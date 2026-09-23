//! U7 — the view algebra, from outside the crate.
//!
//! The *behavioural* acceptance list for this unit — defect #14's
//! `undirect().out_edges(1) ∈ {0, 2}`, `reverse().out_edges(v)` being the
//! in-edges rather than their union, `num_vertices() == vertices().count()`
//! on all six views under four masks, the `ShortMask` refusals and the
//! `find_edge` scan — needs a *populated* graph, and a populated graph cannot
//! exist here: `Bound::new` and `EdgeRef::new` are `pub(crate)` (defects #7
//! and #41), so no downstream type can present itself as a graph, and
//! `AdjList`'s mutators are U5's. Those assertions therefore live in
//! `src/view/mod.rs`'s own `#[cfg(test)]` module, next to `src/graph.rs`'s,
//! and for the same reason.
//!
//! What is left is exactly what an external crate *can* see, and it is not
//! nothing:
//!
//! * **layout** — `gt_core::design` §11's table for all six views, including the two
//!   that decided D3: `KeepAll` erases the mask fields (a const generic would
//!   not), and `Und<Arc<AdjList>>` is the layout
//!   `reinterpret_pointer_cast<ug_t>` produces, obtained by a move;
//! * **the type table** — `Filtered`'s associated iterator types, so a
//!   re-wiring of `Out` to `All` is caught as a type error rather than as a
//!   wrong number;
//! * **the negative guarantees** — `tests/ui/u07_*.rs`, pinning the three
//!   diagnostics this unit is responsible for.

use std::any::TypeId;
use std::sync::Arc;

use gt_core::adj::{
    AdjList, AllEdges, FilterEdges, FilterIncident, FilterVertices, OutEdges, Vertices,
};
use gt_core::graph::{Bidirectional, EdgeList, Endpoints, GraphRef, VertexList};
use gt_core::ids::{EdgeId, VertexId};
use gt_core::view::{Filter, Filtered, KeepAll, MaskFilter, Rev, Und};

// ---------------------------------------------------------------------------
// The six views, named once so the tests below read as a table. `Filtered<_,
// KeepAll>` is the same six with the predicate const-folded away, not a
// seventh: `graph_filtering.hh:127` de-duplicates to six because
// `filt_graph<g, always_true, always_true>` *is* `g`'s shape.
// ---------------------------------------------------------------------------

type D = &'static AdjList;
type U = Und<&'static AdjList>;
type R = Rev<&'static AdjList>;
type FD = Filtered<D, MaskFilter<'static>>;
type FU = Filtered<U, MaskFilter<'static>>;
type FR = Filtered<R, MaskFilter<'static>>;

fn assert_copy<T: Copy>() {}
fn assert_send_sync<T: Send + Sync + 'static>() {}
fn assert_graph_ref<G: GraphRef>() {}
fn assert_bidirectional<G: Bidirectional>() {}
fn assert_vertex_list<G: VertexList>() {}
fn assert_edge_list<G: EdgeList>() {}
fn assert_endpoints<G: Endpoints>() {}
fn assert_type_eq<A: 'static, B: 'static>(what: &str) {
    assert_eq!(
        TypeId::of::<A>(),
        TypeId::of::<B>(),
        "{what}: {} != {}",
        std::any::type_name::<A>(),
        std::any::type_name::<B>()
    );
}

// ===========================================================================
// 1. Layout (`gt_core::design` §11)
// ===========================================================================

/// The three unfiltered views are one machine word: a view is a *pointer*, and
/// `#[repr(transparent)]` plus a private field is what keeps it one after the
/// wrapper. This is the number D1 rests on when it says every view method may
/// take `self` by value.
#[test]
fn an_unfiltered_view_is_one_word() {
    assert_eq!(size_of::<D>(), size_of::<usize>());
    assert_eq!(size_of::<U>(), size_of::<usize>());
    assert_eq!(size_of::<R>(), size_of::<usize>());

    // The `reinterpret_pointer_cast<ug_t>(u)` case (`graph_filtering.cc:92`),
    // which casts a `shared_ptr` to an object never constructed as that type:
    // same layout, obtained by a move.
    assert_eq!(size_of::<Und<Arc<AdjList>>>(), size_of::<Arc<AdjList>>());
    assert_eq!(size_of::<Rev<Arc<AdjList>>>(), size_of::<Arc<AdjList>>());
}

/// `Filtered<G, MaskFilter>` is the view, two `&[u8]` and the two memoised
/// counts: 8 + 32 + 16 = 56 on a 64-bit build with `Raw = u32`.
///
/// And `Filtered<G, KeepAll>` is 24, because `KeepAll` is a ZST and the mask
/// fields are **gone** — not present-and-unread. That difference is the whole
/// of D3's argument against `Filtered<G, const ON: bool>`: a const parameter
/// erases no data, so the unfiltered arm would still carry 32 bytes of
/// dummies. (The design that shipped the const-generic form also shipped a
/// `vmask` field read nowhere, so every filtered measurement it reported was
/// of a no-op.)
#[test]
fn a_filtered_view_costs_exactly_its_filter() {
    assert_eq!(size_of::<MaskFilter<'static>>(), 4 * size_of::<usize>());
    assert_eq!(size_of::<KeepAll>(), 0);

    let counters = 2 * size_of::<usize>();
    assert_eq!(size_of::<FD>(), size_of::<D>() + 32 + counters);
    assert_eq!(size_of::<FD>(), 56);
    assert_eq!(size_of::<FU>(), 56);
    assert_eq!(size_of::<FR>(), 56);

    assert_eq!(size_of::<Filtered<D, KeepAll>>(), size_of::<D>() + counters);
    assert_eq!(size_of::<Filtered<D, KeepAll>>(), 24);
    assert_eq!(size_of::<Filtered<U, KeepAll>>(), 24);
    assert_eq!(size_of::<Filtered<R, KeepAll>>(), 24);
    assert!(
        size_of::<Filtered<D, KeepAll>>() < size_of::<FD>(),
        "the trivial filter must erase the masks, not blank them"
    );
}

/// Every view is `Copy` and thread-shareable, which is what lets `GraphRef`
/// take `self` by value and what lets a view cross into a rayon closure —
/// the two things D1 records a GAT-based trait cannot do at once.
#[test]
fn every_view_is_copy_and_shareable() {
    assert_copy::<D>();
    assert_copy::<U>();
    assert_copy::<R>();
    assert_copy::<FD>();
    assert_copy::<FU>();
    assert_copy::<FR>();
    assert_copy::<Filtered<D, KeepAll>>();

    assert_send_sync::<D>();
    assert_send_sync::<U>();
    assert_send_sync::<R>();
    assert_send_sync::<FD>();
    assert_send_sync::<FU>();
    assert_send_sync::<FR>();
    // The bound is on the `Filter` trait itself, so no filter can smuggle a
    // `!Sync` predicate into a parallel loop.
    assert_send_sync::<MaskFilter<'static>>();
    assert_send_sync::<KeepAll>();
}

// ===========================================================================
// 2. The type table
// ===========================================================================

/// Which iterator a filtered view wraps is a type-level fact, and the one a
/// wrong wiring would get wrong silently: `Out` must wrap the *base's* `Out`
/// and `All` the base's `All`. `AdjList` names all three incidence iterators
/// with the same alias, so this is checkable here only through the wrappers —
/// `Und<G>::Out = G::All` is the interesting row, and it is the type-level
/// form of defect #14.
#[test]
fn the_filtered_iterator_types_are_the_wrapped_ones() {
    type M = MaskFilter<'static>;
    assert_type_eq::<<FD as GraphRef>::Out, FilterIncident<OutEdges<'static>, M>>("FD::Out");
    assert_type_eq::<<FD as GraphRef>::All, FilterIncident<AllEdges<'static>, M>>("FD::All");
    assert_type_eq::<<FD as VertexList>::Vertices, FilterVertices<Vertices, M>>("FD::Vertices");
    assert_type_eq::<<FD as EdgeList>::Edges, FilterEdges<<D as EdgeList>::Edges, M>>("FD::Edges");
    assert_type_eq::<<FD as Bidirectional>::In, FilterIncident<<D as Bidirectional>::In, M>>(
        "FD::In",
    );

    // `Und<G>::Out = G::All`, so the *filtered* undirected view's out-edges
    // are the filtered whole run — anchored, never re-oriented.
    assert_type_eq::<<FU as GraphRef>::Out, FilterIncident<AllEdges<'static>, M>>("FU::Out");
    assert_type_eq::<<FU as GraphRef>::Out, <FU as GraphRef>::All>("FU::Out == FU::All");
    // ... and the reversed view's out-edges are the base's in-edges only.
    assert_type_eq::<<FR as GraphRef>::Out, FilterIncident<<D as Bidirectional>::In, M>>("FR::Out");
    assert_type_eq::<<FR as Bidirectional>::In, FilterIncident<OutEdges<'static>, M>>("FR::In");
}

/// A filtered view is still a graph, on every axis the unfiltered one is —
/// except `Bidirectional`, which `Filtered<Und<_>, _>` must not have, because
/// `Und<_>` does not (defect #12). That row is pinned by
/// `tests/ui/u07_filtered_undirected_has_no_in_edges.rs`.
#[test]
fn a_filtered_view_is_a_graph_on_every_axis() {
    assert_graph_ref::<FD>();
    assert_graph_ref::<FU>();
    assert_graph_ref::<FR>();
    assert_vertex_list::<FD>();
    assert_vertex_list::<FU>();
    assert_edge_list::<FD>();
    assert_edge_list::<FU>();
    assert_edge_list::<FR>();
    assert_endpoints::<FD>();
    assert_endpoints::<FU>();
    assert_endpoints::<FR>();
    assert_bidirectional::<FD>();
    assert_bidirectional::<FR>();
    assert_graph_ref::<Filtered<D, KeepAll>>();
}

// ===========================================================================
// 3. The predicate
// ===========================================================================

/// `KeepAll` is the `TRIVIAL` arm: every probe is a constant and every
/// `F::TRIVIAL` branch in this unit folds away. It is also the only `Filter`
/// an external crate can construct — `MaskFilter`'s fields are private, so a
/// mask reaches a view only through `Filtered::masked`, which validates it
/// against that graph's own bounds. A free-standing
/// `MaskFilter::new(vmask, emask, vbound, ebound)` would be `Copy` and
/// remember nothing, which is the same structural defect as
/// `graph_filtering.cc:42-46` reserving separately from
/// `MaskFilter::operator()`.
#[test]
fn keep_all_keeps_everything_at_compile_time() {
    const { assert!(<KeepAll as Filter>::TRIVIAL) };
    const { assert!(!<MaskFilter<'static> as Filter>::TRIVIAL) };

    for i in [0usize, 1, 7, 4096] {
        assert!(KeepAll.keep_vertex(VertexId::from_index(i)));
        assert!(KeepAll.keep_edge(EdgeId::from_index(i)));
    }
    assert!(KeepAll.keep_vertex(VertexId::from_index(usize::from(u16::MAX))));
    // A ZST, so a `Filtered<G, KeepAll>` is a view plus two counters.
    assert_eq!(size_of::<KeepAll>(), 0);
    assert!(KeepAll.keep_edge(EdgeId::from_index(3)));
}

// ===========================================================================
// 4. The negative guarantees
// ===========================================================================

/// Three diagnostics this unit owns:
///
/// * `g.undirect().reverse()` is `error[E0599]`. `graph_filtering.hh:75-79`
///   guards the same case with `reversed && is_directed_v<directed_t>`, whose
///   effect is to *silently ignore* the request: the undirected view is
///   returned unreversed, and nothing tells the caller. Here there is no
///   `Reverse for Und<_>` impl, so the call does not exist.
/// * `ExactIncidence` on a filtered view is `error[E0277]`. `gt_core::design` §4
///   records this as the resolution of the one question a source design left
///   open: `GraphRef` cannot require `ExactSizeIterator`, because a
///   per-edge-predicate view cannot supply a length without walking — and
///   `Filtered::new` walks *once*, into `num_edges`, rather than on every
///   `len()`.
/// * `in_edges` on a `Filtered<Und<_>, _>` is `error[E0599]`. `Bidirectional
///   for Filtered<G, F>` is bounded on `G: Bidirectional` and `Und<_>` is not,
///   so defect #12's empty in-range (`graph_adaptor.hh:219-227`, summed by
///   `graph_filtered.hh:395-399`) has nowhere to come from.
#[test]
fn the_view_algebra_is_closed_at_the_type_level() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/u07_*.rs");
}
