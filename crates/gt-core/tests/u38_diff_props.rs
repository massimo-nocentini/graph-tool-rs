//! Differential tests: property-map sizing and growth, against graph-tool
//! 3.8's `src/graph/fast_vector_property_map.hh` and the Python
//! `PropertyMap` that decides what a caller ever sees.
//!
//! ## The three C++ functions
//!
//! ```c++
//! // fast_vector_property_map.hh:77-82
//! void reserve(size_t size) const
//! { if (size > _store->size()) _store->resize(size); }
//!
//! // :84-89
//! void soft_reserve(size_t size) const
//! { if (size > _store->size()) resize(std::max(size, 2 * _store->size())); }
//!
//! // :132-137  --- the one that matters
//! reference operator[](const key_type& v) const
//! { auto i = get(_index, v); soft_reserve(i + 1); return (*_store)[i]; }
//! ```
//!
//! `operator[]` is `const` and is what both `get` and `put` call
//! (`:141-167`), so **reading** `g.vp.x[v]` on a never-written vertex both
//! returns the default *and* resizes the store to `max(v + 1, 2 * len)`.
//!
//! ## What a caller sees
//!
//! Not that length. `PropertyMap._get_any` reserves `num_vertices(True)` for a
//! vertex map, `edge_index_range` for an edge map and 1 for a graph property
//! (`graph_tool/__init__.py:363-372`), and `_get_data` then truncates:
//! `self.__map.get_array(n)` with `n = get_num_vertices(False)` (`:974-981`)
//! or `n = g.edge_index_range` (`:1057-1063`). So the geometric slack is
//! invisible from `.a` and visible only to `get_unchecked`, which `reserve`s a
//! caller-supplied size and then indexes without a bound
//! (`fast_vector_property_map.hh:109-118`, `dispatch.hh:171-177`).
//!
//! ## What this port does instead, and why the tests below state both numbers
//!
//! [`DenseProp::sized_for`] is `reserve`: it grows to **exactly** the bound
//! and never shrinks. [`DenseProp::soft_reserve`] grows *capacity* only.
//! [`DenseProp::view`] --- the `&self` read path --- cannot grow at all and
//! reports [`PropError::Undersized`]. The auto-growth on read therefore has to
//! happen at the dispatcher, by calling `sized_for` on read-only maps too, and
//! the tests here pin the shape of what it has to reproduce.

use gt_core::adj::AdjList;
use gt_core::error::PropError;
use gt_core::ids::{EdgeId, EdgeTag, VertexId, VertexTag};
use gt_core::prop::DenseProp;

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

fn graph(n: usize, edges: &[(usize, usize)]) -> AdjList {
    let mut g = AdjList::with_vertices(n);
    for &(s, t) in edges {
        g.add_edge(v(s), v(t)).expect("both endpoints exist");
    }
    g
}

/// `soft_reserve`'s resulting **length**, as graph-tool computes it
/// (`fast_vector_property_map.hh:84-89`).
fn cpp_soft_reserve(len: usize, want: usize) -> usize {
    if want > len { want.max(2 * len) } else { len }
}

/// `reserve`'s resulting length (`:77-82`).
fn cpp_reserve(len: usize, want: usize) -> usize {
    if want > len { want } else { len }
}

// ---------------------------------------------------------------------------

/// `sized_for` is `reserve`, not `soft_reserve`: it grows to exactly the
/// bound, and the two differ by a factor of up to two.
///
/// Hand-computed from `:84-89`, `len` being the store's length before the
/// call and `want` the requested one:
///
/// ```text
///   len  want   reserve   soft_reserve = max(want, 2*len)
///     0     4        4        4
///     4     5        5        8       <-- the doubling wins
///     4   100      100      100       <-- `want` wins
///     8     3        8        8       <-- neither shrinks
/// ```
///
/// The port implements the `reserve` column. That is not observable through
/// `.a` --- `_get_data` truncates to the bound (`__init__.py:974-981`) --- but
/// it is observable to anything that reads `get_storage()` or that hands the
/// map to `get_unchecked` with a size below the slack.
#[test]
fn sized_for_reserves_exactly_where_graph_tool_would_double() {
    let g = graph(4, &[(0, 1), (1, 2)]);
    let mut p: DenseProp<i64, VertexTag> = DenseProp::new(g.graph_id());
    assert_eq!(p.len(), 0, "a fresh map has no store");

    // len 0, want 4.
    p.sized_for(g.vertex_bound()).expect("same graph");
    assert_eq!(p.len(), 4);
    assert_eq!(cpp_reserve(0, 4), 4);
    assert_eq!(cpp_soft_reserve(0, 4), 4, "the two agree from empty");

    // len 4, want 5: `reserve` gives 5, `soft_reserve` gives 8.
    let mut g2 = graph(4, &[]);
    g2.add_vertex().expect("room");
    let mut q: DenseProp<i64, VertexTag> = DenseProp::from_vec(g2.graph_id(), vec![7; 4]);
    q.sized_for(g2.vertex_bound()).expect("same graph");
    assert_eq!(q.len(), 5, "`reserve(5)`");
    assert_eq!(cpp_soft_reserve(4, 5), 8, "graph-tool would have gone to 8");
    assert_eq!(
        q.as_slice(),
        &[7, 7, 7, 7, 0],
        "the existing values survive and the new slot is `T::zero()`"
    );

    // len 4, want 100: both give 100.
    let mut big = AdjList::with_vertices(100);
    big.add_edge(v(0), v(1)).expect("add");
    let mut r: DenseProp<i64, VertexTag> = DenseProp::from_vec(big.graph_id(), vec![1; 4]);
    r.sized_for(big.vertex_bound()).expect("same graph");
    assert_eq!(r.len(), 100);
    assert_eq!(cpp_soft_reserve(4, 100), 100);
}

/// Neither `reserve` nor `sized_for` ever shrinks, and the guard is what makes
/// a map safe to share between two kernels over different bounds.
///
/// `if (size > _store->size())` (`:79`, `:86`) is the whole of it. A map that
/// shrank would hand the second kernel a store missing the first's slots ---
/// and after `swap_remove_vertex` the bound genuinely goes down, so this is
/// reachable and not hypothetical.
#[test]
fn sizing_never_shrinks_even_when_the_bound_does() {
    let mut g = graph(6, &[(0, 1), (4, 5)]);
    let mut p: DenseProp<i64, VertexTag> = DenseProp::new(g.graph_id());
    p.sized_for(g.vertex_bound()).expect("same graph");
    for (i, x) in p.as_mut_slice().iter_mut().enumerate() {
        *x = i as i64 + 1;
    }
    assert_eq!(p.len(), 6);

    // `swap_remove_vertex` pops a block, so the bound drops to 5.
    g.swap_remove_vertex(v(2)).expect("v2 exists");
    assert_eq!(g.vertex_bound().len(), 5);

    p.sized_for(g.vertex_bound()).expect("same graph");
    assert_eq!(p.len(), 6, "`if (size > size())` -- no shrink");
    assert_eq!(cpp_reserve(6, 5), 6);
    assert_eq!(
        p.as_slice(),
        &[1, 2, 3, 4, 5, 6],
        "nothing is dropped or reordered: sizing is not relabelling"
    );
    // The *view* is the bound, though, so a kernel sees exactly five slots.
    assert_eq!(p.view(g.vertex_bound()).expect("same graph").len(), 5);
}

/// **The growth-on-read the port refuses.** `view` is fallible where
/// `operator[]` would silently extend.
///
/// `g.vp.x[v]` on a map shorter than `v + 1` returns `T()` and resizes
/// (`:132-137`), which is why a graph-tool caller never has to size a
/// read-only map. `&self` cannot resize here, so `view` reports
/// [`PropError::Undersized`] with both numbers and the dispatcher is
/// responsible for calling `sized_for` first --- on read-only maps too.
///
/// The failure mode this closes is `get_unchecked(size = 0)`
/// (`fast_vector_property_map.hh:109-113`, reached from `dispatch.hh:171-177`):
/// `reserve(0)` is a no-op, the returned `unchecked` map indexes without a
/// bound, and a kernel then writes past the end of the store.
#[test]
fn a_read_on_a_short_map_is_an_error_where_graph_tool_grows_it() {
    let g = graph(5, &[(0, 1)]);
    let mut p: DenseProp<i64, VertexTag> = DenseProp::from_vec(g.graph_id(), vec![10, 11]);

    let err = p
        .view(g.vertex_bound())
        .expect_err("a two-slot map cannot answer for five vertices");
    assert!(
        matches!(err, PropError::Undersized { have: 2, need: 5 }),
        "got {err:?}"
    );
    // graph-tool's answer to the same question: the default, plus a store of
    // `max(5, 2 * 2) == 5`.
    assert_eq!(cpp_soft_reserve(2, 5), 5);

    // Sizing it first is the port's required two-phase shape.
    p.sized_for(g.vertex_bound()).expect("same graph");
    let view = p.view(g.vertex_bound()).expect("now it is long enough");
    assert_eq!(view.as_slice(), &[10, 11, 0, 0, 0]);
}

/// A vertex map is sized by the vertex *bound* and an edge map by the edge
/// index *range*, and both differ from the live counts.
///
/// `_get_any` (`graph_tool/__init__.py:363-372`):
///
/// ```python
/// if t == "v":   N = g.num_vertices(True)      # the unfiltered count
/// elif t == "e": N = g.edge_index_range        # NOT num_edges()
/// else:          N = 1
/// self.reserve(N)
/// ```
///
/// After removals the edge index space is sparse
/// (`_edge_idx_range` is monotone, `graph_adjacency.hh:636`/`:648`), so a map
/// sized to `num_edges()` would put a live edge's value out of bounds. This is
/// the composition of that fact with the sizing rule.
#[test]
fn an_edge_map_is_sized_by_the_index_range_and_a_vertex_map_by_the_bound() {
    let mut g = graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
    g.remove_edge(EdgeId::from_index(1)).expect("live");
    g.remove_edge(EdgeId::from_index(3)).expect("live");
    assert_eq!((g.num_edges(), g.edge_bound().len()), (3, 5));

    let mut e: DenseProp<f64, EdgeTag> = DenseProp::new(g.graph_id());
    e.sized_for(g.edge_bound()).expect("same graph");
    assert_eq!(e.len(), 5, "`reserve(edge_index_range)`, not `num_edges()`");
    for ed in g.edges() {
        e.as_mut_slice()[ed.id().index()] = ed.id().index() as f64;
    }
    assert_eq!(
        e.as_slice(),
        &[0.0, 0.0, 2.0, 0.0, 4.0],
        "the live ids are 0, 2 and 4; the holes keep the default"
    );

    // A vertex map, by contrast, is sized by `num_vertices(True)` --- the
    // unfiltered index space --- so removing an edge does not touch it.
    let mut vp: DenseProp<i64, VertexTag> = DenseProp::new(g.graph_id());
    vp.sized_for(g.vertex_bound()).expect("same graph");
    assert_eq!(vp.len(), 5);
    assert_eq!(g.vertex_bound().len(), g.num_vertices());
}

/// Growth fills with `T()`, and `sized_for_with` supplies the value for the
/// members that have no context-free one.
///
/// `std::vector<T>::resize(size)` value-initialises, which for every scalar
/// member of graph-tool's value universe is zero and for `std::string` is the
/// empty string. The one member with no such value is
/// `boost::python::object`, whose default construction needs a live
/// interpreter --- which is exactly why the port splits `sized_for`
/// (`T: Zeroed`) from `sized_for_with` (a closure).
#[test]
fn growth_value_initialises_and_the_closure_covers_the_member_that_cannot() {
    let g = graph(4, &[]);

    let mut i: DenseProp<i64, VertexTag> = DenseProp::from_vec(g.graph_id(), vec![5]);
    i.sized_for(g.vertex_bound()).expect("same graph");
    assert_eq!(i.as_slice(), &[5, 0, 0, 0]);

    let mut f: DenseProp<f64, VertexTag> = DenseProp::new(g.graph_id());
    f.sized_for(g.vertex_bound()).expect("same graph");
    assert_eq!(f.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    assert!(f.as_slice().iter().all(|x| x.is_sign_positive()), "+0.0");

    let mut s: DenseProp<String, VertexTag> =
        DenseProp::from_vec(g.graph_id(), vec!["kept".to_owned()]);
    s.sized_for_with(g.vertex_bound(), String::new)
        .expect("same graph");
    assert_eq!(s.as_slice(), &["kept", "", "", ""]);

    // The closure runs once per *new* slot and not at all when the map is
    // already long enough -- `resize_with`'s contract and `reserve`'s guard.
    let mut calls = 0usize;
    s.sized_for_with(g.vertex_bound(), || {
        calls += 1;
        String::new()
    })
    .expect("same graph");
    assert_eq!(calls, 0, "no new slots, no calls");
}

/// `soft_reserve` grows capacity only; in graph-tool it grows the length.
///
/// **Deliberate divergence**, documented on the method. `resize` is what lets
/// `operator[]` extend a map on read, and that behaviour is deliberately not
/// reachable here: the length of a `DenseProp` changes in exactly one place.
/// A caller that expected `soft_reserve` to make slots addressable --- a
/// push-style filler in `gt-io`, say --- gets a map of unchanged length and
/// has to call `sized_for`.
#[test]
fn soft_reserve_grows_capacity_where_graph_tool_grows_the_length() {
    let g = graph(4, &[]);
    let mut p: DenseProp<i64, VertexTag> = DenseProp::from_vec(g.graph_id(), vec![1, 2, 3, 4]);
    assert_eq!(p.len(), 4);

    p.soft_reserve(5);
    assert_eq!(
        p.len(),
        4,
        "capacity only: graph-tool's `soft_reserve(5)` would make this 8"
    );
    assert_eq!(cpp_soft_reserve(4, 5), 8);
    assert_eq!(
        p.as_slice(),
        &[1, 2, 3, 4],
        "and the contents are untouched"
    );

    // It is a hint, so nothing's correctness depends on it: sizing after the
    // hint gives the same answer as sizing without one.
    let mut q: DenseProp<i64, VertexTag> = DenseProp::from_vec(g.graph_id(), vec![1, 2, 3, 4]);
    let mut big = AdjList::with_vertices(9);
    big.add_edge(v(0), v(1)).expect("add");
    let mut a: DenseProp<i64, VertexTag> = DenseProp::from_vec(big.graph_id(), vec![1, 2, 3, 4]);
    a.soft_reserve(9);
    a.sized_for(big.vertex_bound()).expect("same graph");
    q.sized_for(g.vertex_bound()).expect("same graph");
    assert_eq!(a.len(), 9);
    assert_eq!(q.len(), 4);
}

/// A map minted for one graph is refused by another, which graph-tool cannot
/// say at all.
///
/// A `checked_vector_property_map` holds a `shared_ptr<vector<T>>` and an
/// index map and has no notion of *which* graph it belongs to
/// (`fast_vector_property_map.hh:150-152`), so handing a vertex map of `g` to
/// a kernel over `h` is a silent mis-index whenever the two have different
/// sizes and a silent wrong answer when they do not.
#[test]
fn a_map_from_another_graph_is_refused_by_the_bound() {
    let g = graph(4, &[(0, 1)]);
    let h = graph(4, &[(0, 1)]);
    assert_ne!(g.graph_id(), h.graph_id(), "two graphs, two identities");
    assert_eq!(
        g.vertex_bound().len(),
        h.vertex_bound().len(),
        "...and the same size, which is what makes the mistake silent in C++"
    );

    let mut p: DenseProp<i64, VertexTag> = DenseProp::new(g.graph_id());
    let err = p
        .sized_for(h.vertex_bound())
        .expect_err("a map of `g` must not be sized for `h`");
    assert!(matches!(err, PropError::WrongGraph { .. }), "got {err:?}");
    let err = p
        .view(h.vertex_bound())
        .expect_err("nor read through `h`'s bound");
    assert!(matches!(err, PropError::WrongGraph { .. }), "got {err:?}");
    // Its own graph is fine.
    p.sized_for(g.vertex_bound()).expect("same graph");
}
