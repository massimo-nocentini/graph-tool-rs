//! U1 — the foundation layer, asserted rather than inspected.
//!
//! Nothing in `ids.rs`, `dir.rs`, `bound.rs`, `error.rs` or `graph.rs` has a
//! body to write; what they have is a set of *guarantees* that every later unit
//! silently assumes. DESIGN.md §16 is blunt about what happens when those are
//! taken on trust ("one crate's `ln_gamma` was stubbed to `x.ln()` [...] it
//! compiled, it passed its four tests, and it was 72% wrong"), so this file
//! turns each guarantee into an assertion:
//!
//! * **layout** — the widths DESIGN.md §11 tabulates against graph-tool's own
//!   (`AdjEntry` 8 vs 16, a view 8, `Option<GraphId>` 8 with no sentinel);
//! * **totality** — `Field<D>` names every one of `D::N_FIELDS` half-fields and
//!   nothing else, which is the property defect #4's `_dummy` cell lacks;
//! * **uniqueness** — `GraphId::fresh` across 10 000 calls and across threads;
//! * **the view algebra** — the normalisation table verified in DESIGN.md D3,
//!   re-checked here by `TypeId` so a lost `Undirect for Rev<G>` impl cannot
//!   quietly reintroduce `Und<Rev<_>>`;
//! * **the negative guarantees** — seven `trybuild` fixtures under `tests/ui`,
//!   each pinning one diagnostic quoted in DESIGN.md §14. Those are the claims
//!   that regress silently: adding one blanket impl turns a compile error into
//!   graph-tool's original behaviour with nothing to notice it.

use std::any::TypeId;
use std::collections::HashSet;

use gt_core::adj::{AdjList, AllEdges, Edges, InEdges, OutEdges, SwapEnds, Vertices};
use gt_core::bound::{EdgeBound, VertexBound};
use gt_core::dir::{Dir, Directed, Field, HasDir, Undirected};
use gt_core::error::{DispatchError, GraphError, PropError};
use gt_core::graph::{
    Bidirectional, EdgeList, Endpoints, ExactIncidence, GraphBase, GraphOwner, GraphRef, VertexList,
};
use gt_core::ids::{EdgeId, EdgeTag, GraphId, Id, IdTag, MAX_INDEX, Raw, VertexId, VertexTag};
use gt_core::prop::ValueKind;
use gt_core::view::{Rev, Reverse, Und, Undirect};

// ---------------------------------------------------------------------------
// Static assertion helpers. Each is a *use* of the bound, so the assertion is
// the call site's type-check; the bodies are empty on purpose.
// ---------------------------------------------------------------------------

fn assert_copy<T: Copy>() {}
fn assert_send_sync<T: Send + Sync + 'static>() {}
fn assert_type_eq<A: 'static, B: 'static>(what: &str) {
    assert_eq!(
        TypeId::of::<A>(),
        TypeId::of::<B>(),
        "{what}: {} != {}",
        std::any::type_name::<A>(),
        std::any::type_name::<B>()
    );
}

fn assert_graph_base<G: GraphBase>() {}
fn assert_graph_ref<G: GraphRef>() {}
fn assert_bidirectional<G: Bidirectional>() {}
fn assert_vertex_list<G: VertexList>() {}
fn assert_edge_list<G: EdgeList>() {}
fn assert_endpoints<G: Endpoints>() {}
fn assert_exact_incidence<G: ExactIncidence>()
where
    G::Out: ExactSizeIterator,
    G::All: ExactSizeIterator,
{
}

/// The three concrete views, named once so the tests below read as a table.
type D = &'static AdjList;
type U = Und<&'static AdjList>;
type R = Rev<&'static AdjList>;

// ===========================================================================
// 1. Identifier layout and width (D5, DESIGN.md §11)
// ===========================================================================

/// `Id<T>` is `#[repr(transparent)]` over `Raw` and the tag is a
/// `PhantomData<fn() -> T>`, so a tagged id costs exactly what graph-tool's
/// untagged `Vertex` costs. `adj_list<size_t>` (`src/graph/graph.hh:137`) makes
/// that 8 bytes; here it is 4 unless `wide-index` is on.
#[test]
fn an_id_costs_exactly_its_raw_width() {
    assert_eq!(size_of::<VertexId>(), size_of::<Raw>());
    assert_eq!(size_of::<EdgeId>(), size_of::<Raw>());
    assert_eq!(align_of::<VertexId>(), align_of::<Raw>());
    assert_eq!(align_of::<EdgeId>(), align_of::<Raw>());
    assert_eq!(size_of::<VertexId>(), size_of::<EdgeId>());

    #[cfg(not(feature = "wide-index"))]
    {
        assert_eq!(size_of::<Raw>(), 4, "default width is u32 (D5)");
        // Half of `pair<vertex_t, vertex_t>` under `adj_list<size_t>`: eight
        // adjacency entries per 64-byte line against four.
        assert_eq!(size_of::<VertexId>() * 2, 8);
    }
    #[cfg(feature = "wide-index")]
    {
        assert_eq!(size_of::<Raw>(), 8);
    }
}

/// The phantom tag is `fn() -> T`, which is unconditionally `Send + Sync` and
/// covariant; no `T: Send` bound may leak into a public signature.
#[test]
fn ids_are_copy_send_sync_without_bounding_the_tag() {
    assert_copy::<VertexId>();
    assert_copy::<EdgeId>();
    assert_send_sync::<VertexId>();
    assert_send_sync::<EdgeId>();
    assert_send_sync::<GraphId>();
    assert_send_sync::<VertexBound>();
    assert_send_sync::<EdgeBound>();
    assert_copy::<VertexBound>();
    assert_copy::<Field<Directed>>();
}

/// One value of the raw space is reserved so a future niche-carrying id stays
/// expressible. That makes `MAX_INDEX` the *last admissible* index, not the
/// first rejected one — an off-by-one here would silently shrink every graph.
#[test]
fn the_admissible_index_range_is_closed_at_max_index() {
    assert_eq!(MAX_INDEX, (Raw::MAX - 1) as usize);

    assert!(Id::<VertexTag>::new(0).is_some());
    assert!(Id::<VertexTag>::new(MAX_INDEX).is_some());
    assert!(Id::<VertexTag>::new(MAX_INDEX + 1).is_none());
    assert!(Id::<EdgeTag>::new(MAX_INDEX).is_some());
    assert!(Id::<EdgeTag>::new(MAX_INDEX + 1).is_none());

    assert_eq!(VertexId::from_index(MAX_INDEX).index(), MAX_INDEX);
    assert_eq!(VertexId::from_index(MAX_INDEX).raw(), Raw::MAX - 1);
}

#[test]
#[should_panic(expected = "index exceeds the identifier width")]
fn from_index_panics_rather_than_wrapping() {
    // `i as Raw` would truncate to 0 and hand back a live descriptor; this is
    // the one place the port is allowed to be louder than graph-tool, which
    // has no check at all.
    let _ = VertexId::from_index(MAX_INDEX + 1);
}

#[test]
fn index_and_raw_round_trip_and_order_agrees_with_the_integer() {
    let mut ids: Vec<VertexId> = (0..64usize).map(VertexId::from_index).collect();
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(id.index(), i);
        assert_eq!(id.raw(), i as Raw);
    }
    ids.reverse();
    ids.sort();
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(id.index(), i, "Ord must follow the raw index");
    }

    // Hash agrees with Eq, so an id is usable as a map key.
    let set: HashSet<VertexId> = ids.iter().copied().chain(ids.iter().copied()).collect();
    assert_eq!(set.len(), 64);
}

/// `IdTag::NAME` is what every error message in `error.rs` prints, so the two
/// spaces must be distinguishable in a log, not only in the type-checker.
#[test]
fn debug_names_the_space() {
    assert_eq!(format!("{:?}", VertexId::from_index(3)), "vertex#3");
    assert_eq!(format!("{:?}", EdgeId::from_index(3)), "edge#3");
    assert_eq!(VertexTag::NAME, "vertex");
    assert_eq!(EdgeTag::NAME, "edge");
}

// ===========================================================================
// 2. `GraphId` — uniqueness, which is what makes defect #8 a caught error
// ===========================================================================

/// DESIGN.md §11 lists `Option<Group>` at 4 bytes against graph-tool's
/// `int64_t` + a hand-checked `null_group = INT64_MAX` sentinel. `GraphId` is
/// the same trick one level up: `NonZeroU64` means `Option<GraphId>` needs no
/// reserved value and no ~30 hand-written comparisons (`entries.hh:250`).
#[test]
fn graph_id_carries_a_niche() {
    assert_eq!(size_of::<GraphId>(), 8);
    assert_eq!(size_of::<Option<GraphId>>(), size_of::<GraphId>());
    assert!(GraphId::fresh().get() >= 1);
}

#[test]
fn ten_thousand_fresh_ids_are_distinct() {
    const N: usize = 10_000;
    let ids: Vec<GraphId> = (0..N).map(|_| GraphId::fresh()).collect();
    let uniq: HashSet<u64> = ids.iter().map(|g| g.get()).collect();
    assert_eq!(uniq.len(), N, "GraphId::fresh must not repeat");

    // `fetch_add` on one counter: strictly increasing within a thread.
    for w in ids.windows(2) {
        assert!(w[0].get() < w[1].get());
    }
}

#[test]
fn fresh_ids_are_distinct_across_two_threads() {
    const PER_THREAD: usize = 5_000;
    let mint = || -> Vec<u64> { (0..PER_THREAD).map(|_| GraphId::fresh().get()).collect() };

    let (a, b) = std::thread::scope(|s| {
        let ha = s.spawn(mint);
        let hb = s.spawn(mint);
        (ha.join().unwrap(), hb.join().unwrap())
    });

    let uniq: HashSet<u64> = a.iter().chain(b.iter()).copied().collect();
    assert_eq!(
        uniq.len(),
        2 * PER_THREAD,
        "two threads must not be handed the same identity"
    );
    // And each thread still sees its own values in issue order.
    assert!(a.windows(2).all(|w| w[0] < w[1]));
    assert!(b.windows(2).all(|w| w[0] < w[1]));
}

// ===========================================================================
// 3. `Bound` — allocation bound, distinct from cardinality (defect #7, #15)
// ===========================================================================

/// `filt_graph::num_vertices` forwards to the *unfiltered* graph
/// (`graph_filtered.hh:314-318`) precisely so property storage stays correctly
/// sized, and the header admits at `:301-312` that this costs
/// `distance(vi, viend) == num_vertices(g)`. Splitting the two numbers into two
/// *types* is what lets both be right; that costs one `GraphId` alongside the
/// `usize`, and the point of pinning the size is that it stays one.
#[test]
fn a_bound_is_an_identity_plus_a_length_and_nothing_else() {
    assert_eq!(size_of::<VertexBound>(), 2 * size_of::<usize>());
    assert_eq!(size_of::<EdgeBound>(), size_of::<VertexBound>());
    assert_eq!(align_of::<VertexBound>(), align_of::<u64>());
    // Inherited from `GraphId`'s `NonZeroU64`.
    assert_eq!(size_of::<Option<VertexBound>>(), size_of::<VertexBound>());
    // The tag is phantom: the two index spaces have identical layout and are
    // still not interchangeable (see `tests/ui/u01_vertex_and_edge_ids_are_distinct.rs`).
    assert_ne!(TypeId::of::<VertexBound>(), TypeId::of::<EdgeBound>());
}

// The *behaviour* of `Bound` is tested inside `bound.rs`, because `Bound::new`
// is `pub(crate)` and this file is a downstream crate. That is the guarantee,
// not an inconvenience: `tests/ui/u01_bound_new_is_private.rs` asserts the
// `E0624` a forged bound produces.

// ===========================================================================
// 4. Directedness and the totality of `Field<D>` (D4, defect #4)
// ===========================================================================

#[test]
fn dir_constants_match_the_cpp_field_sets() {
    const { assert!(Directed::DIRECTED) };
    const { assert!(!Undirected::DIRECTED) };
    assert_eq!(Directed::NAME, "directed");
    assert_eq!(Undirected::NAME, "undirected");

    // `EntrySet::resize` (`inference/blockmodel/entries.hh:47-56`) allocates
    // `_r_out_field` and `_nr_out_field` always and the two in-fields only
    // under `if constexpr (directed)`. That is the field set, and here it is a
    // number the delta buffer can size an array with.
    assert_eq!(Directed::N_FIELDS, 4);
    assert_eq!(Undirected::N_FIELDS, 2);
}

/// Totality, from both sides: every nameable `Field<D>` lands inside
/// `0..D::N_FIELDS`, **and** every slot in `0..D::N_FIELDS` is named by some
/// constant. Surjectivity is the half that matters — a buffer with an unnamed
/// slot is a slot some code must reach by arithmetic, and a constant outside
/// the range is an out-of-bounds write.
///
/// `get_field` (`entries.hh:108-119`) has neither property: it returns a shared
/// `_dummy` reference for every pair outside the `(r, nr)` plane, so two
/// distinct out-of-plane block pairs accumulate into one cell.
#[test]
fn field_is_total_over_n_fields() {
    fn check<D: Dir>(fields: &[Field<D>]) {
        assert_eq!(
            fields.len(),
            D::N_FIELDS,
            "{}: field table and N_FIELDS disagree",
            D::NAME
        );
        // `N_FIELDS` and `Fields` are two independent declarations of the same
        // number -- `Fields` cannot be written `[Vec<u32>; Self::N_FIELDS]` on
        // stable, so each implementor spells the length twice. Nothing else
        // would notice them drifting: a buffer sized from `Fields` and indexed
        // against `N_FIELDS` would simply stop reaching its last half-field.
        assert_eq!(
            D::Fields::default().as_ref().len(),
            D::N_FIELDS,
            "{}: Fields length and N_FIELDS disagree",
            D::NAME
        );
        let mut covered = vec![false; D::N_FIELDS];
        for f in fields {
            let i = f.index();
            assert!(i < D::N_FIELDS, "{} field {i} is out of plane", D::NAME);
            assert!(!covered[i], "{} field {i} named twice", D::NAME);
            covered[i] = true;
        }
        assert!(
            covered.iter().all(|&c| c),
            "{}: some half-field has no name",
            D::NAME
        );
    }

    check::<Undirected>(&[Field::<Undirected>::R_OUT, Field::<Undirected>::NR_OUT]);
    check::<Directed>(&[
        Field::<Directed>::R_OUT,
        Field::<Directed>::NR_OUT,
        Field::<Directed>::R_IN,
        Field::<Directed>::NR_IN,
    ]);

    // The in-halves are inherent to `Field<Directed>` and unnameable for
    // `Field<Undirected>`; see `tests/ui/u01_field_in_is_directed_only.rs`.
    assert_eq!(Field::<Directed>::R_IN.index(), 2);
    assert_eq!(Field::<Directed>::NR_IN.index(), 3);
    const { assert!(Field::<Directed>::R_IN.index() < Directed::N_FIELDS) };
    const { assert!(Field::<Directed>::NR_IN.index() < Directed::N_FIELDS) };

    // A `Field` indexes an array; it must not also cost one.
    assert_eq!(size_of::<Field<Directed>>(), 1);
    assert_eq!(size_of::<Field<Undirected>>(), 1);

    assert_ne!(Field::<Directed>::R_OUT, Field::<Directed>::NR_OUT);
    assert_eq!(
        format!("{:?}", Field::<Undirected>::NR_OUT),
        "Field<undirected>(1)"
    );
}

/// `HasDir` is separate from `GraphRef` (D4) exactly so a non-`Copy` *owner*
/// can carry directedness. That is what makes `Und<Arc<AdjList>>` — the safe
/// replacement for `reinterpret_pointer_cast<ug_t>` at
/// `graph_filtering.cc:92` — expressible at all.
#[test]
fn owners_carry_directedness_and_wrapping_an_arc_is_free() {
    use std::sync::Arc;
    assert_type_eq::<<AdjList as HasDir>::Dir, Directed>("AdjList");
    assert_type_eq::<<Arc<AdjList> as HasDir>::Dir, Directed>("Arc<AdjList>");
    assert_type_eq::<<Und<Arc<AdjList>> as HasDir>::Dir, Undirected>("Und<Arc<AdjList>>");

    // The layout the cast produces, obtained by a move.
    assert_eq!(size_of::<Und<Arc<AdjList>>>(), size_of::<Arc<AdjList>>());
    assert_eq!(size_of::<Rev<Arc<AdjList>>>(), size_of::<Arc<AdjList>>());
    assert_eq!(size_of::<Arc<AdjList>>(), size_of::<usize>());
}

// ===========================================================================
// 5. The view algebra (D3) — the normalisation table, re-checked
// ===========================================================================

/// DESIGN.md D3 records this table as "verified by `std::any::type_name` on a
/// real build". A table in a document is not a test: deleting
/// `impl Undirect for Rev<G>` reintroduces `Und<Rev<_>>`, which compiles, runs,
/// and is one of the duplicates `hana::to<set_tag>` (`graph_filtering.hh:129`)
/// exists to remove.
#[test]
fn the_view_algebra_normalises() {
    // involution: d.reverse().reverse() == d, as a type
    assert_type_eq::<<<D as Reverse>::Out as Reverse>::Out, D>("reverse . reverse");
    assert_type_eq::<<D as Reverse>::Out, R>("reverse");

    // idempotent: d.undirect().undirect() == Und<&AdjList>
    assert_type_eq::<<D as Undirect>::Out, U>("undirect");
    assert_type_eq::<<U as Undirect>::Out, U>("undirect . undirect");

    // absorbing: d.reverse().undirect() == Und<&AdjList>, not Und<Rev<_>>
    assert_type_eq::<<R as Undirect>::Out, U>("undirect . reverse");
    assert!(
        !std::any::type_name::<<R as Undirect>::Out>().contains("Rev<"),
        "undirecting a reversed view must absorb the Rev, not nest it: {}",
        std::any::type_name::<<R as Undirect>::Out>()
    );

    // `Rev<Und<_>>` is not merely unused, it is ill-formed:
    // see `tests/ui/u01_rev_und_is_unnameable.rs` (E0271).

    // Directedness survives the algebra.
    assert_type_eq::<<D as HasDir>::Dir, Directed>("&AdjList");
    assert_type_eq::<<R as HasDir>::Dir, Directed>("Rev<&AdjList>");
    assert_type_eq::<<U as HasDir>::Dir, Undirected>("Und<&AdjList>");
}

/// Every view method takes `self` by value and every view is `Copy`, so the
/// adaptor chain collapses to one pointer after inlining (DESIGN.md §11).
#[test]
fn a_view_is_one_pointer() {
    assert_eq!(size_of::<D>(), size_of::<usize>());
    assert_eq!(size_of::<U>(), size_of::<usize>());
    assert_eq!(size_of::<R>(), size_of::<usize>());
    assert_copy::<D>();
    assert_copy::<U>();
    assert_copy::<R>();
}

/// The incidence wiring: `Und<G>::Out` is `G::All` (the *whole* run, anchored
/// at the query vertex, as `graph_adaptor.hh:199-207` routes through
/// `_all_edges_out`), and `Rev<G>::Out` is `G::In` **only**, because
/// `graph_reverse.hh:78-80` swaps the two iterator typedefs. A reversed view
/// yielding `out ∪ in` would be an undirected view wearing a directed type.
///
/// Only the parts of that wiring which are *observable from here* are asserted
/// here. `AdjList` names its three incidence iterators with three aliases of
/// one type (`OutEdges = InEdges = AllEdges = IncidentIter`), so
/// `Rev::Out == InEdges` is true of every possible wiring and asserts nothing.
/// The wiring itself is checked in `graph.rs`'s own test module against a mock
/// whose three directions have three *distinct* iterator types, and by value.
#[test]
fn incidence_iterators_are_wired_to_the_cpp_meaning() {
    // Vacuous by aliasing, but pinned so that splitting the aliases later is a
    // decision rather than an accident.
    assert_type_eq::<OutEdges<'static>, AllEdges<'static>>("AdjList's aliases coincide");
    assert_type_eq::<InEdges<'static>, AllEdges<'static>>("AdjList's aliases coincide");
    assert_type_eq::<<D as GraphRef>::Out, OutEdges<'static>>("&AdjList::Out");
    assert_type_eq::<<D as Bidirectional>::In, InEdges<'static>>("&AdjList::In");
    assert_type_eq::<<D as VertexList>::Vertices, Vertices>("vertices");
    assert_type_eq::<<D as EdgeList>::Edges, Edges<'static>>("edges");

    // Not vacuous: a reversed view must re-orient the *global* edge list, or
    // `edges()` and `endpoints()` would disagree about the same edge.
    assert_type_eq::<<R as EdgeList>::Edges, SwapEnds<Edges<'static>>>("Rev::Edges");
    // An undirected view must not: `edges(undirected_adaptor)` forwards
    // unchanged, since each edge is still stored once, in one orientation.
    assert_type_eq::<<U as EdgeList>::Edges, Edges<'static>>("Und::Edges");
}

/// The trait-impl matrix. `Und<_>` is absent from exactly one row, and that
/// absence is defect #12: `in_edges` on an undirected adaptor returns an empty
/// range (`graph_adaptor.hh:219-227`) instead of failing.
#[test]
fn the_trait_matrix_has_the_hole_it_is_supposed_to_have() {
    assert_graph_base::<D>();
    assert_graph_base::<U>();
    assert_graph_base::<R>();

    assert_graph_ref::<D>();
    assert_graph_ref::<U>();
    assert_graph_ref::<R>();

    assert_vertex_list::<D>();
    assert_vertex_list::<U>();
    assert_vertex_list::<R>();

    assert_edge_list::<D>();
    assert_edge_list::<U>();
    assert_edge_list::<R>();

    assert_endpoints::<D>();
    assert_endpoints::<U>();
    assert_endpoints::<R>();

    assert_bidirectional::<D>();
    assert_bidirectional::<R>();
    // assert_bidirectional::<U>() does not compile:
    // see `tests/ui/u01_und_has_no_in_edges.rs` (E0599).

    // `ExactIncidence` is the refinement unfiltered views implement and
    // per-edge-filtered ones cannot, since they would have to count.
    assert_exact_incidence::<D>();
    // NOTE(U3): `graph.rs`'s own module docs and DESIGN.md §11 say the
    // undirected and reversed views implement it too -- and they can, since
    // `IncidentIter` is `ExactSizeIterator` -- but `view/undirected.rs` and
    // `view/reversed.rs` carry no impl yet. The two lines below belong here
    // once they do; they are not written as a blanket impl in this file
    // because the view impls live with the views.
    //   assert_exact_incidence::<U>();
    //   assert_exact_incidence::<R>();

    // The owner lends the directed view, by value, one word wide.
    assert_type_eq::<<AdjList as GraphOwner>::Ref<'static>, D>("AdjList::Ref");
    assert_type_eq::<<std::sync::Arc<AdjList> as GraphOwner>::Ref<'static>, D>("Arc::Ref");
}

// ===========================================================================
// 6. Errors — the messages are the interface (defect #41, #48)
// ===========================================================================

/// `edge(s, t, g)` returns a `{max, max, max}` descriptor on failure
/// (`graph_adjacency.hh:948`), equal under `operator==` to every other failure
/// *and* to a legitimate descriptor with that index. Here failure is `Option`
/// or a named error whose `Display` says which descriptor, which space and
/// which graph — so a log line is diagnostic on its own.
#[test]
fn error_messages_name_the_thing_that_failed() {
    assert_eq!(
        GraphError::NoSuchVertex(VertexId::from_index(7)).to_string(),
        "no such vertex: vertex#7"
    );
    assert_eq!(
        GraphError::NoSuchEdge(EdgeId::from_index(7)).to_string(),
        "no such edge: edge#7"
    );
    assert_eq!(
        GraphError::EdgeIdSpaceExhausted { max: MAX_INDEX }.to_string(),
        format!("edge index space exhausted (max {MAX_INDEX})")
    );
    assert_eq!(
        GraphError::VertexIdSpaceExhausted { max: MAX_INDEX }.to_string(),
        format!("vertex index space exhausted (max {MAX_INDEX})")
    );
    assert_eq!(
        GraphError::Invariant("out_len exceeds len").to_string(),
        "internal invariant violated: out_len exceeds len"
    );

    // Defect #7/#8, as a runtime message: the two identities are both printed,
    // because "wrong graph" with only one number in it is not actionable.
    assert_eq!(
        PropError::WrongGraph {
            owner: 3,
            expected: 4
        }
        .to_string(),
        "property map belongs to graph #3, not graph #4"
    );
    assert_eq!(
        PropError::Undersized { have: 5, need: 9 }.to_string(),
        "property map holds 5 entries, the graph's index bound is 9"
    );
    assert_eq!(
        PropError::ShortMask { have: 5, need: 9 }.to_string(),
        "filter mask holds 5 entries, the index bound is 9"
    );
    assert_eq!(
        PropError::NoConversion {
            from: ValueKind::Str,
            to: ValueKind::I32,
        }
        .to_string(),
        "cannot convert a Str property map to I32"
    );
    assert_eq!(
        PropError::NotReadable.to_string(),
        "property map is not readable"
    );
    assert_eq!(
        PropError::NotWritable.to_string(),
        "property map is not writable"
    );
}

/// `DispatchNotFound`'s message is literally "This is a graph_tool bug. :-("
/// (`src/graph/dispatch.hh:85-88`), so a user handing `string` to an
/// integer-only kernel and a genuine codegen hole are indistinguishable.
/// `DispatchError` carries the axis, what was offered and what is accepted;
/// only the first of the two conditions survives to be reported.
#[test]
fn dispatch_error_reports_the_axis_and_the_accepted_set() {
    let e = DispatchError {
        axis: "edge weight",
        offered: ValueKind::Str,
        accepted: &[ValueKind::I32, ValueKind::I64, ValueKind::F64],
    };
    assert_eq!(
        e.to_string(),
        "edge weight: Str is not one of [I32, I64, F64]"
    );
    assert!(!e.to_string().contains("bug"));
}

#[test]
fn errors_are_values_not_exceptions() {
    // graph-tool throws `ValueException` from six sites reachable inside an
    // OpenMP region (`graph_properties.hh:266, :319, :447, :457`,
    // `graph_copy.cc:99, :181`), where escaping the region is undefined.
    // These are plain `Send + Sync` data.
    assert_send_sync::<GraphError>();
    assert_send_sync::<PropError>();
    assert_send_sync::<DispatchError>();
    assert_eq!(
        GraphError::NoSuchVertex(VertexId::from_index(1)),
        GraphError::NoSuchVertex(VertexId::from_index(1))
    );
    assert_ne!(
        GraphError::NoSuchVertex(VertexId::from_index(1)),
        GraphError::NoSuchVertex(VertexId::from_index(2))
    );
    // `Result<(), GraphError>` is the return type of every `AdjList` mutator,
    // so the discriminant must come out of the error's own padding rather than
    // widening it. graph-tool pays nothing here only because it pays with an
    // exception instead.
    assert_eq!(size_of::<GraphError>(), 3 * size_of::<usize>());
    assert_eq!(
        size_of::<Result<(), GraphError>>(),
        size_of::<GraphError>(),
        "the Ok discriminant must fit in GraphError's niche"
    );
    assert_eq!(
        size_of::<Result<(), PropError>>(),
        size_of::<PropError>(),
        "the Ok discriminant must fit in PropError's niche"
    );
}

// ===========================================================================
// 7. The negative guarantees (DESIGN.md §14)
// ===========================================================================

/// Each fixture pins one diagnostic that DESIGN.md §14 claims to have verified.
/// They are the claims that regress *silently*: a blanket impl, a `pub` on a
/// constructor or a relaxed bound turns a compile error back into graph-tool's
/// original runtime behaviour, and nothing else in the suite would notice.
///
/// | fixture | defect | diagnostic |
/// |---|---|---|
/// | `u01_bound_new_is_private` | #7 `graph_copy.cc:66-73` | `E0624` |
/// | `u01_rev_und_is_unnameable` | #17 `graph_filtering.hh:116-121` | `E0271` |
/// | `u01_und_has_no_in_edges` | #12 `graph_adaptor.hh:219-227` | `E0599` |
/// | `u01_graph_ref_is_not_dyn` | D1 | `E0191` |
/// | `u01_graph_ref_is_not_dyn_projected` | D1 | `E0038` |
/// | `u01_field_in_is_directed_only` | #4 `entries.hh:108-119` | `E0599` |
/// | `u01_vertex_and_edge_ids_are_distinct` | `graph_adjacency.hh:211` | `E0308` |
#[test]
fn negative_guarantees_still_fail_to_compile() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/u01_*.rs");
}
