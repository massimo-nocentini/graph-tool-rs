//! U31 — view dispatch.
//!
//! The C++ side of every claim here is `graph_filtering.hh`: `get_graph_views`
//! (`:101-128`) builds the view set from a `hana::cartesian_product`,
//! `run_action` (`:180-205`) hands each one to a generic lambda, and
//! `gt_dispatch` (`dispatch.hh:60-88`) picks the arm by a linear `typeid`
//! scan that ends in `DispatchNotFound`. The Rust dispatcher is an exhaustive
//! `match` over [`ViewKind`], so the six arms are checked by the compiler and
//! what these tests check is that each arm builds the *right* view.
//!
//! The fixture is one small graph whose answers differ in all six views:
//!
//! ```text
//!   e0: 0 -> 1        e3: 0 -> 1   (parallel to e0)
//!   e1: 1 -> 2        e4: 3 -> 3   (self-loop)
//!   e2: 2 -> 0
//! ```
//!
//! and a mask that drops **e0 but keeps e3**, so a filtered `find_edge(1, 0)`
//! can only answer correctly by scanning past the filtered-out first parallel
//! edge — defect #16 in miniature.

use gt_core::adj::{AdjList, NoLookup};
use gt_core::error::PropError;
use gt_core::graph::{Bidirectional, EdgeList, Endpoints, GraphRef, VertexList};
use gt_core::ids::VertexId;
use gt_core::view::{Reverse, Undirect};
use gt_py::dispatch::{AnyGraph, BidiGraphKernel, DynGraph, GraphKernel, ViewError, ViewKind};

// ---------------------------------------------------------------------------
// fixture
// ---------------------------------------------------------------------------

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

fn fixture() -> AdjList<NoLookup> {
    let mut g = AdjList::with_vertices(4);
    for (s, t) in [(0, 1), (1, 2), (2, 0), (0, 1), (3, 3)] {
        g.add_edge(v(s), v(t)).expect("add_edge");
    }
    g
}

/// Keeps every vertex; drops `e0` and keeps its parallel twin `e3`.
fn masks(g: &AdjList<NoLookup>) -> (Vec<u8>, Vec<u8>) {
    let vmask = vec![1u8; g.vertex_bound().len()];
    let mut emask = vec![1u8; g.edge_bound().len()];
    emask[0] = 0;
    (vmask, emask)
}

// ---------------------------------------------------------------------------
// the kernel: the whole bound set, at every arm
// ---------------------------------------------------------------------------

/// What one arm reports. Every field comes from a different accessor, so a
/// view that implemented only incidence could not produce it — which is the
/// reason `GraphKernel` bounds on `VertexList`, `EdgeList` and `Endpoints`
/// and not on `GraphRef` alone.
#[derive(Debug, PartialEq, Eq)]
struct Report {
    n_vertices: usize,
    n_edges: usize,
    vertices: Vec<usize>,
    /// `(id, source, target)`, in this view's orientation.
    edges: Vec<(usize, usize, usize)>,
    /// `out_edges(anchor)` as `(edge id, neighbour)`.
    out: Vec<(usize, usize)>,
    /// `find_edge(from, to)` as `(id, source, target)`.
    found: Option<(usize, usize, usize)>,
    /// `endpoints(e3)`, in this view's orientation.
    e3: Option<(usize, usize)>,
    /// `out_degree(anchor)`, which must agree with `out`.
    out_degree: usize,
}

#[derive(Clone, Copy)]
struct Probe {
    anchor: VertexId,
    from: VertexId,
    to: VertexId,
}

impl GraphKernel for Probe {
    type Out = Report;

    fn call<G>(self, g: G) -> Report
    where
        G: GraphRef + VertexList + EdgeList + Endpoints,
    {
        Report {
            n_vertices: g.num_vertices(),
            n_edges: g.num_edges(),
            vertices: g.vertices().map(VertexId::index).collect(),
            edges: g
                .edges()
                .map(|e| (e.id().index(), e.source().index(), e.target().index()))
                .collect(),
            out: g
                .out_edges(self.anchor)
                .map(|i| (i.edge.index(), i.other.index()))
                .collect(),
            found: g
                .find_edge(self.from, self.to)
                .map(|e| (e.id().index(), e.source().index(), e.target().index())),
            e3: g
                .endpoints(gt_core::ids::EdgeId::from_index(3))
                .map(|(s, t)| (s.index(), t.index())),
            out_degree: g.out_degree(self.anchor),
        }
    }
}

/// `in_edges(anchor)` as `(edge id, neighbour)`.
#[derive(Clone, Copy)]
struct InProbe(VertexId);

impl BidiGraphKernel for InProbe {
    type Out = Vec<(usize, usize)>;

    fn call<G>(self, g: G) -> Vec<(usize, usize)>
    where
        G: Bidirectional + VertexList + EdgeList + Endpoints,
    {
        // `in_degree` is asserted against the iterator here rather than in a
        // separate test: `in_degree(u, filt_graph)` counting one range while
        // `in_edges` yields another is defect #16's shape.
        let got: Vec<_> = g
            .in_edges(self.0)
            .map(|i| (i.edge.index(), i.other.index()))
            .collect();
        assert_eq!(got.len(), g.in_degree(self.0), "in_degree disagrees");
        got
    }
}

fn probe() -> Probe {
    Probe {
        anchor: v(0),
        from: v(1),
        to: v(0),
    }
}

fn run(kind: ViewKind) -> Report {
    let g = fixture();
    let (vmask, emask) = masks(&g);
    AnyGraph::filtered(&g, kind, &vmask, &emask)
        .dispatch(probe())
        .expect("dispatch")
}

// ---------------------------------------------------------------------------
// the six arms
// ---------------------------------------------------------------------------

/// `&AdjList` — storage orientation, `out_edges` is the out-half.
#[test]
fn the_directed_arm_is_the_storage_graph() {
    assert_eq!(
        run(ViewKind::Directed),
        Report {
            n_vertices: 4,
            n_edges: 5,
            vertices: vec![0, 1, 2, 3],
            // `edge_iterator` walks each block's out-half in vertex order
            // (`graph_adjacency.hh:381-409`), so the parallel edge follows its
            // twin and the global list holds every edge exactly once.
            edges: vec![(0, 0, 1), (3, 0, 1), (1, 1, 2), (2, 2, 0), (4, 3, 3)],
            out: vec![(0, 1), (3, 1)],
            // No 1 -> 0 edge exists in the directed view.
            found: None,
            e3: Some((0, 1)),
            out_degree: 2,
        }
    );
}

/// `Und<&AdjList>` — `out_edges` is `all_edges`, so `2 -> 0` is visible from
/// `0`. `undirected_adaptor::out_edges` returns the whole incidence range
/// (`graph_adaptor.hh:180-196`) for the same reason.
#[test]
fn the_undirected_arm_walks_both_halves() {
    assert_eq!(
        run(ViewKind::Undirected),
        Report {
            n_vertices: 4,
            n_edges: 5,
            vertices: vec![0, 1, 2, 3],
            // `edges(undirected_adaptor)` forwards to the base: each edge
            // once, in storage orientation.
            edges: vec![(0, 0, 1), (3, 0, 1), (1, 1, 2), (2, 2, 0), (4, 3, 3)],
            out: vec![(0, 1), (3, 1), (2, 2)],
            // Tries both orientations and reports the hit in *storage*
            // orientation, where `edge(u, v, undirected_adaptor)` swaps the
            // endpoints of a reverse hit (`graph_adaptor.hh:161`) and so
            // disagrees with its own iteration.
            found: Some((0, 0, 1)),
            e3: Some((0, 1)),
            out_degree: 3,
        }
    );
}

/// `Rev<&AdjList>` — `out_edges` is `in_edges`, and every endpoint pair is
/// exchanged.
#[test]
fn the_reversed_arm_exchanges_both_ends() {
    assert_eq!(
        run(ViewKind::Reversed),
        Report {
            n_vertices: 4,
            n_edges: 5,
            vertices: vec![0, 1, 2, 3],
            edges: vec![(0, 1, 0), (3, 1, 0), (1, 2, 1), (2, 0, 2), (4, 3, 3)],
            out: vec![(2, 2)],
            found: Some((0, 1, 0)),
            e3: Some((1, 0)),
            out_degree: 1,
        }
    );
}

/// `Filtered<&AdjList, MaskFilter>` — `e0` is gone from the edge list, from
/// `out_edges(0)` and from `endpoints`.
///
/// `source(e, filt_graph)` and `target(e, filt_graph)` forward unconditionally
/// (`graph_filtered.hh:341-355`), so in C++ a filtered-out descriptor still
/// resolves; here `endpoints(e0)` is `None` by the same predicate `edges()`
/// uses.
#[test]
fn the_directed_filtered_arm_drops_the_masked_edge() {
    assert_eq!(
        run(ViewKind::DirectedFiltered),
        Report {
            n_vertices: 4,
            n_edges: 4,
            vertices: vec![0, 1, 2, 3],
            edges: vec![(3, 0, 1), (1, 1, 2), (2, 2, 0), (4, 3, 3)],
            out: vec![(3, 1)],
            found: None,
            e3: Some((0, 1)),
            out_degree: 1,
        }
    );
}

/// `Filtered<Und<&AdjList>, MaskFilter>` — the scan, not a filtered lookup.
///
/// `find_edge(1, 0)` meets `e0` first, which the mask removes, and must go on
/// to `e3`. `inner.find_edge(..).filter(..)` would answer `None` here, and a
/// lookup disagreeing with the view's own `out_edges` is defect #16.
#[test]
fn the_undirected_filtered_arm_scans_past_the_masked_parallel_edge() {
    assert_eq!(
        run(ViewKind::UndirectedFiltered),
        Report {
            n_vertices: 4,
            n_edges: 4,
            vertices: vec![0, 1, 2, 3],
            edges: vec![(3, 0, 1), (1, 1, 2), (2, 2, 0), (4, 3, 3)],
            out: vec![(3, 1), (2, 2)],
            found: Some((3, 0, 1)),
            e3: Some((0, 1)),
            out_degree: 2,
        }
    );
}

/// `Filtered<Rev<&AdjList>, MaskFilter>` — the same scan, reported in the
/// reversed view's own orientation.
#[test]
fn the_reversed_filtered_arm_reports_reversed_endpoints() {
    assert_eq!(
        run(ViewKind::ReversedFiltered),
        Report {
            n_vertices: 4,
            n_edges: 4,
            vertices: vec![0, 1, 2, 3],
            edges: vec![(3, 1, 0), (1, 2, 1), (2, 0, 2), (4, 3, 3)],
            out: vec![(2, 2)],
            found: Some((3, 1, 0)),
            e3: Some((1, 0)),
            out_degree: 1,
        }
    );
}

/// All six arms agree on what they do *not* depend on: the vertex set, and
/// the identity of the underlying storage.
#[test]
fn every_arm_runs_and_reports_the_same_vertex_set() {
    for kind in ViewKind::ALL {
        let r = run(kind);
        assert_eq!(r.vertices, vec![0, 1, 2, 3], "{kind:?}");
        assert_eq!(r.n_vertices, r.vertices.len(), "{kind:?}");
        // The honest identity `gt_core::design` section 4 requires: the memoised
        // count and the iterator agree. `filt_graph` returns the unfiltered
        // count (`graph_filtered.hh:316`) and loses it.
        assert_eq!(r.n_edges, r.edges.len(), "{kind:?}");
        assert_eq!(r.out_degree, r.out.len(), "{kind:?}");
    }
}

/// A self-loop is stored in both halves of its vertex's block
/// (`graph_adjacency.hh:1192-1215`), so the undirected view sees it twice and
/// the directed one once. The edge list still holds it exactly once.
#[test]
fn a_self_loop_is_seen_twice_from_the_undirected_view() {
    let g = fixture();
    let loop_probe = |kind| {
        AnyGraph::new(&g, kind)
            .dispatch(Probe {
                anchor: v(3),
                from: v(3),
                to: v(3),
            })
            .expect("dispatch")
    };
    assert_eq!(loop_probe(ViewKind::Directed).out, vec![(4, 3)]);
    assert_eq!(loop_probe(ViewKind::Reversed).out, vec![(4, 3)]);
    assert_eq!(loop_probe(ViewKind::Undirected).out, vec![(4, 3), (4, 3)]);
    assert_eq!(loop_probe(ViewKind::Undirected).out_degree, 2);
    assert_eq!(loop_probe(ViewKind::Undirected).n_edges, 5);
}

/// An unfiltered [`AnyGraph::new`] carries empty masks, which the three
/// unfiltered arms never look at.
#[test]
fn the_unfiltered_arms_ignore_the_masks() {
    let g = fixture();
    for kind in [ViewKind::Directed, ViewKind::Undirected, ViewKind::Reversed] {
        let bare = AnyGraph::new(&g, kind).dispatch(probe()).expect("dispatch");
        assert_eq!(bare.n_edges, 5, "{kind:?}");
    }
}

// ---------------------------------------------------------------------------
// dispatch_bidi
// ---------------------------------------------------------------------------

/// `always_directed` (`graph_filtering.hh:135-138`) is
/// `{directed} x {reversed, not} x {filtered, not}`: four views, each of which
/// has real predecessors.
#[test]
fn dispatch_bidi_runs_on_every_directed_view() {
    let g = fixture();
    let (vmask, emask) = masks(&g);
    let bidi = |kind| {
        AnyGraph::filtered(&g, kind, &vmask, &emask)
            .dispatch_bidi(InProbe(v(0)))
            .expect("dispatch_bidi")
    };

    // `in_edges(0)` is the in-half: only `2 -> 0`.
    assert_eq!(bidi(ViewKind::Directed), vec![(2, 2)]);
    // Reversed: predecessors are the base's successors, both parallel edges.
    assert_eq!(bidi(ViewKind::Reversed), vec![(0, 1), (3, 1)]);
    assert_eq!(bidi(ViewKind::DirectedFiltered), vec![(2, 2)]);
    // ... with `e0` masked out.
    assert_eq!(bidi(ViewKind::ReversedFiltered), vec![(3, 1)]);

    assert_eq!(ViewKind::DIRECTED.len(), 4);
    for kind in ViewKind::DIRECTED {
        assert!(kind.is_directed(), "{kind:?}");
    }
}

/// The two undirected views are refused, not answered.
///
/// `in_edges(v, undirected_adaptor)` returns `make_pair(iter_t(), iter_t())`
/// — a default-constructed *empty* range (`graph_adaptor.hh:219-227`) — so in
/// C++ a predecessor-using algorithm on an undirected view silently computes
/// the wrong answer. Here the arm does not exist.
#[test]
fn dispatch_bidi_refuses_the_undirected_views() {
    let g = fixture();
    let (vmask, emask) = masks(&g);
    for kind in [ViewKind::Undirected, ViewKind::UndirectedFiltered] {
        let err = AnyGraph::filtered(&g, kind, &vmask, &emask)
            .dispatch_bidi(InProbe(v(0)))
            .expect_err("an undirected view has no predecessors");
        assert_eq!(
            err,
            ViewError::NotDirected {
                offered: kind,
                accepted: &ViewKind::DIRECTED,
            }
        );
        // The accepted set is in the message, unlike
        // "This is a graph_tool bug. :-(" (`dispatch.hh:86-88`).
        let text = err.to_string();
        assert!(text.contains("Directed"), "{text}");
        assert!(text.contains(&format!("{kind:?}")), "{text}");
    }
}

/// Every kind is either a `dispatch_bidi` arm or a refusal, and the two sets
/// partition [`ViewKind::ALL`].
#[test]
fn the_bidi_arms_and_the_refusals_partition_the_views() {
    let g = fixture();
    let (vmask, emask) = masks(&g);
    let mut ran = 0;
    let mut refused = 0;
    for kind in ViewKind::ALL {
        match AnyGraph::filtered(&g, kind, &vmask, &emask).dispatch_bidi(InProbe(v(0))) {
            Ok(_) => {
                assert!(kind.is_directed(), "{kind:?}");
                ran += 1;
            }
            Err(e) => {
                assert!(!kind.is_directed(), "{kind:?}");
                assert!(matches!(e, ViewError::NotDirected { .. }), "{e:?}");
                refused += 1;
            }
        }
    }
    assert_eq!((ran, refused), (4, 2));
}

// ---------------------------------------------------------------------------
// masks
// ---------------------------------------------------------------------------

/// A mask shorter than the *unfiltered* index bound is refused by each
/// filtered arm, and by none of the unfiltered ones.
///
/// `graph_filtering.cc:42-46` reserves against `get_edge_index_range()` and
/// `num_vertices(*u)` separately from `MaskFilter::operator()`
/// (`graph_filtering.hh:51-56`), which then reads an unchecked map
/// (`fast_vector_property_map.hh:218-221`, defect #11).
#[test]
fn a_short_mask_is_refused_by_the_filtered_arms() {
    let g = fixture();
    let vmask = vec![1u8; 4];
    let short_e = vec![1u8; 3];

    for kind in [
        ViewKind::DirectedFiltered,
        ViewKind::UndirectedFiltered,
        ViewKind::ReversedFiltered,
    ] {
        let err = AnyGraph::filtered(&g, kind, &vmask, &short_e)
            .dispatch(probe())
            .expect_err("a short edge mask");
        assert_eq!(
            err,
            ViewError::Mask(PropError::ShortMask { have: 3, need: 5 }),
            "{kind:?}"
        );
    }

    let short_v = vec![1u8; 2];
    let emask = vec![1u8; 5];
    let err = AnyGraph::filtered(&g, ViewKind::DirectedFiltered, &short_v, &emask)
        .dispatch(probe())
        .expect_err("a short vertex mask");
    assert_eq!(
        err,
        ViewError::Mask(PropError::ShortMask { have: 2, need: 4 })
    );

    // The unfiltered arms hold no mask at all and cannot fail this way.
    for kind in [ViewKind::Directed, ViewKind::Undirected, ViewKind::Reversed] {
        assert!(
            AnyGraph::filtered(&g, kind, &short_v, &short_e)
                .dispatch(probe())
                .is_ok(),
            "{kind:?}"
        );
    }
}

/// A mask that removes a *vertex* removes every edge incident to it, in the
/// count and in the iteration alike.
#[test]
fn a_vertex_mask_removes_its_incident_edges() {
    let g = fixture();
    let mut vmask = vec![1u8; 4];
    vmask[2] = 0; // drop vertex 2, and with it `1 -> 2` and `2 -> 0`
    let emask = vec![1u8; 5];
    let r = AnyGraph::filtered(&g, ViewKind::DirectedFiltered, &vmask, &emask)
        .dispatch(probe())
        .expect("dispatch");
    assert_eq!(r.n_vertices, 3);
    assert_eq!(r.vertices, vec![0, 1, 3]);
    assert_eq!(r.edges, vec![(0, 0, 1), (3, 0, 1), (4, 3, 3)]);
    assert_eq!(r.n_edges, r.edges.len());
    // `out_edges(0)` must not offer `2` as a neighbour either.
    assert_eq!(r.out, vec![(0, 1), (3, 1)]);
}

// ---------------------------------------------------------------------------
// DynGraph
// ---------------------------------------------------------------------------

/// Internal iteration visits exactly the multiset `out_edges` yields.
///
/// One indirect call per *vertex*, not per edge: the `fold` inside
/// `for_each_out` stays monomorphised in the callee. The point of the
/// assertion is that erasing the view changes nothing observable.
fn dyn_agrees_with_the_static_view<G: GraphRef + VertexList>(g: G, label: &str) {
    let erased: &dyn DynGraph = &g;

    assert_eq!(erased.num_vertices(), g.num_vertices(), "{label}");
    assert_eq!(erased.num_edges(), g.num_edges(), "{label}");

    let mut walked = Vec::new();
    erased.for_each_vertex(&mut |v| walked.push(v.index()));
    let direct: Vec<_> = g.vertices().map(VertexId::index).collect();
    assert_eq!(walked, direct, "{label}");

    let mut total = 0;
    for v in g.vertices() {
        let mut seen = Vec::new();
        erased.for_each_out(v, &mut |i| seen.push((i.edge.index(), i.other.index())));
        let mut expect: Vec<_> = g
            .out_edges(v)
            .map(|i| (i.edge.index(), i.other.index()))
            .collect();
        // The *multiset*: parallel edges and self-loops make the sequence
        // repeat, and a comparison that de-duplicated would not notice.
        assert_eq!(seen.len(), expect.len(), "{label} at {}", v.index());
        total += seen.len();
        let mut seen_sorted = seen;
        seen_sorted.sort_unstable();
        expect.sort_unstable();
        assert_eq!(seen_sorted, expect, "{label} at {}", v.index());
    }
    assert!(total > 0, "{label}: nothing was visited");
}

#[test]
fn dyn_graph_visits_the_same_multiset_as_out_edges() {
    let g = fixture();
    let (vmask, emask) = masks(&g);

    dyn_agrees_with_the_static_view(&g, "directed");
    dyn_agrees_with_the_static_view((&g).undirect(), "undirected");
    dyn_agrees_with_the_static_view((&g).reverse(), "reversed");
    dyn_agrees_with_the_static_view(
        gt_py::dispatch::as_filtered(&g, &vmask, &emask).expect("masked"),
        "directed+filtered",
    );
    dyn_agrees_with_the_static_view(
        gt_py::dispatch::as_filtered((&g).undirect(), &vmask, &emask).expect("masked"),
        "undirected+filtered",
    );
    dyn_agrees_with_the_static_view(
        gt_py::dispatch::as_filtered((&g).reverse(), &vmask, &emask).expect("masked"),
        "reversed+filtered",
    );
}

/// Summing `for_each_out` over an undirected view counts every edge twice and
/// every self-loop twice as well, which is exactly why `num_edges` is not
/// defined by degree-summation anywhere in this port.
#[test]
fn dyn_out_degrees_sum_to_twice_the_undirected_edge_count() {
    let g = fixture();
    let u = (&g).undirect();
    let erased: &dyn DynGraph = &u;
    let mut n = 0;
    erased.for_each_vertex(&mut |v| {
        let mut deg = 0;
        erased.for_each_out(v, &mut |_| deg += 1);
        n += deg;
    });
    assert_eq!(n, 2 * erased.num_edges());
    assert_eq!(erased.num_edges(), 5);
}

// ---------------------------------------------------------------------------
// the inventories, and the negative guarantee
// ---------------------------------------------------------------------------

/// [`ViewKind::ALL`] and [`ViewKind::DIRECTED`] are checked against exhaustive
/// matches at compile time (`error[E0080]` on drift); this only pins the
/// cardinalities a reader expects.
#[test]
fn the_view_inventory_is_the_cartesian_product_minus_one_corner() {
    // 2 (directed) x 2 (reversed) x 2 (filtered) = 8, minus the two
    // undirected-and-reversed corners `hana::filter` drops
    // (`graph_filtering.hh:112-118`).
    assert_eq!(ViewKind::ALL.len(), 6);
    assert_eq!(ViewKind::DIRECTED.len(), 4);
    for (i, kind) in ViewKind::ALL.into_iter().enumerate() {
        assert_eq!(kind.rank(), i);
        assert_eq!(kind.is_directed(), ViewKind::DIRECTED.contains(&kind));
    }
}

/// Defect #48's *eliminated* half: a seventh view is `error[E0004]`, where
/// `dispatch.hh:86-88` gives a runtime `DispatchNotFound` reading "This is a
/// graph_tool bug. :-(". The pass fixture pins that the real `ViewKind` still
/// admits a six-arm match with no catch-all.
#[test]
fn a_seventh_view_kind_does_not_compile() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/u31_six_arms_are_exhaustive.rs");
    t.compile_fail("tests/ui/u31_seventh_view_kind.rs");
}
