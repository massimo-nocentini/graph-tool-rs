//! The view algebra: directed x reversed x filtered, as six types.
//!
//! ## The six, and only the six (DESIGN.md D3)
//!
//! `graph_filtering.hh:67-96` builds six types from three booleans by a
//! `std::conditional_t` tower, drops the impossible corner with a
//! `hana::filter` over a value-level tuple (`:112-118`), and de-duplicates the
//! rest with `hana::to<set_tag>` (`:127`).
//!
//! Here the wrapper types have **private fields** and are reachable only
//! through [`Undirect::undirect`] and [`Reverse::reverse`], whose impls make
//! `undirect` idempotent and absorbing and `reverse` an involution. So
//! `Rev<Und<_>>` is ill-formed (the struct bound rejects it), and `Und<Und<_>>`,
//! `Rev<Rev<_>>` and the whole tower above them are *unnameable* rather than
//! merely unused. That is strictly stronger than the C++: it closes the cases
//! `hana::to<set_tag>` exists to de-duplicate, not only the case the filter
//! removes.
//!
//! ## The flattening that was rejected
//!
//! One competing design collapsed all six into a single
//! `GraphView<'g, const DIRECTED: bool, const REVERSED: bool, const FILTERED: bool>`
//! with const-folded branches. Const-folding is real and measurable, but the
//! flattening is also what destroys edge orientation: a symmetric
//! `{src, tgt, idx}` descriptor cannot record "stored 2->1, traversed from 1",
//! which is the information `undirected_adaptor::source/target`
//! (`graph_adaptor.hh:97-112`) carries. Composed wrappers plus an *anchored*
//! [`Incident`](crate::adj::Incident) keep the orientation and get the same
//! monomorphisation, because `Und<G>::Out = G::All` is resolved at the type
//! level exactly as a const parameter would be.

mod filtered;
mod reversed;
mod undirected;

pub use filtered::{Filter, Filtered, KeepAll, MaskFilter};
pub use reversed::{Rev, Reverse};
pub use undirected::{Und, Undirect};

// ===========================================================================
// U7 — the view algebra, exercised as values.
//
// These live inside the crate, as `graph.rs`'s do and for the same reason:
// `Bound::new` and `EdgeRef::new` are `pub(crate)` (defects #7 and #41), so
// only storage gt-core itself blesses can present itself as a graph, and a
// test graph is storage. `tests/u07_views.rs` carries everything that *can*
// be asserted from outside — layout, the type table, and the compile-fail
// fixture — and nothing that needs a populated graph, because `AdjList`'s
// mutators are U5's.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::filtered::keeps_edge;
    use super::*;
    use crate::adj::{EdgeRef, Incident};
    use crate::bound::{Bound, EdgeBound, VertexBound};
    use crate::dir::{Directed, HasDir};
    use crate::error::PropError;
    use crate::graph::{Bidirectional, EdgeList, Endpoints, GraphBase, GraphRef, VertexList};
    use crate::ids::{EdgeId, GraphId, VertexId};

    // -- storage ------------------------------------------------------------

    macro_rules! incident_iter {
        ($name:ident, $inner:ty) => {
            #[derive(Clone)]
            struct $name<'a>($inner);
            impl Iterator for $name<'_> {
                type Item = Incident;
                fn next(&mut self) -> Option<Incident> {
                    self.0.next().copied()
                }
                fn size_hint(&self) -> (usize, Option<usize>) {
                    self.0.size_hint()
                }
            }
        };
    }

    incident_iter!(MockOut, std::slice::Iter<'a, Incident>);
    incident_iter!(MockIn, std::slice::Iter<'a, Incident>);
    incident_iter!(
        MockAll,
        std::iter::Chain<std::slice::Iter<'a, Incident>, std::slice::Iter<'a, Incident>>
    );

    #[derive(Clone)]
    struct MockVertices(std::ops::Range<usize>);
    impl Iterator for MockVertices {
        type Item = VertexId;
        fn next(&mut self) -> Option<VertexId> {
            self.0.next().map(VertexId::from_index)
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.0.size_hint()
        }
    }

    #[derive(Clone)]
    struct MockEdges<'a>(std::slice::Iter<'a, EdgeRef>);
    impl Iterator for MockEdges<'_> {
        type Item = EdgeRef;
        fn next(&mut self) -> Option<EdgeRef> {
            self.0.next().copied()
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.0.size_hint()
        }
    }

    /// Out- and in-halves held separately, so "which half did the view read"
    /// stays answerable, and `all` is `out ++ in` — the layout
    /// `graph_adjacency.hh:1192-1215` produces, self-loops included (an
    /// `add_edge(s, s)` pushes into both halves of `s`, so it is seen twice
    /// from `s`).
    #[derive(Debug)]
    struct Mock {
        id: GraphId,
        out: Vec<Vec<Incident>>,
        inc: Vec<Vec<Incident>>,
        edges: Vec<EdgeRef>,
    }

    fn v(i: usize) -> VertexId {
        VertexId::from_index(i)
    }
    fn e(i: usize) -> EdgeId {
        EdgeId::from_index(i)
    }

    fn mock(n: usize, spec: &[(usize, usize)]) -> Mock {
        let mut m = Mock {
            id: GraphId::fresh(),
            out: vec![Vec::new(); n],
            inc: vec![Vec::new(); n],
            edges: Vec::new(),
        };
        for (i, &(s, t)) in spec.iter().enumerate() {
            m.out[s].push(Incident {
                other: v(t),
                edge: e(i),
            });
            m.inc[t].push(Incident {
                other: v(s),
                edge: e(i),
            });
            m.edges.push(EdgeRef::new(e(i), v(s), v(t)));
        }
        m
    }

    /// The acceptance path: `0 -e0-> 1 -e1-> 2`.
    fn path() -> Mock {
        mock(3, &[(0, 1), (1, 2)])
    }

    /// Parallel edges (`e1`, `e5`), a self-loop (`e4`) and an isolated vertex
    /// (`4`) — the three shapes a filtered count gets wrong.
    ///
    /// `e0: 0->1`, `e1: 1->2`, `e2: 0->2`, `e3: 2->0`, `e4: 3->3`, `e5: 1->2`.
    fn rich() -> Mock {
        mock(5, &[(0, 1), (1, 2), (0, 2), (2, 0), (3, 3), (1, 2)])
    }

    impl HasDir for &Mock {
        type Dir = Directed;
    }

    impl GraphBase for &Mock {
        fn graph_id(self) -> GraphId {
            self.id
        }
        fn num_vertices(self) -> usize {
            self.out.len()
        }
        fn num_edges(self) -> usize {
            self.edges.len()
        }
        fn vertex_bound(self) -> VertexBound {
            Bound::new(self.id, self.out.len())
        }
        fn edge_bound(self) -> EdgeBound {
            Bound::new(self.id, self.edges.len())
        }
    }

    impl<'g> GraphRef for &'g Mock {
        type Out = MockOut<'g>;
        type All = MockAll<'g>;
        fn out_edges(self, u: VertexId) -> MockOut<'g> {
            MockOut(self.out[u.index()].iter())
        }
        fn all_edges(self, u: VertexId) -> MockAll<'g> {
            MockAll(self.out[u.index()].iter().chain(self.inc[u.index()].iter()))
        }
        fn out_degree(self, u: VertexId) -> usize {
            self.out[u.index()].len()
        }
        fn degree(self, u: VertexId) -> usize {
            self.out[u.index()].len() + self.inc[u.index()].len()
        }
    }

    impl<'g> Bidirectional for &'g Mock {
        type In = MockIn<'g>;
        fn in_edges(self, u: VertexId) -> MockIn<'g> {
            MockIn(self.inc[u.index()].iter())
        }
        fn in_degree(self, u: VertexId) -> usize {
            self.inc[u.index()].len()
        }
    }

    impl VertexList for &Mock {
        type Vertices = MockVertices;
        fn vertices(self) -> MockVertices {
            MockVertices(0..self.out.len())
        }
    }

    impl<'g> EdgeList for &'g Mock {
        type Edges = MockEdges<'g>;
        fn edges(self) -> MockEdges<'g> {
            MockEdges(self.edges.iter())
        }
    }

    impl Endpoints for &Mock {
        fn endpoints(self, i: EdgeId) -> Option<(VertexId, VertexId)> {
            self.edges.get(i.index()).map(|r| (r.source(), r.target()))
        }
        fn opposite(self, i: EdgeId, u: VertexId) -> Option<VertexId> {
            self.edges.get(i.index()).and_then(|r| r.opposite(u))
        }
        fn find_edge(self, s: VertexId, t: VertexId) -> Option<EdgeRef> {
            self.edges
                .iter()
                .find(|r| r.source() == s && r.target() == t)
                .copied()
        }
    }

    impl<'g> Reverse for &'g Mock {
        type Out = Rev<&'g Mock>;
        fn reverse(self) -> Rev<&'g Mock> {
            Rev::wrap(self)
        }
    }

    impl<'g> Undirect for &'g Mock {
        type Out = Und<&'g Mock>;
        fn undirect(self) -> Und<&'g Mock> {
            Und::wrap(self)
        }
    }

    // -- helpers ------------------------------------------------------------

    fn pairs<I: Iterator<Item = Incident>>(it: I) -> Vec<(usize, usize)> {
        it.map(|i| (i.other.index(), i.edge.index())).collect()
    }
    fn sorted<I: Iterator<Item = Incident>>(it: I) -> Vec<(usize, usize)> {
        let mut p = pairs(it);
        p.sort_unstable();
        p
    }
    fn edge_ids<I: Iterator<Item = EdgeRef>>(it: I) -> Vec<usize> {
        it.map(|r| r.id().index()).collect()
    }

    /// Defect #15/#16 in one line, for any view: the two numbers and the two
    /// iterations must agree. `filt_graph` reports the *unfiltered* counts
    /// (`graph_filtered.hh:314-326`) and concedes at `:301-312` that the
    /// identity is thereby lost.
    fn counts_agree<G: GraphRef + VertexList + EdgeList>(g: G, what: &str) {
        assert_eq!(
            g.num_vertices(),
            g.vertices().count(),
            "{what}: num_vertices() != vertices().count()"
        );
        assert_eq!(
            g.num_edges(),
            g.edges().count(),
            "{what}: num_edges() != edges().count()"
        );
        // `fold` is a second implementation of `next` on every filtering
        // adaptor, so it is pinned against it rather than trusted.
        assert_eq!(g.vertices().fold(0, |n, _| n + 1), g.num_vertices());
        assert_eq!(g.edges().fold(0, |n, _| n + 1), g.num_edges());
    }

    /// Every edge of a *directed* view is counted exactly once, at its source.
    fn degrees_sum_to_edges<G>(g: G, what: &str)
    where
        G: GraphRef + VertexList + EdgeList + HasDir<Dir = Directed>,
    {
        let sum: usize = g.vertices().map(|u| g.out_degree(u)).sum();
        assert_eq!(sum, g.num_edges(), "{what}: sum(out_degree) != num_edges");
        let by_iter: usize = g.vertices().map(|u| g.out_edges(u).count()).sum();
        assert_eq!(sum, by_iter, "{what}: out_degree != out_edges().count()");
    }

    const ALL_V: [u8; 5] = [1, 1, 1, 1, 1];
    const ALL_E: [u8; 6] = [1, 1, 1, 1, 1, 1];
    /// vertex 2 removed.
    const NO_V2: [u8; 5] = [1, 1, 0, 1, 1];
    /// edge 5 (the parallel `1->2`) removed.
    const NO_E5: [u8; 6] = [1, 1, 1, 1, 1, 0];
    /// vertex 0 and edge 1 removed.
    const NO_V0: [u8; 5] = [0, 1, 1, 1, 1];
    const NO_E1: [u8; 6] = [1, 0, 1, 1, 1, 1];

    // =======================================================================
    // 1. Defect #14 — the single most important assertion in this unit
    // =======================================================================

    /// `out_edges(u, undirected_adaptor)` routes to `_all_edges_out`
    /// (`graph_adaptor.hh:199-207`), which `graph_adjacency.hh:1102-1108`
    /// implements with an **out**-edge iterator over the whole run, so
    /// `make_out_edge::def` sets `src == u` for the in-half too: every yielded
    /// edge is anchored at the query vertex and `other` is the neighbour.
    /// Read backwards — wiring `out_edges` to a canonical all-edges iterator —
    /// `target(e)` becomes the query vertex itself for half the edges.
    ///
    /// On the path `0 -> 1 -> 2` that is the whole difference between
    /// `{0, 2}` and `{1}`.
    #[test]
    fn undirected_out_edges_yield_the_neighbour_never_the_query_vertex() {
        let p = path();
        let u = (&p).undirect();

        // out-half then in-half: `1 -e1-> 2` before `0 -e0-> 1`.
        assert_eq!(pairs(u.out_edges(v(1))), [(2, 1), (0, 0)]);
        let others: Vec<usize> = u.out_edges(v(1)).map(|i| i.other.index()).collect();
        assert_eq!(sorted(u.out_edges(v(1))), [(0, 0), (2, 1)]);
        assert!(
            !others.contains(&1),
            "out_edges(1) yielded the query vertex: {others:?}"
        );
        assert_eq!(u.out_degree(v(1)), 2);
        assert_eq!(u.degree(v(1)), 2);

        // The endpoints are the two ends of the edge, and `other` is the one
        // that is not the anchor — for every vertex, in every filtered form.
        for &keep_all in &[true, false] {
            let vm = [1u8, 1, 1];
            let em = [1u8, 1];
            let f = Filtered::masked((&p).undirect(), &vm, &em).expect("masks fit");
            for w in (&p).vertices() {
                let it: Vec<Incident> = if keep_all {
                    u.out_edges(w).collect()
                } else {
                    f.out_edges(w).collect()
                };
                for i in it {
                    let (s, t) = (&p).endpoints(i.edge).expect("live edge");
                    assert!(
                        s == w || t == w,
                        "edge {:?} is not incident to {w:?}",
                        i.edge
                    );
                    assert_ne!(
                        i.other, w,
                        "out_edges({w:?}) yielded the query vertex for {:?}",
                        i.edge
                    );
                    assert_eq!(i.other, if s == w { t } else { s });
                }
            }
        }
    }

    // =======================================================================
    // 2. `Rev` is the swap, not the union
    // =======================================================================

    /// `graph_reverse.hh:78-80` swaps the `out_edge_iterator` and
    /// `in_edge_iterator` typedefs: a reversed view's out-edges are the base's
    /// in-edges **only**. The union would be an undirected view wearing a
    /// directed type — and on `rich()` it is a different multiset, not merely
    /// a different order.
    #[test]
    fn reversed_out_edges_are_the_in_edges_and_not_their_union() {
        let m = rich();
        let r = (&m).reverse();
        for u in (&m).vertices() {
            assert_eq!(
                sorted(r.out_edges(u)),
                sorted((&m).in_edges(u)),
                "rev.out_edges({u:?})"
            );
            assert_eq!(sorted(r.in_edges(u)), sorted((&m).out_edges(u)));
            assert_eq!(r.out_degree(u), (&m).in_degree(u));
            assert_eq!(r.in_degree(u), (&m).out_degree(u));

            let union_len = (&m).out_degree(u) + (&m).in_degree(u);
            assert_eq!(r.degree(u), union_len);
            if (&m).out_degree(u) > 0 && (&m).in_degree(u) > 0 {
                assert_ne!(
                    r.out_edges(u).count(),
                    union_len,
                    "rev.out_edges({u:?}) is the union"
                );
            }
        }
        // And through the filter, where `in_edge_pred` is edge-and-source:
        let f = Filtered::masked((&m).reverse(), &ALL_V, &ALL_E).expect("masks fit");
        for u in (&m).vertices() {
            assert_eq!(sorted(f.out_edges(u)), sorted((&m).in_edges(u)));
            assert_eq!(f.out_degree(u), (&m).in_degree(u));
        }
    }

    // =======================================================================
    // 3. Six views, four masks: the counts are the iterations
    // =======================================================================

    #[test]
    fn every_view_counts_exactly_what_it_iterates() {
        let m = rich();
        let cases: [(&str, &[u8; 5], &[u8; 6]); 4] = [
            ("keep all", &ALL_V, &ALL_E),
            ("no vertex 2", &NO_V2, &ALL_E),
            ("no edge 5", &ALL_V, &NO_E5),
            ("no vertex 0, no edge 1", &NO_V0, &NO_E1),
        ];
        for (name, vm, em) in cases {
            let d = Filtered::masked(&m, vm, em).expect("masks fit");
            let u = Filtered::masked((&m).undirect(), vm, em).expect("masks fit");
            let r = Filtered::masked((&m).reverse(), vm, em).expect("masks fit");
            counts_agree(d, &format!("{name}: directed+filtered"));
            counts_agree(u, &format!("{name}: undirected+filtered"));
            counts_agree(r, &format!("{name}: reversed+filtered"));
            degrees_sum_to_edges(d, &format!("{name}: directed+filtered"));
            degrees_sum_to_edges(r, &format!("{name}: reversed+filtered"));

            // Filtering is a property of the edge set, not of the traversal
            // orientation, so all three agree on *which* edges survive.
            assert_eq!(d.num_edges(), u.num_edges(), "{name}: und disagrees");
            assert_eq!(d.num_edges(), r.num_edges(), "{name}: rev disagrees");
            assert_eq!(d.num_vertices(), u.num_vertices());
            assert_eq!(edge_ids(d.edges()), edge_ids(u.edges()));
            assert_eq!(edge_ids(d.edges()), edge_ids(r.edges()));

            // Degree-summation as a definition of `num_edges` double-counts on
            // an undirected view: that is why `EdgeList` is a bound on
            // `Filtered::new` (DESIGN.md section 4).
            let und_sum: usize = u.vertices().map(|w| u.out_degree(w)).sum();
            assert_eq!(
                und_sum,
                2 * u.num_edges(),
                "{name}: undirected degree sum is not 2E"
            );
        }

        // The other three of the six: unfiltered, and trivially filtered.
        counts_agree(&m, "directed");
        counts_agree((&m).undirect(), "undirected");
        counts_agree((&m).reverse(), "reversed");
        counts_agree(Filtered::all(&m), "directed+KeepAll");
        counts_agree(Filtered::all((&m).undirect()), "undirected+KeepAll");
        counts_agree(Filtered::all((&m).reverse()), "reversed+KeepAll");
        degrees_sum_to_edges(&m, "directed");
        degrees_sum_to_edges(Filtered::all(&m), "directed+KeepAll");
    }

    /// The numbers themselves, so that "consistent" cannot be satisfied by a
    /// filter that quietly keeps everything — which is the failure mode of a
    /// design whose `vmask` field was read nowhere (DESIGN.md D3).
    #[test]
    fn the_filter_actually_removes_things() {
        let m = rich();
        assert_eq!(((&m).num_vertices(), (&m).num_edges()), (5, 6));

        let d = Filtered::masked(&m, &NO_V2, &ALL_E).expect("masks fit");
        // Vertex 2 gone takes e1, e2, e3 and e5 with it; e0 and the self-loop
        // e4 survive.
        assert_eq!((d.num_vertices(), d.num_edges()), (4, 2));
        assert_eq!(edge_ids(d.edges()), vec![0, 4]);

        let d = Filtered::masked(&m, &ALL_V, &NO_E5).expect("masks fit");
        assert_eq!((d.num_vertices(), d.num_edges()), (5, 5));
        assert_eq!(edge_ids(d.edges()), vec![0, 1, 2, 3, 4]);

        let d = Filtered::masked(&m, &NO_V0, &NO_E1).expect("masks fit");
        assert_eq!((d.num_vertices(), d.num_edges()), (4, 2));
        assert_eq!(edge_ids(d.edges()), vec![4, 5]);

        // `KeepAll` removes nothing and pays no prepass.
        let t = Filtered::all(&m);
        assert_eq!((t.num_vertices(), t.num_edges()), (5, 6));
        const { assert!(<KeepAll as Filter>::TRIVIAL) };
        const { assert!(!<MaskFilter<'_> as Filter>::TRIVIAL) };
        assert_eq!(size_of::<KeepAll>(), 0);
    }

    /// Per-vertex degrees under a filter, against hand-computed answers:
    /// `out_degree` probes (`graph_filtered.hh:383-392`) rather than
    /// forwarding, `degree` is the whole incidence run, and the anchor is not
    /// itself probed — which is what keeps `sum(out_degree) == num_edges`.
    #[test]
    fn filtered_degrees_are_probed_per_vertex() {
        let m = rich();
        let d = Filtered::masked(&m, &NO_V2, &ALL_E).expect("masks fit");

        assert_eq!(d.out_degree(v(0)), 1); // e0 kept, e2 -> 2 gone
        assert_eq!(d.in_degree(v(0)), 0); // e3 comes from 2
        assert_eq!(d.degree(v(0)), 1);
        assert_eq!(d.out_degree(v(1)), 0); // both 1->2 edges gone
        assert_eq!(d.in_degree(v(1)), 1);
        assert_eq!(d.degree(v(1)), 1);
        assert_eq!(d.out_degree(v(3)), 1); // the self-loop survives
        assert_eq!(d.in_degree(v(3)), 1);
        assert_eq!(d.degree(v(3)), 2); // seen twice from its own vertex
        assert_eq!(d.degree(v(4)), 0);

        // Every view: degree == in + out on a directed one, and the three
        // numbers are what the three iterators yield.
        for w in (&m).vertices() {
            assert_eq!(d.degree(w), d.in_degree(w) + d.out_degree(w));
            assert_eq!(d.out_degree(w), d.out_edges(w).count());
            assert_eq!(d.in_degree(w), d.in_edges(w).count());
            assert_eq!(d.degree(w), d.all_edges(w).count());
        }

        // The vertex that was filtered out still reports its surviving
        // out-edges, exactly as `out_edges(u, filt_graph)` does — it applies
        // `out_edge_pred`, which never tests `u`.
        let d = Filtered::masked(&m, &NO_V0, &ALL_E).expect("masks fit");
        assert_eq!(d.out_degree(v(0)), 2, "the anchor itself is not probed");
        assert_eq!(d.vertices().count(), 4);
    }

    // =======================================================================
    // 4. Bounds survive filtering; cardinalities do not
    // =======================================================================

    /// `filt_graph` forwards `num_vertices` to the unfiltered graph
    /// (`graph_filtered.hh:314-318`) *because* property storage must stay
    /// correctly sized — a real requirement answered by the wrong name. Here
    /// the requirement is `vertex_bound()`, a different type, and
    /// `num_vertices()` is free to be honest.
    #[test]
    fn filtering_moves_the_cardinality_and_leaves_the_bound_alone() {
        let m = rich();
        let d = Filtered::masked(&m, &NO_V2, &NO_E5).expect("masks fit");

        assert_eq!(d.vertex_bound(), (&m).vertex_bound());
        assert_eq!(d.edge_bound(), (&m).edge_bound());
        assert_eq!(d.vertex_bound().len(), 5);
        assert_eq!(d.edge_bound().len(), 6);
        assert_eq!(d.num_vertices(), 4);
        assert_ne!(d.num_vertices(), d.vertex_bound().len());
        assert_eq!(d.graph_id(), (&m).graph_id());

        // And through the wrappers, on all six.
        let u = Filtered::masked((&m).undirect(), &NO_V2, &NO_E5).expect("masks fit");
        let r = Filtered::masked((&m).reverse(), &NO_V2, &NO_E5).expect("masks fit");
        assert_eq!(u.vertex_bound(), (&m).vertex_bound());
        assert_eq!(r.edge_bound(), (&m).edge_bound());
        assert_eq!(Filtered::all(&m).vertex_bound(), (&m).vertex_bound());
    }

    // =======================================================================
    // 5. Masks are validated against this graph's own bounds
    // =======================================================================

    #[test]
    fn a_short_mask_is_refused_by_the_constructor() {
        let m = rich();
        let short_v = [1u8; 4];
        let short_e = [1u8; 5];

        assert_eq!(
            Filtered::masked(&m, &short_v, &ALL_E).unwrap_err(),
            PropError::ShortMask { have: 4, need: 5 }
        );
        assert_eq!(
            Filtered::masked(&m, &ALL_V, &short_e).unwrap_err(),
            PropError::ShortMask { have: 5, need: 6 }
        );
        // Longer than the bound is fine: the bound is a minimum, and a
        // property map grown for a larger graph is still indexable here.
        let long_v = [1u8; 9];
        let long_e = [1u8; 9];
        let ok = Filtered::masked(&m, &long_v, &long_e).expect("a longer mask fits");
        assert_eq!(ok.num_vertices(), 5);
        assert_eq!(ok.num_edges(), 6);
        // Both wrappers validate against the same, unfiltered bound.
        assert!(Filtered::masked((&m).undirect(), &short_v, &ALL_E).is_err());
        assert!(Filtered::masked((&m).reverse(), &ALL_V, &short_e).is_err());
    }

    /// The acceptance question, answered as it actually behaves: **`Err`**,
    /// not borrowck.
    ///
    /// `Filtered::masked` re-validates against *this* graph's bounds, so a
    /// mask admitted by a 3-vertex graph is refused by a 5-vertex one. What a
    /// mask cannot carry is identity — it is a bare `&[u8]`, like
    /// `_vertex_filter_map` — so two graphs of the *same* size accept each
    /// other's masks, here and in `graph_filtering.cc:42-46`. The remaining
    /// door is a `MaskFilter` copied out of `filter()` into `Filtered::new`,
    /// which skips validation; behind it the out-of-range probe answers
    /// "filtered out" (defect #11's `operator[]` reads out of bounds instead),
    /// and the memoised counts stay consistent because both go through the
    /// same predicate.
    #[test]
    fn a_mask_admitted_by_one_graph_is_re_validated_by_the_next() {
        let small = mock(3, &[(0, 1), (1, 2)]);
        let big = rich();
        let vm = [1u8, 1, 1];
        let em = [1u8, 1];

        let s = Filtered::masked(&small, &vm, &em).expect("masks fit the small graph");
        assert_eq!((s.num_vertices(), s.num_edges()), (3, 2));

        assert_eq!(
            Filtered::masked(&big, &vm, &em).unwrap_err(),
            PropError::ShortMask { have: 3, need: 5 },
            "the same masks must be refused by a larger graph"
        );

        // The unvalidated door, and what lies behind it.
        let leaked: MaskFilter<'_> = *s.filter();
        let f = Filtered::new(&big, leaked);
        assert_eq!(f.num_vertices(), 3, "out of range is filtered out");
        assert_eq!(f.num_edges(), 2);
        counts_agree(f, "a foreign mask");
        assert!(!leaked.keep_vertex(v(4)));
        assert!(!leaked.keep_edge(e(5)));
    }

    /// Non-zero keeps: "only the vertices with value different than `False`
    /// are kept" (`graph_tool/__init__.py:3505`).
    #[test]
    fn any_non_zero_keeps() {
        let m = rich();
        let vm = [1u8, 255, 0, 2, 1];
        let f = Filtered::masked(&m, &vm, &ALL_E).expect("masks fit");
        assert_eq!(f.num_vertices(), 4);
        assert_eq!(
            f.vertices().map(|w| w.index()).collect::<Vec<_>>(),
            vec![0, 1, 3, 4]
        );
    }

    // =======================================================================
    // 6. Resolution: `endpoints`, `opposite`, `find_edge`
    // =======================================================================

    /// `source(e, filt_graph)` forwards unconditionally
    /// (`graph_filtered.hh:341-355`), so a filtered-out edge still resolves to
    /// a pair of vertices. Here it is `None`, by the same predicate `edges()`
    /// uses.
    #[test]
    fn resolution_is_none_for_what_iteration_does_not_yield() {
        let m = rich();
        let f = Filtered::masked(&m, &NO_V2, &NO_E5).expect("masks fit");

        assert_eq!(f.endpoints(e(0)), Some((v(0), v(1))));
        assert_eq!(f.endpoints(e(1)), None, "endpoint 2 is filtered out");
        assert_eq!(f.endpoints(e(5)), None, "the edge itself is filtered out");
        assert_eq!(f.endpoints(e(99)), None, "no such edge");
        assert_eq!(f.opposite(e(0), v(0)), Some(v(1)));
        assert_eq!(f.opposite(e(0), v(2)), None, "not an endpoint");
        assert_eq!(f.opposite(e(1), v(1)), None, "the edge is not in the view");

        // Every edge `endpoints` resolves is one `edges()` yields, and no
        // other — the two are the same predicate.
        for i in 0..(&m).num_edges() {
            let resolvable = f.endpoints(e(i)).is_some();
            let yielded = edge_ids(f.edges()).contains(&i);
            assert_eq!(resolvable, yielded, "edge {i}");
        }

        // A reversed filtered view swaps the orientation, not the membership.
        let r = Filtered::masked((&m).reverse(), &NO_V2, &NO_E5).expect("masks fit");
        assert_eq!(r.endpoints(e(0)), Some((v(1), v(0))));
        assert_eq!(r.endpoints(e(1)), None);
        assert_eq!(r.opposite(e(0), v(0)), Some(v(1)));
    }

    /// `edge(u, v, filt_graph)` *scans* the parallel run
    /// (`graph_filtered.hh:502-540`), so a filtered-out first hit does not
    /// hide a surviving second one. A `find_edge` that filtered the unfiltered
    /// lookup's single answer would disagree with its own `out_edges`.
    ///
    /// The C++ scan tests `g._edge_pred(e)` only (`:529`) — the edge mask,
    /// never the endpoints — so its `edge()` returns edges whose `vertices()`
    /// has removed an endpoint. That half is not ported.
    #[test]
    fn find_edge_scans_past_a_filtered_parallel_edge() {
        let m = rich();

        // e1 and e5 are both 1->2. Remove the first.
        let f = Filtered::masked(&m, &ALL_V, &NO_E1).expect("masks fit");
        let hit = f.find_edge(v(1), v(2)).expect("e5 survives");
        assert_eq!(hit.id(), e(5));
        assert_eq!((hit.source(), hit.target()), (v(1), v(2)));
        assert!(
            f.out_edges(v(1)).any(|i| i.edge == e(5)),
            "find_edge and out_edges must agree"
        );

        // Remove both: no edge, not a stale descriptor (defect #41).
        let no_e1_e5 = [1u8, 0, 1, 1, 1, 0];
        let f = Filtered::masked(&m, &ALL_V, &no_e1_e5).expect("masks fit");
        assert!(f.find_edge(v(1), v(2)).is_none());
        // Removing an endpoint removes the edge too, unlike the C++ scan.
        let f = Filtered::masked(&m, &NO_V2, &ALL_E).expect("masks fit");
        assert!(f.find_edge(v(1), v(2)).is_none());
        assert!(f.find_edge(v(0), v(1)).is_some());
        // ... including when it is the *source* that is gone.
        let f = Filtered::masked(&m, &NO_V0, &ALL_E).expect("masks fit");
        assert!(f.find_edge(v(0), v(1)).is_none());
    }

    /// Orientation comes from the view underneath: storage order for the
    /// undirected view (defect #13 — the C++ swaps it at
    /// `graph_adaptor.hh:161` into an orientation its own iteration never
    /// yields), reversed order for the reversed one.
    #[test]
    fn filtered_find_edge_keeps_its_view_s_orientation() {
        let m = rich();
        let u = Filtered::masked((&m).undirect(), &ALL_V, &ALL_E).expect("masks fit");
        let hit = u.find_edge(v(2), v(1)).expect("seen from 2");
        assert_eq!(
            (hit.id(), hit.source(), hit.target()),
            (e(1), v(1), v(2)),
            "the undirected hit must be reported in storage orientation"
        );
        assert_eq!(
            hit.id(),
            (&m).undirect().find_edge(v(2), v(1)).unwrap().id()
        );

        let r = Filtered::masked((&m).reverse(), &ALL_V, &ALL_E).expect("masks fit");
        let hit = r.find_edge(v(1), v(0)).expect("e0 reversed");
        assert_eq!((hit.id(), hit.source(), hit.target()), (e(0), v(1), v(0)));
        assert!(r.find_edge(v(0), v(1)).is_none());

        // The trivial filter takes the underlying lookup, and must answer the
        // same question.
        let t = Filtered::all(&m);
        assert_eq!(t.find_edge(v(0), v(1)).map(|h| h.id()), Some(e(0)));
        assert!(t.find_edge(v(1), v(0)).is_none());
        let t = Filtered::all((&m).undirect());
        assert_eq!(t.find_edge(v(2), v(1)).map(|h| h.id()), Some(e(1)));
    }

    // =======================================================================
    // 7. One predicate, everywhere
    // =======================================================================

    /// `Filtered::new` and `FilterEdges` go through the same `keeps_edge`;
    /// `FilterIncident` goes through `keeps_incident`, which is the same
    /// predicate with the anchor already known kept. Two predicates is how
    /// `num_edges()` comes to disagree with `edges().count()`.
    #[test]
    fn the_incidence_predicate_is_the_edge_predicate_seen_from_a_vertex() {
        let m = rich();
        let f = Filtered::masked(&m, &NO_V2, &NO_E5).expect("masks fit");
        let filter = *f.filter();

        for r in (&m).edges() {
            let kept = keeps_edge(filter, r);
            let s_sees = f.out_edges(r.source()).any(|i| i.edge == r.id());
            let t_sees = f.in_edges(r.target()).any(|i| i.edge == r.id());
            if filter.keep_vertex(r.source()) && filter.keep_vertex(r.target()) {
                assert_eq!(kept, s_sees, "edge {:?} from its source", r.id());
                assert_eq!(kept, t_sees, "edge {:?} from its target", r.id());
            } else {
                assert!(!kept);
            }
        }
    }

    /// `fold` overrides on all three adaptors, pinned against `next` from
    /// every starting position — the check U4 applies to `IncidentIter`, for
    /// the same reason: an override is a second implementation.
    #[test]
    fn filtered_fold_agrees_with_next_from_every_start() {
        let m = rich();
        let f = Filtered::masked(&m, &NO_V2, &NO_E5).expect("masks fit");
        for skip in 0..=6 {
            let mut by_next = f.edges();
            for _ in 0..skip {
                by_next.next();
            }
            let folded = by_next.clone().fold(Vec::new(), |mut a, r| {
                a.push(r.id().index());
                a
            });
            let stepped: Vec<usize> = by_next.map(|r| r.id().index()).collect();
            assert_eq!(folded, stepped, "edges: fold disagrees after {skip}");
        }
        for skip in 0..=5 {
            let mut by_next = f.vertices();
            for _ in 0..skip {
                by_next.next();
            }
            let folded = by_next.clone().fold(Vec::new(), |mut a, w| {
                a.push(w.index());
                a
            });
            let stepped: Vec<usize> = by_next.map(|w| w.index()).collect();
            assert_eq!(folded, stepped, "vertices: fold disagrees after {skip}");
        }
        let mut it = f.all_edges(v(1));
        it.next();
        let folded = it.clone().fold(Vec::new(), |mut a, i| {
            a.push(i.edge.index());
            a
        });
        let stepped: Vec<usize> = it.map(|i| i.edge.index()).collect();
        assert_eq!(folded, stepped);

        // Exhausted stays exhausted, on the trivial arm too.
        let mut t = Filtered::all(&m).edges();
        for _ in 0..6 {
            t.next();
        }
        assert!(t.next().is_none());
        assert!(t.next().is_none());
    }

    /// The upper bound is the unfiltered length and the lower bound is zero:
    /// a filtered iterator does not know its length without walking, which is
    /// why `ExactIncidence` is a separate refinement (DESIGN.md section 4).
    #[test]
    fn a_filtered_iterator_promises_nothing_it_has_not_counted() {
        let m = rich();
        let f = Filtered::masked(&m, &NO_V2, &ALL_E).expect("masks fit");
        assert_eq!(f.edges().size_hint(), (0, Some(6)));
        assert_eq!(f.vertices().size_hint(), (0, Some(5)));
        assert_eq!(f.out_edges(v(0)).size_hint(), (0, Some(2)));
        // ... but the memoised counts are exact, because they were counted.
        assert_eq!(f.num_edges(), 2);
        assert_eq!(f.edges().count(), 2);
    }
}
