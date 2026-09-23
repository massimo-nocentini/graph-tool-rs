//! The graph trait layer.
//!
//! ## Why there are no GATs here ([DESIGN](crate::design) D1)
//!
//! The obvious shape is `trait Incidence { type OutEdges<'a> where Self: 'a;
//! fn out_edges(&self, v) -> Self::OutEdges<'_> }`. It does not work, and the
//! three ways it fails are each fatal to a different part of the port:
//!
//! * `&dyn Incidence` is `error[E0038]: the trait is not dyn compatible
//!   because it contains generic associated type OutEdges`. graph-tool's
//!   entire Python boundary is runtime dispatch over graph kinds; it cannot be
//!   built on such a trait.
//! * `where for<'a> G::OutEdges<'a>: Send` -- the bound rayon needs -- is
//!   accepted for an owned graph and rejected for `Und(&g)` with
//!   `error[E0597]` and the compiler's own note that the GAT's implicit
//!   `where Self: 'a` "implies a `'static` lifetime". So no borrowing view
//!   can ever reach a parallel loop.
//! * a refinement such as `trait SizedIncidence: Incidence where for<'a>
//!   Self::OutEdges<'a>: ExactSizeIterator` is well-formed but unimplementable
//!   for any borrowing wrapper (`error[E0477]`), so there is no answer at all
//!   to the filtered-view question.
//!
//! The shape below puts the lifetime on the *implementing type* instead:
//! [`GraphRef`] is implemented for `&'g AdjList`, for `Und<&'g AdjList>` and
//! so on, so `'g` is an impl parameter and a plain associated type suffices.
//! Every method takes `self` by value because a view is `Copy` and one word
//! wide.
//!
//! ## Why `ExactSizeIterator` is not required
//!
//! A per-edge-predicate filtered view cannot supply it without counting. The
//! bound is therefore dropped from the trait and offered as the separate
//! [`ExactIncidence`] refinement, which unfiltered graphs implement and
//! filtered ones do not.

use crate::adj::{EdgeRef, Incident};
use crate::bound::{EdgeBound, VertexBound};
use crate::dir::{Directed, HasDir};
use crate::ids::{EdgeId, GraphId, VertexId};

/// Counts, bounds and identity.
pub trait GraphBase: Copy {
    /// Identity of the underlying storage. Views forward it unchanged, so a
    /// property map sized for `g` is accepted by `Und(g)`.
    fn graph_id(self) -> GraphId;

    /// The honest number of vertices this view exposes.
    ///
    /// A filtered view returns the *filtered* count, and
    /// `self.vertices().count() == self.num_vertices()` holds.
    /// `filt_graph` returns the unfiltered count (`graph_filtered.hh:316`)
    /// and the header admits at `:301-312` that the identity is thereby lost.
    fn num_vertices(self) -> usize;

    /// The honest number of edges this view exposes, counted from
    /// [`EdgeList::edges`] semantics -- each edge once, never by summing
    /// degrees.
    fn num_edges(self) -> usize;

    /// Allocation bound of the vertex index space, always of the *unfiltered*
    /// storage. This, never [`num_vertices`](Self::num_vertices), is what
    /// sizes a property map.
    fn vertex_bound(self) -> VertexBound;

    /// Allocation bound of the edge index space, always of the *unfiltered*
    /// storage.
    fn edge_bound(self) -> EdgeBound;
}

/// Incidence.
pub trait GraphRef: GraphBase + HasDir {
    /// Out-edges of a vertex.
    type Out: Iterator<Item = Incident> + Clone;
    /// Every incident edge of a vertex, whatever the orientation.
    type All: Iterator<Item = Incident> + Clone;

    /// Out-edges of `v`, **anchored at `v`**: every yielded
    /// [`Incident::other`] is the neighbour.
    fn out_edges(self, v: VertexId) -> Self::Out;

    /// Every incident edge of `v`, anchored at `v`.
    fn all_edges(self, v: VertexId) -> Self::All;

    /// Number of out-edges.
    fn out_degree(self, v: VertexId) -> usize;

    /// Total number of incident edges.
    fn degree(self, v: VertexId) -> usize;
}

/// In-edges. Directed views only.
///
/// `in_edges(v, undirected_adaptor)` returns `make_pair(iter_t(), iter_t())`
/// -- a default-constructed empty range (`graph_adaptor.hh:219-227`) -- so any
/// generic algorithm walking in-edges on an undirected view produces a wrong
/// answer rather than an error. There is no implementation of this trait for
/// [`Und`](crate::view::Und), so the same call is `error[E0599]`.
pub trait Bidirectional: GraphRef + HasDir<Dir = Directed> {
    /// In-edges of a vertex.
    type In: Iterator<Item = Incident> + Clone;

    /// In-edges of `v`, anchored at `v`.
    fn in_edges(self, v: VertexId) -> Self::In;

    /// Number of in-edges.
    fn in_degree(self, v: VertexId) -> usize;
}

/// The vertex set.
pub trait VertexList: GraphBase {
    /// Iterator over the vertices this view exposes.
    type Vertices: Iterator<Item = VertexId> + Clone;
    /// The vertices this view exposes.
    fn vertices(self) -> Self::Vertices;
}

/// The edge set.
///
/// Required by [`Filtered`](crate::view::Filtered) because degree-summation
/// cannot define `num_edges` on an undirected view: it double-counts, and
/// mixing an out-edge predicate with an edge predicate makes the count and the
/// iteration disagree.
pub trait EdgeList: GraphBase {
    /// Iterator over the edges this view exposes.
    type Edges: Iterator<Item = EdgeRef> + Clone;
    /// Each edge exactly once, in this view's orientation.
    fn edges(self) -> Self::Edges;
}

/// Resolving a stored [`EdgeId`] back to its endpoints.
pub trait Endpoints: GraphBase + HasDir {
    /// Endpoints in this view's orientation. O(1).
    fn endpoints(self, e: EdgeId) -> Option<(VertexId, VertexId)>;

    /// The endpoint that is not `v`.
    fn opposite(self, e: EdgeId, v: VertexId) -> Option<VertexId>;

    /// The first edge from `s` to `t` in this view's orientation.
    fn find_edge(self, s: VertexId, t: VertexId) -> Option<EdgeRef>;
}

/// A view whose incidence iterators know their own length.
///
/// The refinement that unfiltered graphs and their undirected/reversed views
/// implement and filtered views do not. Hot kernels that want `len()` or an
/// exact `collect` preallocation bound on this; generic ones do not.
pub trait ExactIncidence: GraphRef
where
    Self::Out: ExactSizeIterator,
    Self::All: ExactSizeIterator,
{
}

/// Something that owns graph storage and can lend a view of it.
///
/// The one place a GAT is appropriate: it is never used as a trait object and
/// never bound higher-ranked.
pub trait GraphOwner: HasDir {
    /// The view type this owner lends.
    type Ref<'a>: GraphRef
    where
        Self: 'a;
    /// Borrow a view.
    fn as_graph(&self) -> Self::Ref<'_>;
}

// ---------------------------------------------------------------------------
// `&AdjList` is the directed view.
// ---------------------------------------------------------------------------

use crate::adj::{AdjList, AllEdges, Edges, InEdges, Lookup, OutEdges, Vertices};

impl<H: Lookup> HasDir for AdjList<H> {
    type Dir = Directed;
}
impl<H: Lookup> HasDir for &AdjList<H> {
    type Dir = Directed;
}
impl<H: Lookup> HasDir for std::sync::Arc<AdjList<H>> {
    type Dir = Directed;
}

impl<H: Lookup> GraphBase for &AdjList<H> {
    #[inline]
    fn graph_id(self) -> GraphId {
        AdjList::graph_id(self)
    }
    #[inline]
    fn num_vertices(self) -> usize {
        AdjList::num_vertices(self)
    }
    #[inline]
    fn num_edges(self) -> usize {
        AdjList::num_edges(self)
    }
    #[inline]
    fn vertex_bound(self) -> VertexBound {
        AdjList::vertex_bound(self)
    }
    #[inline]
    fn edge_bound(self) -> EdgeBound {
        AdjList::edge_bound(self)
    }
}

impl<'g, H: Lookup> GraphRef for &'g AdjList<H> {
    type Out = OutEdges<'g>;
    type All = AllEdges<'g>;
    #[inline]
    fn out_edges(self, v: VertexId) -> Self::Out {
        AdjList::out_edges(self, v)
    }
    #[inline]
    fn all_edges(self, v: VertexId) -> Self::All {
        AdjList::all_edges(self, v)
    }
    #[inline]
    fn out_degree(self, v: VertexId) -> usize {
        AdjList::out_degree(self, v)
    }
    #[inline]
    fn degree(self, v: VertexId) -> usize {
        AdjList::degree(self, v)
    }
}

impl<'g, H: Lookup> Bidirectional for &'g AdjList<H> {
    type In = InEdges<'g>;
    #[inline]
    fn in_edges(self, v: VertexId) -> Self::In {
        AdjList::in_edges(self, v)
    }
    #[inline]
    fn in_degree(self, v: VertexId) -> usize {
        AdjList::in_degree(self, v)
    }
}

impl<H: Lookup> VertexList for &AdjList<H> {
    type Vertices = Vertices;
    #[inline]
    fn vertices(self) -> Vertices {
        AdjList::vertices(self)
    }
}

impl<'g, H: Lookup> EdgeList for &'g AdjList<H> {
    type Edges = Edges<'g>;
    #[inline]
    fn edges(self) -> Edges<'g> {
        AdjList::edges(self)
    }
}

impl<H: Lookup> Endpoints for &AdjList<H> {
    #[inline]
    fn endpoints(self, e: EdgeId) -> Option<(VertexId, VertexId)> {
        AdjList::endpoints(self, e)
    }
    #[inline]
    fn opposite(self, e: EdgeId, v: VertexId) -> Option<VertexId> {
        AdjList::endpoints(self, e).and_then(|(s, t)| {
            if v == s {
                Some(t)
            } else if v == t {
                Some(s)
            } else {
                None
            }
        })
    }
    #[inline]
    fn find_edge(self, s: VertexId, t: VertexId) -> Option<EdgeRef> {
        AdjList::find_edge(self, s, t)
    }
}

impl<H: Lookup> ExactIncidence for &AdjList<H> {}

impl<H: Lookup> GraphOwner for AdjList<H> {
    type Ref<'a>
        = &'a AdjList<H>
    where
        Self: 'a;
    #[inline]
    fn as_graph(&self) -> &AdjList<H> {
        self
    }
}

impl<H: Lookup> GraphOwner for std::sync::Arc<AdjList<H>> {
    type Ref<'a>
        = &'a AdjList<H>
    where
        Self: 'a;
    #[inline]
    fn as_graph(&self) -> &AdjList<H> {
        self
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! The view algebra, exercised against a hand-built reference graph.
    //!
    //! This lives *inside* the crate rather than in `tests/` for a reason that
    //! is itself a guarantee: the trait layer is deliberately not implementable
    //! downstream. `GraphBase::vertex_bound` must return a [`Bound`], whose
    //! constructor is `pub(crate)` (defect #7), and [`EdgeList::edges`] must
    //! yield [`EdgeRef`]s, whose constructor is `pub(crate)` too (defect #41).
    //! So only storage that gt-core itself blesses can present itself as a
    //! graph, and a test graph is storage.
    //!
    //! `AdjList`'s three incidence iterators are the same type alias
    //! (`OutEdges = InEdges = AllEdges = IncidentIter`), so wiring `Rev::Out`
    //! to the wrong half of a real `AdjList` is invisible to a type-level
    //! check. The mock below gives the three directions three *distinct*
    //! types, which makes the wiring checkable both as types and as values.

    use super::*;
    use crate::adj::Incident;
    use crate::bound::Bound;
    use crate::dir::Undirected;
    use crate::view::{Rev, Reverse, Und, Undirect};

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
            impl ExactSizeIterator for $name<'_> {}
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
    }

    #[derive(Clone)]
    struct MockEdges<'a>(std::slice::Iter<'a, EdgeRef>);
    impl Iterator for MockEdges<'_> {
        type Item = EdgeRef;
        fn next(&mut self) -> Option<EdgeRef> {
            self.0.next().copied()
        }
    }

    /// Storage with out- and in-halves held separately, so "which half did the
    /// view read" is answerable.
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

    /// `0 -e0-> 1`, `1 -e1-> 2`, `0 -e2-> 2`, `2 -e3-> 0`.
    fn mock() -> Mock {
        let spec = [(0usize, 1usize), (1, 2), (0, 2), (2, 0)];
        let mut m = Mock {
            id: GraphId::fresh(),
            out: vec![Vec::new(); 3],
            inc: vec![Vec::new(); 3],
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

    impl ExactIncidence for &Mock {}

    // The normalising constructors (D3) are traits, so new storage opts into
    // the algebra by naming its own entry points; `Rev<G>`/`Und<G>` then
    // supply idempotence, involution and absorption for free.
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

    fn pairs<I: Iterator<Item = Incident>>(it: I) -> Vec<(usize, usize)> {
        it.map(|i| (i.other.index(), i.edge.index())).collect()
    }
    fn ends<I: Iterator<Item = EdgeRef>>(it: I) -> Vec<(usize, usize, usize)> {
        it.map(|r| (r.id().index(), r.source().index(), r.target().index()))
            .collect()
    }

    // -----------------------------------------------------------------------

    #[test]
    fn the_base_view_reads_the_half_it_names() {
        let m = mock();
        let g = &m;
        assert_eq!(pairs(g.out_edges(v(0))), [(1, 0), (2, 2)]);
        assert_eq!(pairs(g.in_edges(v(0))), [(2, 3)]);
        assert_eq!(pairs(g.in_edges(v(2))), [(1, 1), (0, 2)]);
        assert_eq!(g.out_degree(v(0)), 2);
        assert_eq!(g.in_degree(v(0)), 1);
        assert_eq!(g.degree(v(0)), 3);
        assert_eq!(g.num_edges(), 4);
        assert_eq!(g.num_vertices(), 3);
        // `num_vertices` is the honest cardinality and `vertex_bound` the
        // allocation bound; `filt_graph` (graph_filtered.hh:314-318) returns
        // the latter from the former's name.
        assert_eq!(g.vertex_bound().len(), 3);
        assert_eq!(g.vertex_bound().graph(), g.graph_id());
        // `distance(vi, viend) == num_vertices(g)`, the identity
        // graph_filtered.hh:301-312 gives up.
        assert_eq!(g.vertices().count(), g.num_vertices());
        assert_eq!(g.edges().count(), g.num_edges());
    }

    /// `reversed_graph`'s `graph_traits` specialisation swaps the
    /// `out_edge_iterator` and `in_edge_iterator` typedefs
    /// (`graph_reverse.hh:78-80`): a reversed view's out-edges are the base
    /// graph's in-edges **only**. A view yielding `base-out ∪ base-in` would be
    /// an undirected view wearing a directed type, and [DESIGN](crate::design) D3 records
    /// that four of six views were wrong that way in one source design.
    #[test]
    fn rev_swaps_the_two_halves_and_does_not_union_them() {
        let m = mock();
        let r = (&m).reverse();

        assert_eq!(pairs(r.out_edges(v(0))), pairs((&m).in_edges(v(0))));
        assert_eq!(pairs(r.out_edges(v(0))), [(2, 3)]);
        assert_eq!(pairs(r.in_edges(v(0))), pairs((&m).out_edges(v(0))));
        assert_eq!(r.out_degree(v(0)), 1);
        assert_eq!(r.in_degree(v(0)), 2);
        // Not the union: that would be 3.
        assert_eq!(r.out_degree(v(0)) + r.in_degree(v(0)), r.degree(v(0)));

        // Type level, which the AdjList aliases cannot show.
        assert_eq!(
            std::any::TypeId::of::<<Rev<&Mock> as GraphRef>::Out>(),
            std::any::TypeId::of::<<&Mock as Bidirectional>::In>()
        );
        assert_eq!(
            std::any::TypeId::of::<<Rev<&Mock> as Bidirectional>::In>(),
            std::any::TypeId::of::<<&Mock as GraphRef>::Out>()
        );
    }

    #[test]
    fn rev_swaps_endpoints_without_mutating_a_descriptor() {
        let m = mock();
        let r = (&m).reverse();
        assert_eq!((&m).endpoints(e(0)), Some((v(0), v(1))));
        assert_eq!(r.endpoints(e(0)), Some((v(1), v(0))));
        assert_eq!(r.endpoints(e(9)), None);

        // `edge(s,t,g)` fails with a `{max,max,max}` descriptor
        // (graph_adjacency.hh:948); here it is `None`.
        assert!(r.find_edge(v(0), v(1)).is_none());
        let hit = r.find_edge(v(1), v(0)).expect("reversed lookup");
        assert_eq!(
            (hit.id().index(), hit.source().index(), hit.target().index()),
            (0, 1, 0)
        );
        // The base descriptor is untouched: `reverse_edge` (`:571`) mutates in
        // place, `EdgeRef::reversed` returns a value.
        let base = (&m).find_edge(v(0), v(1)).unwrap();
        assert_eq!(
            (base.source().index(), base.target().index()),
            (0, 1),
            "the stored orientation must survive a reversed lookup"
        );

        // `edges()` counts and iterates the same set, in swapped orientation.
        assert_eq!(
            ends(r.edges()),
            [(0, 1, 0), (1, 2, 1), (2, 2, 0), (3, 0, 2)]
        );
        assert_eq!(r.edges().count(), r.num_edges());
    }

    /// `out_edges(u, undirected_adaptor)` routes to `_all_edges_out`
    /// (`graph_adaptor.hh:199-207`), which `graph_adjacency.hh:1102-1108`
    /// implements by building an **out**-edge iterator over the whole block, so
    /// `make_out_edge::def` sets `src == u` for the in-half too: every yielded
    /// edge is anchored at the query vertex. That anchoring is the whole of
    /// defect #14, and here it is structural — `Incident::other` is the
    /// neighbour by construction.
    #[test]
    fn und_out_edges_is_the_whole_anchored_run() {
        let m = mock();
        let u = Und::wrap(&m);

        assert_eq!(pairs(u.out_edges(v(2))), pairs((&m).all_edges(v(2))));
        assert_eq!(pairs(u.out_edges(v(2))), [(0, 3), (1, 1), (0, 2)]);
        assert_eq!(u.out_degree(v(2)), 3);
        assert_eq!(u.degree(v(2)), u.out_degree(v(2)));

        // Anchored: `other` is never the query vertex, for every vertex.
        for w in (&m).vertices() {
            for i in u.out_edges(w) {
                let (s, t) = (&m).endpoints(i.edge).unwrap();
                assert!(
                    s == w || t == w,
                    "edge {:?} is not incident to {w:?}",
                    i.edge
                );
                assert_eq!(
                    i.other,
                    if s == w { t } else { s },
                    "out_edges({w:?}) must yield the neighbour, not the query vertex"
                );
            }
        }

        // `Und::Out` is the base graph's `All`, at the type level.
        assert_eq!(
            std::any::TypeId::of::<<Und<&Mock> as GraphRef>::Out>(),
            std::any::TypeId::of::<<&Mock as GraphRef>::All>()
        );
    }

    /// `edge(u, v, undirected_adaptor)` does `std::swap(res.first.s, res.first.t)`
    /// on the reverse hit (`graph_adaptor.hh:160`), so C++'s undirected lookup
    /// reports an orientation its own iteration never produces. Here lookup
    /// returns the **storage** orientation, unswapped: iteration answers
    /// "seen from where" with an anchored `Incident`, lookup answers "which
    /// edge" with a canonical `EdgeRef`, and the two cannot disagree.
    #[test]
    fn und_find_edge_reports_storage_orientation() {
        let m = mock();
        let u = Und::wrap(&m);

        let fwd = u.find_edge(v(1), v(2)).expect("forward hit");
        assert_eq!((fwd.source().index(), fwd.target().index()), (1, 2));

        let rev = u.find_edge(v(2), v(1)).expect("reverse hit");
        assert_eq!(
            (rev.id().index(), rev.source().index(), rev.target().index()),
            (1, 1, 2),
            "the reverse hit must not be swapped into an orientation \
             iteration never yields"
        );

        assert!(u.find_edge(v(1), v(1)).is_none());
        // Iteration agrees: edge 1 is seen from 2 with `other == 1`.
        assert!(u.out_edges(v(2)).any(|i| i.edge == e(1) && i.other == v(1)));
    }

    /// The normalising constructors (D3), checked as values rather than as the
    /// type table in [DESIGN](crate::design). `Rev<Und<_>>` and `Und<Und<_>>` are rejected by
    /// the struct bound itself — see `tests/ui/u01_rev_und_is_unnameable.rs`.
    #[test]
    fn the_constructors_normalise() {
        let m = mock();

        // involution
        let back: &Mock = (&m).reverse().reverse();
        assert_eq!(pairs(back.out_edges(v(0))), pairs((&m).out_edges(v(0))));

        // absorbing: reversing then undirecting drops the Rev entirely, so the
        // result reads the same run as undirecting the forward view.
        let a: Und<&Mock> = (&m).reverse().undirect();
        let b: Und<&Mock> = Und::wrap(&m);
        assert_eq!(pairs(a.out_edges(v(2))), pairs(b.out_edges(v(2))));

        // idempotent
        let c: Und<&Mock> = b.undirect();
        assert_eq!(pairs(c.out_edges(v(2))), pairs(b.out_edges(v(2))));

        // directedness survives
        fn is_directed<G: HasDir<Dir = Directed>>(_: &G) {}
        fn is_undirected<G: HasDir<Dir = Undirected>>(_: &G) {}
        is_directed(&(&m).reverse());
        is_directed(&back);
        is_undirected(&a);
    }

    /// Views forward `graph_id` unchanged, which is what lets a property map
    /// sized for `g` be accepted by `Und(g)` (defect #8's comparison is at
    /// `sized_for`, once, not per access).
    #[test]
    fn views_forward_identity_and_bounds() {
        let m = mock();
        let g = &m;
        let u = Und::wrap(g);
        let r = g.reverse();

        assert_eq!(u.graph_id(), g.graph_id());
        assert_eq!(r.graph_id(), g.graph_id());
        assert_eq!(u.vertex_bound(), g.vertex_bound());
        assert_eq!(r.edge_bound(), g.edge_bound());
        assert_eq!(u.num_vertices(), g.num_vertices());
        assert_eq!(u.num_edges(), g.num_edges());
        assert_eq!(r.num_edges(), g.num_edges());

        // Two graphs never share an identity, so a map from one is refused by
        // the other.
        let m2 = mock();
        assert_ne!(g.graph_id(), (&m2).graph_id());
        assert_ne!(g.vertex_bound(), (&m2).vertex_bound());
    }

    /// `opposite` is the one place a caller can pass a vertex that is not an
    /// endpoint. It answers `None`, where `adj_edge_descriptor`'s `==` (which
    /// compares `idx` only, `graph_adjacency.hh:196`) would have said nothing.
    #[test]
    fn opposite_rejects_a_non_endpoint() {
        let m = mock();
        let g = &m;
        assert_eq!(g.opposite(e(0), v(0)), Some(v(1)));
        assert_eq!(g.opposite(e(0), v(1)), Some(v(0)));
        assert_eq!(g.opposite(e(0), v(2)), None);
        assert_eq!(g.opposite(e(7), v(0)), None);
        // A reversed view changes the orientation, never the incidence.
        assert_eq!(g.reverse().opposite(e(0), v(0)), Some(v(1)));
    }
}
