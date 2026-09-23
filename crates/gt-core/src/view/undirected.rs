//! The undirected view.

use crate::adj::EdgeRef;
use crate::bound::{EdgeBound, VertexBound};
use crate::dir::{Directed, HasDir, Undirected};
use crate::graph::{EdgeList, Endpoints, GraphBase, GraphRef, VertexList};
use crate::ids::{EdgeId, GraphId, VertexId};

use super::reversed::Rev;

/// A directed graph seen as undirected.
///
/// `out_edges` is the *whole* incidence run, anchored at the query vertex --
/// which is exactly what `graph_adaptor.hh:199-207` does by routing to
/// `_all_edges_out`, and what `graph_adjacency.hh:1102-1108` implements by
/// building an `out_edge_iterator` (not an `all_edge_iterator`) over
/// `es.begin()..es.end()`, so `make_out_edge::def` sets `src == u` for the
/// in-half too.
///
/// One of the six source designs read that backwards, wired `out_edges` to a
/// canonical all-edges iterator, and thereby made `target(e)` return the query
/// vertex itself for half the edges. The anchored
/// [`Incident`](crate::adj::Incident) makes that
/// mistake unrepresentable.
///
/// The field is private; construct with [`Undirect::undirect`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Und<G: HasDir<Dir = Directed>>(G);

impl<G: HasDir<Dir = Directed>> Und<G> {
    #[inline]
    pub(crate) const fn wrap(g: G) -> Self {
        Und(g)
    }

    /// The underlying directed view.
    ///
    /// Note this is *not* an escape hatch to directed incidence on an
    /// undirected view by accident: it is explicit at the call site, whereas
    /// `Undirected<G>(pub G)` would let `u.0.in_edges(v)` compile silently.
    #[inline]
    pub fn into_directed(self) -> G {
        self.0
    }
}

impl<G: HasDir<Dir = Directed>> HasDir for Und<G> {
    type Dir = Undirected;
}

/// Viewing a directed graph as undirected.
///
/// Idempotent (`Und<G>::undirect() == Und<G>`) and absorbing
/// (`Rev<G>::undirect() == Und<G>`, since reversing an undirected graph is a
/// no-op). Together these close the algebra at the type level.
pub trait Undirect {
    /// The resulting undirected view.
    type Out: HasDir<Dir = Undirected>;
    /// Take the undirected view.
    fn undirect(self) -> Self::Out;
}

impl<G: HasDir<Dir = Directed>> Undirect for Und<G> {
    type Out = Und<G>;
    #[inline]
    fn undirect(self) -> Und<G> {
        self
    }
}

impl<G: HasDir<Dir = Directed>> Undirect for Rev<G> {
    type Out = Und<G>;
    #[inline]
    fn undirect(self) -> Und<G> {
        Und::wrap(self.into_forward())
    }
}

impl<G: HasDir<Dir = Directed>> GraphBase for Und<G>
where
    G: GraphBase,
{
    #[inline]
    fn graph_id(self) -> GraphId {
        self.0.graph_id()
    }
    #[inline]
    fn num_vertices(self) -> usize {
        self.0.num_vertices()
    }
    #[inline]
    fn num_edges(self) -> usize {
        self.0.num_edges()
    }
    #[inline]
    fn vertex_bound(self) -> VertexBound {
        self.0.vertex_bound()
    }
    #[inline]
    fn edge_bound(self) -> EdgeBound {
        self.0.edge_bound()
    }
}

impl<G: GraphRef + HasDir<Dir = Directed>> GraphRef for Und<G> {
    type Out = G::All;
    type All = G::All;
    #[inline]
    fn out_edges(self, v: VertexId) -> G::All {
        self.0.all_edges(v)
    }
    #[inline]
    fn all_edges(self, v: VertexId) -> G::All {
        self.0.all_edges(v)
    }
    #[inline]
    fn out_degree(self, v: VertexId) -> usize {
        self.0.degree(v)
    }
    #[inline]
    fn degree(self, v: VertexId) -> usize {
        self.0.degree(v)
    }
}

// NOTE: deliberately no `impl Bidirectional for Und<G>`.

impl<G: VertexList + HasDir<Dir = Directed>> VertexList for Und<G> {
    type Vertices = G::Vertices;
    #[inline]
    fn vertices(self) -> G::Vertices {
        self.0.vertices()
    }
}

impl<G: EdgeList + HasDir<Dir = Directed>> EdgeList for Und<G> {
    type Edges = G::Edges;
    /// Each edge exactly once, in storage orientation. `edges(undirected_adaptor)`
    /// forwards to `edges(original)` in the C++ for the same reason.
    #[inline]
    fn edges(self) -> G::Edges {
        self.0.edges()
    }
}

impl<G: Endpoints + HasDir<Dir = Directed>> Endpoints for Und<G> {
    #[inline]
    fn endpoints(self, e: EdgeId) -> Option<(VertexId, VertexId)> {
        self.0.endpoints(e)
    }
    #[inline]
    fn opposite(self, e: EdgeId, v: VertexId) -> Option<VertexId> {
        self.0.opposite(e, v)
    }
    /// Tries both orientations, and returns the hit **in storage orientation**.
    ///
    /// `edge(u, v, undirected_adaptor)` does `std::swap(res.first.s, res.first.t)`
    /// on the reverse hit (`graph_adaptor.hh:161`), which makes the C++'s
    /// undirected lookup disagree with its own iteration. Here iteration
    /// yields anchored incidences and lookup yields a canonical
    /// [`EdgeRef`], so the two cannot disagree: they answer different
    /// questions and say so in their types.
    fn find_edge(self, s: VertexId, t: VertexId) -> Option<EdgeRef> {
        self.0.find_edge(s, t).or_else(|| self.0.find_edge(t, s))
    }
}

/// `Arc<AdjList>` seen as undirected, by moving the handle into the wrapper.
///
/// This is the replacement for `std::reinterpret_pointer_cast<ug_t>(u)`
/// (`graph_filtering.cc:92`), which produces a `shared_ptr` to an
/// `undirected_adaptor` object that was never constructed. `Und<Arc<AdjList>>`
/// is `#[repr(transparent)]` over `Arc<AdjList>` -- the same layout the cast
/// produces -- but is built by a move.
pub fn own_undirected<H: crate::adj::Lookup>(
    g: std::sync::Arc<crate::adj::AdjList<H>>,
) -> Und<std::sync::Arc<crate::adj::AdjList<H>>> {
    Und::wrap(g)
}

impl<H: crate::adj::Lookup> Undirect for std::sync::Arc<crate::adj::AdjList<H>> {
    type Out = Und<std::sync::Arc<crate::adj::AdjList<H>>>;
    #[inline]
    fn undirect(self) -> Self::Out {
        Und::wrap(self)
    }
}

impl<H: crate::adj::Lookup> Undirect for &crate::adj::AdjList<H> {
    type Out = Und<Self>;
    #[inline]
    fn undirect(self) -> Self::Out {
        Und::wrap(self)
    }
}
