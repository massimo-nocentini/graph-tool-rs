//! The reversed view.

use crate::adj::{EdgeRef, SwapEnds};
use crate::bound::{EdgeBound, VertexBound};
use crate::dir::{Directed, HasDir};
use crate::graph::{Bidirectional, EdgeList, Endpoints, GraphBase, GraphRef, VertexList};
use crate::ids::{EdgeId, GraphId, VertexId};

/// A directed graph traversed against its edges.
///
/// `out_edges` is the underlying graph's **in-edges only**. This is what
/// `graph_reverse.hh:78-80` says: `reversed_graph`'s `graph_traits`
/// specialisation swaps the `out_edge_iterator` and `in_edge_iterator`
/// typedefs. A design that yielded the union of both halves would be an
/// undirected view wearing a directed type.
///
/// The field is private; construct with [`Reverse::reverse`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Rev<G: HasDir<Dir = Directed>>(G);

impl<G: HasDir<Dir = Directed>> Rev<G> {
    #[inline]
    pub(crate) const fn wrap(g: G) -> Self {
        Rev(g)
    }

    /// The underlying forward view.
    #[inline]
    pub fn into_forward(self) -> G {
        self.0
    }
}

impl<G: HasDir<Dir = Directed>> HasDir for Rev<G> {
    type Dir = Directed;
}

/// Reversing a directed view. An involution: `g.reverse().reverse()` is `g`,
/// as a *type*, not merely as a value.
///
/// There is no implementation for [`Und`](super::Und), so reversing an
/// undirected view is `error[E0599]` rather than the silent no-op that
/// `graph_filtering.hh:77`'s `reversed && is_directed_v<directed_t>` guard
/// produces.
pub trait Reverse {
    /// The resulting reversed view.
    type Out: HasDir<Dir = Directed>;
    /// Take the reversed view.
    fn reverse(self) -> Self::Out;
}

impl<G: HasDir<Dir = Directed>> Reverse for Rev<G> {
    type Out = G;
    #[inline]
    fn reverse(self) -> G {
        self.0
    }
}

impl<H: crate::adj::Lookup> Reverse for &crate::adj::AdjList<H> {
    type Out = Rev<Self>;
    #[inline]
    fn reverse(self) -> Self::Out {
        Rev::wrap(self)
    }
}

impl<H: crate::adj::Lookup> Reverse for std::sync::Arc<crate::adj::AdjList<H>> {
    type Out = Rev<std::sync::Arc<crate::adj::AdjList<H>>>;
    #[inline]
    fn reverse(self) -> Self::Out {
        Rev::wrap(self)
    }
}

/// `Arc<AdjList>` traversed backwards, by moving the handle into the wrapper.
///
/// Replaces `std::reinterpret_pointer_cast<rg_t>(u)` (`graph_filtering.cc:78`).
pub fn own_reversed<H: crate::adj::Lookup>(
    g: std::sync::Arc<crate::adj::AdjList<H>>,
) -> Rev<std::sync::Arc<crate::adj::AdjList<H>>> {
    Rev::wrap(g)
}

impl<G: GraphBase + HasDir<Dir = Directed>> GraphBase for Rev<G> {
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

impl<G: Bidirectional> GraphRef for Rev<G> {
    type Out = G::In;
    type All = G::All;
    #[inline]
    fn out_edges(self, v: VertexId) -> G::In {
        self.0.in_edges(v)
    }
    #[inline]
    fn all_edges(self, v: VertexId) -> G::All {
        self.0.all_edges(v)
    }
    #[inline]
    fn out_degree(self, v: VertexId) -> usize {
        self.0.in_degree(v)
    }
    #[inline]
    fn degree(self, v: VertexId) -> usize {
        self.0.degree(v)
    }
}

impl<G: Bidirectional> Bidirectional for Rev<G> {
    type In = G::Out;
    #[inline]
    fn in_edges(self, v: VertexId) -> G::Out {
        self.0.out_edges(v)
    }
    #[inline]
    fn in_degree(self, v: VertexId) -> usize {
        self.0.out_degree(v)
    }
}

impl<G: VertexList + HasDir<Dir = Directed>> VertexList for Rev<G> {
    type Vertices = G::Vertices;
    #[inline]
    fn vertices(self) -> G::Vertices {
        self.0.vertices()
    }
}

impl<G: EdgeList + HasDir<Dir = Directed>> EdgeList for Rev<G> {
    type Edges = SwapEnds<G::Edges>;
    #[inline]
    fn edges(self) -> SwapEnds<G::Edges> {
        SwapEnds::new(self.0.edges())
    }
}

impl<G: Endpoints + HasDir<Dir = Directed>> Endpoints for Rev<G> {
    #[inline]
    fn endpoints(self, e: EdgeId) -> Option<(VertexId, VertexId)> {
        self.0.endpoints(e).map(|(s, t)| (t, s))
    }
    #[inline]
    fn opposite(self, e: EdgeId, v: VertexId) -> Option<VertexId> {
        self.0.opposite(e, v)
    }
    #[inline]
    fn find_edge(self, s: VertexId, t: VertexId) -> Option<EdgeRef> {
        self.0.find_edge(t, s).map(EdgeRef::reversed)
    }
}
