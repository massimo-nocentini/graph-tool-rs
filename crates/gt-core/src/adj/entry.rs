//! Adjacency entries and the two edge descriptor types.
//!
//! ## Why there are two (DESIGN.md D2)
//!
//! graph-tool has one, `adj_edge_descriptor {Vertex s, t, idx}`
//! (`graph_adjacency.hh:186-206`), and it is overloaded:
//!
//! * `operator==` compares `idx` **only** (`:196`), so
//!   `edge_descriptor(1,2,7) == edge_descriptor(9,9,7)`;
//! * `reverse_edge` (`:571`) mutates `s`/`t` in place, so a descriptor's
//!   endpoints can disagree with the topology while `==` still says "same
//!   edge";
//! * the default value `{max,max,max}` (`:188-190`) is what `edge(s,t,g)`
//!   returns on failure (`:943`), and compares equal to every other failure.
//!
//! Splitting it gives:
//!
//! * [`Incident`] -- what *incidence iteration* yields. It is **anchored**:
//!   `other` is the neighbour, never the query vertex. This is exactly what
//!   `_all_edges_out` does (`graph_adjacency.hh:1102-1108` builds an
//!   `out_edge_iterator` over the whole list, so `make_out_edge::def` sets
//!   `src == u` for the in-half too) and it is what `graph_adaptor.hh:199-207`
//!   routes an undirected view to.
//! * [`EdgeRef`] -- what the *global* edge list yields, in canonical storage
//!   orientation.
//!
//! Neither implements `PartialEq`/`Hash`. An edge's identity is its
//! [`EdgeId`], obtained by `.id()` or `From`, so porting
//! `unordered_set<edge_descriptor>` (which hashes `e.idx`, `:1618`) to
//! `HashSet<EdgeRef>` does not compile rather than silently changing results.

use crate::ids::{EdgeId, VertexId};

/// One half-edge as stored: the other endpoint, and the edge's dense index.
///
/// The `pair<vertex_t, vertex_t>` of `graph_adjacency.hh:222`, at half the
/// size: 8 bytes with a 32-bit [`Raw`](crate::ids::Raw), so eight per cache
/// line where `adj_list<size_t>` gets four.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AdjEntry {
    /// The endpoint that is not the vertex whose block this is.
    pub other: VertexId,
    /// The edge's dense index.
    pub idx: EdgeId,
}

const _: () = assert!(size_of::<AdjEntry>() == 2 * size_of::<crate::ids::Raw>());

/// An edge as seen *from* a vertex.
///
/// Yielded by [`out_edges`](crate::graph::GraphRef::out_edges),
/// [`in_edges`](crate::graph::Bidirectional::in_edges) and
/// [`all_edges`](crate::graph::GraphRef::all_edges). `other` is always the
/// neighbour.
#[derive(Clone, Copy, Debug)]
pub struct Incident {
    /// The neighbour.
    pub other: VertexId,
    /// The edge's identity.
    pub edge: EdgeId,
}

impl Incident {
    /// The edge's identity.
    #[inline]
    pub const fn id(self) -> EdgeId {
        self.edge
    }
}

impl From<Incident> for EdgeId {
    #[inline]
    fn from(i: Incident) -> EdgeId {
        i.edge
    }
}

/// An edge with both endpoints, in canonical storage orientation.
///
/// Yielded by [`edges`](crate::graph::EdgeList::edges) and returned by
/// [`find_edge`](crate::adj::AdjList::find_edge). There is no null value: a
/// failed lookup is `None`.
#[derive(Clone, Copy, Debug)]
pub struct EdgeRef {
    id: EdgeId,
    src: VertexId,
    tgt: VertexId,
}

impl EdgeRef {
    /// Construct. Crate-private: an `EdgeRef` always comes from a graph.
    #[inline]
    pub(crate) const fn new(id: EdgeId, src: VertexId, tgt: VertexId) -> Self {
        EdgeRef { id, src, tgt }
    }

    /// The edge's identity.
    #[inline]
    pub const fn id(self) -> EdgeId {
        self.id
    }
    /// The stored source.
    #[inline]
    pub const fn source(self) -> VertexId {
        self.src
    }
    /// The stored target.
    #[inline]
    pub const fn target(self) -> VertexId {
        self.tgt
    }

    /// The same edge with its endpoints exchanged.
    ///
    /// Returns a value. `reverse_edge` (`graph_adjacency.hh:571`) mutates the
    /// descriptor in place, and the result is then accepted by `remove_edge`,
    /// which is how a caller-supplied orientation can drive a position lookup
    /// against the wrong half.
    #[inline]
    pub const fn reversed(self) -> Self {
        EdgeRef {
            id: self.id,
            src: self.tgt,
            tgt: self.src,
        }
    }

    /// The endpoint that is not `v`, or `None` if `v` is neither.
    #[inline]
    pub fn opposite(self, v: VertexId) -> Option<VertexId> {
        if v == self.src {
            Some(self.tgt)
        } else if v == self.tgt {
            Some(self.src)
        } else {
            None
        }
    }

    /// View this edge from `v`, if `v` is an endpoint.
    #[inline]
    pub fn anchored_at(self, v: VertexId) -> Option<Incident> {
        self.opposite(v).map(|other| Incident {
            other,
            edge: self.id,
        })
    }
}

impl From<EdgeRef> for EdgeId {
    #[inline]
    fn from(e: EdgeRef) -> EdgeId {
        e.id
    }
}
