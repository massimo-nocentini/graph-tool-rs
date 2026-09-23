//! Error types.
//!
//! Every fallible operation in this crate returns one of these. graph-tool
//! signals the same conditions by `throw ValueException` from six sites
//! reachable inside kernels (`graph_properties.hh:266, :319, :447, :457`,
//! `graph_copy.cc:99, :181`), which can escape an OpenMP region.

use crate::ids::{EdgeId, VertexId};
use crate::prop::ValueKind;

/// Structural failures of a graph operation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GraphError {
    /// A vertex descriptor does not name a live vertex.
    #[error("no such vertex: {0:?}")]
    NoSuchVertex(VertexId),
    /// An edge descriptor does not name a live edge.
    #[error("no such edge: {0:?}")]
    NoSuchEdge(EdgeId),
    /// The dense edge-index space is exhausted.
    #[error("edge index space exhausted (max {max})")]
    EdgeIdSpaceExhausted {
        /// The largest representable index.
        max: usize,
    },
    /// The dense vertex-index space is exhausted.
    #[error("vertex index space exhausted (max {max})")]
    VertexIdSpaceExhausted {
        /// The largest representable index.
        max: usize,
    },
    /// A derived index disagrees with the adjacency. Only reachable from
    /// [`crate::adj::AdjList::validate`].
    #[error("internal invariant violated: {0}")]
    Invariant(&'static str),
}

/// Failures at the property-map boundary.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PropError {
    /// A map sized for one graph was offered to a kernel operating on another.
    ///
    /// This is the class behind `copy_property`'s out-of-bounds write: the
    /// Python guard at `graph_tool/__init__.py:3200` compares *filtered*
    /// counts while `graph_copy.cc:72` writes at *unfiltered* indices.
    #[error("property map belongs to graph #{owner}, not graph #{expected}")]
    WrongGraph {
        /// Identity carried by the map.
        owner: u64,
        /// Identity carried by the bound.
        expected: u64,
    },
    /// A read-only view was requested of a map shorter than the graph's bound.
    #[error("property map holds {have} entries, the graph's index bound is {need}")]
    Undersized {
        /// Current length.
        have: usize,
        /// Required length.
        need: usize,
    },
    /// The dynamic map does not support reading.
    #[error("property map is not readable")]
    NotReadable,
    /// The dynamic map does not support writing.
    #[error("property map is not writable")]
    NotWritable,
    /// No conversion exists between these two members of the value universe.
    #[error("cannot convert a {from:?} property map to {to:?}")]
    NoConversion {
        /// Source member.
        from: ValueKind,
        /// Target member.
        to: ValueKind,
    },
    /// A filter mask is shorter than the index bound it was offered for.
    #[error("filter mask holds {have} entries, the index bound is {need}")]
    ShortMask {
        /// Mask length.
        have: usize,
        /// Required length.
        need: usize,
    },
}

/// The one error a dispatcher can raise after the boundary has been crossed.
///
/// graph-tool throws `DispatchNotFound`, whose message is literally
/// "This is a graph_tool bug. :-(" (`dispatch.hh:86-88`), so a legitimate user
/// type error and a genuine codegen hole are indistinguishable. Here only the
/// first survives, and it carries the accepted set.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{axis}: {offered:?} is not one of {accepted:?}")]
pub struct DispatchError {
    /// Which argument was rejected.
    pub axis: &'static str,
    /// What the caller supplied.
    pub offered: ValueKind,
    /// What this call site accepts.
    pub accepted: &'static [ValueKind],
}
