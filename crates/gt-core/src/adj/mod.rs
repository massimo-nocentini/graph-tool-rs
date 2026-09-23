//! Adjacency storage: the `boost::adj_list` replacement.
//!
//! One contiguous `Vec<AdjEntry>` per vertex, out-entries in `[0, out_len)`
//! and in-entries in `[out_len, len)` -- the same layout as
//! `graph_adjacency.hh:222`, at half the width. Edge indices are dense, via a
//! free list, so edge property maps stay compact.
//!
//! Mutation goes through exactly two private primitives, `splice_in` and
//! `splice_out`, and every entry relocation is reported as a [`Moved`] value.
//! graph-tool instead declares every mutating free function a `friend`
//! (`graph_adjacency.hh:755-818`), so `_n_edges`, `_epos` and `_ehash` are
//! effectively public and each mutator re-derives the invariants by hand --
//! which is how `clear_vertex` comes to decrement `_n_edges` by two for one
//! removed edge (`:1404-1413`).

mod alloc;
mod block;
mod builder;
mod entry;
mod index;
mod iter;
mod list;

pub use alloc::EdgeIds;
pub use block::{Block, End, Moved};
pub use builder::ParBuilder;
pub use entry::{AdjEntry, EdgeRef, Incident};
pub use index::{EHash, EdgeSlot, EdgeSlots, Lookup, NoLookup};
pub use iter::{
    AllEdges, EdgeIdsOf, Edges, FilterEdges, FilterIncident, FilterVertices, InEdges, IncidentIter,
    OutEdges, SwapEnds, Vertices,
};
pub use list::AdjList;

/// The default graph: dense edge indices, O(1) endpoint lookup, no `(s,t)`
/// hash index. Corresponds to graph-tool's `_keep_epos = true`,
/// `_keep_ehash = false`.
pub type Graph = AdjList<NoLookup>;

/// A graph that additionally answers `edge(s, t)` in O(1).
pub type LookupGraph = AdjList<EHash>;
