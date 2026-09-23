//! `adj_edge_descriptor` (`graph_adjacency.hh:211`) is `Vertex s, t, idx;` --
//! three fields of one type, so a vertex descriptor and an edge index are
//! mutually substitutable at every call site in the library, silently.
//!
//! `Id<T: IdTag>` is `#[repr(transparent)]` over `Raw` and costs the same, but
//! the phantom tag makes the two spaces disjoint:
//!
//! ```text
//! error[E0308]: mismatched types
//! ```

use gt_core::ids::{EdgeId, VertexId};

fn wants_a_vertex(_: VertexId) {}

fn main() {
    let e: EdgeId = EdgeId::from_index(3);
    wants_a_vertex(e);
}
