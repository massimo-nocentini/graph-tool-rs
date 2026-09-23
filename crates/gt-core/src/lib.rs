//! # gt-core
//!
//! The foundation of the graph-tool Rust port: identifiers, adjacency storage,
//! the view algebra, the closed property-value universe, property maps, and
//! deterministic parallel primitives.
//!
//! The architecture of record is `graph-tool-rs/docs/DESIGN.md`. Read it before
//! changing any public signature here; several of them encode a decision that
//! was contested, and the reason lives in that document rather than in the
//! code.
//!
//! ## Invariants this crate exists to enforce
//!
//! * A property map cannot be handed to a kernel without first being grown to
//!   the graph's *unfiltered* index bound ([`prop::DenseProp::sized_for`]).
//! * Incident edges are always **anchored**: `out_edges(v)` yields
//!   [`adj::Incident`] whose `other` is the neighbour, never the query vertex.
//! * `num_vertices()` is the honest cardinality; `vertex_bound()` is the
//!   allocation bound. They are different types, not the same `size_t`.
//! * The six graph views are the *only* six: `Und<Und<_>>`, `Rev<Rev<_>>` and
//!   `Rev<Und<_>>` are unnameable, not merely unused.
//! * No `unsafe` anywhere in this crate.

#![forbid(unsafe_code)]
// SKELETON: bodies are `todo!()`, so parameters are unread and private fields
// are unwritten. Both allows are removed by implementation unit U-FINAL.
#![allow(dead_code, unused_variables)]
#![warn(missing_docs)]

pub mod adj;
pub mod bound;
pub mod dir;
pub mod error;
pub mod graph;
pub mod ids;
pub mod par;
pub mod prop;
pub mod view;

pub use bound::{Bound, EdgeBound, VertexBound};
pub use dir::{Dir, Directed, HasDir, Undirected};
pub use error::{GraphError, PropError};
pub use graph::{Bidirectional, EdgeList, Endpoints, GraphBase, GraphOwner, GraphRef, VertexList};
pub use ids::{EdgeId, GraphId, Id, IdTag, Raw, VertexId};
