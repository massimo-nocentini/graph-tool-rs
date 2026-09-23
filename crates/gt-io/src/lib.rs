//! # gt-io
//!
//! Graph and property-map serialisation.
//!
//! ## The ordering contract
//!
//! `AdjList` removes adjacency entries by swapping with the back of the half,
//! where graph-tool's non-`_keep_epos` path erases and shifts
//! (`graph_adjacency.hh:1257-1263`), preserving relative order. So the *order*
//! in which edges come out of [`EdgeList::edges`](gt_core::graph::EdgeList)
//! after removals differs from graph-tool's, although the set does not.
//!
//! Every writer here must therefore either (a) emit in the graph's own edge
//! order and document that the order is unspecified, or (b) sort by
//! [`EdgeId`](gt_core::ids::EdgeId), which is stable. `.gt` and GraphML choose
//! (b), so a round trip through this crate is byte-reproducible.
//!
//! ## `long double`
//!
//! `LongDouble` is an opaque 16-byte payload with no arithmetic
//! ([`gt_core::prop::LongDouble`]). That is what keeps a `.gt` file written by
//! graph-tool loadable and re-savable unchanged even though Rust has no
//! 80-bit float: the bytes round-trip, and the
//! [`Scalar`](gt_core::prop::Scalar) bound keeps the type out of every
//! arithmetic kernel at compile time.

#![forbid(unsafe_code)]
// SKELETON: see gt-core's lib.rs.
#![allow(dead_code, unused_variables)]
#![warn(missing_docs)]

pub mod csv;
pub mod dot;
pub mod error;
pub mod graphml;
pub mod gt;

pub use error::IoError;
