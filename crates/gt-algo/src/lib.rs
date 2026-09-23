//! # gt-algo
//!
//! Algorithms written **once**, generic over
//! [`GraphRef`](gt_core::graph::GraphRef), and therefore valid on all six
//! views without being written six times.
//!
//! Two bounds conventions are used throughout, and the distinction is the
//! reason the trait layer is shaped the way it is:
//!
//! * Most kernels bound on `G: GraphRef + VertexList`. These work on filtered
//!   views, whose incidence iterators cannot report an exact length.
//! * Kernels that genuinely need `len()` or an exact `collect` preallocation
//!   additionally bound on [`ExactIncidence`](gt_core::graph::ExactIncidence),
//!   which unfiltered graphs and their undirected/reversed views implement and
//!   filtered views do not.
//!
//! `graph_filtered.hh` has no analogue of that split: `filt_graph` simply
//! returns whatever the adaptor's `out_degree` says.
//!
//! ## In-edges
//!
//! An algorithm that needs predecessors bounds on
//! [`Bidirectional`](gt_core::graph::Bidirectional). Calling `in_edges` on an
//! undirected view is then `error[E0599]`, where
//! `graph_adaptor.hh:224-233` returns a default-constructed empty range and
//! the algorithm silently produces a wrong answer.

#![forbid(unsafe_code)]
// SKELETON: see gt-core's lib.rs.
#![allow(dead_code, unused_variables)]
#![warn(missing_docs)]

pub mod centrality;
pub mod components;
pub mod degree;
pub mod topology;
pub mod traversal;
