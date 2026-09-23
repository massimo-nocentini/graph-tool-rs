//! D1: `GraphRef` is intentionally **not** a trait object. Every method takes
//! `self` by value and `GraphBase: Copy`, so incidence is monomorphised and a
//! view collapses to one pointer after inlining. The dyn-compatible face for
//! the Python boundary is `gt_py::DynGraph`, which iterates internally -- one
//! indirect call per *vertex*, not per edge.
//!
//! Naming `dyn GraphRef` without projecting its associated types stops at the
//! associated type itself, which is the first thing a caller reaching for a
//! trait object hits:
//!
//! ```text
//! error[E0191]: the value of the associated type `Dir` in `HasDir` must be specified
//! ```
//!
//! See `u01_graph_ref_is_not_dyn_projected.rs` for the same fact once every
//! projection is supplied.

use gt_core::graph::GraphRef;

fn takes_a_trait_object(_: &dyn GraphRef) {}

fn main() {}
