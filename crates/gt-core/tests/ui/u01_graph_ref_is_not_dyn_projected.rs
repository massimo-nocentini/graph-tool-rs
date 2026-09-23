//! D1, continued: supplying every projection does not rescue the trait object.
//! `GraphBase: Copy` and the by-value receivers put `Self: Sized` on the trait,
//! so no vtable can be built at all.
//!
//! This is the single strongest argument for D1's *non*-GAT shape: a GAT-based
//! incidence trait would make `DynGraph` -- and therefore graph-tool's entire
//! runtime-dispatched Python boundary -- impossible outright, with this same
//! `E0038` and no `DynGraph` available as the escape.
//!
//! ```text
//! error[E0038]: the trait `GraphRef` is not dyn compatible
//! ```

use gt_core::adj::{AllEdges, OutEdges};
use gt_core::dir::Directed;
use gt_core::graph::GraphRef;

fn takes_a_trait_object<'g>(
    _: &dyn GraphRef<Dir = Directed, Out = OutEdges<'g>, All = AllEdges<'g>>,
) {
}

fn main() {}
