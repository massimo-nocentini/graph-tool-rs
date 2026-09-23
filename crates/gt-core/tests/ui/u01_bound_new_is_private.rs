//! Defect #7 (`graph_copy.cc:66-73`): `copy_property` sizes its target map from
//! `num_vertices(tgt)` -- the *filtered* count -- and then writes at
//! *unfiltered* indices `index_map[v]`. Nothing in the C++ type system can tell
//! a cardinality from an allocation bound, because both are `size_t`.
//!
//! Here a `Bound` is the allocation bound and can only be minted by the
//! unfiltered storage: `Bound::new` is `pub(crate)`. A downstream crate that
//! tries to forge one from a `usize` gets:
//!
//! ```text
//! error[E0624]: associated function `new` is private
//! ```

use gt_core::bound::{Bound, VertexBound};
use gt_core::ids::{GraphId, VertexTag};

fn main() {
    // A "filtered count" of 7, forged into an allocation bound.
    let forged: VertexBound = Bound::<VertexTag>::new(GraphId::fresh(), 7);
    let _ = forged;
}
