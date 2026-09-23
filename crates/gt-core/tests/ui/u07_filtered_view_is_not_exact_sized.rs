//! DESIGN.md section 4: `GraphRef` does **not** require
//! `ExactSizeIterator`, because a per-edge-predicate view cannot supply a
//! length without walking. `ExactIncidence` is the refinement that unfiltered
//! views implement and filtered ones do not.
//!
//! ```text
//! error[E0277]: the trait bound `Filtered<&AdjList, MaskFilter<'_>>: ExactIncidence` is not satisfied
//! ```

use gt_core::adj::AdjList;
use gt_core::graph::ExactIncidence;
use gt_core::view::{Filtered, MaskFilter};

fn needs_a_length<G: ExactIncidence>(_: G)
where
    G::Out: ExactSizeIterator,
    G::All: ExactSizeIterator,
{
}

fn main() {
    let g = AdjList::new();
    let vmask = [1u8; 8];
    let emask = [1u8; 8];
    let f: Filtered<&AdjList, MaskFilter<'_>> =
        Filtered::masked(&g, &vmask, &emask).expect("masks fit");
    needs_a_length(f);
}
