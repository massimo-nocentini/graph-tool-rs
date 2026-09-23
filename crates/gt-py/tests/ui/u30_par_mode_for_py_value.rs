//! The lattice cannot be mis-edited into unsoundness.
//!
//! `ModeOf::Mode` is bounded `Mode + Allows<Self>`, and `Par: Allows<V>` holds
//! only for a `V` that can cross a thread. So the edit this fixture stands
//! for --
//!
//! ```ignore
//! impl ModeOf for PyValue { type Mode = Par; }
//! ```
//!
//! -- cannot be written, because that impl's associated type would have to
//! discharge exactly the obligation named below. A downstream crate cannot
//! write the impl itself (gt-py already has one, and `PropValue` is sealed in
//! gt-core besides), so the obligation is named directly: this is the same
//! bound the impl would face, in the only form a test crate can reach it.
//!
//! graph-tool has no counterpart. Its predicate is a `bool` computed by hand
//! inside the kernel body, five times over
//! (`graph_properties_copy.cc:35-42, :69-76, :104-111, :145-152`;
//! `graph_properties_copy.hh:36-40`), and there is nothing there for a
//! compiler to check -- which is how the expression came to be wrong in both
//! directions and stay that way in all five copies.

use gt_core::prop::{PropValue, PyValue};
use gt_py::gil::{Allows, Mode, Par};

/// The obligation `impl ModeOf for V { type Mode = M; }` incurs.
fn mode_of<V: PropValue, M: Mode + Allows<V>>() {}

fn main() {
    mode_of::<PyValue, Par>();
}
