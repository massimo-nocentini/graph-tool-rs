//! `unchecked_vector_property_map::operator[]` is `const` and returns a
//! non-const `reference` out of a `shared_ptr` member
//! (`fast_vector_property_map.hh:219-222`), so a *const* C++ property map
//! writes to its store. The port's read-only view is a `&[T]`, and the write
//! traits are implemented for `PropSliceMut` only:
//!
//! ```text
//! error[E0599]: no method named `put` found for struct `PropSlice` in the current scope
//! ```

use gt_core::ids::{Id, VertexTag};
use gt_core::prop::{PropSlice, WriteProp};

fn write(mut w: PropSlice<'_, f64, VertexTag>) {
    w.put(Id::<VertexTag>::from_index(0), 2.0);
}

fn main() {}
