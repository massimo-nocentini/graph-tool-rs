//! `vertex_index_map_t` / `edge_index_map_t` are appended to every
//! *non-*`writable_` property axis (`graph_properties.hh:166-231`), and C++
//! keeps the two kinds of axis apart with a `has_unwritable` bool threaded
//! through `get_seq_type_names` (`:278`) -- a value, checked by hand, at code
//! generation time.
//!
//! Here the identity map simply has no `WriteProp` impl, so the distinction is
//! the trait bound and the check is the compiler's:
//!
//! ```text
//! error[E0599]: no method named `put` found for struct `IndexProp` in the current scope
//! ```

use gt_core::ids::{Id, VertexTag};
use gt_core::prop::{IndexProp, WriteProp};

fn write(mut w: IndexProp<VertexTag>) {
    w.put(Id::<VertexTag>::from_index(0), 7);
}

fn main() {}
