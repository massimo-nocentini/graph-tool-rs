//! Defect #43 (`graph_properties.hh:711-716`): `put(UnityPropertyMap, k, v) {}`
//! is an empty body that satisfies `writable_property_map_tag`, so every one of
//! the 74 call sites that passes a unity map can be handed a write and will
//! discard it in silence.
//!
//! `Unity` implements [`ReadProp`] and nothing else:
//!
//! ```text
//! error[E0599]: no method named `put` found for struct `Unity` in the current scope
//! ```

use gt_core::ids::{Id, VertexTag};
use gt_core::prop::{Unity, WriteProp};

fn main() {
    let mut w: Unity<f64, VertexTag> = Unity::NEW;
    w.put(Id::<VertexTag>::from_index(0), 2.0);
}
