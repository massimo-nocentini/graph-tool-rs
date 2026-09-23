//! The other constant map, same rule. `ConstantPropertyMap`
//! (`graph_properties.hh:670-696`) also derives `put_get_helper` and also
//! accepts writes that go nowhere.
//!
//! ```text
//! error[E0599]: no method named `put` found for struct `Constant` in the current scope
//! ```

use gt_core::ids::{Id, VertexTag};
use gt_core::prop::{Constant, WriteProp};

fn main() {
    let mut w: Constant<f64, VertexTag> = Constant::new(0.5);
    w.put(Id::<VertexTag>::from_index(0), 2.0);
}
