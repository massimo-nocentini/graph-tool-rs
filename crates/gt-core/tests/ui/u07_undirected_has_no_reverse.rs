//! Defect #17's other half. `graph_filtering.hh:75-79` writes the
//! undirected-and-reversed corner out with
//!
//! ```cpp
//! typedef std::conditional_t<reversed && is_directed_v<directed_t>,
//!                            boost::reversed_graph<directed_t>,
//!                            directed_t> reversed_t;
//! ```
//!
//! so asking for a reversed undirected view *succeeds* and hands back the
//! unreversed one. The request is discarded, not refused.
//!
//! Here `Reverse` is implemented for `&AdjList`, `Arc<AdjList>` and `Rev<G>`
//! -- and for no undirected view -- so the call has no meaning to discard.
//!
//! ```text
//! error[E0599]: no method named `reverse` found for struct `Und` in the current scope
//! ```

use gt_core::adj::AdjList;
use gt_core::view::{Reverse, Undirect};

fn main() {
    let g = AdjList::new();
    let _ = (&g).undirect().reverse();
}
