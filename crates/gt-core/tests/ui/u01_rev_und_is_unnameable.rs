//! Defect #17 (`graph_filtering.hh:116-121`): the undirected-and-reversed
//! corner of the six-view product is removed by a `hana::filter` over a
//! *value-level* tuple, which constrains only the generated dispatch table --
//! a caller naming `graph_view_t<false, true, false>` by hand is unaffected.
//!
//! Here the corner is not filtered out of a list, it is ill-formed:
//! `Rev<G>` requires `G: HasDir<Dir = Directed>` and `Und<_>` is `Undirected`.
//!
//! ```text
//! error[E0271]: type mismatch resolving `<Und<&AdjList> as HasDir>::Dir == Directed`
//! ```

use gt_core::adj::AdjList;
use gt_core::view::{Rev, Und};

fn reversed_undirected(_: Rev<Und<&AdjList>>) {}

fn main() {}

// And the *duplicates* `hana::to<hana::set_tag>` (`graph_filtering.hh:129`)
// exists to remove: `Und<Und<_>>` is rejected by the same bound, so the tower
// above the six views does not exist either.
//
// ```text
// error[E0271]: type mismatch resolving `<Und<&AdjList> as HasDir>::Dir == Directed`
// ```
fn undirected_twice(_: Und<Und<&AdjList>>) {}
