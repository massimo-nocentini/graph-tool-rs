//! Defect #12, propagated through the filter.
//!
//! `graph_filtered.hh:395-399` writes `degree(u, g)` as
//!
//! ```cpp
//! return in_degree(u, g) + out_degree(u, g);
//! ```
//!
//! and `graph_adaptor.hh:219-227` gives the undirected adaptor an `in_edges`
//! that returns a default-constructed, always-empty range. The sum therefore
//! compiles, runs, and answers half the degree.
//!
//! Here `Bidirectional for Filtered<G, F>` is bounded on `G: Bidirectional`,
//! and `Und<_>` is not `Bidirectional`, so the filtered undirected view has no
//! `in_edges` to return an empty range from. `Filtered::degree` counts
//! `all_edges` instead, which is why it needs no `Bidirectional` underneath.
//!
//! ```text
//! error[E0599]: the method `in_edges` exists for struct `Filtered<Und<&AdjList>, ...>`,
//!                but its trait bounds were not satisfied
//! ```

use gt_core::adj::AdjList;
use gt_core::graph::Bidirectional;
use gt_core::ids::VertexId;
use gt_core::view::{Filtered, MaskFilter, Undirect};

fn main() {
    let g = AdjList::new();
    let vmask = [1u8; 8];
    let emask = [1u8; 8];
    let f =
        Filtered::<_, MaskFilter<'_>>::masked((&g).undirect(), &vmask, &emask).expect("masks fit");
    let _ = f.in_edges(VertexId::from_index(0));
}
