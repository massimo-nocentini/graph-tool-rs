//! Defect #12 (`graph_adaptor.hh:219-227`): `in_edges(v, undirected_adaptor)`
//! returns `make_pair(iter_t(), iter_t())` -- a default-constructed *empty*
//! range. Every generic algorithm that walks in-edges therefore produces a
//! wrong answer on an undirected view instead of failing.
//!
//! `Bidirectional` is a separate trait requiring `HasDir<Dir = Directed>` and
//! is deliberately not implemented for `Und<_>`:
//!
//! ```text
//! error[E0599]: no method named `in_edges` found for struct `Und<G>` in the current scope
//! ```

use gt_core::adj::AdjList;
use gt_core::graph::Bidirectional;
use gt_core::ids::VertexId;
use gt_core::view::Undirect;

fn main() {
    let g = AdjList::new();
    let u = (&g).undirect();
    let _ = u.in_edges(VertexId::from_index(0));
}
