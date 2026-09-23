//! The other half of the seventh-view guarantee: the *real* `ViewKind` has
//! exactly these six variants, and is not `#[non_exhaustive]`.
//!
//! Without this, `u31_seventh_view_kind.rs` would pin a shape that the
//! dispatcher no longer has: a `#[non_exhaustive]` enum forces every
//! downstream match to carry a `_` arm, which is exactly the catch-all that
//! turns a seventh view back into a runtime `DispatchNotFound`. This file
//! compiles only while a six-arm, catch-all-free match over `ViewKind` is
//! legal from outside gt-py.

use gt_py::dispatch::ViewKind;

fn name(kind: ViewKind) -> &'static str {
    match kind {
        ViewKind::Directed => "&AdjList",
        ViewKind::Undirected => "Und<&AdjList>",
        ViewKind::Reversed => "Rev<&AdjList>",
        ViewKind::DirectedFiltered => "Filtered<&AdjList, MaskFilter>",
        ViewKind::UndirectedFiltered => "Filtered<Und<&AdjList>, MaskFilter>",
        ViewKind::ReversedFiltered => "Filtered<Rev<&AdjList>, MaskFilter>",
    }
}

fn main() {
    for k in ViewKind::ALL {
        assert!(!name(k).is_empty());
    }
}
