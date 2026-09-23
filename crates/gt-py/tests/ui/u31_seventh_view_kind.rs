//! Defect #48's *eliminated* half, pinned.
//!
//! `gt_dispatch` resolves a view by walking the Hana product linearly and
//! comparing `typeid`s (`dispatch.hh:60-85`); when nothing matches it throws
//! `DispatchNotFound`, whose message is "This is a graph_tool bug. :-("
//! (`dispatch.hh:86-88`). Adding a seventh view to `get_graph_views`
//! therefore compiles, links, and fails at *run* time at whichever call site
//! first meets it.
//!
//! `AnyGraph::dispatch` is an exhaustive `match` over `ViewKind` with no
//! catch-all, so the same edit is a compile error. A trybuild fixture cannot
//! add a variant to another crate's enum, so the enum and the dispatcher's
//! arm list are reproduced here verbatim: what is being pinned is that *this
//! shape* -- every variant named, no `_` -- is what rejects the seventh, and
//! `tests/ui/u31_six_arms_are_exhaustive.rs` pins that the real `ViewKind`
//! still has exactly the six arms named below and is not `#[non_exhaustive]`.
//!
//! ```text
//! error[E0004]: non-exhaustive patterns: `ViewKind::Bipartite` not covered
//! ```

#[derive(Clone, Copy)]
enum ViewKind {
    Directed,
    Undirected,
    Reversed,
    DirectedFiltered,
    UndirectedFiltered,
    ReversedFiltered,
    // The seventh view. In C++ this is one more row in the cartesian product.
    Bipartite,
}

fn dispatch(kind: ViewKind) -> &'static str {
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
    let _ = dispatch(ViewKind::Directed);
}
