//! Allocation bounds, as a type distinct from cardinality.
//!
//! `filt_graph` deliberately forwards `num_vertices` to the *unfiltered* graph
//! (`graph_filtered.hh:314-318`) so that property storage stays correctly
//! sized, and the header admits at `:301-312` that this breaks
//! `distance(vi, viend) == num_vertices(g)`.
//!
//! Both numbers are needed and they are different things, so they are
//! different types here: [`crate::graph::GraphBase::num_vertices`] is the
//! honest cardinality, and [`VertexBound`] is the allocation bound. A `Bound`
//! is minted only by the unfiltered graph (`Bound::new` is crate-private) and
//! carries the graph's [`GraphId`], so sizing a map from a filtered count is a
//! type error and sizing it from *another graph's* bound is a caught runtime
//! error.

use crate::ids::{EdgeTag, GraphId, IdTag, VertexTag};
use std::marker::PhantomData;

/// Proof that one identifier space of one graph has exactly `n` slots.
pub struct Bound<K: IdTag> {
    graph: GraphId,
    n: usize,
    _k: PhantomData<fn() -> K>,
}

/// Allocation bound of a graph's vertex index space.
pub type VertexBound = Bound<VertexTag>;
/// Allocation bound of a graph's edge index space.
///
/// Note this is the edge index *range*, not the edge count: graph-tool's
/// `_get_any` uses `g.edge_index_range` (`__init__.py:369`) because the index
/// space is sparse after removals.
pub type EdgeBound = Bound<EdgeTag>;

impl<K: IdTag> Bound<K> {
    /// Mint a bound. Crate-private, so downstream code cannot forge one from a
    /// `usize` -- in particular, not from a filtered cardinality.
    #[inline]
    pub(crate) const fn new(graph: GraphId, n: usize) -> Self {
        Bound {
            graph,
            n,
            _k: PhantomData,
        }
    }

    /// Number of slots.
    #[inline]
    pub const fn len(self) -> usize {
        self.n
    }

    /// Whether the space is empty.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.n == 0
    }

    /// Identity of the graph this bound describes.
    #[inline]
    pub const fn graph(self) -> GraphId {
        self.graph
    }
}

impl<K: IdTag> Clone for Bound<K> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: IdTag> Copy for Bound<K> {}
impl<K: IdTag> PartialEq for Bound<K> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.graph == other.graph && self.n == other.n
    }
}
impl<K: IdTag> Eq for Bound<K> {}
impl<K: IdTag> std::fmt::Debug for Bound<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Bound<{}>(graph #{}, {})",
            K::NAME,
            self.graph.get(),
            self.n
        )
    }
}

#[cfg(test)]
mod tests {
    //! `Bound::new` is `pub(crate)`, so these live here rather than in
    //! `tests/`. That is the guarantee, not an inconvenience:
    //! `tests/ui/u01_bound_new_is_private.rs` pins the `E0624` a downstream
    //! crate gets for trying.

    use super::*;
    use crate::ids::GraphId;

    #[test]
    fn a_bound_is_a_length_plus_the_identity_that_minted_it() {
        let g = GraphId::fresh();
        let b: VertexBound = Bound::new(g, 7);
        assert_eq!(b.len(), 7);
        assert!(!b.is_empty());
        assert_eq!(b.graph(), g);

        let empty: VertexBound = Bound::new(g, 0);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    /// Defect #8. Equality compares the identity *and* the length, so a map
    /// sized for graph A cannot satisfy graph B's bound even when the two
    /// happen to have the same number of vertices — which is the case that
    /// makes the C++ failure silent rather than loud.
    #[test]
    fn two_graphs_of_equal_size_have_unequal_bounds() {
        let a = GraphId::fresh();
        let b = GraphId::fresh();
        assert_ne!(a, b);
        let ba: VertexBound = Bound::new(a, 10);
        let bb: VertexBound = Bound::new(b, 10);
        assert_eq!(ba.len(), bb.len());
        assert_ne!(ba, bb);
        assert_eq!(ba, Bound::new(a, 10));
        assert_ne!(ba, Bound::new(a, 11));
    }

    /// The vertex and edge spaces of *one* graph share an identity and are
    /// still distinct types, because `graph_copy.cc`'s confusion is not only
    /// between graphs.
    #[test]
    fn the_two_index_spaces_are_separate_types_over_one_identity() {
        let g = GraphId::fresh();
        let v: VertexBound = Bound::new(g, 4);
        let e: EdgeBound = Bound::new(g, 9);
        assert_eq!(v.graph(), e.graph());
        assert_ne!(v.len(), e.len());
        // `assert_eq!(v, e)` does not compile: different `K`.
    }

    #[test]
    fn debug_prints_both_numbers() {
        let g = GraphId::fresh();
        let v: VertexBound = Bound::new(g, 4);
        assert_eq!(
            format!("{v:?}"),
            format!("Bound<vertex>(graph #{}, 4)", g.get())
        );
        let e: EdgeBound = Bound::new(g, 4);
        assert_eq!(
            format!("{e:?}"),
            format!("Bound<edge>(graph #{}, 4)", g.get())
        );
    }

    /// A bound is passed by value into every `sized_for` call and must not
    /// cost more than the two words it carries.
    #[test]
    fn a_bound_is_copy_and_two_words() {
        let g = GraphId::fresh();
        let b: VertexBound = Bound::new(g, 3);
        let c = b;
        assert_eq!(b, c);
        assert_eq!(size_of::<VertexBound>(), 2 * size_of::<usize>());
        assert_eq!(size_of::<Option<VertexBound>>(), size_of::<VertexBound>());
    }
}
