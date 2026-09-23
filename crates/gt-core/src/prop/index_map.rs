//! The identity property map.
//!
//! `vertex_index_map_t` and `edge_index_map_t` are appended to every
//! non-`writable_` property axis (`graph_properties.hh:166-231`), so they are
//! present in `vertex_properties`, `scalar_vertex_properties`,
//! `integer_vertex_properties` and their edge counterparts -- the axis names
//! that occur in roughly three quarters of the 324 dispatch call sites.
//!
//! They are *storage-free*: the map returns the descriptor's own index. No
//! vector-owning or slice-holding property-map type can represent them, which
//! is why the kernel-facing map is a trait
//! ([`ReadProp`]) rather than a struct.

use crate::bound::Bound;
use crate::ids::{Id, IdTag};

use super::map::{Owned, ReadProp};

/// Reads back the identifier's own index. Carries no storage.
#[derive(Clone, Copy, Debug)]
pub struct IndexProp<K: IdTag> {
    bound: Bound<K>,
}

impl<K: IdTag> IndexProp<K> {
    /// The identity map over an index space.
    ///
    /// Takes the bound so that the map still knows which graph it belongs to
    /// and how large the space is, for the same checks a
    /// [`DenseProp`](super::DenseProp) makes.
    #[inline]
    pub const fn new(bound: Bound<K>) -> Self {
        IndexProp { bound }
    }

    /// The index space this map covers.
    #[inline]
    pub const fn bound(&self) -> Bound<K> {
        self.bound
    }
}

impl<K: IdTag> ReadProp<K> for IndexProp<K> {
    type Value = i64;
    type Ref<'s>
        = Owned<i64>
    where
        Self: 's;
    #[inline(always)]
    fn get_ref(&self, k: Id<K>) -> Owned<i64> {
        Owned(k.index() as i64)
    }
}

// NOTE: deliberately no `WriteProp`. graph-tool's `writable_*` axes exist
// precisely because the index map cannot be written, and it distinguishes them
// with a `has_unwritable` flag threaded through `get_seq_type_names`
// (`graph_properties.hh:278`). Here the distinction is the trait bound.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{EdgeTag, GraphId, MAX_INDEX, VertexTag};

    /// The whole behaviour: `get(vertex_index, v) == v`, storage-free.
    #[test]
    fn the_identity_map_reads_back_the_index() {
        let g = GraphId::fresh();
        let p: IndexProp<VertexTag> = IndexProp::new(Bound::new(g, 16));
        for i in [0usize, 1, 2, 15, 4096, MAX_INDEX] {
            let k = Id::<VertexTag>::from_index(i);
            assert_eq!(*p.get_ref(k), i as i64);
            assert_eq!(p.get(k), k.index() as i64);
        }
    }

    /// It is an *index* map, not a bounds-checked map: `vertex_index_map_t`
    /// answers for any descriptor, and the bound it carries is there for the
    /// same sizing check a `DenseProp` makes, not to gate reads.
    #[test]
    fn a_read_outside_the_bound_is_still_the_index() {
        let g = GraphId::fresh();
        let p: IndexProp<VertexTag> = IndexProp::new(Bound::new(g, 2));
        assert_eq!(*p.get_ref(Id::<VertexTag>::from_index(99)), 99);
        assert_eq!(p.bound(), Bound::new(g, 2));
        assert_eq!(p.bound().graph(), g);
    }

    /// Storage-free is the reason the kernel-facing map is a trait: no
    /// vector-owning struct can represent this member of the axes at
    /// `graph_properties.hh:166-231`.
    #[test]
    fn the_index_map_carries_nothing_but_its_bound() {
        assert_eq!(
            size_of::<IndexProp<VertexTag>>(),
            size_of::<Bound<VertexTag>>()
        );
        assert_eq!(
            [
                <IndexProp<VertexTag> as ReadProp<VertexTag>>::IS_UNITY,
                <IndexProp<VertexTag> as ReadProp<VertexTag>>::IS_CONSTANT,
            ],
            [false; 2]
        );
    }

    /// The two spaces are separate maps over one identity.
    #[test]
    fn both_index_spaces_have_their_own_map() {
        let g = GraphId::fresh();
        let v: IndexProp<VertexTag> = IndexProp::new(Bound::new(g, 4));
        let e: IndexProp<EdgeTag> = IndexProp::new(Bound::new(g, 9));
        assert_eq!(*v.get_ref(Id::<VertexTag>::from_index(3)), 3);
        assert_eq!(*e.get_ref(Id::<EdgeTag>::from_index(3)), 3);
        // `v.get_ref(Id::<EdgeTag>::from_index(3))` does not compile.
    }
}
