//! The property-map traits a kernel bounds on.

use crate::ids::{EdgeTag, Id, IdTag, VertexTag};
use std::ops::Deref;

/// A by-value result presented as a reference, so that storage-free maps
/// ([`IndexProp`](super::IndexProp), [`Unity`](super::dense::Unity)) can
/// implement the same trait as slice-backed ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Owned<T>(pub T);

impl<T> Deref for Owned<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

/// Reading a property map.
///
/// Access is **by reference**, not by value. `fn get(&self, k) -> Self::Value`
/// costs a heap allocation, a `memcpy` and two deallocations per element for
/// `String` and the seven vector members -- nine of the fifteen -- which is
/// exactly the inner loop of `graph_properties_copy.hh:62`, where the C++
/// `pm[k] = src_ref` reuses the target's existing capacity and allocates
/// nothing in steady state.
pub trait ReadProp<K: IdTag> {
    /// The stored type.
    type Value;
    /// How a read is handed back. `&'s Value` for slice-backed maps, an
    /// [`Owned`] carrier for computed ones.
    type Ref<'s>: Deref<Target = Self::Value>
    where
        Self: 's;

    /// `is_unity_map_v` (`graph_properties.hh:729`), as a const that kernels
    /// branch on. The untaken branch folds away entirely.
    const IS_UNITY: bool = false;
    /// `is_constant_map_v` (`graph_properties.hh:653`).
    const IS_CONSTANT: bool = false;

    /// Read. Panics if `k` is outside this map's bound; a map handed to a
    /// kernel has been sized by [`DenseProp::sized_for`](super::DenseProp::sized_for),
    /// so that cannot happen on the kernel path.
    fn get_ref(&self, k: Id<K>) -> Self::Ref<'_>;

    /// Read a copy.
    #[inline]
    fn get(&self, k: Id<K>) -> Self::Value
    where
        Self::Value: Clone,
    {
        self.get_ref(k).clone()
    }
}

/// Writing a property map.
///
/// Not implemented for [`Unity`](super::dense::Unity) or
/// [`Constant`](super::dense::Constant). `put(UnityPropertyMap, k, v) {}`
/// (`graph_properties.hh:714`) is a silent no-op that satisfies
/// `writable_property_map_tag` and discards every write, at any of the 74 call
/// sites that pass a unity map. Here the same call is a trait-bound error
/// naming `WriteProp`.
pub trait WriteProp<K: IdTag>: ReadProp<K> {
    /// Store a value.
    fn put(&mut self, k: Id<K>, v: Self::Value);
}

/// A property map whose slots are addressable.
pub trait LvalueProp<K: IdTag>: WriteProp<K> {
    /// Borrow a slot mutably, for in-place update without a round trip.
    fn at_mut(&mut self, k: Id<K>) -> &mut Self::Value;
}

/// Convenience alias: a readable vertex property map.
pub trait ReadVertexProp: ReadProp<VertexTag> {}
impl<P: ReadProp<VertexTag>> ReadVertexProp for P {}

/// Convenience alias: a readable edge property map.
pub trait ReadEdgeProp: ReadProp<EdgeTag> {}
impl<P: ReadProp<EdgeTag>> ReadEdgeProp for P {}
