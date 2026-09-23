//! Identifiers.
//!
//! ## The width decision ([DESIGN](crate::design) D5)
//!
//! graph-tool instantiates `adj_list<size_t>` and nothing else
//! (`src/graph/graph.hh:137`), so an adjacency entry costs 16 bytes and a
//! 64-byte cache line holds four. It then stores adjacency *positions* as
//! `pair<uint32_t,uint32_t>` (`graph_adjacency.hh:620`) under that `size_t`
//! vertex, so the two widths already disagree and nothing asserts it.
//!
//! Here the width is **one crate-wide type alias**, not a type parameter.
//! [`Raw`] is `u32` by default (8 entries per line, 2x graph-tool on the
//! dominant memory stream) and `u64` under the `wide-index` feature. Making it
//! an alias rather than a parameter removes an entire monomorphisation axis
//! from every signature in the port -- see the instantiation budget in
//! [DESIGN](crate::design) section 10.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroU64;

/// The raw integer width of every vertex and edge identifier.
#[cfg(not(feature = "wide-index"))]
pub type Raw = u32;
/// The raw integer width of every vertex and edge identifier.
#[cfg(feature = "wide-index")]
pub type Raw = u64;

/// Largest representable index. One value is reserved so that a future
/// niche-carrying id stays free.
pub const MAX_INDEX: usize = (Raw::MAX - 1) as usize;

/// Marker distinguishing one identifier space from another.
///
/// Replaces `adj_edge_descriptor`'s `Vertex s, t, idx` (`graph_adjacency.hh:211`),
/// where all three fields are the same type and are therefore mutually
/// substitutable at every call site.
pub trait IdTag: Copy + Eq + Ord + Hash + fmt::Debug + Send + Sync + 'static {
    /// Human-readable name, used in error messages.
    const NAME: &'static str;
}

/// Tag for the vertex index space.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct VertexTag;
impl IdTag for VertexTag {
    const NAME: &'static str = "vertex";
}

/// Tag for the edge index space.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct EdgeTag;
impl IdTag for EdgeTag {
    const NAME: &'static str = "edge";
}

/// A tagged index into one identifier space.
///
/// `PhantomData<fn() -> T>` keeps `Id` unconditionally `Copy + Send + Sync`
/// and covariant without bounding `T`.
#[repr(transparent)]
pub struct Id<T: IdTag>(Raw, PhantomData<fn() -> T>);

/// A vertex identifier.
pub type VertexId = Id<VertexTag>;
/// An edge identifier. This, and never an [`adj::EdgeRef`](crate::adj::EdgeRef),
/// is an edge's identity.
pub type EdgeId = Id<EdgeTag>;

const _: () = assert!(size_of::<VertexId>() == size_of::<Raw>());
const _: () = assert!(size_of::<EdgeId>() == size_of::<Raw>());

impl<T: IdTag> Id<T> {
    /// Construct from a raw index, or `None` if it exceeds [`MAX_INDEX`].
    #[inline]
    pub const fn new(i: usize) -> Option<Self> {
        if i <= MAX_INDEX {
            Some(Id(i as Raw, PhantomData))
        } else {
            None
        }
    }

    /// Construct from a raw index, panicking if out of range.
    ///
    /// For call sites that have already proved the bound (an iterator over
    /// `0..n`, a value read back out of the same structure).
    #[inline]
    pub const fn from_index(i: usize) -> Self {
        match Self::new(i) {
            Some(id) => id,
            None => panic!("index exceeds the identifier width"),
        }
    }

    /// The index, for use as a slice offset.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The raw representation.
    #[inline]
    pub const fn raw(self) -> Raw {
        self.0
    }
}

// Hand-written so that no `T: Clone`-style bound leaks into public signatures.
impl<T: IdTag> Clone for Id<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: IdTag> Copy for Id<T> {}
impl<T: IdTag> PartialEq for Id<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T: IdTag> Eq for Id<T> {}
impl<T: IdTag> PartialOrd for Id<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: IdTag> Ord for Id<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
impl<T: IdTag> Hash for Id<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}
impl<T: IdTag> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{}", T::NAME, self.0)
    }
}

/// Process-unique identity of one graph.
///
/// Carried by [`crate::bound::Bound`] and by [`crate::prop::DenseProp`] so that
/// sizing a map from graph A and indexing it with graph B's descriptors is
/// caught once, at kernel entry, by a single comparison. graph-tool has no
/// analogue: vertex descriptors are bare `size_t` and the confusion is the
/// mechanism behind the `copy_property` out-of-bounds write
/// (`graph_copy.cc:66-73` sizing from a *filtered* count then writing at
/// *unfiltered* indices).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GraphId(NonZeroU64);

impl GraphId {
    /// Mint a fresh, process-unique identity.
    pub fn fresh() -> Self {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let v = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        GraphId(NonZeroU64::new(v).expect("counter starts at 1"))
    }

    /// The raw value, for diagnostics only.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}
