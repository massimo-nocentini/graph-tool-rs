//! Directedness as a type.
//!
//! Three designs disagreed about how to spell this and each spelling was
//! load-bearing for a different result (DESIGN.md D4):
//!
//! * a bare `const DIRECTED: bool` on the graph trait cannot carry an
//!   associated *field set*, which is what lets an undirected delta buffer
//!   contain no in-field vectors at all rather than empty ones;
//! * a `Dir` associated type on the graph trait cannot be implemented for a
//!   non-`Copy` owner such as `Arc<AdjList>`, which is what makes
//!   `Und<Arc<AdjList>>` -- the safe replacement for the
//!   `reinterpret_pointer_cast` at `graph_filtering.cc:92` -- expressible.
//!
//! Both are therefore kept, and split: [`Dir`] is the type-level directedness
//! with its field set, and [`HasDir`] is a *separate* trait carrying it, so
//! owners can implement `HasDir` without implementing
//! [`GraphRef`](crate::graph::GraphRef).

use std::fmt;
use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

/// Type-level directedness.
pub trait Dir:
    sealed::Sealed + Copy + Clone + Default + fmt::Debug + Send + Sync + 'static
{
    /// Whether in-edges are distinct from out-edges.
    const DIRECTED: bool;
    /// Number of adjacency half-fields an SBM delta buffer needs: 4 when
    /// directed (`r`/`nr` x out/in), 2 when not. Ports
    /// `entries.hh:52`'s `if constexpr (directed)`.
    const N_FIELDS: usize;
    /// Short name for diagnostics.
    const NAME: &'static str;

    /// The delta buffer's field table, as a type: `[Vec<u32>; N_FIELDS]`.
    ///
    /// This is the "associated field set" the module doc above says a bare
    /// `const DIRECTED: bool` cannot carry. `N_FIELDS` alone only lets a
    /// buffer *allocate* the right number of half-fields; it still has to
    /// store them behind a `Vec<Vec<u32>>`, whose header is 24 bytes whatever
    /// the directedness. With the table as an associated array type, an
    /// undirected buffer **contains** no in-field vectors rather than owning
    /// two empty ones, and `size_of::<DeltaBuf<Undirected, _>>()` is strictly
    /// smaller than its directed counterpart.
    ///
    /// The bounds are what a buffer needs and nothing more: `Default` to build
    /// an empty table, `AsRef`/`AsMut` to index it as a slice (so the code is
    /// written once, not per directedness), and `Clone`/`Debug` so `DeltaBuf`
    /// can keep deriving both.
    ///
    /// It cannot be written as `[Vec<u32>; Self::N_FIELDS]` inline: an
    /// associated const is not usable in an array length on stable
    /// (`error: generic parameters may not be used in const operations`), so
    /// each implementor spells its own length.
    type Fields: AsRef<[Vec<u32>]>
        + AsMut<[Vec<u32>]>
        + Default
        + Clone
        + fmt::Debug
        + Send
        + Sync
        + 'static;
}

/// Directed view semantics.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Directed;
impl sealed::Sealed for Directed {}
impl Dir for Directed {
    const DIRECTED: bool = true;
    const N_FIELDS: usize = 4;
    const NAME: &'static str = "directed";
    type Fields = [Vec<u32>; 4];
}

/// Undirected view semantics.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Undirected;
impl sealed::Sealed for Undirected {}
impl Dir for Undirected {
    const DIRECTED: bool = false;
    const N_FIELDS: usize = 2;
    const NAME: &'static str = "undirected";
    type Fields = [Vec<u32>; 2];
}

/// Carries directedness. Implemented by graph views *and* by owners.
pub trait HasDir {
    /// This value's directedness.
    type Dir: Dir;
}

/// A *total* index into the adjacency half-fields of a delta buffer.
///
/// `Field<Undirected>` has exactly two inhabitants and `Field<Directed>` four,
/// so the "impossible" field is not a runtime `unreachable!()` guarded by an
/// undocumented invariant -- which is precisely the shape of the `_dummy`
/// fallthrough at `entries.hh:118` that lets two out-of-plane block pairs
/// silently accumulate into one entry.
pub struct Field<D: Dir>(u8, PhantomData<fn() -> D>);

impl<D: Dir> Field<D> {
    /// The out-half of the source group `r`.
    pub const R_OUT: Self = Field(0, PhantomData);
    /// The out-half of the target group `nr`.
    pub const NR_OUT: Self = Field(1, PhantomData);

    /// Position in a buffer's field table; always `< D::N_FIELDS`.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl Field<Directed> {
    /// The in-half of the source group `r`. Directed graphs only.
    pub const R_IN: Self = Field(2, PhantomData);
    /// The in-half of the target group `nr`. Directed graphs only.
    pub const NR_IN: Self = Field(3, PhantomData);
}

impl<D: Dir> Clone for Field<D> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<D: Dir> Copy for Field<D> {}
impl<D: Dir> PartialEq for Field<D> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<D: Dir> Eq for Field<D> {}
impl<D: Dir> fmt::Debug for Field<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Field<{}>({})", D::NAME, self.0)
    }
}
