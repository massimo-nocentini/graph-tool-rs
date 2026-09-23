//! Property maps and the closed value universe.
//!
//! ## The invariant this module exists for
//!
//! `gt_dispatch::pmap` (`dispatch.hh:171-177`) calls `a.get_unchecked()` with
//! **no size argument**, so `get_unchecked(size_t size = 0)`
//! (`fast_vector_property_map.hh:108`) performs `reserve(0)` -- a no-op, since
//! `reserve` only grows -- and hands the kernel a map whose `operator[]`
//! (`:218-221`) has no bounds check and whose backing vector may be shorter
//! than `num_vertices(g)`. The only thing standing between that and an
//! out-of-bounds write is nine lines of Python (`__init__.py:363-373`).
//!
//! Here, growth and view-creation are **one operation**:
//! [`DenseProp::sized_for`] takes a [`Bound`](crate::bound::Bound), resizes,
//! and returns a slice of exactly that length. There is no way to name a view
//! shorter than its bound, because there is no other constructor.
//!
//! ## The map a kernel receives is a trait, not a struct
//!
//! `vertex_index_map_t` and `edge_index_map_t` are `hana::append`ed to every
//! non-`writable_` property axis (`graph_properties.hh:166-231`), and those
//! axes name roughly three quarters of the 324 dispatch call sites. That
//! member is a *storage-free identity map*: it has no vector and no slice, so
//! a `struct`-shaped property map cannot represent it at all. Kernels
//! therefore bound on [`ReadProp`] / [`WriteProp`], which
//! [`DenseProp`], [`IndexProp`], [`Unity`] and [`Constant`] all implement.

pub mod convert;
pub mod dense;
pub mod dispatch;
pub mod dynamic;
pub mod index_map;
pub mod map;
pub mod value;

pub use convert::{AnyValue, ConvertFrom, FromAny, PyCell, ToAny, convertible};
pub use dense::{
    ConstI64, Constant, DenseProp, EdgeProp, One, PropSlice, PropSliceMut, Unity, VertexProp,
};
pub use dynamic::{
    DynProp, DynWrap, ReadOnlyAdaptor, ReadWriteAdaptor, Refusing, WriteOnlyAdaptor,
};
pub use index_map::IndexProp;
pub use map::{LvalueProp, Owned, ReadProp, WriteProp};
pub use value::{GilFree, LongDouble, PropValue, Scalar, ToF64, ValueKind, Zeroed};

#[cfg(feature = "python")]
pub use value::PyValue;
