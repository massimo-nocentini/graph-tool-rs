//! Type-erased property maps.
//!
//! `DynamicPropertyMapWrap` (`graph_properties.hh:389-435`) type-erases a
//! property map behind a hand-rolled `shared_ptr<ValueConverter>` hierarchy,
//! turning an `N x M` product into `N + M`. The cost model is one virtual call
//! per access, and it must never appear inside a hot loop -- which the header
//! itself says at `:384-387`.
//!
//! The Rust form keeps the cost model exactly and changes two things: it is a
//! `dyn` object, so the cost is visible in the type, and the two `throw`s in
//! `get_dispatch`/`put_dispatch` (`:447`, `:457`) become `Result`, so they
//! cannot escape a parallel region as an exception.
//!
//! ## The three adaptors, and why there are three
//!
//! `ValueConverterImp::get` and `::put` (`graph_properties.hh:444-489`) are
//! tag-dispatched on whether the wrapped map's
//! `boost::property_traits<PropertyMap>::category` is convertible to
//! `readable_property_map_tag` / `writable_property_map_tag`. Two of the four
//! resulting overloads have a body and two `throw`. The same four cases exist
//! here as three constructible adaptors and one that is not a type at all:
//!
//! | boost category | there | here |
//! |---|---|---|
//! | read/write | both bodies run | [`ReadWriteAdaptor`] |
//! | readable only | `put` throws at `:489` | [`ReadOnlyAdaptor`], `dyn_put` is `Err(NotWritable)` |
//! | writable only | `get` throws at `:472` | [`WriteOnlyAdaptor`], `dyn_get` is `Err(NotReadable)` |
//! | neither | both throw | no adaptor exists |
//!
//! The throws are the defect. They are reachable from
//! `DynamicPropertyMapWrap::get`, which `graph_properties_copy.cc` calls from
//! inside an OpenMP loop whose `GILRelease` guard is already inverted
//! ([DESIGN](crate::design) §7); an exception crossing a `#pragma omp parallel` boundary is
//! `std::terminate`. A `Result` cannot do that.

use std::marker::PhantomData;

use crate::error::PropError;
use crate::ids::{Id, IdTag};
use crate::prop::convert::{ConvertFrom, FromAny, ToAny};
use crate::prop::map::{ReadProp, WriteProp};
use crate::prop::value::{PropValue, ValueKind};

/// The object-safe face of a property map.
pub trait DynProp<K: IdTag, V: 'static>: 'static {
    /// Read, converting into `V`.
    fn dyn_get(&self, k: Id<K>) -> Result<V, PropError>;
    /// Write, converting from `V`.
    fn dyn_put(&mut self, k: Id<K>, v: V) -> Result<(), PropError>;
    /// The member this map actually stores.
    fn underlying(&self) -> ValueKind;
}

/// Owning wrapper over a [`DynProp`].
pub struct DynWrap<K: IdTag, V: 'static> {
    inner: Box<dyn DynProp<K, V>>,
}

impl<K: IdTag, V: 'static> DynWrap<K, V> {
    /// Erase a concrete map.
    pub fn new(inner: Box<dyn DynProp<K, V>>) -> Self {
        DynWrap { inner }
    }

    /// Erase a readable and writable map.
    ///
    /// The `choose_converter` scan (`graph_properties.hh:495-510`) that picks
    /// an arm by `std::any_cast` and leaves `_converter == nullptr` — then
    /// `throw boost::bad_lexical_cast` at `:409` — has no counterpart: the arm
    /// is the static type of `map`.
    pub fn wrap<P>(map: P) -> Self
    where
        P: WriteProp<K> + 'static,
        P::Value: ToAny + FromAny,
        V: ToAny + FromAny,
    {
        DynWrap::new(Box::new(ReadWriteAdaptor::new(map)))
    }

    /// Erase a map that can only be read: [`Unity`](super::Unity),
    /// [`Constant`](super::Constant), [`IndexProp`](super::IndexProp).
    pub fn wrap_read_only<P>(map: P) -> Self
    where
        P: ReadProp<K> + 'static,
        P::Value: ToAny + FromAny,
        V: ToAny + FromAny,
    {
        DynWrap::new(Box::new(ReadOnlyAdaptor::new(map)))
    }

    /// Erase a map presented as write-only.
    pub fn wrap_write_only<P>(map: P) -> Self
    where
        P: WriteProp<K> + 'static,
        P::Value: ToAny + FromAny,
        V: ToAny + FromAny,
    {
        DynWrap::new(Box::new(WriteOnlyAdaptor::new(map)))
    }

    /// Read.
    #[inline]
    pub fn get(&self, k: Id<K>) -> Result<V, PropError> {
        self.inner.dyn_get(k)
    }
    /// Write.
    #[inline]
    pub fn put(&mut self, k: Id<K>, v: V) -> Result<(), PropError> {
        self.inner.dyn_put(k, v)
    }
    /// The member this map actually stores.
    ///
    /// `get_underlying_value_type` (`graph_properties.hh:425-428`) returns a
    /// `const std::type_info&` that every caller then compares against
    /// `typeid(...)` by hand — which is the shape that made the inverted
    /// `is_python` predicate at `graph_properties_copy.cc:35` writable in the
    /// first place. A [`ValueKind`] is matchable instead.
    #[inline]
    pub fn underlying(&self) -> ValueKind {
        self.inner.underlying()
    }
}

impl<K: IdTag, V: 'static> std::fmt::Debug for DynWrap<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynWrap")
            .field("key", &K::NAME)
            .field("underlying", &self.underlying())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// The adaptors
// ---------------------------------------------------------------------------

/// Erases a map that is both readable and writable.
///
/// `#[repr(transparent)]`: the adaptor is the map, and the only cost of the
/// erasure is the `Box` and the vtable that [`DynWrap`] adds.
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct ReadWriteAdaptor<P>(P);

/// Erases a map that can only be read.
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct ReadOnlyAdaptor<P>(P);

/// Erases a map presented as write-only.
///
/// [`WriteProp`] is a subtrait of [`ReadProp`], so the wrapped map *could* be
/// read; this adaptor is the deliberate refusal, and the point of having it is
/// that `graph_properties.hh:472`'s `throw` is reachable and this `Err` is not
/// an exception.
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct WriteOnlyAdaptor<P>(P);

macro_rules! adaptor_ctor {
    ($($t:ident),+ $(,)?) => {$(
        impl<P> $t<P> {
            /// Wrap a concrete map.
            #[inline]
            pub const fn new(map: P) -> Self {
                $t(map)
            }
            /// Give the map back, un-erased.
            #[inline]
            pub fn into_inner(self) -> P {
                self.0
            }
            /// Borrow the map.
            #[inline]
            pub const fn inner(&self) -> &P {
                &self.0
            }
        }
    )+};
}
adaptor_ctor!(ReadWriteAdaptor, ReadOnlyAdaptor, WriteOnlyAdaptor);

/// `convert<Value>(boost::get(pmap, k))` (`graph_properties.hh:464`).
#[inline]
fn read_converting<K, V, P>(map: &P, k: Id<K>) -> Result<V, PropError>
where
    K: IdTag,
    V: FromAny,
    P: ReadProp<K>,
    P::Value: ToAny,
{
    // Bound to `&P::Value`, never moved: `graph_properties_copy.hh:62`'s inner
    // loop is what the by-reference `ReadProp::Ref` exists for (`gt_core::design` D6),
    // and a `get(k) -> Value` here would allocate for nine of the fifteen
    // members before the conversion even starts.
    let slot = map.get_ref(k);
    <V as ConvertFrom<P::Value>>::convert_from(&slot)
}

/// `boost::put(pmap, k, convert<val_t>(val))` (`graph_properties.hh:457`,
/// `:481`). The conversion runs *before* the write, so a failed conversion
/// leaves the slot untouched — which the C++ also achieves, by throwing out of
/// the argument evaluation.
#[inline]
fn write_converting<K, V, P>(map: &mut P, k: Id<K>, v: V) -> Result<(), PropError>
where
    K: IdTag,
    V: ToAny,
    P: WriteProp<K>,
    P::Value: FromAny,
{
    let stored = <P::Value as ConvertFrom<V>>::convert_from(&v)?;
    map.put(k, stored);
    Ok(())
}

impl<K, V, P> DynProp<K, V> for ReadWriteAdaptor<P>
where
    K: IdTag,
    V: ToAny + FromAny,
    P: WriteProp<K> + 'static,
    P::Value: ToAny + FromAny,
{
    #[inline]
    fn dyn_get(&self, k: Id<K>) -> Result<V, PropError> {
        read_converting(&self.0, k)
    }
    #[inline]
    fn dyn_put(&mut self, k: Id<K>, v: V) -> Result<(), PropError> {
        write_converting(&mut self.0, k, v)
    }
    #[inline]
    fn underlying(&self) -> ValueKind {
        <P::Value as PropValue>::KIND
    }
}

impl<K, V, P> DynProp<K, V> for ReadOnlyAdaptor<P>
where
    K: IdTag,
    V: ToAny + FromAny,
    P: ReadProp<K> + 'static,
    P::Value: ToAny + FromAny,
{
    #[inline]
    fn dyn_get(&self, k: Id<K>) -> Result<V, PropError> {
        read_converting(&self.0, k)
    }
    /// `throw ValueException("Property map is not writable.")`
    /// (`graph_properties.hh:489`), as a value.
    #[inline]
    fn dyn_put(&mut self, _k: Id<K>, _v: V) -> Result<(), PropError> {
        Err(PropError::NotWritable)
    }
    #[inline]
    fn underlying(&self) -> ValueKind {
        <P::Value as PropValue>::KIND
    }
}

impl<K, V, P> DynProp<K, V> for WriteOnlyAdaptor<P>
where
    K: IdTag,
    V: ToAny + FromAny,
    P: WriteProp<K> + 'static,
    P::Value: ToAny + FromAny,
{
    /// `throw graph_tool::ValueException("Property map is not readable.")`
    /// (`graph_properties.hh:472`), as a value.
    #[inline]
    fn dyn_get(&self, _k: Id<K>) -> Result<V, PropError> {
        Err(PropError::NotReadable)
    }
    #[inline]
    fn dyn_put(&mut self, k: Id<K>, v: V) -> Result<(), PropError> {
        write_converting(&mut self.0, k, v)
    }
    #[inline]
    fn underlying(&self) -> ValueKind {
        <P::Value as PropValue>::KIND
    }
}

// ---------------------------------------------------------------------------
// A map that refuses both directions
// ---------------------------------------------------------------------------

/// The fourth boost category: neither readable nor writable.
///
/// `DynamicPropertyMapWrap` reaches it whenever the wrapped map's category is
/// convertible to neither tag, and then *both* `get` and `put` throw. Nothing
/// in the value universe has that category, so this carries no map at all; it
/// exists so a caller that must produce a `DynWrap` for a member it cannot
/// service has something to produce that is not a panic.
pub struct Refusing<V>(PhantomData<fn() -> V>);

impl<V> Refusing<V> {
    /// The map that refuses.
    pub const NEW: Self = Refusing(PhantomData);
}

impl<V> Default for Refusing<V> {
    #[inline]
    fn default() -> Self {
        Refusing::NEW
    }
}

impl<V> std::fmt::Debug for Refusing<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Refusing")
    }
}

impl<K: IdTag, V: PropValue> DynProp<K, V> for Refusing<V> {
    #[inline]
    fn dyn_get(&self, _k: Id<K>) -> Result<V, PropError> {
        Err(PropError::NotReadable)
    }
    #[inline]
    fn dyn_put(&mut self, _k: Id<K>, _v: V) -> Result<(), PropError> {
        Err(PropError::NotWritable)
    }
    #[inline]
    fn underlying(&self) -> ValueKind {
        <V as PropValue>::KIND
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{GraphId, VertexId, VertexTag};
    use crate::prop::dense::{Constant, DenseProp, Unity};

    fn v(i: usize) -> VertexId {
        VertexId::new(i).expect("small index")
    }

    /// The erasure is one `Box` and one vtable; the adaptor itself is free.
    #[test]
    fn adaptors_are_transparent() {
        assert_eq!(
            size_of::<ReadWriteAdaptor<DenseProp<i64, VertexTag>>>(),
            size_of::<DenseProp<i64, VertexTag>>()
        );
        assert_eq!(size_of::<ReadOnlyAdaptor<Unity<i64, VertexTag>>>(), 0);
        assert_eq!(size_of::<Refusing<i64>>(), 0);
    }

    /// `Unity` implements `ReadProp` only, so it can be erased read-only and
    /// the write is refused rather than silently discarded
    /// (`graph_properties.hh:714`, [DESIGN](crate::design) §5).
    #[test]
    fn unity_erases_read_only() {
        let mut w: DynWrap<VertexTag, String> =
            DynWrap::wrap_read_only(Unity::<i64, VertexTag>::NEW);
        assert_eq!(w.underlying(), ValueKind::I64);
        assert_eq!(w.get(v(3)).unwrap(), "1");
        assert_eq!(w.put(v(3), "5".to_owned()), Err(PropError::NotWritable));
    }

    #[test]
    fn constant_erases_read_only() {
        let w: DynWrap<VertexTag, f64> =
            DynWrap::wrap_read_only(Constant::<i64, VertexTag>::new(7));
        assert_eq!(w.get(v(0)).unwrap(), 7.0);
    }

    #[test]
    fn refusing_refuses_both_directions() {
        let mut w: DynWrap<VertexTag, i64> = DynWrap::new(Box::new(Refusing::<i64>::NEW));
        assert_eq!(w.get(v(0)), Err(PropError::NotReadable));
        assert_eq!(w.put(v(0), 1), Err(PropError::NotWritable));
        assert_eq!(w.underlying(), ValueKind::I64);
    }

    /// The `Debug` impl exists so a `DynWrap` in a test failure names the
    /// member it stores rather than printing `<opaque>`.
    #[test]
    fn debug_names_the_underlying_member() {
        let g = GraphId::fresh();
        let w: DynWrap<VertexTag, i64> =
            DynWrap::wrap(DenseProp::<String, VertexTag>::from_vec(g, vec![]));
        let s = format!("{w:?}");
        assert!(s.contains("vertex"), "{s}");
        assert!(s.contains("Str"), "{s}");
    }
}
