//! The closed 15-member value universe.
//!
//! ## What is on the base trait, and why it matters (DESIGN.md D7)
//!
//! Two of the source designs wrote
//! `trait PropValue: Sealed + Default + Send + Sync + 'static`. Both bounds
//! are wrong, and each breaks a different headline claim:
//!
//! * **`Default`** is unimplementable for the 15th member. pyo3 has no
//!   `impl Default for Py<T>` -- the default `python::object` is `Py_None`,
//!   which needs a live interpreter and an incref. So
//!   `data.resize_with(n, T::default)`, the body of the central sizing
//!   chokepoint, cannot exist for the Python type. Here growth takes a
//!   `FnMut() -> T` closure, which the PyO3 boundary fills by capturing its
//!   `Python<'py>` token; the context-free case is the [`Zeroed`] marker.
//! * **`Send + Sync`** is exactly what defeats the GIL guarantee, because
//!   rayon checks `Send`, not a positive marker such as `GilFree`. With them
//!   on the base trait, `store.as_mut_slice().par_iter_mut()` over a
//!   `python::object` map compiles with no `unsafe` and no diagnostic. They
//!   live on [`GilFree`] instead, and -- belt and braces -- [`PyValue`] is
//!   `!Send` *by construction*, so a kernel that forgets the marker still
//!   cannot reach rayon.
//!
//! `long double` and `vector<long double>` have no Rust equivalent (`f128` is
//! unstable). They are carried as [`LongDouble`], an opaque 16-byte payload
//! with **no arithmetic**: `.gt` files and `PropertyMap.value_type()` round-trip
//! unchanged, and the [`Scalar`] bound excludes the type from every arithmetic
//! kernel at compile time. That is what the subset-bound mechanism is for.

mod sealed {
    /// Seals the value universe. Private, so no downstream crate can name it
    /// and therefore none can add a 16th member -- in particular, none can
    /// re-introduce a `Send + Sync` Python handle under a different newtype.
    pub trait Sealed {}
}

/// The 15 members, as a closed enum.
///
/// Replaces the positionally-coupled pair `value_types`
/// (`graph_properties.hh:61-69`) and `type_names[]` (`:72-76`), which are kept
/// in lockstep by hand and by nothing else.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ValueKind {
    /// `uint8_t`, named "bool" on the Python side.
    Bool = 0,
    /// `int16_t`.
    I16,
    /// `int32_t`.
    I32,
    /// `int64_t`.
    I64,
    /// `double`.
    F64,
    /// `long double`, carried opaquely as [`LongDouble`].
    LongDouble,
    /// `std::string`.
    Str,
    /// `std::vector<uint8_t>`.
    VecBool,
    /// `std::vector<int16_t>`.
    VecI16,
    /// `std::vector<int32_t>`.
    VecI32,
    /// `std::vector<int64_t>`.
    VecI64,
    /// `std::vector<double>`.
    VecF64,
    /// `std::vector<long double>`.
    VecLongDouble,
    /// `std::vector<std::string>`.
    VecStr,
    /// `boost::python::object`.
    PyObject,
}

impl ValueKind {
    /// Every member, in graph-tool's declaration order.
    pub const ALL: [ValueKind; 15] = [
        ValueKind::Bool,
        ValueKind::I16,
        ValueKind::I32,
        ValueKind::I64,
        ValueKind::F64,
        ValueKind::LongDouble,
        ValueKind::Str,
        ValueKind::VecBool,
        ValueKind::VecI16,
        ValueKind::VecI32,
        ValueKind::VecI64,
        ValueKind::VecF64,
        ValueKind::VecLongDouble,
        ValueKind::VecStr,
        ValueKind::PyObject,
    ];

    /// The Python-visible name, byte-for-byte `type_names[]`
    /// (`graph_properties.hh:72-76`). An exhaustive match, so a 16th member
    /// breaks the build here rather than silently mis-naming.
    pub const fn name(self) -> &'static str {
        match self {
            ValueKind::Bool => "bool",
            ValueKind::I16 => "int16_t",
            ValueKind::I32 => "int32_t",
            ValueKind::I64 => "int64_t",
            ValueKind::F64 => "double",
            ValueKind::LongDouble => "long double",
            ValueKind::Str => "string",
            ValueKind::VecBool => "vector<bool>",
            ValueKind::VecI16 => "vector<int16_t>",
            ValueKind::VecI32 => "vector<int32_t>",
            ValueKind::VecI64 => "vector<int64_t>",
            ValueKind::VecF64 => "vector<double>",
            ValueKind::VecLongDouble => "vector<long double>",
            ValueKind::VecStr => "vector<string>",
            ValueKind::PyObject => "python::object",
        }
    }

    /// Parse a Python-visible name.
    ///
    /// The inverse of [`name`](Self::name), and **exact**: `new_property`
    /// (`graph_python_interface.hh:674-691`) drives a `hana::for_each` over
    /// `value_types` and accepts the member whose `type_name == type_names[i]`
    /// (`:659`), with no normalisation of case, whitespace or spelling. A name
    /// that matches nothing there raises `ValueException("Invalid property
    /// type: " + type)` (`:689`); here it is `None`.
    ///
    /// The alias table -- `"int" -> "int32_t"`, `"float" -> "double"`,
    /// `"object" -> "python::object"`, and the `vector<...>` rewrite --
    /// belongs to `_type_alias` (`graph_tool/__init__.py:210-229`), which runs
    /// *above* the C++ boundary and never reaches `type_names[]`. Folding it
    /// in here would make `from_name(name(k))` no longer the only way to
    /// obtain a kind from a string, and would silently accept spellings
    /// graph-tool's own C++ rejects. It stays at the Python boundary.
    ///
    /// `const` so the round-trip against [`name`](Self::name) is proved at
    /// compile time rather than asserted in a test; see the `const` block
    /// below. It matches on the bytes because `str` patterns are not yet
    /// allowed in a `const fn`, which changes nothing about the generated
    /// code: either spelling lowers to a length-bucketed comparison chain,
    /// not to the C++ linear scan.
    pub const fn from_name(s: &str) -> Option<ValueKind> {
        Some(match s.as_bytes() {
            b"bool" => ValueKind::Bool,
            b"int16_t" => ValueKind::I16,
            b"int32_t" => ValueKind::I32,
            b"int64_t" => ValueKind::I64,
            b"double" => ValueKind::F64,
            b"long double" => ValueKind::LongDouble,
            b"string" => ValueKind::Str,
            b"vector<bool>" => ValueKind::VecBool,
            b"vector<int16_t>" => ValueKind::VecI16,
            b"vector<int32_t>" => ValueKind::VecI32,
            b"vector<int64_t>" => ValueKind::VecI64,
            b"vector<double>" => ValueKind::VecF64,
            b"vector<long double>" => ValueKind::VecLongDouble,
            b"vector<string>" => ValueKind::VecStr,
            b"python::object" => ValueKind::PyObject,
            _ => return None,
        })
    }

    /// `hana::filter(value_types, is_scalar)` (`:78-80`), minus
    /// `long double`, which has no arithmetic here. See the module docs.
    pub const fn is_scalar(self) -> bool {
        matches!(
            self,
            ValueKind::Bool | ValueKind::I16 | ValueKind::I32 | ValueKind::I64 | ValueKind::F64
        )
    }

    /// `integer_types` (`:83`).
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            ValueKind::Bool | ValueKind::I16 | ValueKind::I32 | ValueKind::I64
        )
    }

    /// `floating_types` (`:87`).
    pub const fn is_floating(self) -> bool {
        matches!(self, ValueKind::F64 | ValueKind::LongDouble)
    }

    /// `vector_types` (`:91`).
    pub const fn is_vector(self) -> bool {
        self.elem().is_some()
    }

    /// The element kind of a vector member.
    pub const fn elem(self) -> Option<ValueKind> {
        match self {
            ValueKind::VecBool => Some(ValueKind::Bool),
            ValueKind::VecI16 => Some(ValueKind::I16),
            ValueKind::VecI32 => Some(ValueKind::I32),
            ValueKind::VecI64 => Some(ValueKind::I64),
            ValueKind::VecF64 => Some(ValueKind::F64),
            ValueKind::VecLongDouble => Some(ValueKind::LongDouble),
            ValueKind::VecStr => Some(ValueKind::Str),
            _ => None,
        }
    }
}

// `from_name(name(k)) == Some(k)` for all fifteen members, proved at compile
// time. `const _` rather than a named item: it is a proof obligation, not a
// value, and nothing should be able to "use" it.
//
// The pair this replaces -- `value_types` (`graph_properties.hh:61-69`) and
// `type_names[]` (`:72-76`) -- is held in correspondence by **position in two
// separate declarations** and by nothing else; `hana::index_if` (`:656`)
// happily indexes `type_names[]` with whatever index the type tuple yields,
// so inserting a member into one list and not the other mis-names every
// member after it, silently, in `PropertyMap.value_type()` and in every `.gt`
// header written thereafter.
//
// Here the two directions are independent matches, so the same mistake is
// still *writable* -- but it is not *compilable*: a member whose `name` arm
// and `from_name` arm disagree fails this evaluation, and a member missing
// from `from_name` falls into its `_` arm and fails it too. There is no
// runtime path to the inconsistent state, which is why this is a `const`
// block rather than a `#[test]`.
const _: () = {
    let mut i = 0;
    while i < ValueKind::ALL.len() {
        let k = ValueKind::ALL[i];
        // `as u8` because `PartialEq::eq` is not usable in a const context;
        // the discriminants are explicit (`#[repr(u8)]`, `Bool = 0`), so the
        // comparison is exact rather than approximate.
        assert!(
            matches!(ValueKind::from_name(k.name()), Some(g) if g as u8 == k as u8),
            "ValueKind::name and ValueKind::from_name disagree: some member \
             does not round-trip through its type_names[] spelling"
        );
        i += 1;
    }
};

/// A member of the value universe.
///
/// `Elem` replaces the whole `last_type_func_t` family (`dispatch.hh:113`,
/// `:301-385`) by *projection* rather than by search: `velem_dprop_t` and its
/// siblings collapse a dispatch axis to one element, but `hana::for_each`
/// still runs over that element, still performs the three-way `any_cast`
/// probe, and can still reach the `DispatchNotFound` throw. An associated
/// type adds no arm, no probe and no failure mode.
pub trait PropValue: sealed::Sealed + 'static {
    /// This type's member.
    const KIND: ValueKind;
    /// `Self` for scalars and the Python member, the element type for vectors.
    type Elem: PropValue;
}

/// Values that can be constructed with no context: the 14 non-Python members.
pub trait Zeroed: PropValue {
    /// The value a freshly grown property-map slot holds.
    fn zero() -> Self;
}

/// Values that carry no CPython reference and may therefore cross threads.
///
/// Every parallel entry point in the port is bounded by this. Because the
/// supertrait chain reaches the private seal, no crate can add an impl.
pub trait GilFree: PropValue + Clone + Send + Sync {}

/// Widening to `f64`.
///
/// `Into<f64>` cannot serve: there is no `impl From<i64> for f64` (it is
/// lossy), and `int64_t` is graph-tool's most-used scalar property type. A
/// design bounded on `Into<f64>` silently ships a scalar axis with `i64`
/// missing.
pub trait ToF64: Copy {
    /// Widen, accepting the precision loss that `i64 -> f64` can incur.
    fn to_f64(self) -> f64;
}

/// Arithmetic-capable scalar members: `{uint8_t, int16_t, int32_t, int64_t, double}`.
pub trait Scalar: PropValue + GilFree + Copy + ToF64 + PartialOrd {}

/// Opaque carrier for `long double` and the element of `vector<long double>`.
///
/// Storage and round-trip only: no arithmetic, no [`ToF64`], no [`Scalar`].
/// The x86-64 `long double` is 80-bit extended stored in 16 bytes; this
/// preserves the bytes so `.gt` files written by graph-tool load and re-save
/// unchanged.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct LongDouble(pub [u8; 16]);

// `sizeof(long double) == 16` on every ABI graph-tool builds `.gt` files on
// (x86-64 System V: 80-bit extended, padded to 16; AArch64: true binary128,
// also 16). The payload is the *storage*, so a narrower carrier would
// truncate a value on load and write a different one back out.
//
// Alignment deliberately is **not** matched: C++ gives `long double` a
// 16-byte alignment, `[u8; 16]` has 1. Nothing here reinterprets the bytes as
// a float -- that is the whole point of the opaque carrier -- and the size
// equality is what keeps `Vec<LongDouble>`'s element stride equal to the
// C++ array's.
const _: () = assert!(size_of::<LongDouble>() == 16);

macro_rules! impl_scalar_value {
    ($($t:ty => $k:ident),+ $(,)?) => {$(
        impl sealed::Sealed for $t {}
        impl PropValue for $t { const KIND: ValueKind = ValueKind::$k; type Elem = $t; }
        impl Zeroed for $t { #[inline] fn zero() -> Self { 0 as $t } }
        impl GilFree for $t {}
        impl ToF64 for $t { #[inline] fn to_f64(self) -> f64 { self as f64 } }
        impl Scalar for $t {}
    )+};
}
impl_scalar_value!(u8 => Bool, i16 => I16, i32 => I32, i64 => I64, f64 => F64);

impl sealed::Sealed for LongDouble {}
impl PropValue for LongDouble {
    const KIND: ValueKind = ValueKind::LongDouble;
    type Elem = LongDouble;
}
impl Zeroed for LongDouble {
    #[inline]
    fn zero() -> Self {
        LongDouble([0; 16])
    }
}
impl GilFree for LongDouble {}

impl sealed::Sealed for String {}
impl PropValue for String {
    const KIND: ValueKind = ValueKind::Str;
    type Elem = String;
}
impl Zeroed for String {
    #[inline]
    fn zero() -> Self {
        String::new()
    }
}
impl GilFree for String {}

macro_rules! impl_vector_value {
    ($($t:ty => $k:ident),+ $(,)?) => {$(
        impl sealed::Sealed for Vec<$t> {}
        impl PropValue for Vec<$t> { const KIND: ValueKind = ValueKind::$k; type Elem = $t; }
        impl Zeroed for Vec<$t> { #[inline] fn zero() -> Self { Vec::new() } }
        impl GilFree for Vec<$t> {}
    )+};
}
impl_vector_value!(
    u8 => VecBool,
    i16 => VecI16,
    i32 => VecI32,
    i64 => VecI64,
    f64 => VecF64,
    LongDouble => VecLongDouble,
    String => VecStr,
);

/// The Python member of the universe.
///
/// `!Send` and `!Sync` **by construction**. pyo3's `Py<T>` is unconditionally
/// `Send + Sync` (`pyo3-0.22.6/src/instance.rs:943-944`), and both `Clone`
/// and `Drop` do refcount work without requiring a `Python<'py>` token, so a
/// bare handle in a property map reaches rayon with nothing to stop it. The
/// `PhantomData<*mut ()>` is what turns
/// `graph_properties_copy.cc:35-42`'s inverted `is_python` predicate -- which
/// drops the GIL and enables `#pragma omp parallel` in exactly the
/// object-to-object case -- into a compile error.
#[cfg(feature = "python")]
pub struct PyValue {
    obj: pyo3::Py<pyo3::PyAny>,
    _not_send: std::marker::PhantomData<*mut ()>,
}

#[cfg(feature = "python")]
impl PyValue {
    /// Wrap an owned handle.
    pub fn new(obj: pyo3::Py<pyo3::PyAny>) -> Self {
        PyValue {
            obj,
            _not_send: std::marker::PhantomData,
        }
    }
    /// Borrow under a GIL token.
    pub fn bind<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Bound<'py, pyo3::PyAny> {
        self.obj.bind(py).clone()
    }
    /// Duplicate the handle. Requires the token, so the incref is sound.
    pub fn clone_ref(&self, py: pyo3::Python<'_>) -> Self {
        PyValue::new(self.obj.clone_ref(py))
    }
    /// `Py_None`. This is the only way to make a "default" Python value, and
    /// it needs the interpreter -- which is why [`Zeroed`] is not implemented.
    pub fn none(py: pyo3::Python<'_>) -> Self {
        PyValue::new(py.None())
    }
}

#[cfg(feature = "python")]
impl sealed::Sealed for PyValue {}
#[cfg(feature = "python")]
impl PropValue for PyValue {
    const KIND: ValueKind = ValueKind::PyObject;
    /// `velem_dprop_t` needs no `if constexpr (is_same_v<val_t, python::object>)`
    /// special case (`dispatch.hh:344`): the projection simply is the identity.
    type Elem = PyValue;
}
#[cfg(feature = "python")]
impl std::fmt::Debug for PyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PyValue(..)")
    }
}
// NOTE: deliberately no `Zeroed`, no `Clone`, no `GilFree`, no `Send`, no `Sync`.
