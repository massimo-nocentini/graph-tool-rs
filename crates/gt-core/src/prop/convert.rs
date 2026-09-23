//! Conversion between members of the value universe.
//!
//! ## Why `From` cannot serve (DESIGN.md D7)
//!
//! `value_convert.hh:135` handles `is_vector_v<To> && is_vector_v<From>`, and
//! `prop_map_as` narrows as well as widens. Both
//! `impl From<Vec<i32>> for Vec<i64>` and `impl From<f64> for i32` are
//! `error[E0117]` from *any* crate, since `Vec` and the primitives are all
//! foreign. A design bounded on `U: From<P::Value>` therefore buys exactly
//! std's widening scalar conversions and nothing else. [`ConvertFrom`] is
//! crate-local, so every pair in the real lattice is implementable.
//!
//! ## The N+M collapse
//!
//! `prop_map_as` (`graph_properties.cc:69-87`) crosses two 30-wide
//! property-map ranges: 900 instantiations, which then run a
//! `std::is_same_v<decltype(map), decltype(tgt)>` check at runtime to discover
//! that 30 of them are the identity. Here the 15 diagonal cases stay
//! monomorphic and fast, and the 210 off-diagonal ones route through
//! [`AnyValue`]: 15 `to_any` plus 15 `from_any`, i.e. 30 monomorphisations in
//! place of 225 pairwise `convert<S, T>`. The cost is an intermediate
//! allocation on the off-diagonal path, which is cold.
//!
//! The single blanket `impl<S: ToAny, T: FromAny> ConvertFrom<S> for T` is what
//! makes that true: `convert_from` is `#[inline]` and its body is two
//! const-folded guards over `S::KIND` and `T::KIND` around a call into the two
//! erasure functions, so nothing pairwise survives codegen. Measured on the
//! release build of `tests/u33_convert.rs`, which instantiates all 196
//! non-Python ordered pairs: **0** `to_any`, **7** `from_any` and **0**
//! `convert_from` symbols survive, against the 900 leaves `prop_map_as` emits.
//!
//! ## Where this deliberately does *not* follow the C++
//!
//! `convert` (`value_convert.hh:73-200`) tries, in order: identity;
//! `To == python::object`; `From == python::object`; `From == string &&
//! is_scalar_v<To>`; `To == string && is_scalar_v<From>`;
//! `is_convertible_v<From, To>`; elementwise for vector-to-vector; then
//! `convert_dispatch`, which throws. Two rows of that lattice are answered
//! differently here, both on purpose:
//!
//! * **`long double`.** `std::is_scalar_v<long double>` is true, so the C++
//!   converts it to and from every scalar, to `std::string`, and elementwise
//!   inside `vector<long double>`. [`LongDouble`] is an opaque 16-byte payload
//!   with no arithmetic (DESIGN.md D7) precisely so that a value graph-tool
//!   wrote is handed back byte-for-byte rather than routed through an `f64`
//!   that cannot hold it. So `long double` and `vector<long double>` convert
//!   only to themselves; every other pair is
//!   [`NoConversion`](PropError::NoConversion). [`convertible`] says so
//!   statically, at compile time, which the C++ `is_convertible_v` also does —
//!   it simply answers `true`.
//! * **Out-of-range float-to-integer.** `is_convertible_v<double, int16_t>`
//!   selects `To(v)` at `value_convert.hh:127`, i.e. a `static_cast`, and a
//!   `double` outside the destination's range makes that **undefined
//!   behaviour**. Rust's `as` saturates, and this port keeps the saturation:
//!   `1e30_f64` becomes `i16::MAX`, not whatever `cvttsd2si` happened to
//!   leave behind. Integer-to-integer narrowing is *not* a divergence — both
//!   languages wrap modulo 2^N.
//!
//! Everything else is matched pair by pair, including the two `uint8_t`
//! detours at `value_convert.hh:112-116` and `:120-124`, which route the
//! "bool" member through `int` so that it prints as `"1"` rather than as a
//! control character.

use std::any::Any;
use std::str::FromStr;

use crate::error::PropError;
use crate::prop::value::{LongDouble, PropValue, ValueKind};

/// A normalised carrier for cross-type property conversion.
#[derive(Clone, Debug, PartialEq)]
pub enum AnyValue {
    /// Any integral member, widened.
    Int(i64),
    /// `double`.
    Float(f64),
    /// `long double`, opaque.
    Long(LongDouble),
    /// `string`.
    Str(String),
    /// Any integral vector, widened.
    IntVec(Vec<i64>),
    /// `vector<double>`.
    FloatVec(Vec<f64>),
    /// `vector<long double>`, opaque.
    LongVec(Vec<LongDouble>),
    /// `vector<string>`.
    StrVec(Vec<String>),
    /// A Python handle. Never leaves the thread that made it, because the
    /// only value that can occupy it is `!Send`.
    Py(PyCell),
}

impl AnyValue {
    /// The member this carrier presents itself as.
    ///
    /// The carrier is *normalised*, so this is not in general the member the
    /// value came from: every integral member widens into
    /// [`Int`](AnyValue::Int) and reports [`ValueKind::I64`]. It is the kind
    /// named in a [`PropError::NoConversion`] raised by
    /// [`FromAny::from_any`] called directly; [`ConvertFrom::convert_from`]
    /// knows the true source member statically and names *that* instead.
    pub const fn kind(&self) -> ValueKind {
        match self {
            AnyValue::Int(_) => ValueKind::I64,
            AnyValue::Float(_) => ValueKind::F64,
            AnyValue::Long(_) => ValueKind::LongDouble,
            AnyValue::Str(_) => ValueKind::Str,
            AnyValue::IntVec(_) => ValueKind::VecI64,
            AnyValue::FloatVec(_) => ValueKind::VecF64,
            AnyValue::LongVec(_) => ValueKind::VecLongDouble,
            AnyValue::StrVec(_) => ValueKind::VecStr,
            AnyValue::Py(_) => ValueKind::PyObject,
        }
    }
}

/// Opaque slot for the Python member inside [`AnyValue`].
///
/// A distinct type so that `AnyValue` stays nameable in crates built without
/// the `python` feature, while remaining impossible to construct there: with
/// the feature off the payload is [`Infallible`](std::convert::Infallible), so
/// `AnyValue::Py` is an **uninhabited** variant rather than a reachable one
/// carrying nothing.
///
/// With the feature on it holds the handle itself, which is what makes
/// `python::object` a full member of the conversion lattice rather than a
/// 15th member that can only be named. The consequence is deliberate and
/// load-bearing: [`PyValue`](crate::prop::value::PyValue) is `!Send`, so
/// `AnyValue` is `!Send` too in a Python build, and the carrier cannot be the
/// hole through which a handle reaches rayon (DESIGN.md §7).
#[cfg(not(feature = "python"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyCell(#[doc(hidden)] pub(crate) std::convert::Infallible);

/// Opaque slot for the Python member inside [`AnyValue`].
#[cfg(feature = "python")]
pub struct PyCell(#[doc(hidden)] pub(crate) crate::prop::value::PyValue);

#[cfg(feature = "python")]
impl Clone for PyCell {
    /// `Py_INCREF` needs a token, and `PyValue` deliberately has no `Clone`.
    /// Acquiring one here is sound and cannot deadlock a worker: the handle is
    /// `!Send`, so this runs on the thread that made it, and DESIGN.md §7
    /// states explicitly that `Python::with_gil` inside a worker is the
    /// supported shape.
    fn clone(&self) -> Self {
        PyCell(pyo3::Python::with_gil(|py| self.0.clone_ref(py)))
    }
}

#[cfg(feature = "python")]
impl PartialEq for PyCell {
    /// Object identity, matching `operator==` on a `boost::python::object`
    /// handle rather than `__eq__` on the object: comparing two carriers must
    /// not be able to run arbitrary Python.
    fn eq(&self, other: &Self) -> bool {
        use pyo3::types::PyAnyMethods;
        pyo3::Python::with_gil(|py| self.0.bind(py).is(&other.0.bind(py)))
    }
}

#[cfg(feature = "python")]
impl Eq for PyCell {}

#[cfg(feature = "python")]
impl std::fmt::Debug for PyCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PyCell(..)")
    }
}

/// Lossy or lossless conversion within the value universe.
pub trait ConvertFrom<S>: Sized {
    /// Convert, or explain why not.
    fn convert_from(s: &S) -> Result<Self, PropError>;
}

/// Erase a value into the carrier.
pub trait ToAny: PropValue {
    /// Normalise.
    fn to_any(&self) -> AnyValue;
}

/// Recover a value from the carrier.
pub trait FromAny: PropValue + Sized {
    /// Denormalise, or explain why not.
    fn from_any(a: &AnyValue) -> Result<Self, PropError>;

    /// The diagonal of the lattice: duplicate a value of this same member.
    ///
    /// Not `Clone`. Nine of the fifteen members are `Clone` and the
    /// fifteenth deliberately is not — `Py_INCREF` needs a `Python<'py>`
    /// token, which is exactly why `PyValue` has `clone_ref` and no `Clone`
    /// impl (DESIGN.md D7, §7). Bounding [`ConvertFrom`]'s blanket impl on
    /// `T: Clone` would therefore drop `python::object` out of the lattice,
    /// and `convert<python::object, python::object>` is `value_convert.hh:79`,
    /// the very first branch the C++ takes.
    fn from_same(s: &Self) -> Self;
}

// ---------------------------------------------------------------------------
// The static half of the lattice
// ---------------------------------------------------------------------------

/// `long double` carries its bytes and nothing else, so it is on the diagonal
/// and nowhere else. See the module docs.
const fn is_opaque(k: ValueKind) -> bool {
    matches!(k, ValueKind::LongDouble | ValueKind::VecLongDouble)
}

/// Whether a conversion between these two members exists at all.
///
/// The analogue of `is_convertible_v<To, From>` (`value_convert.hh:220-226`),
/// which the C++ computes by instantiating `convert<To, From, true>` and
/// asking whether the result type is `void`. Here it is an ordinary `const fn`
/// over the two [`ValueKind`]s, so [`ConvertFrom::convert_from`] folds it away
/// and an impossible pair compiles to a constant `Err` that never reaches
/// [`ToAny::to_any`] at all.
///
/// **Type level, not value level.** `string -> int32_t` is convertible; the
/// string `"abc"` still fails, exactly as `boost::lexical_cast` still throws.
/// The rules, read off `value_convert.hh:73-165`:
///
/// * the diagonal always converts (`:79`);
/// * `long double` and `vector<long double>` convert only on the diagonal —
///   this port's divergence, see the module docs;
/// * `python::object` converts to and from everything else (`:82`, `:86-107`),
///   including the vector members, whose element-wise `extract` fallback lives
///   at `:96-106`;
/// * otherwise a scalar-or-string converts to a scalar-or-string (`:110`,
///   `:118`, `:126`) and a vector converts to a vector (`:130-144`), and the
///   two groups do not mix: `convert_dispatch`'s primary template throws
///   (`:204-210`).
pub const fn convertible(from: ValueKind, to: ValueKind) -> bool {
    if from as u8 == to as u8 {
        return true;
    }
    if is_opaque(from) || is_opaque(to) {
        return false;
    }
    if matches!(from, ValueKind::PyObject) || matches!(to, ValueKind::PyObject) {
        return true;
    }
    from.is_vector() == to.is_vector()
}

/// A pair that does not exist in the lattice, or a value that did not survive
/// one that does.
///
/// `#[cold]`: every caller is an error path, and keeping the `PropError`
/// construction out of line leaves the conversion bodies branch-free.
#[cold]
fn no_conversion(from: ValueKind, to: ValueKind) -> PropError {
    PropError::NoConversion { from, to }
}

impl<S: ToAny, T: FromAny> ConvertFrom<S> for T {
    #[inline]
    fn convert_from(s: &S) -> Result<T, PropError> {
        // The diagonal. `ValueKind` is injective over the universe -- each of
        // the fifteen types is the sole inhabitant of its member -- so equal
        // kinds mean `S == T` and the downcast cannot fail; it is here because
        // proving that to the compiler is what makes this safe rather than a
        // `transmute`. The guard is a comparison of two associated consts, so
        // the whole block folds away on every off-diagonal leaf and the
        // `TypeId` compare never reaches a real build.
        if S::KIND as u8 == T::KIND as u8
            && let Some(same) = (s as &dyn Any).downcast_ref::<T>()
        {
            return Ok(T::from_same(same));
        }
        // Also const, because `convertible` is a `const fn` over two
        // associated consts: an impossible pair folds to a constant `Err` and
        // never touches the carrier, so `to_any` is not instantiated for it.
        if !convertible(S::KIND, T::KIND) {
            return Err(no_conversion(S::KIND, T::KIND));
        }
        T::from_any(&s.to_any()).map_err(|e| match e {
            // `from_any` only sees the *normalised* member, so it names
            // `int64_t` where the caller passed a `uint8_t` map. The static
            // half of the pair is known here; say the truth.
            PropError::NoConversion { .. } => no_conversion(S::KIND, T::KIND),
            other => other,
        })
    }
}

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

/// `boost::lexical_cast<P>(s)`: strict, whole-string, no surrounding
/// whitespace. `str::parse` agrees on all of that, and also on rejecting
/// `"1.5"` for an integral `P` and `"70000"` for `int16_t`.
fn parse_lex<P: FromStr>(s: &str) -> Option<P> {
    s.parse::<P>().ok()
}

/// `boost::lexical_cast<std::string>(double)`.
///
/// Boost writes through a stream at `lcast_precision<double>::value == 17`
/// significant digits in the default (`%g`-like) float format. Rust's
/// `Display` writes the *shortest* decimal that round-trips. Both round-trip;
/// this one is shorter, and `0.1` prints as `"0.1"` rather than as
/// `"0.10000000000000001"`. The non-finite spellings are matched exactly,
/// which `Display` alone does not do: it writes `NaN`, and every `.gt` and
/// GraphML file in existence says `nan`.
fn fmt_f64(x: f64) -> String {
    if x.is_nan() {
        "nan".to_owned()
    } else {
        // `inf` and `-inf` already agree with boost.
        x.to_string()
    }
}

#[cfg(feature = "python")]
fn py_extract<T>(c: &PyCell, to: ValueKind) -> Result<T, PropError>
where
    T: for<'py> pyo3::FromPyObject<'py>,
{
    use pyo3::types::PyAnyMethods;
    pyo3::Python::with_gil(|py| {
        let bound = c.0.bind(py);
        bound
            .extract::<T>()
            .map_err(|_| no_conversion(ValueKind::PyObject, to))
    })
}

/// The four integral members. `$lex` is the type `boost::lexical_cast` is
/// asked for when the source is a `std::string`, which is **not** always
/// `$t`: `value_convert.hh:112-116` runs `uint8_t` through `int` first, so
/// `"300"` parses as the `int` 300 and only then narrows to 44 — whereas
/// `"5000000000"` is rejected by `lexical_cast<int>` rather than silently
/// wrapping through `int64_t`.
macro_rules! impl_integral_member {
    ($($t:ty => $lex:ty),+ $(,)?) => {$(
        impl ToAny for $t {
            #[inline]
            fn to_any(&self) -> AnyValue {
                AnyValue::Int(i64::from(*self))
            }
        }

        impl FromAny for $t {
            fn from_any(a: &AnyValue) -> Result<Self, PropError> {
                let from = a.kind();
                match a {
                    // `To(v)` (`:127`): modulo 2^N in both languages.
                    AnyValue::Int(i) => Ok(*i as $t),
                    // `To(v)` again, but out of range is UB there and
                    // saturating here. See the module docs.
                    AnyValue::Float(f) => Ok(*f as $t),
                    AnyValue::Str(s) => parse_lex::<$lex>(s)
                        .map(|v| v as $t)
                        .ok_or_else(|| no_conversion(from, Self::KIND)),
                    #[cfg(feature = "python")]
                    AnyValue::Py(c) => py_extract::<$t>(c, Self::KIND),
                    _ => Err(no_conversion(from, Self::KIND)),
                }
            }

            #[inline]
            fn from_same(s: &Self) -> Self {
                *s
            }
        }
    )+};
}

impl_integral_member!(u8 => i32, i16 => i16, i32 => i32, i64 => i64);

impl ToAny for f64 {
    #[inline]
    fn to_any(&self) -> AnyValue {
        AnyValue::Float(*self)
    }
}

impl FromAny for f64 {
    fn from_any(a: &AnyValue) -> Result<Self, PropError> {
        let from = a.kind();
        match a {
            // `is_convertible_v<int64_t, double>` (`:126`): widening, and
            // lossy past 2^53 in both languages.
            AnyValue::Int(i) => Ok(*i as f64),
            AnyValue::Float(f) => Ok(*f),
            AnyValue::Str(s) => {
                parse_lex::<f64>(s).ok_or_else(|| no_conversion(from, Self::KIND))
            }
            #[cfg(feature = "python")]
            AnyValue::Py(c) => py_extract::<f64>(c, Self::KIND),
            _ => Err(no_conversion(from, Self::KIND)),
        }
    }

    #[inline]
    fn from_same(s: &Self) -> Self {
        *s
    }
}

impl ToAny for LongDouble {
    #[inline]
    fn to_any(&self) -> AnyValue {
        AnyValue::Long(*self)
    }
}

impl FromAny for LongDouble {
    fn from_any(a: &AnyValue) -> Result<Self, PropError> {
        match a {
            AnyValue::Long(l) => Ok(*l),
            other => Err(no_conversion(other.kind(), Self::KIND)),
        }
    }

    #[inline]
    fn from_same(s: &Self) -> Self {
        *s
    }
}

impl ToAny for String {
    #[inline]
    fn to_any(&self) -> AnyValue {
        AnyValue::Str(self.clone())
    }
}

impl FromAny for String {
    fn from_any(a: &AnyValue) -> Result<Self, PropError> {
        let from = a.kind();
        match a {
            // `lexical_cast<std::string>(convert<int, uint8_t>(v))`
            // (`:120-124`): the "bool" member prints as `"1"`, never as a
            // control character. Widening into the carrier already did it.
            AnyValue::Int(i) => Ok(i.to_string()),
            AnyValue::Float(f) => Ok(fmt_f64(*f)),
            AnyValue::Str(s) => Ok(s.clone()),
            #[cfg(feature = "python")]
            AnyValue::Py(c) => py_extract::<String>(c, Self::KIND),
            _ => Err(no_conversion(from, Self::KIND)),
        }
    }

    #[inline]
    fn from_same(s: &Self) -> Self {
        s.clone()
    }
}

// ---------------------------------------------------------------------------
// Vectors
//
// `value_convert.hh:130-144`: vector-to-vector is elementwise and only when
// the *elements* convert, and one failed element fails the whole conversion
// (the `try` block at `:75` catches nothing until it unwinds out of the loop).
// `collect::<Result<Vec<_>, _>>()` is the same short-circuit.
// ---------------------------------------------------------------------------

macro_rules! impl_integral_vector_member {
    ($($t:ty => $lex:ty),+ $(,)?) => {$(
        impl ToAny for Vec<$t> {
            #[inline]
            fn to_any(&self) -> AnyValue {
                AnyValue::IntVec(self.iter().map(|&x| i64::from(x)).collect())
            }
        }

        impl FromAny for Vec<$t> {
            fn from_any(a: &AnyValue) -> Result<Self, PropError> {
                let from = a.kind();
                match a {
                    AnyValue::IntVec(v) => Ok(v.iter().map(|&x| x as $t).collect()),
                    AnyValue::FloatVec(v) => Ok(v.iter().map(|&x| x as $t).collect()),
                    AnyValue::StrVec(v) => v
                        .iter()
                        .map(|s| {
                            parse_lex::<$lex>(s)
                                .map(|x| x as $t)
                                .ok_or_else(|| no_conversion(from, Self::KIND))
                        })
                        .collect(),
                    #[cfg(feature = "python")]
                    AnyValue::Py(c) => py_extract::<Vec<$t>>(c, Self::KIND),
                    _ => Err(no_conversion(from, Self::KIND)),
                }
            }

            #[inline]
            fn from_same(s: &Self) -> Self {
                s.clone()
            }
        }
    )+};
}

impl_integral_vector_member!(u8 => i32, i16 => i16, i32 => i32, i64 => i64);

impl ToAny for Vec<f64> {
    #[inline]
    fn to_any(&self) -> AnyValue {
        AnyValue::FloatVec(self.clone())
    }
}

impl FromAny for Vec<f64> {
    fn from_any(a: &AnyValue) -> Result<Self, PropError> {
        let from = a.kind();
        match a {
            AnyValue::IntVec(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            AnyValue::FloatVec(v) => Ok(v.clone()),
            AnyValue::StrVec(v) => v
                .iter()
                .map(|s| parse_lex::<f64>(s).ok_or_else(|| no_conversion(from, Self::KIND)))
                .collect(),
            #[cfg(feature = "python")]
            AnyValue::Py(c) => py_extract::<Vec<f64>>(c, Self::KIND),
            _ => Err(no_conversion(from, Self::KIND)),
        }
    }

    #[inline]
    fn from_same(s: &Self) -> Self {
        s.clone()
    }
}

impl ToAny for Vec<LongDouble> {
    #[inline]
    fn to_any(&self) -> AnyValue {
        AnyValue::LongVec(self.clone())
    }
}

impl FromAny for Vec<LongDouble> {
    fn from_any(a: &AnyValue) -> Result<Self, PropError> {
        match a {
            AnyValue::LongVec(v) => Ok(v.clone()),
            other => Err(no_conversion(other.kind(), Self::KIND)),
        }
    }

    #[inline]
    fn from_same(s: &Self) -> Self {
        s.clone()
    }
}

impl ToAny for Vec<String> {
    #[inline]
    fn to_any(&self) -> AnyValue {
        AnyValue::StrVec(self.clone())
    }
}

impl FromAny for Vec<String> {
    fn from_any(a: &AnyValue) -> Result<Self, PropError> {
        let from = a.kind();
        match a {
            AnyValue::IntVec(v) => Ok(v.iter().map(|x| x.to_string()).collect()),
            AnyValue::FloatVec(v) => Ok(v.iter().map(|&x| fmt_f64(x)).collect()),
            AnyValue::StrVec(v) => Ok(v.clone()),
            #[cfg(feature = "python")]
            AnyValue::Py(c) => py_extract::<Vec<String>>(c, Self::KIND),
            _ => Err(no_conversion(from, Self::KIND)),
        }
    }

    #[inline]
    fn from_same(s: &Self) -> Self {
        s.clone()
    }
}

// ---------------------------------------------------------------------------
// The Python member
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
impl ToAny for crate::prop::value::PyValue {
    fn to_any(&self) -> AnyValue {
        AnyValue::Py(PyCell(pyo3::Python::with_gil(|py| self.clone_ref(py))))
    }
}

#[cfg(feature = "python")]
impl FromAny for crate::prop::value::PyValue {
    /// `convert<python::object, From>` is `boost::python::object(v)`
    /// (`value_convert.hh:81-84`) and cannot fail. It cannot fail here either,
    /// with the one exception the module docs record: `long double` has no
    /// Python spelling that round-trips its bytes, so the opaque carrier
    /// refuses rather than handing out an `f64` that lost the low bits.
    fn from_any(a: &AnyValue) -> Result<Self, PropError> {
        use pyo3::ToPyObject;
        let from = a.kind();
        pyo3::Python::with_gil(|py| {
            let obj = match a {
                AnyValue::Int(i) => i.to_object(py),
                AnyValue::Float(f) => f.to_object(py),
                AnyValue::Str(s) => s.to_object(py),
                AnyValue::IntVec(v) => v.to_object(py),
                AnyValue::FloatVec(v) => v.to_object(py),
                AnyValue::StrVec(v) => v.to_object(py),
                AnyValue::Py(c) => return Ok(Self::from_same(&c.0)),
                AnyValue::Long(_) | AnyValue::LongVec(_) => {
                    return Err(no_conversion(from, Self::KIND));
                }
            };
            Ok(crate::prop::value::PyValue::new(obj))
        })
    }

    fn from_same(s: &Self) -> Self {
        pyo3::Python::with_gil(|py| s.clone_ref(py))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The lattice is closed, and `convertible` is the table. Nothing outside
    /// `ValueKind::ALL` can be a source or a target, so the table is total.
    #[test]
    fn convertible_is_reflexive_and_total() {
        for &k in &ValueKind::ALL {
            assert!(convertible(k, k), "{k:?} does not convert to itself");
        }
    }

    /// `long double` is the diagonal and nothing else. This is the divergence
    /// the module docs record; if it ever stops being one, this test is the
    /// thing that has to be deleted on purpose.
    #[test]
    fn long_double_is_isolated() {
        for &k in &ValueKind::ALL {
            let opaque = matches!(k, ValueKind::LongDouble | ValueKind::VecLongDouble);
            assert_eq!(
                convertible(ValueKind::LongDouble, k),
                opaque && k == ValueKind::LongDouble
            );
            assert_eq!(
                convertible(k, ValueKind::VecLongDouble),
                opaque && k == ValueKind::VecLongDouble
            );
        }
    }

    /// A vector never converts to a scalar or a string, and vice versa:
    /// `convert_dispatch`'s primary template throws (`value_convert.hh:204`).
    #[test]
    fn the_two_groups_do_not_mix() {
        assert!(!convertible(ValueKind::VecI32, ValueKind::Str));
        assert!(!convertible(ValueKind::Str, ValueKind::VecStr));
        assert!(!convertible(ValueKind::F64, ValueKind::VecF64));
        assert!(!convertible(ValueKind::VecF64, ValueKind::I64));
    }

    /// `AnyValue` normalises, so the kind it reports is the widened one.
    #[test]
    fn carrier_kind_is_the_normalised_member() {
        assert_eq!(1u8.to_any().kind(), ValueKind::I64);
        assert_eq!(vec![1i16].to_any().kind(), ValueKind::VecI64);
        assert_eq!(1.0f64.to_any().kind(), ValueKind::F64);
    }

    /// But a `convert_from` error names the *true* pair, not the widened one.
    #[test]
    fn convert_from_names_the_true_pair() {
        let e = <Vec<String> as ConvertFrom<u8>>::convert_from(&7).unwrap_err();
        assert_eq!(
            e,
            PropError::NoConversion {
                from: ValueKind::Bool,
                to: ValueKind::VecStr
            }
        );
    }
}
