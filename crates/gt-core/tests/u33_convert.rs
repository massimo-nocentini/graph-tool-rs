//! U33 — the conversion lattice and type erasure, from outside the crate.
//!
//! Everything asserted here was read in
//! `src/graph/value_convert.hh` and `src/graph/graph_properties.hh` at the
//! stated line. Where this port answers differently from the C++ the test says
//! so and says why; there are exactly two such places, both recorded in
//! `prop::convert`'s module docs:
//!
//! * `long double` is opaque here, so it is the diagonal and nothing else;
//! * an out-of-range `double -> int` saturates here and is undefined
//!   behaviour there (`value_convert.hh:127`).

use gt_core::error::PropError;
use gt_core::ids::{VertexId, VertexTag};
use gt_core::prop::convert::{AnyValue, ConvertFrom, FromAny, ToAny, convertible};
use gt_core::prop::dynamic::{ReadOnlyAdaptor, Refusing, WriteOnlyAdaptor};
use gt_core::prop::value::ValueKind;
use gt_core::prop::{DenseProp, DynWrap, LongDouble, ReadProp, Unity};
use gt_core::GraphId;

// ===========================================================================
// 0. The universe, as the tests see it
// ===========================================================================

/// The fourteen members reachable without the `python` feature, each with a
/// canonical value chosen so that *every* type-level-possible conversion out
/// of it also succeeds at the value level. `"1"` rather than `"x"`, `1.0`
/// rather than `1e30`: this file separates "the pair does not exist" from "the
/// value did not survive", and the sweep below is about the first.
macro_rules! each_target {
    ($src:expr, $out:ident) => {
        probe::<_, u8>($src, &mut $out);
        probe::<_, i16>($src, &mut $out);
        probe::<_, i32>($src, &mut $out);
        probe::<_, i64>($src, &mut $out);
        probe::<_, f64>($src, &mut $out);
        probe::<_, LongDouble>($src, &mut $out);
        probe::<_, String>($src, &mut $out);
        probe::<_, Vec<u8>>($src, &mut $out);
        probe::<_, Vec<i16>>($src, &mut $out);
        probe::<_, Vec<i32>>($src, &mut $out);
        probe::<_, Vec<i64>>($src, &mut $out);
        probe::<_, Vec<f64>>($src, &mut $out);
        probe::<_, Vec<LongDouble>>($src, &mut $out);
        probe::<_, Vec<String>>($src, &mut $out);
    };
}

/// Every source, crossed with [`each_target`]. 14 x 14 = 196 ordered pairs.
macro_rules! every_pair {
    ($out:ident) => {
        each_target!(&1u8, $out);
        each_target!(&1i16, $out);
        each_target!(&1i32, $out);
        each_target!(&1i64, $out);
        each_target!(&1.0f64, $out);
        each_target!(&LongDouble([1; 16]), $out);
        each_target!(&"1".to_owned(), $out);
        each_target!(&vec![1u8], $out);
        each_target!(&vec![1i16], $out);
        each_target!(&vec![1i32], $out);
        each_target!(&vec![1i64], $out);
        each_target!(&vec![1.0f64], $out);
        each_target!(&vec![LongDouble([1; 16])], $out);
        each_target!(&vec!["1".to_owned()], $out);
    };
}

fn probe<S: ToAny, T: FromAny>(s: &S, out: &mut Vec<(ValueKind, ValueKind, bool)>) {
    // If any pair panicked instead of returning, the test binary would die
    // here rather than fail an assertion. That *is* the "never panics" half of
    // the acceptance: `convert` (`value_convert.hh:166-199`) answers a bad pair
    // by throwing `ValueException` out of a function every kernel calls.
    let r = <T as ConvertFrom<S>>::convert_from(s);
    out.push((S::KIND, T::KIND, r.is_ok()));
}

/// `convert<To, From>` with `To == From`.
fn diagonal<T>(v: T)
where
    T: ToAny + FromAny + PartialEq + std::fmt::Debug,
{
    let back = <T as ConvertFrom<T>>::convert_from(&v).unwrap();
    assert_eq!(v, back, "the diagonal is not the identity for {:?}", T::KIND);
}

/// `to_any` then `from_any`, the off-diagonal path's two halves back to back.
fn carrier_round_trip<T>(v: T)
where
    T: ToAny + FromAny + PartialEq + std::fmt::Debug,
{
    let a: AnyValue = v.to_any();
    let back = <T as FromAny>::from_any(&a).unwrap();
    assert_eq!(v, back, "carrier lost information for {:?}", T::KIND);
}

const MEMBERS: usize = if cfg!(feature = "python") { 15 } else { 14 };

// ===========================================================================
// 1. The lattice, pair by pair
// ===========================================================================

/// Every off-diagonal pair either converts or returns `Err(NoConversion)`, and
/// which one it does agrees exactly with the static table.
#[test]
fn every_pair_converts_or_refuses_and_never_panics() {
    let mut out = Vec::new();
    every_pair!(out);
    assert_eq!(out.len(), 14 * 14, "the sweep is not the whole lattice");

    for (from, to, ok) in out {
        assert_eq!(
            ok,
            convertible(from, to),
            "{from:?} -> {to:?}: runtime says {ok}, `convertible` says {}",
            convertible(from, to)
        );
    }
}

/// And the refusal is the right error, naming the real pair rather than the
/// widened carrier.
#[test]
fn refusal_is_no_conversion_naming_the_true_pair() {
    let e = <Vec<i32> as ConvertFrom<i16>>::convert_from(&3).unwrap_err();
    assert_eq!(
        e,
        PropError::NoConversion {
            from: ValueKind::I16,
            to: ValueKind::VecI32
        }
    );
    let e = <String as ConvertFrom<Vec<u8>>>::convert_from(&vec![1]).unwrap_err();
    assert_eq!(
        e,
        PropError::NoConversion {
            from: ValueKind::VecBool,
            to: ValueKind::Str
        }
    );
}

/// The 14 diagonal cases are the identity.
///
/// Not "convertible", not "lossless": equal. `value_convert.hh:77-80` returns
/// `(const To&)v` for `is_same_v<To, From>` before any other branch is
/// considered, and [`FromAny::from_same`] is the same first branch here.
#[test]
fn the_diagonal_is_the_identity() {
    macro_rules! diag {
        ($($v:expr),+ $(,)?) => {$( diagonal($v); )+};
    }
    diag!(
        200u8,
        -300i16,
        -70000i32,
        -5_000_000_000i64,
        -0.125f64,
        LongDouble([3; 16]),
        "a string with spaces".to_owned(),
        vec![0u8, 255],
        vec![i16::MIN, i16::MAX],
        vec![i32::MIN, i32::MAX],
        vec![i64::MIN, i64::MAX],
        vec![f64::MIN, 0.0, f64::MAX],
        vec![LongDouble([7; 16])],
        vec!["a".to_owned(), String::new()],
    );
}

/// `to_any`/`from_any` round-trip every member, which is what makes the
/// N+M collapse sound: the carrier has to be able to hold each member
/// without loss, or the off-diagonal path would be lossy in a way the
/// pairwise C++ is not.
#[test]
fn every_member_round_trips_through_the_carrier() {
    macro_rules! round_trip {
        ($($v:expr),+ $(,)?) => {$( carrier_round_trip($v); )+};
    }
    round_trip!(
        200u8,
        -300i16,
        -70000i32,
        -5_000_000_000i64,
        -0.125f64,
        LongDouble([3; 16]),
        "a string".to_owned(),
        vec![0u8, 255],
        vec![i16::MIN, i16::MAX],
        vec![i32::MIN, i32::MAX],
        vec![i64::MIN, i64::MAX],
        vec![f64::MIN, 0.0, f64::MAX],
        vec![LongDouble([7; 16])],
        vec!["a".to_owned(), String::new()],
    );
}

// ===========================================================================
// 2. `uint8_t` is not a char, it is a bool — `value_convert.hh:112-124`
// ===========================================================================

/// Both detours, in both directions. The comment in the C++ is literally
/// "uint8_t is not char, it is bool!", and what it buys is that the member
/// named `"bool"` prints as a number.
#[test]
fn uint8_routes_through_int_in_both_directions() {
    // `:120-124` — `lexical_cast<string>(convert<int, uint8_t>(v))`.
    assert_eq!(
        <String as ConvertFrom<u8>>::convert_from(&200).unwrap(),
        "200"
    );
    assert_eq!(<String as ConvertFrom<u8>>::convert_from(&0).unwrap(), "0");

    // `:112-116` — `convert<uint8_t, int>(lexical_cast<int>(v))`. The parse is
    // at `int` width and the narrowing happens afterwards, so "300" wraps to
    // 44 and "-1" wraps to 255.
    assert_eq!(<u8 as ConvertFrom<String>>::convert_from(&"7".into()).unwrap(), 7);
    assert_eq!(
        <u8 as ConvertFrom<String>>::convert_from(&"300".into()).unwrap(),
        44
    );
    assert_eq!(
        <u8 as ConvertFrom<String>>::convert_from(&"-1".into()).unwrap(),
        255
    );

    // But the parse really is at `int` width: `lexical_cast<int>` rejects a
    // value that would have fitted in `int64_t`. Routing "5000000000" through
    // the widened carrier instead would have silently produced 0.
    assert_eq!(
        <u8 as ConvertFrom<String>>::convert_from(&"5000000000".into()),
        Err(PropError::NoConversion {
            from: ValueKind::Str,
            to: ValueKind::Bool
        })
    );

    // Elementwise, the vector members take the same detour (`:130-144`).
    assert_eq!(
        <Vec<String> as ConvertFrom<Vec<u8>>>::convert_from(&vec![0, 1, 200]).unwrap(),
        vec!["0", "1", "200"]
    );
    assert_eq!(
        <Vec<u8> as ConvertFrom<Vec<String>>>::convert_from(&vec!["300".into()]).unwrap(),
        vec![44u8]
    );
}

// ===========================================================================
// 3. Narrowing — `is_convertible_v<From, To>` selects `To(v)` at `:127`
// ===========================================================================

/// Integer narrowing wraps modulo 2^N. This is **not** a divergence: the C++
/// `static_cast` to a narrower integral type has been well defined since
/// C++20, and it is what `prop_map_as` relies on.
#[test]
fn integer_narrowing_wraps_exactly_as_static_cast_does() {
    assert_eq!(<i16 as ConvertFrom<i64>>::convert_from(&70000).unwrap(), 4464);
    assert_eq!(<u8 as ConvertFrom<i64>>::convert_from(&300).unwrap(), 44);
    assert_eq!(<u8 as ConvertFrom<i32>>::convert_from(&-1).unwrap(), 255);
    assert_eq!(
        <i32 as ConvertFrom<i64>>::convert_from(&(i64::from(i32::MAX) + 1)).unwrap(),
        i32::MIN
    );
    // Widening is exact for everything a `double` can hold.
    assert_eq!(<f64 as ConvertFrom<i64>>::convert_from(&-3).unwrap(), -3.0);
    assert_eq!(<i64 as ConvertFrom<u8>>::convert_from(&255).unwrap(), 255);
}

/// Float-to-integer truncates toward zero in both languages.
#[test]
fn float_to_integer_truncates_toward_zero() {
    assert_eq!(<i64 as ConvertFrom<f64>>::convert_from(&3.9).unwrap(), 3);
    assert_eq!(<i64 as ConvertFrom<f64>>::convert_from(&-3.9).unwrap(), -3);
    assert_eq!(<i32 as ConvertFrom<f64>>::convert_from(&-0.5).unwrap(), 0);
    assert_eq!(<u8 as ConvertFrom<f64>>::convert_from(&1.75).unwrap(), 1);
}

/// **Deliberate divergence.** `To(v)` for a `double` outside the destination's
/// range is undefined behaviour (`value_convert.hh:127`); on x86-64 it is
/// whatever `cvttsd2si` leaves in the register, which for `int16_t` is the low
/// half of the "integer indefinite" value and looks like a legitimate reading.
/// Rust saturates, and this port keeps the saturation. Rule 4: implement the
/// correct behaviour and say so.
#[test]
fn out_of_range_float_to_integer_saturates_rather_than_being_undefined() {
    assert_eq!(<i16 as ConvertFrom<f64>>::convert_from(&1e30).unwrap(), i16::MAX);
    assert_eq!(
        <i16 as ConvertFrom<f64>>::convert_from(&-1e30).unwrap(),
        i16::MIN
    );
    assert_eq!(<u8 as ConvertFrom<f64>>::convert_from(&300.0).unwrap(), 255);
    assert_eq!(<u8 as ConvertFrom<f64>>::convert_from(&-1.0).unwrap(), 0);
    assert_eq!(<i64 as ConvertFrom<f64>>::convert_from(&f64::NAN).unwrap(), 0);
    assert_eq!(
        <i64 as ConvertFrom<f64>>::convert_from(&f64::INFINITY).unwrap(),
        i64::MAX
    );
    // And elementwise inside a vector, for the same reason.
    assert_eq!(
        <Vec<u8> as ConvertFrom<Vec<f64>>>::convert_from(&vec![-1.0, 300.0]).unwrap(),
        vec![0u8, 255]
    );
}

// ===========================================================================
// 4. `boost::lexical_cast`, both directions — `:110-124`
// ===========================================================================

#[test]
fn string_to_scalar_is_strict_and_whole_string() {
    assert_eq!(<f64 as ConvertFrom<String>>::convert_from(&"1e3".into()).unwrap(), 1000.0);
    assert_eq!(<i64 as ConvertFrom<String>>::convert_from(&"+5".into()).unwrap(), 5);

    for bad in ["", "abc", " 5", "5 ", "1.5", "0x10", "1,5"] {
        let r = <i32 as ConvertFrom<String>>::convert_from(&bad.to_owned());
        assert_eq!(
            r,
            Err(PropError::NoConversion {
                from: ValueKind::Str,
                to: ValueKind::I32
            }),
            "{bad:?} should not parse as int32_t"
        );
    }
    // Out of range for the *destination*, not for the carrier:
    // `lexical_cast<int16_t>("70000")` throws.
    assert!(<i16 as ConvertFrom<String>>::convert_from(&"70000".into()).is_err());
    assert_eq!(<i32 as ConvertFrom<String>>::convert_from(&"70000".into()).unwrap(), 70000);
}

#[test]
fn scalar_to_string_round_trips_and_matches_the_non_finite_spellings() {
    assert_eq!(<String as ConvertFrom<f64>>::convert_from(&1.5).unwrap(), "1.5");
    assert_eq!(<String as ConvertFrom<f64>>::convert_from(&100.0).unwrap(), "100");
    assert_eq!(<String as ConvertFrom<i64>>::convert_from(&-7).unwrap(), "-7");

    // `.gt` and GraphML files say `nan`/`inf`/`-inf`; Rust's `Display` says
    // `NaN`, which is the one place the default formatting had to be
    // overridden.
    assert_eq!(
        <String as ConvertFrom<f64>>::convert_from(&f64::NAN).unwrap(),
        "nan"
    );
    assert_eq!(
        <String as ConvertFrom<f64>>::convert_from(&f64::INFINITY).unwrap(),
        "inf"
    );
    assert_eq!(
        <String as ConvertFrom<f64>>::convert_from(&f64::NEG_INFINITY).unwrap(),
        "-inf"
    );

    // The guarantee boost's `lcast_precision` exists for: the text parses back
    // to the same bits. This port gets there with the shortest such text
    // rather than with 17 significant digits.
    for x in [0.1f64, 1.0 / 3.0, f64::MIN_POSITIVE, -2.220446049250313e-16] {
        let s = <String as ConvertFrom<f64>>::convert_from(&x).unwrap();
        let back = <f64 as ConvertFrom<String>>::convert_from(&s).unwrap();
        assert_eq!(x.to_bits(), back.to_bits(), "{s} did not round-trip");
    }
    assert_eq!(<String as ConvertFrom<f64>>::convert_from(&0.1).unwrap(), "0.1");
}

// ===========================================================================
// 5. Vectors — `:130-144`
// ===========================================================================

#[test]
fn vector_to_vector_is_elementwise_and_fails_whole() {
    assert_eq!(
        <Vec<i32> as ConvertFrom<Vec<String>>>::convert_from(&vec![
            "1".into(),
            "-2".into(),
            "3".into()
        ])
        .unwrap(),
        vec![1, -2, 3]
    );
    assert_eq!(
        <Vec<String> as ConvertFrom<Vec<f64>>>::convert_from(&vec![1.5, 100.0]).unwrap(),
        vec!["1.5", "100"]
    );
    assert_eq!(
        <Vec<f64> as ConvertFrom<Vec<i16>>>::convert_from(&vec![-1, 2]).unwrap(),
        vec![-1.0, 2.0]
    );
    // One bad element fails the conversion: the C++ `try` block at `:75`
    // catches nothing until the loop has already unwound.
    assert!(
        <Vec<i32> as ConvertFrom<Vec<String>>>::convert_from(&vec!["1".into(), "x".into()])
            .is_err()
    );
    // The empty vector converts to the empty vector, for every pair.
    assert_eq!(
        <Vec<String> as ConvertFrom<Vec<i64>>>::convert_from(&vec![]).unwrap(),
        Vec::<String>::new()
    );
}

/// A vector never converts to a scalar or a string, and no scalar or string
/// ever converts to a vector: `convert_dispatch`'s primary template
/// (`value_convert.hh:204-210`) throws `bad_lexical_cast` for exactly those
/// pairs. In particular the `lexical_cast(const std::vector<T>&)` overload
/// that `value_convert.hh:32-45` injects into `namespace boost` — the one that
/// formats `"(1, 2, 3)"` — is unreachable from `convert`, because the branch
/// that would call it is guarded by `is_scalar_v<From>`.
#[test]
fn the_two_groups_never_mix() {
    assert!(<String as ConvertFrom<Vec<i32>>>::convert_from(&vec![1, 2, 3]).is_err());
    assert!(<Vec<String> as ConvertFrom<String>>::convert_from(&"1".into()).is_err());
    assert!(<Vec<f64> as ConvertFrom<f64>>::convert_from(&1.0).is_err());
    assert!(<i64 as ConvertFrom<Vec<i64>>>::convert_from(&vec![1]).is_err());
}

// ===========================================================================
// 6. `long double` — this port's deliberate divergence
// ===========================================================================

/// `std::is_scalar_v<long double>` is true, so the C++ converts it to every
/// scalar and to `std::string`. Here it is a 16-byte opaque payload with no
/// arithmetic (`gt_core::design` D7), so it is the diagonal and nothing else — and
/// what that buys is the round-trip: a value graph-tool wrote into a `.gt`
/// file comes back byte for byte instead of through an `f64` that has 11 fewer
/// bits of mantissa and a much smaller exponent range.
#[test]
fn long_double_is_the_diagonal_and_nothing_else() {
    let ld = LongDouble([0xAB; 16]);
    assert_eq!(<LongDouble as ConvertFrom<LongDouble>>::convert_from(&ld).unwrap(), ld);

    assert_eq!(
        <f64 as ConvertFrom<LongDouble>>::convert_from(&ld),
        Err(PropError::NoConversion {
            from: ValueKind::LongDouble,
            to: ValueKind::F64
        })
    );
    assert!(<String as ConvertFrom<LongDouble>>::convert_from(&ld).is_err());
    assert!(<LongDouble as ConvertFrom<f64>>::convert_from(&1.0).is_err());
    assert!(<LongDouble as ConvertFrom<String>>::convert_from(&"1".into()).is_err());
    assert!(
        <Vec<f64> as ConvertFrom<Vec<LongDouble>>>::convert_from(&vec![ld]).is_err()
    );
    assert!(
        <Vec<LongDouble> as ConvertFrom<Vec<f64>>>::convert_from(&vec![1.0]).is_err()
    );
    // But the bytes survive the diagonal untouched, which is the whole point.
    assert_eq!(
        <Vec<LongDouble> as ConvertFrom<Vec<LongDouble>>>::convert_from(&vec![ld])
            .unwrap()[0]
            .0,
        [0xAB; 16]
    );
}

// ===========================================================================
// 7. The static table
// ===========================================================================

/// `convertible` is `is_convertible_v<To, From>` (`value_convert.hh:220-226`)
/// as a `const fn`, which is what lets an impossible pair fold to a constant
/// `Err` instead of instantiating the carrier for it.
#[test]
fn convertible_is_a_const_fn() {
    const SCALARS: bool = convertible(ValueKind::Str, ValueKind::I64);
    const VECTORS: bool = convertible(ValueKind::VecBool, ValueKind::VecStr);
    const MIXED: bool = convertible(ValueKind::Str, ValueKind::VecStr);
    const OPAQUE: bool = convertible(ValueKind::LongDouble, ValueKind::F64);
    const DIAGONAL: bool = convertible(ValueKind::VecLongDouble, ValueKind::VecLongDouble);
    // Read back through an array so the check is a comparison of values, not
    // an `assert!` on a literal the compiler has already decided: the point is
    // that the five `const` initialisers above compiled at all.
    assert_eq!(
        [SCALARS, VECTORS, DIAGONAL, MIXED, OPAQUE],
        [true, true, true, false, false]
    );
}

/// The shape of the table, counted rather than described.
///
/// 15 x 15 = 225 ordered pairs. `long double` and `vector<long double>` are
/// the diagonal and nothing else (this port's divergence, §6 above);
/// `python::object` reaches everything that is not opaque, in both directions
/// (`value_convert.hh:81-107`); and the remaining twelve members split into
/// six scalars-or-string and six vectors that convert within their group and
/// never across it. That is 99 pairs, and nothing else is a conversion.
#[test]
fn the_table_has_the_shape_the_design_claims() {
    let mut yes = 0usize;
    for &a in &ValueKind::ALL {
        for &b in &ValueKind::ALL {
            if convertible(a, b) {
                yes += 1;
            }
        }
    }

    let n = ValueKind::ALL.len(); // 15, and feature-independent: it is the enum
    let opaque = 2; // long double, vector<long double>
    let py = 1; // python::object
    let group = (n - opaque - py) / 2; // 6 scalar-or-string, 6 vector

    let expected = n                         // the diagonal, always
        + 2 * py * (n - py - opaque)         // python::object, both directions
        + 2 * (group * group - group); // within each group, off-diagonal
    assert_eq!(yes, expected, "the lattice changed shape");
    assert_eq!(yes, 99);
}

// ===========================================================================
// 8. Type erasure — `DynamicPropertyMapWrap`, `graph_properties.hh:389-492`
// ===========================================================================

fn v(i: usize) -> VertexId {
    VertexId::new(i).expect("small index")
}

fn dense_i64(values: Vec<i64>) -> DenseProp<i64, VertexTag> {
    DenseProp::from_vec(GraphId::fresh(), values)
}

/// The N x M product collapses: one `int64_t` map, read and written as three
/// different members, with one adaptor and no pairwise code.
#[test]
fn dyn_wrap_converts_in_both_directions() {
    let mut as_str: DynWrap<VertexTag, String> = DynWrap::wrap(dense_i64(vec![1, -2, 3]));
    assert_eq!(as_str.underlying(), ValueKind::I64);
    assert_eq!(as_str.get(v(1)).unwrap(), "-2");
    as_str.put(v(1), "42".to_owned()).unwrap();
    assert_eq!(as_str.get(v(1)).unwrap(), "42");

    let mut as_f64: DynWrap<VertexTag, f64> = DynWrap::wrap(dense_i64(vec![1, -2, 3]));
    assert_eq!(as_f64.get(v(2)).unwrap(), 3.0);
    // The narrowing on the way back in is `convert<val_t>(val)`
    // (`graph_properties.hh:457`): it happens before the write, at the
    // *stored* member's width.
    as_f64.put(v(2), -3.9).unwrap();
    assert_eq!(as_f64.get(v(2)).unwrap(), -3.0);

    let mut same: DynWrap<VertexTag, i64> = DynWrap::wrap(dense_i64(vec![9]));
    assert_eq!(same.get(v(0)).unwrap(), 9);
    same.put(v(0), 11).unwrap();
    assert_eq!(same.get(v(0)).unwrap(), 11);
}

/// A failed conversion on the way in leaves the slot untouched.
#[test]
fn a_failed_put_does_not_write() {
    let mut w: DynWrap<VertexTag, String> = DynWrap::wrap(dense_i64(vec![7]));
    assert!(w.put(v(0), "not a number".to_owned()).is_err());
    assert_eq!(w.get(v(0)).unwrap(), "7");
}

/// **Defect.** `get_dispatch` and `put_dispatch` `throw ValueException`
/// (`graph_properties.hh:472`, `:489`) from inside a virtual call that
/// `graph_properties_copy.cc` makes within an OpenMP region. Here both are
/// values, and neither can panic.
#[test]
fn unreadable_and_unwritable_maps_return_err_rather_than_throwing() {
    // Readable only. `Unity` has no `WriteProp` impl at all, so this is the
    // only adaptor it can reach — and `put(UnityPropertyMap, k, v) {}`
    // (`graph_properties.hh:714`), the silent no-op, is not expressible.
    let mut ro: DynWrap<VertexTag, i64> = DynWrap::wrap_read_only(Unity::<i64, VertexTag>::NEW);
    assert_eq!(ro.get(v(5)).unwrap(), 1);
    assert_eq!(ro.put(v(5), 2), Err(PropError::NotWritable));

    // Writable only — `:472`'s throw.
    let mut wo: DynWrap<VertexTag, i64> = DynWrap::wrap_write_only(dense_i64(vec![0, 0]));
    assert_eq!(wo.get(v(0)), Err(PropError::NotReadable));
    wo.put(v(0), 5).unwrap();

    // Neither.
    let mut none: DynWrap<VertexTag, i64> = DynWrap::new(Box::new(Refusing::<i64>::NEW));
    assert_eq!(none.get(v(0)), Err(PropError::NotReadable));
    assert_eq!(none.put(v(0), 1), Err(PropError::NotWritable));

    // The un-erased adaptors answer the same way, so a caller that skipped the
    // `Box` is not a second code path.
    let mut bare = ReadOnlyAdaptor::new(dense_i64(vec![1]));
    assert_eq!(
        gt_core::prop::DynProp::<VertexTag, i64>::dyn_put(&mut bare, v(0), 2),
        Err(PropError::NotWritable)
    );
    let bare = WriteOnlyAdaptor::new(dense_i64(vec![1]));
    assert_eq!(
        gt_core::prop::DynProp::<VertexTag, i64>::dyn_get(&bare, v(0)),
        Err(PropError::NotReadable)
    );
}

/// An erased map whose stored member cannot reach the requested one refuses
/// both directions. It does not panic, and it does not silently truncate.
#[test]
fn an_impossible_erasure_refuses_instead_of_panicking() {
    let map: DenseProp<LongDouble, VertexTag> =
        DenseProp::from_vec(GraphId::fresh(), vec![LongDouble([9; 16])]);
    let mut w: DynWrap<VertexTag, i64> = DynWrap::wrap(map);
    assert_eq!(w.underlying(), ValueKind::LongDouble);
    assert_eq!(
        w.get(v(0)),
        Err(PropError::NoConversion {
            from: ValueKind::LongDouble,
            to: ValueKind::I64
        })
    );
    assert_eq!(
        w.put(v(0), 1),
        Err(PropError::NoConversion {
            from: ValueKind::I64,
            to: ValueKind::LongDouble
        })
    );
}

/// The erased map reports the member it actually stores, as a matchable value
/// rather than as the `const std::type_info&` of
/// `get_underlying_value_type` (`graph_properties.hh:425-428`).
#[test]
fn underlying_names_the_stored_member() {
    let g = GraphId::fresh();
    let w: DynWrap<VertexTag, String> =
        DynWrap::wrap(DenseProp::<Vec<f64>, VertexTag>::from_vec(g, vec![]));
    assert_eq!(w.underlying(), ValueKind::VecF64);
    assert_eq!(w.underlying().name(), "vector<double>");

    let w: DynWrap<VertexTag, i64> = DynWrap::wrap_read_only(Unity::<i64, VertexTag>::NEW);
    assert_eq!(w.underlying(), ValueKind::I64);
    // `Unity`'s const marker survives the erasure only at the concrete type;
    // that is the documented cost of `DynamicPropertyMapWrap`
    // (`graph_properties.hh:384-387`) and the reason it must not appear in a
    // kernel.
    assert_eq!(
        [
            <Unity<i64, VertexTag> as ReadProp<VertexTag>>::IS_UNITY,
            <DenseProp<i64, VertexTag> as ReadProp<VertexTag>>::IS_UNITY,
        ],
        [true, false]
    );
}

// ===========================================================================
// 9. The instantiation budget — `gt_core::design` §10
// ===========================================================================

/// `prop_map_as` (`graph_properties.cc:69-87`) crosses two 30-wide ranges and
/// emits 900 leaves. `gt_core::design` §10 claims 45 here: 15 identity + 15 `to_any` +
/// 15 `from_any`. This is that claim, measured on the symbols this very test
/// binary carries — it instantiates all 196 ordered pairs above, so if the
/// lattice were pairwise the count would be in the hundreds.
///
/// Asserted only on a build with the optimiser on. `convert_from` is
/// `#[inline]` and its body is three const-folded tests, so whether its 196
/// monomorphisations leave a symbol behind is an inliner decision, not a
/// design property; at `opt-level = 0` they all do. Run it with
/// `cargo test --release -p gt-core --test u33_convert`.
#[test]
fn the_lattice_is_n_plus_m_not_n_times_m() {
    let exe = std::env::current_exe().expect("test binary path");
    let out = match std::process::Command::new("nm")
        .args(["--defined-only", "-C"])
        .arg(&exe)
        .output()
    {
        Ok(o) if o.status.success() => o.stdout,
        // No `nm`, or a platform whose object format it cannot read. The
        // census is evidence, not a portability requirement.
        _ => return,
    };
    let text = String::from_utf8_lossy(&out);

    // A demangled name may carry a nested item -- `...::from_any::{{closure}}`
    // for the `collect` in the vector arms, `::{{constant}}` for a promoted
    // literal. Cut those off, or one member's `from_any` counts several times.
    let count = |needle: &str| -> usize {
        text.lines()
            .filter_map(|l| l.rsplit_once(' ').map(|(_, sym)| sym.trim()))
            .filter(|s| s.contains("gt_core::prop::convert"))
            .map(|s| match s.find("::{{") {
                Some(i) => &s[..i],
                None => s,
            })
            .filter(|s| s.ends_with(needle))
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    };

    let to_any = count("::to_any");
    let from_any = count("::from_any");
    let pairwise = count("::convert_from");
    let report =
        format!("to_any={to_any}, from_any={from_any}, convert_from={pairwise}, members={MEMBERS}");

    // Always true, in any profile: the erasure functions are one per member.
    assert!(to_any <= MEMBERS, "{report}");
    assert!(from_any <= MEMBERS, "{report}");

    if !cfg!(debug_assertions) {
        // The headline: three per member, never one per pair.
        assert!(
            to_any + from_any + pairwise <= 3 * MEMBERS,
            "the conversion lattice is no longer N+M: {report}"
        );
        assert!(
            pairwise < MEMBERS * MEMBERS,
            "convert_from was emitted pairwise: {report}"
        );
    }
}
