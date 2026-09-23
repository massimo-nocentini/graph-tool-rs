//! U8 — the value universe, checked against `graph_properties.hh` rather than
//! against itself.
//!
//! The value universe is a *transcription*: fifteen C++ types, fifteen
//! spellings, and four `hana::filter`s over them, all of which the Python
//! surface and the `.gt` on-disk format depend on byte-for-byte. A
//! transcription error here is invisible from inside the port — every test
//! that only asks "does `from_name(name(k)) == Some(k)`" passes just as well
//! with `"int_16"` as with `"int16_t"` — so the checks below carry a second,
//! independent copy of the C++ arrays and compare against *that*.
//!
//! The four set predicates get the same treatment: each is spelled out as the
//! member list the corresponding `hana::filter` produces when evaluated by
//! hand over `value_types` (`graph_properties.hh:61-69`), and the one place
//! the port deliberately disagrees with C++ — `long double` is floating but
//! **not** scalar, because it has no arithmetic here — is asserted as a
//! divergence rather than left to be discovered as a discrepancy.

use std::any::TypeId;
use std::collections::BTreeSet;

use gt_core::ids::{GraphId, Id, VertexTag};
use gt_core::prop::{
    DenseProp, GilFree, LongDouble, LvalueProp, PropValue, ReadProp, Scalar, ToF64, ValueKind,
    WriteProp, Zeroed,
};

// ===========================================================================
// 1. The names
// ===========================================================================

/// `type_names[]`, transcribed from `src/graph/graph_properties.hh:71-75`.
///
/// Deliberately a literal, not `ValueKind::ALL.map(ValueKind::name)`: the
/// whole point is to have a copy that did not come from the code under test.
/// Order matters as much as spelling — `hana::index_if(value_types, ...)`
/// (`graph_python_interface.hh:656`, `graph_python_interface_export.cc:45`)
/// indexes this array by the type's position in `value_types`, so the two
/// declarations are coupled positionally and by nothing else.
const CPP_TYPE_NAMES: [&str; 15] = [
    "bool",
    "int16_t",
    "int32_t",
    "int64_t",
    "double",
    "long double",
    "string",
    "vector<bool>",
    "vector<int16_t>",
    "vector<int32_t>",
    "vector<int64_t>",
    "vector<double>",
    "vector<long double>",
    "vector<string>",
    "python::object",
];

/// `value_types` (`graph_properties.hh:61-69`), as the port's members, in the
/// order the C++ tuple declares them.
const CPP_VALUE_TYPES: [ValueKind; 15] = [
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

#[test]
fn every_name_is_byte_identical_to_type_names() {
    assert_eq!(
        ValueKind::ALL.len(),
        CPP_TYPE_NAMES.len(),
        "the universe is closed at 15 members (graph_properties.hh:61-69)"
    );

    for (i, (&k, &expected)) in ValueKind::ALL.iter().zip(&CPP_TYPE_NAMES).enumerate() {
        assert_eq!(
            k.name().as_bytes(),
            expected.as_bytes(),
            "type_names[{i}]: {k:?} is spelled {:?}, graph_properties.hh:71-75 \
             spells it {expected:?}. PropertyMap.value_type() and every .gt \
             header written by this port carry this string verbatim",
            k.name()
        );
    }

    // The one the Python surface is most likely to be broken by: uint8_t's
    // name is "bool", not "uint8_t". `_python_type` (`__init__.py:241`) keys
    // off exactly this string to decide that the map yields `bool`, and
    // `_type_alias` (`:210`) maps *both* "int8_t" and "boolean" onto it.
    assert_eq!(ValueKind::Bool.name(), "bool");
    // ...while "int16_t" and friends keep their C spelling.
    assert_eq!(ValueKind::I16.name(), "int16_t");
    // A single space, not an underscore and not "longdouble".
    assert_eq!(ValueKind::LongDouble.name(), "long double");
    // `boost::python::object` is announced with the *inner* qualification only.
    assert_eq!(ValueKind::PyObject.name(), "python::object");
}

#[test]
fn all_is_the_cpp_declaration_order_with_matching_discriminants() {
    assert_eq!(ValueKind::ALL, CPP_VALUE_TYPES);

    // `#[repr(u8)]` with `Bool = 0` and implicit successors: position in
    // `ALL` is the discriminant is the index into `type_names[]`. Anything
    // that reorders one and not the others is caught here.
    for (i, &k) in ValueKind::ALL.iter().enumerate() {
        assert_eq!(k as u8 as usize, i, "{k:?} is not at its own discriminant");
    }

    let distinct: BTreeSet<ValueKind> = ValueKind::ALL.into_iter().collect();
    assert_eq!(distinct.len(), 15, "ValueKind::ALL repeats a member");

    let names: BTreeSet<&str> = ValueKind::ALL.iter().map(|k| k.name()).collect();
    assert_eq!(names.len(), 15, "two members share a spelling");
}

#[test]
fn every_member_round_trips_through_its_name() {
    for k in ValueKind::ALL {
        assert_eq!(
            ValueKind::from_name(k.name()),
            Some(k),
            "{k:?} does not round-trip through {:?}",
            k.name()
        );
    }
    // And the other direction, from the independent copy of the array.
    for (&name, &k) in CPP_TYPE_NAMES.iter().zip(&CPP_VALUE_TYPES) {
        assert_eq!(ValueKind::from_name(name), Some(k), "from_name({name:?})");
    }
}

#[test]
fn from_name_is_exact_like_new_property() {
    // `new_property` (`graph_python_interface.hh:673-690`) compares
    // `type_name == type_names[i]` with `std::string::operator==` and throws
    // `ValueException("Invalid property type: " + type)` when nothing matches.
    // No trimming, no case folding, no alias expansion.
    for bad in [
        "",
        " ",
        "bool ",
        " bool",
        "BOOL",
        "Bool",
        "Boolean",
        "longdouble",
        "long  double",
        "long_double",
        "LONG DOUBLE",
        "vector<bool >",
        "vector< bool>",
        "vector<bool",
        "vector<>",
        "python::object ",
        "object ",
        "boost::python::object",
        "uint8_t",
        "int8_t",
        "u8",
        "i64",
        "str",
        "f64",
        "double precision",
    ] {
        assert_eq!(
            ValueKind::from_name(bad),
            None,
            "{bad:?} is not in type_names[]; graph_properties.hh has no such \
             spelling and new_property would raise ValueException"
        );
    }

    // The alias table is `_type_alias` (`graph_tool/__init__.py:210-229`),
    // which runs above the C++ boundary: it rewrites the string *before*
    // `new_property` sees it. Accepting aliases here would put a second,
    // divergent spelling table in the port, and would make `name`/`from_name`
    // a non-bijection. Each of these must therefore be rejected, and each is
    // listed with the member the Python layer maps it onto.
    for (alias, target) in [
        ("int8_t", ValueKind::Bool),
        ("boolean", ValueKind::Bool),
        ("short", ValueKind::I16),
        ("int", ValueKind::I32),
        ("unsigned int", ValueKind::I32),
        ("long", ValueKind::I64),
        ("long long", ValueKind::I64),
        ("unsigned long", ValueKind::I64),
        ("object", ValueKind::PyObject),
        ("float", ValueKind::F64),
        ("vector<int>", ValueKind::VecI32),
        ("vector<float>", ValueKind::VecF64),
    ] {
        assert_eq!(
            ValueKind::from_name(alias),
            None,
            "{alias:?} is a Python-layer alias for {target:?}; the C++ \
             boundary does not know it"
        );
        // ...and the canonical spelling it aliases to does parse, so the
        // Python layer has somewhere to send it.
        assert_eq!(ValueKind::from_name(target.name()), Some(target));
    }
}

// ===========================================================================
// 2. The four hana::filter sets
// ===========================================================================

/// The members of `ValueKind::ALL` satisfying `pred`, as a set.
fn set_of(pred: fn(ValueKind) -> bool) -> BTreeSet<ValueKind> {
    ValueKind::ALL.into_iter().filter(|&k| pred(k)).collect()
}

fn set(members: &[ValueKind]) -> BTreeSet<ValueKind> {
    members.iter().copied().collect()
}

#[test]
fn is_scalar_is_the_cpp_filter_minus_long_double() {
    // `scalar_types = hana::filter(value_types, is_scalar)`
    // (`graph_properties.hh:78-80`). `std::is_scalar_v` is true for every
    // arithmetic type, so evaluated by hand over `value_types` it yields SIX
    // members -- `long double` among them.
    let cpp = set(&[
        ValueKind::Bool,
        ValueKind::I16,
        ValueKind::I32,
        ValueKind::I64,
        ValueKind::F64,
        ValueKind::LongDouble,
    ]);
    let ours = set_of(ValueKind::is_scalar);

    // THE ONE DOCUMENTED DIVERGENCE (`gt_core::design`, and the module docs on
    // `prop::value`). `long double` is carried as an opaque 16-byte payload
    // with no arithmetic -- `f128` is unstable, and reinterpreting an 80-bit
    // extended value as `f64` would change what a `.gt` file round-trips to.
    // The `Scalar` bound is what keeps it out of every arithmetic kernel at
    // compile time, so `is_scalar` has to agree with the bound, not with C++.
    let diverges: Vec<ValueKind> = cpp.difference(&ours).copied().collect();
    assert_eq!(
        diverges,
        vec![ValueKind::LongDouble],
        "is_scalar differs from graph_properties.hh:78-80 in some way other \
         than the single documented exclusion of `long double`"
    );
    assert!(
        ours.is_subset(&cpp),
        "is_scalar admits a member std::is_scalar_v rejects"
    );
    assert_eq!(ours.len(), 5);
    assert!(!ValueKind::LongDouble.is_scalar());

    // Nothing that is not an arithmetic C type is scalar: string, the seven
    // vectors and the Python object are all out, in both languages.
    assert!(!ValueKind::Str.is_scalar());
    assert!(!ValueKind::VecF64.is_scalar());
    assert!(!ValueKind::PyObject.is_scalar());
}

#[test]
fn is_integer_is_integer_types() {
    // `integer_types = hana::filter(value_types, is_integral)` (`:82-83`).
    // Note `uint8_t` is integral, so the member spelled "bool" is in the set;
    // `std::string` is not, and neither is any vector.
    assert_eq!(
        set_of(ValueKind::is_integer),
        set(&[
            ValueKind::Bool,
            ValueKind::I16,
            ValueKind::I32,
            ValueKind::I64
        ])
    );
    // `is_integral` implies `is_scalar` in C++, and the port keeps that:
    // no integer member was excluded the way `long double` was.
    assert!(set_of(ValueKind::is_integer).is_subset(&set_of(ValueKind::is_scalar)));
}

#[test]
fn is_floating_is_floating_types_and_is_not_a_subset_of_is_scalar() {
    // `floating_types = hana::filter(value_types, is_floating_point)`
    // (`:86-87`): exactly `double` and `long double`.
    assert_eq!(
        set_of(ValueKind::is_floating),
        set(&[ValueKind::F64, ValueKind::LongDouble])
    );

    // The consequence of the divergence above, spelled out so it cannot be
    // "fixed" by accident. In C++ `floating_types` is a subset of
    // `scalar_types`; here it is not, because `long double` is floating (it
    // is that C type, and `.gt` files say so) while not being arithmetic in
    // this port. Any code that assumed `is_floating => is_scalar` -- a
    // perfectly reasonable reading of the C++ -- is wrong, and this is where
    // it finds out.
    assert!(ValueKind::LongDouble.is_floating());
    assert!(!ValueKind::LongDouble.is_scalar());
    assert!(
        !set_of(ValueKind::is_floating).is_subset(&set_of(ValueKind::is_scalar)),
        "if this ever holds again, `long double` has been readmitted to the \
         arithmetic axis and its bytes are no longer opaque"
    );

    // Disjoint from the integers, as in C++.
    assert!(set_of(ValueKind::is_floating).is_disjoint(&set_of(ValueKind::is_integer)));
}

#[test]
fn is_vector_and_elem_are_vector_types() {
    // `vector_types = transform(append(scalar_types, type<string>),
    //                           t -> type<vector<t>>)` (`:90-93`).
    // `scalar_types` there is the C++ six (long double included), so this is
    // seven members -- and `vector<long double>` really is one of them, which
    // is why the port carries `Vec<LongDouble>` rather than dropping it.
    let cpp_vectors = set(&[
        ValueKind::VecBool,
        ValueKind::VecI16,
        ValueKind::VecI32,
        ValueKind::VecI64,
        ValueKind::VecF64,
        ValueKind::VecLongDouble,
        ValueKind::VecStr,
    ]);
    assert_eq!(set_of(ValueKind::is_vector), cpp_vectors);
    assert_eq!(cpp_vectors.len(), 7);

    for k in ValueKind::ALL {
        match k.elem() {
            Some(e) => {
                assert!(k.is_vector(), "{k:?} has an element but is not a vector");
                // The `transform` is visible in the *name*: the C++ spelling
                // of `vector<T>` is "vector<" + the spelling of T + ">", and
                // `_type_alias`/`_python_type` (`__init__.py:224, :243`) parse
                // it back out with exactly that regex. This ties `elem` and
                // `name` together so neither can drift alone.
                assert_eq!(
                    k.name(),
                    format!("vector<{}>", e.name()),
                    "{k:?}'s name is not vector<{}>",
                    e.name()
                );
                // No vector of vectors: `transform` is applied once.
                assert!(!e.is_vector(), "{k:?} has a vector element type");
                assert_eq!(e.elem(), None);
            }
            None => {
                assert!(!k.is_vector());
                // The non-vector members are exactly scalars + long double +
                // string + the Python object.
                assert!(
                    k.is_scalar()
                        || k == ValueKind::LongDouble
                        || k == ValueKind::Str
                        || k == ValueKind::PyObject,
                    "{k:?} is neither a vector nor an accounted-for leaf"
                );
            }
        }
    }

    // `append(scalar_types, string)` is what puts `vector<string>` in the set
    // while leaving `python::object` out: there is no `vector<object>`
    // member, and a property map of lists of Python objects is a
    // `python::object` map instead.
    assert!(!ValueKind::PyObject.is_vector());
    assert_eq!(ValueKind::PyObject.elem(), None);
}

// ===========================================================================
// 3. The types behind the members
// ===========================================================================

/// `<T as PropValue>::KIND` and `<T as PropValue>::Elem`, checked against the
/// member the C++ tuple assigns that type.
fn assert_member<T: PropValue, E: PropValue>(kind: ValueKind) {
    assert_eq!(
        T::KIND,
        kind,
        "{} has the wrong KIND",
        std::any::type_name::<T>()
    );
    assert_eq!(
        TypeId::of::<T::Elem>(),
        TypeId::of::<E>(),
        "{}::Elem is not {}",
        std::any::type_name::<T>(),
        std::any::type_name::<E>()
    );
    // `Elem`'s own KIND is the member `elem()` names -- the type-level
    // projection and the value-level one agree. This is the property that
    // lets `velem_dprop_t` (`dispatch.hh:344`) disappear entirely.
    assert_eq!(
        Some(<T::Elem as PropValue>::KIND),
        kind.elem().or(Some(kind)),
        "{}'s element projection disagrees with ValueKind::elem",
        std::any::type_name::<T>()
    );
}

#[test]
fn each_rust_type_carries_its_cpp_member() {
    assert_member::<u8, u8>(ValueKind::Bool);
    assert_member::<i16, i16>(ValueKind::I16);
    assert_member::<i32, i32>(ValueKind::I32);
    assert_member::<i64, i64>(ValueKind::I64);
    assert_member::<f64, f64>(ValueKind::F64);
    assert_member::<LongDouble, LongDouble>(ValueKind::LongDouble);
    assert_member::<String, String>(ValueKind::Str);
    assert_member::<Vec<u8>, u8>(ValueKind::VecBool);
    assert_member::<Vec<i16>, i16>(ValueKind::VecI16);
    assert_member::<Vec<i32>, i32>(ValueKind::VecI32);
    assert_member::<Vec<i64>, i64>(ValueKind::VecI64);
    assert_member::<Vec<f64>, f64>(ValueKind::VecF64);
    assert_member::<Vec<LongDouble>, LongDouble>(ValueKind::VecLongDouble);
    assert_member::<Vec<String>, String>(ValueKind::VecStr);
}

fn assert_zeroed<T: Zeroed + PartialEq + std::fmt::Debug>(expected: T) {
    assert_eq!(T::zero(), expected);
}

fn assert_gil_free<T: GilFree>() {}
fn assert_scalar<T: Scalar>() {}

#[test]
fn the_fourteen_non_python_members_are_zeroed_and_gil_free() {
    assert_zeroed(0u8);
    assert_zeroed(0i16);
    assert_zeroed(0i32);
    assert_zeroed(0i64);
    assert_zeroed(0.0f64);
    assert_zeroed(LongDouble([0; 16]));
    assert_zeroed(String::new());
    assert_zeroed(Vec::<u8>::new());
    assert_zeroed(Vec::<i16>::new());
    assert_zeroed(Vec::<i32>::new());
    assert_zeroed(Vec::<i64>::new());
    assert_zeroed(Vec::<f64>::new());
    assert_zeroed(Vec::<LongDouble>::new());
    assert_zeroed(Vec::<String>::new());

    assert_gil_free::<u8>();
    assert_gil_free::<i16>();
    assert_gil_free::<i32>();
    assert_gil_free::<i64>();
    assert_gil_free::<f64>();
    assert_gil_free::<LongDouble>();
    assert_gil_free::<String>();
    assert_gil_free::<Vec<u8>>();
    assert_gil_free::<Vec<String>>();
    assert_gil_free::<Vec<LongDouble>>();

    // `Scalar` is the arithmetic axis: the five, and not `LongDouble`.
    // The negative half of that claim is a compile error, so it lives in
    // `tests/ui/u08_value_subset_violates_kernel_bound.rs`.
    assert_scalar::<u8>();
    assert_scalar::<i16>();
    assert_scalar::<i32>();
    assert_scalar::<i64>();
    assert_scalar::<f64>();
}

#[test]
fn to_f64_widens_i64_which_into_f64_cannot() {
    // The reason `ToF64` exists rather than `Into<f64>`: there is no
    // `impl From<i64> for f64`, and `int64_t` is graph-tool's most-used
    // scalar property member. A `Into<f64>`-bounded design silently drops it.
    assert_eq!(255u8.to_f64(), 255.0);
    assert_eq!((-32768i16).to_f64(), -32768.0);
    assert_eq!(i32::MIN.to_f64(), -2147483648.0);
    assert_eq!(1.5f64.to_f64(), 1.5);

    // Exact up to 2^53, lossy above it. Accepted, and asserted so the loss is
    // a documented property rather than a surprise: `int64_t` property maps
    // reach `double`-typed kernels all through graph-tool too, so this is the
    // C++ behaviour being ported, not a new approximation being introduced.
    assert_eq!((1i64 << 53).to_f64(), 9007199254740992.0);
    assert_eq!(
        ((1i64 << 53) + 1).to_f64(),
        9007199254740992.0,
        "2^53 + 1 is not representable and must round down to 2^53"
    );
    assert_eq!(
        i64::MAX.to_f64(),
        9223372036854775808.0,
        "i64::MAX rounds *up*, to a value one greater than itself"
    );
    assert_eq!((i64::MAX - 1).to_f64(), i64::MAX.to_f64());
}

// ===========================================================================
// 4. long double survives as bytes
// ===========================================================================

/// The x86-64 System V encoding of `1.0L`: a 64-bit significand with its
/// explicit integer bit set, a 15-bit biased exponent of 0x3FFF, and six bytes
/// of padding the ABI leaves *unspecified*. The padding here is deliberately
/// not zero: graph-tool writes `sizeof(long double)` bytes straight into a
/// `.gt` file, so whatever the producer left in those bytes is what a
/// round-trip has to reproduce.
const ONE_L: LongDouble = LongDouble([
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // significand, LE
    0xFF, 0x3F, // sign + biased exponent, LE
    0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, // ABI padding, unspecified
]);

#[test]
fn long_double_is_sixteen_bytes() {
    assert_eq!(size_of::<LongDouble>(), 16);
    // `#[repr(transparent)]` over `[u8; 16]`, so a `Vec<LongDouble>` has the
    // same element stride as the C++ `vector<long double>` it was read from.
    assert_eq!(size_of::<[LongDouble; 4]>(), 64);
    assert_eq!(align_of::<LongDouble>(), 1);
}

#[test]
fn a_long_double_round_trips_through_a_dense_prop_unchanged() {
    let g = GraphId::fresh();
    let zero = LongDouble::zero();
    let mut map: DenseProp<LongDouble, VertexTag> =
        DenseProp::from_vec(g, vec![zero, ONE_L, zero, ONE_L]);

    let v1 = Id::<VertexTag>::from_index(1);
    let v2 = Id::<VertexTag>::from_index(2);

    // Read back through the property-map interface.
    assert_eq!(*map.get_ref(v1), ONE_L);
    assert_eq!(map.get_ref(v1).0, ONE_L.0, "the bytes, one for one");

    // Write through it.
    map.put(v2, ONE_L);
    assert_eq!(*map.get_ref(v2), ONE_L);

    // And through the lvalue path, which is what a kernel gets.
    map.at_mut(v2).0[15] = 0x7F;
    assert_eq!(map.get_ref(v2).0[15], 0x7F);
    assert_eq!(
        &map.get_ref(v2).0[..15],
        &ONE_L.0[..15],
        "an lvalue write to one byte disturbed the others"
    );

    // The bulk path -- the one every kernel actually uses -- sees the same
    // bytes, and nothing has normalised the ABI padding away.
    let bytes: Vec<[u8; 16]> = map.as_slice().iter().map(|d| d.0).collect();
    assert_eq!(bytes[0], [0u8; 16]);
    assert_eq!(bytes[1], ONE_L.0);
    assert_eq!(bytes[1][10..], [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02]);
    assert_eq!(map.len(), 4);
    assert_eq!(map.graph(), g);
}

// ===========================================================================
// 5. The negative guarantees
// ===========================================================================

/// `gt_core::design` §6 quotes both of these diagnostics. Neither has a runtime
/// representation, so neither can be asserted any other way — and both
/// regress silently: dropping the generated `const` assertion turns the first
/// into a *runtime* `DispatchError` that names the wrong type, and widening a
/// kernel bound turns the second into a successful build of a subset whose
/// kernel cannot run.
#[test]
fn value_subset_rejects_transposed_rows_and_bound_violations() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/u08_*.rs");
}
