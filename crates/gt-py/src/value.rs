//! The shipped value subsets, and the coverage guarantee.
//!
//! Each invocation of
//! [`value_subset!`](gt_core::value_subset) is the single source of truth for
//! one dispatch axis: its member list, its accepted-kind array, its
//! narrowing, its kernel trait *with the bound the kernel body needs*, and its
//! exhaustive match, all from one token list.
//!
//! The axes mirror `graph_properties.hh:78-104` and `:166-231`. The one
//! deliberate divergence is that `long double` is not in
//! [`ScalarV`](ScalarV): it has no arithmetic in this port, so the
//! [`Scalar`](gt_core::prop::Scalar) bound excludes it -- which is the
//! mechanism working as intended rather than a gap.

use gt_core::prop::{GilFree, PropValue, Scalar, ValueKind};
use gt_core::value_subset;

value_subset!(
    /// Arithmetic scalars: `{uint8_t, int16_t, int32_t, int64_t, double}`.
    pub ScalarV, ScalarKernel, "value", Scalar,
    { Bool => u8, I16 => i16, I32 => i32, I64 => i64, F64 => f64 }
);

value_subset!(
    /// Integral scalars: `integer_types` (`graph_properties.hh:83`).
    pub IntegerV, IntegerKernel, "value", Scalar,
    { Bool => u8, I16 => i16, I32 => i32, I64 => i64 }
);

value_subset!(
    /// Floating scalars actually usable in arithmetic. `long double` is
    /// excluded; see the module docs.
    pub FloatingV, FloatingKernel, "value", Scalar,
    { F64 => f64 }
);

value_subset!(
    /// Every member except the Python one: the 14 that may cross a thread.
    pub GilFreeV, GilFreeKernel, "value", GilFree,
    {
        Bool => u8, I16 => i16, I32 => i32, I64 => i64, F64 => f64,
        LongDouble => gt_core::prop::LongDouble,
        Str => String,
        VecBool => Vec<u8>, VecI16 => Vec<i16>, VecI32 => Vec<i32>,
        VecI64 => Vec<i64>, VecF64 => Vec<f64>,
        VecLongDouble => Vec<gt_core::prop::LongDouble>,
        VecStr => Vec<String>,
    }
);

value_subset!(
    /// The whole universe, including the Python member. Kernels over this
    /// axis are serial: [`PyValue`](gt_core::prop::PyValue) is `!Send`.
    pub AnyV, AnyKernel, "value", PropValue,
    {
        Bool => u8, I16 => i16, I32 => i32, I64 => i64, F64 => f64,
        LongDouble => gt_core::prop::LongDouble,
        Str => String,
        VecBool => Vec<u8>, VecI16 => Vec<i16>, VecI32 => Vec<i32>,
        VecI64 => Vec<i64>, VecF64 => Vec<f64>,
        VecLongDouble => Vec<gt_core::prop::LongDouble>,
        VecStr => Vec<String>,
        PyObject => gt_core::prop::PyValue,
    }
);

/// Every member reachable from at least one shipped subset.
///
/// This is the mitigation for the one residual hole in the dispatch story.
/// A subset's `narrow` necessarily ends in a `_ => Err` arm, so adding a
/// sixteenth [`ValueKind`] would not break the build at every call site; it
/// would make that member unsupported at runtime wherever no subset lists it.
/// The test below turns that into a build failure at the one place it can be.
pub const fn covered(kind: ValueKind) -> bool {
    matches!(
        kind,
        ValueKind::Bool
            | ValueKind::I16
            | ValueKind::I32
            | ValueKind::I64
            | ValueKind::F64
            | ValueKind::LongDouble
            | ValueKind::Str
            | ValueKind::VecBool
            | ValueKind::VecI16
            | ValueKind::VecI32
            | ValueKind::VecI64
            | ValueKind::VecF64
            | ValueKind::VecLongDouble
            | ValueKind::VecStr
            | ValueKind::PyObject
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_value_kind_is_reachable_from_some_subset() {
        for k in ValueKind::ALL {
            assert!(covered(k), "{k:?} ({}) is in no shipped subset", k.name());
            assert!(AnyV::narrow(k).is_ok(), "{k:?} is not in AnyV");
        }
    }

    #[test]
    fn narrowing_outside_a_subset_names_the_accepted_set() {
        let e = ScalarV::narrow(ValueKind::VecStr).unwrap_err();
        assert_eq!(e.offered, ValueKind::VecStr);
        assert_eq!(e.accepted, ScalarV::ACCEPTED);
    }
}
