//! The universe cannot be re-opened, so mechanism 3 cannot be side-stepped by
//! a sibling newtype.
//!
//! `gt_core::design` §7 records that one earlier design had exactly this hole: a
//! `struct RawPy(Py<PyAny>)` in the Python crate with
//! `impl PropValue for RawPy { type Mode = Par; }` compiled, and ran
//! `Py_INCREF` inside rayon. Here `PropValue`'s supertrait is a `Sealed` that
//! is private to gt-core, so the fifteen members are the only members there
//! are and `RawPy` cannot be a property value at all -- never mind which mode
//! it would claim.

use gt_core::prop::{PropValue, ValueKind};
use pyo3::{Py, PyAny};

struct RawPy(Py<PyAny>);

impl PropValue for RawPy {
    const KIND: ValueKind = ValueKind::PyObject;
    type Elem = RawPy;
}

fn main() {}
