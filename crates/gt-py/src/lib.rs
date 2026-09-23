//! # gt-py
//!
//! The Boost.Python replacement, and the only crate in the workspace that
//! names `pyo3`.
//!
//! ## What the GIL guarantee actually is ([DESIGN](gt_core::design) D7, D9)
//!
//! `graph_properties_copy.cc:35-42` -- and its four verbatim clones at `:69`,
//! `:104`, `:145`, and `graph_properties_copy.hh:36-40` -- computes
//!
//! ```text
//! bool is_python = (tgt != typeid(object) || src != typeid(object));
//! GILRelease gil(!is_python);
//! bool parallel = (num_vertices(g) > thresh && !is_python);
//! ```
//!
//! The connective is wrong and the sense is inverted. When **both** maps hold
//! `python::object`, `is_python` is false, so the GIL is *released* and the
//! loop *is* parallelised -- `Py_INCREF`/`Py_DECREF` across OpenMP threads
//! with no GIL held, in exactly the case that needs it most. And the reason
//! the author re-derived the predicate by hand at all is structural:
//! `gt_dispatch_args::gil_release` (`dispatch.hh:152`) is a single
//! compile-time constant per call site, while the value type is chosen by a
//! *runtime* `typeid` probe, so one flag cannot say "release iff this leaf has
//! no `python::object`".
//!
//! Three independent mechanisms replace it here, and the third is the one
//! that actually holds:
//!
//! 1. Nobody writes the predicate: [`gil::copy_prop`]'s where-clause derives
//!    it as a type-level lattice meet, resolved per monomorphised leaf.
//! 2. The `Par` strategy's bounds (`S: Sync`, `T: Send`) are unsatisfiable for
//!    the Python member, so even a *wrongly edited lattice* fails to compile.
//! 3. [`PyValue`](gt_core::prop::PyValue) is `!Send` **by construction**. This
//!    is the load-bearing one: a positive marker trait only guards the
//!    functions that remember to require it, and `pyo3::Py<T>` is
//!    unconditionally `Send + Sync`
//!    (`pyo3-0.22.6/src/instance.rs:943-944`), so a sibling newtype that
//!    forgets the marker otherwise sails straight into rayon.
//!
//! What Rust does **not** give for free, and the docs here do not claim:
//! `Python::with_gil` inside a rayon closure compiles, because the token is
//! acquired per worker and never crosses a thread boundary. The guarantee is
//! that the *race* is inexpressible, not that the parallel loop is -- and the
//! price is that a Python-valued property map is serialised, which is the
//! correct behaviour and what the C++ intended before the predicate was
//! inverted.

#![warn(missing_docs)]
// SKELETON: see gt-core's lib.rs.
#![allow(dead_code, unused_variables)]
// NOTE: no `forbid(unsafe_code)` here, and only here. pyo3's `#[pymodule]`,
// `#[pyclass]` and `#[pymethods]` macros expand to `unsafe extern "C"` FFI
// glue. No hand-written `unsafe` appears in this crate; see `gt_core::design`
// section 9.
//
// pyo3 0.22's macro expansion predates edition 2024's
// `unsafe_op_in_unsafe_fn`, so the generated bodies trip it. The allow covers
// generated code only and is removed when the workspace moves to a pyo3
// release built for edition 2024.
#![allow(unsafe_op_in_unsafe_fn)]

pub mod dispatch;
pub mod gil;
pub mod module;
pub mod value;

pub use dispatch::{
    AnyGraph, BidiGraphKernel, DynGraph, GraphKernel, ViewError, ViewKind,
};
pub use gil::{CopyStrategy, CopyValue, Meet, Mode, Par, Seq, copy_prop, detach};
