//! DESIGN.md §7, mechanism 3 — the load-bearing one, and defect #30.
//!
//! `pyo3::Py<T>` is unconditionally `Send + Sync`
//! (`pyo3-0.22.6/src/instance.rs:943-944`), and both its `Clone` and its
//! `Drop` do refcount work without requiring a `Python<'py>` token. A property
//! map holding bare handles therefore reaches rayon with nothing at all to
//! stop it, which is what `graph_properties_copy.cc:35-42` does when both maps
//! are `python::object`: `is_python` is `false`, so the GIL is released *and*
//! `#pragma omp parallel` is enabled, and `Py_INCREF`/`Py_DECREF` then run
//! concurrently on unprotected refcounts.
//!
//! `PyValue` carries a `PhantomData<*mut ()>`, so it is `!Send` by
//! construction rather than by a marker trait somebody has to remember to
//! require:
//!
//! ```text
//! error[E0599]: the method `par_iter_mut` exists for struct `Vec<PyValue>`,
//!               but its trait bounds were not satisfied
//! ```

use gt_core::prop::PyValue;
use rayon::prelude::*;

fn main() {
    let mut map: Vec<PyValue> = Vec::new();
    map.par_iter_mut().for_each(|_v| {});
}
