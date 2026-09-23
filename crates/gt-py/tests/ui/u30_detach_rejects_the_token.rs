//! `detach` excludes the token structurally — defect #32.
//!
//! `GILRelease(bool)` (`gil_release.hh:31-35`) saves the thread state only if
//! `PyGILState_Check()` says the GIL is already held, so on a thread that does
//! not hold it the object is a no-op that still *looks* like a release at the
//! call site. Every OpenMP worker is such a thread.
//!
//! `detach`'s `F: Send` says the closure may leave this thread, and a
//! `Python<'py>` may not: it carries a `PhantomData<NotSend>`. So a body that
//! wants the token is not a body that may be detached, and the two cannot be
//! confused.

use gt_py::gil::detach;
use pyo3::Python;

fn main() {
    Python::with_gil(|py| {
        // Capturing `py` is the mistake. It is also the only way to touch a
        // CPython object from inside, which is precisely why it must not
        // compile.
        detach(py, move || py.version().len())
    });
}
