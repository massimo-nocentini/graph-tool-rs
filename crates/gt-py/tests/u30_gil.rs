//! U30 — the GIL lattice, checked from outside the crate.
//!
//! Three kinds of evidence live here, and they are deliberately different
//! kinds, because the guarantee has three parts that no single kind can cover:
//!
//! 1. **Type-level witnesses.** `copy_prop::<f64, f64>` resolving to
//!    [`Par`](gt_py::Par) is not a fact about a run, it is a fact about a
//!    projection, so it is asserted by naming the projection and the expected
//!    type in a `const` — not by reading the source and not by timing
//!    anything. If the lattice is edited, this file stops compiling.
//! 2. **Observations from Python.** "The body calls `allow_threads`" is only
//!    interesting because a *Python thread gets to run*. So a Python thread is
//!    started, blocked on an event, and then watched: it must stay blocked
//!    across a `python::object -> python::object` copy and must get through
//!    during a `double -> double` one. That is precisely the truth table
//!    `graph_properties_copy.cc:35-42` gets backwards in both rows.
//! 3. **`trybuild` fixtures**, for the four diagnostics that have no runtime
//!    representation at all.
//!
//! ## Why the Python-facing tests share a mutex
//!
//! `cargo test` runs the functions in this file on several threads of one
//! process. A test that asserts "no other Python thread ran while I held the
//! token" is only sound if no *sibling test* released the token in the
//! meantime, and [`detach`](gt_py::detach) exists to do exactly that. The
//! mutex makes the observation window exclusive; it is not there to make a
//! flaky test pass.

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use gt_core::adj::AdjList;
use gt_core::error::PropError;
use gt_core::ids::VertexTag;
use gt_core::prop::{DenseProp, PyValue, VertexProp};
use gt_py::gil::{self, Allows, CopyStrategy, Meet, Mode, ModeFor, ModeOf, Par, Seq, copy_prop};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ===========================================================================
// 1. The lattice, as types
// ===========================================================================

/// Type identity, as a bound. `Is<B>` is inhabited for `A` exactly when
/// `A == B`, so `witness::<X, Y>()` is a compile-time `assert_eq!` on types.
///
/// This is the shape the acceptance criterion asks for: the resolution is
/// asserted, never inspected. `assert_eq!(std::any::type_name::<..>(), "Par")`
/// would be inspection, and would keep passing if `Par` were renamed to
/// something that no longer released anything.
trait Is<T: ?Sized> {}
impl<T: ?Sized> Is<T> for T {}
const fn witness<A: Is<B> + ?Sized, B: ?Sized>() {}

/// `Seq` absorbs, on all four corners.
const _: () = {
    witness::<<Par as Meet<Par>>::Out, Par>();
    witness::<<Par as Meet<Seq>>::Out, Seq>();
    witness::<<Seq as Meet<Par>>::Out, Seq>();
    witness::<<Seq as Meet<Seq>>::Out, Seq>();
};

/// The acceptance criterion itself: what `copy_prop` resolves to, per leaf.
///
/// The two middle lines are the rows `is_python = (tgt != object || src !=
/// object)` (`graph_properties_copy.cc:35-37`) answers wrongly: with one
/// Python side the C++ computes `is_python = true` and *keeps* the GIL, and
/// with **two** it computes `false` and releases it while running
/// `Py_INCREF` under `#pragma omp parallel`. Here the meet answers `Seq` for
/// all three.
const _: () = {
    witness::<ModeFor<f64, f64>, Par>();
    witness::<ModeFor<PyValue, f64>, Seq>();
    witness::<ModeFor<f64, PyValue>, Seq>();
    witness::<ModeFor<PyValue, PyValue>, Seq>();
};

/// Every one of the fourteen GIL-free members resolves to `Par` against every
/// other, and every one of them resolves to `Seq` against the Python member.
///
/// Spelled out rather than quantified, because that is the only form in which
/// a compiler can check it.
const _: () = {
    macro_rules! par_against_itself {
        ($($t:ty),+ $(,)?) => {$(
            witness::<<$t as ModeOf>::Mode, Par>();
            witness::<ModeFor<$t, $t>, Par>();
            witness::<ModeFor<$t, PyValue>, Seq>();
            witness::<ModeFor<PyValue, $t>, Seq>();
        )+};
    }
    par_against_itself!(
        u8,
        i16,
        i32,
        i64,
        f64,
        gt_core::prop::LongDouble,
        String,
        Vec<u8>,
        Vec<i16>,
        Vec<i32>,
        Vec<i64>,
        Vec<f64>,
        Vec<gt_core::prop::LongDouble>,
        Vec<String>,
    );
    witness::<<PyValue as ModeOf>::Mode, Seq>();
};

/// `Par` may only be given to a member that can cross a thread.
///
/// The positive half; `tests/ui/u30_par_mode_for_py_value.rs` pins the
/// negative half, which is the one that matters.
const _: () = {
    const fn allowed<M: Allows<V>, V: ?Sized>() {}
    allowed::<Par, f64>();
    allowed::<Seq, f64>();
    allowed::<Seq, PyValue>();
};

/// The strategy chosen for a leaf is the one that can actually serve it.
///
/// `Par: CopyStrategy<PyValue, PyValue, _>` is unnameable (`PyValue` is
/// neither `Sync` nor `Send`), so this is not merely "the meet says `Seq`" --
/// there is no other inhabitant to choose.
const _: () = {
    const fn strategy<M: CopyStrategy<S, T, VertexTag>, S, T>() {}
    strategy::<ModeFor<PyValue, PyValue>, PyValue, PyValue>();
    strategy::<ModeFor<f64, f64>, f64, f64>();
    strategy::<Seq, PyValue, PyValue>();
};

#[test]
fn the_mode_witness_is_a_constant_the_compiler_resolved() {
    // The `const` blocks above are the assertion; this reproduces the whole
    // four-row table as data so that a reader of a CI log sees the answer,
    // rather than having to trust that a silent build checked it.
    //
    // Compare against `graph_properties_copy.cc:35-42`, where the same four
    // rows are `is_python = (tgt != object || src != object)` and the last
    // one -- the `Py_INCREF` loop -- is the one that gets `#pragma omp
    // parallel`.
    let resolved: Vec<(&str, &str, bool)> = vec![
        (
            "double -> double",
            <ModeFor<f64, f64> as Mode>::NAME,
            <ModeFor<f64, f64> as Mode>::PARALLEL,
        ),
        (
            "double -> object",
            <ModeFor<f64, PyValue> as Mode>::NAME,
            <ModeFor<f64, PyValue> as Mode>::PARALLEL,
        ),
        (
            "object -> double",
            <ModeFor<PyValue, f64> as Mode>::NAME,
            <ModeFor<PyValue, f64> as Mode>::PARALLEL,
        ),
        (
            "object -> object",
            <ModeFor<PyValue, PyValue> as Mode>::NAME,
            <ModeFor<PyValue, PyValue> as Mode>::PARALLEL,
        ),
    ];
    assert_eq!(
        resolved,
        vec![
            ("double -> double", "Par", true),
            ("double -> object", "Seq", false),
            ("object -> double", "Seq", false),
            ("object -> object", "Seq", false),
        ]
    );
}

// ===========================================================================
// 2. Python-facing fixtures
// ===========================================================================

/// Serialises the observation window. See the module docs.
static GIL_OBSERVATION: Mutex<()> = Mutex::new(());

fn observation_window() -> MutexGuard<'static, ()> {
    pyo3::prepare_freethreaded_python();
    GIL_OBSERVATION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A Python thread parked on an event, plus the list it appends to once it is
/// released *and* gets the GIL.
///
/// `Thread.start()` itself waits for the worker to begin running, which the
/// worker cannot do without the GIL -- so by the time this returns the worker
/// is inside `ev.wait()`, which has released the GIL back to us. Calling
/// `ev.set()` makes it *runnable*; whether it actually runs is then purely a
/// question of whether this thread ever lets go of the GIL. That is the
/// property under test.
fn park_a_python_thread(py: Python<'_>) -> Bound<'_, PyDict> {
    // One dict, used as *globals*. A function defined by `exec` resolves its
    // free variables in the globals it was compiled against, never in the
    // exec's locals, so handing this in as `locals` would leave `_w` raising
    // `NameError` on a thread nobody is watching -- and the test would then
    // "prove" that the GIL was never released.
    let globals = PyDict::new_bound(py);
    py.run_bound(
        "import threading\n\
         ev = threading.Event()\n\
         done = []\n\
         def _w():\n\
         \x20   ev.wait()\n\
         \x20   done.append(1)\n\
         t = threading.Thread(target=_w, daemon=True)\n\
         t.start()\n",
        Some(&globals),
        None,
    )
    .expect("the fixture is plain threading");
    // The worker is now inside `ev.wait()`, which released the GIL back to
    // us. Setting the event makes it *runnable*; whether it runs is the
    // question.
    globals
        .get_item("ev")
        .unwrap()
        .unwrap()
        .call_method0("set")
        .expect("Event.set");
    globals
}

/// How many times the parked worker has got through.
fn worker_ran(locals: &Bound<'_, PyDict>) -> bool {
    locals
        .get_item("done")
        .unwrap()
        .unwrap()
        .len()
        .expect("list")
        > 0
}

/// A vertex map holding `n` distinct Python integers.
fn py_map(py: Python<'_>, g: &AdjList, n: usize) -> VertexProp<PyValue> {
    DenseProp::from_vec(
        g.graph_id(),
        (0..n).map(|i| PyValue::new(i.into_py(py))).collect(),
    )
}

/// A vertex map holding `n` references to `None`.
fn none_map(py: Python<'_>, g: &AdjList, n: usize) -> VertexProp<PyValue> {
    DenseProp::from_vec(g.graph_id(), (0..n).map(|_| PyValue::none(py)).collect())
}

// ===========================================================================
// 3. Defect #30 and #31 — the two rows, observed from Python
// ===========================================================================

/// `object -> object` must **not** release the GIL, and `double -> double`
/// must.
///
/// This is one test rather than two because the two halves share one parked
/// worker: the negative half establishes that the worker is genuinely blocked
/// on *our* token, and the positive half then shows the same worker getting
/// through. Split apart, the negative half would be unfalsifiable (a worker
/// that never runs for an unrelated reason passes it).
///
/// `graph_properties_copy.cc:35-42` gets both halves backwards: with `tgt` and
/// `src` both `python::object` its `is_python` is `false`, so `GILRelease
/// gil(!is_python)` releases and `#pragma omp parallel if (parallel)` runs the
/// `Py_INCREF` loop across threads; with both `double` its `is_python` is
/// `true`, so the GIL is held across the long loop for no reason at all.
#[test]
fn the_python_leaf_holds_the_gil_and_the_double_leaf_releases_it() {
    let _window = observation_window();
    Python::with_gil(|py| {
        let locals = park_a_python_thread(py);

        // --- Seq: python::object -> python::object -------------------------
        let g = AdjList::with_vertices(64);
        let src = py_map(py, &g, 64);
        let mut dst = none_map(py, &g, 64);

        let until = Instant::now() + Duration::from_millis(250);
        while Instant::now() < until {
            copy_prop(py, g.vertex_bound(), &src, &mut dst).expect("sized for its own graph");
        }
        assert!(
            !worker_ran(&locals),
            "a python::object -> python::object copy released the GIL: that is \
             defect #30, and the whole reason PyValue is !Send"
        );

        // --- Par: double -> double -----------------------------------------
        // Big enough to cross MIN_PAR_ITEMS (300, `openmp.cc:20`) so the
        // chunked path runs, and repeated so that a slow scheduler costs
        // iterations rather than correctness.
        let h = AdjList::with_vertices(4096);
        let fsrc: VertexProp<f64> = DenseProp::from_vec(
            h.graph_id(),
            (0..4096).map(|i| i as f64).collect::<Vec<_>>(),
        );
        let mut fdst: VertexProp<f64> = DenseProp::from_vec(h.graph_id(), vec![0.0; 4096]);

        let deadline = Instant::now() + Duration::from_secs(10);
        let mut released = false;
        while Instant::now() < deadline {
            copy_prop(py, h.vertex_bound(), &fsrc, &mut fdst).expect("sized for its own graph");
            if worker_ran(&locals) {
                released = true;
                break;
            }
        }
        assert!(
            released,
            "a double -> double copy never let a waiting Python thread run, so \
             Par::copy is not going through allow_threads: that is defect #31"
        );
        // And it copied while it was at it.
        assert_eq!(fdst.as_slice()[4095], 4095.0);
    });
}

/// [`detach`](gt_py::detach) on its own, without a property map in the way.
///
/// `GILRelease(bool)` (`gil_release.hh:31-35`) saves the thread state only
/// when `PyGILState_Check()` is already true, so on a worker that never held
/// the GIL it is a silent no-op -- defect #32. There is no analogous state
/// here to be in: `allow_threads` is called on a token, and a token is proof
/// the GIL is held.
#[test]
fn detach_lets_a_blocked_python_thread_through() {
    let _window = observation_window();
    Python::with_gil(|py| {
        let locals = park_a_python_thread(py);

        // Holding the token: the worker is runnable and cannot run.
        std::thread::sleep(Duration::from_millis(150));
        assert!(
            !worker_ran(&locals),
            "a Python thread ran while this thread held the token"
        );

        let deadline = Instant::now() + Duration::from_secs(10);
        let mut ran = false;
        while Instant::now() < deadline {
            gil::detach(py, || std::thread::sleep(Duration::from_millis(20)));
            if worker_ran(&locals) {
                ran = true;
                break;
            }
        }
        assert!(ran, "detach did not release the GIL");
    });
}

// ===========================================================================
// 4. What the serial strategy actually does
// ===========================================================================

/// `convert`'s identity arm for the Python member (`value_convert.hh:78-81`)
/// returns `(const To&)v` -- a new reference to the *same* object, not a copy
/// of it. So must this, and it must do the incref under the token it was
/// handed rather than one it re-acquired.
#[test]
fn copying_a_python_map_shares_the_objects_rather_than_duplicating_them() {
    let _window = observation_window();
    Python::with_gil(|py| {
        let g = AdjList::with_vertices(5);
        let src = py_map(py, &g, 5);
        let mut dst = none_map(py, &g, 5);

        copy_prop(py, g.vertex_bound(), &src, &mut dst).expect("same graph, same length");

        for (i, (s, d)) in src.as_slice().iter().zip(dst.as_slice()).enumerate() {
            assert!(
                d.bind(py).is(&s.bind(py)),
                "slot {i} holds a different object than the source"
            );
            assert_eq!(d.bind(py).extract::<usize>().unwrap(), i);
        }
    });
}

/// The strategy is reachable directly, not only through the meet, and the
/// direct call is the same code.
#[test]
fn the_serial_strategy_can_be_named_and_run_on_its_own() {
    let _window = observation_window();
    Python::with_gil(|py| {
        let g = AdjList::with_vertices(3);
        let src = py_map(py, &g, 3);
        let mut dst = none_map(py, &g, 3);
        <Seq as CopyStrategy<PyValue, PyValue, VertexTag>>::copy(
            py,
            g.vertex_bound(),
            &src,
            &mut dst,
        )
        .expect("same graph, same length");
        assert!(dst.as_slice()[2].bind(py).is(&src.as_slice()[2].bind(py)));
    });
}

// ===========================================================================
// 5. The bound is checked on both maps, before anything is written
// ===========================================================================

/// A map that is too short is refused, not silently extended and not written
/// past.
///
/// This is the one place the Python member forces the port's hand and the
/// forcing is the right way round: `PyValue` has no context-free default, so
/// `DenseProp::sized_for` does not exist for it and the sizing has to happen
/// at the entry point, under the token (DESIGN.md §5). graph-tool's
/// `get_unchecked(size = 0)` (`dispatch.hh:171-177`) hands the kernel the
/// short map instead.
#[test]
fn an_undersized_map_on_either_side_is_refused() {
    let _window = observation_window();
    Python::with_gil(|py| {
        let g = AdjList::with_vertices(5);

        let src = py_map(py, &g, 5);
        let mut short = none_map(py, &g, 3);
        assert_eq!(
            copy_prop(py, g.vertex_bound(), &src, &mut short).unwrap_err(),
            PropError::Undersized { have: 3, need: 5 }
        );

        let short_src = py_map(py, &g, 2);
        let mut dst = none_map(py, &g, 5);
        assert_eq!(
            copy_prop(py, g.vertex_bound(), &short_src, &mut dst).unwrap_err(),
            PropError::Undersized { have: 2, need: 5 }
        );

        // A map longer than the bound is fine: the bound names the prefix.
        let long_src = py_map(py, &g, 9);
        assert!(copy_prop(py, g.vertex_bound(), &long_src, &mut dst).is_ok());
        assert!(
            dst.as_slice()[4]
                .bind(py)
                .is(&long_src.as_slice()[4].bind(py))
        );
    });
}

/// Defect #8, at the copy boundary: identity is checked before length, so two
/// graphs of the same size are still not interchangeable.
///
/// `graph_tool/__init__.py:3200` compares filtered *counts*, which agree in
/// exactly the case that makes `graph_copy.cc:72`'s write land in the wrong
/// map without a diagnostic.
#[test]
fn a_map_belonging_to_another_graph_is_refused_even_at_equal_length() {
    let _window = observation_window();
    Python::with_gil(|py| {
        let a = AdjList::with_vertices(4);
        let b = AdjList::with_vertices(4);
        assert_ne!(a.graph_id(), b.graph_id());

        let src = py_map(py, &a, 4);
        let mut foreign = none_map(py, &b, 4);
        assert_eq!(
            copy_prop(py, a.vertex_bound(), &src, &mut foreign).unwrap_err(),
            PropError::WrongGraph {
                owner: b.graph_id().get(),
                expected: a.graph_id().get(),
            }
        );
        // Nothing was written before the check.
        assert!(foreign.as_slice()[0].bind(py).is_none());
    });
}

/// An empty bound is a legal bound, and copies nothing.
#[test]
fn an_empty_graph_copies_nothing_and_succeeds() {
    let _window = observation_window();
    Python::with_gil(|py| {
        let g = AdjList::new();
        let src: VertexProp<PyValue> = DenseProp::new(g.graph_id());
        let mut dst: VertexProp<PyValue> = DenseProp::new(g.graph_id());
        assert!(copy_prop(py, g.vertex_bound(), &src, &mut dst).is_ok());
        assert_eq!(dst.len(), 0);
    });
}

// ===========================================================================
// 6. The parallel strategy: same answer at any width
// ===========================================================================

/// The chunked path and the short path must agree, value for value.
///
/// 4096 crosses `MIN_PAR_ITEMS` (300, `openmp.cc:20`) and is sixteen
/// `PAR_GRAIN` chunks; 7 does not cross it and runs serially inside
/// `allow_threads`. Both are the same `copy_prop` call.
#[test]
fn the_parallel_and_serial_paths_of_par_produce_the_same_map() {
    let _window = observation_window();
    Python::with_gil(|py| {
        for n in [0usize, 1, 7, 299, 300, 301, 4096] {
            let g = AdjList::with_vertices(n);
            let src: VertexProp<i64> = DenseProp::from_vec(
                g.graph_id(),
                (0..n).map(|i| (i as i64) * 3 - 1).collect::<Vec<_>>(),
            );
            let mut dst: VertexProp<i64> = DenseProp::from_vec(g.graph_id(), vec![0; n]);
            copy_prop(py, g.vertex_bound(), &src, &mut dst).expect("same graph");
            assert_eq!(dst.as_slice(), src.as_slice(), "n = {n}");
        }
    });
}

/// A failing conversion does not abandon the rest of the cover, and the set of
/// work performed is the same at every width.
///
/// `parallel_loop_no_spawn<true>` (`parallel_util.hh:399-437`) sets a
/// thread-private `skip` flag on the first throw, so *which* later iterations
/// run is the OpenMP schedule; the exception itself is then stored into a
/// `std::exception_ptr` shared by every thread in the region
/// (`:438-446`), which is a data race on a refcount. Here the error is a
/// return value and the cover is executed in full, so the resulting map is a
/// function of the input alone.
#[test]
fn a_failing_conversion_still_writes_every_convertible_slot() {
    let _window = observation_window();
    Python::with_gil(|py| {
        for n in [8usize, 4096] {
            let g = AdjList::with_vertices(n);
            // Every third entry is not a number.
            let src: VertexProp<String> = DenseProp::from_vec(
                g.graph_id(),
                (0..n)
                    .map(|i| {
                        if i % 3 == 2 {
                            "not a number".to_string()
                        } else {
                            i.to_string()
                        }
                    })
                    .collect::<Vec<_>>(),
            );
            let mut dst: VertexProp<i64> = DenseProp::from_vec(g.graph_id(), vec![-1; n]);
            let err = copy_prop(py, g.vertex_bound(), &src, &mut dst).unwrap_err();
            assert!(
                matches!(err, PropError::NoConversion { .. }),
                "n = {n}: {err:?}"
            );
            for i in 0..n {
                let expected = if i % 3 == 2 { -1 } else { i as i64 };
                assert_eq!(dst.as_slice()[i], expected, "n = {n}, slot {i}");
            }
        }
    });
}

// ===========================================================================
// 6b. The mixed arms of the lattice
// ===========================================================================

/// `convert`'s two Python-crossing arms (`value_convert.hh:82-107`), end to
/// end through [`copy_prop`].
///
/// These were the hole U30 left behind: the lattice *resolved* the mixed pairs
/// to [`Seq`] (the `const` witnesses above check that much) but no
/// `CopyValue<f64> for PyValue` or `CopyValue<PyValue> for f64` existed, so
/// `copy_prop::<PyValue, f64, _>` was an `E0277` rather than a call. gt-core's
/// `ConvertFrom` now carries both arms under `feature = "python"`, so the
/// bridge in `gil.rs` is a bound relaxation on the `TokenFree` blanket plus one
/// impl for the `PyValue` target, and the column is reachable.
///
/// The direction that matters for fidelity is `python::object -> To`:
/// `value_convert.hh:86-107` is `extract<To>` with an elementwise fallback for
/// vector targets, i.e. it is allowed to *fail per value*, which is why the
/// failing half below asserts `Err` rather than a panic.
#[test]
fn the_mixed_python_arms_are_reachable_in_both_directions() {
    let _window = observation_window();
    Python::with_gil(|py| {
        let g = AdjList::with_vertices(4);

        // `is_same_v<From, boost::python::object>`: extract<double>.
        let src: VertexProp<PyValue> = DenseProp::from_vec(
            g.graph_id(),
            (0..4)
                .map(|i| PyValue::new(pyo3::types::PyFloat::new_bound(py, i as f64).into_any().unbind()))
                .collect(),
        );
        let mut dst: VertexProp<f64> = DenseProp::from_vec(g.graph_id(), vec![0.0; 4]);
        copy_prop(py, g.vertex_bound(), &src, &mut dst).expect("python::object -> double");
        assert_eq!(dst.as_slice(), &[0.0, 1.0, 2.0, 3.0]);

        // `is_same_v<To, boost::python::object>`: object(v), which cannot fail.
        let isrc: VertexProp<i64> = DenseProp::from_vec(g.graph_id(), vec![7, 8, 9, 10]);
        let mut pdst: VertexProp<PyValue> =
            DenseProp::from_vec(g.graph_id(), (0..4).map(|_| PyValue::none(py)).collect());
        copy_prop(py, g.vertex_bound(), &isrc, &mut pdst).expect("int64_t -> python::object");
        let got: Vec<i64> = pdst
            .as_slice()
            .iter()
            .map(|v| v.bind(py).extract::<i64>().expect("round trip"))
            .collect();
        assert_eq!(got, vec![7, 8, 9, 10]);

        // A value `extract<To>` cannot take is an `Err`, not a panic and not a
        // silent zero -- and, per `Seq::copy`, the convertible slots are still
        // written.
        let bad: VertexProp<PyValue> = DenseProp::from_vec(
            g.graph_id(),
            vec![
                PyValue::new(pyo3::types::PyFloat::new_bound(py, 1.0).into_any().unbind()),
                PyValue::new(pyo3::types::PyString::new_bound(py, "not a number").into_any().unbind()),
                PyValue::new(pyo3::types::PyFloat::new_bound(py, 3.0).into_any().unbind()),
                PyValue::new(pyo3::types::PyFloat::new_bound(py, 4.0).into_any().unbind()),
            ],
        );
        let mut out: VertexProp<f64> = DenseProp::from_vec(g.graph_id(), vec![-1.0; 4]);
        assert!(
            copy_prop(py, g.vertex_bound(), &bad, &mut out).is_err(),
            "a value extract<double> cannot take must be an Err"
        );
        assert_eq!(
            [out.as_slice()[0], out.as_slice()[2], out.as_slice()[3]],
            [1.0, 3.0, 4.0],
            "the full cover is still attempted: that is the half \
             parallel_util.hh:399-437 abandons on its `skip` flag"
        );
    });
}

/// Both mixed pairs still meet at [`Seq`], which is what keeps a `PyValue`
/// away from rayon now that the arms exist.
const _: () = {
    witness::<ModeFor<PyValue, f64>, Seq>();
    witness::<ModeFor<f64, PyValue>, Seq>();
};

// ===========================================================================
// 7. The negative guarantees
// ===========================================================================

/// Four diagnostics, none of which has a runtime representation.
///
/// DESIGN.md §7 claims each of them; a claim about a compiler error that is
/// not pinned to the compiler's actual output regresses the first time
/// somebody adds a blanket impl or relaxes a bound, and regresses *silently*,
/// because the suite goes on passing while checking nothing.
#[test]
fn the_lattice_cannot_be_edited_or_bypassed() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/u30_*.rs");
}
