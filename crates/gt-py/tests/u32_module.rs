//! U32 — the extension module.
//!
//! These run the *same* code an `import graph_tool._rust` would: `register`
//! populates a real `PyModule`, and every assertion below goes through the
//! Python object protocol (`__getitem__`, `__setitem__`, bound method calls)
//! rather than through the Rust structs behind them. The only difference from
//! a `python3 -c` smoke test is that the interpreter is embedded, which keeps
//! the whole thing inside `cargo test`.
//!
//! What they are checking, in order of what DESIGN.md cares about:
//!
//! * **auto-grow on read** (§5). `fast_vector_property_map.hh:132-136` does
//!   `soft_reserve(i + 1)` inside the checked `operator[]`, so `g.vp.x[v]` on
//!   a never-written map returns the default. The port has no auto-grow
//!   anywhere except [`DenseProp::sized_for`], so the boundary has to call it
//!   on read-only maps too, and that is the first group of tests.
//! * **the edge index *range*, not the count** (`__init__.py:369`). Sizing an
//!   edge map to `num_edges` after a removal is an out-of-bounds write in the
//!   C++ and an `Undersized` error here.
//! * **`type_names[]` exactly** (`graph_properties.hh:72-76`), with the alias
//!   table left where it belongs, above this boundary.
//! * **errors are `PyErr`s, never panics** (§10). Nothing in the module
//!   unwinds on a caller error, so `panic = "abort"` changes nothing.

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};
use std::sync::Once;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// gt-py does not enable pyo3's `auto-initialize`, so the interpreter is
/// started explicitly. Idempotent, and safe to call from every test.
fn interpreter() {
    static START: Once = Once::new();
    START.call_once(|| {
        pyo3::prepare_freethreaded_python();
    });
}

/// A freshly registered `graph_tool._rust`.
fn module(py: Python<'_>) -> Bound<'_, PyModule> {
    let m = PyModule::new_bound(py, "graph_tool._rust").expect("module");
    gt_py::module::register(&m).expect("register");
    m
}

/// Run `f` with the module in hand.
fn with_module<F, R>(f: F) -> R
where
    F: for<'py> FnOnce(Python<'py>, &Bound<'py, PyModule>) -> R,
{
    interpreter();
    Python::with_gil(|py| {
        let m = module(py);
        f(py, &m)
    })
}

/// An empty `Graph`.
fn graph<'py>(m: &Bound<'py, PyModule>) -> Bound<'py, PyAny> {
    m.getattr("Graph").unwrap().call0().unwrap()
}

/// A graph with `n` isolated vertices.
fn graph_with<'py>(m: &Bound<'py, PyModule>, n: usize) -> Bound<'py, PyAny> {
    let g = graph(m);
    for i in 0..n {
        let v: usize = g
            .call_method0("add_vertex")
            .unwrap()
            .extract()
            .expect("add_vertex returns an index");
        assert_eq!(v, i, "vertices are dense and appended in order");
    }
    g
}

fn vprop<'py>(g: &Bound<'py, PyAny>, ty: &str) -> Bound<'py, PyAny> {
    g.call_method1("new_vertex_property", (ty,)).unwrap()
}

fn eprop<'py>(g: &Bound<'py, PyAny>, ty: &str) -> Bound<'py, PyAny> {
    g.call_method1("new_edge_property", (ty,)).unwrap()
}

fn is_value_error(py: Python<'_>, e: &PyErr) -> bool {
    e.is_instance_of::<PyValueError>(py)
}

// ---------------------------------------------------------------------------
// 1. Graph mutation
// ---------------------------------------------------------------------------

#[test]
fn counts_follow_the_adjacency() {
    with_module(|_py, m| {
        let g = graph_with(m, 4);
        assert_eq!(
            g.call_method0("num_vertices")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            4
        );
        assert_eq!(
            g.call_method0("num_edges")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            0
        );

        let e0: usize = g
            .call_method1("add_edge", (0, 1))
            .unwrap()
            .extract()
            .unwrap();
        let e1: usize = g
            .call_method1("add_edge", (1, 2))
            .unwrap()
            .extract()
            .unwrap();
        let e2: usize = g
            .call_method1("add_edge", (2, 2))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!((e0, e1, e2), (0, 1, 2), "edge indices are dense");
        assert_eq!(
            g.call_method0("num_edges")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            3
        );

        // A self-loop removed by identity. `remove_edge(e, g)`
        // (`graph_adjacency.hh:1252-1312`) reads both end positions up front
        // through two references that alias when `s == t`, then trusts an
        // `_epos` value the first splice rewrote; here the count is derived
        // from the allocator and cannot drift.
        g.call_method1("remove_edge", (e2,)).unwrap();
        assert_eq!(
            g.call_method0("num_edges")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            2
        );
    });
}

#[test]
fn a_bad_endpoint_is_a_value_error_and_not_a_panic() {
    with_module(|py, m| {
        let g = graph_with(m, 2);
        let err = g.call_method1("add_edge", (0, 9)).unwrap_err();
        assert!(is_value_error(py, &err), "{err}");
        // The failed call took no edge index: `add_edge` checks both endpoints
        // before `EdgeIds::alloc`, where `:1190` allocates first.
        assert_eq!(
            g.call_method0("edge_index_range")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            0
        );

        let e: usize = g
            .call_method1("add_edge", (0, 1))
            .unwrap()
            .extract()
            .unwrap();
        g.call_method1("remove_edge", (e,)).unwrap();
        let err = g.call_method1("remove_edge", (e,)).unwrap_err();
        assert!(is_value_error(py, &err), "{err}");
    });
}

#[test]
fn the_edge_index_range_outlives_the_edges() {
    with_module(|_py, m| {
        let g = graph_with(m, 3);
        for pair in [(0, 1), (1, 2), (0, 2)] {
            g.call_method1("add_edge", pair).unwrap();
        }
        g.call_method1("remove_edge", (1,)).unwrap();

        // `_get_any` sizes an edge map to `g.edge_index_range`
        // (`__init__.py:369`) and *not* to `num_edges`, because the index
        // space is sparse after a removal. A map sized to the count would be
        // indexed past its end at index 2.
        assert_eq!(
            g.call_method0("num_edges")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            2
        );
        assert_eq!(
            g.call_method0("edge_index_range")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            3
        );

        let ep = eprop(&g, "int64_t");
        ep.set_item(2, 11i64).unwrap();
        assert_eq!(
            ep.call_method0("allocated")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            3
        );
        // The freed slot still reads, and reads as the default.
        assert_eq!(ep.get_item(1).unwrap().extract::<i64>().unwrap(), 0);
    });
}

// ---------------------------------------------------------------------------
// 2. Auto-grow on read
// ---------------------------------------------------------------------------

#[test]
fn a_never_written_key_reads_back_as_the_default() {
    with_module(|_py, m| {
        let g = graph_with(m, 5);
        let vp = vprop(&g, "double");

        // Nothing is allocated until the map is used, exactly as a fresh
        // `checked_vector_property_map` holds an empty `_store`.
        assert_eq!(
            vp.call_method0("allocated")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            0
        );

        // A *read* grows the store. This is the whole point of DESIGN.md §5's
        // "the dispatcher calls `sized_for` on read-only maps too": without it
        // the read is `PropError::Undersized` and graph-tool returns 0.0.
        assert_eq!(vp.get_item(4).unwrap().extract::<f64>().unwrap(), 0.0);
        assert_eq!(
            vp.call_method0("allocated")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            5,
            "a read sizes the map to the graph's vertex bound"
        );

        vp.set_item(2, 1.5f64).unwrap();
        assert_eq!(vp.get_item(2).unwrap().extract::<f64>().unwrap(), 1.5);
        assert_eq!(vp.get_item(0).unwrap().extract::<f64>().unwrap(), 0.0);
    });
}

#[test]
fn the_map_grows_with_the_graph() {
    with_module(|py, m| {
        let g = graph_with(m, 2);
        let vp = vprop(&g, "int32_t");
        vp.set_item(1, 7i32).unwrap();
        assert_eq!(
            vp.call_method0("allocated")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            2
        );

        // Before the graph grows, vertex 2 does not exist and is refused --
        // where `operator[]` would `soft_reserve(3)` and hand back a slot for
        // a vertex that is not there.
        let err = vp.get_item(2).unwrap_err();
        assert!(err.is_instance_of::<PyIndexError>(py), "{err}");

        g.call_method0("add_vertex").unwrap();
        assert_eq!(vp.get_item(2).unwrap().extract::<i32>().unwrap(), 0);
        assert_eq!(
            vp.call_method0("allocated")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            3
        );
        // Growth never drops what was already there.
        assert_eq!(vp.get_item(1).unwrap().extract::<i32>().unwrap(), 7);
    });
}

#[test]
fn a_rejected_value_leaves_the_map_untouched() {
    with_module(|py, m| {
        let g = graph_with(m, 3);
        let vp = vprop(&g, "int64_t");
        vp.set_item(0, 5i64).unwrap();

        let err = vp.set_item(1, "not a number").unwrap_err();
        assert!(
            err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
            "{err}"
        );
        assert_eq!(vp.get_item(0).unwrap().extract::<i64>().unwrap(), 5);
        assert_eq!(vp.get_item(1).unwrap().extract::<i64>().unwrap(), 0);
    });
}

// ---------------------------------------------------------------------------
// 3. type_names[] and the fifteen members
// ---------------------------------------------------------------------------

#[test]
fn value_types_is_type_names_verbatim() {
    with_module(|_py, m| {
        let got: Vec<String> = m
            .getattr("value_types")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        // `type_names[]`, `graph_properties.hh:72-76`, in declaration order.
        assert_eq!(
            got,
            vec![
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
            ]
        );
    });
}

#[test]
fn every_member_is_constructible_and_names_itself() {
    with_module(|_py, m| {
        let g = graph_with(m, 2);
        let names: Vec<String> = m
            .getattr("value_types")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(names.len(), 15);

        for name in &names {
            let vp = vprop(&g, name);
            assert_eq!(
                vp.call_method0("value_type")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                *name
            );
            assert_eq!(
                vp.call_method0("key_type")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "v"
            );
            let ep = eprop(&g, name);
            assert_eq!(
                ep.call_method0("key_type")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "e"
            );
        }

        // The fifteenth member is a different class, because `PyValue` is
        // `!Send` and `#[pyclass]` requires `Send` (DESIGN.md §7).
        let obj = vprop(&g, "python::object");
        assert_eq!(
            obj.get_type().name().unwrap().to_string(),
            "ObjectPropertyMap"
        );
        let dbl = vprop(&g, "double");
        assert_eq!(dbl.get_type().name().unwrap().to_string(), "PropertyMap");
    });
}

#[test]
fn an_unknown_type_name_is_graph_tools_value_exception() {
    with_module(|py, m| {
        let g = graph(m);
        for bad in [
            "",
            "int",
            "float",
            "object",
            "Bool",
            "vector<int>",
            "long_double",
        ] {
            let err = g.call_method1("new_vertex_property", (bad,)).unwrap_err();
            assert!(is_value_error(py, &err), "{bad:?}: {err}");
            assert!(
                err.to_string().contains("Invalid property type"),
                "{bad:?}: {err}"
            );
        }
    });
}

#[test]
fn each_member_round_trips_one_value() {
    with_module(|py, m| {
        let g = graph_with(m, 2);

        macro_rules! trip {
            ($ty:literal, $value:expr, $as:ty, $default:expr) => {{
                let vp = vprop(&g, $ty);
                let v: $as = $value;
                vp.set_item(0, v.clone().into_py(py)).unwrap();
                assert_eq!(
                    vp.get_item(0).unwrap().extract::<$as>().unwrap(),
                    v,
                    concat!($ty, " does not round-trip")
                );
                let d: $as = $default;
                assert_eq!(
                    vp.get_item(1).unwrap().extract::<$as>().unwrap(),
                    d,
                    concat!($ty, " has the wrong default")
                );
            }};
        }

        trip!("bool", 1u8, u8, 0);
        trip!("int16_t", -7i16, i16, 0);
        trip!("int32_t", 100_000i32, i32, 0);
        trip!("int64_t", 1i64 << 40, i64, 0);
        trip!("double", -0.5f64, f64, 0.0);
        trip!("string", "héllo".to_string(), String, String::new());
        trip!("vector<bool>", vec![1u8, 0, 1], Vec<u8>, Vec::new());
        trip!("vector<int16_t>", vec![1i16, -2], Vec<i16>, Vec::new());
        trip!("vector<int32_t>", vec![3i32, -4], Vec<i32>, Vec::new());
        trip!("vector<int64_t>", vec![5i64, -6], Vec<i64>, Vec::new());
        trip!("vector<double>", vec![0.25f64, 8.0], Vec<f64>, Vec::new());
        trip!(
            "vector<string>",
            vec!["a".to_string(), "".to_string()],
            Vec<String>,
            Vec::new()
        );

        // `long double` is an opaque 16-byte payload with no arithmetic
        // (DESIGN.md D7), so `bytes` is the only lossless Python spelling: a
        // `float` would round an 80-bit extended value through a `double` and
        // write a different one back to the `.gt` file.
        let ld = vprop(&g, "long double");
        let payload: [u8; 16] = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
            0x32, 0x10,
        ];
        ld.set_item(0, PyBytes::new_bound(py, &payload)).unwrap();
        assert_eq!(
            ld.get_item(0).unwrap().extract::<Vec<u8>>().unwrap(),
            payload.to_vec()
        );
        assert_eq!(
            ld.get_item(1).unwrap().extract::<Vec<u8>>().unwrap(),
            vec![0u8; 16],
            "the `long double` default is sixteen zero bytes"
        );
        // A payload of the wrong width is not a `long double`.
        let err = ld
            .set_item(0, PyBytes::new_bound(py, b"short"))
            .unwrap_err();
        assert!(is_value_error(py, &err), "{err}");

        let vld = vprop(&g, "vector<long double>");
        let items = vec![
            PyBytes::new_bound(py, &payload).unbind(),
            PyBytes::new_bound(py, &[0u8; 16]).unbind(),
        ];
        vld.set_item(0, items).unwrap();
        let back: Vec<Vec<u8>> = vld.get_item(0).unwrap().extract().unwrap();
        assert_eq!(back, vec![payload.to_vec(), vec![0u8; 16]]);
        assert!(
            vld.get_item(1)
                .unwrap()
                .extract::<Vec<Vec<u8>>>()
                .unwrap()
                .is_empty()
        );
    });
}

// ---------------------------------------------------------------------------
// 4. The scalar kernel: narrowing, sizing, detach
// ---------------------------------------------------------------------------

#[test]
fn the_scalar_kernel_sizes_a_read_only_map_before_running() {
    with_module(|_py, m| {
        let g = graph_with(m, 6);
        let vp = vprop(&g, "int64_t");
        // Never written, never read: the store is still empty.
        assert_eq!(
            vp.call_method0("allocated")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            0
        );

        // If the entry point handed this straight to the kernel, `view()`
        // would be `Undersized` -- which is exactly the failure `sized_for` on
        // read-only maps exists to prevent.
        let s: f64 = g
            .call_method1("sum_property", (&vp,))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(s, 0.0);
        assert_eq!(
            vp.call_method0("allocated")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            6
        );

        vp.set_item(0, 3i64).unwrap();
        vp.set_item(5, -1i64).unwrap();
        let s: f64 = g
            .call_method1("sum_property", (&vp,))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(s, 2.0);
    });
}

#[test]
fn the_scalar_kernel_reaches_all_five_members_and_refuses_the_rest() {
    with_module(|py, m| {
        let g = graph_with(m, 3);

        // `ScalarV` is `{uint8_t, int16_t, int32_t, int64_t, double}`
        // (`graph_properties.hh:83-89` minus `long double`, which has no
        // arithmetic in this port: `Scalar` excludes it, so it cannot be
        // named in the subset at all).
        macro_rules! reaches {
            ($ty:literal, $v:expr, $sum:expr) => {{
                let vp = vprop(&g, $ty);
                vp.set_item(0, $v).unwrap();
                let s: f64 = g
                    .call_method1("sum_property", (&vp,))
                    .unwrap()
                    .extract()
                    .unwrap();
                assert_eq!(s, $sum, $ty);
            }};
        }
        reaches!("bool", 1u8, 1.0);
        reaches!("int16_t", -2i16, -2.0);
        reaches!("int32_t", 3i32, 3.0);
        reaches!("int64_t", 1i64 << 33, 8_589_934_592.0);
        reaches!("double", 0.5f64, 0.5);

        // Everything else narrows to an error that *names the accepted set*,
        // where `DispatchNotFound`'s message is "This is a graph_tool bug. :-("
        // (`dispatch.hh:86-88`).
        for ty in [
            "long double",
            "string",
            "vector<bool>",
            "vector<double>",
            "vector<string>",
        ] {
            let vp = vprop(&g, ty);
            let err = g.call_method1("sum_property", (&vp,)).unwrap_err();
            assert!(is_value_error(py, &err), "{ty}: {err}");
            let msg = err.to_string();
            assert!(msg.contains("value:"), "{ty}: {msg}");
            assert!(
                msg.contains("F64"),
                "{ty}: {msg} should list the accepted set"
            );
        }
    });
}

#[test]
fn the_kernel_picks_the_bound_from_the_maps_own_key_type() {
    with_module(|_py, m| {
        let g = graph_with(m, 2);
        for pair in [(0, 1), (1, 0), (0, 0)] {
            g.call_method1("add_edge", pair).unwrap();
        }
        g.call_method1("remove_edge", (0,)).unwrap();

        let ep = eprop(&g, "double");
        ep.set_item(1, 2.0f64).unwrap();
        ep.set_item(2, 0.25f64).unwrap();
        let s: f64 = g
            .call_method1("sum_property", (&ep,))
            .unwrap()
            .extract()
            .unwrap();
        // Three slots in the index range, one of them freed and reading 0.0.
        assert_eq!(s, 2.25);
        assert_eq!(
            ep.call_method0("allocated")
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            3
        );
    });
}

#[test]
fn a_map_belonging_to_another_graph_is_refused() {
    with_module(|py, m| {
        let a = graph_with(m, 4);
        let b = graph_with(m, 4);
        let vp = vprop(&b, "double");

        // Same size, different graph. This is defect #8: `graph_copy.cc:66-73`
        // sizes from one count and writes at another's indices, and the Python
        // guard at `__init__.py:3200` compares counts too -- so equal sizes
        // make the C++ failure silent. Here the `GraphId` comparison in
        // `sized_for` catches it.
        let err = a.call_method1("sum_property", (&vp,)).unwrap_err();
        assert!(is_value_error(py, &err), "{err}");
        assert!(err.to_string().contains("belongs to graph"), "{err}");
    });
}

// ---------------------------------------------------------------------------
// 5. The Python member, and exceptions out of a kernel
// ---------------------------------------------------------------------------

#[test]
fn an_object_map_defaults_to_none_and_round_trips() {
    with_module(|py, m| {
        let g = graph_with(m, 3);
        let op = vprop(&g, "python::object");
        assert_eq!(
            op.call_method0("value_type")
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "python::object"
        );

        // `Py<PyAny>` has no `Default` and the default `python::object` is
        // `Py_None`, which needs the interpreter -- which is why `Zeroed` is
        // not a supertrait of `PropValue` and the fill is a closure
        // (DESIGN.md §13.1).
        assert!(op.get_item(0).unwrap().is_none());

        let d = PyDict::new_bound(py);
        d.set_item("k", 1).unwrap();
        op.set_item(1, &d).unwrap();
        let back = op.get_item(1).unwrap();
        assert!(back.is(&d), "the same object comes back, not a copy");
        assert!(op.get_item(2).unwrap().is_none());
    });
}

#[test]
fn an_exception_inside_a_kernel_surfaces_as_a_pyerr() {
    with_module(|py, m| {
        let g = graph_with(m, 4);
        let op = vprop(&g, "python::object");
        for v in 0..4 {
            op.set_item(v, v as i64).unwrap();
        }

        let ns = PyDict::new_bound(py);
        py.run_bound(
            "def boom(x):\n    if x == 2:\n        raise KeyError('kernel said no')\n    return x + 1\n",
            Some(&ns),
            None,
        )
        .unwrap();
        let boom = ns.get_item("boom").unwrap().unwrap();

        // `map_values` is the `Seq` arm: the token is held for the whole loop
        // and `detach` is not reachable, because its `F: Send` bound rejects a
        // closure capturing a `Bound<'py, PyAny>`. The C++ predicate at
        // `graph_properties_copy.cc:35-42` releases the GIL and parallelises
        // in exactly this case.
        let err = op.call_method1("map_values", (&boom,)).unwrap_err();
        assert!(
            err.is_instance_of::<pyo3::exceptions::PyKeyError>(py),
            "the callee's own exception type survives: {err}"
        );
        assert!(err.to_string().contains("kernel said no"), "{err}");

        // Nothing unwound past the boundary and nothing aborted: the map is
        // still a live, usable object, with the slots processed before the
        // raise already updated.
        assert_eq!(op.get_item(0).unwrap().extract::<i64>().unwrap(), 1);
        assert_eq!(op.get_item(1).unwrap().extract::<i64>().unwrap(), 2);
        assert_eq!(op.get_item(2).unwrap().extract::<i64>().unwrap(), 2);

        // And the interpreter is not left with a pending error.
        assert!(!PyErr::occurred(py));

        py.run_bound("def bump(x):\n    return x\n", Some(&ns), None)
            .unwrap();
        let bump = ns.get_item("bump").unwrap().unwrap();
        op.call_method1("map_values", (&bump,)).unwrap();
    });
}

#[test]
fn an_out_of_range_key_is_an_index_error_on_both_map_classes() {
    with_module(|py, m| {
        let g = graph_with(m, 2);
        for map in [vprop(&g, "double"), vprop(&g, "python::object")] {
            let err = map.get_item(2).unwrap_err();
            assert!(err.is_instance_of::<PyIndexError>(py), "{err}");
            let err = map.set_item(9, 1i64).unwrap_err();
            assert!(err.is_instance_of::<PyIndexError>(py), "{err}");
        }
    });
}

// ---------------------------------------------------------------------------
// 6. Registration
// ---------------------------------------------------------------------------

#[test]
fn register_exports_exactly_the_boundary() {
    with_module(|_py, m| {
        for name in ["Graph", "PropertyMap", "ObjectPropertyMap", "value_types"] {
            assert!(m.hasattr(name).unwrap(), "{name} is not registered");
        }
        // The classes advertise the module they belong to, so `repr` and
        // pickling name `graph_tool._rust` rather than `builtins`.
        for name in ["Graph", "PropertyMap", "ObjectPropertyMap"] {
            let cls = m.getattr(name).unwrap();
            assert_eq!(
                cls.getattr("__module__")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "graph_tool._rust",
                "{name}"
            );
        }
    });
}
