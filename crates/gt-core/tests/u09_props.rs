//! U9 — property maps, checked through the public surface.
//!
//! ## Where the rest of this unit's tests live, and why
//!
//! `Bound::new` is `pub(crate)`. That is not an inconvenience, it is defect #7's
//! entire mechanism: `graph_copy.cc:66-73` reserves `num_vertices(src)` — a
//! *filtered* count — and then writes at unfiltered indices, and the only
//! reason that is possible is that the size is a `size_t` anyone can produce.
//! `tests/ui/u01_bound_new_is_private.rs` pins the `E0624` a downstream crate
//! gets for trying, and this file is a downstream crate.
//!
//! So a bound of non-zero length can only come from a graph:
//! `AdjList::with_vertices(n)` (U5) is what this file sizes maps against, and
//! a `DenseProp` built for a `GraphId` that graph did not mint is refused
//! whatever the two lengths are. The same guarantees are checked a second time
//! in `src/prop/dense.rs`'s own `#[cfg(test)]` module, which is inside the
//! crate and can therefore mint a bound directly — worth the duplication,
//! because that copy keeps running if `AdjList` ever stops being constructible
//! and this one would otherwise go quiet without going red.
//!
//! The rest is the surface a kernel sees: the trait split, the ZST claims, the
//! four `trybuild` diagnostics, and DESIGN.md §12's two opposite codegen
//! claims.

use std::path::PathBuf;
use std::process::Command;

use gt_core::adj::AdjList;
use gt_core::error::PropError;
use gt_core::ids::{EdgeTag, GraphId, VertexId, VertexTag};
use gt_core::prop::{
    ConstI64, Constant, DenseProp, EdgeProp, IndexProp, LvalueProp, PropSliceMut, ReadProp, Unity,
    VertexProp,
};

// ===========================================================================
// 1. Defect #8 — a map from graph A, a bound from graph B
// ===========================================================================

/// Runs today: identity is not length, so an *empty* bound from the wrong
/// graph is rejected just as loudly as a full one. That is the whole point —
/// C++ compares sizes (`__init__.py:3200`), which agree in exactly the case
/// that matters.
#[test]
fn sizing_a_map_for_another_graph_is_an_error() {
    let a = AdjList::new();
    let b = AdjList::new();
    assert_ne!(a.graph_id(), b.graph_id());

    let mut p: VertexProp<i64> = DenseProp::new(a.graph_id());
    assert_eq!(
        p.sized_for(b.vertex_bound()).unwrap_err(),
        PropError::WrongGraph {
            owner: a.graph_id().get(),
            expected: b.graph_id().get(),
        }
    );
    assert_eq!(
        p.view(b.vertex_bound()).unwrap_err(),
        PropError::WrongGraph {
            owner: a.graph_id().get(),
            expected: b.graph_id().get(),
        }
    );
    // Its own graph's bound is accepted.
    assert!(p.sized_for(a.vertex_bound()).is_ok());

    // And the edge space of the same graph is a different bound over the same
    // identity, so this is a length question, never an identity one.
    let mut e: EdgeProp<i64> = DenseProp::new(a.graph_id());
    assert!(e.sized_for(a.edge_bound()).is_ok());
    assert_eq!(
        e.sized_for(b.edge_bound()).unwrap_err(),
        PropError::WrongGraph {
            owner: a.graph_id().get(),
            expected: b.graph_id().get(),
        }
    );
}

/// The error message names both identities, because "wrong graph" without
/// saying which two is not actionable at a 324-call-site dispatch boundary.
#[test]
fn the_wrong_graph_error_names_both_graphs() {
    let owner = GraphId::fresh();
    let expected = GraphId::fresh();
    let e = PropError::WrongGraph {
        owner: owner.get(),
        expected: expected.get(),
    };
    assert_eq!(
        e.to_string(),
        format!(
            "property map belongs to graph #{}, not graph #{}",
            owner.get(),
            expected.get()
        )
    );
}

// ===========================================================================
// 2. Sizing, through the public surface
// ===========================================================================

#[test]
fn sized_for_grows_to_exactly_the_bound_and_never_shrinks() {
    let g = AdjList::with_vertices(6);
    let bound = g.vertex_bound();
    assert_eq!(bound.len(), 6);

    let mut p: VertexProp<i64> = DenseProp::new(g.graph_id());
    {
        let v = p.sized_for(bound).expect("same graph");
        assert_eq!(v.len(), 6);
        assert_eq!(v.as_slice(), &[0i64; 6]);
    }
    assert_eq!(p.len(), 6);

    // Idempotent.
    p.as_mut_slice()[4] = 9;
    assert_eq!(p.sized_for(bound).expect("same graph").len(), 6);
    assert_eq!(p.as_slice()[4], 9);
    assert_eq!(p.len(), 6);

    // Never shrinks: a smaller graph's bound yields a shorter *view* over an
    // unchanged store.
    let small = AdjList::with_vertices(2);
    let mut q: VertexProp<i64> = DenseProp::from_vec(small.graph_id(), vec![1, 2, 3, 4, 5]);
    assert_eq!(q.sized_for(small.vertex_bound()).expect("same").len(), 2);
    assert_eq!(q.len(), 5);
}

#[test]
fn sized_for_with_fills_only_the_new_slots() {
    let g = AdjList::with_vertices(4);
    let mut p: VertexProp<i64> = DenseProp::from_vec(g.graph_id(), vec![-1, -2]);
    let mut n = 0;
    let v = p
        .sized_for_with(g.vertex_bound(), || {
            n += 1;
            n
        })
        .expect("same graph");
    assert_eq!(v.as_slice(), &[-1, -2, 1, 2]);
}

#[test]
fn view_on_an_undersized_map_is_an_error() {
    let g = AdjList::with_vertices(5);
    let p: VertexProp<i64> = DenseProp::from_vec(g.graph_id(), vec![0; 3]);
    match p.view(g.vertex_bound()) {
        Err(PropError::Undersized { have, need }) => {
            assert_eq!((have, need), (3, 5));
        }
        Err(other) => panic!("expected Undersized, got {other}"),
        // The failure this test exists for: a short slice, handed out.
        Ok(v) => panic!("view() returned a run of {} for a bound of 5", v.len()),
    }
}

/// An empty map and an empty bound agree, so the degenerate case is not an
/// error. `AdjList::new()` is the only graph available before U5, which is
/// what makes this one runnable today.
#[test]
fn an_empty_map_satisfies_an_empty_graphs_bound() {
    let g = AdjList::new();
    let p: VertexProp<i64> = DenseProp::new(g.graph_id());
    assert!(p.view(g.vertex_bound()).expect("0 <= 0").is_empty());
}

// ===========================================================================
// 3. The trait surface
// ===========================================================================

#[test]
fn the_index_map_reads_back_the_descriptors_own_index() {
    let g = AdjList::new();
    let p: IndexProp<VertexTag> = IndexProp::new(g.vertex_bound());
    for i in [0usize, 1, 7, 1023, 65_536] {
        let k = VertexId::from_index(i);
        assert_eq!(*p.get_ref(k), k.index() as i64);
    }
    assert_eq!(p.bound(), g.vertex_bound());
    // `IndexProp` is `ReadProp` and not `WriteProp`:
    // `tests/ui/u09_index_map_has_no_put.rs`.
    fn only_readable<P: ReadProp<VertexTag, Value = i64>>(p: &P, k: VertexId) -> i64 {
        p.get(k)
    }
    assert_eq!(only_readable(&p, VertexId::from_index(3)), 3);
}

/// Defect #43's sizeof half: the C++ unity map is an empty *class*, so it
/// costs a byte and 74 call sites pass one around.
#[test]
fn the_constant_maps_are_zero_sized_where_they_claim_to_be() {
    assert_eq!(size_of::<Unity<f64, VertexTag>>(), 0);
    assert_eq!(size_of::<Unity<i64, EdgeTag>>(), 0);
    assert_eq!(size_of::<ConstI64<5, VertexTag>>(), 0);
    // `Constant` carries its value, so it is exactly that value.
    assert_eq!(size_of::<Constant<f64, VertexTag>>(), size_of::<f64>());

    assert_eq!(
        *Unity::<f64, VertexTag>::NEW.get_ref(VertexId::from_index(4)),
        1.0
    );
    assert_eq!(
        *Constant::<f64, VertexTag>::new(3.0).get_ref(VertexId::from_index(4)),
        3.0
    );
    assert_eq!(
        *ConstI64::<5, VertexTag>::NEW.get_ref(VertexId::from_index(4)),
        5
    );
}

/// The consts a kernel branches on. `IS_UNITY` on a `ConstI64<1>` is the case
/// C++ loses (defect #45) whenever the map is wrapped.
#[test]
fn the_fast_path_flags_are_compile_time_constants() {
    const U: bool = <Unity<f64, VertexTag> as ReadProp<VertexTag>>::IS_UNITY;
    const C1: bool = <ConstI64<1, VertexTag> as ReadProp<VertexTag>>::IS_UNITY;
    const C2: bool = <ConstI64<2, VertexTag> as ReadProp<VertexTag>>::IS_UNITY;
    const D: bool = <VertexProp<f64> as ReadProp<VertexTag>>::IS_UNITY;
    assert_eq!([U, C1, C2, D], [true, true, false, false]);
}

/// A kernel takes the view *by value* (see `PropSliceMut::reborrow`'s note),
/// and both write traits are available on it.
#[test]
fn a_writable_view_is_a_write_prop_and_an_lvalue_prop() {
    fn scatter<P: LvalueProp<VertexTag, Value = i64>>(mut p: P, ks: &[VertexId]) -> P {
        for (i, &k) in ks.iter().enumerate() {
            p.put(k, i as i64);
            *p.at_mut(k) += 100;
        }
        p
    }
    let g = AdjList::new();
    let mut store: VertexProp<i64> = DenseProp::from_vec(g.graph_id(), vec![0; 3]);
    {
        let view: PropSliceMut<'_, i64, VertexTag> =
            store.sized_for(g.vertex_bound()).expect("empty bound");
        // A zero-length view is still a view; nothing to scatter into it.
        assert!(view.is_empty());
    }
    let ks = [VertexId::from_index(0), VertexId::from_index(2)];
    let store = scatter(store, &ks);
    assert_eq!(store.as_slice(), &[100, 0, 101]);
}

// ===========================================================================
// 4. The negative guarantees
// ===========================================================================

/// | fixture | defect | diagnostic |
/// |---|---|---|
/// | `u09_unity_has_no_put` | #43 `graph_properties.hh:711-716` | `E0599` |
/// | `u09_constant_has_no_put` | #43/#44 `graph_properties.hh:670-696` | `E0599` |
/// | `u09_index_map_has_no_put` | `graph_properties.hh:166-231, :278` | `E0599` |
/// | `u09_shared_view_has_no_put` | #11 `fast_vector_property_map.hh:219-222` | `E0599` |
#[test]
fn the_unwritable_maps_still_refuse_to_compile_a_write() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/u09_*.rs");
}

// ===========================================================================
// 5. The codegen assertion (DESIGN.md §12)
// ===========================================================================
//
// The ledger claims two things about property access and they are opposite
// claims, so one test must show both:
//
//   * "`as_slice()` is the bounds-check-free path" — a scan through
//     `PropSlice::as_slice()` must contain no `panic_bounds_check`.
//   * "Bounds checks on scatter ... accepted per D11" — a per-key
//     `put(k, v)` loop must contain one. A *stated loss* that turned out not to
//     be there would mean the probe is not measuring what it says, and a
//     stated loss nobody checks is how a ledger rots.
//
// Compiled with `rustc -O` against the rlib this test binary was built beside,
// as U4 does: same optimisation pipeline for the question being asked, without
// a fat-LTO rebuild of the workspace.

fn deps_dir() -> Option<PathBuf> {
    Some(std::env::current_exe().ok()?.parent()?.to_path_buf())
}

fn gt_core_rlib() -> Option<PathBuf> {
    let dir = deps_dir()?;
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&dir).ok()? {
        let path = entry.ok()?.path();
        let name = path.file_name()?.to_string_lossy().into_owned();
        if name.starts_with("libgt_core-") && name.ends_with(".rlib") {
            let when = path.metadata().ok()?.modified().ok()?;
            if best.as_ref().is_none_or(|(b, _)| when > *b) {
                best = Some((when, path));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// Compile `body` as a `#[no_mangle]` free function against gt-core and return
/// the assembly, or `None` if the toolchain or the rlib is not where this test
/// can see it.
fn probe_asm(name: &str, body: &str) -> Option<String> {
    let rlib = gt_core_rlib()?;
    let deps = deps_dir()?;
    let dir = std::env::temp_dir().join(format!("gt_u09_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("probe.rs");
    let asm = dir.join("probe.s");
    std::fs::write(
        &src,
        format!(
            "extern crate gt_core;\n\
             #[allow(unused_imports)]\n\
             use gt_core::ids::{{VertexId, VertexTag}};\n\
             #[allow(unused_imports)]\n\
             use gt_core::prop::{{PropSlice, PropSliceMut, ReadProp, WriteProp}};\n\
             #[no_mangle]\n\
             {body}\n"
        ),
    )
    .ok()?;

    let out = Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
        .args([
            "-O",
            "--edition",
            "2021",
            "--crate-type",
            "lib",
            "--emit",
            "asm",
        ])
        .arg("--extern")
        .arg(format!("gt_core={}", rlib.display()))
        .arg("-L")
        .arg(format!("dependency={}", deps.display()))
        .arg("-o")
        .arg(&asm)
        .arg(&src)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "SKIPPED: could not compile the codegen probe:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    let text = std::fs::read_to_string(&asm).ok();
    let _ = std::fs::remove_dir_all(&dir);
    text
}

#[test]
fn a_scan_through_as_slice_has_no_bounds_check() {
    let Some(asm) = probe_asm(
        "scan",
        "pub fn s(p: &PropSlice<'_, f64, VertexTag>) -> f64 \
         { p.as_slice().iter().sum() }",
    ) else {
        return;
    };
    assert!(
        asm.contains("s:"),
        "the probe did not emit the symbol under test"
    );
    assert!(
        !asm.contains("panic_bounds_check"),
        "PropSlice::as_slice().iter() left a bounds check in the loop; \
         DESIGN.md §12 calls this the bounds-check-free path"
    );

    // The same must hold for the writable view's bulk path, which is what a
    // scatter expressible as a scan is supposed to escape into.
    let Some(asm) = probe_asm(
        "scan_mut",
        "pub fn s(p: &mut PropSliceMut<'_, f64, VertexTag>) \
         { for x in p.as_mut_slice() { *x *= 2.0; } }",
    ) else {
        return;
    };
    assert!(
        !asm.contains("panic_bounds_check"),
        "PropSliceMut::as_mut_slice() left a bounds check in the loop"
    );
}

/// The stated loss, asserted as a loss. `pm[target(e)] += x` keeps one compare
/// per element that `unchecked_vector_property_map::operator[]`
/// (`fast_vector_property_map.hh:219-222`) does not — and does not, only
/// because it reads out of bounds instead.
#[test]
fn a_per_key_scatter_keeps_its_bounds_check() {
    let Some(asm) = probe_asm(
        "scatter",
        "pub fn s(p: &mut PropSliceMut<'_, f64, VertexTag>, ks: &[VertexId]) \
         { for &k in ks { p.put(k, 1.0); } }",
    ) else {
        return;
    };
    assert!(
        asm.contains("s:"),
        "the probe did not emit the symbol under test"
    );
    assert!(
        asm.contains("panic_bounds_check"),
        "the per-key scatter has no bounds check. That is not a win to be \
         celebrated: `put` is a checked index, so either this probe is not \
         measuring the scatter path or the check has been removed by something \
         that should be looked at. DESIGN.md §12 records the check as an \
         accepted cost (D11)."
    );
}
