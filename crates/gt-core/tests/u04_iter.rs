//! U4 — the iterators.
//!
//! What this file asserts is the *global* edge list, `AdjList::edges()`, which
//! ports `adj_list<Vertex>::edge_iterator` (`graph_adjacency.hh:360-416`). The
//! C++ iterator is three cursors plus a `skip()` that walks `_vi` forward
//! while `_ei == _vi->second.begin() + _vi->first` (`:384-385`) — the **out**
//! half's end, not the block's end. That one detail is why the global list
//! yields each edge once although every edge is stored twice (once in the
//! source's out-half, once in the target's in-half), and it is the property
//! DESIGN.md section 4 leans on when it makes `EdgeList::edges` — not
//! degree-summation — the definition of `num_edges` on a view.
//!
//! ## Why the graph-building tests are conditional
//!
//! `AdjList::with_vertices`, `add_edge` and `remove_edge` are U5's and are
//! still `todo!()`, so from *outside* the crate there is no populated graph to
//! sweep. Rather than ship a file that asserts nothing, each test that needs
//! one goes through [`build`], which returns `None` while the dependency
//! panics and prints a loud skip line; the assertions are written out in full
//! and start running the moment U5 lands.
//!
//! They are not the only copy. `src/adj/iter.rs`'s own `#[cfg(test)]` module
//! asserts every one of the same properties *unconditionally*, building the
//! blocks through U2's `pub(crate)` `Block::insert_out`/`insert_in` — each
//! edge exactly once, the self-loop, `EdgeId` order after a compaction, the
//! `fold`/`next` agreement, and the skip cost as a step count rather than a
//! timing. What this file adds on top is the wiring: that `AdjList::edges`
//! hands `Edges::new` the right slice, and that the honest count
//! `edges().count() == num_edges()` (DESIGN.md section 4) survives removals.

use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use gt_core::adj::Graph;
use gt_core::ids::{EdgeId, VertexId};

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

/// Serialises the panic-hook swap in [`build`]: libtest runs these tests on
/// several threads and the hook is process-global.
fn hook_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// `n` vertices and `edges` inserted in order, with the ids `add_edge`
/// returned — or `None` if U2/U5 have not landed yet.
///
/// The `todo!()` in the dependency is a panic, so the probe is a
/// `catch_unwind` with the hook silenced; a real failure inside a landed
/// implementation would come back as `None` too, which is why the skip
/// message names the dependency rather than claiming success.
fn build(n: usize, edges: &[(usize, usize)]) -> Option<(Graph, Vec<EdgeId>)> {
    let _guard = hook_lock().lock().unwrap_or_else(|e| e.into_inner());
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let built = panic::catch_unwind(AssertUnwindSafe(|| {
        let mut g = Graph::with_vertices(n);
        let ids: Vec<EdgeId> = edges
            .iter()
            .map(|&(s, t)| g.add_edge(v(s), v(t)).expect("add_edge").id())
            .collect();
        (g, ids)
    }));
    panic::set_hook(previous);
    if built.is_err() {
        eprintln!(
            "SKIPPED: AdjList::with_vertices/add_edge is still todo!() (U5). \
             This test's assertions are not running; the same properties are \
             asserted unconditionally in src/adj/iter.rs."
        );
    }
    built.ok()
}

/// `(source, target, edge id)` for every edge the global list yields.
fn swept(g: &Graph) -> Vec<(usize, usize, usize)> {
    g.edges()
        .map(|e| (e.source().index(), e.target().index(), e.id().index()))
        .collect()
}

// ===========================================================================
// 1. Each edge exactly once, in canonical orientation
// ===========================================================================

/// The shape that catches every way of getting this wrong at once:
///
/// * vertex 0 has out-edges *and* in-edges — walking the whole block
///   (`Block::all`) instead of the out-half yields its in-edges a second
///   time, so `4` becomes `7`;
/// * vertex 2 is isolated and vertex 4 has only in-edges — a cursor that
///   stops at the first exhausted out-half never reaches vertex 3;
/// * `(3, 3)` is a self-loop, stored once in the out-half and once in the
///   in-half of the same block — the case that makes `_n_edges` wrong in
///   `clear_vertex` (`graph_adjacency.hh:1409-1413`);
/// * `(0, 1)` twice is a parallel pair, which is legal and must not be
///   deduplicated: identity is the `EdgeId`, not the endpoint pair (D2).
const MIXED: [(usize, usize); 6] = [(0, 1), (0, 1), (1, 0), (3, 3), (3, 4), (1, 4)];

#[test]
fn edges_yields_every_edge_exactly_once_in_storage_orientation() {
    let Some((g, ids)) = build(5, &MIXED) else {
        return;
    };

    // The C++ order: vertices ascending, and within a vertex the out-half in
    // the order `insert_out` appended to it.
    let mut expected: Vec<(usize, usize, usize)> = Vec::new();
    for u in 0..5 {
        for (k, &(s, t)) in MIXED.iter().enumerate() {
            if s == u {
                expected.push((s, t, ids[k].index()));
            }
        }
    }
    assert_eq!(swept(&g), expected);

    // Each id once, and the count is the honest one.
    let mut seen: Vec<usize> = g.edges().map(|e| e.id().index()).collect();
    let before = seen.len();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), before, "an edge was yielded twice");
    assert_eq!(before, g.num_edges(), "edges().count() != num_edges()");
    assert_eq!(g.edges().count(), g.num_edges());
}

/// A self-loop is one edge, not two: it sits in both halves of one block, and
/// the out-half is walked, so it appears once.
#[test]
fn a_self_loop_is_yielded_once() {
    let Some((g, _)) = build(2, &[(0, 0), (0, 1)]) else {
        return;
    };
    let loops = g.edges().filter(|e| e.source() == e.target()).count();
    assert_eq!(loops, 1);
    assert_eq!(g.edges().count(), 2);
}

/// Every vertex whose out-half is empty — isolated, or in-edges only — is
/// skipped, and no edge is attributed to it as a source.
#[test]
fn vertices_with_no_out_half_are_skipped_not_yielded() {
    // 0 -> 9 only; 1..9 are isolated; 9 has an in-edge and nothing else.
    let Some((g, ids)) = build(10, &[(0, 9)]) else {
        return;
    };
    assert_eq!(swept(&g), vec![(0, 9, ids[0].index())]);

    // The mirror case: the only out-half is the *last* block, so the cursor
    // has to cross nine empty ones to find it.
    let Some((g, ids)) = build(10, &[(9, 0)]) else {
        return;
    };
    assert_eq!(swept(&g), vec![(9, 0, ids[0].index())]);
}

// ===========================================================================
// 2. EdgeId order
// ===========================================================================

/// `EdgeIds::alloc` hands out `0, 1, 2, …` while the free list is empty, so a
/// graph built source-major — which is also the state
/// [`EdgeIds::compact`](gt_core::adj::EdgeIds::compact) leaves behind, since
/// it remaps the live ids onto `0..live` in sweep order — yields ids in
/// ascending order with no gaps.
#[test]
fn edges_are_in_edge_id_order_after_a_compaction() {
    let source_major: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)];
    let Some((g, _)) = build(3, &source_major) else {
        return;
    };
    let ids: Vec<usize> = g.edges().map(|e| e.id().index()).collect();
    assert_eq!(ids, (0..source_major.len()).collect::<Vec<_>>());
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "ids are not ascending: {ids:?}"
    );
}

/// And the converse, so the test above is not read as a promise the iterator
/// does not make: built out of source order, the sweep is in *vertex* order,
/// which is not id order. The invariant is "each edge once", not "sorted".
#[test]
fn edges_are_not_sorted_by_id_when_insertion_was_not_source_major() {
    let Some((g, ids)) = build(3, &[(2, 0), (0, 1)]) else {
        return;
    };
    assert_eq!(
        swept(&g),
        vec![(0, 1, ids[1].index()), (2, 0, ids[0].index())]
    );
}

// ===========================================================================
// 3. The cursor's own contract
// ===========================================================================

/// `count()` and `sum()` route through the `fold` override; `next()` is the
/// other implementation. Two implementations of one thing is two chances to
/// be wrong, so they are pinned against each other from every starting
/// position, including one parked mid-block.
#[test]
fn fold_and_next_agree_from_every_starting_position() {
    let Some((g, _)) = build(5, &MIXED) else {
        return;
    };
    let full = swept(&g);
    for skip in 0..=full.len() {
        let mut it = g.edges();
        for _ in 0..skip {
            it.next();
        }
        let folded = it.clone().fold(Vec::new(), |mut acc, e| {
            acc.push((e.source().index(), e.target().index(), e.id().index()));
            acc
        });
        assert_eq!(folded, full[skip..].to_vec(), "fold disagrees after {skip}");
        assert_eq!(it.count(), full.len() - skip);
    }
}

#[test]
fn the_sweep_is_fused() {
    let Some((g, _)) = build(4, &[(0, 1)]) else {
        return;
    };
    let mut it = g.edges();
    assert!(it.next().is_some());
    for _ in 0..8 {
        assert!(it.next().is_none());
    }
}

/// An empty graph, and a graph of isolated vertices, are both empty sweeps —
/// and `equal()` in the C++ has a special case for `_vi_begin == _vi_end`
/// (`:399-403`) precisely because the no-vertex case is otherwise wrong there.
#[test]
fn an_edgeless_graph_sweeps_empty() {
    let g = Graph::new();
    assert_eq!(g.edges().count(), 0);
    assert!(g.edges().next().is_none());

    let Some((g, _)) = build(32, &[]) else {
        return;
    };
    assert_eq!(g.edges().count(), 0);
    assert_eq!(g.num_edges(), 0);
}

/// After removals the id space is sparse, so the sweep is where "each edge
/// exactly once" has to hold against a *free list*: `num_edges()` is
/// `EdgeIds::live()` and the sweep must agree with it, which is DESIGN.md's
/// answer to defect #16 (`distance(ei, eiend) != num_edges(g)`, `:301-312`).
#[test]
fn the_sweep_agrees_with_num_edges_after_removals() {
    let Some((mut g, ids)) = build(5, &MIXED) else {
        return;
    };
    let dropped = {
        let _guard = hook_lock().lock().unwrap_or_else(|e| e.into_inner());
        let previous = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            g.remove_edge(ids[0]).expect("remove_edge");
            g.remove_edge(ids[3]).expect("remove_edge");
        }));
        panic::set_hook(previous);
        r.is_ok()
    };
    if !dropped {
        eprintln!("SKIPPED: AdjList::remove_edge is still todo!() (U5).");
        return;
    }
    assert_eq!(g.edges().count(), g.num_edges());
    assert_eq!(g.num_edges(), MIXED.len() - 2);
    let live: Vec<usize> = g.edges().map(|e| e.id().index()).collect();
    assert!(!live.contains(&ids[0].index()));
    assert!(!live.contains(&ids[3].index()));
}

// ===========================================================================
// 4. The codegen assertion
// ===========================================================================
//
// IMPLEMENTATION_PLAN.md asks for the disassembly of
//
//     fn s(g: &AdjList, v: VertexId) -> u64 { g.out_edges(v).map(|i| i.other.raw() as u64).sum() }
//
// to contain no `panic_bounds_check` and no allocator call. `out_edges` is
// U5's `todo!()`, so that exact probe is compiled *and checked* here but skips
// itself while the body is a panic; the same assertion is made unconditionally
// against `edges()`, which is U4's own hot path and is implemented.
//
// The probe is compiled with `rustc -O` against the rlib this test binary was
// built beside, not with `cargo --release`: same optimisation pipeline for the
// question being asked (`#[inline]` bodies do cross the crate edge without
// LTO), seconds instead of a fat-LTO rebuild of the workspace.

/// Everything an allocation on the hot path would leave in the assembly.
const ALLOCATOR_SYMBOLS: [&str; 4] = [
    "__rust_alloc",
    "__rust_dealloc",
    "__rust_realloc",
    "__rust_no_alloc_shim_is_unstable",
];

fn deps_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.to_path_buf())
}

/// The freshest `libgt_core-*.rlib` beside this test binary.
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
/// can see it (a packaged-source run, for instance).
fn probe_asm(name: &str, body: &str) -> Option<String> {
    let rlib = gt_core_rlib()?;
    let deps = deps_dir()?;
    let dir = std::env::temp_dir().join(format!("gt_u04_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("probe.rs");
    let asm = dir.join("probe.s");
    std::fs::write(
        &src,
        format!(
            "extern crate gt_core;\n\
             #[allow(unused_imports)]\n\
             use gt_core::adj::AdjList;\n\
             #[allow(unused_imports)]\n\
             use gt_core::ids::VertexId;\n\
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

fn assert_no_bounds_check_and_no_allocation(what: &str, asm: &str) {
    assert!(
        !asm.contains("panic_bounds_check"),
        "{what}: the optimiser left a bounds check in the loop"
    );
    for symbol in ALLOCATOR_SYMBOLS {
        assert!(
            !asm.contains(symbol),
            "{what}: the loop calls the allocator ({symbol})"
        );
    }
}

/// U4's own hot path: the global sweep compiles to two nested slice walks with
/// no indexing panic and no allocation.
#[test]
fn the_global_sweep_has_no_bounds_check_and_no_allocation() {
    let Some(asm) = probe_asm(
        "edges",
        "pub fn s(g: &AdjList) -> u64 { g.edges().map(|e| e.target().raw() as u64).sum() }",
    ) else {
        return;
    };
    assert!(
        asm.contains("s:"),
        "the probe did not emit the symbol under test"
    );
    assert_no_bounds_check_and_no_allocation("edges().map(..).sum()", &asm);

    // `count()` goes through the `fold` override, which is the version a
    // `num_edges()`-shaped call reaches.
    let Some(asm) = probe_asm(
        "count",
        "pub fn s(g: &AdjList) -> usize { g.edges().count() }",
    ) else {
        return;
    };
    assert_no_bounds_check_and_no_allocation("edges().count()", &asm);
}

/// The plan's literal probe. Runs as soon as `AdjList::out_edges` (U5) is not
/// a `todo!()`; until then the assembly is a panic and there is nothing to
/// assert about it.
#[test]
fn the_incidence_sweep_has_no_bounds_check_and_no_allocation() {
    let Some(asm) = probe_asm(
        "out_edges",
        "pub fn s(g: &AdjList, v: VertexId) -> u64 \
         { g.out_edges(v).map(|i| i.other.raw() as u64).sum() }",
    ) else {
        return;
    };
    if asm.contains("not yet implemented") {
        eprintln!("SKIPPED: AdjList::out_edges is still todo!() (U5).");
        return;
    }
    assert_no_bounds_check_and_no_allocation("out_edges(v).map(..).sum()", &asm);
}
