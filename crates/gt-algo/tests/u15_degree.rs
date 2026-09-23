//! U15 -- the degree kernels.
//!
//! The unit's whole claim is that `graph_selectors.hh`'s four-way overload set
//! survives as two associated `const`s, and that both fast paths are *real*:
//!
//! * the unity path must not touch the weight map at all. A `Unity`-shaped map
//!   whose `get_ref` panics proves that at run time, where a disassembly
//!   proves it at the instruction level;
//! * the constant path must **multiply**, not accumulate. `graph_selectors.hh:173`
//!   writes `out_degree(v, g) * weight.c` and cannot be instantiated (defect
//!   #44: `weight.c` names the private `_c` of
//!   `graph_properties.hh:677`), so in graph-tool every constant-weighted
//!   degree silently falls through to the loop. `c = 0.1` over ten edges
//!   separates the two answers exactly: `10.0 * 0.1 == 1.0`, while ten
//!   additions of `0.1` give `0.999...9`.
//!
//! The accumulator is `f64` reached through `ToF64`, not `Into<f64>`, because
//! `i64` -- the only width graph-tool instantiates (`src/graph/graph.hh:137`)
//! -- does not implement `Into<f64>`. `tests/ui/u15_i64_weight_is_admissible.rs`
//! pins that as a compiling program and
//! `tests/ui/u15_string_weight_is_rejected.rs` pins the diagnostic for a
//! member of the value universe that has no numeric reading at all.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use gt_algo::degree::{
    degree_histogram, out_degrees, weighted_degree, weighted_in_degree, weighted_out_degree,
};
use gt_core::adj::AdjList;
use gt_core::ids::{EdgeId, EdgeTag, VertexId};
use gt_core::prop::dense::{Constant, EdgeProp, Unity};
use gt_core::prop::{DenseProp, Owned, ReadProp};
use gt_core::view::{Reverse, Undirect};

// ===========================================================================
// An allocation counter (the shape `tests/u05_adjlist.rs` established)
//
// The acceptance criterion "no allocation per vertex" is a statement about the
// global allocator, which has no safe observer. This is the one `unsafe` in
// the unit; it lives in a test binary, a separate crate from `gt-algo` and so
// not covered by that crate's `#![forbid(unsafe_code)]`, and every operation
// forwards to `System`. The tally is thread-local because `cargo test` runs
// this binary's tests concurrently.
// ===========================================================================

thread_local! {
    /// `-1` while this thread is not counting; otherwise the tally so far.
    static TALLY: Cell<isize> = const { Cell::new(-1) };
}

fn bump() {
    let _ = TALLY.try_with(|t| {
        let n = t.get();
        if n >= 0 {
            t.set(n + 1);
        }
    });
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(p, l, new) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Count the allocations `f` performs on this thread.
fn allocations<R>(f: impl FnOnce() -> R) -> (R, usize) {
    TALLY.with(|t| t.set(0));
    let r = f();
    let n = TALLY.with(|t| {
        let n = t.get();
        t.set(-1);
        n
    });
    (r, usize::try_from(n).expect("the tally was enabled"))
}

// ===========================================================================
// The graph under test
// ===========================================================================

/// Six vertices. Vertex 0 has out-degree **ten** -- the constant-path test
/// needs a degree at which `c * d` and `d` additions of `c` differ in `f64`
/// -- with parallel edges, a self-loop, and one edge back into it so that the
/// in-half is not empty either.
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (0, 1),
    (0, 2),
    (0, 2),
    (0, 3),
    (0, 3),
    (0, 4),
    (0, 4),
    (0, 5),
    (0, 0), // self-loop: one slot in each half of block 0
    (1, 2),
    (2, 0),
];
const N: usize = 6;

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

fn build() -> AdjList {
    let mut g = AdjList::with_vertices(N);
    for &(s, t) in &EDGES {
        g.add_edge(v(s), v(t)).expect("add_edge");
    }
    g
}

/// `1.0 ..= 12.0`. Small integers, so every partial sum is exact and the
/// reference below does not depend on the order storage happens to walk.
fn weights() -> Vec<f64> {
    (0..EDGES.len()).map(|i| i as f64 + 1.0).collect()
}

fn ref_out(w: &[f64], u: usize) -> f64 {
    EDGES
        .iter()
        .enumerate()
        .filter(|&(_, &(s, _))| s == u)
        .map(|(i, _)| w[i])
        .sum()
}
fn ref_in(w: &[f64], u: usize) -> f64 {
    EDGES
        .iter()
        .enumerate()
        .filter(|&(_, &(_, t))| t == u)
        .map(|(i, _)| w[i])
        .sum()
}

// ===========================================================================
// Weight maps that report on themselves
// ===========================================================================

/// A map that claims to be unity and **panics if read**.
///
/// The only way to observe "the fast path did not enter the loop" from safe
/// Rust: `graph_selectors.hh:182-187` returns `out_degree(v, g)` without
/// touching `weight`, and so must this port.
#[derive(Clone, Copy, Default, Debug)]
struct PoisonUnity;

impl ReadProp<EdgeTag> for PoisonUnity {
    type Value = f64;
    type Ref<'s>
        = Owned<f64>
    where
        Self: 's;
    const IS_UNITY: bool = true;
    const IS_CONSTANT: bool = true;
    fn get_ref(&self, _k: EdgeId) -> Owned<f64> {
        panic!("the unity fast path must not read the weight map");
    }
}

/// A constant map that counts its reads. The constant path is allowed exactly
/// one, whatever the degree.
#[derive(Debug)]
struct CountingConstant {
    c: f64,
    reads: Cell<usize>,
}

impl CountingConstant {
    fn new(c: f64) -> Self {
        CountingConstant {
            c,
            reads: Cell::new(0),
        }
    }
}

impl ReadProp<EdgeTag> for CountingConstant {
    type Value = f64;
    type Ref<'s>
        = Owned<f64>
    where
        Self: 's;
    const IS_CONSTANT: bool = true;
    fn get_ref(&self, _k: EdgeId) -> Owned<f64> {
        self.reads.set(self.reads.get() + 1);
        Owned(self.c)
    }
}

// ===========================================================================
// 1. The unity fast path
// ===========================================================================

/// `get_out_degree(v, g, const UnityPropertyMap&)` -- `graph_selectors.hh:182-187`.
#[test]
fn unity_is_the_plain_degree_widened() {
    let g = build();
    let u = Unity::<f64, EdgeTag>::NEW;
    for i in 0..N {
        let x = v(i);
        assert_eq!(weighted_out_degree(&g, x, &u), g.out_degree(x) as f64);
        assert_eq!(weighted_in_degree(&g, x, &u), g.in_degree(x) as f64);
        assert_eq!(weighted_degree(&g, x, &u), g.degree(x) as f64);
    }
    // The structure the rest of the file leans on.
    assert_eq!(g.out_degree(v(0)), 10);
    assert_eq!(g.in_degree(v(0)), 2, "the self-loop and 2 -> 0");
    assert_eq!(g.degree(v(0)), 12);
    assert_eq!(g.num_edges(), EDGES.len());
}

/// The fast path is a *branch that is not taken*, not a cheaper loop: a map
/// that panics on read returns an answer.
#[test]
fn the_unity_path_never_reads_the_map() {
    let g = build();
    let p = PoisonUnity;
    for i in 0..N {
        let x = v(i);
        assert_eq!(weighted_out_degree(&g, x, &p), g.out_degree(x) as f64);
        assert_eq!(weighted_in_degree(&g, x, &p), g.in_degree(x) as f64);
        assert_eq!(weighted_degree(&g, x, &p), g.degree(x) as f64);
    }
}

// ===========================================================================
// 2. The constant fast path -- defect #44
// ===========================================================================

/// `out_degree(v, g) * weight.c`, the overload graph-tool cannot instantiate.
///
/// `0.1` is not representable in binary, so `10 * 0.1` and `0.1 + ... + 0.1`
/// are *different `f64` values*. Asserting the exact product is therefore an
/// assertion that the multiplication happened.
#[test]
fn the_constant_path_multiplies_rather_than_accumulates() {
    let g = build();
    let c = Constant::<f64, EdgeTag>::new(0.1);

    // The number ten additions would have produced.
    let accumulated: f64 = std::iter::repeat_n(0.1f64, 10).sum();
    assert_ne!(accumulated, 1.0, "0.1 must not sum exactly, or this proves nothing");

    assert_eq!(weighted_out_degree(&g, v(0), &c), 1.0);
    assert_eq!(weighted_out_degree(&g, v(0), &c), 10.0 * 0.1);
    assert_ne!(weighted_out_degree(&g, v(0), &c), accumulated);

    // ... and it is `c * degree` at every vertex and in every direction.
    for i in 0..N {
        let x = v(i);
        assert_eq!(weighted_out_degree(&g, x, &c), g.out_degree(x) as f64 * 0.1);
        assert_eq!(weighted_in_degree(&g, x, &c), g.in_degree(x) as f64 * 0.1);
        assert_eq!(weighted_degree(&g, x, &c), g.degree(x) as f64 * 0.1);
    }
}

/// One read, not `degree` reads -- and none at all when the neighbourhood is
/// empty, because there is no key to read at.
#[test]
fn the_constant_path_reads_the_map_once() {
    let g = build();

    let k = CountingConstant::new(0.1);
    assert_eq!(weighted_out_degree(&g, v(0), &k), 1.0);
    assert_eq!(k.reads.get(), 1, "out_degree(0) is 10");

    let k = CountingConstant::new(2.0);
    assert_eq!(weighted_degree(&g, v(0), &k), 24.0);
    assert_eq!(k.reads.get(), 1, "degree(0) is 12");

    // Vertex 5 is a sink: no out-edges, so nothing is read and the answer is
    // zero -- `val_t d = val_t()` (`graph_selectors.hh:193`) with no loop.
    let k = CountingConstant::new(7.0);
    assert_eq!(weighted_out_degree(&g, v(5), &k), 0.0);
    assert_eq!(k.reads.get(), 0);
}

// ===========================================================================
// 3. The general loop
// ===========================================================================

/// `d += get(weight, *e)` (`graph_selectors.hh:190-197`), against a reference
/// computed from the edge list rather than from the adjacency.
#[test]
fn a_dense_map_sums_the_neighbourhood() {
    let g = build();
    let w = weights();
    let wp: EdgeProp<f64> = DenseProp::from_vec(g.graph_id(), w.clone());

    for i in 0..N {
        let x = v(i);
        assert_eq!(weighted_out_degree(&g, x, &wp), ref_out(&w, i), "out {i}");
        assert_eq!(weighted_in_degree(&g, x, &wp), ref_in(&w, i), "in {i}");
        // `total_degreeS::get_total_degree(..., std::true_type, ...)` is
        // `in_degreeS + out_degreeS` (`graph_selectors.hh:227-233`). The
        // self-loop on vertex 0 is in both, which is the point.
        assert_eq!(
            weighted_degree(&g, x, &wp),
            ref_in(&w, i) + ref_out(&w, i),
            "total {i}"
        );
    }

    // Every edge once, from each end: sum of all totals is twice the weight.
    let total: f64 = (0..N).map(|i| weighted_degree(&g, v(i), &wp)).sum();
    assert_eq!(total, 2.0 * w.iter().sum::<f64>());
}

/// The `ToF64` widening, at a magnitude `f64` still represents exactly. An
/// `Into<f64>` bound would not have compiled for `i64` at all; see
/// `tests/ui/u15_i64_weight_is_admissible.rs`.
#[test]
fn i64_weights_are_exact_beyond_two_to_the_53() {
    let g = build();
    let big: i64 = 1 << 60; // 1_152_921_504_606_846_976
    assert!(big > (1i64 << 53));

    let wp: EdgeProp<i64> = DenseProp::from_vec(g.graph_id(), vec![big; EDGES.len()]);

    // Ten out-edges of `1 << 60`: 10 * 2^60 = 2^61 * 5, three mantissa bits.
    let ten = weighted_out_degree(&g, v(0), &wp);
    assert_eq!(ten, 11_529_215_046_068_469_760.0);
    assert_eq!(ten, 10.0 * 1_152_921_504_606_846_976.0);
    // Exact, not merely close: the nearest other `f64` is 2048 away.
    assert_eq!(ten - 11_529_215_046_068_469_760.0, 0.0);

    // The same number reached by i128 arithmetic, so the claim is arithmetic
    // and not a restatement of the kernel.
    let exact = (g.out_degree(v(0)) as i128) * (big as i128);
    assert_eq!(ten, exact as f64);
    assert_eq!(ten as i128, exact, "the round trip is lossless here");

    // Three of them, where two mantissa bits suffice.
    assert_eq!(g.in_degree(v(2)), 3);
    assert_eq!(weighted_in_degree(&g, v(2), &wp), 3_458_764_513_820_540_928.0);
}

/// An empty neighbourhood is `val_t d = val_t()` -- zero, never a NaN and
/// never the previous vertex's answer.
#[test]
fn a_sink_and_a_source_are_zero() {
    let g = build();
    let w = weights();
    let wp: EdgeProp<f64> = DenseProp::from_vec(g.graph_id(), w);
    assert_eq!(weighted_out_degree(&g, v(5), &wp), 0.0);
    assert_eq!(weighted_in_degree(&g, v(3), &wp), 11.0, "edges 4 and 5");
    // A vertex that is not in the graph at all: `out_edges` is an empty range
    // (`adj/list.rs:147-152`), not an out-of-bounds read of `g._edges[v]`
    // (`graph_adjacency.hh:1110-1119`).
    assert_eq!(weighted_out_degree(&g, v(N + 3), &wp), 0.0);
    assert_eq!(weighted_degree(&g, v(N + 3), &wp), 0.0);
}

// ===========================================================================
// 4. The view algebra
// ===========================================================================

/// `get_total_degree(..., std::false_type, ...)` forwards to `out_degreeS`
/// (`graph_selectors.hh:239-244`). Here that is not a special case: `Und<G>`'s
/// out-run and all-run are the same iterator type.
#[test]
fn on_an_undirected_view_total_is_out() {
    let g = build();
    let w = weights();
    let wp: EdgeProp<f64> = DenseProp::from_vec(g.graph_id(), w.clone());
    let u = (&g).undirect();

    for i in 0..N {
        let x = v(i);
        let out = weighted_out_degree(u, x, &wp);
        assert_eq!(out, weighted_degree(u, x, &wp), "und total == und out");
        // ... and both equal the *directed* total.
        assert_eq!(out, weighted_degree(&g, x, &wp));
        assert_eq!(out, ref_in(&w, i) + ref_out(&w, i));
    }

    // Unity over the undirected view is `degree`, not `out_degree`.
    let unity = Unity::<f64, EdgeTag>::NEW;
    assert_eq!(weighted_out_degree(u, v(0), &unity), 12.0);
    assert_eq!(weighted_out_degree(&g, v(0), &unity), 10.0);
}

/// `reversed_graph` swaps the two iterator typedefs (`graph_reverse.hh:78-80`)
/// and nothing else.
#[test]
fn a_reversed_view_swaps_the_two_halves() {
    let g = build();
    let w = weights();
    let wp: EdgeProp<f64> = DenseProp::from_vec(g.graph_id(), w);
    let r = (&g).reverse();

    for i in 0..N {
        let x = v(i);
        assert_eq!(weighted_out_degree(r, x, &wp), weighted_in_degree(&g, x, &wp));
        assert_eq!(weighted_in_degree(r, x, &wp), weighted_out_degree(&g, x, &wp));
        // The total is orientation-free.
        assert_eq!(weighted_degree(r, x, &wp), weighted_degree(&g, x, &wp));
    }
}

// ===========================================================================
// 5. The bulk forms
// ===========================================================================

#[test]
fn out_degrees_fills_every_slot_and_clears_the_stale_ones() {
    let g = build();

    // A map pre-loaded with junk, which is what `get_unchecked(size = 0)`
    // (`dispatch.hh:171-177`) hands a kernel in graph-tool.
    let mut p: DenseProp<i64, _> = DenseProp::from_vec(g.graph_id(), vec![-7i64; N]);
    out_degrees(&g, &mut p);
    assert_eq!(p.as_slice(), &[10, 1, 1, 0, 0, 0]);

    // Idempotent, and the second call still clears.
    out_degrees(&g, &mut p);
    assert_eq!(p.as_slice(), &[10, 1, 1, 0, 0, 0]);
    assert_eq!(p.len(), g.vertex_bound().len());

    // An empty map is grown to the bound, not left short.
    let mut q: DenseProp<i64, _> = DenseProp::new(g.graph_id());
    out_degrees(&g, &mut q);
    assert_eq!(q.len(), N);
    assert_eq!(q.as_slice().iter().sum::<i64>(), EDGES.len() as i64);

    // On the undirected view every vertex reports its whole block.
    let mut u: DenseProp<i64, _> = DenseProp::new(g.graph_id());
    out_degrees((&g).undirect(), &mut u);
    assert_eq!(u.as_slice(), &[12i64, 3, 4, 2, 2, 1][..]);
    assert_eq!(u.as_slice().iter().sum::<i64>(), 2 * EDGES.len() as i64);
}

/// The `GraphId` comparison defect #8 costs, at the one place it is paid.
#[test]
#[should_panic(expected = "must belong to the graph")]
fn out_degrees_refuses_a_map_from_another_graph() {
    let g = build();
    let other = build();
    let mut p: DenseProp<i64, _> = DenseProp::new(other.graph_id());
    out_degrees(&g, &mut p);
}

#[test]
fn the_histogram_is_indexed_by_degree() {
    let g = build();

    // Directed: degree is in + out, and the self-loop counts twice.
    let h = degree_histogram(&g);
    let mut expect = vec![0usize; 13];
    for i in 0..N {
        expect[g.degree(v(i))] += 1;
    }
    expect.truncate(expect.iter().rposition(|&c| c > 0).unwrap() + 1);
    assert_eq!(h, expect);
    assert_eq!(h.len(), 13, "max degree is vertex 0's 12");

    // The two identities a degree histogram has to satisfy.
    assert_eq!(h.iter().sum::<usize>(), g.num_vertices());
    assert_eq!(
        h.iter().enumerate().map(|(d, c)| d * c).sum::<usize>(),
        2 * g.num_edges(),
        "the handshake lemma, counting each edge from both ends"
    );

    // The undirected view sees the same blocks, hence the same histogram.
    assert_eq!(degree_histogram((&g).undirect()), h);
}

#[test]
fn the_histogram_of_a_graph_with_no_edges() {
    let mut g = AdjList::with_vertices(4);
    assert_eq!(degree_histogram(&g), vec![4]);
    // An empty graph has no bins at all, not one bin of zero.
    let e = AdjList::with_vertices(0);
    assert!(degree_histogram(&e).is_empty());
    // One edge lifts exactly two vertices out of bin zero.
    g.add_edge(v(0), v(1)).expect("add_edge");
    assert_eq!(degree_histogram(&g), vec![2, 2]);
}

// ===========================================================================
// 6. No allocation per vertex
// ===========================================================================

/// The signatures take an *iterator-shaped* neighbourhood precisely so that no
/// kernel materialises `&[EdgeId]` per vertex. Nothing below is allowed to
/// reach the global allocator.
#[test]
fn the_kernels_allocate_nothing() {
    let g = build();
    let w = weights();
    let wp: EdgeProp<f64> = DenseProp::from_vec(g.graph_id(), w);
    let unity = Unity::<f64, EdgeTag>::NEW;
    let c = Constant::<f64, EdgeTag>::new(0.25);
    let u = (&g).undirect();

    let (sum, allocs) = allocations(|| {
        let mut acc = 0.0f64;
        for i in 0..N {
            let x = v(i);
            acc += weighted_out_degree(&g, x, &wp);
            acc += weighted_in_degree(&g, x, &wp);
            acc += weighted_degree(&g, x, &wp);
            acc += weighted_out_degree(&g, x, &unity);
            acc += weighted_in_degree(&g, x, &c);
            acc += weighted_degree(u, x, &wp);
        }
        acc
    });
    assert!(sum > 0.0);
    assert_eq!(allocs, 0, "a weighted degree must not allocate");

    // `out_degrees` allocates only to grow the map. Pre-sized, it is a scan.
    let mut p: DenseProp<i64, _> = DenseProp::new(g.graph_id());
    out_degrees(&g, &mut p);
    let ((), allocs) = allocations(|| out_degrees(&g, &mut p));
    assert_eq!(allocs, 0, "a pre-sized out_degrees must not allocate");

    // The histogram grows at most `max_degree + 1` times over the whole scan
    // -- never once per vertex. Six vertices, and far fewer than six growths.
    let (h, allocs) = allocations(|| degree_histogram(&g));
    assert_eq!(h.iter().sum::<usize>(), N);
    assert!(
        allocs < N,
        "the histogram grew {allocs} times for {N} vertices"
    );
}

// ===========================================================================
// 7. The disassembly
// ===========================================================================

/// The `Unity` monomorphisation of [`weighted_out_degree`], pinned to a symbol
/// so that a disassembler can find it.
///
/// `#[inline(never)]` keeps the frame; `no_mangle` keeps the name. The body is
/// the only thing in this file that is *not* about a value.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn u15_unity_out_degree(g: &AdjList, x: VertexId) -> f64 {
    weighted_out_degree(g, x, &Unity::<f64, EdgeTag>::NEW)
}

/// Mnemonics that can transfer control backwards on x86-64.
fn is_branch(mnemonic: &str) -> bool {
    mnemonic.starts_with('j') || mnemonic.starts_with("loop")
}

/// Parse `objdump -d` output for backward control transfers, which is what a
/// loop is at the instruction level.
///
/// Returns `None` when the disassembly could not be obtained or the
/// architecture is not one this parser understands -- the test then reports
/// and passes, because the *semantic* proof is
/// `the_unity_path_never_reads_the_map` above and this is the corroboration.
fn backward_branches(disasm: &str) -> Option<Vec<String>> {
    let mut found = Vec::new();
    let mut saw_any = false;
    for line in disasm.lines() {
        // `   1234:\tbytes...\tmnemonic operands`
        let Some((addr, rest)) = line.split_once(':') else {
            continue;
        };
        let Ok(here) = u64::from_str_radix(addr.trim(), 16) else {
            continue;
        };
        saw_any = true;
        // The instruction text is the last tab-separated column; a wrapped
        // raw-bytes continuation line has no such column and parses to a
        // mnemonic that is a hex byte, which is never a branch.
        let Some(text) = rest.rsplit('\t').next() else {
            continue;
        };
        let mut words = text.split_whitespace();
        let Some(mnemonic) = words.next() else {
            continue;
        };
        if !is_branch(mnemonic) {
            continue;
        }
        let Some(operand) = words.next() else {
            continue;
        };
        // `*%rax` and friends are indirect: not a loop back into this body.
        let Ok(target) = u64::from_str_radix(operand.trim_start_matches("0x"), 16) else {
            continue;
        };
        if target <= here {
            found.push(line.trim().to_owned());
        }
    }
    saw_any.then_some(found)
}

/// Defect #44's fast path, at the instruction level: with `Unity` there is no
/// loop to run.
#[test]
fn the_unity_monomorphisation_contains_no_loop() {
    let g = build();
    // The symbol has to be *used*, or there is nothing to disassemble.
    assert_eq!(u15_unity_out_degree(&g, v(0)), 10.0);
    assert_eq!(u15_unity_out_degree(&g, v(5)), 0.0);

    let exe = std::env::current_exe().expect("test binary path");
    let out = match std::process::Command::new("objdump")
        .arg("-d")
        .arg("--disassemble=u15_unity_out_degree")
        .arg(&exe)
        .output()
    {
        Ok(out) if out.status.success() => out,
        _ => {
            eprintln!("u15: objdump unavailable; the instruction-level check is skipped");
            return;
        }
    };
    let text = String::from_utf8_lossy(&out.stdout);
    if !text.contains("u15_unity_out_degree") {
        eprintln!("u15: symbol not in the disassembly; the check is skipped");
        return;
    }
    let Some(back) = backward_branches(&text) else {
        eprintln!("u15: unrecognised disassembly format; the check is skipped");
        return;
    };
    assert!(
        back.is_empty(),
        "the Unity monomorphisation must contain no loop, found:\n{}\n\nfull body:\n{text}",
        back.join("\n")
    );
}

// ===========================================================================
// 8. The bound is `ToF64`
// ===========================================================================

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/u15_i64_weight_is_admissible.rs");
    t.compile_fail("tests/ui/u15_string_weight_is_rejected.rs");
}
