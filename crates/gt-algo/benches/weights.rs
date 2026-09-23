//! Weighted-degree selectors: the property-map axis of `gt_core::design` section 12.
//!
//! `kernels.rs` measures the *graph* axis (traversal shapes). This file
//! measures the *weight-map* axis, which is the one section 12 makes a
//! structural claim about and which `kernels.rs` never varies: every group
//! there passes `Unity::NEW`.
//!
//! `graph_selectors.hh:109-187` dispatches the same three cases at run time
//! through `get_weight`/`get_out_degree` overloads, and the unity case is
//! `UnityPropertyMap` -- an empty class whose `operator[]` returns `1`, which
//! the C++ compiler is also free to fold. The difference is *where*: there the
//! fold depends on the weight map being a concrete type at the call site;
//! here `W::IS_UNITY` is an associated const and the branch is folded at
//! monomorphisation whatever the call site looks like.
//!
//! The three arms of `weighted_out_degree` (degree.rs:122-136) are:
//!
//! * `IS_UNITY`   -- `g.out_degree(v)`, no edge touched at all;
//! * `IS_CONSTANT`-- one load plus a multiply, no per-edge load;
//! * otherwise    -- one dependent load per incident edge.
//!
//! The gap between the first and the third is the number the ledger's "the
//! `Unity` path folds away" line is worth, and it is a *memory-traffic*
//! number: a `DenseProp<f64, EdgeTag>` over M edges is 8M bytes that the
//! unity arm never reads. Reading it in `EdgeId` order against adjacency
//! order is the realistic access pattern, because `EdgeId`s are handed out in
//! insertion order and the adjacency is grouped by source.
//!
//! `bulk/out_degrees` is the scan form: one pass writing a vertex map, which
//! is where "bounds checks on scatter" (section 12, Losses) is paid.
//!
//! U15 owns the selector bodies.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use gt_algo::degree::{out_degrees, weighted_degree, weighted_out_degree};
use gt_core::adj::AdjList;
use gt_core::ids::{EdgeTag, VertexId};
use gt_core::prop::DenseProp;
use gt_core::prop::dense::{Constant, EdgeProp, Unity};

const N: usize = 100_000;
const M: usize = 500_000;

/// SplitMix64, as everywhere else in this tree: the graph under measurement
/// must not depend on a dependency version.
fn graph(n: usize, m: usize) -> AdjList {
    let mut s = 0x243f_6a88_85a3_08d3u64;
    let mut next = move || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let mut g = AdjList::with_vertices(n);
    for i in 0..n {
        let s = VertexId::from_index(i);
        let t = VertexId::from_index((i + 1) % n);
        g.add_edge(s, t).expect("add_edge");
    }
    for _ in n..m {
        let s = VertexId::from_index((next() % n as u64) as usize);
        let t = VertexId::from_index((next() % n as u64) as usize);
        g.add_edge(s, t).expect("add_edge");
    }
    g
}

/// One weight per edge, in `EdgeId` order.
fn weights(g: &AdjList) -> EdgeProp<f64> {
    let bound = g.edge_bound().len();
    let w: Vec<f64> = (0..bound).map(|i| 1.0 + ((i % 17) as f64)).collect();
    DenseProp::from_vec(g.graph_id(), w)
}

/// `sum_v weighted_out_degree(v)`, once per weight-map kind.
///
/// The same loop, the same graph, the same `f64` accumulator: the only thing
/// that varies is the type of `w`, so the difference between the three
/// numbers is the monomorphisation and nothing else.
fn out_degree(c: &mut Criterion) {
    let g = graph(N, M);
    let dense = weights(&g);
    let konst: Constant<f64, EdgeTag> = Constant::new(2.5);
    let unity: Unity<f64, EdgeTag> = Unity::NEW;

    let mut group = c.benchmark_group("weighted_out_degree");
    // One element per *edge*: the out-degree sweep visits each edge once.
    group.throughput(Throughput::Elements(M as u64));
    group.bench_function("unity", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for v in 0..N {
                acc += weighted_out_degree(&g, VertexId::from_index(v), &unity);
            }
            black_box(acc)
        });
    });
    group.bench_function("constant", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for v in 0..N {
                acc += weighted_out_degree(&g, VertexId::from_index(v), &konst);
            }
            black_box(acc)
        });
    });
    group.bench_function("dense", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for v in 0..N {
                acc += weighted_out_degree(&g, VertexId::from_index(v), &dense);
            }
            black_box(acc)
        });
    });
    group.finish();
}

/// The same three, over `all_edges`: both halves of every block.
///
/// `weighted_degree` is the selector `total_degreeS` (`graph_selectors.hh:
/// 213-246`) ports, and on a directed graph it walks twice the entries the
/// out-degree sweep does -- so the unity arm's advantage should be larger
/// here in absolute terms and the same in relative terms.
fn all_degree(c: &mut Criterion) {
    let g = graph(N, M);
    let dense = weights(&g);
    let unity: Unity<f64, EdgeTag> = Unity::NEW;

    let mut group = c.benchmark_group("weighted_degree");
    group.throughput(Throughput::Elements(2 * M as u64));
    group.bench_function("unity", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for v in 0..N {
                acc += weighted_degree(&g, VertexId::from_index(v), &unity);
            }
            black_box(acc)
        });
    });
    group.bench_function("dense", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for v in 0..N {
                acc += weighted_degree(&g, VertexId::from_index(v), &dense);
            }
            black_box(acc)
        });
    });
    group.finish();
}

/// The bulk form: one pass, one scattered store per vertex.
fn bulk(c: &mut Criterion) {
    let g = graph(N, M);
    let id = g.graph_id();

    let mut group = c.benchmark_group("degree/bulk");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("out_degrees", |b| {
        b.iter_batched_ref(
            || DenseProp::<i64, _>::new(id),
            |out| out_degrees(&g, out),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

// ===========================================================================
// Instruction-level probes for `gt_core::design` section 12.
//
// `gt-algo/tests/u15_degree.rs` disassembles a Unity monomorphisation too,
// but a test binary is built with `[profile.dev]` (opt-level 1, no LTO).
// Section 12's claims are about the *shipped* configuration, and
// `[profile.bench]` inherits `release`: opt-level 3, `lto = "fat"`,
// `codegen-units = 1`. These three symbols are the same call at the three
// weight-map monomorphisations, so `objdump --disassemble=<name>` on this
// binary counts instructions under exactly the settings the ledger describes.
//
// `#[inline(never)]` keeps the frame, `no_mangle` keeps the name, and each is
// called from `main` below so that LTO cannot drop it.
// ===========================================================================

/// `W::IS_UNITY`: the arm that reads nothing from the map.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn gtprobe_unity_out_degree(g: &AdjList, v: VertexId) -> f64 {
    weighted_out_degree(g, v, &Unity::<f64, EdgeTag>::NEW)
}

/// `W::IS_CONSTANT`: one load from the map, then `degree * c`.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn gtprobe_constant_out_degree(g: &AdjList, v: VertexId, w: &Constant<f64, EdgeTag>) -> f64 {
    weighted_out_degree(g, v, w)
}

/// The general arm: one dependent load per incident edge.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn gtprobe_dense_out_degree(g: &AdjList, v: VertexId, w: &EdgeProp<f64>) -> f64 {
    weighted_out_degree(g, v, w)
}

/// Keep the three probes alive through LTO, and check they agree.
fn probes(c: &mut Criterion) {
    let g = graph(1_000, 5_000);
    let dense = weights(&g);
    let konst = Constant::<f64, EdgeTag>::new(1.0);
    let v = VertexId::from_index(7);

    let unity = gtprobe_unity_out_degree(&g, v);
    let constant = gtprobe_constant_out_degree(&g, v, &konst);
    let sum = gtprobe_dense_out_degree(&g, v, &dense);
    assert_eq!(
        unity, constant,
        "the constant arm at c = 1 is the unity arm"
    );
    assert!(sum >= unity, "weights are >= 1 by construction");

    // One token measurement so the group is not empty and the symbols are
    // reachable from `criterion_main`.
    let mut group = c.benchmark_group("probe");
    group.bench_function("unity_one_vertex", |b| {
        b.iter(|| black_box(gtprobe_unity_out_degree(black_box(&g), black_box(v))));
    });
    group.bench_function("dense_one_vertex", |b| {
        b.iter(|| {
            black_box(gtprobe_dense_out_degree(
                black_box(&g),
                black_box(v),
                black_box(&dense),
            ))
        });
    });
    group.finish();
}

criterion_group!(weights_bench, out_degree, all_degree, bulk, probes);
criterion_main!(weights_bench);
