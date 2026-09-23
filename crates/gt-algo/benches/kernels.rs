//! gt-algo kernels, measured through the crate edge.
//!
//! gt_core::design section 10 justifies `lto = "fat"` and `codegen-units = 1` in the
//! release profile on the grounds that the hot paths cross crate boundaries by
//! construction: every kernel here is generic over gt-core's `GraphRef`, and
//! without cross-crate inlining each `Incident` manufacture is a call. That
//! claim is only testable from a bench that lives in *this* crate and calls
//! into gt-core, which is what these do. (`cargo bench` uses `[profile.bench]`,
//! which inherits `release`, so the LTO configuration under test is the one
//! shipped.)
//!
//! The groups are chosen to cover the three shapes the trait layer
//! distinguishes:
//!
//! * incidence-only, over a directed view — `bfs`, `shortest_distances`;
//! * incidence over an *undirected* view — `components`, where D2's anchored
//!   [`Incident`](gt_core::adj::Incident) does the work that
//!   `graph_adjacency.hh:1102-1108`'s whole-block `out_edge_iterator` does;
//! * predecessor-using, bounded on `Bidirectional` — `pagerank`, which cannot
//!   be called on an undirected view at all, where
//!   `graph_adaptor.hh:219-227` would have returned an empty range and a
//!   silently wrong answer (defect #12).
//!
//! `pagerank` also carries a [`Plan`](gt_core::par::Plan): its damping sum is
//! folded in chunk order, so the number below includes the cost of being
//! reproducible across thread counts, which `#pragma omp critical` at
//! `merge_split.hh:1140` does not pay and does not deliver.
//!
//! U13–U17 own the kernel bodies.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use gt_algo::centrality::{PowerIteration, pagerank};
use gt_algo::components::components;
use gt_algo::degree::degree_histogram;
use gt_algo::traversal::{Control, Visitor, bfs, shortest_distances};
use gt_core::adj::{AdjList, Incident};
use gt_core::ids::VertexId;
use gt_core::par::Plan;
use gt_core::prop::DenseProp;
use gt_core::prop::dense::Unity;
use gt_core::view::Undirect;

const N: usize = 100_000;
const M: usize = 500_000;

/// SplitMix64: the graph under test must not depend on a dependency version.
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
    // A ring first, so the graph is connected and the traversals actually
    // traverse: on a pure G(n, m) at m/n = 5 the giant component is large but
    // a benchmark should not depend on how large.
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

/// Counts, and asks for nothing. The cheapest visitor there is, so the number
/// is the traversal's and not the callback's.
#[derive(Default)]
struct Counter {
    discovered: usize,
    examined: usize,
}

impl Visitor for Counter {
    #[inline]
    fn discover(&mut self, _v: VertexId, _depth: usize) -> Control {
        self.discovered += 1;
        Control::Continue
    }
    #[inline]
    fn examine(&mut self, _from: VertexId, _e: Incident) -> Control {
        self.examined += 1;
        Control::Continue
    }
}

fn traversal(c: &mut Criterion) {
    let g = graph(N, M);
    let id = g.graph_id();

    let mut group = c.benchmark_group("traversal");
    group.throughput(Throughput::Elements(M as u64));
    group.bench_function("bfs", |b| {
        b.iter(|| {
            let mut vis = Counter::default();
            bfs(&g, VertexId::from_index(0), &mut vis).expect("bfs");
            black_box(vis.examined)
        });
    });
    group.bench_function("shortest_distances", |b| {
        b.iter_batched_ref(
            || DenseProp::<i64, _>::new(id),
            |dist| {
                shortest_distances(&g, VertexId::from_index(0), dist).expect("sssp");
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn undirected(c: &mut Criterion) {
    let g = graph(N, M);
    let id = g.graph_id();

    let mut group = c.benchmark_group("undirected");
    group.throughput(Throughput::Elements(M as u64));
    group.bench_function("components", |b| {
        b.iter_batched_ref(
            || DenseProp::<i64, _>::new(id),
            |label| black_box(components((&g).undirect(), label)),
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("degree_histogram", |b| {
        b.iter(|| black_box(degree_histogram((&g).undirect()).len()));
    });
    group.finish();
}

fn bidirectional(c: &mut Criterion) {
    let g = graph(N, M);
    let id = g.graph_id();
    let params = PowerIteration {
        epsilon: 1e-6,
        max_iter: 20,
        // Fixed grain: the fold order is the plan's, not the pool's.
        plan: Plan::new(N, 4_096),
    };

    let mut group = c.benchmark_group("bidirectional");
    group.throughput(Throughput::Elements(M as u64));
    group.sample_size(20);
    group.bench_function("pagerank", |b| {
        b.iter_batched_ref(
            || DenseProp::<f64, _>::new(id),
            |rank| black_box(pagerank(&g, &Unity::NEW, 0.85, params, rank)),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(kernels, traversal, undirected, bidirectional);
criterion_main!(kernels);
