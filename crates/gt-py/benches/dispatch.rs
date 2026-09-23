//! What the Python boundary costs per call.
//!
//! gt_core::design section 12 claims "one `TypeId` compare at the boundary, against
//! up to three `any_cast` probes per candidate plus a try/catch on the Python
//! path". The C++ side of that comparison is `dispatch.hh`'s linear scan over
//! a Hana cartesian product, ending — when nothing matches — in a runtime
//! `DispatchNotFound` whose message is "This is a graph_tool bug. :-("
//! (`dispatch.hh:86-88`). Here the scan is an exhaustive `match` over six
//! [`ViewKind`] variants with no catch-all, so a seventh view is
//! `error[E0004]` rather than a runtime failure (defect #48).
//!
//! Two things are worth a number, and only the first is obvious:
//!
//! * `dispatch/<view>` — the per-call cost of choosing an arm. It should be
//!   indistinguishable from calling the kernel directly, and the `direct`
//!   baseline is here so that "should" is checkable.
//! * `dispatch/filtered/*` — section 12 records an accepted loss:
//!   `Filtered::new` costs an O(V + E) prepass where C++ gets O(1) by
//!   returning the wrong number (`graph_filtered.hh:316`, defect #15). Three
//!   of the six arms pay it, so the boundary's cost is not one number.
//!
//! No Python object is touched: `AnyGraph` borrows an `AdjList` and the masks,
//! so this measures dispatch and nothing else. (The target still links
//! libpython, as any binary target in this crate does; the `[lib]` section
//! keeps `cargo build --workspace` free of that, not `--benches`.)
//!
//! U31 owns `dispatch`.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use gt_core::adj::{AdjList, NoLookup};
use gt_core::graph::{EdgeList, Endpoints, GraphRef, VertexList};
use gt_core::ids::VertexId;
use gt_py::dispatch::{AnyGraph, GraphKernel, ViewKind};

const N: usize = 20_000;
const M: usize = 100_000;

fn graph(n: usize, m: usize) -> AdjList<NoLookup> {
    let mut s = 0x243f_6a88_85a3_08d3u64;
    let mut next = move || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let mut g = AdjList::with_vertices(n);
    for _ in 0..m {
        let a = VertexId::from_index((next() % n as u64) as usize);
        let b = VertexId::from_index((next() % n as u64) as usize);
        g.add_edge(a, b).expect("add_edge");
    }
    g
}

/// The smallest kernel that still forces every arm to be instantiated: it
/// touches `vertices`, `edges` and `endpoints`, i.e. the whole read surface
/// `GraphKernel` bounds on, so no arm can be optimised down to a constant.
struct Fingerprint;

impl GraphKernel for Fingerprint {
    type Out = usize;

    fn call<G>(self, g: G) -> usize
    where
        G: GraphRef + VertexList + EdgeList + Endpoints,
    {
        let mut acc = g.num_vertices();
        for e in g.edges() {
            acc = acc
                .wrapping_mul(31)
                .wrapping_add(e.source().index() ^ e.target().index());
        }
        acc
    }
}

fn unfiltered(c: &mut Criterion) {
    let g = graph(N, M);

    let mut group = c.benchmark_group("dispatch");
    group.throughput(Throughput::Elements(M as u64));
    // The floor: the same kernel, monomorphised at the call site with no
    // runtime choice at all.
    group.bench_function("direct", |b| {
        b.iter(|| black_box(Fingerprint.call(&g)));
    });
    for (name, kind) in [
        ("directed", ViewKind::Directed),
        ("undirected", ViewKind::Undirected),
        ("reversed", ViewKind::Reversed),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(
                    AnyGraph::new(&g, kind)
                        .dispatch(Fingerprint)
                        .expect("dispatch"),
                )
            });
        });
    }
    group.finish();
}

fn filtered(c: &mut Criterion) {
    let g = graph(N, M);
    // Keep four vertices in five and nine edges in ten: dense enough that the
    // prepass is the cost being measured and not the survivors.
    let vmask: Vec<u8> = (0..N).map(|i| u8::from(i % 5 != 0)).collect();
    let emask: Vec<u8> = (0..M).map(|i| u8::from(i % 10 != 0)).collect();

    let mut group = c.benchmark_group("dispatch/filtered");
    group.throughput(Throughput::Elements(M as u64));
    for (name, kind) in [
        ("directed", ViewKind::DirectedFiltered),
        ("undirected", ViewKind::UndirectedFiltered),
        ("reversed", ViewKind::ReversedFiltered),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(
                    AnyGraph::filtered(&g, kind, &vmask, &emask)
                        .dispatch(Fingerprint)
                        .expect("dispatch"),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(dispatch, unfiltered, filtered);
criterion_main!(dispatch);
