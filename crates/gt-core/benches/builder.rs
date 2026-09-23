//! `ParBuilder` against serial `add_edge`.
//!
//! graph-tool supports concurrent mutation of a *live* graph
//! (`set_concurrent`, `graph_adjacency.hh:451`, with a per-thread `_free_idx_m`
//! at `:613` and `#pragma omp atomic g._n_edges++` at `:1214`). DESIGN.md
//! defect #51 records that `&mut AdjList` removes that capability rather than
//! fixing it, and that [`ParBuilder`](gt_core::adj::ParBuilder) is the honest
//! replacement: deterministic, chunk-ordered, and therefore giving edge ids
//! that do not depend on the thread count — which graph-tool's per-thread free
//! lists explicitly do.
//!
//! A removed capability is only honestly replaced if the replacement is
//! actually faster than the serial path it is supposed to justify. That is the
//! whole content of this file: the same edge multiset, built both ways, at
//! sizes where the answer is allowed to differ.
//!
//! The plan calls for 10^7 edges. That is the `builder/1e7` rung, and it is
//! deliberately the last one: criterion will run the smaller rungs often and
//! the large one rarely, which is the right ratio for a number that only has
//! to be re-checked when the merge changes.
//!
//! U6 owns `ParBuilder`'s body.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gt_core::adj::{AdjList, NoLookup, ParBuilder};
use gt_core::ids::VertexId;

/// Chunk count is a property of the *plan*, not of the machine: it fixes the
/// merge order and hence the edge-id assignment, so it must not be derived
/// from `available_parallelism`.
const CHUNKS: usize = 64;

/// SplitMix64, mixed with the item index so that a chunk can generate its own
/// slice of the stream without reference to any other chunk. `ParBuilder::fill`
/// hands each chunk only its index; a generator carrying mutable state across
/// chunks would reintroduce exactly the thread-order dependence the builder
/// exists to remove.
#[inline]
fn mix(seed: u64, i: u64) -> u64 {
    let mut z = seed
        .wrapping_add(i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline]
fn endpoint(seed: u64, i: u64, n: usize) -> VertexId {
    VertexId::from_index((mix(seed, i) % n as u64) as usize)
}

/// Serially, through the incremental API.
fn serial(n: usize, m: usize) -> AdjList {
    let mut g = AdjList::with_vertices(n);
    for i in 0..m as u64 {
        let s = endpoint(0x1111, 2 * i, n);
        let t = endpoint(0x1111, 2 * i + 1, n);
        g.add_edge(s, t).expect("add_edge");
    }
    g
}

/// In parallel, through the staging buffers, with the same multiset.
fn parallel(n: usize, m: usize) -> AdjList<NoLookup> {
    let mut b = ParBuilder::new(n, CHUNKS);
    let per = m.div_ceil(CHUNKS);
    b.fill(|k, out| {
        let lo = k * per;
        let hi = ((k + 1) * per).min(m);
        out.reserve(hi.saturating_sub(lo));
        for i in lo..hi {
            let i = i as u64;
            out.push((endpoint(0x1111, 2 * i, n), endpoint(0x1111, 2 * i + 1, n)));
        }
    });
    b.build(NoLookup).expect("build")
}

fn build_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("builder");
    // 1e5 and 1e6 are where the merge overhead is still visible; 1e7 is the
    // size DESIGN.md's bulk-construction claim is about.
    for &(label, n, m) in &[
        ("1e5", 25_000usize, 100_000usize),
        ("1e6", 250_000, 1_000_000),
        ("1e7", 2_500_000, 10_000_000),
    ] {
        group.throughput(Throughput::Elements(m as u64));
        group.sample_size(10);
        group.bench_with_input(BenchmarkId::new("serial", label), &(n, m), |b, &(n, m)| {
            b.iter(|| serial(n, m));
        });
        group.bench_with_input(
            BenchmarkId::new("parallel", label),
            &(n, m),
            |b, &(n, m)| {
                b.iter(|| parallel(n, m));
            },
        );
    }
    group.finish();
}

criterion_group!(builder, build_throughput);
criterion_main!(builder);
