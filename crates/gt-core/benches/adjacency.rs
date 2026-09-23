//! Adjacency storage: the numbers behind DESIGN.md section 12.
//!
//! Each group here corresponds to one line of the ledger or one row of the
//! defect table, and exists so that the line stops being an assertion:
//!
//! * `add_edge/scaling` — defect #18. `graph_adjacency.hh:1192-1215` appends
//!   and swaps; it never calls `insert`. A transliteration that reached for
//!   `Vec::insert` to keep the adjacency sorted would be quadratic and would
//!   still pass every correctness test in the tree. The six-rung ladder is
//!   what makes that visible: a `Vec::insert` doubles the per-element time
//!   every rung, i.e. 32x across the ladder.
//!
//!   The criterion is **not** that the per-element time stays flat, which an
//!   earlier version of this comment claimed and which is false: measured,
//!   it rises 53.8 -> 187.1 ns/edge from 10k to 320k, 3.5x over five
//!   doublings. That is |V| independent `Vec`s growing under random access,
//!   not an insert, and it is recorded in section 12 with the numbers.
//! * `edges/scan` — the real `AdjList` walked through `out_edges`.
//! * `entry_width` — "2x on the adjacency stream", i.e. `AdjEntry` at 8 bytes
//!   against `pair<vertex_t, vertex_t>` at 16 (section 11), with the width as
//!   the only variable. The footprint ratio is exactly 2 and is pinned by
//!   `tests/ledger_layout.rs`; the *time* ratio measured here is 1.22x on a
//!   blocked adjacency and 1.89x on a flat run, which is what section 12 now
//!   says instead of quoting the footprint ratio as a throughput.
//! * `remove_edge` / `clear_vertex` — defect #1. The C++ `clear_vertex`
//!   decrements `_n_edges` by counting `remove_if`'s *moved-from tail*
//!   (`graph_adjacency.hh:1403-1410`); the port is one loop over
//!   `remove_edge`, which is the slower shape and is measured as such.
//! * `swap_remove_vertex` — an openly stated loss: C++ patches endpoints in
//!   place (`:1489-1523`), the port unlinks and relinks each incident edge, so
//!   2–4x is the expected, accepted number.
//!
//! Bodies of the functions under measurement belong to U3 and U5; the shape of
//! the measurement belongs here.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gt_core::adj::AdjList;
use gt_core::ids::{EdgeId, VertexId};

/// SplitMix64.
///
/// The edge stream has to be bit-identical at every rung of the ladder and on
/// every machine, or the scaling curve measures the generator rather than the
/// insertion. Written out rather than taken from `rand` for the same reason
/// `par::plan::Seed` derives per chunk: a benchmark whose input depends on a
/// dependency's version is not a ledger entry.
struct Stream(u64);

impl Stream {
    const fn new(seed: u64) -> Self {
        Stream(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[inline]
    fn vertex(&mut self, n: usize) -> VertexId {
        VertexId::from_index((self.next_u64() % n as u64) as usize)
    }
}

/// `m` edges over `n` vertices, deterministically.
fn edge_stream(n: usize, m: usize, seed: u64) -> Vec<(VertexId, VertexId)> {
    let mut s = Stream::new(seed);
    (0..m).map(|_| (s.vertex(n), s.vertex(n))).collect()
}

/// A populated graph, plus the ids in insertion order.
fn populate(n: usize, edges: &[(VertexId, VertexId)]) -> (AdjList, Vec<EdgeId>) {
    let mut g = AdjList::with_vertices(n);
    let ids = edges
        .iter()
        .map(|&(s, t)| g.add_edge(s, t).expect("add_edge").id())
        .collect();
    (g, ids)
}

/// The ladder. Flat per-element time, or `add_edge` is not O(1).
fn add_edge_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_edge/scaling");
    for &m in &[10_000usize, 20_000, 40_000, 80_000, 160_000, 320_000] {
        let n = m / 4;
        let edges = edge_stream(n, m, 0x5eed_0001);
        group.throughput(Throughput::Elements(m as u64));
        group.bench_with_input(BenchmarkId::from_parameter(m), &edges, |b, edges| {
            b.iter(|| {
                let mut g = AdjList::with_vertices(n);
                for &(s, t) in edges {
                    g.add_edge(s, t).expect("add_edge");
                }
                g
            });
        });
    }
    group.finish();
}

/// Streaming the whole edge set: the 8-vs-16 byte entry claim.
fn edges_scan(c: &mut Criterion) {
    const N: usize = 50_000;
    const M: usize = 400_000;
    let edges = edge_stream(N, M, 0x5eed_0002);
    let (g, _) = populate(N, &edges);

    let mut group = c.benchmark_group("edges");
    group.throughput(Throughput::Elements(M as u64));
    group.bench_function("scan", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for v in 0..N {
                let v = VertexId::from_index(v);
                for e in g.out_edges(v) {
                    acc = acc.wrapping_add(e.other.index());
                }
            }
            black_box(acc)
        });
    });
    group.finish();
}

/// Removal, one edge at a time, in insertion order.
fn remove_edge(c: &mut Criterion) {
    const N: usize = 20_000;
    const M: usize = 80_000;
    let edges = edge_stream(N, M, 0x5eed_0003);

    let mut group = c.benchmark_group("remove_edge");
    group.throughput(Throughput::Elements(M as u64));
    group.bench_function("all", |b| {
        b.iter_batched(
            || populate(N, &edges),
            |(mut g, ids)| {
                for id in ids {
                    g.remove_edge(id).expect("remove_edge");
                }
                g
            },
            criterion::BatchSize::LargeInput,
        );
    });
    group.finish();
}

/// `clear_vertex` over the whole vertex set: defect #1's correct shape.
fn clear_vertex(c: &mut Criterion) {
    const N: usize = 20_000;
    const M: usize = 80_000;
    let edges = edge_stream(N, M, 0x5eed_0004);

    let mut group = c.benchmark_group("clear_vertex");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("all", |b| {
        b.iter_batched(
            || populate(N, &edges).0,
            |mut g| {
                for v in 0..N {
                    g.clear_vertex(VertexId::from_index(v)).expect("clear");
                }
                g
            },
            criterion::BatchSize::LargeInput,
        );
    });
    group.finish();
}

/// The stated 2–4x loss against `:1489-1523`.
fn swap_remove_vertex(c: &mut Criterion) {
    const N: usize = 20_000;
    const M: usize = 80_000;
    const K: usize = 2_000;
    let edges = edge_stream(N, M, 0x5eed_0005);

    let mut group = c.benchmark_group("swap_remove_vertex");
    group.throughput(Throughput::Elements(K as u64));
    group.bench_function("tail", |b| {
        b.iter_batched(
            || populate(N, &edges).0,
            |mut g| {
                // Remove from the back: every victim is the swap partner of
                // nothing, which isolates the unlink/relink cost from the
                // relabelling cost.
                for k in 0..K {
                    g.swap_remove_vertex(VertexId::from_index(N - 1 - k))
                        .expect("swap_remove_vertex");
                }
                g
            },
            criterion::BatchSize::LargeInput,
        );
    });
    group.finish();
}

/// The "2x on the adjacency stream" claim, with the entry width as the only
/// variable, at two working-set sizes.
///
/// Section 12's first Win is `AdjEntry` at 8 bytes against graph-tool's
/// `pair<vertex_t, vertex_t>` at 16 (`graph.hh:137` fixes `vertex_t = size_t`,
/// so `edge_list_t` at `graph_adjacency.hh:224` is a vector of 16-byte pairs).
/// Comparing `edges/scan` against a C++ build would compare two compilers, two
/// allocators and two iterator designs at once, and could not isolate the
/// width.
///
/// These mirrors can. Each pair is `Vec<Vec<_>>` with the *same* degree
/// distribution, built from the same edge stream, scanned by the same loop,
/// differing only in whether an entry is `(u32, u32)` or `(u64, u64)`.
///
/// **Three rungs, because the answer depends on the rung.** Halving the bytes
/// can only halve the time where the scan is bandwidth-bound; `l3` sits
/// inside a server L3 and `dram` well outside one, and both measured 1.22x --
/// the per-vertex indirection, not the width, is what binds. `flat` drops the
/// indirection entirely and measured 1.89x, which is the ceiling the ledger's
/// "2x" was quoting. The footprint ratio itself is pinned by
/// `tests/ledger_layout.rs::the_cxx_entry_is_two_pointers_wide`.
fn entry_width(c: &mut Criterion) {
    // (vertices, edges, label). Entries are `2 * edges`; narrow bytes are
    // `16 * edges`, wide bytes `32 * edges`.
    let rungs: [(usize, usize, &str); 2] = [
        (50_000, 400_000, "l3"),        //  6.4 MB narrow / 12.8 MB wide
        (1_000_000, 8_000_000, "dram"), //  128 MB narrow /  256 MB wide
    ];

    let mut group = c.benchmark_group("entry_width");
    group.sample_size(20);
    for &(n, m, label) in &rungs {
        let edges = edge_stream(n, m, 0x5eed_0002);
        let mut narrow: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        let mut wide: Vec<Vec<(u64, u64)>> = vec![Vec::new(); n];
        for (i, &(s, t)) in edges.iter().enumerate() {
            let (si, ti) = (s.index(), t.index());
            narrow[si].push((ti as u32, i as u32));
            narrow[ti].push((si as u32, i as u32));
            wide[si].push((ti as u64, i as u64));
            wide[ti].push((si as u64, i as u64));
        }
        drop(edges);

        group.throughput(Throughput::Elements(2 * m as u64));
        group.bench_function(BenchmarkId::new("narrow_8B", label), |b| {
            b.iter(|| {
                let mut acc = 0usize;
                for blk in &narrow {
                    for e in blk {
                        acc = acc.wrapping_add(e.0 as usize);
                    }
                }
                black_box(acc)
            });
        });
        group.bench_function(BenchmarkId::new("wide_16B", label), |b| {
            b.iter(|| {
                let mut acc = 0usize;
                for blk in &wide {
                    for e in blk {
                        acc = acc.wrapping_add(e.0 as usize);
                    }
                }
                black_box(acc)
            });
        });
    }

    // The ceiling: the same two widths as one contiguous run, with no
    // per-vertex indirection at all. Halving the bytes can buy at most what
    // this pair shows, and the blocked scans above can only approach it.
    // 16M entries: 128 MB narrow, 256 MB wide.
    const FLAT: usize = 16_000_000;
    let flat_n: Vec<(u32, u32)> = (0..FLAT as u32).map(|i| (i, i)).collect();
    let flat_w: Vec<(u64, u64)> = (0..FLAT as u64).map(|i| (i, i)).collect();
    group.throughput(Throughput::Elements(FLAT as u64));
    group.bench_function("narrow_8B/flat", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for e in &flat_n {
                acc = acc.wrapping_add(e.0 as usize);
            }
            black_box(acc)
        });
    });
    group.bench_function("wide_16B/flat", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for e in &flat_w {
                acc = acc.wrapping_add(e.0 as usize);
            }
            black_box(acc)
        });
    });

    group.finish();
}

criterion_group!(
    adjacency,
    add_edge_scaling,
    edges_scan,
    entry_width,
    remove_edge,
    clear_vertex,
    swap_remove_vertex
);
criterion_main!(adjacency);
