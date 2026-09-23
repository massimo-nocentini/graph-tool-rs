//! The SBM move cycle: record, price, commit.
//!
//! `entropy.rs` benches the *terms* -- `eterm`, `vterm`, the cache lookups,
//! the description lengths -- and says in its own header that `sparse_ds` is
//! not measured there because pricing needs a real `Delta` and only the
//! recording lifecycle can mint one. This file is where that number lives.
//!
//! The three groups are the three halves of `move_vertex`
//! (`blockmodel/state.hh:726-786`) as this port splits them:
//!
//! * `sbm/record` -- `modify_entries` (`entries.hh:322`) +
//!   `get_move_entries` (`state.hh:1062`). One scan of `v`'s incidence,
//!   writing `(r, s, delta)` triples and the before-image into a `DeltaBuf`
//!   whose storage is reused across moves.
//! * `sbm/sparse_ds` -- `entries_dS` (`state.hh:1224-1255`), the hottest
//!   kernel in the library and the one D10 is *for*. The delta is recorded
//!   once, outside the timing loop, so this group is the pricing arithmetic
//!   and nothing else: a contiguous `&[Entry<W>]` walk with both `mrs_before`
//!   values already interned, against the C++'s two cache-cold dependent
//!   loads per entry into `_mrs`.
//! * `sbm/move_vertex` -- the whole cycle, `K` moves per iteration, on a
//!   state reseeded outside the timing loop: record, price, commit, reseat.
//!   `reseat` is the `_b[v] = r` half of `add_partition_node`
//!   (`state.hh:779`) that `commit` structurally cannot do, because
//!   `MoveHeader` carries `r` and `nr` and no `VertexId`.
//!
//! One `Workspace` is created per *iteration* of the sweep and reused across
//! all `K` moves inside it, which is the real call shape (D9: one scratch per
//! rayon worker, never a `Vec` indexed by thread id). `DeltaBuf::begin`
//! (`delta/buf.rs:149`) drains and clears rather than freeing, and only grows
//! the field table, so after the first move of a sweep the cycle allocates
//! nothing -- which is the section 12 claim this file exists to price.
//!
//! The graph is deliberately larger than the fixtures in `tests/u2*`: `_mrs`
//! at `B = 64` is 4096 cells, so the before-image it replaces is genuinely
//! cache-resident, and the win D10 claims is *not* an artefact of a hot
//! matrix. `SLOTS` and `N` are what set the entry count per move; the average
//! degree sets the scan length.
//!
//! U22/U23 own the pricing, U24 the state, U25 the recorder.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gt_core::adj::AdjList;
use gt_core::dir::Directed;
use gt_core::ids::VertexId;
use gt_inference::blockmodel::{BlockCommit, BlockState, Cache, EntropyParams, record, sparse_ds};
use gt_inference::delta::{MoveKey, Workspace};
use gt_inference::ids::Group;

const N: usize = 20_000;
const M: usize = 100_000;
const SLOTS: usize = 64;
/// Moves per timed iteration of the full cycle.
const K: usize = 2_000;

/// SplitMix64.
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
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn grp(i: usize) -> Group {
    Group::new(i as u32).expect("group index in range")
}

fn vid(i: usize) -> VertexId {
    VertexId::from_index(i)
}

/// A graph with a ring spine plus `M - N` random edges, one in five of the
/// random ones a self-loop -- the shape both `dkin`/`dkout` halves handle
/// specially.
fn graph() -> AdjList {
    let mut s = Stream::new(0x5EED_5B30);
    let mut g = AdjList::with_vertices(N);
    for i in 0..N {
        g.add_edge(vid(i), vid((i + 1) % N)).expect("add_edge");
    }
    for _ in N..M {
        let a = s.below(N);
        let b = if s.below(5) == 0 { a } else { s.below(N) };
        g.add_edge(vid(a), vid(b)).expect("add_edge");
    }
    g
}

/// The initial partition: `v mod SLOTS`.
fn partition() -> Vec<u32> {
    (0..N).map(|i| (i % SLOTS) as u32).collect()
}

/// A state holding exactly `b`, seeded from `g`.
fn seed(g: &AdjList, b: &[u32]) -> BlockState<Directed, i64> {
    let mut st = BlockState::<Directed, i64>::new(SLOTS, b.len());
    for (i, &r) in b.iter().enumerate() {
        st.assign(vid(i), grp(r as usize), 1);
    }
    for e in g.edges() {
        st.seed_pair(
            grp(b[e.source().index()] as usize),
            grp(b[e.target().index()] as usize),
            1,
        );
    }
    st
}

fn params() -> EntropyParams {
    EntropyParams {
        deg_corr: true,
        ..EntropyParams::default()
    }
}

/// The scan half: one pass over `v`'s incidence into a reused buffer.
fn recording(c: &mut Criterion) {
    let g = graph();
    let b = partition();
    let st = seed(&g, &b);
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let mut s = Stream::new(0xD317_A001);

    let mut group = c.benchmark_group("sbm/record");
    group.throughput(Throughput::Elements(1));
    group.bench_function("one_move", |bch| {
        bch.iter(|| {
            let v = s.below(N);
            let nr = s.below(SLOTS);
            let mv = MoveKey {
                from: Some(grp(b[v] as usize)),
                to: Some(grp(nr)),
            };
            let nb = |u: VertexId| Group::new(b[u.index()]);
            let rec = record(&st, &g, vid(v), mv, &nb, SLOTS, 1i64, &mut ws);
            black_box(rec.seal().level(0).entries().len())
        });
    });
    group.finish();
}

/// `entries_dS` alone, over deltas of a known entry count.
///
/// Each rung records one move *outside* the timing loop and then prices that
/// same delta repeatedly, so the number is the pricing walk with the entry
/// count on the x-axis. A contiguous scan is linear in it with a flat
/// per-entry cost; a shape that re-derived `mrs` per entry would not be.
fn pricing(c: &mut Criterion) {
    let g = graph();
    let b = partition();
    let st = seed(&g, &b);
    let ea = params();
    let cache = Cache::build(1 << 16);

    // Vertices chosen for their degree, so the rungs differ in entry count
    // rather than in luck. Degree is `2 * M / N = 10` on average; the ring
    // spine guarantees every vertex has at least two.
    let mut by_degree: Vec<(usize, usize)> = (0..N).map(|v| (g.degree(vid(v)), v)).collect();
    by_degree.sort_unstable();
    let picks = [
        by_degree[0].1,
        by_degree[N / 2].1,
        by_degree[N - N / 100].1,
        by_degree[N - 1].1,
    ];

    let mut group = c.benchmark_group("sbm/sparse_ds");
    for &v in &picks {
        let mut ws = Workspace::<Directed, i64>::with_levels(1);
        let nr = (b[v] as usize + 1) % SLOTS;
        let mv = MoveKey {
            from: Some(grp(b[v] as usize)),
            to: Some(grp(nr)),
        };
        let nb = |u: VertexId| Group::new(b[u.index()]);
        let t = record(&st, &g, vid(v), mv, &nb, SLOTS, 1i64, &mut ws).seal();
        let delta = t.level(0);
        let n_entries = delta.entries().len();

        group.throughput(Throughput::Elements(n_entries as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_entries),
            &delta,
            |bch, delta| {
                bch.iter(|| black_box(sparse_ds(*delta, ea, &cache)));
            },
        );
    }
    group.finish();
}

/// The whole cycle: record, price, commit, reseat -- `K` times.
fn move_vertex(c: &mut Criterion) {
    let g = graph();
    let b0 = partition();
    let ea = params();
    let cache = Cache::build(1 << 16);

    let mut group = c.benchmark_group("sbm/move_vertex");
    group.throughput(Throughput::Elements(K as u64));
    group.sample_size(20);
    group.bench_function("sweep", |bch| {
        bch.iter_batched(
            || (seed(&g, &b0), b0.clone(), Stream::new(0xD317_C0DE)),
            |(mut st, mut b, mut s)| {
                // One workspace for the whole sweep: `begin` drains and
                // clears, so only the first move of the sweep can allocate.
                let mut ws = Workspace::<Directed, i64>::with_levels(1);
                let mut acc = 0.0f64;
                for _ in 0..K {
                    let v = s.below(N);
                    let nr = s.below(SLOTS);
                    let mv = MoveKey {
                        from: Some(grp(b[v] as usize)),
                        to: Some(grp(nr)),
                    };
                    let nb = |u: VertexId| Group::new(b[u.index()]);
                    let t = record(&st, &g, vid(v), mv, &nb, SLOTS, 1i64, &mut ws).seal();
                    acc += sparse_ds(t.level(0), ea, &cache);
                    let _ = st.commit(t.into_levels().next().expect("one level"));
                    st.reseat(vid(v), grp(nr));
                    b[v] = nr as u32;
                }
                black_box(acc)
            },
            criterion::BatchSize::LargeInput,
        );
    });
    group.finish();
}

// ===========================================================================
// An instruction-level probe for DESIGN.md section 12.
//
// The ledger's "Wins" list says the before-image removes "two cache-cold
// dependent loads per entry from `entries_dS`, the hottest kernel in the
// library". That is a statement about the shape of one loop, and
// `[profile.bench]` (which inherits `release`: opt-level 3, `lto = "fat"`,
// `codegen-units = 1`) is the configuration it is a statement about.
//
// `objdump --disassemble=gtprobe_sparse_ds` on this binary shows the loop:
// what it loads per entry, whether the entry walk is a contiguous stride, and
// whether `__rust_alloc` appears anywhere in the body.
// ===========================================================================

/// `entries_dS` (`state.hh:1224-1255`) at one monomorphisation.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn gtprobe_sparse_ds(
    d: gt_inference::delta::Delta<'_, Directed, i64>,
    ea: EntropyParams,
    c: &Cache,
) -> f64 {
    sparse_ds(d, ea, c)
}

/// Keep the probe alive through LTO.
fn probe(c: &mut Criterion) {
    let g = graph();
    let b = partition();
    let st = seed(&g, &b);
    let ea = params();
    let cache = Cache::build(1 << 16);
    let mut ws = Workspace::<Directed, i64>::with_levels(1);

    let v = 7usize;
    let nr = (b[v] as usize + 1) % SLOTS;
    let mv = MoveKey {
        from: Some(grp(b[v] as usize)),
        to: Some(grp(nr)),
    };
    let nb = |u: VertexId| Group::new(b[u.index()]);
    let t = record(&st, &g, vid(v), mv, &nb, SLOTS, 1i64, &mut ws).seal();
    let delta = t.level(0);
    assert!(gtprobe_sparse_ds(delta, ea, &cache).is_finite());

    let mut group = c.benchmark_group("probe");
    group.bench_function("sparse_ds_one_delta", |bch| {
        bch.iter(|| black_box(gtprobe_sparse_ds(black_box(delta), ea, black_box(&cache))));
    });
    group.finish();
}

criterion_group!(sbm, recording, pricing, move_vertex, probe);
criterion_main!(sbm);
