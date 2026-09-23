//! The tabulated special functions and the entropy terms built on them.
//!
//! `inference/support/cache.hh:55-57` declares three **global mutable**
//! `std::vector<double>` tables; `init_cache` (`:83-95`) resizes them at `:90`
//! and fills them inside a `#pragma omp parallel for` at `:91-93` while
//! `get_cached` (`:73-81`) reads them at `:80`, and `get_cached` is annotated
//! `[[gnu::const]]` at `:74` — a promise to the optimiser that it reads no
//! memory, which licenses hoisting across the very resize that races with it
//! (defect #34).
//!
//! (This file previously cited `:84-92` and `:72-73`, one or two off. The
//! offsets above were re-read out of the 3.8 tree and now agree with the
//! module doc on `blockmodel::cache`, which owns the reconciliation.)
//!
//! The port's answer is one immutable `Sync` [`Cache`] threaded explicitly.
//! That is unambiguously more correct; it is *not* automatically as fast,
//! because the C++ version's lie is exactly the kind an optimiser rewards.
//! These groups are where that trade is priced rather than asserted:
//!
//! * `cache/*` — the table lookups themselves, in range and out of range. The
//!   out-of-range arms matter because the SBM deliberately feeds `lgamma`
//!   out-of-domain values (`cache.hh:36-48` installs an `ignore_error`
//!   policy), so the fallback is on the real path, not an error path.
//! * `terms/*` — `eterm` and `vterm` at both directedness instantiations. `D`
//!   is a type parameter, so the directed/undirected factor-of-two is folded
//!   at monomorphisation where `entropy.hh:38` branches on a runtime flag.
//! * `dl/partition_dl` — O(B) per evaluation, and what an `audit-full`
//!   recompute spends its time in; and `dl/edges_dl`, its O(1) sibling, at
//!   both directedness instantiations. (`edges_dl` was unreachable from here
//!   until `blockmodel/mod.rs` re-exported it; the omission was recorded
//!   against U22 in this header and has been fixed.)
//!
//! Not benched here: `sparse_ds`. Pricing needs a real `Delta`, which only the
//! recording lifecycle can mint (U19/U20), so that number belongs to a bench
//! written against the SBM state and not to this file. That bench now exists:
//! `benches/sbm.rs`, which records a move, prices it, commits it, and carries
//! the `#[inline(never)]` probe section 12's before-image entry is measured
//! through.
//!
//! U22 owns `Cache`; U23 owns the terms.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use gt_core::dir::{Directed, Undirected};
use gt_inference::blockmodel::{Cache, edges_dl, eterm, partition_dl, vterm};

/// Large enough that the tables do not fit in L1, which is the regime the
/// sweep actually runs in.
const TABLE: usize = 1 << 16;

fn lookups(c: &mut Criterion) {
    let cache = Cache::build(TABLE);
    // A stride that is coprime with the table length, so the access pattern is
    // neither sequential (which would measure the prefetcher) nor random
    // (which would measure the TLB).
    let xs: Vec<f64> = (0..4096u64)
        .map(|i| ((i * 9973) % TABLE as u64) as f64)
        .collect();
    // Beyond the table: the `boost::math::lgamma` fallback path.
    let big: Vec<f64> = (0..4096u64)
        .map(|i| (TABLE as f64) + (i as f64) * 3.5)
        .collect();

    let mut group = c.benchmark_group("cache");
    group.throughput(Throughput::Elements(xs.len() as u64));
    group.bench_function("lgamma1p/tabulated", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for &x in &xs {
                acc += cache.lgamma1p(x);
            }
            black_box(acc)
        });
    });
    group.bench_function("lgamma1p/fallback", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for &x in &big {
                acc += cache.lgamma1p(x);
            }
            black_box(acc)
        });
    });
    group.bench_function("safelog", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for &x in &xs {
                acc += cache.safelog(x);
            }
            black_box(acc)
        });
    });
    group.bench_function("lbinom", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for &x in &xs {
                acc += cache.lbinom(x + 64.0, 32.0);
            }
            black_box(acc)
        });
    });
    group.finish();
}

fn terms(c: &mut Criterion) {
    let cache = Cache::build(TABLE);
    const B: usize = 256;

    let mut group = c.benchmark_group("terms");
    group.throughput(Throughput::Elements((B * B) as u64));
    group.bench_function("eterm/directed", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for r in 0..B {
                for s in 0..B {
                    acc += eterm::<Directed>(r, s, ((r * s) % 97 + 1) as f64, &cache);
                }
            }
            black_box(acc)
        });
    });
    group.bench_function("eterm/undirected", |b| {
        b.iter(|| {
            let mut acc = 0.0f64;
            for r in 0..B {
                for s in 0..B {
                    acc += eterm::<Undirected>(r, s, ((r * s) % 97 + 1) as f64, &cache);
                }
            }
            black_box(acc)
        });
    });
    group.finish();

    let mut group = c.benchmark_group("terms/vterm");
    group.throughput(Throughput::Elements(B as u64));
    for &deg_corr in &[false, true] {
        let name = if deg_corr { "deg_corr" } else { "plain" };
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut acc = 0.0f64;
                for r in 0..B {
                    let mrp = (r % 53 + 1) as f64;
                    acc += vterm::<Directed>(mrp, mrp + 1.0, (r % 11 + 1) as f64, deg_corr, &cache);
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

fn description_lengths(c: &mut Criterion) {
    let cache = Cache::build(TABLE);
    const B: usize = 1_024;
    const N: usize = 100_000;
    let sizes: Vec<usize> = (0..B).map(|r| N / B + (r % 7)).collect();

    let mut group = c.benchmark_group("dl");
    group.throughput(Throughput::Elements(B as u64));
    group.bench_function("partition_dl", |b| {
        b.iter(|| black_box(partition_dl(&sizes, N, &cache)));
    });
    group.finish();

    // `edges_dl` is O(1): two `lbinom` arguments and one lookup. It is grouped
    // per element so that its cost can be read against `partition_dl`'s O(B).
    let mut group = c.benchmark_group("dl/edges_dl");
    group.throughput(Throughput::Elements(1));
    group.bench_function("directed", |b| {
        b.iter(|| black_box(edges_dl::<Directed>(B, N as f64, &cache)));
    });
    group.bench_function("undirected", |b| {
        b.iter(|| black_box(edges_dl::<Undirected>(B, N as f64, &cache)));
    });
    group.finish();
}

criterion_group!(entropy, lookups, terms, description_lengths);
criterion_main!(entropy);
