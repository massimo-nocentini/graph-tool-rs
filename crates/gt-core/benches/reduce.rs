//! Deterministic reductions: what determinism costs.
//!
//! `gt_core::design` section 12 states a loss and five of the six source designs did
//! not: `#pragma omp parallel for reduction(+:S)` (`potts/spec.hh:133, :143`)
//! *licenses* GCC to reassociate and vectorise the accumulation, and Rust's
//! `f64 +=` does not. A plain scan emits `addsd`, never `addpd`.
//!
//! [`chunked_sum`](gt_core::par::chunked_sum) is the payback: `LANES`
//! independent accumulators restore instruction-level parallelism, and on
//! this toolchain the SLP vectoriser packs them as well -- without any
//! reassociation licence, because independent accumulators fed from
//! consecutive slots are a packed add at the *same* association order. The
//! result still does not depend on how the work was split. (An earlier
//! version of this comment said "without restoring SIMD"; the probes below
//! measure `addpd` in both the standalone `-O` and the `[profile.bench]`
//! builds, and section 12 carries both counts.)
//!
//! The `lanes` group is the sweep that says which `LANES` is worth having --
//! a claim about a constant generic parameter is empty without it. Measured,
//! four lanes is 3.61x the strict fold and eight is 3.74x, so four is the
//! knee.
//!
//! The `det_reduce` group measures the other half: a chunk-ordered fold over
//! `plan.chunks()` partials, against graph-tool's `#pragma omp critical`
//! reduction at `merge_split.hh:1140`, whose result is the thread interleaving
//! (defect #27).
//!
//! U10 owns `Plan`/`Seed`; U11 owns `det_reduce` and `chunked_sum`.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gt_core::par::reduce::chunked_sum;
use gt_core::par::{Plan, Seed, det_reduce};

/// Deterministic, and deliberately not near-uniform: a summand distribution
/// with a wide exponent range is where association order actually changes the
/// answer, so it is the one worth timing.
fn summands(n: usize) -> Vec<f64> {
    let mut s = 0x243f_6a88_85a3_08d3u64;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let mantissa = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            let scale = 1.0 + ((i % 17) as f64);
            mantissa * scale
        })
        .collect()
}

// ===========================================================================
// Instruction-level probes for `gt_core::design` section 12's floating-point
// paragraph.
//
// `gt-core/tests/u11_reduce.rs::the_ledger_records_whether_chunked_sum_vectorises`
// already disassembles both forms on every test run, but a test binary is
// built with `[profile.dev]` (opt-level 1). `[profile.bench]` inherits
// `release`: opt-level 3, `lto = "fat"`, `codegen-units = 1` -- the shipped
// configuration, and the one the ledger's `addsd` / `addpd` counts are a
// statement about. These symbols let `objdump --disassemble=<name>` on this
// binary count the instruction mix under exactly those settings.
// ===========================================================================

/// A strict left fold: no reassociation licence, so no packing is legal.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn gtprobe_scan(xs: &[f64]) -> f64 {
    xs.iter().fold(0.0f64, |a, &x| a + x)
}

/// Four independent accumulators at the same association order.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn gtprobe_chunked4(xs: &[f64]) -> f64 {
    chunked_sum::<4>(xs)
}

/// Eight.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn gtprobe_chunked8(xs: &[f64]) -> f64 {
    chunked_sum::<8>(xs)
}

/// The `LANES` sweep.
fn lanes(c: &mut Criterion) {
    const N: usize = 1 << 20;
    let xs = summands(N);

    let mut group = c.benchmark_group("chunked_sum/lanes");
    group.throughput(Throughput::Elements(N as u64));
    // A naive scan is the baseline the ledger's claim is relative to.
    group.bench_function("scan", |b| {
        // `black_box` on the input too, so the baseline and the `chunked_sum`
        // arms below see the same opacity. Without it the two arms differ in
        // what the optimiser is allowed to assume about `xs`, and the ratio
        // measures that difference as well as the association order.
        b.iter(|| gtprobe_scan(black_box(&xs)));
    });
    group.bench_function(BenchmarkId::from_parameter(1), |b| {
        b.iter(|| chunked_sum::<1>(black_box(&xs)));
    });
    group.bench_function(BenchmarkId::from_parameter(2), |b| {
        b.iter(|| chunked_sum::<2>(black_box(&xs)));
    });
    group.bench_function(BenchmarkId::from_parameter(4), |b| {
        b.iter(|| gtprobe_chunked4(black_box(&xs)));
    });
    group.bench_function(BenchmarkId::from_parameter(8), |b| {
        b.iter(|| gtprobe_chunked8(black_box(&xs)));
    });
    group.finish();
}

/// Chunk-ordered parallel fold, swept over grain size.
///
/// Section 8 records that fixed chunking loses to `schedule(runtime)` on ragged
/// workloads and that a small grain recovers most of it. This is where "most"
/// gets a value.
fn ordered_fold(c: &mut Criterion) {
    const N: usize = 1 << 20;
    let xs = summands(N);
    let seed = Seed([0x2a; 32]);

    let mut group = c.benchmark_group("det_reduce/grain");
    group.throughput(Throughput::Elements(N as u64));
    for &grain in &[1_024usize, 8_192, 65_536] {
        group.bench_with_input(BenchmarkId::from_parameter(grain), &grain, |b, &grain| {
            b.iter(|| {
                det_reduce(
                    Plan::new(N, grain),
                    seed,
                    |range, _rng| chunked_sum::<4>(&xs[range]),
                    0.0f64,
                    |a, b| a + b,
                )
            });
        });
    }
    group.finish();
}

criterion_group!(reduce, lanes, ordered_fold);
criterion_main!(reduce);
