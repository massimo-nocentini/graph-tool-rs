//! Reductions with a fixed association order.

use rayon::prelude::*;

use rand_chacha::ChaCha8Rng;
use std::ops::Range;

use super::plan::{Plan, Seed};

/// Map each chunk, then fold the per-chunk results **in chunk order**.
///
/// The fold order is a property of the [`Plan`], so the result is
/// bit-identical across thread counts. graph-tool cannot offer this:
/// `merge_split.hh:1140` reduces under `#pragma omp critical`.
///
/// ## Shape, and why it is this shape
///
/// `map` is called exactly `plan.chunks()` times, once per chunk of the
/// [`Plan`]'s cover, with the generator [`Seed::split`] derives from the
/// **chunk index**. Both are functions of the `Plan` alone, so neither the
/// number of workers rayon happens to have nor the order in which it steals
/// can be observed in the result:
///
/// * the partials land in a `Vec<T>` indexed by chunk — rayon's indexed
///   `collect` is position-preserving, not completion-ordered;
/// * the fold is a plain sequential `fold` over that `Vec`, left to right
///   from `identity`, so the association order is `((id ⊕ p0) ⊕ p1) ⊕ …`
///   whatever the interleaving was.
///
/// `reduce` is therefore **not** required to be associative, commutative or
/// `Send`: it runs on the calling thread. That is deliberate. `log_sum_exp`
/// (`support/util.hh:79-90`) is neither associative nor commutative in
/// `f64`, and `merge_split.hh:1131-1142` folds it under
/// `#pragma omp critical` — the association order there *is* the thread
/// interleaving, which is defect #27.
///
/// One `Vec` of `plan.chunks()` elements is allocated per call. That is the
/// price of the fixed fold order and it is paid once per reduction, never in
/// an inner loop.
pub fn det_reduce<T, M, R>(plan: Plan, seed: Seed, map: M, identity: T, reduce: R) -> T
where
    T: Send,
    M: Fn(Range<usize>, &mut ChaCha8Rng) -> T + Sync + Send,
    R: Fn(T, T) -> T,
{
    let partials: Vec<T> = (0..plan.chunks())
        .into_par_iter()
        .map(|k| {
            let mut rng = seed.split(k as u64);
            map(plan.range(k), &mut rng)
        })
        .collect();

    partials.into_iter().fold(identity, reduce)
}

/// As [`det_reduce`], but each chunk may fail.
///
/// The error is a **return value**. `parallel_loop_spawn`'s shared
/// `std::exception_ptr` (`parallel_util.hh:441-443`), written by every thread
/// in the region, has no expressible counterpart: there is no shared mutable
/// slot in this signature. Which error wins is the lowest chunk index, so even
/// the failure is reproducible, unlike OpenMP's "whichever thread got there".
///
/// ## Every chunk runs
///
/// A failing chunk does not cancel the others, and that is the point rather
/// than an oversight. `parallel_loop_no_spawn<true>`
/// (`parallel_util.hh:399-437`) sets a thread-private `skip` flag so the
/// *rest of that thread's* iterations are abandoned while other threads run
/// on — which iterations get skipped is the schedule, so the set of work
/// actually performed is not reproducible either. Here the cover is executed
/// in full and the lowest-indexed `Err` is selected afterwards, which costs
/// the tail of a failing run and buys a result that is a function of
/// `(plan, seed, map)` and nothing else.
///
/// Rayon's own `collect::<Result<Vec<_>, _>>()` cannot be used for this: it
/// short-circuits and keeps whichever error was stored first, i.e. the one
/// the interleaving chose — the same defect in a different language.
pub fn try_det_reduce<T, E, M, R>(
    plan: Plan,
    seed: Seed,
    map: M,
    identity: T,
    reduce: R,
) -> Result<T, E>
where
    T: Send,
    E: Send,
    M: Fn(Range<usize>, &mut ChaCha8Rng) -> Result<T, E> + Sync + Send,
    R: Fn(T, T) -> T,
{
    let partials: Vec<Result<T, E>> = (0..plan.chunks())
        .into_par_iter()
        .map(|k| {
            let mut rng = seed.split(k as u64);
            map(plan.range(k), &mut rng)
        })
        .collect();

    // Sequential, left to right: the first `Err` encountered is the one with
    // the lowest chunk index, and the successful prefix is folded in exactly
    // the order `det_reduce` would have folded it.
    let mut acc = identity;
    for partial in partials {
        acc = reduce(acc, partial?);
    }
    Ok(acc)
}

/// Multi-accumulator summation over a slice.
///
/// ## The cost nobody else costed (DESIGN.md section 12)
///
/// `#pragma omp parallel for reduction(+:S)` (e.g. `potts/spec.hh:133, :143`)
/// *licenses reassociation*, so GCC may vectorise the accumulation. Rust's
/// `f64 +=` does not, and measurement confirms it: a plain serial scan emits
/// `addsd`, never `addpd`. Five of the six source designs presented
/// entropy-sum performance as parity-or-better without mentioning this.
///
/// This function is where the port pays it back: `LANES` independent
/// accumulators folded at the end, which restores instruction-level
/// parallelism.
///
/// ## What the ledger records, measured
///
/// An earlier reading of §12 also claimed a four-accumulator hand-unroll
/// emits only `addsd`. It does not, and §12 no longer says so: on this
/// toolchain `chunked_sum::<4>` compiles to `addpd`. There is no
/// contradiction with the first half of the paragraph — a *serial* fold
/// genuinely cannot vectorise, because packing it would reassociate it — but
/// `LANES` independent accumulators fed from consecutive slots are a packed
/// add *at the same association order*, so the SLP vectoriser is allowed to
/// take it and does. SIMD is recovered here without any of the licence
/// `reduction(+:S)` needs.
///
/// That changes the size of the loss, not its existence: graph-tool still
/// gets to reassociate a loop this one may not, and explicit lane control
/// waits for `core::simd`.
/// `tests/u11_reduce.rs::the_ledger_records_whether_chunked_sum_vectorises`
/// asserts only what is guaranteed (a serial scan never packs) and prints
/// what it found, because a ledger entry that tests itself is the only kind
/// that stays true.
///
/// ## Exact association order
///
/// Lane `i` accumulates `xs[i], xs[i + LANES], xs[i + 2*LANES], …` in index
/// order; the `xs.len() % LANES` trailing elements go to lanes `0 ..r` in the
/// same way; the lanes are then folded `0.0 + a0 + a1 + … ` in lane order.
/// This is a pure function of `(xs, LANES)`, so unlike `reduction(+:S)` it
/// cannot change with the thread count, the schedule or the compiler's mood.
/// `chunked_sum::<1>` is therefore bit-identical to the naive left-to-right
/// scan, by construction.
///
/// ## Accuracy
///
/// Each lane commits at most `⌈n / LANES⌉` roundings and the final fold adds
/// `LANES - 1` more, so against an exactly-rounded sum the error is bounded by
/// roughly `(n / LANES + LANES) · ε · Σ|xᵢ|` — i.e. `LANES` accumulators are
/// not merely faster than one, they are *more* accurate, because each
/// dependency chain is shorter. The `u11_reduce` test pins this against a
/// Kahan sum. It is not, and does not claim to be, compensated summation:
/// callers needing that should say so.
///
/// Panics at compile time (monomorphisation) if `LANES == 0`.
#[inline]
pub fn chunked_sum<const LANES: usize>(xs: &[f64]) -> f64 {
    const {
        assert!(LANES > 0, "chunked_sum needs at least one accumulator");
    }

    let mut acc = [0.0f64; LANES];

    // `as_chunks`, not `chunks_exact`: the block type is `&[f64; LANES]`, so
    // the inner zip is over two arrays of statically equal length. That is
    // what lets the whole body unroll into LANES independent dependency
    // chains with no bounds check and no loop-carried dependence between
    // lanes -- and, as the ledger note above records, lets the SLP vectoriser
    // pack them.
    let (blocks, rest) = xs.as_chunks::<LANES>();
    for block in blocks {
        for (a, &x) in acc.iter_mut().zip(block) {
            *a += x;
        }
    }
    for (a, &x) in acc.iter_mut().zip(rest) {
        *a += x;
    }

    acc.iter().fold(0.0, |s, &a| s + a)
}

/// A partial sum carrying its chunk index, so a parallel reduction can be
/// folded back in a fixed order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkSum {
    /// Which chunk produced this.
    pub chunk: usize,
    /// The partial sum.
    pub sum: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LANES == 1` is the definition of the naive scan, bit for bit.
    #[test]
    fn one_lane_is_the_naive_scan() {
        let xs: Vec<f64> = (0..1000)
            .map(|i| ((i * 7919) % 101) as f64 * 1e-3)
            .collect();
        let naive = xs.iter().fold(0.0f64, |a, &x| a + x);
        assert_eq!(chunked_sum::<1>(&xs).to_bits(), naive.to_bits());
    }

    /// The remainder is not dropped, and it lands in lane order.
    #[test]
    fn the_remainder_is_summed_into_the_low_lanes() {
        // 7 = 1 full block of 4 plus a remainder of 3.
        let xs = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
        // lane0 = 1 + 16, lane1 = 2 + 32, lane2 = 4 + 64, lane3 = 8.
        assert_eq!(chunked_sum::<4>(&xs), 127.0);
        assert_eq!(chunked_sum::<16>(&xs), 127.0);
    }

    #[test]
    fn an_empty_slice_sums_to_positive_zero() {
        assert_eq!(chunked_sum::<4>(&[]).to_bits(), 0.0f64.to_bits());
    }

    /// Lane assignment must be positional, so a shift changes nothing about
    /// the *count* of terms in each lane when the length is a multiple.
    #[test]
    fn a_full_cover_uses_every_lane_equally() {
        let xs: Vec<f64> = (0..64).map(|i| i as f64).collect();
        assert_eq!(chunked_sum::<4>(&xs), 63.0 * 64.0 / 2.0);
        assert_eq!(chunked_sum::<8>(&xs), 63.0 * 64.0 / 2.0);
    }
}
