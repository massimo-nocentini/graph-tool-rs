//! Work partitioning and reproducible seeding.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::ops::Range;

/// A fixed partition of `0..n` into chunks.
///
/// The chunk count is a function of `n` and the grain **only**. It must never
/// depend on the thread-pool size: it is what fixes both the floating-point
/// fold order and the per-chunk RNG stream, so a run with one worker and a run
/// with sixteen produce bit-identical results.
///
/// The cost is real and is stated rather than hidden: a fixed partition
/// forgoes rayon's adaptive splitting, so on a ragged workload -- blockmodel
/// `virtual_move` cost scales with vertex degree -- load balance is worse than
/// OpenMP's `schedule(runtime)` (`parallel_util.hh:405`). Choose a small grain
/// to recover most of it.
///
/// ## The partition is balanced, not grain-exact
///
/// `chunks()` is `n.div_ceil(grain)`, but the items are then spread *evenly*
/// over that many chunks rather than cut into `grain`-sized pieces with a
/// remainder: every chunk holds `n / chunks` or `n / chunks + 1` items, and
/// never more than `grain`. Both schemes are exact covers; the balanced one is
/// chosen for two reasons.
///
/// * Load balance is the one thing a fixed partition gives up (section 8), so
///   it should not also hand one worker a chunk of 1 while its neighbours hold
///   `grain` -- which is exactly what `k * grain .. min((k+1) * grain, n)`
///   does whenever `grain` barely divides `n` (`n = 1_000_001`, `grain =
///   1_000` gives a final chunk of one item).
/// * `k * (n / chunks) + min(k, n % chunks)` is bounded by `n` at every step,
///   whereas `(k + 1) * grain` can overflow `usize` for a large grain on a
///   64-bit index. There is no branch to get wrong because there is no
///   saturating case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Plan {
    n: usize,
    chunks: usize,
}

impl Plan {
    /// Partition `0..n` into chunks of about `grain` items.
    ///
    /// The empty plan still has one (empty) chunk, so `chunks()` is never `0`
    /// and a fold over `0..chunks()` is never a fold over nothing.
    ///
    /// # Panics
    ///
    /// If `grain == 0`, as `slice::chunks` does: a zero grain has no meaning
    /// to silently repair, and the alternative -- treating it as `1` -- turns
    /// an arithmetic bug in the caller into `n` chunks of one item, which is
    /// slow rather than loud.
    pub fn new(n: usize, grain: usize) -> Self {
        assert!(grain > 0, "Plan::new: grain must be non-zero");
        Plan {
            n,
            chunks: n.div_ceil(grain).max(1),
        }
    }

    /// Total item count.
    #[inline]
    pub const fn len(&self) -> usize {
        self.n
    }
    /// Whether there is nothing to do.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }
    /// Number of chunks.
    #[inline]
    pub const fn chunks(&self) -> usize {
        self.chunks
    }

    /// The `k`th chunk. The chunks are an exact cover of `0..n`.
    ///
    /// # Panics
    ///
    /// If `k >= chunks()`. Returning an empty range for an out-of-range chunk
    /// would make a fold that miscounts its own chunks silently produce a
    /// *plausible* number, and the whole point of the type is that the number
    /// is reproducible for a reason.
    #[inline]
    pub fn range(&self, k: usize) -> Range<usize> {
        assert!(
            k < self.chunks,
            "Plan::range: chunk {k} out of range (chunks = {})",
            self.chunks
        );
        let base = self.n / self.chunks;
        let rem = self.n % self.chunks;
        // `k * base + min(k, rem)`: the first `rem` chunks carry one extra
        // item. Both terms are bounded by `n`, so neither can overflow.
        let start = k * base + if k < rem { k } else { rem };
        let end = start + base + usize::from(k < rem);
        start..end
    }
}

/// Root entropy for a run.
#[derive(Clone, Copy, Debug)]
pub struct Seed(pub [u8; 32]);

impl Seed {
    /// Derive the generator for one chunk.
    ///
    /// Keyed on the *work-item* index, never on a thread id, so the stream a
    /// chunk sees is a function of the [`Plan`] alone.
    ///
    /// Uses a keyed derivation rather than XORing `k` into the seed bytes:
    /// XOR makes `(S, k)` and `(S ^ k, 0)` alias, which is harmless within one
    /// run and a weaker construction for no gain.
    ///
    /// ## What graph-tool does instead
    ///
    /// `parallel_rng.hh:38-42` fills a per-generator vector with copies of the
    /// caller's generator and calls `set_stream(get_rng_stream())` on each,
    /// where `get_rng_stream` (`random.cc:46-50`) is a **mutex-guarded global
    /// counter**: which stream a copy gets is its position in a global arrival
    /// order, not a property of the work. `get(tnum)` (`:56-61`) then indexes
    /// `_rngs[tnum - 1]`, so the stream a piece of work draws from is its
    /// OpenMP thread number and the results move with `OMP_NUM_THREADS`. The
    /// backing map (`:65-69, :73`) is keyed on the *address* of the caller's
    /// generator and is never evicted, so a later generator allocated at a
    /// recycled address inherits the dead one's streams.
    ///
    /// Here there is no global, no cache and no thread number: `k` is the
    /// chunk index from a [`Plan`], and the same `(Seed, k)` is the same
    /// stream in every process, at every thread count, forever.
    ///
    /// ## The construction
    ///
    /// ChaCha8 is its own PRF. The root seed is the key, `k` is the 64-bit
    /// stream (nonce), and the first 32 bytes of that keystream are the
    /// derived seed the returned generator is built from -- i.e. `derived =
    /// H(self.0, k)` with `H` a keyed function, so nothing an adversary or a
    /// bug does to `k` can be undone by a matching change to the seed.
    ///
    /// The second stage is not ceremony. Returning the keyed generator
    /// directly would hand every chunk a generator whose key *is* the root
    /// seed, so recovering one chunk's state would recover the root and with
    /// it every sibling stream; re-seeding from the derived bytes keeps the
    /// chunks independent in both directions. It also leaves each returned
    /// generator at stream 0, word position 0, so a caller that saves and
    /// restores an RNG position sees the same shape of state for every chunk.
    /// The cost is one ChaCha8 block per *chunk*, never per item.
    pub fn split(self, k: u64) -> ChaCha8Rng {
        let mut kdf = ChaCha8Rng::from_seed(self.0);
        kdf.set_stream(k);
        let mut derived = [0u8; 32];
        kdf.fill_bytes(&mut derived);
        ChaCha8Rng::from_seed(derived)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The in-crate copy of the cover check: `tests/u10_plan.rs` runs the
    /// exhaustive sweep from outside, this one keeps running if the public
    /// surface ever moves.
    #[test]
    fn chunks_cover_exactly() {
        for n in 0..64usize {
            for grain in 1..16usize {
                let p = Plan::new(n, grain);
                let mut next = 0;
                for k in 0..p.chunks() {
                    let r = p.range(k);
                    assert_eq!(r.start, next);
                    assert!(r.end >= r.start);
                    next = r.end;
                }
                assert_eq!(next, n);
            }
        }
    }

    #[test]
    fn empty_plan_has_one_empty_chunk() {
        let p = Plan::new(0, 8);
        assert_eq!(p.chunks(), 1);
        assert_eq!(p.range(0), 0..0);
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn a_huge_grain_does_not_overflow() {
        let p = Plan::new(usize::MAX, usize::MAX);
        assert_eq!(p.chunks(), 1);
        assert_eq!(p.range(0), 0..usize::MAX);
    }

    #[test]
    #[should_panic(expected = "grain must be non-zero")]
    fn zero_grain_panics() {
        let _ = Plan::new(10, 0);
    }
}
