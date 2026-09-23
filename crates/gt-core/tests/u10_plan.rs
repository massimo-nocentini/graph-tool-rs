//! U10 — the fixed partition and the chunk-keyed RNG, from outside the crate.
//!
//! Two claims are under test and neither is a property of the code in the
//! usual sense; both are properties of what the code *refuses* to consult.
//!
//! 1. `Plan`'s cover is a function of `(n, grain)`. graph-tool's equivalent is
//!    `#pragma omp for schedule(runtime)` (`parallel_util.hh:400`), whose
//!    partition is `OMP_SCHEDULE` and the thread count, so the association
//!    order of every reduction downstream of it is the interleaving
//!    (`merge_split.hh:1131-1142`, defect #27).
//! 2. `Seed::split` is keyed on the work-item index. graph-tool keys on the
//!    OpenMP thread number — `_rngs[tnum - 1]` (`parallel_rng.hh:56-61`) over a
//!    vector whose streams were handed out by a mutex-guarded global counter
//!    in arrival order (`random.cc:46-50`), cached in a map keyed on the
//!    *address* of the caller's generator and never evicted
//!    (`parallel_rng.hh:65-69, :73`), so a reused address inherits a dead
//!    object's streams (defect #28).
//!
//! The environment-independence claim is checked the only way it can honestly
//! be checked: by re-running this binary as a child process under different
//! values of `RAYON_NUM_THREADS` and comparing digests. Mutating the
//! environment in-process is `unsafe` in edition 2024 and this crate is
//! `forbid(unsafe_code)`; it would also be a lie, since rayon reads the
//! variable once when its global pool is built.

use std::collections::HashSet;
use std::env;
use std::process::Command;

use rand::RngCore;
use rand_chacha::ChaCha8Rng;

use gt_core::par::{Plan, Seed};

// ===========================================================================
// 1. The cover
// ===========================================================================

/// The acceptance check: for every `n` in `0..300` and `grain` in `1..40`,
/// `(0..chunks).flat_map(range)` *is* `0..n` — no gaps, no overlaps, nothing
/// dropped and no empty chunk carrying work that never runs.
#[test]
fn chunks_are_an_exact_cover() {
    for n in 0..300usize {
        for grain in 1..40usize {
            let plan = Plan::new(n, grain);
            let cover: Vec<usize> = (0..plan.chunks()).flat_map(|k| plan.range(k)).collect();
            let expect: Vec<usize> = (0..n).collect();
            assert_eq!(cover, expect, "cover mismatch for n = {n}, grain = {grain}");

            // Stated separately from the flattened equality, because a
            // flattened comparison cannot tell `0..3, 3..3, 3..5` from
            // `0..3, 3..5`: the first has a chunk that will be handed a
            // generator and produce a partial for no items.
            for k in 0..plan.chunks() {
                let r = plan.range(k);
                if n > 0 {
                    assert!(
                        !r.is_empty(),
                        "empty chunk {k} for n = {n}, grain = {grain}"
                    );
                }
                assert!(
                    r.end - r.start <= grain,
                    "chunk {k} exceeds the grain for n = {n}, grain = {grain}"
                );
            }
            assert_eq!(plan.len(), n);
            assert_eq!(plan.is_empty(), n == 0);
        }
    }
}

/// The chunks differ in size by at most one item. A grain-exact partition
/// (`k * grain .. min((k + 1) * grain, n)`) also covers, but leaves a tail of
/// `n % grain`; on `n = 1_000_001, grain = 1_000` that tail is a single item
/// while its 1000 siblings are full, and load balance is precisely what a
/// fixed partition has already spent.
#[test]
fn chunk_sizes_differ_by_at_most_one() {
    for n in 0..300usize {
        for grain in 1..40usize {
            let plan = Plan::new(n, grain);
            let sizes: Vec<usize> = (0..plan.chunks()).map(|k| plan.range(k).len()).collect();
            let lo = *sizes.iter().min().unwrap();
            let hi = *sizes.iter().max().unwrap();
            assert!(
                hi - lo <= 1,
                "ragged partition for n = {n}, grain = {grain}: {sizes:?}"
            );
        }
    }
}

/// `chunks()` is `n.div_ceil(grain)`, floored at one so that a fold over
/// `0..chunks()` is never a fold over nothing.
#[test]
fn chunk_count_is_div_ceil() {
    for n in 0..300usize {
        for grain in 1..40usize {
            assert_eq!(Plan::new(n, grain).chunks(), n.div_ceil(grain).max(1));
        }
    }
}

#[test]
fn the_empty_plan_is_one_empty_chunk() {
    let plan = Plan::new(0, 7);
    assert_eq!(plan.chunks(), 1);
    assert_eq!(plan.range(0), 0..0);
    assert!(plan.is_empty());
}

#[test]
fn a_grain_larger_than_n_is_one_chunk() {
    let plan = Plan::new(5, 1_000);
    assert_eq!(plan.chunks(), 1);
    assert_eq!(plan.range(0), 0..5);
}

#[test]
#[should_panic(expected = "grain must be non-zero")]
fn a_zero_grain_is_rejected() {
    let _ = Plan::new(10, 0);
}

/// An out-of-range chunk is a caller bug, not an empty range: a reduction that
/// miscounts its own chunks must fail, not quietly return a number that looks
/// like an answer.
#[test]
#[should_panic(expected = "out of range")]
fn an_out_of_range_chunk_is_rejected() {
    let plan = Plan::new(10, 4);
    let _ = plan.range(plan.chunks());
}

/// `Plan` is `Copy` and compares by value, so a plan can be a field of a
/// parameter struct (`gt_algo::centrality::PowerIteration`) and be handed to
/// every worker without a `clone`.
#[test]
fn plans_are_values() {
    let a = Plan::new(100, 8);
    let b = a;
    assert_eq!(a, b);
    assert_ne!(a, Plan::new(100, 9));
    assert_ne!(a, Plan::new(101, 8));
}

// ===========================================================================
// 2. Independence from the thread pool
// ===========================================================================

/// A digest of the whole observable partition *and* the stream each chunk
/// draws: 8 bytes from every chunk's generator, folded in chunk order.
///
/// Computed through rayon so the value is produced by however many workers the
/// environment gave us — if any part of `Plan`/`Seed` consulted the pool, this
/// is where it would show.
fn partition_digest() -> u64 {
    use rayon::prelude::*;

    let seed = Seed([0x5a; 32]);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for (n, grain) in [(1usize, 1usize), (997, 13), (65_536, 4_096), (0, 8)] {
        let plan = Plan::new(n, grain);
        let rows: Vec<(usize, usize, [u8; 8])> = (0..plan.chunks())
            .into_par_iter()
            .map(|k| {
                let r = plan.range(k);
                let mut rng = seed.split(k as u64);
                let mut bytes = [0u8; 8];
                rng.fill_bytes(&mut bytes);
                (r.start, r.end, bytes)
            })
            .collect();
        h = fnv(h, &(plan.chunks() as u64).to_le_bytes());
        for (start, end, bytes) in rows {
            h = fnv(h, &(start as u64).to_le_bytes());
            h = fnv(h, &(end as u64).to_le_bytes());
            h = fnv(h, &bytes);
        }
    }
    h
}

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Same `(n, grain)`, same plan — including inside rayon pools of one and of
/// sixteen threads, which is the in-process half of the claim.
#[test]
fn the_plan_does_not_move_with_the_pool_size() {
    let reference: Vec<(usize, usize, usize)> = (1..40)
        .map(|grain| {
            let p = Plan::new(997, grain);
            (grain, p.chunks(), p.range(p.chunks() - 1).end)
        })
        .collect();

    for threads in [1usize, 3, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        let got: Vec<(usize, usize, usize)> = pool.install(|| {
            (1..40)
                .map(|grain| {
                    let p = Plan::new(997, grain);
                    (grain, p.chunks(), p.range(p.chunks() - 1).end)
                })
                .collect()
        });
        assert_eq!(
            got, reference,
            "plan changed inside a {threads}-thread pool"
        );
    }
}

/// The out-of-process half. `RAYON_NUM_THREADS` is read by rayon exactly once,
/// when its global pool is built, so the only honest way to vary it is a fresh
/// process. Child runs print the digest; all of them must agree with the
/// parent's.
///
/// This is the test graph-tool cannot pass: `parallel_rng.hh:56-61` indexes
/// `_rngs[tnum - 1]`, so the same work under a different `OMP_NUM_THREADS`
/// draws from a different stream.
#[test]
fn the_digest_does_not_move_with_rayon_num_threads() {
    let here = partition_digest();
    let exe = env::current_exe().expect("test binary path");

    for threads in ["1", "2", "7", "16"] {
        let out = Command::new(&exe)
            .args([
                "--exact",
                "child_prints_the_partition_digest",
                "--ignored",
                "--nocapture",
            ])
            .env("RAYON_NUM_THREADS", threads)
            .output()
            .expect("re-running the test binary");
        assert!(
            out.status.success(),
            "child with RAYON_NUM_THREADS={threads} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout
            .lines()
            .find_map(|l| l.strip_prefix("U10-DIGEST="))
            .unwrap_or_else(|| panic!("no digest from the child; stdout was:\n{stdout}"));
        let got: u64 = line.trim().parse().expect("digest is a u64");
        assert_eq!(
            got, here,
            "partition digest moved under RAYON_NUM_THREADS={threads}"
        );
    }
}

/// Not a test; the child half of the one above. `#[ignore]` keeps it out of an
/// ordinary run, and it is invoked by name with `--ignored`.
#[test]
#[ignore = "child process of the_digest_does_not_move_with_rayon_num_threads"]
fn child_prints_the_partition_digest() {
    println!("U10-DIGEST={}", partition_digest());
}

// ===========================================================================
// 3. Seeding
// ===========================================================================

/// First `n` bytes of a chunk's stream.
fn prefix(rng: &mut ChaCha8Rng, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    rng.fill_bytes(&mut v);
    v
}

/// Pairwise distinct for `k in 0..1024`. Checked on a 64-byte prefix, which is
/// two ChaCha blocks: a collision on 64 bytes is a collision in the derivation,
/// not a coincidence.
#[test]
fn split_streams_are_pairwise_distinct() {
    let seed = Seed([0x11; 32]);
    let mut seen: HashSet<Vec<u8>> = HashSet::with_capacity(1024);
    for k in 0..1024u64 {
        let p = prefix(&mut seed.split(k), 64);
        assert!(seen.insert(p), "stream {k} collided with an earlier chunk");
    }
    assert_eq!(seen.len(), 1024);
}

/// Distinct *seeds* must also separate, at the same chunk index: the
/// derivation is keyed, so the root seed is not merely an offset.
#[test]
fn distinct_seeds_separate_at_the_same_chunk() {
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    for s in 0..=255u8 {
        let mut bytes = [0u8; 32];
        bytes[0] = s;
        assert!(seen.insert(prefix(&mut Seed(bytes).split(7), 64)));
    }
}

/// Reproducible within a run: `split` is a pure function of `(Seed, k)`, and
/// `Seed` is `Copy` so calling it does not consume anything.
#[test]
fn split_is_a_pure_function() {
    let seed = Seed([0x7e; 32]);
    for k in [0u64, 1, 2, 63, 1023, u64::MAX] {
        let a = prefix(&mut seed.split(k), 128);
        let b = prefix(&mut seed.split(k), 128);
        assert_eq!(a, b, "split({k}) is not reproducible");
    }
}

/// Reproducible *across runs*, which is the claim that actually matters and
/// the one an in-process comparison cannot make. These bytes are the
/// construction's fingerprint: ChaCha8 keyed with the root seed, stream = k,
/// first 32 bytes re-seeding the returned generator. A change to any of that
/// changes published results, so it has to be a deliberate edit to this
/// literal and not a silent refactor.
#[test]
fn split_matches_its_golden_vector() {
    let seed = Seed([0x2a; 32]);
    assert_eq!(prefix(&mut seed.split(0), 16), GOLDEN_0);
    assert_eq!(prefix(&mut seed.split(1), 16), GOLDEN_1);
    assert_eq!(prefix(&mut seed.split(u64::MAX), 16), GOLDEN_MAX);
}

const GOLDEN_0: [u8; 16] = [
    46, 68, 172, 149, 77, 241, 16, 142, 42, 143, 167, 32, 8, 96, 165, 109,
];
const GOLDEN_1: [u8; 16] = [
    170, 45, 217, 25, 3, 40, 11, 24, 112, 149, 181, 255, 238, 136, 227, 67,
];
const GOLDEN_MAX: [u8; 16] = [
    11, 110, 193, 201, 236, 56, 203, 96, 86, 249, 74, 71, 47, 174, 141, 244,
];

/// `(S, k)` and `(S ^ k, 0)` do **not** alias.
///
/// This is the whole reason the derivation is keyed rather than
/// `Seed(s ^ k_bytes)`. Under XOR the two are the same generator, so two
/// different runs — one that split the root seed into chunks, one that was
/// handed a "different" seed — silently share a stream. Nothing in a
/// reduction's output would show it.
#[test]
fn a_seed_xor_k_is_not_the_same_stream_as_chunk_k() {
    let base = [0x33u8; 32];
    // From 1: `(S, 0)` and `(S ^ 0, 0)` are the same generator by definition,
    // and that is not the aliasing anyone was worried about.
    for k in 1..256u64 {
        let via_chunk = prefix(&mut Seed(base).split(k), 64);

        // The natural XOR encodings: little-endian into the first eight
        // bytes, and big-endian into the last eight. Neither may alias.
        let mut le = base;
        for (dst, src) in le.iter_mut().zip(k.to_le_bytes()) {
            *dst ^= src;
        }
        let mut be = base;
        for (dst, src) in be[24..].iter_mut().zip(k.to_be_bytes()) {
            *dst ^= src;
        }

        assert_ne!(
            via_chunk,
            prefix(&mut Seed(le).split(0), 64),
            "k = {k} (le)"
        );
        assert_ne!(
            via_chunk,
            prefix(&mut Seed(be).split(0), 64),
            "k = {k} (be)"
        );
    }
}

/// The derived generator starts at the beginning of its own stream, so a
/// caller that saves and restores an RNG position sees the same shape of state
/// for every chunk — and no chunk is a *continuation* of another, which is the
/// failure mode of handing out copies of one generator at different word
/// positions (`parallel_rng.hh:38-42` copies `rng` and only then re-streams).
#[test]
fn every_chunk_starts_at_the_beginning_of_its_stream() {
    let seed = Seed([0x91; 32]);
    for k in [0u64, 5, 12_345] {
        let rng = seed.split(k);
        assert_eq!(rng.get_stream(), 0);
        assert_eq!(rng.get_word_pos(), 0);
    }
}

/// A chunk's stream must not be a shifted copy of its neighbour's: the first
/// block of chunk `k` may not appear anywhere in the first 16 blocks of chunk
/// `j`. A plain counter-offset derivation would fail this; a keyed one cannot
/// be made to.
#[test]
fn streams_do_not_overlap_by_a_shift() {
    let seed = Seed([0x4d; 32]);
    let marks: Vec<[u8; 16]> = (0..8u64)
        .map(|k| {
            let mut m = [0u8; 16];
            seed.split(k).fill_bytes(&mut m);
            m
        })
        .collect();

    for j in 0..8u64 {
        let long = prefix(&mut seed.split(j), 64 * 16);
        for (k, mark) in marks.iter().enumerate() {
            if k as u64 == j {
                continue;
            }
            assert!(
                !long.windows(16).any(|w| w == mark),
                "stream {j} contains the head of stream {k}"
            );
        }
    }
}

/// The plan and the seed compose the way `det_reduce` composes them: chunk `k`
/// gets `range(k)` and `split(k)`, and that pairing is what must be stable.
#[test]
fn chunk_index_pairs_the_range_with_the_stream() {
    let seed = Seed([0xc0; 32]);
    let plan = Plan::new(1_000, 64);
    let pairs: Vec<(usize, u32)> = (0..plan.chunks())
        .map(|k| (plan.range(k).start, seed.split(k as u64).next_u32()))
        .collect();

    // Re-derived from a *differently built* plan with the same (n, grain):
    // nothing is memoised, nothing is stateful, so the pairing is reproduced.
    let again = Plan::new(1_000, 64);
    for (k, &(start, word)) in pairs.iter().enumerate() {
        assert_eq!(again.range(k).start, start);
        assert_eq!(seed.split(k as u64).next_u32(), word);
    }
}
