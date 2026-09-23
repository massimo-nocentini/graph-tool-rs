//! U11 — deterministic reductions, checked through the public surface.
//!
//! Three defects from `gt_core::design` §14 are on trial here and each one is a
//! *numerical* claim, so each test compares bit patterns or errors rather than
//! asserting that something compiled.
//!
//! * **#27** — `merge_split.hh:1131-1142` accumulates `log_sum_exp` under
//!   `#pragma omp critical (get_move_prob)`. The critical section serialises
//!   the update but it does not order it, so the association order of a
//!   non-associative `f64` operation is whichever thread reached the region
//!   first. Two runs of the same binary, same input, same `OMP_NUM_THREADS`,
//!   disagree in the low bits. Here the fold order is `Plan`'s chunk order and
//!   nothing else, so the answer is bit-identical at 1, 4 and 16 workers.
//! * **#26** — `parallel_loop_spawn` (`parallel_util.hh:438-446`) declares
//!   `std::exception_ptr eptr{}` *outside* `#pragma omp parallel` and assigns
//!   `eptr = loop()` *inside*: every thread in the region writes the same
//!   refcounted handle, unsynchronised, and the surviving exception is a race.
//!   [`try_det_reduce`] returns the error of the **lowest chunk index**.
//! * **§12's stated loss** — `#pragma omp parallel for reduction(+:S)`
//!   (`potts/spec.hh:133, :143`) licenses GCC to reassociate; Rust's `f64 +=`
//!   does not. The last test *records* what the compiler actually emitted for
//!   `chunked_sum::<4>` rather than assuming the ledger is still true.
//!
//! Every test that involves threads runs the same computation in private
//! rayon pools of 1, 4 and 16 workers. One worker is not a degenerate case
//! here, it is the control: it is the only configuration in which the OpenMP
//! original is also deterministic.

use std::path::PathBuf;
use std::process::Command;

use rand::RngCore;
use rayon::ThreadPoolBuilder;

use gt_core::par::reduce::chunked_sum;
use gt_core::par::{ChunkSum, Plan, Seed, det_reduce, try_det_reduce};

/// The worker counts every threaded test is run at.
const THREADS: [usize; 3] = [1, 4, 16];

/// Run `f` inside a private rayon pool of exactly `n` workers.
///
/// A private pool, not `RAYON_NUM_THREADS`: the global pool is built once per
/// process and these tests share a process with every other test in the
/// binary. A claim about thread counts that can only be checked by setting an
/// environment variable is a claim that never gets checked.
fn at_threads<R: Send>(n: usize, f: impl Fn() -> R + Sync + Send) -> R {
    ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("could not build a rayon pool")
        .install(f)
}

// ===========================================================================
// 0. The operation under test
// ===========================================================================

/// Port of `log_sum_exp(T a, T b)`, `support/util.hh:76-90`, with
/// `handle_inf = true` (the default there).
///
/// The `a == b` branch exists for `-inf + -inf`: without it, `exp(b - a)` is
/// `exp(NaN)` and the identity element poisons the fold. C++ marks it
/// `[[unlikely]]`; the equality test is kept exactly, not relaxed to an
/// `is_infinite` test, because `a + ln 2` is also what C++ returns for two
/// equal *finite* operands and the two implementations must agree bit for bit
/// on that path too.
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == b {
        // handles infinity
        return a + std::f64::consts::LN_2;
    }
    if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

/// `N` log-weights with a wide exponent range.
///
/// Near-uniform summands would make this test pass for the wrong reason: the
/// association order of `log_sum_exp` is only observable when the terms differ
/// by enough that `log1p(exp(-d))` lands in different binades. The spread is
/// what gives the bit-identity assertion teeth, and
/// [`the_fold_order_is_observable_at_all`] proves it has them.
fn log_weights(n: usize) -> Vec<f64> {
    let mut rng = Seed([0x5a; 32]).split(0);
    (0..n)
        .map(|_| {
            let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            -60.0 * u + 5.0
        })
        .collect()
}

/// The reduction itself: fold a chunk with `log_sum_exp`, fold the chunks with
/// `log_sum_exp`. `-inf` is the identity, which is why `log_sum_exp` must
/// handle two equal infinities.
fn lse_reduce(xs: &[f64], plan: Plan) -> f64 {
    det_reduce(
        plan,
        Seed([7; 32]),
        |range, _rng| {
            xs[range]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, log_sum_exp)
        },
        f64::NEG_INFINITY,
        log_sum_exp,
    )
}

// ===========================================================================
// 1. Defect #27 — the acceptance test
// ===========================================================================

/// 10 000 values, three thread counts, one bit pattern.
#[test]
fn log_sum_exp_is_bit_identical_across_thread_counts() {
    let xs = log_weights(10_000);
    let plan = Plan::new(xs.len(), 128);
    assert!(
        plan.chunks() > 16,
        "the test is vacuous unless there are more chunks than workers"
    );

    let mut seen: Option<u64> = None;
    for n in THREADS {
        let bits = at_threads(n, || lse_reduce(&xs, plan)).to_bits();
        match seen {
            None => seen = Some(bits),
            Some(first) => assert_eq!(
                bits, first,
                "log_sum_exp over 10 000 values differed at {n} workers \
                 ({bits:#x} vs {first:#x}); this is exactly defect #27, \
                 merge_split.hh:1140"
            ),
        }
    }

    // Repeating inside one pool must also not drift: #27's C++ form varies
    // run to run at a *fixed* thread count, because `#pragma omp critical`
    // serialises without ordering.
    let bits = seen.unwrap();
    for _ in 0..8 {
        assert_eq!(at_threads(4, || lse_reduce(&xs, plan)).to_bits(), bits);
    }
    assert!(bits != f64::NEG_INFINITY.to_bits() && !f64::from_bits(bits).is_nan());
}

/// The previous test would pass trivially if the fold order made no
/// difference. It does: folding the same partials right-to-left changes the
/// answer's low bits, which is precisely the quantity `#pragma omp critical`
/// leaves to the scheduler.
#[test]
fn the_fold_order_is_observable_at_all() {
    let xs = log_weights(10_000);
    let plan = Plan::new(xs.len(), 128);

    let partials: Vec<f64> = (0..plan.chunks())
        .map(|k| {
            xs[plan.range(k)]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, log_sum_exp)
        })
        .collect();

    let forward = partials
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, log_sum_exp);
    let backward = partials
        .iter()
        .rev()
        .copied()
        .fold(f64::NEG_INFINITY, log_sum_exp);

    assert_eq!(forward.to_bits(), lse_reduce(&xs, plan).to_bits());
    assert_ne!(
        forward.to_bits(),
        backward.to_bits(),
        "the summands are too tame for this test to mean anything"
    );
    // ...but only in the low bits: this is a reproducibility defect, not an
    // accuracy one, and overstating it would be the same sin as denying it.
    assert!((forward - backward).abs() < 1e-12 * forward.abs().max(1.0));
}

/// The fold visits chunk 0, then 1, then 2 — not completion order.
///
/// `Vec` concatenation is used rather than a float so the assertion is about
/// the *sequence*, with no room to pass by numerical coincidence.
#[test]
fn the_partials_are_folded_in_chunk_order() {
    let plan = Plan::new(1_000, 7);
    let expected: Vec<usize> = (0..plan.chunks()).map(|k| plan.range(k).start).collect();

    for n in THREADS {
        let got = at_threads(n, || {
            det_reduce(
                plan,
                Seed([1; 32]),
                |range, _rng| vec![range.start],
                Vec::new(),
                |mut a, b| {
                    a.extend(b);
                    a
                },
            )
        });
        assert_eq!(
            got, expected,
            "chunk order was not preserved at {n} workers"
        );
    }
}

/// `map` is called once per chunk, over an exact cover, and never twice.
#[test]
fn the_cover_is_exact_and_each_chunk_is_mapped_once() {
    for &(n, grain) in &[
        (0usize, 4usize),
        (1, 4),
        (7, 4),
        (1_000, 7),
        (1_000, 1_000_000),
    ] {
        let plan = Plan::new(n, grain);
        let visited = at_threads(4, || {
            det_reduce(
                plan,
                Seed([2; 32]),
                |range, _rng| range.collect::<Vec<usize>>(),
                Vec::new(),
                |mut a, b| {
                    a.extend(b);
                    a
                },
            )
        });
        assert_eq!(
            visited,
            (0..n).collect::<Vec<usize>>(),
            "the cover of Plan::new({n}, {grain}) was not 0..{n}, exactly once, in order"
        );
    }
}

/// Defect #28's other half: the stream a chunk draws from is `Seed::split(k)`,
/// so it moves with the chunk index and not with `OMP_NUM_THREADS`.
/// `parallel_rng.hh:56-61` returns `_rngs[tnum - 1]`.
#[test]
fn each_chunk_draws_the_stream_its_index_names() {
    let seed = Seed([0xa5; 32]);
    let plan = Plan::new(4_096, 32);

    let direct: Vec<u64> = (0..plan.chunks() as u64)
        .map(|k| seed.split(k).next_u64())
        .collect();

    for n in THREADS {
        let got = at_threads(n, || {
            det_reduce(
                plan,
                seed,
                |_range, rng| vec![rng.next_u64()],
                Vec::new(),
                |mut a, b| {
                    a.extend(b);
                    a
                },
            )
        });
        assert_eq!(got, direct, "the per-chunk stream moved with the pool size");
    }
    // And the streams are not all the same stream.
    let mut uniq = direct.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(uniq.len(), direct.len());
}

// ===========================================================================
// 2. Defect #26 — the lowest failing chunk wins
// ===========================================================================

/// Several chunks fail; the error is the one from the lowest index, at every
/// thread count and on every repetition.
#[test]
fn the_lowest_failing_chunk_supplies_the_error() {
    let plan = Plan::new(1_000, 10);
    assert_eq!(plan.chunks(), 100);

    // Chunks 3, 10, 17, ... fail. A single failing chunk would not distinguish
    // "lowest" from "whichever"; this one has fourteen of them.
    let failing: Vec<usize> = (0..plan.chunks()).filter(|k| k % 7 == 3).collect();
    assert!(failing.len() > 8 && failing[0] == 3);

    let run = |plan: Plan| {
        try_det_reduce(
            plan,
            Seed([3; 32]),
            |range: std::ops::Range<usize>, _rng: &mut _| {
                let k = range.start / 10;
                if k % 7 == 3 { Err(k) } else { Ok(range.len()) }
            },
            0usize,
            |a: usize, b: usize| a + b,
        )
    };

    for n in THREADS {
        for _ in 0..4 {
            assert_eq!(
                at_threads(n, || run(plan)),
                Err(3),
                "the surviving error was not the lowest chunk's at {n} workers"
            );
        }
    }
}

/// The boundaries: only chunk 0 fails, only the last chunk fails, every chunk
/// fails. The third is the one that would expose a "first writer wins" slot.
#[test]
fn the_selection_holds_at_the_boundaries() {
    let plan = Plan::new(256, 4);
    let last = plan.chunks() - 1;

    let run = |pred: &(dyn Fn(usize) -> bool + Sync)| {
        try_det_reduce(
            plan,
            Seed([4; 32]),
            |range: std::ops::Range<usize>, _rng: &mut _| {
                let k = range.start / 4;
                if pred(k) { Err(k) } else { Ok(0usize) }
            },
            0usize,
            |a: usize, b: usize| a + b,
        )
    };

    for n in THREADS {
        assert_eq!(at_threads(n, || run(&|k| k == 0)), Err(0));
        assert_eq!(at_threads(n, || run(&|k| k == last)), Err(last));
        assert_eq!(at_threads(n, || run(&|_| true)), Err(0));
        assert_eq!(at_threads(n, || run(&|_| false)), Ok(0));
    }
}

/// With no failure, `try_det_reduce` is `det_reduce` bit for bit — the same
/// partials in the same order, not merely the same value by luck.
#[test]
fn the_fallible_fold_matches_the_infallible_one() {
    let xs = log_weights(10_000);
    let plan = Plan::new(xs.len(), 128);

    for n in THREADS {
        let ok: Result<f64, ()> = at_threads(n, || {
            try_det_reduce(
                plan,
                Seed([7; 32]),
                |range, _rng| {
                    Ok(xs[range]
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, log_sum_exp))
                },
                f64::NEG_INFINITY,
                log_sum_exp,
            )
        });
        assert_eq!(ok.unwrap().to_bits(), lse_reduce(&xs, plan).to_bits());
    }
}

/// A chunk that fails must not corrupt the prefix: the successful chunks below
/// the failure are still folded in order, which is what makes a partial result
/// usable by a caller that chooses to inspect one.
#[test]
fn the_successful_prefix_is_folded_before_the_error_is_returned() {
    let plan = Plan::new(100, 10);
    let seen = std::sync::Mutex::new(Vec::<usize>::new());

    let out: Result<usize, usize> = try_det_reduce(
        plan,
        Seed([5; 32]),
        |range: std::ops::Range<usize>, _rng: &mut _| {
            let k = range.start / 10;
            if k == 6 { Err(k) } else { Ok(k) }
        },
        0usize,
        |a: usize, b: usize| {
            seen.lock().unwrap().push(b);
            a + b
        },
    );

    assert_eq!(out, Err(6));
    // reduce ran for chunks 0..6 and stopped: 0+1+2+3+4+5.
    assert_eq!(*seen.lock().unwrap(), vec![0, 1, 2, 3, 4, 5]);
}

// ===========================================================================
// 3. chunked_sum — accuracy against Kahan
// ===========================================================================

/// Exactly-compensated reference. `c` carries the low-order bits that `s + x`
/// discarded, so the result is within one ulp of the exactly-rounded sum for
/// any input this test generates.
fn kahan(xs: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    for &x in xs {
        let y = x - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
    }
    s
}

fn summands(seed: u8, n: usize) -> Vec<f64> {
    let mut rng = Seed([seed; 32]).split(17);
    (0..n)
        .map(|i| {
            let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            // A spread of magnitudes: a sum of same-sized terms hides
            // everything interesting about association order.
            u * (1u64 << (i % 24)) as f64
        })
        .collect()
}

/// The documented bound: `(n / LANES + LANES) · ε · Σ|xᵢ|`.
fn error_bound(n: usize, lanes: usize, sum_abs: f64) -> f64 {
    (n as f64 / lanes as f64 + lanes as f64) * f64::EPSILON * sum_abs
}

/// The acceptance test: `chunked_sum::<4>` against a sequential Kahan sum.
#[test]
fn chunked_sum_4_matches_kahan_within_the_documented_bound() {
    const N: usize = 1 << 16;
    let xs = summands(0x11, N);
    let sum_abs: f64 = kahan(&xs.iter().map(|x| x.abs()).collect::<Vec<_>>());
    let reference = kahan(&xs);

    let bound = error_bound(N, 4, sum_abs);
    let got = chunked_sum::<4>(&xs);
    assert!(
        (got - reference).abs() <= bound,
        "chunked_sum::<4> was off by {:e}, documented bound {bound:e}",
        (got - reference).abs()
    );

    // Same for the other lane counts the bench sweeps.
    for (lanes, got) in [
        (1usize, chunked_sum::<1>(&xs)),
        (2, chunked_sum::<2>(&xs)),
        (8, chunked_sum::<8>(&xs)),
        (16, chunked_sum::<16>(&xs)),
    ] {
        assert!(
            (got - reference).abs() <= error_bound(N, lanes, sum_abs),
            "chunked_sum::<{lanes}> exceeded its bound"
        );
    }
}

/// The accuracy half of the ledger claim: more accumulators means shorter
/// dependency chains means less drift. Averaged over 16 independent inputs,
/// because a single draw is a random walk and could go either way.
#[test]
fn more_lanes_is_more_accurate_on_average() {
    const N: usize = 1 << 15;
    let mut e1 = 0.0f64;
    let mut e8 = 0.0f64;
    for s in 0..16u8 {
        let xs = summands(s.wrapping_add(0x40), N);
        let reference = kahan(&xs);
        e1 += (chunked_sum::<1>(&xs) - reference).abs();
        e8 += (chunked_sum::<8>(&xs) - reference).abs();
    }
    assert!(
        e8 < e1,
        "eight accumulators were not more accurate than one ({e8:e} vs {e1:e}); \
         the doc comment on chunked_sum claims they are"
    );
}

/// `chunked_sum::<1>` *is* the naive scan, bit for bit. If that ever stops
/// being true the remainder handling has drifted.
#[test]
fn one_lane_is_the_naive_scan_bit_for_bit() {
    for n in [0usize, 1, 2, 3, 5, 17, 1_000, 4_097] {
        let xs = summands(0x22, n);
        let naive = xs.iter().fold(0.0f64, |a, &x| a + x);
        assert_eq!(chunked_sum::<1>(&xs).to_bits(), naive.to_bits(), "n = {n}");
    }
}

/// A pure function of `(xs, LANES)`: no thread count, no call history.
#[test]
fn chunked_sum_is_a_pure_function() {
    let xs = summands(0x33, 5_000);
    let first = chunked_sum::<4>(&xs).to_bits();
    for n in THREADS {
        let bits = at_threads(n, || chunked_sum::<4>(&xs)).to_bits();
        assert_eq!(bits, first);
    }
}

/// Every element is counted exactly once, including a remainder shorter than
/// `LANES` and a slice shorter than `LANES` entirely.
#[test]
fn the_remainder_is_never_dropped() {
    for n in 0..40usize {
        let xs: Vec<f64> = (0..n).map(|_| 1.0).collect();
        assert_eq!(chunked_sum::<1>(&xs), n as f64, "LANES=1, n={n}");
        assert_eq!(chunked_sum::<4>(&xs), n as f64, "LANES=4, n={n}");
        assert_eq!(chunked_sum::<7>(&xs), n as f64, "LANES=7, n={n}");
        assert_eq!(chunked_sum::<64>(&xs), n as f64, "LANES=64, n={n}");
    }
}

/// `chunked_sum` composed with `det_reduce` is the shape the bench and
/// `gt-algo`'s PageRank use, and it must still be thread-count-invariant.
#[test]
fn a_chunked_sum_under_det_reduce_is_bit_identical_across_thread_counts() {
    let xs = summands(0x44, 1 << 16);
    let plan = Plan::new(xs.len(), 1_024);

    let mut seen: Option<u64> = None;
    for n in THREADS {
        let bits = at_threads(n, || {
            det_reduce(
                plan,
                Seed([9; 32]),
                |range, _rng| chunked_sum::<4>(&xs[range]),
                0.0f64,
                |a, b| a + b,
            )
        })
        .to_bits();
        match seen {
            None => seen = Some(bits),
            Some(first) => assert_eq!(bits, first, "differed at {n} workers"),
        }
    }
}

/// `ChunkSum` is the hand-rolled form of the same discipline: a partial that
/// knows which chunk it came from, so a caller assembling its own fold cannot
/// lose the order. Kept honest here so the type does not rot unused.
#[test]
fn chunk_sums_carry_the_index_that_orders_them() {
    let xs = summands(0x55, 4_096);
    let plan = Plan::new(xs.len(), 256);

    let mut parts = det_reduce(
        plan,
        Seed([0; 32]),
        |range, _rng| {
            vec![ChunkSum {
                chunk: range.start / 256,
                sum: chunked_sum::<4>(&xs[range]),
            }]
        },
        Vec::new(),
        |mut a, b| {
            a.extend(b);
            a
        },
    );
    assert_eq!(parts.len(), plan.chunks());
    let in_order: Vec<usize> = parts.iter().map(|p| p.chunk).collect();
    assert_eq!(in_order, (0..plan.chunks()).collect::<Vec<_>>());

    // Shuffled and re-sorted by `chunk`, the fold is the same number.
    let straight = parts.iter().fold(0.0f64, |a, p| a + p.sum);
    parts.reverse();
    parts.sort_by_key(|p| p.chunk);
    assert_eq!(
        parts.iter().fold(0.0f64, |a, p| a + p.sum).to_bits(),
        straight.to_bits()
    );
}

// ===========================================================================
// 4. The codegen ledger (`gt_core::design` §12) — recorded, not assumed
// ===========================================================================
//
// §12 states that a plain scan emits `addsd` and never `addpd`, "and so does a
// four-accumulator hand-unroll". The first half is a structural fact: without
// `-ffast-math` LLVM may not reassociate a serial `f64` fold, so it cannot
// vectorise it, and that is asserted. The second half is an empirical claim
// about one compiler on one target, and the SLP vectoriser is entitled to
// falsify it — four independent accumulators fed from consecutive lanes is a
// packed add with *no* reassociation. So this test prints what it found and
// asserts only what is guaranteed.

fn deps_dir() -> Option<PathBuf> {
    Some(std::env::current_exe().ok()?.parent()?.to_path_buf())
}

fn gt_core_rlib() -> Option<PathBuf> {
    let dir = deps_dir()?;
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&dir).ok()? {
        let path = entry.ok()?.path();
        let name = path.file_name()?.to_string_lossy().into_owned();
        if name.starts_with("libgt_core-") && name.ends_with(".rlib") {
            let when = path.metadata().ok()?.modified().ok()?;
            if best.as_ref().is_none_or(|(b, _)| when > *b) {
                best = Some((when, path));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// Compile `body` against gt-core with `-O` and return the assembly, or `None`
/// if the toolchain or the rlib is not where this test can see it.
fn probe_asm(name: &str, body: &str) -> Option<String> {
    let rlib = gt_core_rlib()?;
    let deps = deps_dir()?;
    let dir = std::env::temp_dir().join(format!("gt_u11_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("probe.rs");
    let asm = dir.join("probe.s");
    std::fs::write(
        &src,
        format!(
            "extern crate gt_core;\n\
             #[allow(unused_imports)]\n\
             use gt_core::par::reduce::chunked_sum;\n\
             #[no_mangle]\n\
             {body}\n"
        ),
    )
    .ok()?;

    let out = Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
        .args([
            "-O",
            "--edition",
            "2021",
            "--crate-type",
            "lib",
            "--emit",
            "asm",
        ])
        .arg("--extern")
        .arg(format!("gt_core={}", rlib.display()))
        .arg("-L")
        .arg(format!("dependency={}", deps.display()))
        .arg("-o")
        .arg(&asm)
        .arg(&src)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "SKIPPED: could not compile the codegen probe:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    let text = std::fs::read_to_string(&asm).ok();
    let _ = std::fs::remove_dir_all(&dir);
    text
}

/// Does `asm` contain a packed double-precision add?
fn has_packed_add(asm: &str) -> bool {
    asm.lines()
        .map(str::trim)
        .any(|l| l.starts_with("addpd") || l.starts_with("vaddpd"))
}

fn has_scalar_add(asm: &str) -> bool {
    asm.lines()
        .map(str::trim)
        .any(|l| l.starts_with("addsd") || l.starts_with("vaddsd"))
}

#[test]
fn the_ledger_records_whether_chunked_sum_vectorises() {
    if !cfg!(target_arch = "x86_64") {
        eprintln!("SKIPPED: the §12 claim is about x86-64 mnemonics");
        return;
    }
    let Some(scan) = probe_asm(
        "scan",
        "pub fn s(xs: &[f64]) -> f64 { xs.iter().fold(0.0f64, |a, &x| a + x) }",
    ) else {
        return;
    };
    assert!(scan.contains("s:"), "the probe emitted no symbol");

    // Guaranteed, not measured: a serial `f64` fold is not reassociable, so
    // there is no legal packed form of it.
    assert!(
        !has_packed_add(&scan),
        "a serial f64 fold vectorised; that would mean fast-math semantics \
         crept in, and every bit-identity test above is then meaningless"
    );
    assert!(has_scalar_add(&scan), "the scan emitted no f64 add at all");

    let Some(unrolled) = probe_asm(
        "unrolled",
        "pub fn s(xs: &[f64]) -> f64 { chunked_sum::<4>(xs) }",
    ) else {
        return;
    };
    assert!(unrolled.contains("s:"), "the probe emitted no symbol");

    let packed = has_packed_add(&unrolled);
    let record = format!(
        "LEDGER (gt_core::design §12, U11): {} -O, arch = {}\n  \
         scan            : addpd = false (guaranteed), addsd = true\n  \
         chunked_sum::<4>: addpd = {packed}\n  \
         §12 records addpd = true for the hand-unroll. \
         Measured here: addpd = {packed}. {}\n",
        std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()),
        std::env::consts::ARCH,
        if packed {
            "Agrees with §12 as corrected: LANES independent accumulators \
             are a packed add with no reassociation, and the SLP vectoriser \
             takes it. The determinism claim is unaffected -- the association \
             order is still lane-wise and fixed."
        } else {
            "DISAGREES with §12 as corrected, which records addpd = true \
             on rustc 1.98.1 / x86-64. Not a defect -- the SLP vectoriser is \
             never obliged to pack -- but §12 should then be re-measured."
        }
    );
    println!("{record}");
    // Durable, because a number printed into a captured stdout is not a record.
    if let Some(dir) = deps_dir() {
        let _ = std::fs::write(dir.join("u11_codegen_ledger.txt"), &record);
    }

    // The only assertion: whichever form it took, it is still doing f64 adds
    // and it did not grow a bounds check or a call into the allocator.
    assert!(
        packed || has_scalar_add(&unrolled),
        "chunked_sum::<4> emitted no f64 add at all"
    );
    assert!(
        !unrolled.contains("panic_bounds_check"),
        "chunked_sum left a bounds check in the accumulation loop"
    );
    assert!(
        !unrolled.contains("__rust_alloc"),
        "chunked_sum allocated; it is a steady-state inner loop"
    );
}
