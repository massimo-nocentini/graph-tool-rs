//! U23 -- the entropy functional.
//!
//! The acceptance criterion is that each kernel matches the corresponding C++
//! expression on 10^4 random inputs to 1e-12:
//!
//! | kernel | C++ |
//! |---|---|
//! | [`eterm`] | `inference/blockmodel/entropy.hh:38` (`eterm_d`) |
//! | [`vterm`] | `entropy.hh:73` (`vterm_d`) |
//! | [`eterm_dense`] | `entropy.hh:235` (`eterm_dense_d`) |
//! | [`edges_dl`] | `entropy.hh:293` (`get_edges_dl`) |
//! | [`partition_dl`] | `inference/blockmodel/partition.hh:106` (`get_dl`) |
//!
//! A transcription test alone is circular in one specific way: it routes both
//! sides through the same [`Cache`], so a wrong `lgamma` reproduces itself.
//! Every kernel below therefore *also* gets a check against exact
//! `ln(n!)` arithmetic over the small-integer range, which is the arithmetic
//! the terms actually are, and a structural check that the `D` instantiation
//! is load-bearing -- an `eterm` that ignored `D` would pass the transcription
//! test against an `eterm_ref` that ignored it too.
//!
//! `sparse_ds` and `dense_ds` get their numeric coverage in
//! `blockmodel::entropy`'s own `#[cfg(test)]` module, where the private
//! `sparse_terms`/`dense_terms` kernels can be driven from hand-built entries.
//! `blockmodel/mod.rs` now re-exports both (it did not when this file was
//! first written), so the last section here re-checks them *through the
//! public path* -- the re-export is the whole subject of that test, and a
//! re-export nothing names is a line nobody would notice losing.

use gt_core::dir::{Dir, Directed, Undirected};
use gt_inference::blockmodel::{
    BlockView, Cache, EntropyParams, dense_ds, edges_dl, eterm, eterm_dense, partition_dl,
    sparse_ds, vterm,
};
use gt_inference::delta::{
    EndImage, MoveHeader, MoveKey, Recording, Transition, Workspace,
};
use gt_inference::ids::{BEdge, Epoch, Group, Stamp, StateId};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Enough to cover every random input below without touching the fallback
/// path, so a disagreement is the kernel's and not the table's.
const TABLE: usize = 1 << 14;

/// The acceptance count.
const N_RANDOM: usize = 10_000;

fn cache() -> Cache {
    Cache::build(TABLE)
}

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Relative-or-absolute agreement at the stated tolerance.
#[track_caller]
fn close(a: f64, b: f64, tol: f64, what: &str) {
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!(
        (a - b).abs() <= tol * scale,
        "{what}: {a} vs {b} (|d| = {})",
        (a - b).abs()
    );
}

// ---------------------------------------------------------------------------
// Exact small-integer arithmetic, independent of `Cache`.
// ---------------------------------------------------------------------------

/// `ln(n!)`, summed term by term. Exact to a few ulps for the `n <= 60` used
/// below, and it shares no code with the tabulated `lgamma`.
fn ln_fact(n: u64) -> f64 {
    (2..=n).map(|i| (i as f64).ln()).sum()
}

/// `lbinom_fast` (`inference/support/util.hh:42-47`) with exact factorials,
/// **including its three guards** -- those are semantics, not an
/// implementation detail: `lbinom(n, 0)`, `lbinom(0, k)` and `lbinom(n, k)`
/// for `k >= n` are all zero, which is what makes `eterm_dense` vanish on a
/// block pair with no edges.
fn ln_binom_exact(n: u64, k: u64) -> f64 {
    if n == 0 || k == 0 || k >= n {
        return 0.0;
    }
    ln_fact(n) - ln_fact(k) - ln_fact(n - k)
}

// ---------------------------------------------------------------------------
// Reference transcriptions of the C++ expressions.
// ---------------------------------------------------------------------------

/// `eterm_d` (`entropy.hh:38-60`).
///
/// The first two arms are deliberately not merged: the C++ has three, and a
/// transcription that folds `if constexpr (directed)` into `r != s` is no
/// longer independent of the thing it is checking.
#[allow(clippy::if_same_then_else)]
fn eterm_ref<D: Dir>(r: usize, s: usize, mrs: f64, c: &Cache) -> f64 {
    let val = c.lgamma1p(mrs);
    if D::DIRECTED {
        -val
    } else if r != s {
        -val
    } else {
        -val - mrs * std::f64::consts::LN_2
    }
}

/// `vterm_d` (`entropy.hh:73-89`).
fn vterm_ref<D: Dir>(mrp: f64, mrm: f64, wr: f64, deg_corr: bool, c: &Cache) -> f64 {
    if deg_corr {
        if D::DIRECTED {
            c.lgamma1p(mrp) + c.lgamma1p(mrm)
        } else {
            c.lgamma1p(mrp)
        }
    } else if D::DIRECTED {
        (mrp + mrm) * c.safelog(wr)
    } else {
        mrp * c.safelog(wr)
    }
}

/// `eterm_dense_d` (`entropy.hh:235-257`).
fn eterm_dense_ref<D: Dir>(
    r: usize,
    s: usize,
    ers: f64,
    wr_r: f64,
    wr_s: f64,
    multigraph: bool,
    c: &Cache,
) -> f64 {
    let nrns = if D::DIRECTED || r != s {
        wr_r * wr_s
    } else if multigraph {
        (wr_r * (wr_r + 1.0)) / 2.0
    } else {
        (wr_r * (wr_r - 1.0)) / 2.0
    };
    if multigraph {
        c.lbinom(nrns + ers - 1.0, ers)
    } else {
        c.lbinom(nrns, ers)
    }
}

/// `get_edges_dl` (`entropy.hh:293-297`).
fn edges_dl_ref<D: Dir>(b: i64, e: i64, c: &Cache) -> f64 {
    let bb = if D::DIRECTED {
        b * b
    } else {
        (b * (b + 1)) / 2
    };
    c.lbinom((bb + e - 1) as f64, e as f64)
}

/// `partition_stats_t::get_dl` (`partition.hh:106-117`).
fn partition_dl_ref(sizes: &[usize], n: usize, c: &Cache) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let actual_b = sizes.iter().filter(|&&x| x != 0).count();
    let mut s = c.lbinom((n - 1) as f64, (actual_b - 1) as f64);
    s += c.lgamma1p(n as f64);
    for &nr in sizes {
        s -= c.lgamma1p(nr as f64);
    }
    s += c.safelog(n as f64);
    s
}

// ---------------------------------------------------------------------------
// eterm
// ---------------------------------------------------------------------------

#[test]
fn eterm_matches_the_cpp_expression_on_random_inputs() {
    let c = cache();
    let mut rg = rng(0xE7E4);
    for i in 0..N_RANDOM {
        // Force a self-pair on a third of the draws: `r == s` is the only
        // input that separates the two undirected arms.
        let r: usize = rg.random_range(0..8);
        let s = if i % 3 == 0 { r } else { rg.random_range(0..8) };
        let mrs = rg.random_range(0..4096u32) as f64;

        close(
            eterm::<Directed>(r, s, mrs, &c),
            eterm_ref::<Directed>(r, s, mrs, &c),
            1e-12,
            "eterm directed",
        );
        close(
            eterm::<Undirected>(r, s, mrs, &c),
            eterm_ref::<Undirected>(r, s, mrs, &c),
            1e-12,
            "eterm undirected",
        );
    }
}

#[test]
fn eterm_is_minus_log_factorial() {
    let c = cache();
    for m in 0..=60u64 {
        // Off-diagonal: both instantiations are `-ln(mrs!)`.
        for (r, s) in [(0usize, 1usize), (3, 7)] {
            close(
                eterm::<Directed>(r, s, m as f64, &c),
                -ln_fact(m),
                1e-12,
                "eterm off-diagonal",
            );
            close(
                eterm::<Undirected>(r, s, m as f64, &c),
                -ln_fact(m),
                1e-12,
                "eterm off-diagonal undirected",
            );
        }
        // The undirected self-pair carries the extra `-mrs * log 2`
        // (`entropy.hh:58`).
        close(
            eterm::<Undirected>(2, 2, m as f64, &c),
            -ln_fact(m) - (m as f64) * std::f64::consts::LN_2,
            1e-12,
            "eterm undirected self-pair",
        );
    }
}

/// The directedness parameter has to reach the answer. A kernel that dropped
/// `D` would make these two equal on the diagonal.
#[test]
fn eterm_directedness_is_load_bearing() {
    let c = cache();
    for m in [1.0f64, 2.0, 17.0, 1000.0] {
        let d = eterm::<Directed>(4, 4, m, &c);
        let u = eterm::<Undirected>(4, 4, m, &c);
        // Not `assert_eq!`: the undirected arm is `-val - mrs * LN_2`, so
        // recovering the correction by subtraction reassociates it and can
        // land one ulp away.
        close(d - u, m * std::f64::consts::LN_2, 1e-12, "self-pair correction");
        // Off the diagonal the two agree exactly.
        assert_eq!(eterm::<Directed>(4, 5, m, &c), eterm::<Undirected>(4, 5, m, &c));
    }
    // `mrs == 0` is the one self-pair where the arms coincide.
    assert_eq!(
        eterm::<Directed>(4, 4, 0.0, &c),
        eterm::<Undirected>(4, 4, 0.0, &c)
    );
}

// ---------------------------------------------------------------------------
// vterm
// ---------------------------------------------------------------------------

#[test]
fn vterm_matches_the_cpp_expression_on_random_inputs() {
    let c = cache();
    let mut rg = rng(0x17E4);
    for i in 0..N_RANDOM {
        let mrp = rg.random_range(0..4096u32) as f64;
        let mrm = rg.random_range(0..4096u32) as f64;
        // `wr == 0` is the `safelog` guard (`cache.hh:99-104`); draw it often.
        let wr = if i % 5 == 0 {
            0.0
        } else {
            rg.random_range(1..1024u32) as f64
        };
        let dc = i % 2 == 0;

        close(
            vterm::<Directed>(mrp, mrm, wr, dc, &c),
            vterm_ref::<Directed>(mrp, mrm, wr, dc, &c),
            1e-12,
            "vterm directed",
        );
        close(
            vterm::<Undirected>(mrp, mrm, wr, dc, &c),
            vterm_ref::<Undirected>(mrp, mrm, wr, dc, &c),
            1e-12,
            "vterm undirected",
        );
    }
}

#[test]
fn vterm_is_log_factorial_when_degree_corrected() {
    let c = cache();
    for p in 0..=40u64 {
        for m in [0u64, 1, 7, 40] {
            close(
                vterm::<Directed>(p as f64, m as f64, 99.0, true, &c),
                ln_fact(p) + ln_fact(m),
                1e-12,
                "vterm directed deg_corr",
            );
            close(
                vterm::<Undirected>(p as f64, m as f64, 99.0, true, &c),
                ln_fact(p),
                1e-12,
                "vterm undirected deg_corr ignores mrm",
            );
        }
    }
}

#[test]
fn vterm_is_degree_times_log_size_when_not_degree_corrected() {
    let c = cache();
    for p in [0.0f64, 1.0, 13.0, 4096.0] {
        for m in [0.0f64, 3.0, 99.0] {
            for w in [1.0f64, 2.0, 512.0] {
                close(
                    vterm::<Directed>(p, m, w, false, &c),
                    (p + m) * w.ln(),
                    1e-12,
                    "vterm directed plain",
                );
                close(
                    vterm::<Undirected>(p, m, w, false, &c),
                    p * w.ln(),
                    1e-12,
                    "vterm undirected plain",
                );
            }
            // `safelog(0) == 0` (`cache.hh:102`), not `-inf`: an empty group
            // must contribute nothing rather than poison the sum.
            assert_eq!(vterm::<Directed>(p, m, 0.0, false, &c), 0.0);
            assert_eq!(vterm::<Undirected>(p, m, 0.0, false, &c), 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// eterm_dense
// ---------------------------------------------------------------------------

#[test]
fn eterm_dense_matches_the_cpp_expression_on_random_inputs() {
    let c = cache();
    let mut rg = rng(0xDE45);
    for i in 0..N_RANDOM {
        let r: usize = rg.random_range(0..8);
        let s = if i % 3 == 0 { r } else { rg.random_range(0..8) };
        let wr_r = rg.random_range(0..64u32) as f64;
        let wr_s = rg.random_range(0..64u32) as f64;
        let ers = rg.random_range(0..256u32) as f64;
        let mg = i % 2 == 0;

        close(
            eterm_dense::<Directed>(r, s, ers, wr_r, wr_s, mg, &c),
            eterm_dense_ref::<Directed>(r, s, ers, wr_r, wr_s, mg, &c),
            1e-12,
            "eterm_dense directed",
        );
        close(
            eterm_dense::<Undirected>(r, s, ers, wr_r, wr_s, mg, &c),
            eterm_dense_ref::<Undirected>(r, s, ers, wr_r, wr_s, mg, &c),
            1e-12,
            "eterm_dense undirected",
        );
    }
}

#[test]
fn eterm_dense_is_an_exact_log_binomial() {
    let c = cache();
    for wr_r in 1..=6u64 {
        for wr_s in 1..=6u64 {
            for ers in 0..=8u64 {
                // Off-diagonal / directed: `nrns = wr_r * wr_s`.
                let n = wr_r * wr_s;
                close(
                    eterm_dense::<Directed>(0, 1, ers as f64, wr_r as f64, wr_s as f64, false, &c),
                    ln_binom_exact(n, ers),
                    1e-12,
                    "eterm_dense simple graph",
                );
                close(
                    eterm_dense::<Directed>(0, 1, ers as f64, wr_r as f64, wr_s as f64, true, &c),
                    ln_binom_exact(n + ers - 1, ers),
                    1e-12,
                    "eterm_dense multigraph",
                );
            }
        }
    }
}

/// The undirected self-pair counts *unordered* vertex pairs: `n(n-1)/2`
/// without self-loops and `n(n+1)/2` with them (`entropy.hh:247-250`). The
/// directed arm uses `n^2`, so an instantiation that lost `D` or lost `r == s`
/// would be off by roughly a factor of two in the binomial's first argument.
#[test]
fn eterm_dense_self_pair_uses_the_triangular_count() {
    let c = cache();
    for n in 1..=8u64 {
        for ers in 0..=6u64 {
            close(
                eterm_dense::<Undirected>(3, 3, ers as f64, n as f64, n as f64, false, &c),
                ln_binom_exact(n * (n - 1) / 2, ers),
                1e-12,
                "undirected self-pair, simple",
            );
            close(
                eterm_dense::<Undirected>(3, 3, ers as f64, n as f64, n as f64, true, &c),
                ln_binom_exact(n * (n + 1) / 2 + ers - 1, ers),
                1e-12,
                "undirected self-pair, multigraph",
            );
            // Directed keeps `n^2` even on the diagonal.
            close(
                eterm_dense::<Directed>(3, 3, ers as f64, n as f64, n as f64, false, &c),
                ln_binom_exact(n * n, ers),
                1e-12,
                "directed self-pair",
            );
        }
    }
}

/// A block pair with no edges contributes exactly nothing, whatever the group
/// sizes. This is what lets a sweep over the block *graph* and a sweep over
/// every block *pair* agree, and `dense_ds` relies on it.
#[test]
fn eterm_dense_vanishes_without_edges() {
    let c = cache();
    for wr_r in 0..=9u32 {
        for wr_s in 0..=9u32 {
            for mg in [false, true] {
                assert_eq!(
                    eterm_dense::<Directed>(1, 2, 0.0, wr_r as f64, wr_s as f64, mg, &c),
                    0.0
                );
                assert_eq!(
                    eterm_dense::<Undirected>(1, 1, 0.0, wr_r as f64, wr_r as f64, mg, &c),
                    0.0
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// edges_dl
// ---------------------------------------------------------------------------

#[test]
fn edges_dl_matches_the_cpp_expression_on_random_inputs() {
    let c = cache();
    let mut rg = rng(0xED91);
    for _ in 0..N_RANDOM {
        let b: i64 = rg.random_range(1..90);
        let e: i64 = rg.random_range(0..4000);
        close(
            edges_dl::<Directed>(b as usize, e as f64, &c),
            edges_dl_ref::<Directed>(b, e, &c),
            1e-12,
            "edges_dl directed",
        );
        close(
            edges_dl::<Undirected>(b as usize, e as f64, &c),
            edges_dl_ref::<Undirected>(b, e, &c),
            1e-12,
            "edges_dl undirected",
        );
    }
}

#[test]
fn edges_dl_is_an_exact_log_binomial() {
    let c = cache();
    for b in 1..=7u64 {
        for e in 0..=8u64 {
            close(
                edges_dl::<Directed>(b as usize, e as f64, &c),
                ln_binom_exact(b * b + e - 1, e),
                1e-12,
                "edges_dl directed",
            );
            close(
                edges_dl::<Undirected>(b as usize, e as f64, &c),
                ln_binom_exact(b * (b + 1) / 2 + e - 1, e),
                1e-12,
                "edges_dl undirected",
            );
        }
    }
}

/// `B * B` against `B * (B + 1) / 2` (`entropy.hh:295`): the directed model
/// has strictly more block pairs to spend description length on for `B > 1`.
#[test]
fn edges_dl_directedness_is_load_bearing() {
    let c = cache();
    for b in 2..40usize {
        let d = edges_dl::<Directed>(b, 500.0, &c);
        let u = edges_dl::<Undirected>(b, 500.0, &c);
        assert!(d > u, "B={b}: directed {d} should exceed undirected {u}");
    }
    // `E == 0` hits `lbinom`'s `k == 0` guard in both.
    assert_eq!(edges_dl::<Directed>(9, 0.0, &c), 0.0);
    assert_eq!(edges_dl::<Undirected>(9, 0.0, &c), 0.0);
}

// ---------------------------------------------------------------------------
// partition_dl
// ---------------------------------------------------------------------------

#[test]
fn partition_dl_matches_the_cpp_expression_on_random_inputs() {
    let c = cache();
    let mut rg = rng(0x9A27);
    for i in 0..N_RANDOM {
        let slots = rg.random_range(1..32usize);
        // Empty slots on purpose: `_count` is indexed by group *id*, so an
        // unoccupied group is a zero, and `_actual_B` counts only the rest.
        let sizes: Vec<usize> = (0..slots)
            .map(|_| {
                if rg.random_range(0..4u32) == 0 {
                    0
                } else {
                    rg.random_range(1..64usize)
                }
            })
            .collect();
        let n: usize = sizes.iter().sum();
        if n == 0 {
            assert_eq!(partition_dl(&sizes, n, &c), 0.0, "draw {i}");
            continue;
        }
        close(
            partition_dl(&sizes, n, &c),
            partition_dl_ref(&sizes, n, &c),
            1e-12,
            "partition_dl",
        );
    }
}

#[test]
fn partition_dl_is_exact_for_small_partitions() {
    let c = cache();
    let cases: [&[usize]; 6] = [
        &[1],
        &[3, 2, 1],
        &[4, 4],
        &[5, 0, 3, 0, 1],
        &[1, 1, 1, 1, 1, 1],
        &[9, 0, 0],
    ];
    for sizes in cases {
        let n: usize = sizes.iter().sum();
        let b = sizes.iter().filter(|&&x| x != 0).count();
        let want = ln_binom_exact((n - 1) as u64, (b - 1) as u64) + ln_fact(n as u64)
            - sizes.iter().map(|&x| ln_fact(x as u64)).sum::<f64>()
            + (n as f64).ln();
        close(partition_dl(sizes, n, &c), want, 1e-12, "partition_dl exact");
    }
}

/// `get_dl`'s `_N == 0` early return (`partition.hh:109-110`).
#[test]
fn partition_dl_of_an_empty_partition_is_zero() {
    let c = cache();
    assert_eq!(partition_dl(&[], 0, &c), 0.0);
    assert_eq!(partition_dl(&[0, 0, 0], 0, &c), 0.0);
}

/// Empty group slots are invisible: `lgamma(0 + 1) == 0` drops them from the
/// product, and `_actual_B` never counted them. Padding `_count` with zeros
/// must not move the answer.
#[test]
fn partition_dl_ignores_empty_slots() {
    let c = cache();
    let base = [4usize, 3, 2];
    let n: usize = base.iter().sum();
    let want = partition_dl(&base, n, &c);
    for pad in 1..16usize {
        let mut padded = base.to_vec();
        padded.extend(std::iter::repeat_n(0usize, pad));
        assert_eq!(partition_dl(&padded, n, &c), want, "padded with {pad} zeros");
    }
}

/// `ln(N!) - sum ln(n_r!)` is the log of the multinomial coefficient: the
/// number of labelled partitions with that size sequence. It is *largest* for
/// the most even split, so an even partition costs more description length
/// than a skewed one at the same `N` and `B`, and the difference is exactly
/// the log ratio of the two multinomials.
///
/// The sign of the `-sum lgamma` term (`partition.hh:114-115`) is what makes
/// that so; flipping it would reverse both assertions below.
#[test]
fn partition_dl_costs_the_log_multinomial() {
    let c = cache();
    let even = [25usize, 25, 25, 25];
    let skewed = [97usize, 1, 1, 1];
    let n = 100usize;

    let d_even = partition_dl(&even, n, &c);
    let d_skewed = partition_dl(&skewed, n, &c);
    assert!(
        d_skewed < d_even,
        "skewed {d_skewed} should cost less than even {d_even}"
    );

    // Both share `lbinom(N-1, B-1)`, `lgamma(N+1)` and `log N`, so the whole
    // difference is the `-sum lgamma` term.
    let want: f64 = skewed.iter().map(|&x| ln_fact(x as u64)).sum::<f64>()
        - even.iter().map(|&x| ln_fact(x as u64)).sum::<f64>();
    close(d_skewed - d_even, -want, 1e-12, "log multinomial ratio");
}

// ---------------------------------------------------------------------------
// The two pricing entry points, through the public path
// ---------------------------------------------------------------------------
//
// U23 reported both of these as unreachable from outside `gt-inference`:
// `blockmodel::entropy` is a private module, and `blockmodel/mod.rs` (owned by
// U22) re-exported `edges_dl`, `eterm`, `eterm_dense`, `partition_dl` and
// `vterm` but neither `sparse_ds` nor `dense_ds`. The re-export has since been
// widened. What follows is the test that makes the widening load-bearing: it
// mints a real `Delta` through the recording lifecycle and prices it, using
// only names a downstream crate can write.
//
// The numbers are re-derived here rather than borrowed, so this is a second
// independent check of the kernels and not merely a linkage test.

/// The block-pair matrix, three slots, directed.
const DS_B: usize = 3;

fn ds_group(i: usize) -> Group {
    Group::new(i as u32).expect("not the null group")
}

/// Out- and in-weights of every slot, read off the block-pair matrix.
fn ds_degrees<D: Dir>(mrs: &[i64], slots: usize) -> (Vec<i64>, Vec<i64>) {
    let mut mrp = vec![0i64; slots];
    let mut mrm = vec![0i64; slots];
    for r in 0..slots {
        for t in 0..slots {
            let w = mrs[r * slots + t];
            if w == 0 {
                continue;
            }
            mrp[r] += w;
            if D::DIRECTED {
                mrm[t] += w;
            } else {
                // Undirected: the matrix is stored upper-triangular and a
                // slot's degree counts a self-pair twice (`entropy.hh:80`).
                mrp[t] += w;
            }
        }
    }
    (mrp, mrm)
}

/// The sparse entropy of a whole block-pair matrix, from scratch.
fn ds_sparse_absolute<D: Dir>(mrs: &[i64], wr: &[i64], dc: bool, c: &Cache) -> f64 {
    let (mrp, mrm) = ds_degrees::<D>(mrs, DS_B);
    let mut s = 0.0;
    for r in 0..DS_B {
        for t in 0..DS_B {
            if !D::DIRECTED && t < r {
                continue;
            }
            s += eterm::<D>(r, t, mrs[r * DS_B + t] as f64, c);
        }
    }
    for r in 0..DS_B {
        s += vterm::<D>(mrp[r] as f64, mrm[r] as f64, wr[r] as f64, dc, c);
    }
    s
}

/// The dense entropy of a whole block-pair matrix, from scratch.
fn ds_dense_absolute<D: Dir>(mrs: &[i64], wr: &[i64], multigraph: bool, c: &Cache) -> f64 {
    let mut s = 0.0;
    for r in 0..DS_B {
        for t in 0..DS_B {
            if !D::DIRECTED && t < r {
                continue;
            }
            // `(r, s, mrs, wr_r, wr_s, ..)` -- the three weights are all
            // `f64`, so transposing them compiles and only the numbers say so.
            s += eterm_dense::<D>(
                r,
                t,
                mrs[r * DS_B + t] as f64,
                wr[r] as f64,
                wr[t] as f64,
                multigraph,
                c,
            );
        }
    }
    s
}

/// A minimal `BlockView` over a dense block-pair matrix.
///
/// `dense_ds` reads only `wr` and the pair weights (`state.hh:1143-1171`), so
/// the rest of the trait is present to satisfy the bound and nothing more.
struct MatrixView {
    mrs: Vec<i64>,
    wr: Vec<i64>,
    stamp: Stamp,
}

impl BlockView for MatrixView {
    type D = Directed;
    type W = i64;

    fn stamp(&self) -> Stamp {
        self.stamp
    }

    fn group_of(&self, _v: gt_core::ids::VertexId) -> Option<Group> {
        unreachable!("dense_ds never asks a vertex for its group")
    }

    fn n_groups(&self) -> usize {
        self.wr.iter().filter(|&&w| w != 0).count()
    }

    fn find_me(&self, r: Group, s: Group) -> Option<BEdge> {
        let i = r.index() * DS_B + s.index();
        (self.mrs[i] != 0).then_some(BEdge(i as u32))
    }

    fn mrs(&self, e: BEdge) -> i64 {
        self.mrs[e.0 as usize]
    }

    fn mrp(&self, r: Group) -> i64 {
        (0..DS_B).map(|t| self.mrs[r.index() * DS_B + t]).sum()
    }

    fn mrm(&self, r: Group) -> i64 {
        (0..DS_B).map(|t| self.mrs[t * DS_B + r.index()]).sum()
    }

    fn wr(&self, r: Group) -> i64 {
        self.wr[r.index()]
    }

    fn move_prob(&self, _t: &Transition<'_, Directed, i64>, _v: gt_core::ids::VertexId, _c: f64) -> f64 {
        unreachable!("dense_ds never prices a proposal")
    }
}

/// One vertex of weight 1 carrying one out-edge and one in-edge moves from
/// group 0 to group 1. Returns the before-image, the touches, and the header.
#[allow(clippy::type_complexity)]
fn ds_move() -> (Vec<i64>, Vec<i64>, Vec<(Group, Group, i64)>, MoveHeader<i64>) {
    let before: Vec<i64> = vec![
        4, 2, 1, //
        3, 5, 2, //
        1, 2, 6,
    ];
    let wr = vec![5i64, 4, 3];
    let (r, nr) = (ds_group(0), ds_group(1));
    let touches = vec![
        (r, ds_group(2), -1),
        (nr, ds_group(2), 1),
        (ds_group(1), r, -1),
        (ds_group(1), nr, 1),
    ];
    let (mrp, mrm) = ds_degrees::<Directed>(&before, DS_B);
    let hdr = MoveHeader {
        r: Some(r),
        nr: Some(nr),
        r_img: EndImage {
            mrp: mrp[0],
            mrm: mrm[0],
            wr: wr[0],
        },
        nr_img: EndImage {
            mrp: mrp[1],
            mrm: mrm[1],
            wr: wr[1],
        },
        dkin: 1,
        dkout: 1,
        dr: 1,
        dnr: 1,
    };
    (before, wr, touches, hdr)
}

/// The after-image implied by the touches and the header.
fn ds_after(before: &[i64], wr: &[i64], touches: &[(Group, Group, i64)]) -> (Vec<i64>, Vec<i64>) {
    let mut mrs = before.to_vec();
    for &(r, s, d) in touches {
        mrs[r.index() * DS_B + s.index()] += d;
    }
    let mut w = wr.to_vec();
    w[0] -= 1;
    w[1] += 1;
    (mrs, w)
}

/// `sparse_ds` is callable from outside the crate, and prices the transition.
///
/// This is U23's first blocked item discharged. Every name in the body --
/// `sparse_ds`, `EntropyParams`, `Workspace`, `Recording`, `MoveKey`,
/// `MoveHeader`, `EndImage` -- resolves through `gt_inference`'s public API.
#[test]
fn sparse_ds_is_reachable_and_prices_a_recorded_delta() {
    let c = cache();
    let (before, wr, touches, hdr) = ds_move();
    let (after, wr_after) = ds_after(&before, &wr, &touches);

    for &deg_corr in &[false, true] {
        let mut ws: Workspace<Directed, i64> = Workspace::default();
        let mut rec = Recording::new(&mut ws.stack, hdr, ds_stamp());
        {
            let buf = rec.level_mut(0);
            buf.begin(
                MoveKey {
                    from: hdr.r,
                    to: hdr.nr,
                },
                DS_B,
            );
            let mut resolve = |r: Group, s: Group| {
                let i = r.index() * DS_B + s.index();
                let w = before[i];
                ((w != 0).then_some(BEdge(i as u32)), w)
            };
            for &(r, s, d) in &touches {
                buf.touch_dyn(r, s, d, &mut resolve).expect("in-plane");
            }
        }
        let sealed = rec.seal();

        let priced = sparse_ds(
            sealed.level(0),
            EntropyParams {
                deg_corr,
                ..EntropyParams::default()
            },
            &c,
        );
        let expected = ds_sparse_absolute::<Directed>(&after, &wr_after, deg_corr, &c)
            - ds_sparse_absolute::<Directed>(&before, &wr, deg_corr, &c);

        assert!(
            (priced - expected).abs() < 1e-9,
            "deg_corr={deg_corr}: sparse_ds gave {priced}, from scratch {expected}"
        );
    }
}

/// `dense_ds` is callable from outside the crate, and prices the transition.
///
/// The second half of U23's blocked item. `dense_ds` additionally needs a
/// `BlockView`, so this also pins that the trait is public enough to be
/// implemented downstream.
#[test]
fn dense_ds_is_reachable_and_prices_a_recorded_delta() {
    let c = cache();
    let (before, wr, touches, hdr) = ds_move();
    let (after, wr_after) = ds_after(&before, &wr, &touches);

    for &multigraph in &[false, true] {
        let st = MatrixView {
            mrs: before.clone(),
            wr: wr.clone(),
            stamp: ds_stamp(),
        };

        let mut ws: Workspace<Directed, i64> = Workspace::default();
        let mut rec = Recording::new(&mut ws.stack, hdr, st.stamp());
        {
            let buf = rec.level_mut(0);
            buf.begin(
                MoveKey {
                    from: hdr.r,
                    to: hdr.nr,
                },
                DS_B,
            );
            let mut resolve = |r: Group, s: Group| st.resolve(r, s);
            for &(r, s, d) in &touches {
                buf.touch_dyn(r, s, d, &mut resolve).expect("in-plane");
            }
        }
        let sealed = rec.seal();

        let priced = dense_ds(
            &st,
            sealed.level(0),
            DS_B,
            EntropyParams {
                deg_corr: false,
                multigraph,
                ..EntropyParams::default()
            },
            &c,
        );
        let expected = ds_dense_absolute::<Directed>(&after, &wr_after, multigraph, &c)
            - ds_dense_absolute::<Directed>(&before, &wr, multigraph, &c);

        assert!(
            (priced - expected).abs() < 1e-9,
            "multigraph={multigraph}: dense_ds gave {priced}, from scratch {expected}"
        );
    }
}

fn ds_stamp() -> Stamp {
    Stamp {
        state: StateId::fresh(),
        epoch: Epoch::default(),
    }
}
