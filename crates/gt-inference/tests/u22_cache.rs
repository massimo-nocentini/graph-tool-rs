//! U22 acceptance: the special-function cache.
//!
//! The design phase's single largest evidence failure was a crate whose
//! `ln_gamma` was `x.ln()`. It compiled, it passed its own tests, and the
//! pricing path it fed was wrong on 194 of 269 moves. So the bar here is not
//! "it compiles": every number below is checked against something computed a
//! different way --
//!
//! * `lgamma1p` against a 60-digit `mpmath` reference (`REF`), and
//!   independently against `ln(n!)` accumulated by compensated summation;
//! * `lbinom` against exact `u128` binomial coefficients;
//! * `log_q` against an exact `u128` dynamic program for the restricted
//!   partition counts `q(n, k)` (`int_part.cc:61-72`).
//!
//! `Cache` has no interior mutability and no global backing it, which is the
//! whole point of defect #34's replacement; `cache_is_sync_and_threads_agree`
//! is what pins that.

use std::thread;

use gt_inference::blockmodel::Cache;

// ---------------------------------------------------------------------------
// The high-precision reference.
// ---------------------------------------------------------------------------

/// `(x, ln Gamma(x + 1))` with the second element the correctly rounded `f64`
/// of a 60-decimal-digit `mpmath.loggamma` evaluation.
///
/// Spread over `[0, 10^6]`, and deliberately including:
///
/// * both zeros of `lgamma1p`, at `x = 0` and `x = 1`, where any method with
///   merely good absolute error has unbounded relative error;
/// * their immediate neighbourhoods (`1e-12`, `0.9999999`, `1.0000001`);
/// * integers on both sides of a plausible table edge (63/64/65,
///   65535/65536);
/// * non-integers, which never hit the table however large it is.
///
/// `ln Gamma(3) = ln 2`, so clippy sees the reference value for `x = 2` as a
/// misspelled `f64::consts::LN_2`. It is not: it is what `mpmath` printed.
#[allow(clippy::approx_constant)]
const REF: [(f64, f64); 50] = [
    (0.0, 0.0),
    (1.0, 0.0),
    (2.0, 0.6931471805599453),
    (3.0, 1.791759469228055),
    (4.0, 3.1780538303479458),
    (5.0, 4.787491742782046),
    (7.0, 8.525161361065415),
    (10.0, 15.104412573075516),
    (17.0, 33.50507345013689),
    (20.0, 42.335616460753485),
    (31.0, 78.0922235533153),
    (50.0, 148.47776695177302),
    (63.0, 201.00931639928152),
    (64.0, 205.1681994826412),
    (65.0, 209.34258675253685),
    (100.0, 363.73937555556347),
    (255.0, 1161.7121011184006),
    (1000.0, 5912.128178488163),
    (4096.0, 29978.648060844047),
    (65535.0, 661276.8717651855),
    (65536.0, 661287.9621200744),
    (123457.0, 1323916.216131994),
    (1000000.0, 12815518.384658169),
    (1e-12, -5.772156649007103e-13),
    (1e-08, -5.772156566768626e-09),
    (0.0001, -5.7713342220477625e-05),
    (0.001, -0.0005763935982833696),
    (0.0625, -0.03295710029357782),
    (0.125, -0.06002318412603958),
    (0.25, -0.09827183642181316),
    (0.4, -0.1196129141723713),
    (0.46163214496836236, -0.12148629053584961),
    (0.5, -0.12078223763524522),
    (0.75, -0.08440112102048555),
    (0.9375, -0.02514761940298887),
    (1.0, 0.0),
    (1.0625, 0.027667521522857022),
    (1.25, 0.1248717148923966),
    (1.5, 0.2846828704729192),
    (1.9375, 0.6362508628423761),
    (2.5, 1.2009736023470743),
    (3.75, 2.8085714185757364),
    (7.3, 9.135766871176594),
    (123.456, 474.42143194721274),
    (1234.5678, 7558.151980669254),
    (99999.5, 1051293.4654351394),
    (999999.5, 12815511.476902766),
    (1000000.5, 12815525.292413823),
    (0.9999999, -4.227843026292281e-08),
    (1.0000001, 4.227843675920197e-08),
];

/// The acceptance bound. Measured worst case over the table below and over
/// 6000 uniform samples of `[0, 10^6]` during development: `1.1e-14`.
const REL_TOL: f64 = 1e-13;

fn rel_err(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        (got - want).abs() / want.abs()
    }
}

// ---------------------------------------------------------------------------
// lgamma1p
// ---------------------------------------------------------------------------

#[test]
fn lgamma1p_matches_high_precision_reference_in_and_out_of_table() {
    // `small` tabulates 0..=127 only, so every `REF` point above 127 -- and
    // every non-integer point -- takes the computed path. `big` tabulates
    // 0..=1048575, which swallows all the integer points including 10^6.
    let small = Cache::build(64);
    let big = Cache::build(1_000_000);
    assert!(big.len() > 1_000_000, "10^6 must be a table hit in `big`");
    assert!(small.len() <= 128, "63/64/65 must straddle `small`'s edge");

    for &(x, want) in &REF {
        for (name, c) in [("small", &small), ("big", &big)] {
            let got = c.lgamma1p(x);
            if want == 0.0 {
                // x = 0 and x = 1 are exact zeros of ln Gamma(1 + x), and the
                // implementation is required to return them exactly rather
                // than a rounding of them: a relative bound is vacuous here.
                assert_eq!(got, 0.0, "{name}: lgamma1p({x}) should be exactly 0");
                continue;
            }
            let e = rel_err(got, want);
            assert!(
                e <= REL_TOL,
                "{name}: lgamma1p({x}) = {got}, want {want}, rel err {e:e}"
            );
        }
    }
}

#[test]
fn lgamma1p_table_and_fallback_return_the_same_bits() {
    // The whole cache is only sound if a table hit and a computed value are
    // the same number: a sweep that grows past the table edge must not see a
    // discontinuity in the entropy.
    let small = Cache::build(8);
    let big = Cache::build(4096);
    for i in 0..4096u32 {
        let x = f64::from(i);
        assert_eq!(
            small.lgamma1p(x),
            big.lgamma1p(x),
            "lgamma1p({x}) differs across the table edge"
        );
    }
}

/// An independent check that does not share a line of code with `REF`:
/// `ln(n!) = sum_{i=2}^{n} ln(i)`, accumulated with Neumaier compensation so
/// the summation error stays at one rounding of the total.
#[test]
fn lgamma1p_matches_a_compensated_log_factorial() {
    let c = Cache::build(1 << 12);
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for i in 1..=4096u32 {
        let want = sum + comp;
        let got = c.lgamma1p(f64::from(i - 1));
        if i > 1 {
            let e = rel_err(got, want);
            assert!(e <= REL_TOL, "ln({}!) = {got}, want {want}, rel {e:e}", i - 1);
        }
        // Neumaier: add ln(i) to the running ln((i-1)!) to get ln(i!).
        let term = f64::from(i).ln();
        let t = sum + term;
        comp += if sum.abs() >= term.abs() {
            (sum - t) + term
        } else {
            (term - t) + sum
        };
        sum = t;
    }
}

#[test]
fn lgamma1p_is_not_the_logarithm() {
    // The failure mode this unit exists to make impossible.
    let c = Cache::build(1 << 10);
    // n = 2 is excluded on purpose: ln Gamma(3) = ln(2) = ln(2!), so the
    // stub and the truth genuinely coincide there and the check would pass
    // for the wrong reason.
    for &n in &[5.0f64, 10.0, 100.0, 1000.0] {
        assert!(
            (c.lgamma1p(n) - n.ln()).abs() > 0.1,
            "lgamma1p({n}) collapsed onto ln({n})"
        );
    }
    // ln(10!) = ln(3628800).
    assert!(rel_err(c.lgamma1p(10.0), 3_628_800.0f64.ln()) <= REL_TOL);
    // lgamma1p is convex with a single interior minimum near 0.4616321.
    assert!(c.lgamma1p(0.4616321449683623) < c.lgamma1p(0.2));
    assert!(c.lgamma1p(0.4616321449683623) < c.lgamma1p(0.8));
}

/// `cache.hh:37-49` installs `ignore_error` on the domain, pole, overflow and
/// evaluation handlers precisely so that the SBM's out-of-domain probes come
/// back as NaN instead of throwing. Nothing here may panic.
#[test]
fn lgamma1p_follows_the_ignore_error_policy() {
    let c = Cache::build(16);

    // Poles: x + 1 a non-positive integer.
    for &x in &[-1.0f64, -2.0, -3.0, -17.0] {
        assert!(c.lgamma1p(x).is_nan(), "lgamma1p({x}) should be NaN (pole)");
    }
    // Negative non-integers are in the domain of ln|Gamma|.
    // ln Gamma(0.5) = ln(sqrt(pi)).
    assert!(rel_err(c.lgamma1p(-0.5), std::f64::consts::PI.sqrt().ln()) <= REL_TOL);
    // ln|Gamma(-0.5)| = ln(2 sqrt(pi)).
    assert!(rel_err(c.lgamma1p(-1.5), (2.0 * std::f64::consts::PI.sqrt()).ln()) <= REL_TOL);

    // Infinities and NaN.
    assert_eq!(c.lgamma1p(f64::INFINITY), f64::INFINITY);
    assert!(c.lgamma1p(f64::NEG_INFINITY).is_nan());
    assert!(c.lgamma1p(f64::NAN).is_nan());

    // Huge finite arguments do not overflow: ln Gamma grows like z ln z.
    assert!(c.lgamma1p(1e300).is_finite());
}

#[test]
fn lgamma1p_does_not_truncate_a_non_integer_index() {
    // `get_cached` (cache.hh:73-81) indexes with an integral `x`. This API
    // takes f64, so a table hit must require integrality, not truncation.
    let c = Cache::build(1 << 10);
    assert!(
        (c.lgamma1p(2.5) - c.lgamma1p(2.0)).abs() > 0.1,
        "lgamma1p(2.5) truncated to the lgamma1p(2) table slot"
    );
    assert!(rel_err(c.lgamma1p(2.5), 1.2009736023470743) <= REL_TOL);
}

// ---------------------------------------------------------------------------
// safelog / xlogx
// ---------------------------------------------------------------------------

#[test]
fn safelog_of_zero_is_zero() {
    // `cache.hh:99-104`. Both through the table and through the fallback.
    assert_eq!(Cache::build(1024).safelog(0.0), 0.0);
    assert_eq!(Cache::build(0).safelog(0.0), 0.0);
    assert_eq!(Cache::build(1024).xlogx(0.0), 0.0);
}

#[test]
fn safelog_matches_ln_in_and_out_of_table() {
    let small = Cache::build(4);
    let big = Cache::build(1 << 12);
    for i in 1..=4096u32 {
        let x = f64::from(i);
        assert_eq!(small.safelog(x), x.ln(), "safelog({x}) off table");
        assert_eq!(big.safelog(x), x.ln(), "safelog({x}) on table");
    }
    for &x in &[0.5f64, 1.5, 2.5, 1e-9, 1e9, 7.3] {
        assert_eq!(big.safelog(x), x.ln());
    }
    // Negative: ln of a negative is NaN, as std::log is in C++.
    assert!(big.safelog(-1.0).is_nan());
    assert_eq!(big.safelog(f64::INFINITY), f64::INFINITY);
}

#[test]
fn xlogx_is_x_times_safelog() {
    // `cache.hh:119-124` computes the tabulated xlogx entries exactly this
    // way, so the values must agree bit for bit.
    let c = Cache::build(1 << 10);
    for i in 0..2048u32 {
        let x = f64::from(i);
        assert_eq!(c.xlogx(x), x * c.safelog(x));
    }
}

// ---------------------------------------------------------------------------
// lbinom
// ---------------------------------------------------------------------------

/// Exact `binom(n, k)` by Pascal's rule. `binom(60, 30) = 118264581564861424`
/// fits in a `u64`; `u128` leaves room to spare.
fn exact_binom(n: u32, k: u32) -> u128 {
    if k > n {
        return 0;
    }
    let mut row = vec![0u128; (n + 1) as usize];
    row[0] = 1;
    for i in 1..=n as usize {
        for j in (1..=i).rev() {
            row[j] += row[j - 1];
        }
    }
    row[k as usize]
}

#[test]
fn lbinom_matches_exact_bigint_for_n_up_to_60() {
    let c = Cache::build(1 << 8);
    let mut checked = 0usize;
    for n in 1..=60u32 {
        for k in 1..n {
            let exact = exact_binom(n, k);
            assert!(exact > 0);
            // The exact integer is below 2^57, so its f64 image carries a
            // relative error of at most 2^-53; its logarithm is therefore a
            // reference good to ~1e-17 absolute.
            let want = (exact as f64).ln();
            let got = c.lbinom(f64::from(n), f64::from(k));
            let e = rel_err(got, want);
            assert!(
                e <= 1e-13,
                "lbinom({n}, {k}) = {got}, want ln({exact}) = {want}, rel {e:e}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 1770); // sum_{n=1}^{60} (n - 1)
    // The largest central coefficient, spelled out.
    assert_eq!(exact_binom(60, 30), 118_264_581_564_861_424);
}

#[test]
fn lbinom_guard_cases_match_util_hh() {
    // `util.hh:44`: N == 0 || k == 0 || k >= N  ->  0.
    let c = Cache::build(1 << 8);
    assert_eq!(c.lbinom(0.0, 0.0), 0.0);
    assert_eq!(c.lbinom(0.0, 5.0), 0.0);
    assert_eq!(c.lbinom(10.0, 0.0), 0.0);
    assert_eq!(c.lbinom(10.0, 10.0), 0.0);
    assert_eq!(c.lbinom(10.0, 11.0), 0.0);
    // binom(n, 1) = n.
    for n in 2..200u32 {
        let e = rel_err(c.lbinom(f64::from(n), 1.0), f64::from(n).ln());
        assert!(e <= 1e-13, "lbinom({n}, 1) rel {e:e}");
    }
    // Symmetry, out of the table as well as in it. The bound is absolute and
    // scaled by ln Gamma(n + 1), not relative to the answer, because
    // `lbinom_fast` is a difference of terms of that size: at n = 10^6 the
    // three lgammas are ~1.3e7 and the answer is ~40, so nine digits of the
    // subtraction cancel. `lbinom_careful` (util.hh:49-68) exists for exactly
    // this regime, and its Stirling branch carries a sign error (`- N *
    // log1p(-k/N) - k * log1p(-k/N)`, :62, where the second term should be
    // `+ k * log1p(-k/N)`), which is why nothing here routes through it.
    for &(n, k) in &[(1e6f64, 3.0), (1e6, 500.0), (12345.0, 6172.0)] {
        let a = c.lbinom(n, k);
        let b = c.lbinom(n, n - k);
        let bound = 1e-15 * c.lgamma1p(n);
        assert!((a - b).abs() <= bound, "lbinom({n},{k}) = {a} vs {b}");
    }
}

// ---------------------------------------------------------------------------
// log_q
// ---------------------------------------------------------------------------

/// Exact `q(n, k)`: partitions of `n` into at most `k` parts.
/// `q_rec`, `int_part.cc:61-72`, without the memo.
fn exact_q(n_max: usize) -> Vec<Vec<u128>> {
    // q[n][k] = q[n][k-1] + q[n-k][min(k, n-k)], q[n][1] = 1, q[0][0] = 1.
    let mut q = vec![vec![0u128; n_max + 1]; n_max + 1];
    q[0][0] = 1;
    for n in 1..=n_max {
        q[n][1] = 1;
        for k in 2..=n {
            q[n][k] = q[n][k - 1] + q[n - k][k.min(n - k)];
        }
    }
    q
}

#[test]
fn log_q_matches_the_exact_partition_recursion() {
    const N: usize = 200;
    let c = Cache::build(N);
    let q = exact_q(N);

    // p(200) = 3972999029388: the recursion is being checked against a number
    // the caller can look up, not against itself.
    assert_eq!(q[200][200], 3_972_999_029_388);
    assert_eq!(q[4][4], 5);
    assert_eq!(q[10][10], 42);

    for (n, row) in q.iter().enumerate() {
        for (k, &want) in row.iter().enumerate().take(n + 1) {
            let got = c.log_q(n, k);
            if want == 0 {
                assert_eq!(got, f64::NEG_INFINITY, "log_q({n}, {k})");
            } else {
                let e = rel_err(got, (want as f64).ln());
                assert!(
                    e <= 1e-12 || got.abs() < 1e-15,
                    "log_q({n}, {k}) = {got}, want ln({want}), rel {e:e}"
                );
            }
        }
        // int_part.hh:43: k is clamped to n before anything else.
        assert_eq!(c.log_q(n, n + 7), c.log_q(n, n));
    }
}

#[test]
fn log_q_edge_cases_match_int_part_hh() {
    let c = Cache::build(64);
    // `int_part.hh:45-48`.
    assert_eq!(c.log_q(0, 0), 0.0);
    assert_eq!(c.log_q(5, 0), f64::NEG_INFINITY);
    assert_eq!(c.log_q(0, 5), 0.0); // k clamps to n = 0 first
    // q(n, 1) = 1 for every n >= 1.
    for n in 1..64 {
        assert_eq!(c.log_q(n, 1), 0.0);
    }
}

#[test]
fn log_q_falls_through_to_the_asymptotic_above_the_table() {
    // Below the cap the table answers exactly; above it, `log_q_approx`
    // (int_part.cc:129-139) does, and the two must be of one piece. The
    // asymptotic is an asymptotic, so the bound here is on its documented
    // accuracy, not on rounding.
    let tabulated = Cache::build(400);
    let approx = Cache::build(8); // q_max = 8: everything below is the approximation
    for n in [50usize, 100, 200, 300, 400] {
        for k in [4usize, 10, n / 2, n] {
            let exact = tabulated.log_q(n, k);
            let est = approx.log_q(n, k);
            assert!(est.is_finite(), "log_q_approx({n}, {k}) = {est}");
            let e = rel_err(est, exact);
            assert!(e < 0.05, "log_q_approx({n}, {k}) = {est}, exact {exact}, rel {e:e}");
        }
    }
}

// ---------------------------------------------------------------------------
// The point of the whole exercise: no global, no race.
// ---------------------------------------------------------------------------

#[test]
fn cache_is_sync_and_threads_agree() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Cache>();

    let cache = Cache::build(1 << 12);
    let probe = |c: &Cache| -> Vec<f64> {
        let mut out = Vec::with_capacity(4 * 4096);
        for i in 0..4096u32 {
            let x = f64::from(i);
            out.push(c.lgamma1p(x));
            out.push(c.safelog(x));
            out.push(c.lbinom(x + 64.0, 32.0));
            out.push(c.log_q(i as usize % 1024, 7));
        }
        // Out of table on purpose: the fallback must be reentrant too.
        for i in 0..512u32 {
            let x = 5000.0 + f64::from(i) * 1.5;
            out.push(c.lgamma1p(x));
            out.push(c.safelog(x));
        }
        out
    };

    let expected = probe(&cache);
    let (a, b) = thread::scope(|s| {
        let ta = s.spawn(|| probe(&cache));
        let tb = s.spawn(|| probe(&cache));
        (ta.join().unwrap(), tb.join().unwrap())
    });
    assert_eq!(a, b, "two concurrent readers disagreed");
    assert_eq!(a, expected, "a concurrent reader disagreed with the serial one");
}

#[test]
fn cloning_a_cache_preserves_every_value() {
    let a = Cache::build(1 << 9);
    let b = a.clone();
    for i in 0..1024u32 {
        let x = f64::from(i);
        assert_eq!(a.lgamma1p(x), b.lgamma1p(x));
        assert_eq!(a.safelog(x), b.safelog(x));
    }
}
