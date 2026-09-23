//! Tabulated `lgamma` and `log`.
//!
//! `support/cache.hh:55-57` declares **three global mutable**
//! `std::vector<double>` (`__safelog_cache`, `__xlogx_cache`,
//! `__lgamma_cache`); `init_cache` (`:83-95`) resizes them at `:90` and fills
//! them inside a `#pragma omp parallel for` at `:91-93` while `get_cached`
//! (`:73-81`) reads them at `:80`. Worse, `get_cached` (`:74`),
//! `safelog_fast` (`:106-111`) and `lgamma_fast` (`:139-144`) are annotated
//! `[[gnu::const]]`, which promises the optimiser that they read no memory --
//! licensing hoisting across the very resize that races with them. That is
//! defect #34 in DESIGN.md §17.
//!
//! (The skeleton's original text cited `:56-58`, `:83-91` and `:73`, and the
//! bench header cited `:55-57`, `:84-92` and `:72-73`. The constructs are the
//! same three; the offsets above were re-read out of the 3.8 tree and are the
//! ones this module is written against. U22 owns the reconciliation, so this
//! is it.)
//!
//! Here the tables are one immutable, `Sync` value, threaded explicitly. There
//! is no global to race on and no attribute to lie with.
//!
//! # What is tabulated, and what is not
//!
//! Two tables, not three. `__xlogx_cache` is dropped: [`Cache::xlogx`] is
//! `x * safelog(x)`, one multiply on top of a table this cache already holds,
//! and a third array of the same length buys nothing but a second stream of
//! cache misses. The values are bit-identical to the C++'s, because
//! `xlogx(y) = y * safelog(y)` is exactly how `cache.hh:121-124` computes the
//! entries it stores.
//!
//! The restricted-partition table `__q_cache` (`int_part.cc:30-54`) *is* kept,
//! but triangular and capped -- see [`Cache::log_q`].
//!
//! # The slow paths are on the hot path
//!
//! graph-tool deliberately feeds `lgamma` out-of-domain values and installs an
//! `ignore_error` policy (`cache.hh:37-49`) so that they come back as NaN
//! instead of throwing. Everything below returns NaN or an infinity where
//! `boost::math` with that policy would, and **nothing here panics** -- there
//! is no `assert`, no unwrap, and every table read goes through
//! `slice::get`, so no `panic_bounds_check` survives into the kernels.

use std::f64::consts::{LN_2, PI};

use rayon::prelude::*;

/// Ceiling on the dimension of the restricted-partition table.
///
/// `state.hh:242` and `planted_partition/spec.hh:127` both call
/// `init_q_cache(std::min(std::max(2 * int(_E), 100), 10000))`, and
/// `__q_cache` is a **square** `boost::multi_array`: at that ceiling the C++
/// allocates `10001 * 10001` doubles, 800 MB, to store a table that is zero
/// above the diagonal.
///
/// Here the table is triangular (`k <= n`) and this constant caps `n`, so the
/// worst case is `1025 * 1026 / 2` doubles, 4.2 MB. Above it, `log_q` falls
/// through to the same asymptotic `log_q_approx` (`int_part.cc:129-139`) the
/// C++ uses whenever `n` exceeds whatever `init_q_cache` was last called with.
/// See `deviations`.
const Q_MAX: usize = 1024;

/// `ln(sqrt(2*pi))`.
const LN_SQRT_2PI: f64 = 0.9189385332046727;

/// Euler-Mascheroni constant.
const EULER: f64 = 0.5772156649015329;

/// Lanczos parameter `g = 607/128` (Godfrey's `N = 15` set).
const LANCZOS_G: f64 = 607.0 / 128.0;

/// Godfrey's 15 Lanczos coefficients for `g = 607/128`.
///
/// Measured against a 60-digit `mpmath` reference over `[0, 10^6]`: worst
/// relative error `7.8e-15` on `ln Gamma`, and `1.1e-14` on `lgamma1p` once
/// the two zeros are excised by the series below.
const LANCZOS: [f64; 15] = [
    0.9999999999999971,
    57.15623566586292,
    -59.59796035547549,
    14.136097974741746,
    -0.4919138160976202,
    3.399464998481189e-05,
    4.652362892704858e-05,
    -9.837447530487956e-05,
    0.0001580887032249125,
    -0.00021026444172410488,
    0.00021743961811521265,
    -0.0001643181065367639,
    8.441822398385275e-05,
    -2.6190838401581408e-05,
    3.6899182659531625e-06,
];

/// Half-width of the window around each zero of `lgamma1p` in which the
/// Taylor series is used instead of Lanczos.
///
/// `lgamma1p` vanishes at `x = 0` and `x = 1`, so near those points a method
/// with good *absolute* error has unbounded *relative* error. Inside this
/// window the series is evaluated instead, and it has no cancellation: its
/// leading term is `-EULER * x`.
const SERIES_WINDOW: f64 = 0.25;

/// `(-1)^j * zeta(j + 2) / (j + 2)` for `j = 0 ..= 34`.
///
/// `ln Gamma(1 + x) = -gamma*x + sum_{k>=2} (-1)^k zeta(k) x^k / k`, `|x| < 1`.
/// At `|x| = SERIES_WINDOW` the first omitted term is `0.25^37 / 37 < 1e-23`,
/// which is below the rounding of the leading term by five orders of
/// magnitude.
const ZETA_SERIES: [f64; 35] = [
    0.8224670334241132,
    -0.40068563438653143,
    0.27058080842778454,
    -0.20738555102867398,
    0.1695571769974082,
    -0.1440498967688461,
    0.12550966952474304,
    -0.11133426586956469,
    0.1000994575127818,
    -0.09095401714582904,
    0.083353840546109,
    -0.0769325164113522,
    0.07143294629536133,
    -0.06666870588242046,
    0.06250095514121304,
    -0.058823978658684585,
    0.055555767627403614,
    -0.05263167937961666,
    0.05000004769810169,
    -0.047619070330142226,
    0.04545455629320467,
    -0.04347826605304026,
    0.04166666915034121,
    -0.04000000119214014,
    0.03846153903467518,
    -0.037037037312989324,
    0.035714285847333355,
    -0.034482758684919304,
    0.03333333336437758,
    -0.03225806453115042,
    0.03125000000727597,
    -0.030303030306558044,
    0.029411764707594344,
    -0.02857142857226011,
    0.027777777778181998,
];

/// Tabulated special functions, immutable after construction.
///
/// `Send + Sync` by construction -- every field is a `Box<[f64]>` or a
/// `usize`, and no method takes `&mut self`. One `Cache` is shared by every
/// thread in a sweep; see `tests/u22_cache.rs::cache_is_sync_and_threads_agree`.
#[derive(Clone, Debug)]
pub struct Cache {
    /// `lgamma[i] == ln Gamma(i + 1) == ln(i!)`. Length is a power of two.
    lgamma: Box<[f64]>,
    /// `safelog[i] == ln(i)`, and `safelog[0] == 0` (`cache.hh:99-104`).
    safelog: Box<[f64]>,
    /// Triangular `log q(n, k)` for `n <= q_max`, `k <= n`; row `n` starts at
    /// `n * (n + 1) / 2`.
    q: Box<[f64]>,
    /// Largest `n` for which `q` holds an exact value.
    q_max: usize,
}

impl Cache {
    /// Tabulate up to `n`.
    ///
    /// The tables are sized `get_next_size(n + 1)` (`cache.hh:64-71`), i.e.
    /// the smallest power of two that makes index `n` valid, exactly as
    /// `init_cache` (`:86-90`) sizes them. Unlike the C++ this happens once,
    /// in a constructor, and never again: there is no `init_*_fast` call to
    /// forget and no resize to race with a reader.
    ///
    /// The fill is parallel, as `cache.hh:91` is; it is a pure `map` over a
    /// disjoint index range, so the result does not depend on the thread
    /// count.
    pub fn build(n: usize) -> Self {
        let len = n
            .saturating_add(1)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX);

        let lgamma: Vec<f64> = (0..len)
            .into_par_iter()
            .map(|i| lgamma1p_slow(i as f64))
            .collect();
        let safelog: Vec<f64> = (0..len)
            .into_par_iter()
            .map(|i| safelog_slow(i as f64))
            .collect();

        let q_max = n.min(Q_MAX);
        Self {
            lgamma: lgamma.into_boxed_slice(),
            safelog: safelog.into_boxed_slice(),
            q: build_q(q_max),
            q_max,
        }
    }

    /// The number of tabulated entries; index `i < len()` is a table hit.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.lgamma.len()
    }

    /// Always false -- `build` allocates at least one entry.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lgamma.is_empty()
    }

    /// `lgamma(x + 1)`, falling back to a real `lgamma` out of range.
    ///
    /// graph-tool installs an `ignore_error` policy on `boost::math::lgamma`
    /// (`cache.hh:37-49`) because the SBM deliberately feeds it out-of-domain
    /// values; this matches that behaviour rather than panicking:
    ///
    /// * a pole (`x + 1` a non-positive integer, so `x` a negative integer)
    ///   gives NaN, as `raise_pole_error` under `ignore_error` does;
    /// * `-inf` gives NaN (domain error), `+inf` gives `+inf`;
    /// * NaN in, NaN out.
    ///
    /// `get_cached` (`cache.hh:73-81`) indexes the table with an *integral*
    /// `x`. This takes `f64`, so it checks integrality before indexing: a
    /// truncating index would answer `lgamma1p(2.5)` with `ln(2!)`, which is
    /// not what any C++ call site asks for, because every C++ call site passes
    /// an integer.
    #[inline]
    pub fn lgamma1p(&self, x: f64) -> f64 {
        if x >= 0.0 {
            let i = x as usize;
            if i as f64 == x
                && let Some(&v) = self.lgamma.get(i)
            {
                return v;
            }
        }
        lgamma1p_slow(x)
    }

    /// `log(x)`, returning 0 at `x == 0` (`safelog_fast`, `cache.hh:106-111`).
    #[inline]
    pub fn safelog(&self, x: f64) -> f64 {
        if x >= 0.0 {
            let i = x as usize;
            if i as f64 == x
                && let Some(&v) = self.safelog.get(i)
            {
                return v;
            }
        }
        safelog_slow(x)
    }

    /// `x * log(x)`, zero at `x == 0` (`xlogx_fast`, `cache.hh:126-131`).
    ///
    /// The C++ keeps a third table for this. Here it is one multiply over the
    /// `safelog` table, which is how the C++ computes the entries it stores
    /// (`cache.hh:119-124`), so the values agree bit for bit.
    #[inline]
    pub fn xlogx(&self, x: f64) -> f64 {
        x * self.safelog(x)
    }

    /// `log(binom(n, k))`. `support/util.hh:42-47` (`lbinom_fast`).
    ///
    /// The three guard cases and the association of the two subtractions are
    /// copied from `:44-46`; the association is load-bearing because these are
    /// large cancelling terms and regrouping them changes the last bits.
    #[inline]
    pub fn lbinom(&self, n: f64, k: f64) -> f64 {
        if n == 0.0 || k == 0.0 || k >= n {
            return 0.0;
        }
        (self.lgamma1p(n) - self.lgamma1p(k)) - self.lgamma1p(n - k)
    }

    /// `log q(n, k)`, the restricted-partition count. `support/int_part.hh:39-52`.
    ///
    /// `q(n, k)` is the number of partitions of `n` into at most `k` parts
    /// (`q_rec`, `int_part.cc:61-72`). Exact from the table while
    /// `n <= Q_MAX`, and the asymptotic `log_q_approx` (`:129-139`) above it --
    /// the same two-armed structure as `int_part.hh:49-51`, with a triangular
    /// and capped table in place of a square unbounded one.
    #[inline]
    pub fn log_q(&self, n: usize, k: usize) -> f64 {
        // `int_part.hh:43-48`, minus the `n < 0` arm: `n` is unsigned here.
        let k = k.min(n);
        if n == 0 && k == 0 {
            return 0.0;
        }
        if k == 0 {
            return f64::NEG_INFINITY;
        }
        if n <= self.q_max
            && let Some(&v) = self.q.get(n * (n + 1) / 2 + k)
        {
            return v;
        }
        self.log_q_approx(n, k)
    }

    /// `log_q_approx`, `int_part.cc:129-139`.
    fn log_q_approx(&self, n: usize, k: usize) -> f64 {
        let nf = n as f64;
        let kf = k as f64;
        if kf < nf.powf(0.25) {
            // `log_q_approx_small`, `int_part.cc:110-113`.
            return self.lbinom(nf - 1.0, kf - 1.0) - self.lgamma1p(kf);
        }
        let u = kf / nf.sqrt();
        let v = get_v(u);
        let lf = v.ln()
            - (-(-v).exp() * (1.0 + u * u / 2.0)).ln_1p() / 2.0
            - LN_2 * 3.0 / 2.0
            - u.ln()
            - PI.ln();
        let g = 2.0 * v / u - u * (-(-v).exp()).ln_1p();
        lf - nf.ln() + nf.sqrt() * g
    }
}

// ---------------------------------------------------------------------------
// The uncached kernels.
// ---------------------------------------------------------------------------

/// `safelog`, `cache.hh:97-104`: `log(x)`, and 0 at `x == 0`.
#[inline]
fn safelog_slow(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x.ln() }
}

/// `ln Gamma(x + 1)`, computed rather than looked up.
///
/// Three regimes, chosen so that the *relative* error is bounded everywhere
/// and not merely the absolute one. `lgamma1p` has a zero at `x = 0` and
/// another at `x = 1`; a plain `ln_gamma(x + 1.0)` is accurate to a few
/// units in the last place of a quantity that is itself near zero there, so
/// its relative error blows up. Inside `SERIES_WINDOW` of each zero the
/// Taylor series is used instead: its leading term is `-EULER * x`, there is
/// no cancellation, and the relative error stays at rounding.
fn lgamma1p_slow(x: f64) -> f64 {
    // Exact, and the two values the series and `ln_gamma` would only round to.
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    if x.abs() <= SERIES_WINDOW {
        return lgamma1p_series(x);
    }
    let t = x - 1.0;
    if t.abs() <= SERIES_WINDOW {
        // ln Gamma(2 + t) = ln(1 + t) + ln Gamma(1 + t); both terms have the
        // sign of t, so the sum does not cancel.
        return t.ln_1p() + lgamma1p_series(t);
    }
    ln_gamma(x + 1.0)
}

/// `ln Gamma(1 + x) = -gamma*x + sum_{k>=2} (-1)^k zeta(k) x^k / k`.
///
/// Only called for `|x| <= SERIES_WINDOW`, where it is converged to well
/// below rounding.
#[inline]
fn lgamma1p_series(x: f64) -> f64 {
    let mut p = 0.0;
    for &c in ZETA_SERIES.iter().rev() {
        p = p * x + c;
    }
    (-EULER) * x + x * x * p
}

/// `ln|Gamma(z)|` for real `z`, with graph-tool's `ignore_error` policy
/// (`cache.hh:37-49`) at the poles and at the domain edge.
///
/// `boost::math::lgamma` under `pole_error<ignore_error>` returns NaN at the
/// non-positive integers and under `domain_error<ignore_error>` returns NaN
/// for `-inf`; both are reproduced here, and neither panics.
fn ln_gamma(z: f64) -> f64 {
    if z.is_nan() {
        return z;
    }
    if z == f64::INFINITY {
        return f64::INFINITY;
    }
    if z <= 0.0 {
        // `-inf` is a domain error; a non-positive integer is a pole. Both are
        // NaN under `ignore_error`.
        if z == f64::NEG_INFINITY || z == z.floor() {
            return f64::NAN;
        }
        // Reflection: ln|Gamma(z)| = ln(pi) - ln|sin(pi z)| - ln Gamma(1 - z).
        // `sin(PI * z)` loses every bit of its argument for large |z|, so the
        // range reduction is done first: |sin(pi z)| = sin(pi * frac(z)) with
        // frac(z) in (0, 1).
        let frac = z - z.floor();
        let s = (PI * frac).sin();
        return PI.ln() - s.ln() - ln_gamma(1.0 - z);
    }
    ln_gamma_lanczos(z)
}

/// Lanczos, `g = 607/128`, 15 terms, for `z > 0`.
///
/// `Gamma(z) = sqrt(2 pi) * base^(z - 1/2) * e^-base * A(z)` with
/// `base = z + g - 1/2` and `A(z) = c0 + sum_k c_k / (z - 1 + k)`; taken in
/// logs so that neither `base^(z - 1/2)` nor `e^-base` ever overflows.
#[inline]
fn ln_gamma_lanczos(z: f64) -> f64 {
    let zm1 = z - 1.0;
    let mut sum = LANCZOS[0];
    for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
        sum += c / (zm1 + i as f64);
    }
    let base = zm1 + LANCZOS_G + 0.5;
    LN_SQRT_2PI + (zm1 + 0.5) * base.ln() - base + sum.ln()
}

/// `log_sum_exp`, `support/util.hh:77-90`, with `handle_inf = true`.
///
/// The `a == b` arm is what makes `-inf + -inf` come out as `-inf` rather than
/// NaN, which the `q` recurrence below depends on: every cell starts at
/// `-inf`.
#[inline]
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == b {
        return a + LN_2;
    }
    if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

/// `get_v`, `int_part.cc:115-127`.
///
/// Fixed-point iteration on `v = u * sqrt(spence(exp(-v)))`. The C++ loop has
/// no iteration bound; this one does, so that a `log_q` on a pathological `u`
/// cannot hang a sweep. In practice it converges in under twenty.
fn get_v(u: f64) -> f64 {
    const EPSILON: f64 = 1e-8;
    const MAX_ITER: usize = 1_000;
    let mut v = u;
    let mut delta = 1.0;
    let mut iter = 0;
    while delta > EPSILON && iter < MAX_ITER {
        let n_v = u * spence((-v).exp()).sqrt();
        delta = (n_v - v).abs();
        v = n_v;
        iter += 1;
    }
    v
}

/// Numerator of the Cephes `spence` rational approximation (`spence.cc:69-78`).
///
/// The literals are the C++ file's own digits, character for character, so
/// that the transcription can be diffed against it; they carry more decimals
/// than an `f64` holds, which is what the `allow` is for.
#[allow(clippy::excessive_precision)]
const SPENCE_A: [f64; 8] = [
    4.65128586073990045278E-5,
    7.31589045238094711071E-3,
    1.33847639578309018650E-1,
    8.79691311754530315341E-1,
    2.71149851196553469920E0,
    4.25697156008121755724E0,
    3.29771340985225106936E0,
    1.0,
];

/// Denominator of the Cephes `spence` rational approximation (`spence.cc:80-89`).
#[allow(clippy::excessive_precision)]
const SPENCE_B: [f64; 8] = [
    6.90990488912553276999E-4,
    2.54043763932544379113E-2,
    2.82974860602568089943E-1,
    1.41172597751831069617E0,
    3.63800533345137075418E0,
    5.03278880143316990390E0,
    3.54771340985225096217E0,
    1.0,
];

/// `polevl`, `spence.cc:91-105`.
#[inline]
fn polevl(x: f64, coef: &[f64; 8]) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

/// Cephes dilogarithm, `spence.cc:107-150`, transcribed.
///
/// `spence(x) = -integral_1^x ln(t)/(t - 1) dt`, for `x >= 0`; NaN below zero,
/// which is what the C++ returns too (`:113-114`).
fn spence(x: f64) -> f64 {
    let mut x = x;
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 1.0 {
        return 0.0;
    }
    if x == 0.0 {
        return PI * PI / 6.0;
    }

    let mut flag = 0u32;
    if x > 2.0 {
        x = 1.0 / x;
        flag |= 2;
    }

    let w = if x > 1.5 {
        flag |= 2;
        (1.0 / x) - 1.0
    } else if x < 0.5 {
        flag |= 1;
        -x
    } else {
        x - 1.0
    };

    let mut y = -w * polevl(w, &SPENCE_A) / polevl(w, &SPENCE_B);

    if flag & 1 != 0 {
        y = (PI * PI) / 6.0 - x.ln() * (-x).ln_1p() - y;
    }
    if flag & 2 != 0 {
        let z = x.ln();
        y = -0.5 * z * z - y;
    }
    y
}

/// The triangular `log q(n, k)` table, `int_part.cc:32-54`.
///
/// The C++ builds a square `(n_max + 1)^2` array and fills the whole thing
/// with `-inf` first; the half above the diagonal is never written and never
/// read, because `log_q` clamps `k` to `n` before indexing. Storing only
/// `k <= n` halves the memory and, more usefully, keeps a row contiguous.
fn build_q(q_max: usize) -> Box<[f64]> {
    let len = (q_max + 1) * (q_max + 2) / 2;
    let mut q = vec![f64::NEG_INFINITY; len];
    // `int_part.cc:42`.
    q[0] = 0.0;
    for n in 1..=q_max {
        let row = n * (n + 1) / 2;
        // `int_part.cc:46`: a partition of n into at most one part.
        q[row + 1] = 0.0;
        for k in 2..=n {
            // `:49` -- q[n][k] is still -inf, so this is q[n][k-1].
            let mut v = log_sum_exp(q[row + k], q[row + k - 1]);
            // `:50-51`. `n >= k` holds for the whole loop.
            let m = k.min(n - k);
            let prev = (n - k) * (n - k + 1) / 2 + m;
            v = log_sum_exp(v, q[prev]);
            q[row + k] = v;
        }
    }
    q.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `get_next_size`, `cache.hh:64-71`: `k = 1; while (k < n) k <<= 1`.
    #[test]
    fn table_length_is_the_next_power_of_two() {
        assert_eq!(Cache::build(0).len(), 1);
        assert_eq!(Cache::build(1).len(), 2);
        assert_eq!(Cache::build(2).len(), 4);
        assert_eq!(Cache::build(3).len(), 4);
        assert_eq!(Cache::build(4).len(), 8);
        assert_eq!(Cache::build(1023).len(), 1024);
        assert_eq!(Cache::build(1024).len(), 2048);
    }

    #[test]
    fn spence_matches_known_values() {
        // spence(0) = pi^2/6, spence(1) = 0, spence(2) = -pi^2/12.
        assert_eq!(spence(1.0), 0.0);
        assert!((spence(0.0) - PI * PI / 6.0).abs() < 1e-15);
        assert!((spence(2.0) + PI * PI / 12.0).abs() < 1e-9);
        // spence(x) + spence(1/x) = -ln(x)^2 / 2  (spence.cc:124-127, :146-149)
        for &x in &[0.3f64, 0.7, 1.3, 3.0, 11.0] {
            let lhs = spence(x) + spence(1.0 / x);
            let rhs = -0.5 * x.ln() * x.ln();
            assert!((lhs - rhs).abs() < 1e-9, "x = {x}: {lhs} vs {rhs}");
        }
        assert!(spence(-1.0).is_nan());
    }

    #[test]
    fn log_sum_exp_absorbs_negative_infinity() {
        assert_eq!(
            log_sum_exp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
        assert_eq!(log_sum_exp(f64::NEG_INFINITY, 0.0), 0.0);
        assert_eq!(log_sum_exp(0.0, f64::NEG_INFINITY), 0.0);
        assert!((log_sum_exp(0.0, 0.0) - LN_2).abs() < 1e-16);
    }

    #[test]
    fn lanczos_and_series_agree_where_both_are_valid() {
        // Just outside the series window, the two regimes must meet.
        let x = SERIES_WINDOW;
        let a = lgamma1p_series(x);
        let b = ln_gamma(x + 1.0);
        assert!((a - b).abs() / a.abs() < 1e-13, "{a} vs {b}");
    }
}
