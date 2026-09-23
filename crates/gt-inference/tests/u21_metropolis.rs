//! U21 -- the spec traits and the Metropolis loop.
//!
//! Everything here is checked against `inference/loops/mcmc_loop.hh` and
//! `inference/base/mcmc.hh`, line by line:
//!
//! | subject | C++ | defect |
//! |---|---|---|
//! | `accept` | `mcmc_loop.hh:79-99` | — |
//! | node multiset per schedule | `:121-122`, `:136-139`, `:196-197` | #20 |
//! | `nattempts` / `nmoves` | `:124`, `:161`, `:178`, `:184` | #25 |
//! | the null-move test | `:168-173`, `base/mcmc.hh:104-107` | #21 |
//! | `skip_node`, `init_iter`, `after_step` | `:134`, `:141`, `:189` | #21 |
//! | the misnamed density | `potts/spec.hh:124` (trybuild) | #19 |
//!
//! Two deliberate choices are load-bearing for the sweep tests below and are
//! stated once here.
//!
//! **`beta = ∞` never draws.** `mcmc_loop.hh:82-85` returns `dS < 0` without
//! touching the generator, so a sweep at infinite `beta` consumes the stream
//! only for shuffling and node selection. Every ordering test uses that: the
//! accept/reject decision is a function of `d_entropy`'s sign alone, and the
//! permutation a seed produces cannot be perturbed by how many moves happened
//! to be accepted.
//!
//! **The toy state never mutates its own group labels in `perform`.** A real
//! state does, and `node_state` then changes underneath the next proposal.
//! Keeping the labels fixed makes "which nodes were visited" independent of
//! "which moves were accepted", which is what these tests are about; the one
//! test that does care ([`a_move_to_the_current_state_is_a_null_move`]) sets
//! the label by hand.

use std::collections::BTreeMap;

use gt_inference::metropolis::{
    MetropolisState, MoveScore, Schedule, Step, SweepResult, accept, mcmc_sweep,
};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ===========================================================================
// 0. Fixtures
// ===========================================================================

/// A generator that returns one `u64` for ever, and counts the draws.
///
/// `accept` is specified as much by *when* it touches the generator as by what
/// it does with the value: `mcmc_loop.hh:82-85` and `:90-91` both return
/// without constructing the `uniform_real_distribution` at all. A shared
/// stream between the loop's node selection and its acceptance test makes that
/// observable, so it is asserted rather than assumed.
///
/// `rand`'s `StandardUniform` for `f64` takes the top 53 bits and scales by
/// `2^-53`, so `u64::MAX` is the largest representable draw,
/// `1 - 2^-53 ≈ 0.999999999999999889`, and `0` is exactly `0.0`.
struct FixedRng {
    value: u64,
    draws: u64,
}

impl FixedRng {
    /// The largest draw `[0, 1)` admits: rejects every `a <= 0`.
    fn high() -> Self {
        FixedRng {
            value: u64::MAX,
            draws: 0,
        }
    }
    /// `0.0`: accepts every `a` whose `exp(a)` is strictly positive.
    fn low() -> Self {
        FixedRng { value: 0, draws: 0 }
    }
}

impl RngCore for FixedRng {
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    fn next_u64(&mut self) -> u64 {
        self.draws += 1;
        self.value
    }
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        for chunk in dst.chunks_mut(8) {
            let v = self.next_u64().to_le_bytes();
            let n = chunk.len();
            chunk.copy_from_slice(&v[..n]);
        }
    }
}

fn score(d_entropy: f64, log_hastings: f64) -> MoveScore {
    MoveScore {
        d_entropy,
        log_hastings,
    }
}

/// A `MetropolisState` that does nothing but record what the loop asked it.
///
/// It is the `MetropolisStateBase` of `mcmc_loop.hh:37-76` with the five
/// `// required` members actually required -- which is the whole point of the
/// trait: the C++ base gives each of them a working body, so a state that
/// misspells `virtual_move` inherits `{0, 0}` and `metropolis_accept` then
/// accepts every move unconditionally.
struct Toy {
    // -- configuration ------------------------------------------------------
    nodes: Vec<usize>,
    n_iter: usize,
    beta: f64,
    schedule: Schedule,
    steps: usize,
    /// Proposals, consumed in order and wrapping.
    script: Vec<Step<i32>>,
    cursor: usize,
    /// `node_state`, one per node index.
    group: Vec<i32>,
    d_entropy: f64,
    log_hastings: f64,
    skip: Vec<usize>,
    init_offset: f64,

    // -- observations -------------------------------------------------------
    visits: Vec<usize>,
    /// The node list as `init_iter` saw it, once per sweep.
    order_at_init: Vec<Vec<usize>>,
    scored: Vec<(usize, i32)>,
    performed: Vec<(usize, i32)>,
    after: Vec<(usize, i32)>,
    init_calls: usize,
}

impl Toy {
    /// `n` nodes `0..n`, one sweep, one always-downhill move per visit, at
    /// `beta = ∞` so the sweep is deterministic given the node order.
    fn new(n: usize) -> Self {
        Toy {
            nodes: (0..n).collect(),
            n_iter: 1,
            beta: f64::INFINITY,
            schedule: Schedule::Alternating,
            steps: n,
            script: vec![Step { mv: 1, nsteps: 1 }],
            cursor: 0,
            group: vec![0; n],
            d_entropy: -1.0,
            log_hastings: 0.0,
            skip: Vec::new(),
            init_offset: 0.0,
            visits: Vec::new(),
            order_at_init: Vec::new(),
            scored: Vec::new(),
            performed: Vec::new(),
            after: Vec::new(),
            init_calls: 0,
        }
    }

    fn schedule(mut self, s: Schedule) -> Self {
        self.schedule = s;
        self
    }
    fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }
    fn steps(mut self, n: usize) -> Self {
        self.steps = n;
        self
    }
    fn script(mut self, s: Vec<Step<i32>>) -> Self {
        self.script = s;
        self
    }
    fn uphill(mut self) -> Self {
        self.d_entropy = 1.0;
        self
    }

    /// The visits of sweep `i`, for a sweep-per-node-list schedule.
    fn sweep(&self, i: usize, len: usize) -> &[usize] {
        &self.visits[i * len..(i + 1) * len]
    }
}

const NULL: i32 = i32::MIN;

impl MetropolisState for Toy {
    type Node = usize;
    type Move = i32;
    const NULL_MOVE: i32 = NULL;

    fn nodes(&mut self) -> &mut Vec<usize> {
        &mut self.nodes
    }
    fn n_iter(&self) -> usize {
        self.n_iter
    }
    fn beta(&self) -> f64 {
        self.beta
    }
    fn schedule(&self) -> Schedule {
        self.schedule
    }
    fn steps_per_iter(&mut self) -> usize {
        self.steps
    }
    fn node_state(&self, v: usize) -> i32 {
        self.group[v]
    }
    fn propose<R: Rng + ?Sized>(&mut self, v: usize, _rng: &mut R) -> Step<i32> {
        self.visits.push(v);
        let s = self.script[self.cursor % self.script.len()];
        self.cursor += 1;
        s
    }
    fn score(&mut self, v: usize, s: i32) -> MoveScore {
        self.scored.push((v, s));
        score(self.d_entropy, self.log_hastings)
    }
    fn perform(&mut self, v: usize, s: i32) {
        self.performed.push((v, s));
    }
    fn skip_node(&self, v: usize) -> bool {
        self.skip.contains(&v)
    }
    fn init_iter<R: Rng + ?Sized>(&mut self, _rng: &mut R) -> f64 {
        self.init_calls += 1;
        self.order_at_init.push(self.nodes.clone());
        self.init_offset
    }
    fn after_step(&mut self, v: usize, s: i32) {
        self.after.push((v, s));
    }
}

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn multiset(xs: &[usize]) -> BTreeMap<usize, usize> {
    let mut m = BTreeMap::new();
    for &x in xs {
        *m.entry(x).or_insert(0) += 1;
    }
    m
}

// ===========================================================================
// 1. `accept`: the two limits (`mcmc_loop.hh:79-99`)
// ===========================================================================

/// `mcmc_loop.hh:82-85`. At `beta = ∞` the test is `dS < 0` -- strictly, so a
/// flat move is *rejected* -- and the Hastings ratio does not enter it at all.
#[test]
fn at_zero_temperature_accept_is_a_strict_downhill_test() {
    for &mp in &[-1e9, -3.0, -0.5, 0.0, 0.5, 3.0, 1e9, f64::NEG_INFINITY] {
        let mut r = FixedRng::high();
        assert!(accept(score(-1e-300, mp), f64::INFINITY, &mut r));
        assert!(!accept(score(0.0, mp), f64::INFINITY, &mut r));
        assert!(!accept(score(1e-300, mp), f64::INFINITY, &mut r));
        assert_eq!(
            r.draws, 0,
            "mcmc_loop.hh:82-85 returns before constructing the distribution"
        );
    }
}

/// The acceptance criterion: at `beta = 0` the entropy drops out and the test
/// is `log_hastings > 0`.
///
/// "Exactly when" is the *deterministic* half, `mcmc_loop.hh:89-92`. The other
/// half is a draw against `exp(mP)`, so the boundary is pinned from both
/// sides: with the largest draw `[0, 1)` admits, every `mP < 0` is rejected;
/// with `0.0`, every finite `mP` is accepted. `mP == 0` is accepted under both
/// because `exp(0) = 1` is strictly greater than every draw -- a
/// Hastings ratio of one is an unconditional accept, which is correct
/// Metropolis and is what C++ does.
#[test]
fn at_infinite_temperature_accept_is_the_hastings_ratio() {
    for &ds in &[-7.0, 0.0, 7.0] {
        for &mp in &[1e-12, 0.5, 3.0, 1e9] {
            let mut hi = FixedRng::high();
            assert!(accept(score(ds, mp), 0.0, &mut hi), "mP = {mp} must accept");
            assert_eq!(hi.draws, 0, "a > 0 returns at mcmc_loop.hh:90-91");
        }
        for &mp in &[-1e-9, -0.5, -3.0, -1e9] {
            let mut hi = FixedRng::high();
            assert!(
                !accept(score(ds, mp), 0.0, &mut hi),
                "mP = {mp} must reject"
            );
            assert_eq!(hi.draws, 1, "one draw, at mcmc_loop.hh:95-96");

            let mut lo = FixedRng::low();
            assert_eq!(
                accept(score(ds, mp), 0.0, &mut lo),
                mp.exp() > 0.0,
                "a draw of 0.0 accepts exactly the moves whose exp(a) is                  strictly positive; exp(-1e9) underflows to 0 and is not one"
            );
        }
        // The exact boundary.
        let mut hi = FixedRng::high();
        assert!(accept(score(ds, 0.0), 0.0, &mut hi));
        assert_eq!(hi.draws, 1);
    }
}

/// `double a = -dS * beta + mP;` (`:88`) -- the sign of `dS`, the product with
/// `beta`, and the *sum* with `mP`, each separately.
#[test]
fn the_exponent_is_minus_ds_times_beta_plus_mp() {
    // -(-2) * 3 + (-6) = 0 -> not > 0 -> draw against exp(0) = 1.
    let mut hi = FixedRng::high();
    assert!(accept(score(-2.0, -6.0), 3.0, &mut hi));
    assert_eq!(hi.draws, 1);

    // -(-2) * 3 + (-6.5) = -0.5 < 0 -> rejected by the maximal draw.
    let mut hi = FixedRng::high();
    assert!(!accept(score(-2.0, -6.5), 3.0, &mut hi));

    // -(-2) * 3 + (-5.5) = +0.5 > 0 -> accepted without a draw.
    let mut hi = FixedRng::high();
    assert!(accept(score(-2.0, -5.5), 3.0, &mut hi));
    assert_eq!(hi.draws, 0);

    // beta scales dS and only dS: -(1) * 0.1 + 0.2 = +0.1 > 0.
    let mut hi = FixedRng::high();
    assert!(accept(score(1.0, 0.2), 0.1, &mut hi));
    assert_eq!(hi.draws, 0);
    // ... while beta = 2 turns the same move downhill-negative.
    let mut hi = FixedRng::high();
    assert!(!accept(score(1.0, 0.2), 2.0, &mut hi));
}

/// The stochastic branch really is `P(accept) = exp(a)`, not `a` or `1 - e^a`
/// or a comparison against the wrong tail. 200k draws of a genuine generator
/// against the three-sigma band.
#[test]
fn the_stochastic_branch_accepts_with_probability_exp_a() {
    const N: usize = 200_000;
    let mut r = rng(0xA21);
    for &p in &[0.05_f64, 0.25, 0.5, 0.9] {
        let a = p.ln();
        let mut n = 0usize;
        for _ in 0..N {
            if accept(score(0.0, a), 0.0, &mut r) {
                n += 1;
            }
        }
        let got = n as f64 / N as f64;
        let sigma = (p * (1.0 - p) / N as f64).sqrt();
        assert!(
            (got - p).abs() < 4.0 * sigma,
            "exp({a}) = {p}, observed {got}"
        );
    }
}

/// `exp(-inf) = 0` and `sample < 0` is false for every draw in `[0, 1)`, so an
/// infinitely uphill move is rejected at any finite `beta` -- including
/// `beta = 0`, where `a = mP`.
#[test]
fn an_infinitely_unfavourable_move_is_rejected_at_finite_beta() {
    let mut lo = FixedRng::low();
    assert!(!accept(score(f64::INFINITY, 0.0), 1.0, &mut lo));
    let mut lo = FixedRng::low();
    assert!(!accept(score(0.0, f64::NEG_INFINITY), 0.0, &mut lo));
    // ... and an infinitely favourable one is accepted without a draw.
    let mut hi = FixedRng::high();
    assert!(accept(score(f64::NEG_INFINITY, 0.0), 1.0, &mut hi));
    assert_eq!(hi.draws, 0);
}

// ===========================================================================
// 2. The three schedules (`mcmc_loop.hh:121-122`, `:136-139`, `:196-197`)
// ===========================================================================

/// `Schedule::Shuffled` is C++'s `is_sequential() && !is_deterministic()`:
/// every node exactly once per sweep, in a fresh permutation each sweep.
#[test]
fn shuffled_visits_every_node_exactly_once_per_sweep() {
    const N: usize = 12;
    const SWEEPS: usize = 6;
    let mut s = Toy::new(N).schedule(Schedule::Shuffled).n_iter(SWEEPS);
    mcmc_sweep(&mut s, &mut rng(7));

    assert_eq!(s.visits.len(), N * SWEEPS);
    let all: Vec<usize> = (0..N).collect();
    for i in 0..SWEEPS {
        let sweep = s.sweep(i, N);
        let mut sorted = sweep.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, all, "sweep {i} is not a permutation of the nodes");
    }

    // A permutation each sweep, and not the same one twice in a row.
    let distinct = (1..SWEEPS)
        .filter(|&i| s.sweep(i, N) != s.sweep(i - 1, N))
        .count();
    assert_eq!(distinct, SWEEPS - 1, "the list is reshuffled every sweep");
}

/// `:121-122` shuffles **before** `:134` calls `init_iter`. `init_iter` is
/// where a real state resizes per-sweep scratch, and half of them then read
/// `get_vlist()`; a shuffle after it would hand them the previous order.
#[test]
fn the_shuffle_precedes_init_iter() {
    const N: usize = 9;
    let mut s = Toy::new(N).schedule(Schedule::Shuffled).n_iter(4);
    mcmc_sweep(&mut s, &mut rng(11));

    assert_eq!(s.init_calls, 4, "mcmc_loop.hh:134 runs once per sweep");
    for i in 0..4 {
        assert_eq!(
            s.order_at_init[i],
            s.sweep(i, N),
            "init_iter must see the order the sweep is about to use"
        );
    }
}

/// `Schedule::Alternating` is C++'s `is_sequential() && is_deterministic()`:
/// the list is walked in order and then *reversed* (`:196-197`), so the node
/// a sweep ends on is the node the next sweep starts from.
#[test]
fn alternating_walks_the_list_forwards_then_backwards() {
    const N: usize = 5;
    let mut s = Toy::new(N).schedule(Schedule::Alternating).n_iter(4);
    mcmc_sweep(&mut s, &mut rng(3));

    assert_eq!(s.sweep(0, N), [0, 1, 2, 3, 4]);
    assert_eq!(s.sweep(1, N), [4, 3, 2, 1, 0]);
    assert_eq!(s.sweep(2, N), [0, 1, 2, 3, 4]);
    assert_eq!(s.sweep(3, N), [4, 3, 2, 1, 0]);
    // The reversal is applied after the last sweep too, exactly as at `:196`.
    assert_eq!(s.nodes, [0, 1, 2, 3, 4]);
}

/// The acceptance criterion's reproducibility clause, and more: `Alternating`
/// draws nothing at all for its ordering, so it is reproducible across
/// *different* seeds as well as equal ones.
#[test]
fn alternating_is_reproducible_and_seed_independent() {
    let run = |seed: u64| {
        let mut s = Toy::new(7).schedule(Schedule::Alternating).n_iter(5);
        let out = mcmc_sweep(&mut s, &mut rng(seed));
        (s.visits, out)
    };
    let (a, ra) = run(1);
    let (b, rb) = run(1);
    let (c, rc) = run(0xDEAD_BEEF);
    assert_eq!(a, b, "the same seed must reproduce the sweep");
    assert_eq!(ra, rb);
    assert_eq!(a, c, "Alternating consumes no randomness for its ordering");
    assert_eq!(ra, rc);
}

/// `Schedule::Random` is C++'s `!is_sequential()`: `uniform_sample(vlist, rng)`
/// at `:139`, drawn `steps_per_iter()` times -- `get_N()` at `:125-132`. With
/// replacement, so the multiset is *not* the node set.
#[test]
fn random_draws_steps_per_iter_nodes_with_replacement() {
    const N: usize = 6;
    const STEPS: usize = 500;
    let mut s = Toy::new(N)
        .schedule(Schedule::Random)
        .steps(STEPS)
        .n_iter(3);
    mcmc_sweep(&mut s, &mut rng(19));

    assert_eq!(
        s.visits.len(),
        STEPS * 3,
        "get_N() bounds the step count, not the node count"
    );
    let counts = multiset(&s.visits);
    assert_eq!(counts.len(), N, "every node is reachable");
    for (&v, &c) in &counts {
        assert!(v < N, "a node outside the list was visited");
        // 1500 draws over 6 nodes: mean 250, sd ~15. A 100-wide band catches
        // an off-by-one range or a non-uniform index without being flaky.
        assert!((c as i64 - 250).abs() < 100, "node {v} drawn {c} times");
    }
    // With replacement: 1500 draws over 6 nodes cannot be a permutation.
    assert!(counts.values().any(|&c| c > 1));
}

/// `Random` takes its length from `steps_per_iter()` and *not* from the node
/// list: a one-node list driven for `N` steps is how `merge_split.hh:1464-1472`
/// runs, with `_vlist = {Node()}` and `get_N() = round(_N * min(niter, 1))`.
#[test]
fn random_ignores_the_node_list_length() {
    let mut s = Toy::new(1).schedule(Schedule::Random).steps(37);
    mcmc_sweep(&mut s, &mut rng(5));
    assert_eq!(s.visits, vec![0; 37]);
}

/// A sequential schedule takes its length from the node list and ignores
/// `steps_per_iter()` -- the other half of `:125-132`.
#[test]
fn sequential_schedules_ignore_steps_per_iter() {
    for sch in [Schedule::Shuffled, Schedule::Alternating] {
        let mut s = Toy::new(4).schedule(sch).steps(1000);
        mcmc_sweep(&mut s, &mut rng(2));
        assert_eq!(s.visits.len(), 4, "{sch:?}");
    }
}

/// All three variants, one table: the multiset a single sweep visits.
#[test]
fn every_schedule_visits_the_multiset_it_names() {
    const N: usize = 8;

    let mut seq = Toy::new(N).schedule(Schedule::Alternating);
    mcmc_sweep(&mut seq, &mut rng(1));
    assert_eq!(multiset(&seq.visits), multiset(&(0..N).collect::<Vec<_>>()));

    let mut sh = Toy::new(N).schedule(Schedule::Shuffled);
    mcmc_sweep(&mut sh, &mut rng(1));
    assert_eq!(multiset(&sh.visits), multiset(&(0..N).collect::<Vec<_>>()));

    let mut rnd = Toy::new(N).schedule(Schedule::Random).steps(N);
    mcmc_sweep(&mut rnd, &mut rng(1));
    assert_eq!(rnd.visits.len(), N);
    assert!(rnd.visits.iter().all(|&v| v < N));
    // Distinguishes with-replacement from a permutation: over many sweeps a
    // uniform-with-replacement draw of 8 from 8 is a permutation with
    // probability 8!/8^8 ≈ 0.0024, so 40 sweeps that are all permutations
    // would mean the schedule is not what it says.
    let mut any_repeat = false;
    for seed in 0..40u64 {
        let mut r = Toy::new(N).schedule(Schedule::Random).steps(N);
        mcmc_sweep(&mut r, &mut rng(seed));
        if multiset(&r.visits).len() < N {
            any_repeat = true;
            break;
        }
    }
    assert!(any_repeat, "Random must sample with replacement");
}

// ===========================================================================
// 3. `nattempts` / `nmoves` accumulate `nsteps` (defect #25, `:124`-`:184`)
// ===========================================================================

/// `mcmc_loop.hh:178` and `:184` add `nsteps`, which `:161` sets only inside
/// the `constexpr` non-single-step branch. A `propose` that returned a bare
/// move could not express `merge_split.hh:1409`'s `{_null_move, _nmoves}` at
/// all, and a loop that added 1 would report one attempt for a batch of
/// `_nmoves` node moves.
#[test]
fn attempts_and_moves_accumulate_nsteps_not_one() {
    const N: usize = 4;
    let mut s = Toy::new(N)
        .schedule(Schedule::Alternating)
        .script(vec![Step { mv: 1, nsteps: 7 }]);
    let out = mcmc_sweep(&mut s, &mut rng(1));

    assert_eq!(
        out.nattempts,
        N * 7,
        "nattempts += nsteps (mcmc_loop.hh:178)"
    );
    assert_eq!(out.nmoves, N * 7, "nmoves += nsteps (mcmc_loop.hh:184)");
    assert_eq!(s.performed.len(), N, "one perform_move per proposal");
}

/// The two counters are independent: a rejected batch is attempted and not
/// moved. `beta = ∞` with `dS > 0` rejects without drawing (`:82-85`).
#[test]
fn a_rejected_batch_counts_attempts_but_no_moves() {
    const N: usize = 4;
    let mut s = Toy::new(N)
        .schedule(Schedule::Alternating)
        .uphill()
        .script(vec![Step { mv: 1, nsteps: 5 }]);
    let out = mcmc_sweep(&mut s, &mut rng(1));

    assert_eq!(out.nattempts, N * 5);
    assert_eq!(out.nmoves, 0);
    assert_eq!(out.d_entropy, 0.0, "S += dS only on the accepted branch");
    assert!(s.performed.is_empty());
    assert_eq!(
        s.after.len(),
        N,
        "step() runs for rejected moves too (:189)"
    );
}

/// Varying `nsteps` per proposal, so a loop that used the *last* `nsteps` for
/// every step -- which is what the C++ variable's placement at `:124` invites,
/// since it is declared outside the node loop and assigned inside a lambda --
/// gives a different total.
#[test]
fn nsteps_is_read_per_proposal() {
    let mut s = Toy::new(4).schedule(Schedule::Alternating).script(vec![
        Step { mv: 1, nsteps: 1 },
        Step { mv: 1, nsteps: 2 },
        Step { mv: 1, nsteps: 4 },
        Step { mv: 1, nsteps: 8 },
    ]);
    let out = mcmc_sweep(&mut s, &mut rng(1));
    assert_eq!(out.nattempts, 15);
    assert_eq!(out.nmoves, 15);
}

/// `nsteps = 0` is a proposal that stands for nothing; it must move the
/// counters by nothing while still being proposed, priced and performed.
#[test]
fn a_zero_step_proposal_moves_no_counter() {
    let mut s = Toy::new(3)
        .schedule(Schedule::Alternating)
        .script(vec![Step { mv: 1, nsteps: 0 }]);
    let out = mcmc_sweep(&mut s, &mut rng(1));
    assert_eq!((out.nattempts, out.nmoves), (0, 0));
    assert_eq!(s.performed.len(), 3);
    assert_eq!(out.d_entropy, -3.0);
}

// ===========================================================================
// 4. The null move and `skip_node` (`:141-142`, `:168-173`)
// ===========================================================================

/// `:168-173`. A null proposal is not an attempt: `nattempts += nsteps` is at
/// `:178`, *after* the `continue`. It is also not priced and not stepped.
#[test]
fn a_null_proposal_is_not_an_attempt() {
    let mut s = Toy::new(4).schedule(Schedule::Alternating).script(vec![
        Step { mv: 1, nsteps: 3 },
        Step {
            mv: NULL,
            nsteps: 99,
        },
    ]);
    let out = mcmc_sweep(&mut s, &mut rng(1));

    assert_eq!(s.visits, [0, 1, 2, 3], "the node is still visited");
    assert_eq!(out.nattempts, 6, "two real proposals of three steps each");
    assert_eq!(out.nmoves, 6);
    assert_eq!(s.scored, [(0, 1), (2, 1)], "a null move is never priced");
    assert_eq!(
        s.after,
        [(0, 1), (2, 1)],
        "nor stepped (:189 is unreachable)"
    );
}

/// `base/mcmc.hh:104-107`: `if (s == node_state(v)) return _null_move;`. Every
/// concrete C++ state ends its `move_proposal` with this line and
/// `mcmc.hh:112-114` then guards the *same* condition a second time inside
/// `virtual_move`. Hoisting it into the loop is why `node_state` is a required
/// member here rather than the verbose-print accessor it is at
/// `mcmc_loop.hh:146`.
#[test]
fn a_move_to_the_current_state_is_a_null_move() {
    let mut s = Toy::new(4).schedule(Schedule::Alternating);
    s.group = vec![0, 1, 0, 1]; // nodes 1 and 3 are already in group 1
    let out = mcmc_sweep(&mut s, &mut rng(1));

    assert_eq!(s.visits, [0, 1, 2, 3]);
    assert_eq!(s.scored, [(0, 1), (2, 1)]);
    assert_eq!(out.nattempts, 2);
    assert_eq!(out.nmoves, 2);
    assert_eq!(s.after, [(0, 1), (2, 1)]);
}

/// `:141-142`. A skipped node never reaches `move_proposal`, so it consumes no
/// randomness either -- `mcmc.hh:95-97` skips zero-weight vertices, and a
/// state that proposed for them would desynchronise the stream.
#[test]
fn a_skipped_node_is_never_proposed_for() {
    let mut s = Toy::new(5).schedule(Schedule::Alternating);
    s.skip = vec![1, 3];
    let out = mcmc_sweep(&mut s, &mut rng(1));

    assert_eq!(s.visits, [0, 2, 4]);
    assert_eq!(out.nattempts, 3);
    assert!(s.after.iter().all(|&(v, _)| v != 1 && v != 3));
    // ... and skipping does not disturb the walk: `:196` still reverses the
    // whole list, skipped entries included.
    assert_eq!(s.nodes, [4, 3, 2, 1, 0]);
}

// ===========================================================================
// 5. The accumulated entropy (`:134`, `:185`)
// ===========================================================================

/// `S` is `init_iter`'s offset once per sweep (`:134`) plus `dS` for each
/// *accepted* move (`:185`), and nothing else.
#[test]
fn the_entropy_is_init_iter_plus_the_accepted_deltas() {
    const N: usize = 6;
    const SWEEPS: usize = 3;
    let mut s = Toy::new(N).schedule(Schedule::Alternating).n_iter(SWEEPS);
    s.init_offset = 0.25;
    s.d_entropy = -0.5;
    let out = mcmc_sweep(&mut s, &mut rng(1));

    assert_eq!(out.nmoves, N * SWEEPS);
    let expect = SWEEPS as f64 * 0.25 + (N * SWEEPS) as f64 * -0.5;
    assert!((out.d_entropy - expect).abs() < 1e-12, "{:?}", out);

    // Rejected: the offsets survive, the deltas do not.
    let mut s = Toy::new(N)
        .schedule(Schedule::Alternating)
        .n_iter(SWEEPS)
        .uphill();
    s.init_offset = 0.25;
    let out = mcmc_sweep(&mut s, &mut rng(1));
    assert!((out.d_entropy - 0.75).abs() < 1e-12, "{:?}", out);
}

/// `init_iter` runs even when the sweep visits nothing -- `:134` is outside the
/// node loop.
#[test]
fn init_iter_runs_on_an_empty_node_list() {
    let mut s = Toy::new(0).schedule(Schedule::Alternating).n_iter(3);
    s.init_offset = 2.0;
    let out = mcmc_sweep(&mut s, &mut rng(1));
    assert_eq!(s.init_calls, 3);
    assert_eq!(out.d_entropy, 6.0);
    assert_eq!(out.nattempts, 0);
}

// ===========================================================================
// 6. Degenerate shapes
// ===========================================================================

/// `uniform_sample(vlist, rng)` on an empty list is undefined behaviour in
/// C++ (`samplers.hh` indexes without checking). Here it is a sweep that
/// visits nothing.
#[test]
fn an_empty_node_list_is_not_a_panic() {
    for sch in [Schedule::Random, Schedule::Shuffled, Schedule::Alternating] {
        let mut s = Toy::new(0).schedule(sch).steps(25).n_iter(2);
        let out = mcmc_sweep(&mut s, &mut rng(1));
        assert_eq!(out, SweepResult::default(), "{sch:?}");
        assert!(s.visits.is_empty());
    }
}

/// `for (iter = 0; iter < get_niter(); ++iter)` with `get_niter() == 0` is a
/// loop that does not run -- including `init_iter`.
#[test]
fn zero_iterations_does_nothing_at_all() {
    let mut s = Toy::new(5).n_iter(0);
    s.init_offset = 9.0;
    let out = mcmc_sweep(&mut s, &mut rng(1));
    assert_eq!(out, SweepResult::default());
    assert_eq!(s.init_calls, 0);
    assert!(s.visits.is_empty());
    assert_eq!(s.nodes, [0, 1, 2, 3, 4], "no reversal either");
}

/// `Random` with `steps_per_iter() == 0` draws nothing, and must not draw the
/// node index "just in case" -- the generator state is observable through the
/// next caller.
#[test]
fn random_with_zero_steps_draws_nothing() {
    let mut s = Toy::new(5).schedule(Schedule::Random).steps(0).n_iter(4);
    let mut r = rng(1);
    let before = r.next_u64();
    let mut r = rng(1);
    let out = mcmc_sweep(&mut s, &mut r);
    assert_eq!(out, SweepResult::default());
    assert_eq!(r.next_u64(), before, "the stream was not advanced");
}

/// One node, one sweep, one move: the smallest complete trace, asserted in
/// full so a reordering of the five calls at `:150`-`:189` is visible.
#[test]
fn the_call_order_is_propose_score_perform_step() {
    let mut s = Toy::new(1).schedule(Schedule::Alternating);
    let out = mcmc_sweep(&mut s, &mut rng(1));
    assert_eq!(s.init_calls, 1);
    assert_eq!(s.visits, [0]);
    assert_eq!(s.scored, [(0, 1)]);
    assert_eq!(s.performed, [(0, 1)]);
    assert_eq!(s.after, [(0, 1)]);
    assert_eq!(
        out,
        SweepResult {
            d_entropy: -1.0,
            nattempts: 1,
            nmoves: 1
        }
    );
}

// ===========================================================================
// 7. The negative guarantees (DESIGN.md §14, defect #19)
// ===========================================================================

/// Defect #19 is the reason `GroupProposal` is shaped as it is: the sampler
/// returns its own `log_fwd`, and the reverse density is a required trait item
/// with no default. The two fixtures pin the two ways `potts/spec.hh:124`'s
/// mistake can be written.
///
/// | fixture | diagnostic |
/// |---|---|
/// | `u21_get_move_lprob_is_not_a_trait_item` | `E0407` |
/// | `u21_inherent_lprob_leaves_log_reverse_missing` | `E0046` |
#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/u21_*.rs");
}
