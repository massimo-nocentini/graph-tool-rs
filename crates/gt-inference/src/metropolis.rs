//! The Metropolis loop.
//!
//! Every member `MetropolisStateBase` (`loops/mcmc_loop.hh:36-75`) labels
//! `// required` and then gives a *working* body is genuinely required here:
//! `get_vlist()` returns an empty array (`:39`), `node_state()` returns 0
//! (`:43`), `move_proposal()` returns 0 (`:46`), `virtual_move()` returns
//! `{0,0}` (`:52-53`), `perform_move()` is a no-op (`:56`). Nothing is
//! `= delete`d, so a state that misspells `virtual_move` compiles and reports
//! `dS = 0, mP = 0` for every move -- and `metropolis_accept` (`:78-96`) then
//! accepts unconditionally.

use rand::Rng;
use rand::seq::SliceRandom;

/// The two numbers a move is judged by.
///
/// `base/mcmc.hh:112-128` returns `std::tuple<double, double>` and
/// `mcmc_loop.hh:174` destructures it **positionally**. Transposing them now
/// requires writing the wrong field name.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MoveScore {
    /// Change in the entropy functional.
    pub d_entropy: f64,
    /// Log of the Hastings ratio.
    pub log_hastings: f64,
}

/// A proposal, with how many elementary steps it represents.
///
/// `mcmc_loop.hh:112-118` detects the multi-step case with a `constexpr`
/// probe on `move_proposal`'s *return type*: `merge_split.hh:1201-1203`
/// returns `std::tuple<size_t, size_t>` and the loop does
/// `nattempts += nsteps`. A design whose `propose` returns a bare move cannot
/// express merge-split, multiflip or multilevel at all. Single-step states
/// write `Step { mv, nsteps: 1 }`, and nothing is read positionally.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Step<M> {
    /// The proposed move.
    pub mv: M,
    /// How many elementary steps it stands for.
    pub nsteps: usize,
}

/// How a sweep orders its nodes.
///
/// Collapses `is_sequential()` x `is_deterministic()` (`mcmc_loop.hh:121`,
/// `:196`) into one required method with three legal values. This deletes the
/// misspelled `is_determinisitic` (`:60`) *and* the fourth, unrepresentable
/// C++ combination -- `!sequential && deterministic`, which silently means
/// "random order" because `:196` is guarded by `is_sequential()`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Schedule {
    /// Sample nodes independently with replacement.
    Random,
    /// Visit every node once, in a fresh random order each sweep.
    Shuffled,
    /// Visit every node once, in a fixed order.
    Alternating,
}

/// What a sweep did.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct SweepResult {
    /// Total entropy change.
    pub d_entropy: f64,
    /// Elementary steps attempted.
    pub nattempts: usize,
    /// Elementary steps accepted.
    pub nmoves: usize,
}

/// A state the Metropolis loop can drive.
pub trait MetropolisState: Send {
    /// What a sweep iterates over.
    type Node: Copy + Send + Sync;
    /// What a move is.
    type Move: Copy + PartialEq + Send + Sync;
    /// The value meaning "no move".
    const NULL_MOVE: Self::Move;

    /// The node list, owned by the state so a sweep can shuffle it in place.
    fn nodes(&mut self) -> &mut Vec<Self::Node>;
    /// Sweeps to run.
    fn n_iter(&self) -> usize;
    /// Inverse temperature.
    fn beta(&self) -> f64;
    /// Node ordering.
    fn schedule(&self) -> Schedule;
    /// Elementary steps per sweep, for the [`Schedule::Random`] case.
    /// `merge_split.hh:1470`'s `get_N()`.
    fn steps_per_iter(&mut self) -> usize;
    /// The node's current state, for the null-move test.
    fn node_state(&self, v: Self::Node) -> Self::Move;
    /// Propose.
    fn propose<R: Rng + ?Sized>(&mut self, v: Self::Node, rng: &mut R) -> Step<Self::Move>;
    /// Price.
    fn score(&mut self, v: Self::Node, s: Self::Move) -> MoveScore;
    /// Apply.
    fn perform(&mut self, v: Self::Node, s: Self::Move);

    // -- legitimate defaults: pure no-ops, no policy ------------------------

    /// Skip a node entirely.
    #[inline]
    fn skip_node(&self, v: Self::Node) -> bool {
        false
    }
    /// Per-sweep setup; returns an entropy offset.
    #[inline]
    fn init_iter<R: Rng + ?Sized>(&mut self, rng: &mut R) -> f64 {
        0.0
    }
    /// Per-step bookkeeping.
    #[inline]
    fn after_step(&mut self, v: Self::Node, s: Self::Move) {}
}

/// The Metropolis acceptance test. `mcmc_loop.hh:78-96`.
///
/// Transcribed term for term:
///
/// ```text
/// if (std::isinf(beta))          ->  beta.is_infinite()
///     return dS < 0;             ->  score.d_entropy < 0.0
/// double a = -dS * beta + mP;    ->  -score.d_entropy * beta + score.log_hastings
/// if (a > 0) return true;        ->  a > 0.0
/// std::uniform_real_distribution<> sample;   // [0, 1)
/// return sample(rng) < exp(a);   ->  rng.random::<f64>() < a.exp()
/// ```
///
/// `rand`'s `StandardUniform` for `f64` is `[0, 1)` (`distr/float.rs`,
/// "Multiply-based method; 53 random bits; [0, 1) interval"), which is
/// `std::uniform_real_distribution<double>`'s default range, so the zero
/// measure endpoint convention agrees as well.
///
/// The generator is touched **only** on the `a <= 0` path: an infinite `beta`
/// and an uphill-free move both return without drawing. A sweep's stream is
/// therefore a function of which moves were proposed, exactly as in C++, and
/// two implementations that disagree about when to draw desynchronise
/// visibly rather than subtly.
///
/// `beta.is_infinite()` -- not `beta == f64::INFINITY` -- is `std::isinf`,
/// which is also true at `-inf`. That is faithful and is *not* on the defect
/// list: graph-tool's `beta` is an inverse temperature and the Python layer
/// never passes a negative one, so the branch has no reachable disagreement
/// to fix. Naming a different rule here would be inventing semantics.
#[inline]
pub fn accept<R: Rng + ?Sized>(score: MoveScore, beta: f64, rng: &mut R) -> bool {
    if beta.is_infinite() {
        // `mcmc_loop.hh:82-85`: the zero-temperature limit is a downhill test,
        // and the Hastings ratio does not enter it.
        return score.d_entropy < 0.0;
    }

    let a = -score.d_entropy * beta + score.log_hastings;
    if a > 0.0 {
        return true;
    }
    rng.random::<f64>() < a.exp()
}

/// Run `state.n_iter()` sweeps.
///
/// No `Python` token is in scope inside this function, so
/// `mcmc_loop.hh:101`'s unconditional `GILRelease gil;` has no analogue to get
/// wrong: the GIL is released once, outside, by the PyO3 boundary, and the
/// `F: Send` bound on `allow_threads` means nothing holding a token can be
/// captured into the region.
pub fn mcmc_sweep<S, R>(state: &mut S, rng: &mut R) -> SweepResult
where
    S: MetropolisState,
    R: Rng + ?Sized,
{
    let mut out = SweepResult::default();

    // `:108`. Read once, as C++ does; `beta` is not a per-sweep quantity.
    let beta = state.beta();
    let n_iter = state.n_iter();

    for _ in 0..n_iter {
        let schedule = state.schedule();

        // `:121-122`. The C++ guard is `is_sequential() && !is_deterministic()`
        // -- and `is_deterministic()` is *not* the member `MetropolisStateBase`
        // declares (`:60` spells it `is_determinisitic`), which is defect #20.
        // One enum, three arms, nothing to misspell.
        if schedule == Schedule::Shuffled {
            state.nodes().shuffle(rng);
        }

        // `:134`. After the shuffle, so the generator is consumed in C++'s
        // order and a fixed seed reproduces C++'s node permutation.
        out.d_entropy += state.init_iter(rng);

        // `:125-132`'s `get_N`, with the `constexpr` return-type probe
        // (defect #25) replaced by the schedule. A sequential sweep visits the
        // node list; a random sweep takes the state's own step count. A
        // multi-step state that C++ drives through `get_N()` over a
        // one-element `_vlist` (`merge_split.hh:1464-1472`) ports as a node
        // list of that many entries, which is why this needs no third arm.
        let n = match schedule {
            Schedule::Random => state.steps_per_iter(),
            Schedule::Shuffled | Schedule::Alternating => state.nodes().len(),
        };

        for vi in 0..n {
            // `:138-139`. `nodes()` is re-borrowed per step, never hoisted:
            // `perform` may push or pop nodes, and C++ re-evaluates
            // `vlist[vi]` and `get_N()` at every step for the same reason.
            let v = {
                let nodes = state.nodes();
                if nodes.is_empty() {
                    break;
                }
                match schedule {
                    // `uniform_sample(vlist, rng)`: with replacement.
                    Schedule::Random => {
                        let k = rng.random_range(0..nodes.len());
                        nodes[k]
                    }
                    Schedule::Shuffled | Schedule::Alternating => {
                        if vi >= nodes.len() {
                            break;
                        }
                        nodes[vi]
                    }
                }
            };

            // `:141-142`.
            if state.skip_node(v) {
                continue;
            }

            let step = state.propose(v, rng);
            let s = step.mv;

            // `:168-173`, plus `base/mcmc.hh:105-106`. The second test is
            // hoisted out of the state: every concrete C++ state ends its
            // `move_proposal` with `if (s == node_state(v)) return _null_move;`
            // and every one that forgot would price a self-move that
            // `mcmc.hh:112-114` then has to special-case a second time. One
            // place, and it is the reason `node_state` is a required member
            // here instead of the verbose-print accessor it is at `:146`.
            if s == S::NULL_MOVE || s == state.node_state(v) {
                continue;
            }

            // `:176`. Named fields, so `d_entropy` and `log_hastings` cannot
            // arrive transposed the way `std::tie(dS, mP)` can (defect #23).
            let score = state.score(v, s);

            // `:178`. `nsteps`, not 1 -- defect #25. In C++ this is a variable
            // declared at `:124` and assigned only inside a `constexpr`
            // branch of a lambda, so a state whose `move_proposal` returns a
            // bare move silently reports one attempt for a batch of moves.
            out.nattempts += step.nsteps;

            // `:181-187`. `_force_accept` has no analogue: it is a compile
            // time member of the state in C++ and belongs, if anywhere, to the
            // state's own `score`.
            if accept(score, beta, rng) {
                state.perform(v, s);
                out.nmoves += step.nsteps;
                out.d_entropy += score.d_entropy;
            }

            // `:189`. Runs for accepted and rejected moves alike, but not for
            // skipped or null ones.
            state.after_step(v, s);
        }

        // `:196-197`. The reversal is what makes this schedule *alternating*:
        // sweep 0 runs the list forwards, sweep 1 backwards, and a vertex at
        // the end of one sweep is at the start of the next.
        if schedule == Schedule::Alternating {
            state.nodes().reverse();
        }
    }

    out
}
