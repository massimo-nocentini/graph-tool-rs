//! The state/loop layering.
//!
//! ## The rule that makes Potts-style bugs impossible ([DESIGN](gt_core::design) D8)
//!
//! graph-tool's layering is *mixin-from-below*: `SpecBase::Imp`
//! (`base/spec_base.hh:58-130`) `= delete`s exactly three members --
//! `move_vertex` (`:69`), `virtual_move` (`:70`), `entropy` (`:71`) -- and
//! **defaults fourteen others with working bodies**. A misnamed deleted
//! member is a compile error; a misnamed defaulted one silently inherits the
//! base's behaviour.
//!
//! That is a live bug, verified: `potts/spec.hh:113-117` overrides the
//! proposal with `uniform_int_distribution<group_t> sample(0, _q-1)`, then
//! `:124` defines `get_move_lprob`. `grep -rn get_move_lprob src/` finds
//! exactly that one line, called from nowhere; the loop calls
//! `get_move_prob` (`base/mcmc.hh:123-124`), which Potts does not define, so
//! it inherits `SpecBase::Imp::get_move_prob` (`spec_base.hh:88-109`) --
//! `log(1. - d) - safelog_fast(B)`, the density of the *base's* sampler.
//! The Hastings ratio does not correspond to the kernel being sampled.
//!
//! There is a **second instance one layer up**, not previously reported:
//! `loops/mcmc_loop.hh:60` declares `constexpr bool is_determinisitic()` --
//! misspelled -- while the loop calls `is_deterministic()` at `:121` and
//! `:196`. The default is dead; every concrete state happens to define the
//! correct spelling, so it survives only because nothing relies on it.
//!
//! The rule this module applies:
//!
//! > **A trait method may carry a default body only if that body is either a
//! > pure no-op or derivable from other *required* methods of the same
//! > trait.** A default that encodes policy is not a default; it is a required
//! > method.
//!
//! And, stronger than the rule: [`GroupProposal::propose`] **returns its own
//! forward density**, so there is no second place for the density to live.
//! Requiring `log_move_prob` merely makes the misnaming a compile error;
//! returning `log_fwd` from the sampler means a copy-pasted density in the
//! same impl block cannot silently disagree with a hand-written sampler.

use gt_core::dir::Dir;
use gt_core::ids::VertexId;
use rand::Rng;

use crate::ids::{Group, Weight};

/// Parameters of the entropy functional (degree correction, priors, ...).
pub trait EntropyArgs: Copy + Send + Sync + 'static {}

/// A proposed move, carrying the density of the kernel that produced it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Proposal {
    /// The group proposed.
    pub to: Group,
    /// `log q(r -> to | v)` under the kernel that sampled it.
    pub log_fwd: f64,
}

/// The three members graph-tool `= delete`s, plus the scratch they need.
///
/// `Scratch` is the fix for a signature that cannot be implemented as written.
/// `virtual_move(&self, ...)` looks right, but `blockmodel/state.hh:304` is
/// `get_move_entries(v, r, nr, _m_entries); move_vertex(v, r, nr, _m_entries);`
/// -- virtual_move **mutates** a reusable scratch `EntrySet`, and every one of
/// those C++ members is non-const. `&self` plus the `Sync` supertrait leaves
/// only three options: allocate per Metropolis step (on the hottest path in
/// the library), use a `thread_local` (reintroducing exactly the thread-keyed
/// hidden global that `parallel_rng.hh:56-73` is condemned for, and
/// destroying reproducibility), or take `&mut self` (deleting the lock-free
/// proposal phase). Threading `&mut Self::Scratch` is the fourth, and it makes
/// `_m_entries_pool.resize(get_num_threads())` (`:451`) an explicit per-chunk
/// value instead of a thread-count-indexed global.
pub trait SpecCore: Sync {
    /// Entropy parameters.
    type Args: EntropyArgs;
    /// Per-caller scratch. [`Workspace`](crate::delta::Workspace) for the
    /// block model.
    type Scratch: Default + Send;
    /// Edge/vertex weight type.
    type W: Weight;
    /// Directedness.
    type D: Dir;

    /// Apply a move. `spec_base.hh:69`.
    fn move_vertex(&mut self, v: VertexId, r: Group, nr: Group, sc: &mut Self::Scratch);

    /// Price a move without applying it. `spec_base.hh:70`.
    fn virtual_move(
        &self,
        v: VertexId,
        r: Group,
        nr: Group,
        ea: &Self::Args,
        sc: &mut Self::Scratch,
    ) -> f64;

    /// The absolute entropy. `spec_base.hh:71`.
    fn entropy(&self, ea: &Self::Args) -> f64;
}

/// The proposal kernel **and** its density, in one impl block.
///
/// Bundling them is the substantive fix. In C++ the sampler and the density
/// can come from *different classes* in the inheritance chain; in Rust trait
/// items resolve per-`impl`, so one type supplies both and a disagreement is a
/// local, visible inconsistency rather than an inheritance accident.
///
/// The generator is a **type parameter**, not a fixed concrete type: a
/// concrete `Rng` baked into the trait would make swapping the generator, or
/// testing with a deterministic stub, an edit to every signature.
pub trait GroupProposal: SpecCore {
    /// Sample a move, returning its own forward log-density.
    fn propose<R: Rng + ?Sized>(
        &self,
        v: VertexId,
        r: Group,
        c: f64,
        d: f64,
        rng: &mut R,
    ) -> Option<Proposal>;

    /// `log q(s -> r | v)` for the reverse move, which is never sampled.
    fn log_reverse(&self, v: VertexId, r: Group, s: Group, c: f64, d: f64) -> f64;

    /// The Hastings ratio. Derived, so the loop never hand-swaps the
    /// endpoints.
    ///
    /// `base/mcmc.hh:123-124` is
    /// `pf = get_move_prob(v, r, nr, c, d, false);`
    /// `pb = get_move_prob(v, nr, r, c, d, true);` -- the caller must exchange
    /// `r` and `nr` **and** flip the flag, in agreement, on two adjacent
    /// lines. Here there is nothing to exchange.
    #[inline]
    fn log_hastings(&self, v: VertexId, r: Group, p: &Proposal, c: f64, d: f64) -> f64 {
        self.log_reverse(v, r, p.to, c, d) - p.log_fwd
    }

    /// A local proposal. Derived from [`propose`](Self::propose), so the
    /// default is legitimate (`spec_base.hh:83-86`).
    #[inline]
    fn propose_local<R: Rng + ?Sized>(
        &self,
        v: VertexId,
        r: Group,
        rng: &mut R,
    ) -> Option<Proposal> {
        self.propose(v, r, 0.0, 0.0, rng)
    }
}

/// Pure side effects with legitimate no-op defaults (`spec_base.hh:112-124`).
pub trait GroupHooks: SpecCore {
    /// A group became non-empty.
    #[inline]
    fn occupy_group(&mut self, r: Group) {}
    /// A group became empty.
    #[inline]
    fn vacate_group(&mut self, r: Group) {}
    /// Post-construction fixup.
    #[inline]
    fn post_init(&mut self) {}
}

/// Group merging. Capability as **trait presence**.
///
/// Replaces the duplicated pair `has_merge` on the spec (`spec_base.hh:50`)
/// and `_has_merge` on `Imp` (`:115`) and again on the loop states
/// (`merge_split.hh:99`, `multilevel.hh:114`) -- two independent booleans for
/// one capability, at two scopes, which a derived spec can set
/// inconsistently. `if constexpr (_has_merge)` becomes a where-clause, and
/// there is no second flag to disagree with the first.
pub trait Mergeable: SpecCore {
    /// Merge `s` into `t`.
    fn merge_groups(&mut self, s: Group, t: Group, sc: &mut Self::Scratch);
    /// Price a merge.
    fn virtual_merge(&self, s: Group, t: Group, ea: &Self::Args, sc: &mut Self::Scratch) -> f64;
}

/// Batched moves. Replaces `has_parallel_move` / `_has_parallel_move`
/// (`spec_base.hh:51`, `:119`).
pub trait ParallelMove: SpecCore + Send {
    /// Apply a batch, returning the total entropy change.
    fn move_vertices_batch(&mut self, batch: &[(VertexId, Group, Group)]) -> f64;
}

/// The explicit undo stack of `spec_base.hh:124-129`.
pub trait Checkpointed: SpecCore {
    /// Enable or disable relaxed updates.
    fn relax_update(&mut self, relax: bool);
    /// Snapshot the given vertices.
    fn push_state(&mut self, vs: &[VertexId]);
    /// Pop, optionally discarding the snapshot instead of restoring it.
    fn pop_state(&mut self, discard: bool);
    /// Exchange the top two snapshots.
    fn swap_state(&mut self);
    /// Drop every snapshot.
    fn clear_state(&mut self);
}
