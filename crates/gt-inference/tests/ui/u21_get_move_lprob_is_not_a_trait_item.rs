//! Defect #19 (`potts/spec.hh:113-124`), half one: the *misnamed* density.
//!
//! Potts overrides the proposal with a uniform sampler over `0..q` and then
//! writes its density under the name `get_move_lprob`. The loop calls
//! `get_move_prob` (`base/mcmc.hh:123-124`), which Potts does not define, so
//! C++ silently inherits `SpecBase::Imp::get_move_prob`
//! (`spec_base.hh:88-109`) -- `log(1. - d) - safelog_fast(B)`, the density of
//! a *different* sampler. `grep -rn get_move_lprob src/` finds the definition
//! and no call.
//!
//! In an `impl` block there is no base to fall back to, so the misnamed member
//! is not even a member:
//!
//! ```text
//! error[E0407]: method `get_move_lprob` is not a member of trait `GroupProposal`
//! ```

use gt_core::dir::Directed;
use gt_core::ids::VertexId;
use gt_inference::ids::Group;
use gt_inference::spec::{EntropyArgs, GroupProposal, Proposal, SpecCore};
use rand::Rng;

#[derive(Clone, Copy)]
struct Args;
impl EntropyArgs for Args {}

struct Potts {
    q: u32,
}

impl SpecCore for Potts {
    type Args = Args;
    type Scratch = ();
    type W = i64;
    type D = Directed;

    fn move_vertex(&mut self, _v: VertexId, _r: Group, _nr: Group, _sc: &mut ()) {}

    fn virtual_move(&self, _v: VertexId, _r: Group, _nr: Group, _ea: &Args, _sc: &mut ()) -> f64 {
        0.0
    }

    fn entropy(&self, _ea: &Args) -> f64 {
        0.0
    }
}

impl GroupProposal for Potts {
    fn propose<R: Rng + ?Sized>(
        &self,
        _v: VertexId,
        _r: Group,
        _c: f64,
        _d: f64,
        rng: &mut R,
    ) -> Option<Proposal> {
        let to = Group::new(rng.random_range(0..self.q))?;
        Some(Proposal {
            to,
            log_fwd: -f64::from(self.q).ln(),
        })
    }

    /// `potts/spec.hh:124`, transliterated under its C++ name.
    fn get_move_lprob(&self, _v: VertexId, _r: Group, _s: Group, _c: f64, d: f64) -> f64 {
        (1.0 - d).ln() - f64::from(self.q).ln()
    }

    fn log_reverse(&self, _v: VertexId, _r: Group, _s: Group, _c: f64, _d: f64) -> f64 {
        -f64::from(self.q).ln()
    }
}

fn main() {}
