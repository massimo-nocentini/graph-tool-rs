//! Defect #19, half two: the density moved somewhere the trait cannot see it.
//!
//! The first fixture shows that `get_move_lprob` is not a trait item. The
//! obvious "fix" is to demote it to an inherent method -- which is what
//! `potts/spec.hh:124` effectively is, since nothing dispatches to it. The
//! trait's own reverse density is then missing, and that is the error C++
//! cannot produce: `SpecBase::Imp` supplies a working body for every member it
//! does not `= delete` (`spec_base.hh:58-130`), so the inherited default takes
//! over in silence and the Hastings ratio stops describing the sampler.
//!
//! ```text
//! error[E0046]: not all trait items implemented, missing: `log_reverse`
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

impl Potts {
    /// `potts/spec.hh:124`, now an inherent method: it compiles, and, exactly
    /// as in C++, nothing calls it.
    fn get_move_lprob(&self, _v: VertexId, _r: Group, _s: Group, _c: f64, d: f64) -> f64 {
        (1.0 - d).ln() - f64::from(self.q).ln()
    }
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
}

fn main() {
    let p = Potts { q: 3 };
    let _ = p.get_move_lprob(
        VertexId::from_index(0),
        Group::new(0).unwrap(),
        Group::new(1).unwrap(),
        0.0,
        0.0,
    );
}
