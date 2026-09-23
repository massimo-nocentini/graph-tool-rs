//! U20, the *positive* guarantee: a priced `Delta` is still usable across a
//! `&mut state` call.
//!
//! This is the ownership inversion, and it is the only reason the rest of the
//! design exists. In graph-tool the buffer is `BlockState::_m_entries`
//! (`blockmodel/state.hh:2545`), so the canonical call
//! `this->entries_dS(..., this->_m_entries)` has `&mut self` aliasing
//! `&mut self._m_entries`; transliterating that gives `error[E0499]` and the
//! usual answer is a `RefCell`, which trades the compile error for a runtime
//! panic in the hottest loop in the library.
//!
//! The answer here is that the buffer is the *caller's*
//! ([`Workspace`](gt_inference::delta::Workspace)), so a `Delta` borrows the
//! workspace and not the state, and `&mut st` is disjoint from it by
//! construction. The state need not even be the state the delta was recorded
//! against for the borrow checker to be satisfied -- what keeps the two in
//! step is the `Stamp`, not a lifetime.
//!
//! A `pass` fixture because "this still compiles" regresses exactly as
//! silently as "this no longer fails": the moment somebody gives `Delta` the
//! state's lifetime instead of the workspace's, every negative fixture still
//! passes and this one stops.

use gt_core::dir::Directed;
use gt_inference::delta::{Delta, DeltaStack, MoveHeader, MoveKey, Recording};
use gt_inference::ids::{BEdge, Epoch, Group, Stamp, StateId};

/// Stands in for the block state. Only its mutability matters.
struct Counts {
    mrs: i64,
    epoch: u64,
}

impl Counts {
    /// `&mut self`, taken while a `Delta` is live.
    fn bump(&mut self) {
        self.mrs += 1;
        self.epoch += 1;
    }
}

/// `entries_dS` (`blockmodel/state.hh:1215-1260`), reduced to a contiguous
/// scan over the entries -- the shape the interned before-image buys.
fn price(d: Delta<'_, Directed, i64>) -> i64 {
    let mut s = 0;
    for e in d.entries() {
        s += e.delta * (e.mrs_before + 1);
    }
    s + d.header().dkin + d.header().dkout
}

fn main() {
    let mut stack = DeltaStack::<Directed, i64>::with_levels(1);
    let stamp = Stamp {
        state: StateId::fresh(),
        epoch: Epoch(0),
    };

    let r = Group::new(0).expect("group 0");
    let s = Group::new(1).expect("group 1");

    let hdr = MoveHeader::<i64> {
        r: Some(r),
        nr: Some(s),
        dkin: 2,
        dkout: 3,
        ..MoveHeader::default()
    };

    let mut rec = Recording::new(&mut stack, hdr, stamp);
    rec.level_mut(0).begin(
        MoveKey {
            from: Some(r),
            to: Some(s),
        },
        2,
    );
    let mut resolve = |a: Group, b: Group| -> (Option<BEdge>, i64) {
        (Some(BEdge(0)), (a.index() * 10 + b.index()) as i64)
    };
    rec.level_mut(0)
        .touch_dyn(r, s, 4, &mut resolve)
        .expect("in plane");
    rec.level_mut(0)
        .touch_dyn(s, r, -4, &mut resolve)
        .expect("in plane");
    let t = rec.seal();

    let mut st = Counts { mrs: 0, epoch: 0 };

    // The delta is taken *before* the mutable call.
    let d = t.level(0);
    let before = price(d);

    // ... the state is mutated, exclusively ...
    st.bump();
    st.bump();

    // ... and the delta is still there, unchanged, because it never borrowed
    // the state in the first place. `Delta` is `Copy`, so the first `price`
    // did not consume it either.
    let after = price(d);
    assert_eq!(before, after);
    assert_eq!(before, 4 * (1 + 1) + (-4) * (10 + 1) + 2 + 3);
    assert_eq!(st.mrs, 2);
    assert_eq!(st.epoch, 2);

    // And the transition is still whole: nothing about pricing consumed it.
    assert_eq!(t.n_levels(), 1);
    assert_eq!(t.stamp(), stamp);
}
