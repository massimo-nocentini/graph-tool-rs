//! U20, defect #35: `apply_delta` re-entered double-counts every entry.
//!
//! `apply_delta` (`blockmodel/entries.hh:429-490`) is an ordinary function
//! over an `EntrySet&`. The `EntrySet` is left exactly as it was found -- the
//! entries and their deltas are still there when it returns -- so calling it
//! twice on the same object applies every `update_rs` twice and the block
//! counts silently gain a second copy of the move. Nothing in the C++ marks
//! the set as spent; `empty()` (`:433`) is the only guard and it is false.
//!
//! [`Applied`] is the permission to apply *one level, exactly once*. It is
//! `!Clone` and `!Copy` and every commit takes it by value, so the second
//! application is a use of a moved value.
//!
//! The state here is deliberately not a `BlockCommit`: what is under test is
//! the token, and it must fail for *any* consumer that takes it by value.

use gt_core::dir::Directed;
use gt_inference::delta::{Applied, DeltaStack, MoveHeader, MoveKey, Recording};
use gt_inference::ids::{BEdge, Epoch, Group, Stamp, StateId};

struct Counts {
    mrs: i64,
}

impl Counts {
    /// `update_rs` (`entries.hh:359-418`), reduced to the one line that
    /// double-counts.
    fn commit(&mut self, a: Applied<'_, Directed, i64>) {
        for e in a.entries() {
            self.mrs += e.delta;
        }
    }
}

fn main() {
    let mut stack = DeltaStack::<Directed, i64>::with_levels(1);
    let stamp = Stamp {
        state: StateId::fresh(),
        epoch: Epoch(0),
    };

    let mut rec = Recording::new(&mut stack, MoveHeader::default(), stamp);
    let r = Group::new(0).expect("group 0");
    let s = Group::new(1).expect("group 1");
    rec.level_mut(0).begin(
        MoveKey {
            from: Some(r),
            to: Some(s),
        },
        2,
    );
    let mut resolve = |_r: Group, _s: Group| -> (Option<BEdge>, i64) { (None, 0) };
    rec.level_mut(0)
        .touch_dyn(r, s, 1, &mut resolve)
        .expect("in plane");

    let t = rec.seal();
    let a = t.into_levels().next().expect("one level");

    let mut st = Counts { mrs: 0 };
    st.commit(a);
    st.commit(a);
}
