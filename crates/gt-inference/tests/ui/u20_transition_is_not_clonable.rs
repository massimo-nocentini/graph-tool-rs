//! U20: a sealed transition is not a value that can be duplicated.
//!
//! `Transition` is the whole reified move -- the buffers for every level of
//! the hierarchy plus the shared header and stamp. Duplicating it would hand
//! out a second set of `Applied` tokens for the same buffers and put defect
//! #35 (`apply_delta` re-entered, `blockmodel/entries.hh:429`) back within
//! reach, one level at a time, past the `!Clone` on `Applied` itself.
//!
//! It is therefore neither `Clone` nor `Copy`. The receiver here is an owned
//! `Transition` and the type derefs to nothing, so the autoderef chain has
//! exactly one step and `Clone for Transition` is the only candidate on it:
//! the diagnostic is `error[E0599]`, "no method named `clone`". The
//! well-known `&T: Clone` fallback -- which would hand back a *copy of the
//! reference* and report no error at all -- is reachable only from a receiver
//! that is already a reference, and is why this fixture takes `t` by value.

use gt_core::dir::Directed;
use gt_inference::delta::{DeltaStack, MoveHeader, Recording};
use gt_inference::ids::{Epoch, Stamp, StateId};

fn main() {
    let mut stack = DeltaStack::<Directed, i64>::with_levels(2);
    let stamp = Stamp {
        state: StateId::fresh(),
        epoch: Epoch(0),
    };

    let t = Recording::new(&mut stack, MoveHeader::<i64>::default(), stamp).seal();
    let second = t.clone();
    println!("{}", second.n_levels());
}
