//! `gt_core::design` §6, first half of the `value_subset!` guarantee.
//!
//! `$variant => $ty` is, on its face, an unchecked claim: nothing about the
//! macro's *expansion* requires the named [`ValueKind`] and the named type to
//! be the same member. Transposing two rows therefore builds cleanly under a
//! naive expansion and fails at **runtime**, inside `dispatch`'s
//! `downcast().expect(...)` -- or, worse, in `narrow`, which then reports a
//! `DispatchError` naming the *wrong* type as offered and listing it as
//! accepted. That is strictly less honest than graph-tool's own
//! `DispatchNotFound` ("This is a graph_tool bug. :-(", `dispatch.hh:86-88`),
//! which at least does not lie about what it was given.
//!
//! The generated `const _: () = assert!(matches!(<$ty>::KIND, ...))` turns it
//! into:
//!
//! ```text
//! error[E0080]: evaluation panicked: value_subset! Bad: variant I16 is
//!               mapped to i32, whose KIND differs
//! ```
//!
//! Both rows below are transposed, so both assertions fail: the diagnostic is
//! per-row, not "somewhere in this macro".

use gt_core::prop::Scalar;
use gt_core::value_subset;

value_subset!(
    /// `I16` holds `i16` and `I32` holds `i32`. These are the other way round.
    pub Bad, BadKernel, "value", Scalar,
    { I16 => i32, I32 => i16 }
);

fn main() {}
