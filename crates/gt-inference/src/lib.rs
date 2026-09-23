//! # gt-inference
//!
//! ## The organising idea, which the C++ states nowhere
//!
//! **The state transition is reified as data.** `modify_entries_dispatch`
//! (`blockmodel/entries.hh:227`) turns "move vertex `v` from group `r` to
//! group `nr`" into an explicit sparse `EntrySet`, and that one object is then
//! consumed by three unrelated clients: `entries_dS` prices it,
//! `get_move_prob` synthesises post-move counts from it to compute detailed
//! balance in a state that does not yet exist, and `apply_delta` finally
//! replays it.
//!
//! ## What actually dissolves the borrow conflict (DESIGN.md D10)
//!
//! Not, as one design argued, carrying a before-image in each entry. Three
//! readers are three *shared* borrows and never conflicted with each other;
//! the conflict is that the buffer is a **member of the state**
//! (`blockmodel/state.hh:2545`), so the canonical call
//! `this->entries_dS(..., this->_m_entries)` has `&mut self` aliasing
//! `&mut self._m_entries`.
//!
//! The fix is the **ownership inversion**: the buffer lives in a caller-owned
//! [`Workspace`](delta::Workspace), one per thread. That is what
//! `_m_entries_pool[tid]` (`state.hh:463`) already is, in disguise; making it
//! the only form removes the thread-id indexing, the `set_concurrent` pool
//! resizing, and the aliasing.
//!
//! The before-image *is* kept, for a different and honest reason: it removes
//! two cache-cold dependent loads per entry from the pricing loop -- the
//! `_emat` hash probe and the `_mrs[me]` indirection at `state.hh:1225` --
//! turning `entries_dS` into a contiguous scan over `&[Entry<W>]`. That is a
//! cache-locality optimisation, not a soundness argument, and it is the best
//! single performance idea in this port.

#![forbid(unsafe_code)]
// SKELETON: see gt-core's lib.rs.
#![allow(dead_code, unused_variables)]
#![warn(missing_docs)]

pub mod blockmodel;
pub mod delta;
pub mod ids;
pub mod metropolis;
pub mod spec;

pub use ids::{BEdge, Epoch, Group, Stamp, StateId, Weight};
