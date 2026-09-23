//! The typestate of a transition: record, seal, price, commit, audit.
//!
//! ## Why `commit` consumes a *level*, not a transition ([DESIGN](gt_core::design) D10)
//!
//! Making `commit(&mut self, t: Transition)` take the whole transition by
//! value does buy one thing -- replaying a delta twice is a move error, where
//! `apply_delta` (`entries.hh:429`) has no notion of whether it has already
//! run and double-counts every entry if re-entered. But it also makes two
//! things the design claims to support impossible:
//!
//! * the always-on audit cannot be called, because after `commit` the
//!   transition has moved (`error[E0382]`) and before `commit` the audit
//!   always fails;
//! * the **nested block model cannot be committed at all**: `apply_delta` must
//!   recurse into the coupled state with `*m_entries._next` (`:488`), i.e.
//!   level `l` goes to state `l`, and there is only one value to give away.
//!
//! Making the *level* the unit of consumption keeps the double-apply move
//! error -- [`Applied`] is `!Clone` and taken by value -- and fixes both.
//! [`Receipt`] then carries the before-image past the commit so the audit
//! compiles.

use gt_core::dir::Dir;

use super::buf::DeltaBuf;
use super::entry::{Entry, MoveHeader};
use super::stack::DeltaStack;
use crate::ids::{Stamp, Weight};

/// A transition being filled. Holds the workspace uniquely.
///
/// The lifetime is the **workspace's**, never the state's. That is the
/// type-level statement that lets `st.commit(..)` take `&mut st` while a
/// priced transition is still alive.
pub struct Recording<'w, D: Dir, W: Weight> {
    stack: &'w mut DeltaStack<D, W>,
    hdr: MoveHeader<W>,
    stamp: Stamp,
}

impl<'w, D: Dir, W: Weight> Recording<'w, D, W> {
    /// Begin recording against a state.
    pub fn new(stack: &'w mut DeltaStack<D, W>, hdr: MoveHeader<W>, stamp: Stamp) -> Self {
        Recording { stack, hdr, stamp }
    }

    /// The header being built.
    #[inline]
    pub fn header_mut(&mut self) -> &mut MoveHeader<W> {
        &mut self.hdr
    }

    /// One level's buffer.
    #[inline]
    pub fn level_mut(&mut self, l: usize) -> &mut DeltaBuf<D, W> {
        self.stack.level_mut(l)
    }

    /// Level `l` shared and level `l + 1` mutably, for
    /// [`propagate`](crate::blockmodel::propagate).
    #[inline]
    pub fn below_above(&mut self, l: usize) -> (&DeltaBuf<D, W>, &mut DeltaBuf<D, W>) {
        self.stack.below_above(l)
    }

    /// Freeze. The buffers become read-only for every consumer.
    pub fn seal(self) -> Transition<'w, D, W> {
        Transition {
            stack: self.stack,
            hdr: self.hdr,
            stamp: self.stamp,
        }
    }
}

/// A frozen transition, ready to be priced and then applied.
///
/// Deliberately **not** `Clone` and not `Copy`.
pub struct Transition<'w, D: Dir, W: Weight> {
    stack: &'w DeltaStack<D, W>,
    hdr: MoveHeader<W>,
    stamp: Stamp,
}

impl<'w, D: Dir, W: Weight> Transition<'w, D, W> {
    /// Which state and revision this was recorded against.
    #[inline]
    pub const fn stamp(&self) -> Stamp {
        self.stamp
    }

    /// Number of hierarchy levels.
    #[inline]
    pub fn n_levels(&self) -> usize {
        self.stack.len()
    }

    /// A priceable view of one level. Shared, so several coexist.
    #[inline]
    pub fn level(&self, l: usize) -> Delta<'_, D, W> {
        Delta {
            buf: self.stack.level(l),
            hdr: &self.hdr,
        }
    }

    /// Consume, yielding one [`Applied`] token per level in order.
    pub fn into_levels(self) -> LevelIter<'w, D, W> {
        LevelIter {
            stack: self.stack,
            hdr: self.hdr,
            stamp: self.stamp,
            next: 0,
        }
    }
}

/// One level of a frozen transition. `Copy`, and borrows nothing mutably.
#[derive(Clone, Copy)]
pub struct Delta<'a, D: Dir, W: Weight> {
    buf: &'a DeltaBuf<D, W>,
    hdr: &'a MoveHeader<W>,
}

impl<D: Dir, W: Weight> Delta<'_, D, W> {
    /// The changed block pairs, with their before-images.
    #[inline]
    pub fn entries(&self) -> &[Entry<W>] {
        self.buf.entries()
    }
    /// The per-endpoint before-image and degree deltas.
    #[inline]
    pub fn header(&self) -> &MoveHeader<W> {
        self.hdr
    }
}

/// Permission to apply exactly one level, exactly once.
///
/// `!Clone`, `!Copy`, and consumed by value.
pub struct Applied<'w, D: Dir, W: Weight> {
    buf: &'w DeltaBuf<D, W>,
    hdr: MoveHeader<W>,
    stamp: Stamp,
    level: usize,
}

impl<D: Dir, W: Weight> Applied<'_, D, W> {
    /// Which state and revision this level was recorded against.
    #[inline]
    pub const fn stamp(&self) -> Stamp {
        self.stamp
    }
    /// Which level.
    #[inline]
    pub const fn level(&self) -> usize {
        self.level
    }
    /// The changed block pairs.
    #[inline]
    pub fn entries(&self) -> &[Entry<W>] {
        self.buf.entries()
    }
    /// The header.
    #[inline]
    pub const fn header(&self) -> &MoveHeader<W> {
        &self.hdr
    }
}

/// Yields one [`Applied`] per level, bottom-up.
pub struct LevelIter<'w, D: Dir, W: Weight> {
    stack: &'w DeltaStack<D, W>,
    hdr: MoveHeader<W>,
    stamp: Stamp,
    next: usize,
}

impl<'w, D: Dir, W: Weight> Iterator for LevelIter<'w, D, W> {
    type Item = Applied<'w, D, W>;
    fn next(&mut self) -> Option<Applied<'w, D, W>> {
        if self.next >= self.stack.len() {
            return None;
        }
        let l = self.next;
        self.next += 1;
        Some(Applied {
            buf: self.stack.level(l),
            hdr: self.hdr,
            stamp: self.stamp,
            level: l,
        })
    }
}

/// The before-image, forwarded past a commit so the audit can run.
///
/// Without this, an always-on `audit_commit(st, &transition)` cannot be
/// called: the transition has been moved into `commit`, and calling it
/// *before* the commit fails on every non-zero delta because the state still
/// holds `mrs_before`.
#[derive(Clone, Debug)]
pub struct Receipt<W: Weight> {
    /// The applied entries, with their before-images.
    pub entries: Vec<Entry<W>>,
    /// The header.
    pub hdr: MoveHeader<W>,
    /// The stamp the delta carried.
    pub stamp: Stamp,
    /// Which level was applied.
    pub level: usize,
}
