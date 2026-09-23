//! Row locking for concurrent commits.
//!
//! ## Why this exists at all ([DESIGN](crate::design) D9)
//!
//! Three of the six source designs independently removed graph-tool's
//! concurrent mutation and each scored the removal as a win. It is not a win.
//! `blockmodel/state.hh:152` builds `_group_mutex`, `:343-348` takes ordered
//! pair row locks via `do_lock<lock_t::shared>`, and `:451` sizes
//! `_m_entries_pool` to `get_num_threads()` -- the library commits moves from
//! several threads, and the scratch pool exists precisely for that.
//!
//! What Rust adds is not a prohibition but an enforcement: `partition.hh:78-84`
//! mixes a plain `_count[r] += w` between two `#pragma omp atomic` statements,
//! correct only if every caller holds `_group_mutex[r]` -- a convention, not a
//! type. Here the row can only be reached through its guard.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Two distinct rows, or one row named twice.
///
/// The variants matter. Returning `None` for `r == s` forces every caller to
/// hand-write a diagonal branch, and the diagonal is not an error case: it is
/// the block-graph self-loop `_mrs[r][r]`, which is common. `do_ulock_pair`
/// (`parallel_util.hh:247-256`) handles it transparently by locking once.
pub enum Pair<'a, T> {
    /// `r` and `s` were distinct; the references are in **caller order**.
    Two(&'a mut T, &'a mut T),
    /// `r == s`; one reference.
    Same(&'a mut T),
}

/// Borrow two rows of a slice mutably, in caller order.
///
/// Caller order is the whole point. A version returning `[&mut T; 2]` sorted
/// by index silently transposes: `pair_mut(xs, Group(3), Group(1))` hands back
/// `[&mut xs[1], &mut xs[3]]`, so a caller writing `pr` and `ps` positionally
/// updates the wrong group. graph-tool does not have that bug -- it sorts only
/// for `std::lock`'s deadlock avoidance and keeps `r` and `s` by name -- and a
/// port must not introduce it.
///
/// Returns `None` if either index is out of range, `Same` if `r == s`.
pub fn pair_mut<T>(xs: &mut [T], r: usize, s: usize) -> Option<Pair<'_, T>> {
    // The diagonal first: `get_disjoint_mut` rejects `[r, r]` as overlapping,
    // and `r == s` is not an error here -- it is `_mrs[r][r]`.
    if r == s {
        return xs.get_mut(r).map(Pair::Same);
    }

    // Sorted only to mirror `std::lock`'s deadlock-avoidance discipline in
    // `do_ulock_pair`; the pair is then re-normalised to caller order, which
    // is the part `do_ulock_pair` keeps and a sorted-array API loses. The
    // `Err` covers both out-of-range indices; overlap is impossible here.
    let (lo, hi) = if r < s { (r, s) } else { (s, r) };
    let [a, b] = xs.get_disjoint_mut([lo, hi]).ok()?;

    Some(if r < s {
        Pair::Two(a, b)
    } else {
        Pair::Two(b, a)
    })
}

/// Per-row mutexes, acquired in `(min, max)` order.
///
/// The port of `_group_mutex` (`blockmodel/state.hh:152`).
///
/// ## Two deliberate divergences from the C++
///
/// * `do_lock` (`parallel_util.hh:84-86`) calls `std::lock`, whose order is
///   an unspecified try-lock dance. A fixed `(min, max)` order is equally
///   deadlock-free and additionally *deterministic*, which is the property
///   the rest of `par` is built around.
/// * A `std::mutex` has no poison state. The guarded datum here is `()`, so
///   there is no invariant a panicking closure can leave broken; a poisoned
///   row is recovered rather than propagated, and the panic reaches the
///   caller by itself. Otherwise one panicking commit would wedge its two
///   groups for the rest of the process.
pub struct RowLocks {
    rows: Box<[Mutex<()>]>,
}

/// Acquire one row, ignoring poison. See the note on [`RowLocks`].
#[inline]
fn acquire(m: &Mutex<()>) -> MutexGuard<'_, ()> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl RowLocks {
    /// One mutex per row.
    pub fn new(n: usize) -> Self {
        RowLocks {
            rows: (0..n).map(|_| Mutex::new(())).collect(),
        }
    }

    /// Number of rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    /// Whether there are no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Run `f` holding the locks for `r` and `s`, taken in `(min, max)` order
    /// so that two commits can never deadlock against each other.
    ///
    /// `r == s` locks **once**: `std::sync::Mutex` is not re-entrant, so the
    /// transparent diagonal of `do_ulock_pair` (`parallel_util.hh:247-256`) is
    /// not a convenience here, it is the difference between working and
    /// self-deadlocking.
    ///
    /// # Panics
    ///
    /// If `r` or `s` is not a row, matching `_group_mutex[r]`.
    pub fn with_pair<R>(&self, r: usize, s: usize, f: impl FnOnce() -> R) -> R {
        if r == s {
            return self.with_row(r, f);
        }
        let (lo, hi) = if r < s { (r, s) } else { (s, r) };
        // Both bounds are checked before either mutex is taken, so an
        // out-of-range `hi` cannot panic with `lo` held.
        let (mlo, mhi) = (&self.rows[lo], &self.rows[hi]);
        let _glo = acquire(mlo);
        let _ghi = acquire(mhi);
        f()
    }

    /// Run `f` holding one row's lock.
    ///
    /// # Panics
    ///
    /// If `r` is not a row.
    pub fn with_row<R>(&self, r: usize, f: impl FnOnce() -> R) -> R {
        let _g = acquire(&self.rows[r]);
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D9's named case, by value rather than by index.
    #[test]
    fn pair_mut_is_in_caller_order() {
        let mut xs = [10, 11, 12, 13];
        match pair_mut(&mut xs, 3, 1) {
            Some(Pair::Two(a, b)) => {
                assert_eq!(*a, 13, "first reference must be r's row");
                assert_eq!(*b, 11, "second reference must be s's row");
            }
            _ => panic!("expected Two"),
        }
    }

    #[test]
    fn pair_mut_diagonal_is_same() {
        let mut xs = [10, 11, 12, 13];
        match pair_mut(&mut xs, 2, 2) {
            Some(Pair::Same(a)) => assert_eq!(*a, 12),
            _ => panic!("expected Same"),
        }
    }

    #[test]
    fn pair_mut_out_of_range_is_none() {
        let mut xs = [10, 11, 12, 13];
        assert!(pair_mut(&mut xs, 4, 1).is_none());
        assert!(pair_mut(&mut xs, 1, 4).is_none());
        assert!(pair_mut(&mut xs, 4, 4).is_none());
        let empty: &mut [i32] = &mut [];
        assert!(pair_mut(empty, 0, 0).is_none());
    }

    #[test]
    fn row_locks_are_reentrant_on_the_diagonal() {
        let l = RowLocks::new(4);
        assert_eq!(l.len(), 4);
        assert!(!l.is_empty());
        assert_eq!(l.with_pair(2, 2, || 7), 7);
    }
}
