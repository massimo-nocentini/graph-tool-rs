//! U12 — row locks, checked through the public surface.
//!
//! Two things are under test and they are not the same thing.
//!
//! 1. **[`pair_mut`] returns the two rows in caller order.** `gt_core::design` D9
//!    names the failure exactly: an API handing back `[&mut T; 2]` sorted by
//!    index turns `pair_mut(xs, 3, 1)` into `[&mut xs[1], &mut xs[3]]`, and a
//!    caller that binds `pr, ps` positionally then writes both groups'
//!    counters into each other. That transposition is invisible to any test
//!    that asserts on *indices*, because both orders name the same two rows.
//!    So every assertion below is on the **value** behind the reference.
//! 2. **[`RowLocks`] serialises and does not deadlock.** The diagonal is the
//!    interesting half: `std::sync::Mutex` is not re-entrant, so a `with_pair`
//!    that forgot `do_ulock_pair`'s `if (u != v)` branch
//!    (`parallel_util.hh:247-256`) self-deadlocks on the block-graph self-loop
//!    `_mrs[r][r]` — the single most common pair there is.
//!
//! The stress test increments its counters with a **non-atomic**
//! read-modify-write (`load` / `store`, `Relaxed`, with a yield wedged into
//! the window). That is deliberate: an `AtomicU64::fetch_add` would total up
//! correctly even with the locks removed entirely, and would therefore test
//! nothing but the absence of deadlock. Mutual exclusion is what makes the
//! totals come out, and the `Relaxed` accesses are sound because the mutex's
//! own acquire/release edges order every pair of accesses to a given row.

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use gt_core::par::{Pair, RowLocks, pair_mut};

// ===========================================================================
// 1. `pair_mut` — caller order, the diagonal, and the bounds
// ===========================================================================

/// The acceptance case from the unit spec, asserted by value.
///
/// `xs[i] = 100 + i`, so `xs[3]` and `xs[1]` are distinguishable from each
/// other and from their own indices; a sorted return fails here loudly.
#[test]
fn pair_mut_three_one_is_row_three_then_row_one() {
    let mut xs: Vec<i64> = (0..8).map(|i| 100 + i).collect();

    let Some(Pair::Two(pr, ps)) = pair_mut(&mut xs, 3, 1) else {
        panic!("distinct rows must give Two");
    };
    assert_eq!(*pr, 103, "the first reference is r's row (r = 3)");
    assert_eq!(*ps, 101, "the second reference is s's row (s = 1)");

    // And they are genuinely two independent borrows: write through both.
    *pr = -3;
    *ps = -1;
    assert_eq!(xs[3], -3);
    assert_eq!(xs[1], -1);
}

/// The mirror direction. `r < s` and `r > s` must both come back by name,
/// which is the whole of the sort/re-normalise contract.
#[test]
fn pair_mut_is_in_caller_order_both_ways() {
    let mut xs: Vec<i64> = (0..8).map(|i| 100 + i).collect();

    for (r, s) in [(1usize, 3usize), (3, 1), (0, 7), (7, 0), (2, 3), (3, 2)] {
        let Some(Pair::Two(pr, ps)) = pair_mut(&mut xs, r, s) else {
            panic!("({r}, {s}) are distinct");
        };
        assert_eq!(*pr, 100 + r as i64, "first reference must be row {r}");
        assert_eq!(*ps, 100 + s as i64, "second reference must be row {s}");
    }
}

/// `r == s` is `_mrs[r][r]`, not an error: `Same`, never `None`.
#[test]
fn pair_mut_diagonal_is_same_not_none() {
    let mut xs: Vec<i64> = (0..8).map(|i| 100 + i).collect();

    let Some(Pair::Same(p)) = pair_mut(&mut xs, 3, 3) else {
        panic!("the diagonal must give Same");
    };
    assert_eq!(*p, 103);
    *p = 42;
    assert_eq!(xs[3], 42);
}

#[test]
fn pair_mut_out_of_range_is_none() {
    let mut xs: Vec<i64> = (0..4).map(|i| 100 + i).collect();

    assert!(pair_mut(&mut xs, 4, 1).is_none(), "r out of range");
    assert!(pair_mut(&mut xs, 1, 4).is_none(), "s out of range");
    assert!(pair_mut(&mut xs, 9, 9).is_none(), "diagonal out of range");
    assert!(pair_mut(&mut xs, usize::MAX, 0).is_none(), "no wraparound");
    assert!(pair_mut(&mut xs, 0, usize::MAX).is_none(), "no wraparound");

    let empty: &mut [i64] = &mut [];
    assert!(
        pair_mut(empty, 0, 0).is_none(),
        "no row 0 in an empty slice"
    );

    // The last valid pair still works, so the bound is `len`, not `len - 1`.
    assert!(matches!(pair_mut(&mut xs, 3, 0), Some(Pair::Two(..))));
}

// ===========================================================================
// 2. `RowLocks` — the diagonal, and no deadlock under contention
// ===========================================================================

#[test]
fn row_locks_len_and_empty() {
    let l = RowLocks::new(5);
    assert_eq!(l.len(), 5);
    assert!(!l.is_empty());

    let z = RowLocks::new(0);
    assert_eq!(z.len(), 0);
    assert!(z.is_empty());
}

/// `with_pair(r, r)` must take the row's mutex **once**. If it takes it twice
/// this test does not fail, it hangs — which is exactly what the missing
/// `if (u != v)` branch does in production.
#[test]
fn with_pair_on_the_diagonal_locks_once() {
    let l = RowLocks::new(4);
    assert_eq!(l.with_pair(2, 2, || 7), 7);
    // Sequentially re-enterable: the guard is released on return.
    assert_eq!(l.with_pair(2, 2, || 8), 8);
    assert_eq!(l.with_row(2, || 9), 9);
}

/// A panicking closure poisons the row in `std`'s model. The guarded datum is
/// `()`, so there is no broken invariant to guard; the row must still be
/// usable afterwards, as `std::mutex` (which has no poison state at all)
/// leaves it. Otherwise one bad commit wedges two groups forever.
#[test]
fn a_panicking_closure_does_not_wedge_the_row() {
    let l = RowLocks::new(4);

    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        l.with_pair(1, 2, || panic!("commit blew up"));
    }));
    assert!(caught.is_err(), "the panic must reach the caller");

    assert_eq!(l.with_row(1, || 1), 1, "row 1 still usable");
    assert_eq!(l.with_row(2, || 2), 2, "row 2 still usable");
    assert_eq!(l.with_pair(1, 2, || 3), 3, "the pair still usable");
    assert_eq!(l.with_pair(2, 1, || 4), 4, "and in the other order");
}

// ---------------------------------------------------------------------------
// The stress test.
// ---------------------------------------------------------------------------

const ROWS: usize = 24;
const THREADS: usize = 16;
const OPS_PER_THREAD: usize = 100_000 / THREADS; // 6250; 100_000 pairs total.

/// splitmix64. A local generator rather than `rand`, so the sequence this
/// test asserts against is fixed by this file and cannot drift with a
/// dependency bump — the expected totals are computed from the very same
/// stream the threads consume.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `10^5` random pairs in random order over 16 threads.
///
/// Every thread walks its own randomly ordered pair list, and the lists
/// overlap heavily (24 rows, 16 threads), so `(r, s)` and `(s, r)` are taken
/// concurrently thousands of times. A `with_pair` that locked in caller order
/// instead of `(min, max)` deadlocks here; one that skipped the diagonal
/// branch deadlocks on the ~4% of pairs with `r == s`; one that dropped a
/// guard early loses increments.
#[test]
fn sixteen_threads_take_a_hundred_thousand_pairs() {
    // 1. Build every thread's work list up front, deterministically.
    let work: Vec<Vec<(usize, usize)>> = (0..THREADS)
        .map(|t| {
            let mut st = 0xC0FF_EE00_u64 ^ (t as u64).wrapping_mul(0x0000_0100_0000_01B3);
            (0..OPS_PER_THREAD)
                .map(|_| {
                    let r = (splitmix64(&mut st) % ROWS as u64) as usize;
                    let s = (splitmix64(&mut st) % ROWS as u64) as usize;
                    (r, s)
                })
                .collect()
        })
        .collect();

    // 2. The expected totals, from the same lists, serially.
    let mut expected = [0u64; ROWS];
    let mut diagonal = 0usize;
    for list in &work {
        for &(r, s) in list {
            expected[r] += 1;
            if r == s {
                diagonal += 1;
            } else {
                expected[s] += 1;
            }
        }
    }
    assert!(
        diagonal > 1_000,
        "the diagonal must actually be exercised, got {diagonal}"
    );

    // 3. Run it.
    let locks = RowLocks::new(ROWS);
    let counters: Vec<AtomicU64> = (0..ROWS).map(|_| AtomicU64::new(0)).collect();

    // Non-atomic read-modify-write: correct only under the row's lock.
    let bump = |i: usize, yield_now: bool| {
        let c = &counters[i];
        let v = c.load(Ordering::Relaxed);
        if yield_now {
            thread::yield_now();
        }
        c.store(v + 1, Ordering::Relaxed);
    };

    thread::scope(|scope| {
        for (t, list) in work.iter().enumerate() {
            let locks = &locks;
            let bump = &bump;
            scope.spawn(move || {
                for (i, &(r, s)) in list.iter().enumerate() {
                    // Widen the race window on a slice of the ops; a lost
                    // update then shows up in the totals within a run or two
                    // rather than once in a blue moon.
                    let y = (i + t) % 41 == 0;
                    locks.with_pair(r, s, || {
                        bump(r, y);
                        if r != s {
                            bump(s, y);
                        }
                    });
                }
            });
        }
    });

    // 4. Every counter at its expected total.
    let got: Vec<u64> = counters.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    assert_eq!(got, expected.to_vec(), "lost or duplicated increments");

    let total: u64 = got.iter().sum();
    assert_eq!(
        total,
        2 * (THREADS * OPS_PER_THREAD) as u64 - diagonal as u64
    );
}

/// The same contention with the two entry points mixed. `with_row(r)` taken
/// concurrently with `with_pair(r, s)` must still serialise row `r`, and a
/// pair taken against a bare row must not deadlock either way round.
///
/// The work lists are pre-rolled here too, so this is an *exact* total and
/// not a plausibility range.
#[test]
fn rows_and_pairs_interleave() {
    const N: usize = 8;
    const ITERS: usize = 20_000;

    // `None` = `with_row(r)`, `Some(s)` = `with_pair(r, s)`.
    let work: Vec<Vec<(usize, Option<usize>)>> = (0..THREADS)
        .map(|t| {
            let mut st = 0xDEAD_BEEF_u64 ^ (t as u64).wrapping_mul(0x9E37_79B9);
            (0..ITERS)
                .map(|_| {
                    let r = (splitmix64(&mut st) % N as u64) as usize;
                    let s = (splitmix64(&mut st) % N as u64) as usize;
                    (r, if t % 2 == 0 { None } else { Some(s) })
                })
                .collect()
        })
        .collect();

    let mut expected = [0u64; N];
    for list in &work {
        for &(r, s) in list {
            expected[r] += 1;
            if let Some(s) = s
                && s != r
            {
                expected[s] += 1;
            }
        }
    }

    let locks = RowLocks::new(N);
    let counters: Vec<AtomicU64> = (0..N).map(|_| AtomicU64::new(0)).collect();

    let bump = |i: usize| {
        let c = &counters[i];
        let v = c.load(Ordering::Relaxed);
        c.store(v + 1, Ordering::Relaxed);
    };

    thread::scope(|scope| {
        for list in work.iter() {
            let locks = &locks;
            let bump = &bump;
            scope.spawn(move || {
                for &(r, s) in list {
                    match s {
                        None => locks.with_row(r, || bump(r)),
                        Some(s) => locks.with_pair(r, s, || {
                            bump(r);
                            if r != s {
                                bump(s);
                            }
                        }),
                    }
                }
            });
        }
    });

    let got: Vec<u64> = counters.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    assert_eq!(got, expected.to_vec(), "lost or duplicated increments");
}
