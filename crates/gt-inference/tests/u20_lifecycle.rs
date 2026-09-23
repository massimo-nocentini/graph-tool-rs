//! U20 -- the delta stack and the transition lifecycle.
//!
//! Three things are under test here, and all three are places where the C++
//! relies on an invariant it never writes down.
//!
//! * **The stack.** `EntrySet::_next` (`entries.hh:207`) is a raw pointer,
//!   assigned from another *state's* member (`state.hh:983`) and, on the
//!   parallel path, into `_m_entries_pool`'s heap buffer -- which the very
//!   same function `resize()`s (`:497`, `:500`) and then `clear()`s and
//!   `shrink_to_fit()`s (`:513-526`). Defect #38. Here the chain is a `Vec`
//!   owned by the caller and the disjointness "read level `l`, write level
//!   `l + 1`" that `propagate_entries` (`:1099-1127`) assumes is proved by
//!   `split_at_mut` instead of assumed.
//!
//! * **The lifecycle.** `apply_delta` (`entries.hh:429`) has no notion of
//!   having already run: call it twice on one `EntrySet` and every entry is
//!   counted twice (defect #35). [`Applied`] is `!Clone` and consumed by
//!   value, so the second call does not compile -- pinned by
//!   `tests/ui/u20_applied_is_not_reusable.rs`.
//!
//! * **The audit.** `__test__` is `False` (`base_states.py:33`) and the
//!   delta/absolute cross-check never runs (defect #37). It can only run at
//!   all if the before-image outlives the commit, which is what [`Receipt`]
//!   is for; [`the_receipt_carries_the_before_image_past_the_commit`] is that
//!   check, performed against a state the commit has already mutated.
//!
//! The negative guarantees are `trybuild` fixtures rather than prose, and the
//! *positive* one -- a [`Delta`] held across a `&mut state` call, which is the
//! entire point of the ownership inversion -- is a fixture too, because
//! "this still compiles" regresses as silently as "this no longer fails".

use std::collections::HashMap;

use gt_core::dir::{Directed, Undirected};
use gt_inference::delta::{
    Applied, Delta, DeltaBuf, DeltaStack, EndImage, Entry, MoveHeader, MoveKey, Receipt,
    Recording, Transition, Workspace,
};
use gt_inference::ids::{BEdge, Epoch, Group, Stamp, StateId, Weight};

/// Group count the field tables are sized for.
const B: usize = 6;

fn g(i: u32) -> Group {
    Group::new(i).expect("group index fits")
}

fn stamp() -> Stamp {
    Stamp {
        state: StateId::fresh(),
        epoch: Epoch(7),
    }
}

/// A deterministic before-image, distinguishable per pair, mirroring the
/// `(get_me, mrs)` pair `BlockView::resolve` interns.
fn resolve(r: Group, s: Group) -> (Option<BEdge>, i64) {
    if (r.index() + s.index()).is_multiple_of(3) {
        (None, 0)
    } else {
        (
            Some(BEdge((r.index() * 16 + s.index()) as u32)),
            (r.index() * 100 + s.index()) as i64,
        )
    }
}

/// A move header with recognisable scalars, so that "the header reached every
/// level" is not vacuously true of the default.
fn header(r: u32, nr: u32) -> MoveHeader<i64> {
    MoveHeader {
        r: Group::new(r),
        nr: Group::new(nr),
        r_img: EndImage {
            mrp: 11,
            mrm: 12,
            wr: 13,
        },
        nr_img: EndImage {
            mrp: 21,
            mrm: 22,
            wr: 23,
        },
        dkin: 2,
        dkout: 3,
        dr: -1,
        dnr: 1,
    }
}

/// Fill one level with `(s, t, delta)` triples through the public recorder
/// entry point. Every pair must contain an endpoint of `mv`, exactly as
/// `get_field` (`entries.hh:108-119`) requires.
fn fill(buf: &mut DeltaBuf<Directed, i64>, mv: MoveKey, pairs: &[(u32, u32, i64)]) {
    buf.begin(mv, B);
    for &(s, t, d) in pairs {
        buf.touch_dyn(g(s), g(t), d, &mut resolve)
            .expect("the fixture pairs are in plane");
    }
}

// ===========================================================================
// 1. `with_levels`
// ===========================================================================

/// `count_L` (`state.hh:996-1002`) returns `1` for an uncoupled state, so the
/// bottom of the chain is a level like any other and `with_levels(1)` is the
/// ordinary block model, not a degenerate case.
#[test]
fn with_levels_counts_the_uncoupled_state_itself() {
    for n in 1..=6 {
        let s = DeltaStack::<Directed, i64>::with_levels(n);
        assert_eq!(s.len(), n);
        assert!(!s.is_empty());
        for l in 0..n {
            assert!(s.level(l).entries().is_empty());
            // Each buffer sizes its own field table on its first `begin`;
            // `with_levels` allocates none of them (`entries.hh:48`).
            assert_eq!(s.level(l).table_len(), 0);
            assert!(s.level(l).table_is_clear());
            assert_eq!(s.level(l).field_writes(), 0);
        }
    }
    assert_eq!(DeltaStack::<Directed, i64>::default().len(), 1);
    assert_eq!(Workspace::<Directed, i64>::default().stack.len(), 1);
    assert_eq!(Workspace::<Undirected, i64>::with_levels(3).stack.len(), 3);
}

/// A stack with no levels has nowhere to record, and every later call would
/// fail with an index panic naming this type rather than the caller.
#[test]
#[should_panic(expected = "at least one level")]
fn a_stack_with_no_levels_is_refused() {
    let _ = DeltaStack::<Directed, i64>::with_levels(0);
}

/// The levels are `n` distinct buffers, not `n` clones of one. `vec![x; n]`
/// would also satisfy the length assertion above.
#[test]
fn the_levels_are_independent_buffers() {
    let mut s = DeltaStack::<Directed, i64>::with_levels(4);
    for l in 0..4u32 {
        fill(
            s.level_mut(l as usize),
            MoveKey {
                from: Group::new(l),
                to: Group::new(l + 1),
            },
            &[(l, l + 2, 10 + l as i64)],
        );
    }
    for l in 0..4u32 {
        let b = s.level(l as usize);
        assert_eq!(b.entries().len(), 1);
        assert_eq!(b.move_key().from, Group::new(l));
        assert_eq!(b.delta_of(g(l), g(l + 2)), 10 + l as i64);
    }
}

// ===========================================================================
// 2. `below_above` -- the disjointness `propagate_entries` assumes
// ===========================================================================

/// The acceptance test: level `l` shared and level `l + 1` mutably, at once.
///
/// This is `propagate_entries` (`state.hh:1109-1117`) streamed straight from
/// the below-level entries into the above-level buffer, including the
/// `if (delta == 0) return;` skip at `:1113-1114`. No intermediate vector is
/// materialised -- if `below_above` did not prove the disjointness, this
/// function would not compile at all, which is the whole point of the test.
#[test]
fn below_above_reads_one_level_while_writing_the_next() {
    let mut s = DeltaStack::<Directed, i64>::with_levels(2);

    // The hierarchy projection: level-0 groups 0,2 sit in level-1 group 0,
    // and 1,3 in level-1 group 1.
    let b_of = |r: Group| g(if r.index().is_multiple_of(2) { 0 } else { 1 });

    fill(
        s.level_mut(0),
        MoveKey {
            from: Group::new(0),
            to: Group::new(1),
        },
        &[
            (0, 2, 5),  // -> (0, 0)
            (2, 0, 3),  // -> (0, 0), same projected pair
            (1, 3, 9),  // -> (1, 1)
            (0, 1, 4),  // -> (0, 1)
            (0, 4, 0),  // delta 0: `:1113` skips it
        ],
    );

    let above_mv = MoveKey {
        from: Some(b_of(g(0))),
        to: Some(b_of(g(1))),
    };

    {
        let (below, above) = s.below_above(0);
        // Both borrows are live across this whole block.
        above.begin(above_mv, B);
        let mut res = resolve;
        for e in below.entries() {
            if e.delta == 0 {
                continue;
            }
            above
                .touch_dyn(b_of(e.r), b_of(e.s), e.delta, &mut res)
                .expect("a projected pair contains a projected endpoint");
        }
        // The shared half is a real view of the live buffer, not a snapshot.
        assert_eq!(below.entries().len(), 5);
        assert_eq!(above.entries().len(), 3);
    }

    assert_eq!(s.level(1).delta_of(g(0), g(0)), 8, "5 + 3 merge into one pair");
    assert_eq!(s.level(1).delta_of(g(1), g(1)), 9);
    assert_eq!(s.level(1).delta_of(g(0), g(1)), 4);
    assert_eq!(s.level(1).delta_of(g(1), g(0)), 0, "never touched");
    assert_eq!(s.level(1).entries().len(), 3, "the zero delta was skipped");

    // The below level is untouched by the projection.
    assert_eq!(s.level(0).entries().len(), 5);
    assert_eq!(s.level(0).delta_of(g(0), g(2)), 5);
}

/// Every adjacent pair splits, and the split names the right two levels --
/// `l` below and `l + 1` above, never `l - 1` or `l + 2`.
#[test]
fn below_above_names_l_and_l_plus_one_for_every_adjacent_pair() {
    const N: usize = 5;
    let mut s = DeltaStack::<Directed, i64>::with_levels(N);
    for l in 0..N as u32 {
        fill(
            s.level_mut(l as usize),
            MoveKey {
                from: Group::new(0),
                to: Group::new(1),
            },
            &[(0, l + 1, 100 + l as i64)],
        );
    }
    for l in 0..N - 1 {
        let (below, above) = s.below_above(l);
        assert_eq!(below.entries()[0].delta, 100 + l as i64);
        assert_eq!(above.entries()[0].delta, 101 + l as i64);
    }
}

/// The top level has nothing above it: `propagate_entries` recurses only
/// through `visit_coupled_if` (`state.hh:1119-1126`).
#[test]
#[should_panic]
fn below_above_has_no_level_above_the_top() {
    let mut s = DeltaStack::<Directed, i64>::with_levels(2);
    let _ = s.below_above(1);
}

// ===========================================================================
// 3. `Recording` -> `Transition` -> `Applied`
// ===========================================================================

/// Record one transition over `n` levels, each level carrying a delta that
/// identifies it.
fn record_over(s: &mut DeltaStack<Directed, i64>, hdr: MoveHeader<i64>, st: Stamp) -> usize {
    let n = s.len();
    let mut rec = Recording::new(s, hdr, st);
    for l in 0..n {
        fill(
            rec.level_mut(l),
            MoveKey {
                from: Group::new(0),
                to: Group::new(1),
            },
            &[(0, 2, (l as i64 + 1) * 10), (1, 3, -(l as i64 + 1))],
        );
    }
    // Seal, then drop: the point here is only that the buffers survive.
    let t = rec.seal();
    assert_eq!(t.n_levels(), n);
    n
}

/// The acceptance test: exactly `n_levels()` tokens, bottom-up, each carrying
/// **its own** level index and **the shared** stamp.
#[test]
fn into_levels_yields_one_token_per_level_bottom_up() {
    for n in 1..=5 {
        let mut s = DeltaStack::<Directed, i64>::with_levels(n);
        let st = stamp();
        let hdr = header(0, 1);

        let mut rec = Recording::new(&mut s, hdr, st);
        for l in 0..n {
            fill(
                rec.level_mut(l),
                MoveKey {
                    from: Group::new(0),
                    to: Group::new(1),
                },
                &[(0, 2, (l as i64 + 1) * 10)],
            );
        }
        let t = rec.seal();
        assert_eq!(t.n_levels(), n);
        assert_eq!(t.stamp(), st);

        let mut seen = 0usize;
        for (i, a) in t.into_levels().enumerate() {
            assert_eq!(a.level(), i, "bottom-up, level index carried");
            assert_eq!(a.stamp(), st, "one stamp for the whole transition");
            assert_eq!(a.header(), &hdr);
            // Its own buffer, not level 0's.
            assert_eq!(a.entries().len(), 1);
            assert_eq!(a.entries()[0].delta, (i as i64 + 1) * 10);
            assert_eq!(a.entries()[0].r, g(0));
            assert_eq!(a.entries()[0].s, g(2));
            seen += 1;
        }
        assert_eq!(seen, n, "exactly n_levels() tokens");
    }
}

/// `LevelIter` is fused by construction: the cursor only ever advances, so a
/// second pass over an exhausted iterator cannot hand out a second token for
/// the top level.
#[test]
fn into_levels_is_exhausted_exactly_once() {
    let mut s = DeltaStack::<Directed, i64>::with_levels(3);
    let n = record_over(&mut s, header(0, 1), stamp());
    assert_eq!(n, 3);

    let mut it = Recording::new(&mut s, header(0, 1), stamp())
        .seal()
        .into_levels();
    assert_eq!(it.next().map(|a| a.level()), Some(0));
    assert_eq!(it.next().map(|a| a.level()), Some(1));
    assert_eq!(it.next().map(|a| a.level()), Some(2));
    assert!(it.next().is_none());
    assert!(it.next().is_none());
    assert!(it.next().is_none());
}

/// A sealed transition hands out `Delta`s by shared borrow, so all of them
/// coexist -- three readers were always three shared borrows
/// (`entries_dS`, `get_move_prob`, `apply_delta`); the C++ conflict was that
/// the buffer was a member of the state (`state.hh:2545`).
#[test]
fn several_deltas_of_one_transition_coexist() {
    let mut s = DeltaStack::<Directed, i64>::with_levels(3);
    let hdr = header(0, 1);
    let mut rec = Recording::new(&mut s, hdr, stamp());
    for l in 0..3 {
        fill(
            rec.level_mut(l),
            MoveKey {
                from: Group::new(0),
                to: Group::new(1),
            },
            &[(0, 2, (l as i64 + 1) * 10)],
        );
    }
    let t = rec.seal();

    let d0 = t.level(0);
    let d1 = t.level(1);
    let d2 = t.level(2);
    let again = t.level(0);
    // `Delta` is `Copy`: passing one to a pricing function does not move it.
    let copied: Delta<'_, Directed, i64> = d0;

    assert_eq!(d0.entries()[0].delta, 10);
    assert_eq!(d1.entries()[0].delta, 20);
    assert_eq!(d2.entries()[0].delta, 30);
    assert_eq!(again.entries(), d0.entries());
    assert_eq!(copied.entries(), d0.entries());
    for d in [d0, d1, d2] {
        assert_eq!(d.header(), &hdr, "the header is shared by every level");
    }
}

/// `set_move`/`clear` (`entries.hh:59-65`, `:169-176`) are temporally
/// coupled; `begin` is not, and a second recording against the same stack
/// therefore starts from a clean field table whatever the previous key was.
#[test]
fn a_second_recording_reuses_the_same_workspace() {
    let mut ws = Workspace::<Directed, i64>::with_levels(2);
    let first = stamp();

    // Scoped, so the first recording's borrow of the workspace ends here and
    // the second one can take it: the workspace outlives both, which is what
    // `_m_entries_pool[tid]` (`state.hh:463`) is for.
    {
        let mut rec = Recording::new(&mut ws.stack, header(0, 1), first);
        fill(
            rec.level_mut(0),
            MoveKey {
                from: Group::new(0),
                to: Group::new(1),
            },
            &[(0, 2, 5), (1, 4, 7)],
        );
        let t = rec.seal();
        assert_eq!(t.level(0).entries().len(), 2);
        assert_eq!(t.stamp(), first);
    }

    let second = Stamp {
        state: first.state,
        epoch: Epoch(first.epoch.0 + 1),
    };
    let mut rec = Recording::new(&mut ws.stack, header(2, 3), second);
    fill(
        rec.level_mut(0),
        MoveKey {
            from: Group::new(2),
            to: Group::new(3),
        },
        &[(2, 5, 1)],
    );
    let t = rec.seal();
    assert_eq!(t.stamp(), second);
    assert_eq!(t.level(0).entries().len(), 1, "no residue from the first key");
    assert_eq!(t.level(0).entries()[0].delta, 1);
    // The levels that were never re-filled kept their (empty) state.
    assert_eq!(t.level(1).entries().len(), 0);
}

// ===========================================================================
// 4. The audit: `Receipt` outliving the commit
// ===========================================================================

/// The smallest thing that can consume an [`Applied`]: the `_mrs` half of
/// `update_rs` (`entries.hh:359`), and nothing else.
///
/// Deliberately *not* an implementation of `BlockCommit` -- U20 owns the
/// lifecycle, not the block state, and this stands in for whatever the state
/// turns out to be.
#[derive(Debug, Default)]
struct Counts {
    mrs: HashMap<(usize, usize), i64>,
    epoch: u64,
    commits: usize,
}

impl Counts {
    fn mrs(&self, r: Group, s: Group) -> i64 {
        self.mrs.get(&(r.index(), s.index())).copied().unwrap_or(0)
    }

    /// Consumes the token by value. Replaying it is `error[E0382]`.
    fn commit(&mut self, a: Applied<'_, Directed, i64>) -> Receipt<i64> {
        for e in a.entries() {
            *self.mrs.entry((e.r.index(), e.s.index())).or_insert(0) += e.delta;
        }
        self.epoch += 1;
        self.commits += 1;
        Receipt {
            entries: a.entries().to_vec(),
            hdr: *a.header(),
            stamp: a.stamp(),
            level: a.level(),
        }
    }
}

/// `__test__ = False` (`base_states.py:33`, defect #37): the cross-check
/// cannot run *before* the commit, because the state still holds
/// `mrs_before`, and it cannot run *after* if the transition was moved into
/// the commit. [`Receipt`] is what makes "after" possible.
#[test]
fn the_receipt_carries_the_before_image_past_the_commit() {
    let mut st = Counts::default();
    st.mrs.insert((0, 2), 100);
    st.mrs.insert((1, 3), 40);

    let mut s = DeltaStack::<Directed, i64>::with_levels(1);
    let sstamp = stamp();
    let mut rec = Recording::new(&mut s, header(0, 1), sstamp);
    {
        let buf = rec.level_mut(0);
        buf.begin(
            MoveKey {
                from: Group::new(0),
                to: Group::new(1),
            },
            B,
        );
        // Intern the *state's* before-image, the way `BlockView::resolve`
        // does, rather than the synthetic one.
        let before = [((0usize, 2usize), 100i64), ((1, 3), 40)]
            .into_iter()
            .collect::<HashMap<_, _>>();
        let mut res = |r: Group, s: Group| {
            (
                Some(BEdge((r.index() * 16 + s.index()) as u32)),
                before.get(&(r.index(), s.index())).copied().unwrap_or(0),
            )
        };
        buf.touch_dyn(g(0), g(2), -7, &mut res).expect("in plane");
        buf.touch_dyn(g(1), g(3), 5, &mut res).expect("in plane");
    }
    let t = rec.seal();

    // Price first: three shared readers, then one mutable writer.
    let priced: i64 = t.level(0).entries().iter().map(|e| e.delta).sum();

    let mut it = t.into_levels();
    let a = it.next().expect("one level");
    let r = st.commit(a);
    assert!(it.next().is_none());

    assert_eq!(st.commits, 1);
    assert_eq!(r.level, 0);
    assert_eq!(r.stamp, sstamp);
    assert_eq!(priced, -2);

    // The audit, run against the already-mutated state. This is exactly the
    // O(#entries) check `audit_commit` performs, and it is only expressible
    // because `r` outlived the commit.
    for e in &r.entries {
        assert_eq!(
            st.mrs(e.r, e.s),
            e.mrs_before + e.delta,
            "entry {:?} -> {:?} did not land",
            e.r,
            e.s
        );
    }
    assert_eq!(st.mrs(g(0), g(2)), 93);
    assert_eq!(st.mrs(g(1), g(3)), 45);
}

/// One `Applied` per level, each committed into its own state -- the nested
/// block model, which is the case `commit(Transition)` could not express at
/// all (there is only one value to give away, D10).
#[test]
fn each_level_commits_into_its_own_state() {
    const N: usize = 3;
    let mut states: Vec<Counts> = (0..N).map(|_| Counts::default()).collect();

    let mut s = DeltaStack::<Directed, i64>::with_levels(N);
    let sstamp = stamp();
    let mut rec = Recording::new(&mut s, header(0, 1), sstamp);
    for l in 0..N {
        fill(
            rec.level_mut(l),
            MoveKey {
                from: Group::new(0),
                to: Group::new(1),
            },
            &[(0, 2, (l as i64 + 1) * 10)],
        );
    }

    let mut receipts: Vec<Receipt<i64>> = Vec::new();
    for a in rec.seal().into_levels() {
        let l = a.level();
        receipts.push(states[l].commit(a));
    }

    assert_eq!(receipts.len(), N);
    for (l, r) in receipts.iter().enumerate() {
        assert_eq!(r.level, l);
        assert_eq!(r.stamp, sstamp);
        assert_eq!(states[l].commits, 1, "level {l} was applied exactly once");
        assert_eq!(states[l].mrs(g(0), g(2)), (l as i64 + 1) * 10);
    }
}

/// The tokens are values: they can be collected, reordered, and applied in
/// any order, because each names the level it belongs to. `apply_delta`
/// recurses top-down through `*m_entries._next` (`entries.hh:488`) and has no
/// such freedom.
#[test]
fn the_tokens_may_be_applied_in_any_order() {
    const N: usize = 4;
    let mut s = DeltaStack::<Directed, i64>::with_levels(N);
    let mut rec = Recording::new(&mut s, header(0, 1), stamp());
    for l in 0..N {
        fill(
            rec.level_mut(l),
            MoveKey {
                from: Group::new(0),
                to: Group::new(1),
            },
            &[(0, 2, (l as i64 + 1) * 10)],
        );
    }

    let mut tokens: Vec<_> = rec.seal().into_levels().collect();
    assert_eq!(tokens.len(), N);
    tokens.reverse();

    let mut states: Vec<Counts> = (0..N).map(|_| Counts::default()).collect();
    let mut order = Vec::new();
    for a in tokens {
        let l = a.level();
        order.push(l);
        states[l].commit(a);
    }
    assert_eq!(order, vec![3, 2, 1, 0]);
    for (l, st) in states.iter().enumerate() {
        assert_eq!(st.mrs(g(0), g(2)), (l as i64 + 1) * 10);
        assert_eq!(st.commits, 1);
    }
}

// ===========================================================================
// 5. Shapes
// ===========================================================================

/// `Entry<i64>` is the unit of the pricing scan (`gt_core::design` §15.3), and
/// `Applied`/`Delta` must not silently start copying buffers around.
#[test]
fn the_tokens_are_borrows_not_copies() {
    assert_eq!(
        size_of::<Delta<'_, Directed, i64>>(),
        2 * size_of::<usize>(),
        "a Delta is two pointers: the buffer and the header"
    );
    // A `Transition` borrows the stack; the header travels with it by value.
    assert!(
        size_of::<Transition<'_, Directed, i64>>()
            >= size_of::<usize>() + size_of::<MoveHeader<i64>>()
    );
    // A zero delta is still an entry, and `W::ZERO` is what `insert` seeds it
    // with (`entries.hh:128`).
    assert_eq!(
        Entry::<i64> {
            r: g(0),
            s: g(1),
            delta: 0,
            me: None,
            mrs_before: 0,
        }
        .delta,
        <i64 as Weight>::ZERO
    );
}

// ===========================================================================
// 6. The negative guarantees, and the positive one
// ===========================================================================

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    // Defect #35: `apply_delta` re-entered double-counts every entry.
    t.compile_fail("tests/ui/u20_applied_is_not_reusable.rs");
    // A transition is not a value that can be duplicated and replayed.
    t.compile_fail("tests/ui/u20_transition_is_not_clonable.rs");
    // The ownership inversion itself: a priced `Delta` survives `&mut state`.
    t.pass("tests/ui/u20_delta_survives_a_mut_state.rs");
}
