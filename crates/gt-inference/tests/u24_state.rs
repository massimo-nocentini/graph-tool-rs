//! U24 -- the block state, driven from *outside* the crate.
//!
//! The acceptance tests for `Aggregates`, `BlockState` and the eight-thread
//! `commit_shared` stress live beside the code they exercise, in
//! `src/blockmodel/state.rs`, because `blockmodel/mod.rs` -- U22's file --
//! re-exports only `BlockView`, `BlockCommit`, `BlockCommitShared` and
//! `GroupLocks`, and `mod state;` is private. `Aggregates` and `BlockState`
//! are therefore unreachable from here until that `pub use` grows two names.
//!
//! What *is* reachable is the contract itself, and that is what this file
//! pins: the three traits, `Stamp`/`StateId`, `GroupLocks` and the delta
//! lifecycle are together sufficient for someone outside `gt-inference` to
//! build a committing block state -- including the defect #36 guard, which is
//! the one an epoch cannot express.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;

use gt_core::dir::{Dir, Directed};
use gt_core::ids::VertexId;

use gt_inference::blockmodel::{BlockCommit, BlockCommitShared, BlockView, GroupLocks};
use gt_inference::delta::{
    Applied, Entry, MoveHeader, MoveKey, Receipt, Recording, Transition, Workspace,
};
use gt_inference::ids::{BEdge, Epoch, Group, Stamp, StateId};

fn g(i: u32) -> Group {
    Group::new(i).expect("group index in range")
}

/// A block state built entirely from `gt-inference`'s public surface.
///
/// Dense, like `EMat` (`blockmodel/emat.hh:44-70`): `mrs[r * b + s]`. The
/// counters are atomics so that the row locks are the only thing establishing
/// *order*, which is the property under test -- `partition.hh:78-84` has the
/// atomics and leaves the ordering to convention.
struct TinyState {
    id: StateId,
    epoch: AtomicU64,
    slots: usize,
    mrs: Vec<AtomicI64>,
    mrp: Vec<AtomicI64>,
    mrm: Vec<AtomicI64>,
    wr: Vec<AtomicI64>,
    occupied: Mutex<BTreeSet<Group>>,
    locks: GroupLocks,
}

/// What the state refused, and why.
#[derive(Debug, PartialEq, Eq)]
struct Refused {
    delta: Stamp,
    state: Stamp,
}

impl TinyState {
    fn new(slots: usize) -> Self {
        TinyState {
            id: StateId::fresh(),
            epoch: AtomicU64::new(0),
            slots,
            mrs: (0..slots * slots).map(|_| AtomicI64::new(0)).collect(),
            mrp: (0..slots).map(|_| AtomicI64::new(0)).collect(),
            mrm: (0..slots).map(|_| AtomicI64::new(0)).collect(),
            wr: (0..slots).map(|_| AtomicI64::new(0)).collect(),
            occupied: Mutex::new(BTreeSet::new()),
            locks: GroupLocks::new(slots),
        }
    }

    fn addr(&self, r: Group, s: Group) -> usize {
        r.index() * self.slots + s.index()
    }

    /// `update_rs` (`entries.hh:391-398`), directed arm.
    fn pair(&self, r: Group, s: Group, delta: i64) {
        self.mrs[self.addr(r, s)].fetch_add(delta, Ordering::Relaxed);
        self.mrp[r.index()].fetch_add(delta, Ordering::Relaxed);
        self.mrm[s.index()].fetch_add(delta, Ordering::Relaxed);
    }

    /// `occupy_group`/`vacate_group` (`state.hh:783`, `:795`) fused into the
    /// only writer of `_wr`.
    fn add_wr(&self, r: Group, dw: i64) {
        if dw == 0 {
            return;
        }
        // The occupancy set is the innermost lock, always.
        let mut occ = self.occupied.lock().unwrap();
        let before = self.wr[r.index()].load(Ordering::Relaxed);
        let after = before + dw;
        self.wr[r.index()].store(after, Ordering::Relaxed);
        if before == 0 && after != 0 {
            occ.insert(r);
        } else if before != 0 && after == 0 {
            occ.remove(&r);
        }
    }

    fn check(&self, delta: Stamp, identity_only: bool) -> Result<(), Refused> {
        let mine = self.stamp();
        let ok = if identity_only {
            delta.state == mine.state
        } else {
            delta == mine
        };
        if ok { Ok(()) } else { Err(Refused { delta, state: mine }) }
    }

    fn try_commit(&mut self, a: Applied<'_, Directed, i64>) -> Result<Receipt<i64>, Refused> {
        self.check(a.stamp(), false)?;
        let r = self.apply(&a);
        self.epoch.fetch_add(1, Ordering::Relaxed);
        Ok(r)
    }

    fn try_commit_shared(&self, a: Applied<'_, Directed, i64>) -> Result<Receipt<i64>, Refused> {
        self.check(a.stamp(), true)?;
        let hdr = *a.header();
        for e in a.entries() {
            self.locks.with_pair(e.r, e.s, || self.pair(e.r, e.s, e.delta));
        }
        if let Some(r) = hdr.r {
            self.add_wr(r, -hdr.dr);
        }
        if let Some(nr) = hdr.nr {
            self.add_wr(nr, hdr.dnr);
        }
        self.epoch.fetch_add(1, Ordering::Relaxed);
        Ok(Receipt {
            entries: a.entries().to_vec(),
            hdr,
            stamp: a.stamp(),
            level: a.level(),
        })
    }

    fn apply(&self, a: &Applied<'_, Directed, i64>) -> Receipt<i64> {
        let hdr = *a.header();
        for e in a.entries() {
            self.pair(e.r, e.s, e.delta);
        }
        if let Some(r) = hdr.r {
            self.add_wr(r, -hdr.dr);
        }
        if let Some(nr) = hdr.nr {
            self.add_wr(nr, hdr.dnr);
        }
        Receipt {
            entries: a.entries().to_vec(),
            hdr,
            stamp: a.stamp(),
            level: a.level(),
        }
    }

    fn image(&self) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
        let load = |v: &Vec<AtomicI64>| v.iter().map(|x| x.load(Ordering::Relaxed)).collect();
        (
            load(&self.mrs),
            load(&self.mrp),
            load(&self.mrm),
            load(&self.wr),
        )
    }
}

impl BlockView for TinyState {
    type D = Directed;
    type W = i64;

    fn stamp(&self) -> Stamp {
        Stamp {
            state: self.id,
            epoch: Epoch(self.epoch.load(Ordering::Relaxed)),
        }
    }
    fn group_of(&self, v: VertexId) -> Option<Group> {
        Group::new(v.index() as u32 % self.slots as u32)
    }
    fn n_groups(&self) -> usize {
        self.occupied.lock().unwrap().len()
    }
    fn find_me(&self, r: Group, s: Group) -> Option<BEdge> {
        let a = self.addr(r, s);
        (self.mrs[a].load(Ordering::Relaxed) != 0).then_some(BEdge(a as u32))
    }
    fn mrs(&self, e: BEdge) -> i64 {
        self.mrs[e.0 as usize].load(Ordering::Relaxed)
    }
    fn mrp(&self, r: Group) -> i64 {
        self.mrp[r.index()].load(Ordering::Relaxed)
    }
    fn mrm(&self, r: Group) -> i64 {
        self.mrm[r.index()].load(Ordering::Relaxed)
    }
    fn wr(&self, r: Group) -> i64 {
        self.wr[r.index()].load(Ordering::Relaxed)
    }
    fn move_prob(&self, _t: &Transition<'_, Directed, i64>, _v: VertexId, _c: f64) -> f64 {
        // `state.hh:1733`, the `w == 0` fall-through.
        let b = self.n_groups() as f64;
        if b == 0.0 { 0.0 } else { -b.ln() }
    }
}

impl BlockCommit for TinyState {
    fn commit(&mut self, a: Applied<'_, Directed, i64>) -> Receipt<i64> {
        self.try_commit(a).expect("stamp mismatch")
    }
}

impl BlockCommitShared for TinyState {
    fn commit_shared(&self, a: Applied<'_, Directed, i64>) -> Receipt<i64> {
        self.try_commit_shared(a).expect("stamp mismatch")
    }
}

/// Record and seal one move. The transition's lifetime is the workspace's, so
/// the caller keeps the right to take `&mut st` afterwards.
fn seal<'w, S: BlockView<D = Directed, W = i64>>(
    st: &S,
    ws: &'w mut Workspace<Directed, i64>,
    slots: usize,
    hdr: MoveHeader<i64>,
    pairs: &[(Group, Group, i64)],
) -> Transition<'w, Directed, i64> {
    let mut rec = Recording::new(&mut ws.stack, hdr, st.stamp());
    rec.level_mut(0).begin(
        MoveKey {
            from: hdr.r,
            to: hdr.nr,
        },
        slots,
    );
    let mut resolve = |r, s| st.resolve(r, s);
    for &(a, b, w) in pairs {
        rec.level_mut(0)
            .touch_dyn(a, b, w, &mut resolve)
            .expect("in plane");
    }
    rec.seal()
}

fn header(r: Group, nr: Group, dr: i64, dnr: i64) -> MoveHeader<i64> {
    MoveHeader {
        r: Some(r),
        nr: Some(nr),
        dr,
        dnr,
        ..MoveHeader::default()
    }
}

// ---------------------------------------------------------------------------

/// Defect #36, at the trait boundary: two states built one after the other
/// both sit at epoch 0, so only the [`StateId`] half of the [`Stamp`] can
/// reject the cross-commit.
#[test]
fn a_delta_cannot_be_committed_into_a_sibling_state_at_the_same_epoch() {
    let a = TinyState::new(8);
    let mut b = TinyState::new(8);

    assert_eq!(a.stamp().epoch, Epoch(0));
    assert_eq!(b.stamp().epoch, Epoch(0));
    assert_eq!(
        a.stamp().epoch,
        b.stamp().epoch,
        "the epochs are equal -- that is the point"
    );
    assert_ne!(a.stamp().state, b.stamp().state);

    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let hdr = header(g(0), g(1), 1, 1);
    let t = seal(&a, &mut ws, 8, hdr, &[(g(0), g(2), -3), (g(1), g(2), 3)]);
    let applied = t.into_levels().next().unwrap();

    let before = b.image();
    let refused = b
        .try_commit(applied)
        .expect_err("a delta stamped with `a` must not commit into `b`");
    assert_eq!(refused.delta.state, a.stamp().state);
    assert_eq!(refused.state.state, b.stamp().state);
    assert_eq!(refused.delta.epoch, refused.state.epoch);
    assert_eq!(b.image(), before, "a refused delta changes nothing");
    assert_eq!(b.stamp().epoch, Epoch(0), "and does not advance the epoch");
}

/// The same across the concurrent door.
#[test]
fn commit_shared_also_refuses_a_foreign_identity() {
    let a = TinyState::new(4);
    let b = TinyState::new(4);
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let hdr = header(g(0), g(1), 0, 0);
    let t = seal(&a, &mut ws, 4, hdr, &[(g(0), g(2), 1)]);
    assert!(b.try_commit_shared(t.into_levels().next().unwrap()).is_err());
}

/// A transition recorded against the *right* state commits, and the epoch
/// advances exactly once per level.
#[test]
fn a_matching_stamp_commits_and_advances_the_epoch() {
    let mut st = TinyState::new(8);
    st.add_wr(g(0), 3);
    assert_eq!(st.n_groups(), 1);

    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let hdr = header(g(0), g(1), 1, 1);
    let t = seal(&st, &mut ws, 8, hdr, &[(g(0), g(2), 4), (g(1), g(2), 6)]);

    // The whole point of the ownership inversion: a `Delta` is alive here and
    // `commit` still takes `&mut st` below.
    let d = t.level(0);
    assert_eq!(d.entries().len(), 2);
    let claimed: i64 = d.entries().iter().map(|e: &Entry<i64>| e.delta).sum();

    let receipt = st.commit(t.into_levels().next().unwrap());
    assert_eq!(receipt.level, 0);
    assert_eq!(claimed, 10);
    assert_eq!(st.stamp().epoch, Epoch(1));
    assert_eq!(st.mrs(st.find_me(g(0), g(2)).unwrap()), 4);
    assert_eq!(st.mrs(st.find_me(g(1), g(2)).unwrap()), 6);
    assert_eq!(st.mrm(g(2)), 10);
    assert_eq!((st.wr(g(0)), st.wr(g(1))), (2, 1));
    assert_eq!(st.n_groups(), 2, "the header occupied group 1");

    // Second commit: the epoch the first delta carried is now stale.
    let mut ws2 = Workspace::<Directed, i64>::with_levels(1);
    let t2 = seal(&st, &mut ws2, 8, header(g(1), g(0), 1, 1), &[(g(1), g(2), -6)]);
    st.commit(t2.into_levels().next().unwrap());
    assert_eq!(st.stamp().epoch, Epoch(2));
    // The move back took group 1's last unit of vertex weight, so
    // `vacate_group` (`state.hh:795`) fires and the occupancy count drops.
    assert_eq!((st.wr(g(0)), st.wr(g(1))), (3, 0));
    assert_eq!(st.n_groups(), 1);
    assert!(st.find_me(g(1), g(2)).is_none(), "zero weight is no block edge");
}

/// `n_groups()` is the occupancy set's length; emptying a group removes it.
#[test]
fn n_groups_tracks_occupancy_and_nothing_else() {
    let st = TinyState::new(6);
    assert_eq!(st.n_groups(), 0);
    st.add_wr(g(2), 5);
    st.add_wr(g(4), 1);
    assert_eq!(st.n_groups(), 2);
    st.add_wr(g(2), -5);
    assert_eq!(st.n_groups(), 1);
    st.add_wr(g(4), -1);
    assert_eq!(st.n_groups(), 0);
    // `move_prob` reads it, so the guard flows through (`state.hh:1630`).
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let t = seal(&st, &mut ws, 6, header(g(0), g(1), 0, 0), &[]);
    assert_eq!(st.move_prob(&t, VertexId::from_index(0), f64::INFINITY), 0.0);
    st.add_wr(g(3), 2);
    let p = st.move_prob(&t, VertexId::from_index(0), f64::INFINITY);
    assert_eq!(p, 0.0, "safelog(1) is zero (cache.hh:99-104)");
    st.add_wr(g(5), 2);
    let p = st.move_prob(&t, VertexId::from_index(0), f64::INFINITY);
    assert!((p - -(2f64).ln()).abs() < 1e-15);
}

/// Eight threads over `GroupLocks`, disjoint and overlapping pairs, against a
/// serial replay of the same load.
#[test]
fn eight_threads_of_commit_shared_match_a_serial_replay() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 12_500;
    const SLOTS: usize = 16;

    fn ops(t: usize) -> Vec<(Group, Group, Group)> {
        let a = g(2 * t as u32);
        let b = g(2 * t as u32 + 1);
        let hot = g(15);
        (0..PER_THREAD)
            .map(|i| if i % 2 == 0 { (a, b, a) } else { (a, hot, g(0)) })
            .collect()
    }

    fn run(t: usize, st: &TinyState) {
        let mut ws = Workspace::<Directed, i64>::with_levels(1);
        for (from, to, x) in ops(t) {
            let hdr = header(from, to, 0, 0);
            let tr = seal(st, &mut ws, SLOTS, hdr, &[(from, x, 1), (to, x, 2)]);
            let _ = st.commit_shared(tr.into_levels().next().unwrap());
        }
    }

    let serial = TinyState::new(SLOTS);
    for t in 0..THREADS {
        run(t, &serial);
    }

    let parallel = TinyState::new(SLOTS);
    std::thread::scope(|scope| {
        for t in 0..THREADS {
            let p = &parallel;
            scope.spawn(move || run(t, p));
        }
    });

    assert_eq!(parallel.image(), serial.image());
    assert_eq!(
        parallel.stamp().epoch,
        Epoch((THREADS * PER_THREAD) as u64)
    );
    let total: i64 = serial.image().0.iter().sum();
    assert_eq!(total, 3 * (THREADS * PER_THREAD) as i64);
}

/// `GroupLocks` is deadlock-free under the opposite pair order, and handles
/// the block-graph self-loop `_mrs[r][r]` by locking once
/// (`parallel_util.hh:247-256`).
#[test]
fn group_locks_are_ordered_and_diagonal_safe() {
    let locks = GroupLocks::new(8);
    assert_eq!(locks.with_pair(g(3), g(3), || 7), 7);
    let counter = AtomicI64::new(0);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..50_000 {
                locks.with_pair(g(1), g(6), || {
                    counter.fetch_add(1, Ordering::Relaxed);
                });
            }
        });
        scope.spawn(|| {
            for _ in 0..50_000 {
                locks.with_pair(g(6), g(1), || {
                    counter.fetch_add(1, Ordering::Relaxed);
                });
            }
        });
        scope.spawn(|| {
            for _ in 0..50_000 {
                locks.with_pair(g(1), g(1), || {
                    counter.fetch_add(1, Ordering::Relaxed);
                });
            }
        });
    });
    assert_eq!(counter.load(Ordering::Relaxed), 150_000);
}

/// The trait bounds themselves: `BlockCommitShared: BlockView + Sync`, so a
/// shared reference to a committing state crosses a thread boundary. This is
/// the capability three of the six source designs deleted (DESIGN.md D9).
#[test]
fn a_shared_committing_state_is_sync() {
    fn assert_sync<T: BlockCommitShared>(_: &T) {}
    fn assert_view<T: BlockView<D = Directed, W = i64>>(_: &T) {}
    let st = TinyState::new(2);
    assert_sync(&st);
    assert_view(&st);
    // `Dir` is a type-level constant here, not a runtime flag, so these two
    // are const-evaluated rather than checked at run time.
    const {
        assert!(<TinyState as BlockView>::D::DIRECTED);
        assert!(<TinyState as BlockView>::D::N_FIELDS == 4);
    }
}
