//! The block state: its read face, its write face, and its locks.

use std::collections::{BTreeSet, HashMap};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use gt_core::dir::Dir;
use gt_core::ids::VertexId;
use gt_core::par::{Pair, pair_mut};

use crate::delta::{Applied, EndImage, Entry, Receipt, Transition};
use crate::ids::{BEdge, Epoch, Group, Stamp, StateId, Weight};

/// Everything a reader needs. Shared borrows, so several coexist.
pub trait BlockView {
    /// Directedness.
    type D: Dir;
    /// Weight type.
    type W: Weight;

    /// Identity and revision.
    fn stamp(&self) -> Stamp;

    /// The group of a vertex, or `None` for a zero-weight vertex.
    fn group_of(&self, v: VertexId) -> Option<Group>;

    /// Number of groups currently occupied. Derived from the occupancy set,
    /// never stored: graph-tool keeps `_actual_B` in three places at once
    /// (`blockmodel/state.hh:2518`, `partition.hh:172`, and implicitly as
    /// `_occupied_groups[..].size()`).
    fn n_groups(&self) -> usize;

    /// The block-graph edge `(r, s)`, if it exists.
    fn find_me(&self, r: Group, s: Group) -> Option<BEdge>;
    /// Its weight.
    fn mrs(&self, e: BEdge) -> Self::W;
    /// Out-weight of a group.
    fn mrp(&self, r: Group) -> Self::W;
    /// In-weight of a group.
    fn mrm(&self, r: Group) -> Self::W;
    /// Vertex weight of a group.
    fn wr(&self, r: Group) -> Self::W;

    /// The three scalars for one endpoint, as recorded in a move header.
    #[inline]
    fn end_image(&self, r: Option<Group>) -> EndImage<Self::W> {
        match r {
            None => EndImage::default(),
            Some(r) => EndImage {
                mrp: self.mrp(r),
                mrm: self.mrm(r),
                wr: self.wr(r),
            },
        }
    }

    /// Intern a pair for the recorder: its block edge and current weight.
    #[inline]
    fn resolve(&self, r: Group, s: Group) -> (Option<BEdge>, Self::W) {
        match self.find_me(r, s) {
            Some(e) => (Some(e), self.mrs(e)),
            None => (None, Self::W::ZERO),
        }
    }

    /// The log-probability of the reverse move, synthesised from a recorded
    /// transition.
    ///
    /// This is `get_move_prob` (`state.hh:1628-1734`), the consumer that
    /// prices a state which does not yet exist. It takes `&self` and
    /// `&Transition` -- two shared borrows -- and therefore coexists with the
    /// state-free [`sparse_ds`](super::sparse_ds) and precedes the `&mut self`
    /// commit without any interior mutability.
    fn move_prob(&self, t: &Transition<'_, Self::D, Self::W>, v: VertexId, c: f64) -> f64;
}

/// The write face. Single-threaded.
pub trait BlockCommit: BlockView {
    /// Apply **one level** of a transition, returning its before-image.
    ///
    /// Takes [`Applied`] by value, and `Applied` is `!Clone`, so replaying a
    /// level twice is a move error. `apply_delta` (`entries.hh:429`) has no
    /// such notion and double-counts every entry if re-entered.
    ///
    /// Rejects a delta whose [`Stamp`] does not match -- both the state
    /// identity and the revision. A revision counter alone is not enough: two
    /// freshly built states both start at epoch 0.
    fn commit(&mut self, a: Applied<'_, Self::D, Self::W>) -> Receipt<Self::W>;
}

/// The write face, concurrently.
///
/// Three of the six source designs removed graph-tool's concurrent commit and
/// scored the removal as a win; it is a removed capability.
/// `blockmodel/state.hh:152` builds `_group_mutex`, `:343-348` takes ordered
/// pair row locks, `:451` sizes the scratch pool to `get_num_threads()`.
/// What Rust adds is the enforcement `partition.hh:78-84` lacks, where
/// `_count[r] += w` sits between two `#pragma omp atomic` statements and is
/// correct only if every caller holds `_group_mutex[r]` -- a convention, not a
/// type.
pub trait BlockCommitShared: BlockView + Sync {
    /// Apply one level under this state's row locks.
    fn commit_shared(&self, a: Applied<'_, Self::D, Self::W>) -> Receipt<Self::W>;
}

/// Per-group row locks, taken in `(min, max)` order.
///
/// The port of `_group_mutex` (`blockmodel/state.hh:152`) and
/// `do_lock<lock_t::shared>` (`:343-348`).
pub struct GroupLocks {
    locks: gt_core::par::RowLocks,
}

impl GroupLocks {
    /// One lock per group slot.
    pub fn new(n_groups: usize) -> Self {
        GroupLocks {
            locks: gt_core::par::RowLocks::new(n_groups),
        }
    }

    /// Run `f` holding the locks for both endpoints of a block pair.
    pub fn with_pair<R>(&self, r: Group, s: Group, f: impl FnOnce() -> R) -> R {
        self.locks.with_pair(r.index(), s.index(), f)
    }
}

// ---------------------------------------------------------------------------
// The aggregates themselves.
// ---------------------------------------------------------------------------

/// One group's row: its three scalars and the block pairs it owns.
///
/// `_mrp[r]`, `_mrm[r]`, `_wr[r]` (`blockmodel/state.hh:2500-2546`) plus the
/// slice of `_mrs` reachable from `r`. Keeping the pair weights *inside* the
/// row is what makes the row mutex mean something: `entries.hh:391-398` writes
/// `_mrs[me]`, `_mrp[r]` and `_mrm[s]` from one delta under
/// `do_lock<shared>(r, s)` (`state.hh:343-348`), and if the pair weight lived
/// in a third structure the lock would again be a convention rather than a
/// type.
#[derive(Clone, Debug)]
struct Row<W: Weight> {
    mrp: W,
    mrm: W,
    wr: W,
    /// `other endpoint -> weight`, for the pairs this row is the owner of.
    ///
    /// Sparse, like `EHash`; addressed densely, like `EMat`
    /// (`blockmodel/emat.hh:44-77`). A pair is present iff its weight is
    /// non-zero, which is `update_rs`'s `if (int64_t(mrs) == 0) remove_me`
    /// (`entries.hh:405-427`) stated as an invariant instead of as a step.
    out: HashMap<u32, W>,
}

impl<W: Weight> Default for Row<W> {
    fn default() -> Self {
        Row {
            mrp: W::ZERO,
            mrm: W::ZERO,
            wr: W::ZERO,
            out: HashMap::new(),
        }
    }
}

/// A copy of one group's aggregates.
///
/// For the audit, for `Debug` output, and for tests that need to compare two
/// histories that reached the same state by different routes.
#[derive(Clone, PartialEq, Debug)]
pub struct RowImage<W: Weight> {
    /// Out-weight.
    pub mrp: W,
    /// In-weight.
    pub mrm: W,
    /// Vertex weight.
    pub wr: W,
    /// The block pairs this row owns, `(other endpoint, weight)`, sorted by
    /// endpoint so that two images compare structurally.
    pub pairs: Vec<(u32, W)>,
}

/// Strictly below zero, written so that `NaN` is not "negative".
///
/// [`Weight`] is only `PartialOrd` -- `f64` is a weight -- so `!(w < ZERO)` is
/// both a clippy lint and the wrong shape: it reports `NaN` as non-negative by
/// accident rather than on purpose. `partial_cmp` says which it is.
#[inline]
fn is_negative<W: Weight>(w: W) -> bool {
    w.partial_cmp(&W::ZERO) == Some(std::cmp::Ordering::Less)
}

/// Acquire a row, ignoring poison.
///
/// Same argument as [`gt_core::par::RowLocks`]: a panicking commit must not
/// wedge two groups for the rest of the process, and the panic reaches the
/// caller by itself.
#[inline]
fn acquire<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The same, where the caller already holds `&mut`. No atomic operation.
#[inline]
fn owned<T>(m: &mut Mutex<T>) -> &mut T {
    m.get_mut().unwrap_or_else(PoisonError::into_inner)
}

/// The twelve hand-maintained aggregates, behind one mutator.
///
/// `blockmodel/state.hh:2500-2546` keeps `_mrs`, `_mrp`, `_mrm`, `_wr`, `_N`,
/// `_E`, `_actual_B`, `_ps`, `_deg_stats`, `_groups`, `_occupied_groups`,
/// `_empty_groups`, `_emat`, `_bclabel`, `_bpclabel` and `_egroups` consistent
/// **by hand**, in an order that is transiently inconsistent in *opposite*
/// directions: `remove_partition_node` (`:738`) calls `_ps.remove_item`
/// **before** `_wr[r] -= w`, while `add_partition_node` (`:759`) calls
/// `occupy_group` -- which reads `_wr[r]` -- **before** `_wr[r] += w`.
///
/// Here every derived quantity is written only by
/// [`BlockCommit::commit`], which is not re-entrant, so no window exists.
///
/// ## Three counters collapsed into one
///
/// graph-tool stores the occupied-group count three times: `_actual_B`
/// (`state.hh:2518`), `_occupied_groups[c]` as a set, and `_empty_groups` as
/// its complement, each updated by hand in `occupy_group`/`vacate_group`
/// (`:783`, `:795`). Here there is **one** set, and [`n_groups`](Self::n_groups)
/// is its length. The set is inserted into and removed from only by
/// [`add_wr`](Self::add_wr) -- the only writer of `wr` anywhere in this type --
/// so there is no second copy that can drift away from the first.
///
/// ## Lock order
///
/// Rows are taken in `(min, max)` index order, exactly as `GroupLocks` and
/// `do_lock` (`state.hh:343-348`) prescribe; the occupancy set, when it is
/// needed at all, is taken **last** and is never held while a row is acquired.
/// Both rules together make the shared mutator deadlock-free by construction.
pub struct Aggregates<D: Dir, W: Weight> {
    /// Number of block-graph slots, i.e. `num_vertices(_bg)`.
    slots: usize,
    rows: Box<[Mutex<Row<W>>]>,
    /// The occupancy set. `n_groups()` is its length; there is no `_actual_B`.
    occupied: Mutex<BTreeSet<Group>>,
    /// Scratch for [`apply_entries`](Self::apply_entries): per-slot
    /// `(d mrp, d mrm)`. Reused across calls, so the batched path allocates
    /// nothing in steady state.
    acc: Vec<(W, W)>,
    /// Which slots of `acc` are live. Duplicates are harmless: the flush
    /// zeroes each entry as it applies it, so a second visit adds zero.
    touched: Vec<u32>,
    _d: PhantomData<fn() -> D>,
}

impl<D: Dir, W: Weight> std::fmt::Debug for Aggregates<D, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Aggregates")
            .field("dir", &D::NAME)
            .field("slots", &self.slots)
            .field("n_groups", &self.n_groups())
            .finish_non_exhaustive()
    }
}

impl<D: Dir, W: Weight> Aggregates<D, W> {
    /// Empty aggregates over `slots` block-graph slots.
    ///
    /// `slots` is `num_vertices(_bg)` (`state.hh:1107`), an index bound, not
    /// the occupied-group count [`n_groups`](Self::n_groups) reports.
    ///
    /// # Panics
    ///
    /// If `slots * slots` would not fit a `u32`. Block-edge identity here is
    /// the dense address `owner * slots + other`, the same `_mat[r][s]`
    /// addressing `EMat` uses (`emat.hh:44-70`), and [`BEdge`] is a `u32`.
    pub fn new(slots: usize) -> Self {
        assert!(
            slots
                .checked_mul(slots)
                .is_some_and(|n| n <= u32::MAX as usize),
            "{slots} block-graph slots do not fit the dense `BEdge` address \
             space (emat.hh:44); the limit is 65535"
        );
        Aggregates {
            slots,
            rows: (0..slots).map(|_| Mutex::new(Row::default())).collect(),
            occupied: Mutex::new(BTreeSet::new()),
            acc: vec![(W::ZERO, W::ZERO); slots],
            touched: Vec::new(),
            _d: PhantomData,
        }
    }

    /// Number of block-graph slots.
    #[inline]
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Number of **occupied** groups: `_actual_B` (`state.hh:2518`), derived.
    #[inline]
    pub fn n_groups(&self) -> usize {
        acquire(&self.occupied).len()
    }

    /// The occupied groups, in index order.
    pub fn occupied(&self) -> Vec<Group> {
        acquire(&self.occupied).iter().copied().collect()
    }

    /// Which row owns the pair `(r, s)`, and under which key.
    ///
    /// Directed: row `r`, key `s`. Undirected: `EMat::put_me` writes the same
    /// edge descriptor to `_mat[r][s]` **and** `_mat[s][r]`
    /// (`emat.hh:72-77`), so the pair has exactly one home; the lower index is
    /// it. Either way the owner is one of the two endpoints, so a caller that
    /// already holds both row locks holds the owner's.
    #[inline]
    fn owner(r: Group, s: Group) -> (usize, u32) {
        let (a, b) = (r.index(), s.index());
        if D::DIRECTED || a <= b {
            (a, b as u32)
        } else {
            (b, a as u32)
        }
    }

    /// The dense block-edge address, `_mat[r][s]`.
    #[inline]
    fn address(&self, owner: usize, other: u32) -> BEdge {
        BEdge((owner * self.slots + other as usize) as u32)
    }

    /// Decode one.
    #[inline]
    fn decode(&self, e: BEdge) -> (usize, u32) {
        let a = e.0 as usize;
        (a / self.slots, (a % self.slots) as u32)
    }

    /// The block-graph edge `(r, s)`, if it exists.
    ///
    /// `_emat.get_me(r, s)` (`emat.hh:66-70`) with `null_edge` spelled `None`.
    pub fn find_me(&self, r: Group, s: Group) -> Option<BEdge> {
        let (o, other) = Self::owner(r, s);
        acquire(&self.rows[o])
            .out
            .contains_key(&other)
            .then(|| self.address(o, other))
    }

    /// The weight of a block-graph edge.
    pub fn mrs(&self, e: BEdge) -> W {
        let (o, other) = self.decode(e);
        acquire(&self.rows[o])
            .out
            .get(&other)
            .copied()
            .unwrap_or(W::ZERO)
    }

    /// The weight of the pair `(r, s)`, zero when there is no block edge.
    pub fn pair(&self, r: Group, s: Group) -> W {
        let (o, other) = Self::owner(r, s);
        acquire(&self.rows[o])
            .out
            .get(&other)
            .copied()
            .unwrap_or(W::ZERO)
    }

    /// `_mrp[r]`.
    pub fn mrp(&self, r: Group) -> W {
        acquire(&self.rows[r.index()]).mrp
    }
    /// `_mrm[r]`.
    pub fn mrm(&self, r: Group) -> W {
        acquire(&self.rows[r.index()]).mrm
    }
    /// `_wr[r]`.
    pub fn wr(&self, r: Group) -> W {
        acquire(&self.rows[r.index()]).wr
    }

    /// A copy of every row, for audits and differential tests.
    pub fn image(&self) -> Vec<RowImage<W>> {
        self.rows
            .iter()
            .map(|m| {
                let row = acquire(m);
                let mut pairs: Vec<(u32, W)> = row.out.iter().map(|(&k, &v)| (k, v)).collect();
                pairs.sort_by_key(|p| p.0);
                RowImage {
                    mrp: row.mrp,
                    mrm: row.mrm,
                    wr: row.wr,
                    pairs,
                }
            })
            .collect()
    }

    // -- the per-pair mutator ------------------------------------------------

    /// `_mrs[me] += delta`, creating or removing the block edge.
    ///
    /// `update_rs` (`entries.hh:350-427`) minus the emat bookkeeping, which is
    /// the map key itself: the `boost::add_edge` + `put_me` arm (`:378-384`)
    /// is `entry().or_insert`, and the `remove_me` + `remove_edge` arm
    /// (`:405-427`) is `remove` on a zero result. The C++ needs a re-check
    /// under an upgraded lock there because two threads can race the removal;
    /// here the owner row's mutex is already held by both of them.
    #[inline]
    fn apply_pair(row: &mut Row<W>, other: u32, delta: W) {
        let w = match row.out.entry(other) {
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let w = *o.get() + delta;
                *o.get_mut() = w;
                w
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(delta);
                delta
            }
        };
        debug_assert!(
            !is_negative(w),
            "block-pair weight went negative ({w:?}); entries.hh:401 asserts \
             `_mrs[me] >= 0`"
        );
        if w == W::ZERO {
            row.out.remove(&other);
        }
    }

    /// Apply one changed block pair. The single per-pair mutator.
    ///
    /// `update_rs` (`entries.hh:350`) in full: the pair weight, then
    /// `_mrp[r]`, then `_mrm[s]` when directed and `_mrp[s]` when not
    /// (`:393-398`).
    ///
    /// The `r == s` diagonal is the case that has to be written out. For an
    /// undirected model `atomic_inc(_mrp[r], delta)` followed by
    /// `atomic_inc(_mrp[s], delta)` with `r == s` adds the delta **twice** --
    /// an undirected self-loop contributes two to the group's degree -- which
    /// a naive `Pair::Same` arm that adds once would silently halve.
    ///
    /// # Panics
    ///
    /// If `r` or `s` is not a slot, matching `_mrp[r]`'s unchecked indexing.
    pub fn apply_entry(&mut self, r: Group, s: Group, delta: W) {
        // `entries.hh:354-355`: a zero delta creates no block edge and touches
        // no counter. Skipping it here is what makes `apply_entries` and
        // repeated `apply_entry` agree entry for entry.
        if delta == W::ZERO {
            return;
        }
        // Before any mutation: a half-applied entry is worse than a panic,
        // and `pair_mut` only reports the range failure after `apply_pair`
        // would already have created the block edge.
        assert!(
            r.index() < self.slots && s.index() < self.slots,
            "block pair ({r:?}, {s:?}) is outside the {} block-graph slots",
            self.slots
        );
        let (o, other) = Self::owner(r, s);
        Self::apply_pair(owned(&mut self.rows[o]), other, delta);

        match pair_mut(&mut self.rows, r.index(), s.index()) {
            Some(Pair::Two(pr, ps)) => {
                let (rr, rs) = (owned(pr), owned(ps));
                rr.mrp = rr.mrp + delta;
                if D::DIRECTED {
                    rs.mrm = rs.mrm + delta;
                } else {
                    rs.mrp = rs.mrp + delta;
                }
                debug_assert!(!is_negative(rr.mrp), "entries.hh:402");
            }
            Some(Pair::Same(p)) => {
                let row = owned(p);
                // Both increments land on the same row.
                row.mrp = row.mrp + delta;
                if D::DIRECTED {
                    row.mrm = row.mrm + delta;
                } else {
                    row.mrp = row.mrp + delta;
                }
                debug_assert!(!is_negative(row.mrp), "entries.hh:402");
            }
            None => panic!(
                "block pair ({:?}, {:?}) is outside the {} block-graph slots",
                r, s, self.slots
            ),
        }
    }

    /// Apply a whole level's entries at once.
    ///
    /// Preferred over repeated [`apply_entry`](Self::apply_entry) for a
    /// high-degree move: every entry shares `r` or `nr`, so the endpoint
    /// scalars become one accumulate rather than one read-modify-write per
    /// entry.
    ///
    /// The pair weights are applied in input order, exactly as the loop would;
    /// only the scalar accumulation is reassociated. For an integer `W` -- the
    /// `int64_t` graph-tool actually instantiates -- the two are therefore
    /// bit-identical. For `W = f64` they can differ in the last place, which
    /// is the price of the reassociation and is stated rather than hidden.
    ///
    /// Allocates nothing: the accumulator and its touch list live in `self`
    /// and are reused.
    pub fn apply_entries(&mut self, entries: &[Entry<W>]) {
        let Aggregates {
            slots,
            rows,
            acc,
            touched,
            ..
        } = self;

        #[inline]
        fn bump<W: Weight>(
            acc: &mut [(W, W)],
            touched: &mut Vec<u32>,
            i: usize,
            d: W,
            into_mrm: bool,
        ) {
            let cell = &mut acc[i];
            if *cell == (W::ZERO, W::ZERO) {
                touched.push(i as u32);
            }
            if into_mrm {
                cell.1 = cell.1 + d;
            } else {
                cell.0 = cell.0 + d;
            }
        }

        for e in entries {
            if e.delta == W::ZERO {
                continue;
            }
            let (ri, si) = (e.r.index(), e.s.index());
            assert!(
                ri < *slots && si < *slots,
                "block pair ({:?}, {:?}) is outside the {slots} block-graph slots",
                e.r,
                e.s
            );
            let (o, other) = Self::owner(e.r, e.s);
            Self::apply_pair(owned(&mut rows[o]), other, e.delta);
            bump(acc, touched, ri, e.delta, false);
            bump(acc, touched, si, e.delta, D::DIRECTED);
        }

        for &i in touched.iter() {
            let cell = std::mem::replace(&mut acc[i as usize], (W::ZERO, W::ZERO));
            let row = owned(&mut rows[i as usize]);
            row.mrp = row.mrp + cell.0;
            row.mrm = row.mrm + cell.1;
            debug_assert!(!is_negative(row.mrp), "entries.hh:402");
        }
        touched.clear();
    }

    /// [`apply_entry`](Self::apply_entry) under this state's row locks.
    ///
    /// The two rows are taken in `(min, max)` order -- `do_lock` /
    /// `do_ulock_pair` (`state.hh:343-348`, `parallel_util.hh:247-256`) -- and
    /// `r == s` takes **one** lock, because `std::sync::Mutex` is not
    /// re-entrant.
    ///
    /// The owner of the pair weight is one of the two endpoints
    /// (the private `owner` rule), so no third lock is needed and the write is
    /// reachable only through a guard. That is the enforcement
    /// `partition.hh:78-84` lacks.
    pub fn apply_entry_shared(&self, r: Group, s: Group, delta: W) {
        if delta == W::ZERO {
            return;
        }
        let (ri, si) = (r.index(), s.index());
        assert!(
            ri < self.slots && si < self.slots,
            "block pair ({r:?}, {s:?}) is outside the {} block-graph slots",
            self.slots
        );
        let (o, other) = Self::owner(r, s);

        if ri == si {
            let mut g = acquire(&self.rows[ri]);
            Self::apply_pair(&mut g, other, delta);
            g.mrp = g.mrp + delta;
            if D::DIRECTED {
                g.mrm = g.mrm + delta;
            } else {
                g.mrp = g.mrp + delta;
            }
            return;
        }

        let (lo, hi) = if ri < si { (ri, si) } else { (si, ri) };
        let mut glo = acquire(&self.rows[lo]);
        let mut ghi = acquire(&self.rows[hi]);
        Self::apply_pair(if o == lo { &mut glo } else { &mut ghi }, other, delta);
        let (gr, gs) = if ri == lo {
            (&mut glo, &mut ghi)
        } else {
            (&mut ghi, &mut glo)
        };
        gr.mrp = gr.mrp + delta;
        if D::DIRECTED {
            gs.mrm = gs.mrm + delta;
        } else {
            gs.mrp = gs.mrp + delta;
        }
    }

    // -- vertex weight and occupancy ----------------------------------------

    /// `_wr[r] += dw`, with the occupancy set kept in step.
    ///
    /// The **only** writer of `wr`, which is what makes `n_groups()` unable to
    /// drift. It also fuses `occupy_group`/`vacate_group` (`state.hh:783-812`)
    /// into the weight update, removing the two windows the C++ leaves open in
    /// opposite directions (`:738` removes from `_ps` before decrementing
    /// `_wr`, `:759` reads `_wr` in `occupy_group` before incrementing it).
    pub fn add_wr(&mut self, r: Group, dw: W) {
        let row = owned(&mut self.rows[r.index()]);
        let before = row.wr;
        let after = before + dw;
        row.wr = after;
        debug_assert!(
            !is_negative(after),
            "group {r:?} vertex weight went negative ({after:?})"
        );
        if (before == W::ZERO) != (after == W::ZERO) {
            let occ = owned(&mut self.occupied);
            if after == W::ZERO {
                occ.remove(&r);
            } else {
                occ.insert(r);
            }
        }
    }

    /// The same under this state's locks.
    ///
    /// The occupancy set is taken **while the row lock is held**, so two
    /// threads driving the same group across zero cannot land their occupancy
    /// updates out of order. It is always the innermost lock, so it cannot
    /// participate in a cycle.
    pub fn add_wr_shared(&self, r: Group, dw: W) {
        let mut row = acquire(&self.rows[r.index()]);
        let before = row.wr;
        let after = before + dw;
        row.wr = after;
        debug_assert!(
            !is_negative(after),
            "group {r:?} vertex weight went negative ({after:?})"
        );
        if (before == W::ZERO) != (after == W::ZERO) {
            let mut occ = acquire(&self.occupied);
            if after == W::ZERO {
                occ.remove(&r);
            } else {
                occ.insert(r);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The stamp guard.
// ---------------------------------------------------------------------------

/// A delta offered to the wrong state, or to the wrong revision of it.
///
/// Defect #36. The identity half is the one an epoch cannot cover: two freshly
/// built states both sit at epoch 0, so without [`StateId`] a transition
/// recorded against one commits silently into the other.
#[derive(Clone, Copy, PartialEq, Eq, Debug, thiserror::Error)]
#[error(
    "stamp mismatch: delta carries state #{delta_state}/epoch {delta_epoch}, \
     state is #{state_state}/epoch {state_epoch}"
)]
pub struct StampMismatch {
    /// Identity carried by the delta.
    pub delta_state: u64,
    /// Revision carried by the delta.
    pub delta_epoch: u64,
    /// The state's identity.
    pub state_state: u64,
    /// The state's revision.
    pub state_epoch: u64,
}

impl StampMismatch {
    fn new(delta: Stamp, state: Stamp) -> Self {
        StampMismatch {
            delta_state: delta.state.get(),
            delta_epoch: delta.epoch.0,
            state_state: state.state.get(),
            state_epoch: state.epoch.0,
        }
    }
}

// ---------------------------------------------------------------------------
// A concrete state.
// ---------------------------------------------------------------------------

/// A block state over a fixed number of block-graph slots.
///
/// The concrete [`BlockView`] + [`BlockCommit`] + [`BlockCommitShared`]
/// implementation: `BlockState<D, W>` with `_b`, `_vweight` and the
/// [`Aggregates`]. It is the portion of `blockmodel/state.hh`'s state that the
/// reified transition actually reads and writes; the graph itself is never a
/// member here, because [`record`](super::record) takes it as an argument.
///
/// ## What it deliberately does not carry
///
/// `_egroups`, `_deg_stats`, `_ps`, `_bclabel`/`_bpclabel` and the coupled
/// state chain. Each is a separate hand-maintained structure in the C++ and a
/// separate unit here; none of them is read by pricing or by the delta.
pub struct BlockState<D: Dir, W: Weight> {
    id: StateId,
    /// Bumped by every commit. Atomic because [`commit_shared`] takes `&self`.
    ///
    /// [`commit_shared`]: BlockCommitShared::commit_shared
    epoch: AtomicU64,
    agg: Aggregates<D, W>,
    /// `_b`: the group of each vertex, `None` for a zero-weight vertex
    /// (`state.hh:126-131` sets `_b[v] = _null_group` there).
    b: Vec<Option<Group>>,
    /// `_vweight`.
    vweight: Vec<W>,
}

impl<D: Dir, W: Weight> std::fmt::Debug for BlockState<D, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockState")
            .field("id", &self.id.get())
            .field("epoch", &self.epoch.load(Ordering::Relaxed))
            .field("agg", &self.agg)
            .finish_non_exhaustive()
    }
}

impl<D: Dir, W: Weight> BlockState<D, W> {
    /// A fresh state with `slots` block-graph slots and `n` vertices, every
    /// vertex unassigned.
    ///
    /// The identity is minted here, so two calls never produce states a delta
    /// can be moved between.
    pub fn new(slots: usize, n: usize) -> Self {
        BlockState {
            id: StateId::fresh(),
            epoch: AtomicU64::new(0),
            agg: Aggregates::new(slots),
            b: vec![None; n],
            vweight: vec![W::ZERO; n],
        }
    }

    /// The aggregates.
    #[inline]
    pub fn aggregates(&self) -> &Aggregates<D, W> {
        &self.agg
    }
    /// The aggregates, mutably. Bypasses the stamp guard; for construction.
    #[inline]
    pub fn aggregates_mut(&mut self) -> &mut Aggregates<D, W> {
        &mut self.agg
    }

    /// Number of block-graph slots.
    #[inline]
    pub fn slots(&self) -> usize {
        self.agg.slots()
    }

    /// `_vweight[v]`.
    #[inline]
    pub fn vweight(&self, v: VertexId) -> W {
        self.vweight.get(v.index()).copied().unwrap_or(W::ZERO)
    }

    /// Put vertex `v`, of weight `w`, into group `r`.
    ///
    /// `add_partition_node` (`state.hh:759-777`) minus `_ps`, `_deg_stats` and
    /// `_groups`. If `v` is already placed it is removed first, so this is
    /// `remove_partition_node` + `add_partition_node`, i.e. `move_vertex`.
    pub fn assign(&mut self, v: VertexId, r: Group, w: W) {
        self.unassign(v);
        self.b[v.index()] = Some(r);
        self.vweight[v.index()] = w;
        self.agg.add_wr(r, w);
    }

    /// Write `_b[v] = r` and nothing else. `add_partition_node`'s last line
    /// (`state.hh:779`), separated from the `_wr` half that precedes it.
    ///
    /// # Why this exists
    ///
    /// graph-tool's `move_vertex` does both halves of a move in one place:
    /// `add_partition_node` (`state.hh:759-780`) writes `_wr[r] += w` *and*
    /// `_b[v] = r`. This port splits those two writes across two owners --
    /// the vertex weight rides the delta as `MoveHeader::dr`/`dnr` and lands
    /// in [`Self::apply_level`], while the partition entry has nowhere to
    /// ride, because [`MoveHeader`] carries the two *groups* and no
    /// [`VertexId`]. A commit therefore cannot write `_b` even in principle.
    ///
    /// Neither can the caller use [`Self::assign`] afterwards: that is
    /// `remove_partition_node` + `add_partition_node`, so it moves the vertex
    /// weight a second time and `_wr` ends up double-counting the move the
    /// delta already applied.
    ///
    /// So after every `commit` of a single-vertex move, the caller owes the
    /// state this one call. Until it is made, `group_of(v)` reports the *old*
    /// group while the aggregates report the new one, and the next
    /// [`record`](crate::blockmodel::record) reads that stale `_b` at
    /// `entries.hh:250` (`auto s = b[u]`) and produces a delta that removes
    /// weight from a pair that no longer holds it.
    ///
    /// `_vweight[v]` is deliberately untouched: the moved vertex weighs what
    /// it weighed, and it is `dr == dnr == _vweight[v]` that the commit moved.
    ///
    /// # Panics
    ///
    /// If `v` has never been [`assign`](Self::assign)ed. A vertex with no
    /// group has no weight either, so reseating it would leave `_wr` short by
    /// its weight for ever; that case is [`assign`](Self::assign)'s.
    pub fn reseat(&mut self, v: VertexId, r: Group) {
        let slot = self
            .b
            .get_mut(v.index())
            .expect("the vertex index is within the partition");
        assert!(
            slot.is_some(),
            "vertex {} has no group; use `assign` to place it, which also \
             moves its weight into `_wr` (state.hh:759-780)",
            v.index()
        );
        *slot = Some(r);
    }

    /// Take vertex `v` out of its group. `remove_partition_node`
    /// (`state.hh:738-757`).
    pub fn unassign(&mut self, v: VertexId) {
        if let Some(r) = self.b[v.index()].take() {
            let w = std::mem::replace(&mut self.vweight[v.index()], W::ZERO);
            self.agg.add_wr(r, -w);
        }
    }

    /// The weight of the pair `(r, s)`, zero when there is no block edge.
    ///
    /// `find_me` + `mrs` in one step, without the intermediate [`BEdge`].
    #[inline]
    pub fn pair_weight(&self, r: Group, s: Group) -> W {
        self.agg.pair(r, s)
    }

    /// Seed a block-pair weight directly. For building a state to move from.
    pub fn seed_pair(&mut self, r: Group, s: Group, w: W) {
        self.agg.apply_entry(r, s, w);
    }

    /// Apply one level, returning the before-image, or reject the delta.
    ///
    /// The fallible form of [`BlockCommit::commit`], which is this plus
    /// `unwrap`. Both halves of the [`Stamp`] are checked: identity **and**
    /// revision.
    pub fn try_commit(&mut self, a: Applied<'_, D, W>) -> Result<Receipt<W>, StampMismatch> {
        let mine = self.stamp();
        let theirs = a.stamp();
        if theirs != mine {
            return Err(StampMismatch::new(theirs, mine));
        }
        let receipt = self.apply_level(&a);
        // Only now: a rejected delta must not advance the revision, or the
        // caller's retry would be rejected for a second, different reason.
        self.epoch.store(mine.epoch.0 + 1, Ordering::Relaxed);
        Ok(receipt)
    }

    /// The concurrent form.
    ///
    /// Checks the **identity** half of the stamp only, and says so rather than
    /// pretending. Under concurrent commits the revision half is not a
    /// property any recorder can satisfy: every other thread's commit
    /// invalidates the epoch a delta was recorded against, so enforcing it
    /// would reject every delta but the first. graph-tool has no epoch at all
    /// here (`entries.hh:429` checks nothing); this keeps the half that closes
    /// defect #36 and drops the half that cannot hold.
    ///
    /// The epoch is still advanced, so a stale delta offered to the
    /// single-threaded [`try_commit`](Self::try_commit) afterwards is still
    /// rejected.
    pub fn try_commit_shared(&self, a: Applied<'_, D, W>) -> Result<Receipt<W>, StampMismatch> {
        let theirs = a.stamp();
        if theirs.state != self.id {
            return Err(StampMismatch::new(theirs, self.stamp()));
        }
        let hdr = *a.header();
        for e in a.entries() {
            self.agg.apply_entry_shared(e.r, e.s, e.delta);
        }
        // `_wr` is the header's business, not the entries': `apply_delta`
        // moves edge weight, `add/remove_partition_node` moves vertex weight
        // (`state.hh:738`, `:759`).
        if let Some(r) = hdr.r {
            self.agg.add_wr_shared(r, -hdr.dr);
        }
        if let Some(nr) = hdr.nr {
            self.agg.add_wr_shared(nr, hdr.dnr);
        }
        self.epoch.fetch_add(1, Ordering::Relaxed);
        Ok(Receipt {
            entries: a.entries().to_vec(),
            hdr,
            stamp: theirs,
            level: a.level(),
        })
    }

    /// The body shared by both commit paths, single-threaded.
    fn apply_level(&mut self, a: &Applied<'_, D, W>) -> Receipt<W> {
        let hdr = *a.header();
        let entries = a.entries().to_vec();
        self.agg.apply_entries(a.entries());
        if let Some(r) = hdr.r {
            self.agg.add_wr(r, -hdr.dr);
        }
        if let Some(nr) = hdr.nr {
            self.agg.add_wr(nr, hdr.dnr);
        }
        Receipt {
            entries,
            hdr,
            stamp: a.stamp(),
            level: a.level(),
        }
    }
}

/// `safelog` (`support/cache.hh:99-104`): `log(0)` is zero, not `-inf`.
#[inline]
fn safelog(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x.ln() }
}

impl<D: Dir, W: Weight> BlockView for BlockState<D, W> {
    type D = D;
    type W = W;

    #[inline]
    fn stamp(&self) -> Stamp {
        Stamp {
            state: self.id,
            epoch: Epoch(self.epoch.load(Ordering::Relaxed)),
        }
    }

    #[inline]
    fn group_of(&self, v: VertexId) -> Option<Group> {
        self.b.get(v.index()).copied().flatten()
    }

    #[inline]
    fn n_groups(&self) -> usize {
        self.agg.n_groups()
    }

    #[inline]
    fn find_me(&self, r: Group, s: Group) -> Option<BEdge> {
        self.agg.find_me(r, s)
    }
    #[inline]
    fn mrs(&self, e: BEdge) -> W {
        self.agg.mrs(e)
    }
    #[inline]
    fn mrp(&self, r: Group) -> W {
        self.agg.mrp(r)
    }
    #[inline]
    fn mrm(&self, r: Group) -> W {
        self.agg.mrm(r)
    }
    #[inline]
    fn wr(&self, r: Group) -> W {
        self.agg.wr(r)
    }

    /// `get_move_prob(v, nr, r, c, d = 0, reverse = true)`
    /// (`state.hh:1628-1734`), for the part of it a stateless signature can
    /// express.
    ///
    /// The three group-count guards are ported exactly: the reverse move into
    /// a group that `v` alone occupies has probability `log(d)` (`:1640-1641`),
    /// a vacated source group restores itself to the count (`:1643-1644`), and
    /// the diagonal is not a reverse move at all (`:1634`).
    ///
    /// **What it cannot do.** The neighbour sum at `:1662-1727` iterates
    /// `random_neighbors_range(v, _g)`, and this signature carries no graph --
    /// `move_prob(&self, &Transition, VertexId, f64)` has nowhere to put one.
    /// With `w == 0` the C++ falls through to `:1733`, `log(1 - d) -
    /// safelog_fast(B)`, which is also exactly what the `isinf(c)` arm at
    /// `:1654` returns; that shared expression is what is computed here, and
    /// it is the *correct* answer for an isolated `v` and for `c = inf`. A
    /// state that owns its graph overrides this method; `c` is accepted so
    /// that such an override needs no signature change.
    fn move_prob(&self, t: &Transition<'_, D, W>, v: VertexId, c: f64) -> f64 {
        let hdr = *t.level(0).header();
        // `pb = get_move_prob(v, nr, r, c, d, true)` (`spec.rs:128-129`): the
        // reverse move runs `nr -> r`.
        let (from, to) = (hdr.nr, hdr.r);
        let Some(s) = to else {
            // No target group: the forward move was a removal, so there is no
            // reverse proposal to price.
            return f64::NEG_INFINITY;
        };
        // `:1634`. `r == s` is not a move, so it is not a *reverse* move.
        let reverse = from != Some(s);
        let mut b = self.n_groups();

        if reverse {
            // `:1640-1641`: `d` is the probability of proposing a brand new
            // group and is zero for this signature, so `log(d)` is `-inf`.
            if self.agg.wr(s) == self.vweight(v) {
                return f64::NEG_INFINITY;
            }
            // `:1643-1644`: the source group is about to be vacated, so it is
            // not in the occupancy count and has to be added back.
            if from.is_none_or(|g| self.agg.wr(g) == W::ZERO) {
                b += 1;
            }
        } else if self.agg.wr(s) == W::ZERO {
            // `:1647-1648`.
            return f64::NEG_INFINITY;
        }

        // `c` selects between `:1654` and the neighbour sum; both reduce to
        // the same expression here (see the note above), so it is read and
        // deliberately not branched on.
        let _ = c;
        -safelog(b as f64)
    }
}

impl<D: Dir, W: Weight> BlockCommit for BlockState<D, W> {
    /// # Panics
    ///
    /// If the delta was recorded against a different state or a different
    /// revision. [`try_commit`](BlockState::try_commit) is the same check
    /// without the panic.
    fn commit(&mut self, a: Applied<'_, D, W>) -> Receipt<W> {
        match self.try_commit(a) {
            Ok(r) => r,
            Err(e) => panic!("{e}"),
        }
    }
}

impl<D: Dir, W: Weight> BlockCommitShared for BlockState<D, W> {
    /// # Panics
    ///
    /// If the delta was recorded against a different state. See
    /// [`try_commit_shared`](BlockState::try_commit_shared) for why the
    /// revision half is not enforced here.
    fn commit_shared(&self, a: Applied<'_, D, W>) -> Receipt<W> {
        match self.try_commit_shared(a) {
            Ok(r) => r,
            Err(e) => panic!("{e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{MoveHeader, MoveKey, Recording, Transition, Workspace};
    use gt_core::dir::{Directed, Undirected};

    fn g(i: u32) -> Group {
        Group::new(i).expect("group index in range")
    }
    fn v(i: usize) -> VertexId {
        VertexId::from_index(i)
    }

    /// A deterministic generator. `rand` is a dependency of this crate, but a
    /// three-line LCG makes every failure reproducible from the test name
    /// alone with no seed plumbing.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n
        }
    }

    /// Record and seal one move against `st`.
    ///
    /// The lifetime of the result is the **workspace's**, never `st`'s, which
    /// is the ownership inversion D10 rests on: every caller below takes
    /// `&mut st` while the sealed transition is still alive.
    fn seal_move<'w, S: BlockView>(
        st: &S,
        ws: &'w mut Workspace<S::D, S::W>,
        slots: usize,
        hdr: MoveHeader<S::W>,
        pairs: &[(Group, Group, S::W)],
    ) -> Transition<'w, S::D, S::W> {
        let stamp = st.stamp();
        let mut rec = Recording::new(&mut ws.stack, hdr, stamp);
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
                .expect("test pairs touch an endpoint of the move key");
        }
        rec.seal()
    }

    /// A header that moves no vertex weight: only the entries apply.
    fn plain_header<W: Weight>(r: Option<Group>, nr: Option<Group>) -> MoveHeader<W> {
        MoveHeader {
            r,
            nr,
            ..MoveHeader::default()
        }
    }

    // -- defect #36 ---------------------------------------------------------

    /// The acceptance test: two *fresh* states, so both sit at epoch 0, and a
    /// delta recorded against one must still be refused by the other.
    ///
    /// An epoch alone cannot catch this, which is the whole reason [`Stamp`]
    /// carries a [`StateId`].
    #[test]
    fn commit_rejects_a_foreign_state_at_an_equal_epoch() {
        let mut a = BlockState::<Directed, i64>::new(8, 4);
        let mut b = BlockState::<Directed, i64>::new(8, 4);

        // The precondition that makes the test mean something.
        assert_eq!(a.stamp().epoch, b.stamp().epoch, "both start at epoch 0");
        assert_ne!(a.stamp().state, b.stamp().state);

        a.seed_pair(g(0), g(2), 5);
        b.seed_pair(g(0), g(2), 5);

        let mut ws = Workspace::<Directed, i64>::with_levels(1);
        let hdr = plain_header(Some(g(0)), Some(g(1)));
        let t = seal_move(&a, &mut ws, 8, hdr, &[(g(0), g(2), -1), (g(1), g(2), 1)]);
        let applied = t.into_levels().next().expect("one level");

        let before = b.aggregates().image();
        let err = b
            .try_commit(applied)
            .expect_err("a delta stamped with another state must be rejected");
        assert_eq!(err.delta_state, a.stamp().state.get());
        assert_eq!(err.state_state, b.stamp().state.get());
        assert_eq!(err.delta_epoch, err.state_epoch, "the epochs matched");
        assert_eq!(
            b.aggregates().image(),
            before,
            "a rejected delta changes nothing"
        );
    }

    #[test]
    #[should_panic(expected = "stamp mismatch")]
    fn commit_panics_on_a_foreign_state() {
        let a = BlockState::<Directed, i64>::new(8, 4);
        let mut b = BlockState::<Directed, i64>::new(8, 4);
        let mut ws = Workspace::<Directed, i64>::with_levels(1);
        let hdr = plain_header::<i64>(Some(g(0)), Some(g(1)));
        let t = seal_move(&a, &mut ws, 8, hdr, &[(g(0), g(2), 0)]);
        let _ = b.commit(t.into_levels().next().unwrap());
    }

    #[test]
    #[should_panic(expected = "stamp mismatch")]
    fn commit_shared_panics_on_a_foreign_state() {
        let a = BlockState::<Directed, i64>::new(8, 4);
        let b = BlockState::<Directed, i64>::new(8, 4);
        let mut ws = Workspace::<Directed, i64>::with_levels(1);
        let hdr = plain_header::<i64>(Some(g(0)), Some(g(1)));
        let t = seal_move(&a, &mut ws, 8, hdr, &[(g(0), g(2), 0)]);
        let _ = b.commit_shared(t.into_levels().next().unwrap());
    }

    #[test]
    fn a_stale_epoch_is_rejected_and_a_rejection_does_not_advance_it() {
        let mut st = BlockState::<Directed, i64>::new(8, 4);
        st.seed_pair(g(0), g(2), 10);
        st.seed_pair(g(1), g(2), 10);

        let mut ws = Workspace::<Directed, i64>::with_levels(1);
        let hdr = plain_header(Some(g(0)), Some(g(1)));

        // Two deltas recorded against the *same* revision.
        let stale = {
            let t = seal_move(&st, &mut ws, 8, hdr, &[(g(0), g(2), -1), (g(1), g(2), 1)]);
            t.into_levels().next().unwrap().level()
        };
        assert_eq!(stale, 0);

        let mut ws1 = Workspace::<Directed, i64>::with_levels(1);
        let mut ws2 = Workspace::<Directed, i64>::with_levels(1);
        let t1 = seal_move(&st, &mut ws1, 8, hdr, &[(g(0), g(2), -1), (g(1), g(2), 1)]);
        let t2 = seal_move(&st, &mut ws2, 8, hdr, &[(g(0), g(2), -1), (g(1), g(2), 1)]);

        st.commit(t1.into_levels().next().unwrap());
        assert_eq!(st.stamp().epoch, Epoch(1));

        let epoch_before = st.stamp().epoch;
        let image_before = st.aggregates().image();
        let err = st
            .try_commit(t2.into_levels().next().unwrap())
            .expect_err("epoch 0 is stale once the state is at epoch 1");
        assert_eq!(err.delta_epoch, 0);
        assert_eq!(err.state_epoch, 1);
        assert_eq!(
            st.stamp().epoch,
            epoch_before,
            "a rejection is not a revision"
        );
        assert_eq!(st.aggregates().image(), image_before);
    }

    // -- the per-pair mutator ----------------------------------------------

    /// `entries.hh:354-355`.
    #[test]
    fn a_zero_delta_creates_nothing() {
        let mut a = Aggregates::<Directed, i64>::new(4);
        a.apply_entry(g(0), g(1), 0);
        assert!(a.find_me(g(0), g(1)).is_none());
        assert_eq!(a.mrp(g(0)), 0);
        a.apply_entries(&[Entry {
            r: g(0),
            s: g(1),
            delta: 0,
            me: None,
            mrs_before: 0,
        }]);
        assert!(a.find_me(g(0), g(1)).is_none());
        assert_eq!(a.mrp(g(0)), 0);
    }

    /// `entries.hh:393-398`, directed arm.
    #[test]
    fn directed_routes_the_target_into_mrm() {
        let mut a = Aggregates::<Directed, i64>::new(4);
        a.apply_entry(g(1), g(2), 3);
        assert_eq!((a.mrp(g(1)), a.mrm(g(1))), (3, 0));
        assert_eq!((a.mrp(g(2)), a.mrm(g(2))), (0, 3));
        assert_eq!(a.pair(g(1), g(2)), 3);
        // Directed pairs are not symmetric.
        assert_eq!(a.pair(g(2), g(1)), 0);
        assert!(a.find_me(g(2), g(1)).is_none());

        // The directed diagonal lands on both halves of the same row.
        a.apply_entry(g(3), g(3), 2);
        assert_eq!((a.mrp(g(3)), a.mrm(g(3))), (2, 2));
    }

    /// The `Pair::Same` case that a naive implementation halves.
    ///
    /// Undirected, `r == s`: `atomic_inc(_mrp[r], delta)` and
    /// `atomic_inc(_mrp[s], delta)` (`entries.hh:394`, `:398`) are two
    /// increments of the same counter, because an undirected self-loop
    /// contributes twice to a group's degree.
    #[test]
    fn an_undirected_self_pair_counts_twice() {
        let mut a = Aggregates::<Undirected, i64>::new(4);
        a.apply_entry(g(1), g(1), 3);
        assert_eq!(a.mrp(g(1)), 6, "an undirected self-loop is two half-edges");
        assert_eq!(a.mrm(g(1)), 0, "undirected states never write _mrm");
        assert_eq!(a.pair(g(1), g(1)), 3);

        let mut b = Aggregates::<Undirected, i64>::new(4);
        b.apply_entries(&[Entry {
            r: g(1),
            s: g(1),
            delta: 3,
            me: None,
            mrs_before: 0,
        }]);
        assert_eq!(a.image(), b.image(), "the batched path must agree");
    }

    /// `emat.hh:72-77`: undirected `put_me` writes `_mat[r][s]` *and*
    /// `_mat[s][r]`, so the pair has one identity whichever way round it is
    /// named.
    #[test]
    fn the_undirected_block_edge_is_symmetric() {
        let mut a = Aggregates::<Undirected, i64>::new(6);
        a.apply_entry(g(4), g(1), 7);
        assert_eq!(a.find_me(g(1), g(4)), a.find_me(g(4), g(1)));
        assert_eq!(a.pair(g(1), g(4)), 7);
        assert_eq!(a.pair(g(4), g(1)), 7);
        let e = a.find_me(g(4), g(1)).unwrap();
        assert_eq!(a.mrs(e), 7);
        // The two endpoints each gained the weight once.
        assert_eq!((a.mrp(g(4)), a.mrp(g(1))), (7, 7));
    }

    /// `entries.hh:405-427`: the block edge is removed when its weight
    /// reaches zero, so `get_me` goes back to the null edge.
    #[test]
    fn a_block_edge_disappears_at_zero() {
        let mut a = Aggregates::<Directed, i64>::new(4);
        a.apply_entry(g(0), g(1), 5);
        let e = a.find_me(g(0), g(1)).expect("created");
        assert_eq!(a.mrs(e), 5);
        a.apply_entry(g(0), g(1), -5);
        assert!(a.find_me(g(0), g(1)).is_none(), "zero weight is no edge");
        assert_eq!(a.pair(g(0), g(1)), 0);
        assert_eq!(a.mrs(e), 0, "the stale address still reads as zero");
        assert_eq!((a.mrp(g(0)), a.mrm(g(1))), (0, 0));
        // And it comes back.
        a.apply_entry(g(0), g(1), 2);
        assert_eq!(a.find_me(g(0), g(1)), Some(e));
    }

    // -- apply_entries == repeated apply_entry ------------------------------

    fn batched_equals_serial<D: Dir>(seed: u64) {
        let slots = 8u32;
        let mut one = Aggregates::<D, i64>::new(slots as usize);
        let mut many = Aggregates::<D, i64>::new(slots as usize);

        // Seed both identically and generously, so that the random negative
        // deltas below can never drive a pair weight through zero -- which is
        // a state the C++ asserts against (`entries.hh:401`) and this port
        // debug-asserts against too.
        for r in 0..slots {
            for s in 0..slots {
                one.apply_entry(g(r), g(s), 1000);
                many.apply_entry(g(r), g(s), 1000);
            }
        }
        assert_eq!(one.image(), many.image());

        let mut rng = Lcg::new(seed);
        let entries: Vec<Entry<i64>> = (0..400)
            .map(|_| Entry {
                r: g(rng.below(slots)),
                s: g(rng.below(slots)),
                delta: i64::from(rng.below(9)) - 4,
                me: None,
                mrs_before: 0,
            })
            .collect();

        for e in &entries {
            one.apply_entry(e.r, e.s, e.delta);
        }
        many.apply_entries(&entries);

        assert_eq!(
            one.image(),
            many.image(),
            "{}: apply_entries must equal repeated apply_entry",
            D::NAME
        );
        assert_eq!(one.n_groups(), many.n_groups());
    }

    #[test]
    fn apply_entries_equals_repeated_apply_entry_directed() {
        for seed in 0..16 {
            batched_equals_serial::<Directed>(0xC0FFEE + seed);
        }
    }

    #[test]
    fn apply_entries_equals_repeated_apply_entry_undirected() {
        for seed in 0..16 {
            batched_equals_serial::<Undirected>(0xBEEF + seed);
        }
    }

    /// The batched path reuses its accumulator, so a second call must not see
    /// the first call's residue.
    #[test]
    fn apply_entries_leaves_no_residue() {
        let mut a = Aggregates::<Directed, i64>::new(4);
        let e = |r: u32, s: u32, d: i64| Entry {
            r: g(r),
            s: g(s),
            delta: d,
            me: None,
            mrs_before: 0,
        };
        a.apply_entries(&[e(0, 1, 5), e(0, 2, 5)]);
        assert_eq!(a.mrp(g(0)), 10);
        a.apply_entries(&[e(3, 1, 1)]);
        assert_eq!(a.mrp(g(0)), 10, "group 0 must not move a second time");
        assert_eq!(a.mrp(g(3)), 1);
        assert_eq!(a.mrm(g(1)), 6);
    }

    // -- occupancy ----------------------------------------------------------

    /// `n_groups()` is the length of the occupancy set and nothing else:
    /// there is no `_actual_B` beside it, and no setter that can move one
    /// without the other.
    #[test]
    fn n_groups_is_the_occupancy_set() {
        let mut st = BlockState::<Directed, i64>::new(8, 4);
        assert_eq!(st.n_groups(), 0);

        st.assign(v(0), g(1), 1);
        assert_eq!(st.n_groups(), 1);
        st.assign(v(1), g(1), 1);
        assert_eq!(st.n_groups(), 1, "a second vertex does not occupy twice");
        st.assign(v(2), g(3), 2);
        assert_eq!(st.n_groups(), 2);
        assert_eq!(st.aggregates().occupied(), vec![g(1), g(3)]);

        st.unassign(v(0));
        assert_eq!(st.n_groups(), 2, "group 1 still holds vertex 1");
        st.unassign(v(1));
        assert_eq!(st.n_groups(), 1, "vacate_group (state.hh:795)");
        assert_eq!(st.aggregates().occupied(), vec![g(3)]);
        assert_eq!(st.wr(g(1)), 0);

        // `assign` on an already-placed vertex is `move_vertex`.
        st.assign(v(2), g(5), 2);
        assert_eq!(st.n_groups(), 1);
        assert_eq!(st.aggregates().occupied(), vec![g(5)]);
    }

    /// The same claim under a long random sequence, checked against a scan of
    /// the group weights. If a second copy of the count existed anywhere this
    /// is where it would drift.
    #[test]
    fn n_groups_never_drifts_from_the_weights() {
        let slots = 12u32;
        let n = 40usize;
        let mut st = BlockState::<Undirected, i64>::new(slots as usize, n);
        let mut rng = Lcg::new(0x5EED);

        for step in 0..4000 {
            let u = v(rng.below(n as u32) as usize);
            if rng.below(3) == 0 {
                st.unassign(u);
            } else {
                st.assign(u, g(rng.below(slots)), 1 + i64::from(rng.below(3)));
            }
            let scanned = st.aggregates().image().iter().filter(|r| r.wr != 0).count();
            assert_eq!(st.n_groups(), scanned, "step {step}");
        }
    }

    // -- commit -------------------------------------------------------------

    /// A commit applies the entries (edge weight) *and* the header (vertex
    /// weight), which are the two halves `apply_delta` (`entries.hh:429`) and
    /// `add/remove_partition_node` (`state.hh:738`, `:759`) do separately.
    #[test]
    fn commit_applies_entries_and_the_header() {
        let mut st = BlockState::<Directed, i64>::new(8, 4);
        st.assign(v(0), g(0), 1);
        st.assign(v(1), g(0), 1);
        st.seed_pair(g(0), g(2), 4);
        assert_eq!(st.n_groups(), 1);

        let mut ws = Workspace::<Directed, i64>::with_levels(1);
        let hdr = MoveHeader {
            r: Some(g(0)),
            nr: Some(g(1)),
            r_img: st.end_image(Some(g(0))),
            nr_img: st.end_image(Some(g(1))),
            dkin: 0,
            dkout: 2,
            dr: 1,
            dnr: 1,
        };
        let t = seal_move(&st, &mut ws, 8, hdr, &[(g(0), g(2), -2), (g(1), g(2), 2)]);
        let applied = t.into_levels().next().unwrap();
        assert_eq!(applied.entries().len(), 2);

        let receipt = st.commit(applied);
        assert_eq!(receipt.level, 0);
        assert_eq!(receipt.entries.len(), 2);
        assert_eq!(receipt.hdr.dkout, 2);

        assert_eq!(st.pair_weight(g(0), g(2)), 2);
        assert_eq!(st.pair_weight(g(1), g(2)), 2);
        assert_eq!(st.mrp(g(0)), 2);
        assert_eq!(st.mrp(g(1)), 2);
        assert_eq!(st.mrm(g(2)), 4);
        // The header moved one unit of vertex weight, occupying group 1.
        assert_eq!((st.wr(g(0)), st.wr(g(1))), (1, 1));
        assert_eq!(st.n_groups(), 2);
        assert_eq!(st.stamp().epoch, Epoch(1));

        // The receipt carries the before-image the audit needs.
        assert_eq!(receipt.hdr.r_img.wr, 2);
        assert_eq!(receipt.hdr.nr_img.wr, 0);
        let moved: i64 = receipt.entries.iter().map(|e| e.delta).sum();
        assert_eq!(moved, 0, "a pure move conserves edge weight");
    }

    /// The before-image the recorder interned must be what the state held.
    #[test]
    fn the_recorded_before_image_is_the_states() {
        let mut st = BlockState::<Directed, i64>::new(8, 4);
        st.seed_pair(g(0), g(2), 4);
        let mut ws = Workspace::<Directed, i64>::with_levels(1);
        let hdr = plain_header(Some(g(0)), Some(g(1)));
        let t = seal_move(&st, &mut ws, 8, hdr, &[(g(0), g(2), -1), (g(1), g(2), 1)]);
        for e in t.level(0).entries() {
            assert_eq!(e.mrs_before, st.pair_weight(e.r, e.s));
            assert_eq!(e.me, st.find_me(e.r, e.s));
        }
        let _ = st.commit(t.into_levels().next().unwrap());
    }

    // -- move_prob ----------------------------------------------------------

    /// The three group-count guards of `get_move_prob`
    /// (`state.hh:1634-1648`).
    #[test]
    fn move_prob_ports_the_group_count_guards() {
        let mut st = BlockState::<Directed, i64>::new(8, 4);
        st.assign(v(0), g(0), 1);
        st.assign(v(1), g(0), 1);
        st.assign(v(2), g(1), 1);
        assert_eq!(st.n_groups(), 2);

        let mut ws = Workspace::<Directed, i64>::with_levels(1);
        // Forward move 0 -> 1, so the reverse move is 1 -> 0.
        let hdr = plain_header::<i64>(Some(g(0)), Some(g(1)));
        let t = seal_move(&st, &mut ws, 8, hdr, &[]);
        // `r = nr = g(1)` is occupied and holds more than `v`'s own weight,
        // and `g(0)` is not empty, so `B` stays at 2.
        let p = st.move_prob(&t, v(0), f64::INFINITY);
        assert!((p - -(2f64).ln()).abs() < 1e-15, "{p}");

        // `:1640-1641`: the reverse target is occupied by `v` alone.
        let mut st2 = BlockState::<Directed, i64>::new(8, 4);
        st2.assign(v(0), g(0), 1);
        st2.assign(v(1), g(1), 1);
        let mut ws2 = Workspace::<Directed, i64>::with_levels(1);
        let t2 = seal_move(&st2, &mut ws2, 8, hdr, &[]);
        assert_eq!(st2.move_prob(&t2, v(1), f64::INFINITY), f64::NEG_INFINITY);

        // `:1643-1644`: the reverse move's *source* -- the forward move's
        // `nr` -- is empty, so it is missing from the occupancy count and has
        // to be added back before the count is used.
        let mut st3 = BlockState::<Directed, i64>::new(8, 4);
        st3.assign(v(0), g(2), 1);
        st3.assign(v(1), g(3), 1);
        assert_eq!(st3.n_groups(), 2);
        assert_eq!(st3.wr(g(1)), 0, "the reverse source is empty");
        let mut ws3 = Workspace::<Directed, i64>::with_levels(1);
        let t3 = seal_move(&st3, &mut ws3, 8, hdr, &[]);
        let p3 = st3.move_prob(&t3, v(0), f64::INFINITY);
        assert!(
            (p3 - -(3f64).ln()).abs() < 1e-15,
            "an empty reverse source rejoins B: {p3}"
        );
    }

    // -- the concurrent commit ---------------------------------------------

    /// One thread's share of the stress load.
    ///
    /// Half the moves run over a block pair private to the thread -- the
    /// disjoint case, where the row locks never contend -- and half run one
    /// endpoint into a single shared group, which is the overlapping case the
    /// `(min, max)` ordering exists for.
    fn stress_ops(thread: usize, n: usize) -> Vec<(Group, Group, Group)> {
        let a = g(2 * thread as u32);
        let b = g(2 * thread as u32 + 1);
        let hot = g(15);
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    (a, b, a)
                } else {
                    (a, hot, g(0))
                }
            })
            .collect()
    }

    /// 8 threads, 10^5 `commit_shared` calls, against a serial replay.
    ///
    /// The deltas are positive throughout: reordering across threads is what
    /// is under test, and a mixed-sign load would let a pair weight go
    /// transiently negative under one interleaving and not another, tripping
    /// `entries.hh:401`'s assertion for a reason that has nothing to do with
    /// the locks.
    #[test]
    fn commit_shared_stress_matches_a_serial_replay() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 12_500;
        let slots = 16usize;

        // A `fn` item, not a closure: a closure would be *moved* into the
        // first `spawn` and unavailable to the other seven.
        fn run(t: usize, st: &BlockState<Directed, i64>, slots: usize, per: usize) {
            let mut ws = Workspace::<Directed, i64>::with_levels(1);
            for (from, to, x) in stress_ops(t, per) {
                let hdr = plain_header::<i64>(Some(from), Some(to));
                let tr = seal_move(st, &mut ws, slots, hdr, &[(from, x, 1), (to, x, 2)]);
                let applied = tr.into_levels().next().unwrap();
                // Both sides go through the same door. The epoch has moved on
                // since `seal_move` read it, which is exactly the condition
                // `try_commit_shared` documents as unenforceable; what is
                // being compared here is the arithmetic under reordering.
                let _ = st.commit_shared(applied);
            }
        }

        let serial = BlockState::<Directed, i64>::new(slots, 1);
        for t in 0..THREADS {
            run(t, &serial, slots, PER_THREAD);
        }

        let parallel = BlockState::<Directed, i64>::new(slots, 1);
        std::thread::scope(|scope| {
            for t in 0..THREADS {
                let p = &parallel;
                scope.spawn(move || run(t, p, slots, PER_THREAD));
            }
        });

        assert_eq!(
            parallel.aggregates().image(),
            serial.aggregates().image(),
            "concurrent commits must land the same aggregates as a serial replay"
        );
        assert_eq!(
            parallel.stamp().epoch,
            Epoch((THREADS * PER_THREAD) as u64),
            "every commit advances the revision exactly once"
        );

        // And the totals are what the load actually asked for: each move adds
        // 1 to (from, x) and 2 to (to, x), over 10^5 moves.
        let total: i64 = serial
            .aggregates()
            .image()
            .iter()
            .flat_map(|r| r.pairs.iter().map(|p| p.1))
            .sum();
        assert_eq!(total, 3 * (THREADS * PER_THREAD) as i64);
    }

    /// The occupancy set under concurrency: `add_wr_shared` takes it while the
    /// row lock is held, so two threads driving one group across zero cannot
    /// land their updates out of order.
    #[test]
    fn concurrent_vertex_weight_keeps_the_occupancy_set_exact() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 2_000;
        let agg = Aggregates::<Undirected, i64>::new(4);
        std::thread::scope(|scope| {
            for t in 0..THREADS {
                let a = &agg;
                scope.spawn(move || {
                    for i in 0..PER_THREAD {
                        let r = g(((t + i) % 4) as u32);
                        a.add_wr_shared(r, 1);
                        a.add_wr_shared(r, -1);
                        a.add_wr_shared(r, 1);
                    }
                });
            }
        });
        for r in 0..4 {
            assert!(agg.wr(g(r)) > 0);
        }
        assert_eq!(agg.n_groups(), 4);
        let scanned = agg.image().iter().filter(|r| r.wr != 0).count();
        assert_eq!(agg.n_groups(), scanned);
    }

    /// Row locks are taken in `(min, max)` order, so the *opposite* pair order
    /// cannot deadlock against them. Ported from the reason `do_lock` sorts
    /// (`parallel_util.hh:84-86`).
    #[test]
    fn opposed_pair_orders_do_not_deadlock() {
        let agg = Aggregates::<Directed, i64>::new(4);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for _ in 0..20_000 {
                    agg.apply_entry_shared(g(0), g(3), 1);
                }
            });
            scope.spawn(|| {
                for _ in 0..20_000 {
                    agg.apply_entry_shared(g(3), g(0), 1);
                }
            });
        });
        assert_eq!(agg.pair(g(0), g(3)), 20_000);
        assert_eq!(agg.pair(g(3), g(0)), 20_000);
        // A writes `_mrp[0]`, B writes `_mrm[0]`: `entries.hh:394-396`
        // routes the source into `_mrp` and the target into `_mrm`.
        assert_eq!((agg.mrp(g(0)), agg.mrm(g(0))), (20_000, 20_000));
        assert_eq!((agg.mrp(g(3)), agg.mrm(g(3))), (20_000, 20_000));
    }

    /// The shared diagonal locks once. A second acquisition of the same
    /// `std::sync::Mutex` would hang, which is why the `r == s` arm exists.
    #[test]
    fn the_shared_diagonal_locks_once() {
        let agg = Aggregates::<Undirected, i64>::new(4);
        agg.apply_entry_shared(g(2), g(2), 5);
        assert_eq!(agg.pair(g(2), g(2)), 5);
        assert_eq!(agg.mrp(g(2)), 10);
    }

    #[test]
    #[should_panic(expected = "block pair")]
    fn a_pair_outside_the_slots_panics() {
        let mut agg = Aggregates::<Directed, i64>::new(4);
        agg.apply_entry(g(0), g(9), 1);
    }

    #[test]
    #[should_panic(expected = "dense `BEdge` address")]
    fn too_many_slots_is_refused_rather_than_wrapped() {
        let _ = Aggregates::<Directed, i64>::new(70_000);
    }
}
