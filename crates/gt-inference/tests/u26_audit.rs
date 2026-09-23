//! U26 -- the always-on audit, driven from *outside* the crate.
//!
//! `audit_absolute` is not exercised here: it is `#[cfg(feature =
//! "audit-full")]` and `blockmodel/mod.rs` -- U22's file, which this unit does
//! not own -- lists only `AuditError`, `audit_commit` and `audit_price` in its
//! `pub use`. Its acceptance test (the 10 000-move sweep) therefore lives
//! beside the code, in `src/blockmodel/audit.rs`, where the private module is
//! in scope.
//!
//! Everything else is here, because "the audit composes with a state someone
//! else built" is precisely the claim worth pinning from outside.

use std::cell::Cell;
use std::collections::BTreeMap;

use gt_core::dir::{Dir, Directed, Undirected};
use gt_core::ids::VertexId;

use gt_inference::blockmodel::{
    AuditError, BlockCommit, BlockState, BlockView, Cache, EntropyParams, audit_commit,
    audit_price, sparse_ds,
};
use gt_inference::delta::{MoveHeader, MoveKey, Receipt, Recording, Transition, Workspace};
use gt_inference::ids::{BEdge, Group, Stamp, Weight};

// ---------------------------------------------------------------------------
// A block state built by hand, and the move plan that goes with it.
//
// The entries are produced here by a transcription of
// `modify_entries_dispatch` (`blockmodel/entries.hh:227-320`) independent of
// U25's `record`. That independence is deliberate and kept now that `record`
// has landed: if the audit and the recorder shared a scan, a scan defect
// would cancel. The two are checked against each other -- and the audits are
// driven against the real recorder -- in the wave-8 integration section at the
// bottom of this file.
// ---------------------------------------------------------------------------

fn grp(i: u32) -> Group {
    Group::new(i).expect("group index in range")
}

fn vid(i: usize) -> VertexId {
    VertexId::from_index(i)
}

/// A tiny weighted multigraph with a partition, as plain data.
struct Fix {
    /// Block-graph slots, i.e. `num_vertices(_bg)`.
    slots: usize,
    /// `_b[v]`.
    b: Vec<u32>,
    /// `(source, target, weight)`. For an undirected fixture the pair is
    /// unordered and listed once.
    edges: Vec<(usize, usize, i64)>,
}

impl Fix {
    /// Seed a state: every vertex weighs one, every edge lands on its pair.
    fn build<D: Dir>(&self) -> BlockState<D, i64> {
        let mut st = BlockState::<D, i64>::new(self.slots, self.b.len());
        for (v, &g) in self.b.iter().enumerate() {
            st.assign(vid(v), grp(g), 1);
        }
        for &(u, w, e) in &self.edges {
            st.seed_pair(grp(self.b[u]), grp(self.b[w]), e);
        }
        st
    }
}

/// The entry set and degree magnitudes for "move `v` from `b[v]` to `nr`".
struct Plan {
    pairs: Vec<(Group, Group, i64)>,
    dkin: i64,
    dkout: i64,
}

/// `modify_entries_dispatch` (`entries.hh:227-320`), transcribed.
///
/// The post-move group of a neighbour `u` is `b[u]` unless `u == v`, which is
/// the `nb` ad-hoc property map of `state.hh:451-459`. Self-loops are visited
/// on the out pass only, so their weight is counted once in `dkout` and once
/// in `dkin` -- `_mrp[r]` and `_mrm[r]` each gain one copy of a directed
/// self-loop -- and twice in the undirected `dkout`, because an undirected
/// self-pair contributes `2 * w` to `_mrp[r]` (`entries.hh:393-398` with
/// `r == s`).
fn plan<D: Dir>(f: &Fix, v: usize, nr: u32) -> Plan {
    let r = f.b[v];
    let mut acc: BTreeMap<(u32, u32), i64> = BTreeMap::new();
    let mut add = |a: u32, b: u32, d: i64| *acc.entry((a, b)).or_insert(0) += d;
    let (mut dkin, mut dkout) = (0i64, 0i64);

    for &(a, b, w) in &f.edges {
        if D::DIRECTED {
            if a == v {
                let t = if b == v { nr } else { f.b[b] };
                add(r, f.b[b], -w);
                add(nr, t, w);
                dkout += w;
                if b == v {
                    dkin += w;
                }
            } else if b == v {
                add(f.b[a], r, -w);
                add(f.b[a], nr, w);
                dkin += w;
            }
        } else if a == v && b == v {
            add(r, r, -w);
            add(nr, nr, w);
            dkout += 2 * w;
        } else if a == v {
            add(r, f.b[b], -w);
            add(nr, f.b[b], w);
            dkout += w;
        } else if b == v {
            add(f.b[a], r, -w);
            add(f.b[a], nr, w);
            dkout += w;
        }
    }

    Plan {
        pairs: acc
            .into_iter()
            .map(|((a, b), d)| (grp(a), grp(b), d))
            .collect(),
        dkin,
        dkout,
    }
}

/// The header for that move, snapshotting both endpoints from the live state.
fn header<S: BlockView<W = i64>>(st: &S, r: u32, nr: u32, p: &Plan) -> MoveHeader<i64> {
    MoveHeader {
        r: Some(grp(r)),
        nr: Some(grp(nr)),
        r_img: st.end_image(Some(grp(r))),
        nr_img: st.end_image(Some(grp(nr))),
        dkin: p.dkin,
        dkout: p.dkout,
        // Every vertex in these fixtures weighs one.
        dr: 1,
        dnr: 1,
    }
}

/// Record and seal one move. The lifetime is the workspace's, never `st`'s.
fn seal<'w, D: Dir>(
    st: &BlockState<D, i64>,
    ws: &'w mut Workspace<D, i64>,
    slots: usize,
    hdr: MoveHeader<i64>,
    pairs: &[(Group, Group, i64)],
) -> Transition<'w, D, i64> {
    let stamp = st.stamp();
    let mut rec = Recording::new(&mut ws.stack, hdr, stamp);
    rec.level_mut(0).begin(
        MoveKey {
            from: hdr.r,
            to: hdr.nr,
        },
        slots,
    );
    let mut resolve = |a, b| st.resolve(a, b);
    for &(a, b, w) in pairs {
        rec.level_mut(0)
            .touch_dyn(a, b, w, &mut resolve)
            .expect("every recorded pair touches an endpoint of the move");
    }
    rec.seal()
}

/// The standing directed fixture: six vertices over four occupied groups of
/// six slots, with a self-loop, an antiparallel pair and a parallel edge.
fn directed_fix() -> Fix {
    Fix {
        slots: 6,
        b: vec![0, 0, 1, 1, 2, 2],
        edges: vec![
            (0, 2, 1),
            (0, 3, 2),
            (2, 0, 1),
            (1, 4, 3),
            (4, 1, 1),
            (0, 0, 2),
            (0, 5, 1),
            (5, 0, 1),
            (3, 0, 1),
            (2, 4, 1),
            (0, 2, 2),
        ],
    }
}

fn undirected_fix() -> Fix {
    Fix {
        slots: 5,
        b: vec![0, 0, 1, 1, 2, 3],
        edges: vec![
            (0, 2, 1),
            (0, 3, 2),
            (1, 4, 3),
            (0, 0, 2),
            (0, 5, 1),
            (3, 0, 1),
            (2, 4, 1),
            (0, 1, 2),
        ],
    }
}

/// Seal, price, audit the price, commit, audit the commit.
///
/// The whole point of D10 in eleven lines: `t` borrows the workspace, so the
/// `&mut st` of the commit is free to happen while it is still alive, and the
/// `Receipt` carries the before-image out the other side.
fn full_cycle<D: Dir>(
    st: &mut BlockState<D, i64>,
    f: &Fix,
    v: usize,
    nr: u32,
    params: EntropyParams,
    c: &Cache,
) -> (Receipt<i64>, f64) {
    let p = plan::<D>(f, v, nr);
    let hdr = header(st, f.b[v], nr, &p);
    let mut ws = Workspace::<D, i64>::with_levels(1);
    let t = seal(st, &mut ws, f.slots, hdr, &p.pairs);

    let ds = sparse_ds(t.level(0), params, c);
    audit_price(&*st, &t, 0, params, c, ds).expect("the recorded before-image is the live state");

    let receipt = st.commit(t.into_levels().next().expect("one level"));
    audit_commit(&*st, &receipt).expect("a committed delta audits");
    (receipt, ds)
}

// ---------------------------------------------------------------------------
// Two lying read faces.
// ---------------------------------------------------------------------------

/// Which single field a [`Corrupt`] view misreports, and by how much.
#[derive(Clone, Copy, Debug)]
enum Fault {
    Mrs(u32, u32, i64),
    Mrp(u32, i64),
    Mrm(u32, i64),
    Wr(u32, i64),
}

/// A read face that lies about exactly one field of one group.
///
/// Fault injection through the *trait*, not through the storage: the state,
/// the delta and the receipt are all the real ones, so the test cannot
/// accidentally verify a corruption it also constructed. There is no way to
/// perturb one `_mrs` of a `BlockState` through its public surface without
/// perturbing `_mrp` and `_mrm` too -- `apply_entry` is one mutator by design
/// -- which is the sort of thing the C++'s hand-maintained aggregates make
/// trivially possible and this port makes deliberately hard.
struct Corrupt<'a, S: BlockView> {
    inner: &'a S,
    fault: Fault,
    /// The block edge `Fault::Mrs` names, resolved once against the honest
    /// state so that nothing here has to know how a `BEdge` is addressed.
    edge: Option<BEdge>,
}

impl<'a, S: BlockView> Corrupt<'a, S> {
    fn new(inner: &'a S, fault: Fault) -> Self {
        let edge = match fault {
            Fault::Mrs(r, s, _) => inner.find_me(grp(r), grp(s)),
            _ => None,
        };
        assert!(
            !matches!(fault, Fault::Mrs(..)) || edge.is_some(),
            "the fault names a block pair the state does not hold"
        );
        Corrupt { inner, fault, edge }
    }
}

/// Add `d` to `base` when `target` names `g`.
fn nudge<W: Weight>(base: W, target: Option<(u32, i64)>, g: Group) -> W {
    match target {
        Some((t, d)) if t as usize == g.index() => base + W::from_i64(d),
        _ => base,
    }
}

impl<S: BlockView> BlockView for Corrupt<'_, S> {
    type D = S::D;
    type W = S::W;

    fn stamp(&self) -> Stamp {
        self.inner.stamp()
    }
    fn group_of(&self, v: VertexId) -> Option<Group> {
        self.inner.group_of(v)
    }
    fn n_groups(&self) -> usize {
        self.inner.n_groups()
    }
    fn find_me(&self, r: Group, s: Group) -> Option<BEdge> {
        self.inner.find_me(r, s)
    }
    fn mrs(&self, e: BEdge) -> S::W {
        let w = self.inner.mrs(e);
        match self.fault {
            Fault::Mrs(_, _, d) if self.edge == Some(e) => w + S::W::from_i64(d),
            _ => w,
        }
    }
    fn mrp(&self, r: Group) -> S::W {
        let t = match self.fault {
            Fault::Mrp(g, d) => Some((g, d)),
            _ => None,
        };
        nudge(self.inner.mrp(r), t, r)
    }
    fn mrm(&self, r: Group) -> S::W {
        let t = match self.fault {
            Fault::Mrm(g, d) => Some((g, d)),
            _ => None,
        };
        nudge(self.inner.mrm(r), t, r)
    }
    fn wr(&self, r: Group) -> S::W {
        let t = match self.fault {
            Fault::Wr(g, d) => Some((g, d)),
            _ => None,
        };
        nudge(self.inner.wr(r), t, r)
    }
    fn move_prob(&self, t: &Transition<'_, S::D, S::W>, v: VertexId, c: f64) -> f64 {
        self.inner.move_prob(t, v, c)
    }
}

/// A read face that counts.
#[derive(Default)]
struct Reads {
    find_me: Cell<usize>,
    mrs: Cell<usize>,
    mrp: Cell<usize>,
    mrm: Cell<usize>,
    wr: Cell<usize>,
}

impl Reads {
    fn total(&self) -> usize {
        self.find_me.get() + self.mrs.get() + self.mrp.get() + self.mrm.get() + self.wr.get()
    }
}

fn bump(c: &Cell<usize>) {
    c.set(c.get() + 1);
}

/// Counts every aggregate read the audit performs.
///
/// A clock cannot establish "O(#entries) and not O(B^2)" -- it establishes
/// what the machine did this afternoon. A counter can, which is the same
/// argument `DeltaBuf::field_writes` already makes for `begin`.
struct Spy<'a, S: BlockView> {
    inner: &'a S,
    reads: Reads,
}

impl<S: BlockView> BlockView for Spy<'_, S> {
    type D = S::D;
    type W = S::W;

    fn stamp(&self) -> Stamp {
        self.inner.stamp()
    }
    fn group_of(&self, v: VertexId) -> Option<Group> {
        self.inner.group_of(v)
    }
    fn n_groups(&self) -> usize {
        self.inner.n_groups()
    }
    fn find_me(&self, r: Group, s: Group) -> Option<BEdge> {
        bump(&self.reads.find_me);
        self.inner.find_me(r, s)
    }
    fn mrs(&self, e: BEdge) -> S::W {
        bump(&self.reads.mrs);
        self.inner.mrs(e)
    }
    fn mrp(&self, r: Group) -> S::W {
        bump(&self.reads.mrp);
        self.inner.mrp(r)
    }
    fn mrm(&self, r: Group) -> S::W {
        bump(&self.reads.mrm);
        self.inner.mrm(r)
    }
    fn wr(&self, r: Group) -> S::W {
        bump(&self.reads.wr);
        self.inner.wr(r)
    }
    fn move_prob(&self, t: &Transition<'_, S::D, S::W>, v: VertexId, c: f64) -> f64 {
        self.inner.move_prob(t, v, c)
    }
}

fn params() -> EntropyParams {
    EntropyParams {
        deg_corr: true,
        ..EntropyParams::default()
    }
}

// ---------------------------------------------------------------------------
// The call site. This is the acceptance test `Receipt` exists for.
// ---------------------------------------------------------------------------

/// `audit_commit(st, &receipt)` compiles and passes *after* the commit.
///
/// The `let receipt = st.commit(..)` on the line above the audit is the whole
/// assertion: with `commit` taking the whole `Transition` by value there is no
/// value left to audit (`error[E0382]`). And there is no way to run it early:
/// a `Receipt` is minted by `commit` and by nothing else, so "audit before the
/// commit" -- the check that fails on every non-zero delta by construction --
/// is not a call this API can express.
#[test]
fn audit_commit_passes_at_its_only_possible_call_site() {
    let f = directed_fix();
    let mut st = f.build::<Directed>();

    let p = plan::<Directed>(&f, 0, 3);
    assert!(p.pairs.len() > 4, "the fixture must exercise several pairs");

    let hdr = header(&st, f.b[0], 3, &p);
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let t = seal(&st, &mut ws, f.slots, hdr, &p.pairs);

    let receipt = st.commit(t.into_levels().next().expect("one level"));
    assert_eq!(audit_commit(&st, &receipt), Ok(()));
}

/// The same delta, audited against a state it was never applied to.
///
/// A fresh `BlockState` of the same shape sits at epoch 0 with an identical
/// set of aggregates, so only the `StateId` half of the stamp can tell the
/// two apart -- defect #36, arriving one step later than `try_commit` sees it.
#[test]
fn a_receipt_audited_against_a_foreign_state_is_refused_by_identity_alone() {
    let f = directed_fix();
    let mut st = f.build::<Directed>();
    let twin = f.build::<Directed>();
    let c = Cache::build(1024);
    let (receipt, _) = full_cycle(&mut st, &f, 0, 3, params(), &c);

    let err = audit_commit(&twin, &receipt).expect_err("a foreign state must be refused");
    match err {
        AuditError::Stamp {
            delta_state,
            state_state,
            delta_epoch,
            state_epoch,
        } => {
            assert_ne!(delta_state, state_state);
            assert_eq!((delta_epoch, state_epoch), (0, 0), "both were at epoch 0");
        }
        other => panic!("expected a stamp mismatch, got {other}"),
    }
}

// ---------------------------------------------------------------------------
// Fault injection.
// ---------------------------------------------------------------------------

/// `mrs` off by one: the audit names the *pair*, both endpoints.
#[test]
fn a_corrupt_pair_weight_names_its_two_groups() {
    let f = directed_fix();
    let mut st = f.build::<Directed>();
    let c = Cache::build(1024);
    let (receipt, _) = full_cycle(&mut st, &f, 0, 3, params(), &c);

    for delta in [1i64, -1] {
        // Every entry, not just the first: the audit must not report whichever
        // pair happens to come out of the map ordering.
        for e in &receipt.entries {
            let after = e.mrs_before + e.delta;
            if after == 0 {
                // A pair the commit removed has no block edge to misreport.
                continue;
            }
            let view = Corrupt::new(
                &st,
                Fault::Mrs(e.r.index() as u32, e.s.index() as u32, delta),
            );
            let err = audit_commit(&view, &receipt).expect_err("a corrupt mrs must be caught");
            match err {
                AuditError::EdgeWeight {
                    r,
                    s,
                    expected,
                    found,
                } => {
                    assert_eq!((r, s), (e.r.index(), e.s.index()));
                    assert_eq!(expected, after as f64);
                    assert_eq!(found, (after + delta) as f64);
                }
                other => panic!("expected an edge-weight fault, got {other}"),
            }
        }
    }
}

/// `mrp` off by one: the audit names the group and the field.
#[test]
fn a_corrupt_mrp_names_its_group_and_field() {
    let f = directed_fix();
    let mut st = f.build::<Directed>();
    let c = Cache::build(1024);
    let (receipt, _) = full_cycle(&mut st, &f, 0, 3, params(), &c);

    for (g, img, sign) in [(0u32, receipt.hdr.r_img, -1i64), (3, receipt.hdr.nr_img, 1)] {
        for delta in [1i64, -1] {
            let view = Corrupt::new(&st, Fault::Mrp(g, delta));
            let err = audit_commit(&view, &receipt).expect_err("a corrupt mrp must be caught");
            let expected = (img.mrp + sign * receipt.hdr.dkout) as f64;
            assert_eq!(
                err,
                AuditError::EndScalar {
                    r: g as usize,
                    field: "mrp",
                    expected,
                    found: expected + delta as f64,
                }
            );
        }
    }
}

/// `mrm` off by one, on a directed model where `_mrm` is maintained at all.
#[test]
fn a_corrupt_mrm_names_its_group_and_field() {
    let f = directed_fix();
    let mut st = f.build::<Directed>();
    let c = Cache::build(1024);
    let (receipt, _) = full_cycle(&mut st, &f, 0, 3, params(), &c);

    for (g, img, sign) in [(0u32, receipt.hdr.r_img, -1i64), (3, receipt.hdr.nr_img, 1)] {
        let view = Corrupt::new(&st, Fault::Mrm(g, 1));
        let err = audit_commit(&view, &receipt).expect_err("a corrupt mrm must be caught");
        let expected = (img.mrm + sign * receipt.hdr.dkin) as f64;
        assert_eq!(
            err,
            AuditError::EndScalar {
                r: g as usize,
                field: "mrm",
                expected,
                found: expected + 1.0,
            }
        );
    }
}

/// `wr` off by one: the vertex weight the *header* moves, not the entries.
#[test]
fn a_corrupt_wr_names_its_group_and_field() {
    let f = directed_fix();
    let mut st = f.build::<Directed>();
    let c = Cache::build(1024);
    let (receipt, _) = full_cycle(&mut st, &f, 0, 3, params(), &c);

    for (g, img, moved) in [
        (0u32, receipt.hdr.r_img, -receipt.hdr.dr),
        (3, receipt.hdr.nr_img, receipt.hdr.dnr),
    ] {
        for delta in [1i64, -1] {
            let view = Corrupt::new(&st, Fault::Wr(g, delta));
            let err = audit_commit(&view, &receipt).expect_err("a corrupt wr must be caught");
            let expected = (img.wr + moved) as f64;
            assert_eq!(
                err,
                AuditError::EndScalar {
                    r: g as usize,
                    field: "wr",
                    expected,
                    found: expected + delta as f64,
                }
            );
        }
    }
}

/// An undirected model has no `_mrm` to corrupt, and the audit does not read
/// one.
///
/// `update_rs` writes `_mrp[s]` rather than `_mrm[s]` when the graph is
/// undirected (`entries.hh:393-398`), so `_mrm` is structurally zero and a
/// view that lies about it changes nothing. Asserting that the audit still
/// passes is what pins "the read is skipped" rather than "the read happened to
/// agree".
#[test]
fn an_undirected_audit_neither_reads_nor_checks_mrm() {
    let f = undirected_fix();
    let mut st = f.build::<Undirected>();
    let c = Cache::build(1024);
    let (receipt, _) = full_cycle(&mut st, &f, 0, 4, params(), &c);

    for g in 0..f.slots as u32 {
        let view = Corrupt::new(&st, Fault::Mrm(g, 7));
        assert_eq!(
            audit_commit(&view, &receipt),
            Ok(()),
            "an undirected audit must not consult _mrm"
        );
    }

    // And the corruption is real under a directed reading of the same field.
    let spy = Spy {
        inner: &st,
        reads: Reads::default(),
    };
    audit_commit(&spy, &receipt).expect("the honest undirected state audits");
    assert_eq!(spy.reads.mrm.get(), 0, "no undirected _mrm read at all");
}

// ---------------------------------------------------------------------------
// Cost.
// ---------------------------------------------------------------------------

/// `audit_commit` is O(#entries), counted rather than timed.
///
/// Exactly one `find_me` per entry, one `mrs` per entry that still has a block
/// edge, and three scalars per present endpoint. Nothing scales with the slot
/// count, which is the difference between this and `copy_state_wrap`
/// (`base_states.py:44-59`).
#[test]
fn audit_commit_reads_the_state_a_fixed_number_of_times_per_entry() {
    let f = directed_fix();
    let c = Cache::build(1024);

    // Two moves of visibly different fan-out through the same fixture.
    for (v, nr) in [(0usize, 3u32), (4, 5)] {
        let mut st = f.build::<Directed>();
        let (receipt, _) = full_cycle(&mut st, &f, v, nr, params(), &c);

        let spy = Spy {
            inner: &st,
            reads: Reads::default(),
        };
        assert_eq!(audit_commit(&spy, &receipt), Ok(()));

        let n = receipt.entries.len();
        let live = receipt
            .entries
            .iter()
            .filter(|e| e.mrs_before + e.delta != 0)
            .count();

        assert_eq!(spy.reads.find_me.get(), n, "one pair probe per entry");
        assert_eq!(
            spy.reads.mrs.get(),
            live,
            "no weight read for a removed pair"
        );
        assert_eq!(spy.reads.mrp.get(), 2, "one per endpoint");
        assert_eq!(spy.reads.mrm.get(), 2);
        assert_eq!(spy.reads.wr.get(), 2);
        assert_eq!(spy.reads.total(), n + live + 6);
        assert!(
            spy.reads.total() <= 2 * n + 6,
            "the bound the module header claims"
        );
    }
}

// ---------------------------------------------------------------------------
// `audit_price`.
// ---------------------------------------------------------------------------

/// The claim `sparse_ds` makes is the one `audit_price` recomputes.
#[test]
fn audit_price_agrees_with_the_pricing_it_audits() {
    for deg_corr in [false, true] {
        let ea = EntropyParams {
            deg_corr,
            ..EntropyParams::default()
        };
        let c = Cache::build(1024);

        let f = directed_fix();
        let mut st = f.build::<Directed>();
        let (_, ds) = full_cycle(&mut st, &f, 0, 3, ea, &c);
        assert!(ds.is_finite());

        let u = undirected_fix();
        let mut ust = u.build::<Undirected>();
        let (_, uds) = full_cycle(&mut ust, &u, 0, 4, ea, &c);
        assert!(uds.is_finite());
    }
}

/// A wrong claim is reported with both numbers and the tolerance used.
#[test]
fn audit_price_reports_a_wrong_claim() {
    let f = directed_fix();
    let st = f.build::<Directed>();
    let c = Cache::build(1024);
    let ea = params();

    let p = plan::<Directed>(&f, 0, 3);
    let hdr = header(&st, f.b[0], 3, &p);
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let t = seal(&st, &mut ws, f.slots, hdr, &p.pairs);

    let ds = sparse_ds(t.level(0), ea, &c);
    assert_eq!(audit_price(&st, &t, 0, ea, &c, ds), Ok(()));

    let err = audit_price(&st, &t, 0, ea, &c, ds + 1e-3)
        .expect_err("a claim off by a millilog must be caught");
    match err {
        AuditError::Entropy {
            claimed,
            recomputed,
            tol,
        } => {
            assert_eq!(claimed, ds + 1e-3);
            assert!((recomputed - ds).abs() <= 1e-12 * ds.abs().max(1.0));
            assert!(tol < 1e-6, "tolerance {tol} is not a check");
        }
        other => panic!("expected an entropy mismatch, got {other}"),
    }

    // A NaN claim is a failure, not a pass: `base_states.py:47` asserts
    // `not isnan(S)` separately, which is the same statement.
    assert!(matches!(
        audit_price(&st, &t, 0, ea, &c, f64::NAN),
        Err(AuditError::Entropy { .. })
    ));
}

/// The independence that makes `audit_price` worth running.
///
/// `sparse_ds` reads only the recorded before-image; `audit_price` reads only
/// the live state. A state that disagrees with the snapshot the recorder
/// interned is therefore visible to exactly one of the two, and that is the
/// class of defect -- a stale or mis-resolved `Entry::mrs_before`,
/// `MoveHeader::r_img` -- that pricing structurally cannot see.
#[test]
fn audit_price_catches_a_before_image_that_the_state_contradicts() {
    let f = directed_fix();
    let st = f.build::<Directed>();
    let c = Cache::build(1024);

    // `0 -> 2`, not `0 -> 3`: group 2 is *occupied*, and it has to be. The
    // plain `vterm` is `(mrp + mrm) * log(wr)` (`entropy.hh:84-87`), which is
    // identically zero at `wr in {0, 1}` however wrong `mrp` and `mrm` are, so
    // a move onto an empty group cannot demonstrate anything about them.
    let p = plan::<Directed>(&f, 0, 2);
    let hdr = header(&st, f.b[0], 2, &p);
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let t = seal(&st, &mut ws, f.slots, hdr, &p.pairs);

    // The entry half: one pair's live weight is not what was interned.
    for (i, e) in t.level(0).entries().iter().enumerate() {
        // A pair the move *creates* has no block edge yet, so there is no
        // live `_mrs[me]` for a read face to misreport: `find_me` returns
        // `None` and `live_pair` never asks.
        if e.delta == 0 || e.me.is_none() {
            continue;
        }
        let ea = params();
        let ds = sparse_ds(t.level(0), ea, &c);
        let view = Corrupt::new(&st, Fault::Mrs(e.r.index() as u32, e.s.index() as u32, 1));
        assert!(
            matches!(
                audit_price(&view, &t, 0, ea, &c, ds),
                Err(AuditError::Entropy { .. })
            ),
            "entry {i}: a live mrs that contradicts Entry::mrs_before must be caught"
        );
    }

    // The header half: one endpoint's live scalars are not the snapshot's.
    // Which scalars `vterm` reads is the model's business, and the fault has
    // to be injected into one it reads: the degree-corrected arm
    // (`entropy.hh:77-80`) is `lgamma(mrp + 1) + lgamma(mrm + 1)` and does not
    // look at `wr` at all, while the plain arm looks at all three.
    for (deg_corr, fault) in [
        (true, Fault::Mrp(0, 1)),
        (true, Fault::Mrm(0, 1)),
        (true, Fault::Mrp(2, 1)),
        (true, Fault::Mrm(2, 1)),
        (false, Fault::Mrp(0, 1)),
        (false, Fault::Mrm(2, 1)),
        (false, Fault::Wr(0, 1)),
        (false, Fault::Wr(2, 1)),
    ] {
        let ea = EntropyParams {
            deg_corr,
            ..EntropyParams::default()
        };
        let ds = sparse_ds(t.level(0), ea, &c);
        let view = Corrupt::new(&st, fault);
        assert!(
            matches!(
                audit_price(&view, &t, 0, ea, &c, ds),
                Err(AuditError::Entropy { .. })
            ),
            "a live scalar that contradicts MoveHeader's snapshot must be caught \
             ({fault:?}, deg_corr = {deg_corr})"
        );
    }

    // The converse, stated rather than left to luck: a `wr` fault under
    // `deg_corr` is *correctly* invisible, because the degree-corrected
    // `vterm` never reads `_wr`. Asserting otherwise would be asserting a bug.
    let ea = params();
    let ds = sparse_ds(t.level(0), ea, &c);
    for g in [0u32, 2] {
        let view = Corrupt::new(&st, Fault::Wr(g, 1));
        assert_eq!(
            audit_price(&view, &t, 0, ea, &c, ds),
            Ok(()),
            "the degree-corrected vterm does not read _wr (entropy.hh:77-80)"
        );
    }
}

/// `audit_price` is pre-commit, and says so when it is not.
///
/// The transition borrows the *workspace*, so it outlives an intervening
/// `&mut st` -- the ownership inversion D10 rests on. That makes "priced
/// against a revision that has moved on" an expressible mistake, and the stamp
/// is what catches it.
#[test]
fn audit_price_refuses_a_transition_the_state_has_moved_past() {
    let f = directed_fix();
    let mut st = f.build::<Directed>();
    let c = Cache::build(1024);
    let ea = params();

    let p = plan::<Directed>(&f, 0, 3);
    let hdr = header(&st, f.b[0], 3, &p);
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let t = seal(&st, &mut ws, f.slots, hdr, &p.pairs);
    let ds = sparse_ds(t.level(0), ea, &c);
    assert_eq!(audit_price(&st, &t, 0, ea, &c, ds), Ok(()));

    // Something else commits. `t` is still alive; that is the point.
    let mut ws2 = Workspace::<Directed, i64>::with_levels(1);
    let p2 = plan::<Directed>(&f, 4, 0);
    let hdr2 = header(&st, f.b[4], 0, &p2);
    let t2 = seal(&st, &mut ws2, f.slots, hdr2, &p2.pairs);
    let _ = st.commit(t2.into_levels().next().expect("one level"));

    match audit_price(&st, &t, 0, ea, &c, ds).expect_err("epoch 0 is stale at epoch 1") {
        AuditError::Stamp {
            delta_epoch,
            state_epoch,
            delta_state,
            state_state,
        } => {
            assert_eq!((delta_epoch, state_epoch), (0, 1));
            assert_eq!(delta_state, state_state, "the same state, a later revision");
        }
        other => panic!("expected a stamp mismatch, got {other}"),
    }
}

// ---------------------------------------------------------------------------
// A sweep, with both audits on every move.
// ---------------------------------------------------------------------------

/// A deterministic generator, so a failure reproduces from the test name.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn below(&mut self, n: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) % n
    }
}

/// Two thousand random moves, every one priced, audited, committed and audited
/// again.
///
/// This is the configuration graph-tool ships with switched **off**
/// (`base_states.py:33`). Here it is the default, and it costs O(#entries).
#[test]
fn a_sweep_audits_every_move_in_both_directions() {
    let mut rng = Lcg::new(0x5EED_0026);
    let slots = 8usize;
    let n = 40usize;

    let mut f = Fix {
        slots,
        b: (0..n).map(|i| (i % slots) as u32).collect(),
        edges: Vec::new(),
    };
    for _ in 0..140 {
        let a = rng.below(n as u32) as usize;
        let b = rng.below(n as u32) as usize;
        f.edges.push((a, b, 1 + i64::from(rng.below(3))));
    }

    let ea = params();
    let c = Cache::build(4096);
    let mut st = f.build::<Directed>();
    let mut moves = 0usize;

    for _ in 0..2000 {
        let v = rng.below(n as u32) as usize;
        let nr = rng.below(slots as u32);
        if nr == f.b[v] {
            continue;
        }
        full_cycle(&mut st, &f, v, nr, ea, &c);
        // The state's own `_b` is not written by `commit` (see the note in the
        // unit report); the fixture is the partition of record.
        f.b[v] = nr;
        moves += 1;
    }

    assert!(moves > 1500, "only {moves} moves were attempted");
}

// ===========================================================================
// WAVE-8 INTEGRATION (added by the integrator, not by U26).
//
// Everything above this line produces its entries from *this* file's
// transcription of `modify_entries_dispatch` (`entries.hh:227-320`), because
// when U26 was written `record`/`scan` were still `todo!()`. U25 has since
// landed, so the audits are now driven once against the real recorder.
//
// The independence above is kept deliberately: if every test in the file went
// through `record`, a recorder defect and an audit defect could cancel. What
// these two tests add is the other half -- that U25's recorder and U26's
// audit, transcribed from the same C++ independently, actually agree.
//
// They also exercise the two `pub use` lines the integrator added to
// `blockmodel/mod.rs`: `ScanDir` is named below in a helper that is generic
// over directedness, which is exactly the thing U25 reported it could not
// write from outside the crate.
// ===========================================================================

use gt_core::adj::AdjList;
use gt_core::graph::GraphRef;
use gt_core::view::Undirect;
use gt_inference::blockmodel::{ScanDir, record};

/// The fixture's weighted edge list, expanded into parallel unit edges.
///
/// `record` carries no `_eweight` (U25's signature has none), so multiplicity
/// *is* parallel edges -- the same reading `Fix::build` gives when it seeds a
/// pair at weight `w`.
fn adj_of(f: &Fix) -> AdjList {
    let mut adj = AdjList::with_vertices(f.b.len());
    for &(a, b, w) in &f.edges {
        for _ in 0..w {
            adj.add_edge(vid(a), vid(b)).expect("both vertices exist");
        }
    }
    adj
}

/// The dense matrix of a pair list, with the undirected pair normalised.
///
/// `DeltaBuf` is free to emit an undirected pair in either order (one pair is
/// one cell, not one ordering), so a comparison that fixes the order would be
/// testing the buffer's internal choice rather than the delta.
fn matrix(pairs: &[(Group, Group, i64)], slots: usize, directed: bool) -> Vec<i64> {
    let mut m = vec![0i64; slots * slots];
    for &(r, s, d) in pairs {
        let (x, y) = (r.index(), s.index());
        let (x, y) = if directed || x <= y { (x, y) } else { (y, x) };
        m[x * slots + y] += d;
    }
    m
}

/// Record one move with U25's recorder, price it, audit both ends, commit.
///
/// Generic over directedness through `ScanDir`, which is the whole point of
/// re-exporting it: before that line existed this had to be written twice.
#[allow(clippy::too_many_arguments)]
fn recorded_cycle<D, G, F>(
    st: &mut BlockState<D, i64>,
    g: G,
    v: usize,
    mv: MoveKey,
    nb: &F,
    slots: usize,
    ea: EntropyParams,
    c: &Cache,
) -> (Vec<(Group, Group, i64)>, i64, i64)
where
    D: Dir + ScanDir<G>,
    G: GraphRef,
    F: Fn(VertexId) -> Option<Group>,
{
    let mut ws = Workspace::<D, i64>::with_levels(1);
    let rec = record(&*st, g, vid(v), mv, nb, slots, 1i64, &mut ws);
    let t = rec.seal();
    let hdr = *t.level(0).header();

    let entries: Vec<_> = t
        .level(0)
        .entries()
        .iter()
        .map(|e| (e.r, e.s, e.delta))
        .collect();

    let ds = sparse_ds(t.level(0), ea, c);
    audit_price(&*st, &t, 0, ea, c, ds)
        .expect("the recorder's before-image is the state it was recorded against");

    let receipt = st.commit(t.into_levels().next().expect("one level"));
    audit_commit(&*st, &receipt).expect("a delta from the real recorder audits after commit");

    (entries, hdr.dkin, hdr.dkout)
}

/// U25's recorder and U26's transcription of `entries.hh` agree, pair for
/// pair and degree for degree, on every move of both standing fixtures.
///
/// This is the check that makes the rest of the file mean something: two
/// independent readings of `modify_entries_dispatch` producing the same
/// delta. `r == nr` is included, where the correct delta is zero and
/// graph-tool's is not (`entries.hh:283`).
#[test]
fn the_real_recorder_agrees_with_this_units_transcription() {
    let mut checked = 0usize;

    // -- directed ---------------------------------------------------------
    let f = directed_fix();
    let adj = adj_of(&f);
    for v in 0..f.b.len() {
        for nr in 0..f.slots as u32 {
            let p = plan::<Directed>(&f, v, nr);
            let st = f.build::<Directed>();
            let mut ws = Workspace::<Directed, i64>::with_levels(1);
            let mv = MoveKey {
                from: Some(grp(f.b[v])),
                to: Some(grp(nr)),
            };
            let nb = |u: VertexId| Group::new(if u.index() == v { nr } else { f.b[u.index()] });
            let rec = record(&st, &adj, vid(v), mv, &nb, f.slots, 1i64, &mut ws);
            let t = rec.seal();
            let hdr = *t.level(0).header();
            let got: Vec<_> = t
                .level(0)
                .entries()
                .iter()
                .map(|e| (e.r, e.s, e.delta))
                .collect();

            assert_eq!(
                matrix(&got, f.slots, true),
                matrix(&p.pairs, f.slots, true),
                "directed delta disagrees for v={v} -> nr={nr}"
            );
            assert_eq!(hdr.dkin, p.dkin, "directed dkin for v={v}");
            assert_eq!(hdr.dkout, p.dkout, "directed dkout for v={v}");
            checked += 1;
        }
    }

    // -- undirected -------------------------------------------------------
    let f = undirected_fix();
    let adj = adj_of(&f);
    for v in 0..f.b.len() {
        for nr in 0..f.slots as u32 {
            let p = plan::<Undirected>(&f, v, nr);
            let st = f.build::<Undirected>();
            let mut ws = Workspace::<Undirected, i64>::with_levels(1);
            let mv = MoveKey {
                from: Some(grp(f.b[v])),
                to: Some(grp(nr)),
            };
            let nb = |u: VertexId| Group::new(if u.index() == v { nr } else { f.b[u.index()] });
            let rec = record(&st, adj.undirect(), vid(v), mv, &nb, f.slots, 1i64, &mut ws);
            let t = rec.seal();
            let hdr = *t.level(0).header();
            let got: Vec<_> = t
                .level(0)
                .entries()
                .iter()
                .map(|e| (e.r, e.s, e.delta))
                .collect();

            assert_eq!(
                matrix(&got, f.slots, false),
                matrix(&p.pairs, f.slots, false),
                "undirected delta disagrees for v={v} -> nr={nr}"
            );
            assert_eq!(hdr.dkin, 0, "an undirected view has no in-degree");
            assert_eq!(
                hdr.dkout, p.dkout,
                "undirected dkout for v={v}; a self-loop counts twice"
            );
            checked += 1;
        }
    }

    assert_eq!(checked, 6 * 6 + 6 * 5);
}

/// A sweep in which every delta comes from `record`, both directednesses.
///
/// `a_sweep_audits_every_move_in_both_directions` above builds its entries by
/// hand and, despite its name, only runs the directed arm. This one is the
/// end-to-end path a caller actually takes -- record, price, audit, commit,
/// audit -- and runs it for `Directed` and for `Undirect`ed.
#[test]
fn a_recorded_sweep_prices_and_audits_in_both_directednesses() {
    let mut rng = Lcg::new(0x5EED_2526);
    let slots = 6usize;
    let n = 24usize;

    let mut adj = AdjList::with_vertices(n);
    for _ in 0..90 {
        let a = rng.below(n as u32) as usize;
        // One edge in five is a self-loop: the shape both halves handle
        // specially and the one the audit's `dkout` check pins.
        let b = if rng.below(5) == 0 {
            a
        } else {
            rng.below(n as u32) as usize
        };
        adj.add_edge(vid(a), vid(b)).expect("both vertices exist");
    }
    let b0: Vec<u32> = (0..n).map(|i| (i % slots) as u32).collect();

    let ea = params();
    let c = Cache::build(4096);

    // A state holding exactly the partition `b`, seeded from the graph.
    fn seed<D: Dir>(adj: &AdjList, b: &[u32], slots: usize) -> BlockState<D, i64> {
        let mut st = BlockState::<D, i64>::new(slots, b.len());
        for (i, &r) in b.iter().enumerate() {
            st.assign(vid(i), grp(r), 1);
        }
        for e in adj.edges() {
            st.seed_pair(grp(b[e.source().index()]), grp(b[e.target().index()]), 1);
        }
        st
    }

    let mut moves = 0usize;

    let mut b = b0.clone();
    let mut st = seed::<Directed>(&adj, &b, slots);
    for _ in 0..400 {
        let v = rng.below(n as u32) as usize;
        let nr = rng.below(slots as u32);
        let mv = MoveKey {
            from: Some(grp(b[v])),
            to: Some(grp(nr)),
        };
        // `nb` is `_b`, which is what `get_move_entries` passes
        // (`state.hh:1088`); the state's own `_b` is not written by `commit`.
        let nb = |u: VertexId| Group::new(b[u.index()]);
        recorded_cycle(&mut st, &adj, v, mv, &nb, slots, ea, &c);
        // The `_b[v] = r` of `add_partition_node` (`state.hh:779`). The
        // commit moved the vertex *weight* through `dr`/`dnr`; the partition
        // entry has no room in `MoveHeader`, so the caller owes it. Drop this
        // line and the *next* `record` reads a stale `b[u]` at
        // `entries.hh:250` and the audit catches it on `_mrm`.
        st.reseat(vid(v), grp(nr));
        b[v] = nr;
        moves += 1;
    }

    let mut b = b0.clone();
    let mut st = seed::<Undirected>(&adj, &b, slots);
    for _ in 0..400 {
        let v = rng.below(n as u32) as usize;
        let nr = rng.below(slots as u32);
        let mv = MoveKey {
            from: Some(grp(b[v])),
            to: Some(grp(nr)),
        };
        let nb = |u: VertexId| Group::new(b[u.index()]);
        recorded_cycle(&mut st, adj.undirect(), v, mv, &nb, slots, ea, &c);
        st.reseat(vid(v), grp(nr));
        b[v] = nr;
        moves += 1;
    }

    assert_eq!(moves, 800);
}

/// A commit moves the vertex *weight* but cannot move the partition entry, so
/// `group_of` is stale until the caller calls `reseat`.
///
/// This is the regression test for the defect the wave-8 integration found:
/// `add_partition_node` (`state.hh:759-780`) writes `_wr[r] += w` and
/// `_b[v] = r` together, and this port splits those across the delta and the
/// state. `MoveHeader` carries `r` and `nr` but no `VertexId`, so
/// `apply_level` has nothing to write `_b` with.
///
/// Without the `reseat`, the *next* `record` reads the stale `b[u]` at
/// `entries.hh:250` for a neighbour that has already moved and removes weight
/// from a pair that no longer holds it -- which `audit_commit` catches one
/// move later on `_mrm`. `a_recorded_sweep_prices_and_audits_in_both_directednesses`
/// fails at move 1 if the `reseat` line is deleted (verified by mutation).
#[test]
fn a_commit_moves_the_weight_and_the_caller_owes_the_partition_entry() {
    let f = directed_fix();
    let adj = adj_of(&f);
    let (v, nr) = (0usize, 2u32);
    let r = f.b[v];
    assert_ne!(r, nr);

    let mut st = f.build::<Directed>();
    let wr_r = st.wr(grp(r));
    let wr_nr = st.wr(grp(nr));

    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let mv = MoveKey {
        from: Some(grp(r)),
        to: Some(grp(nr)),
    };
    let nb = |u: VertexId| Group::new(f.b[u.index()]);
    let t = record(&st, &adj, vid(v), mv, &nb, f.slots, 1i64, &mut ws).seal();
    let receipt = st.commit(t.into_levels().next().expect("one level"));
    audit_commit(&st, &receipt).expect("the commit itself is sound");

    // The weight moved...
    assert_eq!(st.wr(grp(r)), wr_r - 1, "`dr` left `r`");
    assert_eq!(st.wr(grp(nr)), wr_nr + 1, "`dnr` entered `nr`");
    // ...and the partition entry did not.
    assert_eq!(
        st.group_of(vid(v)),
        Some(grp(r)),
        "`commit` cannot write `_b`: `MoveHeader` carries no `VertexId`"
    );

    st.reseat(vid(v), grp(nr));
    assert_eq!(st.group_of(vid(v)), Some(grp(nr)));
    // `reseat` is *not* `assign`: it must not move the weight a second time.
    assert_eq!(st.wr(grp(r)), wr_r - 1, "`reseat` re-applied `dr`");
    assert_eq!(st.wr(grp(nr)), wr_nr + 1, "`reseat` re-applied `dnr`");
}
