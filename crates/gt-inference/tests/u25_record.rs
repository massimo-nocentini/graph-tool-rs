//! U25 -- recording a transition, and projecting it up a hierarchy.
//!
//! The acceptance test is differential and deliberately dumb: build a random
//! multigraph with self-loops and parallel edges, count `e_rs` from scratch
//! before the move, count it from scratch after, and require the *recorded*
//! delta to be the difference. Nothing in the reference path knows about
//! half-fields, `single` dispatch, or the self-loop halving -- it iterates
//! edges and adds one -- which is the only way the design-phase pricing bug
//! (an undirected self-loop priced at twice its weight) would have been
//! caught.
//!
//! Two of the three checks are stronger than "the entries look right":
//!
//! * `the_committed_state_equals_a_from_scratch_rebuild` commits the delta and
//!   compares the whole `Aggregates` image -- `e_rs`, `_mrp`, `_mrm` and
//!   `_wr` -- against a state built independently from the after-partition;
//! * `the_entries_match_an_edge_by_edge_specification` restates the delta as a
//!   sum over *edges* (a self-loop once, whatever the traversal does) and
//!   compares pair by pair, which is what pins the halving.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use gt_core::adj::AdjList;
use gt_core::dir::{Directed, Undirected};
use gt_core::ids::VertexId;
use gt_core::view::Undirect;

use gt_inference::blockmodel::{BlockCommit, BlockState, BlockView, RowImage, propagate, record};
use gt_inference::delta::{MoveKey, Workspace};
use gt_inference::ids::Group;

// ===========================================================================
// An allocation counter (the shape `gt-algo/tests/u15_degree.rs` established)
//
// "`propagate` performs no heap allocation per call" is a statement about the
// global allocator, which has no safe observer. This is the one `unsafe` in
// the unit; it lives in a test binary, a separate crate from `gt-inference`
// and so not covered by that crate's `#![forbid(unsafe_code)]`, and every
// operation forwards to `System`.
// ===========================================================================

thread_local! {
    /// `-1` while this thread is not counting; otherwise the tally so far.
    static TALLY: Cell<isize> = const { Cell::new(-1) };
}

fn bump() {
    let _ = TALLY.try_with(|t| {
        let n = t.get();
        if n >= 0 {
            t.set(n + 1);
        }
    });
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(p, l, new) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Count the allocations `f` performs on this thread.
fn allocations<R>(f: impl FnOnce() -> R) -> (R, usize) {
    TALLY.with(|t| t.set(0));
    let r = f();
    let n = TALLY.with(|t| {
        let n = t.get();
        t.set(-1);
        n
    });
    (r, usize::try_from(n).expect("the tally was enabled"))
}

// ===========================================================================
// Fixtures
// ===========================================================================

fn grp(i: u32) -> Group {
    Group::new(i).expect("group index in range")
}
fn vid(i: usize) -> VertexId {
    VertexId::from_index(i)
}

/// A deterministic generator, so every failure reproduces from the case index.
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

/// One random case: a multigraph, a partition, and the move to record.
struct Case {
    adj: AdjList,
    /// `_b`, one group index per vertex.
    b: Vec<u32>,
    /// `nb`, the ad-hoc "group after the move" map (`state.hh:451-459`).
    nb: Vec<u32>,
    slots: usize,
    v: usize,
    nr: u32,
}

/// Self-loops and parallel edges are *deliberately* over-represented: they are
/// the two shapes the recorder handles specially and the two the C++ gets
/// wrong when `single`.
fn case(rng: &mut Lcg) -> Case {
    let n = 3 + rng.below(5) as usize;
    let slots = 2 + rng.below(3) as usize;
    let mut adj = AdjList::with_vertices(n);
    let m = rng.below(13) as usize;
    for _ in 0..m {
        // One edge in four is a self-loop; the rest draw endpoints from a
        // range narrow enough that parallel edges are common.
        let s = rng.below(n as u32) as usize;
        let t = if rng.below(4) == 0 {
            s
        } else {
            rng.below(n as u32) as usize
        };
        adj.add_edge(vid(s), vid(t)).expect("both vertices exist");
    }
    let b: Vec<u32> = (0..n).map(|_| rng.below(slots as u32)).collect();
    let nb: Vec<u32> = (0..n).map(|_| rng.below(slots as u32)).collect();
    let v = rng.below(n as u32) as usize;
    // `nr` is drawn from the whole range, so roughly one case in `slots` is
    // the `single` (`r == nr`) dispatch.
    let nr = rng.below(slots as u32);
    Case {
        adj,
        b,
        nb,
        slots,
        v,
        nr,
    }
}

/// A block state holding exactly the partition `b`: one unit-weight vertex per
/// entry, and one unit of block-edge weight per graph edge.
///
/// This is the from-scratch recount. It never looks at a delta.
fn build<D: gt_core::dir::Dir>(adj: &AdjList, b: &[u32], slots: usize) -> BlockState<D, i64> {
    let mut st = BlockState::<D, i64>::new(slots, adj.num_vertices());
    for (i, &r) in b.iter().enumerate() {
        st.assign(vid(i), grp(r), 1);
    }
    for e in adj.edges() {
        st.seed_pair(grp(b[e.source().index()]), grp(b[e.target().index()]), 1);
    }
    st
}

fn cell(x: u32, y: u32, slots: usize, directed: bool) -> usize {
    let (x, y) = if directed || x <= y { (x, y) } else { (y, x) };
    x as usize * slots + y as usize
}

/// The recorded delta, as the same dense matrix.
fn as_matrix(entries: &[(Group, Group, i64)], slots: usize, directed: bool) -> Vec<i64> {
    let mut m = vec![0i64; slots * slots];
    let mut seen = vec![false; slots * slots];
    for &(r, s, d) in entries {
        let i = cell(r.index() as u32, s.index() as u32, slots, directed);
        assert!(
            !seen[i],
            "the pair ({r:?}, {s:?}) has two entries; one pair is one cell \
             (entries.hh:122-135)"
        );
        seen[i] = true;
        m[i] += d;
    }
    m
}

/// The specification, restated over *edges*: moving `v` from `r` to `nr`
/// rewires every incident edge's endpoint pair, once per edge.
fn spec(c: &Case, directed: bool) -> Vec<i64> {
    let mut m = vec![0i64; c.slots * c.slots];
    let r = c.b[c.v];
    let nr = c.nr;
    // `nb` with the moved vertex's own new group substituted (`:257-259`).
    let after = |u: usize| if u == c.v { nr } else { c.nb[u] };

    for e in c.adj.edges() {
        let (s, t) = (e.source().index(), e.target().index());
        if directed {
            if s == c.v {
                m[cell(r, c.b[t], c.slots, true)] -= 1;
                m[cell(nr, after(t), c.slots, true)] += 1;
            }
            // A directed self-loop is an out-edge only: `:299-300` skips it in
            // the in-pass.
            if t == c.v && s != c.v {
                m[cell(c.b[s], r, c.slots, true)] -= 1;
                m[cell(after(s), nr, c.slots, true)] += 1;
            }
        } else if s == c.v || t == c.v {
            let u = if s == c.v { t } else { s };
            m[cell(r, c.b[u], c.slots, false)] -= 1;
            m[cell(nr, after(u), c.slots, false)] += 1;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// The two recorders. `ScanDir` is not re-exported by `blockmodel/mod.rs`, so
// a caller outside the crate cannot *name* the bound `record` needs -- which
// is exactly why these two are written out rather than being one generic
// helper.
// ---------------------------------------------------------------------------

fn run_directed(
    c: &Case,
    st: &mut BlockState<Directed, i64>,
    mv: MoveKey,
    commit: bool,
) -> Vec<(Group, Group, i64)> {
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let nb = |u: VertexId| Group::new(c.nb[u.index()]);
    let rec = record(&*st, &c.adj, vid(c.v), mv, &nb, c.slots, 1i64, &mut ws);
    let t = rec.seal();
    let entries: Vec<_> = t
        .level(0)
        .entries()
        .iter()
        .map(|e| (e.r, e.s, e.delta))
        .collect();
    if commit {
        st.commit(t.into_levels().next().expect("one level"));
    }
    entries
}

fn run_undirected(
    c: &Case,
    st: &mut BlockState<Undirected, i64>,
    mv: MoveKey,
    commit: bool,
) -> Vec<(Group, Group, i64)> {
    let mut ws = Workspace::<Undirected, i64>::with_levels(1);
    let nb = |u: VertexId| Group::new(c.nb[u.index()]);
    let rec = record(
        &*st,
        c.adj.undirect(),
        vid(c.v),
        mv,
        &nb,
        c.slots,
        1i64,
        &mut ws,
    );
    let t = rec.seal();
    let entries: Vec<_> = t
        .level(0)
        .entries()
        .iter()
        .map(|e| (e.r, e.s, e.delta))
        .collect();
    if commit {
        st.commit(t.into_levels().next().expect("one level"));
    }
    entries
}

fn move_key(c: &Case) -> MoveKey {
    MoveKey {
        from: Some(grp(c.b[c.v])),
        to: Some(grp(c.nr)),
    }
}

// ===========================================================================
// The acceptance test
// ===========================================================================

/// 300 directed and 300 undirected random multigraphs: commit the recorded
/// delta and require the state to equal one built from scratch on the
/// after-partition.
///
/// This is `e_rs` *and* `_mrp`, `_mrm`, `_wr` -- everything
/// `update_rs`/`apply_delta` touches (`entries.hh:350-427`) -- in one
/// comparison, and it includes the `r == nr` cases, where the correct delta is
/// zero and graph-tool's is not (see `record.rs`'s header on the
/// `if constexpr (!single)` guard at `entries.hh:283`).
#[test]
fn the_committed_state_equals_a_from_scratch_rebuild() {
    let mut rng = Lcg::new(0x5eed_0025);
    let (mut singles, mut self_loops) = (0, 0);

    for i in 0..300 {
        let mut c = case(&mut rng);
        // The committed comparison only means something for a single-vertex
        // move, so `nb` is `_b` -- which is what `get_move_entries` passes
        // (`state.hh:1088`).
        c.nb = c.b.clone();
        if c.b[c.v] == c.nr {
            singles += 1;
        }
        if c.adj
            .edges()
            .any(|e| e.source() == vid(c.v) && e.target() == vid(c.v))
        {
            self_loops += 1;
        }

        let mut after = c.b.clone();
        after[c.v] = c.nr;

        let mut st = build::<Directed>(&c.adj, &c.b, c.slots);
        let entries = run_directed(&c, &mut st, move_key(&c), true);
        let want = build::<Directed>(&c.adj, &after, c.slots);
        assert_images(&st, &want, i, "directed", &entries);

        let mut st = build::<Undirected>(&c.adj, &c.b, c.slots);
        let entries = run_undirected(&c, &mut st, move_key(&c), true);
        let want = build::<Undirected>(&c.adj, &after, c.slots);
        assert_images(&st, &want, i, "undirected", &entries);
    }

    // The sweep is only an acceptance test if it actually reached the two
    // shapes it exists for.
    assert!(singles > 20, "only {singles} `r == nr` cases in 300");
    assert!(
        self_loops > 20,
        "only {self_loops} cases with a self-loop at v"
    );
}

fn assert_images<D: gt_core::dir::Dir>(
    got: &BlockState<D, i64>,
    want: &BlockState<D, i64>,
    i: usize,
    tag: &str,
    entries: &[(Group, Group, i64)],
) {
    let a: Vec<RowImage<i64>> = got.aggregates().image();
    let b: Vec<RowImage<i64>> = want.aggregates().image();
    assert_eq!(
        a, b,
        "case {i} ({tag}): the committed delta disagrees with a from-scratch \
         rebuild; entries were {entries:?}"
    );
}

/// The same 600 cases against an edge-by-edge restatement of the delta, with
/// an **independent** `nb`.
///
/// This is the check that pins the halving: `spec` walks each incident edge
/// exactly once, where the undirected recorder walks a self-loop twice and has
/// to correct for it (`entries.hh:274-290`).
#[test]
fn the_entries_match_an_edge_by_edge_specification() {
    let mut rng = Lcg::new(0xf00d_0025);
    for i in 0..300 {
        let c = case(&mut rng);

        let mut st = build::<Directed>(&c.adj, &c.b, c.slots);
        let got = run_directed(&c, &mut st, move_key(&c), false);
        assert_eq!(
            as_matrix(&got, c.slots, true),
            spec(&c, true),
            "case {i} (directed): v={}, r={}, nr={}, b={:?}, nb={:?}",
            c.v,
            c.b[c.v],
            c.nr,
            c.b,
            c.nb
        );

        let mut st = build::<Undirected>(&c.adj, &c.b, c.slots);
        let got = run_undirected(&c, &mut st, move_key(&c), false);
        assert_eq!(
            as_matrix(&got, c.slots, false),
            spec(&c, false),
            "case {i} (undirected): v={}, r={}, nr={}, b={:?}, nb={:?}",
            c.v,
            c.b[c.v],
            c.nr,
            c.b,
            c.nb
        );
    }
}

// ===========================================================================
// The self-loop, in the open
// ===========================================================================

/// A vertex whose only edges are `k` parallel self-loops. Undirected, the
/// recorder traverses each one twice; the delta must still be `-k` on
/// `(r, r)` and `+k` on `(nr, nr)`.
#[test]
fn an_undirected_self_loop_is_halved() {
    for k in 1..4usize {
        let mut adj = AdjList::with_vertices(2);
        for _ in 0..k {
            adj.add_edge(vid(0), vid(0)).expect("v exists");
        }
        let c = Case {
            adj,
            b: vec![0, 1],
            nb: vec![0, 1],
            slots: 3,
            v: 0,
            nr: 2,
        };
        let mut st = build::<Undirected>(&c.adj, &c.b, c.slots);
        let got = run_undirected(&c, &mut st, move_key(&c), false);
        let m = as_matrix(&got, c.slots, false);
        assert_eq!(m[0], -(k as i64), "k={k}: e_rr");
        assert_eq!(m[2 * 3 + 2], k as i64, "k={k}: e_(nr,nr)");
        assert_eq!(m.iter().map(|x| x.abs()).sum::<i64>(), 2 * k as i64);

        // Directed: the self-loop is an out-edge once and skipped by the
        // in-pass, so the same numbers arrive without any halving.
        let mut st = build::<Directed>(&c.adj, &c.b, c.slots);
        let got = run_directed(&c, &mut st, move_key(&c), false);
        let m = as_matrix(&got, c.slots, true);
        assert_eq!(m[0], -(k as i64), "k={k}: directed e_rr");
        assert_eq!(m[2 * 3 + 2], k as i64, "k={k}: directed e_(nr,nr)");
    }
}

/// Defect: `entries.hh:283` guards the `nr` half of the self-loop correction
/// with `if constexpr (!single)` and leaves the `r` half unguarded, so a
/// `single` move over an undirected self-loop records `+w` on `(r, r)` for a
/// transition that changes nothing. The port drops the guard, and the two
/// halves then cancel on the one cell they share.
#[test]
fn a_no_op_move_over_a_self_loop_records_nothing() {
    let mut adj = AdjList::with_vertices(3);
    adj.add_edge(vid(0), vid(0)).expect("v exists");
    adj.add_edge(vid(0), vid(1)).expect("v exists");
    adj.add_edge(vid(0), vid(0)).expect("v exists");
    adj.add_edge(vid(2), vid(0)).expect("v exists");
    let c = Case {
        adj,
        b: vec![0, 1, 0],
        nb: vec![0, 1, 0],
        slots: 2,
        v: 0,
        nr: 0,
    };

    let mut st = build::<Undirected>(&c.adj, &c.b, c.slots);
    let before = st.aggregates().image();
    let got = run_undirected(&c, &mut st, move_key(&c), true);
    assert!(
        got.iter().all(|&(_, _, d)| d == 0),
        "a `single` move must record only zero deltas, got {got:?}"
    );
    assert_eq!(
        st.aggregates().image(),
        before,
        "committing a no-op move changed the state"
    );

    let mut st = build::<Directed>(&c.adj, &c.b, c.slots);
    let before = st.aggregates().image();
    let got = run_directed(&c, &mut st, move_key(&c), true);
    assert!(got.iter().all(|&(_, _, d)| d == 0), "{got:?}");
    assert_eq!(st.aggregates().image(), before);
}

// ===========================================================================
// The header
// ===========================================================================

/// `record` fills the six scalars `entries_dS` reads (`state.hh:1239-1253`)
/// plus the degree deltas, with `dkin`/`dkout` counted the way
/// `state.hh:220-222` counts them: `in_degree` is zero on an undirected view
/// (`graph_adaptor.hh:322-330`) and `out_degree` there is the whole incidence
/// run, so a self-loop counts twice.
#[test]
fn the_header_carries_the_before_image_and_the_degrees() {
    let mut adj = AdjList::with_vertices(3);
    adj.add_edge(vid(0), vid(1)).expect("v exists");
    adj.add_edge(vid(0), vid(0)).expect("v exists");
    adj.add_edge(vid(2), vid(0)).expect("v exists");
    let c = Case {
        adj,
        b: vec![0, 1, 1],
        nb: vec![0, 1, 1],
        slots: 3,
        v: 0,
        nr: 2,
    };

    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let st = build::<Directed>(&c.adj, &c.b, c.slots);
    let nb = |u: VertexId| Group::new(c.nb[u.index()]);
    let t = record(
        &st,
        &c.adj,
        vid(0),
        move_key(&c),
        &nb,
        c.slots,
        1i64,
        &mut ws,
    )
    .seal();
    let h = *t.level(0).header();
    assert_eq!((h.r, h.nr), (Some(grp(0)), Some(grp(2))));
    assert_eq!(
        (h.dkout, h.dkin),
        (2, 2),
        "out {{0->1, 0->0}}, in {{2->0, 0->0}}: a directed self-loop is one \
         out-edge and one in-edge"
    );
    assert_eq!((h.dr, h.dnr), (1, 1));
    assert_eq!(h.r_img.wr, 1, "group 0 holds v alone");
    assert_eq!(h.r_img.mrp, st.aggregates().mrp(grp(0)));
    assert_eq!(h.r_img.mrm, st.aggregates().mrm(grp(0)));
    assert_eq!(h.nr_img.wr, 0, "group 2 is empty before the move");

    let mut ws = Workspace::<Undirected, i64>::with_levels(1);
    let st = build::<Undirected>(&c.adj, &c.b, c.slots);
    let t = record(
        &st,
        c.adj.undirect(),
        vid(0),
        move_key(&c),
        &nb,
        c.slots,
        1i64,
        &mut ws,
    )
    .seal();
    let h = *t.level(0).header();
    assert_eq!(
        (h.dkout, h.dkin),
        (4, 0),
        "an undirected self-loop contributes two to the degree, and \
         `in_degree` on an undirected view is zero"
    );
}

// ===========================================================================
// The null group
// ===========================================================================

/// `auto s = b[u]` (`entries.hh:250`) is unchecked. Here it is reported.
#[test]
#[should_panic(expected = "has no group under `b`")]
fn a_neighbour_with_no_group_is_reported_not_skipped() {
    let mut adj = AdjList::with_vertices(2);
    adj.add_edge(vid(0), vid(1)).expect("v exists");
    let mut st = BlockState::<Directed, i64>::new(2, 2);
    st.assign(vid(0), grp(0), 1);
    // Vertex 1 is left unassigned: `_b[1]` is the null group.
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let nb = |u: VertexId| st.group_of(u);
    let _ = record(
        &st,
        &adj,
        vid(0),
        MoveKey {
            from: Some(grp(0)),
            to: Some(grp(1)),
        },
        &nb,
        2,
        1i64,
        &mut ws,
    );
}

/// The same for the ad-hoc "after" map, which the C++ reads at `:259`.
#[test]
#[should_panic(expected = "has no group under `nb`")]
fn a_neighbour_with_no_new_group_is_reported_not_skipped() {
    let mut adj = AdjList::with_vertices(2);
    adj.add_edge(vid(0), vid(1)).expect("v exists");
    let mut st = BlockState::<Directed, i64>::new(2, 2);
    st.assign(vid(0), grp(0), 1);
    st.assign(vid(1), grp(1), 1);
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let nb = |_: VertexId| None;
    let _ = record(
        &st,
        &adj,
        vid(0),
        MoveKey {
            from: Some(grp(0)),
            to: Some(grp(1)),
        },
        &nb,
        2,
        1i64,
        &mut ws,
    );
}

/// With `r` absent -- the insertion move of [`MoveHeader::r`] -- `b[u]` is
/// never *used*, and a read that the C++ performs and discards must not
/// become a panic the port performs and keeps.
#[test]
fn an_insertion_move_does_not_read_the_old_group() {
    let mut adj = AdjList::with_vertices(2);
    adj.add_edge(vid(0), vid(1)).expect("v exists");
    let mut st = BlockState::<Directed, i64>::new(2, 2);
    st.assign(vid(1), grp(1), 1);
    // Vertex 0 -- the one being *inserted* -- has no group, and neither does
    // its own `b[0]`; only `nb` is consulted.
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let nb = |u: VertexId| st.group_of(u);
    let t = record(
        &st,
        &adj,
        vid(0),
        MoveKey {
            from: None,
            to: Some(grp(0)),
        },
        &nb,
        2,
        1i64,
        &mut ws,
    )
    .seal();
    let e: Vec<_> = t
        .level(0)
        .entries()
        .iter()
        .map(|e| (e.r, e.s, e.delta))
        .collect();
    assert_eq!(e, vec![(grp(0), grp(1), 1)]);
    assert_eq!(t.level(0).header().r, None);
}

// ===========================================================================
// propagate
// ===========================================================================

/// A two-level hierarchy: four groups below, two above, `{0,1} -> 0` and
/// `{2,3} -> 1`.
fn hierarchy() -> (AdjList, Vec<u32>, Vec<u32>) {
    let mut adj = AdjList::with_vertices(5);
    for &(s, t) in &[(0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (0, 0), (0, 2)] {
        adj.add_edge(vid(s), vid(t)).expect("v exists");
    }
    // Vertex -> group below.
    let b = vec![0u32, 1, 2, 3, 2];
    // Group below -> group above.
    let up = vec![0u32, 0, 1, 1];
    (adj, b, up)
}

#[test]
fn propagate_projects_every_entry_and_stays_in_plane() {
    let (adj, b, up) = hierarchy();
    let c = Case {
        adj,
        b: b.clone(),
        nb: b.clone(),
        slots: 4,
        v: 0,
        nr: 3,
    };
    let st = build::<Directed>(&c.adj, &c.b, 4);
    // The level above: its "vertices" are the level below's groups.
    let above = build::<Directed>(&AdjList::with_vertices(4), &up, 2);

    let mut ws = Workspace::<Directed, i64>::with_levels(2);
    let nb = |u: VertexId| Group::new(c.nb[u.index()]);
    let mut rec = record(&st, &c.adj, vid(0), move_key(&c), &nb, 4, 1i64, &mut ws);

    let below: Vec<_> = rec
        .level_mut(0)
        .entries()
        .iter()
        .map(|e| (e.r, e.s, e.delta))
        .collect();
    assert!(!below.is_empty());

    let b_of = |r: Group| Group::new(up[r.index()]);
    let mut resolve = |r: Group, s: Group| above.resolve(r, s);
    propagate(&mut rec, 0, &b_of, 2, &mut resolve).expect("a well-formed hierarchy is in plane");

    let mut want = [0i64; 4];
    for (r, s, d) in below {
        want[up[r.index()] as usize * 2 + up[s.index()] as usize] += d;
    }
    let top = rec.level_mut(1);
    for x in 0..2u32 {
        for y in 0..2u32 {
            assert_eq!(
                top.delta_of(grp(x), grp(y)),
                want[x as usize * 2 + y as usize],
                "projected pair ({x}, {y})"
            );
        }
    }
    // The projected move key: `{_b[r], _b[nr]}` (`state.hh:1103-1106`).
    assert_eq!(top.move_key().from, Some(grp(0)));
    assert_eq!(top.move_key().to, Some(grp(1)));
}

/// `if (delta == 0) return;` (`state.hh:1114`). A `single` move records only
/// zero deltas, so nothing reaches the level above -- not even an empty entry
/// holding a field cell.
#[test]
fn propagate_skips_zero_deltas() {
    let (adj, b, up) = hierarchy();
    let c = Case {
        adj,
        b: b.clone(),
        nb: b.clone(),
        slots: 4,
        v: 0,
        nr: 0,
    };
    let st = build::<Directed>(&c.adj, &c.b, 4);
    let above = build::<Directed>(&AdjList::with_vertices(4), &up, 2);
    let mut ws = Workspace::<Directed, i64>::with_levels(2);
    let nb = |u: VertexId| Group::new(c.nb[u.index()]);
    let mut rec = record(&st, &c.adj, vid(0), move_key(&c), &nb, 4, 1i64, &mut ws);
    assert!(rec.level_mut(0).entries().iter().all(|e| e.delta == 0));

    let b_of = |r: Group| Group::new(up[r.index()]);
    let mut resolve = |r: Group, s: Group| above.resolve(r, s);
    propagate(&mut rec, 0, &b_of, 2, &mut resolve).expect("in plane");
    assert!(rec.level_mut(1).entries().is_empty());
    assert!(rec.level_mut(1).table_is_clear());
}

/// The acceptance criterion: no heap allocation per call, once the buffers are
/// warm. The first call grows the entry vector and the field table; the
/// second must touch neither.
#[test]
fn propagate_allocates_nothing_in_steady_state() {
    let (adj, b, up) = hierarchy();
    let c = Case {
        adj,
        b: b.clone(),
        nb: b.clone(),
        slots: 4,
        v: 0,
        nr: 3,
    };
    let st = build::<Directed>(&c.adj, &c.b, 4);
    let above = build::<Directed>(&AdjList::with_vertices(4), &up, 2);
    let mut ws = Workspace::<Directed, i64>::with_levels(2);
    let nb = |u: VertexId| Group::new(c.nb[u.index()]);
    let mut rec = record(&st, &c.adj, vid(0), move_key(&c), &nb, 4, 1i64, &mut ws);

    let b_of = |r: Group| Group::new(up[r.index()]);
    let mut resolve = |r: Group, s: Group| above.resolve(r, s);

    propagate(&mut rec, 0, &b_of, 2, &mut resolve).expect("in plane");
    let (out, n) = allocations(|| propagate(&mut rec, 0, &b_of, 2, &mut resolve));
    out.expect("in plane");
    assert_eq!(n, 0, "the warm `propagate` allocated {n} times");
}

/// The third acceptance criterion: `propagate` on a well-formed hierarchy
/// never returns `Err(OutOfPlane)`.
///
/// It is not luck, and the sweep below is only the evidence. Every entry a
/// [`DeltaBuf`] holds contains an endpoint of its move key -- that is what
/// makes the O(1) field lookup valid -- and the projected key is the *image*
/// of that key under the same `b_of`. A function image of an endpoint is
/// still an endpoint, so the projected pair contains the projected key's
/// first or second group and `get_field` matches. The C++'s `_dummy`
/// fallthrough (`entries.hh:118`) is therefore dead code guarded by an
/// invariant stated nowhere; here the invariant is the return type, and this
/// test walks 300 random hierarchies -- including projections that collapse
/// every group onto one -- to show the `Err` arm is never taken.
#[test]
fn propagate_is_never_out_of_plane_on_a_well_formed_hierarchy() {
    let mut rng = Lcg::new(0xbeef_0025);
    for i in 0..300 {
        let mut c = case(&mut rng);
        c.nb = c.b.clone();
        let above_slots = 1 + rng.below(3) as usize;
        let up: Vec<u32> = (0..c.slots)
            .map(|_| rng.below(above_slots as u32))
            .collect();

        let st = build::<Directed>(&c.adj, &c.b, c.slots);
        let above = build::<Directed>(&AdjList::with_vertices(c.slots), &up, above_slots);
        let mut ws = Workspace::<Directed, i64>::with_levels(2);
        let nb = |u: VertexId| Group::new(c.nb[u.index()]);
        let mut rec = record(
            &st,
            &c.adj,
            vid(c.v),
            move_key(&c),
            &nb,
            c.slots,
            1i64,
            &mut ws,
        );

        let below: Vec<_> = rec
            .level_mut(0)
            .entries()
            .iter()
            .map(|e| (e.r, e.s, e.delta))
            .collect();

        let b_of = |r: Group| Group::new(up[r.index()]);
        let mut resolve = |r: Group, s: Group| above.resolve(r, s);
        propagate(&mut rec, 0, &b_of, above_slots, &mut resolve)
            .unwrap_or_else(|e| panic!("case {i}: {e}"));

        // And the projection is the sum it claims to be.
        let mut want = vec![0i64; above_slots * above_slots];
        for (r, s, d) in below {
            want[up[r.index()] as usize * above_slots + up[s.index()] as usize] += d;
        }
        let top = rec.level_mut(1);
        for x in 0..above_slots as u32 {
            for y in 0..above_slots as u32 {
                assert_eq!(
                    top.delta_of(grp(x), grp(y)),
                    want[x as usize * above_slots + y as usize],
                    "case {i}: projected pair ({x}, {y})"
                );
            }
        }
    }
}

/// A group with no image one level up is reported, for the same reason a
/// neighbour with no group is.
#[test]
#[should_panic(expected = "has no image one level up")]
fn an_unprojectable_group_is_reported() {
    let (adj, b, up) = hierarchy();
    let c = Case {
        adj,
        b: b.clone(),
        nb: b.clone(),
        slots: 4,
        v: 0,
        nr: 3,
    };
    let st = build::<Directed>(&c.adj, &c.b, 4);
    let above = build::<Directed>(&AdjList::with_vertices(4), &up, 2);
    let mut ws = Workspace::<Directed, i64>::with_levels(2);
    let nb = |u: VertexId| Group::new(c.nb[u.index()]);
    let mut rec = record(&st, &c.adj, vid(0), move_key(&c), &nb, 4, 1i64, &mut ws);

    let b_of = |r: Group| {
        if r.index() == 2 {
            None
        } else {
            Group::new(up[r.index()])
        }
    };
    let mut resolve = |r: Group, s: Group| above.resolve(r, s);
    let _ = propagate(&mut rec, 0, &b_of, 2, &mut resolve);
}
