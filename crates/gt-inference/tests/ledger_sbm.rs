//! `gt_core::design` section 12, pinned: the SBM move cycle allocates nothing in
//! steady state.
//!
//! Section 12's "Wins" list claims that the before-image removes "two
//! cache-cold dependent loads per entry from `entries_dS`, the hottest kernel
//! in the library", and section 12's `Losses` list is silent about allocation
//! in the move cycle -- which is a claim, because graph-tool's own
//! `modify_entries` reuses `_m_entries` for exactly the same reason and the
//! port's `DeltaBuf::begin` (`delta/buf.rs:149`) drains and clears rather than
//! freeing precisely so that it can.
//!
//! A ledger entry that nobody re-derives rots. `benches/sbm.rs` prices the
//! cycle in nanoseconds; this file pins the structural half in a unit the
//! benchmark cannot express, because criterion measures time and the claim is
//! about the allocator.
//!
//! Companion to `gt-core/tests/u11_reduce.rs::the_ledger_records_whether_chunked_sum_vectorises`,
//! which does the same job for section 12's floating-point paragraph.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use gt_core::adj::AdjList;
use gt_core::dir::Directed;
use gt_core::ids::VertexId;
use gt_inference::blockmodel::{BlockCommit, BlockState, Cache, EntropyParams, record, sparse_ds};
use gt_inference::delta::{MoveKey, Workspace};
use gt_inference::ids::Group;

// ---------------------------------------------------------------------------
// An allocation counter.
//
// There is no safe API that observes a call to the global allocator, and the
// claim under test is a statement about allocation rather than about time.
// This is the only `unsafe` in the file; it is in a test binary, a separate
// crate from `gt-inference`, which therefore does not carry the library's
// `#![forbid(unsafe_code)]`, and every operation forwards to `System`.
//
// The tally is thread-local because `cargo test` runs this binary's tests
// concurrently. Same shape as `gt-core/tests/u05_adjlist.rs`.
// ---------------------------------------------------------------------------

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

/// The counter is installed and non-vacuous, or every assertion below is
/// trivially true.
#[test]
fn the_allocation_counter_counts() {
    let (v, n) = allocations(|| vec![0u8; 4096]);
    assert_eq!(v.len(), 4096);
    assert!(n >= 1, "a 4 KiB Vec must reach the global allocator");

    let (_, none) = allocations(|| 1u64 + 1);
    assert_eq!(none, 0, "arithmetic must not reach the allocator");
}

// ---------------------------------------------------------------------------
// The fixture
// ---------------------------------------------------------------------------

const N: usize = 2_000;
const M: usize = 10_000;
const SLOTS: usize = 32;

struct Stream(u64);

impl Stream {
    const fn new(seed: u64) -> Self {
        Stream(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn grp(i: usize) -> Group {
    Group::new(i as u32).expect("group index in range")
}

fn vid(i: usize) -> VertexId {
    VertexId::from_index(i)
}

fn graph() -> AdjList {
    let mut s = Stream::new(0x5EED_11ED);
    let mut g = AdjList::with_vertices(N);
    for i in 0..N {
        g.add_edge(vid(i), vid((i + 1) % N)).expect("add_edge");
    }
    for _ in N..M {
        let a = s.below(N);
        let b = if s.below(5) == 0 { a } else { s.below(N) };
        g.add_edge(vid(a), vid(b)).expect("add_edge");
    }
    g
}

fn seed(g: &AdjList, b: &[u32]) -> BlockState<Directed, i64> {
    let mut st = BlockState::<Directed, i64>::new(SLOTS, b.len());
    for (i, &r) in b.iter().enumerate() {
        st.assign(vid(i), grp(r as usize), 1);
    }
    for e in g.edges() {
        st.seed_pair(
            grp(b[e.source().index()] as usize),
            grp(b[e.target().index()] as usize),
            1,
        );
    }
    st
}

fn params() -> EntropyParams {
    EntropyParams {
        deg_corr: true,
        ..EntropyParams::default()
    }
}

/// One record -> price -> commit -> reseat cycle against a reused workspace.
fn cycle(
    st: &mut BlockState<Directed, i64>,
    g: &AdjList,
    b: &mut [u32],
    ws: &mut Workspace<Directed, i64>,
    cache: &Cache,
    v: usize,
    nr: usize,
) -> f64 {
    let mv = MoveKey {
        from: Some(grp(b[v] as usize)),
        to: Some(grp(nr)),
    };
    let nb = |u: VertexId| Group::new(b[u.index()]);
    let t = record(&*st, g, vid(v), mv, &nb, SLOTS, 1i64, ws).seal();
    let ds = sparse_ds(t.level(0), params(), cache);
    let _ = st.commit(t.into_levels().next().expect("one level"));
    st.reseat(vid(v), grp(nr));
    b[v] = nr as u32;
    ds
}

/// Where the move cycle's allocations actually are, stage by stage.
///
/// Section 12's "Wins" list credits the before-image for removing two
/// cache-cold loads per entry from `entries_dS`, and says nothing about
/// allocation in the cycle around it -- which reads as a claim, because
/// `DeltaBuf::begin` (`delta/buf.rs:149`) drains and clears rather than
/// freeing for exactly that reason, and graph-tool's `modify_entries` reuses
/// `_m_entries` for exactly that reason.
///
/// Measured, it is true of three of the four stages and false of the fourth:
/// [`Receipt`](gt_inference::delta::Receipt) owns
/// `entries: Vec<Entry<W>>` (`delta/lifecycle.rs:203`), so **every commit
/// allocates once and frees once**, whatever the workspace does. That is the
/// price of forwarding the before-image past the commit so that
/// `audit_commit` can run at all -- a deliberate trade, recorded in section
/// 12 rather than denied.
///
/// This test pins the split, so that a regression in `begin` (which would
/// show up as allocations in `record`) cannot hide behind the commit's known
/// one.
#[test]
fn the_move_cycle_allocates_once_per_commit_and_nowhere_else() {
    let g = graph();
    let mut b: Vec<u32> = (0..N).map(|i| (i % SLOTS) as u32).collect();
    let mut st = seed(&g, &b);
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let cache = Cache::build(4096);

    // Warm-up: touch every vertex once, so the entry vector has grown to the
    // largest neighbourhood in the graph and the field table to SLOTS.
    let mut warm = Stream::new(0xC0FF_EE01);
    for v in 0..N {
        let nr = warm.below(SLOTS);
        cycle(&mut st, &g, &mut b, &mut ws, &cache, v, nr);
    }
    for _ in 0..20_000 {
        let v = warm.below(N);
        let nr = warm.below(SLOTS);
        cycle(&mut st, &g, &mut b, &mut ws, &cache, v, nr);
    }

    const MOVES: usize = 10_000;
    let mut s = Stream::new(0xC0FF_EE02);
    let mut acc = 0.0f64;
    let mut a_record = 0usize;
    let mut a_price = 0usize;
    let mut a_commit = 0usize;
    let mut a_reseat = 0usize;

    for _ in 0..MOVES {
        let v = s.below(N);
        let nr = s.below(SLOTS);
        let mv = MoveKey {
            from: Some(grp(b[v] as usize)),
            to: Some(grp(nr)),
        };

        // -- record ------------------------------------------------------
        // The borrow of `b` inside `nb` has to end before `b[v] = nr`, so
        // each stage is scoped rather than held.
        let (t, n) = {
            let nb = |u: VertexId| Group::new(b[u.index()]);
            allocations(|| record(&st, &g, vid(v), mv, &nb, SLOTS, 1i64, &mut ws).seal())
        };
        a_record += n;

        // -- price -------------------------------------------------------
        let (ds, n) = allocations(|| sparse_ds(t.level(0), params(), &cache));
        a_price += n;
        acc += ds;

        // -- commit ------------------------------------------------------
        let (receipt, n) = allocations(|| st.commit(t.into_levels().next().expect("one level")));
        a_commit += n;
        // Dropping the receipt is the matching free; keep it alive past the
        // count so the tally is allocations, not churn.
        assert!(!receipt.entries.is_empty() || receipt.hdr.dkout == 0);

        // -- reseat ------------------------------------------------------
        let (_, n) = allocations(|| st.reseat(vid(v), grp(nr)));
        a_reseat += n;
        b[v] = nr as u32;
    }

    assert!(
        acc.is_finite(),
        "the sweep must actually price something: {acc}"
    );

    // The three stages the ledger's silence is true of.
    assert_eq!(
        a_record, 0,
        "record allocated {a_record} times over {MOVES} moves; \
         DeltaBuf::begin is supposed to drain and clear, not free"
    );
    assert_eq!(a_price, 0, "sparse_ds allocated {a_price} times");
    assert_eq!(a_reseat, 0, "reseat allocated {a_reseat} times");

    // The one it is false of, pinned at exactly one per commit so that a
    // second one cannot appear unnoticed.
    assert_eq!(
        a_commit, MOVES,
        "commit is expected to allocate exactly once per move (Receipt::entries, \
         delta/lifecycle.rs:203); it allocated {a_commit} times over {MOVES} moves"
    );
}

/// `sparse_ds` on its own: a contiguous walk of `&[Entry<W>]`, no allocation
/// at any entry count.
///
/// Separated from the cycle so that a regression in the recorder cannot be
/// mistaken for one in the pricing kernel, which is the one section 12 names.
#[test]
fn entries_ds_allocates_nothing_at_any_entry_count() {
    let g = graph();
    let b: Vec<u32> = (0..N).map(|i| (i % SLOTS) as u32).collect();
    let st = seed(&g, &b);
    let cache = Cache::build(4096);

    // The widest neighbourhood in the graph, so the delta is as long as this
    // fixture can make it.
    let widest = (0..N).max_by_key(|&v| g.degree(vid(v))).expect("N > 0");

    let mut counts = Vec::new();
    for &v in &[0usize, N / 2, widest] {
        let mut ws = Workspace::<Directed, i64>::with_levels(1);
        let nr = (b[v] as usize + 1) % SLOTS;
        let mv = MoveKey {
            from: Some(grp(b[v] as usize)),
            to: Some(grp(nr)),
        };
        let nb = |u: VertexId| Group::new(b[u.index()]);
        let t = record(&st, &g, vid(v), mv, &nb, SLOTS, 1i64, &mut ws).seal();
        let delta = t.level(0);
        let n = delta.entries().len();

        // Once outside the count, so any lazily-initialised table in the
        // cache path is already warm.
        let first = sparse_ds(delta, params(), &cache);
        let (ds, allocs) = allocations(|| {
            let mut acc = 0.0f64;
            for _ in 0..1_000 {
                acc += sparse_ds(delta, params(), &cache);
            }
            acc
        });
        assert_eq!(
            allocs, 0,
            "sparse_ds over {n} entries allocated {allocs} times"
        );
        assert!(
            (ds / 1_000.0 - first).abs() <= first.abs() * 1e-12 + 1e-12,
            "sparse_ds is not a pure function of its delta"
        );
        counts.push(n);
    }

    // The three rungs must genuinely differ, or the loop above proved nothing
    // about how the entry count scales.
    assert!(
        counts.iter().max() > counts.iter().min(),
        "all three deltas had the same entry count: {counts:?}"
    );
}
