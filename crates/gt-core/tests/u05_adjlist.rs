//! U5 -- `AdjList`, the two splice primitives, and the always-on audit.
//!
//! The centrepiece is the model-based proptest: random sequences of
//! `add_edge` / `remove_edge` / `clear_vertex` / `clear_vertex_where` /
//! `swap_remove_vertex` over at most six vertices, with self-loops and
//! parallel edges, run against a `BTreeMap<EdgeId, (V, V)>` reference. After
//! **every** operation three things are checked:
//!
//! * `validate()` is `Ok` -- `check_epos` (`graph_adjacency.hh:686-718`)
//!   running, which in graph-tool it never does: `:698`, `:1226`, `:1277`,
//!   `:1306` and `:1433` are all commented out;
//! * `num_edges()` equals the reference's cardinality -- the assertion
//!   defect #1 fails (`:1403-1410` decrements `_n_edges` by two for one
//!   removed edge, by counting `std::remove_if`'s moved-from tail, which by
//!   that algorithm's definition holds the *kept* elements);
//! * the whole sorted **out-half and in-half** of every vertex match the
//!   reference. Both halves, because the in-half is where a splice that gets
//!   the boundary promotion wrong actually goes wrong, and a `validate` that
//!   inspects only the out-half reports success while the in-half rots.
//!
//! The sequence is replayed against `AdjList<NoLookup>` and `AdjList<EHash>`
//! independently, because the `(s,t)` index is a second derived structure
//! with its own hooks and its own way of being left behind (defect #3).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;

use gt_core::adj::{AdjList, EHash, Lookup, NoLookup};
use gt_core::ids::{EdgeId, VertexId};
use proptest::prelude::*;

// ===========================================================================
// An allocation counter
//
// There is no safe API that observes a call to the global allocator, and the
// unit's acceptance criterion is a statement about allocation ("10 000
// `clear_vertex` calls perform zero allocations after warm-up"), not about
// time. This is the one `unsafe` block in the unit; it is in a test binary,
// which is a separate crate from `gt-core` and does not carry the library's
// `#![forbid(unsafe_code)]`, and every operation in it forwards to `System`.
//
// The tally is thread-local: `cargo test` runs the tests in this binary
// concurrently, so a process-wide counter would be measuring proptest's
// allocations as often as its own.
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
// Helpers
// ===========================================================================

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

fn empty<H: Lookup>(n: usize) -> AdjList<H> {
    let mut g = AdjList::with_lookup(H::default());
    for _ in 0..n {
        g.add_vertex().expect("vertex index space");
    }
    g
}

/// `(edge id, other endpoint)` for one half, sorted: what the half holds, with
/// the swap order deliberately thrown away.
fn half(it: impl Iterator<Item = gt_core::adj::Incident>) -> Vec<(usize, usize)> {
    let mut s: Vec<_> = it.map(|i| (i.edge.index(), i.other.index())).collect();
    s.sort_unstable();
    s
}

// ===========================================================================
// 1. The two primitives, in their smallest forms
// ===========================================================================

#[test]
fn an_empty_graph_is_valid_and_empty() {
    let g = AdjList::new();
    assert_eq!(g.num_vertices(), 0);
    assert_eq!(g.num_edges(), 0);
    assert!(g.validate().is_ok());
    assert_eq!(g.out_edges(v(0)).count(), 0, "an absent vertex is empty");
    assert_eq!(g.degree(v(7)), 0);
    assert!(g.find_edge(v(0), v(1)).is_none());
}

#[test]
fn with_vertices_is_n_isolated_vertices() {
    let g = AdjList::with_vertices(5);
    assert_eq!(g.num_vertices(), 5);
    assert_eq!(g.num_edges(), 0);
    assert_eq!(g.vertices().count(), 5);
    assert!(g.vertices().all(|u| g.degree(u) == 0));
    assert!(g.validate().is_ok());
}

/// `add_edge` links both halves and both indexes; the entry in each half names
/// the *other* endpoint (D2), so the two halves of a self-loop are not a
/// special case anywhere.
#[test]
fn add_edge_links_both_halves() {
    let mut g = AdjList::with_vertices(3);
    let e = g.add_edge(v(0), v(1)).expect("add_edge");
    assert_eq!(e.source(), v(0));
    assert_eq!(e.target(), v(1));
    assert_eq!(g.num_edges(), 1);
    assert_eq!(g.endpoints(e.id()), Some((v(0), v(1))));

    assert_eq!(half(g.out_edges(v(0))), vec![(e.id().index(), 1)]);
    assert_eq!(half(g.in_edges(v(0))), vec![]);
    assert_eq!(half(g.out_edges(v(1))), vec![]);
    assert_eq!(half(g.in_edges(v(1))), vec![(e.id().index(), 0)]);
    assert_eq!((g.out_degree(v(0)), g.in_degree(v(1))), (1, 1));
    assert_eq!(g.degree(v(0)), 1);
    assert!(g.validate().is_ok());
}

/// A failed `add_edge` must not consume an edge index. `add_edge`
/// (`graph_adjacency.hh:1190-1193`) calls `get_free_idx()` *first* and only
/// then indexes `g._edges[s]`.
#[test]
fn add_edge_checks_its_endpoints_before_taking_an_index() {
    let mut g = AdjList::with_vertices(2);
    let before = g.edge_bound().len();
    assert_eq!(
        g.add_edge(v(0), v(9)).err(),
        Some(gt_core::GraphError::NoSuchVertex(v(9)))
    );
    assert_eq!(
        g.add_edge(v(9), v(0)).err(),
        Some(gt_core::GraphError::NoSuchVertex(v(9)))
    );
    assert_eq!(g.edge_bound().len(), before, "an index was consumed");
    assert_eq!(g.num_edges(), 0);
    assert!(g.validate().is_ok());
}

/// A self-loop is one edge with two entries in one block, which is the
/// aliasing `:1252-1257` hides behind `s_es` and `t_es`. Removing it must
/// leave nothing behind in either half.
#[test]
fn a_self_loop_occupies_both_halves_of_one_block() {
    let mut g = AdjList::with_vertices(1);
    let e = g.add_edge(v(0), v(0)).expect("add_edge");
    assert_eq!(g.out_degree(v(0)), 1);
    assert_eq!(g.in_degree(v(0)), 1);
    assert_eq!(g.degree(v(0)), 2);
    assert_eq!(g.num_edges(), 1, "two entries, one edge");
    assert!(g.validate().is_ok());

    g.remove_edge(e.id()).expect("remove_edge");
    assert_eq!(g.degree(v(0)), 0);
    assert_eq!(g.num_edges(), 0);
    assert!(g.validate().is_ok());
}

/// Removing a self-loop's out-entry promotes the loop's **own** in-entry
/// across the out/in boundary. `splice_out` therefore has to re-locate the
/// in-half *after* the out splice; a position read before it is stale by
/// exactly one slot. Two loops plus a witness edge so that the boundary is
/// actually crossed.
#[test]
fn splice_out_relocates_the_in_half_before_it_reads_it() {
    let mut g = AdjList::with_vertices(2);
    let loop_a = g.add_edge(v(0), v(0)).expect("add_edge").id();
    let loop_b = g.add_edge(v(0), v(0)).expect("add_edge").id();
    let other = g.add_edge(v(1), v(0)).expect("add_edge").id();
    assert!(g.validate().is_ok());
    assert_eq!((g.out_degree(v(0)), g.in_degree(v(0))), (2, 3));

    g.remove_edge(loop_a).expect("remove_edge");
    assert!(g.validate().is_ok());
    assert_eq!(g.num_edges(), 2);
    assert_eq!(half(g.out_edges(v(0))), vec![(loop_b.index(), 0)]);
    assert_eq!(half(g.in_edges(v(0))), {
        let mut w = vec![(loop_b.index(), 0), (other.index(), 1)];
        w.sort_unstable();
        w
    });

    g.remove_edge(loop_b).expect("remove_edge");
    assert!(g.validate().is_ok());
    assert_eq!(half(g.in_edges(v(0))), vec![(other.index(), 1)]);
    assert_eq!(g.out_degree(v(0)), 0);
    assert_eq!(g.num_edges(), 1);
}

#[test]
fn remove_edge_is_by_identity_and_rejects_a_dead_id() {
    let mut g = AdjList::with_vertices(2);
    let e = g.add_edge(v(0), v(1)).expect("add_edge").id();
    g.remove_edge(e).expect("remove_edge");
    assert_eq!(g.remove_edge(e), Err(gt_core::GraphError::NoSuchEdge(e)));
    assert_eq!(g.num_edges(), 0);
    assert!(g.validate().is_ok());
}

/// `remove_edge` takes an [`EdgeId`], so there is no orientation left for a
/// caller to get wrong: the id behind `find_edge(t, s)` -- which is `None`
/// here, the edge runs `s -> t` -- and the id behind `find_edge(s, t)`
/// reversed are the same id, and both remove the same edge.
///
/// This is the interaction `reverse_edge` (`graph_adjacency.hh:571`) leaves
/// open: it mutates a descriptor's `s`/`t` in place and `remove_edge` then
/// drives a `_epos` lookup with the swapped endpoints.
#[test]
fn remove_edge_of_a_reversed_descriptor_is_the_same_removal() {
    for reversed_first in [false, true] {
        let mut g = AdjList::with_vertices(2);
        let fwd = g.add_edge(v(0), v(1)).expect("add_edge");

        let found = g.find_edge(v(0), v(1)).expect("the edge is there");
        assert_eq!(found.id(), fwd.id());
        assert!(
            g.find_edge(v(1), v(0)).is_none(),
            "storage is oriented; `edge(t, s, g)` is a different question"
        );

        // `.reversed()` returns a value and keeps the identity.
        let rev = found.reversed();
        assert_eq!(rev.id(), fwd.id());
        assert_eq!((rev.source(), rev.target()), (v(1), v(0)));

        let doomed = if reversed_first { rev.id() } else { fwd.id() };
        g.remove_edge(doomed).expect("remove_edge");
        assert_eq!(g.num_edges(), 0);
        assert_eq!(g.degree(v(0)), 0);
        assert_eq!(g.degree(v(1)), 0);
        assert!(g.validate().is_ok());
    }
}

// ===========================================================================
// 2. `find_edge`
// ===========================================================================

/// `edge(s, t, g)` (`:952-971`) scans whichever of `out_edges(s)` and
/// `in_edges(t)` is shorter. Both arms must answer identically; the test
/// builds a graph where the choice actually flips.
#[test]
fn find_edge_scans_the_shorter_half_and_both_arms_agree() {
    // `padding` decides which of `out_degree(0)` and `in_degree(3)` is
    // smaller, so both arms of `:952-971` run over the same topology and must
    // return the same edge.
    for (out_pad, in_pad) in [(8usize, 0usize), (0, 8)] {
        let mut g: AdjList<NoLookup> = empty(4);
        let wanted = g.add_edge(v(0), v(3)).expect("add_edge").id();
        for _ in 0..out_pad {
            g.add_edge(v(0), v(2)).expect("add_edge");
        }
        for _ in 0..in_pad {
            g.add_edge(v(1), v(3)).expect("add_edge");
        }
        assert_ne!(
            g.out_degree(v(0)),
            g.in_degree(v(3)),
            "the two arms are not distinguished"
        );

        let hit = g.find_edge(v(0), v(3)).expect("present");
        assert_eq!(hit.id(), wanted);
        assert_eq!((hit.source(), hit.target()), (v(0), v(3)));
        assert!(g.find_edge(v(3), v(0)).is_none());
        assert!(g.find_edge(v(2), v(1)).is_none());
        assert!(g.find_edge(v(0), v(99)).is_none(), "out of range is a miss");
        assert!(g.validate().is_ok());
    }
}

/// With an `(s,t)` index the answer is the bucket's first id
/// (`iter->second.front()`, `:949`), and it must be an id the adjacency
/// really holds.
#[test]
fn find_edge_with_a_lookup_agrees_with_the_scan() {
    let mut plain: AdjList<NoLookup> = empty(4);
    let mut hashed: AdjList<EHash> = empty(4);
    let pairs = [(0, 1), (0, 1), (2, 2), (3, 0), (0, 1), (2, 2)];
    for &(s, t) in &pairs {
        let a = plain.add_edge(v(s), v(t)).expect("add_edge").id();
        let b = hashed.add_edge(v(s), v(t)).expect("add_edge").id();
        assert_eq!(a, b, "both allocators issue the same dense ids");
    }
    for s in 0..4 {
        for t in 0..4 {
            assert_eq!(
                plain.find_edge(v(s), v(t)).map(|e| e.id()),
                hashed.find_edge(v(s), v(t)).map(|e| e.id()),
                "the scan and the index disagree about ({s},{t})"
            );
        }
    }
    assert!(plain.validate().is_ok());
    assert!(hashed.validate().is_ok());
}

// ===========================================================================
// 3. `clear_vertex_where` -- defect #1
// ===========================================================================

/// **Defect #1.** `clear_vertex` with `_keep_epos == false`
/// (`graph_adjacency.hh:1343-1414`) computes
///
/// ```c++
/// iter = std::remove_if(es.begin(), es.begin() + pos, pred);
/// k += std::count_if(iter, es.begin() + pos, [](auto& e){ return e.first != v; });
/// ```
///
/// -- counting over `[iter, pos)`, which after `std::remove_if` is the
/// *moved-from tail* and holds unspecified (in practice: the kept) elements.
/// A vertex with a self-loop plus one other out-edge, under a predicate that
/// matches only the self-loop, therefore decrements `_n_edges` by **two** for
/// **one** removed edge.
///
/// Here `num_edges()` is `EdgeIds::live()`, moved only by `alloc` and
/// `release`, so the count cannot be computed wrongly by a caller.
#[test]
fn clearing_only_a_self_loop_removes_exactly_one_edge() {
    let mut g = AdjList::with_vertices(2);
    let loop_ = g.add_edge(v(0), v(0)).expect("add_edge").id();
    let other = g.add_edge(v(0), v(1)).expect("add_edge").id();
    assert_eq!(g.num_edges(), 2);
    assert_eq!(g.degree(v(0)), 3, "the loop is two entries");

    // The predicate matches the self-loop and nothing else.
    g.clear_vertex_where(v(0), |_, other_end| other_end == v(0))
        .expect("clear_vertex_where");

    assert_eq!(g.num_edges(), 1, "defect #1: this is 0 in graph-tool");
    assert_eq!(g.endpoints(loop_), None);
    assert_eq!(g.endpoints(other), Some((v(0), v(1))));
    assert_eq!(half(g.out_edges(v(0))), vec![(other.index(), 1)]);
    assert_eq!(half(g.in_edges(v(0))), vec![]);
    assert_eq!(half(g.in_edges(v(1))), vec![(other.index(), 0)]);
    assert!(g.validate().is_ok());
}

/// The predicate is offered each incident edge exactly once -- including a
/// self-loop, which occupies two entries. `:1421-1429` evaluates `pred(ed)`
/// on both of a loop's entries and then discards the in-half occurrence
/// regardless of the answer, so a `FnMut` with a side effect sees it twice.
#[test]
fn the_predicate_sees_each_incident_edge_once() {
    let mut g = AdjList::with_vertices(3);
    let loop_ = g.add_edge(v(1), v(1)).expect("add_edge").id();
    let out = g.add_edge(v(1), v(2)).expect("add_edge").id();
    let inc = g.add_edge(v(0), v(1)).expect("add_edge").id();

    let mut seen: Vec<(usize, usize)> = Vec::new();
    g.clear_vertex_where(v(1), |id, other_end| {
        seen.push((id.index(), other_end.index()));
        false
    })
    .expect("clear_vertex_where");

    seen.sort_unstable();
    let mut want = vec![(loop_.index(), 1), (out.index(), 2), (inc.index(), 0)];
    want.sort_unstable();
    assert_eq!(seen, want, "a self-loop must be offered once, not twice");
    assert_eq!(g.num_edges(), 3, "a false predicate removes nothing");
    assert!(g.validate().is_ok());
}

#[test]
fn clear_vertex_removes_both_directions() {
    let mut g = AdjList::with_vertices(3);
    g.add_edge(v(0), v(1)).expect("add_edge");
    g.add_edge(v(1), v(0)).expect("add_edge");
    g.add_edge(v(1), v(1)).expect("add_edge");
    let survivor = g.add_edge(v(0), v(2)).expect("add_edge").id();
    assert_eq!(g.num_edges(), 4);

    g.clear_vertex(v(1)).expect("clear_vertex");
    assert_eq!(g.num_edges(), 1);
    assert_eq!(g.degree(v(1)), 0);
    assert_eq!(half(g.out_edges(v(0))), vec![(survivor.index(), 2)]);
    assert_eq!(g.num_vertices(), 3, "the vertex stays");
    assert!(g.validate().is_ok());
}

#[test]
fn clear_vertex_rejects_an_absent_vertex() {
    let mut g = AdjList::with_vertices(2);
    assert_eq!(
        g.clear_vertex(v(5)),
        Err(gt_core::GraphError::NoSuchVertex(v(5)))
    );
}

// ===========================================================================
// 4. `swap_remove_vertex` -- defect #3
// ===========================================================================

#[test]
fn swap_remove_moves_the_last_vertex_into_the_hole() {
    let mut g = AdjList::with_vertices(4);
    g.add_edge(v(0), v(1)).expect("add_edge");
    let kept = g.add_edge(v(3), v(2)).expect("add_edge").id();
    let loop_ = g.add_edge(v(3), v(3)).expect("add_edge").id();

    g.swap_remove_vertex(v(1)).expect("swap_remove_vertex");

    assert_eq!(g.num_vertices(), 3);
    assert_eq!(g.num_edges(), 2, "only (0,1) went away");
    // vertex 3 is now vertex 1, with its identities preserved.
    assert_eq!(g.endpoints(kept), Some((v(1), v(2))));
    assert_eq!(g.endpoints(loop_), Some((v(1), v(1))));
    assert_eq!(half(g.out_edges(v(1))), {
        let mut w = vec![(kept.index(), 2), (loop_.index(), 1)];
        w.sort_unstable();
        w
    });
    assert_eq!(half(g.in_edges(v(2))), vec![(kept.index(), 1)]);
    assert!(g.validate().is_ok());
}

/// **Defect #3.** `remove_vertex_fast` (`:1471-1535`) repairs `_ehash` with
/// `out_edges(back)` before the move and `out_edges(v)` after it, so an
/// *in*-neighbour of `back` -- a vertex `u` holding `back` as a **target** --
/// keeps a bucket keyed on `back`, a vertex that no longer exists.
/// `edge(u, v, g)` then answers false for an edge that is right there.
///
/// The port keys the index on the ordered pair and relabels by unlink-then-
/// relink, so both directions are notified.
#[test]
fn swap_remove_rekeys_the_lookup_for_in_neighbours() {
    let mut g: AdjList<EHash> = empty(4);
    // `back` (vertex 3) is the *target* here: exactly the direction the C++
    // never re-keys.
    let e = g.add_edge(v(2), v(3)).expect("add_edge").id();
    g.add_edge(v(0), v(1)).expect("add_edge");

    g.swap_remove_vertex(v(0)).expect("swap_remove_vertex");

    assert_eq!(g.num_vertices(), 3);
    assert_eq!(g.num_edges(), 1);
    assert_eq!(g.endpoints(e), Some((v(2), v(0))));
    assert_eq!(
        g.find_edge(v(2), v(0)).map(|x| x.id()),
        Some(e),
        "defect #3: the index is still keyed on the dead vertex"
    );
    assert!(g.find_edge(v(2), v(3)).is_none(), "the dead key survived");
    assert!(g.validate().is_ok());
}

#[test]
fn swap_remove_of_the_last_vertex_is_just_a_clear() {
    let mut g = AdjList::with_vertices(3);
    let kept = g.add_edge(v(0), v(1)).expect("add_edge").id();
    g.add_edge(v(2), v(0)).expect("add_edge");
    g.swap_remove_vertex(v(2)).expect("swap_remove_vertex");
    assert_eq!(g.num_vertices(), 2);
    assert_eq!(g.num_edges(), 1);
    assert_eq!(g.endpoints(kept), Some((v(0), v(1))));
    assert!(g.validate().is_ok());
}

#[test]
fn swap_remove_rejects_an_absent_vertex() {
    let mut g = AdjList::with_vertices(1);
    assert_eq!(
        g.swap_remove_vertex(v(3)),
        Err(gt_core::GraphError::NoSuchVertex(v(3)))
    );
    g.swap_remove_vertex(v(0)).expect("swap_remove_vertex");
    assert_eq!(g.num_vertices(), 0);
    assert!(g.validate().is_ok());
}

// ===========================================================================
// 5. Dense indices and `shrink_to_fit`
// ===========================================================================

/// The index space is recycled (`get_free_idx`, `:627-655`, LIFO) but the
/// *bound* never shrinks on its own -- which is what an edge property map is
/// sized to.
#[test]
fn edge_indices_are_recycled_and_the_bound_is_the_range() {
    let mut g = AdjList::with_vertices(2);
    let a = g.add_edge(v(0), v(1)).expect("add_edge").id();
    let b = g.add_edge(v(0), v(1)).expect("add_edge").id();
    assert_eq!((a.index(), b.index()), (0, 1));
    assert_eq!(g.edge_bound().len(), 2);

    g.remove_edge(a).expect("remove_edge");
    assert_eq!(g.num_edges(), 1);
    assert_eq!(g.edge_bound().len(), 2, "the range is sparse after removal");

    let c = g.add_edge(v(1), v(1)).expect("add_edge").id();
    assert_eq!(c, a, "the free list is LIFO");
    assert_eq!(g.edge_bound().len(), 2);
    assert!(g.validate().is_ok());
}

#[test]
fn shrink_to_fit_keeps_the_contents() {
    let mut g = AdjList::with_vertices(3);
    let mut ids = Vec::new();
    for i in 0..64 {
        ids.push(g.add_edge(v(i % 3), v((i + 1) % 3)).expect("add_edge").id());
    }
    for id in ids.drain(..32) {
        g.remove_edge(id).expect("remove_edge");
    }
    let before: Vec<_> = (0..3).map(|i| half(g.all_edges(v(i)))).collect();
    g.shrink_to_fit();
    let after: Vec<_> = (0..3).map(|i| half(g.all_edges(v(i)))).collect();
    assert_eq!(before, after);
    assert_eq!(g.num_edges(), 32);
    assert!(g.validate().is_ok());
}

/// `all_edges` is the out-half followed by the in-half, in that order
/// (`_all_edges_out`, `:1102-1108`), and it is anchored: `other` is the
/// neighbour in both halves.
#[test]
fn all_edges_is_the_out_half_then_the_in_half() {
    let mut g = AdjList::with_vertices(3);
    let out = g.add_edge(v(1), v(2)).expect("add_edge").id();
    let inc = g.add_edge(v(0), v(1)).expect("add_edge").id();

    let all: Vec<_> = g
        .all_edges(v(1))
        .map(|i| (i.edge.index(), i.other.index()))
        .collect();
    assert_eq!(all, vec![(out.index(), 2), (inc.index(), 0)]);
    assert_eq!(all.len(), g.degree(v(1)));
    assert_eq!(
        g.all_edges(v(1)).count(),
        g.out_edges(v(1)).count() + g.in_edges(v(1)).count()
    );
}

// ===========================================================================
// 6. The model
// ===========================================================================

#[derive(Clone, Copy, Debug)]
enum Op {
    AddVertex,
    AddEdge(u8, u8),
    RemoveEdge(u8),
    Clear(u8),
    ClearWhere(u8, u8),
    SwapRemove(u8),
}

/// The naive reference: a vertex count and `EdgeId -> (source, target)`.
#[derive(Clone, Debug, Default)]
struct Model {
    n: usize,
    edges: BTreeMap<EdgeId, (usize, usize)>,
}

/// The predicates `ClearWhere` chooses between. `other` is the endpoint that
/// is not `v` -- `v` itself for a self-loop.
fn pred_of(kind: u8, vertex: usize, id: EdgeId, other: usize) -> bool {
    match kind % 4 {
        0 => true,
        1 => other == vertex,
        2 => id.index().is_multiple_of(2),
        _ => false,
    }
}

impl Model {
    fn clear_where(&mut self, vertex: usize, kind: u8) {
        self.edges.retain(|&id, &mut (s, t)| {
            // The out-half occurrence wins for a self-loop, exactly as the
            // block scan does: `if j >= out_len && other == v { continue }`.
            let other = if s == vertex {
                t
            } else if t == vertex {
                s
            } else {
                return true;
            };
            !pred_of(kind, vertex, id, other)
        });
    }

    fn swap_remove(&mut self, vertex: usize) {
        self.edges
            .retain(|_, &mut (s, t)| s != vertex && t != vertex);
        let back = self.n - 1;
        self.n -= 1;
        if vertex != back {
            for e in self.edges.values_mut() {
                if e.0 == back {
                    e.0 = vertex;
                }
                if e.1 == back {
                    e.1 = vertex;
                }
            }
        }
    }
}

const MAX_VERTICES: usize = 6;

fn check<H: Lookup>(g: &AdjList<H>, m: &Model) -> Result<(), TestCaseError> {
    if let Err(e) = g.validate() {
        return Err(TestCaseError::fail(format!("validate: {e}")));
    }
    prop_assert_eq!(g.num_vertices(), m.n);
    prop_assert_eq!(g.num_edges(), m.edges.len());
    prop_assert_eq!(g.edges().count(), m.edges.len(), "the global edge list");

    for i in 0..m.n {
        let u = v(i);
        let mut want_out: Vec<(usize, usize)> = m
            .edges
            .iter()
            .filter(|&(_, &(s, _))| s == i)
            .map(|(&id, &(_, t))| (id.index(), t))
            .collect();
        want_out.sort_unstable();
        let mut want_in: Vec<(usize, usize)> = m
            .edges
            .iter()
            .filter(|&(_, &(_, t))| t == i)
            .map(|(&id, &(s, _))| (id.index(), s))
            .collect();
        want_in.sort_unstable();

        prop_assert_eq!(
            half(g.out_edges(u)),
            want_out.clone(),
            "out-half of {:?}",
            u
        );
        prop_assert_eq!(half(g.in_edges(u)), want_in.clone(), "in-half of {:?}", u);
        prop_assert_eq!(g.out_degree(u), want_out.len());
        prop_assert_eq!(g.in_degree(u), want_in.len());
        prop_assert_eq!(g.degree(u), want_out.len() + want_in.len());
    }

    // Endpoints, and `find_edge` against the same reference.
    for (&id, &(s, t)) in &m.edges {
        prop_assert_eq!(g.endpoints(id), Some((v(s), v(t))));
        prop_assert!(g.find_edge(v(s), v(t)).is_some());
    }
    Ok(())
}

fn run<H: Lookup>(ops: &[Op]) -> Result<(), TestCaseError> {
    let mut g: AdjList<H> = empty(MAX_VERTICES);
    let mut m = Model {
        n: MAX_VERTICES,
        edges: BTreeMap::new(),
    };
    check(&g, &m)?;

    for &op in ops {
        match op {
            Op::AddVertex => {
                if m.n >= MAX_VERTICES {
                    continue;
                }
                g.add_vertex().expect("vertex space");
                m.n += 1;
            }
            Op::AddEdge(a, b) => {
                if m.n == 0 {
                    continue;
                }
                let (s, t) = (a as usize % m.n, b as usize % m.n);
                let id = g.add_edge(v(s), v(t)).expect("add_edge").id();
                prop_assert!(
                    m.edges.insert(id, (s, t)).is_none(),
                    "id reissued while live"
                );
            }
            Op::RemoveEdge(k) => {
                if m.edges.is_empty() {
                    continue;
                }
                let k = k as usize % m.edges.len();
                let id = *m.edges.keys().nth(k).expect("in range");
                g.remove_edge(id).expect("remove_edge");
                m.edges.remove(&id);
            }
            Op::Clear(a) => {
                if m.n == 0 {
                    continue;
                }
                let vertex = a as usize % m.n;
                g.clear_vertex(v(vertex)).expect("clear_vertex");
                m.clear_where(vertex, 0);
            }
            Op::ClearWhere(a, kind) => {
                if m.n == 0 {
                    continue;
                }
                let vertex = a as usize % m.n;
                g.clear_vertex_where(v(vertex), |id, other| {
                    pred_of(kind, vertex, id, other.index())
                })
                .expect("clear_vertex_where");
                m.clear_where(vertex, kind);
            }
            Op::SwapRemove(a) => {
                if m.n == 0 {
                    continue;
                }
                let vertex = a as usize % m.n;
                g.swap_remove_vertex(v(vertex)).expect("swap_remove_vertex");
                m.swap_remove(vertex);
            }
        }
        check(&g, &m)?;
    }
    Ok(())
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        1 => Just(Op::AddVertex),
        6 => (0u8..8, 0u8..8).prop_map(|(a, b)| Op::AddEdge(a, b)),
        3 => any::<u8>().prop_map(Op::RemoveEdge),
        2 => (0u8..8).prop_map(Op::Clear),
        3 => (0u8..8, 0u8..4).prop_map(|(a, k)| Op::ClearWhere(a, k)),
        2 => (0u8..8).prop_map(Op::SwapRemove),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2048, ..ProptestConfig::default() })]

    /// The centrepiece, without the `(s,t)` index.
    #[test]
    fn model_sequences_agree_without_a_lookup(
        ops in prop::collection::vec(op_strategy(), 1..=16)
    ) {
        run::<NoLookup>(&ops)?;
    }

    /// The same sequences with `EHash` live. `validate()` then also compares
    /// the index against the adjacency in *multiplicity*, so a bucket that
    /// keeps a stale id -- `remove_ehash`'s `_ehpos[es.back()] = pos` write
    /// for `pos == es.size() - 1` (`:743-752`) -- is a failure here.
    #[test]
    fn model_sequences_agree_with_a_lookup(
        ops in prop::collection::vec(op_strategy(), 1..=16)
    ) {
        run::<EHash>(&ops)?;
    }
}

// ===========================================================================
// 7. The allocation ledger
//
// DESIGN.md section 3: "`clear_vertex_where` is one loop over `remove_edge`,
// filling the reusable scratch -- no per-call `Vec`, which a judge measured
// at 2-4 allocations per call in the design that used one."  That is a claim
// about a number, so it is measured.
// ===========================================================================

/// 20 000 vertices, 160 000 edges, deterministic.
fn ladder(n: usize, m: usize) -> Vec<(VertexId, VertexId)> {
    let mut out = Vec::with_capacity(m);
    let mut x = 12_345u64;
    for _ in 0..m {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let s = (x >> 33) as usize % n;
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let t = (x >> 33) as usize % n;
        out.push((v(s), v(t)));
    }
    out
}

#[test]
fn ten_thousand_clear_vertex_calls_allocate_nothing() {
    const N: usize = 20_000;
    const M: usize = 160_000;
    const SWEEP: usize = 10_000;

    let pairs = ladder(N, M);
    let mut g = AdjList::with_vertices(N);
    for &(s, t) in &pairs {
        g.add_edge(s, t).expect("add_edge");
    }

    // Warm-up: one full sweep grows the scratch buffer to the largest degree
    // it will ever see and the free list to the whole index range, then the
    // same edges go back in -- every container is at its steady-state
    // capacity and none of them has been shrunk.
    for i in 0..N {
        g.clear_vertex(v(i)).expect("clear_vertex");
    }
    assert_eq!(g.num_edges(), 0);
    for &(s, t) in &pairs {
        g.add_edge(s, t).expect("add_edge");
    }
    assert_eq!(g.num_edges(), M);

    let edges_before = g.num_edges();
    let (removed, allocations) = allocations(|| {
        let mut before = 0;
        for i in 0..SWEEP {
            before += g.degree(v(i));
            g.clear_vertex(v(i)).expect("clear_vertex");
        }
        before
    });

    assert_eq!(
        allocations, 0,
        "{SWEEP} clear_vertex calls allocated {allocations} times; \
         the scratch buffer is not being reused"
    );
    assert!(removed > 0, "the sweep did nothing");
    assert!(g.num_edges() < edges_before);
    assert!((0..SWEEP).all(|i| g.degree(v(i)) == 0));
    assert!(g.validate().is_ok());
}

/// The counter itself must be able to see an allocation, or the assertion
/// above is vacuous -- the shape of graph-tool's own `__test__ = False`
/// (`base_states.py:33`).
#[test]
fn the_allocation_counter_counts() {
    let (_, n) = allocations(|| {
        let mut v: Vec<u64> = Vec::new();
        for i in 0..1024 {
            v.push(i);
        }
        v
    });
    assert!(n > 0, "the counting allocator is not installed");
    let (_, none) = allocations(|| 1 + 1);
    assert_eq!(none, 0);
}
