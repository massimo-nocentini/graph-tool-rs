//! Differential tests: adjacency mutation and the six views, against
//! graph-tool 3.8's `adj_list`, `undirected_adaptor` and `reversed_graph`.
//!
//! Every expected value in this file was derived **by hand** from
//! `src/graph/graph_adjacency.hh`, `src/graph/graph_adaptor.hh` and
//! `src/graph/graph_reverse.hh` in the 3.8 tree, by simulating the C++ on
//! paper. Nothing here was read off the Rust implementation and turned into an
//! assertion; where the two disagree, the disagreement is stated in the test's
//! doc comment with both values.
//!
//! ## The fixture, and the C++ simulation behind it
//!
//! Five edges added to a four-vertex graph, in this order:
//!
//! ```text
//!   e0: 0 -> 1      e1: 1 -> 2      e2: 2 -> 0
//!   e3: 0 -> 0      e4: 0 -> 1        (v3 isolated)
//! ```
//!
//! graph-tool stores one `vector<pair<Vertex,Vertex>>` per vertex, split by a
//! `size_t pos` into an out-half `[0, pos)` and an in-half `[pos, size)`
//! (`graph_adjacency.hh:219-222`). `add_edge` (`:1190-1237`) is a *push-swap*:
//!
//! ```c++
//! if (s_pos < s_es.size()) {              // in-half is not empty
//!     s_es.push_back(s_es[s_pos]);        // displace its first entry
//!     s_es[s_pos] = {t, idx};             // ...and put the new out-entry there
//! } else {
//!     s_es.emplace_back(t, idx);
//! }
//! s_pos++;
//! t_es.emplace_back(s, idx);              // the in-entry always goes on the back
//! ```
//!
//! Running that on the fixture, vertex by vertex, gives (`(other, idx)`, with
//! `|` marking `pos`):
//!
//! ```text
//!   after e0   v0: (1,0) |                 v1: | (0,0)
//!   after e1   v1: (2,1) | (0,0)           v2: | (1,1)
//!   after e2   v2: (0,2) | (1,1)           v0: (1,0) | (2,2)
//!   after e3   v0: (1,0) (0,3) | (2,2) (0,3)
//!   after e4   v0: (1,0) (0,3) (1,4) | (0,3) (2,2)     v1: (2,1) | (0,0) (0,4)
//! ```
//!
//! The e3 step is the one worth following: `pos` was 1 and the list had two
//! entries, so `(2,2)` was pushed to the back and the self-loop's out-entry
//! took its place; then the self-loop's *in*-entry was appended. A self-loop
//! therefore occupies **two** slots of its vertex's block, one in each half.
//! The e4 step displaces `(2,2)` a second time, which is why the in-half comes
//! out as `(0,3) (2,2)` and not in edge-id order.
//!
//! `LAYOUT` below is that table, transcribed. It is the anchor for the rest of
//! the file: `out_edges` is `[0, pos)`, `in_edges` is `[pos, size)`,
//! `all_edges` is the whole run (`:1076-1098`), and the undirected view's
//! `out_edges` is *also* the whole run (`graph_adaptor.hh:199-207` routing to
//! `_all_edges_out`, `graph_adjacency.hh:1102-1108`).

use gt_core::adj::AdjList;
use gt_core::graph::{Bidirectional, EdgeList, Endpoints, GraphBase, GraphRef, VertexList};
use gt_core::ids::{EdgeId, VertexId};
use gt_core::prop::DenseProp;
use gt_core::view::{Filtered, Reverse, Undirect};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

const N: usize = 4;
/// `(source, target)` in the order `add_edge` is called, so the edge added
/// `i`th carries index `i` (`get_free_idx`, `:627-655`, on a graph that has
/// never freed an index).
const EDGES: [(usize, usize); 5] = [(0, 1), (1, 2), (2, 0), (0, 0), (0, 1)];

/// The block of each vertex as graph-tool builds it, hand-simulated above.
///
/// `(entries, pos)`; each entry is `(other, idx)`.
const LAYOUT: [(&[(usize, usize)], usize); N] = [
    (&[(1, 0), (0, 3), (1, 4), (0, 3), (2, 2)], 3),
    (&[(2, 1), (0, 0), (0, 4)], 1),
    (&[(0, 2), (1, 1)], 1),
    (&[], 0),
];

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

fn fixture() -> AdjList {
    let mut g = AdjList::with_vertices(N);
    for &(s, t) in EDGES.iter() {
        g.add_edge(v(s), v(t)).expect("both endpoints exist");
    }
    g
}

/// `(other, idx)` pairs of an incidence run, in yield order.
fn run<I: Iterator<Item = gt_core::adj::Incident>>(it: I) -> Vec<(usize, usize)> {
    it.map(|i| (i.other.index(), i.edge.index())).collect()
}

/// `(source, target, idx)` triples of an edge-list run, in yield order.
fn list<I: Iterator<Item = gt_core::adj::EdgeRef>>(it: I) -> Vec<(usize, usize, usize)> {
    it.map(|e| (e.source().index(), e.target().index(), e.id().index()))
        .collect()
}

// ---------------------------------------------------------------------------
// 1. The storage layout
// ---------------------------------------------------------------------------

/// The port reproduces `add_edge`'s push-swap entry for entry.
///
/// This is the load-bearing fact the rest of the file rests on: if the blocks
/// were merely *sets* that happened to agree, no ordering claim below --- and
/// no claim about which entry a removal displaces --- would transfer from the
/// C++ to this port. The expected values are `LAYOUT`, simulated on paper from
/// `graph_adjacency.hh:1190-1237`; see the module doc for the derivation.
///
/// Two of the three halves are order-sensitive in a way a naive
/// implementation gets wrong:
///
/// * v0's out-half is `(1,0) (0,3) (1,4)` --- insertion order, because a new
///   out-entry always lands at `pos`, i.e. at the end of the out-half;
/// * v0's in-half is `(0,3) (2,2)` --- **not** insertion order, because the
///   push-swap displaced `(2,2)` to the back twice.
#[test]
fn the_block_layout_reproduces_graph_tools_push_swap_entry_for_entry() {
    let g = fixture();
    for (i, (entries, pos)) in LAYOUT.iter().enumerate() {
        let out: Vec<_> = entries[..*pos].to_vec();
        let inn: Vec<_> = entries[*pos..].to_vec();
        assert_eq!(
            run(g.out_edges(v(i))),
            out,
            "v{i}: out-half is `[0, pos)` (graph_adjacency.hh:1076-1084)"
        );
        assert_eq!(
            run(g.in_edges(v(i))),
            inn,
            "v{i}: in-half is `[pos, size)` (graph_adjacency.hh:1086-1094)"
        );
        assert_eq!(
            run(g.all_edges(v(i))),
            entries.to_vec(),
            "v{i}: `all_edges` is the whole run (graph_adjacency.hh:1120-1129)"
        );
        assert_eq!(
            g.out_degree(v(i)),
            *pos,
            "out_degree is `pes.first` (:1053)"
        );
        assert_eq!(
            g.in_degree(v(i)),
            entries.len() - *pos,
            "in_degree is `es.size() - pos` (:1060)"
        );
        assert_eq!(
            g.degree(v(i)),
            entries.len(),
            "degree is `es.size()` (:1070) -- a self-loop counts twice"
        );
    }
}

/// A self-loop occupies **both** halves of its vertex's block.
///
/// `add_edge(v, v, g)` runs the out-half insert and then `t_es.emplace_back`
/// on the *same* vector (`:1215-1217`, with `s == t`). So `out_degree` and
/// `in_degree` each count it once and `degree` counts it twice --- which is
/// exactly what `_mrp[r] += kout` in the SBM depends on
/// (`blockmodel/state.hh:224`, and see `u35_diff_sbm.rs`).
#[test]
fn a_self_loop_occupies_both_halves_of_its_block() {
    let g = fixture();
    let loops_out = g.out_edges(v(0)).filter(|i| i.other == v(0)).count();
    let loops_in = g.in_edges(v(0)).filter(|i| i.other == v(0)).count();
    let loops_all = g.all_edges(v(0)).filter(|i| i.other == v(0)).count();
    assert_eq!((loops_out, loops_in, loops_all), (1, 1, 2));
    assert_eq!(g.degree(v(0)) - g.out_degree(v(0)) - g.in_degree(v(0)), 0);
    // But there is only one *edge*: the two entries share an index.
    assert_eq!(
        g.all_edges(v(0))
            .filter(|i| i.other == v(0))
            .map(|i| i.edge)
            .collect::<Vec<_>>(),
        vec![EdgeId::from_index(3), EdgeId::from_index(3)]
    );
    assert_eq!(g.num_edges(), EDGES.len());
}

/// The global edge list is vertex-major over the out-halves.
///
/// `adj_list::edge_iterator::skip()` (`:353-362`) advances while
/// `_ei == _vi->second.begin() + _vi->first`, i.e. it stops at `pos`: the
/// global list walks vertices in index order and yields each vertex's
/// **out-half only**. So every edge is named exactly once, at its stored
/// source, and the order is *not* edge-id order.
#[test]
fn the_global_edge_list_is_vertex_major_over_out_halves_only() {
    let g = fixture();
    assert_eq!(
        list(g.edges()),
        vec![(0, 1, 0), (0, 0, 3), (0, 1, 4), (1, 2, 1), (2, 0, 2)],
        "vertex-major over out-halves; `LAYOUT` read down the out-halves"
    );
    assert_eq!(g.edges().count(), g.num_edges());
}

// ---------------------------------------------------------------------------
// 2. Removal, recycling, density
// ---------------------------------------------------------------------------

/// Removal reproduces graph-tool's `_keep_epos == true` path, **not** its
/// default.
///
/// `remove_edge` (`:1244-1315`) has two bodies selected by a runtime flag that
/// defaults to `false` (`:229`, and is turned on from Python by
/// `g.set_fast_edge_removal(True)`):
///
/// * `!_keep_epos` (`:1258-1272`): `std::find_if` then `elist.erase(iter)` ---
///   O(k) and **order-preserving**;
/// * `_keep_epos` (`:1274-1305`): swap the doomed entry with the back of its
///   half, and, for the out-half, additionally swap the vacated middle slot
///   with the very back of the vector --- O(1) and **order-destroying**.
///
/// This port implements only the second. Hand-simulating both on the fixture
/// for `remove_edge(e0)`, where `v0 = (1,0) (0,3) (1,4) | (0,3) (2,2)`:
///
/// ```text
///   erase-and-shift (C++ default):   (0,3) (1,4) | (0,3) (2,2)
///   swap-with-back  (_keep_epos):    (1,4) (0,3) | (2,2) (0,3)
/// ```
///
/// The swap-with-back trace, from `:1277-1297`: `back` is the last out-entry
/// `(1,4)`; `j = _epos[0].first = 0`; `elist[0] = (1,4)`; then because the
/// out-half does not end the vector, `elist[2]` (the slot `(1,4)` vacated) is
/// overwritten with `elist.back() = (2,2)` and the vector is popped; `pos--`.
/// The in-half pass then finds `(0,3)` at index 3, which is the back of the
/// in-half, self-assigns it, and pops --- the "departing edge self-assign" of
/// `:1287-1288`.
///
/// **This is a real divergence from graph-tool's default configuration.** The
/// *set* of edges agrees; the order of the surviving entries in a block, and
/// therefore the order `out_edges`/`edges` yields them in, does not. It is the
/// divergence `gt-io/src/lib.rs` documents and works around by sorting on
/// `EdgeId` before writing.
#[test]
fn removal_is_the_keep_epos_swap_with_back_not_the_default_erase_and_shift() {
    let mut g = fixture();
    g.remove_edge(EdgeId::from_index(0)).expect("e0 is live");

    // What graph-tool's *default* `!_keep_epos` branch would have produced.
    let erase_and_shift_out = vec![(0usize, 3usize), (1, 4)];
    let erase_and_shift_in = vec![(0usize, 3usize), (2, 2)];
    // What `_keep_epos == true` produces, hand-traced above.
    let swap_with_back_out = vec![(1usize, 4usize), (0, 3)];
    let swap_with_back_in = vec![(2usize, 2usize), (0, 3)];

    assert_eq!(run(g.out_edges(v(0))), swap_with_back_out);
    assert_eq!(run(g.in_edges(v(0))), swap_with_back_in);
    assert_ne!(
        run(g.out_edges(v(0))),
        erase_and_shift_out,
        "if this ever passes, the port has silently switched to the C++ default"
    );
    assert_ne!(run(g.in_edges(v(0))), erase_and_shift_in);

    // v1's in-half: `back` is `(0,4)`, `j = _epos[0].second = 1`, no middle
    // swap (the in-half ends the vector), pop. Both C++ branches agree here.
    assert_eq!(run(g.in_edges(v(1))), vec![(0, 4)]);
    assert_eq!(run(g.out_edges(v(1))), vec![(2, 1)]);

    // Whatever the order, the multiset of incidences is graph-tool's.
    let mut all: Vec<_> = (0..N).flat_map(|i| run(g.all_edges(v(i)))).collect();
    all.sort_unstable();
    assert_eq!(
        all,
        vec![
            (0, 2),
            (0, 3),
            (0, 3),
            (0, 4),
            (1, 1),
            (1, 4),
            (2, 1),
            (2, 2)
        ]
    );
    assert_eq!(g.num_edges(), 4);
}

/// The free list is LIFO and the index range never shrinks.
///
/// `put_free_index` is `_free_idx.push_back(idx)` (`:659-664`) and
/// `get_free_idx` is `idx = _free_idx.back(); _free_idx.pop_back()`
/// (`:645-652`), so recycling is last-freed-first. `_edge_idx_range` is only
/// ever incremented (`:636`, `:648`); nothing in `add_edge`/`remove_edge`
/// lowers it, which is why `edge_index_range >= num_edges` after any removal
/// and why `_get_any` sizes an edge property map from the *range*
/// (`graph_tool/__init__.py:369`).
#[test]
fn the_free_list_is_lifo_and_the_index_range_never_shrinks() {
    let mut g = AdjList::with_vertices(3);
    let ids: Vec<EdgeId> = [(0, 1), (1, 2), (2, 0), (0, 2)]
        .iter()
        .map(|&(s, t)| g.add_edge(v(s), v(t)).expect("add").id())
        .collect();
    assert_eq!(
        ids.iter().map(|i| i.index()).collect::<Vec<_>>(),
        vec![0, 1, 2, 3],
        "a graph that has never freed an index allocates `_edge_idx_range++`"
    );
    assert_eq!(g.edge_bound().len(), 4);

    g.remove_edge(ids[1]).expect("e1 is live"); // free list: [1]
    g.remove_edge(ids[3]).expect("e3 is live"); // free list: [1, 3]
    assert_eq!(g.num_edges(), 2);
    assert_eq!(
        g.edge_bound().len(),
        4,
        "`_edge_idx_range` is monotone: the range keeps the holes"
    );

    // LIFO: 3 first, then 1, and only then a fresh index.
    assert_eq!(g.add_edge(v(0), v(0)).expect("add").id().index(), 3);
    assert_eq!(g.add_edge(v(1), v(1)).expect("add").id().index(), 1);
    assert_eq!(g.add_edge(v(2), v(2)).expect("add").id().index(), 4);
    assert_eq!(g.edge_bound().len(), 5);
    assert_eq!(g.num_edges(), 5);
}

/// An edge property map is sized by the index *range*, not the edge count.
///
/// `PropertyMap._get_any` (`graph_tool/__init__.py:363-372`) reserves
/// `g.edge_index_range` for an edge map and `g.num_vertices(True)` for a
/// vertex map --- never `num_edges()`. Reproducing that is what keeps the
/// holes addressable: after two removals the live ids here are `{0, 2}` out of
/// a range of 4, and a map sized to `num_edges() == 2` would put edge 2's
/// value out of bounds.
#[test]
fn an_edge_property_map_is_sized_by_the_index_range_not_the_edge_count() {
    let mut g = AdjList::with_vertices(3);
    let ids: Vec<EdgeId> = [(0, 1), (1, 2), (2, 0), (0, 2)]
        .iter()
        .map(|&(s, t)| g.add_edge(v(s), v(t)).expect("add").id())
        .collect();
    g.remove_edge(ids[1]).expect("live");
    g.remove_edge(ids[3]).expect("live");

    let eb = g.edge_bound();
    assert_eq!((eb.len(), g.num_edges()), (4, 2), "the space stays sparse");

    let mut p: DenseProp<i64, gt_core::ids::EdgeTag> = DenseProp::new(g.graph_id());
    let mut w = p.sized_for(eb).expect("same graph");
    for e in g.edges() {
        w.as_mut_slice()[e.id().index()] = e.id().index() as i64 + 100;
    }
    assert_eq!(
        p.as_slice(),
        &[100, 0, 102, 0],
        "live ids 0 and 2 are written; the two holes keep the default"
    );
    assert_eq!(
        p.len(),
        4,
        "`reserve(edge_index_range)` (fast_vector_property_map.hh:77-82)"
    );
}

/// `clear_vertex` frees in block order, so the next allocations come back in
/// exactly the reverse of it.
///
/// The `_keep_epos` branch (`:1416-1433`) collects the doomed descriptors by
/// walking `es[0..size)` in order --- skipping the in-half occurrence of a
/// self-loop, `(j >= pos && e.first == v)` --- and then calls `remove_edge` on
/// each, so `put_free_index` is called in that same block order. Composed with
/// the LIFO free list, the next `add_edge` calls hand the indices back in
/// reverse block order.
///
/// Fixture: `0->1 (e0)`, `0->2 (e1)`, `3->0 (e2)`, `0->0 (e3)`, giving
/// `v0 = (1,0) (2,1) (0,3) | (3,2) (0,3)` by the same push-swap simulation.
/// The visit order is then `e0, e1, e3, e2` (the last entry, the self-loop's
/// in-half copy, is skipped), so the free list ends `[0, 1, 3, 2]` and pops
/// `2, 3, 1, 0`.
#[test]
fn clear_vertex_frees_in_block_order_and_reallocation_reverses_it() {
    let mut g = AdjList::with_vertices(4);
    for &(s, t) in &[(0, 1), (0, 2), (3, 0), (0, 0)] {
        g.add_edge(v(s), v(t)).expect("add");
    }
    assert_eq!(
        run(g.all_edges(v(0))),
        vec![(1, 0), (2, 1), (0, 3), (3, 2), (0, 3)],
        "push-swap layout, simulated from graph_adjacency.hh:1190-1237"
    );

    g.clear_vertex(v(0)).expect("v0 exists");
    assert_eq!(g.num_edges(), 0, "every edge was incident to v0");
    assert_eq!(g.edge_bound().len(), 4, "the range does not shrink");

    let got: Vec<usize> = (0..4)
        .map(|_| g.add_edge(v(1), v(2)).expect("add").id().index())
        .collect();
    assert_eq!(
        got,
        vec![2, 3, 1, 0],
        "free list [0, 1, 3, 2] popped from the back"
    );
}

/// A predicate passed to `clear_vertex_where` sees a self-loop **once**; the
/// C++ offers it **twice**.
///
/// `clear_vertex`'s `_keep_epos` branch is
/// `if (!pred(ed) || (j >= pos && e.first == v)) continue;` (`:1424`). The
/// short-circuit is in the wrong order: `pred` is evaluated on the in-half
/// occurrence of a self-loop *before* the guard discards it. A `Pred` with a
/// side effect --- and `clear_vertex(v, g, pred)` is reachable from
/// `graph_filtered.hh:573-584` with a filter predicate --- therefore observes
/// one edge twice. The removed *set* is the same either way, which is why the
/// defect is invisible to every existing test of it.
///
/// This port evaluates the guard first, so the count is 4 where graph-tool's
/// is 5. **Deliberate divergence**, and this test is what pins it.
#[test]
fn a_self_loop_reaches_the_clear_vertex_predicate_once_where_the_cpp_offers_it_twice() {
    let mut g = AdjList::with_vertices(4);
    for &(s, t) in &[(0, 1), (0, 2), (3, 0), (0, 0)] {
        g.add_edge(v(s), v(t)).expect("add");
    }
    // v0's block is `(1,0) (2,1) (0,3) | (3,2) (0,3)`: five entries, four
    // edges, the self-loop appearing at index 2 and index 4.
    assert_eq!(g.all_edges(v(0)).count(), 5);

    let mut calls: Vec<(usize, usize)> = Vec::new();
    g.clear_vertex_where(v(0), |id, other| {
        calls.push((id.index(), other.index()));
        true
    })
    .expect("v0 exists");

    assert_eq!(
        calls,
        vec![(0, 1), (1, 2), (3, 0), (2, 3)],
        "block order, with the in-half copy of e3 skipped before the call"
    );
    assert_eq!(
        calls.len(),
        4,
        "graph-tool would call the predicate 5 times"
    );
    assert_eq!(
        calls.iter().filter(|&&(id, _)| id == 3).count(),
        1,
        "graph_adjacency.hh:1424 evaluates `pred` on both copies of a self-loop"
    );
    assert_eq!(g.num_edges(), 0);
}

/// A rejected edge survives `clear_vertex_where`, and the count stays right.
///
/// `_n_edges` is not decremented here at all: it is derived from
/// `EdgeIds::live()`. graph-tool's non-`_keep_epos` `clear_vertex` instead
/// accumulates `k` by hand and then does `g._n_edges -= k` (`:1413`), where
/// `k` is partly a `count_if` over the moved-from tail of a `std::remove_if`
/// (`:1409-1410`) --- a range whose element values the standard leaves
/// unspecified.
#[test]
fn clear_vertex_where_keeps_the_edges_its_predicate_refuses() {
    let mut g = fixture();
    // Drop only the edges of v0 that point at (or come from) vertex 1.
    g.clear_vertex_where(v(0), |_, other| other == v(1))
        .expect("v0 exists");
    assert_eq!(g.num_edges(), 3, "e0 and e4 went; e2 and e3 stayed");
    let mut ids: Vec<usize> = g.edges().map(|e| e.id().index()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3]);
    g.validate().expect("the structure is consistent");
}

/// `swap_remove_vertex` keeps the moved vertex's edge **indices**.
///
/// `remove_vertex_fast` (`:1471-1535`) clears `v`, then relabels the back
/// vertex's endpoints in place; the edge indices are never touched. This port
/// re-splices instead of patching in place --- so that the `(s,t)` lookup is
/// notified for both directions, which `:1479-1529` does not do --- but the
/// identity of each surviving edge is the same.
#[test]
fn swap_remove_vertex_relabels_the_back_vertex_and_keeps_edge_indices() {
    let mut g = AdjList::with_vertices(4);
    // v3 is the back vertex and carries two edges; v1 is the one removed.
    for &(s, t) in &[(0, 1), (1, 2), (3, 0), (2, 3)] {
        g.add_edge(v(s), v(t)).expect("add");
    }
    g.swap_remove_vertex(v(1)).expect("v1 exists");

    assert_eq!(g.num_vertices(), 3);
    assert_eq!(g.num_edges(), 2, "e0 and e1 were incident to v1");
    let mut got = list(g.edges());
    got.sort_unstable_by_key(|&(_, _, i)| i);
    assert_eq!(
        got,
        vec![(1, 0, 2), (2, 1, 3)],
        "`3->0` became `1->0` and `2->3` became `2->1`; indices 2 and 3 survive"
    );
    g.validate().expect("the structure is consistent");
}

// ---------------------------------------------------------------------------
// 3. The six views
// ---------------------------------------------------------------------------

/// The undirected view's `out_edges` is the **whole block**, anchored at the
/// query vertex.
///
/// `out_edges(u, undirected_adaptor<G>)` is `_all_edges_out(u, original)`
/// (`graph_adaptor.hh:199-207`), and `_all_edges_out`
/// (`graph_adjacency.hh:1102-1108`) builds an `out_edge_iterator` --- not an
/// `all_edge_iterator` --- over `es.begin()..es.end()`. So
/// `make_out_edge::def` (`:298-305`) sets `src == u` for the *in*-half entries
/// too: every incidence of the undirected view reports the query vertex as its
/// source and the neighbour as its target, whatever the stored direction.
///
/// That is the claim the design review found three of four candidate designs
/// got wrong (wiring `out_edges` to a canonical all-edges iterator, so that
/// `target(e)` returns the query vertex itself for half the edges). Here the
/// yielded [`Incident`](gt_core::adj::Incident) carries only `other`, so the
/// anchoring is structural: `other` is never the query vertex except for a
/// genuine self-loop.
#[test]
fn the_undirected_view_anchors_every_incidence_at_the_query_vertex() {
    let g = fixture();
    let und = (&g).undirect();

    for (i, (entries, _pos)) in LAYOUT.iter().enumerate() {
        assert_eq!(
            run(und.out_edges(v(i))),
            entries.to_vec(),
            "v{i}: `_all_edges_out` is the whole block in storage order"
        );
        assert_eq!(und.out_degree(v(i)), entries.len(), "degree(u, adaptor)");
        // Anchored: the only way to see the query vertex as `other` is a
        // genuine self-loop.
        for inc in und.out_edges(v(i)) {
            if inc.other == v(i) {
                assert_eq!(
                    g.endpoints(inc.edge),
                    Some((v(i), v(i))),
                    "`other == u` only for a real self-loop"
                );
            }
        }
    }

    // A directed in-edge is reported with the *neighbour* as `other`: e2 is
    // stored `2 -> 0` and appears in v0's undirected run as `(2, 2)`.
    assert!(run(und.out_edges(v(0))).contains(&(2, 2)));
    assert_eq!(g.endpoints(EdgeId::from_index(2)), Some((v(2), v(0))));
}

/// The undirected view sees a self-loop **twice** per vertex.
///
/// It is the whole block, and a self-loop occupies both halves, so
/// `degree(v, undirected_adaptor)` is `degree(v, original) = es.size()`
/// (`graph_adaptor.hh:315-319` -> `graph_adjacency.hh:1068-1072`) and the
/// self-loop contributes 2. This is not an accident that cancels: it is what
/// the SBM's `_mrp[r] += kout` relies on for an undirected self-pair
/// (`blockmodel/entries.hh:393-398` with `r == s`, and `state.hh:224`).
#[test]
fn the_undirected_view_yields_a_self_loop_twice() {
    let g = fixture();
    let und = (&g).undirect();
    assert_eq!(
        und.out_edges(v(0)).filter(|i| i.edge.index() == 3).count(),
        2
    );
    assert_eq!(und.out_degree(v(0)), 5);
    assert_eq!(g.num_edges(), 5, "but the edge *count* is unchanged");
    assert_eq!(
        und.num_edges(),
        5,
        "`num_edges(undirected_adaptor)` forwards to the original (graph_adaptor.hh)"
    );
    // The sum of undirected degrees is 2E, self-loops counting twice.
    let sum: usize = (0..N).map(|i| und.out_degree(v(i))).sum();
    assert_eq!(sum, 2 * g.num_edges());
}

/// `edges(undirected_adaptor)` forwards to the original: each edge once, in
/// storage orientation.
///
/// `graph_adaptor.hh:134-140`. So the undirected view's *global* list and its
/// *incidence* runs disagree about orientation on purpose --- the list is
/// canonical, the runs are anchored --- and a port that made `edges()` also
/// anchored would double every edge.
#[test]
fn the_undirected_global_edge_list_is_the_originals() {
    let g = fixture();
    let und = (&g).undirect();
    assert_eq!(list(und.edges()), list(g.edges()));
    assert_eq!(und.edges().count(), 5);
}

/// The reversed view swaps the *accessors*; it does not rewrite the block.
///
/// `out_edges(u, reversed_graph<G>)` is `in_edges(u, original)`
/// (`graph_reverse.hh:126-131`) and `in_edges(u, rev)` is `out_edges(u, orig)`
/// (`:178-183`); `source(e, rev)` is `target(e, orig)` and vice versa
/// (`:243-256`). Nothing renumbers, nothing copies, and `all_edges(u, rev)` is
/// `all_edges(u, orig)` verbatim (`:187-192`).
#[test]
fn the_reversed_view_swaps_the_accessors_not_the_storage() {
    let g = fixture();
    let rev = (&g).reverse();

    for (i, (entries, pos)) in LAYOUT.iter().enumerate() {
        assert_eq!(
            run(rev.out_edges(v(i))),
            entries[*pos..].to_vec(),
            "v{i}: `out_edges(u, rev) == in_edges(u, orig)`"
        );
        assert_eq!(
            run(rev.in_edges(v(i))),
            entries[..*pos].to_vec(),
            "v{i}: `in_edges(u, rev) == out_edges(u, orig)`"
        );
        assert_eq!(
            run(rev.all_edges(v(i))),
            entries.to_vec(),
            "v{i}: `all_edges(u, rev) == all_edges(u, orig)`, unchanged"
        );
        assert_eq!(rev.out_degree(v(i)), entries.len() - *pos);
        assert_eq!(rev.in_degree(v(i)), *pos);
        assert_eq!(rev.degree(v(i)), entries.len(), "`degree` is unchanged");
    }
    assert_eq!(rev.num_edges(), g.num_edges());
}

/// The reversed global edge list swaps the endpoints and keeps the indices.
///
/// `edges(rev)` forwards to `edges(orig)` unchanged (`graph_reverse.hh:120-124`)
/// and the *descriptors* are therefore identical; what changes is that
/// `source`/`target` are swapped when read through the reversed graph. The
/// observable pair is therefore `(t, s)` with the same `idx`, which is what
/// `SwapEnds` reproduces.
#[test]
fn the_reversed_edge_list_swaps_endpoints_and_keeps_indices() {
    let g = fixture();
    let rev = (&g).reverse();
    assert_eq!(
        list(rev.edges()),
        vec![(1, 0, 0), (0, 0, 3), (1, 0, 4), (2, 1, 1), (0, 2, 2)],
        "`edges(orig)` with `source`/`target` read through the reversal"
    );
    let forward = list(g.edges());
    for (r, f) in list(rev.edges()).iter().zip(forward.iter()) {
        assert_eq!((r.0, r.1, r.2), (f.1, f.0, f.2));
    }
}

/// Reversing twice, and undirecting a reversal, are identities on incidence.
///
/// `get_reversed_graph` (`graph_reverse.hh:55-70`) is a `reinterpret_cast` and
/// returns the graph itself for an undirected one; `undirected_adaptor` over a
/// reversal sees the same block through the same whole-run iterator. Here the
/// algebra is closed at the type level instead (`Rev::reverse() -> G`,
/// `Rev::undirect() -> Und<G>`), and this pins that the *values* agree too.
#[test]
fn the_view_algebra_closes_on_the_same_incidence_runs() {
    let g = fixture();
    let back = (&g).reverse().reverse();
    let und_of_rev = (&g).reverse().undirect();
    let und = (&g).undirect();
    for i in 0..N {
        assert_eq!(run(back.out_edges(v(i))), run(g.out_edges(v(i))));
        assert_eq!(run(back.in_edges(v(i))), run(g.in_edges(v(i))));
        assert_eq!(
            run(und_of_rev.out_edges(v(i))),
            run(und.out_edges(v(i))),
            "reversing an undirected view is a no-op on the whole-run iterator"
        );
    }
}

/// `edge(u, v, undirected_adaptor)` swaps the descriptor's endpoints on the
/// reverse hit; this port returns storage orientation instead.
///
/// `graph_adaptor.hh:146-163`:
///
/// ```c++
/// auto res = edge(u, v, g.original_graph());
/// if (!res.second) {
///     res = edge(v, u, g.original_graph());
///     std::swap(res.first.s, res.first.t);     // <-- here
/// }
/// ```
///
/// So on a graph holding only `0 -> 1`, `edge(1, 0, adaptor)` returns a
/// descriptor whose `s` is 1 and whose `t` is 0 --- an orientation the
/// adjacency does not hold, and one that disagrees with what the adaptor's own
/// `out_edges(1)` reports for the same edge (`src == 1`, `tgt == 0`, by
/// anchoring --- so the two happen to agree here, and disagree the other way
/// round: `edge(0, 1, adaptor)` hits on the first try and keeps `0 -> 1`,
/// while `out_edges(1)` also calls the same edge `1 -> 0`).
///
/// **Deliberate divergence.** `find_edge` here answers with a canonical
/// [`EdgeRef`](gt_core::adj::EdgeRef) in storage orientation, so lookup and
/// iteration cannot disagree: they have different return types and answer
/// different questions.
#[test]
fn undirected_find_edge_returns_storage_orientation_where_the_cpp_swaps() {
    let g = fixture();
    let und = (&g).undirect();
    let rev = (&g).reverse();

    // `1 -> 0` is not stored; `0 -> 1` is (e0 and e4).
    assert!(g.find_edge(v(1), v(0)).is_none());
    let hit = und
        .find_edge(v(1), v(0))
        .expect("undirected lookup succeeds");
    assert_eq!(
        (hit.source().index(), hit.target().index()),
        (0, 1),
        "storage orientation; graph-tool would return `s = 1, t = 0`"
    );
    assert_eq!(und.find_edge(v(0), v(1)).map(|e| e.id()), Some(hit.id()));

    // `edge(u, v, rev)` is `edge(v, u, orig)` (`graph_reverse.hh:165-171`),
    // and the endpoints are then read through the reversal.
    let r = rev.find_edge(v(1), v(0)).expect("reversed lookup succeeds");
    assert_eq!((r.source().index(), r.target().index()), (1, 0));
    assert!(rev.find_edge(v(0), v(1)).is_none());
}

/// `degree(u, filtered)` is `in_degree + out_degree`, counted over the
/// surviving edges only.
///
/// `graph_filtered.hh:395-399`. The filtered view of a directed graph is still
/// directed, so its `out_edges` is the filtered out-half --- and an undirected
/// *filtered* view is the filtered whole run.
#[test]
fn the_filtered_views_degrees_are_the_surviving_incidences() {
    let g = fixture();
    // Drop vertex 2 and edge e4.
    let mut vmask = vec![1u8; N];
    vmask[2] = 0;
    let mut emask = vec![1u8; g.edge_bound().len()];
    emask[4] = 0;

    let f = Filtered::masked(&g, &vmask, &emask).expect("masks cover the bounds");
    assert_eq!(f.num_vertices(), 3);
    assert_eq!(
        f.vertices().count(),
        3,
        "`vertices(filt_graph)` skips the masked vertex"
    );
    // Surviving edges: e0 (0->1) and e3 (0->0). e1 and e2 touch v2; e4 is
    // masked.
    assert_eq!(
        f.edges().map(|e| e.id().index()).collect::<Vec<_>>(),
        vec![0, 3]
    );
    assert_eq!(f.num_edges(), 2);
    assert_eq!(run(f.out_edges(v(0))), vec![(1, 0), (0, 3)]);
    assert_eq!(run(f.in_edges(v(0))), vec![(0, 3)]);
    assert_eq!(
        f.degree(v(0)),
        3,
        "in_degree + out_degree (graph_filtered.hh:395-399); the self-loop counts twice"
    );

    let fu = Filtered::masked((&g).undirect(), &vmask, &emask).expect("masks cover");
    assert_eq!(
        run(fu.out_edges(v(0))),
        vec![(1, 0), (0, 3), (0, 3)],
        "the filtered whole run"
    );
    assert_eq!(fu.out_degree(v(0)), 3);

    let fr = Filtered::masked((&g).reverse(), &vmask, &emask).expect("masks cover");
    assert_eq!(run(fr.out_edges(v(0))), vec![(0, 3)]);
    assert_eq!(run(fr.in_edges(v(0))), vec![(1, 0), (0, 3)]);
}

/// All six views report the same `num_edges` and the same *multiset* of edge
/// identities, and differ only in orientation and in which half a vertex sees.
#[test]
fn the_six_views_agree_on_the_edge_set() {
    let g = fixture();
    let vmask = vec![1u8; N];
    let emask = vec![1u8; g.edge_bound().len()];

    let ids = |mut v: Vec<usize>| {
        v.sort_unstable();
        v
    };
    let base = ids(g.edges().map(|e| e.id().index()).collect());
    assert_eq!(base, vec![0, 1, 2, 3, 4]);

    assert_eq!(
        ids((&g).undirect().edges().map(|e| e.id().index()).collect()),
        base
    );
    assert_eq!(
        ids((&g).reverse().edges().map(|e| e.id().index()).collect()),
        base
    );
    let f = Filtered::masked(&g, &vmask, &emask).expect("full masks");
    assert_eq!(ids(f.edges().map(|e| e.id().index()).collect()), base);
    let fu = Filtered::masked((&g).undirect(), &vmask, &emask).expect("full masks");
    assert_eq!(ids(fu.edges().map(|e| e.id().index()).collect()), base);
    let fr = Filtered::masked((&g).reverse(), &vmask, &emask).expect("full masks");
    assert_eq!(ids(fr.edges().map(|e| e.id().index()).collect()), base);

    for (name, d) in [
        ("directed", g.out_degree(v(0))),
        ("undirected", (&g).undirect().out_degree(v(0))),
        ("reversed", (&g).reverse().out_degree(v(0))),
    ] {
        let want = match name {
            "directed" => 3,
            "undirected" => 5,
            _ => 2,
        };
        assert_eq!(d, want, "out_degree(v0) on the {name} view");
    }
}
