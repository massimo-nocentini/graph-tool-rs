//! U13 — traversal, from outside the crate.
//!
//! The fixed graph below is the one every hand-computed table in this file
//! refers to. It is six vertices and seven edges, added in this order, and the
//! order matters twice: `Block::insert_out` is the O(1) push-swap of
//! `graph_adjacency.hh:1192-1215`, so the *out*-half of a block is in
//! insertion order and a directed DFS has a pinnable visit sequence; the
//! *in*-half is not, because the push-swap displaces its first entry to the
//! back, so nothing here asserts an order on a reversed or undirected view.
//!
//! ```text
//!        e0        e1
//!   0 ───────> 1 ───────> 2
//!   ^          │          │
//!   │ e2       │ e3       │ e6
//!   └───────── 2 <─┘      │
//!              3 ──e4──> 4 <──e5── 5
//! ```
//!
//! (`e2: 2->0`, drawn folded back.)

use gt_algo::traversal::{
    Control, UNREACHABLE, Visitor, bfs, bfs_multi, dfs, dijkstra, shortest_distances,
};
use gt_core::adj::{AdjList, Incident};
use gt_core::graph::{GraphBase, GraphRef, VertexList};
use gt_core::ids::{EdgeId, EdgeTag, VertexId, VertexTag};
use gt_core::prop::dense::Unity;
use gt_core::prop::{Constant, DenseProp, Owned, ReadProp};
use gt_core::view::{Filtered, Reverse, Undirect};

// ---------------------------------------------------------------------------
// The fixture
// ---------------------------------------------------------------------------

const N: usize = 6;
/// `(source, target)` in edge-id order.
const EDGES: [(usize, usize); 7] = [(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (5, 4), (2, 4)];

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

fn fixture() -> AdjList {
    let mut g = AdjList::with_vertices(N);
    for &(s, t) in EDGES.iter() {
        g.add_edge(v(s), v(t)).expect("add_edge");
    }
    g
}

/// Keeps every vertex but `3` and every edge but `e6`.
///
/// Two independent reasons for an edge to disappear, which is the point: `e3`
/// and `e4` go because an *endpoint* went (`keeps_edge` tests both endpoints,
/// `view/filtered.rs`), `e6` goes because the edge mask says so.
fn masks() -> (Vec<u8>, Vec<u8>) {
    let mut vmask = vec![1u8; N];
    vmask[3] = 0;
    let mut emask = vec![1u8; EDGES.len()];
    emask[6] = 0;
    (vmask, emask)
}

// ---------------------------------------------------------------------------
// Visitors used by more than one test
// ---------------------------------------------------------------------------

/// Records the three events in the order they fire.
#[derive(Default, Debug)]
struct Rec {
    /// `(vertex, depth)`.
    discovered: Vec<(usize, usize)>,
    /// `(anchor, edge)`.
    examined: Vec<(usize, usize)>,
    finished: Vec<usize>,
}

impl Rec {
    fn order(&self) -> Vec<usize> {
        self.discovered.iter().map(|&(v, _)| v).collect()
    }
    fn depths(&self) -> Vec<(usize, usize)> {
        let mut d = self.discovered.clone();
        d.sort_unstable();
        d
    }
}

impl Visitor for Rec {
    fn discover(&mut self, v: VertexId, depth: usize) -> Control {
        self.discovered.push((v.index(), depth));
        Control::Continue
    }
    fn examine(&mut self, from: VertexId, e: Incident) -> Control {
        self.examined.push((from.index(), e.edge.index()));
        Control::Continue
    }
    fn finish(&mut self, v: VertexId) {
        self.finished.push(v.index());
    }
}

/// `Rec`, plus one answer that is not `Continue`.
struct Answering {
    rec: Rec,
    /// Answer given when this vertex is discovered.
    at_vertex: Option<(usize, Control)>,
    /// Answer given when this edge is examined.
    at_edge: Option<(usize, Control)>,
}

impl Answering {
    fn at_vertex(i: usize, c: Control) -> Self {
        Answering {
            rec: Rec::default(),
            at_vertex: Some((i, c)),
            at_edge: None,
        }
    }
    fn at_edge(i: usize, c: Control) -> Self {
        Answering {
            rec: Rec::default(),
            at_vertex: None,
            at_edge: Some((i, c)),
        }
    }
}

impl Visitor for Answering {
    fn discover(&mut self, v: VertexId, depth: usize) -> Control {
        self.rec.discover(v, depth);
        match self.at_vertex {
            Some((i, c)) if i == v.index() => c,
            _ => Control::Continue,
        }
    }
    fn examine(&mut self, from: VertexId, e: Incident) -> Control {
        self.rec.examine(from, e);
        match self.at_edge {
            Some((i, c)) if i == e.edge.index() => c,
            _ => Control::Continue,
        }
    }
    fn finish(&mut self, v: VertexId) {
        self.rec.finish(v);
    }
}

// ---------------------------------------------------------------------------
// 1. BFS distances on the six views
// ---------------------------------------------------------------------------

fn dists<G: GraphRef + VertexList>(g: G, source: usize) -> Vec<i64> {
    let mut d = DenseProp::<i64, VertexTag>::new(g.graph_id());
    shortest_distances(g, v(source), &mut d).expect("source is in bounds");
    d.as_slice().to_vec()
}

const M: i64 = UNREACHABLE;

/// The acceptance table. Every row was computed by hand from the drawing at
/// the top of this file; none of them was read off a run.
///
/// The three unfiltered rows are what makes the view algebra worth having:
/// one kernel, three answers, and the *reversed* row is not the directed row
/// (`graph_reverse.hh:78-80` swaps the two iterator typedefs) nor the
/// undirected one. `graph_adaptor.hh:224-233` returns an empty range where the
/// reversed row differs from the undirected one, which is why the difference
/// between rows 2 and 3 is the thing under test.
#[test]
fn bfs_distances_on_the_six_views_match_the_hand_table() {
    let g = fixture();
    let (vmask, emask) = masks();

    // directed: 0 -> 1 -> {2, 3} -> 4; 5 has no in-edge from the component.
    assert_eq!(dists(&g, 0), [0, 1, 2, 2, 3, M], "directed");

    // reversed: only the 0-1-2 cycle can reach 0, and 3 cannot (its one
    // out-edge goes to the sink 4).
    assert_eq!(dists((&g).reverse(), 0), [0, 2, 1, M, M, M], "reversed");

    // undirected: everything is one component.
    assert_eq!(dists((&g).undirect(), 0), [0, 1, 1, 2, 2, 3], "undirected");

    // Filtered: vertex 3 and edge e6 are gone, so e3 and e4 go with the
    // vertex and 4 loses its last inbound edge from the component.
    let fd = Filtered::masked(&g, &vmask, &emask).expect("masks are long enough");
    assert_eq!(dists(fd, 0), [0, 1, 2, M, M, M], "filtered directed");

    let fr = Filtered::masked((&g).reverse(), &vmask, &emask).expect("masks");
    assert_eq!(dists(fr, 0), [0, 2, 1, M, M, M], "filtered reversed");

    let fu = Filtered::masked((&g).undirect(), &vmask, &emask).expect("masks");
    assert_eq!(dists(fu, 0), [0, 1, 1, M, M, M], "filtered undirected");
}

/// The distance map is sized for the graph's *bound*, and every slot in it is
/// written — including the slots of vertices a filtered view does not keep.
/// `graph_copy.cc:66-73` sizes from the filtered count and then writes at
/// unfiltered indices; here the two numbers are different types.
#[test]
fn every_slot_in_the_bound_is_written_even_on_a_filtered_view() {
    let g = fixture();
    let (vmask, emask) = masks();
    let fd = Filtered::masked(&g, &vmask, &emask).expect("masks");
    assert_eq!(fd.num_vertices(), 5, "the honest filtered cardinality");
    assert_eq!(fd.vertex_bound().len(), 6, "the allocation bound");

    let mut d = DenseProp::<i64, VertexTag>::new(g.graph_id());
    shortest_distances(fd, v(0), &mut d).expect("source");
    assert_eq!(d.len(), 6);
    assert_eq!(d.as_slice()[3], UNREACHABLE, "the filtered-out slot");
}

/// Unreached vertices carry `numeric_limits<dist_t>::max()`
/// (`topology/graph_distance.cc:277-279`), which the Python layer spells
/// `numpy.iinfo(dtype).max` (`topology/__init__.py:2097-2098`) — not `-1`,
/// and not "absent".
#[test]
fn unreached_is_the_integer_maximum() {
    let g = fixture();
    assert_eq!(dists(&g, 0)[5], i64::MAX);
    assert_eq!(UNREACHABLE, i64::MAX);
    // A second run over the same map must not leave the first run's numbers
    // behind: the whole run is refilled, not only what is reached.
    let mut d = DenseProp::<i64, VertexTag>::new(g.graph_id());
    shortest_distances(&g, v(0), &mut d).expect("source");
    shortest_distances(&g, v(5), &mut d).expect("source");
    assert_eq!(d.as_slice(), [M, M, M, M, 1, 0]);
}

#[test]
fn a_source_outside_the_bound_is_refused_by_every_entry_point() {
    let g = fixture();
    let bad = v(N + 3);
    let mut rec = Rec::default();
    assert_eq!(
        bfs(&g, bad, &mut rec),
        Err(gt_core::GraphError::NoSuchVertex(bad))
    );
    assert_eq!(
        dfs(&g, bad, &mut rec),
        Err(gt_core::GraphError::NoSuchVertex(bad))
    );
    assert!(rec.discovered.is_empty(), "nothing fired");

    let mut di = DenseProp::<i64, VertexTag>::new(g.graph_id());
    assert_eq!(
        shortest_distances(&g, bad, &mut di),
        Err(gt_core::GraphError::NoSuchVertex(bad))
    );
    let mut df = DenseProp::<f64, VertexTag>::new(g.graph_id());
    assert_eq!(
        dijkstra(&g, bad, &Unity::<f64, EdgeTag>::NEW, &mut df),
        Err(gt_core::GraphError::NoSuchVertex(bad))
    );
}

// ---------------------------------------------------------------------------
// 2. The event contract
// ---------------------------------------------------------------------------

/// `discover` once per vertex, `examine` once per incidence of an *expanded*
/// vertex (boost's `examine_edge` position: before the colour test, so
/// non-tree edges are examined too), `finish` once per expanded vertex.
#[test]
fn bfs_fires_the_three_events_in_the_boost_positions() {
    let g = fixture();
    let mut rec = Rec::default();
    bfs(&g, v(0), &mut rec).expect("source");

    assert_eq!(rec.order(), [0, 1, 2, 3, 4], "5 is unreachable");
    assert_eq!(rec.depths(), [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3)]);

    // Sum of the out-degrees of the five expanded vertices: 1 + 2 + 2 + 1 + 0.
    assert_eq!(rec.examined.len(), 6);
    // e2 (2->0) reaches an already-black vertex and is still examined; that is
    // the `non_tree_edge`/`black_target` case, which has no callback here but
    // must not swallow the `examine_edge` that precedes it.
    assert!(rec.examined.contains(&(2, 2)), "{:?}", rec.examined);
    // e4 (3->4) is examined although 4 was already discovered through e6.
    assert!(rec.examined.contains(&(3, 4)), "{:?}", rec.examined);

    assert_eq!(rec.finished, [0, 1, 2, 3, 4], "FIFO order, one per vertex");
}

/// A multi-source BFS runs one frontier, so a vertex's depth is its distance
/// to the *nearest* source. Boost has no such entry point, and `do_bfs`'s
/// null-source branch (`search/graph_bfs.cc:116-124`) is a different thing: a
/// forest, with the depth restarting at every root.
#[test]
fn bfs_multi_puts_every_source_at_depth_zero() {
    let g = fixture();
    let mut rec = Rec::default();
    bfs_multi(&g, [v(0), v(5)], &mut rec).expect("sources");

    assert_eq!(
        rec.depths(),
        [(0, 0), (1, 1), (2, 2), (3, 2), (4, 1), (5, 0)],
        "4 is one hop from 5, not three from 0"
    );
    // Both sources are discovered before either is expanded.
    assert_eq!(rec.order()[..2], [0, 5]);
}

#[test]
fn bfs_multi_discovers_a_repeated_source_once() {
    let g = fixture();
    let mut rec = Rec::default();
    bfs_multi(&g, [v(0), v(0), v(1)], &mut rec).expect("sources");
    assert_eq!(rec.order(), [0, 1, 2, 3, 4]);
    assert_eq!(rec.depths(), [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)]);
}

#[test]
fn bfs_multi_reports_a_bad_source_and_keeps_the_good_ones() {
    let g = fixture();
    let bad = v(N);
    let mut rec = Rec::default();
    assert_eq!(
        bfs_multi(&g, [v(0), bad], &mut rec),
        Err(gt_core::GraphError::NoSuchVertex(bad))
    );
    assert_eq!(rec.order(), [0], "consumed lazily, as documented");
}

/// `Prune` at `discover`: the vertex exists, and its neighbourhood is not
/// looked at. `finish(v)` and "the neighbours of v were examined" are the same
/// statement, so a pruned vertex is not finished.
#[test]
fn prune_at_discover_suppresses_expansion_and_the_finish() {
    let g = fixture();
    let mut vis = Answering::at_vertex(1, Control::Prune);
    bfs(&g, v(0), &mut vis).expect("source");

    assert_eq!(vis.rec.order(), [0, 1], "2 and 3 are only reachable via 1");
    assert_eq!(vis.rec.examined, [(0, 0)], "1's out-edges were never read");
    assert_eq!(vis.rec.finished, [0], "1 was pruned, so never finished");

    // The same answer in a DFS.
    let mut vis = Answering::at_vertex(1, Control::Prune);
    dfs(&g, v(0), &mut vis).expect("source");
    assert_eq!(vis.rec.order(), [0, 1]);
    assert_eq!(vis.rec.finished, [0]);
}

/// `Prune` at `examine`: one incidence is dropped; the traversal continues
/// with the rest of that vertex's incidences.
#[test]
fn prune_at_examine_drops_one_edge_and_no_more() {
    let g = fixture();
    // e1 is 1->2. Dropping it leaves 2 reachable only through... nothing: 2's
    // only in-edge is e1. So 2 and, with it, e6 to 4 disappear, but 3 (via e3)
    // and 4 (via e4) survive.
    let mut vis = Answering::at_edge(1, Control::Prune);
    bfs(&g, v(0), &mut vis).expect("source");
    assert_eq!(vis.rec.order(), [0, 1, 3, 4]);
    assert_eq!(vis.rec.finished, [0, 1, 3, 4]);
    // The pruned edge was still examined — pruning is the answer to the
    // callback, not a way of skipping it.
    assert!(vis.rec.examined.contains(&(1, 1)));
}

/// `Stop` ends the traversal, and it is `Ok`: `stop_search`
/// (`topology/graph_distance.cc:42`) is thrown to end a *successful* search
/// and caught at `:314` with an empty handler.
#[test]
fn stop_ends_the_traversal_and_is_not_an_error() {
    let g = fixture();

    let mut vis = Answering::at_vertex(2, Control::Stop);
    assert_eq!(bfs(&g, v(0), &mut vis), Ok(()));
    assert_eq!(vis.rec.order(), [0, 1, 2]);
    assert_eq!(vis.rec.finished, [0], "1 was in flight when 2 stopped it");

    let mut vis = Answering::at_edge(0, Control::Stop);
    assert_eq!(bfs(&g, v(0), &mut vis), Ok(()));
    assert_eq!(vis.rec.order(), [0], "stopped on the first incidence");
    assert!(vis.rec.finished.is_empty());

    let mut vis = Answering::at_vertex(2, Control::Stop);
    assert_eq!(dfs(&g, v(0), &mut vis), Ok(()));
    assert_eq!(vis.rec.order(), [0, 1, 2]);
    assert!(vis.rec.finished.is_empty(), "nothing had finished yet");
}

// ---------------------------------------------------------------------------
// 3. DFS
// ---------------------------------------------------------------------------

/// Every reachable vertex, exactly once, on all six views.
#[test]
fn dfs_visits_every_reachable_vertex_exactly_once() {
    let g = fixture();
    let (vmask, emask) = masks();

    fn check<G: GraphRef + VertexList + Copy>(g: G, expected: &[usize], what: &str) {
        let mut rec = Rec::default();
        dfs(g, v(0), &mut rec).expect("source");
        let mut got = rec.order();
        assert_eq!(got.len(), rec.discovered.len(), "{what}: no repeats");
        got.sort_unstable();
        assert_eq!(got, expected, "{what}");
        // Everything discovered was finished, because nothing pruned.
        let mut fin = rec.finished.clone();
        fin.sort_unstable();
        assert_eq!(fin, expected, "{what}: finish pairs with discover");
        // The reachable set is the BFS one, by construction.
        let bfs_reached: Vec<usize> = dists(g, 0)
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d != UNREACHABLE)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(bfs_reached, expected, "{what}: agrees with BFS");
    }

    check(&g, &[0, 1, 2, 3, 4], "directed");
    check((&g).reverse(), &[0, 1, 2], "reversed");
    check((&g).undirect(), &[0, 1, 2, 3, 4, 5], "undirected");
    check(
        Filtered::masked(&g, &vmask, &emask).expect("masks"),
        &[0, 1, 2],
        "filtered directed",
    );
    check(
        Filtered::masked((&g).reverse(), &vmask, &emask).expect("masks"),
        &[0, 1, 2],
        "filtered reversed",
    );
    check(
        Filtered::masked((&g).undirect(), &vmask, &emask).expect("masks"),
        &[0, 1, 2],
        "filtered undirected",
    );
}

/// The visit order of a directed DFS, pinned. This is assertable only because
/// the out-half of a block is in insertion order (`Block::insert_out`), and it
/// is worth pinning because it is what separates a real DFS from a
/// "repeatedly expand the deepest frontier" imitation: `finish` is a genuine
/// post-order, produced by keeping the *iterator* on the stack the way
/// `depth_first_visit` keeps `(u, ei, ei_end)`.
#[test]
fn dfs_is_a_preorder_with_a_post_order_finish() {
    let g = fixture();
    let mut rec = Rec::default();
    dfs(&g, v(0), &mut rec).expect("source");

    assert_eq!(rec.order(), [0, 1, 2, 4, 3]);
    assert_eq!(rec.depths(), [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3)]);
    assert_eq!(rec.finished, [4, 2, 3, 1, 0]);
    // A post-order: a vertex finishes after every vertex it discovered, so the
    // root is last and the first finish is a leaf of the tree. A
    // "re-expand the deepest frontier" imitation finishes 4 then 3 then 2,
    // because it has forgotten where it was in 2's incidence list.
    assert_eq!(rec.finished.last(), Some(&0), "the root finishes last");
    assert_eq!(rec.finished.first(), Some(&4), "the first leaf finishes first");
}

/// A DFS depth is the path length along the tree edge that discovered the
/// vertex, which is *not* the shortest-path distance: vertex 4 sits at DFS
/// depth 3 and BFS depth 3 here only by accident of the fixture, while vertex
/// 3 sits at DFS depth 2 reached through a different parent than in BFS.
#[test]
fn dfs_depth_is_the_tree_depth_not_the_distance() {
    let mut g = AdjList::with_vertices(3);
    g.add_edge(v(0), v(1)).expect("e0");
    g.add_edge(v(1), v(2)).expect("e1");
    g.add_edge(v(0), v(2)).expect("e2");

    let mut rec = Rec::default();
    dfs(&g, v(0), &mut rec).expect("source");
    assert_eq!(rec.depths(), [(0, 0), (1, 1), (2, 2)], "down the long path");

    let mut rec = Rec::default();
    bfs(&g, v(0), &mut rec).expect("source");
    assert_eq!(rec.depths(), [(0, 0), (1, 1), (2, 1)], "the short one");
}

// ---------------------------------------------------------------------------
// 4. Dijkstra
// ---------------------------------------------------------------------------

/// A weight map that is *labelled* unity and would answer nonsense if read.
///
/// This is the acceptance criterion "the weight loop folded away", made into
/// an assertion rather than a disassembly listing: under `W::IS_UNITY` the
/// kernel must take the BFS arm, and the BFS arm never touches the weight
/// map. A `get_ref` that panics fails the test at the exact call that should
/// not exist, on every optimisation level and every target, which an
/// `objdump` pattern would not.
///
/// The disassembly was read once, by hand, and says the same thing more
/// strongly:
///
/// ```text
/// cargo rustc --release -p gt-algo --test u13_traversal -- --emit=asm
/// ```
///
/// emits `gt_algo::traversal::dijkstra::<&AdjList, Unity<f64, EdgeTag>>` as a
/// prologue and a `jmp` straight into
/// `gt_algo::traversal::bfs_from::<&AdjList, dijkstra::Hops, Once<VertexId>>`.
/// The instantiation contains no `BinaryHeap` sift, no `total_cmp`, no
/// `addsd` and no load from the weight map: the whole relaxation loop is
/// absent from the object code rather than merely unexecuted. That listing
/// cannot be asserted portably, so it is recorded here and `PoisonUnity`
/// guards the property that matters.
struct PoisonUnity;

impl ReadProp<EdgeTag> for PoisonUnity {
    type Value = f64;
    type Ref<'s> = Owned<f64>;
    const IS_UNITY: bool = true;
    const IS_CONSTANT: bool = true;
    fn get_ref(&self, k: EdgeId) -> Owned<f64> {
        panic!("the unity fast path read the weight map at {k:?}");
    }
}

fn weighted<G: GraphRef + VertexList>(g: G, source: usize, w: &[f64]) -> Vec<f64> {
    let mut wm = DenseProp::<f64, EdgeTag>::new(g.graph_id());
    {
        let mut view = wm.sized_for(g.edge_bound()).expect("same graph");
        view.as_mut_slice().copy_from_slice(w);
    }
    let mut d = DenseProp::<f64, VertexTag>::new(g.graph_id());
    dijkstra(g, v(source), &wm, &mut d).expect("source");
    d.as_slice().to_vec()
}

fn unity_dists<G: GraphRef + VertexList>(g: G, source: usize) -> Vec<f64> {
    let mut d = DenseProp::<f64, VertexTag>::new(g.graph_id());
    dijkstra(g, v(source), &Unity::<f64, EdgeTag>::NEW, &mut d).expect("source");
    d.as_slice().to_vec()
}

fn as_f64(d: &[i64]) -> Vec<f64> {
    d.iter()
        .map(|&x| {
            if x == UNREACHABLE {
                f64::INFINITY
            } else {
                x as f64
            }
        })
        .collect()
}

/// The acceptance criterion: a unity weight equals the unweighted kernel, on
/// every view.
#[test]
fn dijkstra_with_a_unity_weight_equals_shortest_distances() {
    let g = fixture();
    let (vmask, emask) = masks();

    assert_eq!(unity_dists(&g, 0), as_f64(&dists(&g, 0)), "directed");
    assert_eq!(
        unity_dists((&g).reverse(), 0),
        as_f64(&dists((&g).reverse(), 0)),
        "reversed"
    );
    assert_eq!(
        unity_dists((&g).undirect(), 0),
        as_f64(&dists((&g).undirect(), 0)),
        "undirected"
    );
    let fd = Filtered::masked(&g, &vmask, &emask).expect("masks");
    assert_eq!(unity_dists(fd, 0), as_f64(&dists(fd, 0)), "filtered");
    assert_eq!(
        unity_dists(&g, 5),
        [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0, 0.0],
        "unreached is `inf`, the floating-point half of graph_distance.cc:333"
    );
}

#[test]
fn a_unity_weight_map_is_never_read() {
    let g = fixture();
    let mut d = DenseProp::<f64, VertexTag>::new(g.graph_id());
    dijkstra(&g, v(0), &PoisonUnity, &mut d).expect("source");
    assert_eq!(d.as_slice(), as_f64(&dists(&g, 0)));

    // ... and on an undirected view, where the incidence count is higher and
    // a stray read would be likelier.
    let mut d = DenseProp::<f64, VertexTag>::new(g.graph_id());
    dijkstra((&g).undirect(), v(0), &PoisonUnity, &mut d).expect("source");
    assert_eq!(d.as_slice(), as_f64(&dists((&g).undirect(), 0)));
}

/// A constant map is *not* a unity map unless its constant is one, so this
/// goes through the heap. `graph_selectors.hh:109`'s constant-weight fast path
/// cannot: it reads `weight.c` on a class whose member is the private `_c`
/// (`graph_properties.hh:677`), so the overload is uninstantiable dead code.
#[test]
fn a_constant_weight_goes_through_the_heap_and_scales_the_answer() {
    let g = fixture();
    let two = Constant::<f64, EdgeTag>::new(2.0);
    const { assert!(!<Constant<f64, EdgeTag> as ReadProp<EdgeTag>>::IS_UNITY) };
    const { assert!(<Constant<f64, EdgeTag> as ReadProp<EdgeTag>>::IS_CONSTANT) };

    let mut d = DenseProp::<f64, VertexTag>::new(g.graph_id());
    dijkstra(&g, v(0), &two, &mut d).expect("source");
    let want: Vec<f64> = as_f64(&dists(&g, 0)).iter().map(|x| x * 2.0).collect();
    assert_eq!(d.as_slice(), want);
}

/// The whole point of a weighted search: the cheapest path is not the shortest
/// one. `0 -> 1` costs 10 directly and 2 through `0 -> 2 -> 1`.
#[test]
fn dijkstra_prefers_a_longer_cheaper_path() {
    let mut g = AdjList::with_vertices(4);
    g.add_edge(v(0), v(1)).expect("e0"); //  10
    g.add_edge(v(0), v(2)).expect("e1"); //   1
    g.add_edge(v(2), v(1)).expect("e2"); //   1
    g.add_edge(v(1), v(3)).expect("e3"); //   1
    g.add_edge(v(2), v(3)).expect("e4"); // 100

    let d = weighted(&g, 0, &[10.0, 1.0, 1.0, 1.0, 100.0]);
    assert_eq!(d, [0.0, 2.0, 1.0, 3.0]);
    // The unweighted kernel answers the other question, and differs.
    assert_eq!(dists(&g, 0), [0, 1, 1, 2]);
}

/// A zero-weight edge must not spin: the lazy-deletion pop test is `>`, not
/// `>=`, so an entry whose key equals the recorded distance is still expanded
/// once and a re-push only happens on a strict improvement.
#[test]
fn zero_weights_and_parallel_edges_terminate() {
    let mut g = AdjList::with_vertices(3);
    g.add_edge(v(0), v(1)).expect("e0"); // 0.0
    g.add_edge(v(0), v(1)).expect("e1"); // 0.0, a parallel edge
    g.add_edge(v(1), v(0)).expect("e2"); // 0.0, back again
    g.add_edge(v(1), v(2)).expect("e3"); // 5.0
    let d = weighted(&g, 0, &[0.0, 0.0, 0.0, 5.0]);
    assert_eq!(d, [0.0, 0.0, 5.0]);
}

#[test]
fn a_self_loop_is_examined_and_changes_nothing() {
    let mut g = AdjList::with_vertices(2);
    g.add_edge(v(0), v(0)).expect("e0");
    g.add_edge(v(0), v(1)).expect("e1");

    let mut rec = Rec::default();
    bfs(&g, v(0), &mut rec).expect("source");
    assert_eq!(rec.order(), [0, 1]);
    assert!(rec.examined.contains(&(0, 0)), "the loop was examined");
    assert_eq!(weighted(&g, 0, &[7.0, 1.0]), [0.0, 1.0]);
}

// ---------------------------------------------------------------------------
// 5. Properties
// ---------------------------------------------------------------------------

mod props {
    use super::*;
    use proptest::prelude::*;

    /// A textbook Bellman–Ford over the global edge list, which shares no code
    /// with the kernel under test: a different algorithm, a different data
    /// structure, and no priority queue at all. `do_bf_search`
    /// (`topology/graph_distance.cc:415`) is the same fallback graph-tool
    /// keeps for the negative-weight case.
    fn bellman_ford(g: &AdjList, source: usize, w: &[f64]) -> Vec<f64> {
        let n = g.num_vertices();
        let mut d = vec![f64::INFINITY; n];
        d[source] = 0.0;
        for _ in 0..n {
            let mut changed = false;
            for e in g.edges() {
                let (s, t) = (e.source().index(), e.target().index());
                let alt = d[s] + w[e.id().index()];
                if alt < d[t] {
                    d[t] = alt;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        d
    }

    fn graph_and_weights() -> impl Strategy<Value = (usize, Vec<(usize, usize)>, Vec<f64>)> {
        (2usize..12).prop_flat_map(|n| {
            let edges = prop::collection::vec((0..n, 0..n), 0..30);
            edges.prop_flat_map(move |es| {
                let m = es.len();
                // Small integers, exactly representable, so the two algorithms
                // sum the same values to the same bits whatever the order.
                let ws = prop::collection::vec(0u32..12, m..=m)
                    .prop_map(|v| v.into_iter().map(f64::from).collect::<Vec<f64>>());
                (Just(n), Just(es), ws)
            })
        })
    }

    fn build(n: usize, es: &[(usize, usize)]) -> AdjList {
        let mut g = AdjList::with_vertices(n);
        for &(s, t) in es {
            g.add_edge(v(s), v(t)).expect("in range");
        }
        g
    }

    proptest! {
        /// The heap kernel against Bellman–Ford, on non-negative weights.
        #[test]
        fn dijkstra_agrees_with_bellman_ford((n, es, ws) in graph_and_weights()) {
            let g = build(n, &es);
            for s in 0..n {
                let got = weighted(&g, s, &ws);
                prop_assert_eq!(got, bellman_ford(&g, s, &ws), "source {}", s);
            }
        }

        /// The unity arm against the unweighted kernel — the same claim as the
        /// hand table, over arbitrary shapes.
        #[test]
        fn unity_dijkstra_is_bfs((n, es, _ws) in graph_and_weights()) {
            let g = build(n, &es);
            for s in 0..n {
                prop_assert_eq!(unity_dists(&g, s), as_f64(&dists(&g, s)));
                prop_assert_eq!(unity_dists((&g).undirect(), s),
                                as_f64(&dists((&g).undirect(), s)));
            }
        }

        /// BFS and DFS disagree about order and agree about reachability, on
        /// every view. A `Prune`-free traversal finishes everything it
        /// discovers.
        #[test]
        fn bfs_and_dfs_reach_the_same_set((n, es, _ws) in graph_and_weights()) {
            let g = build(n, &es);
            for s in 0..n {
                let mut b = Rec::default();
                bfs(&g, v(s), &mut b).expect("source");
                let mut d = Rec::default();
                dfs(&g, v(s), &mut d).expect("source");

                let (mut bo, mut dd) = (b.order(), d.order());
                bo.sort_unstable();
                dd.sort_unstable();
                prop_assert_eq!(&bo, &dd);

                let mut bf = b.finished.clone();
                bf.sort_unstable();
                prop_assert_eq!(&bf, &bo, "everything discovered was finished");

                // The BFS depth is never larger than the DFS depth.
                let dfs_depth: std::collections::HashMap<usize, usize> =
                    d.discovered.iter().copied().collect();
                for (u, depth) in b.discovered {
                    prop_assert!(depth <= dfs_depth[&u]);
                }
            }
        }

        /// One `examine` per incidence of an expanded vertex, and the
        /// undirected view examines each edge from both of its endpoints.
        #[test]
        fn examine_counts_the_incidences((n, es, _ws) in graph_and_weights()) {
            let g = build(n, &es);
            let u = (&g).undirect();
            let mut rec = Rec::default();
            bfs(u, v(0), &mut rec).expect("source");
            let expected: usize = rec.finished.iter().map(|&x| u.out_degree(v(x))).sum();
            prop_assert_eq!(rec.examined.len(), expected);
        }
    }
}
