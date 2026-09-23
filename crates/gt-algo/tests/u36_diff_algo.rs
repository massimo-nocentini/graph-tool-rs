//! Differential tests: traversal, distances and components, against
//! graph-tool 3.8's `topology/graph_distance.cc`,
//! `topology/graph_components.hh` and the Python wrappers that decide what a
//! caller actually gets.
//!
//! Every expected table below is hand-derived from the fixture, and every
//! convention it encodes is quoted from the C++ or the Python beside it.
//!
//! ## The fixture
//!
//! Six vertices, six edges, added in this order (so the edge added `i`th is
//! `e_i`, and every out-half is in insertion order):
//!
//! ```text
//!        e0        e1
//!   0 ───────> 1 ───────> 2 ───e3──> 3 ──┐
//!              ^          ^              │ e5 (self-loop)
//!              └───e2─────┘         <────┘
//!                         ^
//!                         └─e4── 4          5 is isolated
//! ```
//!
//! It is chosen so that every view disagrees with every other: `4` is
//! reachable from `0` only undirected, `0` is reachable from `3` only
//! reversed, `{1, 2}` is the one non-trivial strongly connected component, and
//! `5` is isolated so the "number of components" answers differ.
//!
//! ## The three conventions this file pins
//!
//! 1. **The unreachable sentinel.** `shortest_distance` fills the map with
//!    `numpy.iinfo(dtype).max` before the search
//!    (`graph_tool/topology/__init__.py:2097-2100`) and the C++ then writes
//!    only the vertices `tree_edge` reaches (`graph_distance.cc:298-306`), so
//!    an unreached vertex keeps the type's maximum --- 2147483647 for the
//!    default `int32_t`, and visible as such in the docstring's own doctest
//!    (`__init__.py:2026-2038`). The port's distance map is `i64`, so its
//!    sentinel is `i64::MAX`; the *convention* is the same and a vertex that
//!    is merely far away can never collide with it.
//!
//! 2. **Which view you search is the whole question.** `shortest_distance(g,
//!    directed=False)` builds `GraphView(g, directed=False)`
//!    (`__init__.py:2088-2090`), i.e. the `undirected_adaptor`, whose
//!    `out_edges` is the whole incidence block
//!    (`graph_adaptor.hh:199-207`). Passing `directed=None` leaves the graph
//!    as it is. Here that choice is `g` versus `(&g).undirect()` versus
//!    `(&g).reverse()`, and it is a type, not a keyword argument.
//!
//! 3. **`label_components` is two algorithms.** `graph_components.hh:102-129`
//!    tag-dispatches on `graph_traits<Graph>::directed_category`:
//!    `boost::strong_components` for a directed graph and
//!    `boost::connected_components` otherwise. A caller who wants *weak*
//!    components has to remember `GraphView(g, directed=False)` and gets the
//!    strong ones silently if they forget. Here they are two functions with
//!    different bounds.

use gt_algo::components::{UNLABELED, components, strong_components};
use gt_algo::topology::{is_bipartite, is_dag, kcore_decomposition, topological_sort};
use gt_algo::traversal::{Control, UNREACHABLE, Visitor, bfs, dijkstra, shortest_distances};
use gt_core::adj::{AdjList, Incident};
use gt_core::graph::{GraphRef, VertexList};
use gt_core::ids::{EdgeId, EdgeTag, VertexId, VertexTag};
use gt_core::prop::DenseProp;
use gt_core::view::{Reverse, Undirect};

const N: usize = 6;
/// `(source, target)` in `add_edge` order.
const EDGES: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 1), (2, 3), (4, 2), (3, 3)];

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

fn dist_of<G: GraphRef + VertexList>(g: G, source: usize) -> Vec<i64> {
    let mut d: DenseProp<i64, VertexTag> = DenseProp::new(g.graph_id());
    shortest_distances(g, v(source), &mut d).expect("the source exists");
    d.as_slice().to_vec()
}

fn labels_of<G: GraphRef + VertexList>(g: G) -> (Vec<i64>, usize) {
    let mut l: DenseProp<i64, VertexTag> = DenseProp::new(g.graph_id());
    let n = components(g, &mut l);
    (l.as_slice().to_vec(), n)
}

const INF: i64 = UNREACHABLE;

// ---------------------------------------------------------------------------
// 1. Unweighted distances, one table per view
// ---------------------------------------------------------------------------

/// The three views give three different distance tables from the same source.
///
/// Hand-derived from the fixture:
///
/// ```text
///   directed   from 0:  0  1  2  3  INF INF     (4 is upstream, 5 isolated)
///   undirected from 0:  0  1  2  3   3  INF     (4 arrives via 2)
///   reversed   from 0:  0 INF INF INF INF INF   (nothing points at 0)
///   reversed   from 3:  3  2  1  0   2  INF
/// ```
///
/// The reversed table from 3 is the transpose read: `3 <- 2 <- 1 <- 0` gives
/// `d[0] = 3`, and `3 <- 2 <- 4` gives `d[4] = 2`. The self-loop `e5` changes
/// nothing in any of them --- `tree_edge` only fires on an edge to an unvisited
/// vertex (`graph_distance.cc:298`).
#[test]
fn the_distance_table_is_a_function_of_the_view() {
    let g = fixture();
    assert_eq!(dist_of(&g, 0), vec![0, 1, 2, 3, INF, INF], "directed");
    assert_eq!(
        dist_of((&g).undirect(), 0),
        vec![0, 1, 2, 3, 3, INF],
        "undirected: `_all_edges_out` reaches 4 from 2"
    );
    assert_eq!(
        dist_of((&g).reverse(), 0),
        vec![0, INF, INF, INF, INF, INF],
        "reversed: nothing points at 0"
    );
    assert_eq!(dist_of((&g).reverse(), 3), vec![3, 2, 1, 0, 2, INF]);
    // The reversed table from `t` is the directed table *to* `t`: a distance
    // of 3 from 0 to 3 in the forward graph, and of 3 from 3 to 0 reversed.
    assert_eq!(dist_of(&g, 0)[3], dist_of((&g).reverse(), 3)[0]);
    // Undirecting is symmetric; reversing a directed graph is not.
    for s in 0..N {
        let du = dist_of((&g).undirect(), s);
        for (t, &d) in du.iter().enumerate() {
            assert_eq!(
                d,
                dist_of((&g).undirect(), t)[s],
                "undirected distance {s}->{t} is not symmetric"
            );
        }
    }
}

/// An unreached vertex keeps the sentinel, and the sentinel is not a distance.
///
/// `dist_map.set_value(numpy.iinfo(dtype).max)` then `dist_map[source] = 0`
/// (`__init__.py:2097-2100`, `graph_distance.cc:283`): everything the BFS does
/// not reach is left at the type maximum. An implementation that filled with
/// `-1`, or that left the map's previous contents in place, would be
/// indistinguishable on a connected graph and wrong on this one.
#[test]
fn the_unreachable_sentinel_is_the_type_maximum_and_the_source_is_zero() {
    let g = fixture();
    let d = dist_of(&g, 4);
    assert_eq!(d[4], 0, "`dist_map[source] = 0` (graph_distance.cc:283)");
    assert_eq!(
        d,
        vec![INF, 2, 1, 2, 0, INF],
        "4 -> 2 at 1; 2 -> 1 and 2 -> 3 at 2; 0 and 5 are unreachable"
    );
    assert_eq!(UNREACHABLE, i64::MAX);
    // Nothing a real BFS can produce collides with it: the largest finite
    // distance on any graph is `num_vertices - 1`.
    let finite_max = d.iter().copied().filter(|&x| x != INF).max().unwrap();
    assert!(finite_max < N as i64);

    // The map is *re-initialised* by each run, not accumulated into --- the
    // Python re-fills before every call.
    let mut m: DenseProp<i64, VertexTag> = DenseProp::new(g.graph_id());
    shortest_distances(&g, v(0), &mut m).expect("source exists");
    assert_eq!(m.as_slice(), &[0, 1, 2, 3, INF, INF]);
    shortest_distances(&g, v(4), &mut m).expect("source exists");
    assert_eq!(
        m.as_slice(),
        &[INF, 2, 1, 2, 0, INF],
        "a second run must not inherit the first's reachability"
    );
}

/// `bfs` visits in layers and reports the same depths the distance map holds.
///
/// `boost::breadth_first_visit`'s `tree_edge` sets `d = _dist_map[u] + 1`
/// (`graph_distance.cc:296-302`), so a visitor's depth and the map agree by
/// construction. A self-loop and a back edge must not re-fire it.
#[test]
fn bfs_depths_agree_with_the_distance_map_and_ignore_back_edges() {
    let g = fixture();

    struct Rec {
        seen: Vec<(usize, usize)>,
        examined: usize,
    }
    impl Visitor for Rec {
        fn discover(&mut self, v: VertexId, depth: usize) -> Control {
            self.seen.push((v.index(), depth));
            Control::Continue
        }
        fn examine(&mut self, _from: VertexId, _e: Incident) -> Control {
            self.examined += 1;
            Control::Continue
        }
    }

    let mut r = Rec {
        seen: Vec::new(),
        examined: 0,
    };
    bfs(&g, v(0), &mut r).expect("the source exists");
    assert_eq!(
        r.seen,
        vec![(0, 0), (1, 1), (2, 2), (3, 3)],
        "layer order; 4 and 5 are unreachable from 0"
    );
    // Every out-edge of every discovered vertex is examined exactly once:
    // v0 has 1, v1 has 1, v2 has 2, v3 has 1 (the self-loop).
    assert_eq!(r.examined, 5);

    let d = dist_of(&g, 0);
    for (u, depth) in r.seen {
        assert_eq!(d[u], depth as i64, "visitor depth vs distance map at {u}");
    }
}

/// Dijkstra with unit weights reproduces the BFS table, and leaves the
/// unreached at infinity.
///
/// `shortest_distance` picks the distance type from the weight map
/// (`__init__.py:2070-2073`) and fills with `numpy.inf` for a floating map
/// (`:2100`), which is `graph_distance.cc:277-279`'s
/// `is_floating_point ? infinity() : max()`.
#[test]
fn dijkstra_with_unit_weights_is_the_bfs_table_and_infinity_is_the_sentinel() {
    let g = fixture();
    let mut w: DenseProp<f64, EdgeTag> = DenseProp::from_vec(g.graph_id(), vec![1.0; 6]);
    let mut d: DenseProp<f64, VertexTag> = DenseProp::new(g.graph_id());
    dijkstra(
        &g,
        v(0),
        &w.view(g.edge_bound()).expect("same graph"),
        &mut d,
    )
    .expect("the source exists");
    assert_eq!(
        d.as_slice(),
        &[0.0, 1.0, 2.0, 3.0, f64::INFINITY, f64::INFINITY]
    );

    // A weight that makes the long way round cheaper: `0 -> 1` costs 10, so
    // there is no alternative and the table only scales.
    w.as_mut_slice()[0] = 10.0;
    let mut d2: DenseProp<f64, VertexTag> = DenseProp::new(g.graph_id());
    dijkstra(
        &g,
        v(0),
        &w.view(g.edge_bound()).expect("same graph"),
        &mut d2,
    )
    .expect("the source exists");
    assert_eq!(
        d2.as_slice(),
        &[0.0, 10.0, 11.0, 12.0, f64::INFINITY, f64::INFINITY]
    );
}

// ---------------------------------------------------------------------------
// 2. Components: the two algorithms `label_components` hides behind one name
// ---------------------------------------------------------------------------

/// `components` on the directed view is the out-reachability partition, and it
/// is **not** what `label_components` computes for a directed graph.
///
/// `graph_components.hh:102-129` sends a directed graph to
/// `boost::strong_components` and only an undirected one to
/// `boost::connected_components` (which additionally carries
/// `BOOST_STATIC_ASSERT((is_same<directed, undirected_tag>::value))`, so the
/// directed instantiation does not even compile). A Python caller reaches the
/// weak components only by writing `GraphView(g, directed=False)`.
///
/// Hand-derived labels. `components` numbers by root order over
/// `vertices()`, which is `components_recorder::start_vertex` incrementing
/// once per DFS root:
///
/// ```text
///   directed:    root 0 takes {0,1,2,3}=0, root 4 takes {4}=1, root 5 = 2  -> 3
///   undirected:  root 0 takes {0,1,2,3,4}=0, root 5 = 1                    -> 2
/// ```
///
/// Note that the directed answer is not a partition into *weak* components
/// and is not even symmetric in the vertex order: it is "which root reached
/// you first", which is exactly why the port refuses to guess.
#[test]
fn connected_components_needs_the_undirected_view_to_mean_what_it_says() {
    let g = fixture();

    assert_eq!(
        labels_of(&g),
        (vec![0, 0, 0, 0, 1, 2], 3),
        "directed: out-reachability from each unvisited root"
    );
    assert_eq!(
        labels_of((&g).undirect()),
        (vec![0, 0, 0, 0, 0, 1], 2),
        "undirected: the weakly connected components"
    );
    // Reversing gives the *in*-reachability partition, a third answer again:
    // root 0 has no in-edges at all, root 1 reaches {1, 2, 4} backwards, root
    // 3 reaches only itself (its in-edges are `2 -> 3`, already labelled, and
    // its own self-loop), and 5 is alone.
    assert_eq!(labels_of((&g).reverse()), (vec![0, 1, 1, 2, 1, 3], 4));

    // Only the undirected labelling is an equivalence relation: `u ~ v` iff
    // they share a label, symmetric in the roles of `u` and `v`.
    let (lu, _) = labels_of((&g).undirect());
    for &(s, t) in EDGES.iter() {
        assert_eq!(lu[s], lu[t], "an edge cannot cross a weak component");
    }
}

/// `strong_components` numbers in the order SCC roots finish, which makes the
/// labels a reverse topological order of the condensation.
///
/// `boost::tarjan_scc_visitor::finish_vertex` assigns the component and then
/// increments `c_count`, so the first SCC to complete is 0. Hand-traced on the
/// fixture, with `out_edges` in insertion order:
///
/// ```text
///   root 0: disc 0 -> 1 -> 2; 2 -> 1 is a back edge (low[2] = 1);
///           2 -> 3 discovers 3, whose only edge is its own self-loop,
///           so 3 finishes first                      -> {3}    = 0
///           1 finishes as the root of its SCC        -> {1, 2} = 1
///           0 finishes                               -> {0}    = 2
///   root 4: 4 -> 2 is a cross edge to a finished SCC -> {4}    = 3
///   root 5:                                             {5}    = 4
/// ```
///
/// The condensation property is then `label[u] >= label[v]` for every edge
/// `u -> v`, with equality exactly on the edges inside `{1, 2}`.
#[test]
fn strong_components_are_numbered_in_reverse_topological_order() {
    let g = fixture();
    let mut l: DenseProp<i64, VertexTag> = DenseProp::new(g.graph_id());
    let n = strong_components(&g, &mut l);
    let lab = l.as_slice().to_vec();

    assert_eq!(n, 5, "only {{1, 2}} is a non-trivial SCC");
    assert_eq!(lab, vec![2, 1, 1, 0, 3, 4]);

    for &(s, t) in EDGES.iter() {
        assert!(
            lab[s] >= lab[t],
            "edge {s} -> {t} violates the reverse topological order \
             ({} < {})",
            lab[s],
            lab[t]
        );
    }
    assert_eq!(lab[1], lab[2], "1 and 2 are mutually reachable");
    assert_ne!(lab[0], lab[1], "0 is not reachable from 1");

    // A self-loop does not merge anything: `3 -> 3` leaves `{3}` a singleton,
    // which is what `boost::strong_components` does and what a naive
    // "reachable both ways" implementation would get wrong only if it treated
    // the loop as a witness for something else.
    assert_eq!(lab.iter().filter(|&&x| x == lab[3]).count(), 1);
}

/// A vertex the view does not expose reads [`UNLABELED`], where the C++ leaves
/// whatever the property map held.
///
/// `label_components` is documented to write labels "from 0 to N-1"
/// (`graph_components.hh:98-100`) and its `HistogramPropertyMap` counts only
/// the vertices it writes (`:70-77`) --- but on a filtered graph the map is
/// sized for the *unfiltered* index space and the filtered-out vertices keep
/// the map's default, which for a fresh `int32_t` map is `0`, i.e. a valid
/// label. So `c.fa == h.argmax()` (`__init__.py:1431`) can pick up vertices
/// that were never in the view.
#[test]
fn a_vertex_outside_the_view_is_unlabelled_rather_than_label_zero() {
    let g = fixture();
    let (lab, _) = labels_of(&g);
    assert!(
        lab.iter().all(|&x| x != UNLABELED),
        "nothing is filtered here"
    );

    // The same run on a mask that drops vertex 0.
    let mut vmask = vec![1u8; N];
    vmask[0] = 0;
    let emask = vec![1u8; g.edge_bound().len()];
    let f = gt_core::view::Filtered::masked(&g, &vmask, &emask).expect("masks cover");
    let (lab, n) = labels_of(f);
    assert_eq!(lab[0], UNLABELED, "vertex 0 is not in the view");
    assert_eq!(
        (lab[1], lab[2], lab[3]),
        (0, 0, 0),
        "root 1 takes {{1, 2, 3}}"
    );
    assert_eq!((lab[4], lab[5]), (1, 2));
    assert_eq!(n, 3, "the count excludes the filtered vertex's component");
    assert_eq!(UNLABELED, -1);
}

// ---------------------------------------------------------------------------
// 3. Topological order and bipartiteness
// ---------------------------------------------------------------------------

/// `topological_sort` returns sources first, which is boost's output
/// **reversed**.
///
/// `boost::topological_sort` emits in *finish* order, i.e. reverse
/// topological; graph-tool's Python wrapper flips it with
/// `topological_order.a[::-1].copy()`
/// (`graph_tool/topology/__init__.py:1254`), so what a caller sees has every
/// edge pointing forwards in the sequence. A port that returned boost's raw
/// vector would be exactly backwards and would still "look like" a
/// topological order to a reader.
#[test]
fn topological_sort_returns_sources_first_as_the_python_wrapper_does() {
    // The fixture has a cycle and a self-loop, so it is not a DAG at all.
    let g = fixture();
    assert!(!is_dag(&g));
    assert!(
        topological_sort(&g).is_none(),
        "`topological_sort` raises `ValueError` on a cyclic graph \
         (__init__.py:1252-1253)"
    );

    // A DAG: 0 -> 1 -> 3, 0 -> 2 -> 3, 4 isolated.
    let mut d = AdjList::with_vertices(5);
    for &(s, t) in &[(0, 1), (0, 2), (1, 3), (2, 3)] {
        d.add_edge(v(s), v(t)).expect("add");
    }
    assert!(is_dag(&d));
    let order = topological_sort(&d).expect("it is a DAG");
    assert_eq!(order.len(), 5);

    let pos: Vec<usize> = {
        let mut p = vec![usize::MAX; 5];
        for (i, u) in order.iter().enumerate() {
            p[u.index()] = i;
        }
        p
    };
    for &(s, t) in &[(0usize, 1usize), (0, 2), (1, 3), (2, 3)] {
        assert!(
            pos[s] < pos[t],
            "edge {s} -> {t} points backwards in the order"
        );
    }
    // 0 and 4 are both sources, so which of them leads is boost's DFS order
    // and not a documented guarantee; that 0 precedes both its targets is.
    assert!(pos[0] < pos[1] && pos[0] < pos[2]);
}

/// `is_bipartite` two-colours across the **undirected** incidence, and a
/// self-loop or an odd cycle refuses.
///
/// `graph_bipartite.cc` runs on `never_directed` (the `run_action` tag), i.e.
/// on the undirected adaptor, so direction is irrelevant to the answer. The
/// port takes a view instead, and the two disagree only because a directed
/// view cannot see an in-edge at all.
#[test]
fn bipartiteness_is_decided_on_the_undirected_incidence() {
    // A 4-cycle, directed all the way round: bipartite undirected.
    let mut c4 = AdjList::with_vertices(4);
    for &(s, t) in &[(0, 1), (1, 2), (2, 3), (3, 0)] {
        c4.add_edge(v(s), v(t)).expect("add");
    }
    let part = is_bipartite((&c4).undirect()).expect("a 4-cycle is bipartite");
    assert_eq!(part.len(), 4);
    for &(s, t) in &[(0usize, 1usize), (1, 2), (2, 3), (3, 0)] {
        assert_ne!(part[s], part[t], "edge {s} -> {t} is monochromatic");
    }

    // A triangle is not.
    let mut c3 = AdjList::with_vertices(3);
    for &(s, t) in &[(0, 1), (1, 2), (2, 0)] {
        c3.add_edge(v(s), v(t)).expect("add");
    }
    assert!(is_bipartite((&c3).undirect()).is_none());

    // A self-loop is an odd cycle of length one.
    let mut l = AdjList::with_vertices(2);
    l.add_edge(v(0), v(1)).expect("add");
    l.add_edge(v(1), v(1)).expect("add");
    assert!(
        is_bipartite((&l).undirect()).is_none(),
        "a self-loop makes a graph non-bipartite"
    );
}

// ---------------------------------------------------------------------------
// 4. Removals do not disturb the kernels
// ---------------------------------------------------------------------------

/// The kernels index by [`VertexId`] and are sized from the *bound*, so a
/// graph with holes in its edge index space still answers correctly.
///
/// This is the composition of `u34_diff_adjacency`'s recycling facts with the
/// algorithms: after a removal the live edge ids are sparse, and a kernel that
/// walked `0..num_edges` rather than the incidence would silently drop the
/// edges above the count.
#[test]
fn distances_survive_edge_removal_and_index_recycling() {
    let mut g = fixture();
    // Remove `1 -> 2` (e1), then add `1 -> 5`, which recycles index 1.
    g.remove_edge(EdgeId::from_index(1)).expect("e1 is live");
    assert_eq!((g.num_edges(), g.edge_bound().len()), (5, 6));
    let re = g.add_edge(v(1), v(5)).expect("add").id();
    assert_eq!(re.index(), 1, "LIFO recycling hands index 1 back");
    // Live edges are now `0->1 (e0)`, `1->5 (e1)`, `2->1 (e2)`, `2->3 (e3)`,
    // `4->2 (e4)`, `3->3 (e5)`.

    assert_eq!(
        dist_of(&g, 0),
        vec![0, 1, INF, INF, INF, 2],
        "0 -> 1 -> 5; 2, 3 and 4 are all upstream now"
    );
    assert_eq!(
        dist_of(&g, 4),
        vec![INF, 2, 1, 2, 0, 3],
        "4 -> 2 at 1, then 1 and 3 at 2, then 5 at 3"
    );
    assert_eq!(
        dist_of((&g).undirect(), 0),
        vec![0, 1, 2, 3, 3, 2],
        "undirected, every vertex is reachable"
    );

    // The self-loop is the only cycle left, so dropping it makes a DAG.
    assert!(!is_dag(&g));
    g.remove_edge(EdgeId::from_index(5))
        .expect("the self-loop is live");
    assert!(is_dag(&g), "removing the self-loop makes it acyclic");
    let order = topological_sort(&g).expect("it is a DAG");
    let mut pos = [usize::MAX; N];
    for (i, u) in order.iter().enumerate() {
        pos[u.index()] = i;
    }
    for e in [(0usize, 1usize), (1, 5), (2, 1), (2, 3), (4, 2)] {
        assert!(pos[e.0] < pos[e.1], "edge {e:?} points backwards");
    }
}

// ---------------------------------------------------------------------------
// 5. k-core
// ---------------------------------------------------------------------------
//
// `kcore_decomposition` (`topology/graph_kcore.hh:25-80`) reads `degree(v, g)`
// and `all_neighbors_range(v, g)`, which for `adj_list` are both the whole
// incidence block. The Python docstring turns that into two promises
// (`topology/__init__.py:1832-1836`):
//
//   "For directed graphs, the degree is assumed to be the total (in + out)
//    degree."
//   "The algorithm accepts graphs with parallel edges and self loops, in which
//    case these edges contribute to the degree in the usual fashion."
//
// Both of those are testable, and both are places a plausible implementation
// would differ.

fn cores_of<G: GraphRef + VertexList>(g: G) -> (Vec<i64>, usize) {
    let mut c: DenseProp<i64, VertexTag> = DenseProp::new(g.graph_id());
    let k = kcore_decomposition(g, &mut c);
    (c.as_slice().to_vec(), k)
}

/// A triangle with a pendant, peeled by hand.
///
/// Edges `{0,1}, {1,2}, {2,0}, {0,3}`; degrees `3, 2, 2, 1`. The bins start as
/// `bins[1] = [3]`, `bins[2] = [1, 2]`, `bins[3] = [0]`, and the drain is:
///
/// ```text
///   k=1: pop 3 -> core 1; its only neighbour 0 has deg 3 > 1, so 0 drops
///        into bins[2] with deg 2
///   k=2: pop 0, then 2, then 1 -- each has remaining degree 2, so nothing
///        is demoted further
/// ```
///
/// giving `[2, 2, 2, 1]` and a degeneracy of 2.
#[test]
fn the_kcore_of_a_triangle_with_a_pendant_is_hand_computable() {
    let mut g = AdjList::with_vertices(4);
    for &(s, t) in &[(0, 1), (1, 2), (2, 0), (0, 3)] {
        g.add_edge(v(s), v(t)).expect("add");
    }
    assert_eq!(cores_of((&g).undirect()), (vec![2, 2, 2, 1], 2));
    // `kcore_decomposition` reads `degree` and `all_edges`, and
    // `Und<G>::All == G::All`, so the directed and undirected views cannot
    // disagree -- which is the C++'s single `all_graph_views` body.
    assert_eq!(cores_of(&g), cores_of((&g).undirect()));
    assert_eq!(cores_of((&g).reverse()), cores_of(&g));
}

/// A self-loop and a parallel edge raise a vertex's core number, because they
/// raise its degree.
///
/// This is the docstring's "in the usual fashion" made concrete, and it is
/// where an implementation that deduplicated the adjacency --- or that skipped
/// `u == v` --- would disagree.
///
/// * `{0,1}` plus a self-loop on 0: `degree(0) = 3` (the loop occupies both
///   halves of the block), `degree(1) = 1`. Peeling 1 at `k = 1` drops 0 to
///   `deg 2`, and 0 is then popped at `k = 2` --- so **a vertex with one real
///   neighbour has core number 2**.
/// * three parallel `{0,1}` edges: both degrees are 3, neither exceeds the
///   other, and both come out at core 3.
#[test]
fn self_loops_and_parallel_edges_contribute_to_the_core_number() {
    let mut loopy = AdjList::with_vertices(2);
    loopy.add_edge(v(0), v(1)).expect("add");
    loopy.add_edge(v(0), v(0)).expect("add");
    assert_eq!(
        loopy.degree(v(0)),
        3,
        "the self-loop occupies both halves of v0's block"
    );
    assert_eq!(
        cores_of((&loopy).undirect()),
        (vec![2, 1], 2),
        "v0 reaches core 2 with a single neighbour"
    );

    let mut par = AdjList::with_vertices(2);
    for _ in 0..3 {
        par.add_edge(v(0), v(1)).expect("add");
    }
    assert_eq!((par.degree(v(0)), par.degree(v(1))), (3, 3));
    assert_eq!(cores_of((&par).undirect()), (vec![3, 3], 3));

    // The simple `{0,1}` edge alone is a 1-core, which is what makes the two
    // above non-vacuous.
    let mut simple = AdjList::with_vertices(2);
    simple.add_edge(v(0), v(1)).expect("add");
    assert_eq!(cores_of((&simple).undirect()), (vec![1, 1], 1));
}

/// A directed graph's core number uses the **total** degree.
///
/// `0 -> 1`, `1 -> 0`, `1 -> 2`: total degrees 2, 3, 1. Peeling 2 at `k = 1`
/// drops 1 to `deg 2`; then 1 and 0 both come out at `k = 2`, giving
/// `[2, 2, 1]`. An implementation that used `out_degree` would peel 2 first
/// and then 0 (out-degree 1) and get a different answer.
#[test]
fn a_directed_graphs_core_number_is_the_total_degree() {
    let mut g = AdjList::with_vertices(3);
    for &(s, t) in &[(0, 1), (1, 0), (1, 2)] {
        g.add_edge(v(s), v(t)).expect("add");
    }
    assert_eq!(
        (g.degree(v(0)), g.degree(v(1)), g.degree(v(2))),
        (2, 3, 1),
        "in + out"
    );
    assert_eq!(cores_of(&g), (vec![2, 2, 1], 2));
    assert_eq!(
        (g.out_degree(v(0)), g.out_degree(v(1)), g.out_degree(v(2))),
        (1, 2, 0),
        "the out-degrees would give a different peel order"
    );

    // A complete graph on 4 vertices is 3-degenerate however it is oriented.
    let mut k4 = AdjList::with_vertices(4);
    for &(s, t) in &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        k4.add_edge(v(s), v(t)).expect("add");
    }
    assert_eq!(cores_of((&k4).undirect()), (vec![3, 3, 3, 3], 3));
    assert_eq!(cores_of(&k4), cores_of((&k4).undirect()));
}

/// The defining property, checked on a random multigraph.
///
/// "The k-core is a maximal set of vertices such that its induced subgraph
/// only contains vertices with degree larger than or equal to k"
/// (`topology/__init__.py:1829-1830`). For every `k`, the set
/// `S_k = {v : core[v] >= k}` must therefore have minimum induced degree at
/// least `k` --- counting multiplicity and self-loops doubly, since that is
/// the degree the algorithm reads.
///
/// Maximality is the other half, and it follows from `S_{k+1}` failing the
/// same test: if some vertex of `S_k \ S_{k+1}` could stay, it would have been
/// peeled later.
#[test]
fn every_k_core_has_minimum_induced_degree_k() {
    // SplitMix64, so a failure reproduces from the test name.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let n = 40usize;
    let mut g = AdjList::with_vertices(n);
    let mut edges = Vec::new();
    for _ in 0..140 {
        let a = (next() % n as u64) as usize;
        let b = if next() % 7 == 0 {
            a
        } else {
            (next() % n as u64) as usize
        };
        g.add_edge(v(a), v(b)).expect("add");
        edges.push((a, b));
    }

    let (core, degeneracy) = cores_of((&g).undirect());
    assert!(core.iter().all(|&c| c >= 0), "every vertex is labelled");
    assert_eq!(degeneracy, *core.iter().max().expect("non-empty") as usize);

    for k in 1..=degeneracy as i64 {
        let inside: Vec<bool> = core.iter().map(|&c| c >= k).collect();
        for u in 0..n {
            if !inside[u] {
                continue;
            }
            // Induced degree, counting a self-loop twice and a parallel edge
            // with its multiplicity, which is `degree(v, g)` restricted to S.
            let mut d = 0usize;
            for &(a, b) in &edges {
                if a == u && inside[b] {
                    d += 1;
                }
                if b == u && inside[a] {
                    d += 1;
                }
            }
            assert!(
                d >= k as usize,
                "vertex {u} is in the {k}-core with induced degree {d}"
            );
        }
        // Maximality: the (k+1)-core is a proper subset whenever some vertex
        // has core exactly k.
        if core.contains(&k) {
            let bigger = core.iter().filter(|&&c| c > k).count();
            let this = core.iter().filter(|&&c| c >= k).count();
            assert!(bigger < this);
        }
    }
}
