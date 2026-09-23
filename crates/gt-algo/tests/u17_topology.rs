//! U17 — topology, from outside the crate.
//!
//! The acceptance list for this unit is two statements:
//!
//! 1. `topological_sort` returns `None` **exactly** when `is_dag` is false,
//!    and otherwise a valid order;
//! 2. `count_triangles` counts each triangle **once**, on a directed view and
//!    on an undirected one, checked against a brute-force `O(V³)` reference
//!    over 200 small graphs.
//!
//! Both are randomised below, and both references are written from the
//! definition rather than from the kernel: `brute_triangles` is a triple loop
//! over `a < b < c`, `brute_is_dag` is the transitive closure, `brute_bipartite`
//! enumerates all `2ⁿ` colourings, and `brute_reciprocity` is the `min(m_uv,
//! m_vu)` sum of `topology/graph_reciprocity.cc:59-60` spelled out over the
//! full ordered-pair matrix. A reference that shares an idea with the kernel
//! checks nothing.
//!
//! The generated graphs carry **self-loops and parallel edges** on purpose.
//! They are where `graph_clustering.hh:64`'s `mark[u] = w` and `:65`'s
//! `k += w` stop agreeing, and where a per-edge count and a per-incidence
//! count separate.

use gt_algo::topology::{
    count_triangles, global_clustering, is_bipartite, is_dag, reciprocity, topological_sort,
};
use gt_core::adj::AdjList;
use gt_core::graph::{EdgeList, GraphBase};
use gt_core::ids::VertexId;
use gt_core::view::{Filtered, Reverse, Undirect};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

fn graph(n: usize, edges: &[(usize, usize)]) -> AdjList {
    let mut g = AdjList::with_vertices(n);
    for &(s, t) in edges {
        g.add_edge(v(s), v(t)).expect("endpoints are in range");
    }
    g
}

/// SplitMix64: the fixtures must not depend on a dependency version.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// `count` graphs, small enough for an `O(V³)` reference and an exhaustive
/// `2ⁿ` colouring, with self-loops and parallel edges left in.
fn corpus(count: usize) -> Vec<(usize, Vec<(usize, usize)>)> {
    let mut rng = Rng(0x5eed_1234_abcd_ef01);
    (0..count)
        .map(|_| {
            let n = 4 + rng.below(8);
            let m = rng.below(3 * n);
            let edges = (0..m).map(|_| (rng.below(n), rng.below(n))).collect();
            (n, edges)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Brute-force references
// ---------------------------------------------------------------------------

/// The simple undirected adjacency: direction dropped, parallel edges
/// collapsed, self-loops removed.
fn simple(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<bool>> {
    let mut a = vec![vec![false; n]; n];
    for &(s, t) in edges {
        if s != t {
            a[s][t] = true;
            a[t][s] = true;
        }
    }
    a
}

/// `O(V³)`: every unordered triple, tested directly.
fn brute_triangles(a: &[Vec<bool>]) -> u64 {
    let n = a.len();
    let mut t = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                if a[i][j] && a[j][k] && a[i][k] {
                    t += 1;
                }
            }
        }
    }
    t
}

/// `Σ_v C(d_v, 2)`, the connected triples of `clustering/__init__.py:190-193`.
fn brute_triples(a: &[Vec<bool>]) -> u64 {
    a.iter()
        .map(|row| {
            let d = row.iter().filter(|&&x| x).count() as u64;
            d * d.saturating_sub(1) / 2
        })
        .sum()
}

/// Acyclicity by transitive closure: a DAG is a digraph in which no vertex
/// reaches itself. Shares nothing with a DFS.
fn brute_is_dag(n: usize, edges: &[(usize, usize)]) -> bool {
    let mut r = vec![vec![false; n]; n];
    for &(s, t) in edges {
        r[s][t] = true;
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if r[i][k] && r[k][j] {
                    r[i][j] = true;
                }
            }
        }
    }
    (0..n).all(|i| !r[i][i])
}

/// Bipartiteness by exhaustive two-colouring, over the underlying undirected
/// graph *including* self-loops (a self-loop is monochromatic by definition).
fn brute_bipartite(n: usize, edges: &[(usize, usize)]) -> bool {
    assert!(n < 24, "exhaustive colouring is 2ⁿ");
    (0u32..(1 << n)).any(|mask| {
        edges
            .iter()
            .all(|&(s, t)| (mask >> s) & 1 != (mask >> t) & 1)
    })
}

/// `Lbd / L` with `Lbd = Σ_{u≠v} min(m_uv, m_vu)` and `L` the non-self-loop
/// edge count: `topology/graph_reciprocity.cc:44-61`, over the full matrix.
fn brute_reciprocity(n: usize, edges: &[(usize, usize)]) -> f64 {
    let mut m = vec![vec![0u64; n]; n];
    for &(s, t) in edges {
        if s != t {
            m[s][t] += 1;
        }
    }
    let mut lbd = 0u64;
    let mut l = 0u64;
    for (i, row) in m.iter().enumerate() {
        for (j, &forward) in row.iter().enumerate() {
            if i != j {
                lbd += forward.min(m[j][i]);
                l += forward;
            }
        }
    }
    lbd as f64 / l as f64
}

// ---------------------------------------------------------------------------
// Acceptance 1 — the ordering kernels agree, and the order is an order
// ---------------------------------------------------------------------------

/// `topological_sort` is `Some` exactly when `is_dag`, and the order it
/// returns places every arc's source before its target.
///
/// The direction matters: the C++ writes DFS *finish* order through a
/// `back_inserter` (`topology/graph_topological_sort.cc:34`), which is the
/// reverse of what the caller wants, and the Python wrapper repairs it with
/// `[::-1]`. A port that forgets the reversal still returns a permutation of
/// the vertices and still agrees with `is_DAG`; only this assertion sees it.
#[test]
fn topological_sort_is_some_exactly_when_is_dag_and_is_a_valid_order() {
    let mut dags = 0;
    for (n, edges) in corpus(200) {
        let g = graph(n, &edges);
        let expected = brute_is_dag(n, &edges);

        assert_eq!(is_dag(&g), expected, "is_dag: n={n} edges={edges:?}");
        let order = topological_sort(&g);
        assert_eq!(
            order.is_some(),
            is_dag(&g),
            "topological_sort and is_dag disagree: n={n} edges={edges:?}"
        );

        if let Some(order) = order {
            dags += 1;
            assert_eq!(order.len(), n, "every vertex appears");
            let mut pos = vec![usize::MAX; n];
            for (p, u) in order.iter().enumerate() {
                assert_eq!(pos[u.index()], usize::MAX, "no vertex appears twice");
                pos[u.index()] = p;
            }
            for &(s, t) in &edges {
                assert!(
                    pos[s] < pos[t],
                    "arc {s}->{t} is inverted in {order:?} (n={n})"
                );
            }
        }
    }
    assert!(dags > 10, "the corpus must contain DAGs; found {dags}");
}

#[test]
fn a_path_sorts_and_a_cycle_does_not() {
    let path = graph(4, &[(0, 1), (1, 2), (2, 3)]);
    assert!(is_dag(&path));
    assert_eq!(
        topological_sort(&path).unwrap(),
        vec![v(0), v(1), v(2), v(3)]
    );

    let cycle = graph(3, &[(0, 1), (1, 2), (2, 0)]);
    assert!(!is_dag(&cycle));
    assert!(topological_sort(&cycle).is_none());

    // A self-loop is a back edge to a grey vertex, exactly as in
    // `boost::topological_sort`.
    let loopy = graph(2, &[(0, 1), (1, 1)]);
    assert!(!is_dag(&loopy));
    assert!(topological_sort(&loopy).is_none());

    // No edges: every graph is trivially sorted, including the empty one.
    assert!(is_dag(&graph(3, &[])));
    assert_eq!(topological_sort(&graph(0, &[])).unwrap(), vec![]);
}

/// Every incidence of an undirected view is its own back edge, so the only
/// acyclic undirected view is the edgeless one. The point is that the answer
/// is *derived*, not special-cased: `Und<G>::Out` is `G::All`.
#[test]
fn an_undirected_view_is_a_dag_only_when_it_has_no_edges() {
    let g = graph(3, &[(0, 1), (1, 2)]);
    assert!(is_dag(&g));
    assert!(!is_dag((&g).undirect()));

    let empty = graph(3, &[]);
    assert!(is_dag((&empty).undirect()));
}

/// Iterative, not recursive. `boost::depth_first_search` recurses, so this
/// shape overflows the stack in graph-tool.
#[test]
fn a_long_path_does_not_overflow_the_stack() {
    let n = 400_000;
    let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
    let g = graph(n, &edges);
    let order = topological_sort(&g).expect("a path is a DAG");
    assert_eq!(order.len(), n);
    assert_eq!(order[0], v(0));
    assert_eq!(order[n - 1], v(n - 1));
}

// ---------------------------------------------------------------------------
// Acceptance 2 — each triangle counted once, on every view
// ---------------------------------------------------------------------------

/// 200 small graphs against the `O(V³)` reference, on the directed view, the
/// undirected view and the reversed one.
///
/// The three must agree: a triangle is a property of the incidence structure,
/// and none of the three views changes it. Degree summation would report
/// `3 ×`, `6 ×` or `2 ×` the truth depending on which view it was asked of,
/// which is exactly why `get_triangles` needs a `/ 2` at
/// `graph_clustering.hh:92` and another `/ 3` at `:132`.
#[test]
fn count_triangles_counts_each_triangle_once_on_every_view() {
    let mut seen_triangles = 0u64;
    for (n, edges) in corpus(200) {
        let g = graph(n, &edges);
        let expected = brute_triangles(&simple(n, &edges));
        seen_triangles += expected;

        assert_eq!(
            count_triangles(&g),
            expected,
            "directed: n={n} edges={edges:?}"
        );
        assert_eq!(
            count_triangles((&g).undirect()),
            expected,
            "undirected: n={n} edges={edges:?}"
        );
        assert_eq!(
            count_triangles((&g).reverse()),
            expected,
            "reversed: n={n} edges={edges:?}"
        );
    }
    assert!(
        seen_triangles > 100,
        "the corpus must contain triangles; found {seen_triangles}"
    );
}

/// Parallel edges and self-loops multiply a *degree* sum and must not
/// multiply a triangle count.
#[test]
fn parallel_edges_and_self_loops_do_not_manufacture_triangles() {
    let simple_k3 = graph(3, &[(0, 1), (1, 2), (2, 0)]);
    assert_eq!(count_triangles(&simple_k3), 1);

    // Each edge tripled and every vertex looped: the same one triangle.
    let mut edges = Vec::new();
    for &(s, t) in &[(0usize, 1usize), (1, 2), (2, 0)] {
        edges.extend([(s, t), (s, t), (t, s)]);
    }
    edges.extend([(0, 0), (1, 1), (2, 2)]);
    let fat = graph(3, &edges);
    assert_eq!(count_triangles(&fat), 1);
    assert_eq!(count_triangles((&fat).undirect()), 1);

    // A triangle of self-loops is not a triangle.
    assert_eq!(count_triangles(&graph(3, &[(0, 0), (1, 1), (2, 2)])), 0);
}

/// `K_n` has `C(n, 3)` triangles and clustering `1`. Every wrong constant in
/// the witness ordering is visible here: `K5` has 10 triangles, and the
/// six-fold ordered overcount would be 60.
#[test]
fn complete_graphs_have_the_binomial_count_and_unit_clustering() {
    for n in 3..=7usize {
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();
        let g = graph(n, &edges);
        let expected = (n * (n - 1) * (n - 2) / 6) as u64;
        assert_eq!(count_triangles(&g), expected, "K{n}");
        assert!(
            (global_clustering(&g) - 1.0).abs() < 1e-12,
            "K{n} is fully transitive"
        );
    }
}

/// A five-cycle: no triangles, five triples, zero transitivity. A ten-cycle
/// and the Petersen graph pin the same thing at girth 5 with 15 edges.
#[test]
fn triangle_free_graphs_have_zero_clustering() {
    let c5 = graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
    assert_eq!(count_triangles(&c5), 0);
    assert_eq!(global_clustering(&c5), 0.0);

    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..5 {
        edges.push((i, (i + 1) % 5)); // outer pentagon
        edges.push((5 + i, 5 + (i + 2) % 5)); // inner pentagram
        edges.push((i, 5 + i)); // spokes
    }
    let petersen = graph(10, &edges);
    assert_eq!(petersen.num_edges(), 15);
    assert_eq!(count_triangles(&petersen), 0);
    assert_eq!(global_clustering(&petersen), 0.0);
}

// ---------------------------------------------------------------------------
// global_clustering
// ---------------------------------------------------------------------------

/// `c = 3 · triangles / triples` over the whole corpus, and the same value on
/// the undirected view — which is the number graph-tool reports, because
/// `clustering/__init__.py:226-227` undirects before dispatching.
#[test]
fn global_clustering_is_three_triangles_over_triples_on_every_view() {
    for (n, edges) in corpus(200) {
        let g = graph(n, &edges);
        let a = simple(n, &edges);
        let (tri, triples) = (brute_triangles(&a), brute_triples(&a));
        let got = global_clustering(&g);

        if triples == 0 {
            assert!(got.is_nan(), "0/0 is not a coefficient: n={n}");
            continue;
        }
        let expected = 3.0 * tri as f64 / triples as f64;
        assert!(
            (got - expected).abs() < 1e-12,
            "n={n} edges={edges:?}: {got} != {expected}"
        );
        assert_eq!(
            global_clustering((&g).undirect()).to_bits(),
            got.to_bits(),
            "the undirected view is the same graph"
        );
        // The two kernels share a definition, so they must share a number.
        assert!(
            (got - 3.0 * count_triangles(&g) as f64 / triples as f64).abs() < 1e-12
        );
    }
}

/// `double(triangles) / n` at `graph_clustering.hh:116` is `0.0 / 0` for a
/// graph with no connected triple. NaN, not a plausible zero.
#[test]
fn a_graph_without_triples_has_no_coefficient() {
    assert!(global_clustering(&graph(0, &[])).is_nan());
    assert!(global_clustering(&graph(5, &[])).is_nan());
    assert!(global_clustering(&graph(2, &[(0, 1)])).is_nan());
    // One triple, no triangle: defined, and zero.
    assert_eq!(global_clustering(&graph(3, &[(0, 1), (1, 2)])), 0.0);
}

// ---------------------------------------------------------------------------
// is_bipartite
// ---------------------------------------------------------------------------

/// Against exhaustive `2ⁿ` colouring, with the returned map checked edge by
/// edge rather than compared to a chosen one.
#[test]
fn is_bipartite_agrees_with_exhaustive_colouring_and_returns_a_proper_map() {
    let (mut yes, mut no) = (0, 0);
    for (n, edges) in corpus(200) {
        let g = graph(n, &edges);
        let expected = brute_bipartite(n, &edges);
        let got = is_bipartite(&g);
        assert_eq!(got.is_some(), expected, "n={n} edges={edges:?}");

        match got {
            Some(part) => {
                yes += 1;
                assert_eq!(part.len(), n);
                for &(s, t) in &edges {
                    assert_ne!(part[s], part[t], "edge {s}-{t} is monochromatic");
                }
                assert!(part.iter().all(|&c| c <= 1), "the map is a two-colouring");
                // Direction cannot change bipartiteness: the C++ dispatches
                // under `{.tr=never_directed}` (`graph_bipartite.cc:84`).
                assert_eq!(is_bipartite((&g).undirect()), Some(part));
            }
            None => {
                no += 1;
                assert_eq!(is_bipartite((&g).undirect()), None);
            }
        }
    }
    assert!(yes > 10 && no > 10, "corpus is one-sided: {yes}/{no}");
}

/// `gt.is_bipartite(gt.lattice([10, 10]))` is `True`
/// (`topology/__init__.py`, the `is_bipartite` doctest), and the colouring is
/// the parity of `i + j`.
#[test]
fn a_square_lattice_is_bipartite_and_coloured_by_parity() {
    let side = 10usize;
    let mut edges = Vec::new();
    for i in 0..side {
        for j in 0..side {
            let u = i * side + j;
            if i + 1 < side {
                edges.push((u, u + side));
            }
            if j + 1 < side {
                edges.push((u, u + 1));
            }
        }
    }
    let g = graph(side * side, &edges);
    let part = is_bipartite(&g).expect("a lattice is bipartite");

    // Root convention: `boost::is_bipartite` paints the start vertex white and
    // `graph_bipartite.cc:51` stores `part[v] == white`, i.e. 1.
    assert_eq!(part[0], 1);
    for i in 0..side {
        for j in 0..side {
            assert_eq!(part[i * side + j], u8::from((i + j) % 2 == 0));
        }
    }

    // An odd cycle is not.
    assert!(is_bipartite(&graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])).is_none());
    // A self-loop is a monochromatic edge.
    assert!(is_bipartite(&graph(2, &[(0, 0)])).is_none());
    // Isolated vertices are roots of their own component, so all white.
    assert_eq!(is_bipartite(&graph(3, &[])).unwrap(), vec![1, 1, 1]);
}

// ---------------------------------------------------------------------------
// reciprocity
// ---------------------------------------------------------------------------

/// Against the `min(m_uv, m_vu)` matrix sum, over the whole corpus.
#[test]
fn reciprocity_is_the_min_multiplicity_sum() {
    for (n, edges) in corpus(200) {
        let g = graph(n, &edges);
        let expected = brute_reciprocity(n, &edges);
        let got = reciprocity(&g);
        if expected.is_nan() {
            assert!(got.is_nan(), "n={n}: no non-self-loop edge");
        } else {
            assert!(
                (got - expected).abs() < 1e-12,
                "n={n} edges={edges:?}: {got} != {expected}"
            );
        }
    }
}

/// The `edge_reciprocity` doctest (`topology/__init__.py`): one arc is `0.0`,
/// its reverse makes it `1.0`.
#[test]
fn the_reciprocity_doctest() {
    let mut g = AdjList::with_vertices(2);
    g.add_edge(v(0), v(1)).unwrap();
    assert_eq!(reciprocity(&g), 0.0);
    g.add_edge(v(1), v(0)).unwrap();
    assert_eq!(reciprocity(&g), 1.0);

    // Self-loops are outside both counts: `self_loops=False` is the Python
    // default and `:46-47` is the `continue`.
    g.add_edge(v(0), v(0)).unwrap();
    assert_eq!(reciprocity(&g), 1.0);
    assert!(reciprocity(&graph(2, &[(0, 0), (1, 1)])).is_nan());
}

/// `Lbd += min(wr, w_u)` (`graph_reciprocity.cc:59`), not
/// `Lbd += w_u * [wr > 0]`: two arcs one way and one back is `2/3`, not `1`.
#[test]
fn a_parallel_arc_is_reciprocated_only_as_far_as_its_reverse_reaches() {
    let g = graph(2, &[(0, 1), (0, 1), (1, 0)]);
    assert!((reciprocity(&g) - 2.0 / 3.0).abs() < 1e-12);

    let g = graph(2, &[(0, 1), (0, 1), (1, 0), (1, 0)]);
    assert_eq!(reciprocity(&g), 1.0);

    let g = graph(3, &[(0, 1), (1, 0), (1, 2)]);
    assert!((reciprocity(&g) - 2.0 / 3.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Filtered views
// ---------------------------------------------------------------------------

/// A filtered view is a graph, and the counts are the counts of *that* graph.
///
/// `edges()` on a filtered view drops an edge whose endpoint is masked out
/// (`view/filtered.rs:116-118`), so masking a corner of `K4` must remove the
/// three triangles through it — and `num_edges` must agree, which is the
/// identity `filt_graph` gives up at `graph_filtered.hh:301-318`.
#[test]
fn the_counts_follow_a_filtered_view() {
    let k4 = graph(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    assert_eq!(count_triangles(&k4), 4);

    let vmask = [1u8, 1, 1, 0];
    let emask = [1u8; 6];
    let f = Filtered::masked(&k4, &vmask, &emask).expect("masks are graph-sized");
    assert_eq!(f.num_vertices(), 3);
    assert_eq!(f.num_edges(), 3);
    assert_eq!(f.edges().count(), f.num_edges());
    assert_eq!(count_triangles(f), 1);
    assert!((global_clustering(f) - 1.0).abs() < 1e-12);

    // Masking an edge instead: K4 minus one edge keeps two triangles.
    let vmask = [1u8; 4];
    let emask = [0u8, 1, 1, 1, 1, 1];
    let f = Filtered::masked(&k4, &vmask, &emask).expect("masks are graph-sized");
    assert_eq!(f.num_edges(), 5);
    assert_eq!(count_triangles(f), 2);

    // The vertex-filtered view is still a DAG question with an answer.
    let dag = graph(4, &[(0, 1), (1, 2), (2, 3), (3, 1)]);
    assert!(!is_dag(&dag));
    let vmask = [1u8, 1, 1, 0];
    let emask = [1u8; 4];
    let f = Filtered::masked(&dag, &vmask, &emask).expect("masks are graph-sized");
    assert!(is_dag(f));
    assert_eq!(topological_sort(f).unwrap(), vec![v(0), v(1), v(2)]);
}
