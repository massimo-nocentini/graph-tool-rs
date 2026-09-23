//! U16 -- centrality, checked through the public surface.
//!
//! The unit's acceptance criteria, one test each:
//!
//! * PageRank on a fixed 1 000-vertex graph is **bit-identical** at 1, 4 and
//!   16 workers -- `pagerank_is_bit_identical_across_thread_counts`;
//! * and equal to a **dense** power iteration to 1e-12 --
//!   `pagerank_matches_a_dense_power_iteration`;
//! * `betweenness` matches an independent Brandes on **100 small graphs**,
//!   directed and undirected -- `betweenness_matches_a_reference_brandes`.
//!
//! On top of those, one test per defect the module corrects, because each is
//! a wrong *number* rather than a crash and none of them is visible in a run
//! that converges:
//!
//! * the inverted odd-iteration writeback (`graph_pagerank.hh:90-98`,
//!   `graph_eigenvector.hh:94-95`) --
//!   `an_odd_number_of_sweeps_leaves_the_latest_iterate_in_the_callers_map`
//!   and its eigenvector twin. Both pin the arithmetic of a single sweep by
//!   hand, so "returns the previous iterate" is a failure and not a rounding
//!   difference;
//! * division by a zero weighted out-degree (`graph_pagerank.hh:77`) --
//!   `a_zero_weight_predecessor_does_not_poison_the_vector`;
//! * division by a zero norm (`graph_eigenvector.hh:83`) --
//!   `an_edgeless_graph_has_a_zero_eigenvalue_and_no_nans`;
//! * the Brandes distance map that is never reset between sources
//!   (`betweenness_centrality.hpp:341-349`) --
//!   `betweenness_on_a_path_is_the_closed_form`, whose expected values come
//!   from a formula rather than from another implementation, and
//!   `betweenness_is_unaffected_by_an_edge_back_into_an_earlier_source`.

use rayon::ThreadPoolBuilder;

use gt_algo::centrality::{PowerIteration, betweenness, eigenvector, pagerank, pagerank_with};
use gt_core::adj::AdjList;
use gt_core::ids::{EdgeTag, VertexId, VertexTag};
use gt_core::par::Plan;
use gt_core::prop::DenseProp;
use gt_core::prop::dense::{Constant, Unity};
use gt_core::view::Undirect;

/// The worker counts every threaded test is run at. One worker is the
/// control: it is the only configuration in which the OpenMP original is
/// deterministic too.
const THREADS: [usize; 3] = [1, 4, 16];

/// Run `f` inside a private rayon pool of exactly `n` workers. A private
/// pool, not `RAYON_NUM_THREADS`: the global pool is built once per process
/// and these tests share a process.
fn at_threads<R: Send>(n: usize, f: impl Fn() -> R + Sync + Send) -> R {
    ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("could not build a rayon pool")
        .install(f)
}

/// SplitMix64: the graphs under test must not depend on a dependency version.
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

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

/// A graph from an explicit edge list. Edge `k` of the list is edge index
/// `k` of the graph, which is what lets an edge-betweenness comparison be
/// written by position.
fn build(n: usize, edges: &[(usize, usize)]) -> AdjList {
    let mut g = AdjList::with_vertices(n);
    for &(s, t) in edges {
        g.add_edge(v(s), v(t)).expect("add_edge");
    }
    g
}

/// The fixed 1 000-vertex graph the acceptance criteria name. A ring, so the
/// graph is strongly connected, plus 4 000 chords, plus 50 deliberate sinks
/// whose out-edges are removed -- `p_sink` is the term `det_reduce` exists
/// for and a graph without sinks never exercises it.
fn thousand() -> AdjList {
    const N: usize = 1_000;
    let mut r = Rng(0x243f_6a88_85a3_08d3);
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..N {
        edges.push((i, (i + 1) % N));
    }
    for _ in 0..4_000 {
        edges.push((r.below(N), r.below(N)));
    }
    // 50 sinks: drop every out-edge of vertices 17, 37, 57, ...
    let sink = |i: usize| i >= 17 && (i - 17).is_multiple_of(20) && i < 17 + 20 * 50;
    edges.retain(|&(s, _)| !sink(s));
    build(N, &edges)
}

/// Every edge of `g`, indexed by its edge index.
fn edge_table(g: &AdjList) -> Vec<(usize, usize)> {
    let mut t = vec![(usize::MAX, usize::MAX); g.edge_bound().len()];
    for e in g.edges() {
        t[e.id().index()] = (e.source().index(), e.target().index());
    }
    t
}

fn params(n: usize, epsilon: f64, max_iter: usize) -> PowerIteration {
    PowerIteration {
        epsilon,
        max_iter,
        plan: Plan::new(n, 64),
    }
}

// ===========================================================================
// 1. PageRank: the dense reference
// ===========================================================================

/// A dense power iteration, written from `graph_pagerank.hh:56-88` and
/// sharing no code with the kernel: an explicit `n x n` transition matrix, a
/// full row-major inner product per vertex, and a fixed sweep count well past
/// convergence (`0.85^400` is below every representable double).
///
/// The zero-out-degree guard here is the one the C++ omits: a vertex with no
/// out-weight contributes no column, because its rank is already
/// redistributed through `p_sink`.
fn dense_pagerank(
    n: usize,
    edges: &[(usize, usize, f64)],
    pers: &[f64],
    d: f64,
    sweeps: usize,
) -> Vec<f64> {
    let mut deg = vec![0.0f64; n];
    for &(s, _, w) in edges {
        deg[s] += w;
    }
    let mut m = vec![0.0f64; n * n];
    for &(s, t, w) in edges {
        if deg[s] != 0.0 {
            m[t * n + s] += w / deg[s];
        }
    }
    let mut r = pers.to_vec();
    for _ in 0..sweeps {
        let mut p_sink = 0.0;
        for i in 0..n {
            if deg[i] == 0.0 {
                p_sink += r[i];
            }
        }
        let mut nr = vec![0.0f64; n];
        for (t, slot) in nr.iter_mut().enumerate() {
            let row = &m[t * n..(t + 1) * n];
            let mut acc = p_sink * pers[t];
            for (s, &mts) in row.iter().enumerate() {
                acc += mts * r[s];
            }
            *slot = (1.0 - d) * pers[t] + d * acc;
        }
        r = nr;
    }
    r
}

#[test]
fn pagerank_matches_a_dense_power_iteration() {
    let g = thousand();
    let n = g.num_vertices();
    let table = edge_table(&g);
    let weighted: Vec<(usize, usize, f64)> = table.iter().map(|&(s, t)| (s, t, 1.0)).collect();
    let pers = vec![1.0 / n as f64; n];

    let want = dense_pagerank(n, &weighted, &pers, 0.85, 400);

    let mut rank = DenseProp::<f64, VertexTag>::new(g.graph_id());
    let iters = pagerank(
        &g,
        &Unity::<f64, EdgeTag>::NEW,
        0.85,
        params(n, 1e-13, 10_000),
        &mut rank,
    );
    assert!((5..10_000).contains(&iters), "converged in {iters} sweeps");

    let got = rank.as_slice();
    let mut worst = 0.0f64;
    for i in 0..n {
        worst = worst.max((got[i] - want[i]).abs());
    }
    println!("worst absolute difference vs dense reference: {worst:e}");
    assert!(worst < 1e-12, "worst absolute difference {worst:e}");

    // A sanity invariant the dense reference shares: with a personalisation
    // vector summing to one, so does the answer.
    let total: f64 = got.iter().sum();
    assert!((total - 1.0).abs() < 1e-9, "total mass {total}");
}

// ===========================================================================
// 2. PageRank: bit-identity across thread counts
// ===========================================================================

#[test]
fn pagerank_is_bit_identical_across_thread_counts() {
    let g = thousand();
    let n = g.num_vertices();

    let run = || {
        let mut rank = DenseProp::<f64, VertexTag>::new(g.graph_id());
        let iters = pagerank(
            &g,
            &Unity::<f64, EdgeTag>::NEW,
            0.85,
            params(n, 1e-13, 10_000),
            &mut rank,
        );
        (iters, rank.as_slice().to_vec())
    };

    let mut reference: Option<(usize, Vec<f64>)> = None;
    for t in THREADS {
        let got = at_threads(t, run);
        match &reference {
            None => reference = Some(got),
            Some((it, want)) => {
                assert_eq!(got.0, *it, "sweep count moved at {t} workers");
                for (i, (a, b)) in got.1.iter().zip(want).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "vertex {i} differs in the low bits at {t} workers: \
                         {a:e} vs {b:e}"
                    );
                }
            }
        }
    }
}

/// The plan, not the pool, fixes the fold order -- so a *different* plan is
/// allowed to give a different low bit, and this records that the test above
/// is testing something. If both plans agreed bit for bit the determinism
/// claim would be vacuous.
#[test]
fn the_plan_is_what_fixes_the_fold_order() {
    let g = thousand();
    let n = g.num_vertices();

    let with = |grain: usize| {
        let mut rank = DenseProp::<f64, VertexTag>::new(g.graph_id());
        pagerank(
            &g,
            &Unity::<f64, EdgeTag>::NEW,
            0.85,
            PowerIteration {
                epsilon: 1e-13,
                max_iter: 10_000,
                plan: Plan::new(n, grain),
            },
            &mut rank,
        );
        rank.as_slice().to_vec()
    };

    let fine = with(16);
    let coarse = with(1_000);
    // Same answer to well within tolerance...
    for (a, b) in fine.iter().zip(&coarse) {
        assert!((a - b).abs() < 1e-14);
    }
    // ...and not, in general, the same bits.
    assert!(
        fine.iter()
            .zip(&coarse)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "two different covers folded to identical bits everywhere; the \
         determinism test above would then prove nothing"
    );
}

// ===========================================================================
// 3. PageRank: the defects
// ===========================================================================

/// Defect: `graph_pagerank.hh:90-98` copies the *stale* buffer over the fresh
/// one after an odd number of swaps, so a truncated run hands the caller the
/// second-to-last iterate.
///
/// Two vertices, one edge `0 -> 1`, uniform personalisation `1/2`, damping
/// `0.85`, initial rank `[0.5, 0.5]`. Vertex 1 is a sink, so `p_sink = 0.5`:
///
/// * `v = 0`: no in-edges, `r = 0.5 * 0.5 = 0.25`,
///   `new = 0.15 * 0.5 + 0.85 * 0.25 = 0.2875`;
/// * `v = 1`: `r = 0.25 + 0.5 / 1 = 0.75`,
///   `new = 0.15 * 0.5 + 0.85 * 0.75 = 0.7125`.
///
/// One sweep is an odd number of swaps, which is exactly the case the C++
/// gets backwards: it would return `[0.5, 0.5]`.
#[test]
fn an_odd_number_of_sweeps_leaves_the_latest_iterate_in_the_callers_map() {
    let g = build(2, &[(0, 1)]);
    let mut rank = DenseProp::<f64, VertexTag>::new(g.graph_id());
    let iters = pagerank(
        &g,
        &Unity::<f64, EdgeTag>::NEW,
        0.85,
        params(2, 1e-12, 1),
        &mut rank,
    );
    assert_eq!(iters, 1);

    let got = rank.as_slice();
    assert_ne!(got[0], 0.5, "the initial iterate was handed back");
    assert!((got[0] - 0.2875).abs() < 1e-15, "rank[0] = {}", got[0]);
    assert!((got[1] - 0.7125).abs() < 1e-15, "rank[1] = {}", got[1]);

    // Two sweeps is the even case, which the C++ also gets right; check the
    // port did not merely swap which parity is broken.
    let mut rank2 = DenseProp::<f64, VertexTag>::new(g.graph_id());
    assert_eq!(
        pagerank(
            &g,
            &Unity::<f64, EdgeTag>::NEW,
            0.85,
            params(2, 1e-12, 2),
            &mut rank2
        ),
        2
    );
    let want = dense_pagerank(2, &[(0, 1, 1.0)], &[0.5, 0.5], 0.85, 2);
    for (a, b) in rank2.as_slice().iter().zip(&want) {
        assert!((a - b).abs() < 1e-15, "{a} vs {b}");
    }
}

/// Defect: `graph_pagerank.hh:77` divides by `deg[s]` without checking it.
/// Vertex 0 has an out-edge of weight zero, so `:49-50` files it as a sink
/// *and* `:77` divides by its zero degree -- the mass is both redistributed
/// and infinite.
#[test]
fn a_zero_weight_predecessor_does_not_poison_the_vector() {
    let g = build(3, &[(0, 1), (1, 2)]);
    let mut w = DenseProp::<f64, EdgeTag>::from_vec(g.graph_id(), vec![0.0, 1.0]);
    let _ = w.sized_for(g.edge_bound()).expect("size weights");

    let mut rank = DenseProp::<f64, VertexTag>::new(g.graph_id());
    pagerank(&g, &w, 0.85, params(3, 1e-13, 1_000), &mut rank);

    for (i, x) in rank.as_slice().iter().enumerate() {
        assert!(x.is_finite(), "rank[{i}] = {x}");
    }
    // Vertex 0 is a sink under this weighting, so the answer is the one for
    // the graph with that edge deleted.
    let alone = build(3, &[(1, 2)]);
    let mut want = DenseProp::<f64, VertexTag>::new(alone.graph_id());
    pagerank(
        &alone,
        &Unity::<f64, EdgeTag>::NEW,
        0.85,
        params(3, 1e-13, 1_000),
        &mut want,
    );
    for (a, b) in rank.as_slice().iter().zip(want.as_slice()) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
}

/// The personalisation vector is the `PerMap` of `graph_pagerank.hh:34`, and
/// a non-uniform one moves the answer in the documented direction: the
/// docstring's example gives low-degree vertices an artificially high score.
#[test]
fn a_personalisation_vector_biases_the_answer() {
    let g = build(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
    let n = 4;

    let mut flat = DenseProp::<f64, VertexTag>::new(g.graph_id());
    pagerank(
        &g,
        &Unity::<f64, EdgeTag>::NEW,
        0.85,
        params(n, 1e-14, 10_000),
        &mut flat,
    );
    // A directed cycle is vertex-transitive: uniform.
    for x in flat.as_slice() {
        assert!((x - 0.25).abs() < 1e-12, "{x}");
    }

    let pers = DenseProp::<f64, VertexTag>::from_vec(g.graph_id(), vec![0.7, 0.1, 0.1, 0.1]);
    let mut biased = DenseProp::<f64, VertexTag>::new(g.graph_id());
    pagerank_with(
        &g,
        &Unity::<f64, EdgeTag>::NEW,
        &pers,
        0.85,
        params(n, 1e-14, 10_000),
        &mut biased,
    );
    let got = biased.as_slice();
    assert!(got[0] > 0.25, "favoured vertex fell to {}", got[0]);
    assert!(got[0] > got[1] && got[1] > got[2] && got[2] > got[3]);
    let total: f64 = got.iter().sum();
    assert!((total - 1.0).abs() < 1e-9, "total {total}");

    // And it agrees with the dense reference.
    let want = dense_pagerank(
        n,
        &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
        &[0.7, 0.1, 0.1, 0.1],
        0.85,
        400,
    );
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
}

/// A map that arrives already sized is a warm start; only slots the kernel
/// has to create are seeded. Both halves of `centrality/__init__.py`'s
/// `if prop is None` are therefore reachable.
#[test]
fn an_already_sized_map_is_the_initial_iterate() {
    let g = build(2, &[(0, 1)]);
    // Start from the converged answer of the run above's two-sweep case and
    // ask for one more sweep; the result must be the *third* iterate, not
    // the first.
    let start = vec![0.2875, 0.7125];
    let mut rank = DenseProp::<f64, VertexTag>::from_vec(g.graph_id(), start.clone());
    pagerank(
        &g,
        &Unity::<f64, EdgeTag>::NEW,
        0.85,
        params(2, 1e-12, 1),
        &mut rank,
    );
    // p_sink = 0.7125; v0 = 0.15*0.5 + 0.85*(0.7125*0.5) = 0.377_812_5
    let want0 = 0.15 * 0.5 + 0.85 * (0.7125 * 0.5);
    let want1 = 0.15 * 0.5 + 0.85 * (0.7125 * 0.5 + 0.2875);
    assert!((rank.as_slice()[0] - want0).abs() < 1e-15);
    assert!((rank.as_slice()[1] - want1).abs() < 1e-15);
}

#[test]
#[should_panic(expected = "sweep: the plan must cover the output exactly")]
fn a_plan_that_does_not_cover_the_index_space_is_rejected() {
    let g = build(4, &[(0, 1)]);
    let mut rank = DenseProp::<f64, VertexTag>::new(g.graph_id());
    pagerank(
        &g,
        &Unity::<f64, EdgeTag>::NEW,
        0.85,
        params(3, 1e-6, 10),
        &mut rank,
    );
}

// ===========================================================================
// 4. Eigenvector
// ===========================================================================

/// The defining property, checked against the answer rather than against
/// another power iteration: for the converged `c` and reported `eig`,
/// `(A c)[v] == eig * c[v]` for every vertex, where `A` is the weighted
/// in-adjacency `graph_eigenvector.hh:66-70` sums over.
fn eigen_residual(g: &AdjList, w: &[f64], eig: f64, c: &[f64]) -> f64 {
    let table = edge_table(g);
    let mut ac = vec![0.0f64; g.vertex_bound().len()];
    for (ei, &(s, t)) in table.iter().enumerate() {
        if s != usize::MAX {
            ac[t] += w[ei] * c[s];
        }
    }
    let mut worst = 0.0f64;
    for i in 0..g.num_vertices() {
        worst = worst.max((ac[i] - eig * c[i]).abs());
    }
    worst
}

#[test]
fn eigenvector_satisfies_its_own_equation() {
    let mut r = Rng(0xfeed_face_cafe_beef);
    let n = 60;
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        edges.push((i, (i + 1) % n));
    }
    for _ in 0..240 {
        edges.push((r.below(n), r.below(n)));
    }
    let g = build(n, &edges);
    let m = g.edge_bound().len();

    let wv: Vec<f64> = (0..m).map(|i| 1.0 + (i % 5) as f64 * 0.25).collect();
    let mut w = DenseProp::<f64, EdgeTag>::from_vec(g.graph_id(), wv.clone());
    let _ = w.sized_for(g.edge_bound()).expect("size weights");

    let mut c = DenseProp::<f64, VertexTag>::new(g.graph_id());
    let (eig, iters) = eigenvector(&g, &w, params(n, 1e-14, 100_000), &mut c);
    assert!(iters > 1, "did not iterate");
    assert!(eig > 1.0, "eigenvalue {eig}");

    let residual = eigen_residual(&g, &wv, eig, c.as_slice());
    assert!(residual < 1e-10, "residual {residual:e}");

    // Unit L2 norm, which is what the `/= norm` at `:83` maintains.
    let l2: f64 = c.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((l2 - 1.0).abs() < 1e-12, "norm {l2}");
}

/// A directed cycle is vertex-transitive and its adjacency is a permutation
/// matrix: the leading eigenvalue is exactly 1 and the eigenvector uniform.
#[test]
fn a_directed_cycle_has_unit_eigenvalue() {
    let n = 7;
    let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    let g = build(n, &edges);

    let mut c = DenseProp::<f64, VertexTag>::new(g.graph_id());
    let (eig, _) = eigenvector(
        &g,
        &Unity::<f64, EdgeTag>::NEW,
        params(n, 1e-15, 1_000),
        &mut c,
    );
    assert!((eig - 1.0).abs() < 1e-12, "eigenvalue {eig}");
    let want = 1.0 / (n as f64).sqrt();
    for x in c.as_slice() {
        assert!((x - want).abs() < 1e-12, "{x} vs {want}");
    }

    // With a constant weight `w`, the eigenvalue scales to `w`.
    let mut c2 = DenseProp::<f64, VertexTag>::new(g.graph_id());
    let (eig2, _) = eigenvector(
        &g,
        &Constant::<f64, EdgeTag>::new(2.5),
        params(n, 1e-15, 1_000),
        &mut c2,
    );
    assert!((eig2 - 2.5).abs() < 1e-12, "eigenvalue {eig2}");
}

/// Defect: `graph_eigenvector.hh:83` divides by the norm unconditionally, so
/// a graph with no edges returns a vector of `NaN` and a `NaN` eigenvalue.
#[test]
fn an_edgeless_graph_has_a_zero_eigenvalue_and_no_nans() {
    let g = build(5, &[]);
    let mut c = DenseProp::<f64, VertexTag>::new(g.graph_id());
    let (eig, iters) = eigenvector(
        &g,
        &Unity::<f64, EdgeTag>::NEW,
        params(5, 1e-12, 100),
        &mut c,
    );
    assert_eq!(eig, 0.0, "eigenvalue {eig}");
    assert!(iters >= 1);
    for (i, x) in c.as_slice().iter().enumerate() {
        assert!(x.is_finite(), "c[{i}] = {x}");
        assert_eq!(*x, 0.0, "c[{i}] = {x}");
    }
}

/// The eigenvector twin of the inverted writeback. One sweep from the
/// uniform start `1/3` on the path `0 -> 1 -> 2`:
/// `c_temp = [0, 1/3, 1/3]`, `norm = sqrt(2)/3`, so the normalised iterate is
/// `[0, 1/sqrt 2, 1/sqrt 2]`. The C++ would hand back `[1/3, 1/3, 1/3]`.
#[test]
fn an_odd_eigenvector_sweep_leaves_the_latest_iterate_in_the_callers_map() {
    let g = build(3, &[(0, 1), (1, 2)]);
    let mut c = DenseProp::<f64, VertexTag>::new(g.graph_id());
    let (eig, iters) = eigenvector(&g, &Unity::<f64, EdgeTag>::NEW, params(3, 1e-12, 1), &mut c);
    assert_eq!(iters, 1);
    let want_norm = (2.0f64).sqrt() / 3.0;
    assert!((eig - want_norm).abs() < 1e-15, "eig {eig}");
    let h = 1.0 / (2.0f64).sqrt();
    let got = c.as_slice();
    assert!(got[0].abs() < 1e-15, "c[0] = {}", got[0]);
    assert!((got[1] - h).abs() < 1e-15, "c[1] = {}", got[1]);
    assert!((got[2] - h).abs() < 1e-15, "c[2] = {}", got[2]);
}

#[test]
fn eigenvector_is_bit_identical_across_thread_counts() {
    let g = thousand();
    let n = g.num_vertices();
    let run = || {
        let mut c = DenseProp::<f64, VertexTag>::new(g.graph_id());
        let (eig, iters) = eigenvector(
            &g,
            &Unity::<f64, EdgeTag>::NEW,
            params(n, 1e-13, 10_000),
            &mut c,
        );
        (eig.to_bits(), iters, c.as_slice().to_vec())
    };
    let base = at_threads(1, run);
    for t in THREADS {
        let got = at_threads(t, run);
        assert_eq!(got.0, base.0, "eigenvalue moved at {t} workers");
        assert_eq!(got.1, base.1, "sweep count moved at {t} workers");
        for (i, (a, b)) in got.2.iter().zip(&base.2).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "vertex {i} at {t} workers");
        }
    }
}

// ===========================================================================
// 5. Betweenness
// ===========================================================================

/// An independent Brandes, written against a flat edge list rather than the
/// graph traits, following `betweenness_centrality.hpp:296-388` but with the
/// distance map reset at every source (defect 4). Returns unnormalised
/// `(vertex, edge)` betweenness, halved for an undirected run exactly as
/// `divide_centrality_by_two` (`:384-387`) does.
fn reference_brandes(n: usize, edges: &[(usize, usize)], directed: bool) -> (Vec<f64>, Vec<f64>) {
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (ei, &(s, t)) in edges.iter().enumerate() {
        adj[s].push((t, ei));
        if !directed && s != t {
            adj[t].push((s, ei));
        }
    }

    let mut vb = vec![0.0f64; n];
    let mut eb = vec![0.0f64; edges.len()];

    for s in 0..n {
        let mut dist = vec![usize::MAX; n];
        let mut sigma = vec![0.0f64; n];
        let mut dep = vec![0.0f64; n];
        let mut pred: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        let mut order: Vec<usize> = Vec::new();
        let mut head = 0usize;

        dist[s] = 0;
        sigma[s] = 1.0;
        order.push(s);
        while head < order.len() {
            let u = order[head];
            head += 1;
            for &(w, ei) in &adj[u] {
                if w == u {
                    continue;
                }
                if dist[w] == usize::MAX {
                    dist[w] = dist[u] + 1;
                    sigma[w] = sigma[u];
                    pred[w].push((u, ei));
                    order.push(w);
                } else if dist[w] == dist[u] + 1 {
                    sigma[w] += sigma[u];
                    pred[w].push((u, ei));
                }
            }
        }

        for &u in order.iter().rev() {
            for &(p, ei) in &pred[u] {
                let f = (sigma[p] / sigma[u]) * (1.0 + dep[u]);
                dep[p] += f;
                eb[ei] += f;
            }
            if u != s {
                vb[u] += dep[u];
            }
        }
    }

    if !directed {
        for x in vb.iter_mut() {
            *x /= 2.0;
        }
        for x in eb.iter_mut() {
            *x /= 2.0;
        }
    }
    (vb, eb)
}

fn run_betweenness_directed(g: &AdjList) -> (Vec<f64>, Vec<f64>) {
    let mut vb = DenseProp::<f64, VertexTag>::new(g.graph_id());
    let mut eb = DenseProp::<f64, EdgeTag>::new(g.graph_id());
    betweenness(g, &mut vb, &mut eb);
    (vb.as_slice().to_vec(), eb.as_slice().to_vec())
}

fn run_betweenness_undirected(g: &AdjList) -> (Vec<f64>, Vec<f64>) {
    let mut vb = DenseProp::<f64, VertexTag>::new(g.graph_id());
    let mut eb = DenseProp::<f64, EdgeTag>::new(g.graph_id());
    betweenness(g.undirect(), &mut vb, &mut eb);
    (vb.as_slice().to_vec(), eb.as_slice().to_vec())
}

#[test]
fn betweenness_matches_a_reference_brandes() {
    let mut r = Rng(0x0123_4567_89ab_cdef);
    let mut checked = 0usize;

    for case in 0..100 {
        let n = 3 + r.below(10);
        let m = r.below(3 * n + 1);
        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(m);
        for _ in 0..m {
            // Self-loops and parallel edges are deliberate: boost's
            // `non_tree_edge` guard at `:203-204` exists for the first and
            // its `incoming[w].push_back` for the second, where each
            // parallel edge is a distinct shortest path.
            edges.push((r.below(n), r.below(n)));
        }
        let g = build(n, &edges);

        let (gv, ge) = run_betweenness_directed(&g);
        let (wv, we) = reference_brandes(n, &edges, true);
        for i in 0..n {
            assert!(
                (gv[i] - wv[i]).abs() < 1e-9,
                "case {case} directed vertex {i}: {} vs {}",
                gv[i],
                wv[i]
            );
        }
        for i in 0..edges.len() {
            assert!(
                (ge[i] - we[i]).abs() < 1e-9,
                "case {case} directed edge {i}: {} vs {}",
                ge[i],
                we[i]
            );
        }

        let (gv, ge) = run_betweenness_undirected(&g);
        let (wv, we) = reference_brandes(n, &edges, false);
        for i in 0..n {
            assert!(
                (gv[i] - wv[i]).abs() < 1e-9,
                "case {case} undirected vertex {i}: {} vs {}",
                gv[i],
                wv[i]
            );
        }
        for i in 0..edges.len() {
            assert!(
                (ge[i] - we[i]).abs() < 1e-9,
                "case {case} undirected edge {i}: {} vs {}",
                ge[i],
                we[i]
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 100);
}

/// Closed form, so this depends on no other implementation. On a path of `n`
/// vertices the unnormalised betweenness of vertex `i` is `i * (n - 1 - i)`
/// and of the edge `(i, i+1)` is `(i + 1) * (n - 1 - i)`, both for the
/// undirected graph (after `divide_centrality_by_two`) and for the directed
/// path, where only the ordered pairs `s < t` are connected.
#[test]
fn betweenness_on_a_path_is_the_closed_form() {
    const N: usize = 9;
    let edges: Vec<(usize, usize)> = (0..N - 1).map(|i| (i, i + 1)).collect();
    let g = build(N, &edges);

    for (label, (gv, ge)) in [
        ("directed", run_betweenness_directed(&g)),
        ("undirected", run_betweenness_undirected(&g)),
    ] {
        for (i, got) in gv.iter().enumerate().take(N) {
            let want = (i * (N - 1 - i)) as f64;
            assert_eq!(*got, want, "{label} vertex {i}");
        }
        for (i, got) in ge.iter().enumerate().take(N - 1) {
            let want = ((i + 1) * (N - 1 - i)) as f64;
            assert_eq!(*got, want, "{label} edge {i}");
        }
    }
}

/// Defect 4, isolated. `vdistance` is the one scratch vector
/// `betweenness_centrality.hpp:343-348` does not reset, and the only vertex
/// whose distance the BFS never writes is the source. The corruption
/// therefore needs an edge pointing *back* at a source that was already used
/// in the same thread, at the right stale depth. Here betweenness is computed
/// twice: once on the graph, once on the same graph with its vertices
/// relabelled so a different source runs first. A stale root distance makes
/// the two disagree; a reset one cannot.
#[test]
fn betweenness_is_unaffected_by_an_edge_back_into_an_earlier_source() {
    // A 4-cycle with a chord: every vertex is reachable from every other,
    // and each source's BFS ends by examining edges that point back at it.
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)];
    let g = build(4, &edges);
    let (gv, ge) = run_betweenness_directed(&g);
    let (wv, we) = reference_brandes(4, &edges, true);
    for i in 0..4 {
        assert!((gv[i] - wv[i]).abs() < 1e-12, "vertex {i}");
    }
    for i in 0..edges.len() {
        assert!((ge[i] - we[i]).abs() < 1e-12, "edge {i}");
    }

    // Relabelled: vertex k becomes 3 - k, so the source order is reversed.
    let flipped: Vec<(usize, usize)> = edges.iter().map(|&(s, t)| (3 - s, 3 - t)).collect();
    let h = build(4, &flipped);
    let (hv, _) = run_betweenness_directed(&h);
    for i in 0..4 {
        assert!(
            (hv[3 - i] - gv[i]).abs() < 1e-12,
            "relabelling changed vertex {i}: {} vs {}",
            hv[3 - i],
            gv[i]
        );
    }
}

/// Both maps are zeroed before the accumulation
/// (`init_centrality_map`, `:311-312`), so a reused map is not accumulated
/// into twice.
#[test]
fn betweenness_overwrites_the_maps_it_is_given() {
    let edges = [(0, 1), (1, 2), (2, 3)];
    let g = build(4, &edges);
    let mut vb = DenseProp::<f64, VertexTag>::new(g.graph_id());
    let mut eb = DenseProp::<f64, EdgeTag>::new(g.graph_id());
    betweenness(&g, &mut vb, &mut eb);
    let first = (vb.as_slice().to_vec(), eb.as_slice().to_vec());
    betweenness(&g, &mut vb, &mut eb);
    assert_eq!(vb.as_slice(), &first.0[..]);
    assert_eq!(eb.as_slice(), &first.1[..]);
}

#[test]
fn betweenness_is_bit_identical_across_thread_counts() {
    let mut r = Rng(0xdead_beef_1234_5678);
    let n = 220;
    let mut edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    for _ in 0..600 {
        edges.push((r.below(n), r.below(n)));
    }
    let g = build(n, &edges);

    let base = at_threads(1, || run_betweenness_directed(&g));
    for t in THREADS {
        let got = at_threads(t, || run_betweenness_directed(&g));
        for (i, (a, b)) in got.0.iter().zip(&base.0).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "vertex {i} at {t} workers");
        }
        for (i, (a, b)) in got.1.iter().zip(&base.1).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "edge {i} at {t} workers");
        }
    }
}

/// A disconnected graph: the unreachable half contributes nothing, and no
/// slot is left as `NaN` by a `0 / 0` in the dependency accumulation.
#[test]
fn unreachable_vertices_contribute_nothing() {
    let edges = [(0, 1), (1, 2), (3, 4)];
    let g = build(6, &edges);
    let (gv, ge) = run_betweenness_undirected(&g);
    for (i, x) in gv.iter().enumerate() {
        assert!(x.is_finite(), "vertex {i} = {x}");
    }
    // Vertex 1 is the only cut vertex; 5 is isolated.
    assert_eq!(gv[1], 1.0);
    assert_eq!(gv[0], 0.0);
    assert_eq!(gv[5], 0.0);
    assert_eq!(ge[2], 1.0, "the lone edge of the second component");
}

/// The unity fast path (`graph_selectors.hh:181-186`) and the general sum
/// (`:188-197`) must agree exactly, because the first is `n as f64` and the
/// second is `n` additions of `1.0` -- equal for every degree a graph can
/// have.
#[test]
fn the_unity_weight_fast_path_is_the_general_path() {
    let g = thousand();
    let n = g.num_vertices();
    let ones = vec![1.0f64; g.edge_bound().len()];
    let mut explicit = DenseProp::<f64, EdgeTag>::from_vec(g.graph_id(), ones);
    let _ = explicit.sized_for(g.edge_bound()).expect("size weights");

    let mut a = DenseProp::<f64, VertexTag>::new(g.graph_id());
    pagerank(
        &g,
        &Unity::<f64, EdgeTag>::NEW,
        0.85,
        params(n, 1e-13, 10_000),
        &mut a,
    );
    let mut b = DenseProp::<f64, VertexTag>::new(g.graph_id());
    pagerank(&g, &explicit, 0.85, params(n, 1e-13, 10_000), &mut b);

    for (i, (x, y)) in a.as_slice().iter().zip(b.as_slice()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "vertex {i}: {x:e} vs {y:e}");
    }
}
