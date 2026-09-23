//! U14 — components, against independent references.
//!
//! `gt_core::design` §16: *it compiles* is not evidence. Every kernel here is checked
//! against a reference built by a **different algorithm** — union-find for the
//! connected components, Kosaraju's two-pass for the strong ones — so an error
//! shared between implementation and reference would have to be an error in
//! two unrelated formulations at once.
//!
//! The numbering is checked too, not just the partition. `label_components`
//! returns its histogram to Python (`graph_components.hh:110`), and
//! `label_largest_component` indexes the labels with `h.argmax()`
//! (`__init__.py:1431`), so which component is called 0 is observable and a
//! partition-only test would not see it move.

use gt_algo::components::{UNLABELED, components, largest_component_mask, strong_components};
use gt_core::adj::AdjList;
use gt_core::graph::{GraphBase, GraphRef, VertexList};
use gt_core::ids::{VertexId, VertexTag};
use gt_core::prop::DenseProp;
use gt_core::view::{Filtered, Undirect};

// ---------------------------------------------------------------------------
// A deterministic generator. As in `benches/kernels.rs`: the graphs under test
// must not change when a dependency's `rand` version does.
// ---------------------------------------------------------------------------

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

/// G(n, m) with self-loops and parallel edges left in: both are exactly the
/// cases a components kernel must not miscount, and `AdjList` admits both.
fn random_digraph(rng: &mut SplitMix64, n: usize, m: usize) -> (AdjList, Vec<(usize, usize)>) {
    let mut g = AdjList::with_vertices(n);
    let mut spec = Vec::with_capacity(m);
    for _ in 0..m {
        let s = rng.below(n);
        let t = rng.below(n);
        g.add_edge(v(s), v(t)).expect("edge id space");
        spec.push((s, t));
    }
    (g, spec)
}

fn labels(g: &AdjList, run: impl FnOnce(&AdjList, &mut DenseProp<i64, VertexTag>) -> usize) -> (Vec<i64>, usize) {
    let mut map = DenseProp::<i64, VertexTag>::new(g.graph_id());
    let n = run(g, &mut map);
    (map.as_slice().to_vec(), n)
}

// ---------------------------------------------------------------------------
// Reference 1: union-find, for the connected components.
// ---------------------------------------------------------------------------

struct Dsu(Vec<usize>);

impl Dsu {
    fn new(n: usize) -> Self {
        Dsu((0..n).collect())
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.0[x] != x {
            self.0[x] = self.0[self.0[x]];
            x = self.0[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[ra] = rb;
        }
    }
}

/// Connected components by union-find, numbered exactly as
/// `components_recorder` numbers them: the counter advances once per new
/// class, in vertex order, so the class of vertex 0 is 0.
fn reference_connected(n: usize, spec: &[(usize, usize)]) -> (Vec<i64>, usize) {
    let mut dsu = Dsu::new(n);
    for &(s, t) in spec {
        dsu.union(s, t);
    }
    let mut seen = vec![UNLABELED; n];
    let mut out = Vec::with_capacity(n);
    let mut next = 0i64;
    for x in 0..n {
        let r = dsu.find(x);
        if seen[r] == UNLABELED {
            seen[r] = next;
            next += 1;
        }
        out.push(seen[r]);
    }
    (out, next as usize)
}

// ---------------------------------------------------------------------------
// Reference 2: Kosaraju, for the strong components.
//
// Deliberately *not* Tarjan: a second Tarjan would share every assumption with
// the implementation. Kosaraju numbers its components in a different order, so
// only the partition is compared here; the numbering is pinned separately, by
// the reverse-topological property that boost's ordering is equivalent to.
// ---------------------------------------------------------------------------

fn reference_strong(n: usize, spec: &[(usize, usize)]) -> Vec<i64> {
    let mut fwd = vec![Vec::new(); n];
    let mut rev = vec![Vec::new(); n];
    for &(s, t) in spec {
        fwd[s].push(t);
        rev[t].push(s);
    }

    // Pass 1: finish order of a DFS on the forward graph.
    let mut order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    for s in 0..n {
        if seen[s] {
            continue;
        }
        seen[s] = true;
        let mut stack = vec![(s, 0usize)];
        while let Some(&mut (u, ref mut i)) = stack.last_mut() {
            if *i < fwd[u].len() {
                let w = fwd[u][*i];
                *i += 1;
                if !seen[w] {
                    seen[w] = true;
                    stack.push((w, 0));
                }
            } else {
                order.push(u);
                stack.pop();
            }
        }
    }

    // Pass 2: DFS on the reverse graph in reverse finish order.
    let mut comp = vec![UNLABELED; n];
    let mut next = 0i64;
    for &s in order.iter().rev() {
        if comp[s] != UNLABELED {
            continue;
        }
        comp[s] = next;
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            for &w in &rev[u] {
                if comp[w] == UNLABELED {
                    comp[w] = next;
                    stack.push(w);
                }
            }
        }
        next += 1;
    }
    comp
}

/// Two labellings induce the same partition, whatever they call each class.
fn same_partition(a: &[i64], b: &[i64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .enumerate()
            .all(|(i, (&x, &y))| {
                a.iter().zip(b).skip(i).all(|(&x2, &y2)| (x == x2) == (y == y2))
            })
}

// ===========================================================================
// Acceptance: `components(g.undirect())` equals a union-find reference.
// ===========================================================================

#[test]
fn components_of_the_undirected_view_equal_union_find() {
    let mut rng = SplitMix64::new(0x5eed_0014);
    for case in 0..500 {
        let n = 1 + rng.below(40);
        let m = rng.below(2 * n + 1);
        let (g, spec) = random_digraph(&mut rng, n, m);

        let (got, n_got) = labels(&g, |g, map| components(g.undirect(), map));
        let (want, n_want) = reference_connected(n, &spec);

        assert_eq!(n_got, n_want, "case {case}: component count");
        assert_eq!(got, want, "case {case}: labels, including their numbering");
        // The count and the labels are the same statement twice; check the
        // range too, because `label_components` promises `[0, N-1]`
        // (`graph_components.hh:99-101`) and the histogram is indexed by it.
        assert!(got.iter().all(|&c| c >= 0 && (c as usize) < n_got));
    }
}

/// Isolated vertices, self-loops and a vertex with only a self-loop: the three
/// shapes where "an edge joins two classes" is not the whole rule.
#[test]
fn degenerate_shapes_are_components_too() {
    let empty = AdjList::with_vertices(0);
    let (lab, n) = labels(&empty, |g, m| components(g.undirect(), m));
    assert_eq!((lab.as_slice(), n), ([].as_slice(), 0));

    let isolated = AdjList::with_vertices(5);
    let (lab, n) = labels(&isolated, |g, m| components(g.undirect(), m));
    assert_eq!(n, 5);
    assert_eq!(lab, vec![0, 1, 2, 3, 4]);

    let mut loops = AdjList::with_vertices(3);
    loops.add_edge(v(1), v(1)).unwrap();
    loops.add_edge(v(0), v(0)).unwrap();
    let (lab, n) = labels(&loops, |g, m| components(g.undirect(), m));
    assert_eq!(n, 3, "a self-loop joins nothing");
    assert_eq!(lab, vec![0, 1, 2]);

    // Parallel edges collapse into one class and must not create a second.
    let mut parallel = AdjList::with_vertices(2);
    for _ in 0..4 {
        parallel.add_edge(v(0), v(1)).unwrap();
    }
    let (lab, n) = labels(&parallel, |g, m| components(g.undirect(), m));
    assert_eq!((lab, n), (vec![0, 0], 1));
}

/// The directed arm is the documented out-reachability partition, not the weak
/// components: `0 -> 1 <- 2` is one class from 0's side and a second for 2,
/// where `g.undirect()` sees one. This is the distinction
/// `label_components`' tag dispatch (`graph_components.hh:108-113`) makes on
/// the *view type* and a caller can only reach through `GraphView(g,
/// directed=...)`.
#[test]
fn the_directed_arm_is_out_reachability_and_says_so() {
    let mut g = AdjList::with_vertices(3);
    g.add_edge(v(0), v(1)).unwrap();
    g.add_edge(v(2), v(1)).unwrap();

    let (lab, n) = labels(&g, |g, m| components(g, m));
    assert_eq!((lab, n), (vec![0, 0, 1], 2));

    let (lab, n) = labels(&g, |g, m| components(g.undirect(), m));
    assert_eq!((lab, n), (vec![0, 0, 0], 1));
}

// ===========================================================================
// Acceptance: `strong_components` equals a reference Tarjan (here Kosaraju,
// which is stronger) on 500 random digraphs.
// ===========================================================================

#[test]
fn strong_components_equal_a_reference_on_500_random_digraphs() {
    let mut rng = SplitMix64::new(0x5eed_beef);
    for case in 0..500 {
        let n = 1 + rng.below(40);
        // Around m = 2n the giant SCC appears, so the cases span "all
        // singletons" through "one component plus a fringe".
        let m = rng.below(3 * n + 1);
        let (g, spec) = random_digraph(&mut rng, n, m);

        let (got, n_got) = labels(&g, |g, map| strong_components(g, map));
        let want = reference_strong(n, &spec);
        let n_want = want.iter().max().map_or(0, |&c| c as usize + 1);

        assert_eq!(n_got, n_want, "case {case}: SCC count");
        assert!(
            same_partition(&got, &want),
            "case {case}: partitions differ\n got: {got:?}\nwant: {want:?}"
        );

        // The numbering: boost assigns a component number when its root
        // *finishes* (`tarjan_scc_visitor::finish_vertex`), so the labels are a
        // reverse topological order of the condensation. That is the property
        // `label_largest_component`'s `h.argmax()` is indexing into, and it is
        // checkable without a second Tarjan.
        for &(s, t) in &spec {
            assert!(
                got[s] >= got[t],
                "case {case}: edge {s}->{t} labelled {} -> {}, which is not a \
                 reverse topological order",
                got[s],
                got[t]
            );
            assert_eq!(
                got[s] == got[t],
                want[s] == want[t],
                "case {case}: edge {s}->{t} disagrees on mutual reachability"
            );
        }
    }
}

/// A directed ring is one SCC; cutting it makes n of them, in exactly the
/// reverse order of the path. Both halves are hand-checkable, and the second
/// pins the numbering convention against a literal.
#[test]
fn a_ring_is_one_component_and_a_path_is_n() {
    let mut ring = AdjList::with_vertices(6);
    for i in 0..6 {
        ring.add_edge(v(i), v((i + 1) % 6)).unwrap();
    }
    let (lab, n) = labels(&ring, |g, m| strong_components(g, m));
    assert_eq!(n, 1);
    assert_eq!(lab, vec![0; 6]);

    let mut path = AdjList::with_vertices(4);
    for i in 0..3 {
        path.add_edge(v(i), v(i + 1)).unwrap();
    }
    let (lab, n) = labels(&path, |g, m| strong_components(g, m));
    assert_eq!(n, 4);
    // The sink finishes first and is numbered 0.
    assert_eq!(lab, vec![3, 2, 1, 0]);

    // A self-loop does not merge anything and does not make a vertex its own
    // second component either.
    let mut selfy = AdjList::with_vertices(2);
    selfy.add_edge(v(0), v(0)).unwrap();
    selfy.add_edge(v(0), v(1)).unwrap();
    let (lab, n) = labels(&selfy, |g, m| strong_components(g, m));
    assert_eq!((lab, n), (vec![1, 0], 2));
}

/// The two kernels answer different questions on the same graph, which is what
/// the split of `label_components`' tag dispatch is for. `0 -> 1 -> 0`, `2`:
/// two SCCs, two connected components, and they are *not* the same partition.
#[test]
fn strong_and_connected_are_not_the_same_partition() {
    let mut g = AdjList::with_vertices(4);
    g.add_edge(v(0), v(1)).unwrap();
    g.add_edge(v(1), v(0)).unwrap();
    g.add_edge(v(1), v(2)).unwrap();
    g.add_edge(v(3), v(2)).unwrap();

    let (weak, n_weak) = labels(&g, |g, m| components(g.undirect(), m));
    assert_eq!((weak, n_weak), (vec![0, 0, 0, 0], 1));

    let (strong, n_strong) = labels(&g, |g, m| strong_components(g, m));
    assert_eq!(n_strong, 3);
    assert_eq!(strong[0], strong[1], "0 and 1 are mutually reachable");
    assert_ne!(strong[1], strong[2]);
    assert_ne!(strong[3], strong[2]);
    // Reverse topological: the sink `2` is numbered below both its sources.
    assert!(strong[2] < strong[1] && strong[2] < strong[3]);
}

/// Depth without recursion. `boost::depth_first_search` recurses, so this
/// shape is a thread-stack overflow there; here the DFS stack is a `Vec`.
#[test]
fn a_long_path_does_not_need_a_deep_stack() {
    const N: usize = 200_000;
    let mut g = AdjList::with_vertices(N);
    for i in 0..N - 1 {
        g.add_edge(v(i), v(i + 1)).unwrap();
    }
    let (_, n) = labels(&g, |g, m| components(g, m));
    assert_eq!(n, 1);
    let (lab, n) = labels(&g, |g, m| strong_components(g, m));
    assert_eq!(n, N);
    assert_eq!(lab[0], (N - 1) as i64);
    assert_eq!(lab[N - 1], 0);
}

// ===========================================================================
// Acceptance: `largest_component_mask` feeds `Filtered::masked` directly.
// ===========================================================================

#[test]
fn the_largest_component_mask_is_a_view_of_the_right_size() {
    let mut rng = SplitMix64::new(0x5eed_0a5c);
    for case in 0..200 {
        let n = 1 + rng.below(30);
        let m = rng.below(n + 2);
        let (g, spec) = random_digraph(&mut rng, n, m);

        let mask = largest_component_mask((&g).undirect());
        // Sized for the *bound*, which is what `Filtered::masked` demands;
        // `num_vertices` would be the `graph_copy.cc:66-73` error.
        assert_eq!(mask.len(), g.vertex_bound().len(), "case {case}");

        let (want, n_comp) = reference_connected(n, &spec);
        let mut hist = vec![0usize; n_comp];
        for &c in &want {
            hist[c as usize] += 1;
        }
        let best = *hist.iter().max().expect("at least one vertex");
        let best_label = hist.iter().position(|&h| h == best).expect("argmax");

        let expected: Vec<u8> = want
            .iter()
            .map(|&c| u8::from(c as usize == best_label))
            .collect();
        assert_eq!(mask, expected, "case {case}: ties go to the lowest label");

        // ... and straight into the view.
        let emask = vec![1u8; g.edge_bound().len()];
        let view = Filtered::masked((&g).undirect(), &mask, &emask).expect("masks are sized");
        assert_eq!(view.num_vertices(), best, "case {case}: component size");
        assert_eq!(
            view.vertices().count(),
            view.num_vertices(),
            "case {case}: the identity graph_filtered.hh:301-312 gives up"
        );
        // Every edge of the component survives, and no edge leaves it.
        for u in view.vertices() {
            for i in view.out_edges(u) {
                assert_eq!(mask[i.other.index()], 1);
            }
        }
    }
}

/// `h.argmax()` (`__init__.py:1431`) resolves a tie to the *first* maximum,
/// and numpy's first is the lowest label, which is the component of the lowest
/// vertex. Two equal halves pin it.
#[test]
fn a_tie_goes_to_the_component_of_the_lowest_vertex() {
    let mut g = AdjList::with_vertices(4);
    g.add_edge(v(2), v(3)).unwrap();
    g.add_edge(v(0), v(1)).unwrap();
    let mask = largest_component_mask((&g).undirect());
    assert_eq!(mask, vec![1, 1, 0, 0]);
}

#[test]
fn an_empty_view_yields_an_empty_mask_rather_than_an_argmax_error() {
    let g = AdjList::with_vertices(0);
    assert!(largest_component_mask((&g).undirect()).is_empty());
}

// ===========================================================================
// The sentinel, and the filtered view.
// ===========================================================================

/// A vertex the view does not expose is `UNLABELED`, not 0. C++ leaves it
/// holding the property map's default — `0` for a fresh `int32_t` map — which
/// is indistinguishable from a real member of component 0, while the histogram
/// `HistogramPropertyMap` builds counts only the vertices it wrote.
#[test]
fn a_filtered_out_vertex_is_unlabeled_and_not_component_zero() {
    let mut g = AdjList::with_vertices(4);
    g.add_edge(v(0), v(1)).unwrap();
    g.add_edge(v(2), v(3)).unwrap();

    // Keep only {2, 3}: under C++'s default, 0 and 1 would read as component 0
    // and the caller could not tell them from 2 and 3's class.
    //
    // Note the composition order: `Filtered` has no `Undirect` impl, so the
    // filtered undirected view is `Filtered<Und<_>, _>` and not the other way
    // round. That is D3's normalisation doing its job -- there is one spelling
    // of the view, not two that would have to agree.
    let vmask = [0u8, 0, 1, 1];
    let emask = vec![1u8; g.edge_bound().len()];
    let und = Filtered::masked((&g).undirect(), &vmask, &emask).expect("masks are sized");
    let dir = Filtered::masked(&g, &vmask, &emask).expect("masks are sized");

    let mut map = DenseProp::<i64, VertexTag>::new(g.graph_id());
    let n = components(und, &mut map);
    assert_eq!(n, 1);
    assert_eq!(map.as_slice(), &[UNLABELED, UNLABELED, 0, 0]);

    let mut map = DenseProp::<i64, VertexTag>::new(g.graph_id());
    let n = strong_components(dir, &mut map);
    assert_eq!(n, 2);
    assert_eq!(map.as_slice()[0], UNLABELED);
    assert_eq!(map.as_slice()[1], UNLABELED);
    assert!(map.as_slice()[2] >= 0 && map.as_slice()[3] >= 0);

    // And the mask built from a filtered view is still bound-sized, so it can
    // be fed back in.
    let mask = largest_component_mask(und);
    assert_eq!(mask, vec![0, 0, 1, 1]);
}

/// A map reused across two runs must not keep the first run's labels for
/// vertices the second run does not reach.
#[test]
fn a_reused_map_is_reset_before_it_is_written() {
    let mut g = AdjList::with_vertices(3);
    g.add_edge(v(0), v(1)).unwrap();
    g.add_edge(v(1), v(2)).unwrap();

    let mut map = DenseProp::<i64, VertexTag>::new(g.graph_id());
    assert_eq!(components((&g).undirect(), &mut map), 1);
    assert_eq!(map.as_slice(), &[0, 0, 0]);

    let vmask = [1u8, 0, 0];
    let emask = vec![1u8; g.edge_bound().len()];
    let view = Filtered::masked((&g).undirect(), &vmask, &emask).expect("masks are sized");
    assert_eq!(components(view, &mut map), 1);
    assert_eq!(map.as_slice(), &[0, UNLABELED, UNLABELED]);
}

/// The map is minted from the graph's identity, and a map from another graph
/// is refused rather than written through. `Bound`'s `GraphId` comparison is
/// the whole of the check; graph-tool has no analogue.
#[test]
#[should_panic(expected = "different graph")]
fn a_map_from_another_graph_is_refused() {
    let g = AdjList::with_vertices(3);
    let other = AdjList::with_vertices(3);
    let mut map = DenseProp::<i64, VertexTag>::new(other.graph_id());
    let _ = components((&g).undirect(), &mut map);
}
