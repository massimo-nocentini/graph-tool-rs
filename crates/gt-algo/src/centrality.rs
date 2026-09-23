//! Centrality measures.
//!
//! Three kernels, ported from `src/graph/centrality/`:
//!
//! * [`pagerank`] -- `graph_pagerank.hh:30-100`;
//! * [`eigenvector`] -- `graph_eigenvector.hh:41-99`;
//! * [`betweenness`] -- `graph_betweenness.cc:70-94`, which delegates to the
//!   vendored Brandes in
//!   `src/boost-workaround/boost/graph/betweenness_centrality.hpp:296-388`.
//!
//! ## What this module fixes, and why
//!
//! Four defects are corrected here rather than reproduced. Each is a silent
//! wrong *number*, not a crash, which is why none of them shows up in
//! graph-tool's own test suite.
//!
//! 1. **The odd-iteration writeback is inverted** in both power iterations
//!    (`graph_pagerank.hh:90-98`, `graph_eigenvector.hh:94-95`). The loop
//!    maintains the invariant "`rank` names the latest iterate" by swapping
//!    two property-map *handles* at `:84`; the caller's storage is the one
//!    `r_temp` names after an odd number of swaps. The fixup then runs
//!    `put(rank, v, get(r_temp, v))` -- it copies the caller's stale storage
//!    over the fresh scratch and leaves the caller holding the
//!    second-to-last iterate. Convergence hides it (the last two iterates
//!    differ by less than `epsilon` in L1 by construction) but truncation by
//!    `max_iter` does not. Here the two buffers are ordinary `&mut [f64]`
//!    bindings and the final copy, when parity calls for one, goes *into*
//!    the caller's map.
//! 2. **Division by a zero weighted out-degree.** `graph_pagerank.hh:77`
//!    computes `rank[s] * weight[e] / deg[s]` for every predecessor `s`
//!    without checking `deg[s]`. A vertex whose out-edges all carry weight
//!    zero is pushed onto `sinks` at `:50` -- so its mass is *already*
//!    redistributed through `p_sink` -- and is nevertheless divided by at
//!    `:77`, producing `inf` or `NaN` that poisons the whole vector. Here a
//!    zero weighted out-degree contributes nothing, which is the only
//!    reading consistent with `:49-50`. With a unity weight map the test is
//!    unobservable (`deg[s] == 0` implies `s` has no out-edges, so `s` is
//!    never a predecessor) and const-folds away.
//! 3. **Division by a zero norm.** `graph_eigenvector.hh:83` divides by
//!    `sqrt(norm)` unconditionally, so a graph with no edges -- or one whose
//!    iterate has collapsed to zero -- returns a vector of `NaN`. Here a
//!    zero norm leaves the iterate alone.
//! 4. **The Brandes distance map is never reset between sources**
//!    (`betweenness_centrality.hpp:341-349`: `vincoming`, `vpath_count` and
//!    `vdependency` are cleared per source, `vdistance` is not, and the BFS
//!    visitor at `:183-191` only ever *writes* `distance[w]` for a tree
//!    edge, never for the root). The root keeps whatever distance the
//!    previous source's BFS left there, so `non_tree_edge` (`:199-209`) can
//!    match `distance[s] == distance[v] + 1` on an edge pointing back at the
//!    source, push it into `incoming[s]`, and corrupt both `path_count[s]`
//!    and the dependency accumulation. Here every BFS sets `dist[s] = 0`.
//!
//! ## Determinism
//!
//! Every reduction in this module folds in a fixed order, so the result is a
//! function of the input and the [`Plan`] and of nothing else -- not the
//! worker count, not the steal order. `p_sink` goes through
//! [`det_reduce`](gt_core::par::det_reduce) directly; the vertex sweeps use
//! [`sweep`], which is `det_reduce`'s discipline for the case `det_reduce`
//! cannot express (its `map` is an `Fn`, so no chunk can hold a `&mut` into
//! the output). graph-tool reduces `delta`, `norm` and `p_sink` under
//! `reduction(+:...)` (`graph_pagerank.hh:62, :68`,
//! `graph_eigenvector.hh:60, :78`) and the betweenness accumulation under
//! `#pragma omp atomic` (`betweenness_centrality.hpp:275-278`); all four are
//! reassociated by the schedule, so the low bits move with
//! `OMP_NUM_THREADS`.

use std::collections::VecDeque;
use std::ops::Range;

use rayon::prelude::*;

use gt_core::dir::Dir;
use gt_core::graph::{Bidirectional, GraphRef, VertexList};
use gt_core::ids::{EdgeId, EdgeTag, VertexId, VertexTag};
use gt_core::par::{Plan, Seed, det_reduce};
use gt_core::prop::{Constant, DenseProp, ReadProp};

/// Parameters for a power-iteration centrality.
#[derive(Clone, Copy, Debug)]
pub struct PowerIteration {
    /// Convergence threshold on the L1 change per sweep.
    pub epsilon: f64,
    /// Hard cap on sweeps.
    ///
    /// `0` means *no* cap, which is what `max_iter > 0 && iter == max_iter`
    /// (`graph_pagerank.hh:86`) means and what `centrality/__init__.py`
    /// passes for `max_iter=None`.
    pub max_iter: usize,
    /// Partition used for the parallel sweep. Fixed, so the result is
    /// bit-identical across thread counts.
    ///
    /// It partitions the **vertex index space**, not the vertex set:
    /// `plan.len()` must equal `g.vertex_bound().len()`, which is also the
    /// length of the property maps involved. For an unfiltered graph the two
    /// coincide, so `Plan::new(g.num_vertices(), grain)` is what a caller
    /// writes.
    pub plan: Plan,
}

/// The generator [`det_reduce`] derives per chunk is unused by every
/// reduction here -- none of them is randomised -- but the signature demands
/// a root, and a *fixed* root is what keeps the call a pure function of its
/// arguments.
const NO_ENTROPY: Seed = Seed([0u8; 32]);

/// How many chunks [`betweenness`] splits its source list into.
///
/// Fixed, and deliberately not a function of the pool size: each chunk owns a
/// full private accumulator pair, so the count sets both the peak memory
/// (`CHUNKS * (|V| + |E|) * 8` bytes) and the fold order. A thread-derived
/// count would make the answer move with `RAYON_NUM_THREADS`, which is
/// exactly what `#pragma omp atomic` at `betweenness_centrality.hpp:277`
/// does.
const BETWEENNESS_CHUNKS: usize = 16;

// ===========================================================================
// The deterministic scatter-and-reduce
// ===========================================================================

/// Run `body` over the [`Plan`]'s cover of `out`, in parallel, and fold the
/// per-chunk `f64` it returns **in chunk order**.
///
/// This is [`det_reduce`](gt_core::par::det_reduce) for a sweep that also
/// *writes*. `det_reduce`'s `map` is an `Fn(Range, &mut ChaCha8Rng) -> T`, so
/// it can capture only shared state and no chunk can hold a `&mut` into the
/// output; here the output is cut into the plan's chunks up front with
/// `split_at_mut`, so each chunk owns its slice and the borrow checker --
/// not a comment -- is what proves the chunks disjoint.
///
/// The fold is a sequential left-to-right `+` over an index-preserving
/// `collect`, so `sweep` is bit-identical at any worker count, for the same
/// reason `det_reduce` is.
///
/// One `Vec` of `plan.chunks()` slice headers and one of `plan.chunks()`
/// partials are allocated per call -- per *sweep*, never per vertex.
///
/// # Panics
///
/// If `plan.len() != out.len()`.
fn sweep<F>(plan: Plan, out: &mut [f64], body: F) -> f64
where
    F: Fn(Range<usize>, &mut [f64]) -> f64 + Sync + Send,
{
    assert_eq!(
        plan.len(),
        out.len(),
        "sweep: the plan must cover the output exactly"
    );

    let mut pieces: Vec<(Range<usize>, &mut [f64])> = Vec::with_capacity(plan.chunks());
    let mut rest: &mut [f64] = out;
    for k in 0..plan.chunks() {
        let r = plan.range(k);
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(r.len());
        pieces.push((r, head));
        rest = tail;
    }
    debug_assert!(rest.is_empty(), "Plan::range is not an exact cover");

    let partials: Vec<f64> = pieces
        .into_par_iter()
        .map(|(r, piece)| body(r, piece))
        .collect();

    partials.into_iter().fold(0.0, |a, b| a + b)
}

/// Membership in this view's vertex set, indexed by the *unfiltered* vertex
/// index.
///
/// The sweeps walk the index space rather than the vertex set so that a
/// chunk of the [`Plan`] is a contiguous slice of the property map. A
/// filtered view's vertex set is not contiguous, so the walk needs an O(1)
/// membership test and the trait layer offers none; this is it.
fn active_mask<G: VertexList>(g: G) -> Vec<bool> {
    let mut mask = vec![false; g.vertex_bound().len()];
    for v in g.vertices() {
        mask[v.index()] = true;
    }
    mask
}

/// The balanced chunk size a [`Plan`] was built with, recovered so that a
/// second plan over a different length can be cut the same way.
fn grain_of(plan: Plan) -> usize {
    plan.len().div_ceil(plan.chunks()).max(1)
}

/// Fill the slots a `sized_for` call has just created, leaving any the caller
/// already owned untouched.
///
/// `centrality/__init__.py` seeds a fresh map with `pers` (pagerank) or
/// `1/N` (eigenvector) before calling the kernel, and the kernel itself
/// treats whatever it is handed as the initial iterate. Both halves are
/// preserved: a map that arrives already sized is a warm start, a map that
/// arrives empty gets the Python front-end's default.
fn seed_new_slots<P>(data: &mut [f64], had: usize, init: &P)
where
    P: ReadProp<VertexTag, Value = f64>,
{
    for (i, slot) in data.iter_mut().enumerate().skip(had) {
        *slot = init.get(VertexId::from_index(i));
    }
}

// ===========================================================================
// PageRank
// ===========================================================================

/// PageRank.
///
/// The damping sum is folded through [`gt_core::par::det_reduce`], so the
/// value does not depend on the thread count. graph-tool's equivalent reduces
/// under `#pragma omp critical` and therefore does.
///
/// The personalisation vector is uniform (`1/N`), which is the default
/// `pagerank` binds when the caller passes none
/// (`graph_pagerank.cc:43-44`). For a non-uniform one, call
/// [`pagerank_with`].
///
/// `rank` is the initial iterate as well as the output. Slots that
/// [`DenseProp::sized_for`](gt_core::prop::DenseProp::sized_for) has to
/// create are seeded with the personalisation value, so a freshly
/// constructed map reproduces `centrality/__init__.py`'s
/// `prop.fa = 1. / g.num_vertices()`.
///
/// Returns the number of sweeps performed.
///
/// # Panics
///
/// If `params.plan` does not cover `g.vertex_bound()`, or if `rank` belongs
/// to another graph.
pub fn pagerank<G, W>(
    g: G,
    weight: &W,
    damping: f64,
    params: PowerIteration,
    rank: &mut DenseProp<f64, VertexTag>,
) -> usize
where
    G: Bidirectional + VertexList + Sync,
    W: ReadProp<EdgeTag, Value = f64> + Sync,
{
    let n = g.num_vertices();
    let uniform = if n == 0 { 0.0 } else { 1.0 / n as f64 };
    pagerank_with(g, weight, &Constant::new(uniform), damping, params, rank)
}

/// PageRank with a personalisation vector.
///
/// `pers` is `PerMap` of `graph_pagerank.hh:34`. It is *not* normalised here,
/// exactly as the C++ does not normalise it: the docstring's worked example
/// feeds it a vector summing to far more than one.
///
/// Returns the number of sweeps performed.
///
/// # Panics
///
/// As [`pagerank`].
pub fn pagerank_with<G, W, P>(
    g: G,
    weight: &W,
    pers: &P,
    damping: f64,
    params: PowerIteration,
    rank: &mut DenseProp<f64, VertexTag>,
) -> usize
where
    G: Bidirectional + VertexList + Sync,
    W: ReadProp<EdgeTag, Value = f64> + Sync,
    P: ReadProp<VertexTag, Value = f64> + Sync,
{
    let bound = g.vertex_bound();
    let len = bound.len();
    let active = active_mask(g);

    // `init degs` -- `graph_pagerank.hh:44-51`. `out_degree(v, g, weight)` is
    // `graph_selectors.hh:188-197`, a plain sum over the out-edges, with the
    // unity specialisation at `:181-186` that this const branch reproduces.
    let mut deg = vec![0.0f64; len];
    let mut sinks: Vec<VertexId> = Vec::new();
    for v in g.vertices() {
        let k = if W::IS_UNITY {
            g.out_degree(v) as f64
        } else {
            let mut acc = 0.0;
            for inc in g.out_edges(v) {
                acc += weight.get(inc.edge);
            }
            acc
        };
        deg[v.index()] = k;
        if k == 0.0 {
            sinks.push(v);
        }
    }

    let had = rank.len();
    let mut view = rank
        .sized_for(bound)
        .expect("pagerank: rank map does not belong to this graph");
    seed_new_slots(view.as_mut_slice(), had, pers);

    let mut scratch = vec![0.0f64; len];
    let mut cur: &mut [f64] = view.as_mut_slice();
    let mut next: &mut [f64] = &mut scratch;

    // A second plan for the sink list, cut with the caller's grain so that it
    // too is a function of `params` alone.
    let sink_plan = Plan::new(sinks.len(), grain_of(params.plan));

    let d = damping;
    let mut delta = params.epsilon + 1.0;
    let mut iter = 0usize;
    let gr = &g;

    while delta >= params.epsilon {
        let prev: &[f64] = cur;
        let sinks_ref = &sinks;

        // `graph_pagerank.hh:60-65`.
        let p_sink = det_reduce(
            sink_plan,
            NO_ENTROPY,
            |r, _rng| {
                let mut acc = 0.0;
                for s in &sinks_ref[r] {
                    acc += prev[s.index()];
                }
                acc
            },
            0.0,
            |a, b| a + b,
        );

        // `graph_pagerank.hh:67-83`.
        let active_ref = &active;
        let deg_ref = &deg;
        delta = sweep(params.plan, next, |range, out| {
            let base = range.start;
            let mut local = 0.0f64;
            for (offset, slot) in out.iter_mut().enumerate() {
                let i = base + offset;
                if !active_ref[i] {
                    // Not a vertex of this view: carry it through unchanged
                    // rather than letting the buffer swap alternate two
                    // meanings in a slot nothing reads.
                    *slot = prev[i];
                    continue;
                }
                let v = VertexId::from_index(i);
                let pv = pers.get(v);
                let mut r = p_sink * pv;
                for inc in gr.in_edges(v) {
                    let s = inc.other.index();
                    let ds = deg_ref[s];
                    // Defect 2: a zero weighted out-degree is a sink, and a
                    // sink's mass is already in `p_sink`.
                    if ds != 0.0 {
                        r += (prev[s] * weight.get(inc.edge)) / ds;
                    }
                }
                let nv = (1.0 - d) * pv + d * r;
                local += (nv - prev[i]).abs();
                *slot = nv;
            }
            local
        });

        std::mem::swap(&mut cur, &mut next);
        iter += 1;
        if params.max_iter > 0 && iter == params.max_iter {
            break;
        }
    }

    // Defect 1, the right way round: after an odd number of swaps the latest
    // iterate is in the scratch buffer and `next` is the caller's map.
    if iter % 2 == 1 {
        next.copy_from_slice(cur);
    }
    iter
}

// ===========================================================================
// Eigenvector centrality
// ===========================================================================

/// Eigenvector centrality.
///
/// Power iteration on the *in*-adjacency, normalised in L2 each sweep and
/// stopped on the L1 change: `graph_eigenvector.hh:56-92`. Returns
/// `(eigenvalue, sweeps)`, where the eigenvalue is the last norm computed
/// (`:97`) and is `0.0` if no sweep ran.
///
/// `x` is the initial iterate as well as the output; slots created by
/// sizing it are seeded with `1/N`, reproducing
/// `centrality/__init__.py`'s `vprop.fa = 1. / g.num_vertices()`.
///
/// # Panics
///
/// If `params.plan` does not cover `g.vertex_bound()`, or if `x` belongs to
/// another graph.
pub fn eigenvector<G, W>(
    g: G,
    weight: &W,
    params: PowerIteration,
    x: &mut DenseProp<f64, VertexTag>,
) -> (f64, usize)
where
    G: Bidirectional + VertexList + Sync,
    W: ReadProp<EdgeTag, Value = f64> + Sync,
{
    let bound = g.vertex_bound();
    let len = bound.len();
    let active = active_mask(g);
    let n = g.num_vertices();
    let uniform = if n == 0 { 0.0 } else { 1.0 / n as f64 };

    let had = x.len();
    let mut view = x
        .sized_for(bound)
        .expect("eigenvector: centrality map does not belong to this graph");
    seed_new_slots(
        view.as_mut_slice(),
        had,
        &Constant::<f64, VertexTag>::new(uniform),
    );

    let mut scratch = vec![0.0f64; len];
    let mut cur: &mut [f64] = view.as_mut_slice();
    let mut next: &mut [f64] = &mut scratch;

    let mut norm = 0.0f64;
    let mut delta = params.epsilon + 1.0;
    let mut iter = 0usize;
    let gr = &g;
    let active_ref = &active;

    while delta >= params.epsilon {
        let prev: &[f64] = cur;

        // `graph_eigenvector.hh:58-72`: build `c_temp` and accumulate the
        // squared norm in the same pass.
        let sq = sweep(params.plan, next, |range, out| {
            let base = range.start;
            let mut acc = 0.0f64;
            for (offset, slot) in out.iter_mut().enumerate() {
                let i = base + offset;
                if !active_ref[i] {
                    *slot = prev[i];
                    continue;
                }
                let v = VertexId::from_index(i);
                let mut s = 0.0f64;
                for inc in gr.in_edges(v) {
                    s += weight.get(inc.edge) * prev[inc.other.index()];
                }
                *slot = s;
                // `power(c_temp[v], 2)` -- `__gnu_cxx::power` is `x * x`.
                acc += s * s;
            }
            acc
        });

        norm = sq.sqrt();

        // `graph_eigenvector.hh:76-85`. Defect 3: `norm == 0` means the
        // iterate is identically zero, and `0 / 0` is not its normalisation.
        let scale = if norm == 0.0 { 1.0 } else { norm };
        delta = sweep(params.plan, next, |range, out| {
            let base = range.start;
            let mut acc = 0.0f64;
            for (offset, slot) in out.iter_mut().enumerate() {
                let i = base + offset;
                if !active_ref[i] {
                    continue;
                }
                *slot /= scale;
                acc += (*slot - prev[i]).abs();
            }
            acc
        });

        std::mem::swap(&mut cur, &mut next);
        iter += 1;
        if params.max_iter > 0 && iter == params.max_iter {
            break;
        }
    }

    if iter % 2 == 1 {
        next.copy_from_slice(cur);
    }
    (norm, iter)
}

// ===========================================================================
// Betweenness (Brandes)
// ===========================================================================

/// Per-chunk scratch for [`betweenness`].
///
/// Allocated once per chunk and reused across that chunk's sources, which is
/// what `firstprivate(vincoming, vdistance, vdependency, vpath_count)`
/// (`betweenness_centrality.hpp:331-333`) buys in the C++: after the first
/// source no `Vec` in here grows again.
struct Brandes {
    /// BFS distance, `-1` standing in for boost's `white` colour.
    dist: Vec<i64>,
    /// `sigma`: number of shortest paths from the source.
    sigma: Vec<f64>,
    /// `delta`: dependency accumulator.
    dep: Vec<f64>,
    /// Shortest-path DAG predecessors, as `(predecessor, edge)`.
    incoming: Vec<Vec<(VertexId, EdgeId)>>,
    /// Vertices in BFS *dequeue* order; boost pushes these on a stack
    /// (`:173-176`) and pops it, so the accumulation walks this in reverse.
    order: Vec<VertexId>,
    /// The BFS frontier.
    queue: VecDeque<VertexId>,
    /// Vertex-betweenness partial for this chunk.
    vb: Vec<f64>,
    /// Edge-betweenness partial for this chunk.
    eb: Vec<f64>,
}

impl Brandes {
    fn new(vlen: usize, elen: usize) -> Self {
        Brandes {
            dist: vec![-1; vlen],
            sigma: vec![0.0; vlen],
            dep: vec![0.0; vlen],
            incoming: vec![Vec::new(); vlen],
            order: Vec::new(),
            queue: VecDeque::new(),
            vb: vec![0.0; vlen],
            eb: vec![0.0; elen],
        }
    }

    /// One source: BFS, then the reverse-order dependency accumulation.
    fn run<G>(&mut self, g: G, verts: &[VertexId], s: VertexId)
    where
        G: GraphRef,
    {
        // `betweenness_centrality.hpp:343-348`, plus the `dist` reset the C++
        // omits (defect 4).
        for v in verts {
            let i = v.index();
            self.incoming[i].clear();
            self.sigma[i] = 0.0;
            self.dep[i] = 0.0;
            self.dist[i] = -1;
        }
        self.order.clear();
        self.queue.clear();

        let si = s.index();
        self.sigma[si] = 1.0;
        self.dist[si] = 0;
        self.queue.push_back(s);

        while let Some(u) = self.queue.pop_front() {
            // boost calls `examine_vertex` on *dequeue*
            // (`breadth_first_search.hpp`), so this is the order the stack
            // at `:173-176` receives.
            self.order.push(u);
            let ui = u.index();
            let du = self.dist[ui];
            let su = self.sigma[ui];
            for inc in g.out_edges(u) {
                let w = inc.other;
                if w == u {
                    // `if (v == w) return;` -- `:203-204`. A self-loop can
                    // never shorten a path and must never enter `incoming`.
                    continue;
                }
                let wi = w.index();
                if self.dist[wi] < 0 {
                    // `tree_edge`, `:183-191`.
                    self.dist[wi] = du + 1;
                    self.sigma[wi] = su;
                    self.incoming[wi].push((u, inc.edge));
                    self.queue.push_back(w);
                } else if self.dist[wi] == du + 1 {
                    // `non_tree_edge`, `:199-209`. Parallel edges are
                    // distinct shortest paths, and each is pushed.
                    self.sigma[wi] += su;
                    self.incoming[wi].push((u, inc.edge));
                }
            }
        }

        // `betweenness_centrality.hpp:357-377`.
        for idx in (0..self.order.len()).rev() {
            let u = self.order[idx];
            let ui = u.index();
            let su = self.sigma[ui];
            for &(v, e) in &self.incoming[ui] {
                let vi = v.index();
                let factor = (self.sigma[vi] / su) * (1.0 + self.dep[ui]);
                self.dep[vi] += factor;
                self.eb[e.index()] += factor;
            }
            if u != s {
                self.vb[ui] += self.dep[ui];
            }
        }
    }
}

/// Betweenness centrality (Brandes).
///
/// Unweighted, with **every** vertex of the view as a pivot -- the
/// `pivots = list(g.vertices())` default of
/// `graph_tool.centrality.betweenness`. Both maps are zeroed first
/// (`init_centrality_map`, `betweenness_centrality.hpp:311-312`) and the
/// result is *unnormalised*: graph-tool's `norm=True` is a separate pass
/// (`normalize_betweenness`, `graph_betweenness.cc:31-68`).
///
/// On an undirected view every pair is counted from both endpoints, so both
/// maps are halved at the end, exactly as
/// `divide_centrality_by_two` (`:384-387`) does.
///
/// # Panics
///
/// If either map belongs to another graph.
pub fn betweenness<G>(
    g: G,
    vertex_bc: &mut DenseProp<f64, VertexTag>,
    edge_bc: &mut DenseProp<f64, EdgeTag>,
) where
    G: GraphRef + VertexList + Sync,
{
    let vbound = g.vertex_bound();
    let ebound = g.edge_bound();
    let vlen = vbound.len();
    let elen = ebound.len();

    let verts: Vec<VertexId> = g.vertices().collect();

    // `betweenness_centrality.hpp:311-312`.
    let mut vview = vertex_bc
        .sized_for(vbound)
        .expect("betweenness: vertex map does not belong to this graph");
    vview.as_mut_slice().fill(0.0);
    let mut eview = edge_bc
        .sized_for(ebound)
        .expect("betweenness: edge map does not belong to this graph");
    eview.as_mut_slice().fill(0.0);

    let plan = Plan::new(verts.len(), verts.len().div_ceil(BETWEENNESS_CHUNKS).max(1));
    let gr = &g;
    let verts_ref = &verts;

    let partials: Vec<(Vec<f64>, Vec<f64>)> = (0..plan.chunks())
        .into_par_iter()
        .map(|k| {
            let mut scratch = Brandes::new(vlen, elen);
            for &s in &verts_ref[plan.range(k)] {
                scratch.run(*gr, verts_ref, s);
            }
            (scratch.vb, scratch.eb)
        })
        .collect();

    // Sequential, chunk order: the association is the plan's, not the pool's.
    let vout = vview.as_mut_slice();
    let eout = eview.as_mut_slice();
    for (pv, pe) in &partials {
        for (a, b) in vout.iter_mut().zip(pv) {
            *a += *b;
        }
        for (a, b) in eout.iter_mut().zip(pe) {
            *a += *b;
        }
    }

    if !<G::Dir as Dir>::DIRECTED {
        for a in vout.iter_mut() {
            *a /= 2.0;
        }
        for a in eout.iter_mut() {
            *a /= 2.0;
        }
    }
}
