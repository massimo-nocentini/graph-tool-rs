//! Topological predicates and orderings.
//!
//! Six kernels from five different C++ translation units, and the thing they
//! have in common is that each one is a place where "which view am I on"
//! decides the answer.
//!
//! ## The direction each kernel actually reads
//!
//! | kernel | incidence used | why |
//! |---|---|---|
//! | [`is_dag`], [`topological_sort`] | `out_edges` | a topological order is a statement about arcs; `boost::topological_sort` walks out-edges and throws `not_a_dag` on a back edge (`topology/graph_topological_sort.cc:34`) |
//! | [`is_bipartite`] | `all_edges` | `graph_bipartite.cc:84` dispatches under `{.tr=never_directed}` and the Python wrapper builds `GraphView(g, directed=False)` first, so bipartiteness never sees an orientation |
//! | [`global_clustering`] | `all_edges` | `clustering/__init__.py:226-227` undirects a directed graph before calling the C++ at all |
//! | [`count_triangles`], [`reciprocity`] | `edges()` | the count is per *edge*; see below |
//! | [`kcore_decomposition`] | `degree` + `all_edges` | `graph_kcore.hh:41`/`:62` read the whole block, so the core number is the total-degree one on a directed graph and a self-loop counts twice (`topology/__init__.py:1832-1836`) |
//!
//! Because [`Und<G>::All`](gt_core::view::Und) **is** `G::All`
//! (`view/undirected.rs:107-116`), the three direction-insensitive kernels give
//! the same answer on `&g` and on `g.undirect()` by construction rather than by
//! a runtime `is_directed(g)` test. That equality is asserted in
//! `tests/u17_topology.rs`.
//!
//! ## Why the counting kernels are bounded on `EdgeList`
//!
//! `get_triangles` (`clustering/graph_clustering.hh:50-93`) counts ordered
//! neighbour pairs per vertex and then repairs the multiplicity twice: once
//! with `/ 2` for the undirected case at `:92`, and once with `/ 3` at `:132`
//! when the caller wants a triangle *count*. Both repairs are divisions
//! applied to a degree summation, and a degree summation over an undirected
//! view is exactly the arithmetic that makes a filtered `num_edges` wrong
//! ([DESIGN](gt_core::design) D3). Here the kernels enumerate the edge set once, build a
//! sorted simple adjacency, and count each triangle at its unique `u < v < w`
//! witness -- so there is no multiplicity to divide out and no division to get
//! wrong.
//!
//! ## Where this deliberately does not reproduce the C++
//!
//! `get_triangles` marks a neighbour with `mark[u] = w` (`:64`, an
//! *assignment*) while accumulating the degree with `k += w` (`:65`, a
//! *sum*). On a simple graph the two agree. On a multigraph they do not: the
//! numerator then uses the 0/1 adjacency of the underlying simple graph and
//! the denominator uses the multiplicity-weighted one, so the ratio is not
//! `Tr(A³) / Σ_{i≠j}[A²]_{ij}` for any single `A` -- which is the definition
//! `clustering/__init__.py:194-201` states. This port uses the *simple*
//! underlying graph throughout (parallel edges collapsed, self-loops dropped,
//! matching `:60` and `:72`'s `u == v` skips), so numerator and denominator
//! are consistent and `global_clustering == 3 · count_triangles / triples`
//! holds exactly. For a simple graph -- graph-tool's own karate doctest
//! included -- the two implementations agree.

use std::cmp::Ordering;

use gt_core::graph::{Bidirectional, EdgeList, GraphRef, VertexList};
use gt_core::ids::VertexId;
use gt_core::par::{Plan, Seed, det_reduce};

// ---------------------------------------------------------------------------
// Sorted adjacency in CSR form: the shared substrate of the counting kernels.
// ---------------------------------------------------------------------------

/// Vertices per parallel chunk.
///
/// A fixed [`Plan`] gives up load balance (`par/plan.rs:12-19`) and the
/// counting work is ragged -- an intersection costs `O(d_u + d_v)` -- so the
/// grain is small enough that a high-degree row cannot monopolise a chunk,
/// and large enough that a chunk is not one cache line of work.
const GRAIN: usize = 256;

/// The seed is structurally required by [`det_reduce`] and structurally
/// unused: these folds are over `u64`, which is associative, and no chunk
/// draws a random number. It is named rather than inlined so that the fact is
/// stated once.
const NO_RANDOMNESS: Seed = Seed([0; 32]);

/// A sorted adjacency list, one row per vertex index in the view's
/// [`vertex_bound`](gt_core::graph::GraphBase::vertex_bound).
///
/// Two allocations, not one per vertex: rows live end to end in `data` and
/// `start` holds the `n + 1` offsets, so a row is a `&[VertexId]` and the
/// intersection kernel below never touches a pointer it did not just read.
struct Csr {
    /// `n + 1` offsets; row `v` is `data[start[v]..start[v + 1]]`.
    start: Vec<usize>,
    /// Row contents, each row sorted ascending.
    data: Vec<VertexId>,
}

impl Csr {
    /// Row `v`, sorted ascending.
    #[inline]
    fn row(&self, v: usize) -> &[VertexId] {
        &self.data[self.start[v]..self.start[v + 1]]
    }

    /// Number of rows.
    #[inline]
    fn len(&self) -> usize {
        self.start.len() - 1
    }

    /// Turn a degree histogram in `start[1..]` into offsets, and allocate.
    ///
    /// Returns the fill cursor, one per row. Filling from a cursor rather than
    /// from `start` itself is what lets the rows be written in whatever order
    /// the edge enumeration produces.
    fn allocate(mut start: Vec<usize>) -> (Self, Vec<usize>) {
        let mut acc = 0usize;
        for s in &mut start {
            acc += *s;
            *s = acc;
        }
        // `start` now holds the *end* of each row shifted by one slot, which
        // is exactly the exclusive prefix sum: `start[0] == 0` because the
        // histogram was written into `start[1..]`.
        let cursor = start[..start.len() - 1].to_vec();
        let csr = Csr {
            data: vec![VertexId::from_index(0); acc],
            start,
        };
        (csr, cursor)
    }

    /// Sort every row, drop duplicates, and close the gaps.
    ///
    /// The compaction is in place and always writes behind the read head
    /// (`w <= s` at every step, because nothing ever grows), so one buffer
    /// serves for both.
    fn simplify(&mut self) {
        let n = self.len();
        let mut w = 0usize;
        for v in 0..n {
            let s = self.start[v];
            let e = self.start[v + 1];
            let row = &mut self.data[s..e];
            row.sort_unstable();
            // `Vec::dedup` would need a row-shaped `Vec`; this is the same
            // stable partition, written in place over the shared buffer.
            let mut kept = 0usize;
            for k in 0..row.len() {
                if kept == 0 || row[kept - 1] != row[k] {
                    row[kept] = row[k];
                    kept += 1;
                }
            }
            self.data.copy_within(s..s + kept, w);
            self.start[v] = w;
            w += kept;
        }
        self.start[n] = w;
        self.data.truncate(w);
    }

    /// Sort every row, keeping duplicates.
    fn sort_rows(&mut self) {
        for v in 0..self.len() {
            let s = self.start[v];
            let e = self.start[v + 1];
            self.data[s..e].sort_unstable();
        }
    }
}

/// The simple undirected adjacency underlying `g`, built from its **edge
/// set**.
///
/// Self-loops are dropped (`graph_clustering.hh:60`) and parallel edges are
/// collapsed. Two passes over `edges()`, which is `Clone` on every view.
fn simple_adjacency_from_edges<G: EdgeList>(g: G) -> Csr {
    let n = g.vertex_bound().len();
    let mut start = vec![0usize; n + 1];
    for e in g.edges() {
        let (s, t) = (e.source().index(), e.target().index());
        if s != t {
            start[s + 1] += 1;
            start[t + 1] += 1;
        }
    }
    let (mut csr, mut cursor) = Csr::allocate(start);
    for e in g.edges() {
        let (s, t) = (e.source().index(), e.target().index());
        if s != t {
            csr.data[cursor[s]] = e.target();
            cursor[s] += 1;
            csr.data[cursor[t]] = e.source();
            cursor[t] += 1;
        }
    }
    csr.simplify();
    csr
}

/// The simple undirected adjacency underlying `g`, built from its
/// **incidence**.
///
/// The same structure as [`simple_adjacency_from_edges`] for a kernel that has
/// [`VertexList`] but not [`EdgeList`]. Each row is filled from its own
/// vertex's `all_edges`, so symmetry follows from the storage rather than from
/// this function pushing both directions -- and a filtered view whose excluded
/// vertices are absent from `vertices()` simply leaves those rows empty.
///
/// A self-loop occupies a slot in each half of the adjacency block and is
/// therefore yielded *twice* by `all_edges` (`graph_adjacency.hh:1069-1073`);
/// the `other != v` test drops both copies.
fn simple_adjacency_from_incidence<G: GraphRef + VertexList>(g: G) -> Csr {
    let n = g.vertex_bound().len();
    let mut start = vec![0usize; n + 1];
    for v in g.vertices() {
        let mut d = 0usize;
        for i in g.all_edges(v) {
            d += usize::from(i.other != v);
        }
        start[v.index() + 1] = d;
    }
    let (mut csr, mut cursor) = Csr::allocate(start);
    for v in g.vertices() {
        let vi = v.index();
        for i in g.all_edges(v) {
            if i.other != v {
                csr.data[cursor[vi]] = i.other;
                cursor[vi] += 1;
            }
        }
    }
    csr.simplify();
    csr
}

/// Number of elements common to two ascending slices.
///
/// A merge, not a `contains` loop: both rows are already sorted, so the cost
/// is `O(|a| + |b|)` with no allocation and no hashing anywhere in the kernel.
#[inline]
fn intersection_len(a: &[VertexId], b: &[VertexId]) -> u64 {
    let (mut i, mut j, mut c) = (0usize, 0usize, 0u64);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                c += 1;
                i += 1;
                j += 1;
            }
        }
    }
    c
}

/// Triangles witnessed at `u`, plus the connected triples centred on `u`.
///
/// The witness is the unique ordering `u < v < w`, so a triangle is counted
/// **once** over the whole vertex range -- where `get_triangles` counts it
/// six times (twice per corner) and divides by two and then by three
/// (`graph_clustering.hh:92`, `:132`).
#[inline]
fn triangles_and_triples(csr: &Csr, rows: std::ops::Range<usize>) -> (u64, u64) {
    let (mut tri, mut triples) = (0u64, 0u64);
    for u in rows {
        let au = csr.row(u);
        // `(k * k - k2) / 2` at `graph_clustering.hh:92` with unit weights is
        // `C(d, 2)`, written here without the intermediate that overflows.
        let d = au.len() as u64;
        triples += d * d.saturating_sub(1) / 2;

        let uu = VertexId::from_index(u);
        let first = au.partition_point(|&x| x <= uu);
        for (k, &v) in au.iter().enumerate().skip(first) {
            let av = csr.row(v.index());
            // Both suffixes hold only vertices greater than `v`: `au[k + 1..]`
            // because `au` is sorted and `au[k] == v`, `av[hi..]` by search.
            let hi = av.partition_point(|&x| x <= v);
            tri += intersection_len(&au[k + 1..], &av[hi..]);
        }
    }
    (tri, triples)
}

/// [`triangles_and_triples`] over every row, folded in [`Plan`] order.
fn count_over_rows(csr: &Csr) -> (u64, u64) {
    let plan = Plan::new(csr.len(), GRAIN);
    det_reduce(
        plan,
        NO_RANDOMNESS,
        |rows, _rng| triangles_and_triples(csr, rows),
        (0, 0),
        |a, b| (a.0 + b.0, a.1 + b.1),
    )
}

// ---------------------------------------------------------------------------
// Depth-first machinery for the ordering kernels
// ---------------------------------------------------------------------------

/// Not yet reached.
const WHITE: u8 = 0;
/// On the DFS stack.
const GREY: u8 = 1;
/// Finished.
const BLACK: u8 = 2;

/// One iterative DFS over out-edges, recording finish order when asked.
///
/// `RECORD` is a const parameter rather than an `Option<&mut Vec<_>>` so that
/// [`is_dag`] and [`topological_sort`] are the *same* traversal -- the
/// acceptance condition for this unit is that they agree on every graph, and
/// two hand-written walks could drift. The predicate arm monomorphises with
/// the pushes removed.
///
/// Iterative, not recursive: `boost::depth_first_search` recurses, so
/// `topological_sort` on a path of a few hundred thousand vertices is a stack
/// overflow in graph-tool and a heap `Vec` here.
fn dfs_finish_order<G, const RECORD: bool>(g: G, order: &mut Vec<VertexId>) -> bool
where
    G: GraphRef + VertexList,
{
    let mut colour = vec![WHITE; g.vertex_bound().len()];
    // `(vertex, its unconsumed out-edges)`: the explicit form of the recursion
    // variable. `G::Out` is `Clone` but never cloned here; it is moved in.
    let mut stack: Vec<(VertexId, G::Out)> = Vec::new();

    for s in g.vertices() {
        if colour[s.index()] != WHITE {
            continue;
        }
        colour[s.index()] = GREY;
        stack.push((s, g.out_edges(s)));

        while !stack.is_empty() {
            // The top frame is read, advanced and released before the body
            // touches `stack` again: the shorter
            // `while let Some((v, edges)) = stack.last_mut()` holds the borrow
            // across the whole body, so pushing a successor inside it is
            // `error[E0499]` -- the borrow checker saying what the recursive
            // C++ formulation leaves to convention.
            let top = stack.len() - 1;
            let (v, edges) = &mut stack[top];
            let v = *v;
            let step = edges.next();

            match step {
                Some(i) => match colour[i.other.index()] {
                    WHITE => {
                        colour[i.other.index()] = GREY;
                        stack.push((i.other, g.out_edges(i.other)));
                    }
                    // A back edge, which `boost::topo_sort_visitor` reports by
                    // throwing `not_a_dag` (caught at
                    // `graph_topological_sort.cc:51`). A self-loop is a back
                    // edge to a grey vertex too, so it is a cycle here exactly
                    // as it is there.
                    GREY => return false,
                    _ => {}
                },
                None => {
                    colour[v.index()] = BLACK;
                    if RECORD {
                        order.push(v);
                    }
                    stack.pop();
                }
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// The kernels
// ---------------------------------------------------------------------------

/// Whether the directed view is acyclic.
///
/// Exactly `topological_sort(g).is_some()`, and shares its traversal: both are
/// [`dfs_finish_order`], which is the only place a back edge is recognised.
/// graph-tool reaches the same answer through an exception
/// (`graph_topological_sort.cc:38-55`), so `is_DAG` and `topological_sort`
/// there are one C++ entry point returning a `bool` and filling an out
/// parameter.
///
/// A self-loop is a cycle. An undirected view is a DAG only when it has no
/// edges at all, because every incidence is then its own back edge -- which is
/// the honest answer to a question that was asked of the wrong view, not a
/// silent zero.
pub fn is_dag<G>(g: G) -> bool
where
    G: GraphRef + VertexList,
{
    let mut sink = Vec::new();
    dfs_finish_order::<G, false>(g, &mut sink)
}

/// A topological order, or `None` if the view has a cycle.
///
/// Returned **in sort order**: `u` precedes `v` for every arc `u -> v`. The
/// C++ writes DFS finish order through a `back_inserter`
/// (`graph_topological_sort.cc:34`), which is the *reverse* topological order,
/// and the Python wrapper repairs it with `topological_order.a[::-1]`
/// (`topology/__init__.py`). One reversal, here, at the only place that knows
/// the convention.
///
/// `None` rather than graph-tool's `ValueError`: the caller cannot forget to
/// check it, and there is no second entry point that returns a half-filled
/// vector alongside a `false`.
pub fn topological_sort<G>(g: G) -> Option<Vec<VertexId>>
where
    G: GraphRef + VertexList,
{
    let mut order = Vec::with_capacity(g.num_vertices());
    if !dfs_finish_order::<G, true>(g, &mut order) {
        return None;
    }
    order.reverse();
    Some(order)
}

/// Whether the view is bipartite, and the two-colouring if so.
///
/// Direction-insensitive, walking `all_edges`: `graph_bipartite.cc:84`
/// dispatches under `{.tr=never_directed}` and the Python wrapper undirects
/// the graph first, so an orientation never reaches the algorithm.
///
/// The colouring uses graph-tool's convention: each search root is `1` and its
/// neighbours `0`, because `boost::is_bipartite` paints the start vertex
/// `white` (`bipartite.hpp`, `put_property(..., white, on_start_vertex())`)
/// and `graph_bipartite.cc:51` writes `part[v] == white` into the property
/// map. A DFS and a BFS agree on this map -- inside a component the colour is
/// the parity of the distance to the root, whatever order the vertices were
/// reached in -- so the queue below reproduces the C++ map exactly.
///
/// Indices outside the view (a filtered-out vertex) are left `0`. A self-loop
/// makes a graph non-bipartite, as it does in boost, where it is a back edge
/// whose endpoints are one vertex.
///
/// `None` rather than a filled map: boost documents that "if the graph is not
/// bipartite, the contents of the color map are undefined", which the Python
/// wrapper patches up after the fact with `part.a = 0`. An undefined map is
/// not a value this signature can return.
pub fn is_bipartite<G>(g: G) -> Option<Vec<u8>>
where
    G: GraphRef + VertexList,
{
    /// Not yet coloured. Erased to `0` before the map is returned.
    const UNSEEN: u8 = 2;

    let mut colour = vec![UNSEEN; g.vertex_bound().len()];
    let mut queue: Vec<VertexId> = Vec::new();

    for s in g.vertices() {
        if colour[s.index()] != UNSEEN {
            continue;
        }
        // White, i.e. `1` in graph-tool's partition map.
        colour[s.index()] = 1;
        queue.push(s);

        while let Some(v) = queue.pop() {
            let c = colour[v.index()];
            for i in g.all_edges(v) {
                let u = i.other.index();
                if colour[u] == UNSEEN {
                    colour[u] = 1 - c;
                    queue.push(i.other);
                } else if colour[u] == c {
                    return None;
                }
            }
        }
    }

    for c in &mut colour {
        if *c == UNSEEN {
            *c = 0;
        }
    }
    Some(colour)
}

/// Global clustering coefficient (transitivity).
///
/// `c = 3 · triangles / triples`, the definition at
/// `clustering/__init__.py:190-193`, over the simple undirected graph
/// underlying this view -- which is what `global_clustering` computes in
/// graph-tool too, because the Python wrapper replaces a directed graph with
/// `GraphView(g, directed=False)` before dispatching
/// (`clustering/__init__.py:226-227`). So this returns the same number on
/// `&g` and on `g.undirect()`, and both match the C++.
///
/// `triples` is `Σ_v C(d_v, 2)`, which is `(k² - k2) / 2` at
/// `graph_clustering.hh:92` with unit weights. Vertices of degree below two
/// contribute nothing, matching the `out_degree(v, g) > 1` guard at `:56`.
///
/// Returns `f64::NAN` for a graph with no connected triple, which is what
/// `double(triangles) / n` yields at `graph_clustering.hh:116` when `n == 0`.
/// Zero would be a claim -- "this graph has no transitivity" -- that an empty
/// graph does not support.
///
/// The jackknife error term (`:118-131`) is not ported: it is a second
/// statistic, not part of the coefficient, and it belongs with the counts a
/// caller asks for explicitly.
pub fn global_clustering<G>(g: G) -> f64
where
    G: GraphRef + VertexList + Sync,
{
    let csr = simple_adjacency_from_incidence(g);
    let (triangles, triples) = count_over_rows(&csr);
    3.0 * triangles as f64 / triples as f64
}

/// Number of triangles, counted once each.
///
/// Bounded on [`EdgeList`] because the count is per *edge*, not per incidence:
/// summing over out-edges of an undirected view double-counts, which is the
/// same arithmetic error that makes a filtered undirected `num_edges` wrong.
///
/// A triangle is a set of three distinct, pairwise adjacent vertices of the
/// simple graph underlying this view: parallel edges do not multiply it and
/// self-loops do not create one. Each is counted at its unique `u < v < w`
/// witness, so the answer is the same on a directed view and on its
/// [`undirect`](gt_core::view::Undirect)ed counterpart, and no division
/// repairs a multiplicity afterwards.
///
/// graph-tool exposes no directed triangle count at all -- `global_clustering`
/// undirects first -- so the directed arm of `get_triangles`
/// (`graph_clustering.hh:89-92`) counts *transitive triples* `v→u, v→w, u→w`,
/// of which a three-cycle `v→u→w→v` contributes none. That quantity is not a
/// triangle count and `triangles / 3` at `:132` is not even an integer for it.
pub fn count_triangles<G>(g: G) -> u64
where
    G: GraphRef + EdgeList + Sync,
{
    let csr = simple_adjacency_from_edges(g);
    count_over_rows(&csr).0
}

/// Reciprocity: the fraction of edges whose reverse also exists.
///
/// `E↔ / E`, ported from `get_reciprocity`
/// (`topology/graph_reciprocity.cc:29-70`) with unit weights and the Python
/// default `self_loops=False` (`topology/__init__.py`, `edge_reciprocity`).
/// The C++ accumulates, per vertex `v` and per out-neighbour `u`,
/// `Lbd += min(m_uv, m_vu)` and `L += m_vu` (`:59-60`), so a parallel edge
/// counts towards `L` with its multiplicity and towards `Lbd` only as far as
/// the reverse multiplicity supports it. That `min` is reproduced exactly
/// here: run-lengths in one sorted adjacency row against an equal-range probe
/// in the other, rather than the `O(Σ_v d_v²)` rescan at `:54-58`.
///
/// Self-loops are excluded from both numerator and denominator (`:46-47`).
/// A graph with no non-self-loop edge returns `f64::NAN`, as `Lbd / double(L)`
/// does at `:69`.
///
/// Bounded on [`Bidirectional`], so it cannot be asked of an undirected view,
/// where the answer is `1` by construction and the question is a mistake.
pub fn reciprocity<G>(g: G) -> f64
where
    G: Bidirectional + EdgeList,
{
    let n = g.vertex_bound().len();
    let mut start = vec![0usize; n + 1];
    for e in g.edges() {
        if e.source() != e.target() {
            start[e.source().index() + 1] += 1;
        }
    }
    let (mut out, mut cursor) = Csr::allocate(start);
    for e in g.edges() {
        if e.source() != e.target() {
            let s = e.source().index();
            out.data[cursor[s]] = e.target();
            cursor[s] += 1;
        }
    }
    // Sorted but *not* deduplicated: the multiplicities are the whole point of
    // the `min`.
    out.sort_rows();

    let (mut reciprocated, mut total) = (0u64, 0u64);
    for s in 0..n {
        let row = out.row(s);
        total += row.len() as u64;
        let sv = VertexId::from_index(s);
        let mut i = 0usize;
        while i < row.len() {
            let t = row[i];
            let j = i + row[i..].partition_point(|&x| x == t);
            let forward = (j - i) as u64;
            let back = out.row(t.index());
            let lo = back.partition_point(|&x| x < sv);
            let hi = back.partition_point(|&x| x <= sv);
            reciprocated += forward.min((hi - lo) as u64);
            i = j;
        }
    }

    reciprocated as f64 / total as f64
}

// ---------------------------------------------------------------------------

/// The core number of every vertex: Batagelj--Zaversnik bin sort.
///
/// Ports `kcore_decomposition` (`topology/graph_kcore.hh:25-80`), which
/// `do_kcore_decomposition` (`graph_kcore.cc:31-40`) dispatches over
/// `all_graph_views`, so the same body runs on a directed graph, an undirected
/// one and any filtered view of either.
///
/// ## The two incidence choices, which are the whole of the semantics
///
/// The C++ reads `degree(v, g)` (`:41`) and `all_neighbors_range(v, g)`
/// (`:62`). For `adj_list` both are the *whole* block
/// (`graph_adjacency.hh:1068-1072`, `:1163-1170`), so:
///
/// * on a directed graph the degree is the total in + out degree --- which the
///   Python docstring states outright (`topology/__init__.py:1832-1833`);
/// * a **self-loop contributes two**, because it occupies both halves of its
///   vertex's block, and "these edges contribute to the degree in the usual
///   fashion" (`:1835-1836`);
/// * a parallel edge contributes its multiplicity.
///
/// This port keeps those reads --- [`GraphRef::degree`] and
/// [`GraphRef::all_edges`] --- so `kcore_decomposition(&g)` and
/// `kcore_decomposition(g.undirect())` are the same computation, exactly as
/// the C++'s single body is.
///
/// ## The bookkeeping
///
/// `bins[k]` holds the vertices whose *remaining* degree is `k`, and `pos[v]`
/// is `v`'s index inside its bin, so removing a vertex from the middle of a
/// bin is the swap-with-back at `:66-71`. Vertices are drained from the
/// smallest bin upwards; peeling `v` out of bin `k` lowers each still-present
/// neighbour whose remaining degree exceeds `deg[v]` by one and moves it down
/// a bin. `deg[v]` itself is never lowered after `v` is popped, which is what
/// makes the `ku > deg[v]` test at `:64` mean "u has not been peeled yet".
///
/// Returns the largest core number, i.e. the degeneracy of the graph.
///
/// # Panics
///
/// If `core` was minted for a different graph than `g`.
pub fn kcore_decomposition<G>(
    g: G,
    core: &mut gt_core::prop::DenseProp<i64, gt_core::ids::VertexTag>,
) -> usize
where
    G: GraphRef + VertexList,
{
    let bound = g.vertex_bound();
    let mut view = core
        .sized_for(bound)
        .expect("the core map belongs to a different graph");
    let out = view.as_mut_slice();
    // A vertex the view does not expose is never binned and never popped, so
    // it would otherwise keep whatever the map held. `components::UNLABELED`
    // is the same convention; the C++ leaves the map's previous contents.
    out.fill(crate::components::UNLABELED);

    let n = bound.len();
    let mut deg = vec![0usize; n];
    let mut pos = vec![0usize; n];
    // `bins` is sized by the maximum degree, as `:46-47` sizes it.
    let mut bins: Vec<Vec<VertexId>> = Vec::new();
    for v in g.vertices() {
        let k = g.degree(v);
        deg[v.index()] = k;
        if k >= bins.len() {
            bins.resize(k + 1, Vec::new());
        }
        bins[k].push(v);
        pos[v.index()] = bins[k].len() - 1;
    }

    let mut max_core = 0usize;
    for k in 0..bins.len() {
        // `while (!bins_k.empty()) { v = bins_k.back(); bins_k.pop_back(); }`
        // (`:56-59`). The bin can *grow* while it is being drained --- a
        // neighbour demoted into it --- which is why this is a `while` and not
        // an iteration over a snapshot.
        while let Some(v) = bins[k].pop() {
            out[v.index()] = k as i64;
            max_core = max_core.max(k);
            let dv = deg[v.index()];
            for inc in g.all_edges(v) {
                let u = inc.other;
                let ku = deg[u.index()];
                // `if (ku > deg[v])` (`:64`): a peeled neighbour has a
                // remaining degree at most `deg[v]`, and so does a self-loop's
                // `u == v`, so neither is demoted.
                if ku <= dv {
                    continue;
                }
                // Swap `u` out of the middle of `bins[ku]` (`:66-71`).
                let w = *bins[ku].last().expect("u is in this bin");
                let pw = pos[u.index()];
                pos[w.index()] = pw;
                bins[ku][pw] = w;
                bins[ku].pop();
                deg[u.index()] = ku - 1;
                bins[ku - 1].push(u);
                pos[u.index()] = bins[ku - 1].len() - 1;
            }
        }
    }

    max_core
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Unit-local checks of the pieces the public kernels are assembled from.
    //! The behavioural acceptance list lives in `tests/u17_topology.rs`.

    use super::*;

    fn csr(rows: &[&[usize]]) -> Csr {
        let n = rows.len();
        let mut start = vec![0usize; n + 1];
        for (v, r) in rows.iter().enumerate() {
            start[v + 1] = r.len();
        }
        let (mut c, mut cursor) = Csr::allocate(start);
        for (v, r) in rows.iter().enumerate() {
            for &u in *r {
                c.data[cursor[v]] = VertexId::from_index(u);
                cursor[v] += 1;
            }
        }
        c
    }

    #[test]
    fn simplify_sorts_dedups_and_closes_the_gaps() {
        let mut c = csr(&[&[3, 1, 1, 3], &[], &[2, 0, 2], &[0]]);
        c.simplify();
        let got: Vec<Vec<usize>> = (0..c.len())
            .map(|v| c.row(v).iter().map(|x| x.index()).collect())
            .collect();
        assert_eq!(got, vec![vec![1, 3], vec![], vec![0, 2], vec![0]]);
        // Compacted: no gap survives between rows.
        assert_eq!(c.data.len(), 5);
        assert_eq!(c.start, vec![0, 2, 2, 4, 5]);
    }

    #[test]
    fn intersection_is_a_merge_over_ascending_rows() {
        let a: Vec<VertexId> = [1, 4, 7, 9]
            .iter()
            .map(|&i| VertexId::from_index(i))
            .collect();
        let b: Vec<VertexId> = [0, 4, 5, 9, 11]
            .iter()
            .map(|&i| VertexId::from_index(i))
            .collect();
        assert_eq!(intersection_len(&a, &b), 2);
        assert_eq!(intersection_len(&a, &[]), 0);
        assert_eq!(intersection_len(&[], &b), 0);
    }

    /// The witness ordering, on the one graph where every wrong constant is
    /// visible: `K4` has four triangles, and the six-fold overcount that
    /// `graph_clustering.hh` divides away would report 24.
    #[test]
    fn each_triangle_is_witnessed_once() {
        let mut c = csr(&[&[1, 2, 3], &[0, 2, 3], &[0, 1, 3], &[0, 1, 2]]);
        c.simplify();
        let (tri, triples) = triangles_and_triples(&c, 0..c.len());
        assert_eq!(tri, 4);
        assert_eq!(triples, 4 * 3);
        // Every row sees exactly one triangle as `u < v < w` except the last
        // two, which have no room for a `w`.
        assert_eq!(triangles_and_triples(&c, 0..1).0, 3);
        assert_eq!(triangles_and_triples(&c, 1..2).0, 1);
        assert_eq!(triangles_and_triples(&c, 2..4).0, 0);
    }

    /// Chunking is an exact cover, so a partitioned fold is the whole fold --
    /// the property [`Plan`] exists to guarantee.
    #[test]
    fn the_chunked_fold_equals_the_whole_fold() {
        let mut c = csr(&[&[1, 2, 3, 4], &[0, 2], &[0, 1, 3], &[0, 2, 4], &[0, 3], &[]]);
        c.simplify();
        let whole = triangles_and_triples(&c, 0..c.len());
        let plan = Plan::new(c.len(), 2);
        let mut parts = (0u64, 0u64);
        for k in 0..plan.chunks() {
            let p = triangles_and_triples(&c, plan.range(k));
            parts = (parts.0 + p.0, parts.1 + p.1);
        }
        assert_eq!(whole, parts);
        assert_eq!(whole.0, 3);
    }
}
