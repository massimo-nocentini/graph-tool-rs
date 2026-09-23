//! Breadth- and depth-first traversal.
//!
//! ## The visitor contract, stated once
//!
//! `breadth_first_visit` (boost/graph/breadth_first_search.hpp) fires nine
//! events; graph-tool's `BFSVisitorWrapper` (`search/graph_bfs.cc:41-105`)
//! forwards all nine across the Python boundary. Three of them —
//! `discover_vertex`, `examine_edge`, `finish_vertex` — carry the whole of
//! what a traversal is; the other six are re-derivable from the colour map a
//! caller does not have, and each is one more `_vis.attr(...)` lookup per
//! event. [`Visitor`] therefore has exactly the three, and states their
//! order:
//!
//! 1. [`discover`](Visitor::discover) fires once per vertex, at the moment it
//!    is coloured, *before* it is expanded. The source fires it at `depth 0`,
//!    exactly as `breadth_first_visit` calls `discover_vertex(s)` before
//!    entering its loop.
//! 2. [`examine`](Visitor::examine) fires for **every** incidence of an
//!    expanded vertex, tree edge or not — the `examine_edge` position, which
//!    precedes the colour test.
//! 3. [`finish`](Visitor::finish) fires when the last incidence of a vertex
//!    has been examined.
//!
//! [`Control`] is the part boost does not have: its visitors terminate by
//! throwing (`topology/graph_distance.cc:42` defines `struct stop_search {}`
//! and `:62`, `:69`, `:113` throw it from inside the kernel, caught at
//! `:314`). A `throw` from a visitor leaves the algorithm's own state
//! half-updated and is caught by *value* across a template boundary, so a
//! second visitor throwing a second exception type is silently uncaught. Here
//! the traversal asks and the visitor answers:
//!
//! * [`Control::Prune`] from `discover` — the vertex is discovered and never
//!   expanded. It is therefore never finished: `finish(v)` and "the
//!   neighbours of `v` were examined" are the same statement.
//! * [`Control::Prune`] from `examine` — that one incidence is not followed;
//!   the neighbour stays undiscovered unless some other edge reaches it.
//! * [`Control::Stop`] from either — the traversal returns `Ok(())`
//!   immediately, with no `finish` for the vertex in flight. Stopping is a
//!   *result*, not a failure, which is why it is not an `Err`.
//!
//! ## Reachability, not membership
//!
//! A traversal checks its source against
//! [`vertex_bound`](gt_core::graph::GraphBase::vertex_bound) — the allocation
//! bound, the number that sizes the colour array — and answers
//! [`GraphError::NoSuchVertex`](gt_core::GraphError::NoSuchVertex) outside it.
//! It does *not* check that a filtered view keeps the source:
//! `Filtered::out_edges` filters the incidences of its anchor without
//! re-testing the anchor (`view/filtered.rs`, `keeps_incident`), and
//! `filt_graph` behaves the same way. Testing it would cost an O(V) scan per
//! call on the one view that cannot answer it in O(1).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use gt_core::adj::Incident;
use gt_core::graph::{GraphRef, VertexList};
use gt_core::ids::VertexId;
use gt_core::prop::{DenseProp, ReadProp};

/// What a visitor asks the traversal to do next.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Control {
    /// Keep going.
    Continue,
    /// Do not expand this vertex's neighbours.
    Prune,
    /// Stop the whole traversal.
    Stop,
}

/// Callbacks for a traversal.
///
/// Every method has a default, but each default is expressible purely in terms
/// of *doing nothing*, which is the rule this port applies:
///
/// > a trait method may carry a default body only if that body is either a
/// > pure no-op or derivable from other **required** methods of the same
/// > trait. A default that encodes policy is not a default; it is a required
/// > method.
///
/// `MetropolisStateBase` (`loops/mcmc_loop.hh:39-56`) breaks that rule
/// spectacularly: it labels five members `// required` and gives each one a
/// *working* body, so a state that misspells `virtual_move` compiles and
/// reports `dS = 0` for every move.
pub trait Visitor {
    /// Called once per newly discovered vertex.
    fn discover(&mut self, v: VertexId, depth: usize) -> Control {
        // `let _ = ...` rather than `_v`/`_depth`: the parameter names are
        // documentation, and this body stays a no-op once the crate-level
        // `allow(unused_variables)` comes off in U-FINAL.
        let _ = (v, depth);
        Control::Continue
    }
    /// Called for each examined incidence.
    fn examine(&mut self, from: VertexId, e: Incident) -> Control {
        let _ = (from, e);
        Control::Continue
    }
    /// Called when a vertex's neighbours have all been examined.
    fn finish(&mut self, v: VertexId) {
        let _ = v;
    }
}

/// Breadth-first search from one source.
pub fn bfs<G, V>(g: G, source: VertexId, vis: &mut V) -> Result<(), gt_core::GraphError>
where
    G: GraphRef,
    V: Visitor,
{
    bfs_from(g, std::iter::once(source), vis)
}

/// Breadth-first search from several sources at once.
///
/// Every source is at `depth 0` and the frontiers advance together, so the
/// depth a vertex is discovered at is its distance to the *nearest* source.
/// That is not what `do_bfs`'s null-source branch (`search/graph_bfs.cc:116-124`)
/// does: it runs one `breadth_first_visit` per uncovered vertex over a shared
/// colour map, so each root restarts the depth count and the result is a BFS
/// *forest*, not a multi-source BFS. A forest is [`bfs`] in a loop over
/// [`vertices`](gt_core::graph::VertexList::vertices); this is the other thing.
///
/// Sources are consumed lazily: a bad descriptor half-way through the sequence
/// is reported after the sources before it have already been discovered.
pub fn bfs_multi<G, V, I>(g: G, sources: I, vis: &mut V) -> Result<(), gt_core::GraphError>
where
    G: GraphRef,
    V: Visitor,
    I: IntoIterator<Item = VertexId>,
{
    bfs_from(g, sources, vis)
}

/// The one BFS. [`bfs`] is this with a one-element source sequence, which
/// monomorphises to the same loop over `std::iter::Once`.
fn bfs_from<G, V, I>(g: G, sources: I, vis: &mut V) -> Result<(), gt_core::GraphError>
where
    G: GraphRef,
    V: Visitor,
    I: IntoIterator<Item = VertexId>,
{
    let n = g.vertex_bound().len();
    let mut seen = vec![false; n];

    // Level-synchronous, not a `VecDeque<(VertexId, usize)>`: a FIFO queue and
    // a pair of frontiers visit in the same order, and the frontier form keeps
    // the depth out of the queue element (8 bytes per entry against 16) and
    // out of the loop body.
    let mut frontier: Vec<VertexId> = Vec::new();
    let mut next: Vec<VertexId> = Vec::new();

    for s in sources {
        if s.index() >= n {
            return Err(gt_core::GraphError::NoSuchVertex(s));
        }
        if seen[s.index()] {
            continue;
        }
        seen[s.index()] = true;
        match vis.discover(s, 0) {
            Control::Stop => return Ok(()),
            Control::Prune => continue,
            Control::Continue => frontier.push(s),
        }
    }

    let mut depth = 0usize;
    while !frontier.is_empty() {
        for &u in &frontier {
            for inc in g.out_edges(u) {
                match vis.examine(u, inc) {
                    Control::Stop => return Ok(()),
                    Control::Prune => continue,
                    Control::Continue => {}
                }
                let w = inc.other.index();
                if seen[w] {
                    continue;
                }
                seen[w] = true;
                match vis.discover(inc.other, depth + 1) {
                    Control::Stop => return Ok(()),
                    Control::Prune => {}
                    Control::Continue => next.push(inc.other),
                }
            }
            vis.finish(u);
        }
        std::mem::swap(&mut frontier, &mut next);
        next.clear();
        depth += 1;
    }
    Ok(())
}

/// Depth-first search from one source.
///
/// The frontier stores `G::Out` values. Under the GAT trait shape this needs
/// an explicit `G: 'a` that then propagates into every algorithm struct in the
/// port (`error[E0309]: the parameter type G may not live long enough ...
/// required by this bound: where Self: 'a`). With the lifetime on the
/// implementing type, `G::Out` is an ordinary type and the struct is ordinary.
///
/// Keeping the *iterator* on the stack rather than a re-derived index is what
/// makes [`Visitor::finish`] a true post-order: `depth_first_visit`'s stack
/// holds `(u, ei, ei_end)` for exactly this reason. Re-calling `out_edges(u)`
/// on resume would be O(degree) per edge on a filtered view, where the
/// iterator is a predicate chain and not a slice cursor.
pub fn dfs<G, V>(g: G, source: VertexId, vis: &mut V) -> Result<(), gt_core::GraphError>
where
    G: GraphRef,
    V: Visitor,
{
    let n = g.vertex_bound().len();
    if source.index() >= n {
        return Err(gt_core::GraphError::NoSuchVertex(source));
    }
    let mut seen = vec![false; n];
    let mut stack: Vec<(VertexId, G::Out)> = Vec::new();

    seen[source.index()] = true;
    match vis.discover(source, 0) {
        Control::Stop | Control::Prune => return Ok(()),
        Control::Continue => stack.push((source, g.out_edges(source))),
    }

    // `stack[i]` is the vertex at depth `i`, so a newly discovered vertex is
    // at `stack.len()` — no per-vertex depth field, and no second map.
    while let Some(top) = stack.last_mut() {
        let u = top.0;
        let Some(inc) = top.1.next() else {
            vis.finish(u);
            stack.pop();
            continue;
        };
        match vis.examine(u, inc) {
            Control::Stop => return Ok(()),
            Control::Prune => continue,
            Control::Continue => {}
        }
        let w = inc.other;
        if seen[w.index()] {
            continue;
        }
        seen[w.index()] = true;
        match vis.discover(w, stack.len()) {
            Control::Stop => return Ok(()),
            Control::Prune => {}
            Control::Continue => stack.push((w, g.out_edges(w))),
        }
    }
    Ok(())
}

/// The value an unreached vertex carries in an integer distance map.
///
/// `numeric_limits<dist_t>::max()` for an integral `dist_t`
/// (`topology/graph_distance.cc:277-279`), which the Python layer writes over
/// the whole array before the search (`topology/__init__.py:2097-2098`:
/// `dist_map.set_value(numpy.iinfo(dist_map.a.dtype).max)`).
pub const UNREACHABLE: i64 = i64::MAX;

/// Unweighted shortest-path distances from one source.
///
/// The distance map is **sized for the graph**, not for the number of
/// reachable vertices; that is the distinction
/// [`vertex_bound`](gt_core::graph::GraphBase::vertex_bound) exists to make.
///
/// Every slot in the bound is set: the source to `0`, a reached vertex to its
/// hop count, everything else to [`UNREACHABLE`].
///
/// # Panics
///
/// If `dist` belongs to another graph. That is the defect-#8 comparison, made
/// once at kernel entry by [`DenseProp::sized_for`]; it is a misuse of the API
/// rather than a property of the input, and the rest of gt-algo's map-taking
/// kernels ([`components`](crate::components::components),
/// [`pagerank`](crate::centrality::pagerank)) do not return a `Result` at all.
/// The `Result` here is about the *source descriptor*.
pub fn shortest_distances<G>(
    g: G,
    source: VertexId,
    dist: &mut DenseProp<i64, gt_core::ids::VertexTag>,
) -> Result<(), gt_core::GraphError>
where
    G: GraphRef + VertexList,
{
    let bound = g.vertex_bound();
    if source.index() >= bound.len() {
        return Err(gt_core::GraphError::NoSuchVertex(source));
    }
    let mut view = dist
        .sized_for(bound)
        .expect("distance map must belong to this graph");
    let d = view.as_mut_slice();
    d.fill(UNREACHABLE);

    // The depth `discover` is handed *is* the distance, so there is no
    // `_dist_map[u] + 1` and no second read of the map per tree edge
    // (`topology/graph_distance.cc:60`).
    struct Dist<'a> {
        d: &'a mut [i64],
    }
    impl Visitor for Dist<'_> {
        #[inline]
        fn discover(&mut self, v: VertexId, depth: usize) -> Control {
            self.d[v.index()] = depth as i64;
            Control::Continue
        }
    }

    bfs(g, source, &mut Dist { d })
}

/// Weighted shortest-path distances (Dijkstra).
///
/// Generic over the weight map, so a [`Unity`](gt_core::prop::dense::Unity)
/// weight routes into the unweighted fast path at monomorphisation.
///
/// Unreached vertices carry `f64::INFINITY`, which is the floating-point half
/// of `topology/graph_distance.cc:333-335`'s `inf`.
///
/// Weights must be non-negative, as Dijkstra's requires; graph-tool routes the
/// negative case to `bellman_ford_shortest_paths` (`:415-418`) rather than
/// letting `dijkstra_shortest_paths_no_color_map_no_init` produce a wrong
/// answer, and so should a caller here.
///
/// # Panics
///
/// If `dist` belongs to another graph; see [`shortest_distances`].
pub fn dijkstra<G, W>(
    g: G,
    source: VertexId,
    weight: &W,
    dist: &mut DenseProp<f64, gt_core::ids::VertexTag>,
) -> Result<(), gt_core::GraphError>
where
    G: GraphRef + VertexList,
    W: ReadProp<gt_core::ids::EdgeTag, Value = f64>,
{
    let bound = g.vertex_bound();
    if source.index() >= bound.len() {
        return Err(gt_core::GraphError::NoSuchVertex(source));
    }
    let mut view = dist
        .sized_for(bound)
        .expect("distance map must belong to this graph");
    let d = view.as_mut_slice();
    d.fill(f64::INFINITY);

    if W::IS_UNITY {
        // `is_unity_map_v` (`graph_properties.hh:729`) as a `const`: with unit
        // weights the priority queue is a FIFO queue, so this arm *is* the
        // BFS, and the heap arm below is dead code in this instantiation —
        // the weight map is never read, which `tests/u13_traversal.rs`
        // asserts with a map whose `get_ref` panics.
        //
        // graph-tool cannot take this branch: the constant-weight overloads
        // that would (`graph_selectors.hh:109`, `:178`) read `weight.c` on a
        // class whose member is the private `_c` (`graph_properties.hh:677`),
        // so they are uninstantiable dead code.
        struct Hops<'a> {
            d: &'a mut [f64],
        }
        impl Visitor for Hops<'_> {
            #[inline]
            fn discover(&mut self, v: VertexId, depth: usize) -> Control {
                self.d[v.index()] = depth as f64;
                Control::Continue
            }
        }
        return bfs(g, source, &mut Hops { d });
    }

    d[source.index()] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(Step {
        key: 0.0,
        v: source,
    });

    // Lazy deletion rather than decrease-key, which is what
    // `dijkstra_shortest_paths_no_color_map_no_init` does: a stale entry is
    // recognised by its key and dropped.
    while let Some(Step { key, v: u }) = heap.pop() {
        if key > d[u.index()] {
            continue;
        }
        for inc in g.out_edges(u) {
            let w = *weight.get_ref(inc.edge);
            // `closed_plus<dist_t>` (`:333`) is saturating addition; on `f64`
            // the saturation is `inf` and comes for free, and `key` is finite
            // by construction because only finite keys are pushed.
            let alt = key + w;
            let t = inc.other.index();
            if alt < d[t] {
                d[t] = alt;
                heap.push(Step {
                    key: alt,
                    v: inc.other,
                });
            }
        }
    }
    Ok(())
}

/// One entry of Dijkstra's priority queue, ordered so that
/// [`BinaryHeap`] — a max-heap — pops the *smallest* key.
///
/// Ties break on the vertex id, so the pop order is a total order on the
/// entries and the traversal is reproducible; `std::less<dist_t>` on a heap of
/// equal keys is not.
#[derive(Clone, Copy, Debug)]
struct Step {
    key: f64,
    v: VertexId,
}

impl Ord for Step {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // `total_cmp`, not `partial_cmp().unwrap()`: a NaN weight must not
        // turn an ordering into a panic inside the kernel.
        other
            .key
            .total_cmp(&self.key)
            .then_with(|| other.v.cmp(&self.v))
    }
}
impl PartialOrd for Step {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Step {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Step {}

#[cfg(test)]
mod tests {
    //! The parts that are about *this file's* internals. The behavioural
    //! acceptance list is `tests/u13_traversal.rs`.

    use super::*;

    #[test]
    fn the_queue_entry_is_a_min_heap_with_a_deterministic_tie_break() {
        let mut h = BinaryHeap::new();
        for (key, i) in [(2.0, 0usize), (1.0, 5), (1.0, 3), (f64::INFINITY, 1)] {
            h.push(Step {
                key,
                v: VertexId::from_index(i),
            });
        }
        let got: Vec<(f64, usize)> = std::iter::from_fn(|| h.pop())
            .map(|s| (s.key, s.v.index()))
            .collect();
        assert_eq!(
            got,
            [(1.0, 3), (1.0, 5), (2.0, 0), (f64::INFINITY, 1)],
            "smallest key first, then smallest vertex"
        );
    }

    /// `total_cmp` rather than `partial_cmp().unwrap()`: a NaN key orders,
    /// rather than aborting the kernel from inside `BinaryHeap`.
    #[test]
    fn a_nan_key_orders_instead_of_panicking() {
        let a = Step {
            key: f64::NAN,
            v: VertexId::from_index(0),
        };
        let b = Step {
            key: 1.0,
            v: VertexId::from_index(1),
        };
        assert_ne!(a.cmp(&b), Ordering::Equal);
        assert_eq!(a.cmp(&b), b.cmp(&a).reverse());
    }
}
