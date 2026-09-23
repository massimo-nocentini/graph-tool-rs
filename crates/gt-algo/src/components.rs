//! Connected, weakly connected and strongly connected components.
//!
//! The port of `label_components` (`topology/graph_components.hh:102-129`),
//! which is a two-arm tag dispatch: `boost::strong_components` when
//! `directed_category` converts to `directed_tag`, `boost::connected_components`
//! otherwise. That single entry point is split in two here, and the split is
//! the point.
//!
//! `boost::connected_components` carries
//! `BOOST_STATIC_ASSERT((is_same<directed, undirected_tag>::value))`, so the
//! C++ can only reach it through the `std::false_type` arm --- and the tag it
//! branches on comes from `graph_traits<Graph>::directed_category`, i.e. from
//! the *view type*, not from what the caller meant. `GraphView(g,
//! directed=False)` (`__init__.py:1370`) is how a caller says "weak
//! components", and forgetting it silently buys the strong ones instead.
//!
//! Here [`components`] and [`strong_components`] are separate functions and
//! the second is bounded on [`Bidirectional`], which no undirected view
//! implements, so the two are not confusable in either direction: asking for
//! strong components of `g.undirect()` is `error[E0277]`, and asking for
//! connected components is `components(g.undirect())`, written out.
//!
//! ## Labels
//!
//! Both functions reproduce boost's numbering exactly, because
//! `graph_components.hh:110` wraps the caller's map in a
//! `HistogramPropertyMap` whose histogram is indexed *by the label*: the
//! numbering is observable from Python as `hist`, and
//! `label_largest_component` (`__init__.py:1431`) then does
//! `c.fa == h.argmax()`.
//!
//! * [`components`] numbers by root order: the component of the first
//!   unvisited vertex in [`VertexList::vertices`] order is 0, the next is 1.
//!   That is `components_recorder::start_vertex`, which increments the counter
//!   once per DFS root, with `discover_vertex` writing the current value.
//! * [`strong_components`] numbers in the order the SCC *roots finish*, which
//!   is `tarjan_scc_visitor::finish_vertex` incrementing `c_count` after
//!   popping a component off its stack. The observable consequence is that the
//!   labels are a reverse topological order of the condensation: for every
//!   edge `u -> v`, `label[u] >= label[v]`.
//!
//! ## The sentinel
//!
//! Both functions first fill the whole *bound* with [`UNLABELED`] and only
//! then label what the view exposes. On a filtered view that matters: the map
//! is sized for the unfiltered index space (it must be --- the labels are
//! indexed by unfiltered `VertexId`), while only the surviving vertices are
//! reachable from `vertices()`. C++ leaves the rest holding whatever the
//! property map's default was, which for a fresh `int32_t` map is `0` ---
//! indistinguishable from a genuine member of component 0. `label_components`
//! is documented to return labels "from 0 to N-1" (`graph_components.hh:99`)
//! and its histogram counts only the vertices it wrote, so the two disagree on
//! a filtered graph. Here a vertex the view does not expose reads `-1`.

use gt_core::graph::{Bidirectional, GraphRef, VertexList};
use gt_core::ids::{VertexId, VertexTag};
use gt_core::prop::DenseProp;

/// The value a vertex outside the view carries after a labelling run.
///
/// Distinct from every real label, which are `0..n`.
pub const UNLABELED: i64 = -1;

/// Panic message for the one failure `sized_for` can report here. The bound
/// comes from `g` itself, so the only way to reach it is to pass a map minted
/// for a *different* graph --- a caller error, not a runtime condition, and
/// the one `GraphId` comparison of [DESIGN](gt_core::design) D-`Bound` exists to catch it.
const WRONG_GRAPH: &str = "the label map belongs to a different graph";

/// Label each vertex with its component.
///
/// On an [`Und`](gt_core::view::Und) view this is the connected components; on
/// a directed view it is the *out*-reachability partition, which is usually
/// not what the caller means. Callers wanting weak components should pass
/// `g.undirect()` explicitly rather than relying on the algorithm to guess.
///
/// Returns the number of components. Vertices the view does not expose are set
/// to [`UNLABELED`].
///
/// # Panics
///
/// If `label` was minted for a different graph than `g`.
pub fn components<G>(g: G, label: &mut DenseProp<i64, VertexTag>) -> usize
where
    G: GraphRef + VertexList,
{
    let bound = g.vertex_bound();
    let mut view = label.sized_for(bound).expect(WRONG_GRAPH);
    let lab = view.as_mut_slice();
    lab.fill(UNLABELED);

    // One frontier for the whole run, so the per-root cost is amortised to
    // zero allocations after the first component: `clear` keeps the capacity.
    // The frontier holds `VertexId`s and not `G::Out` iterators because the
    // label of a vertex depends only on *which* root reached it, never on the
    // order the root's tree was walked in -- `components_recorder` changes its
    // counter in `start_vertex` alone. So a stack of 4-byte ids is exactly as
    // faithful as boost's recursive DFS and does not risk its stack depth.
    let mut frontier: Vec<VertexId> = Vec::new();
    let mut n_comp: i64 = 0;

    for root in g.vertices() {
        if lab[root.index()] != UNLABELED {
            continue;
        }
        lab[root.index()] = n_comp;
        frontier.clear();
        frontier.push(root);
        while let Some(v) = frontier.pop() {
            for i in g.out_edges(v) {
                let w = i.other.index();
                if lab[w] == UNLABELED {
                    lab[w] = n_comp;
                    frontier.push(i.other);
                }
            }
        }
        n_comp += 1;
    }

    n_comp as usize
}

/// Strongly connected components (Tarjan).
///
/// Bounded on [`Bidirectional`], so it cannot be called on an undirected view
/// at all -- where the answer would be the connected components under a name
/// that promises something else.
///
/// Returns the number of components. Vertices the view does not expose are set
/// to [`UNLABELED`].
///
/// Labels are assigned in the order SCC roots finish, exactly as
/// `boost::strong_components` assigns them, so they are a reverse topological
/// order of the condensation: `label[source] >= label[target]` for every edge,
/// with equality exactly when the two endpoints are mutually reachable.
///
/// # Panics
///
/// If `label` was minted for a different graph than `g`.
pub fn strong_components<G>(g: G, label: &mut DenseProp<i64, VertexTag>) -> usize
where
    G: Bidirectional + VertexList,
{
    let bound = g.vertex_bound();
    let n = bound.len();
    let mut view = label.sized_for(bound).expect(WRONG_GRAPH);
    let lab = view.as_mut_slice();
    lab.fill(UNLABELED);

    /// Discovery time of a vertex the DFS has not reached yet.
    const UNVISITED: usize = usize::MAX;

    let mut disc: Vec<usize> = vec![UNVISITED; n];
    let mut low: Vec<usize> = vec![UNVISITED; n];
    // The Tarjan stack: vertices discovered and not yet assigned a component.
    let mut open: Vec<VertexId> = Vec::new();
    // The explicit DFS stack. `boost::depth_first_search` recurses; a
    // 10^7-vertex path would overflow the thread stack there, and does not
    // here. Each frame owns its own incidence iterator, which is what makes
    // "resume where this vertex left off" expressible without re-scanning.
    let mut frames: Vec<(VertexId, G::Out)> = Vec::new();
    let mut time: usize = 0;
    let mut n_comp: i64 = 0;

    for root in g.vertices() {
        if disc[root.index()] != UNVISITED {
            continue;
        }
        disc[root.index()] = time;
        low[root.index()] = time;
        time += 1;
        open.push(root);
        frames.push((root, g.out_edges(root)));

        // `v` is copied out of the frame and the iterator step taken before
        // anything touches `frames` again, so the frame borrow ends on this
        // line and the recursive `push` below is an ordinary statement rather
        // than an aliasing problem.
        while let Some(&mut (v, ref mut it)) = frames.last_mut() {
            let step = it.next();

            if let Some(i) = step {
                let w = i.other;
                if disc[w.index()] == UNVISITED {
                    disc[w.index()] = time;
                    low[w.index()] = time;
                    time += 1;
                    open.push(w);
                    frames.push((w, g.out_edges(w)));
                } else if lab[w.index()] == UNLABELED {
                    // Discovered and not yet in a component, i.e. still on the
                    // Tarjan stack. This is `tarjan_scc_visitor`'s own test:
                    // it compares `get(m_comp, w)` against the map's sentinel
                    // rather than carrying a separate `on_stack` array.
                    let dw = disc[w.index()];
                    let lv = &mut low[v.index()];
                    if dw < *lv {
                        *lv = dw;
                    }
                } // else: a cross edge into a finished component, ignored.
                continue;
            }

            frames.pop();
            if low[v.index()] == disc[v.index()] {
                // `v` roots an SCC: unwind the open stack down to it.
                while let Some(w) = open.pop() {
                    lab[w.index()] = n_comp;
                    if w == v {
                        break;
                    }
                }
                n_comp += 1;
            }
            if let Some((p, _)) = frames.last() {
                let lv = low[v.index()];
                let lp = &mut low[p.index()];
                if lv < *lp {
                    *lp = lv;
                }
            }
        }
    }

    debug_assert!(open.is_empty(), "every discovered vertex joins a component");
    n_comp as usize
}

/// The largest component, as a vertex mask suitable for
/// [`MaskFilter`](gt_core::view::MaskFilter).
///
/// The mask is indexed by the *unfiltered* vertex index space and has exactly
/// [`vertex_bound`](gt_core::graph::GraphBase::vertex_bound) entries, which is
/// what [`Filtered::masked`](gt_core::view::Filtered::masked) demands: sizing
/// it from `num_vertices` instead is `graph_copy.cc:66-73`'s error, and on a
/// filtered input it is not even a big-enough allocation.
///
/// This is [`components`], not [`strong_components`] --- the bound is
/// `GraphRef`, so a directed view's answer is its out-reachability partition.
/// `label_largest_component` (`__init__.py:1385-1432`) instead inherits
/// `label_components`' tag dispatch and therefore silently means the largest
/// *strong* component on a directed graph. Callers who want that should run
/// [`strong_components`] and take the argmax themselves; callers who want the
/// largest connected component pass `g.undirect()`, which is the explicit
/// spelling of `label_largest_component(g, directed=False)`.
///
/// Ties go to the lowest label, matching `h.argmax()` (`__init__.py:1431`),
/// which numpy resolves to the first maximum. An empty view yields an
/// all-zero mask rather than the `ValueError` numpy's `argmax` raises on an
/// empty histogram.
///
/// # Panics
///
/// Never, for any `g`: the scratch map is minted from `g`'s own id.
pub fn largest_component_mask<G>(g: G) -> Vec<u8>
where
    G: GraphRef + VertexList,
{
    let n = g.vertex_bound().len();
    let mut label = DenseProp::<i64, VertexTag>::new(g.graph_id());
    let n_comp = components(g, &mut label);

    let lab = label.as_slice();
    let mut mask = vec![0u8; n];
    if n_comp == 0 {
        return mask;
    }

    // The histogram `HistogramPropertyMap` builds incrementally
    // (`graph_components.hh:63-74`), built in one pass instead: its `put`
    // hook exists only because boost writes the labels one at a time.
    let mut hist = vec![0usize; n_comp];
    for &c in lab.iter().take(n) {
        if c != UNLABELED {
            hist[c as usize] += 1;
        }
    }

    // `argmax`: first maximum wins.
    let mut best = 0usize;
    for (c, &h) in hist.iter().enumerate() {
        if h > hist[best] {
            best = c;
        }
    }

    let best = best as i64;
    for (slot, &c) in mask.iter_mut().zip(lab.iter()) {
        *slot = u8::from(c == best);
    }
    mask
}
