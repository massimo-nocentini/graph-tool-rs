//! The filtered view.
//!
//! ## Type parameter, not const generic, not runtime predicate (DESIGN.md D3)
//!
//! * **Type parameter** (chosen). [`KeepAll`] is a ZST, so `Filtered<G, KeepAll>`
//!   carries only the two memoised counters and `F::TRIVIAL` const-folds the
//!   predicate chain away. Instantiation count: 3 directedness shapes x 2
//!   filter shapes = 6, matching `graph_filtering.hh:127` exactly.
//! * **Const generic** `Filtered<G, const ON: bool>`: the same six, but a const
//!   erases no *data*, so the mask fields (two fat pointers) exist in the
//!   unfiltered arm and must be populated with dummies. Strictly worse.
//! * **Runtime predicate** (`&dyn Fn(EdgeId) -> bool`): one instantiation
//!   instead of six, a real compile-time win, but the inner loop gains an
//!   unspeculatable indirect call that blocks unrolling. Rejected on hot
//!   paths; acceptable behind the single cold dispatch boundary.

use crate::adj::{EdgeRef, FilterEdges, FilterIncident, FilterVertices, Incident};
use crate::bound::{EdgeBound, VertexBound};
use crate::dir::HasDir;
use crate::error::PropError;
use crate::graph::{Bidirectional, EdgeList, Endpoints, GraphBase, GraphRef, VertexList};
use crate::ids::{EdgeId, GraphId, VertexId};

/// A predicate over both identifier spaces.
///
/// One parameter, not two. `graph_filtering.cc:35` already collapses
/// `_edge_filter_active || _vertex_filter_active` into a single branch.
pub trait Filter: Copy + Send + Sync {
    /// `true` when this filter keeps everything, letting every probe fold away.
    const TRIVIAL: bool;
    /// Whether `v` survives.
    fn keep_vertex(self, v: VertexId) -> bool;
    /// Whether `e` survives.
    fn keep_edge(self, e: EdgeId) -> bool;
}

/// The trivial filter. A ZST.
#[derive(Clone, Copy, Default, Debug)]
pub struct KeepAll;

impl Filter for KeepAll {
    const TRIVIAL: bool = true;
    #[inline(always)]
    fn keep_vertex(self, v: VertexId) -> bool {
        true
    }
    #[inline(always)]
    fn keep_edge(self, e: EdgeId) -> bool {
        true
    }
}

/// A filter backed by two `uint8` masks, as graph-tool's `MaskFilter`
/// (`graph_filtering.hh:42`) is.
///
/// The borrow is load-bearing: while any [`Filtered`] holding this exists, the
/// mask storage cannot be resized or replaced. graph-tool instead relies on
/// the programmer remembering `_gviews.clear()` in `set_vertex_filter_property`
/// (`graph_filtering.cc:179`) and `set_edge_filter_property` (`:196`).
#[derive(Clone, Copy, Debug)]
pub struct MaskFilter<'m> {
    vmask: &'m [u8],
    emask: &'m [u8],
}

impl Filter for MaskFilter<'_> {
    const TRIVIAL: bool = false;

    /// Non-zero keeps, exactly as `set_vertex_filter` documents ("only the
    /// vertices with value different than `False` are kept",
    /// `graph_tool/__init__.py:3505-3507`).
    ///
    /// An out-of-range probe answers *filtered out*. `MaskFilter::operator()`
    /// is `boost::get(_filtered_property, d)` (`graph_filtering.hh:51-56`) on
    /// an **unchecked** map, whose `operator[]` is an unguarded `_data[i]`
    /// (`fast_vector_property_map.hh:218-221`, defect #11): a mask shorter
    /// than the index space reads past the end. That case is unreachable
    /// through [`Filtered::masked`], which refuses a short mask; this clause
    /// is what makes it unreachable through the one remaining door -- a
    /// `MaskFilter` copied out of [`Filtered::filter`] and handed to
    /// [`Filtered::new`] on a *different*, larger graph. There it
    /// under-filters, deterministically, instead of reading out of bounds,
    /// and the memoised counts still agree with iteration because both go
    /// through this predicate.
    #[inline]
    fn keep_vertex(self, v: VertexId) -> bool {
        self.vmask.get(v.index()).is_some_and(|&keep| keep != 0)
    }

    /// Non-zero keeps. Out of range is filtered out; see
    /// [`keep_vertex`](Self::keep_vertex).
    #[inline]
    fn keep_edge(self, e: EdgeId) -> bool {
        self.emask.get(e.index()).is_some_and(|&keep| keep != 0)
    }
}

/// A view restricted to the vertices and edges a [`Filter`] keeps.
///
/// Both counts are memoised at construction from [`EdgeList::edges`] and
/// [`VertexList::vertices`], using the *same* predicate the iterators use, so
/// `num_edges() == edges().count()` holds. Computing the edge count by summing
/// out-degrees instead is how a filtered undirected view comes to report three
/// edges where the answer is one.
#[derive(Clone, Copy, Debug)]
pub struct Filtered<G, F> {
    inner: G,
    filter: F,
    n_vertices: usize,
    n_edges: usize,
}

/// Whether an edge survives, endpoints included. The single predicate shared
/// by [`Filtered::new`] and every filtered iterator.
#[inline]
pub(crate) fn keeps_edge<F: Filter>(f: F, e: EdgeRef) -> bool {
    F::TRIVIAL || (f.keep_edge(e.id()) && f.keep_vertex(e.source()) && f.keep_vertex(e.target()))
}

/// Whether an incidence survives, given that its anchor already survives.
#[inline]
pub(crate) fn keeps_incident<F: Filter>(f: F, i: Incident) -> bool {
    F::TRIVIAL || (f.keep_edge(i.edge) && f.keep_vertex(i.other))
}

impl<G: GraphRef + VertexList + EdgeList, F: Filter> Filtered<G, F> {
    /// Build a filtered view, memoising both honest counts.
    ///
    /// Short-circuits on `F::TRIVIAL`, so the unfiltered arm pays no prepass.
    /// Otherwise O(V + E), which is the price of an honest `num_vertices`;
    /// `filt_graph` gets O(1) by returning the wrong number.
    pub fn new(inner: G, filter: F) -> Self {
        if F::TRIVIAL {
            return Filtered {
                n_vertices: inner.num_vertices(),
                n_edges: inner.num_edges(),
                inner,
                filter,
            };
        }
        // `keeps_edge` here and in `FilterEdges::next` are the *same*
        // function, which is the whole of defect #16: `num_edges` counted by
        // one rule and `edges()` yielded by another is how
        // `distance(ei, eiend) != num_edges(g)` (`graph_filtered.hh:301-312`)
        // happens. Summing degrees instead would be a third rule, and on an
        // undirected view a wrong one.
        let n_vertices = inner.vertices().filter(|&v| filter.keep_vertex(v)).count();
        let n_edges = inner.edges().filter(|&e| keeps_edge(filter, e)).count();
        Filtered {
            inner,
            filter,
            n_vertices,
            n_edges,
        }
    }
}

impl<G: GraphRef + VertexList + EdgeList> Filtered<G, KeepAll> {
    /// The trivially filtered view: same six types, no prepass.
    #[inline]
    pub fn all(inner: G) -> Self {
        Filtered {
            n_vertices: inner.num_vertices(),
            n_edges: inner.num_edges(),
            inner,
            filter: KeepAll,
        }
    }
}

impl<'m, G: GraphRef + VertexList + EdgeList> Filtered<G, MaskFilter<'m>> {
    /// Build a mask-filtered view, validating the masks against *this graph's*
    /// own bounds.
    ///
    /// Validation and use are one statement. A free-standing
    /// `MaskFilter::new(vmask, emask, vbound, ebound)` is `Copy` and remembers
    /// nothing, so a mask admitted for a 3-vertex graph is silently accepted
    /// by a 10-vertex one and panics on first probe -- which is the same
    /// structural defect as `graph_filtering.cc:42-46` reserving separately
    /// from `MaskFilter::operator()`.
    pub fn masked(inner: G, vmask: &'m [u8], emask: &'m [u8]) -> Result<Self, PropError> {
        // Against the *bound*, never against `num_vertices`: the mask is
        // indexed by the unfiltered index space, which is exactly the
        // distinction `graph_filtering.cc:42-46` gets right
        // (`get_edge_index_range()`, `num_vertices(*u)`) and
        // `graph_copy.cc:66-73` gets wrong.
        let vb = inner.vertex_bound();
        if vmask.len() < vb.len() {
            return Err(PropError::ShortMask {
                have: vmask.len(),
                need: vb.len(),
            });
        }
        let eb = inner.edge_bound();
        if emask.len() < eb.len() {
            return Err(PropError::ShortMask {
                have: emask.len(),
                need: eb.len(),
            });
        }
        Ok(Filtered::new(inner, MaskFilter { vmask, emask }))
    }
}

impl<G, F> Filtered<G, F> {
    /// The unfiltered view underneath.
    #[inline]
    pub const fn inner(&self) -> &G {
        &self.inner
    }
    /// The filter.
    #[inline]
    pub const fn filter(&self) -> &F {
        &self.filter
    }
}

impl<G: HasDir, F> HasDir for Filtered<G, F> {
    type Dir = G::Dir;
}

impl<G: GraphBase, F: Filter> GraphBase for Filtered<G, F> {
    #[inline]
    fn graph_id(self) -> GraphId {
        self.inner.graph_id()
    }
    /// The honest filtered cardinality.
    #[inline]
    fn num_vertices(self) -> usize {
        self.n_vertices
    }
    /// The honest filtered cardinality.
    #[inline]
    fn num_edges(self) -> usize {
        self.n_edges
    }
    /// The **unfiltered** allocation bound, forwarded. This is
    /// `graph_filtered.hh:316`'s behaviour, under a name that says what it is.
    #[inline]
    fn vertex_bound(self) -> VertexBound {
        self.inner.vertex_bound()
    }
    /// The **unfiltered** allocation bound, forwarded.
    #[inline]
    fn edge_bound(self) -> EdgeBound {
        self.inner.edge_bound()
    }
}

impl<G: GraphRef, F: Filter> GraphRef for Filtered<G, F> {
    type Out = FilterIncident<G::Out, F>;
    type All = FilterIncident<G::All, F>;
    #[inline]
    fn out_edges(self, v: VertexId) -> Self::Out {
        FilterIncident {
            inner: self.inner.out_edges(v),
            filter: self.filter,
        }
    }
    #[inline]
    fn all_edges(self, v: VertexId) -> Self::All {
        FilterIncident {
            inner: self.inner.all_edges(v),
            filter: self.filter,
        }
    }
    /// O(deg), by probing. `out_degree(u, filt_graph)` counts the filtered
    /// range (`graph_filtered.hh:383-392`) rather than forwarding, and so does
    /// this. The anchor `v` itself is *not* probed, matching
    /// `out_edges(u, g)`, which applies `out_edge_pred` (edge and *target*)
    /// and never tests `u`: the caller reached `v` through `vertices()`, which
    /// already filtered it, and this is what keeps
    /// `sum(out_degree) == num_edges` true over a directed filtered view.
    #[inline]
    fn out_degree(self, v: VertexId) -> usize {
        if F::TRIVIAL {
            return self.inner.out_degree(v);
        }
        self.out_edges(v).count()
    }
    /// O(deg), by probing the whole incidence run.
    ///
    /// Counted from [`all_edges`](GraphRef::all_edges), not as
    /// `in_degree + out_degree` the way `degree(u, filt_graph)`
    /// (`graph_filtered.hh:395-399`) is: that form needs a `Bidirectional`
    /// underneath, so on an undirected view it reaches
    /// `in_edges(u, undirected_adaptor)` -- the default-constructed empty
    /// range of `graph_adaptor.hh:219-227`, defect #12 -- and silently halves
    /// the answer.
    #[inline]
    fn degree(self, v: VertexId) -> usize {
        if F::TRIVIAL {
            return self.inner.degree(v);
        }
        self.all_edges(v).count()
    }
}

impl<G: Bidirectional, F: Filter> Bidirectional for Filtered<G, F> {
    type In = FilterIncident<G::In, F>;
    #[inline]
    fn in_edges(self, v: VertexId) -> Self::In {
        FilterIncident {
            inner: self.inner.in_edges(v),
            filter: self.filter,
        }
    }
    /// O(deg), by probing. `in_edges`' predicate is edge-and-*source*
    /// (`in_edge_pred`), which on an anchored [`Incident`] is edge-and-`other`
    /// -- the same `keeps_incident` the out-half uses, with no orientation
    /// branch, because the anchor carries the orientation.
    #[inline]
    fn in_degree(self, v: VertexId) -> usize {
        if F::TRIVIAL {
            return self.inner.in_degree(v);
        }
        self.in_edges(v).count()
    }
}

impl<G: VertexList, F: Filter> VertexList for Filtered<G, F> {
    type Vertices = FilterVertices<G::Vertices, F>;
    #[inline]
    fn vertices(self) -> Self::Vertices {
        FilterVertices {
            inner: self.inner.vertices(),
            filter: self.filter,
        }
    }
}

impl<G: EdgeList, F: Filter> EdgeList for Filtered<G, F> {
    type Edges = FilterEdges<G::Edges, F>;
    #[inline]
    fn edges(self) -> Self::Edges {
        FilterEdges {
            inner: self.inner.edges(),
            filter: self.filter,
        }
    }
}

impl<G: Endpoints + GraphRef, F: Filter> Endpoints for Filtered<G, F> {
    /// `None` when the edge or either endpoint is filtered out.
    ///
    /// `source(e, filt_graph)` and `target(e, filt_graph)` forward to the
    /// unfiltered graph unconditionally (`graph_filtered.hh:341-355`), so a
    /// descriptor that the view's own `edges()` never yields still resolves to
    /// a pair of vertices its own `vertices()` never yields. The predicate
    /// here is `keeps_edge` -- the same one `edges()` and `num_edges()` use.
    #[inline]
    fn endpoints(self, e: EdgeId) -> Option<(VertexId, VertexId)> {
        let (s, t) = self.inner.endpoints(e)?;
        keeps_edge(self.filter, EdgeRef::new(e, s, t)).then_some((s, t))
    }
    #[inline]
    fn opposite(self, e: EdgeId, v: VertexId) -> Option<VertexId> {
        let (s, t) = self.endpoints(e)?;
        EdgeRef::new(e, s, t).opposite(v)
    }
    /// The first *surviving* `s -> t` edge, in this view's orientation.
    ///
    /// The scan is the point. `inner.find_edge(s, t).filter(..)` would answer
    /// `None` whenever the first parallel edge is filtered out and a later one
    /// survives -- a lookup disagreeing with the view's own `out_edges(s)`,
    /// which is defect #16 in miniature. `edge(u, v, filt_graph)` scans too
    /// (`graph_filtered.hh:502-540` drives `edge_range_iter`), so this is also
    /// what the C++ does; what the C++ then omits is the vertex predicate,
    /// testing only `g._edge_pred(e)` (`:529`), so its `edge()` returns edges
    /// whose endpoints its `vertices()` has removed. Here one predicate
    /// serves both.
    ///
    /// O(deg) when filtering is on, O(1)-or-`Lookup` when it is not.
    fn find_edge(self, s: VertexId, t: VertexId) -> Option<EdgeRef> {
        if F::TRIVIAL {
            return self.inner.find_edge(s, t);
        }
        if !self.filter.keep_vertex(s) {
            return None;
        }
        let f = self.filter;
        let hit = self
            .inner
            .out_edges(s)
            .find(|&i| i.other == t && keeps_incident(f, i))?;
        // Orientation comes from the inner view, so a reversed view reports
        // reversed endpoints and an undirected one reports storage order --
        // each exactly as its own unfiltered `find_edge` would.
        let (es, et) = self.inner.endpoints(hit.edge)?;
        Some(EdgeRef::new(hit.edge, es, et))
    }
}

// ---------------------------------------------------------------------------
// Iterator impls for the filtering adaptors declared in `adj::iter`.
// ---------------------------------------------------------------------------

impl<I: Iterator<Item = Incident>, F: Filter> Iterator for FilterIncident<I, F> {
    type Item = Incident;
    /// `Iterator::find` rather than a hand-rolled `while let`, so the probe
    /// runs inside the inner iterator's own `try_fold` and keeps the
    /// unrolled slice loop that `IncidentIter::fold` exists to expose.
    #[inline]
    fn next(&mut self) -> Option<Incident> {
        if F::TRIVIAL {
            return self.inner.next();
        }
        let f = self.filter;
        self.inner.find(|&i| keeps_incident(f, i))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.inner.size_hint().1)
    }
    #[inline]
    fn fold<B, Fun: FnMut(B, Incident) -> B>(self, init: B, mut fun: Fun) -> B {
        if F::TRIVIAL {
            return self.inner.fold(init, fun);
        }
        let f = self.filter;
        self.inner.fold(init, move |acc, i| {
            if keeps_incident(f, i) {
                fun(acc, i)
            } else {
                acc
            }
        })
    }
}

/// Filtering never resurrects an exhausted iterator, so fusedness survives it.
/// [`ExactSizeIterator`] deliberately does **not**: that is the `ExactIncidence`
/// refinement (DESIGN.md section 4) which filtered views do not implement.
impl<I: std::iter::FusedIterator<Item = Incident>, F: Filter> std::iter::FusedIterator
    for FilterIncident<I, F>
{
}

impl<I: Iterator<Item = EdgeRef>, F: Filter> Iterator for FilterEdges<I, F> {
    type Item = EdgeRef;
    /// `keeps_edge`, the same predicate [`Filtered::new`] counted with --
    /// which is what makes `num_edges() == edges().count()` a fact rather
    /// than a hope.
    #[inline]
    fn next(&mut self) -> Option<EdgeRef> {
        if F::TRIVIAL {
            return self.inner.next();
        }
        let f = self.filter;
        self.inner.find(|&e| keeps_edge(f, e))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.inner.size_hint().1)
    }
    #[inline]
    fn fold<B, Fun: FnMut(B, EdgeRef) -> B>(self, init: B, mut fun: Fun) -> B {
        if F::TRIVIAL {
            return self.inner.fold(init, fun);
        }
        let f = self.filter;
        self.inner.fold(
            init,
            move |acc, e| {
                if keeps_edge(f, e) { fun(acc, e) } else { acc }
            },
        )
    }
}

impl<I: std::iter::FusedIterator<Item = EdgeRef>, F: Filter> std::iter::FusedIterator
    for FilterEdges<I, F>
{
}

impl<I: Iterator<Item = VertexId>, F: Filter> Iterator for FilterVertices<I, F> {
    type Item = VertexId;
    #[inline]
    fn next(&mut self) -> Option<VertexId> {
        if F::TRIVIAL {
            return self.inner.next();
        }
        let f = self.filter;
        self.inner.find(|&v| f.keep_vertex(v))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.inner.size_hint().1)
    }
    #[inline]
    fn fold<B, Fun: FnMut(B, VertexId) -> B>(self, init: B, mut fun: Fun) -> B {
        if F::TRIVIAL {
            return self.inner.fold(init, fun);
        }
        let f = self.filter;
        self.inner.fold(
            init,
            move |acc, v| {
                if f.keep_vertex(v) { fun(acc, v) } else { acc }
            },
        )
    }
}

impl<I: std::iter::FusedIterator<Item = VertexId>, F: Filter> std::iter::FusedIterator
    for FilterVertices<I, F>
{
}
