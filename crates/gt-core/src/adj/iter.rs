//! Iterators.
//!
//! Descriptors are manufactured on dereference and never stored, matching
//! `base_edge_iterator<Dereference>` (`graph_adjacency.hh:255-290`). Because
//! [`Incident`] is *anchored* rather than symmetric, there is no orientation
//! policy to apply and therefore no per-element branch: `all_edge_iterator`'s
//! `make_in_or_out_edge` (`:327-341`) does a `reinterpret_cast` from the CRTP
//! base to the derived iterator on **every dereference**, purely to read
//! `_pos` and decide which way round to build the descriptor. Here the out-,
//! in- and all-iterators differ only in which slice they walk.
//!
//! Every iterator forwards `fold`, so `for_each`/`sum`/`try_fold` reach
//! LLVM's unrolled slice loop rather than going through `next()`.

use super::block::Block;
use super::entry::{AdjEntry, EdgeRef, Incident};
use crate::ids::{EdgeId, VertexId};

/// Walks a contiguous run of adjacency entries, yielding anchored incidences.
#[derive(Clone, Debug)]
pub struct IncidentIter<'a> {
    inner: std::slice::Iter<'a, AdjEntry>,
}

impl<'a> IncidentIter<'a> {
    #[inline]
    pub(crate) fn new(entries: &'a [AdjEntry]) -> Self {
        IncidentIter {
            inner: entries.iter(),
        }
    }

    /// An empty run.
    #[inline]
    pub fn empty() -> Self {
        IncidentIter { inner: [].iter() }
    }
}

impl Iterator for IncidentIter<'_> {
    type Item = Incident;

    #[inline]
    fn next(&mut self) -> Option<Incident> {
        self.inner.next().map(|e| Incident {
            other: e.other,
            edge: e.idx,
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn fold<B, F: FnMut(B, Incident) -> B>(self, init: B, mut f: F) -> B {
        self.inner.fold(init, move |acc, e| {
            f(
                acc,
                Incident {
                    other: e.other,
                    edge: e.idx,
                },
            )
        })
    }
}

impl DoubleEndedIterator for IncidentIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Incident> {
        self.inner.next_back().map(|e| Incident {
            other: e.other,
            edge: e.idx,
        })
    }
}
impl ExactSizeIterator for IncidentIter<'_> {}
impl std::iter::FusedIterator for IncidentIter<'_> {}

/// Out-edges of a vertex.
pub type OutEdges<'a> = IncidentIter<'a>;
/// In-edges of a vertex.
pub type InEdges<'a> = IncidentIter<'a>;
/// All incident edges of a vertex, out-half first. Ports `_all_edges_out`.
pub type AllEdges<'a> = IncidentIter<'a>;

/// The vertex set, as ids.
#[derive(Clone, Debug)]
pub struct Vertices {
    range: std::ops::Range<usize>,
}

impl Vertices {
    #[inline]
    pub(crate) const fn new(n: usize) -> Self {
        Vertices { range: 0..n }
    }
}

impl Iterator for Vertices {
    type Item = VertexId;
    #[inline]
    fn next(&mut self) -> Option<VertexId> {
        self.range.next().map(VertexId::from_index)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
    #[inline]
    fn fold<B, F: FnMut(B, VertexId) -> B>(self, init: B, mut f: F) -> B {
        self.range
            .fold(init, move |acc, i| f(acc, VertexId::from_index(i)))
    }
}
impl ExactSizeIterator for Vertices {}
impl std::iter::FusedIterator for Vertices {}

/// The global edge list: every edge exactly once, in canonical orientation.
///
/// This, not degree-summation, is the basis for `num_edges` on a view.
/// Summing out-degrees over an undirected view double-counts, which is how a
/// filtered undirected view comes to report three edges where the answer is
/// one.
#[derive(Clone, Debug)]
pub struct Edges<'a> {
    blocks: &'a [Block],
    vertex: usize,
    within: usize,
}

impl<'a> Edges<'a> {
    #[inline]
    pub(crate) const fn new(blocks: &'a [Block]) -> Self {
        Edges {
            blocks,
            vertex: 0,
            within: 0,
        }
    }
}

impl Iterator for Edges<'_> {
    type Item = EdgeRef;

    /// Ports `edge_iterator::increment` (`graph_adjacency.hh:393-397`) and the
    /// `skip` it calls (`:381-391`): advance within the current vertex's
    /// **out**-half, and when it is exhausted walk forward over vertices until
    /// one has an out-entry. `skip`'s loop condition is
    /// `_ei == _vi->second.begin() + _vi->first` (`:385`) -- the end of the
    /// out-half, not the end of the block -- which is what makes the global
    /// list yield every edge exactly once even though every edge is stored
    /// twice, once in each endpoint's block. The descriptor is
    /// `(vertex_t(_vi - _vi_begin), _ei->first, _ei->second)` (`:408-409`),
    /// i.e. the *storage* orientation: source is the block's own vertex.
    ///
    /// The cursor is forward-only: `vertex` never decreases and `within`
    /// resets to zero exactly when `vertex` advances, so a full sweep does
    /// O(V + E) work in total. An empty block is tested once for the whole
    /// sweep, not once per edge -- the O(V) stall a "rescan from the first
    /// block" formulation would pay.
    #[inline]
    fn next(&mut self) -> Option<EdgeRef> {
        loop {
            // `?` ends the sweep at the last block and leaves the cursor
            // parked there, which is what makes this iterator fused.
            let out = self.blocks.get(self.vertex)?.out();
            if let Some(e) = out.get(self.within) {
                let src = VertexId::from_index(self.vertex);
                self.within += 1;
                return Some(EdgeRef::new(e.idx, src, e.other));
            }
            self.vertex += 1;
            self.within = 0;
        }
    }

    /// Two nested slice folds, so a full sweep never goes through the
    /// per-element state machine above. `count()`, and therefore
    /// `num_edges() == edges().count()` (DESIGN.md section 4), lands here.
    #[inline]
    fn fold<B, F: FnMut(B, EdgeRef) -> B>(self, init: B, mut f: F) -> B {
        let Edges {
            blocks,
            vertex,
            mut within,
        } = self;
        let mut acc = init;
        for (k, block) in blocks.get(vertex..).unwrap_or_default().iter().enumerate() {
            let src = VertexId::from_index(vertex + k);
            // Only the block the cursor is parked in is entered part-way; the
            // `take` makes every later block start at zero.
            let rest = block
                .out()
                .get(std::mem::take(&mut within)..)
                .unwrap_or_default();
            acc = rest
                .iter()
                .fold(acc, |acc, e| f(acc, EdgeRef::new(e.idx, src, e.other)));
        }
        acc
    }
}
impl std::iter::FusedIterator for Edges<'_> {}

/// Exchanges the endpoints of every [`EdgeRef`] an inner iterator yields.
///
/// The [`EdgeList`](crate::graph::EdgeList) adaptor for
/// [`Rev`](crate::view::Rev).
#[derive(Clone, Debug)]
pub struct SwapEnds<I> {
    inner: I,
}

impl<I> SwapEnds<I> {
    /// Wrap an iterator.
    #[inline]
    pub const fn new(inner: I) -> Self {
        SwapEnds { inner }
    }
}

impl<I: Iterator<Item = EdgeRef>> Iterator for SwapEnds<I> {
    type Item = EdgeRef;
    #[inline]
    fn next(&mut self) -> Option<EdgeRef> {
        self.inner.next().map(EdgeRef::reversed)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    #[inline]
    fn fold<B, F: FnMut(B, EdgeRef) -> B>(self, init: B, mut f: F) -> B {
        self.inner.fold(init, move |acc, e| f(acc, e.reversed()))
    }
}
impl<I: ExactSizeIterator<Item = EdgeRef>> ExactSizeIterator for SwapEnds<I> {}
impl<I: std::iter::FusedIterator<Item = EdgeRef>> std::iter::FusedIterator for SwapEnds<I> {}

/// Yields only the edges whose identity satisfies a predicate.
///
/// Hand-written rather than `std::iter::Filter` so that the view stays `Copy`
/// (a closure-carrying `Filter` is not) and so the trivial-filter fast path
/// stays visible to the optimiser.
#[derive(Clone, Debug)]
pub struct FilterEdges<I, F> {
    pub(crate) inner: I,
    pub(crate) filter: F,
}

/// Yields only the incidences whose edge and neighbour survive a predicate.
#[derive(Clone, Debug)]
pub struct FilterIncident<I, F> {
    pub(crate) inner: I,
    pub(crate) filter: F,
}

/// Yields only the vertices that survive a predicate.
#[derive(Clone, Debug)]
pub struct FilterVertices<I, F> {
    pub(crate) inner: I,
    pub(crate) filter: F,
}

/// The edge identities of an incidence run, for callers that only want ids.
#[derive(Clone, Debug)]
pub struct EdgeIdsOf<I> {
    inner: I,
}

impl<I> EdgeIdsOf<I> {
    /// Wrap an iterator.
    #[inline]
    pub const fn new(inner: I) -> Self {
        EdgeIdsOf { inner }
    }
}

impl<I: Iterator<Item = Incident>> Iterator for EdgeIdsOf<I> {
    type Item = EdgeId;
    #[inline]
    fn next(&mut self) -> Option<EdgeId> {
        self.inner.next().map(|i| i.edge)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    #[inline]
    fn fold<B, F: FnMut(B, EdgeId) -> B>(self, init: B, mut f: F) -> B {
        self.inner.fold(init, move |acc, i| f(acc, i.edge))
    }
}
impl<I: ExactSizeIterator<Item = Incident>> ExactSizeIterator for EdgeIdsOf<I> {}
impl<I: std::iter::FusedIterator<Item = Incident>> std::iter::FusedIterator for EdgeIdsOf<I> {}

// ===========================================================================
// U4 — unit tests
//
// `Block`'s `entries`/`out_len` are private to `adj::block`, so the populated
// sweep is built here the way `AdjList::add_edge` will build it: U2's
// `insert_out`/`insert_in`, which are `pub(crate)`. That keeps every claim in
// the acceptance list testable from inside the module, independently of U5.
// `tests/u04_iter.rs` then re-asserts the same properties through the public
// `AdjList` API, which is the pairing that would catch a `list.rs` that wires
// the wrong slice into `Edges::new`.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: usize) -> VertexId {
        VertexId::from_index(i)
    }
    fn e(i: usize) -> EdgeId {
        EdgeId::from_index(i)
    }

    /// `(other, idx)` pairs, as one vertex's half-edges.
    fn entries(pairs: &[(usize, usize)]) -> Vec<AdjEntry> {
        pairs
            .iter()
            .map(|&(o, i)| AdjEntry {
                other: v(o),
                idx: e(i),
            })
            .collect()
    }

    const RUN: [(usize, usize); 5] = [(3, 0), (7, 4), (3, 9), (0, 2), (11, 6)];

    // -- IncidentIter -------------------------------------------------------

    /// D2: `other` is the neighbour as stored, and the run is yielded in
    /// block order — `base_edge_iterator` is a plain sequence walker over
    /// `es` (`graph_adjacency.hh:255-290`), it does not sort or dedup.
    #[test]
    fn incident_yields_the_run_in_order() {
        let es = entries(&RUN);
        let got: Vec<(usize, usize)> = IncidentIter::new(&es)
            .map(|i| (i.other.index(), i.edge.index()))
            .collect();
        assert_eq!(got, RUN.to_vec());
    }

    /// `fold` is an override, so it is a second implementation of `next` and
    /// has to be pinned against it — including from a partially advanced
    /// iterator, which is the case `for_each`-after-`next` produces.
    #[test]
    fn incident_fold_and_next_agree_from_every_start() {
        let es = entries(&RUN);
        for skip in 0..=RUN.len() {
            let mut by_next = IncidentIter::new(&es);
            for _ in 0..skip {
                by_next.next();
            }
            let folded = by_next.clone().fold(Vec::new(), |mut acc, i| {
                acc.push((i.other.index(), i.edge.index()));
                acc
            });
            let stepped: Vec<(usize, usize)> =
                by_next.map(|i| (i.other.index(), i.edge.index())).collect();
            assert_eq!(folded, stepped, "fold disagrees with next after {skip}");
        }
    }

    /// `ExactSizeIterator` is a promise, not a hint: `Filtered::new` memoises
    /// counts and `ExactIncidence` is a refinement unfiltered views implement
    /// (DESIGN.md section 4). Both are wrong if `size_hint` is loose.
    #[test]
    fn incident_size_hint_is_exact_at_every_step() {
        let es = entries(&RUN);
        let mut it = IncidentIter::new(&es);
        for remaining in (0..=RUN.len()).rev() {
            assert_eq!(it.size_hint(), (remaining, Some(remaining)));
            assert_eq!(it.len(), remaining);
            it.next();
        }
        assert_eq!(it.size_hint(), (0, Some(0)));
        // Fused: exhausted stays exhausted.
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn incident_double_ended_meets_in_the_middle() {
        let es = entries(&RUN);
        let mut it = IncidentIter::new(&es);
        assert_eq!(it.next().unwrap().edge.index(), 0);
        assert_eq!(it.next_back().unwrap().edge.index(), 6);
        assert_eq!(it.next_back().unwrap().edge.index(), 2);
        assert_eq!(it.len(), 2);
        let rest: Vec<usize> = it.map(|i| i.edge.index()).collect();
        assert_eq!(rest, vec![4, 9]);
    }

    #[test]
    fn incident_empty_is_empty() {
        let mut it = IncidentIter::empty();
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
        assert_eq!(IncidentIter::empty().count(), 0);
    }

    // -- Edges: the skip discipline -----------------------------------------

    /// `edge_iterator::skip` (`graph_adjacency.hh:381-391`) walks `_vi`
    /// forward over vertices whose out-half is empty. With *every* block
    /// empty that is the whole sweep, and it must terminate having visited
    /// each block exactly once rather than spinning on one.
    #[test]
    fn edges_over_empty_blocks_yields_nothing() {
        for n in [0usize, 1, 2, 64] {
            let blocks = vec![Block::new(); n];
            assert_eq!(Edges::new(&blocks).count(), 0, "n = {n}");
            assert!(Edges::new(&blocks).next().is_none(), "n = {n}");
            assert_eq!(Edges::new(&blocks).fold(0usize, |a, _| a + 1), 0, "n = {n}");
        }
    }

    /// The cursor is forward-only and advances one block per empty block —
    /// which is what makes a sweep O(V + E) rather than O(V) *per edge*. A
    /// rescanning formulation would still pass the test above; it would not
    /// pass this one, because `vertex` would not be monotone across calls.
    #[test]
    fn edges_cursor_advances_one_block_at_a_time_and_never_rewinds() {
        let blocks = vec![Block::new(); 32];
        let mut it = Edges::new(&blocks);
        let mut seen = it.vertex;
        assert!(it.next().is_none());
        // One `next` on an all-empty list parks the cursor past the end,
        // having stepped through every block once.
        assert!(it.vertex >= seen);
        seen = it.vertex;
        assert_eq!(seen, blocks.len());
        assert_eq!(it.within, 0);
        // Fused: further calls neither yield nor run off the end.
        for _ in 0..4 {
            assert!(it.next().is_none());
            assert_eq!(it.vertex, seen);
        }
    }

    /// `fold` from a cursor parked past the end is not an out-of-range slice
    /// index — `blocks.get(vertex..)` is `None`, not a panic.
    #[test]
    fn edges_fold_from_a_parked_cursor_is_empty() {
        let blocks = vec![Block::new(); 3];
        let mut it = Edges::new(&blocks);
        assert!(it.next().is_none());
        assert_eq!(it.clone().fold(0usize, |a, _| a + 1), 0);
        // And from a cursor deliberately parked beyond the block list.
        it.vertex = 99;
        it.within = 7;
        assert_eq!(it.fold(0usize, |a, _| a + 1), 0);
    }

    // -- Edges: the populated sweep -----------------------------------------

    /// The adjacency `AdjList::add_edge` produces: an out-entry in the
    /// source's block and an in-entry in the target's, edge `i` carrying id
    /// `i` (`EdgeIds::alloc` hands out `0, 1, 2, …` on an empty free list,
    /// which is also the state `compact()` leaves behind).
    fn blocks_of(n: usize, edges: &[(usize, usize)]) -> Vec<Block> {
        let mut blocks = vec![Block::new(); n];
        for (i, &(s, t)) in edges.iter().enumerate() {
            blocks[s].insert_out(AdjEntry {
                other: v(t),
                idx: e(i),
            });
            blocks[t].insert_in(AdjEntry {
                other: v(s),
                idx: e(i),
            });
        }
        blocks
    }

    fn swept(blocks: &[Block]) -> Vec<(usize, usize, usize)> {
        Edges::new(blocks)
            .map(|r| (r.source().index(), r.target().index(), r.id().index()))
            .collect()
    }

    /// One shape, every way of getting the sweep wrong:
    ///
    /// * vertex 0 has out-edges *and* in-edges — walking `Block::all` instead
    ///   of `Block::out` yields its in-entries a second time;
    /// * vertex 2 is isolated and vertex 4 has in-entries only — a cursor
    ///   that stops at the first exhausted out-half never reaches vertex 3;
    /// * `(3, 3)` is a self-loop, stored in both halves of one block;
    /// * `(0, 1)` appears twice — parallel edges are distinct edges, since
    ///   identity is the `EdgeId` and not the endpoint pair (D2).
    const MIXED: [(usize, usize); 6] = [(0, 1), (0, 1), (1, 0), (3, 3), (3, 4), (1, 4)];

    #[test]
    fn edges_yields_every_edge_exactly_once_in_storage_orientation() {
        let blocks = blocks_of(5, &MIXED);
        let mut expected = Vec::new();
        for u in 0..5 {
            for (i, &(s, t)) in MIXED.iter().enumerate() {
                if s == u {
                    expected.push((s, t, i));
                }
            }
        }
        assert_eq!(swept(&blocks), expected);

        let mut ids: Vec<usize> = Edges::new(&blocks).map(|r| r.id().index()).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "an edge was yielded twice");
        assert_eq!(n, MIXED.len());
        assert_eq!(Edges::new(&blocks).count(), MIXED.len());
    }

    /// A self-loop occupies both halves of one block and is still one edge:
    /// `skip`'s bound is the out-half's end (`graph_adjacency.hh:385`).
    #[test]
    fn a_self_loop_is_yielded_once() {
        let blocks = blocks_of(2, &[(0, 0), (0, 1)]);
        assert_eq!(swept(&blocks), vec![(0, 0, 0), (0, 1, 1)]);
    }

    /// Ids ascend when the blocks were filled source-major — the state after
    /// `EdgeIds::compact`, which remaps the live ids onto `0..live`.
    #[test]
    fn ids_ascend_after_a_compaction() {
        let blocks = blocks_of(3, &[(0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]);
        let ids: Vec<usize> = Edges::new(&blocks).map(|r| r.id().index()).collect();
        assert_eq!(ids, (0..6).collect::<Vec<_>>());
    }

    /// And *not* otherwise: the sweep is in vertex order, which is a promise
    /// about "each edge once", not about sortedness.
    #[test]
    fn ids_are_not_sorted_when_insertion_was_not_source_major() {
        let blocks = blocks_of(3, &[(2, 0), (0, 1)]);
        assert_eq!(swept(&blocks), vec![(0, 1, 1), (2, 0, 0)]);
    }

    /// The acceptance claim about cost, as an equality rather than a timing.
    ///
    /// Charge each `next()` the number of blocks it stepped over. A cursor
    /// that rescans from the front would charge O(V) per edge; this one
    /// charges each empty block once for the whole sweep, so the total is
    /// bounded by the number of blocks — here 1000 blocks and 3 edges, where
    /// a rescanning formulation would run up ~3000.
    #[test]
    fn skipping_empty_blocks_costs_each_block_once_for_the_whole_sweep() {
        let n = 1000;
        let blocks = blocks_of(n, &[(0, 1), (500, 2), (999, 3)]);
        let mut it = Edges::new(&blocks);
        let mut steps = 0usize;
        let mut yielded = 0usize;
        let mut at = it.vertex;
        loop {
            let got = it.next();
            assert!(it.vertex >= at, "the cursor rewound");
            steps += it.vertex - at;
            at = it.vertex;
            match got {
                Some(_) => yielded += 1,
                None => break,
            }
        }
        assert_eq!(yielded, 3);
        assert!(
            steps <= n,
            "the sweep stepped {steps} blocks over {n}: that is a rescan"
        );
    }

    #[test]
    fn edges_fold_and_next_agree_from_every_starting_position() {
        let blocks = blocks_of(5, &MIXED);
        let full = swept(&blocks);
        for skip in 0..=full.len() {
            let mut it = Edges::new(&blocks);
            for _ in 0..skip {
                it.next();
            }
            let folded = it.clone().fold(Vec::new(), |mut acc, r| {
                acc.push((r.source().index(), r.target().index(), r.id().index()));
                acc
            });
            assert_eq!(folded, full[skip..].to_vec(), "fold disagrees after {skip}");
            assert_eq!(it.count(), full.len() - skip);
        }
    }

    /// Exhausted stays exhausted, and the cursor does not run past the end.
    #[test]
    fn edges_is_fused_on_a_populated_list() {
        let blocks = blocks_of(4, &[(0, 1)]);
        let mut it = Edges::new(&blocks);
        assert!(it.next().is_some());
        for _ in 0..8 {
            assert!(it.next().is_none());
            assert_eq!(it.vertex, blocks.len());
        }
    }

    // -- Vertices -----------------------------------------------------------

    #[test]
    fn vertices_are_zero_to_n_and_fold_agrees() {
        let n = 7;
        let stepped: Vec<usize> = Vertices::new(n).map(|u| u.index()).collect();
        assert_eq!(stepped, (0..n).collect::<Vec<_>>());
        let folded = Vertices::new(n).fold(Vec::new(), |mut a, u| {
            a.push(u.index());
            a
        });
        assert_eq!(folded, stepped);
        let mut it = Vertices::new(n);
        for remaining in (0..=n).rev() {
            assert_eq!(it.size_hint(), (remaining, Some(remaining)));
            it.next();
        }
    }

    // -- SwapEnds and EdgeIdsOf ---------------------------------------------

    /// `Rev`'s edge list exchanges endpoints and keeps identity: `reverse_edge`
    /// (`graph_adjacency.hh:578-583`) rewrites `s`/`t` and leaves `idx` alone.
    #[test]
    fn swap_ends_exchanges_endpoints_and_keeps_identity() {
        let list = [
            EdgeRef::new(e(2), v(0), v(1)),
            EdgeRef::new(e(5), v(3), v(3)),
        ];
        let got: Vec<(usize, usize, usize)> = SwapEnds::new(list.iter().copied())
            .map(|r| (r.id().index(), r.source().index(), r.target().index()))
            .collect();
        assert_eq!(got, vec![(2, 1, 0), (5, 3, 3)]);
        // fold is an override here too.
        let folded = SwapEnds::new(list.iter().copied()).fold(Vec::new(), |mut a, r| {
            a.push((r.id().index(), r.source().index(), r.target().index()));
            a
        });
        assert_eq!(folded, got);
        assert_eq!(SwapEnds::new(list.iter().copied()).len(), 2);
    }

    #[test]
    fn edge_ids_of_projects_identity_only() {
        let es = entries(&RUN);
        let ids: Vec<usize> = EdgeIdsOf::new(IncidentIter::new(&es))
            .map(|i| i.index())
            .collect();
        assert_eq!(ids, RUN.iter().map(|&(_, i)| i).collect::<Vec<_>>());
        let folded = EdgeIdsOf::new(IncidentIter::new(&es)).fold(Vec::new(), |mut a, i| {
            a.push(i.index());
            a
        });
        assert_eq!(folded, ids);
        assert_eq!(EdgeIdsOf::new(IncidentIter::new(&es)).len(), RUN.len());
    }
}
