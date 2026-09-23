//! The adjacency list itself.

use crate::bound::{EdgeBound, VertexBound};
use crate::error::GraphError;
use crate::ids::{EdgeId, GraphId, VertexId};

use super::alloc::EdgeIds;
use super::block::{Block, End};
use super::entry::{AdjEntry, EdgeRef};
use super::index::{EdgeSlot, EdgeSlots, Lookup, NoLookup};
use super::iter::{AllEdges, Edges, InEdges, OutEdges, Vertices};
use crate::ids::MAX_INDEX;

/// A graph: contiguous adjacency, dense edge indices, O(1) endpoint lookup.
///
/// `H` selects whether an `(s,t)` hash index is maintained. It is the only
/// type parameter; the slot table is unconditional (see
/// the private `index` module) and the index width is the crate-wide
/// [`Raw`](crate::ids::Raw) alias rather than a parameter.
///
/// `AdjList` is *not* directed or undirected. graph-tool's base type is not
/// either -- `graph_filtering.hh:70-73` builds `directed_t` by wrapping
/// `adj_list<size_t>` in an adaptor. Directedness is a property of the
/// [`view`](crate::view), and `&AdjList` is the directed one.
#[derive(Clone, Debug)]
pub struct AdjList<H: Lookup = NoLookup> {
    blocks: Vec<Block>,
    ids: EdgeIds,
    slots: EdgeSlots,
    lookup: H,
    id: GraphId,
    /// Reused across `clear_vertex`/`remove_vertex` so that a steady-state
    /// sweep allocates nothing. graph-tool's non-epos `clear_vertex` is an
    /// in-place `remove_if` with zero allocations; a per-call `Vec` of doomed
    /// descriptors costs two allocations per call on a path this library runs
    /// in loops.
    scratch: Vec<EdgeId>,
}

impl AdjList<NoLookup> {
    /// An empty graph.
    pub fn new() -> Self {
        AdjList::with_lookup(NoLookup)
    }

    /// An empty graph with `n` isolated vertices.
    ///
    /// One `Vec` allocation, not `n` calls to
    /// [`add_vertex`](Self::add_vertex): `add_vertex(g, n)`
    /// (`graph_adjacency.hh:1317-1333`) special-cases the bulk form the same
    /// way, `_edges.resize(v + n)`.
    ///
    /// # Panics
    ///
    /// If `n` exceeds the vertex index space, i.e. `n - 1 > MAX_INDEX`. The
    /// C++ has no bound at all: `Vertex` is also the index type and
    /// `null_vertex()` is `numeric_limits<Vertex>::max()` (`:924`), so a graph
    /// grown past that silently collides with the null descriptor.
    pub fn with_vertices(n: usize) -> Self {
        AdjList::with_vertices_and_lookup(n, NoLookup)
    }
}

impl Default for AdjList<NoLookup> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Lookup> AdjList<H> {
    /// An empty graph with the given `(s,t)` index.
    pub fn with_lookup(lookup: H) -> Self {
        AdjList {
            blocks: Vec::new(),
            ids: EdgeIds::new(),
            slots: EdgeSlots::new(),
            lookup,
            id: GraphId::fresh(),
            scratch: Vec::new(),
        }
    }

    /// An empty graph with `n` isolated vertices and the given `(s,t)` index.
    ///
    /// The generic form of [`AdjList::with_vertices`]; `pub(crate)` because
    /// the public constructor's `AdjList::with_vertices(n)` would otherwise
    /// stop inferring `H = NoLookup` at every call site. `ParBuilder::build`
    /// (U6) is the intended caller.
    pub(crate) fn with_vertices_and_lookup(n: usize, lookup: H) -> Self {
        assert!(
            n == 0 || n - 1 <= MAX_INDEX,
            "vertex index space exhausted (max {MAX_INDEX})"
        );
        let mut g = AdjList::with_lookup(lookup);
        g.blocks = vec![Block::new(); n];
        g
    }

    // -- identity and counts ------------------------------------------------

    /// Process-unique identity, carried by every [`Bound`](crate::bound::Bound)
    /// this graph mints.
    #[inline]
    pub const fn graph_id(&self) -> GraphId {
        self.id
    }

    /// Number of vertices. Vertices are dense, so this is also the bound.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.blocks.len()
    }

    /// Number of live edges. Derived from the allocator, never accumulated.
    #[inline]
    pub const fn num_edges(&self) -> usize {
        self.ids.live()
    }

    /// Allocation bound of the vertex index space.
    #[inline]
    pub fn vertex_bound(&self) -> VertexBound {
        VertexBound::new(self.id, self.blocks.len())
    }

    /// Allocation bound of the edge index space. Larger than
    /// [`num_edges`](Self::num_edges) whenever edges have been removed; this,
    /// not the count, is what an edge property map must be sized to.
    #[inline]
    pub fn edge_bound(&self) -> EdgeBound {
        EdgeBound::new(self.id, self.ids.bound())
    }

    // -- incidence ----------------------------------------------------------

    /// One vertex's adjacency block.
    #[inline]
    pub fn block(&self, v: VertexId) -> Option<&Block> {
        self.blocks.get(v.index())
    }

    /// Out-edges of `v`, anchored at `v`.
    ///
    /// A vertex outside the graph yields an empty range rather than reading
    /// out of bounds: `out_edges(v, g)` indexes `g._edges[v]` unconditionally
    /// (`graph_adjacency.hh:1110-1119`).
    #[inline]
    pub fn out_edges(&self, v: VertexId) -> OutEdges<'_> {
        match self.blocks.get(v.index()) {
            Some(b) => OutEdges::new(b.out()),
            None => OutEdges::empty(),
        }
    }
    /// In-edges of `v`, anchored at `v`.
    #[inline]
    pub fn in_edges(&self, v: VertexId) -> InEdges<'_> {
        match self.blocks.get(v.index()) {
            Some(b) => InEdges::new(b.inc()),
            None => InEdges::empty(),
        }
    }
    /// All incident edges of `v`, anchored at `v`, out-half first.
    ///
    /// Ports `_all_edges_out` (`graph_adjacency.hh:1102-1108`), which walks
    /// the whole block; because [`Incident`](super::Incident) is anchored
    /// there is no per-element orientation branch, where
    /// `all_edge_iterator`'s `make_in_or_out_edge` (`:327-341`) does a
    /// `reinterpret_cast` on every dereference to decide one.
    #[inline]
    pub fn all_edges(&self, v: VertexId) -> AllEdges<'_> {
        match self.blocks.get(v.index()) {
            Some(b) => AllEdges::new(b.all()),
            None => AllEdges::empty(),
        }
    }
    /// O(1).
    #[inline]
    pub fn out_degree(&self, v: VertexId) -> usize {
        self.blocks.get(v.index()).map_or(0, Block::out_degree)
    }
    /// O(1).
    #[inline]
    pub fn in_degree(&self, v: VertexId) -> usize {
        self.blocks.get(v.index()).map_or(0, Block::in_degree)
    }
    /// O(1).
    #[inline]
    pub fn degree(&self, v: VertexId) -> usize {
        self.blocks.get(v.index()).map_or(0, Block::degree)
    }

    /// The vertex set.
    #[inline]
    pub fn vertices(&self) -> Vertices {
        Vertices::new(self.blocks.len())
    }

    /// Every edge exactly once, in canonical orientation.
    #[inline]
    pub fn edges(&self) -> Edges<'_> {
        Edges::new(&self.blocks)
    }

    /// Endpoints of a live edge, O(1) from the slot table.
    #[inline]
    pub fn endpoints(&self, e: EdgeId) -> Option<(VertexId, VertexId)> {
        self.slots.endpoints(e)
    }

    /// The first edge from `s` to `t`, if any.
    ///
    /// Returns `Option`, not a `{max,max,max}` descriptor paired with a `bool`
    /// that callers may drop (`graph_adjacency.hh:943, :949, :962`). O(1) with
    /// [`EHash`](super::EHash), otherwise a scan of the shorter half.
    pub fn find_edge(&self, s: VertexId, t: VertexId) -> Option<EdgeRef> {
        let sb = self.blocks.get(s.index())?;
        let tb = self.blocks.get(t.index())?;
        if H::ENABLED {
            // `iter->second.front()` (`graph_adjacency.hh:949`): the bucket's
            // first id, which is the oldest surviving parallel edge.
            return self
                .lookup
                .find(s, t)
                .first()
                .map(|&id| EdgeRef::new(id, s, t));
        }
        // `:952-971`: scan whichever half is shorter. The two arms answer the
        // same question, so they must agree; they do here because the entry's
        // `other` is the *other* endpoint in either half (D2), where the C++
        // rebuilds an `edge_descriptor(s, t, idx)` per arm from the arguments
        // and could not disagree even if storage did.
        let hit = if sb.out_degree() < tb.in_degree() {
            sb.out().iter().find(|e| e.other == t)
        } else {
            tb.inc().iter().find(|e| e.other == s)
        };
        hit.map(|e| EdgeRef::new(e.idx, s, t))
    }

    // -- mutation -----------------------------------------------------------

    /// Append an isolated vertex.
    ///
    /// `Block::new()` does not allocate, so an isolated vertex costs the 32
    /// bytes of its record and no malloc.
    pub fn add_vertex(&mut self) -> Result<VertexId, GraphError> {
        let v = VertexId::new(self.blocks.len())
            .ok_or(GraphError::VertexIdSpaceExhausted { max: MAX_INDEX })?;
        self.blocks.push(Block::new());
        Ok(v)
    }

    /// Add an edge. O(1) amortised in both halves.
    ///
    /// Both endpoints are checked *before* an index is taken, so a failed call
    /// leaves the index space untouched. `add_edge` (`:1190`) calls
    /// `get_free_idx()` first and indexes `g._edges[s]` afterwards.
    pub fn add_edge(&mut self, s: VertexId, t: VertexId) -> Result<EdgeRef, GraphError> {
        let n = self.blocks.len();
        if s.index() >= n {
            return Err(GraphError::NoSuchVertex(s));
        }
        if t.index() >= n {
            return Err(GraphError::NoSuchVertex(t));
        }
        let id = self.ids.alloc()?;
        match self.splice_in(id, s, t) {
            Ok(e) => {
                self.audit();
                Ok(e)
            }
            Err(err) => {
                // Unreachable given the bounds above, and written anyway: an
                // index taken and not linked is a leak the allocator's own
                // invariant (`live() == bound() - |free|`) would not notice.
                self.ids.release(id);
                Err(err)
            }
        }
    }

    /// Remove an edge, by identity.
    ///
    /// Takes an [`EdgeId`], not a descriptor: the endpoints come from the slot
    /// table, so there is no caller-supplied orientation that can be wrong.
    pub fn remove_edge(&mut self, e: EdgeId) -> Result<(), GraphError> {
        self.splice_out(e)?;
        // The count is `EdgeIds::live()`, moved only here and in `alloc`.
        // `remove_edge` (`:1310-1312`) instead decrements `_n_edges` at the
        // call site, which is why `clear_vertex` can and does get it wrong
        // (`:1403-1410`, defect #1).
        self.ids.release(e);
        self.audit();
        Ok(())
    }

    /// Remove every edge incident to `v`, leaving `v` in place.
    #[inline]
    pub fn clear_vertex(&mut self, v: VertexId) -> Result<(), GraphError> {
        self.clear_vertex_where(v, |_, _| true)
    }

    /// Remove the incident edges of `v` that satisfy `pred`.
    ///
    /// One algorithm: collect the doomed identities into the reusable scratch
    /// buffer, then call [`remove_edge`](Self::remove_edge) on each. graph-tool
    /// has two bodies for this selected by a runtime flag
    /// (`graph_adjacency.hh:1343-1414` and `:1416-1434`); only one of them is
    /// wrong, which is the point.
    /// `pred` is called once per *edge*, as `pred(id, other)`, where `other`
    /// is the endpoint that is not `v` -- `v` itself for a self-loop. A
    /// self-loop occupies both halves of `v`'s block and the C++ offers it to
    /// the predicate **twice** (`:1411-1419`: `pred(ed)` is evaluated before
    /// the `j >= pos && e.first == v` guard discards the in-half occurrence),
    /// which a `FnMut` with a side effect can observe. Here the in-half
    /// occurrence of a self-loop is skipped before the call, so a predicate
    /// sees each incident edge exactly once; the *set* of removed edges is the
    /// same, because the C++ discards that occurrence regardless of the answer.
    pub fn clear_vertex_where<F>(&mut self, v: VertexId, mut pred: F) -> Result<(), GraphError>
    where
        F: FnMut(EdgeId, VertexId) -> bool,
    {
        if v.index() >= self.blocks.len() {
            return Err(GraphError::NoSuchVertex(v));
        }

        // `mem::take` rather than a fresh `Vec`: the buffer keeps its capacity
        // across calls, and the empty `Vec` left behind has no allocation to
        // free. A per-call `Vec<edge_descriptor>` -- which is literally what
        // `:1421-1424` does, `res.reserve(es.size())` -- costs two allocations
        // per call on a path that runs once per vertex.
        let mut scratch = core::mem::take(&mut self.scratch);
        scratch.clear();
        {
            let block = &self.blocks[v.index()];
            let out_len = block.out_degree();
            for (j, e) in block.all().iter().enumerate() {
                if j >= out_len && e.other == v {
                    // The out-half occurrence of this self-loop already
                    // decided its fate.
                    continue;
                }
                if pred(e.idx, e.other) {
                    scratch.push(e.idx);
                }
            }
        }

        let mut outcome = Ok(());
        for &id in &scratch {
            if let Err(err) = self.remove_edge(id) {
                outcome = Err(err);
                break;
            }
        }

        scratch.clear();
        self.scratch = scratch;
        outcome?;
        self.audit();
        Ok(())
    }

    /// Remove `v`, moving the last vertex into its slot and relabelling.
    ///
    /// Relabelling is expressed as unlink-then-relink through the two splice
    /// primitives, so both the slot table and the `(s,t)` index are notified
    /// for *both* directions. `remove_vertex_fast` (`:1471-1535`) instead
    /// patches endpoints in place and touches `_ehash` only via
    /// `out_edges(back)` and `out_edges(v)`, leaving every neighbour that held
    /// `back` as a target keyed on a dead vertex. This is slower and it is why
    /// the class is gone rather than merely absent today.
    pub fn swap_remove_vertex(&mut self, v: VertexId) -> Result<(), GraphError> {
        let n = self.blocks.len();
        if v.index() >= n {
            return Err(GraphError::NoSuchVertex(v));
        }
        // `clear_vertex(v, g)` first, exactly as `:1475`.
        self.clear_vertex(v)?;

        let back = VertexId::from_index(n - 1);
        if v != back {
            // Collect `back`'s incident identities before touching anything:
            // every splice below relocates entries within this very block.
            // A self-loop is named once, by its out-half entry.
            let mut scratch = core::mem::take(&mut self.scratch);
            scratch.clear();
            {
                let block = &self.blocks[back.index()];
                let out_len = block.out_degree();
                for (j, e) in block.all().iter().enumerate() {
                    if j >= out_len && e.other == back {
                        continue;
                    }
                    scratch.push(e.idx);
                }
            }
            let outcome = self.relabel(&scratch, back, v);
            scratch.clear();
            self.scratch = scratch;
            outcome?;
        }

        let emptied = self.blocks.pop();
        debug_assert_eq!(
            emptied.map_or(0, |b| b.degree()),
            0,
            "the vacated block still holds entries"
        );
        self.audit();
        Ok(())
    }

    /// Move every edge in `ids` off `from` and onto `onto`.
    ///
    /// Unlink then relink, per edge, through the two primitives -- so the slot
    /// table *and* the `(s,t)` index are notified for both directions and for
    /// both endpoints. `remove_vertex_fast` (`:1497-1522`) patches
    /// `es[i].first` in place and repairs `_ehash` only through
    /// `out_edges(back)` and `out_edges(v)` (`:1479-1484`, `:1524-1529`), so a
    /// neighbour `u` that held `back` as a **target** keeps a bucket keyed on
    /// a vertex that no longer exists and `edge(u, v, g)` answers false
    /// afterwards. That is defect #3, and it is a consequence of `_ehash`
    /// being keyed on the source alone (`:624`); keying on the ordered pair
    /// leaves no direction to forget.
    ///
    /// The cost is the stated loss: two splices per incident edge where the
    /// C++ writes one word.
    fn relabel(
        &mut self,
        ids: &[EdgeId],
        from: VertexId,
        onto: VertexId,
    ) -> Result<(), GraphError> {
        for &id in ids {
            let (s, t) = self.splice_out(id)?;
            let s = if s == from { onto } else { s };
            let t = if t == from { onto } else { t };
            self.splice_in(id, s, t)?;
        }
        Ok(())
    }

    /// Release excess capacity across all blocks.
    ///
    /// Ports the per-vertex `_edges[i].second.shrink_to_fit()` and the
    /// `_edges.shrink_to_fit()` of `shrink_to_fit` (`graph_adjacency.hh:926-960`).
    /// It does **not** recompute the edge index range: `:928-935` sets
    /// `_edge_idx_range = max(idx) + 1` over the adjacency and drops the free
    /// list above it, which reclaims the tail of the space while every
    /// interior hole survives -- and every edge property map keeps paying for
    /// them. Renumbering is [`EdgeIds::compact`](super::EdgeIds::compact),
    /// which returns the permutation so property maps can follow.
    pub fn shrink_to_fit(&mut self) {
        for b in &mut self.blocks {
            b.shrink();
        }
        self.blocks.shrink_to_fit();
        self.scratch.shrink_to_fit();
    }

    // -- the two mutation primitives ---------------------------------------

    /// Link an already-allocated edge id into both halves.
    ///
    /// `add_edge`'s body from `:1195` on, in order: out-half first (which may
    /// displace the in-half's first entry -- the O(1) push-swap of
    /// `:1203-1210`), *then* the in-half, then the derived indexes. The
    /// displaced entry's notification is applied before the second splice
    /// runs, because for a self-loop the second splice is into the same block.
    fn splice_in(&mut self, id: EdgeId, s: VertexId, t: VertexId) -> Result<EdgeRef, GraphError> {
        let n = self.blocks.len();
        if s.index() >= n {
            return Err(GraphError::NoSuchVertex(s));
        }
        if t.index() >= n {
            return Err(GraphError::NoSuchVertex(t));
        }

        let (out_pos, displaced) =
            self.blocks[s.index()].insert_out(AdjEntry { other: t, idx: id });
        if let Some(m) = displaced {
            // `g.get_epos(s_es.back().second).second = s_es.size() - 1`
            // (`:1208-1209`), as a value rather than a reference into `_epos`.
            self.slots.on_move(m);
        }
        let in_pos = self.blocks[t.index()].insert_in(AdjEntry { other: s, idx: id });

        self.slots.on_insert(
            id,
            EdgeSlot {
                src: s,
                tgt: t,
                out_pos,
                in_pos,
            },
        );
        self.lookup.on_link(s, t, id);
        Ok(EdgeRef::new(id, s, t))
    }

    /// Unlink a live edge from both halves, leaving the id allocated.
    ///
    /// The in-half must be re-located *after* the out-half is spliced, because
    /// the out splice can move it (self-loop, or a boundary promotion). The
    /// borrow checker forces that ordering: `remove_at` returns its
    /// notifications by value, so the `&mut Block` ends before the index hooks
    /// run, and two blocks cannot be borrowed mutably at once -- which is
    /// exactly the `s_es`/`t_es` aliasing that `:1243-1249` hides.
    fn splice_out(&mut self, id: EdgeId) -> Result<(VertexId, VertexId), GraphError> {
        let (s, t) = self.slots.endpoints(id).ok_or(GraphError::NoSuchEdge(id))?;
        let (owner, out_pos) = self
            .slots
            .locate(&self.blocks, id, End::Out)
            .ok_or(GraphError::NoSuchEdge(id))?;
        debug_assert_eq!(owner, s, "the slot's source is not the out-half's owner");

        for m in self.blocks[owner.index()]
            .remove_at(out_pos, End::Out)
            .into_iter()
            .flatten()
        {
            self.slots.on_move(m);
        }

        // Re-located here, and not before the splice above, because that
        // splice can have moved it: removing a self-loop's out-entry promotes
        // the loop's *own* in-entry across the out/in boundary, and a
        // boundary promotion moves an unrelated in-entry in every other case.
        // The C++ reads both positions up front through two references
        // (`s_es`, `t_es`, `:1252-1257`) that are the same object when
        // `s == t`, and the second `remove_e` then trusts a `_epos` value the
        // first one rewrote.
        let (owner, in_pos) =
            self.slots
                .locate(&self.blocks, id, End::In)
                .ok_or(GraphError::Invariant(
                    "a live edge's in-half is not locatable",
                ))?;
        debug_assert_eq!(owner, t, "the slot's target is not the in-half's owner");

        for m in self.blocks[owner.index()]
            .remove_at(in_pos, End::In)
            .into_iter()
            .flatten()
        {
            self.slots.on_move(m);
        }

        self.slots.on_remove(id);
        self.lookup.on_unlink(s, t, id);
        Ok((s, t))
    }

    // -- audit --------------------------------------------------------------

    /// Run [`validate`](Self::validate) after a mutation, in debug builds.
    ///
    /// This is DESIGN.md's answer to defect #46 -- `check_epos` exists in
    /// graph-tool and every call site is commented out (`:698`, `:1226`,
    /// `:1277`, `:1306`, `:1433`) -- wired up.
    ///
    /// It carries a size budget, which the skeleton's wording ("after every
    /// mutation") did not. `validate` is O(V + E); running it unconditionally
    /// after each O(1) splice makes every debug build quadratic, which is how
    /// an always-on audit becomes an audit someone turns off. Below the
    /// budget -- every unit test in this port, and any graph small enough for
    /// a human to have written down -- it runs on every single mutation.
    #[inline]
    fn audit(&self) {
        #[cfg(debug_assertions)]
        {
            /// Entries plus vertices. 4096 makes the audit's own cost O(1) in
            /// the size of the graphs it actually guards.
            const BUDGET: usize = 4096;
            if self.blocks.len() + 2 * self.num_edges() <= BUDGET
                && let Err(e) = self.validate()
            {
                panic!("AdjList invariant broken by the preceding mutation: {e}");
            }
        }
    }

    /// Cross-check every derived index against the adjacency.
    ///
    /// This is `check_epos` (`graph_adjacency.hh:686-718`) actually wired up:
    /// graph-tool has the function and comments out every call site (`:698`,
    /// `:1226`, `:1277`, `:1306`, `:1433`). Called by the private `audit`
    /// after every mutation under `debug_assertions`, and by every test.
    ///
    /// Five claims, in order:
    ///
    /// 1. **Both halves.** Every entry of every block is named by the matching
    ///    half of a live slot, at exactly the position stored there, and the
    ///    slot's endpoints agree with the entry's `other`. `check_epos`
    ///    (`:701-717`) only walks `edges(g)` -- the out-halves -- so an
    ///    in-half position that drifted is invisible to it. Checking one half
    ///    is how a `validate` that never inspects the in-half at all ships.
    /// 2. **Injectivity, hence completeness.** Distinct entries in the same
    ///    half map to distinct `(owner, pos)` pairs, so claim 1 makes the
    ///    out-entries an injection into the live slots; with claim 3 it is a
    ///    bijection.
    /// 3. **The count is the adjacency's.** `num_edges()` equals the number of
    ///    out-entries and the number of in-entries. This is the assertion
    ///    defect #1 fails: `:1403-1410` decrements `_n_edges` by two for one
    ///    removed edge, counting `std::remove_if`'s moved-from tail.
    /// 4. **No orphan slots.** The number of live slots over the whole edge
    ///    index range equals `num_edges()`, so a released id cannot keep a
    ///    live slot, and a live id cannot lack entries.
    /// 5. **The `(s,t)` index agrees**, in multiplicity, not merely in
    ///    membership -- and only when one exists, so the `NoLookup` audit
    ///    allocates nothing and costs nothing.
    pub fn validate(&self) -> Result<(), GraphError> {
        let n = self.blocks.len();
        let mut out_entries = 0usize;
        let mut in_entries = 0usize;

        for (vi, block) in self.blocks.iter().enumerate() {
            let owner = VertexId::from_index(vi);
            let out_len = block.out_degree();
            for (j, entry) in block.all().iter().enumerate() {
                if entry.other.index() >= n {
                    return Err(GraphError::Invariant(
                        "an adjacency entry names a vertex outside the graph",
                    ));
                }
                let end = if j < out_len { End::Out } else { End::In };
                let (src, tgt) = self
                    .slots
                    .endpoints(entry.idx)
                    .ok_or(GraphError::Invariant(
                        "an adjacency entry names a released edge",
                    ))?;
                let (holder, pos) = self.slots.locate(&self.blocks, entry.idx, end).ok_or(
                    GraphError::Invariant("an adjacency entry is unreachable from the slot table"),
                )?;
                if holder != owner || pos as usize != j {
                    return Err(GraphError::Invariant(
                        "the slot table records a different position for an entry",
                    ));
                }
                match end {
                    End::Out => {
                        if src != owner || tgt != entry.other {
                            return Err(GraphError::Invariant(
                                "the slot's endpoints disagree with an out-half entry",
                            ));
                        }
                        out_entries += 1;
                    }
                    End::In => {
                        if tgt != owner || src != entry.other {
                            return Err(GraphError::Invariant(
                                "the slot's endpoints disagree with an in-half entry",
                            ));
                        }
                        in_entries += 1;
                    }
                }
            }
        }

        if out_entries != in_entries {
            return Err(GraphError::Invariant(
                "the out-halves and the in-halves hold different numbers of entries",
            ));
        }
        if out_entries != self.num_edges() {
            return Err(GraphError::Invariant(
                "num_edges() disagrees with the adjacency",
            ));
        }

        let mut live_slots = 0usize;
        for i in 0..self.ids.bound() {
            if self.slots.endpoints(EdgeId::from_index(i)).is_some() {
                live_slots += 1;
            }
        }
        if live_slots != self.num_edges() {
            return Err(GraphError::Invariant(
                "the slot table holds entries the adjacency does not",
            ));
        }

        if H::ENABLED {
            for (vi, block) in self.blocks.iter().enumerate() {
                let src = VertexId::from_index(vi);
                for entry in block.out() {
                    let bucket = self.lookup.find(src, entry.other);
                    if !bucket.contains(&entry.idx) {
                        return Err(GraphError::Invariant(
                            "the (s,t) index does not hold an edge the adjacency does",
                        ));
                    }
                    let multiplicity = block
                        .out()
                        .iter()
                        .filter(|x| x.other == entry.other)
                        .count();
                    if bucket.len() != multiplicity {
                        return Err(GraphError::Invariant(
                            "the (s,t) index disagrees with the adjacency about multiplicity",
                        ));
                    }
                    if bucket
                        .iter()
                        .any(|&id| self.slots.endpoints(id) != Some((src, entry.other)))
                    {
                        return Err(GraphError::Invariant(
                            "the (s,t) index holds an edge whose endpoints are not (s,t)",
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}
