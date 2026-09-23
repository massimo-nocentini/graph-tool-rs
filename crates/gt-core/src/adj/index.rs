//! Derived indexes: the edge slot table, and the optional `(s,t)` lookup.
//!
//! ## The slot table is not optional ([DESIGN](crate::design) D1)
//!
//! graph-tool makes `_epos` a runtime flag (`_keep_epos`, `graph_adjacency.hh:619`)
//! and then writes `clear_vertex` **twice**, once for each setting
//! (`:1344-1414` and `:1415-1434`). Only one of the two is wrong. Here the
//! slot table always exists, so there is one algorithm, and there is no
//! configuration in which two monomorphisations can disagree about whether an
//! input is valid.
//!
//! The slot additionally carries the **endpoints**, not only the positions.
//! That is what lets [`AdjList::remove_edge`](super::AdjList::remove_edge)
//! take a bare [`EdgeId`]: there is no caller-supplied orientation left to be
//! wrong, which retires the `reverse_edge`/`remove_edge` interaction at
//! `graph_adjacency.hh:578` entirely rather than guarding it.
//!
//! ## Liveness without a second table
//!
//! `_epos` (`:620`) is a bare `vector<pair<uint32_t,uint32_t>>`: it says
//! nothing about whether `idx` names a live edge, and every reader has to have
//! established that some other way. Here [`EdgeSlot::out_pos`] carries the
//! answer. [`MAX_INDEX`] reserves the top raw value, so `Raw::MAX` is not a
//! representable adjacency position and can serve as the tombstone --
//! [`EdgeSlots::endpoints`] and [`EdgeSlots::locate`] are therefore total
//! functions on *any* [`EdgeId`], which is what the `Option` in their return
//! types promises.

use crate::ids::{EdgeId, MAX_INDEX, Raw, VertexId};
use rustc_hash::FxHashMap;

use super::block::{Block, End, Moved};

/// Everything the graph knows about one edge besides its adjacency entries.
///
/// 16 bytes at a 32-bit [`Raw`](crate::ids::Raw). Positions are `Raw`, not a
/// hardcoded `u32`: graph-tool stores `_epos` as `pair<uint32_t,uint32_t>`
/// (`:620`) beneath a `size_t` vertex, and nothing asserts the relation.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EdgeSlot {
    /// Stored source.
    pub src: VertexId,
    /// Stored target.
    pub tgt: VertexId,
    /// Position of this edge's entry in `src`'s out-half.
    pub out_pos: Raw,
    /// Position of this edge's entry in `tgt`'s in-half, relative to the block
    /// start (i.e. including the out-half offset).
    pub in_pos: Raw,
}

const _: () = assert!(size_of::<EdgeSlot>() == 4 * size_of::<Raw>());
// The tombstone below is only unambiguous because the identifier space stops
// one short of the raw width (`ids::MAX_INDEX`). If that ever changes, a
// legitimate adjacency position could collide with `EdgeSlot::TOMB`.
const _: () = assert!(MAX_INDEX < Raw::MAX as usize);

impl EdgeSlot {
    /// The slot of an index that names no live edge.
    ///
    /// The endpoints are arbitrary and are never read: every accessor tests
    /// [`live`](Self::live) first.
    const TOMB: EdgeSlot = EdgeSlot {
        src: VertexId::from_index(0),
        tgt: VertexId::from_index(0),
        out_pos: Raw::MAX,
        in_pos: Raw::MAX,
    };

    /// Whether this slot names a live edge.
    #[inline]
    const fn live(self) -> bool {
        self.out_pos != Raw::MAX
    }

    /// The vertex whose block holds the `end` half, and the position within it.
    ///
    /// This is the whole of the orientation logic, in one place. graph-tool
    /// spreads the same two lines over every caller of `_epos` (`:1208`,
    /// `:1225`, `:1295`, `:1301-1304`, `:1517-1518`), each of which re-derives which
    /// member of the pair goes with which endpoint from a descriptor the
    /// caller supplied -- and `reverse_edge` (`:578`) lets that descriptor
    /// disagree with storage.
    #[inline]
    const fn half(self, end: End) -> (VertexId, Raw) {
        match end {
            End::Out => (self.src, self.out_pos),
            End::In => (self.tgt, self.in_pos),
        }
    }
}

/// Narrow a position or a length to [`Raw`].
///
/// Every value that reaches this is bounded by a container the allocator
/// already refused to grow past [`MAX_INDEX`], so the cast is lossless; the
/// assertion is what keeps that a fact rather than a belief.
#[inline]
fn raw(i: usize) -> Raw {
    debug_assert!(i <= MAX_INDEX, "position exceeds the identifier width");
    i as Raw
}

/// The edge slot table.
#[derive(Clone, Debug, Default)]
pub struct EdgeSlots {
    slot: Vec<EdgeSlot>,
}

impl EdgeSlots {
    /// An empty table.
    #[inline]
    pub const fn new() -> Self {
        EdgeSlots { slot: Vec::new() }
    }

    /// Endpoints of a live edge.
    #[inline]
    pub fn endpoints(&self, id: EdgeId) -> Option<(VertexId, VertexId)> {
        let s = *self.slot.get(id.index())?;
        s.live().then_some((s.src, s.tgt))
    }

    /// Locate one half of an edge, verifying the entry actually found there.
    ///
    /// The verification lives *here*, checked against the correct half, rather
    /// than at the caller. An ad-hoc caller-side guard cannot be correct: it
    /// passes whenever the two halves happen to share a position.
    ///
    /// Concretely: `EdgeSlot { out_pos: 0, in_pos: 0, .. }` is a perfectly
    /// ordinary slot -- the edge is the first out-entry of `src` and, because
    /// `src` has no in-edges of its own, also the entry at offset 0 of `tgt`
    /// (a self-loop on an isolated vertex is the smallest case). A guard that
    /// reads `block.all()[pos]` and compares `idx` therefore succeeds for
    /// `End::In` while having read the **out** entry. The check below selects
    /// the half *first* and treats a position on the wrong side of `out_len`
    /// as a miss, so the two halves can never be confused for one another.
    pub fn locate(&self, blocks: &[Block], id: EdgeId, end: End) -> Option<(VertexId, Raw)> {
        let s = *self.slot.get(id.index())?;
        if !s.live() {
            return None;
        }
        let (owner, pos) = s.half(end);
        let entry = blocks.get(owner.index())?.get(pos, end)?;
        (entry.idx == id).then_some((owner, pos))
    }

    /// Record a new edge.
    pub(crate) fn on_insert(&mut self, id: EdgeId, s: EdgeSlot) {
        debug_assert!(
            s.live(),
            "a live edge cannot sit at the reserved position Raw::MAX"
        );
        let i = id.index();
        if self.slot.len() <= i {
            // Ports `set_epos`'s `soft_resize` (`graph_adjacency.hh:672-677`):
            // the table is indexed by the *edge index range*, not the edge
            // count, so it is sparse after removals and grows only at the top.
            self.slot.resize(i + 1, EdgeSlot::TOMB);
        }
        self.slot[i] = s;
    }

    /// Apply a relocation notification.
    pub(crate) fn on_move(&mut self, m: Moved) {
        let Some(slot) = self.slot.get_mut(m.id.index()) else {
            debug_assert!(false, "relocation names an edge with no slot");
            return;
        };
        // A relocation can name the edge currently being spliced out: removing
        // a self-loop's out-entry promotes its own in-entry across the
        // out/in boundary. The slot is still live at that point, and must be
        // updated, or the second half of the splice locates a stale position.
        debug_assert!(slot.live(), "relocation names a released edge");
        match m.end {
            End::Out => slot.out_pos = m.to,
            End::In => slot.in_pos = m.to,
        }
    }

    /// Forget an edge.
    pub(crate) fn on_remove(&mut self, id: EdgeId) {
        if let Some(slot) = self.slot.get_mut(id.index()) {
            debug_assert!(slot.live(), "double release of an edge slot");
            *slot = EdgeSlot::TOMB;
        } else {
            debug_assert!(false, "release names an edge with no slot");
        }
    }

    /// Rebuild from scratch. Ports `rebuild_epos` (`:679-699`).
    ///
    /// `bound` is the edge *index range* (`_edge_idx_range`), not the edge
    /// count: the table is indexed by edge id and the space is sparse after
    /// removals.
    ///
    /// **Serial, deliberately.** `rebuild_epos` runs its vertex loop under
    /// `#pragma omp parallel for` and the plan's skeleton described that as
    /// "par_iter over blocks, writing disjoint slots". The slots are *not*
    /// disjoint per block: the edge `(s,t)` is written by `s`'s block (the
    /// `out_pos` field) and by `t`'s block (the `in_pos` field). C++ gets away
    /// with it because distinct non-bitfield members are distinct memory
    /// locations; in Rust that decomposition of a `&mut [EdgeSlot]` has no
    /// safe expression, and this crate is `#![forbid(unsafe_code)]`. The safe
    /// parallel forms -- an SoA of atomics, or a bucket-by-edge-id scatter --
    /// both cost a second pass over the edge set, and this is a cold path
    /// (bulk construction and compaction), so the single streaming pass wins.
    pub(crate) fn rebuild(&mut self, blocks: &[Block], bound: usize) {
        self.slot.clear();
        self.slot.resize(bound, EdgeSlot::TOMB);
        for (v, block) in blocks.iter().enumerate() {
            let owner = VertexId::from_index(v);
            let out_len = block.out_degree();
            // `for j in 0..es.size() { if (j < pos) _epos[idx].first = j else
            // _epos[idx].second = j }` (`:689-696`), plus the endpoints, which
            // `_epos` does not carry. Both halves agree about them because
            // `AdjEntry::other` is the *other* endpoint in either half (D2),
            // so the order in which blocks are scanned does not matter.
            for (j, e) in block.all().iter().enumerate() {
                let Some(slot) = self.slot.get_mut(e.idx.index()) else {
                    // `rebuild` is also the repair path, so an entry naming an
                    // id outside the range is dropped rather than panicking.
                    debug_assert!(false, "adjacency names an edge outside the index range");
                    continue;
                };
                let pos = raw(j);
                if j < out_len {
                    slot.src = owner;
                    slot.tgt = e.other;
                    slot.out_pos = pos;
                } else {
                    slot.src = e.other;
                    slot.tgt = owner;
                    slot.in_pos = pos;
                }
            }
        }
    }
}

/// Optional `(source, target) -> edges` index.
///
/// One table keyed on the **ordered pair**, replacing
/// `vector<gt_hash_map<Vertex, vector<Vertex>>>` (`graph_adjacency.hh:624`) --
/// one hash table *object* per vertex, keyed by the source only.
///
/// The asymmetry is what causes the second confirmed defect:
/// `remove_vertex_fast` (`:1471-1535`) patches `_ehash` only through
/// `out_edges(back)` and `out_edges(v)`, so a neighbour `u` that held `back`
/// as a *target* keeps a key naming a dead vertex, and `edge(u, v, g)`
/// afterwards returns false. With no privileged endpoint there is no direction
/// a relabelling can forget.
pub trait Lookup: Default + Clone + Send + Sync + 'static {
    /// Whether this index is live. `false` makes every hook compile away.
    const ENABLED: bool;

    /// Note an edge's existence.
    fn on_link(&mut self, s: VertexId, t: VertexId, id: EdgeId);
    /// Note an edge's removal.
    fn on_unlink(&mut self, s: VertexId, t: VertexId, id: EdgeId);
    /// Every edge from `s` to `t`.
    fn find(&self, s: VertexId, t: VertexId) -> &[EdgeId];
    /// Rebuild from scratch. Ports `rebuild_ehash` (`:720-732`).
    fn rebuild(&mut self, blocks: &[Block]);
}

/// No `(s,t)` index; `find_edge` falls back to scanning the shorter half.
#[derive(Clone, Copy, Default, Debug)]
pub struct NoLookup;

impl Lookup for NoLookup {
    const ENABLED: bool = false;
    #[inline]
    fn on_link(&mut self, s: VertexId, t: VertexId, id: EdgeId) {}
    #[inline]
    fn on_unlink(&mut self, s: VertexId, t: VertexId, id: EdgeId) {}
    #[inline]
    fn find(&self, s: VertexId, t: VertexId) -> &[EdgeId] {
        &[]
    }
    #[inline]
    fn rebuild(&mut self, blocks: &[Block]) {}
}

/// Hash index on the ordered endpoint pair.
#[derive(Clone, Default, Debug)]
pub struct EHash {
    map: FxHashMap<(VertexId, VertexId), Vec<EdgeId>>,
    /// Where each edge sits inside its own bucket.
    ///
    /// The port of `_ehpos` (`graph_adjacency.hh:625`), and the reason
    /// [`on_unlink`](Lookup::on_unlink) is O(1) rather than O(multiplicity):
    /// `remove_ehash` (`:743-752`) swap-removes at the recorded position
    /// instead of searching the bucket. Indexed by edge id, so it is sparse
    /// after removals, exactly like [`EdgeSlots`].
    pos: Vec<Raw>,
}

impl Lookup for EHash {
    const ENABLED: bool = true;

    fn on_link(&mut self, s: VertexId, t: VertexId, id: EdgeId) {
        let bucket = self.map.entry((s, t)).or_default();
        bucket.push(id);
        let at = raw(bucket.len() - 1);
        let i = id.index();
        if self.pos.len() <= i {
            self.pos.resize(i + 1, 0);
        }
        self.pos[i] = at;
    }

    fn on_unlink(&mut self, s: VertexId, t: VertexId, id: EdgeId) {
        let Some(&at) = self.pos.get(id.index()) else {
            debug_assert!(false, "unlink of an edge this index never saw");
            return;
        };
        let at = at as usize;
        let Some(bucket) = self.map.get_mut(&(s, t)) else {
            debug_assert!(false, "unlink names a pair this index never saw");
            return;
        };
        if bucket.get(at) != Some(&id) {
            debug_assert!(false, "`pos` disagrees with the bucket");
            return;
        }
        // `remove_ehash` (`:743-752`), minus its stale write: the C++ sets
        // `_ehpos[es.back()] = pos` before popping, which for `pos ==
        // es.size()-1` records a position for the edge it is about to forget.
        bucket.swap_remove(at);
        if let Some(&now) = bucket.get(at) {
            self.pos[now.index()] = raw(at);
        }
        if bucket.is_empty() {
            // `_ehash[e.s].erase(e.t)` (`:750-751`). Not an optimisation: without
            // it the key set of the index stops being the set of adjacent
            // pairs, which is what `rebuild` reconstructs and what
            // `AdjList::validate` compares against.
            self.map.remove(&(s, t));
        }
    }

    fn find(&self, s: VertexId, t: VertexId) -> &[EdgeId] {
        self.map.get(&(s, t)).map_or(&[], Vec::as_slice)
    }

    fn rebuild(&mut self, blocks: &[Block]) {
        self.map.clear();
        self.pos.clear();
        // The out-half alone is the right domain: every edge is an out-entry
        // of exactly one vertex, so this visits each edge exactly once. It is
        // also literally what `rebuild_ehash` iterates (`:728-730`,
        // `out_edges(v, *this)`).
        for (v, block) in blocks.iter().enumerate() {
            let s = VertexId::from_index(v);
            for e in block.out() {
                self.on_link(s, e.other, e.idx);
            }
        }
    }
}

// ===========================================================================
// U3 — unit tests
//
// `EdgeSlots`'s hooks are `pub(crate)`, `Block`'s mutators are `pub(crate)`
// and `EHash`'s side table is private, so nothing below is reachable from
// `tests/u03_index.rs`; what *is* reachable lives there. `Mini` is the splice
// pair `AdjList` will be built from (U5), written once here so that every
// claim about the slot table is checked against a real adjacency rather than
// against a hand-written table that agrees with itself.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::adj::AdjEntry;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

    fn v(i: usize) -> VertexId {
        VertexId::from_index(i)
    }
    fn e(i: usize) -> EdgeId {
        EdgeId::from_index(i)
    }
    fn slot(src: usize, tgt: usize, out_pos: Raw, in_pos: Raw) -> EdgeSlot {
        EdgeSlot {
            src: v(src),
            tgt: v(tgt),
            out_pos,
            in_pos,
        }
    }

    // -- the slot itself ----------------------------------------------------

    #[test]
    fn slot_is_four_raws_and_the_tombstone_is_unrepresentable() {
        assert_eq!(size_of::<EdgeSlot>(), 4 * size_of::<Raw>());
        assert!(!EdgeSlot::TOMB.live());
        assert!(slot(0, 0, 0, 0).live());
        // The largest position any container can hold is still live, so the
        // tombstone costs no representable state.
        assert!(slot(0, 0, raw(MAX_INDEX), raw(MAX_INDEX)).live());
    }

    #[test]
    fn half_names_the_owning_block() {
        let s = slot(3, 7, 1, 4);
        assert_eq!(s.half(End::Out), (v(3), 1));
        assert_eq!(s.half(End::In), (v(7), 4));
        let loop_ = slot(5, 5, 0, 2);
        assert_eq!(loop_.half(End::Out).0, loop_.half(End::In).0);
    }

    // -- bookkeeping --------------------------------------------------------

    #[test]
    fn insert_move_remove_roundtrip() {
        let mut t = EdgeSlots::new();
        t.on_insert(e(2), slot(1, 4, 0, 3));
        assert_eq!(t.endpoints(e(2)), Some((v(1), v(4))));
        // Sparse: ids below the inserted one are tombstones, not endpoints.
        assert_eq!(t.endpoints(e(0)), None);
        assert_eq!(t.endpoints(e(1)), None);
        // Out of range is a miss, not a panic.
        assert_eq!(t.endpoints(e(3)), None);
        assert_eq!(t.endpoints(e(9_999)), None);

        t.on_move(Moved {
            id: e(2),
            end: End::Out,
            to: 5,
        });
        assert_eq!(t.slot[2].half(End::Out), (v(1), 5));
        assert_eq!(
            t.slot[2].half(End::In),
            (v(4), 3),
            "the other half must be untouched"
        );
        t.on_move(Moved {
            id: e(2),
            end: End::In,
            to: 6,
        });
        assert_eq!(t.slot[2].half(End::In), (v(4), 6));

        t.on_remove(e(2));
        assert_eq!(t.endpoints(e(2)), None);
        assert_eq!(t.slot[2], EdgeSlot::TOMB);
    }

    #[test]
    fn reinsert_after_release_reuses_the_slot() {
        let mut t = EdgeSlots::new();
        t.on_insert(e(0), slot(1, 2, 0, 0));
        t.on_remove(e(0));
        t.on_insert(e(0), slot(3, 4, 7, 9));
        assert_eq!(t.endpoints(e(0)), Some((v(3), v(4))));
        assert_eq!(t.slot.len(), 1, "the table grew on a reused index");
    }

    // -- a real adjacency to check the table against ------------------------

    /// The two splice primitives, and nothing else.
    ///
    /// This is `AdjList::{splice_in, splice_out}` (U5) in miniature. It exists
    /// so that the slot table is verified against an adjacency that was built
    /// the way the graph builds one -- `Block::insert_out`'s push-swap, the
    /// promotion `remove_at` performs across the out/in boundary -- rather
    /// than against positions a test made up.
    struct Mini {
        blocks: Vec<Block>,
        slots: EdgeSlots,
        hash: EHash,
        next: usize,
        live: Vec<(usize, usize, EdgeId)>,
    }

    impl Mini {
        fn new(n: usize) -> Self {
            Mini {
                blocks: vec![Block::new(); n],
                slots: EdgeSlots::new(),
                hash: EHash::default(),
                next: 0,
                live: Vec::new(),
            }
        }

        fn add(&mut self, s: usize, t: usize) -> EdgeId {
            let id = e(self.next);
            self.next += 1;
            let (out_pos, moved) = self.blocks[s].insert_out(AdjEntry {
                other: v(t),
                idx: id,
            });
            if let Some(m) = moved {
                self.slots.on_move(m);
            }
            let in_pos = self.blocks[t].insert_in(AdjEntry {
                other: v(s),
                idx: id,
            });
            self.slots.on_insert(
                id,
                EdgeSlot {
                    src: v(s),
                    tgt: v(t),
                    out_pos,
                    in_pos,
                },
            );
            self.hash.on_link(v(s), v(t), id);
            self.live.push((s, t, id));
            id
        }

        fn remove(&mut self, id: EdgeId) {
            let (s, t) = self.slots.endpoints(id).expect("live");
            let (owner, out_pos) = self
                .slots
                .locate(&self.blocks, id, End::Out)
                .expect("the out half is locatable");
            assert_eq!(owner, s);
            for m in self.blocks[s.index()]
                .remove_at(out_pos, End::Out)
                .into_iter()
                .flatten()
            {
                self.slots.on_move(m);
            }
            // Re-located *after* the out splice, which can have moved it: a
            // self-loop's in-entry is promoted across the boundary by the very
            // call above. This ordering is the whole of `splice_out`'s
            // contract, and reading `in_pos` before the first splice is the
            // aliasing bug `graph_adjacency.hh:1252-1257` leaves to the reader.
            let (owner, in_pos) = self
                .slots
                .locate(&self.blocks, id, End::In)
                .expect("the in half is locatable");
            assert_eq!(owner, t);
            for m in self.blocks[t.index()]
                .remove_at(in_pos, End::In)
                .into_iter()
                .flatten()
            {
                self.slots.on_move(m);
            }
            self.slots.on_remove(id);
            self.hash.on_unlink(s, t, id);
            let k = self.live.iter().position(|x| x.2 == id).expect("live");
            self.live.swap_remove(k);
        }

        /// `check_epos` (`graph_adjacency.hh:701-718`), which graph-tool defines
        /// and never calls (`:698`, `:1226`, `:1277`, `:1306` and `:1433` are
        /// all commented out).
        ///
        /// Both directions: every live edge is locatable in both halves and
        /// the entry found there names it, *and* every entry in every block is
        /// accounted for by exactly one half of one slot.
        fn check(&self) {
            let mut seen = 0usize;
            for &(s, t, id) in &self.live {
                assert_eq!(self.slots.endpoints(id), Some((v(s), v(t))));
                let (o, p) = self.slots.locate(&self.blocks, id, End::Out).unwrap();
                assert_eq!(o, v(s));
                let entry = self.blocks[s].get(p, End::Out).unwrap();
                assert_eq!((entry.idx, entry.other), (id, v(t)));
                let (o, p) = self.slots.locate(&self.blocks, id, End::In).unwrap();
                assert_eq!(o, v(t));
                let entry = self.blocks[t].get(p, End::In).unwrap();
                assert_eq!((entry.idx, entry.other), (id, v(s)));
                seen += 2;
            }
            let entries: usize = self.blocks.iter().map(Block::degree).sum();
            assert_eq!(entries, seen, "blocks hold entries no slot points at");
            // …and the `(s,t)` index agrees with the same edge set.
            let mut model: BTreeMap<(usize, usize), Vec<EdgeId>> = BTreeMap::new();
            for &(s, t, id) in &self.live {
                model.entry((s, t)).or_default().push(id);
            }
            for ((s, t), mut want) in model {
                let mut got = self.hash.find(v(s), v(t)).to_vec();
                want.sort_unstable();
                got.sort_unstable();
                assert_eq!(got, want, "EHash disagrees about ({s},{t})");
            }
            assert_eq!(self.hash.map.len(), self.hash_keys());
            check_ehpos(&self.hash);
        }

        fn hash_keys(&self) -> usize {
            let mut ks: Vec<(usize, usize)> = self.live.iter().map(|&(s, t, _)| (s, t)).collect();
            ks.sort_unstable();
            ks.dedup();
            ks.len()
        }
    }

    /// The shape that gets every part of the slot table wrong when one of them
    /// is: a hub with both halves populated, a self-loop, parallel edges, an
    /// isolated vertex, and a vertex that is only ever a target.
    const MIXED: [(usize, usize); 7] = [(0, 1), (0, 1), (1, 0), (3, 3), (3, 4), (1, 4), (0, 0)];

    #[test]
    fn the_table_tracks_the_push_swap_and_the_boundary_promotion() {
        let mut g = Mini::new(5);
        for &(s, t) in &MIXED {
            g.add(s, t);
            g.check();
        }
        // Vertex 0: out-entries first, then in-entries, with the in-half's
        // head repeatedly displaced to the back by `insert_out`.
        assert_eq!(g.blocks[0].out_degree(), 3);
        assert_eq!(g.blocks[0].in_degree(), 2);
        // Removing in insertion order exercises the promotion across the
        // boundary and the swap within each half.
        let ids: Vec<EdgeId> = g.live.iter().map(|x| x.2).collect();
        for id in ids {
            g.remove(id);
            g.check();
        }
        assert!(g.blocks.iter().all(|b| b.degree() == 0));
    }

    #[test]
    fn locate_disagrees_about_the_block_exactly_when_the_ends_differ() {
        let mut g = Mini::new(4);
        let cross = g.add(2, 3);
        let loop_ = g.add(1, 1);
        g.check();

        let out = g.slots.locate(&g.blocks, cross, End::Out).unwrap();
        let inn = g.slots.locate(&g.blocks, cross, End::In).unwrap();
        assert_eq!(out.0, v(2));
        assert_eq!(inn.0, v(3));
        assert_ne!(out.0, inn.0, "src != tgt must name two blocks");

        let out = g.slots.locate(&g.blocks, loop_, End::Out).unwrap();
        let inn = g.slots.locate(&g.blocks, loop_, End::In).unwrap();
        assert_eq!(out.0, inn.0, "a self-loop lives in one block");
        assert_ne!(
            out.1, inn.1,
            "…but in two positions, one per half, never the same entry"
        );
        assert_eq!(g.blocks[1].get(out.1, End::Out).unwrap().idx, loop_);
        assert_eq!(g.blocks[1].get(inn.1, End::In).unwrap().idx, loop_);
    }

    #[test]
    fn locate_is_total() {
        let mut g = Mini::new(3);
        let kept = g.add(0, 1);
        let gone = g.add(1, 2);
        g.remove(gone);

        // A released id: the slot is a tombstone, and the block position it
        // used to name is now occupied by nothing at all.
        assert_eq!(g.slots.locate(&g.blocks, gone, End::Out), None);
        assert_eq!(g.slots.locate(&g.blocks, gone, End::In), None);
        assert_eq!(g.slots.endpoints(gone), None);
        // An id past the end of the table.
        assert_eq!(g.slots.locate(&g.blocks, e(50), End::Out), None);
        assert_eq!(g.slots.locate(&g.blocks, e(50), End::In), None);
        // A block entry overwritten out from under the slot: the graph
        // truncated to one vertex, or the entry replaced by another edge's.
        assert!(g.slots.locate(&g.blocks, kept, End::Out).is_some());
        assert_eq!(g.slots.locate(&g.blocks[..1], kept, End::In), None);
        assert_eq!(g.slots.locate(&[], kept, End::Out), None);

        // …and the case a caller-side guard gets wrong. Vertex 0 holds one
        // out-entry and one in-entry, so positions 0 and 1 both exist; a slot
        // whose halves agree on a number must still not match the other half.
        g.add(2, 0);
        let mut t = EdgeSlots::new();
        t.on_insert(kept, slot(0, 0, 0, 0));
        assert!(
            t.locate(&g.blocks, kept, End::Out).is_some(),
            "position 0 of vertex 0's out-half really is `kept`"
        );
        assert_eq!(
            t.locate(&g.blocks, kept, End::In),
            None,
            "the same number read against the in-half must not match"
        );
    }

    // -- rebuild ------------------------------------------------------------

    #[test]
    fn rebuild_reproduces_the_incremental_table() {
        let mut g = Mini::new(5);
        for &(s, t) in &MIXED {
            g.add(s, t);
        }
        g.remove(e(1));
        g.remove(e(4));
        g.check();

        let mut fresh = EdgeSlots::new();
        fresh.rebuild(&g.blocks, g.next);
        assert_eq!(fresh.slot.len(), g.next);
        assert_eq!(
            fresh.slot, g.slots.slot,
            "rebuild_epos must reproduce the incremental state exactly"
        );

        let mut fresh_hash = EHash::default();
        fresh_hash.rebuild(&g.blocks);
        check_ehpos(&fresh_hash);
        let keys = |h: &EHash| {
            let mut ks: Vec<_> = h.map.keys().copied().collect();
            ks.sort_unstable();
            ks
        };
        assert_eq!(keys(&fresh_hash), keys(&g.hash));
        for k in keys(&g.hash) {
            let mut a = g.hash.find(k.0, k.1).to_vec();
            let mut b = fresh_hash.find(k.0, k.1).to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "rebuild_ehash disagrees about {k:?}");
        }
    }

    #[test]
    fn rebuild_sizes_to_the_index_range_and_tombstones_the_holes() {
        let mut t = EdgeSlots::new();
        t.on_insert(e(4), slot(0, 0, 0, 0));
        t.rebuild(&[], 6);
        assert_eq!(t.slot.len(), 6);
        assert!(t.slot.iter().all(|s| *s == EdgeSlot::TOMB));
        for i in 0..6 {
            assert_eq!(t.endpoints(e(i)), None);
        }
        t.rebuild(&[Block::new(), Block::new()], 0);
        assert!(t.slot.is_empty());
        assert_eq!(t.endpoints(e(0)), None);
    }

    // -- the random sequence ------------------------------------------------

    #[test]
    fn a_random_mutation_sequence_keeps_every_index_exact() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5107_7ab1e);
        for n in [1usize, 2, 3, 8] {
            let mut g = Mini::new(n);
            for _ in 0..400 {
                if !g.live.is_empty() && rng.random_range(0..3) == 0 {
                    let k = rng.random_range(0..g.live.len());
                    let id = g.live[k].2;
                    g.remove(id);
                } else {
                    g.add(rng.random_range(0..n), rng.random_range(0..n));
                }
                g.check();
            }
            // And the rebuilds agree with the state the hooks maintained.
            let mut fresh = EdgeSlots::new();
            fresh.rebuild(&g.blocks, g.next);
            assert_eq!(fresh.slot, g.slots.slot);
        }
    }

    // -- EHash --------------------------------------------------------------

    /// `pos[id]` is the index of `id` in its own bucket, for every live edge,
    /// and no bucket outlives its last edge.
    fn check_ehpos(h: &EHash) {
        for (key, bucket) in &h.map {
            assert!(
                !bucket.is_empty(),
                "empty bucket for {key:?} was not erased"
            );
            for (i, id) in bucket.iter().enumerate() {
                assert_eq!(
                    h.pos[id.index()] as usize,
                    i,
                    "`pos` disagrees with the bucket for {id:?}"
                );
            }
        }
    }

    #[test]
    fn ehash_swap_removes_and_erases_empty_buckets() {
        let mut h = EHash::default();
        for i in 0..3 {
            h.on_link(v(1), v(2), e(i));
        }
        h.on_link(v(2), v(1), e(3));
        check_ehpos(&h);
        assert_eq!(h.find(v(1), v(2)).len(), 3);
        assert_eq!(h.find(v(2), v(1)), &[e(3)]);
        assert_eq!(h.find(v(2), v(2)), &[], "an absent pair is an empty slice");

        // From the middle: the survivor swapped into the hole must have its
        // recorded position rewritten, or the next unlink reads the wrong one.
        h.on_unlink(v(1), v(2), e(0));
        check_ehpos(&h);
        let mut got: Vec<_> = h.find(v(1), v(2)).to_vec();
        got.sort_unstable();
        assert_eq!(got, vec![e(1), e(2)]);

        h.on_unlink(v(1), v(2), e(1));
        h.on_unlink(v(1), v(2), e(2));
        check_ehpos(&h);
        assert_eq!(h.find(v(1), v(2)), &[]);
        assert!(
            !h.map.contains_key(&(v(1), v(2))),
            "the key outlived its last edge"
        );
        h.on_unlink(v(2), v(1), e(3));
        assert!(h.map.is_empty());
    }

    // Statically off / statically on: a `const` item, so a regression is a
    // build failure rather than a test failure.
    const _: () = assert!(!NoLookup::ENABLED);
    const _: () = assert!(EHash::ENABLED);

    #[test]
    fn nolookup_is_a_zero_sized_no_op() {
        assert_eq!(size_of::<NoLookup>(), 0);
        let mut n = NoLookup;
        n.on_link(v(0), v(1), e(0));
        assert_eq!(n.find(v(0), v(1)), &[]);
        n.on_unlink(v(0), v(1), e(0));
        n.rebuild(&[Block::new()]);
    }
}
