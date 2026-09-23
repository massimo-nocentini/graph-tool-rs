//! The dense edge-index allocator.

use crate::error::GraphError;
use crate::ids::{EdgeId, MAX_INDEX};

/// Owner of the edge index space.
///
/// Merges graph-tool's three separate fields -- `_n_edges` (`:607`),
/// `_edge_idx_range` (`:608`) and `_free_idx` (`:612`) -- into one type whose
/// only mutators are [`alloc`](Self::alloc) and [`release`](Self::release).
///
/// `num_edges()` is therefore *derived*, never accumulated by caller
/// arithmetic. That deletes an entire defect class: `clear_vertex`
/// (`graph_adjacency.hh:1404-1413`) computes
/// `k += count_if(iter, es.begin()+pos, ...)` over the moved-from tail of
/// `std::remove_if`, which by that algorithm's definition holds the *kept*
/// elements; a filtered `clear_vertex` (reachable from
/// `graph_filtered.hh:573-584`) on a vertex with a self-loop plus another
/// out-edge decrements `_n_edges` by two for one removed edge.
///
/// ## The invariant
///
/// `live() == bound() - free.len()` holds after every operation and is
/// `debug_assert`ed there. It is the one relation graph-tool cannot state,
/// because its three fields are `friend`-visible to every mutating free
/// function (`:755-818`) and each of them re-derives the count by hand.
#[derive(Clone, Debug, Default)]
pub struct EdgeIds {
    next: usize,
    free: Vec<EdgeId>,
    live: usize,
}

impl EdgeIds {
    /// An empty allocator.
    #[inline]
    pub const fn new() -> Self {
        EdgeIds {
            next: 0,
            free: Vec::new(),
            live: 0,
        }
    }

    /// Take an index from the free list, or extend the range.
    ///
    /// Ports `get_free_idx` (`graph_adjacency.hh:627-655`), serial branch:
    /// LIFO off the back of the free list, else `_edge_idx_range++`. The
    /// concurrent branch (`_free_idx_m[get_thread_num()]`) has no counterpart
    /// -- [DESIGN](crate::design) defect #51: `&mut AdjList` refuses concurrent mutation and
    /// `ParBuilder` replaces the bulk case deterministically, so there is no
    /// per-thread free list whose contents depend on the schedule.
    ///
    /// Exhaustion is a `Result`, not a wrap: `get_free_idx` returns
    /// `_edge_idx_range++` with no bound at all, and `Vertex` is also the
    /// index type, so the range silently collides with `null_vertex()` at the
    /// top of the space.
    pub fn alloc(&mut self) -> Result<EdgeId, GraphError> {
        let id = match self.free.pop() {
            Some(id) => id,
            None => {
                let id = EdgeId::new(self.next)
                    .ok_or(GraphError::EdgeIdSpaceExhausted { max: MAX_INDEX })?;
                self.next += 1;
                id
            }
        };
        self.live += 1;
        debug_assert_eq!(self.live, self.next - self.free.len());
        Ok(id)
    }

    /// Return an index to the free list.
    ///
    /// Ports `put_free_index` (`graph_adjacency.hh:659-664`). `id` must be
    /// live; releasing an id twice, or one this allocator never issued, is a
    /// caller bug that [`AdjList::validate`](super::AdjList::validate) and the
    /// `debug_assert`s here exist to catch. In a release build the count
    /// saturates rather than wrapping, because a `live` of `usize::MAX` would
    /// propagate into every `num_edges()`-sized allocation in the port.
    pub fn release(&mut self, id: EdgeId) {
        debug_assert!(
            id.index() < self.next,
            "release of an id outside the allocated range"
        );
        debug_assert!(self.live > 0, "release with no live edge");
        self.free.push(id);
        self.live = self.live.saturating_sub(1);
        debug_assert_eq!(self.live, self.next - self.free.len());
    }

    /// Number of live edges. This is `num_edges(g)`.
    #[inline]
    pub const fn live(&self) -> usize {
        self.live
    }

    /// Size of the index space. This is `get_edge_index_range(g)`, and it is
    /// what a property map must be sized to -- not [`live`](Self::live), since
    /// the space is sparse after removals.
    #[inline]
    pub const fn bound(&self) -> usize {
        self.next
    }

    /// Largest representable index.
    #[inline]
    pub const fn max_index() -> usize {
        MAX_INDEX
    }

    /// Compact the index space, remapping live edges onto `0..live`.
    ///
    /// Returns the permutation so that callers can rewrite edge property maps.
    ///
    /// ## What "the permutation" means
    ///
    /// The result is indexed by the **old** id and has length the **old**
    /// [`bound`](Self::bound); `p[old.index()]` is `old`'s new id. It is a
    /// total permutation of `0..old_bound`, not a partial map with holes:
    ///
    /// * the live ids go, order-preservingly, onto `0..live()`;
    /// * the ids that were on the free list go onto `live()..old_bound`, also
    ///   order-preservingly.
    ///
    /// So an edge property map is rewritten by permuting in place over its
    /// whole length and then truncating to the new [`bound`](Self::bound) --
    /// no validity test per element, and no sentinel id to represent "was
    /// free", which is what a `live`-length map would have needed. The
    /// adjacency entries and the slot table are rewritten by the same table:
    /// `entry.idx = p[entry.idx.index()]`.
    ///
    /// Afterwards `bound() == live()` and the free list is empty.
    ///
    /// There is no counterpart in graph-tool. `shrink_to_fit` (`:541-576`)
    /// recomputes `_edge_idx_range` as `max(idx) + 1` over the adjacency and
    /// drops the free-list entries above it, which reclaims the tail of the
    /// index space but never renumbers an edge -- so a graph that had its
    /// *interior* edges removed keeps every hole, and every edge property map
    /// keeps paying for them.
    pub fn compact(&mut self) -> Vec<EdgeId> {
        let bound = self.next;
        let mut freed = vec![false; bound];
        for id in &self.free {
            debug_assert!(id.index() < bound, "free id outside the range");
            freed[id.index()] = true;
        }

        let mut perm = Vec::with_capacity(bound);
        let mut live = 0usize;
        // The dead ids are numbered after the live ones, so the map is a
        // permutation of the whole old range and the caller needs no "is this
        // id live" test while rewriting.
        let mut dead = self.live;
        for is_free in freed {
            if is_free {
                perm.push(EdgeId::from_index(dead));
                dead += 1;
            } else {
                perm.push(EdgeId::from_index(live));
                live += 1;
            }
        }
        // Fires if an id was released twice: the free list is then longer than
        // the set of ids it names, and the two counters cross.
        debug_assert_eq!(live, self.live, "free list disagrees with the count");
        debug_assert_eq!(dead, bound, "free list disagrees with the range");

        self.next = self.live;
        self.free.clear();
        perm
    }
}
