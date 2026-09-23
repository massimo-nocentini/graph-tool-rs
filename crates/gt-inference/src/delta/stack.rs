//! Per-thread scratch: the delta buffers for a whole hierarchy.

use gt_core::dir::Dir;

use super::buf::DeltaBuf;
use crate::ids::Weight;

/// One delta buffer per hierarchy level.
///
/// Replaces `EntrySet* _next` (`entries.hh:207`), which is assigned a raw
/// pointer into another *state's* member (`blockmodel/state.hh:983`) and, for
/// parallel moves, into a `std::vector`'s heap buffer that the same code path
/// may `resize()` (`:500`) and `clear(); shrink_to_fit()` (`:513-526`). There
/// is no pointer here, so there is nothing to invalidate.
#[derive(Clone, Debug)]
pub struct DeltaStack<D: Dir, W: Weight> {
    levels: Vec<DeltaBuf<D, W>>,
}

impl<D: Dir, W: Weight> Default for DeltaStack<D, W> {
    fn default() -> Self {
        DeltaStack {
            levels: vec![DeltaBuf::default()],
        }
    }
}

impl<D: Dir, W: Weight> DeltaStack<D, W> {
    /// A stack with `n` levels.
    ///
    /// `n` is `get_L()` (`blockmodel/state.hh:1004`), i.e. the *whole* chain
    /// including the uncoupled state itself -- `count_L` (`:996-1002`) returns
    /// `1` when nothing is coupled, so an ordinary non-nested block state is
    /// `with_levels(1)` and not `with_levels(0)`. The C++ builds the same
    /// chain one `set_next` at a time (`:983`, `:497`): one `EntrySet` per
    /// state, linked by a raw pointer into another state's member. Here the
    /// chain is the `Vec`, so there is no pointer to dangle when the pool
    /// behind it is resized (defect #38, `:495-503`).
    ///
    /// Grown exactly once: the buffers themselves are empty, and each one
    /// sizes its own field table on the first
    /// [`begin`](DeltaBuf::begin).
    ///
    /// # Panics
    ///
    /// If `n == 0`. A stack with no levels is not a smaller hierarchy, it is
    /// a hierarchy whose bottom state has nowhere to record: every later call
    /// -- `level(0)`, `below_above(0)`, the first `touch` -- would fail with
    /// an index panic pointing at this type instead of at the caller that
    /// asked for zero. `count_L` has no zero case either.
    pub fn with_levels(n: usize) -> Self {
        assert!(
            n > 0,
            "a delta stack needs at least one level; `count_L` \
             (blockmodel/state.hh:996) counts the uncoupled state itself"
        );
        DeltaStack {
            levels: (0..n).map(|_| DeltaBuf::default()).collect(),
        }
    }

    /// Number of levels.
    #[inline]
    pub fn len(&self) -> usize {
        self.levels.len()
    }
    /// Whether the stack is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// One level.
    #[inline]
    pub fn level(&self, l: usize) -> &DeltaBuf<D, W> {
        &self.levels[l]
    }

    /// One level, mutably.
    #[inline]
    pub fn level_mut(&mut self, l: usize) -> &mut DeltaBuf<D, W> {
        &mut self.levels[l]
    }

    /// Level `l` shared and level `l + 1` mutably, at the same time.
    ///
    /// `split_at_mut` is how the borrow checker is *told* that "read level
    /// `l`, write level `l + 1`" is disjoint -- the exact disjointness
    /// `propagate_entries` (`state.hh:1099-1117`) assumes and expresses with a
    /// raw pointer.
    ///
    /// # Panics
    ///
    /// If `l + 1 >= len()`. The top level of a hierarchy has nothing above
    /// it, which `propagate_entries` expresses by recursing only through
    /// `visit_coupled_if` (`state.hh:1119-1126`), whose body is skipped for
    /// the `monostate` alternative (`:965-974`); here the caller owes the
    /// same check.
    pub fn below_above(&mut self, l: usize) -> (&DeltaBuf<D, W>, &mut DeltaBuf<D, W>) {
        let (a, b) = self.levels.split_at_mut(l + 1);
        (&a[l], &mut b[0])
    }
}

/// Caller-owned scratch for one thread.
///
/// **The ownership inversion.** The buffer stops being
/// `BlockState::_m_entries` (`blockmodel/state.hh:2545`) and becomes the
/// caller's, which is what `_m_entries_pool[tid]` (`:463`) already is. This
/// alone removes the `&mut self` / `&mut self._m_entries` alias that is the
/// whole borrow problem; nothing else in the design is load-bearing for it.
///
/// Give each rayon worker its own through `map_init`, never a `Vec` indexed by
/// thread id: that removes `set_concurrent`'s pool resizing along with the
/// pointer-invalidation window it opens.
#[derive(Clone, Debug, Default)]
pub struct Workspace<D: Dir, W: Weight> {
    /// The per-level buffers.
    pub stack: DeltaStack<D, W>,
}

impl<D: Dir, W: Weight> Workspace<D, W> {
    /// Scratch for an `n`-level hierarchy.
    pub fn with_levels(n: usize) -> Self {
        Workspace {
            stack: DeltaStack::with_levels(n),
        }
    }
}
