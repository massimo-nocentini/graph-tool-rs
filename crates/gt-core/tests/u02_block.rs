//! U2 -- the adjacency block and the edge-id allocator.
//!
//! Two things are asserted here and nowhere else:
//!
//! * **the layout numbers of `gt_core::design` section 11**, literally. `AdjEntry` at
//!   8 bytes against `pair<vertex_t, vertex_t>` at 16, and `Block` at 32
//!   against `pair<size_t, vector<pair<size_t,size_t>>>`, are the whole of the
//!   "2x on the adjacency stream" claim in section 12. A `usize` that crept
//!   into either would double the memory stream that every BFS, triangle count
//!   and SBM neighbour sweep is bound by, and nothing else in the tree would
//!   notice.
//! * **the edge-id allocator's invariant**, `live() == bound() - |free|`, over
//!   random alloc/release sequences. `num_edges()` is this number
//!   (`adj/list.rs`), so the defect the invariant forecloses is #1: the C++
//!   accumulates `_n_edges` at each call site instead, and `clear_vertex`
//!   (`graph_adjacency.hh:1404-1413`) decrements it by two for one removed
//!   edge by counting `std::remove_if`'s moved-from tail.
//!
//! The splice sequences live in `src/adj/block.rs`'s own test module:
//! `insert_out`, `insert_in` and `remove_at` are `pub(crate)` because
//! gt_core::design section 3 makes mutation "exactly two private primitives", and
//! that encapsulation is worth more than the convenience of reaching them from
//! here.

use gt_core::adj::{Block, EdgeIds, End};
// The layout assertions below are `#[cfg(not(feature = "wide-index"))]`: the
// numbers in `gt_core::design` section 11 are the 32-bit-index ones. These four types
// are named only by those assertions, so the import carries the same gate --
// otherwise `--all-features` builds this file with four unused imports.
#[cfg(not(feature = "wide-index"))]
use gt_core::adj::{AdjEntry, EdgeSlot, Incident, Moved};
use gt_core::ids::{EdgeId, MAX_INDEX, Raw};
use proptest::prelude::*;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// `gt_core::design` section 11, the rows this unit owns. Written as literals rather
/// than as a relation between `size_of`s, because a relation that holds for
/// the wrong reason is exactly how `_epos`'s `uint32_t` came to sit under a
/// `size_t` vertex (`graph_adjacency.hh:620`, defect #5).
#[test]
#[cfg(not(feature = "wide-index"))]
fn the_layout_numbers_are_the_ones_in_the_ledger() {
    assert_eq!(size_of::<AdjEntry>(), 8, "AdjEntry: 8 against C++'s 16");
    assert_eq!(size_of::<Block>(), 32, "Block: parity with the C++ record");
    assert_eq!(size_of::<EdgeSlot>(), 16);
    assert_eq!(size_of::<Incident>(), 8);
    assert_eq!(align_of::<AdjEntry>(), 4);
    // Eight entries per 64-byte line, where `adj_list<size_t>` gets four.
    assert_eq!(64 / size_of::<AdjEntry>(), 8);
}

/// A `Moved` is a machine word and a half and is trivially copyable: it is
/// returned by value from every splice precisely so that the `&mut Block`
/// borrow can end before the derived indexes are touched, and that is only
/// free if the value is this small.
#[test]
#[cfg(not(feature = "wide-index"))]
fn a_notification_costs_nothing_to_return() {
    assert_eq!(size_of::<Moved>(), 12);
    // `End`'s two variants leave a niche, so the `Option` is free and the
    // pair that `remove_at` returns is three words, not five.
    assert_eq!(size_of::<Option<Moved>>(), 12);
    assert_eq!(size_of::<[Option<Moved>; 2]>(), 24);
    fn assert_copy<T: Copy>() {}
    assert_copy::<Moved>();
    assert_copy::<End>();
    assert_copy::<AdjEntry>();
}

// ---------------------------------------------------------------------------
// The empty block
// ---------------------------------------------------------------------------

/// `Vec::new()` does not allocate, so an isolated vertex costs the 32 bytes of
/// its record and no malloc -- the property `gt_core::design` section 3 cites when it
/// rejects an inline small-vector buffer.
#[test]
fn an_isolated_vertex_is_empty_in_both_halves() {
    let b = Block::new();
    assert_eq!(b.degree(), 0);
    assert_eq!(b.out_degree(), 0);
    assert_eq!(b.in_degree(), 0);
    assert!(b.out().is_empty());
    assert!(b.inc().is_empty());
    assert!(b.all().is_empty());
    assert_eq!(b.get(0, End::Out), None);
    assert_eq!(b.get(0, End::In), None);
    assert_eq!(b.get(Raw::MAX, End::In), None);
    assert_eq!(Block::default().degree(), 0);
}

// ---------------------------------------------------------------------------
// The allocator
// ---------------------------------------------------------------------------

#[test]
fn an_empty_allocator_has_nothing_and_bounds_nothing() {
    let ids = EdgeIds::new();
    assert_eq!(ids.live(), 0);
    assert_eq!(ids.bound(), 0);
    assert_eq!(EdgeIds::max_index(), MAX_INDEX);
    assert_eq!(EdgeIds::default().bound(), 0);
}

/// Fresh ids are dense and ascending: `_edge_idx_range++`
/// (`graph_adjacency.hh:640`).
#[test]
fn fresh_ids_are_dense_and_ascending() {
    let mut ids = EdgeIds::new();
    for i in 0..8 {
        assert_eq!(ids.alloc().expect("space"), EdgeId::from_index(i));
        assert_eq!(ids.live(), i + 1);
        assert_eq!(ids.bound(), i + 1);
    }
}

/// `get_free_idx` pops the **back** of `_free_idx` (`:648-649`), so recycling
/// is LIFO. Ported as written: the order is observable through every edge
/// property map, and `gt-io` emits in `EdgeId` order (defect #52), so a
/// gratuitous change of discipline would change what a round-trip file looks
/// like.
#[test]
fn a_released_id_is_reused_lifo_and_the_bound_does_not_grow() {
    let mut ids = EdgeIds::new();
    let e: Vec<_> = (0..4).map(|_| ids.alloc().expect("space")).collect();

    ids.release(e[1]);
    ids.release(e[3]);
    assert_eq!(ids.live(), 2);
    assert_eq!(ids.bound(), 4, "the index space stays as wide as it was");

    assert_eq!(ids.alloc().expect("space"), e[3]);
    assert_eq!(ids.alloc().expect("space"), e[1]);
    assert_eq!(ids.bound(), 4, "recycling must not extend the range");
    assert_eq!(ids.live(), 4);

    // Only now is the free list empty again.
    assert_eq!(ids.alloc().expect("space"), EdgeId::from_index(4));
    assert_eq!(ids.bound(), 5);
}

/// The bound, not the count, is what an edge property map must be sized to:
/// after interior removals the space is sparse and a map of `live()` entries
/// would be indexed out of bounds by the surviving high ids. That is the
/// shape of defect #6/#7.
#[test]
fn the_bound_outlives_the_count() {
    let mut ids = EdgeIds::new();
    let e: Vec<_> = (0..6).map(|_| ids.alloc().expect("space")).collect();
    for id in e.iter().take(5) {
        ids.release(*id);
    }
    assert_eq!(ids.live(), 1);
    assert_eq!(ids.bound(), 6);
    assert!(ids.bound() > ids.live());
}

/// `compact()` returns a **total** permutation of the old index space: live
/// ids onto `0..live()` order-preservingly, freed ids onto the tail. That is
/// what lets a caller permute an edge property map over its whole length and
/// truncate, with no per-element validity test and no sentinel id.
#[test]
fn compact_is_a_permutation_of_the_old_index_space() {
    let mut ids = EdgeIds::new();
    let e: Vec<_> = (0..10).map(|_| ids.alloc().expect("space")).collect();
    for k in [0usize, 3, 4, 9] {
        ids.release(e[k]);
    }
    let live_before = ids.live();
    let bound_before = ids.bound();
    assert_eq!((live_before, bound_before), (6, 10));

    let perm = ids.compact();

    assert_eq!(perm.len(), bound_before);
    let image: HashSet<usize> = perm.iter().map(|p| p.index()).collect();
    assert_eq!(
        image,
        (0..bound_before).collect::<HashSet<_>>(),
        "not a permutation"
    );

    // The survivors, in order, occupy the new dense prefix.
    let survivors = [1usize, 2, 5, 6, 7, 8];
    for (new, &old) in survivors.iter().enumerate() {
        assert_eq!(perm[old].index(), new);
    }
    // The freed ids are pushed past the new bound, so truncating a permuted
    // property map to `bound()` drops exactly them.
    for k in [0usize, 3, 4, 9] {
        assert!(perm[k].index() >= live_before);
    }

    assert_eq!(ids.live(), live_before, "compaction removes no edge");
    assert_eq!(ids.bound(), live_before, "the space is dense again");
    // The free list went with it: the next id is fresh, not recycled.
    assert_eq!(
        ids.alloc().expect("space"),
        EdgeId::from_index(live_before),
        "the free list must not survive a compaction"
    );
}

#[test]
fn compacting_a_dense_space_is_the_identity() {
    let mut ids = EdgeIds::new();
    for _ in 0..5 {
        ids.alloc().expect("space");
    }
    let perm = ids.compact();
    for (i, p) in perm.iter().enumerate() {
        assert_eq!(p.index(), i);
    }
    assert_eq!((ids.live(), ids.bound()), (5, 5));
}

#[test]
fn compacting_an_empty_space_yields_an_empty_permutation() {
    let mut ids = EdgeIds::new();
    assert!(ids.compact().is_empty());
    assert_eq!((ids.live(), ids.bound()), (0, 0));
}

proptest! {
    /// The acceptance invariant, checked after **every** operation.
    ///
    /// `live()` is `num_edges(g)`. The model here is a plain `Vec` of the ids
    /// believed live; the allocator never sees it. Three things are asserted:
    /// the count agrees with the model, the bound never shrinks under
    /// allocation, and no id is ever handed out twice while live -- which is
    /// the one that a hand-rolled free list gets wrong, and which would alias
    /// two edges onto one slot in every edge property map in the port.
    #[test]
    fn allocation_keeps_its_invariant(ops in prop::collection::vec(any::<(bool, u16)>(), 1..500)) {
        let mut ids = EdgeIds::new();
        let mut live: Vec<EdgeId> = Vec::new();
        let mut bound = 0usize;

        for (is_alloc, arg) in ops {
            if is_alloc || live.is_empty() {
                let id = ids.alloc().expect("u16-many allocations fit");
                prop_assert!(!live.contains(&id), "{id:?} was already live");
                prop_assert!(id.index() < ids.bound());
                live.push(id);
            } else {
                let k = arg as usize % live.len();
                let id = live.swap_remove(k);
                ids.release(id);
            }
            prop_assert_eq!(ids.live(), live.len());
            prop_assert!(ids.bound() >= bound, "the index space must not shrink");
            prop_assert!(ids.bound() >= ids.live());
            bound = ids.bound();

            // Every live id is inside the bound: this is the relation a
            // property map is sized on.
            for id in &live {
                prop_assert!(id.index() < ids.bound());
            }
        }

        // And a compaction of whatever state we ended in is a permutation
        // that lands every survivor inside the new, dense bound.
        let before = ids.live();
        let old_bound = ids.bound();
        let mut old: Vec<usize> = live.iter().map(|e| e.index()).collect();
        old.sort_unstable();

        let perm = ids.compact();
        prop_assert_eq!(perm.len(), old_bound);
        let image: HashSet<usize> = perm.iter().map(|p| p.index()).collect();
        prop_assert_eq!(image.len(), old_bound);
        prop_assert_eq!(ids.bound(), before);
        prop_assert_eq!(ids.live(), before);
        for (new, &o) in old.iter().enumerate() {
            prop_assert_eq!(perm[o].index(), new);
        }
    }
}
