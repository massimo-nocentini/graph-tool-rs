//! U3 — the edge slot table and the `(s,t)` lookup, from outside the crate.
//!
//! Two of the four acceptance properties are structural and can only be
//! checked from in here, because they are about what a *user* of `gt-core`
//! can observe:
//!
//! * [`EdgeSlot`] is four [`Raw`]s wide. That is the number DESIGN.md D12
//!   spends to make `remove_edge` take a bare [`EdgeId`] -- 16 bytes per edge
//!   against graph-tool's 8 (`_epos`, `graph_adjacency.hh:620`) -- and the
//!   reason the non-slot configuration does not ship. A silent growth to 24
//!   (an `Option<EdgeSlot>` liveness flag is the obvious way to get there)
//!   changes the storage budget the design was argued on.
//! * [`EdgeSlots::endpoints`] and [`EdgeSlots::locate`] are **total**. Their
//!   `Option` is not decoration: `edge(s,t,g)` in graph-tool returns a
//!   `{max,max,max}` descriptor paired with a `bool` (`:943`), and every
//!   caller that drops the `bool` gets a descriptor that compares equal to
//!   every other failure (`operator==` is `idx`-only, `:196`). Here a miss is
//!   `None` and a released, out-of-range or stale index is a miss rather than
//!   a panic or a lie.
//!
//! The third -- [`EHash`] against a brute-force `HashMap<(V,V), Vec<E>>` -- is
//! reachable here because `Lookup` is a public trait. The fourth, `locate`'s
//! behaviour against a *populated* adjacency, is not: `Block`'s mutators are
//! `pub(crate)`, so that half lives in `adj::index`'s own test module, built
//! on the same two splice primitives `AdjList` will use.

use std::collections::HashMap;

use gt_core::adj::{Block, EHash, EdgeSlot, EdgeSlots, End, Lookup, NoLookup};
use gt_core::ids::{EdgeId, Raw, VertexId};
use proptest::prelude::*;

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}
fn e(i: usize) -> EdgeId {
    EdgeId::from_index(i)
}

// ===========================================================================
// 1. Layout
// ===========================================================================

#[test]
fn an_edge_slot_is_two_endpoints_and_two_positions_and_nothing_else() {
    assert_eq!(size_of::<EdgeSlot>(), 4 * size_of::<Raw>());
    assert_eq!(align_of::<EdgeSlot>(), align_of::<Raw>());
    #[cfg(not(feature = "wide-index"))]
    assert_eq!(size_of::<EdgeSlot>(), 16, "DESIGN.md D12 quotes 16 bytes");
    // Liveness is carried inside the slot (`out_pos == Raw::MAX`), not beside
    // it: `Option<EdgeSlot>` has no niche to use, so the obvious alternative
    // would cost another `Raw`'s worth of padding per edge.
    assert!(size_of::<Option<EdgeSlot>>() > size_of::<EdgeSlot>());
}

#[test]
fn an_empty_table_costs_one_vec_and_allocates_nothing() {
    let t = EdgeSlots::new();
    assert_eq!(size_of_val(&t), size_of::<Vec<EdgeSlot>>());
    assert_eq!(t.endpoints(e(0)), None);
    // `const fn new` -- the table is constructible in a `static`/`const`,
    // which `AdjList::with_lookup` relies on.
    const _: EdgeSlots = EdgeSlots::new();
}

// ===========================================================================
// 2. Totality
// ===========================================================================

#[test]
fn every_lookup_into_an_empty_table_is_a_miss_not_a_panic() {
    let t = EdgeSlots::new();
    let blocks = [Block::new(), Block::new(), Block::new()];
    for id in [e(0), e(1), e(7), e(1_000_000)] {
        assert_eq!(t.endpoints(id), None);
        assert_eq!(t.locate(&blocks, id, End::Out), None);
        assert_eq!(t.locate(&blocks, id, End::In), None);
        assert_eq!(t.locate(&[], id, End::Out), None);
        assert_eq!(t.locate(&[], id, End::In), None);
    }
}

#[test]
fn an_empty_block_answers_neither_half() {
    // The bound `locate` checks against. An empty block has no position in
    // either half, so a slot pointing anywhere into it is a miss -- which is
    // what makes a stale slot survivable rather than an out-of-bounds read.
    let b = Block::new();
    assert_eq!(b.out_degree(), 0);
    assert_eq!(b.degree(), 0);
    for pos in [0, 1, Raw::MAX] {
        assert_eq!(b.get(pos, End::Out), None);
        assert_eq!(b.get(pos, End::In), None);
    }
}

// ===========================================================================
// 3. `NoLookup` really is nothing
// ===========================================================================

// `ENABLED` is what makes every hook on `NoLookup` compile away, so it is
// pinned as a `const` item: a regression here is a build failure, not a test
// failure.
const _: () = assert!(!NoLookup::ENABLED);
const _: () = assert!(EHash::ENABLED);

#[test]
fn the_disabled_lookup_is_zero_sized_and_statically_off() {
    assert_eq!(size_of::<NoLookup>(), 0);

    let mut n = NoLookup;
    n.on_link(v(0), v(1), e(0));
    n.on_link(v(0), v(1), e(1));
    assert_eq!(
        n.find(v(0), v(1)),
        &[],
        "the disabled index must answer `no edges`, so `find_edge` falls back \
         to the scan rather than believing it"
    );
    n.on_unlink(v(0), v(1), e(0));
    n.rebuild(&[Block::new(), Block::new()]);
    assert_eq!(n.find(v(0), v(1)), &[]);
}

#[test]
fn an_ehash_with_no_edges_answers_every_pair_with_an_empty_slice() {
    let mut h = EHash::default();
    h.rebuild(&[Block::new(), Block::new()]);
    for s in 0..2 {
        for t in 0..2 {
            assert_eq!(h.find(v(s), v(t)), &[]);
        }
    }
}

// ===========================================================================
// 4. `EHash` against a brute-force model
// ===========================================================================

/// The naive index: every live `(s, t, id)` triple, bucketed on the spot.
fn brute_force(
    live: &[(VertexId, VertexId, EdgeId)],
) -> HashMap<(VertexId, VertexId), Vec<EdgeId>> {
    let mut m: HashMap<(VertexId, VertexId), Vec<EdgeId>> = HashMap::new();
    for &(s, t, id) in live {
        m.entry((s, t)).or_default().push(id);
    }
    m
}

const N: usize = 5;

/// `find` agrees with the model on **every** pair, present or absent.
///
/// Bucket *order* is deliberately not compared: `remove_ehash`
/// (`graph_adjacency.hh:748-758`) swap-removes, so the order inside a bucket
/// is a function of the removal history and is not part of the contract. What
/// is part of the contract is the multiset -- `find_edge` picks one of them,
/// and a parallel-edge multiset that has lost or gained a member is a wrong
/// answer whatever the order.
fn agree(h: &EHash, live: &[(VertexId, VertexId, EdgeId)]) -> Result<(), TestCaseError> {
    let model = brute_force(live);
    for s in 0..N {
        for t in 0..N {
            let mut want = model.get(&(v(s), v(t))).cloned().unwrap_or_default();
            let mut got = h.find(v(s), v(t)).to_vec();
            want.sort_unstable();
            got.sort_unstable();
            prop_assert_eq!(got, want, "EHash disagrees about ({}, {})", s, t);
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// A random link/unlink sequence, checked after **every** operation.
    ///
    /// Checking only at the end would miss the failure mode that matters: a
    /// swap-remove that forgets to rewrite the displaced edge's recorded
    /// position corrupts the *next* unlink, not the current one.
    #[test]
    fn ehash_equals_the_brute_force_index_after_every_step(
        ops in prop::collection::vec((0usize..N, 0usize..N, 0u8..3, any::<u16>()), 0..120)
    ) {
        let mut h = EHash::default();
        let mut live: Vec<(VertexId, VertexId, EdgeId)> = Vec::new();
        let mut next = 0usize;

        for (s, t, kind, pick) in ops {
            if kind == 0 && !live.is_empty() {
                // Removal order is arbitrary, which is the point: the index
                // must not depend on edges leaving in the order they arrived.
                let k = pick as usize % live.len();
                let (a, b, id) = live.swap_remove(k);
                h.on_unlink(a, b, id);
            } else {
                let id = e(next);
                next += 1;
                h.on_link(v(s), v(t), id);
                live.push((v(s), v(t), id));
            }
            agree(&h, &live)?;
        }

        // Unlinking everything must leave the index empty for every pair,
        // not merely holding empty buckets: `_ehash[e.s].erase(e.t)` (`:757`)
        // is what keeps the key set equal to the set of adjacent pairs.
        while let Some((a, b, id)) = live.pop() {
            h.on_unlink(a, b, id);
            agree(&h, &live)?;
        }
        for s in 0..N {
            for t in 0..N {
                prop_assert!(h.find(v(s), v(t)).is_empty());
            }
        }
    }
}

#[test]
fn parallel_edges_are_distinct_members_of_one_bucket() {
    // Identity is the `EdgeId` (D2), so `(1,2)` added three times is three
    // edges, and `find` must hand back all three. graph-tool's `edge(s,t,g)`
    // returns one descriptor and a `bool`, which is where the multiplicity
    // goes missing at the API boundary.
    let mut h = EHash::default();
    for i in 0..3 {
        h.on_link(v(1), v(2), e(i));
    }
    let mut got = h.find(v(1), v(2)).to_vec();
    got.sort_unstable();
    assert_eq!(got, vec![e(0), e(1), e(2)]);
    // The ordered pair is the key: the reverse direction is a different edge
    // set, which is the asymmetry `remove_vertex_fast` (`:1471-1535`) loses
    // track of by keying on the source alone.
    assert_eq!(h.find(v(2), v(1)), &[]);
}
