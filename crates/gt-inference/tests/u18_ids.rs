//! U18 — the inference identifier layer, asserted rather than inspected.
//!
//! `ids.rs` has exactly one function body in it. What it *has* is a set of
//! guarantees that U19–U26 spend the rest of the crate assuming, and each one
//! replaces a hand-maintained convention in the C++:
//!
//! * **the niche.** `Option<Group>` must be four bytes, or the whole reason
//!   for `NonZeroU32` over a plain `u32` is gone and `Entry<W>` (U19) grows.
//!   graph-tool spends a full `int64_t` plus a sentinel
//!   (`inference/blockmodel/spec.hh:87-88`), and DESIGN.md §11 tabulates 4
//!   against 8.
//! * **the sentinel arithmetic.** `null_group = numeric_limits<int64_t>::max()`
//!   burns the *top* representable value, so `INT64_MAX - 1` is the last
//!   usable group. `Group::new` must reserve `u32::MAX` the same way — and
//!   must round-trip every other index, across the whole range, not just the
//!   small ones a hand-written test would reach for.
//! * **identity, not revision.** Defect #36: two freshly built states both sit
//!   at epoch 0, so a bare epoch cannot tell them apart and a delta priced
//!   against one commits silently into the other. `Stamp` closes that only if
//!   `StateId::fresh` really is distinct — including across threads, which is
//!   where the C++ `_m_entries_pool[tid]` shape (`blockmodel/state.hh:2546`)
//!   says states are actually built.
//! * **the `Weight` contract.** `count_t = int64_t` (`spec.hh:86`) is signed
//!   *so that deltas can be negative*; `to_f64` is what `entries_dS` prices
//!   through. Both halves are asserted, including the point where `i64` stops
//!   being exactly representable in an `f64`, which the C++ never states and
//!   silently relies on.

use std::collections::HashSet;
use std::sync::mpsc;

use gt_inference::ids::{BEdge, Epoch, Group, Stamp, StateId, Weight};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Bound-as-assertion helpers: the check is the call site's type-check.
// ---------------------------------------------------------------------------

fn assert_copy<T: Copy>() {}
fn assert_send_sync<T: Send + Sync + 'static>() {}
fn assert_weight<W: Weight>() {}

// ---------------------------------------------------------------------------
// Layout (DESIGN.md §11)
// ---------------------------------------------------------------------------

#[test]
fn option_group_is_four_bytes() {
    // The acceptance criterion, stated as a runtime assertion as well as the
    // `const _: () = assert!(..)` in the module itself, so that a build that
    // somehow reached here with a wider `Group` still fails.
    assert_eq!(size_of::<Option<Group>>(), 4, "Option<Group>");
    assert_eq!(size_of::<Group>(), 4, "Group");
    assert_eq!(align_of::<Option<Group>>(), 4, "align_of Option<Group>");

    // graph-tool's equivalent is `int64_t` + a sentinel: 8 bytes, and the
    // sentinel check is the programmer's problem.
    assert!(size_of::<Option<Group>>() < size_of::<i64>());
}

#[test]
fn the_remaining_id_widths() {
    assert_eq!(size_of::<BEdge>(), 4, "BEdge");
    assert_eq!(size_of::<StateId>(), 8, "StateId");
    // The niche again: `StateId` is `NonZeroU64`, so the option is free.
    assert_eq!(size_of::<Option<StateId>>(), 8, "Option<StateId>");
    assert_eq!(size_of::<Epoch>(), 8, "Epoch");
    assert_eq!(size_of::<Stamp>(), 16, "Stamp");
}

#[test]
fn the_id_types_are_copy_and_thread_safe() {
    assert_copy::<Group>();
    assert_copy::<BEdge>();
    assert_copy::<StateId>();
    assert_copy::<Epoch>();
    assert_copy::<Stamp>();
    assert_send_sync::<Group>();
    assert_send_sync::<BEdge>();
    assert_send_sync::<StateId>();
    assert_send_sync::<Epoch>();
    assert_send_sync::<Stamp>();
}

// ---------------------------------------------------------------------------
// Group: the round trip, over the whole range
// ---------------------------------------------------------------------------

/// The acceptance criterion: `Group::new(i).index() == i` for the *full*
/// range, which for a `u32` index means all `0..=u32::MAX - 1`.
///
/// This is an exhaustive 2^32 - 1 sweep, not a sample. It is cheap because the
/// body is two integer ops and the workspace dev profile is optimised; it is
/// worth doing exhaustively because the failure mode being excluded is an
/// off-by-one that a `0..1000` test cannot see.
#[test]
fn group_index_round_trips_over_the_full_range() {
    const CHUNK: u32 = 1 << 20;
    let n_chunks = (u32::MAX / CHUNK) + 1;

    let bad: Option<u32> = (0..n_chunks)
        .into_par_iter()
        .filter_map(|c| {
            let lo = c * CHUNK;
            let hi = lo.saturating_add(CHUNK);
            (lo..hi).find(|&i| match Group::new(i) {
                Some(g) => g.index() != i as usize,
                // `u32::MAX` is the reserved null and is checked separately;
                // any *other* `None` is a round-trip failure.
                None => i != u32::MAX,
            })
        })
        .min();

    assert_eq!(bad, None, "first index that failed to round-trip");
}

/// graph-tool burns the top representable `group_t` on `null_group`
/// (`inference/blockmodel/spec.hh:88`, repeated as `_null_group` at
/// `blockmodel/partition.hh:170`), so `INT64_MAX - 1` is the last usable
/// group. `Group::new` reserves `u32::MAX` for exactly the same reason — but
/// here the reservation is the `Option`, so it cannot be forgotten at a use
/// site the way `entries.hh:250`'s unchecked `auto s = b[u]` forgets it.
#[test]
fn group_new_reserves_the_top_index_as_the_null() {
    assert_eq!(Group::new(u32::MAX), None, "the null group");
    let last = Group::new(u32::MAX - 1).expect("u32::MAX - 1 is a real group");
    assert_eq!(last.index(), (u32::MAX - 1) as usize);
    assert_eq!(Group::new(0).expect("group 0").index(), 0);
}

#[test]
fn group_boundaries_round_trip() {
    let mut probes: Vec<u32> = (0..=64).collect();
    for k in 0..32 {
        let p: u32 = 1 << k;
        probes.extend([p.wrapping_sub(1), p, p.wrapping_add(1)]);
    }
    probes.extend([u32::MAX - 2, u32::MAX - 1]);
    for i in probes {
        if i == u32::MAX {
            continue;
        }
        let g = Group::new(i).unwrap_or_else(|| panic!("Group::new({i}) was None"));
        assert_eq!(g.index(), i as usize, "round trip at {i}");
    }
}

/// `Ord` on `Group` must agree with `Ord` on the index it wraps: the block
/// model canonicalises a pair with `if (r > s) swap(r, s)`
/// (`blockmodel/emat.hh`'s key construction), and a `NonZeroU32` that ordered
/// differently from its index would canonicalise to a different key than the
/// C++ does.
#[test]
fn group_order_and_hash_follow_the_index() {
    let gs: Vec<Group> = (0..1024u32).map(|i| Group::new(i).unwrap()).collect();
    for w in gs.windows(2) {
        assert!(w[0] < w[1], "{:?} !< {:?}", w[0], w[1]);
    }
    let mut shuffled: Vec<Group> = gs.iter().rev().copied().collect();
    shuffled.sort_by_key(|g| g.index());
    assert_eq!(shuffled, gs, "sort by index == sort by Group");

    let set: HashSet<Group> = gs.iter().copied().collect();
    assert_eq!(set.len(), gs.len(), "1024 distinct groups hash distinctly");
    assert!(set.contains(&Group::new(7).unwrap()), "hash matches value");
}

#[test]
fn group_debug_prints_the_index_not_the_representation() {
    // The stored `NonZeroU32` is index + 1; a derived `Debug` would print that
    // and every log line in the crate would be off by one.
    assert_eq!(format!("{:?}", Group::new(0).unwrap()), "g0");
    assert_eq!(format!("{:?}", Group::new(41).unwrap()), "g41");
    assert_eq!(format!("{:?}", Option::<Group>::None), "None");
}

// ---------------------------------------------------------------------------
// StateId / Stamp: defect #36
// ---------------------------------------------------------------------------

#[test]
fn state_id_fresh_is_nonzero_and_strictly_increasing() {
    let mut prev = StateId::fresh();
    assert_ne!(prev.get(), 0, "the NonZeroU64 niche must be real");
    for _ in 0..10_000 {
        let next = StateId::fresh();
        assert!(
            next.get() > prev.get(),
            "fresh() went backwards: {} then {}",
            prev.get(),
            next.get()
        );
        assert_ne!(next, prev);
        prev = next;
    }
}

/// The acceptance criterion: distinct *across threads*. `fetch_add` is what
/// makes this true; a `static mut` counter or a `load`/`store` pair would pass
/// the single-threaded test above and fail here.
#[test]
fn state_id_fresh_is_distinct_across_threads() {
    const THREADS: usize = 16;
    const PER_THREAD: usize = 4_000;

    let (tx, rx) = mpsc::channel::<Vec<StateId>>();
    std::thread::scope(|scope| {
        for _ in 0..THREADS {
            let tx = tx.clone();
            scope.spawn(move || {
                let ids: Vec<StateId> = (0..PER_THREAD).map(|_| StateId::fresh()).collect();
                tx.send(ids).expect("receiver alive");
            });
        }
        drop(tx);
    });

    let mut seen = HashSet::with_capacity(THREADS * PER_THREAD);
    let mut total = 0usize;
    for batch in rx {
        for id in batch {
            total += 1;
            assert_ne!(id.get(), 0);
            assert!(seen.insert(id), "duplicate StateId {}", id.get());
        }
    }
    assert_eq!(total, THREADS * PER_THREAD);
    assert_eq!(seen.len(), total, "all {total} ids distinct");
}

/// Defect #36, stated as an assertion. Two freshly built states both sit at
/// epoch 0; a guard that compared only the revision counter would accept a
/// delta priced against one and replay it into the other. The `Stamp` pair
/// makes that a mismatch.
#[test]
fn a_bare_epoch_cannot_tell_two_fresh_states_apart_but_a_stamp_can() {
    let a = Stamp {
        state: StateId::fresh(),
        epoch: Epoch(0),
    };
    let b = Stamp {
        state: StateId::fresh(),
        epoch: Epoch(0),
    };

    assert_eq!(a.epoch, b.epoch, "both fresh states are at epoch 0");
    assert_ne!(a.state, b.state, "but their identities differ");
    assert_ne!(
        a, b,
        "so the stamps differ -- this is what closes defect #36"
    );

    // And the epoch half still discriminates within one state.
    let a1 = Stamp {
        state: a.state,
        epoch: Epoch(a.epoch.0 + 1),
    };
    assert_ne!(a, a1);
    assert_eq!(
        a,
        Stamp {
            state: a.state,
            epoch: Epoch(0)
        }
    );
}

#[test]
fn epoch_defaults_to_zero_and_orders() {
    assert_eq!(Epoch::default(), Epoch(0));
    assert!(Epoch(0) < Epoch(1));
    assert!(Epoch(u64::MAX - 1) < Epoch(u64::MAX));
}

// ---------------------------------------------------------------------------
// Weight
// ---------------------------------------------------------------------------

/// `W::from_i64(i).to_f64() == i as f64` over the range in which `W` is exact.
fn assert_round_trip<W: Weight>(i: i64, what: &str) {
    let w = W::from_i64(i);
    assert_eq!(w.to_f64(), i as f64, "{what}: from_i64({i}).to_f64()");
}

#[test]
fn weight_impls_are_weights() {
    assert_weight::<i32>();
    assert_weight::<i64>();
    assert_weight::<f64>();
}

/// `i32`'s exact range is the whole type: every `i32` is representable in an
/// `f64` (24 bits of significand short of 53 is plenty), so this sweep has no
/// exceptions. Checked exhaustively at the boundaries and by a strided sweep
/// over the rest.
#[test]
fn weight_i32_round_trips_over_its_exact_range() {
    for i in [
        0i64,
        1,
        -1,
        i32::MAX as i64,
        i32::MIN as i64,
        i32::MAX as i64 - 1,
        i32::MIN as i64 + 1,
        1 << 23,
        -(1 << 23),
        (1 << 24) + 1,
    ] {
        assert_round_trip::<i32>(i, "i32");
    }
    // Strided sweep across the full i32 range: 2^32 / 2^13 = 524288 probes.
    let mut i = i32::MIN as i64;
    while i <= i32::MAX as i64 {
        assert_round_trip::<i32>(i, "i32 sweep");
        i += 1 << 13;
    }
    assert_round_trip::<i32>(i32::MAX as i64, "i32 top");
}

/// `i64`'s *exact* range for `to_f64` is `|i| <= 2^53`: beyond that an `f64`
/// cannot name every integer. graph-tool prices with `double` throughout
/// (`entropy.hh`'s `xlogx`/`lbinom` chain) and never says so; this pins the
/// boundary instead of discovering it as drift in an entropy audit.
#[test]
fn weight_i64_round_trips_within_its_exact_range() {
    const EXACT: i64 = 1 << 53;
    for i in [0i64, 1, -1, 2, -2, EXACT - 1, EXACT, -(EXACT - 1), -EXACT] {
        assert_round_trip::<i64>(i, "i64");
    }
    let mut i = -EXACT;
    while i < EXACT {
        assert_round_trip::<i64>(i, "i64 sweep");
        i += 1 << 40;
    }

    // Just past the boundary, `to_f64` is a rounding — documented, not a bug,
    // and the reason the audit in U24 carries a tolerance.
    let past = EXACT + 1;
    assert_eq!(
        Weight::to_f64(past),
        EXACT as f64,
        "2^53 + 1 rounds down to 2^53 in an f64"
    );
}

#[test]
fn weight_f64_round_trips_and_carries_fractions() {
    for i in [0i64, 1, -1, 1 << 53, -(1 << 53), i64::MAX / 2] {
        assert_round_trip::<f64>(i, "f64");
    }
    // The only impl that is not integral: a fractional weight survives.
    let half = 0.5f64;
    assert_eq!(half.to_f64(), 0.5);
    assert_eq!(half + half, f64::from_i64(1));
}

/// `count_t = int64_t` (`inference/blockmodel/spec.hh:86`) is signed *so that
/// a delta can be negative* — `insert_delta_rnr` in `entries.hh` inserts
/// `-ew` for the outgoing half of a move. The `Neg` bound on `Weight` is what
/// carries that across, and it is why no unsigned count type can implement it.
#[test]
fn weight_algebra_admits_negative_deltas() {
    fn check<W: Weight>(what: &str) {
        assert_eq!(W::ZERO, W::default(), "{what}: ZERO == Default");
        assert_eq!(W::ZERO.to_f64(), 0.0, "{what}: ZERO widens to 0.0");

        let three = W::from_i64(3);
        let five = W::from_i64(5);
        assert_eq!((three + five).to_f64(), 8.0, "{what}: add");
        assert_eq!((three - five).to_f64(), -2.0, "{what}: sub goes negative");
        assert_eq!((-three).to_f64(), -3.0, "{what}: neg");
        assert_eq!(three + (-three), W::ZERO, "{what}: w + (-w) == ZERO");
        assert_eq!(W::ZERO - five, -five, "{what}: 0 - w == -w");
        assert!(-five < W::ZERO, "{what}: negatives compare below zero");
        assert!(three < five, "{what}: order");
    }
    check::<i32>("i32");
    check::<i64>("i64");
    check::<f64>("f64");
}

/// `i32` is the narrow count type of DESIGN.md §13's open question. Its
/// `from_i64` is a narrowing cast, which is exactly C++'s implicit
/// `int64_t -> int32_t` conversion; the point of the test is that the
/// behaviour is *pinned*, so that a later unit cannot quietly swap in a
/// saturating or panicking cast and change what a 64-bit count means when it
/// reaches a 32-bit state.
#[test]
fn weight_i32_from_i64_truncates_like_the_cpp_implicit_conversion() {
    assert_eq!(i32::from_i64(i32::MAX as i64), i32::MAX);
    assert_eq!(i32::from_i64(i32::MIN as i64), i32::MIN);
    assert_eq!(i32::from_i64(i32::MAX as i64 + 1), i32::MIN);
    assert_eq!(i32::from_i64((1i64 << 32) + 7), 7);
    assert_eq!(i32::from_i64(-1), -1);
}
