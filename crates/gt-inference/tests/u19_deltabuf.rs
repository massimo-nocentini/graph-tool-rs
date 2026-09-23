//! U19 -- the delta buffer.
//!
//! The acceptance tests for `EntrySet`'s replacement. Three of them are about
//! things the C++ gets wrong or leaves to an unwritten invariant:
//!
//! * `clear()` (`entries.hh:169-176`) re-derives each cell address from
//!   `_rnr`, so it is only correct if it runs *before* the new key is
//!   assigned -- a temporal coupling `set_move` (`:59-65`) happens to honour
//!   and nothing states. [`residue`](begin_with_a_new_key_leaves_no_residue)
//!   pins that `begin` does not care.
//! * the same loop is O(#entries) only because each entry names its cell;
//!   [`cost`](begin_costs_one_write_per_entry_not_one_per_group) pins that
//!   with a counter rather than a clock.
//! * `get_field` (`:108-119`) returns the shared `_dummy` cell for a pair
//!   touching neither endpoint, so two such pairs accumulate into one entry
//!   (defect #4). Here the pair is rejected and nothing is written.

use std::collections::HashMap;

use gt_core::dir::{Dir, Directed, Undirected};
use gt_inference::delta::{DeltaBuf, MoveKey, OutOfPlane};
use gt_inference::ids::{BEdge, Group};

use proptest::prelude::*;

const B: usize = 6;

fn g(i: u32) -> Group {
    Group::new(i).expect("group index fits")
}

/// The pair identity the field table implements: directed pairs are ordered,
/// undirected ones are not, because `get_field_rnr<_, false>` indexes the
/// *out* half for both orientations when `directed` is false
/// (`entries.hh:100-102`).
fn canon<D: Dir>(s: Group, t: Group) -> (Group, Group) {
    if D::DIRECTED || s <= t { (s, t) } else { (t, s) }
}

/// A deterministic before-image, distinguishable per pair.
fn resolve(r: Group, s: Group) -> (Option<BEdge>, i64) {
    if (r.index() + s.index()).is_multiple_of(3) {
        (None, 0)
    } else {
        (
            Some(BEdge((r.index() * 16 + s.index()) as u32)),
            (r.index() * 100 + s.index()) as i64,
        )
    }
}

fn fresh<D: Dir>(mv: MoveKey) -> DeltaBuf<D, i64> {
    let mut d = DeltaBuf::<D, i64>::default();
    d.begin(mv, B);
    d
}

/// Every pair, recorded or not, is queried -- the reference is total.
fn check_against<D: Dir>(d: &DeltaBuf<D, i64>, model: &HashMap<(Group, Group), i64>) {
    for s in 0..B as u32 {
        for t in 0..B as u32 {
            let want = model
                .get(&canon::<D>(g(s), g(t)))
                .copied()
                .unwrap_or_default();
            assert_eq!(d.delta_of(g(s), g(t)), want, "delta_of({s}, {t})");
        }
    }
    assert_eq!(
        d.entries().len(),
        model.len(),
        "one entry per distinct touched pair"
    );
    let mut seen: HashMap<(Group, Group), usize> = HashMap::new();
    for e in d.entries() {
        *seen.entry(canon::<D>(e.r, e.s)).or_default() += 1;
        assert_eq!(d.me_of(e.r, e.s), Some(e.me));
        assert_eq!(d.me_of(e.r, e.s), Some(resolve(e.r, e.s).0));
        assert_eq!(d.mrs_before_of(e.r, e.s), Some(resolve(e.r, e.s).1));
        assert_eq!(d.delta_of(e.r, e.s), e.delta);
    }
    for (k, n) in seen {
        assert_eq!(n, 1, "{k:?} appears {n} times in entries()");
    }
}

/// One generated touch: which endpoint it hangs off, the other group, and the
/// weight. `side == 4` is the out-of-plane case.
fn ops() -> impl Strategy<Value = Vec<(u8, u32, u32, i64)>> {
    prop::collection::vec(
        (0u8..5, 0u32..B as u32, 0u32..B as u32, -4i64..5),
        0..40,
    )
}

fn run<D: Dir>(mv: MoveKey, ops: &[(u8, u32, u32, i64)]) {
    let mut d = fresh::<D>(mv);
    let mut model: HashMap<(Group, Group), i64> = HashMap::new();

    for &(side, a, b, w) in ops {
        let anchor = match side & 1 {
            0 => mv.from,
            _ => mv.to,
        };
        let pair = match (side, anchor) {
            (4, _) => {
                // Both ends away from the plane, unless the move key happens
                // to contain them.
                (g(2 + a % 4), g(2 + b % 4))
            }
            (_, None) => continue,
            (0 | 1, Some(x)) => (x, g(a)),
            (_, Some(x)) => (g(a), x),
        };
        let (s, t) = pair;
        let before = d.entries().to_vec();
        let writes = d.field_writes();
        match d.touch_dyn(s, t, w, &mut |r, q| resolve(r, q)) {
            Ok(()) => {
                *model.entry(canon::<D>(s, t)).or_default() += w;
            }
            Err(OutOfPlane(es, et)) => {
                assert_eq!((es, et), (s, t));
                assert_ne!(mv.from, Some(s));
                assert_ne!(mv.from, Some(t));
                assert_ne!(mv.to, Some(s));
                assert_ne!(mv.to, Some(t));
                // Defect #4: nothing is recorded, nothing is aliased.
                assert_eq!(d.entries(), before.as_slice());
                assert_eq!(d.field_writes(), writes);
            }
        }
    }
    check_against::<D>(&d, &model);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// The directed buffer against a `HashMap<(Group, Group), i64>`.
    #[test]
    fn directed_touches_match_a_hashmap(ops in ops()) {
        run::<Directed>(MoveKey { from: Group::new(0), to: Group::new(1) }, &ops);
    }

    /// The undirected buffer, whose pair identity is unordered.
    #[test]
    fn undirected_touches_match_a_hashmap(ops in ops()) {
        run::<Undirected>(MoveKey { from: Group::new(0), to: Group::new(1) }, &ops);
    }

    /// `single` (`entries.hh:322-332`): a null target is a real move key, not
    /// a degenerate one, and half the plane simply does not exist.
    #[test]
    fn a_single_endpoint_move_is_priced_the_same_way(ops in ops()) {
        run::<Directed>(MoveKey { from: Group::new(3), to: None }, &ops);
    }
}

/// Recording, then beginning a **different** move, leaves no live cell --
/// including in the previous key's plane, which a `clear()` that re-derives
/// addresses from the *new* key would miss entirely.
#[test]
fn begin_with_a_new_key_leaves_no_residue() {
    let first = MoveKey {
        from: Group::new(0),
        to: Group::new(1),
    };
    let second = MoveKey {
        from: Group::new(2),
        to: Group::new(3),
    };
    let mut d = fresh::<Directed>(first);
    for o in 0..B as u32 {
        d.touch_dyn(g(0), g(o), 3, &mut |r, s| resolve(r, s)).unwrap();
        d.touch_dyn(g(o), g(0), 5, &mut |r, s| resolve(r, s)).unwrap();
        d.touch_dyn(g(1), g(o), 7, &mut |r, s| resolve(r, s)).unwrap();
    }
    assert!(!d.entries().is_empty());
    assert!(!d.table_is_clear());

    d.begin(second, B);
    assert!(d.entries().is_empty());
    assert!(d.table_is_clear(), "every field slot is null");
    assert_eq!(d.move_key(), second);
    for s in 0..B as u32 {
        for t in 0..B as u32 {
            assert_eq!(d.delta_of(g(s), g(t)), 0);
            assert_eq!(d.me_of(g(s), g(t)), None);
        }
    }

    // And the old plane is genuinely empty, not merely unreachable from the
    // new key.
    d.begin(first, B);
    assert!(d.table_is_clear());
    for s in 0..B as u32 {
        for t in 0..B as u32 {
            assert_eq!(d.delta_of(g(s), g(t)), 0);
        }
    }
}

/// `begin` writes one cell per entry, whatever the group count is. Asserted
/// on the write counter, not on a clock.
#[test]
fn begin_costs_one_write_per_entry_not_one_per_group() {
    const BIG: usize = 4096;
    let mv = MoveKey {
        from: Group::new(0),
        to: Group::new(1),
    };
    let mut d = DeltaBuf::<Directed, i64>::default();

    d.begin(mv, BIG);
    assert_eq!(d.field_writes(), 0, "an empty buffer resets nothing");
    assert_eq!(d.table_len(), BIG);

    for _ in 0..10 {
        d.touch_dyn(g(0), g(7), 1, &mut |r, s| resolve(r, s)).unwrap();
        d.touch_dyn(g(7), g(0), 1, &mut |r, s| resolve(r, s)).unwrap();
        d.touch_dyn(g(1), g(9), 1, &mut |r, s| resolve(r, s)).unwrap();
    }
    let n = d.entries().len();
    assert_eq!(n, 3);

    let before = d.field_writes();
    d.begin(
        MoveKey {
            from: Group::new(4),
            to: Group::new(5),
        },
        BIG,
    );
    assert_eq!(d.field_writes() - before, n as u64);
    assert!(d.table_is_clear());

    // The table only ever grows (`entries.hh:63`).
    d.begin(mv, 4);
    assert_eq!(d.table_len(), BIG);
}

/// Defect #4. The C++ hands an out-of-plane pair the single `_dummy` cell
/// (`entries.hh:118, :223`), so the *second* such pair finds it occupied and
/// adds its weight to the first pair's entry.
#[test]
fn an_out_of_plane_pair_is_rejected_and_changes_nothing() {
    let mv = MoveKey {
        from: Group::new(0),
        to: Group::new(1),
    };
    let mut d = fresh::<Directed>(mv);
    d.touch_dyn(g(0), g(2), 11, &mut |r, s| resolve(r, s)).unwrap();

    let snapshot = d.entries().to_vec();
    let writes = d.field_writes();

    assert_eq!(
        d.touch_dyn(g(2), g(3), 100, &mut |r, s| resolve(r, s)),
        Err(OutOfPlane(g(2), g(3)))
    );
    assert_eq!(
        d.touch_dyn(g(4), g(5), 200, &mut |r, s| resolve(r, s)),
        Err(OutOfPlane(g(4), g(5)))
    );

    assert_eq!(d.entries(), snapshot.as_slice());
    assert_eq!(d.field_writes(), writes);
    assert_eq!(d.delta_of(g(2), g(3)), 0);
    assert_eq!(d.delta_of(g(4), g(5)), 0);
    assert_eq!(d.me_of(g(2), g(3)), None);
    // The one live entry is untouched: no aliasing happened.
    assert_eq!(d.delta_of(g(0), g(2)), 11);
    assert_eq!(d.entries().len(), 1);
}

/// The undirected buffer allocates **two** half-field vectors, not four empty
/// ones -- `entries.hh:52`'s `if constexpr (directed)`.
///
/// Two forms of the same claim, and both now hold.
///
/// The *storage* form is on the table, which is the O(B) part: an undirected
/// buffer owns `2 * B` cells against the directed buffer's `4 * B`.
///
/// The *type* form -- `size_of::<DeltaBuf<Undirected, _>>()` strictly less
/// than the directed one -- needs the field table's type to vary with `D`. It
/// was asserted only as `<=` when this unit landed, because `Vec<Vec<u32>>`
/// is one 24-byte header whatever `D` is, and an array length cannot be an
/// associated const on stable (`error: generic parameters may not be used in
/// const operations`). `gt_core::dir::Dir` now carries `type Fields`
/// (`[Vec<u32>; 4]` / `[Vec<u32>; 2]`), which is the associated field set its
/// own module doc always said a bare `const DIRECTED: bool` could not carry,
/// so the strict inequality is asserted below.
#[test]
fn the_undirected_table_is_half_the_directed_one() {
    assert_eq!(Undirected::N_FIELDS, 2);
    assert_eq!(Directed::N_FIELDS, 4);

    let mv = MoveKey {
        from: Group::new(0),
        to: Group::new(1),
    };
    let u = fresh::<Undirected>(mv);
    let d = fresh::<Directed>(mv);

    let cells = |n: usize, len: usize| n * len;
    assert_eq!(cells(Undirected::N_FIELDS, u.table_len()), 2 * B);
    assert_eq!(cells(Directed::N_FIELDS, d.table_len()), 4 * B);
    assert!(
        cells(Undirected::N_FIELDS, u.table_len())
            < cells(Directed::N_FIELDS, d.table_len())
    );

    // Strict: the undirected buffer *contains* two field vectors, the
    // directed one four, so the difference is two `Vec` headers (48 bytes).
    assert!(
        size_of::<DeltaBuf<Undirected, i64>>() < size_of::<DeltaBuf<Directed, i64>>(),
        "undirected {} is not strictly smaller than directed {}",
        size_of::<DeltaBuf<Undirected, i64>>(),
        size_of::<DeltaBuf<Directed, i64>>()
    );
    assert_eq!(
        size_of::<DeltaBuf<Directed, i64>>() - size_of::<DeltaBuf<Undirected, i64>>(),
        2 * size_of::<Vec<u32>>()
    );
}

/// An undirected lookup is symmetric and a directed one is not: `(r, nr)` and
/// `(nr, r)` are two entries directed, one undirected.
#[test]
fn directedness_decides_pair_identity() {
    let mv = MoveKey {
        from: Group::new(0),
        to: Group::new(1),
    };

    let mut u = fresh::<Undirected>(mv);
    u.touch_dyn(g(0), g(4), 2, &mut |r, s| resolve(r, s)).unwrap();
    u.touch_dyn(g(4), g(0), 3, &mut |r, s| resolve(r, s)).unwrap();
    assert_eq!(u.entries().len(), 1);
    assert_eq!(u.delta_of(g(0), g(4)), 5);
    assert_eq!(u.delta_of(g(4), g(0)), 5);

    let mut d = fresh::<Directed>(mv);
    d.touch_dyn(g(0), g(4), 2, &mut |r, s| resolve(r, s)).unwrap();
    d.touch_dyn(g(4), g(0), 3, &mut |r, s| resolve(r, s)).unwrap();
    assert_eq!(d.entries().len(), 2);
    assert_eq!(d.delta_of(g(0), g(4)), 2);
    assert_eq!(d.delta_of(g(4), g(0)), 3);

    // The self-pair has one cell in both, because `get_field_rnr` routes
    // `s == t` to the out half (`entries.hh:97`).
    let mut d2 = fresh::<Directed>(mv);
    d2.touch_dyn(g(0), g(0), 6, &mut |r, s| resolve(r, s)).unwrap();
    d2.touch_dyn(g(0), g(0), -1, &mut |r, s| resolve(r, s)).unwrap();
    assert_eq!(d2.entries().len(), 1);
    assert_eq!(d2.delta_of(g(0), g(0)), 5);
}

/// A group beyond the table is "not recorded", not an out-of-bounds read --
/// `get_delta` indexes `_r_out_field[t]` unchecked (`entries.hh:156-162`).
#[test]
fn a_group_past_the_table_reads_as_zero() {
    let mv = MoveKey {
        from: Group::new(0),
        to: Group::new(1),
    };
    let mut d = DeltaBuf::<Directed, i64>::default();
    d.begin(mv, 2);
    assert_eq!(d.delta_of(g(0), g(9999)), 0);
    assert_eq!(d.me_of(g(0), g(9999)), None);
    assert_eq!(d.mrs_before_of(g(9999), g(1)), None);
}
