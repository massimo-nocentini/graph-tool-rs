//! The sparse delta buffer: graph-tool's `EntrySet`, made total.

use gt_core::dir::{Dir, Directed, Field};

use super::entry::Entry;
use crate::ids::{BEdge, Group, Weight};

/// The empty field cell.
///
/// `EntrySet::_null` (`entries.hh:198`) is `numeric_limits<uint32_t>::max()`
/// even though the field vectors are `uint32_t` and the sentinel is compared
/// against a `size_t`; here the sentinel and the cell have one type.
const NULL: u32 = u32::MAX;

/// Field-table slots, in `Field<D>`'s order. `const`-checked against it below,
/// because this module indexes the table arithmetically (`fi & 1` selects the
/// endpoint, `fi >= 2` the half) and that arithmetic is only meaningful if the
/// two orders agree.
const F_R_OUT: usize = 0;
const F_NR_OUT: usize = 1;
const F_R_IN: usize = 2;
const F_NR_IN: usize = 3;

const _: () = {
    assert!(Field::<Directed>::R_OUT.index() == F_R_OUT);
    assert!(Field::<Directed>::NR_OUT.index() == F_NR_OUT);
    assert!(Field::<Directed>::R_IN.index() == F_R_IN);
    assert!(Field::<Directed>::NR_IN.index() == F_NR_IN);
    // `fi & 1` picks the endpoint and `fi >= 2` the half only if the four
    // constants are laid out `{r,nr} x {out,in}` in exactly that nesting.
    assert!(F_R_OUT + 2 == F_R_IN);
    assert!(F_NR_OUT + 2 == F_NR_IN);
};

/// Which move a buffer is currently recording.
///
/// `EntrySet::_rnr` (`entries.hh:200`), with `null_group` spelled `None`.
/// `modify_entries` (`:322-332`) never records `from == to`: the `r == nr`
/// case dispatches to `single`, which sets `to` to the null group and folds
/// both endpoints onto the `r` half-fields. A `MoveKey` whose two endpoints
/// are equal and present is therefore not a value this buffer expects, and
/// the crate-private `DeltaBuf::touch` behind [`DeltaBuf::touch_dyn`]
/// `debug_assert`s so.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct MoveKey {
    /// Source group.
    pub from: Option<Group>,
    /// Target group.
    pub to: Option<Group>,
}

/// Back-pointer from an entry to the field slot that indexes it.
///
/// Storing one per entry is what makes `begin` branch-free **and independent
/// of the previous move key**. `EntrySet::clear` (`entries.hh:169-176`)
/// re-derives each field address through `get_field`, which reads `_rnr`, so
/// `set_move` must call `clear()` *before* assigning `_rnr` -- a temporal
/// coupling stated nowhere.
///
/// Eight bytes, not a packed `u32`. A `field:2 | other:30` packing saves four
/// bytes per entry in a loop that runs once per cleared slot, and caps the
/// group count at 2^30 -- a capability the C++ has and there is no reason to
/// give up for that.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SlotRef {
    /// The other endpoint keying the slot.
    pub other: u32,
    /// Which half-field.
    pub field: u8,
}

/// Interns the block-graph edge and its current weight for a `(r, s)` pair.
///
/// A closure rather than a trait method, so that this module never needs to
/// know what a block state is: the reified transition is *data*, and saying so
/// structurally means the delta types do not depend on the state types.
pub trait Resolve<W: Weight>: FnMut(Group, Group) -> (Option<BEdge>, W) {}
impl<W: Weight, F: FnMut(Group, Group) -> (Option<BEdge>, W)> Resolve<W> for F {}

/// A block pair touching neither endpoint of the current move.
///
/// `get_field` (`entries.hh:108-119`) falls through to `return _dummy;` -- a
/// single `uint32_t` member shared by *every* out-of-plane pair (`:223`) --
/// and `insert_delta_dispatch` then keys an entry on it, so two distinct pairs
/// silently accumulate into one. The path is reachable only from
/// `propagate_entries`, where the projected pair provably contains an
/// endpoint, so the C++ bug is latent and guarded by an invariant written
/// nowhere. Here there is no dummy cell to alias into.
#[derive(Clone, Copy, PartialEq, Eq, Debug, thiserror::Error)]
#[error("block pair ({0:?}, {1:?}) touches neither endpoint of the recorded move")]
pub struct OutOfPlane(pub Group, pub Group);

/// One level's sparse record of "which block pairs change, and by how much".
///
/// The field table is [`D::Fields`](Dir::Fields), the associated *array*
/// `[Vec<u32>; D::N_FIELDS]`, so an undirected buffer holds **two**
/// half-fields and a directed one four -- the port of `entries.hh:52`'s
/// `if constexpr (directed)`. Because the table is an array and not a
/// `Vec<Vec<u32>>`, the undirected buffer does not merely *allocate* less: it
/// is a strictly smaller type, which `size_of` asserts. Lookup is O(1) with
/// no hashing, which is valid because every changed pair contains `r` or
/// `nr`.
#[derive(Clone, Debug)]
pub struct DeltaBuf<D: Dir, W: Weight> {
    mv: MoveKey,
    fields: D::Fields,
    entries: Vec<Entry<W>>,
    slots: Vec<SlotRef>,
    /// Cumulative count of field cells written by [`DeltaBuf::begin`].
    ///
    /// Not instrumentation for its own sake: "`begin` is O(#entries)" is the
    /// one claim of this type that a timing test cannot establish and a
    /// counter can. One increment per store, in a loop that already stores.
    field_writes: u64,
    _d: std::marker::PhantomData<fn() -> D>,
}

impl<D: Dir, W: Weight> Default for DeltaBuf<D, W> {
    fn default() -> Self {
        DeltaBuf {
            mv: MoveKey::default(),
            fields: D::Fields::default(),
            entries: Vec::new(),
            slots: Vec::new(),
            field_writes: 0,
            _d: std::marker::PhantomData,
        }
    }
}

/// The endpoint a half-field names is absent from the move key.
///
/// Out of line and `#[cold]`: `modify_entries_dispatch` guards every
/// `insert_delta_rnr` with `if (r != null_group)` / `if (nr != null_group)`
/// (`entries.hh:245-255`), so this is a recorder bug, not a data case.
#[cold]
#[inline(never)]
fn absent_endpoint(fi: usize) -> ! {
    panic!("touch on half-field {fi} whose move-key endpoint is absent");
}

impl<D: Dir, W: Weight> DeltaBuf<D, W> {
    /// Start recording a new move, clearing in O(#entries).
    ///
    /// Only ever grows the field tables, matching `entries.hh:48`'s
    /// amortisation. Correct in any order relative to the move key, unlike
    /// `set_move`/`clear`.
    pub fn begin(&mut self, mv: MoveKey, n_groups: usize) {
        // `clear()` (`entries.hh:169-176`) walks `_entries` and re-derives
        // each address with `get_field(r, s)`, i.e. through `_rnr`. Walking
        // the recorded addresses instead is the same O(#entries) and drops
        // the dependency on the key entirely -- so assigning `mv` first,
        // last, or not at all cannot leave a live cell behind.
        for slot in self.slots.drain(..) {
            self.fields.as_mut()[slot.field as usize][slot.other as usize] = NULL;
            self.field_writes += 1;
        }
        self.entries.clear();
        self.mv = mv;

        // `set_move` (`:59-65`): grow only, null-filled.
        if n_groups > self.fields.as_ref()[0].len() {
            for f in self.fields.as_mut() {
                f.resize(n_groups, NULL);
            }
        }
    }

    /// The move this buffer records.
    #[inline]
    pub const fn move_key(&self) -> MoveKey {
        self.mv
    }

    /// The recorded entries.
    #[inline]
    pub fn entries(&self) -> &[Entry<W>] {
        &self.entries
    }

    /// Number of groups the field table currently covers.
    ///
    /// `_r_out_field.size()` (`entries.hh:63`). Grows, never shrinks.
    #[inline]
    pub fn table_len(&self) -> usize {
        self.fields.as_ref()[0].len()
    }

    /// Cumulative number of field cells [`begin`](Self::begin) has written.
    ///
    /// Exactly one per entry cleared, and nothing else touches the table on
    /// the reset path. A test asserts this against `entries().len()` to pin
    /// the "O(#entries), not O(B)" claim without measuring a clock.
    #[inline]
    pub const fn field_writes(&self) -> u64 {
        self.field_writes
    }

    /// Whether every cell of the field table is null.
    ///
    /// O(`D::N_FIELDS` * `table_len()`), i.e. the cost `begin` deliberately
    /// does *not* pay. For the audit and for tests; never on a hot path.
    pub fn table_is_clear(&self) -> bool {
        self.fields.as_ref().iter().flatten().all(|&c| c == NULL)
    }

    /// The half-field the pair `(s, t)` is keyed by, or `None` when it
    /// touches neither endpoint.
    ///
    /// `get_field` (`entries.hh:108-119`) branch for branch and **in order**:
    /// the order is load-bearing, because the pair `(r, nr)` matches two of
    /// the four tests and the first one wins. The fallthrough returns `None`
    /// where the C++ returns `_dummy`.
    #[inline]
    fn slot_of(&self, s: Group, t: Group) -> Option<(usize, usize)> {
        if self.mv.from == Some(s) {
            return Some((F_R_OUT, t.index()));
        }
        if self.mv.from == Some(t) {
            return Some((Self::sink_half(F_R_OUT, s, t), s.index()));
        }
        if self.mv.to == Some(s) {
            return Some((F_NR_OUT, t.index()));
        }
        if self.mv.to == Some(t) {
            return Some((Self::sink_half(F_NR_OUT, s, t), s.index()));
        }
        None
    }

    /// `get_field_rnr<First, Source = false>` (`entries.hh:90-103`).
    ///
    /// Directed: `(s == t) ? out_field[t] : in_field[s]` -- an endpoint's own
    /// self-pair has no in-half, so it lives in the out-half whichever call
    /// site reaches it. Undirected: `out_field[s]`, there being no in-half at
    /// all, which `Field<Undirected>` states in the type.
    #[inline]
    fn sink_half(out: usize, s: Group, t: Group) -> usize {
        if D::DIRECTED && s != t { out + 2 } else { out }
    }

    /// Record a weight change on one half-field.
    ///
    /// The static path, and the only one the recorder uses. `Field<D>` is
    /// total and `ADD` is a const generic, so this is
    /// `insert_delta_rnr<First, Source, Add>` (`entries.hh:142`) one for one,
    /// with all three parameters resolved at monomorphisation and no
    /// fallthrough to reach.
    ///
    /// `other` is the *other* endpoint of the pair: `t` for an out-half and
    /// `s` for an in-half, matching `get_field_rnr`'s indexing. The caller
    /// owes the same routing decision `modify_entries_dispatch` makes at
    /// `entries.hh:258-261` and `:311-314` -- a pair containing both `r` and
    /// `nr` belongs to `r`'s half-fields, because `get_field` tests `r`
    /// first. Violating that would key one pair under two cells;
    /// `debug_assert` catches it.
    ///
    /// # Panics
    ///
    /// If the endpoint `f` names is absent from the move key, or if `other`
    /// is beyond the group count given to [`begin`](Self::begin).
    pub(crate) fn touch<const ADD: bool>(
        &mut self,
        f: Field<D>,
        other: Group,
        w: W,
        resolve: &mut impl super::Resolve<W>,
    ) {
        let fi = f.index();
        // `fi & 1`: out/in halves of the same endpoint are two apart.
        let own = match if fi & 1 == 0 { self.mv.from } else { self.mv.to } {
            Some(g) => g,
            None => absent_endpoint(fi),
        };
        let is_in = fi >= 2;
        let (s, t) = if is_in { (other, own) } else { (own, other) };
        // The `s == t` arm of `get_field_rnr`, hoisted: an in-half touch on
        // the endpoint's own self-pair is an out-half touch.
        let fi = if is_in && other == own { fi - 2 } else { fi };

        debug_assert_eq!(
            self.slot_of(s, t),
            Some((fi, other.index())),
            "half-field {fi} is not where `get_field` routes ({s:?}, {t:?}) \
             under {:?}",
            self.mv
        );
        self.insert::<ADD>(fi, other.index(), s, t, w, resolve);
    }

    /// Record a weight change on an arbitrary pair.
    ///
    /// `insert_delta` (`entries.hh:149`), made **total**. Its only caller is
    /// [`propagate`](crate::blockmodel::propagate), which is exactly where the
    /// C++'s latent `_dummy` aliasing lives. The error is provably unreachable
    /// given the hierarchy invariant, and is returned rather than asserted
    /// because the projection is a caller-supplied closure.
    ///
    /// On `Err` the buffer is **unchanged**: nothing is resolved, no entry is
    /// created, no cell is written. That is defect #4 -- the C++ creates an
    /// entry keyed on `_dummy`, which the next out-of-plane pair then finds
    /// already occupied and adds to.
    pub fn touch_dyn(
        &mut self,
        s: Group,
        t: Group,
        w: W,
        resolve: &mut impl super::Resolve<W>,
    ) -> Result<(), OutOfPlane> {
        let Some((fi, oi)) = self.slot_of(s, t) else {
            return Err(OutOfPlane(s, t));
        };
        // `insert_delta` is only ever instantiated with `Add = true`
        // (`state.hh:1113`); a sign change is expressed in the delta.
        self.insert::<true>(fi, oi, s, t, w, resolve);
        Ok(())
    }

    /// `insert_delta_dispatch` (`entries.hh:122-135`), plus the interning the
    /// C++ defers to `get_mes` (`:181-191`).
    ///
    /// The before-image is read exactly once per *entry*, not once per touch:
    /// the resolve closure runs only on the cell that was null.
    #[inline]
    fn insert<const ADD: bool>(
        &mut self,
        fi: usize,
        oi: usize,
        s: Group,
        t: Group,
        w: W,
        resolve: &mut impl super::Resolve<W>,
    ) {
        let cell = self.fields.as_ref()[fi][oi];
        let i = if cell == NULL {
            let i = self.entries.len();
            debug_assert!(i < NULL as usize, "entry index collides with the null cell");
            let (me, mrs_before) = resolve(s, t);
            self.entries.push(Entry {
                r: s,
                s: t,
                delta: W::ZERO,
                me,
                mrs_before,
            });
            self.slots.push(SlotRef {
                other: oi as u32,
                field: fi as u8,
            });
            self.fields.as_mut()[fi][oi] = i as u32;
            i
        } else {
            cell as usize
        };
        let e = &mut self.entries[i];
        e.delta = if ADD { e.delta + w } else { e.delta - w };
    }

    /// The entry index for a pair, if it has one.
    #[inline]
    fn lookup(&self, s: Group, t: Group) -> Option<usize> {
        let (fi, oi) = self.slot_of(s, t)?;
        // `get_delta` indexes `_r_out_field[t]` unchecked (`entries.hh:156`);
        // a group past the table is "not recorded", not a wild read.
        match self.fields.as_ref()[fi].get(oi).copied() {
            Some(c) if c != NULL => Some(c as usize),
            _ => None,
        }
    }

    /// The accumulated change for a pair. `entries.hh:156`, honestly `&self`.
    pub fn delta_of(&self, s: Group, t: Group) -> W {
        match self.lookup(s, t) {
            Some(i) => self.entries[i].delta,
            // `_zero` (`entries.hh:224`), which `get_delta` returns by
            // reference for both the null-cell and the out-of-plane case.
            None => W::ZERO,
        }
    }

    /// The interned block-graph edge for a pair, if the pair is recorded.
    ///
    /// A genuine shared-borrow read. `get_mes` (`entries.hh:181-191`) lazily
    /// extends `_mes` and is reached from the `get_move_prob` *read* path
    /// (`state.hh:1661`) through `cget_field`'s
    /// `const_cast<EntrySet*>(this)` (`:121-123`). Interning `me` eagerly is
    /// what makes this an ordinary `&self` method.
    ///
    /// `None` means "not recorded", where `get_me` (`:194-199`) falls back to
    /// `emat.get_me(r, s)`; the caller owns that fallback, because this type
    /// has no access to the state.
    pub fn me_of(&self, s: Group, t: Group) -> Option<Option<BEdge>> {
        self.lookup(s, t).map(|i| self.entries[i].me)
    }

    /// The pair's weight before the move, if the pair is recorded.
    ///
    /// The other half of what `get_mes`'s caller reads at `state.hh:1225`.
    pub fn mrs_before_of(&self, s: Group, t: Group) -> Option<W> {
        self.lookup(s, t).map(|i| self.entries[i].mrs_before)
    }
}

// ---------------------------------------------------------------------------
// `touch` is `pub(crate)` -- the static half-field path belongs to the
// recorder -- so its tests live here rather than in `tests/u19_deltabuf.rs`.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use gt_core::dir::Undirected;

    fn g(i: u32) -> Group {
        Group::new(i).expect("group index fits")
    }

    /// A resolve closure that interns a distinguishable edge per pair and
    /// counts its own calls.
    fn probe(calls: &mut usize) -> impl super::super::Resolve<i64> + '_ {
        move |r: Group, s: Group| {
            *calls += 1;
            (
                Some(BEdge((r.index() * 16 + s.index()) as u32)),
                (r.index() * 100 + s.index()) as i64,
            )
        }
    }

    fn buf<D: Dir>(from: u32, to: Option<u32>, b: usize) -> DeltaBuf<D, i64> {
        let mut d = DeltaBuf::<D, i64>::default();
        d.begin(
            MoveKey {
                from: Group::new(from),
                to: to.and_then(Group::new),
            },
            b,
        );
        d
    }

    /// Every `(half-field, other)` the recorder is allowed to use routes to
    /// the same cell `get_field` would pick for the pair it denotes -- which
    /// is what makes one pair one entry.
    #[test]
    fn the_static_and_dynamic_paths_pick_the_same_cell() {
        const B: usize = 5;
        for &f in &[
            Field::<Directed>::R_OUT,
            Field::<Directed>::NR_OUT,
            Field::<Directed>::R_IN,
            Field::<Directed>::NR_IN,
        ] {
            for o in 0..B as u32 {
                let mut a = buf::<Directed>(0, Some(1), B);
                let fi = f.index();
                let own = if fi & 1 == 0 { g(0) } else { g(1) };
                let (s, t) = if fi >= 2 {
                    (g(o), own)
                } else {
                    (own, g(o))
                };
                // The routing the C++ recorder owes; see `touch`.
                if a.slot_of(s, t) != Some((if fi >= 2 && g(o) == own { fi - 2 } else { fi }, o as usize))
                {
                    continue;
                }
                let mut n = 0;
                a.touch::<true>(f, g(o), 7, &mut probe(&mut n));

                let mut b2 = buf::<Directed>(0, Some(1), B);
                let mut m = 0;
                b2.touch_dyn(s, t, 7, &mut probe(&mut m)).expect("in plane");

                assert_eq!(a.entries(), b2.entries(), "field {fi}, other {o}");
                assert_eq!(a.delta_of(s, t), 7);
                assert_eq!(n, 1);
                assert_eq!(m, 1);
            }
        }
    }

    /// `get_field_rnr<First, false>`'s `(s == t) ? out_field[t]`
    /// (`entries.hh:97`): an endpoint's self-pair has one cell, reached from
    /// both the out-edge and the in-edge pass of the recorder.
    #[test]
    fn the_self_pair_of_an_endpoint_has_exactly_one_cell() {
        let mut d = buf::<Directed>(0, Some(1), 4);
        let mut n = 0;
        {
            let mut r = probe(&mut n);
            // `insert_delta_rnr<true, true, false>(r, s)` with `s == r`.
            d.touch::<false>(Field::<Directed>::R_OUT, g(0), 5, &mut r);
            // `insert_delta_rnr<true, false, false>(s, r)` with `s == r`.
            d.touch::<false>(Field::<Directed>::R_IN, g(0), 3, &mut r);
        }
        assert_eq!(d.entries().len(), 1);
        assert_eq!(n, 1, "the before-image is interned once per entry");
        assert_eq!(d.delta_of(g(0), g(0)), -8);
        assert_eq!(d.entries()[0].r, g(0));
        assert_eq!(d.entries()[0].s, g(0));
    }

    /// `(r, nr)` and `(nr, r)` are different directed pairs and get different
    /// cells; undirected they are one pair and get one.
    #[test]
    fn the_cross_pair_is_two_entries_directed_and_one_undirected() {
        let mut d = buf::<Directed>(0, Some(1), 4);
        let mut n = 0;
        {
            let mut r = probe(&mut n);
            d.touch::<true>(Field::<Directed>::R_OUT, g(1), 2, &mut r);
            d.touch::<true>(Field::<Directed>::R_IN, g(1), 4, &mut r);
        }
        assert_eq!(d.entries().len(), 2);
        assert_eq!(d.delta_of(g(0), g(1)), 2);
        assert_eq!(d.delta_of(g(1), g(0)), 4);

        let mut u = buf::<Undirected>(0, Some(1), 4);
        let mut m = 0;
        {
            let mut r = probe(&mut m);
            u.touch::<true>(Field::<Undirected>::R_OUT, g(1), 2, &mut r);
            // The `else` arm of `entries.hh:258-261`: a pair containing both
            // endpoints is routed to `r`'s half-field, not `nr`'s.
            u.touch::<true>(Field::<Undirected>::R_OUT, g(1), 4, &mut r);
        }
        assert_eq!(u.entries().len(), 1);
        assert_eq!(u.delta_of(g(0), g(1)), 6);
        assert_eq!(u.delta_of(g(1), g(0)), 6, "undirected lookup is symmetric");
    }

    /// `single` (`entries.hh:322-332`): `r == nr` folds both endpoints onto
    /// the `r` half-fields and leaves `to` null.
    #[test]
    fn a_single_endpoint_move_uses_the_r_half_fields_only() {
        let mut d = buf::<Directed>(2, None, 6);
        let mut n = 0;
        {
            let mut r = probe(&mut n);
            d.touch::<false>(Field::<Directed>::R_OUT, g(4), 9, &mut r);
            d.touch::<true>(Field::<Directed>::R_OUT, g(4), 9, &mut r);
        }
        assert_eq!(d.entries().len(), 1);
        assert_eq!(d.delta_of(g(2), g(4)), 0, "a zero entry is still an entry");
        assert_eq!(d.me_of(g(2), g(4)), Some(Some(BEdge(2 * 16 + 4))));
        assert_eq!(d.mrs_before_of(g(2), g(4)), Some(204));
    }

    #[test]
    #[should_panic(expected = "whose move-key endpoint is absent")]
    fn touching_an_absent_endpoint_is_a_recorder_bug() {
        let mut d = buf::<Directed>(2, None, 6);
        let mut n = 0;
        d.touch::<true>(Field::<Directed>::NR_OUT, g(1), 1, &mut probe(&mut n));
    }

    /// The pair `(nr, r)` belongs to `r`'s in-half because `get_field` tests
    /// `r` first; routing it to `nr`'s out-half would key one pair twice.
    #[test]
    #[should_panic(expected = "is not where `get_field` routes")]
    fn misrouting_a_pair_that_contains_both_endpoints_is_caught() {
        let mut d = buf::<Directed>(0, Some(1), 4);
        let mut n = 0;
        d.touch::<true>(Field::<Directed>::NR_OUT, g(0), 1, &mut probe(&mut n));
    }
}
