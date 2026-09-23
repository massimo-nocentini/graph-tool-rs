//! The delta-versus-absolute cross-check, cheap enough to always run.
//!
//! `base_states.py:33` sets `__test__ = False`, and the wrapper at `:44-59`
//! shows why: when enabled it **copies the whole state and recomputes the
//! whole entropy** per call, i.e. O(E + B^2). So the invariant that the
//! incremental pricing agrees with the absolute functional is checked by
//! nobody, in the default configuration, ever.
//!
//! Both checks below are O(#entries + 2) with the same access pattern as
//! pricing, so a debug build roughly doubles the per-move cost instead of
//! multiplying it. They run on every commit under `debug_assertions`.
//!
//! One honest limitation, which the source design did not state: the `eterm`
//! half of [`audit_price`] is algebraically identical to the pricing
//! expression -- both compute `eterm(after) - eterm(before)` -- so a *wrong*
//! `eterm` reproduces itself and the check passes too. What is genuinely
//! independent is the *source of the before-image*: [`sparse_ds`] reads the
//! snapshot the recorder interned ([`Entry::mrs_before`],
//! [`MoveHeader::r_img`]), while [`audit_price`] re-reads the live state. A
//! stale or mis-resolved before-image -- the one class of recorder defect
//! pricing cannot see, because pricing never touches the state -- is caught
//! here and nowhere else. The from-scratch recompute survives behind the
//! `audit-full` feature, for a periodic sweep-level drift check.
//!
//! [`sparse_ds`]: super::sparse_ds
//!
//! ## When each one runs
//!
//! [`audit_price`] takes `&Transition`, and [`Transition::into_levels`]
//! consumes the transition, so it is structurally a **pre-commit** check: the
//! state it reads is the state the delta was recorded against.
//! [`audit_commit`] takes a [`Receipt`], which is what survives the commit, so
//! it is structurally a **post-commit** check. Neither can be called at the
//! wrong moment by accident; that is the whole point of D10 making the *level*
//! the unit of consumption.

use crate::delta::{EndImage, Entry, MoveHeader, Receipt, Transition};
use crate::ids::{Group, Stamp, Weight};
use gt_core::dir::Dir;

use super::cache::Cache;
use super::entropy::{EntropyParams, eterm, vterm};
use super::state::BlockView;

/// What an audit found.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum AuditError {
    /// A block pair's post-state weight is not `mrs_before + delta`.
    #[error("pair ({r}, {s}): expected {expected}, found {found}")]
    EdgeWeight {
        /// Source group index.
        r: usize,
        /// Target group index.
        s: usize,
        /// `mrs_before + delta`.
        expected: f64,
        /// What the state holds.
        found: f64,
    },
    /// An endpoint's scalars do not match the header's snapshot plus the
    /// move's own degree deltas.
    #[error("group {r}: {field} expected {expected}, found {found}")]
    EndScalar {
        /// Group index.
        r: usize,
        /// Which of `mrp`/`mrm`/`wr`.
        field: &'static str,
        /// Expected value.
        expected: f64,
        /// Found value.
        found: f64,
    },
    /// The claimed entropy change disagrees with the recomputation.
    #[error("dS: claimed {claimed}, recomputed {recomputed} (tolerance {tol})")]
    Entropy {
        /// What pricing returned.
        claimed: f64,
        /// What the audit recomputed.
        recomputed: f64,
        /// The tolerance applied.
        tol: f64,
    },
    /// The delta was recorded against a different state or revision.
    #[error(
        "stamp mismatch: delta carries state #{delta_state}/epoch {delta_epoch}, state is #{state_state}/epoch {state_epoch}"
    )]
    Stamp {
        /// Identity carried by the delta.
        delta_state: u64,
        /// Revision carried by the delta.
        delta_epoch: u64,
        /// The state's identity.
        state_state: u64,
        /// The state's revision.
        state_epoch: u64,
    },
}

// ---------------------------------------------------------------------------
// Tolerance.
// ---------------------------------------------------------------------------

/// The absolute floor, matching `math.isclose(S, Salt, abs_tol=1e-8)`
/// (`base_states.py:56`) but one decade tighter: graph-tool compares two
/// *independent* O(E + B^2) sums, where this compares one incremental step
/// against a recomputation of the same terms.
const ABS_TOL: f64 = 1e-9;

/// The relative floor. `eterm`/`vterm` are sums of `lgamma` values that grow
/// like `E log E`, so a purely absolute tolerance becomes a test of the
/// floating-point unit rather than of the pricing once the state is large.
const REL_TOL: f64 = 1e-12;

/// The tolerance applied to a `dS` comparison at this magnitude.
#[inline]
fn tolerance(a: f64, b: f64) -> f64 {
    ABS_TOL + REL_TOL * a.abs().max(b.abs())
}

/// Compare a claim against a recomputation.
///
/// Written as `!(diff <= tol)` rather than `diff > tol` on purpose: every
/// comparison against `NaN` is false, so a `NaN` on either side -- including
/// the `inf - inf` of two infinite entropies -- reports rather than passes.
/// That is `base_states.py:47`'s `assert not isnan(S) and not isinf(S)`,
/// folded into the same check instead of bolted on before it.
#[inline]
fn agree(claimed: f64, recomputed: f64) -> Result<(), AuditError> {
    let tol = tolerance(claimed, recomputed);
    if (claimed - recomputed).abs() <= tol {
        Ok(())
    } else {
        Err(AuditError::Entropy {
            claimed,
            recomputed,
            tol,
        })
    }
}

/// A [`Stamp`] pair, as an error.
#[inline]
fn stamp_error(delta: Stamp, state: Stamp) -> AuditError {
    AuditError::Stamp {
        delta_state: delta.state.get(),
        delta_epoch: delta.epoch.0,
        state_state: state.state.get(),
        state_epoch: state.epoch.0,
    }
}

// ---------------------------------------------------------------------------
// State reads.
// ---------------------------------------------------------------------------

/// The weight the state currently holds for the pair `(r, s)`.
///
/// Two reads -- the `_emat.get_me` probe and the `_mrs[me]` indirection of
/// `state.hh:1225` -- and exactly the pair of reads the pricing loop was
/// relieved of by interning the before-image. The audit pays them back once
/// per entry, which is what makes it O(#entries) and not O(B^2).
#[inline]
fn live_pair<S: BlockView>(st: &S, r: Group, s: Group) -> S::W {
    match st.find_me(r, s) {
        Some(e) => st.mrs(e),
        None => S::W::ZERO,
    }
}

/// One group's three scalars, reading `_mrm` only where the model has one.
///
/// `update_rs` writes `_mrm[s]` under `if constexpr (is_directed)` and
/// `_mrp[s]` otherwise (`entries.hh:393-398`), so an undirected `_mrm` is
/// structurally zero: reading it would be a mutex acquisition bought for a
/// value `vterm` (`entropy.hh:84-87`) does not look at.
#[inline]
fn scalars<S: BlockView>(st: &S, g: Group) -> (f64, f64, f64) {
    let mrp = st.mrp(g).to_f64();
    let mrm = if <S::D as Dir>::DIRECTED {
        st.mrm(g).to_f64()
    } else {
        0.0
    };
    (mrp, mrm, st.wr(g).to_f64())
}

// ---------------------------------------------------------------------------
// The post-commit audit.
// ---------------------------------------------------------------------------

/// One endpoint's three scalars against the header's snapshot plus its deltas.
///
/// `d_mrp`, `d_mrm` and `d_wr` are *signed*: the source endpoint passes
/// `(-dkout, -dkin, -dr)` and the target `(+dkout, +dkin, +dnr)`, which is
/// `state.hh:1244` / `:1253` with this port's magnitude convention (see
/// `entropy`'s module header).
fn check_end<S: BlockView>(
    st: &S,
    g: Group,
    img: EndImage<S::W>,
    d_mrp: S::W,
    d_mrm: S::W,
    d_wr: S::W,
) -> Result<(), AuditError> {
    let fail = |field, expected: S::W, found: S::W| AuditError::EndScalar {
        r: g.index(),
        field,
        expected: expected.to_f64(),
        found: found.to_f64(),
    };

    // `_mrp[r]`. This is the check with teeth: the entries move the edge
    // weight and the header claims the degree, and nothing but this equality
    // ties the two together. A recorder that misses one incident edge
    // produces a delta that prices and commits perfectly and lands here.
    let expected = img.mrp + d_mrp;
    let found = st.mrp(g);
    if found != expected {
        return Err(fail("mrp", expected, found));
    }

    if <S::D as Dir>::DIRECTED {
        let expected = img.mrm + d_mrm;
        let found = st.mrm(g);
        if found != expected {
            return Err(fail("mrm", expected, found));
        }
    }

    // `_wr[r]`: the header's business alone (`state.hh:738`, `:759`); no
    // entry touches it.
    let expected = img.wr + d_wr;
    let found = st.wr(g);
    if found != expected {
        return Err(fail("wr", expected, found));
    }

    Ok(())
}

/// After a commit: every pair's new weight equals `mrs_before + delta`, and
/// every endpoint scalar matches the header plus the degree deltas.
///
/// Takes a [`Receipt`], not a `&Transition`: the transition has been consumed
/// by the commit, and checking *before* the commit is not the same check --
/// it fails on every non-zero delta by construction.
///
/// ## What it costs
///
/// `2 * entries.len()` pair reads plus three scalar reads per present endpoint
/// (two when undirected), and nothing else. No allocation, no `B^2` sweep, no
/// state copy -- which is the entire difference from `copy_state_wrap`
/// (`base_states.py:44-59`), and the reason this one is not behind a switch
/// that ships off.
///
/// ## What it assumes
///
/// That the receipt lists each block pair at most once. [`DeltaBuf`] keys one
/// entry per field cell and so guarantees it
/// (`crate::delta::DeltaBuf::touch_dyn`); a hand-built [`Receipt`] that
/// repeats a pair would have its second occurrence checked against a weight
/// the first already moved.
///
/// [`DeltaBuf`]: crate::delta::DeltaBuf
pub fn audit_commit<S: BlockView>(st: &S, r: &Receipt<S::W>) -> Result<(), AuditError> {
    // The commit consumed one revision. A receipt whose epoch has *not* been
    // overtaken was never applied to this state -- the caller is auditing the
    // wrong pair -- and a foreign `StateId` is defect #36 arriving one step
    // late. `>=` rather than `+ 1 ==`, because `commit_shared` advances the
    // epoch once per concurrent commit and this one is not the only writer.
    let state = st.stamp();
    if r.stamp.state != state.state || r.stamp.epoch >= state.epoch {
        return Err(stamp_error(r.stamp, state));
    }

    for e in &r.entries {
        let expected = e.mrs_before + e.delta;
        let found = live_pair(st, e.r, e.s);
        if found != expected {
            return Err(AuditError::EdgeWeight {
                r: e.r.index(),
                s: e.s.index(),
                expected: expected.to_f64(),
                found: found.to_f64(),
            });
        }
    }

    let h = &r.hdr;
    let zero = S::W::ZERO;
    match (h.r, h.nr) {
        // `modify_entries` never records `from == to` (`entries.hh:322-332`;
        // `virtual_move_groups` returns 0 at `state.hh:1341`), but if one
        // arrives, `apply_level` applies `-dr` and `+dnr` to the *same* row
        // and the two-endpoint arm below would check each half against the
        // whole. The net move is the identity, so that is what is checked.
        (Some(a), Some(b)) if a == b => check_end(st, a, h.r_img, zero, zero, zero),
        _ => {
            if let Some(g) = h.r {
                check_end(st, g, h.r_img, -h.dkout, -h.dkin, -h.dr)?;
            }
            if let Some(g) = h.nr {
                check_end(st, g, h.nr_img, h.dkout, h.dkin, h.dnr)?;
            }
            Ok(())
        }
    }
}

/// The arithmetic of [`audit_price`], over live state reads.
///
/// Deliberately *not* factored with `entropy::sparse_terms`: the point of the
/// check is that the two expressions take their before-image from different
/// places, and sharing a body would delete the only independence there is.
fn recompute_ds<S, D: Dir, W: Weight>(
    st: &S,
    entries: &[Entry<W>],
    hdr: &MoveHeader<W>,
    deg_corr: bool,
    c: &Cache,
) -> f64
where
    S: BlockView<D = D, W = W>,
{
    let mut ds = 0.0;

    // `state.hh:1224-1232`, read the way the C++ reads it: `_emat.get_me(t, w)`
    // then `_mrs[me]`, live, rather than `Entry::mrs_before`.
    for e in entries {
        let (r, s) = (e.r.index(), e.s.index());
        let ers = live_pair(st, e.r, e.s).to_f64();
        ds += eterm::<D>(r, s, ers + e.delta.to_f64(), c) - eterm::<D>(r, s, ers, c);
    }

    let dkin = hdr.dkin.to_f64();
    let dkout = hdr.dkout.to_f64();

    // `state.hh:1239-1246`, against `_mrp`/`_mrm`/`_wr` rather than
    // `MoveHeader::r_img`.
    if let Some(g) = hdr.r {
        let (mrp, mrm, wr) = scalars(st, g);
        ds += vterm::<D>(mrp - dkout, mrm - dkin, wr - hdr.dr.to_f64(), deg_corr, c);
        ds -= vterm::<D>(mrp, mrm, wr, deg_corr, c);
    }

    // `state.hh:1248-1255`
    if let Some(g) = hdr.nr {
        let (mrp, mrm, wr) = scalars(st, g);
        ds += vterm::<D>(mrp + dkout, mrm + dkin, wr + hdr.dnr.to_f64(), deg_corr, c);
        ds -= vterm::<D>(mrp, mrm, wr, deg_corr, c);
    }

    ds
}

/// Recompute the entropy change locally and compare it with the claim.
///
/// Only terms touching an entry or an endpoint group change, so this costs the
/// same order as pricing rather than the O(E + B^2) of a full recompute.
///
/// This is the **sparse** adjacency delta, i.e. the claim
/// [`sparse_ds`](super::sparse_ds) makes and no other. It deliberately does
/// not add a partition or degree description length: `sparse_ds` does not
/// either (`get_delta_partition_dl` is applied by the caller at
/// `state.hh:1352`), and an audit that priced more terms than the thing it
/// audits would report every correct move as wrong. Only
/// [`EntropyParams::deg_corr`] is therefore read, exactly as `sparse_ds` reads
/// only that field.
///
/// The dense branch has no analogue here: `dense_ds` already reads the state
/// for every term (`state.hh:1145-1219`), so a "recompute from the state"
/// audit of it would be the identity function.
pub fn audit_price<S, D: Dir, W: Weight>(
    st: &S,
    t: &Transition<'_, D, W>,
    level: usize,
    params: EntropyParams,
    c: &Cache,
    ds_claimed: f64,
) -> Result<(), AuditError>
where
    S: BlockView<D = D, W = W>,
{
    // Pre-commit, so both halves of the stamp must match exactly: pricing a
    // delta against a state it was not recorded against is the same defect as
    // committing one there, one step earlier and much cheaper to find.
    let state = st.stamp();
    let delta = t.stamp();
    if delta != state {
        return Err(stamp_error(delta, state));
    }

    let d = t.level(level);
    let recomputed = recompute_ds::<S, D, W>(st, d.entries(), d.header(), params.deg_corr, c);
    agree(ds_claimed, recomputed)
}

/// The expensive global recompute, for a periodic sweep-level drift check.
///
/// `sparse_entropy` (`entropy.hh:185-206`): `eterm` over every block-graph
/// edge and `vterm` over every block-graph vertex. The `ea.constants` block
/// (`:207-221`) is not here -- `get_deg_entropy` and `get_parallel_entropy`
/// are functions of the *graph*, which [`BlockView`] does not carry, and both
/// are invariant under a vertex move, so neither can drift against an
/// accumulated `sparse_ds`.
///
/// **Deviation from the skeleton ([DESIGN](gt_core::design) rule 7).** `slots` was added, for
/// the same reason `dense_ds` needed it: the sum runs over the block graph,
/// [`BlockView::n_groups`] is `_actual_B` -- the number of *occupied* groups,
/// not an index bound -- and the trait offers no way to enumerate block-graph
/// edges. `slots` is `num_vertices(_bg)`, exactly the third argument of
/// `m_entries.set_move(r, s, num_vertices(_bg))` (`state.hh:1107`).
///
/// Sweeping every slot pair rather than the block graph's edge list costs
/// O(slots^2) instead of O(|E_bg|) and gives the same number: `find_me`
/// returns `None` exactly where the pair weight is zero -- `update_rs` removes
/// the block edge there (`entries.hh:405-427`) -- and `eterm(r, s, 0)` is
/// `-lgamma(1)`, i.e. zero, in both arms (`entropy.hh:44-59`). An unoccupied
/// slot's `vterm` is zero for the same reason in the degree-corrected arm and
/// because `xlogx(0) == 0` (`cache.hh:126-131`) in the other.
///
/// # Panics
///
/// If `slots` reaches [`Group`]'s sentinel. `Group::new(u32::MAX)` is `None`
/// (the port of `null_group = INT64_MAX`), so the top slot is not addressable
/// and silently skipping it would drop a real row from the sum.
#[cfg(feature = "audit-full")]
pub fn audit_absolute<S: BlockView>(
    st: &S,
    slots: usize,
    params: EntropyParams,
    c: &Cache,
    claimed: f64,
) -> Result<(), AuditError> {
    assert!(
        slots < u32::MAX as usize,
        "{slots} block-graph slots reaches the null-group sentinel \
         (blockmodel/spec.hh:87); the limit is {}",
        u32::MAX as usize - 1
    );
    let group = |i: usize| Group::new(i as u32).expect("slot index below the sentinel");

    let mut s = 0.0;
    for ri in 0..slots {
        let rg = group(ri);

        // `parallel_edge_loop_no_spawn(bg, ...)` (`entropy.hh:193-198`). An
        // undirected block graph holds one edge per unordered pair -- `put_me`
        // writes the same descriptor to `_mat[r][s]` and `_mat[s][r]`
        // (`emat.hh:72-77`) -- so the lower triangle would double-count it.
        let first = if <S::D as Dir>::DIRECTED { 0 } else { ri };
        for si in first..slots {
            let sg = group(si);
            if let Some(e) = st.find_me(rg, sg) {
                s += eterm::<S::D>(ri, si, st.mrs(e).to_f64(), c);
            }
        }

        // `parallel_vertex_loop_no_spawn(bg, ...)` (`entropy.hh:200-206`).
        let (mrp, mrm, wr) = scalars(st, rg);
        s += vterm::<S::D>(mrp, mrm, wr, params.deg_corr, c);
    }

    agree(claimed, s)
}

// ---------------------------------------------------------------------------
// The `audit-full` acceptance test.
//
// It lives here and not in `tests/u26_audit.rs` because `audit_absolute` is
// feature-gated and `blockmodel/mod.rs` -- U22's file, which U26 does not own
// -- lists only `AuditError`, `audit_commit` and `audit_price` in its
// `pub use`. Everything else U26 promises is pinned from outside the crate.
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "audit-full"))]
mod sweep {
    use std::collections::BTreeMap;

    use gt_core::dir::Directed;
    use gt_core::ids::VertexId;

    use super::*;
    use crate::blockmodel::state::{BlockCommit, BlockState};
    use crate::delta::{MoveKey, Recording, Workspace};

    fn grp(i: u32) -> Group {
        Group::new(i).expect("group index in range")
    }

    /// A deterministic generator, so a failure reproduces from the test name.
    struct Lcg(u64);
    impl Lcg {
        fn below(&mut self, n: u32) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as u32) % n
        }
    }

    /// Read the recomputation out of [`audit_absolute`] by handing it a claim
    /// it cannot accept.
    ///
    /// There is deliberately no getter. The function's job is to *judge*, and
    /// a test that wants the number pays for it with a comparison that is
    /// guaranteed to fail: every comparison against `NaN` is false, which is
    /// the same property that makes `agree` report a `NaN` claim.
    fn absolute_of<S: BlockView>(st: &S, slots: usize, ea: EntropyParams, c: &Cache) -> f64 {
        match audit_absolute(st, slots, ea, c, f64::NAN) {
            Err(AuditError::Entropy { recomputed, .. }) => recomputed,
            other => panic!("a NaN claim must never be accepted, got {other:?}"),
        }
    }

    /// `modify_entries_dispatch` (`entries.hh:227-320`) for a directed graph,
    /// transcribed independently of U25's `record` so that a defect in either
    /// one cannot cancel against the other. Kept independent now that `record`
    /// has landed; the two are checked against each other from
    /// `tests/u26_audit.rs`.
    fn plan(
        b: &[u32],
        edges: &[(usize, usize, i64)],
        v: usize,
        nr: u32,
    ) -> (Vec<(Group, Group, i64)>, i64, i64) {
        let r = b[v];
        let mut acc: BTreeMap<(u32, u32), i64> = BTreeMap::new();
        let (mut dkin, mut dkout) = (0i64, 0i64);
        for &(a, t, w) in edges {
            if a == v {
                let after = if t == v { nr } else { b[t] };
                *acc.entry((r, b[t])).or_insert(0) -= w;
                *acc.entry((nr, after)).or_insert(0) += w;
                dkout += w;
                if t == v {
                    dkin += w;
                }
            } else if t == v {
                *acc.entry((b[a], r)).or_insert(0) -= w;
                *acc.entry((b[a], nr)).or_insert(0) += w;
                dkin += w;
            }
        }
        let pairs = acc
            .into_iter()
            .map(|((a, t), d)| (grp(a), grp(t), d))
            .collect();
        (pairs, dkin, dkout)
    }

    /// Ten thousand moves, each priced incrementally, against one from-scratch
    /// recomputation of the whole functional at the end.
    ///
    /// This is `copy_state_wrap` (`base_states.py:44-59`) doing the job it was
    /// written for -- once per sweep instead of once per call, and without
    /// copying the state. The always-on audits run on every move as well, so a
    /// failure here after they all passed would mean the *incremental
    /// expression* is wrong rather than the bookkeeping, which is exactly the
    /// split the two checks exist to make.
    #[test]
    fn ten_thousand_priced_moves_agree_with_one_absolute_recompute() {
        const SLOTS: usize = 8;
        const N: usize = 40;
        const MOVES: usize = 10_000;

        for deg_corr in [false, true] {
            let ea = EntropyParams {
                deg_corr,
                ..EntropyParams::default()
            };
            let c = Cache::build(4096);
            let mut rng = Lcg(0x5EED_0026 | 1);

            let mut b: Vec<u32> = (0..N).map(|i| (i % SLOTS) as u32).collect();
            let mut edges = Vec::new();
            for _ in 0..140 {
                let a = rng.below(N as u32) as usize;
                let t = rng.below(N as u32) as usize;
                edges.push((a, t, 1 + i64::from(rng.below(3))));
            }

            let mut st = BlockState::<Directed, i64>::new(SLOTS, N);
            for (v, &g) in b.iter().enumerate() {
                st.assign(VertexId::from_index(v), grp(g), 1);
            }
            for &(a, t, w) in &edges {
                st.seed_pair(grp(b[a]), grp(b[t]), w);
            }

            let s0 = absolute_of(&st, SLOTS, ea, &c);
            assert!(
                s0.is_finite() && s0 != 0.0,
                "the fixture must have an entropy"
            );
            assert_eq!(audit_absolute(&st, SLOTS, ea, &c, s0), Ok(()));

            let mut acc = 0.0f64;
            let mut applied = 0usize;
            for _ in 0..MOVES {
                let v = rng.below(N as u32) as usize;
                let nr = rng.below(SLOTS as u32);
                if nr == b[v] {
                    continue;
                }
                let (pairs, dkin, dkout) = plan(&b, &edges, v, nr);
                let hdr = MoveHeader {
                    r: Some(grp(b[v])),
                    nr: Some(grp(nr)),
                    r_img: st.end_image(Some(grp(b[v]))),
                    nr_img: st.end_image(Some(grp(nr))),
                    dkin,
                    dkout,
                    dr: 1,
                    dnr: 1,
                };

                let mut ws = Workspace::<Directed, i64>::with_levels(1);
                let t = {
                    let mut rec = Recording::new(&mut ws.stack, hdr, st.stamp());
                    rec.level_mut(0).begin(
                        MoveKey {
                            from: hdr.r,
                            to: hdr.nr,
                        },
                        SLOTS,
                    );
                    let mut resolve = |x, y| st.resolve(x, y);
                    for &(x, y, w) in &pairs {
                        rec.level_mut(0)
                            .touch_dyn(x, y, w, &mut resolve)
                            .expect("every pair touches an endpoint of the move");
                    }
                    rec.seal()
                };

                let ds = crate::blockmodel::sparse_ds(t.level(0), ea, &c);
                assert_eq!(audit_price(&st, &t, 0, ea, &c, ds), Ok(()));

                let receipt = st.commit(t.into_levels().next().expect("one level"));
                assert_eq!(audit_commit(&st, &receipt), Ok(()));

                acc += ds;
                b[v] = nr;
                applied += 1;
            }

            assert!(applied > 8_000, "only {applied} of {MOVES} moves ran");

            // The acceptance bound, asserted on the raw difference and not
            // merely delegated to the audit's own tolerance.
            let end = absolute_of(&st, SLOTS, ea, &c);
            let drift = (s0 + acc) - end;
            assert!(
                drift.abs() < 1e-9,
                "deg_corr = {deg_corr}: {applied} incremental moves drifted \
                 {drift:e} from the absolute functional"
            );
            assert_eq!(audit_absolute(&st, SLOTS, ea, &c, s0 + acc), Ok(()));
        }
    }
}
