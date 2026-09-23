//! The entropy functional.
//!
//! ## Sign conventions, stated once
//!
//! graph-tool carries the endpoint deltas *signed*: `entries_dS` is called as
//! `entries_dS(s, t, -n, n, ...)` (`blockmodel/state.hh:1349`), so its `dr` is
//! already negative and every use is a bare `+=` (`:1158`, `:1244`). The
//! before-image this port records instead names the two halves
//! ([`MoveHeader::dr`](crate::delta::MoveHeader::dr) is "vertex weight
//! *leaving* `r`", [`dnr`](crate::delta::MoveHeader::dnr) is "vertex weight
//! *entering* `nr`"), exactly as `dkin`/`dkout` are already magnitudes with
//! the sign supplied at the use site (`:1244` subtracts, `:1253` adds). So all
//! four header deltas are non-negative here and this module applies
//!
//! ```text
//!   wr[r]  -> wr[r]  - dr      mrp[r]  -> mrp[r]  - dkout   mrm[r]  -> mrm[r]  - dkin
//!   wr[nr] -> wr[nr] + dnr     mrp[nr] -> mrp[nr] + dkout   mrm[nr] -> mrm[nr] + dkin
//! ```
//!
//! which is `state.hh:1244-1254` term for term.

use gt_core::dir::Dir;

use super::cache::Cache;
use super::state::BlockView;
use crate::delta::{Delta, Entry, MoveHeader};
use crate::ids::{Group, Weight};

/// Degree correction and prior switches.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct EntropyParams {
    /// Degree-corrected variant.
    pub deg_corr: bool,
    /// Treat parallel edges as distinguishable.
    pub multigraph: bool,
    /// Include the description length of the partition.
    pub partition_dl: bool,
    /// Include the description length of the edge counts.
    pub degree_dl: bool,
}

impl crate::spec::EntropyArgs for EntropyParams {}

/// The edge term. `inference/blockmodel/entropy.hh:38` (`eterm_d`).
#[inline]
pub fn eterm<D: Dir>(r: usize, s: usize, mrs: f64, c: &Cache) -> f64 {
    debug_assert!(
        mrs >= 0.0,
        "eterm: mrs must be non-negative (entropy.hh:40)"
    );

    let val = c.lgamma1p(mrs);

    // `entropy.hh:44-59`: the `-mrs * log(2)` correction is the undirected
    // self-loop arm only, and `D::DIRECTED` folds the outer branch away at
    // monomorphisation where the C++ has `if constexpr (directed)`.
    if D::DIRECTED || r != s {
        -val
    } else {
        -val - mrs * std::f64::consts::LN_2
    }
}

/// The vertex term. `entropy.hh:73` (`vterm_d`).
#[inline]
pub fn vterm<D: Dir>(mrp: f64, mrm: f64, wr: f64, deg_corr: bool, c: &Cache) -> f64 {
    if deg_corr {
        // `entropy.hh:77-80`
        if D::DIRECTED {
            c.lgamma1p(mrp) + c.lgamma1p(mrm)
        } else {
            c.lgamma1p(mrp)
        }
    } else {
        // `entropy.hh:84-87`
        if D::DIRECTED {
            (mrp + mrm) * c.safelog(wr)
        } else {
            mrp * c.safelog(wr)
        }
    }
}

/// The dense-model edge term. `entropy.hh:235` (`eterm_dense_d`).
///
/// **Deviation from the skeleton ([DESIGN](gt_core::design) rule 7).** `r` and `s` were added.
/// `eterm_dense_d` branches on `directed || r != s` (`entropy.hh:241`) to pick
/// between `wr_r * wr_s` and the triangular self-pair count, and the two
/// disagree by a factor of roughly two; with only the two vertex counts in
/// hand the undirected self-loop arm is unreachable and the term is silently
/// wrong for every diagonal block pair. There is no way to recover `r == s`
/// from `wr_r` and `wr_s`.
#[inline]
pub fn eterm_dense<D: Dir>(
    r: usize,
    s: usize,
    mrs: f64,
    wr_r: f64,
    wr_s: f64,
    multigraph: bool,
    c: &Cache,
) -> f64 {
    debug_assert!(
        wr_r + wr_s >= 0.0,
        "eterm_dense: wr_r + wr_s must be non-negative (entropy.hh:238)"
    );

    // `entropy.hh:240-251`. The C++ accumulates into a `uint64_t`; the two
    // triangular forms are products of consecutive integers and therefore
    // exactly halvable, so evaluating them in `f64` is exact over the whole
    // range where `wr * wr` is exact at all.
    let nrns = if D::DIRECTED || r != s {
        wr_r * wr_s
    } else if multigraph {
        (wr_r * (wr_r + 1.0)) / 2.0
    } else {
        (wr_r * (wr_r - 1.0)) / 2.0
    };

    // `entropy.hh:253-256`
    if multigraph {
        c.lbinom(nrns + mrs - 1.0, mrs)
    } else {
        c.lbinom(nrns, mrs)
    }
}

/// The description length of the edge counts. `entropy.hh:293`.
///
/// **Deviation from the skeleton ([DESIGN](gt_core::design) rule 7).** The `D: Dir` parameter
/// was added: `get_edges_dl` chooses `B * B` or `B * (B + 1) / 2` on
/// `is_directed(g)` (`entropy.hh:295`), and the skeleton carried no
/// directedness at all.
pub fn edges_dl<D: Dir>(n_groups: usize, n_edges: f64, c: &Cache) -> f64 {
    let b = n_groups as f64;
    let bb = if D::DIRECTED {
        b * b
    } else {
        (b * (b + 1.0)) / 2.0
    };
    c.lbinom(bb + n_edges - 1.0, n_edges)
}

/// The description length of the partition. `partition.hh:106`.
///
/// `sizes` is `_count`, one slot per group *index* (zero for an unoccupied
/// one), and `n` is `_N`, the total vertex weight. The group count the prior
/// is written against is `_actual_B`, the number of **occupied** groups, which
/// is derived here rather than carried: `lgamma(0 + 1) == 0`, so the empty
/// slots drop out of the product term on their own and only the
/// `lbinom(N - 1, B - 1)` term needs the distinction.
pub fn partition_dl(sizes: &[usize], n: usize, c: &Cache) -> f64 {
    // `partition.hh:109-110`
    if n == 0 {
        return 0.0;
    }

    let actual_b = sizes.iter().filter(|&&nr| nr != 0).count();
    if actual_b == 0 {
        // Unreachable for a consistent `(sizes, n)`: `n > 0` implies some slot
        // is occupied. graph-tool reaches the same answer by accident -- its
        // `_actual_B - 1` underflows a `size_t` and `lbinom_fast`'s
        // `cmp_greater_equal(k, N)` guard (`util.hh:44`) then returns 0 -- but
        // relying on unsigned wraparound to land on the right branch is not
        // something to reproduce.
        debug_assert!(false, "partition_dl: N > 0 with every group empty");
        return 0.0;
    }

    let nf = n as f64;
    let mut s = c.lbinom(nf - 1.0, actual_b as f64 - 1.0);
    s += c.lgamma1p(nf);
    for &nr in sizes {
        s -= c.lgamma1p(nr as f64);
    }
    s += c.safelog(nf);
    s
}

/// Price a sparse-model transition **with no access to the state at all**.
///
/// This is the payoff of the entry-carried before-image: the loop is a
/// contiguous scan over `&[Entry<W>]` with no pointer chasing into `_emat` or
/// `_mrs`, where the C++ (`blockmodel/state.hh:1223-1233`) performs a hash
/// lookup and an indirect load per entry into structures sized O(B) and O(E)
/// that are cold at this point in the sweep.
///
/// Because it borrows only the delta and the cache, it composes freely with
/// [`move_prob`](super::state::BlockView::move_prob), which borrows `&state`,
/// and with the `&mut state` that follows.
pub fn sparse_ds<D: Dir, W: Weight>(
    delta: Delta<'_, D, W>,
    params: EntropyParams,
    c: &Cache,
) -> f64 {
    sparse_terms::<D, W>(delta.entries(), delta.header(), params.deg_corr, c)
}

/// The arithmetic of [`sparse_ds`], over the two pieces of the before-image it
/// actually reads.
///
/// Split out so that the kernel can be exercised on hand-built entries: a
/// [`Delta`] can only be minted by the recording lifecycle, and `Entry` /
/// `MoveHeader` are plain public data.
fn sparse_terms<D: Dir, W: Weight>(
    entries: &[Entry<W>],
    hdr: &MoveHeader<W>,
    deg_corr: bool,
    c: &Cache,
) -> f64 {
    let mut ds = 0.0;

    // `state.hh:1224-1232`. Note the C++ does *not* skip a zero delta here
    // (unlike the dense branch at `:1176`), and neither does this: for a
    // finite `ers` the term is exactly zero anyway, and for a non-finite one
    // skipping would turn a `NaN` into a `0.0` and hide the defect.
    for e in entries {
        let (r, s) = (e.r.index(), e.s.index());
        let ers = e.mrs_before;
        ds += eterm::<D>(r, s, (ers + e.delta).to_f64(), c) - eterm::<D>(r, s, ers.to_f64(), c);
    }

    let dkin = hdr.dkin.to_f64();
    let dkout = hdr.dkout.to_f64();

    // `state.hh:1239-1246`
    if hdr.r.is_some() {
        let i = hdr.r_img;
        let (mrp, mrm, wr) = (i.mrp.to_f64(), i.mrm.to_f64(), i.wr.to_f64());
        ds += vterm::<D>(mrp - dkout, mrm - dkin, wr - hdr.dr.to_f64(), deg_corr, c);
        ds -= vterm::<D>(mrp, mrm, wr, deg_corr, c);
    }

    // `state.hh:1248-1255`
    if hdr.nr.is_some() {
        let i = hdr.nr_img;
        let (mrp, mrm, wr) = (i.mrp.to_f64(), i.mrm.to_f64(), i.wr.to_f64());
        ds += vterm::<D>(mrp + dkout, mrm + dkin, wr + hdr.dnr.to_f64(), deg_corr, c);
        ds -= vterm::<D>(mrp, mrm, wr, deg_corr, c);
    }

    ds
}

/// Price a dense-model transition.
///
/// This one genuinely needs the state, and the signature says so. The dense
/// branch (`state.hh:1145-1219`) sweeps `out_edges_range(t, _bg)` and
/// `in_edges_range`, i.e. it reads block-graph edges that are **not** in the
/// entry set. That is fine -- it is a shared borrow and composes with the
/// other readers -- but it means the state-free pricing signature holds for
/// the sparse model only, and presenting it as unconditional would be false.
///
/// **Deviation from the skeleton ([DESIGN](gt_core::design) rule 7).** `slots` was added: it
/// is `num_vertices(_bg)`, the number of block-graph *slots*, exactly the
/// third argument of `m_entries.set_move(r, s, num_vertices(_bg))`
/// (`state.hh:1107`). [`BlockView`] exposes no way to enumerate block-graph
/// edges and its `n_groups()` is `_actual_B`, the number of *occupied* groups,
/// which is not an index bound. Without a slot count the two sweeps at
/// `:1188` and `:1197` have no domain.
///
/// The sweep here is over every slot rather than over `t`'s incident block
/// edges, which is a wider set but not a different answer: for a pair with no
/// block edge, `ers == 0` makes both `lbinom` calls hit the `k == 0` guard
/// (`util.hh:44`) and the pair contributes exactly `0 - 0`. It costs O(B)
/// instead of O(deg_bg(t)); the dense model is not the hot path, and D10
/// already records that this branch cannot be made state-free.
pub fn dense_ds<S, D: Dir, W: Weight>(
    state: &S,
    delta: Delta<'_, D, W>,
    slots: usize,
    params: EntropyParams,
    c: &Cache,
) -> f64
where
    S: super::state::BlockView<D = D, W = W>,
{
    // `state.hh:1140-1141` / `entropy.hh:274-275`.
    assert!(
        !params.deg_corr,
        "dense entropy for the degree-corrected model is not implemented \
         (entropy.hh:275, state.hh:1141)"
    );
    dense_terms::<S, D, W>(
        state,
        delta.entries(),
        delta.header(),
        slots,
        params.multigraph,
        c,
    )
}

/// The arithmetic of [`dense_ds`]. Split out for the same reason as
/// [`sparse_terms`].
fn dense_terms<S, D: Dir, W: Weight>(
    st: &S,
    entries: &[Entry<W>],
    hdr: &MoveHeader<W>,
    slots: usize,
    multigraph: bool,
    c: &Cache,
) -> f64
where
    S: BlockView<D = D, W = W>,
{
    let mut ds = 0.0;

    // `state.hh:1173-1180`
    for e in entries {
        if e.delta == W::ZERO {
            continue;
        }
        ds += dense_pair_ds::<S, D, W>(
            st,
            hdr,
            e.r,
            e.s,
            e.mrs_before.to_f64(),
            e.delta.to_f64(),
            multigraph,
            c,
        );
    }

    let dr = hdr.dr.to_f64();
    let dnr = hdr.dnr.to_f64();

    // `state.hh:1182-1220`. The two endpoints, in order, with the `r == nr`
    // break that stops the diagonal pair being counted twice.
    for end in [hdr.r, hdr.nr] {
        let Some(t) = end else {
            continue; // `:1184-1185`
        };

        // `:1186`
        let moved = (hdr.r == Some(t) && dr != 0.0) || (hdr.nr == Some(t) && dnr != 0.0);
        if moved {
            for ui in 0..slots {
                let Some(u) = Group::new(ui as u32) else {
                    continue;
                };
                if hdr.r == Some(u) || hdr.nr == Some(u) {
                    continue;
                }
                // `:1188-1195`, the out-half.
                if recorded_delta::<D, W>(entries, t, u) == W::ZERO {
                    let ers = pair_weight(st, t, u);
                    ds += dense_pair_ds::<S, D, W>(st, hdr, t, u, ers, 0.0, multigraph, c);
                }
                // `:1197-1204`, the in-half. `in_edges` on an undirected
                // adaptor is an *empty* range (`graph_adaptor.hh:219-227`), so
                // this pass exists for the directed instantiation only --
                // running it undirected would double-count every pair.
                if D::DIRECTED && recorded_delta::<D, W>(entries, u, t) == W::ZERO {
                    let ers = pair_weight(st, u, t);
                    ds += dense_pair_ds::<S, D, W>(st, hdr, u, t, ers, 0.0, multigraph, c);
                }
            }
        }

        // `:1207-1208`
        if let Some(r) = hdr.r
            && recorded_delta::<D, W>(entries, t, r) == W::ZERO
        {
            let ers = pair_weight(st, t, r);
            ds += dense_pair_ds::<S, D, W>(st, hdr, t, r, ers, 0.0, multigraph, c);
        }

        // `:1212-1213`
        if hdr.r == hdr.nr {
            break;
        }

        // `:1215-1217`
        if let Some(nr) = hdr.nr {
            let distinct = D::DIRECTED || Some(t) != hdr.r;
            if distinct && recorded_delta::<D, W>(entries, t, nr) == W::ZERO {
                let ers = pair_weight(st, t, nr);
                ds += dense_pair_ds::<S, D, W>(st, hdr, t, nr, ers, 0.0, multigraph, c);
            }
        }
    }

    ds
}

/// `get_dS` (`state.hh:1143-1171`): one block pair, before and after.
#[allow(clippy::too_many_arguments)]
#[inline]
fn dense_pair_ds<S, D: Dir, W: Weight>(
    st: &S,
    hdr: &MoveHeader<W>,
    t: Group,
    u: Group,
    ers: f64,
    d: f64,
    multigraph: bool,
    c: &Cache,
) -> f64
where
    S: BlockView<D = D, W = W>,
{
    let wt = st.wr(t).to_f64();
    let wu = st.wr(u).to_f64();

    let mut dd = -eterm_dense::<D>(t.index(), u.index(), ers, wt, wu, multigraph, c);
    dd += eterm_dense::<D>(
        t.index(),
        u.index(),
        ers + d,
        shifted(hdr, t, wt),
        shifted(hdr, u, wu),
        multigraph,
        c,
    );

    debug_assert!(
        dd.is_finite(),
        "dense_pair_ds: non-finite term (state.hh:1169)"
    );
    dd
}

/// A group's vertex weight after the move. `state.hh:1157-1165`, with the sign
/// convention of this module's header comment.
#[inline]
fn shifted<W: Weight>(hdr: &MoveHeader<W>, g: Group, w: f64) -> f64 {
    let mut w = w;
    if hdr.r == Some(g) {
        w -= hdr.dr.to_f64();
    }
    if hdr.nr == Some(g) {
        w += hdr.dnr.to_f64();
    }
    w
}

/// The block pair's weight *before* the move, read from the state.
#[inline]
fn pair_weight<S: BlockView>(st: &S, r: Group, s: Group) -> f64 {
    match st.find_me(r, s) {
        Some(e) => st.mrs(e).to_f64(),
        None => 0.0,
    }
}

/// `EntrySet::get_delta` (`entries.hh:156`) over the sealed entry slice.
///
/// Orientation matters only when directed. Undirected lookups are symmetric
/// because `get_field`'s undirected arm keys both `(s, t)` and `(t, s)` on the
/// same out-field slot (`entries.hh:99-105`), so one unordered block pair is
/// one entry however the recorder happened to name its ends.
///
/// A linear scan, not a map: the entry set of one move is the distinct block
/// pairs touched by one vertex's incidence, which is small and contiguous, and
/// this path runs only in the dense branch.
#[inline]
fn recorded_delta<D: Dir, W: Weight>(entries: &[Entry<W>], a: Group, b: Group) -> W {
    for e in entries {
        let hit = if D::DIRECTED {
            e.r == a && e.s == b
        } else {
            (e.r == a && e.s == b) || (e.r == b && e.s == a)
        };
        if hit {
            return e.delta;
        }
    }
    W::ZERO
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{EndImage, MoveKey, Recording, Transition, Workspace};
    use crate::ids::{BEdge, Epoch, Stamp, StateId};
    use gt_core::dir::{Directed, Undirected};
    use gt_core::ids::VertexId;

    const B: usize = 3;

    fn cache() -> Cache {
        Cache::build(1 << 12)
    }

    fn g(i: u32) -> Group {
        Group::new(i).unwrap()
    }

    fn stamp() -> Stamp {
        Stamp {
            state: StateId::fresh(),
            epoch: Epoch(0),
        }
    }

    // ---- reference recomputes, transcribed from `entropy.hh` -------------

    /// `mrp` / `mrm` of a dense block-weight matrix.
    ///
    /// Undirected block graphs store one entry per *unordered* pair, and the
    /// self-pair counts twice in the degree, which is what makes
    /// `eterm_d`'s `- mrs * log(2)` arm (`entropy.hh:58`) the right correction.
    fn degrees<D: Dir>(mrs: &[i64], slots: usize) -> (Vec<i64>, Vec<i64>) {
        let mut mrp = vec![0i64; slots];
        let mut mrm = vec![0i64; slots];
        for r in 0..slots {
            for t in 0..slots {
                let e = mrs[r * slots + t];
                if D::DIRECTED {
                    mrp[r] += e;
                    mrm[t] += e;
                } else if t > r {
                    mrp[r] += e;
                    mrp[t] += e;
                } else if t == r {
                    mrp[r] += 2 * e;
                }
            }
        }
        (mrp, mrm)
    }

    /// `sparse_entropy` (`entropy.hh:185-226`), adjacency terms only: the two
    /// `parallel_*` loops over `bg`, summed from scratch.
    fn sparse_absolute<D: Dir>(mrs: &[i64], wr: &[i64], slots: usize, dc: bool, c: &Cache) -> f64 {
        let (mrp, mrm) = degrees::<D>(mrs, slots);
        let mut s = 0.0;
        for r in 0..slots {
            for t in 0..slots {
                if !D::DIRECTED && t < r {
                    continue;
                }
                s += eterm::<D>(r, t, mrs[r * slots + t] as f64, c);
            }
        }
        for r in 0..slots {
            s += vterm::<D>(mrp[r] as f64, mrm[r] as f64, wr[r] as f64, dc, c);
        }
        s
    }

    /// `dense_entropy` (`entropy.hh:269-289`), summed over every block pair
    /// rather than over the block graph's edges -- the same number, because
    /// `eterm_dense` is exactly zero wherever `ers` is (`util.hh:44`).
    fn dense_absolute<D: Dir>(mrs: &[i64], wr: &[i64], slots: usize, mg: bool, c: &Cache) -> f64 {
        let mut s = 0.0;
        for r in 0..slots {
            for t in 0..slots {
                if !D::DIRECTED && t < r {
                    continue;
                }
                s += eterm_dense::<D>(
                    r,
                    t,
                    mrs[r * slots + t] as f64,
                    wr[r] as f64,
                    wr[t] as f64,
                    mg,
                    c,
                );
            }
        }
        s
    }

    // ---- a stand-in block state -----------------------------------------
    //
    // `dense_ds` reads `wr`, `find_me` and `mrs` and nothing else. The rest of
    // `BlockView` is left `unimplemented!()` on purpose: if the dense pricing
    // ever starts reading `mrp`, `group_of` or `move_prob` these tests say so
    // instead of quietly agreeing.
    struct Stub {
        wr: Vec<i64>,
        mrs: Vec<i64>,
        slots: usize,
        directed: bool,
    }

    impl Stub {
        fn key(&self, r: Group, s: Group) -> usize {
            let (a, b) = (r.index(), s.index());
            let (a, b) = if self.directed || a <= b {
                (a, b)
            } else {
                (b, a)
            };
            a * self.slots + b
        }
    }

    macro_rules! impl_stub {
        ($name:ident, $dir:ty) => {
            struct $name(Stub);
            impl BlockView for $name {
                type D = $dir;
                type W = i64;
                fn stamp(&self) -> Stamp {
                    unimplemented!("dense_ds does not read the stamp")
                }
                fn group_of(&self, _v: VertexId) -> Option<Group> {
                    unimplemented!("dense_ds does not read the partition")
                }
                fn n_groups(&self) -> usize {
                    self.0.wr.iter().filter(|&&w| w != 0).count()
                }
                fn find_me(&self, r: Group, s: Group) -> Option<BEdge> {
                    let k = self.0.key(r, s);
                    (self.0.mrs[k] != 0).then(|| BEdge(k as u32))
                }
                fn mrs(&self, e: BEdge) -> i64 {
                    self.0.mrs[e.0 as usize]
                }
                fn mrp(&self, _r: Group) -> i64 {
                    unimplemented!("dense_ds does not read mrp")
                }
                fn mrm(&self, _r: Group) -> i64 {
                    unimplemented!("dense_ds does not read mrm")
                }
                fn wr(&self, r: Group) -> i64 {
                    self.0.wr[r.index()]
                }
                fn move_prob(
                    &self,
                    _t: &Transition<'_, Self::D, Self::W>,
                    _v: VertexId,
                    _c: f64,
                ) -> f64 {
                    unimplemented!("dense_ds does not read move_prob")
                }
            }
        };
    }

    impl_stub!(DirStub, Directed);
    impl_stub!(UndStub, Undirected);

    // ---- the state-free signature (the acceptance criterion) ------------

    /// Nothing but a [`Delta`] and a [`Cache`] is in scope. If `sparse_ds`
    /// ever grew a state parameter, this would stop compiling.
    fn prices_with_only_a_delta_and_a_cache<D: Dir, W: Weight>(
        delta: Delta<'_, D, W>,
        c: &Cache,
    ) -> f64 {
        sparse_ds(delta, EntropyParams::default(), c)
    }

    /// The directed three-group scenario: one vertex of weight 1 leaves group
    /// 0 for group 1, carrying one out-edge to a group-2 vertex and one
    /// in-edge from a group-1 vertex.
    ///
    /// Returns `(before, wr_before, entries, header)`.
    #[allow(clippy::type_complexity)]
    fn directed_move() -> (
        Vec<i64>,
        Vec<i64>,
        Vec<(Group, Group, i64)>,
        MoveHeader<i64>,
    ) {
        let before: Vec<i64> = vec![
            4, 2, 1, //
            3, 5, 2, //
            1, 2, 6,
        ];
        let wr_before = vec![5i64, 4, 3];
        let (r, nr) = (g(0), g(1));
        let touches = vec![
            (r, g(2), -1), // the out-edge leaves (0,2)
            (nr, g(2), 1), //             and arrives at (1,2)
            (g(1), r, -1), // the in-edge leaves (1,0)
            (g(1), nr, 1), //             and arrives at (1,1)
        ];
        let (mrp, mrm) = degrees::<Directed>(&before, B);
        let hdr = MoveHeader {
            r: Some(r),
            nr: Some(nr),
            r_img: EndImage {
                mrp: mrp[0],
                mrm: mrm[0],
                wr: wr_before[0],
            },
            nr_img: EndImage {
                mrp: mrp[1],
                mrm: mrm[1],
                wr: wr_before[1],
            },
            dkin: 1,
            dkout: 1,
            dr: 1,
            dnr: 1,
        };
        (before, wr_before, touches, hdr)
    }

    fn as_entries(before: &[i64], touches: &[(Group, Group, i64)]) -> Vec<Entry<i64>> {
        touches
            .iter()
            .map(|&(r, s, d)| Entry {
                r,
                s,
                delta: d,
                me: None,
                mrs_before: before[r.index() * B + s.index()],
            })
            .collect()
    }

    /// The acceptance test proper: price a **real** `Delta`, minted by the
    /// recording lifecycle, with nothing but the delta and the cache in scope,
    /// and check it against two from-scratch absolute entropies.
    #[test]
    fn sparse_ds_prices_a_real_delta_with_no_state() {
        let c = cache();
        let (before, wr_before, touches, hdr) = directed_move();

        let mut after = before.clone();
        for &(r, s, d) in &touches {
            after[r.index() * B + s.index()] += d;
        }
        let mut wr_after = wr_before.clone();
        wr_after[0] -= 1;
        wr_after[1] += 1;

        for &deg_corr in &[false, true] {
            let mut ws: Workspace<Directed, i64> = Workspace::default();
            let mut rec = Recording::new(&mut ws.stack, hdr, stamp());
            {
                let buf = rec.level_mut(0);
                buf.begin(
                    MoveKey {
                        from: hdr.r,
                        to: hdr.nr,
                    },
                    B,
                );
                let mut resolve = |r: Group, s: Group| {
                    let w = before[r.index() * B + s.index()];
                    (
                        (w != 0).then(|| BEdge((r.index() * B + s.index()) as u32)),
                        w,
                    )
                };
                for &(r, s, d) in &touches {
                    buf.touch_dyn(r, s, d, &mut resolve).expect("in-plane");
                }
            }
            let sealed = rec.seal();

            // Only a `Delta` and a `Cache` cross this call. The `deg_corr`
            // arm goes through `sparse_ds` directly because the state-free
            // helper hard-codes the default parameters.
            let priced = if deg_corr {
                sparse_ds(
                    sealed.level(0),
                    EntropyParams {
                        deg_corr: true,
                        ..EntropyParams::default()
                    },
                    &c,
                )
            } else {
                prices_with_only_a_delta_and_a_cache(sealed.level(0), &c)
            };

            let expected = sparse_absolute::<Directed>(&after, &wr_after, B, deg_corr, &c)
                - sparse_absolute::<Directed>(&before, &wr_before, B, deg_corr, &c);
            assert!(
                (priced - expected).abs() < 1e-9,
                "deg_corr={deg_corr}: delta priced {priced}, from scratch {expected}"
            );
        }
    }

    /// The same numbers, reached through the hand-built entry slice, so that a
    /// disagreement localises to the recorder rather than to this kernel.
    #[test]
    fn sparse_ds_matches_a_from_scratch_recompute() {
        let c = cache();
        let (before, wr_before, touches, hdr) = directed_move();
        let entries = as_entries(&before, &touches);

        let mut after = before.clone();
        for &(r, s, d) in &touches {
            after[r.index() * B + s.index()] += d;
        }
        let mut wr_after = wr_before.clone();
        wr_after[0] -= 1;
        wr_after[1] += 1;

        for &deg_corr in &[false, true] {
            let priced = sparse_terms::<Directed, i64>(&entries, &hdr, deg_corr, &c);
            let expected = sparse_absolute::<Directed>(&after, &wr_after, B, deg_corr, &c)
                - sparse_absolute::<Directed>(&before, &wr_before, B, deg_corr, &c);
            assert!(
                (priced - expected).abs() < 1e-9,
                "deg_corr={deg_corr}: priced {priced}, from scratch {expected}"
            );
        }
    }

    /// Undirected: the `- mrs * log(2)` self-pair arm and the `mrp`-only
    /// `vterm` must both be exercised, so the moved vertex's two edges land on
    /// a pair that stays off-diagonal while `mrp` moves.
    #[test]
    fn sparse_ds_matches_a_from_scratch_recompute_undirected() {
        let c = cache();
        // Canonical (`r <= s`) storage, mirrored for the degree helper.
        let mut before = vec![0i64; B * B];
        for a in 0..B {
            for b in a..B {
                let w = ((a * 2 + b * 3) % 4) as i64;
                before[a * B + b] = w;
                before[b * B + a] = w;
            }
        }
        let wr_before = vec![6i64, 5, 4];
        let (r, nr) = (g(0), g(1));
        let touches = [(r, g(2), -2i64), (nr, g(2), 2)];

        let (mrp, _) = degrees::<Undirected>(&before, B);
        let hdr = MoveHeader {
            r: Some(r),
            nr: Some(nr),
            r_img: EndImage {
                mrp: mrp[0],
                mrm: 0,
                wr: wr_before[0],
            },
            nr_img: EndImage {
                mrp: mrp[1],
                mrm: 0,
                wr: wr_before[1],
            },
            dkin: 0,
            dkout: 2,
            dr: 1,
            dnr: 1,
        };

        let entries: Vec<Entry<i64>> = touches
            .iter()
            .map(|&(a, b, d)| Entry {
                r: a,
                s: b,
                delta: d,
                me: None,
                mrs_before: before[a.index() * B + b.index()],
            })
            .collect();

        let mut after = before.clone();
        for &(a, b, d) in &touches {
            after[a.index() * B + b.index()] += d;
            if a != b {
                after[b.index() * B + a.index()] += d;
            }
        }
        let mut wr_after = wr_before.clone();
        wr_after[0] -= 1;
        wr_after[1] += 1;

        for &deg_corr in &[false, true] {
            let priced = sparse_terms::<Undirected, i64>(&entries, &hdr, deg_corr, &c);
            let expected = sparse_absolute::<Undirected>(&after, &wr_after, B, deg_corr, &c)
                - sparse_absolute::<Undirected>(&before, &wr_before, B, deg_corr, &c);
            assert!(
                (priced - expected).abs() < 1e-9,
                "deg_corr={deg_corr}: priced {priced}, from scratch {expected}"
            );
        }
    }

    #[test]
    fn sparse_ds_of_the_empty_transition_is_zero() {
        let c = cache();
        let hdr = MoveHeader::<i64>::default();
        for &deg_corr in &[false, true] {
            assert_eq!(sparse_terms::<Directed, i64>(&[], &hdr, deg_corr, &c), 0.0);
            assert_eq!(
                sparse_terms::<Undirected, i64>(&[], &hdr, deg_corr, &c),
                0.0
            );
        }
    }

    /// A move with no degree and no vertex-weight change prices at *exactly*
    /// zero, not within a tolerance: the two `vterm` halves are called with
    /// identical arguments and must cancel bit for bit.
    ///
    /// Note what is **not** asserted: that `r == nr` prices at zero when the
    /// degree deltas are non-zero. It does not, and neither does the C++ --
    /// `entries_dS` has no `r == nr` arm at all. The filter lives one level up,
    /// at `virtual_move_groups`'s `if (s == t || n == 0) return 0`
    /// (`state.hh:1341`), and reproducing it here would put the same policy in
    /// two places.
    #[test]
    fn sparse_ds_of_a_weightless_move_is_exactly_zero() {
        let c = cache();
        let img = EndImage {
            mrp: 7,
            mrm: 5,
            wr: 3,
        };
        let hdr = MoveHeader {
            r: Some(g(1)),
            nr: Some(g(2)),
            r_img: img,
            nr_img: EndImage {
                mrp: 4,
                mrm: 2,
                wr: 9,
            },
            dkin: 0,
            dkout: 0,
            dr: 0,
            dnr: 0,
        };
        for &deg_corr in &[false, true] {
            assert_eq!(
                sparse_terms::<Directed, i64>(&[], &hdr, deg_corr, &c),
                0.0,
                "deg_corr={deg_corr}"
            );
            assert_eq!(
                sparse_terms::<Undirected, i64>(&[], &hdr, deg_corr, &c),
                0.0,
                "deg_corr={deg_corr}"
            );
        }
    }

    /// A single-endpoint move -- `r == None`, the insertion case
    /// (`MoveHeader::r`'s documentation) -- must price only the `nr` half.
    #[test]
    fn sparse_ds_handles_a_half_move() {
        let c = cache();
        let hdr = MoveHeader {
            r: None,
            nr: Some(g(2)),
            r_img: EndImage {
                mrp: 99,
                mrm: 99,
                wr: 99,
            }, // must be ignored
            nr_img: EndImage {
                mrp: 4,
                mrm: 3,
                wr: 2,
            },
            dkin: 1,
            dkout: 2,
            dr: 0,
            dnr: 1,
        };
        let got = sparse_terms::<Directed, i64>(&[], &hdr, true, &c);
        let want =
            vterm::<Directed>(6.0, 4.0, 3.0, true, &c) - vterm::<Directed>(4.0, 3.0, 2.0, true, &c);
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    // ---- dense_ds --------------------------------------------------------

    #[test]
    fn dense_ds_matches_a_from_scratch_recompute_directed() {
        let c = cache();
        const N: usize = 4;
        let before: Vec<i64> = (0..N * N).map(|i| ((i * 7) % 5) as i64).collect();
        let wr_before = vec![6i64, 5, 4, 3];

        let (r, nr) = (g(0), g(2));
        let entries = vec![
            Entry {
                r,
                s: g(3),
                delta: -1,
                me: None,
                mrs_before: before[3],
            },
            Entry {
                r: nr,
                s: g(3),
                delta: 1,
                me: None,
                mrs_before: before[2 * N + 3],
            },
        ];
        let hdr = MoveHeader {
            r: Some(r),
            nr: Some(nr),
            r_img: EndImage::default(),
            nr_img: EndImage::default(),
            dkin: 0,
            dkout: 1,
            dr: 1,
            dnr: 1,
        };

        let mut after = before.clone();
        for e in &entries {
            after[e.r.index() * N + e.s.index()] += e.delta;
        }
        let mut wr_after = wr_before.clone();
        wr_after[0] -= 1;
        wr_after[2] += 1;

        for &mg in &[false, true] {
            let st = DirStub(Stub {
                wr: wr_before.clone(),
                mrs: before.clone(),
                slots: N,
                directed: true,
            });
            let priced = dense_terms::<DirStub, Directed, i64>(&st, &entries, &hdr, N, mg, &c);
            let expected = dense_absolute::<Directed>(&after, &wr_after, N, mg, &c)
                - dense_absolute::<Directed>(&before, &wr_before, N, mg, &c);
            assert!(
                (priced - expected).abs() < 1e-9,
                "multigraph={mg}: priced {priced}, from scratch {expected}"
            );
        }
    }

    #[test]
    fn dense_ds_matches_a_from_scratch_recompute_undirected() {
        let c = cache();
        const N: usize = 4;
        let mut before = vec![0i64; N * N];
        for a in 0..N {
            for b in a..N {
                let w = ((a * 3 + b * 5) % 4) as i64;
                before[a * N + b] = w;
                before[b * N + a] = w;
            }
        }
        let wr_before = vec![7i64, 6, 5, 4];

        let (r, nr) = (g(1), g(3));
        let entries = vec![
            Entry {
                r,
                s: g(0),
                delta: -1,
                me: None,
                mrs_before: before[N],
            },
            Entry {
                r: nr,
                s: g(0),
                delta: 1,
                me: None,
                mrs_before: before[3 * N],
            },
        ];
        let hdr = MoveHeader {
            r: Some(r),
            nr: Some(nr),
            r_img: EndImage::default(),
            nr_img: EndImage::default(),
            dkin: 0,
            dkout: 1,
            dr: 1,
            dnr: 1,
        };

        let mut after = before.clone();
        for e in &entries {
            let (a, b) = (e.r.index(), e.s.index());
            after[a * N + b] += e.delta;
            if a != b {
                after[b * N + a] += e.delta;
            }
        }
        let mut wr_after = wr_before.clone();
        wr_after[1] -= 1;
        wr_after[3] += 1;

        for &mg in &[false, true] {
            let st = UndStub(Stub {
                wr: wr_before.clone(),
                mrs: before.clone(),
                slots: N,
                directed: false,
            });
            let priced = dense_terms::<UndStub, Undirected, i64>(&st, &entries, &hdr, N, mg, &c);
            let expected = dense_absolute::<Undirected>(&after, &wr_after, N, mg, &c)
                - dense_absolute::<Undirected>(&before, &wr_before, N, mg, &c);
            assert!(
                (priced - expected).abs() < 1e-9,
                "multigraph={mg}: priced {priced}, from scratch {expected}"
            );
        }
    }

    /// A move whose vertex weight does not change touches no `wr`, so the
    /// O(B) sweep must contribute exactly nothing and only the entries count.
    #[test]
    fn dense_ds_with_no_vertex_weight_change_prices_only_the_entries() {
        let c = cache();
        const N: usize = 4;
        let before: Vec<i64> = (0..N * N).map(|i| ((i * 3) % 4) as i64).collect();
        let wr = vec![5i64, 5, 5, 5];
        let (r, nr) = (g(0), g(1));
        let entries = vec![Entry {
            r,
            s: nr,
            delta: 3,
            me: None,
            mrs_before: before[1],
        }];
        let hdr = MoveHeader {
            r: Some(r),
            nr: Some(nr),
            r_img: EndImage::default(),
            nr_img: EndImage::default(),
            dkin: 0,
            dkout: 0,
            dr: 0,
            dnr: 0,
        };
        let mut after = before.clone();
        after[1] += 3;

        for &mg in &[false, true] {
            let st = DirStub(Stub {
                wr: wr.clone(),
                mrs: before.clone(),
                slots: N,
                directed: true,
            });
            let priced = dense_terms::<DirStub, Directed, i64>(&st, &entries, &hdr, N, mg, &c);
            let expected = dense_absolute::<Directed>(&after, &wr, N, mg, &c)
                - dense_absolute::<Directed>(&before, &wr, N, mg, &c);
            assert!((priced - expected).abs() < 1e-12, "{priced} vs {expected}");
        }
    }

    /// `entries.hh:99-105`: the undirected field table keys `(s, t)` and
    /// `(t, s)` on the same slot, so a lookup must not depend on the
    /// orientation the recorder happened to pick.
    #[test]
    fn recorded_delta_is_symmetric_only_when_undirected() {
        let entries = [Entry {
            r: g(1),
            s: g(2),
            delta: 7i64,
            me: None,
            mrs_before: 0,
        }];
        assert_eq!(recorded_delta::<Directed, i64>(&entries, g(1), g(2)), 7);
        assert_eq!(recorded_delta::<Directed, i64>(&entries, g(2), g(1)), 0);
        assert_eq!(recorded_delta::<Undirected, i64>(&entries, g(1), g(2)), 7);
        assert_eq!(recorded_delta::<Undirected, i64>(&entries, g(2), g(1)), 7);
    }
}
