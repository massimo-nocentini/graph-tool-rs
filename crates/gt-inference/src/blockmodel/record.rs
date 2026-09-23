//! Recording a transition, and projecting it up a hierarchy.
//!
//! ## The three things this module owes the C++, and the one it refuses it
//!
//! `modify_entries_dispatch` (`blockmodel/entries.hh:227-320`) is thirty lines
//! of template with four independent axes -- `single`, directedness, `r`
//! present, `nr` present -- and every one of them is resolved at
//! monomorphisation here exactly as it is there: `SINGLE` is a const generic,
//! `D::DIRECTED` is an associated const, and the two endpoint tests are
//! `Option::is_some` on a four-byte niche.
//!
//! Three details the port must not lose, and does not:
//!
//! * the undirected self-loop is **traversed twice** -- `out_edges_range` on
//!   an undirected view walks the whole adjacency block, out-half and in-half
//!   (`graph_adaptor.hh:199-207`, `graph_adjacency.hh:1102-1108`) -- so its
//!   weight is accumulated and halved (`entries.hh:274-290`);
//! * a pair containing *both* endpoints belongs to `r`'s half-fields, because
//!   `get_field` (`:108-119`) tests `r` first. Routing it anywhere else keys
//!   one pair under two cells, and the `else` arms at `:258-261`/`:311-314`
//!   are that routing;
//! * `auto s = b[u]` (`:250`, `:304`) is read unchecked although
//!   `state.hh:126-131` deliberately sets `_b[v]` to the null group for
//!   zero-weight vertices. Here that value is an `Option<Group>` and a `None`
//!   is **reported**, never skipped: skipping produces a delta quietly short
//!   by one edge's weight, which is strictly worse than the C++'s
//!   out-of-range index because nothing downstream can detect it. See
//!   [`no_group`].
//!
//! The one it refuses is the `single` self-loop correction. `entries.hh:283`
//! guards the `nr` half of the halving with `if constexpr (!single)` while the
//! `r` half runs unguarded, so a `single` (`r == nr`) move over an undirected
//! self-loop records `+w` on the pair `(r, r)` for a transition that changes
//! nothing. It is latent in graph-tool -- `move_vertex` returns at
//! `state.hh:280` when `r == nr` and `virtual_move_groups` at `:1341`, so the
//! entries are computed and thrown away -- but it is still a wrong delta, and
//! [`scan`] drops the guard. With `SINGLE` the two halves then land on the
//! same cell (`nr`'s half-fields *are* `r`'s, `:231`) and cancel, which is the
//! correct answer.

use gt_core::dir::{Dir, Directed, Field, HasDir, Undirected};
use gt_core::graph::{Bidirectional, GraphRef};
use gt_core::ids::VertexId;

use super::state::BlockView;
use crate::delta::{DeltaBuf, MoveHeader, MoveKey, OutOfPlane, Recording, Workspace};
use crate::ids::{Group, Weight};

// ---------------------------------------------------------------------------
// The two failures that must be loud
// ---------------------------------------------------------------------------

/// A vertex the recorder needs a group for has none.
///
/// `#[cold]` and out of line: this is the `auto s = b[u]` of `entries.hh:250`
/// and `:304`, where the C++ indexes `_mrs`/`_emat` with
/// `numeric_limits<int64_t>::max()` and reads whatever is there. A skip would
/// be quieter and worse -- the recorded delta would be short by this edge's
/// weight and every consumer would price, apply and audit it without noticing
/// -- so the only two honest answers are a wrong index and a stopped process.
#[cold]
#[inline(never)]
fn no_group(u: VertexId, map: &'static str) -> ! {
    panic!(
        "vertex {} has no group under `{map}`: a zero-weight vertex \
         (`state.hh:126-131` sets `_b[v]` to the null group there) is incident \
         to the vertex being moved, so the recorded delta would be short by \
         this edge's weight",
        u.index()
    )
}

/// A group the hierarchy projection has no image for.
///
/// `propagate_entries` reads `_b[t]` unchecked (`state.hh:1113`). Same
/// argument as [`no_group`]: an unprojectable group cannot be dropped from the
/// delta, because the level above would then be short by that pair's weight.
#[cold]
#[inline(never)]
fn no_image(r: Group) -> ! {
    panic!(
        "group {r:?} has no image one level up: `propagate_entries` \
         (`state.hh:1110-1115`) projects every entry endpoint through `_b`, \
         and dropping one would leave the level above short by that pair's \
         weight"
    )
}

/// `b[u]`, the group of a neighbour **before** the move.
#[inline]
fn group_of<S: BlockView>(st: &S, u: VertexId) -> Group {
    match st.group_of(u) {
        Some(g) => g,
        None => no_group(u, "b"),
    }
}

/// `nb[u]`, the group of a neighbour **after** the move.
#[inline]
fn new_group<F: Fn(VertexId) -> Option<Group>>(nb: &F, u: VertexId) -> Group {
    match nb(u) {
        Some(g) => g,
        None => no_group(u, "nb"),
    }
}

// ---------------------------------------------------------------------------
// The directedness-dependent half of the recorder
// ---------------------------------------------------------------------------

/// The part of [`scan`] that only exists when the graph is directed.
///
/// ## Why this trait exists rather than an `if D::DIRECTED` branch
///
/// `entries.hh:292` is `if constexpr (is_directed_v<g_t>)` around a loop over
/// `in_edges_range(v, g)`, and `:308` names `_r_in_field`. Neither has an
/// undirected spelling *at all* in this port, and deliberately so:
///
/// * [`Bidirectional`] is implemented for no undirected view, because
///   `in_edges(v, undirected_adaptor)` returns a default-constructed empty
///   range (`graph_adaptor.hh:219-227`) and a generic algorithm that walked it
///   would get a wrong answer rather than an error;
/// * `Field::<Undirected>` has exactly two inhabitants, so `R_IN` is
///   unnameable there -- which is the whole point of
///   [`Field`](gt_core::dir::Field) being total.
///
/// A plain `if D::DIRECTED { g.in_edges(v) }` therefore does not typecheck for
/// generic `D`, however dead the branch is. Dispatching on the *directedness
/// type* does, costs nothing (both methods monomorphise to either the loop or
/// an empty body), and moves the `G: Bidirectional` requirement to exactly the
/// impl that needs it: a directed block model can only be scanned through a
/// view that has in-edges, and an undirected one only through a view whose
/// `out_edges` is the whole incidence run.
pub trait ScanDir<G: GraphRef>: Dir + Sized {
    /// The half-field a pair `(x, r)` -- `r` as the *sink* -- is keyed by.
    ///
    /// `get_field_rnr<true, false>` (`entries.hh:90-103`): `_r_in_field` when
    /// directed, and `_r_out_field` when not, there being no in-half to speak
    /// of. [`DeltaBuf::touch`] folds the `s == t` case onto the out-half by
    /// itself, exactly as `:97` does.
    fn r_sink() -> Field<Self>;

    /// `in_degree(v, g, eweight)` with a unit `eweight`.
    ///
    /// Zero when undirected: `in_degree(u, undirected_adaptor)` is literally
    /// `return 0` (`graph_adaptor.hh:322-330`), which is why
    /// `state.hh:225-226` guards `_mrm[r] += kin` with `if constexpr
    /// (is_directed_v<g_t>)`.
    fn in_degree<W: Weight>(g: G, v: VertexId) -> W;

    /// The in-edge pass of `modify_entries_dispatch` (`entries.hh:292-318`).
    ///
    /// A no-op when undirected, where the C++ does not instantiate it either.
    fn in_pass<const SINGLE: bool, S, F>(
        st: &S,
        g: G,
        v: VertexId,
        mv: MoveKey,
        nb: &F,
        buf: &mut DeltaBuf<Self, S::W>,
    ) where
        S: BlockView<D = Self>,
        F: Fn(VertexId) -> Option<Group>;
}

impl<G: Bidirectional> ScanDir<G> for Directed {
    #[inline]
    fn r_sink() -> Field<Directed> {
        Field::<Directed>::R_IN
    }

    #[inline]
    fn in_degree<W: Weight>(g: G, v: VertexId) -> W {
        W::from_i64(g.in_degree(v) as i64)
    }

    fn in_pass<const SINGLE: bool, S, F>(
        st: &S,
        g: G,
        v: VertexId,
        mv: MoveKey,
        nb: &F,
        buf: &mut DeltaBuf<Directed, S::W>,
    ) where
        S: BlockView<D = Directed>,
        F: Fn(VertexId) -> Option<Group>,
    {
        let one = S::W::from_i64(1);
        // `insert_delta_rnr<single, false, true>`: with `single` the `nr`
        // half-fields *are* `r`'s (`:231` nulls the second endpoint).
        let nr_in = if SINGLE {
            Field::<Directed>::R_IN
        } else {
            Field::<Directed>::NR_IN
        };
        let mut resolve = |a: Group, b: Group| st.resolve(a, b);

        for inc in g.in_edges(v) {
            let u = inc.other;
            // `:299-300`. The self-loop was already counted by the out-pass.
            if u == v {
                continue;
            }

            if mv.from.is_some() {
                // `insert_delta_rnr<true, false, false>(s, r, ew)` (`:306`).
                buf.touch::<false>(Field::<Directed>::R_IN, group_of(st, u), one, &mut resolve);
            }

            if let Some(nr) = mv.to {
                let s = new_group(nb, u);
                if Some(s) != mv.from {
                    // `insert_delta_rnr<single, false, true>(s, nr, ew)`
                    // (`:313`).
                    buf.touch::<true>(nr_in, s, one, &mut resolve);
                } else {
                    // `insert_delta_rnr<true, true, true>(s, nr, ew)` (`:315`)
                    // with `s == r`: the pair `(r, nr)`, which `get_field`
                    // routes to `r`'s out-half because it tests `r` first.
                    buf.touch::<true>(Field::<Directed>::R_OUT, nr, one, &mut resolve);
                }
            }
        }
    }
}

impl<G: GraphRef + HasDir<Dir = Undirected>> ScanDir<G> for Undirected {
    #[inline]
    fn r_sink() -> Field<Undirected> {
        Field::<Undirected>::R_OUT
    }

    #[inline]
    fn in_degree<W: Weight>(_g: G, _v: VertexId) -> W {
        W::ZERO
    }

    /// `if constexpr (is_directed_v<g_t>)` (`entries.hh:292`), not taken.
    #[inline]
    fn in_pass<const SINGLE: bool, S, F>(
        _st: &S,
        _g: G,
        _v: VertexId,
        _mv: MoveKey,
        _nb: &F,
        _buf: &mut DeltaBuf<Undirected, S::W>,
    ) where
        S: BlockView<D = Undirected>,
        F: Fn(VertexId) -> Option<Group>,
    {
    }
}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------

/// Record the transition "move `v` from `mv.from` to `mv.to`".
///
/// `st` is borrowed **shared** and does not appear in the return type: the
/// recording's lifetime is the *workspace's*. That is the whole ownership
/// inversion, and it is why the caller can subsequently take `&mut st`.
///
/// Ports `modify_entries` (`entries.hh:322`) + `get_move_entries`
/// (`state.hh:1062`). `nb` is the `_vset`-backed ad-hoc property map of
/// `state.hh:451-459`, here an ordinary closure rather than a
/// `make_adhoc_prop`.
///
/// `slots` is `num_vertices(_bg)`, the block-graph index bound that
/// `modify_entries` takes as `B` (`entries.hh:228`); `vw` is `_vweight[v]`,
/// the `n` of `virtual_move_groups` (`state.hh:1331`, `:1394`). Both are
/// parameters for the same reason: [`BlockView`] exposes the *occupied* group
/// count and no per-vertex weight, so neither can be recovered from `st`. See
/// the crate docs' note on `dr`'s sign -- graph-tool passes `-n` and `+n` into
/// `entries_dS` (`:1349`), while [`MoveHeader::dr`] is the weight *leaving*
/// `r` and is therefore positive.
///
/// Only level 0 is recorded. `get_move_entries`' `visit_coupled_if`
/// (`state.hh:1068-1075`) is [`propagate`], which the caller runs once per
/// level because the projection `_b` differs per level and is not reachable
/// from a single `st`.
///
/// # Panics
///
/// If a neighbour of `v` has no group. See this module's `no_group`.
#[allow(clippy::too_many_arguments)]
pub fn record<'w, S, G, F>(
    st: &S,
    g: G,
    v: VertexId,
    mv: MoveKey,
    nb: &F,
    slots: usize,
    vw: S::W,
    ws: &'w mut Workspace<S::D, S::W>,
) -> Recording<'w, S::D, S::W>
where
    S: BlockView,
    S::D: ScanDir<G>,
    G: GraphRef,
    F: Fn(VertexId) -> Option<Group>,
{
    // `_degs[v] = {kin, kout}` (`state.hh:220-222`) with a unit `_eweight`:
    // this recorder carries no edge-weight map, so an edge's weight is one and
    // multiplicity is parallel edges. `out_degree` on an undirected view is
    // `degree` on the underlying graph (`graph_adaptor.hh:315-319`), i.e. a
    // self-loop counts twice -- which is what `_mrp[r] += kout` (`:224`)
    // expects.
    let hdr = MoveHeader {
        r: mv.from,
        nr: mv.to,
        r_img: st.end_image(mv.from),
        nr_img: st.end_image(mv.to),
        dkin: <S::D as ScanDir<G>>::in_degree(g, v),
        dkout: S::W::from_i64(g.out_degree(v) as i64),
        dr: vw,
        dnr: vw,
    };
    let stamp = st.stamp();
    let mut rec = Recording::new(&mut ws.stack, hdr, stamp);

    // `modify_entries` (`entries.hh:322-331`): the dispatch *is* the
    // comparison, and `single` is the only thing it decides.
    if mv.from == mv.to {
        scan::<true, S, G, F, S::D>(st, g, v, mv, nb, slots, rec.level_mut(0));
    } else {
        scan::<false, S, G, F, S::D>(st, g, v, mv, nb, slots, rec.level_mut(0));
    }
    rec
}

// ---------------------------------------------------------------------------
// scan
// ---------------------------------------------------------------------------

/// The single-pass recorder. `modify_entries_dispatch<single>`
/// (`entries.hh:227-320`).
///
/// `SINGLE` is a const generic exactly as C++'s `single` is a template
/// parameter, and it selects the half-field at every touch site, so the
/// monomorphised body has the same shape. `D::DIRECTED` gates the in-edge pass
/// with no runtime branch.
///
/// Two details the C++ encodes and a port must not lose: the undirected
/// self-loop weight is accumulated and **halved** (`:276-292`, with
/// `assert(self_weight % 2 == 0)`), and `auto s = b[u]` (`:250`, `:304`) is
/// unchecked although `state.hh:126-128` sets `_b[v]` to the null group for
/// zero-weight vertices. Here that is an `Option<Group>`; a `None` neighbour
/// must be `debug_assert`ed or reported, never silently skipped, because a
/// skip produces a delta quietly short by one edge's weight.
///
/// `slots` is `B`, the block-graph index bound `set_move` grows the field
/// table to (`entries.hh:63-64`). Calling this is what *begins* the buffer;
/// anything it held is cleared first, in O(#entries).
///
/// # Panics
///
/// If a neighbour of `v` has no group (this module's `no_group`), or if a
/// group index is at or beyond `slots`.
pub fn scan<const SINGLE: bool, S, G, F, D: ScanDir<G>>(
    st: &S,
    g: G,
    v: VertexId,
    mv: MoveKey,
    nb: &F,
    slots: usize,
    buf: &mut DeltaBuf<D, S::W>,
) where
    S: BlockView<D = D>,
    G: GraphRef,
    F: Fn(VertexId) -> Option<Group>,
{
    debug_assert_eq!(
        SINGLE,
        mv.from == mv.to,
        "`modify_entries` (entries.hh:324) dispatches `single` on exactly \
         `r == nr`"
    );

    // `set_move(r, single ? null_group : nr, B)` (`:231`). `mv` keeps both
    // endpoints -- the C++'s local `nr` stays non-null under `single` and is
    // still tested at `:255` -- while the *buffer's* key folds `nr` onto `r`.
    buf.begin(
        MoveKey {
            from: mv.from,
            to: if SINGLE { None } else { mv.to },
        },
        slots,
    );

    let one = S::W::from_i64(1);
    // `insert_delta_rnr<single, true, _>`: the `nr` out-half, which is `r`'s
    // when `single`.
    let nr_out = if SINGLE {
        Field::<D>::R_OUT
    } else {
        Field::<D>::NR_OUT
    };
    let r_sink = <D as ScanDir<G>>::r_sink();
    let mut resolve = |a: Group, b: Group| st.resolve(a, b);

    // Undirected only: how many times a self-loop was *traversed*. With a unit
    // edge weight this is `self_weight` (`:239`) exactly, and counting rather
    // than summing is what lets the halving at `:277` be an integer division
    // by two on a count -- [`Weight`] has no `Div`, and `to_f64`-and-back
    // would be lossy for the very type (`f64`) it would be introduced for.
    let mut self_traversals = 0usize;

    for inc in g.out_edges(v) {
        let u = inc.other;

        if mv.from.is_some() {
            // `insert_delta_rnr<true, true, false>(r, s, ew)` (`:252`).
            // `b[u]` is read only where the C++ *uses* it: with `r` null the
            // value at `:250` is dead, and a dead read must not panic.
            buf.touch::<false>(Field::<D>::R_OUT, group_of(st, u), one, &mut resolve);
        }

        if let Some(nr) = mv.to {
            // `:257-259`: `v` itself is already in `nr`, whatever `nb` says.
            let s = if u == v { nr } else { new_group(nb, u) };
            if Some(s) != mv.from {
                // `insert_delta_rnr<single, true, true>(nr, s, ew)` (`:262`).
                buf.touch::<true>(nr_out, s, one, &mut resolve);
            } else {
                // `insert_delta_rnr<true, false, true>(nr, s, ew)` (`:264`)
                // with `s == r`: the pair `(nr, r)`, keyed by `r`'s sink half
                // because `get_field` tests `r` first.
                buf.touch::<true>(r_sink, nr, one, &mut resolve);
            }
        }

        if !D::DIRECTED && u == v {
            // `:269-272`.
            self_traversals += 1;
        }
    }

    <D as ScanDir<G>>::in_pass::<SINGLE, S, F>(st, g, v, mv, nb, buf);

    if !D::DIRECTED && self_traversals != 0 {
        // `assert(int64_t(self_weight) % 2 == 0)` (`:279`): an undirected view
        // yields a self-loop from both halves of the adjacency block, so the
        // traversal count is even by construction.
        debug_assert_eq!(
            self_traversals % 2,
            0,
            "an undirected self-loop is traversed twice (entries.hh:279)"
        );
        let half = S::W::from_i64((self_traversals / 2) as i64);
        if let Some(r) = mv.from {
            // `insert_delta_rnr<true, true, true>(r, r, self_weight)` (`:282`).
            buf.touch::<true>(Field::<D>::R_OUT, r, half, &mut resolve);
        }
        if let Some(nr) = mv.to {
            // `insert_delta_rnr<single, true, false>(nr, nr, self_weight)`
            // (`:288`), **without** the `if constexpr (!single)` of `:283`.
            // See this module's header: with `single` the guard leaves `+w` on
            // `(r, r)` for a transition that changes nothing, and dropping it
            // makes the two halves cancel on the one cell they share.
            buf.touch::<false>(nr_out, nr, half, &mut resolve);
        }
    }
}

// ---------------------------------------------------------------------------
// propagate
// ---------------------------------------------------------------------------

/// Project level `l`'s entries onto level `l + 1`.
///
/// Ports `propagate_entries` (`state.hh:1099-1117`). Streams straight from the
/// below-level entries into the above-level buffer: materialising an
/// intermediate vector is not required by the borrow checker
/// ([`DeltaStack::below_above`](crate::delta::DeltaStack::below_above) already
/// proves the disjointness) and `entries_op` allocates nothing.
///
/// `b_of` is level `l + 1`'s `_b`, indexed by a level-`l` group -- the block
/// graph's vertices *are* the level below's groups. `slots` is that level's
/// `num_vertices(_bg)` (`:1107`).
///
/// The above-level key is `{_b[r], _b[nr]}` (`:1103-1106`), with one
/// normalisation the C++ does not perform: two groups that share a parent
/// project to one, and a [`MoveKey`] whose endpoints are equal and present is
/// not a value [`DeltaBuf`] expects. Collapsing it to a single endpoint is
/// what `set_move(r, single ? null : nr, ..)` does one level down and selects
/// the identical cells -- `get_field` tests `_rnr.first` first, so `nr`'s
/// half-fields are unreachable either way.
///
/// Allocates nothing once the buffers are warm: `begin` drains into the
/// existing field table and the entry vector is reused.
///
/// # Panics
///
/// If `b_of` has no image for an entry endpoint (this module's `no_image`),
/// or if `l + 1 >= rec.n_levels()`.
pub fn propagate<W: Weight, D: Dir, P>(
    rec: &mut Recording<'_, D, W>,
    l: usize,
    b_of: &P,
    slots: usize,
    resolve: &mut impl crate::delta::Resolve<W>,
) -> Result<(), OutOfPlane>
where
    P: Fn(Group) -> Option<Group>,
{
    let (below, above) = rec.below_above(l);

    // `group_t r = (u != _null_group) ? _b[u] : _null_group` (`:1103-1105`):
    // an absent endpoint stays absent, and an unplaced group is null too --
    // the C++ reaches the same value by reading `_b[u]` unchecked.
    let mv = below.move_key();
    let from = mv.from.and_then(b_of);
    let to = mv.to.and_then(b_of);
    let to = if to == from { None } else { to };
    above.begin(MoveKey { from, to }, slots);

    for e in below.entries() {
        // `if (delta == 0) return;` (`:1114`). A zero entry projects to a zero
        // entry; creating it would only cost the above level a cell.
        if e.delta == W::ZERO {
            continue;
        }
        let r = match b_of(e.r) {
            Some(g) => g,
            None => no_image(e.r),
        };
        let s = match b_of(e.s) {
            Some(g) => g,
            None => no_image(e.s),
        };
        // `insert_delta<true>(_b[t], _b[w], delta)` (`:1115-1116`), made
        // total: the C++ falls through `get_field` to the shared `_dummy`
        // cell here, where two out-of-plane pairs accumulate into one entry.
        above.touch_dyn(r, s, e.delta, resolve)?;
    }
    Ok(())
}
