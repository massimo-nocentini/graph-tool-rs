//! What one recorded transition consists of.

use crate::ids::{BEdge, Group, Weight};

/// One changed block pair.
///
/// `me` and `mrs_before` are the **before-image**, interned at record time.
/// Carrying them is free: the sparse pricing loop already performs exactly
/// this `_emat.get_me(t, w)` lookup per entry (`blockmodel/state.hh:1225`),
/// and `update_rs` performs it again (`entries.hh:359`). Paying it once and
/// letting both consumers read it removes two cache-cold dependent loads per
/// entry from the hottest kernel in the library.
///
/// It is *not* free everywhere, and the cost is stated rather than hidden:
/// `virtual_move_groups` returns early when `s == t || n == 0`
/// (`state.hh:1340-1342`) and a clabel-rejected move returns `+inf` at
/// `:1346`, so those paths now pay one block-edge lookup per distinct entry
/// that the C++ skips.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Entry<W: Weight> {
    /// Source group of the changed pair.
    pub r: Group,
    /// Target group of the changed pair.
    pub s: Group,
    /// How much the `(r, s)` edge weight changes.
    pub delta: W,
    /// The block-graph edge, if it already existed.
    pub me: Option<BEdge>,
    /// Its weight before the move.
    pub mrs_before: W,
}

/// The per-endpoint scalars the pricing function needs.
///
/// Exactly the six values `entries_dS` reads at `state.hh:1239-1253`.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct EndImage<W: Weight> {
    /// Out-weight of the group.
    pub mrp: W,
    /// In-weight of the group.
    pub mrm: W,
    /// Vertex weight of the group.
    pub wr: W,
}

/// The non-entry half of the before-image: six scalars plus the move's own
/// degree deltas.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MoveHeader<W: Weight> {
    /// Source group, or `None` for a single-endpoint (insertion) move.
    pub r: Option<Group>,
    /// Target group, or `None` for a removal move.
    pub nr: Option<Group>,
    /// `r`'s scalars before the move.
    pub r_img: EndImage<W>,
    /// `nr`'s scalars before the move.
    pub nr_img: EndImage<W>,
    /// In-degree contribution of the moved vertex.
    pub dkin: W,
    /// Out-degree contribution of the moved vertex.
    pub dkout: W,
    /// Vertex weight leaving `r`.
    pub dr: W,
    /// Vertex weight entering `nr`.
    pub dnr: W,
}

impl<W: Weight> Default for MoveHeader<W> {
    fn default() -> Self {
        MoveHeader {
            r: None,
            nr: None,
            r_img: EndImage::default(),
            nr_img: EndImage::default(),
            dkin: W::ZERO,
            dkout: W::ZERO,
            dr: W::ZERO,
            dnr: W::ZERO,
        }
    }
}
