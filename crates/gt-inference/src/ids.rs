//! Group and block-edge identifiers, and the transition stamp.

use std::fmt;
use std::num::{NonZeroU32, NonZeroU64};
use std::ops::{Add, Neg, Sub};
use std::sync::atomic::{AtomicU64, Ordering};

/// A group (block) identifier.
///
/// `NonZeroU32` so that `Option<Group>` is four bytes. graph-tool uses
/// `group_t = int64_t` with `null_group = INT64_MAX`
/// (`inference/blockmodel/spec.hh:87-88` -- the blockmodel `spec.hh`, not
/// `inference/base/`, which holds `graph_spec_base.hh`/`spec_base.hh` and
/// defines no `group_t`; `blockmodel/partition.hh:170` repeats the sentinel
/// as `_null_group`), checked by
/// hand about thirty times in `blockmodel/state.hh`; and `entries.hh:250`
/// reads `auto s = b[u]` *unchecked*, although `state.hh:126-131`
/// deliberately sets `_b[v] = _null_group` for zero-weight vertices. Missing
/// the check here is a compile error.
///
/// The sentinel's *arithmetic* is reproduced exactly, not merely its effect:
/// C++ burns the top representable `group_t` on the null, so the last usable
/// index is `INT64_MAX - 1`. Here [`Group::new`] maps `u32::MAX` to `None`
/// for the same reason, leaving `0..=u32::MAX - 1` addressable.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Group(NonZeroU32);

const _: () = assert!(size_of::<Option<Group>>() == 4);

impl Group {
    /// From a zero-based index.
    #[inline]
    pub const fn new(i: u32) -> Option<Group> {
        match NonZeroU32::new(i.wrapping_add(1)) {
            Some(n) => Some(Group(n)),
            None => None,
        }
    }
    /// The zero-based index.
    #[inline]
    pub const fn index(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

impl fmt::Debug for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "g{}", self.index())
    }
}

/// An edge of the *block* graph.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BEdge(pub u32);

/// Process-unique identity of one inference state.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StateId(NonZeroU64);

impl StateId {
    /// Mint a fresh identity.
    ///
    /// Process-wide, monotone, and never zero -- so the `NonZeroU64` niche is
    /// real and `Option<StateId>` stays eight bytes. `Relaxed` is the whole
    /// ordering contract: `fetch_add` is atomic under every ordering, the
    /// counter publishes no other memory, and the only property any caller
    /// depends on is that two calls cannot observe the same value.
    pub fn fresh() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let v = NEXT.fetch_add(1, Ordering::Relaxed);
        match NonZeroU64::new(v) {
            Some(n) => StateId(n),
            // Reached only after 2^64 - 1 states in one process, at which
            // point the counter has wrapped onto live identities and defect
            // #36 -- a delta committed into the wrong state -- is open again.
            // Aborting loudly is the only honest answer; silently wrapping is
            // not.
            None => panic!("StateId counter exhausted: 2^64 states in one process"),
        }
    }
    /// The raw value, for diagnostics.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// A state's revision counter, bumped on every commit.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Epoch(pub u64);

/// Identity **and** revision.
///
/// A revision counter alone does not identify a state: two freshly built
/// states both sit at epoch 0, so a transition recorded against one commits
/// silently into the other and the guard never fires. That was a real hole in
/// the source design and it costs one extra `u64` comparison to close.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Stamp {
    /// Which state.
    pub state: StateId,
    /// Which revision of it.
    pub epoch: Epoch,
}

/// Edge and vertex weights carried by the block model.
///
/// graph-tool carries counts as `int64_t` precisely so that deltas can be
/// negative; the `Neg` bound preserves that and excludes unsigned counts.
pub trait Weight:
    Copy
    + Default
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + fmt::Debug
    + Send
    + Sync
    + 'static
{
    /// The additive identity.
    const ZERO: Self;
    /// Widen for entropy arithmetic.
    fn to_f64(self) -> f64;
    /// Narrow from a count.
    fn from_i64(i: i64) -> Self;
}

impl Weight for i32 {
    const ZERO: i32 = 0;
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_i64(i: i64) -> i32 {
        i as i32
    }
}
impl Weight for i64 {
    const ZERO: i64 = 0;
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_i64(i: i64) -> i64 {
        i
    }
}
impl Weight for f64 {
    const ZERO: f64 = 0.0;
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn from_i64(i: i64) -> f64 {
        i as f64
    }
}
