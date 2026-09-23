//! Vector-backed property maps, and the constant/unity maps.

use crate::bound::Bound;
use crate::error::PropError;
use crate::ids::{EdgeTag, GraphId, Id, IdTag, VertexTag};
use std::marker::PhantomData;

use super::map::{LvalueProp, Owned, ReadProp, WriteProp};

/// A property map backed by a dense vector, owning its storage.
///
/// `checked_vector_property_map` (`fast_vector_property_map.hh:47`) holds a
/// `shared_ptr<vector<T>>`, so `soft_reserve` (`:84`) -- which is `const` and
/// reachable from any aliasing copy -- can reallocate underneath a `T&`
/// returned by `operator[]` (`:132-137`), with no diagnostic. `DenseProp` owns its
/// `Vec` and hands out views through `&mut self`, so the same sequence is a
/// borrow-check error.
#[derive(Clone, Debug)]
pub struct DenseProp<T, K: IdTag> {
    graph: GraphId,
    data: Vec<T>,
    _k: PhantomData<fn() -> K>,
}

/// A vertex property map.
pub type VertexProp<T> = DenseProp<T, VertexTag>;
/// An edge property map.
pub type EdgeProp<T> = DenseProp<T, EdgeTag>;

impl<T, K: IdTag> DenseProp<T, K> {
    /// An empty map belonging to `graph`.
    pub fn new(graph: GraphId) -> Self {
        DenseProp {
            graph,
            data: Vec::new(),
            _k: PhantomData,
        }
    }

    /// Adopt existing storage.
    pub fn from_vec(graph: GraphId, data: Vec<T>) -> Self {
        DenseProp {
            graph,
            data,
            _k: PhantomData,
        }
    }

    /// The graph this map belongs to.
    #[inline]
    pub const fn graph(&self) -> GraphId {
        self.graph
    }

    /// Current length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Whether the map holds nothing.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The whole store, for bulk kernels.
    ///
    /// This is the bounds-check-free path, and it needs no `unsafe`: an
    /// iterator over a slice carries its own length, so LLVM removes the
    /// check that indexing cannot. Prefer it over per-key access in any scan.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    /// The whole store, mutably.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Grow to `bound` and hand out a writable view, in one operation.
    ///
    /// **This is the replacement for `get_unchecked(size = 0)`.** There is no
    /// overload without a bound, so an under-sized view is not a value that
    /// exists. The `graph` comparison catches the other half of the
    /// `copy_property` defect: a map sized for one graph offered to a kernel
    /// operating on another.
    pub fn sized_for(&mut self, bound: Bound<K>) -> Result<PropSliceMut<'_, T, K>, PropError>
    where
        T: super::value::Zeroed,
    {
        // `T::zero` is a `fn() -> T`, hence an `FnMut() -> T`: the two entry
        // points are one implementation, and the `Zeroed` bound buys only the
        // absence of a closure at the call site. `gt_core::design` section 13.1: the
        // `Default` supertrait this replaces is unimplementable for the 15th
        // member of the value universe.
        self.sized_for_with(bound, T::zero)
    }

    /// As [`sized_for`](Self::sized_for), but filling new slots from a
    /// closure.
    ///
    /// The Python member has no context-free default -- `Py<PyAny>` is not
    /// `Default` and `Py_None` needs the interpreter -- so the PyO3 boundary
    /// uses this form, capturing its `Python<'py>` token in `fill`.
    pub fn sized_for_with<F>(
        &mut self,
        bound: Bound<K>,
        fill: F,
    ) -> Result<PropSliceMut<'_, T, K>, PropError>
    where
        F: FnMut() -> T,
    {
        self.check_graph(bound)?;
        let n = bound.len();
        // `resize_with` grows to *exactly* `n` and is a no-op when the store is
        // already at least that long, which is both halves of
        // `reserve(size)`'s `if (size > _store->size()) resize(size)`
        // (`fast_vector_property_map.hh:77-82`). In particular it never
        // shrinks: two kernels over the same map with different bounds -- the
        // vertex map of a graph that has since lost vertices, say -- must not
        // hand the second one a store that drops the first one's slots.
        if self.data.len() < n {
            self.data.resize_with(n, fill);
        }
        Ok(PropSliceMut::new(&mut self.data[..n]))
    }

    /// The single `GraphId` comparison defect #8 costs.
    ///
    /// graph-tool has no analogue, because a vertex descriptor is a bare
    /// `size_t`: `graph_copy.cc:66-73` reserves `num_vertices(src)` -- the
    /// *filtered* count -- and then writes at unfiltered indices, and the
    /// Python guard at `__init__.py:3200` compares filtered counts too. Here
    /// the two graphs are distinguishable even when their sizes agree, which
    /// is exactly the case that makes the C++ failure silent.
    #[inline]
    fn check_graph(&self, bound: Bound<K>) -> Result<(), PropError> {
        if self.graph != bound.graph() {
            return Err(PropError::WrongGraph {
                owner: self.graph.get(),
                expected: bound.graph().get(),
            });
        }
        Ok(())
    }

    /// A read-only view, if the map is already large enough.
    ///
    /// Fallible by construction: `&self` cannot grow. graph-tool's checked map
    /// auto-grows on read (`fast_vector_property_map.hh:132-137`), so
    /// `g.vp.x[v]` on a never-written map returns the default; the dispatcher
    /// reproduces that by calling [`sized_for`](Self::sized_for) on read-only
    /// maps too, which is the one place the two-phase shape survives.
    pub fn view(&self, bound: Bound<K>) -> Result<PropSlice<'_, T, K>, PropError> {
        self.check_graph(bound)?;
        let n = bound.len();
        if self.data.len() < n {
            // Not `&self.data[..]`. Handing back the short run is precisely
            // `get_unchecked(size = 0)` (`dispatch.hh:171-177`), whose
            // `reserve(0)` is a no-op and whose result is a map the kernel
            // then indexes past the end of (`:219-222`).
            return Err(PropError::Undersized {
                have: self.data.len(),
                need: n,
            });
        }
        Ok(PropSlice::new(&self.data[..n]))
    }

    /// Geometric growth for push-style filling. Ports `soft_reserve` (`:84`),
    /// but nothing's correctness depends on it having been called.
    ///
    /// **Deliberate divergence.** C++ `soft_reserve` calls `resize`, so it
    /// grows the *size*, and that is what makes `operator[]`
    /// (`fast_vector_property_map.hh:132-137`) able to silently extend a map
    /// on read -- the auto-growth the port pushes into
    /// [`sized_for`](Self::sized_for). Here it grows *capacity* only: the
    /// length of a `DenseProp` changes at exactly one place, and a hint cannot
    /// manufacture a `T` in any case (14 of the 15 members have
    /// [`Zeroed`](super::value::Zeroed); the Python one has no context-free
    /// value at all).
    pub fn soft_reserve(&mut self, n: usize) {
        let have = self.data.len();
        if n > have {
            // `std::max(size, 2 * _store->size())` (`:84-89`), with the
            // multiplication saturated: `usize * 2` is a panic under
            // `[profile.dev] overflow-checks`, and a hint has no business
            // aborting.
            let want = n.max(have.saturating_mul(2));
            self.data.reserve(want - have);
        }
    }
}

impl<T: Clone, K: IdTag> ReadProp<K> for DenseProp<T, K> {
    type Value = T;
    type Ref<'s>
        = &'s T
    where
        Self: 's;
    #[inline]
    fn get_ref(&self, k: Id<K>) -> &T {
        &self.data[k.index()]
    }
}

impl<T: Clone, K: IdTag> WriteProp<K> for DenseProp<T, K> {
    #[inline]
    fn put(&mut self, k: Id<K>, v: T) {
        self.data[k.index()] = v;
    }
}

impl<T: Clone, K: IdTag> LvalueProp<K> for DenseProp<T, K> {
    #[inline]
    fn at_mut(&mut self, k: Id<K>) -> &mut T {
        &mut self.data[k.index()]
    }
}

/// A read-only view of a correctly sized map.
///
/// `unchecked_vector_property_map` (`fast_vector_property_map.hh:178`) is
/// obtained from its checked counterpart by
/// `reinterpret_cast<unchecked_t&>(*this)` out of a **private** base. Here the
/// relation is an ordinary reborrow, so the store is frozen for the view's
/// lifetime.
#[derive(Debug)]
pub struct PropSlice<'a, T, K: IdTag> {
    data: &'a [T],
    _k: PhantomData<fn() -> K>,
}

/// A writable view of a correctly sized map.
#[derive(Debug)]
pub struct PropSliceMut<'a, T, K: IdTag> {
    data: &'a mut [T],
    _k: PhantomData<fn() -> K>,
}

impl<'a, T, K: IdTag> PropSlice<'a, T, K> {
    pub(crate) fn new(data: &'a [T]) -> Self {
        PropSlice {
            data,
            _k: PhantomData,
        }
    }
    /// The whole run, for bulk kernels.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }
    /// Length, which equals the bound this view was made for.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Whether the run is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, T, K: IdTag> PropSliceMut<'a, T, K> {
    pub(crate) fn new(data: &'a mut [T]) -> Self {
        PropSliceMut {
            data,
            _k: PhantomData,
        }
    }
    /// The whole run, for bulk kernels.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }
    /// The whole run, mutably: the bounds-check-free scatter path.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }
    /// Length, which equals the bound this view was made for.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Whether the run is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// Reborrow, so a view can be passed by value into a kernel and reused.
    ///
    /// Kernels should take `PropSliceMut` **by value**, not `&mut PropSliceMut`:
    /// by value, the inner `&mut [T]` reaches LLVM as a direct `noalias`
    /// argument; behind a reference it becomes a pointer load and the aliasing
    /// information is lost.
    #[inline]
    pub fn reborrow(&mut self) -> PropSliceMut<'_, T, K> {
        PropSliceMut::new(self.data)
    }
    /// Downgrade to a read-only view.
    #[inline]
    pub fn as_shared(&self) -> PropSlice<'_, T, K> {
        PropSlice::new(self.data)
    }
}

impl<T: Clone, K: IdTag> ReadProp<K> for PropSlice<'_, T, K> {
    type Value = T;
    type Ref<'s>
        = &'s T
    where
        Self: 's;
    #[inline]
    fn get_ref(&self, k: Id<K>) -> &T {
        &self.data[k.index()]
    }
}

impl<T: Clone, K: IdTag> ReadProp<K> for PropSliceMut<'_, T, K> {
    type Value = T;
    type Ref<'s>
        = &'s T
    where
        Self: 's;
    #[inline]
    fn get_ref(&self, k: Id<K>) -> &T {
        &self.data[k.index()]
    }
}

impl<T: Clone, K: IdTag> WriteProp<K> for PropSliceMut<'_, T, K> {
    #[inline]
    fn put(&mut self, k: Id<K>, v: T) {
        self.data[k.index()] = v;
    }
}

impl<T: Clone, K: IdTag> LvalueProp<K> for PropSliceMut<'_, T, K> {
    #[inline]
    fn at_mut(&mut self, k: Id<K>) -> &mut T {
        &mut self.data[k.index()]
    }
}

// ---------------------------------------------------------------------------
// Constant maps
// ---------------------------------------------------------------------------

/// The multiplicative identity, as a `const` so that it folds.
pub trait One: Copy {
    /// One.
    const ONE: Self;
}
impl One for f64 {
    const ONE: f64 = 1.0;
}
impl One for i64 {
    const ONE: i64 = 1;
}
impl One for i32 {
    const ONE: i32 = 1;
}

/// A map that reads `1` everywhere. A true ZST.
///
/// `UnityPropertyMap` (`graph_properties.hh:699-711`) is an empty class
/// deriving `boost::put_get_helper`, so `sizeof == 1`; 74 call sites pass it.
/// `PhantomData<fn() -> (T, K)>` keeps this `Copy + Send + Sync + Default`
/// regardless of `T` and makes it genuinely zero-sized.
pub struct Unity<T, K: IdTag>(PhantomData<fn() -> (T, K)>);

const _: () = assert!(size_of::<Unity<f64, VertexTag>>() == 0);

impl<T, K: IdTag> Unity<T, K> {
    /// The map.
    pub const NEW: Self = Unity(PhantomData);
}

// Hand-written, for the reason `Id`'s are (`ids.rs`): `#[derive(Clone)]` on a
// type carrying `PhantomData<fn() -> (T, K)>` still emits `T: Clone, K: Clone`
// bounds, and `#[derive(Default)]` emits `K: Default` -- which no `IdTag` is,
// so the derived `Default` is uninhabited for every real instantiation. The
// doc above says "regardless of `T`"; these make that true.
impl<T, K: IdTag> Clone for Unity<T, K> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, K: IdTag> Copy for Unity<T, K> {}
impl<T, K: IdTag> Default for Unity<T, K> {
    #[inline]
    fn default() -> Self {
        Unity::NEW
    }
}
impl<T, K: IdTag> std::fmt::Debug for Unity<T, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unity<{}>", K::NAME)
    }
}

impl<T: One + Clone, K: IdTag> ReadProp<K> for Unity<T, K> {
    type Value = T;
    type Ref<'s>
        = Owned<T>
    where
        Self: 's;
    const IS_UNITY: bool = true;
    const IS_CONSTANT: bool = true;
    #[inline(always)]
    fn get_ref(&self, k: Id<K>) -> Owned<T> {
        Owned(T::ONE)
    }
}

/// A map that reads the same value everywhere.
///
/// The field is `c` and it is **public** -- which is exactly what
/// `graph_selectors.hh:109` and `:178` try to read on a class whose member is
/// the *private* `_c` (`graph_properties.hh:677`). Both of those C++ overloads
/// are therefore uninstantiable dead code, so the "constant weight" fast path
/// silently does not exist.
#[derive(Clone, Copy, Debug)]
pub struct Constant<T, K: IdTag> {
    /// The value.
    pub c: T,
    _k: PhantomData<fn() -> K>,
}

impl<T, K: IdTag> Constant<T, K> {
    /// A map reading `c` everywhere.
    pub const fn new(c: T) -> Self {
        Constant { c, _k: PhantomData }
    }
}

// `T: Default`, and nothing about `K`: see the note on `Unity`.
impl<T: Default, K: IdTag> Default for Constant<T, K> {
    #[inline]
    fn default() -> Self {
        Constant::new(T::default())
    }
}

impl<T: Clone, K: IdTag> ReadProp<K> for Constant<T, K> {
    type Value = T;
    type Ref<'s>
        = &'s T
    where
        Self: 's;
    const IS_CONSTANT: bool = true;
    #[inline(always)]
    fn get_ref(&self, k: Id<K>) -> &T {
        &self.c
    }
}

/// A constant map whose value lives in the type. Zero-sized, and routes into
/// the unity fast path automatically when `C == 1`. No C++ equivalent.
pub struct ConstI64<const C: i64, K: IdTag>(PhantomData<fn() -> K>);

impl<const C: i64, K: IdTag> ConstI64<C, K> {
    /// The map.
    pub const NEW: Self = ConstI64(PhantomData);
}

impl<const C: i64, K: IdTag> Clone for ConstI64<C, K> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<const C: i64, K: IdTag> Copy for ConstI64<C, K> {}
impl<const C: i64, K: IdTag> Default for ConstI64<C, K> {
    #[inline]
    fn default() -> Self {
        ConstI64::NEW
    }
}
impl<const C: i64, K: IdTag> std::fmt::Debug for ConstI64<C, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConstI64<{C}, {}>", K::NAME)
    }
}

impl<const C: i64, K: IdTag> ReadProp<K> for ConstI64<C, K> {
    type Value = i64;
    type Ref<'s>
        = Owned<i64>
    where
        Self: 's;
    const IS_CONSTANT: bool = true;
    const IS_UNITY: bool = C == 1;
    #[inline(always)]
    fn get_ref(&self, k: Id<K>) -> Owned<i64> {
        Owned(C)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    //! `Bound::new` is `pub(crate)` (defect #7: that is the mechanism, and
    //! `tests/ui/u01_bound_new_is_private.rs` pins the `E0624` a downstream
    //! crate gets for trying), so a bound of non-zero length cannot be minted
    //! from `tests/`. Until `AdjList::add_vertex` (U5) lands there is no
    //! external way to obtain one either, so the sizing guarantees are checked
    //! here, inside the crate, and `tests/u09_props.rs` re-checks them through
    //! the public surface as soon as U5 makes that possible.

    use super::*;
    use crate::ids::GraphId;
    use proptest::prelude::*;

    /// Two graphs, deliberately of the same size, and a map belonging to the
    /// first. Sameness of size is the point: it is what makes the C++
    /// confusion silent.
    fn two_graphs() -> (GraphId, GraphId) {
        (GraphId::fresh(), GraphId::fresh())
    }

    // -- sized_for ----------------------------------------------------------

    #[test]
    fn sized_for_grows_to_exactly_the_bound_and_is_idempotent() {
        let g = GraphId::fresh();
        let mut p: VertexProp<i64> = DenseProp::new(g);
        assert_eq!(p.len(), 0);

        {
            let v = p.sized_for(Bound::new(g, 5)).expect("same graph");
            assert_eq!(v.len(), 5);
            assert_eq!(v.as_slice(), &[0i64; 5]);
        }
        // *Exactly* `bound.len()`: `reserve(size)` is `resize(size)`, not a
        // geometric step (`fast_vector_property_map.hh:77-82`). A map that
        // over-grew here would report a length no graph asked for, and
        // `view()` would then accept a bound the graph cannot produce.
        assert_eq!(p.len(), 5);

        // Idempotent: the second call resizes nothing and preserves content.
        p.as_mut_slice()[2] = 42;
        let v = p.sized_for(Bound::new(g, 5)).expect("same graph");
        assert_eq!(v.len(), 5);
        assert_eq!(v.as_slice()[2], 42);
        assert_eq!(p.len(), 5);
    }

    #[test]
    fn sized_for_never_shrinks_and_still_hands_back_exactly_the_bound() {
        let g = GraphId::fresh();
        let mut p: VertexProp<i64> = DenseProp::from_vec(g, vec![1, 2, 3, 4, 5, 6, 7]);

        {
            let v = p.sized_for(Bound::new(g, 3)).expect("same graph");
            // The view is the bound, not the store.
            assert_eq!(v.len(), 3);
            assert_eq!(v.as_slice(), &[1, 2, 3]);
        }
        // The store is untouched: `reserve` only grows (`:77-82`), and a
        // shrink here would silently discard slots another live bound still
        // covers.
        assert_eq!(p.len(), 7);
        assert_eq!(p.as_slice(), &[1, 2, 3, 4, 5, 6, 7]);

        // A zero bound is the degenerate case of the same rule.
        assert_eq!(p.sized_for(Bound::new(g, 0)).expect("same graph").len(), 0);
        assert_eq!(p.len(), 7);
    }

    /// Defect #8. The two graphs have the *same* number of slots, so a design
    /// that compared lengths would accept this.
    #[test]
    fn sized_for_refuses_another_graphs_bound() {
        let (a, b) = two_graphs();
        let mut p: VertexProp<i64> = DenseProp::from_vec(a, vec![0; 4]);

        let err = p.sized_for(Bound::new(b, 4)).unwrap_err();
        assert_eq!(
            err,
            PropError::WrongGraph {
                owner: a.get(),
                expected: b.get(),
            }
        );
        // And it is refused *before* any resize: the store is unchanged.
        assert_eq!(p.len(), 4);
        assert!(p.sized_for(Bound::new(b, 9)).is_err());
        assert_eq!(p.len(), 4);
    }

    /// The edge space of the *same* graph is a different `K`, so
    /// `p.sized_for(g.edge_bound())` on a vertex map is a type error rather
    /// than a length comparison. Asserted by construction: this compiles only
    /// because the tags agree.
    #[test]
    fn the_two_index_spaces_do_not_mix() {
        let g = GraphId::fresh();
        let mut e: EdgeProp<i64> = DenseProp::new(g);
        assert_eq!(e.sized_for(Bound::new(g, 6)).expect("same graph").len(), 6);
        // `e.sized_for(Bound::<VertexTag>::new(g, 6))` does not compile.
    }

    // -- sized_for_with -----------------------------------------------------

    #[test]
    fn sized_for_with_fills_new_slots_and_leaves_existing_ones() {
        let g = GraphId::fresh();
        let mut p: VertexProp<i64> = DenseProp::from_vec(g, vec![10, 11]);

        let mut next = 100;
        {
            let v = p
                .sized_for_with(Bound::new(g, 5), || {
                    next += 1;
                    next
                })
                .expect("same graph");
            assert_eq!(v.as_slice(), &[10, 11, 101, 102, 103]);
        }

        // A second call that needs no growth must not call the closure at all:
        // `resize_with` on a no-op range does not, and a `fill` capturing a
        // `Python<'py>` token has no business running once per already-filled
        // slot.
        let mut calls = 0;
        let v = p
            .sized_for_with(Bound::new(g, 5), || {
                calls += 1;
                0
            })
            .expect("same graph");
        assert_eq!(v.len(), 5);
        assert_eq!(calls, 0);
    }

    /// The reason `sized_for_with` exists at all ([DESIGN](crate::design) section 13.1): the
    /// 15th member has no context-free default, so growth must be able to take
    /// a closure over borrowed context. `String` stands in for the borrow.
    #[test]
    fn sized_for_with_accepts_a_closure_over_borrowed_context() {
        let g = GraphId::fresh();
        let token = String::from("py");
        let mut p: VertexProp<String> = DenseProp::new(g);
        let v = p
            .sized_for_with(Bound::new(g, 3), || token.clone())
            .expect("same graph");
        assert_eq!(v.as_slice(), &["py", "py", "py"]);
    }

    #[test]
    fn sized_for_is_sized_for_with_over_zero() {
        let g = GraphId::fresh();
        let mut a: VertexProp<i64> = DenseProp::new(g);
        let mut b: VertexProp<i64> = DenseProp::new(g);
        let left = a.sized_for(Bound::new(g, 4)).expect("same graph");
        let right = b
            .sized_for_with(Bound::new(g, 4), i64::default)
            .expect("same graph");
        assert_eq!(left.as_slice(), right.as_slice());
    }

    // -- view ---------------------------------------------------------------

    /// The whole point of the type: an under-sized read is an `Err`, never a
    /// short slice. `get_unchecked(size = 0)` (`dispatch.hh:170-177`) returns
    /// the short map instead, and `unchecked_vector_property_map::operator[]`
    /// (`:219-222`) then indexes past its end.
    #[test]
    fn view_on_an_undersized_map_is_an_error_not_a_short_slice() {
        let g = GraphId::fresh();
        let p: VertexProp<i64> = DenseProp::from_vec(g, vec![7, 8, 9]);

        let err = p.view(Bound::new(g, 5)).unwrap_err();
        assert_eq!(err, PropError::Undersized { have: 3, need: 5 });

        // Exactly at the bound, and beyond it, are both fine.
        assert_eq!(p.view(Bound::new(g, 3)).expect("exact").len(), 3);
        assert_eq!(p.view(Bound::new(g, 2)).expect("longer").len(), 2);
        assert_eq!(
            p.view(Bound::new(g, 2)).expect("longer").as_slice(),
            &[7, 8]
        );
    }

    #[test]
    fn view_refuses_another_graphs_bound_before_it_compares_lengths() {
        let (a, b) = two_graphs();
        let p: VertexProp<i64> = DenseProp::from_vec(a, vec![0; 2]);
        // Wrong graph *and* too short: the identity is the more specific
        // failure, and reporting `Undersized` here would send a caller off to
        // grow a map that will never fit.
        assert_eq!(
            p.view(Bound::new(b, 9)).unwrap_err(),
            PropError::WrongGraph {
                owner: a.get(),
                expected: b.get(),
            }
        );
    }

    #[test]
    fn an_empty_map_satisfies_an_empty_bound() {
        let g = GraphId::fresh();
        let p: VertexProp<i64> = DenseProp::new(g);
        let v = p.view(Bound::new(g, 0)).expect("0 <= 0");
        assert!(v.is_empty());
    }

    /// What `sized_for` buys the caller of `view`: the two-phase shape
    /// [DESIGN](crate::design) section 5 moves out of Python.
    #[test]
    fn sizing_first_makes_the_read_only_view_succeed() {
        let g = GraphId::fresh();
        let bound = Bound::new(g, 6);
        let mut p: VertexProp<i64> = DenseProp::new(g);
        assert!(p.view(bound).is_err());
        assert_eq!(p.sized_for(bound).expect("same graph").len(), 6);
        assert_eq!(p.view(bound).expect("now sized").len(), 6);
    }

    // -- the views themselves ------------------------------------------------

    #[test]
    fn writes_through_the_view_reach_the_store() {
        let g = GraphId::fresh();
        let mut p: VertexProp<i64> = DenseProp::new(g);
        {
            let mut v = p.sized_for(Bound::new(g, 4)).expect("same graph");
            v.put(Id::from_index(1), 11);
            *v.at_mut(Id::from_index(3)) += 30;
            // `as_shared` and `reborrow` are views of the same run.
            assert_eq!(v.as_shared().as_slice(), &[0, 11, 0, 30]);
            let mut r = v.reborrow();
            r.as_mut_slice()[0] = 5;
        }
        assert_eq!(p.as_slice(), &[5, 11, 0, 30]);
        assert_eq!(*p.get_ref(Id::from_index(1)), 11);
    }

    // -- soft_reserve --------------------------------------------------------

    #[test]
    fn soft_reserve_grows_capacity_geometrically_and_never_the_length() {
        let g = GraphId::fresh();
        let mut p: VertexProp<i64> = DenseProp::from_vec(g, vec![0; 8]);

        // `std::max(size, 2 * _store->size())` (`:84-89`).
        p.soft_reserve(9);
        assert!(p.as_slice().len() == 8, "soft_reserve must not resize");
        assert!(
            p.data.capacity() >= 16,
            "capacity {} is below the geometric step",
            p.data.capacity()
        );

        // A larger request wins over the doubling.
        p.soft_reserve(100);
        assert!(p.data.capacity() >= 100);
        assert_eq!(p.len(), 8);

        // A request at or below the length is a no-op, exactly as
        // `if (size > _store->size())` makes it.
        let before = p.data.capacity();
        p.soft_reserve(8);
        p.soft_reserve(0);
        assert_eq!(p.data.capacity(), before);
        assert_eq!(p.len(), 8);
    }

    /// `2 * _store->size()` is unguarded in C++. Here the same expression runs
    /// under `[profile.dev] overflow-checks = true`, so it is saturated: a
    /// hint has no business aborting the process.
    #[test]
    fn soft_reserve_does_not_overflow_on_an_absurd_request() {
        let g = GraphId::fresh();
        let mut p: VertexProp<i64> = DenseProp::new(g);
        // An empty store: `max(n, 0)`, and `reserve` is what would fail, not
        // the arithmetic. Only the arithmetic is under test, so ask for a
        // length that cannot allocate but can be doubled.
        p.soft_reserve(0);
        assert_eq!(p.len(), 0);

        let mut q: VertexProp<u8> = DenseProp::from_vec(g, vec![0; 4]);
        q.soft_reserve(4);
        assert_eq!(q.len(), 4);
    }

    // -- the constant maps ---------------------------------------------------

    /// Defect #43's other half: `UnityPropertyMap` is an empty *class*, so
    /// `sizeof == 1` and 74 call sites pass a byte around. This is 0.
    #[test]
    fn unity_is_a_true_zero_sized_type() {
        assert_eq!(size_of::<Unity<f64, VertexTag>>(), 0);
        assert_eq!(size_of::<Unity<i64, EdgeTag>>(), 0);
        assert_eq!(size_of::<ConstI64<3, VertexTag>>(), 0);

        let u: Unity<f64, VertexTag> = Unity::NEW;
        assert_eq!(*u.get_ref(Id::from_index(9)), 1.0);
        // `assert_eq!` on a pair rather than two `assert!`s: both flags are
        // associated *consts*, so `assert!(FLAG)` is a constant assertion and
        // clippy is right to say so. The value under test is the pair.
        assert_eq!(
            (
                <Unity<f64, VertexTag> as ReadProp<VertexTag>>::IS_UNITY,
                <Unity<f64, VertexTag> as ReadProp<VertexTag>>::IS_CONSTANT,
            ),
            (true, true)
        );
    }

    /// Defect #44: `graph_selectors.hh:109, :178` read `weight.c` on a class
    /// whose member is the private `_c`, so both overloads are dead code.
    #[test]
    fn constants_field_is_public_and_the_unity_fast_path_is_value_directed() {
        let c: Constant<f64, VertexTag> = Constant::new(2.5);
        assert_eq!(c.c, 2.5);
        assert_eq!(*c.get_ref(Id::from_index(0)), 2.5);
        assert_eq!(
            (
                <Constant<f64, VertexTag> as ReadProp<VertexTag>>::IS_CONSTANT,
                <Constant<f64, VertexTag> as ReadProp<VertexTag>>::IS_UNITY,
            ),
            (true, false)
        );

        // `ConstI64<1>` routes into the unity fast path automatically, which
        // is the thing `is_unity_map<ConvertedPropertyMap<...>>` (defect #45)
        // loses in C++.
        assert_eq!(
            (
                <ConstI64<1, VertexTag> as ReadProp<VertexTag>>::IS_UNITY,
                <ConstI64<2, VertexTag> as ReadProp<VertexTag>>::IS_UNITY,
            ),
            (true, false)
        );
        assert_eq!(
            *ConstI64::<7, VertexTag>::default().get_ref(Id::from_index(3)),
            7
        );
    }

    // -- the model ----------------------------------------------------------

    /// One operation on a property map, as a kernel or the dispatcher would
    /// perform it.
    #[derive(Clone, Debug)]
    enum Op {
        /// `sized_for(bound)`.
        SizedFor(usize),
        /// `sized_for_with(bound, || v)`.
        SizedWith(usize, i64),
        /// `view(bound)`.
        View(usize),
        /// `soft_reserve(n)` -- a hint, so the model does not move.
        SoftReserve(usize),
        /// A checked write at a live slot.
        Put(usize, i64),
        /// The same call, with a bound minted by a *different* graph.
        Foreign(usize),
    }

    fn any_op() -> impl Strategy<Value = Op> {
        prop_oneof![
            (0usize..24).prop_map(Op::SizedFor),
            (0usize..24, -9i64..9).prop_map(|(n, v)| Op::SizedWith(n, v)),
            (0usize..24).prop_map(Op::View),
            (0usize..48).prop_map(Op::SoftReserve),
            (0usize..24, -9i64..9).prop_map(|(i, v)| Op::Put(i, v)),
            (0usize..24).prop_map(Op::Foreign),
        ]
    }

    proptest! {
        /// [DESIGN](crate::design) section 16's model-based check, for the sizing chokepoint:
        /// a `Vec<i64>` is the reference, and the store must equal it after
        /// every operation in the sequence.
        ///
        /// The three claims that only a *sequence* can falsify are the ones
        /// this is here for: growth is monotone (no operation shortens the
        /// store), a view is exactly its bound however long the store has
        /// become, and a rejected call -- wrong graph, or too short -- leaves
        /// the store exactly as it was. The C++ counterpart fails the last of
        /// those by construction: `operator[]` calls `soft_reserve(i + 1)`
        /// (`fast_vector_property_map.hh:132-137`), so a *read* at a stale
        /// index silently doubles the store.
        #[test]
        fn a_sequence_of_operations_agrees_with_a_naive_model(
            ops in prop::collection::vec(any_op(), 1..40)
        ) {
            let g = GraphId::fresh();
            let other = GraphId::fresh();
            let mut p: VertexProp<i64> = DenseProp::new(g);
            let mut model: Vec<i64> = Vec::new();

            for op in ops {
                let before = model.len();
                match op {
                    Op::SizedFor(n) => {
                        let v = p.sized_for(Bound::new(g, n)).expect("same graph");
                        prop_assert_eq!(v.len(), n);
                        if model.len() < n {
                            model.resize(n, 0);
                        }
                    }
                    Op::SizedWith(n, fill) => {
                        let v = p
                            .sized_for_with(Bound::new(g, n), || fill)
                            .expect("same graph");
                        prop_assert_eq!(v.len(), n);
                        if model.len() < n {
                            model.resize(n, fill);
                        }
                    }
                    Op::View(n) => match p.view(Bound::new(g, n)) {
                        Ok(v) => {
                            prop_assert!(model.len() >= n);
                            prop_assert_eq!(v.len(), n);
                            prop_assert_eq!(v.as_slice(), &model[..n]);
                        }
                        Err(e) => {
                            prop_assert!(model.len() < n);
                            prop_assert_eq!(
                                e,
                                PropError::Undersized {
                                    have: model.len(),
                                    need: n,
                                }
                            );
                        }
                    },
                    Op::SoftReserve(n) => {
                        p.soft_reserve(n);
                    }
                    Op::Put(i, v) => {
                        if i < model.len() {
                            p.put(Id::from_index(i), v);
                            model[i] = v;
                        }
                    }
                    Op::Foreign(n) => {
                        prop_assert_eq!(
                            p.sized_for(Bound::new(other, n)).unwrap_err(),
                            PropError::WrongGraph {
                                owner: g.get(),
                                expected: other.get(),
                            }
                        );
                        prop_assert!(p.view(Bound::new(other, n)).is_err());
                    }
                }
                // The store *is* the model, after every single step.
                prop_assert_eq!(p.as_slice(), model.as_slice());
                // Monotone: nothing in this API shortens a map.
                prop_assert!(p.len() >= before);
            }
        }
    }

    /// A slice-backed map is not constant and not unity, so the kernel branch
    /// that folds away for `Unity` is live for it.
    #[test]
    fn a_dense_map_claims_neither_fast_path() {
        assert_eq!(
            [
                <VertexProp<f64> as ReadProp<VertexTag>>::IS_UNITY,
                <VertexProp<f64> as ReadProp<VertexTag>>::IS_CONSTANT,
                <PropSlice<'_, f64, VertexTag> as ReadProp<VertexTag>>::IS_UNITY,
                <PropSliceMut<'_, f64, VertexTag> as ReadProp<VertexTag>>::IS_CONSTANT,
            ],
            [false; 4]
        );
    }
}
