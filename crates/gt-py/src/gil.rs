//! The GIL discipline, as a type-level lattice.

use gt_core::bound::Bound;
use gt_core::error::PropError;
use gt_core::ids::IdTag;
use gt_core::prop::{DenseProp, PropValue, PyValue, ValueKind};
use pyo3::Python;
use rayon::prelude::*;

mod sealed {
    pub trait Sealed {}
}

/// Whether an operation may leave the calling thread.
pub trait Mode: sealed::Sealed {
    /// Whether work in this mode may leave the calling thread.
    ///
    /// This is the *witness*: the resolution of the lattice is a `const` that
    /// a `const` block can assert, so "`f64 -> f64` is parallel" is proved at
    /// compile time rather than inferred by reading the source. See the
    /// `const` block below [`copy_prop`].
    const PARALLEL: bool;
    /// The mode's own name, for diagnostics.
    const NAME: &'static str;
}

/// Parallel: no CPython reference is touched, the GIL is released.
pub struct Par;
/// Serial: a CPython reference is touched, the GIL is held throughout.
pub struct Seq;

impl sealed::Sealed for Par {}
impl sealed::Sealed for Seq {}
impl Mode for Par {
    const PARALLEL: bool = true;
    const NAME: &'static str = "Par";
}
impl Mode for Seq {
    const PARALLEL: bool = false;
    const NAME: &'static str = "Seq";
}

/// The greatest lower bound of two modes. `Seq` absorbs.
///
/// There is no `||` here for a human to invert.
pub trait Meet<Rhs: Mode>: Mode {
    /// The meet.
    type Out: Mode;
}
impl Meet<Par> for Par {
    type Out = Par;
}
impl Meet<Seq> for Par {
    type Out = Seq;
}
impl Meet<Par> for Seq {
    type Out = Seq;
}
impl Meet<Seq> for Seq {
    type Out = Seq;
}

/// A mode that may be given to `V`.
///
/// This is the bound on [`ModeOf::Mode`], and it is what makes the policy
/// table below *unable* to be mis-edited: `Par` permits only a value that can
/// cross a thread, so writing `impl ModeOf for PyValue { type Mode = Par; }`
/// is `error[E0277]: the trait bound `PyValue: Sync` is not satisfied` rather
/// than a silently faster and unsound build. `tests/ui/u30_par_mode_for_py_value.rs`
/// pins that diagnostic.
///
/// The C++ has no analogue because its predicate is a `bool` computed inside
/// the kernel body (`graph_properties_copy.cc:35-42`): there is nothing for a
/// compiler to check, and in fact the expression is wrong in both directions.
pub trait Allows<V: ?Sized>: Mode {}
impl<V: ?Sized + Sync + Send> Allows<V> for Par {}
impl<V: ?Sized> Allows<V> for Seq {}

/// Which mode a member of the value universe forces.
///
/// Note this is *not* an open policy table: [`PropValue`] is sealed in
/// gt-core, so the only types that can appear here are the fifteen. A newtype
/// over a raw `pyo3::Py<PyAny>` cannot be given `Mode = Par`, because it
/// cannot be a property value at all.
pub trait ModeOf: PropValue {
    /// The mode this member forces.
    ///
    /// `Allows<Self>` is not decoration: it is the second of DESIGN.md §7's
    /// three mechanisms, moved from the strategy impls up onto the table
    /// itself, so that the table cannot be edited into unsoundness even by
    /// someone who never reads the strategy bounds.
    type Mode: Mode + Allows<Self>;
}

/// The fourteen members whose value can be moved with no token in hand.
///
/// A **local** marker, deliberately, rather than a use of gt-core's
/// [`GilFree`](gt_core::prop::GilFree). The two say the same thing about the
/// same fourteen types, but only a trait defined *here* lets the two
/// [`CopyValue`] impls below be proved disjoint: for an upstream trait rustc
/// must assume "upstream crates may add a new impl of `GilFree` for `PyValue`
/// in future versions" and rejects the pair as overlapping (`E0119`), whereas
/// nothing outside gt-py can ever implement this one.
///
/// It is minted by the same macro as [`ModeOf`], from the same token list, so
/// the two tables cannot drift apart -- which is the failure mode
/// `value_types` / `type_names[]` (`graph_properties.hh:61-76`) has by
/// construction.
pub trait TokenFree: PropValue {}

macro_rules! par_mode {
    ($($t:ty),+ $(,)?) => {$(
        impl ModeOf for $t { type Mode = Par; }
        impl TokenFree for $t {}
    )+};
}
par_mode!(
    u8,
    i16,
    i32,
    i64,
    f64,
    gt_core::prop::LongDouble,
    String,
    Vec<u8>,
    Vec<i16>,
    Vec<i32>,
    Vec<i64>,
    Vec<f64>,
    Vec<gt_core::prop::LongDouble>,
    Vec<String>,
);

impl ModeOf for PyValue {
    type Mode = Seq;
}

/// The mode a `S -> T` copy resolves to.
///
/// Nothing in the port computes this; the compiler does. It is written here
/// only so that a `const` block, a test or a reader can *name* the answer.
pub type ModeFor<S, T> = <<S as ModeOf>::Mode as Meet<<T as ModeOf>::Mode>>::Out;

/// Copying one value of the universe into another, **with the token**.
///
/// ## Why [`ConvertFrom`](gt_core::prop::ConvertFrom) does not serve the
/// serial half
///
/// `convert<To, From>` (`value_convert.hh:73-105`) has three arms that touch
/// CPython: `is_same_v<To, From>` on `python::object`, `is_same_v<To, object>`
/// and `is_same_v<From, object>`. All three need a live interpreter, and
/// `ConvertFrom::convert_from(&S) -> Result<T, _>` has no token in its
/// signature to give them -- it cannot, because gt-core builds without pyo3 at
/// all. gt-core's diagonal therefore reaches the interpreter by *re-acquiring*
/// the token (`Python::with_gil` inside `PyCell`'s `Clone`), once per value.
///
/// That is sound, and DESIGN.md §7 says so explicitly. It is also exactly the
/// work `Seq` exists to avoid: `Seq::copy` was handed a `Python<'_>` and holds
/// it for the whole loop, so re-deriving one per element is a TLS probe and a
/// guard construction per property-map entry, paid for nothing. The
/// token-carrying conversion therefore lives here, next to the mode that
/// guarantees the token exists, and its two impl families are precisely the
/// lattice's two halves:
///
/// * [`TokenFree`] -> [`TokenFree`] delegates to gt-core's `ConvertFrom` and
///   ignores the token, because no CPython reference is touched;
/// * `PyValue -> PyValue` is `Py::clone_ref`, which *takes* the token.
///
/// Note what this trait cannot be used for: [`Par::copy`](Par) does **not**
/// bound on it, and could not. Its body is the inside of
/// [`detach`](crate::gil::detach), where `Python<'_>` -- being `!Send` -- cannot be
/// captured. The parallel path is therefore bounded on the tokenless
/// `ConvertFrom` by construction, not by convention.
pub trait CopyValue<S: PropValue>: PropValue + Sized {
    /// Produce the target value for one key.
    fn copy_value(py: Python<'_>, s: &S) -> Result<Self, PropError>;
}

/// Every conversion whose **target** is GIL-free, delegating to gt-core's
/// tokenless `ConvertFrom`. Disjoint from the two impls below because
/// [`TokenFree`] is local to this crate and [`PyValue`] is not one of its
/// members; see its docs for why that distinction is the whole reason it
/// exists.
///
/// `S` is deliberately **not** bounded on [`TokenFree`], only on `PropValue`.
/// That is what admits `S = PyValue`, i.e. `convert`'s
/// `is_same_v<From, boost::python::object>` arm (`value_convert.hh:86-107`):
/// `extract<To>` with an elementwise fallback for vector targets. gt-core
/// carries that arm (`prop/convert.rs`, `feature = "python"`), so the column
/// is reachable here the moment the target is one of the fourteen. The token
/// is ignored on this path because gt-core re-derives its own; see the note on
/// the `PyValue` target impl below for what that costs and why it is still the
/// right trade for now.
impl<S, T> CopyValue<S> for T
where
    S: PropValue,
    T: TokenFree + gt_core::prop::ConvertFrom<S>,
{
    #[inline]
    fn copy_value(_py: Python<'_>, s: &S) -> Result<T, PropError> {
        T::convert_from(s)
    }
}

/// `convert`'s `is_same_v<To, boost::python::object>` arm
/// (`value_convert.hh:82-85`): `boost::python::object(v)` never fails, and
/// neither does its counterpart here.
///
/// **Cost, recorded rather than hidden.** gt-core's `ConvertFrom` has no token
/// in its signature -- it cannot, because gt-core builds without pyo3 -- so it
/// re-acquires one with `Python::with_gil` per value, which on this path is a
/// TLS probe and a guard construction per property-map entry that `Seq` was
/// already handed a token to avoid. The re-entrant acquisition is sound
/// (DESIGN.md section 7), and correctness does not depend on removing it. The
/// saving needs a token parameter on gt-core's `FromAny`, which is gt-core's
/// call to make; when it exists, this body becomes the direct
/// `ToPyObject::to_object(py, s)` and this note goes away.
impl<S> CopyValue<S> for PyValue
where
    S: TokenFree,
    PyValue: gt_core::prop::ConvertFrom<S>,
{
    #[inline]
    fn copy_value(_py: Python<'_>, s: &S) -> Result<PyValue, PropError> {
        <PyValue as gt_core::prop::ConvertFrom<S>>::convert_from(s)
    }
}

/// `convert`'s `is_same_v<To, From>` arm (`value_convert.hh:78-81`) for the
/// Python member: the C++ returns `(const To&)v`, i.e. a new reference to the
/// *same* object, and so does this. The incref is sound because the token is
/// a parameter.
impl CopyValue<PyValue> for PyValue {
    #[inline]
    fn copy_value(py: Python<'_>, s: &PyValue) -> Result<PyValue, PropError> {
        Ok(s.clone_ref(py))
    }
}

/// How to copy one property map into another.
///
/// The `Python<'_>` token is a **parameter of the trait method**, not
/// something the call site decides to pass. So `Par` cannot be implemented
/// without `allow_threads` and `Seq` cannot run without the token held.
///
/// This closes the half of the C++ truth table that a serial-versus-parallel
/// switch alone leaves open: for `int -> int`, `graph_properties_copy.cc`
/// *holds* the GIL across a long serial loop, blocking every other Python
/// thread for the duration.
pub trait CopyStrategy<S, T, K: IdTag> {
    /// Copy `src` into `dst`.
    fn copy(
        py: Python<'_>,
        bound: Bound<K>,
        src: &DenseProp<S, K>,
        dst: &mut DenseProp<T, K>,
    ) -> Result<(), PropError>;
}

/// graph-tool's parallel cut-off: `__openmp_min_thresh = 300`
/// (`openmp.cc:20`), read by `graph_properties_copy.cc:42` as
/// `num_vertices(g) > get_openmp_min_thresh()`.
///
/// The *sense* of the port's use differs from the C++'s in one respect worth
/// stating: here the threshold governs only whether rayon is entered. The GIL
/// is released either way, because releasing it is a property of the value
/// types and not of the workload size. In the C++ the two decisions are welded
/// to one another through `is_python`, which is how defect #31 -- the GIL held
/// across a long serial `int -> int` loop -- comes about.
const MIN_PAR_ITEMS: usize = 300;

/// Items per parallel chunk.
///
/// A **constant**, deliberately: DESIGN.md §8 forbids any partition that is a
/// function of the pool size, so that a run with one worker and a run with
/// sixteen perform the same work in the same grouping. Nothing here folds
/// floating point, so the grouping cannot move a result; what it fixes is
/// *which* error a failing copy reports (see [`Par::copy`](Par)).
const PAR_GRAIN: usize = 256;

/// The two runs a copy operates on, each checked against the bound.
///
/// Both maps must already be sized: DESIGN.md §5 and U32 put the sizing at the
/// Python entry point, *before* `detach`, which is the only place it can
/// happen for a map of [`PyValue`] (the member has no context-free default, so
/// `DenseProp::sized_for` is not available to it). Handing back the short run
/// instead -- `get_unchecked(size = 0)` (`dispatch.hh:171-177`) -- is the
/// defect `Bound` exists to prevent.
fn runs<'s, 'd, S, T, K: IdTag>(
    bound: Bound<K>,
    src: &'s DenseProp<S, K>,
    dst: &'d mut DenseProp<T, K>,
) -> Result<(&'s [S], &'d mut [T]), PropError> {
    let n = bound.len();
    check(dst.graph(), dst.len(), bound)?;
    check(src.graph(), src.len(), bound)?;
    Ok((&src.as_slice()[..n], &mut dst.as_mut_slice()[..n]))
}

/// One map's half of [`runs`]' check: identity first, then length.
///
/// Identity first because it is the one the C++ never makes: the Python guard
/// at `__init__.py:3200` compares *filtered counts*, which agree in exactly
/// the case that makes `graph_copy.cc:72`'s out-of-bounds write silent.
#[inline]
fn check<K: IdTag>(
    owner: gt_core::ids::GraphId,
    have: usize,
    bound: Bound<K>,
) -> Result<(), PropError> {
    if owner != bound.graph() {
        return Err(PropError::WrongGraph {
            owner: owner.get(),
            expected: bound.graph().get(),
        });
    }
    if have < bound.len() {
        return Err(PropError::Undersized {
            have,
            need: bound.len(),
        });
    }
    Ok(())
}

/// Convert one contiguous run, keeping the **lowest-indexed** failure.
///
/// The whole run is executed even after a failure. `parallel_loop_no_spawn<true>`
/// (`parallel_util.hh:399-437`) instead sets a thread-private `skip` flag, so
/// which iterations are abandoned is the OpenMP schedule and the set of work
/// actually performed is not reproducible. Here it is a function of the inputs
/// alone -- and, because `Seq` runs the full cover too, it does not depend on
/// which arm of the lattice was selected either.
#[inline]
fn convert_run<S, T>(base: usize, s: &[S], d: &mut [T]) -> Option<(usize, PropError)>
where
    T: gt_core::prop::ConvertFrom<S>,
{
    let mut first: Option<(usize, PropError)> = None;
    for (i, (slot, v)) in d.iter_mut().zip(s).enumerate() {
        match T::convert_from(v) {
            Ok(c) => *slot = c,
            Err(e) => {
                if first.is_none() {
                    first = Some((base + i, e));
                }
            }
        }
    }
    first
}

/// Lowest index wins. Commutative and associative, so rayon's tree shape --
/// which *is* a function of the pool size -- cannot be observed in the result.
#[inline]
fn earlier(
    a: Option<(usize, PropError)>,
    b: Option<(usize, PropError)>,
) -> Option<(usize, PropError)> {
    match (a, b) {
        (None, other) | (other, None) => other,
        (Some(x), Some(y)) => Some(if x.0 <= y.0 { x } else { y }),
    }
}

impl<S, T, K: IdTag> CopyStrategy<S, T, K> for Seq
where
    S: PropValue,
    T: CopyValue<S>,
{
    /// Holds the token for the whole loop.
    ///
    /// `py` is threaded through every single conversion, so there is no
    /// formulation of this body that quietly drops the GIL -- and equally, no
    /// formulation that reaches rayon, since the token cannot cross a thread.
    ///
    /// **Deviation from the skeleton, recorded here because it is load
    /// bearing.** The skeleton bounded this impl on
    /// `S: PropValue + Clone, T: PropValue + ConvertFrom<S>`.
    ///
    /// `S: Clone` is not satisfiable by the one member this impl exists to
    /// serve. [`PyValue`] is `!Clone` *by construction* — `Py_INCREF` needs a
    /// token, so duplication is `clone_ref(py)` and nothing else — which means
    /// `copy_prop::<PyValue, PyValue, _>` could not even be **named** under
    /// the skeleton's bounds, and the entire `Seq` half of the lattice would
    /// have been dead code. It is dropped, and nothing here needs it: a
    /// conversion reads `&S`.
    ///
    /// `ConvertFrom<PyValue>` *is* satisfiable, but only by re-acquiring a
    /// token per value; [`CopyValue`] is the same conversion using the one
    /// already in hand. See its docs.
    fn copy(
        py: Python<'_>,
        bound: Bound<K>,
        src: &DenseProp<S, K>,
        dst: &mut DenseProp<T, K>,
    ) -> Result<(), PropError> {
        let (s, d) = runs(bound, src, dst)?;
        let mut first: Option<(usize, PropError)> = None;
        for (i, (slot, v)) in d.iter_mut().zip(s).enumerate() {
            match T::copy_value(py, v) {
                Ok(c) => *slot = c,
                Err(e) => {
                    if first.is_none() {
                        first = Some((i, e));
                    }
                }
            }
        }
        match first {
            None => Ok(()),
            Some((_, e)) => Err(e),
        }
    }
}

impl<S, T, K: IdTag> CopyStrategy<S, T, K> for Par
where
    S: PropValue + Clone + Sync,
    T: PropValue + gt_core::prop::ConvertFrom<S> + Send,
    K: Send + Sync,
{
    /// Body is `py.allow_threads(|| ... par_iter_mut() ...)`.
    ///
    /// The token is *consumed* by [`detach`] and cannot enter the closure, so
    /// the conversion this body can reach is the tokenless `ConvertFrom` and
    /// nothing else. That is why the `Par` half needs no discipline from its
    /// author: the alternative does not typecheck.
    ///
    /// Note the release is unconditional, including on the short path below
    /// [`MIN_PAR_ITEMS`]. Defect #31 is the opposite choice -- `int -> int`
    /// keeps the GIL for the whole loop (`graph_properties_copy.cc:38`)
    /// because `is_python` is `true` there, so every other Python thread in
    /// the process stalls for the duration of a copy that touches no Python
    /// object at all.
    fn copy(
        py: Python<'_>,
        bound: Bound<K>,
        src: &DenseProp<S, K>,
        dst: &mut DenseProp<T, K>,
    ) -> Result<(), PropError> {
        let (s, d) = runs(bound, src, dst)?;
        let failure = detach(py, move || {
            if d.len() < MIN_PAR_ITEMS {
                return convert_run(0, s, d);
            }
            d.par_chunks_mut(PAR_GRAIN)
                .zip(s.par_chunks(PAR_GRAIN))
                .enumerate()
                .map(|(c, (dc, sc))| convert_run(c * PAR_GRAIN, sc, dc))
                .reduce(|| None, earlier)
        });
        match failure {
            None => Ok(()),
            Some((_, e)) => Err(e),
        }
    }
}

/// Copy one property map into another, choosing the mode by lattice meet.
///
/// The predicate is *derived*, per monomorphised leaf, which is precisely what
/// a single compile-time `gil_release` flag structurally cannot do.
pub fn copy_prop<S, T, K: IdTag>(
    py: Python<'_>,
    bound: Bound<K>,
    src: &DenseProp<S, K>,
    dst: &mut DenseProp<T, K>,
) -> Result<(), PropError>
where
    S: ModeOf,
    T: ModeOf,
    S::Mode: Meet<T::Mode>,
    <S::Mode as Meet<T::Mode>>::Out: CopyStrategy<S, T, K>,
{
    <<S::Mode as Meet<T::Mode>>::Out as CopyStrategy<S, T, K>>::copy(py, bound, src, dst)
}

// The truth table graph-tool computes by hand, five times, and gets wrong in
// both directions (`graph_properties_copy.cc:35-42, :69-76, :104-111,
// :145-152`; `graph_properties_copy.hh:36-40`). Here it is not computed at
// all -- it is *resolved*, and this block makes the compiler state the answer
// for the four corners. A `const` block rather than a `#[test]` because there
// is no runtime path to a different answer: if this ever disagrees, the crate
// does not build.
const _: () = {
    assert!(<ModeFor<f64, f64> as Mode>::PARALLEL);
    assert!(<ModeFor<i64, Vec<String>> as Mode>::PARALLEL);
    assert!(!<ModeFor<PyValue, PyValue> as Mode>::PARALLEL);
    // The two rows the C++ `||` gets backwards: one Python side is enough to
    // force `Seq`, and `object -> object` is emphatically not the case that
    // may be parallelised.
    assert!(!<ModeFor<f64, PyValue> as Mode>::PARALLEL);
    assert!(!<ModeFor<PyValue, f64> as Mode>::PARALLEL);
};

/// Run a GIL-free region.
///
/// `F: Send` means nothing carrying a `Python<'py>` or a `Bound<'py, _>` can
/// be captured in. This is [`GILRelease`](https://git.skewed.de) with the
/// bound attached, and the bound is the part graph-tool cannot express:
/// `GILRelease(bool)` (`gil_release.hh:31-35`) saves the thread state only if
/// `PyGILState_Check()`, so it is a **silent no-op on any thread that does not
/// already hold the GIL** -- which is every OpenMP worker.
#[inline]
pub fn detach<R, F>(py: Python<'_>, f: F) -> R
where
    F: Send + FnOnce() -> R,
    R: Send,
{
    py.allow_threads(f)
}

/// Whether a member forces serial execution. Runtime mirror of the lattice,
/// for the erased dispatch path.
pub const fn is_serial(kind: ValueKind) -> bool {
    matches!(kind, ValueKind::PyObject)
}

#[cfg(test)]
mod tests {
    //! The runtime half. The compile-time half is the `const` block above and
    //! `tests/ui/u30_*.rs`; the cross-crate half is `tests/u30_gil.rs`.

    use super::*;

    /// `is_serial` is a *mirror*, and a mirror that drifts is worse than no
    /// mirror: `graph_properties.hh:61-76` keeps two such lists in step by
    /// position alone. One arm per member, checked against the lattice.
    #[test]
    fn the_runtime_mirror_agrees_with_the_lattice_member_by_member() {
        macro_rules! agree {
            ($($t:ty),+ $(,)?) => {$(
                assert_eq!(
                    is_serial(<$t as PropValue>::KIND),
                    !<<$t as ModeOf>::Mode as Mode>::PARALLEL,
                    "is_serial({:?}) disagrees with the lattice",
                    <$t as PropValue>::KIND,
                );
            )+};
        }
        agree!(
            u8,
            i16,
            i32,
            i64,
            f64,
            gt_core::prop::LongDouble,
            String,
            Vec<u8>,
            Vec<i16>,
            Vec<i32>,
            Vec<i64>,
            Vec<f64>,
            Vec<gt_core::prop::LongDouble>,
            Vec<String>,
            PyValue,
        );
    }

    /// Exactly one of the fifteen is serial, and it is the Python one.
    #[test]
    fn exactly_one_member_is_serial() {
        let serial: Vec<ValueKind> = ValueKind::ALL
            .into_iter()
            .filter(|&k| is_serial(k))
            .collect();
        assert_eq!(serial, vec![ValueKind::PyObject]);
    }

    /// The grain is a constant. DESIGN.md §8's whole claim is that no
    /// partition may be a function of the pool size, and the cheapest way for
    /// that to regress is for someone to "improve" this into
    /// `rayon::current_num_threads()`.
    #[test]
    fn the_parallel_partition_does_not_depend_on_the_pool() {
        let chunks = |n: usize| n.div_ceil(PAR_GRAIN);
        assert_eq!(chunks(MIN_PAR_ITEMS), 2);
        assert_eq!(chunks(PAR_GRAIN), 1);
        assert_eq!(chunks(PAR_GRAIN + 1), 2);
        assert_eq!(chunks(0), 0);
    }

    /// Lowest index wins, whichever order the tree folds in.
    #[test]
    fn the_reported_failure_is_the_lowest_indexed_one() {
        let a = Some((7usize, PropError::NotReadable));
        let b = Some((3usize, PropError::NotWritable));
        assert_eq!(earlier(a.clone(), b.clone()), b);
        assert_eq!(earlier(b.clone(), a.clone()), b);
        assert_eq!(earlier(None, a.clone()), a);
        assert_eq!(earlier(a.clone(), None), a);
        assert_eq!(earlier(None, None), None);
        // A tie keeps the left operand, so the fold is total rather than
        // merely "usually deterministic".
        let c = Some((3usize, PropError::NotReadable));
        assert_eq!(earlier(c.clone(), b.clone()), c);
    }
}
