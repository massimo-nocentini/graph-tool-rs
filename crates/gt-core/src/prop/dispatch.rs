//! Monomorphising dispatch from a runtime [`ValueKind`] to a static type.
//!
//! ## The two failure modes this closes
//!
//! `dispatcher` (`dispatch.hh:186-286`) walks a **linear scan** over a Hana
//! product guarded by a mutable `bool found`, doing up to three `any_cast`
//! probes per candidate (`:248-267`) plus a `try`/`catch(bad_extract)` per
//! candidate on the `python::object` path. If nothing matches it throws
//! `DispatchNotFound`, whose message is "This is a graph_tool bug. :-("
//! (`:86-88`) -- so a user type error and a missing instantiation are
//! indistinguishable.
//!
//! [`value_subset!`] generates the member list, the accepted-kind array, the
//! narrowing, the kernel trait *with its bound*, and the exhaustive match,
//! **from one token list**. Two consequences:
//!
//! 1. A member whose type the kernel cannot handle does not compile. Putting
//!    `Str => String` into a subset bounded by [`Scalar`](super::Scalar) is
//!    `error[E0277] ... required by a bound in <Kernel>::call`, pointed at the
//!    macro invocation.
//! 2. A row whose [`ValueKind`] and type disagree does not compile. That
//!    matters: transposing two rows otherwise builds cleanly and produces a
//!    *runtime* error naming the wrong type as offered and listing it as
//!    accepted -- strictly worse than `DispatchNotFound`, which at least
//!    prints honestly.
//!
//! ## What remains possible ([DESIGN](crate::design) defect table, row 34)
//!
//! The narrowing ends in a `_ => Err` arm, because a subset is by definition
//! partial. So adding a 16th [`ValueKind`] does not break the build at every
//! call site; it makes that member unsupported at runtime wherever no subset
//! lists it. The mitigation is a coverage test asserting that every
//! `ValueKind::ALL` member appears in at least one shipped subset. This is the
//! one place the "no `DispatchNotFound`" claim is only partial, and it is
//! stated rather than hidden.

use crate::ids::IdTag;
use crate::prop::dense::DenseProp;
use crate::prop::value::{PropValue, ValueKind};
use std::any::Any;
use std::marker::PhantomData;

/// A type-erased property map at the dispatch boundary.
///
/// One `TypeId` comparison, where `std::any` needs `any_cast<T>` then
/// `any_cast<reference_wrapper<T>>` then `any_cast<shared_ptr<T>>` per
/// candidate (`dispatch.hh:248-268`): Rust's ownership model collapses that
/// fan-out, because the boundary always hands out `&'a mut`.
pub struct AnyMap<'a, K: IdTag> {
    kind: ValueKind,
    cell: &'a mut dyn Any,
    _k: PhantomData<fn() -> K>,
}

impl<'a, K: IdTag> AnyMap<'a, K> {
    /// Erase a concrete map.
    pub fn new<V: PropValue>(m: &'a mut DenseProp<V, K>) -> Self {
        AnyMap {
            kind: V::KIND,
            cell: m,
            _k: PhantomData,
        }
    }

    /// Which member this map stores.
    #[inline]
    pub const fn kind(&self) -> ValueKind {
        self.kind
    }

    /// Recover the concrete map. Crate-visible: callers go through a
    /// generated `dispatch`, which performs this once per call and discharges
    /// it against the const assertion.
    #[inline]
    pub fn downcast<V: PropValue>(&mut self) -> Option<&mut DenseProp<V, K>> {
        self.cell.downcast_mut::<DenseProp<V, K>>()
    }
}

/// A generic operation over the whole 15-member universe.
///
/// Replaces `hana::for_each(value_types, f)`. Object-unsafe by construction,
/// therefore always monomorphised -- the same property as the C++ generic
/// lambda, which is what makes this a like-for-like port rather than a
/// regression to dynamic dispatch.
pub trait KindVisitor {
    /// What the operation produces.
    type Out;
    /// Run at one concrete member.
    fn visit<V: PropValue>(self) -> Self::Out;
}

/// Declare a subset of the value universe together with its dispatcher.
///
/// ```ignore
/// value_subset!(
///     /// The arithmetic scalars.
///     pub ScalarV, ScalarKernel, "value", gt_core::prop::Scalar,
///     { Bool => u8, I16 => i16, I32 => i32, I64 => i64, F64 => f64 }
/// );
/// ```
#[macro_export]
macro_rules! value_subset {
    (
        $(#[$meta:meta])*
        $vis:vis $name:ident, $kernel:ident, $axis:literal, $bound:path,
        { $($variant:ident => $ty:ty),+ $(,)? }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        $vis enum $name { $(
            #[doc = concat!("The `", stringify!($variant), "` member.")]
            $variant
        ),+ }

        // The row-agreement proof. Without it, `$variant => $ty` is an
        // unchecked assertion and transposing two rows compiles.
        $(
            const _: () = assert!(
                matches!(<$ty as $crate::prop::PropValue>::KIND, $crate::prop::ValueKind::$variant),
                concat!(
                    "value_subset! ", stringify!($name), ": variant ", stringify!($variant),
                    " is mapped to ", stringify!($ty), ", whose KIND differs"
                )
            );
        )+

        #[doc = concat!("A kernel monomorphised over every member of [`", stringify!($name), "`].")]
        $vis trait $kernel<K: $crate::ids::IdTag> {
            /// What the kernel produces.
            type Out;
            /// Run at one concrete member, with the map already downcast.
            fn call<V: $bound>(self, map: &mut $crate::prop::DenseProp<V, K>) -> Self::Out;
        }

        impl $name {
            /// Every member this subset accepts, for the error message.
            $vis const ACCEPTED: &'static [$crate::prop::ValueKind] =
                &[$($crate::prop::ValueKind::$variant),+];

            /// Narrow a runtime kind into this subset.
            ///
            /// The **only** fallible step in a dispatch. Everything after it
            /// is total.
            $vis fn narrow(k: $crate::prop::ValueKind)
                -> ::core::result::Result<Self, $crate::error::DispatchError>
            {
                match k {
                    $($crate::prop::ValueKind::$variant => ::core::result::Result::Ok($name::$variant),)+
                    other => ::core::result::Result::Err($crate::error::DispatchError {
                        axis: $axis,
                        offered: other,
                        accepted: Self::ACCEPTED,
                    }),
                }
            }

            /// Run `kernel` at the concrete member, handing it the downcast map.
            ///
            /// Exhaustive: no `bool found`, no linear scan, no second failure
            /// point inside the kernel body.
            #[inline]
            $vis fn dispatch<K: $crate::ids::IdTag, T: $kernel<K>>(
                self,
                map: &mut $crate::prop::dispatch::AnyMap<'_, K>,
                kernel: T,
            ) -> T::Out {
                match self {
                    $($name::$variant => kernel.call::<$ty>(
                        map.downcast::<$ty>()
                            .expect("kind/type agreement is proved by value_subset!'s const assertion"),
                    ),)+
                }
            }
        }
    };
}
