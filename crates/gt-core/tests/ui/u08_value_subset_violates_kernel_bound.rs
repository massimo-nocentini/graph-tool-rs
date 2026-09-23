//! DESIGN.md §6, second half of the `value_subset!` guarantee.
//!
//! The kernel trait is generated *with its bound*, so a subset cannot list a
//! member whose type the kernel body could not handle. `std::string` is a
//! perfectly good member of the value universe (`graph_properties.hh:66`) and
//! a perfectly bad member of an arithmetic axis; graph-tool's answer is that
//! `scalar_types` (`:78-80`) is a separate `hana::filter` maintained beside
//! the dispatch macros, and putting the wrong tuple in a `GT_DISPATCH` call
//! fails deep inside the kernel body with an error about `operator+`.
//!
//! Here it fails at the *declaration*:
//!
//! ```text
//! error[E0277]: the trait bound `String: Scalar` is not satisfied
//! ```
//!
//! The same diagnostic is what keeps `long double` out of every arithmetic
//! axis: [`LongDouble`](gt_core::prop::LongDouble) deliberately implements
//! neither `ToF64` nor `Scalar`, so `LongDouble => gt_core::prop::LongDouble`
//! in a `Scalar`-bounded subset is this error too.

use gt_core::prop::Scalar;
use gt_core::value_subset;

value_subset!(
    /// `Str` is a genuine member; `String` is genuinely not a `Scalar`.
    pub Bad2, Bad2Kernel, "value", Scalar,
    { Str => String }
);

fn main() {}
