//! Deterministic parallelism.
//!
//! Three things graph-tool's OpenMP layer gets wrong, all structural:
//!
//! * `parallel_loop_spawn` (`parallel_util.hh:438-446`) declares
//!   `std::exception_ptr eptr{}` *outside* `#pragma omp parallel` and assigns
//!   it *inside*, so every thread races on a refcounted handle. Four verbatim
//!   copies live in `graph_properties_copy.cc`. Here the error is a **return
//!   value** ([`try_det_reduce`]); there is no shared slot to race on.
//! * `merge_split.hh:1131-1142` accumulates `log_sum_exp` under
//!   `#pragma omp critical`, so the floating-point association order is the
//!   thread interleaving and the result is not reproducible even at fixed
//!   thread count. [`Plan`] fixes the chunk count and the fold order.
//! * `parallel_rng.hh:56-61` returns `_rngs[tnum - 1]`, so results depend on
//!   `OMP_NUM_THREADS`, and `get_rngs` (`:183-191`) caches streams in a
//!   process-global map keyed on the *address* of the caller's generator and
//!   never evicts -- so a freed generator whose address is reused inherits the
//!   previous object's streams. [`Seed::split`] is keyed on the **chunk
//!   index**.

pub mod locks;
pub mod plan;
pub mod reduce;

pub use locks::{Pair, RowLocks, pair_mut};
pub use plan::{Plan, Seed};
pub use reduce::{ChunkSum, det_reduce, try_det_reduce};
