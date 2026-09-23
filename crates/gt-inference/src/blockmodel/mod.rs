//! The stochastic block model.

mod audit;
mod cache;
mod entropy;
mod record;
mod state;

pub use audit::{AuditError, audit_commit, audit_price};
// `audit_absolute` is the O(E + B^2) from-scratch recompute behind the
// `audit-full` feature (Cargo.toml: the replacement for graph-tool's
// `__test__ = True`). U26 declared it `pub` but it was unreachable from
// outside the crate, so its own acceptance sweep had to live in a private
// `mod sweep` inside audit.rs. Exported here under the same feature gate that
// guards the cost, so the drift check can be driven from an integration test
// or a bench.
#[cfg(feature = "audit-full")]
pub use audit::audit_absolute;
pub use cache::Cache;
// `entropy` is a private module, so anything it declares `pub` is still
// unreachable from outside the crate unless it is named here. Three items were
// missing and are now listed:
//
// * `edges_dl` (`entropy.hh:293-298`), a description-length term of the same
//   functional as its sibling `partition_dl`, which *was* listed;
// * `sparse_ds` and `dense_ds` (`state.hh:1224-1255` / `:1138-1220`), the two
//   pricing entry points. Both are `pub`, both take only public types
//   (`delta::Delta`, `EntropyParams`, `Cache`, and for `dense_ds` a
//   `state::BlockView`), and without this line no integration test and no
//   bench outside `gt-inference` can name either one.
pub use entropy::{
    EntropyParams, dense_ds, edges_dl, eterm, eterm_dense, partition_dl, sparse_ds, vterm,
};
// `ScanDir` is a bound on `record`/`scan`, so it has to be nameable outside
// the crate: without it a caller can only write *monomorphic* wrappers
// (`record::<_, Directed, _>`), never one generic over directedness, because
// the `S::D: ScanDir<G>` obligation cannot be written down. Calling
// `record`/`scan` at a concrete directedness never needs to name it.
pub use record::{ScanDir, propagate, record, scan};
// `state` is private too. Alongside the three traits and `GroupLocks`, the
// concrete implementation U24 landed has to be nameable from outside the
// crate or it is dead on arrival: `Aggregates` is the storage the pricing and
// commit paths both address, `BlockState` is the only type in the workspace
// that implements all three traits, `RowImage` is what `Aggregates::image`
// returns, and `StampMismatch` is the error half of `BlockState`'s inherent
// `try_commit` / `try_commit_shared` -- which exist precisely because
// `BlockCommit::commit` returns `Receipt` by value and can refuse only by
// panicking.
pub use state::{
    Aggregates, BlockCommit, BlockCommitShared, BlockState, BlockView, GroupLocks, RowImage,
    StampMismatch,
};
