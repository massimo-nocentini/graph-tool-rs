//! Defect #4 (`entries.hh:108-119`, `:223`): `get_field` falls through to a
//! shared `_dummy` cell whenever neither endpoint is in the `(r, nr)` plane, so
//! two *different* out-of-plane block pairs silently accumulate into one entry.
//! The undirected case has the same shape from the other side: `resize`
//! (`entries.hh:45-56`) only allocates `_r_in_field`/`_nr_in_field` under
//! `if constexpr (directed)`, yet the accessors that name them exist
//! unconditionally.
//!
//! Here `Field<D>` is *total* over `D::N_FIELDS`: the in-half constants are
//! inherent to `Field<Directed>` and are not nameable at all for
//! `Field<Undirected>`, which therefore has exactly the two inhabitants its
//! `N_FIELDS = 2` promises.
//!
//! ```text
//! error[E0599]: no associated function or constant named `R_IN` found for struct `gt_core::dir::Field<Undirected>` in the current scope
//! ```

use gt_core::dir::{Field, Undirected};

fn main() {
    let _ = Field::<Undirected>::R_IN;
}
