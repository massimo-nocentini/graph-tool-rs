//! # Implementation plan
//!
//! The skeleton in `crates/` compiles and its module tree is **complete**. No unit
//! below needs to add a module, a `use` in a crate root, or a re-export: every
//! public item the design calls for is already declared. Implementers fill
//! `todo!()` bodies.
//!
//! Each `todo!()` in the tree is tagged with its unit number, so
//! `grep -rn 'todo!("U5' crates` is the exact worklist for unit 5.
//!
//! ## Rules
//!
//! 1. **File ownership is disjoint and exhaustive.** A unit edits only the files
//!    listed under *Owns*. If a unit believes it needs to touch a file it does not
//!    own, that is a design error: stop and raise it against [DESIGN](crate::design)
//!    rather than editing across the boundary.
//! 2. **Crate roots are frozen.** The five `crates/*/src/lib.rs` files are owned by
//!    **U-FINAL** and by nothing else. The module tree is already declared.
//! 3. **Tests go in the unit's own file**, named after the unit, e.g.
//!    `crates/gt-core/tests/u05_adjlist.rs`. Never in a shared test file.
//! 4. **`cargo build --workspace` and `cargo clippy --workspace --all-targets
//!    --all-features` must pass after every unit**, with zero warnings. They pass
//!    now; keep them passing.
//! 5. **A unit is not done because it compiles.** §16 of [DESIGN](crate::design) exists because
//!    every source design said "verified by compiling it" and shipped a defect its
//!    own claims denied. Each unit's acceptance test below is the minimum, not the
//!    target.
//! 6. `todo!()` bodies may be left in a file a unit owns *only* if the unit's
//!    description says so (a few units deliberately land in two passes).
//!
//! ## Dependency order
//!
//! ```text
//! U0 ──┬─> U1 ─┬─> U2 ─> U3 ─> U4 ─> U5 ─┬─> U6
//!      │       │                          ├─> U7
//!      │       ├─> U8 ─> U9 ─> U33        │
//!      │       └─> U10 ─> U11             │
//!      │             └──> U12             │
//!      │                                  ├─> U13..U17   (gt-algo)
//!      │                                  ├─> U18 ─> U19 ─> U20 ─> U21
//!      │                                  │              └─> U22 ─> U23 ─> U24 ─> U25 ─> U26
//!      │                                  ├─> U27 ─> U28, U29
//!      │                                  └─> U30 ─> U31 ─> U32
//!      └────────────────────────────────────────────────────────> U-FINAL
//! ```
//!
//! U2–U5 are a chain (storage is one object). Everything after U5 parallelises
//! widely: U6/U7, U13–U17, U18–U26, U27–U29 and U30–U32 are four independent
//! streams.
//!
//! ---
//!
//! ## U0 — Test harness
//!
//! **Owns:** `Cargo.toml` (workspace), `crates/gt-core/Cargo.toml`,
//! `crates/gt-algo/Cargo.toml`, `crates/gt-inference/Cargo.toml`,
//! `crates/gt-io/Cargo.toml`, `crates/gt-py/Cargo.toml`.
//!
//! **Fills:** nothing in `src/`. Adds `[dev-dependencies]` to each crate:
//! `proptest`, `criterion`, `trybuild`; registers `[[bench]]` targets with
//! `harness = false`.
//!
//! **Acceptance:** `cargo build --workspace --all-targets` and
//! `cargo clippy --workspace --all-targets --all-features` pass with zero
//! warnings. `cargo bench --no-run` succeeds.
//!
//! ---
//!
//! ## U1 — Identifiers, directedness, bounds, errors, trait layer
//!
//! **Owns:** `crates/gt-core/src/ids.rs`, `crates/gt-core/src/dir.rs`,
//! `crates/gt-core/src/bound.rs`, `crates/gt-core/src/error.rs`,
//! `crates/gt-core/src/graph.rs`. **Tests:** `crates/gt-core/tests/u01_foundation.rs`.
//!
//! **Signatures to fill:** `GraphId::fresh` is already implemented; nothing else
//! in these files has a body to write. The unit's work is the *tests*, because
//! these files encode the guarantees everything downstream assumes.
//!
//! **Acceptance (all must be assertions, not inspection):**
//! * `size_of::<VertexId>() == size_of::<Raw>()`, `size_of::<Bound<VertexTag>>()`
//!   stable, `Field<Undirected>` has exactly two inhabitants and
//!   `Field::<Directed>::R_IN.index() < Directed::N_FIELDS`.
//! * `GraphId::fresh()` returns distinct values across 10 000 calls and from two
//!   threads.
//! * **trybuild compile-fail fixtures**, each asserting the exact diagnostic:
//!   - `Bound::new` from a downstream crate → `E0624`.
//!   - `Rev<Und<&AdjList>>` → `E0271`.
//!   - `Und<&AdjList>::in_edges` → `E0599`.
//!   - `&dyn GraphRef` → compile error (the trait is intentionally not a trait
//!     object; `DynGraph` is).
//!
//!   These four are quoted in [DESIGN](crate::design) §14 and must not be allowed to regress.
//!
//! ---
//!
//! ## U2 — Adjacency block and the edge-id allocator
//!
//! **Owns:** `crates/gt-core/src/adj/mod.rs`, `crates/gt-core/src/adj/entry.rs`,
//! `crates/gt-core/src/adj/block.rs`, `crates/gt-core/src/adj/alloc.rs`. **Tests:** `crates/gt-core/tests/u02_block.rs`.
//!
//! **Fills:** `Block::{get, insert_out, insert_in, remove_at, shrink}`,
//! `EdgeIds::{alloc, release, compact}`.
//!
//! `insert_out` **must** be the O(1) push-swap of `graph_adjacency.hh:1192-1215`
//! (push `entries[out_len]` to the back, overwrite the slot). An `insert` at the
//! boundary is O(in-degree) and was measured quadratic; see [DESIGN](crate::design) §3.
//!
//! **Acceptance:**
//! * `size_of::<Block>() == 32` and `size_of::<AdjEntry>() == 8`, asserted.
//! * Property test: random `insert_out`/`insert_in`/`remove_at` sequences against a
//!   naive `(Vec<AdjEntry>, usize)` model; after every op the out-half and in-half
//!   contents match the model as multisets, and every returned `Moved` names an
//!   entry that really is at `to`.
//! * `EdgeIds`: `live() == next() - free.len()` after every op; `alloc` after
//!   `release` reuses the released id; `compact()` is a permutation.
//! * **Benchmark asserting non-quadratic `insert_out`**: 10k/20k/40k/80k
//!   out-insertions onto a vertex with that many pre-existing in-edges must scale
//!   within 2.5× per doubling, not 4×.
//!
//! ---
//!
//! ## U3 — The edge slot table and the `(s,t)` lookup
//!
//! **Owns:** `crates/gt-core/src/adj/index.rs`.
//! **Tests:** `crates/gt-core/tests/u03_index.rs`.
//!
//! **Fills:** `EdgeSlots::{endpoints, locate, on_insert, on_move, on_remove,
//! rebuild}`; `EHash::{on_link, on_unlink, find, rebuild}`.
//!
//! `locate` performs the identity check **itself**, against the half named by
//! `end`, bounds-checked. A caller-side guard passes whenever the two halves share
//! a position; see [DESIGN](crate::design) D12.
//!
//! **Acceptance:**
//! * `size_of::<EdgeSlot>() == 4 * size_of::<Raw>()`.
//! * `locate(blocks, id, Out)` and `locate(blocks, id, In)` disagree about the
//!   block whenever `src != tgt`, and agree when `src == tgt`.
//! * `locate` returns `None`, never panics, for a released id, an out-of-range id,
//!   and a slot whose block entry has been overwritten.
//! * `EHash` after a random link/unlink sequence equals a brute-force
//!   `HashMap<(V,V), Vec<E>>`; `rebuild` reproduces the incremental state exactly.
//!
//! ---
//!
//! ## U4 — Iterators
//!
//! **Owns:** `crates/gt-core/src/adj/iter.rs`.
//! **Tests:** `crates/gt-core/tests/u04_iter.rs`.
//!
//! **Fills:** `Edges::next` (the only `todo!()`).
//!
//! **Acceptance:**
//! * `edges()` yields each edge **exactly once**, in `EdgeId` order after a
//!   compaction, and skips empty blocks without an O(V) stall per edge.
//! * `IncidentIter::fold` and `next` agree; `size_hint` is exact.
//! * Codegen assertion: the disassembly of
//!   `fn s(g: &AdjList, v: VertexId) -> u64 { g.out_edges(v).map(|i| i.other.raw() as u64).sum() }`
//!   in `--release` contains **no** `panic_bounds_check` and no allocator call.
//!
//! ---
//!
//! ## U5 — `AdjList`: the two splice primitives
//!
//! **Owns:** `crates/gt-core/src/adj/list.rs`.
//! **Tests:** `crates/gt-core/tests/u05_adjlist.rs`.
//!
//! **Fills:** everything except `new`/`with_lookup`, which are done. In
//! particular `splice_in`, `splice_out`, `add_edge`, `remove_edge`,
//! `clear_vertex_where`, `swap_remove_vertex`, `find_edge`, `validate`.
//!
//! `splice_out` must re-locate the in-half **after** splicing the out-half; the
//! borrow checker enforces the ordering, but the code must not work around it.
//! `clear_vertex_where` uses `self.scratch`, not a fresh `Vec`.
//! `swap_remove_vertex` relabels by unlink-then-relink so both index hooks fire
//! for both directions.
//!
//! **Acceptance:**
//! * **Model-based proptest, the centrepiece of this unit.** 2 000+ cases ×
//!   {`NoLookup`, `EHash`}, ≤ 6 vertices, sequences of up to 16
//!   `add_edge`/`remove_edge`/`clear_vertex`/`clear_vertex_where`/`swap_remove_vertex`
//!   including self-loops and parallel edges. After **every** operation:
//!   `validate()` is `Ok`; `num_edges()` matches a `BTreeMap<EdgeId,(V,V)>`
//!   reference; the full sorted out-half **and in-half** match the reference.
//!   (Checking only the out-half is what let one source design ship a `validate`
//!   that never inspected the in-half at all.)
//! * `clear_vertex` on a vertex with a self-loop plus one other out-edge, under a
//!   predicate matching only the self-loop, decrements `num_edges()` by exactly 1.
//!   This is defect #1.
//! * `remove_edge` of an id obtained from `find_edge(t, s).reversed()` behaves
//!   identically to `remove_edge` of the forward id (there is no orientation to
//!   get wrong; assert it).
//! * Allocation assertion: 10 000 `clear_vertex` calls on a 20k/160k graph perform
//!   **zero** allocations after warm-up.
//!
//! ---
//!
//! ## U6 — Deterministic parallel construction
//!
//! **Owns:** `crates/gt-core/src/adj/builder.rs`.
//! **Tests:** `crates/gt-core/tests/u06_builder.rs`.
//!
//! **Fills:** `ParBuilder::{new, push, fill, build}`.
//!
//! **Acceptance:** the same input built with `RAYON_NUM_THREADS` = 1, 4 and 16
//! produces **byte-identical** edge-id assignments and an identical `validate()`ing
//! graph. A throughput benchmark against serial `add_edge` on 10⁷ edges.
//!
//! ---
//!
//! ## U7 — The view algebra
//!
//! **Owns:** `crates/gt-core/src/view/mod.rs`,
//! `crates/gt-core/src/view/undirected.rs`,
//! `crates/gt-core/src/view/reversed.rs`,
//! `crates/gt-core/src/view/filtered.rs`.
//! **Tests:** `crates/gt-core/tests/u07_views.rs`.
//!
//! **Fills:** `MaskFilter::{keep_vertex, keep_edge}`; `Filtered::{new, masked,
//! out_degree, degree, in_degree}`; `Filtered`'s `Endpoints` impl; the `Iterator`
//! impls for `FilterIncident`, `FilterEdges`, `FilterVertices`.
//!
//! `Filtered::new` and `FilterEdges` must both go through `keeps_edge`. Two
//! predicates is how `num_edges()` comes to disagree with `edges().count()`.
//!
//! **Acceptance:**
//! * On the path `0→1→2`: `g.undirect().out_edges(1)` yields `other ∈ {0, 2}` —
//!   **never 1**. This is defect #14 and it is the single most important assertion
//!   in this unit.
//! * `g.reverse().out_edges(v)` equals `g.in_edges(v)` as a multiset, and is
//!   **not** their union.
//! * For every one of the six views: `num_vertices() == vertices().count()` and
//!   `num_edges() == edges().count()`, on a graph with a filter that removes
//!   vertices, edges, and both.
//! * `vertex_bound()` is unchanged by filtering; `num_vertices()` is not.
//! * `Filtered::masked` with a mask shorter than the bound returns
//!   `Err(ShortMask)`, and a mask validated against graph A cannot be used with
//!   graph B (borrowck or `Err`, assert which).
//! * `size_of` assertions for all six views ([DESIGN](crate::design) §11).
//! * trybuild: `g.undirect().reverse()` → `E0599`.
//!
//! ---
//!
//! ## U8 — The value universe and the subset macro
//!
//! **Owns:** `crates/gt-core/src/prop/value.rs`,
//! `crates/gt-core/src/prop/dispatch.rs`.
//! **Tests:** `crates/gt-core/tests/u08_value.rs`.
//!
//! **Fills:** `ValueKind::from_name`.
//!
//! **Acceptance:**
//! * `ValueKind::ALL` round-trips through `name()`/`from_name()`, and every name
//!   is **byte-identical** to `graph_properties.hh:71-75` — including `"bool"` for
//!   `uint8_t`, which the Python `PropertyMap.value_type()` surface depends on.
//! * `is_scalar`/`is_integer`/`is_floating`/`is_vector`/`elem` agree with the C++
//!   `hana::filter` sets (`:78-104`), with the one documented divergence
//!   (`long double` is not `Scalar`) asserted explicitly so it cannot drift
//!   silently.
//! * trybuild fixtures: a transposed `value_subset!` row → `E0080` with the
//!   generated message; a member violating the kernel bound → `E0277`. Both are
//!   quoted in [DESIGN](crate::design) §6.
//! * `size_of::<LongDouble>() == 16`; a `LongDouble` round-trips through a
//!   `DenseProp` unchanged.
//!
//! ---
//!
//! ## U9 — Property maps
//!
//! **Owns:** `crates/gt-core/src/prop/mod.rs`, `crates/gt-core/src/prop/map.rs`,
//! `crates/gt-core/src/prop/dense.rs`, `crates/gt-core/src/prop/index_map.rs`. **Tests:** `crates/gt-core/tests/u09_props.rs`.
//!
//! **Fills:** `DenseProp::{sized_for, sized_for_with, view, soft_reserve}`.
//!
//! **Acceptance:**
//! * `sized_for` on a map belonging to a different graph returns
//!   `Err(WrongGraph)`. This is defect #8 and #7.
//! * `sized_for` grows to exactly `bound.len()`, is idempotent, and never shrinks.
//! * `sized_for_with` fills new slots from the closure and leaves existing ones.
//! * `view` on an under-sized map returns `Err(Undersized)`, never a short slice.
//! * `size_of::<Unity<f64, VertexTag>>() == 0`, asserted.
//! * `IndexProp::get_ref(k) == k.index()`, and `IndexProp` satisfies
//!   `ReadProp` but **not** `WriteProp` (trybuild → `E0599`).
//! * trybuild: `Unity::put` → `E0599` (defect #43).
//! * Codegen assertion: a scan through `PropSlice::as_slice()` has no
//!   `panic_bounds_check`; a per-key scatter does (documented cost, [DESIGN](crate::design)
//!   §12).
//!
//! ---
//!
//! ## U10 — Work partitioning and seeding
//!
//! **Owns:** `crates/gt-core/src/par/mod.rs`, `crates/gt-core/src/par/plan.rs`.
//! **Tests:** `crates/gt-core/tests/u10_plan.rs`.
//!
//! **Fills:** `Plan::{new, range}`, `Seed::split`.
//!
//! **Acceptance:**
//! * Exhaustive exact-cover check: for every `n` in `0..300` and `grain` in
//!   `1..40`, `(0..plan.chunks()).flat_map(|k| plan.range(k))` equals `0..n`
//!   exactly — no gaps, no overlaps, no empty trailing chunk.
//! * `plan.chunks()` is a pure function of `(n, grain)` and observably independent
//!   of `RAYON_NUM_THREADS`.
//! * `Seed::split(k)` streams are pairwise distinct for `k in 0..1024` and
//!   reproducible across runs; `(S, k)` and `(S ^ k, 0)` do **not** alias.
//!
//! ---
//!
//! ## U11 — Reductions
//!
//! **Owns:** `crates/gt-core/src/par/reduce.rs`.
//! **Tests:** `crates/gt-core/tests/u11_reduce.rs`.
//!
//! **Fills:** `det_reduce`, `try_det_reduce`, `chunked_sum`.
//!
//! **Acceptance:**
//! * A `log_sum_exp` reduction over 10 000 random values is **bit-identical**
//!   (`a.to_bits() == b.to_bits()`) at 1, 4 and 16 rayon threads. This is defect
//!   #27.
//! * `try_det_reduce` with several failing chunks returns the error from the
//!   **lowest** chunk index, deterministically, at every thread count. This is
//!   defect #26.
//! * `chunked_sum::<4>` matches a sequential Kahan sum to within the documented
//!   tolerance, and a disassembly assertion records whether `addpd` is present —
//!   the test must *record* the answer, not assume it ([DESIGN](crate::design) §12).
//!
//! ---
//!
//! ## U12 — Row locks
//!
//! **Owns:** `crates/gt-core/src/par/locks.rs`.
//! **Tests:** `crates/gt-core/tests/u12_locks.rs`.
//!
//! **Fills:** `pair_mut`, `RowLocks::{new, with_pair, with_row}`.
//!
//! **Acceptance:**
//! * **`pair_mut(xs, 3, 1)` returns `Two(&mut xs[3], &mut xs[1])` — in that
//!   order.** Write the test as "the first reference is `r`'s row" and assert the
//!   *value*, not the index. A sorted return silently transposes; see [DESIGN](crate::design)
//!   D9.
//! * `pair_mut(xs, 3, 3)` returns `Same`, not `None`.
//! * `pair_mut` with an out-of-range index returns `None`.
//! * A 16-thread stress test taking 10⁵ random pairs in random order completes
//!   without deadlock and leaves every counter at its expected total.
//!
//! ---
//!
//! ## U13 — Traversal
//!
//! **Owns:** `crates/gt-algo/src/traversal.rs`.
//! **Tests:** `crates/gt-algo/tests/u13_traversal.rs`.
//!
//! **Fills:** `bfs`, `bfs_multi`, `dfs`, `shortest_distances`, `dijkstra`.
//!
//! **Acceptance:** BFS distances on the six views of a fixed graph match a
//! hand-computed table; `dfs` visits every reachable vertex exactly once;
//! `Control::{Prune, Stop}` are honoured; `dijkstra` with a `Unity` weight equals
//! `shortest_distances` **and** its disassembly shows the weight loop folded away.
//!
//! ---
//!
//! ## U14 — Components
//!
//! **Owns:** `crates/gt-algo/src/components.rs`.
//! **Tests:** `crates/gt-algo/tests/u14_components.rs`.
//!
//! **Fills:** `components`, `strong_components`, `largest_component_mask`.
//!
//! **Acceptance:** `components(g.undirect())` equals a union-find reference;
//! `strong_components` equals a reference Tarjan on 500 random digraphs;
//! `largest_component_mask` fed straight into `Filtered::masked` yields a view
//! whose `num_vertices()` equals the component size.
//!
//! ---
//!
//! ## U15 — Degree kernels
//!
//! **Owns:** `crates/gt-algo/src/degree.rs`.
//! **Tests:** `crates/gt-algo/tests/u15_degree.rs`.
//!
//! **Fills:** `weighted_out_degree`, `weighted_in_degree`, `weighted_degree`,
//! `out_degrees`, `degree_histogram`.
//!
//! **Acceptance:**
//! * `weighted_out_degree` with `Unity` equals `out_degree` as an `f64`, and the
//!   disassembly of that monomorphisation contains **no loop** (defect #44's fast
//!   path, which does not exist in the C++).
//! * With `Constant { c }` it equals `c * out_degree`.
//! * With an `i64` `DenseProp` it is exact for values beyond 2⁵³ where `f64`
//!   allows — i.e. assert the `ToF64` widening is used, not `Into<f64>` (which
//!   would not compile; keep a trybuild fixture proving `i64` is admissible).
//! * No allocation per vertex: assert with an instrumented allocator.
//!
//! ---
//!
//! ## U16 — Centrality
//!
//! **Owns:** `crates/gt-algo/src/centrality.rs`.
//! **Tests:** `crates/gt-algo/tests/u16_centrality.rs`.
//!
//! **Fills:** `pagerank`, `eigenvector`, `betweenness`.
//!
//! **Acceptance:** PageRank on a fixed 1 000-vertex graph is **bit-identical** at
//! 1 and 16 threads (this is what `det_reduce` is for); values match a dense
//! power-iteration reference to 1e-12; `betweenness` matches a reference Brandes
//! on 100 small graphs.
//!
//! ---
//!
//! ## U17 — Topology
//!
//! **Owns:** `crates/gt-algo/src/topology.rs`.
//! **Tests:** `crates/gt-algo/tests/u17_topology.rs`.
//!
//! **Fills:** `is_dag`, `topological_sort`, `is_bipartite`, `global_clustering`,
//! `count_triangles`, `reciprocity`.
//!
//! **Acceptance:** `topological_sort` returns `None` exactly when `is_dag` is
//! false and otherwise a valid order; `count_triangles` counts each triangle
//! **once** on both directed and undirected views (this is where degree-summation
//! double-counts — assert against a brute-force O(V³) reference on 200 small
//! graphs).
//!
//! ---
//!
//! ## U18 — Inference identifiers
//!
//! **Owns:** `crates/gt-inference/src/ids.rs`.
//! **Tests:** `crates/gt-inference/tests/u18_ids.rs`.
//!
//! **Fills:** `StateId::fresh`.
//!
//! **Acceptance:** `size_of::<Option<Group>>() == 4`, asserted;
//! `Group::new(i).index() == i` for the full range; `StateId::fresh` distinct
//! across threads; the `Weight` impls round-trip `to_f64`/`from_i64` within their
//! exact range.
//!
//! ---
//!
//! ## U19 — The delta buffer
//!
//! **Owns:** `crates/gt-inference/src/delta/mod.rs`,
//! `crates/gt-inference/src/delta/entry.rs`,
//! `crates/gt-inference/src/delta/buf.rs`. **Tests:** `crates/gt-inference/tests/u19_deltabuf.rs`.
//!
//! **Fills:** `DeltaBuf::{begin, touch, touch_dyn, delta_of, me_of}`.
//!
//! `begin` resets exactly the slots recorded in `self.slots` and must be correct
//! **regardless of the order** relative to the move key — unlike `EntrySet::clear`
//! (`entries.hh:169-176`), which re-derives addresses from `_rnr`.
//!
//! **Acceptance:**
//! * Property test: random `touch` sequences; `delta_of(s, t)` equals a
//!   `HashMap<(Group,Group), W>` reference; `entries()` contains each touched pair
//!   exactly once.
//! * `begin` with a **different** move key after a previous recording leaves no
//!   residue: every field slot is null and `delta_of` is zero for every pair.
//! * `begin` is O(#entries): assert the number of field-slot writes, not the
//!   wall clock.
//! * `touch_dyn` on a pair touching neither endpoint returns `Err(OutOfPlane)` and
//!   leaves the buffer unchanged. This is defect #4.
//! * `size_of::<DeltaBuf<Undirected, i64>>() < size_of::<DeltaBuf<Directed, i64>>()`
//!   — the undirected buffer holds two field vectors, not four empty ones.
//!
//! ---
//!
//! ## U20 — Stack and lifecycle
//!
//! **Owns:** `crates/gt-inference/src/delta/stack.rs`,
//! `crates/gt-inference/src/delta/lifecycle.rs`.
//! **Tests:** `crates/gt-inference/tests/u20_lifecycle.rs`.
//!
//! **Fills:** `DeltaStack::with_levels`. (`below_above`, `Recording`,
//! `Transition`, `Delta`, `Applied`, `LevelIter` are complete; this unit's real
//! work is the tests and the trybuild fixtures.)
//!
//! **Acceptance:**
//! * `below_above(l)` hands out level `l` shared and `l+1` mutably at once.
//! * `into_levels()` yields exactly `n_levels()` `Applied` tokens, bottom-up, each
//!   carrying its own level index and the shared stamp.
//! * trybuild fixtures, each asserting the exact diagnostic:
//!   - applying the same `Applied` twice → `E0382` (defect #35);
//!   - `Transition::clone()` → `E0599`;
//!   - holding a `Delta` across a `&mut state` call → **compiles** (this is the
//!     whole point of the ownership inversion; assert the positive case too).
//!
//! ---
//!
//! ## U21 — Spec traits and the Metropolis loop
//!
//! **Owns:** `crates/gt-inference/src/spec.rs`,
//! `crates/gt-inference/src/metropolis.rs`.
//! **Tests:** `crates/gt-inference/tests/u21_metropolis.rs`.
//!
//! **Fills:** `accept`, `mcmc_sweep`.
//!
//! **Acceptance:**
//! * A toy `MetropolisState` at `beta = 0` accepts exactly when `log_hastings > 0`;
//!   at `beta = ∞` accepts exactly when `d_entropy < 0`.
//! * All three `Schedule` variants visit the right node multiset;
//!   `Schedule::Alternating` is reproducible across runs with a fixed seed.
//! * `nattempts`/`nmoves` accumulate `Step::nsteps`, not 1 (defect #25).
//! * trybuild fixtures:
//!   - a `GroupProposal` impl defining `get_move_lprob` inside the impl block →
//!     `E0407`;
//!   - the same in an inherent impl, leaving `log_reverse` unimplemented →
//!     `E0046`. Together these are defect #19 and they are the reason this trait
//!     is shaped as it is.
//!
//! ---
//!
//! ## U22 — The special-function cache
//!
//! **Owns:** `crates/gt-inference/src/blockmodel/mod.rs`,
//! `crates/gt-inference/src/blockmodel/cache.rs`.
//! **Tests:** `crates/gt-inference/tests/u22_cache.rs`.
//!
//! **Fills:** `Cache::{build, lgamma1p, safelog, lbinom, log_q}`.
//!
//! **`lgamma1p` must be a real Lanczos (or equivalent) `lgamma`, not a stub.** The
//! single largest evidence failure in the design phase was a crate whose
//! `ln_gamma` was `x.ln()`, which hid a pricing path that was wrong on 72% of
//! moves. It must also match graph-tool's `ignore_error` policy
//! (`cache.hh:36-48`) on out-of-domain input rather than panicking.
//!
//! **Acceptance:** `lgamma1p` agrees with a high-precision reference to 1e-13
//! relative over `[0, 10⁶]`, both inside and outside the table; `safelog(0) == 0`;
//! `lbinom` matches an exact big-integer computation for `n ≤ 60`; `Cache` is
//! `Sync` and two threads reading it concurrently agree.
//!
//! ---
//!
//! ## U23 — The entropy functional
//!
//! **Owns:** `crates/gt-inference/src/blockmodel/entropy.rs`.
//! **Tests:** `crates/gt-inference/tests/u23_entropy.rs`.
//!
//! **Fills:** `eterm`, `vterm`, `eterm_dense`, `edges_dl`, `partition_dl`,
//! `sparse_ds`, `dense_ds`.
//!
//! **Acceptance:** each kernel matches the corresponding C++ expression
//! (`entropy.hh:38, :73, :235, :293`; `partition.hh:106`) on 10⁴ random inputs to
//! 1e-12. `sparse_ds` takes **no state argument** — assert that by calling it with
//! only a `Delta` and a `Cache` in scope.
//!
//! ---
//!
//! ## U24 — Block state
//!
//! **Owns:** `crates/gt-inference/src/blockmodel/state.rs`.
//! **Tests:** `crates/gt-inference/tests/u24_state.rs`.
//!
//! **Fills:** `Aggregates::{apply_entry, apply_entries}`, plus a concrete
//! `BlockView` + `BlockCommit` + `BlockCommitShared` implementation for the
//! tests to drive.
//!
//! **Acceptance:**
//! * `commit` rejects an `Applied` whose `Stamp` names a different `StateId`
//!   **even when the epochs match** — build two fresh states, record against one,
//!   commit into the other, assert `Err`/panic. This is defect #36, and an epoch
//!   alone does not catch it.
//! * `apply_entries` and repeated `apply_entry` produce identical aggregates.
//! * `n_groups()` is derived from the occupancy set; there is no second stored
//!   copy to drift.
//! * 8-thread `commit_shared` stress: 10⁵ moves over disjoint and overlapping
//!   block pairs; final aggregates equal a serial replay.
//!
//! ---
//!
//! ## U25 — Recording and propagation
//!
//! **Owns:** `crates/gt-inference/src/blockmodel/record.rs`.
//! **Tests:** `crates/gt-inference/tests/u25_record.rs`.
//!
//! **Fills:** `record`, `scan`, `propagate`.
//!
//! `scan` must halve the doubly-traversed undirected self-loop weight
//! (`entries.hh:276-292`). A `None` neighbour group must be reported or
//! `debug_assert`ed, **never silently skipped** — a skip produces a delta quietly
//! short by one edge's weight, which is a worse failure mode than the C++'s
//! out-of-range index.
//!
//! **Acceptance:**
//! * Differential test, 300 random undirected **and** 300 random directed
//!   multigraphs with self-loops and parallel edges: the recorded delta equals a
//!   brute-force recount of `e_rs` before and after the move. This is the test
//!   that would have caught the design-phase pricing bug.
//! * `propagate` performs **no** heap allocation per call (instrumented
//!   allocator).
//! * `propagate` on a well-formed hierarchy never returns `Err(OutOfPlane)`.
//!
//! ---
//!
//! ## U26 — The always-on audit
//!
//! **Owns:** `crates/gt-inference/src/blockmodel/audit.rs`.
//! **Tests:** `crates/gt-inference/tests/u26_audit.rs`.
//!
//! **Fills:** `audit_commit`, `audit_price`, `audit_absolute`.
//!
//! **Acceptance:**
//! * `audit_commit(st, &receipt)` compiles and passes **after** a commit —
//!   assert the call site, since that is the whole reason `Receipt` exists.
//! * Fault injection: corrupt one `mrs` / one `mrp` / one `wr` by ±1 and assert
//!   the audit names the right group and field.
//! * Cost: `audit_commit` performs O(#entries) state reads, asserted by counting.
//! * Under `--features audit-full`, `audit_absolute` agrees with the accumulated
//!   delta sum over a 10 000-move sweep to 1e-9.
//!
//! ---
//!
//! ## U27 — The `.gt` binary format
//!
//! **Owns:** `crates/gt-io/src/error.rs`, `crates/gt-io/src/gt.rs`.
//! **Tests:** `crates/gt-io/tests/u27_gt.rs`.
//!
//! **Fills:** `gt::{read, write}`.
//!
//! **Acceptance:** round-trip of a graph with all 15 property types (including
//! `long double` and a Python-valued map, the latter skipped when the `python`
//! feature is off) is byte-identical; a real graph-tool-written `.gt` fixture
//! loads and re-saves byte-identically; a truncated file gives `BadMagic` or
//! `Parse`, never a panic; a file declaring an out-of-range index gives
//! `IndexOutOfRange`.
//!
//! ---
//!
//! ## U28 — GraphML and DOT
//!
//! **Owns:** `crates/gt-io/src/graphml.rs`, `crates/gt-io/src/dot.rs`.
//! **Tests:** `crates/gt-io/tests/u28_text.rs`.
//!
//! **Acceptance:** GraphML round-trip preserves typed properties and edge order;
//! DOT round-trip preserves topology (property typing is best-effort and the test
//! must assert *that*, not more); malformed input yields `Parse` with the correct
//! line number.
//!
//! ---
//!
//! ## U29 — Delimited edge lists
//!
//! **Owns:** `crates/gt-io/src/csv.rs`.
//! **Tests:** `crates/gt-io/tests/u29_csv.rs`.
//!
//! **Acceptance:** symbolic endpoints intern deterministically (first-seen order);
//! `read_edge_list` over `ParBuilder` produces identical edge ids at 1 and 16
//! threads; `write_edge_list` emits in `EdgeId` order.
//!
//! ---
//!
//! ## U30 — The GIL lattice
//!
//! **Owns:** `crates/gt-py/src/gil.rs`.
//! **Tests:** `crates/gt-py/tests/u30_gil.rs`.
//!
//! **Fills:** both `CopyStrategy::copy` impls.
//!
//! **Acceptance:**
//! * `copy_prop::<f64, f64>` resolves to `Par` and its body calls
//!   `allow_threads`; `copy_prop::<PyValue, PyValue>` resolves to `Seq` and holds
//!   the token. Assert the resolution with a type-level witness, not by
//!   inspection.
//! * trybuild fixtures, each asserting the exact diagnostic:
//!   - `Vec<PyValue>::par_iter_mut()` → `E0599` (defect #30, already verified);
//!   - a hand-written `impl ModeOf for PyValue { type Mode = Par; }` →
//!     `E0277: PyValue: Sync is not satisfied`, i.e. the lattice cannot be
//!     mis-edited into unsoundness;
//!   - `struct RawPy(Py<PyAny>); impl PropValue for RawPy` → a seal error,
//!     proving the universe cannot be re-opened.
//! * `detach` rejects a closure capturing a `Python<'_>` (`E0277`).
//!
//! ---
//!
//! ## U31 — View dispatch
//!
//! **Owns:** `crates/gt-py/src/dispatch.rs`.
//! **Tests:** `crates/gt-py/tests/u31_dispatch.rs`.
//!
//! **Fills:** `AnyGraph::{dispatch, dispatch_bidi}`.
//!
//! **Acceptance:**
//! * All six arms construct and run a kernel that exercises `out_edges`,
//!   `vertices`, `edges` and `find_edge` — i.e. the kernel's full bound set,
//!   because a view implementing only incidence cannot back a generic algorithm.
//! * `dispatch_bidi` has exactly three arms and a kernel calling `in_edges`
//!   compiles on it.
//! * `&dyn DynGraph` works; assert `for_each_out` visits the same multiset as
//!   `out_edges`.
//! * trybuild: adding a seventh `ViewKind` variant → `E0004` (defect #48's
//!   *eliminated* half).
//!
//! ---
//!
//! ## U32 — The extension module
//!
//! **Owns:** `crates/gt-py/src/value.rs`, `crates/gt-py/src/module.rs`.
//! **Tests:** `crates/gt-py/tests/u32_module.rs`.
//!
//! **Fills:** `PyGraph`'s methods, `register`.
//!
//! **Every entry point must size *all* maps, including read-only ones**, before
//! entering `detach`. That preserves graph-tool's auto-grow-on-read semantics
//! (`fast_vector_property_map.hh:129`); see [DESIGN](crate::design) §5.
//!
//! **Acceptance:** `every_value_kind_is_reachable_from_some_subset` (already
//! written) passes; a Python-level smoke test under
//! `--features extension-module` builds a graph, sets a property, reads it back
//! from a never-written key and gets the default; an exception raised inside a
//! kernel surfaces as a `PyErr` and does not abort under `panic = "abort"` —
//! if it does, that is a finding against [DESIGN](crate::design) §10 and must be raised.
//!
//! ---
//!
//! ## U33 — Conversions and type erasure
//!
//! **Owns:** `crates/gt-core/src/prop/convert.rs`,
//! `crates/gt-core/src/prop/dynamic.rs`.
//! **Tests:** `crates/gt-core/tests/u33_convert.rs`.
//!
//! **Fills:** `ConvertFrom` for the real lattice (`value_convert.hh:73-200`),
//! `ToAny`/`FromAny` for all 15 members, and a `DynProp` adaptor over `DenseProp`.
//!
//! **Acceptance:**
//! * The 15 diagonal cases are the identity and allocate nothing beyond the value
//!   itself.
//! * Every off-diagonal pair either converts correctly or returns
//!   `Err(NoConversion)` — never panics, never silently truncates without saying
//!   so. Compare against `value_convert.hh` pair by pair.
//! * `to_any`/`from_any` round-trip every member.
//! * `DynWrap::{get, put}` return `Err`, never `panic`, for an unreadable or
//!   unwritable map (defect: `graph_properties.hh:447, :457` throw).
//! * Instantiation count assertion: `nm` on a release build shows **45**
//!   conversion monomorphisations for the full lattice, not 225 ([DESIGN](crate::design) §10).
//!
//! ---
//!
//! ## U-FINAL — Remove the skeleton allowances
//!
//! **Owns:** `crates/gt-core/src/lib.rs`, `crates/gt-algo/src/lib.rs`,
//! `crates/gt-inference/src/lib.rs`, `crates/gt-io/src/lib.rs`,
//! `crates/gt-py/src/lib.rs`.
//!
//! **Fills:** deletes `#![allow(dead_code, unused_variables)]` from all five
//! roots, and the two pyo3-related allows from gt-py once the workspace moves to a
//! pyo3 release built for edition 2024.
//!
//! **Acceptance:** `cargo clippy --workspace --all-targets --all-features` passes
//! with zero warnings **without** those allows. `grep -rn 'todo!' crates` returns
//! nothing. `grep -rn 'unsafe' crates/gt-core crates/gt-algo crates/gt-inference
//! crates/gt-io` returns only the `forbid` attributes.
//!
//! ---
//!
//! ## Cross-cutting, owned by no single unit
//!
//! These are tracked in [DESIGN](crate::design) §15 and must not be started as part of a unit
//! above without a design amendment:
//!
//! * the inference second dispatch layer (`GEN_DISPATCH`, name-indexed resolved
//!   types) — roughly 13 000 of the ~35 000 C++ leaves;
//! * heterogeneous coupled states (`coupled_state_t`);
//! * the numpy/FFI property-map boundary, which is where any future `unsafe` will
//!   accumulate;
//! * differential testing against a live graph-tool build. **Nothing in the defect
//!   table is settled until the two implementations agree on real graphs.**
