# graph-tool 3.8 → Rust: the architecture of record

This document is authoritative. Where it disagrees with a comment in the code,
the code is wrong. Where a public signature here cannot be made to compile, the
signature was changed and the change is recorded in §13.

Everything asserted below about the C++ was read in
`../src/graph` at the stated line. Everything asserted about the Rust was
compiled with rustc 1.98.1; the layout numbers and the compiler diagnostics in
§9 and §10 are copied from a run, not recalled.

---

## 0. What this port is, and what it is not

graph-tool is a dynamically typed Python frontend over monomorphic C++ kernels.
The C++ recovers the static type at the boundary by a linear `any_cast` scan
over a Hana cartesian product of graph views × property value types, then hands
the resolved types to a template. That shape is not an accident and this port
does not try to remove it: **a dynamically typed frontend still needs the
product.**

What the port changes is the *kind of failure* available at each layer:

| layer | graph-tool | here |
|---|---|---|
| a dispatch axis has no arm | runtime `DispatchNotFound`, message: "This is a graph_tool bug. :-(" (`dispatch.hh:86-88`) | exhaustive `match`; a new view is `error[E0004]` |
| a state supplies a member under the wrong name | silently inherits a *working* default (`spec_base.hh:88-109`) | `error[E0407]` / `error[E0046]` |
| a property map is shorter than the graph | unchecked write (`dispatch.hh:174` → `fast_vector_property_map.hh:218`) | the short view is not a constructible value |
| an undirected view is asked for in-edges | empty range (`graph_adaptor.hh:224-233`) | `error[E0599]` |
| a Python-valued map reaches a parallel loop | happens, with the GIL released (`graph_properties_copy.cc:35-42`) | `error[E0599]`: the type is `!Send` |
| two derived indexes disagree | possible; `check_epos` exists and every call site is commented out (`:1206`, `:1289`, `:1433`) | `validate()` runs after every mutation under `debug_assertions` |

It is **not** a performance rewrite. §12 is an honest ledger, and it contains
three places where this port is slower than the C++ and says so.

---

## 1. Crate layout

```
graph-tool-rs/
  Cargo.toml                     workspace, resolver 3, edition 2024
  crates/
    gt-core        ids · dir · adj · view · prop · par · bound · graph
    gt-algo        traversal · components · degree · centrality · topology
    gt-inference   delta · spec · metropolis · blockmodel
    gt-io          gt · graphml · dot · csv
    gt-py          gil · dispatch · value · module     ← the only crate naming pyo3
  docs/
    DESIGN.md                    this file
    IMPLEMENTATION_PLAN.md       numbered units with disjoint file ownership
```

Five crates, not the eleven the source designs proposed between them. Two
reasons. First, the `value_subset!` macro and the types it names
(`ValueKind`, `DispatchError`, `PropValue`) **must live in the same crate**: a
`macro_rules!` body expanding `$crate::ValueKind` resolves `$crate` to the crate
that *defines the macro*, so splitting them gives `error[E0433]: could not find
ValueKind in crate $crate` at every invocation. Second, crate boundaries are
recompilation boundaries and there are ~21 000 monomorphised leaves to get
through; five is already coarse enough to hurt (§10).

`gt-core` carries two optional features:

* `python` — compiles `prop::value::PyValue`. gt-py enables it. The Python
  member of the value universe lives in gt-core, not gt-py, because
  `PropValue`'s seal is a **private** module: if the trait could be implemented
  downstream, a newtype over a bare `pyo3::Py<PyAny>` could re-enter the
  universe with `Send + Sync` intact and defeat §7 entirely.
* `wide-index` — swaps `ids::Raw` from `u32` to `u64`.

---

## 2. The contested decisions

Six independent designs produced four graph traits (three of them named
`GraphRef`), three view mechanisms, three edge descriptors, five property-map
types, four branding schemes and two index widths. Both judges concluded the
set was not implementable as a merge. These are the resolutions, each with the
evidence that decided it.

### D1 — The graph trait is **not** GAT-based

**Decided:** `trait GraphRef: GraphBase + HasDir` with plain associated types,
implemented on `&'g AdjList` and on view *values*, every method taking `self`
by value.

**Rejected:** `trait Incidence { type OutEdges<'a> where Self: 'a; fn out_edges(&self, v) -> Self::OutEdges<'_> }`.

The GAT form fails three ways, each fatal to a different part of the port:

* `&dyn Incidence` is `error[E0038]: the trait is not dyn compatible because it
  contains generic associated type OutEdges`. graph-tool's whole Python
  boundary is runtime dispatch over graph kinds. It cannot be built on that
  trait.
* `where for<'a> G::OutEdges<'a>: Send` — the bound rayon needs — is accepted
  for an owned graph and rejected for any *borrowing view* with
  `error[E0597]` and rustc's own note that the GAT's implicit `where Self: 'a`
  "implies a `'static` lifetime". So `Und(&g)` could never reach a parallel
  loop.
* The usual escape, a refinement trait
  `trait SizedIncidence: Incidence where for<'a> Self::OutEdges<'a>: ExactSizeIterator`,
  is well-formed and unimplementable for any borrowing wrapper
  (`error[E0477]`). So the filtered-view question would have *no* answer.

Putting the lifetime on the implementing type costs nothing: `Self::Out` is
still `OutEdges<'g>`, `fold` still forwards to `slice::Iter::fold`, and views
are one machine word (verified: `size_of::<Und<&AdjList>>() == 8`).

### D2 — Incidence is **anchored**; identity is `EdgeId` and nothing else

**Decided:** two descriptor types.

* `Incident { other: VertexId, edge: EdgeId }`, yielded by `out_edges`,
  `in_edges`, `all_edges`. `other` is **always the neighbour**.
* `EdgeRef { id, src, tgt }`, yielded by `edges()` and `find_edge`, in
  canonical storage orientation.

Neither implements `PartialEq` or `Hash`. Verified: `a == b` on two `EdgeRef`s
is `error[E0369]`, and `HashSet<EdgeRef>::insert` is `error[E0599]`. Identity
goes through `.id()` or `From`.

**Why anchored.** Two of the six designs got undirected traversal silently
wrong, in opposite directions, and both misread the same C++. The ground truth:

```c++
// graph_adaptor.hh:199-207
out_edges(u, undirected_adaptor<Graph>& g) { return _all_edges_out(u, g.original_graph()); }
// graph_adjacency.hh:1102-1108   — note the iterator type
_all_edges_out(Vertex v, const adj_list<Vertex>& g) {
    typedef typename adj_list<Vertex>::out_edge_iterator ei_t;   // OUT, not ALL
    return {ei_t(v, es.begin()), ei_t(v, es.end())};             // the WHOLE block
}
// graph_adjacency.hh:298-306
struct make_out_edge { static edge_descriptor def(vertex_t src, const pair& v, Iter&&)
    { return edge_descriptor(src, v.first, v.second); } };       // src == u, always
```

So graph-tool's undirected view **normalises**: `src == u` for the in-half too,
and the `std::swap` at `graph_adaptor.hh:161` exists to make `edge(u,v,g)`
*agree* with that. The view is self-consistent; what one design mistook for an
inconsistency was `all_edges(v, g)` on a *directed* `adj_list`, a different
function.

A symmetric `{src, tgt, idx}` POD cannot record "stored 2→1, traversed from 1".
`Incident` does not have to: the field is named `other` and there is no second
interpretation. This also makes the filtering adaptors cheaper — `FilterIncident`
needs only the iterator and the filter, never the graph, because the
neighbour is already in hand.

**Why no `Eq`.** `operator==` on `adj_edge_descriptor` compares `idx` **only**
(`:196`) and `hash` is `e.idx` (`:1618`), so `unordered_set<edge_descriptor>`
dedupes correctly — which undirected algorithms rely on. One design "fixed"
this by deriving equality over all three fields, which silently changes results
for every such set. Refusing to implement `Eq` at all makes the port of that
code a compile error that points at the right line.

### D3 — Views are composed newtypes with **normalising constructors**

**Decided:** `Und<G>` and `Rev<G>`, `#[repr(transparent)]`, **private fields**,
reachable only through `Undirect::undirect` and `Reverse::reverse`, whose impls
make `undirect` idempotent and absorbing and `reverse` an involution. Filtering
is a **type parameter** `Filtered<G, F: Filter>` with a `KeepAll` ZST.

**Rejected:** flattening all six into
`GraphView<'g, const D: bool, const R: bool, const F: bool>`.

Const-folding is real — one judge measured 27 vs 45 instructions for the
unfiltered and filtered arms of the flattened form. But flattening is also
exactly what throws away the orientation information that
`undirected_adaptor::source/target` carries, and the design that shipped it had
four of six views semantically wrong (reversed yielding base-out ∪ base-in
where `graph_reverse.hh:78-80` swaps the iterator typedefs and means base-in
*only*). Composed wrappers plus an anchored `Incident` keep the orientation and
get the same monomorphisation, because `Und<G>::Out = G::All` is resolved at
the type level exactly as a const parameter would be.

Verified, by `std::any::type_name` on a real build:

```
d.reverse().reverse()   = &AdjList                  (involution)
d.reverse().undirect()  = Und<&AdjList>             (absorbing, not Und<Rev<_>>)
d.undirect().undirect() = Und<&AdjList>             (idempotent)
Rev<Und<&AdjList>>      = error[E0271]: type mismatch resolving
                          <Und<&AdjList> as HasDir>::Dir == Directed
```

This is **stronger** than `graph_filtering.hh`. C++ drops the impossible corner
with a `hana::filter` (`:112-118`) and de-duplicates the rest with
`hana::to<set_tag>` (`:127`); the struct bound alone kills only the filtered
case, and the *duplicates* — `Und<Und<_>>`, `Rev<Rev<_>>` and the tower above
them — are what normalising constructors close. Private fields are what make
that closure real: `Undirected<G>(pub G)` would let `u.0.in_edges(v)` compile
and defeat the `Bidirectional` separation entirely.

**Why a type parameter for filtering, not a const generic or a runtime
predicate.** A const generic erases no *data*, so the mask fields would exist
in the unfiltered arm and need dummy values; a runtime `&dyn Fn` gives one
instantiation instead of six — a real compile-time win — at the cost of an
unspeculatable indirect call in the inner loop. Type parameter with a ZST for
"off" gives the same six as C++ with neither cost. (Corroboration: the design
that chose the const-generic form shipped a `vmask` field that was **read
nowhere**, so every filtered measurement it reported was of a no-op.)

### D4 — Directedness is split: `Dir` *and* `HasDir`

`trait Dir { const DIRECTED: bool; const N_FIELDS: usize; }` carries the
type-level directedness **and its field set** — which is what lets an
undirected SBM delta buffer hold two adjacency half-field vectors instead of
four, rather than four empty ones.

`trait HasDir { type Dir: Dir }` is separate so that non-`Copy` **owners** can
carry directedness. `Arc<AdjList>: HasDir` is what makes
`Und<Arc<AdjList>>` — the safe replacement for
`std::reinterpret_pointer_cast<ug_t>(u)` (`graph_filtering.cc:92`) — expressible
at all. Verified: `size_of::<Und<Arc<AdjList>>>() == size_of::<Arc<AdjList>>() == 8`,
the same layout the cast produces, obtained by a move.

`Field<D>` replaces the `InFields for ()` trait one design used, whose `slot()`
was an `unreachable!()` in safe code, unreachable only because every caller
branched on `D::DIRECTED` first — an invariant stated in a comment and nowhere
in a type. That is the same species of defect as the `_dummy` fallthrough it
was meant to fix.

### D5 — Index width is a crate-wide **alias**, not a type parameter

`pub type Raw = u32;` (`u64` under `wide-index`). `Id<T: IdTag>` is a
newtype over it.

The designs disagreed: one made `I: RawId` a parameter defaulted to `u32` and
built its whole cache-line argument on it; one used `usize`; one used `u32`.
The parameter version is worse than either fixed choice, because it multiplies
every generic algorithm in the port by two for a capability nobody has asked
for. Making it an alias removes an entire monomorphisation axis from every
signature and still leaves `u64` reachable by a whole-build flag.

The width itself: verified `size_of::<AdjEntry>() == 8`, against 16 for
`adj_list<size_t>`'s `pair<vertex_t, vertex_t>` — eight entries per 64-byte
line against four, on the memory stream that dominates BFS, triangle counting
and SBM neighbour sweeps. Ceiling 4.29 × 10⁹ vertices.

Note graph-tool already has *two* widths and asserts neither: `_epos` is
`vector<pair<uint32_t,uint32_t>>` (`graph_adjacency.hh:620`) beneath a `size_t`
vertex. Here positions are `Raw` too, so that cannot recur — a claim one design
made in its rationale and contradicted six lines later in its own code.

### D6 — The kernel-facing property map is a **trait**

`trait ReadProp<K: IdTag>` with `type Ref<'s>: Deref<Target = Self::Value>`.
Implementations: `DenseProp` (owns a `Vec`), `PropSlice`/`PropSliceMut`
(views), `IndexProp` (**no storage**), `Unity` (ZST), `Constant`, `ConstI64`.

This is not stylistic. `vertex_index_map_t` and `edge_index_map_t` are
`hana::append`ed to every non-`writable_` property axis:

```c++
// graph_properties.hh:166-180
inline constexpr auto vertex_properties =
    hana::append(get_vprop_types(value_types), hana::type<vertex_index_map_t>());
inline constexpr auto scalar_vertex_properties =
    hana::append(get_vprop_types(scalar_types), hana::type<vertex_index_map_t>());
```

Those axis names appear in roughly three quarters of the 324 dispatch call
sites. `vertex_index_map_t` is a *storage-free identity map*: it has no vector
and no slice. **All five property-map designs proposed by the source lenses
model a map as an owned `Vec<T>` or an `&mut [T]`, so none of them can
represent the most common argument the dispatcher passes.** Only a trait can.

Reads return `Self::Ref<'s>: Deref`, not `Self::Value`. Returning by value
costs a heap allocation, a `memcpy` and two deallocations per element for
`String` and the seven vector members — nine of the fifteen — which is exactly
the inner loop of `graph_properties_copy.hh:62`, where the C++
`pm[k] = src_ref` reuses the target's capacity and allocates nothing.

### D7 — The value universe: no `Default`, no `Send`, no `Sync` on the base trait

```rust
pub trait PropValue: sealed::Sealed + 'static { const KIND: ValueKind; type Elem: PropValue; }
pub trait Zeroed:  PropValue { fn zero() -> Self; }      // the 14 non-Python members
pub trait GilFree: PropValue + Clone + Send + Sync {}    // the 14 again, as a marker
pub trait ToF64:   Copy      { fn to_f64(self) -> f64; }
pub trait Scalar:  PropValue + GilFree + Copy + ToF64 + PartialOrd {}
```

Two source designs wrote
`PropValue: Sealed + Default + Send + Sync + 'static`. Both bounds are wrong:

* **`Default`** is unimplementable for the 15th member. pyo3 has no
  `impl Default for Py<T>`; the default `python::object` is `Py_None`, which
  needs a live interpreter and an incref. So `data.resize_with(n, T::default)`
  — the body of the central sizing chokepoint — cannot exist for it. Growth
  here takes a `FnMut() -> T` closure (`sized_for_with`), which the PyO3
  boundary fills by capturing its token; `sized_for` is the `Zeroed`
  convenience.
* **`Send + Sync`** is precisely what defeats the GIL guarantee, because rayon
  checks `Send`, not a positive marker. With them on the base trait,
  `store.as_mut_slice().par_iter_mut()` over a Python-valued map compiles with
  no `unsafe` and no diagnostic.

`long double` and `vector<long double>` are carried as `LongDouble([u8; 16])`,
opaque, with **no arithmetic**. `.gt` files and `PropertyMap.value_type()`
round-trip unchanged, and the `Scalar` bound excludes the type from every
arithmetic kernel at compile time. Three designs called this a dead end; it is
a one-variant decision, and it is what the subset-bound mechanism is for.

`ToF64` rather than `Into<f64>`: there is **no** `impl From<i64> for f64` (it is
lossy), and `int64_t` is graph-tool's most-used scalar property type. A design
bounded on `Into<f64>` ships a "scalar" axis with `i64` silently missing.

### D8 — Layering: the default-body rule, and the sampler returns its own density

> **A trait method may carry a default body only if that body is either a pure
> no-op or derivable from other *required* methods of the same trait.** A
> default that encodes policy is not a default; it is a required method.

graph-tool's `SpecBase::Imp` (`spec_base.hh:58-130`) `= delete`s three members
and defaults fourteen with *working* bodies, so a misnamed deleted member is a
compile error and a misnamed defaulted one silently inherits. Confirmed live
bug: `potts/spec.hh:113-117` overrides the proposal with
`uniform_int_distribution<group_t> sample(0, _q-1)`, then `:124` defines
`get_move_lprob`. `grep -rn get_move_lprob src/` finds that one line, called
from nowhere. The loop calls `get_move_prob` (`base/mcmc.hh:123-124`), which
Potts does not define, so it inherits `spec_base.hh:88-109` —
`log(1. - d) - safelog_fast(B)`, the density of the *base's* sampler. The
Hastings ratio does not correspond to the kernel being sampled.

**A second instance, one layer up, not previously reported:**
`loops/mcmc_loop.hh:60` declares `constexpr bool is_determinisitic()` —
misspelled — while the loop calls `is_deterministic()` at `:121` and `:196`.
The default is dead. It survives only because every concrete state happens to
define the correct spelling.

The rule alone would make both a compile error. This port goes further:
`GroupProposal::propose` **returns `Proposal { to, log_fwd }`**, so there is no
second place for the forward density to live. Requiring a `log_move_prob`
method merely makes *misnaming* fatal; returning the density from the sampler
means a copy-pasted density in the same `impl` block cannot silently disagree
with a hand-written sampler. (One judge compiled exactly that: a correct
uniform sampler plus a pasted base density, in one impl block, giving a
Hastings ratio of −1.505 where the answer is 0.)

Capabilities are **trait presence**: `Mergeable`, `ParallelMove`,
`Checkpointed`. That deletes the duplicated `has_merge` (`spec_base.hh:50`) /
`_has_merge` (`:115`, and again at `merge_split.hh:99`, `multilevel.hh:114`) —
two independent booleans for one capability, at two scopes, which a derived
spec can set inconsistently.

`MoveScore { d_entropy, log_hastings }` replaces `std::tuple<double,double>`
destructured positionally at `mcmc_loop.hh:174`. `Step { mv, nsteps }` replaces
the `constexpr` return-type probe at `:112-118` — without it, merge-split
(`merge_split.hh:1201-1203` returns `tuple<size_t,size_t>`), multiflip and
multilevel cannot implement the trait at all.

### D9 — Concurrency: a scratch type, row locks, and a builder

Three source designs independently removed graph-tool's concurrent mutation and
**each scored the removal as a win**. It is a removed capability:
`blockmodel/state.hh:152` builds `_group_mutex`, `:343-348` takes ordered pair
row locks, `:451` sizes `_m_entries_pool` to `get_num_threads()`.

Resolution, in three parts:

1. **`SpecCore::Scratch`.** `virtual_move(&self, ...)` cannot be implemented as
   written: `blockmodel/state.hh:304` is
   `get_move_entries(v, r, nr, _m_entries); move_vertex(v, r, nr, _m_entries);`
   — `virtual_move` **mutates** a reusable scratch `EntrySet`, and every one of
   those C++ members is non-const. `&self` plus the `Sync` supertrait leaves
   only three options, all bad: allocate per Metropolis step (on the hottest
   path in the library); use a `thread_local` (reintroducing exactly the
   thread-keyed hidden global that `parallel_rng.hh:56-73` is condemned for,
   and destroying reproducibility); or take `&mut self` (deleting the lock-free
   proposal phase the whole four-phase argument rests on). Threading
   `&mut Self::Scratch` is the fourth, and it makes
   `_m_entries_pool.resize(get_num_threads())` an explicit per-chunk value.
2. **`BlockCommitShared` + `GroupLocks`.** Shared `&self`, `Box<[Mutex<_>]>`
   taken in `(min, max)` order. What Rust adds is not a prohibition but the
   enforcement `partition.hh:78-84` lacks, where `_count[r] += w` sits between
   two `#pragma omp atomic` statements and is correct only if every caller
   holds `_group_mutex[r]`.
3. **`ParBuilder`** for bulk construction, replacing `set_concurrent(true)`
   (`graph_adjacency.hh:451`) + per-thread `_free_idx_m` (`:613`). Edge ids are
   assigned by a deterministic merge in chunk order, so they do not depend on
   the thread count — which graph-tool's per-thread free lists explicitly do.

An "owner-computes" scheme with no locks was proposed and is **provably wrong
for the block graph**: `entries.hh:391-398` writes `_mrs[me]`, `_mrp[r]` *and*
`_mrm[s]` from one delta, so no partition by group makes every counter
single-owner.

`pair_mut` returns `Option<Pair<'_, T>>` with `Two(&mut, &mut)` **in caller
order** and `Same(&mut)`. A version returning `[&mut T; 2]` sorted by index
silently transposes — `pair_mut(xs, 3, 1)` hands back `[&mut xs[1], &mut xs[3]]`
— which is the same class of defect `MoveScore`'s named fields exist to
prevent, and graph-tool does *not* have it (`do_ulock_pair` sorts only for
`std::lock`, and the caller keeps `r` and `s` by name). `Same` exists because
`r == s` is the block-graph self-loop `_mrs[r][r]`, not an error case.

### D10 — The SBM lifecycle: the *level* is the unit of consumption

```
Recording<'w>  --seal()-->  Transition<'w>  --into_levels()-->  Applied<'w> (per level)
                                |                                     |
                                +-- level(l) -> Delta<'a> (Copy)       +-- commit -> Receipt
```

`Transition` and `Applied` are `!Clone`; `commit` takes `Applied` **by value**,
so replaying a level twice is a move error where `apply_delta`
(`entries.hh:429`) has no such notion and double-counts every entry if
re-entered.

**Why per level and not per transition.** Taking the whole `Transition` by value
makes two things impossible that the design claims to support. The always-on
audit cannot be called (after `commit` the value has moved, `error[E0382]`; and
before `commit` the audit fails by construction on every non-zero delta). And
the **nested block model cannot be committed at all**: `apply_delta` must
recurse into the coupled state with `*m_entries._next` (`:488`) — level *l*
goes to state *l* — and there is only one value to give away. `Receipt` carries
the before-image past the commit so the audit compiles.

`Stamp { state: StateId, epoch: Epoch }`, not a bare epoch. A revision counter
does not identify a state: two freshly built states both sit at epoch 0, so a
transition recorded against one commits silently into the other and the guard
never fires. One extra `u64` comparison per commit.

**The ownership inversion is what dissolves the borrow conflict**, and the
before-image is *not*. Three readers are three shared borrows and never
conflicted with each other; the conflict is that the buffer is a member of the
state (`state.hh:2545`), so `this->entries_dS(..., this->_m_entries)` has
`&mut self` aliasing `&mut self._m_entries`. Moving the buffer into a
caller-owned `Workspace` removes it, full stop.

The before-image is kept for a different and honest reason: it removes two
cache-cold dependent loads per entry from the pricing loop — the `_emat` hash
probe and the `_mrs[me]` indirection at `state.hh:1225` — turning `entries_dS`
into a contiguous scan over `&[Entry<W>]`. That is the best single performance
idea in this port, and a judge verified it prices 269/269 random moves exactly
against a from-scratch entropy. It is also **sufficient only for the sparse
model**: the dense branch (`state.hh:1145-1219`) sweeps `out_edges_range(t, _bg)`
and genuinely needs the state. Both signatures exist and say which is which.

### D11 — No branded-index `unsafe`. At all.

Three designs proposed a generative-lifetime branding layer
(`scope(|Domain<'id>| ...)`, `BEdgeId<'id>`, `get_unchecked`) to elide bounds
checks. It is dropped. Three reasons, in order of weight:

1. **The one that shipped was unsound.** Its `brand_edge` was a *safe* function
   taking a provenance-free `Copy` descriptor from any graph, bound-checked
   only under `debug_assert!`, feeding `get_unchecked`. A judge got a SIGSEGV
   out of it from safe code in a release build.
2. **It does not pay on the shape that matters.** Measured: on a contiguous
   scan the branded form saves one `cmp`/`jae` per element (17 → 10
   instructions) with no unrolling and no vectorisation — low single digits,
   not the claimed 2–4×. On the *re-entry* shape that actually dominates graph
   kernels (`_b[source(e, g)]`, an index read out of one map and used in
   another) it was measured at **26 instructions against 17** for plain safe
   indexing: a net pessimisation.
3. **The zero-unsafe path recovers most of it anyway.** `DenseProp::as_slice`
   and `PropSliceMut::as_mut_slice` hand out `&[T]` / `&mut [T]`, and an
   iterator over a slice carries its own length, so LLVM removes the check that
   indexing cannot. No `unsafe`, no brand, no viral `'id` in every signature,
   no `#[pyclass]` problem.

Cross-graph confusion is caught instead by a **runtime `GraphId` comparison at
kernel entry** — one compare per kernel, not per access. `Bound<K>` and
`DenseProp<T, K>` both carry a `GraphId`, and `sized_for` returns
`Err(PropError::WrongGraph)`. That is cheaper than a brand, catches the case
the brand was *actually* needed for (`copy_property`), and costs no ergonomics.

### D12 — `remove_edge` takes an `EdgeId`, and the slot table is unconditional

`EdgeSlot { src, tgt, out_pos, in_pos }` — 16 bytes, verified — so
`endpoints(e)` and `remove_edge(e)` are O(1) and there is **no caller-supplied
orientation left to be wrong**.

This answers, affirmatively, the open question one design left: should the
non-slot configuration ship? No. graph-tool makes `_epos` a runtime flag
(`_keep_epos`, `:617`) and then writes `clear_vertex` **twice**, once per
setting (`:1343-1414` and `:1416-1434`) — and only one of the two is wrong.
With one configuration there is one algorithm.

It also retires the `reverse_edge`/`remove_edge` interaction entirely rather
than guarding it. A design that kept a positions-only slot table and accepted a
descriptor had `remove_edge(e.reversed())` — using its own advertised
replacement for `graph_adjacency.hh:571` — panic via `out_len - 1` wrapping to
`usize::MAX`, with its two configurations disagreeing about whether the input
was even valid.

---

## 3. Storage

`AdjList<H: Lookup = NoLookup>`: `Vec<Block>`, `EdgeIds`, `EdgeSlots`, the
lookup, a `GraphId`, and a reusable `Vec<EdgeId>` scratch.

**`Block` is 32 bytes** (verified), matching graph-tool's
`pair<size_t, vector<pair<size_t,size_t>>>` vertex record exactly, with no
malloc for an isolated vertex (`Vec::new()` does not allocate). An inline
small-vector buffer was measured at **48 bytes** — a 50% larger vertex array,
streamed by every `for v in vertices { for e in out_edges(v) }` kernel — and
was rejected: it helps low-degree vertices, which are not the ones carrying the
edge traffic.

`add_edge` is **O(1)**, by the trick at `graph_adjacency.hh:1192-1215`:

```c++
if (s_pos < s_es.size()) {           // in-list is not empty
    s_es.push_back(s_es[s_pos]);     // push the first in-edge to the back
    s_es[s_pos] = {t, idx};          // overwrite its slot
}
```

One source design transliterated the surrounding code and replaced this with
`es.insert(split, e)`, which is O(in-degree). A judge measured it quadratic:
20.8 / 51.6 / 207.6 / 876.3 ms for 10k / 20k / 40k / 80k insertions on a hub.
On the high-in-degree hubs this library exists to analyse, that is the
difference between a port and a regression.

Mutation is exactly two private primitives, `splice_in` and `splice_out`, and
every relocation is reported as a `Moved` **value** — so the `&mut Block`
borrow ends before the index hooks run, which is what forces the correct
ordering in `splice_out` and what graph-tool has to remember by hand
(`:1243-1249` binds `s_es` and `t_es` as two references that alias whenever
`s == t`).

`num_edges()` is `EdgeIds::live()`, derived, never accumulated by caller
arithmetic. `clear_vertex_where` is one loop over `remove_edge`, filling the
reusable scratch — no per-call `Vec`, which a judge measured at 2–4 allocations
per call in the design that used one.

---

## 4. Views

Six types, closed by construction (D3). `Filtered::new` memoises both honest
counts from `EdgeList::edges` and `VertexList::vertices` using the **same**
predicate the iterators use, so `num_edges() == edges().count()` holds.

Degree-summation cannot define `num_edges` on an undirected view: it
double-counts, and mixing an out-edge predicate with an edge predicate makes
the count and the iteration disagree. (One design reported 3 where the answer
was 1, and 2 vs 4 even with the trivial filter.) That is why `EdgeList` is a
bound on `Filtered::new` and why `edges()` is on the trait at all.

`Filtered::masked` validates the masks against **this graph's own bounds**, so
validation and use are one statement. A free-standing
`MaskFilter::new(vmask, emask, vb, eb)` is `Copy` and remembers nothing: a mask
admitted for a 3-vertex graph is silently accepted by a 10-vertex one and
panics on first probe — the same structural defect as
`graph_filtering.cc:42-46` reserving separately from `MaskFilter::operator()`.

`ExactIncidence` is the refinement unfiltered views implement and filtered ones
do not. `GraphRef` itself does **not** require `ExactSizeIterator`: that bound
excludes filtered graphs (`error[E0277]`) and, under a GAT trait, cannot even
be added as a refinement (`error[E0477]`). This is the resolution of the one
question a design explicitly left open.

There is no `&G → &Und<G>` reference conversion. The `ref-cast` route was
dropped: it needs a dependency, and the value it produces is not a graph unless
`GraphRef` is also implemented for `&Und<AdjList>` — which the design that
proposed it never did, so its only use of its only dependency produced an inert
value. `Und<Arc<AdjList>>` by move covers the `reinterpret_pointer_cast` case
at the same layout.

---

## 5. Property maps and the value universe

`DenseProp::sized_for(bound) -> Result<PropSliceMut, PropError>` is the single
chokepoint. Growth and view-creation are one operation, so an under-sized view
is not a value that exists.

What it replaces:

```c++
// dispatch.hh:171-177      — note: no size argument
static auto& pmap(boost::checked_vector_property_map<Type,IndexMap>& a)
{ if constexpr (dargs.uncheck) return a.get_unchecked(); else return a; }
// fast_vector_property_map.hh:108
unchecked_t& get_unchecked(size_t size = 0) { reserve(size); return reinterpret_cast<unchecked_t&>(*this); }
// :77    reserve only grows, so reserve(0) is a no-op
// :218   unchecked operator[] has no bounds check
```

So every kernel behind `gt_dispatch` receives a raw indexed map whose length was
never checked against the graph, and the only guarantee is nine lines of Python
(`__init__.py:363-373`). That guarantee is already violated: `graph_copy.cc:66-73`
reserves `num_vertices(src)` — the *filtered* count on a filtered view — and then
writes at `index_map[v]`, *unfiltered* indices built at `:222`, while the Python
guard at `__init__.py:3200` also compares filtered counts.

`view()` (shared) is fallible because `&self` cannot grow. Note the consequence,
which one judge caught and is worth stating: graph-tool's checked map
auto-grows on read (`:129`), so `g.vp.x[v]` on a never-written map returns the
default. To preserve that, **the dispatcher calls `sized_for` on read-only maps
too**. The two-phase reserve-then-hand-out does not disappear; it moves from
Python into one place in Rust (`gt-py::module`), where forgetting it is
impossible rather than merely unlikely.

One consequence of that, found while wiring the boundary up and recorded here
because it constrains every future entry point: sizing *cannot* be pushed down
into the kernel. `DenseProp::sized_for` needs `T: Zeroed`, and none of the
`value_subset!` axis traits — `Scalar`, `GilFree`, `PropValue` — implies it,
precisely because `Zeroed` is the property the Python member lacks. So a kernel
generic over its axis has no way to size its own map, and the sizing has to
happen at the dispatched entry point while the concrete member is still in
hand. An entry point that skips it hands its kernel
`PropError::Undersized` on exactly the case this paragraph exists to
preserve — a never-written map.

`Unity` is a true ZST (verified: 0 bytes, against 1 for the C++ empty class)
and implements `ReadProp` **only**. `put(UnityPropertyMap, k, v) {}`
(`graph_properties.hh:714`) is a silent no-op satisfying
`writable_property_map_tag` at any of the 74 call sites; here the same call is
`error[E0599]` (verified).

`Constant`'s field is `c` and it is **public** — which is what
`graph_selectors.hh:109` and `:178` try to read on a class whose member is the
private `_c` (`graph_properties.hh:677`). Both of those C++ overloads are
uninstantiable dead code: graph-tool's constant-weight fast path silently does
not exist.

Conversion uses a crate-local `ConvertFrom`, never `From`. `impl From<Vec<i32>> for Vec<i64>`
and `impl From<f64> for i32` are both `error[E0117]` from *any* crate, and
`value_convert.hh:135` handles vector-to-vector while `prop_map_as` narrows, so
those are the common cases.

---

## 6. Dispatch

`value_subset!` generates, **from one token list**: the member enum, the
accepted-kind array, `narrow`, the kernel trait *with its bound*, and the
exhaustive match that also performs the downcast. Two consequences, both
verified:

```
// a member the kernel's bound cannot handle
value_subset!(pub Bad2, Bad2Kernel, "value", Scalar, { Str => String });
  → error[E0277]: the trait bound `String: Scalar` is not satisfied

// a transposed row
value_subset!(pub Bad, BadKernel, "value", Scalar, { I16 => i32, I32 => i16 });
  → error[E0080]: evaluation panicked: value_subset! Bad: variant I16 is
                  mapped to i32, whose KIND differs
```

The second matters more than it looks. Without the const assertion, `$v => $t`
is an unchecked claim: transposing two rows compiles cleanly and produces a
*runtime* `DispatchError` naming the **wrong** type as offered and listing it as
accepted — strictly worse than `DispatchNotFound`, which at least prints
honestly.

Because the downcast happens inside the generated match,
**`narrow` is the only fallible step** and kernel bodies carry no second
`Result`. Otherwise every one of the ~324 call sites reintroduces
`DispatchNotFound` in its own body.

`AnyGraph::dispatch` is an exhaustive match over six `ViewKind`s with no
catch-all: a seventh is `error[E0004]`.

`DynGraph` is the dyn-compatible face for the Python boundary — internal
iteration, one indirect call per **vertex** rather than per edge. A GAT-based
incidence trait makes this impossible outright (`error[E0038]`), which is the
single strongest argument for D1.

---

## 7. The Python boundary and the GIL

The C++ defect, confirmed in all five copies
(`graph_properties_copy.cc:35-42, :69-76, :104-111, :145-152`;
`graph_properties_copy.hh:36-40`):

```c++
bool is_python = (get_underlying_value_type(tgt) != typeid(boost::python::object) ||
                  get_underlying_value_type(src) != typeid(boost::python::object));
GILRelease gil(!is_python);
bool parallel = (num_vertices(g) > get_openmp_min_thresh() && !is_python);
```

The connective is wrong and the sense is inverted. `object → object` gives
`is_python == false`, so the GIL is **released** *and* the loop **is**
parallelised: `Py_INCREF`/`Py_DECREF` across OpenMP threads with no GIL, in
exactly the case that needs it most. And `int → int` gives the opposite
pessimisation: the GIL held across a long serial loop.

Why the author re-derived it by hand at all is structural:
`gt_dispatch_args::gil_release` (`dispatch.hh:152`) is a single compile-time
constant per call site, while the value type is chosen by a *runtime* `typeid`
probe. One flag cannot say "release iff this leaf has no `python::object`".

Three mechanisms replace it, and it is worth being precise about which one
carries the weight:

1. **Nobody writes the predicate.** `copy_prop`'s where-clause derives it as
   `<S::Mode as Meet<T::Mode>>::Out`, resolved per monomorphised leaf.
2. **A wrongly edited lattice still fails.** `Par`'s bounds (`S: Sync`,
   `T: Send`) are unsatisfiable for the Python member.
3. **`PyValue` is `!Send` by construction** — `PhantomData<*mut ()>`. *This is
   the load-bearing one.* A positive marker trait guards only the functions
   that remember to require it, and `pyo3::Py<T>` is unconditionally
   `Send + Sync` (`pyo3-0.22.6/src/instance.rs:943-944`), with `Clone` and
   `Drop` both doing refcount work without a token. Verified:
   `Vec<PyValue>::par_iter_mut()` is
   `error[E0599]: ... trait bounds were not satisfied`.

And the seal is what keeps (3) from being bypassed by a sibling newtype:
`PropValue`'s `sealed::Sealed` is private to gt-core, and `PyValue` lives in
gt-core, so `struct RawPy(Py<PyAny>)` in gt-py **cannot be a property value at
all**. (One design had exactly that hole: `impl PropValue for RawPy { type Mode = Par; }`
compiled and ran `Py_INCREF` in rayon.)

Also fixed, and the half a mode-lattice alone leaves open: `CopyStrategy::copy`
takes the `Python<'_>` token **as a parameter of the trait method**, so `Par`
cannot be implemented without `allow_threads` and `Seq` cannot run without the
token held. That closes the `int → int` row.

**What is *not* claimed.** `Python::with_gil` inside a rayon closure compiles —
the token is acquired per worker and never crosses a thread boundary. The
guarantee is that the **race** is inexpressible, not that the parallel loop is;
and the price is that a Python-valued map is serialised, which is the correct
behaviour and what the C++ intended before the predicate was inverted.

---

## 8. Determinism

| graph-tool | here |
|---|---|
| `std::exception_ptr eptr{}` declared outside `#pragma omp parallel`, assigned inside by every thread (`parallel_util.hh:438-446`, + 4 copies in `graph_properties_copy.cc`) | `try_det_reduce` returns `Result`; there is no shared slot in the signature |
| `log_sum_exp` accumulated under `#pragma omp critical` (`merge_split.hh:1131-1142`), so association order is the thread interleaving | `Plan` fixes the chunk count and the fold order; bit-identical across thread counts |
| `_rngs[tnum - 1]` (`parallel_rng.hh:56-61`), so results depend on `OMP_NUM_THREADS` | `Seed::split(chunk_index)` |
| `get_rngs` caches in a process-global map keyed on the *address* of the caller's generator, never evicted (`:183-191`) — a freed generator whose address is reused inherits stale streams | no global cache |
| `set_concurrent` assigns edge ids from per-thread free lists, explicitly non-deterministically | `ParBuilder` merges in chunk order |

The cost is stated: a fixed partition forgoes rayon's adaptive splitting, so on
a ragged workload — blockmodel `virtual_move` cost scales with vertex degree —
load balance is worse than OpenMP's `schedule(runtime)` (`parallel_util.hh:405`).

---

## 9. Unsafe budget

**Zero hand-written `unsafe` in the entire workspace.**

| crate | attribute | verified |
|---|---|---|
| gt-core | `#![forbid(unsafe_code)]` | compiles |
| gt-algo | `#![forbid(unsafe_code)]` | compiles |
| gt-inference | `#![forbid(unsafe_code)]` | compiles |
| gt-io | `#![forbid(unsafe_code)]` | compiles |
| gt-py | no `forbid` — pyo3's `#[pyclass]`/`#[pymethods]`/`#[pymodule]` expand to `unsafe extern "C"` FFI glue | `grep -c 'unsafe' crates/gt-py/src` finds only the attribute comment |

`forbid`, not `deny`, so no downstream module can re-allow it.

Deliberately **not** spent:

* Branded `get_unchecked` on property maps — see D11. Unsound in the one design
  that shipped it, and a net pessimisation on the re-entry access pattern.
* `transmute` for the view types — there is no cast to make; views are values.
* `unsafe impl Send/Sync` for the id types. One design wrote four of these with
  the justification that `PhantomData<fn(&'id ()) -> &'id ()>` "does not give
  the auto-derive in every position". That is false, and the cost of writing
  them anyway is that the auto-trait check is permanently suppressed: the same
  struct with a `*mut u8` field would still be `Send`.
* Atomic counters on the block-graph aggregates — `RowLocks` is off the hot
  path by construction and a `lock xadd` is ~20 cycles against 1.

What is *not* under our control: `rayon`, `rustc-hash`, `rand_chacha` and
`pyo3` contain `unsafe` internally. These are widely audited dependencies, not
code we write.

---

## 10. Instantiation budget

Methodology for the C++ figures: a census over all **324** `run_action` /
`gt_dispatch` call sites in `src/` (paren-balanced argument extraction,
`hana::concat` treated as a sum, `hana::append` as +1, the implicit 6-view axis
added for `run_action` sites naming no view range) gives **22 233 direct leaf
instantiations**, median 12 per site. Largest: `graph_properties_map_values*.cc`
at 1 440 each, `graph_properties_copy_imp1.cc` at 1 080,
`prop_map_as` (`graph_properties.cc:69`) at 30 × 30 = 900. The commonly quoted
~35 440 additionally counts the inference module's *second* dispatch layer
(`GEN_DISPATCH`, `graph_state.hh:384-410`). 275 `.cc` files exist under `src/`.

| axis | graph-tool | here | note |
|---|---:|---:|---|
| graph views | 6 | 6 | parity; both exclude undirected∧reversed |
| full value universe | 15 | 15 | parity |
| scalar subset | 6 | 5 | `long double` is storage-only (D7) |
| integer subset | 4 | 4 | parity |
| vertex-prop axis (values + index map) | 16 | 16 | `IndexProp` is the 16th |
| index width | 1 (`size_t`) | **1** | would be 2 if `I` were a parameter (D5) |
| slot table | 1 emitted, 2 code paths | **1** | `Epos` always on (D12) |
| `(s,t)` lookup | 1 emitted, runtime flag | **1** default | `EHash` arms emitted only for actions that declare edge lookup |
| conversion lattice (`prop_map_as`) | **900** | **45** | 15 identity + 15 `to_any` + 15 `from_any` (D6, §5) |
| dependent axes (`last_type_func_t`, 7 families) | 1 recursion level + leaf each | **0** | `<V as PropValue>::Elem` is a projection, not an axis |

**Estimated direct leaves here: ≈ 21 000.** That is 22 233 − ~855 (conversion
lattice) − ~400 (dependent-axis families in the group/ungroup/copy code), i.e.
**parity within 5%, not a win.** Anyone claiming a large monomorphisation
saving from Rust here is mistaken; the product is a property of the problem.

Where Rust is genuinely worse: graph-tool splits instantiations across 277
translation units and can use `extern template`. Rust has neither —
`-Z share-generics` is nightly-only. Five crates is far coarser parallelism
than 277 TUs. Mitigations actually applied:

* `codegen-units = 256` and `opt-level = 1` in `[profile.dev]`; ~21 000 leaves
  means parallelism matters more than per-unit quality in a debug build.
* `[profile.release] codegen-units = 1, lto = "fat"` — the opposite trade, and
  correct there because the hot paths cross crate boundaries by construction
  (gt-algo kernels generic over gt-core's `GraphRef`, gt-inference pricing over
  gt-core's `ReadProp`). Without cross-crate inlining every `Incident`
  manufacture and every `IS_UNITY` const-fold is lost at the crate edge.
* `panic = "abort"` in release: no kernel in gt-core/gt-algo/gt-inference is
  exception-safe by design (nor are their C++ equivalents), and unwind tables
  across 21 000 leaves are pure size. The PyO3 boundary converts `Result` into
  `PyErr`; it never relies on `catch_unwind`.
* `[profile.dev] overflow-checks = true` — this catches the `_n_edges -= 2`
  class directly.

Where Rust is genuinely better on build time: there is no preprocessor.
graph-tool parses ~4.8M lines per build (48.5× amplification across those 277
TUs); `gt-core` is parsed once. The net is a front-end win and a back-end loss,
and it should be *measured* on one fat module before the crate split is
considered final.

**Not designed yet:** the inference module's second dispatch layer
(`GEN_DISPATCH` + `all_types_func_wrap`, `graph_state.hh:203-256`), which
indexes resolved types **by parameter name** rather than positionally. The
positional `last_type_func_t` case is solved by `PropValue::Elem`; the
name-indexed case needs a generated struct of associated types and is roughly
13 000 of the ~35 000 figure. This is an open item, not a solved one.

---

## 11. Verified layout

Measured on a real build (`size_of`), for the default `Raw = u32`:

| type | bytes | graph-tool counterpart | bytes |
|---|---:|---|---:|
| `AdjEntry` | **8** | `pair<vertex_t, vertex_t>` | 16 |
| `Block` | **32** | `pair<size_t, vector<...>>` | 32 (+ a malloc per non-empty vertex) |
| `EdgeSlot` | **16** | `pair<uint32_t,uint32_t>` in `_epos` | 8 (positions only, no endpoints) |
| `Incident` | **8** | — | — |
| `EdgeRef` | **12** | `adj_edge_descriptor` | 24 |
| `&AdjList` / `Und<&AdjList>` / `Rev<&AdjList>` | **8 / 8 / 8** | view refs | 8 |
| `Und<Arc<AdjList>>` | **8** | `shared_ptr<ug_t>` from the cast | 8 |
| `Filtered<&AdjList, MaskFilter>` | **56** | `filt_graph` (ref + 5 predicates) | larger |
| `Unity<f64, VertexTag>` | **0** | `UnityPropertyMap` (empty class) | 1 |
| `Option<Group>` | **4** | `int64_t` + `null_group` sentinel | 8 |
| `Option<VertexId>` | 8 | `size_t` + sentinel | 8 |

`Und`, `Rev` and `Filtered` are `Copy`; every view method takes `self` by
value, so the adaptor chain collapses to one pointer after inlining.

---

## 12. Performance ledger, honestly

### Wins, measured or structural

Numbers in this section were taken on one machine and are labelled with what
was measured and how, because a ledger entry without a method is a memory.
Unless another profile is named, "measured" means `[profile.bench]` (which
inherits `release`: `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`) on an
Intel Xeon Gold 6238R (2.20 GHz, 1 MiB L2/core, 38.5 MiB L3/socket), one
thread pinned with `taskset`, criterion at 20–30 samples, rustc 1.98.1,
baseline `x86-64` (no `target-cpu=native`: every packed add below is SSE2
`addpd`, two lanes, not AVX's four). The box carried a load average of ~33 of
112 cores. Every number below is therefore single-threaded; `det_reduce`,
`pagerank` and `ParBuilder` have benches but **no numbers are recorded for
them here**, because a rayon result taken on a box with 33 cores already busy
measures the box. That remains the standing open item it has been since the
builder's throughput ladder was first attempted.

To re-take any of it:

```
cargo bench -p gt-core      --bench adjacency   # entry_width, add_edge/scaling
cargo bench -p gt-core      --bench reduce      # chunked_sum/lanes
cargo bench -p gt-algo      --bench weights     # Unity / Constant / DenseProp
cargo bench -p gt-algo      --bench kernels     # bfs, components, shortest_distances
cargo bench -p gt-inference --bench sbm         # record, sparse_ds, move_vertex
cargo test  -p gt-core      --test ledger_layout -- --nocapture   # section 11
cargo test  -p gt-inference --test ledger_sbm                     # allocation counts
objdump -d --disassemble=gtprobe_<name> target/release/deps/<bench>-*
```

The `gtprobe_*` symbols are `#[inline(never)] #[unsafe(no_mangle)]` wrappers
that exist only so that `objdump` has something to name; they live in the
bench files beside the group they justify.

* **2× on the adjacency stream, as a footprint. 1.22× as a time.**
  `AdjEntry` is 8 bytes against graph-tool's 16 -- `graph.hh:137` fixes
  `vertex_t = size_t`, so `edge_list_t` (`graph_adjacency.hh:224`) has a
  16-byte element -- and `tests/ledger_layout.rs` re-derives that ratio from
  `size_of::<usize>()` on every run rather than restating the 2.

  The *time* ratio was not measured until now, and it is not 2. `benches/
  adjacency.rs::entry_width` scans two `Vec<Vec<_>>` mirrors of the same
  graph, same degree distribution, same loop, differing only in whether an
  entry is `(u32,u32)` or `(u64,u64)`:

  | working set | 8 B | 16 B | ratio |
  |---|---:|---:|---:|
  | 0.8 M entries (6.4 / 12.8 MB, in L3) | 884.8 µs | 1.083 ms | **1.22×** |
  | 16 M entries (128 / 256 MB, out of L3) | 57.67 ms | 70.43 ms | **1.22×** |
  | 16 M entries, one flat run, no per-vertex block | 13.85 ms | 26.19 ms | **1.89×** |

  The flat row is the ceiling and it reaches the claimed 2×. The blocked rows
  do not, at either working set, because a per-vertex allocation is a pointer
  chase per vertex and neither width comes near memory bandwidth there (2.22
  and 3.63 GB/s, against 9.24 and 9.78 GB/s on the flat run). **The entry width
  is worth 1.22× on the adjacency as it is actually laid out**; the remaining
  0.6× is available only to a CSR-flattened adjacency, which this port does
  not have. Corrected from "2× on the adjacency stream", which quoted the
  footprint ratio as though it were the throughput ratio.

* **The before-image.** Two cache-cold dependent loads per entry removed from
  `entries_dS`, the hottest kernel in the library (D10). Confirmed at the
  instruction level rather than argued: `state.hh:1226-1231` does
  `_emat.get_me(t, w)` and then `_mrs[me]`, two dependent loads per entry,
  while `objdump --disassemble=gtprobe_sparse_ds` on `benches/sbm.rs` shows
  the port's loop reading `0x8(%r12,%rbx,1)` and `0x10(%r12,%rbx,1)` --
  `mrs_before` and `delta`, both out of the entry itself -- and advancing with
  `add $0x20,%rbx`, one contiguous 32-byte stride. The price paid for it is
  the entry width: `Entry<W>` is 32 bytes because it interns `me` and
  `mrs_before`.

  Measured (`benches/sbm.rs::sbm/sparse_ds`, deltas of 4/17/24/32 entries,
  66.98 / 180.55 / 240.44 / 310.42 ns): a least-squares fit is **8.69 ns per
  entry plus 32.4 ns fixed** -- the fixed part being the two `vterm` pairs at
  `r` and `nr` -- and all four points sit within 0.3% of that line. A shape
  that re-derived `mrs` per entry could not be linear in the entry count at a
  flat per-entry cost.

* **`Block` at 32 bytes with no per-vertex malloc** for isolated vertices --
  **which is parity with graph-tool, not a win over it.** Its vertex record
  (`graph_adjacency.hh:225`, `pair<size_t, edge_list_t>`) is 8 + 24 = 32 too,
  and `add_vertex` (`:1318-1336`) `emplace_back`s a default-constructed
  `vector`, which does not allocate either. Both halves are parity. Listed
  here because it is a property worth *keeping* -- an inline small-vector
  buffer was measured at 48 bytes and rejected -- not because it is an
  advantage. `tests/ledger_layout.rs::block_is_one_vec_plus_one_word_and_an_empty_one_does_not_allocate`
  pins it.
* **Monomorphised `Dir` / `Filter` / `Lookup`** delete the runtime flag tests
  C++ performs inside `add_edge` (`:1195`, `:1213`), `remove_edge` (`:1251`)
  and `reverse_edge` (`:576`). Visible in the `entries_dS` disassembly above:
  `eterm::<D>` branches on `D::DIRECTED || r != s`, so at the `Directed`
  instantiation the `r` and `s` fields of `Entry` are **never loaded at all**
  -- the 694-instruction body of `sparse_terms::<Directed, i64>` touches only
  offsets `0x8` and `0x10` of each 32-byte entry.

* **The `Unity` weight map costs nothing, measured three ways.**
  `weighted_out_degree` (`gt-algo/src/degree.rs:122-136`) selects one of three
  arms on `W::IS_UNITY` / `W::IS_CONSTANT`, and section 11's `Unity` row (0
  bytes) is only half the claim. `benches/weights.rs` carries three
  `#[inline(never)]` probes at the three monomorphisations, so
  `objdump --disassemble=gtprobe_*_out_degree` settles it in the shipped
  profile rather than in a test binary:

  | arm | instructions | backward branch | loads from the map |
  |---|---:|---|---|
  | `Unity` | **14** (10 on the empty-block path) | none | **none, at any address** |
  | `Constant` | 34 | none | one `mulsd (%rdx)`, hoisted out of nothing |
  | `DenseProp<f64>` | 29, of which a **7-instruction loop** | yes | one `addsd (%rcx,%rdi,8)` per incident edge |

  "Folds to a couple of instructions" was the standing informal claim; 14 is
  the measured number. Static size is the wrong axis anyway -- `Constant` is
  the *largest* of the three and the second cheapest -- so what the table is
  for is the other two columns: whether there is a loop over the
  neighbourhood, and whether the map is read at all. Wall clock, 100 000
  vertices / 500 000 edges:

  | sweep | `Unity` | `DenseProp<f64>` | ratio |
  |---|---:|---:|---:|
  | `sum_v weighted_out_degree` (500 k incidences) | 130.4 µs | 2.242 ms | **17.2×** |
  | `sum_v weighted_degree` (1 M incidences) | 125.2 µs | 4.424 ms | **35.3×** |

  The unity arm does not scale with the edge count because it does not read
  edges: both rows are ~1.3 ns per *vertex*. `Constant` is 147.5 µs, i.e.
  1.13× the unity arm and 15× cheaper than the general one, which is the
  `IS_CONSTANT` arm earning its place. The `DenseProp` inner loop retains its
  bounds check and a reachable `panic_bounds_check`, as D11 accepts.
* **No per-dereference `reinterpret_cast`.** `make_in_or_out_edge`
  (`graph_adjacency.hh:328-341`) casts the CRTP base to the derived iterator on
  every dereference just to read `_pos`. An anchored `Incident` needs neither.
* **One `TypeId` compare at the boundary**, against up to three `any_cast`
  probes per candidate plus a `try`/`catch` on the Python path.
* **One `Vec` allocation per property map**, against `shared_ptr<vector<T>>`'s
  two plus atomic refcount traffic on every copy into a parallel region.

### Losses, stated

* **No floating-point reassociation.** `#pragma omp parallel for reduction(+:S)`
  (`potts/spec.hh:133, :143`) *licenses* GCC to vectorise the accumulation.
  Rust's `f64 +=` does not, and measurement confirms it: a serial scan emits
  `addsd`, never `addpd`. **Five of the six source designs reported
  entropy-sum performance as parity-or-better without mentioning this.**
  `par::reduce::chunked_sum::<LANES>` restores instruction-level parallelism,
  and on the current toolchain the SLP vectoriser also packs the lanes --
  without any reassociation licence, because `LANES` independent accumulators
  fed from consecutive slots are a packed add *at the same association order*.
  Corrected from an earlier reading of this paragraph, which claimed a
  four-accumulator hand-unroll emits only `addsd`; it does not.

  **The instruction mix, in two profiles, because it differs between them.**
  The numbers this paragraph used to carry -- "the scan is 9 `addsd` / 0
  `addpd`, and `chunked_sum::<4>` is 18 `addpd`" -- were taken with a
  standalone `rustc -O` probe, and they reproduce there exactly. They are not
  the shipped configuration. `benches/reduce.rs` carries `#[inline(never)]`
  probes so the same two forms can be disassembled out of `[profile.bench]`:

  | form | `rustc -O`, standalone | `[profile.bench]` (opt-level 3, fat LTO, CGU 1) |
  |---|---|---|
  | serial `f64` fold | 37 insns, 9 `addsd`, **0 `addpd`** | 37 insns, 9 `addsd`, **0 `addpd`** |
  | `chunked_sum::<4>` | 117 insns, 13 `addsd`, **18 `addpd`** | 69 insns, 7 `addsd`, **6 `addpd`** |
  | `chunked_sum::<8>` | -- | 85 insns, 15 `addsd`, **4 `addpd`** |

  The claim that matters is identical in both columns and is the one that is
  structurally guaranteed: the scan never packs, the lane form does. The
  `addpd` *count* is an unroll-factor artefact and should not have been stated
  as a constant. Every `addpd` here is SSE2, two lanes; the port ships
  baseline `x86-64`, so `-C target-cpu=native` would change these numbers
  again.

  **What the licence is worth, which this paragraph never said.** 1 Mi `f64`
  (8 MB), `benches/reduce.rs::chunked_sum/lanes`:

  | form | time | vs. the strict fold |
  |---|---:|---:|
  | serial `f64` fold | 1.247 ms | 1.00× |
  | `chunked_sum::<1>` | 1.246 ms | 1.00× (it *is* the fold) |
  | `chunked_sum::<2>` | 625.2 µs | 1.99× |
  | `chunked_sum::<4>` | 345.8 µs | **3.61×** |
  | `chunked_sum::<8>` | 333.6 µs | 3.74× |

  So a strict left fold costs **3.6×** against this port's own deterministic
  lane form, and that is the size of the thing `reduction(+:S)` licenses.
  `chunked_sum` recovers essentially all of it at four lanes (3.61 of a 3.74
  ceiling). The reason it can is that the strict fold is latency-bound rather
  than bandwidth-bound: 1.19 ns per element is 6.72 GB/s, against the
  9.24-9.78 GB/s this same machine sustained on the flat adjacency scan two
  entries above, so the `addsd` dependency chain is the binding constraint
  and independent accumulators go straight at it. The loss that remains is
  the licence itself,
  at every site that still spells a reduction as `f64 +=`: graph-tool may
  reassociate a loop this port may not, so each such site pays up to 3.6×
  until it is converted. Explicit lane control still waits for `core::simd`.

  The claim is no longer taken on trust:
  `tests/u11_reduce.rs::the_ledger_records_whether_chunked_sum_vectorises`
  disassembles both forms on every run, asserts only what is structurally
  guaranteed (a serial `f64` fold can never pack), and writes what it found
  to `target/debug/deps/u11_codegen_ledger.txt`.
* **Bounds checks on scatter.** `pm[target(e)] += x` keeps a compare per
  element that `unchecked_vector_property_map::operator[]` does not. Accepted
  per D11; `as_mut_slice()` is the escape for anything expressible as a scan.

* **One allocation per SBM commit, which nothing else in the move cycle
  pays.** `Receipt` owns `entries: Vec<Entry<W>>` (`delta/lifecycle.rs:203`)
  so that the before-image survives into `audit_commit`, and building it is a
  fresh heap object on every move. Measured with a counting `GlobalAlloc`
  (`gt-inference/tests/ledger_sbm.rs`), over 10 000 moves after warm-up:

  | stage | allocations per move |
  |---|---:|
  | `record` (`modify_entries` + `get_move_entries`) | **0** |
  | `sparse_ds` (`entries_dS`) | **0** |
  | `commit` | **1** |
  | `reseat` | **0** |

  `DeltaBuf::begin` (`delta/buf.rs:149`) really does drain and clear rather
  than free, and the entry vector reaches its high-water mark and stays
  there -- the recorder allocated nothing at all across 10 000 moves once
  warm. The commit's one is structural, not an oversight: it is what D10's
  "the level is the unit of consumption" costs when the audit has to run
  after the state has already moved. It is a malloc/free pair against a
  1.97 µs move, so it is small; it is recorded because the surrounding
  design reads as though the cycle were allocation-free, and it is not.
  The whole cycle measures 1.97 µs per move at |V| = 20 000, |E| = 100 000,
  B = 64 (`benches/sbm.rs::sbm/move_vertex`), of which `record` is 1.31 µs.

* **`add_edge` is O(1) amortised, but its constant grows with the graph.**
  Defect #18's detector (`benches/adjacency.rs::add_edge/scaling`) exists to
  catch a `Vec::insert` transliteration, which would double the per-edge time
  every rung. It does not double -- but it does not stay flat either, which
  the bench's own comment used to claim:

  | edges (vertices = edges/4) | 10 k | 20 k | 40 k | 80 k | 160 k | 320 k |
  |---|---:|---:|---:|---:|---:|---:|
  | ns per edge | 53.8 | 57.0 | 80.0 | 89.8 | 107.9 | 187.1 |

  3.5× over five doublings, against 32× for a quadratic insert and 1× for a
  flat append: the detector's purpose is intact and its stated acceptance
  criterion was wrong. The shape is what growing |V| independent `Vec`s under
  random access costs -- reallocation plus a cache miss per touched block --
  and the port has no way to avoid it today, because there is no
  `reserve_edges` on `AdjList` (the open item U6 recorded against D9: the
  builder can count degrees in parallel but cannot size the blocks, since
  `AdjList`'s fields are private to `adj/list.rs`). Sizing every block once
  is the fix and it is unwritten.
* **`swap_remove_vertex` is 2–4× slower.** C++ patches endpoints in place
  (`:1489-1523`); here it is unlink+relink per incident edge, which is the only
  way to make the stale-index class impossible rather than merely absent today.
* **`Filtered::new` costs an O(V + E) prepass.** C++ gets O(1) by returning the
  wrong number.
* **Fixed chunking loses to `schedule(runtime)`** on ragged workloads (§8).
* **Coarse recompilation.** Five crates against 277 TUs (§10).
* **Adjacency order changes after removals.** Swap-with-back against
  erase-and-shift (`:1257-1263`). `gt-io` therefore emits in `EdgeId` order,
  not adjacency order, so round-trips stay byte-reproducible.

### Withdrawn claims

* Branded-index `unsafe` "4× unroll / 2–4× on gather-bound sums": measured at
  17 → 10 instructions on a scan and **a regression** on re-entry (D11).
* Const-generic filtered-view measurements from one source design: taken
  against a `vmask` that was never read.
* "Zero allocation, better than C++" for `clear_vertex`: only true with the
  reusable scratch, which is why `AdjList` carries one.
* **"2× on the adjacency stream" as a throughput number.** The footprint
  ratio is exactly 2 and survives; the time ratio is 1.22× on the blocked
  adjacency this port actually has, and reaches 1.89× only on a flat
  contiguous run. Measured both ways above.
* **`Block`'s "no per-vertex malloc" as a win.** It is parity: graph-tool's
  vertex record is 32 bytes too and its `add_vertex` does not allocate for an
  isolated vertex either. Reclassified above, not deleted -- it is still a
  property this port has to keep.
* **`chunked_sum::<4>` "is 18 `addpd`" as a constant.** True of the
  standalone `rustc -O` probe it was taken from, and 6 in the shipped
  `[profile.bench]` build. The unroll factor is not a design property; that
  the serial fold emits **zero** packed adds in both is.

---

## 13. Changes forced by the compiler

Every item here is a signature that could not be written as designed.

1. **`Fill` with an associated context was dropped.** A blanket
   `impl<T: FillFree> Fill for T` overlaps a manual `impl Fill for PyValue`,
   and Rust cannot prove `PyValue: !FillFree`. Replaced by `Zeroed` (context-free,
   14 members) plus `sized_for_with(bound, fill: impl FnMut() -> T)` for the
   Python member. Strictly more general.
2. **`InFields` with `unreachable!()` was replaced by `Field<D>`** and a
   `Vec<Vec<u32>>` field table of length `D::N_FIELDS`. No unreachable arm.
3. **`Slots`/`Scan` removed as a type parameter** (D12); `EdgeSlots` is a
   field, and it carries endpoints.
4. **`I: RawId` removed as a type parameter**; `ids::Raw` is a crate alias (D5).
5. **`MaskFilter::new` removed.** Only `Filtered::masked` constructs one, so
   validation and use are one statement (§4).
6. **`ExactIncidence`** carries its `ExactSizeIterator` requirement as a
   `where` clause on the trait declaration; as a supertrait bound on
   `GraphRef` it excludes filtered views.
7. **`value_subset!` lives in gt-core**, with the types it names. Splitting
   macro and types across crates gives `error[E0433]` on every `$crate::` path.
8. **`ParBuilder::fill(make: F)`** — `gen` is a reserved keyword in edition
   2024.
9. **rand 0.9, not 0.8.** `rng.gen()` needs `r#gen` under edition 2024.
10. **`[profile.bench] panic` removed** — cargo ignores it and warns.
11. **`DynGraph`'s blanket impl needs a `GraphBaseExt` helper trait**, because
    `self.num_vertices()` inside it resolves to the trait method being defined
    and recurses.
12. **gt-py carries `#![allow(unsafe_op_in_unsafe_fn)]` and a module-scoped
    `#![allow(clippy::useless_conversion)]`.** pyo3 0.22's macro expansion
    predates edition 2024. Both are removed when the workspace moves to a pyo3
    release built for it.
13. **`PyValue` lives in gt-core behind `feature = "python"`**, not in gt-py,
    so that `PropValue`'s seal cannot be re-opened (§7).
14. **`Scalar` excludes `long double`**; `LongDouble` is opaque (D7).
15. **`ToF64` replaces `Into<f64>`** — no `From<i64> for f64` exists (D7).
16. **`ref-cast` dropped**, with the `&G → &Und<G>` conversion (§4).
17. **`SlotRef` is 8 bytes**, not a packed `u32`. The `field:2 | other:30`
    packing caps the group count at 2³⁰ — a capability the C++ has — to save
    four bytes in a loop that runs once per cleared slot.
18. **`AdjList` carries a reusable `Vec<EdgeId>` scratch**, rather than a
    per-call `Vec` or an inline small-vector that spills above degree 16.
19. **`pair_mut` returns `Option<Pair<'_, T>>`** in caller order, with a `Same`
    variant (D9).
20. **`Rng` is a method-level type parameter**, not a fixed concrete type.

---

## 14. Defect table

`graph-tool defect → Rust mechanism`. "Eliminated" means the failure has no
representation, and the cited error was produced by the compiler.

| # | defect (file:line) | status | mechanism |
|---:|---|---|---|
| 1 | `clear_vertex` counts `remove_if`'s moved-from tail, decrementing `_n_edges` by 2 for one removed edge (`graph_adjacency.hh:1403-1410`) | **eliminated** | `num_edges()` is `EdgeIds::live()`, mutated only in `alloc`/`release`; `clear_vertex_where` is one loop over `remove_edge` |
| 2 | two divergent `clear_vertex` bodies, one wrong (`:1343-1414` vs `:1416-1434`) | **eliminated** | one configuration (D12), therefore one algorithm |
| 3 | `remove_vertex_fast` leaves `_ehash` keyed on a dead vertex for in-neighbours (`:1471-1535`) | **eliminated** | `Lookup` keyed on the ordered pair; relabelling is unlink+relink through the two splices |
| 4 | `_dummy` fallthrough aliases two out-of-plane pairs into one entry (`entries.hh:108-119, :223`) | **eliminated** | `Field<D>` is total; `touch_dyn` returns `Err(OutOfPlane)`; no dummy cell exists |
| 5 | `_epos` is `uint32_t` beneath a `size_t` vertex (`:620`) | **eliminated** | positions are `Raw`; `EdgeSlot` static-asserted at `4 * size_of::<Raw>()` |
| 6 | `get_unchecked(size = 0)` hands kernels unvalidated maps (`dispatch.hh:174`) | **eliminated** | `sized_for(bound)` fuses growth and view; no other constructor |
| 7 | `copy_property` sizes from a *filtered* count and writes at *unfiltered* indices (`graph_copy.cc:66-73`) | **eliminated** | `Bound` is minted only by the unfiltered graph, `Bound::new` is `pub(crate)` → `error[E0624]` |
| 8 | a map from graph A indexed by graph B's descriptors | **eliminated** (runtime) | `GraphId` compared once at `sized_for`; `PropError::WrongGraph` |
| 9 | `reinterpret_pointer_cast` to an object never constructed as that type (`graph_filtering.cc:78, :92`) | **eliminated** | `Und<Arc<AdjList>>` / `Rev<Arc<AdjList>>` by move, same layout (verified 8 = 8) |
| 10 | `reinterpret_cast<Graph&>(*this)` out of a private base (`graph_adaptor.hh:46`, `graph_reverse.hh:47`, `fast_vector_property_map.hh:111`) | **eliminated** | no inheritance, no casts, `#![forbid(unsafe_code)]` |
| 11 | `unchecked_vector_property_map::operator[]` reads out of bounds (`:218-221`) | **eliminated** | `ReadProp::get_ref` is checked; `as_slice()` for elision |
| 12 | `in_edges` on an undirected view returns an empty range (`graph_adaptor.hh:224-233`) | **eliminated** | no `Bidirectional for Und<_>` → `error[E0599]` (verified) |
| 13 | `edge(u,v,undirected)` swaps endpoints, disagreeing with iteration (`:161`) | **eliminated** | one rule: `out_edges` yields anchored `Incident`, `find_edge` yields canonical `EdgeRef`; different questions, different types |
| 14 | undirected traversal reads the wrong endpoint | **eliminated** | `Incident.other` is the neighbour by construction (D2) |
| 15 | `num_vertices` on a filtered view is the unfiltered count (`graph_filtered.hh:316`) | **eliminated** | `num_vertices()` honest, `vertex_bound()` separate type |
| 16 | `distance(ei, eiend) != num_edges(g)` (`:301-312`) | **eliminated** | both counted from `edges()` with the same predicate |
| 17 | undirected∧reversed corner removed by a value-level `hana::filter` a caller can bypass (`graph_filtering.hh:112-118`) | **eliminated** | `Rev<Und<_>>` → `error[E0271]` (verified); duplicates unnameable too |
| 18 | `add_edge` transliterated as `insert` → quadratic | **avoided** | O(1) push-swap, `graph_adjacency.hh:1192-1215` |
| 19 | Potts scores a uniform proposal with the base's density (`potts/spec.hh:124`) | **eliminated** | `propose` returns `log_fwd`; no second home for the density (D8) |
| 20 | `is_determinisitic` typo makes the default dead (`mcmc_loop.hh:60`) | **eliminated** | one required `schedule() -> Schedule` |
| 21 | five `// required` members with working bodies (`mcmc_loop.hh:39-56`) | **eliminated** | all required; the default-body rule (D8) |
| 22 | `has_merge` / `_has_merge` duplicated at two scopes | **eliminated** | capability = trait presence |
| 23 | `(dS, a)` tuple transposed positionally (`mcmc_loop.hh:174`) | **eliminated** | `MoveScore { d_entropy, log_hastings }` |
| 24 | `(r, s, dir)` hand-swapped on adjacent lines (`mcmc.hh:123-124`) | **eliminated** | defaulted `log_hastings` |
| 25 | multi-step states inexpressible without a return-type probe (`:112-118`) | **eliminated** | `Step { mv, nsteps }` + `steps_per_iter` |
| 26 | shared `std::exception_ptr` written by every thread (`parallel_util.hh:441`, ×5) | **eliminated** | `try_det_reduce -> Result`; no shared slot in the signature, and the surviving error is the lowest chunk's -- `tests/u11_reduce.rs::the_lowest_failing_chunk_supplies_the_error` |
| 27 | FP reduction order is the thread interleaving (`merge_split.hh:1140`) | **eliminated** | `Plan` fixes chunks and fold order; `tests/u11_reduce.rs::log_sum_exp_is_bit_identical_across_thread_counts` |
| 28 | RNG keyed on thread id (`parallel_rng.hh:56-61`) | **eliminated** | `Seed::split(chunk)`; pinned by `tests/u11_reduce.rs::each_chunk_draws_the_stream_its_index_names` |
| 29 | RNG cache keyed on a reusable address, never evicted (`:65-69`, `:73`) | **eliminated** | no global cache; `Seed::split` is pure, pinned by `tests/u10_plan.rs::split_is_a_pure_function` |
| 30 | inverted GIL predicate releases the GIL *and* parallelises `python::object` (`graph_properties_copy.cc:35-42`, ×5) | **eliminated** | `PyValue: !Send` → `error[E0599]` (verified); lattice; token-carrying `CopyStrategy` |
| 31 | GIL held across a long serial `int → int` loop (same lines) | **eliminated** | `Par::copy`'s body *is* `allow_threads` |
| 32 | `GILRelease` is a silent no-op on a thread not already holding the GIL (`gil_release.hh:31-35`) | **eliminated** | `detach`'s `F: Send` excludes the token structurally |
| 33 | `_count[r] += w` non-atomic between two `omp atomic` (`partition.hh:78-84`) | **eliminated** | `RowLocks`; the row is reachable only through its guard |
| 34 | three global mutable caches + `[[gnu::const]]` lying about it (`cache.hh:56, :73, :102, :137`) | **eliminated** | `&Cache`, immutable, `Sync`, threaded explicitly |
| 35 | `apply_delta` double-apply corrupts counts (`entries.hh:429`) | **eliminated** | `Applied` is `!Clone`, consumed by value |
| 36 | a delta applied to the wrong state | **eliminated** | `Stamp { StateId, Epoch }`; an epoch alone does not identify a state |
| 37 | `__test__ = False`: the delta/absolute cross-check is O(E + B²) and off (`base_states.py:33, :44-59`) | **eliminated** | `Receipt` + O(#entries) `audit_commit` on every commit under `debug_assertions` |
| 38 | `EntrySet* _next` points into a pool that may be resized (`state.hh:495-503`) | **eliminated** | `DeltaStack` + `split_at_mut` |
| 39 | `s_es`/`t_es` alias when `s == t` (`:1243-1249`) | **eliminated** | borrowck forbids two `&mut` blocks; `splice_out` is forced to sequence |
| 40 | `null_group = INT64_MAX` checked by hand ~30 times, and `b[u]` is unchecked (`entries.hh:250`) | **eliminated** | `Option<Group>`, `NonZeroU32` niche, 4 bytes (verified) |
| 41 | `edge(s,t,g)` returns a `{max,max,max}` descriptor equal to every other failure (`:943`) | **eliminated** | `find_edge -> Option<EdgeRef>` |
| 42 | `edge_descriptor` equality ignores endpoints; `reverse_edge` mutates in place (`:196`, `:571`) | **eliminated** | no `Eq`/`Hash` on `EdgeRef` (verified `error[E0369]`, `error[E0599]`); `reversed()` returns a value |
| 43 | `put(UnityPropertyMap, ...)` silently discards writes (`graph_properties.hh:714`), 74 call sites | **eliminated** | `Unity` implements `ReadProp` only → `error[E0599]` (verified) |
| 44 | `graph_selectors.hh:109, :178` read `weight.c` on a class whose member is private `_c`; both overloads are dead | **eliminated** | `Constant { pub c: T }` |
| 45 | `is_unity_map<ConvertedPropertyMap<UnityPropertyMap<..>>>` is false, silently losing the fast path | **eliminated** | conversion wrappers forward `IS_UNITY`/`IS_CONSTANT` |
| 46 | `check_epos` exists and every call site is commented out (`:686`, `:1206`, `:1289`, `:1433`) | **eliminated** | `validate()` under `debug_assertions`, checking **both** halves |
| 47 | a subset's member list and its dispatch scan are independent artifacts that can drift | **eliminated** | one token list generates both; transposition is `error[E0080]` (verified) |
| 48 | `DispatchNotFound` cannot distinguish a user type error from a codegen hole (`dispatch.hh:86-88`) | **partial** | views are exhaustive (`error[E0004]`); a *value* subset necessarily ends in `_ => Err`, so a 16th `ValueKind` is a runtime error where no subset lists it. Mitigated by the `every_value_kind_is_reachable_from_some_subset` test |
| 49 | `long double` precision | **partial** | `LongDouble([u8;16])` round-trips `.gt` and the type name; arithmetic unavailable, and the `Scalar` bound says so |
| 50 | `EdgeId` ABA after free-list recycling: an id held across remove+add names a different edge | **still possible** | inherited deliberately; a generation counter would break the 4-words-per-edge layout. `gt-core/id-generations` adds a debug-build side table |
| 51 | concurrent *graph* mutation (`set_concurrent`, `:451`) | **changed** | `&mut AdjList` refuses it; `ParBuilder` replaces the bulk case deterministically. Incremental concurrent `add_edge` is **not** supported |
| 52 | adjacency order after removals | **changed** | swap-with-back, not erase-and-shift. `gt-io` emits in `EdgeId` order so round-trips stay reproducible |
| 53 | the inference second dispatch layer (`GEN_DISPATCH`, name-indexed resolved types, `graph_state.hh:203-256`) | **not designed** | ~13 000 of the ~35 000 leaves. Open item, §10 |

---

## 15. What is still open

1. **The `GEN_DISPATCH` second layer** (§10, row 53). `all_types_func_wrap`
   indexes resolved types **by parameter name**; `PropValue::Elem` solves only
   the positional case. Likely a generated struct of associated types.
2. **Heterogeneous coupled states.** `coupled_state_t` (`state.hh:110`) is a
   `std::variant` over pointers to *differently instantiated* states. A
   homogeneous `Vec<S>` covers the nested SBM only; layered/overlap/weighted
   need an enum (combinatorial) or `Box<dyn CoupledLevel>` (a virtual call per
   level per move — small against O(#entries) work, but not zero).
3. **Whether a 32-bit count type is worth a second `Weight` impl.** `Entry<i64>`
   is 32 bytes; `Entry<i32>` would be 16, i.e. two per cache line in the
   `entries_op` loop.
4. **Vectorising the reductions** (§12) once `core::simd` stabilises, and
   whether it can be done without reintroducing nondeterminism.
5. **Property-map storage across FFI.** C++ hands `std::vector<T>&` to
   Boost.Python directly (`fast_vector_property_map.hh:224`). A numpy view over
   `DenseProp`'s `Vec<T>` reintroduces the aliasing `&mut` was protecting: a
   Python-side array view outliving a `sized_for` resize. This is where the
   remaining `unsafe` would accumulate and it needs a designed answer.
6. **Whether `EHash` should ship at all**, or whether `find_edge`'s
   shorter-half scan is enough in practice. Shipping it doubles the
   `AdjList` monomorphisations reachable from generic kernels.
7. **Measuring one fat module** (`graph_properties.cc`, ~1 090 leaves by the
   census) against its C++ translation unit, before the five-crate split is
   called final.

---

## 16. Testing discipline this port requires

Every source design said "verified by compiling it" and, in most cases, the
compiled artifact contained a defect the design's own claims denied. The
sharpest example: one crate's `ln_gamma` was stubbed to `x.ln()`, no test
asserted a number, and the pricing path was wrong on **194 of 269** moves —
because the `emat` was keyed on an unordered pair while lookup canonicalised to
`(min, max)`. It compiled, it passed its four tests, and it was 72% wrong.

So: *it compiles* is not evidence. Required from the first implementation unit:

* **proptest**, model-based, against a naive reference — for `AdjList`
  (add/remove/clear/swap\_remove sequences, `validate()` after every op, in-half
  cross-checked against the model) and for the SBM (random moves priced against
  a from-scratch entropy with a *real* `lgamma`).
* **trybuild** compile-fail fixtures pinning every negative guarantee in §14.
  The six verified in this document must become fixtures, or they will regress
  silently the first time someone adds a blanket impl.
* **criterion** from day one: the ledger in §12 has numbers in it, and numbers
  rot.
* **codegen assertions** on the hot loops: no `panic_bounds_check`, no
  `__rust_alloc`, in the disassembly of the named kernels.
* **differential testing against graph-tool itself**, on real graphs, with real
  numbers. Nothing in §14 is settled until the two implementations agree.
