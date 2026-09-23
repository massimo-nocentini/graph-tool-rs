//! Differential tests: the stochastic block model, against graph-tool 3.8's
//! `inference/blockmodel/{entropy,partition,entries,state}.hh`.
//!
//! ## 1. The check graph-tool ships disabled
//!
//! `graph_tool/inference/base_states.py:33` is `__test__ = False`, and
//! `mcmc_sweep_wrap` (`:64-89`) only runs when it is true:
//!
//! ```python
//! Si = self.entropy(**entropy_args)          # from scratch
//! ret = func(self, *args, **kwargs)          # the sweep
//! dS = ret[0]                                # the accumulated delta
//! Sf = self.entropy(**entropy_args)          # from scratch again
//! assert math.isclose(dS, (Sf - Si), abs_tol=1e-8), ...
//! ```
//!
//! That is the only thing in graph-tool that ties the O(#entries) incremental
//! pricing to the O(E + B^2) definition, and a user has to call
//! `graph_tool.inference.set_test(True)` to get it. Here
//! [`the_delta_sum_equals_the_absolute_entropy_difference`] is that assertion,
//! in the default `cargo test` run, at graph-tool's own tolerance, over both
//! directednesses and both degree-correction settings.
//!
//! The port already has [`audit_price`](gt_inference::blockmodel::audit_price)
//! and [`audit_commit`](gt_inference::blockmodel::audit_commit), which run on
//! every move --- but they check the delta against *the state's own
//! aggregates*, so a scan that miscounts an edge and a commit that applies
//! that same miscount agree with each other perfectly. Only a recompute from
//! the **graph and the partition** closes that loop, and only
//! `audit_absolute` does that today, behind the `audit-full` feature and
//! reading `BlockView` rather than the graph.
//!
//! The recompute here is deliberately *not* `audit_absolute`. It is written
//! from `entropy.hh:185-206` against `(edges, b)` directly, with its own
//! `ln(n!)` (a plain sum of logs, so it shares no code with
//! [`Cache`](gt_inference::blockmodel::Cache)), and it re-derives `mrs`,
//! `mrp`, `mrm` and `wr` from scratch. Three independent readings therefore
//! have to agree: the recorder's delta, the state's aggregates, and this file.
//!
//! ## 2. Hand-computed terms
//!
//! Section 3 below prices two three-vertex graphs entirely on paper, from
//! `eterm_d` (`entropy.hh:38-59`), `vterm_d` (`:73-87`), `eterm_dense_d`
//! (`:235-258`), `get_edges_dl` (`:293-298`) and `partition::get_dl`
//! (`partition.hh:106-117`). Those are closed forms in `ln 2`, `ln 3` and
//! `ln 6`, which is what makes them checkable without a reference
//! implementation.

use std::collections::BTreeMap;

use gt_core::adj::AdjList;
use gt_core::dir::{Dir, Directed, Undirected};
use gt_core::ids::VertexId;
use gt_core::view::Undirect;

use gt_inference::blockmodel::{
    BlockCommit, BlockState, BlockView, Cache, EntropyParams, audit_commit, audit_price, edges_dl,
    eterm, eterm_dense, partition_dl, record, sparse_ds, vterm,
};
use gt_inference::delta::{MoveKey, Workspace};
use gt_inference::ids::Group;

// ---------------------------------------------------------------------------
// Independent arithmetic
//
// None of this calls into `Cache`. `lgamma_fast(n + 1)` is `ln(n!)` for the
// integer arguments every SBM call site passes (`cache.hh:139-144` tabulates
// `safe_lgamma(y)` at integer `y`), so a plain sum of logs is the definition
// and not an approximation of one.
// ---------------------------------------------------------------------------

/// `ln(n!)`, i.e. `lgamma_fast(n + 1)` at integer `n`.
fn ln_fact(n: u64) -> f64 {
    let mut s = 0.0f64;
    for k in 2..=n {
        s += (k as f64).ln();
    }
    s
}

/// `safelog` (`cache.hh:106-111`): `log(x)`, and zero at zero.
fn safelog(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x.ln() }
}

/// `lbinom_fast` (`support/util.hh:42-47`), guards included.
fn lbinom(n: u64, k: u64) -> f64 {
    if n == 0 || k == 0 || k >= n {
        return 0.0;
    }
    (ln_fact(n) - ln_fact(k)) - ln_fact(n - k)
}

/// `eterm_d` (`entropy.hh:38-59`).
fn eterm_ref(r: usize, s: usize, mrs: u64, directed: bool) -> f64 {
    let val = ln_fact(mrs);
    if directed || r != s {
        -val
    } else {
        -val - (mrs as f64) * std::f64::consts::LN_2
    }
}

/// `vterm_d` (`entropy.hh:73-87`).
fn vterm_ref(mrp: u64, mrm: u64, wr: u64, deg_corr: bool, directed: bool) -> f64 {
    if deg_corr {
        if directed {
            ln_fact(mrp) + ln_fact(mrm)
        } else {
            ln_fact(mrp)
        }
    } else if directed {
        ((mrp + mrm) as f64) * safelog(wr as f64)
    } else {
        (mrp as f64) * safelog(wr as f64)
    }
}

// ---------------------------------------------------------------------------
// The from-scratch model, over (edges, partition)
// ---------------------------------------------------------------------------

/// `_mrs`, `_mrp`, `_mrm` and `_wr`, re-derived from the graph and `_b`.
///
/// The accumulation rule is `update_rs` (`blockmodel/entries.hh:391-398`):
/// `_mrs[me] += d`, `_mrp[r] += d`, and then `_mrm[s] += d` when directed,
/// `_mrp[s] += d` when not --- so an **undirected** pair contributes to
/// `_mrp` twice, and an undirected self-pair (`r == s`) contributes `2 * d` to
/// the one counter. That is also why `_mrp[r] += kout` with an undirected
/// `kout = degree(v)` (`state.hh:224`, `graph_adaptor.hh:315-319`) is
/// consistent: an undirected self-loop is traversed twice.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Aggr {
    /// Keyed by the ordered pair when directed, by `(min, max)` when not.
    mrs: BTreeMap<(usize, usize), u64>,
    mrp: Vec<u64>,
    mrm: Vec<u64>,
    wr: Vec<u64>,
}

fn aggregate(edges: &[(usize, usize)], b: &[u32], slots: usize, directed: bool) -> Aggr {
    let mut a = Aggr {
        mrs: BTreeMap::new(),
        mrp: vec![0; slots],
        mrm: vec![0; slots],
        wr: vec![0; slots],
    };
    for &g in b {
        a.wr[g as usize] += 1;
    }
    for &(u, w) in edges {
        let (r, s) = (b[u] as usize, b[w] as usize);
        let key = if directed || r <= s { (r, s) } else { (s, r) };
        *a.mrs.entry(key).or_insert(0) += 1;
        a.mrp[r] += 1;
        if directed {
            a.mrm[s] += 1;
        } else {
            a.mrp[s] += 1;
        }
    }
    a
}

/// `sparse_entropy` (`entropy.hh:185-206`), minus the `ea.constants` block.
///
/// The two omitted terms, `get_deg_entropy` (`:104-123`) and
/// `get_parallel_entropy` (`:153-182`), are functions of the graph and the
/// edge weights alone and are therefore invariant under a vertex move ---
/// which [`the_move_invariant_constants_really_are_invariant`] checks rather
/// than assumes.
fn absolute_entropy(
    edges: &[(usize, usize)],
    b: &[u32],
    slots: usize,
    directed: bool,
    deg_corr: bool,
) -> f64 {
    let a = aggregate(edges, b, slots, directed);
    let mut s = 0.0;
    // `parallel_edge_loop_no_spawn(bg, ...)`: one term per block-graph edge.
    // A pair with zero weight has no block edge at all (`entries.hh:405-427`
    // removes it), and `eterm(r, s, 0)` is `-lgamma(1) == 0` in both arms, so
    // iterating the non-zero pairs is the same sum.
    for (&(r, sx), &mrs) in &a.mrs {
        s += eterm_ref(r, sx, mrs, directed);
    }
    // `parallel_vertex_loop_no_spawn(bg, ...)`: one term per block-graph
    // vertex, occupied or not.
    for r in 0..slots {
        s += vterm_ref(a.mrp[r], a.mrm[r], a.wr[r], deg_corr, directed);
    }
    s
}

/// `get_deg_entropy` summed over vertices (`entropy.hh:104-123`) plus
/// `get_parallel_entropy` (`:153-182`), at unit edge weight.
fn move_invariant_constants(edges: &[(usize, usize)], n: usize, directed: bool) -> f64 {
    let (mut kin, mut kout) = (vec![0u64; n], vec![0u64; n]);
    for &(u, w) in edges {
        kout[u] += 1;
        if directed {
            kin[w] += 1;
        } else {
            kout[w] += 1;
        }
    }
    let mut s = 0.0;
    for v in 0..n {
        // `deg_entropy(k) == -lgamma(k + 1)` (`:97-101`).
        s += -ln_fact(kout[v]);
        if directed {
            s += -ln_fact(kin[v]);
        }
    }
    // `get_parallel_entropy`: for each vertex, bucket the out-neighbours and
    // add `parallel_term` (`:135-151`). An undirected graph skips `u < v`, so
    // each unordered pair is counted once, at its larger endpoint.
    let mut adj: Vec<BTreeMap<usize, u64>> = vec![BTreeMap::new(); n];
    for &(u, w) in edges {
        *adj[u].entry(w).or_insert(0) += 1;
        if !directed && u != w {
            *adj[w].entry(u).or_insert(0) += 1;
        }
    }
    for (v, row) in adj.iter().enumerate().take(n) {
        for (&u, &m) in row {
            if !directed && u < v {
                continue;
            }
            if m > 1 {
                if u == v && !directed {
                    s += ln_fact(m / 2) + (m as f64) * std::f64::consts::LN_2 / 2.0;
                } else {
                    s += ln_fact(m);
                }
            }
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn vid(i: usize) -> VertexId {
    VertexId::from_index(i)
}

fn grp(i: u32) -> Group {
    Group::new(i).expect("group index below the null-group sentinel")
}

/// SplitMix64 --- deterministic, so a failure reproduces from the test name.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// A random multigraph with parallel edges and self-loops, plus a partition.
fn fixture(seed: u64, n: usize, m: usize, slots: usize) -> (Vec<(usize, usize)>, Vec<u32>) {
    let mut rng = Rng(seed);
    let mut edges = Vec::with_capacity(m);
    for _ in 0..m {
        let u = rng.below(n);
        // One in six is a self-loop: the shape the recorder halves, the state
        // double-counts into `_mrp`, and `eterm` gives its own `-m log 2` arm.
        let w = if rng.below(6) == 0 { u } else { rng.below(n) };
        edges.push((u, w));
    }
    let b = (0..n).map(|_| rng.below(slots) as u32).collect();
    (edges, b)
}

fn build_graph(edges: &[(usize, usize)], n: usize) -> AdjList {
    let mut g = AdjList::with_vertices(n);
    for &(u, w) in edges {
        g.add_edge(vid(u), vid(w)).expect("both endpoints exist");
    }
    g
}

/// Seed a state that holds exactly `(graph, b)`.
///
/// `assign` is `add_partition_node` (`state.hh:759-780`) and `seed_pair` is
/// one `update_rs`, so this is the state graph-tool would build by moving
/// every vertex into its group from an empty partition.
fn seed<D: Dir>(g: &AdjList, b: &[u32], slots: usize) -> BlockState<D, i64> {
    let mut st = BlockState::<D, i64>::new(slots, b.len());
    for (i, &r) in b.iter().enumerate() {
        st.assign(vid(i), grp(r), 1);
    }
    for e in g.edges() {
        st.seed_pair(grp(b[e.source().index()]), grp(b[e.target().index()]), 1);
    }
    st
}

/// The state's aggregates, read back through [`BlockView`] into the same shape
/// [`aggregate`] produces.
fn read_back<D: Dir>(st: &BlockState<D, i64>, slots: usize) -> Aggr {
    let mut a = Aggr {
        mrs: BTreeMap::new(),
        mrp: vec![0; slots],
        mrm: vec![0; slots],
        wr: vec![0; slots],
    };
    for r in 0..slots {
        a.mrp[r] = st.mrp(grp(r as u32)) as u64;
        a.mrm[r] = st.mrm(grp(r as u32)) as u64;
        a.wr[r] = st.wr(grp(r as u32)) as u64;
        let first = if D::DIRECTED { 0 } else { r };
        for s in first..slots {
            if let Some(e) = st.find_me(grp(r as u32), grp(s as u32)) {
                let w = st.mrs(e);
                if w != 0 {
                    a.mrs.insert((r, s), w as u64);
                }
            }
        }
    }
    a
}

const TOL: f64 = 1e-8;

// ---------------------------------------------------------------------------
// 1. The check graph-tool disables
// ---------------------------------------------------------------------------

/// `mcmc_sweep_wrap`'s assertion (`base_states.py:80-87`), always on.
///
/// For each of the four models --- directed/undirected x degree-corrected or
/// not --- take a random multigraph and a random partition, record 500 vertex
/// moves through [`record`], price each one with
/// [`sparse_ds`](gt_inference::blockmodel::sparse_ds), commit it, and compare
/// the accumulated delta against the difference of two from-scratch entropies
/// computed from `(edges, b)` alone, at graph-tool's own `abs_tol = 1e-8`.
///
/// Along the way the state's own aggregates are read back and compared,
/// element by element, against the same from-scratch `mrs`/`mrp`/`mrm`/`wr` ---
/// so a drift is localised to a block pair rather than showing up only as a
/// number at the end.
///
/// Moves with `r == nr` are included deliberately. They must price to exactly
/// zero, and the C++'s `if constexpr (!single)` guard at `entries.hh:283`
/// leaves `+w` on the `(r, r)` cell for an undirected self-loop in precisely
/// that case; this port drops the guard so the two halves cancel, and a sweep
/// that never proposed a null move could not tell.
#[test]
fn the_delta_sum_equals_the_absolute_entropy_difference() {
    let c = Cache::build(8192);
    let (n, m, slots) = (30usize, 120usize, 6usize);

    for (seed_v, deg_corr) in [(0xD1FF_0001u64, false), (0xD1FF_0002, true)] {
        let (edges, b0) = fixture(seed_v, n, m, slots);
        let g = build_graph(&edges, n);
        let params = EntropyParams {
            deg_corr,
            multigraph: true,
            partition_dl: false,
            degree_dl: false,
        };

        // -- directed -----------------------------------------------------
        {
            let mut b = b0.clone();
            let mut st = seed::<Directed>(&g, &b, slots);
            assert_eq!(
                read_back(&st, slots),
                aggregate(&edges, &b, slots, true),
                "the seeded directed state disagrees with the from-scratch aggregates"
            );
            let s_i = absolute_entropy(&edges, &b, slots, true, deg_corr);
            let mut acc = 0.0;
            let mut rng = Rng(seed_v ^ 0xAAAA);
            for step in 0..500 {
                let v = rng.below(n);
                let nr = rng.below(slots) as u32;
                // `virtual_move_groups` returns 0 for `s == t` *before*
                // `entries_dS` runs (`state.hh:1341-1342`), and `mcmc_sweep`
                // never proposes one anyway (`loops/mcmc_loop.hh:168-173`).
                // `sparse_ds` is `entries_dS`, which has no such guard --- see
                // `a_null_move_is_zero_in_graph_tool_and_is_not_here`.
                if nr == b[v] {
                    continue;
                }
                let mv = MoveKey {
                    from: Some(grp(b[v])),
                    to: Some(grp(nr)),
                };
                let nb = |u: VertexId| Group::new(b[u.index()]);
                let mut ws = Workspace::<Directed, i64>::with_levels(1);
                let t = record(&st, &g, vid(v), mv, &nb, slots, 1i64, &mut ws).seal();
                let ds = sparse_ds(t.level(0), params, &c);
                audit_price(&st, &t, 0, params, &c, ds).expect("price audits");
                let receipt = st.commit(t.into_levels().next().expect("one level"));
                audit_commit(&st, &receipt).expect("commit audits");
                st.reseat(vid(v), grp(nr));
                b[v] = nr;
                acc += ds;

                if step % 100 == 0 {
                    assert_eq!(
                        read_back(&st, slots),
                        aggregate(&edges, &b, slots, true),
                        "directed aggregates drifted at step {step}"
                    );
                }
            }
            let s_f = absolute_entropy(&edges, &b, slots, true, deg_corr);
            assert!(
                (acc - (s_f - s_i)).abs() <= TOL,
                "directed deg_corr={deg_corr}: inconsistent entropy delta \
                 (reported {acc:.12}, actual {:.12}, diff {:e})",
                s_f - s_i,
                acc - (s_f - s_i)
            );
        }

        // -- undirected ---------------------------------------------------
        {
            let mut b = b0.clone();
            let mut st = seed::<Undirected>(&g, &b, slots);
            assert_eq!(
                read_back(&st, slots),
                aggregate(&edges, &b, slots, false),
                "the seeded undirected state disagrees with the from-scratch aggregates"
            );
            let s_i = absolute_entropy(&edges, &b, slots, false, deg_corr);
            let mut acc = 0.0;
            let mut rng = Rng(seed_v ^ 0x5555);
            for step in 0..500 {
                let v = rng.below(n);
                let nr = rng.below(slots) as u32;
                // `virtual_move_groups` returns 0 for `s == t` *before*
                // `entries_dS` runs (`state.hh:1341-1342`), and `mcmc_sweep`
                // never proposes one anyway (`loops/mcmc_loop.hh:168-173`).
                // `sparse_ds` is `entries_dS`, which has no such guard --- see
                // `a_null_move_is_zero_in_graph_tool_and_is_not_here`.
                if nr == b[v] {
                    continue;
                }
                let mv = MoveKey {
                    from: Some(grp(b[v])),
                    to: Some(grp(nr)),
                };
                let nb = |u: VertexId| Group::new(b[u.index()]);
                let mut ws = Workspace::<Undirected, i64>::with_levels(1);
                let t = record(&st, (&g).undirect(), vid(v), mv, &nb, slots, 1i64, &mut ws).seal();
                let ds = sparse_ds(t.level(0), params, &c);
                audit_price(&st, &t, 0, params, &c, ds).expect("price audits");
                let receipt = st.commit(t.into_levels().next().expect("one level"));
                audit_commit(&st, &receipt).expect("commit audits");
                st.reseat(vid(v), grp(nr));
                b[v] = nr;
                acc += ds;

                if step % 100 == 0 {
                    assert_eq!(
                        read_back(&st, slots),
                        aggregate(&edges, &b, slots, false),
                        "undirected aggregates drifted at step {step}"
                    );
                }
            }
            let s_f = absolute_entropy(&edges, &b, slots, false, deg_corr);
            assert!(
                (acc - (s_f - s_i)).abs() <= TOL,
                "undirected deg_corr={deg_corr}: inconsistent entropy delta \
                 (reported {acc:.12}, actual {:.12}, diff {:e})",
                s_f - s_i,
                acc - (s_f - s_i)
            );
        }
    }
}

/// **REAL DIVERGENCE.** A null move (`r == nr`) is zero in graph-tool and is
/// not zero here.
///
/// `virtual_move_groups` (`blockmodel/state.hh:1330-1387`) records the entries
/// and then short-circuits *before* pricing them:
///
/// ```c++
/// get_move_entries(v, s, t, m_entries);
/// if (s == t || n == 0)
///     return 0;                                   // state.hh:1341-1342
/// ...
/// double dS = entries_dS(s, t, -n, n, kin, kout, ea, m_entries);
/// ```
///
/// So `entries_dS` --- which is what [`sparse_ds`] ports --- is never called
/// with `s == t` in graph-tool, and its behaviour there is unspecified. This
/// port exposes `sparse_ds` as *the* pricing entry point and has no
/// `virtual_move_groups` analogue: `SpecCore::virtual_move` (`spec.rs:87`,
/// `spec_base.hh:70`) is where that guard belongs and there is no SBM impl of
/// it yet. The `mcmc_sweep` loop does hold the line --- `metropolis.rs:245`
/// refuses `s == state.node_state(v)`, which is `loops/mcmc_loop.hh:168-173`
/// plus `base/mcmc.hh:105-106` --- so the defect is unreachable *through the
/// sweep* and reachable through every other caller, including
/// `tests/u26_audit.rs`'s own sweep.
///
/// The *entry* half is right: every recorded delta is exactly zero, including
/// the undirected self-loop cell where `entries.hh:283`'s
/// `if constexpr (!single)` guard would have left `+w`. The residual is the
/// vterm half of `entries_dS` (`state.hh:1239-1255`), whose two blocks
/// (`if (r != null)` and `if (nr != null)`) both fire against the *same* group
/// image when `r == nr`, giving the second difference of `vterm` instead of
/// zero:
///
/// ```text
///   [vterm(mrp - dkout, mrm - dkin, wr - dr) - vterm(mrp, mrm, wr)]
/// + [vterm(mrp + dkout, mrm + dkin, wr + dnr) - vterm(mrp, mrm, wr)]
/// ```
///
/// On the fixture below every group holds one vertex, so `wr == 1`,
/// `safelog(1) == 0` and `safelog(0) == 0` kill the first bracket; the second
/// is `(mrp + dkout + mrm + dkin) * log(wr + 1)`. Directed:
/// `(2 + 2 + 2 + 2) * log 2 = 8 log 2`. Undirected: `mrm` is ignored and
/// `dkout` is `degree(v) = 4` (the self-loop counts twice), so
/// `(4 + 4) * log 2 = 8 log 2` again.
#[test]
fn a_null_move_is_zero_in_graph_tool_and_is_not_here() {
    let c = Cache::build(1024);
    let slots = 3usize;
    // v0 carries a self-loop, one out-edge and one in-edge; every group holds
    // exactly one vertex.
    let edges = [(0usize, 0usize), (0, 1), (2, 0), (1, 2)];
    let b = [0u32, 1, 2];
    let g = build_graph(&edges, 3);
    let params = EntropyParams {
        deg_corr: false,
        multigraph: true,
        partition_dl: false,
        degree_dl: false,
    };
    let mv = MoveKey {
        from: Some(grp(0)),
        to: Some(grp(0)),
    };
    let nb = |u: VertexId| Group::new(b[u.index()]);
    let want = 8.0 * LN2;

    // -- directed ---------------------------------------------------------
    let mut st = seed::<Directed>(&g, &b, slots);
    let mut ws = Workspace::<Directed, i64>::with_levels(1);
    let t = record(&st, &g, vid(0), mv, &nb, slots, 1i64, &mut ws).seal();
    assert_eq!(
        t.level(0).header().dkout,
        2,
        "out_degree(v0): the self-loop and `0 -> 1`"
    );
    assert_eq!(
        t.level(0).header().dkin,
        2,
        "in_degree(v0): the self-loop and `2 -> 0`"
    );
    assert!(
        t.level(0).entries().iter().all(|e| e.delta == 0),
        "every recorded entry cancels: {:?}",
        t.level(0)
            .entries()
            .iter()
            .map(|e| (e.r.index(), e.s.index(), e.delta))
            .collect::<Vec<_>>()
    );
    let ds = sparse_ds(t.level(0), params, &c);
    close(ds, want, "directed null move");
    assert_ne!(ds, 0.0, "graph-tool returns 0 here (state.hh:1341-1342)");
    // The commit itself is sound: a delta of all zeros changes nothing.
    let before = read_back(&st, slots);
    st.commit(t.into_levels().next().expect("one level"));
    assert_eq!(
        read_back(&st, slots),
        before,
        "a null move leaves the state alone; only the *price* is wrong"
    );

    // -- undirected -------------------------------------------------------
    let mut st = seed::<Undirected>(&g, &b, slots);
    let mut ws = Workspace::<Undirected, i64>::with_levels(1);
    let t = record(&st, (&g).undirect(), vid(0), mv, &nb, slots, 1i64, &mut ws).seal();
    assert_eq!(
        t.level(0).header().dkout,
        4,
        "degree(v0) on the undirected view: the self-loop counts twice"
    );
    assert_eq!(
        t.level(0).header().dkin,
        0,
        "an undirected view has no in-half"
    );
    assert!(
        t.level(0).entries().iter().all(|e| e.delta == 0),
        "the `(r, r)` self-loop cell cancels -- this port drops the \
         `if constexpr (!single)` of entries.hh:283, which would leave `+w`"
    );
    let ds = sparse_ds(t.level(0), params, &c);
    close(ds, want, "undirected null move");
    let before = read_back(&st, slots);
    st.commit(t.into_levels().next().expect("one level"));
    assert_eq!(read_back(&st, slots), before);
}

/// The two terms `sparse_entropy` skips under `ea.constants` really are
/// invariant under a vertex move.
///
/// `get_deg_entropy` reads `_degs[v]` and `get_parallel_entropy` reads the
/// graph and `_eweight` (`entropy.hh:207-221`); neither mentions `_b`. Dropping
/// them from the from-scratch entropy in this file is therefore not a
/// convenience --- it is required for the delta comparison to mean anything,
/// and this pins it.
#[test]
fn the_move_invariant_constants_really_are_invariant() {
    let (edges, b0) = fixture(0xC0FFEE, 20, 70, 4);
    for directed in [true, false] {
        let base = move_invariant_constants(&edges, 20, directed);
        let mut rng = Rng(7);
        let mut b = b0.clone();
        for _ in 0..50 {
            b[rng.below(20)] = rng.below(4) as u32;
            assert_eq!(
                move_invariant_constants(&edges, 20, directed),
                base,
                "the constants depend on the partition"
            );
        }
        // ...and they are not vacuously zero on this fixture.
        assert!(base != 0.0);
    }
}

// ---------------------------------------------------------------------------
// 2. The port's `Cache` against the definition
// ---------------------------------------------------------------------------

/// `Cache::lgamma1p`, `safelog` and `lbinom` agree with the definitions.
///
/// `lgamma_fast` is a table of `safe_lgamma(y)` at integer `y`
/// (`cache.hh:139-144`), `safelog_fast` of `safelog(y)` (`:106-116`), and
/// `lbinom_fast` is the three-guard expression at `support/util.hh:42-47` with
/// **that** association --- `(lgamma(N+1) - lgamma(k+1)) - lgamma(N-k+1)` ---
/// which is load-bearing for the last bits. All three are checked here against
/// a sum of logs, on and off the end of the table.
#[test]
fn the_cache_agrees_with_the_definitions_it_tabulates() {
    // Deliberately small, so the fall-off-the-table path is exercised too:
    // `build(n)` sizes to `get_next_size(n + 1)` (`cache.hh:64-71`), i.e. 64.
    let c = Cache::build(50);
    assert_eq!(c.len(), 64, "`get_next_size(51)`");

    for n in 0u64..=200 {
        let want = ln_fact(n);
        let got = c.lgamma1p(n as f64);
        assert!(
            (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
            "lgamma1p({n}): {got} vs ln({n}!) = {want}"
        );
        assert_eq!(c.safelog(n as f64), safelog(n as f64), "safelog({n})");
        assert_eq!(c.xlogx(n as f64), (n as f64) * safelog(n as f64));
    }
    assert_eq!(c.safelog(0.0), 0.0, "safelog(0) == 0, not -inf");
    assert_eq!(c.lgamma1p(0.0), 0.0);
    assert_eq!(c.lgamma1p(1.0), 0.0);

    // The three `lbinom_fast` guards, each reached.
    assert_eq!(c.lbinom(0.0, 0.0), 0.0, "N == 0");
    assert_eq!(c.lbinom(5.0, 0.0), 0.0, "k == 0");
    assert_eq!(c.lbinom(5.0, 5.0), 0.0, "k >= N");
    assert_eq!(c.lbinom(5.0, 9.0), 0.0, "k >= N");
    for (n, k) in [(7u64, 4u64), (10, 3), (100, 7), (200, 100)] {
        let want = lbinom(n, k);
        let got = c.lbinom(n as f64, k as f64);
        assert!(
            (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
            "lbinom({n}, {k}): {got} vs {want}"
        );
    }
    // `binom(7, 4) == 35`.
    assert!((c.lbinom(7.0, 4.0) - 35f64.ln()).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// 3. Hand-computed terms on a three-vertex graph
// ---------------------------------------------------------------------------
//
// Directed fixture D: vertices {0, 1, 2}, b = [0, 0, 1], edges
// `0->1`, `1->0`, `0->2`, `2->2`.
//
//   wr  = [2, 1]
//   mrs = {(0,0): 2, (0,1): 1, (1,1): 1}
//   mrp = [3, 1]        (edges leaving group 0: 0->1, 1->0, 0->2)
//   mrm = [2, 2]        (edges entering group 0: 0->1, 1->0)
//
// Undirected fixture U: vertices {0, 1, 2}, b = [0, 0, 1], edges
// `{0,1}`, `{0,2}`, `{2,2}`.
//
//   wr  = [2, 1]
//   mrs = {(0,0): 1, (0,1): 1, (1,1): 1}
//   mrp = [3, 3]        (each edge adds one to *each* endpoint's group, so
//                        the self-loop adds two to group 1)
//   mrm = [0, 0]        (unused when undirected)
//
// ---------------------------------------------------------------------------

const LN2: f64 = std::f64::consts::LN_2;

fn close(a: f64, b: f64, what: &str) {
    assert!(
        (a - b).abs() < 1e-12,
        "{what}: got {a:.15}, hand-computed {b:.15}"
    );
}

/// `eterm` and `vterm` against values computed on paper.
///
/// The undirected `r == s` arm is the interesting one: `eterm_d` subtracts
/// `mrs * log(2)` there (`entropy.hh:50-57`) and nowhere else, because an
/// undirected block self-pair counts each of its `mrs` edges in both
/// directions.
#[test]
fn the_entropy_terms_match_hand_computed_values() {
    let c = Cache::build(256);

    // -- eterm ------------------------------------------------------------
    close(
        eterm::<Directed>(0, 0, 2.0, &c),
        -ln_fact(2),
        "directed (0,0)",
    );
    close(
        eterm::<Directed>(0, 0, 2.0, &c),
        -LN2,
        "directed (0,0) closed form",
    );
    close(eterm::<Directed>(0, 1, 1.0, &c), 0.0, "directed (0,1)");
    close(eterm::<Directed>(1, 1, 1.0, &c), 0.0, "directed (1,1)");
    close(
        eterm::<Undirected>(0, 0, 1.0, &c),
        -LN2,
        "undirected (0,0): -ln(1!) - 1*ln 2",
    );
    close(eterm::<Undirected>(0, 1, 1.0, &c), 0.0, "undirected (0,1)");
    close(
        eterm::<Undirected>(0, 0, 3.0, &c),
        -(6f64.ln()) - 3.0 * LN2,
        "undirected (0,0) at mrs = 3: -ln(3!) - 3 ln 2",
    );

    // -- vterm ------------------------------------------------------------
    close(
        vterm::<Directed>(3.0, 2.0, 2.0, false, &c),
        5.0 * LN2,
        "directed non-deg-corr: (mrp + mrm) log wr",
    );
    close(
        vterm::<Directed>(1.0, 2.0, 1.0, false, &c),
        0.0,
        "log(1) == 0",
    );
    close(
        vterm::<Directed>(3.0, 2.0, 2.0, true, &c),
        6f64.ln() + LN2,
        "directed deg-corr: ln(3!) + ln(2!)",
    );
    close(
        vterm::<Undirected>(3.0, 0.0, 2.0, false, &c),
        3.0 * LN2,
        "undirected non-deg-corr: mrp log wr, `mrm` ignored",
    );
    close(
        vterm::<Undirected>(3.0, 99.0, 2.0, true, &c),
        6f64.ln(),
        "undirected deg-corr: ln(3!), `mrm` ignored (entropy.hh:79-80)",
    );
    close(
        vterm::<Undirected>(4.0, 0.0, 0.0, false, &c),
        0.0,
        "safelog(0) == 0 keeps an empty group's term finite",
    );
}

/// Fixture D and fixture U priced entirely on paper, and reproduced by a real
/// [`BlockState`] seeded from a real [`AdjList`].
///
/// ```text
///   D, non-deg-corr:  -ln 2  +  5 ln 2            =  4 ln 2
///   D, deg-corr:      -ln 2  +  (ln 6 + ln 2) + ln 2  =  ln 12
///   U, non-deg-corr:  -2 ln 2  +  3 ln 2         =  ln 2
///   U, deg-corr:      -2 ln 2  +  2 ln 6         =  2 ln 3
/// ```
#[test]
fn a_three_vertex_graph_is_priced_by_hand() {
    let slots = 2usize;
    let b = [0u32, 0, 1];

    // -- directed ---------------------------------------------------------
    let d_edges = [(0usize, 1usize), (1, 0), (0, 2), (2, 2)];
    let dg = build_graph(&d_edges, 3);
    let d_agg = aggregate(&d_edges, &b, slots, true);
    assert_eq!(d_agg.wr, vec![2, 1]);
    assert_eq!(d_agg.mrp, vec![3, 1]);
    assert_eq!(d_agg.mrm, vec![2, 2]);
    assert_eq!(
        d_agg.mrs.iter().map(|(&k, &w)| (k, w)).collect::<Vec<_>>(),
        vec![((0, 0), 2), ((0, 1), 1), ((1, 1), 1)]
    );
    assert_eq!(
        read_back(&seed::<Directed>(&dg, &b, slots), slots),
        d_agg,
        "the state built from the graph holds the hand-computed aggregates"
    );
    close(
        absolute_entropy(&d_edges, &b, slots, true, false),
        4.0 * LN2,
        "D, non-deg-corr",
    );
    close(
        absolute_entropy(&d_edges, &b, slots, true, true),
        12f64.ln(),
        "D, deg-corr",
    );

    // -- undirected -------------------------------------------------------
    let u_edges = [(0usize, 1usize), (0, 2), (2, 2)];
    let ug = build_graph(&u_edges, 3);
    let u_agg = aggregate(&u_edges, &b, slots, false);
    assert_eq!(u_agg.wr, vec![2, 1]);
    assert_eq!(
        u_agg.mrp,
        vec![3, 3],
        "the undirected self-loop adds two to group 1 (entries.hh:397)"
    );
    assert_eq!(
        u_agg.mrs.iter().map(|(&k, &w)| (k, w)).collect::<Vec<_>>(),
        vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
    );
    assert_eq!(read_back(&seed::<Undirected>(&ug, &b, slots), slots), u_agg);
    close(
        absolute_entropy(&u_edges, &b, slots, false, false),
        LN2,
        "U, non-deg-corr",
    );
    close(
        absolute_entropy(&u_edges, &b, slots, false, true),
        2.0 * 3f64.ln(),
        "U, deg-corr",
    );
}

/// `get_edges_dl` (`entropy.hh:293-298`) on paper.
///
/// `BB` is `B * B` when directed and `B * (B + 1) / 2` when not, and the term
/// is `lbinom(BB + E - 1, E)`. For fixture D (`B = 2`, `E = 4`) that is
/// `lbinom(7, 4) = ln 35`; for fixture U (`B = 2`, `E = 3`),
/// `lbinom(5, 3) = ln 10`.
#[test]
fn the_edges_description_length_matches_hand_computed_values() {
    let c = Cache::build(256);
    close(
        edges_dl::<Directed>(2, 4.0, &c),
        35f64.ln(),
        "directed B=2 E=4",
    );
    close(
        edges_dl::<Undirected>(2, 3.0, &c),
        10f64.ln(),
        "undirected B=2 E=3",
    );
    // A single group: directed `BB = 1`, so `lbinom(E, E) == 0` by the
    // `k >= N` guard; undirected `BB = 1` likewise.
    close(edges_dl::<Directed>(1, 9.0, &c), 0.0, "B = 1 is free");
    close(edges_dl::<Undirected>(1, 9.0, &c), 0.0, "B = 1 is free");
}

/// `partition::get_dl` (`partition.hh:106-117`) on paper.
///
/// `lbinom(N - 1, B - 1) + lgamma(N + 1) - sum lgamma(n_r + 1) + safelog(N)`,
/// where `B` is `_actual_B`, the number of **occupied** groups --- so an empty
/// slot must not raise it. With `sizes = [2, 1]` and `N = 3`:
/// `ln 2 + ln 6 - ln 2 - 0 + ln 3 = ln 18`.
#[test]
fn the_partition_description_length_matches_hand_computed_values() {
    let c = Cache::build(256);
    close(partition_dl(&[2, 1], 3, &c), 18f64.ln(), "sizes [2, 1]");
    close(
        partition_dl(&[2, 0, 1, 0], 3, &c),
        18f64.ln(),
        "empty slots do not count towards `_actual_B`",
    );
    close(partition_dl(&[], 0, &c), 0.0, "`if (_N == 0) return 0`");
    // Everything in one group: `lbinom(N-1, 0) == 0`, `lgamma(N+1)` cancels
    // the single `-lgamma(n_r + 1)`, leaving `log N`.
    close(partition_dl(&[5], 5, &c), 5f64.ln(), "one group");
}

/// `eterm_dense_d` (`entropy.hh:235-258`) on paper, including the arm that
/// needs `r` and `s` and cannot be recovered from `wr_r` and `wr_s`.
///
/// ```text
///   nrns = wr_r * wr_s                     if directed or r != s
///        = wr_r * (wr_r + 1) / 2           if undirected, r == s, multigraph
///        = wr_r * (wr_r - 1) / 2           if undirected, r == s, simple
///   term = lbinom(nrns + ers - 1, ers)     if multigraph
///        = lbinom(nrns, ers)               otherwise
/// ```
#[test]
fn the_dense_edge_term_matches_hand_computed_values() {
    let c = Cache::build(256);
    // Undirected diagonal, wr = 2, ers = 1.
    close(
        eterm_dense::<Undirected>(0, 0, 1.0, 2.0, 2.0, true, &c),
        3f64.ln(),
        "multigraph: nrns = 2*3/2 = 3, lbinom(3, 1) = ln 3",
    );
    close(
        eterm_dense::<Undirected>(0, 0, 1.0, 2.0, 2.0, false, &c),
        0.0,
        "simple: nrns = 2*1/2 = 1, lbinom(1, 1) = 0 by the `k >= N` guard",
    );
    // The same two vertex counts off the diagonal take the product arm...
    close(
        eterm_dense::<Undirected>(0, 1, 1.0, 2.0, 2.0, true, &c),
        4f64.ln(),
        "off-diagonal: nrns = 4, lbinom(4, 1) = ln 4",
    );
    // ...and a *directed* diagonal takes it too (`directed || r != s`).
    close(
        eterm_dense::<Directed>(0, 0, 1.0, 2.0, 2.0, true, &c),
        4f64.ln(),
        "directed diagonal is still the product",
    );
    close(
        eterm_dense::<Directed>(0, 1, 1.0, 2.0, 1.0, true, &c),
        2f64.ln(),
        "nrns = 2, lbinom(2, 1) = ln 2",
    );
    // This is the divergence the skeleton's signature could not express: with
    // only `wr_r` and `wr_s` in hand the two undirected arms are
    // indistinguishable, and they differ by roughly a factor of two.
    assert_ne!(
        eterm_dense::<Undirected>(0, 0, 1.0, 2.0, 2.0, true, &c),
        eterm_dense::<Undirected>(0, 1, 1.0, 2.0, 2.0, true, &c)
    );
}
