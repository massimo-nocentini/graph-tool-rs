//! U6 -- deterministic parallel construction.
//!
//! The unit's acceptance criterion is a statement about a *process*: "the same
//! input built with `RAYON_NUM_THREADS` = 1, 4 and 16 produces byte-identical
//! edge-id assignments and an identical `validate()`ing graph". That variable
//! is read by rayon's **global** pool, once, when the pool is first used, so
//! three values of it cannot be observed inside one test binary. The test
//! below therefore re-executes this binary three times with the variable set,
//! has each child write a fingerprint of the graph it built to a file, and
//! compares the files. (The complementary in-process check, over explicit
//! `rayon::ThreadPool`s of 1, 4 and 16 threads, is a `#[cfg(test)]` module in
//! `src/adj/builder.rs`; `rayon` is a dependency of `gt-core` and not of its
//! test binaries, so that half cannot live here.)
//!
//! The fingerprint is not the edge *set*. It is the whole observable layout:
//! the id-to-endpoint map across the entire index range, and every vertex's
//! block in storage order with its out/in split. A builder that produced the
//! right edges in a thread-dependent order would pass a set comparison and
//! fail this one -- and the order is observable, because `out_edges(v)` yields
//! storage order and every kernel in the port iterates it.
//!
//! What graph-tool does here instead: `set_concurrent(true)`
//! (`graph_adjacency.hh:451`) dices the free list across `get_num_threads()`
//! per-thread lists (`:460-466`) and `get_free_idx` (`:627-655`) then hands
//! out whichever index the *taking worker's* list holds, with
//! `#pragma omp atomic capture idx = _edge_idx_range++` (`:635`) beneath it.
//! `graph_merge.hh:103-108` turns that on when
//! `num_vertices(g) > get_openmp_min_thresh() && get_num_threads() > 1`, so
//! the edge indices of a merged graph -- and therefore every edge property
//! map indexed by them -- depend on the machine.

use std::collections::{BTreeMap, HashSet};
use std::process::Command;
use std::thread::ThreadId;

use gt_core::adj::{AdjList, EHash, EdgeIds, Lookup, NoLookup, ParBuilder};
use gt_core::error::GraphError;
use gt_core::ids::{EdgeId, VertexId};
use proptest::prelude::*;

// ===========================================================================
// The input, and the fingerprint
// ===========================================================================

/// SplitMix64 mixed with the item index: a chunk generates its own slice of
/// the stream from its own index alone, which is what `fill`'s determinism
/// contract requires of a generator (and what `parallel_rng.hh:56-61`'s
/// `_rngs[tnum - 1]` is not).
fn mix(seed: u64, i: u64) -> u64 {
    let mut z = seed
        .wrapping_add(i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

const N: usize = 1_024;
const CHUNKS: usize = 61;
const PER: usize = 97;

/// Chunk `k`'s pairs. Self-loops (one in eight) and parallel edges (the
/// endpoint space is deliberately small) are both frequent, because those are
/// the cases where `insert_out`'s push-swap (`graph_adjacency.hh:1203-1210`)
/// actually moves an entry and the block layout becomes order-sensitive.
fn chunk_pairs(k: usize, out: &mut Vec<(VertexId, VertexId)>) {
    out.reserve(PER);
    for j in 0..PER {
        let i = (k * PER + j) as u64;
        let s = mix(0x5eed, 2 * i) as usize % N;
        let t = if mix(0x5eed, 4 * i).is_multiple_of(8) {
            s
        } else {
            mix(0x5eed, 2 * i + 1) as usize % N
        };
        out.push((VertexId::from_index(s), VertexId::from_index(t)));
    }
}

/// The same pairs, in merge order, without the builder.
fn expected_sequence() -> Vec<(VertexId, VertexId)> {
    let mut all = Vec::new();
    for k in 0..CHUNKS {
        chunk_pairs(k, &mut all);
    }
    all
}

/// Every observable byte of a built graph: counts, the id-to-endpoint map over
/// the whole index range (holes included), and each vertex's block in storage
/// order with the out/in boundary.
fn fingerprint<H: Lookup>(g: &AdjList<H>) -> Vec<u64> {
    let mut f = vec![
        g.num_vertices() as u64,
        g.num_edges() as u64,
        g.edge_bound().len() as u64,
    ];
    for i in 0..g.edge_bound().len() {
        match g.endpoints(EdgeId::from_index(i)) {
            Some((s, t)) => f.extend([1, s.index() as u64, t.index() as u64]),
            None => f.extend([0, 0, 0]),
        }
    }
    for v in g.vertices() {
        let b = g.block(v).expect("vertex in range");
        f.push(b.out_degree() as u64);
        f.push(b.degree() as u64);
        for e in b.all() {
            f.push(e.other.index() as u64);
            f.push(e.idx.index() as u64);
        }
    }
    f
}

fn to_bytes(f: &[u64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(f.len() * 8);
    for x in f {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

// ===========================================================================
// The acceptance test: RAYON_NUM_THREADS = 1, 4, 16
// ===========================================================================

/// Set by the parent on each child; its value is where the child writes.
const OUT_VAR: &str = "GT_U06_FINGERPRINT_OUT";

/// libtest names an integration test by its function path, which for a
/// top-level `#[test]` is just the name.
const SELF: &str = "the_thread_count_changes_neither_the_edge_ids_nor_the_layout";

#[test]
fn the_thread_count_changes_neither_the_edge_ids_nor_the_layout() {
    // ---- child ----------------------------------------------------------
    if let Ok(path) = std::env::var(OUT_VAR) {
        // How many distinct threads rayon actually ran the generators on.
        // Without it a build in which `RAYON_NUM_THREADS` was ignored would
        // pass this test by never having been parallel at all.
        let seen = std::sync::Mutex::new(HashSet::<ThreadId>::new());
        let mut b = ParBuilder::new(N, CHUNKS);
        b.fill(|k, out| {
            seen.lock()
                .expect("poisoned")
                .insert(std::thread::current().id());
            chunk_pairs(k, out);
        });
        let g = b.build(NoLookup).expect("build");
        g.validate().expect("the built graph must validate");

        let workers = seen.into_inner().expect("poisoned").len() as u64;
        let mut payload = workers.to_le_bytes().to_vec();
        payload.extend_from_slice(&to_bytes(&fingerprint(&g)));
        std::fs::write(&path, payload).expect("write fingerprint");
        return;
    }

    // ---- parent ---------------------------------------------------------
    let exe = std::env::current_exe().expect("current_exe");
    let dir = std::env::temp_dir();
    let mut results: Vec<(usize, u64, Vec<u8>)> = Vec::new();

    for threads in [1usize, 4, 16] {
        let path = dir.join(format!("gt-u06-{}-{threads}.fp", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let status = Command::new(&exe)
            .args(["--exact", SELF, "--test-threads=1"])
            .env("RAYON_NUM_THREADS", threads.to_string())
            .env(OUT_VAR, &path)
            .status()
            .expect("re-exec the test binary");
        assert!(status.success(), "child with {threads} threads failed");

        let blob = std::fs::read(&path).expect("child wrote no fingerprint");
        let _ = std::fs::remove_file(&path);
        assert!(
            blob.len() > 8,
            "empty fingerprint from the {threads}-thread child"
        );
        let workers = u64::from_le_bytes(blob[..8].try_into().expect("8 bytes"));
        results.push((threads, workers, blob[8..].to_vec()));
    }

    // RAYON_NUM_THREADS=1 means one worker, and rayon does not run injected
    // jobs on the injecting thread: seeing exactly one is the evidence that
    // the variable reached the pool at all.
    assert_eq!(
        results[0].1, 1,
        "RAYON_NUM_THREADS=1 did not produce a single-worker pool"
    );

    let (_, _, first) = &results[0];
    for (threads, workers, blob) in &results[1..] {
        assert_eq!(
            first, blob,
            "the graph built with RAYON_NUM_THREADS={threads} ({workers} worker(s) observed) \
             is not byte-identical to the one built on a single thread"
        );
    }

    // And the ids really are the staging order, not merely a stable order.
    let seq = expected_sequence();
    let g = {
        let mut b = ParBuilder::new(N, CHUNKS);
        b.fill(chunk_pairs);
        b.build(NoLookup).expect("build")
    };
    assert_eq!(g.num_edges(), seq.len());
    assert_eq!(to_bytes(&fingerprint(&g)), *first);
}

// ===========================================================================
// The merge is the incremental splice, in chunk order
// ===========================================================================

/// Build the same edges through `add_edge`, one at a time.
fn serially<H: Lookup>(n: usize, seq: &[(VertexId, VertexId)], lookup: H) -> AdjList<H> {
    let mut g = AdjList::with_lookup(lookup);
    for _ in 0..n {
        g.add_vertex().expect("add_vertex");
    }
    for &(s, t) in seq {
        g.add_edge(s, t).expect("add_edge");
    }
    g
}

#[test]
fn the_built_graph_is_the_serially_built_graph() {
    let seq = expected_sequence();
    let mut b = ParBuilder::new(N, CHUNKS);
    b.fill(chunk_pairs);
    let built = b.build(NoLookup).expect("build");
    built.validate().expect("validate");

    let serial = serially(N, &seq, NoLookup);
    serial.validate().expect("validate");

    // Not "the same edges": the same adjacency layout, entry for entry. The
    // builder has no second, bulk-only construction path that could disagree
    // with the incremental one about the block layout, the slot table or the
    // index range.
    assert_eq!(fingerprint(&built), fingerprint(&serial));
}

#[test]
fn the_lookup_index_is_built_too() {
    let seq = expected_sequence();
    let mut b = ParBuilder::new(N, CHUNKS);
    b.fill(chunk_pairs);
    let built = b.build(EHash::default()).expect("build");
    // `validate()` checks the (s,t) index in multiplicity, not membership.
    built.validate().expect("validate");
    assert_eq!(
        fingerprint(&built),
        fingerprint(&serially(N, &seq, EHash::default()))
    );

    // And it answers. `find_edge` returns the bucket's first id, which is the
    // oldest surviving parallel edge -- so for the first occurrence of a pair
    // in the merge order it is that occurrence's own id.
    let mut first_seen: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (i, &(s, t)) in seq.iter().enumerate() {
        first_seen.entry((s.index(), t.index())).or_insert(i);
    }
    for (&(s, t), &i) in &first_seen {
        let e = built
            .find_edge(VertexId::from_index(s), VertexId::from_index(t))
            .expect("the edge was staged");
        assert_eq!(e.id().index(), i);
    }
}

#[test]
fn edge_ids_follow_the_chunk_order_and_not_the_push_order() {
    // Two edges per chunk, staged back to front: chunk 3's pair is pushed
    // first and must still receive the *last* two ids.
    let mut b = ParBuilder::new(8, 4);
    for k in (0..4).rev() {
        b.push(k, VertexId::from_index(k), VertexId::from_index(k + 1));
        b.push(k, VertexId::from_index(k + 4), VertexId::from_index(k));
    }
    let g = b.build(NoLookup).expect("build");
    g.validate().expect("validate");
    assert_eq!(g.num_edges(), 8);
    for k in 0..4usize {
        assert_eq!(
            g.endpoints(EdgeId::from_index(2 * k)),
            Some((VertexId::from_index(k), VertexId::from_index(k + 1)))
        );
        assert_eq!(
            g.endpoints(EdgeId::from_index(2 * k + 1)),
            Some((VertexId::from_index(k + 4), VertexId::from_index(k)))
        );
    }
}

#[test]
fn where_the_chunk_boundaries_fall_does_not_matter_only_the_concatenation() {
    let seq = expected_sequence();
    let mut fps = Vec::new();
    for chunks in [1usize, 7, 61, 4_096] {
        let per = seq.len().div_ceil(chunks);
        let mut b = ParBuilder::new(N, chunks);
        for (i, &(s, t)) in seq.iter().enumerate() {
            b.push(i / per.max(1), s, t);
        }
        let g = b.build(NoLookup).expect("build");
        g.validate().expect("validate");
        fps.push(fingerprint(&g));
    }
    for f in &fps[1..] {
        assert_eq!(
            &fps[0], f,
            "the merge depends on more than the concatenation"
        );
    }
}

// ===========================================================================
// Degenerate and failing inputs
// ===========================================================================

#[test]
fn an_endpoint_outside_the_graph_is_an_error_and_yields_no_graph() {
    let mut b = ParBuilder::new(3, 2);
    b.push(0, VertexId::from_index(0), VertexId::from_index(1));
    b.push(1, VertexId::from_index(2), VertexId::from_index(3));
    assert_eq!(
        b.build(NoLookup).map(|g| g.num_edges()),
        Err(GraphError::NoSuchVertex(VertexId::from_index(3)))
    );

    // The failing pair is reported even when it is the *source*, and the
    // check happens before an index is taken, exactly as in `add_edge`.
    let mut b = ParBuilder::new(3, 1);
    b.push(0, VertexId::from_index(9), VertexId::from_index(0));
    assert_eq!(
        b.build(EHash::default()).map(|g| g.num_edges()),
        Err(GraphError::NoSuchVertex(VertexId::from_index(9)))
    );
}

#[test]
fn a_vertex_count_past_the_index_space_is_an_error_not_a_panic() {
    // `add_vertex(g, n)` (`graph_adjacency.hh:1318-1333`) is an unchecked
    // `_edges.resize(v + n)`, and `Vertex` doubles as the index type, so the
    // overflowing graph collides silently with `null_vertex()`
    // (`graph_adjacency.hh:539`).
    //
    // `max_index() + 2` is itself a `usize` overflow under `wide-index` on a
    // 64-bit target, where `MAX_INDEX` is `usize::MAX - 1`: there the
    // "past the index space" vertex count is not a representable `usize` at
    // all, so the guard is unreachable rather than untested. `checked_add`
    // states which of the two configurations is running instead of panicking
    // in `[profile.dev]`'s overflow checks before the assertion is reached.
    let Some(past) = EdgeIds::max_index().checked_add(2) else {
        assert_eq!(
            EdgeIds::max_index(),
            usize::MAX - 1,
            "`MAX_INDEX + 2` overflowed a `usize` at an index width that \
             leaves room above `MAX_INDEX`; the guard below is reachable and \
             should have been exercised"
        );
        return;
    };
    let b = ParBuilder::new(past, 0);
    assert_eq!(
        b.build(NoLookup).map(|g| g.num_vertices()),
        Err(GraphError::VertexIdSpaceExhausted {
            max: EdgeIds::max_index()
        })
    );
}

#[test]
fn an_empty_builder_builds_an_empty_graph() {
    let g = ParBuilder::new(0, 8).build(NoLookup).expect("build");
    assert_eq!((g.num_vertices(), g.num_edges()), (0, 0));
    assert_eq!(g.edge_bound().len(), 0);
    g.validate().expect("validate");

    // Chunk count is reported as given: it is part of the plan, and a builder
    // that rounded it would be a builder with a different merge order.
    assert_eq!(ParBuilder::new(4, 61).chunks(), 61);
}

#[test]
fn isolated_vertices_survive_the_merge() {
    let mut b = ParBuilder::new(5, 1);
    b.push(0, VertexId::from_index(4), VertexId::from_index(4));
    let g = b.build(NoLookup).expect("build");
    g.validate().expect("validate");
    assert_eq!(g.num_vertices(), 5);
    assert_eq!(g.num_edges(), 1);
    // A self-loop occupies both halves of the one block.
    assert_eq!(g.out_degree(VertexId::from_index(4)), 1);
    assert_eq!(g.in_degree(VertexId::from_index(4)), 1);
    for v in 0..4 {
        assert_eq!(g.degree(VertexId::from_index(v)), 0);
    }
}

// ===========================================================================
// Model-based: any chunking of any edge sequence
// ===========================================================================

/// `(n_vertices, chunk sizes, pairs)`, with a small vertex space so that
/// self-loops and parallel edges are the common case rather than the corner.
fn input() -> impl Strategy<Value = (usize, Vec<usize>, Vec<(usize, usize)>)> {
    (1usize..7).prop_flat_map(|n| {
        (
            Just(n),
            prop::collection::vec(0usize..4, 1..8),
            prop::collection::vec((0..n, 0..n), 0..24),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    /// Whatever the chunking, the built graph is the graph the same pairs
    /// would have produced through `add_edge` in concatenation order, it
    /// validates, and edge `i` is the `i`-th pair.
    #[test]
    fn any_chunking_merges_to_the_serial_graph((n, sizes, pairs) in input()) {
        // Deal the pairs into the chunks by the generated sizes, cycling.
        let chunks = sizes.len();
        let mut b = ParBuilder::new(n, chunks);
        let mut seq: Vec<Vec<(VertexId, VertexId)>> = vec![Vec::new(); chunks];
        let mut k = 0usize;
        let mut room = sizes[0];
        for &(s, t) in &pairs {
            while room == 0 && k + 1 < chunks {
                k += 1;
                room = sizes[k];
            }
            let (s, t) = (VertexId::from_index(s), VertexId::from_index(t));
            b.push(k, s, t);
            seq[k].push((s, t));
            room = room.saturating_sub(1);
        }
        let flat: Vec<(VertexId, VertexId)> = seq.concat();

        let built = b.build(NoLookup).expect("build");
        prop_assert!(built.validate().is_ok());
        prop_assert_eq!(built.num_edges(), flat.len());
        prop_assert_eq!(built.edge_bound().len(), flat.len());

        for (i, &(s, t)) in flat.iter().enumerate() {
            prop_assert_eq!(built.endpoints(EdgeId::from_index(i)), Some((s, t)));
        }

        let serial = serially(n, &flat, NoLookup);
        prop_assert_eq!(fingerprint(&built), fingerprint(&serial));

        // The same claim once more with the (s,t) index switched on, because
        // it is a second derived structure with its own hooks.
        let mut hb = ParBuilder::new(n, chunks);
        for (k, c) in seq.iter().enumerate() {
            for &(s, t) in c {
                hb.push(k, s, t);
            }
        }
        let hashed = hb.build(EHash::default()).expect("build");
        prop_assert!(hashed.validate().is_ok());
        prop_assert_eq!(fingerprint(&hashed), fingerprint(&serially(n, &flat, EHash::default())));
    }
}
