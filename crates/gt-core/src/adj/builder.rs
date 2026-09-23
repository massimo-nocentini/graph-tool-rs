//! Parallel bulk construction.
//!
//! ## What this replaces, and why it is a different shape (DESIGN.md D9)
//!
//! graph-tool supports concurrent mutation of a *live* graph:
//! `set_concurrent(true)` (`graph_adjacency.hh:451`), a per-thread
//! `_free_idx_m` (`:613`), `#pragma omp atomic g._n_edges++` (`:1220`) and
//! `#pragma omp atomic capture idx = _edge_idx_range++` (`:635`). It is
//! correct only if the caller guarantees vertex-disjointness, and nothing
//! checks that.
//!
//! `&mut AdjList` refuses all of it, and three of the six source designs
//! recorded that refusal as a win. It is not a win, it is a removed
//! capability; the honest replacement is this builder. Each worker stages
//! `(s, t)` pairs into its own buffer, edge-id ranges are reserved with one
//! atomic each, and a deterministic merge splices them in chunk order -- so
//! the resulting edge ids do not depend on the thread count, which
//! graph-tool's per-thread free lists explicitly do.
//!
//! ## What "does not depend on the thread count" costs graph-tool
//!
//! `graph_merge.hh:103-108` is the shape of a real caller:
//!
//! ```text
//! parallel = (parallel && (num_vertices(g) > get_openmp_min_thresh())
//!                      && (get_num_threads() > 1));
//! base_graph(ug).set_concurrent(parallel);
//! ```
//!
//! so whether the merged graph's edge indices come off one free list or off
//! `get_num_threads()` of them is decided by the machine and by a runtime
//! threshold. Every edge property map written by index afterwards inherits
//! that. Here the analogous decision is [`ParBuilder::new`]'s `chunks`
//! argument: a number the *caller* fixes, that is part of the plan and not of
//! the deployment, and that is why it must never be derived from
//! `available_parallelism()`.
//!
//! ## The merge is serial, and that is the point
//!
//! [`fill`](ParBuilder::fill) is the parallel phase: generation, I/O decoding,
//! neighbour sampling -- whatever produces the pairs, done on all cores with
//! each chunk seeing only its own index. [`build`](ParBuilder::build) is then
//! one pass of the ordinary [`AdjList::add_edge`] splice over the chunks in
//! index order. Two consequences, both deliberate:
//!
//! * the splice is the same code the incremental API runs, so a graph built
//!   here is *byte-identical* to the same edges added one at a time in chunk
//!   order -- there is no second, bulk-only construction path that could
//!   disagree with the first about the adjacency layout, the slot table or
//!   the `(s,t)` index;
//! * `build` is `O(E)` serial work. The speed-up over a fully serial pipeline
//!   is Amdahl's on the staging phase, never more, and `benches/builder.rs`
//!   measures exactly that rather than asserting it.
//!
//! ## What the benchmark actually shows
//!
//! `benches/builder.rs` drives both paths with a SplitMix64 generator at
//! 10^5, 10^6 and 10^7 edges -- the *cheapest imaginable* staging phase, and
//! therefore this shape's worst case. There the two paths are within noise of
//! each other: the splice is the entire cost, and the builder's own
//! contribution is one streamed write and one streamed read of eight bytes
//! per edge. Nothing here claims a speed-up on that input, and the ledger
//! entry is worth taking only on an idle machine. The work a real caller does
//! per edge is what moves off the critical path -- `graph_merge.hh:95-101`
//! weighs each edge and `:120-135` locks its endpoints, `graph_contract_edges.hh`
//! hashes them -- and that work is precisely what `set_concurrent` was turned
//! on for.

use crate::error::GraphError;
use crate::ids::{MAX_INDEX, VertexId};

use rayon::prelude::*;

use super::index::Lookup;
use super::list::AdjList;

/// Stages edges off-thread, then merges them deterministically.
#[derive(Debug, Default)]
pub struct ParBuilder {
    chunks: Vec<Vec<(VertexId, VertexId)>>,
    n_vertices: usize,
}

impl ParBuilder {
    /// A builder for a graph with `n` vertices and `chunks` staging buffers.
    ///
    /// `chunks` must not depend on the thread pool size: it fixes the merge
    /// order and therefore the edge-id assignment.
    ///
    /// The count is taken literally -- `new(n, 0)` is the same degenerate
    /// builder as [`ParBuilder::default`] and builds an edgeless graph -- so
    /// that [`chunks`](Self::chunks) always reports back what the caller
    /// asked for. A builder whose chunk count was silently rounded up to one
    /// would be a builder whose merge order was not the caller's.
    pub fn new(n_vertices: usize, chunks: usize) -> Self {
        ParBuilder {
            chunks: (0..chunks).map(|_| Vec::new()).collect(),
            n_vertices,
        }
    }

    /// Number of staging buffers.
    #[inline]
    pub fn chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Stage one edge into chunk `k`.
    ///
    /// Endpoints are *not* checked here. They are checked by
    /// [`build`](Self::build), through the same
    /// [`AdjList::add_edge`] bounds test every other caller gets, which keeps
    /// staging a pure `push` and keeps the graph's range check in exactly one
    /// place.
    ///
    /// # Panics
    ///
    /// If `chunk >= self.chunks()`.
    #[inline]
    pub fn push(&mut self, chunk: usize, s: VertexId, t: VertexId) {
        let n = self.chunks.len();
        match self.chunks.get_mut(chunk) {
            Some(buf) => buf.push((s, t)),
            None => panic!("chunk {chunk} out of range for a builder with {n} chunks"),
        }
    }

    /// Fill every chunk in parallel from a per-chunk generator.
    ///
    /// `make(k, buf)` is handed the chunk's index and its buffer, and nothing
    /// else. That is the whole determinism contract, and it is the same one
    /// [`Seed::split`](crate::par::plan::Seed::split) states: a generator that
    /// is a pure function of `(k, buf)` produces the same staged content for
    /// every thread count, whereas one carrying state across chunks -- a
    /// shared RNG, an atomic counter, `rayon::current_thread_index()` -- is
    /// exactly `_rngs[tnum - 1]` (`parallel_rng.hh:56-61`) again.
    ///
    /// The buffer arrives as it is, so `fill` appends: calling it twice adds
    /// two generations of edges to each chunk, and a caller that wants
    /// replacement clears the buffer itself. Chunk *k*'s content always
    /// precedes chunk *k+1*'s in the merge no matter which worker produced
    /// it, or when.
    pub fn fill<F>(&mut self, make: F)
    where
        F: Fn(usize, &mut Vec<(VertexId, VertexId)>) + Sync + Send,
    {
        self.chunks
            .par_iter_mut()
            .enumerate()
            .for_each(|(k, buf)| make(k, buf));
    }

    /// Merge in chunk order. Edge ids are `0..n_edges` in that order.
    ///
    /// The graph starts empty, so [`EdgeIds::alloc`](super::EdgeIds) hands out
    /// `0, 1, 2, ...` with an empty free list; the *i*-th staged pair in chunk
    /// order is therefore edge `i`, and
    /// [`endpoints(EdgeId::from_index(i))`](AdjList::endpoints) reads it back.
    /// graph-tool's concurrent `get_free_idx` (`graph_adjacency.hh:627-655`)
    /// cannot promise this: with `_concurrent` set, `_free_idx_m` is diced up
    /// `pos++ % nt` (`:464`) across `get_num_threads()` lists and the index an
    /// edge receives depends on which worker took it.
    ///
    /// # Errors
    ///
    /// * [`GraphError::VertexIdSpaceExhausted`] if the vertex count does not
    ///   fit the index space. The equivalent bulk `add_vertex(g, n)`
    ///   (`:1318-1333`) is an unchecked `_edges.resize(v + n)`, and `Vertex`
    ///   is also the index type, so the overflowing graph collides silently
    ///   with `null_vertex()`.
    /// * [`GraphError::NoSuchVertex`] if a staged endpoint is outside
    ///   `0..n_vertices`.
    /// * [`GraphError::EdgeIdSpaceExhausted`] if more edges were staged than
    ///   the index space holds.
    ///
    /// Any of them drops the partially merged graph, so a failed `build`
    /// yields no graph rather than a truncated one.
    pub fn build<H: Lookup>(self, lookup: H) -> Result<AdjList<H>, GraphError> {
        // Checked before a block is allocated: `with_vertices_and_lookup`
        // asserts the same bound, and a bulk constructor that aborts the
        // process on a large input is not a constructor a library can offer.
        if self.n_vertices != 0 && self.n_vertices - 1 > MAX_INDEX {
            return Err(GraphError::VertexIdSpaceExhausted { max: MAX_INDEX });
        }
        let mut g = AdjList::with_vertices_and_lookup(self.n_vertices, lookup);
        // `into_iter`, so each staging buffer is freed as soon as it has been
        // spliced: peak residency is the graph plus *one* chunk, not the graph
        // plus the whole staged multiset. At the 10^7-edge rung of
        // `benches/builder.rs` that is 80 MiB of `(s, t)` pairs that do not
        // have to coexist with the 400 MiB of adjacency they become.
        for chunk in self.chunks {
            for (s, t) in chunk {
                g.add_edge(s, t)?;
            }
        }
        Ok(g)
    }
}

// ---------------------------------------------------------------------------
// The thread-count tests live here rather than in `tests/u06_builder.rs`
// because an explicit `rayon::ThreadPool` is the only way to vary the pool
// size inside one process, and `rayon` is a dependency of `gt-core`, not of
// its test binaries. The integration test covers the same claim the way the
// plan words it -- `RAYON_NUM_THREADS` in the environment, which only the
// *global* pool reads, and only once per process -- by re-executing itself.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::adj::{EHash, NoLookup};
    use crate::ids::EdgeId;

    /// Every byte of the built graph that a caller can observe: the counts,
    /// the id-to-endpoint map over the whole index range, and each vertex's
    /// block in storage order with its out/in split. Two graphs with equal
    /// fingerprints have equal adjacency *layout*, not merely equal edge sets
    /// -- which is what "byte-identical" has to mean for a structure whose
    /// insertion order is visible through `out_edges`.
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

    /// SplitMix64 of `(seed, i)`: a chunk can produce its own slice of the
    /// stream from its own index, with no cross-chunk state. This is what
    /// `fill`'s contract asks of a generator.
    fn mix(seed: u64, i: u64) -> u64 {
        let mut z = seed
            .wrapping_add(i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    const N: usize = 512;
    const CHUNKS: usize = 37;
    const PER: usize = 53;

    fn staged() -> ParBuilder {
        let mut b = ParBuilder::new(N, CHUNKS);
        b.fill(|k, out| {
            for j in 0..PER {
                let i = (k * PER + j) as u64;
                // A tenth of the pairs are self-loops and the endpoint space
                // is small enough that parallel edges are common: both are
                // cases where the splice's push-swap moves an entry, so the
                // layout is order-sensitive and the fingerprint has teeth.
                let s = mix(0xA5A5, 2 * i) as usize % N;
                let t = if mix(0xA5A5, 4 * i).is_multiple_of(10) {
                    s
                } else {
                    mix(0xA5A5, 2 * i + 1) as usize % N
                };
                out.push((VertexId::from_index(s), VertexId::from_index(t)));
            }
        });
        b
    }

    #[test]
    fn the_pool_size_changes_nothing() {
        let mut seen: Option<Vec<u64>> = None;
        for threads in [1usize, 4, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let g = pool
                .install(|| staged().build(NoLookup))
                .expect("build must succeed");
            g.validate().expect("the built graph must validate");
            assert_eq!(g.num_edges(), CHUNKS * PER);
            let f = fingerprint(&g);
            match &seen {
                None => seen = Some(f),
                Some(first) => assert_eq!(
                    *first, f,
                    "the graph built on {threads} threads differs from the one built on 1"
                ),
            }
        }
    }

    /// The `(s,t)` index is a second derived structure with its own hooks; it
    /// has to be thread-count-independent too, and `validate()` checks it in
    /// multiplicity rather than membership.
    fn hashed(threads: usize) -> Vec<u64> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        let g = pool
            .install(|| staged().build(EHash::default()))
            .expect("build must succeed");
        g.validate().expect("the built graph must validate");
        fingerprint(&g)
    }

    #[test]
    fn the_pool_size_changes_nothing_with_a_lookup_index() {
        assert_eq!(hashed(1), hashed(16));
    }

    #[test]
    fn fill_appends_rather_than_replacing() {
        let mut b = ParBuilder::new(4, 2);
        let one = |k: usize, out: &mut Vec<(VertexId, VertexId)>| {
            out.push((VertexId::from_index(k), VertexId::from_index(k + 1)));
        };
        b.fill(one);
        b.fill(one);
        let g = b.build(NoLookup).expect("build");
        assert_eq!(g.num_edges(), 4);
        // Chunk order survives the second generation: chunk 0's two edges are
        // ids 0 and 1, chunk 1's are 2 and 3.
        for id in 0..4 {
            let k = id / 2;
            assert_eq!(
                g.endpoints(EdgeId::from_index(id)),
                Some((VertexId::from_index(k), VertexId::from_index(k + 1)))
            );
        }
    }

    #[test]
    fn a_builder_with_no_chunks_builds_the_edgeless_graph() {
        let b = ParBuilder::new(3, 0);
        assert_eq!(b.chunks(), 0);
        let g = b.build(NoLookup).expect("build");
        assert_eq!((g.num_vertices(), g.num_edges()), (3, 0));
        g.validate().expect("validate");
        let d = ParBuilder::default();
        assert_eq!(d.chunks(), 0);
        assert_eq!(d.build(NoLookup).expect("build").num_vertices(), 0);
    }

    #[test]
    #[should_panic(expected = "chunk 2 out of range")]
    fn pushing_past_the_last_chunk_panics() {
        let mut b = ParBuilder::new(2, 2);
        b.push(2, VertexId::from_index(0), VertexId::from_index(1));
    }
}
