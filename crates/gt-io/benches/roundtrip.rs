//! `.gt` write/read round-trip.
//!
//! `gt_core::design` defect #52: this port removes edges with swap-with-back where
//! `graph_adjacency.hh:1257-1263` erases and shifts, so **adjacency order
//! after a removal differs from graph-tool's**. The port's answer is that
//! `gt::write` emits in [`EdgeId`](gt_core::ids::EdgeId) order rather than
//! adjacency order, which makes the output a function of the graph and not of
//! its removal history.
//!
//! That is a correctness decision with a cost — a pass over the edge set in id
//! order instead of a straight walk of the adjacency stream — and section 12
//! is a ledger, so the cost is measured. `write/dense` is the graph as built;
//! `write/after_removals` is the same graph with a fifth of its edges removed,
//! which is where the two orders actually diverge and where the reordering
//! pass has the least locality.
//!
//! `read` is measured separately rather than only as a round trip, because the
//! two halves have different bottlenecks and a single round-trip number hides
//! which one moved.
//!
//! U27 owns the format.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use gt_core::adj::{AdjList, NoLookup};
use gt_core::ids::{EdgeId, VertexId};
use gt_io::gt::{Document, read, write};

const N: usize = 50_000;
const M: usize = 250_000;

fn build(n: usize, m: usize) -> (AdjList<NoLookup>, Vec<EdgeId>) {
    let mut s = 0x243f_6a88_85a3_08d3u64;
    let mut next = move || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let mut g = AdjList::with_vertices(n);
    let mut ids = Vec::with_capacity(m);
    for _ in 0..m {
        let a = VertexId::from_index((next() % n as u64) as usize);
        let b = VertexId::from_index((next() % n as u64) as usize);
        ids.push(g.add_edge(a, b).expect("add_edge").id());
    }
    (g, ids)
}

fn document(graph: AdjList<NoLookup>) -> Document<NoLookup> {
    Document {
        graph,
        directed: true,
        // No property maps: this file is about the graph half of the format.
        // Property serialisation is per-`ValueKind` and belongs in its own
        // group once the fifteen members are written.
        properties: Vec::new(),
        // U27: the header comment is preserved verbatim on read so that a
        // file written by an older graph-tool re-saves byte-identically;
        // `None` means "generate this version's".
        comment: None,
    }
}

fn writing(c: &mut Criterion) {
    let (dense, ids) = build(N, M);
    let sparse = {
        let (mut g, ids) = build(N, M);
        for id in ids.iter().step_by(5) {
            g.remove_edge(*id).expect("remove_edge");
        }
        g
    };
    let removed = M - M.div_ceil(5);
    let dense_doc = document(dense);
    let sparse_doc = document(sparse);
    let _ = ids;

    let mut group = c.benchmark_group("write");
    group.throughput(Throughput::Elements(M as u64));
    group.bench_function("dense", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(1 << 22);
            write(&mut out, &dense_doc).expect("write");
            out
        });
    });
    group.throughput(Throughput::Elements(removed as u64));
    group.bench_function("after_removals", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(1 << 22);
            write(&mut out, &sparse_doc).expect("write");
            out
        });
    });
    group.finish();
}

fn reading(c: &mut Criterion) {
    let (g, _) = build(N, M);
    let doc = document(g);
    let mut bytes = Vec::with_capacity(1 << 22);
    write(&mut bytes, &doc).expect("write");

    let mut group = c.benchmark_group("read");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("gt", |b| {
        b.iter(|| {
            let doc = read(bytes.as_slice(), NoLookup).expect("read");
            black_box(doc.properties.len())
        });
    });
    group.finish();
}

criterion_group!(roundtrip, writing, reading);
criterion_main!(roundtrip);
