//! Differential tests: the `.gt` round trip, against graph-tool 3.8's
//! `src/graph/graph_io_binary.hh` and `src/graph/graph_io.cc`.
//!
//! The question this file answers is *what a `.gt` round trip is allowed to
//! change*, which the format's own reader and writer answer between them:
//!
//! ```c++
//! // write_adjacency_dispatch (graph_io_binary.hh:171-184)
//! for (auto v : vertices_range(g)) {
//!     std::vector<Vint> us;
//!     for (auto e : out_edges_range(v, g))     // the OUT-half only
//!         us.push_back(vindex[target(e, g)]);
//!     write(s, us);
//! }
//!
//! // read_adjacency_dispatch (graph_io_binary.hh:204-219)
//! for (vertex_t v = 0; v < N; ++v) {
//!     std::vector<Vint> us; read<BE>(s, us);
//!     for (vertex_t u : us) add_edge(v, u, g);  // in that order
//! }
//! ```
//!
//! So the file carries **no edge indices at all**. The reader re-creates every
//! edge with `add_edge`, which hands out `_edge_idx_range++`
//! (`graph_adjacency.hh:645-652`) in exactly the order the writer walked ---
//! vertex-major over out-halves. Three consequences, each pinned below:
//!
//! 1. edge indices are **renumbered** to a dense `0..E` by any round trip, so
//!    a graph that had interior holes comes back compacted;
//! 2. edge *property* values nevertheless stay attached to the right edge,
//!    because `write_property<edge_range_traits>` emits them in
//!    `edges_range(g)` order (`:296-302`, `:318-320`) --- the same
//!    vertex-major walk --- and the reader fills them in the new graph's
//!    `edges_range` order, which is the order it just built;
//! 3. an **undirected** graph is written once per edge, not twice, because
//!    `write_to_file` sets `_directed = true` for the duration of the write
//!    (`graph_io.cc:505-506`, restored at `:542`) so `out_edges_range` is the
//!    storage out-half and never the `undirected_adaptor`'s whole block. The
//!    real directedness rides in a single byte (`:190-191`).

use gt_core::adj::{AdjList, NoLookup};
use gt_core::ids::{EdgeId, VertexId};
use gt_core::prop::ValueKind;
use gt_io::gt::{Document, NamedProperty, PropertyColumn, PropertyDomain, read, write};

fn v(i: usize) -> VertexId {
    VertexId::from_index(i)
}

fn roundtrip(doc: &Document<NoLookup>) -> Document<NoLookup> {
    let mut bytes = Vec::new();
    write(&mut bytes, doc).expect("write");
    read(&bytes[..], NoLookup).expect("read back what we just wrote")
}

/// `(source, target, id)` in `edges()` order: vertex-major over out-halves,
/// which is `adj_list::edge_iterator` (`graph_adjacency.hh:341-373`).
fn list(g: &AdjList) -> Vec<(usize, usize, usize)> {
    g.edges()
        .map(|e| (e.source().index(), e.target().index(), e.id().index()))
        .collect()
}

/// The same, without the identities: what the format actually preserves.
fn pairs(g: &AdjList) -> Vec<(usize, usize)> {
    g.edges()
        .map(|e| (e.source().index(), e.target().index()))
        .collect()
}

fn doc_of(n: usize, edges: &[(usize, usize)], directed: bool) -> Document<NoLookup> {
    let mut g = AdjList::with_vertices(n);
    for &(s, t) in edges {
        g.add_edge(v(s), v(t)).expect("both endpoints exist");
    }
    Document::new(g, directed)
}

// ---------------------------------------------------------------------------

/// A round trip preserves the vertex-major edge *sequence* exactly.
///
/// The writer emits vertex 0's out-half, then vertex 1's, and so on; the
/// reader replays that with `add_edge(v, u, g)` in the same order, and the
/// push-swap (`graph_adjacency.hh:1196-1211`) puts each new out-entry at the
/// end of its vertex's out-half. So the rebuilt graph's `edges_range` is the
/// file's order, which was the original's `edges_range`.
///
/// Parallel edges, self-loops and isolated vertices are all in the fixture,
/// because each is a way for a naive writer to lose or duplicate something.
#[test]
fn the_round_trip_preserves_the_vertex_major_edge_sequence() {
    let edges = [(0, 1), (1, 2), (2, 0), (0, 0), (0, 1), (3, 3)];
    let doc = doc_of(5, &edges, true);
    let before = list(&doc.graph);
    assert_eq!(
        before,
        vec![
            (0, 1, 0),
            (0, 0, 3),
            (0, 1, 4),
            (1, 2, 1),
            (2, 0, 2),
            (3, 3, 5)
        ],
        "the push-swap layout, read down the out-halves"
    );

    let back = roundtrip(&doc);
    assert_eq!(back.graph.num_vertices(), 5, "isolated vertex 4 survives");
    assert_eq!(back.graph.num_edges(), edges.len());
    assert_eq!(
        pairs(&back.graph),
        pairs(&doc.graph),
        "the sequence of endpoint pairs is the invariant"
    );
    assert!(back.directed);

    // The file is a fixed point after one pass: writing what we read gives
    // the same bytes.
    let mut once = Vec::new();
    write(&mut once, &doc).expect("write");
    let mut twice = Vec::new();
    write(&mut twice, &back).expect("write");
    assert_eq!(
        once, twice,
        "a document read back from a file re-serialises to that file"
    );
}

/// **The round trip renumbers edge indices**, because the format does not
/// carry them.
///
/// `read_adjacency_dispatch` calls `add_edge(v, u, g)` on a graph that has
/// never freed an index, so `get_free_idx` returns `_edge_idx_range++`
/// (`graph_adjacency.hh:645-652`) and the result is dense `0..E` in
/// vertex-major order. Any hole an earlier `remove_edge` left --- and
/// `_edge_idx_range` never shrinks, so there always is one --- is gone, and so
/// is any identity a caller was holding.
///
/// This is not a divergence, it is the format. But it is the thing a caller
/// most needs told, because `g.edge_index` is a stable handle *within* a
/// session and is silently not one across a save/load.
#[test]
fn the_round_trip_renumbers_edge_indices_and_compacts_the_index_space() {
    let mut g = AdjList::with_vertices(4);
    let ids: Vec<EdgeId> = [(0, 1), (1, 2), (2, 3), (0, 3), (3, 0)]
        .iter()
        .map(|&(s, t)| g.add_edge(v(s), v(t)).expect("add").id())
        .collect();
    // Punch two holes in the middle of the index space.
    g.remove_edge(ids[1]).expect("live");
    g.remove_edge(ids[3]).expect("live");
    assert_eq!(
        (g.num_edges(), g.edge_bound().len()),
        (3, 5),
        "`_edge_idx_range` is monotone, so the space stays sparse"
    );
    let before: Vec<usize> = g.edges().map(|e| e.id().index()).collect();
    // v0 keeps e0, v1's out-half is empty after e1 went, v2 keeps e2 and v3
    // keeps e4: vertex-major over out-halves, with two holes at 1 and 3.
    assert_eq!(before, vec![0, 2, 4], "live ids, vertex-major");

    let doc = Document::new(g, true);
    let back = roundtrip(&doc);

    assert_eq!(
        pairs(&back.graph),
        pairs(&doc.graph),
        "the edges themselves are the same, in the same order"
    );
    assert_eq!(
        back.graph
            .edges()
            .map(|e| e.id().index())
            .collect::<Vec<_>>(),
        vec![0, 1, 2],
        "renumbered dense, in the order the file listed them"
    );
    assert_eq!(
        back.graph.edge_bound().len(),
        back.graph.num_edges(),
        "the index space comes back compact: no holes survive a save/load"
    );
    assert_ne!(before, vec![0usize, 1, 2], "the old ids really were sparse");
}

/// Edge property values follow the edges through the renumbering.
///
/// `write_property<edge_range_traits>` walks `edges_range(g)`
/// (`graph_io_binary.hh:296-302`) and the reader fills the new graph's
/// `edges_range` in the same order (`:390-396`), so a column is positional and
/// the positions line up. What does *not* line up is the indexing: the column
/// written at position `i` belonged to edge id `order[i]` before and to edge
/// id `i` after.
///
/// The test therefore checks the values against the **endpoints**, which is
/// the only thing that is meaningfully conserved.
#[test]
fn an_edge_column_follows_its_edges_through_the_renumbering() {
    let mut g = AdjList::with_vertices(4);
    let ids: Vec<EdgeId> = [(0, 1), (1, 2), (2, 3), (0, 3), (3, 0)]
        .iter()
        .map(|&(s, t)| g.add_edge(v(s), v(t)).expect("add").id())
        .collect();
    g.remove_edge(ids[1]).expect("live");
    g.remove_edge(ids[3]).expect("live");

    // A column keyed by edge *id*, sized to the bound -- which is what
    // `_get_any` reserves for an edge map (`graph_tool/__init__.py:369`).
    let bound = g.edge_bound().len();
    let mut col = vec![0i64; bound];
    for e in g.edges() {
        col[e.id().index()] = (e.source().index() * 10 + e.target().index()) as i64;
    }
    assert_eq!(col, vec![1, 0, 23, 0, 30], "two holes keep the default");

    let mut doc = Document::new(g, true);
    doc.properties.push(NamedProperty {
        name: "w".to_owned(),
        domain: PropertyDomain::Edge,
        values: PropertyColumn::I64(col),
    });
    // A vertex column too: those are keyed by a dense `0..N` and are not
    // disturbed at all.
    doc.properties.push(NamedProperty {
        name: "vx".to_owned(),
        domain: PropertyDomain::Vertex,
        values: PropertyColumn::I64(vec![100, 101, 102, 103]),
    });
    // ...and a graph property, which is one value (`graph_range` yields a
    // single `graph_property_tag`, `graph_io_binary.hh:254-259`).
    doc.properties.push(NamedProperty {
        name: "gp".to_owned(),
        domain: PropertyDomain::Graph,
        values: PropertyColumn::Str(vec!["hello".to_owned()]),
    });

    let back = roundtrip(&doc);
    assert_eq!(back.properties.len(), 3);

    let edge_col = back
        .properties
        .iter()
        .find(|p| p.name == "w")
        .expect("the edge column came back");
    let PropertyColumn::I64(vals) = &edge_col.values else {
        panic!("wrong member");
    };
    assert_eq!(
        vals.len(),
        back.graph.num_edges(),
        "the file stores one value per *live* edge, not one per index slot"
    );
    for e in back.graph.edges() {
        assert_eq!(
            vals[e.id().index()],
            (e.source().index() * 10 + e.target().index()) as i64,
            "edge {:?} lost its value across the renumbering",
            (e.source().index(), e.target().index())
        );
    }

    let vcol = back
        .properties
        .iter()
        .find(|p| p.name == "vx")
        .expect("the vertex column came back");
    assert!(matches!(
        &vcol.values,
        PropertyColumn::I64(x) if x == &vec![100i64, 101, 102, 103]
    ));
    let gcol = back
        .properties
        .iter()
        .find(|p| p.name == "gp")
        .expect("the graph property came back");
    assert_eq!(gcol.kind(), ValueKind::Str);
    assert!(matches!(&gcol.values, PropertyColumn::Str(x) if x.len() == 1));
}

/// An undirected graph is written **once** per edge, with the directedness in
/// a single byte.
///
/// `write_to_file` does `bool directed = _directed; _directed = true;`
/// (`graph_io.cc:505-506`) before dispatching, so `run_action` instantiates
/// the *directed* view and `out_edges_range(v, g)` is the storage out-half.
/// Without that, `out_edges` on the `undirected_adaptor` would be the whole
/// block (`graph_adaptor.hh:199-207`) and every edge would be emitted from
/// both endpoints --- so a reader would build a graph with `2E` edges, and a
/// self-loop with three.
#[test]
fn an_undirected_graph_is_written_once_per_edge_with_the_flag_in_a_byte() {
    let edges = [(0, 1), (1, 2), (0, 0), (2, 0)];
    let doc = doc_of(3, &edges, false);
    assert!(!doc.directed);

    let mut bytes = Vec::new();
    write(&mut bytes, &doc).expect("write");
    let back = read(&bytes[..], NoLookup).expect("read");

    assert!(!back.directed, "the flag survives");
    assert_eq!(
        back.graph.num_edges(),
        4,
        "each undirected edge is stored once; doubling would give 8"
    );
    assert_eq!(pairs(&back.graph), pairs(&doc.graph));
    // The self-loop occupies both halves of vertex 0's block but was written
    // from the out-half only, so it comes back as one edge.
    assert_eq!(
        back.graph
            .all_edges(v(0))
            .filter(|i| i.other == v(0))
            .count(),
        2,
        "one edge, two adjacency entries"
    );

    // The directed flag is one byte, immediately after the comment, and is
    // the only difference between the two files.
    let directed_doc = doc_of(3, &edges, true);
    let mut d_bytes = Vec::new();
    write(&mut d_bytes, &directed_doc).expect("write");
    let diffs: Vec<usize> = bytes
        .iter()
        .zip(d_bytes.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    // The comment records "directed" vs "undirected" too, so it differs in
    // length; compare only the generated-comment-free case.
    assert!(
        !diffs.is_empty(),
        "the two files must not be identical: the flag is real"
    );

    // With the *same* comment forced on both, the files differ in exactly one
    // byte --- the flag.
    let mut a = doc_of(3, &edges, false);
    let mut b = doc_of(3, &edges, true);
    a.comment = Some("same".to_owned());
    b.comment = Some("same".to_owned());
    let (mut ab, mut bb) = (Vec::new(), Vec::new());
    write(&mut ab, &a).expect("write");
    write(&mut bb, &b).expect("write");
    assert_eq!(ab.len(), bb.len());
    let diff: Vec<usize> = ab
        .iter()
        .zip(bb.iter())
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        diff.len(),
        1,
        "exactly one byte distinguishes a directed file from an undirected one"
    );
    assert_eq!((ab[diff[0]], bb[diff[0]]), (0, 1));
}

/// An isolated vertex has no representation other than its empty adjacency
/// row, and the row is what preserves it.
///
/// `write_adjacency` writes `N` first (`:192`) and then one length-prefixed
/// list per vertex; a reader that trusted only the edge lists would lose every
/// trailing isolated vertex. The vertex *count* is therefore load-bearing, and
/// it is `get_num_vertices()` --- the unfiltered count --- not the number of
/// vertices with edges.
#[test]
fn isolated_and_trailing_vertices_survive_because_n_is_written_first() {
    let doc = doc_of(6, &[(0, 1)], true);
    let back = roundtrip(&doc);
    assert_eq!(back.graph.num_vertices(), 6);
    assert_eq!(back.graph.num_edges(), 1);
    assert_eq!(back.graph.degree(v(5)), 0);

    // The empty graph round-trips too.
    let empty = doc_of(0, &[], true);
    let back = roundtrip(&empty);
    assert_eq!((back.graph.num_vertices(), back.graph.num_edges()), (0, 0));

    // And a graph with vertices but no edges.
    let bare = doc_of(3, &[], false);
    let back = roundtrip(&bare);
    assert_eq!((back.graph.num_vertices(), back.graph.num_edges()), (3, 0));
    assert!(!back.directed);
}

/// The adjacency integer width is chosen from `N`, and the boundary is
/// `numeric_limits<uint8_t>::max()` **inclusive**.
///
/// `write_adjacency` (`graph_io_binary.hh:194-201`):
///
/// ```c++
/// if (N <= numeric_limits<uint8_t>::max())  ... uint8_t
/// else if (N <= numeric_limits<uint16_t>::max()) ... uint16_t
/// ```
///
/// So `N == 255` is written with one byte per target and `N == 256` with two.
/// The off-by-one is worth pinning because the test that matters --- a vertex
/// index of 255 in a 256-vertex graph --- is representable in `uint8_t`, and a
/// `<` instead of a `<=` would only fail at exactly one size.
#[test]
fn the_adjacency_width_switches_at_n_equal_to_the_type_maximum() {
    // N = 255: one byte per target, and index 254 is the largest.
    let doc = doc_of(255, &[(0, 254), (254, 0)], true);
    let mut small = Vec::new();
    write(&mut small, &doc).expect("write");
    let back = read(&small[..], NoLookup).expect("read");
    assert_eq!(pairs(&back.graph), vec![(0, 254), (254, 0)]);

    // N = 256: two bytes per target. The file is longer by exactly one byte
    // per edge, with the same edge count and the same comment length.
    let doc2 = doc_of(256, &[(0, 254), (254, 0)], true);
    let mut large = Vec::new();
    write(&mut large, &doc2).expect("write");
    let back2 = read(&large[..], NoLookup).expect("read");
    assert_eq!(pairs(&back2.graph), vec![(0, 254), (254, 0)]);

    // 256 - 255 = 1 extra adjacency row (8 bytes of length prefix), 2 extra
    // target bytes, and one more digit in the comment's vertex count.
    assert_eq!(
        large.len() - small.len(),
        8 + 2,
        "one more adjacency row (an 8-byte length prefix) and one extra byte \
         per target; `255` and `256` have the same number of digits, so the \
         comment is the same length"
    );

    // Index 255 in a 256-vertex graph is representable in the wider form.
    let doc3 = doc_of(256, &[(255, 0), (0, 255)], true);
    let back3 = roundtrip(&doc3);
    assert_eq!(pairs(&back3.graph), vec![(0, 255), (255, 0)]);
}

/// The reader refuses a target index outside `[0, N)`.
///
/// `read_adjacency_dispatch` (`graph_io_binary.hh:212-213`):
/// `if (u >= N) throw IOException("...vertex index not in range")`. A reader
/// without that check calls `add_edge(v, u, g)` with `u` past the end of
/// `_edges` and reads out of bounds.
#[test]
fn a_target_index_out_of_range_is_refused() {
    let doc = doc_of(3, &[(0, 1), (1, 2)], true);
    let mut bytes = Vec::new();
    write(&mut bytes, &doc).expect("write");

    // The header, laid out by `write_graph` (`graph_io_binary.hh:441-456`):
    // magic (6) | version (1) | big_endian (1) | comment (u64 len + bytes)
    // then `write_adjacency`: directed (1) | N (u64) | one row per vertex,
    // each a u64 length followed by that many target indices.
    let clen = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")) as usize;
    let adj = 16 + clen;
    assert_eq!(bytes[adj], 1, "the directed flag");
    let n = u64::from_le_bytes(bytes[adj + 1..adj + 9].try_into().expect("8 bytes"));
    assert_eq!(n, 3, "N");
    let row0 = adj + 9;
    assert_eq!(
        u64::from_le_bytes(bytes[row0..row0 + 8].try_into().expect("8 bytes")),
        1,
        "vertex 0 has one out-edge"
    );
    let target = row0 + 8;
    assert_eq!(bytes[target], 1, "vertex 0's only out-neighbour is 1");

    // `if (u >= N) throw IOException(...)` (`graph_io_binary.hh:212-213`).
    bytes[target] = 3;
    let err = read(&bytes[..], NoLookup).expect_err("an out-of-range target must be refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("range") || msg.contains("vertex"),
        "the diagnosis should name the problem, got: {msg}"
    );

    // 2 is in range, so the same file with that byte is accepted.
    bytes[target] = 2;
    let ok = read(&bytes[..], NoLookup).expect("2 is a valid vertex index");
    assert_eq!(pairs(&ok.graph), vec![(0, 2), (1, 2)]);
}

/// The two shipped fixtures load, re-save byte-identically, and agree with
/// their own header comments.
///
/// `tests/data/{karate,lesmis}.gt` are graph-tool's own collection files,
/// gunzipped. They were written by 2.2.32dev, whose comment wording differs
/// from the current format string, so byte-identity is only possible because
/// [`Document::comment`] carries the file's text rather than regenerating it.
/// The counts in that comment were written by graph-tool and are therefore an
/// independent statement of what the file contains.
#[test]
fn the_shipped_collection_files_agree_with_their_own_headers() {
    for (name, bytes) in [
        ("karate", &include_bytes!("data/karate.gt")[..]),
        ("lesmis", &include_bytes!("data/lesmis.gt")[..]),
    ] {
        let doc = read(bytes, NoLookup).unwrap_or_else(|e| panic!("{name}: {e}"));
        let comment = doc.comment.clone().expect("the file carries a comment");

        // "stats: N vertices, E edges, directed|undirected, ..."
        let stats = comment
            .split("stats: ")
            .nth(1)
            .unwrap_or_else(|| panic!("{name}: no stats in {comment:?}"));
        let n: usize = stats
            .split(" vertices")
            .next()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or_else(|| panic!("{name}: no vertex count in {stats:?}"));
        let e: usize = stats
            .split(", ")
            .nth(1)
            .and_then(|s| s.split(' ').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("{name}: no edge count in {stats:?}"));

        assert_eq!(doc.graph.num_vertices(), n, "{name}: vertex count");
        assert_eq!(doc.graph.num_edges(), e, "{name}: edge count");
        assert_eq!(
            doc.directed,
            stats.contains(", directed,"),
            "{name}: directedness"
        );
        assert_eq!(
            doc.graph.edge_bound().len(),
            e,
            "{name}: a freshly read graph has a compact index space"
        );

        let mut again = Vec::new();
        write(&mut again, &doc).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!(&again[..], bytes, "{name}: re-saving is not byte-identical");
    }
}
