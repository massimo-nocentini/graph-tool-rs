//! U27 acceptance: the `.gt` binary format.
//!
//! The reference is `src/graph/graph_io_binary.hh`. Every assertion here is
//! either a layout fact taken from that header (cited at the assertion) or a
//! property of the round trip that the header's own reader depends on.
//!
//! The two fixtures in `tests/data/` are **not** synthesised: they are
//! `src/graph_tool/collection/karate.gt.gz` and `lesmis.gt.gz` as shipped,
//! gunzipped. They were written by graph-tool 2.2.32dev, so their header
//! comment is not the one this version generates -- which is precisely why
//! `Document::comment` exists, and why "load and re-save byte-identically" is
//! a real test rather than a tautology.

use std::collections::BTreeMap;

use gt_core::adj::{AdjList, NoLookup};
use gt_core::ids::{EdgeId, VertexId};
use gt_core::prop::{LongDouble, ValueKind};
use gt_io::IoError;
use gt_io::gt::{
    ColumnValue, Document, MAGIC, NamedProperty, PropertyColumn, PropertyDomain, VERSION, read,
    write,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn to_bytes(doc: &Document<NoLookup>) -> Vec<u8> {
    let mut out = Vec::new();
    write(&mut out, doc).expect("write");
    out
}

fn ld(seed: u8) -> LongDouble {
    let mut b = [0u8; 16];
    for (i, x) in b.iter_mut().enumerate() {
        *x = seed.wrapping_mul(31).wrapping_add(i as u8);
    }
    LongDouble(b)
}

/// One column of `n` values of the given member, deterministic in `n` and the
/// member, and deliberately awkward: empty inner vectors, an empty string, a
/// NaN, a subnormal, and a non-ASCII string all appear.
fn column(kind: ValueKind, n: usize) -> PropertyColumn {
    let s = |i: usize| match i % 4 {
        0 => String::new(),
        1 => format!("value {i}"),
        2 => "héllo ⛾ wörld".to_owned(),
        _ => "\u{0}embedded nul".to_owned(),
    };
    let f = |i: usize| match i % 5 {
        0 => 0.0,
        1 => -0.0,
        2 => f64::NAN,
        3 => f64::from_bits(1), // subnormal
        _ => i as f64 * 1.5,
    };
    match kind {
        ValueKind::Bool => PropertyColumn::Bool((0..n).map(|i| (i % 2) as u8).collect()),
        ValueKind::I16 => PropertyColumn::I16((0..n).map(|i| i as i16 - 7).collect()),
        ValueKind::I32 => PropertyColumn::I32((0..n).map(|i| i as i32 * -1000).collect()),
        ValueKind::I64 => PropertyColumn::I64((0..n).map(|i| i as i64 * 1_000_000_007).collect()),
        ValueKind::F64 => PropertyColumn::F64((0..n).map(f).collect()),
        ValueKind::LongDouble => PropertyColumn::LongDouble((0..n).map(|i| ld(i as u8)).collect()),
        ValueKind::Str => PropertyColumn::Str((0..n).map(s).collect()),
        ValueKind::VecBool => PropertyColumn::VecBool(
            (0..n)
                .map(|i| (0..i % 3).map(|j| j as u8).collect())
                .collect(),
        ),
        ValueKind::VecI16 => PropertyColumn::VecI16(
            (0..n)
                .map(|i| (0..i % 4).map(|j| j as i16 - 2).collect())
                .collect(),
        ),
        ValueKind::VecI32 => PropertyColumn::VecI32(
            (0..n)
                .map(|i| (0..i % 5).map(|j| j as i32 - 2).collect())
                .collect(),
        ),
        ValueKind::VecI64 => PropertyColumn::VecI64(
            (0..n)
                .map(|i| (0..i % 3).map(|j| j as i64 * -5).collect())
                .collect(),
        ),
        ValueKind::VecF64 => {
            PropertyColumn::VecF64((0..n).map(|i| (0..i % 4).map(f).collect()).collect())
        }
        ValueKind::VecLongDouble => PropertyColumn::VecLongDouble(
            (0..n)
                .map(|i| (0..i % 3).map(|j| ld((i + j) as u8)).collect())
                .collect(),
        ),
        ValueKind::VecStr => {
            PropertyColumn::VecStr((0..n).map(|i| (0..i % 4).map(s).collect()).collect())
        }
        // The format stores a pickle payload (`graph_io_binary.hh:86-90` ->
        // `graph_io.cc:62-68` -> `gt_io.py:63-67`), so these are real protocol-5
        // pickles of small objects: `pickle.dumps(i, -1)`.
        ValueKind::PyObject => PropertyColumn::PyObject(
            (0..n)
                .map(|i| {
                    let mut v = vec![0x80, 0x05, 0x4b, i as u8];
                    v.extend_from_slice(b".");
                    v
                })
                .collect(),
        ),
    }
}

/// Five vertices, eight edges, including a self-loop, a parallel pair and a
/// vertex with no out-edges -- the three shapes that make the per-vertex
/// adjacency lists non-uniform.
fn sample_graph() -> AdjList<NoLookup> {
    let mut g = AdjList::with_vertices(5);
    let v = |i: usize| VertexId::from_index(i);
    for (a, b) in [
        (0, 1),
        (0, 1), // parallel
        (1, 2),
        (2, 2), // self-loop
        (3, 0),
        (3, 4),
        (0, 4),
        (4, 3),
    ] {
        g.add_edge(v(a), v(b)).expect("add_edge");
    }
    g
}

/// Every property of every member, on every domain: 45 columns.
fn sample_document() -> Document<NoLookup> {
    let g = sample_graph();
    let (nv, ne) = (g.num_vertices(), g.num_edges());
    let mut properties = Vec::new();
    for (domain, n) in [
        (PropertyDomain::Graph, 1),
        (PropertyDomain::Vertex, nv),
        (PropertyDomain::Edge, ne),
    ] {
        for kind in ValueKind::ALL {
            properties.push(NamedProperty {
                name: format!("{:?}_{}", domain, kind.name()),
                domain,
                values: column(kind, n),
            });
        }
    }
    Document {
        graph: g,
        directed: true,
        properties,
        comment: None,
    }
}

/// The order `write` emits edges in, and therefore the order the reader
/// re-assigns ids in: out-edges of each vertex, ascending vertex, ascending
/// `EdgeId` within a vertex.
fn traversal(g: &AdjList<NoLookup>) -> Vec<usize> {
    let mut out = Vec::with_capacity(g.num_edges());
    for v in g.vertices() {
        let mut e: Vec<EdgeId> = g.out_edges(v).map(|i| i.edge).collect();
        e.sort_unstable();
        out.extend(e.iter().map(|id| id.index()));
    }
    out
}

fn adjacency(g: &AdjList<NoLookup>) -> Vec<Vec<usize>> {
    g.vertices()
        .map(|v| {
            let mut e: Vec<EdgeId> = g.out_edges(v).map(|i| i.edge).collect();
            e.sort_unstable();
            e.iter()
                .map(|id| g.endpoints(*id).expect("live edge").1.index())
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Layout facts
// ---------------------------------------------------------------------------

/// `_magic` and `_magic_length` (`graph_io_binary.hh:30-32`).
#[test]
fn the_header_is_the_bytes_the_c_writes() {
    assert_eq!(MAGIC, b"\xe2\x9b\xbe gt");
    assert_eq!(MAGIC.len(), 6);
    assert_eq!(VERSION, 1);

    let doc = Document::new(AdjList::with_vertices(3), false);
    let b = to_bytes(&doc);
    assert_eq!(&b[..6], MAGIC);
    assert_eq!(b[6], 1, "version");
    assert_eq!(
        b[7],
        u8::from(cfg!(target_endian = "big")),
        "is_bigendian() of the writing host (`:447`)"
    );
}

/// `write_graph`'s comment (`:449-456`), regenerated when the document does
/// not carry one.
#[test]
fn the_generated_comment_is_the_c_format_string() {
    let mut doc = Document::new(sample_graph(), false);
    doc.properties.push(NamedProperty {
        name: "x".into(),
        domain: PropertyDomain::Vertex,
        values: column(ValueKind::I64, 5),
    });
    let comment_of = |b: &[u8]| {
        let len = u64::from_ne_bytes(b[8..16].try_into().unwrap()) as usize;
        std::str::from_utf8(&b[16..16 + len]).unwrap().to_owned()
    };

    let b = to_bytes(&doc);
    assert_eq!(
        comment_of(&b),
        "graph-tool binary file version 1 (https://graph-tool.skewed.de), \
         stats: 5 vertices, 8 edges, undirected, 0 graph props, 1 vertex props, 0 edge props"
    );

    doc.directed = true;
    let comment = comment_of(&to_bytes(&doc));
    assert!(
        comment.contains("8 edges, directed, 0 graph props"),
        "{comment}"
    );

    // A comment the document carries is emitted verbatim, which is the only
    // way a file written by an older graph-tool re-saves unchanged.
    doc.comment = Some("written by something else entirely".into());
    let b = to_bytes(&doc);
    assert_eq!(comment_of(&b), "written by something else entirely");
    let back = read(b.as_slice(), NoLookup).expect("read");
    assert_eq!(
        back.comment.as_deref(),
        Some("written by something else entirely")
    );
    assert_eq!(to_bytes(&back), b);
}

/// `write_adjacency` picks the narrowest of `uint8/16/32/64_t` that can hold
/// an index below `N` (`:194-201`), and `read_adjacency` re-derives the same
/// choice from `N` alone (`:234-241`). The boundary is `N <= 255`, not
/// `N < 255`.
#[test]
fn the_index_width_switches_at_the_c_boundaries() {
    for (n, width) in [(1usize, 1), (255, 1), (256, 2), (65_535, 2), (65_536, 4)] {
        let mut g = AdjList::with_vertices(n);
        g.add_edge(VertexId::from_index(0), VertexId::from_index(n - 1))
            .expect("add_edge");
        let b = to_bytes(&Document::new(g, true));

        let clen = u64::from_ne_bytes(b[8..16].try_into().unwrap()) as usize;
        let adj = 16 + clen;
        assert_eq!(b[adj], 1, "directed byte");
        assert_eq!(
            u64::from_ne_bytes(b[adj + 1..adj + 9].try_into().unwrap()),
            n as u64
        );
        // Vertex 0's list: a u64 count of 1 followed by one index of `width`
        // bytes whose value is n-1.
        let list = adj + 9;
        assert_eq!(u64::from_ne_bytes(b[list..list + 8].try_into().unwrap()), 1);
        let mut idx = [0u8; 8];
        idx[..width].copy_from_slice(&b[list + 8..list + 8 + width]);
        assert_eq!(
            u64::from_ne_bytes(idx),
            (n - 1) as u64,
            "N = {n} should use a {width}-byte index"
        );

        let back = read(b.as_slice(), NoLookup).expect("read");
        assert_eq!(back.graph.num_vertices(), n);
        assert_eq!(back.graph.num_edges(), 1);
    }
}

/// The three `property_type` discriminants (`:248-253`) and the value-type
/// index, which is the position in `val_types` (`:257`, `:317`) and therefore
/// in `value_types` (`graph_properties.hh:61-69`).
#[test]
fn the_domain_and_value_type_bytes_are_the_c_discriminants() {
    assert_eq!(PropertyDomain::Graph as u8, 0);
    assert_eq!(PropertyDomain::Vertex as u8, 1);
    assert_eq!(PropertyDomain::Edge as u8, 2);
    assert_eq!(PropertyDomain::from_byte(3), None);

    for (i, kind) in ValueKind::ALL.iter().enumerate() {
        assert_eq!(*kind as u8, i as u8, "{}", kind.name());
    }

    let mut doc = Document::new(AdjList::with_vertices(2), true);
    doc.properties.push(NamedProperty {
        name: "w".into(),
        domain: PropertyDomain::Vertex,
        values: column(ValueKind::VecF64, 2),
    });
    let b = to_bytes(&doc);
    // ... nprops, then: domain, name, type index.
    // domain(1) + name length(8) + "w"(1) + type index(1) + two vector<double>
    // values, the first empty and the second one element: 8 + (8 + 8).
    let tail = &b[b.len() - (1 + 8 + 1 + 1 + 8 + 8 + 8)..];
    assert_eq!(tail[0], PropertyDomain::Vertex as u8);
    assert_eq!(u64::from_ne_bytes(tail[1..9].try_into().unwrap()), 1);
    assert_eq!(tail[9], b'w');
    assert_eq!(tail[10], ValueKind::VecF64 as u8);
}

// ---------------------------------------------------------------------------
// Acceptance: the fifteen-member round trip
// ---------------------------------------------------------------------------

#[test]
fn all_fifteen_members_round_trip_byte_identically() {
    let doc = sample_document();
    assert_eq!(doc.properties.len(), 45);

    let first = to_bytes(&doc);
    let back = read(first.as_slice(), NoLookup).expect("read");
    let second = to_bytes(&back);

    assert_eq!(first, second, "write . read . write != write");

    assert_eq!(back.directed, doc.directed);
    assert_eq!(back.graph.num_vertices(), doc.graph.num_vertices());
    assert_eq!(back.graph.num_edges(), doc.graph.num_edges());
    assert_eq!(adjacency(&back.graph), adjacency(&doc.graph));
    back.graph.validate().expect("validate");

    // Values, not just bytes. NaN != NaN, so `F64` and `VecF64` are compared
    // on the bit patterns the format actually stores.
    // An edge column is keyed by `EdgeId`, and a load re-assigns ids in
    // traversal order -- exactly as graph-tool's does, since the reader binds
    // the i-th stored value to the i-th edge of `edges_range(g)`
    // (`graph_io_binary.hh:398-399`). So the column comes back permuted by the
    // traversal, and *that* is what has to be checked, not naive equality.
    let order = traversal(&doc.graph);
    assert_ne!(
        order,
        (0..doc.graph.num_edges()).collect::<Vec<_>>(),
        "the sample graph must actually permute, or this checks nothing"
    );
    assert_eq!(back.properties.len(), doc.properties.len());
    for (a, b) in doc.properties.iter().zip(&back.properties) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.domain, b.domain);
        assert_eq!(a.kind(), b.kind());
        let expected = if a.domain == PropertyDomain::Edge {
            reorder(&a.values, &order)
        } else {
            a.values.clone()
        };
        match (&expected, &b.values) {
            (PropertyColumn::F64(x), PropertyColumn::F64(y)) => {
                assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()));
            }
            (PropertyColumn::VecF64(x), PropertyColumn::VecF64(y)) => {
                assert!(
                    x.iter()
                        .zip(y)
                        .all(|(p, q)| p.iter().zip(q).all(|(u, v)| u.to_bits() == v.to_bits()))
                );
            }
            (x, y) => assert_eq!(x, y, "{}", a.name),
        }
    }
}

/// `column[order[k]]` for each k, for every member.
fn reorder(c: &PropertyColumn, order: &[usize]) -> PropertyColumn {
    macro_rules! pick {
        ($v:expr, $variant:ident) => {
            PropertyColumn::$variant(order.iter().map(|&i| $v[i].clone()).collect())
        };
    }
    match c {
        PropertyColumn::Bool(v) => pick!(v, Bool),
        PropertyColumn::I16(v) => pick!(v, I16),
        PropertyColumn::I32(v) => pick!(v, I32),
        PropertyColumn::I64(v) => pick!(v, I64),
        PropertyColumn::F64(v) => pick!(v, F64),
        PropertyColumn::LongDouble(v) => pick!(v, LongDouble),
        PropertyColumn::Str(v) => pick!(v, Str),
        PropertyColumn::VecBool(v) => pick!(v, VecBool),
        PropertyColumn::VecI16(v) => pick!(v, VecI16),
        PropertyColumn::VecI32(v) => pick!(v, VecI32),
        PropertyColumn::VecI64(v) => pick!(v, VecI64),
        PropertyColumn::VecF64(v) => pick!(v, VecF64),
        PropertyColumn::VecLongDouble(v) => pick!(v, VecLongDouble),
        PropertyColumn::VecStr(v) => pick!(v, VecStr),
        PropertyColumn::PyObject(v) => pick!(v, PyObject),
    }
}

/// The Python member needs no interpreter: the format stores the pickle
/// payload as a `std::string` (`:86-90`), so the bytes are the value.
///
/// If this crate ever gains a `python` feature, that feature turns the payload
/// into a live handle at the *boundary*; it does not change the file, and this
/// test is what pins that.
#[test]
fn the_python_member_round_trips_without_an_interpreter() {
    let mut doc = Document::new(sample_graph(), true);
    let payloads: Vec<Vec<u8>> = vec![
        b"\x80\x05N.".to_vec(),                 // None
        b"\x80\x05\x88.".to_vec(),              // True
        Vec::new(),                             // an empty payload is legal
        b"\x80\x05]\x94(K\x01K\x02e.".to_vec(), // [1, 2]
        vec![0xff; 300],                        // not valid UTF-8, and not text
    ];
    doc.properties.push(NamedProperty {
        name: "obj".into(),
        domain: PropertyDomain::Vertex,
        values: PropertyColumn::PyObject(payloads.clone()),
    });

    let bytes = to_bytes(&doc);
    let back = read(bytes.as_slice(), NoLookup).expect("read");
    assert_eq!(
        back.properties[0].values,
        PropertyColumn::PyObject(payloads)
    );
    assert_eq!(back.properties[0].kind(), ValueKind::PyObject);
    assert_eq!(to_bytes(&back), bytes);
}

/// `LongDouble` is sixteen opaque bytes with no arithmetic
/// (`prop/value.rs`), which is the whole reason a graph-tool file carrying
/// `long double` can be re-saved unchanged by a language with no 80-bit float.
#[test]
fn long_double_survives_as_bytes() {
    let mut doc = Document::new(AdjList::with_vertices(3), true);
    let vals: Vec<LongDouble> = (0..3).map(|i| ld(i as u8 + 200)).collect();
    doc.properties.push(NamedProperty {
        name: "ld".into(),
        domain: PropertyDomain::Vertex,
        values: PropertyColumn::LongDouble(vals.clone()),
    });
    let bytes = to_bytes(&doc);
    // 16 bytes each, exactly `sizeof(long double)`, with no padding between.
    assert!(bytes.ends_with(&vals.iter().flat_map(|v| v.0).collect::<Vec<u8>>()));

    let back = read(bytes.as_slice(), NoLookup).expect("read");
    assert_eq!(back.properties[0].values, PropertyColumn::LongDouble(vals));
    assert_eq!(to_bytes(&back), bytes);
}

// ---------------------------------------------------------------------------
// Acceptance: a real graph-tool file
// ---------------------------------------------------------------------------

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

#[test]
fn karate_loads_and_re_saves_byte_identically() {
    let bytes = fixture("karate.gt");
    let doc = read(bytes.as_slice(), NoLookup).expect("read karate.gt");

    // The header's own "stats:" clause, as graph-tool 2.2.32dev wrote it.
    assert_eq!(doc.graph.num_vertices(), 34);
    assert_eq!(doc.graph.num_edges(), 78);
    assert!(!doc.directed);
    doc.graph.validate().expect("validate");
    assert!(
        doc.comment
            .as_deref()
            .expect("read fills the comment")
            .contains("34 vertices, 78 edges, undirected"),
    );

    let by_name: BTreeMap<&str, &NamedProperty> = doc
        .properties
        .iter()
        .map(|p| (p.name.as_str(), p))
        .collect();
    assert_eq!(by_name.len(), 3);
    assert_eq!(by_name["description"].kind(), ValueKind::Str);
    assert_eq!(by_name["description"].domain, PropertyDomain::Graph);
    assert_eq!(by_name["readme"].domain, PropertyDomain::Graph);
    assert_eq!(by_name["pos"].kind(), ValueKind::VecF64);
    assert_eq!(by_name["pos"].domain, PropertyDomain::Vertex);
    assert_eq!(by_name["pos"].values.len(), 34);
    // Zachary's karate club: vertex 0 is the instructor, degree 16.
    assert_eq!(doc.graph.degree(VertexId::from_index(0)), 16);

    if cfg!(target_endian = "little") {
        assert_eq!(to_bytes(&doc), bytes, "re-save is not byte-identical");
    }
}

#[test]
fn lesmis_loads_and_re_saves_byte_identically() {
    let bytes = fixture("lesmis.gt");
    let doc = read(bytes.as_slice(), NoLookup).expect("read lesmis.gt");

    assert_eq!(doc.graph.num_vertices(), 77);
    assert_eq!(doc.graph.num_edges(), 254);
    assert!(!doc.directed);
    doc.graph.validate().expect("validate");

    // All three domains in one file, which is what makes this the fixture that
    // exercises the grouped write order of `write_graph` (`:461-466`).
    let doms: Vec<(&str, PropertyDomain, ValueKind)> = doc
        .properties
        .iter()
        .map(|p| (p.name.as_str(), p.domain, p.kind()))
        .collect();
    assert_eq!(
        doms,
        vec![
            ("description", PropertyDomain::Graph, ValueKind::Str),
            ("readme", PropertyDomain::Graph, ValueKind::Str),
            ("label", PropertyDomain::Vertex, ValueKind::Str),
            ("pos", PropertyDomain::Vertex, ValueKind::VecF64),
            ("value", PropertyDomain::Edge, ValueKind::F64),
        ]
    );
    assert_eq!(doc.properties[4].values.len(), 254, "one value per edge");
    let labels = <String as ColumnValue>::column_slice(&doc.properties[2].values).expect("strings");
    assert_eq!(labels[0], "Myriel");
    assert!(labels.iter().any(|l| l == "Valjean"));

    if cfg!(target_endian = "little") {
        assert_eq!(to_bytes(&doc), bytes, "re-save is not byte-identical");
    }
}

/// Edge property values are positional: the reader binds the `i`-th value to
/// the `i`-th edge of `edges_range(g)` (`:398-399`). So the value that came
/// back for an edge has to be the one that belongs to *that* edge, not merely
/// the right multiset.
#[test]
fn an_edge_column_stays_attached_to_its_edge() {
    let bytes = fixture("lesmis.gt");
    let doc = read(bytes.as_slice(), NoLookup).expect("read");
    let w = <f64 as ColumnValue>::column_slice(&doc.properties[4].values).expect("doubles");

    // The first adjacency entry in the file is Napoleon(1) -> Myriel(0) -- the
    // store is directed even though the graph is not -- and its weight is 1.
    // Vertex 0 has no *out*-edges at all, which is what makes this a real
    // check that the column is keyed by `EdgeId` and not by position.
    assert_eq!(doc.graph.out_degree(VertexId::from_index(0)), 0);
    assert_eq!(doc.graph.in_degree(VertexId::from_index(0)), 10);
    let e0 = doc
        .graph
        .find_edge(VertexId::from_index(1), VertexId::from_index(0))
        .expect("edge 1-0");
    assert_eq!(e0.id().index(), 0);
    assert_eq!(w[e0.id().index()], 1.0);

    // The heaviest interaction in Les Miserables is Valjean -- Javert, 31.
    let heaviest = w.iter().cloned().fold(f64::MIN, f64::max);
    assert_eq!(heaviest, 31.0);
    let peak = doc
        .graph
        .edges()
        .find(|e| w[e.id().index()] == heaviest)
        .expect("the heaviest edge");
    assert_eq!((peak.source().index(), peak.target().index()), (26, 11));

    // And the association survives the round trip.
    let again = read(to_bytes(&doc).as_slice(), NoLookup).expect("read");
    let w2 = <f64 as ColumnValue>::column_slice(&again.properties[4].values).expect("doubles");
    assert_eq!(w, w2);
}

// ---------------------------------------------------------------------------
// Ordering: defect #52
// ---------------------------------------------------------------------------

/// `AdjList` removes an adjacency entry by swapping with the back of the half
/// where `graph_adjacency.hh:1257-1263` erases and shifts, so adjacency order
/// after a removal is this port's, not graph-tool's. `write` therefore sorts
/// each vertex's out-half by `EdgeId`, which makes the output a function of
/// the graph and not of how it was reached.
#[test]
fn the_output_does_not_depend_on_the_removal_history() {
    let v = |i: usize| VertexId::from_index(i);
    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let build = || {
        let mut g = AdjList::with_vertices(4);
        let ids: Vec<EdgeId> = edges
            .iter()
            .map(|(a, b)| g.add_edge(v(*a), v(*b)).expect("add_edge").id())
            .collect();
        (g, ids)
    };

    let (a, _) = build();

    // Reach the *same* graph -- same ids on the same endpoints -- by removing
    // an edge and putting it straight back. `EdgeIds::release` pushes onto a
    // free list that `alloc` pops (`adj/alloc.rs:58-71`, porting
    // `get_free_index`/`put_free_index`, `graph_adjacency.hh:645-664`), so the
    // id returns; the adjacency entry does not return to its old slot, because
    // removal swaps with the back of the half rather than shifting
    // (`:1257-1263` does shift, which is defect #52).
    let (mut b, idb) = build();
    b.remove_edge(idb[1]).expect("remove");
    let again = b.add_edge(v(0), v(2)).expect("add_edge");
    assert_eq!(again.id(), idb[1], "the free list must hand the id back");

    // Same edges, same ids, different adjacency order.
    let ao: Vec<EdgeId> = a.out_edges(v(0)).map(|i| i.edge).collect();
    let bo: Vec<EdgeId> = b.out_edges(v(0)).map(|i| i.edge).collect();
    assert_ne!(ao, bo, "the histories must actually diverge for this test");
    assert_eq!(
        ao.iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        bo.iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
    );
    assert_eq!(adjacency(&a), adjacency(&b));

    // A column keyed by `EdgeId` therefore has to come out the same way too.
    let col = || PropertyColumn::I64((0..6).map(|i| i as i64 * 11).collect());
    let doc = |g: AdjList<NoLookup>| {
        let mut d = Document::new(g, true);
        d.properties.push(NamedProperty {
            name: "w".into(),
            domain: PropertyDomain::Edge,
            values: col(),
        });
        d
    };
    assert_eq!(to_bytes(&doc(a)), to_bytes(&doc(b)));
}

/// After a removal the edge index space is sparse, so an edge column is keyed
/// by `EdgeId` and sized to `edge_index_range`, not to `num_edges` -- the same
/// reason `_get_any` uses the range (`graph_tool/__init__.py:369`).
#[test]
fn a_sparse_edge_index_space_writes_the_live_slots() {
    let mut g = AdjList::with_vertices(3);
    let v = |i: usize| VertexId::from_index(i);
    let ids: Vec<EdgeId> = [(0, 1), (1, 2), (0, 2), (2, 0)]
        .iter()
        .map(|(a, b)| g.add_edge(v(*a), v(*b)).expect("add_edge").id())
        .collect();
    g.remove_edge(ids[1]).expect("remove");
    assert_eq!(g.num_edges(), 3);
    assert_eq!(g.edge_bound().len(), 4, "the hole is still in the space");

    // A column over the whole index range; slot 1 is the dead edge and must
    // never reach the file.
    let mut doc = Document::new(g, true);
    doc.properties.push(NamedProperty {
        name: "w".into(),
        domain: PropertyDomain::Edge,
        values: PropertyColumn::I64(vec![100, -1, 300, 400]),
    });

    let back = read(to_bytes(&doc).as_slice(), NoLookup).expect("read");
    assert_eq!(back.graph.num_edges(), 3);
    let w = <i64 as ColumnValue>::column_slice(&back.properties[0].values).expect("i64");
    assert_eq!(w.len(), 3, "the reader re-densifies the index space");
    assert!(
        !w.contains(&-1),
        "the dead slot leaked into the file: {w:?}"
    );

    // And each surviving value is still on its own edge.
    let pairs: Vec<(usize, usize, i64)> = back
        .graph
        .edges()
        .map(|e| (e.source().index(), e.target().index(), w[e.id().index()]))
        .collect();
    assert_eq!(pairs, vec![(0, 1, 100), (0, 2, 300), (2, 0, 400)]);
}

/// A short column is a caught error, not a silent out-of-bounds read.
/// `write_property_dispatch` calls `prop[x]` (`:319-320`) on an
/// `unchecked_vector_property_map`, which has no bounds check
/// (`fast_vector_property_map.hh:218-221`).
#[test]
fn a_short_column_is_refused_rather_than_read_past() {
    let mut doc = Document::new(sample_graph(), true);
    doc.properties.push(NamedProperty {
        name: "short".into(),
        domain: PropertyDomain::Vertex,
        values: PropertyColumn::I32(vec![1, 2, 3]), // five vertices
    });
    let mut out = Vec::new();
    match write(&mut out, &doc) {
        Err(IoError::ShortProperty { name, have, need }) => {
            assert_eq!((name.as_str(), have, need), ("short", 3, 5));
        }
        other => panic!("expected ShortProperty, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Acceptance: malformed input is diagnosed, never a panic
// ---------------------------------------------------------------------------

#[test]
fn every_truncation_is_bad_magic_or_parse() {
    for name in ["karate.gt", "lesmis.gt"] {
        let bytes = fixture(name);
        for cut in 0..bytes.len() {
            match read(&bytes[..cut], NoLookup) {
                Err(IoError::BadMagic { .. }) if cut < MAGIC.len() => {}
                Err(IoError::Parse { .. }) if cut >= MAGIC.len() => {}
                other => panic!("{name} truncated to {cut} bytes: {other:?}"),
            }
        }
        // And the untruncated file still reads.
        read(bytes.as_slice(), NoLookup).expect("full file");
    }
}

#[test]
fn a_synthetic_document_also_survives_every_truncation() {
    let bytes = to_bytes(&sample_document());
    for cut in 0..bytes.len() {
        let r = read(&bytes[..cut], NoLookup);
        assert!(
            matches!(
                r,
                Err(IoError::BadMagic { .. }) | Err(IoError::Parse { .. })
            ),
            "truncated to {cut}: {r:?}"
        );
    }
}

#[test]
fn a_foreign_file_is_bad_magic() {
    match read(&b"<?xml version=\"1.0\"?>"[..], NoLookup) {
        Err(IoError::BadMagic { expected, found }) => {
            assert_eq!(expected, MAGIC);
            assert_eq!(found, b"<?xml ");
        }
        other => panic!("expected BadMagic, got {other:?}"),
    }
    // An empty stream too: `found` is simply short.
    match read(&[][..], NoLookup) {
        Err(IoError::BadMagic { found, .. }) => assert!(found.is_empty()),
        other => panic!("expected BadMagic, got {other:?}"),
    }
}

#[test]
fn an_unknown_version_is_reported_as_such() {
    let mut bytes = to_bytes(&Document::new(AdjList::with_vertices(2), true));
    bytes[6] = 2;
    match read(bytes.as_slice(), NoLookup) {
        Err(IoError::UnsupportedVersion(2)) => {}
        other => panic!("expected UnsupportedVersion(2), got {other:?}"),
    }
}

/// `:215-217`: the bound is `N`, and the check happens before `add_edge`.
#[test]
fn an_out_of_range_vertex_index_is_reported_with_its_bound() {
    let mut g = AdjList::with_vertices(3);
    g.add_edge(VertexId::from_index(0), VertexId::from_index(2))
        .expect("add_edge");
    let mut bytes = to_bytes(&Document::new(g, true));
    // N = 3, so indices are one byte wide; the last byte of vertex 0's list is
    // the target index. Find it: header, then directed, N, count, index.
    let clen = u64::from_ne_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let idx = 16 + clen + 1 + 8 + 8;
    assert_eq!(bytes[idx], 2);
    bytes[idx] = 3; // == N
    match read(bytes.as_slice(), NoLookup) {
        Err(IoError::IndexOutOfRange { kind, index, bound }) => {
            assert_eq!((kind, index, bound), ("vertex", 3, 3));
        }
        other => panic!("expected IndexOutOfRange, got {other:?}"),
    }
    bytes[idx] = 255;
    assert!(matches!(
        read(bytes.as_slice(), NoLookup),
        Err(IoError::IndexOutOfRange { index: 255, .. })
    ));
}

/// `val_types` is `value_types` with `size_t` appended (`:257-258`), so index
/// 15 exists in the dispatch table and is never written: the `size_t` overload
/// emits `int64_t`'s index instead (`:335`). Anything from 15 up is refused.
#[test]
fn an_unknown_value_type_index_is_refused() {
    let mut doc = Document::new(AdjList::with_vertices(2), true);
    doc.properties.push(NamedProperty {
        name: "p".into(),
        domain: PropertyDomain::Vertex,
        values: column(ValueKind::I64, 2),
    });
    let bytes = to_bytes(&doc);
    let at = bytes.len() - 2 * 8 - 1;
    assert_eq!(bytes[at], ValueKind::I64 as u8);

    for bad in [15u8, 16, 200, 255] {
        let mut b = bytes.clone();
        b[at] = bad;
        match read(b.as_slice(), NoLookup) {
            Err(IoError::UnknownValueType(s)) => assert!(s.contains(&bad.to_string()), "{s}"),
            other => panic!("value type {bad}: expected UnknownValueType, got {other:?}"),
        }
    }
}

/// The `default:` arm of the domain switch (`:504-506`).
#[test]
fn an_unknown_property_domain_is_a_parse_error() {
    let mut doc = Document::new(AdjList::with_vertices(2), true);
    doc.properties.push(NamedProperty {
        name: "p".into(),
        domain: PropertyDomain::Vertex,
        values: column(ValueKind::I64, 2),
    });
    let mut bytes = to_bytes(&doc);
    let at = bytes.len() - 2 * 8 - 1 - 1 - 8 - 1;
    assert_eq!(bytes[at], PropertyDomain::Vertex as u8);
    bytes[at] = 7;
    match read(bytes.as_slice(), NoLookup) {
        Err(IoError::Parse { msg, .. }) => assert!(msg.contains("domain 7"), "{msg}"),
        other => panic!("expected Parse, got {other:?}"),
    }
}

/// `std::string` is a byte string, `String` is not. A payload that is not
/// UTF-8 is named and refused rather than lost or panicked on.
#[test]
fn a_non_utf8_string_payload_is_a_malformed_property() {
    let mut doc = Document::new(AdjList::with_vertices(1), true);
    doc.properties.push(NamedProperty {
        name: "label".into(),
        domain: PropertyDomain::Vertex,
        values: PropertyColumn::Str(vec!["ab".into()]),
    });
    let mut bytes = to_bytes(&doc);
    let n = bytes.len();
    bytes[n - 2] = 0xff;
    match read(bytes.as_slice(), NoLookup) {
        Err(IoError::MalformedProperty { name, declared }) => {
            assert_eq!((name.as_str(), declared), ("label", ValueKind::Str));
        }
        other => panic!("expected MalformedProperty, got {other:?}"),
    }
}

/// A declared length that no file could satisfy must not become a `resize`.
/// `read(std::istream&, std::vector<T>&)` does exactly that at `:111`.
#[test]
fn an_absurd_declared_length_does_not_allocate() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.push(VERSION);
    bytes.push(0); // little-endian
    bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // comment length
    bytes.extend_from_slice(b"short");
    match read(bytes.as_slice(), NoLookup) {
        Err(IoError::Parse { msg, .. }) => assert!(msg.contains("are present"), "{msg}"),
        other => panic!("expected Parse, got {other:?}"),
    }

    // The same for a vertex count, and for a property column's inner vector.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&[VERSION, 0]);
    bytes.extend_from_slice(&0u64.to_le_bytes()); // empty comment
    bytes.push(1); // directed
    bytes.extend_from_slice(&(1u64 << 40).to_le_bytes()); // N
    assert!(matches!(
        read(bytes.as_slice(), NoLookup),
        Err(IoError::Parse { .. }) | Err(IoError::IndexWidthExceeded)
    ));
}

// ---------------------------------------------------------------------------
// Endianness
// ---------------------------------------------------------------------------

/// `read<BE>` byte-swaps every multi-byte field when the file's declared order
/// is not the host's (`:45-52`). Re-encoding a little-endian file as
/// big-endian and reading it back must give the same document.
#[test]
fn a_big_endian_file_reads_the_same_as_its_little_endian_twin() {
    if cfg!(target_endian = "big") {
        return; // the swap below assumes the fixture is host-order
    }
    let doc = sample_document();
    let le = to_bytes(&doc);
    let be = swap_to_big_endian(&le);
    assert_ne!(le, be);

    let a = read(le.as_slice(), NoLookup).expect("read le");
    let b = read(be.as_slice(), NoLookup).expect("read be");
    assert_eq!(b.comment, a.comment);
    assert_eq!(adjacency(&b.graph), adjacency(&a.graph));
    assert_eq!(b.properties.len(), a.properties.len());
    for (x, y) in a.properties.iter().zip(&b.properties) {
        assert_eq!(x.name, y.name);
        assert_eq!(x.kind(), y.kind());
        if !matches!(x.kind(), ValueKind::F64 | ValueKind::VecF64) {
            assert_eq!(x.values, y.values, "{}", x.name);
        }
    }
}

/// Re-encode a host-order `.gt` stream as its big-endian twin, by walking the
/// layout and reversing every multi-byte field. This is the test's own
/// independent implementation of the format, which is the point: if it and
/// `gt::read` agree, they agree about the layout and not merely about each
/// other's bugs.
fn swap_to_big_endian(src: &[u8]) -> Vec<u8> {
    struct W<'a> {
        src: &'a [u8],
        at: usize,
        out: Vec<u8>,
    }
    impl W<'_> {
        fn copy(&mut self, n: usize) {
            self.out.extend_from_slice(&self.src[self.at..self.at + n]);
            self.at += n;
        }
        /// Reverse one field of `n` bytes, and -- for the `n <= 8` fields that
        /// are lengths and counts -- return its host-order value.
        fn swap(&mut self, n: usize) -> u64 {
            let mut b = self.src[self.at..self.at + n].to_vec();
            let mut v = [0u8; 8];
            v[..n.min(8)].copy_from_slice(&b[..n.min(8)]);
            b.reverse();
            self.out.extend_from_slice(&b);
            self.at += n;
            u64::from_le_bytes(v)
        }
        fn string(&mut self) {
            let n = self.swap(8) as usize;
            self.copy(n);
        }
        fn vector(&mut self, elem: usize) {
            let n = self.swap(8) as usize;
            for _ in 0..n {
                self.swap(elem);
            }
        }
    }

    let mut w = W {
        src,
        at: 0,
        out: Vec::new(),
    };
    w.copy(6); // magic
    w.copy(1); // version
    w.out.push(1); // big-endian from here on
    w.at += 1;
    w.string(); // comment
    w.copy(1); // directed
    let n = w.swap(8) as usize;
    let width = if n <= 0xff {
        1
    } else if n <= 0xffff {
        2
    } else if n <= 0xffff_ffff {
        4
    } else {
        8
    };
    let mut edges = 0usize;
    for _ in 0..n {
        let k = w.swap(8) as usize;
        for _ in 0..k {
            w.swap(width);
        }
        edges += k;
    }
    let nprops = w.swap(8);
    for _ in 0..nprops {
        let domain = w.src[w.at];
        w.copy(1);
        w.string(); // name
        let kind = ValueKind::ALL[w.src[w.at] as usize];
        w.copy(1);
        let count = match domain {
            0 => 1,
            1 => n,
            _ => edges,
        };
        for _ in 0..count {
            match kind {
                ValueKind::Bool => w.copy(1),
                ValueKind::I16 => {
                    w.swap(2);
                }
                ValueKind::I32 => {
                    w.swap(4);
                }
                ValueKind::I64 | ValueKind::F64 => {
                    w.swap(8);
                }
                ValueKind::LongDouble => {
                    w.swap(16);
                }
                // A byte run: the length swaps, the bytes do not.
                ValueKind::Str | ValueKind::VecBool | ValueKind::PyObject => w.string(),
                ValueKind::VecI16 => w.vector(2),
                ValueKind::VecI32 => w.vector(4),
                ValueKind::VecI64 | ValueKind::VecF64 => w.vector(8),
                ValueKind::VecLongDouble => w.vector(16),
                ValueKind::VecStr => {
                    let k = w.swap(8);
                    for _ in 0..k {
                        w.string();
                    }
                }
            }
        }
    }
    assert_eq!(w.at, src.len(), "the re-encoder did not consume the file");
    w.out
}

// ---------------------------------------------------------------------------
// The erased column
// ---------------------------------------------------------------------------

#[test]
fn a_column_knows_its_member_and_cannot_disagree_with_it() {
    for kind in ValueKind::ALL {
        let c = column(kind, 4);
        assert_eq!(c.kind(), kind, "{}", kind.name());
        assert_eq!(c.len(), 4);
        assert!(!c.is_empty());
        assert_eq!(PropertyColumn::empty(kind).kind(), kind);
        assert!(PropertyColumn::empty(kind).is_empty());
    }
}

#[test]
fn column_value_matches_the_prop_value_universe() {
    assert_eq!(<u8 as ColumnValue>::KIND, ValueKind::Bool);
    assert_eq!(<Vec<u8> as ColumnValue>::KIND, ValueKind::VecBool);
    assert_eq!(<Vec<String> as ColumnValue>::KIND, ValueKind::VecStr);
    assert_eq!(<LongDouble as ColumnValue>::KIND, ValueKind::LongDouble);

    let c = <i32 as ColumnValue>::into_column(vec![1, 2, 3]);
    assert_eq!(c.kind(), ValueKind::I32);
    assert_eq!(<i32 as ColumnValue>::column_slice(&c), Some(&[1, 2, 3][..]));
    // Asking for the wrong member is `None`, not a reinterpretation.
    assert_eq!(<i64 as ColumnValue>::column_slice(&c), None);
    // `python::object` shares its representation with `vector<bool>` and is
    // therefore *not* reachable through `ColumnValue`.
    let py = PropertyColumn::PyObject(vec![vec![1, 2]]);
    assert_eq!(<Vec<u8> as ColumnValue>::column_slice(&py), None);
}

// ---------------------------------------------------------------------------
// Degenerate graphs
// ---------------------------------------------------------------------------

#[test]
fn the_empty_graph_round_trips() {
    let doc = Document::new(AdjList::new(), false);
    let bytes = to_bytes(&doc);
    let back = read(bytes.as_slice(), NoLookup).expect("read");
    assert_eq!(back.graph.num_vertices(), 0);
    assert_eq!(back.graph.num_edges(), 0);
    assert!(!back.directed);
    assert!(back.properties.is_empty());
    assert_eq!(to_bytes(&back), bytes);
}

#[test]
fn isolated_vertices_and_a_graph_property_on_an_edgeless_graph() {
    let mut doc = Document::new(AdjList::with_vertices(4), true);
    doc.properties.push(NamedProperty {
        name: "name".into(),
        domain: PropertyDomain::Graph,
        values: PropertyColumn::Str(vec!["four dots".into()]),
    });
    let bytes = to_bytes(&doc);
    let back = read(bytes.as_slice(), NoLookup).expect("read");
    assert_eq!(back.graph.num_vertices(), 4);
    assert_eq!(back.graph.num_edges(), 0);
    assert_eq!(back.properties[0].values.len(), 1);
    assert_eq!(to_bytes(&back), bytes);
}

/// The undirected flag is metadata; storage is always the directed adjacency,
/// because `write_to_file` sets `_directed = true` around the write
/// (`graph_io.cc:506-507`). An undirected graph must therefore emit each edge
/// once, not twice.
#[test]
fn an_undirected_graph_emits_each_edge_once() {
    let mut g = AdjList::with_vertices(2);
    g.add_edge(VertexId::from_index(0), VertexId::from_index(1))
        .expect("add_edge");
    let doc = Document::new(g, false);
    let back = read(to_bytes(&doc).as_slice(), NoLookup).expect("read");
    assert!(!back.directed);
    assert_eq!(back.graph.num_edges(), 1);
    assert_eq!(back.graph.out_degree(VertexId::from_index(0)), 1);
    assert_eq!(back.graph.out_degree(VertexId::from_index(1)), 0);
    assert_eq!(back.graph.in_degree(VertexId::from_index(1)), 1);
}
