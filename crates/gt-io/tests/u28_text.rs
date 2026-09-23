//! U28 acceptance: GraphML and DOT.
//!
//! The references are `src/boost-workaround/boost/graph/graphml.hpp`,
//! `src/graph/graphml.cpp`, `src/boost-workaround/boost/graph/graphviz.hpp`
//! and `src/graph/read_graphviz_new.cpp`. Every assertion is either a layout
//! fact taken from one of those (cited at the assertion) or a property of the
//! round trip that the C++ reader depends on.
//!
//! The three things the unit specification asks for are
//! [`graphml_round_trip_preserves_typed_properties_and_edge_order`],
//! [`dot_round_trip_preserves_topology_but_not_vertex_identity`] and the two
//! `names_the_line` tests.

use std::collections::BTreeMap;

use gt_core::adj::{AdjList, NoLookup};
use gt_core::ids::{EdgeId, VertexId};
use gt_core::prop::{LongDouble, ValueKind};
use gt_io::IoError;
use gt_io::dot;
use gt_io::graphml;
use gt_io::gt::{Document, NamedProperty, PropertyColumn, PropertyDomain};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A little directed graph: a path, a parallel edge, a self-loop and an
/// isolated vertex, so the writers have to cope with all four.
fn graph(n: usize, edges: &[(usize, usize)]) -> AdjList<NoLookup> {
    let mut g = AdjList::new();
    for _ in 0..n {
        g.add_vertex().expect("add_vertex");
    }
    for &(s, t) in edges {
        g.add_edge(VertexId::from_index(s), VertexId::from_index(t))
            .expect("add_edge");
    }
    g
}

fn fixture() -> Document<NoLookup> {
    let g = graph(5, &[(0, 1), (1, 2), (0, 1), (3, 3), (2, 0)]);
    Document::new(g, true)
}

fn ld(seed: u8) -> LongDouble {
    // A normal 80-bit value: significand with the integer bit set, a plain
    // exponent. The payload is never interpreted as a float anywhere in the
    // port, so this only has to be a value the text form can name.
    let mut b = [0u8; 16];
    let frac = 0x8000_0000_0000_0000u64 | (u64::from(seed) << 40) | 0x0001_2345;
    b[..8].copy_from_slice(&frac.to_le_bytes());
    b[8..10].copy_from_slice(&(16383u16 + u16::from(seed)).to_le_bytes());
    LongDouble(b)
}

/// One column of `n` values of the given member.
///
/// Deliberately awkward -- an empty inner vector, an empty string, a
/// subnormal, a non-ASCII string, a value needing XML escaping -- but with
/// *no* NaN and no value whose text is empty at every position:
///
/// * a NaN never compares equal to itself, so a column holding one cannot be
///   checked with `==` (the `.gt` path compares bytes and can);
/// * `write_graphml` omits a `<data>` whose text is empty
///   (`graphml.hpp:436-437`), so a column that is empty everywhere is not
///   named by the file at all and does not come back. That is asserted on its
///   own in `graphml_omits_empty_values_and_loses_an_all_empty_column`.
fn column(kind: ValueKind, n: usize) -> PropertyColumn {
    let s = |i: usize| match i % 4 {
        0 => "value <1> & \"2\"".to_owned(),
        1 => format!("value {i}"),
        2 => "héllo ⛾ wörld".to_owned(),
        _ => String::new(),
    };
    let f = |i: usize| match i % 5 {
        0 => 0.0,
        1 => -0.0,
        2 => 1.0 / 3.0,
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
        ValueKind::VecBool => {
            PropertyColumn::VecBool((0..n).map(|i| (0..i % 3).map(|j| j as u8).collect()).collect())
        }
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
                .map(|i| (0..i % 3).map(|j| j as i64 * -7).collect())
                .collect(),
        ),
        ValueKind::VecF64 => PropertyColumn::VecF64(
            (0..n)
                .map(|i| (0..i % 4).map(|j| f(i + j)).collect())
                .collect(),
        ),
        ValueKind::VecLongDouble => PropertyColumn::VecLongDouble(
            (0..n)
                .map(|i| (0..i % 3).map(|j| ld((i + j) as u8)).collect())
                .collect(),
        ),
        ValueKind::VecStr => PropertyColumn::VecStr(
            (0..n)
                .map(|i| (0..i % 4).map(|j| s(i + j)).collect())
                .collect(),
        ),
        ValueKind::PyObject => PropertyColumn::PyObject(
            (0..n)
                .map(|i| (0..i % 5).map(|j| (i * 31 + j) as u8).collect())
                .collect(),
        ),
    }
}

fn prop(name: &str, domain: PropertyDomain, values: PropertyColumn) -> NamedProperty {
    NamedProperty {
        name: name.to_owned(),
        domain,
        values,
    }
}

fn by_key(doc: &Document<NoLookup>) -> BTreeMap<(String, u8), &PropertyColumn> {
    doc.properties
        .iter()
        .map(|p| ((p.name.clone(), p.domain as u8), &p.values))
        .collect()
}

fn to_graphml(doc: &Document<NoLookup>) -> String {
    let mut out = Vec::new();
    graphml::write(&mut out, doc).expect("write graphml");
    String::from_utf8(out).expect("graphml is utf-8")
}

fn to_dot(doc: &Document<NoLookup>) -> String {
    let mut out = Vec::new();
    dot::write(&mut out, doc).expect("write dot");
    String::from_utf8(out).expect("dot is utf-8")
}

fn edge_pairs(doc: &Document<NoLookup>) -> Vec<(usize, usize)> {
    let mut e: Vec<(EdgeId, usize, usize)> = doc
        .graph
        .edges()
        .map(|e| (e.id(), e.source().index(), e.target().index()))
        .collect();
    e.sort_unstable_by_key(|t| t.0);
    e.into_iter().map(|(_, s, t)| (s, t)).collect()
}

fn line_of(e: &IoError) -> usize {
    match e {
        IoError::Parse { line, .. } => *line,
        other => panic!("expected a parse error, got {other}"),
    }
}

// ===========================================================================
// GraphML
// ===========================================================================

/// The acceptance property: every member of the value universe survives a
/// round trip with its type, and the edge order is the one that was written.
#[test]
fn graphml_round_trip_preserves_typed_properties_and_edge_order() {
    let mut doc = fixture();
    let nv = doc.graph.num_vertices();
    let ne = doc.graph.num_edges();
    for (i, k) in ValueKind::ALL.into_iter().enumerate() {
        doc.properties
            .push(prop(&format!("v{i:02}"), PropertyDomain::Vertex, column(k, nv)));
        doc.properties
            .push(prop(&format!("e{i:02}"), PropertyDomain::Edge, column(k, ne)));
        // A graph column holds exactly one value, and that value must not be
        // empty -- see `column`'s documentation. Index 2 of a three-long
        // column is non-empty for every member.
        doc.properties.push(prop(
            &format!("g{i:02}"),
            PropertyDomain::Graph,
            keep_last(column(k, 3)),
        ));
    }

    let text = to_graphml(&doc);
    let back: Document<NoLookup> =
        graphml::read(text.as_bytes(), NoLookup).expect("read back what we wrote");

    assert_eq!(back.directed, doc.directed);
    assert_eq!(back.graph.num_vertices(), nv);
    assert_eq!(back.graph.num_edges(), ne);
    // Edge order: the i-th `<edge>` becomes EdgeId i, and the writer emits in
    // EdgeId order, so the endpoints come back in the same sequence.
    assert_eq!(edge_pairs(&back), edge_pairs(&doc));

    let want = by_key(&doc);
    let got = by_key(&back);
    assert_eq!(got.keys().collect::<Vec<_>>(), want.keys().collect::<Vec<_>>());
    for (k, v) in &want {
        assert_eq!(got[k], *v, "column {k:?}");
    }

    // Writing what we read gives the same bytes: the format is a fixed point.
    assert_eq!(to_graphml(&back), text);
}

/// The last value of a column, as a column of one: a graph property's
/// domain has exactly one element.
fn keep_last(mut c: PropertyColumn) -> PropertyColumn {
    macro_rules! trim {
        ($v:expr) => {{
            let last = $v.pop().expect("a non-empty column");
            $v.clear();
            $v.push(last);
        }};
    }
    match &mut c {
        PropertyColumn::Bool(v) => trim!(v),
        PropertyColumn::I16(v) => trim!(v),
        PropertyColumn::I32(v) => trim!(v),
        PropertyColumn::I64(v) => trim!(v),
        PropertyColumn::F64(v) => trim!(v),
        PropertyColumn::LongDouble(v) => trim!(v),
        PropertyColumn::Str(v) => trim!(v),
        PropertyColumn::VecBool(v) => trim!(v),
        PropertyColumn::VecI16(v) => trim!(v),
        PropertyColumn::VecI32(v) => trim!(v),
        PropertyColumn::VecI64(v) => trim!(v),
        PropertyColumn::VecF64(v) => trim!(v),
        PropertyColumn::VecLongDouble(v) => trim!(v),
        PropertyColumn::VecStr(v) => trim!(v),
        PropertyColumn::PyObject(v) => trim!(v),
    }
    c
}

/// The layout, byte for byte, against `write_graphml` (`graphml.hpp:366-527`).
#[test]
fn graphml_layout_matches_write_graphml() {
    let mut doc = Document::new(graph(2, &[(0, 1)]), true);
    doc.properties.push(prop(
        "w",
        PropertyDomain::Vertex,
        PropertyColumn::I64(vec![7, -3]),
    ));
    assert_eq!(
        to_graphml(&doc),
        concat!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
            "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n",
            "         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n",
            "         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns",
            " http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n\n",
            "  <!-- property keys -->\n",
            "  <key id=\"key0\" for=\"node\" attr.name=\"w\" attr.type=\"long\" />\n",
            "\n  <graph id=\"G\" edgedefault=\"directed\" parse.nodeids=\"canonical\"",
            " parse.edgeids=\"canonical\" parse.order=\"nodesfirst\">\n\n",
            "   <!-- graph properties -->\n",
            "\n   <!-- vertices -->\n",
            "    <node id=\"n0\">\n",
            "      <data key=\"key0\">7</data>\n",
            "    </node>\n",
            "    <node id=\"n1\">\n",
            "      <data key=\"key0\">-3</data>\n",
            "    </node>\n",
            "\n   <!-- edges -->\n",
            "    <edge id=\"e0\" source=\"n0\" target=\"n1\">\n",
            "    </edge>\n",
            "\n  </graph>\n</graphml>\n",
        )
    );
}

/// `dp` is a `std::multimap`, so its three walks are sorted by name
/// (`graphml.hpp:383`, `:430`, `:458`) whatever order the properties arrived
/// in -- and the key ids follow that order.
#[test]
fn graphml_keys_are_numbered_in_name_order() {
    let mut doc = Document::new(graph(1, &[]), true);
    for name in ["zebra", "alpha", "middle"] {
        doc.properties.push(prop(
            name,
            PropertyDomain::Vertex,
            PropertyColumn::I32(vec![1]),
        ));
    }
    let text = to_graphml(&doc);
    let keys: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("<key "))
        .map(|l| l.split("attr.name=\"").nth(1).unwrap().split('"').next().unwrap())
        .collect();
    assert_eq!(keys, ["alpha", "middle", "zebra"]);
    assert!(text.contains("<key id=\"key0\" for=\"node\" attr.name=\"alpha\""));
    assert!(text.contains("<key id=\"key2\" for=\"node\" attr.name=\"zebra\""));
}

/// `parse.nodeids="free"`: the ids are interned in first-seen order and kept
/// in `_graphml_vertex_id` (`graphml.cpp:323-343`), which the writer then
/// uses again instead of `n0`, `n1` (`graphml.hpp:451-455`).
#[test]
fn graphml_free_node_ids_are_stored_and_written_back() {
    let src = r#"<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="k0" for="node" attr.name="label" attr.type="string" />
  <graph id="G" edgedefault="undirected" parse.nodeids="free" parse.edgeids="free">
    <node id="alice"><data key="k0">A</data></node>
    <node id="bob"><data key="k0">B</data></node>
    <edge id="friendship" source="alice" target="bob" />
  </graph>
</graphml>
"#;
    let doc: Document<NoLookup> = graphml::read(src.as_bytes(), NoLookup).expect("read");
    assert!(!doc.directed);
    assert_eq!(doc.graph.num_vertices(), 2);
    assert_eq!(doc.graph.num_edges(), 1);

    let ids = by_key(&doc);
    assert_eq!(
        ids[&(graphml::VERTEX_ID_KEY.to_owned(), PropertyDomain::Vertex as u8)],
        &PropertyColumn::Str(vec!["alice".to_owned(), "bob".to_owned()])
    );
    assert_eq!(
        ids[&(graphml::EDGE_ID_KEY.to_owned(), PropertyDomain::Edge as u8)],
        &PropertyColumn::Str(vec!["friendship".to_owned()])
    );

    let text = to_graphml(&doc);
    assert!(text.contains("parse.nodeids=\"free\""));
    assert!(text.contains("parse.edgeids=\"free\""));
    assert!(text.contains("<node id=\"alice\">"));
    assert!(text.contains("<edge id=\"friendship\" source=\"alice\" target=\"bob\">"));
    // The id properties are node ids, not `<key>`s.
    assert!(!text.contains("_graphml_vertex_id"));

    let again: Document<NoLookup> = graphml::read(text.as_bytes(), NoLookup).expect("re-read");
    assert_eq!(by_key(&again).len(), by_key(&doc).len());
    assert_eq!(to_graphml(&again), text);
}

/// `<default>` applies to every element created after it, and only to keys of
/// the matching kind (`graphml.cpp:333-339`, `:381-386`).
#[test]
fn graphml_key_defaults_reach_new_elements() {
    let src = r#"<graphml>
  <key id="k0" for="node" attr.name="weight" attr.type="int"><default>42</default></key>
  <key id="k1" for="edge" attr.name="w" attr.type="float"><default>1.5</default></key>
  <graph edgedefault="directed" parse.nodeids="canonical">
    <node id="n0"/>
    <node id="n1"><data key="k0">7</data></node>
    <edge source="n0" target="n1"/>
  </graph>
</graphml>
"#;
    let doc: Document<NoLookup> = graphml::read(src.as_bytes(), NoLookup).expect("read");
    let cols = by_key(&doc);
    assert_eq!(
        cols[&("weight".to_owned(), PropertyDomain::Vertex as u8)],
        &PropertyColumn::I32(vec![42, 7])
    );
    assert_eq!(
        cols[&("w".to_owned(), PropertyDomain::Edge as u8)],
        &PropertyColumn::F64(vec![1.5])
    );
}

/// `put_property`'s boolean table (`graphml.hpp:246-252`) and
/// `lexical_cast<uint8_t>`'s trip through `int` (`str_repr.hh:52-55`).
#[test]
fn graphml_boolean_spellings_and_char_truncation() {
    let src = r#"<graphml>
  <key id="b" for="node" attr.name="flag" attr.type="boolean"/>
  <graph edgedefault="undirected" parse.nodeids="canonical">
    <node id="n0"><data key="b">true</data></node>
    <node id="n1"><data key="b">False</data></node>
    <node id="n2"><data key="b">300</data></node>
    <node id="n3"><data key="b">-1</data></node>
  </graph>
</graphml>
"#;
    let doc: Document<NoLookup> = graphml::read(src.as_bytes(), NoLookup).expect("read");
    assert_eq!(
        by_key(&doc)[&("flag".to_owned(), PropertyDomain::Vertex as u8)],
        &PropertyColumn::Bool(vec![1, 0, 44, 255])
    );
}

/// The type table is graph-tool's, not GraphML's: `attr.type="double"` is
/// `long double` and `attr.type="float"` is IEEE double
/// (`graphml.cpp:27-31`).
#[test]
fn graphml_type_names_are_graph_tools_own() {
    let mut doc = Document::new(graph(1, &[]), true);
    doc.properties.push(prop(
        "d",
        PropertyDomain::Vertex,
        PropertyColumn::F64(vec![0.5]),
    ));
    doc.properties.push(prop(
        "l",
        PropertyDomain::Vertex,
        PropertyColumn::LongDouble(vec![ld(0)]),
    ));
    let text = to_graphml(&doc);
    assert!(text.contains("attr.name=\"d\" attr.type=\"float\""));
    assert!(text.contains("attr.name=\"l\" attr.type=\"double\""));

    let back: Document<NoLookup> = graphml::read(text.as_bytes(), NoLookup).expect("read");
    let cols = by_key(&back);
    assert_eq!(
        cols[&("d".to_owned(), PropertyDomain::Vertex as u8)].kind(),
        ValueKind::F64
    );
    assert_eq!(
        cols[&("l".to_owned(), PropertyDomain::Vertex as u8)].kind(),
        ValueKind::LongDouble
    );
}

/// `if (val.empty()) continue` (`graphml.hpp:436`, `:468`, `:516`).
#[test]
fn graphml_omits_empty_values_and_loses_an_all_empty_column() {
    let mut doc = Document::new(graph(2, &[]), true);
    doc.properties.push(prop(
        "some",
        PropertyDomain::Vertex,
        PropertyColumn::Str(vec![String::new(), "here".to_owned()]),
    ));
    doc.properties.push(prop(
        "none",
        PropertyDomain::Vertex,
        PropertyColumn::Str(vec![String::new(), String::new()]),
    ));
    let text = to_graphml(&doc);
    // Both keys are declared...
    assert!(text.contains("attr.name=\"some\""));
    assert!(text.contains("attr.name=\"none\""));
    // ...but only one datum is written, so only one column comes back.
    assert_eq!(text.matches("<data").count(), 1);

    let back: Document<NoLookup> = graphml::read(text.as_bytes(), NoLookup).expect("read");
    let cols = by_key(&back);
    assert_eq!(
        cols[&("some".to_owned(), PropertyDomain::Vertex as u8)],
        &PropertyColumn::Str(vec![String::new(), "here".to_owned()])
    );
    assert!(!cols.contains_key(&("none".to_owned(), PropertyDomain::Vertex as u8)));
}

/// Malformed input is a `Parse` naming the line, never a panic.
#[test]
fn graphml_malformed_input_names_the_line() {
    // 1234567
    let cases: &[(&str, usize, &str)] = &[
        (
            "<graphml>\n  <graph edgedefault=\"directed\">\n    <node id=\"n0\">\n  </graph>\n</graphml>\n",
            4,
            "mismatched end tag",
        ),
        (
            "<graphml>\n<graph parse.nodeids=\"canonical\">\n<node id=\"xyz\"/>\n</graph>\n</graphml>\n",
            3,
            "a canonical id that is not a number",
        ),
        (
            "<graphml>\n<key id=\"k\" for=\"node\" attr.name=\"a\" attr.type=\"quaternion\"/>\n<graph parse.nodeids=\"canonical\">\n<node id=\"n0\"><data key=\"k\">1</data></node>\n</graph>\n</graphml>\n",
            4,
            "an unknown attr.type",
        ),
        (
            "<graphml>\n<key id=\"k\" for=\"node\" attr.name=\"a\" attr.type=\"int\"/>\n<graph parse.nodeids=\"canonical\">\n<node id=\"n0\">\n<data key=\"k\">not a number</data>\n</node>\n</graph>\n</graphml>\n",
            5,
            "a value that does not parse",
        ),
        (
            "<graphml>\n<graph parse.nodeids=\"canonical\">\n<node id=\"n0\" />\n",
            4,
            "an unclosed element at end of input",
        ),
        (
            "<graphml>\n<graph>\n<node id=\"n0\" bad />\n</graph>\n</graphml>\n",
            3,
            "an attribute with no value",
        ),
        (
            "<graphml>\n<graph parse.nodeids=\"canonical\">\n<node id=\"n0\">&nope;</node>\n</graph>\n</graphml>\n",
            3,
            "an undefined entity",
        ),
    ];
    for (src, line, what) in cases {
        let err = graphml::read(src.as_bytes(), NoLookup)
            .err()
            .unwrap_or_else(|| panic!("{what} should not be accepted"));
        assert_eq!(line_of(&err), *line, "{what}: {err}");
    }
}

/// Comments, processing instructions, CDATA and a namespace prefix are all
/// skipped or resolved; expat does the same (`graphml.cpp:57`, `:109`).
#[test]
fn graphml_accepts_the_xml_furniture() {
    let src = r#"<?xml version="1.0"?>
<!-- a comment with <tags/> and & inside -->
<g:graphml xmlns:g="http://graphml.graphdrawing.org/xmlns">
  <g:key id="k" for="node" attr.name="s" attr.type="string"/>
  <g:graph edgedefault="directed" parse.nodeids="canonical">
    <g:node id="n0"><g:data key="k"><![CDATA[raw < & > text]]></g:data></g:node>
    <g:node id="n1"><g:data key="k">&lt;escaped&gt; &amp; &#65;</g:data></g:node>
  </g:graph>
</g:graphml>
"#;
    let doc: Document<NoLookup> = graphml::read(src.as_bytes(), NoLookup).expect("read");
    assert_eq!(
        by_key(&doc)[&("s".to_owned(), PropertyDomain::Vertex as u8)],
        &PropertyColumn::Str(vec![
            "raw < & > text".to_owned(),
            "<escaped> & A".to_owned()
        ])
    );
}

/// A column shorter than its domain is refused before a byte reaches the
/// stream -- the condition `unchecked_vector_property_map::operator[]` cannot
/// detect (`fast_vector_property_map.hh:218-221`).
#[test]
fn graphml_refuses_a_short_column() {
    let mut doc = Document::new(graph(4, &[]), true);
    doc.properties.push(prop(
        "short",
        PropertyDomain::Vertex,
        PropertyColumn::I32(vec![1, 2]),
    ));
    let mut out = Vec::new();
    match graphml::write(&mut out, &doc) {
        Err(IoError::ShortProperty { name, have, need }) => {
            assert_eq!((name.as_str(), have, need), ("short", 2, 4));
        }
        other => panic!("expected ShortProperty, got {other:?}"),
    }
    assert!(out.is_empty(), "nothing may be written before the refusal");
}

// ===========================================================================
// DOT
// ===========================================================================

/// The acceptance property, and the honest version of it: DOT preserves the
/// topology, and it preserves vertex *identity* only up to the node names,
/// because `translate_results_to_graph` creates vertices in `std::map` order
/// (`read_graphviz_new.cpp:754-757`).
#[test]
fn dot_round_trip_preserves_topology_but_not_vertex_identity() {
    // Twelve vertices, so that "10" sorts before "2".
    let doc = Document::new(
        graph(
            12,
            &[(0, 1), (10, 2), (2, 10), (11, 0), (3, 3), (1, 1), (9, 10)],
        ),
        true,
    );
    let text = to_dot(&doc);
    let back: Document<NoLookup> = dot::read(text.as_bytes(), NoLookup).expect("read dot");

    assert_eq!(back.graph.num_vertices(), 12);
    assert_eq!(back.graph.num_edges(), doc.graph.num_edges());
    assert!(back.directed);

    let names = by_key(&back)[&(dot::NODE_ID_KEY.to_owned(), PropertyDomain::Vertex as u8)];
    let PropertyColumn::Str(names) = names else {
        panic!("node names are strings");
    };
    // Lexicographic, not numeric: this is the reordering, pinned.
    assert_eq!(names[0], "0");
    assert_eq!(names[1], "1");
    assert_eq!(names[2], "10");
    assert_eq!(names[3], "11");
    assert_eq!(names[4], "2");

    // Topology, mapped through the names, edge for edge and in order.
    let mapped: Vec<(usize, usize)> = edge_pairs(&back)
        .into_iter()
        .map(|(s, t)| {
            (
                names[s].parse::<usize>().expect("a name we wrote"),
                names[t].parse::<usize>().expect("a name we wrote"),
            )
        })
        .collect();
    assert_eq!(mapped, edge_pairs(&doc));

    // Not a fixed point at the first hop -- the vertices have been
    // renumbered, so the node lines come out in the new order -- but one at
    // the second, because the names are already sorted by then.
    let rewritten = to_dot(&back);
    assert_ne!(rewritten, text);
    let again: Document<NoLookup> = dot::read(rewritten.as_bytes(), NoLookup).expect("read again");
    assert_eq!(to_dot(&again), rewritten);
}

/// Property typing is best-effort, and "best effort" means *string*: DOT
/// carries no types at all (`graphviz.hpp:785-790`, `graph_io.cc:217-261`).
/// This asserts exactly that, and no more.
#[test]
fn dot_property_typing_is_strings_only() {
    let mut doc = Document::new(graph(2, &[(0, 1)]), true);
    doc.properties.push(prop(
        "weight",
        PropertyDomain::Edge,
        PropertyColumn::I64(vec![-17]),
    ));
    doc.properties.push(prop(
        "x",
        PropertyDomain::Vertex,
        PropertyColumn::F64(vec![0.5, 1.0 / 3.0]),
    ));
    let text = to_dot(&doc);
    let back: Document<NoLookup> = dot::read(text.as_bytes(), NoLookup).expect("read dot");
    let cols = by_key(&back);

    let weight = cols[&("weight".to_owned(), PropertyDomain::Edge as u8)];
    assert_eq!(weight.kind(), ValueKind::Str, "not I64: DOT has no types");
    assert_eq!(weight, &PropertyColumn::Str(vec!["-17".to_owned()]));

    let x = cols[&("x".to_owned(), PropertyDomain::Vertex as u8)];
    assert_eq!(x.kind(), ValueKind::Str);
    // The text is `print_float`'s, digit for digit (`str_repr.hh:62-69`).
    assert_eq!(
        x,
        &PropertyColumn::Str(vec!["0.5".to_owned(), "0.33333333333333331".to_owned()])
    );
}

/// The layout, byte for byte, against `write_graphviz` (`graphviz.hpp:270-288`)
/// -- including the two spaces before an edge's attribute list, which are one
/// from the edge line and one from the attribute writer.
#[test]
fn dot_layout_matches_write_graphviz() {
    let mut doc = Document::new(graph(3, &[(0, 1), (1, 2)]), true);
    doc.properties.push(prop(
        "w",
        PropertyDomain::Edge,
        PropertyColumn::I32(vec![3, 4]),
    ));
    doc.properties.push(prop(
        "color",
        PropertyDomain::Vertex,
        PropertyColumn::Str(vec!["red".to_owned(), "a b".to_owned(), String::new()]),
    ));
    assert_eq!(
        to_dot(&doc),
        concat!(
            "digraph G {\n",
            "0 [color=red];\n",
            "1 [color=\"a b\"];\n",
            "2 [color=\"\"];\n",
            "0->1  [w=3];\n",
            "1->2  [w=4];\n",
            "}\n",
        )
    );

    let undirected = Document::new(graph(2, &[(0, 1)]), false);
    assert_eq!(to_dot(&undirected), "graph G {\n0;\n1;\n0--1 ;\n}\n");
}

/// A vertex property named `vertex_name` becomes the node ids and is not also
/// an attribute (`graph_io.cc:389-404`, `graphviz.hpp:557-570`).
#[test]
fn dot_uses_vertex_name_for_the_node_ids() {
    let mut doc = Document::new(graph(2, &[(0, 1)]), true);
    doc.properties.push(prop(
        dot::NODE_ID_KEY,
        PropertyDomain::Vertex,
        PropertyColumn::Str(vec!["alice".to_owned(), "bob".to_owned()]),
    ));
    assert_eq!(
        to_dot(&doc),
        "digraph G {\nalice;\nbob;\nalice->bob ;\n}\n"
    );
}

/// `strict` drops self-loops and repeated endpoint pairs
/// (`read_graphviz_new.cpp:679-687`); without it both are kept, because this
/// port's adjacency is a multigraph.
#[test]
fn dot_strict_drops_loops_and_parallel_edges() {
    let src = "strict digraph { a -> b; a -> b; a -> a; b -> a; }";
    let doc: Document<NoLookup> = dot::read(src.as_bytes(), NoLookup).expect("read");
    assert_eq!(doc.graph.num_edges(), 2);

    let loose: Document<NoLookup> = dot::read(
        "digraph { a -> b; a -> b; a -> a; b -> a; }".as_bytes(),
        NoLookup,
    )
    .expect("read");
    assert_eq!(loose.graph.num_edges(), 4);
}

/// An edge endpoint that is a subgraph expands to every node the subgraph
/// holds, in sorted order (`read_graphviz_new.cpp:634-677`); node defaults
/// are those in force at a node's *first mention* (`:599-601`).
#[test]
fn dot_subgraphs_expand_and_defaults_apply_at_first_mention() {
    let src = r#"digraph {
  node [color=blue];
  a;
  subgraph s { node [color=red]; c; b }
  a -> subgraph s;
  node [color=green];
  a;
  d;
}"#;
    let doc: Document<NoLookup> = dot::read(src.as_bytes(), NoLookup).expect("read");
    let cols = by_key(&doc);
    let PropertyColumn::Str(names) =
        cols[&(dot::NODE_ID_KEY.to_owned(), PropertyDomain::Vertex as u8)]
    else {
        panic!("names are strings")
    };
    assert_eq!(names, &["a", "b", "c", "d"]);
    // `a` keeps the default it was first seen with; `d` gets the later one.
    assert_eq!(
        cols[&("color".to_owned(), PropertyDomain::Vertex as u8)],
        &PropertyColumn::Str(vec![
            "blue".to_owned(),
            "red".to_owned(),
            "red".to_owned(),
            "green".to_owned()
        ])
    );
    // a -> {b, c}, in the set's order.
    assert_eq!(edge_pairs(&doc), vec![(0, 1), (0, 2)]);
}

/// `name = value` at graph level is a graph property; the writer drops it,
/// because `write_graphviz` is given `default_writer()` for graph properties
/// (`graphviz.hpp:626`).
#[test]
fn dot_reads_graph_properties_and_the_writer_drops_them() {
    let doc: Document<NoLookup> =
        dot::read("digraph { bgcolor=yellow; a -> b; }".as_bytes(), NoLookup).expect("read");
    assert_eq!(
        by_key(&doc)[&("bgcolor".to_owned(), PropertyDomain::Graph as u8)],
        &PropertyColumn::Str(vec!["yellow".to_owned()])
    );
    assert!(!to_dot(&doc).contains("bgcolor"));
}

/// Quoting, concatenation, comments and the keyword case-insensitivity of
/// `to_lower_copy` (`read_graphviz_new.cpp:180`).
#[test]
fn dot_accepts_the_lexical_furniture() {
    let src = concat!(
        "# a hash comment\n",
        "DiGraph \"my graph\" { // a slash comment\n",
        "  /* a block\n     comment */\n",
        "  \"a b\" -> \"c\" + \"d\" [label=\"x\\\"y\"];\n",
        "}\n"
    );
    let doc: Document<NoLookup> = dot::read(src.as_bytes(), NoLookup).expect("read");
    let cols = by_key(&doc);
    let PropertyColumn::Str(names) =
        cols[&(dot::NODE_ID_KEY.to_owned(), PropertyDomain::Vertex as u8)]
    else {
        panic!("names are strings")
    };
    assert_eq!(names, &["a b", "cd"]);
    assert_eq!(
        cols[&("label".to_owned(), PropertyDomain::Edge as u8)],
        &PropertyColumn::Str(vec!["x\"y".to_owned()])
    );
}

/// Malformed input is a `Parse` naming the line, never a panic. The C++
/// reports no position at all; the wording is otherwise its own.
#[test]
fn dot_malformed_input_names_the_line() {
    let cases: &[(&str, usize, &str)] = &[
        ("digraph G {\n  a -> b;\n  c ->;\n}\n", 3, "no endpoint"),
        ("digraph G {\n  a -- b;\n}\n", 2, "-- in a digraph"),
        ("graph G {\n  a -> b;\n}\n", 2, "-> in a graph"),
        ("digraph G {\n  a;\n", 3, "no closing brace"),
        ("hypergraph G {\n}\n", 1, "not graph or digraph"),
        ("digraph G {\n  a [x=];\n}\n", 2, "no attribute value"),
        ("digraph G {\n  a;\n  %\n}\n", 3, "an invalid character"),
        (
            "digraph G {\n  a [label=\"unterminated];\n}\n",
            2,
            "an unclosed quoted string",
        ),
        ("digraph G {\n}\njunk\n", 3, "trailing tokens"),
    ];
    for (src, line, what) in cases {
        let err = dot::read(src.as_bytes(), NoLookup)
            .err()
            .unwrap_or_else(|| panic!("{what} should not be accepted"));
        assert_eq!(line_of(&err), *line, "{what}: {err}");
    }
}

// ===========================================================================
// Both
// ===========================================================================

/// Neither reader may panic on arbitrary input, and every prefix of a valid
/// file is arbitrary input.
#[test]
fn truncation_is_an_error_not_a_panic() {
    let mut doc = fixture();
    doc.properties.push(prop(
        "s",
        PropertyDomain::Vertex,
        column(ValueKind::Str, 5),
    ));
    for text in [to_graphml(&doc), to_dot(&doc)] {
        for cut in 0..text.len() {
            if !text.is_char_boundary(cut) {
                continue;
            }
            let head = &text[..cut];
            // Either it parses (a prefix can be a whole valid document only
            // for DOT's `digraph G {}`, which no prefix here is) or it is an
            // error -- but never a panic and never a wrong line.
            if let Err(e) = graphml::read(head.as_bytes(), NoLookup) {
                let _ = line_of(&e);
            }
            if let Err(e) = dot::read(head.as_bytes(), NoLookup) {
                let _ = line_of(&e);
            }
        }
    }
}
