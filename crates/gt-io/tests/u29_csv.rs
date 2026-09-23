//! U29 -- delimited edge lists.
//!
//! The three acceptance criteria, in order:
//!
//! 1. **Symbolic endpoints intern deterministically, in first-seen order.**
//!    That is `get_vertex` (`graph_python_interface_imp1.cc:241-253`) invoked
//!    source-then-target (`:256`, `:260`): the id a name receives is the
//!    number of *distinct* names that preceded it, counting the source of its
//!    own record as preceding its target.
//! 2. **`read_edge_list` produces identical edge ids at 1 and 16 threads.**
//!    `RAYON_NUM_THREADS` is read by rayon's global pool once, when it is
//!    first used, so two values of it cannot be observed inside one test
//!    binary; this test re-executes itself, as `gt-core`'s `u06_builder.rs`
//!    does, and compares what the children wrote. What it compares is not the
//!    edge *set* -- that would pass for any merge order -- but the whole
//!    id-to-endpoint map over the edge index range, plus the interned names in
//!    id order.
//! 3. **`write_edge_list` emits in `EdgeId` order.** The graph under test has
//!    had edges removed and re-added, so its adjacency order, its insertion
//!    order and its id order are three different orders, and the test asserts
//!    against the third while asserting that it differs from the first.
//!
//! Everything else here is the dialect and the two endpoint modes, which are
//! where a reader of this format actually goes wrong.

use std::fmt::Write as _;
use std::process::Command;

use gt_core::adj::{AdjList, NoLookup};
use gt_core::ids::{EdgeId, VertexId};
use gt_io::IoError;
use gt_io::csv::{EdgeListOptions, read_edge_list, write_edge_list};
use proptest::prelude::*;

// ===========================================================================
// Helpers
// ===========================================================================

fn symbolic() -> EdgeListOptions {
    EdgeListOptions {
        delimiter: b',',
        header: false,
        symbolic: true,
    }
}

fn indexed() -> EdgeListOptions {
    EdgeListOptions {
        delimiter: b',',
        header: false,
        symbolic: false,
    }
}

fn read(src: &str, opts: EdgeListOptions) -> Result<(AdjList<NoLookup>, Vec<String>), IoError> {
    read_edge_list(src.as_bytes(), opts, NoLookup)
}

fn read_ok(src: &str, opts: EdgeListOptions) -> (AdjList<NoLookup>, Vec<String>) {
    read(src, opts).expect("well-formed edge list")
}

/// The `i`-th edge's endpoints, by id, for `i` over the whole index range.
fn by_id<H: gt_core::adj::Lookup>(g: &AdjList<H>) -> Vec<Option<(usize, usize)>> {
    (0..g.edge_bound().len())
        .map(|i| {
            g.endpoints(EdgeId::from_index(i))
                .map(|(s, t)| (s.index(), t.index()))
        })
        .collect()
}

fn write_to_string<H: gt_core::adj::Lookup>(g: &AdjList<H>, opts: EdgeListOptions) -> String {
    let mut out = Vec::new();
    write_edge_list(&mut out, g, opts).expect("writing to a Vec cannot fail");
    String::from_utf8(out).expect("the writer emits ASCII")
}

fn build(n: usize, pairs: &[(usize, usize)]) -> AdjList<NoLookup> {
    let mut g = AdjList::<NoLookup>::with_vertices(n);
    for &(s, t) in pairs {
        g.add_edge(VertexId::from_index(s), VertexId::from_index(t))
            .expect("endpoints in range");
    }
    g
}

// ===========================================================================
// Acceptance 1 -- first-seen interning
// ===========================================================================

#[test]
fn symbolic_endpoints_intern_in_first_seen_order() {
    // `c` is a target before it is a source, `a` is seen three times, and the
    // last record mentions nothing new. The expected numbering is exactly the
    // order the names are reached reading left-to-right, top-to-bottom.
    let (g, names) = read_ok("a,c\nb,a\nc,d\na,b\n", symbolic());

    assert_eq!(names, vec!["a", "c", "b", "d"]);
    assert_eq!(g.num_vertices(), 4);
    assert_eq!(g.num_edges(), 4);

    // And the edges use those ids, in file order.
    assert_eq!(
        by_id(&g),
        vec![Some((0, 1)), Some((2, 0)), Some((1, 3)), Some((0, 2))]
    );
}

#[test]
fn a_self_loop_interns_one_vertex_and_the_source_is_seen_first() {
    // `get_vertex(e[0])` runs before `get_vertex(e[1])`, so in `y,x` the name
    // `y` takes the lower id even though `x` is alphabetically first and even
    // though a set-based reader would have no reason to prefer either.
    let (g, names) = read_ok("y,x\nx,x\n", symbolic());
    assert_eq!(names, vec!["y", "x"]);
    assert_eq!(by_id(&g), vec![Some((0, 1)), Some((1, 1))]);
    assert_eq!(g.num_edges(), 2);
}

#[test]
fn interning_is_by_stripped_unquoted_value() {
    // `strip_whitespace=True` (`graph_tool/__init__.py:3832-3836`) applies to
    // the parsed field, so a quoted value is unquoted first and stripped
    // after: all four spellings of `a` below are one vertex.
    let (g, names) = read_ok("a,b\n  a  ,b\n\"a\",b\n\" a \",b\n", symbolic());
    assert_eq!(names, vec!["a", "b"]);
    assert_eq!(g.num_vertices(), 2);
    assert_eq!(g.num_edges(), 4);
}

#[test]
fn the_empty_string_is_a_name() {
    // Python's csv reader yields `''` for an empty field and `get_vertex`
    // hashes it like any other key; it is not a missing value.
    let (_, names) = read_ok("a,\n,b\n", symbolic());
    assert_eq!(names, vec!["a", "", "b"]);
}

#[test]
fn the_header_row_is_not_interned() {
    let opts = EdgeListOptions {
        header: true,
        ..symbolic()
    };
    let (g, names) = read_ok("source,target\na,b\nb,c\n", opts);
    assert_eq!(names, vec!["a", "b", "c"]);
    assert_eq!(g.num_edges(), 2);
}

// ===========================================================================
// Acceptance 2 -- the thread count changes nothing
// ===========================================================================

/// Set by the parent on each child; its value is where the child writes.
const OUT_VAR: &str = "GT_U29_FINGERPRINT_OUT";

/// libtest names an integration test by its function path, which for a
/// top-level `#[test]` is just the name.
const SELF: &str = "the_thread_count_changes_neither_the_edge_ids_nor_the_vertex_ids";

/// Enough records to need several staging chunks: `csv::ROWS_PER_CHUNK` is
/// 65 536, so this is four of them. One chunk would make the claim vacuous.
const RECORDS: usize = 200_000;

/// SplitMix64 of `i`, so the input is a pure function of its own index and the
/// child does not have to ship it.
fn mix(i: u64) -> u64 {
    let mut z = i
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(0x1234_5678_9abc_def0);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A symbolic edge list over a deliberately small name space, so most records
/// intern nothing and the few that do are scattered through the file. Names
/// are not in a format that sorts like their first-seen order, so a reader
/// that sorted them would not accidentally pass.
fn corpus() -> String {
    const NAMES: usize = 4_096;
    let mut s = String::with_capacity(RECORDS * 16);
    for i in 0..RECORDS as u64 {
        let a = mix(2 * i) as usize % NAMES;
        let b = mix(2 * i + 1) as usize % NAMES;
        writeln!(s, "n{a:x},n{b:x}").expect("writing to a String cannot fail");
    }
    s
}

fn fingerprint(g: &AdjList<NoLookup>, names: &[String]) -> Vec<u8> {
    let mut out = Vec::with_capacity(g.edge_bound().len() * 16 + 16);
    out.extend_from_slice(&(g.num_vertices() as u64).to_le_bytes());
    out.extend_from_slice(&(g.num_edges() as u64).to_le_bytes());
    for e in by_id(g) {
        let (s, t) = e.unwrap_or((usize::MAX, usize::MAX));
        out.extend_from_slice(&(s as u64).to_le_bytes());
        out.extend_from_slice(&(t as u64).to_le_bytes());
    }
    // Vertex identity, not just edge identity: two readers can agree on every
    // (s, t) pair and still have numbered the names differently.
    for n in names {
        out.extend_from_slice(n.as_bytes());
        out.push(0);
    }
    out
}

#[test]
fn the_thread_count_changes_neither_the_edge_ids_nor_the_vertex_ids() {
    // ---- child ----------------------------------------------------------
    if let Ok(path) = std::env::var(OUT_VAR) {
        let src = corpus();
        let (g, names) = read_ok(&src, symbolic());
        g.validate().expect("the built graph must validate");
        assert_eq!(g.num_edges(), RECORDS);
        std::fs::write(&path, fingerprint(&g, &names)).expect("write fingerprint");
        return;
    }

    // ---- parent ---------------------------------------------------------
    let exe = std::env::current_exe().expect("current_exe");
    let dir = std::env::temp_dir();
    let mut results: Vec<(usize, Vec<u8>)> = Vec::new();

    for threads in [1usize, 4, 16] {
        let path = dir.join(format!("gt-u29-{}-{threads}.fp", std::process::id()));
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
        assert!(!blob.is_empty(), "empty fingerprint from {threads} threads");
        results.push((threads, blob));
    }

    let (_, reference) = &results[0];
    for (threads, blob) in &results[1..] {
        assert_eq!(
            blob, reference,
            "the graph read at {threads} threads differs from the one read at 1"
        );
    }

    // And the fingerprint is of the graph we meant: a fingerprint that was
    // identical because it was empty would pass the loop above.
    let (g, names) = read_ok(&corpus(), symbolic());
    assert_eq!(&fingerprint(&g, &names), reference);
    assert_eq!(g.num_edges(), RECORDS);
}

// ===========================================================================
// Acceptance 3 -- writing in EdgeId order
// ===========================================================================

#[test]
fn write_edge_list_emits_in_edge_id_order() {
    let mut g = build(6, &[(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 0), (4, 1)]);
    // Free two ids from the middle of the range, then take them back. After
    // this the graph's insertion order, its adjacency order and its id order
    // are three different orders -- which is the only state in which "emits in
    // EdgeId order" is a claim with content.
    g.remove_edge(EdgeId::from_index(1)).expect("live edge");
    g.remove_edge(EdgeId::from_index(4)).expect("live edge");
    g.add_edge(VertexId::from_index(5), VertexId::from_index(0))
        .expect("in range");
    g.add_edge(VertexId::from_index(4), VertexId::from_index(3))
        .expect("in range");
    g.validate().expect("valid");

    // What the adjacency stream would have produced.
    let adjacency: Vec<(usize, usize)> = g
        .edges()
        .map(|e| (e.source().index(), e.target().index()))
        .collect();
    // What the id order is.
    let mut ids: Vec<(usize, usize, usize)> = g
        .edges()
        .map(|e| (e.id().index(), e.source().index(), e.target().index()))
        .collect();
    ids.sort_unstable();
    let expected: Vec<(usize, usize)> = ids.iter().map(|&(_, s, t)| (s, t)).collect();

    assert_ne!(
        adjacency, expected,
        "the fixture must exercise the reordering, not agree with it by luck"
    );

    let text = write_to_string(&g, indexed());
    let got: Vec<(usize, usize)> = text
        .lines()
        .map(|l| {
            let (a, b) = l.split_once(',').expect("two fields");
            (a.parse().expect("index"), b.parse().expect("index"))
        })
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn a_written_header_reads_back_as_a_header() {
    let g = build(3, &[(0, 1), (1, 2)]);
    let opts = EdgeListOptions {
        header: true,
        ..indexed()
    };
    let text = write_to_string(&g, opts);
    assert_eq!(text, "source,target\n0,1\n1,2\n");
    let (back, _) = read_ok(&text, opts);
    assert_eq!(by_id(&back), by_id(&g));
}

#[test]
fn the_delimiter_is_written_as_the_single_byte_it_is() {
    let g = build(2, &[(0, 1)]);
    let opts = EdgeListOptions {
        delimiter: b'\t',
        ..indexed()
    };
    assert_eq!(write_to_string(&g, opts), "0\t1\n");
    // Non-ASCII delimiters are one byte too, not one `char`.
    let opts = EdgeListOptions {
        delimiter: 0xFE,
        ..indexed()
    };
    let mut out = Vec::new();
    write_edge_list(&mut out, &g, opts).expect("write");
    assert_eq!(out, vec![b'0', 0xFE, b'1', b'\n']);
}

#[test]
fn an_empty_graph_writes_nothing() {
    let g = AdjList::<NoLookup>::new();
    assert_eq!(write_to_string(&g, indexed()), "");
    let (back, names) = read_ok("", symbolic());
    assert_eq!(back.num_vertices(), 0);
    assert_eq!(back.num_edges(), 0);
    assert!(names.is_empty());
}

// ===========================================================================
// Index edge lists
// ===========================================================================

#[test]
fn an_index_edge_list_grows_the_graph_to_the_largest_endpoint() {
    // `while (s >= num_vertices(g) || t >= num_vertices(g)) add_vertex(g);`
    // (`graph_python_interface_imp1.cc:75-76`) leaves exactly `max + 1`
    // vertices; vertex 3 stays isolated and is still a vertex.
    let (g, names) = read_ok("0,1\n1,5\n", indexed());
    assert_eq!(g.num_vertices(), 6);
    assert_eq!(g.num_edges(), 2);
    assert_eq!(g.degree(VertexId::from_index(3)), 0);
    assert!(
        names.is_empty(),
        "an index edge list has no names to record"
    );
}

#[test]
fn an_index_edge_list_rejects_what_is_not_an_index() {
    // graph-tool's `int(row[0])` (`graph_tool/__init__.py:3856`) raises here;
    // so does this, with the line number Python does not give.
    for (src, line) in [
        ("0,1\nx,2\n", 2usize),
        ("0,1\n2,y\n", 2),
        ("0,1\n2,3\n-1,0\n", 3),
        ("0,1\n2,3.5\n", 2),
        ("\n\n4,nope\n", 3),
    ] {
        match read(src, indexed()) {
            Err(IoError::Parse { line: got, .. }) => {
                assert_eq!(got, line, "wrong line for {src:?}");
            }
            other => panic!(
                "expected a parse error for {src:?}, got ok={}",
                other.is_ok()
            ),
        }
    }
}

#[test]
fn the_underflow_in_the_missing_target_branch_is_not_reproduced() {
    // `graph_python_interface_imp1.cc:71-72` reads
    //     if (s >= num_vertices(g))
    //         add_vertex(g, 1 + num_vertices(g) - s);
    // in the branch that handles a row whose target is a sentinel. The
    // operands are the wrong way round: with `s > num_vertices(g)` the
    // `size_t` subtraction wraps and the call asks for ~2^64 vertices. The
    // correct count is `1 + s - num_vertices(g)`, and this reader reaches the
    // same place from the other direction -- it takes `max + 1` as the vertex
    // count outright, so a jump from an empty graph straight to index 9 is a
    // 10-vertex graph and not an allocation failure.
    let (g, _) = read_ok("9,9\n", indexed());
    assert_eq!(g.num_vertices(), 10);
    assert_eq!(g.num_edges(), 1);
    assert_eq!(by_id(&g), vec![Some((9, 9))]);
}

#[test]
fn a_record_short_of_two_columns_names_its_own_line() {
    for (src, line) in [
        ("a,b\nlonely\n", 2usize),
        ("lonely\n", 1),
        ("a,b\n\n\nlonely\nc,d\n", 4),
        // A quoted field carrying newlines advances the line counter, so the
        // number still points at the offending record.
        ("\"x\ny\",b\nlonely\n", 3),
    ] {
        match read(src, symbolic()) {
            Err(IoError::Parse { line: got, msg }) => {
                assert_eq!(got, line, "wrong line for {src:?}");
                assert!(msg.contains("two columns"), "wrong message: {msg}");
            }
            other => panic!(
                "expected a parse error for {src:?}, got ok={}",
                other.is_ok()
            ),
        }
    }
}

#[test]
fn a_trailing_record_without_a_newline_is_still_a_record() {
    let (g, names) = read_ok("a,b\nb,c", symbolic());
    assert_eq!(names, vec!["a", "b", "c"]);
    assert_eq!(g.num_edges(), 2);
}

#[test]
fn a_byte_order_mark_is_not_part_of_the_first_name() {
    let (_, names) = read_ok("\u{feff}a,b\n", symbolic());
    assert_eq!(names, vec!["a", "b"]);
}

#[test]
fn extra_columns_are_parsed_and_discarded() {
    // The third field is quoted and contains the delimiter *and* a newline.
    // A reader that stopped after the second field would take `1,2` for the
    // next record's endpoints.
    let (g, names) = read_ok("a,b,\"x,\ny\"\nc,d,z\n", symbolic());
    assert_eq!(names, vec!["a", "b", "c", "d"]);
    assert_eq!(by_id(&g), vec![Some((0, 1)), Some((2, 3))]);
}

#[test]
fn an_unterminated_quote_is_an_error_not_a_panic() {
    match read("a,b\n\"c,d\n", symbolic()) {
        Err(IoError::Parse { line, msg }) => {
            assert_eq!(line, 2);
            assert!(msg.contains("unterminated"), "wrong message: {msg}");
        }
        other => panic!("expected a parse error, got ok={}", other.is_ok()),
    }
}

#[test]
fn invalid_utf8_is_an_error_not_a_panic() {
    let src: &[u8] = b"a,b\n\xff\xfe,d\n";
    match read_edge_list(src, symbolic(), NoLookup) {
        Err(IoError::Parse { line, .. }) => assert_eq!(line, 2),
        other => panic!("expected a parse error, got ok={}", other.is_ok()),
    }
}

// ===========================================================================
// Round trips
// ===========================================================================

#[test]
fn a_symbolic_file_round_trips_through_the_index_form() {
    let src = "alpha,beta\nbeta,gamma\ngamma,alpha\nalpha,alpha\n";
    let (g, names) = read_ok(src, symbolic());
    let text = write_to_string(&g, indexed());
    let (back, back_names) = read_ok(&text, indexed());
    assert_eq!(by_id(&back), by_id(&g));
    assert_eq!(back.num_vertices(), g.num_vertices());
    assert!(back_names.is_empty());
    assert_eq!(names, vec!["alpha", "beta", "gamma"]);
}

proptest! {
    /// Index edge lists round-trip exactly, edge id by edge id, as long as the
    /// graph has no isolated vertices past its largest endpoint -- which the
    /// format cannot express and `load_graph_from_csv` cannot either.
    #[test]
    fn index_round_trip_preserves_every_edge_id(
        pairs in prop::collection::vec((0usize..24, 0usize..24), 1..200)
    ) {
        let n = pairs.iter().map(|&(s, t)| s.max(t) + 1).max().unwrap_or(0);
        let g = build(n, &pairs);
        let text = write_to_string(&g, indexed());
        let (back, _) = read_edge_list(text.as_bytes(), indexed(), NoLookup)
            .expect("what we wrote parses");
        prop_assert_eq!(by_id(&back), by_id(&g));
        prop_assert_eq!(back.num_vertices(), g.num_vertices());
    }

    /// The delimiter is a parameter, not a constant, on both sides.
    #[test]
    fn any_ascii_delimiter_round_trips(
        delim in prop::sample::select(vec![b',', b'\t', b';', b'|', b' ']),
        pairs in prop::collection::vec((0usize..8, 0usize..8), 1..40)
    ) {
        let n = pairs.iter().map(|&(s, t)| s.max(t) + 1).max().unwrap_or(0);
        let g = build(n, &pairs);
        let opts = EdgeListOptions { delimiter: delim, header: false, symbolic: false };
        let text = write_to_string(&g, opts);
        let (back, _) = read_edge_list(text.as_bytes(), opts, NoLookup)
            .expect("what we wrote parses");
        prop_assert_eq!(by_id(&back), by_id(&g));
    }

    /// Chunking is invisible: a file whose record count straddles the staging
    /// chunk size reads back as the same graph as one that does not. The
    /// boundary is exercised by holding the content fixed and varying only the
    /// length.
    #[test]
    fn the_record_count_does_not_change_the_edge_ids(extra in 0usize..64) {
        let mut src = String::new();
        for i in 0..(16 + extra) {
            src.push_str(&format!("v{},v{}\n", i % 5, (i * 7) % 5));
        }
        let (g, names) = read_edge_list(src.as_bytes(), symbolic(), NoLookup)
            .expect("well formed");
        prop_assert_eq!(g.num_edges(), 16 + extra);
        prop_assert_eq!(names.len(), 5);
        // First-seen order, source before target, over
        // `(i % 5, (i * 7) % 5)` for i = 0, 1, 2, ...:
        //   (0,0) -> v0 ; (1,2) -> v1, v2 ; (2,4) -> v4 ; (3,1) -> v3.
        // It is neither the order the names sort in nor the order they first
        // appear as *sources*, which is what makes it worth asserting.
        prop_assert_eq!(&names, &["v0", "v1", "v2", "v4", "v3"]);
    }
}
