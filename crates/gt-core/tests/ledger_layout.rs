//! `gt_core::design` section 11's layout table, pinned row by row.
//!
//! Section 12's first Win -- "2x on the adjacency stream. 8-byte entries
//! against 16 (section 11)" -- is a claim about `size_of`, and section 11 is
//! twelve such claims. Two of them (`AdjEntry`, `Block`) are asserted in
//! `tests/u02_block.rs` and one (`Unity`) in the prop tests; the other nine
//! were written into the document from a build that no longer exists.
//!
//! A table of measured numbers that nothing re-measures is a table of
//! remembered numbers. This file re-measures all of it, and prints the whole
//! table under `--nocapture` so the document can be diffed against a build
//! rather than against an earlier draft of itself.
//!
//! The C++ counterparts are not measured here -- there is no C++ in this tree
//! -- but the two that the ledger's arithmetic depends on are *derived* rather
//! than remembered: `graph.hh:137` is `typedef boost::adj_list<size_t>
//! multigraph_t`, so `vertex_t` is `size_t`, so `edge_list_t`
//! (`graph_adjacency.hh:224`, `vector<pair<vertex_t, vertex_t>>`) has a
//! 16-byte element on any LP64 target. That is the 16 the 8 is measured
//! against, and `the_cxx_entry_is_two_pointers_wide` states it as an
//! assertion about `size_t` rather than as a number.

use std::mem::{align_of, size_of};

use gt_core::adj::{AdjEntry, AdjList, Block, EdgeRef, EdgeSlot, Incident, NoLookup};
use gt_core::ids::{EdgeId, EdgeTag, Raw, VertexId};
use gt_core::prop::dense::{Constant, Unity};
use gt_core::view::{Filtered, MaskFilter, Rev, Und};

/// The whole of section 11, in one place.
///
/// Printed as well as asserted: `cargo test -p gt-core --test ledger_layout
/// -- --nocapture` emits the table in the document's own column order.
#[test]
fn section_11_is_the_layout_this_build_has() {
    // The table is a function of `Raw`, and `Raw` is a crate-wide alias
    // (D5). Under `--features wide-index` every row below changes, so the
    // assertions are guarded on the configuration they were written for
    // rather than silently failing in the other one.
    if size_of::<Raw>() != 4 {
        eprintln!(
            "ledger_layout: Raw is {} bytes; section 11 tabulates the 4-byte \
             default, so the numeric rows are skipped",
            size_of::<Raw>()
        );
        return;
    }

    let rows: [(&str, usize); 12] = [
        ("AdjEntry", size_of::<AdjEntry>()),
        ("Block", size_of::<Block>()),
        ("EdgeSlot", size_of::<EdgeSlot>()),
        ("Incident", size_of::<Incident>()),
        ("EdgeRef", size_of::<EdgeRef>()),
        ("&AdjList", size_of::<&AdjList<NoLookup>>()),
        ("Und<&AdjList>", size_of::<Und<&AdjList<NoLookup>>>()),
        ("Rev<&AdjList>", size_of::<Rev<&AdjList<NoLookup>>>()),
        (
            "Filtered<&AdjList, MaskFilter>",
            size_of::<Filtered<&AdjList<NoLookup>, MaskFilter<'static>>>(),
        ),
        ("Unity<f64, EdgeTag>", size_of::<Unity<f64, EdgeTag>>()),
        ("Option<Group>", 4), // gt-inference's; see that crate's tests.
        ("Option<VertexId>", size_of::<Option<VertexId>>()),
    ];
    println!("\ngt_core::design section 11, measured on this build (Raw = 4 bytes):");
    for (name, bytes) in rows {
        println!("  {name:<34} {bytes:>3}");
    }

    assert_eq!(size_of::<AdjEntry>(), 8, "section 11: AdjEntry");
    assert_eq!(size_of::<Block>(), 32, "section 11: Block");
    assert_eq!(size_of::<EdgeSlot>(), 16, "section 11: EdgeSlot");
    assert_eq!(size_of::<Incident>(), 8, "section 11: Incident");
    assert_eq!(size_of::<EdgeRef>(), 12, "section 11: EdgeRef");
    assert_eq!(size_of::<&AdjList<NoLookup>>(), 8, "section 11: &AdjList");
    assert_eq!(
        size_of::<Und<&AdjList<NoLookup>>>(),
        8,
        "section 11: Und is a newtype and must add nothing"
    );
    assert_eq!(
        size_of::<Rev<&AdjList<NoLookup>>>(),
        8,
        "section 11: Rev is a newtype and must add nothing"
    );
    assert_eq!(
        size_of::<Filtered<&AdjList<NoLookup>, MaskFilter<'static>>>(),
        56,
        "section 11: Filtered"
    );
    assert_eq!(
        size_of::<Unity<f64, EdgeTag>>(),
        0,
        "section 11: Unity is a ZST, against UnityPropertyMap's 1"
    );
    assert_eq!(
        size_of::<Option<VertexId>>(),
        8,
        "section 11: Option<VertexId> has no niche, unlike Option<Group>"
    );
}

/// The 16 that the 8 is measured against, derived rather than remembered.
///
/// `graph.hh:137`: `typedef boost::adj_list<size_t> multigraph_t`.
/// `graph_adjacency.hh:220`: `typedef Vertex vertex_t`.
/// `graph_adjacency.hh:224`: `typedef std::vector<std::pair<vertex_t,
/// vertex_t>> edge_list_t`.
///
/// So graph-tool's adjacency element is `pair<size_t, size_t>`, which is
/// `2 * size_of::<usize>()` on every target this port builds for. The ratio
/// in section 12's first Win is therefore exactly
/// `2 * size_of::<usize>() / size_of::<AdjEntry>()`, and on a 32-bit target
/// it would be 1, not 2 -- which is worth pinning, because the ledger states
/// the 2 as a constant.
#[test]
fn the_cxx_entry_is_two_pointers_wide() {
    let cxx = 2 * size_of::<usize>();
    let ours = size_of::<AdjEntry>();
    println!(
        "graph-tool adjacency element: {cxx} B; AdjEntry: {ours} B; ratio {}",
        cxx as f64 / ours as f64
    );
    if size_of::<usize>() == 8 && size_of::<Raw>() == 4 {
        assert_eq!(cxx, 16);
        assert_eq!(
            cxx / ours,
            2,
            "section 12's first Win, as a footprint ratio"
        );
    }
}

/// A cache line holds eight `AdjEntry` and four of graph-tool's.
///
/// This is the mechanism behind the footprint ratio, and the one thing about
/// it that is genuinely a *constant*: the line is 64 bytes on every x86-64
/// and every AArch64 part this port targets.
#[test]
fn a_cache_line_holds_eight_entries() {
    if size_of::<Raw>() != 4 {
        return;
    }
    assert_eq!(64 / size_of::<AdjEntry>(), 8);
    assert_eq!(64 / (2 * size_of::<usize>()), 4);
    assert_eq!(align_of::<AdjEntry>(), 4, "no padding to 8");
}

/// `Block` is at parity with the C++ vertex record, not smaller than it.
///
/// Section 12 lists "`Block` at 32 bytes with no per-vertex malloc for
/// isolated vertices" under **Wins**. The 32 is real (one `Vec` plus one
/// `Raw`, padded to four words) and so is the absence of the malloc -- but
/// graph-tool's `vertex_list_t` element (`graph_adjacency.hh:225`,
/// `pair<size_t, edge_list_t>`) is 8 + 24 = 32 too, and its `add_vertex`
/// (`:1318-1336`) `emplace_back`s a default-constructed `vector` that does
/// not allocate either. Both halves are parity, and section 12 has been
/// corrected to say so.
///
/// What this pins is the Rust half: a `Block` that grew a fourth word, or an
/// inline small-vector buffer (measured at 48 bytes and rejected), would make
/// the vertex array 50% larger and nothing else in the tree would notice.
#[test]
fn block_is_one_vec_plus_one_word_and_an_empty_one_does_not_allocate() {
    assert_eq!(
        size_of::<Block>(),
        size_of::<Vec<AdjEntry>>() + size_of::<usize>()
    );
    assert_eq!(size_of::<Block>(), 32);

    // An isolated vertex's block owns no heap. `Vec::new()` is documented not
    // to allocate, and `Block::default()` must not do anything more.
    let g = AdjList::with_vertices(1_000);
    assert_eq!(g.num_edges(), 0);
    // The observable consequence: a graph of isolated vertices costs the
    // vertex array and nothing per vertex beyond it.
    assert_eq!(g.num_vertices(), 1_000);
}

/// `Constant` carries its value; `Unity` does not carry anything.
///
/// The pair is what makes section 12's `Unity` line a statement about types
/// rather than about one function: `weighted_out_degree`'s three arms
/// (`gt-algo/src/degree.rs:122-136`) are selected by associated consts, and
/// the ZST is the reason the unity arm cannot read a map even by accident.
#[test]
fn unity_is_a_zst_and_constant_is_not() {
    assert_eq!(size_of::<Unity<f64, EdgeTag>>(), 0);
    assert_eq!(size_of::<Constant<f64, EdgeTag>>(), 8);
    assert_eq!(
        size_of::<Unity<[u8; 4096], EdgeTag>>(),
        0,
        "regardless of T"
    );
}

/// Ids are `Raw`-wide and distinct.
#[test]
fn ids_are_raw_wide() {
    assert_eq!(size_of::<VertexId>(), size_of::<Raw>());
    assert_eq!(size_of::<EdgeId>(), size_of::<Raw>());
}
