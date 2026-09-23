//! Degree and weighted-degree kernels.
//!
//! These are the port of `graph_selectors.hh:104-129`, where three separate
//! overloads exist to give a constant-weight fast path. Two of the three are
//! *uninstantiable dead code*: `:109` and `:178` read `weight.c` on
//! `ConstantPropertyMap`, whose member is the private `_c`
//! (`graph_properties.hh:677`), so the constant-weight fast path silently does
//! not exist in graph-tool.
//!
//! ## What the three overload sets become here
//!
//! `in_degreeS`, `out_degreeS` and `total_degreeS` each dispatch on four
//! things at once in C++: the weight map's *type* (`no_weightS`,
//! `UnityPropertyMap`, `ConstantPropertyMap`, anything else) and the graph's
//! `directed_category`. Here the second axis is the view algebra and the first
//! is two associated `const`s:
//!
//! * `W::IS_UNITY` reproduces `get_out_degree(v, g, const UnityPropertyMap&)`
//!   (`graph_selectors.hh:182-187`) -- return the plain degree;
//! * `W::IS_CONSTANT` reproduces `out_degree(v, g) * weight.c`
//!   (`:173-179`), the overload that cannot be instantiated in C++ because
//!   `weight.c` does not name an accessible member. [`Constant`]'s field is
//!   public, so the fast path exists here (defect #44);
//! * everything else is `:190-197`'s accumulation loop.
//!
//! Both consts are `const`, so the branch is resolved at monomorphisation and
//! the untaken arms are dead code before LLVM sees them: the `Unity`
//! instantiation of [`weighted_out_degree`] contains no loop at all.
//!
//! ## Directedness
//!
//! `get_in_degree(v, g, std::false_type, Weight&&)` (`graph_selectors.hh:133-148`)
//! returns `val_t()` -- zero -- for an undirected graph, so C++ answers a
//! question it cannot answer with a plausible number. [`weighted_in_degree`]
//! is bounded on [`Bidirectional`] instead, which [`Und`](gt_core::view::Und)
//! deliberately does not implement, so the same call is `error[E0599]`.
//!
//! `get_total_degree(..., std::false_type, ...)` (`:239-244`) forwards to
//! `out_degreeS`, and that identity survives here without a special case:
//! `Und<G>::all_edges` and `Und<G>::out_edges` are the same run
//! (`view/undirected.rs:106-116`), so [`weighted_degree`] and
//! [`weighted_out_degree`] agree on an undirected view by construction.
//!
//! ## Accumulator width
//!
//! C++ accumulates in `property_traits<Weight>::value_type`, so an `int32_t`
//! weight map sums into an `int32_t` and overflows silently at 2³¹. Every
//! kernel here accumulates in `f64` via [`ToF64`], which is exact to 2⁵³ for
//! the integer members and cannot wrap. The bound is [`ToF64`] and not
//! `Into<f64>` precisely so that `i64` -- the width graph-tool actually
//! instantiates -- is admissible; `i64: Into<f64>` does not hold. See
//! `tests/ui/u15_i64_weight_is_admissible.rs`.

use gt_core::adj::Incident;
use gt_core::graph::{Bidirectional, GraphRef, VertexList};
use gt_core::ids::{EdgeTag, VertexId, VertexTag};
use gt_core::prop::{DenseProp, ReadProp, ToF64};

// ---------------------------------------------------------------------------
// The two shared bodies
// ---------------------------------------------------------------------------

/// `d += get(weight, *e)` over a neighbourhood (`graph_selectors.hh:190-197`).
///
/// Takes the iterator by value and never collects: the C++ walks
/// `out_edges_range(v, g)` straight out of storage, and materialising
/// `&[EdgeId]` per vertex would cost an allocation per vertex.
#[inline(always)]
fn sum_weights<I, W>(w: &W, edges: I) -> f64
where
    I: Iterator<Item = Incident>,
    W: ReadProp<EdgeTag>,
    W::Value: ToF64,
{
    let mut acc = 0.0f64;
    for i in edges {
        // `get_ref`, not `get`: the read is by reference, so a weight map over
        // a non-`Copy` member would still not clone here. `ToF64` is `Copy`,
        // so the deref is a register move.
        acc += (*w.get_ref(i.edge)).to_f64();
    }
    acc
}

/// `out_degree(v, g) * weight.c` (`graph_selectors.hh:173-179`).
///
/// The constant is read from the *first surviving incident edge* rather than
/// from a synthesised key: a filtered view's neighbourhood may be empty while
/// the underlying edge space is not, and a map is only obliged to answer for
/// keys inside its bound. One `next()` is O(1), so the result still contains
/// no loop.
///
/// Multiplication, not repeated addition: that is what makes this a fast path
/// rather than a rewriting of [`sum_weights`], and for a large degree the two
/// differ in `f64` -- `c * n` is one correctly rounded operation where the
/// sum accumulates `n - 1` roundings.
#[inline(always)]
fn constant_weight<I, W>(w: &W, mut edges: I, degree: usize) -> f64
where
    I: Iterator<Item = Incident>,
    W: ReadProp<EdgeTag>,
    W::Value: ToF64,
{
    match edges.next() {
        None => 0.0,
        Some(i) => degree as f64 * (*w.get_ref(i.edge)).to_f64(),
    }
}

// ---------------------------------------------------------------------------
// The three selectors
// ---------------------------------------------------------------------------

/// Sum of a weight map over `v`'s out-edges.
///
/// Takes an iterator-shaped neighbourhood, not a slice: materialising
/// `&[EdgeId]` per vertex costs an allocation per vertex, where C++ walks
/// `out_edges_range(v, g)` straight out of storage.
///
/// `W::IS_UNITY` folds the whole loop away, leaving the degree.
#[inline]
pub fn weighted_out_degree<G, W>(g: G, v: VertexId, w: &W) -> f64
where
    G: GraphRef,
    W: ReadProp<EdgeTag>,
    W::Value: ToF64,
{
    if W::IS_UNITY {
        // `graph_selectors.hh:182-187`. Nothing is read from `w` at all, so
        // this arm is the only code the `Unity` monomorphisation contains.
        g.out_degree(v) as f64
    } else if W::IS_CONSTANT {
        constant_weight(w, g.out_edges(v), g.out_degree(v))
    } else {
        sum_weights(w, g.out_edges(v))
    }
}

/// Sum of a weight map over `v`'s in-edges.
///
/// Bounded on [`Bidirectional`]: there is no undirected arm returning a
/// plausible zero (`graph_selectors.hh:133-148`).
#[inline]
pub fn weighted_in_degree<G, W>(g: G, v: VertexId, w: &W) -> f64
where
    G: Bidirectional,
    W: ReadProp<EdgeTag>,
    W::Value: ToF64,
{
    if W::IS_UNITY {
        g.in_degree(v) as f64
    } else if W::IS_CONSTANT {
        constant_weight(w, g.in_edges(v), g.in_degree(v))
    } else {
        sum_weights(w, g.in_edges(v))
    }
}

/// Sum of a weight map over every edge incident to `v`.
///
/// `total_degreeS` (`graph_selectors.hh:213-246`) is `in_degreeS + out_degreeS`
/// on a directed graph and `out_degreeS` alone on an undirected one. Walking
/// `all_edges` is both at once, because the adjacency block *is* the two
/// halves concatenated and an undirected view's `all_edges` is that same run
/// (D2). A self-loop occupies one slot in each half and is therefore counted
/// twice, which is what `degree(v, g)` (`graph_adjacency.hh:1069-1073`) says.
#[inline]
pub fn weighted_degree<G, W>(g: G, v: VertexId, w: &W) -> f64
where
    G: GraphRef,
    W: ReadProp<EdgeTag>,
    W::Value: ToF64,
{
    if W::IS_UNITY {
        g.degree(v) as f64
    } else if W::IS_CONSTANT {
        constant_weight(w, g.all_edges(v), g.degree(v))
    } else {
        sum_weights(w, g.all_edges(v))
    }
}

// ---------------------------------------------------------------------------
// Bulk forms
// ---------------------------------------------------------------------------

/// Fill a vertex map with out-degrees.
///
/// The map is grown to `g.vertex_bound()` -- the *unfiltered* allocation
/// bound, never [`num_vertices`](gt_core::graph::GraphBase::num_vertices) --
/// and then **the whole run is cleared** before the surviving vertices are
/// written. Both halves matter:
///
/// * sizing from the bound is the half `copy_property` gets wrong
///   (`graph_copy.cc:66-73` reserves the filtered count and writes at
///   unfiltered indices);
/// * clearing is what makes the answer a function of `g` alone. A map reused
///   across two calls, or one filled by an earlier kernel, would otherwise
///   report a stale degree at every vertex this view filters out -- and
///   `get_unchecked(size = 0)` (`dispatch.hh:171-177`) hands graph-tool
///   exactly such a map.
///
/// Panics if `out` belongs to a different graph than `g`; a map reaching a
/// kernel has been sized for it, and the `GraphId` comparison happens once
/// here rather than per access.
pub fn out_degrees<G>(g: G, out: &mut DenseProp<i64, VertexTag>)
where
    G: GraphRef + VertexList,
{
    let mut view = out
        .sized_for(g.vertex_bound())
        .expect("the out map must belong to the graph being measured");
    let slots = view.as_mut_slice();
    slots.fill(0);
    for v in g.vertices() {
        // `vertices()` yields only indices inside the bound the slice was cut
        // to, so this is one bounds check LLVM removes rather than a class of
        // out-of-range write.
        slots[v.index()] = g.out_degree(v) as i64;
    }
}

/// The degree histogram.
///
/// `hist[d]` is the number of vertices of total degree `d`, so the result has
/// length `max_degree + 1` and is empty for a graph with no vertices. "Total"
/// is [`GraphRef::degree`]: in- plus out-degree on a directed view, the whole
/// adjacency block on an undirected one, which is `total_degreeS`
/// (`graph_selectors.hh:213-246`) in both cases.
///
/// Unit-width bins from zero, not `get_vertex_histogram`'s `[min, max]` range
/// (`stats/graph_histograms.cc:44-58`): the index *is* the degree, so the
/// caller needs no bin table to read the answer back, and `hist.iter().sum()`
/// is `num_vertices` exactly. graph-tool's own binning is where defect #46
/// lives (`histogram/graph_histogram.hh:944`, `trim_points` re-adding without
/// clearing); there is nothing to trim here.
pub fn degree_histogram<G>(g: G) -> Vec<usize>
where
    G: GraphRef + VertexList,
{
    let mut hist: Vec<usize> = Vec::new();
    for v in g.vertices() {
        let d = g.degree(v);
        if d >= hist.len() {
            // Amortised: the vector grows at most `max_degree + 1` times over
            // the whole scan, never once per vertex.
            hist.resize(d + 1, 0);
        }
        hist[d] += 1;
    }
    hist
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    //! In-crate checks of the const-folding contract. The behavioural suite is
    //! `tests/u15_degree.rs`; what is here is what needs no `AdjList` at all.

    use super::*;
    use gt_core::prop::dense::{ConstI64, Constant, EdgeProp, Unity};

    /// The two consts the three kernels branch on, at the types the port
    /// actually passes. `ConstI64<1>` routing into the unity arm is the case
    /// `is_unity_map_v` (`graph_properties.hh:729`) has no way to express.
    #[test]
    fn the_branch_is_a_const() {
        const U: bool = <Unity<f64, EdgeTag> as ReadProp<EdgeTag>>::IS_UNITY;
        const C: bool = <Constant<f64, EdgeTag> as ReadProp<EdgeTag>>::IS_CONSTANT;
        const CU: bool = <Constant<f64, EdgeTag> as ReadProp<EdgeTag>>::IS_UNITY;
        const K1: bool = <ConstI64<1, EdgeTag> as ReadProp<EdgeTag>>::IS_UNITY;
        const K2: bool = <ConstI64<2, EdgeTag> as ReadProp<EdgeTag>>::IS_UNITY;
        const D: bool = <EdgeProp<f64> as ReadProp<EdgeTag>>::IS_CONSTANT;
        // `const` blocks: the whole claim is that these fold, so the check
        // belongs at compile time rather than at run time.
        const { assert!(U && C && K1) };
        const { assert!(!CU && !K2 && !D) };
    }

    /// `i64` is admissible, which is the whole reason the bound is [`ToF64`]:
    /// `i64: Into<f64>` does not hold, so an `Into<f64>` bound would have
    /// excluded the one width graph-tool instantiates
    /// (`src/graph/graph.hh:137`).
    #[test]
    fn to_f64_widens_i64_exactly_where_f64_allows() {
        let big: i64 = 1 << 60;
        assert_eq!(big.to_f64(), 1_152_921_504_606_846_976.0);
        // Three of them: 3 * 2^60 needs two mantissa bits and is exact.
        assert_eq!(3.0 * big.to_f64(), 3_458_764_513_820_540_928.0);
        // And the sum route agrees, because nothing here is ever rounded.
        assert_eq!(big.to_f64() + big.to_f64() + big.to_f64(), 3.0 * big.to_f64());
    }
}
