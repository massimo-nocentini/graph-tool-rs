//! U15: the weight bound is `ToF64`, and `i64` therefore passes it.
//!
//! graph-tool instantiates `adj_list<size_t>` and its scalar property members
//! include `int64_t` (`graph_properties.hh`'s `scalar_properties`), so an
//! `i64`-valued edge weight is not a corner case: it is the common one.
//!
//! The obvious Rust bound for "can be read as a float" is `Into<f64>`, and it
//! is **wrong**: `impl From<i64> for f64` does not exist, because the
//! conversion is lossy above 2^53. An `Into<f64>` bound would have silently
//! excluded `i64` from every weighted-degree call site, and the failure would
//! have shown up as `error[E0277]` at the *caller*. `ToF64::to_f64` widens
//! explicitly and accepts that rounding, which is what
//! `boost::property_traits<Weight>::value_type` arithmetic does anyway.
//!
//! This fixture is a `pass` case: it compiles and runs, and the assertion is
//! that the widening is exact at a magnitude past 2^53.

use gt_algo::degree::weighted_out_degree;
use gt_core::adj::AdjList;
use gt_core::ids::VertexId;
use gt_core::prop::dense::EdgeProp;
use gt_core::prop::{DenseProp, ToF64};

/// Nothing calls this; naming the bound is the point.
fn requires_to_f64<T: ToF64>(x: T) -> f64 {
    x.to_f64()
}

fn main() {
    let mut g = AdjList::with_vertices(2);
    let a = VertexId::from_index(0);
    let b = VertexId::from_index(1);
    g.add_edge(a, b).expect("add_edge");
    g.add_edge(a, b).expect("add_edge");
    g.add_edge(a, b).expect("add_edge");

    let big: i64 = 1 << 60;
    let w: EdgeProp<i64> = DenseProp::from_vec(g.graph_id(), vec![big; 3]);

    // 3 * 2^60 needs two mantissa bits, so the answer is exact.
    assert_eq!(weighted_out_degree(&g, a, &w), 3_458_764_513_820_540_928.0);
    assert_eq!(requires_to_f64(big), 1_152_921_504_606_846_976.0);
}
