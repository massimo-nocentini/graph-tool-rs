//! U15: a weight map over a non-numeric member of the value universe.
//!
//! `scalar_edge_properties` is a *separate* axis from `edge_properties` in
//! graph-tool (`graph_properties.hh`), and keeping the two apart is a runtime
//! check: `get_edge_histogram` throws `ValueException("Edge property must be
//! of scalar type.")` from `stats/graph_histograms.cc:66-67` when the caller
//! gets it wrong. Nothing stops a *C++* caller of `out_degreeS` from passing a
//! `string` map; `d += get(weight, *e)` then fails deep inside the template.
//!
//! Here the axis is the `W::Value: ToF64` bound, so the same mistake is
//! rejected at the call site:
//!
//! ```text
//! error[E0277]: the trait bound `String: ToF64` is not satisfied
//! ```

use gt_algo::degree::weighted_out_degree;
use gt_core::adj::AdjList;
use gt_core::ids::VertexId;
use gt_core::prop::DenseProp;
use gt_core::prop::dense::EdgeProp;

fn main() {
    let g = AdjList::with_vertices(2);
    let w: EdgeProp<String> = DenseProp::new(g.graph_id());
    let _ = weighted_out_degree(&g, VertexId::from_index(0), &w);
}
