# graph-tool-rs

A Rust port of [graph-tool](https://graph-tool.skewed.de) 3.8, the Python
library for graph analysis and network inference built on C++ kernels.

The port keeps graph-tool's shape (a dynamically typed Python frontend over
statically typed kernels) but moves failures to compile time wherever it can.
Missing dispatch arms, undersized property maps, and Python values reaching
parallel loops become type errors instead of runtime bugs. It is not a
performance rewrite.

The workspace has five crates:

| crate          | contents                                                        |
|----------------|-----------------------------------------------------------------|
| `gt-core`      | identifiers, adjacency storage, views, property maps, parallelism |
| `gt-algo`      | traversal, components, degree, centrality, topology             |
| `gt-inference` | stochastic block model and Metropolis machinery                 |
| `gt-io`        | `.gt`, GraphML, DOT and CSV formats                             |
| `gt-py`        | the PyO3 extension module                                       |

**Status:** work in progress, built unit by unit following the implementation
plan. A few `todo!()` bodies remain.

## Documentation

The API docs, the design document and the implementation plan are all rustdoc:

* Online: <https://massimo-nocentini.github.io/graph-tool-rs/>
* Design: [`gt_core::design`](https://massimo-nocentini.github.io/graph-tool-rs/gt_core/design/index.html)
* Plan: [`gt_core::implementation_plan`](https://massimo-nocentini.github.io/graph-tool-rs/gt_core/implementation_plan/index.html)

To rebuild them locally into `docs/`:

```sh
make doc
```

## Building

Requires Rust 1.98 or newer.

```sh
cargo build --workspace
cargo test --workspace
```

## License

LGPL-3.0-or-later, the same as graph-tool.
