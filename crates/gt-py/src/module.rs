//! The importable extension module.
//!
//! Every entry point follows the same three-step shape, and the order matters:
//!
//! 1. take the `Python<'py>` token and borrow the graph and the property maps;
//! 2. size every map with
//!    [`DenseProp::sized_for`](gt_core::prop::DenseProp::sized_for) -- read-only
//!    maps too, because graph-tool's checked map auto-grows on read
//!    (`fast_vector_property_map.hh:132-136`; the plan's `:129` is three lines
//!    above the `soft_reserve(i + 1)` that actually does it), so `g.vp.x[v]`
//!    on a never-written map must keep returning the default;
//! 3. hand the sized views to a kernel inside [`detach`](crate::gil::detach),
//!    whose `F: Send` bound is what keeps the token out of the region.
//!
//! Step 2 is the two-phase reserve-then-hand-out that `_get_any`
//! (`graph_tool/__init__.py:363-373`) implements in nine lines of Python. It
//! does not disappear; it moves from Python into one place in Rust, where the
//! type system makes forgetting it impossible rather than merely unlikely.
//!
//! ## Where the fifteenth member went
//!
//! `#[pyclass]` requires `Send`, and [`PyValue`] is `!Send` by construction
//! ([DESIGN](gt_core::design) section 7). That is not an obstacle to work around; it is the
//! guarantee arriving at the boundary, and the only honest way to spell it is
//! **two classes**:
//!
//! * [`PyPropertyMap`] carries the fourteen GIL-free members. It is `Send`,
//!   so CPython may hand it to any thread, and its kernels run inside
//!   [`detach`](crate::gil::detach).
//! * [`PyObjectPropertyMap`] carries `python::object`. It is
//!   `#[pyclass(unsendable)]`, because the store it holds is `!Send` and the
//!   compiler will not be persuaded otherwise, and every one of its loops
//!   holds the token.
//!
//! Folding both into one class would have forced `unsendable` onto the
//! fourteen members that do not need it, which is the pessimisation
//! `graph_properties_copy.cc:35-42` makes in the other direction -- and one
//! `unsendable` class is a smaller lie than fourteen.
//!
//! ## Errors
//!
//! No entry point here panics on a caller error. `GraphError`, `PropError` and
//! `DispatchError` become a `PyErr` at the boundary and nothing above relies
//! on `catch_unwind`, which is what [DESIGN](gt_core::design) section 10 promises when
//! `[profile.release] panic = "abort"` removes unwinding entirely. The
//! exception classes are graph-tool's own translation table
//! (`graph_bind.cc:63-76`): `ValueException -> ValueError`,
//! `GraphException -> RuntimeError`.

// pyo3 0.22's `#[pymethods]` expansion wraps every fallible return in an
// `Into<PyErr>` that is the identity when the error already is a `PyErr`.
// Scoped to this module, which contains nothing but pyo3 glue.
#![allow(clippy::useless_conversion)]

use crate::gil::detach;
use crate::value::{ScalarKernel, ScalarV};
use gt_core::bound::Bound as IndexBound;
use gt_core::error::{DispatchError, GraphError, PropError};
use gt_core::ids::{EdgeId, EdgeTag, GraphId, IdTag, VertexId, VertexTag};
use gt_core::prop::dispatch::AnyMap;
use gt_core::prop::{DenseProp, LongDouble, PyValue, Scalar, ValueKind};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

// ---------------------------------------------------------------------------
// Error translation
// ---------------------------------------------------------------------------

/// `GraphException -> RuntimeError`, `ValueException -> ValueError`
/// (`graph_bind.cc:67-73`).
///
/// A descriptor that does not name a live element is the condition
/// `graph_tool/__init__.py:2149` raises `ValueError("Invalid vertex index")`
/// for; the two exhaustion cases and the invariant breach have no graph-tool
/// analogue at all -- `Vertex` is also the index type there and a graph grown
/// past `null_vertex()` silently collides with it (`graph_adjacency.hh:924`).
fn graph_err(e: GraphError) -> PyErr {
    match e {
        GraphError::NoSuchVertex(_) | GraphError::NoSuchEdge(_) => {
            PyValueError::new_err(e.to_string())
        }
        GraphError::EdgeIdSpaceExhausted { .. }
        | GraphError::VertexIdSpaceExhausted { .. }
        | GraphError::Invariant(_) => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Every `PropError` is a `ValueException` in graph-tool's vocabulary: the six
/// sites that can raise one (`graph_properties.hh:266, :319, :447, :457`,
/// `graph_copy.cc:99, :181`) all throw that type.
fn prop_err(e: PropError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The narrowing failure. graph-tool's counterpart is `DispatchNotFound`,
/// whose message is "This is a graph_tool bug. :-(" (`dispatch.hh:86-88`);
/// this one names the offered member and the accepted set.
fn dispatch_err(e: DispatchError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A key outside the index space of its own graph.
///
/// graph-tool has no such check on this path: the checked map's `operator[]`
/// does `soft_reserve(i + 1)` (`fast_vector_property_map.hh:132-136`) and
/// happily allocates `i + 1` slots for any `i`, so `g.vp.x[1 << 40]` on a
/// five-vertex graph is an allocation, not an error. **Deliberate
/// divergence:** a [`Bound`](gt_core::bound::Bound) is minted only by a graph,
/// so the only length a map here can grow to is that graph's index bound, and
/// a key past it is an `IndexError`. The auto-grow that matters -- a
/// never-written key *inside* the bound reading back as the default -- is
/// preserved exactly, by sizing before every read.
fn key_error<K: IdTag>(key: usize, bound: usize) -> PyErr {
    PyIndexError::new_err(format!(
        "{} index {key} is outside this graph's index bound of {bound}",
        K::NAME
    ))
}

// ---------------------------------------------------------------------------
// Python <-> value-universe bridging
// ---------------------------------------------------------------------------

/// How one member of the value universe crosses the boundary.
///
/// Twelve of the fourteen GIL-free members delegate to pyo3's own conversions.
/// [`LongDouble`] and `Vec<LongDouble>` cannot: the type is an *opaque
/// 16-byte payload* with no arithmetic ([DESIGN](gt_core::design) D7), so the only lossless
/// Python spelling is `bytes`. Exposing it as a `float` would round-trip an
/// 80-bit extended value through a `double` and write a different one back to
/// the `.gt` file, which is the whole failure the opaque carrier exists to
/// prevent.
trait PyBridge: Sized {
    /// Hand the value to Python.
    fn to_py(&self, py: Python<'_>) -> PyObject;
    /// Take a value from Python, or fail with a `PyErr`.
    fn from_py(ob: &Bound<'_, PyAny>) -> PyResult<Self>;
}

macro_rules! bridge_native {
    ($($t:ty),+ $(,)?) => {$(
        impl PyBridge for $t {
            #[inline]
            fn to_py(&self, py: Python<'_>) -> PyObject {
                self.clone().into_py(py)
            }
            #[inline]
            fn from_py(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
                ob.extract::<$t>()
            }
        }
    )+};
}

bridge_native!(
    u8,
    i16,
    i32,
    i64,
    f64,
    String,
    Vec<u8>,
    Vec<i16>,
    Vec<i32>,
    Vec<i64>,
    Vec<f64>,
    Vec<String>,
);

impl PyBridge for LongDouble {
    fn to_py(&self, py: Python<'_>) -> PyObject {
        PyBytes::new_bound(py, &self.0).unbind().into_any()
    }

    fn from_py(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let raw = ob.downcast::<PyBytes>()?.as_bytes();
        // `sizeof(long double) == 16` on every ABI graph-tool writes `.gt`
        // files on; a shorter or longer payload is not a `long double`.
        let exact: [u8; 16] = raw.try_into().map_err(|_| {
            PyValueError::new_err(format!(
                "long double is an opaque 16-byte payload, got {} bytes",
                raw.len()
            ))
        })?;
        Ok(LongDouble(exact))
    }
}

impl PyBridge for Vec<LongDouble> {
    fn to_py(&self, py: Python<'_>) -> PyObject {
        let items: Vec<PyObject> = self.iter().map(|v| v.to_py(py)).collect();
        PyList::new_bound(py, items).unbind().into_any()
    }

    fn from_py(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut out = Vec::new();
        for item in ob.iter()? {
            out.push(LongDouble::from_py(&item?)?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// The erased store for the fourteen GIL-free members
// ---------------------------------------------------------------------------

/// Generates the fourteen-armed store from one token list, in the same shape
/// -- and for the same reason -- as
/// [`value_subset!`](gt_core::value_subset): the member list, the
/// construction, the key accessors and the erasure all come from one place, so
/// a member cannot be added to one and forgotten in another.
macro_rules! gil_free_store {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        /// One property map of one key space, with its member chosen at
        /// runtime.
        enum Store<K: IdTag> {
            $( $variant(DenseProp<$ty, K>), )+
        }

        impl<K: IdTag> Store<K> {
            /// An empty map of `kind` belonging to `graph`, or `None` for the
            /// one member this store cannot hold.
            fn new(graph: GraphId, kind: ValueKind) -> Option<Self> {
                match kind {
                    $( ValueKind::$variant => Some(Store::$variant(DenseProp::new(graph))), )+
                    ValueKind::PyObject => None,
                }
            }

            /// Which member this map stores.
            fn kind(&self) -> ValueKind {
                match self { $( Store::$variant(_) => ValueKind::$variant, )+ }
            }

            /// Slots currently allocated. Not the graph's size: it is the
            /// number `sized_for` has grown the store to so far.
            fn allocated(&self) -> usize {
                match self { $( Store::$variant(m) => m.len(), )+ }
            }

            /// Read one key, **sizing first**.
            ///
            /// This is the auto-grow-on-read of
            /// `fast_vector_property_map.hh:132-136`, kept honest: a key that
            /// was never written reads back as `Zeroed::zero`, which is what
            /// `(*_store)[i]` returns after `soft_reserve(i + 1)` there.
            fn get(
                &mut self,
                py: Python<'_>,
                bound: IndexBound<K>,
                key: usize,
            ) -> PyResult<PyObject> {
                match self { $( Store::$variant(m) => {
                    let view = m.sized_for(bound).map_err(prop_err)?;
                    let slot = view
                        .as_slice()
                        .get(key)
                        .ok_or_else(|| key_error::<K>(key, bound.len()))?;
                    Ok(PyBridge::to_py(slot, py))
                } )+ }
            }

            /// Write one key.
            ///
            /// The conversion runs *before* the store is sized, so a rejected
            /// value leaves the map exactly as it was.
            fn set(
                &mut self,
                bound: IndexBound<K>,
                key: usize,
                value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                match self { $( Store::$variant(m) => {
                    let v = <$ty as PyBridge>::from_py(value)?;
                    let mut view = m.sized_for(bound).map_err(prop_err)?;
                    let slot = view
                        .as_mut_slice()
                        .get_mut(key)
                        .ok_or_else(|| key_error::<K>(key, bound.len()))?;
                    *slot = v;
                    Ok(())
                } )+ }
            }

            /// Grow the store to `bound`, filling new slots with the member's
            /// default.
            ///
            /// **This is step 2, and it is an entry-point obligation rather
            /// than a kernel one.** It has to be: a kernel is bounded by its
            /// axis' trait -- [`Scalar`] for this one -- and none of the axis
            /// traits implies [`Zeroed`](gt_core::prop::Zeroed), because
            /// `Zeroed` is exactly the property the fifteenth member does not
            /// have. Only here, where the concrete type is still in hand, is
            /// the default constructible at all. `gt_dispatch::pmap`
            /// (`dispatch.hh:171-177`) has the same information available and
            /// throws it away by calling `get_unchecked()` with no size.
            fn size(&mut self, bound: IndexBound<K>) -> Result<(), PropError> {
                match self { $( Store::$variant(m) => { m.sized_for(bound)?; } )+ }
                Ok(())
            }

            /// Erase to the dispatch boundary.
            ///
            /// One `TypeId` comparison downstream, where `dispatch.hh:248-268`
            /// needs up to three `any_cast` probes per candidate.
            fn erase(&mut self) -> AnyMap<'_, K> {
                match self { $( Store::$variant(m) => AnyMap::new(m), )+ }
            }
        }
    };
}

gil_free_store!(
    Bool => u8,
    I16 => i16,
    I32 => i32,
    I64 => i64,
    F64 => f64,
    LongDouble => LongDouble,
    Str => String,
    VecBool => Vec<u8>,
    VecI16 => Vec<i16>,
    VecI32 => Vec<i32>,
    VecI64 => Vec<i64>,
    VecF64 => Vec<f64>,
    VecLongDouble => Vec<LongDouble>,
    VecStr => Vec<String>,
);

/// Which index space a map is keyed on. graph-tool's `key_type()` answers
/// `"v"`, `"e"` or `"g"`; the graph-wide case is not a property *map* here,
/// so there are two.
enum KeyStore {
    /// Keyed on vertices.
    Vertex(Store<VertexTag>),
    /// Keyed on edges.
    Edge(Store<EdgeTag>),
}

/// The same choice for the Python member.
enum ObjStore {
    /// Keyed on vertices.
    Vertex(DenseProp<PyValue, VertexTag>),
    /// Keyed on edges.
    Edge(DenseProp<PyValue, EdgeTag>),
}

// ---------------------------------------------------------------------------
// The scalar kernel
// ---------------------------------------------------------------------------

/// Sum a scalar-valued map, in a GIL-free region.
///
/// Step 3 of the shape, and only step 3. The map reached this point already
/// sized by [`Store::size`], so `view()` -- which is fallible precisely
/// because `&self` cannot grow -- succeeds. Skipping the sizing would make
/// this kernel fail on a map the caller has never written, where graph-tool
/// returns the default (`fast_vector_property_map.hh:132-136`); doing it here
/// instead is not possible, because [`Scalar`] does not imply
/// [`Zeroed`](gt_core::prop::Zeroed).
///
/// The `Python<'py>` token lives in the kernel *struct*, never in the closure:
/// [`detach`]'s `F: Send` bound is what makes that distinction enforceable,
/// and the slice is `Send` only because `V: Scalar` implies
/// [`GilFree`](gt_core::prop::GilFree).
struct SumScalar<'py, K: IdTag> {
    py: Python<'py>,
    bound: IndexBound<K>,
}

impl<K: IdTag> ScalarKernel<K> for SumScalar<'_, K> {
    type Out = Result<f64, PropError>;

    fn call<V: Scalar>(self, map: &mut DenseProp<V, K>) -> Self::Out {
        let view = map.view(self.bound)?;
        let data: &[V] = view.as_slice();
        // The Python member cannot reach this line: it is not a `Scalar`, and
        // `&[PyValue]` is not `Send` in any case.
        Ok(detach(self.py, move || {
            data.iter().map(|v| v.to_f64()).sum::<f64>()
        }))
    }
}

// ---------------------------------------------------------------------------
// The graph
// ---------------------------------------------------------------------------

/// A graph, owned by Python.
#[pyclass(name = "Graph", module = "graph_tool._rust")]
pub struct PyGraph {
    inner: gt_core::adj::Graph,
}

#[pymethods]
impl PyGraph {
    /// An empty graph.
    #[new]
    fn new() -> Self {
        PyGraph {
            inner: gt_core::adj::Graph::new(),
        }
    }

    /// Number of vertices.
    fn num_vertices(&self) -> usize {
        self.inner.num_vertices()
    }

    /// Number of edges.
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Size of the edge index space, which exceeds
    /// [`num_edges`](Self::num_edges) once an edge has been removed.
    ///
    /// This is `g.edge_index_range` (`graph_tool/__init__.py:369`), and it is
    /// the number an edge property map must be sized to -- not the count.
    fn edge_index_range(&self) -> usize {
        self.inner.edge_bound().len()
    }

    /// Append an isolated vertex.
    fn add_vertex(&mut self) -> PyResult<usize> {
        self.inner
            .add_vertex()
            .map(|v| v.index())
            .map_err(graph_err)
    }

    /// Add an edge, returning its dense index.
    fn add_edge(&mut self, s: usize, t: usize) -> PyResult<usize> {
        let sv = VertexId::new(s).ok_or_else(|| PyValueError::new_err(invalid_vertex(s)))?;
        let tv = VertexId::new(t).ok_or_else(|| PyValueError::new_err(invalid_vertex(t)))?;
        self.inner
            .add_edge(sv, tv)
            .map(|e| e.id().index())
            .map_err(graph_err)
    }

    /// Remove an edge by index.
    ///
    /// Takes the index, never a descriptor: the endpoints come from the slot
    /// table, so there is no caller-supplied orientation that can disagree
    /// with storage. `remove_edge(e, g)` (`graph_adjacency.hh:1310-1312`)
    /// trusts the descriptor's own `s`/`t` and decrements `_n_edges` at the
    /// call site, which is the mechanism behind `clear_vertex`'s double
    /// decrement (`:1403-1413`).
    fn remove_edge(&mut self, e: usize) -> PyResult<()> {
        let id = EdgeId::new(e).ok_or_else(|| PyValueError::new_err(invalid_edge(e)))?;
        self.inner.remove_edge(id).map_err(graph_err)
    }

    /// A new vertex property map of the named type.
    ///
    /// `type` is `type_names[]` exactly (`graph_properties.hh:72-76`), and an
    /// unknown spelling raises `ValueError("Invalid property type: " + type)`
    /// as `new_property` does (`graph_python_interface.hh:689`). The alias
    /// table -- `"int"`, `"float"`, `"object"` -- belongs to `_type_alias`
    /// (`graph_tool/__init__.py:210-229`), which runs *above* this boundary
    /// and never reaches `type_names[]`.
    fn new_vertex_property(slf: &Bound<'_, PyGraph>, value_type: &str) -> PyResult<PyObject> {
        new_property(slf, value_type, false)
    }

    /// A new edge property map of the named type.
    fn new_edge_property(slf: &Bound<'_, PyGraph>, value_type: &str) -> PyResult<PyObject> {
        new_property(slf, value_type, true)
    }

    /// Sum a scalar-valued property map over this graph's index space.
    ///
    /// The worked example of the three-step shape. Note that the map is
    /// read-only to the kernel and is **sized anyway** ([DESIGN](gt_core::design) section 5),
    /// and that the bound comes from the map's own key type exactly as
    /// `_get_any` chooses between `num_vertices` and `edge_index_range`
    /// (`graph_tool/__init__.py:363-373`).
    fn sum_property(&self, py: Python<'_>, prop: &mut PyPropertyMap) -> PyResult<f64> {
        // The only fallible step of the dispatch, and it happens before any
        // map is touched.
        let member = ScalarV::narrow(prop.kind).map_err(dispatch_err)?;
        match &mut prop.store {
            KeyStore::Vertex(s) => {
                let bound = self.inner.vertex_bound();
                // Step 2, for a map the kernel only reads.
                s.size(bound).map_err(prop_err)?;
                let mut erased = s.erase();
                member
                    .dispatch(&mut erased, SumScalar { py, bound })
                    .map_err(prop_err)
            }
            KeyStore::Edge(s) => {
                let bound = self.inner.edge_bound();
                s.size(bound).map_err(prop_err)?;
                let mut erased = s.erase();
                member
                    .dispatch(&mut erased, SumScalar { py, bound })
                    .map_err(prop_err)
            }
        }
    }
}

/// `ValueError("Invalid vertex index: %d")`, the message
/// `graph_tool/__init__.py:2149` raises.
fn invalid_vertex(i: usize) -> String {
    format!("Invalid vertex index: {i}")
}

/// The edge-side counterpart.
fn invalid_edge(i: usize) -> String {
    format!("Invalid edge index: {i}")
}

/// Shared body of the two `new_*_property` methods.
///
/// The fifteenth member gets a different class, for the reason in the module
/// docs; `value_type()` and `key_type()` read the same on both, so Python code
/// that only asks those two questions does not have to know.
fn new_property(slf: &Bound<'_, PyGraph>, value_type: &str, edge: bool) -> PyResult<PyObject> {
    let py = slf.py();
    let kind = ValueKind::from_name(value_type)
        .ok_or_else(|| PyValueError::new_err(format!("Invalid property type: {value_type}")))?;
    let graph_id = slf.try_borrow()?.inner.graph_id();
    let handle: Py<PyGraph> = slf.clone().unbind();

    if kind == ValueKind::PyObject {
        let store = if edge {
            ObjStore::Edge(DenseProp::new(graph_id))
        } else {
            ObjStore::Vertex(DenseProp::new(graph_id))
        };
        return Ok(Py::new(
            py,
            PyObjectPropertyMap {
                graph: handle,
                store,
            },
        )?
        .into_any());
    }

    // `Store::new` is generic in `K`, and which one is fixed here and nowhere
    // else -- so a vertex map can never later be sized from an edge bound.
    // It returns `None` only for `PyObject`, which the branch above took;
    // spelled as an error rather than an `expect` so that a sixteenth member
    // turns a coverage gap into a `ValueError` and not a panic at the
    // boundary (`gt_core::design` defect table, row 34).
    let missing = || PyValueError::new_err(format!("Invalid property type: {value_type}"));
    let store = if edge {
        KeyStore::Edge(Store::new(graph_id, kind).ok_or_else(missing)?)
    } else {
        KeyStore::Vertex(Store::new(graph_id, kind).ok_or_else(missing)?)
    };
    Ok(Py::new(
        py,
        PyPropertyMap {
            kind,
            graph: handle,
            store,
        },
    )?
    .into_any())
}

// ---------------------------------------------------------------------------
// The property maps
// ---------------------------------------------------------------------------

/// A property map, owned by Python.
///
/// Carries any of the fourteen GIL-free members. The fifteenth lives in
/// [`PyObjectPropertyMap`]; see the module docs for why that is two classes
/// and not one.
#[pyclass(name = "PropertyMap", module = "graph_tool._rust")]
pub struct PyPropertyMap {
    kind: gt_core::prop::ValueKind,
    graph: Py<PyGraph>,
    store: KeyStore,
}

impl PyPropertyMap {
    /// The graph's two index bounds, read and released before anything else
    /// happens.
    ///
    /// A [`Bound`](gt_core::bound::Bound) is two `Copy` words carrying the
    /// graph's identity, so holding the `PyRef` across the rest of the call is
    /// unnecessary -- and holding it would turn any re-entrant mutation from
    /// Python into a borrow error rather than letting it proceed.
    fn bounds(&self, py: Python<'_>) -> PyResult<(IndexBound<VertexTag>, IndexBound<EdgeTag>)> {
        let g = self.graph.bind(py).try_borrow()?;
        Ok((g.inner.vertex_bound(), g.inner.edge_bound()))
    }
}

#[pymethods]
impl PyPropertyMap {
    /// The Python-visible type name, byte-for-byte graph-tool's
    /// `type_names[]`.
    fn value_type(&self) -> &'static str {
        self.kind.name()
    }

    /// `"v"` or `"e"`, as `PropertyMap.key_type()`.
    fn key_type(&self) -> &'static str {
        match self.store {
            KeyStore::Vertex(_) => "v",
            KeyStore::Edge(_) => "e",
        }
    }

    /// Slots the store has grown to so far. Observable because the growth is
    /// observable in graph-tool too, through `PropertyMap.get_array()`.
    fn allocated(&self) -> usize {
        match &self.store {
            KeyStore::Vertex(s) => s.allocated(),
            KeyStore::Edge(s) => s.allocated(),
        }
    }

    /// Read one key. A never-written key inside the graph's index bound
    /// returns the member's default.
    fn __getitem__(&mut self, py: Python<'_>, key: usize) -> PyResult<PyObject> {
        let (vb, eb) = self.bounds(py)?;
        match &mut self.store {
            KeyStore::Vertex(s) => s.get(py, vb, key),
            KeyStore::Edge(s) => s.get(py, eb, key),
        }
    }

    /// Write one key.
    fn __setitem__(
        &mut self,
        py: Python<'_>,
        key: usize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let (vb, eb) = self.bounds(py)?;
        match &mut self.store {
            KeyStore::Vertex(s) => s.set(vb, key, value),
            KeyStore::Edge(s) => s.set(eb, key, value),
        }
    }
}

/// A `python::object` property map, owned by Python.
///
/// `unsendable` is not a concession: [`PyValue`] is `!Send` by construction
/// ([DESIGN](gt_core::design) section 7, mechanism 3), so a store of them is `!Send` and
/// `#[pyclass]` will not accept it without the marker. That is the inverted
/// `is_python` predicate of `graph_properties_copy.cc:35-42` -- which releases
/// the GIL and runs `#pragma omp parallel` in exactly the object-to-object
/// case -- showing up as a type error instead of a data race.
#[pyclass(unsendable, name = "ObjectPropertyMap", module = "graph_tool._rust")]
pub struct PyObjectPropertyMap {
    graph: Py<PyGraph>,
    store: ObjStore,
}

impl PyObjectPropertyMap {
    /// As [`PyPropertyMap::bounds`].
    fn bounds(&self, py: Python<'_>) -> PyResult<(IndexBound<VertexTag>, IndexBound<EdgeTag>)> {
        let g = self.graph.bind(py).try_borrow()?;
        Ok((g.inner.vertex_bound(), g.inner.edge_bound()))
    }
}

/// Size a Python-valued map and read one key.
///
/// `sized_for_with`, not `sized_for`: `Py<PyAny>` has no `Default` and the
/// default `python::object` is `Py_None`, which needs the interpreter. That is
/// why [`Zeroed`](gt_core::prop::Zeroed) is not a supertrait of
/// [`PropValue`](gt_core::prop::PropValue) -- see [DESIGN](gt_core::design) section 13.1.
fn obj_get<K: IdTag>(
    py: Python<'_>,
    map: &mut DenseProp<PyValue, K>,
    bound: IndexBound<K>,
    key: usize,
) -> PyResult<PyObject> {
    let view = map
        .sized_for_with(bound, || PyValue::none(py))
        .map_err(prop_err)?;
    let slot = view
        .as_slice()
        .get(key)
        .ok_or_else(|| key_error::<K>(key, bound.len()))?;
    Ok(slot.bind(py).unbind())
}

/// Size a Python-valued map and write one key.
fn obj_set<K: IdTag>(
    py: Python<'_>,
    map: &mut DenseProp<PyValue, K>,
    bound: IndexBound<K>,
    key: usize,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut view = map
        .sized_for_with(bound, || PyValue::none(py))
        .map_err(prop_err)?;
    let slot = view
        .as_mut_slice()
        .get_mut(key)
        .ok_or_else(|| key_error::<K>(key, bound.len()))?;
    *slot = PyValue::new(value.clone().unbind());
    Ok(())
}

/// Apply a Python callable to every slot, serially, with the token held.
///
/// This is the `Seq` arm of the lattice arriving at the boundary, and it is
/// the *correct* behaviour that `graph_properties_copy.cc:35-42` intended
/// before the predicate was inverted. There is no `detach` here and there
/// cannot be one: the closure captures `f`, a `Bound<'py, PyAny>`, and
/// [`detach`](crate::gil::detach)'s `F: Send` bound rejects it.
///
/// An exception from `f` propagates as a `PyErr` through `?`. Nothing
/// unwinds, so `[profile.release] panic = "abort"` changes nothing about this
/// path -- which is the claim [DESIGN](gt_core::design) section 10 makes for the whole
/// boundary.
fn obj_map<K: IdTag>(
    py: Python<'_>,
    map: &mut DenseProp<PyValue, K>,
    bound: IndexBound<K>,
    f: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut view = map
        .sized_for_with(bound, || PyValue::none(py))
        .map_err(prop_err)?;
    for slot in view.as_mut_slice() {
        let out = f.call1((slot.bind(py),))?;
        *slot = PyValue::new(out.unbind());
    }
    Ok(())
}

#[pymethods]
impl PyObjectPropertyMap {
    /// Always `"python::object"`.
    fn value_type(&self) -> &'static str {
        ValueKind::PyObject.name()
    }

    /// `"v"` or `"e"`.
    fn key_type(&self) -> &'static str {
        match self.store {
            ObjStore::Vertex(_) => "v",
            ObjStore::Edge(_) => "e",
        }
    }

    /// Slots the store has grown to so far.
    fn allocated(&self) -> usize {
        match &self.store {
            ObjStore::Vertex(m) => m.len(),
            ObjStore::Edge(m) => m.len(),
        }
    }

    /// Read one key. A never-written key inside the bound reads back as
    /// `None`.
    fn __getitem__(&mut self, py: Python<'_>, key: usize) -> PyResult<PyObject> {
        let (vb, eb) = self.bounds(py)?;
        match &mut self.store {
            ObjStore::Vertex(m) => obj_get(py, m, vb, key),
            ObjStore::Edge(m) => obj_get(py, m, eb, key),
        }
    }

    /// Write one key.
    fn __setitem__(
        &mut self,
        py: Python<'_>,
        key: usize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let (vb, eb) = self.bounds(py)?;
        match &mut self.store {
            ObjStore::Vertex(m) => obj_set(py, m, vb, key, value),
            ObjStore::Edge(m) => obj_set(py, m, eb, key, value),
        }
    }

    /// Replace every slot with `f(slot)`, serially.
    fn map_values(&mut self, py: Python<'_>, f: &Bound<'_, PyAny>) -> PyResult<()> {
        let (vb, eb) = self.bounds(py)?;
        match &mut self.store {
            ObjStore::Vertex(m) => obj_map(py, m, vb, f),
            ObjStore::Edge(m) => obj_map(py, m, eb, f),
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// `type_names[]` (`graph_properties.hh:72-76`), in declaration order.
///
/// One list, derived from the enum, where the C++ keeps two -- `value_types`
/// and `type_names[]` -- in correspondence by position and by nothing else.
#[pyfunction]
fn value_types() -> Vec<&'static str> {
    ValueKind::ALL.iter().map(|k| k.name()).collect()
}

/// Register the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyPropertyMap>()?;
    m.add_class::<PyObjectPropertyMap>()?;
    m.add_function(wrap_pyfunction!(value_types, m)?)?;
    Ok(())
}

/// The CPython entry point, `graph_tool._rust`.
///
/// Compiled only under the `extension-module` feature, so
/// `cargo build --workspace` never needs to link libpython.
///
/// **Note for whoever owns `crates/gt-py/Cargo.toml`:** the comment there
/// says the feature "adds `cdylib` through pyo3's own feature". It does not.
/// A Cargo feature cannot add a crate type; `pyo3/extension-module` only
/// stops the build script emitting `-l python3.x`. With
/// `crate-type = ["rlib"]` the `PyInit__rust` symbol below is compiled and
/// then discarded, and no importable `.so` is produced. The fix is one word,
/// `crate-type = ["rlib", "cdylib"]`, which was verified out of tree: with it,
/// a debug *and* a `panic = "abort"` release build both import and pass the
/// Python-level smoke test.
#[cfg(feature = "extension-module")]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(m)
}
