//! The native binary `.gt` format.
//!
//! Ports `src/graph/graph_io_binary.hh` in full. The layout, byte for byte:
//!
//! ```text
//! "\u{26fe} gt"          6 bytes, `_magic` (`graph_io_binary.hh:30-31`)
//! version               u8, `_version == 1` (`:32`)
//! big_endian            u8, `is_bigendian()` of the *writing* host (`:447`)
//! comment               string (`:449-456`)
//! directed              u8  \
//! N                     u64  |  `write_adjacency` (`:186-202`)
//! N x vector<Vint>           /  `Vint` is the narrowest of u8/u16/u32/u64 that holds N-1
//! nprops                u64 (`:459-460`)
//! nprops x property          `write_property` (`:367-382`)
//! ```
//!
//! where a *string* is a `u64` length followed by that many bytes (`:71-76`),
//! a *vector* is a `u64` count followed by that many fixed-width elements
//! (`:62-68`), and a *property* is a domain byte (`:248-253`), a name string,
//! a value-type index byte, and then one value per element of the domain's
//! range -- one for a graph property, `N` for a vertex property, `num_edges`
//! for an edge property.
//!
//! ## Endianness
//!
//! Every multi-byte field after the `big_endian` flag is in the byte order
//! that flag declares; [`write`] emits the host's own order and sets the flag
//! to match, exactly as `write_graph` does (`:447`). The *comment*, however,
//! is read by the C++ with `read<false>` (`:534`) -- that is, always as
//! little-endian -- while it is written with the host's order (`:456`), so a
//! big-endian build of graph-tool writes a header it cannot itself read. This
//! port reads the comment's length in the order the flag declares, which is
//! identical on every little-endian host and correct on the others.
//!
//! ## `python::object`
//!
//! The C++ serialises the Python member as `lexical_cast<std::string>(v)`
//! (`:86-90`), and that cast is `object_pickler` (`graph_io.cc:62-68`), which
//! is `pickle.dump(obj, sstream, -1)` (`graph_tool/gt_io.py:63-67`). So a
//! `python::object` column *in the file* is a column of opaque pickle
//! payloads, and nothing about reading or writing one needs an interpreter.
//! [`PropertyColumn::PyObject`] carries those payloads as bytes, for the same
//! reason [`LongDouble`] carries sixteen: the bytes round-trip unchanged
//! whether or not this build has the `python` feature, and unpickling is the
//! boundary layer's business, not the format's.
//!
//! ## Ordering
//!
//! Edge property values are positional: the reader binds the `i`-th value to
//! the `i`-th edge of `edges_range(g)` (`:398-399`), so the adjacency section
//! and every edge property column must agree on one traversal order. This port
//! uses *out-edges of each vertex in ascending vertex order, and within a
//! vertex in ascending [`EdgeId`](gt_core::ids::EdgeId) order*. On a graph
//! that has never had an edge removed that is byte-for-byte graph-tool's own
//! order; after removals it is the port's, because `AdjList` removes by
//! swapping with the back of the half where `graph_adjacency.hh:1257-1263`
//! erases and shifts (DESIGN.md defect #52). Sorting makes the output a
//! function of the graph rather than of its removal history.

use std::io::{BufReader, BufWriter, Read, Write};

use gt_core::adj::{AdjList, Incident, Lookup};
use gt_core::error::GraphError;
use gt_core::ids::{MAX_INDEX, VertexId};
use gt_core::prop::{LongDouble, ValueKind};

use crate::error::IoError;

/// `_magic` (`graph_io_binary.hh:30`): U+26FE SESQUIQUADRATE, a space, `gt`.
pub const MAGIC: &[u8] = "⛾ gt".as_bytes();

// `_magic_length` (`:31`) is a *separate* constant from `_magic` in the C++,
// and `strncmp(magic, _magic, _magic_length)` (`:524`) trusts it. Here the
// length is derived from the bytes and the agreement is proved at compile
// time, so the two cannot drift apart.
const _: () = assert!(MAGIC.len() == 6);

/// `_version` (`graph_io_binary.hh:32`).
pub const VERSION: u8 = 1;

/// Number of elements a reader will reserve before it has seen the bytes.
///
/// `read(std::istream&, std::vector<T>&)` (`:107-117`) does `v.resize(size)`
/// on a length it has not validated, so a corrupt or truncated file asks for
/// that allocation before the stream can refuse it. Growth here is driven by
/// the bytes that actually arrive; this is only the head start.
const RESERVE_CAP: usize = 1024;

/// A property map read from, or to be written to, a file.
///
/// Named, typed and *valued*: the reader reconstructs a whole column, so the
/// caller does not have to declare the concrete
/// [`DenseProp`](gt_core::prop::DenseProp) in advance and does not have to
/// re-read the stream to fill it.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedProperty {
    /// The property's name, as the Python layer sees it.
    pub name: String,
    /// Which identifier space it is keyed on.
    pub domain: PropertyDomain,
    /// The values, one per element of `domain`'s range.
    ///
    /// A vertex column is indexed by [`VertexId`](gt_core::ids::VertexId), an
    /// edge column by [`EdgeId`](gt_core::ids::EdgeId) -- *not* by position in
    /// the file's traversal order. A graph column holds exactly one value.
    pub values: PropertyColumn,
}

impl NamedProperty {
    /// Which member of the value universe this column stores.
    ///
    /// A method rather than a field: `value_types` and `type_names[]`
    /// (`graph_properties.hh:61-76`) are two declarations held in
    /// correspondence by position and by nothing else, and a `kind` field
    /// beside a `values` field is the same shape of mistake one level down --
    /// it can disagree with the payload it describes, and the writer would
    /// then emit a type byte that the values do not match.
    #[inline]
    pub fn kind(&self) -> ValueKind {
        self.values.kind()
    }
}

/// Which identifier space a property map is keyed on.
///
/// The `property_type` enum (`graph_io_binary.hh:248-253`); the discriminants
/// are the bytes in the file.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PropertyDomain {
    /// One value per graph.
    Graph = 0,
    /// One value per vertex.
    Vertex = 1,
    /// One value per edge.
    Edge = 2,
}

impl PropertyDomain {
    /// Parse a domain byte, or `None` for the `default:` arm at `:504-506`.
    #[inline]
    pub const fn from_byte(b: u8) -> Option<PropertyDomain> {
        match b {
            0 => Some(PropertyDomain::Graph),
            1 => Some(PropertyDomain::Vertex),
            2 => Some(PropertyDomain::Edge),
            _ => None,
        }
    }
}

/// One property map's values, erased over the fifteen-member value universe.
///
/// One variant per [`ValueKind`], carrying the column rather than a single
/// value: the `.gt` layout is columnar, and so is
/// [`DenseProp`](gt_core::prop::DenseProp), so neither direction needs a
/// per-element `dyn` hop.
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyColumn {
    /// `uint8_t`, named "bool" on the Python side.
    Bool(Vec<u8>),
    /// `int16_t`.
    I16(Vec<i16>),
    /// `int32_t`.
    I32(Vec<i32>),
    /// `int64_t`.
    I64(Vec<i64>),
    /// `double`.
    F64(Vec<f64>),
    /// `long double`, opaque.
    LongDouble(Vec<LongDouble>),
    /// `std::string`.
    Str(Vec<String>),
    /// `std::vector<uint8_t>`.
    VecBool(Vec<Vec<u8>>),
    /// `std::vector<int16_t>`.
    VecI16(Vec<Vec<i16>>),
    /// `std::vector<int32_t>`.
    VecI32(Vec<Vec<i32>>),
    /// `std::vector<int64_t>`.
    VecI64(Vec<Vec<i64>>),
    /// `std::vector<double>`.
    VecF64(Vec<Vec<f64>>),
    /// `std::vector<long double>`, opaque.
    VecLongDouble(Vec<Vec<LongDouble>>),
    /// `std::vector<std::string>`.
    VecStr(Vec<Vec<String>>),
    /// `boost::python::object`, as the pickle payload the format actually
    /// stores. See the module docs.
    PyObject(Vec<Vec<u8>>),
}

macro_rules! column_arms {
    ($self:expr, $col:ident => $e:expr) => {
        match $self {
            PropertyColumn::Bool($col) => $e,
            PropertyColumn::I16($col) => $e,
            PropertyColumn::I32($col) => $e,
            PropertyColumn::I64($col) => $e,
            PropertyColumn::F64($col) => $e,
            PropertyColumn::LongDouble($col) => $e,
            PropertyColumn::Str($col) => $e,
            PropertyColumn::VecBool($col) => $e,
            PropertyColumn::VecI16($col) => $e,
            PropertyColumn::VecI32($col) => $e,
            PropertyColumn::VecI64($col) => $e,
            PropertyColumn::VecF64($col) => $e,
            PropertyColumn::VecLongDouble($col) => $e,
            PropertyColumn::VecStr($col) => $e,
            PropertyColumn::PyObject($col) => $e,
        }
    };
}

impl PropertyColumn {
    /// Which member of the value universe this column stores.
    #[inline]
    pub const fn kind(&self) -> ValueKind {
        match self {
            PropertyColumn::Bool(_) => ValueKind::Bool,
            PropertyColumn::I16(_) => ValueKind::I16,
            PropertyColumn::I32(_) => ValueKind::I32,
            PropertyColumn::I64(_) => ValueKind::I64,
            PropertyColumn::F64(_) => ValueKind::F64,
            PropertyColumn::LongDouble(_) => ValueKind::LongDouble,
            PropertyColumn::Str(_) => ValueKind::Str,
            PropertyColumn::VecBool(_) => ValueKind::VecBool,
            PropertyColumn::VecI16(_) => ValueKind::VecI16,
            PropertyColumn::VecI32(_) => ValueKind::VecI32,
            PropertyColumn::VecI64(_) => ValueKind::VecI64,
            PropertyColumn::VecF64(_) => ValueKind::VecF64,
            PropertyColumn::VecLongDouble(_) => ValueKind::VecLongDouble,
            PropertyColumn::VecStr(_) => ValueKind::VecStr,
            PropertyColumn::PyObject(_) => ValueKind::PyObject,
        }
    }

    /// An empty column of the given member.
    pub fn empty(kind: ValueKind) -> PropertyColumn {
        match kind {
            ValueKind::Bool => PropertyColumn::Bool(Vec::new()),
            ValueKind::I16 => PropertyColumn::I16(Vec::new()),
            ValueKind::I32 => PropertyColumn::I32(Vec::new()),
            ValueKind::I64 => PropertyColumn::I64(Vec::new()),
            ValueKind::F64 => PropertyColumn::F64(Vec::new()),
            ValueKind::LongDouble => PropertyColumn::LongDouble(Vec::new()),
            ValueKind::Str => PropertyColumn::Str(Vec::new()),
            ValueKind::VecBool => PropertyColumn::VecBool(Vec::new()),
            ValueKind::VecI16 => PropertyColumn::VecI16(Vec::new()),
            ValueKind::VecI32 => PropertyColumn::VecI32(Vec::new()),
            ValueKind::VecI64 => PropertyColumn::VecI64(Vec::new()),
            ValueKind::VecF64 => PropertyColumn::VecF64(Vec::new()),
            ValueKind::VecLongDouble => PropertyColumn::VecLongDouble(Vec::new()),
            ValueKind::VecStr => PropertyColumn::VecStr(Vec::new()),
            ValueKind::PyObject => PropertyColumn::PyObject(Vec::new()),
        }
    }

    /// Number of values.
    #[inline]
    pub fn len(&self) -> usize {
        column_arms!(self, v => v.len())
    }

    /// Whether the column holds nothing.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A Rust type that can be a `.gt` property column.
///
/// The mapping is [`PropValue`](gt_core::prop::PropValue)'s, member for
/// member, with one exception it cannot express: `python::object` and
/// `vector<bool>` are both `Vec<Vec<u8>>` here, because this crate stores the
/// Python member as its pickle payload rather than as a live handle. `Vec<u8>`
/// therefore means `vector<bool>`, and a `python::object` column is built
/// with [`PropertyColumn::PyObject`] directly.
pub trait ColumnValue: Sized {
    /// The member this type occupies.
    const KIND: ValueKind;
    /// Wrap a column.
    fn into_column(values: Vec<Self>) -> PropertyColumn;
    /// Borrow a column of this type, or `None` if it stores another member.
    fn column_slice(column: &PropertyColumn) -> Option<&[Self]>;
}

macro_rules! impl_column_value {
    ($($t:ty => $v:ident),+ $(,)?) => {$(
        impl ColumnValue for $t {
            const KIND: ValueKind = ValueKind::$v;
            #[inline]
            fn into_column(values: Vec<Self>) -> PropertyColumn {
                PropertyColumn::$v(values)
            }
            #[inline]
            fn column_slice(column: &PropertyColumn) -> Option<&[Self]> {
                match column {
                    PropertyColumn::$v(v) => Some(v),
                    _ => None,
                }
            }
        }
    )+};
}
impl_column_value!(
    u8 => Bool,
    i16 => I16,
    i32 => I32,
    i64 => I64,
    f64 => F64,
    LongDouble => LongDouble,
    String => Str,
    Vec<u8> => VecBool,
    Vec<i16> => VecI16,
    Vec<i32> => VecI32,
    Vec<i64> => VecI64,
    Vec<f64> => VecF64,
    Vec<LongDouble> => VecLongDouble,
    Vec<String> => VecStr,
);

/// Everything a `.gt` file holds.
#[derive(Clone, Debug)]
pub struct Document<H: Lookup> {
    /// The graph.
    pub graph: AdjList<H>,
    /// Whether the file declares the graph directed.
    ///
    /// Storage is always the directed adjacency: `write_to_file` sets
    /// `_directed = true` for the duration of the write (`graph_io.cc:507`)
    /// so that an undirected graph still emits each edge once, from its
    /// stored source, and records the flag separately.
    pub directed: bool,
    /// The property maps, erased.
    pub properties: Vec<NamedProperty>,
    /// The header comment, verbatim.
    ///
    /// `None` on write means "generate the one this version of graph-tool
    /// would" (`graph_io_binary.hh:449-456`). [`read`] fills it with what the
    /// file actually said, which is the only way a file written by an older
    /// graph-tool can be re-saved byte-identically: the comment embeds the
    /// writing version's own text, and the shipped collection (for instance
    /// `karate.gt.gz`) still carries 2.2.32dev's wording.
    pub comment: Option<String>,
}

impl<H: Lookup> Document<H> {
    /// A document with no properties and no comment.
    pub fn new(graph: AdjList<H>, directed: bool) -> Self {
        Document {
            graph,
            directed,
            properties: Vec::new(),
            comment: None,
        }
    }

    /// `write_graph`'s comment (`graph_io_binary.hh:449-456`), regenerated.
    fn generated_comment(&self) -> String {
        let (ng, nv, ne) = self.domain_counts();
        format!(
            "graph-tool binary file version {VERSION} (https://graph-tool.skewed.de), stats: {} vertices, {} edges, {}, {ng} graph props, {nv} vertex props, {ne} edge props",
            self.graph.num_vertices(),
            self.graph.num_edges(),
            if self.directed {
                "directed"
            } else {
                "undirected"
            },
        )
    }

    fn domain_counts(&self) -> (usize, usize, usize) {
        let mut c = (0usize, 0usize, 0usize);
        for p in &self.properties {
            match p.domain {
                PropertyDomain::Graph => c.0 += 1,
                PropertyDomain::Vertex => c.1 += 1,
                PropertyDomain::Edge => c.2 += 1,
            }
        }
        c
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// The one shape of syntax error a binary format has.
///
/// `Parse::line` is a text-format concept, so it is zero here and the byte
/// offset goes in the message. graph-tool signals the same condition by
/// arming the stream with `failbit | eofbit` (`graph_io.cc:346`) and catching
/// `ios_base::failure` two frames up, which loses the offset entirely.
fn corrupt(at: u64, msg: impl std::fmt::Display) -> IoError {
    IoError::Parse {
        line: 0,
        msg: format!("at byte {at}: {msg}"),
    }
}

fn from_graph_error(e: GraphError) -> IoError {
    match e {
        GraphError::VertexIdSpaceExhausted { .. } | GraphError::EdgeIdSpaceExhausted { .. } => {
            IoError::IndexWidthExceeded
        }
        other => IoError::Parse {
            line: 0,
            msg: other.to_string(),
        },
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// A counted, endian-aware byte source.
struct Source<R: Read> {
    r: R,
    /// Byte offset, for diagnostics. graph-tool reports none.
    pos: u64,
    /// What the file's `big_endian` flag said.
    big_endian: bool,
}

/// `byte_swap<BE>` (`graph_io_binary.hh:45-52`): a no-op when the file's
/// declared order is the host's, a full reversal of the field otherwise.
macro_rules! read_scalar {
    ($name:ident, $t:ty) => {
        #[inline]
        fn $name(&mut self) -> Result<$t, IoError> {
            let mut b = [0u8; size_of::<$t>()];
            self.exact(&mut b)?;
            Ok(if self.big_endian {
                <$t>::from_be_bytes(b)
            } else {
                <$t>::from_le_bytes(b)
            })
        }
    };
}

impl<R: Read> Source<R> {
    fn new(r: R) -> Self {
        Source {
            r,
            pos: 0,
            big_endian: false,
        }
    }

    /// Fill `buf` or fail with [`IoError::Parse`].
    ///
    /// `read_exact`'s `UnexpectedEof` is remapped rather than forwarded: a
    /// truncated file is a malformed file, not an I/O failure, and the
    /// acceptance criterion for this unit is that truncation is diagnosed and
    /// never panics.
    fn exact(&mut self, buf: &mut [u8]) -> Result<(), IoError> {
        match self.r.read_exact(buf) {
            Ok(()) => {
                self.pos += buf.len() as u64;
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Err(corrupt(
                self.pos,
                format_args!("unexpected end of file, wanted {} more bytes", buf.len()),
            )),
            Err(e) => Err(IoError::Io(e)),
        }
    }

    #[inline]
    fn u8(&mut self) -> Result<u8, IoError> {
        let mut b = [0u8; 1];
        self.exact(&mut b)?;
        Ok(b[0])
    }

    read_scalar!(i16, i16);
    read_scalar!(i32, i32);
    read_scalar!(i64, i64);
    read_scalar!(u16, u16);
    read_scalar!(u32, u32);
    read_scalar!(u64, u64);
    read_scalar!(f64, f64);

    /// `long double` is carried opaquely, so the "swap" is the whole field
    /// reversed -- which is what `byte_swap<BE>` does to it (`:50-51`).
    fn long_double(&mut self) -> Result<LongDouble, IoError> {
        let mut b = [0u8; 16];
        self.exact(&mut b)?;
        if self.big_endian != cfg!(target_endian = "big") {
            b.reverse();
        }
        Ok(LongDouble(b))
    }

    /// A length-prefixed byte run (`:127-134`).
    ///
    /// The length is *not* trusted into an allocation: growth follows the
    /// bytes that arrive, so a corrupt length yields a parse error rather than
    /// the `v.resize(size)` the C++ performs first (`:131`).
    fn bytes(&mut self) -> Result<Vec<u8>, IoError> {
        let n = self.u64()?;
        let at = self.pos;
        let mut out = Vec::new();
        let got = self
            .r
            .by_ref()
            .take(n)
            .read_to_end(&mut out)
            .map_err(IoError::Io)?;
        self.pos += got as u64;
        if got as u64 != n {
            return Err(corrupt(
                at,
                format_args!("string declares {n} bytes, {got} are present"),
            ));
        }
        Ok(out)
    }

    fn string(&mut self, what: &str) -> Result<String, IoError> {
        let at = self.pos;
        let b = self.bytes()?;
        String::from_utf8(b).map_err(|e| {
            corrupt(
                at,
                format_args!(
                    "{what} is not valid UTF-8 at offset {}",
                    e.utf8_error().valid_up_to()
                ),
            )
        })
    }

    /// The element count of a vector field, with the reservation capped.
    fn count(&mut self) -> Result<(usize, usize), IoError> {
        let n = self.u64()?;
        let n = usize::try_from(n).map_err(|_| {
            corrupt(
                self.pos,
                format_args!("vector of {n} exceeds the address space"),
            )
        })?;
        Ok((n, n.min(RESERVE_CAP)))
    }
}

/// Read a `.gt` stream.
pub fn read<R: Read, H: Lookup>(r: R, lookup: H) -> Result<Document<H>, IoError> {
    let mut src = Source::new(BufReader::new(r));

    // `strncmp(magic, _magic, _magic_length) != 0` (`:524`). A short read is
    // reported as a bad magic rather than as truncation: it is the first
    // thing in the file, so "not a .gt file" is the better diagnosis.
    let mut magic = [0u8; 6];
    let mut got = 0;
    while got < magic.len() {
        match src.r.read(&mut magic[got..]).map_err(IoError::Io)? {
            0 => break,
            n => got += n,
        }
    }
    src.pos += got as u64;
    if got != magic.len() || magic != *MAGIC {
        return Err(IoError::BadMagic {
            expected: MAGIC,
            found: magic[..got].to_vec(),
        });
    }

    let version = src.u8()?;
    if version != VERSION {
        return Err(IoError::UnsupportedVersion(version));
    }
    // `read<false>` on a one-byte field (`:531-532`): the swap is a no-op, so
    // the flag is readable before the order it declares is known.
    src.big_endian = src.u8()? != 0;
    let comment = src.string("the header comment")?;

    let (graph, directed, num_edges) = read_adjacency(&mut src, lookup)?;

    let nprops = src.u64()?;
    let mut properties = Vec::new();
    for _ in 0..nprops {
        let at = src.pos;
        let byte = src.u8()?;
        let domain = PropertyDomain::from_byte(byte)
            .ok_or_else(|| corrupt(at, format_args!("invalid property domain {byte}")))?;
        let name = src.string("a property name")?;
        let at = src.pos;
        let index = src.u8()?;
        // `val_types` is `value_types` with `size_t` appended (`:257-258`), so
        // index 15 is a dispatch-table artefact that `write_property_dispatch`
        // never emits -- its `size_t` overload writes `int64_t`'s index
        // instead (`:335`). Rejecting it here costs nothing a real file uses.
        let kind = ValueKind::ALL
            .get(index as usize)
            .copied()
            .ok_or_else(|| IoError::UnknownValueType(format!("value type index {index}")))?;
        let n = match domain {
            PropertyDomain::Graph => 1,
            PropertyDomain::Vertex => graph.num_vertices(),
            PropertyDomain::Edge => num_edges,
        };
        let values = read_column(&mut src, kind, n, &name)?;
        properties.push(NamedProperty {
            name,
            domain,
            values,
        });
    }

    Ok(Document {
        graph,
        directed,
        properties,
        comment: Some(comment),
    })
}

/// `read_adjacency` (`:223-244`), returning the edge count as well.
fn read_adjacency<R: Read, H: Lookup>(
    src: &mut Source<R>,
    lookup: H,
) -> Result<(AdjList<H>, bool, usize), IoError> {
    let directed = src.u8()? != 0;
    let n = src.u64()?;
    let n = usize::try_from(n).map_err(|_| IoError::IndexWidthExceeded)?;
    if n > 0 && n - 1 > MAX_INDEX {
        return Err(IoError::IndexWidthExceeded);
    }

    let mut g = AdjList::with_lookup(lookup);
    let width = vint_width(n as u64);
    let mut edges = 0usize;

    // `add_vertex(g, N)` (`:232`) allocates all N blocks before a single
    // adjacency byte is read, so a file whose N is corrupt asks for the
    // allocation first and discovers the truncation second. Growing to the
    // highest index actually mentioned, and only then to N, makes the two
    // happen in the other order; for a well-formed file the result is
    // identical, because every list is read and N blocks exist at the end.
    for v in 0..n {
        let (k, _) = src.count()?;
        grow_to(&mut g, v + 1)?;
        let v = VertexId::from_index(v);
        for _ in 0..k {
            let u = read_vint(src, width)?;
            // `:215-217`, verbatim, including the bound being N and not the
            // number of vertices created so far.
            if u >= n as u64 {
                return Err(IoError::IndexOutOfRange {
                    kind: "vertex",
                    index: u,
                    bound: n as u64,
                });
            }
            let u = VertexId::from_index(u as usize);
            grow_to(&mut g, u.index() + 1)?;
            let e = g.add_edge(v, u).map_err(from_graph_error)?;
            // The reader binds the i-th edge property value to the i-th edge
            // of this traversal, so the traversal index and the edge id must
            // agree. On a graph built only by `add_edge` they do, because the
            // free list is empty and `alloc` hands out 0, 1, 2, ...
            debug_assert_eq!(e.id().index(), edges);
            edges += 1;
        }
    }
    grow_to(&mut g, n)?;

    Ok((g, directed, edges))
}

fn grow_to<H: Lookup>(g: &mut AdjList<H>, n: usize) -> Result<(), IoError> {
    while g.num_vertices() < n {
        g.add_vertex().map_err(from_graph_error)?;
    }
    Ok(())
}

/// The `Vint` dispatch of `write_adjacency`/`read_adjacency` (`:194-201`,
/// `:234-241`), as a width in bytes.
const fn vint_width(n: u64) -> u8 {
    if n <= u8::MAX as u64 {
        1
    } else if n <= u16::MAX as u64 {
        2
    } else if n <= u32::MAX as u64 {
        4
    } else {
        8
    }
}

fn read_vint<R: Read>(src: &mut Source<R>, width: u8) -> Result<u64, IoError> {
    Ok(match width {
        1 => u64::from(src.u8()?),
        2 => u64::from(src.u16()?),
        4 => u64::from(src.u32()?),
        _ => src.u64()?,
    })
}

/// A property payload that declares `string` but is not UTF-8.
///
/// `std::string` is a byte string, so the C++ accepts anything here; the value
/// universe spells the member `String` (`prop/value.rs`), so this port has to
/// say so rather than silently substitute. Truncation is *not* routed through
/// here -- that is a `Parse`, raised by `Source::bytes` before this is called.
fn utf8(b: Vec<u8>, name: &str, declared: ValueKind) -> Result<String, IoError> {
    String::from_utf8(b).map_err(|_| IoError::MalformedProperty {
        name: name.to_owned(),
        declared,
    })
}

/// Read `n` values of one member.
fn read_column<R: Read>(
    src: &mut Source<R>,
    kind: ValueKind,
    n: usize,
    name: &str,
) -> Result<PropertyColumn, IoError> {
    /// `for (auto x : range) read<BE>(s, prop[x])` (`:398-399`), with the
    /// reservation capped at [`RESERVE_CAP`].
    macro_rules! column {
        ($variant:ident, $read:expr) => {{
            let mut out = Vec::with_capacity(n.min(RESERVE_CAP));
            for _ in 0..n {
                out.push($read?);
            }
            PropertyColumn::$variant(out)
        }};
    }

    /// A vector-valued member: a count and then that many elements (`:107-117`).
    macro_rules! vector_column {
        ($variant:ident, $elem:ident) => {{
            let mut out = Vec::with_capacity(n.min(RESERVE_CAP));
            for _ in 0..n {
                let (k, cap) = src.count()?;
                let mut inner = Vec::with_capacity(cap);
                for _ in 0..k {
                    inner.push(src.$elem()?);
                }
                out.push(inner);
            }
            PropertyColumn::$variant(out)
        }};
    }

    Ok(match kind {
        ValueKind::Bool => column!(Bool, src.u8()),
        ValueKind::I16 => column!(I16, src.i16()),
        ValueKind::I32 => column!(I32, src.i32()),
        ValueKind::I64 => column!(I64, src.i64()),
        ValueKind::F64 => column!(F64, src.f64()),
        ValueKind::LongDouble => column!(LongDouble, src.long_double()),
        ValueKind::Str => {
            let mut out = Vec::with_capacity(n.min(RESERVE_CAP));
            for _ in 0..n {
                out.push(utf8(src.bytes()?, name, ValueKind::Str)?);
            }
            PropertyColumn::Str(out)
        }
        ValueKind::VecBool => {
            // `vector<uint8_t>` is a byte run, so it takes the bulk path
            // rather than one `read_exact` per element.
            let mut out = Vec::with_capacity(n.min(RESERVE_CAP));
            for _ in 0..n {
                out.push(src.bytes()?);
            }
            PropertyColumn::VecBool(out)
        }
        ValueKind::VecI16 => vector_column!(VecI16, i16),
        ValueKind::VecI32 => vector_column!(VecI32, i32),
        ValueKind::VecI64 => vector_column!(VecI64, i64),
        ValueKind::VecF64 => vector_column!(VecF64, f64),
        ValueKind::VecLongDouble => vector_column!(VecLongDouble, long_double),
        ValueKind::VecStr => {
            let mut out = Vec::with_capacity(n.min(RESERVE_CAP));
            for _ in 0..n {
                let (k, cap) = src.count()?;
                let mut inner = Vec::with_capacity(cap);
                for _ in 0..k {
                    inner.push(utf8(src.bytes()?, name, ValueKind::VecStr)?);
                }
                out.push(inner);
            }
            PropertyColumn::VecStr(out)
        }
        // A pickle payload, kept opaque: `read(s, boost::python::object&)`
        // (`:156-162`) reads a `std::string` and then unpickles it, and the
        // unpickling is the boundary layer's business.
        ValueKind::PyObject => {
            let mut out = Vec::with_capacity(n.min(RESERVE_CAP));
            for _ in 0..n {
                out.push(src.bytes()?);
            }
            PropertyColumn::PyObject(out)
        }
    })
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// A byte sink that writes the host's own order, as `write` does (`:55-59`).
struct Sink<W: Write> {
    w: W,
}

macro_rules! write_scalar {
    ($name:ident, $t:ty) => {
        #[inline]
        fn $name(&mut self, v: $t) -> Result<(), IoError> {
            self.raw(&v.to_ne_bytes())
        }
    };
}

impl<W: Write> Sink<W> {
    #[inline]
    fn raw(&mut self, b: &[u8]) -> Result<(), IoError> {
        self.w.write_all(b).map_err(IoError::Io)
    }

    #[inline]
    fn u8(&mut self, v: u8) -> Result<(), IoError> {
        self.raw(&[v])
    }

    write_scalar!(i16, i16);
    write_scalar!(i32, i32);
    write_scalar!(i64, i64);
    write_scalar!(u16, u16);
    write_scalar!(u32, u32);
    write_scalar!(u64, u64);
    write_scalar!(f64, f64);

    #[inline]
    fn long_double(&mut self, v: LongDouble) -> Result<(), IoError> {
        self.raw(&v.0)
    }

    /// A length-prefixed byte run (`:71-76`).
    fn bytes(&mut self, b: &[u8]) -> Result<(), IoError> {
        self.u64(b.len() as u64)?;
        self.raw(b)
    }
}

/// Write a `.gt` stream.
///
/// Edges are emitted in [`EdgeId`](gt_core::ids::EdgeId) order within each
/// vertex's out-half, not in adjacency order, so the output does not depend on
/// the removal history. See the module docs.
pub fn write<W: Write, H: Lookup>(w: W, doc: &Document<H>) -> Result<(), IoError> {
    // One buffer for many small fields; the C++ writes into a
    // `filtering_stream`, which is buffered for the same reason.
    let mut sink = Sink {
        w: BufWriter::with_capacity(1 << 16, w),
    };

    sink.raw(MAGIC)?;
    sink.u8(VERSION)?;
    sink.u8(u8::from(cfg!(target_endian = "big")))?;
    match &doc.comment {
        Some(c) => sink.bytes(c.as_bytes())?,
        None => sink.bytes(doc.generated_comment().as_bytes())?,
    }

    let order = write_adjacency(&mut sink, doc)?;

    let (ng, nv, ne) = doc.domain_counts();
    sink.u64((ng + nv + ne) as u64)?;
    // Grouped by domain, in graph/vertex/edge order, preserving the relative
    // order within each group: `write_graph` walks three separate vectors
    // (`:461-466`), and a file graph-tool wrote is therefore already grouped.
    for domain in [
        PropertyDomain::Graph,
        PropertyDomain::Vertex,
        PropertyDomain::Edge,
    ] {
        for p in doc.properties.iter().filter(|p| p.domain == domain) {
            write_property(&mut sink, doc, p, &order)?;
        }
    }

    // `BufWriter` swallows a write error in its own `Drop`; the C++ arms the
    // stream with `failbit` (`graph_io.cc:485`) so that the same condition
    // throws. Flushing explicitly is what turns it back into a `Result`.
    sink.w.flush().map_err(IoError::Io)
}

/// `write_adjacency` (`:186-202`), returning the canonical edge order so that
/// every edge property column can be emitted against it.
fn write_adjacency<W: Write, H: Lookup>(
    sink: &mut Sink<W>,
    doc: &Document<H>,
) -> Result<Vec<usize>, IoError> {
    let g = &doc.graph;
    let n = g.num_vertices();
    sink.u8(u8::from(doc.directed))?;
    sink.u64(n as u64)?;

    let width = vint_width(n as u64);
    let mut order = Vec::with_capacity(g.num_edges());
    // Reused across vertices: a per-vertex `Vec` would be one allocation per
    // vertex on a path that runs once per file.
    let mut scratch: Vec<Incident> = Vec::new();

    for v in g.vertices() {
        scratch.clear();
        scratch.extend(g.out_edges(v));
        // Already sorted on a graph that has never had an edge removed, which
        // is the case that has to stay byte-identical to graph-tool's output.
        scratch.sort_unstable_by_key(|i| i.edge);
        sink.u64(scratch.len() as u64)?;
        for i in &scratch {
            let u = i.other.index() as u64;
            match width {
                1 => sink.u8(u as u8)?,
                2 => sink.u16(u as u16)?,
                4 => sink.u32(u as u32)?,
                _ => sink.u64(u)?,
            }
            order.push(i.edge.index());
        }
    }
    Ok(order)
}

/// `write_property` (`:367-382`).
fn write_property<W: Write, H: Lookup>(
    sink: &mut Sink<W>,
    doc: &Document<H>,
    p: &NamedProperty,
    order: &[usize],
) -> Result<(), IoError> {
    // Checked before a byte of this property reaches the sink: a refused
    // write should not leave half a property header in the stream.
    //
    // A vertex column is keyed by vertex index and a graph column holds one
    // value; an edge column is keyed by `EdgeId`, and the file's positional
    // order is therefore the traversal `order`, not `0..num_edges`.
    let need = match p.domain {
        PropertyDomain::Graph => 1,
        PropertyDomain::Vertex => doc.graph.num_vertices(),
        // The *bound*, not the count: after a removal the live ids are sparse,
        // which is exactly why `_get_any` sizes from `edge_index_range`
        // (`graph_tool/__init__.py:369`).
        PropertyDomain::Edge => doc.graph.edge_bound().len(),
    };
    if p.values.len() < need {
        return Err(IoError::ShortProperty {
            name: p.name.clone(),
            have: p.values.len(),
            need,
        });
    }

    sink.u8(p.domain as u8)?;
    sink.bytes(p.name.as_bytes())?;
    sink.u8(p.kind() as u8)?;

    match p.domain {
        PropertyDomain::Graph => write_values(sink, &p.values, std::iter::once(0)),
        PropertyDomain::Vertex => write_values(sink, &p.values, 0..doc.graph.num_vertices()),
        PropertyDomain::Edge => write_values(sink, &p.values, order.iter().copied()),
    }
}

/// Emit `column[i]` for each `i`, in the order given.
///
/// Monomorphised over the index iterator rather than taking `&[usize]`, so the
/// vertex case walks a range with no indirection and no `order` vector to
/// build.
fn write_values<W, I>(
    sink: &mut Sink<W>,
    column: &PropertyColumn,
    indices: I,
) -> Result<(), IoError>
where
    W: Write,
    I: Iterator<Item = usize>,
{
    /// One scalar per element.
    macro_rules! scalars {
        ($v:expr, $put:ident) => {{
            for i in indices {
                sink.$put($v[i])?;
            }
        }};
    }
    /// A count and then that many elements (`:62-68`).
    macro_rules! vectors {
        ($v:expr, $put:ident) => {{
            for i in indices {
                let inner = &$v[i];
                sink.u64(inner.len() as u64)?;
                for x in inner {
                    sink.$put(*x)?;
                }
            }
        }};
    }

    match column {
        PropertyColumn::Bool(v) => scalars!(v, u8),
        PropertyColumn::I16(v) => scalars!(v, i16),
        PropertyColumn::I32(v) => scalars!(v, i32),
        PropertyColumn::I64(v) => scalars!(v, i64),
        PropertyColumn::F64(v) => scalars!(v, f64),
        PropertyColumn::LongDouble(v) => scalars!(v, long_double),
        PropertyColumn::Str(v) => {
            for i in indices {
                sink.bytes(v[i].as_bytes())?;
            }
        }
        // `vector<uint8_t>` and a pickle payload are both byte runs.
        PropertyColumn::VecBool(v) | PropertyColumn::PyObject(v) => {
            for i in indices {
                sink.bytes(&v[i])?;
            }
        }
        PropertyColumn::VecI16(v) => vectors!(v, i16),
        PropertyColumn::VecI32(v) => vectors!(v, i32),
        PropertyColumn::VecI64(v) => vectors!(v, i64),
        PropertyColumn::VecF64(v) => vectors!(v, f64),
        PropertyColumn::VecLongDouble(v) => vectors!(v, long_double),
        PropertyColumn::VecStr(v) => {
            for i in indices {
                let inner = &v[i];
                sink.u64(inner.len() as u64)?;
                for s in inner {
                    sink.bytes(s.as_bytes())?;
                }
            }
        }
    }
    Ok(())
}
