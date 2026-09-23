//! I/O errors.

use gt_core::prop::ValueKind;

/// Anything that can go wrong reading or writing a graph.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// Underlying stream failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The file is not in the expected format.
    #[error("bad magic: expected {expected:?}, found {found:?}")]
    BadMagic {
        /// What the format requires.
        expected: &'static [u8],
        /// What was there.
        found: Vec<u8>,
    },
    /// The file declares a format version this build cannot read.
    #[error("unsupported format version {0}")]
    UnsupportedVersion(u8),
    /// A property map declares a value type this build does not know.
    #[error("unknown value type name {0:?}")]
    UnknownValueType(String),
    /// A property map's declared type and payload disagree.
    #[error("property {name:?} declares {declared:?} but its payload is not readable as that")]
    MalformedProperty {
        /// The property's name.
        name: String,
        /// The declared member.
        declared: ValueKind,
    },
    /// A property column handed to a writer does not cover the graph.
    ///
    /// The C++ cannot raise this: `write_property_dispatch` walks the graph's
    /// own range and calls `prop[x]` (`graph_io_binary.hh:319-320`) on an
    /// `unchecked_vector_property_map`, whose `operator[]`
    /// (`fast_vector_property_map.hh:218-221`) has no bounds check and whose
    /// backing vector may be shorter than the range -- which is the class of
    /// defect `prop::DenseProp::sized_for` exists to close.
    #[error("property {name:?} holds {have} values, the graph needs {need}")]
    ShortProperty {
        /// The property's name.
        name: String,
        /// How many values the column holds.
        have: usize,
        /// How many the graph's index space requires.
        need: usize,
    },
    /// A descriptor in the file is out of range.
    #[error("{kind} index {index} exceeds the declared bound {bound}")]
    IndexOutOfRange {
        /// "vertex" or "edge".
        kind: &'static str,
        /// The offending index.
        index: u64,
        /// The declared bound.
        bound: u64,
    },
    /// The graph is too large for the configured index width.
    #[error("graph exceeds the index width; rebuild with the `wide-index` feature")]
    IndexWidthExceeded,
    /// Syntax error in a text format.
    #[error("parse error at line {line}: {msg}")]
    Parse {
        /// One-based line number.
        line: usize,
        /// What was wrong.
        msg: String,
    },
}
