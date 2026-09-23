//! Edge lists as delimited text.
//!
//! This is the port of `load_graph_from_csv`
//! (`graph_tool/__init__.py:3768-3893`) together with the two C++ kernels it
//! dispatches into: `add_edge_list` (`graph_python_interface_imp1.cc:36-94`)
//! for an index edge list, and `add_edge_list_hash::numpy_dispatch`
//! (`:217-272`) for a symbolic one.
//!
//! ## What the format is, exactly
//!
//! One record per line, at least two fields per record, the first two being
//! the source and the target. graph-tool hands the file to Python's
//! [`csv.reader`] with `delimiter=","`, `quotechar='"'` and
//! `strip_whitespace=True`, so the accepted grammar is Python's default
//! dialect, not "split on the delimiter": a field that opens with `"` runs to
//! the matching `"`, a doubled `""` inside it is one literal quote, and such a
//! field may contain the delimiter and even a newline. Both readers here
//! implement that dialect. Fields past the second are parsed and discarded --
//! the edge properties `load_graph_from_csv` builds out of them are a
//! property-map concern, and [`EdgeListOptions`] has no place to put them --
//! but they are still *parsed*, because a quoted third field is allowed to
//! contain the record terminator and a reader that stopped at the second
//! field would mis-frame the next record.
//!
//! [`csv.reader`]: https://docs.python.org/3/library/csv.html
//!
//! ## Determinism
//!
//! Two things fix the graph this module builds, and neither is the machine it
//! runs on:
//!
//! * **Vertex ids** come from first-seen order over the records, source
//!   before target within a record. That is `get_vertex`
//!   (`graph_python_interface_imp1.cc:241-253`) called as `s` then `t`
//!   (`:256-260`), and it is sequential in graph-tool too -- the whole hashed
//!   path runs under one `GILRelease` with one `vertices` map.
//! * **Edge ids** come from [`ParBuilder`], whose chunk count this module
//!   derives from the *record count* alone ([`ROWS_PER_CHUNK`]) and never from
//!   `available_parallelism()` or the size of the rayon pool. So the `i`-th
//!   record is always edge `i`, at one worker or at sixteen.
//!
//! The second point is what `set_concurrent` costs graph-tool: with per-thread
//! free lists (`graph_adjacency.hh:460-466`, `:627-655`) the index an edge
//! receives depends on which worker took it, and every edge property map
//! written by index afterwards inherits that.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{BufWriter, Read, Write};

use gt_core::adj::{AdjList, Lookup, ParBuilder};
use gt_core::error::GraphError;
use gt_core::graph::{EdgeList, GraphRef, VertexList};
use gt_core::ids::{MAX_INDEX, VertexId};

use crate::error::IoError;

/// Records per staging chunk.
///
/// A constant, so the chunk count is a pure function of the input. See the
/// module documentation, and [`ParBuilder::new`]'s warning that a chunk count
/// derived from the thread pool is a chunk count that makes the edge ids
/// depend on the deployment.
const ROWS_PER_CHUNK: usize = 1 << 16;

/// How to interpret a delimited edge list.
#[derive(Clone, Copy, Debug)]
pub struct EdgeListOptions {
    /// Field separator.
    pub delimiter: u8,
    /// Whether the first row names the columns.
    pub header: bool,
    /// Whether endpoints are names to be interned rather than indices.
    pub symbolic: bool,
}

impl Default for EdgeListOptions {
    fn default() -> Self {
        EdgeListOptions {
            delimiter: b',',
            header: false,
            symbolic: true,
        }
    }
}

// ===========================================================================
// Reading
// ===========================================================================

/// Build a graph from a delimited edge list.
///
/// Uses [`ParBuilder`](gt_core::adj::ParBuilder) when the input can be chunked,
/// so the edge ids do not depend on the number of worker threads.
///
/// The second half of the result is the vertex-name map: `names[v.index()]` is
/// the string `v` was interned from. It is `load_graph_from_csv`'s `g.vp.name`
/// (`graph_tool/__init__.py:3891-3892`), and like it, it is empty when
/// `opts.symbolic` is false -- there the fields *are* the indices and there is
/// nothing to record.
///
/// # Semantics
///
/// * Endpoints are stripped of leading and trailing ASCII whitespace, which is
///   `load_graph_from_csv`'s `strip_whitespace=True` default (`:3832-3836`).
///   Stripping happens *after* unquoting, exactly as it does there, so
///   `" a "` and `a` name the same vertex.
/// * With `opts.symbolic`, vertices are numbered in first-seen order, source
///   before target. The empty string is a name like any other.
/// * Without it, each field must parse as a `usize` and the graph gets
///   `max + 1` vertices -- the fixed point of
///   `while (s >= num_vertices(g) || t >= num_vertices(g)) add_vertex(g);`
///   (`graph_python_interface_imp1.cc:75-76`).
/// * Blank records are skipped. `load_graph_from_csv` does not skip them: a
///   blank line reaches `add_edge_list` as `[]` and the unconditional
///   `row[0]` at `graph_tool/__init__.py:3856` raises `IndexError`. A trailing
///   newline is not a syntax error in any other reader of this format, and it
///   is not one here.
///
/// # Errors
///
/// * [`IoError::Io`] from the stream.
/// * [`IoError::Parse`], with the one-based line the record *started* on, for
///   a record with fewer than two fields, an unterminated quoted field, a
///   field that is not UTF-8, or -- when `opts.symbolic` is false -- a field
///   that is not a non-negative integer.
/// * [`IoError::IndexWidthExceeded`] if the vertex or edge count does not fit
///   [`Raw`](gt_core::ids::Raw).
pub fn read_edge_list<R: Read, H: Lookup>(
    mut r: R,
    opts: EdgeListOptions,
    lookup: H,
) -> Result<(AdjList<H>, Vec<String>), IoError> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    // A UTF-8 BOM is a byte-order mark, not part of the first vertex's name.
    let body: &[u8] = buf.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(&buf);

    let mut cur = Cursor::new();
    if opts.header {
        cur.skip_record(body, opts.delimiter)?;
    }

    // Phase 1, sequential: resolve every record to a pair of ids. This is the
    // phase that fixes vertex identity, and it is the phase that reports every
    // syntax error -- so the parallel phase below cannot fail, and the error a
    // malformed file produces does not depend on which worker saw it first.
    let mut pairs: Vec<(VertexId, VertexId)> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut index: HashMap<String, VertexId> = HashMap::new();
    let mut n_vertices = 0usize;

    while let Some(rec) = cur.next_record(body, opts.delimiter)? {
        let (s, t) = if opts.symbolic {
            // Source first, then target: `graph_python_interface_imp1.cc:256`
            // and `:260`. A record `a,a` interns one vertex; a record `b,a`
            // after `a,b` interns none.
            let s = intern(&mut index, &mut names, rec.source)?;
            let t = intern(&mut index, &mut names, rec.target)?;
            (s, t)
        } else {
            let s = parse_index(&rec.source, rec.line)?;
            let t = parse_index(&rec.target, rec.line)?;
            n_vertices = n_vertices.max(s.index() + 1).max(t.index() + 1);
            (s, t)
        };
        pairs.push((s, t));
    }
    if opts.symbolic {
        n_vertices = names.len();
    }

    // Phase 2, parallel: scatter into the staging buffers. `fill` hands chunk
    // `k` nothing but `k`, which is the whole determinism contract, and here
    // the generator is a slice copy -- a pure function of `k` by construction.
    let chunks = pairs.len().div_ceil(ROWS_PER_CHUNK).max(1);
    let mut builder = ParBuilder::new(n_vertices, chunks);
    let staged = &pairs[..];
    builder.fill(|k, out| {
        let lo = (k * ROWS_PER_CHUNK).min(staged.len());
        let hi = (lo + ROWS_PER_CHUNK).min(staged.len());
        out.extend_from_slice(&staged[lo..hi]);
    });
    // Dropped before `build`, so the staged pairs never coexist with the
    // adjacency they become.
    drop(pairs);

    let g = builder
        .build(lookup)
        .map_err(|e| build_error(e, n_vertices))?;
    Ok((g, names))
}

/// Translate the builder's failure into this crate's vocabulary.
///
/// `NoSuchVertex` is unreachable: phase 1 either interns the endpoint (so it
/// is below `names.len()`) or widens `n_vertices` past it. It is mapped rather
/// than asserted because a reader that aborts the process on a malformed file
/// is not a reader.
fn build_error(e: GraphError, n_vertices: usize) -> IoError {
    match e {
        GraphError::VertexIdSpaceExhausted { .. } | GraphError::EdgeIdSpaceExhausted { .. } => {
            IoError::IndexWidthExceeded
        }
        GraphError::NoSuchVertex(v) => IoError::IndexOutOfRange {
            kind: "vertex",
            index: v.index() as u64,
            bound: n_vertices as u64,
        },
        other => IoError::Parse {
            line: 0,
            msg: other.to_string(),
        },
    }
}

/// First-seen interning.
///
/// The `to_owned` on the miss path is the one allocation per *vertex*; a hit
/// allocates nothing, which matters because a real edge list mentions each
/// name `degree(v)` times.
fn intern(
    index: &mut HashMap<String, VertexId>,
    names: &mut Vec<String>,
    name: Cow<'_, str>,
) -> Result<VertexId, IoError> {
    if let Some(&v) = index.get(name.as_ref()) {
        return Ok(v);
    }
    let v = VertexId::new(names.len()).ok_or(IoError::IndexWidthExceeded)?;
    let owned = name.into_owned();
    names.push(owned.clone());
    index.insert(owned, v);
    Ok(v)
}

/// An index edge list's endpoint.
fn parse_index(field: &str, line: usize) -> Result<VertexId, IoError> {
    let n: usize = field.parse().map_err(|_| IoError::Parse {
        line,
        msg: format!("expected a vertex index, found {field:?}"),
    })?;
    if n > MAX_INDEX {
        return Err(IoError::IndexWidthExceeded);
    }
    Ok(VertexId::from_index(n))
}

// ===========================================================================
// The dialect
// ===========================================================================

/// Where a field stopped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stop {
    /// At a delimiter, which was consumed. Another field follows.
    Field,
    /// At a record terminator, which was *not* consumed.
    Record,
    /// At end of input.
    Eof,
}

/// One record's first two fields, and the line it began on.
struct Record<'a> {
    line: usize,
    source: Cow<'a, str>,
    target: Cow<'a, str>,
}

/// A position in the buffer, plus the physical line that position is on.
///
/// The line counter is advanced by newlines *inside* quoted fields too, so the
/// number in a [`IoError::Parse`] is the line an editor would put the cursor
/// on.
struct Cursor {
    pos: usize,
    line: usize,
}

impl Cursor {
    fn new() -> Self {
        Cursor { pos: 0, line: 1 }
    }

    /// Consume a `\n`, `\r` or `\r\n` if one is next.
    fn eat_terminator(&mut self, buf: &[u8]) {
        match buf.get(self.pos) {
            Some(&b'\r') => {
                self.pos += 1;
                if buf.get(self.pos) == Some(&b'\n') {
                    self.pos += 1;
                }
                self.line += 1;
            }
            Some(&b'\n') => {
                self.pos += 1;
                self.line += 1;
            }
            _ => {}
        }
    }

    /// Read one field, leaving the cursor after the delimiter that ended it
    /// (or on the record terminator).
    ///
    /// The returned value is borrowed from `buf` whenever the field carried no
    /// quotes, which is the common case and the reason this is not a
    /// `String`-per-field parser.
    fn field<'a>(&mut self, buf: &'a [u8], delim: u8) -> Result<(Cow<'a, str>, Stop), IoError> {
        let start_line = self.line;
        let start = self.pos;
        // `Some` exactly when the field needed rewriting: it opened with a
        // quote, so the quotes themselves and any doubled `""` have to go.
        let mut owned: Option<Vec<u8>> = None;
        let mut in_quotes = false;

        let (stop, end) = loop {
            let Some(&c) = buf.get(self.pos) else {
                break (Stop::Eof, self.pos);
            };
            if in_quotes {
                if c == b'"' {
                    if buf.get(self.pos + 1) == Some(&b'"') {
                        // `""` -> one literal quote (Python's `doublequote`).
                        owned.as_mut().expect("quoted").push(b'"');
                        self.pos += 2;
                    } else {
                        in_quotes = false;
                        self.pos += 1;
                    }
                } else {
                    if c == b'\n' {
                        self.line += 1;
                    }
                    owned.as_mut().expect("quoted").push(c);
                    self.pos += 1;
                }
            } else if c == delim {
                let end = self.pos;
                self.pos += 1;
                break (Stop::Field, end);
            } else if c == b'\n' || c == b'\r' {
                break (Stop::Record, self.pos);
            } else if c == b'"' && self.pos == start {
                // A quote opens a field only at its start; anywhere else it is
                // an ordinary character, which is what `QUOTE_MINIMAL` does on
                // the way in.
                owned = Some(Vec::new());
                in_quotes = true;
                self.pos += 1;
            } else {
                // Text after a closing quote is appended, not rejected: that
                // is the reader's behaviour, and rejecting it would turn a
                // file graph-tool loads into a file this crate refuses.
                if let Some(o) = owned.as_mut() {
                    o.push(c);
                }
                self.pos += 1;
            }
        };

        if in_quotes {
            return Err(IoError::Parse {
                line: start_line,
                msg: "unterminated quoted field".to_owned(),
            });
        }

        let value = match owned {
            Some(v) => {
                let s = String::from_utf8(v).map_err(|_| utf8_error(start_line))?;
                match s.trim() {
                    t if t.len() == s.len() => Cow::Owned(s),
                    t => Cow::Owned(t.to_owned()),
                }
            }
            None => {
                let s =
                    std::str::from_utf8(&buf[start..end]).map_err(|_| utf8_error(start_line))?;
                Cow::Borrowed(s.trim())
            }
        };
        Ok((value, stop))
    }

    /// Skip whatever is left of the current record, terminator included.
    fn drain_record(&mut self, buf: &[u8], delim: u8) -> Result<(), IoError> {
        loop {
            let (_, stop) = self.field(buf, delim)?;
            match stop {
                Stop::Field => continue,
                Stop::Record => {
                    self.eat_terminator(buf);
                    return Ok(());
                }
                Stop::Eof => return Ok(()),
            }
        }
    }

    /// Skip one whole record, blank lines first. This is `skip_first`
    /// (`graph_tool/__init__.py:3838-3839`), which discards the line without
    /// looking at how many columns it has.
    fn skip_record(&mut self, buf: &[u8], delim: u8) -> Result<(), IoError> {
        while matches!(buf.get(self.pos), Some(&b'\n') | Some(&b'\r')) {
            self.eat_terminator(buf);
        }
        if self.pos >= buf.len() {
            return Ok(());
        }
        self.drain_record(buf, delim)
    }

    /// The next non-blank record, or `None` at end of input.
    fn next_record<'a>(&mut self, buf: &'a [u8], delim: u8) -> Result<Option<Record<'a>>, IoError> {
        loop {
            if self.pos >= buf.len() {
                return Ok(None);
            }
            if matches!(buf[self.pos], b'\n' | b'\r') {
                self.eat_terminator(buf);
                continue;
            }
            let line = self.line;
            let (source, stop) = self.field(buf, delim)?;
            if stop != Stop::Field {
                if stop == Stop::Record {
                    self.eat_terminator(buf);
                }
                // A line of nothing but whitespace is blank, not malformed.
                if source.is_empty() {
                    continue;
                }
                return Err(too_few(line));
            }
            let (target, stop) = self.field(buf, delim)?;
            match stop {
                Stop::Field => self.drain_record(buf, delim)?,
                Stop::Record => self.eat_terminator(buf),
                Stop::Eof => {}
            }
            return Ok(Some(Record {
                line,
                source,
                target,
            }));
        }
    }
}

fn utf8_error(line: usize) -> IoError {
    IoError::Parse {
        line,
        msg: "field is not valid UTF-8".to_owned(),
    }
}

/// graph-tool says this as `"Second dimension in edge list must be of size (at
/// least) two"` (`graph_python_interface_imp1.cc:52`) on the numpy path, and
/// as an `IndexError` with no context on the iterator path.
fn too_few(line: usize) -> IoError {
    IoError::Parse {
        line,
        msg: "expected at least two columns".to_owned(),
    }
}

// ===========================================================================
// Writing
// ===========================================================================

/// Write a view as a delimited edge list, in
/// [`EdgeId`](gt_core::ids::EdgeId) order.
///
/// Option (b) of the crate's ordering contract: `AdjList` removes adjacency
/// entries by swapping with the back of the half, so `EdgeList::edges` yields
/// an order that depends on the removal history, while `EdgeId` does not. One
/// sort of `num_edges()` ids buys a byte-reproducible file.
///
/// Endpoints are written as vertex *indices*, `opts.symbolic` notwithstanding:
/// there is no name map in this signature to write instead, and inventing one
/// from the indices would produce a file that reads back as a different graph.
/// `opts.header` prepends `source<delim>target`, so a file written with it set
/// reads back with it set.
///
/// Isolated vertices are not representable in this format and are not written;
/// a round trip through it preserves the edge set and the vertex *count* only
/// as far as the largest endpoint. That is the format's limitation, and it is
/// `load_graph_from_csv`'s too.
pub fn write_edge_list<W: Write, G>(w: W, g: G, opts: EdgeListOptions) -> Result<(), IoError>
where
    G: GraphRef + VertexList + EdgeList,
{
    let mut out = BufWriter::new(w);
    if opts.header {
        out.write_all(b"source")?;
        out.write_all(&[opts.delimiter])?;
        out.write_all(b"target\n")?;
    }

    let mut edges: Vec<_> = Vec::with_capacity(g.num_edges());
    edges.extend(g.edges().map(|e| (e.id().index(), e.source(), e.target())));
    // Ids are unique, so this is a total order and the unstable sort is
    // deterministic.
    edges.sort_unstable_by_key(|&(id, _, _)| id);

    // One reusable scratch buffer: the steady-state inner loop allocates
    // nothing.
    let mut line = Vec::with_capacity(32);
    for (_, s, t) in edges {
        line.clear();
        write_index(&mut line, s.index());
        line.push(opts.delimiter);
        write_index(&mut line, t.index());
        line.push(b'\n');
        out.write_all(&line)?;
    }
    out.flush()?;
    Ok(())
}

/// Decimal, appended. `write!` would go through `fmt::Arguments` and a
/// formatting machine this does not need.
fn write_index(out: &mut Vec<u8>, mut n: usize) {
    let mut digits = [0u8; 20];
    let mut i = digits.len();
    loop {
        i -= 1;
        digits[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    out.extend_from_slice(&digits[i..]);
}

// ===========================================================================
// Unit tests for the dialect itself. The acceptance criteria live in
// `tests/u29_csv.rs`.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn fields(src: &str, delim: u8) -> Vec<(usize, String, String)> {
        let mut cur = Cursor::new();
        let mut out = Vec::new();
        while let Some(r) = cur
            .next_record(src.as_bytes(), delim)
            .expect("well-formed input")
        {
            out.push((r.line, r.source.into_owned(), r.target.into_owned()));
        }
        out
    }

    #[test]
    fn a_quoted_field_may_hold_the_delimiter_and_a_newline() {
        let got = fields("\"a,1\",\"b\nc\",x\nd,e\n", b',');
        assert_eq!(
            got,
            vec![
                (1, "a,1".to_owned(), "b\nc".to_owned()),
                (3, "d".to_owned(), "e".to_owned()),
            ]
        );
    }

    #[test]
    fn doubled_quotes_collapse() {
        let got = fields("\"a\"\"b\",c\n", b',');
        assert_eq!(got, vec![(1, "a\"b".to_owned(), "c".to_owned())]);
    }

    #[test]
    fn a_quote_inside_an_unquoted_field_is_literal() {
        let got = fields("a\"b,c\n", b',');
        assert_eq!(got, vec![(1, "a\"b".to_owned(), "c".to_owned())]);
    }

    #[test]
    fn whitespace_is_stripped_after_unquoting() {
        let got = fields("  a  ,\t b \t\n\" c \", d \n", b',');
        assert_eq!(
            got,
            vec![
                (1, "a".to_owned(), "b".to_owned()),
                (2, "c".to_owned(), "d".to_owned()),
            ]
        );
    }

    #[test]
    fn crlf_and_blank_lines() {
        let got = fields("a,b\r\n\r\n\r\nc,d\r\n", b',');
        assert_eq!(
            got,
            vec![
                (1, "a".to_owned(), "b".to_owned()),
                (4, "c".to_owned(), "d".to_owned()),
            ]
        );
    }

    #[test]
    fn a_record_with_one_field_names_its_own_line() {
        let mut cur = Cursor::new();
        let src = b"a,b\nlonely\nc,d\n";
        cur.next_record(src, b',').expect("first").expect("some");
        match cur.next_record(src, b',') {
            Err(IoError::Parse { line, .. }) => assert_eq!(line, 2),
            other => panic!(
                "expected a parse error, got {other:?}",
                other = other.is_ok()
            ),
        }
    }

    #[test]
    fn an_unterminated_quote_is_an_error_not_a_panic() {
        let mut cur = Cursor::new();
        assert!(matches!(
            cur.next_record(b"\"a,b\n", b','),
            Err(IoError::Parse { .. })
        ));
    }

    #[test]
    fn write_index_matches_display() {
        for n in [0usize, 1, 9, 10, 99, 100, 12345, usize::MAX] {
            let mut v = Vec::new();
            write_index(&mut v, n);
            assert_eq!(String::from_utf8(v).expect("ascii"), n.to_string());
        }
    }
}
