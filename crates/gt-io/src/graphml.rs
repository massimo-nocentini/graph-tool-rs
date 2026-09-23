//! GraphML.
//!
//! Ports the writer of `src/boost-workaround/boost/graph/graphml.hpp`
//! (`write_graphml`, `:353-528`) and the reader of `src/graph/graphml.cpp`
//! (`graphml_reader`, `:44-479`), together with the `mutate_graph_impl`
//! glue that sits between the reader and the property maps
//! (`graphml.hpp:93-286`).
//!
//! ## The type-name table
//!
//! GraphML's `attr.type` is *not* [`ValueKind::name`]. It is
//! `prop_type_names[]` (`graphml.cpp:27-31`) indexed by position in
//! `prop_value_types` (`graphml.hpp:38-43`), and neither the names nor the
//! order match `type_names[]`/`value_types` in `graph_properties.hh`:
//!
//! | this port's [`ValueKind`] | `attr.type` | note |
//! |---|---|---|
//! | `F64` (`double`) | `float` | |
//! | `LongDouble` (`long double`) | `double` | *not* IEEE double |
//! | `VecF64` | `vector_float` | |
//! | `VecLongDouble` | `vector_double` | |
//! | `PyObject` | `python_object` | base64 of the pickle payload |
//!
//! So a GraphML file written by another tool that declares `attr.type="double"`
//! -- which in the GraphML specification means IEEE double -- is read by
//! graph-tool as `long double`. That is a compatibility hazard, not a defect
//! in the code, and this port reproduces it: changing the table would make
//! graph-tool unable to read files this crate writes.
//!
//! ## `long double` in a text format
//!
//! [`LongDouble`] is an opaque 16-byte payload with no arithmetic
//! (DESIGN.md row 49), and a text format needs a *representation*. This module
//! writes it as a C99 hexadecimal float -- `0x8p-3` for `1.0` -- because that
//! is exact, and because graph-tool's own reader takes long doubles through
//! `sscanf("%La")` (`str_repr.hh:81-84`), which accepts precisely that form.
//! Decimal input is accepted too, by parsing it as `f64` and widening, which
//! is exact for every value a `double` can hold and is what a file *written*
//! by graph-tool contains (`print_float`, `:62-69`). The interpretation is the
//! x86-64 80-bit extended layout; see [`long_double_to_text`].
//!
//! ## Ordering
//!
//! `dynamic_properties` is a `std::multimap<std::string, ...>`, so
//! `write_graphml`'s three walks over `dp` (`graphml.hpp:383`, `:430`, `:458`)
//! visit the properties **sorted by name**, whatever order they were added in.
//! [`write`] sorts a stable copy of the index range the same way, so the key
//! ids and the `<data>` order are a function of the document alone. Edges are
//! emitted in [`EdgeId`] order, per the crate-level ordering contract.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::io::{BufWriter, Read, Write};

use gt_core::adj::{AdjList, Lookup};
use gt_core::error::GraphError;
use gt_core::ids::{EdgeId, MAX_INDEX, VertexId};
use gt_core::prop::{LongDouble, ValueKind};

use crate::error::IoError;
use crate::gt::{Document, NamedProperty, PropertyColumn, PropertyDomain};

/// The vertex property `read_graphml` fills with the file's own node ids when
/// they are not canonical (`graphml.cpp:340-342`).
pub const VERTEX_ID_KEY: &str = "_graphml_vertex_id";

/// The edge property `read_graphml` fills with the file's own edge ids when
/// they are not canonical (`graphml.cpp:388-389`).
pub const EDGE_ID_KEY: &str = "_graphml_edge_id";

// ===========================================================================
// The value universe as text
//
// This section is shared with `dot.rs`: both formats print a property value
// with the same `lexical_cast<std::string>` (`graphml.hpp:326-339` and
// `graphviz.hpp:536`), so they must agree digit for digit. It lives here
// rather than in a module of its own because `lib.rs` belongs to another unit
// and a new module would have to be declared there.
// ===========================================================================

/// `prop_type_names[]` (`graphml.cpp:27-31`), by member.
pub(crate) const fn graphml_type_name(k: ValueKind) -> &'static str {
    match k {
        ValueKind::Bool => "boolean",
        ValueKind::I16 => "short",
        ValueKind::I32 => "int",
        ValueKind::I64 => "long",
        ValueKind::F64 => "float",
        ValueKind::LongDouble => "double",
        ValueKind::VecBool => "vector_boolean",
        ValueKind::VecI16 => "vector_short",
        ValueKind::VecI32 => "vector_int",
        ValueKind::VecI64 => "vector_long",
        ValueKind::VecF64 => "vector_float",
        ValueKind::VecLongDouble => "vector_double",
        ValueKind::VecStr => "vector_string",
        ValueKind::Str => "string",
        ValueKind::PyObject => "python_object",
    }
}

/// The inverse of [`graphml_type_name`]; `None` is `put_property`'s
/// `type_found == false` (`graphml.hpp:170-172`).
pub(crate) const fn graphml_type_kind(name: &str) -> Option<ValueKind> {
    Some(match name.as_bytes() {
        b"boolean" => ValueKind::Bool,
        b"short" => ValueKind::I16,
        b"int" => ValueKind::I32,
        b"long" => ValueKind::I64,
        b"float" => ValueKind::F64,
        b"double" => ValueKind::LongDouble,
        b"vector_boolean" => ValueKind::VecBool,
        b"vector_short" => ValueKind::VecI16,
        b"vector_int" => ValueKind::VecI32,
        b"vector_long" => ValueKind::VecI64,
        b"vector_float" => ValueKind::VecF64,
        b"vector_double" => ValueKind::VecLongDouble,
        b"vector_string" => ValueKind::VecStr,
        b"string" => ValueKind::Str,
        b"python_object" => ValueKind::PyObject,
        _ => return None,
    })
}

// Proved here rather than asserted in a test: the two tables above are the
// same table read in opposite directions, and nothing else may drift. The
// C++ pair is `prop_value_types` and `prop_type_names[]`, held in
// correspondence by position and by nothing else.
const fn round_trips(k: ValueKind) -> bool {
    match graphml_type_kind(graphml_type_name(k)) {
        Some(back) => back as u8 == k as u8,
        None => false,
    }
}
const _: () = {
    let mut i = 0;
    while i < ValueKind::ALL.len() {
        assert!(round_trips(ValueKind::ALL[i]));
        i += 1;
    }
};

/// `print_float` (`str_repr.hh:61-69`): `max_digits10` significant digits, the
/// `C` locale, and `ostream`'s default float format -- that is, `%.17g`.
///
/// Rust has no `%g`, so this derives it from `{:.16e}`: take the decimal
/// exponent `x`, and use a fixed form with `16 - x` fraction digits when
/// `-4 <= x < 17` and a scientific form otherwise, in both cases with trailing
/// zeros removed. The exponent is written with a sign and at least two digits,
/// which is what `ostream` does.
pub(crate) fn print_float(v: f64) -> String {
    if v.is_nan() {
        // glibc's `ostream` prints the sign of a NaN; `sscanf("%la")` reads
        // both spellings back.
        return if v.is_sign_negative() { "-nan" } else { "nan" }.to_owned();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf" } else { "inf" }.to_owned();
    }

    /// `std::numeric_limits<double>::max_digits10`.
    const P: usize = 17;
    let sci = format!("{:.*e}", P - 1, v);
    let (mantissa, exp) = sci.split_once('e').expect("Rust always emits the e");
    let x: i32 = exp.parse().expect("Rust always emits a decimal exponent");

    if x < -4 || x >= P as i32 {
        let mut s = trim_zeros(mantissa).to_owned();
        let _ = write!(s, "e{}{:02}", if x < 0 { '-' } else { '+' }, x.abs());
        s
    } else {
        let places = (P as i32 - 1 - x) as usize;
        trim_zeros(&format!("{v:.places$}")).to_owned()
    }
}

/// Drop a trailing run of zeros in the fraction, and the point with it.
fn trim_zeros(s: &str) -> &str {
    if !s.contains('.') {
        return s;
    }
    let s = s.trim_end_matches('0');
    s.strip_suffix('.').unwrap_or(s)
}

/// An exact C99 hexadecimal float for the opaque 16-byte `long double`.
///
/// The payload is read as the x86-64 System V layout: bytes 0..8 are the
/// 64-bit significand *including* its explicit integer bit, bytes 8..10 are
/// the sign and the 15-bit exponent, bytes 10..16 are padding. That is the
/// layout every `.gt` file in circulation carries, and it is the layout
/// `%La` prints. On a target whose `long double` is IEEE binary128 the bytes
/// mean something else; this module says so rather than pretending, and the
/// binary `.gt` path -- which never interprets the payload -- is unaffected.
///
/// The padding bytes are *not* representable in text: a value that goes
/// through a text format comes back with bytes 10..16 zeroed.
pub(crate) fn long_double_to_text(v: LongDouble) -> String {
    let frac = u64::from_le_bytes([
        v.0[0], v.0[1], v.0[2], v.0[3], v.0[4], v.0[5], v.0[6], v.0[7],
    ]);
    let se = u16::from_le_bytes([v.0[8], v.0[9]]);
    let sign = if se & 0x8000 != 0 { "-" } else { "" };
    let exp = i32::from(se & 0x7FFF);

    if exp == 0x7FFF {
        // Infinity is the integer bit alone; everything else is a NaN.
        return if frac == 0x8000_0000_0000_0000 {
            format!("{sign}inf")
        } else {
            format!("{sign}nan")
        };
    }
    if frac == 0 && exp == 0 {
        return format!("{sign}0x0p+0");
    }

    // value = frac * 2^(e - 63), and `d0.d1..d15` denotes frac / 2^60, so the
    // printed exponent is e - 3. `e` is -16382 for the subnormal range, where
    // the integer bit is clear.
    let e = if exp == 0 { -16382 } else { exp - 16383 };
    let digits = format!("{frac:016x}");
    let (head, tail) = digits.split_at(1);
    let rest = trim_zeros_hex(tail);
    if rest.is_empty() {
        format!("{sign}0x{head}p{:+}", e - 3)
    } else {
        format!("{sign}0x{head}.{rest}p{:+}", e - 3)
    }
}

fn trim_zeros_hex(s: &str) -> &str {
    s.trim_end_matches('0')
}

/// `lexical_cast<long double>` (`str_repr.hh:140-149`), as far as an opaque
/// payload allows: an exact inverse of [`long_double_to_text`] for the
/// hexadecimal form, and a widening of the `f64` parse for the decimal form.
pub(crate) fn long_double_from_text(s: &str) -> Result<LongDouble, String> {
    let s = s.trim();
    let (sign, body) = match s.strip_prefix('-') {
        Some(rest) => (0x8000u16, rest),
        None => (0u16, s.strip_prefix('+').unwrap_or(s)),
    };
    let lower = body.to_ascii_lowercase();
    if lower == "nan" {
        return Ok(compose_long_double(sign, 0x7FFF, 0xC000_0000_0000_0000));
    }
    if lower == "inf" || lower == "infinity" {
        return Ok(compose_long_double(sign, 0x7FFF, 0x8000_0000_0000_0000));
    }

    if let Some(hex) = lower.strip_prefix("0x") {
        let (frac, e2) = parse_hex_significand(hex)?;
        if frac == 0 {
            return Ok(compose_long_double(sign, 0, 0));
        }
        // Normalise so the integer bit is bit 63.
        let shift = frac.leading_zeros();
        let frac = frac << shift;
        let e = e2 + 63 - i64::from(shift);
        return Ok(scale_long_double(sign, frac, e));
    }

    let v: f64 = body
        .parse()
        .map_err(|_| format!("not a floating point value: {s:?}"))?;
    Ok(widen_f64(if sign != 0 { -v } else { v }))
}

/// The significand of a hexadecimal float as an integer, and the power of two
/// it must be multiplied by: the pair `(v, e)` with `value == v * 2^e`.
///
/// A significand with more than sixteen significant hexadecimal digits does
/// not fit `u64` and is truncated towards zero. `%La` never emits more than
/// sixteen, so that only costs precision on hand-written input.
fn parse_hex_significand(hex: &str) -> Result<(u64, i64), String> {
    let (digits, pexp) = match hex.split_once('p') {
        Some((d, e)) => (
            d,
            e.trim_start_matches('+')
                .parse::<i64>()
                .map_err(|_| format!("bad binary exponent in {hex:?}"))?,
        ),
        None => (hex, 0),
    };
    let (int_part, frac_part) = match digits.split_once('.') {
        Some((a, b)) => (a, b),
        None => (digits, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return Err(format!("no hexadecimal digits in {hex:?}"));
    }

    // Room for one more digit, or the value is already as precise as `u64`
    // can carry and further digits only move the exponent.
    const ROOM: u64 = u64::MAX >> 4;
    let mut value: u64 = 0;
    let mut e: i64 = 0;
    for c in int_part.chars() {
        let d = digit16(c)?;
        if value <= ROOM {
            value = (value << 4) | u64::from(d);
        } else {
            e += 4;
        }
    }
    for c in frac_part.chars() {
        let d = digit16(c)?;
        if value <= ROOM {
            value = (value << 4) | u64::from(d);
            e -= 4;
        }
    }
    Ok((value, e + pexp))
}

fn digit16(c: char) -> Result<u32, String> {
    c.to_digit(16)
        .ok_or_else(|| format!("bad hexadecimal digit {c:?}"))
}

/// Build the payload for `(frac / 2^63) * 2^e`, where `frac` carries its
/// integer bit at bit 63 -- that is, `e` is the exponent of the unit bit, and
/// the stored exponent is just `e` biased by 16383.
fn scale_long_double(sign: u16, frac: u64, e: i64) -> LongDouble {
    let biased = e + 16383;
    if biased >= 0x7FFF {
        return compose_long_double(sign, 0x7FFF, 0x8000_0000_0000_0000);
    }
    if biased <= 0 {
        // Subnormal: shift the significand right into the fixed exponent.
        let shift = 1 - biased;
        if shift >= 64 {
            return compose_long_double(sign, 0, 0);
        }
        return compose_long_double(sign, 0, frac >> shift);
    }
    compose_long_double(sign, biased as u16, frac)
}

fn compose_long_double(sign: u16, exp: u16, frac: u64) -> LongDouble {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&frac.to_le_bytes());
    b[8..10].copy_from_slice(&(sign | exp).to_le_bytes());
    LongDouble(b)
}

/// Widen an IEEE double into the 80-bit extended payload. Exact.
fn widen_f64(v: f64) -> LongDouble {
    let bits = v.to_bits();
    let sign = if bits >> 63 != 0 { 0x8000u16 } else { 0 };
    let exp = ((bits >> 52) & 0x7FF) as i64;
    let frac = bits & 0x000F_FFFF_FFFF_FFFF;
    if exp == 0x7FF {
        return if frac == 0 {
            compose_long_double(sign, 0x7FFF, 0x8000_0000_0000_0000)
        } else {
            compose_long_double(sign, 0x7FFF, 0xC000_0000_0000_0000)
        };
    }
    if exp == 0 {
        if frac == 0 {
            return compose_long_double(sign, 0, 0);
        }
        // A double's subnormal is an extended normal: shift the leading one
        // up to bit 63 and pay for it in the exponent.
        let shift = frac.leading_zeros();
        let significand = frac << shift;
        let e = -1022 - (i64::from(shift) - 11);
        return scale_long_double(sign, significand, e);
    }
    compose_long_double(sign, (exp - 1023 + 16383) as u16, 0x8000_0000_0000_0000 | (frac << 11))
}

/// `lexical_cast<double>` (`str_repr.hh:122-131`).
///
/// `sscanf("%la")` accepts a hexadecimal float, so this does too; it does not
/// accept the trailing garbage `sscanf` would silently stop at, because a
/// reader that takes `12abc` for `12` turns a corrupt file into a graph.
pub(crate) fn parse_f64(s: &str) -> Result<f64, String> {
    let t = s.trim();
    let (neg, body) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if body.len() > 2 && (body.starts_with("0x") || body.starts_with("0X")) {
        let (frac, e) = parse_hex_significand(&body[2..].to_ascii_lowercase())?;
        let v = (frac as f64) * exp2(e);
        return Ok(if neg { -v } else { v });
    }
    t.parse()
        .map_err(|_| format!("not a floating point value: {s:?}"))
}

/// `2^e` without `powi`'s repeated multiplication error for large `|e|`.
fn exp2(e: i64) -> f64 {
    let mut acc = 1.0f64;
    let mut n = e;
    while n > 1023 {
        acc *= f64::from_bits(0x7FE0_0000_0000_0000); // 2^1023
        n -= 1023;
    }
    while n < -1022 {
        acc *= f64::from_bits(0x0010_0000_0000_0000); // 2^-1022
        n += 1022;
    }
    acc * f64::from_bits((((n + 1023) as u64) & 0x7FF) << 52)
}

/// `lexical_cast<uint8_t>` (`str_repr.hh:52-55`): through `int`, then
/// truncated, so `300` is `44` and `-1` is `255`.
fn parse_byte(s: &str) -> Result<u8, String> {
    let t = s.trim();
    let v: i32 = t
        .parse()
        .map_err(|_| format!("not an integer: {s:?}"))?;
    Ok(v as u8)
}

fn parse_int<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.trim()
        .parse()
        .map_err(|_| format!("not an integer: {s:?}"))
}

/// `operator>>(istream&, vector<Type>&)` (`graph_util.hh:457-476`): split on
/// `,`, trim each field, cast each. The empty string is the empty vector.
fn parse_vec<T>(s: &str, f: impl Fn(&str) -> Result<T, String>) -> Result<Vec<T>, String> {
    let t = s.trim();
    if t.is_empty() {
        return Ok(Vec::new());
    }
    t.split(',').map(|x| f(x.trim())).collect()
}

/// `operator>>(istream&, vector<string>&)` (`str_repr.hh:178-207`): split on
/// the two-character sequence `", "`, then undo the escaping, in that order.
fn parse_str_vec(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(", ")
        .map(|part| part.replace(",\\ ", ", ").replace("\\\\", "\\"))
        .collect()
}

/// `operator<<(ostream&, const vector<string>&)` (`str_repr.hh:161-176`).
fn print_str_vec(v: &[String]) -> String {
    let mut out = String::new();
    for (i, s) in v.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&s.replace('\\', "\\\\").replace(", ", ",\\ "));
    }
    out
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// `base64_encode` (`base64.cc:27-38`).
pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        let take = chunk.len() + 1;
        for i in 0..take {
            out.push(B64[((n >> (18 - 6 * i)) & 0x3F) as usize] as char);
        }
        for _ in take..4 {
            out.push('=');
        }
    }
    out
}

/// `base64_decode` (`base64.cc:40-63`).
pub(crate) fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    for c in s.bytes() {
        if c == b'=' {
            break;
        }
        // Whitespace is what an XML pretty-printer inserts; expat hands it to
        // the reader and `binary_from_base64` would take it for a digit.
        if c.is_ascii_whitespace() {
            continue;
        }
        let v = B64
            .iter()
            .position(|&x| x == c)
            .ok_or_else(|| format!("not base64: {:?}", c as char))?;
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xFF) as u8);
        }
    }
    Ok(out)
}

/// `print_value<prop_value_types>` (`graphml.hpp:341-349`) for one element of
/// a column, or `None` when the column is too short for `i`.
pub(crate) fn value_at(column: &PropertyColumn, i: usize) -> Option<String> {
    Some(match column {
        // "chars should be printed as numbers" (`str_repr.hh:40-48`).
        PropertyColumn::Bool(v) => i32::from(*v.get(i)?).to_string(),
        PropertyColumn::I16(v) => v.get(i)?.to_string(),
        PropertyColumn::I32(v) => v.get(i)?.to_string(),
        PropertyColumn::I64(v) => v.get(i)?.to_string(),
        PropertyColumn::F64(v) => print_float(*v.get(i)?),
        PropertyColumn::LongDouble(v) => long_double_to_text(*v.get(i)?),
        PropertyColumn::Str(v) => v.get(i)?.clone(),
        PropertyColumn::VecBool(v) => join(v.get(i)?, |x| i32::from(*x).to_string()),
        PropertyColumn::VecI16(v) => join(v.get(i)?, i16::to_string),
        PropertyColumn::VecI32(v) => join(v.get(i)?, i32::to_string),
        PropertyColumn::VecI64(v) => join(v.get(i)?, i64::to_string),
        PropertyColumn::VecF64(v) => join(v.get(i)?, |x| print_float(*x)),
        PropertyColumn::VecLongDouble(v) => join(v.get(i)?, |x| long_double_to_text(*x)),
        PropertyColumn::VecStr(v) => print_str_vec(v.get(i)?),
        PropertyColumn::PyObject(v) => base64_encode(v.get(i)?),
    })
}

/// `operator<<(ostream&, const vector<Type>&)` (`graph_util.hh:446-456`).
fn join<T>(v: &[T], f: impl Fn(&T) -> String) -> String {
    let mut out = String::new();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&f(x));
    }
    out
}

/// One value into position `i` of a column that is already long enough.
///
/// `put_property::operator()` (`graphml.hpp:239-268`), including the
/// `boolean` spelling table at `:246-252`.
pub(crate) fn store_at(column: &mut PropertyColumn, i: usize, text: &str) -> Result<(), String> {
    match column {
        PropertyColumn::Bool(v) => v[i] = parse_byte(&boolean_spelling(text))?,
        PropertyColumn::I16(v) => v[i] = parse_int(text)?,
        PropertyColumn::I32(v) => v[i] = parse_int(text)?,
        PropertyColumn::I64(v) => v[i] = parse_int(text)?,
        PropertyColumn::F64(v) => v[i] = parse_f64(text)?,
        PropertyColumn::LongDouble(v) => v[i] = long_double_from_text(text)?,
        PropertyColumn::Str(v) => v[i] = text.to_owned(),
        PropertyColumn::VecBool(v) => v[i] = parse_vec(text, parse_byte)?,
        PropertyColumn::VecI16(v) => v[i] = parse_vec(text, parse_int)?,
        PropertyColumn::VecI32(v) => v[i] = parse_vec(text, parse_int)?,
        PropertyColumn::VecI64(v) => v[i] = parse_vec(text, parse_int)?,
        PropertyColumn::VecF64(v) => v[i] = parse_vec(text, parse_f64)?,
        PropertyColumn::VecLongDouble(v) => v[i] = parse_vec(text, long_double_from_text)?,
        PropertyColumn::VecStr(v) => v[i] = parse_str_vec(text),
        PropertyColumn::PyObject(v) => v[i] = base64_decode(text)?,
    }
    Ok(())
}

/// `val == "true" || val == "True"` and the two negatives (`graphml.hpp:248-251`).
fn boolean_spelling(text: &str) -> Cow<'_, str> {
    match text {
        "true" | "True" => Cow::Borrowed("1"),
        "false" | "False" => Cow::Borrowed("0"),
        other => Cow::Borrowed(other),
    }
}

/// Extend a column with default-constructed values up to `n`.
///
/// This is `vector_property_map::operator[]`'s implicit resize
/// (`fast_vector_property_map.hh`): a key never written keeps the member's
/// default, which is zero, the empty string or the empty vector.
pub(crate) fn grow_column(column: &mut PropertyColumn, n: usize) {
    macro_rules! grow {
        ($v:expr) => {
            if $v.len() < n {
                $v.resize_with(n, Default::default)
            }
        };
    }
    match column {
        PropertyColumn::Bool(v) => grow!(v),
        PropertyColumn::I16(v) => grow!(v),
        PropertyColumn::I32(v) => grow!(v),
        PropertyColumn::I64(v) => grow!(v),
        PropertyColumn::F64(v) => grow!(v),
        PropertyColumn::LongDouble(v) => grow!(v),
        PropertyColumn::Str(v) => grow!(v),
        PropertyColumn::VecBool(v) => grow!(v),
        PropertyColumn::VecI16(v) => grow!(v),
        PropertyColumn::VecI32(v) => grow!(v),
        PropertyColumn::VecI64(v) => grow!(v),
        PropertyColumn::VecF64(v) => grow!(v),
        PropertyColumn::VecLongDouble(v) => grow!(v),
        PropertyColumn::VecStr(v) => grow!(v),
        PropertyColumn::PyObject(v) => grow!(v),
    }
}

/// The `dynamic_properties` a text reader fills, as columns.
///
/// `dp` creates a map the first time a name is written and infers the member
/// from the value (`graph_io.cc:209-265`); the same name written later with a
/// different member throws `bad_any_cast` there and is an error here.
#[derive(Default)]
pub(crate) struct Columns {
    props: Vec<NamedProperty>,
    /// One name index per domain, so a lookup borrows the name it was handed
    /// instead of allocating a key for every value in the file.
    index: [HashMap<String, usize>; 3],
}

impl Columns {
    /// `put(name, dp, key, value)` for a key at position `at` of its domain.
    pub(crate) fn put(
        &mut self,
        name: &str,
        domain: PropertyDomain,
        kind: ValueKind,
        at: usize,
        text: &str,
    ) -> Result<(), String> {
        let by_name = &mut self.index[domain as usize];
        let slot = match by_name.get(name) {
            Some(&i) => i,
            None => {
                self.props.push(NamedProperty {
                    name: name.to_owned(),
                    domain,
                    values: PropertyColumn::empty(kind),
                });
                by_name.insert(name.to_owned(), self.props.len() - 1);
                self.props.len() - 1
            }
        };
        let column = &mut self.props[slot].values;
        if column.kind() != kind {
            return Err(format!(
                "property {name:?} was read as {} and is now declared {}",
                graphml_type_name(column.kind()),
                graphml_type_name(kind)
            ));
        }
        grow_column(column, at + 1);
        store_at(column, at, text).map_err(|e| format!("invalid value for key {name:?}: {e}"))
    }

    /// Size every column to its domain and hand the columns over.
    ///
    /// A property map must cover its whole index space before it can be
    /// written again ([`IoError::ShortProperty`]), and a key mentioned only
    /// for vertex 3 covers nothing else until this runs.
    pub(crate) fn finish(mut self, n_vertices: usize, edge_bound: usize) -> Vec<NamedProperty> {
        for p in &mut self.props {
            let n = match p.domain {
                PropertyDomain::Graph => 1,
                PropertyDomain::Vertex => n_vertices,
                PropertyDomain::Edge => edge_bound,
            };
            grow_column(&mut p.values, n);
        }
        self.props
    }
}

// ===========================================================================
// Reading
// ===========================================================================

/// Read a GraphML document.
///
/// Ports `graphml_reader` (`graphml.cpp:44-479`) as graph-tool drives it:
/// `read_graphml(stream, g, dp, true, true, true, ...)` (`graph_io.cc:337`),
/// that is `store_ids = true`, `integer_vertices = true` and
/// `ignore_directedness = true`.
///
/// * With `parse.nodeids="canonical"` a node id is `n` followed by the vertex
///   index (`graphml.cpp:288-320`); the first character is dropped whatever it
///   is, and the rest must parse.
/// * Otherwise node ids are interned in first-seen order and kept in the
///   vertex property [`VERTEX_ID_KEY`], which is what lets [`write`] emit the
///   file's own ids again.
/// * Edges are added in document order, so the `i`-th `<edge>` is
///   [`EdgeId`] `i`, and edge property columns are keyed by that.
/// * The directedness is the disjunction of every `edgedefault="directed"`
///   and every directed `<edge>`: `flip_directed` never clears
///   (`graphml.hpp:120-124`).
///
/// # Errors
///
/// [`IoError::Parse`], with the one-based line the offending token starts on,
/// for malformed XML, an unknown `attr.type`, a value that does not parse as
/// its declared type, or a node id that is not canonical where the file says
/// it is.
pub fn read<R: Read, H: Lookup>(mut r: R, lookup: H) -> Result<Document<H>, IoError> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    let text = std::str::from_utf8(&buf).map_err(|e| IoError::Parse {
        line: line_of(&buf, e.valid_up_to()),
        msg: "input is not valid UTF-8".to_owned(),
    })?;

    let mut rd = Reader::new(lookup);
    rd.run(text)?;
    rd.finish()
}

fn line_of(buf: &[u8], at: usize) -> usize {
    1 + buf[..at.min(buf.len())]
        .iter()
        .filter(|&&c| c == b'\n')
        .count()
}

/// A `<key>` declaration (`graphml.cpp:166-207`).
struct Key {
    /// `for=` -- only `graph`, `node` and `edge` carry data here.
    domain: Option<PropertyDomain>,
    name: String,
    kind: Option<ValueKind>,
    /// The raw `attr.type`, kept for the diagnostic when `kind` is `None`.
    type_name: String,
}

/// What a `<data>` element currently applies to (`graphml.cpp:94-99`).
#[derive(Clone, Copy)]
enum Active {
    /// Before any `<graph>`: the C++ reads an uninitialised `m_descriptor_kind`
    /// here. Ignoring the datum is the only defined behaviour available.
    None,
    Graph,
    Vertex(VertexId),
    Edge(EdgeId),
}

struct Reader<H: Lookup> {
    g: AdjList<H>,
    directed: bool,
    canonical_vertices: bool,
    canonical_edges: bool,
    keys: HashMap<String, Key>,
    /// Key id -> default text, ordered: `m_key_default` is a `std::map`
    /// (`graphml.cpp:455`) and the order decides which default lands first.
    defaults: BTreeMap<String, String>,
    vertex: HashMap<String, VertexId>,
    cols: Columns,
    active_key: String,
    active: Active,
    chardata: String,
}

impl<H: Lookup> Reader<H> {
    fn new(lookup: H) -> Self {
        Reader {
            g: AdjList::with_lookup(lookup),
            directed: false,
            canonical_vertices: false,
            canonical_edges: false,
            keys: HashMap::new(),
            defaults: BTreeMap::new(),
            vertex: HashMap::new(),
            cols: Columns::default(),
            active_key: String::new(),
            active: Active::None,
            chardata: String::new(),
        }
    }

    fn run(&mut self, text: &str) -> Result<(), IoError> {
        let mut xml = Xml::new(text);
        let mut stack: Vec<String> = Vec::new();
        loop {
            let line = xml.line;
            match xml.next()? {
                Event::Eof => {
                    if let Some(open) = stack.pop() {
                        return Err(parse_at(line, format_args!("unclosed element <{open}>")));
                    }
                    return Ok(());
                }
                Event::Text(t) => self.chardata.push_str(&t),
                Event::Start { name, attrs, empty } => {
                    self.start(local(name), &attrs, line)?;
                    if empty {
                        self.end(local(name), line)?;
                    } else {
                        stack.push(local(name).to_owned());
                    }
                }
                Event::End { name } => {
                    match stack.pop() {
                        Some(open) if open == local(name) => {}
                        Some(open) => {
                            return Err(parse_at(
                                line,
                                format_args!("</{}> closes <{open}>", local(name)),
                            ));
                        }
                        None => {
                            return Err(parse_at(
                                line,
                                format_args!("</{}> with nothing open", local(name)),
                            ));
                        }
                    }
                    self.end(local(name), line)?;
                }
            }
        }
    }

    /// `on_start_element` (`graphml.cpp:102-242`).
    fn start(&mut self, name: &str, attrs: &[(String, String)], line: usize) -> Result<(), IoError> {
        match name {
            "edge" => {
                let mut id = "";
                let mut source = "";
                let mut target = "";
                for (a, v) in attrs {
                    match a.as_str() {
                        "id" => id = v,
                        "source" => source = v,
                        "target" => target = v,
                        // `graphml.cpp:125` compares this attribute with
                        // "directed", but GraphML spells it "true"/"false"
                        // (and `edgedefault`, which *is* spelled "directed",
                        // is a different attribute). Both spellings are taken
                        // as directed here: the C++ reading is a defect that
                        // silently loses the directedness of a per-edge
                        // declaration, and `flip_directed` only ever sets.
                        "directed" if v == "directed" || v == "true" => self.directed = true,
                        _ => {}
                    }
                }
                self.active = Active::Edge(self.handle_edge(id, source, target, line)?);
            }
            "node" => {
                let mut id = "";
                for (a, v) in attrs {
                    if a == "id" {
                        id = v;
                    }
                }
                self.active = Active::Vertex(self.handle_vertex(id, line)?);
            }
            "data" => {
                for (a, v) in attrs {
                    if a == "key" {
                        self.active_key = v.clone();
                    }
                }
            }
            "key" => {
                let mut id = String::new();
                let mut key = Key {
                    domain: None,
                    name: String::new(),
                    kind: None,
                    type_name: String::new(),
                };
                for (a, v) in attrs {
                    match a.as_str() {
                        "id" => id = v.clone(),
                        "attr.name" => key.name = v.clone(),
                        "attr.type" => {
                            key.kind = graphml_type_kind(v);
                            key.type_name = v.clone();
                        }
                        "for" => {
                            key.domain = match v.as_str() {
                                "graph" => Some(PropertyDomain::Graph),
                                "node" => Some(PropertyDomain::Vertex),
                                "edge" => Some(PropertyDomain::Edge),
                                // `hyperedge`, `port`, `endpoint` and `all`
                                // are recognised and carry no data
                                // (`graphml.cpp:186-189`).
                                "hyperedge" | "port" | "endpoint" | "all" => None,
                                other => {
                                    return Err(parse_at(
                                        line,
                                        format_args!("unrecognized key kind '{other}'"),
                                    ));
                                }
                            };
                        }
                        _ => {}
                    }
                }
                self.active_key = id.clone();
                self.keys.insert(id, key);
            }
            "graph" => {
                for (a, v) in attrs {
                    match a.as_str() {
                        // `flip_directed` never clears (`graphml.hpp:120-124`),
                        // so the document is directed if anything says so.
                        "edgedefault" if v == "directed" => self.directed = true,
                        "parse.nodeids" => self.canonical_vertices = v == "canonical",
                        "parse.edgeids" => self.canonical_edges = v == "canonical",
                        _ => {}
                    }
                }
                self.active = Active::Graph;
            }
            _ => {}
        }
        // `self->m_character_data.clear()` (`graphml.cpp:241`), for every
        // element, not only the ones above.
        self.chardata.clear();
        Ok(())
    }

    /// `on_end_element` (`graphml.cpp:244-274`).
    fn end(&mut self, name: &str, line: usize) -> Result<(), IoError> {
        match name {
            "data" => {
                let key = std::mem::take(&mut self.active_key);
                let value = std::mem::take(&mut self.chardata);
                let r = self.apply(&key, self.active, &value);
                self.active_key = key;
                self.chardata = value;
                r.map_err(|e| parse_at(line, e))?;
            }
            "default" => {
                self.defaults
                    .insert(self.active_key.clone(), self.chardata.clone());
            }
            _ => {}
        }
        Ok(())
    }

    /// `set_{graph,vertex,edge}_property` (`graphml.hpp:150-225`).
    fn apply(&mut self, key_id: &str, to: Active, value: &str) -> Result<(), String> {
        let Some(key) = self.keys.get(key_id) else {
            // An unknown key id: `m_key_name[key_id]` default-constructs in
            // the C++ and writes a property named "" of type "", which then
            // fails `type_found`. Saying so is more useful.
            return Err(format!("no <key> declares id {key_id:?}"));
        };
        let Some(domain) = key.domain else {
            return Ok(());
        };
        let kind = key.kind.ok_or_else(|| {
            format!(
                "unrecognized type {:?} for key {}",
                key.type_name, key.name
            )
        })?;
        let name = key.name.clone();
        let at = match (domain, to) {
            (PropertyDomain::Graph, Active::Graph) => 0,
            (PropertyDomain::Vertex, Active::Vertex(v)) => v.index(),
            (PropertyDomain::Edge, Active::Edge(e)) => e.index(),
            // The descriptor and the key disagree: the C++ `any_cast` in
            // `set_*_property` throws `bad_any_cast`, which is not caught.
            (_, Active::None) => return Ok(()),
            _ => {
                return Err(format!(
                    "key {name:?} is declared for={} but appears on another element",
                    match domain {
                        PropertyDomain::Graph => "graph",
                        PropertyDomain::Vertex => "node",
                        PropertyDomain::Edge => "edge",
                    }
                ));
            }
        };
        self.cols.put(&name, domain, kind, at, value)
    }

    /// `handle_vertex` (`graphml.cpp:283-345`), on the
    /// `integer_vertices == true` path.
    fn handle_vertex(&mut self, id: &str, line: usize) -> Result<VertexId, IoError> {
        let (v, is_new) = if self.canonical_vertices {
            let n: usize = strip_head(id).parse().map_err(|_| {
                parse_at(line, format_args!("invalid vertex: {id}"))
            })?;
            if n > MAX_INDEX {
                return Err(IoError::IndexWidthExceeded);
            }
            let is_new = self.g.num_vertices() <= n;
            while self.g.num_vertices() <= n {
                self.g.add_vertex().map_err(graph_error)?;
            }
            (VertexId::from_index(n), is_new)
        } else if let Some(&v) = self.vertex.get(id) {
            (v, false)
        } else {
            let v = self.g.add_vertex().map_err(graph_error)?;
            self.vertex.insert(id.to_owned(), v);
            (v, true)
        };

        if is_new {
            self.apply_defaults(PropertyDomain::Vertex, Active::Vertex(v), line)?;
            if !self.canonical_vertices {
                self.cols
                    .put(VERTEX_ID_KEY, PropertyDomain::Vertex, ValueKind::Str, v.index(), id)
                    .map_err(|e| parse_at(line, e))?;
            }
        }
        Ok(v)
    }

    /// `handle_edge` (`graphml.cpp:364-392`).
    fn handle_edge(
        &mut self,
        id: &str,
        source: &str,
        target: &str,
        line: usize,
    ) -> Result<EdgeId, IoError> {
        let s = self.handle_vertex(source, line)?;
        let t = self.handle_vertex(target, line)?;
        let e = self.g.add_edge(s, t).map_err(graph_error)?.id();
        self.apply_defaults(PropertyDomain::Edge, Active::Edge(e), line)?;
        if !self.canonical_edges {
            self.cols
                .put(EDGE_ID_KEY, PropertyDomain::Edge, ValueKind::Str, e.index(), id)
                .map_err(|e| parse_at(line, e))?;
        }
        Ok(e)
    }

    /// The `<default>` of every key of this domain, applied to a new element
    /// (`graphml.cpp:333-339`, `:381-386`).
    fn apply_defaults(
        &mut self,
        domain: PropertyDomain,
        to: Active,
        line: usize,
    ) -> Result<(), IoError> {
        if self.defaults.is_empty() {
            return Ok(());
        }
        let pending: Vec<(String, String)> = self
            .defaults
            .iter()
            .filter(|(id, _)| self.keys.get(id.as_str()).and_then(|k| k.domain) == Some(domain))
            .map(|(id, v)| (id.clone(), v.clone()))
            .collect();
        for (id, v) in pending {
            self.apply(&id, to, &v).map_err(|e| parse_at(line, e))?;
        }
        Ok(())
    }

    fn finish(self) -> Result<Document<H>, IoError> {
        let n = self.g.num_vertices();
        let m = self.g.edge_bound().len();
        Ok(Document {
            graph: self.g,
            directed: self.directed,
            properties: self.cols.finish(n, m),
            comment: None,
        })
    }
}

/// Drop the first character of a canonical node id (`std::string(v,1)`,
/// `graphml.cpp:295`). Character, not byte: the C++ would split a multi-byte
/// code point, and the remainder would fail to parse either way.
fn strip_head(id: &str) -> &str {
    let mut it = id.chars();
    it.next();
    it.as_str()
}

pub(crate) fn parse_at(line: usize, msg: impl std::fmt::Display) -> IoError {
    IoError::Parse {
        line,
        msg: msg.to_string(),
    }
}

pub(crate) fn graph_error(e: GraphError) -> IoError {
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

/// The local part of an expanded name.
///
/// expat is created with `XML_ParserCreateNS(0, '|')` and the reader strips
/// the GraphML namespace URI (`graphml.cpp:57`, `:109`), so a document in a
/// *different* default namespace reads as empty there. Stripping whatever
/// prefix is present accepts those files instead of silently producing an
/// empty graph.
fn local(name: &str) -> &str {
    match name.rsplit_once(':') {
        Some((_, tail)) => tail,
        None => name,
    }
}

// ---------------------------------------------------------------------------
// A small XML scanner
// ---------------------------------------------------------------------------

/// One XML event. Text is decoded; attribute values are decoded.
enum Event<'a> {
    Start {
        name: &'a str,
        attrs: Vec<(String, String)>,
        empty: bool,
    },
    End {
        name: &'a str,
    },
    Text(Cow<'a, str>),
    Eof,
}

/// A byte scanner over XML, counting lines.
///
/// This replaces expat. It implements the subset GraphML uses -- elements,
/// attributes, character data, CDATA, comments, processing instructions and a
/// skipped DOCTYPE -- and rejects everything else with the line number, which
/// is what `XML_GetCurrentLineNumber` gives the C++ (`graphml.cpp:74`).
/// Entities beyond the five predefined ones and numeric references are not
/// supported; neither are they by any GraphML writer.
struct Xml<'a> {
    s: &'a str,
    b: &'a [u8],
    pos: usize,
    line: usize,
}

impl<'a> Xml<'a> {
    fn new(s: &'a str) -> Self {
        Xml {
            s,
            b: s.as_bytes(),
            pos: 0,
            line: 1,
        }
    }

    fn bump(&mut self, n: usize) {
        let end = (self.pos + n).min(self.b.len());
        self.line += self.b[self.pos..end].iter().filter(|&&c| c == b'\n').count();
        self.pos = end;
    }

    fn rest(&self) -> &'a str {
        &self.s[self.pos..]
    }

    fn err<T>(&self, msg: impl std::fmt::Display) -> Result<T, IoError> {
        Err(parse_at(self.line, msg))
    }

    /// Consume through the first occurrence of `close`, which must be there.
    fn through(&mut self, close: &str, what: &str) -> Result<(), IoError> {
        match self.rest().find(close) {
            Some(i) => {
                self.bump(i + close.len());
                Ok(())
            }
            None => self.err(format_args!("unterminated {what}")),
        }
    }

    fn skip_ws(&mut self) {
        while let Some(&c) = self.b.get(self.pos) {
            if c.is_ascii_whitespace() {
                self.bump(1);
            } else {
                break;
            }
        }
    }

    fn next(&mut self) -> Result<Event<'a>, IoError> {
        loop {
            let Some(&c) = self.b.get(self.pos) else {
                return Ok(Event::Eof);
            };
            if c != b'<' {
                let end = self.rest().find('<').map_or(self.b.len(), |i| self.pos + i);
                let raw = &self.s[self.pos..end];
                self.bump(end - self.pos);
                return Ok(Event::Text(decode_entities(raw, self.line)?));
            }
            let rest = self.rest();
            if rest.starts_with("<!--") {
                self.bump(4);
                self.through("-->", "comment")?;
                continue;
            }
            if rest.starts_with("<![CDATA[") {
                self.bump(9);
                let start = self.pos;
                match self.rest().find("]]>") {
                    Some(i) => {
                        let raw = &self.s[start..start + i];
                        self.bump(i + 3);
                        return Ok(Event::Text(Cow::Borrowed(raw)));
                    }
                    None => return self.err("unterminated CDATA section"),
                }
            }
            if rest.starts_with("<?") {
                self.bump(2);
                self.through("?>", "processing instruction")?;
                continue;
            }
            if rest.starts_with("<!") {
                // A DOCTYPE, with an optional internal subset.
                self.bump(2);
                let mut depth = 0usize;
                loop {
                    match self.b.get(self.pos) {
                        None => return self.err("unterminated declaration"),
                        Some(b'[') => depth += 1,
                        Some(b']') => depth = depth.saturating_sub(1),
                        Some(b'>') if depth == 0 => {
                            self.bump(1);
                            break;
                        }
                        Some(_) => {}
                    }
                    self.bump(1);
                }
                continue;
            }
            if rest.starts_with("</") {
                self.bump(2);
                let name = self.name()?;
                self.skip_ws();
                if self.b.get(self.pos) == Some(&b'>') {
                    self.bump(1);
                    return Ok(Event::End { name });
                }
                return self.err(format_args!("expected '>' to close </{name}"));
            }

            self.bump(1);
            let name = self.name()?;
            let mut attrs = Vec::new();
            loop {
                self.skip_ws();
                if self.rest().starts_with("/>") {
                    self.bump(2);
                    return Ok(Event::Start {
                        name,
                        attrs,
                        empty: true,
                    });
                }
                if self.b.get(self.pos) == Some(&b'>') {
                    self.bump(1);
                    return Ok(Event::Start {
                        name,
                        attrs,
                        empty: false,
                    });
                }
                let a = self.name()?;
                self.skip_ws();
                if self.b.get(self.pos) != Some(&b'=') {
                    return self.err(format_args!("expected '=' after attribute {a:?}"));
                }
                self.bump(1);
                self.skip_ws();
                let quote = match self.b.get(self.pos) {
                    Some(&q @ (b'"' | b'\'')) => q,
                    _ => return self.err(format_args!("attribute {a:?} is not quoted")),
                };
                self.bump(1);
                let start = self.pos;
                let end = match self.rest().find(quote as char) {
                    Some(i) => start + i,
                    None => return self.err(format_args!("unterminated value for {a:?}")),
                };
                let raw = &self.s[start..end];
                let line = self.line;
                self.bump(end - start + 1);
                attrs.push((a.to_owned(), decode_entities(raw, line)?.into_owned()));
            }
        }
    }

    /// An element or attribute name.
    fn name(&mut self) -> Result<&'a str, IoError> {
        let start = self.pos;
        while let Some(&c) = self.b.get(self.pos) {
            if c.is_ascii_whitespace() || matches!(c, b'/' | b'>' | b'=' | b'<') {
                break;
            }
            self.bump(1);
        }
        if self.pos == start {
            return self.err("expected a name");
        }
        Ok(&self.s[start..self.pos])
    }
}

/// The five predefined entities and numeric character references.
///
/// The inverse of `protect_xml_string` (`graphml.cpp:33-41`), which is
/// `boost::archive::iterators::xml_escape`: `& < > " '`, and nothing else.
fn decode_entities(s: &str, line: usize) -> Result<Cow<'_, str>, IoError> {
    if !s.contains('&') {
        return Ok(Cow::Borrowed(s));
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let end = tail
            .find(';')
            .ok_or_else(|| parse_at(line, "unterminated entity reference"))?;
        let body = &tail[1..end];
        match body {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            _ => {
                let code = if let Some(hex) = body.strip_prefix("#x").or(body.strip_prefix("#X")) {
                    u32::from_str_radix(hex, 16).ok()
                } else if let Some(dec) = body.strip_prefix('#') {
                    dec.parse().ok()
                } else {
                    None
                };
                let c = code.and_then(char::from_u32).ok_or_else(|| {
                    parse_at(line, format_args!("undefined entity &{body};"))
                })?;
                out.push(c);
            }
        }
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    Ok(Cow::Owned(out))
}

// ===========================================================================
// Writing
// ===========================================================================

/// `protect_xml_string` (`graphml.cpp:33-41`).
///
/// Control characters are *not* escaped, because XML 1.0 has no
/// representation for them at all -- not even a numeric reference. A
/// `string` property is a byte string in the value universe, so a value
/// carrying one produces a file expat will refuse; the C++ has the same hole
/// and the binary `.gt` format is the format that carries such a value.
fn protect(s: &str) -> Cow<'_, str> {
    if !s.contains(['&', '<', '>', '"', '\'']) {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    Cow::Owned(out)
}

/// The edges of a graph in [`EdgeId`] order.
///
/// See the crate documentation: adjacency order is not stable across
/// removals here, so every reproducible writer sorts.
pub(crate) fn edges_by_id<H: Lookup>(g: &AdjList<H>) -> Vec<(EdgeId, VertexId, VertexId)> {
    let mut v: Vec<(EdgeId, VertexId, VertexId)> = g
        .edges()
        .map(|e| (e.id(), e.source(), e.target()))
        .collect();
    v.sort_unstable_by_key(|t| t.0);
    v
}

/// How many values a column must hold to cover its domain.
pub(crate) fn domain_len<H: Lookup>(doc: &Document<H>, d: PropertyDomain) -> usize {
    match d {
        PropertyDomain::Graph => 1,
        PropertyDomain::Vertex => doc.graph.num_vertices(),
        PropertyDomain::Edge => doc.graph.edge_bound().len(),
    }
}

/// `dp`'s iteration order: by name, ties in insertion order
/// (`std::multimap`).
pub(crate) fn dp_order<H: Lookup>(doc: &Document<H>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..doc.properties.len()).collect();
    order.sort_by(|&a, &b| doc.properties[a].name.cmp(&doc.properties[b].name));
    order
}

/// Every column covers its domain, or the first one that does not.
pub(crate) fn check_columns<H: Lookup>(doc: &Document<H>) -> Result<(), IoError> {
    for p in &doc.properties {
        let need = domain_len(doc, p.domain);
        if p.values.len() < need {
            return Err(IoError::ShortProperty {
                name: p.name.clone(),
                have: p.values.len(),
                need,
            });
        }
    }
    Ok(())
}

/// Write a GraphML document, edges in [`EdgeId`](gt_core::ids::EdgeId) order.
///
/// Ports `write_graphml` (`graphml.hpp:353-528`) as graph-tool calls it, with
/// `ordered_vertices = true` (`graph_io.cc:421`):
///
/// * Properties are emitted in name order, and the key ids `key0`, `key1`,
///   ... follow that order.
/// * A vertex property named [`VERTEX_ID_KEY`] becomes the node ids and is
///   not emitted as a key; without one the ids are canonical, `n` followed by
///   the vertex index. [`EDGE_ID_KEY`] does the same for edges.
/// * A value whose text is empty is omitted (`:436-437`, `:468-469`,
///   `:516-517`). Reading the file back restores it, because an omitted datum
///   leaves the member's default -- but a column *all* of whose values are
///   empty is not reconstructed at all, since nothing then names it. That is
///   the C++ behaviour and it is why [`gt`](crate::gt) is the format for
///   round-tripping a document unchanged.
///
/// # Errors
///
/// [`IoError::ShortProperty`] if a column does not cover its domain, and
/// [`IoError::Io`] from the stream.
pub fn write<W: Write, H: Lookup>(w: W, doc: &Document<H>) -> Result<(), IoError> {
    let mut out = BufWriter::with_capacity(1 << 16, w);
    check_columns(doc)?;
    let order = dp_order(doc);

    out.write_all(
        b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
          <graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n         \
          xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n         \
          xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns \
          http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n\n",
    )?;
    out.write_all(b"  <!-- property keys -->\n")?;

    let vertex_ids = order
        .iter()
        .copied()
        .find(|&i| is_id_column(&doc.properties[i], PropertyDomain::Vertex, VERTEX_ID_KEY));
    let edge_ids = order
        .iter()
        .copied()
        .find(|&i| is_id_column(&doc.properties[i], PropertyDomain::Edge, EDGE_ID_KEY));

    // Key ids, assigned in `dp` order over the properties that get one.
    let mut key_id: HashMap<usize, String> = HashMap::new();
    let mut count = 0usize;
    for &i in &order {
        if Some(i) == vertex_ids || Some(i) == edge_ids {
            continue;
        }
        let p = &doc.properties[i];
        let id = format!("key{count}");
        count += 1;
        writeln!(
            out,
            "  <key id=\"{}\" for=\"{}\" attr.name=\"{}\" attr.type=\"{}\" />",
            protect(&id),
            match p.domain {
                PropertyDomain::Graph => "graph",
                PropertyDomain::Vertex => "node",
                PropertyDomain::Edge => "edge",
            },
            protect(&p.name),
            protect(graphml_type_name(p.kind())),
        )?;
        key_id.insert(i, id);
    }

    writeln!(
        out,
        "\n  <graph id=\"G\" edgedefault=\"{}\" parse.nodeids=\"{}\" parse.edgeids=\"{}\" parse.order=\"nodesfirst\">\n",
        if doc.directed { "directed" } else { "undirected" },
        if vertex_ids.is_none() { "canonical" } else { "free" },
        if edge_ids.is_none() { "canonical" } else { "free" },
    )?;

    out.write_all(b"   <!-- graph properties -->\n")?;
    for &i in &order {
        let p = &doc.properties[i];
        if p.domain != PropertyDomain::Graph {
            continue;
        }
        if let Some(val) = nonempty(&p.values, 0) {
            writeln!(
                out,
                "   <data key=\"{}\">{}</data>",
                protect(&key_id[&i]),
                protect(&val)
            )?;
        }
    }

    out.write_all(b"\n   <!-- vertices -->\n")?;
    for v in doc.graph.vertices() {
        let id = match vertex_ids {
            Some(i) => value_at(&doc.properties[i].values, v.index()).unwrap_or_default(),
            None => format!("n{}", v.index()),
        };
        writeln!(out, "    <node id=\"{}\">", protect(&id))?;
        for &i in &order {
            let p = &doc.properties[i];
            if p.domain != PropertyDomain::Vertex || Some(i) == vertex_ids {
                continue;
            }
            if let Some(val) = nonempty(&p.values, v.index()) {
                writeln!(
                    out,
                    "      <data key=\"{}\">{}</data>",
                    protect(&key_id[&i]),
                    protect(&val)
                )?;
            }
        }
        out.write_all(b"    </node>\n")?;
    }

    out.write_all(b"\n   <!-- edges -->\n")?;
    let names = |v: VertexId| match vertex_ids {
        Some(i) => value_at(&doc.properties[i].values, v.index()).unwrap_or_default(),
        None => format!("n{}", v.index()),
    };
    for (n, (e, s, t)) in edges_by_id(&doc.graph).into_iter().enumerate() {
        let id = match edge_ids {
            Some(i) => value_at(&doc.properties[i].values, e.index()).unwrap_or_default(),
            None => format!("e{n}"),
        };
        writeln!(
            out,
            "    <edge id=\"{}\" source=\"{}\" target=\"{}\">",
            protect(&id),
            protect(&names(s)),
            protect(&names(t)),
        )?;
        for &i in &order {
            let p = &doc.properties[i];
            if p.domain != PropertyDomain::Edge || Some(i) == edge_ids {
                continue;
            }
            if let Some(val) = nonempty(&p.values, e.index()) {
                writeln!(
                    out,
                    "      <data key=\"{}\">{}</data>",
                    protect(&key_id[&i]),
                    protect(&val)
                )?;
            }
        }
        out.write_all(b"    </edge>\n")?;
    }

    out.write_all(b"\n  </graph>\n</graphml>\n")?;
    // `BufWriter::drop` swallows the error; the C++ arms the stream with
    // `failbit` (`graph_io.cc:485`) so the same condition throws.
    out.flush().map_err(IoError::Io)
}

fn is_id_column(p: &NamedProperty, domain: PropertyDomain, name: &str) -> bool {
    p.domain == domain && p.name == name && p.kind() == ValueKind::Str
}

/// `if (val.empty()) continue;` (`graphml.hpp:436`).
fn nonempty(column: &PropertyColumn, i: usize) -> Option<String> {
    value_at(column, i).filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_float_is_percent_17g() {
        // Each of these is what `std::ostream` with `setprecision(17)` emits
        // (`str_repr.hh:62-69`), checked against C++ on x86-64.
        assert_eq!(print_float(0.0), "0");
        assert_eq!(print_float(-0.0), "-0");
        assert_eq!(print_float(1.5), "1.5");
        assert_eq!(print_float(0.1), "0.10000000000000001");
        assert_eq!(print_float(1e16), "10000000000000000");
        assert_eq!(print_float(1e17), "1e+17");
        assert_eq!(print_float(1e300), "1.0000000000000001e+300");
        assert_eq!(print_float(-2.5e-7), "-2.4999999999999999e-07");
        assert_eq!(print_float(1.0 / 3.0), "0.33333333333333331");
        assert_eq!(print_float(5e-324), "4.9406564584124654e-324");
        assert_eq!(print_float(f64::MAX), "1.7976931348623157e+308");
        assert_eq!(print_float(f64::INFINITY), "inf");
        assert_eq!(print_float(f64::NAN), "nan");
    }

    #[test]
    fn floats_round_trip_through_their_text() {
        for v in [
            0.0f64,
            -0.0,
            1.0,
            0.1,
            1.0 / 3.0,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::MAX,
            -12345.6789e-30,
        ] {
            let s = print_float(v);
            assert_eq!(parse_f64(&s).unwrap().to_bits(), v.to_bits(), "{s}");
        }
    }

    #[test]
    fn long_double_text_is_the_c99_hex_form() {
        // 1.0L on x86-64: significand 0x8000000000000000, exponent 16383.
        let one = compose_long_double(0, 16383, 0x8000_0000_0000_0000);
        assert_eq!(long_double_to_text(one), "0x8p-3");
        assert_eq!(long_double_from_text("0x8p-3").unwrap(), one);
        // glibc prints 1.0L as exactly that.
        assert_eq!(long_double_to_text(compose_long_double(0, 0, 0)), "0x0p+0");
        assert_eq!(
            long_double_to_text(compose_long_double(0x8000, 0x7FFF, 0x8000_0000_0000_0000)),
            "-inf"
        );
    }

    #[test]
    fn long_double_round_trips_its_own_text() {
        for &(sign, exp, frac) in &[
            (0u16, 16383u16, 0x8000_0000_0000_0000u64),
            (0x8000, 16383, 0xC000_0000_0000_0000),
            (0, 1, 0x8000_0000_0000_0001),
            (0, 32766, 0xFFFF_FFFF_FFFF_FFFF),
            (0, 0, 0x0000_0000_0000_0001),
            (0x8000, 0, 0),
        ] {
            let v = compose_long_double(sign, exp, frac);
            let s = long_double_to_text(v);
            assert_eq!(long_double_from_text(&s).unwrap(), v, "{s}");
        }
    }

    #[test]
    fn decimal_long_double_widens_exactly() {
        for v in [0.5f64, -3.25, 1e-300, 1e300, 0.1] {
            let ld = long_double_from_text(&print_float(v)).unwrap();
            // Widening is exact, so the hex form names the same real number.
            let back = long_double_from_text(&long_double_to_text(ld)).unwrap();
            assert_eq!(ld, back);
        }
    }

    #[test]
    fn base64_matches_the_boost_pipeline() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        for s in [&b""[..], b"f", b"fo", b"foo", b"foob", b"\x00\xff\x80"] {
            assert_eq!(base64_decode(&base64_encode(s)).unwrap(), s);
        }
    }

    #[test]
    fn string_vectors_escape_their_separator() {
        let v = vec![
            "a, b".to_owned(),
            "back\\slash".to_owned(),
            String::new(),
            "plain".to_owned(),
        ];
        let s = print_str_vec(&v);
        assert_eq!(parse_str_vec(&s), v);
    }
}
