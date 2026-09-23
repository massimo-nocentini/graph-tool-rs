//! Graphviz DOT.
//!
//! Ports the reader of `src/graph/read_graphviz_new.cpp` (the tokenizer at
//! `:138-312`, the parser at `:361-723` and `translate_results_to_graph` at
//! `:752-777`) and the writer of
//! `src/boost-workaround/boost/graph/graphviz.hpp`
//! (`write_graphviz`, `:252-288`, with `dynamic_vertex_properties_writer` and
//! `dynamic_properties_writer`, `:520-577`), as `graph_io.cc:334` and `:414`
//! call them.
//!
//! ## What DOT does *not* carry
//!
//! * **Types.** Every attribute is a string, on the way in and on the way
//!   out: `set_node_property` hands `dp` a `std::string`
//!   (`graphviz.hpp:785-790`), and `create_dynamic_map` therefore builds a
//!   `string` map (`graph_io.cc:217-261`). A document read back from DOT has
//!   [`ValueKind::Str`](gt_core::prop::ValueKind::Str) columns even where the
//!   document written out had `int64_t` ones. This is the "best-effort
//!   typing" the unit specification means, and [`read`] does not try to guess
//!   better: guessing would make `1`, `1.0` and `true` three different
//!   members depending on the data.
//! * **Graph properties.** `write_graphviz` passes `default_writer()` as the
//!   graph-property writer (`graphviz.hpp:626`), which emits nothing, so
//!   [`write`] drops them -- exactly as `g.save("x.dot")` does. [`read`] does
//!   parse `name = value` statements into graph properties, because files
//!   from other tools have them.
//! * **Vertex identity.** `translate_results_to_graph` walks
//!   `std::map<node_name, properties>` (`:754`), so vertices are created in
//!   *lexicographic order of their names*, not in the order the file mentions
//!   them. A ten-vertex graph written by [`write`] and read back has vertex
//!   `2` where it had vertex `10`. The names survive in the vertex property
//!   [`NODE_ID_KEY`], which is how the mapping is recovered, and it is why
//!   the round-trip test asserts topology *up to* that map.
//!
//! ## Line numbers
//!
//! `bad_graphviz_syntax` carries the offending token and no position at all
//! (`graphviz.hpp:673-679`, `read_graphviz_new.cpp:115-117`). [`IoError::Parse`]
//! has a line, so the lexer here counts them; the message keeps the C++
//! wording and its `(token is ...)` tail.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{BufWriter, Read, Write};

use gt_core::adj::{AdjList, Lookup};
use gt_core::ids::VertexId;
use gt_core::prop::ValueKind;

use crate::error::IoError;
use crate::graphml::{
    Columns, check_columns, domain_len, dp_order, edges_by_id, graph_error, parse_at, value_at,
};
use crate::gt::{Document, PropertyDomain};

/// The vertex property the reader fills with each node's name, and the writer
/// uses as the node name when it is present.
///
/// `read_graphviz(stream, *_mg, dp, "vertex_name", true, ...)`
/// (`graph_io.cc:334`) and `graphviz_insert_index` (`:389-404`), which looks
/// for exactly this name before falling back to the vertex index.
pub const NODE_ID_KEY: &str = "vertex_name";

/// The name `graphviz_insert_index` gives the index map when there is no
/// [`NODE_ID_KEY`] (`graph_io.cc:399`). The writer never emits it as an
/// attribute, because it *is* the node name.
pub const NODE_INDEX_KEY: &str = "vertex_id";

/// The root graph's key in `parser_result::graph_props`
/// (`read_graphviz_new.cpp:376`).
const ROOT: &str = "___root___";

// ===========================================================================
// Tokens
// ===========================================================================

#[derive(Clone, PartialEq, Eq, Debug)]
enum Tok {
    Strict,
    Graph,
    Digraph,
    Node,
    Edge,
    Subgraph,
    LBrace,
    RBrace,
    Semi,
    Equal,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Plus,
    LParen,
    RParen,
    At,
    DashGreater,
    DashDash,
    Ident(String),
    /// Only ever seen inside the lexer: string concatenation turns it into an
    /// [`Tok::Ident`] before the parser sees it (`read_graphviz_new.cpp:294-307`).
    Quoted(String),
    Eof,
}

impl Tok {
    /// `operator<<(ostream&, const token&)` (`read_graphviz_new.cpp:75-104`).
    fn describe(&self) -> String {
        let (tag, value) = match self {
            Tok::Strict => ("<strict>", ""),
            Tok::Graph => ("<graph>", ""),
            Tok::Digraph => ("<digraph>", ""),
            Tok::Node => ("<node>", ""),
            Tok::Edge => ("<edge>", ""),
            Tok::Subgraph => ("<subgraph>", ""),
            Tok::LBrace => ("<left_brace>", "{"),
            Tok::RBrace => ("<right_brace>", "}"),
            Tok::Semi => ("<semicolon>", ";"),
            Tok::Equal => ("<equal>", "="),
            Tok::LBracket => ("<left_bracket>", "["),
            Tok::RBracket => ("<right_bracket>", "]"),
            Tok::Comma => ("<comma>", ","),
            Tok::Colon => ("<colon>", ":"),
            Tok::Plus => ("<plus>", "+"),
            Tok::LParen => ("<left_paren>", "("),
            Tok::RParen => ("<right_paren>", ")"),
            Tok::At => ("<at>", "@"),
            Tok::DashGreater => ("<dash-greater>", "->"),
            Tok::DashDash => ("<dash-dash>", "--"),
            Tok::Ident(s) => ("<identifier>", s.as_str()),
            Tok::Quoted(s) => ("<quoted_string>", s.as_str()),
            Tok::Eof => ("<eof>", ""),
        };
        format!("{tag} '{value}'")
    }

    fn ident(&self) -> Option<&str> {
        match self {
            Tok::Ident(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct Token {
    kind: Tok,
    line: usize,
}

/// `tokenizer` (`read_graphviz_new.cpp:119-312`), as a hand-written scanner.
///
/// The C++ drives seven precompiled regexes; the order they are tried in is
/// load-bearing and is preserved: keyword/identifier, punctuation, number,
/// quoted string, HTML string. In particular `-` is punctuation only in `->`
/// and `--`, so `-1` lexes as the number it is.
struct Lexer<'a> {
    s: &'a str,
    b: &'a [u8],
    pos: usize,
    line: usize,
    /// `tokenizer::lookahead` (`:121`), used only by string concatenation.
    peeked: Option<Token>,
}

impl<'a> Lexer<'a> {
    fn new(s: &'a str) -> Self {
        Lexer {
            s,
            b: s.as_bytes(),
            pos: 0,
            line: 1,
            peeked: None,
        }
    }

    fn bump(&mut self, n: usize) {
        let end = (self.pos + n).min(self.b.len());
        self.line += self.b[self.pos..end].iter().filter(|&&c| c == b'\n').count();
        self.pos = end;
    }

    /// `lex_error` (`read_graphviz_new.cpp:107-113`).
    fn lex_error<T>(&self, msg: &str) -> Result<T, IoError> {
        Err(parse_at(
            self.line,
            match self.b.get(self.pos) {
                None => format!("{msg} (at end of input)"),
                Some(&c) => format!("{msg} (char is '{}')", c as char),
            },
        ))
    }

    fn skip_to_eol(&mut self) {
        while let Some(&c) = self.b.get(self.pos) {
            if c == b'\n' {
                return;
            }
            self.bump(1);
        }
    }

    /// `stuff_to_skip` (`:139-144`): whitespace, `//` and `/* */` comments,
    /// `#` comments at the start of a line, and a backslash-newline.
    fn skip(&mut self) -> Result<(), IoError> {
        loop {
            let Some(&c) = self.b.get(self.pos) else {
                return Ok(());
            };
            if c.is_ascii_whitespace() {
                self.bump(1);
                continue;
            }
            if c == b'#' && (self.pos == 0 || self.b[self.pos - 1] == b'\n') {
                self.skip_to_eol();
                continue;
            }
            if c == b'\\' && self.b.get(self.pos + 1) == Some(&b'\n') {
                self.bump(2);
                continue;
            }
            if c == b'/' {
                match self.b.get(self.pos + 1) {
                    Some(b'/') => {
                        self.skip_to_eol();
                        continue;
                    }
                    Some(b'*') => {
                        self.bump(2);
                        match self.s[self.pos..].find("*/") {
                            Some(i) => {
                                self.bump(i + 2);
                                continue;
                            }
                            None => return self.lex_error("Unclosed comment"),
                        }
                    }
                    _ => return Ok(()),
                }
            }
            return Ok(());
        }
    }

    /// `get_token_raw` (`:166-284`).
    fn raw(&mut self) -> Result<Token, IoError> {
        if let Some(t) = self.peeked.take() {
            return Ok(t);
        }
        self.skip()?;
        let line = self.line;
        let Some(&c) = self.b.get(self.pos) else {
            return Ok(Token {
                kind: Tok::Eof,
                line,
            });
        };

        // basic_id_token: "\\A([[:alpha:]_](?:\\w*))" (`:145`). The character
        // class is the C locale's, so a non-ASCII name must be quoted -- and
        // `escape_dot_string` quotes it, so a file this crate writes reads.
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = self.pos;
            while let Some(&c) = self.b.get(self.pos) {
                if c.is_ascii_alphanumeric() || c == b'_' {
                    self.bump(1);
                } else {
                    break;
                }
            }
            let word = &self.s[start..self.pos];
            let kind = match word.to_ascii_lowercase().as_str() {
                "strict" => Tok::Strict,
                "graph" => Tok::Graph,
                "digraph" => Tok::Digraph,
                "node" => Tok::Node,
                "edge" => Tok::Edge,
                "subgraph" => Tok::Subgraph,
                _ => Tok::Ident(word.to_owned()),
            };
            return Ok(Token { kind, line });
        }

        // punctuation_token: "\\A([][{};=,:+()@]|[-][>-])" (`:146`).
        let punct = match c {
            b'[' => Some(Tok::LBracket),
            b']' => Some(Tok::RBracket),
            b'{' => Some(Tok::LBrace),
            b'}' => Some(Tok::RBrace),
            b';' => Some(Tok::Semi),
            b'=' => Some(Tok::Equal),
            b',' => Some(Tok::Comma),
            b':' => Some(Tok::Colon),
            b'+' => Some(Tok::Plus),
            b'(' => Some(Tok::LParen),
            b')' => Some(Tok::RParen),
            b'@' => Some(Tok::At),
            b'-' => match self.b.get(self.pos + 1) {
                Some(b'>') => Some(Tok::DashGreater),
                Some(b'-') => Some(Tok::DashDash),
                // A `-` that starts a number falls through to the number
                // rule; anything else is an invalid character.
                _ => None,
            },
            _ => None,
        };
        if let Some(kind) = punct {
            self.bump(if matches!(kind, Tok::DashGreater | Tok::DashDash) {
                2
            } else {
                1
            });
            return Ok(Token { kind, line });
        }

        // number_token: "\\A([-]?(?:(?:\\.\\d+)|(?:\\d+(?:\\.\\d*)?)))" (`:147`).
        if (c == b'-' || c == b'.' || c.is_ascii_digit())
            && let Some(n) = self.number_len()
        {
            let text = self.s[self.pos..self.pos + n].to_owned();
            self.bump(n);
            return Ok(Token {
                kind: Tok::Ident(text),
                line,
            });
        }

        // quoted_string_token: "\\A(\"(?:[^\"\\\\]|(?:[\\\\].))*\")" (`:148`).
        if c == b'"' {
            return self.quoted(line);
        }

        if c == b'<' {
            return self.html(line);
        }

        self.lex_error("Invalid character")
    }

    /// The length of a number token at the cursor, if there is one.
    fn number_len(&self) -> Option<usize> {
        let mut i = self.pos;
        if self.b.get(i) == Some(&b'-') {
            i += 1;
        }
        let digits = |i: &mut usize| {
            let start = *i;
            while self.b.get(*i).is_some_and(u8::is_ascii_digit) {
                *i += 1;
            }
            *i - start
        };
        if self.b.get(i) == Some(&b'.') {
            i += 1;
            let mut j = i;
            if digits(&mut j) == 0 {
                return None;
            }
            return Some(j - self.pos);
        }
        let mut j = i;
        if digits(&mut j) == 0 {
            return None;
        }
        if self.b.get(j) == Some(&b'.') {
            j += 1;
            digits(&mut j);
        }
        Some(j - self.pos)
    }

    /// A quoted string, with the quotes removed and `\"` unescaped
    /// (`:230-245`). A backslash-newline is a line continuation and both
    /// characters go.
    ///
    /// The C++ regex's `[\\].` cannot match a backslash-newline, because `.`
    /// excludes newlines by default -- so the unescaping loop at `:239-242`
    /// is unreachable there and such a string is a lex error instead. This
    /// implements the loop's evident intent, which is also what Graphviz
    /// itself does.
    fn quoted(&mut self, line: usize) -> Result<Token, IoError> {
        self.bump(1);
        let mut out = String::new();
        // A multi-line string that never closes is reported on the line it
        // *opened*, which is the line a reader has to go and look at; the
        // C++ reports no position at all.
        let unclosed = || parse_at(line, "Unclosed quoted string (at end of input)");
        loop {
            let Some(&c) = self.b.get(self.pos) else {
                return Err(unclosed());
            };
            match c {
                b'"' => {
                    self.bump(1);
                    return Ok(Token {
                        kind: Tok::Quoted(out),
                        line,
                    });
                }
                b'\\' => match self.b.get(self.pos + 1) {
                    None => return Err(unclosed()),
                    Some(b'"') => {
                        out.push('"');
                        self.bump(2);
                    }
                    Some(b'\n') => self.bump(2),
                    // "Unescape quotes in the middle, but nothing else (see
                    // format spec)" (`:238`): the backslash stays.
                    Some(_) => {
                        out.push('\\');
                        self.bump(1);
                        let ch = self.s[self.pos..].chars().next().expect("not at the end");
                        out.push(ch);
                        self.bump(ch.len_utf8());
                    }
                },
                _ => {
                    let ch = self.s[self.pos..].chars().next().expect("not at the end");
                    out.push(ch);
                    self.bump(ch.len_utf8());
                }
            }
        }
    }

    /// An HTML string `<...>`, kept verbatim, angle brackets included
    /// (`:252-279`).
    fn html(&mut self, line: usize) -> Result<Token, IoError> {
        let start = self.pos;
        let mut depth = 0i32;
        loop {
            let Some(&c) = self.b.get(self.pos) else {
                return self.lex_error("Unclosed HTML string");
            };
            if c != b'<' {
                if c == b'>' {
                    self.bump(1);
                    depth -= 1;
                    if depth <= 0 {
                        break;
                    }
                    continue;
                }
                self.bump(1);
                continue;
            }
            if self.s[self.pos..].starts_with("<![CDATA[") {
                match self.s[self.pos..].find("]]>") {
                    Some(i) => {
                        self.bump(i + 3);
                        continue;
                    }
                    None => return self.lex_error("Invalid contents in HTML string"),
                }
            }
            self.bump(1);
            depth += 1;
        }
        Ok(Token {
            kind: Tok::Ident(self.s[start..self.pos].to_owned()),
            line,
        })
    }

    fn peek_raw(&mut self) -> Result<Tok, IoError> {
        if self.peeked.is_none() {
            self.peeked = Some(self.raw()?);
        }
        Ok(self.peeked.as_ref().expect("just filled").kind.clone())
    }

    /// `get_token` (`:294-307`): string concatenation with `+`, and a quoted
    /// string never reaches the parser as one.
    fn next(&mut self) -> Result<Token, IoError> {
        let t = self.raw()?;
        let Tok::Quoted(mut s) = t.kind else {
            return Ok(t);
        };
        while self.peek_raw()? == Tok::Plus {
            self.raw()?;
            let t2 = self.raw()?;
            match t2.kind {
                Tok::Quoted(s2) => s.push_str(&s2),
                _ => {
                    return self.lex_error("Must have quoted string after string concatenation");
                }
            }
        }
        Ok(Token {
            kind: Tok::Ident(s),
            line: t.line,
        })
    }
}

// ===========================================================================
// The parse tree
// ===========================================================================

type Props = BTreeMap<String, String>;

/// `node_and_port` (`read_graphviz_new.hpp:52-68`). The derived order is the
/// hand-written `operator<`: name, then angle, then location.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct NodePort {
    name: String,
    angle: String,
    location: Vec<String>,
}

#[derive(Clone, Debug)]
enum Endpoint {
    Node(NodePort),
    Subgraph(String),
}

struct EdgeInfo {
    source: NodePort,
    target: NodePort,
    props: Props,
}

#[derive(Clone, Default)]
struct SubgraphInfo {
    def_node: Props,
    def_edge: Props,
    members: Vec<(bool, String)>,
}

#[derive(Default)]
struct Parsed {
    directed: bool,
    strict: bool,
    nodes: BTreeMap<String, Props>,
    edges: Vec<EdgeInfo>,
    graph_props: BTreeMap<String, Props>,
}

struct Parser<'a> {
    lex: Lexer<'a>,
    lookahead: Option<Token>,
    r: Parsed,
    subgraphs: BTreeMap<String, SubgraphInfo>,
    current: String,
    sgcounter: usize,
    existing: BTreeSet<(String, String)>,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        let mut p = Parser {
            lex: Lexer::new(s),
            lookahead: None,
            r: Parsed::default(),
            subgraphs: BTreeMap::new(),
            current: ROOT.to_owned(),
            sgcounter: 0,
            existing: BTreeSet::new(),
        };
        p.subgraphs.insert(ROOT.to_owned(), SubgraphInfo::default());
        p.r.graph_props.insert(ROOT.to_owned(), Props::new());
        p
    }

    fn get(&mut self) -> Result<Token, IoError> {
        match self.lookahead.take() {
            Some(t) => Ok(t),
            None => self.lex.next(),
        }
    }

    fn peek(&mut self) -> Result<&Token, IoError> {
        if self.lookahead.is_none() {
            self.lookahead = Some(self.lex.next()?);
        }
        Ok(self.lookahead.as_ref().expect("just filled"))
    }

    fn peek_kind(&mut self) -> Result<Tok, IoError> {
        Ok(self.peek()?.kind.clone())
    }

    /// `parser::error` (`read_graphviz_new.cpp:400-402`), which reports the
    /// token it is *looking at*, not the one it consumed.
    fn error<T>(&mut self, msg: &str) -> Result<T, IoError> {
        let t = self.peek()?;
        let (line, described) = (t.line, t.kind.describe());
        Err(parse_at(line, format!("{msg} (token is \"{described}\")")))
    }

    fn eat(&mut self, k: &Tok) -> Result<bool, IoError> {
        if self.peek()?.kind == *k {
            self.get()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn current_info(&mut self) -> &mut SubgraphInfo {
        self.subgraphs.entry(self.current.clone()).or_default()
    }

    fn current_graph_props(&mut self) -> &mut Props {
        self.r.graph_props.entry(self.current.clone()).or_default()
    }

    /// `parse_graph` (`:404-432`), with `want_directed == 2`: graph-tool
    /// always passes `ignore_directedness = true` (`graph_io.cc:334`), so the
    /// file decides and no mismatch is an error.
    fn parse_graph(&mut self) -> Result<(), IoError> {
        if self.eat(&Tok::Strict)? {
            self.r.strict = true;
        }
        match self.peek_kind()? {
            Tok::Graph => self.r.directed = false,
            Tok::Digraph => self.r.directed = true,
            _ => return self.error("Wanted \"graph\" or \"digraph\""),
        }
        self.get()?;
        match self.peek_kind()? {
            Tok::Ident(_) => {
                self.get()?;
            }
            Tok::LBrace => {}
            _ => return self.error("Wanted a graph name or left brace"),
        }
        if !self.eat(&Tok::LBrace)? {
            return self.error("Wanted a left brace to start the graph");
        }
        self.parse_stmt_list()?;
        if !self.eat(&Tok::RBrace)? {
            return self.error("Wanted a right brace to end the graph");
        }
        if self.peek_kind()? != Tok::Eof {
            return self.error("Wanted end of file");
        }
        Ok(())
    }

    /// `parse_stmt_list` (`:434-441`).
    fn parse_stmt_list(&mut self) -> Result<(), IoError> {
        loop {
            if self.peek_kind()? == Tok::RBrace {
                return Ok(());
            }
            self.parse_stmt()?;
            self.eat(&Tok::Semi)?;
        }
    }

    /// `parse_stmt` (`:443-485`).
    fn parse_stmt(&mut self) -> Result<(), IoError> {
        match self.peek_kind()? {
            Tok::Node | Tok::Edge | Tok::Graph => self.parse_attr_stmt(),
            Tok::Subgraph | Tok::LBrace | Tok::Ident(_) => {
                let id = self.get()?;
                if id.kind.ident().is_some() && self.peek_kind()? == Tok::Equal {
                    self.get()?;
                    let Tok::Ident(_) = self.peek_kind()? else {
                        return self.error("Wanted identifier as right side of =");
                    };
                    let id2 = self.get()?;
                    let key = id.kind.ident().expect("checked").to_owned();
                    let val = id2.kind.ident().expect("checked").to_owned();
                    self.current_graph_props().insert(key, val);
                    return Ok(());
                }
                let ep = self.parse_endpoint_rest(id)?;
                if matches!(self.peek_kind()?, Tok::DashDash | Tok::DashGreater) {
                    return self.parse_edge_stmt(ep);
                }
                match ep {
                    Endpoint::Node(np) => {
                        // The node already exists: `parse_node_and_port` gave
                        // it the defaults in force at its first mention.
                        let mut here = Props::new();
                        if self.peek_kind()? == Tok::LBracket {
                            self.parse_attr_list(&mut here)?;
                        }
                        let entry = self.r.nodes.entry(np.name.clone()).or_default();
                        for (k, v) in here {
                            entry.insert(k, v);
                        }
                        self.current_info().members.push((false, np.name));
                    }
                    Endpoint::Subgraph(name) => {
                        self.current_info().members.push((true, name));
                    }
                }
                Ok(())
            }
            _ => self.error("Invalid start token for statement"),
        }
    }

    /// `parse_attr_stmt` (`:487-494`).
    fn parse_attr_stmt(&mut self) -> Result<(), IoError> {
        let which = self.get()?.kind;
        let mut props = match which {
            Tok::Graph => self.current_graph_props().clone(),
            Tok::Node => self.current_info().def_node.clone(),
            Tok::Edge => self.current_info().def_edge.clone(),
            _ => unreachable!("parse_stmt only routes the three keywords here"),
        };
        self.parse_attr_list(&mut props)?;
        match which {
            Tok::Graph => *self.current_graph_props() = props,
            Tok::Node => self.current_info().def_node = props,
            Tok::Edge => self.current_info().def_edge = props,
            _ => unreachable!(),
        }
        Ok(())
    }

    /// `parse_endpoint` (`:496-508`).
    fn parse_endpoint(&mut self) -> Result<Endpoint, IoError> {
        match self.peek_kind()? {
            Tok::Subgraph | Tok::LBrace | Tok::Ident(_) => {
                let first = self.get()?;
                self.parse_endpoint_rest(first)
            }
            _ => self.error("Wanted \"subgraph\", \"{\", or identifier to start node or subgraph"),
        }
    }

    /// `parse_endpoint_rest` (`:510-516`).
    fn parse_endpoint_rest(&mut self, first: Token) -> Result<Endpoint, IoError> {
        match first.kind {
            Tok::Subgraph | Tok::LBrace => {
                Ok(Endpoint::Subgraph(self.parse_subgraph(&first.kind)?))
            }
            _ => Ok(Endpoint::Node(self.parse_node_and_port(&first)?)),
        }
    }

    /// `parse_subgraph` (`:518-546`).
    fn parse_subgraph(&mut self, first: &Tok) -> Result<String, IoError> {
        let mut name = String::new();
        let mut anonymous = true;
        if *first == Tok::Subgraph
            && let Tok::Ident(id) = self.peek_kind()?
        {
            self.get()?;
            name = id;
            anonymous = false;
        }
        if anonymous {
            self.sgcounter += 1;
            name = format!("___subgraph_{}", self.sgcounter);
        }
        if !self.subgraphs.contains_key(&name) {
            // "Initialize properties and defaults ... Except member list"
            // (`:531-532`).
            let mut copy = self.current_info().clone();
            copy.members.clear();
            self.subgraphs.insert(name.clone(), copy);
        }
        if *first == Tok::Subgraph && self.peek_kind()? != Tok::LBrace {
            if anonymous {
                return self.error("Subgraph reference needs a name");
            }
            return Ok(name);
        }
        let old = std::mem::replace(&mut self.current, name.clone());
        if !self.eat(&Tok::LBrace)? {
            return self.error("Wanted left brace to start subgraph");
        }
        self.parse_stmt_list()?;
        if !self.eat(&Tok::RBrace)? {
            return self.error("Wanted right brace to end subgraph");
        }
        self.current = old;
        Ok(name)
    }

    /// `parse_node_and_port` (`:548-601`).
    fn parse_node_and_port(&mut self, name: &Token) -> Result<NodePort, IoError> {
        let mut id = NodePort {
            name: name.kind.ident().unwrap_or_default().to_owned(),
            ..NodePort::default()
        };
        loop {
            match self.peek_kind()? {
                Tok::At => {
                    self.get()?;
                    let Tok::Ident(a) = self.peek_kind()? else {
                        return self.error("Wanted identifier as port angle");
                    };
                    if !id.angle.is_empty() {
                        return self.error("Duplicate port angle");
                    }
                    self.get()?;
                    id.angle = a;
                }
                Tok::Colon => {
                    self.get()?;
                    if !id.location.is_empty() {
                        return self.error("Duplicate port location");
                    }
                    match self.peek_kind()? {
                        Tok::Ident(first) => {
                            self.get()?;
                            id.location.push(first);
                            if self.peek_kind()? == Tok::Colon {
                                self.get()?;
                                let Tok::Ident(second) = self.peek_kind()? else {
                                    return self.error("Wanted identifier as port location");
                                };
                                self.get()?;
                                id.location.push(second);
                            }
                        }
                        Tok::LParen => {
                            self.get()?;
                            let Tok::Ident(a) = self.peek_kind()? else {
                                return self
                                    .error("Wanted identifier as first element of port location");
                            };
                            self.get()?;
                            id.location.push(a);
                            if !self.eat(&Tok::Comma)? {
                                return self.error("Wanted comma between parts of port location");
                            }
                            let Tok::Ident(b) = self.peek_kind()? else {
                                return self
                                    .error("Wanted identifier as second element of port location");
                            };
                            self.get()?;
                            id.location.push(b);
                            if !self.eat(&Tok::RParen)? {
                                return self
                                    .error("Wanted right parenthesis to close port location");
                            }
                        }
                        _ => {
                            return self
                                .error("Wanted identifier or left parenthesis as start of port location");
                        }
                    }
                }
                _ => break,
            }
        }
        if !self.r.nodes.contains_key(&id.name) {
            // First mention: the node takes the node defaults in force here.
            let defaults = self.current_info().def_node.clone();
            self.r.nodes.insert(id.name.clone(), defaults);
        }
        Ok(id)
    }

    /// `parse_edge_stmt` (`:603-632`).
    fn parse_edge_stmt(&mut self, lhs: Endpoint) -> Result<(), IoError> {
        let mut chain = vec![lhs];
        loop {
            match self.peek_kind()? {
                Tok::DashDash => {
                    if self.r.directed {
                        return self.error("Using -- in directed graph");
                    }
                    self.get()?;
                    let ep = self.parse_endpoint()?;
                    chain.push(ep);
                }
                Tok::DashGreater => {
                    if !self.r.directed {
                        return self.error("Using -> in undirected graph");
                    }
                    self.get()?;
                    let ep = self.parse_endpoint()?;
                    chain.push(ep);
                }
                _ => break,
            }
        }
        let mut props = self.current_info().def_edge.clone();
        if self.peek_kind()? == Tok::LBracket {
            self.parse_attr_list(&mut props)?;
        }
        for pair in chain.windows(2) {
            self.do_orig_edge(&pair[0], &pair[1], &props);
        }
        Ok(())
    }

    /// `do_orig_edge` (`:634-643`): a subgraph endpoint expands to every node
    /// it holds, in `std::set` order.
    fn do_orig_edge(&mut self, src: &Endpoint, tgt: &Endpoint, props: &Props) {
        let sources = self.members_of(src);
        let targets = self.members_of(tgt);
        for s in &sources {
            for t in &targets {
                self.do_edge(s, t, props);
            }
        }
    }

    /// `get_recursive_members` (`:645-677`), including the `done` set that
    /// keeps a cyclic containment from looping.
    fn members_of(&self, ep: &Endpoint) -> BTreeSet<NodePort> {
        let mut result = BTreeSet::new();
        let mut work = vec![ep.clone()];
        let mut done: BTreeSet<String> = BTreeSet::new();
        while let Some(ep) = work.pop() {
            match ep {
                Endpoint::Subgraph(name) => {
                    if done.insert(name.clone())
                        && let Some(info) = self.subgraphs.get(&name)
                    {
                        for (is_subgraph, member) in &info.members {
                            if *is_subgraph {
                                work.push(Endpoint::Subgraph(member.clone()));
                            } else {
                                work.push(Endpoint::Node(NodePort {
                                    name: member.clone(),
                                    ..NodePort::default()
                                }));
                            }
                        }
                    }
                }
                Endpoint::Node(np) => {
                    result.insert(np);
                }
            }
        }
        result
    }

    /// `do_edge` (`:679-695`): in a `strict` graph a self-loop and a repeated
    /// pair are both dropped.
    fn do_edge(&mut self, src: &NodePort, tgt: &NodePort, props: &Props) {
        if self.r.strict {
            if src.name == tgt.name {
                return;
            }
            if !self
                .existing
                .insert((src.name.clone(), tgt.name.clone()))
            {
                return;
            }
        }
        self.r.edges.push(EdgeInfo {
            source: src.clone(),
            target: tgt.clone(),
            props: props.clone(),
        });
    }

    /// `parse_attr_list` (`:697-722`): several bracketed lists in a row are
    /// one list, and a bare name is `name=true`.
    fn parse_attr_list(&mut self, props: &mut Props) -> Result<(), IoError> {
        loop {
            if !self.eat(&Tok::LBracket)? {
                return self.error("Wanted left bracket to start attribute list");
            }
            loop {
                match self.peek_kind()? {
                    Tok::RBracket => break,
                    Tok::Ident(lhs) => {
                        self.get()?;
                        let mut rhs = "true".to_owned();
                        if self.eat(&Tok::Equal)? {
                            let Tok::Ident(v) = self.peek_kind()? else {
                                return self.error("Wanted identifier as value of attributed");
                            };
                            self.get()?;
                            rhs = v;
                        }
                        props.insert(lhs, rhs);
                    }
                    _ => return self.error("Wanted identifier as name of attribute"),
                }
                if self.eat(&Tok::Comma)? {
                    continue;
                }
                break;
            }
            if !self.eat(&Tok::RBracket)? {
                return self.error("Wanted right bracket to end attribute list");
            }
            if self.peek_kind()? != Tok::LBracket {
                return Ok(());
            }
        }
    }
}

// ===========================================================================
// Reading
// ===========================================================================

/// Read a DOT document. Property typing is best-effort: DOT has none.
///
/// Every attribute becomes a [`ValueKind::Str`] column, and each vertex
/// additionally gets [`NODE_ID_KEY`] holding the name the file gave it.
///
/// * Vertices are created in lexicographic order of their names, not in the
///   order of first mention (`read_graphviz_new.cpp:754-757`). See the module
///   documentation.
/// * Edges are created in file order, so the `i`-th edge statement is
///   [`EdgeId`](gt_core::ids::EdgeId) `i` -- except that an edge whose
///   endpoint is a subgraph expands to one edge per member pair, in the
///   members' sorted order (`:634-643`).
/// * `strict` drops self-loops and repeated endpoint pairs (`:679-687`);
///   without it, parallel edges are kept, because this port's adjacency
///   accepts them.
/// * The result's `directed` is the file's own `graph`/`digraph` keyword:
///   graph-tool passes `ignore_directedness = true` (`graph_io.cc:334`), so a
///   file never conflicts with the graph it is read into.
///
/// # Errors
///
/// [`IoError::Parse`], with the line of the offending token, for any lexical
/// or syntactic error; [`IoError::IndexWidthExceeded`] if the file has more
/// vertices or edges than [`Raw`](gt_core::ids::Raw) can index.
pub fn read<R: Read, H: Lookup>(mut r: R, lookup: H) -> Result<Document<H>, IoError> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    let text = std::str::from_utf8(&buf).map_err(|e| IoError::Parse {
        line: 1 + buf[..e.valid_up_to()].iter().filter(|&&c| c == b'\n').count(),
        msg: "input is not valid UTF-8".to_owned(),
    })?;

    let mut p = Parser::new(text);
    p.parse_graph()?;
    let parsed = p.r;

    let mut g: AdjList<H> = AdjList::with_lookup(lookup);
    let mut cols = Columns::default();
    let mut of: HashMap<&str, VertexId> = HashMap::with_capacity(parsed.nodes.len());

    // `translate_results_to_graph` (`:752-777`).
    for (name, props) in &parsed.nodes {
        let v = g.add_vertex().map_err(graph_error)?;
        of.insert(name.as_str(), v);
        // `put(node_id_prop_, dp_, v, node)` (`graphviz.hpp:762`).
        put(&mut cols, NODE_ID_KEY, PropertyDomain::Vertex, v.index(), name)?;
        for (k, val) in props {
            put(&mut cols, k, PropertyDomain::Vertex, v.index(), val)?;
        }
    }
    for e in &parsed.edges {
        let s = of[e.source.name.as_str()];
        let t = of[e.target.name.as_str()];
        let id = g.add_edge(s, t).map_err(graph_error)?.id();
        for (k, val) in &e.props {
            put(&mut cols, k, PropertyDomain::Edge, id.index(), val)?;
        }
    }
    if let Some(gp) = parsed.graph_props.get(ROOT) {
        for (k, val) in gp {
            put(&mut cols, k, PropertyDomain::Graph, 0, val)?;
        }
    }

    let (nv, ne) = (g.num_vertices(), g.edge_bound().len());
    Ok(Document {
        graph: g,
        directed: parsed.directed,
        properties: cols.finish(nv, ne),
        comment: None,
    })
}

/// Every DOT attribute is a string, so this can only fail on a name that
/// already names a column of another member -- which cannot happen here,
/// since this reader creates none.
fn put(
    cols: &mut Columns,
    name: &str,
    domain: PropertyDomain,
    at: usize,
    value: &str,
) -> Result<(), IoError> {
    cols.put(name, domain, ValueKind::Str, at, value)
        .map_err(|e| parse_at(0, e))
}

// ===========================================================================
// Writing
// ===========================================================================

/// `escape_dot_string` (`graphviz.hpp:74-83`): unquoted when the whole string
/// matches `((alpha|_)\w*) | (-?((\.\d+)|(\d+(\.\d*)?)))`, otherwise quoted
/// with `"` escaped -- and nothing else escaped, which is what the reader's
/// "unescape quotes in the middle, but nothing else" expects.
fn escape_dot(s: &str) -> Cow<'_, str> {
    if is_plain_id(s) {
        return Cow::Borrowed(s);
    }
    Cow::Owned(format!("\"{}\"", s.replace('"', "\\\"")))
}

fn is_plain_id(s: &str) -> bool {
    let b = s.as_bytes();
    let Some(&first) = b.first() else {
        return false;
    };
    if first.is_ascii_alphabetic() || first == b'_' {
        return b[1..].iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_');
    }
    let t = s.strip_prefix('-').unwrap_or(s);
    let digits = |x: &str| !x.is_empty() && x.bytes().all(|c| c.is_ascii_digit());
    if let Some(frac) = t.strip_prefix('.') {
        return digits(frac);
    }
    match t.split_once('.') {
        Some((int, frac)) => digits(int) && frac.bytes().all(|c| c.is_ascii_digit()),
        None => digits(t),
    }
}

/// Write a DOT document.
///
/// Ports `write_graphviz` with `dynamic_properties` (`graphviz.hpp:606-630`)
/// as `do_write_to_file` calls it (`graph_io.cc:414-419`), byte for byte,
/// including the two spaces before an edge's attribute list -- the edge line
/// ends with a space and the attribute writer opens with one
/// (`graphviz.hpp:283-286`, `:527`).
///
/// * Node names come from the vertex property [`NODE_ID_KEY`] when there is
///   one, and from the vertex index otherwise (`graph_io.cc:389-404`). The
///   property that supplies them is not also emitted as an attribute, and
///   neither is one named [`NODE_INDEX_KEY`], which is the index map
///   graph-tool inserts for exactly this purpose.
/// * Attributes are emitted in property-name order, per `dp`'s
///   `std::multimap`.
/// * Edges are emitted in [`EdgeId`](gt_core::ids::EdgeId) order.
/// * **Graph properties are dropped**, because the C++ passes
///   `default_writer()` for them. Use [`gt`](crate::gt) or
///   [`graphml`](crate::graphml) if they matter.
///
/// # Errors
///
/// [`IoError::ShortProperty`] if a column does not cover its domain, and
/// [`IoError::Io`] from the stream.
pub fn write<W: Write, H: Lookup>(w: W, doc: &Document<H>) -> Result<(), IoError> {
    let mut out = BufWriter::with_capacity(1 << 16, w);
    check_columns(doc)?;
    let order = dp_order(doc);

    let node_id = order.iter().copied().find(|&i| {
        doc.properties[i].domain == PropertyDomain::Vertex && doc.properties[i].name == NODE_ID_KEY
    });

    let name_of = |v: VertexId| match node_id {
        Some(i) => value_at(&doc.properties[i].values, v.index()).unwrap_or_default(),
        None => v.index().to_string(),
    };

    let (keyword, arrow) = if doc.directed {
        ("digraph", "->")
    } else {
        ("graph", "--")
    };
    writeln!(out, "{keyword} G {{")?;

    for v in doc.graph.vertices() {
        write!(out, "{}", escape_dot(&name_of(v)))?;
        attributes(&mut out, doc, &order, PropertyDomain::Vertex, v.index(), node_id)?;
        out.write_all(b";\n")?;
    }

    for (e, s, t) in edges_by_id(&doc.graph) {
        write!(
            out,
            "{}{arrow}{} ",
            escape_dot(&name_of(s)),
            escape_dot(&name_of(t))
        )?;
        attributes(&mut out, doc, &order, PropertyDomain::Edge, e.index(), None)?;
        out.write_all(b";\n")?;
    }

    out.write_all(b"}\n")?;
    out.flush().map_err(IoError::Io)
}

/// `dynamic_properties_writer::operator()` (`graphviz.hpp:524-542`) and its
/// vertex twin, which additionally skips the node-id property (`:557-570`).
fn attributes<W: Write, H: Lookup>(
    out: &mut W,
    doc: &Document<H>,
    order: &[usize],
    domain: PropertyDomain,
    at: usize,
    skip: Option<usize>,
) -> Result<(), IoError> {
    debug_assert!(at < domain_len(doc, domain));
    let mut first = true;
    for &i in order {
        let p = &doc.properties[i];
        if p.domain != domain || Some(i) == skip {
            continue;
        }
        // The index map graph-tool inserts under this name *is* the node
        // name; it is never also an attribute.
        if domain == PropertyDomain::Vertex && skip.is_none() && p.name == NODE_INDEX_KEY {
            continue;
        }
        let Some(val) = value_at(&p.values, at) else {
            continue;
        };
        out.write_all(if first { b" [" } else { b", " })?;
        first = false;
        write!(out, "{}={}", p.name, escape_dot(&val))?;
    }
    if !first {
        out.write_all(b"]")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_follows_the_xpressive_grammar() {
        for plain in ["a", "_x9", "G", "0", "-1", "1.5", ".5", "-.5", "12."] {
            assert_eq!(escape_dot(plain), plain, "{plain}");
        }
        for quoted in ["", "a b", "1a", "a-b", "hé", "--", "1.2.3"] {
            assert!(escape_dot(quoted).starts_with('"'), "{quoted}");
        }
        assert_eq!(escape_dot("say \"hi\""), "\"say \\\"hi\\\"\"");
    }

    #[test]
    fn the_lexer_reports_the_line_it_stopped_on() {
        let mut lex = Lexer::new("digraph G {\n  a -> b;\n  @\n}\n");
        let mut last = lex.next().expect("digraph");
        loop {
            match lex.next() {
                Ok(t) if t.kind == Tok::Eof => break,
                Ok(t) => last = t,
                Err(e) => panic!("{e}"),
            }
        }
        // `@` is a token in this grammar, on line 3.
        assert_eq!(last.kind, Tok::RBrace);
        assert_eq!(last.line, 4);
    }

    #[test]
    fn numbers_and_arrows_do_not_collide() {
        let mut lex = Lexer::new("-1 -> -.5 -- 2.");
        let mut kinds = Vec::new();
        loop {
            let t = lex.next().expect("lex");
            if t.kind == Tok::Eof {
                break;
            }
            kinds.push(t.kind);
        }
        assert_eq!(
            kinds,
            vec![
                Tok::Ident("-1".to_owned()),
                Tok::DashGreater,
                Tok::Ident("-.5".to_owned()),
                Tok::DashDash,
                Tok::Ident("2.".to_owned()),
            ]
        );
    }

    #[test]
    fn quoted_strings_concatenate_and_unescape() {
        let mut lex = Lexer::new("\"a\\\"b\" + \"c\" \"d\\\ne\"");
        assert_eq!(lex.next().unwrap().kind, Tok::Ident("a\"bc".to_owned()));
        assert_eq!(lex.next().unwrap().kind, Tok::Ident("de".to_owned()));
    }
}
