//! U0 — the test harness itself.
//!
//! This unit adds no code to `src/`. What it delivers is a *configuration*:
//! three dev-dependencies, seven `[[bench]]` registrations, and two profile
//! switches. Every one of those is the kind of thing that breaks silently —
//! a missing `harness = false` turns criterion's `main` into a link error with
//! an unhelpful message, a `[profile.test] debug-assertions = false` turns six
//! rows of `gt_core::design`'s defect table into dead code while the suite stays
//! green — so U0 gets tests like every other unit.
//!
//! gt_core::design section 16 is blunt about why: *"it compiles" is not evidence*.
//! The same applies to the harness. A harness that is present but misconfigured
//! is worse than an absent one, because it reports success.

use std::collections::BTreeSet;
use std::hint::black_box;
use std::path::{Path, PathBuf};

use gt_core::ids::{EdgeId, MAX_INDEX, Raw, VertexId};
use proptest::prelude::*;

// ===========================================================================
// 1. The profile switches the defect table depends on
// ===========================================================================

/// Six "eliminated" rows of gt_core::design section 14 are cashed out as checks that
/// exist **only** under `debug_assertions`:
///
/// * #46 — `check_epos` exists in graph-tool and every call site is commented
///   out (`graph_adjacency.hh:686, :1206, :1289, :1433`); the port's answer is
///   `AdjList::validate()` after every mutation.
/// * #37 — `__test__ = False` (`base_states.py:33`) leaves the delta/absolute
///   cross-check off; the port's answer is `audit_commit` on every commit.
/// * #8 — the `GraphId` compare in `sized_for`.
///
/// Every one of those is a `debug_assert`-shaped guarantee. Run the suite in a
/// profile without `debug_assertions` and it still passes — while checking
/// none of them. That is precisely graph-tool's failure mode, reproduced in
/// the port's own test run, so it is worth one assertion.
///
/// This is also why `cargo test --release` is not a supported way to run this
/// suite: it is not a faster run of the same checks, it is a different and
/// much weaker set of checks.
#[test]
fn the_test_profile_keeps_the_invariant_checks_on() {
    // Not `cfg!(debug_assertions)`: that asks the compiler what it was told,
    // where the thing actually relied upon is that a `debug_assert!` *body*
    // runs. Observing the body's side effect answers the real question, and
    // answers it the same way a `validate()` call inside one would.
    let mut body_ran = false;
    debug_assert!({
        body_ran = true;
        true
    });
    assert!(
        body_ran,
        "debug_assert! bodies do not run, so AdjList::validate, audit_commit \
         and the sized_for graph check are all compiled out: this run proves \
         nothing about gt_core::design section 14. Run `cargo test` without --release."
    );
}

/// `[profile.dev] overflow-checks = true` is justified in `gt_core::design` section 10
/// by exactly one failure: defect #1, where `clear_vertex` decrements
/// `_n_edges` by two for one removed edge because it counts `remove_if`'s
/// moved-from tail (`graph_adjacency.hh:1403-1410`). In the port the same
/// mistake is a `usize` subtraction below zero, which is a panic *only* if the
/// profile says so — and silently `usize::MAX` if it does not, after which
/// `num_edges()` is nonsense and every downstream assertion compares two wrong
/// numbers.
///
/// Cargo applies `[profile.dev]` to test targets, so this holds for
/// `cargo test`; the assertion is what keeps that true.
#[test]
fn overflow_checks_reach_the_test_profile() {
    // `black_box` on both operands: a constant-folded `0u32 - 1u32` is a
    // compile error rather than a panic, which would test nothing.
    let zero = black_box(0u32);
    let one = black_box(1u32);
    let wrapped = std::panic::catch_unwind(move || zero - one);
    assert!(
        wrapped.is_err(),
        "unsigned subtraction below zero wrapped instead of panicking: \
         overflow-checks is off in this profile, and defect #1's mechanism \
         (a `usize` count that cannot go negative) is not being enforced"
    );
}

// ===========================================================================
// 2. The manifests
// ===========================================================================

fn workspace_root() -> PathBuf {
    // `crates/gt-core` -> `graph-tool-rs`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/<crate>/ has two ancestors")
        .to_path_buf()
}

/// The crates U0 is responsible for configuring.
const CRATES: [&str; 5] = ["gt-core", "gt-algo", "gt-inference", "gt-io", "gt-py"];

/// The three dependencies `gt_core::design` section 16 makes mandatory.
const HARNESS_DEPS: [&str; 3] = ["criterion", "proptest", "trybuild"];

/// One `[[bench]]` stanza.
#[derive(Debug, Default)]
struct BenchTarget {
    name: Option<String>,
    harness: Option<bool>,
}

#[derive(Debug, Default)]
struct Manifest {
    dev_dependencies: BTreeSet<String>,
    workspace_dependencies: BTreeSet<String>,
    benches: Vec<BenchTarget>,
}

/// A deliberately small reader for the shape this test asks about.
///
/// Pulling in a TOML parser would mean a fourth dev-dependency whose own
/// correctness this test would then be assuming, in order to check three
/// keys. The manifests in this workspace are hand-written, one key per line,
/// and that is the subset parsed here: section headers, `key = ...`, comments
/// and blank lines. Anything else (inline tables spanning lines, arrays of
/// tables inside arrays) would be silently misread, so the assertions below
/// only ever ask about keys this shape can see.
fn read_manifest(path: &Path) -> Manifest {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} is not readable: {e}", path.display()));
    let mut m = Manifest::default();
    let mut section = String::new();
    let mut continuation = false;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if continuation {
            // Inside a multi-line inline table or array; it ends at the line
            // whose trailing brace closes it.
            if line.starts_with('}') || line.starts_with(']') {
                continuation = false;
            }
            continue;
        }
        if line.starts_with('[') {
            section = line.to_string();
            if section == "[[bench]]" {
                m.benches.push(BenchTarget::default());
            }
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        // An unterminated `{`/`[` opens a continuation.
        let opens = value.ends_with('{') || value.ends_with('[');
        if opens {
            continuation = true;
        }

        match section.as_str() {
            "[dev-dependencies]" => {
                m.dev_dependencies.insert(key.to_string());
            }
            "[workspace.dependencies]" => {
                m.workspace_dependencies.insert(key.to_string());
            }
            "[[bench]]" => {
                let target = m.benches.last_mut().expect("a [[bench]] header came first");
                match key {
                    "name" => target.name = Some(value.trim_matches('"').to_string()),
                    "harness" => target.harness = Some(value == "true"),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    m
}

/// Every crate must be able to run the three kinds of test the design calls
/// for. A crate missing one of them does not fail loudly; it simply never
/// grows that kind of test, which is how a "verified by compiling it" tree
/// comes about in the first place.
#[test]
fn every_crate_carries_the_three_harness_dependencies() {
    let root = workspace_root();

    let ws = read_manifest(&root.join("Cargo.toml"));
    for dep in HARNESS_DEPS {
        assert!(
            ws.workspace_dependencies.contains(dep),
            "`{dep}` is not pinned in [workspace.dependencies]; a per-crate \
             version would let two crates drift onto different releases"
        );
    }

    for krate in CRATES {
        let m = read_manifest(&root.join("crates").join(krate).join("Cargo.toml"));
        for dep in HARNESS_DEPS {
            assert!(
                m.dev_dependencies.contains(dep),
                "crates/{krate}/Cargo.toml has no `{dep}` dev-dependency"
            );
        }
    }
}

/// The failure this catches: a `benches/foo.rs` that exists and is not
/// registered, or registered without `harness = false`.
///
/// Neither is a missing file, so neither shows up as "not found". An
/// unregistered bench is simply never compiled — `cargo bench` reports success
/// having run nothing, which is the same class of silent pass as
/// `__test__ = False` (defect #37). A bench registered *with* the default
/// harness fails at link time instead, because libtest supplies a `main` and
/// so does `criterion_main!`, and the resulting duplicate-symbol error names
/// neither cause.
#[test]
fn every_bench_file_is_registered_with_harness_false() {
    let root = workspace_root();

    for krate in CRATES {
        let dir = root.join("crates").join(krate);
        let m = read_manifest(&dir.join("Cargo.toml"));

        let registered: BTreeSet<String> = m
            .benches
            .iter()
            .map(|b| {
                let name = b
                    .name
                    .clone()
                    .unwrap_or_else(|| panic!("a [[bench]] in {krate} has no name"));
                assert_eq!(
                    b.harness,
                    Some(false),
                    "[[bench]] {name} in {krate} must set `harness = false`: \
                     criterion_main! supplies its own main"
                );
                name
            })
            .collect();

        let bench_dir = dir.join("benches");
        let on_disk: BTreeSet<String> = if bench_dir.is_dir() {
            std::fs::read_dir(&bench_dir)
                .expect("benches/ is readable")
                .map(|e| e.expect("a readable directory entry").path())
                .filter(|p| p.extension().is_some_and(|x| x == "rs"))
                .map(|p| {
                    p.file_stem()
                        .expect("a .rs file has a stem")
                        .to_string_lossy()
                        .into_owned()
                })
                .collect()
        } else {
            BTreeSet::new()
        };

        assert_eq!(
            registered, on_disk,
            "crates/{krate}: the [[bench]] stanzas and benches/*.rs disagree. \
             Left is registered, right is on disk."
        );
    }
}

// ===========================================================================
// 3. The harness dependencies actually link and run
// ===========================================================================

/// trybuild is the load-bearing dependency for `gt_core::design` section 14: six of
/// its rows are claims about a *diagnostic*, and a diagnostic has no runtime
/// representation to assert on.
///
/// Coerced to a function pointer rather than called: constructing
/// `TestCases` and dropping it is a no-op only by implementation detail, and
/// the fixtures themselves belong to the units that own the guarantees.
#[test]
fn trybuild_links() {
    let _new: fn() -> trybuild::TestCases = trybuild::TestCases::new;
}

/// criterion is a dev-dependency of every crate but a *compile* dependency of
/// the bench targets only, so nothing in `cargo test` would notice it being
/// misconfigured. `Throughput` is the type every bench in the tree reports
/// through, and the variants used are `Elements` and `Bytes`.
#[test]
fn criterion_links() {
    assert!(matches!(
        criterion::Throughput::Elements(4),
        criterion::Throughput::Elements(4)
    ));
    assert!(matches!(
        criterion::Throughput::Bytes(4),
        criterion::Throughput::Bytes(4)
    ));
}

// ===========================================================================
// 4. proptest runs, against a real property
// ===========================================================================

proptest! {
    /// The point of this one is the *harness*: it proves `proptest!` expands,
    /// generates its default 256 cases and reports through libtest. The
    /// property is chosen to be genuinely true rather than trivially so:
    /// `Id` is `#[repr(transparent)]` over `Raw`, and `index()`/`raw()` are
    /// two views of the same field, so a discrepancy would mean the id had
    /// been widened or narrowed somewhere between construction and read —
    /// which is defect #5 (`_epos` is `uint32_t` beneath a `size_t` vertex,
    /// `graph_adjacency.hh:620`) in its Rust shape.
    #[test]
    fn an_id_round_trips_through_its_index(i in 0usize..=MAX_INDEX) {
        let v = VertexId::from_index(i);
        prop_assert_eq!(v.index(), i);
        prop_assert_eq!(v.raw(), i as Raw);
        prop_assert_eq!(VertexId::new(i).map(VertexId::index), Some(i));

        // The same field, a different space. `adj_edge_descriptor` makes
        // `s`, `t` and `idx` the same type (`:206`); here the two spaces do
        // not convert, and the only thing they share is the arithmetic.
        let e = EdgeId::from_index(i);
        prop_assert_eq!(e.index(), v.index());
    }

    /// One value is reserved above [`MAX_INDEX`] so a future niche stays free,
    /// and `new` is the checked constructor that knows it.
    #[test]
    fn indices_above_the_maximum_are_refused(k in 1usize..=4096) {
        // Under `wide-index`, `MAX_INDEX` is `u64::MAX - 1`, so on a 64-bit
        // target there is at most one representable index above it and the
        // property is vacuous for larger `k`. Saying so beats asserting it.
        if let Some(j) = MAX_INDEX.checked_add(k) {
            prop_assert!(VertexId::new(j).is_none());
            prop_assert!(EdgeId::new(j).is_none());
        }
    }
}
