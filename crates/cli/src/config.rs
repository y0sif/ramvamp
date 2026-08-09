//! Named profiles: the config file, and the precedence chain around it.
//!
//! The dials named here — `context`, `cache_bytes`, `threads`, `prefill`,
//! `prefill_chunk` — are the ones that decide whether a run fits in memory and
//! how fast it goes. They were already runtime values before this module
//! existed; what a profile adds is a *name* for a combination of them, so that
//! a small chat setup and a wide-context agent setup are `--profile chat` and
//! `--profile agent` rather than four flags each, remembered by hand.
//!
//! # Precedence
//!
//! Lowest to highest: **built-in default -> profile file -> environment
//! variable -> explicit CLI flag.**
//!
//! That is the rule `main.rs` already documents for its `Option` flags — a
//! flag is `Option` precisely so that an unset one does not beat
//! `RAMVAMP_CONTEXT` — with one layer inserted *underneath* the environment.
//! A file the user edited months ago must not beat a variable they exported in
//! this shell, and neither may beat what they typed on this command line.
//!
//! [`Dials::resolve`] is the whole of that rule, as a pure function over three
//! plain structs ([`Flags`], [`Env`], [`Profile`]), so the ordering is tested
//! directly rather than through `main` — no process environment, no file on
//! disk and no model.
//!
//! # The file
//!
//! ```json
//! {
//!   "version": 1,
//!   "default_profile": "chat",
//!   "profiles": {
//!     "chat":  { "context": 4096,  "cache_bytes": "1440M" },
//!     "agent": { "context": 16384, "cache_bytes": "1440M" }
//!   }
//! }
//! ```
//!
//! Read from `--config <PATH>` if given, else `$XDG_CONFIG_HOME/ramvamp/
//! config.json`, else `~/.config/ramvamp/config.json`.
//!
//! **A missing file is not an error.** It means the built-in defaults, which
//! are the measured, published 4K/11-slot configuration; nothing here ships a
//! file, writes one, or creates the directory. A file that exists and does not
//! parse *is* an error naming the path, and so is an unrecognized key:
//! silently ignoring a configuration the user wrote is the failure this
//! project refuses everywhere else, and `"contxt": 16384` would otherwise widen
//! nothing and say nothing. A `--config` that names a file which is not there
//! is an error for the same reason — the user typed that path.
//!
//! Every value is checked against exactly the range its flag is checked
//! against, and `cache_bytes` goes through [`crate::parse_bytes`], the parser
//! behind `--cache-bytes`, so `"1440M"` in the file means the byte count
//! `--cache-bytes 1440M` means. There is no second parser to drift.
//!
//! # Why the parse is hand-written
//!
//! `serde`'s derive would say this in a third of the lines and get the
//! strictness for free from `#[serde(deny_unknown_fields)]` — the pattern
//! `crates/repack/src/install/state.rs` follows. It is not used here because
//! this crate depends on `serde_json` but not on `serde`, and the whole
//! feature is meant to live inside `crates/cli/src` without adding a
//! dependency edge. What is given up is the derive. What is not given up is
//! the behaviour: unknown keys are refused at the root *and* inside every
//! profile, with the recognized ones listed, and
//! `an_unknown_key_is_refused_at_every_level` pins it.

use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, bail};
use clap::Args;
use ramvamp_core::format::{ExpertsLayout, Manifest};
use ramvamp_core::io::{Footprint, FootprintError};
use ramvamp_core::model::{
    DEFAULT_CACHE_BYTES, ForwardState, PrefillConfig, PrefillMode, RuntimeConfig,
};
use serde_json::Value;

use crate::repl::DEFAULT_CONTEXT;
use crate::{
    ContextArgs, MAX_THREADS, PrefillArgs, RuntimeArgs, parse_bytes, parse_context_var,
    parse_prefill_mode,
};

/// Schema version this build reads. A file that declares anything else is
/// refused rather than read optimistically, exactly as `install-state.json`
/// treats its own version.
pub const CONFIG_VERSION: u64 = 1;

/// Parse cap for the config file. A hand-written file is a few hundred bytes;
/// the cap is what stops a hostile or accidental path (a core dump, a video)
/// from being read into memory before it is rejected.
const MAX_CONFIG_BYTES: u64 = 256 * 1024;

/// Directory under `$XDG_CONFIG_HOME` (or `~/.config`) the file lives in.
const CONFIG_DIR: &str = "ramvamp";

/// The file's name inside [`CONFIG_DIR`].
const CONFIG_FILE: &str = "config.json";

/// Keys the root object may carry, in the order a message lists them.
const ROOT_KEYS: [&str; 3] = ["default_profile", "profiles", "version"];

/// Keys one profile object may carry. Each one is a dial with a flag, and the
/// spelling is the flag's with `-` written `_`.
const PROFILE_KEYS: [&str; 5] = [
    "cache_bytes",
    "context",
    "prefill",
    "prefill_chunk",
    "threads",
];

// ---------------------------------------------------------------------------
// the flags that pick a profile
// ---------------------------------------------------------------------------

/// `--profile` and `--config`, flattened into every command that builds a
/// [`ForwardState`] and into `plan`.
///
/// Deliberately *not* on `tokenize`: it runs no model and reads no dial, so
/// accepting a profile there would be a flag that does nothing.
#[derive(Args, Debug, Clone, Default)]
pub struct ProfileArgs {
    /// Named profile from the config file. Unset: the file's
    /// `default_profile`, or the built-in defaults when there is neither.
    /// A name the file does not define is an error listing the ones it does.
    #[arg(long, value_name = "NAME")]
    pub profile: Option<String>,

    /// Config file to read instead of the search path
    /// (`$XDG_CONFIG_HOME/ramvamp/config.json`, else
    /// `~/.config/ramvamp/config.json`). A file named here has to exist; one
    /// merely searched for does not.
    #[arg(long, value_name = "PATH", conflicts_with = "no_config")]
    pub config: Option<PathBuf>,

    /// Ignore any config file and use the built-in defaults.
    ///
    /// For measurement. Every published number in `docs/experiments/` assumes
    /// a known configuration, and a config file in the operator's home would
    /// otherwise change what a benchmark measures without appearing anywhere
    /// in the entry that records it. `scripts/cold_bench.py` passes this so a
    /// cold run means the same thing on a machine that has profiles as on one
    /// that does not.
    ///
    /// Env vars and explicit flags still apply: this removes one layer of the
    /// precedence chain, it does not pin the whole configuration.
    #[arg(long)]
    pub no_config: bool,
}

// ---------------------------------------------------------------------------
// the file
// ---------------------------------------------------------------------------

/// One named profile. Every dial is optional and an absent one falls through
/// to the layer below (see [`Dials::resolve`]), so a profile that sets only
/// `context` changes only `context`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Profile {
    /// `context`: the KV-cache window in tokens.
    pub context: Option<usize>,
    /// `cache_bytes`: the expert-cache byte budget, parsed by
    /// [`crate::parse_bytes`] so the suffixes match `--cache-bytes`.
    pub cache_bytes: Option<u64>,
    /// `threads`: compute shards, counting the decode thread.
    pub threads: Option<usize>,
    /// `prefill`: `sweep` or `token-major`.
    pub prefill: Option<PrefillMode>,
    /// `prefill_chunk`: positions carried through the model together.
    pub prefill_chunk: Option<usize>,
}

/// A parsed config file, or the empty stand-in for not having one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Config {
    /// The file actually read. `None` when there is no file, which is the
    /// default state of a fresh install.
    path: Option<PathBuf>,
    /// Where a file was looked for, whether or not one was found. Carried so
    /// "there is no profile called that" can say where it looked.
    searched: Option<PathBuf>,
    /// `default_profile`, already checked to name a profile that exists.
    default_profile: Option<String>,
    /// The profiles, by name. A `BTreeMap` so every message that lists them
    /// lists them in the same order.
    profiles: BTreeMap<String, Profile>,
}

impl Config {
    /// Load the config file: `explicit` if given, else the search path, else
    /// nothing at all.
    ///
    /// # Errors
    ///
    /// A file that cannot be read, is larger than [`MAX_CONFIG_BYTES`], is not
    /// JSON, is not the shape this build reads, or carries a key this build
    /// does not recognize. A *searched* path that does not exist is not an
    /// error; an `explicit` one that does not exist is.
    /// No config at all, as if the search found nothing.
    ///
    /// What `--no-config` resolves to. Distinct from a config that happens to
    /// define no profiles: this one was never looked for, so a malformed file
    /// on disk cannot fail a run that asked to ignore it.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn load(explicit: Option<&Path>) -> anyhow::Result<Self> {
        let (path, required) = match explicit {
            Some(path) => (Some(path.to_path_buf()), true),
            None => (default_config_path(), false),
        };
        let Some(path) = path else {
            // No `--config`, no `$XDG_CONFIG_HOME`, no `$HOME`: there is
            // nowhere to look, which is the same as finding nothing.
            return Ok(Self::default());
        };
        match read_capped(&path)? {
            Some(bytes) => Self::parse(&path, &bytes),
            None if required => bail!(
                "--config {}: no such file. A config file is optional, but one \
                 named on the command line has to exist.",
                path.display()
            ),
            None => Ok(Self {
                path: None,
                searched: Some(path),
                ..Self::default()
            }),
        }
    }

    /// Parse the bytes of a config file. Separate from reading it so the
    /// schema can be tested without a filesystem; `path` is only ever used to
    /// name the file in a message.
    pub fn parse(path: &Path, bytes: &[u8]) -> anyhow::Result<Self> {
        let root: Value = serde_json::from_slice(bytes)
            .with_context(|| format!("parsing the config file {}", path.display()))?;
        let root = object(&root, path, "the config file")?;
        for key in root.keys() {
            reject_unknown(path, key, &ROOT_KEYS, "the config file")?;
        }

        let version = root
            .get("version")
            .with_context(|| {
                format!(
                    "{}: no \"version\". This build reads version {CONFIG_VERSION}.",
                    path.display()
                )
            })?
            .as_u64()
            .with_context(|| {
                format!(
                    "{}: \"version\" must be a whole number, and this build reads \
                     {CONFIG_VERSION}",
                    path.display()
                )
            })?;
        if version != CONFIG_VERSION {
            bail!(
                "{}: config version {version} (this build reads {CONFIG_VERSION})",
                path.display()
            );
        }

        let mut profiles = BTreeMap::new();
        if let Some(value) = root.get("profiles") {
            let listed = object(value, path, "\"profiles\"")?;
            for (name, value) in listed {
                if name.trim().is_empty() {
                    bail!("{}: a profile name cannot be empty", path.display());
                }
                profiles.insert(name.clone(), parse_profile(path, name, value)?);
            }
        }

        let default_profile = match root.get("default_profile") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let name = value.as_str().with_context(|| {
                    format!("{}: \"default_profile\" must be a string", path.display())
                })?;
                if !profiles.contains_key(name) {
                    bail!(
                        "{}: default_profile {name:?} is not one of the profiles it defines{}",
                        path.display(),
                        list_names(&profiles),
                    );
                }
                Some(name.to_owned())
            }
        };

        Ok(Self {
            path: Some(path.to_path_buf()),
            searched: Some(path.to_path_buf()),
            default_profile,
            profiles,
        })
    }

    /// The profile this run uses: `wanted` if given, else `default_profile`,
    /// else the built-in defaults.
    ///
    /// # Errors
    ///
    /// A `wanted` name the file does not define. The message lists the names
    /// that do exist, because that is the question a wrong one raises.
    pub fn select(&self, wanted: Option<&str>) -> anyhow::Result<Selection> {
        let Some(name) = wanted.or(self.default_profile.as_deref()) else {
            return Ok(Selection {
                name: None,
                file: self.path.clone(),
                searched: self.searched.clone(),
                profile: Profile::default(),
            });
        };
        let Some(profile) = self.profiles.get(name) else {
            // `default_profile` is checked at parse time, so this is always
            // the user's `--profile`. It is still written as a lookup rather
            // than an unwrap, because a message beats a panic either way.
            match &self.path {
                Some(path) => bail!(
                    "no profile called {name:?} in {}{}",
                    path.display(),
                    list_names(&self.profiles),
                ),
                None => bail!(
                    "no profile called {name:?}: there is no config file{}. \
                     Profiles live in a JSON file with a \"profiles\" object; \
                     see --help.",
                    match &self.searched {
                        Some(path) => format!(" at {}", path.display()),
                        None => String::new(),
                    }
                ),
            }
        };
        Ok(Selection {
            name: Some(name.to_owned()),
            file: self.path.clone(),
            searched: self.searched.clone(),
            profile: *profile,
        })
    }
}

/// The profile a run selected, and where it came from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Selection {
    /// The profile's name, or `None` for the built-in defaults.
    pub name: Option<String>,
    /// The config file it was read from, if there was one.
    pub file: Option<PathBuf>,
    /// Where a config file was looked for.
    pub searched: Option<PathBuf>,
    /// The dials it sets.
    pub profile: Profile,
}

/// Read at most [`MAX_CONFIG_BYTES`] from `path`. `Ok(None)` means the file is
/// not there, which is the one I/O outcome that is not a failure.
fn read_capped(path: &Path) -> anyhow::Result<Option<Vec<u8>>> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        // Everything else — a permission denial, a symlink loop (ELOOP), a
        // path that is not a directory — is reported with the path, never
        // treated as "no config".
        Err(e) => {
            return Err(e).with_context(|| format!("opening the config file {}", path.display()));
        }
    };
    let mut bytes = Vec::new();
    // A directory at this path opens on Linux and fails here (EISDIR), which
    // is the message the user should see.
    file.take(MAX_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading the config file {}", path.display()))?;
    if bytes.len() as u64 > MAX_CONFIG_BYTES {
        bail!(
            "{} is larger than the {MAX_CONFIG_BYTES}-byte cap on a config file",
            path.display()
        );
    }
    Ok(Some(bytes))
}

/// `$XDG_CONFIG_HOME/ramvamp/config.json`, else `~/.config/ramvamp/config.json`,
/// else nowhere.
///
/// `XDG_CONFIG_HOME` is honoured only when it is absolute, which is what the
/// specification requires of it; a relative value would otherwise make the
/// config path depend on the working directory.
fn default_config_path() -> Option<PathBuf> {
    let base = match std::env::var_os("XDG_CONFIG_HOME") {
        Some(value) if !value.is_empty() && Path::new(&value).is_absolute() => PathBuf::from(value),
        _ => PathBuf::from(std::env::var_os("HOME")?).join(".config"),
    };
    Some(base.join(CONFIG_DIR).join(CONFIG_FILE))
}

/// `value` as a JSON object, or a message naming what was expected where.
fn object<'a>(
    value: &'a Value,
    path: &Path,
    what: &str,
) -> anyhow::Result<&'a serde_json::Map<String, Value>> {
    value.as_object().with_context(|| {
        format!(
            "{}: {what} must be a JSON object, found {}",
            path.display(),
            kind_of(value)
        )
    })
}

/// Refuse a key this build does not recognize, listing the ones it does.
///
/// This is what `#[serde(deny_unknown_fields)]` buys, written out: a config
/// whose `"contxt"` is silently ignored is a user wondering why their context
/// never widened.
fn reject_unknown(path: &Path, key: &str, known: &[&str], what: &str) -> anyhow::Result<()> {
    if known.contains(&key) {
        return Ok(());
    }
    bail!(
        "{}: unknown key {key:?} in {what}; this build reads {}",
        path.display(),
        known.join(", "),
    )
}

/// Parse one profile object. Every key is optional; every value is checked
/// against the range its flag is checked against.
fn parse_profile(path: &Path, name: &str, value: &Value) -> anyhow::Result<Profile> {
    let fields = object(value, path, &format!("profile {name:?}"))?;
    let mut profile = Profile::default();
    for (key, value) in fields {
        reject_unknown(path, key, &PROFILE_KEYS, &format!("profile {name:?}"))?;
        // A `null` is "say nothing about this dial", which is what leaving the
        // key out means; anything else is checked.
        if value.is_null() {
            continue;
        }
        let at = || format!("{}: profile {name:?}: \"{key}\"", path.display());
        match key.as_str() {
            "context" => {
                profile.context = Some(positive(value, &at(), "a context window in tokens")?);
            }
            "prefill_chunk" => {
                profile.prefill_chunk = Some(positive(value, &at(), "a chunk size in positions")?);
            }
            "threads" => {
                let threads = positive(value, &at(), "a thread count")?;
                if threads as u64 > MAX_THREADS {
                    bail!("{}: {threads} exceeds the maximum of {MAX_THREADS}", at());
                }
                profile.threads = Some(threads);
            }
            "cache_bytes" => {
                // The flag's own parser, so `"1440M"` here is the byte count
                // `--cache-bytes 1440M` is. A JSON number goes through it too,
                // which is how `0` and `-1` get the flag's exact refusals.
                let text = match value {
                    Value::String(text) => text.clone(),
                    Value::Number(number) => number.to_string(),
                    other => bail!(
                        "{}: expected a byte budget like \"1440M\" or a whole number of \
                         bytes, found {}",
                        at(),
                        kind_of(other)
                    ),
                };
                profile.cache_bytes =
                    Some(parse_bytes(&text).map_err(|e| anyhow::anyhow!("{}: {e}", at()))?);
            }
            "prefill" => {
                let text = value
                    .as_str()
                    .with_context(|| format!("{}: expected \"sweep\" or \"token-major\"", at()))?;
                profile.prefill =
                    Some(parse_prefill_mode(text).map_err(|e| anyhow::anyhow!("{}: {e}", at()))?);
            }
            // Unreachable: `reject_unknown` accepts exactly the keys these
            // arms handle. Written as a refusal rather than an `unreachable!`
            // so that adding a key to `PROFILE_KEYS` and forgetting its arm is
            // a message about this build, not a panic on somebody's config.
            other => bail!(
                "{}: this build recognizes {other:?} but does not read it",
                path.display()
            ),
        }
    }
    Ok(profile)
}

/// A JSON value as a positive `usize`, or a message saying what was wanted.
///
/// Everything a hostile file can put here lands in the same refusal: a
/// negative number, a float, a string, `0`, and a value past `usize`.
fn positive(value: &Value, at: &str, want: &str) -> anyhow::Result<usize> {
    let number = value.as_u64().filter(|&n| n > 0).with_context(|| {
        format!(
            "{at}: expected {want} (a whole number above zero), found {}",
            kind_of(value)
        )
    })?;
    usize::try_from(number)
        .map_err(|_| anyhow::anyhow!("{at}: {number} is too large for this machine"))
}

/// A value's JSON type, plus the value itself when it is small enough to be
/// worth quoting back.
fn kind_of(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(b) => format!("the boolean {b}"),
        Value::Number(n) => format!("the number {n}"),
        Value::String(s) if s.chars().count() <= 32 => format!("the string {s:?}"),
        Value::String(_) => "a string".to_owned(),
        Value::Array(_) => "an array".to_owned(),
        Value::Object(_) => "an object".to_owned(),
    }
}

/// `; it defines: agent, chat`, or a note that it defines none.
fn list_names(profiles: &BTreeMap<String, Profile>) -> String {
    if profiles.is_empty() {
        return "; it defines no profiles".to_owned();
    }
    format!(
        "; it defines: {}",
        profiles.keys().cloned().collect::<Vec<_>>().join(", ")
    )
}

// ---------------------------------------------------------------------------
// precedence
// ---------------------------------------------------------------------------

/// Which layer of the chain a dial's value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The built-in default: what a machine with no configuration gets.
    Default,
    /// A profile in the config file.
    Profile,
    /// An environment variable.
    Env,
    /// A flag on this command line.
    Flag,
}

impl Source {
    /// The word `plan` prints in the source column.
    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Profile => "profile",
            Self::Env => "env",
            Self::Flag => "flag",
        }
    }
}

/// One dial's resolved value and the layer that won it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dial<T> {
    /// The value the run uses.
    pub value: T,
    /// Where it came from.
    pub source: Source,
}

/// **The precedence rule, and the only place it is written.**
///
/// Flag beats environment beats file beats built-in default. Every dial goes
/// through this, so none of them can drift into its own ordering.
fn pick<T>(flag: Option<T>, env: Option<T>, file: Option<T>, default: T) -> Dial<T> {
    match (flag, env, file) {
        (Some(value), _, _) => Dial {
            value,
            source: Source::Flag,
        },
        (None, Some(value), _) => Dial {
            value,
            source: Source::Env,
        },
        (None, None, Some(value)) => Dial {
            value,
            source: Source::Profile,
        },
        (None, None, None) => Dial {
            value: default,
            source: Source::Default,
        },
    }
}

/// What this command line asked for. Every field is `Option` for the reason
/// `main.rs` documents on [`ContextArgs`]: an unset flag must not beat a
/// variable, and now must not beat a profile either.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Flags {
    /// `--context`.
    pub context: Option<usize>,
    /// `--cache-bytes`.
    pub cache_bytes: Option<u64>,
    /// `--threads`.
    pub threads: Option<usize>,
    /// `--prefill`.
    pub prefill: Option<PrefillMode>,
    /// `--prefill-chunk`.
    pub prefill_chunk: Option<usize>,
}

impl Flags {
    /// The dials the parsed argument structs carry, gathered into one value.
    pub fn from_args(context: ContextArgs, prefill: PrefillArgs, runtime: &RuntimeArgs) -> Self {
        Self {
            context: context.context,
            cache_bytes: runtime.cache_bytes,
            threads: runtime.threads,
            prefill: prefill.prefill,
            prefill_chunk: prefill.prefill_chunk,
        }
    }
}

/// What the environment asked for.
///
/// Only three dials have a variable today. `threads` and `cache_bytes` have
/// none, so their `Env` entries do not exist rather than being permanently
/// `None` — a variable that does not exist cannot be resolved from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Env {
    /// `RAMVAMP_CONTEXT`.
    pub context: Option<usize>,
    /// `RAMVAMP_PREFILL`.
    pub prefill: Option<PrefillMode>,
    /// `RAMVAMP_PREFILL_CHUNK`.
    pub prefill_chunk: Option<usize>,
}

impl Env {
    /// Read the three variables.
    ///
    /// An unusable value is ignored rather than fatal, which is what the
    /// runtime already does with all three: a variable a shell profile set
    /// years ago must not be able to stop the binary from starting.
    /// `RAMVAMP_CONTEXT` says so on stderr through [`parse_context_var`]; the
    /// two prefill variables are re-read by
    /// [`PrefillConfig::from_env`](ramvamp_core::model::PrefillConfig::from_env)
    /// when the state is built, which is where their warning comes from, so
    /// warning again here would print it twice.
    pub fn read() -> Self {
        Self {
            context: parse_context_var(std::env::var("RAMVAMP_CONTEXT").ok()),
            prefill: std::env::var("RAMVAMP_PREFILL")
                .ok()
                .and_then(|raw| parse_prefill_mode(&raw).ok()),
            prefill_chunk: std::env::var("RAMVAMP_PREFILL_CHUNK")
                .ok()
                .and_then(|raw| raw.trim().parse::<usize>().ok())
                .filter(|&chunk| chunk > 0),
        }
    }
}

/// Every dial this run will use, with the layer each one came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dials {
    /// The profile these were resolved against, for the report.
    pub selection: Selection,
    /// Context window in tokens.
    pub context: Dial<usize>,
    /// Expert-cache byte budget.
    pub cache_bytes: Dial<u64>,
    /// Compute shards; `None` is the runtime's own topology detection.
    pub threads: Dial<Option<usize>>,
    /// Prefill path.
    pub prefill: Dial<PrefillMode>,
    /// Prefill chunk in positions.
    pub prefill_chunk: Dial<usize>,
}

impl Dials {
    /// Apply the precedence chain. A pure function: everything it reads is an
    /// argument, which is what lets the ordering be tested without a process
    /// environment or a file.
    pub fn resolve(selection: Selection, flags: &Flags, env: &Env) -> Self {
        let file = selection.profile;
        let defaults = PrefillConfig::default();
        Self {
            context: pick(flags.context, env.context, file.context, DEFAULT_CONTEXT),
            cache_bytes: pick(
                flags.cache_bytes,
                None,
                file.cache_bytes,
                DEFAULT_CACHE_BYTES,
            ),
            // The built-in default for `threads` is "let the pool detect the
            // topology", which is `None` — so the value here is itself an
            // `Option` and the layers are `Option<Option<usize>>`.
            threads: pick(flags.threads.map(Some), None, file.threads.map(Some), None),
            prefill: pick(flags.prefill, env.prefill, file.prefill, defaults.mode),
            prefill_chunk: pick(
                flags.prefill_chunk,
                env.prefill_chunk,
                file.prefill_chunk,
                defaults.chunk,
            ),
            selection,
        }
    }

    /// The decode-time runtime dials these resolve to.
    pub fn runtime_config(&self) -> RuntimeConfig {
        RuntimeConfig {
            cache_bytes: self.cache_bytes.value,
            threads: self.threads.value,
            pin: true,
        }
    }

    /// `base` with the two resolved prefill dials applied.
    ///
    /// The sweep's own sub-dials (`experts_per_window`, `windows_in_flight`)
    /// come from `base` untouched: they are not part of this chain, and the
    /// state seeds them itself.
    pub fn prefill_config(&self, base: PrefillConfig) -> PrefillConfig {
        PrefillConfig {
            mode: self.prefill.value,
            chunk: self.prefill_chunk.value,
            ..base
        }
    }

    /// Apply the prefill dials to a freshly built state.
    pub fn apply_prefill(&self, state: &mut ForwardState) -> anyhow::Result<()> {
        let config = self.prefill_config(state.prefill_config());
        state.set_prefill_config(config)?;
        Ok(())
    }

    /// What this configuration will hold resident, from the same predictor
    /// `ForwardState::with_config` prices itself with.
    ///
    /// The arithmetic is [`Footprint::project`]'s and only ever
    /// [`Footprint::project`]'s; what this adds is that it is asked about the
    /// *resolved* dials, so `plan` cannot answer for a configuration the run
    /// would not have used.
    pub fn project(
        &self,
        manifest: &Manifest,
        layout: &ExpertsLayout,
    ) -> Result<Footprint, FootprintError> {
        Footprint::project(manifest, layout, self.cache_bytes.value, self.context.value)
    }
}

/// Load the config file, select the profile, and resolve every dial: the one
/// entry point every command uses.
///
/// # Errors
///
/// A config file that exists and is not readable as one, or a `--profile` name
/// it does not define.
pub fn resolve_dials(
    profile: &ProfileArgs,
    context: ContextArgs,
    prefill: PrefillArgs,
    runtime: &RuntimeArgs,
) -> anyhow::Result<Dials> {
    // `--no-config` removes the file layer entirely rather than loading it and
    // ignoring what it says: a config that cannot be read cannot be
    // half-applied, and a malformed file should not fail a run that asked not
    // to use one. `--profile` alongside it is a contradiction worth naming.
    let config = if profile.no_config {
        if let Some(name) = profile.profile.as_deref() {
            anyhow::bail!(
                "--profile {name:?} asks for a profile and --no-config asks for none; \
                 drop one of them"
            );
        }
        Config::none()
    } else {
        Config::load(profile.config.as_deref())?
    };
    let selection = config.select(profile.profile.as_deref())?;
    let flags = Flags::from_args(context, prefill, runtime);
    Ok(Dials::resolve(selection, &flags, &Env::read()))
}

// ---------------------------------------------------------------------------
// the report `plan` prints
// ---------------------------------------------------------------------------

/// One mebibyte.
const MIB: u128 = 1 << 20;

/// `bytes` as whole mebibytes, rounded to nearest, grouped with `,` — the way
/// `docs/architecture.md` writes every figure in the memory contract, and the
/// way [`Footprint`]'s own `Display` writes its terms.
///
/// A copy of `io::slots::mib`, which is `pub(crate)` to that crate. Six lines
/// of presentation; the arithmetic that must not be duplicated is the
/// projection's, and that is [`Footprint::project`]'s alone.
fn mib(bytes: u128) -> String {
    let whole = (bytes + MIB / 2) / MIB;
    let digits = whole.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.char_indices() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// The exact text `plan` prints, as lines.
///
/// Separate from the printing so a test can read it, exactly as
/// `stream_stats_lines` and `prefill_timing_lines` are in `main.rs`.
///
/// The projected line is [`Footprint`]'s own `Display`, verbatim: printing it
/// any other way would mean a second copy of the tenant breakdown, and the two
/// would drift the first time a tenant changed.
pub fn plan_lines(
    dials: &Dials,
    model: &Path,
    footprint: &Footprint,
    ceiling: Option<u64>,
) -> Vec<String> {
    let selection = &dials.selection;
    let mut lines = vec![
        match (&selection.name, &selection.file, &selection.searched) {
            (Some(name), Some(file), _) => format!("profile: {name} (from {})", file.display()),
            // A name with no file cannot be selected, so this is unreachable
            // through `Config::select`; written out rather than unwrapped.
            (Some(name), None, _) => format!("profile: {name}"),
            (None, Some(file), _) => format!(
                "profile: none (built-in defaults; {} sets no default_profile)",
                file.display()
            ),
            (None, None, Some(searched)) => format!(
                "profile: none (built-in defaults; no config file at {})",
                searched.display()
            ),
            (None, None, None) => "profile: none (built-in defaults; no config path)".to_owned(),
        },
    ];

    let threads = match dials.threads.value {
        Some(threads) => threads.to_string(),
        None => "auto".to_owned(),
    };
    let prefill = match dials.prefill.value {
        PrefillMode::Sweep => "sweep",
        PrefillMode::TokenMajor => "token-major",
    };
    for (name, value, source) in [
        (
            "context",
            dials.context.value.to_string(),
            dials.context.source,
        ),
        (
            "cache-bytes",
            format!("{} MiB", mib(u128::from(dials.cache_bytes.value))),
            dials.cache_bytes.source,
        ),
        ("threads", threads, dials.threads.source),
        ("prefill", prefill.to_owned(), dials.prefill.source),
        (
            "prefill-chunk",
            dials.prefill_chunk.value.to_string(),
            dials.prefill_chunk.source,
        ),
    ] {
        lines.push(format!("  {name:<13}  {value:<12}  {}", source.label()));
    }

    lines.push(String::new());
    lines.push(format!("model:     {}", model.display()));
    lines.push(format!("projected: {footprint}"));
    match ceiling {
        Some(limit) => {
            lines.push(format!("available: {} MiB", mib(u128::from(limit))));
            let total = footprint.total();
            lines.push(if footprint.fits_within(limit) {
                format!(
                    "verdict:   fits, {} MiB spare",
                    mib(u128::from(limit) - total)
                )
            } else {
                format!(
                    "verdict:   does not fit, {} MiB over",
                    mib(total - u128::from(limit))
                )
            });
        }
        None => {
            // `resident_ceiling` returns `None` only when the kernel will not
            // say, which is not a verdict of "fits".
            lines.push("available: unknown (the kernel would not say)".to_owned());
            lines.push("verdict:   cannot be judged without a memory limit".to_owned());
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ramvamp_core::format::{
        ArchInfo, ExpertsLayout, FileEntry, LayerLayout, Manifest, QuantInfo, RVMP_VERSION,
        SourceInfo,
    };

    use super::*;

    /// A path that names a config file without one existing, for the messages
    /// that only ever print it.
    fn at(name: &str) -> PathBuf {
        PathBuf::from(format!("/nowhere/{name}/config.json"))
    }

    fn parse(json: &str) -> anyhow::Result<Config> {
        Config::parse(&at("t"), json.as_bytes())
    }

    /// A file with the two profiles the module docs show.
    const SAMPLE: &str = r#"{
        "version": 1,
        "default_profile": "chat",
        "profiles": {
            "chat":  { "context": 4096,  "cache_bytes": "1440M" },
            "agent": { "context": 16384, "cache_bytes": "1440M", "threads": 4,
                       "prefill": "token-major", "prefill_chunk": 256 }
        }
    }"#;

    // -----------------------------------------------------------------
    // precedence
    // -----------------------------------------------------------------

    /// **The rule this module exists for**, tested where it lives rather than
    /// through `main`: for a dial with all four layers present the flag wins;
    /// drop the flag and the environment wins; drop that and the file wins;
    /// drop that and the built-in wins.
    #[test]
    fn a_flag_beats_the_environment_beats_the_file_beats_the_built_in() {
        let profile = Profile {
            context: Some(16_384),
            cache_bytes: Some(512 << 20),
            threads: Some(4),
            prefill: Some(PrefillMode::TokenMajor),
            prefill_chunk: Some(256),
        };
        let selection = Selection {
            name: Some("agent".to_owned()),
            file: Some(at("agent")),
            searched: Some(at("agent")),
            profile,
        };
        let flags = Flags {
            context: Some(1024),
            cache_bytes: Some(64 << 20),
            threads: Some(3),
            prefill: Some(PrefillMode::Sweep),
            prefill_chunk: Some(8),
        };
        let env = Env {
            context: Some(2048),
            prefill: Some(PrefillMode::Sweep),
            prefill_chunk: Some(64),
        };

        // All four layers: the command line wins every dial it names.
        let all = Dials::resolve(selection.clone(), &flags, &env);
        assert_eq!(all.context.value, 1024);
        assert_eq!(all.context.source, Source::Flag);
        assert_eq!(all.cache_bytes.value, 64 << 20);
        assert_eq!(all.cache_bytes.source, Source::Flag);
        assert_eq!(all.threads.value, Some(3));
        assert_eq!(all.threads.source, Source::Flag);
        assert_eq!(all.prefill.value, PrefillMode::Sweep);
        assert_eq!(all.prefill.source, Source::Flag);
        assert_eq!(all.prefill_chunk.value, 8);
        assert_eq!(all.prefill_chunk.source, Source::Flag);

        // No flags: the environment wins the three dials it has variables for,
        // and the file keeps the two it does not.
        let no_flags = Dials::resolve(selection.clone(), &Flags::default(), &env);
        assert_eq!(no_flags.context.value, 2048);
        assert_eq!(no_flags.context.source, Source::Env);
        assert_eq!(no_flags.prefill.value, PrefillMode::Sweep);
        assert_eq!(no_flags.prefill.source, Source::Env);
        assert_eq!(no_flags.prefill_chunk.value, 64);
        assert_eq!(no_flags.prefill_chunk.source, Source::Env);
        assert_eq!(no_flags.cache_bytes.value, 512 << 20);
        assert_eq!(no_flags.cache_bytes.source, Source::Profile);
        assert_eq!(no_flags.threads.value, Some(4));
        assert_eq!(no_flags.threads.source, Source::Profile);

        // Neither: the file wins everything it sets.
        let file_only = Dials::resolve(selection, &Flags::default(), &Env::default());
        assert_eq!(file_only.context.value, 16_384);
        assert_eq!(file_only.context.source, Source::Profile);
        assert_eq!(file_only.prefill.value, PrefillMode::TokenMajor);
        assert_eq!(file_only.prefill.source, Source::Profile);
        assert_eq!(file_only.prefill_chunk.value, 256);
        assert_eq!(file_only.prefill_chunk.source, Source::Profile);

        // And with nothing at all, the built-in defaults — which are the
        // measured, published configuration, unchanged by this feature.
        let bare = Dials::resolve(Selection::default(), &Flags::default(), &Env::default());
        assert_eq!(bare.context.value, DEFAULT_CONTEXT);
        assert_eq!(bare.cache_bytes.value, DEFAULT_CACHE_BYTES);
        assert_eq!(bare.threads.value, None);
        assert_eq!(bare.prefill.value, PrefillMode::Sweep);
        assert_eq!(bare.prefill_chunk.value, PrefillConfig::default().chunk);
        for source in [
            bare.context.source,
            bare.cache_bytes.source,
            bare.threads.source,
            bare.prefill.source,
            bare.prefill_chunk.source,
        ] {
            assert_eq!(source, Source::Default);
        }
        // The default path is what `RuntimeConfig::default()` describes, term
        // for term: no profile can be what a bare command line gets.
        assert_eq!(bare.runtime_config(), RuntimeConfig::default());
        assert_eq!(
            bare.prefill_config(PrefillConfig::default()),
            PrefillConfig::default()
        );
    }

    /// A profile that sets one dial changes one dial. The rest fall through to
    /// the layer below, which is what makes a partial profile safe to write.
    #[test]
    fn an_absent_key_falls_through_to_the_layer_below() {
        let config = parse(SAMPLE).expect("the sample parses");
        let selection = config.select(Some("chat")).unwrap();
        let dials = Dials::resolve(selection, &Flags::default(), &Env::default());
        assert_eq!(dials.context.value, 4096);
        assert_eq!(dials.context.source, Source::Profile);
        assert_eq!(dials.threads.value, None);
        assert_eq!(dials.threads.source, Source::Default);
        assert_eq!(dials.prefill_chunk.source, Source::Default);
    }

    // -----------------------------------------------------------------
    // the file
    // -----------------------------------------------------------------

    /// A machine with no config file is the default machine, and it must not
    /// be an error there. A `--config` the user typed is the other case: that
    /// file has to be there.
    #[test]
    fn a_missing_config_file_is_not_an_error_unless_it_was_named() {
        let missing = std::env::temp_dir().join(format!(
            "ramvamp-no-such-config-{}.json",
            std::process::id()
        ));
        assert!(!missing.exists());

        // The mechanism, hermetically: a path that is searched and not found
        // reads as "no config", not as a failure. `Config::load(None)` itself
        // is deliberately not called here — it would read whatever the
        // developer running the suite happens to have in `~/.config`.
        assert!(read_capped(&missing).unwrap().is_none());

        // ... and a `Config` that found nothing still answers "no profile"
        // rather than refusing to run.
        let searched = Config {
            path: None,
            searched: Some(missing.clone()),
            ..Config::default()
        };
        let selection = searched.select(None).expect("no profile is a fine answer");
        assert_eq!(selection.profile, Profile::default());
        assert_eq!(selection.name, None);

        let err = Config::load(Some(&missing)).unwrap_err().to_string();
        assert!(err.contains("no such file"), "{err}");
        assert!(err.contains(&missing.display().to_string()), "{err}");
    }

    /// A file that exists and does not parse is refused, and the message names
    /// the path — a config that was ignored silently is the failure this whole
    /// module is against.
    #[test]
    fn a_malformed_config_is_refused_and_names_the_path() {
        let path = at("broken");
        for bad in [
            "",
            "{",
            "not json at all",
            "[1, 2, 3]",
            "{\"version\": \"one\"}",
            "{\"version\": 2}",
            "{}",
            "{\"version\": 1, \"profiles\": 7}",
            "{\"version\": 1, \"profiles\": {\"chat\": 4096}}",
            "{\"version\": 1, \"default_profile\": \"nope\", \"profiles\": {}}",
            "{\"version\": 1, \"default_profile\": 7, \"profiles\": {}}",
        ] {
            let err = Config::parse(&path, bad.as_bytes())
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("/nowhere/broken/config.json"),
                "{bad:?} reported as {err:?} without the path"
            );
        }
        // The version mismatch says which version this build reads, since that
        // is the only actionable part of it.
        let err = Config::parse(&path, b"{\"version\": 2}")
            .unwrap_err()
            .to_string();
        assert!(err.contains("config version 2"), "{err}");
        assert!(err.contains("reads 1"), "{err}");
    }

    /// What `deny_unknown_fields` buys, pinned: a typo is a loud error at the
    /// root and inside a profile, and the message lists what is recognized.
    #[test]
    fn an_unknown_key_is_refused_at_every_level() {
        let err = parse("{\"version\": 1, \"profils\": {}}")
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown key \"profils\""), "{err}");
        assert!(err.contains("default_profile, profiles, version"), "{err}");

        let err = parse("{\"version\": 1, \"profiles\": {\"chat\": {\"contxt\": 16384}}}")
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown key \"contxt\""), "{err}");
        assert!(err.contains("profile \"chat\""), "{err}");
        assert!(
            err.contains("cache_bytes, context, prefill, prefill_chunk, threads"),
            "{err}"
        );
    }

    /// Every way a hostile or mistyped value can be wrong is a message, never
    /// a panic and never a silently clamped dial.
    #[test]
    fn a_hostile_value_is_a_message_rather_than_a_panic() {
        let profile = |body: &str| {
            parse(&format!(
                "{{\"version\": 1, \"profiles\": {{\"p\": {body}}}}}"
            ))
        };
        for bad in [
            "{\"context\": 0}",
            "{\"context\": -1}",
            "{\"context\": 1.5}",
            "{\"context\": \"16384\"}",
            "{\"threads\": 0}",
            "{\"threads\": 99999}",
            "{\"prefill_chunk\": 0}",
            "{\"prefill\": \"layer-major\"}",
            "{\"prefill\": 1}",
            "{\"cache_bytes\": \"1440X\"}",
            "{\"cache_bytes\": 0}",
            "{\"cache_bytes\": -5}",
            "{\"cache_bytes\": true}",
            "{\"cache_bytes\": [1440]}",
        ] {
            assert!(profile(bad).is_err(), "{bad} should not parse");
        }
        // An explicit null says nothing, exactly as leaving the key out does.
        let config = profile("{\"context\": null, \"threads\": null}").unwrap();
        assert_eq!(
            config.select(Some("p")).unwrap().profile,
            Profile::default()
        );
        // The `threads` ceiling is the flag's, named in the message.
        let err = profile("{\"threads\": 99999}").unwrap_err().to_string();
        assert!(err.contains("maximum of 256"), "{err}");

        // A context nothing could ever allocate is *not* a parse error — the
        // flag has no upper bound either, and the trained-context check and the
        // projection are the two things entitled to refuse a window. What
        // matters is that it is refused rather than overflowing something: the
        // KV term for `usize::MAX` positions is arithmetic that has to fail,
        // not wrap.
        let config = profile(&format!("{{\"context\": {}}}", u64::MAX)).unwrap();
        let dials = Dials::resolve(
            config.select(Some("p")).unwrap(),
            &Flags::default(),
            &Env::default(),
        );
        let (manifest, layout) = v0_install();
        assert!(
            dials.project(&manifest, &layout).is_err(),
            "a context of {} must be refused by the projection",
            usize::MAX
        );
    }

    /// `cache_bytes` in the file is the flag's parser, so every spelling the
    /// flag takes means the same number here — including the suffixes, which
    /// are binary.
    #[test]
    fn cache_bytes_in_the_file_parses_exactly_as_the_flag_does() {
        for text in ["1440M", "1440MiB", "1.40625G", "1509949440", "  1440M  "] {
            let config = parse(&format!(
                "{{\"version\": 1, \"profiles\": {{\"p\": {{\"cache_bytes\": \"{text}\"}}}}}}"
            ))
            .unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert_eq!(
                config.select(Some("p")).unwrap().profile.cache_bytes,
                Some(parse_bytes(text).unwrap()),
                "{text:?}"
            );
        }
        // And a bare JSON number is the same byte count as the same digits in
        // a string, which is what `--cache-bytes 1509949440` means.
        let config =
            parse("{\"version\": 1, \"profiles\": {\"p\": {\"cache_bytes\": 1509949440}}}")
                .unwrap();
        assert_eq!(
            config.select(Some("p")).unwrap().profile.cache_bytes,
            Some(1_509_949_440)
        );
        // The documented default really is the documented number, on both
        // paths: this is what keeps a bare command line byte-identical.
        assert_eq!(parse_bytes("1440M").unwrap(), DEFAULT_CACHE_BYTES);
    }

    /// A profile name the file does not define is an error listing the ones it
    /// does — the question a wrong name actually raises.
    #[test]
    fn an_unknown_profile_lists_the_ones_that_exist() {
        let config = parse(SAMPLE).unwrap();
        let err = config.select(Some("agnet")).unwrap_err().to_string();
        assert!(err.contains("no profile called \"agnet\""), "{err}");
        assert!(err.contains("it defines: agent, chat"), "{err}");
        assert!(err.contains("/nowhere/t/config.json"), "{err}");

        // With no file at all the message says so rather than listing nothing.
        let err = Config::default()
            .select(Some("agent"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("there is no config file"), "{err}");

        // A file that defines none says that too.
        let err = parse("{\"version\": 1, \"profiles\": {}}")
            .unwrap()
            .select(Some("agent"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("defines no profiles"), "{err}");
    }

    /// `default_profile` picks the profile when `--profile` does not, and
    /// `--profile` beats it.
    #[test]
    fn the_default_profile_is_used_when_no_name_is_given() {
        let config = parse(SAMPLE).unwrap();
        let selected = config.select(None).unwrap();
        assert_eq!(selected.name.as_deref(), Some("chat"));
        assert_eq!(selected.profile.context, Some(4096));
        assert_eq!(
            config.select(Some("agent")).unwrap().profile.context,
            Some(16_384)
        );
        // Without a `default_profile`, no name and no dials.
        let bare = parse("{\"version\": 1, \"profiles\": {\"a\": {}}}").unwrap();
        let selected = bare.select(None).unwrap();
        assert_eq!(selected.name, None);
        assert_eq!(selected.profile, Profile::default());
    }

    // -----------------------------------------------------------------
    // the projection `plan` prints
    // -----------------------------------------------------------------

    /// The v0 install's metadata, declared rather than installed: the real
    /// 48-layer geometry, the real `common.bin` size, and the two real
    /// per-layer blob strides. The same fixture `io::slots`' own tests use,
    /// which is why the numbers below are the memory contract's.
    fn v0_install() -> (Manifest, ExpertsLayout) {
        /// `common.bin` on Qwen3-30B-A3B.
        const COMMON_BYTES: u64 = 1_073_051_648;
        /// Q6_K-down layers pack the wider blob; the rest are pure Q4_K.
        const STRIDES: [u64; 2] = [3_059_712, 2_654_208];

        let manifest = Manifest {
            rvmp_version: RVMP_VERSION,
            model_id: "qwen3-30b-a3b-instruct-2507".to_owned(),
            source: SourceInfo {
                hf_repo: "test/qwen3".to_owned(),
                revision: "0".repeat(40),
                file: "qwen3-Q4_K_M.gguf".to_owned(),
                sha256: "0".repeat(64),
            },
            arch: ArchInfo {
                n_layers: 48,
                n_experts: 128,
                top_k: 8,
                hidden: 2048,
                moe_intermediate: 768,
                n_heads: 32,
                n_kv_heads: 4,
                head_dim: 128,
                vocab: 151_936,
                context_length: 262_144,
                rope_theta: 10_000_000.0,
                rms_eps: 1e-6,
                norm_topk_prob: true,
                tie_embeddings: false,
                shared_expert: false,
                sliding_window: None,
            },
            quant: QuantInfo {
                scheme: "gguf".to_owned(),
                tensor_types: BTreeMap::new(),
            },
            common_tensors: BTreeMap::new(),
            files: BTreeMap::from([(
                "common.bin".to_owned(),
                FileEntry {
                    size: COMMON_BYTES,
                    sha256: "0".repeat(64),
                },
            )]),
        };
        let layout = ExpertsLayout {
            layers: (0..48)
                .map(|layer| LayerLayout {
                    file: ramvamp_core::format::layer_file_name(layer),
                    stride: STRIDES[usize::from(layer >= 24)],
                    n_experts: 128,
                    projections: Vec::new(),
                })
                .collect(),
        };
        (manifest, layout)
    }

    /// The dry run prices the **resolved** dials with the runtime's own
    /// predictor, and prints that prediction verbatim. Anything else — a
    /// second copy of the arithmetic, or a projection of the flags rather than
    /// of what the run would use — is the drift this asserts against.
    #[test]
    fn the_projection_is_the_runtime_predictor_over_the_resolved_dials() {
        let (manifest, layout) = v0_install();
        let config = parse(SAMPLE).unwrap();
        // The profile alone decides both terms: no flag, no variable.
        let dials = Dials::resolve(
            config.select(Some("agent")).unwrap(),
            &Flags::default(),
            &Env::default(),
        );
        assert_eq!(dials.context.value, 16_384);
        assert_eq!(dials.cache_bytes.value, 1440 << 20);

        let projected = dials.project(&manifest, &layout).expect("projects");
        assert_eq!(
            projected,
            Footprint::project(&manifest, &layout, 1440 << 20, 16_384).unwrap(),
            "the plan must ask the predictor the runtime asks, about the dials the \
             run would use"
        );
        // And the KV term really did move with the profile's context.
        assert_eq!(projected.kv, 16_384 * 96 * 1024);

        let lines = plan_lines(&dials, Path::new("/m.rvmp"), &projected, Some(16 << 30));
        assert!(
            lines.contains(&format!("projected: {projected}")),
            "{lines:#?}"
        );
    }

    /// The report answers the question it exists for: which layer won each
    /// dial, and whether the total fits. Under a 3 GiB ceiling the same
    /// configuration that fits a workstation must say it does not, and by how
    /// much.
    #[test]
    fn the_plan_report_names_every_source_and_both_verdicts() {
        let (manifest, layout) = v0_install();
        let config = parse(SAMPLE).unwrap();
        let dials = Dials::resolve(
            config.select(Some("agent")).unwrap(),
            &Flags::default(),
            &Env::default(),
        );
        let projected = dials.project(&manifest, &layout).unwrap();

        let lines = plan_lines(&dials, Path::new("/m.rvmp"), &projected, Some(16 << 30));
        assert_eq!(lines[0], "profile: agent (from /nowhere/t/config.json)");
        assert_eq!(lines[1], "  context        16384         profile");
        assert_eq!(lines[2], "  cache-bytes    1,440 MiB     profile");
        assert_eq!(lines[3], "  threads        4             profile");
        assert_eq!(lines[4], "  prefill        token-major   profile");
        assert_eq!(lines[5], "  prefill-chunk  256           profile");
        assert_eq!(lines[7], "model:     /m.rvmp");
        assert!(lines[9].starts_with("available: 16,384 MiB"), "{lines:#?}");
        assert!(lines[10].starts_with("verdict:   fits"), "{lines:#?}");

        // The same configuration under the published 3 GiB cgroup.
        let lines = plan_lines(&dials, Path::new("/m.rvmp"), &projected, Some(3 << 30));
        assert_eq!(lines[9], "available: 3,072 MiB");
        assert_eq!(lines[10], "verdict:   does not fit, 1,041 MiB over");

        // A kernel that will not say is not a verdict of "fits".
        let lines = plan_lines(&dials, Path::new("/m.rvmp"), &projected, None);
        assert!(lines[10].contains("cannot be judged"), "{lines:#?}");

        // The built-in default, which is the published 4K/11-slot
        // configuration, does fit the 3 GiB budget — and says so with every
        // source reading `default`.
        let bare = Dials::resolve(Selection::default(), &Flags::default(), &Env::default());
        let projected = bare.project(&manifest, &layout).unwrap();
        let lines = plan_lines(&bare, Path::new("/m.rvmp"), &projected, Some(3 << 30));
        assert_eq!(
            lines[0],
            "profile: none (built-in defaults; no config path)"
        );
        assert_eq!(lines[1], "  context        4096          default");
        assert_eq!(lines[3], "  threads        auto          default");
        assert_eq!(lines[10], "verdict:   fits, 111 MiB spare");
    }

    /// The MiB column is the memory contract's rounding and grouping, which is
    /// what makes the projected and available lines comparable at a glance.
    #[test]
    fn mib_rounds_and_groups_the_way_the_memory_contract_does() {
        assert_eq!(mib(0), "0");
        assert_eq!(mib(MIB), "1");
        assert_eq!(mib(MIB / 2), "1", "rounds to nearest");
        assert_eq!(mib(1440 * MIB), "1,440");
        assert_eq!(mib(3 << 30), "3,072");
        assert_eq!(mib(u128::from(u64::MAX)), "17,592,186,044,416");
    }
}
