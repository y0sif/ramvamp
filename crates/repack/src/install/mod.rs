//! Streaming installer: execute a [`RepackPlan`] against a source,
//! producing a complete `.rvmp` install directory.
//!
//! The executor walks the source in large sequential windows (the Xet CDN
//! signs URLs per exact byte range, so per-tensor requests are not an
//! option; see `executor`), demuxing each window into preallocated
//! destination files with positioned writes. Progress is checkpointed to
//! `install-state.json` inside the `.partial` staging directory so an
//! interrupted install resumes from its last durable window. Finalization
//! writes `experts/layout.json`, hashes every data file, writes
//! `manifest.json` last, atomically promotes the staging directory, and
//! re-verifies the result. A pidfile lock (`<target>.lock`) serializes
//! install, resume, and discard on the same target.
//!
//! # `source.sha256`: the window digest-of-digests
//!
//! The manifest's `source.sha256` is **not** the plain SHA-256 of the
//! source GGUF. Maintaining one running whole-file hasher would make
//! resume impossible mid-stream (sha2 exposes no resumable midstate), so
//! the installer records one SHA-256 per window over the source bytes
//! that window actually copied (copy-op overlap spans in ascending source
//! order; alignment padding the plan never copies is excluded), and
//! `source.sha256` is the SHA-256 over the concatenation of those raw
//! 32-byte per-window digests — a merkle-ish digest-of-digests. It is
//! deterministic: uninterrupted and resumed installs of the same source,
//! plan, and window size produce the identical value (the value does
//! depend on the window size). The per-window digests in the state file
//! are the transfer-integrity evidence while an install is in flight;
//! comparing the final value against an audited upstream whole-file hash
//! is a post-v0 nicety. The scheme and value are printed at finalize.

mod executor;
mod lock;
mod state;

#[cfg(test)]
mod tests;

pub use state::{InstallState, STATE_FILE, STATE_VERSION};

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use ramvamp_core::format::{
    self, COMMON_FILE, FileEntry, LAYOUT_FILE, MANIFEST_FILE, Manifest, RVMP_VERSION, SourceInfo,
};

use crate::plan::{CopyOp, RepackPlan};
use crate::source::{RangeRead, SourceError};

use executor::{DestFiles, Executor, WindowGeometry};
use lock::InstallLock;

/// Default transfer window: 32 MiB of source per range request.
pub const DEFAULT_WINDOW_BYTES: u64 = 32 * 1024 * 1024;

/// Largest accepted transfer window: 1 GiB, the same ceiling the CLI's
/// `--window-mib 1..=1024` enforces for fresh installs. Also applied to
/// `window_bytes` loaded from a resume state file, so a hostile state
/// file cannot drive an unbounded scratch allocation.
pub const MAX_WINDOW_BYTES: u64 = 1024 * 1024 * 1024;

/// Default checkpoint batch: fsync destinations and advance the durable
/// state every this many windows.
pub const DEFAULT_FSYNC_EVERY_WINDOWS: u64 = 8;

/// Error installing, resuming, or discarding a `.rvmp` install.
#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    /// Error from the `.rvmp` format layer (manifest/layout/verify).
    #[error(transparent)]
    Format(#[from] format::FormatError),
    /// Error reading the source.
    #[error(transparent)]
    Source(#[from] SourceError),
    /// A window transfer from the source failed (after the source layer's
    /// own retries). Progress up to the last checkpoint is durable, so the
    /// hint mirrors [`InstallError::NoSpace`]: resume, don't restart.
    #[error(
        "source read failed at window {window}; downloaded progress up to \
         the last checkpoint is saved — re-run with --resume to continue \
         from there"
    )]
    SourceRead {
        /// Index of the window whose transfer failed.
        window: u64,
        /// Underlying source error.
        #[source]
        source: SourceError,
    },
    /// A filesystem operation failed.
    #[error("{}: {source}", path.display())]
    Io {
        /// Path the operation was acting on.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// The filesystem ran out of space (ENOSPC, surfaced with context).
    #[error(
        "out of disk space writing {} (the install needs the full planned \
         size up front; free space and resume with --resume)",
        path.display()
    )]
    NoSpace {
        /// Path being written when space ran out.
        path: PathBuf,
    },
    /// The final install directory already exists.
    #[error("install target {} already exists (pass --overwrite to replace it)", .0.display())]
    TargetExists(PathBuf),
    /// A partial install exists and neither resume nor overwrite was asked.
    #[error(
        "partial install {} exists (pass --resume to continue it or --overwrite to restart)",
        .0.display()
    )]
    PartialExists(PathBuf),
    /// Overwrite refused: the existing target does not look like an
    /// install, so deleting it could destroy unrelated data.
    #[error(
        "refusing to overwrite {}: not a .rvmp install (no manifest.json) and not empty",
        .0.display()
    )]
    OverwriteRefused(PathBuf),
    /// Another process holds the install lock.
    #[error("another install owns {} (pid {pid}); remove the file if that pid is gone", path.display())]
    Locked {
        /// Lock file path.
        path: PathBuf,
        /// Pid recorded in the lock file (0 if unreadable/contended).
        pid: u32,
    },
    /// The resume state does not match this source/plan.
    #[error("resume state mismatch: {0}")]
    StateMismatch(String),
    /// A durable window's destination bytes no longer hash to the digest
    /// recorded when they were downloaded.
    #[error(
        "window {window} readback digest mismatch: destination bytes no longer match \
         the recorded source digest (disk corruption or tampering); restart with --overwrite"
    )]
    WindowDigestMismatch {
        /// Index of the failing window.
        window: u64,
    },
    /// The plan cannot be executed (internal inconsistency).
    #[error("plan invalid for execution: {0}")]
    BadPlan(String),
    /// Invalid installer options.
    #[error("invalid options: {0}")]
    BadOptions(String),
    /// Nothing to discard at the given target.
    #[error("nothing to discard: {} does not exist", .0.display())]
    NothingToDiscard(PathBuf),
    /// The test-only abort hook fired.
    #[error("aborted by test hook after {windows} windows")]
    TestAbort {
        /// Windows processed when the hook fired.
        windows: u64,
    },
}

/// Attach path context to an I/O error, surfacing ENOSPC distinctly.
pub(crate) fn io_err(path: &Path, source: io::Error) -> InstallError {
    if source.kind() == io::ErrorKind::StorageFull {
        InstallError::NoSpace {
            path: path.to_path_buf(),
        }
    } else {
        InstallError::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}

/// Lowercase hex of a byte string.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Identity of the source checkpoint, for the resume state and the
/// manifest's `source` section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcePin {
    /// Resolved URL (remote) or canonical path (local); the resume
    /// identity a partial install is pinned to.
    pub url: String,
    /// Hugging Face repo for the manifest (`"(local)"` for local files).
    pub hf_repo: String,
    /// Repo revision for the manifest (`"(local)"` for local files).
    pub revision: String,
    /// Source file name, e.g. `...Q4_K_M.gguf`.
    pub file: String,
}

/// Installer knobs.
#[derive(Debug, Clone)]
pub struct InstallOptions {
    /// Transfer window size in bytes ([`DEFAULT_WINDOW_BYTES`]). A resume
    /// adopts the window size recorded in the state file instead.
    pub window_bytes: u64,
    /// Checkpoint batch ([`DEFAULT_FSYNC_EVERY_WINDOWS`]).
    pub fsync_every_windows: u64,
    /// Continue a matching partial install instead of failing on one.
    pub resume: bool,
    /// Replace an existing install or partial from scratch.
    pub overwrite: bool,
    /// Skip the post-promotion hash self-check.
    pub skip_verify: bool,
    /// Test-only: abort after this many windows this session.
    pub fail_after_windows: Option<u64>,
}

impl Default for InstallOptions {
    fn default() -> Self {
        InstallOptions {
            window_bytes: DEFAULT_WINDOW_BYTES,
            fsync_every_windows: DEFAULT_FSYNC_EVERY_WINDOWS,
            resume: false,
            overwrite: false,
            skip_verify: false,
            fail_after_windows: None,
        }
    }
}

/// What an install run did, for display.
#[derive(Debug, Clone)]
pub struct InstallReport {
    /// The promoted install directory.
    pub final_dir: PathBuf,
    /// `source.sha256` recorded in the manifest: the window
    /// digest-of-digests (see the module docs).
    pub source_sha256: String,
    /// Total windows in the transfer geometry.
    pub windows_total: u64,
    /// Windows already durable when this session started (0 when fresh).
    pub windows_resumed: u64,
    /// Copy-op bytes written this session.
    pub bytes_copied: u64,
    /// Whether the post-promotion self-check ran and passed.
    pub verified: bool,
}

/// Install (or resume installing) `plan` from `source` into `final_dir`.
///
/// Streams the source in sequential windows with a single reusable window
/// buffer, checkpoints durable progress into the `.partial` staging
/// directory, then finalizes: `experts/layout.json`, per-file SHA-256
/// hashes, `manifest.json` (written last), atomic promotion, and a
/// hash self-check (skippable). See the module docs for the
/// `source.sha256` digest-of-digests scheme.
pub fn install(
    source: &dyn RangeRead,
    plan: &RepackPlan,
    pin: &SourcePin,
    final_dir: &Path,
    opts: &InstallOptions,
) -> Result<InstallReport, InstallError> {
    if opts.window_bytes == 0 {
        return Err(InstallError::BadOptions("window size is zero".to_owned()));
    }
    if opts.window_bytes > MAX_WINDOW_BYTES {
        return Err(InstallError::BadOptions(format!(
            "window size {} exceeds the {MAX_WINDOW_BYTES}-byte cap",
            opts.window_bytes
        )));
    }
    if opts.fsync_every_windows == 0 {
        return Err(InstallError::BadOptions(
            "fsync batch size is zero".to_owned(),
        ));
    }
    let _lock = InstallLock::acquire(final_dir)?;

    if final_dir.symlink_metadata().is_ok() {
        if opts.overwrite {
            remove_existing_install(final_dir)?;
        } else {
            return Err(InstallError::TargetExists(final_dir.to_path_buf()));
        }
    }

    // Sort the copy map by source offset (the plan emits common tensors in
    // name order and expert ops interleaved gate/up/down, neither of which
    // is source order) and validate it before trusting any offset.
    let mut ops = plan.copy_ops.clone();
    ops.sort_by_key(|op| op.src.start);
    let sizes = dest_sizes(plan);
    validate_ops(&ops, source.len(), &sizes)?;
    let fingerprint = state::plan_fingerprint(&ops);

    let partial = format::partial_dir(final_dir);
    let (mut st, resume) = prepare_state(&partial, source, pin, &fingerprint, opts)?;
    if let Some(report) = maybe_recover_complete_partial(&partial, final_dir, opts, resume)? {
        return Ok(report);
    }

    let geom = WindowGeometry::from_ops(&ops, st.window_bytes)?;
    let n_windows = geom.n_windows();
    if st.durable_windows > n_windows {
        return Err(InstallError::StateMismatch(format!(
            "state claims {} durable windows, geometry has {n_windows}",
            st.durable_windows
        )));
    }
    let windows_resumed = st.durable_windows;

    let dests = DestFiles::open(&partial, &sizes, resume)?;
    // The one fixed scratch buffer for the whole install (hard rule: no
    // tensor or shard is ever materialized whole).
    let mut buf = vec![0u8; st.window_bytes as usize];

    if resume && windows_resumed > 0 {
        let spot = pick_window(windows_resumed);
        executor::verify_window_readback(&ops, geom, &dests, &st, spot, &mut buf)?;
        tracing::info!(
            window = spot,
            durable = windows_resumed,
            "resume spot-check passed; continuing from window {windows_resumed}"
        );
    }

    let outcome = Executor {
        source,
        ops: &ops,
        geom,
        dests: &dests,
        partial: &partial,
        fsync_every_windows: opts.fsync_every_windows,
        fail_after_windows: opts.fail_after_windows,
    }
    .run(&mut st, &mut buf)?;
    drop(buf);
    drop(dests);

    let manifest = finalize(&partial, plan, pin, &st, &sizes)?;
    promote_with_cleanup(&partial, final_dir)?;
    tracing::info!(
        source_sha256 = manifest.source.sha256.as_str(),
        windows = n_windows,
        window_bytes = st.window_bytes,
        "finalized install; source.sha256 is the per-window digest-of-digests \
         (not the plain file hash; see the install module docs)"
    );

    let verified = if opts.skip_verify {
        false
    } else {
        self_check(final_dir, &manifest)?;
        true
    };
    Ok(InstallReport {
        final_dir: final_dir.to_path_buf(),
        source_sha256: manifest.source.sha256,
        windows_total: n_windows,
        windows_resumed,
        bytes_copied: outcome.bytes_copied,
        verified,
    })
}

/// Delete the partial directory for `final_dir` (computed with
/// [`format::partial_dir`], so it always matches the target path). Takes
/// the same advisory lock as install/resume. Returns the removed path.
pub fn discard_partial(final_dir: &Path) -> Result<PathBuf, InstallError> {
    let _lock = InstallLock::acquire(final_dir)?;
    let partial = format::partial_dir(final_dir);
    if partial.symlink_metadata().is_err() {
        return Err(InstallError::NothingToDiscard(partial));
    }
    fs::remove_dir_all(&partial).map_err(|e| io_err(&partial, e))?;
    tracing::info!(dir = %partial.display(), "discarded partial install");
    Ok(partial)
}

/// Planned destination data files: `common.bin` plus every layer file.
fn dest_sizes(plan: &RepackPlan) -> BTreeMap<String, u64> {
    let mut sizes = plan.layer_file_sizes.clone();
    sizes.insert(COMMON_FILE.to_owned(), plan.common_size);
    sizes
}

/// Validate a sorted copy map against the source length and the planned
/// destination sizes: non-empty ops, disjoint ascending source ranges,
/// every destination span inside its planned file, and destination spans
/// disjoint within each file.
fn validate_ops(
    ops: &[CopyOp],
    source_len: u64,
    sizes: &BTreeMap<String, u64>,
) -> Result<(), InstallError> {
    if ops.is_empty() {
        return Err(InstallError::BadPlan("empty copy map".to_owned()));
    }
    let mut prev_end = 0u64;
    for op in ops {
        if op.is_empty() {
            return Err(InstallError::BadPlan(format!(
                "empty copy op for {:?}",
                op.dst_file
            )));
        }
        if op.src.start < prev_end {
            return Err(InstallError::BadPlan(format!(
                "overlapping source ranges near offset {}",
                op.src.start
            )));
        }
        prev_end = op.src.end;
        if op.src.end > source_len {
            return Err(InstallError::BadPlan(format!(
                "op source range {}..{} exceeds source length {source_len}",
                op.src.start, op.src.end
            )));
        }
        let size = sizes.get(&op.dst_file).copied().ok_or_else(|| {
            InstallError::BadPlan(format!("copy op targets unknown file {:?}", op.dst_file))
        })?;
        let dst_end = op.dst_offset.checked_add(op.len()).ok_or_else(|| {
            InstallError::BadPlan(format!("destination overflow in {:?}", op.dst_file))
        })?;
        if dst_end > size {
            return Err(InstallError::BadPlan(format!(
                "op writes to {} past planned size {size} of {:?}",
                dst_end, op.dst_file
            )));
        }
    }
    // Destination disjointness: within each file, no two ops may write
    // overlapping spans. A planner bug that double-writes a destination
    // range would otherwise only surface later as a baffling
    // WindowDigestMismatch "corruption" on resume.
    let mut dst_spans: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
    for op in ops {
        // dst_offset + len cannot overflow: checked above.
        dst_spans
            .entry(op.dst_file.as_str())
            .or_default()
            .push((op.dst_offset, op.dst_offset + op.len()));
    }
    for (file, spans) in &mut dst_spans {
        spans.sort_unstable();
        for pair in spans.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(InstallError::BadPlan(format!(
                    "overlapping destination spans in {file:?} near offset {}",
                    pair[1].0
                )));
            }
        }
    }
    Ok(())
}

/// Decide fresh-vs-resume for the partial directory and produce the
/// working state. Returns `(state, resuming)`.
fn prepare_state(
    partial: &Path,
    source: &dyn RangeRead,
    pin: &SourcePin,
    fingerprint: &str,
    opts: &InstallOptions,
) -> Result<(InstallState, bool), InstallError> {
    let partial_exists = partial.symlink_metadata().is_ok();
    if partial_exists && opts.overwrite {
        fs::remove_dir_all(partial).map_err(|e| io_err(partial, e))?;
    } else if partial_exists && !opts.resume {
        return Err(InstallError::PartialExists(partial.to_path_buf()));
    }

    if partial_exists && opts.resume && !opts.overwrite {
        if format::is_complete(partial) {
            // Crash window between manifest write and promotion; the
            // caller promotes it as-is. Return a placeholder state (never
            // used: the recovery path returns before the executor runs).
            let st = fresh_state(pin, source.len(), fingerprint, opts);
            return Ok((st, true));
        }
        let st = state::load_state(partial)?;
        if st.source_url != pin.url {
            return Err(InstallError::StateMismatch(format!(
                "partial was downloading {:?}, not {:?}; restart with --overwrite",
                st.source_url, pin.url
            )));
        }
        if st.source_len != source.len() {
            return Err(InstallError::StateMismatch(format!(
                "source is {} bytes, partial expected {}; restart with --overwrite",
                source.len(),
                st.source_len
            )));
        }
        if st.plan_fingerprint != fingerprint {
            return Err(InstallError::StateMismatch(
                "plan fingerprint differs (source or planner changed); \
                 restart with --overwrite"
                    .to_owned(),
            ));
        }
        if st.window_bytes != opts.window_bytes {
            tracing::info!(
                state_window = st.window_bytes,
                requested_window = opts.window_bytes,
                "resuming with the window size recorded in the state file"
            );
        }
        return Ok((st, true));
    }

    // Fresh install (also: --resume with nothing to resume).
    if opts.resume && !partial_exists {
        tracing::info!(dir = %partial.display(), "no partial install to resume; starting fresh");
    }
    fs::create_dir_all(partial).map_err(|e| io_err(partial, e))?;
    let st = fresh_state(pin, source.len(), fingerprint, opts);
    state::write_state(partial, &st)?;
    Ok((st, false))
}

fn fresh_state(
    pin: &SourcePin,
    source_len: u64,
    fingerprint: &str,
    opts: &InstallOptions,
) -> InstallState {
    InstallState {
        state_version: STATE_VERSION,
        source_url: pin.url.clone(),
        source_len,
        plan_fingerprint: fingerprint.to_owned(),
        window_bytes: opts.window_bytes,
        durable_windows: 0,
        window_digests: Vec::new(),
    }
}

/// If a resumed partial already contains a valid manifest (crash between
/// manifest write and promotion), promote it as-is and report.
fn maybe_recover_complete_partial(
    partial: &Path,
    final_dir: &Path,
    opts: &InstallOptions,
    resume: bool,
) -> Result<Option<InstallReport>, InstallError> {
    if !(resume && format::is_complete(partial)) {
        return Ok(None);
    }
    tracing::info!(
        dir = %partial.display(),
        "partial install already has a valid manifest; promoting it"
    );
    let manifest = format::load_manifest(partial)?;
    promote_with_cleanup(partial, final_dir)?;
    let verified = if opts.skip_verify {
        false
    } else {
        self_check(final_dir, &manifest)?;
        true
    };
    Ok(Some(InstallReport {
        final_dir: final_dir.to_path_buf(),
        source_sha256: manifest.source.sha256,
        windows_total: 0,
        windows_resumed: 0,
        bytes_copied: 0,
        verified,
    }))
}

/// Finalize a fully-transferred partial: write the layout, hash every
/// data file, and write the manifest last. Returns the manifest.
fn finalize(
    partial: &Path,
    plan: &RepackPlan,
    pin: &SourcePin,
    st: &InstallState,
    sizes: &BTreeMap<String, u64>,
) -> Result<Manifest, InstallError> {
    format::write_layout(partial, &plan.layout)?;

    let mut files = BTreeMap::new();
    for (name, &size) in sizes {
        let path = partial.join(name);
        files.insert(
            name.clone(),
            FileEntry {
                size,
                sha256: format::sha256_file(&path)?,
            },
        );
    }
    let layout_path = partial.join(LAYOUT_FILE);
    let layout_size = fs::metadata(&layout_path)
        .map_err(|e| io_err(&layout_path, e))?
        .len();
    files.insert(
        LAYOUT_FILE.to_owned(),
        FileEntry {
            size: layout_size,
            sha256: format::sha256_file(&layout_path)?,
        },
    );

    let manifest = Manifest {
        rvmp_version: RVMP_VERSION,
        model_id: model_id_from_file(&pin.file),
        source: SourceInfo {
            hf_repo: pin.hf_repo.clone(),
            revision: pin.revision.clone(),
            file: pin.file.clone(),
            sha256: source_digest_of_digests(&st.window_digests)?,
        },
        arch: plan.arch.clone(),
        quant: plan.quant.clone(),
        common_tensors: plan.common_tensors.clone(),
        files,
    };
    format::write_manifest(partial, &manifest)?;
    Ok(manifest)
}

/// Remove the state file (it must never appear in a finished install),
/// then atomically promote the partial directory.
fn promote_with_cleanup(partial: &Path, final_dir: &Path) -> Result<(), InstallError> {
    let state_path = partial.join(STATE_FILE);
    match fs::remove_file(&state_path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(io_err(&state_path, e)),
    }
    format::promote(partial, final_dir)?;
    Ok(())
}

/// Post-promotion self-check: re-hash every file against the manifest and
/// cross-check the layout.
fn self_check(final_dir: &Path, manifest: &Manifest) -> Result<(), InstallError> {
    format::verify_files(final_dir, manifest)?;
    let layout = format::load_layout(final_dir)?;
    layout.validate_against(manifest)?;
    Ok(())
}

/// `source.sha256` = SHA-256 over the concatenated raw 32-byte per-window
/// digests, in window order (see the module docs for why).
fn source_digest_of_digests(window_digests: &[String]) -> Result<String, InstallError> {
    let mut hasher = Sha256::new();
    for (window, digest) in window_digests.iter().enumerate() {
        let raw = decode_hex_digest(digest).ok_or_else(|| {
            InstallError::StateMismatch(format!("malformed digest for window {window}"))
        })?;
        hasher.update(raw);
    }
    Ok(hex(&hasher.finalize()))
}

/// Decode a 64-char hex SHA-256 into raw bytes.
fn decode_hex_digest(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 || !s.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, &[hi, lo]) in s.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let hi = (hi as char).to_digit(16)?;
        let lo = (lo as char).to_digit(16)?;
        out[i] = (hi * 16 + lo) as u8;
    }
    Some(out)
}

/// Stable install identifier derived from the source file name: the stem
/// lowercased, with anything outside `[a-z0-9._-]` replaced by `-`.
fn model_id_from_file(file: &str) -> String {
    let stem = file.strip_suffix(".gguf").unwrap_or(file);
    let id: String = stem
        .chars()
        .map(|c| {
            let c = c.to_ascii_lowercase();
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    if id.is_empty() {
        "model".to_owned()
    } else {
        id
    }
}

/// `--overwrite` target removal, guarded: only delete a directory that
/// contains a `manifest.json` (a real or staged install) or is empty.
fn remove_existing_install(final_dir: &Path) -> Result<(), InstallError> {
    let metadata = final_dir
        .symlink_metadata()
        .map_err(|e| io_err(final_dir, e))?;
    if !metadata.is_dir() {
        return Err(InstallError::OverwriteRefused(final_dir.to_path_buf()));
    }
    let has_manifest = final_dir.join(MANIFEST_FILE).symlink_metadata().is_ok();
    let is_empty = fs::read_dir(final_dir)
        .map_err(|e| io_err(final_dir, e))?
        .next()
        .is_none();
    if !(has_manifest || is_empty) {
        return Err(InstallError::OverwriteRefused(final_dir.to_path_buf()));
    }
    fs::remove_dir_all(final_dir).map_err(|e| io_err(final_dir, e))?;
    tracing::info!(dir = %final_dir.display(), "removed existing install (--overwrite)");
    Ok(())
}

/// Pseudo-random window index in `0..n` for the resume spot-check. Not
/// cryptographic; it only needs to vary across resumes.
fn pick_window(n: u64) -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let mut x = (nanos ^ (u64::from(std::process::id()) << 32)) | 1;
    // xorshift64
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x % n.max(1)
}
