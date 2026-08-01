//! Windowed streaming executor: large sequential source reads demuxed
//! into positioned writes on preallocated destination files.
//!
//! The Hugging Face Xet CDN signs each download URL for one exact byte
//! range (any other range 403s), so per-tensor range requests are off the
//! table: the executor walks the source in fixed sequential windows
//! (default 32 MiB), one `RangeRead::read_at` per window into a single
//! reusable buffer, and demuxes every copy-op span overlapping the window
//! to its destination offset with `pwrite` (`FileExt::write_all_at`).
//! Scratch is exactly one window buffer plus small bookkeeping; no tensor,
//! shard, or file is ever materialized whole (hard rule). Ops larger than
//! a window are written span-by-span across consecutive windows.
//!
//! Durability: after every `fsync_every_windows` completed windows (and
//! after the last), destination files are fsync'd first, then the state
//! file advances `durable_windows` — so the state never claims durability
//! the data files do not have. A resume restarts from the first
//! non-durable window.

use std::collections::BTreeMap;
use std::fs;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use sha2::{Digest, Sha256};

use super::state::{InstallState, write_state};
use super::{InstallError, hex, io_err};
use crate::plan::CopyOp;
use crate::source::RangeRead;

/// Emit a progress line roughly every this many source bytes.
const PROGRESS_EVERY_BYTES: u64 = 1 << 30;

/// Fixed mapping from window indices to absolute source byte ranges.
///
/// Windows tile the contiguous source region `[region_start, region_end)`
/// spanned by the sorted copy map (the GGUF data section the plan actually
/// copies from; the header before the first op is never fetched). The
/// geometry is fully determined by the sorted ops and the window size, so
/// a resume reconstructs identical window boundaries.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WindowGeometry {
    /// First source byte any op copies.
    pub region_start: u64,
    /// One past the last source byte any op copies.
    pub region_end: u64,
    /// Window size in bytes (the last window may be shorter).
    pub window_bytes: u64,
}

impl WindowGeometry {
    /// Geometry for a non-empty, sorted op list.
    pub(crate) fn from_ops(ops: &[CopyOp], window_bytes: u64) -> Result<Self, InstallError> {
        let (Some(first), Some(last)) = (ops.first(), ops.last()) else {
            return Err(InstallError::BadPlan("empty copy map".to_owned()));
        };
        if window_bytes == 0 {
            return Err(InstallError::BadOptions("window size is zero".to_owned()));
        }
        Ok(WindowGeometry {
            region_start: first.src.start,
            region_end: last.src.end,
            window_bytes,
        })
    }

    /// Total number of windows.
    pub(crate) fn n_windows(&self) -> u64 {
        (self.region_end - self.region_start).div_ceil(self.window_bytes)
    }

    /// Absolute source byte range of window `k` (must be `< n_windows()`).
    pub(crate) fn range(&self, k: u64) -> Range<u64> {
        let start = self.region_start + k * self.window_bytes;
        let end = (start + self.window_bytes).min(self.region_end);
        start..end
    }
}

/// Visit every copy-op span overlapping `range`, in ascending source
/// order: `f(op, overlap_start, overlap_end)` with absolute source
/// offsets. `ops` must be sorted by `src.start` and disjoint.
pub(crate) fn for_each_overlap<F>(
    ops: &[CopyOp],
    range: &Range<u64>,
    mut f: F,
) -> Result<(), InstallError>
where
    F: FnMut(&CopyOp, u64, u64) -> Result<(), InstallError>,
{
    let first = ops.partition_point(|op| op.src.end <= range.start);
    for op in &ops[first..] {
        if op.src.start >= range.end {
            break;
        }
        let overlap_start = op.src.start.max(range.start);
        let overlap_end = op.src.end.min(range.end);
        if overlap_end > overlap_start {
            f(op, overlap_start, overlap_end)?;
        }
    }
    Ok(())
}

/// Destination files of one install, opened once and preallocated to
/// their final sizes.
#[derive(Debug)]
pub(crate) struct DestFiles {
    root: PathBuf,
    files: BTreeMap<String, fs::File>,
}

impl DestFiles {
    /// Open (and on a fresh install create + `set_len`-preallocate) every
    /// destination file under `root`. On resume, each file must already
    /// exist with exactly its planned size — anything else means the
    /// partial directory does not belong to this plan.
    ///
    /// `set_len` reserves the namespace but not blocks (sparse), so
    /// ENOSPC can still surface mid-write; writes map it to a clear
    /// [`InstallError::NoSpace`].
    pub(crate) fn open(
        root: &Path,
        sizes: &BTreeMap<String, u64>,
        resume: bool,
    ) -> Result<Self, InstallError> {
        let mut files = BTreeMap::new();
        for (name, &size) in sizes {
            let path = root.join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| io_err(parent, e))?;
            }
            if resume {
                let metadata = fs::metadata(&path).map_err(|_| {
                    InstallError::StateMismatch(format!(
                        "destination file {} is missing; restart with --overwrite",
                        path.display()
                    ))
                })?;
                if metadata.len() != size {
                    return Err(InstallError::StateMismatch(format!(
                        "destination file {} is {} bytes, plan says {size}; \
                         restart with --overwrite",
                        path.display(),
                        metadata.len()
                    )));
                }
            }
            let file = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(!resume)
                .open(&path)
                .map_err(|e| io_err(&path, e))?;
            if !resume {
                file.set_len(size).map_err(|e| io_err(&path, e))?;
            }
            files.insert(name.clone(), file);
        }
        Ok(DestFiles {
            root: root.to_owned(),
            files,
        })
    }

    /// Handle for an install-relative destination name.
    pub(crate) fn get(&self, name: &str) -> Result<&fs::File, InstallError> {
        self.files
            .get(name)
            .ok_or_else(|| InstallError::BadPlan(format!("copy op targets unknown file {name:?}")))
    }

    /// Path of an install-relative destination name.
    fn path_of(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// fsync every destination file.
    pub(crate) fn sync_all(&self) -> Result<(), InstallError> {
        for (name, file) in &self.files {
            file.sync_all()
                .map_err(|e| io_err(&self.path_of(name), e))?;
        }
        Ok(())
    }
}

/// Re-hash one durable window from the *destination* files and compare
/// against the digest recorded in the state file: reads back every op
/// span the window wrote (ascending source order, same order the digest
/// was computed in) into `buf` and re-hashes. Cheap spot-check that the
/// bytes claimed durable are still the bytes that were downloaded.
pub(crate) fn verify_window_readback(
    ops: &[CopyOp],
    geom: WindowGeometry,
    dests: &DestFiles,
    state: &InstallState,
    window: u64,
    buf: &mut [u8],
) -> Result<(), InstallError> {
    let expected = state
        .window_digests
        .get(window as usize)
        .ok_or_else(|| InstallError::StateMismatch(format!("no digest for window {window}")))?;
    let range = geom.range(window);
    let mut hasher = Sha256::new();
    for_each_overlap(ops, &range, |op, overlap_start, overlap_end| {
        let len = (overlap_end - overlap_start) as usize;
        let dst_offset = op.dst_offset + (overlap_start - op.src.start);
        let span = &mut buf[..len];
        dests
            .get(&op.dst_file)?
            .read_exact_at(span, dst_offset)
            .map_err(|e| io_err(&dests.path_of(&op.dst_file), e))?;
        hasher.update(span);
        Ok(())
    })?;
    if !hex(&hasher.finalize()).eq_ignore_ascii_case(expected) {
        return Err(InstallError::WindowDigestMismatch { window });
    }
    Ok(())
}

/// Numbers from one executor run (the current session only).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RunOutcome {
    /// Copy-op bytes written this session (excludes source gap bytes).
    pub bytes_copied: u64,
}

/// One windowed transfer session over an open set of destinations.
pub(crate) struct Executor<'a> {
    /// Source being copied from.
    pub source: &'a dyn RangeRead,
    /// Copy map, sorted by `src.start`, disjoint, validated.
    pub ops: &'a [CopyOp],
    /// Window geometry derived from `ops` and the state's window size.
    pub geom: WindowGeometry,
    /// Open, size-checked destination files.
    pub dests: &'a DestFiles,
    /// Partial directory the state file lives in.
    pub partial: &'a Path,
    /// Batch size for fsync + state checkpointing.
    pub fsync_every_windows: u64,
    /// Test knob: abort with [`InstallError::TestAbort`] after this many
    /// windows have been processed this session (checkpoint batching still
    /// applies, so fewer windows may be durable).
    pub fail_after_windows: Option<u64>,
}

impl Executor<'_> {
    /// Run from the state's first non-durable window to the end, keeping
    /// `state` and its on-disk copy in sync at every checkpoint.
    pub(crate) fn run(
        &self,
        state: &mut InstallState,
        buf: &mut [u8],
    ) -> Result<RunOutcome, InstallError> {
        let n_windows = self.geom.n_windows();
        let start_window = state.durable_windows;
        let total_bytes = self.geom.region_end - self.geom.region_start;
        let session_started = Instant::now();
        let mut session_window_bytes = 0u64;
        let mut bytes_copied = 0u64;
        let mut windows_since_sync = 0u64;
        let mut next_progress = PROGRESS_EVERY_BYTES;

        for k in start_window..n_windows {
            let range = self.geom.range(k);
            let window_len = (range.end - range.start) as usize;
            let window = &mut buf[..window_len];
            self.source.read_at(range.start, window)?;

            let mut hasher = Sha256::new();
            for_each_overlap(self.ops, &range, |op, overlap_start, overlap_end| {
                let lo = (overlap_start - range.start) as usize;
                let hi = (overlap_end - range.start) as usize;
                let span = &window[lo..hi];
                hasher.update(span);
                let dst_offset = op.dst_offset + (overlap_start - op.src.start);
                self.dests
                    .get(&op.dst_file)?
                    .write_all_at(span, dst_offset)
                    .map_err(|e| io_err(&self.dests.path_of(&op.dst_file), e))?;
                bytes_copied += (hi - lo) as u64;
                Ok(())
            })?;
            state.window_digests.push(hex(&hasher.finalize()));
            windows_since_sync += 1;
            session_window_bytes += window_len as u64;

            let last = k + 1 == n_windows;
            if windows_since_sync >= self.fsync_every_windows || last {
                // Data first, then the state that claims it durable.
                self.dests.sync_all()?;
                state.durable_windows = k + 1;
                write_state(self.partial, state)?;
                windows_since_sync = 0;
            }

            if session_window_bytes >= next_progress {
                next_progress += PROGRESS_EVERY_BYTES;
                let done = range.end - self.geom.region_start;
                let pct = 100.0 * done as f64 / total_bytes as f64;
                let secs = session_started.elapsed().as_secs_f64().max(1e-9);
                let rate = session_window_bytes as f64 / secs;
                let eta = (total_bytes - done) as f64 / rate.max(1.0);
                tracing::info!(
                    "install progress: {pct:.1}% ({done} of {total_bytes} bytes), \
                     {:.1} MiB/s, eta {eta:.0}s",
                    rate / (1024.0 * 1024.0)
                );
            }

            if let Some(fail_after) = self.fail_after_windows
                && k + 1 - start_window >= fail_after
            {
                return Err(InstallError::TestAbort { windows: k + 1 });
            }
        }
        Ok(RunOutcome { bytes_copied })
    }
}
