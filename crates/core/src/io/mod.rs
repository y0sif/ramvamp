//! Expert streaming and caching.
//!
//! - Common weights: read-only `mmap` (they are touched every token, so the
//!   page cache keeps them resident).
//! - Routed experts: explicit reads, never demand paging. On Linux with the
//!   `io-uring` feature (default), misses are fetched with io_uring +
//!   O_DIRECT into pre-allocated page-aligned slot buffers; the portable
//!   fallback uses positioned reads (`pread`) on a small thread pool.
//! - Cache: per-layer fixed slot arrays with LFU eviction over frequency
//!   counters indexed by expert id, sized `n_experts`, whose counts survive
//!   eviction (ghost history; EXP-005). No cross-layer prefetch: measured
//!   predictability is too low to pay for speculative reads.
//!
//! Concurrency invariant: a slot being filled by an in-flight read, or still
//! owned by queued compute, is never reassigned.
//!
//! # File access layer
//!
//! What exists today is the synchronous file-access layer beneath that
//! design:
//!
//! - [`MappedCommon`]: the read-only map of `common.bin`.
//! - [`ExpertReader`]: lazy per-layer file handles + positioned blob reads,
//!   returning [`ExpertView`]s over the gate/up/down slabs.
//! - [`LoadOptions`]: the load-time integrity dial.
//!
//! Integrity at load: `manifest.json` is the trust root — it carries the
//! digests but has no recorded digest of its own, so its gate is schema
//! validation ([`Manifest::validate`](crate::format::Manifest::validate)).
//! `common.bin` and `experts/layout.json` are hashed against the manifest
//! when the model opens (the ~1 GiB `common.bin` hash is an accepted
//! one-time cost); each `experts/layer_NN.bin` is hashed on first open.
//! [`LoadOptions::skip_hashes`] skips every SHA-256 check for fast dev
//! iteration; size checks always run.

mod cache;
mod common;
mod error;
mod expert;
mod slots;
#[cfg(test)]
pub(crate) mod testutil;

use std::fs;
use std::path::Path;

pub use cache::{CacheError, CachePlan, CacheStats, LayerCache, MAX_SLOTS, STUCK_PROTECTED_PLANS};
pub use common::MappedCommon;
pub use error::IoError;
pub use expert::{ExpertReader, ExpertSlab, ExpertView};
pub use slots::{MAX_POOL_BYTES, SLOT_ALIGN, SlotError, SlotGuard, SlotPool};

use crate::format::{FormatError, Manifest, sha256_file};
use crate::kernels::quants::QuantFormat;

/// Knobs for opening an installed model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoadOptions {
    /// Skip every SHA-256 integrity check (load-time metadata hashes and
    /// first-open layer-file hashes). Development flag only: size checks
    /// still run, but corruption goes undetected. Defaults to `false`.
    pub skip_hashes: bool,
}

/// Map a lowercase on-disk quant name (as the repacker writes it) to the
/// weight format the kernels compute. Returns `None` for anything that is
/// not a storable weight format — including `q8_k`, which is
/// activation-only and never on disk.
pub fn parse_quant_format(name: &str) -> Option<QuantFormat> {
    match name {
        "q4_k" => Some(QuantFormat::Q4_K),
        "q5_k" => Some(QuantFormat::Q5_K),
        "q6_k" => Some(QuantFormat::Q6_K),
        "q8_0" => Some(QuantFormat::Q8_0),
        _ => None,
    }
}

/// Check one manifest-listed file on disk: exact size always, streaming
/// SHA-256 unless [`LoadOptions::skip_hashes`] is set.
///
/// # Errors
///
/// [`IoError::Format`] wrapping the missing-entry, size-mismatch,
/// hash-mismatch, or I/O failure.
pub fn verify_named_file(
    dir: &Path,
    manifest: &Manifest,
    name: &str,
    options: LoadOptions,
) -> Result<(), IoError> {
    let entry = manifest
        .files
        .get(name)
        .ok_or_else(|| FormatError::MissingFileEntry(name.to_owned()))?;
    let path = dir.join(name);
    let metadata = fs::metadata(&path).map_err(|e| IoError::io(&path, e))?;
    if metadata.len() != entry.size {
        return Err(FormatError::SizeMismatch {
            name: name.to_owned(),
            expected: entry.size,
            actual: metadata.len(),
        }
        .into());
    }
    if !options.skip_hashes {
        let actual = sha256_file(&path)?;
        if !actual.eq_ignore_ascii_case(&entry.sha256) {
            return Err(FormatError::HashMismatch {
                name: name.to_owned(),
                expected: entry.sha256.clone(),
                actual,
            }
            .into());
        }
        tracing::debug!(file = name, bytes = entry.size, "file verified");
    }
    Ok(())
}

/// Narrow a metadata `u64` into `usize`, naming the value on failure.
pub(crate) fn to_usize(value: u64, what: &'static str) -> Result<usize, IoError> {
    usize::try_from(value).map_err(|_| IoError::TooLarge { what, value })
}
