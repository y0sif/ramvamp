//! Resume state for an in-progress install: `install-state.json` inside
//! the partial directory.
//!
//! The state file pins the identity of the transfer — source URL, source
//! size, and a fingerprint of the sorted copy map — plus the window size
//! the transfer was started with, how many windows are fully durable on
//! disk, and one SHA-256 digest per durable window over the source bytes
//! that window actually copied. A resume must match the identity exactly;
//! anything else requires a restart (`--overwrite`). The file is removed
//! before promotion so it never appears in a finished install.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{InstallError, hex, io_err};
use crate::plan::CopyOp;

/// Name of the resume-state file inside the partial directory.
pub const STATE_FILE: &str = "install-state.json";

/// Schema version this build reads and writes.
pub const STATE_VERSION: u32 = 1;

/// Parse cap for the state file. A 1 TiB source at 32 MiB windows needs a
/// few MiB of digests; the cap bounds a hostile file's allocation.
const MAX_STATE_BYTES: u64 = 16 * 1024 * 1024;

/// Durable progress record for a windowed install.
///
/// Invariant maintained by the executor: `window_digests.len() ==
/// durable_windows`, and every window below `durable_windows` has been
/// written to its destination files *and* fsync'd before the state file
/// recording it was itself durably written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstallState {
    /// Schema version; must equal [`STATE_VERSION`].
    pub state_version: u32,
    /// Identity of the source: resolved URL for remote sources, canonical
    /// path for local ones.
    pub source_url: String,
    /// Total size of the source in bytes at install start.
    pub source_len: u64,
    /// SHA-256 hex over the serialized sorted copy map
    /// ([`plan_fingerprint`]); pins the plan geometry exactly.
    pub plan_fingerprint: String,
    /// Window size in bytes the transfer was started with. Window
    /// boundaries (and therefore per-window digests) depend on it, so a
    /// resume always adopts this value.
    pub window_bytes: u64,
    /// Windows `0..durable_windows` are fully written and fsync'd.
    pub durable_windows: u64,
    /// Per-window SHA-256 hex digest over the source bytes the window
    /// copied: the copy-op overlap spans in ascending source order,
    /// excluding alignment padding the plan never copies. Length equals
    /// `durable_windows`.
    pub window_digests: Vec<String>,
}

/// Fingerprint of a sorted copy map: SHA-256 over every op's source range,
/// destination offset, and destination file name, in order. Two plans with
/// the same fingerprint describe byte-identical transfers.
pub(crate) fn plan_fingerprint(ops: &[CopyOp]) -> String {
    let mut hasher = Sha256::new();
    for op in ops {
        hasher.update(op.src.start.to_le_bytes());
        hasher.update(op.src.end.to_le_bytes());
        hasher.update(op.dst_offset.to_le_bytes());
        hasher.update((op.dst_file.len() as u64).to_le_bytes());
        hasher.update(op.dst_file.as_bytes());
    }
    hex(&hasher.finalize())
}

/// Load and structurally check `<partial>/install-state.json`.
///
/// A missing file maps to a [`InstallError::StateMismatch`] with guidance
/// (the install crashed before its first checkpoint, or the directory is
/// not one of ours); identity checks against the current plan are the
/// caller's job.
pub(crate) fn load_state(partial: &Path) -> Result<InstallState, InstallError> {
    let path = partial.join(STATE_FILE);
    let file = match fs::File::open(&path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(InstallError::StateMismatch(format!(
                "{} is missing (crashed before the first checkpoint, or not a ramvamp \
                 partial install); restart with --overwrite",
                path.display()
            )));
        }
        Err(e) => return Err(io_err(&path, e)),
    };
    let mut data = Vec::new();
    file.take(MAX_STATE_BYTES + 1)
        .read_to_end(&mut data)
        .map_err(|e| io_err(&path, e))?;
    if data.len() as u64 > MAX_STATE_BYTES {
        return Err(InstallError::StateMismatch(format!(
            "{} exceeds the {MAX_STATE_BYTES}-byte cap",
            path.display()
        )));
    }
    let state: InstallState = serde_json::from_slice(&data).map_err(|e| {
        InstallError::StateMismatch(format!("{} does not parse: {e}", path.display()))
    })?;
    if state.state_version != STATE_VERSION {
        return Err(InstallError::StateMismatch(format!(
            "state_version {} (this build supports {STATE_VERSION})",
            state.state_version
        )));
    }
    if state.window_bytes == 0 {
        return Err(InstallError::StateMismatch(
            "window_bytes is zero".to_owned(),
        ));
    }
    if state.window_digests.len() as u64 != state.durable_windows {
        return Err(InstallError::StateMismatch(format!(
            "{} window digests recorded for {} durable windows",
            state.window_digests.len(),
            state.durable_windows
        )));
    }
    Ok(state)
}

/// Durably write the state file: temp file, fsync, rename over the
/// destination, fsync the directory. Mirrors the manifest writer's
/// discipline so a crash never leaves a torn state file.
pub(crate) fn write_state(partial: &Path, state: &InstallState) -> Result<(), InstallError> {
    let path = partial.join(STATE_FILE);
    let json = serde_json::to_vec_pretty(state).map_err(|e| {
        // Serializing our own struct cannot fail in practice; typed anyway.
        InstallError::StateMismatch(format!("serializing install state: {e}"))
    })?;
    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(".tmp");
    let tmp = PathBuf::from(tmp_name);
    {
        let mut file = fs::File::create(&tmp).map_err(|e| io_err(&tmp, e))?;
        file.write_all(&json).map_err(|e| io_err(&tmp, e))?;
        file.sync_all().map_err(|e| io_err(&tmp, e))?;
    }
    fs::rename(&tmp, &path).map_err(|e| io_err(&tmp, e))?;
    let dir = fs::File::open(partial).map_err(|e| io_err(partial, e))?;
    dir.sync_all().map_err(|e| io_err(partial, e))?;
    Ok(())
}
