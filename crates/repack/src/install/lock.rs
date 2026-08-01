//! Advisory per-target install lock.
//!
//! std exposes no `flock(2)` and the crate adds no new dependencies, so
//! mutual exclusion uses the portable O_EXCL pidfile pattern: a sibling
//! `<target>.lock` file created with `create_new` (O_CREAT|O_EXCL),
//! containing the holder's pid, removed on drop. A leftover lock whose pid
//! no longer exists in `/proc` is stale and gets broken. Limitations
//! accepted for an advisory single-machine lock: a recycled pid can make a
//! stale lock look live, and breaking a stale lock races other breakers
//! (bounded retries; `create_new` guarantees a single winner).

use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::{InstallError, io_err};

/// Suffix appended to the final directory name to form the lock file,
/// mirroring how `.partial` names the staging directory.
const LOCK_SUFFIX: &str = ".lock";

/// Create-attempts before giving up on a contended/thrashing lock.
const ACQUIRE_ATTEMPTS: u32 = 3;

/// Lock file guarding one install target: `<final_dir>.lock`, as a
/// sibling of `final_dir` (same construction as
/// [`ramvamp_core::format::partial_dir`]).
pub(crate) fn lock_path(final_dir: &Path) -> PathBuf {
    match final_dir.file_name() {
        Some(file_name) => {
            let mut name = file_name.to_os_string();
            name.push(LOCK_SUFFIX);
            final_dir.parent().unwrap_or(Path::new("")).join(name)
        }
        None => {
            let mut name = final_dir.as_os_str().to_os_string();
            name.push(LOCK_SUFFIX);
            PathBuf::from(name)
        }
    }
}

/// Held advisory lock on an install target. Serializes install, resume,
/// and discard on the same target; released (unlinked) on drop.
#[derive(Debug)]
pub(crate) struct InstallLock {
    path: PathBuf,
}

impl InstallLock {
    /// Acquire the lock for `final_dir`, breaking a stale lock (holder pid
    /// absent from `/proc`) if one is found.
    pub(crate) fn acquire(final_dir: &Path) -> Result<Self, InstallError> {
        let path = lock_path(final_dir);
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|e| io_err(parent, e))?;
        }
        let mut last_pid = 0u32;
        for _ in 0..ACQUIRE_ATTEMPTS {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    let pid = std::process::id();
                    file.write_all(pid.to_string().as_bytes())
                        .and_then(|()| file.sync_all())
                        .map_err(|e| io_err(&path, e))?;
                    tracing::debug!(lock = %path.display(), pid, "acquired install lock");
                    return Ok(InstallLock { path });
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    match holder_pid(&path) {
                        Some(pid) if pid_alive(pid) => {
                            return Err(InstallError::Locked { path, pid });
                        }
                        other => {
                            // Stale (dead pid) or unreadable/garbage lock:
                            // break it and retry create_new.
                            last_pid = other.unwrap_or(0);
                            tracing::warn!(
                                lock = %path.display(),
                                stale_pid = last_pid,
                                "breaking stale install lock"
                            );
                            let _ = fs::remove_file(&path);
                        }
                    }
                }
                Err(e) => return Err(io_err(&path, e)),
            }
        }
        Err(InstallError::Locked {
            path,
            pid: last_pid,
        })
    }
}

impl Drop for InstallLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Pid recorded in a lock file, if it parses.
fn holder_pid(path: &Path) -> Option<u32> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Whether a pid currently exists on this machine (Linux `/proc`).
fn pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}
