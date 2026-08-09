//! The O_DIRECT capability layer.
//!
//! Expert reads bypass the page cache or the memory contract does not hold:
//! the same 1.4 GiB of expert traffic peaked at 1,092.2 MiB of cgroup memory
//! buffered against 5.0 MiB with O_DIRECT (provisional, EXP-009). So this
//! module opens layer files with `O_DIRECT` — and then refuses to believe
//! that it worked.
//!
//! # Why the open is not evidence
//!
//! A successful `O_DIRECT` open is not a promise that reads bypass anything.
//! Confirmed ways to get a full byte count, no error, and a page cache that
//! grew anyway:
//!
//! - **Misalignment.** btrfs wants 4096 (its `sectorsize`) on the offset, the
//!   length *and* the destination address, not the device's 512-byte logical
//!   block size.
//! - **Compressed extents**, and **DUP/RAID data profiles**.
//! - **A destination whose pages are not faulted in.** btrfs runs direct
//!   reads with page faults disabled (`fs/btrfs/direct-io.c`) and completes
//!   the transfer through `filemap_read()` when it cannot fault the
//!   destination. [`SlotPool`](crate::io::SlotPool) pre-faults the whole slab
//!   for exactly this reason, and the probe below reads into a real slot so
//!   that the pre-fault is on the path being measured.
//! - **The filesystem lying by design.** tmpfs (since 6.6) and loop-backed
//!   filesystems accept the open and do buffered I/O.
//!
//! Nor can the requirement be asked for: `statx(STATX_DIOALIGN)` is
//! unimplemented on btrfs, f2fs and erofs answer with zeros, and NFS
//! fabricates an answer without asking the server.
//!
//! # So it is measured
//!
//! [`probe`] does one aligned read through the handle under test and checks
//! whether the page cache grew, with `posix_fadvise(POSIX_FADV_DONTNEED)` for
//! a clean baseline and `mincore` over a separate mapping of the same file as
//! the residency signal. Three syscall pairs and two 4096-byte reads, once at
//! startup.
//!
//! Two details are load-bearing:
//!
//! - **`mincore`, not `preadv2(RWF_NOWAIT)`.** The `RWF_NOWAIT` trick is
//!   cheaper and it is wrong here: measured on this project's own btrfs
//!   volume, a NOWAIT read returned a full 4096 bytes for a range `mincore`
//!   and `fincore(1)` both reported as *not* resident, i.e. it reported
//!   "cached" for data a genuine O_DIRECT read had not cached. False
//!   positives in that direction would fail an install that is in fact fine.
//! - **A positive control.** After concluding "the cache did not grow" the
//!   probe reads the same range *buffered* and requires the cache to grow.
//!   Without it, any host where the residency signal is unavailable —
//!   tmpfs answers `EOPNOTSUPP`, a container may block the mapping — would
//!   silently look like a clean O_DIRECT pass. If the control fails the probe
//!   reports [`DirectFault::NoResidencySignal`] and the caller degrades.
//!
//! # `mincore` does not answer for files this process does not own
//!
//! Since Linux 4.19 (`mm/mincore.c`, "make mincore() more conservative")
//! `mincore` reveals page-cache residency for a file-backed mapping only when
//! the caller passes `inode_owner_or_capable()` or could open the file for
//! writing. When it refuses it does **not** fail: it fills the vector with
//! `1`, reporting everything as resident. Ownership is what decides it, not
//! the mode bits — `chmod 444` on your own file still answers truthfully.
//!
//! That turns every deployment where the install is not owned by the running
//! user — a model unpacked into `/opt` by root, a bind-mount with a uid
//! mismatch, `ramvamp-repack` run under a different account — into a probe
//! that can never clear its baseline and always reports
//! [`DirectFault::DirtyBaseline`], naming the wrong cause for a filesystem
//! where O_DIRECT is working perfectly. So the condition is checked before the
//! measurement is attempted and reported as [`DirectFault::ResidencyDenied`].
//! The verdict is still a degradation — nothing here can prove the bypass
//! without a residency signal — but it names the real fault, and it is a
//! permission problem the operator can fix.
//!
//! Everything degrades: a machine with no working O_DIRECT still runs, with
//! a warning and a page-cache-charged budget.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;

use memmap2::MmapOptions;

/// Alignment the O_DIRECT path requires of the file offset, the transfer
/// length, and the destination address.
///
/// btrfs's `sectorsize`, which is also [`SLOT_ALIGN`](crate::io::SLOT_ALIGN)
/// and every Linux page size's common divisor. The device's 512-byte logical
/// block size is *not* sufficient and the real requirement cannot be probed
/// (`STATX_DIOALIGN` is unimplemented on btrfs), so it is fixed here.
pub const DIO_ALIGN: u64 = 4096;

/// `O_DIRECT` for the platforms whose value we hard-code, `None` elsewhere.
///
/// The constant is 0o0040000 on Linux/x86, x86-64, ARM, AArch64, RISC-V,
/// s390x and LoongArch, and something else on each of mips, powerpc, sparc
/// and alpha. Rather than encode values we cannot test, other targets get no
/// O_DIRECT and run buffered — they are outside the Linux/x86-64-first
/// support target anyway, and it is not read from `libc` because a wrong
/// constant would be a silent buffered fallback rather than an error. Other
/// operating systems have no `O_DIRECT` at all (macOS spells the idea
/// `F_NOCACHE`), so they take the same path.
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "loongarch64",
    )
))]
const O_DIRECT: Option<i32> = Some(0o0040000);

/// `O_DIRECT` is not hard-coded for this target; see the other arm.
#[cfg(not(all(
    target_os = "linux",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "loongarch64",
    )
)))]
const O_DIRECT: Option<i32> = None;

/// Why direct I/O is unavailable, or available but not bypassing anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectFault {
    /// This build does not know `O_DIRECT`'s value for this architecture.
    Unsupported,
    /// The kernel refused an `O_DIRECT` open — the filesystem does not
    /// support direct I/O at all.
    OpenRefused,
    /// The file is shorter than one aligned probe window.
    FileTooSmall,
    /// The layout's blob stride is not a multiple of [`DIO_ALIGN`], so expert
    /// reads would be misaligned in both offset and length. Nothing about the
    /// filesystem: the install's geometry rules direct I/O out.
    UnalignedGeometry,
    /// The probe buffer is too small, or its address is not [`DIO_ALIGN`]
    /// aligned. A caller bug: slot buffers are aligned by construction.
    BadProbeBuffer,
    /// The aligned probe read failed outright, so reads through this handle
    /// cannot be used at all.
    ReadFailed,
    /// The aligned probe read returned fewer bytes than asked for.
    ShortRead,
    /// `POSIX_FADV_DONTNEED` did not clear the probe window, so "the page
    /// cache grew" could not be distinguished from "it was already warm".
    DirtyBaseline,
    /// The read completed and the page cache grew: a silent buffered
    /// fallback, which is the outcome this whole module exists to catch.
    PageCacheGrew,
    /// The residency signal itself does not work here: a buffered read of the
    /// same range left `mincore` reporting nothing resident, so a clean
    /// result would have been meaningless.
    NoResidencySignal,
    /// The kernel will not reveal this file's page-cache residency to this
    /// process. Since Linux 4.19 `mincore` reports a file-backed range as
    /// *fully resident* — silently, without an error — unless the caller owns
    /// the inode or could open the file for writing, so no measurement over
    /// this install is meaningful. Says nothing about O_DIRECT: the install is
    /// simply owned by another user (root-owned `/opt`, a bind-mount with a
    /// uid mismatch, a repack run under a different account). Run as the
    /// owner of the install to get a verdict.
    ResidencyDenied,
    /// `mmap`, `mincore` or `posix_fadvise` failed during the probe.
    ProbeFailed,
}

impl fmt::Display for DirectFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::Unsupported => "O_DIRECT is not implemented for this architecture",
            Self::OpenRefused => "the filesystem refused an O_DIRECT open",
            Self::FileTooSmall => "the layer file is shorter than one aligned block",
            Self::UnalignedGeometry => "the layout's blob stride is not 4096-aligned",
            Self::BadProbeBuffer => "the probe buffer is misaligned or too small",
            Self::ReadFailed => "an aligned O_DIRECT read failed",
            Self::ShortRead => "an aligned O_DIRECT read came back short",
            Self::DirtyBaseline => "the page cache could not be cleared for the probe window",
            Self::PageCacheGrew => "the read populated the page cache: a silent buffered fallback",
            Self::NoResidencySignal => {
                "mincore reports nothing resident even after a buffered read"
            }
            Self::ResidencyDenied => {
                "mincore will not report page-cache residency for a file this \
                 process neither owns nor may write, so O_DIRECT cannot be \
                 measured on this install"
            }
            Self::ProbeFailed => "the probe's own mmap/mincore/fadvise failed",
        };
        f.write_str(text)
    }
}

/// What [`probe`] concluded about direct I/O on one install.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectSupport {
    /// An aligned read through the O_DIRECT handle left the page cache
    /// untouched, and a buffered read of the same range did not — so the
    /// signal works and the bypass is real.
    Verified,
    /// Reads work, but they are not bypassing the page cache (or it cannot
    /// be shown that they are). The handle stays usable; the budget does not
    /// hold.
    Degraded(DirectFault),
    /// The O_DIRECT handle cannot serve reads at all. The caller must reopen
    /// buffered.
    Unusable(DirectFault),
}

impl DirectSupport {
    /// Whether expert reads through this handle bypass the page cache.
    pub fn is_verified(self) -> bool {
        self == Self::Verified
    }

    /// Whether the O_DIRECT handle can serve reads, bypass or not.
    pub fn is_usable(self) -> bool {
        !matches!(self, Self::Unusable(_))
    }

    /// The fault behind a non-verified outcome.
    pub fn fault(self) -> Option<DirectFault> {
        match self {
            Self::Verified => None,
            Self::Degraded(fault) | Self::Unusable(fault) => Some(fault),
        }
    }
}

impl fmt::Display for DirectSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verified => f.write_str("verified"),
            Self::Degraded(fault) => write!(f, "degraded: {fault}"),
            Self::Unusable(fault) => write!(f, "unusable: {fault}"),
        }
    }
}

/// Open `path` read-only, with `O_DIRECT` when `direct` is set.
///
/// Returns the handle and whether `O_DIRECT` was actually applied: a
/// filesystem that refuses the flag (`EINVAL`) gets a second, buffered open
/// rather than an error, because a machine without direct I/O must still run.
///
/// # Errors
///
/// [`io::Error`] from the open itself, once the O_DIRECT retry has been
/// exhausted.
pub fn open(path: &Path, direct: bool) -> io::Result<(File, bool)> {
    if direct && let Some(flag) = O_DIRECT {
        match OpenOptions::new().read(true).custom_flags(flag).open(path) {
            Ok(file) => return Ok((file, true)),
            Err(error) => {
                tracing::debug!(
                    path = %path.display(),
                    %error,
                    "O_DIRECT open refused, retrying buffered"
                );
            }
        }
    }
    Ok((File::open(path)?, false))
}

/// Measure whether reads of `path` through an `O_DIRECT` handle populate the
/// page cache.
///
/// `buf` is the read destination and must be at least [`DIO_ALIGN`] bytes,
/// [`DIO_ALIGN`]-aligned, and already faulted in — pass a slice of a
/// [`SlotPool`](crate::io::SlotPool) slot, which is all three by construction
/// and is what real reads will land in. Its first [`DIO_ALIGN`] bytes are
/// overwritten with the head of the file.
///
/// Never fails: every way the measurement can go wrong is a
/// [`DirectSupport`] the caller can act on. The window sampled is the first
/// block of the file, which is one extent's worth of evidence — enough to
/// catch a filesystem-wide fallback (tmpfs, an unsupported open, a
/// compress-force mount), not enough to prove every extent of a multi-gigabyte
/// file is uncompressed.
pub fn probe(path: &Path, buf: &mut [u8]) -> DirectSupport {
    let window = DIO_ALIGN as usize;
    if buf.len() < window || !is_aligned(buf.as_ptr() as usize as u64) {
        return DirectSupport::Unusable(DirectFault::BadProbeBuffer);
    }
    let buf = &mut buf[..window];

    let Some(_) = O_DIRECT else {
        return DirectSupport::Unusable(DirectFault::Unsupported);
    };
    let (direct, applied) = match open(path, true) {
        Ok(opened) => opened,
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "cannot open layer file for probing");
            return DirectSupport::Unusable(DirectFault::OpenRefused);
        }
    };
    if !applied {
        return DirectSupport::Unusable(DirectFault::OpenRefused);
    }
    let Ok(buffered) = File::open(path) else {
        return DirectSupport::Unusable(DirectFault::OpenRefused);
    };
    match buffered.metadata() {
        Ok(meta) if meta.len() >= DIO_ALIGN => {}
        Ok(_) => return DirectSupport::Unusable(DirectFault::FileTooSmall),
        Err(_) => return DirectSupport::Unusable(DirectFault::ProbeFailed),
    }

    // Before any measurement: `mincore` answers "everything is resident" for
    // a file this process neither owns nor may write, which would make the
    // baseline below permanently dirty and blame the filesystem for a
    // permission fact. See the module docs.
    if !residency_is_visible(path, &buffered) {
        tracing::warn!(
            path = %path.display(),
            "the install is owned by another user, so the kernel will not \
             report its page-cache residency; O_DIRECT cannot be verified"
        );
        return DirectSupport::Degraded(DirectFault::ResidencyDenied);
    }

    // Baseline: nothing of the window may be resident, or "the cache grew"
    // cannot be told from "the cache was already warm". tmpfs stops here,
    // which is correct: on tmpfs the page cache *is* the file.
    match clear_window(&buffered, window) {
        Ok(0) => {}
        Ok(_) => return DirectSupport::Degraded(DirectFault::DirtyBaseline),
        Err(error) => {
            tracing::debug!(path = %path.display(), %error, "residency baseline failed");
            return DirectSupport::Degraded(DirectFault::ProbeFailed);
        }
    }

    // The read under test. Aligned offset, aligned length, aligned and
    // pre-faulted destination: everything btrfs asks for.
    match direct.read_at(buf, 0) {
        Ok(read) if read == window => {}
        Ok(read) => {
            tracing::debug!(path = %path.display(), read, "short O_DIRECT probe read");
            return DirectSupport::Unusable(DirectFault::ShortRead);
        }
        Err(error) => {
            tracing::debug!(path = %path.display(), %error, "O_DIRECT probe read failed");
            return DirectSupport::Unusable(DirectFault::ReadFailed);
        }
    }
    let grew = match resident_pages(&buffered, window) {
        Ok(pages) => pages,
        Err(_) => return DirectSupport::Degraded(DirectFault::ProbeFailed),
    };

    let verdict = if grew > 0 {
        DirectSupport::Degraded(DirectFault::PageCacheGrew)
    } else {
        // Positive control: a buffered read of the same range must show up,
        // or the clean result above proved nothing.
        let mut control = vec![0u8; window];
        match buffered.read_at(&mut control, 0) {
            Ok(read) if read == window => match resident_pages(&buffered, window) {
                Ok(0) => DirectSupport::Degraded(DirectFault::NoResidencySignal),
                Ok(_) => DirectSupport::Verified,
                Err(_) => DirectSupport::Degraded(DirectFault::ProbeFailed),
            },
            _ => DirectSupport::Degraded(DirectFault::ProbeFailed),
        }
    };
    // Leave the window as we found it; the control read above deliberately
    // dirtied it.
    let _ = fadvise_dontneed(&buffered, 0, DIO_ALIGN);
    verdict
}

/// Whether `value` satisfies the direct-I/O alignment requirement.
pub fn is_aligned(value: u64) -> bool {
    value.is_multiple_of(DIO_ALIGN)
}

/// Whether [`resident_pages`] will tell the truth about `file`.
///
/// `mincore` reveals page-cache residency for a file-backed mapping only when
/// `can_do_mincore()` passes, which since Linux 4.19 means
/// `inode_owner_or_capable() || file_permission(MAY_WRITE) == 0`. Otherwise it
/// reports the whole range resident and returns success, so a caller that does
/// not ask this question first cannot tell a warm cache from a refused answer.
///
/// This mirrors the kernel's test rather than probing for it, because there is
/// no range whose true residency is known in advance to compare against — the
/// refusal is indistinguishable from a genuinely warm file. Conservative in the
/// safe direction: a `false` here degrades the verdict, it never claims one.
#[cfg(target_os = "linux")]
fn residency_is_visible(path: &Path, file: &File) -> bool {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;

    // SAFETY: `geteuid` takes no arguments and cannot fail.
    let euid = unsafe { libc::geteuid() };
    // `inode_owner_or_capable()`: the owner, or anyone with CAP_FOWNER over
    // the inode, which root always has.
    if euid == 0 {
        return true;
    }
    if file.metadata().is_ok_and(|meta| meta.uid() == euid) {
        return true;
    }
    // Or write permission, checked against the effective ids the way the
    // kernel's own permission check is.
    let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: `c_path` is a live NUL-terminated string for the call's
    // duration and `faccessat` only reads it.
    let rc = unsafe {
        libc::faccessat(
            libc::AT_FDCWD,
            c_path.as_ptr(),
            libc::W_OK,
            libc::AT_EACCESS,
        )
    };
    rc == 0
}

/// Off Linux the probe short-circuits long before residency is consulted.
#[cfg(not(target_os = "linux"))]
fn residency_is_visible(_path: &Path, _file: &File) -> bool {
    false
}

/// Evict `file[..len]` from the page cache and report what is left.
///
/// `POSIX_FADV_DONTNEED` cannot drop a *dirty* page, and the model files are
/// dirty for as long as writeback takes right after an install — a real
/// window, since installing and then running is the obvious thing to do. So
/// a first failure is answered by writing the window back (`sync_file_range`
/// over 4096 bytes, not an `fsync` of a multi-gigabyte file) and trying
/// again. What cannot be cleared after that is not dirt: on tmpfs the page
/// *is* the file and never goes away, which is exactly the answer the probe
/// wants there.
///
/// # Errors
///
/// [`io::Error`] when the residency check itself fails; a failing `fadvise`
/// or writeback is not an error, it just leaves pages behind.
fn clear_window(file: &File, len: usize) -> io::Result<usize> {
    let _ = fadvise_dontneed(file, 0, len as u64);
    let resident = resident_pages(file, len)?;
    if resident == 0 {
        return Ok(0);
    }
    if writeback(file, 0, len as u64).is_err() {
        return Ok(resident);
    }
    let _ = fadvise_dontneed(file, 0, len as u64);
    resident_pages(file, len)
}

/// Write `offset..offset + len` of `file` back, so its pages become clean and
/// droppable.
///
/// # Errors
///
/// [`io::Error`] from `sync_file_range`, including on platforms without it.
#[cfg(target_os = "linux")]
fn writeback(file: &File, offset: u64, len: u64) -> io::Result<()> {
    let (Ok(offset), Ok(len)) = (i64::try_from(offset), i64::try_from(len)) else {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    };
    // SAFETY: `file` keeps the descriptor open across the call, which takes
    // no pointers. A read-only descriptor is accepted: this asks the kernel
    // to flush its own dirty pages, not to write through the handle.
    let rc = unsafe {
        libc::sync_file_range(
            file.as_raw_fd(),
            offset,
            len,
            libc::SYNC_FILE_RANGE_WAIT_BEFORE
                | libc::SYNC_FILE_RANGE_WRITE
                | libc::SYNC_FILE_RANGE_WAIT_AFTER,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// `sync_file_range` is Linux-only; elsewhere there is nothing to try.
#[cfg(not(target_os = "linux"))]
fn writeback(_file: &File, _offset: u64, _len: u64) -> io::Result<()> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

/// Drop `offset..offset + len` of `file` from the page cache.
///
/// Advisory: the kernel ignores this for dirty or otherwise pinned pages,
/// which is why the caller checks the result with [`resident_pages`] rather
/// than trusting it.
///
/// # Errors
///
/// [`io::Error`] carrying `posix_fadvise`'s errno, which the call returns
/// directly rather than through `errno`.
#[cfg(target_os = "linux")]
pub fn fadvise_dontneed(file: &File, offset: u64, len: u64) -> io::Result<()> {
    let (Ok(offset), Ok(len)) = (i64::try_from(offset), i64::try_from(len)) else {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    };
    // SAFETY: `file` keeps the descriptor open across the call, and
    // `posix_fadvise` only reads the arguments. Its return value *is* the
    // errno; it does not set `errno` or return -1.
    let rc = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        )
    };
    if rc != 0 {
        return Err(io::Error::from_raw_os_error(rc));
    }
    Ok(())
}

/// `posix_fadvise` is not portable enough to rely on off Linux, and nothing
/// here needs it there: without `O_DIRECT` the whole probe short-circuits.
#[cfg(not(target_os = "linux"))]
pub fn fadvise_dontneed(_file: &File, _offset: u64, _len: u64) -> io::Result<()> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

/// Number of pages of `file[..len]` currently resident in the page cache.
///
/// `mincore` over a fresh private mapping of the range, which is what
/// `fincore(1)` does. Mapping the range does not read it: an untouched
/// mapping holds no page references, so this neither warms the cache nor
/// stops [`fadvise_dontneed`] from clearing it.
///
/// **Only meaningful when [`residency_is_visible`] holds.** For a file the
/// caller neither owns nor may write, `mincore` reports every page resident
/// and returns success rather than refusing, so this answers `len / page_size`
/// no matter what the cache actually holds.
///
/// # Errors
///
/// [`io::Error`] when the mapping or the `mincore` call fails — including
/// filesystems where `mincore` is unsupported.
#[cfg(target_os = "linux")]
pub fn resident_pages(file: &File, len: usize) -> io::Result<usize> {
    if len == 0 {
        return Ok(0);
    }
    // SAFETY: the mapping is private and read-only and is never written
    // through; it is dropped at the end of this function. The install
    // directory is treated as immutable while the model is open, the same
    // contract `MappedCommon` relies on.
    let map = unsafe { MmapOptions::new().len(len).map(file)? };
    let pages = len.div_ceil(page_size());
    let mut resident = vec![0u8; pages];
    // SAFETY: `map` is `len` readable bytes at a page-aligned address (mmap
    // returns page-aligned addresses and no offset was applied), and
    // `resident` has one byte per page of that range, which is exactly what
    // `mincore` writes.
    let rc = unsafe {
        libc::mincore(
            map.as_ptr() as *mut libc::c_void,
            len,
            resident.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(resident.iter().filter(|byte| *byte & 1 == 1).count())
}

/// `mincore`'s signature and semantics vary off Linux, and the probe never
/// gets this far there.
#[cfg(not(target_os = "linux"))]
pub fn resident_pages(_file: &File, _len: usize) -> io::Result<usize> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

/// This host's page size, or 4096 if the kernel will not say.
fn page_size() -> usize {
    // SAFETY: `sysconf` takes an integer and returns one; no pointers.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if size > 0 {
        size as usize
    } else {
        DIO_ALIGN as usize
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::build_install;
    use super::*;
    use crate::io::SlotPool;
    use std::path::PathBuf;

    /// A 4096-aligned, pre-faulted probe buffer, from the real slot pool so
    /// that the probe measures what expert reads will actually do. Two pages
    /// wide, so that a misaligned-but-long-enough slice can be carved out of
    /// it.
    struct Scratch {
        pool: SlotPool,
    }

    impl Scratch {
        fn new() -> Self {
            Self {
                pool: SlotPool::new(1, &[2 * DIO_ALIGN]).expect("two-page pool"),
            }
        }

        fn with<R>(&self, f: impl FnOnce(&mut [u8]) -> R) -> R {
            let mut slot = self.pool.acquire(0).expect("free slot");
            f(slot.as_mut_slice())
        }
    }

    /// A directory that is definitely tmpfs on Linux, or `None`.
    fn shm_dir() -> Option<PathBuf> {
        let dir = PathBuf::from("/dev/shm").join(format!("ramvamp-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok().map(|()| dir)
    }

    /// Whether an outcome implies the probe got as far as its O_DIRECT read.
    fn read_happened(outcome: DirectSupport) -> bool {
        matches!(
            outcome,
            DirectSupport::Verified
                | DirectSupport::Degraded(
                    DirectFault::PageCacheGrew | DirectFault::NoResidencySignal
                )
        )
    }

    #[test]
    fn probe_reads_the_head_of_the_file_whatever_it_concludes() {
        let fx = build_install("direct-probe");
        let path = fx.root.join(&fx.layout.layers[0].file);
        let expected = std::fs::read(&path).unwrap();
        let scratch = Scratch::new();
        let (outcome, got) = scratch.with(|buf| (probe(&path, buf), buf.to_vec()));
        println!("probe outcome on the test filesystem: {outcome}");

        // The host is allowed to have no direct I/O at all. What it is not
        // allowed to do is fail an aligned read of an aligned file into an
        // aligned, pre-faulted buffer: that would be our geometry being
        // wrong, not the filesystem's business.
        assert!(
            !matches!(
                outcome.fault(),
                Some(DirectFault::ReadFailed | DirectFault::ShortRead)
            ),
            "aligned O_DIRECT read failed: {outcome}"
        );
        if read_happened(outcome) {
            let window = DIO_ALIGN as usize;
            assert_eq!(
                &got[..window],
                &expected[..window],
                "probe read wrong bytes"
            );
        }
        // A verified outcome carries no fault and vice versa.
        assert_eq!(outcome.is_verified(), outcome.fault().is_none());
    }

    #[test]
    fn probe_separates_a_compressed_extent_from_a_plain_one() {
        // Not a filesystem test with a fixed answer — it is the same probe
        // over two files that differ only in how well they compress, on
        // whatever filesystem the fixtures land on. On a `compress=` btrfs
        // mount the compressible one is stored as `encoded` extents and its
        // O_DIRECT reads silently fall back to `filemap_read` (measured: one
        // 4096-byte direct read left a whole 128 KiB extent in the page
        // cache), while the incompressible one bypasses cleanly. Quantized
        // expert weights are the incompressible case; the fixture's pattern
        // bytes are the other one.
        let fx = build_install("direct-compressible");
        let path = fx.root.join("incompressible.bin");
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut bytes = Vec::with_capacity(1 << 20);
        while bytes.len() < (1 << 20) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.extend_from_slice(&state.to_le_bytes());
        }
        {
            use std::io::Write;
            let mut file = File::create(&path).unwrap();
            file.write_all(&bytes).unwrap();
            file.sync_all().unwrap();
        }
        let scratch = Scratch::new();
        let plain = scratch.with(|buf| probe(&path, buf));
        let packed = scratch.with(|buf| probe(&fx.root.join(&fx.layout.layers[0].file), buf));
        println!("probe: incompressible={plain} compressible={packed}");

        for outcome in [plain, packed] {
            assert!(
                !matches!(
                    outcome.fault(),
                    Some(DirectFault::ReadFailed | DirectFault::ShortRead)
                ),
                "aligned O_DIRECT read failed: {outcome}"
            );
        }
        // Whatever the host does, the two must not disagree in the direction
        // that would mean the probe is noise: a filesystem that bypasses the
        // page cache for compressible data must do so for incompressible data
        // too.
        if packed.is_verified() {
            assert!(plain.is_verified(), "verified only for compressible data");
        }
    }

    #[test]
    fn probe_refuses_to_claim_direct_io_on_tmpfs() {
        let Some(dir) = shm_dir() else {
            return; // no /dev/shm (not Linux, or a locked-down container)
        };
        let path = dir.join("probe.bin");
        std::fs::write(&path, vec![7u8; 64 * 1024]).unwrap();
        let scratch = Scratch::new();
        let outcome = scratch.with(|buf| probe(&path, buf));
        println!("probe outcome on tmpfs: {outcome}");
        // tmpfs accepts the O_DIRECT open and then does buffered I/O. It must
        // not come back verified, by whichever route it fails: the pages are
        // the page cache, so the baseline cannot be cleared, and `mincore`
        // there answers EOPNOTSUPP.
        assert!(!outcome.is_verified(), "tmpfs claimed real O_DIRECT");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn probe_rejects_a_misaligned_or_short_buffer() {
        let fx = build_install("direct-badbuf");
        let path = fx.root.join(&fx.layout.layers[0].file);
        let scratch = Scratch::new();
        let short = scratch.with(|buf| probe(&path, &mut buf[..8]));
        assert_eq!(
            short,
            DirectSupport::Unusable(DirectFault::BadProbeBuffer),
            "a short buffer must be rejected"
        );
        let misaligned = scratch.with(|buf| probe(&path, &mut buf[8..]));
        assert_eq!(
            misaligned,
            DirectSupport::Unusable(DirectFault::BadProbeBuffer),
            "a misaligned buffer must be rejected"
        );
    }

    #[test]
    fn probe_reports_a_missing_file_rather_than_failing() {
        let scratch = Scratch::new();
        let outcome = scratch.with(|buf| probe(Path::new("/nonexistent/layer_00.bin"), buf));
        assert!(!outcome.is_usable());
        assert_eq!(outcome.fault(), Some(DirectFault::OpenRefused));
    }

    #[test]
    fn residency_tracks_a_buffered_read() {
        let fx = build_install("direct-residency");
        let path = fx.root.join(&fx.layout.layers[0].file);
        let file = File::open(&path).unwrap();
        let window = DIO_ALIGN as usize;
        if fadvise_dontneed(&file, 0, DIO_ALIGN).is_err() {
            return;
        }
        let Ok(before) = resident_pages(&file, window) else {
            return; // no residency signal on this filesystem (tmpfs)
        };
        let mut buf = vec![0u8; window];
        file.read_at(&mut buf, 0).unwrap();
        let after = resident_pages(&file, window).unwrap();
        // A buffered read is exactly the thing the cache must notice. If the
        // baseline could not be cleared the check is vacuous, so it is only
        // asserted when the baseline was clean.
        if before == 0 {
            assert!(after > 0, "buffered read left nothing resident");
        }
    }

    /// A file that exists, is big enough to probe, and belongs to somebody
    /// else — the shape every root-installed model has. `None` when the test
    /// is running as root (which passes `inode_owner_or_capable` for every
    /// inode) or when no such file is reachable.
    #[cfg(target_os = "linux")]
    fn foreign_file() -> Option<PathBuf> {
        use std::os::unix::fs::MetadataExt;

        // SAFETY: `geteuid` takes no arguments and cannot fail.
        if unsafe { libc::geteuid() } == 0 {
            return None;
        }
        [
            "/usr/lib/libc.so.6",
            "/usr/lib/x86_64-linux-gnu/libc.so.6",
            "/usr/bin/env",
            "/bin/sh",
        ]
        .into_iter()
        .map(PathBuf::from)
        .find(|path| {
            let Ok(meta) = std::fs::metadata(path) else {
                return false;
            };
            // SAFETY: as above.
            meta.is_file()
                && meta.len() >= DIO_ALIGN
                && meta.uid() != unsafe { libc::geteuid() }
                && File::open(path).is_ok_and(|file| !residency_is_visible(path, &file))
        })
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn residency_is_visible_for_a_file_this_process_owns() {
        // The kernel's rule is ownership, not the mode bits: a read-only file
        // of our own still answers.
        let fx = build_install("direct-residency-owner");
        let path = fx.root.join(&fx.layout.layers[0].file);
        let file = File::open(&path).unwrap();
        assert!(
            residency_is_visible(&path, &file),
            "our own install must have a residency signal"
        );
        let mut mode = std::fs::metadata(&path).unwrap().permissions();
        mode.set_readonly(true);
        std::fs::set_permissions(&path, mode).unwrap();
        assert!(
            residency_is_visible(&path, &file),
            "chmod 444 on our own file must not hide residency"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn probe_names_the_permission_fault_on_a_file_owned_by_someone_else() {
        // The deployment shape no fixture can reproduce: the install belongs
        // to root and the runtime does not. `mincore` then reports the whole
        // window resident without erroring, which used to surface as
        // `DirtyBaseline` — the filesystem blamed for a permission fact.
        let Some(path) = foreign_file() else {
            eprintln!("skipping: no foreign-owned probe target (running as root?)");
            return;
        };
        let scratch = Scratch::new();
        let outcome = scratch.with(|buf| probe(&path, buf));
        println!("probe outcome on {}: {outcome}", path.display());
        assert_eq!(
            outcome,
            DirectSupport::Degraded(DirectFault::ResidencyDenied),
            "a file we do not own must report the permission fault, not a \
             filesystem one"
        );
        // And the reason it matters: the raw signal really does lie here.
        let file = File::open(&path).unwrap();
        let _ = fadvise_dontneed(&file, 0, DIO_ALIGN);
        let pages = resident_pages(&file, DIO_ALIGN as usize).unwrap();
        assert_eq!(
            pages,
            (DIO_ALIGN as usize).div_ceil(page_size()),
            "mincore was expected to claim full residency for a foreign file"
        );
    }

    #[test]
    fn alignment_helper_matches_the_documented_requirement() {
        assert!(is_aligned(0));
        assert!(is_aligned(DIO_ALIGN));
        assert!(is_aligned(3_059_712)); // the audited Qwen3 Q6_K-down stride
        assert!(is_aligned(2_654_208)); // and the other one
        assert!(!is_aligned(512)); // the device block size is not enough
        assert!(!is_aligned(DIO_ALIGN + 1));
    }

    #[test]
    fn open_applies_o_direct_only_when_asked() {
        let fx = build_install("direct-open");
        let path = fx.root.join(&fx.layout.layers[0].file);
        let (buffered, applied) = open(&path, false).unwrap();
        assert!(!applied, "O_DIRECT applied when it was not requested");
        // A buffered handle takes an unaligned read; that is the point of it.
        let mut small = vec![0u8; 7];
        buffered.read_at(&mut small, 3).unwrap();

        let (direct, applied) = open(&path, true).unwrap();
        assert_eq!(
            direct.metadata().unwrap().len(),
            buffered.metadata().unwrap().len()
        );
        if applied {
            // An aligned read must work through the direct handle; the
            // destination alignment is the caller's job, so use a slot.
            let scratch = Scratch::new();
            let read = scratch.with(|buf| direct.read_at(&mut buf[..DIO_ALIGN as usize], 0));
            assert_eq!(read.unwrap(), DIO_ALIGN as usize);
        }
    }
}
