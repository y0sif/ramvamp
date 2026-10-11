//! Read-only mapping of `common.bin`.
//!
//! The common core (embeddings, lm_head, attention, routers, norms) is
//! touched every token. By default it is `mmap`'d read-only and the page
//! cache keeps it resident, for as long as nothing else needs the memory;
//! with [`CommonMemory::Resident`] it is read once into anonymous memory
//! instead, which nothing but swap can take away. Integrity is checked
//! before the bytes are handed out: the file's size must match the manifest
//! entry, and unless hashes are skipped its SHA-256 must too (~1 GiB for the
//! v0 model — a one-time load cost the design accepts).

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::format::{COMMON_FILE, CommonTensor, Manifest};

use super::direct::{DIO_ALIGN, O_DIRECT};
use super::{CommonMemory, IoError, LoadOptions, to_usize, verify_named_file};

/// Bytes per read when loading [`CommonMemory::Resident`]: large enough to
/// keep the drive streaming, a multiple of [`DIO_ALIGN`] as `O_DIRECT`
/// requires.
const RESIDENT_CHUNK: usize = 16 << 20;

/// Read-only memory map of an install's `common.bin`.
///
/// The mapping is private and read-only; nothing here ever writes through
/// it. The install directory is treated as immutable while the model is
/// open — truncating the file underneath a live map would fault, which is
/// the standard mmap contract, not something this layer can defend against.
#[derive(Debug)]
pub struct MappedCommon {
    /// The file's bytes: the file mapping itself, or an anonymous region
    /// read from it and then made read-only. The anonymous region is rounded
    /// up to [`DIO_ALIGN`], so only the first `len` bytes are the file.
    map: Mmap,
    len: usize,
    memory: CommonMemory,
}

impl MappedCommon {
    /// Map `<dir>/common.bin` after checking it against the manifest.
    ///
    /// The size check always runs; the SHA-256 check is skipped when
    /// [`LoadOptions::skip_hashes`] is set.
    ///
    /// # Errors
    ///
    /// [`IoError::Format`] when the manifest has no `common.bin` entry or
    /// the file fails its size/hash check; [`IoError::Io`] when opening or
    /// mapping fails.
    pub fn open(dir: &Path, manifest: &Manifest, options: LoadOptions) -> Result<Self, IoError> {
        verify_named_file(dir, manifest, COMMON_FILE, options)?;
        let path = dir.join(COMMON_FILE);
        let file = File::open(&path).map_err(|e| IoError::io(&path, e))?;
        let len = file.metadata().map_err(|e| IoError::io(&path, e))?.len();
        let len = to_usize(len, "common.bin length")?;
        let map = match options.common_memory {
            // SAFETY: the map is read-only and private. The install
            // directory is treated as immutable while open (see the struct
            // docs); the size was just verified against the manifest.
            CommonMemory::Mapped => unsafe { Mmap::map(&file) },
            CommonMemory::Resident => read_resident(&path, &file, len),
        }
        .map_err(|e| IoError::io(&path, e))?;
        tracing::debug!(
            path = %path.display(),
            bytes = len,
            memory = ?options.common_memory,
            "loaded common weights"
        );
        Ok(Self {
            map,
            len,
            memory: options.common_memory,
        })
    }

    /// The whole file as a byte slice.
    pub fn bytes(&self) -> &[u8] {
        &self.map[..self.len]
    }

    /// Length of the file in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the file is empty (never true for a valid install).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How the bytes are held.
    pub fn memory(&self) -> CommonMemory {
        self.memory
    }

    /// Bounds-checked byte slice of one tensor from the manifest index.
    ///
    /// # Errors
    ///
    /// [`IoError::RangeOutOfBounds`] when the entry's range does not fit
    /// the mapping (a manifest/file inconsistency the validator should
    /// have caught); [`IoError::TooLarge`] on a 32-bit-overflowing offset.
    pub fn tensor(&self, name: &str, entry: &CommonTensor) -> Result<&[u8], IoError> {
        let offset = to_usize(entry.offset, "common tensor offset")?;
        let len = to_usize(entry.len, "common tensor len")?;
        self.bytes()
            .get(offset..offset.saturating_add(len))
            .ok_or_else(|| IoError::RangeOutOfBounds {
                what: format!("common tensor {name:?}"),
                offset: entry.offset,
                len: entry.len,
                available: self.len as u64,
            })
    }
}

/// Read `len` bytes of `path` into a fresh anonymous region and make it
/// read-only.
///
/// The region comes from `mmap(MAP_ANONYMOUS)`, so it is page-aligned, which
/// is what `O_DIRECT` needs of the buffer, and its length is rounded up to
/// [`DIO_ALIGN`], so every read is a whole number of aligned units even at
/// the end of the file. Where the file system refuses `O_DIRECT` (tmpfs, for
/// one) the read falls back to a buffered one.
///
/// **`O_DIRECT` alone does not keep the file out of the page cache.** btrfs
/// serves a direct read of a *compressed* extent through the page cache, and
/// the installed `common.bin` is mostly compressed extents (430 of them on the
/// reference install; the expert files have none). Without help, loading
/// would leave the whole file cached beside its anonymous copy, twice the
/// footprint, which in a 3 GB cgroup is the difference between fitting and
/// reclaiming (EXP-031). So every chunk is dropped from the page cache right
/// after it is read, whichever path read it, and the transient cost is one
/// chunk.
fn read_resident(path: &Path, buffered: &File, len: usize) -> std::io::Result<Mmap> {
    let align = DIO_ALIGN as usize;
    let rounded = len.div_ceil(align).max(1) * align;
    let mut region = MmapOptions::new().len(rounded).map_anon()?;
    let direct = O_DIRECT.and_then(|flag| {
        OpenOptions::new()
            .read(true)
            .custom_flags(flag)
            .open(path)
            .ok()
    });
    let mut filled = 0usize;
    if let Some(file) = direct {
        while filled < len {
            let end = (filled + RESIDENT_CHUNK).min(rounded);
            match file.read_at(&mut region[filled..end], filled as u64) {
                Ok(0) => break,
                // A short read is only legitimate at the end of the file;
                // anywhere else the next aligned read picks up from it.
                Ok(n) => {
                    drop_cached(buffered, filled, n);
                    filled += n;
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                // The open succeeded but this file system will not do the
                // transfer: finish buffered from where we are.
                Err(e) if e.raw_os_error() == Some(libc::EINVAL) => break,
                Err(e) => return Err(e),
            }
        }
    }
    while filled < len {
        let end = (filled + RESIDENT_CHUNK).min(len);
        let mut file = buffered;
        std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(filled as u64))?;
        file.read_exact(&mut region[filled..end])?;
        drop_cached(buffered, filled, end - filled);
        filled = end;
    }
    region.make_read_only()
}

/// Ask the kernel to drop `len` bytes at `offset` of `file` from the page
/// cache. Best effort: `POSIX_FADV_DONTNEED` only drops clean pages nothing
/// maps, which is all a just-read range of an unmapped file is, and a
/// failure costs memory, not correctness.
fn drop_cached(file: &File, offset: usize, len: usize) {
    // SAFETY: a plain syscall on a valid descriptor, with no pointers.
    unsafe {
        libc::posix_fadvise(
            std::os::fd::AsRawFd::as_raw_fd(file),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::build_install;
    use super::*;
    use crate::format::FormatError;

    #[test]
    fn maps_and_slices_common() {
        let fx = build_install("mapped-common");
        let common = MappedCommon::open(&fx.root, &fx.manifest, LoadOptions::default()).unwrap();
        assert_eq!(common.len() as u64, fx.manifest.files[COMMON_FILE].size);
        assert!(!common.is_empty());
        let entry = &fx.manifest.common_tensors["output_norm.weight"];
        let bytes = common.tensor("output_norm.weight", entry).unwrap();
        assert_eq!(bytes.len() as u64, entry.len);
        let whole = common.bytes();
        let offset = entry.offset as usize;
        assert_eq!(&whole[offset..offset + bytes.len()], bytes);
    }

    /// Resident and mapped hand out the same bytes and the same tensors. The
    /// fixture's `common.bin` is not a multiple of the 4 KiB alignment, so the
    /// rounded tail is exercised; under a tmpfs temp dir the `O_DIRECT` open
    /// is refused and the buffered fallback runs instead.
    #[test]
    fn resident_and_mapped_hold_the_same_bytes() {
        let fx = build_install("resident-common");
        let mapped = MappedCommon::open(&fx.root, &fx.manifest, LoadOptions::default()).unwrap();
        let resident = MappedCommon::open(
            &fx.root,
            &fx.manifest,
            LoadOptions {
                common_memory: CommonMemory::Resident,
                ..LoadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(resident.memory(), CommonMemory::Resident);
        assert_eq!(mapped.memory(), CommonMemory::Mapped);
        assert_eq!(resident.len(), mapped.len());
        assert_eq!(resident.bytes(), mapped.bytes());
        for (name, entry) in &fx.manifest.common_tensors {
            assert_eq!(
                resident.tensor(name, entry).unwrap(),
                mapped.tensor(name, entry).unwrap(),
                "{name}"
            );
        }
    }

    /// Reading through `O_DIRECT` on a real file system, where the temp dir
    /// supports it: the same bytes as a plain read, for lengths on and off the
    /// alignment.
    #[test]
    fn read_resident_matches_a_plain_read_at_awkward_lengths() {
        let dir = std::env::current_dir()
            .unwrap()
            .join("target")
            .join("resident-read-test");
        std::fs::create_dir_all(&dir).unwrap();
        for len in [1usize, 4095, 4096, 4097, (16 << 20) + 513] {
            let path = dir.join(format!("blob-{len}"));
            let data: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
            std::fs::write(&path, &data).unwrap();
            let file = File::open(&path).unwrap();
            let region = read_resident(&path, &file, len).unwrap();
            assert_eq!(&region[..len], &data[..], "len {len}");
            std::fs::remove_file(&path).unwrap();
        }
    }

    #[test]
    fn rejects_corrupt_common_unless_hashes_skipped() {
        let fx = build_install("mapped-corrupt");
        let path = fx.root.join(COMMON_FILE);
        let mut data = std::fs::read(&path).unwrap();
        data[10] ^= 0xff;
        std::fs::write(&path, data).unwrap();

        let err = MappedCommon::open(&fx.root, &fx.manifest, LoadOptions::default()).unwrap_err();
        assert!(matches!(
            err,
            IoError::Format(FormatError::HashMismatch { name, .. }) if name == COMMON_FILE
        ));
        MappedCommon::open(
            &fx.root,
            &fx.manifest,
            LoadOptions {
                skip_hashes: true,
                ..LoadOptions::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn size_check_survives_skip_hashes() {
        let fx = build_install("mapped-truncated");
        let path = fx.root.join(COMMON_FILE);
        let data = std::fs::read(&path).unwrap();
        std::fs::write(&path, &data[..data.len() - 64]).unwrap();
        let err = MappedCommon::open(
            &fx.root,
            &fx.manifest,
            LoadOptions {
                skip_hashes: true,
                ..LoadOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            IoError::Format(FormatError::SizeMismatch { name, .. }) if name == COMMON_FILE
        ));
    }

    #[test]
    fn tensor_range_is_bounds_checked() {
        let fx = build_install("mapped-bounds");
        let common = MappedCommon::open(&fx.root, &fx.manifest, LoadOptions::default()).unwrap();
        let bogus = CommonTensor {
            offset: common.len() as u64,
            len: 64,
            dtype: "f32".to_owned(),
        };
        assert!(matches!(
            common.tensor("bogus", &bogus).unwrap_err(),
            IoError::RangeOutOfBounds { .. }
        ));
    }
}
