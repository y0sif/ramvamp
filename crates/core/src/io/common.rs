//! Read-only mapping of `common.bin`.
//!
//! The common core (embeddings, lm_head, attention, routers, norms) is
//! touched every token, so it is `mmap`'d read-only and the page cache
//! keeps it resident. Integrity is checked before the map is handed out:
//! the file's size must match the manifest entry, and unless hashes are
//! skipped its SHA-256 must too (~1 GiB for the v0 model — a one-time
//! load cost the design accepts).

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::format::{COMMON_FILE, CommonTensor, Manifest};

use super::{IoError, LoadOptions, to_usize, verify_named_file};

/// Read-only memory map of an install's `common.bin`.
///
/// The mapping is private and read-only; nothing here ever writes through
/// it. The install directory is treated as immutable while the model is
/// open — truncating the file underneath a live map would fault, which is
/// the standard mmap contract, not something this layer can defend against.
#[derive(Debug)]
pub struct MappedCommon {
    map: Mmap,
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
        // SAFETY: the map is read-only and private. The install directory
        // is treated as immutable while open (see the struct docs); the
        // size was just verified against the manifest.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| IoError::io(&path, e))?;
        tracing::debug!(
            path = %path.display(),
            bytes = map.len(),
            "mapped common weights"
        );
        Ok(Self { map })
    }

    /// The whole file as a byte slice.
    pub fn bytes(&self) -> &[u8] {
        &self.map
    }

    /// Length of the mapping in bytes.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the mapping is empty (never true for a valid install).
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
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
        self.map
            .get(offset..offset + len)
            .ok_or_else(|| IoError::RangeOutOfBounds {
                what: format!("common tensor {name:?}"),
                offset: entry.offset,
                len: entry.len,
                available: self.map.len() as u64,
            })
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
