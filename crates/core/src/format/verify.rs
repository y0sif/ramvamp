//! Streaming SHA-256 hashing and on-disk integrity verification.
//!
//! Hashing streams through a fixed 128 KiB buffer (matching the reference
//! NVMe's `max_hw_sectors_kb=128`), so a multi-gigabyte layer file is never
//! materialized in memory — the same rule the repacker and runtime live by.

use std::fmt::Write as _;
use std::fs;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use super::manifest::check_file_name;
use super::{FormatError, Manifest};

/// Fixed hashing buffer size.
const HASH_BUF_BYTES: usize = 128 * 1024;

/// Compute the SHA-256 of a file as a lowercase hex string, streaming
/// through a fixed-size buffer.
pub fn sha256_file(path: &Path) -> Result<String, FormatError> {
    let mut file = fs::File::open(path).map_err(|e| FormatError::io(path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_BUF_BYTES];
    loop {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => hasher.update(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(FormatError::io(path, e)),
        }
    }
    Ok(to_hex(&hasher.finalize()))
}

/// Check every `files` entry of the manifest against the install directory:
/// exact size match first (cheap), then a streaming SHA-256 compare
/// (case-insensitive on the manifest side).
///
/// Fails on the first mismatch. File names are re-checked for path safety
/// so a hostile manifest cannot make us hash files outside `dir` even when
/// `validate()` was skipped.
pub fn verify_files(dir: &Path, manifest: &Manifest) -> Result<(), FormatError> {
    for (name, entry) in &manifest.files {
        check_file_name(name)?;
        let path = dir.join(name);
        let metadata = fs::metadata(&path).map_err(|e| FormatError::io(&path, e))?;
        if metadata.len() != entry.size {
            return Err(FormatError::SizeMismatch {
                name: name.clone(),
                expected: entry.size,
                actual: metadata.len(),
            });
        }
        let actual = sha256_file(&path)?;
        if !actual.eq_ignore_ascii_case(&entry.sha256) {
            return Err(FormatError::HashMismatch {
                name: name.clone(),
                expected: entry.sha256.clone(),
                actual,
            });
        }
        tracing::debug!(file = name.as_str(), bytes = entry.size, "hash verified");
    }
    tracing::debug!(
        dir = %dir.display(),
        files = manifest.files.len(),
        "all files verified"
    );
    Ok(())
}

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::super::testutil::{TempDir, sample_manifest};
    use super::*;

    /// Write deterministic bytes for every manifest file entry, then point
    /// the entries at the real on-disk sizes and hashes.
    fn write_install(dir: &Path, manifest: &mut Manifest) {
        fs::create_dir_all(dir.join("experts")).unwrap();
        for (name, entry) in &manifest.files {
            let data: Vec<u8> = (0..entry.size)
                .map(|i| (i.wrapping_mul(31).wrapping_add(name.len() as u64) % 251) as u8)
                .collect();
            fs::write(dir.join(name), data).unwrap();
        }
        let mut fixed = BTreeMap::new();
        for name in manifest.files.keys() {
            let path = dir.join(name);
            fixed.insert(
                name.clone(),
                super::super::FileEntry {
                    size: fs::metadata(&path).unwrap().len(),
                    sha256: sha256_file(&path).unwrap(),
                },
            );
        }
        manifest.files = fixed;
    }

    #[test]
    fn sha256_matches_known_vectors() {
        let tmp = TempDir::new("sha-vectors");
        let empty = tmp.path().join("empty");
        fs::write(&empty, b"").unwrap();
        assert_eq!(
            sha256_file(&empty).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let abc = tmp.path().join("abc");
        fs::write(&abc, b"abc").unwrap();
        assert_eq!(
            sha256_file(&abc).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_streams_across_buffer_boundary() {
        let tmp = TempDir::new("sha-large");
        let path = tmp.path().join("big");
        // 300 KiB: crosses the 128 KiB buffer twice, with a partial tail.
        let data: Vec<u8> = (0..300 * 1024).map(|i| (i % 253) as u8).collect();
        fs::write(&path, &data).unwrap();
        assert_eq!(sha256_file(&path).unwrap(), to_hex(&Sha256::digest(&data)));
    }

    #[test]
    fn sha256_missing_file_is_io_error() {
        let tmp = TempDir::new("sha-missing");
        assert!(matches!(
            sha256_file(&tmp.path().join("absent")).unwrap_err(),
            FormatError::Io { .. }
        ));
    }

    #[test]
    fn verify_accepts_intact_install() {
        let tmp = TempDir::new("verify-ok");
        let mut manifest = sample_manifest();
        write_install(tmp.path(), &mut manifest);
        manifest.validate().unwrap();
        verify_files(tmp.path(), &manifest).unwrap();
    }

    #[test]
    fn verify_detects_corruption() {
        let tmp = TempDir::new("verify-flip");
        let mut manifest = sample_manifest();
        write_install(tmp.path(), &mut manifest);
        let victim = tmp.path().join("experts/layer_00.bin");
        let mut data = fs::read(&victim).unwrap();
        data[100] ^= 0xff;
        fs::write(&victim, data).unwrap();
        match verify_files(tmp.path(), &manifest).unwrap_err() {
            FormatError::HashMismatch { name, .. } => {
                assert_eq!(name, "experts/layer_00.bin");
            }
            other => panic!("expected HashMismatch, got {other:?}"),
        }
    }

    #[test]
    fn verify_detects_truncation() {
        let tmp = TempDir::new("verify-trunc");
        let mut manifest = sample_manifest();
        write_install(tmp.path(), &mut manifest);
        let victim = tmp.path().join("common.bin");
        let data = fs::read(&victim).unwrap();
        fs::write(&victim, &data[..data.len() - 1]).unwrap();
        assert!(matches!(
            verify_files(tmp.path(), &manifest).unwrap_err(),
            FormatError::SizeMismatch { .. }
        ));
    }

    #[test]
    fn verify_detects_missing_file() {
        let tmp = TempDir::new("verify-missing");
        let mut manifest = sample_manifest();
        write_install(tmp.path(), &mut manifest);
        fs::remove_file(tmp.path().join("experts/layout.json")).unwrap();
        assert!(matches!(
            verify_files(tmp.path(), &manifest).unwrap_err(),
            FormatError::Io { .. }
        ));
    }

    #[test]
    fn verify_rejects_unsafe_names_without_touching_disk() {
        let tmp = TempDir::new("verify-unsafe");
        let mut manifest = sample_manifest();
        manifest.files.insert(
            "../outside.bin".to_owned(),
            super::super::FileEntry {
                size: 0,
                sha256: super::super::testutil::ZERO_SHA.to_owned(),
            },
        );
        assert!(matches!(
            verify_files(tmp.path(), &manifest).unwrap_err(),
            FormatError::UnsafeFileName(_)
        ));
    }
}
