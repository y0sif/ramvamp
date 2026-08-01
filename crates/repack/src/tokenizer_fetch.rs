//! Tokenizer sidecar fetch: place the pinned Hugging Face tokenizer files
//! into an installed `.rvmp` directory's `tokenizer/` subdirectory and
//! record them in `manifest.json`.
//!
//! # The tokenizer pin is not the GGUF pin
//!
//! The model weights come from a community *quantization* repo (the GGUF
//! pin: `PIN_REPO`/`PIN_REVISION` in the repack CLI, audited in
//! docs/architecture.md "Model pin"), which does not ship the HF tokenizer
//! files. Those are fetched from the *upstream model repo* instead, at a
//! pin of its own: [`TOKENIZER_REPO`] at commit [`TOKENIZER_REVISION`].
//! The two pins belong to different repos and move independently; bumping
//! one never implies bumping the other.
//!
//! # Mechanics
//!
//! [`fetch_tokenizer`] streams each of [`TOKENIZER_FILES`] from the source
//! (pinned remote revision, or a local directory as the `--tokenizer-dir`
//! escape hatch) into `<install>/tokenizer/<name>` via temp file + fsync +
//! rename, hashes the durable bytes, and returns the manifest-relative
//! [`FileEntry`] map. [`amend_manifest`] rewrites `manifest.json`
//! atomically with those entries added (they are extra files on top of the
//! schema's required set, so validation passes) and re-runs
//! [`format::verify_files`] as a self-check. [`fetch_and_amend`] chains
//! both behind an [`format::is_complete`] gate: partial installs are
//! refused. Re-running is idempotent — files are overwritten and
//! re-hashed, entries are overwritten in place.
//!
//! Transfers go through a fixed-size chunk buffer with a hard per-file cap
//! ([`MAX_TOKENIZER_FILE_BYTES`]); even these small sidecar files are
//! never materialized whole in memory (hard rule).

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use ramvamp_core::format::{self, FileEntry};

use crate::source::{LocalFile, RangeRead, RemoteFile, SourceError};

/// Upstream model repo the tokenizer files come from. This is the
/// tokenizer-source pin, distinct from the GGUF quantization-repo pin.
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3-30B-A3B-Instruct-2507";

/// Pinned revision (commit hash) of [`TOKENIZER_REPO`]. Belongs to the
/// tokenizer source only; it is unrelated to the GGUF pin's revision.
pub const TOKENIZER_REVISION: &str = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe";

/// Files fetched into the install's `tokenizer/` subdirectory: the HF
/// tokenizer, its config (chat template), and the generation defaults.
pub const TOKENIZER_FILES: [&str; 3] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
];

/// Install-relative subdirectory the tokenizer files land in.
pub const TOKENIZER_SUBDIR: &str = "tokenizer";

/// Hard per-file size cap: 64 MiB. The largest real file
/// (`tokenizer.json`) is ~11 MB; anything past the cap is rejected before
/// a byte is transferred, so a wrong URL or hostile server cannot drive an
/// unbounded download.
pub const MAX_TOKENIZER_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// Fixed transfer chunk: files stream through one reusable buffer of at
/// most this size (never a whole-file allocation).
const FETCH_CHUNK_BYTES: u64 = 8 * 1024 * 1024;

/// Where the tokenizer files come from.
#[derive(Debug, Clone)]
pub enum TokenizerSource {
    /// The pinned remote revision: [`TOKENIZER_REPO`] at
    /// [`TOKENIZER_REVISION`] over ranged HTTP.
    Pinned,
    /// A local directory containing the three [`TOKENIZER_FILES`]
    /// (the `--tokenizer-dir` escape hatch; offline installs).
    LocalDir(PathBuf),
}

/// Error fetching tokenizer files or amending the manifest.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerFetchError {
    /// Error from the `.rvmp` format layer (manifest load/write, hashing,
    /// verification).
    #[error(transparent)]
    Format(#[from] format::FormatError),
    /// Error opening or reading the tokenizer source.
    #[error(transparent)]
    Source(#[from] SourceError),
    /// A filesystem operation on the destination failed.
    #[error("{}: {source}", path.display())]
    Io {
        /// Path the operation was acting on.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// A source file exceeds the per-file cap.
    #[error("tokenizer file {name:?} is {size} bytes, over the {max}-byte per-file cap")]
    FileTooLarge {
        /// File name within the tokenizer source.
        name: String,
        /// Size the source reports.
        size: u64,
        /// The cap that was exceeded ([`MAX_TOKENIZER_FILE_BYTES`]).
        max: u64,
    },
    /// The target directory is not a complete install.
    #[error(
        "{} is not a complete .rvmp install (missing or invalid manifest.json); \
         finish the model install first",
        .0.display()
    )]
    NotComplete(PathBuf),
}

/// Attach path context to an I/O error.
fn io_err(path: &Path, source: io::Error) -> TokenizerFetchError {
    TokenizerFetchError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Fetch the three tokenizer files from `source` into
/// `<dir>/tokenizer/<name>` (temp file + fsync + rename each) and return
/// the manifest entries, keyed by install-relative path
/// (`tokenizer/tokenizer.json`, ...). Hashes are computed from the durable
/// bytes on disk after the rename.
///
/// Does not gate on install completeness and does not touch the manifest;
/// use [`fetch_and_amend`] for the full guarded flow.
pub fn fetch_tokenizer(
    dir: &Path,
    source: TokenizerSource,
) -> Result<BTreeMap<String, FileEntry>, TokenizerFetchError> {
    let dest_dir = dir.join(TOKENIZER_SUBDIR);
    fs::create_dir_all(&dest_dir).map_err(|e| io_err(&dest_dir, e))?;
    let mut entries = BTreeMap::new();
    for name in TOKENIZER_FILES {
        let reader: Box<dyn RangeRead> = match &source {
            TokenizerSource::Pinned => {
                let url = format!(
                    "https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REVISION}/{name}"
                );
                tracing::info!(url = url.as_str(), "fetching tokenizer file");
                Box::new(RemoteFile::open(&url)?)
            }
            TokenizerSource::LocalDir(src) => {
                let path = src.join(name);
                tracing::info!(path = %path.display(), "copying tokenizer file");
                Box::new(LocalFile::open(&path)?)
            }
        };
        let entry = copy_to(&dest_dir, name, reader.as_ref(), FETCH_CHUNK_BYTES)?;
        tracing::debug!(
            file = name,
            bytes = entry.size,
            sha256 = entry.sha256.as_str(),
            "tokenizer file durable"
        );
        entries.insert(format!("{TOKENIZER_SUBDIR}/{name}"), entry);
    }
    Ok(entries)
}

/// Insert/overwrite `entries` in the manifest's `files` map and rewrite
/// `manifest.json` in `dir` (core's [`format::write_manifest`] is
/// temp + fsync + rename, and validates the full manifest first — the
/// tokenizer entries are extra files beyond the required set, so a valid
/// manifest stays valid). Then re-verify the amended entries on disk as a
/// self-check.
///
/// The self-check is scoped to the tokenizer entries just written: the
/// model data files were hash-verified by the install itself, and
/// re-hashing ~17 GiB on every tokenizer amendment would be pure waste.
/// A full re-check remains available via `verify-install`.
pub fn amend_manifest(
    dir: &Path,
    entries: &BTreeMap<String, FileEntry>,
) -> Result<(), TokenizerFetchError> {
    let mut manifest = format::load_manifest(dir)?;
    for (name, entry) in entries {
        manifest.files.insert(name.clone(), entry.clone());
    }
    format::write_manifest(dir, &manifest)?;
    // Restrict the verification view to the amended entries only.
    let mut scoped = manifest.clone();
    scoped.files = entries.clone();
    format::verify_files(dir, &scoped)?;
    tracing::info!(
        dir = %dir.display(),
        entries = entries.len(),
        "manifest amended with tokenizer files; amended entries re-verified"
    );
    Ok(())
}

/// The full guarded flow: refuse anything that is not a complete install
/// ([`format::is_complete`]), fetch the tokenizer files, amend the
/// manifest, self-verify. Idempotent: re-running overwrites the files and
/// their manifest entries with freshly hashed copies.
pub fn fetch_and_amend(
    dir: &Path,
    source: TokenizerSource,
) -> Result<BTreeMap<String, FileEntry>, TokenizerFetchError> {
    if !format::is_complete(dir) {
        return Err(TokenizerFetchError::NotComplete(dir.to_path_buf()));
    }
    let entries = fetch_tokenizer(dir, source)?;
    amend_manifest(dir, &entries)?;
    Ok(entries)
}

/// Stream `source` into `<dest_dir>/<name>` through a bounded chunk
/// buffer: cap check first, then temp file + fsync + rename + directory
/// fsync, then hash the durable bytes. Returns the file's manifest entry.
fn copy_to(
    dest_dir: &Path,
    name: &str,
    source: &dyn RangeRead,
    chunk_bytes: u64,
) -> Result<FileEntry, TokenizerFetchError> {
    debug_assert!(chunk_bytes > 0);
    let total = source.len();
    if total > MAX_TOKENIZER_FILE_BYTES {
        return Err(TokenizerFetchError::FileTooLarge {
            name: name.to_owned(),
            size: total,
            max: MAX_TOKENIZER_FILE_BYTES,
        });
    }
    let path = dest_dir.join(name);
    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(".tmp");
    let tmp = PathBuf::from(tmp_name);
    {
        let mut file = fs::File::create(&tmp).map_err(|e| io_err(&tmp, e))?;
        // One reusable chunk buffer; never a whole-file allocation.
        let mut buf = vec![0u8; total.min(chunk_bytes) as usize];
        let mut offset = 0u64;
        while offset < total {
            let n = (total - offset).min(chunk_bytes) as usize;
            let chunk = &mut buf[..n];
            source.read_at(offset, chunk)?;
            file.write_all(chunk).map_err(|e| io_err(&tmp, e))?;
            offset += n as u64;
        }
        file.sync_all().map_err(|e| io_err(&tmp, e))?;
    }
    fs::rename(&tmp, &path).map_err(|e| io_err(&tmp, e))?;
    fsync_dir(dest_dir)?;
    // Hash after the rename so the digest describes the durable bytes.
    let sha256 = format::sha256_file(&path)?;
    Ok(FileEntry {
        size: total,
        sha256,
    })
}

/// Fsync a directory so a rename inside it is durable.
fn fsync_dir(dir: &Path) -> Result<(), TokenizerFetchError> {
    let handle = fs::File::open(dir).map_err(|e| io_err(dir, e))?;
    handle.sync_all().map_err(|e| io_err(dir, e))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use ramvamp_core::format::{
        ArchInfo, FormatError, MANIFEST_FILE, Manifest, QuantInfo, RVMP_VERSION, SourceInfo,
    };
    use sha2::{Digest, Sha256};

    use super::*;

    /// Unique directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let pid = std::process::id();
            let path =
                std::env::temp_dir().join(format!("ramvamp-tokenizer-{tag}-{pid}-{n}-{nanos}"));
            fs::create_dir_all(&path).expect("create test temp dir");
            TempDir(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Build a minimal but genuinely complete `.rvmp` install at `dir`:
    /// real data files, real hashes, a manifest that validates, written
    /// with core's own `write_manifest`. (Core's test fixtures are
    /// `#[cfg(test)]`-internal to core, so the fake install is built here
    /// from public types only.)
    fn fake_install(dir: &Path) -> Manifest {
        fs::create_dir_all(dir.join("experts")).unwrap();
        let common: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        fs::write(dir.join("common.bin"), &common).unwrap();
        fs::write(dir.join("experts/layout.json"), b"{}").unwrap();
        let layer: Vec<u8> = (0..8192u32).map(|i| (i % 253) as u8).collect();
        fs::write(dir.join("experts/layer_00.bin"), &layer).unwrap();

        let mut files = BTreeMap::new();
        for name in ["common.bin", "experts/layout.json", "experts/layer_00.bin"] {
            let path = dir.join(name);
            files.insert(
                name.to_owned(),
                FileEntry {
                    size: fs::metadata(&path).unwrap().len(),
                    sha256: format::sha256_file(&path).unwrap(),
                },
            );
        }
        let manifest = Manifest {
            rvmp_version: RVMP_VERSION,
            model_id: "test-tokenizer".to_owned(),
            source: SourceInfo {
                hf_repo: "test/repo".to_owned(),
                revision: "deadbeef".to_owned(),
                file: "tiny-Q4_K_M.gguf".to_owned(),
                sha256: "0".repeat(64),
            },
            arch: ArchInfo {
                n_layers: 1,
                n_experts: 4,
                top_k: 2,
                hidden: 64,
                moe_intermediate: 96,
                n_heads: 8,
                n_kv_heads: 2,
                head_dim: 8,
                vocab: 512,
                context_length: 4096,
                rope_theta: 1e7,
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
            files,
        };
        format::write_manifest(dir, &manifest).unwrap();
        assert!(format::is_complete(dir));
        manifest
    }

    /// A local tokenizer source directory holding all three files.
    fn tokenizer_src(dir: &Path) -> PathBuf {
        let src = dir.join("tok-src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("tokenizer.json"), br#"{"model":{"type":"BPE"}}"#).unwrap();
        fs::write(
            src.join("tokenizer_config.json"),
            br#"{"chat_template":"t"}"#,
        )
        .unwrap();
        fs::write(
            src.join("generation_config.json"),
            br#"{"temperature":0.7}"#,
        )
        .unwrap();
        src
    }

    #[test]
    fn local_dir_fetch_amends_manifest_and_verifies() {
        let tmp = TempDir::new("local-fetch");
        let install = tmp.path().join("model.rvmp");
        let before = fake_install(&install);
        let src = tokenizer_src(tmp.path());

        let entries = fetch_and_amend(&install, TokenizerSource::LocalDir(src.clone())).unwrap();
        assert_eq!(entries.len(), 3);
        let expected_keys = [
            "tokenizer/generation_config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ];
        assert_eq!(
            entries.keys().map(String::as_str).collect::<Vec<_>>(),
            expected_keys
        );

        // The manifest gained exactly the three entries, still validates,
        // and every file (old and new) verifies on disk.
        let manifest = format::load_manifest(&install).unwrap();
        manifest.validate().unwrap();
        assert_eq!(manifest.files.len(), before.files.len() + 3);
        for key in expected_keys {
            assert_eq!(manifest.files[key], entries[key]);
        }
        format::verify_files(&install, &manifest).unwrap();

        // Bytes are copied verbatim.
        for name in TOKENIZER_FILES {
            assert_eq!(
                fs::read(install.join(TOKENIZER_SUBDIR).join(name)).unwrap(),
                fs::read(src.join(name)).unwrap(),
                "{name} differs from its source"
            );
        }
        // No temp files left behind.
        assert!(!install.join("tokenizer/tokenizer.json.tmp").exists());
        assert!(!install.join("manifest.json.tmp").exists());
    }

    #[test]
    fn refuses_incomplete_install() {
        let tmp = TempDir::new("incomplete");
        let src = tokenizer_src(tmp.path());

        // Nonexistent directory.
        let absent = tmp.path().join("absent.rvmp");
        let err = fetch_and_amend(&absent, TokenizerSource::LocalDir(src.clone())).unwrap_err();
        assert!(matches!(err, TokenizerFetchError::NotComplete(_)));

        // Partial-style directory: exists, but no parseable manifest.
        let partial = tmp.path().join("model.rvmp.partial");
        fs::create_dir_all(&partial).unwrap();
        fs::write(partial.join(MANIFEST_FILE), b"not json").unwrap();
        let err = fetch_and_amend(&partial, TokenizerSource::LocalDir(src)).unwrap_err();
        assert!(matches!(err, TokenizerFetchError::NotComplete(_)));
        // Nothing was written into the refused directory.
        assert!(!partial.join(TOKENIZER_SUBDIR).exists());
    }

    #[test]
    fn rerun_overwrites_and_rehashes() {
        let tmp = TempDir::new("rerun");
        let install = tmp.path().join("model.rvmp");
        let before = fake_install(&install);
        let src = tokenizer_src(tmp.path());

        let first = fetch_and_amend(&install, TokenizerSource::LocalDir(src.clone())).unwrap();

        // Change one source file and re-run: same entry count, new hash.
        fs::write(src.join("tokenizer.json"), br#"{"model":{"type":"WP"}}"#).unwrap();
        let second = fetch_and_amend(&install, TokenizerSource::LocalDir(src.clone())).unwrap();
        assert_eq!(second.len(), 3);
        assert_ne!(
            first["tokenizer/tokenizer.json"].sha256,
            second["tokenizer/tokenizer.json"].sha256
        );
        assert_eq!(
            first["tokenizer/generation_config.json"],
            second["tokenizer/generation_config.json"]
        );

        let manifest = format::load_manifest(&install).unwrap();
        assert_eq!(manifest.files.len(), before.files.len() + 3);
        assert_eq!(
            manifest.files["tokenizer/tokenizer.json"],
            second["tokenizer/tokenizer.json"]
        );
        format::verify_files(&install, &manifest).unwrap();
        assert_eq!(
            fs::read(install.join("tokenizer/tokenizer.json")).unwrap(),
            fs::read(src.join("tokenizer.json")).unwrap()
        );
    }

    #[test]
    fn per_file_cap_is_enforced_before_reading() {
        let tmp = TempDir::new("cap");
        let install = tmp.path().join("model.rvmp");
        fake_install(&install);
        let src = tokenizer_src(tmp.path());

        // A sparse file over the cap: rejected on its reported size, so
        // the test never materializes 64 MiB of data.
        let big = fs::File::create(src.join("tokenizer.json")).unwrap();
        big.set_len(MAX_TOKENIZER_FILE_BYTES + 1).unwrap();
        drop(big);

        let err = fetch_and_amend(&install, TokenizerSource::LocalDir(src)).unwrap_err();
        match err {
            TokenizerFetchError::FileTooLarge { name, size, max } => {
                assert_eq!(name, "tokenizer.json");
                assert_eq!(size, MAX_TOKENIZER_FILE_BYTES + 1);
                assert_eq!(max, MAX_TOKENIZER_FILE_BYTES);
            }
            other => panic!("expected FileTooLarge, got {other:?}"),
        }
        // The oversized file never landed, and the manifest is untouched.
        assert!(!install.join("tokenizer/tokenizer.json").exists());
        let manifest = format::load_manifest(&install).unwrap();
        assert!(!manifest.files.keys().any(|k| k.starts_with("tokenizer/")));
        format::verify_files(&install, &manifest).unwrap();
    }

    #[test]
    fn amend_self_check_reads_only_tokenizer_entries() {
        let tmp = TempDir::new("scoped-verify");
        let install = tmp.path().join("model.rvmp");
        fake_install(&install);
        let src = tokenizer_src(tmp.path());

        // Corrupt common.bin (same size, different bytes): a full-manifest
        // verify would fail on its hash, so a successful amend proves the
        // self-check never read the model data files.
        let len = fs::metadata(install.join("common.bin")).unwrap().len();
        fs::write(install.join("common.bin"), vec![0xa5u8; len as usize]).unwrap();

        let entries = fetch_and_amend(&install, TokenizerSource::LocalDir(src)).unwrap();
        assert_eq!(entries.len(), 3);

        // The amendment landed and the tokenizer entries verify in
        // isolation...
        let manifest = format::load_manifest(&install).unwrap();
        manifest.validate().unwrap();
        assert_eq!(
            manifest.files["tokenizer/tokenizer.json"],
            entries["tokenizer/tokenizer.json"]
        );
        let mut scoped = manifest.clone();
        scoped.files = entries;
        format::verify_files(&install, &scoped).unwrap();

        // ...while an explicit full verify (the verify-install path) still
        // catches the data-file corruption.
        match format::verify_files(&install, &manifest).unwrap_err() {
            FormatError::HashMismatch { name, .. } => assert_eq!(name, "common.bin"),
            other => panic!("expected HashMismatch on common.bin, got {other:?}"),
        }
    }

    #[test]
    fn missing_source_file_is_typed_error() {
        let tmp = TempDir::new("missing-src");
        let install = tmp.path().join("model.rvmp");
        fake_install(&install);
        let src = tokenizer_src(tmp.path());
        fs::remove_file(src.join("generation_config.json")).unwrap();

        let err = fetch_and_amend(&install, TokenizerSource::LocalDir(src)).unwrap_err();
        assert!(matches!(
            err,
            TokenizerFetchError::Source(SourceError::Open { .. })
        ));
        // The manifest was never amended.
        let manifest = format::load_manifest(&install).unwrap();
        assert!(!manifest.files.keys().any(|k| k.starts_with("tokenizer/")));
    }

    #[test]
    fn copy_streams_in_bounded_chunks() {
        let tmp = TempDir::new("chunks");
        let dest = tmp.path().join("out");
        fs::create_dir_all(&dest).unwrap();

        // 100 bytes through 7-byte chunks: 14 full chunks + a 2-byte tail.
        let data: Vec<u8> = (0..100u8).collect();
        let slice: &[u8] = &data;
        let entry = copy_to(&dest, "tokenizer.json", &slice, 7).unwrap();
        assert_eq!(entry.size, 100);
        assert_eq!(fs::read(dest.join("tokenizer.json")).unwrap(), data);

        let mut hasher = Sha256::new();
        hasher.update(&data);
        let expected = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(entry.sha256, expected);

        // Empty source: valid zero-byte copy, hash of the empty string.
        let empty: &[u8] = &[];
        let entry = copy_to(&dest, "tokenizer_config.json", &empty, 7).unwrap();
        assert_eq!(entry.size, 0);
        assert_eq!(
            entry.sha256,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
