//! Install-time writers and atomic promotion.
//!
//! The repacker builds an install inside `<final>.partial/` (a sibling of
//! the final directory, so `rename(2)` stays on one filesystem). Data files
//! land first; `manifest.json` is written last via temp file + fsync +
//! rename, then [`promote`] re-validates the manifest and renames the whole
//! partial directory to its final name. A crash at any point leaves either
//! a rejectable partial (no parseable manifest at the final path) or a
//! complete install — never a half-promoted one.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

use super::{ExpertsLayout, FormatError, LAYOUT_FILE, MANIFEST_FILE, Manifest};

/// Suffix appended to the final directory name to form the staging
/// directory, e.g. `model.rvmp` -> `model.rvmp.partial`.
pub const PARTIAL_SUFFIX: &str = ".partial";

/// Parse cap for `manifest.json` / `layout.json`. Real files are a few
/// hundred KiB at most; the cap keeps a hostile file from driving an
/// unbounded allocation.
pub const MAX_JSON_BYTES: u64 = 16 * 1024 * 1024;

/// The staging directory for an install: `<final_dir>.partial`, as a
/// sibling of `final_dir` so promotion is a single same-filesystem rename.
pub fn partial_dir(final_dir: &Path) -> PathBuf {
    let mut name = final_dir.as_os_str().to_os_string();
    name.push(PARTIAL_SUFFIX);
    PathBuf::from(name)
}

/// Validate the manifest, then durably write it to `<dir>/manifest.json`
/// (temp file, fsync, rename, directory fsync). `dir` is normally the
/// staging directory from [`partial_dir`]; writing the manifest is the last
/// step before [`promote`].
pub fn write_manifest(dir: &Path, manifest: &Manifest) -> Result<(), FormatError> {
    manifest.validate()?;
    write_json_durable(dir, MANIFEST_FILE, manifest, "manifest")
}

/// Validate the layout's intrinsic invariants, then durably write it to
/// `<dir>/experts/layout.json`. Cross-checks against the manifest
/// ([`ExpertsLayout::validate_against`]) are the caller's job once both
/// halves exist.
pub fn write_layout(dir: &Path, layout: &ExpertsLayout) -> Result<(), FormatError> {
    layout.validate()?;
    write_json_durable(dir, LAYOUT_FILE, layout, "experts layout")
}

/// Load and parse `<dir>/manifest.json`. Parsing only — call
/// [`Manifest::validate`] before trusting the contents.
pub fn load_manifest(dir: &Path) -> Result<Manifest, FormatError> {
    read_json_bounded(&dir.join(MANIFEST_FILE))
}

/// Load and parse `<dir>/experts/layout.json`. Parsing only — call
/// [`ExpertsLayout::validate_against`] before trusting the contents.
pub fn load_layout(dir: &Path) -> Result<ExpertsLayout, FormatError> {
    read_json_bounded(&dir.join(LAYOUT_FILE))
}

/// A complete install is a directory whose `manifest.json` exists and
/// parses. Anything else is a partial install the loader must reject.
pub fn is_complete(dir: &Path) -> bool {
    match load_manifest(dir) {
        Ok(_) => true,
        Err(err) => {
            tracing::debug!(dir = %dir.display(), error = %err, "install is not complete");
            false
        }
    }
}

/// Atomically promote a staged install: re-validate its manifest, then
/// rename `partial_dir` to `final_dir` and fsync the parent directory.
///
/// Fails without touching anything if the manifest is missing or invalid,
/// or if `final_dir` already exists. Integrity of the data files is
/// [`verify_files`](super::verify_files)' job, not promotion's.
pub fn promote(partial_dir: &Path, final_dir: &Path) -> Result<(), FormatError> {
    let manifest = load_manifest(partial_dir)?;
    manifest.validate()?;
    if final_dir.symlink_metadata().is_ok() {
        return Err(FormatError::AlreadyExists(final_dir.to_path_buf()));
    }
    fs::rename(partial_dir, final_dir).map_err(|e| FormatError::io(partial_dir, e))?;
    fsync_dir(parent_of(final_dir))?;
    tracing::info!(
        model = manifest.model_id.as_str(),
        dir = %final_dir.display(),
        "promoted install"
    );
    Ok(())
}

/// Serialize `value` as pretty JSON and durably write it to `dir/rel`:
/// write to `<path>.tmp`, fsync the file, rename over the destination,
/// fsync the containing directory.
fn write_json_durable<T: Serialize>(
    dir: &Path,
    rel: &str,
    value: &T,
    what: &'static str,
) -> Result<(), FormatError> {
    let path = dir.join(rel);
    let parent = path.parent().unwrap_or(dir).to_path_buf();
    fs::create_dir_all(&parent).map_err(|e| FormatError::io(&parent, e))?;

    let json = serde_json::to_vec_pretty(value)
        .map_err(|source| FormatError::Serialize { what, source })?;

    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(".tmp");
    let tmp = PathBuf::from(tmp_name);
    {
        let mut file = fs::File::create(&tmp).map_err(|e| FormatError::io(&tmp, e))?;
        file.write_all(&json)
            .map_err(|e| FormatError::io(&tmp, e))?;
        file.sync_all().map_err(|e| FormatError::io(&tmp, e))?;
    }
    fs::rename(&tmp, &path).map_err(|e| FormatError::io(&tmp, e))?;
    fsync_dir(&parent)?;
    tracing::debug!(path = %path.display(), bytes = json.len(), what, "wrote metadata file");
    Ok(())
}

/// Read a JSON file with a hard size cap, then parse it.
fn read_json_bounded<T: DeserializeOwned>(path: &Path) -> Result<T, FormatError> {
    let file = fs::File::open(path).map_err(|e| FormatError::io(path, e))?;
    let mut data = Vec::new();
    file.take(MAX_JSON_BYTES + 1)
        .read_to_end(&mut data)
        .map_err(|e| FormatError::io(path, e))?;
    if data.len() as u64 > MAX_JSON_BYTES {
        return Err(FormatError::FileTooLarge {
            path: path.to_path_buf(),
            max: MAX_JSON_BYTES,
        });
    }
    serde_json::from_slice(&data).map_err(|source| FormatError::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// Fsync a directory so a rename inside it is durable.
fn fsync_dir(dir: &Path) -> Result<(), FormatError> {
    let handle = fs::File::open(dir).map_err(|e| FormatError::io(dir, e))?;
    handle.sync_all().map_err(|e| FormatError::io(dir, e))
}

/// Parent directory of `path`, falling back to `.` for bare relative names.
fn parent_of(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{TempDir, sample_layout, sample_manifest};
    use super::*;

    #[test]
    fn partial_dir_appends_suffix() {
        assert_eq!(
            partial_dir(Path::new("/models/qwen.rvmp")),
            Path::new("/models/qwen.rvmp.partial")
        );
    }

    #[test]
    fn install_flow_write_then_promote() {
        let tmp = TempDir::new("install-flow");
        let final_dir = tmp.path().join("model.rvmp");
        let staging = partial_dir(&final_dir);

        let manifest = sample_manifest();
        let layout = sample_layout();
        write_layout(&staging, &layout).unwrap();
        write_manifest(&staging, &manifest).unwrap();
        assert!(!staging.join("manifest.json.tmp").exists());

        assert!(is_complete(&staging));
        assert!(!is_complete(&final_dir));

        promote(&staging, &final_dir).unwrap();
        assert!(!staging.exists());
        assert!(is_complete(&final_dir));
        assert_eq!(load_manifest(&final_dir).unwrap(), manifest);
        assert_eq!(load_layout(&final_dir).unwrap(), layout);

        // A second promotion has nothing to promote.
        assert!(matches!(
            promote(&staging, &final_dir).unwrap_err(),
            FormatError::Io { .. }
        ));
    }

    #[test]
    fn write_manifest_rejects_invalid_without_writing() {
        let tmp = TempDir::new("install-invalid-write");
        let staging = tmp.path().join("model.rvmp.partial");
        let mut manifest = sample_manifest();
        manifest.arch.top_k = manifest.arch.n_experts + 1;
        assert!(matches!(
            write_manifest(&staging, &manifest).unwrap_err(),
            FormatError::InvalidArch(_)
        ));
        assert!(!staging.join(MANIFEST_FILE).exists());
    }

    #[test]
    fn promote_rejects_existing_target() {
        let tmp = TempDir::new("install-target-exists");
        let final_dir = tmp.path().join("model.rvmp");
        fs::create_dir_all(&final_dir).unwrap();
        let staging = partial_dir(&final_dir);
        write_manifest(&staging, &sample_manifest()).unwrap();

        assert!(matches!(
            promote(&staging, &final_dir).unwrap_err(),
            FormatError::AlreadyExists(_)
        ));
        // The staged install is untouched and still promotable elsewhere.
        assert!(is_complete(&staging));
    }

    #[test]
    fn promote_rejects_invalid_manifest() {
        let tmp = TempDir::new("install-invalid-promote");
        let staging = tmp.path().join("model.rvmp.partial");
        fs::create_dir_all(&staging).unwrap();
        let mut manifest = sample_manifest();
        manifest.rvmp_version = 99;
        // Bypass write_manifest's validation to simulate a corrupt stage.
        fs::write(
            staging.join(MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let final_dir = tmp.path().join("model.rvmp");
        assert!(matches!(
            promote(&staging, &final_dir).unwrap_err(),
            FormatError::UnsupportedVersion { found: 99, .. }
        ));
        assert!(staging.exists());
        assert!(!final_dir.exists());
    }

    #[test]
    fn is_complete_rejects_missing_and_garbage() {
        let tmp = TempDir::new("install-incomplete");
        assert!(!is_complete(&tmp.path().join("absent")));

        let garbage = tmp.path().join("garbage.rvmp");
        fs::create_dir_all(&garbage).unwrap();
        fs::write(garbage.join(MANIFEST_FILE), b"not json").unwrap();
        assert!(!is_complete(&garbage));
    }

    #[test]
    fn load_manifest_enforces_size_cap() {
        let tmp = TempDir::new("install-oversize");
        let dir = tmp.path().join("model.rvmp");
        fs::create_dir_all(&dir).unwrap();
        let oversized = vec![b' '; MAX_JSON_BYTES as usize + 1];
        fs::write(dir.join(MANIFEST_FILE), oversized).unwrap();
        assert!(matches!(
            load_manifest(&dir).unwrap_err(),
            FormatError::FileTooLarge { .. }
        ));
    }
}
