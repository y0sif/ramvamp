//! Typed errors for the `.rvmp` format layer.

use std::path::{Path, PathBuf};

use thiserror::Error;

/// Errors from parsing, validating, verifying, or installing `.rvmp` data.
///
/// Manifests, layouts, and bytes on disk are untrusted until verified, so
/// every failure mode is a typed, recoverable error; this layer never panics
/// on bad input.
#[derive(Debug, Error)]
pub enum FormatError {
    /// A filesystem operation failed.
    #[error("{}: {source}", path.display())]
    Io {
        /// Path the operation was acting on.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A JSON file exists but does not parse into its schema.
    #[error("{}: invalid JSON: {source}", path.display())]
    Json {
        /// Path of the offending file.
        path: PathBuf,
        /// Underlying parse error.
        #[source]
        source: serde_json::Error,
    },

    /// Serializing one of our own types failed (a bug, not bad input).
    #[error("serializing {what}: {source}")]
    Serialize {
        /// What was being serialized.
        what: &'static str,
        /// Underlying serialization error.
        #[source]
        source: serde_json::Error,
    },

    /// A JSON metadata file exceeds the parse size cap.
    #[error("{}: larger than the {max}-byte cap for format metadata", path.display())]
    FileTooLarge {
        /// Path of the oversized file.
        path: PathBuf,
        /// The cap that was exceeded.
        max: u64,
    },

    /// `rvmp_version` is not one this build understands.
    #[error("unsupported rvmp_version {found} (this build supports {supported})")]
    UnsupportedVersion {
        /// Version found in the manifest.
        found: u32,
        /// Version this build supports.
        supported: u32,
    },

    /// `quant.scheme` is not a known scheme.
    #[error("unknown quant scheme {scheme:?} (known: {known:?})")]
    UnknownQuantScheme {
        /// Scheme found in the manifest.
        scheme: String,
        /// Schemes this build understands.
        known: &'static [&'static str],
    },

    /// An architecture field fails a sanity bound.
    #[error("invalid architecture: {0}")]
    InvalidArch(String),

    /// A manifest field is malformed beyond what serde can express.
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),

    /// An experts layout entry is malformed.
    #[error("invalid experts layout: {0}")]
    InvalidLayout(String),

    /// An offset or stride violates its documented alignment.
    #[error("{what} {name:?}: {value} is not a multiple of {align}")]
    Misaligned {
        /// What kind of value is misaligned.
        what: &'static str,
        /// Which tensor/file/projection it belongs to.
        name: String,
        /// The misaligned value.
        value: u64,
        /// The required alignment.
        align: u64,
    },

    /// A referenced file has no `files` entry in the manifest.
    #[error("no manifest files entry for {0:?}")]
    MissingFileEntry(String),

    /// A manifest file name could escape the install directory.
    #[error("unsafe file name {0:?}: must be relative with no `..` components")]
    UnsafeFileName(String),

    /// On-disk size differs from the manifest.
    #[error("{name:?}: manifest says {expected} bytes, found {actual}")]
    SizeMismatch {
        /// Install-relative file name.
        name: String,
        /// Size recorded in the manifest.
        expected: u64,
        /// Size found on disk.
        actual: u64,
    },

    /// On-disk SHA-256 differs from the manifest.
    #[error("{name:?}: sha256 mismatch (manifest {expected}, computed {actual})")]
    HashMismatch {
        /// Install-relative file name.
        name: String,
        /// Digest recorded in the manifest.
        expected: String,
        /// Digest computed from disk.
        actual: String,
    },

    /// Promotion target already exists.
    #[error("install target {} already exists", .0.display())]
    AlreadyExists(PathBuf),
}

impl FormatError {
    /// Attach path context to an I/O error.
    pub(crate) fn io(path: &Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}
