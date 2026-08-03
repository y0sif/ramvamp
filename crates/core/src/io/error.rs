//! Typed errors for the file-access layer.

use std::path::PathBuf;

use thiserror::Error;

use crate::format::{FormatError, ProjectionName};

/// Errors from mapping, verifying, or reading installed model files.
///
/// Everything on disk is untrusted until verified, so every failure mode is
/// a typed, recoverable error; this layer never panics on bad input.
#[derive(Debug, Error)]
pub enum IoError {
    /// A format-layer failure: missing manifest entry, size mismatch,
    /// hash mismatch, or an I/O error raised while hashing.
    #[error(transparent)]
    Format(#[from] FormatError),

    /// A filesystem operation failed.
    #[error("{}: {source}", path.display())]
    Io {
        /// Path the operation was acting on.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A layer index is at or past the layout's layer count.
    #[error("layer {layer} out of range ({n_layers} layers)")]
    LayerOutOfRange {
        /// Requested layer index.
        layer: u32,
        /// Number of layers in the layout.
        n_layers: u32,
    },

    /// An expert index is at or past the layer's expert count.
    #[error("layer {layer}: expert {expert} out of range ({n_experts} experts)")]
    ExpertOutOfRange {
        /// Layer the request was for.
        layer: u32,
        /// Requested expert index.
        expert: u32,
        /// Experts per layer.
        n_experts: u32,
    },

    /// A layer's layout lacks one of the three FFN projections the
    /// streaming FFN needs.
    #[error("layer {layer}: layout has no {name:?} projection")]
    MissingProjection {
        /// Layer whose layout is incomplete.
        layer: u32,
        /// The absent projection.
        name: ProjectionName,
    },

    /// A layout quant name is not a weight format this build can compute.
    #[error("{what}: unknown quant format {quant:?}")]
    UnknownQuant {
        /// Which slab or tensor declared the format.
        what: String,
        /// The unrecognized format name.
        quant: String,
    },

    /// A byte range does not fit the file or buffer it indexes.
    #[error("{what}: range {offset}+{len} out of bounds ({available} bytes)")]
    RangeOutOfBounds {
        /// What was being sliced.
        what: String,
        /// Start of the requested range.
        offset: u64,
        /// Length of the requested range.
        len: u64,
        /// Bytes actually available.
        available: u64,
    },

    /// A 64-bit size in the metadata does not fit this platform's `usize`.
    #[error("{what}: {value} does not fit usize on this platform")]
    TooLarge {
        /// What the value describes.
        what: &'static str,
        /// The oversized value.
        value: u64,
    },

    /// Sizing, allocating, or leasing an expert slot buffer failed.
    #[error(transparent)]
    Slots(#[from] crate::io::SlotError),

    /// Planning one routing step against a layer's slots failed.
    #[error(transparent)]
    Cache(#[from] crate::io::CacheError),
}

impl IoError {
    /// Attach path context to an I/O error.
    pub(crate) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}
