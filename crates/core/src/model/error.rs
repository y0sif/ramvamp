//! Typed errors for model loading and weight access.

use thiserror::Error;

use crate::format::FormatError;
use crate::io::IoError;
use crate::kernels::KernelError;

/// Errors from loading an installed model or accessing its weights.
///
/// The manifest and layout are untrusted metadata: every mismatch between
/// what the architecture implies and what the install declares is a typed
/// error naming the offending tensor. No panics on bad input.
#[derive(Debug, Error)]
pub enum ModelError {
    /// A format-layer failure (parse, validation, size, or hash).
    #[error(transparent)]
    Format(#[from] FormatError),

    /// A file-access failure (mapping, verification, expert reads).
    #[error(transparent)]
    Io(#[from] IoError),

    /// Kernel-side geometry failure (row/block divisibility).
    #[error(transparent)]
    Kernel(#[from] KernelError),

    /// A tensor the architecture requires is absent from the install.
    #[error("required tensor {0:?} is missing")]
    MissingTensor(String),

    /// A tensor's declared dtype is not what the audited type map allows.
    #[error("tensor {tensor:?}: dtype {found:?}, expected {expected}")]
    WrongDtype {
        /// The offending tensor.
        tensor: String,
        /// Human-readable allowed set, e.g. `"q4_k or q6_k"`.
        expected: &'static str,
        /// The dtype the install declared.
        found: String,
    },

    /// A tensor's byte length disagrees with its expected shape.
    #[error("tensor {tensor:?}: {found} bytes, expected {expected}")]
    WrongSize {
        /// The offending tensor.
        tensor: String,
        /// Bytes the shape table demands.
        expected: u64,
        /// Bytes the install declared.
        found: u64,
    },

    /// A layer index is at or past the model's layer count.
    #[error("layer {layer} out of range ({n_layers} layers)")]
    LayerOutOfRange {
        /// Requested layer index.
        layer: u32,
        /// Layers in the model.
        n_layers: u32,
    },

    /// A token id is at or past the vocabulary size.
    #[error("token {token} out of range (vocab {vocab})")]
    TokenOutOfRange {
        /// Requested token id.
        token: u32,
        /// Vocabulary size.
        vocab: u32,
    },

    /// An output buffer has the wrong length.
    #[error("output buffer holds {found} floats, expected {expected}")]
    OutputLen {
        /// Required length.
        expected: usize,
        /// Provided length.
        found: usize,
    },
}
