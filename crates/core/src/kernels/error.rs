//! Unified typed error for every kernel family.
//!
//! One `KernelError` subsumes the former `quants::KernelError` and
//! `primitives::PrimitiveError` (both born on this branch, so the merge is
//! breaking-change-free) so the backend trait surfaces a single error type.
//! Variant names and semantics are unchanged from the originals; both
//! submodules re-export this type under their old names.
//!
//! Kernel inputs are either weight bytes from disk (untrusted until
//! length-checked) or our own shape bookkeeping; per the crate rules
//! (`ramvamp-core` never panics on untrusted input) every mismatch is a
//! typed, recoverable error. Validation happens once per public call; inner
//! loops assume validated lengths.

use super::quants::QuantFormat;
use thiserror::Error;

/// Typed errors from kernel input validation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum KernelError {
    /// A row's weight count is not a multiple of the format's block width.
    #[error(
        "{format:?} row of {in_dim} weights: not a multiple of the {block_weights}-weight block"
    )]
    IndivisibleRow {
        /// Format the row was declared as.
        format: QuantFormat,
        /// Offending row length in weights.
        in_dim: usize,
        /// The format's block width.
        block_weights: usize,
    },

    /// A weight-row byte slice is not a whole number of blocks.
    #[error("{format:?} row of {len} bytes: not a multiple of the {block_bytes}-byte block")]
    RowBytesNotBlockMultiple {
        /// Format the bytes were declared as.
        format: QuantFormat,
        /// Offending slice length in bytes.
        len: usize,
        /// The format's block size in bytes.
        block_bytes: usize,
    },

    /// A dot product's weight row and activation row disagree on length.
    #[error(
        "dot length mismatch: weight row has {weight_blocks} blocks, activations {activation_blocks}"
    )]
    BlockCountMismatch {
        /// Blocks in the weight row.
        weight_blocks: usize,
        /// Blocks in the activation row.
        activation_blocks: usize,
    },

    /// A quantizer's input floats and output blocks disagree on length.
    #[error(
        "{format:?} quantize: {floats} input floats do not fill {out_blocks} output blocks exactly"
    )]
    QuantizeLenMismatch {
        /// Target activation format.
        format: QuantFormat,
        /// Input length in f32 elements.
        floats: usize,
        /// Output length in blocks.
        out_blocks: usize,
    },

    /// Two buffers that must have equal lengths do not.
    #[error("{what}: length mismatch ({left} vs {right})")]
    LengthMismatch {
        /// Which call and which pair of buffers disagree.
        what: &'static str,
        /// Length of the first (usually destination) buffer.
        left: usize,
        /// Length of the second buffer, or the expected length.
        right: usize,
    },

    /// An operation that reduces over its input received an empty slice.
    #[error("{what}: input is empty")]
    Empty {
        /// Which call received the empty input.
        what: &'static str,
    },

    /// A RoPE head dimension is zero or odd (NeoX rotates index pairs).
    #[error("invalid head_dim {head_dim}: must be nonzero and even")]
    InvalidHeadDim {
        /// The rejected head dimension.
        head_dim: usize,
    },

    /// A documented-but-unimplemented variant was called.
    #[error("{what} is not implemented")]
    Unimplemented {
        /// Which variant is missing.
        what: &'static str,
    },

    /// A driver was asked to pair a weight format with an activation type it
    /// does not support (e.g. Q8_0 weights against Q8_K activations).
    #[error("{what}: unsupported weight format {format:?}")]
    UnsupportedFormat {
        /// Which call rejected the format.
        what: &'static str,
        /// The rejected weight format.
        format: QuantFormat,
    },
}
