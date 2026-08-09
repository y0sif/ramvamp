//! Quantized block formats and scalar reference kernels.
//!
//! Everything here mirrors upstream ggml (`ggml-common.h` block layouts,
//! `ggml-quants.c` reference quantizers, dequantizers, and scalar
//! `vec_dot_*` fallbacks) so that the bytes we stream from a repacked GGUF
//! compute the same numbers llama.cpp would compute on them. The audited v0
//! model needs weights in Q4_K, Q5_K, Q6_K, and Q8_0, activations in Q8_K
//! (for the k-quant dots) and Q8_0 (for the `q8_0 x q8_0` dot); Q4_0 appears
//! nowhere in the audited pin and is not implemented.
//!
//! Weight rows are consumed as raw byte slices (`&[u8]`) straight from the
//! mmap'd common core or a streamed expert slab: zero-copy, no `#[repr(C)]`
//! casting. Activation blocks are plain Rust structs produced per token by
//! the quantizers here; they never touch disk.
//!
//! # Alignment
//!
//! The scalar kernels assume nothing beyond byte alignment. Packed
//! sub-tensor offsets may be only 2-byte aligned, so every multi-byte field
//! (f16 scales, i16 bsums) is read with `from_le_bytes` on individual byte
//! pairs, never by casting pointers. Vectorized ports must re-state and test
//! their own alignment assumptions.
//!
//! # Validation contract
//!
//! Public entry points validate slice lengths once up front and return
//! [`KernelError`] on mismatch; inner loops then index fixed-size block
//! arrays with in-bounds constant offsets and cannot panic.

pub mod avx2;
mod blocks;
mod dequant;
mod dot;
mod f16;
mod quantize;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

pub use blocks::{BlockQ8_0, BlockQ8K};
pub use dequant::{
    dequantize_row_q4_k, dequantize_row_q5_k, dequantize_row_q6_k, dequantize_row_q8_0,
};
pub use dot::{vec_dot_q4_k_q8_k, vec_dot_q5_k_q8_k, vec_dot_q6_k_q8_k, vec_dot_q8_0_q8_0};
pub use f16::{f16_to_f32, f32_to_f16};
pub use quantize::{quantize_row_q8_0, quantize_row_q8_k};

// Unified kernel error (`kernels/error.rs`); re-exported here under its old
// home so this module's API is unchanged.
pub use super::KernelError;

/// Quantization formats the kernels understand.
///
/// Block geometry matches `GgmlType` in `ramvamp-repack` (the repacker
/// copies these bytes unchanged, so the sizes are the GGUF sizes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    /// 4-bit k-quant super-block: 144 bytes / 256 weights.
    Q4_K,
    /// 5-bit k-quant super-block: 176 bytes / 256 weights.
    Q5_K,
    /// 6-bit k-quant super-block: 210 bytes / 256 weights.
    Q6_K,
    /// Legacy 8-bit block: 34 bytes / 32 weights.
    Q8_0,
    /// 8-bit k-quant activation format: 292 bytes / 256 weights.
    Q8_K,
}

impl QuantFormat {
    /// Weights per quantization block.
    pub const fn block_weights(self) -> usize {
        match self {
            Self::Q8_0 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
        }
    }

    /// Bytes per quantization block (packed on-disk form).
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_0 => 34,
            Self::Q8_K => 292,
        }
    }

    /// Packed bytes for one row of `in_dim` weights.
    ///
    /// Blocks span the innermost (contiguous) axis, so `in_dim` must be a
    /// multiple of [`Self::block_weights`].
    pub fn row_bytes(self, in_dim: usize) -> Result<usize, KernelError> {
        if !in_dim.is_multiple_of(self.block_weights()) {
            return Err(KernelError::IndivisibleRow {
                format: self,
                in_dim,
                block_weights: self.block_weights(),
            });
        }
        Ok(in_dim / self.block_weights() * self.block_bytes())
    }
}

// Compile-time consistency between the enum geometry and the block layout
// offsets in `blocks`.
const _: () = {
    assert!(QuantFormat::Q4_K.block_bytes() == blocks::q4k::BYTES);
    assert!(QuantFormat::Q5_K.block_bytes() == blocks::q5k::BYTES);
    assert!(QuantFormat::Q6_K.block_bytes() == blocks::q6k::BYTES);
    assert!(QuantFormat::Q8_0.block_bytes() == blocks::q8_0::BYTES);
};
