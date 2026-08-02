//! Scalar reference dequantizers, one per weight format.
//!
//! Each mirrors the corresponding `dequantize_row_*` in ggml's
//! `ggml-quants.c`, including iteration order, so the outputs are
//! bit-identical to llama.cpp's scalar dequantization of the same bytes.
//! These are correctness references for the dot kernels and later
//! validation tooling, not hot paths.

use super::blocks::{QK_K, QK8_0, f16_at, get_scale_min_k4, q4k, q5k, q6k, q8_0, scales_at};
use super::{KernelError, QuantFormat};

/// Split a weight row into whole blocks, or fail with a typed error.
fn blocks(
    bytes: &[u8],
    format: QuantFormat,
) -> Result<std::slice::ChunksExact<'_, u8>, KernelError> {
    let block_bytes = format.block_bytes();
    if bytes.len() % block_bytes != 0 {
        return Err(KernelError::RowBytesNotBlockMultiple {
            format,
            len: bytes.len(),
            block_bytes,
        });
    }
    Ok(bytes.chunks_exact(block_bytes))
}

/// Dequantize a row of packed Q4_K blocks (144 bytes / 256 weights each).
///
/// `w = d*sc*q - dmin*m` per 32-weight sub-block; mirrors ggml's
/// `dequantize_row_q4_K`.
pub fn dequantize_row_q4_k(bytes: &[u8]) -> Result<Vec<f32>, KernelError> {
    let blocks = blocks(bytes, QuantFormat::Q4_K)?;
    let mut out = Vec::with_capacity(blocks.len() * QK_K);
    for block in blocks {
        let d = f16_at(block, q4k::D);
        let min = f16_at(block, q4k::DMIN);
        let scales = scales_at(block, q4k::SCALES);
        let qs = &block[q4k::QS..q4k::BYTES];
        // Four 64-weight spans; each consumes 32 qs bytes (low nibbles then
        // high nibbles) and two (scale, min) pairs.
        for span in 0..4 {
            let (sc1, m1) = get_scale_min_k4(2 * span, scales);
            let (sc2, m2) = get_scale_min_k4(2 * span + 1, scales);
            let d1 = d * f32::from(sc1);
            let n1 = min * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let n2 = min * f32::from(m2);
            let q = &qs[32 * span..32 * span + 32];
            for &b in q {
                out.push(d1 * f32::from(b & 0xF) - n1);
            }
            for &b in q {
                out.push(d2 * f32::from(b >> 4) - n2);
            }
        }
    }
    Ok(out)
}

/// Dequantize a row of packed Q5_K blocks (176 bytes / 256 weights each).
///
/// Q4_K's affine scheme with a 5th bit per weight pulled from `qh`;
/// mirrors ggml's `dequantize_row_q5_K` (the `u1`/`u2` shifting-mask
/// ordering).
pub fn dequantize_row_q5_k(bytes: &[u8]) -> Result<Vec<f32>, KernelError> {
    let blocks = blocks(bytes, QuantFormat::Q5_K)?;
    let mut out = Vec::with_capacity(blocks.len() * QK_K);
    for block in blocks {
        let d = f16_at(block, q5k::D);
        let min = f16_at(block, q5k::DMIN);
        let scales = scales_at(block, q5k::SCALES);
        let qh = &block[q5k::QH..q5k::QS];
        let qs = &block[q5k::QS..q5k::BYTES];
        for span in 0..4 {
            let (sc1, m1) = get_scale_min_k4(2 * span, scales);
            let (sc2, m2) = get_scale_min_k4(2 * span + 1, scales);
            let d1 = d * f32::from(sc1);
            let n1 = min * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let n2 = min * f32::from(m2);
            // qh is indexed 0..32 in every span; the span selects the bit.
            let u1 = 1u8 << (2 * span);
            let u2 = 2u8 << (2 * span);
            let q = &qs[32 * span..32 * span + 32];
            for (l, &b) in q.iter().enumerate() {
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                out.push(d1 * f32::from((b & 0xF) + hi) - n1);
            }
            for (l, &b) in q.iter().enumerate() {
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                out.push(d2 * f32::from((b >> 4) + hi) - n2);
            }
        }
    }
    Ok(out)
}

/// Dequantize a row of packed Q6_K blocks (210 bytes / 256 weights each).
///
/// `w = d * scales[sub16] * q`, `q` in [-32, 31]; mirrors ggml's
/// `dequantize_row_q6_K` (two 128-weight halves, each writing four
/// interleaved 32-weight groups).
pub fn dequantize_row_q6_k(bytes: &[u8]) -> Result<Vec<f32>, KernelError> {
    let blocks = blocks(bytes, QuantFormat::Q6_K)?;
    let n_blocks = blocks.len();
    let mut out = vec![0f32; n_blocks * QK_K];
    for (i, block) in blocks.enumerate() {
        let d = f16_at(block, q6k::D);
        let y = &mut out[i * QK_K..(i + 1) * QK_K];
        // Two 128-weight halves; each uses 64 ql bytes, 32 qh bytes, and
        // 8 of the 16 sub-block scales.
        for half in 0..2 {
            let ql = &block[q6k::QL + 64 * half..];
            let qh = &block[q6k::QH + 32 * half..];
            let sc = &block[q6k::SCALES + 8 * half..];
            let y = &mut y[128 * half..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = (i32::from(ql[l] & 0xF) | ((i32::from(qh[l]) & 3) << 4)) - 32;
                let q2 = (i32::from(ql[l + 32] & 0xF) | ((i32::from(qh[l] >> 2) & 3) << 4)) - 32;
                let q3 = (i32::from(ql[l] >> 4) | ((i32::from(qh[l] >> 4) & 3) << 4)) - 32;
                let q4 = (i32::from(ql[l + 32] >> 4) | ((i32::from(qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * f32::from(sc[is] as i8) * q1 as f32;
                y[l + 32] = d * f32::from(sc[is + 2] as i8) * q2 as f32;
                y[l + 64] = d * f32::from(sc[is + 4] as i8) * q3 as f32;
                y[l + 96] = d * f32::from(sc[is + 6] as i8) * q4 as f32;
            }
        }
    }
    Ok(out)
}

/// Dequantize a row of packed Q8_0 blocks (34 bytes / 32 weights each).
///
/// `w = d * q`; mirrors ggml's `dequantize_row_q8_0`.
pub fn dequantize_row_q8_0(bytes: &[u8]) -> Result<Vec<f32>, KernelError> {
    let blocks = blocks(bytes, QuantFormat::Q8_0)?;
    let mut out = Vec::with_capacity(blocks.len() * QK8_0);
    for block in blocks {
        let d = f16_at(block, q8_0::D);
        for &b in &block[q8_0::QS..q8_0::BYTES] {
            out.push(d * f32::from(b as i8));
        }
    }
    Ok(out)
}
