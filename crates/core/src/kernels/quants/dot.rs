//! Scalar reference dot products: packed weight rows against quantized
//! activation rows.
//!
//! Each mirrors the scalar (non-SIMD) fallback of the corresponding
//! `ggml_vec_dot_*` in ggml's `ggml-quants.c`: expand the packed weights to
//! int8 in ggml's order, accumulate `weight * activation` products in
//! integers per sub-block, apply the sub-block scale while still in
//! integers, and only then touch floats -- with the k-quant mins folded in
//! through the activation `bsums`. This is the ground truth the AVX2 ports
//! will be validated against, and it keeps us numerically aligned with
//! llama.cpp.
//!
//! One deliberate deviation from ggml's code shape: ggml accumulates into
//! eight strided `int32` lanes (`aux16`/`aux32`) and eight float `sums`
//! lanes that are added together at the end; we keep a single `i32` per
//! sub-block and one running f32. The integer result is identical (integer
//! addition is associative and nothing can overflow `i32`: worst case
//! `|scale| * |q| * 127 * block` is well under 2^31), so only the float
//! summation order differs -- immaterial next to the documented kernel
//! tolerance and irrelevant to the integer-exactness the AVX2 comparison
//! relies on.

use super::blocks::{
    BlockQ8_0, BlockQ8K, QK_K, f16_at, get_scale_min_k4, q4k, q5k, q6k, q8_0, scales_at,
};
use super::{KernelError, QuantFormat};

/// Validate a weight row against an activation row and split it into
/// blocks.
fn weight_blocks<'a>(
    bytes: &'a [u8],
    format: QuantFormat,
    activation_blocks: usize,
) -> Result<std::slice::ChunksExact<'a, u8>, KernelError> {
    let block_bytes = format.block_bytes();
    if !bytes.len().is_multiple_of(block_bytes) {
        return Err(KernelError::RowBytesNotBlockMultiple {
            format,
            len: bytes.len(),
            block_bytes,
        });
    }
    let weight_blocks = bytes.len() / block_bytes;
    if weight_blocks != activation_blocks {
        return Err(KernelError::BlockCountMismatch {
            weight_blocks,
            activation_blocks,
        });
    }
    Ok(bytes.chunks_exact(block_bytes))
}

/// The shared k-quant tail for q4_K/q5_K: scaled integer dot over eight
/// 32-weight sub-blocks plus the min correction via `bsums`.
///
/// Returns the block's contribution `d*sumi - dmin*sumi_mins` exactly as
/// ggml's scalar `vec_dot_q4_K_q8_K` / `vec_dot_q5_K_q8_K` compute it.
#[inline]
fn k_quant_affine_block(
    aux: &[i8; QK_K],
    scales: &[u8; 12],
    y: &BlockQ8K,
    d: f32,
    dmin: f32,
) -> f32 {
    let mut sc = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..8 {
        let (s, m) = get_scale_min_k4(j, scales);
        sc[j] = s;
        mins[j] = m;
    }
    // Min correction: bsums are per-16 sums, mins are per-32, so each min
    // multiplies two adjacent bsums.
    let mut sumi_mins = 0i32;
    for (g, &bs) in y.bsums.iter().enumerate() {
        sumi_mins += i32::from(bs) * i32::from(mins[g / 2]);
    }
    // Scaled integer dot: per 32-weight sub-block, sum the int8 products,
    // then multiply by the 6-bit sub-block scale.
    let mut sumi = 0i32;
    for j in 0..8 {
        let mut s = 0i32;
        for l in 0..32 {
            s += i32::from(aux[32 * j + l]) * i32::from(y.qs[32 * j + l]);
        }
        sumi += i32::from(sc[j]) * s;
    }
    d * sumi as f32 - dmin * sumi_mins as f32
}

/// Dot product of a packed Q4_K weight row with Q8_K activations.
///
/// Scalar reference mirroring ggml's `ggml_vec_dot_q4_K_q8_K` fallback.
/// `weight_row.len()` must be `acts.len() * 144`.
pub fn vec_dot_q4_k_q8_k(weight_row: &[u8], acts: &[BlockQ8K]) -> Result<f32, KernelError> {
    let blocks = weight_blocks(weight_row, QuantFormat::Q4_K, acts.len())?;
    let mut sumf = 0f32;
    for (block, y) in blocks.zip(acts) {
        let d = f16_at(block, q4k::D) * y.d;
        let dmin = f16_at(block, q4k::DMIN) * y.d;
        let qs = &block[q4k::QS..q4k::BYTES];
        // Expand nibbles in ggml order: per 64-weight span, 32 low nibbles
        // then 32 high nibbles.
        let mut aux = [0i8; QK_K];
        for span in 0..4 {
            let q = &qs[32 * span..32 * span + 32];
            for l in 0..32 {
                aux[64 * span + l] = (q[l] & 0xF) as i8;
                aux[64 * span + 32 + l] = (q[l] >> 4) as i8;
            }
        }
        sumf += k_quant_affine_block(&aux, scales_at(block, q4k::SCALES), y, d, dmin);
    }
    Ok(sumf)
}

/// Dot product of a packed Q5_K weight row with Q8_K activations.
///
/// Scalar reference mirroring ggml's `ggml_vec_dot_q5_K_q8_K` fallback.
/// `weight_row.len()` must be `acts.len() * 176`.
pub fn vec_dot_q5_k_q8_k(weight_row: &[u8], acts: &[BlockQ8K]) -> Result<f32, KernelError> {
    let blocks = weight_blocks(weight_row, QuantFormat::Q5_K, acts.len())?;
    let mut sumf = 0f32;
    for (block, y) in blocks.zip(acts) {
        let d = f16_at(block, q5k::D) * y.d;
        let dmin = f16_at(block, q5k::DMIN) * y.d;
        let qh = &block[q5k::QH..q5k::QS];
        let qs = &block[q5k::QS..q5k::BYTES];
        // Expand to 5 bits in ggml order: nibbles as in q4_K, plus bit j of
        // qh[l] as the 5th bit of sub-block j's weight l.
        let mut aux = [0i8; QK_K];
        for j in 0..8 {
            let q = &qs[32 * (j / 2)..32 * (j / 2) + 32];
            let mask = 1u8 << j;
            for l in 0..32 {
                let low = if j % 2 == 0 { q[l] & 0xF } else { q[l] >> 4 };
                let hi = if qh[l] & mask != 0 { 16 } else { 0 };
                aux[32 * j + l] = (low + hi) as i8;
            }
        }
        sumf += k_quant_affine_block(&aux, scales_at(block, q5k::SCALES), y, d, dmin);
    }
    Ok(sumf)
}

/// Dot product of a packed Q6_K weight row with Q8_K activations.
///
/// Scalar reference mirroring ggml's `ggml_vec_dot_q6_K_q8_K` fallback:
/// sixteen 16-weight sub-blocks with signed int8 scales, no mins.
/// `weight_row.len()` must be `acts.len() * 210`.
pub fn vec_dot_q6_k_q8_k(weight_row: &[u8], acts: &[BlockQ8K]) -> Result<f32, KernelError> {
    let blocks = weight_blocks(weight_row, QuantFormat::Q6_K, acts.len())?;
    let mut sumf = 0f32;
    for (block, y) in blocks.zip(acts) {
        // Expand to signed 6-bit values in ggml order (two 128-weight
        // halves of four interleaved 32-weight groups).
        let mut aux = [0i8; QK_K];
        for half in 0..2 {
            let ql = &block[q6k::QL + 64 * half..];
            let qh = &block[q6k::QH + 32 * half..];
            let a = &mut aux[128 * half..];
            for l in 0..32 {
                let h = i32::from(qh[l]);
                a[l] = ((i32::from(ql[l] & 0xF) | ((h & 3) << 4)) - 32) as i8;
                a[l + 32] = ((i32::from(ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32) as i8;
                a[l + 64] = ((i32::from(ql[l] >> 4) | (((h >> 4) & 3) << 4)) - 32) as i8;
                a[l + 96] = ((i32::from(ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32) as i8;
            }
        }
        // Scaled integer dot over sixteen 16-weight sub-blocks.
        let mut sumi = 0i32;
        for g in 0..16 {
            let scale = i32::from(block[q6k::SCALES + g] as i8);
            let mut s = 0i32;
            for l in 0..16 {
                s += i32::from(aux[16 * g + l]) * i32::from(y.qs[16 * g + l]);
            }
            sumi += scale * s;
        }
        sumf += f16_at(block, q6k::D) * y.d * sumi as f32;
    }
    Ok(sumf)
}

/// Dot product of a packed Q8_0 weight row with Q8_0 activations.
///
/// Scalar reference mirroring ggml's `ggml_vec_dot_q8_0_q8_0` fallback.
/// `weight_row.len()` must be `acts.len() * 34`.
pub fn vec_dot_q8_0_q8_0(weight_row: &[u8], acts: &[BlockQ8_0]) -> Result<f32, KernelError> {
    let blocks = weight_blocks(weight_row, QuantFormat::Q8_0, acts.len())?;
    let mut sumf = 0f32;
    for (block, y) in blocks.zip(acts) {
        let mut sumi = 0i32;
        for (l, &b) in block[q8_0::QS..q8_0::BYTES].iter().enumerate() {
            sumi += i32::from(b as i8) * i32::from(y.qs[l]);
        }
        sumf += sumi as f32 * (f16_at(block, q8_0::D) * y.d);
    }
    Ok(sumf)
}
