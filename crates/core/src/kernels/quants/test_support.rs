//! Test-only helpers: weight-format quantizers that produce valid packed
//! blocks, and a tiny seeded RNG (no `rand` dependency).
//!
//! The quantizers approximate ggml's `quantize_row_q4_K` family: same
//! block layout and scale packing, but simple one-pass scale selection
//! (ggml refines scales iteratively). They exist to generate valid,
//! representative blocks for round-trip and ground-truth tests -- the
//! runtime never quantizes weights (the repacker copies GGUF bytes
//! unchanged).
//!
//! Scale selection detail: 6-bit sub-scales round *up* (`ceil`), so the
//! decoded step is never smaller than the ideal step and requantizing
//! against the decoded scales cannot clamp-amplify the error. Round-trip
//! error is then at most half a step plus half a min-step; tests assert
//! one full step plus one min-step for slack.

use super::BlockQ8_0;
use super::blocks::{QK_K, q4k, q5k, q6k};
use super::f16::{f16_to_f32, f32_to_f16};
use super::quantize::quantize_row_q8_0;

/// Deterministic 64-bit LCG (Knuth MMIX constants), seeded explicitly.
pub(super) struct Lcg(pub u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform-ish f32 in [-1, 1) from the LCG's high bits (low LCG bits
    /// are weak).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Pack eight 6-bit (scale, min) pairs into the 12-byte k-quant scale
/// field; exact inverse of `get_scale_min_k4`, ordered like ggml's
/// `quantize_row_q4_K_ref`.
fn pack_scales_k4(ls: &[u8; 8], lm: &[u8; 8]) -> [u8; 12] {
    let mut s = [0u8; 12];
    for j in 0..8 {
        if j < 4 {
            s[j] = ls[j] & 63;
            s[j + 4] = lm[j] & 63;
        } else {
            s[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4);
            s[j - 4] |= (ls[j] >> 4) << 6;
            s[j] |= (lm[j] >> 4) << 6;
        }
    }
    s
}

/// Shared affine sub-block scale selection for q4_K/q5_K.
///
/// Returns (packed scales, decoded d, decoded dmin, per-sub-block decoded
/// (step, min) pairs) for eight 32-weight sub-blocks with `levels`
/// quantization levels (15 for q4_K, 31 for q5_K).
#[allow(clippy::type_complexity)]
fn affine_scales(x: &[f32], levels: f32) -> ([u8; 12], f32, f32, [(f32, f32); 8]) {
    let mut scale_f = [0f32; 8];
    let mut min_f = [0f32; 8];
    for j in 0..8 {
        let sub = &x[32 * j..32 * j + 32];
        let vmin = sub.iter().copied().fold(f32::INFINITY, f32::min);
        let vmax = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        min_f[j] = (-vmin).max(0.0);
        scale_f[j] = ((vmax + min_f[j]) / levels).max(0.0);
    }
    let max_scale = scale_f.iter().copied().fold(0f32, f32::max);
    let max_min = min_f.iter().copied().fold(0f32, f32::max);
    let d = f16_to_f32(f32_to_f16(max_scale / 63.0));
    let dmin = f16_to_f32(f32_to_f16(max_min / 63.0));
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    for j in 0..8 {
        if d > 0.0 {
            ls[j] = (scale_f[j] / d).ceil().clamp(0.0, 63.0) as u8;
        }
        if dmin > 0.0 {
            lm[j] = (min_f[j] / dmin).round().clamp(0.0, 63.0) as u8;
        }
    }
    let mut decoded = [(0f32, 0f32); 8];
    for j in 0..8 {
        decoded[j] = (d * f32::from(ls[j]), dmin * f32::from(lm[j]));
    }
    (pack_scales_k4(&ls, &lm), d, dmin, decoded)
}

/// Quantize sub-block values against decoded (step, min): the q index in
/// [0, levels].
fn affine_q(v: f32, step: f32, min: f32, levels: f32) -> u8 {
    if step <= 0.0 {
        return 0;
    }
    ((v + min) / step).round().clamp(0.0, levels) as u8
}

/// Quantize 256-multiples of f32 into packed Q4_K blocks (144 B each).
pub(super) fn quantize_row_q4_k_test(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK_K, 0);
    let mut out = Vec::with_capacity(x.len() / QK_K * q4k::BYTES);
    for xs in x.chunks_exact(QK_K) {
        let (scales, d, dmin, decoded) = affine_scales(xs, 15.0);
        let mut q = [0u8; QK_K];
        for (j, &(step, min)) in decoded.iter().enumerate() {
            for l in 0..32 {
                q[32 * j + l] = affine_q(xs[32 * j + l], step, min, 15.0);
            }
        }
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        out.extend_from_slice(&scales);
        for span in 0..4 {
            for l in 0..32 {
                out.push(q[64 * span + l] | (q[64 * span + 32 + l] << 4));
            }
        }
    }
    out
}

/// Quantize 256-multiples of f32 into packed Q5_K blocks (176 B each).
pub(super) fn quantize_row_q5_k_test(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK_K, 0);
    let mut out = Vec::with_capacity(x.len() / QK_K * q5k::BYTES);
    for xs in x.chunks_exact(QK_K) {
        let (scales, d, dmin, decoded) = affine_scales(xs, 31.0);
        let mut q = [0u8; QK_K];
        for (j, &(step, min)) in decoded.iter().enumerate() {
            for l in 0..32 {
                q[32 * j + l] = affine_q(xs[32 * j + l], step, min, 31.0);
            }
        }
        let mut qh = [0u8; 32];
        for j in 0..8 {
            for l in 0..32 {
                if q[32 * j + l] >= 16 {
                    qh[l] |= 1 << j;
                }
            }
        }
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        out.extend_from_slice(&scales);
        out.extend_from_slice(&qh);
        for span in 0..4 {
            for l in 0..32 {
                out.push((q[64 * span + l] & 0xF) | ((q[64 * span + 32 + l] & 0xF) << 4));
            }
        }
    }
    out
}

/// Quantize 256-multiples of f32 into packed Q6_K blocks (210 B each).
pub(super) fn quantize_row_q6_k_test(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK_K, 0);
    let mut out = Vec::with_capacity(x.len() / QK_K * q6k::BYTES);
    for xs in x.chunks_exact(QK_K) {
        // Sixteen symmetric 16-weight sub-blocks, q in [-32, 31] (we use
        // [-31, 31] so +/-amax quantize symmetrically), signed i8 scales.
        let mut scale_f = [0f32; 16];
        for (g, s) in scale_f.iter_mut().enumerate() {
            let amax = xs[16 * g..16 * g + 16]
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()));
            *s = amax / 31.0;
        }
        let max_scale = scale_f.iter().copied().fold(0f32, f32::max);
        let d = f16_to_f32(f32_to_f16(max_scale / 127.0));
        let mut ls = [0u8; 16];
        for g in 0..16 {
            if d > 0.0 {
                ls[g] = (scale_f[g] / d).ceil().clamp(0.0, 127.0) as u8;
            }
        }
        let mut q = [0u8; QK_K]; // Biased: stored value = q + 32 in 0..64.
        for g in 0..16 {
            let step = d * f32::from(ls[g]);
            for l in 0..16 {
                let v = if step > 0.0 {
                    (xs[16 * g + l] / step).round().clamp(-32.0, 31.0) as i32
                } else {
                    0
                };
                q[16 * g + l] = (v + 32) as u8;
            }
        }
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for half in 0..2 {
            let base = 128 * half;
            for l in 0..32 {
                let v1 = q[base + l];
                let v2 = q[base + l + 32];
                let v3 = q[base + l + 64];
                let v4 = q[base + l + 96];
                ql[64 * half + l] = (v1 & 0xF) | ((v3 & 0xF) << 4);
                ql[64 * half + l + 32] = (v2 & 0xF) | ((v4 & 0xF) << 4);
                qh[32 * half + l] =
                    (v1 >> 4) | ((v2 >> 4) << 2) | ((v3 >> 4) << 4) | ((v4 >> 4) << 6);
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        for &s in &ls {
            out.push(s); // Positive i8 scales; negative covered by raw-byte tests.
        }
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    out
}

/// Quantize 32-multiples of f32 into packed Q8_0 blocks (34 B each),
/// via the production activation quantizer.
pub(super) fn quantize_row_q8_0_test(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % 32, 0);
    let mut blocks = vec![BlockQ8_0::default(); x.len() / 32];
    quantize_row_q8_0(x, &mut blocks).expect("length checked");
    let mut out = Vec::with_capacity(blocks.len() * 34);
    for b in &blocks {
        // b.d is already f16-rounded, so this conversion is exact.
        out.extend_from_slice(&f32_to_f16(b.d).to_le_bytes());
        out.extend(b.qs.iter().map(|&q| q as u8));
    }
    out
}
