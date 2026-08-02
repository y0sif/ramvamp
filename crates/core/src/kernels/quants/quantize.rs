//! Activation quantizers: f32 rows into Q8_0 / Q8_K blocks.
//!
//! Both mirror ggml's reference quantizers (`quantize_row_q8_0_ref`,
//! `quantize_row_q8_K_ref` in `ggml-quants.c`) so our activation integers
//! match llama.cpp's bit-for-bit on identical inputs.

use super::blocks::{BlockQ8_0, BlockQ8K, QK_K, QK8_0};
use super::f16::{f16_to_f32, f32_to_f16};
use super::{KernelError, QuantFormat};

/// Check that `floats` fills `out_blocks` blocks of `width` exactly.
fn check_len(
    format: QuantFormat,
    floats: usize,
    out_blocks: usize,
    width: usize,
) -> Result<(), KernelError> {
    if floats != out_blocks * width {
        return Err(KernelError::QuantizeLenMismatch {
            format,
            floats,
            out_blocks,
        });
    }
    Ok(())
}

/// Quantize a row of f32 activations into Q8_0 blocks (32 values each).
///
/// Mirrors ggml's `quantize_row_q8_0_ref`: `d = amax / 127`, values scaled
/// by the reciprocal of the *unrounded* `d` and rounded half away from zero
/// (`roundf`). The stored [`BlockQ8_0::d`] is the f16-rounded value ggml
/// would write to its `ggml_half` field, so downstream arithmetic matches
/// llama.cpp exactly.
///
/// `x.len()` must equal `out.len() * 32`.
pub fn quantize_row_q8_0(x: &[f32], out: &mut [BlockQ8_0]) -> Result<(), KernelError> {
    check_len(QuantFormat::Q8_0, x.len(), out.len(), QK8_0)?;
    for (xs, block) in x.chunks_exact(QK8_0).zip(out.iter_mut()) {
        let mut amax = 0f32;
        for &v in xs {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = f16_to_f32(f32_to_f16(d));
        for (q, &v) in block.qs.iter_mut().zip(xs) {
            // |v * id| <= 127 up to rounding error, so the saturating
            // float-to-int cast can never actually clip.
            *q = (v * id).round() as i8;
        }
    }
    Ok(())
}

/// Quantize a row of f32 activations into Q8_K blocks (256 values each),
/// filling per-16-group `bsums`.
///
/// Mirrors ggml's `quantize_row_q8_K_ref`: the scale anchors the *signed*
/// value of largest magnitude to -127 (`iscale = -127 / max`), values round
/// half to even (ggml's `nearest_int`), the quantized value is clamped to
/// at most 127, and `d = 1 / iscale` is stored as f32 (`block_q8_K.d` is a
/// float upstream, no f16 rounding). An all-zero input produces a zeroed
/// block; ggml leaves `bsums` uninitialized in that case (harmless there
/// because `d = 0`), we zero them.
///
/// `x.len()` must equal `out.len() * 256`.
pub fn quantize_row_q8_k(x: &[f32], out: &mut [BlockQ8K]) -> Result<(), KernelError> {
    check_len(QuantFormat::Q8_K, x.len(), out.len(), QK_K)?;
    for (xs, block) in x.chunks_exact(QK_K).zip(out.iter_mut()) {
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in xs {
            let av = v.abs();
            if av > amax {
                amax = av;
                max = v;
            }
        }
        if amax == 0.0 {
            *block = BlockQ8K::default();
            continue;
        }
        let iscale = -127.0 / max;
        for (q, &v) in block.qs.iter_mut().zip(xs) {
            // nearest_int(iscale * v) is in [-128, 127] because
            // |iscale * v| <= 127 up to rounding error; the clamp only
            // enforces the same MIN(127, v) cap ggml applies.
            let n = (iscale * v).round_ties_even() as i32;
            *q = n.clamp(-128, 127) as i8;
        }
        for (g, sum) in block.bsums.iter_mut().enumerate() {
            let mut s = 0i32;
            for &q in &block.qs[16 * g..16 * g + 16] {
                s += i32::from(q);
            }
            // Sum of 16 int8 values: always fits i16.
            *sum = s as i16;
        }
        block.d = 1.0 / iscale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_scales_and_signs() {
        let mut x = [0f32; 32];
        x[0] = 2.0; // amax
        x[1] = -1.0;
        x[31] = 0.5;
        let mut out = [BlockQ8_0::default()];
        quantize_row_q8_0(&x, &mut out).unwrap();
        let b = &out[0];
        // d = 2/127 rounded through f16.
        assert_eq!(b.d, f16_to_f32(f32_to_f16(2.0 / 127.0)));
        assert_eq!(b.qs[0], 127);
        assert_eq!(b.qs[1], -64); // roundf(-63.5) = -64: half away from zero.
        assert_eq!(b.qs[31], 32); // roundf(31.75) = 32.
        assert_eq!(b.qs[2], 0);
    }

    #[test]
    fn q8_0_zero_row() {
        let x = [0f32; 64];
        let mut out = [BlockQ8_0 {
            d: 9.0,
            qs: [7; 32],
        }; 2];
        quantize_row_q8_0(&x, &mut out).unwrap();
        for b in &out {
            assert_eq!(b.d, 0.0);
            assert!(b.qs.iter().all(|&q| q == 0));
        }
    }

    #[test]
    fn q8_k_anchors_extreme_to_minus_127() {
        let mut x = [0f32; 256];
        x[10] = -3.0; // Largest magnitude, negative: iscale = 127/3 > 0.
        x[20] = 1.5;
        let mut out = [BlockQ8K::default()];
        quantize_row_q8_k(&x, &mut out).unwrap();
        let b = &out[0];
        // iscale = -127 / -3 = +127/3: the extreme lands on -127, d > 0.
        assert_eq!(b.qs[10], -127);
        assert_eq!(b.qs[20], 64); // 127/3 * 1.5 = 63.5, ties to even 64.
        assert_eq!(b.d, 1.0 / (-127.0f32 / -3.0));
    }

    #[test]
    fn q8_k_positive_extreme_maps_to_minus_127() {
        let mut x = [0f32; 256];
        x[0] = 4.0;
        x[1] = -2.0;
        let mut out = [BlockQ8K::default()];
        quantize_row_q8_k(&x, &mut out).unwrap();
        // iscale = -127/4: the positive extreme lands on -127, d < 0.
        assert_eq!(out[0].qs[0], -127);
        assert_eq!(out[0].qs[1], 64); // -127/4 * -2 = 63.5, ties to even 64.
        assert_eq!(out[0].d, 1.0 / (-127.0f32 / 4.0));
    }

    #[test]
    fn q8_k_zero_row_zeroes_everything() {
        let x = [0f32; 256];
        let mut out = [BlockQ8K {
            d: 5.0,
            qs: [3; 256],
            bsums: [99; 16],
        }];
        quantize_row_q8_k(&x, &mut out).unwrap();
        assert_eq!(out[0], BlockQ8K::default());
    }

    #[test]
    fn quantize_len_mismatch_errors() {
        let x = [0f32; 100];
        let mut out8 = [BlockQ8_0::default(); 3];
        let err = quantize_row_q8_0(&x, &mut out8).unwrap_err();
        assert_eq!(
            err,
            KernelError::QuantizeLenMismatch {
                format: QuantFormat::Q8_0,
                floats: 100,
                out_blocks: 3,
            }
        );
        let mut outk = [BlockQ8K::default()];
        let err = quantize_row_q8_k(&x, &mut outk).unwrap_err();
        assert_eq!(
            err,
            KernelError::QuantizeLenMismatch {
                format: QuantFormat::Q8_K,
                floats: 100,
                out_blocks: 1,
            }
        );
    }
}
