//! RMSNorm.
//!
//! Qwen3 variant (used everywhere in the v0 model, including the per-head
//! QK-RMSNorm over `[128]`): `y_i = x_i / sqrt(mean(x^2) + eps) * w_i`.
//! The weight multiplies directly — there is no `(1 + w)` term. Gemma
//! checkpoints store `w` such that the effective scale is `(1 + w)`; Gemma 4
//! (model #2) will need that variant, stubbed here as [`rmsnorm_gemma`].
//!
//! The mean-of-squares accumulates in `f64` (ggml does the same via
//! `ggml_float`); the elementwise scale is applied in `f32`.

use super::PrimitiveError;

/// Inverse RMS factor `1 / sqrt(mean(x^2) + eps)`, accumulated in `f64`.
///
/// Caller guarantees `x` is non-empty.
#[inline]
fn inv_rms(x: &[f32], eps: f32) -> f32 {
    let mut sum = 0.0f64;
    for &v in x {
        let v = f64::from(v);
        sum += v * v;
    }
    let mean = sum / x.len() as f64;
    (mean + f64::from(eps)).sqrt().recip() as f32
}

/// RMSNorm: `out_i = x_i / sqrt(mean(x^2) + eps) * weight_i`.
///
/// Qwen3/Llama convention: the weight multiplies directly (no `+1`).
/// v0 uses `eps = 1e-6` (see `docs/architecture.md`, "Model pin").
///
/// # Errors
///
/// [`PrimitiveError::Empty`] if `x` is empty (the mean is undefined);
/// [`PrimitiveError::LengthMismatch`] if `weight` or `out` differ in length
/// from `x`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<(), PrimitiveError> {
    if x.is_empty() {
        return Err(PrimitiveError::Empty { what: "rmsnorm" });
    }
    if x.len() != weight.len() {
        return Err(PrimitiveError::LengthMismatch {
            what: "rmsnorm: x vs weight",
            left: x.len(),
            right: weight.len(),
        });
    }
    if x.len() != out.len() {
        return Err(PrimitiveError::LengthMismatch {
            what: "rmsnorm: x vs out",
            left: x.len(),
            right: out.len(),
        });
    }
    let inv = inv_rms(x, eps);
    for ((y, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *y = v * inv * w;
    }
    Ok(())
}

/// In-place [`rmsnorm`]: `x_i <- x_i / sqrt(mean(x^2) + eps) * weight_i`.
///
/// Bit-identical to the out-of-place variant (same accumulation and the same
/// `v * inv * w` evaluation order).
///
/// # Errors
///
/// [`PrimitiveError::Empty`] if `x` is empty;
/// [`PrimitiveError::LengthMismatch`] if `weight` differs in length from `x`.
pub fn rmsnorm_in_place(x: &mut [f32], weight: &[f32], eps: f32) -> Result<(), PrimitiveError> {
    if x.is_empty() {
        return Err(PrimitiveError::Empty {
            what: "rmsnorm_in_place",
        });
    }
    if x.len() != weight.len() {
        return Err(PrimitiveError::LengthMismatch {
            what: "rmsnorm_in_place: x vs weight",
            left: x.len(),
            right: weight.len(),
        });
    }
    let inv = inv_rms(x, eps);
    for (v, &w) in x.iter_mut().zip(weight) {
        *v = *v * inv * w;
    }
    Ok(())
}

/// Gemma-variant RMSNorm: `out_i = x_i / sqrt(mean(x^2) + eps) * (1 + w_i)`.
///
/// Gemma checkpoints store the norm weight zero-centered, so the effective
/// scale is `(1 + w)`. Not needed for the v0 Qwen3 pin; Gemma 4 26B-A4B
/// (model #2, see `docs/architecture.md`, "v0 scope caps") requires it.
///
/// # Errors
///
/// Always returns [`PrimitiveError::Unimplemented`] for now.
pub fn rmsnorm_gemma(
    _x: &[f32],
    _weight: &[f32],
    _eps: f32,
    _out: &mut [f32],
) -> Result<(), PrimitiveError> {
    Err(PrimitiveError::Unimplemented {
        what: "rmsnorm_gemma",
    })
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Rng, assert_close};
    use super::*;

    /// Naive `f64` reference: `y_i = x_i / sqrt(mean(x^2) + eps) * w_i`.
    fn rmsnorm_ref(x: &[f32], weight: &[f32], eps: f32) -> Vec<f64> {
        let n = x.len() as f64;
        let sum: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let scale = 1.0 / (sum / n + f64::from(eps)).sqrt();
        x.iter()
            .zip(weight)
            .map(|(&v, &w)| f64::from(v) * scale * f64::from(w))
            .collect()
    }

    const EPS: f32 = 1e-6; // v0 model pin: rms_norm_eps

    fn check_against_ref(x: &[f32], weight: &[f32]) {
        let reference = rmsnorm_ref(x, weight, EPS);
        let mut out = vec![0.0f32; x.len()];
        rmsnorm(x, weight, EPS, &mut out).unwrap();
        for (&got, &want) in out.iter().zip(&reference) {
            assert_close(got, want, 1e-6, 1e-30);
        }
        // In-place variant must match the out-of-place result exactly.
        let mut inplace = x.to_vec();
        rmsnorm_in_place(&mut inplace, weight, EPS).unwrap();
        assert_eq!(out, inplace);
    }

    #[test]
    fn matches_f64_reference_random() {
        let mut rng = Rng::new(0xA11CE);
        for &n in &[1usize, 2, 128, 2048] {
            let x = rng.vec_in(n, -4.0, 4.0);
            let w = rng.vec_in(n, 0.25, 2.0);
            check_against_ref(&x, &w);
        }
    }

    #[test]
    fn matches_f64_reference_large_magnitudes() {
        let mut rng = Rng::new(0xBEEF);
        let x = rng.vec_in(2048, -1e18, 1e18);
        let w = rng.vec_in(2048, 0.5, 1.5);
        check_against_ref(&x, &w);
    }

    #[test]
    fn matches_f64_reference_all_equal() {
        let x = vec![3.0f32; 512];
        let w = vec![1.0f32; 512];
        // rms = 3 (up to eps), so outputs are ~1.
        check_against_ref(&x, &w);
    }

    #[test]
    fn matches_f64_reference_single_element() {
        check_against_ref(&[7.5f32], &[2.0f32]);
        check_against_ref(&[-1e-3f32], &[1.0f32]);
    }

    #[test]
    fn matches_f64_reference_denormals() {
        // Sub-normal inputs: mean(x^2) underflows toward 0, eps dominates.
        let x = vec![1.0e-40f32, -2.0e-40, 5.0e-41, 1.4e-45];
        let w = vec![1.0f32; 4];
        check_against_ref(&x, &w);
    }

    #[test]
    fn weight_scales_directly_no_plus_one() {
        // Qwen3 convention: w = 0 must zero the output (Gemma's (1+w) would not).
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [0.0f32; 4];
        let mut out = [1.0f32; 4];
        rmsnorm(&x, &w, EPS, &mut out).unwrap();
        assert_eq!(out, [0.0f32; 4]);
    }

    #[test]
    fn qk_norm_shape_128() {
        // Per-head QK-RMSNorm operates on [128] slices (model pin).
        let mut rng = Rng::new(0x128);
        let x = rng.vec_in(128, -10.0, 10.0);
        let w = rng.vec_in(128, 0.5, 1.5);
        check_against_ref(&x, &w);
    }

    #[test]
    fn length_mismatch_errors() {
        let mut out = [0.0f32; 4];
        assert_eq!(
            rmsnorm(&[1.0; 4], &[1.0; 3], EPS, &mut out),
            Err(PrimitiveError::LengthMismatch {
                what: "rmsnorm: x vs weight",
                left: 4,
                right: 3,
            })
        );
        assert_eq!(
            rmsnorm(&[1.0; 4], &[1.0; 4], EPS, &mut out[..2]),
            Err(PrimitiveError::LengthMismatch {
                what: "rmsnorm: x vs out",
                left: 4,
                right: 2,
            })
        );
        assert_eq!(
            rmsnorm_in_place(&mut out, &[1.0; 3], EPS),
            Err(PrimitiveError::LengthMismatch {
                what: "rmsnorm_in_place: x vs weight",
                left: 4,
                right: 3,
            })
        );
    }

    #[test]
    fn empty_input_errors() {
        let mut out: [f32; 0] = [];
        assert_eq!(
            rmsnorm(&[], &[], EPS, &mut out),
            Err(PrimitiveError::Empty { what: "rmsnorm" })
        );
        assert_eq!(
            rmsnorm_in_place(&mut out, &[], EPS),
            Err(PrimitiveError::Empty {
                what: "rmsnorm_in_place",
            })
        );
    }

    #[test]
    fn gemma_variant_is_unimplemented() {
        let mut out = [0.0f32; 2];
        assert_eq!(
            rmsnorm_gemma(&[1.0, 2.0], &[1.0, 1.0], EPS, &mut out),
            Err(PrimitiveError::Unimplemented {
                what: "rmsnorm_gemma",
            })
        );
    }
}
