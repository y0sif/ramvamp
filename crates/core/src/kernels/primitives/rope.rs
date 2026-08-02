//! NeoX-style rotary position embedding (RoPE).
//!
//! Convention: the "rotate_half" / NeoX layout, exactly as in HF
//! `transformers` (`apply_rotary_pos_emb` with `rotate_half` in
//! `models/llama/modeling_llama.py`, reused by `models/qwen3_moe`) and ggml's
//! `GGML_ROPE_TYPE_NEOX` mode (`ggml_rope_ext`), which llama.cpp selects for
//! Qwen3. Rotation pairs are `(i, i + head_dim/2)` — *not* the interleaved
//! `(2i, 2i+1)` pairing of the original GPT-J/RoFormer formulation.
//!
//! For each `i` in `0..head_dim/2`:
//!
//! ```text
//! inv_freq_i = theta_base^(-2i / head_dim)
//! angle      = position * inv_freq_i
//! x[i]            <- x[i] * cos(angle) - x[i + half] * sin(angle)
//! x[i + half]     <- x[i] * sin(angle) + x[i + half] * cos(angle)
//! ```
//!
//! v0 model pin: `head_dim = 128`, `rope_theta = 1e7`, no scaling/YaRN.
//!
//! Precision: `inv_freq`, `angle`, and `sin`/`cos` are computed in `f64`,
//! then applied in `f32`. (ggml computes angles in `f32`; going through `f64`
//! only tightens the result toward the exact rotation, which matters for the
//! logit-level KL gate at large positions.) No frequency cache yet — this is
//! not the perf-critical path; add one behind a profile if it shows up.

use super::PrimitiveError;

/// Rotate one head in place. Caller guarantees `head.len() == 2 * half` and
/// `half > 0`.
fn rotate_head(head: &mut [f32], half: usize, position: u32, theta_base: f32) {
    let pos = f64::from(position);
    let base = f64::from(theta_base);
    let dim = (2 * half) as f64;
    let (lo, hi) = head.split_at_mut(half);
    for (i, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
        let inv_freq = base.powf(-2.0 * i as f64 / dim);
        let (sin, cos) = (pos * inv_freq).sin_cos();
        let (sin, cos) = (sin as f32, cos as f32);
        let x0 = *a;
        let x1 = *b;
        *a = x0 * cos - x1 * sin;
        *b = x0 * sin + x1 * cos;
    }
}

/// Apply NeoX RoPE in place to one head: `q_or_k` is a contiguous
/// `[head_dim]` slice of a query or key head at sequence `position`.
///
/// # Errors
///
/// [`PrimitiveError::InvalidHeadDim`] if `head_dim` is zero or odd;
/// [`PrimitiveError::LengthMismatch`] if `q_or_k.len() != head_dim`.
pub fn rope_neox(
    q_or_k: &mut [f32],
    head_dim: usize,
    position: u32,
    theta_base: f32,
) -> Result<(), PrimitiveError> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(PrimitiveError::InvalidHeadDim { head_dim });
    }
    if q_or_k.len() != head_dim {
        return Err(PrimitiveError::LengthMismatch {
            what: "rope_neox: q_or_k vs head_dim",
            left: q_or_k.len(),
            right: head_dim,
        });
    }
    rotate_head(q_or_k, head_dim / 2, position, theta_base);
    Ok(())
}

/// Apply NeoX RoPE in place to `n_heads` contiguous heads laid out as
/// `[n_heads * head_dim]` (row-major, one `[head_dim]` slice per head), all
/// at the same sequence `position`. Every head gets the identical rotation.
///
/// # Errors
///
/// [`PrimitiveError::InvalidHeadDim`] if `head_dim` is zero or odd;
/// [`PrimitiveError::LengthMismatch`] if `x.len() != n_heads * head_dim`.
pub fn rope_neox_heads(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    position: u32,
    theta_base: f32,
) -> Result<(), PrimitiveError> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(PrimitiveError::InvalidHeadDim { head_dim });
    }
    let expected = n_heads.checked_mul(head_dim);
    if expected != Some(x.len()) {
        return Err(PrimitiveError::LengthMismatch {
            what: "rope_neox_heads: x vs n_heads * head_dim",
            left: x.len(),
            right: n_heads.saturating_mul(head_dim),
        });
    }
    let half = head_dim / 2;
    for head in x.chunks_exact_mut(head_dim) {
        rotate_head(head, half, position, theta_base);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Rng, assert_close};
    use super::*;

    /// v0 model pin: rope_theta.
    const THETA: f32 = 1e7;

    /// Naive `f64` reference of the NeoX/rotate_half formula.
    fn rope_ref(x: &[f32], position: u32, theta_base: f32) -> Vec<f64> {
        let half = x.len() / 2;
        let dim = x.len() as f64;
        let mut out: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        for i in 0..half {
            let inv_freq = f64::from(theta_base).powf(-2.0 * i as f64 / dim);
            let angle = f64::from(position) * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = out[i];
            let x1 = out[i + half];
            out[i] = x0 * cos - x1 * sin;
            out[i + half] = x0 * sin + x1 * cos;
        }
        out
    }

    #[test]
    fn position_zero_is_identity() {
        let mut rng = Rng::new(0x0);
        for &dim in &[4usize, 128] {
            let original = rng.vec_in(dim, -5.0, 5.0);
            let mut x = original.clone();
            rope_neox(&mut x, dim, 0, THETA).unwrap();
            assert_eq!(x, original, "position 0 must be an exact identity");
        }
    }

    #[test]
    fn known_angles_head_dim_4() {
        // Hand-computed f64 values for head [1, 2, 3, 4], head_dim 4,
        // theta 1e7. Pairs are (0, 2) and (1, 3);
        // inv_freq = [1, 1e7^(-1/2) = 3.1622776601683794e-4].
        let cases: [(u32, [f64; 4]); 2] = [
            (
                1,
                [
                    -1.9841106485555495,
                    1.9987349889570154,
                    2.4623779024123156,
                    4.000632255521494,
                ],
            ),
            (
                3,
                [
                    -1.413352520780047,
                    1.9962043673770753,
                    -2.828857481741469,
                    4.001895566311631,
                ],
            ),
        ];
        for (position, expected) in cases {
            let mut x = [1.0f32, 2.0, 3.0, 4.0];
            rope_neox(&mut x, 4, position, THETA).unwrap();
            for (&got, &want) in x.iter().zip(&expected) {
                assert_close(got, want, 1e-6, 1e-30);
            }
        }
    }

    #[test]
    fn matches_f64_reference_head_dim_128() {
        let mut rng = Rng::new(0xD1);
        for &position in &[1u32, 2, 17, 4096, 100_000] {
            let original = rng.vec_in(128, -3.0, 3.0);
            let reference = rope_ref(&original, position, THETA);
            let mut x = original.clone();
            rope_neox(&mut x, 128, position, THETA).unwrap();
            for (&got, &want) in x.iter().zip(&reference) {
                // f32 rotation error is absolute in the *input* magnitude
                // (products round at ~ulp(3) here), so when x0*cos - x1*sin
                // cancels toward zero a pure output-relative bound is
                // unattainable; the 1e-6 abs floor is ~1e-6 of input scale.
                assert_close(got, want, 1e-6, 1e-6);
            }
        }
    }

    #[test]
    fn preserves_per_pair_magnitude() {
        // RoPE is a rotation in each (i, i + half) plane: the pair magnitude
        // sqrt(x_i^2 + x_{i+half}^2) is invariant.
        let mut rng = Rng::new(0xF00D);
        let dim = 128;
        let half = dim / 2;
        let original = rng.vec_in(dim, -8.0, 8.0);
        let mut x = original.clone();
        rope_neox(&mut x, dim, 90_210, THETA).unwrap();
        for i in 0..half {
            let before = f64::from(original[i]).hypot(f64::from(original[i + half]));
            let after = f64::from(x[i]).hypot(f64::from(x[i + half]));
            assert_close(after as f32, before, 1e-5, 1e-30);
        }
    }

    #[test]
    fn multi_head_matches_per_head() {
        let mut rng = Rng::new(0xCAFE);
        let (n_heads, dim) = (4usize, 128usize);
        let mut flat = rng.vec_in(n_heads * dim, -2.0, 2.0);
        let mut per_head = flat.clone();
        rope_neox_heads(&mut flat, n_heads, dim, 7, THETA).unwrap();
        for head in per_head.chunks_exact_mut(dim) {
            rope_neox(head, dim, 7, THETA).unwrap();
        }
        assert_eq!(flat, per_head);
    }

    #[test]
    fn invalid_head_dim_errors() {
        let mut x = [0.0f32; 3];
        assert_eq!(
            rope_neox(&mut x, 3, 1, THETA),
            Err(PrimitiveError::InvalidHeadDim { head_dim: 3 })
        );
        assert_eq!(
            rope_neox(&mut [], 0, 1, THETA),
            Err(PrimitiveError::InvalidHeadDim { head_dim: 0 })
        );
        assert_eq!(
            rope_neox_heads(&mut x, 1, 3, 1, THETA),
            Err(PrimitiveError::InvalidHeadDim { head_dim: 3 })
        );
    }

    #[test]
    fn length_mismatch_errors() {
        let mut x = [0.0f32; 6];
        assert_eq!(
            rope_neox(&mut x, 4, 1, THETA),
            Err(PrimitiveError::LengthMismatch {
                what: "rope_neox: q_or_k vs head_dim",
                left: 6,
                right: 4,
            })
        );
        assert_eq!(
            rope_neox_heads(&mut x, 2, 4, 1, THETA),
            Err(PrimitiveError::LengthMismatch {
                what: "rope_neox_heads: x vs n_heads * head_dim",
                left: 6,
                right: 8,
            })
        );
    }
}
