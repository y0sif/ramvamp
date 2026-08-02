//! SiLU activation and the SwiGLU gate/up combine.
//!
//! v0 model pin: the MoE FFN is SwiGLU with SiLU
//! (`silu(gate_proj(x)) * up_proj(x)`, then `down_proj`).

use super::PrimitiveError;

/// SiLU (a.k.a. swish) of one value: `v * sigmoid(v) = v / (1 + e^-v)`.
///
/// Saturates cleanly at the `f32` extremes: `e^-v` overflowing to `+inf`
/// for very negative `v` yields `-0.0`, and underflowing to `0` for very
/// positive `v` yields `v`.
#[inline]
fn silu_one(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// In-place SiLU: `x_i <- x_i / (1 + e^(-x_i))`.
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu_one(*v);
    }
}

/// SwiGLU FFN combine, in place into `gate`: `gate_i <- silu(gate_i) * up_i`.
///
/// # Errors
///
/// [`PrimitiveError::LengthMismatch`] if `gate` and `up` differ in length.
pub fn swiglu_combine(gate: &mut [f32], up: &[f32]) -> Result<(), PrimitiveError> {
    if gate.len() != up.len() {
        return Err(PrimitiveError::LengthMismatch {
            what: "swiglu_combine: gate vs up",
            left: gate.len(),
            right: up.len(),
        });
    }
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = silu_one(*g) * u;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Rng, assert_close};
    use super::*;

    /// `f64` sigmoid: `1 / (1 + e^-v)`.
    fn sigmoid_ref(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    /// `f64` SiLU reference via the sigmoid formula: `v * sigmoid(v)`.
    fn silu_ref(v: f32) -> f64 {
        let v = f64::from(v);
        v * sigmoid_ref(v)
    }

    /// Extremes: +-inf-adjacent magnitudes, the f32 `exp` overflow boundary
    /// (~88.7), denormals, and zero.
    const EDGE_CASES: [f32; 11] = [
        -1.0e30, -100.0, -88.0, -10.0, -1.0e-40, 0.0, 1.0e-40, 10.0, 88.0, 100.0, 1.0e30,
    ];

    #[test]
    fn silu_matches_f64_sigmoid_formula() {
        let mut rng = Rng::new(0x51D);
        let mut x = rng.vec_in(1024, -20.0, 20.0);
        x.extend_from_slice(&EDGE_CASES);
        let mut got = x.clone();
        silu(&mut got);
        for (&g, &v) in got.iter().zip(&x) {
            assert_close(g, silu_ref(v), 1e-6, 1e-30);
        }
    }

    #[test]
    fn swiglu_matches_f64_sigmoid_formula() {
        let mut rng = Rng::new(0x5716);
        let mut gate = rng.vec_in(768, -12.0, 12.0);
        let up = rng.vec_in(768, -3.0, 3.0);
        let original_gate = gate.clone();
        swiglu_combine(&mut gate, &up).unwrap();
        for ((&got, &g0), &u) in gate.iter().zip(&original_gate).zip(&up) {
            let want = silu_ref(g0) * f64::from(u);
            assert_close(got, want, 1e-6, 1e-30);
        }
    }

    #[test]
    fn swiglu_edge_cases() {
        let mut gate = EDGE_CASES.to_vec();
        let up = vec![2.0f32; gate.len()];
        let original_gate = gate.clone();
        swiglu_combine(&mut gate, &up).unwrap();
        for (&got, &g0) in gate.iter().zip(&original_gate) {
            assert_close(got, silu_ref(g0) * 2.0, 1e-6, 1e-30);
        }
    }

    #[test]
    fn swiglu_length_mismatch_errors() {
        let mut gate = [1.0f32; 4];
        assert_eq!(
            swiglu_combine(&mut gate, &[1.0; 3]),
            Err(PrimitiveError::LengthMismatch {
                what: "swiglu_combine: gate vs up",
                left: 4,
                right: 3,
            })
        );
    }
}
