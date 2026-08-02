//! Numerically stable softmax.

use super::PrimitiveError;

/// In-place softmax: `x_i <- exp(x_i - max(x)) / sum_j exp(x_j - max(x))`.
///
/// Max-subtraction keeps the exponentials in range for arbitrarily large
/// inputs. The subtraction, `exp`, and the normalizer all run in `f64`
/// (ggml accumulates its normalizer in `ggml_float = double`; we also take
/// the exponent argument in `f64` because an `f32` `v - max` alone costs up
/// to ~1e-6 relative in the exponential), then every element is scaled by
/// the normalizer's `f32` reciprocal. Non-finite inputs propagate: a `NaN`,
/// or an all-`-inf` slice, yields `NaN` outputs rather than a panic.
///
/// # Errors
///
/// [`PrimitiveError::Empty`] if `x` is empty.
pub fn softmax(x: &mut [f32]) -> Result<(), PrimitiveError> {
    if x.is_empty() {
        return Err(PrimitiveError::Empty { what: "softmax" });
    }
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let max = f64::from(max);
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        let e = (f64::from(*v) - max).exp();
        *v = e as f32;
        sum += e;
    }
    let inv = sum.recip() as f32;
    for v in x.iter_mut() {
        *v *= inv;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Rng, assert_close};
    use super::*;

    /// Naive `f64` reference softmax (max-subtracted for the same domain).
    fn softmax_ref(x: &[f32]) -> Vec<f64> {
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = x
            .iter()
            .map(|&v| (f64::from(v) - f64::from(max)).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    }

    fn check_against_ref(x: &[f32]) {
        let reference = softmax_ref(x);
        let mut got = x.to_vec();
        softmax(&mut got).unwrap();
        for (&g, &want) in got.iter().zip(&reference) {
            // abs floor 1e-10: probabilities that underflow toward f32
            // denormals cannot hold 1e-6 relative precision.
            assert_close(g, want, 1e-6, 1e-10);
        }
        let total: f64 = got.iter().map(|&v| f64::from(v)).sum();
        assert_close(total as f32, 1.0, 1e-5, 0.0);
    }

    #[test]
    fn matches_f64_reference_random() {
        let mut rng = Rng::new(0x50F7);
        for &n in &[2usize, 20, 128, 2048] {
            let x = rng.vec_in(n, -10.0, 10.0);
            check_against_ref(&x);
        }
    }

    #[test]
    fn matches_f64_reference_large_magnitudes() {
        // Would overflow exp() without max subtraction.
        check_against_ref(&[3.0e4f32, 3.0001e4, 2.9999e4]);
        check_against_ref(&[-3.0e38f32, 3.0e38]);
        let mut rng = Rng::new(0xB16);
        let x = rng.vec_in(512, -1.0e30, 1.0e30);
        check_against_ref(&x);
    }

    #[test]
    fn matches_f64_reference_all_equal() {
        for &n in &[1usize, 3, 1000] {
            let x = vec![42.5f32; n];
            let mut got = x.clone();
            softmax(&mut got).unwrap();
            for &g in &got {
                assert_close(g, 1.0 / n as f64, 1e-6, 0.0);
            }
        }
    }

    #[test]
    fn single_element_is_one() {
        let mut x = [-1234.5f32];
        softmax(&mut x).unwrap();
        assert_eq!(x, [1.0f32]);
    }

    #[test]
    fn matches_f64_reference_denormals() {
        // Sub-normal inputs: differences are ~0, so this approaches uniform.
        check_against_ref(&[1.0e-40f32, -2.0e-40, 1.4e-45, 0.0]);
    }

    #[test]
    fn empty_input_errors() {
        let mut x: [f32; 0] = [];
        assert_eq!(
            softmax(&mut x),
            Err(PrimitiveError::Empty { what: "softmax" })
        );
    }
}
