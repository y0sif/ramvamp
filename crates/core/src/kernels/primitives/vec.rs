//! Small vector helpers: residual add and scalar scale.

use super::PrimitiveError;

/// Elementwise in-place add: `a_i <- a_i + b_i` (residual connections).
///
/// # Errors
///
/// [`PrimitiveError::LengthMismatch`] if `a` and `b` differ in length.
pub fn vec_add(a: &mut [f32], b: &[f32]) -> Result<(), PrimitiveError> {
    if a.len() != b.len() {
        return Err(PrimitiveError::LengthMismatch {
            what: "vec_add: a vs b",
            left: a.len(),
            right: b.len(),
        });
    }
    for (x, &y) in a.iter_mut().zip(b) {
        *x += y;
    }
    Ok(())
}

/// In-place scalar scale: `x_i <- x_i * s` (e.g. router weight application).
pub fn vec_scale(x: &mut [f32], s: f32) {
    for v in x.iter_mut() {
        *v *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::Rng;
    use super::*;

    #[test]
    fn vec_add_adds_elementwise() {
        let mut rng = Rng::new(0xADD);
        let a0 = rng.vec_in(2048, -100.0, 100.0);
        let b = rng.vec_in(2048, -100.0, 100.0);
        let mut a = a0.clone();
        vec_add(&mut a, &b).unwrap();
        for ((&got, &x), &y) in a.iter().zip(&a0).zip(&b) {
            assert_eq!(got, x + y);
        }
        // Empty slices are a valid no-op.
        vec_add(&mut [], &[]).unwrap();
    }

    #[test]
    fn vec_add_length_mismatch_errors() {
        let mut a = [1.0f32; 4];
        assert_eq!(
            vec_add(&mut a, &[1.0; 5]),
            Err(PrimitiveError::LengthMismatch {
                what: "vec_add: a vs b",
                left: 4,
                right: 5,
            })
        );
    }

    #[test]
    fn vec_scale_scales_elementwise() {
        let mut rng = Rng::new(0x5CA1E);
        let x0 = rng.vec_in(1024, -50.0, 50.0);
        let mut x = x0.clone();
        vec_scale(&mut x, 0.375);
        for (&got, &v) in x.iter().zip(&x0) {
            assert_eq!(got, v * 0.375);
        }
        vec_scale(&mut x, 0.0);
        assert!(x.iter().all(|&v| v == 0.0));
        vec_scale(&mut [], 2.0);
    }
}
