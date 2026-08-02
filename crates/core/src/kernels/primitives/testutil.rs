//! Deterministic test helpers: a tiny xorshift64* PRNG (no dev-deps) and a
//! mixed absolute/relative tolerance assertion.

/// xorshift64* PRNG. Deterministic, seedable, good enough for test vectors.
pub struct Rng(u64);

impl Rng {
    /// Create a generator from a nonzero-coerced seed.
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform `f32` in `[lo, hi)`, from the top 24 bits of the next draw.
    pub fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * unit
    }

    /// A vector of `n` uniform draws from `[lo, hi)`.
    pub fn vec_in(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32_in(lo, hi)).collect()
    }
}

/// Assert `actual` is within `abs_tol` absolute or `rel_tol` relative of the
/// `f64` reference `expected`.
///
/// The absolute floor exists for outputs that underflow into `f32`
/// denormals, where relative precision is physically unavailable.
pub fn assert_close(actual: f32, expected: f64, rel_tol: f64, abs_tol: f64) {
    let a = f64::from(actual);
    let diff = (a - expected).abs();
    if diff <= abs_tol {
        return;
    }
    let rel = diff / expected.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel <= rel_tol,
        "actual {actual:e} vs expected {expected:e}: rel err {rel:.3e} > {rel_tol:.1e} \
         (abs diff {diff:.3e} > {abs_tol:.1e})"
    );
}
