//! Single-threaded quantized GEMV: one packed weight matrix times one
//! quantized activation row.
//!
//! The forward pass is token-at-a-time, so every projection is a GEMV over
//! a row-major packed weight matrix (`out_dim` rows of `in_dim` weights,
//! each row a whole number of quantization blocks). Each output element is
//! one `vec_dot` of a weight row against the pre-quantized activation row,
//! dispatched through [`super::quants::avx2`] (AVX2+FMA when the CPU has
//! it, scalar otherwise).
//!
//! Two entry points, one per activation type: [`gemv_q8_k`] for the k-quant
//! weight formats (Q4_K / Q5_K / Q6_K against Q8_K activations) and
//! [`gemv_q8_0`] for Q8_0 weights against Q8_0 activations. That covers
//! every audited v0 projection (in-dims 2048 / 768 / 4096, out-dims from
//! 512 up to the 151936-row lm_head).
//!
//! Sizes are validated once up front; the per-row kernels re-check their
//! own row length, which after this validation cannot fail (the checks are
//! a few integer compares per row, noise next to the dot itself).
//! Single-threaded by design — parallelism lands with the phase-4 forward
//! pass, not here.

use super::KernelError;
use super::quants::{BlockQ8_0, BlockQ8K, QuantFormat, avx2};

/// Validate the shared GEMV geometry: `weight` is `out_dim` rows of
/// `row_bytes`, `out` holds `out_dim` floats, and the activation row covers
/// `in_dim` weights. Returns `row_bytes`.
fn validate(
    what: &'static str,
    format: QuantFormat,
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    activation_blocks: usize,
    out_len: usize,
) -> Result<usize, KernelError> {
    let row_bytes = format.row_bytes(in_dim)?;
    match row_bytes.checked_mul(out_dim) {
        Some(expected) if expected == weight.len() => {}
        _ => {
            return Err(KernelError::LengthMismatch {
                what,
                left: weight.len(),
                right: row_bytes.saturating_mul(out_dim),
            });
        }
    }
    if out_len != out_dim {
        return Err(KernelError::LengthMismatch {
            what,
            left: out_len,
            right: out_dim,
        });
    }
    let weight_blocks = in_dim / format.block_weights();
    if weight_blocks != activation_blocks {
        return Err(KernelError::BlockCountMismatch {
            weight_blocks,
            activation_blocks,
        });
    }
    Ok(row_bytes)
}

/// GEMV of a k-quant weight matrix (Q4_K, Q5_K, or Q6_K) against a Q8_K
/// activation row: `out[r] = weight_row_r . acts` for `r` in `0..out_dim`.
///
/// `weight` is the row-major packed matrix (`out_dim * row_bytes(in_dim)`
/// bytes, rows contiguous); `acts` must hold `in_dim / 256` blocks; `out`
/// must hold `out_dim` floats.
///
/// # Errors
///
/// [`KernelError::UnsupportedFormat`] for non-k-quant formats;
/// [`KernelError::IndivisibleRow`], [`KernelError::LengthMismatch`], or
/// [`KernelError::BlockCountMismatch`] when the sizes disagree.
pub fn gemv_q8_k(
    format: QuantFormat,
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    acts: &[BlockQ8K],
    out: &mut [f32],
) -> Result<(), KernelError> {
    let dot = match format {
        QuantFormat::Q4_K => avx2::vec_dot_q4_k_q8_k,
        QuantFormat::Q5_K => avx2::vec_dot_q5_k_q8_k,
        QuantFormat::Q6_K => avx2::vec_dot_q6_k_q8_k,
        QuantFormat::Q8_0 | QuantFormat::Q8_K => {
            return Err(KernelError::UnsupportedFormat {
                what: "gemv_q8_k",
                format,
            });
        }
    };
    let row_bytes = validate(
        "gemv_q8_k: weight bytes vs rows / out vs out_dim",
        format,
        weight,
        in_dim,
        out_dim,
        acts.len(),
        out.len(),
    )?;
    for (r, o) in out.iter_mut().enumerate() {
        *o = dot(&weight[r * row_bytes..(r + 1) * row_bytes], acts, false)?;
    }
    Ok(())
}

/// GEMV of a Q8_0 weight matrix against a Q8_0 activation row:
/// `out[r] = weight_row_r . acts` for `r` in `0..out_dim`.
///
/// `weight` is the row-major packed matrix (`out_dim * row_bytes(in_dim)`
/// bytes); `acts` must hold `in_dim / 32` blocks; `out` must hold `out_dim`
/// floats.
///
/// # Errors
///
/// [`KernelError::IndivisibleRow`], [`KernelError::LengthMismatch`], or
/// [`KernelError::BlockCountMismatch`] when the sizes disagree.
pub fn gemv_q8_0(
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    acts: &[BlockQ8_0],
    out: &mut [f32],
) -> Result<(), KernelError> {
    let row_bytes = validate(
        "gemv_q8_0: weight bytes vs rows / out vs out_dim",
        QuantFormat::Q8_0,
        weight,
        in_dim,
        out_dim,
        acts.len(),
        out.len(),
    )?;
    for (r, o) in out.iter_mut().enumerate() {
        *o = avx2::vec_dot_q8_0_q8_0(&weight[r * row_bytes..(r + 1) * row_bytes], acts, false)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::quants::{quantize_row_q8_0, quantize_row_q8_k};
    use super::*;

    /// Deterministic LCG (same constants as the quants test support, which
    /// is not visible from this module).
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }

        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }

        /// Random block bytes with small, sane f16 scales planted at
        /// `scale_offs` (keeps the float side finite and non-denormal).
        fn block_bytes(&mut self, format: QuantFormat, scale_offs: &[usize]) -> Vec<u8> {
            let mut b: Vec<u8> = (0..format.block_bytes())
                .map(|_| (self.next_u64() >> 32) as u8)
                .collect();
            for &off in scale_offs {
                let d = super::super::quants::f32_to_f16(0.01 + 0.02 * (self.next_f32() + 1.0));
                b[off..off + 2].copy_from_slice(&d.to_le_bytes());
            }
            b
        }
    }

    /// A row-major packed matrix of structurally valid random blocks.
    fn synth_matrix(format: QuantFormat, in_dim: usize, out_dim: usize, seed: u64) -> Vec<u8> {
        let scale_offs: &[usize] = match format {
            QuantFormat::Q4_K | QuantFormat::Q5_K => &[0, 2], // d, dmin
            QuantFormat::Q6_K => &[208],
            _ => &[0],
        };
        let mut rng = Lcg(seed);
        let blocks_per_row = in_dim / format.block_weights();
        let mut w = Vec::with_capacity(out_dim * blocks_per_row * format.block_bytes());
        for _ in 0..out_dim * blocks_per_row {
            w.extend_from_slice(&rng.block_bytes(format, scale_offs));
        }
        w
    }

    fn q8_k_acts(n: usize, seed: u64) -> Vec<BlockQ8K> {
        let mut rng = Lcg(seed);
        let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 3.0).collect();
        let mut acts = vec![BlockQ8K::default(); n / 256];
        quantize_row_q8_k(&x, &mut acts).unwrap();
        acts
    }

    #[test]
    fn gemv_q8_k_matches_per_row_dots() {
        // Audited-shape slices (out_dim trimmed to keep the test fast; the
        // row walk is identical for any out_dim).
        for (format, in_dim, out_dim) in [
            (QuantFormat::Q4_K, 2048, 64),
            (QuantFormat::Q5_K, 4096, 32),
            (QuantFormat::Q6_K, 768, 96),
        ] {
            let w = synth_matrix(format, in_dim, out_dim, 0x6E0 ^ in_dim as u64);
            let acts = q8_k_acts(in_dim, 0xAC ^ in_dim as u64);
            let mut out = vec![0f32; out_dim];
            gemv_q8_k(format, &w, in_dim, out_dim, &acts, &mut out).unwrap();
            let dot = match format {
                QuantFormat::Q4_K => avx2::vec_dot_q4_k_q8_k,
                QuantFormat::Q5_K => avx2::vec_dot_q5_k_q8_k,
                QuantFormat::Q6_K => avx2::vec_dot_q6_k_q8_k,
                _ => unreachable!(),
            };
            let row_bytes = format.row_bytes(in_dim).unwrap();
            for (r, &o) in out.iter().enumerate() {
                let want = dot(&w[r * row_bytes..(r + 1) * row_bytes], &acts, false).unwrap();
                assert_eq!(o.to_bits(), want.to_bits(), "{format:?} row {r}");
            }
        }
    }

    #[test]
    fn gemv_q8_0_matches_per_row_dots() {
        let (in_dim, out_dim) = (2048, 64);
        let w = synth_matrix(QuantFormat::Q8_0, in_dim, out_dim, 0x8_0);
        let mut rng = Lcg(0xAC7);
        let x: Vec<f32> = (0..in_dim).map(|_| rng.next_f32() * 3.0).collect();
        let mut acts = vec![BlockQ8_0::default(); in_dim / 32];
        quantize_row_q8_0(&x, &mut acts).unwrap();
        let mut out = vec![0f32; out_dim];
        gemv_q8_0(&w, in_dim, out_dim, &acts, &mut out).unwrap();
        let row_bytes = QuantFormat::Q8_0.row_bytes(in_dim).unwrap();
        for (r, &o) in out.iter().enumerate() {
            let want =
                avx2::vec_dot_q8_0_q8_0(&w[r * row_bytes..(r + 1) * row_bytes], &acts, false)
                    .unwrap();
            assert_eq!(o.to_bits(), want.to_bits(), "row {r}");
        }
    }

    #[test]
    fn rejects_wrong_formats() {
        let acts = vec![BlockQ8K::default()];
        let mut out = [0f32; 1];
        for format in [QuantFormat::Q8_0, QuantFormat::Q8_K] {
            assert_eq!(
                gemv_q8_k(format, &[], 256, 1, &acts, &mut out).unwrap_err(),
                KernelError::UnsupportedFormat {
                    what: "gemv_q8_k",
                    format,
                },
            );
        }
    }

    #[test]
    fn rejects_bad_sizes() {
        let acts = vec![BlockQ8K::default(); 8]; // 2048 weights
        let mut out = vec![0f32; 4];
        let row = QuantFormat::Q4_K.row_bytes(2048).unwrap(); // 1152

        // in_dim not a block multiple.
        assert_eq!(
            gemv_q8_k(QuantFormat::Q4_K, &[0; 1152], 2000, 1, &acts, &mut out[..1]).unwrap_err(),
            KernelError::IndivisibleRow {
                format: QuantFormat::Q4_K,
                in_dim: 2000,
                block_weights: 256,
            },
        );
        // Weight buffer one row short.
        assert_eq!(
            gemv_q8_k(
                QuantFormat::Q4_K,
                &vec![0; row * 3],
                2048,
                4,
                &acts,
                &mut out
            )
            .unwrap_err(),
            KernelError::LengthMismatch {
                what: "gemv_q8_k: weight bytes vs rows / out vs out_dim",
                left: row * 3,
                right: row * 4,
            },
        );
        // Output buffer wrong length.
        assert_eq!(
            gemv_q8_k(
                QuantFormat::Q4_K,
                &vec![0; row * 4],
                2048,
                4,
                &acts,
                &mut out[..2]
            )
            .unwrap_err(),
            KernelError::LengthMismatch {
                what: "gemv_q8_k: weight bytes vs rows / out vs out_dim",
                left: 2,
                right: 4,
            },
        );
        // Activation row too short.
        assert_eq!(
            gemv_q8_k(
                QuantFormat::Q4_K,
                &vec![0; row * 4],
                2048,
                4,
                &acts[..4],
                &mut out
            )
            .unwrap_err(),
            KernelError::BlockCountMismatch {
                weight_blocks: 8,
                activation_blocks: 4,
            },
        );
        // Q8_0 side: truncated weights.
        let acts0 = vec![BlockQ8_0::default(); 64];
        assert_eq!(
            gemv_q8_0(&[0; 2176], 2048, 2, &acts0, &mut out[..2]).unwrap_err(),
            KernelError::LengthMismatch {
                what: "gemv_q8_0: weight bytes vs rows / out vs out_dim",
                left: 2176,
                right: 4352,
            },
        );
    }

    #[test]
    fn zero_out_dim_is_a_no_op() {
        let acts = q8_k_acts(256, 1);
        let mut out: [f32; 0] = [];
        gemv_q8_k(QuantFormat::Q4_K, &[], 256, 0, &acts, &mut out).unwrap();
    }
}
