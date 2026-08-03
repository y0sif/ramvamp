//! Quantized GEMV: one packed weight matrix times one quantized activation
//! row, over the whole matrix or over a half-open range of output rows.
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
//! # Row-range parallelism and bit-exactness
//!
//! [`gemv_q8_k_rows`] and [`gemv_q8_0_rows`] compute a contiguous slice of
//! output rows instead of all of them. That is the *only* partitioning this
//! module offers, and deliberately so:
//!
//! - Output rows are independent. `out[r]` is one `vec_dot` of weight row
//!   `r` against the shared, immutable activation row; it reads no other
//!   row and no other output. So for any partition of `0..out_dim` into
//!   row ranges, the concatenated results are **bit-identical** to the
//!   whole-matrix call — not "within tolerance", identical, because each
//!   element is produced by literally the same kernel call on the same
//!   bytes. Phase 5's acceptance gate (phase-5 logits bit-identical to
//!   phase 4) rests on this, so the whole-matrix entry points are thin
//!   wrappers over the row-range ones: one code path, no drift.
//! - Splitting *within* a row would not be bit-exact. The AVX2 and scalar
//!   dot kernels fix their float accumulation order per super-block, and
//!   float addition is not associative, so summing partial dots would
//!   change the low bits. No such split exists here and none may be added.
//!
//! `out` is the destination for `rows` alone, i.e. `out.len() == rows.len()`,
//! not the full `out_dim` buffer. Callers hand each worker a disjoint
//! `&mut [f32]` carved out of one output buffer with `split_at_mut` /
//! `chunks_mut`, which the borrow checker already proves non-overlapping;
//! taking the full buffer per worker instead would force either aliasing
//! `unsafe` or a per-thread copy. Geometry validation still describes the
//! whole matrix (`weight` is all `out_dim` rows), and `rows` is validated
//! against `out_dim`.
//!
//! Sizes are validated once up front; the per-row kernels re-check their
//! own row length, which after this validation cannot fail (the checks are
//! a few integer compares per row, noise next to the dot itself).

use super::KernelError;
use super::quants::{BlockQ8_0, BlockQ8K, QuantFormat, avx2};
use std::ops::Range;
use std::sync::OnceLock;

/// Diagnostic escape hatch: `RAMVAMP_FORCE_SCALAR` (any value) pins every
/// GEMV to the scalar reference kernels, so AVX2-vs-scalar output deltas
/// can be measured end-to-end (the same A/B the benches do per-kernel).
/// Read once; not a supported production knob.
fn force_scalar() -> bool {
    static FORCE: OnceLock<bool> = OnceLock::new();
    *FORCE.get_or_init(|| std::env::var_os("RAMVAMP_FORCE_SCALAR").is_some())
}

/// `what` for an inverted row range (`start > end`).
const ROW_RANGE_ORDER: &str = "gemv rows: row range start vs end";

/// `what` for a row range that runs past the last row of the matrix.
const ROW_RANGE_BOUND: &str = "gemv rows: row range end vs out_dim";

/// The dispatching k-quant row-dot kernels all share this signature
/// (`weight_row`, `acts`, `force_scalar`).
type DotQ8K = fn(&[u8], &[BlockQ8K], bool) -> Result<f32, KernelError>;

/// A whole packed weight matrix: `out_dim` rows of `in_dim` `format`
/// weights, rows contiguous. Bundled so the shared implementations pass one
/// value instead of four positional arguments.
#[derive(Debug, Clone, Copy)]
struct Matrix<'a> {
    /// Weight quantization format.
    format: QuantFormat,
    /// Row-major packed bytes for all `out_dim` rows.
    bytes: &'a [u8],
    /// Weights per row.
    in_dim: usize,
    /// Rows in the whole matrix (not in the requested range).
    out_dim: usize,
}

/// Validate GEMV geometry: `weight` is `out_dim` whole rows, `rows` is a
/// well-formed sub-range of `0..out_dim`, `out` holds exactly `rows.len()`
/// floats, and the activation row covers `in_dim` weights. Returns
/// `row_bytes`.
///
/// The weight slice always describes the *whole* matrix even when only a
/// range is computed, so the full-matrix checks are unconditional and the
/// range is checked against `out_dim` on top of them.
fn validate(
    what: &'static str,
    m: Matrix<'_>,
    activation_blocks: usize,
    rows: &Range<usize>,
    out_len: usize,
) -> Result<usize, KernelError> {
    let row_bytes = m.format.row_bytes(m.in_dim)?;
    match row_bytes.checked_mul(m.out_dim) {
        Some(expected) if expected == m.bytes.len() => {}
        _ => {
            return Err(KernelError::LengthMismatch {
                what,
                left: m.bytes.len(),
                right: row_bytes.saturating_mul(m.out_dim),
            });
        }
    }
    if rows.start > rows.end {
        return Err(KernelError::LengthMismatch {
            what: ROW_RANGE_ORDER,
            left: rows.start,
            right: rows.end,
        });
    }
    if rows.end > m.out_dim {
        return Err(KernelError::LengthMismatch {
            what: ROW_RANGE_BOUND,
            left: rows.end,
            right: m.out_dim,
        });
    }
    // Safe: `start <= end` was just established.
    let wanted = rows.end - rows.start;
    if out_len != wanted {
        return Err(KernelError::LengthMismatch {
            what,
            left: out_len,
            right: wanted,
        });
    }
    let weight_blocks = m.in_dim / m.format.block_weights();
    if weight_blocks != activation_blocks {
        return Err(KernelError::BlockCountMismatch {
            weight_blocks,
            activation_blocks,
        });
    }
    Ok(row_bytes)
}

/// Resolve the k-quant row-dot kernel for `format`, rejecting the formats
/// that never pair with Q8_K activations.
fn k_quant_dot(what: &'static str, format: QuantFormat) -> Result<DotQ8K, KernelError> {
    match format {
        QuantFormat::Q4_K => Ok(avx2::vec_dot_q4_k_q8_k),
        QuantFormat::Q5_K => Ok(avx2::vec_dot_q5_k_q8_k),
        QuantFormat::Q6_K => Ok(avx2::vec_dot_q6_k_q8_k),
        QuantFormat::Q8_0 | QuantFormat::Q8_K => {
            Err(KernelError::UnsupportedFormat { what, format })
        }
    }
}

/// The one k-quant GEMV loop: `out[i] = row(rows.start + i) . acts`.
///
/// `scalar` is threaded rather than read from the environment so tests and
/// benches can drive both kernel paths in one process.
fn gemv_q8_k_impl(
    dot: DotQ8K,
    what: &'static str,
    m: Matrix<'_>,
    acts: &[BlockQ8K],
    rows: Range<usize>,
    out: &mut [f32],
    scalar: bool,
) -> Result<(), KernelError> {
    let row_bytes = validate(what, m, acts.len(), &rows, out.len())?;
    for (i, o) in out.iter_mut().enumerate() {
        // `r < out_dim` and `out_dim * row_bytes == bytes.len()`, so neither
        // the multiply nor the slice can overflow or go out of bounds.
        let r = rows.start + i;
        *o = dot(&m.bytes[r * row_bytes..(r + 1) * row_bytes], acts, scalar)?;
    }
    Ok(())
}

/// The one Q8_0 GEMV loop; see [`gemv_q8_k_impl`].
fn gemv_q8_0_impl(
    what: &'static str,
    m: Matrix<'_>,
    acts: &[BlockQ8_0],
    rows: Range<usize>,
    out: &mut [f32],
    scalar: bool,
) -> Result<(), KernelError> {
    let row_bytes = validate(what, m, acts.len(), &rows, out.len())?;
    for (i, o) in out.iter_mut().enumerate() {
        let r = rows.start + i;
        *o = avx2::vec_dot_q8_0_q8_0(&m.bytes[r * row_bytes..(r + 1) * row_bytes], acts, scalar)?;
    }
    Ok(())
}

/// GEMV of a k-quant weight matrix (Q4_K, Q5_K, or Q6_K) against a Q8_K
/// activation row: `out[r] = weight_row_r . acts` for `r` in `0..out_dim`.
///
/// `weight` is the row-major packed matrix (`out_dim * row_bytes(in_dim)`
/// bytes, rows contiguous); `acts` must hold `in_dim / 256` blocks; `out`
/// must hold `out_dim` floats.
///
/// Exactly [`gemv_q8_k_rows`] over `0..out_dim`, so the whole-matrix and
/// partitioned results are bit-identical by construction.
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
    let dot = k_quant_dot("gemv_q8_k", format)?;
    gemv_q8_k_impl(
        dot,
        "gemv_q8_k: weight bytes vs rows / out vs out_dim",
        Matrix {
            format,
            bytes: weight,
            in_dim,
            out_dim,
        },
        acts,
        0..out_dim,
        out,
        force_scalar(),
    )
}

/// [`gemv_q8_k`] restricted to the output rows in `rows`:
/// `out[i] = weight_row_(rows.start + i) . acts`.
///
/// `weight`, `in_dim`, and `out_dim` still describe the **whole** matrix;
/// `out` holds exactly `rows.len()` floats and receives only those rows.
/// Concatenating the results of any partition of `0..out_dim` reproduces
/// [`gemv_q8_k`] bit for bit (see the module docs), which is what makes the
/// phase-5 compute pool safe to fan out.
///
/// # Errors
///
/// As [`gemv_q8_k`], plus [`KernelError::LengthMismatch`] when `rows` is
/// inverted, runs past `out_dim`, or disagrees with `out.len()`. Never
/// panics on an out-of-bounds or inverted range.
pub fn gemv_q8_k_rows(
    format: QuantFormat,
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    acts: &[BlockQ8K],
    rows: Range<usize>,
    out: &mut [f32],
) -> Result<(), KernelError> {
    let dot = k_quant_dot("gemv_q8_k_rows", format)?;
    gemv_q8_k_impl(
        dot,
        "gemv_q8_k_rows: weight bytes vs rows / out vs range",
        Matrix {
            format,
            bytes: weight,
            in_dim,
            out_dim,
        },
        acts,
        rows,
        out,
        force_scalar(),
    )
}

/// GEMV of a Q8_0 weight matrix against a Q8_0 activation row:
/// `out[r] = weight_row_r . acts` for `r` in `0..out_dim`.
///
/// `weight` is the row-major packed matrix (`out_dim * row_bytes(in_dim)`
/// bytes); `acts` must hold `in_dim / 32` blocks; `out` must hold `out_dim`
/// floats.
///
/// Exactly [`gemv_q8_0_rows`] over `0..out_dim`.
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
    gemv_q8_0_impl(
        "gemv_q8_0: weight bytes vs rows / out vs out_dim",
        Matrix {
            format: QuantFormat::Q8_0,
            bytes: weight,
            in_dim,
            out_dim,
        },
        acts,
        0..out_dim,
        out,
        force_scalar(),
    )
}

/// [`gemv_q8_0`] restricted to the output rows in `rows`:
/// `out[i] = weight_row_(rows.start + i) . acts`.
///
/// `weight`, `in_dim`, and `out_dim` still describe the **whole** matrix;
/// `out` holds exactly `rows.len()` floats. Bit-identical to [`gemv_q8_0`]
/// when the ranges of a partition are concatenated (see the module docs).
///
/// # Errors
///
/// As [`gemv_q8_0`], plus [`KernelError::LengthMismatch`] when `rows` is
/// inverted, runs past `out_dim`, or disagrees with `out.len()`. Never
/// panics on an out-of-bounds or inverted range.
pub fn gemv_q8_0_rows(
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    acts: &[BlockQ8_0],
    rows: Range<usize>,
    out: &mut [f32],
) -> Result<(), KernelError> {
    gemv_q8_0_impl(
        "gemv_q8_0_rows: weight bytes vs rows / out vs range",
        Matrix {
            format: QuantFormat::Q8_0,
            bytes: weight,
            in_dim,
            out_dim,
        },
        acts,
        rows,
        out,
        force_scalar(),
    )
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

    fn q8_0_acts(n: usize, seed: u64) -> Vec<BlockQ8_0> {
        let mut rng = Lcg(seed);
        let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 3.0).collect();
        let mut acts = vec![BlockQ8_0::default(); n / 32];
        quantize_row_q8_0(&x, &mut acts).unwrap();
        acts
    }

    /// A synthesized matrix plus its geometry. Exists so the bit-identity
    /// helpers can reach the shared internals and drive both kernel paths
    /// through the explicit `scalar` flag (`RAMVAMP_FORCE_SCALAR` is read
    /// once per process, so it cannot A/B anything from inside a test).
    struct Fixture {
        /// Human-readable projection name, for assertion messages.
        name: &'static str,
        format: QuantFormat,
        in_dim: usize,
        out_dim: usize,
        weight: Vec<u8>,
    }

    impl Fixture {
        fn new(name: &'static str, format: QuantFormat, in_dim: usize, out_dim: usize) -> Self {
            let seed = 0x6E0 ^ (in_dim as u64) << 20 ^ (out_dim as u64) << 4 ^ format as u64;
            Self {
                name,
                format,
                in_dim,
                out_dim,
                weight: synth_matrix(format, in_dim, out_dim, seed),
            }
        }

        fn matrix(&self) -> Matrix<'_> {
            Matrix {
                format: self.format,
                bytes: &self.weight,
                in_dim: self.in_dim,
                out_dim: self.out_dim,
            }
        }
    }

    /// The real v0 projection shapes and formats (`docs/architecture.md`:
    /// hidden 2048, attention inner 4096, GQA 32:4 so kv_dim 512,
    /// moe_intermediate 768). `lm_head` is Q6_K 2048 -> 151936; its out_dim
    /// is trimmed to 1024 here because the row walk is independent of
    /// out_dim and a 255 MB synthetic matrix is not a unit test.
    fn k_quant_fixtures() -> Vec<Fixture> {
        vec![
            Fixture::new("expert gate (Q4_K)", QuantFormat::Q4_K, 2048, 768),
            Fixture::new("expert up (Q4_K)", QuantFormat::Q4_K, 2048, 768),
            Fixture::new("expert down (Q6_K layers)", QuantFormat::Q6_K, 768, 2048),
            Fixture::new("expert down (Q4_K layers)", QuantFormat::Q4_K, 768, 2048),
            Fixture::new("attn_q (Q4_K)", QuantFormat::Q4_K, 2048, 4096),
            Fixture::new("attn_v (Q6_K layers)", QuantFormat::Q6_K, 2048, 512),
            Fixture::new("attn_v (Q4_K layers)", QuantFormat::Q4_K, 2048, 512),
            Fixture::new("attn_output (Q5_K)", QuantFormat::Q5_K, 4096, 2048),
            Fixture::new(
                "lm_head (Q6_K, out_dim trimmed)",
                QuantFormat::Q6_K,
                2048,
                1024,
            ),
        ]
    }

    /// Contiguous ranges of `chunk` rows covering `0..out_dim`; the last is
    /// short whenever `chunk` does not divide `out_dim`.
    fn chunked(out_dim: usize, chunk: usize) -> Vec<Range<usize>> {
        let chunk = chunk.max(1);
        let mut v = Vec::new();
        let mut s = 0;
        while s < out_dim {
            let e = (s + chunk).min(out_dim);
            v.push(s..e);
            s = e;
        }
        if v.is_empty() {
            v.push(0..0);
        }
        v
    }

    /// Every partition of `0..out_dim` the bit-identity test replays. All
    /// are in ascending order and tile the range exactly (the helper
    /// asserts that), so they can be walked with `split_at_mut` the same
    /// way the phase-5 pool will carve up one output buffer.
    ///
    /// Requires `out_dim >= 6` (every real projection is >= 512 rows).
    fn partitions(out_dim: usize) -> Vec<(&'static str, Vec<Range<usize>>)> {
        assert!(out_dim >= 6, "partition schemes assume a real out_dim");
        let half = out_dim / 2;
        let third = out_dim / 3;

        // 6-way, ceil-chunked: what a 6-thread pool does with `div_ceil`.
        // 768 -> six exact 128s; 2048 -> five 342s and a 338; 4096 -> five
        // 683s and a 681; 512 -> five 86s and an 82.
        let pool_ceil = chunked(out_dim, out_dim.div_ceil(6));

        // 6-way, floor-chunked with the remainder folded into the last
        // range (the other natural 6-thread split, uneven the other way).
        let floor = (out_dim / 6).max(1);
        let mut pool_floor: Vec<Range<usize>> =
            (0..5).map(|i| i * floor..(i + 1) * floor).collect();
        pool_floor.push(5 * floor..out_dim);

        vec![
            // Built via `chunked` rather than written out: a one-element
            // collection of a range trips `clippy::single_range_in_vec_init`.
            ("whole", chunked(out_dim, out_dim)),
            ("halves", vec![0..half, half..out_dim]),
            ("pool6-ceil", pool_ceil),
            ("pool6-floor-remainder", pool_floor),
            // Prime stride: leaves a remainder for every real out_dim
            // (768 % 7 = 5, 2048 % 7 = 4, 4096 % 7 = 1, 512 % 7 = 1, 1024 % 7 = 2).
            ("stride7-remainder", chunked(out_dim, 7)),
            // Single-row ranges everywhere.
            ("single-rows", (0..out_dim).map(|r| r..r + 1).collect()),
            // Empty ranges at the front, middle, and back, mixed with
            // single-row ranges and a long tail.
            (
                "uneven-with-empties",
                vec![
                    0..0,
                    0..1,
                    1..1,
                    1..third,
                    third..third,
                    third..out_dim - 1,
                    out_dim - 1..out_dim,
                    out_dim..out_dim,
                ],
            ),
            // An idle worker in the middle of an otherwise even split.
            (
                "empty-middle",
                vec![0..half, half..half, half..out_dim, out_dim..out_dim],
            ),
        ]
    }

    /// Replay every partition through `run` and assert bit-identity against
    /// the whole-matrix result.
    ///
    /// The destination is pre-filled with NaN, so a range that silently
    /// skipped a row would leave a NaN bit pattern and fail the comparison.
    fn assert_partitions_bit_identical<F>(f: &Fixture, scalar: bool, mut run: F)
    where
        F: FnMut(&Fixture, Range<usize>, &mut [f32], bool),
    {
        let path = if scalar { "scalar" } else { "avx2" };
        let mut full = vec![f32::NAN; f.out_dim];
        run(f, 0..f.out_dim, &mut full, scalar);
        assert!(
            full.iter().all(|v| v.is_finite()),
            "{} [{path}]: fixture produced a non-finite reference",
            f.name,
        );

        for (scheme, ranges) in partitions(f.out_dim) {
            let mut got = vec![f32::NAN; f.out_dim];
            let mut rest: &mut [f32] = &mut got;
            let mut cursor = 0usize;
            for r in &ranges {
                assert_eq!(
                    r.start, cursor,
                    "{} [{path}] {scheme}: ranges must tile 0..out_dim in order",
                    f.name,
                );
                let (head, tail) = rest.split_at_mut(r.len());
                run(f, r.clone(), head, scalar);
                rest = tail;
                cursor = r.end;
            }
            assert_eq!(
                cursor, f.out_dim,
                "{} [{path}] {scheme}: short tiling",
                f.name
            );
            assert!(rest.is_empty());

            for (r, (&g, &w)) in got.iter().zip(&full).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{} [{path}] {scheme}: row {r} differs ({g} vs {w})",
                    f.name,
                );
            }
        }
    }

    /// Drive the k-quant row range through the shared internals with an
    /// explicit kernel-path flag.
    fn run_k(f: &Fixture, acts: &[BlockQ8K], rows: Range<usize>, out: &mut [f32], scalar: bool) {
        let dot = k_quant_dot("test", f.format).unwrap();
        gemv_q8_k_impl(dot, "test", f.matrix(), acts, rows, out, scalar).unwrap();
    }

    /// Drive the Q8_0 row range through the shared internals.
    fn run_0(f: &Fixture, acts: &[BlockQ8_0], rows: Range<usize>, out: &mut [f32], scalar: bool) {
        gemv_q8_0_impl("test", f.matrix(), acts, rows, out, scalar).unwrap();
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

    /// The load-bearing property of phase 5: any row partition of a k-quant
    /// GEMV reassembles bit-for-bit into the whole-matrix result, on both
    /// the AVX2 and the scalar kernel path, at the real projection shapes.
    #[test]
    fn k_quant_row_partitions_are_bit_identical() {
        for f in k_quant_fixtures() {
            let acts = q8_k_acts(f.in_dim, 0xAC ^ f.in_dim as u64);
            for scalar in [false, true] {
                assert_partitions_bit_identical(&f, scalar, |f, rows, out, scalar| {
                    run_k(f, &acts, rows, out, scalar)
                });
            }
        }
    }

    /// Same property for the Q8_0 x Q8_0 path (`attn_k`, 2048 -> 512).
    #[test]
    fn q8_0_row_partitions_are_bit_identical() {
        let f = Fixture::new("attn_k (Q8_0)", QuantFormat::Q8_0, 2048, 512);
        let acts = q8_0_acts(f.in_dim, 0xAC7);
        for scalar in [false, true] {
            assert_partitions_bit_identical(&f, scalar, |f, rows, out, scalar| {
                run_0(f, &acts, rows, out, scalar)
            });
        }
    }

    /// The public row-range entry points agree with the public whole-matrix
    /// ones (the wrappers really are the same code path), and the 6-way
    /// split works through actual threads holding disjoint `&mut [f32]`
    /// sub-slices of one output buffer — the phase-5 call shape.
    #[test]
    fn public_row_range_entry_points_match_whole_matrix() {
        // Expert gate: Q4_K 2048 -> 768, split six ways across threads.
        let f = Fixture::new("expert gate (Q4_K)", QuantFormat::Q4_K, 2048, 768);
        let acts = q8_k_acts(f.in_dim, 0x51CE);
        let mut want = vec![0f32; f.out_dim];
        gemv_q8_k(f.format, &f.weight, f.in_dim, f.out_dim, &acts, &mut want).unwrap();

        let chunk = f.out_dim.div_ceil(6);
        let mut got = vec![f32::NAN; f.out_dim];
        std::thread::scope(|s| {
            for (i, part) in got.chunks_mut(chunk).enumerate() {
                let (f, acts) = (&f, &acts);
                let rows = i * chunk..i * chunk + part.len();
                s.spawn(move || {
                    gemv_q8_k_rows(f.format, &f.weight, f.in_dim, f.out_dim, acts, rows, part)
                        .unwrap();
                });
            }
        });
        for (r, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "gemv_q8_k_rows row {r}");
        }

        // attn_k: Q8_0 2048 -> 512, uneven three-way split, single-threaded.
        let f0 = Fixture::new("attn_k (Q8_0)", QuantFormat::Q8_0, 2048, 512);
        let acts0 = q8_0_acts(f0.in_dim, 0x51CF);
        let mut want0 = vec![0f32; f0.out_dim];
        gemv_q8_0(&f0.weight, f0.in_dim, f0.out_dim, &acts0, &mut want0).unwrap();
        let mut got0 = vec![f32::NAN; f0.out_dim];
        let mut rest: &mut [f32] = &mut got0;
        let mut cursor = 0usize;
        for end in [1usize, 333, 512] {
            let (head, tail) = rest.split_at_mut(end - cursor);
            gemv_q8_0_rows(&f0.weight, f0.in_dim, f0.out_dim, &acts0, cursor..end, head).unwrap();
            rest = tail;
            cursor = end;
        }
        for (r, (&g, &w)) in got0.iter().zip(&want0).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "gemv_q8_0_rows row {r}");
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
            assert_eq!(
                gemv_q8_k_rows(format, &[], 256, 1, &acts, 0..1, &mut out).unwrap_err(),
                KernelError::UnsupportedFormat {
                    what: "gemv_q8_k_rows",
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

    /// Build a deliberately inverted range. Written as a struct literal
    /// because `clippy::reversed_empty_ranges` (deny-by-default) rejects
    /// `3..1` written out — but rejecting such a range without panicking is
    /// exactly what is under test here.
    fn raw_range(start: usize, end: usize) -> Range<usize> {
        Range { start, end }
    }

    /// Every malformed row range is a typed error, never a panic: the
    /// range is checked against the *whole* matrix's `out_dim`, and the
    /// full-matrix geometry is still validated first.
    #[test]
    fn rejects_bad_row_ranges() {
        let acts = vec![BlockQ8K::default(); 8]; // 2048 weights
        let row = QuantFormat::Q4_K.row_bytes(2048).unwrap();
        let w = vec![0u8; row * 4]; // out_dim 4
        let mut out = vec![0f32; 4];

        let rows_err = |rows: Range<usize>, out: &mut [f32]| {
            gemv_q8_k_rows(QuantFormat::Q4_K, &w, 2048, 4, &acts, rows, out).unwrap_err()
        };

        // Inverted range.
        assert_eq!(
            rows_err(raw_range(3, 1), &mut []),
            KernelError::LengthMismatch {
                what: ROW_RANGE_ORDER,
                left: 3,
                right: 1,
            },
        );
        // End past the last row.
        assert_eq!(
            rows_err(2..5, &mut out[..3]),
            KernelError::LengthMismatch {
                what: ROW_RANGE_BOUND,
                left: 5,
                right: 4,
            },
        );
        // Empty range starting past the last row.
        assert_eq!(
            rows_err(9..9, &mut []),
            KernelError::LengthMismatch {
                what: ROW_RANGE_BOUND,
                left: 9,
                right: 4,
            },
        );
        // Absurd range: must not overflow or panic on `end - start`.
        assert_eq!(
            rows_err(usize::MAX - 1..usize::MAX, &mut []),
            KernelError::LengthMismatch {
                what: ROW_RANGE_BOUND,
                left: usize::MAX,
                right: 4,
            },
        );
        assert_eq!(
            rows_err(raw_range(usize::MAX, 0), &mut []),
            KernelError::LengthMismatch {
                what: ROW_RANGE_ORDER,
                left: usize::MAX,
                right: 0,
            },
        );
        // `out` must be exactly the range, not the whole output buffer.
        assert_eq!(
            rows_err(1..3, &mut out),
            KernelError::LengthMismatch {
                what: "gemv_q8_k_rows: weight bytes vs rows / out vs range",
                left: 4,
                right: 2,
            },
        );
        // Full-matrix geometry is still checked, before the range.
        assert_eq!(
            gemv_q8_k_rows(
                QuantFormat::Q4_K,
                &w[..row * 3],
                2048,
                4,
                &acts,
                0..1,
                &mut out[..1]
            )
            .unwrap_err(),
            KernelError::LengthMismatch {
                what: "gemv_q8_k_rows: weight bytes vs rows / out vs range",
                left: row * 3,
                right: row * 4,
            },
        );

        // Same range checks on the Q8_0 entry point.
        let acts0 = vec![BlockQ8_0::default(); 64];
        let row0 = QuantFormat::Q8_0.row_bytes(2048).unwrap();
        let w0 = vec![0u8; row0 * 2];
        assert_eq!(
            gemv_q8_0_rows(&w0, 2048, 2, &acts0, raw_range(2, 1), &mut []).unwrap_err(),
            KernelError::LengthMismatch {
                what: ROW_RANGE_ORDER,
                left: 2,
                right: 1,
            },
        );
        assert_eq!(
            gemv_q8_0_rows(&w0, 2048, 2, &acts0, 0..3, &mut out[..3]).unwrap_err(),
            KernelError::LengthMismatch {
                what: ROW_RANGE_BOUND,
                left: 3,
                right: 2,
            },
        );
        assert_eq!(
            gemv_q8_0_rows(&w0, 2048, 2, &acts0, 0..2, &mut out).unwrap_err(),
            KernelError::LengthMismatch {
                what: "gemv_q8_0_rows: weight bytes vs rows / out vs range",
                left: 4,
                right: 2,
            },
        );
    }

    #[test]
    fn zero_out_dim_is_a_no_op() {
        let acts = q8_k_acts(256, 1);
        let mut out: [f32; 0] = [];
        gemv_q8_k(QuantFormat::Q4_K, &[], 256, 0, &acts, &mut out).unwrap();
        gemv_q8_k_rows(QuantFormat::Q4_K, &[], 256, 0, &acts, 0..0, &mut out).unwrap();
    }

    /// An empty range in the middle of a real matrix writes nothing and
    /// leaves the caller's buffer untouched.
    #[test]
    fn empty_row_range_writes_nothing() {
        let f = Fixture::new("expert gate (Q4_K)", QuantFormat::Q4_K, 2048, 768);
        let acts = q8_k_acts(f.in_dim, 0xE377);
        let mut out = vec![7.5f32; f.out_dim];
        {
            let (_, mid) = out.split_at_mut(300);
            let (empty, _) = mid.split_at_mut(0);
            gemv_q8_k_rows(
                f.format,
                &f.weight,
                f.in_dim,
                f.out_dim,
                &acts,
                300..300,
                empty,
            )
            .unwrap();
        }
        assert!(out.iter().all(|&v| v == 7.5));
    }
}
