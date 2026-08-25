//! **PRESERVED PROTOTYPE. NOT A MERGE CANDIDATE. NOTHING CALLS THIS.**
//!
//! This module exists so that a figure quoted in `docs/roadmap.md` and
//! `docs/experiments.md` has code behind it instead of a memory of code. It
//! is the reconstruction of the hoisted Q4_K prototype whose warm diagnostic
//! (0.482-0.495 ms against production's 0.551-0.568 ms, so **1.15-1.18x**)
//! is cited by EXP-026 as the CPU-side half of any GPU-vs-CPU ratio. It
//! lives on branch `perf/q4k-row-hoist-prototype` under the project's
//! standing rule that measured-and-set-aside work gets a branch rather than
//! a deletion. `main` carries none of it.
//!
//! # The asymmetry it removes
//!
//! [`super::gemv_q8_k_batched`] loops weight rows outer and activations
//! inner, and each element is one call of `vec_dot_q4_k_q8_k` on the packed
//! row. That call re-does, on every one of the `n_acts` passes over the same
//! row, work that depends only on the row:
//!
//! 1. the f16 to f32 conversion of `d` and `dmin`, once per super-block;
//! 2. the unpack of the 12-byte packed field into eight 6-bit scales and
//!    eight 6-bit mins;
//! 3. the nibble split, expanding 128 packed bytes into 256 int8 weights in
//!    ggml's order.
//!
//! A 512-token prefill chunk routes about 32 rows to an expert, so that is
//! about 32x more of (1), (2) and (3) than the arithmetic needs. This module
//! does all three once per weight row into scratch, then runs the *same*
//! integer inner loop against each activation.
//!
//! # What is and is not changed
//!
//! The AVX2 inner loop is production's, instruction for instruction, with
//! one difference and only one: where production loads 32 packed bytes and
//! derives `q4l`/`q4h` with an `and` and a shift, this loads the two 32-byte
//! halves straight out of the pre-expanded row. That elision is the whole
//! point of the hoist. `maddubs` and `madd` see identical values, the
//! sub-block scale is applied in integers exactly as before, the mins still
//! fold through `bsums` per activation (they depend on the activation, so
//! they cannot be hoisted), and the float accumulators are production's:
//! eight f32 lanes plus four for the min correction, summed once per
//! `(row, activation)` pair.
//!
//! **The expansion itself is deliberately scalar**, as the original
//! prototype's was. A vectorized expansion would be faster; leaving it
//! scalar makes the resulting figure a *floor* on what closing the asymmetry
//! is worth, which is the only honest way to quote a number from a
//! prototype.
//!
//! # Bit-identity, measured rather than assumed
//!
//! Because the hoist moves *what is recomputed* and not what is computed,
//! this reconstruction turns out to be **bit-identical** to
//! [`super::gemv_q8_k_batched`] on every shape tested below, which is what
//! rule 4 in `docs/experiments.md` demands of a change claiming identical
//! math. That is a property of **this reconstruction**, established here by
//! a `to_bits()` comparison; the original prototype the 1.15-1.18x figure
//! came from was never checked for it, so the figure must not be read as a
//! verified bit-identical A/B. It also holds only while both arms dispatch
//! the same way, which is why `RAMVAMP_FORCE_SCALAR` is honoured here too.
//!
//! Bit-identity is not an argument for merging. The figure behind this is a
//! warm diagnostic taken on a throttled, uncgrouped box before the harness
//! had convergence gating; the roadmap lists the hoist as an **open
//! decision** for the author, and it stays open until there is a cold,
//! in-cgroup measurement to take it on.
//!
//! # Cost this trades against
//!
//! The hoist buys its arithmetic saving with a per-row scratch buffer:
//! `in_dim` bytes of expanded weights (2,048 B for a v0 expert row) against
//! the 1,152 B packed row production keeps in L1. That is the trade a
//! decision on this has to weigh, and it is why the prototype is preserved
//! rather than merged on the strength of one warm number.
//!
//! # Self-contained on purpose
//!
//! Nothing in `quants` is made more visible for this. The block offsets, the
//! scale unpack and the two horizontal sums are restated here so that the
//! only change this branch makes to a shipping file is one `mod` line, which
//! keeps the preserved prototype from drifting into the production kernels.
//! The tests check the restatements against the real kernel end to end.

use std::ops::Range;
use std::sync::OnceLock;

use super::KernelError;
use super::quants::{BlockQ8K, QuantFormat, avx2, f16_to_f32};

/// Weights per k-quant super-block.
const QK_K: usize = 256;
/// Packed bytes per Q4_K super-block.
const BLOCK_BYTES: usize = 144;
/// Byte offset of `d: f16` inside a Q4_K block.
const OFF_D: usize = 0;
/// Byte offset of `dmin: f16`.
const OFF_DMIN: usize = 2;
/// Byte offset of the 12-byte packed `(scale, min)` field.
const OFF_SCALES: usize = 4;
/// Byte offset of the 128 packed nibble bytes.
const OFF_QS: usize = 16;
/// 32-weight sub-blocks per super-block.
const SUBBLOCKS: usize = 8;

// The restated geometry has to agree with the format the rest of the crate
// defines, or the prototype would be measuring a different kernel.
const _: () = assert!(QuantFormat::Q4_K.block_bytes() == BLOCK_BYTES);
const _: () = assert!(QuantFormat::Q4_K.block_weights() == QK_K);

/// One weight row, expanded once, reused across every activation.
///
/// Flat vectors rather than per-block structs: the buffers are allocated
/// once per call and refilled per row, so a prefill-shaped call
/// (`out_dim` rows) does one allocation and not `out_dim` of them.
struct HoistedRow {
    /// `[b][256]` weights as unsigned nibbles in ggml order: per 64-weight
    /// span, 32 low nibbles then 32 high. Unsigned because `maddubs` wants
    /// the weight operand unsigned.
    aux: Vec<u8>,
    /// `[b][8]` 6-bit sub-block scales.
    sc: Vec<u8>,
    /// `[b][8]` 6-bit sub-block mins.
    mins: Vec<u8>,
    /// `[b]` super-block scale, f16-converted once.
    d: Vec<f32>,
    /// `[b]` super-block min scale, f16-converted once.
    dmin: Vec<f32>,
}

impl HoistedRow {
    fn new(n_blocks: usize) -> Self {
        Self {
            aux: vec![0u8; n_blocks * QK_K],
            sc: vec![0u8; n_blocks * SUBBLOCKS],
            mins: vec![0u8; n_blocks * SUBBLOCKS],
            d: vec![0f32; n_blocks],
            dmin: vec![0f32; n_blocks],
        }
    }

    /// Do every row-only piece of the dot, once, for one packed weight row.
    ///
    /// Deliberately scalar. See the module docs: a vectorized expansion
    /// would make the prototype's figure less conservative, and the figure
    /// is only useful as a floor.
    fn fill(&mut self, row: &[u8], n_blocks: usize) {
        for b in 0..n_blocks {
            let block = &row[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            self.d[b] = f16_at(block, OFF_D);
            self.dmin[b] = f16_at(block, OFF_DMIN);

            let mut scales = [0u8; 12];
            scales.copy_from_slice(&block[OFF_SCALES..OFF_SCALES + 12]);
            let (sc, mins) = unpack_scales_mins(&scales);
            self.sc[b * SUBBLOCKS..(b + 1) * SUBBLOCKS].copy_from_slice(&sc);
            self.mins[b * SUBBLOCKS..(b + 1) * SUBBLOCKS].copy_from_slice(&mins);

            let qs = &block[OFF_QS..BLOCK_BYTES];
            let aux = &mut self.aux[b * QK_K..(b + 1) * QK_K];
            for span in 0..4 {
                let q = &qs[32 * span..32 * span + 32];
                for l in 0..32 {
                    aux[64 * span + l] = q[l] & 0xF;
                    aux[64 * span + 32 + l] = q[l] >> 4;
                }
            }
        }
    }
}

/// Read a little-endian f16 at `off` and widen it, without assuming any
/// alignment beyond 1 (a packed sub-tensor offset guarantees only 2).
#[inline]
fn f16_at(block: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[off], block[off + 1]]))
}

/// Unpack the 12-byte k-quant scale field into eight 6-bit scales and eight
/// 6-bit mins, via ggml's three-word mask trick.
///
/// A restatement of the production unpack, which is private to `quants`.
/// The end-to-end bit-identity test below is what proves the restatement.
fn unpack_scales_mins(scales: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let mut utmp = [
        u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]),
        u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]),
        u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]),
        0u32,
    ];
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    let mut sc = [0u8; 8];
    let mut mins = [0u8; 8];
    sc[..4].copy_from_slice(&utmp[0].to_le_bytes());
    sc[4..].copy_from_slice(&utmp[1].to_le_bytes());
    mins[..4].copy_from_slice(&utmp[2].to_le_bytes());
    mins[4..].copy_from_slice(&utmp[3].to_le_bytes());
    (sc, mins)
}

/// Batched Q4_K x Q8_K GEMV with the per-row dequant setup hoisted out of
/// the activation loop.
///
/// Same contract as [`super::gemv_q8_k_batched`] restricted to Q4_K:
/// `weight`, `in_dim` and `out_dim` describe the whole matrix, `acts` is the
/// flat concatenation of `n_acts` activation rows of `in_dim / 256` blocks
/// each, and `out` holds exactly `rows.len() * n_acts` floats in
/// `[weight_row][activation]` order.
///
/// # Errors
///
/// [`KernelError::IndivisibleRow`] when `in_dim` is not a multiple of 256,
/// [`KernelError::LengthMismatch`] when `weight`, `rows` or `out` disagree
/// with the declared geometry, and [`KernelError::BlockCountMismatch`] when
/// `acts` does not hold `n_acts * (in_dim / 256)` blocks.
pub fn gemv_q4_k_q8_k_batched_hoisted(
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    acts: &[BlockQ8K],
    n_acts: usize,
    rows: Range<usize>,
    out: &mut [f32],
) -> Result<(), KernelError> {
    let row_bytes = QuantFormat::Q4_K.row_bytes(in_dim)?;
    let n_blocks = in_dim / QK_K;
    if weight.len() != out_dim * row_bytes {
        return Err(KernelError::LengthMismatch {
            what: "q4k_row_hoist: weight bytes vs out_dim x row_bytes",
            left: weight.len(),
            right: out_dim * row_bytes,
        });
    }
    if rows.start > rows.end || rows.end > out_dim {
        return Err(KernelError::LengthMismatch {
            what: "q4k_row_hoist: row range vs out_dim",
            left: rows.end,
            right: out_dim,
        });
    }
    if out.len() != rows.len() * n_acts {
        return Err(KernelError::LengthMismatch {
            what: "q4k_row_hoist: out vs range x n_acts",
            left: out.len(),
            right: rows.len() * n_acts,
        });
    }
    if acts.len() != n_acts * n_blocks {
        return Err(KernelError::BlockCountMismatch {
            weight_blocks: n_acts * n_blocks,
            activation_blocks: acts.len(),
        });
    }
    if n_acts == 0 || rows.is_empty() {
        return Ok(());
    }

    // One allocation for the whole call, refilled per weight row.
    let mut hoisted = HoistedRow::new(n_blocks);
    for (i, dst) in out.chunks_mut(n_acts).enumerate() {
        let r = rows.start + i;
        hoisted.fill(&weight[r * row_bytes..(r + 1) * row_bytes], n_blocks);
        for (a, o) in dst.iter_mut().enumerate() {
            let act_row = &acts[a * n_blocks..(a + 1) * n_blocks];
            *o = dot_expanded(&hoisted, act_row);
        }
    }
    Ok(())
}

/// `RAMVAMP_FORCE_SCALAR`, honoured for the same reason `gemv.rs` honours
/// it: with the env var set, production takes its scalar dot, and a
/// prototype that stayed on AVX2 would stop being comparable to it.
fn force_scalar() -> bool {
    static FORCE: OnceLock<bool> = OnceLock::new();
    *FORCE.get_or_init(|| std::env::var_os("RAMVAMP_FORCE_SCALAR").is_some())
}

/// One `(expanded weight row, activation row)` dot.
fn dot_expanded(row: &HoistedRow, acts: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if !force_scalar() && avx2::avx2_fma_available() {
        // SAFETY: AVX2+FMA presence was checked at runtime just above, and
        // `HoistedRow` sizes every buffer at `n_blocks` times its stride,
        // which is `acts.len()` times its stride by the caller's validation.
        return unsafe { x86::dot_expanded(row, acts) };
    }
    dot_expanded_scalar(row, acts)
}

/// Scalar reference for [`dot_expanded`], mirroring the scalar
/// `vec_dot_q4_k_q8_k` over pre-expanded weights.
fn dot_expanded_scalar(row: &HoistedRow, acts: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (b, y) in acts.iter().enumerate() {
        let d = row.d[b] * y.d;
        let dmin = row.dmin[b] * y.d;
        let mins = &row.mins[b * SUBBLOCKS..(b + 1) * SUBBLOCKS];
        let sc = &row.sc[b * SUBBLOCKS..(b + 1) * SUBBLOCKS];
        let aux = &row.aux[b * QK_K..(b + 1) * QK_K];

        let mut sumi_mins = 0i32;
        for (g, &bs) in y.bsums.iter().enumerate() {
            sumi_mins += i32::from(bs) * i32::from(mins[g / 2]);
        }
        let mut sumi = 0i32;
        for j in 0..SUBBLOCKS {
            let mut s = 0i32;
            for l in 0..32 {
                s += i32::from(aux[32 * j + l]) * i32::from(y.qs[32 * j + l]);
            }
            sumi += i32::from(sc[j]) * s;
        }
        sumf += d * sumi as f32 - dmin * sumi_mins as f32;
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::{BlockQ8K, HoistedRow, QK_K, SUBBLOCKS};

    /// Horizontal sum of the eight f32 lanes. Production's, restated.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum_ps_8(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }

    /// Horizontal sum of the four f32 lanes. Production's, restated.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum_ps_4(v: __m128) -> f32 {
        let s = _mm_add_ps(v, _mm_movehl_ps(v, v));
        let s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }

    /// The min correction, per activation. Production's, restated.
    ///
    /// This one cannot be hoisted: `mins` is a property of the weight row
    /// and `bsums` is a property of the activation, so the product exists
    /// only per pair. The *unpack* that produced `mins` is what moved.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mins_dot(y: &BlockQ8K, mins: &[u8]) -> __m128i {
        // SAFETY: `y.bsums` is 32 bytes and `mins` is 8 (a `SUBBLOCKS`
        // stride of `HoistedRow::mins`), both read unaligned; the rest is
        // register-only.
        unsafe {
            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr().cast());
            let q8s = _mm_hadd_epi16(
                _mm256_castsi256_si128(q8sums),
                _mm256_extracti128_si256::<1>(q8sums),
            );
            let minsv = _mm_cvtepu8_epi16(_mm_loadl_epi64(mins.as_ptr().cast()));
            _mm_madd_epi16(minsv, q8s)
        }
    }

    /// Production's `vec_dot_q4_k_q8_k` inner loop over a pre-expanded row.
    ///
    /// The only difference from `quants::avx2::x86::vec_dot_q4_k_q8_k` is
    /// where `q4l` and `q4h` come from: there they are an `and` and a shift
    /// of 32 freshly loaded packed bytes, done once per `(row, activation)`
    /// pair; here they are two loads out of the row expanded once. Every
    /// other instruction, and every value either sees, is the same.
    ///
    /// # Safety
    ///
    /// The caller must have checked AVX2+FMA, and `row`'s buffers must be
    /// sized for at least `acts.len()` super-blocks.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_expanded(row: &HoistedRow, acts: &[BlockQ8K]) -> f32 {
        // SAFETY (whole body): the caller guarantees AVX2+FMA and the
        // buffer sizes, so every `add`/`loadu` below reads inside one
        // 256-byte `aux` stride, one 8-byte `sc` stride, or `y.qs`
        // (256 B); every load is unaligned by construction, which is what
        // `_mm256_loadu_si256` requires.
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let mut acc_m = _mm_setzero_ps();
            for (b, y) in acts.iter().enumerate() {
                let d = row.d[b] * y.d;
                let dmin = row.dmin[b] * y.d;

                let prod = mins_dot(y, &row.mins[b * SUBBLOCKS..(b + 1) * SUBBLOCKS]);
                acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                let aux = row.aux.as_ptr().add(b * QK_K);
                let sc = &row.sc[b * SUBBLOCKS..(b + 1) * SUBBLOCKS];
                let mut sumi = _mm256_setzero_si256();
                for j in 0..4 {
                    // Hoisted: no `and`, no `srli`. The 64 weights of span
                    // `j` are already split into 32 low then 32 high.
                    let q4l = _mm256_loadu_si256(aux.add(64 * j).cast());
                    let q4h = _mm256_loadu_si256(aux.add(64 * j + 32).cast());
                    let q8l = _mm256_loadu_si256(y.qs.as_ptr().add(64 * j).cast());
                    let q8h = _mm256_loadu_si256(y.qs.as_ptr().add(64 * j + 32).cast());
                    let p16l = _mm256_madd_epi16(
                        _mm256_set1_epi16(i16::from(sc[2 * j])),
                        _mm256_maddubs_epi16(q4l, q8l),
                    );
                    let p16h = _mm256_madd_epi16(
                        _mm256_set1_epi16(i16::from(sc[2 * j + 1])),
                        _mm256_maddubs_epi16(q4h, q8h),
                    );
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum_ps_8(acc) - hsum_ps_4(acc_m)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::gemv_q8_k_batched;
    use crate::kernels::quants::quantize_row_q8_k;

    /// splitmix64, so a fixture is reproducible without a dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u32 << 24) as f32
        }
    }

    /// A packed Q4_K matrix with plausible f16 scales and random payload.
    fn packed_matrix(in_dim: usize, out_dim: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        let row_bytes = in_dim / QK_K * BLOCK_BYTES;
        let mut bytes = vec![0u8; out_dim * row_bytes];
        for block in bytes.chunks_exact_mut(BLOCK_BYTES) {
            let d = crate::kernels::quants::f32_to_f16(0.01 + rng.unit() * 0.02);
            let dmin = crate::kernels::quants::f32_to_f16(0.005 + rng.unit() * 0.01);
            block[OFF_D..OFF_D + 2].copy_from_slice(&d.to_le_bytes());
            block[OFF_DMIN..OFF_DMIN + 2].copy_from_slice(&dmin.to_le_bytes());
            for b in &mut block[OFF_SCALES..] {
                *b = (rng.next() >> 24) as u8;
            }
        }
        bytes
    }

    fn acts(n_acts: usize, in_dim: usize, seed: u64) -> Vec<BlockQ8K> {
        let per_row = in_dim / QK_K;
        let mut rng = Rng(seed);
        let mut out = vec![BlockQ8K::default(); n_acts * per_row];
        let mut row = vec![0f32; in_dim];
        for a in 0..n_acts {
            for v in &mut row {
                *v = (rng.unit() - 0.5) * 4.0;
            }
            quantize_row_q8_k(&row, &mut out[a * per_row..(a + 1) * per_row]).expect("quantize");
        }
        out
    }

    /// The prototype against the production kernel it is a prototype of, at
    /// the v0 expert-GEMV shape and two smaller ones.
    ///
    /// The hoist moves *what is recomputed*, not what is computed: the
    /// integer path sees identical values and the float accumulators are
    /// production's, so under rule 4 this must produce identical bits and
    /// not merely agree inside a tolerance. Any drift here means the
    /// restated scale unpack or the restated block offsets are wrong, which
    /// is exactly what this test is for.
    #[test]
    fn hoisted_matches_production_bit_for_bit() {
        for &(in_dim, out_dim, n_acts) in &[(2048, 768, 32), (256, 5, 1), (512, 17, 7)] {
            let weight = packed_matrix(in_dim, out_dim, 0x5EED_0BAD_C0FF_EE01);
            let a = acts(n_acts, in_dim, 0xA11C_E5EE_D000_0001);

            let mut want = vec![0f32; out_dim * n_acts];
            gemv_q8_k_batched(
                QuantFormat::Q4_K,
                &weight,
                in_dim,
                out_dim,
                &a,
                n_acts,
                0..out_dim,
                &mut want,
            )
            .expect("production");

            let mut got = vec![0f32; out_dim * n_acts];
            gemv_q4_k_q8_k_batched_hoisted(
                &weight,
                in_dim,
                out_dim,
                &a,
                n_acts,
                0..out_dim,
                &mut got,
            )
            .expect("hoisted");

            for (i, (w, g)) in want.iter().zip(&got).enumerate() {
                assert_eq!(
                    w.to_bits(),
                    g.to_bits(),
                    "output {i} at ({in_dim}, {out_dim}, {n_acts}): {w} vs {g}"
                );
            }
        }
    }

    /// The scalar arm of the prototype against the AVX2 arm, so the
    /// non-x86 path is covered by the same fixture. Float summation order
    /// differs between them exactly as it does in production, so this is a
    /// tolerance check and not a bit check.
    #[test]
    fn scalar_and_avx2_arms_agree() {
        let (in_dim, out_dim, n_acts) = (2048, 64, 8);
        let weight = packed_matrix(in_dim, out_dim, 0x0BAD_F00D_1234_5678);
        let a = acts(n_acts, in_dim, 0xFEED_FACE_0000_0001);
        let n_blocks = in_dim / QK_K;
        let row_bytes = n_blocks * BLOCK_BYTES;

        let mut hoisted = HoistedRow::new(n_blocks);
        for r in 0..out_dim {
            hoisted.fill(&weight[r * row_bytes..(r + 1) * row_bytes], n_blocks);
            for act in 0..n_acts {
                let act_row = &a[act * n_blocks..(act + 1) * n_blocks];
                let s = dot_expanded_scalar(&hoisted, act_row);
                let v = dot_expanded(&hoisted, act_row);
                let tol = 1e-4 * s.abs().max(1e-3);
                assert!(
                    (s - v).abs() <= tol,
                    "row {r} act {act}: scalar {s} vs dispatched {v}"
                );
            }
        }
    }

    /// A row range is the shape the compute pool actually calls with, and
    /// the concatenation of any partition must reproduce the whole-matrix
    /// result bit for bit, exactly as production's row ranges do.
    #[test]
    fn row_ranges_concatenate_bit_for_bit() {
        let (in_dim, out_dim, n_acts) = (512, 24, 5);
        let weight = packed_matrix(in_dim, out_dim, 0x1111_2222_3333_4444);
        let a = acts(n_acts, in_dim, 0x5555_6666_7777_8888);

        let mut whole = vec![0f32; out_dim * n_acts];
        gemv_q4_k_q8_k_batched_hoisted(
            &weight,
            in_dim,
            out_dim,
            &a,
            n_acts,
            0..out_dim,
            &mut whole,
        )
        .expect("whole");

        let mut parts = vec![0f32; out_dim * n_acts];
        for range in [0..7usize, 7..7, 7..20, 20..out_dim] {
            let dst = &mut parts[range.start * n_acts..range.end * n_acts];
            gemv_q4_k_q8_k_batched_hoisted(
                &weight,
                in_dim,
                out_dim,
                &a,
                n_acts,
                range.clone(),
                dst,
            )
            .expect("part");
        }
        for (i, (w, p)) in whole.iter().zip(&parts).enumerate() {
            assert_eq!(w.to_bits(), p.to_bits(), "output {i}");
        }
    }

    /// Geometry mismatches are typed errors, not panics: the prototype is
    /// still `ramvamp-core` code and the crate rule holds here too.
    #[test]
    fn bad_geometry_is_a_typed_error() {
        let weight = packed_matrix(256, 4, 1);
        let a = acts(2, 256, 2);
        let mut out = vec![0f32; 8];

        assert!(matches!(
            gemv_q4_k_q8_k_batched_hoisted(&weight, 250, 4, &a, 2, 0..4, &mut out),
            Err(KernelError::IndivisibleRow { .. })
        ));
        assert!(matches!(
            gemv_q4_k_q8_k_batched_hoisted(&weight, 256, 5, &a, 2, 0..5, &mut out),
            Err(KernelError::LengthMismatch { .. })
        ));
        assert!(matches!(
            gemv_q4_k_q8_k_batched_hoisted(&weight, 256, 4, &a, 2, 0..9, &mut out),
            Err(KernelError::LengthMismatch { .. })
        ));
        assert!(matches!(
            gemv_q4_k_q8_k_batched_hoisted(&weight, 256, 4, &a, 3, 0..4, &mut out),
            Err(KernelError::LengthMismatch { .. })
        ));
        let mut short = vec![0f32; 4];
        assert!(matches!(
            gemv_q4_k_q8_k_batched_hoisted(&weight, 256, 4, &a, 1, 0..4, &mut short),
            Err(KernelError::BlockCountMismatch { .. })
        ));
    }
}
