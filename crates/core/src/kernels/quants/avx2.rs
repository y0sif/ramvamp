//! Runtime-dispatched AVX2+FMA kernels with scalar fallback.
//!
//! The public functions here are the production dispatch layer: each picks
//! the AVX2+FMA implementation when the CPU supports both (checked via
//! `is_x86_feature_detected!`, which caches after the first probe) and falls
//! back to the scalar reference in the parent module otherwise. The
//! `force_scalar` parameter forces the scalar path so tests and benches can
//! compare both implementations through an identical call shape; there is no
//! environment-variable switch. On non-x86_64 targets everything compiles to
//! the scalar path.
//!
//! The AVX2 bodies mirror ggml's x86 recipes (`ggml-cpu/arch/x86/quants.c`):
//! the k-quant 6-bit scale unpack via the `0x3f3f3f3f` / `0x0f0f0f0f` /
//! `0x03030303` word-mask trick, `_mm256_maddubs_epi16` for the
//! nibble-vs-int8 pair products (operand order matters: the *first* operand
//! is treated as unsigned, so the packed weights go first and the signed
//! activations second), `_mm256_madd_epi16` to apply sub-block scales while
//! widening pair sums into i32 lanes, the k-quant mins folded in from the
//! activation `bsums` with `_mm_madd_epi16`, and one `_mm256_fmadd_ps` per
//! super-block.
//!
//! # Numerics
//!
//! The integer part of every dot product is exact and identical to the
//! scalar reference: no `maddubs` i16 saturation is reachable (worst case
//! `2 * 63 * 128 = 16128` for q6_K weights against an i8 activation, and
//! `2 * 128 * 127 = 32512` for q8_0 whose activations the quantizer keeps in
//! `[-127, 127]`; both under 32767), and the i32 scale/accumulate stages are
//! orders of magnitude below overflow. Only the order of the final float
//! accumulation differs (eight lanes reduced at the end vs one running sum),
//! so AVX2-vs-scalar tests compare with a tolerance scaled by the sum of
//! per-element term magnitudes.
//!
//! The two activation quantizers are byte-identical to the scalar reference
//! by construction. This is a deliberate deviation from ggml's AVX2
//! quantizers where their rounding differs from the scalar reference:
//! ggml's AVX2 `quantize_row_q8_0` rounds with `_mm256_round_ps` (ties to
//! even) and scales by `127/amax`, while its scalar reference uses `roundf`
//! (ties away from zero) and `1/d`; we match the SCALAR reference — SIMD
//! ties-away rounding via an exact tie fix-up, scale math kept in scalar
//! f32 identical to the reference. `quantize_row_q8_k` needs no fix-up:
//! `_mm256_cvtps_epi32` under the default MXCSR rounding mode is exactly
//! the reference's `round_ties_even`.
//!
//! # Alignment
//!
//! Weight rows may be only 1-byte aligned (packed sub-tensor offsets
//! guarantee as little as 2-byte alignment, and a row inside a block can
//! start anywhere): every vector load here is `_mm256_loadu_*` /
//! `_mm_loadu_*`, assuming nothing beyond byte alignment. Tested against a
//! deliberately misaligned copy in `misaligned_weight_row_matches`.

use super::{BlockQ8_0, BlockQ8K, KernelError, QuantFormat, dot, quantize};

/// True when the running CPU supports AVX2 and FMA.
///
/// `is_x86_feature_detected!` caches the CPUID probe, so calling this per
/// row costs an atomic load.
#[inline]
pub fn avx2_fma_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Validate a weight row against an activation row (same checks and error
/// precedence as the scalar kernels).
#[cfg(target_arch = "x86_64")]
fn validate_row(
    bytes: &[u8],
    format: QuantFormat,
    activation_blocks: usize,
) -> Result<(), KernelError> {
    let block_bytes = format.block_bytes();
    if bytes.len() % block_bytes != 0 {
        return Err(KernelError::RowBytesNotBlockMultiple {
            format,
            len: bytes.len(),
            block_bytes,
        });
    }
    let weight_blocks = bytes.len() / block_bytes;
    if weight_blocks != activation_blocks {
        return Err(KernelError::BlockCountMismatch {
            weight_blocks,
            activation_blocks,
        });
    }
    Ok(())
}

/// Validate a quantizer input against its output blocks (same check as the
/// scalar quantizers).
#[cfg(target_arch = "x86_64")]
fn validate_quantize(
    format: QuantFormat,
    floats: usize,
    out_blocks: usize,
) -> Result<(), KernelError> {
    if floats != out_blocks * format.block_weights() {
        return Err(KernelError::QuantizeLenMismatch {
            format,
            floats,
            out_blocks,
        });
    }
    Ok(())
}

/// Dot product of a packed Q4_K weight row with Q8_K activations,
/// dispatched to AVX2+FMA when available (else the scalar reference).
///
/// Same contract as [`super::vec_dot_q4_k_q8_k`]; `force_scalar` pins the
/// scalar path for A/B comparison.
pub fn vec_dot_q4_k_q8_k(
    weight_row: &[u8],
    acts: &[BlockQ8K],
    force_scalar: bool,
) -> Result<f32, KernelError> {
    #[cfg(target_arch = "x86_64")]
    if !force_scalar && avx2_fma_available() {
        validate_row(weight_row, QuantFormat::Q4_K, acts.len())?;
        // SAFETY: AVX2+FMA presence was checked at runtime just above.
        return Ok(unsafe { x86::vec_dot_q4_k_q8_k(weight_row, acts) });
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = force_scalar;
    dot::vec_dot_q4_k_q8_k(weight_row, acts)
}

/// Dot product of a packed Q5_K weight row with Q8_K activations,
/// dispatched to AVX2+FMA when available (else the scalar reference).
///
/// Same contract as [`super::vec_dot_q5_k_q8_k`]; `force_scalar` pins the
/// scalar path for A/B comparison.
pub fn vec_dot_q5_k_q8_k(
    weight_row: &[u8],
    acts: &[BlockQ8K],
    force_scalar: bool,
) -> Result<f32, KernelError> {
    #[cfg(target_arch = "x86_64")]
    if !force_scalar && avx2_fma_available() {
        validate_row(weight_row, QuantFormat::Q5_K, acts.len())?;
        // SAFETY: AVX2+FMA presence was checked at runtime just above.
        return Ok(unsafe { x86::vec_dot_q5_k_q8_k(weight_row, acts) });
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = force_scalar;
    dot::vec_dot_q5_k_q8_k(weight_row, acts)
}

/// Dot product of a packed Q6_K weight row with Q8_K activations,
/// dispatched to AVX2+FMA when available (else the scalar reference).
///
/// Same contract as [`super::vec_dot_q6_k_q8_k`]; `force_scalar` pins the
/// scalar path for A/B comparison.
pub fn vec_dot_q6_k_q8_k(
    weight_row: &[u8],
    acts: &[BlockQ8K],
    force_scalar: bool,
) -> Result<f32, KernelError> {
    #[cfg(target_arch = "x86_64")]
    if !force_scalar && avx2_fma_available() {
        validate_row(weight_row, QuantFormat::Q6_K, acts.len())?;
        // SAFETY: AVX2+FMA presence was checked at runtime just above.
        return Ok(unsafe { x86::vec_dot_q6_k_q8_k(weight_row, acts) });
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = force_scalar;
    dot::vec_dot_q6_k_q8_k(weight_row, acts)
}

/// Dot product of a packed Q8_0 weight row with Q8_0 activations,
/// dispatched to AVX2+FMA when available (else the scalar reference).
///
/// Same contract as [`super::vec_dot_q8_0_q8_0`]; `force_scalar` pins the
/// scalar path for A/B comparison.
pub fn vec_dot_q8_0_q8_0(
    weight_row: &[u8],
    acts: &[BlockQ8_0],
    force_scalar: bool,
) -> Result<f32, KernelError> {
    #[cfg(target_arch = "x86_64")]
    if !force_scalar && avx2_fma_available() {
        validate_row(weight_row, QuantFormat::Q8_0, acts.len())?;
        // SAFETY: AVX2+FMA presence was checked at runtime just above.
        return Ok(unsafe { x86::vec_dot_q8_0_q8_0(weight_row, acts) });
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = force_scalar;
    dot::vec_dot_q8_0_q8_0(weight_row, acts)
}

/// Quantize a row of f32 activations into Q8_0 blocks, dispatched to
/// AVX2+FMA when available (else the scalar reference).
///
/// Same contract as [`super::quantize_row_q8_0`], and byte-identical output
/// on every input: the SIMD path reproduces the reference's
/// half-away-from-zero rounding and its exact `1/d` scale math (deviating
/// from ggml's AVX2 quantizer, which ties to even — see the module docs).
/// `force_scalar` pins the scalar path for A/B comparison.
pub fn quantize_row_q8_0(
    x: &[f32],
    out: &mut [BlockQ8_0],
    force_scalar: bool,
) -> Result<(), KernelError> {
    #[cfg(target_arch = "x86_64")]
    if !force_scalar && avx2_fma_available() {
        validate_quantize(QuantFormat::Q8_0, x.len(), out.len())?;
        // SAFETY: AVX2+FMA presence was checked at runtime just above.
        unsafe { x86::quantize_row_q8_0(x, out) };
        return Ok(());
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = force_scalar;
    quantize::quantize_row_q8_0(x, out)
}

/// Quantize a row of f32 activations into Q8_K blocks, dispatched to
/// AVX2+FMA when available (else the scalar reference).
///
/// Same contract as [`super::quantize_row_q8_k`], and byte-identical output
/// on every input (`_mm256_cvtps_epi32` under default rounding is exactly
/// the reference's `round_ties_even`; `iscale`/`d` math stays in scalar f32
/// identical to the reference). `force_scalar` pins the scalar path for A/B
/// comparison.
pub fn quantize_row_q8_k(
    x: &[f32],
    out: &mut [BlockQ8K],
    force_scalar: bool,
) -> Result<(), KernelError> {
    #[cfg(target_arch = "x86_64")]
    if !force_scalar && avx2_fma_available() {
        validate_quantize(QuantFormat::Q8_K, x.len(), out.len())?;
        // SAFETY: AVX2+FMA presence was checked at runtime just above.
        unsafe { x86::quantize_row_q8_k(x, out) };
        return Ok(());
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = force_scalar;
    quantize::quantize_row_q8_k(x, out)
}

/// AVX2+FMA implementations. Every function in here is
/// `#[target_feature(enable = "avx2,fma")]` and must only be called after a
/// runtime feature check (the dispatch wrappers above are the only callers
/// besides tests, and both check).
#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::super::blocks::{QK_K, QK8_0, f16_at, q4k, q5k, q6k, q8_0, scales_at};
    use super::super::f16::{f16_to_f32, f32_to_f16};
    use super::super::{BlockQ8_0, BlockQ8K};

    /// Unpack the 12-byte k-quant scale field into eight 6-bit scales and
    /// eight 6-bit mins via ggml's three-word mask trick (`kmask1/2/3` in
    /// `ggml-cpu/arch/x86/quants.c`). Bit-equivalent to eight calls of
    /// `get_scale_min_k4` (tested).
    #[inline]
    pub(super) fn unpack_scales_mins(scales: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;
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

    /// Horizontal sum of the eight f32 lanes.
    ///
    /// Register-only intrinsics: safe under the enabled target features.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum_ps_8(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }

    /// Horizontal sum of the four f32 lanes.
    ///
    /// Register-only intrinsics: safe under the enabled target features.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum_ps_4(v: __m128) -> f32 {
        let s = _mm_add_ps(v, _mm_movehl_ps(v, v));
        let s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }

    /// Horizontal max of the eight f32 lanes.
    ///
    /// Register-only intrinsics: safe under the enabled target features.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hmax_ps_8(v: __m256) -> f32 {
        let m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        let m = _mm_max_ss(m, _mm_movehdup_ps(m));
        _mm_cvtss_f32(m)
    }

    /// Horizontal sum of the eight i32 lanes.
    ///
    /// Register-only intrinsics: safe under the enabled target features.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum_epi32_8(v: __m256i) -> i32 {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b0100_1110>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b1011_0001>(s));
        _mm_cvtsi128_si32(s)
    }

    /// Per-block min correction for q4_K/q5_K: `sum_g bsums[g] * mins[g/2]`
    /// as an `__m128i` of four partial i32 sums, computed as ggml does —
    /// `_mm_hadd_epi16` folds the sixteen per-16 `bsums` into eight per-32
    /// sums (exact: |bsum| <= 2048, pairs <= 4096, well inside i16), then
    /// `_mm_madd_epi16` multiplies by the 6-bit mins and widens to i32.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mins_dot(y: &BlockQ8K, mins: &[u8; 8]) -> __m128i {
        // SAFETY: `y.bsums` is 32 bytes and `mins` is 8 bytes, both read
        // with unaligned loads; everything else is register-only.
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

    /// AVX2 port of the scalar `vec_dot_q4_k_q8_k`. Caller must have
    /// validated `weight_row.len() == acts.len() * 144` and checked
    /// AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q4_k_q8_k(weight_row: &[u8], acts: &[BlockQ8K]) -> f32 {
        // SAFETY (whole body): caller guarantees AVX2+FMA and the length
        // contract, so every `add`/`loadu` below reads inside `block`
        // (144 B), `qs` (128 B at offset 16), or `y.qs` (256 B); loads are
        // unaligned by construction.
        unsafe {
            let m4 = _mm256_set1_epi8(0xF);
            let mut acc = _mm256_setzero_ps();
            let mut acc_m = _mm_setzero_ps();
            for (block, y) in weight_row.chunks_exact(q4k::BYTES).zip(acts) {
                let d = f16_at(block, q4k::D) * y.d;
                let dmin = f16_at(block, q4k::DMIN) * y.d;
                let (sc, mins) = unpack_scales_mins(scales_at(block, q4k::SCALES));

                // Min correction via bsums, accumulated separately so the
                // block loop stays integer-only until one fmadd each.
                let prod = mins_dot(y, &mins);
                acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                let qs = &block[q4k::QS..q4k::BYTES];
                let mut sumi = _mm256_setzero_si256();
                for j in 0..4 {
                    // 32 packed bytes = 64 weights: low nibbles are weights
                    // 64j..64j+32, high nibbles 64j+32..64j+64 (ggml order).
                    let q4bits = _mm256_loadu_si256(qs.as_ptr().add(32 * j).cast());
                    let q4l = _mm256_and_si256(q4bits, m4);
                    let q4h = _mm256_and_si256(_mm256_srli_epi16::<4>(q4bits), m4);
                    let q8l = _mm256_loadu_si256(y.qs.as_ptr().add(64 * j).cast());
                    let q8h = _mm256_loadu_si256(y.qs.as_ptr().add(64 * j + 32).cast());
                    // maddubs: unsigned nibbles x signed i8 -> i16 pair
                    // sums (max 2*15*128, no saturation); madd applies the
                    // 6-bit sub-block scale and widens to i32.
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

    /// AVX2 port of the scalar `vec_dot_q5_k_q8_k`. Caller must have
    /// validated `weight_row.len() == acts.len() * 176` and checked
    /// AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q5_k_q8_k(weight_row: &[u8], acts: &[BlockQ8K]) -> f32 {
        // SAFETY (whole body): caller guarantees AVX2+FMA and the length
        // contract, so every load reads inside `block` (176 B: qh 32 B at
        // 16, qs 128 B at 48) or `y.qs` (256 B); loads are unaligned by
        // construction.
        unsafe {
            let m4 = _mm256_set1_epi8(0xF);
            let mut acc = _mm256_setzero_ps();
            let mut acc_m = _mm_setzero_ps();
            for (block, y) in weight_row.chunks_exact(q5k::BYTES).zip(acts) {
                let d = f16_at(block, q5k::D) * y.d;
                let dmin = f16_at(block, q5k::DMIN) * y.d;
                let (sc, mins) = unpack_scales_mins(scales_at(block, q5k::SCALES));

                let prod = mins_dot(y, &mins);
                acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

                let qh = &block[q5k::QH..q5k::QS];
                let qs = &block[q5k::QS..q5k::BYTES];
                let hbits = _mm256_loadu_si256(qh.as_ptr().cast());
                let mut hmask = _mm256_set1_epi8(1);
                let mut sumi = _mm256_setzero_si256();
                for j in 0..4 {
                    let q5bits = _mm256_loadu_si256(qs.as_ptr().add(32 * j).cast());
                    // 5th bit for sub-block 2j: bit 2j of each qh byte,
                    // masked to a single bit per byte so the 16-lane shifts
                    // cannot smear across bytes, then placed at bit 4.
                    let h0 = _mm256_slli_epi16::<4>(_mm256_srl_epi16(
                        _mm256_and_si256(hbits, hmask),
                        _mm_cvtsi32_si128(2 * j as i32),
                    ));
                    let q5_0 = _mm256_add_epi8(_mm256_and_si256(q5bits, m4), h0);
                    hmask = _mm256_slli_epi16::<1>(hmask);
                    // 5th bit for sub-block 2j+1 (bit 2j+1).
                    let h1 = _mm256_slli_epi16::<4>(_mm256_srl_epi16(
                        _mm256_and_si256(hbits, hmask),
                        _mm_cvtsi32_si128(2 * j as i32 + 1),
                    ));
                    let q5_1 =
                        _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16::<4>(q5bits), m4), h1);
                    hmask = _mm256_slli_epi16::<1>(hmask);

                    let q8l = _mm256_loadu_si256(y.qs.as_ptr().add(64 * j).cast());
                    let q8h = _mm256_loadu_si256(y.qs.as_ptr().add(64 * j + 32).cast());
                    // maddubs: 5-bit unsigned (<= 31) x i8, max 2*31*128 —
                    // no i16 saturation.
                    let p16l = _mm256_madd_epi16(
                        _mm256_set1_epi16(i16::from(sc[2 * j])),
                        _mm256_maddubs_epi16(q5_0, q8l),
                    );
                    let p16h = _mm256_madd_epi16(
                        _mm256_set1_epi16(i16::from(sc[2 * j + 1])),
                        _mm256_maddubs_epi16(q5_1, q8h),
                    );
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum_ps_8(acc) - hsum_ps_4(acc_m)
        }
    }

    /// A 16-lane i16 vector holding `lo` in lanes 0..8 and `hi` in lanes
    /// 8..16 — the per-16-weight sub-block scale layout `madd_epi16` needs
    /// for one 32-weight vector.
    ///
    /// Register-only intrinsics: safe under the enabled target features.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn scale_pair(lo: i8, hi: i8) -> __m256i {
        _mm256_set_m128i(_mm_set1_epi16(i16::from(hi)), _mm_set1_epi16(i16::from(lo)))
    }

    /// AVX2 port of the scalar `vec_dot_q6_k_q8_k`. Caller must have
    /// validated `weight_row.len() == acts.len() * 210` and checked
    /// AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q6_k_q8_k(weight_row: &[u8], acts: &[BlockQ8K]) -> f32 {
        // SAFETY (whole body): caller guarantees AVX2+FMA and the length
        // contract, so every load reads inside `block` (210 B: ql 128 B at
        // 0, qh 64 B at 128, scales 16 B at 192) or `y.qs` (256 B); loads
        // are unaligned by construction.
        unsafe {
            let m4 = _mm256_set1_epi8(0xF);
            let m2 = _mm256_set1_epi8(3);
            let m32s = _mm256_set1_epi8(32);
            let mut acc = _mm256_setzero_ps();
            for (block, y) in weight_row.chunks_exact(q6k::BYTES).zip(acts) {
                let d = f16_at(block, q6k::D) * y.d;
                let scs = &block[q6k::SCALES..q6k::SCALES + 16];
                let mut sumi = _mm256_setzero_si256();
                for half in 0..2 {
                    let base = 128 * half; // weight offset of this half
                    let q4bits1 =
                        _mm256_loadu_si256(block.as_ptr().add(q6k::QL + 64 * half).cast());
                    let q4bits2 =
                        _mm256_loadu_si256(block.as_ptr().add(q6k::QL + 64 * half + 32).cast());
                    let bits_h = _mm256_loadu_si256(block.as_ptr().add(q6k::QH + 32 * half).cast());

                    // Two high bits per weight, placed at bits 4..6; OR'd
                    // onto the low nibbles this yields the biased 6-bit
                    // value in [0, 63].
                    let q4h_0 = _mm256_slli_epi16::<4>(_mm256_and_si256(bits_h, m2));
                    let q4h_1 = _mm256_slli_epi16::<4>(_mm256_and_si256(
                        _mm256_srli_epi16::<2>(bits_h),
                        m2,
                    ));
                    let q4h_2 = _mm256_slli_epi16::<4>(_mm256_and_si256(
                        _mm256_srli_epi16::<4>(bits_h),
                        m2,
                    ));
                    let q4h_3 = _mm256_slli_epi16::<4>(_mm256_and_si256(
                        _mm256_srli_epi16::<6>(bits_h),
                        m2,
                    ));

                    let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                    let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                    let q4_2 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(q4bits1), m4),
                        q4h_2,
                    );
                    let q4_3 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(q4bits2), m4),
                        q4h_3,
                    );

                    let q8_0 = _mm256_loadu_si256(y.qs.as_ptr().add(base).cast());
                    let q8_1 = _mm256_loadu_si256(y.qs.as_ptr().add(base + 32).cast());
                    let q8_2 = _mm256_loadu_si256(y.qs.as_ptr().add(base + 64).cast());
                    let q8_3 = _mm256_loadu_si256(y.qs.as_ptr().add(base + 96).cast());

                    // (q - 32) * a as i16 pair sums: maddubs(q, a) (max
                    // 2*63*128) minus maddubs(32, a) (max 2*32*128); the
                    // difference stays within +-24576, no i16 overflow.
                    let p16_0 = _mm256_sub_epi16(
                        _mm256_maddubs_epi16(q4_0, q8_0),
                        _mm256_maddubs_epi16(m32s, q8_0),
                    );
                    let p16_1 = _mm256_sub_epi16(
                        _mm256_maddubs_epi16(q4_1, q8_1),
                        _mm256_maddubs_epi16(m32s, q8_1),
                    );
                    let p16_2 = _mm256_sub_epi16(
                        _mm256_maddubs_epi16(q4_2, q8_2),
                        _mm256_maddubs_epi16(m32s, q8_2),
                    );
                    let p16_3 = _mm256_sub_epi16(
                        _mm256_maddubs_epi16(q4_3, q8_3),
                        _mm256_maddubs_epi16(m32s, q8_3),
                    );

                    // Apply the signed i8 sub-block scales (two per
                    // 32-weight vector) and widen to i32.
                    let s = 8 * half;
                    let p32_0 =
                        _mm256_madd_epi16(scale_pair(scs[s] as i8, scs[s + 1] as i8), p16_0);
                    let p32_1 =
                        _mm256_madd_epi16(scale_pair(scs[s + 2] as i8, scs[s + 3] as i8), p16_1);
                    let p32_2 =
                        _mm256_madd_epi16(scale_pair(scs[s + 4] as i8, scs[s + 5] as i8), p16_2);
                    let p32_3 =
                        _mm256_madd_epi16(scale_pair(scs[s + 6] as i8, scs[s + 7] as i8), p16_3);

                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p32_0, p32_1));
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p32_2, p32_3));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum_ps_8(acc)
        }
    }

    /// AVX2 port of the scalar `vec_dot_q8_0_q8_0`. Caller must have
    /// validated `weight_row.len() == acts.len() * 34` and checked
    /// AVX2+FMA.
    ///
    /// The signed x signed product runs through ggml's `sign` trick:
    /// `|x| * sign(y, x)` feeds `maddubs` with the weight as the unsigned
    /// operand. A weight byte of -128 is handled (|x| = 128 fits the
    /// unsigned operand); the activation side relies on the
    /// [`BlockQ8_0::qs`] contract of values in [-127, 127].
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vec_dot_q8_0_q8_0(weight_row: &[u8], acts: &[BlockQ8_0]) -> f32 {
        // SAFETY (whole body): caller guarantees AVX2+FMA and the length
        // contract, so the loads read the 32 `qs` bytes at offset 2 of each
        // 34-byte block and the 32 bytes of `y.qs`; loads are unaligned by
        // construction.
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            for (block, y) in weight_row.chunks_exact(q8_0::BYTES).zip(acts) {
                let d = f16_at(block, q8_0::D) * y.d;
                let qx = _mm256_loadu_si256(block.as_ptr().add(q8_0::QS).cast());
                let qy = _mm256_loadu_si256(y.qs.as_ptr().cast());
                let ax = _mm256_sign_epi8(qx, qx);
                let sy = _mm256_sign_epi8(qy, qx);
                // Pair sums max 2*128*127 = 32512: no i16 saturation.
                let dot16 = _mm256_maddubs_epi16(ax, sy);
                let sum32 = _mm256_madd_epi16(ones, dot16);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum32), acc);
            }
            hsum_ps_8(acc)
        }
    }

    /// Pack four vectors of 8 i32 (all within [-128, 127]) into 32 i8 in
    /// value order and store them unaligned at `dst`.
    ///
    /// # Safety
    ///
    /// Caller guarantees AVX2 and that `dst` is valid for a 32-byte write.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn pack_store_i8(dst: *mut i8, i0: __m256i, i1: __m256i, i2: __m256i, i3: __m256i) {
        // SAFETY: register-only packing; the store contract is the
        // function's documented safety requirement. Saturation in the packs
        // cannot trigger for values in [-128, 127]. The permutation is
        // ggml's `perm` from its AVX2 `quantize_row_q8_0`: it restores
        // value order after the two lane-interleaving `packs` steps.
        unsafe {
            let p01 = _mm256_packs_epi32(i0, i1);
            let p23 = _mm256_packs_epi32(i2, i3);
            let p = _mm256_packs_epi16(p01, p23);
            let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            let p = _mm256_permutevar8x32_epi32(p, perm);
            _mm256_storeu_si256(dst.cast(), p);
        }
    }

    /// Round eight lanes half away from zero to integral floats, exactly
    /// matching `f32::round` for |x| < 2^23.
    ///
    /// `_mm256_round_ps` only offers ties-to-even, so ties are fixed up:
    /// `x - rte(x)` is exact (Sterbenz), equals +-0.5 exactly at a true
    /// tie, and the tie was rounded toward zero iff that difference has the
    /// sign of `x` — bump those lanes one further away.
    /// Register-only intrinsics: safe under the enabled target features.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn round_away_ps(x: __m256) -> __m256 {
        let t = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(x);
        let diff = _mm256_sub_ps(x, t);
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        let tie_up = _mm256_and_ps(
            _mm256_cmp_ps::<_CMP_EQ_OQ>(diff, _mm256_set1_ps(0.5)),
            _mm256_cmp_ps::<_CMP_GT_OQ>(x, zero),
        );
        let tie_dn = _mm256_and_ps(
            _mm256_cmp_ps::<_CMP_EQ_OQ>(diff, _mm256_set1_ps(-0.5)),
            _mm256_cmp_ps::<_CMP_LT_OQ>(x, zero),
        );
        let t = _mm256_add_ps(t, _mm256_and_ps(tie_up, one));
        _mm256_sub_ps(t, _mm256_and_ps(tie_dn, one))
    }

    /// AVX2 port of the scalar `quantize_row_q8_0`, byte-identical to it.
    /// Caller must have validated `x.len() == out.len() * 32` and checked
    /// AVX2+FMA.
    ///
    /// The scale math (`d = amax / 127`, `id = 1/d`, f16 rounding of `d`)
    /// runs in scalar f32 exactly as the reference does; `amax` via SIMD max
    /// is exact because IEEE max is order-independent for finite inputs.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn quantize_row_q8_0(x: &[f32], out: &mut [BlockQ8_0]) {
        // SAFETY (whole body): caller guarantees AVX2+FMA and the length
        // contract; each iteration reads 32 f32 from `xs` and writes the 32
        // `qs` bytes of one block, all in bounds, all unaligned ops.
        unsafe {
            let sign_mask = _mm256_set1_ps(-0.0);
            for (xs, block) in x.chunks_exact(QK8_0).zip(out.iter_mut()) {
                let v0 = _mm256_loadu_ps(xs.as_ptr());
                let v1 = _mm256_loadu_ps(xs.as_ptr().add(8));
                let v2 = _mm256_loadu_ps(xs.as_ptr().add(16));
                let v3 = _mm256_loadu_ps(xs.as_ptr().add(24));
                let a01 = _mm256_max_ps(
                    _mm256_andnot_ps(sign_mask, v0),
                    _mm256_andnot_ps(sign_mask, v1),
                );
                let a23 = _mm256_max_ps(
                    _mm256_andnot_ps(sign_mask, v2),
                    _mm256_andnot_ps(sign_mask, v3),
                );
                let amax = hmax_ps_8(_mm256_max_ps(a01, a23));

                // Scalar scale math, identical to the reference.
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                block.d = f16_to_f32(f32_to_f16(d));

                let idv = _mm256_set1_ps(id);
                let i0 = _mm256_cvtps_epi32(round_away_ps(_mm256_mul_ps(v0, idv)));
                let i1 = _mm256_cvtps_epi32(round_away_ps(_mm256_mul_ps(v1, idv)));
                let i2 = _mm256_cvtps_epi32(round_away_ps(_mm256_mul_ps(v2, idv)));
                let i3 = _mm256_cvtps_epi32(round_away_ps(_mm256_mul_ps(v3, idv)));
                pack_store_i8(block.qs.as_mut_ptr(), i0, i1, i2, i3);
            }
        }
    }

    /// AVX2 port of the scalar `quantize_row_q8_k`, byte-identical to it.
    /// Caller must have validated `x.len() == out.len() * 256` and checked
    /// AVX2+FMA.
    ///
    /// The signed extreme is recovered exactly as the scalar loop selects
    /// it: SIMD gives `amax` (order-independent), then the *first* element
    /// with `|v| == amax` is the one whose strict-greater update set the
    /// scalar loop's final `max`. `iscale`/`d` math stays in scalar f32;
    /// `_mm256_cvtps_epi32` under the default MXCSR rounding is exactly
    /// `round_ties_even`, and the `packs` saturation to [-128, 127] is
    /// exactly the reference's clamp (unreachable anyway for |q| <= 127).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn quantize_row_q8_k(x: &[f32], out: &mut [BlockQ8K]) {
        // SAFETY (whole body): caller guarantees AVX2+FMA and the length
        // contract; each iteration reads 256 f32 from `xs` and writes one
        // block's `qs`/`bsums`, all in bounds, all unaligned ops.
        unsafe {
            let sign_mask = _mm256_set1_ps(-0.0);
            for (xs, block) in x.chunks_exact(QK_K).zip(out.iter_mut()) {
                let mut amax_v = _mm256_setzero_ps();
                for k in 0..QK_K / 8 {
                    let v = _mm256_loadu_ps(xs.as_ptr().add(8 * k));
                    amax_v = _mm256_max_ps(amax_v, _mm256_andnot_ps(sign_mask, v));
                }
                let amax = hmax_ps_8(amax_v);
                if amax == 0.0 {
                    *block = BlockQ8K::default();
                    continue;
                }

                // First element whose |v| equals amax == the element the
                // scalar strict-greater scan settled on.
                let target = _mm256_set1_ps(amax);
                let mut max = 0.0f32;
                for k in 0..QK_K / 8 {
                    let v = _mm256_loadu_ps(xs.as_ptr().add(8 * k));
                    let eq = _mm256_cmp_ps::<_CMP_EQ_OQ>(_mm256_andnot_ps(sign_mask, v), target);
                    let m = _mm256_movemask_ps(eq);
                    if m != 0 {
                        max = xs[8 * k + m.trailing_zeros() as usize];
                        break;
                    }
                }

                let iscale = -127.0 / max;
                let iv = _mm256_set1_ps(iscale);
                for c in 0..QK_K / 32 {
                    let base = 32 * c;
                    let i0 = _mm256_cvtps_epi32(_mm256_mul_ps(
                        _mm256_loadu_ps(xs.as_ptr().add(base)),
                        iv,
                    ));
                    let i1 = _mm256_cvtps_epi32(_mm256_mul_ps(
                        _mm256_loadu_ps(xs.as_ptr().add(base + 8)),
                        iv,
                    ));
                    let i2 = _mm256_cvtps_epi32(_mm256_mul_ps(
                        _mm256_loadu_ps(xs.as_ptr().add(base + 16)),
                        iv,
                    ));
                    let i3 = _mm256_cvtps_epi32(_mm256_mul_ps(
                        _mm256_loadu_ps(xs.as_ptr().add(base + 24)),
                        iv,
                    ));
                    // Per-16 group sums from the i32 lanes (exact; order
                    // does not matter in integers). Sum of 16 i8 fits i16.
                    block.bsums[2 * c] = hsum_epi32_8(_mm256_add_epi32(i0, i1)) as i16;
                    block.bsums[2 * c + 1] = hsum_epi32_8(_mm256_add_epi32(i2, i3)) as i16;
                    pack_store_i8(block.qs.as_mut_ptr().add(base), i0, i1, i2, i3);
                }
                block.d = 1.0 / iscale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        Lcg, quantize_row_q4_k_test, quantize_row_q5_k_test, quantize_row_q6_k_test,
        quantize_row_q8_0_test,
    };
    use super::super::{
        dequantize_row_q4_k, dequantize_row_q5_k, dequantize_row_q6_k, dequantize_row_q8_0,
        f32_to_f16, quantize,
    };
    use super::*;

    /// The dims the AVX2 ports must match scalar on (audited in-dims).
    const DIMS: [usize; 3] = [2048, 768, 4096];

    /// A dispatched k-quant dot entry point.
    type DispatchDot = fn(&[u8], &[BlockQ8K], bool) -> Result<f32, KernelError>;

    /// Assert the AVX2 and scalar dots agree within 1e-6 relative to the
    /// sum of |w_i * a_i| (they share exact integer structure; only float
    /// summation order differs).
    fn assert_close(avx2: f32, scalar: f32, w: &[f32], a: &[f32]) {
        assert_eq!(w.len(), a.len());
        let mag: f64 = w
            .iter()
            .zip(a)
            .map(|(&wi, &ai)| (f64::from(wi) * f64::from(ai)).abs())
            .sum();
        let tol = 1e-6 * mag.max(1e-20);
        let err = (f64::from(avx2) - f64::from(scalar)).abs();
        assert!(
            err <= tol,
            "avx2 {avx2} vs scalar {scalar}: err {err} > tol {tol} (mag {mag})"
        );
    }

    fn q8_k_acts(x: &[f32]) -> Vec<BlockQ8K> {
        let mut acts = vec![BlockQ8K::default(); x.len() / 256];
        quantize::quantize_row_q8_k(x, &mut acts).unwrap();
        acts
    }

    fn dequant_q8_k(blocks: &[BlockQ8K]) -> Vec<f32> {
        blocks
            .iter()
            .flat_map(|b| b.qs.iter().map(move |&q| b.d * f32::from(q)))
            .collect()
    }

    fn dequant_q8_0_act(blocks: &[BlockQ8_0]) -> Vec<f32> {
        blocks
            .iter()
            .flat_map(|b| b.qs.iter().map(move |&q| b.d * f32::from(q)))
            .collect()
    }

    /// Compare one k-quant kernel against scalar across the audited dims.
    fn k_quant_avx2_vs_scalar(
        quant: fn(&[f32]) -> Vec<u8>,
        dequant: fn(&[u8]) -> Result<Vec<f32>, KernelError>,
        dot: DispatchDot,
        seed: u64,
    ) {
        if !avx2_fma_available() {
            return; // Nothing to compare on this machine.
        }
        for n in DIMS {
            let mut rng = Lcg(seed ^ n as u64);
            let xw: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0).collect();
            let xa: Vec<f32> = (0..n).map(|_| rng.next_f32() * 3.0).collect();
            let wb = quant(&xw);
            let acts = q8_k_acts(&xa);
            let scalar = dot(&wb, &acts, true).unwrap();
            let fast = dot(&wb, &acts, false).unwrap();
            assert_close(fast, scalar, &dequant(&wb).unwrap(), &dequant_q8_k(&acts));
        }
    }

    #[test]
    fn avx2_vs_scalar_q4_k() {
        k_quant_avx2_vs_scalar(
            quantize_row_q4_k_test,
            dequantize_row_q4_k,
            vec_dot_q4_k_q8_k,
            0xB4,
        );
    }

    #[test]
    fn avx2_vs_scalar_q5_k() {
        k_quant_avx2_vs_scalar(
            quantize_row_q5_k_test,
            dequantize_row_q5_k,
            vec_dot_q5_k_q8_k,
            0xB5,
        );
    }

    #[test]
    fn avx2_vs_scalar_q6_k() {
        k_quant_avx2_vs_scalar(
            quantize_row_q6_k_test,
            dequantize_row_q6_k,
            vec_dot_q6_k_q8_k,
            0xB6,
        );
    }

    #[test]
    fn avx2_vs_scalar_q8_0() {
        if !avx2_fma_available() {
            return;
        }
        for n in DIMS {
            let mut rng = Lcg(0xB8 ^ n as u64);
            let xw: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0).collect();
            let xa: Vec<f32> = (0..n).map(|_| rng.next_f32() * 3.0).collect();
            let wb = quantize_row_q8_0_test(&xw);
            let mut acts = vec![BlockQ8_0::default(); n / 32];
            quantize::quantize_row_q8_0(&xa, &mut acts).unwrap();
            let scalar = vec_dot_q8_0_q8_0(&wb, &acts, true).unwrap();
            let fast = vec_dot_q8_0_q8_0(&wb, &acts, false).unwrap();
            assert_close(
                fast,
                scalar,
                &dequantize_row_q8_0(&wb).unwrap(),
                &dequant_q8_0_act(&acts),
            );
        }
    }

    #[test]
    fn force_scalar_is_bitwise_scalar() {
        let mut rng = Lcg(0xF5);
        let xw: Vec<f32> = (0..768).map(|_| rng.next_f32() * 4.0).collect();
        let xa: Vec<f32> = (0..768).map(|_| rng.next_f32() * 3.0).collect();
        let wb = quantize_row_q4_k_test(&xw);
        let acts = q8_k_acts(&xa);
        let forced = vec_dot_q4_k_q8_k(&wb, &acts, true).unwrap();
        let reference = dot::vec_dot_q4_k_q8_k(&wb, &acts).unwrap();
        assert_eq!(forced.to_bits(), reference.to_bits());
    }

    // --------------------------------------------------- adversarial blocks
    // Same raw-block cases as the scalar suite (max/zero scales, all-min).

    fn build_q4k(d: f32, dmin: f32, scales: [u8; 12], qs: u8) -> Vec<u8> {
        let mut b = Vec::with_capacity(144);
        b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        b.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        b.extend_from_slice(&scales);
        b.extend_from_slice(&[qs; 128]);
        b
    }

    fn build_q5k(d: f32, dmin: f32, scales: [u8; 12], qh: u8, qs: u8) -> Vec<u8> {
        let mut b = Vec::with_capacity(176);
        b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        b.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        b.extend_from_slice(&scales);
        b.extend_from_slice(&[qh; 32]);
        b.extend_from_slice(&[qs; 128]);
        b
    }

    fn build_q6k(ql: u8, qh: u8, scale: i8, d: f32) -> Vec<u8> {
        let mut b = Vec::with_capacity(210);
        b.extend_from_slice(&[ql; 128]);
        b.extend_from_slice(&[qh; 64]);
        b.extend_from_slice(&[scale as u8; 16]);
        b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        b
    }

    /// Scales field encoding sc = 0, m = 63 for all eight sub-blocks.
    const ALL_MIN_SCALES: [u8; 12] = [0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xF0, 0xF0, 0xF0, 0xF0];

    fn random_q8_k_block(seed: u64) -> Vec<BlockQ8K> {
        let mut rng = Lcg(seed);
        let xa: Vec<f32> = (0..256).map(|_| rng.next_f32() * 2.0).collect();
        q8_k_acts(&xa)
    }

    #[test]
    fn adversarial_q4_k_matches_scalar() {
        if !avx2_fma_available() {
            return;
        }
        let acts = random_q8_k_block(0xE4);
        let a = dequant_q8_k(&acts);
        let cases = [
            build_q4k(1.0, 0.5, [0xFF; 12], 0xFF),
            build_q4k(4.0, 2.0, [0; 12], 0x5A),
            build_q4k(0.0, 0.0, [0x3F; 12], 0xFF),
            build_q4k(1.0, 0.5, ALL_MIN_SCALES, 0x00),
        ];
        for bytes in &cases {
            let scalar = vec_dot_q4_k_q8_k(bytes, &acts, true).unwrap();
            let fast = vec_dot_q4_k_q8_k(bytes, &acts, false).unwrap();
            assert_close(fast, scalar, &dequantize_row_q4_k(bytes).unwrap(), &a);
        }
        // Zero-scale block against zero activations: exactly zero on both.
        let zero_acts = vec![BlockQ8K::default()];
        assert_eq!(
            vec_dot_q4_k_q8_k(&cases[1], &zero_acts, false).unwrap(),
            0.0
        );
    }

    #[test]
    fn adversarial_q5_k_matches_scalar() {
        if !avx2_fma_available() {
            return;
        }
        let acts = random_q8_k_block(0xE5);
        let a = dequant_q8_k(&acts);
        let cases = [
            build_q5k(1.0, 0.5, [0xFF; 12], 0xFF, 0xFF),
            build_q5k(1.0, 0.5, [0xFF; 12], 0x00, 0xFF),
            build_q5k(4.0, 2.0, [0; 12], 0xAA, 0x5A),
            build_q5k(1.0, 0.5, ALL_MIN_SCALES, 0xAA, 0x00),
        ];
        for bytes in &cases {
            let scalar = vec_dot_q5_k_q8_k(bytes, &acts, true).unwrap();
            let fast = vec_dot_q5_k_q8_k(bytes, &acts, false).unwrap();
            assert_close(fast, scalar, &dequantize_row_q5_k(bytes).unwrap(), &a);
        }
    }

    #[test]
    fn adversarial_q6_k_matches_scalar() {
        if !avx2_fma_available() {
            return;
        }
        let acts = random_q8_k_block(0xE6);
        let a = dequant_q8_k(&acts);
        let cases = [
            build_q6k(0xFF, 0xFF, -128, 1.0),
            build_q6k(0xFF, 0xFF, 127, 0.25),
            build_q6k(0x00, 0x00, 127, 1.0),
            build_q6k(0x5A, 0xA5, 0, 4.0),
        ];
        for bytes in &cases {
            let scalar = vec_dot_q6_k_q8_k(bytes, &acts, true).unwrap();
            let fast = vec_dot_q6_k_q8_k(bytes, &acts, false).unwrap();
            assert_close(fast, scalar, &dequantize_row_q6_k(bytes).unwrap(), &a);
        }
    }

    #[test]
    fn adversarial_q8_0_matches_scalar() {
        if !avx2_fma_available() {
            return;
        }
        // Max-magnitude f16 scale, weight bytes of -128, activations at
        // the +-127 contract edge.
        let mut wb = Vec::new();
        wb.extend_from_slice(&0x7BFFu16.to_le_bytes()); // d = 65504.0
        wb.extend_from_slice(&[0x80; 32]); // q = -128
        let acts = [BlockQ8_0 {
            d: 65504.0,
            qs: [127; 32],
        }];
        let scalar = vec_dot_q8_0_q8_0(&wb, &acts, true).unwrap();
        let fast = vec_dot_q8_0_q8_0(&wb, &acts, false).unwrap();
        assert_close(
            fast,
            scalar,
            &dequantize_row_q8_0(&wb).unwrap(),
            &dequant_q8_0_act(&acts),
        );
        assert_eq!(
            vec_dot_q8_0_q8_0(&wb, &[BlockQ8_0::default()], false).unwrap(),
            0.0
        );
    }

    // ---------------------------------------------------------- quantizers

    /// Inputs that exercise rounding ties, zero blocks, flat blocks, and
    /// the sign of the extreme.
    fn quantizer_inputs(n: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg(seed);
        let mut cases = vec![
            (0..n).map(|_| rng.next_f32() * 8.0).collect::<Vec<f32>>(),
            vec![0.0; n],
            vec![-3.75; n],
            vec![12.5; n],
        ];
        // Tie-heavy: amax 2.0 with +-1.0 elsewhere makes v*id land on
        // exact .5 boundaries (the scalar suite's q8_0 case, row-wide).
        let mut tie = vec![0.0f32; n];
        for (i, v) in tie.iter_mut().enumerate() {
            *v = match i % 4 {
                0 => 2.0,
                1 => -1.0,
                2 => 0.5,
                _ => -0.5,
            };
        }
        cases.push(tie);
        // Negative extreme first, positive extreme first.
        let mut neg = vec![0.5f32; n];
        neg[0] = -4.0;
        cases.push(neg);
        let mut pos = vec![-0.5f32; n];
        pos[1] = 4.0;
        cases.push(pos);
        // Duplicate extremes with opposite signs: the first one must win.
        let mut dup = vec![0.25f32; n];
        dup[3] = -6.0;
        dup[7] = 6.0;
        cases.push(dup);
        cases
    }

    #[test]
    fn quantize_q8_0_bytes_identical_to_scalar() {
        if !avx2_fma_available() {
            return;
        }
        for n in DIMS {
            for x in quantizer_inputs(n, 0xC0 ^ n as u64) {
                let mut scalar = vec![BlockQ8_0::default(); n / 32];
                quantize::quantize_row_q8_0(&x, &mut scalar).unwrap();
                let mut fast = vec![BlockQ8_0::default(); n / 32];
                quantize_row_q8_0(&x, &mut fast, false).unwrap();
                for (s, f) in scalar.iter().zip(&fast) {
                    assert_eq!(s.d.to_bits(), f.d.to_bits());
                    assert_eq!(s.qs, f.qs);
                }
            }
        }
    }

    #[test]
    fn quantize_q8_k_bytes_identical_to_scalar() {
        if !avx2_fma_available() {
            return;
        }
        for n in DIMS {
            for x in quantizer_inputs(n, 0xC8 ^ n as u64) {
                let mut scalar = vec![BlockQ8K::default(); n / 256];
                quantize::quantize_row_q8_k(&x, &mut scalar).unwrap();
                let mut fast = vec![BlockQ8K::default(); n / 256];
                quantize_row_q8_k(&x, &mut fast, false).unwrap();
                for (s, f) in scalar.iter().zip(&fast) {
                    assert_eq!(s.d.to_bits(), f.d.to_bits());
                    assert_eq!(s.qs, f.qs);
                    assert_eq!(s.bsums, f.bsums);
                }
            }
        }
    }

    // ----------------------------------------------- alignment & validation

    #[test]
    fn misaligned_weight_row_matches() {
        if !avx2_fma_available() {
            return;
        }
        let mut rng = Lcg(0xA11);
        let xw: Vec<f32> = (0..2048).map(|_| rng.next_f32() * 4.0).collect();
        let xa: Vec<f32> = (0..2048).map(|_| rng.next_f32() * 3.0).collect();
        let acts = q8_k_acts(&xa);
        for (wb, dot) in [
            (
                quantize_row_q4_k_test(&xw),
                vec_dot_q4_k_q8_k as DispatchDot,
            ),
            (quantize_row_q5_k_test(&xw), vec_dot_q5_k_q8_k),
            (quantize_row_q6_k_test(&xw), vec_dot_q6_k_q8_k),
        ] {
            let aligned = dot(&wb, &acts, false).unwrap();
            // Force a 1-byte-misaligned view of the same bytes.
            let mut shifted = vec![0u8; wb.len() + 1];
            shifted[1..].copy_from_slice(&wb);
            let misaligned = dot(&shifted[1..], &acts, false).unwrap();
            assert_eq!(aligned.to_bits(), misaligned.to_bits());
        }
        // q8_0 path (distinct loads).
        let wb = quantize_row_q8_0_test(&xw);
        let mut acts0 = vec![BlockQ8_0::default(); 2048 / 32];
        quantize::quantize_row_q8_0(&xa, &mut acts0).unwrap();
        let aligned = vec_dot_q8_0_q8_0(&wb, &acts0, false).unwrap();
        let mut shifted = vec![0u8; wb.len() + 1];
        shifted[1..].copy_from_slice(&wb);
        let misaligned = vec_dot_q8_0_q8_0(&shifted[1..], &acts0, false).unwrap();
        assert_eq!(aligned.to_bits(), misaligned.to_bits());
    }

    #[test]
    fn dispatch_validates_like_scalar() {
        let acts2 = vec![BlockQ8K::default(); 2];
        assert_eq!(
            vec_dot_q4_k_q8_k(&[0; 144], &acts2, false).unwrap_err(),
            KernelError::BlockCountMismatch {
                weight_blocks: 1,
                activation_blocks: 2,
            },
        );
        assert_eq!(
            vec_dot_q5_k_q8_k(&[0; 175], &acts2, false).unwrap_err(),
            KernelError::RowBytesNotBlockMultiple {
                format: QuantFormat::Q5_K,
                len: 175,
                block_bytes: 176,
            },
        );
        assert_eq!(
            vec_dot_q6_k_q8_k(&[0; 630], &acts2, false).unwrap_err(),
            KernelError::BlockCountMismatch {
                weight_blocks: 3,
                activation_blocks: 2,
            },
        );
        assert_eq!(
            vec_dot_q8_0_q8_0(&[0; 35], &[BlockQ8_0::default(); 8], false).unwrap_err(),
            KernelError::RowBytesNotBlockMultiple {
                format: QuantFormat::Q8_0,
                len: 35,
                block_bytes: 34,
            },
        );
        let x = [0f32; 100];
        assert_eq!(
            quantize_row_q8_0(&x, &mut [BlockQ8_0::default(); 3], false).unwrap_err(),
            KernelError::QuantizeLenMismatch {
                format: QuantFormat::Q8_0,
                floats: 100,
                out_blocks: 3,
            }
        );
        assert_eq!(
            quantize_row_q8_k(&x, &mut [BlockQ8K::default()], false).unwrap_err(),
            KernelError::QuantizeLenMismatch {
                format: QuantFormat::Q8_K,
                floats: 100,
                out_blocks: 1,
            }
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn scale_unpack_matches_get_scale_min_k4() {
        use super::super::blocks::get_scale_min_k4;
        let mut rng = Lcg(0x5CA1E5);
        for _ in 0..64 {
            let mut scales = [0u8; 12];
            for b in &mut scales {
                *b = (rng.next_f32() * 128.0 + 128.0) as u8;
            }
            let (sc, mins) = super::x86::unpack_scales_mins(&scales);
            for j in 0..8 {
                let (s, m) = get_scale_min_k4(j, &scales);
                assert_eq!(sc[j], s, "scale {j} of {scales:?}");
                assert_eq!(mins[j], m, "min {j} of {scales:?}");
            }
        }
    }
}
