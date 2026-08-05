//! AVX2 + F16C bodies for the GQA attention kernel.
//!
//! Reached only through the dispatch in the parent module, which checks
//! `avx2` and `f16c` at runtime (they are separate CPUID bits, and F16C is
//! *not* covered by `quants::avx2::avx2_fma_available`) and bounds `head_dim`
//! against [`super::MAX_SIMD_HEAD_DIM`]. Every entry point here is
//! `#[target_feature(enable = "avx2,f16c")] unsafe fn`.
//!
//! # Bit-neutrality
//!
//! This module produces **the same f32 bits** as the scalar reference in the
//! parent module, on every input, with zero tolerance
//! (`simd_matches_scalar_bit_for_bit`, and transitively
//! `restructured_kernel_is_bit_identical_to_head_major_reference`, which pins
//! both against the pre-wave-1 head-major nest). Three separate arguments,
//! one per piece:
//!
//! **(a) f16 to f32.** `vcvtph2ps` (`_mm256_cvtph_ps`) implements the same
//! widening as [`f16_to_f32`]. Every f16 — normal, subnormal, signed zero,
//! infinity — is exactly representable in f32, so the conversion is exact and
//! there is nothing to round; MXCSR's rounding mode cannot apply to an exact
//! operation, and the instruction does not flush subnormal inputs. Proven by
//! sweeping **all 65 536 f16 bit patterns** in
//! `cvtph_matches_scalar_on_every_non_nan_pattern`. The one scoped exception
//! is a **signalling** NaN: [`f16_to_f32`] reproduces the payload without
//! forcing the quiet bit while `vcvtph2ps` quiets it. That is scoped exactly
//! as `quants::avx2` scopes the analogous quantizer case — a non-finite value
//! in the KV cache means the pass already failed upstream — and NaN patterns
//! are excluded from the sweep's bit comparison (they are still asserted to
//! come back as NaN).
//!
//! **(b) The QK dot.** The scalar order is one f32 accumulator per
//! `(query head, position)` walking `i = 0..head_dim` **ascending**, with a
//! separate multiply and add. Eight lanes over `i` would reassociate that
//! chain into eight partials plus a horizontal tree — different bits, so it
//! is not done here. The lanes run over the **GQA group** instead, an axis
//! that is already independent: the group's queries are transposed once per
//! kv head into `[i][lane]`, and lane `g` then accumulates
//! `sum_i qt[i][g] * kb[i]` over `i` ascending in its own f32, one
//! `_mm256_mul_ps` and one `_mm256_add_ps` per `i` — the same two roundings
//! in the same order as `acc += qv * kv`, per lane. Positions are blocked by
//! [`T_BLOCK`] purely to run that many *independent* accumulator chains and
//! fill the vector add latency; each `(position, lane)` pair still keeps a
//! single chain over `i`. `scale` is applied once at the end, to the finished
//! accumulator, never folded per element.
//!
//! **(c) The V reduction.** `out[h][i] += w * v[i]` accumulates over `t`
//! ascending, per output element. `i` is a pure elementwise axis, so eight
//! lanes over `i` is exact; `t` is the reduction axis and stays strictly
//! sequential — the eight-lane accumulator is loaded from and stored back to
//! `out` on every `t`, so element `i` sees exactly the scalar sequence
//! `o = o + w_t * v_t`.
//!
//! **No FMA anywhere in this module.** `acc += qv * kv` and `*o += w * vv`
//! are a multiply *and* an add — two roundings. `_mm256_fmadd_ps` and
//! `f32::mul_add` collapse them into one and change the result bits, so they
//! are forbidden here even though the host has FMA and even though the
//! instruction is otherwise the obvious one to reach for. `f16c` is enabled
//! alongside `avx2`; `fma` deliberately is not.
//!
//! There is no online, streaming or flash-style rescaled softmax here either,
//! for the same reason and the one recorded in the parent module: the causal
//! limit is a *length*, and masked positions are absent from the sum rather
//! than zero-weighted.
//!
//! # Alignment
//!
//! Nothing beyond 2 bytes is assumed. The K/V planes are sub-slices of the
//! cache's `Vec<u16>` and a head slice sits at a `kv_head * head_dim` element
//! offset, so with an odd `head_dim` a row's head slice is only 2-byte
//! aligned; the f32 buffers (`out`, the conversion row, the score runs) are
//! only 4-byte aligned for the same reason. Every load and store here is
//! therefore an unaligned form — `_mm_loadu_si128`, `_mm256_loadu_ps`,
//! `_mm256_storeu_ps` — and the claim is tested three ways: `widen` against a
//! deliberately shifted view, `qk_scores` against a shifted plane, and
//! end-to-end geometries with `head_dim = 9` in the parent module's
//! `simd_matches_scalar_bit_for_bit` and
//! `misaligned_head_slices_match_the_scalar_reference`.

use std::arch::x86_64::*;

use super::MAX_SIMD_HEAD_DIM;
use crate::kernels::quants::f16_to_f32;

/// Query heads dotted per AVX2 vector: one lane of the group per lane of the
/// register. Fixed by the register width (8 f32), not tunable.
pub(super) const LANES: usize = 8;

/// Cached positions converted and dotted per block.
///
/// Different positions are independent accumulators, so a block of four runs
/// four independent dependency chains over `i` and fills the vector add's
/// ~4-cycle latency. It does **not** reassociate anything: chain `j` is
/// position `t + j`'s own accumulator, still walking `i` ascending.
pub(super) const T_BLOCK: usize = 4;

/// Widen f16 bit patterns into f32, eight at a time.
///
/// Bit-identical to [`f16_to_f32`] elementwise on every non-NaN pattern (see
/// the module docs on signalling NaN). The tail below eight elements uses
/// [`f16_to_f32`] itself.
///
/// # Safety
///
/// Requires AVX2 + F16C. Both slices are read and written unaligned, so any
/// address is fine; the shorter length wins, so no bound is assumed.
#[target_feature(enable = "avx2,f16c")]
unsafe fn widen(src: &[u16], dst: &mut [f32]) {
    // SAFETY: `i + LANES <= n <= src.len()` and `n <= dst.len()` on every
    // iteration of the wide loop, so the 16-byte load and the 32-byte store
    // both stay in bounds. Both are the unaligned forms.
    unsafe {
        let n = src.len().min(dst.len());
        let mut i = 0;
        while i + LANES <= n {
            let half = _mm_loadu_si128(src.as_ptr().add(i).cast());
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(half));
            i += LANES;
        }
        while i < n {
            dst[i] = f16_to_f32(src[i]);
            i += 1;
        }
    }
}

/// Widen `N` consecutive cached rows' head slices into `kb`, laid out
/// `[j][i]` with `j` the row within the block.
///
/// # Safety
///
/// Requires AVX2 + F16C. `plane` must hold `(t + N) * kv_dim` elements,
/// `col + head_dim <= kv_dim`, and `kb` must hold `N * head_dim` f32.
#[target_feature(enable = "avx2,f16c")]
unsafe fn widen_rows<const N: usize>(
    plane: &[u16],
    kv_dim: usize,
    col: usize,
    head_dim: usize,
    t: usize,
    kb: &mut [f32],
) {
    // SAFETY: the caller's bounds make every sub-slice below in range; the
    // inner call needs only AVX2 + F16C, which this function's own
    // `target_feature` provides.
    unsafe {
        for j in 0..N {
            let start = (t + j) * kv_dim + col;
            let dst = &mut kb[j * head_dim..][..head_dim];
            widen(&plane[start..start + head_dim], dst);
        }
    }
}

/// Dot one block of `N` cached positions against `LANES` query heads.
///
/// `qt` is the group's query block transposed to `[i][lane]`; `kb` is the
/// block's `N` widened K rows laid out `[j][i]`. Lane `g` of chain `j`
/// accumulates `sum_i qt[i][g] * kb[j][i]` over `i` **ascending** in a single
/// f32, one multiply and one separate add per `i` — bit-identical to the
/// scalar `acc += qv * kv`, per lane. `scale` multiplies the finished
/// accumulator once, exactly as the scalar `run[t] = scale * acc` does.
///
/// The `i` axis is a single chain per `(j, lane)`: it is the `for i` loop
/// below, and nothing splits it.
///
/// # Safety
///
/// Requires AVX2 + F16C. `qt` must hold `head_dim * LANES` f32 and `kb` must
/// hold `N * head_dim` f32.
#[target_feature(enable = "avx2,f16c")]
unsafe fn dot_block<const N: usize>(
    qt: &[f32],
    kb: &[f32],
    head_dim: usize,
    scale: f32,
) -> [[f32; LANES]; N] {
    // SAFETY: `i < head_dim` and `j < N`, so `i * LANES + LANES <= qt.len()`
    // and `j * head_dim + i < kb.len()` under the caller's bounds. Every
    // vector load is the unaligned form. No FMA: the multiply and the add
    // are separate instructions, which is what keeps the two roundings.
    unsafe {
        let qt_ptr = qt.as_ptr();
        let kb_ptr = kb.as_ptr();
        let mut acc = [_mm256_setzero_ps(); N];
        for i in 0..head_dim {
            let qv = _mm256_loadu_ps(qt_ptr.add(i * LANES));
            for (j, a) in acc.iter_mut().enumerate() {
                let kv = _mm256_broadcast_ss(&*kb_ptr.add(j * head_dim + i));
                *a = _mm256_add_ps(*a, _mm256_mul_ps(qv, kv));
            }
        }
        let s = _mm256_set1_ps(scale);
        let mut out = [[0.0f32; LANES]; N];
        for (dst, a) in out.iter_mut().zip(acc.iter()) {
            _mm256_storeu_ps(dst.as_mut_ptr(), _mm256_mul_ps(s, *a));
        }
        out
    }
}

/// Scatter a finished block of scores into the group-major score buffer,
/// keeping only the lanes that carry a real query head.
///
/// Plain scalar stores: `scores` is `[g][t]` with a `positions`-long run per
/// query head, which is the layout [`super::softmax`] must see.
fn store_block<const N: usize>(
    block: &[[f32; LANES]; N],
    base: usize,
    lanes: usize,
    positions: usize,
    t: usize,
    scores: &mut [f32],
) {
    for (j, row) in block.iter().enumerate() {
        for (g, &s) in row.iter().take(lanes).enumerate() {
            scores[(base + g) * positions + t + j] = s;
        }
    }
}

/// Phase 1 for one kv head: `scores[g][t] = scale * (q_g . K[t, kv_head])`
/// for every query head `g` in the GQA group and every attended position.
///
/// Bit-identical to the parent module's `qk_scores_scalar` (module docs,
/// argument (b)). The group is walked in chunks of [`LANES`] query heads; a
/// final short chunk pads the unused lanes with zeros and discards their
/// results, which is what makes `group = 1` (plain MHA) correct rather than
/// special.
///
/// Two stack buffers are carved per call: the transposed query block
/// (`LANES * MAX_SIMD_HEAD_DIM` f32) and the K conversion block
/// (`T_BLOCK * MAX_SIMD_HEAD_DIM` f32), 12 KiB together. They are fully
/// overwritten before they are read; Rust's zero-initialization of them is
/// the only cost, ~1500 cycles per attention call at the v0 pin's four kv
/// heads, against a call that does milliseconds of work at a full 4096-position
/// context. They are *not* carved from the caller's scratch because
/// [`super::scratch_len`] is a pinned contract that a later lane sizes an
/// arena from, and widening it to fund a vectorization detail would be the
/// wrong trade.
///
/// # Safety
///
/// Requires AVX2 + F16C. `q_group` must hold `group * head_dim` f32,
/// `k_rows` exactly `positions` rows of `kv_dim`, `col + head_dim <= kv_dim`,
/// `scores` at least `group * positions` f32, and
/// `head_dim <= MAX_SIMD_HEAD_DIM`.
#[target_feature(enable = "avx2,f16c")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn qk_scores(
    q_group: &[f32],
    k_rows: &[u16],
    kv_dim: usize,
    col: usize,
    head_dim: usize,
    group: usize,
    positions: usize,
    scale: f32,
    scores: &mut [f32],
) {
    // SAFETY: `head_dim <= MAX_SIMD_HEAD_DIM` bounds both stack buffers, and
    // the caller's row/column bounds keep every plane slice in range. The
    // inner helpers need only AVX2 + F16C, which this function's own
    // `target_feature` provides.
    unsafe {
        let mut qt = [0.0f32; LANES * MAX_SIMD_HEAD_DIM];
        let mut kb = [0.0f32; T_BLOCK * MAX_SIMD_HEAD_DIM];

        let mut base = 0;
        while base < group {
            let lanes = (group - base).min(LANES);

            // Transpose this chunk of the group to `[i][lane]`, once per kv
            // head. Lane `g` becomes query head `base + g`'s own accumulator;
            // lanes past the chunk are zeroed and their results dropped.
            for g in 0..LANES {
                if g < lanes {
                    let q_h = &q_group[(base + g) * head_dim..][..head_dim];
                    for (i, &v) in q_h.iter().enumerate() {
                        qt[i * LANES + g] = v;
                    }
                } else {
                    for i in 0..head_dim {
                        qt[i * LANES + g] = 0.0;
                    }
                }
            }

            let mut t = 0;
            while t + T_BLOCK <= positions {
                widen_rows::<T_BLOCK>(k_rows, kv_dim, col, head_dim, t, &mut kb);
                let block = dot_block::<T_BLOCK>(&qt, &kb, head_dim, scale);
                store_block(&block, base, lanes, positions, t, scores);
                t += T_BLOCK;
            }
            while t < positions {
                widen_rows::<1>(k_rows, kv_dim, col, head_dim, t, &mut kb);
                let block = dot_block::<1>(&qt, &kb, head_dim, scale);
                store_block(&block, base, lanes, positions, t, scores);
                t += 1;
            }

            base += LANES;
        }
    }
}

/// Phase 2 for one kv head: `out[g] += sum_t scores[g][t] * V[t, kv_head]`,
/// with `out_group` already zeroed by the caller.
///
/// Bit-identical to the parent module's `v_reduce_scalar` (module docs,
/// argument (c)): eight lanes over `i`, which is elementwise, and `t` left
/// strictly sequential, which is the reduction. The multiply and the add are
/// separate instructions — no FMA.
///
/// # Safety
///
/// Requires AVX2 + F16C. `v_rows` must hold exactly `positions` rows of
/// `kv_dim`, `col + head_dim <= kv_dim`, `vbuf` must hold `head_dim` f32,
/// `scores` at least `group * positions` and `out_group` `group * head_dim`.
#[target_feature(enable = "avx2,f16c")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn v_reduce(
    v_rows: &[u16],
    kv_dim: usize,
    col: usize,
    head_dim: usize,
    positions: usize,
    scores: &[f32],
    vbuf: &mut [f32],
    out_group: &mut [f32],
) {
    // SAFETY: `i + LANES <= head_dim <= out_h.len() == vbuf.len()` on every
    // iteration of the wide loop, so both loads and the store stay in bounds;
    // all three are unaligned forms, which is what lets `out_h` and `vbuf`
    // sit at 4-byte alignment. `widen` needs only AVX2 + F16C.
    unsafe {
        for t in 0..positions {
            let start = t * kv_dim + col;
            widen(&v_rows[start..start + head_dim], vbuf);
            for (out_h, run) in out_group
                .chunks_exact_mut(head_dim)
                .zip(scores.chunks_exact(positions))
            {
                let w = run[t];
                let wv = _mm256_set1_ps(w);
                let mut i = 0;
                while i + LANES <= head_dim {
                    let o = _mm256_loadu_ps(out_h.as_ptr().add(i));
                    let v = _mm256_loadu_ps(vbuf.as_ptr().add(i));
                    let acc = _mm256_add_ps(o, _mm256_mul_ps(wv, v));
                    _mm256_storeu_ps(out_h.as_mut_ptr().add(i), acc);
                    i += LANES;
                }
                while i < head_dim {
                    out_h[i] += w * vbuf[i];
                    i += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::attention::avx2_f16c_available;

    /// Every f16 bit pattern that is not a NaN must widen to exactly the bits
    /// [`f16_to_f32`] produces — normals, subnormals, both zeros, both
    /// infinities, all 65 536 patterns swept. This is the whole evidence for
    /// argument (a) in the module docs, and it is the mirror of
    /// `quants::f16::f16_round_trip_all_finite`.
    ///
    /// NaN patterns are excluded from the bit comparison and asserted to stay
    /// NaN instead: `vcvtph2ps` may set the quiet bit on a signalling NaN
    /// where the scalar conversion reproduces the payload untouched. A
    /// non-finite value in the KV cache means the pass already failed
    /// upstream, so the claim is scoped, not defended — exactly as
    /// `quants::avx2` scopes the same case for its quantizers.
    #[test]
    fn cvtph_matches_scalar_on_every_non_nan_pattern() {
        if !avx2_f16c_available() {
            return;
        }
        let src: Vec<u16> = (0..=u16::MAX).collect();
        let mut dst = vec![0.0f32; src.len()];
        // SAFETY: AVX2 + F16C checked immediately above.
        unsafe { widen(&src, &mut dst) };
        for (bits, &got) in src.iter().copied().zip(&dst) {
            let want = f16_to_f32(bits);
            let is_nan = (bits >> 10) & 0x1F == 0x1F && bits & 0x03FF != 0;
            if is_nan {
                assert!(got.is_nan(), "bits {bits:#06x} widened to {got}, want NaN");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "bits {bits:#06x}: vcvtph2ps {got:e} vs f16_to_f32 {want:e}"
                );
            }
        }
    }

    /// The wide load assumes nothing beyond the 2-byte alignment of `&[u16]`:
    /// the same bytes read through a one-element-shifted view must widen to
    /// the same f32 bits. Modelled on `quants::avx2::misaligned_weight_row_matches`.
    ///
    /// The tail (`len % 8`) is swept too, so the scalar remainder is covered
    /// at every offset.
    #[test]
    fn widen_is_indifferent_to_alignment() {
        if !avx2_f16c_available() {
            return;
        }
        for len in [1usize, 7, 8, 9, 16, 23, 64, 129] {
            let base: Vec<u16> = (0..len)
                .map(|i| (i as u16).wrapping_mul(2477) ^ 0x3C11)
                .collect();
            let mut aligned = vec![0.0f32; len];
            // SAFETY: AVX2 + F16C checked above.
            unsafe { widen(&base, &mut aligned) };

            let mut shifted = vec![0u16; len + 1];
            shifted[1..].copy_from_slice(&base);
            let mut misaligned = vec![0.0f32; len];
            // SAFETY: AVX2 + F16C checked above.
            unsafe { widen(&shifted[1..], &mut misaligned) };

            for (i, (&a, &m)) in aligned.iter().zip(&misaligned).enumerate() {
                assert_eq!(a.to_bits(), m.to_bits(), "len {len} elem {i}");
            }
        }
    }

    /// The same claim one level up: `qk_scores` against a K plane shifted by
    /// one `u16`, which moves every head slice off any 4-byte boundary. Same
    /// contents, so the scores must match bit for bit.
    #[test]
    fn qk_scores_is_indifferent_to_plane_alignment() {
        if !avx2_f16c_available() {
            return;
        }
        // head_dim 9 exercises one wide load plus a scalar tail per row, and
        // an odd row stride so successive rows land on odd element offsets.
        let (group, head_dim, n_kv, positions) = (3usize, 9usize, 2usize, 11usize);
        let kv_dim = n_kv * head_dim;
        let col = head_dim; // kv head 1.
        let plane: Vec<u16> = (0..positions * kv_dim)
            .map(|i| (i as u16).wrapping_mul(40503) ^ 0x2E7B)
            .collect();
        let q: Vec<f32> = (0..group * head_dim)
            .map(|i| i as f32 * 0.017 - 0.5)
            .collect();

        let mut want = vec![0.0f32; group * positions];
        // SAFETY: AVX2 + F16C checked above; the plane holds `positions` rows
        // of `kv_dim`, `col + head_dim == kv_dim`, and `head_dim` is small.
        unsafe {
            qk_scores(
                &q, &plane, kv_dim, col, head_dim, group, positions, 0.125, &mut want,
            )
        };

        let mut shifted = vec![0u16; plane.len() + 1];
        shifted[1..].copy_from_slice(&plane);
        let mut got = vec![0.0f32; group * positions];
        // SAFETY: as above, against the shifted view of the same bytes.
        unsafe {
            qk_scores(
                &q,
                &shifted[1..],
                kv_dim,
                col,
                head_dim,
                group,
                positions,
                0.125,
                &mut got,
            )
        };

        for (i, (&w, &g)) in want.iter().zip(&got).enumerate() {
            assert_eq!(w.to_bits(), g.to_bits(), "score {i}");
        }
    }
}
