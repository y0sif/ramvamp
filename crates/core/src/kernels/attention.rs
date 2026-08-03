//! Single-token GQA decode attention over the FP16 KV cache.
//!
//! Step 1 of the decode loop (`docs/architecture.md`, "Decode loop"): one
//! query vector — all heads, already QK-RMSNormed and RoPE-rotated by the
//! caller — attends over every cached position of one layer. Grouped-query
//! attention maps query head `h` to kv head `h / (n_q_heads / n_kv_heads)`;
//! the v0 pin's 32:4 geometry gives groups of 8, so heads 0-7 read kv head
//! 0, heads 8-15 kv head 1, and so on.
//!
//! Precision policy: K and V are stored as f16 bits and converted to f32
//! per element on the fly ([`f16_to_f32`] is exact) — no dequantized plane
//! copies. The QK dots and the V reduction accumulate in f32, standard
//! practice for f16 attention (ggml's f16 `vec_dot` does the same): at the
//! 4K-position v0 ceiling the f32 accumulation error sits well below the
//! f16 storage error, so f64 accumulators would buy nothing here. The
//! softmax itself ([`softmax`]) takes its exponentials and normalizer in
//! f64 per the primitives' policy.
//!
//! `scale` is caller-provided (`1 / sqrt(head_dim)` for Qwen3) — never
//! hardcoded, because Gemma-family models fold query scaling differently.
//!
//! Alignment: reads assume nothing beyond the natural 2-byte alignment of
//! `&[u16]`; every conversion is a scalar element load, no wide loads (per
//! the kernels module's alignment rule).

use super::KernelError;
use super::primitives::softmax;
use super::quants::f16_to_f32;
use crate::kv::{KvCache, KvError};
use thiserror::Error;

/// Typed errors from decode attention.
///
/// Attention is the first kernel that reads a non-kernel data structure
/// (the KV cache), so cache-side failures pass through as [`KvError`] and
/// primitive-side failures as [`KernelError`], both transparent, next to
/// this kernel's own shape checks. Folding these into the unified
/// `KernelError` is the error-module owner's call when attention joins the
/// backend trait.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AttentionError {
    /// KV cache access failed (layer index out of range).
    #[error(transparent)]
    Kv(#[from] KvError),

    /// A primitive kernel rejected its input. Unreachable after this
    /// module's own validation; kept for lossless propagation.
    #[error(transparent)]
    Kernel(#[from] KernelError),

    /// `q` is empty or not a whole number of `head_dim` heads.
    #[error("decode_attention: q length {q_len} is not a nonzero multiple of head_dim {head_dim}")]
    QLenIndivisible {
        /// Offending query length.
        q_len: usize,
        /// The cache's per-head dimension.
        head_dim: usize,
    },

    /// The query head count is not a multiple of the kv head count.
    #[error(
        "decode_attention: {n_q_heads} query heads do not group evenly over \
         {n_kv_heads} kv heads"
    )]
    GqaGroupMismatch {
        /// Query heads implied by `q.len() / head_dim`.
        n_q_heads: usize,
        /// The cache's kv head count.
        n_kv_heads: usize,
    },

    /// The output buffer does not match the query length.
    #[error("decode_attention: out length {out_len}, expected {expected} (same as q)")]
    OutLenMismatch {
        /// Offending output length.
        out_len: usize,
        /// Expected output length (`q.len()`).
        expected: usize,
    },

    /// The layer has no cached positions — attention over an empty history
    /// is undefined (the caller must append the current token's K/V first).
    #[error("decode_attention: layer {layer} has no cached positions")]
    EmptyLayer {
        /// The empty layer.
        layer: usize,
    },
}

/// Reusable per-position score buffer for [`decode_attention`].
///
/// Holds one f32 per cached position. The buffer grows to the high-water
/// sequence length and is then reused as-is, so a scratch constructed once
/// (ideally via [`Self::with_capacity`] at the cache's capacity, 4096 for
/// v0) allocates at most once and never again on the decode path.
#[derive(Debug, Default)]
pub struct AttentionScratch {
    scores: Vec<f32>,
}

impl AttentionScratch {
    /// An empty scratch; the first [`decode_attention`] call sizes it.
    pub fn new() -> Self {
        Self::default()
    }

    /// A scratch preallocated for `positions` cached positions, so the hot
    /// path never allocates at all.
    pub fn with_capacity(positions: usize) -> Self {
        Self {
            scores: Vec::with_capacity(positions),
        }
    }

    /// The score buffer resized to `len` (reallocates only past the
    /// high-water mark).
    fn scores_mut(&mut self, len: usize) -> &mut [f32] {
        self.scores.resize(len, 0.0);
        &mut self.scores[..len]
    }
}

/// Single-token GQA decode attention for one layer:
/// `out_h = sum_t softmax_t(scale * q_h . K[t, kv(h)]) * V[t, kv(h)]`
/// over all cached positions `t`, with `kv(h) = h / (n_q_heads / n_kv_heads)`.
///
/// `q` is the full post-RoPE query `[n_q_heads * head_dim]` (4096 for the
/// v0 pin); `n_q_heads` is derived from `q.len()` and the cache's
/// `head_dim`, everything else comes from the cache dims. `out` receives
/// the concatenated per-head context vectors, same layout and length as
/// `q`; it is fully overwritten. The current token's K/V must already be
/// appended (position `t = len - 1` attends to itself).
///
/// # Errors
///
/// [`AttentionError::Kv`] for a bad layer index; [`AttentionError`]'s shape
/// variants for `q`/`out`/GQA mismatches; [`AttentionError::EmptyLayer`]
/// when the layer holds no positions. `out` is untouched on error.
pub fn decode_attention(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    scale: f32,
    scratch: &mut AttentionScratch,
    out: &mut [f32],
) -> Result<(), AttentionError> {
    let head_dim = cache.head_dim();
    let n_kv_heads = cache.n_kv_heads();
    let kv_dim = cache.kv_dim();

    if q.is_empty() || q.len() % head_dim != 0 {
        return Err(AttentionError::QLenIndivisible {
            q_len: q.len(),
            head_dim,
        });
    }
    let n_q_heads = q.len() / head_dim;
    if n_q_heads % n_kv_heads != 0 {
        return Err(AttentionError::GqaGroupMismatch {
            n_q_heads,
            n_kv_heads,
        });
    }
    if out.len() != q.len() {
        return Err(AttentionError::OutLenMismatch {
            out_len: out.len(),
            expected: q.len(),
        });
    }
    let len = cache.len(layer)?;
    if len == 0 {
        return Err(AttentionError::EmptyLayer { layer });
    }
    let k_plane = cache.k_layer(layer)?;
    let v_plane = cache.v_layer(layer)?;

    let group = n_q_heads / n_kv_heads;
    let scores = scratch.scores_mut(len);

    for (h, out_h) in out.chunks_exact_mut(head_dim).enumerate() {
        let kv_head = h / group;
        let col = kv_head * head_dim;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];

        // scores[t] = scale * (q_h . K[t, kv_head]), f32 accumulate.
        for (row, score) in k_plane.chunks_exact(kv_dim).zip(scores.iter_mut()) {
            let k_th = &row[col..col + head_dim];
            let mut acc = 0.0f32;
            for (&qv, &kb) in q_h.iter().zip(k_th) {
                acc += qv * f16_to_f32(kb);
            }
            *score = scale * acc;
        }

        softmax(scores)?;

        // out_h = sum_t scores[t] * V[t, kv_head].
        out_h.fill(0.0);
        for (row, &w) in v_plane.chunks_exact(kv_dim).zip(scores.iter()) {
            let v_th = &row[col..col + head_dim];
            for (o, &vb) in out_h.iter_mut().zip(v_th) {
                *o += w * f16_to_f32(vb);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::quants::f32_to_f16;

    // Local copies of the deterministic PRNG and tolerance assertion from
    // `primitives::testutil` — that module is `#[cfg(test)]`-private to
    // `primitives`, and `primitives/mod.rs` is frozen on this branch, so it
    // cannot be re-exported from here.

    /// xorshift64* PRNG, seedable and deterministic.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
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

        fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
            let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
            lo + (hi - lo) * unit
        }

        fn vec_in(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
            (0..n).map(|_| self.f32_in(lo, hi)).collect()
        }
    }

    /// Assert `actual` is within `abs_tol` absolute or `rel_tol` relative
    /// of the f64 reference `expected`.
    fn assert_close(actual: f32, expected: f64, rel_tol: f64, abs_tol: f64) {
        let diff = (f64::from(actual) - expected).abs();
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

    /// Naive full-precision reference: f64 attention over f32 K/V rows
    /// (`rows[t]` is one position, head-major `[n_kv_heads * head_dim]`).
    fn reference_attention(
        q: &[f32],
        k_rows: &[Vec<f32>],
        v_rows: &[Vec<f32>],
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f64> {
        let n_q_heads = q.len() / head_dim;
        let group = n_q_heads / n_kv_heads;
        let len = k_rows.len();
        let mut out = vec![0.0f64; q.len()];
        for h in 0..n_q_heads {
            let col = (h / group) * head_dim;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores: Vec<f64> = (0..len)
                .map(|t| {
                    let k_th = &k_rows[t][col..col + head_dim];
                    let dot: f64 = q_h
                        .iter()
                        .zip(k_th)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    dot * f64::from(scale)
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0f64;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for (t, &w) in scores.iter().enumerate() {
                let v_th = &v_rows[t][col..col + head_dim];
                for (i, &v) in v_th.iter().enumerate() {
                    out[h * head_dim + i] += w * f64::from(v);
                }
            }
        }
        out
    }

    /// One f16 round trip, elementwise: what the cache actually stores.
    fn round_trip(rows: &[Vec<f32>]) -> Vec<Vec<f32>> {
        rows.iter()
            .map(|r| r.iter().map(|&x| f16_to_f32(f32_to_f16(x))).collect())
            .collect()
    }

    /// Gate 1: decode_attention vs the f64 reference, v0 pin geometry
    /// (32 q-heads : 4 kv-heads, head_dim 128), multiple layers, seq lens
    /// 1 / 2 / 17 / 256. Two comparisons per point: against the exact f32
    /// inputs (tolerance dominated by f16 storage, ~1e-3 abs on unit-scale
    /// data) and against the f16-round-tripped inputs (tight tolerance —
    /// only accumulation order differs, which pins the arithmetic itself).
    #[test]
    fn matches_f64_reference_across_layers_and_lengths() {
        let (n_layers, n_kv, n_q, head_dim, cap) = (3usize, 4usize, 32usize, 128usize, 256usize);
        let kv_dim = n_kv * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut rng = Rng::new(0xA77E);
        let mut cache = KvCache::new(n_layers, n_kv, head_dim, cap).unwrap();
        let mut k_exact: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut v_exact: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut scratch = AttentionScratch::new();

        let mut appended = 0usize;
        for &target in &[1usize, 2, 17, 256] {
            while appended < target {
                for layer in 0..n_layers {
                    let k = rng.vec_in(kv_dim, -1.0, 1.0);
                    let v = rng.vec_in(kv_dim, -1.0, 1.0);
                    cache.append(layer, &k, &v).unwrap();
                    k_exact[layer].push(k);
                    v_exact[layer].push(v);
                }
                appended += 1;
            }
            assert_eq!(cache.seq_len().unwrap(), target);

            for layer in 0..n_layers {
                let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
                let mut out = vec![0.0f32; q.len()];
                decode_attention(&q, &cache, layer, scale, &mut scratch, &mut out).unwrap();

                let want = reference_attention(
                    &q,
                    &k_exact[layer],
                    &v_exact[layer],
                    n_kv,
                    head_dim,
                    scale,
                );
                for (&got, &w) in out.iter().zip(&want) {
                    assert_close(got, w, 1e-3, 1.5e-3);
                }

                let want_rt = reference_attention(
                    &q,
                    &round_trip(&k_exact[layer]),
                    &round_trip(&v_exact[layer]),
                    n_kv,
                    head_dim,
                    scale,
                );
                for (&got, &w) in out.iter().zip(&want_rt) {
                    assert_close(got, w, 1e-4, 5e-5);
                }
            }
        }
    }

    /// Gate 2: the GQA head->group assignment is exactly `h / 8` for 32:4.
    /// Four cached positions; kv head `j`'s keys make position `t = j` the
    /// runaway softmax winner, and `V[t, j]` carries the marker `10j + t`
    /// (scaled). Every query head must come back with the marker of kv head
    /// `h / 8` at the winner `t = h / 8` — in particular q-heads 16..24 must
    /// read only kv head 2. Any other mapping (h / 4, h % 4, identity)
    /// lands on a different marker, 0.11 apart vs a 1e-3 gate.
    #[test]
    fn gqa_maps_query_head_h_to_kv_head_h_div_8() {
        let (n_kv, n_q, head_dim) = (4usize, 32usize, 16usize);
        let kv_dim = n_kv * head_dim;
        let mut cache = KvCache::new(1, n_kv, head_dim, 4).unwrap();
        let marker = |j: usize, t: usize| (10 * j + t) as f32 * 0.01;
        for t in 0..4 {
            let mut k = vec![0.0f32; kv_dim];
            let mut v = vec![0.0f32; kv_dim];
            for j in 0..n_kv {
                let sign = if t == j { 1.0f32 } else { -1.0 };
                for i in 0..head_dim {
                    // q = all-ones => score(t) = +-8 before scaling.
                    k[j * head_dim + i] = sign * 8.0 / head_dim as f32;
                    v[j * head_dim + i] = marker(j, t);
                }
            }
            cache.append(0, &k, &v).unwrap();
        }
        let q = vec![1.0f32; n_q * head_dim];
        let mut out = vec![0.0f32; q.len()];
        let mut scratch = AttentionScratch::new();
        decode_attention(&q, &cache, 0, 1.0, &mut scratch, &mut out).unwrap();
        for h in 0..n_q {
            let j = h / 8; // The pinned mapping.
            let want = marker(j, j); // Winner position t == j.
            for i in 0..head_dim {
                let got = out[h * head_dim + i];
                assert!(
                    (got - want).abs() < 1e-3,
                    "q head {h} elem {i}: got {got}, want kv head {j} marker {want}"
                );
            }
        }
    }

    /// Gate 3: a single cached position makes softmax degenerate (weight
    /// exactly 1.0), so each head's output is exactly the f16-round-tripped
    /// V row of its kv head — bit-exact, and within f16 tolerance of the
    /// original f32 V.
    #[test]
    fn single_position_returns_v_row() {
        let (n_kv, n_q, head_dim) = (4usize, 32usize, 128usize);
        let kv_dim = n_kv * head_dim;
        let mut rng = Rng::new(0x5EED);
        let mut cache = KvCache::new(2, n_kv, head_dim, 8).unwrap();
        let mut v_by_layer = Vec::new();
        for layer in 0..2 {
            let k = rng.vec_in(kv_dim, -1.0, 1.0);
            let v = rng.vec_in(kv_dim, -1.0, 1.0);
            cache.append(layer, &k, &v).unwrap();
            v_by_layer.push(v);
        }
        let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
        let mut scratch = AttentionScratch::new();
        for (layer, v) in v_by_layer.iter().enumerate() {
            let mut out = vec![0.0f32; q.len()];
            let scale = 1.0 / (head_dim as f32).sqrt();
            decode_attention(&q, &cache, layer, scale, &mut scratch, &mut out).unwrap();
            for h in 0..n_q {
                let col = (h / 8) * head_dim;
                for i in 0..head_dim {
                    let exact = f16_to_f32(f32_to_f16(v[col + i]));
                    assert_eq!(
                        out[h * head_dim + i],
                        exact,
                        "layer {layer} head {h} elem {i}"
                    );
                    assert_close(out[h * head_dim + i], f64::from(v[col + i]), 4.9e-4, 6e-8);
                }
            }
        }
    }

    /// Gate 4 (attention side): every shape violation is a typed error and
    /// `out` is untouched.
    #[test]
    fn typed_errors_on_bad_shapes() {
        let (n_kv, head_dim) = (4usize, 8usize);
        let mut cache = KvCache::new(1, n_kv, head_dim, 4).unwrap();
        let mut scratch = AttentionScratch::new();
        let q = vec![0.0f32; 32 * head_dim];
        let mut out = vec![7.0f32; q.len()];

        // Empty layer (nothing appended yet).
        assert_eq!(
            decode_attention(&q, &cache, 0, 1.0, &mut scratch, &mut out).unwrap_err(),
            AttentionError::EmptyLayer { layer: 0 }
        );

        let row = vec![0.5f32; n_kv * head_dim];
        cache.append(0, &row, &row).unwrap();

        // Layer out of range passes the cache error through.
        assert_eq!(
            decode_attention(&q, &cache, 1, 1.0, &mut scratch, &mut out).unwrap_err(),
            AttentionError::Kv(KvError::LayerOutOfRange {
                layer: 1,
                n_layers: 1,
            })
        );

        // q not a whole number of heads / empty q.
        assert_eq!(
            decode_attention(&q[..13], &cache, 0, 1.0, &mut scratch, &mut out).unwrap_err(),
            AttentionError::QLenIndivisible {
                q_len: 13,
                head_dim,
            }
        );
        assert_eq!(
            decode_attention(&[], &cache, 0, 1.0, &mut scratch, &mut out).unwrap_err(),
            AttentionError::QLenIndivisible { q_len: 0, head_dim }
        );

        // 5 query heads cannot group over 4 kv heads.
        assert_eq!(
            decode_attention(&q[..5 * head_dim], &cache, 0, 1.0, &mut scratch, &mut out)
                .unwrap_err(),
            AttentionError::GqaGroupMismatch {
                n_q_heads: 5,
                n_kv_heads: 4,
            }
        );

        // Output length must equal q length.
        assert_eq!(
            decode_attention(&q, &cache, 0, 1.0, &mut scratch, &mut out[..7]).unwrap_err(),
            AttentionError::OutLenMismatch {
                out_len: 7,
                expected: q.len(),
            }
        );

        // No error path wrote anything.
        assert!(out.iter().all(|&x| x == 7.0));
    }

    /// Gate 6: scale sensitivity. Two positions with QK dots 0 and 1, V
    /// rows 0 and 1, so `out = sigmoid(scale)` analytically; doubling the
    /// scale must square the softmax odds ratio.
    #[test]
    fn doubling_scale_squares_the_odds_ratio() {
        let head_dim = 4usize;
        let mut cache = KvCache::new(1, 1, head_dim, 2).unwrap();
        // q . k0 = 0, q . k1 = 1 (all values exact in f16).
        cache.append(0, &[0.0, 0.0, 0.0, 0.0], &[0.0; 4]).unwrap();
        cache.append(0, &[1.0, 0.0, 0.0, 0.0], &[1.0; 4]).unwrap();
        let q = [1.0f32, 0.0, 0.0, 0.0];
        let mut scratch = AttentionScratch::new();
        let p_at = |scale: f32, scratch: &mut AttentionScratch| {
            let mut out = [0.0f32; 4];
            decode_attention(&q, &cache, 0, scale, scratch, &mut out).unwrap();
            f64::from(out[0]) // = softmax weight of position 1.
        };
        let s = 0.7f32;
        let p1 = p_at(s, &mut scratch);
        let p2 = p_at(2.0 * s, &mut scratch);
        let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
        assert_close(p1 as f32, sigmoid(f64::from(s)), 1e-5, 0.0);
        assert_close(p2 as f32, sigmoid(2.0 * f64::from(s)), 1e-5, 0.0);
        // Doubling the scale squares each pairwise odds ratio.
        let odds = |p: f64| p / (1.0 - p);
        assert_close(odds(p2) as f32, odds(p1).powi(2), 1e-4, 0.0);
        // Sanity: larger scale sharpens toward the winning position.
        assert!(p2 > p1);
    }

    /// The scratch buffer reaches its high-water mark once and is then
    /// reused without reallocating, including for shorter sequences.
    #[test]
    fn scratch_reuses_allocation() {
        let mut scratch = AttentionScratch::with_capacity(17);
        let ptr = scratch.scores.as_ptr();
        assert_eq!(scratch.scores_mut(17).len(), 17);
        assert_eq!(scratch.scores.as_ptr(), ptr);
        assert_eq!(scratch.scores_mut(3).len(), 3);
        assert_eq!(scratch.scores.as_ptr(), ptr);
        assert_eq!(scratch.scores_mut(17).len(), 17);
        assert_eq!(scratch.scores.as_ptr(), ptr);
        assert_eq!(scratch.scores.capacity(), 17);
    }
}
