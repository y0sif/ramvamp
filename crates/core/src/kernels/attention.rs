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
//! Causality: [`decode_attention`] has no mask because it does not need one
//! — the decode loop appends the current token's K/V and calls immediately,
//! so the layer holds exactly positions `0..=current`. Chunked layer-major
//! prefill breaks that invariant on purpose (a whole chunk's K/V lands in
//! layer `L` before any of the chunk's rows are attended), so it uses
//! [`attention_at`], which takes the row's position explicitly and stops the
//! sum there. Both entry points run the same private body, so the masked
//! form is the unmasked form against a shorter cache — identical f32
//! operations in identical order, bit for bit.
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
    #[error("attention: q length {q_len} is not a nonzero multiple of head_dim {head_dim}")]
    QLenIndivisible {
        /// Offending query length.
        q_len: usize,
        /// The cache's per-head dimension.
        head_dim: usize,
    },

    /// The query head count is not a multiple of the kv head count.
    #[error(
        "attention: {n_q_heads} query heads do not group evenly over \
         {n_kv_heads} kv heads"
    )]
    GqaGroupMismatch {
        /// Query heads implied by `q.len() / head_dim`.
        n_q_heads: usize,
        /// The cache's kv head count.
        n_kv_heads: usize,
    },

    /// The output buffer does not match the query length.
    #[error("attention: out length {out_len}, expected {expected} (same as q)")]
    OutLenMismatch {
        /// Offending output length.
        out_len: usize,
        /// Expected output length (`q.len()`).
        expected: usize,
    },

    /// There is nothing to attend over: either the layer holds no cached
    /// positions, or [`attention_at`] was asked for `positions == 0`.
    /// Attention over an empty history is undefined — the caller must append
    /// the current token's K/V first, and a row always attends to at least
    /// itself (`positions = p + 1 >= 1`).
    #[error("attention: layer {layer} has no positions to attend over")]
    EmptyLayer {
        /// The layer with nothing to attend over.
        layer: usize,
    },
}

/// Reusable per-position score buffer for [`decode_attention`] and
/// [`attention_at`].
///
/// Holds one f32 per *attended* position — the full cached length for
/// [`decode_attention`], `p + 1` for [`attention_at`]. The buffer grows to
/// the high-water sequence length and is then reused as-is, so a scratch
/// constructed once (ideally via [`Self::with_capacity`] at the cache's
/// capacity, 4096 for v0) allocates at most once and never again on the
/// decode or prefill path. Shrinking for a shorter row only truncates, and
/// every retained entry is overwritten before it is read, so a short call
/// after a long one can never pick up a stale score.
#[derive(Debug, Default)]
pub struct AttentionScratch {
    scores: Vec<f32>,
}

impl AttentionScratch {
    /// An empty scratch; the first attention call sizes it.
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
    // `None` = "every position the layer holds", which is the decode
    // invariant above. Delegating means the masked and unmasked paths cannot
    // drift apart (`decode_attention_matches_attention_at_at_full_length`).
    attention_impl(q, cache, layer, None, scale, scratch, out)
}

/// Position-limited GQA attention for one layer: exactly
/// [`decode_attention`], except the sum runs over cached positions
/// `0..positions` instead of over the whole layer.
///
/// This is the causal mask for chunked layer-major prefill, where a whole
/// chunk's K/V is appended to layer `L` before any of the chunk's rows are
/// attended. The row at absolute position `p` passes `positions = p + 1` and
/// therefore cannot see the future rows already sitting in the cache behind
/// it. Arguments are otherwise [`decode_attention`]'s.
///
/// The mask is *structural*, not additive. Positions `>= positions` are
/// absent from the score buffer, from the softmax normalizer, and from the V
/// reduction — precisely as they are absent from a cache that only holds
/// `positions` rows. Nothing is `-inf`-biased, nothing is zero-weighted and
/// summed anyway, and the softmax stays the single-pass max / f64-exp /
/// f64-normalize of [`softmax`]; this is deliberately *not* an online or
/// flash-style rescaled softmax, because that would reassociate the
/// reduction and break the byte-identical-logits gate. The f32 operations
/// and their order are the ones [`decode_attention`] performs against a
/// `positions`-row cache, so the two agree bit for bit
/// (`attention_at_is_bit_identical_to_truncated_decode`).
///
/// # Errors
///
/// Everything [`decode_attention`] returns, plus
/// [`AttentionError::EmptyLayer`] for `positions == 0` (nothing to attend
/// over; a row always attends to at least itself) and [`AttentionError::Kv`]
/// wrapping [`KvError::PositionOutOfRange`] when `positions` exceeds the
/// positions the layer actually holds — never a silent truncation, never a
/// panic. `out` is untouched on error.
pub fn attention_at(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    positions: usize,
    scale: f32,
    scratch: &mut AttentionScratch,
    out: &mut [f32],
) -> Result<(), AttentionError> {
    attention_impl(q, cache, layer, Some(positions), scale, scratch, out)
}

/// The one attention body. `limit` is `Some(positions)` for the causal
/// prefill form and `None` for "the whole layer" (decode).
fn attention_impl(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    limit: Option<usize>,
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
    let positions = limit.unwrap_or(len);
    if positions == 0 {
        return Err(AttentionError::EmptyLayer { layer });
    }
    if positions > len {
        // Reported, never truncated: a prefill driver asking for a position
        // the layer has not been given yet is a bug, not a shorter row.
        return Err(AttentionError::Kv(KvError::PositionOutOfRange {
            layer,
            pos: positions - 1,
            len,
        }));
    }
    let k_plane = cache.k_layer(layer)?;
    let v_plane = cache.v_layer(layer)?;

    let group = n_q_heads / n_kv_heads;
    // The causal limit lives entirely in this length. Both reductions below
    // zip the layer's rows against `scores`, so they stop after `positions`
    // rows and every later row is absent from the score buffer, the softmax
    // normalizer, and the V sum — not zero-weighted, absent. That is what
    // makes the limited call bit-identical to an unlimited call against a
    // `positions`-row cache.
    let scores = scratch.scores_mut(positions);

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

    /// Geometries for the position-limited gates: `(n_layers, n_kv_heads,
    /// n_q_heads, head_dim, n_positions)` — the v0 pin (32:4, group 8), MHA
    /// (group 1), MQA (8 q-heads over 1 kv head) and group 2, with head_dim
    /// 128 / 16 / 8 / 4 and more than one layer so the per-layer plane
    /// offset is exercised alongside the position limit.
    const LIMITED_GEOMETRIES: [(usize, usize, usize, usize, usize); 4] = [
        (2, 4, 32, 128, 5),
        (1, 2, 2, 16, 9),
        (1, 1, 8, 8, 7),
        (2, 3, 6, 4, 12),
    ];

    /// Fill a fresh cache with `n_pos` random positions per layer, returning
    /// the cache and the f32 rows that produced it (`rows[layer][pos]`).
    #[allow(clippy::type_complexity)]
    fn random_cache(
        rng: &mut Rng,
        n_layers: usize,
        n_kv: usize,
        head_dim: usize,
        n_pos: usize,
    ) -> (KvCache, Vec<Vec<(Vec<f32>, Vec<f32>)>>) {
        let kv_dim = n_kv * head_dim;
        let rows: Vec<Vec<(Vec<f32>, Vec<f32>)>> = (0..n_layers)
            .map(|_| {
                (0..n_pos)
                    .map(|_| (rng.vec_in(kv_dim, -1.0, 1.0), rng.vec_in(kv_dim, -1.0, 1.0)))
                    .collect()
            })
            .collect();
        let mut cache = KvCache::new(n_layers, n_kv, head_dim, n_pos.max(1)).unwrap();
        for (layer, layer_rows) in rows.iter().enumerate() {
            for (k, v) in layer_rows {
                cache.append(layer, k, v).unwrap();
            }
        }
        (cache, rows)
    }

    /// Gate 7 (Phase 6 causal mask, the central claim): for every prefix
    /// length `n`, `attention_at(.., positions = n)` against a full-length
    /// cache is *bit-identical* to `decode_attention` against a cache holding
    /// exactly the first `n` rows. That is the whole correctness argument for
    /// chunked layer-major prefill: limiting the sum is not a mask applied to
    /// a longer computation, it *is* the shorter computation — same f32
    /// operations, same order, so `to_bits()` equality must hold with zero
    /// tolerance. Any reassociation (an online / flash-style rescaled
    /// softmax, an additive `-inf` bias, a zero-weighted tail that still
    /// enters the V sum) breaks this test on the first geometry.
    ///
    /// The scratch is primed at full length before each limited call, so a
    /// stale score surviving into a shorter row would also fail here.
    #[test]
    fn attention_at_is_bit_identical_to_truncated_decode() {
        let mut rng = Rng::new(0xB17D_E17E);
        let mut scratch = AttentionScratch::new();
        for &(n_layers, n_kv, n_q, head_dim, n_pos) in &LIMITED_GEOMETRIES {
            let scale = 1.0 / (head_dim as f32).sqrt();
            let (full, rows) = random_cache(&mut rng, n_layers, n_kv, head_dim, n_pos);
            let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
            let mut prime = vec![0.0f32; q.len()];

            for (layer, layer_rows) in rows.iter().enumerate() {
                for n in 1..=n_pos {
                    // Reference: a cache that only ever held `n` positions,
                    // i.e. exactly what the decode loop sees at position n-1.
                    let mut trunc = KvCache::new(n_layers, n_kv, head_dim, n).unwrap();
                    for (k, v) in layer_rows.iter().take(n) {
                        trunc.append(layer, k, v).unwrap();
                    }

                    // Drive the scratch to its high-water mark first: the
                    // limited call must not read the tail it leaves behind.
                    attention_at(&q, &full, layer, n_pos, scale, &mut scratch, &mut prime).unwrap();

                    let mut got = vec![0.0f32; q.len()];
                    attention_at(&q, &full, layer, n, scale, &mut scratch, &mut got).unwrap();
                    let mut want = vec![0.0f32; q.len()];
                    decode_attention(&q, &trunc, layer, scale, &mut scratch, &mut want).unwrap();

                    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "geometry ({n_layers}, {n_kv}, {n_q}, {head_dim}) layer {layer} \
                             positions {n} elem {i}: attention_at {g:e} vs truncated \
                             decode_attention {w:e}"
                        );
                    }
                }
            }
        }
    }

    /// Gate 8: `decode_attention` delegates to the same body, so asking
    /// `attention_at` for the layer's whole cached length must reproduce it
    /// bit for bit. Layers are deliberately ragged (3 / 1 / 6 positions) so
    /// the full length is per-layer, not a shared sequence length.
    #[test]
    fn decode_attention_matches_attention_at_at_full_length() {
        let (n_kv, n_q, head_dim) = (4usize, 32usize, 128usize);
        let kv_dim = n_kv * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut rng = Rng::new(0xDE1E_6A7E);
        let lens = [3usize, 1, 6];
        let mut cache = KvCache::new(lens.len(), n_kv, head_dim, 6).unwrap();
        for (layer, &len) in lens.iter().enumerate() {
            for _ in 0..len {
                let k = rng.vec_in(kv_dim, -1.0, 1.0);
                let v = rng.vec_in(kv_dim, -1.0, 1.0);
                cache.append(layer, &k, &v).unwrap();
            }
        }
        let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
        let mut scratch = AttentionScratch::new();
        for (layer, &len) in lens.iter().enumerate() {
            assert_eq!(cache.len(layer).unwrap(), len);
            let mut want = vec![0.0f32; q.len()];
            decode_attention(&q, &cache, layer, scale, &mut scratch, &mut want).unwrap();
            let mut got = vec![0.0f32; q.len()];
            attention_at(&q, &cache, layer, len, scale, &mut scratch, &mut got).unwrap();
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "layer {layer} elem {i}");
            }
        }
    }

    /// Gate 9: every `positions` violation is a typed error and `out` is
    /// untouched — `ramvamp-core` must not panic on untrusted input, and a
    /// driver asking past the end must be told, not silently truncated.
    #[test]
    fn attention_at_typed_errors_on_positions() {
        let (n_kv, head_dim) = (2usize, 8usize);
        let mut cache = KvCache::new(2, n_kv, head_dim, 4).unwrap();
        let row = vec![0.5f32; n_kv * head_dim];
        for _ in 0..3 {
            cache.append(0, &row, &row).unwrap();
        }
        // Layer 1 stays empty on purpose.
        let q = vec![0.25f32; 4 * head_dim];
        let mut out = vec![7.0f32; q.len()];
        let mut scratch = AttentionScratch::new();

        // positions == 0: nothing to attend over. A row always attends to at
        // least itself, so this is a caller bug, reported not tolerated.
        assert_eq!(
            attention_at(&q, &cache, 0, 0, 1.0, &mut scratch, &mut out).unwrap_err(),
            AttentionError::EmptyLayer { layer: 0 }
        );

        // One past the end, well past it, and the saturating case: reported
        // against the layer's real length, never truncated, never overflowing.
        for want in [4usize, 9, usize::MAX] {
            assert_eq!(
                attention_at(&q, &cache, 0, want, 1.0, &mut scratch, &mut out).unwrap_err(),
                AttentionError::Kv(KvError::PositionOutOfRange {
                    layer: 0,
                    pos: want - 1,
                    len: 3,
                })
            );
        }

        // An empty layer rejects even one position, and `positions == 0` on
        // it is still the empty-history error.
        assert_eq!(
            attention_at(&q, &cache, 1, 1, 1.0, &mut scratch, &mut out).unwrap_err(),
            AttentionError::Kv(KvError::PositionOutOfRange {
                layer: 1,
                pos: 0,
                len: 0,
            })
        );
        assert_eq!(
            attention_at(&q, &cache, 1, 0, 1.0, &mut scratch, &mut out).unwrap_err(),
            AttentionError::EmptyLayer { layer: 1 }
        );

        // Layer and shape checks are the shared ones, and shapes are still
        // checked before `positions`.
        assert_eq!(
            attention_at(&q, &cache, 2, 1, 1.0, &mut scratch, &mut out).unwrap_err(),
            AttentionError::Kv(KvError::LayerOutOfRange {
                layer: 2,
                n_layers: 2,
            })
        );
        assert_eq!(
            attention_at(&q[..3], &cache, 0, usize::MAX, 1.0, &mut scratch, &mut out).unwrap_err(),
            AttentionError::QLenIndivisible { q_len: 3, head_dim }
        );

        // No error path wrote anything.
        assert!(out.iter().all(|&x| x == 7.0));
    }
}
