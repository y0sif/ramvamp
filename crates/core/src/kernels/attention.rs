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
//! `&[u16]` and the 4-byte alignment of `&[f32]` (per the kernels module's
//! alignment rule). The scalar path converts one element at a time, so it
//! assumes nothing at all. The AVX2 path ([`x86`]) uses wide loads, and every
//! one of them is an **unaligned** form — `_mm_loadu_si128` for the f16 K/V
//! head slices, `_mm256_loadu_ps` / `_mm256_storeu_ps` for the f32 query
//! transpose, conversion row, score runs and output. That is not belt and
//! braces: a head slice sits at a `kv_head * head_dim` element offset inside
//! a row of `n_kv_heads * head_dim`, so an odd `head_dim` puts it on a
//! 2-byte boundary and the matching `out` sub-slice on a 4-byte one. Tested
//! at `head_dim = 9` end to end
//! (`misaligned_head_slices_match_the_scalar_reference`) and against
//! deliberately shifted views inside [`x86`].
//!
//! Loop order: **kv-head outer, query head inner**. Each K and each V
//! element is converted from f16 exactly once and then reused by every query
//! head in its GQA group (8 of them at the v0 pin) out of a `head_dim`-long
//! f32 buffer. The head-major order this replaced re-converted every element
//! once per query head in the group, and phase 7's wave-0 bench measured
//! that conversion plus the serial f32 accumulate as the whole cost of the
//! kernel — 1.54 ns per element against only 0.17 GB/s of unique K+V bytes,
//! i.e. compute-bound, not memory-bound.
//!
//! The restructure is bit-neutral by construction, which is the only reason
//! it is allowed here. `h = kv_head * group + g` with `kv_head` outer and
//! `g` inner visits the query heads in the same ascending order as the old
//! `kv_head = h / group`; [`f16_to_f32`] is an exact widening, so a buffered
//! element is bit-for-bit the value the inline conversion produced; the QK
//! dot still accumulates over `i` ascending into one f32 with separate
//! multiply and add (no FMA — the two roundings are load-bearing); and the V
//! reduction still accumulates over `t` ascending, per output element. Same
//! operands, same order, same roundings
//! (`restructured_kernel_is_bit_identical_to_head_major_reference`).
//!
//! Vectorization: phase 7's wave 2 added an AVX2 + F16C path for both phases
//! ([`x86`]), dispatched at runtime and **bit-identical** to the scalar
//! reference below, which is what the whole gate rests on. `vcvtph2ps`
//! widens eight f16 at a time and is exact; the QK dot puts the eight vector
//! lanes on the **GQA group**, an axis whose accumulators are already
//! independent, so each lane still walks `i` ascending in one f32 with a
//! separate multiply and add; the V reduction puts them on `i`, which is
//! elementwise, and leaves `t` sequential. No FMA anywhere — `acc += qv * kv`
//! is two roundings and `_mm256_fmadd_ps` would make it one. The full
//! argument lives in [`x86`]'s module docs.
//!
//! Dispatch follows the convention `quants::avx2` set: the path is chosen by
//! a runtime CPUID probe ([`avx2_f16c_available`]) and can be pinned with a
//! `force_scalar` parameter threaded through the `*_dispatch` entry points,
//! never an environment variable, so tests and benches can drive both paths
//! in one process.

use super::KernelError;
use super::primitives::softmax;
use super::quants::f16_to_f32;
use crate::kv::{KvCache, KvError};
use thiserror::Error;

#[cfg(target_arch = "x86_64")]
mod x86;

/// The largest `head_dim` the AVX2 path covers.
///
/// [`x86::qk_scores`] carves its transposed query block and its K conversion
/// block from fixed stack arrays sized by this constant, rather than from the
/// caller's scratch: [`scratch_len`] is a pinned contract that a later lane
/// sizes a prefill arena from, and widening it to fund a vectorization detail
/// would be the wrong trade. Geometries past this bound fall through to the
/// scalar reference, which has no such limit. 256 is twice the v0 pin's
/// `head_dim` and covers every head size in current use.
#[cfg(target_arch = "x86_64")]
const MAX_SIMD_HEAD_DIM: usize = 256;

/// True when the running CPU supports both AVX2 and F16C.
///
/// F16C is a **separate CPUID bit** from AVX2 and FMA, so this is not
/// [`super::quants::avx2::avx2_fma_available`] and must not be folded into
/// it: this kernel needs `vcvtph2ps` and deliberately does not want FMA.
/// `is_x86_feature_detected!` caches the probe, so calling this per kv head
/// costs an atomic load.
#[inline]
pub fn avx2_f16c_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// True when a call at this `head_dim` really takes the AVX2 path, i.e. when
/// `force_scalar` selects between two *different* kernels rather than running
/// the same one twice.
///
/// This is [`avx2_f16c_available`] **and** the [`MAX_SIMD_HEAD_DIM`] envelope.
/// The second half matters to tests: a geometry outside the envelope falls
/// back silently and correctly, and a bit-identity test swept over
/// `force_scalar` on such a geometry would compare the scalar kernel with
/// itself while looking like a two-path proof. `simd_matches_scalar_bit_for_bit`
/// asserts this instead of assuming it.
pub fn avx2_f16c_path_covers(head_dim: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        head_dim <= MAX_SIMD_HEAD_DIM && avx2_f16c_available()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = head_dim;
        false
    }
}

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

    /// A caller-provided scratch slice ([`attention_at_in`] /
    /// [`decode_attention_in`]) is shorter than this call's geometry needs.
    /// Reported, never worked around: silently attending over fewer
    /// positions would be a wrong answer, and panicking is not an option in
    /// this crate. Size the slice with [`scratch_len`].
    #[error("attention: scratch length {len}, need {need} f32 for this geometry")]
    ScratchTooShort {
        /// Length of the slice the caller passed.
        len: usize,
        /// Length [`scratch_len`] requires for this call.
        need: usize,
    },
}

/// The f32 scratch one attention call needs at a given geometry:
///
/// ```text
/// scratch_len = (n_q_heads / n_kv_heads) * max_positions + 2 * head_dim
/// ```
///
/// The first term is the **group-major score block**: one contiguous
/// `max_positions`-long run per query head in the GQA group, so [`softmax`]
/// always receives a contiguous slice and never a strided view (a strided
/// softmax would be a different reduction, and this kernel's output is
/// pinned to the bit). The second term is the two `head_dim`-long f32
/// conversion buffers — one K row and one V row, each widened from f16 once
/// per position and then reused by the whole group.
///
/// At the v0 pin (32 q heads : 4 kv heads, `head_dim` 128, `max_positions`
/// 4096, so `group` = 8) that is `8 * 4096 + 2 * 128 = 33_024` f32 =
/// **129 KiB**: 128 KiB of scores and 1 KiB of conversion buffers. A caller
/// carving per-shard scratch out of the prefill arena sizes each shard's
/// slice from this function.
///
/// Saturating and division-safe, so a nonsensical geometry returns a length
/// rather than panicking: `n_kv_heads == 0` contributes no score block, and
/// an overflowing product saturates (the length check in
/// [`attention_at_in`] then rejects the call).
pub fn scratch_len(
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_positions: usize,
) -> usize {
    let group = n_q_heads.checked_div(n_kv_heads).unwrap_or(0);
    scratch_len_of(group, head_dim, max_positions)
}

/// [`scratch_len`] from an already-derived GQA group size — the one place
/// the carve arithmetic is written down.
fn scratch_len_of(group: usize, head_dim: usize, max_positions: usize) -> usize {
    group
        .saturating_mul(max_positions)
        .saturating_add(head_dim.saturating_mul(2))
}

/// Reusable owning scratch for [`decode_attention`] and [`attention_at`].
///
/// Holds the whole carve [`scratch_len`] describes: the group-major score
/// block plus the two conversion buffers. The buffer is sized on the first
/// call from the cache's *capacity* rather than that call's position count,
/// so it reaches its final length immediately and is then reused as-is for
/// every later call, however long the sequence grows — one allocation, then
/// never again on the decode or prefill path. Every entry is overwritten
/// before it is read, so a short call after a long one can never pick up a
/// stale score.
///
/// Wave-2 shard code wants scratch carved from the prefill arena instead of
/// the heap; that is what [`attention_at_in`] and [`decode_attention_in`]
/// are for. This type is the owning convenience wrapper over them.
#[derive(Debug, Default)]
pub struct AttentionScratch {
    buf: Vec<f32>,
}

impl AttentionScratch {
    /// An empty scratch; the first attention call sizes it.
    pub fn new() -> Self {
        Self::default()
    }

    /// A scratch with `slots` f32 reserved up front.
    ///
    /// This is a *lower bound*, not the final length: the buffer holds
    /// [`scratch_len`]'s whole carve, which at the v0 pin is `group` (8)
    /// times the position count plus the conversion buffers. Passing a
    /// position count therefore still costs one growth on the first call.
    /// To allocate exactly once, pass
    /// `scratch_len(n_q_heads, n_kv_heads, head_dim, cache.capacity())`.
    pub fn with_capacity(slots: usize) -> Self {
        Self {
            buf: Vec::with_capacity(slots),
        }
    }

    /// The buffer resized to `len` (reallocates only past the high-water
    /// mark).
    fn buf_mut(&mut self, len: usize) -> &mut [f32] {
        self.buf.resize(len, 0.0);
        &mut self.buf[..len]
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
    attention_owned(q, cache, layer, None, scale, scratch, out, false)
}

/// [`decode_attention`] with the kernel path pinned instead of probed.
///
/// `force_scalar = true` runs the scalar reference; `false` is exactly
/// [`decode_attention`], i.e. the runtime AVX2 + F16C dispatch. The two are
/// bit-identical — that is the gate, not a tolerance — so production code
/// calls [`decode_attention`] and this exists only so tests and benches can
/// drive both implementations in one process, the convention `gemv` and
/// `quants::avx2` already use.
///
/// # Errors
///
/// Exactly [`decode_attention`]'s.
pub fn decode_attention_dispatch(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    scale: f32,
    scratch: &mut AttentionScratch,
    out: &mut [f32],
    force_scalar: bool,
) -> Result<(), AttentionError> {
    attention_owned(q, cache, layer, None, scale, scratch, out, force_scalar)
}

/// [`decode_attention`] against caller-provided scratch instead of an owning
/// [`AttentionScratch`].
///
/// Same arithmetic, same result, bit for bit — only where the scores and the
/// f16-to-f32 conversion buffers live changes. Size `scratch` with
/// [`scratch_len`]; anything longer is accepted and the tail ignored, so one
/// arena carve sized at the cache's capacity serves every call.
///
/// # Errors
///
/// Everything [`decode_attention`] returns, plus
/// [`AttentionError::ScratchTooShort`] when `scratch` is smaller than
/// [`scratch_len`] requires. `out` is untouched on error.
pub fn decode_attention_in(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    scale: f32,
    scratch: &mut [f32],
    out: &mut [f32],
) -> Result<(), AttentionError> {
    attention_borrowed(q, cache, layer, None, scale, scratch, out, false)
}

/// [`decode_attention_in`] with the kernel path pinned instead of probed; see
/// [`decode_attention_dispatch`].
///
/// # Errors
///
/// Exactly [`decode_attention_in`]'s.
pub fn decode_attention_in_dispatch(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    scale: f32,
    scratch: &mut [f32],
    out: &mut [f32],
    force_scalar: bool,
) -> Result<(), AttentionError> {
    attention_borrowed(q, cache, layer, None, scale, scratch, out, force_scalar)
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
    attention_owned(q, cache, layer, Some(positions), scale, scratch, out, false)
}

/// [`attention_at`] with the kernel path pinned instead of probed; see
/// [`decode_attention_dispatch`].
///
/// # Errors
///
/// Exactly [`attention_at`]'s.
#[allow(clippy::too_many_arguments)]
pub fn attention_at_dispatch(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    positions: usize,
    scale: f32,
    scratch: &mut AttentionScratch,
    out: &mut [f32],
    force_scalar: bool,
) -> Result<(), AttentionError> {
    attention_owned(
        q,
        cache,
        layer,
        Some(positions),
        scale,
        scratch,
        out,
        force_scalar,
    )
}

/// [`attention_at`] against caller-provided scratch instead of an owning
/// [`AttentionScratch`].
///
/// This is the entry point for sharded prefill: rows are independent, so a
/// shard per compute thread needs a scratch per shard, and the project rule
/// is that prefill costs no additional bytes against the memory budget — so
/// the scratch comes from the prefill arena, which cannot hand out a `Vec`.
/// Size the carve with [`scratch_len`]; anything longer is accepted and the
/// tail ignored.
///
/// The scores and the conversion buffers are the only state a call keeps, so
/// two calls with disjoint scratch slices and disjoint `out` slices are
/// independent — and each one is bit-identical to the single-threaded
/// [`attention_at`] on the same row.
///
/// # Errors
///
/// Everything [`attention_at`] returns, plus
/// [`AttentionError::ScratchTooShort`] when `scratch` is smaller than
/// [`scratch_len`] requires. `out` is untouched on error.
pub fn attention_at_in(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    positions: usize,
    scale: f32,
    scratch: &mut [f32],
    out: &mut [f32],
) -> Result<(), AttentionError> {
    attention_borrowed(q, cache, layer, Some(positions), scale, scratch, out, false)
}

/// [`attention_at_in`] with the kernel path pinned instead of probed; see
/// [`decode_attention_dispatch`].
///
/// # Errors
///
/// Exactly [`attention_at_in`]'s.
#[allow(clippy::too_many_arguments)]
pub fn attention_at_in_dispatch(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    positions: usize,
    scale: f32,
    scratch: &mut [f32],
    out: &mut [f32],
    force_scalar: bool,
) -> Result<(), AttentionError> {
    attention_borrowed(
        q,
        cache,
        layer,
        Some(positions),
        scale,
        scratch,
        out,
        force_scalar,
    )
}

/// The validated geometry of one attention call: what the shape checks
/// establish, computed once and handed to the body.
struct Plan {
    head_dim: usize,
    kv_dim: usize,
    /// Query heads per kv head, `n_q_heads / n_kv_heads` (>= 1).
    group: usize,
    /// Positions actually attended over — the causal limit.
    positions: usize,
}

impl Plan {
    /// The scratch this exact call needs.
    fn scratch_len(&self) -> usize {
        scratch_len_of(self.group, self.head_dim, self.positions)
    }
}

/// Every shape and range check, in the order the entry points promise: `q`,
/// then the GQA grouping, then `out`, then the layer, then `positions`.
/// Nothing is written before this returns `Ok`.
fn plan(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    limit: Option<usize>,
    out: &[f32],
) -> Result<Plan, AttentionError> {
    let head_dim = cache.head_dim();
    let n_kv_heads = cache.n_kv_heads();

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
    Ok(Plan {
        head_dim,
        kv_dim: cache.kv_dim(),
        group: n_q_heads / n_kv_heads,
        positions,
    })
}

/// The owning-scratch path: validate, size the `Vec`, run the body.
#[allow(clippy::too_many_arguments)]
fn attention_owned(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    limit: Option<usize>,
    scale: f32,
    scratch: &mut AttentionScratch,
    out: &mut [f32],
    force_scalar: bool,
) -> Result<(), AttentionError> {
    let plan = plan(q, cache, layer, limit, out)?;
    // Size from the cache's capacity, not this call's `positions`: capacity
    // is the high-water mark of every call this cache can ever serve, so the
    // buffer is allocated once on the first call and never grown again as
    // the sequence advances. `.max` is belt and braces — `positions <=
    // cache.len(layer) <= cache.capacity()` already holds here.
    let need = scratch_len_of(
        plan.group,
        plan.head_dim,
        cache.capacity().max(plan.positions),
    );
    let buf = scratch.buf_mut(need);
    attention_body(q, cache, layer, &plan, scale, buf, out, force_scalar)
}

/// The borrowed-scratch path: validate, check the carve fits, run the body.
#[allow(clippy::too_many_arguments)]
fn attention_borrowed(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    limit: Option<usize>,
    scale: f32,
    scratch: &mut [f32],
    out: &mut [f32],
    force_scalar: bool,
) -> Result<(), AttentionError> {
    let plan = plan(q, cache, layer, limit, out)?;
    let need = plan.scratch_len();
    if scratch.len() < need {
        return Err(AttentionError::ScratchTooShort {
            len: scratch.len(),
            need,
        });
    }
    attention_body(q, cache, layer, &plan, scale, scratch, out, force_scalar)
}

/// The one attention body, kv-head outer. `scratch` is at least
/// `plan.scratch_len()` long; anything past the carve is ignored.
///
/// The causal limit lives entirely in `plan.positions`: it is the length of
/// each per-head score run and the trip count of both `t` loops, so every
/// later row of the layer is absent from the scores, from the softmax
/// normalizer, and from the V sum — not zero-weighted, absent. That is what
/// makes the limited call bit-identical to an unlimited call against a
/// `positions`-row cache, and it is why there is no online or flash-style
/// rescaled softmax here.
#[allow(clippy::too_many_arguments)]
fn attention_body(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    plan: &Plan,
    scale: f32,
    scratch: &mut [f32],
    out: &mut [f32],
    force_scalar: bool,
) -> Result<(), AttentionError> {
    let &Plan {
        head_dim,
        kv_dim,
        group,
        positions,
    } = plan;
    // `plan` proved `positions <= cache.len(layer)`, and `k_layer`/`v_layer`
    // return exactly `len(layer)` rows of `kv_dim`, so these are the first
    // `positions` rows and the multiply cannot overflow. Bounding the planes
    // once here is what lets both the scalar and the AVX2 phases index rows
    // directly instead of re-deriving the limit.
    let k_rows = &cache.k_layer(layer)?[..positions * kv_dim];
    let v_rows = &cache.v_layer(layer)?[..positions * kv_dim];

    // The carve: a group-major score block (one contiguous `positions`-long
    // run per query head in the group, so `softmax` gets a contiguous slice)
    // then one K and one V conversion row. Both splits are in range because
    // the callers checked `scratch.len() >= scratch_len_of(..)`.
    let (scores, conv) = scratch.split_at_mut(group * positions);
    let (kbuf, vbuf) = conv.split_at_mut(head_dim);
    let vbuf = &mut vbuf[..head_dim];

    let group_dim = group * head_dim;
    for (kv_head, (q_group, out_group)) in q
        .chunks_exact(group_dim)
        .zip(out.chunks_exact_mut(group_dim))
        .enumerate()
    {
        let col = kv_head * head_dim;

        // Phase 1 — scores for the whole group. K[t, kv_head] is widened
        // once per element here and read by all `group` query heads.
        qk_scores(
            q_group,
            k_rows,
            kv_dim,
            col,
            head_dim,
            group,
            positions,
            scale,
            kbuf,
            scores,
            force_scalar,
        );

        for run in scores.chunks_exact_mut(positions) {
            softmax(run)?;
        }

        // Phase 2 — out[h] = sum_t scores[g][t] * V[t, kv_head], with V
        // widened once per element and every output element still
        // accumulated over `t` ascending.
        out_group.fill(0.0);
        v_reduce(
            v_rows,
            kv_dim,
            col,
            head_dim,
            positions,
            scores,
            vbuf,
            out_group,
            force_scalar,
        );
    }
    Ok(())
}

/// Phase 1 for one kv head, dispatched: AVX2 + F16C when the CPU has both and
/// the geometry fits, else the scalar reference.
///
/// Both write the same bits (see the module docs and [`x86`]), so this is a
/// performance choice and never a numerical one. `force_scalar` is threaded
/// rather than read from the environment so tests and benches can drive both
/// paths in one process.
#[allow(clippy::too_many_arguments)]
fn qk_scores(
    q_group: &[f32],
    k_rows: &[u16],
    kv_dim: usize,
    col: usize,
    head_dim: usize,
    group: usize,
    positions: usize,
    scale: f32,
    kbuf: &mut [f32],
    scores: &mut [f32],
    force_scalar: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if !force_scalar && avx2_f16c_path_covers(head_dim) {
        // SAFETY: AVX2 and F16C presence was checked at runtime just above,
        // and `head_dim <= MAX_SIMD_HEAD_DIM` bounds the stack buffers the
        // body carves. `k_rows` holds exactly `positions` rows of `kv_dim`,
        // `col + head_dim <= kv_dim` because `col = kv_head * head_dim` with
        // `kv_head < n_kv_heads`, `q_group` holds `group * head_dim` f32 and
        // `scores` holds `group * positions`.
        unsafe {
            x86::qk_scores(
                q_group, k_rows, kv_dim, col, head_dim, group, positions, scale, scores,
            );
        }
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = force_scalar;
    qk_scores_scalar(
        q_group, k_rows, kv_dim, col, head_dim, positions, scale, kbuf, scores,
    );
}

/// The scalar reference for phase 1: `scores[g][t] = scale * (q_g . K[t])`,
/// accumulated over `i` ascending in one f32 with a separate multiply and
/// add, scaled once at the end.
///
/// This is the arithmetic every other path is pinned to, so it is written
/// once and never specialized.
#[allow(clippy::too_many_arguments)]
fn qk_scores_scalar(
    q_group: &[f32],
    k_rows: &[u16],
    kv_dim: usize,
    col: usize,
    head_dim: usize,
    positions: usize,
    scale: f32,
    kbuf: &mut [f32],
    scores: &mut [f32],
) {
    for (t, k_row) in k_rows.chunks_exact(kv_dim).take(positions).enumerate() {
        let k_th = &k_row[col..col + head_dim];
        for (dst, &bits) in kbuf.iter_mut().zip(k_th) {
            *dst = f16_to_f32(bits);
        }
        for (q_h, run) in q_group
            .chunks_exact(head_dim)
            .zip(scores.chunks_exact_mut(positions))
        {
            let mut acc = 0.0f32;
            for (&qv, &kv) in q_h.iter().zip(kbuf.iter()) {
                acc += qv * kv;
            }
            run[t] = scale * acc;
        }
    }
}

/// Phase 2 for one kv head, dispatched; see [`qk_scores`]. `out_group` is
/// already zeroed by the caller.
#[allow(clippy::too_many_arguments)]
fn v_reduce(
    v_rows: &[u16],
    kv_dim: usize,
    col: usize,
    head_dim: usize,
    positions: usize,
    scores: &[f32],
    vbuf: &mut [f32],
    out_group: &mut [f32],
    force_scalar: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if !force_scalar && avx2_f16c_path_covers(head_dim) {
        // SAFETY: AVX2 and F16C presence was checked at runtime just above.
        // `v_rows` holds exactly `positions` rows of `kv_dim`,
        // `col + head_dim <= kv_dim`, `vbuf` is exactly `head_dim` f32,
        // `scores` is `group * positions` and `out_group` `group * head_dim`.
        unsafe {
            x86::v_reduce(
                v_rows, kv_dim, col, head_dim, positions, scores, vbuf, out_group,
            );
        }
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = force_scalar;
    v_reduce_scalar(
        v_rows, kv_dim, col, head_dim, positions, scores, vbuf, out_group,
    );
}

/// The scalar reference for phase 2: `out[h][i] += w_t * V[t][i]`, every
/// output element accumulated over `t` ascending.
#[allow(clippy::too_many_arguments)]
fn v_reduce_scalar(
    v_rows: &[u16],
    kv_dim: usize,
    col: usize,
    head_dim: usize,
    positions: usize,
    scores: &[f32],
    vbuf: &mut [f32],
    out_group: &mut [f32],
) {
    for (t, v_row) in v_rows.chunks_exact(kv_dim).take(positions).enumerate() {
        let v_th = &v_row[col..col + head_dim];
        for (dst, &bits) in vbuf.iter_mut().zip(v_th) {
            *dst = f16_to_f32(bits);
        }
        for (out_h, run) in out_group
            .chunks_exact_mut(head_dim)
            .zip(scores.chunks_exact(positions))
        {
            let w = run[t];
            for (o, &vv) in out_h.iter_mut().zip(vbuf.iter()) {
                *o += w * vv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::quants::f32_to_f16;
    use std::io::Write;

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
    ///
    /// The reservation is now the whole carve, not just the positions: at
    /// 32:4 heads, head_dim 8 and 17 positions that is `8 * 17 + 2 * 8 = 152`
    /// f32. Constructed at that size, the buffer must never allocate again —
    /// pointer identity and capacity are pinned across grow / shrink /
    /// regrow, which is the one thing this test exists to guarantee.
    #[test]
    fn scratch_reuses_allocation() {
        let need = scratch_len(32, 4, 8, 17);
        assert_eq!(need, 8 * 17 + 2 * 8);
        let mut scratch = AttentionScratch::with_capacity(need);
        let ptr = scratch.buf.as_ptr();
        assert_eq!(scratch.buf_mut(need).len(), need);
        assert_eq!(scratch.buf.as_ptr(), ptr);
        assert_eq!(scratch.buf_mut(scratch_len(32, 4, 8, 3)).len(), 8 * 3 + 16);
        assert_eq!(scratch.buf.as_ptr(), ptr);
        assert_eq!(scratch.buf_mut(need).len(), need);
        assert_eq!(scratch.buf.as_ptr(), ptr);
        assert_eq!(scratch.buf.capacity(), need);
    }

    /// The same guarantee through the public entry point: a scratch built at
    /// `scratch_len(.., cache.capacity())` allocates once and then holds its
    /// pointer across a growing sequence, because the first call sizes it
    /// from the cache's capacity rather than that call's position count.
    #[test]
    fn scratch_allocates_once_across_a_growing_sequence() {
        let (n_kv, n_q, head_dim, cap) = (2usize, 8usize, 16usize, 32usize);
        let mut rng = Rng::new(0x00A1_10C1);
        let (cache, _rows) = random_cache(&mut rng, 1, n_kv, head_dim, cap);
        let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
        let mut out = vec![0.0f32; q.len()];
        let scale = 1.0 / (head_dim as f32).sqrt();

        let need = scratch_len(n_q, n_kv, head_dim, cap);
        let mut scratch = AttentionScratch::with_capacity(need);
        let ptr = scratch.buf.as_ptr();
        for positions in 1..=cap {
            attention_at(&q, &cache, 0, positions, scale, &mut scratch, &mut out).unwrap();
            assert_eq!(scratch.buf.as_ptr(), ptr, "reallocated at {positions}");
            assert_eq!(scratch.buf.len(), need);
        }
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

    /// The pre-restructure kernel body, copied verbatim from the head-major
    /// loop nest this wave replaced: query head outer, `kv_head = h / group`,
    /// and the f16->f32 conversion inline in both reduction loops, so every
    /// K and V element is re-converted once per query head in its GQA group.
    /// The causal limit is the length of `scores`, which both reductions zip
    /// against — exactly as it was.
    ///
    /// This exists only as the reference for
    /// `restructured_kernel_is_bit_identical_to_head_major_reference`. It is
    /// deliberately *not* factored against the live kernel: the whole point
    /// is that it is the old arithmetic, written the old way.
    fn head_major_reference(
        q: &[f32],
        cache: &KvCache,
        layer: usize,
        positions: usize,
        scale: f32,
        out: &mut [f32],
    ) {
        let head_dim = cache.head_dim();
        let n_kv_heads = cache.n_kv_heads();
        let kv_dim = cache.kv_dim();
        let n_q_heads = q.len() / head_dim;
        let group = n_q_heads / n_kv_heads;
        let k_plane = cache.k_layer(layer).unwrap();
        let v_plane = cache.v_layer(layer).unwrap();
        let mut scores = vec![0.0f32; positions];

        for (h, out_h) in out.chunks_exact_mut(head_dim).enumerate() {
            let kv_head = h / group;
            let col = kv_head * head_dim;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];

            for (row, score) in k_plane.chunks_exact(kv_dim).zip(scores.iter_mut()) {
                let k_th = &row[col..col + head_dim];
                let mut acc = 0.0f32;
                for (&qv, &kb) in q_h.iter().zip(k_th) {
                    acc += qv * f16_to_f32(kb);
                }
                *score = scale * acc;
            }

            softmax(&mut scores).unwrap();

            out_h.fill(0.0);
            for (row, &w) in v_plane.chunks_exact(kv_dim).zip(scores.iter()) {
                let v_th = &row[col..col + head_dim];
                for (o, &vb) in out_h.iter_mut().zip(v_th) {
                    *o += w * f16_to_f32(vb);
                }
            }
        }
    }

    /// Gate 10 (phase 7 wave 1, the central claim of the restructure): the
    /// kv-head-outer kernel, which widens each K and V element from f16 once
    /// per position instead of once per query head, is **bit-identical** to
    /// the head-major nest it replaced.
    ///
    /// That is the whole argument for the hoist. `f16_to_f32` is an exact
    /// widening, so a buffered element is the value the inline conversion
    /// produced; `h = kv_head * group + g` visits the same query heads as
    /// `kv_head = h / group`; the QK dot still runs over `i` ascending in one
    /// f32 with the scale applied once at the end; the V sum still runs over
    /// `t` ascending per output element. Nothing reassociates, so `to_bits()`
    /// equality must hold with zero tolerance. Introducing an FMA, folding
    /// the scale per element, transposing the reduction, or letting softmax
    /// see anything but a contiguous per-head run all break this test.
    ///
    /// Coverage: all four `LIMITED_GEOMETRIES` (GQA groups 8, 1, 8 and 2 —
    /// group 1 is the degenerate case where the group-major score block is a
    /// single run) plus the real v0 pin (32 q heads : 4 kv heads, head_dim
    /// 128) at a prime full length; every layer; position counts 1, the
    /// primes up to the geometry's length, and the full length; and all four
    /// entry points, owning and borrowed, limited and unlimited.
    #[test]
    fn restructured_kernel_is_bit_identical_to_head_major_reference() {
        let mut rng = Rng::new(0x0401_57ED);
        let mut geometries = LIMITED_GEOMETRIES.to_vec();
        geometries.push((1, 4, 32, 128, 67));

        for (n_layers, n_kv, n_q, head_dim, n_pos) in geometries {
            let scale = 1.0 / (head_dim as f32).sqrt();
            let (cache, _rows) = random_cache(&mut rng, n_layers, n_kv, head_dim, n_pos);
            let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
            let mut owned = AttentionScratch::new();

            let mut lengths: Vec<usize> = [1usize, 2, 3, 5, 7, 11, 13, 31, 61]
                .into_iter()
                .filter(|&p| p < n_pos)
                .collect();
            lengths.push(n_pos);

            for layer in 0..n_layers {
                for &positions in &lengths {
                    let mut want = vec![0.0f32; q.len()];
                    head_major_reference(&q, &cache, layer, positions, scale, &mut want);

                    // Owning scratch, limited form.
                    let mut got = vec![0.0f32; q.len()];
                    attention_at(&q, &cache, layer, positions, scale, &mut owned, &mut got)
                        .unwrap();

                    // Borrowed scratch, sized exactly by `scratch_len`.
                    let need = scratch_len(n_q, n_kv, head_dim, positions);
                    let mut carve = vec![0.0f32; need];
                    let mut got_in = vec![0.0f32; q.len()];
                    attention_at_in(&q, &cache, layer, positions, scale, &mut carve, &mut got_in)
                        .unwrap();

                    // The unlimited forms, where the limit is the whole layer.
                    let mut got_decode = vec![0.0f32; q.len()];
                    let mut got_decode_in = vec![0.0f32; q.len()];
                    if positions == n_pos {
                        decode_attention(&q, &cache, layer, scale, &mut owned, &mut got_decode)
                            .unwrap();
                        decode_attention_in(
                            &q,
                            &cache,
                            layer,
                            scale,
                            &mut carve,
                            &mut got_decode_in,
                        )
                        .unwrap();
                    } else {
                        got_decode.copy_from_slice(&got);
                        got_decode_in.copy_from_slice(&got);
                    }

                    for (i, &w) in want.iter().enumerate() {
                        let label = format!(
                            "geometry ({n_layers}, {n_kv}, {n_q}, {head_dim}) layer {layer} \
                             positions {positions} elem {i}"
                        );
                        assert_eq!(
                            got[i].to_bits(),
                            w.to_bits(),
                            "{label}: attention_at {:e} vs head-major reference {w:e}",
                            got[i]
                        );
                        assert_eq!(
                            got_in[i].to_bits(),
                            w.to_bits(),
                            "{label}: attention_at_in {:e} vs head-major reference {w:e}",
                            got_in[i]
                        );
                        assert_eq!(got_decode[i].to_bits(), w.to_bits(), "{label}: decode");
                        assert_eq!(
                            got_decode_in[i].to_bits(),
                            w.to_bits(),
                            "{label}: decode_in"
                        );
                    }
                }
            }
        }
    }

    /// `scratch_len` is the pinned carve arithmetic — a later lane sizes an
    /// arena from it, so the formula is a contract, not an implementation
    /// detail. It is also total: no division by zero, no overflow panic.
    #[test]
    fn scratch_len_pins_the_carve() {
        // v0 pin: 8 * 4096 scores + 2 * 128 conversion = 33 024 f32 = 129 KiB.
        assert_eq!(scratch_len(32, 4, 128, 4096), 33_024);
        assert_eq!(scratch_len(32, 4, 128, 4096) * 4, 132_096);
        // Group 1 (MHA): the score block is a single positions-long run.
        assert_eq!(scratch_len(8, 8, 64, 100), 100 + 128);
        // Group 8 over one kv head (MQA).
        assert_eq!(scratch_len(8, 1, 8, 7), 8 * 7 + 16);
        // Total on nonsense: no panic, no wraparound.
        assert_eq!(scratch_len(4, 0, 8, 10), 16);
        assert_eq!(scratch_len(usize::MAX, 1, 1, usize::MAX), usize::MAX);
        assert_eq!(scratch_len(2, 1, usize::MAX, 4), usize::MAX);
    }

    /// A short borrowed scratch is a typed error: not a panic, and not a
    /// silent truncation to however many positions happen to fit — that
    /// would be a wrong answer wearing a right answer's shape. Shape errors
    /// still take priority over it, and no error path writes `out`.
    #[test]
    fn attention_in_typed_errors_on_short_scratch() {
        let (n_kv, n_q, head_dim, n_pos) = (2usize, 8usize, 16usize, 5usize);
        let mut rng = Rng::new(0x05C2_47C4);
        let (cache, _rows) = random_cache(&mut rng, 1, n_kv, head_dim, n_pos);
        let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![7.0f32; q.len()];

        let need = scratch_len(n_q, n_kv, head_dim, n_pos);
        assert_eq!(need, 4 * n_pos + 2 * head_dim);

        // One short, empty, and short by only the conversion buffers.
        for len in [0usize, 1, need - 1, 4 * n_pos] {
            let mut carve = vec![0.0f32; len];
            assert_eq!(
                attention_at_in(&q, &cache, 0, n_pos, scale, &mut carve, &mut out).unwrap_err(),
                AttentionError::ScratchTooShort { len, need }
            );
            assert_eq!(
                decode_attention_in(&q, &cache, 0, scale, &mut carve, &mut out).unwrap_err(),
                AttentionError::ScratchTooShort { len, need }
            );
        }

        // A shorter row needs a shorter carve, and the error says so.
        let mut carve = vec![0.0f32; 4 * 2];
        assert_eq!(
            attention_at_in(&q, &cache, 0, 2, scale, &mut carve, &mut out).unwrap_err(),
            AttentionError::ScratchTooShort {
                len: 8,
                need: scratch_len(n_q, n_kv, head_dim, 2),
            }
        );

        // Shape and range checks come first: a bad `q` is a `q` error even
        // with an empty scratch, and `positions` past the end is still the
        // cache's error.
        let mut empty: Vec<f32> = Vec::new();
        assert_eq!(
            attention_at_in(&q[..3], &cache, 0, n_pos, scale, &mut empty, &mut out).unwrap_err(),
            AttentionError::QLenIndivisible { q_len: 3, head_dim }
        );
        assert_eq!(
            attention_at_in(&q, &cache, 0, n_pos + 1, scale, &mut empty, &mut out).unwrap_err(),
            AttentionError::Kv(KvError::PositionOutOfRange {
                layer: 0,
                pos: n_pos,
                len: n_pos,
            })
        );

        // Nothing above wrote to `out`.
        assert!(out.iter().all(|&x| x == 7.0));

        // Exactly `scratch_len` is enough, and a longer carve is accepted
        // with the tail ignored — both bit-identical to the owning form.
        let mut owned = AttentionScratch::new();
        let mut want = vec![0.0f32; q.len()];
        attention_at(&q, &cache, 0, n_pos, scale, &mut owned, &mut want).unwrap();
        for len in [need, need + 1, need * 3] {
            let mut carve = vec![-1.0f32; len];
            let mut got = vec![0.0f32; q.len()];
            attention_at_in(&q, &cache, 0, n_pos, scale, &mut carve, &mut got).unwrap();
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "carve {len} elem {i}");
            }
        }
    }

    /// Group 1 (`n_q_heads == n_kv_heads`, plain MHA) is the degenerate case
    /// of the group-major score layout: one run, no stride, and each query
    /// head reads its own kv head. It is easy to get wrong by assuming the
    /// group is the outer stride, so it gets its own gate — against the
    /// head-major reference, through every entry point, and with the head
    /// mapping checked directly by giving each kv head a distinct winner.
    #[test]
    fn group_of_one_maps_each_query_head_to_its_own_kv_head() {
        let (n_kv, n_q, head_dim, n_pos) = (4usize, 4usize, 8usize, 6usize);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut rng = Rng::new(0x0612_00F1);
        let (cache, _rows) = random_cache(&mut rng, 2, n_kv, head_dim, n_pos);
        let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);

        assert_eq!(
            scratch_len(n_q, n_kv, head_dim, n_pos),
            n_pos + 2 * head_dim
        );

        let mut owned = AttentionScratch::new();
        for layer in 0..2 {
            for positions in 1..=n_pos {
                let mut want = vec![0.0f32; q.len()];
                head_major_reference(&q, &cache, layer, positions, scale, &mut want);

                let mut got = vec![0.0f32; q.len()];
                attention_at(&q, &cache, layer, positions, scale, &mut owned, &mut got).unwrap();
                let mut carve = vec![0.0f32; scratch_len(n_q, n_kv, head_dim, positions)];
                let mut got_in = vec![0.0f32; q.len()];
                attention_at_in(&q, &cache, layer, positions, scale, &mut carve, &mut got_in)
                    .unwrap();

                for (i, &w) in want.iter().enumerate() {
                    assert_eq!(got[i].to_bits(), w.to_bits(), "layer {layer} elem {i}");
                    assert_eq!(got_in[i].to_bits(), w.to_bits(), "layer {layer} elem {i}");
                }
            }
        }

        // Head mapping, independent of the reference: kv head j wins at
        // position j and carries marker 10j + t, so query head h (== kv head
        // h at group 1) must come back with marker(h, h).
        let mut marked = KvCache::new(1, n_kv, head_dim, n_kv).unwrap();
        let marker = |j: usize, t: usize| (10 * j + t) as f32 * 0.01;
        for t in 0..n_kv {
            let mut k = vec![0.0f32; n_kv * head_dim];
            let mut v = vec![0.0f32; n_kv * head_dim];
            for j in 0..n_kv {
                let sign = if t == j { 1.0f32 } else { -1.0 };
                for i in 0..head_dim {
                    k[j * head_dim + i] = sign * 8.0 / head_dim as f32;
                    v[j * head_dim + i] = marker(j, t);
                }
            }
            marked.append(0, &k, &v).unwrap();
        }
        let ones = vec![1.0f32; n_q * head_dim];
        let mut out = vec![0.0f32; ones.len()];
        let mut carve = vec![0.0f32; scratch_len(n_q, n_kv, head_dim, n_kv)];
        decode_attention_in(&ones, &marked, 0, 1.0, &mut carve, &mut out).unwrap();
        for h in 0..n_q {
            for i in 0..head_dim {
                let got = out[h * head_dim + i];
                let want = marker(h, h);
                assert!(
                    (got - want).abs() < 1e-3,
                    "q head {h} elem {i}: got {got}, want kv head {h} marker {want}"
                );
            }
        }
    }

    // ----------------------------------------- AVX2 + F16C vs scalar (wave 2)

    /// The device against a silent single-path pass, ported from `gemv`
    /// (`gemv.rs`, `KernelPaths` / `kernel_paths`) with this kernel's probe
    /// swapped in.
    ///
    /// Sweeping `force_scalar` over `[false, true]` only selects between two
    /// implementations when the host has AVX2 **and** F16C; on a host without
    /// them `false` runs the very same scalar kernel as `true`, and both
    /// iterations prove nothing while still looking like a two-path proof. So
    /// the sweep is narrowed to the paths that exist and the shortfall is
    /// announced rather than asserted away — "scalar otherwise" is a real
    /// configuration of this module, so a bare
    /// `assert!(avx2_f16c_available())` would turn a portable gate into one
    /// that cannot pass on an ARM, macOS, or pre-Ivy-Bridge box.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum KernelPaths {
        /// AVX2+F16C is live: `[false, true]` really is two kernels.
        Both,
        /// Scalar only. The AVX2-vs-scalar half of the claim is not tested.
        ScalarOnly,
    }

    impl KernelPaths {
        /// The `force_scalar` flags worth sweeping on this host.
        fn flags(self) -> &'static [bool] {
            match self {
                KernelPaths::Both => &[false, true],
                KernelPaths::ScalarOnly => &[true],
            }
        }
    }

    /// Resolve [`KernelPaths`] for this host, announcing `claim` as untested
    /// when only one path exists.
    ///
    /// The notice goes straight to `stderr` rather than through `eprintln!`
    /// because libtest captures the print macros and replays them only for
    /// *failing* tests; a skip nobody can see on a green run is precisely the
    /// silent single-path pass this exists to prevent.
    ///
    /// # Panics
    ///
    /// On an x86_64 host whose CPU reports AVX2 and F16C while
    /// [`avx2_f16c_available`] does not. That is a dispatch bug, not an
    /// unsupported host: every attention call would quietly run the scalar
    /// kernel on a machine that has the vector one, and no test would say so.
    fn kernel_paths(claim: &str) -> KernelPaths {
        let dispatch = avx2_f16c_available();
        #[cfg(target_arch = "x86_64")]
        {
            let cpu = std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("f16c");
            assert_eq!(
                dispatch, cpu,
                "kernel dispatch disagrees with this x86_64 CPU: CPUID reports \
                 avx2+f16c = {cpu}, `avx2_f16c_available()` reports {dispatch}. \
                 Every attention call dispatches on the latter, so this is a \
                 live mis-dispatch, not an unsupported host."
            );
        }
        if dispatch {
            return KernelPaths::Both;
        }
        let mut err = std::io::stderr();
        let _ = writeln!(
            err,
            "\n\
             ##########################################################\n\
             # NOT PROVEN ON THIS HOST: {claim}\n\
             #\n\
             # This host has no AVX2+F16C, so `force_scalar = false`\n\
             # dispatches to the same scalar kernel as `force_scalar =\n\
             # true`. The scalar arithmetic was exercised in full; the\n\
             # AVX2-vs-scalar half of phase 7's bit-identity gate was NOT\n\
             # tested. A green run here is not evidence for it. Re-run the\n\
             # kernel tests on an AVX2+F16C machine before relying on it.\n\
             ##########################################################\n"
        );
        let _ = err.flush();
        KernelPaths::ScalarOnly
    }

    /// Geometries for the wave-2 dispatch gates: the four
    /// [`LIMITED_GEOMETRIES`] (GQA groups 8 / 1 / 8 / 2, `head_dim` 128 / 16 /
    /// 8 / 4), the real v0 pin at a length that exercises the position
    /// blocking *and* its tail (67 = 16 blocks of 4, then 3), and two odd
    /// `head_dim` geometries whose head slices cannot be better than 2-byte
    /// aligned.
    fn dispatch_geometries() -> Vec<(usize, usize, usize, usize, usize)> {
        let mut g = LIMITED_GEOMETRIES.to_vec();
        g.push((1, 4, 32, 128, 67));
        g.push((1, 1, 8, 9, 11));
        g.push((2, 3, 6, 9, 13));
        g
    }

    /// The position counts to sweep for a geometry: 1, exact multiples of the
    /// position block, one either side of them, primes, and the full length.
    fn dispatch_lengths(n_pos: usize) -> Vec<usize> {
        let mut lengths: Vec<usize> = [1usize, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 31, 61]
            .into_iter()
            .filter(|&p| p < n_pos)
            .collect();
        lengths.push(n_pos);
        lengths
    }

    /// Gate 11 (phase 7 wave 2, the central claim of the vectorization): the
    /// AVX2 + F16C path is **bit-identical** to the scalar reference, and both
    /// are bit-identical to the pre-wave-1 head-major nest.
    ///
    /// Every f32 the kernel produces is pinned with `to_bits()` and zero
    /// tolerance, through all four entry points, on all seven geometries, at
    /// every position count in [`dispatch_lengths`], on every layer. That is
    /// the whole argument for the vectorization: `vcvtph2ps` is an exact
    /// widening, the QK dot's eight lanes ride the GQA group (whose
    /// accumulators are already independent) so each lane still walks `i`
    /// ascending in one f32 with a separate multiply and add, and the V
    /// reduction's eight lanes ride `i` (elementwise) while `t` stays
    /// sequential. An `_mm256_fmadd_ps` anywhere, eight lanes over `i` in the
    /// dot, a horizontal tree, folding `scale` per element, or splitting the
    /// `t` axis all break this test on the first geometry.
    ///
    /// The comparison is also asserted to be a real two-path comparison: on a
    /// host with the instructions, every geometry here must be inside
    /// [`avx2_f16c_path_covers`]'s envelope, or `force_scalar = false` would
    /// silently run the scalar kernel a second time.
    #[test]
    fn simd_matches_scalar_bit_for_bit() {
        let paths = kernel_paths("the AVX2+F16C attention path vs the scalar reference");
        let mut rng = Rng::new(0x5119_D0A7);

        for (n_layers, n_kv, n_q, head_dim, n_pos) in dispatch_geometries() {
            if paths == KernelPaths::Both {
                assert!(
                    avx2_f16c_path_covers(head_dim),
                    "geometry ({n_layers}, {n_kv}, {n_q}, {head_dim}) is outside the AVX2 \
                     envelope, so sweeping `force_scalar` here would run the scalar kernel \
                     twice and report it as a two-path proof"
                );
            }
            let scale = 1.0 / (head_dim as f32).sqrt();
            let (cache, _rows) = random_cache(&mut rng, n_layers, n_kv, head_dim, n_pos);
            let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
            let mut owned = AttentionScratch::new();

            for layer in 0..n_layers {
                for &positions in &dispatch_lengths(n_pos) {
                    let mut want = vec![0.0f32; q.len()];
                    head_major_reference(&q, &cache, layer, positions, scale, &mut want);

                    let need = scratch_len(n_q, n_kv, head_dim, positions);
                    // One entry per swept flag, in `flags()` order.
                    let mut by_path: Vec<Vec<f32>> = Vec::new();

                    for &force_scalar in paths.flags() {
                        let mut got = vec![0.0f32; q.len()];
                        attention_at_dispatch(
                            &q,
                            &cache,
                            layer,
                            positions,
                            scale,
                            &mut owned,
                            &mut got,
                            force_scalar,
                        )
                        .unwrap();

                        let mut carve = vec![0.0f32; need];
                        let mut got_in = vec![0.0f32; q.len()];
                        attention_at_in_dispatch(
                            &q,
                            &cache,
                            layer,
                            positions,
                            scale,
                            &mut carve,
                            &mut got_in,
                            force_scalar,
                        )
                        .unwrap();

                        let mut got_decode = got.clone();
                        let mut got_decode_in = got.clone();
                        if positions == n_pos {
                            decode_attention_dispatch(
                                &q,
                                &cache,
                                layer,
                                scale,
                                &mut owned,
                                &mut got_decode,
                                force_scalar,
                            )
                            .unwrap();
                            decode_attention_in_dispatch(
                                &q,
                                &cache,
                                layer,
                                scale,
                                &mut carve,
                                &mut got_decode_in,
                                force_scalar,
                            )
                            .unwrap();
                        }

                        for (i, &w) in want.iter().enumerate() {
                            let label = format!(
                                "geometry ({n_layers}, {n_kv}, {n_q}, {head_dim}) layer \
                                 {layer} positions {positions} force_scalar {force_scalar} \
                                 elem {i}"
                            );
                            assert_eq!(
                                got[i].to_bits(),
                                w.to_bits(),
                                "{label}: attention_at_dispatch {:e} vs head-major \
                                 reference {w:e}",
                                got[i]
                            );
                            assert_eq!(got_in[i].to_bits(), w.to_bits(), "{label}: at_in");
                            assert_eq!(got_decode[i].to_bits(), w.to_bits(), "{label}: decode");
                            assert_eq!(
                                got_decode_in[i].to_bits(),
                                w.to_bits(),
                                "{label}: decode_in"
                            );
                        }
                        by_path.push(got);
                    }

                    // The pairwise statement, so a failure names the two
                    // paths rather than only the reference. `flags()` yields
                    // `[false, true]` here, so entry 0 is the AVX2 run and
                    // entry 1 the scalar one.
                    if paths == KernelPaths::Both {
                        let (simd, scalar) = (&by_path[0], &by_path[1]);
                        for (i, (&s, &r)) in simd.iter().zip(scalar).enumerate() {
                            assert_eq!(
                                s.to_bits(),
                                r.to_bits(),
                                "geometry ({n_layers}, {n_kv}, {n_q}, {head_dim}) layer \
                                 {layer} positions {positions} elem {i}: AVX2 {s:e} vs \
                                 scalar {r:e}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The alignment claim, end to end: an odd `head_dim` puts a head's K/V
    /// slice on a 2-byte boundary (the row stride `n_kv_heads * head_dim` is
    /// odd, and the column offset `kv_head * head_dim` is odd for odd
    /// `kv_head`) and the matching `out` and score sub-slices on 4-byte ones.
    /// Every wide load and store in the AVX2 path is an unaligned form, so
    /// the result must still be the scalar reference's bits exactly.
    ///
    /// `head_dim` 9 and 17 both leave a scalar tail after the wide loads, so
    /// the remainder path is covered at those offsets too.
    #[test]
    fn misaligned_head_slices_match_the_scalar_reference() {
        let paths = kernel_paths("the AVX2+F16C path on 2-byte-aligned head slices");
        let mut rng = Rng::new(0x0A11_6EDD);
        for &(n_kv, n_q, head_dim, n_pos) in &[
            (1usize, 8usize, 9usize, 14usize),
            (3, 6, 9, 11),
            (3, 3, 17, 9),
            (5, 5, 17, 13),
        ] {
            let scale = 1.0 / (head_dim as f32).sqrt();
            let (cache, _rows) = random_cache(&mut rng, 2, n_kv, head_dim, n_pos);
            let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
            let mut owned = AttentionScratch::new();

            for layer in 0..2 {
                for positions in 1..=n_pos {
                    let mut want = vec![0.0f32; q.len()];
                    attention_at_dispatch(
                        &q, &cache, layer, positions, scale, &mut owned, &mut want, true,
                    )
                    .unwrap();
                    let mut got = vec![0.0f32; q.len()];
                    attention_at_dispatch(
                        &q, &cache, layer, positions, scale, &mut owned, &mut got, false,
                    )
                    .unwrap();
                    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "({n_kv}, {n_q}, {head_dim}) layer {layer} positions \
                             {positions} elem {i}: AVX2 {g:e} vs scalar {w:e}"
                        );
                    }
                }
            }
        }
        // Announced, not asserted: on a host without the instructions the
        // loop above compared the scalar kernel with itself.
        let _ = paths;
    }
}
