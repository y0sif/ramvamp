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
//! element is converted from f16 once per kv head and then reused by every
//! query head in its GQA group (8 of them at the v0 pin) out of a
//! `head_dim`-long f32 buffer, instead of being re-converted per query head.
//! The head-major order this replaced re-converted every element once per
//! query head in the group, and phase 7's wave-0 bench measured that
//! conversion plus the serial f32 accumulate as the whole cost of the kernel
//! — 1.54 ns per element against only 0.17 GB/s of unique K+V bytes, i.e.
//! compute-bound, not memory-bound.
//!
//! **How often "once per kv head" really is.** On the scalar reference it is
//! literally once, for K and V, at every `group`. On the AVX2 path it is
//! literally once for V, and for K it holds only while `group <= 8`.
//! `x86::qk_scores` walks the group in chunks of `x86::LANES` (8) and the
//! whole position sweep — the K widening included — sits *inside* that chunk
//! loop, so K is widened `ceil(group / 8)` times per kv head. Counted in
//! `x86::widen_rows` at `n_kv = 1`, `head_dim = 128`, 16 positions, against
//! the unique element count: K is 1.00x for `group` 1-8, 2.00x for 9-16,
//! 3.00x for 17-24 and 4.00x for 25-32, while V is 1.00x throughout and the
//! scalar path is 1.00x throughout. **The v0 pin is `group = 8`, where the
//! ratio is exactly 1.00**, and so is the attention bench's geometry — so the
//! bench's "unique bytes == touched bytes" model and the 0.17 GB/s
//! compute-bound reading above are unaffected. Groups past 8 are covered by
//! the correctness sweep (`dispatch_geometries` carries 9, 12, 17 and 32) but
//! are not a v0 shape.
//!
//! The redundancy is deliberately **not** hoisted out of the chunk loop. Doing
//! so needs the position axis outer and the group axis inner, and then the
//! transposed query block — currently one chunk's worth, `LANES *
//! MAX_SIMD_HEAD_DIM` f32 of fixed stack — has to hold the *whole* group, a
//! size bounded by nothing in the geometry, or be re-transposed once per
//! position block. Both trade a bounded stack carve for an unbounded one or a
//! new per-position cost, to remove a factor that is 1 at every geometry v0
//! runs. The honest bound is worth more than the micro-optimization.
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
//! Sharding: the axis a compute pool fans this kernel across is the **kv
//! head** ([`decode_attention_kv_range_in`], [`attention_at_kv_range_in`]),
//! and it is the only axis that costs nothing. A partition of the kv heads
//! gives each kv head to exactly one shard, and a kv head's widening cost is
//! whatever the whole call already paid for it — once per element on the
//! scalar path and for V, `ceil(group / 8)` times for K on the AVX2 path,
//! 1.00x either way at the v0 pin (see the once-per-kv-head note above) — so
//! cutting there duplicates no conversion **relative to the whole call**,
//! whatever `group` is; the `group` query heads of one kv head are
//! contiguous and disjoint in both `q` and `out`, so a shard needs no gather,
//! no scatter and no permutation; and the AVX2 QK dot keeps all eight of its
//! lanes on the GQA group, which stays whole. Cutting the **query head** axis
//! instead — a strided slab of heads per shard — narrows the group, and the
//! group is exactly what those lanes ride. Measured on a 185H (warm, `n_kv`
//! 4, `head_dim` 128, 4096 positions, min of 9): the whole `group = 8` call
//! took 4293.8 µs, a `group = 2` slab 2659.0 µs, a `group = 1` slab 1981.1
//! µs. Eight `group = 1` slabs therefore total ~15.8 ms of CPU against 4.3 ms
//! for the one whole call (derived from those measurements) — seven of the
//! eight lanes carry zeros.
//!
//! `group` is consequently derived from the **full** `q` and never from the
//! range: a shard is handed the whole query vector and told *which* kv heads
//! to compute, not a narrowed view of it. Narrowing `q` would re-derive a
//! smaller `n_q_heads`, hence a smaller `group`, and silently re-map the
//! remaining query heads onto the wrong kv heads.
//!
//! `out`, by contrast, is the range's **own** buffer — exactly
//! `kv_heads.len() * group * head_dim` f32, written from index zero. `q` can be
//! shared across shards because it is `&[f32]`; `out` cannot, so the range form
//! takes the sub-slice rather than the whole vector and indexes it from zero.
//! A caller fanning this out hands each shard a genuinely disjoint `&mut [f32]`
//! (`split_at_mut`, or one `chunks_mut(group * head_dim)` chunk per kv head)
//! instead of `group`-many overlapping views of one allocation, which is
//! undefined behaviour under Stacked and Tree Borrows whether or not the writes
//! land in different elements. Calls over disjoint ranges with disjoint scratch
//! are therefore independent by construction, and their union is the whole call
//! bit for bit (`kv_range_union_is_bit_identical_to_the_whole_call`).
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
use core::ops::Range;
use thiserror::Error;

#[cfg(target_arch = "x86_64")]
mod x86;

/// The largest `head_dim` the AVX2 path covers.
///
/// `x86::qk_scores` carves its transposed query block and its K conversion
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

    /// The output buffer is not the length this call writes.
    ///
    /// That is `q.len()` for the whole-vector entry points and
    /// `kv_heads.len() * group * head_dim` for the kv-range ones — the
    /// **range's own** window, indexed from zero, not a window inside the full
    /// vector. Reported rather than absorbed: writing a prefix of a longer
    /// buffer would leave a shard's heads at whatever was there before, which
    /// is a wrong answer with a right answer's shape.
    #[error("attention: out length {out_len}, expected {expected} for this call")]
    OutLenMismatch {
        /// Offending output length.
        out_len: usize,
        /// Length this call writes: `kv_heads.len() * group * head_dim`,
        /// which is `q.len()` when the range is the whole sweep.
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

    /// A kv-head range ([`decode_attention_kv_range_in`] /
    /// [`attention_at_kv_range_in`]) is reversed (`start > end`) or reaches
    /// past the cache's kv heads. Reported rather than clamped: a shard
    /// silently computing fewer heads than it was assigned leaves the rest of
    /// `out` holding whatever was there before, which is a wrong answer with a
    /// right answer's shape. An *empty* in-range request (`start == end <=
    /// n_kv_heads`) is not this error — it is the legal no-op a surplus shard
    /// gets.
    #[error(
        "attention: kv head range {start}..{end} is not a sub-range of \
         0..{n_kv_heads}"
    )]
    KvHeadRangeOutOfRange {
        /// First kv head requested.
        start: usize,
        /// One past the last kv head requested.
        end: usize,
        /// The cache's kv head count.
        n_kv_heads: usize,
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
///
/// **Precedence.** Checks run in a fixed order and the *first* failure is the
/// one reported: `q`'s length, then the GQA grouping, then `layer`, then the
/// position count, then the kv-head range (which this entry point always
/// passes), then `out`'s length. `out` is validated **last** — see the note on
/// the private `plan` — so a call that gets both `layer` and `out` wrong is
/// told about `layer`, not about `out`. Pinned by
/// `error_precedence_is_pinned_from_q_to_out`.
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
///
/// **Precedence.** [`decode_attention`]'s order — `q`, GQA, `layer`,
/// positions, kv range, `out` — with the scratch length checked **after** all
/// of it, because the length it is checked against is this call's whole
/// validated geometry. A call that is short on scratch *and* wrong on `out` is
/// therefore told about `out`. Pinned by
/// `error_precedence_is_pinned_from_q_to_out`.
pub fn decode_attention_in(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    scale: f32,
    scratch: &mut [f32],
    out: &mut [f32],
) -> Result<(), AttentionError> {
    attention_borrowed(
        q,
        cache,
        layer,
        None,
        scale,
        0..cache.n_kv_heads(),
        scratch,
        out,
        false,
    )
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
    attention_borrowed(
        q,
        cache,
        layer,
        None,
        scale,
        0..cache.n_kv_heads(),
        scratch,
        out,
        force_scalar,
    )
}

/// As [`decode_attention_in`], but computing only the query heads belonging to
/// the kv heads in `kv_heads`.
///
/// This is the shard entry point: the compute pool fans decode attention
/// across threads by **kv head**, one contiguous range per shard. See the
/// module docs for why that axis and not the query head; the short version is
/// that it is the only cut that duplicates no f16-to-f32 conversion, needs no
/// gather or scatter, and leaves the AVX2 dot's eight lanes full.
///
/// The two vector arguments are **asymmetric**, and deliberately so:
///
/// - `q` is the **full** query vector, all `n_q_heads` heads, exactly as
///   [`decode_attention_in`] takes it. `group` is derived from it and is
///   therefore the same on every shard; handing this function a narrowed `q`
///   instead would re-derive a smaller `group` and re-map the heads onto the
///   wrong kv heads. It is `&[f32]`, so every shard sharing it is fine.
/// - `out` is the range's **own** buffer: exactly `kv_heads.len() * group *
///   head_dim` f32, written from index **zero**, and fully overwritten. It is
///   not a window inside the full output vector — the caller carves that window
///   and passes it. Anything else is [`AttentionError::OutLenMismatch`].
///
/// The asymmetry is what lets a fan-out be sound. `out` is `&mut [f32]`, and
/// two live `&mut` over one allocation are undefined behaviour under Stacked
/// and Tree Borrows even when the writes never collide, so a shard cannot be
/// handed the whole vector and trusted to stay in its lane. Taking the
/// sub-slice moves that from an argument to a type: the caller splits `out`
/// once (`split_at_mut`, or `chunks_mut(group * head_dim)`) and the disjointness
/// is structural.
///
/// `scratch` is sized exactly as [`decode_attention_in`]'s, by
/// [`scratch_len`] at the **full** geometry — the score block and the two
/// conversion rows are reused from one kv head to the next, so the
/// requirement does not shrink with the range and one uniform carve serves
/// every shard.
///
/// An empty `kv_heads` (`start == end`, in range) is a legal no-op for surplus
/// shards: it writes nothing and computes nothing, and its `out` is
/// zero-length, which is what `kv_heads.len() * group * head_dim` evaluates to.
/// It is still validated like any other call, so a shard handed a broken
/// geometry reports it rather than returning `Ok` by accident.
///
/// Concatenating the outputs of any partition of `0..n_kv_heads`, in ascending
/// range order, reproduces [`decode_attention_in`]'s output **bit for bit**
/// (`kv_range_union_is_bit_identical_to_the_whole_call`): each kv head's
/// arithmetic depends on nothing but its own slice of `q` and the cache, so
/// dropping the other kv heads' iterations changes no operand and no order.
///
/// # Errors
///
/// Everything [`decode_attention_in`] returns, plus
/// [`AttentionError::KvHeadRangeOutOfRange`] when `kv_heads` is reversed or
/// reaches past the cache's kv heads, and [`AttentionError::OutLenMismatch`]
/// when `out` is not the range's own length. `out` is untouched on error.
pub fn decode_attention_kv_range_in(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    scale: f32,
    kv_heads: Range<usize>,
    scratch: &mut [f32],
    out: &mut [f32],
) -> Result<(), AttentionError> {
    attention_borrowed(q, cache, layer, None, scale, kv_heads, scratch, out, false)
}

/// [`decode_attention_kv_range_in`] with the kernel path pinned instead of
/// probed; see [`decode_attention_dispatch`].
///
/// # Errors
///
/// Exactly [`decode_attention_kv_range_in`]'s.
#[allow(clippy::too_many_arguments)]
pub fn decode_attention_kv_range_in_dispatch(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    scale: f32,
    kv_heads: Range<usize>,
    scratch: &mut [f32],
    out: &mut [f32],
    force_scalar: bool,
) -> Result<(), AttentionError> {
    attention_borrowed(
        q,
        cache,
        layer,
        None,
        scale,
        kv_heads,
        scratch,
        out,
        force_scalar,
    )
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
///
/// **Precedence.** [`decode_attention`]'s order — `q`, GQA, `layer`,
/// `positions`, kv range (always passed here), `out` — with the caller's
/// explicit `positions` taking the slot the layer's own length takes there. So
/// `positions = 0` against a mis-sized `out` reports
/// [`AttentionError::EmptyLayer`], and `positions` past the end reports
/// [`KvError::PositionOutOfRange`]; neither reports `out`. Pinned by
/// `error_precedence_is_pinned_from_q_to_out`.
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
///
/// **Precedence.** [`attention_at`]'s order — `q`, GQA, `layer`, `positions`,
/// kv range, `out` — with the scratch length checked **after** all of it, as in
/// [`decode_attention_in`]. Pinned by
/// `error_precedence_is_pinned_from_q_to_out`.
pub fn attention_at_in(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    positions: usize,
    scale: f32,
    scratch: &mut [f32],
    out: &mut [f32],
) -> Result<(), AttentionError> {
    attention_borrowed(
        q,
        cache,
        layer,
        Some(positions),
        scale,
        0..cache.n_kv_heads(),
        scratch,
        out,
        false,
    )
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
        0..cache.n_kv_heads(),
        scratch,
        out,
        force_scalar,
    )
}

/// [`decode_attention_kv_range_in`] with an explicit causal `positions` limit
/// — the prefill shape of the shard entry point.
///
/// This is [`attention_at_in`] restricted to the query heads of `kv_heads`,
/// exactly as [`decode_attention_kv_range_in`] restricts
/// [`decode_attention_in`]. Every clause of that function's contract applies
/// unchanged: `q` is the full query vector and `group` comes from it, `out` is
/// the range's own `kv_heads.len() * group * head_dim` buffer written from
/// index zero, `scratch` is sized by [`scratch_len`] at the full geometry, and
/// an empty in-range `kv_heads` is a validated no-op with a zero-length `out`.
///
/// The two limits compose without interacting: `positions` bounds the `t`
/// axis and `kv_heads` bounds the kv-head axis, so this call is bit-identical
/// to the matching window of [`attention_at_in`]'s output at the same
/// `positions`.
///
/// # Errors
///
/// Everything [`attention_at_in`] returns, plus
/// [`AttentionError::KvHeadRangeOutOfRange`] when `kv_heads` is reversed or
/// reaches past the cache's kv heads, and [`AttentionError::OutLenMismatch`]
/// when `out` is not the range's own length. `out` is untouched on error.
#[allow(clippy::too_many_arguments)]
pub fn attention_at_kv_range_in(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    positions: usize,
    scale: f32,
    kv_heads: Range<usize>,
    scratch: &mut [f32],
    out: &mut [f32],
) -> Result<(), AttentionError> {
    attention_borrowed(
        q,
        cache,
        layer,
        Some(positions),
        scale,
        kv_heads,
        scratch,
        out,
        false,
    )
}

/// [`attention_at_kv_range_in`] with the kernel path pinned instead of probed;
/// see [`decode_attention_dispatch`].
///
/// # Errors
///
/// Exactly [`attention_at_kv_range_in`]'s.
#[allow(clippy::too_many_arguments)]
pub fn attention_at_kv_range_in_dispatch(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    positions: usize,
    scale: f32,
    kv_heads: Range<usize>,
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
        kv_heads,
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
    /// Query heads per kv head, `n_q_heads / n_kv_heads` (>= 1). Always
    /// derived from the **full** `q`, never from `kv_heads` — see the module
    /// docs on sharding.
    group: usize,
    /// Positions actually attended over — the causal limit.
    positions: usize,
    /// The kv heads this call computes, a validated sub-range of
    /// `0..n_kv_heads`. Empty means "compute nothing".
    kv_heads: Range<usize>,
}

impl Plan {
    /// The scratch this exact call needs.
    ///
    /// Deliberately independent of [`Plan::kv_heads`]: the score block and the
    /// two conversion rows are reused from one kv head to the next, so a shard
    /// computing a single kv head needs exactly what a shard computing all of
    /// them needs. That is what lets one carve sized by [`scratch_len`] at the
    /// full geometry serve every shard, whatever range it is given.
    fn scratch_len(&self) -> usize {
        scratch_len_of(self.group, self.head_dim, self.positions)
    }
}

/// Every shape and range check, in the order the entry points promise: `q`,
/// then the GQA grouping, then the layer, then `positions`, then `kv_heads`,
/// then `out`. Nothing is written before this returns `Ok`.
///
/// `kv_heads` is checked next to last on purpose: everything before it is
/// pinned by the typed-error tests, and the whole-vector entry points pass
/// `0..n_kv_heads`, which can never fail it (`KvCache::new` rejects a zero
/// `n_kv_heads`), so a shard with a broken geometry is still told what is
/// actually broken rather than being told about its range.
///
/// `out` is checked **last** because its required length is
/// `kv_heads.len() * group * head_dim` — a function of the *validated* range.
/// Until the range is known to be a sub-range of `0..n_kv_heads` there is no
/// expected length to compare against, and reporting a mismatch against a
/// meaningless expectation would be worse than reporting the range. For the
/// whole-vector entry points that expected length is exactly `q.len()`, so
/// they see the same [`AttentionError::OutLenMismatch`] payload they always
/// did.
///
/// The *payload* is unchanged; the **precedence is not**, and that is
/// semver-visible on the four whole-vector `pub fn`s. `out` used to be
/// validated before the layer and before `positions`, when its expected length
/// was just `q.len()` and nothing had to be known first. So for the same bad
/// input the reported variant moved:
///
/// | call | before | now |
/// |---|---|---|
/// | `decode_attention` with a bad layer and a short `out` | `OutLenMismatch` | `Kv(LayerOutOfRange)` |
/// | `attention_at` with `positions == 0` and a short `out` | `OutLenMismatch` | `EmptyLayer` |
/// | `attention_at` with `positions > len` and a short `out` | `OutLenMismatch` | `Kv(PositionOutOfRange)` |
///
/// Every ordering this function establishes — `q`, GQA, layer, positions, kv
/// range, `out` — is pinned end to end by
/// `error_precedence_is_pinned_from_q_to_out`, so a future reorder is a test
/// failure rather than a silent behaviour change for a downstream caller
/// matching on the variant.
fn plan(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    limit: Option<usize>,
    kv_heads: Range<usize>,
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
    if kv_heads.start > kv_heads.end || kv_heads.end > n_kv_heads {
        return Err(AttentionError::KvHeadRangeOutOfRange {
            start: kv_heads.start,
            end: kv_heads.end,
            n_kv_heads,
        });
    }
    let group = n_q_heads / n_kv_heads;
    // Cannot overflow: `kv_heads.end <= n_kv_heads` was just established, so
    // this is at most `n_kv_heads * group * head_dim`, which is `q.len()` —
    // a length that already exists.
    let expected = kv_heads.len() * group * head_dim;
    if out.len() != expected {
        return Err(AttentionError::OutLenMismatch {
            out_len: out.len(),
            expected,
        });
    }
    Ok(Plan {
        head_dim,
        kv_dim: cache.kv_dim(),
        group,
        positions,
        kv_heads,
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
    // The owning form is whole-vector only: there is no sharded caller that
    // wants a `Vec` per shard, so the range is always the full sweep.
    let plan = plan(q, cache, layer, limit, 0..cache.n_kv_heads(), out)?;
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
    kv_heads: Range<usize>,
    scratch: &mut [f32],
    out: &mut [f32],
    force_scalar: bool,
) -> Result<(), AttentionError> {
    // Validation runs in full even for an empty range, so a surplus shard is
    // told about a bad geometry exactly as its busy siblings are, and the
    // scratch requirement it reports is the range-independent one. Only the
    // *work* is skipped, in `attention_body`'s loop.
    let plan = plan(q, cache, layer, limit, kv_heads, out)?;
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
///
/// The shard limit lives entirely in `plan.kv_heads`, which is the trip count
/// of the outer loop and nothing else. One iteration reads only `q`'s
/// `group_dim`-long window for its own kv head and writes only its own chunk of
/// `out`, and it overwrites the whole score carve before reading any of it, so
/// iterations neither observe nor disturb one another. Dropping iterations
/// therefore leaves the surviving ones bit-identical to the whole sweep, and an
/// empty range is a loop that runs zero times: nothing written, no error.
///
/// The two vectors are indexed differently, which is the whole point of the
/// range form. `q` is the **full** vector, so its window is at the *absolute*
/// offset `kv_head * group_dim`. `out` is the **range's own** buffer, so its
/// chunks run from zero — the `n`th kv head of the range writes the `n`th
/// chunk, whatever absolute head that is. That is what lets a caller hand each
/// shard a disjoint `&mut [f32]` instead of `n` aliasing views of one
/// allocation.
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
        ..
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

    // `group >= 1` and `KvCache::new` rejects a zero `head_dim`, so `group_dim`
    // is nonzero and `chunks_exact_mut` cannot panic. `plan` proved
    // `out.len() == kv_heads.len() * group_dim`, so the zip walks exactly one
    // whole chunk per kv head and leaves no remainder.
    let group_dim = group * head_dim;
    for (kv_head, out_group) in plan.kv_heads.clone().zip(out.chunks_exact_mut(group_dim)) {
        // `q` is the full vector, so its window is at the absolute offset. In
        // range without a bounds check to spare: `plan` proved
        // `kv_heads.end <= n_kv_heads` and `q.len() == n_kv_heads * group_dim`,
        // so `lo + group_dim <= q.len()` for every `kv_head` this loop yields,
        // and the product cannot overflow.
        let lo = kv_head * group_dim;
        let q_group = &q[lo..lo + group_dim];
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

    /// The **order** the checks run in, pinned end to end through every entry
    /// point that validates an `out`: `q`, then the GQA grouping, then `layer`,
    /// then `positions`, then the kv-head range, then `out`, then the scratch
    /// carve.
    ///
    /// `typed_errors_on_bad_shapes` and `attention_at_typed_errors_on_positions`
    /// pin the *variants*: each of their cases breaks one thing and passes a
    /// correctly-sized `out`, so none of them can tell which check ran first.
    /// This test breaks several at once and asserts the earliest one wins.
    ///
    /// It exists because the order is semver-visible and it **changed**. `out`
    /// used to be validated before the layer and before `positions`, when its
    /// expected length was just `q.len()`; it is now
    /// `kv_heads.len() * group * head_dim`, a function of the validated range,
    /// so it moved to the end (see `plan`'s docs). The three calls that moved —
    /// `decode_attention` with a bad layer, `attention_at` with `positions == 0`
    /// and with `positions > len`, each against a mis-sized `out` — are all
    /// below, and a future reorder fails here instead of silently changing the
    /// variant a downstream caller matches on.
    #[test]
    fn error_precedence_is_pinned_from_q_to_out() {
        let (n_layers, n_kv, n_q, head_dim, n_pos) = (2usize, 4usize, 8usize, 8usize, 3usize);
        let cap = 4usize;
        let mut cache = KvCache::new(n_layers, n_kv, head_dim, cap).unwrap();
        let row = vec![0.5f32; n_kv * head_dim];
        for _ in 0..n_pos {
            cache.append(0, &row, &row).unwrap();
        }
        // Layer 1 stays empty on purpose: it is how the unlimited (decode)
        // forms reach the position check, which they cannot be handed directly.
        let q = vec![0.25f32; n_q * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group_dim = (n_q / n_kv) * head_dim;
        // Long enough for every call below whose plan succeeds, so the scratch
        // check never fires ahead of the one under test. The scratch level gets
        // its own block at the end.
        let need = scratch_len(n_q, n_kv, head_dim, cap);

        // A length no call here ever wants: not `q.len()`, and not
        // `k * group_dim` for any `k`.
        const WRONG_OUT: usize = 7;
        // Past the cache's kv heads, so the range forms trip level 5 unless
        // something earlier fires first.
        let bad_range = n_kv + 1..n_kv + 1;
        let range_forms = ["decode_attention_kv_range_in", "attention_at_kv_range_in"];

        // One broken call through all six entry points. `positions` is used by
        // the limited forms only and `kv_heads` by the range forms only, so a
        // probe can carry a fault at every level at once and each entry point
        // reports the first level it can actually see. `out` is NaN-poisoned:
        // a rejected call that wrote anything is caught right here.
        let probe = |q_arg: &[f32],
                     layer: usize,
                     positions: usize,
                     out_len: usize,
                     kv_heads: Range<usize>|
         -> Vec<(&'static str, AttentionError)> {
            let mut owned = AttentionScratch::new();
            let mut carve = vec![0.0f32; need];
            let mut out = vec![f32::NAN; out_len];
            let got: Vec<(&'static str, AttentionError)> = vec![
                (
                    "decode_attention",
                    decode_attention(q_arg, &cache, layer, scale, &mut owned, &mut out)
                        .unwrap_err(),
                ),
                (
                    "decode_attention_in",
                    decode_attention_in(q_arg, &cache, layer, scale, &mut carve, &mut out)
                        .unwrap_err(),
                ),
                (
                    "attention_at",
                    attention_at(q_arg, &cache, layer, positions, scale, &mut owned, &mut out)
                        .unwrap_err(),
                ),
                (
                    "attention_at_in",
                    attention_at_in(q_arg, &cache, layer, positions, scale, &mut carve, &mut out)
                        .unwrap_err(),
                ),
                (
                    "decode_attention_kv_range_in",
                    decode_attention_kv_range_in(
                        q_arg,
                        &cache,
                        layer,
                        scale,
                        kv_heads.clone(),
                        &mut carve,
                        &mut out,
                    )
                    .unwrap_err(),
                ),
                (
                    "attention_at_kv_range_in",
                    attention_at_kv_range_in(
                        q_arg, &cache, layer, positions, scale, kv_heads, &mut carve, &mut out,
                    )
                    .unwrap_err(),
                ),
            ];
            assert!(
                out.iter().all(|x| x.is_nan()),
                "a rejected call wrote to `out`"
            );
            got
        };

        // Levels 1-3 — `q`, the GQA grouping, `layer`. Every entry point checks
        // these identically, so one expectation covers all six. Each probe also
        // carries a wrong `out`, a `positions` past the end and a broken range,
        // and none of those is what comes back.
        let bad_layer = n_layers + 3;
        for (label, q_arg, want) in [
            (
                "q length",
                &q[..13],
                AttentionError::QLenIndivisible {
                    q_len: 13,
                    head_dim,
                },
            ),
            (
                "GQA grouping",
                &q[..5 * head_dim],
                AttentionError::GqaGroupMismatch {
                    n_q_heads: 5,
                    n_kv_heads: n_kv,
                },
            ),
            (
                "layer",
                &q[..],
                AttentionError::Kv(KvError::LayerOutOfRange {
                    layer: bad_layer,
                    n_layers,
                }),
            ),
        ] {
            for (name, err) in probe(q_arg, bad_layer, n_pos + 9, WRONG_OUT, bad_range.clone()) {
                assert_eq!(err, want, "{label} beats everything after it: {name}");
            }
        }

        // Level 4, the empty-history shape. All six agree: the unlimited forms
        // take layer 1's own length (0) and the limited ones are handed 0.
        for (name, err) in probe(&q, 1, 0, WRONG_OUT, bad_range.clone()) {
            assert_eq!(
                err,
                AttentionError::EmptyLayer { layer: 1 },
                "an empty layer beats the range and `out`: {name}"
            );
        }

        // Level 4 again, `positions` past the end — which only the limited
        // forms can be handed. The unlimited forms take layer 0's own length,
        // which is valid, so they fall through to the *next* level they can
        // see: the range for the range form, `out` for the whole-vector ones.
        // That split is the precedence claim, not an accident of this input.
        for (name, err) in probe(&q, 0, n_pos + 9, WRONG_OUT, bad_range.clone()) {
            let want = match name {
                "attention_at" | "attention_at_in" | "attention_at_kv_range_in" => {
                    AttentionError::Kv(KvError::PositionOutOfRange {
                        layer: 0,
                        pos: n_pos + 8,
                        len: n_pos,
                    })
                }
                "decode_attention_kv_range_in" => AttentionError::KvHeadRangeOutOfRange {
                    start: bad_range.start,
                    end: bad_range.end,
                    n_kv_heads: n_kv,
                },
                _ => AttentionError::OutLenMismatch {
                    out_len: WRONG_OUT,
                    expected: q.len(),
                },
            };
            assert_eq!(err, want, "`positions` past the end: {name}");
        }

        // Level 5 — the kv range, with everything before it valid. The
        // whole-vector entry points pass `0..n_kv_heads` and therefore cannot
        // reach this error at all; they report `out` instead.
        for (name, err) in probe(&q, 0, n_pos, WRONG_OUT, bad_range.clone()) {
            let want = if range_forms.contains(&name) {
                AttentionError::KvHeadRangeOutOfRange {
                    start: bad_range.start,
                    end: bad_range.end,
                    n_kv_heads: n_kv,
                }
            } else {
                AttentionError::OutLenMismatch {
                    out_len: WRONG_OUT,
                    expected: q.len(),
                }
            };
            assert_eq!(err, want, "the range beats `out`: {name}");
        }

        // Level 6 — `out` alone, against a whole range and against a partial
        // one, so the expected length really is the *range's* and not `q`'s.
        for (kv_heads, heads) in [(0..n_kv, n_kv), (1..3, 2)] {
            for (name, err) in probe(&q, 0, n_pos, WRONG_OUT, kv_heads) {
                let expected = if range_forms.contains(&name) {
                    heads * group_dim
                } else {
                    q.len()
                };
                assert_eq!(
                    err,
                    AttentionError::OutLenMismatch {
                        out_len: WRONG_OUT,
                        expected,
                    },
                    "`out` is last, over {heads} kv heads: {name}"
                );
            }
        }

        // Level 7 — the scratch carve, checked after the whole plan. Sized from
        // this call's own `positions` (3), not from the cache's capacity, so
        // one f32 short really is short.
        let need_at = scratch_len(n_q, n_kv, head_dim, n_pos);
        let mut short = vec![0.0f32; need_at - 1];
        let mut wrong = vec![f32::NAN; WRONG_OUT];
        for (name, err) in [
            (
                "decode_attention_in",
                decode_attention_in(&q, &cache, 0, scale, &mut short, &mut wrong).unwrap_err(),
            ),
            (
                "attention_at_in",
                attention_at_in(&q, &cache, 0, n_pos, scale, &mut short, &mut wrong).unwrap_err(),
            ),
        ] {
            assert_eq!(
                err,
                AttentionError::OutLenMismatch {
                    out_len: WRONG_OUT,
                    expected: q.len(),
                },
                "`out` beats the scratch carve: {name}"
            );
        }
        assert!(wrong.iter().all(|x| x.is_nan()));
        let mut right = vec![f32::NAN; q.len()];
        for (name, err) in [
            (
                "decode_attention_in",
                decode_attention_in(&q, &cache, 0, scale, &mut short, &mut right).unwrap_err(),
            ),
            (
                "attention_at_in",
                attention_at_in(&q, &cache, 0, n_pos, scale, &mut short, &mut right).unwrap_err(),
            ),
        ] {
            assert_eq!(
                err,
                AttentionError::ScratchTooShort {
                    len: need_at - 1,
                    need: need_at,
                },
                "with `out` right, the scratch is what is reported: {name}"
            );
        }
        assert!(right.iter().all(|x| x.is_nan()));
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
    /// blocking *and* its tail (67 = 16 blocks of 4, then 3), two odd
    /// `head_dim` geometries whose head slices cannot be better than 2-byte
    /// aligned, and the group / `head_dim` rows below.
    ///
    /// # The `group > LANES` rows
    ///
    /// `x86::qk_scores` walks the GQA group in chunks of
    /// `x86::LANES` (`while base < group { .. base += LANES }`) and
    /// `x86::store_block` offsets its stores by `base`. Every group in the
    /// original sweep was 8 or less, so that loop ran **exactly once** on
    /// every tested geometry and the `base != 0` arm of the store never
    /// executed — while an MQA-shaped model reaches it straight through the
    /// public API. These rows execute it:
    ///
    /// - `(1, 1, 32, 128, 9)` — MQA at the v0 `head_dim`: group 32, so four
    ///   chunks at `base` 0 / 8 / 16 / 24, three of them storing at a nonzero
    ///   base, all with a full eight lanes.
    /// - `(1, 2, 34, 16, 13)` — group 17: three chunks, the last one a
    ///   **single** lane at `base = 16`, which is the only configuration that
    ///   combines a nonzero base with the zero-padded lanes whose results must
    ///   be discarded.
    /// - `(1, 1, 12, 9, 11)` — group 12 at an odd `head_dim`: two chunks, a
    ///   short four-lane chunk at `base = 8`, on head slices that are only
    ///   2-byte aligned and with a scalar tail inside `widen`.
    ///
    /// # The `head_dim` boundary rows
    ///
    /// `x86::qk_scores` carves both of its stack buffers from
    /// [`MAX_SIMD_HEAD_DIM`], so the interesting `head_dim` is the bound
    /// itself, not the v0 pin's 128:
    ///
    /// - `(1, 1, 8, 256, 5)` — `head_dim == MAX_SIMD_HEAD_DIM`: `qt` is
    ///   indexed to its last element (`255 * LANES + 7`) and `widen_rows`
    ///   fills `kb` exactly. One off-by-one in either carve shows here and
    ///   nowhere else.
    /// - `(1, 2, 4, 255, 3)` — one below the bound and odd, so the same
    ///   near-full carve lands on 2-byte-aligned head slices with a seven-
    ///   element scalar tail after 31 wide loads.
    ///
    /// The other side of that bound (`head_dim = 257`, outside the envelope
    /// and falling back to the scalar kernel) cannot live here — the sweep
    /// asserts every row is inside [`avx2_f16c_path_covers`] — so it has its
    /// own gate, `simd_envelope_boundary_at_max_simd_head_dim`.
    fn dispatch_geometries() -> Vec<(usize, usize, usize, usize, usize)> {
        let mut g = LIMITED_GEOMETRIES.to_vec();
        g.push((1, 4, 32, 128, 67));
        g.push((1, 1, 8, 9, 11));
        g.push((2, 3, 6, 9, 13));
        // Groups past LANES: the chunk loop and `store_block`'s `base != 0`.
        g.push((1, 1, 32, 128, 9));
        g.push((1, 2, 34, 16, 13));
        g.push((1, 1, 12, 9, 11));
        // The `MAX_SIMD_HEAD_DIM` boundary, on it and just under it.
        g.push((1, 1, 8, 256, 5));
        g.push((1, 2, 4, 255, 3));
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
    /// tolerance, through all four entry points, on every geometry in
    /// [`dispatch_geometries`] — including the GQA groups past
    /// `x86::LANES` that drive the chunk loop and the `head_dim` sitting
    /// exactly on [`MAX_SIMD_HEAD_DIM`] — at
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

    /// Both sides of the AVX2 envelope's upper bound.
    ///
    /// `head_dim = MAX_SIMD_HEAD_DIM` is *inside* it — `dispatch_geometries`
    /// sweeps 256 end to end, which is where the stack-buffer carves in
    /// `x86::qk_scores` are indexed to their last element. `head_dim` one past
    /// the bound is *outside* it, and the contract there is a silent, correct
    /// fallback to the scalar kernel, which nothing tested: a geometry that
    /// large cannot live in the dispatch sweep, because that sweep asserts
    /// every row is inside the envelope precisely so `force_scalar` never
    /// compares the scalar kernel with itself.
    ///
    /// So the fallback is pinned here instead, against the pre-vectorization
    /// head-major nest, so that raising or lowering `MAX_SIMD_HEAD_DIM` cannot
    /// quietly change what a large-`head_dim` model computes.
    #[test]
    fn simd_envelope_boundary_at_max_simd_head_dim() {
        // On a host with the instructions, 256 dispatches to AVX2 and 257 does
        // not. On a host without them, neither does, and the first assertion
        // still holds because both sides are false.
        assert_eq!(avx2_f16c_path_covers(256), avx2_f16c_available());
        assert!(!avx2_f16c_path_covers(257));
        assert!(!avx2_f16c_path_covers(usize::MAX));

        let (n_layers, n_kv, n_q, head_dim, n_pos) = (2usize, 2usize, 4usize, 257usize, 3usize);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut rng = Rng::new(0x2011_B0DD);
        let (cache, _rows) = random_cache(&mut rng, n_layers, n_kv, head_dim, n_pos);
        let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
        let mut owned = AttentionScratch::new();

        for layer in 0..n_layers {
            for positions in 1..=n_pos {
                let mut want = vec![0.0f32; q.len()];
                head_major_reference(&q, &cache, layer, positions, scale, &mut want);
                let mut got = vec![0.0f32; q.len()];
                attention_at(&q, &cache, layer, positions, scale, &mut owned, &mut got).unwrap();
                for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "head_dim {head_dim} layer {layer} positions {positions} elem {i}: \
                         fallback {g:e} vs head-major reference {w:e}"
                    );
                }
            }
        }
    }

    // ------------------------------------------- kv-head range form (wave 3)

    /// Gate 12 (phase 7 wave 3, the claim the shard fan-out rests on): the
    /// concatenation of per-kv-head range calls is **bit-identical** to one
    /// whole-vector call, and a range call fills its own buffer and nothing
    /// else.
    ///
    /// This is what makes fanning attention across the compute pool by kv head
    /// a scheduling change and not a numerical one. It holds because `group`
    /// is derived from the full `q` on every shard — so shard `k` computes
    /// query heads `k * group .. (k + 1) * group` against kv head `k`, exactly
    /// the heads the whole sweep's `k`th iteration computes — and because that
    /// iteration reads only its own `group_dim` window of `q`, overwrites the
    /// whole score carve before reading any of it, and zeroes its own `out`
    /// chunk. Nothing is carried between kv heads, so dropping the other
    /// iterations changes no operand and no order.
    ///
    /// `out` is now the range's **own** buffer rather than the full vector, so
    /// every shard here is handed a genuinely disjoint `&mut [f32]` carved from
    /// the answer buffer with a safe primitive — an index window, `split_at_mut`
    /// or `chunks_exact_mut`. The buffer is still sentinel-filled and still
    /// asserted element by element outside the window, which is now a statement
    /// about the *caller's* offset arithmetic: the borrow checker stops the
    /// kernel from reaching outside its slice, but nothing stops a caller from
    /// carving the wrong slice, and a shard writing the right bits into the
    /// wrong place is exactly as wrong as before.
    ///
    /// Swept over every geometry in [`dispatch_geometries`] (GQA groups 1
    /// through 32, `head_dim` 4 through 256), every layer, every position
    /// count in [`dispatch_lengths`] including 1 and the full length, and both
    /// kernel paths — because a shard split that were bit-exact on the scalar
    /// path and not on the AVX2 one would be the same defect.
    ///
    /// Four statements per point: each single-kv-head shard in isolation, an
    /// uneven two-shard partition sharing one `out` through `split_at_mut`, an
    /// empty range with a zero-length `out` writing nothing, and the unlimited
    /// (decode) form at full length over `chunks_exact_mut`.
    #[test]
    fn kv_range_union_is_bit_identical_to_the_whole_call() {
        let paths = kernel_paths("the kv-head range entry points against the AVX2+F16C path");
        let mut rng = Rng::new(0x5EA5_0EDF);
        // Not a value attention can produce here: `out` is a convex
        // combination of f16-round-tripped V rows drawn from [-1, 1].
        const SENTINEL: f32 = -7.5;

        for (n_layers, n_kv, n_q, head_dim, n_pos) in dispatch_geometries() {
            let scale = 1.0 / (head_dim as f32).sqrt();
            let (cache, _rows) = random_cache(&mut rng, n_layers, n_kv, head_dim, n_pos);
            let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
            let group_dim = (n_q / n_kv) * head_dim;

            for layer in 0..n_layers {
                for &positions in &dispatch_lengths(n_pos) {
                    // The carve is the *full* geometry's, not the range's:
                    // that is the contract a sharded caller sizes its arena
                    // from, so every shard below gets exactly this much.
                    let need = scratch_len(n_q, n_kv, head_dim, positions);

                    for &force_scalar in paths.flags() {
                        let label = format!(
                            "geometry ({n_layers}, {n_kv}, {n_q}, {head_dim}) layer {layer} \
                             positions {positions} force_scalar {force_scalar}"
                        );

                        let mut carve = vec![0.0f32; need];
                        let mut want = vec![0.0f32; q.len()];
                        attention_at_in_dispatch(
                            &q,
                            &cache,
                            layer,
                            positions,
                            scale,
                            &mut carve,
                            &mut want,
                            force_scalar,
                        )
                        .unwrap();

                        // (1) One shard per kv head, each writing into its own
                        // window of a sentinel-filled buffer. The window must
                        // hold the whole call's bits; everything else must
                        // still be the sentinel, which pins the caller's offset
                        // arithmetic. The windows partition `out`, so this is
                        // both the concatenation claim and the untouched one.
                        for kv_head in 0..n_kv {
                            let mut shard = vec![0.0f32; need];
                            let mut got = vec![SENTINEL; q.len()];
                            let window = kv_head * group_dim..(kv_head + 1) * group_dim;
                            attention_at_kv_range_in_dispatch(
                                &q,
                                &cache,
                                layer,
                                positions,
                                scale,
                                kv_head..kv_head + 1,
                                &mut shard,
                                &mut got[window.clone()],
                                force_scalar,
                            )
                            .unwrap();
                            for (i, &g) in got.iter().enumerate() {
                                let w = if window.contains(&i) {
                                    want[i]
                                } else {
                                    SENTINEL
                                };
                                assert_eq!(
                                    g.to_bits(),
                                    w.to_bits(),
                                    "{label} kv head {kv_head} elem {i}: range {g:e} vs \
                                     {} {w:e}",
                                    if window.contains(&i) {
                                        "whole call"
                                    } else {
                                        "untouched sentinel"
                                    }
                                );
                            }
                        }

                        // (2) An uneven two-shard partition over one `out`,
                        // which is how the pool actually uses this: the split
                        // point is not a head boundary of any other axis, and
                        // the two calls must still reassemble the whole call.
                        // `split_at_mut` is the point — the two shards hold
                        // provably disjoint `&mut [f32]`, not two views of the
                        // same allocation. At `n_kv == 1` the low half is the
                        // empty range against a zero-length slice.
                        let split = n_kv / 2;
                        let mut got = vec![SENTINEL; q.len()];
                        {
                            let (lo, hi) = got.split_at_mut(split * group_dim);
                            for (range, dst) in [(0..split, lo), (split..n_kv, hi)] {
                                let mut shard = vec![0.0f32; need];
                                attention_at_kv_range_in_dispatch(
                                    &q,
                                    &cache,
                                    layer,
                                    positions,
                                    scale,
                                    range,
                                    &mut shard,
                                    dst,
                                    force_scalar,
                                )
                                .unwrap();
                            }
                        }
                        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                            assert_eq!(
                                g.to_bits(),
                                w.to_bits(),
                                "{label} split at {split} elem {i}: {g:e} vs whole call {w:e}"
                            );
                        }

                        // (3) The surplus-shard case: an empty range with a
                        // zero-length `out` is a no-op, not an error and not a
                        // write. Once as a zero-length carve out of a live
                        // buffer (which must stay untouched) and once as a bare
                        // empty slice, because a surplus shard may have no
                        // buffer to carve from at all.
                        let mut idle = vec![SENTINEL; q.len()];
                        let mut shard = vec![0.0f32; need];
                        for range in [0..0, n_kv..n_kv] {
                            attention_at_kv_range_in_dispatch(
                                &q,
                                &cache,
                                layer,
                                positions,
                                scale,
                                range.clone(),
                                &mut shard,
                                &mut idle[..0],
                                force_scalar,
                            )
                            .unwrap();
                            attention_at_kv_range_in_dispatch(
                                &q,
                                &cache,
                                layer,
                                positions,
                                scale,
                                range,
                                &mut shard,
                                &mut [],
                                force_scalar,
                            )
                            .unwrap();
                        }
                        assert!(
                            idle.iter().all(|&x| x.to_bits() == SENTINEL.to_bits()),
                            "{label}: an empty kv range wrote to `out`"
                        );

                        // (4) The unlimited (decode) form of the same claim,
                        // where the limit is the layer's whole length. The
                        // per-kv-head chunks come from `chunks_exact_mut`, the
                        // other safe way a caller can carve this.
                        if positions == n_pos {
                            let mut want_decode = vec![0.0f32; q.len()];
                            decode_attention_in_dispatch(
                                &q,
                                &cache,
                                layer,
                                scale,
                                &mut carve,
                                &mut want_decode,
                                force_scalar,
                            )
                            .unwrap();
                            let mut got_decode = vec![SENTINEL; q.len()];
                            for (kv_head, dst) in got_decode.chunks_exact_mut(group_dim).enumerate()
                            {
                                let mut shard = vec![0.0f32; need];
                                decode_attention_kv_range_in_dispatch(
                                    &q,
                                    &cache,
                                    layer,
                                    scale,
                                    kv_head..kv_head + 1,
                                    &mut shard,
                                    dst,
                                    force_scalar,
                                )
                                .unwrap();
                            }
                            for (i, (&g, &w)) in got_decode.iter().zip(&want_decode).enumerate() {
                                assert_eq!(
                                    g.to_bits(),
                                    w.to_bits(),
                                    "{label} decode elem {i}: concatenated ranges {g:e} vs \
                                     whole call {w:e}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The range form derives `group` from the **full** `q`, never from the
    /// range — the trap the query-head slab design fell into, and the one
    /// mistake here that would still produce plausible-looking numbers.
    ///
    /// A shard handed a *narrowed* `q` would re-derive `n_q_heads` from that
    /// slice and hence a smaller `group`, silently re-mapping the surviving
    /// query heads onto the wrong kv heads. This pins the correct mapping
    /// directly rather than through a reference implementation that could
    /// share the bug: kv head `j` wins at position `j` and its V rows carry
    /// the marker `10j + t`, so a call restricted to `j..j+1` must fill query
    /// heads `j * group .. (j + 1) * group` with `marker(j, j)` and nothing
    /// else. Any other derivation of `group` lands on a different marker,
    /// 0.1 apart against a 1e-3 gate.
    ///
    /// `out` is now the range's own chunk, so what the `f32::NAN` poison pins
    /// is the *pair* of derivations that have to agree: the caller carves the
    /// chunk at `j * group * head_dim` from its own `group`, and the kernel
    /// fills it from `group` re-derived from the full `q`. Narrowing `q` moves
    /// the second and not the first, which shows up here as a marker in the
    /// wrong chunk or a `NaN` left in the right one — exactly as it did when
    /// `out` was the whole vector.
    #[test]
    fn kv_range_derives_group_from_the_full_query() {
        let (n_kv, n_q, head_dim) = (4usize, 32usize, 16usize);
        let group = n_q / n_kv;
        let group_dim = group * head_dim;
        let kv_dim = n_kv * head_dim;
        let mut cache = KvCache::new(1, n_kv, head_dim, n_kv).unwrap();
        let marker = |j: usize, t: usize| (10 * j + t) as f32 * 0.01;
        for t in 0..n_kv {
            let mut k = vec![0.0f32; kv_dim];
            let mut v = vec![0.0f32; kv_dim];
            for j in 0..n_kv {
                let sign = if t == j { 1.0f32 } else { -1.0 };
                for i in 0..head_dim {
                    k[j * head_dim + i] = sign * 8.0 / head_dim as f32;
                    v[j * head_dim + i] = marker(j, t);
                }
            }
            cache.append(0, &k, &v).unwrap();
        }

        let q = vec![1.0f32; n_q * head_dim];
        let need = scratch_len(n_q, n_kv, head_dim, n_kv);
        for kv_head in 0..n_kv {
            let mut carve = vec![0.0f32; need];
            let mut out = vec![f32::NAN; q.len()];
            decode_attention_kv_range_in(
                &q,
                &cache,
                0,
                1.0,
                kv_head..kv_head + 1,
                &mut carve,
                &mut out[kv_head * group_dim..(kv_head + 1) * group_dim],
            )
            .unwrap();
            for h in 0..n_q {
                for i in 0..head_dim {
                    let got = out[h * head_dim + i];
                    if h / group == kv_head {
                        let want = marker(kv_head, kv_head);
                        assert!(
                            (got - want).abs() < 1e-3,
                            "kv head {kv_head} q head {h} elem {i}: got {got}, want \
                             marker {want} — `group` was not derived from the full q"
                        );
                    } else {
                        assert!(
                            got.is_nan(),
                            "kv head {kv_head} wrote q head {h} elem {i} ({got}), which \
                             belongs to kv head {}",
                            h / group
                        );
                    }
                }
            }
        }
    }

    /// Every bad kv range is a typed error, `out` is untouched, and the error
    /// precedence is the documented one: `q`, then the GQA grouping, then the
    /// layer, then `positions`, then the range, then `out`. A shard is never
    /// told about its range when something earlier is wrong, and — the point of
    /// the ordering — the whole-vector entry points cannot reach the range
    /// error at all.
    ///
    /// `out` moved to the end when it became the range's own buffer: its
    /// required length is `kv_heads.len() * group * head_dim`, so there is no
    /// expected length to report until the range is known to be in range. That
    /// is the one ordering this test pins differently from the wave-3 original,
    /// and it is asserted rather than assumed below.
    ///
    /// The scratch requirement is pinned here too, because a sharded caller
    /// sizes one uniform arena carve from it: a single-kv-head range needs
    /// exactly [`scratch_len`] at the **full** geometry, not a range-scaled
    /// fraction of it, and one f32 short is reported against that same number.
    /// The scratch check runs *after* the whole plan, so every call in that
    /// block hands `out` the length its range demands — a mis-sized `out` would
    /// mask the scratch error, which is itself worth pinning.
    // A reversed range is the input under test, not a mistake in writing one:
    // `clippy::reversed_empty_ranges` exists to catch `3..1` where `1..3` was
    // meant, and here `3..1` is exactly what a caller must be told about.
    #[allow(clippy::reversed_empty_ranges)]
    #[test]
    fn kv_range_typed_errors_and_scratch_contract() {
        let (n_layers, n_kv, n_q, head_dim, n_pos) = (2usize, 4usize, 8usize, 8usize, 5usize);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut rng = Rng::new(0x0BAD_2A46);
        let (cache, _rows) = random_cache(&mut rng, n_layers, n_kv, head_dim, n_pos);
        let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
        let need = scratch_len(n_q, n_kv, head_dim, n_pos);
        let mut carve = vec![0.0f32; need];
        let mut out = vec![7.0f32; q.len()];

        // Reversed, one past the end, empty but past the end, and the
        // saturating case. All four are the range error, none is a panic.
        for range in [3..1, 0..n_kv + 1, n_kv..n_kv + 1, n_kv + 1..n_kv + 1, 1..0] {
            let (start, end) = (range.start, range.end);
            assert_eq!(
                decode_attention_kv_range_in(
                    &q,
                    &cache,
                    0,
                    scale,
                    range.clone(),
                    &mut carve,
                    &mut out
                )
                .unwrap_err(),
                AttentionError::KvHeadRangeOutOfRange {
                    start,
                    end,
                    n_kv_heads: n_kv,
                }
            );
            assert_eq!(
                attention_at_kv_range_in(&q, &cache, 0, n_pos, scale, range, &mut carve, &mut out)
                    .unwrap_err(),
                AttentionError::KvHeadRangeOutOfRange {
                    start,
                    end,
                    n_kv_heads: n_kv,
                }
            );
        }
        assert_eq!(
            decode_attention_kv_range_in(&q, &cache, 0, scale, usize::MAX..0, &mut carve, &mut out)
                .unwrap_err(),
            AttentionError::KvHeadRangeOutOfRange {
                start: usize::MAX,
                end: 0,
                n_kv_heads: n_kv,
            }
        );

        // Precedence: everything the whole-vector form checks still comes
        // first, so a shard with a broken geometry is told what is actually
        // broken. The range is deliberately invalid in all four.
        let bad = 9..9;
        assert_eq!(
            decode_attention_kv_range_in(
                &q[..3],
                &cache,
                0,
                scale,
                bad.clone(),
                &mut carve,
                &mut out
            )
            .unwrap_err(),
            AttentionError::QLenIndivisible { q_len: 3, head_dim }
        );
        assert_eq!(
            decode_attention_kv_range_in(
                &q[..3 * head_dim],
                &cache,
                0,
                scale,
                bad.clone(),
                &mut carve,
                &mut out
            )
            .unwrap_err(),
            AttentionError::GqaGroupMismatch {
                n_q_heads: 3,
                n_kv_heads: n_kv,
            }
        );
        // `out` is the one check that now comes *after* the range, because the
        // length it is checked against is derived from the range. A call that
        // gets both wrong is told about the range.
        assert_eq!(
            decode_attention_kv_range_in(
                &q,
                &cache,
                0,
                scale,
                bad.clone(),
                &mut carve,
                &mut out[..7]
            )
            .unwrap_err(),
            AttentionError::KvHeadRangeOutOfRange {
                start: 9,
                end: 9,
                n_kv_heads: n_kv,
            }
        );
        assert_eq!(
            decode_attention_kv_range_in(
                &q,
                &cache,
                n_layers,
                scale,
                bad.clone(),
                &mut carve,
                &mut out
            )
            .unwrap_err(),
            AttentionError::Kv(KvError::LayerOutOfRange {
                layer: n_layers,
                n_layers,
            })
        );
        assert_eq!(
            attention_at_kv_range_in(
                &q,
                &cache,
                0,
                n_pos + 1,
                scale,
                bad.clone(),
                &mut carve,
                &mut out
            )
            .unwrap_err(),
            AttentionError::Kv(KvError::PositionOutOfRange {
                layer: 0,
                pos: n_pos,
                len: n_pos,
            })
        );
        assert_eq!(
            attention_at_kv_range_in(&q, &cache, 0, 0, scale, bad.clone(), &mut carve, &mut out)
                .unwrap_err(),
            AttentionError::EmptyLayer { layer: 0 }
        );

        // The scratch requirement does not shrink with the range: a shard
        // computing one kv head of four still needs the whole carve, and the
        // shortfall is reported against it — including for an empty range, so
        // a surplus shard is held to the same contract as a busy one.
        let group_dim = (n_q / n_kv) * head_dim;
        for range in [0..1, 0..n_kv, 2..2] {
            // The range's own `out` length, so the scratch error is what
            // surfaces rather than an `OutLenMismatch` masking it.
            let win = range.len() * group_dim;
            for len in [0usize, need - 1] {
                let mut short = vec![0.0f32; len];
                assert_eq!(
                    attention_at_kv_range_in(
                        &q,
                        &cache,
                        0,
                        n_pos,
                        scale,
                        range.clone(),
                        &mut short,
                        &mut out[..win]
                    )
                    .unwrap_err(),
                    AttentionError::ScratchTooShort { len, need },
                    "range {range:?} carve {len}"
                );
            }
            // Exactly `scratch_len` is enough for any range.
            let mut exact = vec![0.0f32; need];
            let mut sink = vec![0.0f32; win];
            attention_at_kv_range_in(&q, &cache, 0, n_pos, scale, range, &mut exact, &mut sink)
                .unwrap();
        }

        // No error path above wrote anything.
        assert!(out.iter().all(|&x| x == 7.0));
    }

    /// An `out` that is not the range's own length is a typed error — not a
    /// panic, and above all not a silently written prefix.
    ///
    /// This is the guard on the contract change that removed the decode
    /// fan-out's aliasing. `out` used to be the **full** output vector for
    /// every range, with a shard writing the window its kv heads owned; it is
    /// now the range's own buffer, written from index zero. The two shapes
    /// differ only in *length*, so the old call — full vector, partial range —
    /// is exactly what must not be quietly accepted: writing the leading
    /// `kv_heads.len() * group * head_dim` elements of a full vector would put
    /// every shard's answer at kv head 0's offset and leave every other head
    /// stale, a wrong answer with a right answer's shape. It gets its own case
    /// below, on every partial range.
    ///
    /// Swept over partial, whole and empty ranges, through both range entry
    /// points, at every neighbouring length: one short, one long, zero for a
    /// non-empty range, non-zero for an empty one, and the full vector. `out`
    /// is `f32::NAN`-poisoned both ways round — a rejected call must leave
    /// every element `NaN`, and an accepted one must leave none, so a partial
    /// write is caught from either side.
    #[test]
    fn kv_range_typed_error_on_wrong_length_out() {
        let (n_kv, n_q, head_dim, n_pos) = (4usize, 8usize, 8usize, 5usize);
        let group_dim = (n_q / n_kv) * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut rng = Rng::new(0x0117_0A7E);
        let (cache, _rows) = random_cache(&mut rng, 1, n_kv, head_dim, n_pos);
        let q = rng.vec_in(n_q * head_dim, -1.0, 1.0);
        let need = scratch_len(n_q, n_kv, head_dim, n_pos);
        let mut carve = vec![0.0f32; need];

        for range in [0..1, 0..2, 1..n_kv, 0..n_kv, 2..2] {
            let expected = range.len() * group_dim;
            // `q.len()` is the old full-vector contract; the others bracket
            // the right answer on both sides and cover the empty range.
            let mut wrong = vec![q.len(), expected + 1, 0, 1];
            if expected > 0 {
                wrong.push(expected - 1);
            }
            wrong.retain(|&len| len != expected);
            wrong.sort_unstable();
            wrong.dedup();

            for len in wrong {
                let mut out = vec![f32::NAN; len];
                assert_eq!(
                    decode_attention_kv_range_in(
                        &q,
                        &cache,
                        0,
                        scale,
                        range.clone(),
                        &mut carve,
                        &mut out
                    )
                    .unwrap_err(),
                    AttentionError::OutLenMismatch {
                        out_len: len,
                        expected,
                    },
                    "decode, range {range:?}, out {len}"
                );
                assert_eq!(
                    attention_at_kv_range_in(
                        &q,
                        &cache,
                        0,
                        n_pos,
                        scale,
                        range.clone(),
                        &mut carve,
                        &mut out
                    )
                    .unwrap_err(),
                    AttentionError::OutLenMismatch {
                        out_len: len,
                        expected,
                    },
                    "attention_at, range {range:?}, out {len}"
                );
                assert!(
                    out.iter().all(|x| x.is_nan()),
                    "range {range:?} out {len}: a rejected call wrote to `out`"
                );
            }

            // The range's own length is accepted, and every element of it is
            // written — no poison survives a successful call.
            for probe in 0..2 {
                let mut out = vec![f32::NAN; expected];
                if probe == 0 {
                    decode_attention_kv_range_in(
                        &q,
                        &cache,
                        0,
                        scale,
                        range.clone(),
                        &mut carve,
                        &mut out,
                    )
                    .unwrap();
                } else {
                    attention_at_kv_range_in(
                        &q,
                        &cache,
                        0,
                        n_pos,
                        scale,
                        range.clone(),
                        &mut carve,
                        &mut out,
                    )
                    .unwrap();
                }
                assert!(
                    out.iter().all(|x| x.is_finite()),
                    "range {range:?} probe {probe}: an accepted call left `out` unwritten"
                );
            }
        }
    }
}
