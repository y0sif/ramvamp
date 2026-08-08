//! Single-token forward pass over a loaded model.
//!
//! [`forward_token`] runs one token through every layer against a
//! [`ForwardState`] that owns the KV cache, the expert streamer, the pinned
//! compute pool, and every scratch buffer; in **steady state** the layer loop
//! allocates nothing.
//!
//! The qualifier is load-bearing rather than decorative. Everything this
//! module owns is sized once in [`ForwardState::new`] and never grows, and
//! so is the streamer's per-step bookkeeping: the six `Vec`s that would
//! otherwise grow on first use — `CachePlan`'s hit and miss lists, the open
//! step's hit, miss and protected lists, and the in-flight read table — are
//! all preallocated to `top_k` in [`ExpertStream::new`]. They live on the
//! streamer, one set per process, not on anything the layer loop rebuilds
//! per layer, so this is six allocations at construction and none after.
//! Every step clears and reuses them, and the layer loop's only remaining
//! allocations are [`Model::embed`]'s dequant `Vec` and, above this module,
//! the stream decoder's `String`. The claim holds from token 0.
//!
//! # Decode loop shape
//!
//! This is the phase-5 shape from `docs/architecture.md` ("Decode loop").
//! Per layer, once routing has produced the renormalized top-k:
//!
//! 1. [`ExpertStream::begin_layer`] plans the step against the layer's slot
//!    cache and submits a read for every miss. It does not block.
//! 2. **Every cache hit** is computed, row-parallel across the compute pool,
//!    each expert's `[hidden]` output staged into its own slot of
//!    [`ForwardState`]'s staging buffer. This is the work that overlaps the
//!    in-flight reads; Qwen3 has no shared expert, so hits are all there is
//!    to overlap with.
//! 3. [`ExpertStream::await_misses`] blocks until every submitted read has
//!    landed.
//! 4. **Every miss** is computed the same way, as one unit.
//! 5. The staged outputs are reduced in fixed top-k order:
//!    `acc += w[i] * staged[i]` for `i` in `0..top_k`.
//! 6. [`ExpertStream::end_layer`] hands the slots back.
//!
//! Steps 2 and 4 are deliberately coarse: per-expert progressive execution
//! as completions land is measured-and-rejected upstream (TurboFieldfare's
//! DEC-17, 4.799 -> 4.648 tok/s *with divergent output*), and flash-moe and
//! the llama.cpp prototype independently converged on the same two-phase
//! shape. Coarse hit/miss splitting was worth 14.4% for TF (DEC-18).
//!
//! # Bit-identity with the phase-4 baseline
//!
//! Phase 4 accumulated `acc += w_i * d_i` sequentially in top-k order.
//! Staging each expert's `[hidden]` output and reducing afterwards in that
//! same top-k order is the *identical* sequence of f32 operations, so the
//! order experts are actually computed in cannot move a single bit. The
//! renormalization sum (`topk.iter().map(|(_, w)| w).sum()`) is order
//! dependent too, and is left exactly as it was.
//!
//! Row-parallel GEMV is bit-identical by construction: each output row is an
//! independent dot product of one weight row against the shared activation
//! row, and [`crate::threads::shard_range`] is a pure function of
//! `(rows, shards)`. No dot product is ever split *within* a row — the
//! kernels fix their float accumulation order per super-block, and float
//! addition is not associative. See `kernels/gemv.rs`.
//!
//! That argument never said a fan-out may cover only one matrix, and phase 9
//! stopped assuming it did. A phase's routed experts run their gate and up
//! projections as **one** job over the concatenated row space, and their down
//! projections as one more. A partition of a concatenated row space restricts
//! to a partition of every
//! constituent matrix's own rows, so each row is still the same whole-row dot
//! on the same bytes, and the fused form is bit-identical to the sequence it
//! replaces (see [`run_expert_window`] and `kernels/gemv.rs`).
//!
//! # Reference fidelity
//!
//! Layer structure follows HF `transformers`
//! `models/qwen3_moe/modeling_qwen3_moe.py` exactly:
//!
//! - `Qwen3MoeDecoderLayer.forward` (pre-norm residual order):
//!   `residual = x; x = input_layernorm(x); x = self_attn(x);
//!   x = residual + x; residual = x; x = post_attention_layernorm(x);
//!   x = mlp(x); x = residual + x`.
//! - `Qwen3MoeAttention.forward`: per-head QK-RMSNorm is applied AFTER the
//!   q/k projections reshape into heads and BEFORE RoPE
//!   (`q_norm(q_proj(x).view(heads))` then `apply_rotary_pos_emb`). The
//!   `[head_dim]` norm weight is shared across heads.
//! - `Qwen3MoeSparseMoeBlock.forward`: router logits are computed in f32,
//!   softmax over all `n_experts` in f32, `topk` selects the `top_k`
//!   largest probabilities, and with `norm_topk_prob` the selected weights
//!   are renormalized to sum to 1 before weighting the expert outputs.
//!
//! Quantized arithmetic pairs each weight format with its audited
//! activation format (see `docs/architecture.md`, "Source quantization"):
//! k-quant weights (q4_k / q5_k / q6_k) dot Q8_K activations, the q8_0
//! `attn_k` dots Q8_0 activations. Activations are quantized once per
//! distinct input vector, not once per consumer.

use std::cell::Cell;
use std::fmt;
use std::ops::Range;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::format::ArchInfo;
use crate::io::{
    ExpertStream, IoError, StreamMode, StreamPhase, StreamStats, SweepError, SweepPlan,
};
use crate::kernels::KernelError;
use crate::kernels::attention::{
    AttentionError, decode_attention_in, decode_attention_kv_range_in,
    scratch_len as attention_scratch_len,
};
use crate::kernels::primitives::{
    rmsnorm, rmsnorm_in_place, rope_neox_heads, softmax, swiglu_combine, vec_add,
};
use crate::kernels::quants::{
    BlockQ8_0, BlockQ8K, QuantFormat, quantize_row_q8_0, quantize_row_q8_k,
};
// The fused row-space entry points come from `kernels::gemv` by path rather
// than through `kernels`'s re-export list: `pub mod gemv` already makes the
// path public, and this seam does not own `kernels/mod.rs`.
use crate::kernels::gemv::{FusedPart, fused_out_dim, fused_row_parts, gemv_q8_k_fused_rows};
use crate::kernels::{gemv_q8_0_rows, gemv_q8_k_rows};
use crate::kv::{KvCache, KvError};
use crate::threads::{ComputePool, PoolConfig};
use thiserror::Error;

use super::error::ModelError;
use super::prefill::{Phase, PhaseClock, PrefillConfig, PrefillMode, PrefillTiming, SendPtr};
use super::weights::Model;

/// Default total expert-cache budget: 1,440 MiB.
///
/// The dial is a **total** byte budget for the whole model, divided by the
/// summed per-layer blob strides to get slots per layer. On Qwen3-30B-A3B one
/// slot across all 48 layers costs `24 * 3,059,712 + 24 * 2,654,208` =
/// 137,134,080 B = 130.78 MiB, so 11 slots/layer costs **1,438.59 MiB** and
/// the budget must clear that. 1,438 MiB does not: it floors to 10.
///
/// 11 slots/layer is what the 3 GiB memory contract leaves once anonymous
/// runtime allocations are counted (EXP-012): 1,438.59 pool + 1,023.34 mmap'd
/// core + 384 KV + 115.1 anon = 2,961.03 MiB, 111.0 MiB spare. 12 would be
/// 19.8 MiB over.
pub const DEFAULT_CACHE_BYTES: u64 = 1440 * 1024 * 1024;

/// Runtime dials for [`ForwardState::with_config`].
///
/// These are the knobs that decide how much memory the expert cache may
/// hold and how wide the compute fan-out is. Neither changes the arithmetic:
/// the cache only changes *when* bytes are fetched, and row partitioning is
/// bit-identical by construction (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// Total expert-cache budget in bytes, divided across the layer count.
    /// Defaults to [`DEFAULT_CACHE_BYTES`].
    pub cache_bytes: u64,

    /// Compute shards (threads, counting the decode thread itself).
    /// `None` takes the pool's own topology detection, which picks one
    /// shard per physical P-core.
    pub threads: Option<usize>,

    /// Pin compute threads (and the decode thread) to their CPUs. Failures
    /// degrade to unpinned; see [`crate::threads`].
    pub pin: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            cache_bytes: DEFAULT_CACHE_BYTES,
            threads: None,
            pin: true,
        }
    }
}

impl RuntimeConfig {
    /// A tiny unpinned configuration for unit tests over the synthetic
    /// fixture installs: two shards, a 4 MiB budget, no affinity changes.
    ///
    /// Pinning is off because `cargo test` runs test bodies on many threads
    /// at once and a pin outlives the state that requested it, which would
    /// funnel the whole suite onto one core.
    #[cfg(test)]
    pub(crate) fn testing() -> Self {
        Self {
            cache_bytes: 4 * 1024 * 1024,
            threads: Some(2),
            pin: false,
        }
    }
}

/// Typed errors from the forward pass.
#[derive(Debug, Error)]
pub enum ForwardError {
    /// Weight access failed (bad layer/token index, load-time invariant).
    #[error(transparent)]
    Model(#[from] ModelError),

    /// A kernel rejected its input (quantize, gemv, or primitive shapes).
    #[error(transparent)]
    Kernel(#[from] KernelError),

    /// Decode attention failed.
    #[error(transparent)]
    Attention(#[from] AttentionError),

    /// KV cache append/access failed (including a full cache).
    #[error(transparent)]
    Kv(#[from] KvError),

    /// An expert blob read failed.
    #[error(transparent)]
    Io(#[from] IoError),

    /// An architecture dimension is not a whole number of activation
    /// quantization blocks, so the forward pass cannot run on it.
    #[error("forward: {what} = {dim} is not a multiple of the {block}-weight activation block")]
    UnsupportedDim {
        /// Which dimension is indivisible.
        what: &'static str,
        /// The offending dimension.
        dim: usize,
        /// The activation block width it must divide into.
        block: usize,
    },

    /// The router's `top_k` is zero or exceeds the expert count.
    #[error("forward: top_k {top_k} invalid for {n_experts} experts")]
    InvalidTopK {
        /// Configured experts per token.
        top_k: usize,
        /// Routed experts per layer.
        n_experts: usize,
    },

    /// The streamer's per-layer plan named an expert slot outside the
    /// routed set, or named one twice. Reported rather than trusted:
    /// staging into the wrong slot would silently weight the wrong expert.
    #[error(
        "forward: layer {layer}: the expert stream plan is not a permutation \
         of the {top_k} routed experts (bad or repeated index {index})"
    )]
    StreamPlanIndex {
        /// The layer whose plan was rejected.
        layer: u32,
        /// The offending index into the routed set.
        index: usize,
        /// Routed experts this layer.
        top_k: usize,
    },

    /// The streamer's hits and misses together did not cover every routed
    /// expert, so some staging slot would have been reduced stale.
    #[error(
        "forward: layer {layer}: the expert stream plan covered {covered} of {top_k} routed experts"
    )]
    StreamPlanCoverage {
        /// The layer whose plan was rejected.
        layer: u32,
        /// Routed experts actually computed.
        covered: usize,
        /// Routed experts this layer.
        top_k: usize,
    },

    /// `position` does not continue the KV cache: tokens must be fed
    /// strictly in sequence order.
    #[error("forward: position {position}, but the kv cache holds {expected} positions")]
    PositionMismatch {
        /// The requested position.
        position: usize,
        /// The position the cache expects next.
        expected: usize,
    },

    /// `position` does not fit the RoPE kernel's `u32` position type.
    #[error("forward: position {position} exceeds u32::MAX")]
    PositionOverflow {
        /// The requested position.
        position: usize,
    },

    /// [`forward_token`] was handed a model whose architecture is not the
    /// one the [`ForwardState`] was built from. Every buffer in the state —
    /// the KV geometry, the staging slots, the activation blocks — is sized
    /// from the construction-time model, so the pair must agree.
    #[error(
        "forward: this state was built for a model with {what} = {built}, \
         but the model it was given has {what} = {given}"
    )]
    ArchMismatch {
        /// The architecture field that differs.
        what: &'static str,
        /// Its value in the model the state was built from.
        built: u32,
        /// Its value in the model passed to [`forward_token`].
        given: u32,
    },

    /// A chunked-prefill layer sweep failed. See
    /// [`crate::model::prefill_prompt`].
    #[error(transparent)]
    Sweep(#[from] SweepError),

    /// The expert slot slab cannot host a prefill chunk beside the sweep
    /// ring, even narrowed to a single row with one window in flight.
    #[error(
        "prefill: the expert slot pool holds {available} B, but the narrowest \
         chunk needs {needed} B of scratch plus its sweep ring"
    )]
    PrefillScratch {
        /// Bytes the narrowest viable carve would have needed.
        needed: u64,
        /// Bytes the slot pool actually reserved.
        available: u64,
    },

    /// The prefill scratch span cannot be aligned for a buffer the chunk
    /// needs. Ruled out by the slab's page-aligned base; checked rather than
    /// assumed, because the alternative is an unaligned reinterpretation.
    #[error("prefill: the scratch span cannot be aligned to {align} B")]
    PrefillScratchAlign {
        /// The alignment that could not be met.
        align: usize,
    },

    /// The configured prefill chunk is not a usable number of positions.
    #[error("prefill: chunk size {chunk} is not a usable number of positions")]
    InvalidPrefillChunk {
        /// The rejected chunk size.
        chunk: usize,
    },

    /// Prefill was handed an empty prompt: there is nothing to run and no
    /// logits to produce.
    #[error("prefill: no tokens to prefill")]
    EmptyPrefill,

    /// A layer sweep finished without computing every routed `(row, slot)`
    /// pair, so some staging slot would have been reduced stale — the
    /// chunked analogue of [`ForwardError::StreamPlanCoverage`].
    #[error("prefill: layer {layer}: the sweep covered {covered} of {expected} routed rows")]
    PrefillCoverage {
        /// The layer whose sweep came up short.
        layer: u32,
        /// Routed `(row, slot)` pairs actually computed.
        covered: usize,
        /// Routed `(row, slot)` pairs the chunk asked for.
        expected: usize,
    },

    /// A router selection named an expert the layer does not have. Reported
    /// rather than trusted: it would otherwise index past the routing index.
    #[error("prefill: routed expert {expert} is outside the layer's {n_experts} experts")]
    RoutedExpertOutOfRange {
        /// The offending expert id.
        expert: u32,
        /// Routed experts the layer has.
        n_experts: usize,
    },

    /// One chunk row selected the same expert twice, so more `(row, slot)`
    /// pairs route to it than the chunk has rows and its batch would not fit
    /// the scratch. Unreachable through the router — top-k selection blanks
    /// each winner — and refused rather than allowed to index past a buffer.
    #[error("prefill: expert {expert} is routed {count} times by only {rows} chunk rows")]
    RepeatedRoutedExpert {
        /// The over-subscribed expert.
        expert: u32,
        /// `(row, slot)` pairs that named it.
        count: usize,
        /// Rows the chunk actually holds.
        rows: usize,
    },
}

/// The architecture dimensions a [`ForwardState`]'s buffers were sized from.
///
/// [`forward_token`] takes the model and the state independently, and reads
/// `top_k`/`hidden`/`moe_intermediate` from the *argument* model while
/// `expert_staged`, `expert_done`, the activation blocks and the KV cache
/// were all sized from the *construction-time* model. Pairing a state with a
/// wider-`top_k` model would index out of bounds inside [`run_plan`] rather
/// than report, so the pair is fingerprinted and checked once per token.
///
/// Only the dimensions that size something are compared. `rope_theta`,
/// `rms_eps` and the rest are read fresh from the argument model every token
/// and nothing caches them, so they are a caller's business, not an
/// invariant of this state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ArchFingerprint {
    n_layers: u32,
    n_experts: u32,
    top_k: u32,
    hidden: u32,
    moe_intermediate: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    vocab: u32,
}

impl ArchFingerprint {
    fn of(arch: &ArchInfo) -> Self {
        Self {
            n_layers: arch.n_layers,
            n_experts: arch.n_experts,
            top_k: arch.top_k,
            hidden: arch.hidden,
            moe_intermediate: arch.moe_intermediate,
            n_heads: arch.n_heads,
            n_kv_heads: arch.n_kv_heads,
            head_dim: arch.head_dim,
            vocab: arch.vocab,
        }
    }

    /// Name the first dimension of `arch` that this state was not built for.
    fn check(&self, arch: &ArchInfo) -> Result<(), ForwardError> {
        let given = Self::of(arch);
        for (what, built, given) in [
            ("n_layers", self.n_layers, given.n_layers),
            ("n_experts", self.n_experts, given.n_experts),
            ("top_k", self.top_k, given.top_k),
            ("hidden", self.hidden, given.hidden),
            (
                "moe_intermediate",
                self.moe_intermediate,
                given.moe_intermediate,
            ),
            ("n_heads", self.n_heads, given.n_heads),
            ("n_kv_heads", self.n_kv_heads, given.n_kv_heads),
            ("head_dim", self.head_dim, given.head_dim),
            ("vocab", self.vocab, given.vocab),
        ] {
            if built != given {
                return Err(ForwardError::ArchMismatch { what, built, given });
            }
        }
        Ok(())
    }
}

/// Owned per-sequence state for [`forward_token`]: the KV cache, the expert
/// streamer, the pinned compute pool, and every scratch buffer the pass
/// writes, preallocated once so no per-token allocation happens after
/// construction.
pub struct ForwardState {
    /// The architecture every buffer below was sized from. Checked against
    /// the model [`forward_token`] is handed, once per token.
    arch: ArchFingerprint,
    kv: KvCache,
    /// Per-shard attention scratch,
    /// `[min(n_kv_heads, shards)][attention_scratch_len]`.
    ///
    /// One whole carve per compute shard that can reach it, rather than one
    /// shared buffer, because [`pool_decode_attention`] runs a kv-head range on
    /// every shard at once and the scores are the only state a call keeps.
    /// Sized from the context cap up front, so nothing on the decode path ever
    /// reallocates.
    ///
    /// **`min(n_kv_heads, shards)`, not `shards`.** The fan-out's unit count is
    /// `shards`, but its work axis is the kv head: [`kv_head_range`] hands the
    /// slots past `n_kv_heads` an empty range, and [`pool_decode_attention`]'s
    /// closure returns on an empty range *before* it touches scratch. A carve
    /// for a surplus slot could therefore never be read or written — at the v0
    /// pin that was 2 of 6 carves, 264,192 B zero-filled at construction and
    /// dead for the process's life. The reachable count is the bound the
    /// closure's own indexing argument establishes (see there), so this is the
    /// size that argument actually supports.
    ///
    /// **Bytes.** One carve is `scratch_len(32, 4, 128, 4096) = 33_024` f32 =
    /// 132,096 B at the v0 pin, so this field is `min(n_kv_heads, shards) *
    /// 132_096` B: **528,384 B (516 KiB)** on the six-shard reference machine,
    /// where `n_kv_heads = 4` is the binding term — and it stays 528,384 B on
    /// any wider pool, because the cap is the cache's kv head count. Decode
    /// attention held exactly one such carve before the fan-out, so the
    /// fan-out's *new* anonymous memory is `(min(n_kv_heads, shards) - 1) *
    /// 132_096` B = **396,288 B (387 KiB)** at four reachable shards. The
    /// query-head slab also carried two `[n_heads * head_dim]` permutation
    /// buffers, `2 * 4096 * 4 = 32,768 B`, which the kv-head split does not
    /// need — a kv head's query heads are already contiguous — so they are
    /// gone, and the figure c78981e recorded (693,248 B, "+677 KiB net") comes
    /// down by that and by the 264,192 B of unreachable carves above.
    ///
    /// This is **not** the prefill arena's `attn_shards` carve, which is
    /// `shards * attention_scratch_len` and stays that way: prefill fans out
    /// over *rows*, so every shard there is reachable and the 792,576 B
    /// `prefill::tests` pins is a different buffer with a different bound.
    ///
    /// This is heap, not arena: there is no [`crate::io::PrefillSession`]
    /// outside a prefill, which is why prefill's equivalent comes out of the
    /// session arena instead (see `prefill::scratch_bytes`) and decode's does
    /// not. It is charged against a **provisional** ~111 MiB of headroom
    /// (EXP-014, provisional under that entry's own provenance correction),
    /// which already carries an unexplained 99-105 MiB residual between
    /// EXP-014 and EXP-018. 387 KiB is small against both, but it is not free
    /// and the headroom it is charged against is not measured.
    attn_scratch: Vec<f32>,
    /// Residual stream (`[hidden]`).
    hidden: Vec<f32>,
    /// RMSNorm output (`[hidden]`), reused by every norm site.
    normed: Vec<f32>,
    /// Q8_K quantization of a `[hidden]` vector (`hidden / 256` blocks).
    acts_q8k_hidden: Vec<BlockQ8K>,
    /// Q8_K quantization of one `[moe_intermediate]` vector per routed expert
    /// in a fused window (see [`FfnScratch`]).
    acts_q8k_moe: Vec<BlockQ8K>,
    /// Q8_K quantization of a `[n_heads * head_dim]` vector.
    acts_q8k_attn: Vec<BlockQ8K>,
    /// Q8_0 quantization of a `[hidden]` vector (`hidden / 32` blocks),
    /// consumed by the q8_0 `attn_k` gemv.
    acts_q8_0_hidden: Vec<BlockQ8_0>,
    /// Query projection output (`[n_heads * head_dim]`).
    q: Vec<f32>,
    /// Key projection output (`[n_kv_heads * head_dim]`).
    k: Vec<f32>,
    /// Value projection output (`[n_kv_heads * head_dim]`).
    v: Vec<f32>,
    /// Attention context output (`[n_heads * head_dim]`).
    attn_out: Vec<f32>,
    /// Output projection result (`[hidden]`).
    o_proj: Vec<f32>,
    /// Router logits (`[n_experts]`).
    router_logits: Vec<f32>,
    /// Router probabilities (`[n_experts]`), consumed by top-k selection.
    router_probs: Vec<f32>,
    /// Selected `(expert, weight)` pairs (`top_k` entries).
    topk: Vec<(u32, f32)>,
    /// The routed expert ids alone (`top_k` entries), the streamer's input.
    expert_ids: Vec<u32>,
    /// Gate and up projection outputs for one fused window of routed
    /// experts, `[slot][gate | up][moe_intermediate]`; the gate half then
    /// holds the SwiGLU combination in place. Sized for the window rather
    /// than for one expert because the window is one fan-out — see
    /// [`FfnScratch`] for the byte count.
    gate_up: Vec<f32>,
    /// Per-expert staged down-projection outputs, `top_k * hidden` f32
    /// (64 KiB at the v0 dims). Reduced in fixed top-k order after both
    /// compute phases, which is what keeps the pass bit-identical.
    expert_staged: Vec<f32>,
    /// Which staging slots this layer's plan has filled (`top_k` flags).
    expert_done: Vec<bool>,
    /// Weighted expert accumulator (`[hidden]`).
    expert_acc: Vec<f32>,
    /// Output logits (`[vocab]`), valid after a `want_logits` pass.
    logits: Vec<f32>,
    /// Pinned row-parallel compute pool.
    pool: ComputePool,
    /// Expert cache + io_uring/O_DIRECT streamer.
    stream: ExpertStream,

    /// Which prefill path [`crate::model::prefill_prompt`] takes, and how
    /// wide its chunks are. Seeded from [`PrefillConfig::from_env`].
    prefill: PrefillConfig,
    /// The sweep's per-layer window geometry, reused across every layer of
    /// every chunk so only the first layer of the first prefill allocates.
    sweep_plan: SweepPlan,
    /// Deduplicated routed expert ids for the layer being swept, preallocated
    /// to `n_experts`. Lives here rather than in the session scratch because
    /// [`crate::io::PrefillSession::split`] takes it *while* handing that
    /// scratch back, and the two may not alias.
    routed: Vec<u32>,
    /// Where the last [`crate::model::prefill_prompt`] call's wall time went.
    ///
    /// Rearmed and zeroed by every prefill, so it describes one prompt rather
    /// than the process — the opposite of the streaming counters beside it.
    /// `forward_token` charges into it too, but only while a prefill has armed
    /// it, which is what keeps decode out of a prefill's numbers.
    prefill_timing: PrefillTiming,
    /// Where the decode that followed the last prefill spent its wall time.
    ///
    /// The same five phases and the same `PhaseClock`, charged by the same
    /// [`forward_token`] — the only difference is which accumulator the clock
    /// points at, which `prefill_charging` decides. Unlike `prefill_timing`
    /// this one *accumulates* across tokens: a per-token split would be noise,
    /// and the question it answers ("what share of decode is attention now")
    /// is a question about a run.
    ///
    /// Zeroed when a prefill is armed, so it describes the decode belonging to
    /// the most recent prompt rather than a whole chat session.
    decode_timing: PrefillTiming,
    /// The pooled-GEMV sub-split of that decode: for each site, this core's
    /// own share against the barrier wait.
    ///
    /// One level below `decode_timing`, charged by the same `forward_token`
    /// over regions that sit *inside* [`Phase::Projections`] and
    /// [`Phase::ExpertCompute`], and zeroed by exactly the events that zero
    /// `decode_timing` so the two blocks always describe the same tokens.
    /// EXP-023 measured those two coarse phases at ~248 ms of a 532 ms token
    /// and flat in context; this says whether that is arithmetic or waiting.
    decode_gemv: GemvSplit,
    /// Whether [`forward_token`] should charge phases into `prefill_timing`.
    ///
    /// True only while [`crate::model::prefill_prompt`] is running the
    /// token-major path. It is not a user-facing dial — instrumentation is
    /// always on for prefill — but decode and the token-major prefill run the
    /// *same* instrumented function, and folding a generated token's phases
    /// into a prompt's split would make every number in it a lie the moment
    /// generation started.
    prefill_charging: bool,
}

impl fmt::Debug for ForwardState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardState")
            .field("context_cap", &self.kv.capacity())
            .field("top_k", &self.topk.capacity())
            .field("shards", &self.pool.shards())
            .field("slots_per_layer", &self.stream.slots_per_layer())
            .field("cache_bytes", &self.stream.cache_bytes())
            .field("stream_mode", &self.stream.mode())
            .finish_non_exhaustive()
    }
}

impl ForwardState {
    /// Preallocate all state for a model and a context cap (positions the
    /// KV cache can hold; v0 runs with 4096), using [`RuntimeConfig`]'s
    /// defaults.
    ///
    /// # Errors
    ///
    /// [`ForwardError::UnsupportedDim`] when an architecture dimension is
    /// not a whole number of activation blocks;
    /// [`ForwardError::InvalidTopK`] on a nonsensical router config;
    /// [`ForwardError::Kv`] when the KV geometry is rejected;
    /// [`ForwardError::Io`] when the expert streamer cannot open the
    /// install or size its slot pool.
    pub fn new(model: &Model, context_cap: usize) -> Result<Self, ForwardError> {
        Self::with_config(model, context_cap, RuntimeConfig::default())
    }

    /// [`ForwardState::new`] with explicit runtime dials.
    ///
    /// # Errors
    ///
    /// Exactly [`ForwardState::new`]'s.
    pub fn with_config(
        model: &Model,
        context_cap: usize,
        config: RuntimeConfig,
    ) -> Result<Self, ForwardError> {
        let arch = model.arch();
        let hidden = arch.hidden as usize;
        let q_dim = arch.n_heads as usize * arch.head_dim as usize;
        let kv_dim = arch.n_kv_heads as usize * arch.head_dim as usize;
        let moe = arch.moe_intermediate as usize;
        let n_experts = arch.n_experts as usize;
        let top_k = arch.top_k as usize;
        let vocab = arch.vocab as usize;

        const QK_K: usize = 256;
        const QK8_0: usize = 32;
        for (what, dim, block) in [
            ("hidden", hidden, QK_K),
            ("hidden", hidden, QK8_0),
            ("moe_intermediate", moe, QK_K),
            ("n_heads * head_dim", q_dim, QK_K),
        ] {
            if dim % block != 0 {
                return Err(ForwardError::UnsupportedDim { what, dim, block });
            }
        }
        if top_k == 0 || top_k > n_experts {
            return Err(ForwardError::InvalidTopK { top_k, n_experts });
        }
        // Experts in one fused window, and so the width the window scratch is
        // sized for. A wider `top_k` is walked in several windows rather than
        // refused, so this caps an allocation and nothing else.
        let window = top_k.min(MAX_FUSED_EXPERTS);

        let kv = KvCache::new(
            arch.n_layers as usize,
            arch.n_kv_heads as usize,
            arch.head_dim as usize,
            context_cap,
        )?;

        let pool = ComputePool::with_config(PoolConfig {
            shards: config.threads,
            pin: config.pin,
            pin_caller: config.pin,
            inline_caller: true,
        });
        let stream = ExpertStream::new(
            model.dir(),
            model.manifest(),
            model.layout(),
            config.cache_bytes,
            model.load_options(),
        )?;
        tracing::info!(
            shards = pool.shards(),
            slots_per_layer = stream.slots_per_layer(),
            cache_bytes = stream.cache_bytes(),
            mode = ?stream.mode(),
            "decode runtime ready"
        );

        // One whole attention carve per *reachable* shard, sized from the
        // context cap and never grown: `scratch_len` is the *whole* buffer the
        // kernel needs (a group-major score block plus two conversion rows),
        // not just the score count, so passing the cap alone would
        // under-reserve by the GQA group factor and cost a reallocation on the
        // first token. `min(n_kv_heads, shards)` because `pool_decode_attention`
        // fans out over kv heads: a slot past `n_kv_heads` takes an empty range
        // and returns before touching scratch, so a carve for it is allocated,
        // zero-filled and never read. See the field's docs for the bytes. The
        // `.max(1)` is belt and braces — `KvCache::new` above already rejected a
        // zero `n_kv_heads` and a pool always has at least one shard — but the
        // serial fall-through hands the *whole* buffer to one call, so a
        // zero-carve buffer would fail every decode with `ScratchTooShort`.
        let attn_shard = attention_scratch_len(
            arch.n_heads as usize,
            arch.n_kv_heads as usize,
            arch.head_dim as usize,
            context_cap,
        );
        let attn_carves = (arch.n_kv_heads as usize).min(pool.shards()).max(1);
        Ok(Self {
            arch: ArchFingerprint::of(arch),
            kv,
            attn_scratch: vec![0.0; attn_shard.saturating_mul(attn_carves)],
            hidden: vec![0.0; hidden],
            normed: vec![0.0; hidden],
            acts_q8k_hidden: vec![BlockQ8K::default(); hidden / QK_K],
            acts_q8k_moe: vec![BlockQ8K::default(); window * (moe / QK_K)],
            acts_q8k_attn: vec![BlockQ8K::default(); q_dim / QK_K],
            acts_q8_0_hidden: vec![BlockQ8_0::default(); hidden / QK8_0],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn_out: vec![0.0; q_dim],
            o_proj: vec![0.0; hidden],
            router_logits: vec![0.0; n_experts],
            router_probs: vec![0.0; n_experts],
            topk: Vec::with_capacity(top_k),
            expert_ids: Vec::with_capacity(top_k),
            gate_up: vec![0.0; 2 * window * moe],
            expert_staged: vec![0.0; top_k * hidden],
            expert_done: vec![false; top_k],
            expert_acc: vec![0.0; hidden],
            logits: vec![0.0; vocab],
            pool,
            stream,
            prefill: PrefillConfig::from_env(),
            sweep_plan: SweepPlan::new(),
            routed: Vec::with_capacity(n_experts),
            prefill_timing: PrefillTiming::default(),
            decode_timing: PrefillTiming::default(),
            decode_gemv: GemvSplit::default(),
            prefill_charging: false,
        })
    }

    /// The prefill dials this state runs with.
    pub fn prefill_config(&self) -> PrefillConfig {
        self.prefill
    }

    /// Replace the prefill dials.
    ///
    /// This is the path-selection and chunk-size dial the CLI wires up; it is
    /// deliberately not a field on [`RuntimeConfig`], whose fields are
    /// constructed positionally by another crate.
    ///
    /// # Errors
    ///
    /// Whatever [`PrefillConfig::validate`] refuses:
    /// [`ForwardError::InvalidPrefillChunk`] for a zero chunk, and
    /// [`ForwardError::Sweep`] for a degenerate or oversized sweep dial.
    pub fn set_prefill_config(&mut self, config: PrefillConfig) -> Result<(), ForwardError> {
        self.prefill = config.validate()?;
        Ok(())
    }

    /// The last `want_logits` pass's `[vocab]` output.
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Positions appended to one layer of the KV cache.
    ///
    /// Unlike [`ForwardState::seq_len`] this does not require the layers to
    /// agree, which is what makes it usable during a chunked prefill (see
    /// [`crate::model::prefill_prompt`]).
    ///
    /// # Errors
    ///
    /// [`ForwardError::Kv`] when `layer` is outside the cache.
    pub fn kv_len(&self, layer: usize) -> Result<usize, ForwardError> {
        Ok(self.kv.len(layer)?)
    }

    /// Drop every cached position, keeping every allocation.
    ///
    /// For starting a fresh sequence on an existing state — a new REPL
    /// conversation, say — without rebuilding the ~1,438 MiB expert slot pool,
    /// the io_uring ring and the pinned compute pool, which is what
    /// constructing a new [`ForwardState`] costs.
    ///
    /// The KV planes are not zeroed: every read is bounded by the per-layer
    /// cursor this resets, so stale bits are unreachable. The expert cache,
    /// its LFU history and the streaming counters all survive on purpose —
    /// they describe the process, not the sequence.
    ///
    /// [`ForwardState::prefill_timing`] does **not** survive, for the same
    /// reason inverted: it describes one prefill of one sequence, and a
    /// per-phase split left over from the sequence that was just dropped would
    /// be read as belonging to the one that replaced it.
    pub fn reset(&mut self) {
        self.kv.clear();
        self.prefill_timing = PrefillTiming::default();
        self.decode_timing = PrefillTiming::default();
        self.decode_gemv = GemvSplit::default();
        self.prefill_charging = false;
    }

    /// Rewind the cached sequence to its first `new_len` positions, keeping
    /// every allocation.
    ///
    /// [`ForwardState::reset`] restricted to a prefix, and correct for the
    /// same reason at both levels:
    ///
    /// - **The cache.** Only the per-layer cursors move; no plane is zeroed,
    ///   because every read is bounded by the cursor this resets, so the f16
    ///   bits of the dropped suffix are unreachable and the next append
    ///   overwrites them in place. See [`KvCache::truncate`].
    /// - **Everything else.** Nothing outside the KV cache carries
    ///   sequence-position state: the RoPE position, the causal mask width and
    ///   the attention span are all derived from the cursor on every pass, and
    ///   [`forward_token`] re-validates `position == seq_len()` before using
    ///   it. There is no second place a rewind would have to reach.
    ///
    /// What survives is what [`ForwardState::reset`] deliberately preserves,
    /// and for the reason it gives: the expert slot cache, its LFU history and
    /// the streaming counters describe the process, not the sequence, and a
    /// rewind of the sequence is even less of an event for them than a reset
    /// is — the same experts are about to be read again. The timing and
    /// diagnostic fields are dropped, exactly as [`ForwardState::reset`] drops
    /// them: they describe one prefill of one sequence, and a split left over
    /// from a prompt that has just been cut back would be read as belonging to
    /// the one that replaces it.
    ///
    /// This is what lets a stateless chat request reuse the cache up to its
    /// longest common prefix with the previous turn and re-prefill only the
    /// divergent suffix.
    ///
    /// # Errors
    ///
    /// [`ForwardError::Kv`] wrapping
    /// [`KvError::RaggedLayers`](crate::kv::KvError::RaggedLayers) if a
    /// previous pass failed mid-token and left the cache ragged (use
    /// [`ForwardState::reset`] to recover), or
    /// [`KvError::TruncateBeyondLength`](crate::kv::KvError::TruncateBeyondLength)
    /// if `new_len` is past the cached length. Nothing moves on either error.
    pub fn truncate(&mut self, new_len: usize) -> Result<(), ForwardError> {
        self.kv.truncate(new_len)?;
        self.prefill_timing = PrefillTiming::default();
        self.decode_timing = PrefillTiming::default();
        self.decode_gemv = GemvSplit::default();
        self.prefill_charging = false;
        Ok(())
    }

    /// Borrow the pieces the chunked prefill driver needs, all at once.
    ///
    /// Field-by-field so the compute pool, the streamer, the KV cache and the
    /// scratch buffers can be held simultaneously — the same destructuring
    /// [`forward_token`] does, exposed to the sibling module.
    pub(super) fn prefill_parts(&mut self) -> PrefillParts<'_> {
        PrefillParts {
            kv: &mut self.kv,
            topk: &mut self.topk,
            logits: &mut self.logits,
            pool: &mut self.pool,
            stream: &mut self.stream,
            sweep_plan: &mut self.sweep_plan,
            routed: &mut self.routed,
            timing: &mut self.prefill_timing,
        }
    }

    /// Drop the previous prefill's numbers and start charging this one.
    ///
    /// Paired with [`ForwardState::close_prefill_timing`], which must run
    /// however the prefill ended: a state left charging would fold the next
    /// decode token's phases into a prompt that is already over.
    pub(super) fn arm_prefill_timing(&mut self, mode: PrefillMode, tokens: usize) {
        self.prefill_timing = PrefillTiming::started(mode, tokens);
        // The decode split is dropped with the prefill split it belongs
        // beside: in a chat session the numbers under this turn's heading must
        // be this turn's, not the session's running total. The GEMV sub-split
        // goes with it, or the two blocks would be over different token sets.
        self.decode_timing = PrefillTiming::default();
        self.decode_gemv = GemvSplit::default();
        self.prefill_charging = true;
    }

    /// Record the whole `prefill_prompt` call's wall time and stop charging.
    pub(super) fn close_prefill_timing(&mut self, total: Duration) {
        self.prefill_timing.total = total;
        self.prefill_charging = false;
    }

    /// Refuse a model this state's buffers were not sized for.
    ///
    /// # Errors
    ///
    /// [`ForwardError::ArchMismatch`] naming the first dimension that differs.
    pub(super) fn check_arch(&self, arch: &ArchInfo) -> Result<(), ForwardError> {
        self.arch.check(arch)
    }

    /// All appended K rows of one layer, f16 bits. Test-only: the planes are
    /// an implementation detail, and this exists so the prefill tests can
    /// compare what two paths left behind.
    #[cfg(test)]
    pub(super) fn kv_k_layer(&self, layer: usize) -> Result<&[u16], ForwardError> {
        Ok(self.kv.k_layer(layer)?)
    }

    /// All appended V rows of one layer; see [`ForwardState::kv_k_layer`].
    #[cfg(test)]
    pub(super) fn kv_v_layer(&self, layer: usize) -> Result<&[u16], ForwardError> {
        Ok(self.kv.v_layer(layer)?)
    }
}

/// The [`ForwardState`] pieces one chunked prefill borrows, split out so they
/// can be held at the same time.
pub(super) struct PrefillParts<'a> {
    pub(super) kv: &'a mut KvCache,
    /// `(expert, weight)` staging for the route sink.
    pub(super) topk: &'a mut Vec<(u32, f32)>,
    pub(super) logits: &'a mut Vec<f32>,
    pub(super) pool: &'a mut ComputePool,
    pub(super) stream: &'a mut ExpertStream,
    pub(super) sweep_plan: &'a mut SweepPlan,
    pub(super) routed: &'a mut Vec<u32>,
    /// The per-phase timing the chunk driver charges into.
    pub(super) timing: &'a mut PrefillTiming,
}

impl ForwardState {
    /// Positions appended so far (the position the next token must use).
    ///
    /// # Errors
    ///
    /// [`ForwardError::Kv`] if a previous pass failed mid-layer and left
    /// the cache ragged.
    pub fn seq_len(&self) -> Result<usize, ForwardError> {
        Ok(self.kv.seq_len()?)
    }

    /// The KV cache's position capacity (the `context_cap` given at
    /// construction).
    pub fn context_cap(&self) -> usize {
        self.kv.capacity()
    }

    /// Compute shards a GEMV is fanned out across, counting the decode
    /// thread itself.
    pub fn shards(&self) -> usize {
        self.pool.shards()
    }

    /// Expert slots the cache holds per layer, derived from the byte budget.
    pub fn slots_per_layer(&self) -> u32 {
        self.stream.slots_per_layer()
    }

    /// Total bytes the expert slot pool actually reserved.
    pub fn cache_bytes(&self) -> u64 {
        self.stream.cache_bytes()
    }

    /// The submission path the streamer achieved (which is not always the
    /// one it asked for; O_DIRECT can be silently downgraded).
    pub fn stream_mode(&self) -> StreamMode {
        self.stream.mode()
    }

    /// Cumulative expert-streaming counters since this state was built,
    /// summed over **every** phase.
    ///
    /// This is a whole-process figure and reads as one. **A steady-state
    /// decode hit rate has to come from [`ForwardState::stream_stats_in`]**,
    /// and what "hit rate" even means depends on which prefill path ran:
    ///
    /// - [`PrefillMode::Sweep`](super::PrefillMode::Sweep), the default: prefill bypasses the expert
    ///   cache entirely and reports through the sweep counters instead
    ///   ([`StreamStats::sweep_bytes_read`],
    ///   [`StreamStats::sweep_windows`] and their siblings). It contributes
    ///   **no** cache accesses, so [`StreamStats::hit_rate`] over the whole
    ///   run is a decode figure — but [`StreamStats::bytes_read`] is not the
    ///   whole story any more, because the prompt's bytes are in
    ///   `sweep_bytes_read` and nowhere else. A phase with zero
    ///   [`StreamStats::accesses`] is not an idle phase; ask
    ///   [`StreamStats::is_idle`].
    /// - [`PrefillMode::TokenMajor`](super::PrefillMode::TokenMajor), the A/B path: prefill *is* decode, one
    ///   [`forward_token`] per prompt token through the same cache, so a
    ///   prompt's worth of cold misses lands in these totals and folds into
    ///   any hit rate quoted from them. EXP-013 was published from exactly
    ///   that mistake, when this was the only path there was.
    ///
    /// Split by phase before quoting either one.
    pub fn stream_stats(&self) -> StreamStats {
        self.stream.stats()
    }

    /// The share of [`ForwardState::stream_stats`] recorded while the stream
    /// was in `phase`.
    pub fn stream_stats_in(&self, phase: StreamPhase) -> StreamStats {
        self.stream.stats_in(phase)
    }

    /// Where the **last** [`crate::model::prefill_prompt`] call's wall time
    /// went, split by phase.
    ///
    /// Unlike [`ForwardState::stream_stats`] this is not cumulative: every
    /// prefill zeroes it, and so does [`ForwardState::reset`]. In a multi-turn
    /// session it therefore describes the most recent turn's prompt.
    /// [`PrefillTiming::ran`] says whether it describes anything at all.
    ///
    /// Both prefill paths report through it, so
    /// [`PrefillMode::Sweep`](super::PrefillMode::Sweep) and
    /// [`PrefillMode::TokenMajor`](super::PrefillMode::TokenMajor) can be
    /// compared phase by phase; [`PrefillTiming::mode`] says which one these
    /// numbers came from.
    pub fn prefill_timing(&self) -> PrefillTiming {
        self.prefill_timing
    }

    /// Where the decode **since the last prefill** spent its wall time, split
    /// by the same phases.
    ///
    /// The counterpart to [`ForwardState::prefill_timing`], and read the same
    /// way, with three differences worth knowing:
    ///
    /// - [`PrefillTiming::mode`] is always `None` — decode has no path choice
    ///   to report — so [`PrefillTiming::ran`] is not the question to ask
    ///   here. [`PrefillTiming::tokens`] is: it counts tokens decoded, and is
    ///   zero when nothing has.
    /// - [`PrefillTiming::total`] is the sum of the per-token wall times, so
    ///   it excludes whatever the sampler and the generation loop do between
    ///   tokens. It is therefore slightly under the decode wall time a caller
    ///   measures around the loop, and [`PrefillTiming::other`] stays a
    ///   statement about `forward_token` rather than about the whole loop.
    /// - It survives across the whole generation and is zeroed by the next
    ///   prefill (and by [`ForwardState::reset`]), because the question is
    ///   what a *run* spent, not what one token did.
    pub fn decode_timing(&self) -> PrefillTiming {
        self.decode_timing
    }

    /// The pooled-GEMV sub-split of that same decode, one level finer:
    /// `(label, own, wait, scatters)` for `projections`, `experts`, `lm_head`
    /// and `router`, in that order.
    ///
    /// `own` is the submitting thread's time from just before a fan-out to the
    /// moment its **own** shard finished; `wait` is from there to the fan-out
    /// returning, i.e. the compute pool's barrier. Reading the two apart is the
    /// whole point: EXP-023 left ~248 ms of a 532 ms decode token in
    /// [`PrefillTiming::projections`] plus [`PrefillTiming::expert_compute`],
    /// flat in context and far under the throughput a single AVX2 core sustains
    /// on an L2-resident fixture — which phase 9 established is not a valid
    /// reference for a site that streams its weights cold, see
    /// [`GemvBucket`] — and a `wait`-heavy split blames the even row partition
    /// on a hybrid part while an `own`-heavy one blames the kernel or the
    /// memory system.
    ///
    /// `router` is **not** pooled — it is a serial `dot_f32` loop on the decode
    /// thread — so its `wait` is zero by construction, not measured.
    ///
    /// The four buckets are disjoint spans strictly inside the two coarse
    /// phases above, so their sum can never exceed
    /// `projections + expert_compute`; the gap is the non-GEMV work those
    /// phases also cover (the softmax and top-k scan, SwiGLU, the intermediate
    /// quantization, the expert view carves). Same lifetime as
    /// [`ForwardState::decode_timing`]: accumulated across a run's tokens,
    /// zeroed by the next prefill and by [`ForwardState::reset`].
    ///
    /// Returned as plain tuples rather than a struct because `ramvamp-core`
    /// does not print, exactly as [`PrefillTiming::phases`] is.
    pub fn decode_gemv_split(&self) -> [(&'static str, Duration, Duration, u64); 4] {
        self.decode_gemv.rows()
    }

    /// Attribute every expert request from the next [`forward_token`] on to
    /// `phase`.
    ///
    /// The stream sees one layer at a time and has no notion of a token
    /// boundary, let alone of where a prompt ends, so the phase is the
    /// caller's to declare. [`crate::generate::generate`] declares it per
    /// token; a caller driving [`forward_token`] itself owns it. Everything
    /// before the first call is attributed to [`StreamPhase::Prefill`].
    pub fn set_stream_phase(&mut self, phase: StreamPhase) {
        self.stream.set_phase(phase);
    }
}

/// Observer for the router's decision, called once per layer per token
/// with `(layer, top_k)` where `top_k` is the final `(expert, weight)`
/// selection in routed order (descending router probability, weights
/// already renormalized when `norm_topk_prob`).
///
/// This exists so the offline expert-cache simulator can capture real
/// routing traces (`scripts/lfu_sim.py`); the runtime passes `None` and
/// pays one null check per layer. The sink is called after the routing
/// decision is final and before any expert is read, so it observes but
/// cannot influence numerics.
pub type ExpertRouteSink<'a> = &'a mut dyn FnMut(u32, &[(u32, f32)]);

/// Plain f32 dot product with f32 accumulation, matching the reference
/// router matvec (HF computes router logits in f32).
#[inline]
pub(super) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        acc += x * y;
    }
    acc
}

/// Remember the error the **lowest-indexed** failing shard reported, without
/// panicking out of a worker thread.
///
/// Keyed on `shard.index` rather than on arrival order on purpose. Two shards
/// can fail in the same job — a chunk whose scratch is short fails on every
/// shard at once — and "whichever reached the mutex first" makes the reported
/// error a race between threads, so the same input reports a different error
/// from run to run and a failure cannot be reproduced from the message alone.
/// Lowest index is one extra comparison under a lock that is only taken on an
/// error path, and it makes the report a pure function of the input.
///
/// Generic over the error because the GEMV fan-outs report [`KernelError`] and
/// the attention fan-outs report [`AttentionError`]; the bookkeeping is the
/// same either way.
pub(super) fn record<E>(slot: &Mutex<Option<(usize, E)>>, index: usize, err: E) {
    let mut held = slot.lock().unwrap_or_else(PoisonError::into_inner);
    match held.as_ref() {
        Some((held_index, _)) if *held_index <= index => {}
        _ => *held = Some((index, err)),
    }
}

/// Take whatever [`record`] stored, dropping the shard index it was keyed on.
pub(super) fn taken<E>(slot: Mutex<Option<(usize, E)>>) -> Result<(), E> {
    match slot.into_inner().unwrap_or_else(PoisonError::into_inner) {
        Some((_, err)) => Err(err),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Where a decode token's GEMV time went: this core, or the barrier
// ---------------------------------------------------------------------------

/// One pooled site of the decode GEMV sub-split, as charged by [`GemvClock`].
///
/// Strictly finer than [`Phase`], and strictly *inside* it: every site here is
/// already inside [`Phase::Projections`] or [`Phase::ExpertCompute`], which is
/// what makes the two instruments checkable against each other rather than
/// merely printable side by side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemvSite {
    /// `attn_q`, `attn_k`, `attn_v`, `attn_output`: **four** fan-outs a
    /// layer, against resident (mmap'd) weights.
    Projections,
    /// gate, up and down for every routed expert, fused per phase: **two**
    /// fan-outs — a gate/up job and a down job — per window of each of the
    /// hit and miss phases that has any experts in it. At the v0 pin
    /// (`top_k` 8, so one window) that is two to four a layer, against
    /// streamed weights. Not `3 * top_k` any more; see
    /// [`run_expert_window`].
    Experts,
    /// The `[vocab]` head. One fan-out a token, and only when logits are
    /// wanted.
    LmHead,
    /// The f32 router matvec. **Not pooled** — it runs serial on the decode
    /// thread through [`dot_f32`] — so it has no barrier to wait at and its
    /// `wait` is structurally zero rather than measured.
    Router,
}

/// One site's decode cost, split into the part this core computed and the part
/// it spent at the barrier.
///
/// # What the split means
///
/// The pool runs shard 0 **inline on the submitting thread**
/// ([`PoolConfig::inline_caller`], which [`ForwardState::with_config`] always
/// sets): the submitter publishes the job, runs shard 0 itself, and only then
/// joins at the barrier. So the decode thread's own timeline through one
/// fan-out is
///
/// ```text
/// t0 --[ set-up + publish + this core's shard ]-- t1 --[ barrier ]-- t2
/// ```
///
/// and `own = t1 - t0`, `wait = t2 - t1`. EXP-023 left ~248 ms/token of GEMV
/// unexplained at 8.04 GB/s aggregate, and these two numbers separate the two
/// candidate causes: a `wait` near zero says the kernel (or the memory system
/// under it) is genuinely slow on this core, and a large `wait` says the even
/// [`shard_range`](crate::threads::shard_range) row split is wrong for a hybrid
/// 6 P + 8 E + 2 LP-E part and the stragglers are the cost.
///
/// # What that gap is *not* measured against
///
/// It used to be quoted against EXP-001's 9.61 GB/s on one warm core. That
/// comparison is refuted: EXP-001 dots a matrix small enough to sit in L2,
/// while decode reads every expert byte once from DRAM, so the two figures are
/// not the same quantity and the shortfall between them was an artefact of
/// pairing them. What phase 9 measured on the fused decode instead, warm at
/// ctx 512 over three runs (pooled GEMV bucket medians): **11.00 GB/s
/// aggregate**, and forcing a single shard moves that bucket from 11.42 s to
/// 16.35 s — **six cores buy 1.43x**, not 6x. A dispatch-bound site would scale
/// with cores; this one does not, so it is bound by the memory system, and the
/// `own`/`wait` split below is read against that and not against a
/// cache-resident kernel number.
///
/// # What lands on which side
///
/// `own` carries the [`Mutex`] the failure slot needs, the job descriptor's
/// construction, the publish (including the `futex` wake when workers are
/// parked) and this core's `1/shards` of the rows. `wait` carries the barrier
/// spin, the park, and the uncontended `take_panic` lock on the way out. Both
/// are attributed to the *fan-out*, which is what the question is about; the
/// components are named here so nobody reads `own` as pure arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GemvBucket {
    /// Set-up, publish and this core's share of the rows.
    own: Duration,
    /// The barrier: stragglers, worker wake latency, and whatever the even row
    /// split costs on cores of different speeds. Always zero for
    /// [`GemvSite::Router`], which never fans out.
    wait: Duration,
    /// Fan-outs charged, so ms/scatter is derivable without another counter.
    scatters: u64,
}

/// Every [`GemvSite`]'s bucket, for the decode since the last prefill.
///
/// Accumulated across tokens exactly as [`ForwardState::decode_timing`] is, and
/// zeroed by the same two events, so the two blocks always describe the same
/// tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GemvSplit {
    projections: GemvBucket,
    experts: GemvBucket,
    lm_head: GemvBucket,
    router: GemvBucket,
}

impl GemvSplit {
    /// The bucket `site` charges.
    fn bucket(&mut self, site: GemvSite) -> &mut GemvBucket {
        match site {
            GemvSite::Projections => &mut self.projections,
            GemvSite::Experts => &mut self.experts,
            GemvSite::LmHead => &mut self.lm_head,
            GemvSite::Router => &mut self.router,
        }
    }

    /// Every bucket in report order, as `(label, own, wait, scatters)`.
    ///
    /// An array of plain tuples rather than a struct for the same reason
    /// [`PrefillTiming::phases`] is one: `ramvamp-core` does not print, and the
    /// renderer wants rows.
    fn rows(&self) -> [(&'static str, Duration, Duration, u64); 4] {
        [
            (
                "projections",
                self.projections.own,
                self.projections.wait,
                self.projections.scatters,
            ),
            (
                "experts",
                self.experts.own,
                self.experts.wait,
                self.experts.scatters,
            ),
            (
                "lm_head",
                self.lm_head.own,
                self.lm_head.wait,
                self.lm_head.scatters,
            ),
            (
                "router",
                self.router.own,
                self.router.wait,
                self.router.scatters,
            ),
        ]
    }
}

thread_local! {
    /// When the pooled GEMV in flight finished **this thread's** shard.
    ///
    /// This is the one piece of the instrument that cannot live on the
    /// submitting thread's side of the call: `t1` is the instant
    /// [`ComputePool::run`] returns from `f(Shard { index: 0, .. })`, which is
    /// *inside* `run`, and the pool exposes no hook there. It is read from the
    /// closure instead, under two conditions that keep the worker hot path
    /// untouched:
    ///
    /// - only when the sub-split is armed, which the closure captures as a
    ///   plain `bool`, so a prefill's fan-outs and every non-decode caller take
    ///   a predicted-not-taken branch and nothing else;
    /// - only from `shard.index == 0`, which under
    ///   [`PoolConfig::inline_caller`] is the submitting thread itself. Every
    ///   worker shard evaluates one comparison and skips.
    ///
    /// So no worker ever reads a clock, no worker ever writes shared state, and
    /// the wake/park pattern the instrument is trying to measure is not itself
    /// perturbed. Thread-local rather than atomic for the same reason: the
    /// value never crosses a thread, and an atomic would put a store on the
    /// only line where a straggler's timing matters.
    ///
    /// Without `inline_caller` shard 0 is a worker, this cell is never written
    /// on the submitting thread, and [`GemvClock::close`] falls back to
    /// `own = t2 - t0`, `wait = 0`. That is a degenerate report, not a wrong
    /// one, and it is unreachable from [`ForwardState`], which hard-codes
    /// `inline_caller: true`.
    static SHARD0_DONE: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// A stopwatch over a [`GemvSplit`], charged per fan-out rather than per
/// region.
///
/// The counterpart of [`PhaseClock`] one level down, and armed the same way:
/// `None` charges nowhere and — this is the point — reads no clock at all, so a
/// disarmed pass costs one `Option` check per site.
///
/// **Three clock reads per pooled GEMV**, against [`PhaseClock`]'s one per
/// region boundary. Per decoded token at the v0 pin (48 layers, `top_k` 8):
/// `48 * 4 * 3` = 576 for the projections, at most `48 * 4 * 3` = 576 for the
/// experts (two fused fan-outs for each non-empty phase, and a layer has one
/// or two), 3 for `lm_head` and `48 * 2` = 96 for the serial router, so
/// **1,251 reads a token at worst and 963 when every layer's experts are all
/// hits or all misses** — derived by counting the sites, not measured. At the
/// ~27 ns `Instant::now()` measures on this machine's vDSO (the figure
/// [`forward_token_traced`] already quotes) that is ~34 µs against the ~532 ms
/// token EXP-023 measured: 0.006% of it, and an order below the run-to-run
/// noise of anything the split is read against.
///
/// It was 4,131 reads before phase 9 fused the expert fan-outs (`48 * 4 * 3`
/// plus `48 * 8 * 3 * 3`), against a design budget of 4,500. The instrument
/// did not get cheaper; there are simply far fewer jobs to charge.
struct GemvClock<'a> {
    /// `None` when this pass charges nowhere.
    split: Option<&'a mut GemvSplit>,
}

impl<'a> GemvClock<'a> {
    /// Charge into `split` only when `armed`.
    ///
    /// Armed for decode and disarmed for the token-major prefill, which is the
    /// same rule — and the same reason — as [`PhaseClock::new`]'s: the two
    /// paths run the *same* instrumented function, and a prompt's fan-outs
    /// belong to no decode.
    fn new(split: &'a mut GemvSplit, armed: bool) -> Self {
        Self {
            split: armed.then_some(split),
        }
    }

    /// A clock that charges nowhere, for a caller with no [`GemvSplit`] to
    /// hand it — the prefill path, and the tests that drive the expert phase
    /// directly.
    fn disarmed() -> Self {
        Self { split: None }
    }

    /// Open one fan-out: clear the previous stamp and read the clock.
    ///
    /// `None` on a disarmed clock, which is also the `stamp` flag the pooled
    /// helpers take — passing `opened.is_some()` is what keeps the closure's
    /// branch and this clock from ever disagreeing about whether a stamp is
    /// coming.
    #[inline]
    fn open(&self) -> Option<Instant> {
        self.split.as_ref()?;
        SHARD0_DONE.set(None);
        Some(Instant::now())
    }

    /// Close one fan-out, charging `own` and `wait` to `site`.
    #[inline]
    fn close(&mut self, site: GemvSite, opened: Option<Instant>) {
        let (Some(split), Some(t0)) = (self.split.as_deref_mut(), opened) else {
            return;
        };
        let t2 = Instant::now();
        // `None` when the closure never ran on this thread: `rows == 0`, which
        // `ComputePool::run` returns from without calling `f` at all. The whole
        // call is then this thread's and there was no barrier, which is exactly
        // what `t1 = t2` records.
        let t1 = SHARD0_DONE.replace(None).unwrap_or(t2);
        let bucket = split.bucket(site);
        // Saturating for the same reason `PhaseClock::charge` saturates: a
        // non-monotonic platform clock must not panic in library code.
        bucket.own += t1.saturating_duration_since(t0);
        bucket.wait += t2.saturating_duration_since(t1);
        bucket.scatters += 1;
    }

    /// Close a region that never fanned out, charging all of it to `own`.
    ///
    /// For [`GemvSite::Router`], whose matvec is serial on the decode thread:
    /// there is no barrier, so `wait` stays zero by construction rather than
    /// by measuring a zero.
    #[inline]
    fn close_serial(&mut self, site: GemvSite, opened: Option<Instant>) {
        let (Some(split), Some(t0)) = (self.split.as_deref_mut(), opened) else {
            return;
        };
        let now = Instant::now();
        let bucket = split.bucket(site);
        bucket.own += now.saturating_duration_since(t0);
        bucket.scatters += 1;
    }
}

/// Whole-matrix k-quant GEMV, fanned out over contiguous output-row ranges.
///
/// Bit-identical to `gemv_q8_k` over the same operands: the pool's shards
/// tile `0..out_dim` in ascending contiguous order and each output row is an
/// independent dot product (see the module docs).
pub(super) fn pool_gemv_q8_k(
    pool: &mut ComputePool,
    format: QuantFormat,
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    acts: &[BlockQ8K],
    out: &mut [f32],
) -> Result<(), KernelError> {
    pool_gemv_q8_k_stamped(pool, format, weight, in_dim, out_dim, acts, out, false)
}

/// [`pool_gemv_q8_k`], stamping [`SHARD0_DONE`] when `stamp` is set.
///
/// Split out rather than folded into the public helper so that every caller
/// that does not want the decode sub-split — the sweep, the token-major
/// prefill, the tests — keeps the shorter signature and passes no flag.
/// Numerically it is the same call: the stamp is a clock read after the
/// kernel has written its rows, and it moves no bit and no row boundary.
#[allow(clippy::too_many_arguments)]
fn pool_gemv_q8_k_stamped(
    pool: &mut ComputePool,
    format: QuantFormat,
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    acts: &[BlockQ8K],
    out: &mut [f32],
    stamp: bool,
) -> Result<(), KernelError> {
    let failure: Mutex<Option<(usize, KernelError)>> = Mutex::new(None);
    pool.scatter(out, |shard, chunk| {
        let index = shard.index;
        if let Err(err) = gemv_q8_k_rows(format, weight, in_dim, out_dim, acts, shard.rows, chunk) {
            record(&failure, index, err);
        }
        // Last, so it is the end of this core's arithmetic and not the middle
        // of it. See `SHARD0_DONE` for why a worker never reaches the read.
        if stamp && index == 0 {
            SHARD0_DONE.set(Some(Instant::now()));
        }
    });
    taken(failure)
}

/// A **fused** k-quant GEMV: several independent matrices' output rows as one
/// row space, fanned out as a single job over contiguous ranges of it.
///
/// Bit-identical to running each part through [`pool_gemv_q8_k`] in turn. The
/// pool's shards tile the fused space contiguously and ascending; that tiling
/// restricts to a contiguous ascending tiling of every part's own
/// `0..out_dim`; and each output row is one whole-row dot against its part's
/// own activation row. See [`crate::kernels::gemv`]'s module docs — this
/// helper adds the fan-out and nothing else.
///
/// The `out.len()` check is load-bearing rather than defensive:
/// [`ComputePool::scatter`] shards over `out`, so an `out` shorter than the
/// fused row space would quietly compute a prefix of it and report success.
fn pool_gemv_q8_k_fused(
    pool: &mut ComputePool,
    parts: &[FusedPart<'_>],
    out: &mut [f32],
    stamp: bool,
) -> Result<(), KernelError> {
    let total = fused_out_dim(parts);
    if out.len() != total {
        return Err(KernelError::LengthMismatch {
            what: FUSED_OUT_VS_SPACE,
            left: out.len(),
            right: total,
        });
    }
    let failure: Mutex<Option<(usize, KernelError)>> = Mutex::new(None);
    pool.scatter(out, |shard, chunk| {
        let index = shard.index;
        if let Err(err) = gemv_q8_k_fused_rows(parts, shard.rows, chunk) {
            record(&failure, index, err);
        }
        if stamp && index == 0 {
            SHARD0_DONE.set(Some(Instant::now()));
        }
    });
    taken(failure)
}

/// `what` when a fused destination does not match the fused row space.
const FUSED_OUT_VS_SPACE: &str = "pool fused gemv: out vs fused out_dim";

/// `what` when a fused part's destination span is not inside `dst`.
const FUSED_SPAN_BOUND: &str = "pool fused gemv: destination span end vs dst";

/// `what` when two fused parts' destination spans overlap or descend.
const FUSED_SPAN_ORDER: &str = "pool fused gemv: destination span start vs previous end";

/// `what` when `at` does not name one destination per part.
const FUSED_SPAN_COUNT: &str = "pool fused gemv: destination spans vs parts";

/// [`pool_gemv_q8_k_fused`] with one destination span per part instead of one
/// contiguous buffer: part `p`'s output row `r` lands at `dst[at[p] + r]`.
///
/// The expert-down job needs this. A phase's routed experts share one fused
/// row space, but each stages its `[hidden]` output into the slot of
/// `expert_staged` that its **routed** index names, and a phase's plan is a
/// subset of the routed set — so the destination is contiguous per part and
/// gapped across parts, which [`ComputePool::scatter`] cannot express.
///
/// `at` must name one span per part, strictly non-overlapping and ascending
/// (`at[p] + out_dim(p) <= at[p + 1]`), with every span inside `dst`. That is
/// exactly the disjointness the per-shard `&mut [f32]` reconstructions rest
/// on, so it is checked here rather than assumed by the caller.
///
/// # Errors
///
/// [`KernelError::LengthMismatch`] when `at` disagrees with `parts`, a span
/// runs past `dst`, or two spans overlap or descend; otherwise whatever the
/// kernel reports for the lowest-indexed shard that fails.
fn pool_gemv_q8_k_fused_at(
    pool: &mut ComputePool,
    parts: &[FusedPart<'_>],
    at: &[usize],
    dst: &mut [f32],
    stamp: bool,
) -> Result<(), KernelError> {
    if parts.len() != at.len() {
        return Err(KernelError::LengthMismatch {
            what: FUSED_SPAN_COUNT,
            left: at.len(),
            right: parts.len(),
        });
    }
    // Establishes both halves of the safety argument below in one walk:
    // every span ends inside `dst`, and no span starts before the previous
    // one ended. Saturating so an absurd `out_dim` reports a bound failure
    // rather than wrapping into a span that looks in range.
    let mut watermark = 0usize;
    for (part, &start) in parts.iter().zip(at) {
        let end = start.saturating_add(part.out_dim);
        if start < watermark {
            return Err(KernelError::LengthMismatch {
                what: FUSED_SPAN_ORDER,
                left: start,
                right: watermark,
            });
        }
        if end > dst.len() {
            return Err(KernelError::LengthMismatch {
                what: FUSED_SPAN_BOUND,
                left: end,
                right: dst.len(),
            });
        }
        watermark = end;
    }
    let total = fused_out_dim(parts);
    let failure: Mutex<Option<(usize, KernelError)>> = Mutex::new(None);
    let base = SendPtr(dst.as_mut_ptr());
    pool.run(total, |shard| {
        let index = shard.index;
        for (p, local) in fused_row_parts(parts, shard.rows.clone()) {
            let part = parts[p];
            let start = at[p] + local.start;
            let len = local.len();
            // SAFETY: the same invariant as `pool_decode_attention`'s —
            // **shard windows are the image of a tiling under an injective
            // map**. `fused_row_parts` carries the pool's tiling of
            // `0..total` to, per part, a tiling of that part's own
            // `0..out_dim` (`the_fused_row_map_covers_every_part_row_exactly_once`),
            // so distinct shards never name the same `(part, local row)`;
            // and the spans `at` names are pairwise disjoint by the
            // watermark walk above, so distinct parts never name the same
            // element of `dst` either. In bounds because
            // `at[p] + local.end <= at[p] + out_dim <= dst.len()`, also from
            // that walk. `dst` is mutably borrowed by this frame for the
            // whole call and `run` joins before returning, so no view
            // outlives the borrow.
            let chunk = unsafe { std::slice::from_raw_parts_mut(base.get().add(start), len) };
            if let Err(err) = gemv_q8_k_rows(
                part.format,
                part.weight,
                part.in_dim,
                part.out_dim,
                part.acts,
                local,
                chunk,
            ) {
                record(&failure, index, err);
                // Deterministic: the reported error is the lowest shard's,
                // and within a shard the first part in fused order.
                break;
            }
        }
        if stamp && index == 0 {
            SHARD0_DONE.set(Some(Instant::now()));
        }
    });
    taken(failure)
}

/// Whole-matrix Q8_0 GEMV, fanned out the same way as [`pool_gemv_q8_k`].
fn pool_gemv_q8_0(
    pool: &mut ComputePool,
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    acts: &[BlockQ8_0],
    out: &mut [f32],
    stamp: bool,
) -> Result<(), KernelError> {
    let failure: Mutex<Option<(usize, KernelError)>> = Mutex::new(None);
    pool.scatter(out, |shard, chunk| {
        let index = shard.index;
        if let Err(err) = gemv_q8_0_rows(weight, in_dim, out_dim, acts, shard.rows, chunk) {
            record(&failure, index, err);
        }
        if stamp && index == 0 {
            SHARD0_DONE.set(Some(Instant::now()));
        }
    });
    taken(failure)
}

/// Positions below which decode attention stays on the submitting thread.
///
/// A pure function of the geometry, never of a clock: a gate that read the
/// wall time would make the shard partition — and therefore the reduction
/// order — depend on how busy the machine was, and the whole bit-identity
/// argument rests on the partition being a pure function of its inputs.
///
/// **Measured, not assumed.** An in-process harness at the v0 pin geometry
/// (32 q heads : 4 kv heads : head_dim 128, six pinned shards on a 185H, warm),
/// timing serial [`decode_attention_in`] against this fan-out over the ladder
/// 1, 2, 4, 8, ... 4096 positions — median *and* min of 101 timed units after
/// 11 warmups, five repeats — and asserting the two agree to the bit at every
/// rung. One `ComputePool` per process, because a second pool built after
/// `pin_caller` has narrowed the caller's mask sees one CPU, pins nothing, and
/// reports the fan-out at ~0.4x.
///
/// From **8 positions upward** the fan-out won every rung of every repeat on
/// both statistics, by 1.66x to 2.56x, with no crossover anywhere above it.
/// Below 8 the answer is not stable: at 1, 2 and 4 positions the *median* swung
/// between 0.37x and 2.23x across the five repeats while the *min* stayed ahead
/// (1.15x to 2.10x) in 14 of 15 measurements. Both arms cost 3-14 µs there, so
/// what the medians are recording is a preempted worker on a machine shared
/// with other build lanes, not a property of the kernel — but a number that
/// swings either side of 1.0 is not a measurement that the fan-out wins, so the
/// gate is set at the lowest rung where it is one.
///
/// This costs nothing real. Positions 1..7 occur only on the first seven tokens
/// of a session, where one layer's attention is single-digit microseconds; the
/// gate is a pure function of `kv.len(layer)` and moves no bit either way,
/// since both paths are the same arithmetic in the same order.
///
/// The shape behind the numbers: the serial call is ~2.5 µs fixed plus ~0.95 µs
/// per position, and a fan-out is ~1.3 µs of barrier (measured directly,
/// `pool.run(6, |_| {})`) plus the same per-position work spread over
/// `min(n_kv_heads, shards)` workers, so the crossover belonging in the
/// single-digit positions is what the model predicts.
const ATTENTION_FANOUT_MIN_POSITIONS: usize = 8;

/// One decode step's GQA attention for one layer, fanned out across the
/// compute pool as **kv-head ranges**.
///
/// # Why the kv head is the axis
///
/// [`decode_attention_kv_range_in`] is kv-head-outer: it widens each K and V
/// element from f16 once per kv head and then reuses it across that head's
/// whole GQA group, and its AVX2 QK dot rides its eight lanes on that group.
/// Cutting the kv-head axis duplicates no conversion and narrows no vector —
/// every shard runs a full `group = 8` call over its own kv heads — and a kv
/// head's `group` query heads are contiguous in both `q` and `out`, so a shard
/// needs no gather, no scatter and no permutation buffer.
///
/// "Once per kv head" is exact on the scalar path and for V on both paths; on
/// the AVX2 path K is widened `ceil(group / 8)` times per kv head, because the
/// position sweep sits inside the kernel's eight-query-head chunk loop. At the
/// v0 pin's `group = 8` that factor is 1, and it is a property of the *kernel*,
/// identical in the serial call and in every shard — the split neither creates
/// it nor changes it. See `kernels::attention`'s module docs for the measured
/// ratios.
///
/// Cutting the **query-head** axis instead, which is what this function did
/// before, narrows the group that those eight lanes ride. A `k = 1` strided
/// slab issues the same vector-op count as a whole `group = 8` call with seven
/// lanes carrying zeros: measured on a 185H (warm, `n_kv` 4, `head_dim` 128,
/// 4096 positions) the whole call took 4293.8 µs and a `group = 1` slab 1981.1
/// µs, so eight slabs cost ~15.8 ms of CPU against 4.3 ms (derived; see
/// `kernels::attention`'s module docs). End to end the slab fan-out returned
/// ~1.8x at six shards for a 3.1x increase in total CPU work — work taken from
/// the same pinned cores the streamed-expert GEMVs need — and **inverted at
/// short context**: 64 positions measured serial 89 µs against fan-out 195 µs,
/// or 0.46x, and EXP-014's decode baseline sits at ~69 positions.
///
/// # The unit count, and why the split cannot disagree with the closure
///
/// The job is submitted as `pool.shards()` **unit slots**, not as `n_kv_heads`
/// rows, and each slot is mapped to a kv-head range by [`kv_head_range`]. That
/// indirection is a **correctness invariant, not an optimisation**, and it is
/// two separate facts:
///
/// 1. [`ComputePool::run`] runs a job with fewer rows than shards inline as a
///    single `Shard { index: 0, count: 1, rows: 0..rows }`. Submitting
///    `n_kv_heads` rows would therefore serialise the whole call on every
///    machine whose pool is wider than the cache's kv head count — which is
///    every non-hybrid part, where `Topology::detect` falls back to
///    `min(#allowed CPUs, available_parallelism())` (12, 16, 24, 32) against
///    `n_kv_heads = 4`. `rows == pool.shards()` is the one row count that
///    cannot degenerate.
/// 2. The split is read from `shard.rows`, the closure's **own argument**, and
///    never from `pool.shards()` recomputed inside the body. Whatever
///    [`ComputePool::run`] chooses to do, the `shard.rows` it hands out tile
///    `0..rows`, and [`kv_head_range`] carries a tiling of `0..units` to a
///    tiling of `0..n_kv_heads`; so every kv head is computed exactly once
///    whether the job ran on `shards` threads or collapsed to one. There is no
///    second source of truth for the two to disagree about. The old query-head
///    split computed its permutation from `pool.shards()` while the closure
///    could receive `count = 1`, which is why its `group < shards` guard was
///    load-bearing for correctness rather than for payoff.
///
/// Slots past `n_kv_heads`, when the pool is wider than the cache, receive an
/// **empty** kv range, which [`decode_attention_kv_range_in`] documents as a
/// legal no-op. The fan-out therefore reaches `min(n_kv_heads, pool.shards())`
/// workers on any machine instead of silently reaching one.
///
/// # How the shards share `out`
///
/// They do not. A shard is handed the **full** `q` — `&[f32]`, so sharing it
/// costs nothing and `group` stays derived from all `n_q_heads`, which is the
/// trap the query-head slab fell into — and its **own** `kv_heads.len() *
/// group_dim` window of `out`, indexed from zero. Every shard therefore
/// reconstructs exactly one `&mut [f32]`, over exactly the elements it owns.
///
/// That is not a tidiness point. Before, each shard rebuilt a `&mut [f32]` over
/// the *whole* of `out` and the writes were argued to be disjoint; but two
/// simultaneously live `&mut` over one allocation are undefined behaviour under
/// both Stacked and Tree Borrows whether or not they ever touch the same
/// element, because creating the second invalidates the first. Windowing the
/// slice replaces that argument with a structure: `kv_head_range` tiles
/// `0..n_kv_heads` and `group_dim` is a constant stride, so the windows are
/// pairwise disjoint by construction rather than by appeal to the kernel's
/// contract.
///
/// # Bit-neutrality
///
/// A kv head's arithmetic depends on nothing but its own slice of `q`, the
/// immutable cache planes and its own scratch: the same K and V rows widened
/// the same way, each score accumulated over `i` ascending in one f32, the same
/// whole-row [`softmax`] over the same contiguous `positions`-long run, and the
/// V reduction over `t` ascending. Nothing is reduced across shards, nothing
/// accumulates in completion order, and the partition is a pure function of
/// `(n_kv_heads, pool.shards())`. The kernel pins the union of any partition
/// against the whole call in `kv_range_union_is_bit_identical_to_the_whole_call`;
/// `the_kv_range_split_is_bit_identical_to_the_whole_call` below pins this
/// function at every shard count, to the bit, with no tolerance.
///
/// # What it buys
///
/// The unit count is `n_kv_heads`, which is 4 at the v0 pin, not `shards`,
/// which is 6 — so six shards buy at most four, and that is the axis's real
/// ceiling now that redundancy is gone. Against it, this kernel-level fan-out
/// measures (185H, warm, six pinned shards, v0 pin geometry, median of 101,
/// five repeats) **~2.4x at 4096 positions and ~1.9x at 64**, against the
/// slab's 1.8x at 4096 and 0.46x at 64. The gap from 2.4x to the 4x ceiling is
/// the barrier, the two shards with no kv head to take, and multi-core clock
/// scaling; the machine also carried other build lanes while this was taken, a
/// contention the fan-out arm pays for and the serial arm does not, so treat
/// these as lower bounds.
///
/// The arithmetic itself is not duplicated — the fan-out widens each K and V
/// element exactly as many times as the serial call does, which is a structural
/// property of cutting the kv-head axis, not a measurement: the partition gives
/// each kv head to exactly one shard, so no shard repeats another's work. (That
/// count is once per element at the v0 pin; the kernel's own AVX2 K-widening
/// factor of `ceil(group / 8)` is 1 at `group = 8` and is the same on both
/// sides of the comparison either way.) Against that, the slab measured a 3.1x
/// increase in total CPU work taken from the same cores the streamed-expert
/// GEMVs need. These are **decode-attention** figures, not
/// end-to-end token throughput: attention's share of a decode token is the
/// question `decode_timing` exists to answer and is not settled here.
///
/// # Errors
///
/// [`ForwardError::Attention`] from the lowest-indexed shard that reports one;
/// on the serial path, whatever [`decode_attention_in`] returns. `out` is
/// **not** guaranteed untouched on error: shards write into `out` as they go
/// and a later shard may have completed before an earlier one failed. A caller
/// that sees an error must discard the token, which is what `forward_token`'s
/// contract already requires.
fn pool_decode_attention(
    pool: &mut ComputePool,
    kv: &KvCache,
    layer: usize,
    scale: f32,
    q: &[f32],
    scratch: &mut [f32],
    out: &mut [f32],
) -> Result<(), ForwardError> {
    let n_kv_heads = kv.n_kv_heads();
    let units = pool.shards();
    // Division-safe and error-safe rather than validated: a bad layer index or
    // an empty layer falls through to the single call below, which is the entry
    // point that owns those errors. A mismatched `out` does the same, and that
    // is also what makes the raw split below infallible.
    let positions = kv.len(layer).unwrap_or(0);
    // One kv head's slice of `out`, `group * head_dim` wide. Derived from the
    // **full** `q` (`q.len() == n_kv_heads * group * head_dim`), never from a
    // shard's range — the same derivation the kernel makes, for the same
    // reason. A `q` that is not a whole number of kv heads gives a stride that
    // would not tile `out`, so it joins the fall-through cases rather than
    // being split here; the single call below is the entry point that owns
    // `QLenIndivisible` and `GqaGroupMismatch`.
    let group_dim = q.len().checked_div(n_kv_heads).unwrap_or(0);
    if units < 2
        || n_kv_heads < 2
        || positions < ATTENTION_FANOUT_MIN_POSITIONS
        || out.len() != q.len()
        || group_dim == 0
        || group_dim * n_kv_heads != q.len()
    {
        decode_attention_in(q, kv, layer, scale, scratch, out)?;
        return Ok(());
    }

    // One equal carve per *reachable* unit slot, which is `min(n_kv_heads,
    // units)` and not `units`: slots past that take an empty kv range and
    // return below without touching `scratch`, so carving for them would divide
    // the buffer into slices that can never be read. `ForwardState` sizes its
    // buffer to exactly this many carves for the same reason.
    //
    // In bounds without a length check, by the same integer-division argument
    // as before, over the tighter bound:
    //
    // - A shard reaches the two lines below only if its kv range is nonempty.
    //   `kv_head_range`'s `first(u)` is constant for `u >= reachable` — it is
    //   `n_kv_heads` there — so a nonempty range forces
    //   `shard.rows.start < reachable`.
    // - `shard.rows` is `shard_range(units, count, index)`, whose start is
    //   `index * (units / count) + min(index, units % count)`. That is `>=
    //   index` when `units / count >= 1`, and exactly `index` in the only other
    //   case that yields a nonempty range (`units / count == 0` and `index <
    //   units % count`). `ComputePool::run`'s single-shard fallback hands out
    //   `index = 0`, which satisfies it trivially.
    //
    // So `index <= shard.rows.start < reachable`, hence `(index + 1) *
    // shard_scratch <= reachable * shard_scratch <= scratch.len()` by integer
    // division alone. (There used to be a length check here, against
    // `units * (scratch.len() / units)`, which is unsatisfiable — so the
    // `PrefillScratch` its doc promised could never be returned.) A carve too
    // short for the geometry is not silently absorbed either: the kernel
    // reports `ScratchTooShort` per shard, and `record` keeps the lowest
    // shard's.
    //
    // `reachable >= 2`: the fall-through above already returned unless both
    // `units >= 2` and `n_kv_heads >= 2`, so the division is safe.
    let reachable = n_kv_heads.min(units);
    let shard_scratch = scratch.len() / reachable;
    let failure: Mutex<Option<(usize, AttentionError)>> = Mutex::new(None);
    let out_base = SendPtr(out.as_mut_ptr());
    let scratch_base = SendPtr(scratch.as_mut_ptr());
    pool.run(units, |shard| {
        let kv_heads = kv_head_range(n_kv_heads, units, &shard.rows);
        if kv_heads.is_empty() {
            // The kernel accepts this as a no-op; returning early keeps a
            // surplus shard from touching `out` or `scratch` at all.
            return;
        }
        let index = shard.index;
        let lo = kv_heads.start * group_dim;
        let len = kv_heads.len() * group_dim;
        // SAFETY (`out`): the invariant is that **shard windows are the image
        // of a tiling under an injective map**, so no two are ever live at
        // once over the same element. `kv_head_range` carries the pool's
        // tiling of `0..units` to a tiling of `0..n_kv_heads`
        // (`the_kv_head_split_tiles_the_kv_heads`), and scaling a range by the
        // constant stride `group_dim` preserves disjointness; so distinct
        // shards reconstruct disjoint `&mut [f32]`, and each shard reconstructs
        // exactly one. In bounds because `kv_heads.end <= n_kv_heads` and
        // `n_kv_heads * group_dim == q.len() == out.len()`, both established
        // above. Nothing here rests on what the kernel does with the slice: it
        // is handed only the elements it owns, so a kernel that wrote outside
        // its range would be a wrong answer rather than undefined behaviour.
        //
        // SAFETY (`scratch`): the same invariant. Shard indices are distinct
        // across the job and each is visited by exactly one thread, so
        // `index * shard_scratch` names a disjoint `shard_scratch`-long run, in
        // bounds by the `index < reachable` argument above — which applies
        // because the empty-range return sits ahead of this point.
        //
        // Both buffers are mutably borrowed by this frame for the whole call
        // and `run` joins before returning, so no view outlives the borrow.
        let mine_out = unsafe { std::slice::from_raw_parts_mut(out_base.get().add(lo), len) };
        let mine_scratch = unsafe {
            std::slice::from_raw_parts_mut(
                scratch_base.get().add(index * shard_scratch),
                shard_scratch,
            )
        };
        if let Err(err) =
            decode_attention_kv_range_in(q, kv, layer, scale, kv_heads, mine_scratch, mine_out)
        {
            record(&failure, index, err);
        }
    });
    taken(failure)?;
    Ok(())
}

/// The kv heads owned by a shard holding unit slots `slots` out of `units`.
///
/// The one place the decode split is written down. Unit slot `u` owns
/// [`shard_range`](crate::threads::shard_range)`(n_kv_heads, units, u)`; those
/// ranges are contiguous, ascending
/// and tile `0..n_kv_heads`, so the union over a *contiguous run* of slots is
/// itself contiguous and is exactly `[first(slots.start), first(slots.end))`,
/// where `first(u)` is slot `u`'s start. Feeding this a tiling of `0..units`
/// therefore returns a tiling of `0..n_kv_heads`, which is the property
/// [`pool_decode_attention`] relies on to stay correct under
/// [`ComputePool::run`]'s single-shard fallback.
///
/// Written as one local closure rather than two
/// [`shard_range`](crate::threads::shard_range) calls because
/// `shard_range(_, units, units)` is `0..0` by its own out-of-range rule, while
/// the end of the last slot is precisely the `units` boundary — `n_kv_heads`.
///
/// An empty result (`start == end`) is the legal no-op a surplus shard gets
/// when the pool is wider than the cache has kv heads.
fn kv_head_range(n_kv_heads: usize, units: usize, slots: &Range<usize>) -> Range<usize> {
    if units == 0 {
        return 0..0;
    }
    let (base, rem) = (n_kv_heads / units, n_kv_heads % units);
    let first = |u: usize| {
        let u = u.min(units);
        u * base + u.min(rem)
    };
    let lo = first(slots.start);
    lo..first(slots.end).max(lo)
}

/// The three MoE dimensions the expert phase needs, bundled so the phase
/// helpers take one value instead of three positional `usize`s.
#[derive(Clone, Copy, Debug)]
struct MoeDims {
    hidden: usize,
    moe: usize,
    top_k: usize,
}

/// `what` when a window holds more experts than the parts buffers can carry.
const FUSED_WINDOW_PARTS: &str = "expert window: experts vs MAX_FUSED_EXPERTS";

/// `what` when the gate/up scratch is short for the window's fused row space.
const FUSED_WINDOW_GATE_UP: &str = "expert window: gate_up vs 2 * experts * moe_intermediate";

/// `what` when the intermediate blocks are short for the window's experts.
const FUSED_WINDOW_ACTS: &str = "expert window: intermediate blocks vs experts";

/// Routed experts one fused expert job may cover, and so the widest window
/// [`run_plan`] walks at a time.
///
/// The gate/up job carries two matrices per expert, so its parts buffer is
/// `2 * MAX_FUSED_EXPERTS` and lives on the stack: this runs twice a layer,
/// 48 layers a token, and the whole point of fusing is to stop paying
/// per-job set-up. A `top_k` wider than this is not refused — the plan is
/// walked in windows of this many *routed indices* — so the constant is a
/// buffer size and not an architecture limit. The v0 pin routes 8, so every
/// plan is one window.
const MAX_FUSED_EXPERTS: usize = 16;

/// The scratch one window of routed experts writes, reused by every window of
/// every phase of every layer.
///
/// Sized for a whole window rather than for one expert, because a window's
/// gate and up projections are a *single* fan-out and all of their outputs
/// are live at once. At the v0 pin (`top_k` 8, `moe_intermediate` 768) that is
/// 49,152 B of `gate_up` and 7,008 B of `acts_moe`, against 6,144 B and 876 B
/// for the per-expert form it replaces: **49,140 B more**, charged against the
/// same provisional decode headroom as [`ForwardState::attn_scratch`].
struct FfnScratch<'a> {
    /// Q8_K quantization of the layer's normed residual (`[hidden]`), shared
    /// by every routed expert's gate and up — which is what lets all of them
    /// share one fused row space.
    acts_hidden: &'a [BlockQ8K],
    /// Q8_K quantization of each window slot's SwiGLU output: one row of
    /// `moe_intermediate / 256` blocks per slot, because the down job reads
    /// all of them at once.
    acts_moe: &'a mut [BlockQ8K],
    /// Gate and up outputs for the window, `[slot][gate | up][moe]`. That is
    /// the fused row space's own layout, so the fan-out writes it directly
    /// and the SwiGLU reads one slot as a single `2 * moe` run.
    gate_up: &'a mut [f32],
}

/// Everything the expert phase writes, borrowed for one layer.
struct MoeScratch<'a> {
    /// Per-expert FFN scratch.
    ffn: FfnScratch<'a>,
    /// `top_k * hidden` staged expert outputs.
    staged: &'a mut [f32],
    /// Which staging slots are filled (`top_k` flags).
    done: &'a mut [bool],
}

/// gate/up -> SwiGLU -> down for one **window** of routed experts, staging
/// each one's `[hidden]` output into its own top-k slot of `scratch.staged`.
///
/// Two fan-outs, not `3 * window.len()`. Every expert's gate and up read the
/// same Q8_K activation row — the layer's normed residual, quantized once by
/// the caller — so their output rows concatenate into one fused row space
/// ([`FusedPart`]) and go out as a single job; the down projections then
/// concatenate the same way, each reading its *own* SwiGLU intermediate. The
/// two cheap `[moe_intermediate]` element-wise steps between them stay on the
/// decode thread, one expert at a time, exactly as before.
///
/// Phase 9 measured the **pre-fusion** decode fan-out at ~27 µs of
/// submitter-side set-up against ~32 µs of per-worker arithmetic, with 64% of
/// pooled GEMV time at the barrier and six cores buying only 1.16x over one;
/// the jobs were too small to pay for themselves. Both µs figures come off the
/// same warm ctx-512 arms rather than off a stopwatch around one job: fusing
/// moves no arithmetic, so the 16.35 s a forced single shard spends in the
/// pooled GEMV bucket is the serial arithmetic, and over the pre-fusion count
/// of 84,735 fan-outs that is 193 µs a fan-out, one sixth of which — this
/// core's shard — is ~32 µs; the set-up is what is left of the submitter's own
/// 5.04 s once that sixth is taken out, `(5.04 - 16.35 / 6) / 84,735`.
///
/// This is that arithmetic re-packed into fewer, larger jobs, and it moves no
/// bit: see [`crate::kernels::gemv`]'s fused-row-space docs. Fused, the same
/// arm has six cores buying **1.43x**, which is both what the re-packing bought
/// and what says the site is now bound by the memory system rather than by
/// dispatch (see [`GemvBucket`]).
///
/// `window` is `(routed index, slot)` in **ascending routed index**, which is
/// what makes the down job's staging spans ascending and provably disjoint.
fn run_expert_window(
    stream: &ExpertStream,
    pool: &mut ComputePool,
    layer: u32,
    dims: MoeDims,
    window: &[(usize, u32)],
    scratch: &mut MoeScratch<'_>,
    gemv: &mut GemvClock<'_>,
) -> Result<(), ForwardError> {
    let MoeScratch { ffn, staged, .. } = scratch;
    let n = window.len();
    let blocks = dims.moe / QuantFormat::Q8_K.block_weights();
    let acts_hidden: &[BlockQ8K] = ffn.acts_hidden;

    // Everything below indexes fixed-capacity buffers with widths derived from
    // `dims`, which is read off the *argument* model, while the scratch was
    // sized from the construction-time one and the parts arrays from
    // `MAX_FUSED_EXPERTS`. `ArchFingerprint::check` and `run_plan`'s windowing
    // make all three agree on the decode path, so none of these can fail
    // there — but a slice that panics when they disagree is not something
    // library code should rest on a check two frames up, and `run_plan` is
    // reachable from a caller with a hand-built plan. Reported, not asserted.
    let gate_up_len = 2usize.saturating_mul(n).saturating_mul(dims.moe);
    for (what, have, need) in [
        (FUSED_WINDOW_PARTS, MAX_FUSED_EXPERTS, n),
        (FUSED_WINDOW_GATE_UP, ffn.gate_up.len(), gate_up_len),
        (
            FUSED_WINDOW_ACTS,
            ffn.acts_moe.len(),
            n.saturating_mul(blocks),
        ),
    ] {
        if have < need {
            return Err(KernelError::LengthMismatch {
                what,
                left: have,
                right: need,
            }
            .into());
        }
    }

    // One pass over the window carves every slab it needs, so a slot's view
    // is resolved once rather than once per projection. `ExpertSlab` borrows
    // the streamer's slot bytes, not the `ExpertView`, so the view can go.
    let mut gate_up = [FusedPart::empty(); 2 * MAX_FUSED_EXPERTS];
    let mut down = [FusedPart::empty(); MAX_FUSED_EXPERTS];
    for (j, &(_, slot)) in window.iter().enumerate() {
        let view = stream.view(layer, slot)?;
        let (gate_slab, up_slab, down_slab) = (view.gate(), view.up(), view.down());
        gate_up[2 * j] = FusedPart {
            format: gate_slab.format,
            weight: gate_slab.bytes,
            in_dim: dims.hidden,
            out_dim: dims.moe,
            acts: acts_hidden,
        };
        gate_up[2 * j + 1] = FusedPart {
            format: up_slab.format,
            weight: up_slab.bytes,
            in_dim: dims.hidden,
            out_dim: dims.moe,
            acts: acts_hidden,
        };
        // `acts` is filled in below, once the SwiGLU has produced it.
        down[j] = FusedPart {
            format: down_slab.format,
            weight: down_slab.bytes,
            in_dim: dims.moe,
            out_dim: dims.hidden,
            acts: &[],
        };
    }

    let combined = &mut ffn.gate_up[..gate_up_len];
    let opened = gemv.open();
    pool_gemv_q8_k_fused(pool, &gate_up[..2 * n], combined, opened.is_some())?;
    gemv.close(GemvSite::Experts, opened);

    // Indexed rather than `chunks_exact_mut(2 * moe)`, which panics on a zero
    // chunk width; a degenerate `moe_intermediate` has to fall out as `n`
    // empty slots, not as a panic.
    for j in 0..n {
        let slot = &mut combined[j * 2 * dims.moe..(j + 1) * 2 * dims.moe];
        let (gate, up) = slot.split_at_mut(dims.moe);
        swiglu_combine(gate, up)?;
        quantize_row_q8_k(gate, &mut ffn.acts_moe[j * blocks..(j + 1) * blocks])?;
    }

    // The down job's destinations are the routed slots of `staged`, which the
    // phase's plan covers only in part, so they are contiguous per expert and
    // gapped across experts — ascending because `window` is.
    let intermediates: &[BlockQ8K] = ffn.acts_moe;
    let mut at = [0usize; MAX_FUSED_EXPERTS];
    for (j, &(index, _)) in window.iter().enumerate() {
        down[j].acts = &intermediates[j * blocks..(j + 1) * blocks];
        at[j] = index * dims.hidden;
    }
    let opened = gemv.open();
    pool_gemv_q8_k_fused_at(pool, &down[..n], &at[..n], staged, opened.is_some())?;
    gemv.close(GemvSite::Experts, opened);
    Ok(())
}

/// Compute one phase of a layer's plan — the cache hits, or the misses —
/// as a single unit, staging each expert's output into its top-k slot.
///
/// `plan` is `(index into the routed set, slot)`, exactly the encoding
/// [`ExpertStream::hits`] and [`ExpertStream::misses`] report. It is passed
/// in rather than read off `stream` so that the anti-staleness guard below
/// is reachable from a test with a hand-built plan: it is the check that
/// stops a stale `expert_staged` slot from being reduced into the residual,
/// and it has to be pinned by something other than a healthy streamer.
///
/// The whole plan is validated **before** any arithmetic. A window below is
/// one fan-out, so there is no longer a "between two experts" for a bad index
/// to be caught at, and a plan that is not a partial permutation of the
/// routed set must not reach the pool at all. The flags this sets on the way
/// through are what a repeat within the plan, or across the two phases, trips
/// on — unchanged, only earlier.
///
/// The two phases are still separate calls, so **no fusion crosses the
/// hit/miss boundary**: a resident expert's arithmetic never waits on a
/// missing expert's read, which is the overlap the two-phase decode exists
/// for and is worth more than the two fan-outs merging them would save.
///
/// `stream` is taken shared: every [`ExpertView`](crate::io::ExpertView)
/// borrows it for the length
/// of the phase, and nothing here needs to mutate it.
fn run_plan(
    stream: &ExpertStream,
    pool: &mut ComputePool,
    layer: u32,
    dims: MoeDims,
    plan: &[(usize, u32)],
    scratch: &mut MoeScratch<'_>,
    gemv: &mut GemvClock<'_>,
) -> Result<(), ForwardError> {
    for &(index, _) in plan {
        if index >= dims.top_k || scratch.done[index] {
            return Err(ForwardError::StreamPlanIndex {
                layer,
                index,
                top_k: dims.top_k,
            });
        }
        scratch.done[index] = true;
    }

    // Windows are ranges of the *routed* index rather than of the plan's own
    // order, so a window's staging spans come out ascending and disjoint,
    // which is what `pool_gemv_q8_k_fused_at` needs before it hands shards
    // raw sub-slices of `staged`. The scan is `top_k` finds over a plan of at
    // most `top_k` entries — 64 integer compares at the v0 pin, against the
    // ~27 µs of set-up phase 9 measured a fan-out to cost (`run_expert_window`
    // carries that figure's derivation).
    let mut window = [(0usize, 0u32); MAX_FUSED_EXPERTS];
    let mut base = 0usize;
    while base < dims.top_k {
        let end = base.saturating_add(MAX_FUSED_EXPERTS).min(dims.top_k);
        let mut n = 0usize;
        for index in base..end {
            if let Some(&(_, slot)) = plan.iter().find(|&&(i, _)| i == index) {
                window[n] = (index, slot);
                n += 1;
            }
        }
        if n > 0 {
            run_expert_window(stream, pool, layer, dims, &window[..n], scratch, gemv)?;
        }
        base = end;
    }
    Ok(())
}

/// The overlapped body of one layer's expert phase, between
/// [`ExpertStream::begin_layer`] and [`ExpertStream::end_layer`], with the
/// step's reads drained on every error path.
///
/// The drain is not optional. The hits phase runs *before*
/// [`ExpertStream::await_misses`], so a failure in it — a shard's
/// [`KernelError`], a rejected plan, a slot whose view cannot be carved —
/// returns with this step's misses still in the ring, writing into slots the
/// cache is protecting. [`ExpertStream::end_layer`] then cannot release a
/// `Filling` slot, `outstanding` never drains, and every subsequent
/// [`ExpertStream::begin_layer`] on *any* layer is refused for the rest of
/// the process. [`ExpertStream::begin_layer`]'s own failure path already
/// drains for the same reason; this is the symmetric half.
fn stream_experts(
    stream: &mut ExpertStream,
    pool: &mut ComputePool,
    layer: u32,
    dims: MoeDims,
    scratch: &mut MoeScratch<'_>,
    clock: &mut PhaseClock<'_>,
    gemv: &mut GemvClock<'_>,
) -> Result<(), ForwardError> {
    let outcome = stage_expert_phases(stream, pool, layer, dims, scratch, clock, gemv);
    if outcome.is_err()
        && let Err(drain) = stream.await_misses()
    {
        // The drain itself failed, which means the streamer has already
        // invalidated or leaked the slots involved: the reads can no longer
        // strand anything. The first error is the one worth propagating.
        tracing::error!(
            layer,
            error = %drain,
            "draining a failed expert phase also failed"
        );
    }
    outcome
}

/// Hits first — they are the work that covers the in-flight reads — then a
/// single block on every miss, then the misses as one unit. Neither phase
/// reduces anything: both stage into `scratch.staged`, and the caller
/// reduces in fixed top-k order.
///
/// Every exit from here is an error the caller must drain behind; see
/// [`stream_experts`].
///
/// The three phase boundaries are what let a token-major prefill report the
/// same split a swept one does: the two `run_plan` calls are expert
/// arithmetic, and the block between them is expert I/O. `await_misses`'s
/// blocked time is separately (and independently) counted by the streamer as
/// [`StreamStats::io_wait`], so nothing here re-times the drive.
#[allow(clippy::too_many_arguments)]
fn stage_expert_phases(
    stream: &mut ExpertStream,
    pool: &mut ComputePool,
    layer: u32,
    dims: MoeDims,
    scratch: &mut MoeScratch<'_>,
    clock: &mut PhaseClock<'_>,
    gemv: &mut GemvClock<'_>,
) -> Result<(), ForwardError> {
    scratch.done.fill(false);
    run_plan(stream, pool, layer, dims, stream.hits(), scratch, gemv)?;
    clock.charge(Phase::ExpertCompute);
    stream.await_misses()?;
    clock.charge(Phase::ExpertIo);
    run_plan(stream, pool, layer, dims, stream.misses(), scratch, gemv)?;
    clock.charge(Phase::ExpertCompute);
    let covered = scratch.done.iter().filter(|filled| **filled).count();
    if covered != dims.top_k {
        return Err(ForwardError::StreamPlanCoverage {
            layer,
            covered,
            top_k: dims.top_k,
        });
    }
    Ok(())
}

/// Run one token through the whole model at `position`.
///
/// `position` must equal the number of positions already in the KV cache
/// (tokens are fed strictly in order; prefill is this call in a loop). With
/// `want_logits` the final norm + lm_head run and the `[vocab]` logits are
/// returned; without it the tail is skipped (prefill of non-final prompt
/// tokens) and `None` is returned.
///
/// # Errors
///
/// [`ForwardError::PositionMismatch`] / [`ForwardError::PositionOverflow`]
/// on a bad `position`; [`ForwardError::Model`] on a bad `token_id`;
/// [`ForwardError::Kv`] when the context cap is exhausted; kernel, cache,
/// and I/O failures pass through typed. On error the state must be assumed
/// mid-token (the KV cache may be ragged) and discarded.
pub fn forward_token<'s>(
    model: &Model,
    state: &'s mut ForwardState,
    token_id: u32,
    position: usize,
    want_logits: bool,
) -> Result<Option<&'s [f32]>, ForwardError> {
    forward_token_traced(model, state, token_id, position, want_logits, None)
}

/// [`forward_token`] with an optional [`ExpertRouteSink`] observing every
/// layer's routing decision.
///
/// Numerically identical to [`forward_token`]: the sink runs after the
/// top-k selection and renormalization are final and touches no state.
/// With `on_route` `None` the only cost is one `Option` check per layer,
/// and nothing is allocated either way.
///
/// # Errors
///
/// Exactly [`forward_token`]'s.
pub fn forward_token_traced<'s>(
    model: &Model,
    state: &'s mut ForwardState,
    token_id: u32,
    position: usize,
    want_logits: bool,
    mut on_route: Option<ExpertRouteSink<'_>>,
) -> Result<Option<&'s [f32]>, ForwardError> {
    let arch = model.arch();
    // `dims` below is read off this model, while every buffer in `state` was
    // sized from the one it was built with. A mismatched pair would index
    // out of bounds inside `run_plan`, so it is refused up front.
    state.arch.check(arch)?;
    let n_heads = arch.n_heads as usize;
    let n_kv_heads = arch.n_kv_heads as usize;
    let head_dim = arch.head_dim as usize;
    let hidden = arch.hidden as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let dims = MoeDims {
        hidden,
        moe: arch.moe_intermediate as usize,
        top_k: arch.top_k as usize,
    };
    let eps = arch.rms_eps as f32;
    let theta = arch.rope_theta as f32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Field-by-field, so the compute pool, the streamer and the scratch
    // buffers can be borrowed independently inside the layer loop.
    let ForwardState {
        // Already checked against `arch` above.
        arch: _,
        kv,
        attn_scratch,
        hidden: residual,
        normed,
        acts_q8k_hidden,
        acts_q8k_moe,
        acts_q8k_attn,
        acts_q8_0_hidden,
        q,
        k,
        v,
        attn_out,
        o_proj,
        router_logits,
        router_probs,
        topk,
        expert_ids,
        gate_up,
        expert_staged,
        expert_done,
        expert_acc,
        logits,
        pool,
        stream,
        // Prefill-only state; `forward_token` neither reads nor advances it.
        prefill: _,
        sweep_plan: _,
        routed: _,
        prefill_timing,
        decode_timing,
        decode_gemv,
        prefill_charging,
    } = state;

    // The same instrumented pass, charging into whichever accumulator this
    // call belongs to: the prompt's while `prefill_prompt` is running the
    // token-major path, the run's decode split otherwise. One accumulator or
    // the other, never both — a token counted twice would make the two splits
    // sum past the wall time they are read against.
    //
    // **This reverses a documented hot-path choice, deliberately.** Until
    // c78981e the comment here promised that "a decode step therefore pays
    // exactly one `Instant::now()` for the whole token and a predicted-not-
    // taken branch per boundary", because `PhaseClock::new(_, prefill_charging)`
    // left decode's clock unarmed. Decode's clock is now always armed, so a
    // decode token pays one `clock_gettime` per phase boundary instead: 15
    // `clock.charge` sites per layer (12 here, 3 in `stage_expert_phases`) x 48
    // layers, plus 2 on the logits tail, plus the one `PhaseClock::decoding`
    // takes and the one `close` takes, is **724 reads per token** — derived by
    // counting the sites, not measured. `Instant::now()` measures ~27 ns on
    // this machine's vDSO (median of 31 x 1000, same harness as
    // `ATTENTION_FANOUT_MIN_POSITIONS`), so ~20 µs against a token EXP-018 puts
    // in the hundreds of milliseconds: order 1e-4 of the token, well under the
    // run-to-run noise of anything it is read against.
    //
    // It is kept armed rather than gated because the decode phase split is the
    // only instrument that says what share of a decode token attention now
    // costs, which is the question phase 7 exists to answer, and a dial that
    // has to be turned on is a dial that is off when the interesting run
    // happens. The honest statement is that decode instrumentation is no longer
    // free, that its cost is ~20 µs/token, and that this comment — not the old
    // promise — is the current contract.
    //
    // The GEMV sub-split rides the same choice one level down: armed for
    // decode, disarmed for the token-major prefill, so a prompt's fan-outs
    // never land in a decode's `own`/`wait`. Its own cost is a further ~1,251
    // clock reads (~34 µs) a decoded token and zero on the prefill path; see
    // [`GemvClock`] for the count and the arithmetic behind it.
    let (mut clock, mut gemv) = if *prefill_charging {
        (PhaseClock::new(prefill_timing, true), GemvClock::disarmed())
    } else {
        (
            PhaseClock::decoding(decode_timing),
            GemvClock::new(decode_gemv, true),
        )
    };

    let expected = kv.seq_len()?;
    if position != expected {
        return Err(ForwardError::PositionMismatch { position, expected });
    }
    let rope_pos =
        u32::try_from(position).map_err(|_| ForwardError::PositionOverflow { position })?;

    model.embed(token_id, residual)?;

    for layer in 0..model.n_layers() {
        let lw = model.layer(layer)?;
        let layer_idx = layer as usize;

        // Attention block: residual = hidden; x = attn_norm(hidden).
        rmsnorm(residual, lw.attn_norm, eps, normed)?;

        // Quantize the normed input both ways: Q8_K for the k-quant
        // projections, Q8_0 for the q8_0 attn_k projection.
        quantize_row_q8_k(normed, acts_q8k_hidden)?;
        quantize_row_q8_0(normed, acts_q8_0_hidden)?;
        clock.charge(Phase::Elementwise);

        let opened = gemv.open();
        pool_gemv_q8_k_stamped(
            pool,
            lw.attn_q.format,
            lw.attn_q.bytes,
            hidden,
            q_dim,
            acts_q8k_hidden,
            q,
            opened.is_some(),
        )?;
        gemv.close(GemvSite::Projections, opened);
        let opened = gemv.open();
        pool_gemv_q8_0(
            pool,
            lw.attn_k.bytes,
            hidden,
            kv_dim,
            acts_q8_0_hidden,
            k,
            opened.is_some(),
        )?;
        gemv.close(GemvSite::Projections, opened);
        let opened = gemv.open();
        pool_gemv_q8_k_stamped(
            pool,
            lw.attn_v.format,
            lw.attn_v.bytes,
            hidden,
            kv_dim,
            acts_q8k_hidden,
            v,
            opened.is_some(),
        )?;
        gemv.close(GemvSite::Projections, opened);
        clock.charge(Phase::Projections);

        // Per-head QK-RMSNorm, then RoPE — HF order: q_norm/k_norm apply
        // after the projection reshape and before rotary embedding
        // (`Qwen3MoeAttention.forward`). The [head_dim] weight is shared
        // across heads; same eps as the layer norms.
        for head in q.chunks_exact_mut(head_dim) {
            rmsnorm_in_place(head, lw.attn_q_norm, eps)?;
        }
        for head in k.chunks_exact_mut(head_dim) {
            rmsnorm_in_place(head, lw.attn_k_norm, eps)?;
        }
        rope_neox_heads(q, n_heads, head_dim, rope_pos, theta)?;
        rope_neox_heads(k, n_kv_heads, head_dim, rope_pos, theta)?;

        kv.append(layer_idx, k, v)?;
        clock.charge(Phase::Elementwise);
        pool_decode_attention(pool, kv, layer_idx, scale, q, attn_scratch, attn_out)?;
        clock.charge(Phase::Attention);

        // Output projection (q5_k, Q8_K activations) and residual add.
        quantize_row_q8_k(attn_out, acts_q8k_attn)?;
        clock.charge(Phase::Elementwise);
        let opened = gemv.open();
        pool_gemv_q8_k_stamped(
            pool,
            lw.attn_output.format,
            lw.attn_output.bytes,
            q_dim,
            hidden,
            acts_q8k_attn,
            o_proj,
            opened.is_some(),
        )?;
        gemv.close(GemvSite::Projections, opened);
        clock.charge(Phase::Projections);
        vec_add(residual, o_proj)?;

        // MoE block: residual = hidden; x = ffn_norm(hidden).
        rmsnorm(residual, lw.ffn_norm, eps, normed)?;
        clock.charge(Phase::Elementwise);

        // Router: f32 matvec (rows validated `[n_experts, hidden]` at
        // load), softmax over all experts in f32, top-k by probability
        // (equivalent to top-k by logit; first index wins ties like
        // torch.topk), then renormalize when norm_topk_prob.
        //
        // The matvec is the one GEMV on this path that never reaches the
        // compute pool, so the sub-split charges it serially: all `own`, no
        // barrier to wait at. Its two clock reads bracket the loop alone, not
        // the softmax and top-k scan the coarse `Projections` region also
        // covers, which is why the two instruments do not sum to each other
        // here.
        let opened = gemv.open();
        for (row, logit) in lw
            .router
            .data()
            .chunks_exact(hidden)
            .zip(router_logits.iter_mut())
        {
            *logit = dot_f32(row, normed);
        }
        gemv.close_serial(GemvSite::Router, opened);
        router_probs.copy_from_slice(router_logits);
        softmax(router_probs)?;
        topk.clear();
        for _ in 0..dims.top_k {
            let mut best_e = 0usize;
            let mut best_p = f32::NEG_INFINITY;
            for (e, &p) in router_probs.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best_e = e;
                }
            }
            topk.push((best_e as u32, best_p));
            router_probs[best_e] = f32::NEG_INFINITY;
        }
        if arch.norm_topk_prob {
            let sum: f32 = topk.iter().map(|&(_, w)| w).sum();
            for (_, w) in topk.iter_mut() {
                *w /= sum;
            }
        }
        // Charged to `Projections`: the f32 router matvec dominates the
        // softmax, the top-k scan and the sink it shares this region with.
        clock.charge(Phase::Projections);
        if let Some(sink) = on_route.as_deref_mut() {
            sink(layer, topk);
        }

        // Experts: quantize the normed input once, submit every miss, then
        // compute hits and misses as two coarse units, staging each
        // expert's `[hidden]` output.
        quantize_row_q8_k(normed, acts_q8k_hidden)?;
        expert_ids.clear();
        expert_ids.extend(topk.iter().map(|&(expert, _)| expert));
        clock.charge(Phase::Elementwise);

        stream.begin_layer(layer, expert_ids)?;
        clock.charge(Phase::ExpertIo);
        let mut scratch = MoeScratch {
            ffn: FfnScratch {
                acts_hidden: acts_q8k_hidden,
                acts_moe: acts_q8k_moe,
                gate_up,
            },
            staged: expert_staged,
            done: expert_done,
        };
        let phases = stream_experts(
            stream,
            pool,
            layer,
            dims,
            &mut scratch,
            &mut clock,
            &mut gemv,
        );
        stream.end_layer(layer);
        clock.charge(Phase::ExpertIo);
        phases?;

        // Fixed-order reduction: identical to the phase-4 sequential
        // `acc += w_i * d_i`, whatever order the experts were computed in.
        expert_acc.fill(0.0);
        for (i, &(_, weight)) in topk.iter().enumerate() {
            let staged = &expert_staged[i * hidden..(i + 1) * hidden];
            for (acc, &d) in expert_acc.iter_mut().zip(staged) {
                *acc += weight * d;
            }
        }
        vec_add(residual, expert_acc)?;
        clock.charge(Phase::Elementwise);
    }

    if !want_logits {
        clock.close();
        return Ok(None);
    }
    rmsnorm(residual, model.final_norm(), eps, normed)?;
    quantize_row_q8_k(normed, acts_q8k_hidden)?;
    clock.charge(Phase::Elementwise);
    let head = model.lm_head();
    let opened = gemv.open();
    pool_gemv_q8_k_stamped(
        pool,
        head.format,
        head.bytes,
        head.in_dim,
        head.out_dim,
        acts_q8k_hidden,
        logits,
        opened.is_some(),
    )?;
    gemv.close(GemvSite::LmHead, opened);
    clock.charge(Phase::Projections);
    clock.close();
    Ok(Some(logits))
}

/// Test support shared with the generation-loop tests: the io fixture's
/// synthetic quantized tensors stamp every block scale at d = 0.5, which
/// is fine for single-kernel tests but conditions a whole forward pass
/// terribly (value projections overflow the KV cache's f16 range and the
/// NaN residual then quantizes to all-zero activation blocks). Real
/// checkpoints carry per-block scales orders of magnitude smaller, so
/// these helpers re-stamp the install with small scales.
#[cfg(test)]
pub(crate) mod testsupport {
    use std::collections::BTreeMap;
    use std::path::Path;

    use crate::format::{CommonTensor, LayerLayout};
    use crate::io::parse_quant_format;
    use crate::io::testutil::Fixture;
    use crate::kernels::quants::{QuantFormat, f32_to_f16};

    /// Re-stamp every quantized block's f16 scale field(s) in `bytes` to
    /// 2^-12 (exact in f16). Offsets per `kernels/quants/blocks.rs`.
    fn restamp_scales(bytes: &mut [u8], format: QuantFormat) {
        let d = f32_to_f16(f32::powi(2.0, -12)).to_le_bytes();
        for block in bytes.chunks_exact_mut(format.block_bytes()) {
            match format {
                // d and dmin lead the block.
                QuantFormat::Q4_K | QuantFormat::Q5_K => {
                    block[0..2].copy_from_slice(&d);
                    block[2..4].copy_from_slice(&d);
                }
                // d trails at offset 208.
                QuantFormat::Q6_K => block[208..210].copy_from_slice(&d),
                // d leads the 32-weight block.
                QuantFormat::Q8_0 => block[0..2].copy_from_slice(&d),
                QuantFormat::Q8_K => unreachable!("activation-only format"),
            }
        }
    }

    /// Rewrite the install's quantized tensors (common + expert files)
    /// with tempered block scales so a full pass stays in f16 range.
    /// Load afterwards with `skip_hashes` (the manifest digests are stale).
    pub(crate) fn temper_install(fx: &Fixture) {
        temper_parts(&fx.root, &fx.manifest.common_tensors, &fx.layout.layers);
    }

    /// [`temper_install`] against an install described by its parts rather
    /// than by an `io::testutil::Fixture`.
    ///
    /// The shared builder in `io/testutil.rs` hard-codes one geometry and
    /// keeps its `TempDir` private, so a test that needs a *different*
    /// geometry (`prefill.rs`'s wide fixture: `q_dim != hidden`, more than
    /// one Q8_K block per row, 16 experts) cannot produce a `Fixture` to
    /// pass here. This takes exactly the three things the tempering reads.
    pub(crate) fn temper_parts(
        root: &Path,
        common_tensors: &BTreeMap<String, CommonTensor>,
        layers: &[LayerLayout],
    ) {
        let path = root.join("common.bin");
        let mut bytes = std::fs::read(&path).unwrap();
        for tensor in common_tensors.values() {
            if let Some(format) = parse_quant_format(&tensor.dtype) {
                let start = tensor.offset as usize;
                restamp_scales(&mut bytes[start..start + tensor.len as usize], format);
            }
        }
        std::fs::write(&path, bytes).unwrap();

        for layer in layers {
            let path = root.join(&layer.file);
            let mut bytes = std::fs::read(&path).unwrap();
            for expert in 0..layer.n_experts {
                let base = expert as usize * layer.stride as usize;
                for projection in &layer.projections {
                    let format = parse_quant_format(&projection.quant).unwrap();
                    let start = base + projection.offset_in_blob as usize;
                    restamp_scales(&mut bytes[start..start + projection.len as usize], format);
                }
            }
            std::fs::write(&path, bytes).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testsupport::temper_install;
    use super::*;
    use crate::io::LoadOptions;
    use crate::io::testutil::{Fixture, VOCAB, build_install};

    const SKIP: LoadOptions = LoadOptions {
        skip_hashes: true,
        verify_layer_hashes: false,
    };

    fn load_fixture(tag: &str) -> (Fixture, Model) {
        let fx = build_install(tag);
        temper_install(&fx);
        let model = Model::load(&fx.root, SKIP).unwrap();
        (fx, model)
    }

    /// A state over the tiny fixture: two unpinned shards, small budget.
    pub(crate) fn state(model: &Model, context_cap: usize) -> ForwardState {
        ForwardState::with_config(model, context_cap, RuntimeConfig::testing()).unwrap()
    }

    /// Run `ids` through a fresh state and return the final logits.
    fn run(model: &Model, state: &mut ForwardState, ids: &[u32]) -> Vec<f32> {
        let last = ids.len() - 1;
        for (pos, &id) in ids.iter().enumerate() {
            let want = pos == last;
            let out = forward_token(model, state, id, pos, want).unwrap();
            assert_eq!(out.is_some(), want);
            if let Some(logits) = out {
                return logits.to_vec();
            }
        }
        unreachable!("last token always requests logits");
    }

    /// A cheap deterministic spread; the split under test is a permutation,
    /// so what matters is that the values differ, not what they are.
    fn spread(i: usize) -> f32 {
        ((i * 37 % 101) as f32 - 50.0) / 64.0
    }

    /// A job in which several shards fail reports the **lowest-indexed**
    /// shard's error, whatever order the shards reached the mutex in.
    ///
    /// The property that makes a fan-out failure reproducible from its message.
    /// Driven directly rather than through a pool, because the thing under test
    /// is precisely what happens when arrival order and index order disagree —
    /// and a real pool cannot be made to reverse them on demand.
    #[test]
    fn a_multi_shard_failure_reports_the_lowest_shard_index() {
        let err = |layer| AttentionError::EmptyLayer { layer };
        for arrival in [[3usize, 1, 2, 0], [0, 1, 2, 3], [1, 0, 3, 2]] {
            let slot: Mutex<Option<(usize, AttentionError)>> = Mutex::new(None);
            for index in arrival {
                record(&slot, index, err(index));
            }
            assert_eq!(
                taken(slot).unwrap_err(),
                err(0),
                "arrival order {arrival:?} must not change the reported error"
            );
        }
    }

    /// The decode split is a tiling of the kv heads: every kv head is computed
    /// exactly once, whatever [`ComputePool::run`] hands the closure.
    ///
    /// This is the property [`pool_decode_attention`]'s correctness rests on,
    /// tested directly on [`kv_head_range`] so a failure names the split rather
    /// than the kernel. The two shapes that matter are covered explicitly:
    /// `units > n_kv_heads` (the surplus-shard case every non-hybrid CPU hits,
    /// where the tail slots must come back empty rather than duplicating work)
    /// and the single-slot-covering-everything case that
    /// [`ComputePool::run`]'s inline fallback produces.
    #[test]
    fn the_kv_head_split_tiles_the_kv_heads() {
        for (n_kv_heads, units) in [
            (4usize, 6usize),
            (4, 4),
            (4, 32),
            (4, 3),
            (4, 2),
            (3, 2),
            (8, 6),
            (1, 6),
            (0, 6),
        ] {
            // The normal path: `run` hands each shard one unit slot.
            let mut covered = vec![0u32; n_kv_heads];
            let mut empty = 0usize;
            for slot in 0..units {
                let range = kv_head_range(n_kv_heads, units, &(slot..slot + 1));
                assert!(range.start <= range.end, "kv {n_kv_heads} units {units}");
                assert!(range.end <= n_kv_heads, "kv {n_kv_heads} units {units}");
                if range.is_empty() {
                    empty += 1;
                }
                for head in range {
                    covered[head] += 1;
                }
            }
            assert!(
                covered.iter().all(|&n| n == 1),
                "kv {n_kv_heads} units {units}: {covered:?}"
            );
            assert_eq!(
                empty,
                units - n_kv_heads.min(units),
                "kv {n_kv_heads} units {units}: surplus slots must be empty no-ops"
            );

            // The fallback path: one shard receives `0..units` and must
            // therefore receive every kv head.
            assert_eq!(
                kv_head_range(n_kv_heads, units, &(0..units)),
                0..n_kv_heads,
                "kv {n_kv_heads} units {units}: single-shard fallback"
            );
        }
    }

    /// The sharded decode attention is bit-identical to the single whole-`q`
    /// call, on every geometry, every layer and every shard count.
    ///
    /// The decode half of the parallel-attention argument, isolated from the
    /// forward pass so a failure names the split rather than the model. A shard
    /// is a contiguous range of kv heads; it reads the whole `q`, writes its
    /// own `kv_heads.len() * group * head_dim` window of `out`, and its
    /// arithmetic touches nothing but its own slice of `q`, its own K and V
    /// rows and its own scratch — the same rows, the same softmax over the same
    /// contiguous run, the same reduction order. `to_bits`, no tolerance: the
    /// point is that no reduction was reassociated, and a tolerance would hide
    /// exactly that.
    ///
    /// The window carve is what this test is really pinning now that a shard is
    /// handed a sub-slice rather than the whole vector. The kernel writes its
    /// buffer from index zero, so it can no longer place a head wrongly on its
    /// own; the offset arithmetic here can, and a shard writing the right bits
    /// at the wrong offset is exactly the failure `f32::NAN` poisoning and
    /// element-wise `to_bits` catch.
    ///
    /// Coverage. **Geometries**: an uneven kv/shard split at group 4, an
    /// MQA-shaped single kv head (which takes the serial fall-through), plain
    /// MHA at group 1, and a group that divides none of the shard counts.
    /// **Layers**: ragged on purpose, so `positions` is a per-layer property —
    /// 1 and 7 are below [`ATTENTION_FANOUT_MIN_POSITIONS`] and must agree via
    /// the serial path, 8 is exactly the threshold, 17 is a full fan-out.
    /// **Shards**: past `n_kv_heads` on purpose. That is the surplus-shard case
    /// — the one the old query-head split silently declined to run at all — and
    /// it is where an off-by-one in the empty-range handling would either drop
    /// a kv head (stale `out`, caught as a surviving `NaN`) or compute one
    /// twice.
    #[test]
    fn the_kv_range_split_is_bit_identical_to_the_whole_call() {
        // (n_kv_heads, group, head_dim).
        const GEOMETRIES: [(usize, usize, usize); 4] =
            [(3, 4, 8), (1, 8, 8), (4, 1, 16), (2, 3, 4)];
        // Per-layer cached lengths; the cache is this deep.
        const LENS: [usize; 4] = [1, 7, 8, 17];

        /// One geometry's cache, query and per-layer reference outputs.
        struct Case {
            label: String,
            q: Vec<f32>,
            kv: KvCache,
            scale: f32,
            carve: usize,
            want: Vec<Vec<f32>>,
        }

        let cap = LENS.iter().copied().max().unwrap();
        let cases: Vec<Case> = GEOMETRIES
            .into_iter()
            .map(|(n_kv_heads, group, head_dim)| {
                let n_heads = n_kv_heads * group;
                let (q_dim, kv_dim) = (n_heads * head_dim, n_kv_heads * head_dim);
                let mut kv = KvCache::new(LENS.len(), n_kv_heads, head_dim, cap).unwrap();
                for (layer, &len) in LENS.iter().enumerate() {
                    for t in 0..len {
                        let k: Vec<f32> = (0..kv_dim).map(|i| spread(t * 7 + i + layer)).collect();
                        let v: Vec<f32> = (0..kv_dim)
                            .map(|i| spread(t * 11 + i + 3 + layer * 5))
                            .collect();
                        kv.append(layer, &k, &v).unwrap();
                    }
                }
                let q: Vec<f32> = (0..q_dim).map(|i| spread(i * 5 + 1)).collect();
                let scale = 1.0 / (head_dim as f32).sqrt();
                let carve = attention_scratch_len(n_heads, n_kv_heads, head_dim, cap);
                let want: Vec<Vec<f32>> = (0..LENS.len())
                    .map(|layer| {
                        let mut w = vec![0.0f32; q_dim];
                        decode_attention_in(&q, &kv, layer, scale, &mut vec![0.0; carve], &mut w)
                            .unwrap();
                        assert!(
                            w.iter().any(|v| *v != 0.0),
                            "the reference is degenerate at layer {layer}"
                        );
                        w
                    })
                    .collect();
                Case {
                    label: format!("({n_kv_heads}, {group}, {head_dim})"),
                    q,
                    kv,
                    scale,
                    carve,
                    want,
                }
            })
            .collect();

        for shards in [1usize, 2, 3, 4, 5, 8] {
            let mut pool = ComputePool::with_config(PoolConfig {
                shards: Some(shards),
                pin: false,
                pin_caller: false,
                inline_caller: true,
            });
            for case in &cases {
                // Exactly what `ForwardState` allocates: one carve per
                // *reachable* shard. Sizing this to `pool.shards()` instead
                // would leave slack past the last carve and hide an off-by-one
                // in the closure's `index * shard_scratch` window, which is the
                // half of this test that is not about bits.
                let reachable = case.kv.n_kv_heads().min(pool.shards()).max(1);
                let mut scratch = vec![0.0f32; case.carve * reachable];
                for layer in 0..LENS.len() {
                    // Poisoned, not zeroed: a kv head no shard claimed would
                    // otherwise read as a plausible zero rather than a failure.
                    let mut got = vec![f32::NAN; case.q.len()];
                    pool_decode_attention(
                        &mut pool,
                        &case.kv,
                        layer,
                        case.scale,
                        &case.q,
                        &mut scratch,
                        &mut got,
                    )
                    .unwrap();
                    for (i, (a, b)) in got.iter().zip(&case.want[layer]).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{} shards {shards} layer {layer} element {i}",
                            case.label
                        );
                    }
                }
            }
        }
    }

    /// [`ForwardState::attn_scratch`] holds one carve per **reachable** shard,
    /// `min(n_kv_heads, shards)`, not one per pool shard.
    ///
    /// [`pool_decode_attention`] fans out over kv heads, so a unit slot past
    /// `n_kv_heads` takes an empty range from [`kv_head_range`] and returns
    /// before it touches scratch. A carve for such a slot is allocated,
    /// zero-filled at construction and then unreachable for the life of the
    /// process — 2 of 6 at the v0 pin, 264,192 B of the 660,480 B the field's
    /// doc used to charge the fan-out with.
    ///
    /// The floor of one carve is pinned too: the serial fall-through hands the
    /// **whole** buffer to a single [`decode_attention_in`], so a zero-carve
    /// buffer would fail every decode with `ScratchTooShort`.
    #[test]
    fn attn_scratch_holds_one_carve_per_reachable_shard() {
        let (_fx, model) = load_fixture("fwd-attn-carves");
        let arch = model.arch();
        let n_kv = arch.n_kv_heads as usize;
        assert!(n_kv >= 2, "the fixture must have a kv head axis to split");
        let cap = 16usize;
        let carve = attention_scratch_len(arch.n_heads as usize, n_kv, arch.head_dim as usize, cap);
        for threads in [1usize, 2, n_kv, n_kv + 3, n_kv * 4] {
            let config = RuntimeConfig {
                cache_bytes: 4 * 1024 * 1024,
                threads: Some(threads),
                pin: false,
            };
            let st = ForwardState::with_config(&model, cap, config).unwrap();
            let shards = st.pool.shards();
            assert_eq!(
                st.attn_scratch.len(),
                carve * n_kv.min(shards),
                "threads {threads} (pool gave {shards} shards)"
            );
            assert!(
                st.attn_scratch.len() >= carve,
                "threads {threads}: the serial fall-through needs a whole carve"
            );
        }
    }

    #[test]
    fn logits_are_finite_and_vocab_shaped() {
        let (_fx, model) = load_fixture("fwd-finite");
        let mut st = state(&model, 8);
        assert_eq!(st.context_cap(), 8);
        let logits = run(&model, &mut st, &[1, 2, 3]);
        assert_eq!(logits.len(), VOCAB);
        assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");
        // The patterned fixture is not degenerate: logits differ.
        assert!(logits.iter().any(|&v| v != logits[0]));
        assert_eq!(st.seq_len().unwrap(), 3);
    }

    #[test]
    fn identical_inputs_are_bitwise_deterministic() {
        let (_fx, model) = load_fixture("fwd-determinism");
        let mut a = state(&model, 8);
        let mut b = state(&model, 8);
        let la = run(&model, &mut a, &[5, 0, 7]);
        let lb = run(&model, &mut b, &[5, 0, 7]);
        let bits_a: Vec<u32> = la.iter().map(|v| v.to_bits()).collect();
        let bits_b: Vec<u32> = lb.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits_a, bits_b);
    }

    /// The load-bearing phase-5 property, exercised end to end on the
    /// fixture: shard count and cache size are scheduling decisions, so a
    /// one-shard cold-cache run and a four-shard run whose cache is already
    /// warm must agree bit for bit.
    ///
    /// The prompt is 12 tokens rather than 3 so that it crosses
    /// [`ATTENTION_FANOUT_MIN_POSITIONS`]: the last four tokens run decode
    /// attention through the kv-head fan-out on the multi-shard arms and
    /// serially on the one-shard arm, which is exactly the comparison this test
    /// exists to make and which a 3-token prompt would skip entirely. At
    /// `threads = 4` against the fixture's 2 kv heads it also covers the
    /// surplus-shard case, where two of the four slots take an empty range.
    #[test]
    fn shard_count_and_cache_size_do_not_move_a_bit() {
        let (_fx, model) = load_fixture("fwd-shards");
        let prompt: [u32; 12] = [5, 0, 7, 3, 11, 2, 9, 1, 14, 6, 8, 4];

        let single = RuntimeConfig {
            cache_bytes: 4 * 1024 * 1024,
            threads: Some(1),
            pin: false,
        };
        let mut one = ForwardState::with_config(&model, 16, single).unwrap();
        let want = run(&model, &mut one, &prompt);

        // The tightest budget the cache accepts — one slot per routed
        // expert, i.e. `top_k` slots per layer — so nothing survives a step
        // and every request after the first is an eviction miss. That is the
        // maximum amount of cache churn there is, and it must still not move
        // a bit.
        let layout = model.layout();
        let widest = layout.layers.iter().map(|l| l.stride).max().unwrap();
        let tightest = widest * layout.layers.len() as u64 * u64::from(model.arch().top_k.max(1));

        for threads in [2usize, 4] {
            for cache_bytes in [tightest, 4 * 1024 * 1024] {
                let config = RuntimeConfig {
                    cache_bytes,
                    threads: Some(threads),
                    pin: false,
                };
                let mut many = ForwardState::with_config(&model, 16, config).unwrap();
                // Warm the cache, then re-run the same prompt.
                let _ = run(&model, &mut many, &prompt);
                let mut again = ForwardState::with_config(&model, 16, config).unwrap();
                let got = run(&model, &mut again, &prompt);
                for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "threads={threads} cache_bytes={cache_bytes}: logit {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn zeroed_lm_head_yields_all_zero_logits() {
        // Degenerate hand-verifiable config: zero every output.weight byte
        // (a zeroed q6_k block has d = 0, so every dequantized weight is 0)
        // => every logit is exactly 0.0 => softmax would be uniform.
        let fx = build_install("fwd-zero-head");
        temper_install(&fx);
        let entry = &fx.manifest.common_tensors["output.weight"];
        let path = fx.root.join("common.bin");
        let mut bytes = std::fs::read(&path).unwrap();
        let (start, len) = (entry.offset as usize, entry.len as usize);
        bytes[start..start + len].fill(0);
        std::fs::write(&path, bytes).unwrap();

        let model = Model::load(&fx.root, SKIP).unwrap();
        let mut st = state(&model, 4);
        let logits = run(&model, &mut st, &[3]);
        assert!(logits.iter().all(|&v| v == 0.0), "{logits:?}");
    }

    #[test]
    fn positions_must_be_sequential() {
        let (_fx, model) = load_fixture("fwd-positions");
        let mut st = state(&model, 4);
        assert!(matches!(
            forward_token(&model, &mut st, 0, 1, false).unwrap_err(),
            ForwardError::PositionMismatch {
                position: 1,
                expected: 0
            }
        ));
        forward_token(&model, &mut st, 0, 0, false).unwrap();
        assert!(matches!(
            forward_token(&model, &mut st, 0, 0, false).unwrap_err(),
            ForwardError::PositionMismatch {
                position: 0,
                expected: 1
            }
        ));
    }

    #[test]
    fn context_cap_exhaustion_is_typed() {
        let (_fx, model) = load_fixture("fwd-cap");
        let mut st = state(&model, 2);
        forward_token(&model, &mut st, 0, 0, false).unwrap();
        forward_token(&model, &mut st, 1, 1, false).unwrap();
        assert!(matches!(
            forward_token(&model, &mut st, 2, 2, false).unwrap_err(),
            ForwardError::Kv(KvError::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn bad_token_is_typed() {
        let (_fx, model) = load_fixture("fwd-token");
        let mut st = state(&model, 4);
        assert!(matches!(
            forward_token(&model, &mut st, VOCAB as u32, 0, true).unwrap_err(),
            ForwardError::Model(ModelError::TokenOutOfRange { .. })
        ));
    }

    #[test]
    fn runtime_dials_are_reported() {
        let (_fx, model) = load_fixture("fwd-dials");
        let st = state(&model, 4);
        assert_eq!(st.shards(), 2);
        assert!(st.slots_per_layer() >= 1);
        assert!(st.cache_bytes() > 0);
        let stats = st.stream_stats();
        assert_eq!(stats.hits + stats.misses, 0, "no experts read yet");
        // `Debug` must not depend on the streamer being `Debug`.
        assert!(format!("{st:?}").contains("ForwardState"));
    }

    /// Borrow the pieces of a state the expert phase needs, exactly the way
    /// the layer loop does, and hand them to `body` alongside a
    /// [`MoeScratch`] over `staged`/`done`.
    ///
    /// The staging buffers are supplied by the caller rather than taken from
    /// the state, because two of these tests deliberately run with a `top_k`
    /// the state was not built for.
    fn with_expert_scratch<R>(
        st: &mut ForwardState,
        staged: &mut [f32],
        done: &mut [bool],
        body: impl FnOnce(&mut ExpertStream, &mut ComputePool, &mut MoeScratch<'_>) -> R,
    ) -> R {
        let ForwardState {
            acts_q8k_hidden,
            acts_q8k_moe,
            gate_up,
            pool,
            stream,
            ..
        } = st;
        let mut scratch = MoeScratch {
            ffn: FfnScratch {
                acts_hidden: &acts_q8k_hidden[..],
                acts_moe: &mut acts_q8k_moe[..],
                gate_up: &mut gate_up[..],
            },
            staged,
            done,
        };
        body(stream, pool, &mut scratch)
    }

    /// The fixture's MoE geometry with `top_k` overridden.
    fn dims_with_top_k(model: &Model, top_k: usize) -> MoeDims {
        MoeDims {
            hidden: model.arch().hidden as usize,
            moe: model.arch().moe_intermediate as usize,
            top_k,
        }
    }

    /// Warm one expert into layer `layer`'s cache and report the slot it
    /// landed in. The slot stays `Ready` after the step closes.
    fn warm(stream: &mut ExpertStream, layer: u32, expert: u32) -> u32 {
        stream.begin_layer(layer, &[expert]).unwrap();
        let slot = stream.misses()[0].1;
        stream.await_misses().unwrap();
        stream.end_layer(layer);
        slot
    }

    /// F1. The hits phase runs *before* `await_misses`, so a failure in it
    /// returns with the step's misses still in the ring, writing into slots
    /// the cache is protecting. Without a drain, `end_layer` cannot release a
    /// `Filling` slot, `outstanding` never reaches zero, and every later
    /// `begin_layer` on any layer is refused for the life of the process.
    #[test]
    fn a_failed_hits_phase_drains_its_in_flight_reads() {
        let (_fx, model) = load_fixture("fwd-drain");
        assert_eq!(model.arch().top_k, 2, "the fixture routes two experts");
        let hidden = model.arch().hidden as usize;
        let mut st = state(&model, 4);

        // A `top_k` of 1 with a two-expert step: the hits phase rejects the
        // plan (request index 1 is outside 0..1) before it ever blocks.
        let dims = dims_with_top_k(&model, 1);
        let mut staged = vec![0.0f32; hidden];
        let mut done = vec![false; 1];

        with_expert_scratch(&mut st, &mut staged, &mut done, |stream, pool, scratch| {
            warm(stream, 0, 0);

            // One resident expert (a hit, to be computed first) and one cold
            // one (a miss, in flight while that happens).
            stream.begin_layer(0, &[1, 0]).unwrap();
            assert_eq!(stream.hits().len(), 1, "expert 0 is resident");
            assert_eq!(stream.misses().len(), 1, "expert 1 is cold");
            assert_eq!(stream.hits()[0].0, 1, "the hit is the second routed id");

            let mut timing = PrefillTiming::default();
            let err = stream_experts(
                stream,
                pool,
                0,
                dims,
                scratch,
                &mut PhaseClock::new(&mut timing, false),
                &mut GemvClock::disarmed(),
            )
            .unwrap_err();
            assert!(
                matches!(
                    err,
                    ForwardError::StreamPlanIndex {
                        layer: 0,
                        index: 1,
                        top_k: 1
                    }
                ),
                "{err}"
            );

            // The drain is the fix: the step can be closed, and the next one
            // is accepted. Before it, this `begin_layer` failed with "reads
            // from an earlier step were never awaited" — and so did every
            // other one, on every layer, forever.
            stream.end_layer(0);
            stream.begin_layer(1, &[0, 1]).unwrap();
            stream.await_misses().unwrap();
            stream.end_layer(1);
        });

        // End to end: a whole token still runs on this state.
        let logits = run(&model, &mut st, &[1, 2]);
        assert_eq!(logits.len(), VOCAB);
        assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");
    }

    /// F2. `StreamPlanIndex` is what stops a stale `expert_staged` slot from
    /// being reduced into the residual, so it is pinned against a hand-built
    /// plan rather than only against a healthy streamer: an index outside the
    /// routed set, and an index the plan already filled.
    #[test]
    fn a_plan_that_is_not_a_permutation_is_rejected() {
        let (_fx, model) = load_fixture("fwd-plan-index");
        let hidden = model.arch().hidden as usize;
        let mut st = state(&model, 4);
        let dims = dims_with_top_k(&model, 2);
        let mut staged = vec![0.0f32; 2 * hidden];
        let mut done = vec![false; 2];

        with_expert_scratch(&mut st, &mut staged, &mut done, |stream, pool, scratch| {
            let slot = warm(stream, 0, 0);

            // Past the end of the routed set: nothing is computed at all.
            scratch.done.fill(false);
            let err = run_plan(
                stream,
                pool,
                0,
                dims,
                &[(2, slot)],
                scratch,
                &mut GemvClock::disarmed(),
            )
            .unwrap_err();
            assert!(
                matches!(
                    err,
                    ForwardError::StreamPlanIndex {
                        layer: 0,
                        index: 2,
                        top_k: 2
                    }
                ),
                "{err}"
            );
            assert_eq!(scratch.done, &[false, false]);

            // Repeated: the first occurrence claims slot 0, the second is
            // refused because that slot is already claimed. Reducing it twice
            // would weight one expert twice and leave the other's slot stale.
            // Since phase 9 the whole plan is validated before the first
            // fan-out, so nothing is computed here at all — strictly earlier
            // than before, and the flag the guard leaves behind is the same.
            scratch.done.fill(false);
            let err = run_plan(
                stream,
                pool,
                0,
                dims,
                &[(0, slot), (0, slot)],
                scratch,
                &mut GemvClock::disarmed(),
            )
            .unwrap_err();
            assert!(
                matches!(
                    err,
                    ForwardError::StreamPlanIndex {
                        layer: 0,
                        index: 0,
                        top_k: 2
                    }
                ),
                "{err}"
            );
            assert_eq!(scratch.done, &[true, false], "only the first was taken");
        });
    }

    /// F2. `StreamPlanCoverage` is the other half: a plan that is a valid
    /// permutation but a *short* one leaves a staging slot holding the
    /// previous layer's expert output, which the fixed-order reduction would
    /// then weight as if it were this layer's.
    ///
    /// A duplicate expert id in one request is exactly that plan — the
    /// streamer resolves it once and lists it once — so this is the shape the
    /// guard would really see, not a synthetic one.
    #[test]
    fn a_plan_that_covers_fewer_than_top_k_experts_is_rejected() {
        let (_fx, model) = load_fixture("fwd-plan-coverage");
        let hidden = model.arch().hidden as usize;
        let mut st = state(&model, 4);
        let dims = dims_with_top_k(&model, 2);
        let mut staged = vec![0.0f32; 2 * hidden];
        let mut done = vec![false; 2];

        with_expert_scratch(&mut st, &mut staged, &mut done, |stream, pool, scratch| {
            stream.begin_layer(0, &[1, 1]).unwrap();
            assert_eq!(
                stream.hits().len() + stream.misses().len(),
                1,
                "a repeated id resolves once"
            );
            let mut timing = PrefillTiming::default();
            let err = stream_experts(
                stream,
                pool,
                0,
                dims,
                scratch,
                &mut PhaseClock::new(&mut timing, false),
                &mut GemvClock::disarmed(),
            )
            .unwrap_err();
            assert!(
                matches!(
                    err,
                    ForwardError::StreamPlanCoverage {
                        layer: 0,
                        covered: 1,
                        top_k: 2
                    }
                ),
                "{err}"
            );

            // Coverage fails after both phases, so nothing is in flight and
            // the stream is still usable.
            stream.end_layer(0);
            stream.begin_layer(0, &[0, 1]).unwrap();
            stream.await_misses().unwrap();
            stream.end_layer(0);
        });
    }

    /// A packed Q4_K matrix with planted f16 scales and a deterministic body.
    /// Structurally valid, finite, and different for every `seed`, which is
    /// all the fused fan-out tests need of it — they compare against the
    /// unfused call on the *same* bytes.
    fn synth_q4_k(in_dim: usize, out_dim: usize, seed: u8) -> Vec<u8> {
        let block_bytes = QuantFormat::Q4_K.block_bytes();
        let blocks = out_dim * (in_dim / QuantFormat::Q4_K.block_weights());
        let mut w = vec![0u8; blocks * block_bytes];
        let d = crate::kernels::quants::f32_to_f16(f32::powi(2.0, -8)).to_le_bytes();
        for (b, block) in w.chunks_exact_mut(block_bytes).enumerate() {
            // d and dmin lead a Q4_K super-block; the rest is scales and qs,
            // both of which are valid for any bit pattern.
            block[0..2].copy_from_slice(&d);
            block[2..4].copy_from_slice(&d);
            for (i, byte) in block[4..].iter_mut().enumerate() {
                *byte = (i as u8)
                    .wrapping_mul(31)
                    .wrapping_add(b as u8)
                    .wrapping_add(seed);
            }
        }
        w
    }

    /// The gapped fused fan-out — the one that reconstructs `&mut [f32]` from
    /// a raw pointer per shard — writes each part into exactly its own span of
    /// `dst`, bit-identically to the unfused whole-matrix call, and touches
    /// nothing between the spans.
    ///
    /// Driven directly rather than through the expert phase so that the
    /// disjointness the `unsafe` rests on is pinned by something other than
    /// "`run_plan` happens to build ascending spans today", and so that the
    /// spans can be made deliberately gapped, which a real routed plan only
    /// sometimes is.
    #[test]
    fn a_gapped_fused_fan_out_writes_only_its_own_spans() {
        let (in_dim, out_dim) = (256usize, 40usize);
        let weights: Vec<Vec<u8>> = (0..3).map(|s| synth_q4_k(in_dim, out_dim, s * 7)).collect();
        let x: Vec<f32> = (0..in_dim).map(spread).collect();
        let mut acts = vec![BlockQ8K::default(); in_dim / QuantFormat::Q8_K.block_weights()];
        quantize_row_q8_k(&x, &mut acts).unwrap();
        let parts: Vec<FusedPart<'_>> = weights
            .iter()
            .map(|w| FusedPart {
                format: QuantFormat::Q4_K,
                weight: w,
                in_dim,
                out_dim,
                acts: &acts,
            })
            .collect();

        // Deliberately gapped and not starting at zero: the spans a routed
        // plan of experts 1, 2 and 4 out of a top-8 would name.
        let at = [out_dim, 2 * out_dim, 4 * out_dim];
        const FILL: f32 = 1.25;

        for shards in [1usize, 2, 3, 5] {
            let mut pool = ComputePool::with_config(PoolConfig {
                shards: Some(shards),
                pin: false,
                pin_caller: false,
                inline_caller: true,
            });
            let mut dst = vec![FILL; 8 * out_dim];
            pool_gemv_q8_k_fused_at(&mut pool, &parts, &at, &mut dst, false).unwrap();

            for (p, &start) in at.iter().enumerate() {
                let mut want = vec![f32::NAN; out_dim];
                crate::kernels::gemv_q8_k(
                    QuantFormat::Q4_K,
                    &weights[p],
                    in_dim,
                    out_dim,
                    &acts,
                    &mut want,
                )
                .unwrap();
                assert!(
                    want.iter().all(|v| v.is_finite() && *v != FILL),
                    "shards={shards}: part {p} reference is degenerate: {want:?}"
                );
                for (r, (&g, &w)) in dst[start..start + out_dim].iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "shards={shards}: part {p} row {r} ({g} vs {w})"
                    );
                }
            }
            // Every element outside a span is still the fill, so no shard
            // wrote past the range it owns.
            for (i, &v) in dst.iter().enumerate() {
                let inside = at.iter().any(|&s| (s..s + out_dim).contains(&i));
                assert!(
                    inside || v == FILL,
                    "shards={shards}: element {i} outside every span was written ({v})"
                );
            }
        }

        // The disjointness the `unsafe` rests on is checked, not assumed.
        let mut pool = ComputePool::with_config(PoolConfig {
            shards: Some(2),
            pin: false,
            pin_caller: false,
            inline_caller: true,
        });
        let mut dst = vec![0.0f32; 8 * out_dim];
        let overlap = [0usize, out_dim - 1, 4 * out_dim];
        assert!(matches!(
            pool_gemv_q8_k_fused_at(&mut pool, &parts, &overlap, &mut dst, false).unwrap_err(),
            KernelError::LengthMismatch {
                what: FUSED_SPAN_ORDER,
                ..
            }
        ));
        let descending = [2 * out_dim, 0, 4 * out_dim];
        assert!(matches!(
            pool_gemv_q8_k_fused_at(&mut pool, &parts, &descending, &mut dst, false).unwrap_err(),
            KernelError::LengthMismatch {
                what: FUSED_SPAN_ORDER,
                ..
            }
        ));
        let past_end = [0usize, out_dim, 8 * out_dim];
        assert!(matches!(
            pool_gemv_q8_k_fused_at(&mut pool, &parts, &past_end, &mut dst, false).unwrap_err(),
            KernelError::LengthMismatch {
                what: FUSED_SPAN_BOUND,
                ..
            }
        ));
        assert!(matches!(
            pool_gemv_q8_k_fused_at(&mut pool, &parts, &at[..2], &mut dst, false).unwrap_err(),
            KernelError::LengthMismatch {
                what: FUSED_SPAN_COUNT,
                ..
            }
        ));
        // And a contiguous fused destination has to be the whole row space,
        // or `scatter` would shard over a prefix of it and report success.
        assert!(matches!(
            pool_gemv_q8_k_fused(&mut pool, &parts, &mut dst[..3 * out_dim - 1], false)
                .unwrap_err(),
            KernelError::LengthMismatch {
                what: FUSED_OUT_VS_SPACE,
                ..
            }
        ));
    }

    /// One expert's FFN the way phase 8 ran it: three separate whole-matrix
    /// fan-outs with the SwiGLU and the intermediate quantization between
    /// them. This is the sequence [`run_expert_window`] claims to reproduce
    /// bit for bit, so it is written out here rather than derived from it.
    fn expert_ffn_unfused(
        stream: &ExpertStream,
        pool: &mut ComputePool,
        layer: u32,
        dims: MoeDims,
        slot: u32,
        acts_hidden: &[BlockQ8K],
        out: &mut [f32],
    ) {
        let view = stream.view(layer, slot).unwrap();
        let (g, u, d) = (view.gate(), view.up(), view.down());
        let mut gate = vec![0.0f32; dims.moe];
        let mut up = vec![0.0f32; dims.moe];
        let hidden = dims.hidden;
        pool_gemv_q8_k(
            pool,
            g.format,
            g.bytes,
            hidden,
            dims.moe,
            acts_hidden,
            &mut gate,
        )
        .unwrap();
        pool_gemv_q8_k(
            pool,
            u.format,
            u.bytes,
            hidden,
            dims.moe,
            acts_hidden,
            &mut up,
        )
        .unwrap();
        swiglu_combine(&mut gate, &up).unwrap();
        let mut mid = vec![BlockQ8K::default(); dims.moe / QuantFormat::Q8_K.block_weights()];
        quantize_row_q8_k(&gate, &mut mid).unwrap();
        pool_gemv_q8_k(pool, d.format, d.bytes, dims.moe, hidden, &mid, out).unwrap();
    }

    /// Fill a state's `[hidden]` activation blocks with a real quantized row.
    ///
    /// Without this every block is `BlockQ8K::default()` — `d = 0`, `qs` all
    /// zero — so every dot in the layer is exactly `0.0`, every expert stages
    /// the same all-zero vector, and a comparison between two experts (or
    /// between fused and unfused) is vacuously true.
    fn plant_activations(st: &mut ForwardState, hidden: usize) {
        let x: Vec<f32> = (0..hidden).map(spread).collect();
        quantize_row_q8_k(&x, &mut st.acts_q8k_hidden).unwrap();
    }

    /// The claim phase 9 wave 1 rests on, at the layer above the kernel: a
    /// fused expert phase stages exactly what the per-expert sequence of
    /// whole-matrix fan-outs stages, bit for bit — and it stages each
    /// expert's own output in that expert's own routed slot.
    ///
    /// The routing half is not decoration. Fusing gives one job `2 * n`
    /// matrices sharing one activation row and then `n` matrices each reading
    /// a *different* intermediate into a *different* staging slot, so the two
    /// ways to get it wrong are pairing an expert with another's intermediate
    /// and pairing it with another's slot. Running the same two experts under
    /// swapped routed indices catches both: the staged halves must swap with
    /// them, which they cannot do if either pairing is positional.
    #[test]
    fn a_fused_expert_phase_matches_the_per_expert_sequence_bitwise() {
        let (_fx, model) = load_fixture("fwd-fused-experts");
        let hidden = model.arch().hidden as usize;
        let mut st = state(&model, 4);
        let dims = dims_with_top_k(&model, 2);
        plant_activations(&mut st, hidden);

        let mut staged = vec![f32::NAN; 2 * hidden];
        let mut done = vec![false; 2];
        let mut want = vec![f32::NAN; 2 * hidden];
        let mut got = vec![f32::NAN; 2 * hidden];
        let mut swapped_got = vec![f32::NAN; 2 * hidden];

        with_expert_scratch(&mut st, &mut staged, &mut done, |stream, pool, scratch| {
            // Two distinct experts, both resident: the shape `run_plan` sees
            // for a miss phase that has just been awaited.
            stream.begin_layer(0, &[0, 1]).unwrap();
            stream.await_misses().unwrap();
            let mut plan: Vec<(usize, u32)> = stream.hits().to_vec();
            plan.extend_from_slice(stream.misses());
            plan.sort_unstable();
            assert_eq!(plan.len(), 2, "the fixture routes two distinct experts");
            assert_ne!(plan[0].1, plan[1].1, "two experts, two slots");

            let acts: Vec<BlockQ8K> = scratch.ffn.acts_hidden.to_vec();
            for &(index, slot) in &plan {
                expert_ffn_unfused(
                    stream,
                    pool,
                    0,
                    dims,
                    slot,
                    &acts,
                    &mut want[index * hidden..(index + 1) * hidden],
                );
            }
            assert!(
                want.iter().all(|v| v.is_finite()),
                "the reference is not finite: {want:?}"
            );

            scratch.done.fill(false);
            run_plan(
                stream,
                pool,
                0,
                dims,
                &plan,
                scratch,
                &mut GemvClock::disarmed(),
            )
            .unwrap();
            assert_eq!(scratch.done, &[true, true]);
            got.copy_from_slice(scratch.staged);

            // Same experts, swapped routed indices.
            let swapped: Vec<(usize, u32)> = plan
                .iter()
                .map(|&(index, slot)| (1 - index, slot))
                .collect();
            scratch.done.fill(false);
            scratch.staged.fill(f32::NAN);
            run_plan(
                stream,
                pool,
                0,
                dims,
                &swapped,
                scratch,
                &mut GemvClock::disarmed(),
            )
            .unwrap();
            swapped_got.copy_from_slice(scratch.staged);
            stream.end_layer(0);
        });

        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "fused expert phase differs at slot {} element {} ({g} vs {w})",
                i / hidden,
                i % hidden,
            );
        }
        // The two experts really do produce different vectors, so the swap
        // below is a test and not a tautology.
        assert!(
            want[..hidden]
                .iter()
                .zip(&want[hidden..])
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "the fixture's two experts stage the same vector"
        );
        for (i, (&g, &w)) in swapped_got[..hidden]
            .iter()
            .zip(&want[hidden..])
            .enumerate()
        {
            assert_eq!(g.to_bits(), w.to_bits(), "swapped slot 0 element {i}");
        }
        for (i, (&g, &w)) in swapped_got[hidden..]
            .iter()
            .zip(&want[..hidden])
            .enumerate()
        {
            assert_eq!(g.to_bits(), w.to_bits(), "swapped slot 1 element {i}");
        }
    }

    /// A window position and a routed index are two different numbers since
    /// fusion, and the down job stages by the **routed index**.
    ///
    /// Before fusion there was no window position: the body ran one expert at a
    /// time and `index` was the only number in scope, so
    /// `scratch.staged[index * hidden..]` could not be confused with anything.
    /// The fused down job carries a parts array, so a slot's position in it
    /// (`j`, `0..n`) now exists alongside its routed index, and
    /// `at[j] = index * dims.hidden` is the single line that keeps them apart.
    ///
    /// A **complete** plan cannot tell the two apart, which is why the
    /// bit-identity test above does not.
    /// [`run_plan`]'s window scan walks `index` ascending and packs its matches
    /// densely, so when every routed index is present `j == index` identically
    /// and `at[j] = j * dims.hidden` names the same spans. Swapping the routed
    /// indices swaps which slot sits at each position and still leaves them
    /// equal, so `at[j] = j * dims.hidden` survives every test that aims at
    /// this invariant: measured, that mutation leaves the bit-identity test
    /// above, the fan-out count test below and the two plan-validation tests
    /// all green, and the only other test in the crate it fails is
    /// `prefill::tests::wide_paths_leave_the_same_kv_cache` — which compares
    /// two prefill paths end to end and so reports "the paths disagree", not
    /// "the staging span is named by the wrong number".
    ///
    /// Decode's two phases are what make them differ: they partition the routed
    /// set by whatever the cache happened to hold, so a phase whose hits are
    /// `{1, 3}` runs a window with `j` in `{0, 1}` against `index` in `{1, 3}`.
    /// Staging by `j` there would weight each expert's output with another
    /// expert's router probability at the fixed-order reduction and leave the
    /// routed slots it skipped holding the *previous* layer's bytes: wrong
    /// logits, no panic, and [`pool_gemv_q8_k_fused_at`]'s disjointness check
    /// satisfied either way, since `[0, hidden]` is as ascending and disjoint as
    /// `[hidden, 3 * hidden]` is.
    ///
    /// So the plans here are deliberately **partial and non-prefix**: one
    /// expert at routed index 1, then two at 1 and 3 of a top-4. The one-expert
    /// case is also the only bit assertion on a **single-part** fused down job,
    /// a geometry every phase with exactly one expert reaches —
    /// `a_gapped_fused_fan_out_writes_only_its_own_spans` always fans out three
    /// parts, and `an_expert_phase_is_two_fan_outs_however_many_experts_it_has`
    /// runs a one-expert plan but only counts the jobs.
    #[test]
    fn a_partial_plan_stages_by_routed_index_and_not_by_window_position() {
        let (_fx, model) = load_fixture("fwd-fused-partial");
        let hidden = model.arch().hidden as usize;
        let mut st = state(&model, 4);
        plant_activations(&mut st, hidden);

        // A `top_k` the state was not built for, which is the case
        // `with_expert_scratch` takes the staging buffers from the caller for:
        // the routed set has to be wide enough for a plan to skip indices, and
        // the *window* scratch is unaffected because a window is at most as
        // wide as the plan (one expert, then two).
        let top_k = 4usize;
        let dims = dims_with_top_k(&model, top_k);
        let mut staged = vec![f32::NAN; top_k * hidden];
        let mut done = vec![false; top_k];
        let mut want_a = vec![f32::NAN; hidden];
        let mut want_b = vec![f32::NAN; hidden];
        let mut one = vec![f32::NAN; top_k * hidden];
        let mut two = vec![f32::NAN; top_k * hidden];

        with_expert_scratch(&mut st, &mut staged, &mut done, |stream, pool, scratch| {
            stream.begin_layer(0, &[0, 1]).unwrap();
            stream.await_misses().unwrap();
            let mut resident: Vec<(usize, u32)> = stream.hits().to_vec();
            resident.extend_from_slice(stream.misses());
            resident.sort_unstable();
            assert_eq!(resident.len(), 2, "the fixture routes two distinct experts");
            let (slot_a, slot_b) = (resident[0].1, resident[1].1);
            assert_ne!(slot_a, slot_b, "two experts, two slots");

            let acts: Vec<BlockQ8K> = scratch.ffn.acts_hidden.to_vec();
            expert_ffn_unfused(stream, pool, 0, dims, slot_a, &acts, &mut want_a);
            expert_ffn_unfused(stream, pool, 0, dims, slot_b, &acts, &mut want_b);

            // One expert routed at index 1: window position 0 against routed
            // index 1, and a down job of exactly one part.
            scratch.done.fill(false);
            scratch.staged.fill(f32::NAN);
            run_plan(
                stream,
                pool,
                0,
                dims,
                &[(1, slot_a)],
                scratch,
                &mut GemvClock::disarmed(),
            )
            .unwrap();
            assert_eq!(scratch.done, &[false, true, false, false]);
            one.copy_from_slice(scratch.staged);

            // Two experts routed at 1 and 3: window positions 0 and 1, which is
            // the shape a phase takes whenever the cache holds some of a
            // layer's experts and not others.
            scratch.done.fill(false);
            scratch.staged.fill(f32::NAN);
            run_plan(
                stream,
                pool,
                0,
                dims,
                &[(1, slot_a), (3, slot_b)],
                scratch,
                &mut GemvClock::disarmed(),
            )
            .unwrap();
            assert_eq!(scratch.done, &[false, true, false, true]);
            two.copy_from_slice(scratch.staged);

            stream.end_layer(0);
        });

        // The fill is `NAN` and no expert output is one, so "nothing was
        // written here" and "the wrong expert was written here" cannot be
        // mistaken for each other.
        assert!(
            want_a.iter().chain(&want_b).all(|v| v.is_finite()),
            "a reference is not finite"
        );
        // And the two experts really do stage different vectors, or every
        // slot-pairing assertion below would hold for the wrong reason.
        assert!(
            want_a
                .iter()
                .zip(&want_b)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "the fixture's two experts stage the same vector"
        );

        let slot = |buf: &[f32], i: usize| -> Vec<u32> {
            buf[i * hidden..(i + 1) * hidden]
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let untouched =
            |buf: &[f32], i: usize| buf[i * hidden..(i + 1) * hidden].iter().all(|v| v.is_nan());

        // Staged by routed index, the lone expert lands in slot 1. Staged by
        // window position it would land in slot 0.
        assert_eq!(
            slot(&one, 1),
            slot(&want_a, 0),
            "a single-part down job staged the wrong slot, or staged it wrong"
        );
        for i in [0usize, 2, 3] {
            assert!(
                untouched(&one, i),
                "a plan covering routed index 1 alone wrote staging slot {i}"
            );
        }

        // Two experts, two gaps: positions 0 and 1 against indices 1 and 3.
        assert_eq!(slot(&two, 1), slot(&want_a, 0), "routed index 1");
        assert_eq!(slot(&two, 3), slot(&want_b, 0), "routed index 3");
        for i in [0usize, 2] {
            assert!(
                untouched(&two, i),
                "a plan covering routed indices 1 and 3 wrote staging slot {i}"
            );
        }
    }

    /// The count the coarse-versus-fine test can no longer pin exactly: an
    /// expert phase is **two** fan-outs — one gate/up job and one down job —
    /// however many experts its plan holds, and an empty plan is none.
    ///
    /// This is the whole of phase 9 wave 1 in one assertion. Before it, a
    /// two-expert phase was six fan-outs and a one-expert phase three.
    #[test]
    fn an_expert_phase_is_two_fan_outs_however_many_experts_it_has() {
        let (_fx, model) = load_fixture("fwd-fused-count");
        let hidden = model.arch().hidden as usize;
        let mut st = state(&model, 4);
        let dims = dims_with_top_k(&model, 2);
        plant_activations(&mut st, hidden);

        let mut staged = vec![0.0f32; 2 * hidden];
        let mut done = vec![false; 2];
        let mut split = GemvSplit::default();

        with_expert_scratch(&mut st, &mut staged, &mut done, |stream, pool, scratch| {
            stream.begin_layer(0, &[0, 1]).unwrap();
            stream.await_misses().unwrap();
            let mut plan: Vec<(usize, u32)> = stream.hits().to_vec();
            plan.extend_from_slice(stream.misses());
            plan.sort_unstable();
            assert_eq!(plan.len(), 2);

            let mut gemv = GemvClock::new(&mut split, true);

            // An empty plan fans nothing out at all.
            scratch.done.fill(false);
            run_plan(stream, pool, 0, dims, &[], scratch, &mut gemv).unwrap();

            // One expert: still exactly two.
            scratch.done.fill(false);
            run_plan(stream, pool, 0, dims, &plan[..1], scratch, &mut gemv).unwrap();

            // Both experts: two more, not four more.
            scratch.done.fill(false);
            run_plan(stream, pool, 0, dims, &plan, scratch, &mut gemv).unwrap();

            stream.end_layer(0);
        });

        assert_eq!(
            split.experts.scatters, 4,
            "an empty plan, a one-expert plan and a two-expert plan must charge \
             0 + 2 + 2 fan-outs: {split:?}"
        );
        assert_eq!(
            (
                split.projections.scatters,
                split.lm_head.scatters,
                split.router.scatters
            ),
            (0, 0, 0),
            "the expert phase charged another site: {split:?}"
        );
    }

    /// F8. `forward_token` takes the model and the state independently, and
    /// reads `top_k`/`hidden` off the *argument* model while the staging
    /// buffers were sized from the construction-time one. A mismatched pair
    /// used to index out of bounds inside `run_plan`; it is now typed.
    #[test]
    fn a_state_refuses_a_model_it_was_not_built_for() {
        let (_fx, wide) = load_fixture("fwd-arch-wide");
        let (_fx2, narrow) = load_fixture("fwd-arch-narrow");
        let mut st = state(&narrow, 4);
        // Same geometry: the fingerprint is about dimensions, not identity,
        // so two separate installs of the same architecture must pair.
        forward_token(&wide, &mut st, 1, 0, false).unwrap();

        // Now a state built for a narrower router, handed a model whose
        // top_k is one wider than its staging buffer holds.
        let fingerprint = ArchFingerprint {
            top_k: narrow.arch().top_k - 1,
            ..st.arch
        };
        st.arch = fingerprint;
        let err = forward_token(&narrow, &mut st, 1, 1, false).unwrap_err();
        assert!(
            matches!(
                err,
                ForwardError::ArchMismatch {
                    what: "top_k",
                    built: 1,
                    given: 2
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn default_runtime_config_is_the_documented_budget() {
        let config = RuntimeConfig::default();
        assert_eq!(config.cache_bytes, DEFAULT_CACHE_BYTES);
        assert_eq!(config.cache_bytes, 1_509_949_440);
        assert_eq!(config.threads, None);
        assert!(config.pin);
    }

    /// The default budget must clear 11 slots/layer on the audited Qwen3
    /// geometry, and it clears it by only 1.4 MiB. One slot across all 48
    /// layers costs `24 * 3,059,712 + 24 * 2,654,208` = 137,134,080 B, so 11
    /// slots is 1,508,474,880 B (1,438.59 MiB). A budget expressed as a round
    /// 1,438 MiB floors to **10**, which is the bug this pins: the dial
    /// divides and floors, so the default has to clear the boundary, not sit
    /// on it.
    #[test]
    fn the_default_budget_clears_eleven_slots_per_layer() {
        const PER_SLOT_ALL_LAYERS: u64 = 24 * 3_059_712 + 24 * 2_654_208;
        assert_eq!(PER_SLOT_ALL_LAYERS, 137_134_080);

        assert_eq!(DEFAULT_CACHE_BYTES / PER_SLOT_ALL_LAYERS, 11);
        assert_eq!(1438 * 1024 * 1024 / PER_SLOT_ALL_LAYERS, 10);
    }

    // -----------------------------------------------------------------------
    // The decode GEMV sub-split
    // -----------------------------------------------------------------------

    /// `(own + wait)` over every bucket, plus the serial router.
    fn gemv_accounted(split: &[(&'static str, Duration, Duration, u64); 4]) -> Duration {
        split
            .iter()
            .map(|&(_, own, wait, _)| own + wait)
            .sum::<Duration>()
    }

    /// The scatter count of one labelled bucket.
    fn gemv_scatters(split: &[(&'static str, Duration, Duration, u64); 4], want: &str) -> u64 {
        split
            .iter()
            .find(|&&(label, ..)| label == want)
            .unwrap_or_else(|| panic!("no {want} bucket in {split:?}"))
            .3
    }

    /// The sub-split and the coarse split are two instruments over the same
    /// decode, and they must not be able to disagree.
    ///
    /// Four directions, all checked over a real multi-token decode:
    ///
    /// - **Counts.** Every fan-out is charged exactly once, to the bucket the
    ///   geometry says: `4` projections a layer (`attn_q`, `attn_k`, `attn_v`,
    ///   `attn_output`), a serial router matvec a layer, one `lm_head` a
    ///   token. A GEMV that slipped out of the sub-split, or one charged
    ///   twice, moves one of these exact numbers.
    ///
    ///   The expert count is the one that is not a constant any more: phase 9
    ///   fuses a phase's experts into a gate/up job and a down job, so a layer
    ///   charges `2` when every routed expert is a cache hit (or every one a
    ///   miss) and `4` when the plan splits across both phases — which is a
    ///   property of the cache on the day, not of the geometry. So the bucket
    ///   is pinned by the invariant that *is* structural (two fan-outs per
    ///   non-empty phase, one or two non-empty phases a layer, hence even and
    ///   inside `[2, 4]` per layer) and the exact per-phase count is pinned
    ///   separately, and exactly, by
    ///   `an_expert_phase_is_two_fan_outs_however_many_experts_it_has`.
    /// - **Containment.** Every sub-split region sits strictly inside
    ///   `Phase::Projections` or `Phase::ExpertCompute`, so the sub-split can
    ///   never exceed those two coarse phases together. A region charged to the
    ///   wrong coarse phase — attention, say — breaks this immediately.
    /// - **Coverage.** And it has to be most of them, or a bucket has silently
    ///   stopped charging. The gap is the non-GEMV work those phases also cover
    ///   (the softmax and top-k scan, SwiGLU, the intermediate quantization, the
    ///   expert view carves); it measures ~6% here, and the assertion is set at
    ///   50% so that it pins the instrument rather than the fixture's SwiGLU.
    /// - **Liveness.** At least one pooled barrier measured non-zero, which is
    ///   the only thing that distinguishes "the barrier is free" from "the
    ///   shard-0 stamp never arrived and `own` swallowed the whole call".
    ///
    /// No absolute duration is asserted anywhere, for the reason the prefill
    /// timing tests give: this machine runs the model under a cgroup while the
    /// tests run.
    #[test]
    fn the_gemv_sub_split_cannot_disagree_with_the_coarse_decode_split() {
        let (_fx, model) = load_fixture("fwd-gemv-subsplit");
        let mut st = state(&model, 16);
        let layers = model.n_layers() as u64;
        let top_k = model.arch().top_k as u64;
        let tokens = 4u64;

        for pos in 0..tokens as usize {
            forward_token(&model, &mut st, (pos % VOCAB) as u32, pos, true).unwrap();
        }

        let split = st.decode_gemv_split();
        let coarse = st.decode_timing();
        assert_eq!(
            coarse.tokens, tokens,
            "the two blocks cover the same tokens"
        );

        assert_eq!(
            gemv_scatters(&split, "projections"),
            4 * layers * tokens,
            "attn_q, attn_k, attn_v, attn_output, each once a layer: {split:?}"
        );
        let experts = gemv_scatters(&split, "experts");
        assert_eq!(
            experts % 2,
            0,
            "an expert phase charges a gate/up job and a down job or neither, \
             so the count is even: {split:?}"
        );
        assert!(
            experts >= 2 * layers * tokens && experts <= 4 * layers * tokens,
            "every layer runs one or two non-empty expert phases at two fused \
             fan-outs each, so {experts} must sit in \
             [{}, {}]: {split:?}",
            2 * layers * tokens,
            4 * layers * tokens,
        );
        // The fusion is the point, and the bound above is evidence of it only
        // while the fixture routes enough experts for the pre-fusion form to
        // have charged more than that bound allows. `3 * top_k` a layer against
        // a fused at-most-`4` a layer means a `top_k` of 1 makes the two
        // coincide and the assertion above prove nothing about fusion at all.
        // So the fixture's own power is what is asserted here; the regression
        // it would otherwise catch — a return to a fan-out per projection per
        // expert — already breaks the upper bound above, which is strictly
        // tighter for every `top_k` this can pass at.
        assert!(
            3 * top_k > 4,
            "the fixture routes {top_k} expert(s), so the pre-fusion per-expert \
             form would have charged no more than the fused bound above: that \
             bound is no longer evidence that anything is fused"
        );
        assert_eq!(
            gemv_scatters(&split, "lm_head"),
            tokens,
            "one head fan-out a token: {split:?}"
        );
        assert_eq!(
            gemv_scatters(&split, "router"),
            layers * tokens,
            "one serial matvec a layer: {split:?}"
        );

        let accounted = gemv_accounted(&split);
        let inside = coarse.projections + coarse.expert_compute;
        assert!(
            accounted <= inside,
            "the sub-split {accounted:?} exceeds the coarse phases {inside:?} it \
             lives inside — a fan-out is charged to the wrong phase, or twice: \
             {split:?}"
        );
        assert!(
            accounted > Duration::ZERO,
            "a decode that ran charged nothing: {split:?}"
        );
        // And the other side of the sum: the gap between the two instruments is
        // only the non-GEMV work those phases also cover, so the sub-split has
        // to be *most* of them. Measured on this fixture it is ~94%; the floor
        // is set at half, which is eight times the observed gap and still
        // catches a whole bucket that stopped charging (the smallest, `lm_head`,
        // is worth more than that share at the v0 pin). A tighter bound would
        // be measuring the fixture's SwiGLU and top-k scan, not the instrument.
        assert!(
            accounted * 2 >= inside,
            "the sub-split {accounted:?} is under half the coarse phases \
             {inside:?} it tiles: a site has stopped charging: {split:?}"
        );
        for &(label, own, _, _) in &split {
            assert!(
                own > Duration::ZERO,
                "{label} charged no own time: {split:?}"
            );
        }

        // The instrument's own failure mode: if the shard-0 stamp never reaches
        // the submitting thread, `close` falls back to `t1 = t2` and every
        // pooled bucket reports its whole fan-out as `own` with a `wait` of
        // exactly zero — which reads like a finding ("the barrier costs
        // nothing") rather than like a dead instrument. Over the several dozen
        // real fan-outs above, a barrier that measures zero every single time
        // is that failure and nothing else.
        let pooled_wait: Duration = split[..3].iter().map(|&(_, _, wait, _)| wait).sum();
        assert!(
            pooled_wait > Duration::ZERO,
            "every pooled fan-out reported a zero barrier: the shard-0 stamp is \
             not reaching the submitting thread, so `own` is the whole call and \
             the split says nothing: {split:?}"
        );

        // The router never fans out, so it has no barrier to wait at.
        let (_, _, router_wait, _) = split[3];
        assert_eq!(split[3].0, "router");
        assert_eq!(
            router_wait,
            Duration::ZERO,
            "the serial router cannot wait at a barrier it never reaches"
        );

        // Same lifetime as the coarse split: a reset drops both.
        st.reset();
        assert_eq!(
            st.decode_gemv_split(),
            GemvSplit::default().rows(),
            "a reset state reports no decode GEMVs at all"
        );
    }

    /// A disarmed [`GemvClock`] reads no clock and touches no state.
    ///
    /// "No clock reads" is not directly observable, so this asserts the
    /// structural cause of it: [`GemvClock::open`] returns before it does
    /// anything at all, which is why it can neither clear [`SHARD0_DONE`] nor
    /// call `Instant::now`. A sentinel left in the cell across the whole
    /// open/close/close_serial cycle is what pins that.
    #[test]
    fn a_disarmed_gemv_clock_does_nothing() {
        let sentinel = Instant::now();
        SHARD0_DONE.set(Some(sentinel));

        let mut clock = GemvClock::disarmed();
        assert!(
            clock.open().is_none(),
            "a disarmed clock has no instant to hand back, so it read none"
        );
        assert_eq!(
            SHARD0_DONE.get(),
            Some(sentinel),
            "`open` cleared the stamp, so it did more than return"
        );

        // And charging is a no-op even when handed an instant by a caller that
        // armed and then disarmed.
        clock.close(GemvSite::Experts, Some(sentinel));
        clock.close_serial(GemvSite::Router, Some(sentinel));
        assert_eq!(
            SHARD0_DONE.get(),
            Some(sentinel),
            "a disarmed close consumed the stamp"
        );

        // An armed clock over its own split does charge, and clears the stamp
        // on the way in.
        let mut split = GemvSplit::default();
        let mut clock = GemvClock::new(&mut split, true);
        let opened = clock.open();
        assert!(opened.is_some());
        assert_eq!(SHARD0_DONE.get(), None, "`open` arms a fresh stamp");
        clock.close_serial(GemvSite::Router, opened);
        assert_eq!(split.router.scatters, 1);
        assert_eq!(split.router.wait, Duration::ZERO);

        SHARD0_DONE.set(None);
    }
}
