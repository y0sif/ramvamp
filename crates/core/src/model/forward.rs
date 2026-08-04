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

use std::fmt;
use std::sync::{Mutex, PoisonError};

use crate::format::ArchInfo;
use crate::io::{
    ExpertStream, ExpertView, IoError, StreamMode, StreamPhase, StreamStats, SweepError, SweepPlan,
};
use crate::kernels::KernelError;
use crate::kernels::attention::{AttentionError, AttentionScratch, decode_attention};
use crate::kernels::primitives::{
    rmsnorm, rmsnorm_in_place, rope_neox_heads, softmax, swiglu_combine, vec_add,
};
use crate::kernels::quants::{
    BlockQ8_0, BlockQ8K, QuantFormat, quantize_row_q8_0, quantize_row_q8_k,
};
use crate::kernels::{gemv_q8_0_rows, gemv_q8_k_rows};
use crate::kv::{KvCache, KvError};
use crate::threads::{ComputePool, PoolConfig};
use thiserror::Error;

use super::error::ModelError;
use super::prefill::PrefillConfig;
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
    attn_scratch: AttentionScratch,
    /// Residual stream (`[hidden]`).
    hidden: Vec<f32>,
    /// RMSNorm output (`[hidden]`), reused by every norm site.
    normed: Vec<f32>,
    /// Q8_K quantization of a `[hidden]` vector (`hidden / 256` blocks).
    acts_q8k_hidden: Vec<BlockQ8K>,
    /// Q8_K quantization of a `[moe_intermediate]` vector.
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
    /// Expert gate projection output (`[moe_intermediate]`), then the
    /// SwiGLU-combined value in place. Shared: experts are computed one at
    /// a time, each one fanned out across the pool.
    gate: Vec<f32>,
    /// Expert up projection output (`[moe_intermediate]`).
    up: Vec<f32>,
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

        Ok(Self {
            arch: ArchFingerprint::of(arch),
            kv,
            attn_scratch: AttentionScratch::with_capacity(context_cap),
            hidden: vec![0.0; hidden],
            normed: vec![0.0; hidden],
            acts_q8k_hidden: vec![BlockQ8K::default(); hidden / QK_K],
            acts_q8k_moe: vec![BlockQ8K::default(); moe / QK_K],
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
            gate: vec![0.0; moe],
            up: vec![0.0; moe],
            expert_staged: vec![0.0; top_k * hidden],
            expert_done: vec![false; top_k],
            expert_acc: vec![0.0; hidden],
            logits: vec![0.0; vocab],
            pool,
            stream,
            prefill: PrefillConfig::from_env(),
            sweep_plan: SweepPlan::new(),
            routed: Vec::with_capacity(n_experts),
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
    pub fn reset(&mut self) {
        self.kv.clear();
    }

    /// Borrow the pieces the chunked prefill driver needs, all at once.
    ///
    /// Field-by-field so the compute pool, the streamer, the KV cache and the
    /// scratch buffers can be held simultaneously — the same destructuring
    /// [`forward_token`] does, exposed to the sibling module.
    pub(super) fn prefill_parts(&mut self) -> PrefillParts<'_> {
        PrefillParts {
            kv: &mut self.kv,
            attn_scratch: &mut self.attn_scratch,
            topk: &mut self.topk,
            logits: &mut self.logits,
            pool: &mut self.pool,
            stream: &mut self.stream,
            sweep_plan: &mut self.sweep_plan,
            routed: &mut self.routed,
        }
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
    pub(super) attn_scratch: &'a mut AttentionScratch,
    /// `(expert, weight)` staging for the route sink.
    pub(super) topk: &'a mut Vec<(u32, f32)>,
    pub(super) logits: &'a mut Vec<f32>,
    pub(super) pool: &'a mut ComputePool,
    pub(super) stream: &'a mut ExpertStream,
    pub(super) sweep_plan: &'a mut SweepPlan,
    pub(super) routed: &'a mut Vec<u32>,
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

/// Remember the first kernel error any shard reported, without panicking
/// out of a worker thread.
pub(super) fn record(slot: &Mutex<Option<KernelError>>, err: KernelError) {
    let mut held = slot.lock().unwrap_or_else(PoisonError::into_inner);
    if held.is_none() {
        *held = Some(err);
    }
}

/// Take whatever [`record`] stored.
pub(super) fn taken(slot: Mutex<Option<KernelError>>) -> Result<(), KernelError> {
    match slot.into_inner().unwrap_or_else(PoisonError::into_inner) {
        Some(err) => Err(err),
        None => Ok(()),
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
    let failure: Mutex<Option<KernelError>> = Mutex::new(None);
    pool.scatter(out, |shard, chunk| {
        if let Err(err) = gemv_q8_k_rows(format, weight, in_dim, out_dim, acts, shard.rows, chunk) {
            record(&failure, err);
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
) -> Result<(), KernelError> {
    let failure: Mutex<Option<KernelError>> = Mutex::new(None);
    pool.scatter(out, |shard, chunk| {
        if let Err(err) = gemv_q8_0_rows(weight, in_dim, out_dim, acts, shard.rows, chunk) {
            record(&failure, err);
        }
    });
    taken(failure)
}

/// The three MoE dimensions the expert phase needs, bundled so the phase
/// helpers take one value instead of three positional `usize`s.
#[derive(Clone, Copy, Debug)]
struct MoeDims {
    hidden: usize,
    moe: usize,
    top_k: usize,
}

/// The scratch one expert's FFN writes, reused by every expert in a layer:
/// experts are computed one at a time, each fanned out across the pool.
struct FfnScratch<'a> {
    /// Q8_K quantization of the layer's normed residual (`[hidden]`).
    acts_hidden: &'a [BlockQ8K],
    /// Q8_K quantization of the SwiGLU output (`[moe_intermediate]`).
    acts_moe: &'a mut [BlockQ8K],
    /// Gate projection, then the SwiGLU combination in place.
    gate: &'a mut [f32],
    /// Up projection.
    up: &'a mut [f32],
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

/// gate/up -> SwiGLU -> down for one expert, staging `[hidden]` into `out`.
///
/// Each of the three GEMVs is row-parallel across the pool; the two cheap
/// `[moe_intermediate]` element-wise steps between them stay on the decode
/// thread.
fn expert_ffn(
    pool: &mut ComputePool,
    view: &ExpertView<'_>,
    dims: MoeDims,
    scratch: &mut FfnScratch<'_>,
    out: &mut [f32],
) -> Result<(), ForwardError> {
    let gate_slab = view.gate();
    let up_slab = view.up();
    let down_slab = view.down();
    pool_gemv_q8_k(
        pool,
        gate_slab.format,
        gate_slab.bytes,
        dims.hidden,
        dims.moe,
        scratch.acts_hidden,
        scratch.gate,
    )?;
    pool_gemv_q8_k(
        pool,
        up_slab.format,
        up_slab.bytes,
        dims.hidden,
        dims.moe,
        scratch.acts_hidden,
        scratch.up,
    )?;
    swiglu_combine(scratch.gate, scratch.up)?;
    quantize_row_q8_k(scratch.gate, scratch.acts_moe)?;
    pool_gemv_q8_k(
        pool,
        down_slab.format,
        down_slab.bytes,
        dims.moe,
        dims.hidden,
        scratch.acts_moe,
        out,
    )?;
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
/// `stream` is taken shared: every [`ExpertView`] borrows it for the length
/// of the phase, and nothing here needs to mutate it.
fn run_plan(
    stream: &ExpertStream,
    pool: &mut ComputePool,
    layer: u32,
    dims: MoeDims,
    plan: &[(usize, u32)],
    scratch: &mut MoeScratch<'_>,
) -> Result<(), ForwardError> {
    for &(index, slot) in plan {
        if index >= dims.top_k || scratch.done[index] {
            return Err(ForwardError::StreamPlanIndex {
                layer,
                index,
                top_k: dims.top_k,
            });
        }
        let view = stream.view(layer, slot)?;
        let out = &mut scratch.staged[index * dims.hidden..(index + 1) * dims.hidden];
        expert_ffn(pool, &view, dims, &mut scratch.ffn, out)?;
        scratch.done[index] = true;
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
) -> Result<(), ForwardError> {
    let outcome = stage_expert_phases(stream, pool, layer, dims, scratch);
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
fn stage_expert_phases(
    stream: &mut ExpertStream,
    pool: &mut ComputePool,
    layer: u32,
    dims: MoeDims,
    scratch: &mut MoeScratch<'_>,
) -> Result<(), ForwardError> {
    scratch.done.fill(false);
    run_plan(stream, pool, layer, dims, stream.hits(), scratch)?;
    stream.await_misses()?;
    run_plan(stream, pool, layer, dims, stream.misses(), scratch)?;
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
        gate,
        up,
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
    } = state;

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

        pool_gemv_q8_k(
            pool,
            lw.attn_q.format,
            lw.attn_q.bytes,
            hidden,
            q_dim,
            acts_q8k_hidden,
            q,
        )?;
        pool_gemv_q8_0(pool, lw.attn_k.bytes, hidden, kv_dim, acts_q8_0_hidden, k)?;
        pool_gemv_q8_k(
            pool,
            lw.attn_v.format,
            lw.attn_v.bytes,
            hidden,
            kv_dim,
            acts_q8k_hidden,
            v,
        )?;

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
        decode_attention(q, kv, layer_idx, scale, attn_scratch, attn_out)?;

        // Output projection (q5_k, Q8_K activations) and residual add.
        quantize_row_q8_k(attn_out, acts_q8k_attn)?;
        pool_gemv_q8_k(
            pool,
            lw.attn_output.format,
            lw.attn_output.bytes,
            q_dim,
            hidden,
            acts_q8k_attn,
            o_proj,
        )?;
        vec_add(residual, o_proj)?;

        // MoE block: residual = hidden; x = ffn_norm(hidden).
        rmsnorm(residual, lw.ffn_norm, eps, normed)?;

        // Router: f32 matvec (rows validated `[n_experts, hidden]` at
        // load), softmax over all experts in f32, top-k by probability
        // (equivalent to top-k by logit; first index wins ties like
        // torch.topk), then renormalize when norm_topk_prob.
        for (row, logit) in lw
            .router
            .data()
            .chunks_exact(hidden)
            .zip(router_logits.iter_mut())
        {
            *logit = dot_f32(row, normed);
        }
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
        if let Some(sink) = on_route.as_deref_mut() {
            sink(layer, topk);
        }

        // Experts: quantize the normed input once, submit every miss, then
        // compute hits and misses as two coarse units, staging each
        // expert's `[hidden]` output.
        quantize_row_q8_k(normed, acts_q8k_hidden)?;
        expert_ids.clear();
        expert_ids.extend(topk.iter().map(|&(expert, _)| expert));

        stream.begin_layer(layer, expert_ids)?;
        let mut scratch = MoeScratch {
            ffn: FfnScratch {
                acts_hidden: acts_q8k_hidden,
                acts_moe: acts_q8k_moe,
                gate,
                up,
            },
            staged: expert_staged,
            done: expert_done,
        };
        let phases = stream_experts(stream, pool, layer, dims, &mut scratch);
        stream.end_layer(layer);
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
    }

    if !want_logits {
        return Ok(None);
    }
    rmsnorm(residual, model.final_norm(), eps, normed)?;
    quantize_row_q8_k(normed, acts_q8k_hidden)?;
    let head = model.lm_head();
    pool_gemv_q8_k(
        pool,
        head.format,
        head.bytes,
        head.in_dim,
        head.out_dim,
        acts_q8k_hidden,
        logits,
    )?;
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
    #[test]
    fn shard_count_and_cache_size_do_not_move_a_bit() {
        let (_fx, model) = load_fixture("fwd-shards");

        let single = RuntimeConfig {
            cache_bytes: 4 * 1024 * 1024,
            threads: Some(1),
            pin: false,
        };
        let mut one = ForwardState::with_config(&model, 8, single).unwrap();
        let want = run(&model, &mut one, &[5, 0, 7]);

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
                let mut many = ForwardState::with_config(&model, 8, config).unwrap();
                // Warm the cache, then re-run the same prompt.
                let _ = run(&model, &mut many, &[5, 0, 7]);
                let mut again = ForwardState::with_config(&model, 8, config).unwrap();
                let got = run(&model, &mut again, &[5, 0, 7]);
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
            gate,
            up,
            pool,
            stream,
            ..
        } = st;
        let mut scratch = MoeScratch {
            ffn: FfnScratch {
                acts_hidden: &acts_q8k_hidden[..],
                acts_moe: &mut acts_q8k_moe[..],
                gate: &mut gate[..],
                up: &mut up[..],
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

            let err = stream_experts(stream, pool, 0, dims, scratch).unwrap_err();
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
            let err = run_plan(stream, pool, 0, dims, &[(2, slot)], scratch).unwrap_err();
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

            // Repeated: the first occurrence stages, the second is refused
            // because slot 0 is already filled. Reducing it twice would
            // weight one expert twice and leave the other's slot stale.
            scratch.done.fill(false);
            let err =
                run_plan(stream, pool, 0, dims, &[(0, slot), (0, slot)], scratch).unwrap_err();
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
            assert_eq!(scratch.done, &[true, false], "only the first ran");
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
            let err = stream_experts(stream, pool, 0, dims, scratch).unwrap_err();
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
}
