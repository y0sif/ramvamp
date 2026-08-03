//! Single-token forward pass over a loaded model.
//!
//! [`forward_token`] runs one token through every layer against a
//! [`ForwardState`] that owns the KV cache and every scratch buffer; the
//! layer loop itself allocates nothing after [`ForwardState::new`]
//! (remaining small per-token allocations: [`Model::embed`]'s dequant `Vec`
//! and the stream decoder's `String`). This is the
//! single-threaded, buffered-`pread` baseline of the decode loop in
//! `docs/architecture.md` ("Decode loop"): no LFU cache, no io_uring
//! overlap yet — misses read synchronously through
//! [`Model::expert_reader`].
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

use crate::io::IoError;
use crate::kernels::KernelError;
use crate::kernels::attention::{AttentionError, AttentionScratch, decode_attention};
use crate::kernels::primitives::{
    rmsnorm, rmsnorm_in_place, rope_neox_heads, softmax, swiglu_combine, vec_add,
};
use crate::kernels::quants::{BlockQ8_0, BlockQ8K, quantize_row_q8_0, quantize_row_q8_k};
use crate::kernels::{gemv_q8_0, gemv_q8_k};
use crate::kv::{KvCache, KvError};
use thiserror::Error;

use super::error::ModelError;
use super::weights::Model;

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
}

/// Owned per-sequence state for [`forward_token`]: the KV cache plus every
/// scratch buffer the pass writes, preallocated once so no per-token
/// allocation happens after construction.
#[derive(Debug)]
pub struct ForwardState {
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
    /// Expert gate projection output (`[moe_intermediate]`), then the
    /// SwiGLU-combined value in place.
    gate: Vec<f32>,
    /// Expert up projection output (`[moe_intermediate]`).
    up: Vec<f32>,
    /// Expert down projection output (`[hidden]`).
    expert_down: Vec<f32>,
    /// Weighted expert accumulator (`[hidden]`).
    expert_acc: Vec<f32>,
    /// Expert blob read buffer, capacity = the largest layer stride.
    expert_buf: Vec<u8>,
    /// Output logits (`[vocab]`), valid after a `want_logits` pass.
    logits: Vec<f32>,
}

impl ForwardState {
    /// Preallocate all state for a model and a context cap (positions the
    /// KV cache can hold; v0 runs with 4096).
    ///
    /// # Errors
    ///
    /// [`ForwardError::UnsupportedDim`] when an architecture dimension is
    /// not a whole number of activation blocks;
    /// [`ForwardError::InvalidTopK`] on a nonsensical router config;
    /// [`ForwardError::Kv`] when the KV geometry is rejected.
    pub fn new(model: &Model, context_cap: usize) -> Result<Self, ForwardError> {
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
        let reader = model.expert_reader();
        let max_stride = (0..reader.n_layers())
            .filter_map(|l| reader.stride(l))
            .max()
            .unwrap_or(0);
        let max_stride = usize::try_from(max_stride).map_err(|_| ForwardError::UnsupportedDim {
            what: "expert stride",
            dim: usize::MAX,
            block: 1,
        })?;

        Ok(Self {
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
            gate: vec![0.0; moe],
            up: vec![0.0; moe],
            expert_down: vec![0.0; hidden],
            expert_acc: vec![0.0; hidden],
            expert_buf: Vec::with_capacity(max_stride),
            logits: vec![0.0; vocab],
        })
    }

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
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        acc += x * y;
    }
    acc
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
    let n_heads = arch.n_heads as usize;
    let n_kv_heads = arch.n_kv_heads as usize;
    let head_dim = arch.head_dim as usize;
    let hidden = arch.hidden as usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let moe = arch.moe_intermediate as usize;
    let top_k = arch.top_k as usize;
    let eps = arch.rms_eps as f32;
    let theta = arch.rope_theta as f32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let expected = state.kv.seq_len()?;
    if position != expected {
        return Err(ForwardError::PositionMismatch { position, expected });
    }
    let rope_pos =
        u32::try_from(position).map_err(|_| ForwardError::PositionOverflow { position })?;

    model.embed(token_id, &mut state.hidden)?;

    for layer in 0..model.n_layers() {
        let lw = model.layer(layer)?;
        let layer_idx = layer as usize;

        // Attention block: residual = hidden; x = attn_norm(hidden).
        rmsnorm(&state.hidden, lw.attn_norm, eps, &mut state.normed)?;

        // Quantize the normed input both ways: Q8_K for the k-quant
        // projections, Q8_0 for the q8_0 attn_k projection.
        quantize_row_q8_k(&state.normed, &mut state.acts_q8k_hidden)?;
        quantize_row_q8_0(&state.normed, &mut state.acts_q8_0_hidden)?;

        gemv_q8_k(
            lw.attn_q.format,
            lw.attn_q.bytes,
            hidden,
            q_dim,
            &state.acts_q8k_hidden,
            &mut state.q,
        )?;
        gemv_q8_0(
            lw.attn_k.bytes,
            hidden,
            kv_dim,
            &state.acts_q8_0_hidden,
            &mut state.k,
        )?;
        gemv_q8_k(
            lw.attn_v.format,
            lw.attn_v.bytes,
            hidden,
            kv_dim,
            &state.acts_q8k_hidden,
            &mut state.v,
        )?;

        // Per-head QK-RMSNorm, then RoPE — HF order: q_norm/k_norm apply
        // after the projection reshape and before rotary embedding
        // (`Qwen3MoeAttention.forward`). The [head_dim] weight is shared
        // across heads; same eps as the layer norms.
        for head in state.q.chunks_exact_mut(head_dim) {
            rmsnorm_in_place(head, lw.attn_q_norm, eps)?;
        }
        for head in state.k.chunks_exact_mut(head_dim) {
            rmsnorm_in_place(head, lw.attn_k_norm, eps)?;
        }
        rope_neox_heads(&mut state.q, n_heads, head_dim, rope_pos, theta)?;
        rope_neox_heads(&mut state.k, n_kv_heads, head_dim, rope_pos, theta)?;

        state.kv.append(layer_idx, &state.k, &state.v)?;
        decode_attention(
            &state.q,
            &state.kv,
            layer_idx,
            scale,
            &mut state.attn_scratch,
            &mut state.attn_out,
        )?;

        // Output projection (q5_k, Q8_K activations) and residual add.
        quantize_row_q8_k(&state.attn_out, &mut state.acts_q8k_attn)?;
        gemv_q8_k(
            lw.attn_output.format,
            lw.attn_output.bytes,
            q_dim,
            hidden,
            &state.acts_q8k_attn,
            &mut state.o_proj,
        )?;
        vec_add(&mut state.hidden, &state.o_proj)?;

        // MoE block: residual = hidden; x = ffn_norm(hidden).
        rmsnorm(&state.hidden, lw.ffn_norm, eps, &mut state.normed)?;

        // Router: f32 matvec (rows validated `[n_experts, hidden]` at
        // load), softmax over all experts in f32, top-k by probability
        // (equivalent to top-k by logit; first index wins ties like
        // torch.topk), then renormalize when norm_topk_prob.
        for (row, logit) in lw
            .router
            .data()
            .chunks_exact(hidden)
            .zip(state.router_logits.iter_mut())
        {
            *logit = dot_f32(row, &state.normed);
        }
        state.router_probs.copy_from_slice(&state.router_logits);
        softmax(&mut state.router_probs)?;
        state.topk.clear();
        for _ in 0..top_k {
            let mut best_e = 0usize;
            let mut best_p = f32::NEG_INFINITY;
            for (e, &p) in state.router_probs.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best_e = e;
                }
            }
            state.topk.push((best_e as u32, best_p));
            state.router_probs[best_e] = f32::NEG_INFINITY;
        }
        if arch.norm_topk_prob {
            let sum: f32 = state.topk.iter().map(|&(_, w)| w).sum();
            for (_, w) in state.topk.iter_mut() {
                *w /= sum;
            }
        }
        if let Some(sink) = on_route.as_deref_mut() {
            sink(layer, &state.topk);
        }

        // Experts: quantize the normed input once, then gate/up -> SwiGLU
        // -> down per selected expert, accumulating the weighted outputs.
        quantize_row_q8_k(&state.normed, &mut state.acts_q8k_hidden)?;
        state.expert_acc.fill(0.0);
        for i in 0..top_k {
            let (expert, weight) = state.topk[i];
            let view = model
                .expert_reader()
                .read_expert(layer, expert, &mut state.expert_buf)?;
            let gate_slab = view.gate();
            let up_slab = view.up();
            let down_slab = view.down();
            gemv_q8_k(
                gate_slab.format,
                gate_slab.bytes,
                hidden,
                moe,
                &state.acts_q8k_hidden,
                &mut state.gate,
            )?;
            gemv_q8_k(
                up_slab.format,
                up_slab.bytes,
                hidden,
                moe,
                &state.acts_q8k_hidden,
                &mut state.up,
            )?;
            swiglu_combine(&mut state.gate, &state.up)?;
            quantize_row_q8_k(&state.gate, &mut state.acts_q8k_moe)?;
            gemv_q8_k(
                down_slab.format,
                down_slab.bytes,
                moe,
                hidden,
                &state.acts_q8k_moe,
                &mut state.expert_down,
            )?;
            for (acc, &d) in state.expert_acc.iter_mut().zip(&state.expert_down) {
                *acc += weight * d;
            }
        }
        vec_add(&mut state.hidden, &state.expert_acc)?;
    }

    if !want_logits {
        return Ok(None);
    }
    rmsnorm(&state.hidden, model.final_norm(), eps, &mut state.normed)?;
    quantize_row_q8_k(&state.normed, &mut state.acts_q8k_hidden)?;
    let head = model.lm_head();
    gemv_q8_k(
        head.format,
        head.bytes,
        head.in_dim,
        head.out_dim,
        &state.acts_q8k_hidden,
        &mut state.logits,
    )?;
    Ok(Some(&state.logits))
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
        let path = fx.root.join("common.bin");
        let mut bytes = std::fs::read(&path).unwrap();
        for tensor in fx.manifest.common_tensors.values() {
            if let Some(format) = parse_quant_format(&tensor.dtype) {
                let start = tensor.offset as usize;
                restamp_scales(&mut bytes[start..start + tensor.len as usize], format);
            }
        }
        std::fs::write(&path, bytes).unwrap();

        for layer in &fx.layout.layers {
            let path = fx.root.join(&layer.file);
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

    const SKIP: LoadOptions = LoadOptions { skip_hashes: true };

    fn load_fixture(tag: &str) -> (Fixture, Model) {
        let fx = build_install(tag);
        temper_install(&fx);
        let model = Model::load(&fx.root, SKIP).unwrap();
        (fx, model)
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
        let mut state = ForwardState::new(&model, 8).unwrap();
        assert_eq!(state.context_cap(), 8);
        let logits = run(&model, &mut state, &[1, 2, 3]);
        assert_eq!(logits.len(), VOCAB);
        assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");
        // The patterned fixture is not degenerate: logits differ.
        assert!(logits.iter().any(|&v| v != logits[0]));
        assert_eq!(state.seq_len().unwrap(), 3);
    }

    #[test]
    fn identical_inputs_are_bitwise_deterministic() {
        let (_fx, model) = load_fixture("fwd-determinism");
        let mut a = ForwardState::new(&model, 8).unwrap();
        let mut b = ForwardState::new(&model, 8).unwrap();
        let la = run(&model, &mut a, &[5, 0, 7]);
        let lb = run(&model, &mut b, &[5, 0, 7]);
        let bits_a: Vec<u32> = la.iter().map(|v| v.to_bits()).collect();
        let bits_b: Vec<u32> = lb.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits_a, bits_b);
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
        let mut state = ForwardState::new(&model, 4).unwrap();
        let logits = run(&model, &mut state, &[3]);
        assert!(logits.iter().all(|&v| v == 0.0), "{logits:?}");
    }

    #[test]
    fn positions_must_be_sequential() {
        let (_fx, model) = load_fixture("fwd-positions");
        let mut state = ForwardState::new(&model, 4).unwrap();
        assert!(matches!(
            forward_token(&model, &mut state, 0, 1, false).unwrap_err(),
            ForwardError::PositionMismatch {
                position: 1,
                expected: 0
            }
        ));
        forward_token(&model, &mut state, 0, 0, false).unwrap();
        assert!(matches!(
            forward_token(&model, &mut state, 0, 0, false).unwrap_err(),
            ForwardError::PositionMismatch {
                position: 0,
                expected: 1
            }
        ));
    }

    #[test]
    fn context_cap_exhaustion_is_typed() {
        let (_fx, model) = load_fixture("fwd-cap");
        let mut state = ForwardState::new(&model, 2).unwrap();
        forward_token(&model, &mut state, 0, 0, false).unwrap();
        forward_token(&model, &mut state, 1, 1, false).unwrap();
        assert!(matches!(
            forward_token(&model, &mut state, 2, 2, false).unwrap_err(),
            ForwardError::Kv(KvError::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn bad_token_is_typed() {
        let (_fx, model) = load_fixture("fwd-token");
        let mut state = ForwardState::new(&model, 4).unwrap();
        assert!(matches!(
            forward_token(&model, &mut state, VOCAB as u32, 0, true).unwrap_err(),
            ForwardError::Model(ModelError::TokenOutOfRange { .. })
        ));
    }
}
