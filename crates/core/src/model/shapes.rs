//! Shape and dtype expectations derived from the manifest's `ArchInfo`.
//!
//! Every packed matrix is stored GGUF-style: rows span the input
//! (contiguous) axis, so a projection that maps `in_dim -> out_dim` is
//! `out_dim` packed rows of `in_dim` weights — exactly the layout the gemv
//! kernels consume. The embedding is the same shape read the other way:
//! GGUF `token_embd.weight` has ne-order dims `[hidden, vocab]`, i.e.
//! `vocab` rows of `hidden` weights, each row one token's embedding vector
//! (confirmed against the planner's ne-order check and the audited
//! manifest: `len = vocab * row_bytes(hidden)`).
//!
//! Required dtypes are pinned to the audited per-tensor map of the v0
//! source (Q4_K_M GGUF, audited 2026-08-01, see `docs/architecture.md`):
//!
//! | tensor | dtype | rows x in_dim |
//! | --- | --- | --- |
//! | `token_embd.weight` | q4_k | vocab x hidden |
//! | `output.weight` (lm_head) | q6_k | vocab x hidden |
//! | `output_norm.weight` | f32 | hidden |
//! | `blk.N.attn_q.weight` | q4_k | (n_heads*head_dim) x hidden |
//! | `blk.N.attn_k.weight` | q8_0 | (n_kv_heads*head_dim) x hidden |
//! | `blk.N.attn_v.weight` | q4_k or q6_k | (n_kv_heads*head_dim) x hidden |
//! | `blk.N.attn_output.weight` | q5_k | hidden x (n_heads*head_dim) |
//! | `blk.N.attn_norm.weight`, `blk.N.ffn_norm.weight` | f32 | hidden |
//! | `blk.N.attn_q_norm.weight`, `blk.N.attn_k_norm.weight` | f32 | head_dim |
//! | `blk.N.ffn_gate_inp.weight` (router) | f32 | n_experts x hidden |
//! | `blk.N.ffn_gate_exps` / `ffn_up_exps` | q4_k | moe_intermediate x hidden |
//! | `blk.N.ffn_down_exps` | q4_k or q6_k | hidden x moe_intermediate |
//!
//! `attn_v` is Q6_K on exactly the Q6_K-down layers of the audited file;
//! the loader accepts either format per layer independently rather than
//! enforcing that coupling.

use crate::format::ArchInfo;
use crate::io::parse_quant_format;
use crate::kernels::quants::QuantFormat;

/// The dtypes a tensor slot may declare, with a display name for errors.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AllowedQuant {
    /// Human-readable allowed set.
    pub(crate) desc: &'static str,
    /// Accepted formats.
    pub(crate) formats: &'static [QuantFormat],
}

impl AllowedQuant {
    /// Match a manifest dtype string against the allowed set.
    pub(crate) fn match_dtype(&self, dtype: &str) -> Option<QuantFormat> {
        let format = parse_quant_format(dtype)?;
        self.formats.contains(&format).then_some(format)
    }
}

/// `q4_k` only.
pub(crate) const Q4K: AllowedQuant = AllowedQuant {
    desc: "q4_k",
    formats: &[QuantFormat::Q4_K],
};

/// `q5_k` only.
pub(crate) const Q5K: AllowedQuant = AllowedQuant {
    desc: "q5_k",
    formats: &[QuantFormat::Q5_K],
};

/// `q6_k` only.
pub(crate) const Q6K: AllowedQuant = AllowedQuant {
    desc: "q6_k",
    formats: &[QuantFormat::Q6_K],
};

/// `q8_0` only.
pub(crate) const Q8_0: AllowedQuant = AllowedQuant {
    desc: "q8_0",
    formats: &[QuantFormat::Q8_0],
};

/// `q4_k` or `q6_k` (the per-layer split of `attn_v` / `ffn_down_exps`).
pub(crate) const Q4K_OR_Q6K: AllowedQuant = AllowedQuant {
    desc: "q4_k or q6_k",
    formats: &[QuantFormat::Q4_K, QuantFormat::Q6_K],
};

/// Dimensions every expectation derives from, precomputed once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Dims {
    /// Hidden (embedding) dimension.
    pub(crate) hidden: usize,
    /// Attention inner width: `n_heads * head_dim` (4096 for v0 — q_proj
    /// up-projects, o_proj projects back down).
    pub(crate) q_dim: usize,
    /// KV width: `n_kv_heads * head_dim`.
    pub(crate) kv_dim: usize,
    /// Per-head norm width (QK-RMSNorm weights are `[head_dim]`).
    pub(crate) head_dim: usize,
    /// Vocabulary size (embedding/lm_head rows).
    pub(crate) vocab: usize,
    /// Routed experts per layer (router rows).
    pub(crate) n_experts: usize,
    /// Per-expert FFN intermediate dimension.
    pub(crate) moe_intermediate: usize,
}

impl Dims {
    /// Derive the dimension table from validated architecture facts.
    pub(crate) fn from_arch(arch: &ArchInfo) -> Self {
        Self {
            hidden: arch.hidden as usize,
            q_dim: arch.n_heads as usize * arch.head_dim as usize,
            kv_dim: arch.n_kv_heads as usize * arch.head_dim as usize,
            head_dim: arch.head_dim as usize,
            vocab: arch.vocab as usize,
            n_experts: arch.n_experts as usize,
            moe_intermediate: arch.moe_intermediate as usize,
        }
    }
}
