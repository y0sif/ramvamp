//! Architecture configuration and the per-layer forward pass.
//!
//! The runtime is generic over a fine-grained MoE shape: N layers, E routed
//! experts per layer, top-k routing, optional shared expert, optional
//! sliding-window attention on a subset of layers. v0 pins one concrete
//! model, Qwen3-30B-A3B (no shared expert, full attention); Gemma 4 26B-A4B
//! (shared expert, 25/30 sliding-window layers) is the second target and the
//! proof of generality. Exact dimensions are loaded from the manifest, which
//! the repacker derives from the source `config.json`.
//!
//! Decode step per layer: attention + router on resident weights; read back
//! the top-k expert IDs; plan hits/misses against the layer cache; start
//! misses' reads while cache-hit expert work (and the shared expert, if any)
//! runs; combine branches; layer tail. Every GEMV on that path — the four
//! attention projections, all three expert projections, and `lm_head` —
//! fans out over contiguous output-row ranges across a pinned compute pool,
//! which is bit-identical to the whole-matrix call by construction.
//!
//! # Loading layer
//!
//! What exists today is the loading half: [`Model::load`] opens an
//! installed `.rvmp` directory, verifies it (see
//! [`LoadOptions`]), and validates every resident tensor and expert slab
//! against a shape table derived from the manifest's `ArchInfo` plus the
//! audited per-tensor dtype map (the table lives in `shapes`, documented
//! there). Access is typed:
//!
//! - [`Model::layer`] -> [`LayerWeights`]: zero-copy [`QuantTensor`] views
//!   for the four attention projections, decoded `&[f32]` norms, and the
//!   router as an [`F32Tensor`] with row access.
//! - [`Model::embedding_row`] / [`Model::embed`]: one token's packed q4_k
//!   row, or its dequantized `hidden`-vector.
//! - [`Model::lm_head`] / [`Model::final_norm`]: the layer-48 tail.
//! - [`Model::expert_reader`]: the validated expert-blob geometry — the
//!   per-layer gate/up/down slab table that every `ExpertView` is carved
//!   with, plus one synchronous positioned read of a single blob. This is
//!   **not** the streaming side: `io::ExpertStream` owns the slot pool, the
//!   per-layer cache and the io_uring/pread submission path, and borrows
//!   this reader only to carve a slot it has already filled. The reader's
//!   own `read_expert` is the portable one-blob-at-a-time baseline the
//!   streamer was built beside; nothing on the decode path calls it, and it
//!   is exercised only by tests.
//!
//! # Forward pass
//!
//! [`forward_token`] runs one token through every layer against a
//! [`ForwardState`] (KV cache + preallocated scratch); see `forward` for
//! the reference-fidelity notes (HF `Qwen3MoeDecoderLayer` order,
//! QK-RMSNorm before RoPE, router softmax-then-top-k).
//!
//! # Prefill
//!
//! [`prefill_prompt`] consumes a whole prompt. By default it runs the
//! chunked layer-major sweep (`prefill`): up to
//! [`DEFAULT_PREFILL_CHUNK`] positions are carried through the model
//! together and each layer's expert file is streamed once per chunk, which
//! takes expert traffic from ~1,097 MB per token to ~34 MB.
//! [`PrefillMode::TokenMajor`] keeps the old `forward_token`-in-a-loop path
//! selectable in the same binary, which is what makes the byte-identical
//! -logits A/B a unit test rather than a cross-build comparison.

mod error;
mod forward;
mod prefill;
mod shapes;
mod weights;

pub use crate::io::{LoadOptions, StreamPhase};
pub use error::ModelError;
#[cfg(test)]
pub(crate) use forward::testsupport;
pub use forward::{
    DEFAULT_CACHE_BYTES, ExpertRouteSink, ForwardError, ForwardState, RuntimeConfig, forward_token,
    forward_token_traced,
};
pub use prefill::{
    DEFAULT_PREFILL_CHUNK, PrefillConfig, PrefillMode, PrefillProgressSink, PrefillRouteSink,
    PrefillTiming, prefill_prompt, prefill_prompt_with_progress,
};
pub use weights::{F32Tensor, LayerWeights, Model, QuantTensor};
