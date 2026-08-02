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
//! runs; combine branches; layer tail.
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
//! - [`Model::expert_reader`]: the streaming side's file access.

mod error;
mod shapes;
mod weights;

pub use crate::io::LoadOptions;
pub use error::ModelError;
pub use weights::{F32Tensor, LayerWeights, Model, QuantTensor};
