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
