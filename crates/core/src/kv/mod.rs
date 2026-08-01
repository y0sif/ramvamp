//! KV cache.
//!
//! FP16 K/V storage sized by the model's attention pattern: linear append-only
//! storage for full-attention layers; bounded ring buffers for sliding-window
//! layers (needed for Gemma 4's 25 SWA layers, where rings keep the cache
//! flat as context grows). Under a ~3 GB budget the KV cache is the
//! third-largest resident tenant after the expert slot pool and the common
//! weights, so its bounds are part of the memory contract, not an
//! implementation detail.
