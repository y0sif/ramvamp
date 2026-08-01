//! Compute kernels behind a backend trait.
//!
//! The first backend is CPU: runtime-detected AVX2 paths for quantized GEMV,
//! grouped MoE FFN (fused dequant + activation + weighted reduction), RMSNorm,
//! RoPE, attention, and sampling. A Vulkan backend is planned behind the same
//! trait; keeping the boundary here from day one is deliberate.
//!
//! Layout rule learned the hard way upstream: packed sub-tensor offsets may
//! guarantee less alignment than a wide load assumes. Every vectorized path
//! must state, and test, the alignment it relies on.
