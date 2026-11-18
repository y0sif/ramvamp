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
//!
//! # Numerics: identical math means identical bits
//!
//! A restructuring here (a new partition across threads, a fused or batched
//! job, a SIMD path, a new loop order) reproduces the bits of the code it
//! replaces, and is tested bit for bit, not within a tolerance. Everything
//! downstream rests on that: the prefill sweep matches token-major decode,
//! and any thread count matches any other. The rules that keep it:
//!
//! - Vector lanes and thread shards run only along axes that are already
//!   independent: output rows ([`gemv`]); prefill's query rows, kv heads,
//!   the GQA group and the elementwise head dimension ([`attention`]). A reduction chain is never
//!   split, so a dot is never split within a row.
//! - No FMA where the scalar reference does a separate multiply and add (two
//!   roundings); `_mm256_fmadd_ps` and `f32::mul_add` would make it one.
//! - No online, streaming or flash-style rescaled softmax. The causal limit is
//!   a length: masked positions are absent from the sum, not zero-weighted.
//!
//! The AVX2 dots in [`quants::avx2`] are the one scoped exception: their
//! float accumulation order differs from the scalar reference by design and
//! is tested within a tolerance. Every path built on them (row ranges, fused
//! rows, batched tiles) is still bit-identical to the single call it
//! replaces. Each file's own docs state its piece of the argument.

pub mod attention;
pub mod backend;
mod error;
pub mod gemv;

pub mod quants;

pub mod primitives;

pub use backend::{Backend, CpuBackend};
pub use error::KernelError;
pub use gemv::{
    gemv_q8_0, gemv_q8_0_batched, gemv_q8_0_rows, gemv_q8_k, gemv_q8_k_batched, gemv_q8_k_rows,
};
