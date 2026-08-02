//! Scalar `f32` primitive kernels: RMSNorm, NeoX RoPE, softmax, SiLU/SwiGLU,
//! and small vector helpers for the decode loop.
//!
//! These are the non-quantized building blocks of the forward pass (see
//! `docs/architecture.md`, "Decode loop"). They are deliberately written as
//! simple auto-vectorizable loops; explicit SIMD lives with the quantized
//! GEMV kernels, which dominate runtime. Alignment: no assumption beyond the
//! natural 4-byte alignment of `&[f32]` — safe for packed sub-tensor offsets
//! that only guarantee 2-byte alignment once the data has been copied into
//! `f32` buffers.
//!
//! Precision policy (documented so validation gates stay meaningful):
//!
//! - Reductions (RMSNorm mean-of-squares, softmax normalizer) accumulate in
//!   `f64`, matching ggml's `ggml_float = double` accumulators; softmax also
//!   takes its exponentials in `f64` (see [`softmax`]).
//! - RoPE angles and their sin/cos are computed in `f64`, then applied to the
//!   activations in `f32`. Other elementwise math is plain `f32`.
//!
//! Validation policy: every public call validates lengths once up front and
//! returns a typed [`PrimitiveError`] — never panics on bad input. Inner
//! loops assume validated lengths.

mod activation;
mod error;
mod norm;
mod rope;
mod softmax;
#[cfg(test)]
mod testutil;
mod vec;

pub use activation::{silu, swiglu_combine};
pub use error::PrimitiveError;
pub use norm::{rmsnorm, rmsnorm_gemma, rmsnorm_in_place};
pub use rope::{rope_neox, rope_neox_heads};
pub use softmax::softmax;
pub use vec::{vec_add, vec_scale};
