//! The backend trait boundary: the seam a future non-CPU backend (Vulkan is
//! the planned second, see the module docs in [`super`]) plugs into.
//!
//! Deliberately minimal seed: only the operations the phase-4 forward pass
//! consumes today, all delegating to the free functions that own the actual
//! implementations. Phase 4 will grow this trait as the decode loop firms
//! up (attention, sampling, the grouped MoE FFN) — resist adding surface
//! before a second caller or a second backend needs it.

use super::KernelError;
use super::gemv;
use super::primitives;
use super::quants::{BlockQ8_0, BlockQ8K, QuantFormat};

/// Compute backend for the forward pass.
///
/// Object-safe on purpose (the model holds a `Box<dyn Backend>` or a
/// generic; phase 4 decides). All methods keep the exact contracts of the
/// free functions they mirror — see [`gemv::gemv_q8_k`],
/// [`gemv::gemv_q8_0`], [`primitives::rmsnorm`],
/// [`primitives::rope_neox_heads`], [`primitives::softmax`], and
/// [`primitives::swiglu_combine`].
pub trait Backend {
    /// K-quant GEMV: Q4_K / Q5_K / Q6_K weights against Q8_K activations.
    fn gemv_q8_k(
        &self,
        format: QuantFormat,
        weight: &[u8],
        in_dim: usize,
        out_dim: usize,
        acts: &[BlockQ8K],
        out: &mut [f32],
    ) -> Result<(), KernelError>;

    /// Q8_0 GEMV: Q8_0 weights against Q8_0 activations.
    fn gemv_q8_0(
        &self,
        weight: &[u8],
        in_dim: usize,
        out_dim: usize,
        acts: &[BlockQ8_0],
        out: &mut [f32],
    ) -> Result<(), KernelError>;

    /// RMSNorm, Qwen3 convention (weight multiplies directly).
    fn rmsnorm(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
        out: &mut [f32],
    ) -> Result<(), KernelError>;

    /// NeoX RoPE applied in place to `n_heads` contiguous heads.
    fn rope_neox_heads(
        &self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        position: u32,
        theta_base: f32,
    ) -> Result<(), KernelError>;

    /// Numerically stable in-place softmax.
    fn softmax(&self, x: &mut [f32]) -> Result<(), KernelError>;

    /// SwiGLU combine in place into `gate`: `gate_i <- silu(gate_i) * up_i`.
    fn swiglu_combine(&self, gate: &mut [f32], up: &[f32]) -> Result<(), KernelError>;
}

/// The CPU backend: thin delegation to the free kernel functions (which do
/// their own AVX2/scalar dispatch where it matters).
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn gemv_q8_k(
        &self,
        format: QuantFormat,
        weight: &[u8],
        in_dim: usize,
        out_dim: usize,
        acts: &[BlockQ8K],
        out: &mut [f32],
    ) -> Result<(), KernelError> {
        gemv::gemv_q8_k(format, weight, in_dim, out_dim, acts, out)
    }

    fn gemv_q8_0(
        &self,
        weight: &[u8],
        in_dim: usize,
        out_dim: usize,
        acts: &[BlockQ8_0],
        out: &mut [f32],
    ) -> Result<(), KernelError> {
        gemv::gemv_q8_0(weight, in_dim, out_dim, acts, out)
    }

    fn rmsnorm(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
        out: &mut [f32],
    ) -> Result<(), KernelError> {
        primitives::rmsnorm(x, weight, eps, out)
    }

    fn rope_neox_heads(
        &self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        position: u32,
        theta_base: f32,
    ) -> Result<(), KernelError> {
        primitives::rope_neox_heads(x, n_heads, head_dim, position, theta_base)
    }

    fn softmax(&self, x: &mut [f32]) -> Result<(), KernelError> {
        primitives::softmax(x)
    }

    fn swiglu_combine(&self, gate: &mut [f32], up: &[f32]) -> Result<(), KernelError> {
        primitives::swiglu_combine(gate, up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The trait must stay object-safe (phase 4 may hold `dyn Backend`).
    #[test]
    fn object_safe_and_delegates() {
        let backend: &dyn Backend = &CpuBackend;

        // rmsnorm delegates (identical output to the free function).
        let x = [3.0f32, 4.0, 12.0, 84.0];
        let w = [1.0f32, 0.5, 2.0, 1.0];
        let mut via_trait = [0f32; 4];
        backend.rmsnorm(&x, &w, 1e-6, &mut via_trait).unwrap();
        let mut direct = [0f32; 4];
        primitives::rmsnorm(&x, &w, 1e-6, &mut direct).unwrap();
        assert_eq!(via_trait, direct);

        // softmax delegates.
        let mut s = [0.0f32, 1.0, 2.0];
        backend.softmax(&mut s).unwrap();
        let total: f32 = s.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);

        // swiglu delegates, and errors pass through unchanged.
        let mut gate = [1.0f32; 2];
        assert_eq!(
            backend.swiglu_combine(&mut gate, &[1.0; 3]).unwrap_err(),
            KernelError::LengthMismatch {
                what: "swiglu_combine: gate vs up",
                left: 2,
                right: 3,
            }
        );

        // rope delegates (position 0 is the identity).
        let mut h = [1.0f32, 2.0, 3.0, 4.0];
        backend.rope_neox_heads(&mut h, 1, 4, 0, 1e7).unwrap();
        assert_eq!(h, [1.0, 2.0, 3.0, 4.0]);

        // gemv delegates: a zero Q8_0 matrix maps anything to zero.
        let acts = vec![BlockQ8_0::default(); 8]; // 256 weights
        let weight = vec![0u8; 2 * 8 * 34];
        let mut out = [1.0f32; 2];
        backend.gemv_q8_0(&weight, 256, 2, &acts, &mut out).unwrap();
        assert_eq!(out, [0.0, 0.0]);

        // gemv_q8_k rejects non-k-quant formats through the trait too.
        let kacts = vec![BlockQ8K::default()];
        let mut kout = [0f32; 1];
        assert_eq!(
            backend
                .gemv_q8_k(QuantFormat::Q8_0, &[], 256, 1, &kacts, &mut kout)
                .unwrap_err(),
            KernelError::UnsupportedFormat {
                what: "gemv_q8_k",
                format: QuantFormat::Q8_0,
            }
        );
    }
}
