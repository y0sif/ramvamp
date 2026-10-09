//! Arm A: the production CPU path, driven exactly the way prefill drives it.
//!
//! `model::prefill::scatter_rows` shards `0..out_dim` over the pinned
//! compute pool and hands each shard the contiguous `rows.len() * n_acts`
//! sub-slice of `out` that belongs to it, then calls `gemv_q8_k_batched`
//! unchanged. That private helper is reproduced here rather than
//! approximated, so arm A is the same kernel under the same threading.

use anyhow::{Result, anyhow};
use std::sync::Mutex;

use ramvamp_core::kernels::gemv_q8_k_batched;
use ramvamp_core::kernels::q4k_row_hoist::gemv_q4_k_q8_k_batched_hoisted;
use ramvamp_core::kernels::quants::{BlockQ8K, QuantFormat};
use ramvamp_core::threads::ComputePool;

/// A raw pointer into the caller's `out`, made shareable across the pool.
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);

// SAFETY: the pointer is only ever offset into disjoint sub-ranges of the
// one `&mut [f32]` the caller lends for the whole call, one range per shard,
// and `ComputePool::run` joins every shard before returning.
unsafe impl Send for SendPtr {}
// SAFETY: as above; the pointer is shared, the bytes it reaches are not.
unsafe impl Sync for SendPtr {}

impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Which CPU kernel the arm runs. `Production` is what prefill calls today;
/// `Hoisted` is the preserved row-hoist prototype from
/// `perf/q4k-row-hoist-prototype`, which does the per-row dequant setup once
/// per row instead of once per activation, the way the shader does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpuKernel {
    Production,
    Hoisted,
}

/// Batched Q4_K x Q8_K GEMV over the whole matrix, sharded across the pool.
///
/// `out` is `[weight_row][activation]`, `out_dim * n_acts` floats, exactly
/// the layout `gemv_q8_k_batched` writes.
#[allow(clippy::too_many_arguments)]
pub fn gemv_batched_pooled(
    kernel: CpuKernel,
    pool: &mut ComputePool,
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    acts: &[BlockQ8K],
    n_acts: usize,
    out: &mut [f32],
) -> Result<()> {
    assert_eq!(out.len(), out_dim * n_acts, "out vs out_dim * n_acts");
    if n_acts == 0 || out_dim == 0 {
        return Ok(());
    }
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let base = SendPtr(out.as_mut_ptr());
    pool.run(out_dim, |shard| {
        let start = shard.rows.start * n_acts;
        let len = shard.rows.len() * n_acts;
        // SAFETY: `shard_range` yields disjoint sub-ranges of `0..out_dim`,
        // each visited by exactly one shard, so scaling them by `n_acts`
        // yields disjoint sub-ranges of `0..out_dim * n_acts`, which is
        // `out.len()` (asserted above). `out` is mutably borrowed for the
        // whole call and `run` joins before returning.
        let chunk = unsafe { std::slice::from_raw_parts_mut(base.get().add(start), len) };
        let result = match kernel {
            CpuKernel::Production => gemv_q8_k_batched(
                QuantFormat::Q4_K,
                weight,
                in_dim,
                out_dim,
                acts,
                n_acts,
                shard.rows.clone(),
                chunk,
            ),
            CpuKernel::Hoisted => gemv_q4_k_q8_k_batched_hoisted(
                weight,
                in_dim,
                out_dim,
                acts,
                n_acts,
                shard.rows.clone(),
                chunk,
            ),
        };
        if let Err(err) = result {
            let mut slot = failure.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = Some(err.to_string());
            }
        }
    });
    match failure.into_inner().unwrap_or_else(|e| e.into_inner()) {
        Some(err) => Err(anyhow!("{kernel:?} CPU kernel failed on a shard: {err}")),
        None => Ok(()),
    }
}
