//! KV cache.
//!
//! FP16 K/V storage sized by the model's attention pattern: linear append-only
//! storage for full-attention layers; bounded ring buffers for sliding-window
//! layers (needed for Gemma 4's 25 SWA layers, where rings keep the cache
//! flat as context grows). Under a ~3 GB budget the KV cache is the
//! third-largest resident tenant after the expert slot pool and the common
//! weights, so its bounds are part of the memory contract, not an
//! implementation detail.
//!
//! v0 implements the linear variant only: [`KvCache`], one pair of
//! preallocated K and V planes per layer, each `[capacity, n_kv_heads *
//! head_dim]` of f16 bits (`u16`), written once per layer per token and
//! never moved. For the v0 pin (48 layers, 4 kv heads x 128 head_dim, full
//! attention on every layer) one token costs 48 layers x 2 planes x 512
//! elements x 2 B = 96 KiB across the model, so the 4K-token v0 context cap
//! is 384 MiB, allocated once in [`KvCache::new`] — the "KV cache FP16"
//! tenant in `docs/architecture.md`'s memory contract.
//!
//! Values convert f32 -> f16 on append (round to nearest-even, see
//! [`f32_to_f16`]) and are served back as raw bits; the attention kernel
//! converts per element on the fly. Each layer keeps its own write cursor:
//! the decode loop appends once per layer per token, so cursors advance
//! uniformly across a token while remaining individually representable
//! mid-token. [`KvCache::seq_len`] re-checks that uniformity and returns a
//! typed error on ragged lengths instead of asserting (`ramvamp-core` never
//! panics on bad input).
//!
//! The Gemma SWA ring variant will replace the append cursor with a modular
//! write index on windowed layers. Recorded decision: no `KvStorage` trait
//! yet — the linear cache's public surface (`append`, row views, lengths,
//! dims) is the contract a ring must also satisfy, and the trait gets
//! extracted when the second implementation exists, not before (same rule
//! as the backend trait: no surface before a second user).

use crate::kernels::quants::f32_to_f16;
use thiserror::Error;

/// Typed errors from KV cache construction and access.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum KvError {
    /// Construction rejected the requested geometry.
    #[error(
        "invalid kv cache dims (n_layers {n_layers}, n_kv_heads {n_kv_heads}, \
         head_dim {head_dim}, capacity {capacity}): {reason}"
    )]
    InvalidDims {
        /// Requested layer count.
        n_layers: usize,
        /// Requested kv-head count.
        n_kv_heads: usize,
        /// Requested per-head dimension.
        head_dim: usize,
        /// Requested position capacity.
        capacity: usize,
        /// Why the geometry was rejected.
        reason: &'static str,
    },

    /// A layer index is outside the cache.
    #[error("kv cache layer {layer} out of range: cache has {n_layers} layers")]
    LayerOutOfRange {
        /// The rejected layer index.
        layer: usize,
        /// Number of layers in the cache.
        n_layers: usize,
    },

    /// An appended K or V row has the wrong number of floats.
    #[error("kv cache {what} row has {got} floats, expected {expected} (n_kv_heads * head_dim)")]
    RowLenMismatch {
        /// Which row ("k" or "v") had the wrong length.
        what: &'static str,
        /// Offending row length.
        got: usize,
        /// Expected row length.
        expected: usize,
    },

    /// An append would exceed the layer's preallocated capacity.
    #[error("kv cache layer {layer} is full: capacity {capacity} positions")]
    CapacityExceeded {
        /// The full layer.
        layer: usize,
        /// The fixed position capacity.
        capacity: usize,
    },

    /// A position index is past the layer's appended length.
    #[error("kv cache position {pos} out of range: layer {layer} holds {len} positions")]
    PositionOutOfRange {
        /// The layer that was read.
        layer: usize,
        /// The rejected position.
        pos: usize,
        /// Positions currently held by that layer.
        len: usize,
    },

    /// Layers hold unequal lengths where a uniform sequence length was
    /// required.
    #[error(
        "kv cache layers hold unequal lengths ({shortest} vs {longest}): \
         appends must advance every layer once per token"
    )]
    RaggedLayers {
        /// Smallest per-layer length.
        shortest: usize,
        /// Largest per-layer length.
        longest: usize,
    },
}

/// Linear append-only FP16 KV cache for full-attention layers.
///
/// Storage is one flat `u16` (f16 bits) buffer per plane (K and V), laid out
/// as `n_layers` contiguous planes of `[capacity, n_kv_heads * head_dim]`
/// rows. Everything is allocated once in [`Self::new`]; `append` only
/// converts and writes, so no growth, movement, or reallocation ever happens
/// on the decode path.
#[derive(Debug)]
pub struct KvCache {
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    /// Row width in elements: `n_kv_heads * head_dim`.
    kv_dim: usize,
    /// Per-layer plane stride in elements: `capacity * kv_dim`.
    plane: usize,
    /// K planes for all layers, f16 bits.
    k: Vec<u16>,
    /// V planes for all layers, f16 bits.
    v: Vec<u16>,
    /// Per-layer write cursors (positions appended so far).
    lens: Vec<usize>,
}

impl KvCache {
    /// Preallocate a cache of `n_layers` K/V plane pairs, each holding
    /// `capacity` positions of `n_kv_heads * head_dim` f16 values.
    ///
    /// v0 sizing (from the caller): 48 layers, 4 kv heads, head_dim 128,
    /// capacity 4096 — 2 x 48 x 4096 x 512 x 2 B = 384 MiB, the whole-run
    /// allocation for the KV tenant.
    ///
    /// # Errors
    ///
    /// [`KvError::InvalidDims`] if any dimension is zero (a zero `head_dim`
    /// or `n_kv_heads` would poison every downstream shape computation) or
    /// if the total element count overflows `usize`.
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Self, KvError> {
        let invalid = |reason| KvError::InvalidDims {
            n_layers,
            n_kv_heads,
            head_dim,
            capacity,
            reason,
        };
        if n_layers == 0 || n_kv_heads == 0 || head_dim == 0 || capacity == 0 {
            return Err(invalid("all dims must be nonzero"));
        }
        let kv_dim = n_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| invalid("row width overflows usize"))?;
        let plane = capacity
            .checked_mul(kv_dim)
            .ok_or_else(|| invalid("plane size overflows usize"))?;
        let total = n_layers
            .checked_mul(plane)
            .ok_or_else(|| invalid("total size overflows usize"))?;
        Ok(Self {
            n_layers,
            n_kv_heads,
            head_dim,
            capacity,
            kv_dim,
            plane,
            k: vec![0; total],
            v: vec![0; total],
            lens: vec![0; n_layers],
        })
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Number of kv heads per position.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Fixed position capacity per layer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Row width in elements: `n_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Positions appended to `layer` so far.
    ///
    /// # Errors
    ///
    /// [`KvError::LayerOutOfRange`] if `layer >= n_layers`.
    pub fn len(&self, layer: usize) -> Result<usize, KvError> {
        self.check_layer(layer)?;
        Ok(self.lens[layer])
    }

    /// Whether no positions have been appended to any layer.
    pub fn is_empty(&self) -> bool {
        self.lens.iter().all(|&len| len == 0)
    }

    /// The uniform sequence length across all layers.
    ///
    /// The decode loop appends once per layer per token, so between tokens
    /// every layer holds the same count; this is the checked accessor for
    /// that invariant.
    ///
    /// # Errors
    ///
    /// [`KvError::RaggedLayers`] if layers disagree (an append sequence was
    /// abandoned mid-token) — reported instead of asserted so a caller bug
    /// surfaces as a recoverable error, never a panic.
    pub fn seq_len(&self) -> Result<usize, KvError> {
        // `new` guarantees at least one layer, so the fold always observes
        // a cursor; written without unwrap so there is no panic path at all.
        let (shortest, longest) = self
            .lens
            .iter()
            .fold((usize::MAX, 0), |(lo, hi), &len| (lo.min(len), hi.max(len)));
        if shortest != longest {
            return Err(KvError::RaggedLayers { shortest, longest });
        }
        Ok(shortest)
    }

    /// Append one position's K and V rows to `layer`, converting f32 -> f16
    /// (round to nearest-even).
    ///
    /// `k` and `v` must each hold `n_kv_heads * head_dim` floats (512 for
    /// the v0 pin), laid out head-major: `[head 0 | head 1 | ...]`.
    ///
    /// # Errors
    ///
    /// [`KvError::LayerOutOfRange`], [`KvError::RowLenMismatch`], or
    /// [`KvError::CapacityExceeded`] when the layer is full — nothing is
    /// written and the cursor does not advance on any error.
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) -> Result<(), KvError> {
        self.check_layer(layer)?;
        if k.len() != self.kv_dim {
            return Err(KvError::RowLenMismatch {
                what: "k",
                got: k.len(),
                expected: self.kv_dim,
            });
        }
        if v.len() != self.kv_dim {
            return Err(KvError::RowLenMismatch {
                what: "v",
                got: v.len(),
                expected: self.kv_dim,
            });
        }
        let len = self.lens[layer];
        if len == self.capacity {
            return Err(KvError::CapacityExceeded {
                layer,
                capacity: self.capacity,
            });
        }
        let base = layer * self.plane + len * self.kv_dim;
        for (dst, &src) in self.k[base..base + self.kv_dim].iter_mut().zip(k) {
            *dst = f32_to_f16(src);
        }
        for (dst, &src) in self.v[base..base + self.kv_dim].iter_mut().zip(v) {
            *dst = f32_to_f16(src);
        }
        self.lens[layer] = len + 1;
        Ok(())
    }

    /// The K row (f16 bits, `kv_dim` elements) for `pos` in `layer`.
    ///
    /// # Errors
    ///
    /// [`KvError::LayerOutOfRange`] or [`KvError::PositionOutOfRange`] if
    /// `pos` has not been appended yet.
    pub fn k_row(&self, layer: usize, pos: usize) -> Result<&[u16], KvError> {
        self.row(&self.k, layer, pos)
    }

    /// The V row (f16 bits, `kv_dim` elements) for `pos` in `layer`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::k_row`].
    pub fn v_row(&self, layer: usize, pos: usize) -> Result<&[u16], KvError> {
        self.row(&self.v, layer, pos)
    }

    /// All appended K rows of `layer` as one contiguous slice:
    /// `[len(layer), kv_dim]` row-major f16 bits. This is the attention
    /// kernel's bulk view — one bounds check for the whole layer instead of
    /// one per position.
    ///
    /// # Errors
    ///
    /// [`KvError::LayerOutOfRange`] if `layer >= n_layers`.
    pub fn k_layer(&self, layer: usize) -> Result<&[u16], KvError> {
        self.layer_view(&self.k, layer)
    }

    /// All appended V rows of `layer`; see [`Self::k_layer`].
    ///
    /// # Errors
    ///
    /// [`KvError::LayerOutOfRange`] if `layer >= n_layers`.
    pub fn v_layer(&self, layer: usize) -> Result<&[u16], KvError> {
        self.layer_view(&self.v, layer)
    }

    fn check_layer(&self, layer: usize) -> Result<(), KvError> {
        if layer >= self.n_layers {
            return Err(KvError::LayerOutOfRange {
                layer,
                n_layers: self.n_layers,
            });
        }
        Ok(())
    }

    fn row<'a>(&self, plane: &'a [u16], layer: usize, pos: usize) -> Result<&'a [u16], KvError> {
        self.check_layer(layer)?;
        let len = self.lens[layer];
        if pos >= len {
            return Err(KvError::PositionOutOfRange { layer, pos, len });
        }
        let start = layer * self.plane + pos * self.kv_dim;
        Ok(&plane[start..start + self.kv_dim])
    }

    fn layer_view<'a>(&self, plane: &'a [u16], layer: usize) -> Result<&'a [u16], KvError> {
        self.check_layer(layer)?;
        let start = layer * self.plane;
        Ok(&plane[start..start + self.lens[layer] * self.kv_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::quants::f16_to_f32;

    /// Deterministic patterned float in [-1, 1) that exercises f16 rounding
    /// (values are not exactly representable in half precision).
    fn sample(i: usize) -> f32 {
        let h = (i as u32).wrapping_mul(2_654_435_761);
        (h % 20_000) as f32 / 10_000.0 - 1.0
    }

    fn filled_row(tag: usize, kv_dim: usize) -> Vec<f32> {
        (0..kv_dim).map(|i| sample(tag * kv_dim + i)).collect()
    }

    #[test]
    fn new_rejects_zero_dims_and_overflow() {
        for (l, h, d, c) in [
            (0, 4, 128, 8),
            (48, 0, 128, 8),
            (48, 4, 0, 8),
            (48, 4, 128, 0),
        ] {
            assert_eq!(
                KvCache::new(l, h, d, c).unwrap_err(),
                KvError::InvalidDims {
                    n_layers: l,
                    n_kv_heads: h,
                    head_dim: d,
                    capacity: c,
                    reason: "all dims must be nonzero",
                }
            );
        }
        assert!(matches!(
            KvCache::new(usize::MAX, usize::MAX, 2, 2).unwrap_err(),
            KvError::InvalidDims { .. }
        ));
    }

    #[test]
    fn f16_round_trip_through_cache() {
        let kv_dim = 2 * 8;
        let mut cache = KvCache::new(2, 2, 8, 4).unwrap();
        let k = filled_row(0, kv_dim);
        let v = filled_row(1, kv_dim);
        cache.append(0, &k, &v).unwrap();
        cache.append(1, &v, &k).unwrap(); // Layers store independent data.

        // Stored bits are exactly the nearest-even f16 of the input, so the
        // read-back value is within one half-precision rounding step
        // (<= 2^-11 relative) of the source by construction.
        let k_bits = cache.k_row(0, 0).unwrap();
        let v_bits = cache.v_row(0, 0).unwrap();
        for i in 0..kv_dim {
            assert_eq!(k_bits[i], f32_to_f16(k[i]));
            assert_eq!(v_bits[i], f32_to_f16(v[i]));
            let back = f16_to_f32(k_bits[i]);
            assert!(
                (back - k[i]).abs() <= k[i].abs() * 4.9e-4 + 6.0e-8,
                "index {i}: {back} vs {}",
                k[i]
            );
        }
        // Layer 1 got the swapped rows, not layer 0's.
        assert_eq!(cache.k_row(1, 0).unwrap()[0], f32_to_f16(v[0]));
    }

    #[test]
    fn append_advances_rows_and_layer_views() {
        let kv_dim = 3 * 4;
        let mut cache = KvCache::new(1, 3, 4, 8).unwrap();
        let rows: Vec<Vec<f32>> = (0..5).map(|t| filled_row(t, kv_dim)).collect();
        for row in &rows {
            cache.append(0, row, row).unwrap();
        }
        assert_eq!(cache.len(0).unwrap(), 5);
        assert_eq!(cache.seq_len().unwrap(), 5);
        assert!(!cache.is_empty());
        let k_all = cache.k_layer(0).unwrap();
        assert_eq!(k_all.len(), 5 * kv_dim);
        for (t, row) in rows.iter().enumerate() {
            let via_row = cache.k_row(0, t).unwrap();
            assert_eq!(&k_all[t * kv_dim..(t + 1) * kv_dim], via_row);
            assert_eq!(via_row[3], f32_to_f16(row[3]));
        }
    }

    #[test]
    fn typed_errors_on_misuse() {
        let mut cache = KvCache::new(2, 2, 4, 3).unwrap();
        let row = vec![0.5f32; 8];

        // Layer bounds on every accessor.
        let oor = KvError::LayerOutOfRange {
            layer: 2,
            n_layers: 2,
        };
        assert_eq!(cache.append(2, &row, &row).unwrap_err(), oor);
        assert_eq!(cache.len(2).unwrap_err(), oor);
        assert_eq!(cache.k_row(2, 0).unwrap_err(), oor);
        assert_eq!(cache.v_row(2, 0).unwrap_err(), oor);
        assert_eq!(cache.k_layer(2).unwrap_err(), oor);
        assert_eq!(cache.v_layer(2).unwrap_err(), oor);

        // Row length mismatches, k and v separately.
        assert_eq!(
            cache.append(0, &row[..7], &row).unwrap_err(),
            KvError::RowLenMismatch {
                what: "k",
                got: 7,
                expected: 8,
            }
        );
        assert_eq!(
            cache.append(0, &row, &[]).unwrap_err(),
            KvError::RowLenMismatch {
                what: "v",
                got: 0,
                expected: 8,
            }
        );

        // Reads past the appended length.
        cache.append(0, &row, &row).unwrap();
        assert_eq!(
            cache.k_row(0, 1).unwrap_err(),
            KvError::PositionOutOfRange {
                layer: 0,
                pos: 1,
                len: 1,
            }
        );
        assert_eq!(
            cache.v_row(1, 0).unwrap_err(),
            KvError::PositionOutOfRange {
                layer: 1,
                pos: 0,
                len: 0,
            }
        );

        // Capacity overflow: fourth append to a capacity-3 layer fails and
        // does not advance the cursor.
        cache.append(0, &row, &row).unwrap();
        cache.append(0, &row, &row).unwrap();
        assert_eq!(
            cache.append(0, &row, &row).unwrap_err(),
            KvError::CapacityExceeded {
                layer: 0,
                capacity: 3,
            }
        );
        assert_eq!(cache.len(0).unwrap(), 3);
    }

    #[test]
    fn seq_len_rejects_ragged_layers() {
        let mut cache = KvCache::new(3, 1, 4, 4).unwrap();
        assert_eq!(cache.seq_len().unwrap(), 0);
        assert!(cache.is_empty());
        let row = [1.0f32; 4];
        cache.append(0, &row, &row).unwrap();
        assert_eq!(
            cache.seq_len().unwrap_err(),
            KvError::RaggedLayers {
                shortest: 0,
                longest: 1,
            }
        );
        cache.append(1, &row, &row).unwrap();
        cache.append(2, &row, &row).unwrap();
        assert_eq!(cache.seq_len().unwrap(), 1);
    }
}
