//! Block layouts mirroring ggml's `ggml-common.h`, plus the activation-side
//! block structs.
//!
//! Weight blocks are described as byte offsets into the packed on-disk form
//! (all fields little-endian, no padding -- the C structs are naturally
//! packed). Dequantization formulas, with `f16()` meaning f16-to-f32:
//!
//! - `block_q8_0` (34 B / 32 weights): `d: f16` at 0, `qs: [i8; 32]` at 2.
//!   `w[j] = f16(d) * qs[j]`.
//! - `block_q4_K` (144 B / 256 weights): `d: f16` at 0, `dmin: f16` at 2,
//!   `scales: [u8; 12]` at 4 (eight 6-bit (scale, min) pairs, packed per
//!   [`get_scale_min_k4`]), `qs: [u8; 128]` at 16. Eight sub-blocks of 32.
//!   For sub-block `j` with unpacked `(sc, m)`:
//!   `w = f16(d)*sc*q - f16(dmin)*m`, `q` a 4-bit unsigned value. Nibble
//!   order: within each 64-weight span, `qs` byte `l` holds weight `l` in
//!   its low nibble and weight `l + 32` in its high nibble.
//! - `block_q5_K` (176 B / 256 weights): `d: f16` at 0, `dmin: f16` at 2,
//!   `scales: [u8; 12]` at 4 (same packing as q4_K), `qh: [u8; 32]` at 16,
//!   `qs: [u8; 128]` at 48. Same affine formula as q4_K with a 5-bit `q`:
//!   the low 4 bits come from `qs` (q4_K nibble order) and the 5th bit from
//!   `qh` -- for the 64-weight span `s` (0..4), weight `l` of the low-nibble
//!   half uses `qh[l] & (1 << 2s)` and weight `l` of the high-nibble half
//!   uses `qh[l] & (2 << 2s)`.
//! - `block_q6_K` (210 B / 256 weights): `ql: [u8; 128]` at 0,
//!   `qh: [u8; 64]` at 128, `scales: [i8; 16]` at 192, `d: f16` at 208.
//!   Sixteen sub-blocks of 16: `w = f16(d) * scales[i/16] * q` with
//!   `q = ((low 4 bits) | (2 bits from qh) << 4) - 32` in [-32, 31]. Within
//!   each 128-weight half (using half-local `ql`/`qh`/`scales` slices and
//!   `l` in 0..32): weight `l` is `ql[l] & 0xF` with `qh[l] & 3`; weight
//!   `l + 32` is `ql[l + 32] & 0xF` with `(qh[l] >> 2) & 3`; weight `l + 64`
//!   is `ql[l] >> 4` with `(qh[l] >> 4) & 3`; weight `l + 96` is
//!   `ql[l + 32] >> 4` with `(qh[l] >> 6) & 3`.
//! - `block_q8_K` (292 B / 256 weights, activation-only here): `d: f32`,
//!   `qs: [i8; 256]`, `bsums: [i16; 16]` -- `bsums[g]` is the sum of the
//!   16-element group `qs[16g .. 16g + 16]`, which lets the k-quant dots
//!   apply the per-sub-block mins without touching `qs` again.
//!   `w[j] = d * qs[j]`. Never packed to bytes in ramvamp; see [`BlockQ8K`].

use super::f16::f16_to_f32;

/// Weights per k-quant super-block (ggml `QK_K`).
pub(super) const QK_K: usize = 256;
/// Weights per legacy q8_0 block (ggml `QK8_0`).
pub(super) const QK8_0: usize = 32;

/// Byte offsets inside a packed `block_q4_K`.
pub(super) mod q4k {
    /// `d: f16`.
    pub const D: usize = 0;
    /// `dmin: f16`.
    pub const DMIN: usize = 2;
    /// `scales: [u8; 12]`.
    pub const SCALES: usize = 4;
    /// `qs: [u8; 128]`.
    pub const QS: usize = 16;
    /// Total block size.
    pub const BYTES: usize = 144;
}

/// Byte offsets inside a packed `block_q5_K`.
pub(super) mod q5k {
    /// `d: f16`.
    pub const D: usize = 0;
    /// `dmin: f16`.
    pub const DMIN: usize = 2;
    /// `scales: [u8; 12]`.
    pub const SCALES: usize = 4;
    /// `qh: [u8; 32]`.
    pub const QH: usize = 16;
    /// `qs: [u8; 128]`.
    pub const QS: usize = 48;
    /// Total block size.
    pub const BYTES: usize = 176;
}

/// Byte offsets inside a packed `block_q6_K`.
pub(super) mod q6k {
    /// `ql: [u8; 128]`.
    pub const QL: usize = 0;
    /// `qh: [u8; 64]`.
    pub const QH: usize = 128;
    /// `scales: [i8; 16]`.
    pub const SCALES: usize = 192;
    /// `d: f16`.
    pub const D: usize = 208;
    /// Total block size.
    pub const BYTES: usize = 210;
}

/// Byte offsets inside a packed `block_q8_0`.
pub(super) mod q8_0 {
    /// `d: f16`.
    pub const D: usize = 0;
    /// `qs: [i8; 32]`.
    pub const QS: usize = 2;
    /// Total block size.
    pub const BYTES: usize = 34;
}

/// Read the little-endian f16 at `off` in a block and widen to f32.
///
/// Byte-wise load: no alignment assumed beyond 1 byte.
#[inline]
pub(super) fn f16_at(block: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[off], block[off + 1]]))
}

/// Borrow the 12-byte k-quant scale field out of a block.
///
/// Infallible for callers that pass in-bounds constant offsets into a
/// length-validated block; `expect` documents the invariant.
#[inline]
pub(super) fn scales_at(block: &[u8], off: usize) -> &[u8; 12] {
    block[off..off + 12]
        .try_into()
        .expect("in-bounds 12-byte field")
}

/// Unpack the 6-bit (scale, min) pair for sub-block `j` (0..8) from the
/// 12-byte k-quant scale field.
///
/// Mirrors ggml's `get_scale_min_k4`: pairs 0..4 use the low 6 bits of
/// bytes `j` / `j + 4`; pairs 4..8 use the low nibbles of bytes `j + 4` as
/// low bits and the previously unused high 2 bits of bytes `j - 4` / `j` as
/// high bits.
#[inline]
pub(super) fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// One Q8_K activation block: 256 int8 weights with an f32 scale and
/// per-16-element group sums.
///
/// In-memory analogue of ggml's `block_q8_K`. Produced per token by
/// [`super::quantize_row_q8_k`]; never serialized. `bsums[g]` must equal
/// `qs[16g..16g + 16].iter().sum()` -- the k-quant dot products rely on it
/// to fold the per-sub-block mins in via one multiply per group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockQ8K {
    /// Dequantization scale: `w[j] = d * qs[j]`.
    pub d: f32,
    /// Quantized values in [-127, 127].
    pub qs: [i8; QK_K],
    /// Sum of each 16-element group of `qs`.
    pub bsums: [i16; QK_K / 16],
}

impl Default for BlockQ8K {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        }
    }
}

/// One Q8_0 activation block: 32 int8 weights with a single scale.
///
/// In-memory analogue of ggml's `block_q8_0`, except `d` is kept as f32.
/// [`super::quantize_row_q8_0`] stores the f16-rounded value (the f32 you
/// get from reading back the f16 ggml would have written), so arithmetic
/// against it matches llama.cpp's `vec_dot_q8_0_q8_0` exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub struct BlockQ8_0 {
    /// Dequantization scale, already rounded through f16: `w[j] = d * qs[j]`.
    pub d: f32,
    /// Quantized values in [-127, 127].
    pub qs: [i8; QK8_0],
}

impl Default for BlockQ8_0 {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK8_0],
        }
    }
}
