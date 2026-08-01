//! Data types shared by the GGUF parser: ggml tensor storage types,
//! metadata values, and tensor index entries.

/// ggml tensor storage type.
///
/// Numeric ids follow the `ggml_type` enum in upstream ggml
/// (`ggml/include/ggml.h` in ggml-org/llama.cpp); GGUF stores these ids
/// verbatim in tensor index entries and they are stable across GGUF v2/v3.
/// Only the types that can appear in the Q4_K_M-family sources ramvamp
/// repacks (plus the bring-up formats) are accepted; any other id is
/// rejected at parse time with `GgufError::UnsupportedTensorType`. Skipped
/// ids for reference: 3 Q4_1, 6 Q5_0, 7 Q5_1, 9 Q8_1, 10 Q2_K, 11 Q3_K,
/// 16..=29 i-quants/int/f64 types (4 and 5 were removed upstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    /// IEEE 754 float32. 1 weight / 4 bytes.
    F32 = 0,
    /// IEEE 754 float16. 1 weight / 2 bytes.
    F16 = 1,
    /// Legacy 4-bit block quant: f16 scale + 32 4-bit weights.
    /// 32 weights / 18 bytes.
    Q4_0 = 2,
    /// Legacy 8-bit block quant: f16 scale + 32 int8 weights.
    /// 32 weights / 34 bytes.
    Q8_0 = 8,
    /// 4-bit k-quant super-block: f16 d + f16 dmin + 12 B packed 6-bit
    /// sub-scales + 128 B nibbles. 256 weights / 144 bytes.
    Q4_K = 12,
    /// 5-bit k-quant super-block: Q4_K layout + 32 B of high bits.
    /// 256 weights / 176 bytes.
    Q5_K = 13,
    /// 6-bit k-quant super-block: 128 B low nibbles + 64 B high bits +
    /// 16 int8 sub-scales + f16 d. 256 weights / 210 bytes.
    Q6_K = 14,
    /// 8-bit k-quant (activation-side format): f32 d + 256 int8 + 16 i16
    /// block sums. 256 weights / 292 bytes. Rarely stored in files.
    Q8_K = 15,
    /// bfloat16. 1 weight / 2 bytes.
    BF16 = 30,
}

impl GgmlType {
    /// Map a raw GGUF/ggml type id to a supported type.
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            30 => Self::BF16,
            _ => return None,
        })
    }

    /// Weights per quantization block. Blocks span the innermost axis, so
    /// `dims[0]` must be divisible by this.
    pub const fn block_weights(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
        }
    }

    /// Bytes per quantization block.
    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
        }
    }
}

/// A GGUF metadata value. Arrays may nest one level (array of arrays of
/// scalars/strings); the parser rejects anything deeper.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// The value as a u64, if it is any unsigned integer or a non-negative
    /// signed integer. Writers disagree on integer widths (the spec says
    /// `general.alignment` is uint32, files vary), so lookups widen.
    pub fn as_uint(&self) -> Option<u64> {
        match *self {
            MetaValue::U8(v) => Some(u64::from(v)),
            MetaValue::U16(v) => Some(u64::from(v)),
            MetaValue::U32(v) => Some(u64::from(v)),
            MetaValue::U64(v) => Some(v),
            MetaValue::I8(v) if v >= 0 => Some(v as u64),
            MetaValue::I16(v) if v >= 0 => Some(v as u64),
            MetaValue::I32(v) if v >= 0 => Some(v as u64),
            MetaValue::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    /// The value as a string slice, if it is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// One entry of the GGUF tensor index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name, e.g. `blk.0.ffn_gate_exps.weight`.
    pub name: String,
    /// Dimensions in ggml `ne` order: `dims[0]` is the innermost
    /// (contiguous) axis; for 3-D `*_exps` tensors `dims[2]` is the expert
    /// axis (slowest).
    pub dims: Vec<u64>,
    /// Storage type of the tensor data.
    pub ggml_type: GgmlType,
    /// Byte offset of the tensor data relative to the start of the data
    /// section. Always a multiple of `general.alignment`.
    pub rel_offset: u64,
    /// Exact byte size of the tensor data, computed from `dims` and the
    /// type's block geometry.
    pub size_bytes: u64,
}
