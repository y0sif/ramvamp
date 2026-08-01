//! Typed errors for GGUF parsing. Everything a malformed or hostile file
//! can trigger maps to a variant here; the parser never panics.

use crate::gguf::types::GgmlType;
use crate::source::SourceError;

/// Error parsing a GGUF file or computing ranges from its index.
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    /// The underlying source failed.
    #[error(transparent)]
    Source(#[from] SourceError),

    /// The file does not start with the `GGUF` magic.
    #[error("not a GGUF file: bad magic {found:02x?}")]
    BadMagic { found: [u8; 4] },

    /// Unsupported container version (little-endian v2/v3 only).
    #[error("unsupported GGUF version {0} (little-endian v2/v3 only)")]
    UnsupportedVersion(u32),

    /// The file ended in the middle of the header.
    #[error("truncated file: wanted {wanted} bytes at offset {offset}, file is {file_len} bytes")]
    UnexpectedEof {
        offset: u64,
        wanted: u64,
        file_len: u64,
    },

    /// The header region (metadata + tensor index) exceeds the cap.
    #[error("header exceeds {limit} bytes")]
    HeaderTooLarge { limit: u64 },

    /// The header claims more tensors than we accept.
    #[error("tensor count {count} exceeds limit {limit}")]
    TooManyTensors { count: u64, limit: u64 },

    /// The header claims more metadata key-values than we accept.
    #[error("metadata key-value count {count} exceeds limit {limit}")]
    TooManyMetadataKv { count: u64, limit: u64 },

    /// A string length field exceeds the cap.
    #[error("string of {len} bytes at offset {offset} exceeds limit {max}")]
    StringTooLong { offset: u64, len: u64, max: u64 },

    /// A string is not valid UTF-8.
    #[error("invalid UTF-8 in string at offset {offset}")]
    InvalidUtf8 { offset: u64 },

    /// A metadata value carries an unknown type id.
    #[error("unknown metadata value type {ty} at offset {offset}")]
    UnknownValueType { ty: u32, offset: u64 },

    /// Arrays may nest at most one level (array of arrays of scalars).
    #[error("metadata array at offset {offset} nests deeper than one level")]
    ArrayTooDeep { offset: u64 },

    /// An array claims more elements than the remaining file could encode.
    #[error(
        "array of {count} elements at offset {offset} cannot fit in {remaining} remaining bytes"
    )]
    ArrayTooLong {
        count: u64,
        offset: u64,
        remaining: u64,
    },

    /// Total metadata array elements exceed the global budget.
    #[error("metadata exceeds {limit} total array elements")]
    MetadataTooLarge { limit: u64 },

    /// A bool byte was neither 0 nor 1.
    #[error("invalid bool value {value} at offset {offset}")]
    InvalidBool { value: u8, offset: u64 },

    /// The same metadata key appeared twice.
    #[error("duplicate metadata key {0:?}")]
    DuplicateKey(String),

    /// The same tensor name appeared twice in the index.
    #[error("duplicate tensor name {0:?}")]
    DuplicateTensor(String),

    /// `general.alignment` is present but not an unsigned integer.
    #[error("general.alignment must be an unsigned integer")]
    AlignmentNotUint,

    /// `general.alignment` is zero, not a power of two, or too large.
    #[error("invalid general.alignment {0}: must be a power of two <= 1 MiB")]
    BadAlignment(u64),

    /// A tensor's dimension count is outside 1..=4.
    #[error("tensor {name:?}: invalid dimension count {n_dims} (must be 1..=4)")]
    BadDimCount { name: String, n_dims: u32 },

    /// A tensor has a zero-sized dimension.
    #[error("tensor {name:?}: dimension {axis} is zero")]
    ZeroDim { name: String, axis: usize },

    /// A tensor uses a ggml type id we do not handle.
    #[error("tensor {name:?}: unknown or unsupported ggml type id {ty}")]
    UnsupportedTensorType { name: String, ty: u32 },

    /// `dims[0]` is not divisible by the type's quantization block width.
    #[error("tensor {name:?}: dims[0]={dim0} not divisible by {ty:?} block width {block}")]
    DimNotBlockAligned {
        name: String,
        dim0: u64,
        ty: GgmlType,
        block: u64,
    },

    /// Element count or byte size overflows u64.
    #[error("tensor {name:?}: element count or byte size overflows u64")]
    SizeOverflow { name: String },

    /// A tensor's data offset is not aligned to `general.alignment`.
    #[error("tensor {name:?}: data offset {rel_offset} not aligned to {alignment}")]
    MisalignedTensor {
        name: String,
        rel_offset: u64,
        alignment: u64,
    },

    /// A tensor's byte range extends past the end of the data section.
    #[error(
        "tensor {name:?}: range {rel_offset}+{size_bytes} exceeds {data_len}-byte data section"
    )]
    TensorOutOfBounds {
        name: String,
        rel_offset: u64,
        size_bytes: u64,
        data_len: u64,
    },

    /// Expert-slab math requires a 3-D `*_exps` tensor.
    #[error("tensor {name:?} is {n_dims}-D; expert slabs need a 3-D tensor")]
    NotExpertTensor { name: String, n_dims: usize },

    /// The requested expert index is past the expert axis.
    #[error("expert index {expert_idx} out of range for tensor {name:?} with {n_experts} experts")]
    ExpertOutOfRange {
        name: String,
        expert_idx: u64,
        n_experts: u64,
    },

    /// The tensor's size is not divisible by its expert count.
    #[error("tensor {name:?}: size {size_bytes} not divisible by {n_experts} experts")]
    SlabNotDivisible {
        name: String,
        size_bytes: u64,
        n_experts: u64,
    },
}
