//! GGUF container parsing (v3, plus read-compatible v2).
//!
//! Parses the header of a GGUF file — magic, version, metadata key-values,
//! and the tensor index — over a [`RangeRead`] source without ever
//! materializing the file. Tensor data is never read here; the parser only
//! computes and validates byte ranges so the repacker can stream slabs with
//! bounded scratch.
//!
//! GGUF files are untrusted input. Every count, length, and offset from the
//! file is bounds-checked before use, string/array/tensor counts are capped
//! (see the `MAX_*` constants), all arithmetic on parsed values is checked,
//! and failures are typed [`GgufError`]s — never panics, never unbounded
//! allocation. Only little-endian files are supported (a big-endian v3
//! file's version field reads back byte-swapped and is rejected as an
//! unsupported version).

mod error;
mod parse;
mod types;

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) mod testutil;

pub use error::GgufError;
pub use types::{GgmlType, MetaValue, TensorInfo};

use std::collections::BTreeMap;
use std::ops::Range;

use crate::source::RangeRead;

/// Maximum tensor count accepted from a header.
pub const MAX_TENSORS: u64 = 100_000;
/// Maximum metadata key-value count accepted from a header.
pub const MAX_METADATA_KV: u64 = 100_000;
/// Maximum byte length of any metadata string or key (1 MiB).
pub const MAX_STRING_LEN: u64 = 1 << 20;
/// Maximum byte length of a tensor name. The GGUF spec says 64 bytes; we
/// allow slack for out-of-spec writers while staying bounded.
pub const MAX_TENSOR_NAME_LEN: u64 = 256;
/// Maximum tensor rank (`GGML_MAX_DIMS`).
pub const MAX_DIMS: u32 = 4;
/// Largest accepted `general.alignment` (must also be a power of two).
pub const MAX_ALIGNMENT: u64 = 1 << 20;
/// Data-section alignment used when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// Total metadata array elements accepted across the whole header (1 Mi).
/// Real tokenizer arrays are ~300K elements, so 1 Mi gives margin while
/// capping heap amplification: each declared element becomes an in-memory
/// [`MetaValue`] many times larger than its encoded byte, so a small
/// crafted file must not be able to drive hundreds of MiB of parser heap.
/// Bounds parser heap usage together with `MAX_HEADER_BYTES`.
pub const MAX_META_ELEMENTS: u64 = 1 << 20;
/// Maximum size of the header region (metadata + tensor index). Real
/// GGUF headers, tokenizer included, are tens of MiB; 256 MiB is generous.
pub const MAX_HEADER_BYTES: u64 = 256 << 20;

/// A parsed GGUF header: metadata and tensor index, plus the geometry of
/// the data section. Holds no tensor data and no reference to the source.
#[derive(Debug)]
pub struct GgufFile {
    /// Metadata key-value pairs, sorted by key.
    pub metadata: BTreeMap<String, MetaValue>,
    // Private so it cannot be mutated out from under `by_name`, which
    // indexes into it; read access goes through [`GgufFile::tensors`].
    tensors: Vec<TensorInfo>,
    version: u32,
    alignment: u64,
    data_start: u64,
    data_len: u64,
    by_name: BTreeMap<String, usize>,
}

impl GgufFile {
    /// Parse the header of a GGUF file from `source`.
    ///
    /// Validates the magic, version (little-endian v2/v3), all metadata,
    /// and every tensor index entry: known ggml type, `dims[0]` divisible
    /// by the type's block width, offset aligned to `general.alignment`,
    /// and the computed byte range inside the data section.
    pub fn parse(source: &dyn RangeRead) -> Result<GgufFile, GgufError> {
        parse::parse(source)
    }

    /// GGUF container version (2 or 3).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Data-section alignment (`general.alignment`, default 32).
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Absolute file offset where the tensor-data section starts.
    pub fn data_section_offset(&self) -> u64 {
        self.data_start
    }

    /// Size of the tensor-data section in bytes (through end of file).
    pub fn data_section_len(&self) -> u64 {
        self.data_len
    }

    /// Tensor index entries in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Look up a tensor index entry by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    /// Absolute file offset of `tensor`'s data.
    ///
    /// Meaningful for [`TensorInfo`]s obtained from this file; parsing
    /// already validated that `rel_offset + size_bytes` lies inside the
    /// data section.
    pub fn data_offset(&self, tensor: &TensorInfo) -> u64 {
        self.data_start.saturating_add(tensor.rel_offset)
    }

    /// Absolute byte range of one expert's contiguous slab in a 3-D
    /// `*_exps` tensor (expert = slowest axis, `dims[2]`).
    ///
    /// The slab is `size_bytes / dims[2]` bytes; expert `i` occupies
    /// `data_offset + i * slab .. data_offset + (i + 1) * slab`.
    pub fn expert_slab(
        &self,
        tensor: &TensorInfo,
        expert_idx: u64,
    ) -> Result<Range<u64>, GgufError> {
        if tensor.dims.len() != 3 {
            return Err(GgufError::NotExpertTensor {
                name: tensor.name.clone(),
                n_dims: tensor.dims.len(),
            });
        }
        let n_experts = tensor.dims[2];
        if expert_idx >= n_experts {
            return Err(GgufError::ExpertOutOfRange {
                name: tensor.name.clone(),
                expert_idx,
                n_experts,
            });
        }
        // Always true for tensors this parser produced (size is a multiple
        // of every dim beyond the block axis), but `TensorInfo` fields are
        // public, so re-validate instead of trusting the caller.
        if !tensor.size_bytes.is_multiple_of(n_experts) {
            return Err(GgufError::SlabNotDivisible {
                name: tensor.name.clone(),
                size_bytes: tensor.size_bytes,
                n_experts,
            });
        }
        let slab = tensor.size_bytes / n_experts;
        let overflow = || GgufError::SizeOverflow {
            name: tensor.name.clone(),
        };
        let start = expert_idx
            .checked_mul(slab)
            .and_then(|off| self.data_offset(tensor).checked_add(off))
            .ok_or_else(overflow)?;
        let end = start.checked_add(slab).ok_or_else(overflow)?;
        Ok(start..end)
    }
}
