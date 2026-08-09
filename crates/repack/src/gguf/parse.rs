//! Streaming GGUF header parser over a [`RangeRead`] source.
//!
//! Reads forward through the header with a small chunked buffer; the file
//! is never materialized. Every count, length, and offset from the file is
//! treated as hostile: bounded before any allocation, checked arithmetic
//! throughout, typed errors on every failure path.

use std::collections::BTreeMap;

use crate::gguf::error::GgufError;
use crate::gguf::types::{GgmlType, MetaValue, TensorInfo};
use crate::gguf::{
    DEFAULT_ALIGNMENT, GgufFile, MAX_ALIGNMENT, MAX_DIMS, MAX_HEADER_BYTES, MAX_META_ELEMENTS,
    MAX_METADATA_KV, MAX_STRING_LEN, MAX_TENSOR_NAME_LEN, MAX_TENSORS,
};
use crate::source::RangeRead;

/// GGUF metadata value type ids (`gguf_metadata_value_type` in the spec).
const T_UINT8: u32 = 0;
const T_INT8: u32 = 1;
const T_UINT16: u32 = 2;
const T_INT16: u32 = 3;
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_UINT64: u32 = 10;
const T_INT64: u32 = 11;
const T_FLOAT64: u32 = 12;

/// Read-ahead chunk for the forward cursor.
const CHUNK: usize = 64 * 1024;

/// Forward-only cursor over a `RangeRead` with a small internal buffer, so
/// header parsing does not issue one positioned read per primitive.
struct Cursor<'a> {
    src: &'a dyn RangeRead,
    src_len: u64,
    pos: u64,
    buf: Vec<u8>,
    buf_start: u64,
}

impl<'a> Cursor<'a> {
    fn new(src: &'a dyn RangeRead) -> Self {
        Cursor {
            src,
            src_len: src.len(),
            pos: 0,
            buf: Vec::new(),
            buf_start: 0,
        }
    }

    fn pos(&self) -> u64 {
        self.pos
    }

    fn remaining(&self) -> u64 {
        self.src_len.saturating_sub(self.pos)
    }

    /// Hard cap on how far into the file the header may extend. Checked
    /// before every key-value, array element, and tensor entry, so parser
    /// heap usage is bounded regardless of file contents.
    fn check_header_cap(&self) -> Result<(), GgufError> {
        if self.pos > MAX_HEADER_BYTES {
            return Err(GgufError::HeaderTooLarge {
                limit: MAX_HEADER_BYTES,
            });
        }
        Ok(())
    }

    fn read_exact(&mut self, out: &mut [u8]) -> Result<(), GgufError> {
        let mut done = 0usize;
        while done < out.len() {
            let buf_end = self.buf_start.saturating_add(self.buf.len() as u64);
            if self.pos >= self.buf_start && self.pos < buf_end {
                let start = (self.pos - self.buf_start) as usize;
                let n = (self.buf.len() - start).min(out.len() - done);
                out[done..done + n].copy_from_slice(&self.buf[start..start + n]);
                done += n;
                self.pos += n as u64;
            } else {
                let remaining = self.remaining();
                if remaining == 0 {
                    return Err(GgufError::UnexpectedEof {
                        offset: self.pos,
                        wanted: (out.len() - done) as u64,
                        file_len: self.src_len,
                    });
                }
                let n = remaining.min(CHUNK as u64) as usize;
                self.buf.resize(n, 0);
                self.src.read_at(self.pos, &mut self.buf)?;
                self.buf_start = self.pos;
            }
        }
        Ok(())
    }

    fn read_arr<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let mut b = [0u8; N];
        self.read_exact(&mut b)?;
        Ok(b)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_arr::<1>()?[0])
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.read_arr()?))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.read_arr()?))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.read_arr()?))
    }

    /// Read a GGUF string (u64 length + UTF-8 bytes), capped at `max`.
    fn read_string(&mut self, max: u64) -> Result<String, GgufError> {
        let at = self.pos;
        let len = self.read_u64()?;
        if len > max {
            return Err(GgufError::StringTooLong {
                offset: at,
                len,
                max,
            });
        }
        let remaining = self.remaining();
        if len > remaining {
            return Err(GgufError::UnexpectedEof {
                offset: self.pos,
                wanted: len,
                file_len: self.src_len,
            });
        }
        // len <= MAX_STRING_LEN, so this allocation is bounded.
        let mut bytes = vec![0u8; len as usize];
        self.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8 { offset: at })
    }
}

/// Minimum encoded size in bytes of one value of metadata type `ty`. Used
/// to bound array element counts against the remaining file size before
/// allocating anything.
fn min_encoded_size(ty: u32) -> Option<u64> {
    Some(match ty {
        T_UINT8 | T_INT8 | T_BOOL => 1,
        T_UINT16 | T_INT16 => 2,
        T_UINT32 | T_INT32 | T_FLOAT32 => 4,
        T_UINT64 | T_INT64 | T_FLOAT64 => 8,
        T_STRING => 8, // u64 length prefix
        T_ARRAY => 12, // u32 element type + u64 count
        _ => return None,
    })
}

/// Parse one metadata value of type `ty`. `depth` counts enclosing arrays:
/// a top-level array parses its elements at depth 1; an array encountered
/// at depth >= 2 (i.e. nested more than one level) is rejected. `budget`
/// is the global remaining array-element allowance.
fn parse_value(
    cur: &mut Cursor,
    ty: u32,
    depth: u32,
    budget: &mut u64,
) -> Result<MetaValue, GgufError> {
    cur.check_header_cap()?;
    let at = cur.pos();
    Ok(match ty {
        T_UINT8 => MetaValue::U8(cur.read_u8()?),
        T_INT8 => MetaValue::I8(cur.read_u8()? as i8),
        T_UINT16 => MetaValue::U16(cur.read_u16()?),
        T_INT16 => MetaValue::I16(cur.read_u16()? as i16),
        T_UINT32 => MetaValue::U32(cur.read_u32()?),
        T_INT32 => MetaValue::I32(cur.read_u32()? as i32),
        T_FLOAT32 => MetaValue::F32(f32::from_le_bytes(cur.read_arr()?)),
        T_BOOL => match cur.read_u8()? {
            0 => MetaValue::Bool(false),
            1 => MetaValue::Bool(true),
            value => return Err(GgufError::InvalidBool { value, offset: at }),
        },
        T_STRING => MetaValue::String(cur.read_string(MAX_STRING_LEN)?),
        T_UINT64 => MetaValue::U64(cur.read_u64()?),
        T_INT64 => MetaValue::I64(cur.read_u64()? as i64),
        T_FLOAT64 => MetaValue::F64(f64::from_le_bytes(cur.read_arr()?)),
        T_ARRAY => {
            if depth >= 2 {
                return Err(GgufError::ArrayTooDeep { offset: at });
            }
            let elem_ty = cur.read_u32()?;
            let count = cur.read_u64()?;
            let min = min_encoded_size(elem_ty).ok_or(GgufError::UnknownValueType {
                ty: elem_ty,
                offset: at,
            })?;
            // Every element consumes at least `min` bytes, so a count the
            // remaining file cannot encode is rejected before allocation.
            let remaining = cur.remaining();
            if count > remaining / min {
                return Err(GgufError::ArrayTooLong {
                    count,
                    offset: at,
                    remaining,
                });
            }
            if count > *budget {
                return Err(GgufError::MetadataTooLarge {
                    limit: MAX_META_ELEMENTS,
                });
            }
            *budget -= count;
            // Capacity is clamped; the vec grows only as elements are
            // actually parsed out of the file.
            let mut items = Vec::with_capacity(count.min(4096) as usize);
            for _ in 0..count {
                items.push(parse_value(cur, elem_ty, depth + 1, budget)?);
            }
            MetaValue::Array(items)
        }
        _ => return Err(GgufError::UnknownValueType { ty, offset: at }),
    })
}

/// Exact byte size of a tensor from its dims and type block geometry.
fn tensor_byte_size(name: &str, dims: &[u64], ty: GgmlType) -> Result<u64, GgufError> {
    let block = ty.block_weights();
    if !dims[0].is_multiple_of(block) {
        return Err(GgufError::DimNotBlockAligned {
            name: name.to_owned(),
            dim0: dims[0],
            ty,
            block,
        });
    }
    let mut n_elems: u64 = 1;
    for &d in dims {
        n_elems = n_elems
            .checked_mul(d)
            .ok_or_else(|| GgufError::SizeOverflow {
                name: name.to_owned(),
            })?;
    }
    // Exact: dims[0] is block-divisible, so n_elems is too.
    let n_blocks = n_elems / block;
    n_blocks
        .checked_mul(ty.block_bytes())
        .ok_or_else(|| GgufError::SizeOverflow {
            name: name.to_owned(),
        })
}

/// Round `x` up to a multiple of the power-of-two `a`.
fn align_up(x: u64, a: u64) -> Option<u64> {
    debug_assert!(a.is_power_of_two());
    let rem = x % a;
    if rem == 0 {
        Some(x)
    } else {
        x.checked_add(a - rem)
    }
}

pub(crate) fn parse(src: &dyn RangeRead) -> Result<GgufFile, GgufError> {
    let mut cur = Cursor::new(src);

    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(GgufError::BadMagic { found: magic });
    }
    let version = cur.read_u32()?;
    // v2 and v3 share this layout for little-endian files; v3 only added
    // the possibility of big-endian encoding, which we reject (a big-endian
    // file's version field reads back byte-swapped and fails here).
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = cur.read_u64()?;
    if tensor_count > MAX_TENSORS {
        return Err(GgufError::TooManyTensors {
            count: tensor_count,
            limit: MAX_TENSORS,
        });
    }
    let kv_count = cur.read_u64()?;
    if kv_count > MAX_METADATA_KV {
        return Err(GgufError::TooManyMetadataKv {
            count: kv_count,
            limit: MAX_METADATA_KV,
        });
    }

    let mut metadata = BTreeMap::new();
    let mut budget = MAX_META_ELEMENTS;
    for _ in 0..kv_count {
        cur.check_header_cap()?;
        let key = cur.read_string(MAX_STRING_LEN)?;
        let ty = cur.read_u32()?;
        let value = parse_value(&mut cur, ty, 0, &mut budget)?;
        if metadata.contains_key(&key) {
            return Err(GgufError::DuplicateKey(key));
        }
        metadata.insert(key, value);
    }

    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(v) => {
            let a = v.as_uint().ok_or(GgufError::AlignmentNotUint)?;
            if !a.is_power_of_two() || a > MAX_ALIGNMENT {
                return Err(GgufError::BadAlignment(a));
            }
            a
        }
    };

    let mut tensors: Vec<TensorInfo> = Vec::with_capacity(tensor_count.min(1024) as usize);
    let mut by_name: BTreeMap<String, usize> = BTreeMap::new();
    for _ in 0..tensor_count {
        cur.check_header_cap()?;
        let name = cur.read_string(MAX_TENSOR_NAME_LEN)?;
        let n_dims = cur.read_u32()?;
        if n_dims == 0 || n_dims > MAX_DIMS {
            return Err(GgufError::BadDimCount { name, n_dims });
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for axis in 0..n_dims as usize {
            let d = cur.read_u64()?;
            if d == 0 {
                return Err(GgufError::ZeroDim { name, axis });
            }
            dims.push(d);
        }
        let ty_raw = cur.read_u32()?;
        let Some(ggml_type) = GgmlType::from_u32(ty_raw) else {
            return Err(GgufError::UnsupportedTensorType { name, ty: ty_raw });
        };
        let rel_offset = cur.read_u64()?;
        let size_bytes = tensor_byte_size(&name, &dims, ggml_type)?;
        if by_name.insert(name.clone(), tensors.len()).is_some() {
            return Err(GgufError::DuplicateTensor(name));
        }
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            rel_offset,
            size_bytes,
        });
    }
    cur.check_header_cap()?;

    // Data section: after the index, padded up to the alignment. pos is
    // capped at MAX_HEADER_BYTES + one entry, so align_up cannot overflow
    // in practice; map the impossible case to a typed error anyway.
    let data_start = align_up(cur.pos(), alignment).ok_or(GgufError::HeaderTooLarge {
        limit: MAX_HEADER_BYTES,
    })?;
    let data_len = src.len().saturating_sub(data_start);
    for t in &tensors {
        if t.rel_offset % alignment != 0 {
            return Err(GgufError::MisalignedTensor {
                name: t.name.clone(),
                rel_offset: t.rel_offset,
                alignment,
            });
        }
        let end =
            t.rel_offset
                .checked_add(t.size_bytes)
                .ok_or_else(|| GgufError::SizeOverflow {
                    name: t.name.clone(),
                })?;
        if end > data_len {
            return Err(GgufError::TensorOutOfBounds {
                name: t.name.clone(),
                rel_offset: t.rel_offset,
                size_bytes: t.size_bytes,
                data_len,
            });
        }
    }

    tracing::debug!(
        version,
        tensor_count,
        kv_count,
        alignment,
        data_start,
        data_len,
        "parsed GGUF header"
    );

    Ok(GgufFile {
        metadata,
        tensors,
        version,
        alignment,
        data_start,
        data_len,
        by_name,
    })
}
