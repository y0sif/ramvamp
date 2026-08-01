//! Test-only builders that emit GGUF bytes — valid fixtures and
//! deliberately broken ones. Never compiled into the crate proper.

use crate::gguf::types::GgmlType;

pub(crate) fn put_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

pub(crate) fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// GGUF string: u64 length + UTF-8 bytes.
pub(crate) fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

/// Bare header: magic + version + tensor count + kv count.
pub(crate) fn header(version: u32, tensor_count: u64, kv_count: u64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    put_u32(&mut out, version);
    put_u64(&mut out, tensor_count);
    put_u64(&mut out, kv_count);
    out
}

/// Deterministic patterned data: byte `j` of tensor number `idx`.
pub(crate) fn pattern_byte(idx: usize, j: u64) -> u8 {
    ((idx as u64)
        .wrapping_mul(131)
        .wrapping_add(j.wrapping_mul(7))
        .wrapping_add(13)
        % 251) as u8
}

pub(crate) struct TensorSpec {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Raw type id written to the index; defaults to `ty as u32`.
    pub ty_override: Option<u32>,
    /// Rel offset written to the index; defaults to sequential aligned
    /// placement. Data is always laid out at the computed placement.
    pub offset_override: Option<u64>,
}

impl TensorSpec {
    pub fn new(name: &str, dims: &[u64], ty: GgmlType) -> Self {
        TensorSpec {
            name: name.to_owned(),
            dims: dims.to_vec(),
            ty,
            ty_override: None,
            offset_override: None,
        }
    }

    /// Size used for layout. Matches the parser's math for valid specs;
    /// saturates instead of panicking so broken specs still build.
    fn layout_size(&self) -> u64 {
        let mut n: u64 = 1;
        for &d in &self.dims {
            n = n.saturating_mul(d);
        }
        n.div_ceil(self.ty.block_weights())
            .saturating_mul(self.ty.block_bytes())
    }
}

pub(crate) struct FixtureBuilder {
    pub version: u32,
    pub alignment: u64,
    pub emit_alignment_kv: bool,
    pub all_types_metadata: bool,
    pub tensors: Vec<TensorSpec>,
    /// Drop this many bytes from the end of the built file.
    pub truncate_tail: u64,
}

impl Default for FixtureBuilder {
    fn default() -> Self {
        FixtureBuilder {
            version: 3,
            alignment: 32,
            emit_alignment_kv: true,
            all_types_metadata: false,
            tensors: Vec::new(),
            truncate_tail: 0,
        }
    }
}

impl FixtureBuilder {
    /// The standard tiny-MoE fixture: a Q4_K gate-experts tensor, an F32
    /// norm, a Q6_K down-experts tensor, and one tensor of every other
    /// supported storage type.
    pub fn moe() -> Self {
        FixtureBuilder {
            all_types_metadata: true,
            tensors: vec![
                TensorSpec::new("blk.0.ffn_gate_exps.weight", &[256, 64, 4], GgmlType::Q4_K),
                TensorSpec::new("blk.0.attn_norm.weight", &[64], GgmlType::F32),
                TensorSpec::new("blk.0.ffn_down_exps.weight", &[256, 8, 4], GgmlType::Q6_K),
                TensorSpec::new("blk.0.attn_q.weight", &[32, 4], GgmlType::F16),
                TensorSpec::new("blk.0.q8.weight", &[32, 2], GgmlType::Q8_0),
                TensorSpec::new("blk.0.q4legacy.weight", &[32], GgmlType::Q4_0),
                TensorSpec::new("blk.0.bf.weight", &[16], GgmlType::BF16),
                TensorSpec::new("blk.0.q5.weight", &[256], GgmlType::Q5_K),
            ],
            ..FixtureBuilder::default()
        }
    }

    pub fn build(&self) -> Vec<u8> {
        let mut out = header(self.version, self.tensors.len() as u64, 0);
        let kv_count_at = out.len() - 8;

        // Metadata.
        let mut kv_count: u64 = 1;
        put_str(&mut out, "general.architecture");
        put_u32(&mut out, 8);
        put_str(&mut out, "ramvamp-test");
        if self.emit_alignment_kv {
            put_str(&mut out, "general.alignment");
            put_u32(&mut out, 4);
            put_u32(&mut out, self.alignment as u32);
            kv_count += 1;
        }
        if self.all_types_metadata {
            put_str(&mut out, "test.u8");
            put_u32(&mut out, 0);
            put_u8(&mut out, 200);
            put_str(&mut out, "test.i8");
            put_u32(&mut out, 1);
            put_u8(&mut out, (-5i8) as u8);
            put_str(&mut out, "test.u16");
            put_u32(&mut out, 2);
            out.extend_from_slice(&65500u16.to_le_bytes());
            put_str(&mut out, "test.i16");
            put_u32(&mut out, 3);
            out.extend_from_slice(&(-1234i16).to_le_bytes());
            put_str(&mut out, "test.u32");
            put_u32(&mut out, 4);
            put_u32(&mut out, 7_000_000);
            put_str(&mut out, "test.i32");
            put_u32(&mut out, 5);
            out.extend_from_slice(&(-7i32).to_le_bytes());
            put_str(&mut out, "test.f32");
            put_u32(&mut out, 6);
            out.extend_from_slice(&1.5f32.to_le_bytes());
            put_str(&mut out, "test.bool");
            put_u32(&mut out, 7);
            put_u8(&mut out, 1);
            put_str(&mut out, "test.str");
            put_u32(&mut out, 8);
            put_str(&mut out, "hello");
            put_str(&mut out, "test.u64");
            put_u32(&mut out, 10);
            put_u64(&mut out, 1 << 40);
            put_str(&mut out, "test.i64");
            put_u32(&mut out, 11);
            out.extend_from_slice(&(-(1i64 << 40)).to_le_bytes());
            put_str(&mut out, "test.f64");
            put_u32(&mut out, 12);
            out.extend_from_slice(&(-2.25f64).to_le_bytes());
            // Array of strings.
            put_str(&mut out, "test.arr_str");
            put_u32(&mut out, 9);
            put_u32(&mut out, 8);
            put_u64(&mut out, 2);
            put_str(&mut out, "a");
            put_str(&mut out, "bb");
            // Array of arrays of u32 (one level of nesting).
            put_str(&mut out, "test.arr_nested");
            put_u32(&mut out, 9);
            put_u32(&mut out, 9);
            put_u64(&mut out, 2);
            put_u32(&mut out, 4);
            put_u64(&mut out, 2);
            put_u32(&mut out, 1);
            put_u32(&mut out, 2);
            put_u32(&mut out, 4);
            put_u64(&mut out, 1);
            put_u32(&mut out, 3);
            kv_count += 14;
        }
        out[kv_count_at..kv_count_at + 8].copy_from_slice(&kv_count.to_le_bytes());

        // Tensor index; data laid out sequentially, aligned.
        let align = self.alignment.max(1);
        let mut cursor: u64 = 0;
        let mut placements: Vec<(u64, u64)> = Vec::new();
        for t in &self.tensors {
            let size = t.layout_size();
            let rem = cursor % align;
            let place = if rem == 0 {
                cursor
            } else {
                cursor + (align - rem)
            };
            put_str(&mut out, &t.name);
            put_u32(&mut out, t.dims.len() as u32);
            for &d in &t.dims {
                put_u64(&mut out, d);
            }
            put_u32(&mut out, t.ty_override.unwrap_or(t.ty as u32));
            put_u64(&mut out, t.offset_override.unwrap_or(place));
            placements.push((place, size));
            cursor = place + size;
        }

        // Pad to the data section, then patterned data with zero-fill gaps.
        let rem = (out.len() as u64) % align;
        if rem != 0 {
            out.resize(out.len() + (align - rem) as usize, 0);
        }
        let data_start = out.len() as u64;
        for (idx, &(place, size)) in placements.iter().enumerate() {
            let abs = (data_start + place) as usize;
            if out.len() < abs {
                out.resize(abs, 0);
            }
            for j in 0..size {
                out.push(pattern_byte(idx, j));
            }
        }

        if self.truncate_tail > 0 {
            let new_len = out.len().saturating_sub(self.truncate_tail as usize);
            out.truncate(new_len);
        }
        out
    }
}
