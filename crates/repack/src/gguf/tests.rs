//! Unit tests: fixture round-trips, expert-slab math, and malformed-input
//! rejection. Panics (`unwrap`/`assert`) are fine here; the parser itself
//! must never panic.

use super::testutil::{
    FixtureBuilder, TensorSpec, header, pattern_byte, put_str, put_u8, put_u32, put_u64,
};
use super::{
    GgmlType, GgufError, GgufFile, MAX_HEADER_BYTES, MAX_META_ELEMENTS, MAX_METADATA_KV,
    MAX_STRING_LEN, MetaValue, TensorInfo,
};
use crate::source::{LocalFile, RangeRead, SourceError};

fn parse_bytes(bytes: &[u8]) -> Result<GgufFile, GgufError> {
    GgufFile::parse(&bytes)
}

/// (rel_offset, size_bytes) expected for `FixtureBuilder::moe()` with
/// alignment 32, in tensor order.
const MOE_LAYOUT: [(u64, u64); 8] = [
    (0, 36864),    // Q4_K [256,64,4]: 65536/256 blocks * 144 B
    (36864, 256),  // F32 [64]
    (37120, 6720), // Q6_K [256,8,4]: 8192/256 blocks * 210 B
    (43840, 256),  // F16 [32,4]
    (44096, 68),   // Q8_0 [32,2]: 2 blocks * 34 B
    (44192, 18),   // Q4_0 [32]: 1 block, aligned up from 44164
    (44224, 32),   // BF16 [16]
    (44256, 176),  // Q5_K [256]: 1 block
];

#[test]
fn parses_moe_fixture() {
    let bytes = FixtureBuilder::moe().build();
    let f = parse_bytes(&bytes).unwrap();
    assert_eq!(f.version(), 3);
    assert_eq!(f.alignment(), 32);
    assert_eq!(f.tensors().len(), 8);
    assert_eq!(
        f.metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str),
        Some("ramvamp-test")
    );

    let gate = f.tensor("blk.0.ffn_gate_exps.weight").unwrap();
    assert_eq!(gate.dims, vec![256, 64, 4]);
    assert_eq!(gate.ggml_type, GgmlType::Q4_K);
    assert!(f.tensor("no.such.tensor").is_none());
    assert_eq!(f.data_section_offset() % 32, 0);
    assert!(f.data_section_len() >= 44432);
}

#[test]
fn tensor_sizes_offsets_and_data_bytes() {
    let bytes = FixtureBuilder::moe().build();
    let f = parse_bytes(&bytes).unwrap();
    for (idx, t) in f.tensors().iter().enumerate() {
        let (rel, size) = MOE_LAYOUT[idx];
        assert_eq!(t.rel_offset, rel, "rel_offset of {}", t.name);
        assert_eq!(t.size_bytes, size, "size_bytes of {}", t.name);
        let off = f.data_offset(t);
        assert_eq!(off, f.data_section_offset() + rel);
        assert_eq!(
            off % f.alignment(),
            0,
            "data offset alignment of {}",
            t.name
        );
        let got = &bytes[off as usize..(off + size) as usize];
        let want: Vec<u8> = (0..size).map(|j| pattern_byte(idx, j)).collect();
        assert_eq!(got, &want[..], "data bytes of {}", t.name);
    }
}

#[test]
fn metadata_all_types_round_trip() {
    use MetaValue as MV;
    let bytes = FixtureBuilder::moe().build();
    let f = parse_bytes(&bytes).unwrap();
    let m = &f.metadata;
    assert_eq!(m.get("test.u8"), Some(&MV::U8(200)));
    assert_eq!(m.get("test.i8"), Some(&MV::I8(-5)));
    assert_eq!(m.get("test.u16"), Some(&MV::U16(65500)));
    assert_eq!(m.get("test.i16"), Some(&MV::I16(-1234)));
    assert_eq!(m.get("test.u32"), Some(&MV::U32(7_000_000)));
    assert_eq!(m.get("test.i32"), Some(&MV::I32(-7)));
    assert_eq!(m.get("test.f32"), Some(&MV::F32(1.5)));
    assert_eq!(m.get("test.bool"), Some(&MV::Bool(true)));
    assert_eq!(m.get("test.str"), Some(&MV::String("hello".into())));
    assert_eq!(m.get("test.u64"), Some(&MV::U64(1 << 40)));
    assert_eq!(m.get("test.i64"), Some(&MV::I64(-(1i64 << 40))));
    assert_eq!(m.get("test.f64"), Some(&MV::F64(-2.25)));
    assert_eq!(
        m.get("test.arr_str"),
        Some(&MV::Array(vec![
            MV::String("a".into()),
            MV::String("bb".into())
        ]))
    );
    assert_eq!(
        m.get("test.arr_nested"),
        Some(&MV::Array(vec![
            MV::Array(vec![MV::U32(1), MV::U32(2)]),
            MV::Array(vec![MV::U32(3)]),
        ]))
    );
    assert_eq!(
        m.get("general.alignment").and_then(MetaValue::as_uint),
        Some(32)
    );
}

#[test]
fn accepts_v2() {
    let mut b = FixtureBuilder::moe();
    b.version = 2;
    let f = parse_bytes(&b.build()).unwrap();
    assert_eq!(f.version(), 2);
    assert_eq!(f.tensors().len(), 8);
}

#[test]
fn default_alignment_when_kv_absent() {
    let mut b = FixtureBuilder::moe();
    b.emit_alignment_kv = false;
    let f = parse_bytes(&b.build()).unwrap();
    assert_eq!(f.alignment(), 32);
    assert!(!f.metadata.contains_key("general.alignment"));
}

#[test]
fn custom_alignment_respected() {
    let mut b = FixtureBuilder::moe();
    b.alignment = 64;
    let f = parse_bytes(&b.build()).unwrap();
    assert_eq!(f.alignment(), 64);
    for t in f.tensors() {
        assert_eq!(t.rel_offset % 64, 0);
        assert_eq!(f.data_offset(t) % 64, 0);
    }
}

#[test]
fn local_file_source_round_trip() {
    let bytes = FixtureBuilder::moe().build();
    let path = std::env::temp_dir().join(format!("ramvamp-gguf-test-{}.gguf", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let src = LocalFile::open(&path).unwrap();
    assert_eq!(RangeRead::len(&src), bytes.len() as u64);
    let f = GgufFile::parse(&src).unwrap();
    assert_eq!(f.tensors().len(), 8);
    let gate = f.tensor("blk.0.ffn_gate_exps.weight").unwrap();
    assert_eq!(gate.size_bytes, 36864);
    // Slab bytes read through the LocalFile match the in-memory fixture.
    let r = f.expert_slab(gate, 1).unwrap();
    let mut buf = vec![0u8; (r.end - r.start) as usize];
    src.read_at(r.start, &mut buf).unwrap();
    assert_eq!(&buf[..], &bytes[r.start as usize..r.end as usize]);
    std::fs::remove_file(&path).ok();
}

#[test]
fn expert_slabs_disjoint_contiguous_in_bounds_exact() {
    let bytes = FixtureBuilder::moe().build();
    let f = parse_bytes(&bytes).unwrap();
    for (idx, name, n_experts) in [
        (0usize, "blk.0.ffn_gate_exps.weight", 4u64),
        (2usize, "blk.0.ffn_down_exps.weight", 4u64),
    ] {
        let t = f.tensor(name).unwrap();
        let slab = t.size_bytes / n_experts;
        let mut prev_end = f.data_offset(t);
        for e in 0..n_experts {
            let r = f.expert_slab(t, e).unwrap();
            assert_eq!(r.start, prev_end, "expert {e} of {name} not contiguous");
            assert_eq!(r.end - r.start, slab);
            assert!(
                r.end <= bytes.len() as u64,
                "expert {e} of {name} out of bounds"
            );
            let got = &bytes[r.start as usize..r.end as usize];
            let want: Vec<u8> = (0..slab).map(|k| pattern_byte(idx, e * slab + k)).collect();
            assert_eq!(got, &want[..], "expert {e} of {name} bytes");
            prev_end = r.end;
        }
        assert_eq!(prev_end, f.data_offset(t) + t.size_bytes);
    }
    // Verified slab widths from the block tables.
    let gate = f.tensor("blk.0.ffn_gate_exps.weight").unwrap();
    let r = f.expert_slab(gate, 0).unwrap();
    assert_eq!(r.end - r.start, 9216); // 64 rows * 144 B
    let down = f.tensor("blk.0.ffn_down_exps.weight").unwrap();
    let r = f.expert_slab(down, 0).unwrap();
    assert_eq!(r.end - r.start, 1680); // 8 rows * 210 B
}

#[test]
fn expert_slab_rejects_non_3d() {
    let bytes = FixtureBuilder::moe().build();
    let f = parse_bytes(&bytes).unwrap();
    let norm = f.tensor("blk.0.attn_norm.weight").unwrap();
    let err = f.expert_slab(norm, 0).unwrap_err();
    assert!(matches!(err, GgufError::NotExpertTensor { n_dims: 1, .. }));
}

#[test]
fn expert_slab_rejects_out_of_range_index() {
    let bytes = FixtureBuilder::moe().build();
    let f = parse_bytes(&bytes).unwrap();
    let gate = f.tensor("blk.0.ffn_gate_exps.weight").unwrap();
    let err = f.expert_slab(gate, 4).unwrap_err();
    assert!(matches!(
        err,
        GgufError::ExpertOutOfRange {
            expert_idx: 4,
            n_experts: 4,
            ..
        }
    ));
}

#[test]
fn expert_slab_rejects_non_divisible_size() {
    let bytes = FixtureBuilder::moe().build();
    let f = parse_bytes(&bytes).unwrap();
    // Hand-built info (fields are public): 100 B over 3 experts.
    let bogus = TensorInfo {
        name: "bogus".into(),
        dims: vec![4, 4, 3],
        ggml_type: GgmlType::F32,
        rel_offset: 0,
        size_bytes: 100,
    };
    let err = f.expert_slab(&bogus, 0).unwrap_err();
    assert!(matches!(
        err,
        GgufError::SlabNotDivisible { n_experts: 3, .. }
    ));
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = FixtureBuilder::moe().build();
    bytes[0] = b'X';
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(err, GgufError::BadMagic { .. }));
}

#[test]
fn rejects_unsupported_versions() {
    for version in [1u32, 42, 0x0300_0000 /* big-endian v3 */] {
        let mut bytes = FixtureBuilder::moe().build();
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        let err = parse_bytes(&bytes).unwrap_err();
        assert!(
            matches!(err, GgufError::UnsupportedVersion(v) if v == version),
            "version {version}"
        );
    }
}

#[test]
fn rejects_truncated_header() {
    let bytes = FixtureBuilder::moe().build();
    // Inside the fixed header.
    let err = parse_bytes(&bytes[..12]).unwrap_err();
    assert!(matches!(err, GgufError::UnexpectedEof { .. }));
    // Inside the first metadata key.
    let err = parse_bytes(&bytes[..30]).unwrap_err();
    assert!(matches!(err, GgufError::UnexpectedEof { .. }));
}

#[test]
fn rejects_empty_input() {
    let err = parse_bytes(&[]).unwrap_err();
    assert!(matches!(err, GgufError::UnexpectedEof { .. }));
}

#[test]
fn rejects_absurd_tensor_count() {
    let mut bytes = FixtureBuilder::moe().build();
    bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(
        err,
        GgufError::TooManyTensors {
            count: u64::MAX,
            ..
        }
    ));
}

#[test]
fn rejects_absurd_metadata_kv_count() {
    let mut bytes = FixtureBuilder::moe().build();
    bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(
        err,
        GgufError::TooManyMetadataKv {
            count: u64::MAX,
            ..
        }
    ));
}

#[test]
fn rejects_misaligned_tensor_offset() {
    let mut b = FixtureBuilder::default();
    let mut t = TensorSpec::new("t", &[32], GgmlType::F32);
    t.offset_override = Some(1);
    b.tensors.push(t);
    let err = parse_bytes(&b.build()).unwrap_err();
    assert!(matches!(
        err,
        GgufError::MisalignedTensor {
            rel_offset: 1,
            alignment: 32,
            ..
        }
    ));
}

#[test]
fn rejects_dims_not_block_divisible() {
    let mut b = FixtureBuilder::default();
    b.tensors
        .push(TensorSpec::new("t", &[100, 4], GgmlType::Q4_K));
    let err = parse_bytes(&b.build()).unwrap_err();
    assert!(matches!(
        err,
        GgufError::DimNotBlockAligned {
            dim0: 100,
            block: 256,
            ..
        }
    ));
}

#[test]
fn rejects_truncated_data() {
    let mut b = FixtureBuilder::moe();
    b.truncate_tail = 10;
    let err = parse_bytes(&b.build()).unwrap_err();
    match err {
        GgufError::TensorOutOfBounds { name, .. } => assert_eq!(name, "blk.0.q5.weight"),
        other => panic!("expected TensorOutOfBounds, got {other:?}"),
    }
}

#[test]
fn rejects_string_too_long() {
    let mut bytes = header(3, 0, 1);
    put_str(&mut bytes, "k");
    put_u32(&mut bytes, 8); // string value
    put_u64(&mut bytes, 2 << 20); // claims 2 MiB
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(err, GgufError::StringTooLong { len, .. } if len == 2 << 20));
}

#[test]
fn rejects_array_nested_too_deep() {
    let mut bytes = header(3, 0, 1);
    put_str(&mut bytes, "k");
    put_u32(&mut bytes, 9); // array
    put_u32(&mut bytes, 9); // of arrays
    put_u64(&mut bytes, 1);
    put_u32(&mut bytes, 9); // of arrays -- two levels down, rejected
    put_u64(&mut bytes, 1);
    put_u64(&mut bytes, 0); // filler so bounds checks pass first
    put_u32(&mut bytes, 0);
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(err, GgufError::ArrayTooDeep { .. }));
}

#[test]
fn rejects_absurd_array_count() {
    let mut bytes = header(3, 0, 1);
    put_str(&mut bytes, "k");
    put_u32(&mut bytes, 9); // array
    put_u32(&mut bytes, 0); // of u8
    put_u64(&mut bytes, u64::MAX);
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(
        err,
        GgufError::ArrayTooLong {
            count: u64::MAX,
            ..
        }
    ));
}

#[test]
fn rejects_bad_bool() {
    let mut bytes = header(3, 0, 1);
    put_str(&mut bytes, "k");
    put_u32(&mut bytes, 7); // bool
    put_u8(&mut bytes, 2);
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(err, GgufError::InvalidBool { value: 2, .. }));
}

#[test]
fn rejects_unknown_value_type() {
    let mut bytes = header(3, 0, 1);
    put_str(&mut bytes, "k");
    put_u32(&mut bytes, 99);
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(err, GgufError::UnknownValueType { ty: 99, .. }));
}

#[test]
fn rejects_unknown_tensor_type() {
    let mut b = FixtureBuilder::default();
    let mut t = TensorSpec::new("t", &[4], GgmlType::F32);
    t.ty_override = Some(26); // GGML_TYPE_I32: real upstream id we don't support
    b.tensors.push(t);
    let err = parse_bytes(&b.build()).unwrap_err();
    assert!(matches!(
        err,
        GgufError::UnsupportedTensorType { ty: 26, .. }
    ));
}

#[test]
fn rejects_duplicate_tensor_names() {
    let mut b = FixtureBuilder::default();
    b.tensors.push(TensorSpec::new("dup", &[32], GgmlType::F32));
    b.tensors.push(TensorSpec::new("dup", &[32], GgmlType::F32));
    let err = parse_bytes(&b.build()).unwrap_err();
    assert!(matches!(err, GgufError::DuplicateTensor(name) if name == "dup"));
}

#[test]
fn rejects_duplicate_metadata_keys() {
    let mut bytes = header(3, 0, 2);
    put_str(&mut bytes, "dup");
    put_u32(&mut bytes, 0);
    put_u8(&mut bytes, 1);
    put_str(&mut bytes, "dup");
    put_u32(&mut bytes, 0);
    put_u8(&mut bytes, 2);
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(err, GgufError::DuplicateKey(k) if k == "dup"));
}

#[test]
fn rejects_bad_alignment() {
    for bad in [31u64, 0] {
        let b = FixtureBuilder {
            alignment: bad,
            ..FixtureBuilder::default()
        };
        let err = parse_bytes(&b.build()).unwrap_err();
        assert!(
            matches!(err, GgufError::BadAlignment(a) if a == bad),
            "alignment {bad}"
        );
    }
}

#[test]
fn rejects_alignment_of_wrong_type() {
    let mut bytes = header(3, 0, 1);
    put_str(&mut bytes, "general.alignment");
    put_u32(&mut bytes, 8); // string
    put_str(&mut bytes, "x");
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(err, GgufError::AlignmentNotUint));
}

#[test]
fn rejects_zero_dim() {
    let mut b = FixtureBuilder::default();
    b.tensors.push(TensorSpec::new("t", &[0], GgmlType::F32));
    let err = parse_bytes(&b.build()).unwrap_err();
    assert!(matches!(err, GgufError::ZeroDim { axis: 0, .. }));
}

#[test]
fn rejects_bad_dim_count() {
    for n_dims in [0u32, 5] {
        let mut bytes = header(3, 1, 0);
        put_str(&mut bytes, "t");
        put_u32(&mut bytes, n_dims);
        let err = parse_bytes(&bytes).unwrap_err();
        assert!(
            matches!(err, GgufError::BadDimCount { n_dims: n, .. } if n == n_dims),
            "n_dims {n_dims}"
        );
    }
}

#[test]
fn rejects_size_overflow() {
    let mut bytes = header(3, 1, 0);
    put_str(&mut bytes, "t");
    put_u32(&mut bytes, 3);
    put_u64(&mut bytes, 1 << 32);
    put_u64(&mut bytes, 1 << 32);
    put_u64(&mut bytes, 256);
    put_u32(&mut bytes, 0); // F32
    put_u64(&mut bytes, 0);
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(err, GgufError::SizeOverflow { .. }));
}

#[test]
fn rejects_metadata_element_budget_on_declared_count() {
    // One array declaring just over the global element budget, plus one
    // filler byte per declared element so the per-array remaining-bytes
    // check passes and the global budget is what rejects. The parser must
    // fail on the declared count alone, before materializing any element.
    let count = MAX_META_ELEMENTS + 1;
    let mut bytes = header(3, 0, 1);
    put_str(&mut bytes, "k");
    put_u32(&mut bytes, 9); // array
    put_u32(&mut bytes, 0); // of u8
    put_u64(&mut bytes, count);
    bytes.resize(bytes.len() + count as usize, 0);
    let err = parse_bytes(&bytes).unwrap_err();
    assert!(matches!(
        err,
        GgufError::MetadataTooLarge { limit } if limit == MAX_META_ELEMENTS
    ));
}

/// Encoded size of one [`EndlessMetadataSource`] record: key length + 8-byte
/// key + value type + value length + `MAX_STRING_LEN` value bytes.
const RECORD_LEN: u64 = 8 + 8 + 4 + 8 + MAX_STRING_LEN;

/// A GGUF byte stream synthesized on read: a real header declaring
/// `MAX_METADATA_KV` key-values, then an endless run of valid metadata
/// records (unique 8-byte key, `MAX_STRING_LEN` string value). The declared
/// length is ~100 GiB but the test materializes nothing; every field passes
/// its own per-item check, so only the running header-region cap can stop
/// the stream.
struct EndlessMetadataSource;

impl EndlessMetadataSource {
    /// 28-byte encoded prefix of record `r`: key `k<r:07>` + string value
    /// type + value length.
    fn record_prefix(r: u64) -> [u8; 28] {
        let mut p = [0u8; 28];
        p[..8].copy_from_slice(&8u64.to_le_bytes());
        p[8..16].copy_from_slice(format!("k{r:07}").as_bytes());
        p[16..20].copy_from_slice(&8u32.to_le_bytes());
        p[20..28].copy_from_slice(&MAX_STRING_LEN.to_le_bytes());
        p
    }
}

impl RangeRead for EndlessMetadataSource {
    fn len(&self) -> u64 {
        24 + MAX_METADATA_KV * RECORD_LEN
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), SourceError> {
        let want = buf.len() as u64;
        if offset.checked_add(want).is_none_or(|end| end > self.len()) {
            return Err(SourceError::OutOfBounds {
                offset,
                len: want,
                source_len: self.len(),
            });
        }
        let head = header(3, 0, MAX_METADATA_KV);
        let base = head.len() as u64;
        let mut pos = offset;
        let mut done = 0usize;
        while done < buf.len() {
            let n = if pos < base {
                let take = ((base - pos) as usize).min(buf.len() - done);
                buf[done..done + take].copy_from_slice(&head[pos as usize..pos as usize + take]);
                take
            } else {
                let rec_off = (pos - base) % RECORD_LEN;
                if rec_off < 28 {
                    let prefix = Self::record_prefix((pos - base) / RECORD_LEN);
                    let take = ((28 - rec_off) as usize).min(buf.len() - done);
                    buf[done..done + take]
                        .copy_from_slice(&prefix[rec_off as usize..rec_off as usize + take]);
                    take
                } else {
                    let take = ((RECORD_LEN - rec_off) as usize).min(buf.len() - done);
                    buf[done..done + take].fill(b'x');
                    take
                }
            };
            pos += n as u64;
            done += n;
        }
        Ok(())
    }
}

#[test]
fn rejects_header_region_over_cap() {
    // The declared header region (`MAX_METADATA_KV` records of ~1 MiB) is
    // far beyond the cap while every individual count stays within its own
    // limit. The cap must trip just past MAX_HEADER_BYTES instead of
    // streaming the whole declaration.
    let err = GgufFile::parse(&EndlessMetadataSource).unwrap_err();
    assert!(matches!(
        err,
        GgufError::HeaderTooLarge { limit } if limit == MAX_HEADER_BYTES
    ));
}
