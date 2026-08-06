//! Shared synthetic `.rvmp` install for io/ and model/ tests.
//!
//! Builds a tiny but fully valid install with the format module's public
//! API. [`build_install`] builds [`NARROW`]: 2 MoE layers, 4 experts,
//! hidden 256 (k-quant rows need a multiple of 256 weights, so "small"
//! bottoms out at one super-block per row), 4 heads of 64, GQA 4:2,
//! vocab 32. Layer 0 mirrors a Q6_K-down layer (`attn_v`/`ffn_down_exps`
//! Q6_K), layer 1 is pure Q4_K — the same per-tensor type mix as the
//! audited Qwen3 pin, shrunk.
//!
//! [`build_install_with`] builds any other [`Geometry`], which is what
//! `model/prefill.rs`'s wide bit-identity fixture is: the same install
//! shape at widths chosen so no two of the prefill driver's dimensions
//! can accidentally agree.
//!
//! Every quantized tensor is filled with deterministic pattern bytes whose
//! per-block f16 scales are stamped with real values, so dequantizing any
//! row yields finite floats. F32 tensors hold the geometry's payload
//! values — [`f32_pattern`] for [`NARROW`] — that tests can recompute.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use crate::format::{
    ArchInfo, CommonTensor, ExpertsLayout, FileEntry, LAYOUT_FILE, LayerLayout, Manifest,
    Projection, ProjectionName, QuantInfo, RVMP_VERSION, SourceInfo, layer_file_name, sha256_file,
    testutil::TempDir, write_layout, write_manifest,
};
use crate::kernels::quants::{QuantFormat, f32_to_f16};

/// Layers in the fixture.
pub(crate) const N_LAYERS: u32 = 2;
/// Experts per layer.
pub(crate) const N_EXPERTS: u32 = 4;
/// Hidden dimension.
pub(crate) const HIDDEN: usize = 256;
/// Per-expert FFN intermediate dimension.
pub(crate) const MOE_INTERMEDIATE: usize = 256;
/// Query heads.
pub(crate) const N_HEADS: usize = 4;
/// KV heads.
pub(crate) const N_KV_HEADS: usize = 2;
/// Head dimension.
pub(crate) const HEAD_DIM: usize = 64;
/// Vocabulary size.
pub(crate) const VOCAB: usize = 32;

/// Everything that distinguishes one fixture install from another:
/// the widths, the expert counts, the manifest's identity strings, and
/// the payloads its f32 tensors carry.
///
/// The install *shape* — the tensor name set, the per-tensor quant type
/// mix, the 64-byte common alignment, the 4096-byte expert slab
/// alignment — is not here, because it is the thing every fixture shares
/// and [`build_install_with`] is its single source.
pub(crate) struct Geometry {
    /// Layers in the fixture.
    pub(crate) n_layers: u32,
    /// Experts per layer.
    pub(crate) n_experts: u32,
    /// Experts per token.
    pub(crate) top_k: u32,
    /// Hidden dimension.
    pub(crate) hidden: usize,
    /// Per-expert FFN intermediate dimension.
    pub(crate) moe_intermediate: usize,
    /// Query heads.
    pub(crate) n_heads: usize,
    /// KV heads.
    pub(crate) n_kv_heads: usize,
    /// Head dimension.
    pub(crate) head_dim: usize,
    /// Vocabulary size.
    pub(crate) vocab: usize,
    /// `model_id` written to the manifest.
    pub(crate) model_id: &'static str,
    /// `source.hf_repo` written to the manifest.
    pub(crate) hf_repo: &'static str,
    /// `source.file` written to the manifest.
    pub(crate) source_file: &'static str,
    /// Payload of an f32 norm tensor, by tensor name and element index.
    pub(crate) norm_value: fn(&str, usize) -> f32,
    /// Payload of a router (`ffn_gate_inp`) tensor, same arguments.
    pub(crate) router_value: fn(&str, usize) -> f32,
}

impl Geometry {
    /// `n_heads * head_dim`.
    fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// `n_kv_heads * head_dim`.
    fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// The architecture facts this geometry describes.
    pub(crate) fn arch(&self) -> ArchInfo {
        ArchInfo {
            n_layers: self.n_layers,
            n_experts: self.n_experts,
            top_k: self.top_k,
            hidden: self.hidden as u32,
            moe_intermediate: self.moe_intermediate as u32,
            n_heads: self.n_heads as u32,
            n_kv_heads: self.n_kv_heads as u32,
            head_dim: self.head_dim as u32,
            vocab: self.vocab as u32,
            context_length: 4096,
            rope_theta: 1e7,
            rms_eps: 1e-6,
            norm_topk_prob: true,
            tie_embeddings: false,
            shared_expert: false,
            sliding_window: None,
        }
    }
}

/// The geometry [`build_install`] builds, and the one every io/ test and
/// most model/ tests run against.
pub(crate) const NARROW: Geometry = Geometry {
    n_layers: N_LAYERS,
    n_experts: N_EXPERTS,
    top_k: 2,
    hidden: HIDDEN,
    moe_intermediate: MOE_INTERMEDIATE,
    n_heads: N_HEADS,
    n_kv_heads: N_KV_HEADS,
    head_dim: HEAD_DIM,
    vocab: VOCAB,
    model_id: "fixture-moe-2l",
    hf_repo: "test/fixture",
    source_file: "fixture-Q4_K_M.gguf",
    norm_value: f32_pattern,
    router_value: f32_pattern,
};

/// A complete install in a self-cleaning temp dir.
pub(crate) struct Fixture {
    _tmp: TempDir,
    /// The install directory.
    pub(crate) root: PathBuf,
    /// The manifest as written.
    pub(crate) manifest: Manifest,
    /// The layout as written.
    pub(crate) layout: ExpertsLayout,
}

/// Deterministic f32 payload for tensor `name` at element `index`.
pub(crate) fn f32_pattern(name: &str, index: usize) -> f32 {
    (index as f32) * 0.25 - (name.len() as f32) * 0.5
}

/// Deterministic filler byte for quantized payloads.
fn pattern_byte(seed: usize, block: usize, byte: usize) -> u8 {
    (seed.wrapping_mul(31) ^ block.wrapping_mul(7) ^ byte.wrapping_mul(3)) as u8
}

/// Packed bytes for `rows` quantized rows of `in_dim` weights: pattern
/// bytes with valid f16 scales stamped at each block's scale offsets.
fn quant_bytes(format: QuantFormat, rows: usize, in_dim: usize, seed: usize) -> Vec<u8> {
    let row_bytes = format.row_bytes(in_dim).expect("fixture dims divide");
    let block_bytes = format.block_bytes();
    let n_blocks = rows * row_bytes / block_bytes;
    let mut out = vec![0u8; rows * row_bytes];
    for block in 0..n_blocks {
        let base = block * block_bytes;
        for byte in 0..block_bytes {
            out[base + byte] = pattern_byte(seed, block, byte);
        }
        let d = f32_to_f16(0.5).to_le_bytes();
        let dmin = f32_to_f16(0.25).to_le_bytes();
        match format {
            // d/dmin lead the block (blocks.rs q4k/q5k offsets 0 and 2).
            QuantFormat::Q4_K | QuantFormat::Q5_K => {
                out[base..base + 2].copy_from_slice(&d);
                out[base + 2..base + 4].copy_from_slice(&dmin);
            }
            // d trails ql/qh/scales at offset 208 (blocks.rs q6k).
            QuantFormat::Q6_K => out[base + 208..base + 210].copy_from_slice(&d),
            // d leads the 32-weight block (blocks.rs q8_0).
            QuantFormat::Q8_0 => out[base..base + 2].copy_from_slice(&d),
            QuantFormat::Q8_K => unreachable!("activation-only format"),
        }
    }
    out
}

/// LE bytes of `elements` values of `payload` for tensor `name`.
fn f32_bytes(name: &str, elements: usize, payload: fn(&str, usize) -> f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(elements * 4);
    for i in 0..elements {
        out.extend_from_slice(&payload(name, i).to_le_bytes());
    }
    out
}

/// The narrow fixture's architecture facts.
pub(crate) fn fixture_arch() -> ArchInfo {
    NARROW.arch()
}

/// One common tensor's dtype and generated payload.
struct TensorDef {
    dtype: &'static str,
    bytes: Vec<u8>,
}

/// Every common tensor of the fixture, keyed by GGUF name — the exact name
/// set the repack planner produces for qwen3moe.
fn common_defs(g: &Geometry) -> BTreeMap<String, TensorDef> {
    let mut defs = BTreeMap::new();
    let mut quant = |name: String, format: QuantFormat, rows: usize, in_dim: usize| {
        let dtype = match format {
            QuantFormat::Q4_K => "q4_k",
            QuantFormat::Q5_K => "q5_k",
            QuantFormat::Q6_K => "q6_k",
            QuantFormat::Q8_0 => "q8_0",
            QuantFormat::Q8_K => unreachable!("activation-only format"),
        };
        let bytes = quant_bytes(format, rows, in_dim, name.len() + rows);
        defs.insert(name, TensorDef { dtype, bytes });
    };
    quant(
        "token_embd.weight".to_owned(),
        QuantFormat::Q4_K,
        g.vocab,
        g.hidden,
    );
    quant(
        "output.weight".to_owned(),
        QuantFormat::Q6_K,
        g.vocab,
        g.hidden,
    );
    let q_dim = g.q_dim();
    let kv_dim = g.kv_dim();
    for layer in 0..g.n_layers {
        quant(
            format!("blk.{layer}.attn_q.weight"),
            QuantFormat::Q4_K,
            q_dim,
            g.hidden,
        );
        quant(
            format!("blk.{layer}.attn_k.weight"),
            QuantFormat::Q8_0,
            kv_dim,
            g.hidden,
        );
        // Layer 0 mirrors the audited Q6_K-down layers; layer 1 is Q4_K.
        let v_format = if layer == 0 {
            QuantFormat::Q6_K
        } else {
            QuantFormat::Q4_K
        };
        quant(
            format!("blk.{layer}.attn_v.weight"),
            v_format,
            kv_dim,
            g.hidden,
        );
        quant(
            format!("blk.{layer}.attn_output.weight"),
            QuantFormat::Q5_K,
            g.hidden,
            q_dim,
        );
    }
    let mut f32_def = |name: String, elements: usize, payload: fn(&str, usize) -> f32| {
        let bytes = f32_bytes(&name, elements, payload);
        defs.insert(
            name,
            TensorDef {
                dtype: "f32",
                bytes,
            },
        );
    };
    f32_def("output_norm.weight".to_owned(), g.hidden, g.norm_value);
    for layer in 0..g.n_layers {
        for (stem, elements) in [
            ("attn_norm", g.hidden),
            ("ffn_norm", g.hidden),
            ("attn_q_norm", g.head_dim),
            ("attn_k_norm", g.head_dim),
        ] {
            f32_def(format!("blk.{layer}.{stem}.weight"), elements, g.norm_value);
        }
        f32_def(
            format!("blk.{layer}.ffn_gate_inp.weight"),
            g.n_experts as usize * g.hidden,
            g.router_value,
        );
    }
    defs
}

/// Build a complete, hash-consistent install of [`NARROW`] under a fresh
/// temp dir.
pub(crate) fn build_install(tag: &str) -> Fixture {
    build_install_with(tag, &NARROW)
}

/// [`build_install`] at an arbitrary [`Geometry`].
///
/// The returned [`Fixture`] owns the `TempDir`, so the install lives
/// exactly as long as the caller holds it.
///
/// The manifest's digests are taken from the bytes as written here; a
/// caller that re-stamps block scales afterwards (`temper_install`) must
/// load with `skip_hashes`.
pub(crate) fn build_install_with(tag: &str, g: &Geometry) -> Fixture {
    let tmp = TempDir::new(tag);
    let root = tmp.path().join("model.rvmp");
    fs::create_dir_all(root.join("experts")).expect("create install dirs");

    // common.bin: name order, offsets aligned to 64 (mirrors the planner).
    let defs = common_defs(g);
    let mut common_tensors = BTreeMap::new();
    let mut common = Vec::new();
    for (name, def) in &defs {
        let offset = (common.len() as u64).next_multiple_of(64);
        common.resize(offset as usize, 0);
        common.extend_from_slice(&def.bytes);
        common_tensors.insert(
            name.clone(),
            CommonTensor {
                offset,
                len: def.bytes.len() as u64,
                dtype: def.dtype.to_owned(),
            },
        );
    }
    fs::write(root.join("common.bin"), &common).expect("write common.bin");

    // Expert layers: gate/up Q4_K everywhere, down Q6_K on layer 0 only.
    let mut layers = Vec::new();
    for layer in 0..g.n_layers {
        let down_format = if layer == 0 {
            QuantFormat::Q6_K
        } else {
            QuantFormat::Q4_K
        };
        let slabs: [(ProjectionName, QuantFormat, usize, usize); 3] = [
            (
                ProjectionName::Gate,
                QuantFormat::Q4_K,
                g.moe_intermediate,
                g.hidden,
            ),
            (
                ProjectionName::Up,
                QuantFormat::Q4_K,
                g.moe_intermediate,
                g.hidden,
            ),
            (
                ProjectionName::Down,
                down_format,
                g.hidden,
                g.moe_intermediate,
            ),
        ];
        let mut projections = Vec::new();
        let mut cursor: u64 = 0;
        for (name, format, rows, in_dim) in slabs {
            let offset_in_blob = cursor.next_multiple_of(4096);
            let len = (format.row_bytes(in_dim).unwrap() * rows) as u64;
            projections.push(Projection {
                name,
                offset_in_blob,
                len,
                quant: match format {
                    QuantFormat::Q4_K => "q4_k",
                    QuantFormat::Q6_K => "q6_k",
                    _ => unreachable!("fixture expert formats"),
                }
                .to_owned(),
            });
            cursor = offset_in_blob + len;
        }
        let stride = cursor.next_multiple_of(4096);

        let mut file_bytes = vec![0u8; (stride * u64::from(g.n_experts)) as usize];
        for expert in 0..g.n_experts {
            let blob_base = (u64::from(expert) * stride) as usize;
            for (p, ((_, format, rows, in_dim), projection)) in
                slabs.iter().zip(&projections).enumerate()
            {
                let seed = layer as usize * 1009 + expert as usize * 101 + p * 13;
                let bytes = quant_bytes(*format, *rows, *in_dim, seed);
                let start = blob_base + projection.offset_in_blob as usize;
                file_bytes[start..start + bytes.len()].copy_from_slice(&bytes);
            }
        }
        let file = layer_file_name(layer);
        fs::write(root.join(&file), &file_bytes).expect("write layer file");
        layers.push(LayerLayout {
            file,
            stride,
            n_experts: g.n_experts,
            projections,
        });
    }
    let layout = ExpertsLayout { layers };
    write_layout(&root, &layout).expect("write layout");

    // File entries from what is actually on disk.
    let mut files = BTreeMap::new();
    let mut record = |name: String| {
        let path = root.join(&name);
        files.insert(
            name,
            FileEntry {
                size: fs::metadata(&path).unwrap().len(),
                sha256: sha256_file(&path).unwrap(),
            },
        );
    };
    record("common.bin".to_owned());
    record(LAYOUT_FILE.to_owned());
    for layer in 0..g.n_layers {
        record(layer_file_name(layer));
    }

    let mut tensor_types: BTreeMap<String, String> = defs
        .iter()
        .map(|(name, def)| (name.clone(), def.dtype.to_owned()))
        .collect();
    for (layer, layout_layer) in layout.layers.iter().enumerate() {
        for projection in &layout_layer.projections {
            let stem = match projection.name {
                ProjectionName::Gate => "ffn_gate_exps",
                ProjectionName::Up => "ffn_up_exps",
                ProjectionName::Down => "ffn_down_exps",
            };
            tensor_types.insert(
                format!("blk.{layer}.{stem}.weight"),
                projection.quant.clone(),
            );
        }
    }

    let manifest = Manifest {
        rvmp_version: RVMP_VERSION,
        model_id: g.model_id.to_owned(),
        source: SourceInfo {
            hf_repo: g.hf_repo.to_owned(),
            revision: "deadbeef".to_owned(),
            file: g.source_file.to_owned(),
            sha256: "0".repeat(64),
        },
        arch: g.arch(),
        quant: QuantInfo {
            scheme: "gguf".to_owned(),
            tensor_types,
        },
        common_tensors,
        files,
    };
    write_manifest(&root, &manifest).expect("write manifest");

    Fixture {
        _tmp: tmp,
        root,
        manifest,
        layout,
    }
}

/// Persist a mutated layout and refresh its manifest entry so hashes and
/// sizes stay consistent with disk.
pub(crate) fn rewrite_layout(fixture: &mut Fixture, layout: &ExpertsLayout) {
    write_layout(&fixture.root, layout).expect("rewrite layout");
    let path = fixture.root.join(LAYOUT_FILE);
    fixture.manifest.files.insert(
        LAYOUT_FILE.to_owned(),
        FileEntry {
            size: fs::metadata(&path).unwrap().len(),
            sha256: sha256_file(&path).unwrap(),
        },
    );
    write_manifest(&fixture.root, &fixture.manifest).expect("rewrite manifest");
    fixture.layout = layout.clone();
}
