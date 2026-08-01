//! Repack planning: derive everything the executor and manifest writer
//! need from a parsed GGUF header.
//!
//! [`RepackPlan::from_gguf`] extracts the architecture facts, classifies
//! every tensor as either an expert tensor (`blk.N.ffn_{gate,up,down}_exps`)
//! or a `common.bin` resident, computes the aligned `common.bin` index and
//! per-layer expert-blob layout, and emits a byte-exact [`CopyOp`] map the
//! executor can stream. Planning reads only the parsed header — no tensor
//! data is touched here, and the copy map describes byte-identical moves
//! only (the repacker never requantizes; see the hard rules in CLAUDE.md).

use std::collections::BTreeMap;
use std::ops::Range;

use ramvamp_core::format::{
    ArchInfo, COMMON_FILE, COMMON_TENSOR_ALIGN, CommonTensor, EXPERT_BLOB_ALIGN, ExpertsLayout,
    LayerLayout, MAX_EXPERTS, MAX_LAYERS, Projection, ProjectionName, QuantInfo, layer_file_name,
};

use crate::gguf::{GgmlType, GgufError, GgufFile, TensorInfo};

/// The only source architecture this build repacks. Broadened later.
pub const SUPPORTED_ARCH: &str = "qwen3moe";

/// Error building a repack plan from a parsed GGUF header.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// `general.architecture` is not [`SUPPORTED_ARCH`].
    #[error("unsupported architecture {found:?} (only {SUPPORTED_ARCH:?} for now)")]
    UnsupportedArch { found: String },
    /// A required metadata key is absent. Carries the exact key name.
    #[error("metadata key {0:?} is missing")]
    MissingKey(&'static str),
    /// A metadata key exists but has the wrong value type.
    #[error("metadata key {key:?}: expected {expected}")]
    BadKeyType {
        key: &'static str,
        expected: &'static str,
    },
    /// A metadata value is outside the range the format supports.
    #[error("metadata key {key:?}: value {value} out of range (1..={max})")]
    KeyOutOfRange {
        key: &'static str,
        value: u64,
        max: u64,
    },
    /// A tensor the plan requires is absent from the index.
    #[error("required tensor {0:?} is missing")]
    MissingTensor(String),
    /// A tensor has the wrong dimensionality for its role.
    #[error("tensor {name:?}: expected {expected} dimensions, found {found}")]
    BadTensorRank {
        name: String,
        expected: usize,
        found: usize,
    },
    /// `token_embd.weight` does not look like ne-order `[hidden, vocab]`.
    #[error(
        "token_embd.weight dims[0]={dim0} does not match embedding_length {hidden}; \
         ne-order [hidden, vocab] assumption violated"
    )]
    TokenEmbdShape { dim0: u64, hidden: u64 },
    /// An expert tensor's expert axis disagrees with `expert_count`.
    #[error("tensor {name:?}: expert axis {found} does not match expert_count {expected}")]
    ExpertCountMismatch {
        name: String,
        found: u64,
        expected: u64,
    },
    /// An expert tensor names a layer at or past `block_count`.
    #[error("tensor {name:?}: layer {layer} out of range (block_count {block_count})")]
    LayerOutOfRange {
        name: String,
        layer: u32,
        block_count: u32,
    },
    /// A layer is missing one of its three expert projections.
    #[error("layer {layer}: expert tensor {name:?} is missing")]
    MissingExpertTensor { layer: u32, name: String },
    /// An expert tensor's size is not divisible by the expert count.
    #[error("tensor {name:?}: size {size_bytes} not divisible by {n_experts} experts")]
    SlabNotDivisible {
        name: String,
        size_bytes: u64,
        n_experts: u64,
    },
    /// Layout arithmetic overflowed u64.
    #[error("arithmetic overflow computing layout for {0:?}")]
    Overflow(String),
    /// Slab-range computation failed in the GGUF layer.
    #[error(transparent)]
    Gguf(#[from] GgufError),
}

/// One byte-identical copy the executor must perform: `src` is an absolute
/// byte range in the source GGUF, written to `dst_file` at `dst_offset`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyOp {
    /// Absolute source byte range in the GGUF file.
    pub src: Range<u64>,
    /// Install-relative destination file (`common.bin` or a layer file).
    pub dst_file: String,
    /// Byte offset within the destination file.
    pub dst_offset: u64,
}

impl CopyOp {
    /// Length of the copy in bytes.
    pub fn len(&self) -> u64 {
        self.src.end - self.src.start
    }

    /// Whether the copy is empty (never true for planner-produced ops).
    pub fn is_empty(&self) -> bool {
        self.src.end <= self.src.start
    }
}

/// Aggregate numbers for display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Totals {
    /// Sum of all tensor payload bytes in the source — what a full repack
    /// must transfer. Excludes the GGUF header itself.
    pub download_bytes: u64,
    /// Sum of planned output data-file sizes: `common.bin` plus every
    /// expert layer file. Excludes manifest/layout JSON and tokenizer.
    pub installed_bytes: u64,
    /// Tensor count per lowercase ggml type name.
    pub tensor_type_counts: BTreeMap<String, u64>,
    /// Layers whose `ffn_down_exps` is not Q4_K (the wider Q6_K layers of
    /// a standard Q4_K_M file), ascending.
    pub q6k_down_layers: Vec<u32>,
}

/// Everything the future executor and manifest writer need, derived from
/// one parsed GGUF header.
#[derive(Debug)]
pub struct RepackPlan {
    /// Architecture facts for `manifest.json`.
    pub arch: ArchInfo,
    /// Quantization map for `manifest.json`: every tensor name to its
    /// lowercase ggml type name, scheme `"gguf"`.
    pub quant: QuantInfo,
    /// `common.bin` index: name to aligned offset/len/dtype, offsets
    /// [`COMMON_TENSOR_ALIGN`]-aligned, laid out in name order.
    pub common_tensors: BTreeMap<String, CommonTensor>,
    /// Total planned size of `common.bin` in bytes.
    pub common_size: u64,
    /// Per-layer expert blob geometry for `experts/layout.json`.
    pub layout: ExpertsLayout,
    /// Planned size of every expert layer file (`n_experts * stride`),
    /// keyed by install-relative file name.
    pub layer_file_sizes: BTreeMap<String, u64>,
    /// Byte-identical copies covering every common tensor and every expert
    /// slab: common tensors in name order, then layers in order (expert
    /// index outer, gate/up/down inner).
    pub copy_ops: Vec<CopyOp>,
    /// Aggregate numbers for display.
    pub totals: Totals,
}

impl RepackPlan {
    /// Trained context length (`qwen3moe.context_length`), from the arch.
    pub fn context_length(&self) -> u64 {
        self.arch.context_length
    }

    /// Build the full repack plan from a parsed GGUF header.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, PlanError> {
        let arch_name = gguf
            .metadata
            .get(KEY_ARCH)
            .ok_or(PlanError::MissingKey(KEY_ARCH))?
            .as_str()
            .ok_or(PlanError::BadKeyType {
                key: KEY_ARCH,
                expected: "string",
            })?;
        if arch_name != SUPPORTED_ARCH {
            return Err(PlanError::UnsupportedArch {
                found: arch_name.to_owned(),
            });
        }

        let n_layers = meta_u32(gguf, KEY_BLOCK_COUNT, MAX_LAYERS)?;
        let hidden = meta_u32(gguf, KEY_EMBEDDING_LENGTH, u32::MAX)?;
        let n_experts = meta_u32(gguf, KEY_EXPERT_COUNT, MAX_EXPERTS)?;
        let top_k = meta_u32(gguf, KEY_EXPERT_USED_COUNT, u32::MAX)?;
        let moe_intermediate = meta_u32(gguf, KEY_EXPERT_FF_LENGTH, u32::MAX)?;
        let n_heads = meta_u32(gguf, KEY_HEAD_COUNT, u32::MAX)?;
        let n_kv_heads = meta_u32(gguf, KEY_HEAD_COUNT_KV, u32::MAX)?;
        let head_dim = meta_u32(gguf, KEY_KEY_LENGTH, u32::MAX)?;
        let rope_theta = meta_float(gguf, KEY_ROPE_FREQ_BASE)?;
        let rms_eps = meta_float(gguf, KEY_RMS_EPSILON)?;
        let context_length = meta_uint(gguf, KEY_CONTEXT_LENGTH)?;

        // Vocab comes from the embedding matrix, not metadata. GGUF stores
        // dims in ggml `ne` order: dims[0] is the innermost (contiguous)
        // axis, which for `token_embd.weight` is the hidden dim — the shape
        // is [hidden, vocab], NOT [vocab, hidden]. Verified against the
        // fixture in the tests below (hidden innermost, vocab = dims[1])
        // and cross-checked against `embedding_length` here so a swapped
        // assumption fails loudly instead of yielding a garbage vocab.
        let token_embd = gguf
            .tensor(TOKEN_EMBD)
            .ok_or_else(|| PlanError::MissingTensor(TOKEN_EMBD.to_owned()))?;
        if token_embd.dims.len() != 2 {
            return Err(PlanError::BadTensorRank {
                name: TOKEN_EMBD.to_owned(),
                expected: 2,
                found: token_embd.dims.len(),
            });
        }
        if token_embd.dims[0] != u64::from(hidden) {
            return Err(PlanError::TokenEmbdShape {
                dim0: token_embd.dims[0],
                hidden: u64::from(hidden),
            });
        }
        let vocab = u32::try_from(token_embd.dims[1]).map_err(|_| PlanError::KeyOutOfRange {
            key: "token_embd.weight dims[1]",
            value: token_embd.dims[1],
            max: u64::from(u32::MAX),
        })?;

        let arch = ArchInfo {
            n_layers,
            n_experts,
            top_k,
            hidden,
            moe_intermediate,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab,
            context_length,
            rope_theta,
            rms_eps,
            // Not stored in GGUF metadata. Qwen3-MoE renormalizes the
            // top-k router weights after selection (`norm_topk_prob: true`
            // in the HF config; llama.cpp hardcodes the same for qwen3moe),
            // so it is hardcoded here for the qwen3moe arch.
            norm_topk_prob: true,
            // Untied lm_head ships as a separate `output.weight` tensor;
            // tied models omit it and reuse the embedding.
            tie_embeddings: gguf.tensor(OUTPUT_WEIGHT).is_none(),
            // qwen3moe has no shared (always-active) expert and full
            // attention on all layers.
            shared_expert: false,
            sliding_window: None,
        };

        // Classify every tensor: expert projections by layer, or common.
        let mut layers: Vec<LayerTensors> = (0..n_layers as usize)
            .map(|_| LayerTensors::default())
            .collect();
        let mut common: BTreeMap<&str, &TensorInfo> = BTreeMap::new();
        for t in gguf.tensors() {
            match classify(&t.name) {
                Class::Expert { layer, proj } => {
                    if t.dims.len() != 3 {
                        return Err(PlanError::BadTensorRank {
                            name: t.name.clone(),
                            expected: 3,
                            found: t.dims.len(),
                        });
                    }
                    if t.dims[2] != u64::from(n_experts) {
                        return Err(PlanError::ExpertCountMismatch {
                            name: t.name.clone(),
                            found: t.dims[2],
                            expected: u64::from(n_experts),
                        });
                    }
                    if layer >= n_layers {
                        return Err(PlanError::LayerOutOfRange {
                            name: t.name.clone(),
                            layer,
                            block_count: n_layers,
                        });
                    }
                    // The parser rejects duplicate tensor names, so each
                    // slot is written at most once.
                    layers[layer as usize].set(proj, t);
                }
                Class::Common => {
                    common.insert(&t.name, t);
                }
            }
        }

        let mut copy_ops = Vec::new();

        // common.bin: deterministic name order (BTreeMap iteration), each
        // offset aligned up to COMMON_TENSOR_ALIGN.
        let mut common_tensors = BTreeMap::new();
        let mut cursor: u64 = 0;
        for (&name, t) in &common {
            let offset = align_up(cursor, COMMON_TENSOR_ALIGN)
                .ok_or_else(|| PlanError::Overflow(name.to_owned()))?;
            let end = offset
                .checked_add(t.size_bytes)
                .ok_or_else(|| PlanError::Overflow(name.to_owned()))?;
            common_tensors.insert(
                name.to_owned(),
                CommonTensor {
                    offset,
                    len: t.size_bytes,
                    dtype: ggml_type_name(t.ggml_type).to_owned(),
                },
            );
            let src_start = gguf.data_offset(t);
            copy_ops.push(CopyOp {
                src: src_start..src_start + t.size_bytes,
                dst_file: COMMON_FILE.to_owned(),
                dst_offset: offset,
            });
            cursor = end;
        }
        let common_size = cursor;

        // Expert layout: per layer, blob = gate + up + down slabs, each
        // slab offset aligned up to EXPERT_BLOB_ALIGN within the blob;
        // stride = aligned blob size, file = n_experts fixed-stride blobs.
        let mut layer_layouts = Vec::with_capacity(n_layers as usize);
        let mut layer_file_sizes = BTreeMap::new();
        let mut q6k_down_layers = Vec::new();
        for (n, lt) in layers.iter().enumerate() {
            let layer = n as u32;
            let tensors = lt.require(layer)?;
            let n_exp = u64::from(n_experts);
            let overflow = || PlanError::Overflow(layer_file_name(layer));

            // Slab offsets within one blob, gate/up/down order.
            let mut projections = Vec::with_capacity(3);
            let mut blob_cursor: u64 = 0;
            for (proj_name, t) in tensors {
                let slab = slab_len(t, n_exp)?;
                let offset_in_blob =
                    align_up(blob_cursor, EXPERT_BLOB_ALIGN).ok_or_else(overflow)?;
                projections.push(Projection {
                    name: proj_name,
                    offset_in_blob,
                    len: slab,
                    quant: ggml_type_name(t.ggml_type).to_owned(),
                });
                blob_cursor = offset_in_blob.checked_add(slab).ok_or_else(overflow)?;
            }
            let stride = align_up(blob_cursor, EXPERT_BLOB_ALIGN).ok_or_else(overflow)?;
            let file = layer_file_name(layer);
            let file_size = n_exp.checked_mul(stride).ok_or_else(overflow)?;

            // Copy map: one op per expert per projection.
            for expert in 0..n_exp {
                let blob_base = expert.checked_mul(stride).ok_or_else(overflow)?;
                for ((_, t), proj) in tensors.iter().zip(&projections) {
                    let src = gguf.expert_slab(t, expert)?;
                    let dst_offset = blob_base
                        .checked_add(proj.offset_in_blob)
                        .ok_or_else(overflow)?;
                    copy_ops.push(CopyOp {
                        src,
                        dst_file: file.clone(),
                        dst_offset,
                    });
                }
            }

            if tensors[2].1.ggml_type != GgmlType::Q4_K {
                q6k_down_layers.push(layer);
            }
            layer_file_sizes.insert(file.clone(), file_size);
            layer_layouts.push(LayerLayout {
                file,
                stride,
                n_experts,
                projections,
            });
        }
        let layout = ExpertsLayout {
            layers: layer_layouts,
        };

        // Quant map and totals over every tensor in the file.
        let mut tensor_types = BTreeMap::new();
        let mut tensor_type_counts: BTreeMap<String, u64> = BTreeMap::new();
        let mut download_bytes: u64 = 0;
        for t in gguf.tensors() {
            let ty = ggml_type_name(t.ggml_type);
            tensor_types.insert(t.name.clone(), ty.to_owned());
            *tensor_type_counts.entry(ty.to_owned()).or_insert(0) += 1;
            download_bytes = download_bytes
                .checked_add(t.size_bytes)
                .ok_or_else(|| PlanError::Overflow(t.name.clone()))?;
        }
        let mut installed_bytes = common_size;
        for size in layer_file_sizes.values() {
            installed_bytes = installed_bytes
                .checked_add(*size)
                .ok_or_else(|| PlanError::Overflow("installed size".to_owned()))?;
        }

        Ok(RepackPlan {
            arch,
            quant: QuantInfo {
                scheme: "gguf".to_owned(),
                tensor_types,
            },
            common_tensors,
            common_size,
            layout,
            layer_file_sizes,
            copy_ops,
            totals: Totals {
                download_bytes,
                installed_bytes,
                tensor_type_counts,
                q6k_down_layers,
            },
        })
    }
}

/// Lowercase ggml type name, matching llama.cpp's spelling minus case.
pub fn ggml_type_name(ty: GgmlType) -> &'static str {
    match ty {
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::Q4_0 => "q4_0",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::Q4_K => "q4_k",
        GgmlType::Q5_K => "q5_k",
        GgmlType::Q6_K => "q6_k",
        GgmlType::Q8_K => "q8_k",
        GgmlType::BF16 => "bf16",
    }
}

const KEY_ARCH: &str = "general.architecture";
const KEY_BLOCK_COUNT: &str = "qwen3moe.block_count";
const KEY_EMBEDDING_LENGTH: &str = "qwen3moe.embedding_length";
const KEY_EXPERT_COUNT: &str = "qwen3moe.expert_count";
const KEY_EXPERT_USED_COUNT: &str = "qwen3moe.expert_used_count";
const KEY_EXPERT_FF_LENGTH: &str = "qwen3moe.expert_feed_forward_length";
const KEY_HEAD_COUNT: &str = "qwen3moe.attention.head_count";
const KEY_HEAD_COUNT_KV: &str = "qwen3moe.attention.head_count_kv";
const KEY_KEY_LENGTH: &str = "qwen3moe.attention.key_length";
const KEY_ROPE_FREQ_BASE: &str = "qwen3moe.rope.freq_base";
const KEY_RMS_EPSILON: &str = "qwen3moe.attention.layer_norm_rms_epsilon";
const KEY_CONTEXT_LENGTH: &str = "qwen3moe.context_length";

const TOKEN_EMBD: &str = "token_embd.weight";
const OUTPUT_WEIGHT: &str = "output.weight";

/// Result of classifying a tensor name.
enum Class {
    /// One of the three routed-expert projections of a layer.
    Expert { layer: u32, proj: ProjectionName },
    /// Everything else: packed into `common.bin`.
    Common,
}

/// Classify a tensor name: `blk.{N}.ffn_{gate,up,down}_exps.weight` is an
/// expert tensor, everything else is common.
fn classify(name: &str) -> Class {
    let Some(rest) = name.strip_prefix("blk.") else {
        return Class::Common;
    };
    let Some(dot) = rest.find('.') else {
        return Class::Common;
    };
    let (num, tail) = rest.split_at(dot);
    let proj = match tail {
        ".ffn_gate_exps.weight" => ProjectionName::Gate,
        ".ffn_up_exps.weight" => ProjectionName::Up,
        ".ffn_down_exps.weight" => ProjectionName::Down,
        _ => return Class::Common,
    };
    match num.parse::<u32>() {
        Ok(layer) => Class::Expert { layer, proj },
        Err(_) => Class::Common,
    }
}

/// The three expert projections collected for one layer.
#[derive(Default)]
struct LayerTensors<'a> {
    gate: Option<&'a TensorInfo>,
    up: Option<&'a TensorInfo>,
    down: Option<&'a TensorInfo>,
}

impl<'a> LayerTensors<'a> {
    fn set(&mut self, proj: ProjectionName, t: &'a TensorInfo) {
        match proj {
            ProjectionName::Gate => self.gate = Some(t),
            ProjectionName::Up => self.up = Some(t),
            ProjectionName::Down => self.down = Some(t),
        }
    }

    /// All three projections in gate/up/down order, or the exact missing
    /// tensor name.
    fn require(&self, layer: u32) -> Result<[(ProjectionName, &'a TensorInfo); 3], PlanError> {
        let missing = |stem: &str| PlanError::MissingExpertTensor {
            layer,
            name: format!("blk.{layer}.{stem}.weight"),
        };
        Ok([
            (
                ProjectionName::Gate,
                self.gate.ok_or_else(|| missing("ffn_gate_exps"))?,
            ),
            (
                ProjectionName::Up,
                self.up.ok_or_else(|| missing("ffn_up_exps"))?,
            ),
            (
                ProjectionName::Down,
                self.down.ok_or_else(|| missing("ffn_down_exps"))?,
            ),
        ])
    }
}

/// Bytes of one expert's contiguous slab of a 3-D `*_exps` tensor.
fn slab_len(t: &TensorInfo, n_experts: u64) -> Result<u64, PlanError> {
    if n_experts == 0 || t.size_bytes % n_experts != 0 {
        return Err(PlanError::SlabNotDivisible {
            name: t.name.clone(),
            size_bytes: t.size_bytes,
            n_experts,
        });
    }
    Ok(t.size_bytes / n_experts)
}

/// Round `v` up to the next multiple of `align` (a power of two or any
/// nonzero value); `None` on overflow.
fn align_up(v: u64, align: u64) -> Option<u64> {
    let rem = v % align;
    if rem == 0 {
        Some(v)
    } else {
        v.checked_add(align - rem)
    }
}

/// Required unsigned-integer metadata value, widened across writer widths.
fn meta_uint(gguf: &GgufFile, key: &'static str) -> Result<u64, PlanError> {
    gguf.metadata
        .get(key)
        .ok_or(PlanError::MissingKey(key))?
        .as_uint()
        .ok_or(PlanError::BadKeyType {
            key,
            expected: "unsigned integer",
        })
}

/// [`meta_uint`] narrowed into `1..=max` for `ArchInfo`'s u32 fields.
fn meta_u32(gguf: &GgufFile, key: &'static str, max: u32) -> Result<u32, PlanError> {
    let v = meta_uint(gguf, key)?;
    if v == 0 || v > u64::from(max) {
        return Err(PlanError::KeyOutOfRange {
            key,
            value: v,
            max: u64::from(max),
        });
    }
    Ok(v as u32)
}

/// Required float metadata value; integer-typed values widen (some writers
/// store `freq_base` as an integer).
fn meta_float(gguf: &GgufFile, key: &'static str) -> Result<f64, PlanError> {
    use crate::gguf::MetaValue;
    let value = gguf.metadata.get(key).ok_or(PlanError::MissingKey(key))?;
    match *value {
        MetaValue::F32(v) => Ok(f64::from(v)),
        MetaValue::F64(v) => Ok(v),
        _ => value
            .as_uint()
            .map(|v| v as f64)
            .ok_or(PlanError::BadKeyType {
                key,
                expected: "float",
            }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::MetaValue;
    use crate::gguf::testutil::{FixtureBuilder, TensorSpec};

    /// Two MoE layers, four experts. Layer 0's down projection is Q6_K,
    /// layer 1 is pure Q4_K. `attn_k_norm` is 32 bytes so the common
    /// layout must insert alignment padding after it.
    fn base_tensors(untied: bool) -> Vec<TensorSpec> {
        let mut tensors = vec![
            TensorSpec::new("token_embd.weight", &[64, 512], GgmlType::F16),
            TensorSpec::new("blk.0.attn_k_norm.weight", &[8], GgmlType::F32),
            TensorSpec::new("blk.0.attn_norm.weight", &[64], GgmlType::F32),
            TensorSpec::new("blk.0.ffn_gate_exps.weight", &[256, 8, 4], GgmlType::Q4_K),
            TensorSpec::new("blk.0.ffn_up_exps.weight", &[256, 8, 4], GgmlType::Q4_K),
            TensorSpec::new("blk.0.ffn_down_exps.weight", &[256, 8, 4], GgmlType::Q6_K),
            TensorSpec::new("blk.1.ffn_gate_exps.weight", &[256, 8, 4], GgmlType::Q4_K),
            TensorSpec::new("blk.1.ffn_up_exps.weight", &[256, 8, 4], GgmlType::Q4_K),
            TensorSpec::new("blk.1.ffn_down_exps.weight", &[256, 8, 4], GgmlType::Q4_K),
            TensorSpec::new("output_norm.weight", &[64], GgmlType::F32),
        ];
        if untied {
            tensors.push(TensorSpec::new("output.weight", &[64, 512], GgmlType::F16));
        }
        tensors
    }

    fn parse_fixture(tensors: Vec<TensorSpec>) -> GgufFile {
        let bytes = FixtureBuilder {
            tensors,
            ..FixtureBuilder::default()
        }
        .build();
        let slice: &[u8] = &bytes;
        GgufFile::parse(&slice).expect("fixture parses")
    }

    /// The qwen3moe.* metadata the planner requires. The parsed fixture's
    /// `metadata` field is public, so keys are injected post-parse; widths
    /// are mixed on purpose to exercise `as_uint` widening.
    fn inject_meta(gguf: &mut GgufFile) {
        let m = &mut gguf.metadata;
        m.insert(
            "general.architecture".to_owned(),
            MetaValue::String("qwen3moe".to_owned()),
        );
        m.insert("qwen3moe.block_count".to_owned(), MetaValue::U32(2));
        m.insert("qwen3moe.embedding_length".to_owned(), MetaValue::U64(64));
        m.insert("qwen3moe.expert_count".to_owned(), MetaValue::U32(4));
        m.insert("qwen3moe.expert_used_count".to_owned(), MetaValue::U16(2));
        m.insert(
            "qwen3moe.expert_feed_forward_length".to_owned(),
            MetaValue::U32(8),
        );
        m.insert(
            "qwen3moe.attention.head_count".to_owned(),
            MetaValue::U32(8),
        );
        m.insert(
            "qwen3moe.attention.head_count_kv".to_owned(),
            MetaValue::U32(2),
        );
        m.insert(
            "qwen3moe.attention.key_length".to_owned(),
            MetaValue::U32(8),
        );
        m.insert("qwen3moe.rope.freq_base".to_owned(), MetaValue::F32(1e7));
        m.insert(
            "qwen3moe.attention.layer_norm_rms_epsilon".to_owned(),
            MetaValue::F32(1e-6),
        );
        m.insert("qwen3moe.context_length".to_owned(), MetaValue::U64(4096));
    }

    fn qwen_fixture(untied: bool) -> GgufFile {
        let mut gguf = parse_fixture(base_tensors(untied));
        inject_meta(&mut gguf);
        gguf
    }

    #[test]
    fn arch_extraction_matches_fixture_metadata() {
        let plan = RepackPlan::from_gguf(&qwen_fixture(true)).unwrap();
        let a = &plan.arch;
        assert_eq!(a.n_layers, 2);
        assert_eq!(a.n_experts, 4);
        assert_eq!(a.top_k, 2);
        assert_eq!(a.hidden, 64);
        assert_eq!(a.moe_intermediate, 8);
        assert_eq!(a.n_heads, 8);
        assert_eq!(a.n_kv_heads, 2);
        assert_eq!(a.head_dim, 8);
        // ne-order check: token_embd is [hidden, vocab], vocab = dims[1].
        assert_eq!(a.vocab, 512);
        assert_eq!(a.rope_theta, 1e7);
        assert!((a.rms_eps - 1e-6).abs() < 1e-12);
        assert!(a.norm_topk_prob);
        assert!(!a.tie_embeddings);
        assert!(!a.shared_expert);
        assert_eq!(a.sliding_window, None);
        assert_eq!(a.context_length, 4096);
        assert_eq!(plan.context_length(), 4096);
        assert_eq!(plan.quant.scheme, "gguf");
    }

    #[test]
    fn tie_detection_both_ways() {
        let tied = RepackPlan::from_gguf(&qwen_fixture(false)).unwrap();
        assert!(tied.arch.tie_embeddings);
        let untied = RepackPlan::from_gguf(&qwen_fixture(true)).unwrap();
        assert!(!untied.arch.tie_embeddings);
    }

    #[test]
    fn missing_key_error_names_the_key() {
        let mut gguf = qwen_fixture(true);
        gguf.metadata.remove("qwen3moe.expert_count");
        let err = RepackPlan::from_gguf(&gguf).unwrap_err();
        assert!(matches!(
            err,
            PlanError::MissingKey("qwen3moe.expert_count")
        ));
    }

    #[test]
    fn non_qwen3moe_arch_rejected() {
        let gguf = parse_fixture(base_tensors(true));
        // No injection: architecture stays "ramvamp-test".
        let err = RepackPlan::from_gguf(&gguf).unwrap_err();
        assert!(matches!(err, PlanError::UnsupportedArch { found } if found == "ramvamp-test"));
    }

    #[test]
    fn token_embd_shape_guard_fires_on_hidden_mismatch() {
        let mut gguf = qwen_fixture(true);
        gguf.metadata
            .insert("qwen3moe.embedding_length".to_owned(), MetaValue::U32(128));
        let err = RepackPlan::from_gguf(&gguf).unwrap_err();
        assert!(matches!(
            err,
            PlanError::TokenEmbdShape {
                dim0: 64,
                hidden: 128
            }
        ));
    }

    #[test]
    fn classification_splits_experts_from_common() {
        let plan = RepackPlan::from_gguf(&qwen_fixture(true)).unwrap();
        let common: Vec<&str> = plan.common_tensors.keys().map(String::as_str).collect();
        assert_eq!(
            common,
            [
                "blk.0.attn_k_norm.weight",
                "blk.0.attn_norm.weight",
                "output.weight",
                "output_norm.weight",
                "token_embd.weight",
            ]
        );
        assert_eq!(plan.layout.layers.len(), 2);
        // Every tensor, expert or common, appears in the quant map.
        assert_eq!(plan.quant.tensor_types.len(), 11);
        assert_eq!(
            plan.quant.tensor_types.get("blk.0.ffn_down_exps.weight"),
            Some(&"q6_k".to_owned())
        );
        assert_eq!(
            plan.totals.tensor_type_counts,
            BTreeMap::from([
                ("f16".to_owned(), 2),
                ("f32".to_owned(), 3),
                ("q4_k".to_owned(), 5),
                ("q6_k".to_owned(), 1),
            ])
        );
    }

    #[test]
    fn common_layout_is_aligned_name_ordered_and_packed() {
        let plan = RepackPlan::from_gguf(&qwen_fixture(true)).unwrap();
        let get = |name: &str| plan.common_tensors.get(name).unwrap();
        // Name order with 64-byte alignment: the 32-byte k_norm forces a
        // padding gap before the next tensor.
        assert_eq!(
            (
                get("blk.0.attn_k_norm.weight").offset,
                get("blk.0.attn_k_norm.weight").len
            ),
            (0, 32)
        );
        assert_eq!(
            (
                get("blk.0.attn_norm.weight").offset,
                get("blk.0.attn_norm.weight").len
            ),
            (64, 256)
        );
        assert_eq!(
            (get("output.weight").offset, get("output.weight").len),
            (320, 65536)
        );
        assert_eq!(
            (
                get("output_norm.weight").offset,
                get("output_norm.weight").len
            ),
            (65856, 256)
        );
        assert_eq!(
            (
                get("token_embd.weight").offset,
                get("token_embd.weight").len
            ),
            (66112, 65536)
        );
        assert_eq!(plan.common_size, 131648);
        for t in plan.common_tensors.values() {
            assert_eq!(t.offset % COMMON_TENSOR_ALIGN, 0);
        }
        assert_eq!(get("blk.0.attn_norm.weight").dtype, "f32");
        assert_eq!(get("token_embd.weight").dtype, "f16");
    }

    #[test]
    fn expert_layout_alignment_and_stride() {
        let plan = RepackPlan::from_gguf(&qwen_fixture(true)).unwrap();
        // Q4_K slab: (256*8*4/256)*144/4 experts = 1152 B.
        // Q6_K slab: (256*8*4/256)*210/4 experts = 1680 B.
        let l0 = &plan.layout.layers[0];
        assert_eq!(l0.file, "experts/layer_00.bin");
        assert_eq!(l0.n_experts, 4);
        let offs: Vec<(ProjectionName, u64, u64, &str)> = l0
            .projections
            .iter()
            .map(|p| (p.name, p.offset_in_blob, p.len, p.quant.as_str()))
            .collect();
        assert_eq!(
            offs,
            [
                (ProjectionName::Gate, 0, 1152, "q4_k"),
                (ProjectionName::Up, 4096, 1152, "q4_k"),
                (ProjectionName::Down, 8192, 1680, "q6_k"),
            ]
        );
        // 8192 + 1680 = 9872, aligned up to 3 pages.
        assert_eq!(l0.stride, 12288);

        let l1 = &plan.layout.layers[1];
        assert_eq!(l1.file, "experts/layer_01.bin");
        assert_eq!(l1.projections[2].len, 1152);
        assert_eq!(l1.projections[2].quant, "q4_k");
        assert_eq!(l1.stride, 12288);

        assert_eq!(
            plan.layer_file_sizes,
            BTreeMap::from([
                ("experts/layer_00.bin".to_owned(), 4 * 12288),
                ("experts/layer_01.bin".to_owned(), 4 * 12288),
            ])
        );
        assert_eq!(plan.totals.q6k_down_layers, [0]);
        assert_eq!(
            plan.totals.installed_bytes,
            plan.common_size + 2 * 4 * 12288
        );
    }

    #[test]
    fn copy_map_is_disjoint_and_covers_every_tensor() {
        let gguf = qwen_fixture(true);
        let plan = RepackPlan::from_gguf(&gguf).unwrap();
        // 5 common tensors + 2 layers * 4 experts * 3 projections.
        assert_eq!(plan.copy_ops.len(), 5 + 24);

        // Destination ranges must be disjoint per file and in bounds.
        let mut by_file: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
        for op in &plan.copy_ops {
            assert!(!op.is_empty());
            by_file
                .entry(op.dst_file.as_str())
                .or_default()
                .push((op.dst_offset, op.dst_offset + op.len()));
        }
        for (file, ranges) in &mut by_file {
            let file_size = if *file == COMMON_FILE {
                plan.common_size
            } else {
                *plan.layer_file_sizes.get(*file).expect("known dst file")
            };
            ranges.sort_unstable();
            for pair in ranges.windows(2) {
                assert!(pair[0].1 <= pair[1].0, "dst overlap in {file}");
            }
            assert!(
                ranges.last().unwrap().1 <= file_size,
                "dst out of bounds in {file}"
            );
        }

        // Source ranges must be disjoint and cover every tensor exactly.
        let mut srcs: Vec<(u64, u64)> = plan
            .copy_ops
            .iter()
            .map(|op| (op.src.start, op.src.end))
            .collect();
        srcs.sort_unstable();
        for pair in srcs.windows(2) {
            assert!(pair[0].1 <= pair[1].0, "src ranges overlap");
        }
        let total_copied: u64 = plan.copy_ops.iter().map(CopyOp::len).sum();
        assert_eq!(total_copied, plan.totals.download_bytes);
        for t in gguf.tensors() {
            let start = gguf.data_offset(t);
            let end = start + t.size_bytes;
            let covered: u64 = plan
                .copy_ops
                .iter()
                .filter(|op| op.src.start >= start && op.src.end <= end)
                .map(|op| op.len())
                .sum();
            assert_eq!(covered, t.size_bytes, "tensor {} not fully covered", t.name);
        }
    }

    #[test]
    fn expert_axis_must_match_expert_count() {
        let mut gguf = qwen_fixture(true);
        gguf.metadata
            .insert("qwen3moe.expert_count".to_owned(), MetaValue::U32(8));
        let err = RepackPlan::from_gguf(&gguf).unwrap_err();
        assert!(matches!(
            err,
            PlanError::ExpertCountMismatch {
                found: 4,
                expected: 8,
                ..
            }
        ));
    }

    #[test]
    fn expert_layer_out_of_range_rejected() {
        let mut tensors = base_tensors(true);
        tensors.push(TensorSpec::new(
            "blk.5.ffn_gate_exps.weight",
            &[256, 8, 4],
            GgmlType::Q4_K,
        ));
        let mut gguf = parse_fixture(tensors);
        inject_meta(&mut gguf);
        let err = RepackPlan::from_gguf(&gguf).unwrap_err();
        assert!(matches!(
            err,
            PlanError::LayerOutOfRange {
                layer: 5,
                block_count: 2,
                ..
            }
        ));
    }

    #[test]
    fn missing_expert_projection_names_layer_and_tensor() {
        let tensors: Vec<TensorSpec> = base_tensors(true)
            .into_iter()
            .filter(|t| t.name != "blk.1.ffn_up_exps.weight")
            .collect();
        let mut gguf = parse_fixture(tensors);
        inject_meta(&mut gguf);
        let err = RepackPlan::from_gguf(&gguf).unwrap_err();
        assert!(matches!(
            err,
            PlanError::MissingExpertTensor { layer: 1, name } if name == "blk.1.ffn_up_exps.weight"
        ));
    }

    #[test]
    fn align_up_math() {
        assert_eq!(align_up(0, 4096), Some(0));
        assert_eq!(align_up(1, 4096), Some(4096));
        assert_eq!(align_up(4096, 4096), Some(4096));
        assert_eq!(align_up(4097, 4096), Some(8192));
        assert_eq!(align_up(u64::MAX, 4096), None);
    }
}
