//! Serde types and validation for `manifest.json` (schema v1).
//!
//! The manifest is the install's source of truth: model identity, the
//! pinned source checkpoint, architecture facts the runtime may rely on
//! without re-deriving them, the quantization map frozen from the source
//! audit, the tensor index into `common.bin`, and the size + SHA-256 of
//! every data file. It is written last and atomically promoted, so its
//! presence marks a complete install.

use std::collections::BTreeMap;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use super::{
    COMMON_FILE, COMMON_TENSOR_ALIGN, FormatError, KNOWN_QUANT_SCHEMES, LAYOUT_FILE, RVMP_VERSION,
};

/// Validation cap on `arch.n_layers`. Bounds the work `validate` does per
/// manifest, so a hostile manifest cannot drive unbounded loops.
pub const MAX_LAYERS: u32 = 1024;

/// Validation cap on `arch.n_experts`; same purpose as [`MAX_LAYERS`].
pub const MAX_EXPERTS: u32 = 8192;

/// `manifest.json`, schema v1 (frozen at implementation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Schema version; must equal [`RVMP_VERSION`].
    pub rvmp_version: u32,
    /// Stable install identifier, e.g. `qwen3-30b-a3b-instruct-2507`.
    pub model_id: String,
    /// Pin of the source checkpoint this install was repacked from.
    pub source: SourceInfo,
    /// Architecture facts the runtime may rely on.
    pub arch: ArchInfo,
    /// Quantization scheme and per-tensor type map.
    pub quant: QuantInfo,
    /// Index into `common.bin`: tensor name to byte range and dtype.
    /// Offsets are [`COMMON_TENSOR_ALIGN`]-byte aligned.
    pub common_tensors: BTreeMap<String, CommonTensor>,
    /// Size and SHA-256 of every data file, keyed by install-relative path.
    pub files: BTreeMap<String, FileEntry>,
}

/// The pinned source checkpoint (`source` in the manifest).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceInfo {
    /// Hugging Face repo the source file came from.
    pub hf_repo: String,
    /// Exact repo revision (commit hash) the file was fetched at.
    pub revision: String,
    /// Source file name within the repo, e.g. `...Q4_K_M.gguf`.
    pub file: String,
    /// SHA-256 of the source file.
    pub sha256: String,
}

/// Architecture facts (`arch` in the manifest).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchInfo {
    /// Transformer layer count.
    pub n_layers: u32,
    /// Routed experts per layer.
    pub n_experts: u32,
    /// Experts activated per token.
    pub top_k: u32,
    /// Hidden (embedding) dimension.
    pub hidden: u32,
    /// Per-expert FFN intermediate dimension.
    pub moe_intermediate: u32,
    /// Attention query heads.
    pub n_heads: u32,
    /// Attention KV heads (GQA groups).
    pub n_kv_heads: u32,
    /// Dimension of each attention head.
    pub head_dim: u32,
    /// Vocabulary size.
    pub vocab: u32,
    /// RoPE base frequency.
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    pub rms_eps: f64,
    /// Renormalize the top-k router weights after selection.
    pub norm_topk_prob: bool,
    /// Embedding and lm_head share weights.
    pub tie_embeddings: bool,
    /// Model has a shared (always-active) expert.
    pub shared_expert: bool,
    /// Sliding-window size, or `None` for full attention on all layers.
    pub sliding_window: Option<u32>,
}

/// Quantization description (`quant` in the manifest).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantInfo {
    /// Source quantization scheme; must be one of [`KNOWN_QUANT_SCHEMES`].
    pub scheme: String,
    /// Per-tensor quantized type map frozen from the source-file audit,
    /// e.g. `"ffn_down_exps" -> "q6_k"`.
    pub tensor_types: BTreeMap<String, String>,
}

/// One entry of the `common.bin` tensor index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommonTensor {
    /// Byte offset in `common.bin`; must be [`COMMON_TENSOR_ALIGN`]-aligned.
    pub offset: u64,
    /// Byte length of the packed tensor data.
    pub len: u64,
    /// Quantized/packed dtype of the bytes at this range, e.g. `q6_k`, `f32`.
    pub dtype: String,
}

/// Size and integrity record for one install file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    /// File size in bytes.
    pub size: u64,
    /// Lowercase (accepted case-insensitively) SHA-256 hex digest.
    pub sha256: String,
}

/// Canonical install-relative name of a layer's expert file:
/// `experts/layer_NN.bin` (two digits minimum, grows past layer 99).
pub fn layer_file_name(layer: u32) -> String {
    format!("experts/layer_{layer:02}.bin")
}

impl Manifest {
    /// Validate everything the schema itself cannot express.
    ///
    /// Checks, in order: schema version, non-empty identity fields, known
    /// quant scheme, architecture sanity bounds, file-name safety and
    /// digest format for every `files` entry, presence of the required
    /// entries (`common.bin`, `experts/layout.json`, and
    /// `experts/layer_NN.bin` for every layer in `arch`), and the
    /// `common_tensors` index (64-byte-aligned offsets, in-bounds,
    /// non-overlapping ranges).
    pub fn validate(&self) -> Result<(), FormatError> {
        if self.rvmp_version != RVMP_VERSION {
            return Err(FormatError::UnsupportedVersion {
                found: self.rvmp_version,
                supported: RVMP_VERSION,
            });
        }
        if self.model_id.is_empty() {
            return Err(FormatError::InvalidManifest("model_id is empty".to_owned()));
        }
        if !KNOWN_QUANT_SCHEMES.contains(&self.quant.scheme.as_str()) {
            return Err(FormatError::UnknownQuantScheme {
                scheme: self.quant.scheme.clone(),
                known: KNOWN_QUANT_SCHEMES,
            });
        }
        for (tensor, ty) in &self.quant.tensor_types {
            if tensor.is_empty() || ty.is_empty() {
                return Err(FormatError::InvalidManifest(format!(
                    "quant.tensor_types entry {tensor:?} -> {ty:?} has an empty name or type"
                )));
            }
        }
        if !is_sha256_hex(&self.source.sha256) {
            return Err(FormatError::InvalidManifest(
                "source.sha256 is not a 64-char hex digest".to_owned(),
            ));
        }
        self.arch.validate()?;

        for (name, entry) in &self.files {
            check_file_name(name)?;
            if !is_sha256_hex(&entry.sha256) {
                return Err(FormatError::InvalidManifest(format!(
                    "files[{name:?}].sha256 is not a 64-char hex digest"
                )));
            }
        }
        let common = self
            .files
            .get(COMMON_FILE)
            .ok_or_else(|| FormatError::MissingFileEntry(COMMON_FILE.to_owned()))?;
        if !self.files.contains_key(LAYOUT_FILE) {
            return Err(FormatError::MissingFileEntry(LAYOUT_FILE.to_owned()));
        }
        // Bounded by MAX_LAYERS via arch.validate() above.
        for layer in 0..self.arch.n_layers {
            let name = layer_file_name(layer);
            if !self.files.contains_key(&name) {
                return Err(FormatError::MissingFileEntry(name));
            }
        }

        let mut ranges: Vec<(u64, u64, &str)> = Vec::with_capacity(self.common_tensors.len());
        for (name, tensor) in &self.common_tensors {
            if tensor.offset % COMMON_TENSOR_ALIGN != 0 {
                return Err(FormatError::Misaligned {
                    what: "common tensor offset",
                    name: name.clone(),
                    value: tensor.offset,
                    align: COMMON_TENSOR_ALIGN,
                });
            }
            if tensor.len == 0 || tensor.dtype.is_empty() {
                return Err(FormatError::InvalidManifest(format!(
                    "common tensor {name:?} has a zero length or empty dtype"
                )));
            }
            let end = tensor.offset.checked_add(tensor.len).ok_or_else(|| {
                FormatError::InvalidManifest(format!(
                    "common tensor {name:?}: offset + len overflows"
                ))
            })?;
            if end > common.size {
                return Err(FormatError::InvalidManifest(format!(
                    "common tensor {name:?} ends at byte {end}, past common.bin ({} bytes)",
                    common.size
                )));
            }
            ranges.push((tensor.offset, end, name.as_str()));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(FormatError::InvalidManifest(format!(
                    "common tensors {:?} and {:?} overlap",
                    pair[0].2, pair[1].2
                )));
            }
        }
        Ok(())
    }
}

impl ArchInfo {
    /// Sanity bounds only; model-specific correctness is the repacker's job.
    fn validate(&self) -> Result<(), FormatError> {
        let err = |msg: String| Err(FormatError::InvalidArch(msg));
        if self.n_layers == 0 || self.n_layers > MAX_LAYERS {
            return err(format!(
                "n_layers {} not in 1..={MAX_LAYERS}",
                self.n_layers
            ));
        }
        if self.n_experts == 0 || self.n_experts > MAX_EXPERTS {
            return err(format!(
                "n_experts {} not in 1..={MAX_EXPERTS}",
                self.n_experts
            ));
        }
        if self.top_k == 0 || self.top_k > self.n_experts {
            return err(format!(
                "top_k {} not in 1..=n_experts ({})",
                self.top_k, self.n_experts
            ));
        }
        for (field, value) in [
            ("hidden", self.hidden),
            ("moe_intermediate", self.moe_intermediate),
            ("n_heads", self.n_heads),
            ("n_kv_heads", self.n_kv_heads),
            ("head_dim", self.head_dim),
            ("vocab", self.vocab),
        ] {
            if value == 0 {
                return err(format!("{field} must be nonzero"));
            }
        }
        if self.n_heads % self.n_kv_heads != 0 {
            return err(format!(
                "n_heads {} not divisible by n_kv_heads {} (GQA needs whole groups)",
                self.n_heads, self.n_kv_heads
            ));
        }
        if self.rope_theta <= 0.0 || !self.rope_theta.is_finite() {
            return err(format!(
                "rope_theta {} must be finite and positive",
                self.rope_theta
            ));
        }
        if self.rms_eps <= 0.0 || !self.rms_eps.is_finite() {
            return err(format!(
                "rms_eps {} must be finite and positive",
                self.rms_eps
            ));
        }
        if self.sliding_window == Some(0) {
            return err("sliding_window is 0 (use null for full attention)".to_owned());
        }
        Ok(())
    }
}

/// Reject manifest file names that could escape the install directory:
/// every path component must be a normal name (no `..`, no leading `/`,
/// no `.`), and the name must be non-empty.
pub(crate) fn check_file_name(name: &str) -> Result<(), FormatError> {
    let path = Path::new(name);
    let safe = !name.is_empty()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)));
    if safe {
        Ok(())
    } else {
        Err(FormatError::UnsafeFileName(name.to_owned()))
    }
}

fn is_sha256_hex(digest: &str) -> bool {
    digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{ZERO_SHA, sample_manifest};
    use super::*;

    #[test]
    fn serde_roundtrip() {
        let manifest = sample_manifest();
        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let back: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, manifest);
    }

    #[test]
    fn parses_v1_schema_json() {
        // Key names as frozen in docs/architecture.md plus common_tensors.
        let json = format!(
            r#"{{
              "rvmp_version": 1,
              "model_id": "qwen3-30b-a3b-instruct-2507",
              "source": {{ "hf_repo": "r", "revision": "v", "file": "f-Q4_K_M.gguf",
                          "sha256": "{ZERO_SHA}" }},
              "arch": {{ "n_layers": 48, "n_experts": 128, "top_k": 8, "hidden": 2048,
                        "moe_intermediate": 768, "n_heads": 32, "n_kv_heads": 4,
                        "head_dim": 128, "vocab": 151936, "rope_theta": 1e7,
                        "rms_eps": 1e-6, "norm_topk_prob": true, "tie_embeddings": false,
                        "shared_expert": false, "sliding_window": null }},
              "quant": {{ "scheme": "gguf", "tensor_types": {{ "ffn_down_exps": "q6_k" }} }},
              "common_tensors": {{ "token_embd.weight": {{ "offset": 0, "len": 1024,
                                                          "dtype": "q4_k" }} }},
              "files": {{ "common.bin": {{ "size": 4096, "sha256": "{ZERO_SHA}" }} }}
            }}"#
        );
        let manifest: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest.arch.n_layers, 48);
        assert_eq!(manifest.arch.sliding_window, None);
        assert_eq!(manifest.common_tensors["token_embd.weight"].len, 1024);
    }

    #[test]
    fn rejects_unknown_manifest_field() {
        let mut value = serde_json::to_value(sample_manifest()).unwrap();
        value["surprise"] = serde_json::json!(1);
        assert!(serde_json::from_value::<Manifest>(value).is_err());
    }

    #[test]
    fn validates_sample() {
        sample_manifest().validate().unwrap();
    }

    #[test]
    fn rejects_wrong_version() {
        let mut manifest = sample_manifest();
        manifest.rvmp_version = 2;
        let err = manifest.validate().unwrap_err();
        assert!(matches!(
            err,
            FormatError::UnsupportedVersion {
                found: 2,
                supported: RVMP_VERSION
            }
        ));
    }

    #[test]
    fn rejects_top_k_above_n_experts() {
        let mut manifest = sample_manifest();
        manifest.arch.top_k = manifest.arch.n_experts + 1;
        assert!(matches!(
            manifest.validate().unwrap_err(),
            FormatError::InvalidArch(_)
        ));
    }

    #[test]
    fn rejects_zero_layers_and_layer_cap() {
        let mut manifest = sample_manifest();
        manifest.arch.n_layers = 0;
        assert!(matches!(
            manifest.validate().unwrap_err(),
            FormatError::InvalidArch(_)
        ));
        manifest.arch.n_layers = MAX_LAYERS + 1;
        assert!(matches!(
            manifest.validate().unwrap_err(),
            FormatError::InvalidArch(_)
        ));
    }

    #[test]
    fn rejects_unknown_quant_scheme() {
        let mut manifest = sample_manifest();
        manifest.quant.scheme = "awq".to_owned();
        assert!(matches!(
            manifest.validate().unwrap_err(),
            FormatError::UnknownQuantScheme { .. }
        ));
    }

    #[test]
    fn rejects_missing_layer_file_entry() {
        let mut manifest = sample_manifest();
        manifest.files.remove("experts/layer_01.bin");
        let err = manifest.validate().unwrap_err();
        match err {
            FormatError::MissingFileEntry(name) => assert_eq!(name, "experts/layer_01.bin"),
            other => panic!("expected MissingFileEntry, got {other:?}"),
        }
    }

    #[test]
    fn rejects_misaligned_common_tensor() {
        let mut manifest = sample_manifest();
        manifest
            .common_tensors
            .get_mut("output_norm.weight")
            .unwrap()
            .offset = 2049;
        assert!(matches!(
            manifest.validate().unwrap_err(),
            FormatError::Misaligned {
                value: 2049,
                align: COMMON_TENSOR_ALIGN,
                ..
            }
        ));
    }

    #[test]
    fn rejects_common_tensor_past_eof() {
        let mut manifest = sample_manifest();
        manifest
            .common_tensors
            .get_mut("output_norm.weight")
            .unwrap()
            .len = 1 << 40;
        assert!(matches!(
            manifest.validate().unwrap_err(),
            FormatError::InvalidManifest(_)
        ));
    }

    #[test]
    fn rejects_overlapping_common_tensors() {
        let mut manifest = sample_manifest();
        manifest
            .common_tensors
            .get_mut("token_embd.weight")
            .unwrap()
            .len = 2112; // ends at 2112, past output_norm.weight's start (2048)
        assert!(matches!(
            manifest.validate().unwrap_err(),
            FormatError::InvalidManifest(_)
        ));
    }

    #[test]
    fn rejects_traversal_file_names() {
        for bad in ["../evil.bin", "/etc/passwd", "a/../b", "", "./x"] {
            let mut manifest = sample_manifest();
            manifest.files.insert(
                bad.to_owned(),
                FileEntry {
                    size: 1,
                    sha256: ZERO_SHA.to_owned(),
                },
            );
            assert!(
                matches!(
                    manifest.validate().unwrap_err(),
                    FormatError::UnsafeFileName(_)
                ),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn rejects_malformed_digest() {
        let mut manifest = sample_manifest();
        manifest.files.get_mut("common.bin").unwrap().sha256 = "not-hex".to_owned();
        assert!(matches!(
            manifest.validate().unwrap_err(),
            FormatError::InvalidManifest(_)
        ));
    }
}
