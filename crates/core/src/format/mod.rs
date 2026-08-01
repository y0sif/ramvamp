//! The ramvamp packed model format.
//!
//! An installed model is a directory:
//!
//! ```text
//! model.rvmp/
//!   manifest.json        # architecture, quantization, file sizes, SHA-256 hashes
//!   common.bin           # embeddings/head, attention, routers, norms, shared experts
//!   tokenizer/           # HF tokenizer.json + chat template
//!   experts/
//!     layout.json        # sub-tensor offsets within each expert blob
//!     layer_00.bin ...   # one file per layer: fixed-stride, page-aligned expert blobs
//! ```
//!
//! Invariants:
//! - Expert blobs have a fixed stride and 4 KiB (page) alignment so a single
//!   O_DIRECT read fetches exactly one expert.
//! - Quantized values are copied from the source checkpoint byte-for-byte;
//!   the repacker changes layout, never numerics.
//! - `manifest.json` is written last; a directory without it is a partial
//!   install and must be rejected by the loader.
//!
//! On-disk schemas (v1, frozen at implementation):
//! - `manifest.json` -> [`Manifest`]: model identity, source checkpoint pin,
//!   architecture facts the runtime may rely on, quantization map, the
//!   tensor index into `common.bin` (offsets 64-byte aligned), and size +
//!   SHA-256 for every data file.
//! - `experts/layout.json` -> [`ExpertsLayout`]: per-layer blob stride (a
//!   4 KiB multiple) and per-projection offsets (4 KiB aligned) within a
//!   blob. Strides are uniform within a layer but may differ across layers
//!   (Q6_K `down` layers are wider than pure-Q4_K layers).
//!
//! Install flow: the repacker writes everything into `<dir>.partial/`,
//! writes `manifest.json` last ([`write_manifest`]: temp file, fsync,
//! rename), then [`promote`]s the partial directory to its final name with
//! one `rename(2)` after the manifest re-validates. [`is_complete`] is the
//! loader-side gate; [`verify_files`] re-hashes data files against the
//! manifest using a fixed-size streaming buffer.
//!
//! Everything parsed here is untrusted until validated: typed
//! [`FormatError`]s, no panics, and no allocation whose size is driven by
//! parsed values (JSON reads are capped, validation loops are bounded).

mod error;
mod install;
mod layout;
mod manifest;
mod verify;

pub use error::FormatError;
pub use install::{
    MAX_JSON_BYTES, PARTIAL_SUFFIX, is_complete, load_layout, load_manifest, partial_dir, promote,
    write_layout, write_manifest,
};
pub use layout::{ExpertsLayout, LayerLayout, Projection, ProjectionName};
pub use manifest::{
    ArchInfo, CommonTensor, FileEntry, MAX_EXPERTS, MAX_LAYERS, Manifest, QuantInfo, SourceInfo,
    layer_file_name,
};
pub use verify::{sha256_file, verify_files};

/// Manifest schema version this build reads and writes.
pub const RVMP_VERSION: u32 = 1;

/// Required alignment for tensor offsets inside `common.bin`.
///
/// 64 bytes = one cache line; vectorized loads over the mmap'd common
/// weights never straddle an unaligned base.
pub const COMMON_TENSOR_ALIGN: u64 = 64;

/// Required alignment for expert blob strides and for projection slab
/// offsets within a blob: 4 KiB, the page size O_DIRECT reads need.
pub const EXPERT_BLOB_ALIGN: u64 = 4096;

/// Install-relative path of the manifest.
pub const MANIFEST_FILE: &str = "manifest.json";

/// Install-relative path of the packed common weights.
pub const COMMON_FILE: &str = "common.bin";

/// Install-relative path of the experts layout.
pub const LAYOUT_FILE: &str = "experts/layout.json";

/// Quantization schemes this build understands.
pub const KNOWN_QUANT_SCHEMES: &[&str] = &["gguf"];

#[cfg(test)]
pub(crate) mod testutil {
    //! Shared fixtures for format tests: a small consistent manifest/layout
    //! pair and a self-cleaning temp directory.

    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        ArchInfo, CommonTensor, ExpertsLayout, FileEntry, LayerLayout, Manifest, Projection,
        ProjectionName, QuantInfo, RVMP_VERSION, SourceInfo,
    };

    /// A syntactically valid SHA-256 hex digest for entries whose bytes are
    /// never hashed by the test.
    pub(crate) const ZERO_SHA: &str =
        "0000000000000000000000000000000000000000000000000000000000000000";

    /// Two MoE layers, four experts, strides differing per layer; consistent
    /// with [`sample_layout`].
    pub(crate) fn sample_manifest() -> Manifest {
        let mut files = BTreeMap::new();
        files.insert(
            "common.bin".to_owned(),
            FileEntry {
                size: 4096,
                sha256: ZERO_SHA.to_owned(),
            },
        );
        files.insert(
            "experts/layout.json".to_owned(),
            FileEntry {
                size: 2,
                sha256: ZERO_SHA.to_owned(),
            },
        );
        files.insert(
            "experts/layer_00.bin".to_owned(),
            FileEntry {
                size: 4 * 12288,
                sha256: ZERO_SHA.to_owned(),
            },
        );
        files.insert(
            "experts/layer_01.bin".to_owned(),
            FileEntry {
                size: 4 * 16384,
                sha256: ZERO_SHA.to_owned(),
            },
        );

        let mut common_tensors = BTreeMap::new();
        common_tensors.insert(
            "token_embd.weight".to_owned(),
            CommonTensor {
                offset: 0,
                len: 2048,
                dtype: "q4_k".to_owned(),
            },
        );
        common_tensors.insert(
            "output_norm.weight".to_owned(),
            CommonTensor {
                offset: 2048,
                len: 256,
                dtype: "f32".to_owned(),
            },
        );

        let mut tensor_types = BTreeMap::new();
        tensor_types.insert("ffn_gate_exps".to_owned(), "q4_k".to_owned());
        tensor_types.insert("ffn_down_exps".to_owned(), "q6_k".to_owned());

        Manifest {
            rvmp_version: RVMP_VERSION,
            model_id: "test-moe-2l".to_owned(),
            source: SourceInfo {
                hf_repo: "test/repo".to_owned(),
                revision: "0123abcd".to_owned(),
                file: "test-Q4_K_M.gguf".to_owned(),
                sha256: ZERO_SHA.to_owned(),
            },
            arch: ArchInfo {
                n_layers: 2,
                n_experts: 4,
                top_k: 2,
                hidden: 64,
                moe_intermediate: 96,
                n_heads: 8,
                n_kv_heads: 2,
                head_dim: 8,
                vocab: 512,
                rope_theta: 1e7,
                rms_eps: 1e-6,
                norm_topk_prob: true,
                tie_embeddings: false,
                shared_expert: false,
                sliding_window: None,
            },
            quant: QuantInfo {
                scheme: "gguf".to_owned(),
                tensor_types,
            },
            common_tensors,
            files,
        }
    }

    /// Layout matching [`sample_manifest`]: layer 0 stride 12 KiB, layer 1
    /// stride 16 KiB (Q6_K down), gate/up/down each 4 KiB-aligned.
    pub(crate) fn sample_layout() -> ExpertsLayout {
        ExpertsLayout {
            layers: vec![
                LayerLayout {
                    file: "experts/layer_00.bin".to_owned(),
                    stride: 12288,
                    n_experts: 4,
                    projections: vec![
                        Projection {
                            name: ProjectionName::Gate,
                            offset_in_blob: 0,
                            len: 1024,
                            quant: "q4_k".to_owned(),
                        },
                        Projection {
                            name: ProjectionName::Up,
                            offset_in_blob: 4096,
                            len: 1024,
                            quant: "q4_k".to_owned(),
                        },
                        Projection {
                            name: ProjectionName::Down,
                            offset_in_blob: 8192,
                            len: 2048,
                            quant: "q6_k".to_owned(),
                        },
                    ],
                },
                LayerLayout {
                    file: "experts/layer_01.bin".to_owned(),
                    stride: 16384,
                    n_experts: 4,
                    projections: vec![
                        Projection {
                            name: ProjectionName::Gate,
                            offset_in_blob: 0,
                            len: 2048,
                            quant: "q4_k".to_owned(),
                        },
                        Projection {
                            name: ProjectionName::Up,
                            offset_in_blob: 4096,
                            len: 2048,
                            quant: "q4_k".to_owned(),
                        },
                        Projection {
                            name: ProjectionName::Down,
                            offset_in_blob: 8192,
                            len: 4096,
                            quant: "q6_k".to_owned(),
                        },
                    ],
                },
            ],
        }
    }

    /// Unique directory under the system temp dir, removed on drop.
    pub(crate) struct TempDir(PathBuf);

    impl TempDir {
        pub(crate) fn new(tag: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let pid = std::process::id();
            let path = std::env::temp_dir().join(format!("ramvamp-format-{tag}-{pid}-{n}-{nanos}"));
            std::fs::create_dir_all(&path).expect("create test temp dir");
            TempDir(path)
        }

        pub(crate) fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
