//! Model loading and typed weight access.
//!
//! [`Model::load`] turns an install directory into typed, validated weight
//! views. Load-time policy:
//!
//! - The manifest is parsed and validated, `experts/layout.json` and
//!   `common.bin` are verified against it (sizes always, hashes unless
//!   [`LoadOptions::skip_hashes`]), and `common.bin` is mapped read-only.
//! - Every tensor in the shape table (see [`super::shapes`]) must exist
//!   with the required dtype and exact byte size; any miss is a typed
//!   error naming the tensor.
//! - Big quantized matrices stay in the mmap and are handed out as
//!   [`QuantTensor`] byte views. Small f32 tensors (norms) are decoded to
//!   owned `Vec<f32>` at load — copied via `chunks_exact(4)` +
//!   `from_le_bytes`, never by casting the map. Routers are decoded the
//!   same way into per-layer [`F32Tensor`]s: ~1 MiB per layer, 48 MiB
//!   total for v0 — an accepted load-time copy so router matvecs read
//!   plain `&[f32]` rows.

use std::path::Path;

use crate::format::{
    ArchInfo, LAYOUT_FILE, LayerLayout, Manifest, ProjectionName, load_layout, load_manifest,
};
use crate::io::{ExpertReader, LoadOptions, MappedCommon, to_usize, verify_named_file};
use crate::kernels::KernelError;
use crate::kernels::quants::{
    QuantFormat, dequantize_row_q4_k, dequantize_row_q5_k, dequantize_row_q6_k, dequantize_row_q8_0,
};

use super::error::ModelError;
use super::shapes::{self, AllowedQuant, Dims};

/// A packed quantized matrix view into the common map.
///
/// Rows span the input axis: `out_dim` rows of `in_dim` weights, each row
/// `bytes.len() / out_dim` packed bytes — the exact operand layout of
/// [`crate::kernels::gemv_q8_k`] / [`crate::kernels::gemv_q8_0`].
#[derive(Debug, Clone, Copy)]
pub struct QuantTensor<'a> {
    /// Row-major packed weight bytes.
    pub bytes: &'a [u8],
    /// Quantization format of every row.
    pub format: QuantFormat,
    /// Weights per row (input dimension).
    pub in_dim: usize,
    /// Number of rows (output dimension).
    pub out_dim: usize,
}

impl<'a> QuantTensor<'a> {
    /// Packed bytes per row. Exact by construction: the loader verified
    /// `bytes.len() == out_dim * row_bytes(in_dim)`.
    pub fn row_bytes(&self) -> usize {
        self.bytes.len() / self.out_dim
    }

    /// One packed row, or `None` past `out_dim`.
    pub fn row(&self, row: usize) -> Option<&'a [u8]> {
        if row >= self.out_dim {
            return None;
        }
        let row_bytes = self.row_bytes();
        Some(&self.bytes[row * row_bytes..(row + 1) * row_bytes])
    }
}

/// An owned f32 matrix decoded from the install at load time.
#[derive(Debug, Clone, PartialEq)]
pub struct F32Tensor {
    data: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl F32Tensor {
    /// All values, row-major.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Values per row (input dimension).
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Number of rows (output dimension).
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// One row, or `None` past `out_dim`.
    pub fn row(&self, row: usize) -> Option<&[f32]> {
        if row >= self.out_dim {
            return None;
        }
        Some(&self.data[row * self.in_dim..(row + 1) * self.in_dim])
    }
}

/// Typed access to one layer's resident weights.
///
/// Projections are zero-copy views into the common map; norms and the
/// router were decoded to f32 at load.
#[derive(Debug, Clone, Copy)]
pub struct LayerWeights<'a> {
    /// Query projection (`hidden -> n_heads*head_dim`, q4_k).
    pub attn_q: QuantTensor<'a>,
    /// Key projection (`hidden -> n_kv_heads*head_dim`, q8_0).
    pub attn_k: QuantTensor<'a>,
    /// Value projection (`hidden -> n_kv_heads*head_dim`, q4_k or q6_k).
    pub attn_v: QuantTensor<'a>,
    /// Output projection (`n_heads*head_dim -> hidden`, q5_k).
    pub attn_output: QuantTensor<'a>,
    /// Pre-attention RMSNorm weight (`[hidden]`).
    pub attn_norm: &'a [f32],
    /// Pre-FFN RMSNorm weight (`[hidden]`).
    pub ffn_norm: &'a [f32],
    /// Per-head query RMSNorm weight (`[head_dim]`).
    pub attn_q_norm: &'a [f32],
    /// Per-head key RMSNorm weight (`[head_dim]`).
    pub attn_k_norm: &'a [f32],
    /// Router / expert-gate weight (`n_experts` rows of `hidden`).
    pub router: &'a F32Tensor,
}

/// Resolved location and shape of one quantized tensor in the map.
#[derive(Debug, Clone, Copy)]
struct TensorSpec {
    offset: usize,
    len: usize,
    format: QuantFormat,
    in_dim: usize,
    out_dim: usize,
}

/// One layer's resolved metadata and owned decodes.
#[derive(Debug)]
struct LayerMeta {
    attn_q: TensorSpec,
    attn_k: TensorSpec,
    attn_v: TensorSpec,
    attn_output: TensorSpec,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    router: F32Tensor,
}

/// A loaded, validated model: mapped common weights, decoded norms and
/// routers, and lazy expert-file access.
#[derive(Debug)]
pub struct Model {
    manifest: Manifest,
    common: MappedCommon,
    experts: ExpertReader,
    embedding: TensorSpec,
    lm_head: TensorSpec,
    final_norm: Vec<f32>,
    layers: Vec<LayerMeta>,
}

impl Model {
    /// Load an installed model directory.
    ///
    /// Steps, in order: parse + validate `manifest.json`; verify and parse
    /// `experts/layout.json`; verify and map `common.bin`; resolve every
    /// expected tensor against the shape table (dtype + exact size);
    /// validate the expert slab shapes per layer; build the lazy
    /// [`ExpertReader`] (layer files hash on first use).
    ///
    /// # Errors
    ///
    /// [`ModelError::MissingTensor`] / [`ModelError::WrongDtype`] /
    /// [`ModelError::WrongSize`] name the offending tensor; format and
    /// I/O failures pass through as [`ModelError::Format`] /
    /// [`ModelError::Io`].
    pub fn load(dir: impl AsRef<Path>, options: LoadOptions) -> Result<Self, ModelError> {
        let dir = dir.as_ref();
        let manifest = load_manifest(dir)?;
        manifest.validate()?;
        verify_named_file(dir, &manifest, LAYOUT_FILE, options)?;
        let layout = load_layout(dir)?;
        layout.validate_against(&manifest)?;
        let common = MappedCommon::open(dir, &manifest, options)?;

        let dims = Dims::from_arch(&manifest.arch);
        let embedding = require_quant(
            &manifest,
            &common,
            "token_embd.weight".to_owned(),
            shapes::Q4K,
            dims.hidden,
            dims.vocab,
        )?;
        let lm_head = if manifest.arch.tie_embeddings {
            embedding
        } else {
            require_quant(
                &manifest,
                &common,
                "output.weight".to_owned(),
                shapes::Q6K,
                dims.hidden,
                dims.vocab,
            )?
        };
        let final_norm = require_f32(
            &manifest,
            &common,
            "output_norm.weight".to_owned(),
            1,
            dims.hidden,
        )?;

        let n_layers = manifest.arch.n_layers;
        let mut layers = Vec::with_capacity(n_layers as usize);
        for layer in 0..n_layers {
            validate_expert_layer(&layout.layers[layer as usize], layer, dims)?;
            let quant = |stem: &str, allowed: AllowedQuant, in_dim: usize, out_dim: usize| {
                require_quant(
                    &manifest,
                    &common,
                    format!("blk.{layer}.{stem}.weight"),
                    allowed,
                    in_dim,
                    out_dim,
                )
            };
            let f32s = |stem: &str, rows: usize, cols: usize| {
                require_f32(
                    &manifest,
                    &common,
                    format!("blk.{layer}.{stem}.weight"),
                    rows,
                    cols,
                )
            };
            let router_data = f32s("ffn_gate_inp", dims.n_experts, dims.hidden)?;
            layers.push(LayerMeta {
                attn_q: quant("attn_q", shapes::Q4K, dims.hidden, dims.q_dim)?,
                attn_k: quant("attn_k", shapes::Q8_0, dims.hidden, dims.kv_dim)?,
                attn_v: quant("attn_v", shapes::Q4K_OR_Q6K, dims.hidden, dims.kv_dim)?,
                attn_output: quant("attn_output", shapes::Q5K, dims.q_dim, dims.hidden)?,
                attn_norm: f32s("attn_norm", 1, dims.hidden)?,
                ffn_norm: f32s("ffn_norm", 1, dims.hidden)?,
                attn_q_norm: f32s("attn_q_norm", 1, dims.head_dim)?,
                attn_k_norm: f32s("attn_k_norm", 1, dims.head_dim)?,
                router: F32Tensor {
                    data: router_data,
                    in_dim: dims.hidden,
                    out_dim: dims.n_experts,
                },
            });
        }

        let experts = ExpertReader::new(dir, &manifest, &layout, options)?;
        tracing::info!(
            model = manifest.model_id.as_str(),
            layers = n_layers,
            "model loaded"
        );
        Ok(Self {
            manifest,
            common,
            experts,
            embedding,
            lm_head,
            final_norm,
            layers,
        })
    }

    /// Architecture facts from the manifest.
    pub fn arch(&self) -> &ArchInfo {
        &self.manifest.arch
    }

    /// The validated manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Transformer layer count.
    pub fn n_layers(&self) -> u32 {
        self.manifest.arch.n_layers
    }

    /// Typed weight views for one layer.
    ///
    /// # Errors
    ///
    /// [`ModelError::LayerOutOfRange`] past the last layer.
    pub fn layer(&self, layer: u32) -> Result<LayerWeights<'_>, ModelError> {
        let meta = self
            .layers
            .get(layer as usize)
            .ok_or(ModelError::LayerOutOfRange {
                layer,
                n_layers: self.n_layers(),
            })?;
        Ok(LayerWeights {
            attn_q: self.view(meta.attn_q),
            attn_k: self.view(meta.attn_k),
            attn_v: self.view(meta.attn_v),
            attn_output: self.view(meta.attn_output),
            attn_norm: &meta.attn_norm,
            ffn_norm: &meta.ffn_norm,
            attn_q_norm: &meta.attn_q_norm,
            attn_k_norm: &meta.attn_k_norm,
            router: &meta.router,
        })
    }

    /// One token's packed embedding row (q4_k bytes of a `hidden`-vector).
    ///
    /// # Errors
    ///
    /// [`ModelError::TokenOutOfRange`] at or past the vocab size.
    pub fn embedding_row(&self, token: u32) -> Result<&[u8], ModelError> {
        let vocab = self.embedding.out_dim;
        if (token as usize) >= vocab {
            return Err(ModelError::TokenOutOfRange {
                token,
                vocab: self.manifest.arch.vocab,
            });
        }
        let tensor = self.view(self.embedding);
        // In range: token < out_dim just checked.
        Ok(tensor.row(token as usize).expect("token bounds checked"))
    }

    /// Dequantize one token's embedding into `out` (`hidden` floats).
    ///
    /// Convenience wrapper over [`Self::embedding_row`] and the row
    /// dequantizer; allocates one temporary row.
    ///
    /// # Errors
    ///
    /// [`ModelError::TokenOutOfRange`] on a bad token id;
    /// [`ModelError::OutputLen`] when `out` is not `hidden` long.
    pub fn embed(&self, token: u32, out: &mut [f32]) -> Result<(), ModelError> {
        let hidden = self.embedding.in_dim;
        if out.len() != hidden {
            return Err(ModelError::OutputLen {
                expected: hidden,
                found: out.len(),
            });
        }
        let row = self.embedding_row(token)?;
        let values = dequantize_row(self.embedding.format, row)?;
        out.copy_from_slice(&values);
        Ok(())
    }

    /// The lm_head matrix (`hidden -> vocab`; the embedding when tied).
    pub fn lm_head(&self) -> QuantTensor<'_> {
        self.view(self.lm_head)
    }

    /// The final RMSNorm weight (`[hidden]`).
    pub fn final_norm(&self) -> &[f32] {
        &self.final_norm
    }

    /// The lazy expert-file reader.
    pub fn expert_reader(&self) -> &ExpertReader {
        &self.experts
    }

    /// Materialize a view from a resolved spec. In bounds by construction:
    /// every spec's range was bounds-checked against the map at load.
    fn view(&self, spec: TensorSpec) -> QuantTensor<'_> {
        QuantTensor {
            bytes: &self.common.bytes()[spec.offset..spec.offset + spec.len],
            format: spec.format,
            in_dim: spec.in_dim,
            out_dim: spec.out_dim,
        }
    }
}

/// Dequantize one packed row of any storable weight format.
fn dequantize_row(format: QuantFormat, bytes: &[u8]) -> Result<Vec<f32>, KernelError> {
    match format {
        QuantFormat::Q4_K => dequantize_row_q4_k(bytes),
        QuantFormat::Q5_K => dequantize_row_q5_k(bytes),
        QuantFormat::Q6_K => dequantize_row_q6_k(bytes),
        QuantFormat::Q8_0 => dequantize_row_q8_0(bytes),
        QuantFormat::Q8_K => Err(KernelError::UnsupportedFormat {
            what: "dequantize_row",
            format,
        }),
    }
}

/// Decode raw little-endian f32 bytes by copy — never by casting the mmap
/// (the map guarantees no alignment for `f32` reads).
fn f32_from_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Resolve one quantized common tensor against its expectation.
fn require_quant(
    manifest: &Manifest,
    common: &MappedCommon,
    name: String,
    allowed: AllowedQuant,
    in_dim: usize,
    out_dim: usize,
) -> Result<TensorSpec, ModelError> {
    let entry = manifest
        .common_tensors
        .get(&name)
        .ok_or_else(|| ModelError::MissingTensor(name.clone()))?;
    let format = allowed
        .match_dtype(&entry.dtype)
        .ok_or_else(|| ModelError::WrongDtype {
            tensor: name.clone(),
            expected: allowed.desc,
            found: entry.dtype.clone(),
        })?;
    let row_bytes = format.row_bytes(in_dim)?;
    let expected = row_bytes as u64 * out_dim as u64;
    if entry.len != expected {
        return Err(ModelError::WrongSize {
            tensor: name,
            expected,
            found: entry.len,
        });
    }
    // Bounds-check the range against the map once, here.
    let bytes = common.tensor(&name, entry)?;
    Ok(TensorSpec {
        offset: to_usize(entry.offset, "common tensor offset")?,
        len: bytes.len(),
        format,
        in_dim,
        out_dim,
    })
}

/// Resolve and decode one f32 common tensor (`rows` rows of `cols`).
fn require_f32(
    manifest: &Manifest,
    common: &MappedCommon,
    name: String,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, ModelError> {
    let entry = manifest
        .common_tensors
        .get(&name)
        .ok_or_else(|| ModelError::MissingTensor(name.clone()))?;
    if entry.dtype != "f32" {
        return Err(ModelError::WrongDtype {
            tensor: name,
            expected: "f32",
            found: entry.dtype.clone(),
        });
    }
    let expected = 4 * rows as u64 * cols as u64;
    if entry.len != expected {
        return Err(ModelError::WrongSize {
            tensor: name,
            expected,
            found: entry.len,
        });
    }
    Ok(f32_from_le(common.tensor(&name, entry)?))
}

/// Validate one layer's expert slab shapes against the audited map:
/// gate/up q4_k `hidden -> moe_intermediate`, down q4_k-or-q6_k
/// `moe_intermediate -> hidden`. Errors name the canonical GGUF tensor.
fn validate_expert_layer(
    layer_layout: &LayerLayout,
    layer: u32,
    dims: Dims,
) -> Result<(), ModelError> {
    let expectations = [
        (
            ProjectionName::Gate,
            "ffn_gate_exps",
            shapes::Q4K,
            dims.hidden,
            dims.moe_intermediate,
        ),
        (
            ProjectionName::Up,
            "ffn_up_exps",
            shapes::Q4K,
            dims.hidden,
            dims.moe_intermediate,
        ),
        (
            ProjectionName::Down,
            "ffn_down_exps",
            shapes::Q4K_OR_Q6K,
            dims.moe_intermediate,
            dims.hidden,
        ),
    ];
    for (proj_name, stem, allowed, in_dim, out_dim) in expectations {
        let name = format!("blk.{layer}.{stem}.weight");
        let projection = layer_layout
            .projections
            .iter()
            .find(|p| p.name == proj_name)
            .ok_or_else(|| ModelError::MissingTensor(name.clone()))?;
        let format =
            allowed
                .match_dtype(&projection.quant)
                .ok_or_else(|| ModelError::WrongDtype {
                    tensor: name.clone(),
                    expected: allowed.desc,
                    found: projection.quant.clone(),
                })?;
        let row_bytes = format.row_bytes(in_dim)?;
        let expected = row_bytes as u64 * out_dim as u64;
        if projection.len != expected {
            return Err(ModelError::WrongSize {
                tensor: name,
                expected,
                found: projection.len,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{FormatError, write_manifest};
    use crate::io::IoError;
    use crate::io::testutil::{
        HEAD_DIM, HIDDEN, N_EXPERTS, VOCAB, build_install, f32_pattern, fixture_arch,
        rewrite_layout,
    };

    const SKIP: LoadOptions = LoadOptions { skip_hashes: true };

    #[test]
    fn load_happy_path_with_hashes() {
        let fx = build_install("model-happy");
        let model = Model::load(&fx.root, LoadOptions::default()).unwrap();
        assert_eq!(model.arch(), &fixture_arch());
        assert_eq!(model.n_layers(), 2);
        assert_eq!(model.manifest().model_id, "fixture-moe-2l");
    }

    #[test]
    fn accessors_have_expected_shapes_and_formats() {
        let fx = build_install("model-shapes");
        let model = Model::load(&fx.root, SKIP).unwrap();

        let head = model.lm_head();
        assert_eq!(head.format, QuantFormat::Q6_K);
        assert_eq!((head.in_dim, head.out_dim), (HIDDEN, VOCAB));
        assert_eq!(head.bytes.len(), VOCAB * 210);
        assert_eq!(head.row_bytes(), 210);
        assert!(head.row(VOCAB).is_none());
        assert_eq!(model.final_norm().len(), HIDDEN);

        let l0 = model.layer(0).unwrap();
        assert_eq!(l0.attn_q.format, QuantFormat::Q4_K);
        assert_eq!((l0.attn_q.in_dim, l0.attn_q.out_dim), (HIDDEN, 256));
        assert_eq!(l0.attn_q.bytes.len(), 256 * 144);
        assert_eq!(l0.attn_k.format, QuantFormat::Q8_0);
        assert_eq!((l0.attn_k.in_dim, l0.attn_k.out_dim), (HIDDEN, 128));
        assert_eq!(l0.attn_k.bytes.len(), 128 * 272);
        assert_eq!(l0.attn_v.format, QuantFormat::Q6_K);
        assert_eq!(l0.attn_v.bytes.len(), 128 * 210);
        assert_eq!(l0.attn_output.format, QuantFormat::Q5_K);
        assert_eq!(
            (l0.attn_output.in_dim, l0.attn_output.out_dim),
            (256, HIDDEN)
        );
        assert_eq!(l0.attn_output.bytes.len(), HIDDEN * 176);
        assert_eq!(l0.attn_norm.len(), HIDDEN);
        assert_eq!(l0.ffn_norm.len(), HIDDEN);
        assert_eq!(l0.attn_q_norm.len(), HEAD_DIM);
        assert_eq!(l0.attn_k_norm.len(), HEAD_DIM);
        assert_eq!(l0.router.out_dim(), N_EXPERTS as usize);
        assert_eq!(l0.router.in_dim(), HIDDEN);
        assert_eq!(l0.router.row(3).unwrap().len(), HIDDEN);
        assert!(l0.router.row(4).is_none());

        // Layer 1 is the pure-Q4_K flavor.
        let l1 = model.layer(1).unwrap();
        assert_eq!(l1.attn_v.format, QuantFormat::Q4_K);
        assert_eq!(l1.attn_v.bytes.len(), 128 * 144);

        assert!(matches!(
            model.layer(2).unwrap_err(),
            ModelError::LayerOutOfRange {
                layer: 2,
                n_layers: 2
            }
        ));
    }

    #[test]
    fn f32_decode_matches_written_pattern() {
        let fx = build_install("model-f32");
        let model = Model::load(&fx.root, SKIP).unwrap();
        let l0 = model.layer(0).unwrap();
        for (i, &value) in l0.attn_norm.iter().enumerate() {
            assert_eq!(value, f32_pattern("blk.0.attn_norm.weight", i));
        }
        for (i, &value) in model.final_norm().iter().enumerate() {
            assert_eq!(value, f32_pattern("output_norm.weight", i));
        }
        // Router rows are row-major slices of the same pattern.
        let row2 = l0.router.row(2).unwrap();
        for (c, &value) in row2.iter().enumerate() {
            assert_eq!(
                value,
                f32_pattern("blk.0.ffn_gate_inp.weight", 2 * HIDDEN + c)
            );
        }
    }

    #[test]
    fn f32_from_le_known_bytes() {
        let bytes = [
            0x00, 0x00, 0x80, 0x3f, // 1.0
            0x00, 0x00, 0x80, 0xbf, // -1.0
            0x00, 0x00, 0x00, 0x00, // 0.0
        ];
        assert_eq!(f32_from_le(&bytes), vec![1.0, -1.0, 0.0]);
    }

    #[test]
    fn embedding_row_and_embed_roundtrip() {
        let fx = build_install("model-embed");
        let model = Model::load(&fx.root, SKIP).unwrap();

        let row = model.embedding_row(5).unwrap();
        assert_eq!(row.len(), 144); // one q4_k super-block per 256-wide row
        let expected = dequantize_row_q4_k(row).unwrap();

        let mut out = vec![0.0f32; HIDDEN];
        model.embed(5, &mut out).unwrap();
        assert_eq!(out, expected);
        assert!(out.iter().all(|v| v.is_finite()));

        assert!(matches!(
            model.embedding_row(VOCAB as u32).unwrap_err(),
            ModelError::TokenOutOfRange { token, vocab }
                if token == VOCAB as u32 && vocab == VOCAB as u32
        ));
        let mut short = vec![0.0f32; HIDDEN - 1];
        assert!(matches!(
            model.embed(0, &mut short).unwrap_err(),
            ModelError::OutputLen { expected, found }
                if expected == HIDDEN && found == HIDDEN - 1
        ));
    }

    #[test]
    fn missing_tensor_is_named() {
        let mut fx = build_install("model-missing");
        fx.manifest.common_tensors.remove("blk.1.ffn_norm.weight");
        write_manifest(&fx.root, &fx.manifest).unwrap();
        assert!(matches!(
            Model::load(&fx.root, SKIP).unwrap_err(),
            ModelError::MissingTensor(name) if name == "blk.1.ffn_norm.weight"
        ));
    }

    #[test]
    fn wrong_dtype_is_named() {
        let mut fx = build_install("model-dtype");
        fx.manifest
            .common_tensors
            .get_mut("blk.0.attn_k.weight")
            .unwrap()
            .dtype = "q4_k".to_owned();
        write_manifest(&fx.root, &fx.manifest).unwrap();
        assert!(matches!(
            Model::load(&fx.root, SKIP).unwrap_err(),
            ModelError::WrongDtype { tensor, expected: "q8_0", found }
                if tensor == "blk.0.attn_k.weight" && found == "q4_k"
        ));
    }

    #[test]
    fn wrong_size_is_named() {
        let mut fx = build_install("model-size");
        fx.manifest
            .common_tensors
            .get_mut("blk.0.attn_q.weight")
            .unwrap()
            .len -= 144;
        write_manifest(&fx.root, &fx.manifest).unwrap();
        assert!(matches!(
            Model::load(&fx.root, SKIP).unwrap_err(),
            ModelError::WrongSize { tensor, expected, found }
                if tensor == "blk.0.attn_q.weight"
                    && expected == 256 * 144
                    && found == 256 * 144 - 144
        ));
    }

    #[test]
    fn truncated_common_fails_even_without_hashes() {
        let fx = build_install("model-truncated");
        let path = fx.root.join("common.bin");
        let data = std::fs::read(&path).unwrap();
        std::fs::write(&path, &data[..data.len() - 128]).unwrap();
        assert!(matches!(
            Model::load(&fx.root, SKIP).unwrap_err(),
            ModelError::Io(IoError::Format(FormatError::SizeMismatch { name, .. }))
                if name == "common.bin"
        ));
    }

    #[test]
    fn skip_hashes_toggles_corruption_detection() {
        let fx = build_install("model-hashes");
        let path = fx.root.join("common.bin");
        let mut data = std::fs::read(&path).unwrap();
        data[999] ^= 0xff;
        std::fs::write(&path, data).unwrap();
        assert!(matches!(
            Model::load(&fx.root, LoadOptions::default()).unwrap_err(),
            ModelError::Io(IoError::Format(FormatError::HashMismatch { name, .. }))
                if name == "common.bin"
        ));
        Model::load(&fx.root, SKIP).unwrap();
    }

    #[test]
    fn expert_slab_size_mismatch_is_named() {
        let mut fx = build_install("model-expert-size");
        let mut layout = fx.layout.clone();
        layout.layers[0].projections[2].len -= 210;
        rewrite_layout(&mut fx, &layout);
        assert!(matches!(
            Model::load(&fx.root, SKIP).unwrap_err(),
            ModelError::WrongSize { tensor, .. } if tensor == "blk.0.ffn_down_exps.weight"
        ));
    }

    #[test]
    fn expert_slab_dtype_mismatch_is_named() {
        let mut fx = build_install("model-expert-dtype");
        let mut layout = fx.layout.clone();
        layout.layers[1].projections[0].quant = "q8_0".to_owned();
        rewrite_layout(&mut fx, &layout);
        assert!(matches!(
            Model::load(&fx.root, SKIP).unwrap_err(),
            ModelError::WrongDtype { tensor, expected: "q4_k", found }
                if tensor == "blk.1.ffn_gate_exps.weight" && found == "q8_0"
        ));
    }

    #[test]
    fn expert_reader_is_wired_through() {
        let fx = build_install("model-experts");
        let model = Model::load(&fx.root, SKIP).unwrap();
        let reader = model.expert_reader();
        assert_eq!(reader.n_layers(), 2);
        let mut buf = Vec::new();
        let view = reader.read_expert(0, 1, &mut buf).unwrap();
        assert_eq!(view.gate().format, QuantFormat::Q4_K);
        assert_eq!(view.down().format, QuantFormat::Q6_K);
        // Slab sizes match the shape table: 256x256 rows of one block.
        assert_eq!(view.gate().bytes.len(), 256 * 144);
        assert_eq!(view.down().bytes.len(), 256 * 210);
    }

    #[test]
    fn tied_embeddings_reuse_the_embedding_as_lm_head() {
        let mut fx = build_install("model-tied");
        fx.manifest.arch.tie_embeddings = true;
        fx.manifest.common_tensors.remove("output.weight");
        write_manifest(&fx.root, &fx.manifest).unwrap();
        let model = Model::load(&fx.root, SKIP).unwrap();
        let head = model.lm_head();
        assert_eq!(head.format, QuantFormat::Q4_K);
        assert_eq!((head.in_dim, head.out_dim), (HIDDEN, VOCAB));
        assert_eq!(head.bytes.len(), VOCAB * 144);
        assert_eq!(head.row(0).unwrap(), model.embedding_row(0).unwrap());
    }
}
