//! Serde types and validation for `experts/layout.json` (schema v1).
//!
//! Each layer's expert file is an array of `n_experts` fixed-stride blobs;
//! blob `i` starts at byte `i * stride`, so one O_DIRECT read of `stride`
//! bytes fetches exactly one expert. Within a blob, the gate/up/down
//! projection slabs sit at 4 KiB-aligned offsets. Stride is uniform within
//! a layer but may differ across layers: in a Q4_K_M source, layers whose
//! `down` projection is Q6_K pack a wider blob than pure-Q4_K layers.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::manifest::check_file_name;
use super::{EXPERT_BLOB_ALIGN, FormatError, Manifest};

/// `experts/layout.json`: one entry per transformer layer, in layer order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertsLayout {
    /// Per-layer blob geometry, index = layer index.
    pub layers: Vec<LayerLayout>,
}

/// Blob geometry for one layer's expert file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerLayout {
    /// Install-relative expert file, e.g. `experts/layer_00.bin`.
    pub file: String,
    /// Bytes per expert blob; a multiple of [`EXPERT_BLOB_ALIGN`].
    pub stride: u64,
    /// Number of blobs in the file; must match `arch.n_experts`.
    pub n_experts: u32,
    /// Projection slabs within each blob.
    pub projections: Vec<Projection>,
}

/// One projection slab inside an expert blob.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Projection {
    /// Which FFN projection this slab is.
    pub name: ProjectionName,
    /// Byte offset within the blob; [`EXPERT_BLOB_ALIGN`]-aligned.
    pub offset_in_blob: u64,
    /// Byte length of the packed slab.
    pub len: u64,
    /// Quantized type of the slab, e.g. `q4_k`, `q6_k`.
    pub quant: String,
}

/// FFN projection identity; serialized as `"gate"` / `"up"` / `"down"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectionName {
    /// SwiGLU gate projection.
    Gate,
    /// Up projection.
    Up,
    /// Down projection (Q6_K on some layers of a Q4_K_M source).
    Down,
}

impl ExpertsLayout {
    /// Validate the layout's intrinsic invariants: safe file names, nonzero
    /// 4 KiB-multiple strides, and per-layer projections that are unique,
    /// 4 KiB-aligned, non-overlapping, and contained within the stride.
    pub fn validate(&self) -> Result<(), FormatError> {
        if self.layers.is_empty() {
            return Err(FormatError::InvalidLayout("no layers".to_owned()));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            layer.validate(index)?;
        }
        Ok(())
    }

    /// [`ExpertsLayout::validate`] plus cross-checks against the manifest:
    /// layer count matches `arch.n_layers`, per-layer `n_experts` matches
    /// `arch.n_experts`, every referenced file has a unique `files` entry,
    /// and each entry's size is exactly `n_experts * stride`.
    pub fn validate_against(&self, manifest: &Manifest) -> Result<(), FormatError> {
        self.validate()?;
        let n_layers = manifest.arch.n_layers as usize;
        if self.layers.len() != n_layers {
            return Err(FormatError::InvalidLayout(format!(
                "layout has {} layers, arch has {n_layers}",
                self.layers.len()
            )));
        }
        let mut seen_files = BTreeSet::new();
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.n_experts != manifest.arch.n_experts {
                return Err(FormatError::InvalidLayout(format!(
                    "layer {index}: n_experts {} does not match arch.n_experts {}",
                    layer.n_experts, manifest.arch.n_experts
                )));
            }
            if !seen_files.insert(layer.file.as_str()) {
                return Err(FormatError::InvalidLayout(format!(
                    "layer {index}: file {:?} already used by an earlier layer",
                    layer.file
                )));
            }
            let entry = manifest
                .files
                .get(&layer.file)
                .ok_or_else(|| FormatError::MissingFileEntry(layer.file.clone()))?;
            let expected = layer
                .stride
                .checked_mul(u64::from(layer.n_experts))
                .ok_or_else(|| {
                    FormatError::InvalidLayout(format!(
                        "layer {index}: n_experts * stride overflows"
                    ))
                })?;
            if entry.size != expected {
                return Err(FormatError::InvalidLayout(format!(
                    "layer {index} ({:?}): files entry says {} bytes, n_experts * stride is {expected}",
                    layer.file, entry.size
                )));
            }
        }
        Ok(())
    }
}

impl LayerLayout {
    fn validate(&self, index: usize) -> Result<(), FormatError> {
        check_file_name(&self.file)?;
        if self.stride == 0 {
            return Err(FormatError::InvalidLayout(format!(
                "layer {index}: stride is zero"
            )));
        }
        if self.stride % EXPERT_BLOB_ALIGN != 0 {
            return Err(FormatError::Misaligned {
                what: "expert blob stride",
                name: self.file.clone(),
                value: self.stride,
                align: EXPERT_BLOB_ALIGN,
            });
        }
        if self.n_experts == 0 {
            return Err(FormatError::InvalidLayout(format!(
                "layer {index}: n_experts is zero"
            )));
        }
        if self.projections.is_empty() {
            return Err(FormatError::InvalidLayout(format!(
                "layer {index}: no projections"
            )));
        }
        let mut seen = [false; 3];
        let mut ranges: Vec<(u64, u64)> = Vec::with_capacity(self.projections.len());
        for projection in &self.projections {
            let slot = projection.name as usize;
            if seen[slot] {
                return Err(FormatError::InvalidLayout(format!(
                    "layer {index}: duplicate {:?} projection",
                    projection.name
                )));
            }
            seen[slot] = true;
            if projection.len == 0 || projection.quant.is_empty() {
                return Err(FormatError::InvalidLayout(format!(
                    "layer {index}: {:?} projection has a zero length or empty quant",
                    projection.name
                )));
            }
            if projection.offset_in_blob % EXPERT_BLOB_ALIGN != 0 {
                return Err(FormatError::Misaligned {
                    what: "projection offset",
                    name: format!("layer {index} {:?}", projection.name),
                    value: projection.offset_in_blob,
                    align: EXPERT_BLOB_ALIGN,
                });
            }
            let end = projection
                .offset_in_blob
                .checked_add(projection.len)
                .ok_or_else(|| {
                    FormatError::InvalidLayout(format!(
                        "layer {index}: {:?} projection offset + len overflows",
                        projection.name
                    ))
                })?;
            if end > self.stride {
                return Err(FormatError::InvalidLayout(format!(
                    "layer {index}: {:?} projection ends at byte {end}, past stride {}",
                    projection.name, self.stride
                )));
            }
            ranges.push((projection.offset_in_blob, end));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(FormatError::InvalidLayout(format!(
                    "layer {index}: projections overlap"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{sample_layout, sample_manifest};
    use super::*;

    #[test]
    fn serde_roundtrip() {
        let layout = sample_layout();
        let json = serde_json::to_string_pretty(&layout).unwrap();
        let back: ExpertsLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(back, layout);
    }

    #[test]
    fn projection_names_serialize_lowercase() {
        let json = serde_json::to_string(&ProjectionName::Down).unwrap();
        assert_eq!(json, r#""down""#);
        let back: ProjectionName = serde_json::from_str(r#""gate""#).unwrap();
        assert_eq!(back, ProjectionName::Gate);
    }

    #[test]
    fn validates_sample_standalone_and_against_manifest() {
        let layout = sample_layout();
        layout.validate().unwrap();
        layout.validate_against(&sample_manifest()).unwrap();
    }

    #[test]
    fn rejects_stride_not_page_multiple() {
        let mut layout = sample_layout();
        layout.layers[0].stride = 12000;
        assert!(matches!(
            layout.validate().unwrap_err(),
            FormatError::Misaligned {
                what: "expert blob stride",
                value: 12000,
                ..
            }
        ));
    }

    #[test]
    fn rejects_unaligned_projection_offset() {
        let mut layout = sample_layout();
        layout.layers[1].projections[1].offset_in_blob = 4100;
        assert!(matches!(
            layout.validate().unwrap_err(),
            FormatError::Misaligned {
                what: "projection offset",
                value: 4100,
                ..
            }
        ));
    }

    #[test]
    fn rejects_projection_past_stride() {
        let mut layout = sample_layout();
        layout.layers[0].projections[2].len = 5000; // 8192 + 5000 > 12288
        assert!(matches!(
            layout.validate().unwrap_err(),
            FormatError::InvalidLayout(_)
        ));
    }

    #[test]
    fn rejects_overlapping_projections() {
        let mut layout = sample_layout();
        layout.layers[0].projections[0].len = 4097; // gate spills into up's slab
        assert!(matches!(
            layout.validate().unwrap_err(),
            FormatError::InvalidLayout(_)
        ));
    }

    #[test]
    fn rejects_duplicate_projection_name() {
        let mut layout = sample_layout();
        layout.layers[0].projections[1].name = ProjectionName::Gate;
        assert!(matches!(
            layout.validate().unwrap_err(),
            FormatError::InvalidLayout(_)
        ));
    }

    #[test]
    fn cross_check_rejects_layer_count_mismatch() {
        let mut layout = sample_layout();
        layout.layers.pop();
        assert!(matches!(
            layout.validate_against(&sample_manifest()).unwrap_err(),
            FormatError::InvalidLayout(_)
        ));
    }

    #[test]
    fn cross_check_rejects_file_size_mismatch() {
        let mut manifest = sample_manifest();
        manifest.files.get_mut("experts/layer_00.bin").unwrap().size -= 1;
        assert!(matches!(
            sample_layout().validate_against(&manifest).unwrap_err(),
            FormatError::InvalidLayout(_)
        ));
    }

    #[test]
    fn cross_check_rejects_missing_file_entry() {
        let mut layout = sample_layout();
        layout.layers[1].file = "experts/other.bin".to_owned();
        assert!(matches!(
            layout.validate_against(&sample_manifest()).unwrap_err(),
            FormatError::MissingFileEntry(_)
        ));
    }

    #[test]
    fn cross_check_rejects_expert_count_mismatch() {
        let mut layout = sample_layout();
        layout.layers[0].n_experts = 8;
        assert!(matches!(
            layout.validate_against(&sample_manifest()).unwrap_err(),
            FormatError::InvalidLayout(_)
        ));
    }
}
