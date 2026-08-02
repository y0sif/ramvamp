//! Synchronous expert blob reads.
//!
//! [`ExpertReader`] is the positioned-read path over the per-layer expert
//! files: one `pread` of `stride` bytes fetches exactly one expert blob
//! into a caller-owned buffer, and [`ExpertView`] carves the blob into its
//! gate/up/down projection slabs using the validated layout. Layer files
//! are opened lazily (handles cached per layer) and hashed on first open
//! unless [`LoadOptions::skip_hashes`] is set; the size check always runs.
//!
//! This is the portable baseline the io_uring + O_DIRECT streamer will sit
//! beside: same blob geometry, same validation, different submission path.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::format::{ExpertsLayout, Manifest, Projection, ProjectionName, sha256_file};
use crate::kernels::quants::QuantFormat;

use super::{IoError, LoadOptions, parse_quant_format, to_usize};

/// One projection slab inside a fetched expert blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlab<'a> {
    /// Packed quantized bytes of the slab.
    pub bytes: &'a [u8],
    /// Quantization format of those bytes.
    pub format: QuantFormat,
}

/// Byte range and format of one slab within a blob, resolved at
/// construction so per-read work is slicing only.
#[derive(Debug, Clone, Copy)]
struct SlabSpec {
    offset: usize,
    len: usize,
    format: QuantFormat,
}

/// One expert blob, fetched into a caller-owned buffer.
///
/// The three projection accessors return slices of that buffer at the
/// layout's offsets; the ranges were validated against the stride when the
/// reader was built, so slicing cannot fail.
#[derive(Debug)]
pub struct ExpertView<'a> {
    bytes: &'a [u8],
    /// Indexed as `[gate, up, down]`.
    slabs: [SlabSpec; 3],
}

impl<'a> ExpertView<'a> {
    /// SwiGLU gate projection slab.
    pub fn gate(&self) -> ExpertSlab<'a> {
        self.slab(ProjectionName::Gate)
    }

    /// Up projection slab.
    pub fn up(&self) -> ExpertSlab<'a> {
        self.slab(ProjectionName::Up)
    }

    /// Down projection slab.
    pub fn down(&self) -> ExpertSlab<'a> {
        self.slab(ProjectionName::Down)
    }

    /// Slab of the named projection.
    pub fn slab(&self, name: ProjectionName) -> ExpertSlab<'a> {
        let spec = self.slabs[name as usize];
        ExpertSlab {
            // In bounds by construction: projection end <= stride ==
            // bytes.len(), enforced by the layout validator and read_expert.
            bytes: &self.bytes[spec.offset..spec.offset + spec.len],
            format: spec.format,
        }
    }

    /// The whole blob, padding included.
    pub fn blob(&self) -> &'a [u8] {
        self.bytes
    }
}

/// Per-layer state: resolved geometry plus the lazily opened file handle.
#[derive(Debug)]
struct LayerFile {
    /// Absolute path of the layer file.
    path: PathBuf,
    /// Install-relative name, for error messages.
    name: String,
    stride: u64,
    stride_usize: usize,
    n_experts: u32,
    expected_size: u64,
    expected_sha256: String,
    slabs: [SlabSpec; 3],
    file: OnceLock<File>,
}

/// Lazy positioned-read access to every layer's expert file.
///
/// Construction validates the layout against the manifest and resolves the
/// gate/up/down slab geometry for every layer; all three projections must
/// be present with known quant formats. Files open on first use; the first
/// open size-checks the file and (unless skipped) hashes it against the
/// manifest. Concurrent first reads of one layer may both hash the file —
/// benign, one handle wins.
#[derive(Debug)]
pub struct ExpertReader {
    layers: Vec<LayerFile>,
    skip_hashes: bool,
}

impl ExpertReader {
    /// Build a reader over `<dir>`'s expert files.
    ///
    /// # Errors
    ///
    /// [`IoError::Format`] when the layout fails validation against the
    /// manifest; [`IoError::MissingProjection`] when a layer lacks gate,
    /// up, or down; [`IoError::UnknownQuant`] when a slab's quant name is
    /// not a weight format this build computes.
    pub fn new(
        dir: &Path,
        manifest: &Manifest,
        layout: &ExpertsLayout,
        options: LoadOptions,
    ) -> Result<Self, IoError> {
        layout.validate_against(manifest)?;
        let mut layers = Vec::with_capacity(layout.layers.len());
        for (index, layer) in layout.layers.iter().enumerate() {
            let layer_idx = index as u32;
            let find = |name: ProjectionName| -> Result<SlabSpec, IoError> {
                let projection = layer.projections.iter().find(|p| p.name == name).ok_or(
                    IoError::MissingProjection {
                        layer: layer_idx,
                        name,
                    },
                )?;
                slab_spec(layer_idx, projection)
            };
            let slabs = [
                find(ProjectionName::Gate)?,
                find(ProjectionName::Up)?,
                find(ProjectionName::Down)?,
            ];
            // Present and size-consistent per validate_against.
            let entry = &manifest.files[&layer.file];
            layers.push(LayerFile {
                path: dir.join(&layer.file),
                name: layer.file.clone(),
                stride: layer.stride,
                stride_usize: to_usize(layer.stride, "expert blob stride")?,
                n_experts: layer.n_experts,
                expected_size: entry.size,
                expected_sha256: entry.sha256.clone(),
                slabs,
                file: OnceLock::new(),
            });
        }
        Ok(Self {
            layers,
            skip_hashes: options.skip_hashes,
        })
    }

    /// Number of layers the reader serves.
    pub fn n_layers(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Blob stride of one layer in bytes, or `None` past the last layer.
    pub fn stride(&self, layer: u32) -> Option<u64> {
        self.layers.get(layer as usize).map(|l| l.stride)
    }

    /// Read one expert blob into `buf` (resized to the layer's stride) and
    /// return the view over its projection slabs.
    ///
    /// # Errors
    ///
    /// [`IoError::LayerOutOfRange`] / [`IoError::ExpertOutOfRange`] on bad
    /// indices; [`IoError::Format`] when the file fails its first-open
    /// size or hash check; [`IoError::Io`] when the read itself fails.
    pub fn read_expert<'buf>(
        &self,
        layer: u32,
        expert: u32,
        buf: &'buf mut Vec<u8>,
    ) -> Result<ExpertView<'buf>, IoError> {
        let state = self
            .layers
            .get(layer as usize)
            .ok_or(IoError::LayerOutOfRange {
                layer,
                n_layers: self.n_layers(),
            })?;
        if expert >= state.n_experts {
            return Err(IoError::ExpertOutOfRange {
                layer,
                expert,
                n_experts: state.n_experts,
            });
        }
        let file = self.layer_file(state)?;
        buf.resize(state.stride_usize, 0);
        // In bounds: expert < n_experts and file size == n_experts * stride.
        let offset = u64::from(expert) * state.stride;
        file.read_exact_at(buf, offset)
            .map_err(|e| IoError::io(&state.path, e))?;
        Ok(ExpertView {
            bytes: buf,
            slabs: state.slabs,
        })
    }

    /// The layer's cached file handle, opening and verifying on first use.
    fn layer_file<'a>(&self, state: &'a LayerFile) -> Result<&'a File, IoError> {
        if let Some(file) = state.file.get() {
            return Ok(file);
        }
        let file = File::open(&state.path).map_err(|e| IoError::io(&state.path, e))?;
        let size = file
            .metadata()
            .map_err(|e| IoError::io(&state.path, e))?
            .len();
        if size != state.expected_size {
            return Err(crate::format::FormatError::SizeMismatch {
                name: state.name.clone(),
                expected: state.expected_size,
                actual: size,
            }
            .into());
        }
        if !self.skip_hashes {
            let actual = sha256_file(&state.path)?;
            if !actual.eq_ignore_ascii_case(&state.expected_sha256) {
                return Err(crate::format::FormatError::HashMismatch {
                    name: state.name.clone(),
                    expected: state.expected_sha256.clone(),
                    actual,
                }
                .into());
            }
            tracing::debug!(
                file = state.name.as_str(),
                bytes = size,
                "layer file verified"
            );
        }
        Ok(state.file.get_or_init(|| file))
    }
}

/// Resolve one layout projection into a slab spec.
fn slab_spec(layer: u32, projection: &Projection) -> Result<SlabSpec, IoError> {
    let format = parse_quant_format(&projection.quant).ok_or_else(|| IoError::UnknownQuant {
        what: format!("layer {layer} {:?} projection", projection.name),
        quant: projection.quant.clone(),
    })?;
    Ok(SlabSpec {
        offset: to_usize(projection.offset_in_blob, "projection offset")?,
        len: to_usize(projection.len, "projection len")?,
        format,
    })
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{self, build_install};
    use super::*;
    use crate::format::FormatError;

    fn reader(fx: &testutil::Fixture, options: LoadOptions) -> ExpertReader {
        ExpertReader::new(&fx.root, &fx.manifest, &fx.layout, options).unwrap()
    }

    #[test]
    fn reads_expert_and_slabs_match_layout_offsets() {
        let fx = build_install("expert-slabs");
        let reader = reader(&fx, LoadOptions::default());
        assert_eq!(reader.n_layers(), 2);

        let mut buf = Vec::new();
        for (layer, expert) in [(0u32, 0u32), (0, 3), (1, 2)] {
            let view = reader.read_expert(layer, expert, &mut buf).unwrap();
            let layer_layout = &fx.layout.layers[layer as usize];
            assert_eq!(view.blob().len() as u64, layer_layout.stride);

            let file_bytes = std::fs::read(fx.root.join(&layer_layout.file)).unwrap();
            let base = (u64::from(expert) * layer_layout.stride) as usize;
            for projection in &layer_layout.projections {
                let slab = view.slab(projection.name);
                let start = base + projection.offset_in_blob as usize;
                let end = start + projection.len as usize;
                assert_eq!(slab.bytes, &file_bytes[start..end], "{:?}", projection.name);
                assert_eq!(slab.format, parse_quant_format(&projection.quant).unwrap());
            }
        }
        // Named accessors agree with slab-by-name.
        let view = reader.read_expert(0, 1, &mut buf).unwrap();
        assert_eq!(view.gate(), view.slab(ProjectionName::Gate));
        assert_eq!(view.up(), view.slab(ProjectionName::Up));
        assert_eq!(view.down(), view.slab(ProjectionName::Down));
        assert_eq!(view.down().format, QuantFormat::Q6_K);
        assert_eq!(view.gate().format, QuantFormat::Q4_K);
    }

    #[test]
    fn rejects_out_of_range_indices() {
        let fx = build_install("expert-bounds");
        let reader = reader(&fx, LoadOptions { skip_hashes: true });
        let mut buf = Vec::new();
        assert!(matches!(
            reader.read_expert(2, 0, &mut buf).unwrap_err(),
            IoError::LayerOutOfRange {
                layer: 2,
                n_layers: 2
            }
        ));
        assert!(matches!(
            reader.read_expert(0, 4, &mut buf).unwrap_err(),
            IoError::ExpertOutOfRange {
                layer: 0,
                expert: 4,
                n_experts: 4
            }
        ));
    }

    #[test]
    fn first_open_hash_check_toggles_with_skip_hashes() {
        let fx = build_install("expert-hash");
        let victim = fx.root.join(&fx.layout.layers[0].file);
        let mut data = std::fs::read(&victim).unwrap();
        data[5000] ^= 0xff;
        std::fs::write(&victim, data).unwrap();

        let strict = reader(&fx, LoadOptions::default());
        let mut buf = Vec::new();
        assert!(matches!(
            strict.read_expert(0, 0, &mut buf).unwrap_err(),
            IoError::Format(FormatError::HashMismatch { .. })
        ));
        // The untouched layer still reads.
        strict.read_expert(1, 0, &mut buf).unwrap();

        let lax = reader(&fx, LoadOptions { skip_hashes: true });
        lax.read_expert(0, 0, &mut buf).unwrap();
    }

    #[test]
    fn size_check_runs_even_with_hashes_skipped() {
        let fx = build_install("expert-size");
        let victim = fx.root.join(&fx.layout.layers[1].file);
        let data = std::fs::read(&victim).unwrap();
        std::fs::write(&victim, &data[..data.len() - 4096]).unwrap();
        let lax = reader(&fx, LoadOptions { skip_hashes: true });
        let mut buf = Vec::new();
        assert!(matches!(
            lax.read_expert(1, 0, &mut buf).unwrap_err(),
            IoError::Format(FormatError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_layout_missing_a_projection() {
        let fx = build_install("expert-missing-proj");
        let mut layout = fx.layout.clone();
        layout.layers[0]
            .projections
            .retain(|p| p.name != ProjectionName::Down);
        let err =
            ExpertReader::new(&fx.root, &fx.manifest, &layout, LoadOptions::default()).unwrap_err();
        assert!(matches!(
            err,
            IoError::MissingProjection {
                layer: 0,
                name: ProjectionName::Down
            }
        ));
    }

    #[test]
    fn rejects_unknown_slab_quant() {
        let fx = build_install("expert-bad-quant");
        let mut layout = fx.layout.clone();
        layout.layers[1].projections[0].quant = "f16".to_owned();
        let err =
            ExpertReader::new(&fx.root, &fx.manifest, &layout, LoadOptions::default()).unwrap_err();
        assert!(matches!(
            err,
            IoError::UnknownQuant { quant, .. } if quant == "f16"
        ));
    }
}
