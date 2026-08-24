//! Inputs for the spike: one real Q4_K expert gate slab out of an installed
//! `.rvmp`, and a deterministic batch of Q8_K activation rows.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail};
use ramvamp_core::format::{ProjectionName, load_layout, load_manifest};
use ramvamp_core::kernels::quants::{BlockQ8K, quantize_row_q8_k};

/// Weights per k-quant super-block.
pub const QK_K: usize = 256;

/// One expert's packed gate slab plus the geometry needed to read it.
pub struct GateSlab {
    /// Packed Q4_K bytes, row-major, `out_dim` rows of `row_bytes`.
    pub bytes: Vec<u8>,
    /// Weights per row (the model's hidden size).
    pub in_dim: usize,
    /// Rows (the MoE intermediate size).
    pub out_dim: usize,
    /// Where the bytes came from, for the report.
    pub provenance: String,
    /// Whether these bytes are made up.
    ///
    /// A boolean rather than a substring test on [`Self::provenance`],
    /// because the verdict gates on it: a run on synthetic weights is not a
    /// run on a real expert slab, and the difference has to be machine
    /// checkable, not a matter of a reader noticing a banner eighty lines
    /// above the number.
    pub synthetic: bool,
}

/// Read expert `expert`'s gate slab out of layer `layer` of an installed
/// `.rvmp` directory.
///
/// The layout file gives the per-layer blob stride and the gate slab's
/// offset inside a blob, so one seek plus one read lands exactly on the
/// packed bytes. Nothing is materialized beyond the one slab.
///
/// `in_dim` is the caller's *expectation*, not the authority. It is checked
/// against `manifest.json`'s `arch.hidden`, and the derived row count against
/// `arch.moe_intermediate`, because the whole geometry here is otherwise
/// inferred from one number: point this at an install with a different hidden
/// size and `len / row_bytes` reads half-rows as rows, reports a plausible
/// shape, and still passes the numerics gate, since both arms then compute
/// the same wrong thing on the same wrong shape. Proving the bytes are real
/// is not the same as proving they are the right bytes, so the manifest is
/// read and the mismatch is fatal rather than silent.
pub fn load_gate_slab(
    install: &Path,
    layer: usize,
    expert: usize,
    in_dim: usize,
) -> Result<GateSlab> {
    let manifest = load_manifest(install).context("reading manifest.json")?;
    let hidden = manifest.arch.hidden as usize;
    if hidden != in_dim {
        bail!(
            "this spike is built for a hidden size of {in_dim} (one weight row) and \
             {}'s manifest declares arch.hidden = {hidden}. Reading it anyway would derive \
             the row count from the wrong row length and report a plausible geometry that \
             is not the model's",
            manifest.model_id
        );
    }
    if !hidden.is_multiple_of(QK_K) {
        bail!("arch.hidden = {hidden} is not a whole number of {QK_K}-weight super-blocks");
    }

    let layout = load_layout(install).context("reading experts/layout.json")?;
    if layout.layers.len() != manifest.arch.n_layers as usize {
        bail!(
            "experts/layout.json has {} layers and the manifest declares {}; the two \
             describe different installs",
            layout.layers.len(),
            manifest.arch.n_layers
        );
    }
    let layer_layout = layout
        .layers
        .get(layer)
        .with_context(|| format!("layer {layer} is past the end of the layout"))?;
    if layer_layout.n_experts != manifest.arch.n_experts {
        bail!(
            "layer {layer}'s layout declares {} experts and the manifest declares {}",
            layer_layout.n_experts,
            manifest.arch.n_experts
        );
    }
    if expert >= layer_layout.n_experts as usize {
        bail!(
            "expert {expert} is past the layer's {} experts",
            layer_layout.n_experts
        );
    }
    let gate = layer_layout
        .projections
        .iter()
        .find(|p| p.name == ProjectionName::Gate)
        .with_context(|| format!("layer {layer} has no gate projection"))?;
    if gate.quant != "q4_k" {
        bail!(
            "layer {layer} gate is {:?}, this spike only implements q4_k",
            gate.quant
        );
    }

    let row_bytes = in_dim / QK_K * 144;
    let len = usize::try_from(gate.len).context("gate slab length does not fit in usize")?;
    // A zero-length slab divides evenly by any row size, so the divisibility
    // check below would pass it through with `out_dim == 0`: the caller then
    // divides by zero to print bytes per row, and `Buffer::new` would create
    // a zero-size Vulkan buffer, which VUID-VkBufferCreateInfo-size-00912
    // forbids outright.
    if len == 0 {
        bail!("layer {layer} expert {expert} gate slab is zero bytes long");
    }
    if !len.is_multiple_of(row_bytes) {
        bail!("gate slab of {len} B is not a whole number of {row_bytes} B rows");
    }
    let out_dim = len / row_bytes;
    // The row count the bytes imply, against the row count the model
    // declares. These agreeing is what makes `out_dim` the MoE intermediate
    // size rather than a number that happens to divide the slab.
    let moe_intermediate = manifest.arch.moe_intermediate as usize;
    if out_dim != moe_intermediate {
        bail!(
            "layer {layer} expert {expert}'s gate slab is {len} B, which is {out_dim} rows of \
             {row_bytes} B, and the manifest declares arch.moe_intermediate = \
             {moe_intermediate}"
        );
    }

    let path = install.join(&layer_layout.file);
    let offset = layer_layout
        .stride
        .checked_mul(expert as u64)
        .and_then(|base| base.checked_add(gate.offset_in_blob))
        .context("expert blob offset overflows")?;

    let mut file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .with_context(|| format!("seeking to byte {offset} of {}", path.display()))?;
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)
        .with_context(|| format!("reading {len} B of {}", path.display()))?;

    Ok(GateSlab {
        bytes,
        in_dim,
        out_dim,
        provenance: format!(
            "REAL {} layer {layer} expert {expert} gate (q4_k, {len} B at byte {offset}, \
             {out_dim}x{in_dim} checked against {}'s arch.moe_intermediate/arch.hidden)",
            path.display(),
            manifest.model_id
        ),
        synthetic: false,
    })
}

/// Synthetic stand-in for [`load_gate_slab`], used only when no install is
/// present. Reported as such: quantized bytes carry structure, and a result
/// measured on noise is not the same claim.
pub fn synthetic_gate_slab(in_dim: usize, out_dim: usize) -> GateSlab {
    let row_bytes = in_dim / QK_K * 144;
    let mut rng = Rng::new(0x5EED_0BAD_C0FF_EE01);
    let mut bytes = vec![0u8; out_dim * row_bytes];
    for block in bytes.chunks_exact_mut(144) {
        // Plausible f16 d/dmin (small positive normals) so the affine
        // reconstruction stays in a sane range.
        let d = ramvamp_core::kernels::quants::f32_to_f16(0.01 + rng.unit() * 0.02);
        let dmin = ramvamp_core::kernels::quants::f32_to_f16(0.005 + rng.unit() * 0.01);
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        for b in &mut block[4..] {
            *b = (rng.next() >> 24) as u8;
        }
    }
    GateSlab {
        bytes,
        in_dim,
        out_dim,
        provenance: format!("SYNTHETIC q4_k bytes ({out_dim}x{in_dim}), NOT A REAL EXPERT SLAB"),
        synthetic: true,
    }
}

/// Quantize `n_acts` deterministic pseudo-random activation rows to Q8_K.
///
/// Returns the flat concatenation `gemv_q8_k_batched` wants: `n_acts` rows
/// of `in_dim / 256` blocks each, activation-row-major.
pub fn build_acts(n_acts: usize, in_dim: usize, seed: u64) -> Result<Vec<BlockQ8K>> {
    let blocks_per_row = in_dim / QK_K;
    let mut rng = Rng::new(seed);
    let mut out = vec![BlockQ8K::default(); n_acts * blocks_per_row];
    let mut row = vec![0f32; in_dim];
    for a in 0..n_acts {
        // Roughly unit-variance post-RMSNorm-looking values.
        for v in &mut row {
            *v = (rng.unit() - 0.5) * 4.0;
        }
        quantize_row_q8_k(&row, &mut out[a * blocks_per_row..(a + 1) * blocks_per_row])
            .context("quantizing an activation row to q8_k")?;
    }
    Ok(out)
}

/// splitmix64: fixed seed in, reproducible stream out.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}
