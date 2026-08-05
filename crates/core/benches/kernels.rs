//! Warm-cache, single-thread microbenches for the quantized kernels:
//! scalar reference vs AVX2 dispatch for every dot kernel and both
//! activation quantizers, plus the decode-attention context sweep.
//!
//! Deliberately no criterion (keeps the dependency tree lean): a plain
//! `std::time::Instant` harness with warmup and median-of-31 runs. Numbers
//! from here are diagnostics per the experiment-log rules — published
//! end-to-end numbers come from cold runs inside the 3 GB cgroup, not from
//! this harness.
//!
//! Run with `cargo bench -p ramvamp-core`.

use std::time::Instant;

use ramvamp_core::kernels::attention::{AttentionScratch, decode_attention};
use ramvamp_core::kernels::quants::{
    BlockQ8_0, BlockQ8K, QuantFormat, avx2, f32_to_f16, quantize_row_q8_0, quantize_row_q8_k,
};
use ramvamp_core::kv::KvCache;

/// Deterministic 64-bit LCG (Knuth MMIX constants).
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform-ish f32 in [-1, 1) from the high bits.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// A row-major packed matrix of structurally valid random blocks: random
/// quant bytes with small, sane f16 scales planted at the format's scale
/// offsets (avoids inf/NaN/denormal scales that would distort FP timing).
fn synth_matrix(format: QuantFormat, in_dim: usize, out_dim: usize, rng: &mut Lcg) -> Vec<u8> {
    let scale_offs: &[usize] = match format {
        QuantFormat::Q4_K | QuantFormat::Q5_K => &[0, 2], // d, dmin
        QuantFormat::Q6_K => &[208],
        _ => &[0],
    };
    let block_bytes = format.block_bytes();
    let blocks = in_dim / format.block_weights() * out_dim;
    let mut w = vec![0u8; blocks * block_bytes];
    for block in w.chunks_exact_mut(block_bytes) {
        for b in block.iter_mut() {
            *b = (rng.next_u64() >> 32) as u8;
        }
        for &off in scale_offs {
            let d = f32_to_f16(0.01 + 0.02 * (rng.next_f32() + 1.0));
            block[off..off + 2].copy_from_slice(&d.to_le_bytes());
        }
    }
    w
}

/// A dispatched k-quant dot entry point.
type KDot = fn(&[u8], &[BlockQ8K], bool) -> Result<f32, ramvamp_core::kernels::KernelError>;

const WARMUP: usize = 5;
const RUNS: usize = 31;

/// Median wall time of `runs` timed calls after `warmup` untimed ones.
///
/// `runs` must be odd so the median is a real sample rather than an average
/// of two. Callers whose timed unit is long (the attention sweep) dial the
/// counts down instead of skipping ladder rungs.
fn median_ns_n<F: FnMut() -> f32>(runs: usize, warmup: usize, mut f: F) -> f64 {
    let mut sink = 0f32;
    for _ in 0..warmup {
        sink += f();
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        sink += f();
        samples.push(t.elapsed().as_nanos() as f64);
    }
    std::hint::black_box(sink);
    samples.sort_by(f64::total_cmp);
    samples[runs / 2]
}

/// Median wall time of `RUNS` timed calls after `WARMUP` untimed ones.
fn median_ns<F: FnMut() -> f32>(f: F) -> f64 {
    median_ns_n(RUNS, WARMUP, f)
}

struct Row {
    name: &'static str,
    shape: String,
    scalar_ns_per_row: f64,
    avx2_ns_per_row: f64,
    bytes_per_row: usize,
}

fn print_table(rows: &[Row]) {
    println!(
        "{:<16} {:<14} {:>13} {:>11} {:>13} {:>11} {:>9}",
        "kernel", "shape", "scalar ns/row", "scalar GB/s", "avx2 ns/row", "avx2 GB/s", "speedup"
    );
    for r in rows {
        let gbs = |ns: f64| r.bytes_per_row as f64 / ns;
        println!(
            "{:<16} {:<14} {:>13.0} {:>11.2} {:>13.0} {:>11.2} {:>8.2}x",
            r.name,
            r.shape,
            r.scalar_ns_per_row,
            gbs(r.scalar_ns_per_row),
            r.avx2_ns_per_row,
            gbs(r.avx2_ns_per_row),
            r.scalar_ns_per_row / r.avx2_ns_per_row,
        );
    }
}

/// Bench one dot kernel over a full matrix sweep; returns ns/row.
fn bench_dot_q8_k(
    dot: KDot,
    weight: &[u8],
    row_bytes: usize,
    acts: &[BlockQ8K],
    force_scalar: bool,
) -> f64 {
    let rows = weight.len() / row_bytes;
    median_ns(|| {
        let mut acc = 0f32;
        for row in weight.chunks_exact(row_bytes) {
            acc += dot(row, acts, force_scalar).unwrap();
        }
        acc
    }) / rows as f64
}

fn bench_dot_q8_0(weight: &[u8], row_bytes: usize, acts: &[BlockQ8_0], force_scalar: bool) -> f64 {
    let rows = weight.len() / row_bytes;
    median_ns(|| {
        let mut acc = 0f32;
        for row in weight.chunks_exact(row_bytes) {
            acc += avx2::vec_dot_q8_0_q8_0(row, acts, force_scalar).unwrap();
        }
        acc
    }) / rows as f64
}

// ---------------------------------------------------------------------------
// Decode attention vs context length.
// ---------------------------------------------------------------------------

/// v0 pin geometry, from `models/qwen3.rvmp/manifest.json`: 48 layers, 32
/// query heads over 4 kv heads (GQA group 8), head_dim 128 — so q_dim 4096
/// and kv_dim 512 — against the runtime's 4096-position context cap.
const ATTN_LAYERS: usize = 48;
const ATTN_Q_HEADS: usize = 32;
const ATTN_KV_HEADS: usize = 4;
const ATTN_HEAD_DIM: usize = 128;
const ATTN_CAP: usize = 4096;

/// Context lengths both arms sweep. The cache grows through the ladder in
/// place, so every rung is measured against the same allocation.
const CONTEXT_LADDER: [usize; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

/// `(runs, warmup)` per ladder rung for the single-layer arm: cheap enough
/// to keep the standard median-of-31 everywhere.
const ARM_A_SAMPLES: [(usize, usize); 7] = [(RUNS, WARMUP); 7];

/// `(runs, warmup)` per ladder rung for the 48-layer arm, aligned with
/// [`CONTEXT_LADDER`]. One timed unit there is a whole token's attention and
/// grows linearly with context (seconds at the top rung), so the sample count
/// comes down at the long end rather than the ladder losing rungs. Every
/// entry is still an odd run count after untimed warmups, as [`median_ns`]
/// does.
const ARM_B_SAMPLES: [(usize, usize); 7] =
    [(31, 3), (31, 3), (21, 2), (15, 2), (11, 1), (7, 1), (5, 1)];

/// One ladder rung of one arm.
struct AttnRow {
    /// Cached positions the kernel attended over.
    context: usize,
    /// Layers swept per timed unit (1 for arm A, 48 for arm B).
    layers: usize,
    /// Timed runs the median came from.
    runs: usize,
    /// Median wall nanoseconds per timed unit.
    ns_per_unit: f64,
}

impl AttnRow {
    /// Wall milliseconds for one token's worth of attention: arm B measures
    /// that directly, arm A is scaled by the model's layer count so the two
    /// arms sit in the same units.
    fn ms_per_token(&self) -> f64 {
        self.ns_per_unit * (ATTN_LAYERS / self.layers) as f64 / 1e6
    }

    /// Nanoseconds per cached position per timed unit. Flat across the
    /// ladder means the cost is linear in context.
    fn ns_per_pos(&self) -> f64 {
        self.ns_per_unit / self.context as f64
    }

    /// The same figure normalized to a single layer, so arm A and arm B are
    /// directly comparable.
    fn ns_per_pos_per_layer(&self) -> f64 {
        self.ns_per_pos() / self.layers as f64
    }

    /// Unique K+V bytes the timed unit walks: `layers * context * kv_dim`
    /// f16 elements across two planes.
    fn bytes(&self) -> usize {
        self.layers * self.context * ATTN_KV_HEADS * ATTN_HEAD_DIM * 2 * 2
    }
}

/// Sweep [`decode_attention`] over [`CONTEXT_LADDER`] on a `layers`-deep
/// cache, timing one call per layer per unit.
///
/// Arm A (`layers = 1`, 8 MiB of planes) is the pure kernel curve, mostly
/// cache-resident. Arm B (`layers = ATTN_LAYERS`, **384 MiB** of planes —
/// the KV tenant's whole v0 budget, so this arm alone dominates the bench's
/// footprint) makes one timed unit a full 48-layer sweep, i.e. exactly one
/// decoded token's attention: by the time layer `L` comes round again its
/// planes have not been touched since the previous unit, which is the
/// residency the real decode loop sees.
///
/// The query vector, the K/V fill and the scratch are all built outside the
/// timed region, and the scratch is preallocated at the context cap so it
/// never reallocates mid-measurement.
fn bench_attention_arm(
    layers: usize,
    schedule: &[(usize, usize); 7],
    rng: &mut Lcg,
) -> Vec<AttnRow> {
    let kv_dim = ATTN_KV_HEADS * ATTN_HEAD_DIM;
    let scale = 1.0 / (ATTN_HEAD_DIM as f32).sqrt();
    let mut cache = KvCache::new(layers, ATTN_KV_HEADS, ATTN_HEAD_DIM, ATTN_CAP).unwrap();
    let q: Vec<f32> = (0..ATTN_Q_HEADS * ATTN_HEAD_DIM)
        .map(|_| rng.next_f32())
        .collect();
    let mut out = vec![0.0f32; q.len()];
    let mut scratch = AttentionScratch::with_capacity(ATTN_CAP);
    let mut k_row = vec![0.0f32; kv_dim];
    let mut v_row = vec![0.0f32; kv_dim];

    let mut filled = 0usize;
    let mut rows = Vec::with_capacity(CONTEXT_LADDER.len());
    for (&context, &(runs, warmup)) in CONTEXT_LADDER.iter().zip(schedule) {
        // Grow the cache to this rung. Values are uniform in [-1, 1), which
        // keeps the QK scores near unit scale: no denormals, no softmax
        // overflow, and nothing value-dependent for the kernel to branch on.
        while filled < context {
            for layer in 0..layers {
                for (k, v) in k_row.iter_mut().zip(v_row.iter_mut()) {
                    *k = rng.next_f32();
                    *v = rng.next_f32();
                }
                cache.append(layer, &k_row, &v_row).unwrap();
            }
            filled += 1;
        }

        let ns_per_unit = median_ns_n(runs, warmup, || {
            let mut sink = 0f32;
            for layer in 0..layers {
                decode_attention(&q, &cache, layer, scale, &mut scratch, &mut out)
                    .expect("decode_attention");
                sink += out[0];
            }
            sink
        });
        rows.push(AttnRow {
            context,
            layers,
            runs,
            ns_per_unit,
        });
    }
    rows
}

fn print_attn_table(arm: &str, rows: &[AttnRow]) {
    println!("{arm}");
    println!(
        "{:>7} {:>5} {:>14} {:>10} {:>10} {:>14} {:>10}",
        "context", "runs", "ns/unit", "ms/token", "ns/pos", "ns/pos/layer", "eff GB/s"
    );
    for r in rows {
        println!(
            "{:>7} {:>5} {:>14.0} {:>10.3} {:>10.1} {:>14.3} {:>10.2}",
            r.context,
            r.runs,
            r.ns_per_unit,
            r.ms_per_token(),
            r.ns_per_pos(),
            r.ns_per_pos_per_layer(),
            r.bytes() as f64 / r.ns_per_unit,
        );
    }
}

/// `ns/pos` at context 4096 over `ns/pos` at context 512 — 1.0 means the
/// cost is exactly linear in context.
fn linearity_ratio(rows: &[AttnRow]) -> Option<f64> {
    let at = |context: usize| {
        rows.iter()
            .find(|r| r.context == context)
            .map(AttnRow::ns_per_pos)
    };
    Some(at(4096)? / at(512)?)
}

fn main() {
    println!(
        "kernels microbench: warm-cache, single-thread, median of {RUNS} runs \
         ({WARMUP} warmup); avx2+fma detected: {}",
        avx2::avx2_fma_available()
    );
    println!("GB/s = packed row bytes / ns per row (quantizers: f32 input bytes)\n");

    let mut rng = Lcg(0xBE7C);
    let out_dim = 2048usize;
    let mut rows = Vec::new();

    // Dot kernels: k-quants at the 2048-in shape plus the q6_k 768-in down
    // shape; q8_0 at 2048-in.
    let acts_2048 = {
        let x: Vec<f32> = (0..2048).map(|_| rng.next_f32() * 3.0).collect();
        let mut a = vec![BlockQ8K::default(); 2048 / 256];
        quantize_row_q8_k(&x, &mut a).unwrap();
        a
    };
    let acts_768 = {
        let x: Vec<f32> = (0..768).map(|_| rng.next_f32() * 3.0).collect();
        let mut a = vec![BlockQ8K::default(); 768 / 256];
        quantize_row_q8_k(&x, &mut a).unwrap();
        a
    };

    let k_cases: [(&'static str, QuantFormat, usize, KDot, &[BlockQ8K]); 4] = [
        (
            "q4_k x q8_k",
            QuantFormat::Q4_K,
            2048,
            avx2::vec_dot_q4_k_q8_k,
            &acts_2048,
        ),
        (
            "q5_k x q8_k",
            QuantFormat::Q5_K,
            2048,
            avx2::vec_dot_q5_k_q8_k,
            &acts_2048,
        ),
        (
            "q6_k x q8_k",
            QuantFormat::Q6_K,
            2048,
            avx2::vec_dot_q6_k_q8_k,
            &acts_2048,
        ),
        (
            "q6_k x q8_k",
            QuantFormat::Q6_K,
            768,
            avx2::vec_dot_q6_k_q8_k,
            &acts_768,
        ),
    ];
    for (name, format, in_dim, dot, acts) in k_cases {
        let row_bytes = format.row_bytes(in_dim).unwrap();
        let weight = synth_matrix(format, in_dim, out_dim, &mut rng);
        rows.push(Row {
            name,
            shape: format!("{in_dim}x{out_dim}"),
            scalar_ns_per_row: bench_dot_q8_k(dot, &weight, row_bytes, acts, true),
            avx2_ns_per_row: bench_dot_q8_k(dot, &weight, row_bytes, acts, false),
            bytes_per_row: row_bytes,
        });
    }

    {
        let in_dim = 2048;
        let row_bytes = QuantFormat::Q8_0.row_bytes(in_dim).unwrap();
        let weight = synth_matrix(QuantFormat::Q8_0, in_dim, out_dim, &mut rng);
        let x: Vec<f32> = (0..in_dim).map(|_| rng.next_f32() * 3.0).collect();
        let mut acts = vec![BlockQ8_0::default(); in_dim / 32];
        quantize_row_q8_0(&x, &mut acts).unwrap();
        rows.push(Row {
            name: "q8_0 x q8_0",
            shape: format!("{in_dim}x{out_dim}"),
            scalar_ns_per_row: bench_dot_q8_0(&weight, row_bytes, &acts, true),
            avx2_ns_per_row: bench_dot_q8_0(&weight, row_bytes, &acts, false),
            bytes_per_row: row_bytes,
        });
    }

    // Activation quantizers over one 2048-float row per run.
    {
        let n = 2048;
        let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0).collect();
        let mut out_k = vec![BlockQ8K::default(); n / 256];
        let scalar = median_ns(|| {
            avx2::quantize_row_q8_k(&x, &mut out_k, true).unwrap();
            out_k[0].d
        });
        let fast = median_ns(|| {
            avx2::quantize_row_q8_k(&x, &mut out_k, false).unwrap();
            out_k[0].d
        });
        rows.push(Row {
            name: "quantize q8_k",
            shape: format!("{n}"),
            scalar_ns_per_row: scalar,
            avx2_ns_per_row: fast,
            bytes_per_row: n * 4,
        });

        let mut out_0 = vec![BlockQ8_0::default(); n / 32];
        let scalar = median_ns(|| {
            avx2::quantize_row_q8_0(&x, &mut out_0, true).unwrap();
            out_0[0].d
        });
        let fast = median_ns(|| {
            avx2::quantize_row_q8_0(&x, &mut out_0, false).unwrap();
            out_0[0].d
        });
        rows.push(Row {
            name: "quantize q8_0",
            shape: format!("{n}"),
            scalar_ns_per_row: scalar,
            avx2_ns_per_row: fast,
            bytes_per_row: n * 4,
        });
    }

    print_table(&rows);

    // Decode attention against context length. Arm A runs and is dropped
    // before arm B allocates, so the 384 MiB arm is the only large tenant
    // alive at any moment.
    println!(
        "\ndecode attention vs context (v0 pin: {ATTN_Q_HEADS} q-heads : {ATTN_KV_HEADS} kv-heads, \
         head_dim {ATTN_HEAD_DIM}, scale 1/sqrt({ATTN_HEAD_DIM}), cap {ATTN_CAP})"
    );
    println!(
        "ms/token normalizes both arms to one token's {ATTN_LAYERS} layers; ns/pos = ns per timed \
         unit / context;\neff GB/s counts the unique K+V bytes of the walked planes \
         (layers * context * kv_dim * 2 B * 2 planes) — the\nkernel re-reads each kv head's \
         columns once per query head in its group, so the bytes it actually\ntouches are 8x this.\n"
    );

    let arm_a = bench_attention_arm(1, &ARM_A_SAMPLES, &mut rng);
    print_attn_table(
        "arm A: 1 layer (8 MiB planes), timed unit = one decode_attention call",
        &arm_a,
    );
    println!();
    let arm_b = bench_attention_arm(ATTN_LAYERS, &ARM_B_SAMPLES, &mut rng);
    print_attn_table(
        "arm B: 48 layers (384 MiB planes), timed unit = one token (all 48 layers)",
        &arm_b,
    );

    println!();
    for (arm, rows) in [("arm A", &arm_a), ("arm B", &arm_b)] {
        match linearity_ratio(rows) {
            Some(ratio) => println!(
                "linearity {arm}: ns/pos(4096) / ns/pos(512) = {ratio:.3}x \
                 (1.0 = cost linear in context)"
            ),
            None => println!("linearity {arm}: ladder is missing 512 or 4096, no ratio"),
        }
    }
}
