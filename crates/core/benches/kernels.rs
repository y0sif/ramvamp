//! Warm-cache, single-thread microbenches for the quantized kernels:
//! scalar reference vs AVX2 dispatch for every dot kernel and both
//! activation quantizers, plus the decode-attention context sweep.
//!
//! Deliberately no criterion (keeps the dependency tree lean): a plain
//! `std::time::Instant` harness with untimed warmups and an odd-count median.
//! The dot and quantizer table runs 31/5 everywhere; the attention arms dial
//! the counts down as one timed unit grows into the seconds, and print the
//! `runs`/`warmup` they actually used per rung. Numbers from here are
//! diagnostics per the experiment-log rules — published end-to-end numbers
//! come from cold runs inside the 3 GB cgroup, not from this harness.
//!
//! Run with `cargo bench -p ramvamp-core`.

use std::time::Instant;

use ramvamp_core::kernels::attention::{AttentionScratch, decode_attention, scratch_len};
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

/// Ladder rungs re-measured at the end of each arm as a **time-into-run
/// control**. Must be a subset of [`CONTEXT_LADDER`], ascending.
///
/// The ladder is walked strictly ascending, so "slower at the top rung" and
/// "slower later in the run" are perfectly confounded: a 48-layer arm is
/// ~44 s of continuous single-core FP work, which is the same order as this
/// laptop part's PL1 time constant, and a thermal/DVFS ramp has the same
/// sign and monotonicity as a residency effect. Re-running a cheap early
/// rung *after* the expensive ones separates them: if the repeat matches the
/// original, the clock held and the ladder's drift is footprint; if the
/// repeat is materially slower at identical context and identical read set,
/// the drift is time-into-run and the residency reading is contaminated.
///
/// The repeat is set up to touch exactly the bytes the original rung touched:
/// the planes are allocated at [`ATTN_CAP`] up front and never move, so
/// [`KvCache::clear`] plus a refill to `context` leaves the kernel walking the
/// same leading rows of the same allocation. The *addresses* are identical; the
/// *values* are not, because the refill draws fresh f16s from the same RNG
/// stream rather than replaying the old ones. Nothing in this kernel branches
/// on a value and both fills are uniform in [-1, 1), so the comparison holds —
/// but "same read set" here means same footprint, not byte-for-byte replay.
const DRIFT_RUNGS: [usize; 2] = [64, 512];

/// One ladder rung of one arm.
struct AttnRow {
    /// Cached positions the kernel attended over.
    context: usize,
    /// Layers swept per timed unit (1 for arm A, 48 for arm B).
    layers: usize,
    /// Timed runs the median came from.
    runs: usize,
    /// Untimed warmup units run before the timed ones.
    warmup: usize,
    /// Median wall nanoseconds per timed unit.
    ns_per_unit: f64,
}

impl AttnRow {
    /// Wall milliseconds for one token's worth of attention: arm B measures
    /// that directly, arm A is scaled by the model's layer count so the two
    /// arms sit in the same units.
    ///
    /// The layer ratio is taken in f64: as integer division it is silently
    /// correct only for `layers` in {1, 48} and under-reports by 32x at,
    /// say, `layers = 32`.
    fn ms_per_token(&self) -> f64 {
        self.ns_per_unit * (ATTN_LAYERS as f64 / self.layers as f64) / 1e6
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

    /// K+V bytes the timed unit walks: `layers * context * kv_dim` f16
    /// elements across two planes.
    ///
    /// This is both the *unique* byte count and the *touched* byte count **at
    /// this bench's geometry**, and the two have been the same number since
    /// 8fc3c4a made the kernel kv-head-outer: each K and V head slice is
    /// widened from f16 once per position and reused across that head's whole
    /// GQA group. The pre-8fc3c4a kernel re-read every kv head's columns once
    /// per query head, so its touched count was `group` (8) times this one.
    /// Anything derived from `eff GB/s` — the "0.17 GB/s, therefore
    /// compute-bound" reading in particular — has to use this figure as printed
    /// and must not be scaled by the group.
    ///
    /// The equality is `group`-dependent on the AVX2 path and is not a general
    /// claim about the kernel. `x86::qk_scores` walks the GQA group in chunks
    /// of eight query heads with the position sweep *inside* that loop, so K is
    /// widened `ceil(group / 8)` times per kv head — 1x for `group` 1-8, 2x for
    /// 9-16, 3x for 17-24, 4x for 25-32 (counted directly in `x86::widen_rows`);
    /// V is 1x always, and the scalar path is 1x always. This bench is pinned to
    /// [`ATTN_Q_HEADS`]:[`ATTN_KV_HEADS`] = 32:4, i.e. `group = 8`, exactly
    /// where the factor is 1 — which is why "unique == touched" holds here. A
    /// future arm at a wider group would have to scale the K half of this
    /// figure by `ceil(group / 8)` before calling it a touched-byte count.
    fn bytes(&self) -> usize {
        self.layers * self.context * ATTN_KV_HEADS * ATTN_HEAD_DIM * 2 * 2
    }
}

/// One [`DRIFT_RUNGS`] rung measured twice: once in ladder order, once again
/// after the whole arm has run.
struct DriftRow {
    /// Cached positions, identical in both measurements.
    context: usize,
    /// Timed runs, identical in both measurements.
    runs: usize,
    /// Untimed warmups, identical in both measurements.
    warmup: usize,
    /// The ladder's original median for this rung.
    first_ns: f64,
    /// The end-of-arm repeat's median for the same rung.
    repeat_ns: f64,
}

impl DriftRow {
    /// `repeat / first`. Above 1.0, the same work at the same context got
    /// slower purely by running later — a clock/thermal effect, not a
    /// footprint effect.
    fn ratio(&self) -> f64 {
        self.repeat_ns / self.first_ns
    }
}

/// Everything one timed attention unit needs except the cache, so the ladder
/// and the drift control time byte-identical closures.
struct AttnUnit<'a> {
    layers: usize,
    q: &'a [f32],
    scale: f32,
    scratch: AttentionScratch,
    out: Vec<f32>,
}

impl AttnUnit<'_> {
    /// Median wall ns for one timed unit: `layers` back-to-back
    /// [`decode_attention`] calls over `cache`.
    ///
    /// Both `black_box`es are load-bearing, not decoration. `black_box(&out)`
    /// makes the whole output buffer escape, so none of the 4096 output
    /// stores (nor the 32 softmaxes behind them) can be proved dead — the
    /// root manifest builds benches with `lto = "thin"` and
    /// `codegen-units = 1`, so `decode_attention` is inlinable across the
    /// crate boundary and a version cheap enough to inline is a version
    /// whose dead stores LLVM can see. Observing `out[0]` alone would leave
    /// 31 of 32 heads eliminable, which would read as a speedup. Summing the
    /// buffer instead would also work but costs a serial 4096-add chain per
    /// call: ~1.7% at the shortest arm-A rung and ~0.03% at the longest, a
    /// per-call constant that would tilt the linearity probe. `black_box` on
    /// the reference is free.
    ///
    /// `black_box(q)` and `black_box(cache)` stop the inputs from being
    /// treated as loop-invariant across the `layers` calls. In the real
    /// decode loop `q` is different at every layer, so this is also the
    /// faithful shape.
    fn median_ns(&mut self, cache: &KvCache, runs: usize, warmup: usize) -> f64 {
        let layers = self.layers;
        let q = self.q;
        let scale = self.scale;
        let scratch = &mut self.scratch;
        let out = &mut self.out;
        median_ns_n(runs, warmup, || {
            let mut sink = 0f32;
            for layer in 0..layers {
                decode_attention(
                    std::hint::black_box(q),
                    std::hint::black_box(cache),
                    layer,
                    scale,
                    scratch,
                    out,
                )
                .expect("decode_attention");
                sink += std::hint::black_box(&out[..])[0];
            }
            sink
        })
    }
}

/// Append positions to every layer until each cursor reaches `context`.
///
/// Values are uniform in [-1, 1), which keeps the QK scores near unit scale:
/// no denormals, no softmax overflow, and nothing value-dependent for the
/// kernel to branch on.
fn fill_to(
    cache: &mut KvCache,
    layers: usize,
    filled: &mut usize,
    context: usize,
    k_row: &mut [f32],
    v_row: &mut [f32],
    rng: &mut Lcg,
) {
    while *filled < context {
        for layer in 0..layers {
            for (k, v) in k_row.iter_mut().zip(v_row.iter_mut()) {
                *k = rng.next_f32();
                *v = rng.next_f32();
            }
            cache.append(layer, k_row, v_row).unwrap();
        }
        *filled += 1;
    }
}

/// Sweep [`decode_attention`] over [`CONTEXT_LADDER`] on a `layers`-deep
/// cache, timing one call per layer per unit, then re-measure
/// [`DRIFT_RUNGS`] at the end as a time-into-run control.
///
/// Arm A (`layers = 1`, 8 MiB of planes) is the pure kernel curve, mostly
/// cache-resident. Arm B (`layers = ATTN_LAYERS`, **384 MiB** of planes —
/// the KV tenant's whole v0 budget, so this arm alone dominates the bench's
/// footprint) makes one timed unit a full 48-layer sweep, i.e. exactly one
/// decoded token's attention.
///
/// What arm B is and is not: it is a **lower bound on the eviction the real
/// decode loop inflicts between two attention calls on the same layer**, not
/// a reproduction of it. Between consecutive `decode_attention` calls the
/// real loop runs that layer's four quantized GEMVs, an f32 router matvec
/// and `stream_experts` staging 8 experts x 3 `[2048, 768]` matrices — tens
/// of MiB of streamed weights against the ~8 MiB of KV one layer's attention
/// touches at the top rung. Arm B runs 48 attention calls back to back and
/// nothing else, so it evicts strictly less. The bound is close to tight at
/// large `context`, where the 384 MiB of planes alone overrun every cache;
/// below roughly `context = 1024` the whole live plane set fits in L3 and
/// arm B is measuring a cache-resident sweep that the real loop never gets.
/// Read its residency tax as a floor, and only at the long end.
///
/// The query vector, the K/V fill and the scratch are all built outside the
/// timed region, and the scratch is reserved at the **whole carve**
/// [`scratch_len`] describes for this geometry at [`ATTN_CAP`], so it never
/// reallocates — not in the timed region and not in warmup either. Reserving
/// `ATTN_CAP` alone would be an 8x under-reserve (the carve is `group * cap +
/// 2 * head_dim` = 33,024 f32, not 4,096), which is verbatim the mistake
/// `with_capacity`'s own doc warns about; the growth would land in warmup and
/// corrupt no published number, but the sentence claiming it cannot happen
/// would still be false.
fn bench_attention_arm(
    layers: usize,
    schedule: &[(usize, usize); 7],
    rng: &mut Lcg,
) -> (Vec<AttnRow>, Vec<DriftRow>) {
    let kv_dim = ATTN_KV_HEADS * ATTN_HEAD_DIM;
    let scale = 1.0 / (ATTN_HEAD_DIM as f32).sqrt();
    let mut cache = KvCache::new(layers, ATTN_KV_HEADS, ATTN_HEAD_DIM, ATTN_CAP).unwrap();
    let q: Vec<f32> = (0..ATTN_Q_HEADS * ATTN_HEAD_DIM)
        .map(|_| rng.next_f32())
        .collect();
    let mut unit = AttnUnit {
        layers,
        q: &q,
        scale,
        scratch: AttentionScratch::with_capacity(scratch_len(
            ATTN_Q_HEADS,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            ATTN_CAP,
        )),
        out: vec![0.0f32; ATTN_Q_HEADS * ATTN_HEAD_DIM],
    };
    let mut k_row = vec![0.0f32; kv_dim];
    let mut v_row = vec![0.0f32; kv_dim];

    let mut filled = 0usize;
    let mut rows = Vec::with_capacity(CONTEXT_LADDER.len());
    for (&context, &(runs, warmup)) in CONTEXT_LADDER.iter().zip(schedule) {
        fill_to(
            &mut cache,
            layers,
            &mut filled,
            context,
            &mut k_row,
            &mut v_row,
            rng,
        );
        let ns_per_unit = unit.median_ns(&cache, runs, warmup);
        rows.push(AttnRow {
            context,
            layers,
            runs,
            warmup,
            ns_per_unit,
        });
    }

    // Time-into-run control. `clear` only rewinds the per-layer cursors —
    // the planes stay put — so refilling to an early rung leaves the kernel
    // reading the same leading rows of the same allocation it read the first
    // time, with the same call, the same sample counts and the same geometry.
    // The only difference between the two measurements is when they ran.
    cache.clear();
    filled = 0;
    let mut drift = Vec::with_capacity(DRIFT_RUNGS.len());
    for &context in &DRIFT_RUNGS {
        let Some((runs, warmup, first_ns)) = rows
            .iter()
            .find(|r| r.context == context)
            .map(|r| (r.runs, r.warmup, r.ns_per_unit))
        else {
            continue;
        };
        fill_to(
            &mut cache,
            layers,
            &mut filled,
            context,
            &mut k_row,
            &mut v_row,
            rng,
        );
        let repeat_ns = unit.median_ns(&cache, runs, warmup);
        drift.push(DriftRow {
            context,
            runs,
            warmup,
            first_ns,
            repeat_ns,
        });
    }

    (rows, drift)
}

fn print_attn_table(arm: &str, rows: &[AttnRow]) {
    println!("{arm}");
    println!(
        "{:>7} {:>5} {:>7} {:>14} {:>10} {:>10} {:>14} {:>10}",
        "context", "runs", "warmup", "ns/unit", "ms/token", "ns/pos", "ns/pos/layer", "eff GB/s"
    );
    for r in rows {
        println!(
            "{:>7} {:>5} {:>7} {:>14.0} {:>10.3} {:>10.1} {:>14.3} {:>10.2}",
            r.context,
            r.runs,
            r.warmup,
            r.ns_per_unit,
            r.ms_per_token(),
            r.ns_per_pos(),
            r.ns_per_pos_per_layer(),
            r.bytes() as f64 / r.ns_per_unit,
        );
    }
}

fn print_drift_table(arm: &str, drift: &[DriftRow]) {
    println!(
        "drift control {arm}: early rungs re-measured after the whole arm, same context, \
         same sample counts,\nsame ADDRESSES — `clear` rewinds the cursors and the planes stay \
         put, so the repeat walks the same\nbytes of the same allocation. NOT the same VALUES: \
         the refill draws fresh RNG f16s into those\nrows, so the repeat attends a different \
         random K/V. That is immaterial to this kernel (no data-\ndependent branching, no \
         denormals, uniform [-1,1) either way) but it is not a byte-for-byte replay.\nratio > 1 \
         means the same work at the same footprint got slower by running later \
         (clock/thermal),\nwhich is the alternative explanation for the ladder's own upward \
         ns/pos drift."
    );
    println!(
        "{:>7} {:>5} {:>7} {:>14} {:>14} {:>8}",
        "context", "runs", "warmup", "first ns/unit", "repeat ns/unit", "ratio"
    );
    for d in drift {
        println!(
            "{:>7} {:>5} {:>7} {:>14.0} {:>14.0} {:>7.3}x",
            d.context,
            d.runs,
            d.warmup,
            d.first_ns,
            d.repeat_ns,
            d.ratio(),
        );
    }
}

/// `max(ns/pos) / min(ns/pos)` over every measured rung — 1.0 means the cost
/// is exactly linear in context.
///
/// Deliberately not a two-point ratio: anchoring on N=512 and N=4096 threw
/// away five of seven measured cells and put half the statistic's weight on
/// one cell whose ~2% run-to-run noise swung the printed figure between
/// 0.997x and 1.021x with no kernel change. max/min over the whole ladder is
/// what the results doc computed by hand anyway.
fn linearity_ratio(rows: &[AttnRow]) -> Option<f64> {
    let mut it = rows.iter().map(AttnRow::ns_per_pos);
    let first = it.next()?;
    let (lo, hi) = it.fold((first, first), |(lo, hi), v| (lo.min(v), hi.max(v)));
    Some(hi / lo)
}

fn main() {
    println!(
        "kernels microbench: warm-cache, single-thread; avx2+fma detected: {}",
        avx2::avx2_fma_available()
    );
    println!(
        "dot + quantizer table below: median of {RUNS} runs ({WARMUP} warmup). The attention \
         tables do NOT use that\nregime — they print their own runs and warmup per rung."
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
         (layers * context * kv_dim * 2 B * 2 planes). Since 8fc3c4a\nthe kernel is kv-head-outer \
         and widens each K/V head slice exactly ONCE per position, reusing it\nacross the whole \
         GQA group, so this is the exact byte count it touches — not one eighth of it.\nThe \
         pre-8fc3c4a kernel re-read each kv head's columns once per query head, i.e. 8x this; any\n\
         reading that divides an eff GB/s figure by 8 (or multiplies a byte count by it) is \
         reading a\nkernel that no longer exists.\n"
    );

    let (arm_a, drift_a) = bench_attention_arm(1, &ARM_A_SAMPLES, &mut rng);
    print_attn_table(
        "arm A: 1 layer (8 MiB planes), timed unit = one decode_attention call",
        &arm_a,
    );
    println!();
    print_drift_table("arm A", &drift_a);
    println!();
    let (arm_b, drift_b) = bench_attention_arm(ATTN_LAYERS, &ARM_B_SAMPLES, &mut rng);
    print_attn_table(
        "arm B: 48 layers (384 MiB planes), timed unit = one token (all 48 layers). Its \
         residency is a\nLOWER bound on the real decode loop's: the real loop also runs that \
         layer's GEMVs, the router\nand 8 streamed experts between two attention calls. \
         Optimistic below context ~1024.",
        &arm_b,
    );
    println!();
    print_drift_table("arm B", &drift_b);

    println!();
    for (arm, rows) in [("arm A", &arm_a), ("arm B", &arm_b)] {
        match linearity_ratio(rows) {
            Some(ratio) => println!(
                "linearity {arm}: max(ns/pos) / min(ns/pos) over all {} rungs = {ratio:.3}x \
                 (1.0 = cost linear in context)",
                rows.len()
            ),
            None => println!("linearity {arm}: no measured rungs, no ratio"),
        }
    }
}
