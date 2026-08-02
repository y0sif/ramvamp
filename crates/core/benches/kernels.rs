//! Warm-cache, single-thread microbenches for the quantized kernels:
//! scalar reference vs AVX2 dispatch for every dot kernel and both
//! activation quantizers.
//!
//! Deliberately no criterion (keeps the dependency tree lean): a plain
//! `std::time::Instant` harness with warmup and median-of-31 runs. Numbers
//! from here are diagnostics per the experiment-log rules — published
//! end-to-end numbers come from cold runs inside the 3 GB cgroup, not from
//! this harness.
//!
//! Run with `cargo bench -p ramvamp-core`.

use std::time::Instant;

use ramvamp_core::kernels::quants::{
    BlockQ8_0, BlockQ8K, QuantFormat, avx2, f32_to_f16, quantize_row_q8_0, quantize_row_q8_k,
};

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

/// Median wall time of `RUNS` timed calls after `WARMUP` untimed ones.
fn median_ns<F: FnMut() -> f32>(mut f: F) -> f64 {
    let mut sink = 0f32;
    for _ in 0..WARMUP {
        sink += f();
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        sink += f();
        samples.push(t.elapsed().as_nanos() as f64);
    }
    std::hint::black_box(sink);
    samples.sort_by(f64::total_cmp);
    samples[RUNS / 2]
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
}
