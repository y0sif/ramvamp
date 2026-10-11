//! Prefill's batched k-quant GEMV, single thread: one dot per (weight row,
//! activation row) against the four-row tile, at the v0 expert shapes.
//!
//! Two shapes: one expert's Q4_K `gate` (768 rows x 2048 in) and one
//! expert's Q6_K `down` (2048 rows x 768 in), each against the activation
//! rows a 512-token chunk routes to it and smaller counts, since the tile's
//! gain depends on how many rows share a weight row. The two arms alternate
//! block by block, and the outputs are compared bit for bit before anything
//! is timed.
//!
//! Warm, single-thread, synthetic weights: a diagnostic under rule 2 of
//! `docs/experiments.md`, never a published number.
//!
//! Run with `cargo bench -p ramvamp-core --bench batched`.

use std::hint::black_box;
use std::time::Instant;

use ramvamp_core::kernels::KernelError;
use ramvamp_core::kernels::quants::{BlockQ8K, QuantFormat, avx2, f32_to_f16, quantize_row_q8_k};

const ROUNDS: usize = 15;
const BLOCK: usize = 20;

/// Deterministic 64-bit LCG (Knuth MMIX constants), as in `kernels.rs`.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

type OneDot = fn(&[u8], &[BlockQ8K], bool) -> Result<f32, KernelError>;
type RowsDot = fn(&[u8], &[BlockQ8K], usize, &mut [f32], bool) -> Result<(), KernelError>;

/// One weight format at one shape.
struct Case {
    name: &'static str,
    format: QuantFormat,
    in_dim: usize,
    out_dim: usize,
    one: OneDot,
    rows: RowsDot,
}

/// Random packed blocks with sane f16 scales, as in `kernels.rs`.
fn synth(case: &Case, rng: &mut Lcg) -> Vec<u8> {
    let scale_offs: &[usize] = match case.format {
        QuantFormat::Q6_K => &[208],
        _ => &[0, 2],
    };
    let bytes = case.format.block_bytes();
    let mut w = vec![0u8; case.in_dim / 256 * case.out_dim * bytes];
    for block in w.chunks_exact_mut(bytes) {
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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The pre-tile batched loop: one single-row dot per output element.
fn per_row(case: &Case, weight: &[u8], acts: &[BlockQ8K], n_acts: usize, out: &mut [f32]) {
    let row_bytes = weight.len() / case.out_dim;
    let blocks = case.in_dim / 256;
    for (r, dst) in out.chunks_mut(n_acts).enumerate() {
        let row = &weight[r * row_bytes..(r + 1) * row_bytes];
        for (a, o) in dst.iter_mut().enumerate() {
            *o = (case.one)(row, &acts[a * blocks..(a + 1) * blocks], false).expect("dot");
        }
    }
}

/// The tiled loop `gemv_q8_k_batched` now runs for this format.
fn tiled(case: &Case, weight: &[u8], acts: &[BlockQ8K], n_acts: usize, out: &mut [f32]) {
    let row_bytes = weight.len() / case.out_dim;
    for (r, dst) in out.chunks_mut(n_acts).enumerate() {
        let row = &weight[r * row_bytes..(r + 1) * row_bytes];
        (case.rows)(row, acts, n_acts, dst, false).expect("tile");
    }
}

fn main() {
    println!(
        "batched k-quant x q8_k, single thread, avx2+fma: {}",
        avx2::avx2_fma_available()
    );
    println!("median of {ROUNDS} round medians, {BLOCK} timed passes per arm per round");
    let cases = [
        Case {
            name: "q4_k gate, 768 rows x 2048 in",
            format: QuantFormat::Q4_K,
            in_dim: 2048,
            out_dim: 768,
            one: avx2::vec_dot_q4_k_q8_k,
            rows: avx2::vec_dot_q4_k_q8_k_rows,
        },
        Case {
            name: "q6_k down, 2048 rows x 768 in",
            format: QuantFormat::Q6_K,
            in_dim: 768,
            out_dim: 2048,
            one: avx2::vec_dot_q6_k_q8_k,
            rows: avx2::vec_dot_q6_k_q8_k_rows,
        },
    ];
    let mut rng = Lcg(0xBA7C);
    for case in &cases {
        println!("\n{}", case.name);
        println!(
            "{:>7} {:>14} {:>14} {:>8}",
            "n_acts", "per-row ms", "tiled ms", "ratio"
        );
        let weight = synth(case, &mut rng);
        for n_acts in [1usize, 2, 4, 8, 16, 32] {
            let x: Vec<f32> = (0..case.in_dim * n_acts)
                .map(|_| rng.next_f32() * 3.0)
                .collect();
            let mut acts = vec![BlockQ8K::default(); case.in_dim / 256 * n_acts];
            quantize_row_q8_k(&x, &mut acts).expect("quantize");
            let mut a = vec![0f32; case.out_dim * n_acts];
            let mut b = vec![0f32; case.out_dim * n_acts];
            per_row(case, &weight, &acts, n_acts, &mut a);
            tiled(case, &weight, &acts, n_acts, &mut b);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{}: the tile is not bit-identical at n_acts {n_acts}",
                case.name
            );
            let (mut ta, mut tb) = (Vec::new(), Vec::new());
            for round in 0..ROUNDS {
                for arm in 0..2 {
                    let first = (arm + round) % 2 == 0;
                    let mut times = Vec::with_capacity(BLOCK);
                    for _ in 0..BLOCK {
                        let t = Instant::now();
                        if first {
                            per_row(case, &weight, &acts, n_acts, &mut a);
                        } else {
                            tiled(case, &weight, &acts, n_acts, &mut b);
                        }
                        times.push(t.elapsed().as_secs_f64() * 1e3);
                        black_box((&a, &b));
                    }
                    let m = median(&mut times);
                    if first {
                        ta.push(m);
                    } else {
                        tb.push(m);
                    }
                }
            }
            let (ma, mb) = (median(&mut ta), median(&mut tb));
            println!("{n_acts:>7} {ma:>14.3} {mb:>14.3} {:>7.2}x", ma / mb);
        }
    }
}
