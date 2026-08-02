//! Cross-format tests: quantize/dequantize round trips, dot products
//! against f64 ground truth, adversarial blocks, and length validation.

use super::blocks::{f16_at, get_scale_min_k4, q4k, q5k, q6k, scales_at};
use super::test_support::{
    Lcg, quantize_row_q4_k_test, quantize_row_q5_k_test, quantize_row_q6_k_test,
    quantize_row_q8_0_test,
};
use super::*;

/// Dequantize Q8_K activation blocks (d * q per value).
fn dequant_q8_k(blocks: &[BlockQ8K]) -> Vec<f32> {
    blocks
        .iter()
        .flat_map(|b| b.qs.iter().map(move |&q| b.d * f32::from(q)))
        .collect()
}

/// Dequantize Q8_0 activation blocks (d * q per value).
fn dequant_q8_0_act(blocks: &[BlockQ8_0]) -> Vec<f32> {
    blocks
        .iter()
        .flat_map(|b| b.qs.iter().map(move |&q| b.d * f32::from(q)))
        .collect()
}

/// Assert a scalar dot matches the f64 dot of the dequantized rows within
/// relative tolerance 1e-5, scaled by the sum of term magnitudes (the
/// numerically meaningful denominator when the true dot cancels toward
/// zero).
fn assert_dot_close(dot: f32, w: &[f32], a: &[f32]) {
    assert_eq!(w.len(), a.len());
    let mut r = 0f64;
    let mut mag = 0f64;
    for (&wi, &ai) in w.iter().zip(a) {
        let p = f64::from(wi) * f64::from(ai);
        r += p;
        mag += p.abs();
    }
    let tol = 1e-5 * mag.max(1e-20);
    let err = (f64::from(dot) - r).abs();
    assert!(
        err <= tol,
        "dot {dot} vs ref {r}: err {err} > tol {tol} (mag {mag})"
    );
}

// ---------------------------------------------------------------- round trip

#[test]
fn round_trip_q4_k() {
    let mut rng = Lcg(42);
    let x: Vec<f32> = (0..512).map(|_| rng.next_f32() * 8.0).collect();
    let bytes = quantize_row_q4_k_test(&x);
    let w = dequantize_row_q4_k(&bytes).unwrap();
    for (bi, block) in bytes.chunks_exact(q4k::BYTES).enumerate() {
        let d = f16_at(block, q4k::D);
        let dmin = f16_at(block, q4k::DMIN);
        let scales = scales_at(block, q4k::SCALES);
        for j in 0..8 {
            let (sc, _) = get_scale_min_k4(j, scales);
            // One quantization step plus one min step of slack.
            let tol = d * f32::from(sc) + dmin + 1e-6;
            for l in 0..32 {
                let i = 256 * bi + 32 * j + l;
                let err = (w[i] - x[i]).abs();
                assert!(err <= tol, "q4_K w[{i}]: {} vs {} (tol {tol})", w[i], x[i]);
            }
        }
    }
}

#[test]
fn round_trip_q5_k() {
    let mut rng = Lcg(43);
    let x: Vec<f32> = (0..512).map(|_| rng.next_f32() * 8.0).collect();
    let bytes = quantize_row_q5_k_test(&x);
    let w = dequantize_row_q5_k(&bytes).unwrap();
    for (bi, block) in bytes.chunks_exact(q5k::BYTES).enumerate() {
        let d = f16_at(block, q5k::D);
        let dmin = f16_at(block, q5k::DMIN);
        let scales = scales_at(block, q5k::SCALES);
        for j in 0..8 {
            let (sc, _) = get_scale_min_k4(j, scales);
            let tol = d * f32::from(sc) + dmin + 1e-6;
            for l in 0..32 {
                let i = 256 * bi + 32 * j + l;
                let err = (w[i] - x[i]).abs();
                assert!(err <= tol, "q5_K w[{i}]: {} vs {} (tol {tol})", w[i], x[i]);
            }
        }
    }
}

#[test]
fn round_trip_q6_k() {
    let mut rng = Lcg(44);
    let x: Vec<f32> = (0..512).map(|_| rng.next_f32() * 8.0).collect();
    let bytes = quantize_row_q6_k_test(&x);
    let w = dequantize_row_q6_k(&bytes).unwrap();
    for (bi, block) in bytes.chunks_exact(q6k::BYTES).enumerate() {
        let d = f16_at(block, q6k::D);
        for g in 0..16 {
            let sc = f32::from(block[q6k::SCALES + g] as i8);
            let tol = (d * sc).abs() + 1e-6;
            for l in 0..16 {
                let i = 256 * bi + 16 * g + l;
                let err = (w[i] - x[i]).abs();
                assert!(err <= tol, "q6_K w[{i}]: {} vs {} (tol {tol})", w[i], x[i]);
            }
        }
    }
}

#[test]
fn round_trip_q8_0() {
    let mut rng = Lcg(45);
    let x: Vec<f32> = (0..128).map(|_| rng.next_f32() * 8.0).collect();
    let bytes = quantize_row_q8_0_test(&x);
    let w = dequantize_row_q8_0(&bytes).unwrap();
    for (bi, block) in bytes.chunks_exact(34).enumerate() {
        let d = f16_at(block, 0);
        for l in 0..32 {
            let i = 32 * bi + l;
            let err = (w[i] - x[i]).abs();
            assert!(
                err <= d + 1e-6,
                "q8_0 w[{i}]: {} vs {} (tol {d})",
                w[i],
                x[i]
            );
        }
    }
}

#[test]
fn round_trip_extreme_and_flat_blocks() {
    // Constant blocks (zero scale everywhere), all-negative and
    // all-positive blocks; the affine formats must reproduce the constant
    // within a min step.
    for &c in &[0.0f32, -3.75, 12.5] {
        let x = [c; 256];
        let tol45 = c.abs() / 63.0 + 1e-3;
        for w in [
            dequantize_row_q4_k(&quantize_row_q4_k_test(&x)).unwrap(),
            dequantize_row_q5_k(&quantize_row_q5_k_test(&x)).unwrap(),
        ] {
            for &v in &w {
                assert!((v - c).abs() <= tol45, "flat {c}: got {v}");
            }
        }
        let wq6 = dequantize_row_q6_k(&quantize_row_q6_k_test(&x)).unwrap();
        let tol6 = c.abs() / 31.0 + 1e-3;
        for &v in &wq6 {
            assert!((v - c).abs() <= tol6, "flat q6 {c}: got {v}");
        }
    }
}

// ------------------------------------------------------------- ground truth

fn k_quant_ground_truth(
    quant: fn(&[f32]) -> Vec<u8>,
    dequant: fn(&[u8]) -> Result<Vec<f32>, KernelError>,
    dot: fn(&[u8], &[BlockQ8K]) -> Result<f32, KernelError>,
    seed: u64,
) {
    for n in [2048usize, 768] {
        let mut rng = Lcg(seed ^ n as u64);
        let xw: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0).collect();
        let xa: Vec<f32> = (0..n).map(|_| rng.next_f32() * 3.0).collect();
        let wb = quant(&xw);
        let mut acts = vec![BlockQ8K::default(); n / 256];
        quantize_row_q8_k(&xa, &mut acts).unwrap();
        let d = dot(&wb, &acts).unwrap();
        assert_dot_close(d, &dequant(&wb).unwrap(), &dequant_q8_k(&acts));
    }
}

#[test]
fn dot_ground_truth_q4_k() {
    k_quant_ground_truth(
        quantize_row_q4_k_test,
        dequantize_row_q4_k,
        vec_dot_q4_k_q8_k,
        0xA4,
    );
}

#[test]
fn dot_ground_truth_q5_k() {
    k_quant_ground_truth(
        quantize_row_q5_k_test,
        dequantize_row_q5_k,
        vec_dot_q5_k_q8_k,
        0xA5,
    );
}

#[test]
fn dot_ground_truth_q6_k() {
    k_quant_ground_truth(
        quantize_row_q6_k_test,
        dequantize_row_q6_k,
        vec_dot_q6_k_q8_k,
        0xA6,
    );
}

#[test]
fn dot_ground_truth_q8_0() {
    for n in [2048usize, 768] {
        let mut rng = Lcg(0x80 ^ n as u64);
        let xw: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0).collect();
        let xa: Vec<f32> = (0..n).map(|_| rng.next_f32() * 3.0).collect();
        let wb = quantize_row_q8_0_test(&xw);
        let mut acts = vec![BlockQ8_0::default(); n / 32];
        quantize_row_q8_0(&xa, &mut acts).unwrap();
        let d = vec_dot_q8_0_q8_0(&wb, &acts).unwrap();
        assert_dot_close(
            d,
            &dequantize_row_q8_0(&wb).unwrap(),
            &dequant_q8_0_act(&acts),
        );
    }
}

// -------------------------------------------------------------- adversarial

/// Build one raw q4_K block from parts.
fn build_q4k(d: f32, dmin: f32, scales: [u8; 12], qs: u8) -> Vec<u8> {
    let mut b = Vec::with_capacity(q4k::BYTES);
    b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    b.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
    b.extend_from_slice(&scales);
    b.extend_from_slice(&[qs; 128]);
    b
}

/// Build one raw q5_K block from parts.
fn build_q5k(d: f32, dmin: f32, scales: [u8; 12], qh: u8, qs: u8) -> Vec<u8> {
    let mut b = Vec::with_capacity(q5k::BYTES);
    b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    b.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
    b.extend_from_slice(&scales);
    b.extend_from_slice(&[qh; 32]);
    b.extend_from_slice(&[qs; 128]);
    b
}

/// Build one raw q6_K block from parts.
fn build_q6k(ql: u8, qh: u8, scale: i8, d: f32) -> Vec<u8> {
    let mut b = Vec::with_capacity(q6k::BYTES);
    b.extend_from_slice(&[ql; 128]);
    b.extend_from_slice(&[qh; 64]);
    b.extend_from_slice(&[scale as u8; 16]);
    b.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    b
}

/// Scales field encoding sc = 0, m = 63 for all eight sub-blocks.
const ALL_MIN_SCALES: [u8; 12] = [0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xF0, 0xF0, 0xF0, 0xF0];

fn random_q8_k_block(seed: u64) -> Vec<BlockQ8K> {
    let mut rng = Lcg(seed);
    let xa: Vec<f32> = (0..256).map(|_| rng.next_f32() * 2.0).collect();
    let mut acts = vec![BlockQ8K::default()];
    quantize_row_q8_k(&xa, &mut acts).unwrap();
    acts
}

#[test]
fn adversarial_q4_k() {
    let acts = random_q8_k_block(0xD4);
    let a = dequant_q8_k(&acts);
    let cases = [
        build_q4k(1.0, 0.5, [0xFF; 12], 0xFF), // Max-magnitude scales and values.
        build_q4k(4.0, 2.0, [0; 12], 0x5A),    // Zero scales: every weight 0.
        build_q4k(0.0, 0.0, [0x3F; 12], 0xFF), // Zero d/dmin.
        build_q4k(1.0, 0.5, ALL_MIN_SCALES, 0x00), // All-min: only the bsums path.
    ];
    for bytes in &cases {
        let d = vec_dot_q4_k_q8_k(bytes, &acts).unwrap();
        assert_dot_close(d, &dequantize_row_q4_k(bytes).unwrap(), &a);
    }
    // Zero-scale block against zero activations: exactly zero.
    let zero_acts = vec![BlockQ8K::default()];
    assert_eq!(vec_dot_q4_k_q8_k(&cases[1], &zero_acts).unwrap(), 0.0);
}

#[test]
fn adversarial_q5_k() {
    let acts = random_q8_k_block(0xD5);
    let a = dequant_q8_k(&acts);
    let cases = [
        build_q5k(1.0, 0.5, [0xFF; 12], 0xFF, 0xFF), // Everything maxed: q = 31.
        build_q5k(1.0, 0.5, [0xFF; 12], 0x00, 0xFF), // High bit clear everywhere.
        build_q5k(4.0, 2.0, [0; 12], 0xAA, 0x5A),    // Zero scales.
        build_q5k(1.0, 0.5, ALL_MIN_SCALES, 0xAA, 0x00), // Min path + stray high bits.
    ];
    for bytes in &cases {
        let d = vec_dot_q5_k_q8_k(bytes, &acts).unwrap();
        assert_dot_close(d, &dequantize_row_q5_k(bytes).unwrap(), &a);
    }
}

#[test]
fn adversarial_q6_k() {
    let acts = random_q8_k_block(0xD6);
    let a = dequant_q8_k(&acts);
    let cases = [
        build_q6k(0xFF, 0xFF, -128, 1.0), // Most negative scale, max q bits.
        build_q6k(0xFF, 0xFF, 127, 0.25), // Most positive scale.
        build_q6k(0x00, 0x00, 127, 1.0),  // q = -32 everywhere (all-min analogue).
        build_q6k(0x5A, 0xA5, 0, 4.0),    // Zero scales.
    ];
    for bytes in &cases {
        let d = vec_dot_q6_k_q8_k(bytes, &acts).unwrap();
        assert_dot_close(d, &dequantize_row_q6_k(bytes).unwrap(), &a);
    }
}

#[test]
fn adversarial_q8_0() {
    // Max-magnitude f16 scale on the weight side, extreme int8 values.
    let mut wb = Vec::new();
    wb.extend_from_slice(&0x7BFFu16.to_le_bytes()); // d = 65504.0
    wb.extend_from_slice(&[0x80; 32]); // q = -128
    let acts = [BlockQ8_0 {
        d: 65504.0,
        qs: [127; 32],
    }];
    let d = vec_dot_q8_0_q8_0(&wb, &acts).unwrap();
    assert_dot_close(
        d,
        &dequantize_row_q8_0(&wb).unwrap(),
        &dequant_q8_0_act(&acts),
    );
    // Same weights against zero activations: exactly zero.
    assert_eq!(
        vec_dot_q8_0_q8_0(&wb, &[BlockQ8_0::default()]).unwrap(),
        0.0
    );
}

// -------------------------------------------------------------------- bsums

#[test]
fn q8_k_bsums_match_recomputed_group_sums() {
    let mut rng = Lcg(9);
    let x: Vec<f32> = (0..768).map(|_| rng.next_f32() * 5.0).collect();
    let mut acts = vec![BlockQ8K::default(); 3];
    quantize_row_q8_k(&x, &mut acts).unwrap();
    for b in &acts {
        assert!(b.d != 0.0);
        for (g, &bs) in b.bsums.iter().enumerate() {
            let s: i32 = b.qs[16 * g..16 * g + 16]
                .iter()
                .map(|&q| i32::from(q))
                .sum();
            assert_eq!(i32::from(bs), s, "bsums[{g}]");
        }
    }
}

// -------------------------------------------------------- length validation

#[test]
fn dequantize_rejects_partial_blocks() {
    let big = [0u8; 512];
    for (format, ok_len) in [
        (QuantFormat::Q4_K, 144usize),
        (QuantFormat::Q5_K, 176),
        (QuantFormat::Q6_K, 210),
        (QuantFormat::Q8_0, 34),
    ] {
        let dequant: fn(&[u8]) -> Result<Vec<f32>, KernelError> = match format {
            QuantFormat::Q4_K => dequantize_row_q4_k,
            QuantFormat::Q5_K => dequantize_row_q5_k,
            QuantFormat::Q6_K => dequantize_row_q6_k,
            QuantFormat::Q8_0 => dequantize_row_q8_0,
            QuantFormat::Q8_K => unreachable!(),
        };
        // Truncated and oversized by one byte; empty and exact succeed.
        for bad in [ok_len - 1, ok_len + 1] {
            assert_eq!(
                dequant(&big[..bad]).unwrap_err(),
                KernelError::RowBytesNotBlockMultiple {
                    format,
                    len: bad,
                    block_bytes: ok_len,
                },
            );
        }
        assert_eq!(dequant(&[]).unwrap(), Vec::<f32>::new());
        assert_eq!(
            dequant(&big[..ok_len]).unwrap().len(),
            format.block_weights()
        );
    }
}

#[test]
fn dots_reject_bad_lengths() {
    let acts2 = vec![BlockQ8K::default(); 2];
    // One weight block against two activation blocks.
    assert_eq!(
        vec_dot_q4_k_q8_k(&[0; 144], &acts2).unwrap_err(),
        KernelError::BlockCountMismatch {
            weight_blocks: 1,
            activation_blocks: 2,
        },
    );
    // Truncated weight row fails before the count check.
    assert_eq!(
        vec_dot_q5_k_q8_k(&[0; 175], &acts2).unwrap_err(),
        KernelError::RowBytesNotBlockMultiple {
            format: QuantFormat::Q5_K,
            len: 175,
            block_bytes: 176,
        },
    );
    assert_eq!(
        vec_dot_q6_k_q8_k(&[0; 630], &acts2).unwrap_err(),
        KernelError::BlockCountMismatch {
            weight_blocks: 3,
            activation_blocks: 2,
        },
    );
    let acts8 = vec![BlockQ8_0::default(); 8];
    assert_eq!(
        vec_dot_q8_0_q8_0(&[0; 35], &acts8).unwrap_err(),
        KernelError::RowBytesNotBlockMultiple {
            format: QuantFormat::Q8_0,
            len: 35,
            block_bytes: 34,
        },
    );
    // Matching lengths succeed (zero rows dot to zero).
    assert_eq!(vec_dot_q8_0_q8_0(&[0; 272], &acts8).unwrap(), 0.0);
}

#[test]
fn row_bytes_helper() {
    assert_eq!(QuantFormat::Q4_K.row_bytes(2048).unwrap(), 1152);
    assert_eq!(QuantFormat::Q5_K.row_bytes(768).unwrap(), 528);
    assert_eq!(QuantFormat::Q6_K.row_bytes(768).unwrap(), 630);
    assert_eq!(QuantFormat::Q8_0.row_bytes(2048).unwrap(), 2176);
    assert_eq!(QuantFormat::Q8_K.row_bytes(256).unwrap(), 292);
    assert_eq!(QuantFormat::Q4_K.row_bytes(0).unwrap(), 0);
    assert_eq!(
        QuantFormat::Q4_K.row_bytes(100).unwrap_err(),
        KernelError::IndivisibleRow {
            format: QuantFormat::Q4_K,
            in_dim: 100,
            block_weights: 256,
        },
    );
    assert_eq!(
        QuantFormat::Q8_0.row_bytes(33).unwrap_err(),
        KernelError::IndivisibleRow {
            format: QuantFormat::Q8_0,
            in_dim: 33,
            block_weights: 32,
        },
    );
}
