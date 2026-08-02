//! IEEE 754 binary16 <-> binary32 bit conversion.
//!
//! ggml stores block scales as f16 (`ggml_half`); the kernels need an exact
//! software conversion with no dependency. Both directions handle signed
//! zero, subnormals, infinities, and NaN. `f32_to_f16` rounds to nearest,
//! ties to even, matching hardware `vcvtps2ph` and ggml's table-driven
//! `GGML_FP32_TO_FP16` on every input (both implement the same IEEE
//! operation).

/// 2^-24 as an exact f32: the weight of one f16 subnormal ulp.
const F16_SUBNORMAL_ULP: f32 = f32::from_bits(0x3380_0000);

/// Convert IEEE binary16 bits to an f32.
///
/// Exact: every f16 value (including subnormals) is representable in f32.
/// NaN payloads are preserved in the top 10 fraction bits.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let frac = u32::from(bits & 0x03FF);
    match exp {
        // Zero or subnormal: value = (-1)^s * frac * 2^-24, exact in f32.
        0 => {
            let magnitude = frac as f32 * F16_SUBNORMAL_ULP;
            f32::from_bits(magnitude.to_bits() | sign)
        }
        // Inf / NaN: max exponent, fraction shifted into place.
        0x1F => f32::from_bits(sign | 0x7F80_0000 | (frac << 13)),
        // Normal: rebias exponent 15 -> 127.
        _ => f32::from_bits(sign | (u32::from(exp) + 112) << 23 | (frac << 13)),
    }
}

/// Convert an f32 to IEEE binary16 bits, rounding to nearest-even.
///
/// Overflow (|x| > 65519.996...) becomes signed infinity; values below the
/// smallest subnormal round to signed zero; NaN stays NaN (quiet bit forced
/// so a payload that would truncate to zero cannot turn into infinity).
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf / NaN.
        if frac == 0 {
            return sign | 0x7C00;
        }
        return sign | 0x7C00 | 0x0200 | ((frac >> 13) as u16 & 0x03FF);
    }

    // Unbiased-for-f16 exponent: f16 stored exponent would be exp - 112.
    let e16 = exp - 112;

    if e16 >= 0x1F {
        // Too large for f16: overflow to infinity.
        return sign | 0x7C00;
    }

    if e16 <= 0 {
        // Subnormal (or zero) in f16. Shift the full 24-bit significand
        // (implicit bit included; zero for f32 zero/subnormals, which all
        // underflow anyway) right so the value's weight is 2^-24 per ulp,
        // rounding half to even. e16 <= -11 shifts everything out to zero.
        if e16 < -10 {
            return sign;
        }
        let mant = if exp == 0 { frac } else { frac | 0x0080_0000 };
        let shift = (14 - e16) as u32;
        let ulp = 1u32 << shift;
        let half = ulp >> 1;
        let mut v = mant >> shift;
        let rem = mant & (ulp - 1);
        if rem > half || (rem == half && v & 1 == 1) {
            v += 1; // May carry into the smallest normal: correct encoding.
        }
        return sign | v as u16;
    }

    // Normal: round the 23-bit fraction to 10 bits, half to even. A carry
    // out of the fraction increments the exponent field, and a carry into
    // exponent 31 yields infinity -- both correct because the encoding is
    // monotonic.
    let mut v = ((e16 as u32) << 10) | (frac >> 13);
    let rem = frac & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_to_f32_known_vectors() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000) == 0.0 && f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC100), -2.5);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // Largest finite f16.
        assert_eq!(f16_to_f32(0x0001), F16_SUBNORMAL_ULP); // Smallest subnormal.
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * F16_SUBNORMAL_ULP); // Largest subnormal.
        assert_eq!(f16_to_f32(0x8001), -F16_SUBNORMAL_ULP);
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
        assert!(f16_to_f32(0xFE01).is_nan());
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95); // 1/3 rounded to f16.
    }

    #[test]
    fn f32_to_f16_known_vectors() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.5), 0xC100);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF);
        assert_eq!(f32_to_f16(65520.0), 0x7C00); // Ties-to-even overflow to inf.
        assert_eq!(f32_to_f16(1e9), 0x7C00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
        assert_eq!(f32_to_f16(F16_SUBNORMAL_ULP), 0x0001);
        assert_eq!(f32_to_f16(0.5 * F16_SUBNORMAL_ULP), 0x0000); // Half an ulp, ties to even 0.
        assert_eq!(f32_to_f16(0.75 * F16_SUBNORMAL_ULP), 0x0001); // Rounds up.
        assert_eq!(f32_to_f16(1e-10), 0x0000); // Deep underflow.
        assert_eq!(f32_to_f16(-1e-10), 0x8000);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        // NaN with a payload that truncates to zero must stay NaN.
        assert!(f16_to_f32(f32_to_f16(f32::from_bits(0x7F80_0001))).is_nan());
    }

    #[test]
    fn f16_round_trip_all_finite() {
        // Every finite f16 bit pattern must survive f16 -> f32 -> f16.
        for bits in 0u16..=0xFFFF {
            let exp = (bits >> 10) & 0x1F;
            if exp == 0x1F {
                continue; // Inf/NaN round-trip checked separately.
            }
            let f = f16_to_f32(bits);
            assert_eq!(
                f32_to_f16(f),
                bits,
                "bits {bits:#06x} -> {f} did not round-trip"
            );
        }
    }

    #[test]
    fn f32_to_f16_nearest_even_on_normals() {
        // 1.0 + 1 f16-ulp = 1.0009765625; halfway point 1.00048828125
        // must tie to even (0x3C00), just above must round up (0x3C01).
        assert_eq!(f32_to_f16(1.000_488_3), 0x3C00);
        assert_eq!(f32_to_f16(1.000_488_4), 0x3C01);
        // Halfway between 0x3C01 and 0x3C02 ties to even 0x3C02.
        assert_eq!(f32_to_f16(1.001_464_8), 0x3C02);
    }
}
