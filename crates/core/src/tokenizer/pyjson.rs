//! JSON serialization that reproduces CPython's `json.dumps` byte for byte.
//!
//! Qwen3's chat template renders every tool definition through Jinja's
//! `tool | tojson`. In `transformers` that filter is not Jinja's own — it is
//! rebound to a thin wrapper over the standard library
//! (`transformers/utils/chat_template_utils.py`, unchanged across 4.51 and
//! 5.14):
//!
//! ```python
//! def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
//!     return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
//!                       separators=separators, sort_keys=sort_keys)
//! ```
//!
//! So the text the model is trained to see is `json.dumps(x,
//! ensure_ascii=False)` with default separators. [`serde_json::to_string`]
//! produces *valid* JSON for the same value but not the *same* JSON, and it
//! differs in four independent ways — each of which changes the prompt
//! silently, because nothing downstream can tell a re-spelled tool schema
//! from a correct one:
//!
//! | | `json.dumps` | `serde_json` |
//! |---|---|---|
//! | separators | `", "` and `": "` | `","` and `":"` |
//! | key order | insertion order | sorted (`BTreeMap`) |
//! | small floats | `1e-06`, `1.5e-07` | `1e-6`, `1.5e-7` |
//! | float ties | half to even, `…05.2` | half away from zero, `…05.3` |
//!
//! The key-order row is fixed workspace-wide by the `preserve_order` feature
//! in the root `Cargo.toml`, which swaps [`serde_json::Map`] for an
//! `IndexMap`; the other three are fixed here by [`PyJsonFormatter`]. The
//! last row is the rarest and the least obvious — see
//! [`even_tie_alternative`] for what it is and how often it bites.
//!
//! # Known limits
//!
//! - **Integers wider than 64 bits.** Without `arbitrary_precision`,
//!   `serde_json` parses an integer outside `i64`/`u64` into an `f64`, so a
//!   schema containing `10**25` round-trips as `1e+25` where Python keeps
//!   `10000000000000000000000000`. Tool schemas carry counts and bounds, not
//!   bignums, so this is recorded rather than fixed
//!   (`known_limit_big_integers_collapse_to_f64`).
//! - **Non-finite floats.** `json.dumps` emits the bare words `NaN`,
//!   `Infinity` and `-Infinity`, which are not JSON. This module does not
//!   reproduce that, and cannot be asked to: [`serde_json::Number`] refuses
//!   to hold a non-finite value, so a [`Value`] tree never contains one.
//! - **`f32`.** Python has one float type. [`Value`] only ever holds `f64`,
//!   so `f32` is unreachable through [`to_string`]; [`PyJsonFormatter`]
//!   still formats it with the same rules applied to the shortest decimal
//!   that identifies the `f32`, for callers that drive it over other types.

use std::fmt::Write as _;
use std::io;

use serde::Serialize;
use serde::ser::Error as _;
use serde_json::Value;
use serde_json::ser::Formatter;

/// Serializes `value` exactly as `json.dumps(value, ensure_ascii=False)`
/// would.
///
/// Errors only if the underlying serializer does; writing to an in-memory
/// buffer has no failure mode of its own.
pub fn to_string(value: &Value) -> Result<String, serde_json::Error> {
    let mut buf = Vec::with_capacity(128);
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, PyJsonFormatter);
    value.serialize(&mut serializer)?;
    // `serde_json` only ever writes UTF-8, so this cannot fail. Converting
    // the impossible case into an error keeps the "no panics on untrusted
    // input" rule literal instead of relying on that argument staying true.
    String::from_utf8(buf).map_err(serde_json::Error::custom)
}

/// A [`Formatter`] whose output matches CPython's `json.dumps` defaults.
///
/// Only the three points of divergence are overridden. String escaping is
/// left to `serde_json`, which already agrees with `ensure_ascii=False`: it
/// escapes exactly `"`, `\` and the C0 controls (with Python's same `\b`
/// `\t` `\n` `\f` `\r` shorthands and lowercase `\u00XX` for the rest), and
/// passes `/`, `DEL`, `U+00A0`, `U+2028` and every non-ASCII character
/// through raw.
#[derive(Debug, Clone, Copy, Default)]
pub struct PyJsonFormatter;

impl Formatter for PyJsonFormatter {
    /// `", "` between array elements, where the default writes `","`.
    #[inline]
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    /// `", "` between object members, where the default writes `","`.
    #[inline]
    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    /// `": "` between key and value, where the default writes `":"`.
    #[inline]
    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }

    /// See [`PyJsonFormatter`]'s note on `f32`.
    fn write_f32<W>(&mut self, writer: &mut W, value: f32) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(layout(&format!("{value:e}")).as_bytes())
    }

    /// Python's `repr` layout rather than `serde_json`'s; see
    /// [`python_repr`].
    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(python_repr(value).as_bytes())
    }
}

/// `repr(value)` as CPython would print it.
///
/// Both languages start from the shortest decimal digit string that
/// round-trips, so the digits nearly always agree and only the layout has to
/// be rebuilt ([`layout`]). [`even_tie_alternative`] repairs the one case
/// where the digits themselves disagree.
fn python_repr(value: f64) -> String {
    let shortest = format!("{value:e}");
    match even_tie_alternative(value, &shortest) {
        Some(even) => layout(&even),
        None => layout(&shortest),
    }
}

/// Significant digits in the exact decimal expansion of the widest `f64`.
///
/// Every double is a dyadic rational, so its decimal expansion terminates;
/// 767 digits covers the longest one. Asking `{:.*e}` for that many gives
/// the exact value followed by zeros, which is what makes the
/// exactly-representable test in [`even_tie_alternative`] exact rather than
/// approximate.
const EXACT_F64_DIGITS: usize = 767;

/// The even-last-digit spelling of `value`, when Rust and CPython break a
/// tie differently.
///
/// A double occasionally sits *exactly* halfway between the two shortest
/// decimals that identify it — `1050218469363005.25` is equidistant from
/// `…05.2` and `…05.3`, and both round-trip. Rust rounds such a tie away
/// from zero; CPython's `dtoa` rounds it to an even last digit. Measured
/// over 205,716 doubles (uniform bit patterns, decade sweeps, and
/// power-of-ten neighbours) this is the *only* way the two disagree, and it
/// hits 0.016% of them, concentrated between 1e13 and 1e16 where a
/// 17-digit shortest form meets a short exact expansion.
///
/// Returns `Some` only when all three hold, so a false positive would need
/// all three to be wrong at once:
///
/// 1. Rust's last digit is odd — the only case where rounding away from zero
///    and rounding to even can pick different digits.
/// 2. The neighbour one below round-trips to the same double, i.e. it is an
///    equally short spelling and not merely a nearby number.
/// 3. `value`'s exact expansion is that neighbour followed by a single `5`,
///    which is what "exactly halfway" means.
///
/// If Rust ever changed to round ties toward zero, step 2 would reject the
/// candidate and the shortest form would be used unchanged: this can fail to
/// fix a divergence, never introduce one.
fn even_tie_alternative(value: f64, shortest: &str) -> Option<String> {
    let (mantissa, exponent) = shortest.split_once('e')?;
    let last = *mantissa.as_bytes().last()?;
    if !last.is_ascii_digit() || last % 2 == 0 {
        return None;
    }

    let mut lower = String::with_capacity(shortest.len());
    lower.push_str(mantissa.get(..mantissa.len() - 1)?);
    lower.push(char::from(last - 1));
    lower.push('e');
    lower.push_str(exponent);
    if lower.parse::<f64>().ok()? != value {
        return None;
    }

    let exact = format!("{value:.*e}", EXACT_F64_DIGITS - 1);
    let (exact_mantissa, exact_exponent) = exact.split_once('e')?;
    if exact_exponent != exponent {
        return None;
    }
    let mut midpoint = significant_digits(lower.split_once('e')?.0);
    midpoint.push('5');
    let exact_digits = significant_digits(exact_mantissa);
    let tail = exact_digits.strip_prefix(&midpoint)?;
    if tail.bytes().all(|b| b == b'0') {
        Some(lower)
    } else {
        None
    }
}

/// The digits of a `{:e}` mantissa, without its sign or decimal point.
fn significant_digits(mantissa: &str) -> String {
    mantissa
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>()
}

/// The decimal-point position above which `repr` switches to exponential
/// form.
///
/// CPython's `format_float_short` (`Objects/stringlib/formatter.h` via
/// `pystrtod.c`) writes a float as `0.DIGITS * 10^decpt` and picks
/// exponential form iff `decpt <= -4 || decpt > 16`. That is why `1e15`
/// prints as `1000000000000000.0` (decpt 16) but `1e16` as `1e+16`
/// (decpt 17), and `0.0001` stays decimal (decpt -3) while `1e-05` does not
/// (decpt -4).
const MAX_FIXED_DECPT: i32 = 16;

/// The mirror of [`MAX_FIXED_DECPT`] on the small side.
const MIN_FIXED_DECPT: i32 = -3;

/// Rewrites an exponential form like `{:e}` produces into Python's `repr`
/// layout.
///
/// Taking `{:e}` as input hands us the digits already separated from the
/// exponent, which is the whole of the work — Rust's `{}` would instead
/// expand `5e-324` to 751 characters.
///
/// The two layouts, given digits `D` and `decpt` (the value is
/// `0.D * 10^decpt`):
///
/// - exponential, iff `decpt` is outside `MIN_FIXED_DECPT..=MAX_FIXED_DECPT`:
///   one digit, a fractional part only if there is more than one digit, then
///   `e`, a mandatory sign, and the exponent padded to **at least** two
///   digits — `1e+16`, `1e-06`, `1.5e-07`, `5e-324`.
/// - decimal otherwise, always with a fractional part, because a Python
///   float never prints as a bare integer — `100.0`, `-0.0`, `0.0001`.
///
/// Input that is not in `<mantissa>e<exponent>` form is returned unchanged.
/// That is unreachable in practice: only `NaN` and the infinities format
/// without an `e`, and `serde_json`'s serializer intercepts those before the
/// formatter sees them (it writes `null`). The branch exists so that the
/// function is total rather than as a code path with behaviour to rely on.
fn layout(exp_form: &str) -> String {
    let Some((mantissa, exponent)) = exp_form.split_once('e') else {
        return exp_form.to_owned();
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return exp_form.to_owned();
    };
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if digits.is_empty() {
        return exp_form.to_owned();
    }

    // `{:e}` normalizes to one digit before the point, so the value is
    // `0.digits * 10^(exponent + 1)`.
    let decpt = exponent + 1;
    let mut out = String::with_capacity(digits.len() + sign.len() + 8);
    out.push_str(sign);

    if !(MIN_FIXED_DECPT..=MAX_FIXED_DECPT).contains(&decpt) {
        let (lead, rest) = digits.split_at(1);
        out.push_str(lead);
        if !rest.is_empty() {
            out.push('.');
            out.push_str(rest);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        let magnitude = exponent.unsigned_abs();
        if magnitude < 10 {
            out.push('0');
        }
        // Writing into a `String` is infallible; `fmt::Write` still returns
        // a `Result`, and discarding it is the documented idiom.
        let _ = write!(out, "{magnitude}");
        return out;
    }

    // In the decimal range `decpt` is at least -3, so the casts below are
    // taken only on a value already known to be positive and small.
    if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt as usize >= digits.len() {
        let zeros = decpt as usize - digits.len();
        out.push_str(&digits);
        for _ in 0..zeros {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        let (whole, fraction) = digits.split_at(decpt as usize);
        out.push_str(whole);
        out.push('.');
        out.push_str(fraction);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Every expectation in this module was produced by running
    /// `json.dumps(value, ensure_ascii=False)` under CPython 3.13 and
    /// copying the result, rather than reasoned out by hand.
    fn py(value: &Value) -> String {
        to_string(value).expect("serializing a Value to a Vec cannot fail")
    }

    fn float(value: f64) -> String {
        py(&json!(value))
    }

    #[test]
    fn separators_match_python_for_objects_and_arrays() {
        assert_eq!(py(&json!({"a": 1, "b": 2})), r#"{"a": 1, "b": 2}"#);
        assert_eq!(py(&json!([1, 2, 3])), "[1, 2, 3]");
    }

    #[test]
    fn key_order_is_insertion_order_not_sorted() {
        // Deliberately non-alphabetical: `serde_json` without
        // `preserve_order` would emit apple, beta, mango, zebra.
        let value = json!({"zebra": 1, "apple": 2, "mango": 3, "beta": 4});
        assert_eq!(
            py(&value),
            r#"{"zebra": 1, "apple": 2, "mango": 3, "beta": 4}"#
        );
    }

    #[test]
    fn nesting_gets_separators_at_every_level() {
        let value = json!({
            "outer": {"b": [1, 2, {"z": true, "a": null}], "a": "x"},
            "list": [[], {}, [1]],
        });
        assert_eq!(
            py(&value),
            r#"{"outer": {"b": [1, 2, {"z": true, "a": null}], "a": "x"}, "list": [[], {}, [1]]}"#
        );
    }

    #[test]
    fn empty_containers_have_no_stray_separator() {
        assert_eq!(py(&json!({})), "{}");
        assert_eq!(py(&json!([])), "[]");
        assert_eq!(py(&json!({"a": {}, "b": []})), r#"{"a": {}, "b": []}"#);
    }

    #[test]
    fn scalars_render_as_python_spells_them() {
        assert_eq!(py(&Value::Null), "null");
        assert_eq!(py(&json!(true)), "true");
        assert_eq!(py(&json!(false)), "false");
        assert_eq!(py(&json!(42)), "42");
        assert_eq!(py(&json!(-7)), "-7");
    }

    #[test]
    fn strings_are_raw_except_the_json_mandatory_escapes() {
        // `"` and `\` must be escaped and newline and tab cannot be literal;
        // everything else survives verbatim under `ensure_ascii=False`,
        // including `/` (Python never escapes it), DEL, U+00A0, the
        // line-separator U+2028 that JavaScript encoders special-case, an
        // emoji outside the BMP, and CJK.
        let input = "q\"b\\s\nt\ts/ \u{7f} \u{a0} \u{2028} \u{1f980} 世界";
        let expected = concat!(
            r#""q\"b\\s\nt\ts/ "#,
            "\u{7f} \u{a0} \u{2028} \u{1f980} 世界\""
        );
        assert_eq!(py(&json!(input)), expected);
    }

    #[test]
    fn c0_controls_use_pythons_escape_set() {
        let input: String = (0..0x20u8).map(char::from).collect();
        assert_eq!(
            py(&json!(input)),
            r#""\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\b\t\n\u000b\f\r\u000e\u000f\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017\u0018\u0019\u001a\u001b\u001c\u001d\u001e\u001f""#
        );
    }

    #[test]
    fn floats_serde_json_already_agreed_on_still_agree() {
        assert_eq!(float(1e16), "1e+16");
        assert_eq!(float(1e21), "1e+21");
        assert_eq!(float(1e30), "1e+30");
        assert_eq!(float(1e100), "1e+100");
        assert_eq!(float(-0.0), "-0.0");
        assert_eq!(float(100.0), "100.0");
        assert_eq!(float(1e15), "1000000000000000.0");
    }

    #[test]
    fn floats_serde_json_got_wrong_now_match_python() {
        // `serde_json` writes 0.00001, 1e-6, 1e-7, 1.5e-7 for these: it
        // switches to exponential form one decade later than Python, and
        // never pads the exponent.
        assert_eq!(float(1e-5), "1e-05");
        assert_eq!(float(1e-6), "1e-06");
        assert_eq!(float(1e-7), "1e-07");
        assert_eq!(float(1.5e-7), "1.5e-07");
    }

    #[test]
    fn exponential_form_starts_exactly_where_python_switches() {
        // Large side: the switch is between decpt 16 and 17.
        assert_eq!(float(9999999999999998.0), "9999999999999998.0");
        assert_eq!(float(1e17), "1e+17");
        // Small side: between decpt -3 and -4.
        assert_eq!(float(1e-3), "0.001");
        assert_eq!(float(1e-4), "0.0001");
        assert_eq!(float(9.99e-5), "9.99e-05");
        assert_eq!(float(2.5e-8), "2.5e-08");
    }

    #[test]
    fn ordinary_and_signed_floats_keep_their_shortest_digits() {
        assert_eq!(float(0.0), "0.0");
        assert_eq!(float(0.1), "0.1");
        assert_eq!(float(1.5), "1.5");
        assert_eq!(float(-2.5), "-2.5");
        assert_eq!(float(123.456), "123.456");
        assert_eq!(float(1.2345678901234567), "1.2345678901234567");
        assert_eq!(float(123456789.0), "123456789.0");
        assert_eq!(float(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(float(-1e-6), "-1e-06");
    }

    #[test]
    fn extreme_floats_need_no_exponent_padding() {
        // Three-digit exponents are already wide enough; padding is a
        // minimum, not a fixed width.
        assert_eq!(float(f64::MAX), "1.7976931348623157e+308");
        assert_eq!(float(f64::MIN_POSITIVE), "2.2250738585072014e-308");
        assert_eq!(float(1e-320), "1e-320");
        assert_eq!(float(5e-324), "5e-324");
    }

    #[test]
    fn exact_midpoint_ties_round_the_last_digit_to_even() {
        // Each of these sits exactly halfway between two equally short
        // decimals: 0x430d…69ea is 1050218469363005.25, so both …05.2 and
        // …05.3 round-trip. Rust's `{:e}` rounds away from zero and picks
        // the odd one; CPython rounds to even. Bit patterns rather than
        // literals, so the test cannot be weakened by a lossy transcription.
        assert_eq!(
            float(f64::from_bits(0x430d_d958_566c_69ea)),
            "1050218469363005.2"
        );
        assert_eq!(
            float(f64::from_bits(0x42ea_7939_a330_de54)),
            "232863682889458.62"
        );
        assert_eq!(
            float(f64::from_bits(0xc2b7_ae7d_2b3f_3310)),
            "-26038191734579.062"
        );
        assert_eq!(
            float(f64::from_bits(0xc308_c375_b9c2_e7ea)),
            "-871288729525501.2"
        );
    }

    #[test]
    fn near_ties_keep_the_shortest_digits_untouched() {
        // The tie repair must not fire on a value that merely has an odd
        // last digit, nor on one whose lower neighbour round-trips without
        // being equidistant. 5e-324 is the sharpest case: `4e-324` parses
        // back to the same subnormal, but the exact value is
        // 4.94…e-324, not 4.5e-324.
        assert_eq!(float(5e-324), "5e-324");
        assert_eq!(float(0.3), "0.3");
        assert_eq!(float(1e-7), "1e-07");
        assert_eq!(float(9.007199254740993e15), "9007199254740992.0");
    }

    #[test]
    fn known_limit_big_integers_collapse_to_f64() {
        // Python prints 10000000000000000000000000. `serde_json` without
        // `arbitrary_precision` cannot represent an integer outside
        // i64/u64, so it parses this as a float and we faithfully print the
        // float. Documented, not fixed.
        let value: Value =
            serde_json::from_str("10000000000000000000000000").expect("valid JSON number");
        assert!(value.is_f64());
        assert_eq!(py(&value), "1e+25");
    }

    #[test]
    fn non_finite_floats_cannot_enter_a_value() {
        // `json.dumps` would emit the non-JSON words NaN / Infinity here.
        // There is nothing to match, because the tree cannot hold them.
        assert!(serde_json::Number::from_f64(f64::NAN).is_none());
        assert!(serde_json::Number::from_f64(f64::INFINITY).is_none());
        assert!(serde_json::Number::from_f64(f64::NEG_INFINITY).is_none());
    }

    #[test]
    fn qwen_tool_definition_matches_python_end_to_end() {
        let tool = json!({
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "description": "Get current temperature at a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for, in the format \"City, State, Country\"."
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit to return the temperature in.",
                            "default": "celsius"
                        },
                        "tolerance": {"type": "number", "default": 1e-6}
                    },
                    "required": ["location"]
                }
            }
        });
        assert_eq!(
            py(&tool),
            concat!(
                r#"{"type": "function", "function": {"name": "get_current_temperature", "#,
                r#""description": "Get current temperature at a location.", "parameters": "#,
                r#"{"type": "object", "properties": {"location": {"type": "string", "#,
                r#""description": "The location to get the temperature for, in the format "#,
                r#"\"City, State, Country\"."}, "unit": {"type": "string", "enum": "#,
                r#"["celsius", "fahrenheit"], "description": "The unit to return the "#,
                r#"temperature in.", "default": "celsius"}, "tolerance": {"type": "number", "#,
                r#""default": 1e-06}}, "required": ["location"]}}}"#
            )
        );
    }
}
