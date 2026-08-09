//! The characters the panel is drawn with, in a Unicode set and an ASCII one.
//!
//! # The invariant
//!
//! **Every glyph in both sets is exactly one `char` and exactly one column
//! wide.** `panel.rs` budgets columns by counting characters, so a glyph
//! whose character count is not its column count silently corrupts every
//! width calculation downstream of it: the progress bar overruns, the status
//! row clips in the wrong place, and the cursor lands somewhere other than
//! where the user is typing. `PanelSpan::width` and `INDENT` carry the same
//! rule on the other side of it.
//!
//! That is why there are no emoji here and never will be. Emoji are
//! double-width and most are more than one `char` once a variation selector
//! or a ZWJ sequence is involved; either half of that breaks the arithmetic.
//! The chosen code points are all from Box Drawing, Block Elements, General
//! Punctuation or Latin-1, all of which are unambiguously narrow.
//! [`tests::every_glyph_is_one_char_and_one_column`] enforces it.
//!
//! # Why there is an ASCII set at all
//!
//! A terminal that is not in a UTF-8 locale renders `━` as two or three
//! mojibake characters, which is both ugly and — because it is more than one
//! column — wrong. Falling back to `=` is not a downgrade so much as the only
//! honest option, and it costs nothing: the two sets have the same shape, so
//! the panel's layout code never learns which one it is holding.

use std::sync::OnceLock;

/// The panel's characters.
///
/// A plain struct of `&'static str` rather than `char` so callers can
/// `push_str` and `repeat` without a conversion at every site, which is what
/// the bar and rule code actually wants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Glyphs {
    /// Leading mark on the state row, coloured by phase.
    pub(crate) chip: &'static str,
    /// A filled cell of the progress bar.
    pub(crate) bar_full: &'static str,
    /// The one cell between filled and empty, for the fractional remainder.
    pub(crate) bar_partial: &'static str,
    /// An empty cell of the progress bar's track — and, repeated, the
    /// horizontal rule, which is the same character doing the same job.
    pub(crate) bar_empty: &'static str,
    /// Elbow that opens the detail line under the row it belongs to.
    pub(crate) gutter: &'static str,
    /// Marks the row the user types into.
    pub(crate) prompt: &'static str,
    /// Between two metrics on one row.
    pub(crate) separator: &'static str,
}

impl Glyphs {
    /// Box drawing, block elements and punctuation: narrow everywhere, and
    /// present in every font that has been near a terminal.
    pub(crate) const UNICODE: Glyphs = Glyphs {
        chip: "\u{258C}",        // ▌ LEFT HALF BLOCK
        bar_full: "\u{2501}",    // ━ BOX DRAWINGS HEAVY HORIZONTAL
        bar_partial: "\u{2578}", // ╸ BOX DRAWINGS HEAVY LEFT
        bar_empty: "\u{2500}",   // ─ BOX DRAWINGS LIGHT HORIZONTAL
        gutter: "\u{2514}",      // └ BOX DRAWINGS LIGHT UP AND RIGHT
        prompt: "\u{203A}",      // › SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
        separator: "\u{00B7}",   // · MIDDLE DOT
    };

    /// The same shapes in the invariant subset, for a terminal that is not in
    /// a UTF-8 locale.
    pub(crate) const ASCII: Glyphs = Glyphs {
        chip: "|",
        bar_full: "=",
        bar_partial: "-",
        bar_empty: "-",
        gutter: "+",
        prompt: ">",
        separator: "|",
    };

    /// The set this terminal can render, resolved once per process.
    pub(crate) fn detect() -> &'static Glyphs {
        static GLYPHS: OnceLock<Glyphs> = OnceLock::new();
        GLYPHS.get_or_init(|| {
            if wants_ascii(|key| std::env::var(key).ok()) {
                Glyphs::ASCII
            } else {
                Glyphs::UNICODE
            }
        })
    }
}

/// Whether this environment gets the ASCII set.
///
/// Takes its lookup as a closure so the decision can be tested without
/// touching the process environment: `std::env::set_var` is `unsafe` in
/// edition 2024, and mutating it under a threaded test harness is a data race
/// against every other test in the binary.
///
/// Three ways in, in order:
///
/// - `RAMVAMP_ASCII` set to anything but `0`, the explicit override, which
///   wins over a locale that claims otherwise.
/// - `TERM=dumb`, which by definition renders nothing interesting.
/// - The locale. POSIX precedence: the first of `LC_ALL`, `LC_CTYPE`, `LANG`
///   that is set and non-empty decides, and the rest are not consulted — so
///   `LC_ALL=C` forces ASCII even next to `LANG=en_US.UTF-8`. An environment
///   with none of the three set is the `C` locale, which is not UTF-8, so it
///   gets ASCII too.
fn wants_ascii(env: impl Fn(&str) -> Option<String>) -> bool {
    if env("RAMVAMP_ASCII").is_some_and(|flag| flag != "0") {
        return true;
    }
    if env("TERM").as_deref() == Some("dumb") {
        return true;
    }
    ["LC_ALL", "LC_CTYPE", "LANG"]
        .into_iter()
        .find_map(|key| env(key).filter(|value| !value.is_empty()))
        .is_none_or(|locale| !is_utf8(&locale))
}

/// Whether a locale string names UTF-8, matching `utf8` or `utf-8` in any
/// case — `en_US.UTF-8`, `C.utf8` and `en_GB.Utf-8` are all the same claim.
fn is_utf8(locale: &str) -> bool {
    let locale = locale.to_ascii_lowercase();
    locale.contains("utf8") || locale.contains("utf-8")
}

#[cfg(test)]
mod tests {
    use ratatui::text::Span;

    use super::*;

    /// Every glyph of one set, so a test cannot pass by forgetting a field.
    fn all(set: &Glyphs) -> [&'static str; 7] {
        let Glyphs {
            chip,
            bar_full,
            bar_partial,
            bar_empty,
            gutter,
            prompt,
            separator,
        } = *set;
        [
            chip,
            bar_full,
            bar_partial,
            bar_empty,
            gutter,
            prompt,
            separator,
        ]
    }

    /// An environment built from pairs, for [`wants_ascii`].
    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |key| {
            pairs
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| (*value).to_string())
        }
    }

    #[test]
    fn every_glyph_is_one_char_and_one_column() {
        for (name, set) in [("unicode", Glyphs::UNICODE), ("ascii", Glyphs::ASCII)] {
            for glyph in all(&set) {
                assert_eq!(
                    glyph.chars().count(),
                    1,
                    "{name} glyph {glyph:?} is not one char"
                );
                // The same width the renderer will use: ratatui's `Span::width`
                // is unicode-width, which is what decides how many columns the
                // terminal actually spends. `panel.rs` budgets by char count,
                // so the two have to agree.
                assert_eq!(
                    Span::raw(glyph).width(),
                    1,
                    "{name} glyph {glyph:?} is not one column"
                );
            }
        }
    }

    #[test]
    fn the_ascii_set_is_ascii() {
        for glyph in all(&Glyphs::ASCII) {
            assert!(glyph.is_ascii(), "{glyph:?} is not ASCII");
        }
    }

    #[test]
    fn the_unicode_set_is_the_documented_code_points() {
        assert_eq!(all(&Glyphs::UNICODE), ["▌", "━", "╸", "─", "└", "›", "·"]);
        assert_eq!(all(&Glyphs::ASCII), ["|", "=", "-", "-", "+", ">", "|"]);
    }

    #[test]
    fn a_utf8_locale_keeps_the_unicode_set() {
        assert!(!wants_ascii(env(&[
            ("TERM", "xterm-256color"),
            ("LANG", "en_US.UTF-8"),
        ])));
        // Spelling and case are both allowed to vary.
        for locale in ["C.utf8", "en_GB.Utf-8", "de_DE.utf-8", "ja_JP.UTF8"] {
            assert!(
                !wants_ascii(env(&[("LC_ALL", locale)])),
                "{locale} was refused"
            );
        }
    }

    #[test]
    fn a_non_utf8_locale_falls_back_to_ascii() {
        for locale in ["C", "POSIX", "en_US", "en_US.ISO-8859-1", "ru_RU.KOI8-R"] {
            assert!(wants_ascii(env(&[("LANG", locale)])), "{locale} was taken");
        }
    }

    #[test]
    fn an_environment_with_no_locale_at_all_falls_back_to_ascii() {
        assert!(wants_ascii(env(&[])));
        assert!(wants_ascii(env(&[("TERM", "xterm-256color")])));
    }

    #[test]
    fn locale_precedence_stops_at_the_first_one_that_is_set() {
        // LC_ALL wins outright, in both directions.
        assert!(wants_ascii(env(&[
            ("LC_ALL", "C"),
            ("LC_CTYPE", "en_US.UTF-8"),
            ("LANG", "en_US.UTF-8"),
        ])));
        assert!(!wants_ascii(env(&[
            ("LC_ALL", "en_US.UTF-8"),
            ("LANG", "C"),
        ])));
        // Then LC_CTYPE, then LANG.
        assert!(wants_ascii(env(&[
            ("LC_CTYPE", "POSIX"),
            ("LANG", "en_US.UTF-8"),
        ])));
        assert!(!wants_ascii(env(&[
            ("LC_CTYPE", "en_US.UTF-8"),
            ("LANG", "C"),
        ])));
        // An empty value is not set, which is how setlocale reads it too.
        assert!(!wants_ascii(env(&[
            ("LC_ALL", ""),
            ("LANG", "en_US.UTF-8")
        ])));
        assert!(wants_ascii(env(&[
            ("LC_ALL", ""),
            ("LC_CTYPE", ""),
            ("LANG", "")
        ])));
    }

    #[test]
    fn a_dumb_terminal_falls_back_to_ascii() {
        assert!(wants_ascii(env(&[
            ("TERM", "dumb"),
            ("LANG", "en_US.UTF-8"),
        ])));
        // Only exactly "dumb"; "xterm" and friends say nothing about UTF-8.
        assert!(!wants_ascii(env(&[
            ("TERM", "dumb-something"),
            ("LANG", "en_US.UTF-8"),
        ])));
    }

    #[test]
    fn the_override_forces_ascii_over_a_utf8_locale() {
        assert!(wants_ascii(env(&[
            ("RAMVAMP_ASCII", "1"),
            ("LANG", "en_US.UTF-8"),
        ])));
        // Set to anything but "0" counts, including an empty value.
        for flag in ["1", "yes", "true", "", "0 "] {
            assert!(
                wants_ascii(env(&[("RAMVAMP_ASCII", flag), ("LANG", "en_US.UTF-8")])),
                "{flag:?} did not force ASCII"
            );
        }
    }

    #[test]
    fn the_override_set_to_zero_defers_to_everything_else() {
        assert!(!wants_ascii(env(&[
            ("RAMVAMP_ASCII", "0"),
            ("LANG", "en_US.UTF-8"),
        ])));
        // It is an opt-in, not an opt-out: it cannot force Unicode onto a
        // locale that cannot render it.
        assert!(wants_ascii(env(&[("RAMVAMP_ASCII", "0"), ("LANG", "C")])));
        assert!(wants_ascii(env(&[
            ("RAMVAMP_ASCII", "0"),
            ("TERM", "dumb"),
            ("LANG", "en_US.UTF-8"),
        ])));
    }

    #[test]
    fn detect_resolves_once_and_returns_the_same_set() {
        let first = Glyphs::detect();
        let second = Glyphs::detect();
        assert!(std::ptr::eq(first, second));
        assert!(*first == Glyphs::UNICODE || *first == Glyphs::ASCII);
    }
}
