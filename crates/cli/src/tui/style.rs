//! The panel's colour roles, resolved once and shared.
//!
//! # Why the roles are named rather than coloured
//!
//! The panel sits in the user's terminal, under the user's theme, next to the
//! user's shell prompt. A palette of hardcoded hex would fight all three: what
//! reads as "muted grey" on the author's background is invisible on somebody
//! else's. So nothing here names a colour. Each role names a *job* — the
//! focal metric, the phase that is running, the structure between them — and
//! the role decides how to say it in whatever palette is actually installed.
//!
//! # The two layers
//!
//! Everything resolves to ANSI 16 first, because that is the layer every
//! terminal implements and the only one the user can retheme. [`accent`],
//! [`live`] and [`alert`] never leave it: they are cyan, green and red, and
//! they follow the sixteen colours the user configured.
//!
//! [`muted`] and [`faint`] are the exception, and the reason this module
//! exists. They carry the *secondary* layer — metrics, hints, rules, the
//! empty half of a progress bar — which needs to sit visibly below the body
//! text without becoming a colour of its own. The portable spelling for that
//! is SGR 2, and SGR 2 is widely unimplemented; where it is ignored the whole
//! secondary layer snaps back to [`primary`] and the panel goes flat. So when
//! the terminal's actual foreground and background are known, those two roles
//! upgrade to a truecolor blend along the line between them — literally
//! "*this* theme's text, faded toward *this* theme's background".
//!
//! That upgrade is the only thing truecolor is used for. No backgrounds are
//! painted, no other role changes, and the fallback is not a degraded layout:
//! it is the same rows, the same widths, the same characters, with
//! [`Modifier::DIM`] where a blend would have been.
//!
//! # The ground is not probed
//!
//! Learning the terminal's foreground and background means asking it — OSC 10
//! and OSC 11 — and the answer comes back on stdin, in the same queue as the
//! user's keystrokes. That probe is not currently safe here, so [`Palette::detect`]
//! does not run one and every terminal gets the [`Palette::fallback`] path.
//! See [`ground`] for the three reasons.

use std::sync::OnceLock;

use ratatui::style::{Color, Modifier, Style};

/// How much foreground survives in [`Palette::muted`].
///
/// Chosen to sit clearly below body text while staying comfortably readable
/// at small sizes; it is the layer most of the panel's numbers live in.
const MUTED_ALPHA: f32 = 0.62;

/// How much foreground survives in [`Palette::faint`].
///
/// Structure only — rules and the empty half of the progress bar — so it is
/// allowed to be barely there.
const FAINT_ALPHA: f32 = 0.34;

/// `ESC [ 2 m`: SGR 2, "faint". The portable spelling of the secondary layer,
/// and the one the fallback path uses for both secondary roles.
const SGR_FAINT: &str = "\x1b[2m";

/// `ESC [ 22 m`: SGR 22, which turns off both bold and faint. Closes a span
/// opened with [`SGR_FAINT`].
const SGR_FAINT_OFF: &str = "\x1b[22m";

/// `ESC [ 39 m`: SGR 39, default foreground. Closes a span opened with an
/// `ESC [ 38 ; 2 ; …` truecolor foreground.
const SGR_FG_DEFAULT: &str = "\x1b[39m";

/// A 24-bit colour, as a terminal reports one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Rgb {
    r: u8,
    g: u8,
    b: u8,
}

impl Rgb {
    pub(crate) const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

/// `fg * alpha + bg * (1 - alpha)`, per channel.
///
/// A point on the straight line between the two grounds, so `alpha == 1.0` is
/// the foreground exactly and `alpha == 0.0` is the background exactly. No
/// gamma correction and no perceptual space: the roles are tuned by eye
/// against the linear result, and a fancier curve would only move the numbers
/// the tuning already accounts for.
fn blend(fg: Rgb, bg: Rgb, alpha: f32) -> Rgb {
    Rgb::new(
        blend_channel(fg.r, bg.r, alpha),
        blend_channel(fg.g, bg.g, alpha),
        blend_channel(fg.b, bg.b, alpha),
    )
}

/// One channel of [`blend`], rounded to nearest rather than truncated.
///
/// The clamp is belt and braces — the result of a convex combination of two
/// `u8`s cannot leave `0..=255` — but it is what makes the `as u8` cast
/// total rather than something to reason about.
fn blend_channel(fg: u8, bg: u8, alpha: f32) -> u8 {
    let value = f32::from(fg) * alpha + f32::from(bg) * (1.0 - alpha);
    value.round().clamp(0.0, 255.0) as u8
}

/// The colour roles the panel draws in.
///
/// Build one with [`Palette::detect`] in the application and
/// [`Palette::fallback`] in a test that wants the no-truecolor path pinned.
/// Both are cheap to call; `detect` is cached and the accessors just copy a
/// [`Style`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Palette {
    muted: Style,
    faint: Style,
    muted_sgr: String,
    muted_sgr_end: &'static str,
}

impl Palette {
    /// The palette for this terminal, resolved once per process.
    ///
    /// Never fails and never blocks: see [`ground`] for why there is nothing
    /// to wait for.
    pub(crate) fn detect() -> &'static Palette {
        static PALETTE: OnceLock<Palette> = OnceLock::new();
        PALETTE.get_or_init(|| match ground() {
            Some((fg, bg)) => Palette::truecolor(fg, bg),
            None => Palette::fallback(),
        })
    }

    /// The palette for a terminal whose foreground and background are known.
    ///
    /// Only [`Palette::muted`] and [`Palette::faint`] differ from
    /// [`Palette::fallback`]; every other role is byte-identical, which is
    /// what lets the panel's snapshot tests hold across both paths.
    pub(crate) fn truecolor(fg: Rgb, bg: Rgb) -> Palette {
        let muted = blend(fg, bg, MUTED_ALPHA);
        let faint = blend(fg, bg, FAINT_ALPHA);
        Palette {
            muted: Style::default().fg(Color::Rgb(muted.r, muted.g, muted.b)),
            faint: Style::default().fg(Color::Rgb(faint.r, faint.g, faint.b)),
            muted_sgr: format!("\x1b[38;2;{};{};{}m", muted.r, muted.g, muted.b),
            muted_sgr_end: SGR_FG_DEFAULT,
        }
    }

    /// The palette for a terminal whose ground we do not know.
    ///
    /// Both secondary roles collapse onto [`Modifier::DIM`]. Nothing else
    /// changes — same rows, same widths, same characters.
    pub(crate) fn fallback() -> Palette {
        let dim = Style::default().add_modifier(Modifier::DIM);
        Palette {
            muted: dim,
            faint: dim,
            muted_sgr: SGR_FAINT.to_string(),
            muted_sgr_end: SGR_FAINT_OFF,
        }
    }

    /// Body text, the phase word, and the one metric the row is about.
    ///
    /// Deliberately no SGR at all: the terminal's own default foreground is
    /// the brightest thing the panel is allowed to be, so that everything
    /// else can be read as quieter than it.
    pub(crate) fn primary(&self) -> Style {
        Style::default()
    }

    /// Prefilling.
    pub(crate) fn accent(&self) -> Style {
        Style::default().fg(Color::Cyan)
    }

    /// Decoding.
    pub(crate) fn live(&self) -> Style {
        Style::default().fg(Color::Green)
    }

    /// Context near full, and errors.
    pub(crate) fn alert(&self) -> Style {
        Style::default().fg(Color::Red)
    }

    /// The input prompt marker.
    ///
    /// Weight rather than colour, because it marks the one row the user types
    /// into and has to stay legible under every theme including a monochrome
    /// one.
    pub(crate) fn strong(&self) -> Style {
        Style::default().add_modifier(Modifier::BOLD)
    }

    /// Secondary text: metrics, hints, anything read after the phase word.
    pub(crate) fn muted(&self) -> Style {
        self.muted
    }

    /// Structure: rules, and the empty track of the progress bar.
    pub(crate) fn faint(&self) -> Style {
        self.faint
    }

    /// Raw SGR opener for writing muted text straight into the scroll region,
    /// where ratatui is not involved.
    ///
    /// Either `ESC [ 38 ; 2 ; r ; g ; b m` or `ESC [ 2 m`, depending on which
    /// path [`Palette::detect`] took.
    ///
    /// Two rules for callers, both learned the hard way:
    ///
    /// - **A span must never cross a newline.** The scroll region is the
    ///   terminal's own scrollback; a style still open when the line ends is
    ///   inherited by every line after it, and by everything the user scrolls
    ///   back to. Open it, write the run, close it, *then* emit the newline.
    /// - **Close with [`Palette::muted_sgr_end`], never `ESC [ 0 m`.** SGR 0
    ///   resets every attribute there is, including the ones the user's shell
    ///   set outside the region and expects to still be there when the
    ///   process exits. The matching closer turns off exactly what the opener
    ///   turned on: `ESC [ 39 m` for the truecolor foreground, `ESC [ 22 m`
    ///   for SGR 2.
    pub(crate) fn muted_sgr(&self) -> &str {
        &self.muted_sgr
    }

    /// The closer for a span opened with [`Palette::muted_sgr`].
    pub(crate) fn muted_sgr_end(&self) -> &'static str {
        self.muted_sgr_end
    }
}

/// The terminal's foreground and background, when they can be had without
/// costing the user a keystroke.
///
/// Always `None`. Reading them means writing OSC 10 and OSC 11 and reading
/// the replies off stdin, and on this crate's terminal stack that cannot be
/// done safely:
///
/// 1. **crossterm does not parse OSC.** Its `parse_event` sends `ESC ]` down
///    the "unrecognised escape" path, so an OSC 10 reply arrives at the
///    application as `Alt+]`, then one `KeyCode::Char` per byte of
///    `11;rgb:…`, then `Alt+\` — roughly two dozen keystrokes injected into
///    the input line. So the reply cannot be collected through
///    `event::read`.
/// 2. **A raw read on fd 0 cannot be undone.** crossterm's own DSR query
///    survives this because `InternalEventReader` stashes non-matching events
///    in a `skipped_events` buffer and hands them back on the next `read`;
///    that buffer and the `Filter` trait that drives it are `pub(crate)`, so
///    there is no way to put bytes back. And a tty supports no peek — `poll`
///    reports that a byte is available, never which byte, and `MSG_PEEK` is a
///    socket call. Anything typed during the reply window is therefore read
///    and destroyed.
/// 3. **The call site is not ours to order.** This function runs behind a
///    `OnceLock` on first use, which is somewhere inside a panel draw — after
///    `Harness::enter` has already started crossterm's reader. A raw read at
///    that point races crossterm's internal buffer for the same fd.
///
/// (1) and (3) are the ones that would have to change. If crossterm ever
/// parses OSC replies into an `Event`, the whole probe becomes four safe
/// lines and the only edit here is to fill this function in;
/// [`Palette::truecolor`] on the other side is already built and tested.
fn ground() -> Option<(Rgb, Rgb)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dark theme's ground: near-white text on near-black.
    const DARK_FG: Rgb = Rgb::new(0xEC, 0xEB, 0xF0);
    const DARK_BG: Rgb = Rgb::new(0x20, 0x1F, 0x26);

    /// A light theme's ground, at the extremes, where the blend is easiest to
    /// check by hand.
    const LIGHT_FG: Rgb = Rgb::new(0x00, 0x00, 0x00);
    const LIGHT_BG: Rgb = Rgb::new(0xFF, 0xFF, 0xFF);

    #[test]
    fn blend_matches_hand_computed_values_on_a_dark_ground() {
        // 236*0.62 + 32*0.38 = 158.48
        // 235*0.62 + 31*0.38 = 157.48
        // 240*0.62 + 38*0.38 = 163.24
        assert_eq!(
            blend(DARK_FG, DARK_BG, MUTED_ALPHA),
            Rgb::new(158, 157, 163)
        );
        // 236*0.34 + 32*0.66 = 101.36
        // 235*0.34 + 31*0.66 = 100.36
        // 240*0.34 + 38*0.66 = 106.68  (rounds up, it does not truncate)
        assert_eq!(
            blend(DARK_FG, DARK_BG, FAINT_ALPHA),
            Rgb::new(101, 100, 107)
        );
    }

    #[test]
    fn blend_matches_hand_computed_values_on_a_light_ground() {
        // 0*0.62 + 255*0.38 = 96.9
        assert_eq!(blend(LIGHT_FG, LIGHT_BG, MUTED_ALPHA), Rgb::new(97, 97, 97));
        // 0*0.34 + 255*0.66 = 168.3
        assert_eq!(
            blend(LIGHT_FG, LIGHT_BG, FAINT_ALPHA),
            Rgb::new(168, 168, 168)
        );
    }

    #[test]
    fn blend_is_exact_at_the_ends_of_the_range() {
        assert_eq!(blend(DARK_FG, DARK_BG, 1.0), DARK_FG);
        assert_eq!(blend(DARK_FG, DARK_BG, 0.0), DARK_BG);
        assert_eq!(blend(LIGHT_FG, LIGHT_BG, 1.0), LIGHT_FG);
        assert_eq!(blend(LIGHT_FG, LIGHT_BG, 0.0), LIGHT_BG);
    }

    #[test]
    fn blend_stays_in_range_for_alphas_outside_it() {
        // Nothing in this module passes these, but the cast has to be total.
        for alpha in [-4.0, -0.5, 1.5, 9.0, f32::NAN] {
            let _ = blend(DARK_FG, DARK_BG, alpha);
            let _ = blend(LIGHT_FG, LIGHT_BG, alpha);
        }
    }

    #[test]
    fn truecolor_upgrades_exactly_the_two_secondary_roles() {
        let palette = Palette::truecolor(DARK_FG, DARK_BG);
        assert_eq!(
            palette.muted(),
            Style::default().fg(Color::Rgb(158, 157, 163))
        );
        assert_eq!(
            palette.faint(),
            Style::default().fg(Color::Rgb(101, 100, 107))
        );
    }

    #[test]
    fn fallback_dims_both_secondary_roles() {
        let palette = Palette::fallback();
        let dim = Style::default().add_modifier(Modifier::DIM);
        assert_eq!(palette.muted(), dim);
        assert_eq!(palette.faint(), dim);
    }

    #[test]
    fn accents_stay_ansi_on_both_paths() {
        // The whole point: cyan/green/red follow the user's configured
        // sixteen colours and never become a hex the theme cannot move.
        for palette in [Palette::truecolor(DARK_FG, DARK_BG), Palette::fallback()] {
            assert_eq!(palette.accent(), Style::default().fg(Color::Cyan));
            assert_eq!(palette.live(), Style::default().fg(Color::Green));
            assert_eq!(palette.alert(), Style::default().fg(Color::Red));
        }
    }

    #[test]
    fn primary_and_strong_are_identical_on_both_paths() {
        for palette in [Palette::truecolor(LIGHT_FG, LIGHT_BG), Palette::fallback()] {
            assert_eq!(palette.primary(), Style::default());
            assert_eq!(
                palette.strong(),
                Style::default().add_modifier(Modifier::BOLD)
            );
        }
    }

    #[test]
    fn muted_sgr_emits_a_truecolor_foreground_when_the_ground_is_known() {
        let palette = Palette::truecolor(DARK_FG, DARK_BG);
        assert_eq!(palette.muted_sgr(), "\x1b[38;2;158;157;163m");
        assert_eq!(palette.muted_sgr_end(), "\x1b[39m");
        let light = Palette::truecolor(LIGHT_FG, LIGHT_BG);
        assert_eq!(light.muted_sgr(), "\x1b[38;2;97;97;97m");
    }

    #[test]
    fn muted_sgr_emits_sgr_two_when_the_ground_is_unknown() {
        let palette = Palette::fallback();
        assert_eq!(palette.muted_sgr(), "\x1b[2m");
        assert_eq!(palette.muted_sgr_end(), "\x1b[22m");
    }

    #[test]
    fn no_closer_is_a_full_reset() {
        // SGR 0 would clobber attributes the user's shell set outside the
        // scroll region, so neither path may reach for it.
        for palette in [Palette::truecolor(DARK_FG, DARK_BG), Palette::fallback()] {
            assert_ne!(palette.muted_sgr_end(), "\x1b[0m");
            assert!(palette.muted_sgr().starts_with("\x1b["));
            assert!(palette.muted_sgr().ends_with('m'));
        }
    }

    #[test]
    fn detect_resolves_once_and_returns_the_same_palette() {
        let first = Palette::detect();
        let second = Palette::detect();
        assert!(std::ptr::eq(first, second));
    }

    #[test]
    fn detect_falls_back_while_the_ground_cannot_be_probed() {
        // Pins the documented behaviour in `ground`: no probe, so no
        // truecolor, so the secondary roles are DIM everywhere.
        assert!(ground().is_none());
        assert_eq!(*Palette::detect(), Palette::fallback());
    }
}
