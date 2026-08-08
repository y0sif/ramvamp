//! Everything the pinned panel is made of, as pure functions over owned
//! data: the layout arithmetic, the DECSTBM escape sequences, the prefill
//! progress bar, and the status row.
//!
//! Nothing here opens, queries or writes a terminal, which is the point —
//! the parts of the harness worth testing are the parts a CI runner with no
//! TTY can still reach. [`super::Harness`] is the only thing that needs a
//! real terminal, and it is deliberately thin over this module.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};

use super::input::LineEditor;
use super::{Phase, Status};

/// Rows the pinned panel wants: a rule, a status row, a detail row, and the
/// input line.
///
/// A shorter terminal gets fewer rows, dropped from the top — see
/// [`Layout::new`] and [`panel_view`].
pub(crate) const PANEL_H: u16 = 4;

/// What the input line is prefixed with. ASCII on purpose: the windowing
/// arithmetic in [`LineEditor::view`] counts characters, so a prompt whose
/// character count is not its column count would put the cursor in the wrong
/// place.
pub(crate) const PROMPT: &str = "> ";

/// Widest progress bar we will draw, brackets included.
const BAR_MAX: usize = 24;

/// Narrowest bar worth drawing, brackets included (`[██]`). Below this the
/// bar is dropped and the `done/total` counts carry the information alone.
const BAR_MIN: usize = 4;

/// `ESC [ r`: DECSTBM reset, i.e. the scroll region is the whole screen
/// again.
pub(crate) const RESET_SCROLL_REGION: &str = "\x1b[r";

/// `ESC [ top ; bottom r`: DECSTBM, set the scroll region to those rows.
///
/// Both bounds are 1-based and inclusive, as the control sequence wants
/// them. Setting the region also homes the cursor, so every caller has to
/// move the cursor afterwards.
pub(crate) fn set_scroll_region(top: u16, bottom: u16) -> String {
    format!("\x1b[{top};{bottom}r")
}

/// How the screen is split between the natively-scrolling transcript and the
/// pinned panel.
///
/// Construction is total: a terminal reporting `0x0` mid-resize, or one row
/// tall, produces a `Layout` rather than an error. What it does not produce
/// is a *usable* one — see [`Layout::usable`], which the harness checks
/// before it sets a scroll region or draws anything.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Layout {
    width: u16,
    height: u16,
    panel_h: u16,
}

impl Layout {
    /// Split a terminal of this size.
    pub(crate) fn new(width: u16, height: u16) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        // Never more than half the screen, and never the last row: the
        // transcript is the thing the user came for, and a panel that has
        // eaten four of five rows is worse than one that has dropped its
        // rule and its detail line.
        let panel_h = PANEL_H.min(height / 2).max(1);
        Self {
            width,
            height,
            panel_h,
        }
    }

    /// Whether there is room for both a scroll region and a legible panel.
    ///
    /// Below this the harness sets no scroll region and draws nothing, so a
    /// terminal dragged down to nothing degrades to an ordinary one instead
    /// of to a panic.
    pub(crate) fn usable(&self) -> bool {
        self.width >= 4 && self.height >= 2
    }

    pub(crate) fn width(&self) -> u16 {
        self.width
    }

    /// Last row of the scroll region, 1-based, as DECSTBM wants it.
    pub(crate) fn region_bottom(&self) -> u16 {
        self.height.saturating_sub(self.panel_h).max(1)
    }

    /// Rows the panel needs reserved below the cursor before the region is
    /// set.
    pub(crate) fn panel_h(&self) -> u16 {
        self.panel_h
    }

    /// The panel in 0-based terminal coordinates, which is what
    /// `Viewport::Fixed` is specified in.
    pub(crate) fn panel_rect(&self) -> Rect {
        Rect::new(
            0,
            self.height.saturating_sub(self.panel_h),
            self.width,
            self.panel_h,
        )
    }
}

/// The prefill progress bar, as a plain string of block characters.
///
/// `width` is the whole thing including the brackets, so it can be handed
/// straight to a column budget. Below three columns there is nothing useful
/// to draw and the result is empty. A `total` of zero reads as complete
/// rather than dividing by zero.
///
/// Only fills the last cell at literally `done == total`: a bar that reads
/// full while the run is still going is worse than no bar.
pub(crate) fn progress_bar(done: usize, total: usize, width: usize) -> String {
    if width < 3 {
        return String::new();
    }
    let inner = width - 2;
    let filled = if total == 0 {
        inner
    } else {
        // u64 so a caller with an absurd position count cannot overflow the
        // multiply on a 32-bit target.
        let done = done.min(total) as u64;
        (done * inner as u64 / total as u64) as usize
    };
    format!("[{}{}]", "█".repeat(filled), "░".repeat(inner - filled))
}

/// The status row: phase, the prefill bar when there is one, live rate,
/// context use, and expert-cache hit rate.
///
/// Degrades by dropping segments from the right — hit rate, then context,
/// then the rate — and by shrinking the bar, so that on a narrow terminal
/// what survives is the phase and how far through prefill we are. The result
/// is always at most `width` characters.
pub(crate) fn status_line(status: &Status, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let label = phase_label(status.phase);
    let counts = status
        .prefill
        .map(|(done, total)| format!("{done}/{total}"));
    let counts_w = counts.as_ref().map_or(0, |c| 1 + c.chars().count());

    let mut tail: Vec<String> = Vec::new();
    if let Some(rate) = rate(status) {
        tail.push(format!("{rate:.1} tok/s"));
    }
    tail.push(format!("ctx {}/{}", status.context.0, status.context.1));
    if let Some(hit) = status.hit_rate {
        tail.push(format!(
            "hit {:.0}%",
            f64::from(hit.clamp(0.0, 1.0)) * 100.0
        ));
    }

    let fixed = |tail: &[String]| -> usize {
        label.chars().count() + counts_w + tail.iter().map(|s| 2 + s.chars().count()).sum::<usize>()
    };
    while !tail.is_empty() && fixed(&tail) > width {
        tail.pop();
    }

    let mut line = String::from(label);
    if let (Some((done, total)), Some(counts)) = (status.prefill, counts.as_ref()) {
        // Whatever the fixed part leaves over, minus the space in front of
        // it, is the bar's budget.
        let bar_width = width
            .saturating_sub(fixed(&tail))
            .saturating_sub(1)
            .min(BAR_MAX);
        if bar_width >= BAR_MIN {
            line.push(' ');
            line.push_str(&progress_bar(done, total, bar_width));
        }
        line.push(' ');
        line.push_str(counts);
    }
    for segment in &tail {
        line.push_str("  ");
        line.push_str(segment);
    }
    clip(&line, width)
}

/// How fast the *active* phase is going, or `None` when nothing is running
/// or nothing has happened yet.
fn rate(status: &Status) -> Option<f64> {
    let seconds = status.elapsed.as_secs_f64();
    if seconds <= 0.0 {
        return None;
    }
    let done = match status.phase {
        Phase::Idle => return None,
        Phase::Prefill => status.prefill.map(|(done, _)| done)?,
        Phase::Decode => status.tokens,
    };
    if done == 0 {
        return None;
    }
    Some(done as f64 / seconds)
}

fn phase_label(phase: Phase) -> &'static str {
    match phase {
        Phase::Idle => "idle",
        Phase::Prefill => "prefill",
        Phase::Decode => "decode",
    }
}

/// Truncate to `width` *characters*, which is the column count for
/// everything this module builds (ASCII plus single-width box and block
/// drawing).
pub(crate) fn clip(text: &str, width: usize) -> String {
    text.chars().take(width).collect()
}

/// One drawn row of the panel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PanelRow {
    pub(crate) text: String,
    pub(crate) style: Style,
}

impl PanelRow {
    fn plain(text: String) -> Self {
        Self {
            text,
            style: Style::default(),
        }
    }

    fn dim(text: String) -> Self {
        Self {
            text,
            style: Style::default().add_modifier(Modifier::DIM),
        }
    }
}

/// The panel as rows to draw plus where the terminal cursor belongs, both in
/// absolute terminal coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PanelView {
    pub(crate) rows: Vec<PanelRow>,
    pub(crate) cursor: (u16, u16),
}

/// Lay the panel out for this size, status and input line.
///
/// Rows are chosen bottom-up, so a panel squeezed onto a short terminal
/// keeps the input line first, then the status row, then the detail line,
/// then the rule.
pub(crate) fn panel_view(layout: Layout, status: &Status, editor: &LineEditor) -> PanelView {
    let rect = layout.panel_rect();
    let width = layout.width() as usize;
    let panel_h = layout.panel_h();

    // Leave the last cell of the last row alone. Writing the bottom-right
    // cell with autowrap on arms the terminal's pending-wrap flag on a row
    // outside the scroll region, and terminals disagree about what happens
    // next; not writing it costs one column and no arguments.
    let input_w = width.saturating_sub(PROMPT.chars().count() + 1);
    let (text, cursor_col) = editor.view(input_w);

    let mut bottom_up = vec![PanelRow::plain(format!("{PROMPT}{text}"))];
    if panel_h >= 2 {
        bottom_up.push(PanelRow::plain(status_line(status, width)));
    }
    if panel_h >= 3 {
        let detail = status.detail.as_deref().unwrap_or_default();
        bottom_up.push(PanelRow::dim(clip(detail, width)));
    }
    if panel_h >= 4 {
        bottom_up.push(PanelRow::dim("─".repeat(width)));
    }
    bottom_up.reverse();

    let cursor_x = (PROMPT.chars().count() + cursor_col).min(width.saturating_sub(1)) as u16;
    let cursor_y = rect.y + rect.height.saturating_sub(1);
    PanelView {
        rows: bottom_up,
        cursor: (cursor_x, cursor_y),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn prefill_status() -> Status {
        Status {
            phase: Phase::Prefill,
            prefill: Some((1024, 3961)),
            tokens: 0,
            elapsed: Duration::from_secs(93),
            context: (1024, 4096),
            hit_rate: Some(0.87),
            detail: None,
        }
    }

    #[test]
    fn scroll_region_sequences_are_decstbm() {
        assert_eq!(set_scroll_region(1, 20), "\x1b[1;20r");
        assert_eq!(set_scroll_region(1, 1), "\x1b[1;1r");
        assert_eq!(RESET_SCROLL_REGION, "\x1b[r");
    }

    #[test]
    fn progress_bar_at_the_ends_and_in_between() {
        assert_eq!(progress_bar(0, 3961, 10), "[░░░░░░░░]");
        assert_eq!(progress_bar(3961, 3961, 10), "[████████]");
        assert_eq!(progress_bar(1980, 3961, 10), "[███░░░░░]");
        // 99.9% is not 100%: the last cell only fills when the run is done.
        assert_eq!(progress_bar(3960, 3961, 10), "[███████░]");
    }

    #[test]
    fn progress_bar_survives_narrow_widths_and_a_zero_total() {
        assert_eq!(progress_bar(1, 2, 3), "[░]");
        assert_eq!(progress_bar(2, 2, 3), "[█]");
        assert_eq!(progress_bar(1, 2, 2), "");
        assert_eq!(progress_bar(1, 2, 0), "");
        // Nothing to do is done, not a division by zero.
        assert_eq!(progress_bar(0, 0, 6), "[████]");
    }

    #[test]
    fn status_line_shows_the_whole_row_at_eighty_columns() {
        let line = status_line(&prefill_status(), 80);
        assert_eq!(
            line,
            "prefill [█████░░░░░░░░░░░░░░░░░] 1024/3961  11.0 tok/s  ctx 1024/4096  hit 87%"
        );
        assert!(line.chars().count() <= 80);
    }

    #[test]
    fn status_line_drops_segments_from_the_right_as_it_narrows() {
        let status = prefill_status();
        // Wide enough for everything but the bar.
        assert_eq!(
            status_line(&status, 55),
            "prefill 1024/3961  11.0 tok/s  ctx 1024/4096  hit 87%"
        );
        // Hit rate goes first, then the context, then the rate.
        assert_eq!(
            status_line(&status, 45),
            "prefill 1024/3961  11.0 tok/s  ctx 1024/4096"
        );
        assert_eq!(status_line(&status, 30), "prefill 1024/3961  11.0 tok/s");
        assert_eq!(status_line(&status, 20), "prefill 1024/3961");
    }

    #[test]
    fn status_line_never_exceeds_its_width() {
        let status = prefill_status();
        for width in 0..=120 {
            let line = status_line(&status, width);
            assert!(
                line.chars().count() <= width,
                "width {width} produced {line:?}"
            );
        }
    }

    #[test]
    fn status_line_reports_decode_and_idle_without_a_bar() {
        let decode = Status {
            phase: Phase::Decode,
            prefill: None,
            tokens: 40,
            elapsed: Duration::from_secs(20),
            context: (4001, 4096),
            hit_rate: Some(0.5),
            detail: None,
        };
        assert_eq!(
            status_line(&decode, 80),
            "decode  2.0 tok/s  ctx 4001/4096  hit 50%"
        );
        let idle = Status {
            context: (0, 4096),
            ..Status::default()
        };
        assert_eq!(status_line(&idle, 80), "idle  ctx 0/4096");
    }

    #[test]
    fn layout_splits_a_normal_terminal() {
        let layout = Layout::new(80, 24);
        assert!(layout.usable());
        assert_eq!(layout.panel_h(), 4);
        assert_eq!(layout.region_bottom(), 20);
        assert_eq!(layout.panel_rect(), Rect::new(0, 20, 80, 4));
    }

    #[test]
    fn layout_never_gives_the_panel_more_than_half_the_screen() {
        // (height, expected panel rows, expected region bottom)
        for (height, panel_h, bottom) in [
            (24, 4, 20),
            (8, 4, 4),
            (7, 3, 4),
            (5, 2, 3),
            (3, 1, 2),
            (2, 1, 1),
        ] {
            let layout = Layout::new(80, height);
            assert!(layout.usable(), "{height} rows");
            assert_eq!(layout.panel_h(), panel_h, "{height} rows");
            assert_eq!(layout.region_bottom(), bottom, "{height} rows");
            assert_eq!(
                layout.panel_rect(),
                Rect::new(0, bottom, 80, panel_h),
                "{height} rows"
            );
        }
    }

    #[test]
    fn layout_refuses_to_be_usable_when_there_is_no_room() {
        for (width, height) in [(0, 0), (80, 1), (3, 24), (1, 1)] {
            let layout = Layout::new(width, height);
            assert!(!layout.usable(), "{width}x{height} claimed to be usable");
            // Still total: nothing here may panic or produce a zero region.
            assert!(layout.region_bottom() >= 1);
            assert!(layout.panel_h() >= 1);
        }
    }

    #[test]
    fn panel_view_puts_the_cursor_in_the_input_line() {
        let mut editor = LineEditor::default();
        editor.insert_char('h');
        editor.insert_char('i');
        let view = panel_view(Layout::new(80, 24), &prefill_status(), &editor);
        assert_eq!(view.rows.len(), 4);
        assert_eq!(view.rows[0].text.chars().count(), 80);
        assert_eq!(view.rows[3].text, "> hi");
        // Row 23 is the last row of an 80x24 terminal; column 4 is one past
        // "> hi".
        assert_eq!(view.cursor, (4, 23));
    }

    #[test]
    fn panel_view_keeps_the_input_line_when_the_panel_is_squeezed() {
        let editor = LineEditor::default();
        let view = panel_view(Layout::new(80, 2), &prefill_status(), &editor);
        assert_eq!(view.rows.len(), 1);
        assert_eq!(view.rows[0].text, "> ");
        assert_eq!(view.cursor, (2, 1));
    }

    #[test]
    fn panel_view_stays_inside_a_very_narrow_terminal() {
        let mut editor = LineEditor::default();
        editor.insert_str("a long line that cannot possibly fit");
        let view = panel_view(Layout::new(4, 24), &prefill_status(), &editor);
        for row in &view.rows {
            assert!(row.text.chars().count() <= 4, "{:?}", row.text);
        }
        assert!(view.cursor.0 <= 3);
    }
}
