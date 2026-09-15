//! Everything the pinned panel is made of, as pure functions over owned
//! data: the layout arithmetic, the DECSTBM escape sequences, the prefill
//! progress bar, and the four rows.
//!
//! Nothing here opens, queries or writes a terminal, which is the point —
//! the parts of the harness worth testing are the parts a CI runner with no
//! TTY can still reach. [`super::Harness`] is the only thing that needs a
//! real terminal, and it is deliberately thin over this module.
//!
//! # The shape of a row
//!
//! Every row is a *left cluster* and a *right cluster* with padding between
//! them:
//!
//! ```text
//! ▌ decode                              47 tok · 2.1 tok/s   ctx 1071/4096 · 26%
//! └┬─┘                                  └───────────────────────────────────────┘
//!  └ two columns of gutter, always                  right cluster, always
//!    the same width, so the three                   ending two columns short
//!    states never shift the text                    of the right edge
//! ```
//!
//! Two numbers do all the work and are the reason the panel does not jump
//! when the phase changes: content starts at column [`INDENT`] on every row of
//! every state, and the right cluster ends [`MARGIN`] columns short of the
//! right edge. A row that cannot afford everything drops right-cluster
//! segments from the right; it never moves what is left.
//!
//! # Colour is a role, never a colour
//!
//! Nothing here names a colour or an SGR code. [`Palette`] resolves the roles
//! and [`Glyphs`] resolves the characters, both handed in rather than looked
//! up, so a test can pin either one. The rule the rows follow:
//!
//! - the chip says what is running: `accent` prefilling, `live` decoding,
//!   `muted` idle;
//! - the phase word and **one** metric per state are `primary` — the ETA
//!   while prefilling, tok/s while decoding, nothing at all when idle;
//! - everything else on the state and detail rows is `muted`;
//! - the rule, and the empty track of the bar, are `faint`;
//! - the prompt marker is `strong` in every state, because input stays live
//!   while a turn runs.

use ratatui::layout::Rect;
use ratatui::style::Style;

use super::glyphs::Glyphs;
use super::input::LineEditor;
use super::style::Palette;
use super::{Phase, Prefilling, Status};
use crate::human_bytes;

/// Rows the pinned panel wants: the rule (or the prefill bar), the state row,
/// the detail row, and the input line.
///
/// A shorter terminal gets fewer rows — see [`Layout::new`] and
/// [`panel_view`].
pub(crate) const PANEL_H: u16 = 4;

/// Columns before any row's content: the chip and its space, the detail row's
/// blank gutter, the prompt and its space. Identical on every row so the eye
/// has one left edge to follow.
const INDENT: usize = 2;

/// Columns kept clear at the right end of the state and detail rows.
///
/// Two rather than none because a metric that ends in the last column reads
/// as clipped whether it is or not, and because a terminal that reflows on
/// resize has somewhere to put the difference.
const MARGIN: usize = 2;

/// Columns between two segments of a right cluster. Wide enough that
/// `1024/3961 · 26%` and `~4m 26s left` read as two facts rather than one.
const GAP: usize = 3;

/// What the input row says when nothing is typed and nothing is running.
const PLACEHOLDER: &str = "Ask ramvamp anything";

/// The keys, on the idle detail row.
const HINTS: [&str; 3] = ["enter send", "ctrl-c stop", "ctrl-d quit"];

/// The detail row's right cluster while a turn runs, after the clock.
const INTERRUPT_HINT: &str = "esc to interrupt";

/// Context use, in percent, at which the `ctx` segment turns `alert`.
///
/// A turn refused for want of context is the one panel number that costs the
/// user a retype, so it stops being a metric and starts being a warning
/// before it is too late to shorten the next message.
const CTX_ALERT: u64 = 95;

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

// ---------------------------------------------------------------------------
// layout
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// rows
// ---------------------------------------------------------------------------

/// A run of one styled text within a row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PanelSpan {
    pub(crate) text: String,
    pub(crate) style: Style,
}

impl PanelSpan {
    fn new(text: impl Into<String>, style: Style) -> Self {
        Self {
            text: text.into(),
            style,
        }
    }

    /// Columns this span occupies.
    ///
    /// Characters, not bytes: every glyph the panel draws is one column wide
    /// (`glyphs.rs` enforces it) and so is every digit and letter it formats,
    /// so a character count *is* a column count here.
    fn width(&self) -> usize {
        self.text.chars().count()
    }
}

/// One drawn row of the panel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PanelRow {
    pub(crate) spans: Vec<PanelSpan>,
}

impl PanelRow {
    /// The row as the terminal will show it, styling dropped. The shape every
    /// snapshot asserts on, and the only thing here the binary does not need:
    /// what it draws is the spans.
    #[cfg(test)]
    pub(crate) fn text(&self) -> String {
        self.spans.iter().map(|span| span.text.as_str()).collect()
    }

    fn width(&self) -> usize {
        spans_width(&self.spans)
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
/// The visual order is fixed — rule, state, detail, input — and so is the
/// order rows are given up in when the terminal is short: the input line
/// survives longest, then the state row, then the detail row, then the rule.
/// A row never moves relative to the ones that outlive it.
pub(crate) fn panel_view(
    layout: Layout,
    status: &Status,
    editor: &LineEditor,
    palette: &Palette,
    glyphs: &Glyphs,
) -> PanelView {
    let rect = layout.panel_rect();
    let width = layout.width() as usize;
    let panel_h = layout.panel_h();

    let mut rows = Vec::with_capacity(usize::from(panel_h));
    if panel_h >= 4 {
        rows.push(rule_row(status, width, palette, glyphs));
    }
    if panel_h >= 2 {
        rows.push(state_row(status, width, palette, glyphs));
    }
    if panel_h >= 3 {
        rows.push(detail_row(status, width, palette, glyphs));
    }
    let (input, cursor_col) = input_row(status, editor, width, palette, glyphs);
    rows.push(input);

    let cursor_x = cursor_col.min(width.saturating_sub(1)) as u16;
    let cursor_y = rect.y + rect.height.saturating_sub(1);
    PanelView {
        rows,
        cursor: (cursor_x, cursor_y),
    }
}

/// Row one: the prefill bar, or the rule that stands in for it.
///
/// Full bleed and full width in both states, so the row above the panel is
/// the same rule whatever is running and the bar arrives without moving
/// anything.
fn rule_row(status: &Status, width: usize, palette: &Palette, glyphs: &Glyphs) -> PanelRow {
    if let (
        Phase::Prefill,
        Some(Prefilling {
            done,
            total: Some(total),
        }),
    ) = (status.phase, status.prefill)
    {
        return bar_row(done, total, width, palette, glyphs);
    }
    PanelRow {
        spans: vec![PanelSpan::new(
            glyphs.bar_empty.repeat(width),
            palette.faint(),
        )],
    }
}

/// Row two: the chip, the phase, and the metrics.
fn state_row(status: &Status, width: usize, palette: &Palette, glyphs: &Glyphs) -> PanelRow {
    let chip_style = match status.phase {
        Phase::Idle => palette.muted(),
        Phase::Prefill => palette.accent(),
        Phase::Decode => palette.live(),
    };
    let base = vec![
        PanelSpan::new(glyphs.chip, chip_style),
        PanelSpan::new(" ", palette.primary()),
        PanelSpan::new(phase_word(status.phase), palette.primary()),
    ];
    // The model name is worth a row of its own only when nothing is running;
    // during a turn the row is about the turn.
    let named = match (status.phase, status.model.as_deref()) {
        (Phase::Idle, Some(model)) if !model.is_empty() => {
            let mut spans = base.clone();
            spans.push(PanelSpan::new(
                format!(" {} {model}", glyphs.separator),
                palette.muted(),
            ));
            spans
        }
        _ => base.clone(),
    };

    let right = state_segments(status, palette, glyphs);
    // Dropping the model name is cheaper than dropping a metric, and cheaper
    // still than clipping it to something that is no longer a name: it is the
    // only thing on the row that is not a number, and the only thing on it the
    // user already knows.
    let named_w = spans_width(&named);
    let left = if named_w > width
        || (fit_count(named_w, &right, width) == 0
            && fit_count(spans_width(&base), &right, width) > 0)
    {
        base
    } else {
        named
    };
    compose(left, right, width)
}

/// Row three: what the phase is doing, and how long it has been doing it.
fn detail_row(status: &Status, width: usize, palette: &Palette, glyphs: &Glyphs) -> PanelRow {
    let mut left = vec![PanelSpan::new(" ".repeat(INDENT), palette.primary())];
    let running = status.phase != Phase::Idle;
    let text = match status.detail.as_deref() {
        Some(detail) => detail.to_owned(),
        None if running => detail_facts(status, glyphs),
        None => HINTS.join(&format!(" {} ", glyphs.separator)),
    };
    if !text.is_empty() {
        if running {
            left.push(PanelSpan::new(
                format!("{} ", glyphs.gutter),
                palette.muted(),
            ));
        }
        left.push(PanelSpan::new(text, palette.muted()));
    }

    let right = if running {
        vec![vec![PanelSpan::new(
            format!(
                "{} {} {INTERRUPT_HINT}",
                clock(status.elapsed),
                glyphs.separator
            ),
            palette.muted(),
        )]]
    } else {
        Vec::new()
    };
    compose(left, right, width)
}

/// Row four: the prompt marker and whatever is being typed.
///
/// Returns the cursor's column as well, because the two cannot be computed
/// apart: the editor windows its own text and only it knows where the caret
/// landed inside that window.
fn input_row(
    status: &Status,
    editor: &LineEditor,
    width: usize,
    palette: &Palette,
    glyphs: &Glyphs,
) -> (PanelRow, usize) {
    // Leave the last cell of the row alone. Writing the bottom-right cell
    // with autowrap on arms the terminal's pending-wrap flag on a row outside
    // the scroll region, and terminals disagree about what happens next; not
    // writing it costs one column and no arguments.
    let field = width.saturating_sub(INDENT + 1);
    let (text, cursor_col) = editor.view(field);
    let mut spans = vec![
        PanelSpan::new(glyphs.prompt, palette.strong()),
        PanelSpan::new(" ", palette.primary()),
    ];
    if text.is_empty() && status.phase == Phase::Idle && status.detail.is_none() {
        // A placeholder, not text: quieter than anything the user could type,
        // and gone the moment they type it.
        spans.push(PanelSpan::new(clip(PLACEHOLDER, field), palette.faint()));
    } else {
        spans.push(PanelSpan::new(text, palette.primary()));
    }
    let mut row = PanelRow { spans };
    clip_row(&mut row, width);
    (row, INDENT + cursor_col)
}

/// The state row's right cluster, in the order it is read and dropped.
///
/// Exactly one segment is `primary` while something is running — the ETA
/// prefilling, the rate decoding — and none at all when idle, which is what
/// makes the focal metric findable without reading the row.
fn state_segments(status: &Status, palette: &Palette, glyphs: &Glyphs) -> Vec<Vec<PanelSpan>> {
    let dot = format!(" {} ", glyphs.separator);
    let mut segments: Vec<Vec<PanelSpan>> = Vec::new();
    match (status.phase, status.prefill) {
        (
            Phase::Prefill,
            Some(Prefilling {
                done,
                total: Some(total),
            }),
        ) => {
            let mut counts = format!("{done}/{total}");
            if let Some(percent) = percent(done as u64, total as u64) {
                counts.push_str(&format!("{dot}{percent}%"));
            }
            segments.push(vec![PanelSpan::new(counts, palette.muted())]);
            if let Some(eta) = status.eta {
                segments.push(vec![PanelSpan::new(
                    format!("~{} left", clock(eta)),
                    palette.primary(),
                )]);
            }
            if let Some(rate) = status.rate {
                segments.push(vec![PanelSpan::new(rate_text(rate), palette.muted())]);
            }
        }
        // No total to divide by: there is no percent, no ETA and no bar, so
        // the count and the rate carry the row on their own.
        (Phase::Prefill, Some(Prefilling { done, total: None })) => {
            segments.push(counted(done, "tok", status.rate, &dot, palette));
        }
        (Phase::Decode, _) => {
            segments.push(counted(status.tokens, "tok", status.rate, &dot, palette));
            segments.push(vec![context_span(status, palette, glyphs)]);
        }
        (Phase::Prefill, None) => {}
        (Phase::Idle, _) => {
            segments.push(vec![context_span(status, palette, glyphs)]);
            if let Some(hit) = status.hit_rate {
                segments.push(vec![PanelSpan::new(hit_text(hit), palette.muted())]);
            }
        }
    }
    segments
}

/// `47 tok · 2.1 tok/s`, with the rate — the focal metric — `primary` and the
/// count that gives it scale `muted`.
fn counted(
    count: usize,
    unit: &str,
    rate: Option<f64>,
    dot: &str,
    palette: &Palette,
) -> Vec<PanelSpan> {
    match rate {
        Some(rate) => vec![
            PanelSpan::new(format!("{count} {unit}{dot}"), palette.muted()),
            PanelSpan::new(rate_text(rate), palette.primary()),
        ],
        None => vec![PanelSpan::new(format!("{count} {unit}"), palette.muted())],
    }
}

/// `ctx 1071/4096 · 26%`, `alert` once the context is nearly spent.
fn context_span(status: &Status, palette: &Palette, glyphs: &Glyphs) -> PanelSpan {
    let (used, cap) = status.context;
    let mut text = format!("ctx {used}/{cap}");
    let share = percent(used as u64, cap as u64);
    // Nothing has been said yet: `0%` is noise next to `0/4096`.
    if let Some(share) = share.filter(|_| used > 0) {
        text.push_str(&format!(" {} {share}%", glyphs.separator));
    }
    let style = if share.is_some_and(|share| share >= CTX_ALERT) {
        palette.alert()
    } else {
        palette.muted()
    };
    PanelSpan::new(text, style)
}

/// The detail row's left text: what the phase is spending its time on, out of
/// the counters the runtime actually reports.
///
/// Prefill has no byte figure. `GenerateProgress::PrefillChunk` carries
/// positions and nothing else, and the `ForwardState` that holds the counters
/// is mutably borrowed by the generate call for its whole duration, so the UI
/// thread cannot read them between chunks. The segment is left out rather
/// than filled with a number derived from something else.
fn detail_facts(status: &Status, glyphs: &Glyphs) -> String {
    let mut parts: Vec<String> = Vec::new();
    match status.phase {
        Phase::Prefill => {
            if let Some((chunk, chunks)) = status.chunk {
                parts.push(format!("chunk {chunk}/{chunks}"));
            }
        }
        Phase::Decode => {
            if let Some(hit) = status.hit_rate {
                parts.push(hit_text(hit));
            }
        }
        Phase::Idle => {}
    }
    if let Some(bytes) = status.read_bytes {
        parts.push(format!("{} read", human_bytes(bytes)));
    }
    parts.join(&format!(" {} ", glyphs.separator))
}

fn phase_word(phase: Phase) -> &'static str {
    match phase {
        Phase::Idle => "ready",
        Phase::Prefill => "prefill",
        Phase::Decode => "decode",
    }
}

fn rate_text(rate: f64) -> String {
    format!("{rate:.1} tok/s")
}

fn hit_text(hit: f32) -> String {
    format!("hit {:.0}%", f64::from(hit.clamp(0.0, 1.0)) * 100.0)
}

/// `done * 100 / total`, truncated, or `None` when there is nothing to divide
/// by.
///
/// Truncated rather than rounded on purpose: a bar that has not reached the
/// end must not be able to say 100%.
pub(crate) fn percent(done: u64, total: u64) -> Option<u64> {
    (total > 0).then(|| done.min(total) * 100 / total)
}

/// `1m 33s`. Zero-padded seconds, so a live clock never changes width and
/// never reflows the row it sits in.
pub(crate) fn clock(elapsed: std::time::Duration) -> String {
    let seconds = elapsed.as_secs();
    format!("{}m {:02}s", seconds / 60, seconds % 60)
}

// ---------------------------------------------------------------------------
// the prefill bar
// ---------------------------------------------------------------------------

/// How many cells of a `width`-wide bar are filled, and whether the cell after
/// them is a half.
///
/// Two rules, both about honesty at the ends:
///
/// - the last cell fills only at literally `done == total`, so a bar that
///   reads full means the prefill *is* done;
/// - `total == 0` is nothing to do, which is done, not a division by zero.
pub(crate) fn bar_cells(done: usize, total: usize, width: usize) -> (usize, bool) {
    if width == 0 {
        return (0, false);
    }
    if done >= total {
        return (width, false);
    }
    let position = done as f64 / total as f64 * width as f64;
    let mut filled = position.floor().max(0.0) as usize;
    let mut half = (position - filled as f64) >= 0.5;
    if filled >= width {
        // 99.9% is not 100%.
        filled = width - 1;
        half = false;
    }
    (filled, half)
}

fn bar_row(
    done: usize,
    total: usize,
    width: usize,
    palette: &Palette,
    glyphs: &Glyphs,
) -> PanelRow {
    let (filled, half) = bar_cells(done, total, width);
    let empty = width - filled - usize::from(half);
    let mut spans = Vec::with_capacity(3);
    if filled > 0 {
        spans.push(PanelSpan::new(
            glyphs.bar_full.repeat(filled),
            palette.accent(),
        ));
    }
    if half {
        spans.push(PanelSpan::new(glyphs.bar_partial, palette.accent()));
    }
    if empty > 0 {
        spans.push(PanelSpan::new(
            glyphs.bar_empty.repeat(empty),
            palette.faint(),
        ));
    }
    PanelRow { spans }
}

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

fn spans_width(spans: &[PanelSpan]) -> usize {
    spans.iter().map(PanelSpan::width).sum()
}

/// Columns `segments` needs, separators included.
fn cluster_width(segments: &[Vec<PanelSpan>]) -> usize {
    let text: usize = segments.iter().map(|segment| spans_width(segment)).sum();
    text + GAP * segments.len().saturating_sub(1)
}

/// How many of `segments` survive beside a left cluster of `left_w` columns.
fn fit_count(left_w: usize, segments: &[Vec<PanelSpan>], width: usize) -> usize {
    let mut keep = segments.len();
    while keep > 0 && left_w + 1 + cluster_width(&segments[..keep]) + MARGIN > width {
        keep -= 1;
    }
    keep
}

/// Put a left cluster and a right cluster on one row `width` columns wide.
///
/// Right-cluster segments are dropped from the right until what is left fits
/// with at least one column between the clusters and [`MARGIN`] columns clear
/// at the end. The left cluster is clipped only once there is no right
/// cluster left to give up.
fn compose(left: Vec<PanelSpan>, right: Vec<Vec<PanelSpan>>, width: usize) -> PanelRow {
    if width == 0 {
        return PanelRow::default();
    }
    let left_w = spans_width(&left);
    let keep = fit_count(left_w, &right, width);
    let mut row = PanelRow { spans: left };
    if keep == 0 {
        clip_row(&mut row, width);
        pad_row(&mut row, width);
        return row;
    }
    let right = &right[..keep];
    let pad = width - MARGIN - left_w - cluster_width(right);
    row.spans
        .push(PanelSpan::new(" ".repeat(pad), Style::default()));
    for (index, segment) in right.iter().enumerate() {
        if index > 0 {
            row.spans
                .push(PanelSpan::new(" ".repeat(GAP), Style::default()));
        }
        row.spans.extend(segment.iter().cloned());
    }
    pad_row(&mut row, width);
    row
}

/// Drop whatever will not fit in `width` columns, span by span.
fn clip_row(row: &mut PanelRow, width: usize) {
    let mut used = 0;
    let mut keep = 0;
    for span in &mut row.spans {
        let span_w = span.width();
        if used + span_w <= width {
            used += span_w;
            keep += 1;
            continue;
        }
        span.text = clip(&span.text, width - used);
        keep += usize::from(!span.text.is_empty());
        break;
    }
    row.spans.truncate(keep);
}

/// Fill the row out to `width` with unstyled blanks.
///
/// Unstyled on purpose: a padded row must not extend a colour or an underline
/// to the right margin, which is the mistake that makes a status bar look
/// like a selection.
fn pad_row(row: &mut PanelRow, width: usize) {
    let short = width.saturating_sub(row.width());
    if short > 0 {
        row.spans
            .push(PanelSpan::new(" ".repeat(short), Style::default()));
    }
}

/// Truncate to `width` *characters*, which is the column count for
/// everything this module builds (ASCII plus single-width box and block
/// drawing).
pub(crate) fn clip(text: &str, width: usize) -> String {
    text.chars().take(width).collect()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// The palette every snapshot is taken under: the one every terminal
    /// gets today, since `Palette::detect` does not probe (see `style.rs`).
    fn palette() -> Palette {
        Palette::fallback()
    }

    fn view(status: &Status, editor: &LineEditor) -> PanelView {
        panel_view(
            Layout::new(80, 24),
            status,
            editor,
            &palette(),
            &Glyphs::UNICODE,
        )
    }

    fn rows(status: &Status, editor: &LineEditor) -> Vec<String> {
        view(status, editor)
            .rows
            .iter()
            .map(PanelRow::text)
            .collect()
    }

    /// Idle after a turn: a model name, a context share, a hit rate, and no
    /// focal metric at all.
    fn idle_status() -> Status {
        Status {
            phase: Phase::Idle,
            model: Some("Qwen3-30B-A3B".to_owned()),
            context: (1024, 4096),
            hit_rate: Some(0.87),
            ..Status::default()
        }
    }

    /// A quarter of the way through a 3,961-position prompt at the ~11 tok/s
    /// v0 manages, which is the case the bar exists for.
    fn prefill_status() -> Status {
        Status {
            phase: Phase::Prefill,
            model: Some("Qwen3-30B-A3B".to_owned()),
            prefill: Some(Prefilling {
                done: 1024,
                total: Some(3961),
            }),
            chunk: Some((2, 8)),
            elapsed: Duration::from_secs(93),
            context: (1024, 4096),
            rate: Some(1024.0 / 93.0),
            eta: Some(Duration::from_secs_f64(2937.0 * 93.0 / 1024.0)),
            ..Status::default()
        }
    }

    /// Forty-seven tokens in at about the 2 tok/s v0 decodes at.
    fn decode_status() -> Status {
        Status {
            phase: Phase::Decode,
            model: Some("Qwen3-30B-A3B".to_owned()),
            tokens: 47,
            elapsed: Duration::from_secs(22),
            context: (1071, 4096),
            hit_rate: Some(0.87),
            read_bytes: Some(3_650_722_202),
            rate: Some(47.0 / 22.0),
            ..Status::default()
        }
    }

    // -- the three states, exactly ------------------------------------------

    #[test]
    fn idle_after_a_turn_at_eighty_columns() {
        assert_eq!(
            rows(&idle_status(), &LineEditor::default()),
            [
                "────────────────────────────────────────────────────────────────────────────────",
                "▌ ready · Qwen3-30B-A3B                          ctx 1024/4096 · 25%   hit 87%  ",
                "  enter send · ctrl-c stop · ctrl-d quit                                        ",
                "› Ask ramvamp anything",
            ]
        );
    }

    #[test]
    fn prefill_at_eighty_columns() {
        assert_eq!(
            rows(&prefill_status(), &LineEditor::default()),
            [
                "━━━━━━━━━━━━━━━━━━━━╸───────────────────────────────────────────────────────────",
                "▌ prefill                          1024/3961 · 25%   ~4m 26s left   11.0 tok/s  ",
                "  └ chunk 2/8                                        1m 33s · esc to interrupt  ",
                "› ",
            ]
        );
    }

    #[test]
    fn decode_at_eighty_columns() {
        let mut editor = LineEditor::default();
        editor.insert_str("what happens if I");
        assert_eq!(
            rows(&decode_status(), &editor),
            [
                "────────────────────────────────────────────────────────────────────────────────",
                "▌ decode                              47 tok · 2.1 tok/s   ctx 1071/4096 · 26%  ",
                "  └ hit 87% · 3.4 GiB read                           0m 22s · esc to interrupt  ",
                "› what happens if I",
            ]
        );
    }

    #[test]
    fn a_fresh_start_claims_no_context_share_and_no_hit_rate() {
        let fresh = Status {
            phase: Phase::Idle,
            model: Some("Qwen3-30B-A3B".to_owned()),
            context: (0, 4096),
            ..Status::default()
        };
        let row = &rows(&fresh, &LineEditor::default())[1];
        assert_eq!(
            row.trim_end(),
            "▌ ready · Qwen3-30B-A3B                                             ctx 0/4096"
        );
        // `hit 0%` would be a claim about a cache that has not been asked for
        // anything yet.
        assert!(!row.contains("hit"), "{row:?}");
        assert!(!row.contains('%'), "{row:?}");
    }

    #[test]
    fn the_ascii_set_draws_the_same_rows_in_the_invariant_subset() {
        let view = panel_view(
            Layout::new(80, 24),
            &prefill_status(),
            &LineEditor::default(),
            &palette(),
            &Glyphs::ASCII,
        );
        let ascii_rows: Vec<String> = view.rows.iter().map(PanelRow::text).collect();
        let unicode = rows(&prefill_status(), &LineEditor::default());
        assert_eq!(
            ascii_rows
                .iter()
                .map(|row| row.trim_end())
                .collect::<Vec<_>>(),
            [
                "====================------------------------------------------------------------",
                "| prefill                          1024/3961 | 25%   ~4m 26s left   11.0 tok/s",
                "  + chunk 2/8                                        1m 33s | esc to interrupt",
                ">",
            ]
        );
        for (ascii, unicode) in ascii_rows.iter().zip(&unicode) {
            // Same shape, same columns, different characters: the layout code
            // never learns which set it is holding.
            assert!(ascii.is_ascii(), "{ascii:?} left the ASCII subset");
            assert_eq!(
                ascii.chars().count(),
                unicode.chars().count(),
                "{ascii:?} against {unicode:?}"
            );
        }
    }

    // -- the invariants -----------------------------------------------------

    #[test]
    fn every_state_puts_its_content_in_the_same_place() {
        for (name, status) in [
            ("idle", idle_status()),
            ("prefill", prefill_status()),
            ("decode", decode_status()),
        ] {
            let rows = rows(&status, &LineEditor::default());
            assert_eq!(rows.len(), 4, "{name}");
            // Row one is the rule or the bar, full bleed, full width.
            assert_eq!(rows[0].chars().count(), 80, "{name} rule");
            assert!(
                rows[0].chars().all(|c| "─━╸".contains(c)),
                "{name} rule: {:?}",
                rows[0]
            );
            // Rows two to four all start with two columns of gutter and put
            // their content in the third.
            for (index, row) in rows.iter().enumerate().skip(1) {
                let prefix: String = row.chars().take(INDENT).collect();
                assert!(
                    ["▌ ", "  ", "› "].contains(&prefix.as_str()),
                    "{name} row {index}: {row:?}"
                );
                assert_ne!(
                    row.chars().nth(INDENT),
                    Some(' '),
                    "{name} row {index} starts late: {row:?}"
                );
            }
            // And rows two and three end their metrics two columns short.
            for index in [1, 2] {
                let row = &rows[index];
                if row.trim_end().chars().count() > INDENT {
                    assert_eq!(row.chars().count(), 80, "{name} row {index}: {row:?}");
                }
            }
        }
    }

    #[test]
    fn the_right_cluster_ends_two_columns_short_of_the_edge() {
        for (name, status, rows_with_metrics) in [
            ("idle", idle_status(), vec![1]),
            ("prefill", prefill_status(), vec![1, 2]),
            ("decode", decode_status(), vec![1, 2]),
        ] {
            let rows = rows(&status, &LineEditor::default());
            for index in rows_with_metrics {
                assert_eq!(
                    rows[index].trim_end().chars().count(),
                    78,
                    "{name} row {index}: {:?}",
                    rows[index]
                );
            }
        }
    }

    #[test]
    fn exactly_one_metric_is_focal_while_something_is_running() {
        let palette = palette();
        for (name, status, focal) in [
            ("idle", idle_status(), 0),
            ("prefill", prefill_status(), 1),
            ("decode", decode_status(), 1),
        ] {
            let segments = state_segments(&status, &palette, &Glyphs::UNICODE);
            let primary = segments
                .iter()
                .flatten()
                .filter(|span| span.style == palette.primary() && !span.text.trim().is_empty())
                .count();
            assert_eq!(primary, focal, "{name}: {segments:?}");
        }
    }

    #[test]
    fn the_rows_never_move_between_states() {
        // The one invariant a user feels: a phase change must not shift a
        // row, or the eye has to find the numbers again every time.
        let mut kinds: Vec<Vec<String>> = Vec::new();
        for status in [idle_status(), prefill_status(), decode_status()] {
            kinds.push(
                rows(&status, &LineEditor::default())
                    .iter()
                    .map(|row| row.chars().take(INDENT).collect::<String>())
                    .collect(),
            );
        }
        for kind in &kinds {
            assert_eq!(kind.len(), 4);
            assert_eq!(kind[1], "▌ ");
            assert_eq!(kind[2], "  ");
            assert_eq!(kind[3], "› ");
        }
    }

    #[test]
    fn the_context_segment_warns_before_the_context_is_spent() {
        let palette = palette();
        let mut status = idle_status();
        for (used, alert) in [
            (0, false),
            (1024, false),
            (3891, false),
            (3892, true),
            (4096, true),
        ] {
            status.context = (used, 4096);
            let span = context_span(&status, &palette, &Glyphs::UNICODE);
            assert_eq!(
                span.style == palette.alert(),
                alert,
                "{used}/4096 was {:?}",
                span.style
            );
        }
    }

    // -- the bar ------------------------------------------------------------

    #[test]
    fn the_bar_fills_by_halves_and_only_reads_full_when_it_is() {
        // Nothing done: an empty track, full width.
        assert_eq!(bar_cells(0, 3961, 80), (0, false));
        // The approved half-step: 20.68 cells is twenty and a half.
        assert_eq!(bar_cells(1024, 3961, 80), (20, true));
        // Just under a whole cell of remainder rounds down, not up.
        assert_eq!(bar_cells(1000, 3961, 80), (20, false));
        // 99.9% is not 100%: the last cell stays a half.
        assert_eq!(bar_cells(3960, 3961, 80), (79, true));
        assert_eq!(bar_cells(3960, 3961, 10), (9, true));
        // And exactly 100% is.
        assert_eq!(bar_cells(3961, 3961, 80), (80, false));
        // Nothing to do is done, not a division by zero.
        assert_eq!(bar_cells(0, 0, 80), (80, false));
        // Degenerate widths and overruns stay in range.
        assert_eq!(bar_cells(1, 2, 0), (0, false));
        assert_eq!(bar_cells(9_000, 3961, 80), (80, false));
    }

    #[test]
    fn the_bar_never_draws_more_cells_than_it_was_given() {
        let palette = palette();
        for width in 0..=120usize {
            for done in [0, 1, 512, 3960, 3961, 9_000] {
                let (filled, half) = bar_cells(done, 3961, width);
                assert!(filled + usize::from(half) <= width, "{done} at {width}");
                let row = bar_row(done, 3961, width, &palette, &Glyphs::UNICODE);
                assert_eq!(row.text().chars().count(), width, "{done} at {width}");
            }
        }
    }

    #[test]
    fn an_unknown_total_gets_no_bar_and_no_percent() {
        let status = Status {
            phase: Phase::Prefill,
            prefill: Some(Prefilling {
                done: 1024,
                total: None,
            }),
            elapsed: Duration::from_secs(93),
            rate: Some(11.0),
            // An ETA cannot be had without a total, and the panel must not
            // invent one.
            eta: None,
            ..Status::default()
        };
        let rows = rows(&status, &LineEditor::default());
        assert_eq!(
            rows[0],
            "────────────────────────────────────────────────────────────────────────────────",
            "an unknown total draws the plain rule, not an indeterminate bar"
        );
        assert_eq!(
            rows[1].trim_end(),
            "▌ prefill                                                1024 tok · 11.0 tok/s"
        );
        assert!(!rows[1].contains('%'), "{:?}", rows[1]);
        assert!(!rows[1].contains("left"), "{:?}", rows[1]);
    }

    // -- degradation --------------------------------------------------------

    #[test]
    fn segments_are_dropped_from_the_right_as_the_terminal_narrows() {
        let status = decode_status();
        let editor = LineEditor::default();
        let palette = palette();
        let row = |width: u16| {
            panel_view(
                Layout::new(width, 24),
                &status,
                &editor,
                &palette,
                &Glyphs::UNICODE,
            )
            .rows[1]
                .text()
                .trim_end()
                .to_owned()
        };
        assert_eq!(
            row(80),
            "▌ decode                              47 tok · 2.1 tok/s   ctx 1071/4096 · 26%"
        );
        // The context goes first, leaving the focal metric.
        assert_eq!(row(45), "▌ decode                 47 tok · 2.1 tok/s");
        // Then the metric, leaving the phase.
        assert_eq!(row(20), "▌ decode");
    }

    #[test]
    fn the_model_name_is_given_up_before_the_last_metric() {
        let status = idle_status();
        let editor = LineEditor::default();
        let palette = palette();
        let row = |width: u16| {
            panel_view(
                Layout::new(width, 24),
                &status,
                &editor,
                &palette,
                &Glyphs::UNICODE,
            )
            .rows[1]
                .text()
                .trim_end()
                .to_owned()
        };
        assert_eq!(
            row(80),
            "▌ ready · Qwen3-30B-A3B                          ctx 1024/4096 · 25%   hit 87%"
        );
        // Wide enough for the name and one metric.
        assert_eq!(row(45), "▌ ready · Qwen3-30B-A3B ctx 1024/4096 · 25%");
        // Not wide enough for both: the number stays, the name goes.
        assert_eq!(row(30), "▌ ready  ctx 1024/4096 · 25%");
        // Not wide enough for either.
        assert_eq!(row(12), "▌ ready");
    }

    #[test]
    fn the_bar_keeps_its_full_width_however_narrow_the_terminal_is() {
        let status = prefill_status();
        let editor = LineEditor::default();
        let palette = palette();
        for width in 4..=120u16 {
            let view = panel_view(
                Layout::new(width, 24),
                &status,
                &editor,
                &palette,
                &Glyphs::UNICODE,
            );
            assert_eq!(
                view.rows[0].text().chars().count(),
                usize::from(width),
                "{width} columns"
            );
        }
    }

    #[test]
    fn no_row_ever_exceeds_its_width() {
        let palette = palette();
        for status in [idle_status(), prefill_status(), decode_status()] {
            let mut editor = LineEditor::default();
            editor.insert_str("a line long enough to have to be windowed by the editor");
            for width in 1..=120u16 {
                for height in 1..=10u16 {
                    let layout = Layout::new(width, height);
                    let view = panel_view(layout, &status, &editor, &palette, &Glyphs::UNICODE);
                    assert_eq!(
                        view.rows.len(),
                        usize::from(layout.panel_h()),
                        "{width}x{height}"
                    );
                    for row in &view.rows {
                        assert!(
                            row.text().chars().count() <= usize::from(width),
                            "{width}x{height}: {:?}",
                            row.text()
                        );
                    }
                    assert!(view.cursor.0 < width, "{width}x{height}");
                }
            }
        }
    }

    // -- layout -------------------------------------------------------------

    #[test]
    fn scroll_region_sequences_are_decstbm() {
        assert_eq!(set_scroll_region(1, 20), "\x1b[1;20r");
        assert_eq!(set_scroll_region(1, 1), "\x1b[1;1r");
        assert_eq!(RESET_SCROLL_REGION, "\x1b[r");
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
    fn a_squeezed_panel_gives_up_rows_from_the_bottom_of_the_priority_list() {
        let status = decode_status();
        let editor = LineEditor::default();
        let palette = palette();
        let kinds = |height: u16| -> Vec<String> {
            panel_view(
                Layout::new(80, height),
                &status,
                &editor,
                &palette,
                &Glyphs::UNICODE,
            )
            .rows
            .iter()
            .map(|row| row.text().chars().take(INDENT).collect::<String>())
            .collect()
        };
        // The rule goes first, then the detail row; the input line and the
        // state row are the last two standing, in that order.
        assert_eq!(kinds(24), ["──", "▌ ", "  ", "› "]);
        assert_eq!(kinds(7), ["▌ ", "  ", "› "]);
        assert_eq!(kinds(5), ["▌ ", "› "]);
        assert_eq!(kinds(3), ["› "]);
    }

    #[test]
    fn the_cursor_follows_the_input_line() {
        let mut editor = LineEditor::default();
        editor.insert_str("hi");
        let typed = view(&idle_status(), &editor);
        assert_eq!(typed.rows[3].text(), "› hi");
        // Row 23 is the last row of an 80x24 terminal; column 4 is one past
        // "› hi".
        assert_eq!(typed.cursor, (4, 23));
        // The placeholder is not text: the caret stays at the start of it.
        assert_eq!(view(&idle_status(), &LineEditor::default()).cursor, (2, 23));
    }

    #[test]
    fn the_placeholder_only_shows_when_nothing_is_typed_and_nothing_is_running() {
        let empty = LineEditor::default();
        assert_eq!(rows(&idle_status(), &empty)[3], "› Ask ramvamp anything");
        assert_eq!(rows(&prefill_status(), &empty)[3], "› ");
        assert_eq!(rows(&decode_status(), &empty)[3], "› ");
        let mut typed = LineEditor::default();
        typed.insert_char('a');
        assert_eq!(rows(&idle_status(), &typed)[3], "› a");
    }

    #[test]
    fn a_transient_detail_replaces_the_row_it_lands_on() {
        let mut status = idle_status();
        status.detail = Some("loading the model...".to_owned());
        assert_eq!(
            rows(&status, &LineEditor::default())[2].trim_end(),
            "  loading the model..."
        );
        let mut status = decode_status();
        status.detail = Some("stopping this reply...".to_owned());
        assert_eq!(
            rows(&status, &LineEditor::default())[2].trim_end(),
            "  └ stopping this reply...                           0m 22s · esc to interrupt"
        );
    }

    #[test]
    fn clock_and_percent_are_stable_and_truncating() {
        assert_eq!(clock(Duration::ZERO), "0m 00s");
        assert_eq!(clock(Duration::from_secs(9)), "0m 09s");
        assert_eq!(clock(Duration::from_secs(93)), "1m 33s");
        assert_eq!(clock(Duration::from_secs(266)), "4m 26s");
        assert_eq!(clock(Duration::from_secs(3_600)), "60m 00s");
        // Truncated, so 99.9% cannot round up to a hundred.
        assert_eq!(percent(0, 0), None);
        assert_eq!(percent(0, 4096), Some(0));
        assert_eq!(percent(1024, 4096), Some(25));
        assert_eq!(percent(3960, 3961), Some(99));
        assert_eq!(percent(3961, 3961), Some(100));
        assert_eq!(percent(9_000, 3961), Some(100));
    }
}
