//! A terminal harness for the chat REPL: the transcript scrolls natively and
//! a small status panel stays pinned to the bottom of the screen.
//!
//! # Why it exists
//!
//! v0 prefills at about 11 tok/s and decodes at about 2. A 4K prompt is
//! therefore roughly six minutes during which a line-oriented REPL prints
//! nothing at all. The single feature that pays for this module is the
//! prefill progress bar; everything else is scaffolding around it.
//!
//! # How the screen is split
//!
//! The terminal keeps the transcript. A DECSTBM scroll region
//! (`ESC [ top ; bottom r`) is set over every row above the panel, and
//! transcript text is written into it with an ordinary `Print`. The terminal
//! then wraps it, scrolls it, and — the part that matters — lets the user
//! select and copy it with soft-wrap metadata intact.
//!
//! ratatui draws the panel and nothing else, through
//! [`Viewport::Fixed`](ratatui::Viewport::Fixed) pinned to the bottom rows.
//! Two ratatui features are ruled out on purpose:
//!
//! - `Terminal::insert_before` allocates a full-width `Buffer` and emits
//!   every cell with no diff. That writes trailing spaces, extends styled
//!   backgrounds to the right margin, and destroys the soft-wrap metadata, so
//!   selecting a wrapped paragraph copies hard breaks.
//! - `Viewport::Inline` duplicates itself into scrollback on resize under a
//!   draw-every-tick pattern (ratatui issue #2666, reproduced on 0.30.2).
//!
//! Neither is called here, which is also why the `scrolling-regions` cargo
//! feature is not enabled: it only alters `insert_before`.
//!
//! # Wiring
//!
//! This module is presentation only. Nothing in it loads a model, and
//! [`selftest`] drives the whole surface with fabricated numbers so the panel
//! can be reviewed in seconds rather than in six minutes. A later change
//! points the real generate loop at [`Harness`].

mod input;
mod panel;

use std::io::{self, Write as _};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind, KeyModifiers,
};
use ratatui::crossterm::style::Print;
use ratatui::crossterm::terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode, size};
use ratatui::crossterm::tty::IsTty as _;
use ratatui::crossterm::{QueueableCommand as _, cursor};
use ratatui::text::{Line, Span};
use ratatui::{Frame, Terminal, TerminalOptions, Viewport};

use crate::repl::CONTEXT_CAP;
use input::LineEditor;
use panel::Layout;

/// What the runtime is doing right now.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Phase {
    #[default]
    Idle,
    Prefill,
    Decode,
}

/// Everything the panel shows, as of one instant.
///
/// The caller owns this: the harness never derives a number for itself, so
/// what the panel says and what the run reports cannot drift.
#[derive(Clone, Debug, Default)]
pub struct Status {
    pub phase: Phase,
    /// `(done, total)` prompt positions, while prefill is running.
    pub prefill: Option<(usize, usize)>,
    /// Tokens decoded so far this turn.
    pub tokens: usize,
    /// Time the current phase has been running.
    pub elapsed: Duration,
    /// `(used, cap)` context positions.
    pub context: (usize, usize),
    /// Expert cache hit rate, `0.0..=1.0`.
    pub hit_rate: Option<f32>,
    /// One short line, e.g. an error.
    pub detail: Option<String>,
}

/// Something the user did.
#[derive(Clone, Debug)]
pub enum UiEvent {
    /// Enter on a non-empty line. May contain newlines, from a paste.
    Submit(String),
    /// Ctrl-C while a turn is running.
    Interrupt,
    /// Ctrl-D on an empty line, or Ctrl-C while idle.
    Exit,
    /// The terminal was resized. The caller may ignore this; the harness has
    /// already redrawn.
    Redraw,
}

/// Set while the terminal is in the state [`Harness::enter`] left it in.
///
/// Global because the panic hook is global. It is also how [`Harness`]
/// notices that a *caught* unwind (this crate raises two of them as control
/// flow, see `ChatAbort` in `main.rs`) ran the hook and reset the terminal
/// underneath a harness that is still alive: the next call re-arms instead
/// of drawing into a terminal that is no longer in raw mode.
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// The pinned panel and the scroll region above it.
///
/// [`Drop`] restores the terminal, and so does the panic hook installed by
/// [`Harness::enter`]; either is safe to run twice.
pub struct Harness {
    term: Terminal<CrosstermBackend<io::Stdout>>,
    layout: Layout,
    editor: LineEditor,
    status: Status,
    /// Set by [`Harness::set_status`], cleared by a draw.
    dirty: bool,
    /// Throttles status-driven redraws; input and resize always redraw.
    last_draw: Instant,
    /// Set by [`Harness::suspend`]: the terminal is the user's again and
    /// this harness must not touch it.
    suspended: bool,
}

/// Floor on status-driven redraws. Prefill advances a few dozen times a
/// second and there is no point drawing faster than that.
const REDRAW_INTERVAL: Duration = Duration::from_millis(33);

/// Why [`Harness::apply_layout`] is claiming the scroll region, which decides
/// whether the rows it is about to park on are ours to erase.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arrival {
    /// First time in. The row is where a cursor query said we ended up after
    /// the reserving newlines, or `None` if the terminal would not say; the
    /// rows around it are the user's output and must survive.
    Enter(Option<u16>),
    /// A resize or a re-arm. The panel moved, so the rows it used to occupy
    /// are ours to clean up.
    Relayout,
}

impl Harness {
    /// Take over the bottom of the terminal.
    ///
    /// Fails when stdout is not a terminal, which is also what keeps this
    /// out of headless runs: `scripts/cold_bench.py` and the sweep scripts
    /// parse the timing lines the CLI writes, and a status row is not
    /// something they should ever have to see.
    pub fn enter() -> anyhow::Result<Self> {
        let mut out = io::stdout();
        anyhow::ensure!(
            out.is_tty(),
            "the terminal harness needs a terminal on stdout"
        );
        let (width, height) = size().unwrap_or((80, 24));
        let layout = Layout::new(width, height);

        install_panic_hook();

        // Order copied from OpenAI Codex CLI's TUI setup and CodeWhale's
        // terminal guard (both MIT/Apache-2.0, compatible with this
        // project's MIT OR Apache-2.0): raw mode, then bracketed paste, then
        // reserve the panel rows, then claim the scroll region.
        enable_raw_mode().context("entering raw mode")?;
        ACTIVE.store(true, Ordering::SeqCst);
        out.queue(EnableBracketedPaste)?;
        // Print the panel's rows so they exist below whatever was on screen.
        // No scroll region is set yet, so this scrolls the whole screen and
        // the transcript above it goes to scrollback intact.
        for _ in 0..layout.panel_h() {
            out.queue(Print("\r\n"))?;
        }
        out.flush()?;

        let mut harness = Self {
            term: build_terminal(layout)?,
            layout,
            editor: LineEditor::default(),
            status: Status::default(),
            dirty: true,
            last_draw: Instant::now() - REDRAW_INTERVAL,
            suspended: false,
        };
        // Where the newlines left us, so entering does not open a gap under
        // a half-full screen. A terminal that will not answer the query is
        // not an error; the transcript just starts at the bottom of the
        // region. Measured cost: 3 ms against a terminal that answers, and a
        // one-off 2 s against one that never does — crossterm's DSR timeout,
        // which it does not expose. Once per session, so it stays.
        let park = cursor::position().ok().map(|(_, row)| row);
        harness.apply_layout(Arrival::Enter(park))?;
        harness.draw()?;
        Ok(harness)
    }

    /// Write model output, user turns, banners — anything that belongs in
    /// the scrollback — into the scroll region.
    ///
    /// Partial lines are expected: decoded tokens arrive mid-word and the
    /// cursor simply stays where it is. Newlines are translated to CRLF,
    /// because raw mode does not do that for us.
    pub fn write_transcript(&mut self, text: &str) -> io::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if self.suspended {
            // The terminal belongs to the caller again; still deliver the
            // text, just without the escape juggling.
            let mut out = io::stdout();
            out.write_all(text.as_bytes())?;
            return out.flush();
        }
        self.rearm()?;
        let mut out = io::stdout();
        // The panel draw left the cursor on the input line; DECRC puts it
        // back where the transcript left off, and DECSC records where this
        // write ended up — after any scrolling it caused.
        out.queue(cursor::RestorePosition)?;
        out.write_all(to_crlf(text).as_bytes())?;
        out.queue(cursor::SavePosition)?;
        out.flush()?;
        self.dirty = true;
        self.draw()
    }

    /// Replace what the panel is showing.
    ///
    /// Cheap and infallible by contract, so it can be called per decoded
    /// token or per prefilled position. The draw it triggers is throttled
    /// and its errors are swallowed; a caller that wants to see a failure
    /// should look at [`Harness::poll`], which it has to call anyway to
    /// notice Ctrl-C.
    pub fn set_status(&mut self, status: Status) {
        self.status = status;
        self.dirty = true;
        let _ = self.maybe_draw();
    }

    /// Wait up to `timeout` for the user to do something.
    ///
    /// Polls crossterm on the calling thread, so there is no input pump
    /// thread and therefore no dead-reader heartbeat to get wrong. A timeout
    /// is not an event: the harness has already refreshed the panel and
    /// returns `None`.
    pub fn poll(&mut self, timeout: Duration) -> io::Result<Option<UiEvent>> {
        if self.suspended {
            return Ok(None);
        }
        self.rearm()?;
        self.maybe_draw()?;
        if !event::poll(timeout)? {
            self.maybe_draw()?;
            return Ok(None);
        }
        let event = event::read()?;
        let out = self.handle(event)?;
        self.draw()?;
        Ok(out)
    }

    /// Give the terminal back, for a clean error exit or a `Drop`.
    ///
    /// Idempotent. Afterwards [`Harness::write_transcript`] degrades to a
    /// plain write and [`Harness::poll`] reports nothing.
    pub fn suspend(&mut self) -> io::Result<()> {
        if self.suspended {
            return Ok(());
        }
        self.suspended = true;
        ACTIVE.store(false, Ordering::SeqCst);
        restore_terminal()
    }

    fn handle(&mut self, event: Event) -> io::Result<Option<UiEvent>> {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                Ok(self.handle_key(key.code, key.modifiers))
            }
            Event::Paste(text) => {
                self.editor.insert_paste(&text);
                Ok(None)
            }
            Event::Resize(width, height) => {
                self.layout = Layout::new(width, height);
                self.term = build_terminal(self.layout)?;
                self.apply_layout(Arrival::Relayout)?;
                Ok(Some(UiEvent::Redraw))
            }
            _ => Ok(None),
        }
    }

    fn handle_key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> Option<UiEvent> {
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);
        let alt = modifiers.contains(KeyModifiers::ALT);
        match code {
            // No signal handler: in raw mode Ctrl-C is just a keystroke.
            // It stops the turn that is running, and leaves when there is
            // nothing to stop.
            KeyCode::Char('c') if ctrl => Some(if self.status.phase == Phase::Idle {
                UiEvent::Exit
            } else {
                UiEvent::Interrupt
            }),
            KeyCode::Char('d') if ctrl => self.editor.is_empty().then_some(UiEvent::Exit),
            KeyCode::Char('u') if ctrl => {
                self.editor.clear();
                None
            }
            KeyCode::Char('a') if ctrl => {
                self.editor.home();
                None
            }
            KeyCode::Char('e') if ctrl => {
                self.editor.end();
                None
            }
            KeyCode::Char(c) if !ctrl && !alt => {
                self.editor.insert_char(c);
                None
            }
            KeyCode::Backspace => {
                self.editor.backspace();
                None
            }
            KeyCode::Left => {
                self.editor.left();
                None
            }
            KeyCode::Right => {
                self.editor.right();
                None
            }
            KeyCode::Home => {
                self.editor.home();
                None
            }
            KeyCode::End => {
                self.editor.end();
                None
            }
            KeyCode::Enter => {
                if self.editor.text().trim().is_empty() {
                    // Whitespace only: reprompt, do not disturb the
                    // transcript. Same rule the line REPL uses.
                    self.editor.clear();
                    None
                } else {
                    Some(UiEvent::Submit(self.editor.take()))
                }
            }
            _ => None,
        }
    }

    /// Claim the scroll region for the current layout and park the
    /// transcript cursor inside it.
    ///
    /// Rows already in scrollback are not reflowed — that is out of scope,
    /// and this is where it would have to happen.
    fn apply_layout(&mut self, arrival: Arrival) -> io::Result<()> {
        let mut out = io::stdout();
        if !self.layout.usable() {
            // Too small to split. Hand the whole screen back rather than
            // leaving a region that spans one row.
            out.queue(Print(panel::RESET_SCROLL_REGION))?;
            return out.flush();
        }
        let bottom = self.layout.region_bottom();
        out.queue(Print(panel::set_scroll_region(1, bottom)))?;
        // DECSTBM homes the cursor, so this move is mandatory, not tidiness.
        let park = match arrival {
            Arrival::Enter(row) => row.unwrap_or(u16::MAX),
            Arrival::Relayout => u16::MAX,
        };
        out.queue(cursor::MoveTo(0, park.min(bottom - 1)))?;
        if arrival == Arrival::Relayout {
            // The panel moved, so the row the transcript is about to resume
            // on, and the band the panel is about to move into, still hold
            // the old panel's characters. Wiping from here down costs
            // nothing — ratatui repaints the panel from an empty buffer — and
            // stops the next token being written on top of a stale status
            // row. Rows *above* this one are scrollback and stay as they are.
            //
            // Entering must not do this: the parked row is then the user's
            // last line of output, not ours.
            out.queue(Clear(ClearType::FromCursorDown))?;
        }
        out.queue(cursor::SavePosition)?;
        out.flush()
    }

    /// Re-enter after a caught unwind ran the panic hook.
    fn rearm(&mut self) -> io::Result<()> {
        if ACTIVE.load(Ordering::SeqCst) {
            return Ok(());
        }
        enable_raw_mode()?;
        let mut out = io::stdout();
        out.queue(EnableBracketedPaste)?;
        out.flush()?;
        ACTIVE.store(true, Ordering::SeqCst);
        self.apply_layout(Arrival::Relayout)?;
        self.dirty = true;
        Ok(())
    }

    fn maybe_draw(&mut self) -> io::Result<()> {
        if self.dirty && self.last_draw.elapsed() >= REDRAW_INTERVAL {
            self.draw()?;
        }
        Ok(())
    }

    fn draw(&mut self) -> io::Result<()> {
        if self.suspended || !self.layout.usable() {
            self.dirty = false;
            return Ok(());
        }
        let view = panel::panel_view(self.layout, &self.status, &self.editor);
        self.term.draw(|frame| render(frame, &view))?;
        self.dirty = false;
        self.last_draw = Instant::now();
        Ok(())
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        let _ = self.suspend();
    }
}

fn build_terminal(layout: Layout) -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    // A fixed viewport is never autoresized, and `Terminal::resize` moves a
    // horizontally shrinking one to row 0 — straight over the transcript.
    // Rebuilding costs two empty buffers and guarantees the next draw
    // repaints every panel cell.
    Terminal::with_options(
        CrosstermBackend::new(io::stdout()),
        TerminalOptions {
            viewport: Viewport::Fixed(layout.panel_rect()),
        },
    )
}

fn render(frame: &mut Frame, view: &panel::PanelView) {
    let area = frame.area();
    if area.width == 0 || area.height == 0 {
        return;
    }
    for (offset, row) in view.rows.iter().enumerate() {
        let Ok(offset) = u16::try_from(offset) else {
            break;
        };
        if offset >= area.height {
            break;
        }
        let rect = ratatui::layout::Rect::new(area.x, area.y + offset, area.width, 1);
        frame.render_widget(Line::from(Span::styled(row.text.as_str(), row.style)), rect);
    }
    // Setting a cursor position is also what keeps the cursor *visible*:
    // ratatui hides it for a frame that does not ask for one.
    frame.set_cursor_position(view.cursor);
}

/// Translate line endings for raw mode, where a bare LF moves down without
/// returning to column 0.
fn to_crlf(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push_str("\r\n");
            }
            '\n' => out.push_str("\r\n"),
            c => out.push(c),
        }
    }
    out
}

/// Undo [`Harness::enter`], in reverse.
///
/// Every step runs even when an earlier one failed, so a half-initialised
/// terminal still recovers; the first error is the one reported.
fn restore_terminal() -> io::Result<()> {
    let mut first: Option<io::Error> = None;
    macro_rules! step {
        ($call:expr) => {
            if let Err(error) = $call {
                first.get_or_insert(error);
            }
        };
    }
    let mut out = io::stdout();
    step!(out.queue(DisableBracketedPaste));
    step!(out.queue(Print(panel::RESET_SCROLL_REGION)));
    // Below the transcript rather than below the panel: DECRC lands on the
    // line the transcript stopped at, the newline steps past it, and the
    // clear takes the panel with it. What is left in scrollback is the
    // conversation and nothing else.
    step!(out.queue(cursor::RestorePosition));
    step!(out.queue(Print("\r\n")));
    step!(out.queue(Clear(ClearType::FromCursorDown)));
    step!(out.queue(cursor::Show));
    step!(out.flush());
    step!(disable_raw_mode());
    match first {
        None => Ok(()),
        Some(error) => Err(error),
    }
}

/// Restore the terminal on panic, then let the hook that was already there
/// decide what to print.
///
/// Chaining matters: `main.rs` installs `hush_control_flow_panics`, which
/// swallows two deliberate unwinds this crate uses as control flow. Replacing
/// it would turn those into "thread panicked" dumps.
fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if ACTIVE.swap(false, Ordering::SeqCst) {
                let _ = restore_terminal();
            }
            previous(info);
        }));
    });
}

// ---------------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------------

/// Drive the whole harness with fabricated numbers and no model.
///
/// Hidden entry point, wired in `main.rs`:
///
/// ```text
/// RAMVAMP_TUI_SELFTEST=1 cargo run -p ramvamp
/// ```
///
/// Runs one fake turn immediately — a 3,961-position prefill compressed into
/// a few seconds, then about forty tokens of decode at roughly the 2 tok/s
/// the real runtime manages — and then waits for more. Ctrl-C stops a turn,
/// Ctrl-D leaves.
pub fn selftest() -> anyhow::Result<()> {
    let mut harness = Harness::enter()?;
    harness.write_transcript(BANNER)?;

    let mut outcome = fake_turn(&mut harness, "Why is prefill the slow part?")?;
    while outcome != Outcome::Exit {
        harness.set_status(idle_status());
        match harness.poll(Duration::from_millis(100))? {
            Some(UiEvent::Submit(line)) => outcome = fake_turn(&mut harness, &line)?,
            Some(UiEvent::Exit) => break,
            _ => {}
        }
    }
    harness.write_transcript("\nself-test done.\n")?;
    harness.suspend()?;
    Ok(())
}

const BANNER: &str = "\
ramvamp terminal harness — self-test. No model is loaded and every number \
below is fabricated.\r\n\
The transcript above the rule is written by the terminal itself, so it \
scrolls, wraps and selects natively.\r\n\
Type to edit the input line, paste multi-line text, Enter to run another \
fake turn, Ctrl-C to interrupt one, Ctrl-D to leave.\n";

/// Fabricated prompt length: a full 4K-ish prompt, which is the case the
/// progress bar exists for.
const FAKE_PROMPT_POSITIONS: usize = 3961;

/// Fabricated decode length.
const FAKE_DECODE_TOKENS: usize = 40;

/// Positions per second the fabricated prefill clock is scaled to, so the
/// status row shows the rate v0 actually manages.
const FAKE_PREFILL_RATE: f64 = 11.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Outcome {
    Finished,
    Interrupted,
    Exit,
}

fn idle_status() -> Status {
    Status {
        phase: Phase::Idle,
        context: (0, CONTEXT_CAP),
        detail: Some("idle — Enter runs a fake turn, Ctrl-D leaves".to_string()),
        ..Status::default()
    }
}

fn fake_turn(harness: &mut Harness, prompt: &str) -> io::Result<Outcome> {
    harness.write_transcript(&format!("\nyou> {prompt}\n"))?;

    // Prefill. The real thing is 3,961 positions at about 11 tok/s, i.e. six
    // minutes; this walks the same bar in about three seconds so the panel
    // can actually be reviewed. The *clock* is fabricated to match the real
    // rate rather than measured, so the row reads like a real run instead of
    // claiming four figures of tok/s.
    let mut done = 0;
    while done < FAKE_PROMPT_POSITIONS {
        done = (done + 53).min(FAKE_PROMPT_POSITIONS);
        harness.set_status(Status {
            phase: Phase::Prefill,
            prefill: Some((done, FAKE_PROMPT_POSITIONS)),
            tokens: 0,
            elapsed: Duration::from_secs_f64(done as f64 / FAKE_PREFILL_RATE),
            context: (done, CONTEXT_CAP),
            hit_rate: Some(fake_hit_rate(done)),
            detail: None,
        });
        match harness.poll(Duration::from_millis(40))? {
            Some(UiEvent::Interrupt) => return interrupted(harness),
            Some(UiEvent::Exit) => return Ok(Outcome::Exit),
            _ => {}
        }
    }

    // Decode, at roughly the real 2 tok/s, in fragments that land mid-word
    // on purpose: that is what the real token stream looks like, and it is
    // what `write_transcript` has to survive.
    let started = Instant::now();
    harness.set_status(decode_status(0, Duration::ZERO));
    harness.write_transcript("bot> ")?;
    for (index, piece) in fake_tokens(LOREM, FAKE_DECODE_TOKENS).iter().enumerate() {
        let deadline = Instant::now() + Duration::from_millis(500);
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            match harness.poll(remaining)? {
                Some(UiEvent::Interrupt) => return interrupted(harness),
                Some(UiEvent::Exit) => return Ok(Outcome::Exit),
                _ => {}
            }
        }
        harness.write_transcript(piece)?;
        harness.set_status(decode_status(index + 1, started.elapsed()));
    }
    harness.write_transcript("\n")?;
    Ok(Outcome::Finished)
}

/// Decode is the one phase the self-test runs at the real speed, so its
/// clock is measured rather than fabricated.
fn decode_status(tokens: usize, elapsed: Duration) -> Status {
    Status {
        phase: Phase::Decode,
        prefill: None,
        tokens,
        elapsed,
        context: (FAKE_PROMPT_POSITIONS + tokens, CONTEXT_CAP),
        hit_rate: Some(fake_hit_rate(FAKE_PROMPT_POSITIONS + tokens)),
        detail: None,
    }
}

fn interrupted(harness: &mut Harness) -> io::Result<Outcome> {
    harness.write_transcript("\n[interrupted]\n")?;
    Ok(Outcome::Interrupted)
}

/// A hit rate that climbs and wobbles, so the panel is exercised rather than
/// shown one constant.
fn fake_hit_rate(step: usize) -> f32 {
    let warm = 1.0 - (-(step as f32) / 900.0).exp();
    (0.55 + 0.4 * warm + 0.02 * ((step as f32) / 7.0).sin()).clamp(0.0, 1.0)
}

const LOREM: &str = "Experts stream from NVMe on demand, so the resident set \
stays near three gigabytes while the model on disk is thirty billion \
parameters wide. Prefill dominates the wall clock, which is exactly why this \
panel exists.";

/// Chop `text` into `count` roughly equal runs of characters.
///
/// Character counts rather than word boundaries, because a real token stream
/// does not respect words either.
fn fake_tokens(text: &str, count: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if count == 0 || chars.is_empty() {
        return Vec::new();
    }
    let count = count.min(chars.len());
    (0..count)
        .map(|index| {
            let start = index * chars.len() / count;
            let end = (index + 1) * chars.len() / count;
            chars[start..end].iter().collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crlf_translation_is_idempotent_over_mixed_line_endings() {
        assert_eq!(to_crlf("a\nb"), "a\r\nb");
        assert_eq!(to_crlf("a\r\nb"), "a\r\nb");
        assert_eq!(to_crlf("a\rb"), "a\r\nb");
        assert_eq!(to_crlf("a\n\nb"), "a\r\n\r\nb");
        assert_eq!(to_crlf(""), "");
        // A partial line, which is what a decoded token usually is.
        assert_eq!(to_crlf("resid"), "resid");
    }

    #[test]
    fn fake_tokens_covers_the_text_exactly_once() {
        let pieces = fake_tokens(LOREM, FAKE_DECODE_TOKENS);
        assert_eq!(pieces.len(), FAKE_DECODE_TOKENS);
        assert_eq!(pieces.concat(), LOREM);
        assert!(pieces.iter().all(|piece| !piece.is_empty()));
    }

    #[test]
    fn fake_tokens_handles_degenerate_requests() {
        assert!(fake_tokens(LOREM, 0).is_empty());
        assert!(fake_tokens("", 40).is_empty());
        assert_eq!(fake_tokens("ab", 40).len(), 2);
    }

    #[test]
    fn fake_hit_rate_stays_a_rate() {
        for step in [0, 1, 100, 3961, 100_000] {
            let rate = fake_hit_rate(step);
            assert!((0.0..=1.0).contains(&rate), "step {step} gave {rate}");
        }
    }
}
