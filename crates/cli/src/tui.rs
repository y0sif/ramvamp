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
//! # What moves, and how often
//!
//! As little as possible. The panel redraws on a *state* change at once — a
//! new phase, a new detail line, a new chunk — and otherwise at [`TICK`], once
//! a second, which is faster than any number on it can meaningfully change.
//! For [`SETTLE`] after a phase change nothing time-driven redraws at all, so
//! arriving at a phase is one event rather than a flicker. There is no
//! spinner: the prefill bar's half-step is the only thing that animates, and
//! at a realistic prompt size its head advances about every two seconds.
//!
//! # Wiring
//!
//! [`Harness`], [`panel`] and [`input`] are presentation only: nothing in them
//! loads a model, and [`selftest`] drives the whole surface with fabricated
//! numbers so the panel can be reviewed in seconds rather than in six minutes.
//!
//! [`session`] is the other half — it owns the model on a worker thread and
//! feeds this one by message. That split is forced rather than stylistic; see
//! its module docs.
//!
//! [`agent`] is the machine-facing side of the same panel: the semantic tree
//! taria publishes, and the decision of what an agent input means. What it
//! returns is a [`Plan`] — a key press or a paste — which
//! [`Harness::apply_agent_input`] runs through the handlers the keyboard
//! already goes through, so an agent can reach no state a keyboard cannot.

mod agent;
mod glyphs;
mod input;
mod panel;
mod session;
mod style;

pub use session::run_chat_tui;

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
use taria_ratatui::InputStatus;
use taria_ratatui::taria::AgentInput;

// The self-test's fabricated numbers, and only those: it runs before argument
// parsing and has no `--context` to honour, so the window it draws against is
// the default one. Everything the real harness shows comes from `session.rs`,
// which is given the resolved cap.
use crate::repl::DEFAULT_CONTEXT;
use agent::Plan;
use glyphs::Glyphs;
use input::LineEditor;
use panel::Layout;
use style::Palette;

/// What the runtime is doing right now.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Phase {
    #[default]
    Idle,
    Prefill,
    Decode,
}

/// How far through the prompt a prefill is.
///
/// The total is an `Option` because the panel has to be able to say something
/// honest without one: a prefill whose length is not known draws no bar, no
/// percent and no ETA, rather than an indeterminate sweep that implies
/// progress nobody measured.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Prefilling {
    /// Prompt positions committed so far.
    pub done: usize,
    /// Positions this prefill will consume, when that is known.
    pub total: Option<usize>,
}

/// Everything the panel shows, as of one instant.
///
/// The caller owns this: the harness never derives a number for itself, so
/// what the panel says and what the run reports cannot drift. The two derived
/// numbers on it — [`Status::rate`] and [`Status::eta`] — are derived by the
/// UI thread from events it already received, and they are fields here rather
/// than arithmetic in `panel.rs` so that a snapshot test is a pure function of
/// its input and never of a clock.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Status {
    pub phase: Phase,
    /// What the model is called, for the idle row.
    pub model: Option<String>,
    /// Where the prefill is, while one is running.
    pub prefill: Option<Prefilling>,
    /// `(chunk, chunks)` of the chunked prefill sweep.
    pub chunk: Option<(usize, usize)>,
    /// Tokens decoded so far this turn.
    pub tokens: usize,
    /// Time the current phase has been running.
    pub elapsed: Duration,
    /// `(used, cap)` context positions.
    pub context: (usize, usize),
    /// Expert cache hit rate, `0.0..=1.0`.
    pub hit_rate: Option<f32>,
    /// Bytes this turn has streamed from NVMe, where the runtime reports
    /// them.
    pub read_bytes: Option<u64>,
    /// Positions or tokens per second, over a rolling window.
    pub rate: Option<f64>,
    /// What is left of the prefill at that rate.
    pub eta: Option<Duration>,
    /// One short line that replaces whatever the detail row would have said.
    pub detail: Option<String>,
}

/// What a status is *about*, as opposed to what its numbers happen to be:
/// the phase, the line it is saying, the chunk it is on, and the model it is
/// talking to. See [`Status::state`].
type StatusKind<'a> = (
    Phase,
    Option<&'a str>,
    Option<(usize, usize)>,
    Option<&'a str>,
);

impl Status {
    /// The parts of a status whose change is a *state* change rather than a
    /// tick: what is running, what it is doing, and what it is called.
    ///
    /// A change here redraws immediately; a change in the numbers waits for
    /// the next [`TICK`]. That is the whole of the panel's motion policy.
    fn state(&self) -> StatusKind<'_> {
        (
            self.phase,
            self.detail.as_deref(),
            self.chunk,
            self.model.as_deref(),
        )
    }
}

/// Something the user did.
#[derive(Clone, Debug)]
pub enum UiEvent {
    /// Enter on a non-empty line. May contain newlines, from a paste.
    Submit(String),
    /// Ctrl-C or Esc while a turn is running.
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
    palette: &'static Palette,
    glyphs: &'static Glyphs,
    /// Set by [`Harness::set_status`], cleared by a draw.
    dirty: bool,
    /// Throttles status-driven redraws; input, resize and state changes
    /// always redraw.
    last_draw: Instant,
    /// When the phase last changed, which is what [`SETTLE`] is measured
    /// from.
    phase_since: Instant,
    /// Set by [`Harness::suspend`]: the terminal is the user's again and
    /// this harness must not touch it.
    suspended: bool,
}

/// Floor on status-driven redraws.
///
/// A second, because nothing on the panel says anything new faster than that:
/// the clock counts whole seconds, and at 3,961 positions across 80 columns
/// the bar's half-step lands about every two. Anything faster is motion for
/// its own sake in a window the user is going to be looking at for minutes.
const TICK: Duration = Duration::from_millis(1_000);

/// How long a phase change buys before anything *time-driven* redraws again.
///
/// Arriving at a phase should read as one event, and for this long after one
/// nothing moves on its own: not the clock, not the bar's half-step. What
/// still redraws inside the window is a state change (a detail line arriving,
/// a chunk landing) and a keystroke — those are information, not animation,
/// and holding them back would only make the panel feel slow.
const SETTLE: Duration = Duration::from_millis(400);

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

        let now = Instant::now();
        let mut harness = Self {
            term: build_terminal(layout)?,
            layout,
            editor: LineEditor::default(),
            status: Status::default(),
            palette: Palette::detect(),
            glyphs: Glyphs::detect(),
            dirty: true,
            last_draw: now - TICK,
            phase_since: now - SETTLE,
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

    /// The colour roles and the characters this terminal resolved to, for
    /// whatever is writing straight into the scroll region.
    pub fn marks(&self) -> (&'static Palette, &'static Glyphs) {
        (self.palette, self.glyphs)
    }

    /// The line the user is part way through typing.
    ///
    /// Read-only, and it exists for one caller: [`agent::build_nodes`] has to
    /// publish the draft as the value of the input node, and the editor is
    /// the harness's rather than the session's. Nothing may write through
    /// this — what the user typed is the keyboard's to change.
    pub(crate) fn editor(&self) -> &LineEditor {
        &self.editor
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
    /// token or per prefilled position. The draw it triggers is throttled —
    /// see [`TICK`] and [`SETTLE`], and [`Status::state`] for what jumps the
    /// queue — and its errors are swallowed; a caller that wants to see a
    /// failure should look at [`Harness::poll`], which it has to call anyway
    /// to notice Ctrl-C.
    pub fn set_status(&mut self, status: Status) {
        let changed = status.state() != self.status.state();
        if status.phase != self.status.phase {
            self.phase_since = Instant::now();
        }
        self.status = status;
        self.dirty = true;
        let _ = if changed {
            self.draw()
        } else {
            self.maybe_draw()
        };
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
        // A keystroke is not animation: what the user typed echoes now,
        // whatever the tick and the settle window have to say about it.
        self.draw()?;
        Ok(out)
    }

    /// Apply one agent input, and say what it earned.
    ///
    /// Two steps, and the split is the point: [`agent::plan`] decides — purely,
    /// out of the phase and the draft, and unit-tested against the very tree
    /// that advertised the action — and this executes the [`Plan`] it returned
    /// through the handlers a keyboard already uses. Nothing here decides
    /// anything, so there is no second copy of the routing table to drift.
    ///
    /// The [`UiEvent`]s come back in a `Vec` because one input can be several
    /// keystrokes: `type_text("a\nb\n")` is two turns, and dropping either
    /// would lose one.
    pub(crate) fn apply_agent_input(&mut self, input: &AgentInput) -> (InputStatus, Vec<UiEvent>) {
        let (plan, status) = agent::plan(input, self.status.phase, self.editor.text());
        let mut events = Vec::new();
        self.run_plan(plan, &mut events);
        (status, events)
    }

    /// Run a decided [`Plan`], collecting whatever the key handler produced.
    fn run_plan(&mut self, plan: Plan, events: &mut Vec<UiEvent>) {
        match plan {
            Plan::Key(code, modifiers) => events.extend(self.feed_key(code, modifiers)),
            Plan::Paste(text) => self.feed_paste(&text),
            Plan::PasteThenSubmit(text) => {
                self.feed_paste(&text);
                events.extend(self.feed_key(KeyCode::Enter, KeyModifiers::NONE));
            }
            Plan::Submit => events.extend(self.feed_key(KeyCode::Enter, KeyModifiers::NONE)),
            Plan::Nothing => {}
            // One level deep in practice, and this is the only place that
            // matters: a plan is a list of keystrokes, not a tree of them.
            Plan::Sequence(steps) => {
                for step in steps {
                    self.run_plan(step, events);
                }
            }
        }
    }

    /// Feed one key press through the app's own key handler.
    ///
    /// Private, and deliberately: an agent input reaches it only through
    /// [`Harness::apply_agent_input`], so there is nowhere in the session that
    /// can synthesize a keystroke without a decided plan and the ack that goes
    /// with it.
    ///
    /// The draw mirrors [`Harness::poll`] — what the input did echoes now
    /// rather than at the next tick — and its error is swallowed for the
    /// reason [`Harness::set_status`] swallows one: the caller polls the
    /// terminal every pass anyway, and that is where a failure surfaces.
    fn feed_key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> Option<UiEvent> {
        let out = self.handle_key(code, modifiers);
        let _ = self.draw();
        out
    }

    /// Insert text through the bracketed-paste path, exactly as
    /// [`Event::Paste`] does.
    fn feed_paste(&mut self, text: &str) {
        self.editor.insert_paste(text);
        let _ = self.draw();
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
        let running = self.status.phase != Phase::Idle;
        match code {
            // No signal handler: in raw mode Ctrl-C is just a keystroke.
            // It stops the turn that is running, and leaves when there is
            // nothing to stop.
            KeyCode::Char('c') if ctrl => Some(if running {
                UiEvent::Interrupt
            } else {
                UiEvent::Exit
            }),
            // Esc is the key the detail row advertises while a turn runs,
            // because it is the one that cannot be mistaken for "leave".
            // Idle it does nothing: there is no turn to stop, and quitting on
            // a stray Esc from a half-parsed escape sequence would be a
            // spectacular way to lose a conversation.
            KeyCode::Esc => running.then_some(UiEvent::Interrupt),
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
        if self.dirty && self.last_draw.elapsed() >= TICK && self.phase_since.elapsed() >= SETTLE {
            self.draw()?;
        }
        Ok(())
    }

    fn draw(&mut self) -> io::Result<()> {
        if self.suspended || !self.layout.usable() {
            self.dirty = false;
            return Ok(());
        }
        let view = panel::panel_view(
            self.layout,
            &self.status,
            &self.editor,
            self.palette,
            self.glyphs,
        );
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
        let spans: Vec<Span<'_>> = row
            .spans
            .iter()
            .map(|span| Span::styled(span.text.as_str(), span.style))
            .collect();
        frame.render_widget(Line::from(spans), rect);
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
// the transcript
// ---------------------------------------------------------------------------

/// Everything written into the scroll region, and the two bytes of styling it
/// is allowed.
///
/// # What is styled, and what is not
///
/// The prefixes, and nothing else. Model prose goes in at full strength with
/// no colouring, no timestamps and no rules between turns: it is the thing the
/// user came for, and a transcript that decorates it is a transcript that
/// cannot be pasted into an issue. What does the work is the blank line either
/// side of a turn.
///
/// # Three rules, all learned the hard way
///
/// - **A span never crosses a newline.** This is the terminal's own
///   scrollback: a style still open at the end of a line is inherited by every
///   line after it, including everything the user scrolls back to. Multi-line
///   text is styled one line at a time.
/// - **A span is closed with [`Palette::muted_sgr_end`], never `ESC [ 0 m`.**
///   SGR 0 resets attributes the user's shell set outside the region and
///   expects to still be there when the process exits.
/// - **SGR is zero width.** Column accounting is unaffected, and so is a
///   selection: what the user copies is the text, prefixes included, with no
///   escape bytes of ours in the middle of a word.
struct Ribbon {
    palette: &'static Palette,
    /// The user's mark: the same character the input row is prompted with, so
    /// a turn in the scrollback and the row it was typed on match.
    turn_mark: &'static str,
    /// The model's mark. The one character the transcript needs that the
    /// panel's glyph set does not carry — it marks a turn rather than drawing
    /// a panel — so it is resolved here, against the same two sets.
    speech_mark: &'static str,
    /// Newlines at the tail of what has been written. Two is a blank line.
    newlines: usize,
    /// A model turn is open, and the next chunk continues its line.
    speaking: bool,
}

/// Where a [`Ribbon`] puts what it writes.
///
/// A trait with one implementation in the binary, so that the blank lines and
/// the escape bytes — the whole of what this module promises about the
/// transcript — can be asserted on without a terminal.
trait Ink {
    fn ink(&mut self, text: &str) -> io::Result<()>;
}

impl Ink for Harness {
    fn ink(&mut self, text: &str) -> io::Result<()> {
        self.write_transcript(text)
    }
}

/// `ESC [ 1 m` / `ESC [ 22 m`: bold on, bold off.
const SGR_BOLD: &str = "\x1b[1m";
const SGR_BOLD_OFF: &str = "\x1b[22m";

/// `ESC [ 31 m` / `ESC [ 39 m`: red foreground, and default foreground.
const SGR_RED: &str = "\x1b[31m";
const SGR_FG_DEFAULT: &str = "\x1b[39m";

impl Ribbon {
    fn new(palette: &'static Palette, glyphs: &'static Glyphs) -> Self {
        // Two, so the first thing written does not open with a blank line it
        // has nothing above.
        Self {
            palette,
            turn_mark: glyphs.prompt,
            speech_mark: if *glyphs == Glyphs::ASCII {
                "*"
            } else {
                "\u{2022}" // • BULLET: one char, one column, like the rest.
            },
            newlines: 2,
            speaking: false,
        }
    }

    /// `ESC[1m` `{muted}` `› ` `{end}` `ESC[22m`: bold *and* muted, so the
    /// marker is findable when scrolling and quieter than the words after it.
    fn user_mark(&self) -> String {
        format!(
            "{SGR_BOLD}{}{} {}{SGR_BOLD_OFF}",
            self.palette.muted_sgr(),
            self.turn_mark,
            self.palette.muted_sgr_end(),
        )
    }

    /// `{muted}` `• ` `{end}`, and then the reply at full strength.
    fn speech_mark(&self) -> String {
        format!(
            "{}{} {}",
            self.palette.muted_sgr(),
            self.speech_mark,
            self.palette.muted_sgr_end(),
        )
    }

    /// `ESC[31m` `• ` `ESC[39m`: the one place the transcript raises its
    /// voice.
    fn system_mark(&self) -> String {
        format!("{SGR_RED}{} {SGR_FG_DEFAULT}", self.speech_mark)
    }

    /// One line of secondary text, opened and closed within the line.
    fn muted_line(&self, line: &str) -> String {
        format!(
            "{}{line}{}",
            self.palette.muted_sgr(),
            self.palette.muted_sgr_end()
        )
    }

    /// The user's turn: a bold muted marker, then their words untouched.
    fn user(&mut self, out: &mut impl Ink, text: &str) -> io::Result<()> {
        self.open(out)?;
        out.ink(&self.user_mark())?;
        // Trailing newlines are the gap's business, not the text's.
        out.ink(text.trim_end_matches('\n'))?;
        out.ink("\n")?;
        self.newlines = 1;
        Ok(())
    }

    /// A chunk of the model's reply, exactly as it was produced.
    fn say(&mut self, out: &mut impl Ink, text: &str) -> io::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if !self.speaking {
            self.open(out)?;
            out.ink(&self.speech_mark())?;
            self.newlines = 0;
            self.speaking = true;
        }
        out.ink(text)?;
        self.newlines = text.chars().rev().take_while(|c| *c == '\n').count();
        Ok(())
    }

    /// A refusal, a save, an error: red marker, muted text, one line at a
    /// time.
    fn system(&mut self, out: &mut impl Ink, text: &str) -> io::Result<()> {
        self.open(out)?;
        for (index, line) in text.trim_end_matches('\n').split('\n').enumerate() {
            if index == 0 {
                out.ink(&self.system_mark())?;
            }
            if !line.is_empty() {
                out.ink(&self.muted_line(line))?;
            }
            out.ink("\n")?;
        }
        self.newlines = 1;
        Ok(())
    }

    /// Close a reply that stopped mid-word, so whatever comes next starts on
    /// a line of its own.
    fn hush(&mut self, out: &mut impl Ink) -> io::Result<()> {
        if self.speaking {
            self.speaking = false;
            if self.newlines == 0 {
                out.ink("\n")?;
                self.newlines = 1;
            }
        }
        Ok(())
    }

    /// Close whatever was open and leave exactly one blank line above the
    /// turn about to be written.
    fn open(&mut self, out: &mut impl Ink) -> io::Result<()> {
        self.hush(out)?;
        while self.newlines < 2 {
            out.ink("\n")?;
            self.newlines += 1;
        }
        Ok(())
    }
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
/// RAMVAMP_TUI_SELFTEST=1 RAMVAMP_ASCII=1 cargo run -p ramvamp
/// ```
///
/// Walks all four looks in one run — a fresh idle panel with nothing to
/// report, a prefill advancing across a 3,961-position prompt, a decode
/// streaming tokens, and the idle panel that a finished turn leaves behind —
/// and then waits for more. Enter runs another fake turn, Esc or Ctrl-C stops
/// one, Ctrl-D leaves. `RAMVAMP_ASCII=1` is honoured throughout, because the
/// glyph set is resolved by `Glyphs::detect` and never chosen here.
pub fn selftest() -> anyhow::Result<()> {
    let mut harness = Harness::enter()?;
    let (palette, glyphs) = harness.marks();
    let mut ribbon = Ribbon::new(palette, glyphs);
    ribbon.system(&mut harness, BANNER)?;

    // Look one: a panel with nothing behind it. No hit rate, no context
    // share, no focal metric — the state the user meets first.
    let mut outcome = if let Some(UiEvent::Exit) =
        hold(&mut harness, &fresh_status(), Duration::from_millis(1_800))?
    {
        Outcome::Exit
    } else {
        Outcome::Finished
    };
    if outcome != Outcome::Exit {
        outcome = fake_turn(&mut harness, &mut ribbon, "Why is prefill the slow part?")?;
    }
    while outcome != Outcome::Exit {
        harness.set_status(idle_status());
        match harness.poll(Duration::from_millis(100))? {
            Some(UiEvent::Submit(line)) => outcome = fake_turn(&mut harness, &mut ribbon, &line)?,
            Some(UiEvent::Exit) => break,
            _ => {}
        }
    }
    ribbon.system(&mut harness, "self-test done.")?;
    harness.suspend()?;
    Ok(())
}

const BANNER: &str = "\
ramvamp terminal harness — self-test. No model is loaded and every number \
below is fabricated.\n\
The transcript above the panel is written by the terminal itself, so it \
scrolls, wraps and selects natively.\n\
Type to edit the input line, paste multi-line text, Enter to run another \
fake turn, Esc to interrupt one, Ctrl-D to leave.";

/// Fabricated prompt length: a full 4K-ish prompt, which is the case the
/// progress bar exists for.
const FAKE_PROMPT_POSITIONS: usize = 3961;

/// Fabricated prefill chunk width, matching the runtime's default.
const FAKE_CHUNK: usize = 512;

/// Fabricated decode length.
const FAKE_DECODE_TOKENS: usize = 40;

/// Positions per second the fabricated prefill clock is scaled to, so the
/// status row shows the rate v0 actually manages.
const FAKE_PREFILL_RATE: f64 = 11.0;

/// Tokens per second the fabricated decode clock is scaled to.
const FAKE_DECODE_RATE: f64 = 2.1;

/// Bytes the fabricated decode claims to have streamed per token.
const FAKE_BYTES_PER_TOKEN: u64 = 78 * 1024 * 1024;

/// What the fabricated model is called.
const FAKE_MODEL: &str = "Qwen3-30B-A3B";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Outcome {
    Finished,
    Interrupted,
    Exit,
}

/// The panel with nothing behind it: no turn has run, so there is no hit rate
/// to report and no context to report a share of.
fn fresh_status() -> Status {
    Status {
        phase: Phase::Idle,
        model: Some(FAKE_MODEL.to_owned()),
        context: (0, DEFAULT_CONTEXT),
        ..Status::default()
    }
}

/// The panel a finished turn leaves behind.
fn idle_status() -> Status {
    Status {
        phase: Phase::Idle,
        model: Some(FAKE_MODEL.to_owned()),
        context: (FAKE_PROMPT_POSITIONS + FAKE_DECODE_TOKENS, DEFAULT_CONTEXT),
        hit_rate: Some(fake_hit_rate(FAKE_PROMPT_POSITIONS + FAKE_DECODE_TOKENS)),
        ..Status::default()
    }
}

/// Show one status for a while, keeping the panel and the keyboard live.
fn hold(harness: &mut Harness, status: &Status, span: Duration) -> io::Result<Option<UiEvent>> {
    harness.set_status(status.clone());
    let deadline = Instant::now() + span;
    while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
        if let Some(event @ (UiEvent::Exit | UiEvent::Interrupt)) = harness.poll(remaining)? {
            return Ok(Some(event));
        }
    }
    Ok(None)
}

fn fake_turn(harness: &mut Harness, ribbon: &mut Ribbon, prompt: &str) -> io::Result<Outcome> {
    ribbon.user(harness, prompt)?;

    // Prefill. The real thing is 3,961 positions at about 11 tok/s, i.e. six
    // minutes; this walks the same bar in a few seconds so the panel can
    // actually be reviewed. The *clock* is fabricated to match the real rate
    // rather than measured, so the row reads like a real run instead of
    // claiming four figures of tok/s.
    let chunks = FAKE_PROMPT_POSITIONS.div_ceil(FAKE_CHUNK);
    let mut done = 0;
    while done < FAKE_PROMPT_POSITIONS {
        done = (done + 53).min(FAKE_PROMPT_POSITIONS);
        let remaining = (FAKE_PROMPT_POSITIONS - done) as f64 / FAKE_PREFILL_RATE;
        harness.set_status(Status {
            phase: Phase::Prefill,
            model: Some(FAKE_MODEL.to_owned()),
            prefill: Some(Prefilling {
                done,
                total: Some(FAKE_PROMPT_POSITIONS),
            }),
            chunk: Some((done.div_ceil(FAKE_CHUNK).max(1), chunks)),
            elapsed: Duration::from_secs_f64(done as f64 / FAKE_PREFILL_RATE),
            context: (done, DEFAULT_CONTEXT),
            rate: Some(FAKE_PREFILL_RATE),
            eta: Some(Duration::from_secs_f64(remaining)),
            // No byte figure: the runtime reports none between prefill
            // chunks, and the self-test does not get to invent one.
            ..Status::default()
        });
        match harness.poll(Duration::from_millis(40))? {
            Some(UiEvent::Interrupt) => return interrupted(harness, ribbon),
            Some(UiEvent::Exit) => return Ok(Outcome::Exit),
            _ => {}
        }
    }

    // Decode, at roughly the real 2 tok/s, in fragments that land mid-word
    // on purpose: that is what the real token stream looks like, and it is
    // what `write_transcript` has to survive.
    harness.set_status(decode_status(0));
    for (index, piece) in fake_tokens(LOREM, FAKE_DECODE_TOKENS).iter().enumerate() {
        let deadline = Instant::now() + Duration::from_millis(500);
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            match harness.poll(remaining)? {
                Some(UiEvent::Interrupt) => return interrupted(harness, ribbon),
                Some(UiEvent::Exit) => return Ok(Outcome::Exit),
                _ => {}
            }
        }
        ribbon.say(harness, piece)?;
        harness.set_status(decode_status(index + 1));
    }
    ribbon.hush(harness)?;
    Ok(Outcome::Finished)
}

/// Decode's numbers, at the rate the real runtime manages rather than the one
/// the self-test is walking through.
fn decode_status(tokens: usize) -> Status {
    Status {
        phase: Phase::Decode,
        model: Some(FAKE_MODEL.to_owned()),
        tokens,
        elapsed: Duration::from_secs_f64(tokens as f64 / FAKE_DECODE_RATE),
        context: (FAKE_PROMPT_POSITIONS + tokens, DEFAULT_CONTEXT),
        hit_rate: Some(fake_hit_rate(FAKE_PROMPT_POSITIONS + tokens)),
        read_bytes: Some(tokens as u64 * FAKE_BYTES_PER_TOKEN),
        rate: (tokens > 0).then_some(FAKE_DECODE_RATE),
        ..Status::default()
    }
}

fn interrupted(harness: &mut Harness, ribbon: &mut Ribbon) -> io::Result<Outcome> {
    ribbon.system(harness, "[interrupted]")?;
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

    /// A transcript in a `String`, so what the ribbon writes can be read back
    /// byte for byte.
    #[derive(Default)]
    struct Sheet(String);

    impl Ink for Sheet {
        fn ink(&mut self, text: &str) -> io::Result<()> {
            self.0.push_str(text);
            Ok(())
        }
    }

    fn ribbon(glyphs: &'static Glyphs) -> Ribbon {
        // The palette every terminal resolves to today; `style.rs` pins that.
        Ribbon::new(Palette::detect(), glyphs)
    }

    /// The prefixes, to the byte. Everything the transcript is allowed to
    /// style is in these four strings.
    #[test]
    fn the_transcript_marks_are_exactly_the_documented_bytes() {
        let unicode = ribbon(&Glyphs::UNICODE);
        assert_eq!(unicode.user_mark(), "\x1b[1m\x1b[2m› \x1b[22m\x1b[22m");
        assert_eq!(unicode.speech_mark(), "\x1b[2m• \x1b[22m");
        assert_eq!(unicode.system_mark(), "\x1b[31m• \x1b[39m");
        assert_eq!(unicode.muted_line("saved"), "\x1b[2msaved\x1b[22m");
        // SGR 0 would reset attributes the user's shell set outside the
        // scroll region and expects back when the process exits.
        for mark in [
            unicode.user_mark(),
            unicode.speech_mark(),
            unicode.system_mark(),
            unicode.muted_line("x"),
        ] {
            assert!(!mark.contains("\x1b[0m"), "{mark:?}");
        }
        // Same shapes in the invariant subset.
        let ascii = ribbon(&Glyphs::ASCII);
        assert_eq!(ascii.user_mark(), "\x1b[1m\x1b[2m> \x1b[22m\x1b[22m");
        assert_eq!(ascii.speech_mark(), "\x1b[2m* \x1b[22m");
        assert_eq!(ascii.system_mark(), "\x1b[31m* \x1b[39m");
    }

    /// One blank line either side of every turn, however the turns arrive —
    /// which is what makes the transcript readable without a single rule,
    /// timestamp or colour in the prose.
    #[test]
    fn every_turn_gets_a_blank_line_either_side_of_it() {
        let mut sheet = Sheet::default();
        let mut ribbon = ribbon(&Glyphs::UNICODE);
        ribbon.system(&mut sheet, "ramvamp chat.").unwrap();
        ribbon.user(&mut sheet, "why is prefill slow?").unwrap();
        // A reply arrives in fragments that land mid-word.
        ribbon.say(&mut sheet, "Experts stream ").unwrap();
        ribbon.say(&mut sheet, "from NVMe.").unwrap();
        ribbon.hush(&mut sheet).unwrap();
        ribbon.user(&mut sheet, "thanks").unwrap();

        let plain = strip_sgr(&sheet.0);
        assert_eq!(
            plain,
            "• ramvamp chat.\n\n› why is prefill slow?\n\n• Experts stream from NVMe.\n\n› thanks\n"
        );
        // Never two blank lines, at any point.
        assert!(!plain.contains("\n\n\n"), "{plain:?}");
    }

    /// The blank line is a *gap*, not a newline count: a turn that already
    /// ends on a blank line does not get another, and one that stops
    /// mid-word gets its line closed first.
    #[test]
    fn the_gap_does_not_double_up_or_leave_a_turn_open() {
        let mut sheet = Sheet::default();
        let mut ribbon = ribbon(&Glyphs::UNICODE);
        // A reply whose last fragment is a newline.
        ribbon.say(&mut sheet, "one\n").unwrap();
        ribbon.system(&mut sheet, "[interrupted]").unwrap();
        // And a multi-line turn, which is what a paste produces.
        ribbon.user(&mut sheet, "first\nsecond\n").unwrap();
        assert_eq!(
            strip_sgr(&sheet.0),
            "• one\n\n• [interrupted]\n\n› first\nsecond\n"
        );
    }

    /// A style left open at the end of a line is inherited by every line
    /// after it, including everything already in the scrollback.
    #[test]
    fn no_transcript_span_ever_crosses_a_newline() {
        let mut sheet = Sheet::default();
        let mut ribbon = ribbon(&Glyphs::UNICODE);
        // Multi-line system text is the case that has to be split: `/help` is
        // eight lines long.
        ribbon
            .system(&mut sheet, "commands:\n  /help\n\n  /exit")
            .unwrap();
        ribbon.user(&mut sheet, "a\nb").unwrap();
        ribbon.say(&mut sheet, "one\ntwo\n").unwrap();

        let mut open: Option<&str> = None;
        for (index, piece) in sheet.0.split('\n').enumerate() {
            for mark in piece.match_indices('\x1b').map(|(at, _)| &piece[at..]) {
                let code = mark
                    .strip_prefix("\x1b[")
                    .and_then(|rest| rest.split('m').next())
                    .expect("an SGR sequence");
                open = match code {
                    "22" | "39" | "0" => None,
                    _ => Some("open"),
                };
            }
            assert!(
                open.is_none(),
                "line {index} of {:?} left a span open",
                sheet.0
            );
        }
    }

    /// Everything but the SGR sequences, which are zero width and must not
    /// affect what the user copies.
    fn strip_sgr(text: &str) -> String {
        let mut out = String::new();
        let mut rest = text;
        while let Some(at) = rest.find('\x1b') {
            out.push_str(&rest[..at]);
            let end = rest[at..].find('m').expect("an SGR sequence") + at;
            rest = &rest[end + 1..];
        }
        out.push_str(rest);
        out
    }

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

    /// The four looks the self-test walks are four *different* looks, and the
    /// fresh one claims nothing it has not measured.
    #[test]
    fn the_self_test_walks_four_distinct_states() {
        let states = [
            fresh_status(),
            Status {
                phase: Phase::Prefill,
                prefill: Some(Prefilling {
                    done: 1024,
                    total: Some(FAKE_PROMPT_POSITIONS),
                }),
                ..Status::default()
            },
            decode_status(FAKE_DECODE_TOKENS),
            idle_status(),
        ];
        for (index, state) in states.iter().enumerate() {
            for other in &states[index + 1..] {
                assert_ne!(state, other);
            }
        }
        assert_eq!(fresh_status().hit_rate, None);
        assert_eq!(fresh_status().context, (0, DEFAULT_CONTEXT));
        assert!(idle_status().hit_rate.is_some());
        // The prefill look is the one with a bar, so it is the one that has
        // to have a total to draw it against.
        assert_eq!(
            decode_status(1).prefill,
            None,
            "decode draws the rule, not a bar"
        );
    }

    /// A status change that is only a number waits for the tick; a change in
    /// what is *happening* does not.
    #[test]
    fn a_state_change_is_more_than_a_new_number() {
        let quiet = decode_status(1);
        let mut ticked = quiet.clone();
        ticked.tokens = 2;
        ticked.elapsed = Duration::from_secs(9);
        ticked.read_bytes = Some(1);
        assert_eq!(quiet.state(), ticked.state());

        for changed in [
            Status {
                phase: Phase::Idle,
                ..quiet.clone()
            },
            Status {
                detail: Some("stopping this reply...".to_owned()),
                ..quiet.clone()
            },
            Status {
                chunk: Some((2, 8)),
                ..quiet.clone()
            },
            Status {
                model: None,
                ..quiet.clone()
            },
        ] {
            assert_ne!(quiet.state(), changed.state());
        }
    }
}
