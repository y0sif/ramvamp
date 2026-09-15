//! `chat --tui`: the terminal harness driven by the real model.
//!
//! # Two threads, and why it has to be two
//!
//! A generate call holds `&mut ForwardState` from the first prefill chunk to
//! the last sampled token, and on a 4K prompt that is minutes. Nothing else
//! can read the state while it runs, so the only way to draw a live panel is
//! to put the model somewhere the drawing thread is not.
//!
//! `ForwardState` is `Send` and deliberately **not** `Sync` (`main.rs` asserts
//! both at compile time): it owns an `Arena` wrapping a `NonNull<u8>` that must
//! never escape its `&mut ExpertStream`. So the state *moves* onto a worker
//! thread and the UI never sees it — not by reference, not behind a lock. Every
//! number the panel shows arrives as a message.
//!
//! ```text
//! UI thread                             worker thread
//!   Harness::poll ──Command::Submit────▶ owns Model + ForwardState
//!                 ◀──Output::Token───── generate_from_with_progress
//!                 ◀──Output::Progress──
//!                 ◀──Output::TurnDone──
//!                 ──interrupt: true ──▶ (checked in on_token)
//! ```
//!
//! Two `std::sync::mpsc` channels and one `AtomicBool`. No async, no runtime.
//!
//! # What is reused
//!
//! All of the REPL's logic: [`parse_repl_input`], [`Transcript`],
//! [`TurnCodec`], [`plan_turn`], [`turn_seed`]. This module adds a front end
//! and a thread boundary, not a second implementation of the chat policy — and
//! in particular not a second copy of the context accounting, which is the
//! part that is easy to get subtly wrong.
//!
//! # Agent input
//!
//! The taria layer is a second source of the same events. An agent input is
//! decided by [`agent::plan`], executed by
//! [`Harness::apply_agent_input`](super::Harness::apply_agent_input) through
//! the key handler and the paste path the keyboard already uses, and whatever
//! [`UiEvent`] that produces is applied by [`steer`] — the same function the
//! keyboard's events go through. So there is one implementation of what Submit
//! and Interrupt mean, and an agent cannot reach a state a keyboard cannot.
//!
//! # Interrupt
//!
//! Unchanged from the line REPL: `on_token` sees the flag and leaves by
//! [`panic_any(ChatAbort)`](std::panic::panic_any), caught a few frames up.
//! That unwind point is between two forward passes with no expert read in
//! flight and a whole KV cache, which is what makes keeping the state across
//! an abort sound; it is the reason the mechanism is a deliberate unwind and
//! not a return value. The only difference here is where the flag comes from —
//! a keystroke in raw mode rather than SIGINT, so `chat --tui` installs no
//! signal handler at all.
//!
//! A consequence worth stating: `on_token` is the only place the flag is read,
//! so a stop during *prefill* is honoured at the first decoded token rather
//! than immediately. Prefill has no comparably safe unwind point and inventing
//! one is not in scope; the panel says so instead, on the detail row, until
//! the first chunk lands.

use std::cell::Cell;
use std::collections::VecDeque;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::{Context as _, bail};
use ramvamp_core::generate::{GenerateParams, GenerateProgress, generate_from_with_progress};
use ramvamp_core::io::StreamStats;
use ramvamp_core::model::{ForwardState, Model, StreamPhase};
use ramvamp_core::tokenizer::{ChatMessage, Role, RvmpTokenizer};
use taria_ratatui::{InputStatus, TariaLayer};

use super::agent::Liveness;
use super::{Harness, Phase, Prefilling, Ribbon, Status, UiEvent, agent};
use crate::config::Dials;
use crate::repl::{
    REPL_HELP, ReplInput, Transcript, TurnCodec, TurnPlan, parse_repl_input, plan_turn, turn_seed,
};
use crate::{ChatAbort, ChatArgs, human_bytes, hush_control_flow_panics, load_tokenizer};

/// How long the UI blocks on a keystroke before refreshing the panel.
///
/// Also the worst-case latency between the worker producing a token and the
/// transcript showing it. Decode runs at about 2 tok/s, so this is two orders
/// of magnitude inside "immediately".
const POLL: Duration = Duration::from_millis(40);

/// Stack for the worker thread.
///
/// The default for a spawned thread is 2 MiB, and the forward pass has only
/// ever run on the main thread's 8 MiB. Matching it means moving the model to
/// a worker cannot be what overflows a stack. It is address space, not
/// residency, so it does not come out of the 3 GB budget.
const WORKER_STACK: usize = 8 * 1024 * 1024;

const GREETING: &str = "\
ramvamp chat. Type a message and press Enter. /help lists the commands, \
Esc stops a reply, Ctrl-D leaves.";

// ---------------------------------------------------------------------------
// the messages
// ---------------------------------------------------------------------------

/// What the UI asks the worker to do.
///
/// Everything that needs the model, the tokenizer or the transcript is a
/// command: the worker owns all three, so the UI thread holds no part of the
/// conversation and cannot disagree with it.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    Submit(String),
    Reset,
    Save(PathBuf),
    /// Finish what you are doing and stop. Sent once, on the way out.
    Shutdown,
}

/// What the worker tells the UI.
#[derive(Debug)]
enum Output {
    /// The model is up: the load banner, and what the model is called.
    Ready { banner: String, model: String },
    /// A chunk of reply text, exactly as `on_token` produced it.
    Token(String),
    /// Where the run is, as of one progress event.
    Progress(Snapshot),
    /// The turn is over, one way or another.
    TurnDone { context: usize },
    /// One line for the transcript: a refusal, a save, a warning.
    Notice(String),
    /// The worker is giving up and this is why.
    Failed(String),
}

/// The numbers the panel shows, as the worker measured them.
///
/// Deliberately without a clock: the worker cannot send one while it is inside
/// a generate call, so a duration here would be however stale the last chunk
/// boundary left it. The UI times the phase instead (see [`Screen`]), which is
/// the one number it can honestly derive — every count in here is the worker's.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Snapshot {
    phase: Phase,
    prefill: Option<Prefilling>,
    /// `(chunk, chunks)` of the prefill sweep, once its width is known.
    chunk: Option<(usize, usize)>,
    tokens: usize,
    context: usize,
    hit_rate: Option<f32>,
    /// Bytes this turn has streamed, where the runtime reports them.
    read_bytes: Option<u64>,
}

impl Snapshot {
    /// Nothing is running and the conversation stands at `context`.
    fn idle(context: usize) -> Self {
        Self {
            phase: Phase::Idle,
            prefill: None,
            chunk: None,
            tokens: 0,
            context,
            hit_rate: None,
            read_bytes: None,
        }
    }

    /// What the rolling rate is measured over in this phase, if anything.
    fn progress(&self) -> Option<usize> {
        match self.phase {
            Phase::Idle => None,
            Phase::Prefill => self.prefill.map(|prefill| prefill.done),
            Phase::Decode => Some(self.tokens),
        }
    }
}

/// A generate call's progress event as the panel wants it.
///
/// `fed` is what the KV cache held before this turn and `prompt` is how many
/// ids this turn feeds, so the context figure is the conversation's position
/// rather than this call's offset into it. `width` is the prefill chunk the
/// first event revealed, and `read_base` is where the decode byte counter
/// stood when the turn started, so what the panel shows is this turn's
/// streaming rather than the session's.
fn snapshot_of(
    event: GenerateProgress,
    fed: usize,
    prompt: usize,
    width: usize,
    read_base: u64,
) -> Snapshot {
    match event {
        GenerateProgress::PrefillChunk {
            positions_done,
            positions_total,
        } => Snapshot {
            phase: Phase::Prefill,
            prefill: Some(Prefilling {
                done: positions_done,
                total: Some(positions_total),
            }),
            chunk: chunk_of(positions_done, positions_total, width),
            tokens: 0,
            context: fed + positions_done,
            // The chunked sweep bypasses the expert cache, so there is no hit
            // rate to report until decode starts asking it for anything. Nor
            // is there a byte figure: `PrefillChunk` carries positions and
            // nothing else, and the `ForwardState` that holds the counters is
            // borrowed by the generate call for its whole duration.
            hit_rate: None,
            read_bytes: None,
        },
        GenerateProgress::DecodeToken { index, stats } => Snapshot {
            phase: Phase::Decode,
            prefill: None,
            chunk: None,
            tokens: index + 1,
            context: fed + prompt + index,
            hit_rate: hit_rate(&stats),
            read_bytes: Some((stats.bytes_read + stats.sweep_bytes_read).saturating_sub(read_base)),
        },
    }
}

/// Which chunk of the sweep `done` positions is, and how many chunks the
/// prompt is, given the `width` the first event revealed.
///
/// `None` unless there is something to count: a width of one is
/// [`PrefillMode::TokenMajor`](ramvamp_core::model::PrefillMode) reporting per
/// token rather than per chunk, and a prompt that is one chunk long has a
/// progress bar for exactly this reason.
fn chunk_of(done: usize, total: usize, width: usize) -> Option<(usize, usize)> {
    if width < 2 {
        return None;
    }
    let chunks = total.div_ceil(width);
    (chunks >= 2).then(|| (done.div_ceil(width).clamp(1, chunks), chunks))
}

/// Share of routed experts the cache had ready, `hits` against `hits +
/// misses`.
///
/// Pending hits are left out of both halves on purpose: they issue no read and
/// avoid none either, so counting them as served would flatter the rate and
/// counting them as misses would blame the cache for a read it did not make.
/// `None` before anything has been routed, which is not the same as zero.
fn hit_rate(stats: &StreamStats) -> Option<f32> {
    let resolved = stats.hits + stats.misses;
    (resolved > 0).then(|| stats.hits as f32 / resolved as f32)
}

// ---------------------------------------------------------------------------
// the UI thread
// ---------------------------------------------------------------------------

/// Run `chat --tui`.
///
/// Fails before the model is touched when stdout is not a terminal, which is
/// what keeps the harness out of `scripts/cold_bench.py` and the sweep
/// scripts: they parse the timing lines the line REPL writes to stderr, and
/// this path writes none of them anywhere.
pub fn run_chat_tui(args: ChatArgs, dials: Dials) -> anyhow::Result<()> {
    // Before the harness, and therefore before raw mode and before the DSR
    // cursor query: binding a socket is the kind of thing that can block on a
    // filesystem, and doing it while the terminal is half taken over is how a
    // session ends up wedged. `bind_or_disabled` cannot fail by design — a
    // layer that could not bind is inert and answers every method — so taria
    // can never be the reason `chat --tui` does not start. It prints nothing
    // itself either; reporting is ours to place, after `suspend`.
    let mut layer = TariaLayer::bind_or_disabled("ramvamp");
    let mut harness = Harness::enter()
        .context("chat --tui needs an interactive terminal; drop --tui for the line REPL")?;
    // *After* `Harness::enter`, so the hush hook wraps the terminal-restoring
    // one rather than the other way round. A ChatAbort then stops at the hush
    // hook and the panel is left alone; a real panic passes through it, the
    // terminal is restored, and only then is the backtrace printed.
    hush_control_flow_panics();

    // Read before `dials` is moved into the worker. This is the *configured*
    // window, not the one the manifest has vetted: the trained-context refusal
    // belongs to the worker, which owns the model directory, and it arrives
    // here as an `Output::Failed` that ends the session anyway.
    let context_cap = dials.context.value;

    let (commands, from_ui) = mpsc::channel::<Command>();
    let (to_ui, outputs) = mpsc::channel::<Output>();
    let interrupt = Arc::new(AtomicBool::new(false));
    let worker = {
        let interrupt = Arc::clone(&interrupt);
        std::thread::Builder::new()
            .name("ramvamp-chat".to_owned())
            .stack_size(WORKER_STACK)
            .spawn(move || worker_main(args, &dials, &from_ui, &to_ui, &interrupt))
            .context("spawning the chat worker")?
    };

    let ended = drive(
        &mut harness,
        &mut layer,
        &commands,
        &outputs,
        &interrupt,
        context_cap,
    );

    // The terminal comes back before anything else is said, whichever way the
    // session ended — including a `drive` that failed mid-draw.
    let _ = harness.suspend();
    // And only now is printing safe: the scroll region is reset, raw mode is
    // off and stderr is the user's again. Nothing in `drive` may report any of
    // this, because in raw mode a bare newline does not return to column 0.
    report_agent_layer(&layer);
    match ended {
        Ok(Ended { failure, join }) => {
            if join {
                if worker.join().is_err() {
                    // The panic itself has already been reported by the hook.
                    bail!("the chat worker thread panicked");
                }
            } else {
                eprintln!("leaving with the model still working; the thread goes with the process");
            }
            match failure {
                Some(reason) => Err(anyhow::anyhow!(reason)),
                None => Ok(()),
            }
        }
        Err(error) => Err(error),
    }
}

/// Say what the agent layer could not do, once the terminal is the user's
/// again.
///
/// Every one of these is traffic that went nowhere, and the layer keeps the
/// counts precisely because it must not print them itself. Silence is the
/// normal case: a layer that bound and served without losing anything has
/// nothing to add to the end of a chat session.
fn report_agent_layer(layer: &TariaLayer) {
    // Not on stderr. Nobody asked for the agent socket, and the ordinary way
    // to fail to bind it is to already be running `chat --tui` in another
    // window — comparing two models side by side is a thing people do, and the
    // second one greeting them with an error about a feature they never asked
    // for is worse than silence. `RUST_LOG=ramvamp=debug` has it for whoever
    // was actually looking for the socket. The counters below stay on stderr:
    // they only fire when they are non-zero, which means an agent really was
    // connected and really did lose something.
    if let Some(error) = layer.bind_error() {
        tracing::debug!(%error, "the agent socket did not come up; running without it");
    }
    let dropped = layer.dropped_inputs();
    if dropped > 0 {
        eprintln!("dropped {dropped} agent input(s): the chat loop could not keep up");
    }
    let stale = layer.stale_inputs();
    if stale > 0 {
        eprintln!(
            "discarded {stale} agent input(s): the bridge connection they arrived on ended first"
        );
    }
    let unknown = layer.unknown_inputs();
    if unknown > 0 {
        eprintln!(
            "could not read {unknown} agent input(s): the bridge speaks a newer taria than this \
             build; raise the taria-ratatui dependency"
        );
    }
    let acks = layer.dropped_acks();
    if acks > 0 {
        eprintln!(
            "lost the answer to {acks} agent input(s): the bridge read them slower than the \
             session answered, so those agent calls timed out instead"
        );
    }
    let cut = layer.truncated_snapshots();
    if cut > 0 {
        // Names the depth measured and a node found at it, which is the only
        // way to find a branch that ran away in a tree generated from data.
        match layer.last_truncation() {
            Some(branch) => {
                eprintln!("published {cut} agent snapshot(s) with a branch cut: {branch}")
            }
            None => eprintln!("published {cut} agent snapshot(s) with a branch cut"),
        }
    }
}

/// How the session ended, from the UI's side.
struct Ended {
    /// The worker's fatal error, if it had one.
    failure: Option<String>,
    /// Whether the worker can be joined without making the user wait. False
    /// when the user asked twice to leave and the worker had not stopped.
    join: bool,
}

/// Progress samples, oldest first, for a rate that describes the last stretch
/// of a run rather than the whole of it.
///
/// A cumulative rate — everything done over everything elapsed — is wrong in
/// the one place a rate is worth showing. The first seconds of a prefill
/// include the model's first touch of every expert file, so the cumulative
/// figure starts low, climbs for ten or twenty seconds, and drags an ETA
/// computed from it minutes off. A window that forgets the beginning settles
/// within a chunk or two and then stays put.
#[derive(Debug, Default)]
struct Meter {
    samples: VecDeque<(Instant, usize)>,
}

/// Samples the rolling rate is measured over. Prefill reports once per chunk
/// and decode once per token, so this is a minute or two of either.
const WINDOW: usize = 64;

impl Meter {
    fn clear(&mut self) {
        self.samples.clear();
    }

    fn observe(&mut self, at: Instant, done: usize) {
        self.samples.push_back((at, done));
        while self.samples.len() > WINDOW {
            self.samples.pop_front();
        }
    }

    /// Positions or tokens per second across the window, or `None` while
    /// there is not yet enough of one to divide by.
    fn rate(&self) -> Option<f64> {
        let (first_at, first_done) = *self.samples.front()?;
        let (last_at, last_done) = *self.samples.back()?;
        let seconds = last_at.saturating_duration_since(first_at).as_secs_f64();
        let done = last_done.checked_sub(first_done)?;
        (seconds > 0.0 && done > 0).then(|| done as f64 / seconds)
    }

    /// How long `remaining` more would take at that rate.
    fn eta(&self, remaining: usize) -> Option<Duration> {
        let seconds = remaining as f64 / self.rate()?;
        Duration::try_from_secs_f64(seconds).ok()
    }
}

/// Everything the UI thread knows, which is only what the worker told it plus
/// a clock.
struct Screen {
    snapshot: Snapshot,
    /// The rolling tail of the conversation, for the agent tree.
    ///
    /// The human transcript lives in the terminal's scrollback and nowhere
    /// else (see `tui.rs`), so this is the only copy anything in this process
    /// can read back. It is a tap, never a replacement: every push here sits
    /// beside the `ribbon` call that did the writing.
    transcript: agent::TranscriptRing,
    /// When the current phase started, by the UI's own monotonic clock.
    ///
    /// The worker cannot send a duration from inside a generate call, so this
    /// is the honest way to have a live rate: the counts are the worker's, the
    /// elapsed time is measured here, and a phase change restarts it.
    phase_started: Instant,
    /// The rolling rate behind the panel's tok/s and its ETA.
    meter: Meter,
    /// What the model is called, once the worker has loaded one.
    model: Option<String>,
    /// The context window this run resolved `--context` to, which is the
    /// denominator of the panel's `ctx used/cap`.
    ///
    /// Resolved by the UI thread from the same [`ContextArgs`] the worker
    /// reads, before `args` is moved across the channel — not sent back by the
    /// worker, because the panel has to be able to draw a denominator during
    /// the tens of seconds the model is still loading.
    ///
    /// [`ContextArgs`]: crate::ContextArgs
    context_cap: usize,
    detail: Option<String>,
    /// A draw is worth doing.
    dirty: bool,
}

impl Screen {
    fn loading(context_cap: usize) -> Self {
        Self {
            snapshot: Snapshot::idle(0),
            transcript: agent::TranscriptRing::default(),
            phase_started: Instant::now(),
            meter: Meter::default(),
            model: None,
            context_cap,
            detail: Some("loading the model...".to_owned()),
            dirty: true,
        }
    }

    fn observe(&mut self, snapshot: Snapshot) {
        self.observe_at(snapshot, Instant::now());
    }

    /// [`Screen::observe`] with the clock passed in, so the restart rule is
    /// testable without racing a monotonic clock's resolution.
    fn observe_at(&mut self, snapshot: Snapshot, now: Instant) {
        if snapshot.phase != self.snapshot.phase {
            self.phase_started = now;
            // Prefill's rate is not decode's, and neither is measured across
            // the boundary between them.
            self.meter.clear();
        }
        if let Some(done) = snapshot.progress() {
            self.meter.observe(now, done);
        }
        self.snapshot = snapshot;
        self.dirty = true;
    }

    fn say(&mut self, detail: impl Into<String>) {
        self.detail = Some(detail.into());
        self.dirty = true;
    }

    /// Drop the transient line, so the detail row goes back to reporting what
    /// the phase is actually doing.
    fn clear_detail(&mut self) {
        if self.detail.take().is_some() {
            self.dirty = true;
        }
    }

    fn status(&self) -> Status {
        let remaining = self
            .snapshot
            .prefill
            .and_then(|prefill| Some(prefill.total?.saturating_sub(prefill.done)));
        Status {
            phase: self.snapshot.phase,
            model: self.model.clone(),
            prefill: self.snapshot.prefill,
            chunk: self.snapshot.chunk,
            tokens: self.snapshot.tokens,
            elapsed: self.phase_started.elapsed(),
            context: (self.snapshot.context, self.context_cap),
            hit_rate: self.snapshot.hit_rate,
            read_bytes: self.snapshot.read_bytes,
            rate: self.meter.rate(),
            eta: remaining.and_then(|remaining| self.meter.eta(remaining)),
            detail: self.detail.clone(),
        }
    }
}

/// What a submitted line means. Separate from the loop so every command the
/// line REPL understands can be checked without a terminal.
#[derive(Debug, PartialEq, Eq)]
enum Routed {
    /// Whitespace: reprompt, disturb nothing.
    Nothing,
    /// Answered here, in the transcript: `/help`, a mistyped command.
    Local(String),
    /// `/exit` and friends.
    Leave,
    /// Anything the worker owns.
    Send(Command),
}

fn route(line: &str) -> Routed {
    match parse_repl_input(line) {
        ReplInput::Blank => Routed::Nothing,
        ReplInput::Exit => Routed::Leave,
        ReplInput::Help => Routed::Local(REPL_HELP.to_owned()),
        ReplInput::BadCommand(reason) => Routed::Local(reason),
        ReplInput::Reset => Routed::Send(Command::Reset),
        ReplInput::Save(path) => Routed::Send(Command::Save(path)),
        ReplInput::Message(text) => Routed::Send(Command::Submit(text)),
    }
}

/// The UI loop: drain the worker, refresh the panel, read a keystroke.
fn drive(
    harness: &mut Harness,
    layer: &mut TariaLayer,
    commands: &Sender<Command>,
    outputs: &Receiver<Output>,
    interrupt: &AtomicBool,
    context_cap: usize,
) -> anyhow::Result<Ended> {
    let mut screen = Screen::loading(context_cap);
    let (palette, glyphs) = harness.marks();
    let mut ribbon = Ribbon::new(palette, glyphs);
    let mut failure: Option<String> = None;
    ribbon.system(harness, GREETING)?;
    screen.transcript.push_system(GREETING);

    'session: loop {
        // Everything the worker has said since the last pass. Draining rather
        // than taking one keeps a burst of tokens from being paced by POLL.
        loop {
            match outputs.try_recv() {
                Ok(output) => {
                    if absorb(
                        harness,
                        &mut ribbon,
                        &mut screen,
                        &mut failure,
                        interrupt,
                        output,
                    )? {
                        break 'session;
                    }
                }
                Err(TryRecvError::Empty) => break,
                // The worker is gone and said nothing about it; nothing left
                // to drive.
                Err(TryRecvError::Disconnected) => break 'session,
            }
        }

        // Idle is the only state whose row does not change on its own, so it
        // is the only one that does not get a redraw per pass. The tree is
        // built only when there is a layer to publish it to: `publish` returns
        // immediately on an inert layer, but the tree is ~8 nodes and a copy of
        // the transcript tail, and a user with no agent should not pay for one
        // 25 times a second.
        //
        // The two of them describe one instant, so the `Status` behind them is
        // built once — and on a pass that wants neither, which idle with no
        // bridge attached is every pass, not at all. It used to be built and
        // then cloned whatever the pass was doing, which is four `String`
        // clones a pass for two readers that were often both absent.
        let redraw = screen.dirty || screen.snapshot.phase != Phase::Idle;
        if redraw || layer.is_enabled() {
            let status = screen.status();
            // The agent's view goes out first only so that the panel's can take
            // the `Status` by value rather than a copy of it; nothing observes
            // the order, and both are the same instant either way. Unthrottled,
            // because the layer skips a tree identical to the last one it
            // published — which is what the rounding in `agent.rs` is for — so
            // a pass that changed nothing costs a comparison.
            if layer.is_enabled() {
                layer.publish(agent::build_nodes(
                    &status,
                    harness.editor(),
                    &screen.transcript,
                    Liveness::Live,
                ));
            }
            if redraw {
                harness.set_status(status);
                screen.dirty = false;
            }
        }

        // Agent input is drained either side of the blocking poll: once before
        // it, so what arrived during the last pass is applied without waiting
        // out another POLL, and once after, so an input that landed while the
        // loop was parked is applied in the same pass as the keystroke that
        // woke it. Both drains apply their input where they stand, so the
        // harness sees them in the order the drains ran — which is not the
        // order they arrived in: an agent input that landed during the poll is
        // applied after the keystroke that woke it, even if it got there
        // first. Only the events they produced are held back, to keep the one
        // set of arms below.
        let mut events: Vec<UiEvent> = Vec::new();
        pump_agent(harness, layer, &mut events);
        events.extend(harness.poll(POLL)?);
        pump_agent(harness, layer, &mut events);
        for event in events {
            if steer(
                harness,
                &mut ribbon,
                &mut screen,
                commands,
                interrupt,
                event,
            )? {
                break 'session;
            }
        }
    }

    // Leaving. The worker is asked to stop twice over — the flag for a reply
    // in progress, the command for the loop around it — and then watched until
    // it drops its end of the channel.
    interrupt.store(true, Ordering::SeqCst);
    let _ = commands.send(Command::Shutdown);
    screen.say("stopping; Ctrl-C again to leave without waiting");
    screen.dirty = true;
    let mut join = true;
    // Before the first pass rather than after the first quiet one: from here on
    // every input is acked `Ignored` where it stands, and a `Live` tree left
    // standing over that window goes on advertising four actions the drain now
    // refuses.
    publish_wind_down(harness, layer, &screen, &screen.status());
    loop {
        // Acked, deliberately not applied. An input that arrives now cannot be
        // honoured — the tree has stopped advertising anything and the session
        // is already leaving — but it must still be answered, or it sits there
        // until the agent's own timeout, which is as long as the in-flight
        // prefill this loop is waiting out. Applying one would be worse than
        // useless: an agent could press Ctrl-C into the poll below, set
        // `join = false` and abandon the worker thread the user is waiting on.
        // Drained on every pass rather than only when the worker goes quiet,
        // because a reply still streaming out is exactly when an agent is
        // likely to be acting.
        layer.drain_acking(|_| InputStatus::Ignored);
        match outputs.try_recv() {
            Ok(output) => {
                // A fatal error raised on the way out still gets reported, but
                // it cannot cut the wait short: the point of this loop is to
                // see the channel close.
                absorb(
                    harness,
                    &mut ribbon,
                    &mut screen,
                    &mut failure,
                    interrupt,
                    output,
                )?;
                screen.say("stopping; Ctrl-C again to leave without waiting");
            }
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {
                let status = screen.status();
                // Winding down is a state an agent can be watching, so the
                // tree must not go stale while it happens.
                publish_wind_down(harness, layer, &screen, &status);
                harness.set_status(status);
                // A prefill in flight cannot be cut short (see the module
                // docs), so a user who does not want to wait it out says so
                // again and the thread is left to the process exit.
                if matches!(
                    harness.poll(POLL)?,
                    Some(UiEvent::Exit | UiEvent::Interrupt)
                ) {
                    join = false;
                    break;
                }
            }
        }
    }
    Ok(Ended { failure, join })
}

/// Publish the wind-down tree: every value the live one carries, and nothing to
/// act on.
///
/// Both places the wind-down loop can leave a tree behind go through here. It
/// must not go stale, because an agent watching a session end still has to be
/// able to read it — and it must not lie either: nothing in that loop is
/// applied, so `WindingDown` strips every action set and the tree advertises
/// nothing, which is the same answer the drain gives. A `Live` tree left
/// standing while the worker streams its last tokens, or while a channel that
/// was already `Disconnected` breaks the loop on its first pass, would be
/// offering `focus`, `set_value`, `activate` and a way out for as long as that
/// window lasts. The verdict would stay honest; the tree would not.
fn publish_wind_down(harness: &Harness, layer: &mut TariaLayer, screen: &Screen, status: &Status) {
    if layer.is_enabled() {
        layer.publish(agent::build_nodes(
            status,
            harness.editor(),
            &screen.transcript,
            Liveness::WindingDown,
        ));
    }
}

/// Hand every agent input that has arrived to the harness, and collect the
/// [`UiEvent`]s it produced.
///
/// [`drain_acking`](TariaLayer::drain_acking) acks each input as the handler
/// returns its verdict, so an act this app looked at and deliberately did
/// nothing with — a node that is gone, an action this phase does not offer, a
/// `set_value` with nothing to set — reaches the agent as `Ignored` rather
/// than as a silence it has to time out. The events are collected rather than
/// applied here because applying one needs the ribbon, the screen and the
/// command channel, none of which this closure can borrow while the harness is
/// borrowed; they go through [`steer`] with the keyboard's, which is the whole
/// point.
fn pump_agent(harness: &mut Harness, layer: &TariaLayer, events: &mut Vec<UiEvent>) {
    layer.drain_acking(|input| {
        let (status, produced) = harness.apply_agent_input(&input);
        events.extend(produced);
        status
    });
}

/// Apply one [`UiEvent`], whoever produced it, and say whether the session is
/// over.
///
/// The one set of arms both input paths go through. A keystroke and an agent
/// act arrive here having already been reduced to the same four events by
/// [`Harness`], so `/reset` typed at the panel and `/reset` submitted by an
/// agent are not two implementations of the same command — they are one, and
/// there is no second copy to drift.
fn steer(
    harness: &mut Harness,
    ribbon: &mut Ribbon,
    screen: &mut Screen,
    commands: &Sender<Command>,
    interrupt: &AtomicBool,
    event: UiEvent,
) -> io::Result<bool> {
    match event {
        UiEvent::Submit(line) => match route(&line) {
            Routed::Nothing => {}
            Routed::Local(text) => {
                ribbon.system(harness, &text)?;
                screen.transcript.push_system(&text);
            }
            Routed::Leave => return Ok(true),
            Routed::Send(command) => {
                if let Command::Submit(message) = &command {
                    ribbon.user(harness, message)?;
                    screen.transcript.push_user(message);
                    // Before the worker has said a word: from here Esc means
                    // "stop this", not "leave", and the panel stops claiming
                    // to be idle during the seconds before the first prefill
                    // chunk lands.
                    screen.observe(Snapshot {
                        phase: Phase::Prefill,
                        ..Snapshot::idle(screen.snapshot.context)
                    });
                    screen.say("prefilling; the stop lands at the first token");
                }
                if commands.send(command).is_err() {
                    return Ok(true);
                }
            }
        },
        UiEvent::Interrupt => {
            interrupt.store(true, Ordering::SeqCst);
            screen.say("stopping at the next token...");
        }
        UiEvent::Exit => return Ok(true),
        // The harness has already redrawn; nothing here owns anything that a
        // resize changes.
        UiEvent::Redraw => {}
    }
    Ok(false)
}

/// Apply one worker message to the screen. Returns whether the session is over.
///
/// This is also the only place the interrupt flag is cleared, and it is the
/// UI's to clear for the same reason it is the UI's to set: the flag means
/// "stop the reply the panel is showing", so it stops meaning anything at the
/// instant the panel stops showing one. Clearing it on the worker instead
/// leaves a window either side of a turn boundary in which a Ctrl-C is lost or
/// lands on the wrong turn.
fn absorb(
    harness: &mut Harness,
    ribbon: &mut Ribbon,
    screen: &mut Screen,
    failure: &mut Option<String>,
    interrupt: &AtomicBool,
    output: Output,
) -> io::Result<bool> {
    match output {
        Output::Ready { banner, model } => {
            ribbon.system(harness, &banner)?;
            screen.transcript.push_system(&banner);
            screen.model = Some(model);
            // The keys the idle row shows are the panel's own; nothing needs
            // to be said over them.
            screen.clear_detail();
        }
        Output::Token(text) => {
            ribbon.say(harness, &text)?;
            screen.transcript.push_model(&text);
        }
        Output::Progress(snapshot) => {
            // The first event of a phase is where the row stops being told
            // what is about to happen and starts reporting what is.
            screen.clear_detail();
            screen.observe(snapshot);
        }
        Output::Notice(text) => {
            ribbon.system(harness, &text)?;
            screen.transcript.push_system(&text);
        }
        Output::TurnDone { context } => {
            ribbon.hush(harness)?;
            screen.observe(Snapshot::idle(context));
            screen.clear_detail();
            interrupt.store(false, Ordering::SeqCst);
        }
        Output::Failed(reason) => {
            ribbon.system(harness, &reason)?;
            screen.transcript.push_system(&reason);
            *failure = Some(reason);
            return Ok(true);
        }
    }
    Ok(false)
}

// ---------------------------------------------------------------------------
// the worker thread
// ---------------------------------------------------------------------------

/// The next command to serve out of what the UI has queued, or `None` to stop.
///
/// In order, with one exception: a `Shutdown` anywhere in the queue outranks
/// everything in front of it. Type-ahead behaves as it does at a line prompt
/// right up until the user leaves, at which point the messages they typed and
/// then abandoned must not each cost a full prefill before the process can
/// exit.
fn next_command(pending: &mut VecDeque<Command>) -> Option<Command> {
    if pending.iter().any(|command| *command == Command::Shutdown) {
        return None;
    }
    pending.pop_front()
}

/// Send one message, and say whether the UI is still listening.
///
/// A UI that has gone is not an error the worker can report to anyone — there
/// is nobody left to report it to — so every send site treats a closed channel
/// as the end of the session rather than as something to unwrap.
fn tell(outputs: &Sender<Output>, output: Output) -> bool {
    outputs.send(output).is_ok()
}

fn worker_main(
    args: ChatArgs,
    dials: &Dials,
    commands: &Receiver<Command>,
    outputs: &Sender<Output>,
    interrupt: &AtomicBool,
) {
    if let Err(error) = worker_session(args, dials, commands, outputs, interrupt) {
        // The UI may already be gone, in which case there is nobody to tell
        // and nothing to do about it.
        let _ = outputs.send(Output::Failed(format!("{error:#}")));
    }
}

/// The whole model-owning half: load, then serve commands until asked to stop.
///
/// The setup mirrors `run_chat` deliberately — the same seed assembly, the
/// same one-state-per-session rule, the same banner — so the two front ends
/// cannot drift into describing different runtimes.
fn worker_session(
    args: ChatArgs,
    dials: &Dials,
    commands: &Receiver<Command>,
    outputs: &Sender<Output>,
    interrupt: &AtomicBool,
) -> anyhow::Result<()> {
    let model_dir = args.model.as_path();
    let tokenizer = load_tokenizer(model_dir)?;
    // The same dial the UI thread read, plus the manifest's veto. A refusal
    // here reaches the user as `Output::Failed`.
    let context = crate::checked_context(dials, model_dir)?;
    let sanitizer = tokenizer.content_sanitizer();

    let mut seed: Vec<ChatMessage> = Vec::new();
    if let Some(system) = args.system.as_deref() {
        seed.push(ChatMessage::system(system));
    }
    if let Some(path) = args.messages_file.as_deref() {
        let seeded = crate::read_messages_file(path)?;
        let two_systems =
            !seed.is_empty() && seeded.first().is_some_and(|m| m.role == Role::System);
        if two_systems
            && !tell(
                outputs,
                Output::Notice(format!(
                    "warning: --system and a {} that also starts with a system turn; \
                     both are sent, --system first",
                    path.display()
                )),
            )
        {
            return Ok(());
        }
        seed.extend(seeded);
    }
    // Sanitized once, here, so `/reset` cannot restore an unsanitized seed.
    let mut transcript = Transcript::new(tokenizer.sanitize_messages(&seed));
    let codec = TurnCodec::new(&tokenizer)?;

    let load_start = Instant::now();
    let model = Model::load(model_dir, args.runtime.load_options())
        .with_context(|| format!("loading model from {}", model_dir.display()))?;
    // Built once for the whole session: every turn continues this cache rather
    // than rebuilding the slot pool, the ring and the compute pool.
    let mut state = ForwardState::with_config(&model, context, dials.runtime_config())?;
    dials.apply_prefill(&mut state)?;

    let mut params = GenerateParams::from_defaults(tokenizer.sampling_defaults());
    params.max_new = args.max_new;
    params.greedy = args.greedy;
    if let Some(t) = args.temperature {
        params.temperature = t;
    }
    if let Some(k) = args.top_k {
        params.top_k = Some(k);
    }
    if let Some(p) = args.top_p {
        params.top_p = p;
    }
    let base_seed = args.seed.unwrap_or(params.seed);

    let banner = format!(
        "model loaded in {:.2}s; context cap {context}, --max-new {} reserved per turn; \
         {} compute shards, {} expert slots/layer from a {} budget, {} reads",
        load_start.elapsed().as_secs_f64(),
        args.max_new,
        state.shards(),
        state.slots_per_layer(),
        human_bytes(state.cache_bytes()),
        state.stream_mode(),
    );
    if !tell(
        outputs,
        Output::Ready {
            banner,
            // The install's own identifier, verbatim: a prettier name would
            // have to be guessed at, and the panel is not the place to guess
            // which model the user is talking to.
            model: model.manifest().model_id.clone(),
        },
    ) {
        return Ok(());
    }

    let mut turn: u64 = 0;
    // Every id this conversation consists of; see `chat_turn` in `main.rs`.
    // Empty means "nothing prefilled yet", which is where `/reset` puts it.
    let mut history: Vec<u32> = Vec::new();

    // Commands are served strictly in order, so typing ahead during a long
    // turn behaves as it does at a line prompt — with one exception: a
    // `Shutdown` already in the queue jumps it. Without that, leaving mid-reply
    // with two messages typed ahead would run both of them to completion first,
    // and the join on the way out would take minutes.
    let mut pending: VecDeque<Command> = VecDeque::new();
    loop {
        if pending.is_empty() {
            // `recv` fails only when the UI has dropped its sender, which is
            // the one shutdown path that does not send `Command::Shutdown`.
            match commands.recv() {
                Ok(command) => pending.push_back(command),
                Err(_) => break,
            }
        }
        loop {
            match commands.try_recv() {
                Ok(command) => pending.push_back(command),
                Err(TryRecvError::Empty) => break,
                // A UI that has gone cannot be answered, so whatever it typed
                // ahead is not worth a prefill.
                Err(TryRecvError::Disconnected) => {
                    pending.clear();
                    break;
                }
            }
        }
        let Some(command) = next_command(&mut pending) else {
            break;
        };
        match command {
            // Unreachable: `next_command` takes this branch first. Kept so the
            // loop is correct on its own terms rather than on that one's.
            Command::Shutdown => break,
            Command::Reset => {
                let dropped = transcript.live_turns();
                transcript.reset();
                // Drops every cached position and keeps every allocation, so
                // the next turn re-prefills the seed and nothing else.
                state.reset();
                history.clear();
                if !tell(
                    outputs,
                    Output::Notice(format!("reset: dropped {dropped} turns, kept the seed")),
                ) || !tell(outputs, Output::TurnDone { context: 0 })
                {
                    break;
                }
            }
            Command::Save(path) => {
                let saved = transcript
                    .to_json()
                    .context("serializing the transcript")
                    .and_then(|json| {
                        std::fs::write(&path, json)
                            .with_context(|| format!("writing {}", path.display()))
                    });
                let notice = match saved {
                    Ok(()) => format!(
                        "saved {} messages to {}",
                        transcript.messages().len(),
                        path.display()
                    ),
                    // A bad path is the user's typo, not a reason to lose the
                    // conversation.
                    Err(error) => format!("save failed: {error:#}"),
                };
                if !tell(outputs, Output::Notice(notice)) {
                    break;
                }
            }
            Command::Submit(message) => {
                let plan = plan_turn(
                    &tokenizer,
                    &codec,
                    &mut transcript,
                    history.len(),
                    &message,
                    params.max_new,
                    context,
                )?;
                let new_ids = match plan {
                    TurnPlan::Ready { new_ids, .. } => new_ids,
                    TurnPlan::Refused(reason) => {
                        // A refused turn still reports a `TurnDone`: it is what
                        // takes the panel back to idle, and what clears the
                        // interrupt flag.
                        if !tell(outputs, Output::Notice(reason))
                            || !tell(
                                outputs,
                                Output::TurnDone {
                                    context: state.seq_len().unwrap_or(0),
                                },
                            )
                        {
                            break;
                        }
                        continue;
                    }
                };

                params.seed = turn_seed(base_seed, turn);
                turn += 1;
                let outcome = run_turn(
                    &model,
                    &mut state,
                    &tokenizer,
                    &mut history,
                    &new_ids,
                    &params,
                    outputs,
                    interrupt,
                )?;
                // The partial reply is kept: it is what the model actually
                // said and what the next turn's context has to contain to stay
                // coherent. The cache already holds it — this is the
                // human-readable copy.
                transcript.push(sanitizer, Role::Assistant, &outcome.reply);
                if outcome.interrupted
                    && !tell(
                        outputs,
                        Output::Notice(format!(
                            "[interrupted after {} bytes; kept as the reply]",
                            outcome.reply.len()
                        )),
                    )
                {
                    break;
                }
                let done = Output::TurnDone {
                    context: state.seq_len().unwrap_or(0),
                };
                if !tell(outputs, done) || outcome.ui_gone {
                    break;
                }
            }
        }
    }
    Ok(())
}

/// What one turn produced.
struct TurnOutcome {
    reply: String,
    interrupted: bool,
    /// The UI stopped listening mid-turn.
    ui_gone: bool,
}

/// Stream one reply into the conversation already in `state`, reporting as it
/// goes.
///
/// The bookkeeping is `chat_turn`'s, for the reasons documented there: one
/// state for the session, `state.seq_len()` as the only authority on what the
/// cache holds, and the ids the model *emitted* extended onto the history —
/// never a re-encoding of the reply text. What is added is the progress
/// callback and an interrupt that arrives as a flag rather than as a signal.
#[allow(clippy::too_many_arguments)]
fn run_turn(
    model: &Model,
    state: &mut ForwardState,
    tokenizer: &RvmpTokenizer,
    history: &mut Vec<u32>,
    new_ids: &[u32],
    params: &GenerateParams,
    outputs: &Sender<Output>,
    interrupt: &AtomicBool,
) -> anyhow::Result<TurnOutcome> {
    history.extend_from_slice(new_ids);
    let fed = state.seq_len().context("reading the KV cache position")?;
    if fed > history.len() {
        // Unreachable unless the two drift apart, which would mean prefilling
        // ids the model never saw. Refuse rather than slice-panic.
        bail!(
            "chat: the KV cache holds {fed} positions but the conversation is only {} ids long",
            history.len()
        );
    }
    let prompt = history.len() - fed;

    // Where this turn's streaming starts. The state's counters run from
    // session start and `reset` keeps them, so the panel's byte figure is the
    // delta against this or it is a session total wearing a turn's label.
    let at_turn_start = state.stream_stats_in(StreamPhase::Decode);
    let read_base = at_turn_start.bytes_read + at_turn_start.sweep_bytes_read;
    // The sweep narrows its chunk to whatever the slot slab can host, so the
    // configured width is not necessarily the width in use. The first event is,
    // and it is the only place that width is observable from out here.
    let chunk_width = Cell::new(0usize);
    let mut reply = String::new();
    // Only read on the interrupted path, where `generate_from_with_progress`
    // never returns its stats: the unwind leaves `on_token` before the id is
    // fed, so this ends up holding exactly what `generated_ids` would have —
    // every id sampled, the last of them not yet in the cache. On the ordinary
    // path the stats are authoritative and this is ignored, which is why the
    // duplicate call made to flush a trailing partial character does not have
    // to be filtered out.
    let mut spoken: Vec<u32> = Vec::new();

    // A UI that has stopped listening is not an error, it is a reason to stop;
    // the send is never allowed to panic the worker.
    let gone = Cell::new(false);
    let post = |output: Output| {
        if outputs.send(output).is_err() {
            gone.set(true);
        }
    };

    let outcome = {
        let mut on_progress = |event: GenerateProgress| {
            if let GenerateProgress::PrefillChunk { positions_done, .. } = event
                && chunk_width.get() == 0
            {
                chunk_width.set(positions_done);
            }
            post(Output::Progress(snapshot_of(
                event,
                fed,
                prompt,
                chunk_width.get(),
                read_base,
            )));
        };
        let mut on_token = |id: u32, text: &str| {
            reply.push_str(text);
            spoken.push(id);
            post(Output::Token(text.to_owned()));
            if interrupt.load(Ordering::SeqCst) || gone.get() {
                // The callback cannot report anything, so leaving is an
                // unwind. It happens between two forward passes, with no
                // expert read in flight and no worker fanned out, which is the
                // only point in the loop where that is cheap — and it is also
                // the only point where the KV cache is whole, which is what
                // makes keeping the state across an abort sound.
                std::panic::panic_any(ChatAbort);
            }
        };
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            generate_from_with_progress(
                model,
                state,
                tokenizer,
                &history[fed..],
                fed,
                params,
                Some(&mut on_progress),
                &mut on_token,
            )
        }))
    };
    let ui_gone = gone.get();

    match outcome {
        Ok(stats) => {
            let stats = stats?;
            history.extend_from_slice(&stats.generated_ids);
            Ok(TurnOutcome {
                reply,
                interrupted: false,
                ui_gone,
            })
        }
        Err(payload) => {
            if payload.downcast_ref::<ChatAbort>().is_none() {
                // Somebody else's panic: re-raise it untouched.
                std::panic::resume_unwind(payload);
            }
            // The partial reply is in the cache up to its second-to-last id,
            // so it has to be in the history too — otherwise the next turn
            // would prefill from a position the model reached by a route the
            // conversation no longer records.
            history.extend_from_slice(&spoken);
            Ok(TurnOutcome {
                reply,
                interrupted: true,
                ui_gone,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::DEFAULT_CONTEXT;

    fn stats(hits: u64, misses: u64, pending: u64) -> StreamStats {
        StreamStats {
            hits,
            misses,
            pending_hits: pending,
            ..StreamStats::default()
        }
    }

    /// Every command the line REPL understands has to work here too, including
    /// the `//` escape — a front end that quietly dropped one would send
    /// `/exit` to the model.
    #[test]
    fn the_panel_routes_every_command_the_line_repl_has() {
        assert_eq!(route(""), Routed::Nothing);
        assert_eq!(route("   \t "), Routed::Nothing);
        assert_eq!(route("/exit"), Routed::Leave);
        assert_eq!(route("/quit"), Routed::Leave);
        assert_eq!(route("/q"), Routed::Leave);
        assert_eq!(route("/reset"), Routed::Send(Command::Reset));
        assert_eq!(route("/clear"), Routed::Send(Command::Reset));
        assert_eq!(
            route("/save chat.json"),
            Routed::Send(Command::Save(PathBuf::from("chat.json")))
        );
        assert_eq!(route("/help"), Routed::Local(REPL_HELP.to_owned()));
        assert_eq!(
            route("hello there"),
            Routed::Send(Command::Submit("hello there".to_owned()))
        );
        // The escape hatch: `//exit` is a message, not a command.
        assert_eq!(
            route("//exit"),
            Routed::Send(Command::Submit("/exit".to_owned()))
        );
        // A mistyped command is answered here, never sent to the model.
        let Routed::Local(reason) = route("/rest") else {
            panic!("/rest should be answered locally");
        };
        assert!(reason.contains("unknown command /rest"), "{reason}");
        let Routed::Local(reason) = route("/save") else {
            panic!("/save with no path should be answered locally");
        };
        assert!(reason.contains("needs a path"), "{reason}");
    }

    /// The progress events the runtime reports, as the panel reads them. The
    /// context figure is the *conversation's* position, so a turn continuing a
    /// warm cache does not restart the context bar at zero.
    #[test]
    fn progress_events_become_panel_numbers() {
        let prefill = snapshot_of(
            GenerateProgress::PrefillChunk {
                positions_done: 1024,
                positions_total: 3961,
            },
            1000,
            3961,
            512,
            0,
        );
        assert_eq!(prefill.phase, Phase::Prefill);
        assert_eq!(
            prefill.prefill,
            Some(Prefilling {
                done: 1024,
                total: Some(3961)
            })
        );
        assert_eq!(prefill.chunk, Some((2, 8)));
        assert_eq!(prefill.context, 2024);
        assert_eq!(prefill.tokens, 0);
        // The sweep bypasses the cache, so there is no rate to claim yet —
        // and it reports no bytes either.
        assert_eq!(prefill.hit_rate, None);
        assert_eq!(prefill.read_bytes, None);

        let decode = snapshot_of(
            GenerateProgress::DecodeToken {
                index: 0,
                stats: StreamStats {
                    bytes_read: 3_000_000_000,
                    sweep_bytes_read: 500_000_000,
                    ..stats(30, 10, 5)
                },
            },
            1000,
            3961,
            512,
            1_000_000_000,
        );
        assert_eq!(decode.phase, Phase::Decode);
        assert_eq!(decode.prefill, None);
        assert_eq!(decode.chunk, None);
        // `index` is 0-based over the tokens generated; the panel counts them.
        assert_eq!(decode.tokens, 1);
        assert_eq!(decode.context, 4961);
        assert_eq!(decode.hit_rate, Some(0.75));
        // This turn's streaming, not the session's: the counters run from the
        // state's construction and the baseline comes off them.
        assert_eq!(decode.read_bytes, Some(2_500_000_000));
    }

    /// The sweep's chunk width is not the configured one — it is narrowed to
    /// whatever the slot slab can host — so the panel takes it from the first
    /// event and counts from there. A width that is not a chunking says so by
    /// reporting nothing.
    #[test]
    fn the_chunk_counter_is_derived_from_the_width_the_run_revealed() {
        assert_eq!(chunk_of(512, 3961, 512), Some((1, 8)));
        assert_eq!(chunk_of(1024, 3961, 512), Some((2, 8)));
        assert_eq!(chunk_of(3961, 3961, 512), Some((8, 8)));
        // The last chunk is short, and it is still the eighth of eight.
        assert_eq!(chunk_of(3900, 3961, 512), Some((8, 8)));
        // Position zero is in the first chunk, not the zeroth.
        assert_eq!(chunk_of(0, 3961, 512), Some((1, 8)));
        // Nothing to count: one chunk, per-token reporting, or no width yet.
        assert_eq!(chunk_of(100, 400, 512), None);
        assert_eq!(chunk_of(100, 3961, 1), None);
        assert_eq!(chunk_of(100, 3961, 0), None);
    }

    /// Pending hits issue no read and avoid none, so they belong in neither
    /// half of the rate; and nothing routed is not a zero percent hit rate.
    #[test]
    fn the_hit_rate_counts_hits_against_hits_and_misses() {
        assert_eq!(hit_rate(&stats(0, 0, 0)), None);
        // Pending hits are excluded from both halves: 30/(30+10), not 35/45.
        assert_eq!(hit_rate(&stats(30, 10, 5)), Some(0.75));
        assert_eq!(hit_rate(&stats(0, 4, 0)), Some(0.0));
        assert_eq!(hit_rate(&stats(4, 0, 0)), Some(1.0));
        // Whatever the counters say, the panel is handed a rate.
        for (hits, misses) in [(1, 3), (7, 0), (0, 9), (u32::MAX as u64, 1)] {
            let rate = hit_rate(&stats(hits, misses, 0)).expect("something was routed");
            assert!((0.0..=1.0).contains(&rate), "{hits}/{misses} gave {rate}");
        }
    }

    /// A prefill snapshot at `done` of 3,961 positions.
    fn prefilling(done: usize) -> Snapshot {
        Snapshot {
            phase: Phase::Prefill,
            prefill: Some(Prefilling {
                done,
                total: Some(3961),
            }),
            chunk: chunk_of(done, 3961, 512),
            context: done,
            ..Snapshot::idle(done)
        }
    }

    /// The panel's `ctx used/cap` denominator is the window this run was
    /// configured with, not a compiled-in 4096 — and it is right from the
    /// first frame, while the model is still loading, which is why the UI
    /// thread resolves it instead of waiting for the worker to say.
    #[test]
    fn the_panel_counts_against_the_configured_window() {
        let mut screen = Screen::loading(16_384);
        assert_eq!(screen.status().context, (0, 16_384));
        screen.observe(Snapshot::idle(9000));
        assert_eq!(screen.status().context, (9000, 16_384));
        // A window the default would have called overfull is merely half used.
        assert!(screen.status().context.0 > DEFAULT_CONTEXT);
    }

    /// The panel's clock belongs to the UI, and a phase change restarts it —
    /// otherwise decode's rate would be computed over prefill's minutes.
    #[test]
    fn the_phase_clock_restarts_when_the_phase_does() {
        let mut screen = Screen::loading(DEFAULT_CONTEXT);
        assert_eq!(screen.status().phase, Phase::Idle);
        assert_eq!(screen.status().context, (0, DEFAULT_CONTEXT));

        let prefill_at = Instant::now();
        screen.observe_at(prefilling(512), prefill_at);
        assert_eq!(screen.phase_started, prefill_at);

        // Another chunk of the same phase keeps the clock running, so the rate
        // is over the whole prefill and not over the last chunk.
        screen.observe_at(prefilling(1024), prefill_at + Duration::from_secs(46));
        assert_eq!(screen.phase_started, prefill_at);
        let status = screen.status();
        assert_eq!(
            status.prefill,
            Some(Prefilling {
                done: 1024,
                total: Some(3961)
            })
        );
        assert_eq!(status.context, (1024, DEFAULT_CONTEXT));

        // Decode restarts it, or its rate would be computed over prefill's
        // minutes.
        let decode_at = prefill_at + Duration::from_secs(360);
        screen.observe_at(
            Snapshot {
                phase: Phase::Decode,
                tokens: 1,
                context: 3962,
                hit_rate: Some(0.5),
                ..Snapshot::idle(3962)
            },
            decode_at,
        );
        assert_eq!(screen.phase_started, decode_at);
        assert_eq!(screen.status().tokens, 1);
        assert_eq!(screen.status().hit_rate, Some(0.5));
        // And the rate went with it: decode has one sample, which is not a
        // rate, rather than prefill's.
        assert_eq!(screen.status().rate, None);
    }

    /// The rate the panel shows is over a window, not over the whole run, and
    /// the ETA is that rate against what is left.
    ///
    /// This is the case the window exists for: the first stretch of a prefill
    /// is slow — it is where every expert file is touched for the first time —
    /// and a cumulative rate carries that stretch for the rest of the run, so
    /// an ETA derived from it is minutes long and visibly wrong.
    #[test]
    fn the_rate_is_rolling_and_the_eta_follows_it() {
        let start = Instant::now();
        let steady = |step: usize| {
            start + Duration::from_secs(64) + Duration::from_secs_f64(step as f64 * 32.0 / 11.0)
        };
        let mut screen = Screen::loading(DEFAULT_CONTEXT);
        screen.observe_at(prefilling(0), start);
        screen.observe_at(prefilling(128), start + Duration::from_secs(64));
        let slow = screen.status().rate.expect("two samples is a rate");
        assert!((slow - 2.0).abs() < 0.01, "{slow}");

        // Thirty steady chunks: the slow start is still in the window, so the
        // rate has climbed without having caught up.
        for step in 1..=30 {
            screen.observe_at(prefilling(128 + step * 32), steady(step));
        }
        let climbing = screen.status().rate.expect("a window of samples");
        assert!(climbing > slow && climbing < 11.0, "{climbing}");

        // Far enough in that the slow start has fallen out of the window
        // entirely, and the rate is the one the run is actually managing.
        for step in 31..=94 {
            screen.observe_at(prefilling(128 + step * 32), steady(step));
        }
        let status = screen.status();
        let rate = status.rate.expect("a full window");
        assert!((rate - 11.0).abs() < 0.001, "{rate}");
        // 3,961 - 3,136 positions left, at eleven a second.
        let eta = status.eta.expect("an ETA once there is a rate");
        assert!((eta.as_secs_f64() - 75.0).abs() < 0.01, "{eta:?}");
    }

    /// One sample is not a rate, and neither is a stalled one; the panel is
    /// handed nothing rather than a zero, an infinity or a NaN.
    #[test]
    fn a_rate_needs_two_samples_and_some_progress_between_them() {
        let start = Instant::now();
        let meter = |samples: &[(u64, usize)]| {
            let mut meter = Meter::default();
            for (at, done) in samples {
                meter.observe(start + Duration::from_secs(*at), *done);
            }
            meter
        };
        assert_eq!(meter(&[]).rate(), None, "no samples");
        assert_eq!(meter(&[(0, 10)]).rate(), None, "one sample");
        assert_eq!(meter(&[(0, 10), (0, 20)]).rate(), None, "no time");
        assert_eq!(meter(&[(0, 10), (10, 10)]).rate(), None, "no progress");
        // A count that went backwards is not a negative rate.
        assert_eq!(meter(&[(0, 20), (10, 10)]).rate(), None, "backwards");

        let running = meter(&[(0, 10), (10, 20), (20, 30)]);
        assert_eq!(running.rate(), Some(1.0));
        assert_eq!(running.eta(30), Some(Duration::from_secs(30)));
        assert_eq!(running.eta(0), Some(Duration::ZERO));

        // The window forgets, so it never grows without bound and the rate
        // stays the recent one.
        let mut long = Meter::default();
        for step in 0..1_000u64 {
            long.observe(start + Duration::from_secs(step), step as usize);
        }
        assert_eq!(long.samples.len(), WINDOW);
        assert_eq!(long.rate(), Some(1.0));
    }

    /// Leaving has to be prompt even when the user typed ahead: two queued
    /// messages behind a `/exit` are two full prefills the process would
    /// otherwise have to sit through before it could join the worker.
    #[test]
    fn a_queued_shutdown_outranks_the_messages_in_front_of_it() {
        let mut pending: VecDeque<Command> = VecDeque::new();
        assert_eq!(next_command(&mut pending), None, "an empty queue stops");

        pending.push_back(Command::Submit("first".to_owned()));
        pending.push_back(Command::Reset);
        assert_eq!(
            next_command(&mut pending),
            Some(Command::Submit("first".to_owned())),
            "ordinary commands are served in order"
        );

        pending.push_back(Command::Shutdown);
        pending.push_back(Command::Submit("never sent".to_owned()));
        assert_eq!(next_command(&mut pending), None);
        // And it keeps saying so, rather than draining past it.
        assert_eq!(next_command(&mut pending), None);
    }

    /// The harness reads `Status::phase` to decide whether Ctrl-C stops a
    /// reply or leaves, so the screen must not still claim to be idle while a
    /// turn it has already submitted is warming up.
    #[test]
    fn a_submitted_turn_is_not_idle_before_the_first_progress_event() {
        let mut screen = Screen::loading(DEFAULT_CONTEXT);
        screen.observe(Snapshot {
            phase: Phase::Prefill,
            ..Snapshot::idle(0)
        });
        assert_eq!(screen.status().phase, Phase::Prefill);
        assert_eq!(screen.status().prefill, None, "no counts to claim yet");
        // And no bar, no percent, no ETA to go with them.
        assert_eq!(screen.status().eta, None);

        screen.observe(Snapshot::idle(4096));
        assert_eq!(screen.status().phase, Phase::Idle);
        assert_eq!(screen.status().context, (4096, DEFAULT_CONTEXT));
    }
}
