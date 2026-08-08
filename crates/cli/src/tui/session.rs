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
//! [`TurnCodec`], [`plan_turn`], [`turn_seed`], [`PhaseStats`]. This module
//! adds a front end and a thread boundary, not a second implementation of the
//! chat policy — and in particular not a second copy of the context
//! accounting, which is the part that is easy to get subtly wrong.
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
//! so Ctrl-C during *prefill* is honoured at the first decoded token rather
//! than immediately. Prefill has no comparably safe unwind point and inventing
//! one is not in scope; the panel says so instead.

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

use super::{Harness, Phase, Status, UiEvent};
use crate::repl::{
    CONTEXT_CAP, PhaseStats, REPL_HELP, ReplInput, Transcript, TurnCodec, TurnPlan,
    parse_repl_input, plan_turn, turn_seed,
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
Ctrl-C stops a reply, Ctrl-D leaves.\n";

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
    /// The model is up. The text is the load banner.
    Ready(String),
    /// A chunk of reply text, exactly as `on_token` produced it.
    Token(String),
    /// Where the run is, as of one progress event.
    Progress(Snapshot),
    /// The turn is over, one way or another.
    TurnDone { context: usize, detail: String },
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
    prefill: Option<(usize, usize)>,
    tokens: usize,
    context: usize,
    hit_rate: Option<f32>,
}

impl Snapshot {
    /// Nothing is running and the conversation stands at `context`.
    fn idle(context: usize) -> Self {
        Self {
            phase: Phase::Idle,
            prefill: None,
            tokens: 0,
            context,
            hit_rate: None,
        }
    }
}

/// A generate call's progress event as the panel wants it.
///
/// `fed` is what the KV cache held before this turn and `prompt` is how many
/// ids this turn feeds, so the context figure is the conversation's position
/// rather than this call's offset into it.
fn snapshot_of(event: GenerateProgress, fed: usize, prompt: usize) -> Snapshot {
    match event {
        GenerateProgress::PrefillChunk {
            positions_done,
            positions_total,
        } => Snapshot {
            phase: Phase::Prefill,
            prefill: Some((positions_done, positions_total)),
            tokens: 0,
            context: fed + positions_done,
            // The chunked sweep bypasses the expert cache, so there is no hit
            // rate to report until decode starts asking it for anything.
            hit_rate: None,
        },
        GenerateProgress::DecodeToken { index, stats } => Snapshot {
            phase: Phase::Decode,
            prefill: None,
            tokens: index + 1,
            context: fed + prompt + index,
            hit_rate: hit_rate(&stats),
        },
    }
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
pub fn run_chat_tui(args: ChatArgs) -> anyhow::Result<()> {
    let mut harness = Harness::enter()
        .context("chat --tui needs an interactive terminal; drop --tui for the line REPL")?;
    // *After* `Harness::enter`, so the hush hook wraps the terminal-restoring
    // one rather than the other way round. A ChatAbort then stops at the hush
    // hook and the panel is left alone; a real panic passes through it, the
    // terminal is restored, and only then is the backtrace printed.
    hush_control_flow_panics();
    harness.write_transcript(GREETING)?;

    let (commands, from_ui) = mpsc::channel::<Command>();
    let (to_ui, outputs) = mpsc::channel::<Output>();
    let interrupt = Arc::new(AtomicBool::new(false));
    let worker = {
        let interrupt = Arc::clone(&interrupt);
        std::thread::Builder::new()
            .name("ramvamp-chat".to_owned())
            .stack_size(WORKER_STACK)
            .spawn(move || worker_main(args, &from_ui, &to_ui, &interrupt))
            .context("spawning the chat worker")?
    };

    let ended = drive(&mut harness, &commands, &outputs, &interrupt);

    // The terminal comes back before anything else is said, whichever way the
    // session ended — including a `drive` that failed mid-draw.
    let _ = harness.suspend();
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

/// How the session ended, from the UI's side.
struct Ended {
    /// The worker's fatal error, if it had one.
    failure: Option<String>,
    /// Whether the worker can be joined without making the user wait. False
    /// when the user asked twice to leave and the worker had not stopped.
    join: bool,
}

/// Everything the UI thread knows, which is only what the worker told it plus
/// a clock.
struct Screen {
    snapshot: Snapshot,
    /// When the current phase started, by the UI's own monotonic clock.
    ///
    /// The worker cannot send a duration from inside a generate call, so this
    /// is the honest way to have a live rate: the counts are the worker's, the
    /// elapsed time is measured here, and a phase change restarts it.
    phase_started: Instant,
    detail: Option<String>,
    /// Set while the transcript's last write was a token, so a notice knows to
    /// break the line first.
    speaking: bool,
    /// A draw is worth doing.
    dirty: bool,
}

impl Screen {
    fn loading() -> Self {
        Self {
            snapshot: Snapshot::idle(0),
            phase_started: Instant::now(),
            detail: Some("loading the model...".to_owned()),
            speaking: false,
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
        }
        self.snapshot = snapshot;
        self.dirty = true;
    }

    fn say(&mut self, detail: impl Into<String>) {
        self.detail = Some(detail.into());
        self.dirty = true;
    }

    fn status(&self) -> Status {
        Status {
            phase: self.snapshot.phase,
            prefill: self.snapshot.prefill,
            tokens: self.snapshot.tokens,
            elapsed: self.phase_started.elapsed(),
            context: (self.snapshot.context, CONTEXT_CAP),
            hit_rate: self.snapshot.hit_rate,
            detail: self.detail.clone(),
        }
    }

    /// One whole line of chrome, after closing a reply that was mid-word.
    fn line(&mut self, harness: &mut Harness, text: &str) -> io::Result<()> {
        if self.speaking {
            harness.write_transcript("\n")?;
            self.speaking = false;
        }
        harness.write_transcript(text)
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
    commands: &Sender<Command>,
    outputs: &Receiver<Output>,
    interrupt: &AtomicBool,
) -> anyhow::Result<Ended> {
    let mut screen = Screen::loading();
    let mut failure: Option<String> = None;

    'session: loop {
        // Everything the worker has said since the last pass. Draining rather
        // than taking one keeps a burst of tokens from being paced by POLL.
        loop {
            match outputs.try_recv() {
                Ok(output) => {
                    if absorb(harness, &mut screen, &mut failure, interrupt, output)? {
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
        // is the only one that does not get a redraw per pass.
        if screen.dirty || screen.snapshot.phase != Phase::Idle {
            harness.set_status(screen.status());
            screen.dirty = false;
        }

        match harness.poll(POLL)? {
            Some(UiEvent::Submit(line)) => match route(&line) {
                Routed::Nothing => {}
                Routed::Local(text) => screen.line(harness, &format!("\n{text}\n"))?,
                Routed::Leave => break 'session,
                Routed::Send(command) => {
                    if let Command::Submit(message) = &command {
                        screen.line(harness, &format!("\nyou> {message}\n"))?;
                        // Before the worker has said a word: from here Ctrl-C
                        // means "stop this", not "leave", and the panel stops
                        // claiming to be idle during the seconds before the
                        // first prefill chunk lands.
                        screen.observe(Snapshot {
                            phase: Phase::Prefill,
                            ..Snapshot::idle(screen.snapshot.context)
                        });
                        screen.say("prefilling; Ctrl-C stops the reply at its first token");
                    }
                    if commands.send(command).is_err() {
                        break 'session;
                    }
                }
            },
            Some(UiEvent::Interrupt) => {
                interrupt.store(true, Ordering::SeqCst);
                screen.say("stopping this reply...");
            }
            Some(UiEvent::Exit) => break 'session,
            _ => {}
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
    loop {
        match outputs.try_recv() {
            Ok(output) => {
                // A fatal error raised on the way out still gets reported, but
                // it cannot cut the wait short: the point of this loop is to
                // see the channel close.
                absorb(harness, &mut screen, &mut failure, interrupt, output)?;
                screen.say("stopping; Ctrl-C again to leave without waiting");
            }
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {
                harness.set_status(screen.status());
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
    screen: &mut Screen,
    failure: &mut Option<String>,
    interrupt: &AtomicBool,
    output: Output,
) -> io::Result<bool> {
    match output {
        Output::Ready(banner) => {
            screen.line(harness, &format!("{banner}\n"))?;
            screen.say("idle - Enter sends, /help lists the commands");
        }
        Output::Token(text) => {
            if !screen.speaking {
                harness.write_transcript("bot> ")?;
                screen.speaking = true;
            }
            harness.write_transcript(&text)?;
        }
        Output::Progress(snapshot) => {
            if screen.snapshot.phase != snapshot.phase && snapshot.phase == Phase::Decode {
                screen.say("decoding; Ctrl-C stops the reply");
            }
            screen.observe(snapshot);
        }
        Output::Notice(text) => screen.line(harness, &format!("{text}\n"))?,
        Output::TurnDone { context, detail } => {
            if screen.speaking {
                harness.write_transcript("\n")?;
                screen.speaking = false;
            }
            screen.observe(Snapshot::idle(context));
            screen.say(detail);
            interrupt.store(false, Ordering::SeqCst);
        }
        Output::Failed(reason) => {
            screen.line(harness, &format!("\n{reason}\n"))?;
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
    commands: &Receiver<Command>,
    outputs: &Sender<Output>,
    interrupt: &AtomicBool,
) {
    if let Err(error) = worker_session(args, commands, outputs, interrupt) {
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
    commands: &Receiver<Command>,
    outputs: &Sender<Output>,
    interrupt: &AtomicBool,
) -> anyhow::Result<()> {
    let model_dir = args.model.as_path();
    let tokenizer = load_tokenizer(model_dir)?;
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
    let mut state = ForwardState::with_config(&model, CONTEXT_CAP, args.runtime.runtime_config())?;
    args.prefill.apply(&mut state)?;

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
        "model loaded in {:.2}s; context cap {CONTEXT_CAP}, --max-new {} reserved per turn; \
         {} compute shards, {} expert slots/layer from a {} budget, {} reads",
        load_start.elapsed().as_secs_f64(),
        args.max_new,
        state.shards(),
        state.slots_per_layer(),
        human_bytes(state.cache_bytes()),
        state.stream_mode(),
    );
    if !tell(outputs, Output::Ready(banner)) {
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
                ) || !tell(
                    outputs,
                    Output::TurnDone {
                        context: 0,
                        detail: "reset - the next turn prefills the seed again".to_owned(),
                    },
                ) {
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
                                    detail: "the turn was refused; nothing was sent".to_owned(),
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
                    detail: turn_detail(&outcome),
                };
                if !tell(outputs, done) || outcome.ui_gone {
                    break;
                }
            }
        }
    }
    Ok(())
}

/// What one turn produced, and what it cost.
struct TurnOutcome {
    reply: String,
    interrupted: bool,
    /// Tokens the model actually produced, from the stats on the ordinary path
    /// and from the ids seen on the aborted one.
    generated: usize,
    /// The streaming counters for this turn alone.
    span: PhaseStats,
    elapsed: Duration,
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
    // session start and `reset` keeps them, so the summary below is the delta
    // against this or it is a session total wearing a turn's label.
    let at_turn_start = PhaseStats::take(state);
    let started = Instant::now();
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
        let mut on_progress =
            |event: GenerateProgress| post(Output::Progress(snapshot_of(event, fed, prompt)));
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
                generated: stats.generated,
                span: PhaseStats::take(state).since(&at_turn_start),
                elapsed: started.elapsed(),
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
            let generated = spoken.len();
            history.extend_from_slice(&spoken);
            Ok(TurnOutcome {
                reply,
                interrupted: true,
                generated,
                span: PhaseStats::take(state).since(&at_turn_start),
                elapsed: started.elapsed(),
                ui_gone,
            })
        }
    }
}

/// The one line the panel shows between turns.
///
/// Deliberately **not** any of the `report_*` footers: those go to stderr in a
/// shape `scripts/cold_bench.py` and the phase 8/9 sweeps parse, and a front
/// end that reworded them would break a benchmark rather than a screen. This
/// says the same things in a sentence that cannot be mistaken for one of them
/// — no `tok/s`, no `prefill:`, no `decode:` — and it never leaves stdout.
fn turn_detail(outcome: &TurnOutcome) -> String {
    let decode = outcome.span.phase(StreamPhase::Decode);
    let prefill = outcome.span.phase(StreamPhase::Prefill);
    let streamed =
        prefill.bytes_read + prefill.sweep_bytes_read + decode.bytes_read + decode.sweep_bytes_read;
    let resolved = decode.hits + decode.misses;
    let experts = if resolved == 0 {
        "no experts routed".to_owned()
    } else {
        format!("{} of {resolved} experts warm", decode.hits)
    };
    format!(
        "last turn: {} tokens in {:.1} s{}; {experts}; {} streamed",
        outcome.generated,
        outcome.elapsed.as_secs_f64(),
        if outcome.interrupted {
            " (stopped)"
        } else {
            ""
        },
        human_bytes(streamed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(hits: u64, misses: u64, pending: u64) -> StreamStats {
        StreamStats {
            hits,
            misses,
            pending_hits: pending,
            ..StreamStats::default()
        }
    }

    /// A turn span in which only decode streamed anything, laid out in
    /// `StreamPhase::ALL` order rather than by casting the enum.
    fn decode_only(decode: StreamStats) -> PhaseStats {
        let mut span = PhaseStats::default();
        for (slot, phase) in span.0.iter_mut().zip(StreamPhase::ALL) {
            if phase == StreamPhase::Decode {
                *slot = decode;
            }
        }
        span
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
                positions_done: 512,
                positions_total: 3961,
            },
            1000,
            3961,
        );
        assert_eq!(prefill.phase, Phase::Prefill);
        assert_eq!(prefill.prefill, Some((512, 3961)));
        assert_eq!(prefill.context, 1512);
        assert_eq!(prefill.tokens, 0);
        // The sweep bypasses the cache, so there is no rate to claim yet.
        assert_eq!(prefill.hit_rate, None);

        let decode = snapshot_of(
            GenerateProgress::DecodeToken {
                index: 0,
                stats: stats(30, 10, 5),
            },
            1000,
            3961,
        );
        assert_eq!(decode.phase, Phase::Decode);
        assert_eq!(decode.prefill, None);
        // `index` is 0-based over the tokens generated; the panel counts them.
        assert_eq!(decode.tokens, 1);
        assert_eq!(decode.context, 4961);
        assert_eq!(decode.hit_rate, Some(0.75));
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

    /// The idle row has to say something true about the turn that just ran,
    /// and it has to say it in words no benchmark parser is looking for:
    /// `scripts/cold_bench.py`'s `TIMING_RE` and the phase 8/9 sweeps key off
    /// `prefill:`, `decode:` and `tok/s`.
    #[test]
    fn the_idle_detail_cannot_be_mistaken_for_a_timing_line() {
        let span = decode_only(StreamStats {
            hits: 812,
            misses: 212,
            bytes_read: 3 * 1024 * 1024 * 1024,
            ..StreamStats::default()
        });
        let detail = turn_detail(&TurnOutcome {
            reply: "hello".to_owned(),
            interrupted: false,
            generated: 37,
            span,
            elapsed: Duration::from_millis(18_400),
            ui_gone: false,
        });
        assert_eq!(
            detail,
            "last turn: 37 tokens in 18.4 s; 812 of 1024 experts warm; 3.0 GiB streamed"
        );
        for forbidden in ["tok/s", "prefill:", "decode:"] {
            assert!(
                !detail.contains(forbidden),
                "{detail:?} contains {forbidden}"
            );
        }

        // An aborted turn says so rather than reporting a stop it never had.
        let stopped = turn_detail(&TurnOutcome {
            reply: String::new(),
            interrupted: true,
            generated: 2,
            span: PhaseStats::default(),
            elapsed: Duration::from_millis(900),
            ui_gone: false,
        });
        assert_eq!(
            stopped,
            "last turn: 2 tokens in 0.9 s (stopped); no experts routed; 0 B streamed"
        );
    }

    /// The panel's clock belongs to the UI, and a phase change restarts it —
    /// otherwise decode's rate would be computed over prefill's minutes.
    #[test]
    fn the_phase_clock_restarts_when_the_phase_does() {
        let mut screen = Screen::loading();
        assert_eq!(screen.status().phase, Phase::Idle);
        assert_eq!(screen.status().context, (0, CONTEXT_CAP));

        let prefill_at = Instant::now();
        screen.observe_at(
            Snapshot {
                phase: Phase::Prefill,
                prefill: Some((512, 3961)),
                tokens: 0,
                context: 512,
                hit_rate: None,
            },
            prefill_at,
        );
        assert_eq!(screen.phase_started, prefill_at);

        // Another chunk of the same phase keeps the clock running, so the rate
        // is over the whole prefill and not over the last chunk.
        screen.observe_at(
            Snapshot {
                phase: Phase::Prefill,
                prefill: Some((1024, 3961)),
                tokens: 0,
                context: 1024,
                hit_rate: None,
            },
            prefill_at + Duration::from_secs(46),
        );
        assert_eq!(screen.phase_started, prefill_at);
        let status = screen.status();
        assert_eq!(status.prefill, Some((1024, 3961)));
        assert_eq!(status.context, (1024, CONTEXT_CAP));

        // Decode restarts it, or its rate would be computed over prefill's
        // minutes.
        let decode_at = prefill_at + Duration::from_secs(360);
        screen.observe_at(
            Snapshot {
                phase: Phase::Decode,
                prefill: None,
                tokens: 1,
                context: 3962,
                hit_rate: Some(0.5),
            },
            decode_at,
        );
        assert_eq!(screen.phase_started, decode_at);
        assert_eq!(screen.status().tokens, 1);
        assert_eq!(screen.status().hit_rate, Some(0.5));
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
        let mut screen = Screen::loading();
        screen.observe(Snapshot {
            phase: Phase::Prefill,
            ..Snapshot::idle(0)
        });
        assert_eq!(screen.status().phase, Phase::Prefill);
        assert_eq!(screen.status().prefill, None, "no counts to claim yet");

        screen.observe(Snapshot::idle(4096));
        assert_eq!(screen.status().phase, Phase::Idle);
        assert_eq!(screen.status().context, (4096, CONTEXT_CAP));
    }
}
