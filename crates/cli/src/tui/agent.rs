//! The machine-facing view of the session: a rolling copy of the conversation
//! and the semantic node tree taria publishes beside every frame.
//!
//! # Why the ring exists
//!
//! The panel's *human* transcript has no in-process buffer, on purpose (see
//! the module docs of [`super`]): a decoded token goes `Output::Token` ->
//! [`Ribbon::say`](super::Ribbon::say) -> stdout, and the terminal's own
//! scrollback keeps it so it wraps, selects and copies natively. Nothing in
//! this process can read it back. An agent asking what the model just said
//! would therefore be told nothing at all, so [`TranscriptRing`] is a passive
//! tap alongside the `ribbon.*` calls that keeps the last few kilobytes for
//! the tree and changes nothing the user sees.
//!
//! # Why everything here is pure
//!
//! [`build_nodes`] is a function of a [`Status`], a [`LineEditor`] and a
//! ring — state in, nodes out, no terminal and no clock. That is what makes
//! the invariants the tree has to hold (exactly one focused node, exactly one
//! advertised way out of every phase, an action advertised only where it does
//! something) unit tests rather than things to hope for.
//!
//! [`plan`] is the same trick on the way back in: an [`AgentInput`], a phase
//! and the current draft go in, and what comes out is a [`Plan`] — a keystroke
//! or a paste — plus the [`InputStatus`] the input has earned. A [`Harness`]
//! cannot be built without a terminal, so a decision function that took one
//! could not be tested at all; this one is checked against the very tree that
//! advertised the action, which is the test that keeps the two honest.
//!
//! [`Harness`]: super::Harness
//!
//! # Why the numbers are rounded before they are published
//!
//! The adapter dedups identical trees, so publishing on every pass of the UI
//! loop is free — but only for as long as the tree holds still. A value that
//! moves every frame republishes the whole tree every frame and buries the
//! changes an agent came for. So the clock is left out entirely (an agent
//! cannot act on it), the rate is carried to one decimal, and percentages are
//! whole integers: exactly the precision a reader could act on and not one
//! digit more.

use ratatui::crossterm::event::{KeyCode, KeyModifiers};
use taria_ratatui::taria::{Action, AgentInput, Node, Role};
use taria_ratatui::{InputStatus, to_crossterm_key};

use super::input::{LineEditor, normalise_paste};
// The panel's own arithmetic, not a copy of it: a number an agent reads and a
// number the panel draws have to be the same number, and two implementations
// of `done * 100 / total` are two places for that to stop being true.
use super::panel::{clock, percent};
use super::{Phase, Status};
use crate::human_bytes;

/// Bytes of conversation the ring keeps.
///
/// About two screens of prose. Big enough that an agent reading after a
/// reply sees the whole of it, small enough that the value is not the
/// dominant cost of every snapshot: the tree is republished whenever a token
/// lands, and the ring's tail is by far its largest field.
const TRANSCRIPT_CAP: usize = 6 * 1024;

/// Speaker marks in the retained text.
///
/// Plain words rather than the panel's `›` and `•`: the ring is read by an
/// agent, which gets nothing from a glyph that the terminal had to be probed
/// for, and `you:` / `ramvamp:` survive being quoted back into a prompt.
const YOU: &str = "you: ";
const MODEL: &str = "ramvamp: ";
const NOTE: &str = "note: ";

/// What [`TranscriptRing::trim`] actually trims to.
///
/// The longest speaker mark, held back from [`TRANSCRIPT_CAP`] rather than
/// spent after the cut: a trim that lands inside a turn puts that turn's mark
/// back in front of what is left, and reserving the room first is what stops
/// putting it back from pushing the tail over the cap again.
const TAIL_CAP: usize = TRANSCRIPT_CAP - MODEL.len();

/// Between two facts in a node value. Fixed rather than taken from
/// [`Glyphs`](super::glyphs::Glyphs): what the terminal can draw has no
/// bearing on what an agent can read.
const SEPARATOR: &str = " · ";

// ---------------------------------------------------------------------------
// the transcript ring
// ---------------------------------------------------------------------------

/// A bounded rolling tail of the conversation, marked by speaker.
///
/// Turns are appended and the front is dropped a whole line at a time once
/// the tail is over [`TRANSCRIPT_CAP`], so what an agent reads always begins
/// at the start of a line. Every method is total: there is nothing here that
/// can fail, and nothing that can panic on a multi-byte character.
#[derive(Debug, Default)]
pub(crate) struct TranscriptRing {
    text: String,
    /// A model turn is open, so the next chunk continues its line rather than
    /// opening one of its own. The same rule [`Ribbon`](super::Ribbon)
    /// follows, for the same reason: tokens arrive mid-word and one entry per
    /// token would be unreadable.
    speaking: bool,
}

impl TranscriptRing {
    /// The user's turn, verbatim.
    pub(crate) fn push_user(&mut self, text: &str) {
        self.turn(YOU, text);
    }

    /// A chunk of the model's reply.
    ///
    /// Chunks coalesce: the first one opens a `ramvamp:` turn and every chunk
    /// after it appends to the same turn, until something else is pushed.
    pub(crate) fn push_model(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if !self.speaking {
            self.close();
            self.text.push_str(MODEL);
            self.speaking = true;
        }
        self.text.push_str(text);
        self.trim();
    }

    /// A banner, a refusal, a save: whatever the panel says in its own voice.
    pub(crate) fn push_system(&mut self, text: &str) {
        self.turn(NOTE, text);
    }

    /// The retained tail, oldest first.
    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    /// One complete turn: close whatever was open, mark the speaker, and end
    /// on a line of its own so the next trim has a boundary to cut at.
    fn turn(&mut self, mark: &str, text: &str) {
        self.close();
        self.text.push_str(mark);
        self.text.push_str(text.trim_end_matches('\n'));
        self.text.push('\n');
        self.trim();
    }

    /// Close a model turn that stopped mid-word.
    fn close(&mut self) {
        if self.speaking {
            self.speaking = false;
            if !self.text.ends_with('\n') {
                self.text.push('\n');
            }
        }
    }

    /// Drop the front until the tail fits, and keep what is left attributed.
    ///
    /// Whole lines first, so the tail starts at the start of a line rather
    /// than in the middle of a word. Starting at a line is not the same as
    /// starting at a *turn*, which is what an agent needs: a reply with
    /// newlines in it is stored as one marked line and however many unmarked
    /// continuations, so a cut landing inside one used to leave the tail
    /// opening on a paragraph with nobody's name on it. The mark of the turn
    /// the cut ran into is therefore put back in front of what is left, and
    /// [`TAIL_CAP`] is the room held back for it.
    ///
    /// A single line longer than the whole budget — a reply that has streamed
    /// six kilobytes without a newline — has no line boundary to cut at, so it
    /// is cut on a character boundary instead; slicing a `char` in half would
    /// produce a `String` that is not UTF-8, which is the one thing `drain`
    /// will panic on. That tail is re-marked too: a fragment of a word is
    /// still a fragment of somebody's word.
    fn trim(&mut self) {
        if self.text.len() <= TAIL_CAP {
            return;
        }
        // The speaker of the last turn the cut ran into, so what survives it
        // can be marked with the same name.
        let mut owner: Option<&'static str> = None;
        let mut cut = 0;
        while self.text.len() - cut > TAIL_CAP {
            let rest = &self.text[cut..];
            if let Some(mark) = turn_mark(rest) {
                owner = Some(mark);
            }
            let Some(at) = rest.find('\n') else {
                break;
            };
            cut += at + 1;
        }
        if self.text.len() - cut > TAIL_CAP {
            let want = self.text.len() - TAIL_CAP;
            cut = (want..=self.text.len())
                .find(|at| self.text.is_char_boundary(*at))
                .unwrap_or(self.text.len());
        }
        self.text.drain(..cut);
        if let Some(mark) = owner
            && !self.text.is_empty()
            && turn_mark(&self.text).is_none()
        {
            self.text.insert_str(0, mark);
        }
    }
}

/// The speaker a line opens, if it opens one.
///
/// What tells a turn's first line from its continuations, and so the one
/// thing [`TranscriptRing::trim`] needs to know to hand back a tail that says
/// who is talking.
fn turn_mark(line: &str) -> Option<&'static str> {
    [YOU, MODEL, NOTE]
        .into_iter()
        .find(|mark| line.starts_with(mark))
}

// ---------------------------------------------------------------------------
// the tree
// ---------------------------------------------------------------------------

/// Whether the session is still acting on what the tree advertises.
///
/// The wind-down loop keeps publishing, because a tree that freezes while the
/// worker is asked to stop leaves an agent reading a session that ended
/// minutes ago. What it no longer does is *apply* anything: every input it
/// drains is acked `Ignored` on the spot, so nothing waits out the bridge's
/// window, and an agent cannot reach the session's own exit path from there.
/// A tree that went on advertising `focus`, `set_value`, `activate` and the
/// way out through that window would be advertising four actions that cannot
/// work — the one thing the tree is never allowed to do — so it advertises
/// none of them instead, and the tree and the verdict agree again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Liveness {
    /// The UI loop is reading input and acting on it.
    Live,
    /// Leaving: the tree still reports, and offers nothing.
    WindingDown,
}

/// The semantic tree for one instant of the session.
///
/// Returned flat: the adapter wraps these in the `app` root it generates from
/// the label passed to `bind_or_disabled`, so an `app` node here would nest a
/// second one inside the first. Nine nodes at most, none with children, which
/// leaves the tree four levels clear of `taria::MAX_NODE_DEPTH`.
///
/// Node ids are the contract an agent holds across calls, so they are fixed
/// strings and the same string names the same thing in every phase: a node
/// whose data is absent is left out rather than renamed or reused.
pub(crate) fn build_nodes(
    status: &Status,
    editor: &LineEditor,
    transcript: &TranscriptRing,
    liveness: Liveness,
) -> Vec<Node> {
    let running = status.phase != Phase::Idle;
    let live = liveness == Liveness::Live;
    let mut nodes = vec![
        // `Log` rather than `Text`: the value is a rolling tail, so what an
        // agent reads is a suffix of the conversation and not the whole of
        // it, and the role is what says so.
        Node::new("transcript", Role::Log)
            .label("Conversation, most recent first dropped")
            .value(transcript.text()),
        // `Status`, not `ProgressBar`: this says what is running, never how
        // much of it is left. The fraction, when there is one, is `progress`.
        Node::new("phase", Role::Status)
            .label("What the runtime is doing")
            .value(phase_word(status.phase)),
    ];
    if let Some(model) = status.model.as_deref() {
        nodes.push(Node::new("model", Role::Text).label("Model").value(model));
    }
    if let Some(progress) = progress_value(status) {
        nodes.push(
            Node::new("progress", Role::ProgressBar)
                .label("Prefill progress")
                .value(progress),
        );
    }
    // `Chart`: the panel's right cluster is a row of numbers whose rendering
    // carries nothing an agent can use, so the numbers themselves are the
    // value.
    nodes.push(
        Node::new("stats", Role::Chart)
            .label("Throughput, context and cache")
            .value(stats_value(status)),
    );
    if let Some(detail) = status.detail.as_deref() {
        nodes.push(
            Node::new("detail", Role::Status)
                .label("What the panel is saying")
                .value(detail),
        );
    }
    nodes.push(input_node(editor, live));
    // Exactly one advertised way out, whatever the phase: stop a turn that is
    // running, leave when there is nothing to stop. That is the same rule the
    // keyboard follows — Ctrl-C interrupts while busy and exits while idle —
    // so the tree and the key handler cannot disagree about what is possible.
    // Winding down there is no way out to offer: the loop is already leaving
    // and the input it drains on the way is acked `Ignored`.
    let out = if running { "stop" } else { "quit" };
    let out = Node::new(out, Role::Button).label(out);
    nodes.push(if live {
        out.action(Action::Activate)
    } else {
        out
    });
    nodes
}

/// The input line: always the focused node, because the panel's keyboard has
/// nowhere else to be.
///
/// Focused whether or not the session is still taking input — the keyboard
/// really is there, and the human's Ctrl-C still works — but it advertises
/// nothing once the session is winding down, because nothing it could
/// advertise would be applied.
fn input_node(editor: &LineEditor, live: bool) -> Node {
    let node = Node::new("input", Role::TextInput)
        .label("Ask ramvamp anything")
        .value(editor.text())
        .focused(true);
    if !live {
        return node;
    }
    // `Focus` is what an agent acts on to aim `type_text` at a surface, so it
    // is advertised even though this app has only the one: without it the tree
    // never says where typed text lands.
    let mut node = node.actions([Action::Focus, Action::SetValue]);
    // `Activate` only when the draft would actually submit. Enter on a blank
    // buffer reprompts and disturbs nothing (see `Harness::handle_key`), and
    // the condition here is that handler's own, whitespace included, so the
    // two cannot drift into advertising a no-op.
    if !editor.text().trim().is_empty() {
        node = node.action(Action::Activate);
    }
    node
}

/// What the phase is, in the agent's words rather than the panel's: the panel
/// is fitting a word into a column and this is not.
fn phase_word(phase: Phase) -> &'static str {
    match phase {
        Phase::Idle => "ready",
        Phase::Prefill => "prefilling",
        Phase::Decode => "decoding",
    }
}

/// The prefill bar as one line of facts, or `None` when there is no fraction
/// to report.
///
/// Gated on a *known* total for the reason the panel draws no bar without
/// one: a prefill whose length nobody measured has no percentage and no ETA,
/// and inventing an indeterminate one implies progress that was never
/// measured.
fn progress_value(status: &Status) -> Option<String> {
    if status.phase != Phase::Prefill {
        return None;
    }
    let prefill = status.prefill?;
    let total = prefill.total?;
    let mut parts: Vec<String> = Vec::new();
    if let Some((chunk, chunks)) = status.chunk {
        parts.push(format!("chunk {chunk}/{chunks}"));
    }
    if let Some(share) = percent(prefill.done as u64, total as u64) {
        parts.push(format!("{share}%"));
    }
    if let Some(eta) = status.eta {
        parts.push(format!("~{} left", clock(eta)));
    }
    (!parts.is_empty()).then(|| parts.join(SEPARATOR))
}

/// Every number the panel's right cluster carries, rounded to what an agent
/// could act on.
///
/// The elapsed clock is deliberately not among them. It changes every second
/// whatever else does, so publishing it would put a new tree on the wire on
/// every tick of a six-minute prefill, and no agent can do anything with the
/// second it lands on.
fn stats_value(status: &Status) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(rate) = status.rate {
        parts.push(format!("{rate:.1} tok/s"));
    }
    if status.tokens > 0 {
        parts.push(format!("{} tok", status.tokens));
    }
    let (used, cap) = status.context;
    parts.push(format!("ctx {used}/{cap}"));
    // Nothing said yet: `0%` is noise next to `0/4096`, the same judgement
    // the panel's context segment makes.
    if let Some(share) = percent(used as u64, cap as u64).filter(|_| used > 0) {
        parts.push(format!("{share}%"));
    }
    if let Some(hit) = status.hit_rate {
        parts.push(format!(
            "hit {:.0}%",
            f64::from(hit.clamp(0.0, 1.0)) * 100.0
        ));
    }
    // The bytes this turn has streamed from NVMe, in the panel's own words.
    // It is the number this whole project is about — a 30B model in three
    // gigabytes of RAM is bought with reads — and until now it was the one
    // figure on the panel an agent could not reach at all. One decimal of a
    // binary unit is both what the detail row shows and about as fine as a
    // value republished on every token can afford to be.
    if let Some(bytes) = status.read_bytes {
        parts.push(format!("{} read", human_bytes(bytes)));
    }
    parts.join(SEPARATOR)
}

// ---------------------------------------------------------------------------
// agent input
// ---------------------------------------------------------------------------

/// What one agent input becomes, said in the only two things this app takes
/// from a terminal: a key press and a bracketed paste.
///
/// # Why the vocabulary is the safety argument
///
/// Every variant is something a person sitting at this keyboard can already
/// do. [`Key`](Self::Key) goes to
/// [`Harness::handle_key`](super::Harness::handle_key), which is the single
/// key dispatcher in the whole TUI; [`Paste`](Self::Paste) goes to
/// [`LineEditor::insert_paste`], which is where a bracketed paste already
/// lands; [`Submit`](Self::Submit) is Enter through that same handler. There
/// is no variant that reaches the editor, the worker or the session directly,
/// and none that names a command.
///
/// So "an agent cannot reach a state a keyboard cannot" is not a rule this
/// module remembers to follow. It is a property of the type: the only way to
/// break it is to add a variant that is neither a keystroke nor a paste, which
/// is a thing a reviewer can see in the enum rather than a thing they have to
/// find by reading every arm of [`plan`].
///
/// [`Sequence`](Self::Sequence) is composition, not a sixth capability. One
/// input needs more than one keyboard action to carry out — a `set_value`,
/// which is the line cleared and then pasted over — and a sequence of
/// keystrokes is still only keystrokes. It never nests more than one level
/// deep, and [`MAX_PLAN_STEPS`] bounds its length: a step here can be an
/// Enter, an Enter is a model turn, and an input whose step count grows with
/// its own length is an input that can queue more work than anyone can
/// interrupt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Plan {
    /// Feed through the key handler, exactly as crossterm would deliver it.
    Key(KeyCode, KeyModifiers),
    /// Insert through the bracketed-paste path, verbatim.
    Paste(String),
    /// A paste and then Enter: the typed text, submitted as one turn.
    PasteThenSubmit(String),
    /// Enter through the key handler.
    Submit,
    /// Do nothing at all. The ack beside it says whether that is because
    /// there was nothing to do (`Ignored`) or because it was already done
    /// (`Delivered`, for a focus move onto the only focusable thing here).
    Nothing,
    /// Several of the above, in order.
    Sequence(Vec<Plan>),
}

/// The most steps one agent input may expand into.
///
/// Defence in depth rather than a limit anything reaches: the one site that
/// builds a sequence builds two steps. What the cap stops is a future routing
/// change reintroducing an expansion that scales with the input — the shape of
/// the bug in which a single 4 KiB `type_text`, well inside the bridge's own
/// sanctioned limit, became two thousand `Submit`s on an unbounded channel and
/// two thousand queued model turns, run back to back without the keyboard
/// being read in between.
const MAX_PLAN_STEPS: usize = 4;

/// Decide what an agent input does, without doing any of it.
///
/// Returns the [`Plan`] to run and the [`InputStatus`] the input has earned.
/// `Delivered` means this app could act on it; `Ignored` means it looked and
/// deliberately did nothing, which is the answer an agent waiting on an effect
/// needs so it can stop waiting. A key that parses but is bound to nothing is
/// `Delivered` — the same verdict a person gets for pressing an unbound key.
///
/// Total by construction: every arm returns, nothing here can fail, and a node
/// id or an action this app does not know falls through to
/// `(Nothing, Ignored)`.
///
/// # The lint, and why `Unknown` has an arm of its own
///
/// [`AgentInput`] is `#[non_exhaustive]`, so a wildcard arm is mandatory; but
/// neither rustc nor default clippy can tell the arm the attribute demands
/// from one quietly swallowing a variant this app ought to handle. The
/// restriction lint below names every known variant a wildcard covers, so
/// deleting the `Text` arm fails the build instead of answering every
/// `type_text` with `Ignored`. It only sees the wildcard while `Unknown` is an
/// arm of its own: folded in as `Unknown | _` it goes quiet again, which is
/// the failure mode it exists to prevent.
#[warn(clippy::wildcard_enum_match_arm)]
pub(crate) fn plan(input: &AgentInput, phase: Phase, draft: &str) -> (Plan, InputStatus) {
    match input {
        AgentInput::Act {
            node,
            action,
            value,
            ..
        } => plan_act(node.0.as_str(), action, value.as_deref(), phase, draft),
        // The raw fallback, and the one input that is *supposed* to meet the
        // bindings: a key is a key press, and it lands wherever the keyboard
        // would have landed it.
        AgentInput::Key { key, .. } => match to_crossterm_key(key) {
            Some(key) => (Plan::Key(key.code, key.modifiers), InputStatus::Delivered),
            // The grammar rejected it, or it names a key this adapter cannot
            // lower. Either way there is no keystroke to deliver, and
            // approximating one would put a press into the app that nobody
            // asked for.
            None => (Plan::Nothing, InputStatus::Ignored),
        },
        AgentInput::Text { text, .. } => plan_text(text, draft),
        // The layer answers this variant itself and never hands it over. It is
        // named here only so the lint above can still see the wildcard.
        AgentInput::Unknown => (Plan::Nothing, InputStatus::Ignored),
        // An input kind taria learned after this was written. Answered the way
        // every other input this app cannot act on is answered, so the agent
        // hears "nothing happened" in one round trip rather than waiting out
        // the bridge's window.
        _ => (Plan::Nothing, InputStatus::Ignored),
    }
}

/// One semantic act, addressed by a node id from [`build_nodes`].
///
/// The conditions are the tree's own, on purpose: `input` advertises
/// `Activate` exactly when the draft would submit, `stop` exists only while a
/// turn is running and `quit` only while none is. An act on a node the tree is
/// not publishing, or an action it does not advertise there, is `Ignored`
/// rather than quietly no-op'd — the tree and the verdict have to agree, or
/// the advertisement is a lie.
fn plan_act(
    node: &str,
    action: &Action,
    value: Option<&str>,
    phase: Phase,
    draft: &str,
) -> (Plan, InputStatus) {
    let running = phase != Phase::Idle;
    match (node, action) {
        // Enter on a blank line reprompts and disturbs nothing, so it is not
        // advertised and not delivered: an agent that acted on it would be
        // waiting for a turn that is not coming.
        ("input", Action::Activate) if !draft.trim().is_empty() => {
            (Plan::Submit, InputStatus::Delivered)
        }
        // Ctrl-U then a paste, which is how a person replaces a draft. Not
        // `LineEditor::clear` and not an assignment: both would be this module
        // editing the buffer behind the key handler's back.
        ("input", Action::SetValue) => match value {
            Some(value) => {
                let mut steps = vec![Plan::Key(KeyCode::Char('u'), KeyModifiers::CONTROL)];
                if !value.is_empty() {
                    steps.push(Plan::Paste(value.to_owned()));
                }
                (sequence(steps), InputStatus::Delivered)
            }
            // `set_value` with nothing to set. Ignored rather than read as
            // "clear the line": an agent that meant to clear it can say so
            // with an empty string.
            None => (Plan::Nothing, InputStatus::Ignored),
        },
        // Delivered, with nothing to do. This panel has one focusable surface
        // and the keyboard is always on it, so the move an agent asked for is
        // already true — and `Focus` is what the tree advertises as the way to
        // aim `type_text`, so answering it `Ignored` would tell an agent that
        // aiming failed right before its text lands correctly.
        ("input", Action::Focus) => (Plan::Nothing, InputStatus::Delivered),
        // Esc rather than Ctrl-C: while a turn runs Esc is unambiguously
        // "stop this", where Ctrl-C means stop or leave depending on the
        // phase, and a `stop` that could leave the session is not a stop.
        ("stop", Action::Activate) if running => (
            Plan::Key(KeyCode::Esc, KeyModifiers::NONE),
            InputStatus::Delivered,
        ),
        // And Ctrl-C while idle, which is the keyboard's own way out.
        ("quit", Action::Activate) if !running => (
            Plan::Key(KeyCode::Char('c'), KeyModifiers::CONTROL),
            InputStatus::Delivered,
        ),
        _ => (Plan::Nothing, InputStatus::Ignored),
    }
}

/// Type literal characters into the prompt buffer.
///
/// # Text goes to the buffer, never to the bindings
///
/// This app's typing surface is the prompt line, so typed text is pasted into
/// it and never lowered into
/// [`Harness::handle_key`](super::Harness::handle_key) one character at a
/// time.
///
/// Every binding here is Ctrl-qualified, so lowering text through the key
/// handler would *mostly* work today — which is what makes it a trap rather
/// than a shortcut. It survives only until a plain character is bound to
/// something, at which point an agent's prose starts pressing it: taria's own
/// demo lost a task to a `type_text` whose characters were read as "delete"
/// and "yes". It is also already wrong for what the key handler does not
/// normalise — a stray control character in an agent's text would be inserted
/// into the buffer as itself, where the paste path drops it.
///
/// # One `type_text` is at most one turn
///
/// Only a *trailing* newline submits. Interior newlines stay in the buffer as
/// themselves, so `type_text("a\nb\n")` pastes `a\nb` and submits it as a
/// single turn — exactly what a person pasting that text and pressing Enter
/// gets — and `type_text("a\nb")` pastes it and leaves it sitting there.
///
/// This is not a retreat from taria's `'\n'`-is-Enter convention, it is the
/// case the convention carves out. `AgentInput::Text` documents the lowering
/// as "a convention for adapters and not a rule this crate can enforce: an app
/// whose typing surface is not made of key events reads the characters itself
/// and decides there what a newline or a tab means". This app's typing surface
/// is a `String`, not a stream of key events, and it decided for its human
/// users already: `insert_paste` keeps newlines so a pasted paragraph is one
/// turn, which is the fix recorded in `input.rs` and pinned by
/// `a_multi_line_paste_is_one_turn` there. Agent text now gets the same
/// answer, so `type_text`, `set_value` and a human paste all keep interior
/// newlines literal, and there is no asymmetry left between them.
///
/// The alternative was measured and is the reason this changed. An Enter per
/// newline made `type_text` of `"a\n"` repeated 2,048 times — 4,096
/// characters, inside the limit the bridge itself sanctions — into 2,048
/// `Submit`s, 2,048 `Command::Submit`s on an unbounded channel, and days of
/// queued prefill and decode that the session ran through without reading the
/// keyboard in between. No key press can produce more than one turn, so no
/// agent input may either; that property is worth more than a newline
/// convention this app was never obliged to follow.
///
/// # What is delivered and what is not
///
/// Everything else is the paste path's: CRLF and bare CR become LF (so a
/// trailing one submits like LF), a tab becomes a space, and other C0 controls
/// are dropped. Text that is nothing but those is `Ignored` — there is no
/// keystroke left in it to deliver — and so is text whose only effect would be
/// an Enter on a buffer that is blank once the text is in it, because
/// `handle_key` clears such a buffer, reprompts and starts no turn. That is
/// the very condition `input`'s `Activate` is gated on, on purpose: the same
/// non-effect has to earn the same verdict whichever of the two ways an agent
/// asks for it.
fn plan_text(text: &str, draft: &str) -> (Plan, InputStatus) {
    let text = normalise_paste(text);
    if text.is_empty() {
        return (Plan::Nothing, InputStatus::Ignored);
    }
    // The one newline that is Enter, if the text carries it at all. Every
    // other newline is a character being typed.
    let Some(typed) = text.strip_suffix('\n') else {
        return (Plan::Paste(text), InputStatus::Delivered);
    };
    // Enter on a whitespace-only buffer reprompts and disturbs nothing, and
    // the buffer this would submit is the draft with `typed` somewhere in it:
    // blank exactly when both are. Ignored rather than delivered, so an agent
    // is not left waiting on a turn that never starts.
    if draft.trim().is_empty() && typed.trim().is_empty() {
        return (Plan::Nothing, InputStatus::Ignored);
    }
    if typed.is_empty() {
        return (Plan::Submit, InputStatus::Delivered);
    }
    (
        Plan::PasteThenSubmit(typed.to_owned()),
        InputStatus::Delivered,
    )
}

/// Fold a list of steps into the smallest [`Plan`] that expresses it, so that
/// a one-step plan has exactly one spelling.
///
/// Truncated to [`MAX_PLAN_STEPS`]. Nothing here builds a longer one; the
/// truncation is what keeps that true of whatever is written next.
fn sequence(mut steps: Vec<Plan>) -> Plan {
    steps.truncate(MAX_PLAN_STEPS);
    match steps.len() {
        0 => Plan::Nothing,
        1 => steps.remove(0),
        _ => Plan::Sequence(steps),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use taria_ratatui::taria::NodeId;

    use super::*;
    use crate::tui::Prefilling;

    // -- fixtures ------------------------------------------------------------
    //
    // The same three canonical states `panel.rs` takes its snapshots under, so
    // a value that appears in both places can be read against the row it came
    // from.

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
    /// v0 manages.
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

    fn editor(text: &str) -> LineEditor {
        let mut editor = LineEditor::default();
        editor.insert_str(text);
        editor
    }

    fn nodes(status: &Status, draft: &str) -> Vec<Node> {
        build_nodes(
            status,
            &editor(draft),
            &TranscriptRing::default(),
            Liveness::Live,
        )
    }

    fn winding_down(status: &Status, draft: &str) -> Vec<Node> {
        build_nodes(
            status,
            &editor(draft),
            &TranscriptRing::default(),
            Liveness::WindingDown,
        )
    }

    fn find<'a>(nodes: &'a [Node], id: &str) -> Option<&'a Node> {
        nodes.iter().find(|node| node.id.0 == id)
    }

    fn ids(nodes: &[Node]) -> Vec<&str> {
        nodes.iter().map(|node| node.id.0.as_str()).collect()
    }

    fn value<'a>(nodes: &'a [Node], id: &str) -> &'a str {
        find(nodes, id)
            .and_then(|node| node.value.as_deref())
            .unwrap_or_else(|| panic!("no {id} node with a value"))
    }

    fn states() -> [(&'static str, Status); 3] {
        [
            ("idle", idle_status()),
            ("prefill", prefill_status()),
            ("decode", decode_status()),
        ]
    }

    // -- the tree ------------------------------------------------------------

    /// A raw key lands on exactly one place, so exactly one node may claim it.
    /// Two focused nodes tell an agent that a keystroke could go to either,
    /// and none tells it the app is not listening — in this panel the input
    /// line is always listening, including while a turn runs.
    #[test]
    fn exactly_one_node_is_focused_in_every_phase() {
        for (name, status) in states() {
            for draft in ["", "why is prefill slow?"] {
                let nodes = nodes(&status, draft);
                let focused: Vec<&str> = nodes
                    .iter()
                    .filter(|node| node.focused)
                    .map(|node| node.id.0.as_str())
                    .collect();
                assert_eq!(focused, ["input"], "{name} with draft {draft:?}");
                // The nodes are returned flat; the adapter's `app` root is the
                // only parent and it is not ours to build.
                assert!(
                    nodes.iter().all(|node| node.children.is_empty()),
                    "{name}: the tree must stay flat"
                );
            }
        }
    }

    /// Enter on a blank buffer reprompts and changes nothing, so advertising
    /// `activate` there would be an action that does nothing — the one thing
    /// the tree is never allowed to claim, because an agent that acts on it
    /// waits for an effect that is not coming.
    #[test]
    fn the_input_advertises_activate_only_when_the_draft_would_submit() {
        for (name, status) in states() {
            for blank in ["", "   ", "\t", "\n", " \n\t "] {
                let nodes = nodes(&status, blank);
                let input = find(&nodes, "input").expect("an input node");
                assert_eq!(
                    input.actions,
                    vec![Action::Focus, Action::SetValue],
                    "{name} with a blank draft {blank:?}"
                );
            }
            let nodes = nodes(&status, "  hello  ");
            let input = find(&nodes, "input").expect("an input node");
            assert_eq!(
                input.actions,
                vec![Action::Focus, Action::SetValue, Action::Activate],
                "{name} with a draft that would submit"
            );
            // And the draft is published as typed, not as trimmed: an agent
            // reading it back has to see what the editor holds.
            assert_eq!(input.value.as_deref(), Some("  hello  "), "{name}");
        }
    }

    /// A state an agent can enter and cannot leave by anything the tree
    /// advertises is a trap, and the only escape left is the raw-key
    /// fallback. One way out per phase, and only one, so there is never a
    /// question which node ends the thing that is running.
    #[test]
    fn every_phase_advertises_exactly_one_way_out() {
        for (name, status) in states() {
            let nodes = nodes(&status, "");
            let actionable: Vec<&str> = nodes
                .iter()
                .filter(|node| !node.actions.is_empty() && node.id.0 != "input")
                .map(|node| node.id.0.as_str())
                .collect();
            let expected = if status.phase == Phase::Idle {
                "quit"
            } else {
                "stop"
            };
            assert_eq!(actionable, [expected], "{name}");
            let out = find(&nodes, expected).expect("a way out");
            assert_eq!(out.actions, vec![Action::Activate], "{name}");
            assert_eq!(out.label.as_deref(), Some(expected), "{name}");
            // The other one is absent rather than present and inert.
            let absent = if expected == "quit" { "stop" } else { "quit" };
            assert!(find(&nodes, absent).is_none(), "{name} published {absent}");
        }
    }

    /// Winding down, the loop publishes but applies nothing: every input it
    /// drains is acked `Ignored` where it stands. So the tree must advertise
    /// nothing at all — an `activate` offered through that window is an
    /// action that cannot work, and the window is as long as an in-flight
    /// prefill, which cannot be cut short. The values keep moving, because an
    /// agent watching a session end still has to be able to read it.
    #[test]
    fn the_wind_down_tree_advertises_nothing_anywhere() {
        for (name, status) in states() {
            for draft in ["", "   ", "why is prefill slow?"] {
                let live = nodes(&status, draft);
                let winding = winding_down(&status, draft);
                for node in &winding {
                    assert!(
                        node.actions.is_empty(),
                        "{name} with draft {draft:?}: {} still offers {:?}",
                        node.id.0,
                        node.actions
                    );
                }
                // Same ids, same roles, same values: only the offers are gone.
                assert_eq!(ids(&winding), ids(&live), "{name} with draft {draft:?}");
                for (winding, live) in winding.iter().zip(live.iter()) {
                    assert_eq!(winding.role, live.role, "{name}");
                    assert_eq!(winding.value, live.value, "{name}");
                    assert_eq!(winding.focused, live.focused, "{name}");
                }
            }
        }
    }

    /// An agent holds a node id across calls: it reads the tree, decides, and
    /// acts on an id from that read. A turn starting in between must not
    /// change what an id names, so every id that survives a phase change
    /// names the same thing on the other side of it.
    #[test]
    fn node_ids_are_stable_across_phases() {
        assert_eq!(
            ids(&nodes(&idle_status(), "hi")),
            ["transcript", "phase", "model", "stats", "input", "quit"]
        );
        assert_eq!(
            ids(&nodes(&prefill_status(), "hi")),
            [
                "transcript",
                "phase",
                "model",
                "progress",
                "stats",
                "input",
                "stop"
            ]
        );
        assert_eq!(
            ids(&nodes(&decode_status(), "hi")),
            ["transcript", "phase", "model", "stats", "input", "stop"]
        );
        // Every id is unique, and the roles a shared id carries never change
        // with the phase: `stats` is a chart whatever is running.
        for (name, status) in states() {
            let nodes = nodes(&status, "");
            let mut seen = ids(&nodes);
            seen.sort_unstable();
            let len = seen.len();
            seen.dedup();
            assert_eq!(seen.len(), len, "{name} published a duplicate id");
            assert_eq!(
                find(&nodes, "transcript").unwrap().role,
                Role::Log,
                "{name}"
            );
            assert_eq!(find(&nodes, "phase").unwrap().role, Role::Status, "{name}");
            assert_eq!(find(&nodes, "stats").unwrap().role, Role::Chart, "{name}");
            assert_eq!(
                find(&nodes, "input").unwrap().role,
                Role::TextInput,
                "{name}"
            );
        }
    }

    /// The numbers, spelled exactly as the panel's own rows spell them, so a
    /// transcript of an agent session and a screenshot of the panel cannot
    /// be read as describing different runs.
    #[test]
    fn the_values_carry_the_panel_numbers_at_agent_precision() {
        let decode = nodes(&decode_status(), "");
        assert_eq!(
            value(&decode, "stats"),
            "2.1 tok/s · 47 tok · ctx 1071/4096 · 26% · hit 87% · 3.4 GiB read"
        );
        assert_eq!(value(&decode, "phase"), "decoding");
        assert_eq!(value(&decode, "model"), "Qwen3-30B-A3B");

        let prefill = nodes(&prefill_status(), "");
        assert_eq!(
            value(&prefill, "progress"),
            "chunk 2/8 · 25% · ~4m 26s left"
        );
        assert_eq!(value(&prefill, "phase"), "prefilling");

        let idle = nodes(&idle_status(), "");
        assert_eq!(value(&idle, "stats"), "ctx 1024/4096 · 25% · hit 87%");
        assert_eq!(value(&idle, "phase"), "ready");
    }

    /// The tree dedups on equality, so a value that moves every frame
    /// republishes the whole tree every frame and buries the changes an agent
    /// came for. Nothing below the precision an agent could act on may reach
    /// a value — and the clock, which moves whatever else does, is not in the
    /// tree at all.
    #[test]
    fn quantized_values_hold_still_while_only_the_sub_precision_part_moves() {
        let base = decode_status();
        let mut jittered = base.clone();
        // A tenth of a percent of cache hit, a hundredth of a token per
        // second, three positions of context: none of it is actionable.
        jittered.rate = Some(47.0 / 22.0 + 0.009);
        jittered.hit_rate = Some(0.8749);
        jittered.elapsed = Duration::from_secs(9_999);
        // And a megabyte of expert reads, which at these rates lands several
        // times a second: a tenth of a binary unit is where it starts meaning
        // something.
        jittered.read_bytes = Some(3_650_722_202 + 1_000_000);
        assert_eq!(stats_value(&base), stats_value(&jittered));

        // A whole tok/s, a whole percent, another token, or a tenth of a
        // gibibyte of reads does move it.
        for moved in [
            Status {
                rate: Some(3.2),
                ..base.clone()
            },
            Status {
                hit_rate: Some(0.91),
                ..base.clone()
            },
            Status {
                tokens: 48,
                ..base.clone()
            },
            Status {
                read_bytes: Some(4_000_000_000),
                ..base.clone()
            },
        ] {
            assert_ne!(stats_value(&base), stats_value(&moved));
        }

        // The prefill bar is the other live value, and the second it lands on
        // is as far as it goes.
        let prefill = prefill_status();
        let mut later = prefill.clone();
        later.eta = Some(Duration::from_secs_f64(266.9));
        later.elapsed = Duration::from_secs(94);
        assert_eq!(progress_value(&prefill), progress_value(&later));

        // And no node anywhere quotes the elapsed clock, in any phase.
        for (name, status) in states() {
            for node in nodes(&status, "") {
                let value = node.value.unwrap_or_default();
                assert!(!value.contains("1m 33s"), "{name}: {value:?}");
                assert!(!value.contains("0m 22s"), "{name}: {value:?}");
            }
        }
    }

    /// A prefill whose length nobody measured has no percentage and no ETA,
    /// and the panel draws no bar for it. The tree says the same thing by
    /// leaving the node out: a `progress` node with nothing in it would claim
    /// a fraction that was never measured.
    #[test]
    fn the_progress_node_appears_only_while_a_measured_prefill_runs() {
        assert_eq!(progress_value(&idle_status()), None);
        assert_eq!(progress_value(&decode_status()), None);
        // Submitted, but the first chunk has not landed: the phase is prefill
        // and there is nothing to count yet.
        let warming = Status {
            phase: Phase::Prefill,
            ..Status::default()
        };
        assert_eq!(progress_value(&warming), None);
        // A prefill running against an unknown total.
        let unmeasured = Status {
            phase: Phase::Prefill,
            prefill: Some(Prefilling {
                done: 512,
                total: None,
            }),
            ..Status::default()
        };
        assert_eq!(progress_value(&unmeasured), None);
        assert!(find(&nodes(&warming, ""), "progress").is_none());
        assert!(progress_value(&prefill_status()).is_some());
    }

    /// The optional rows are optional in the tree too. A node published with
    /// nothing behind it tells an agent about something that is not there.
    #[test]
    fn a_row_the_panel_is_not_showing_has_no_node() {
        let bare = nodes(&Status::default(), "");
        assert!(find(&bare, "model").is_none(), "no model loaded yet");
        assert!(find(&bare, "detail").is_none(), "no transient line");
        // A fresh context claims no share, exactly as the panel's row does.
        assert_eq!(value(&bare, "stats"), "ctx 0/0");

        let loading = nodes(
            &Status {
                detail: Some("loading the model...".to_owned()),
                ..Status::default()
            },
            "",
        );
        assert_eq!(value(&loading, "detail"), "loading the model...");
        assert_eq!(find(&loading, "detail").unwrap().role, Role::Status);
    }

    // -- the ring ------------------------------------------------------------

    /// Tokens arrive mid-word, a few characters at a time. One entry per
    /// chunk would turn a single reply into a hundred lines of `ramvamp:`
    /// and leave no room in the ring for the turn before it.
    #[test]
    fn the_ring_coalesces_streamed_chunks_into_one_model_turn() {
        let mut ring = TranscriptRing::default();
        ring.push_user("why is prefill slow?");
        for chunk in ["Experts ", "stream ", "from ", "NVMe."] {
            ring.push_model(chunk);
        }
        ring.push_user("thanks");
        assert_eq!(
            ring.text(),
            "you: why is prefill slow?\nramvamp: Experts stream from NVMe.\nyou: thanks\n"
        );
        assert_eq!(ring.text().matches("ramvamp:").count(), 1);
    }

    /// Who said what, in the retained text itself: an agent reads one string
    /// and has to be able to tell the model's words from its own and from the
    /// panel's.
    #[test]
    fn every_turn_in_the_ring_names_its_speaker() {
        let mut ring = TranscriptRing::default();
        ring.push_system("ramvamp chat.");
        ring.push_user("hello");
        ring.push_model("hi");
        // A system notice closes an open reply rather than running into it.
        ring.push_system("[interrupted after 2 bytes; kept as the reply]");
        assert_eq!(
            ring.text(),
            "note: ramvamp chat.\nyou: hello\nramvamp: hi\n\
             note: [interrupted after 2 bytes; kept as the reply]\n"
        );
        // Every line the ring holds is either a speaker's or a continuation of
        // one, and no turn is ever left open at the end.
        assert!(ring.text().ends_with('\n'));
    }

    /// A multi-line turn is one turn: a paste keeps its shape, and the
    /// trailing newlines a reply ends on do not become blank lines that eat
    /// the budget.
    #[test]
    fn the_ring_keeps_a_multi_line_turn_whole() {
        let mut ring = TranscriptRing::default();
        ring.push_user("first\nsecond\n\n");
        ring.push_model("one\ntwo\n");
        ring.push_user("done");
        assert_eq!(
            ring.text(),
            "you: first\nsecond\nramvamp: one\ntwo\nyou: done\n"
        );
    }

    /// The ring is a tail, not a log: it has to stay under its cap whatever
    /// it is fed, and it has to cut at a line boundary so an agent never
    /// reads a fragment of a word as if it were the start of a turn.
    #[test]
    fn the_ring_trims_from_the_front_and_stays_under_the_cap() {
        let mut ring = TranscriptRing::default();
        for turn in 0..400 {
            ring.push_user(&format!("turn {turn} {}", "x".repeat(60)));
            ring.push_model(&format!("reply {turn} {}", "y".repeat(60)));
        }
        assert!(
            ring.text().len() <= TRANSCRIPT_CAP,
            "{} bytes",
            ring.text().len()
        );
        // The oldest turns are gone and the newest are there.
        assert!(!ring.text().contains("turn 0 "));
        assert!(ring.text().contains("turn 399 "));
        assert!(ring.text().contains("reply 399 "));
        // What is left starts at the start of a turn, never mid-word.
        let first = ring.text().lines().next().unwrap_or_default();
        assert!(
            first.starts_with("you: ") || first.starts_with("ramvamp: "),
            "{first:?}"
        );
    }

    /// A trim that cuts into a turn takes the line carrying the speaker mark
    /// with it, so the mark goes back on what is left. Without that, an agent
    /// reading the tail of a long reply opens on a paragraph with nobody's
    /// name on it and cannot tell the model's words from its own.
    #[test]
    fn a_trimmed_tail_still_says_who_is_speaking() {
        let mut ring = TranscriptRing::default();
        ring.push_user("why is prefill slow?");
        // One reply, many lines, far more than the ring can hold.
        for line in 0..400 {
            ring.push_model(&format!("paragraph {line} {}\n", "y".repeat(60)));
        }
        assert!(ring.text().len() <= TRANSCRIPT_CAP);
        let first = ring.text().lines().next().unwrap_or_default();
        assert!(first.starts_with(MODEL), "{first:?}");
        // And it is the reply's own continuation that was re-marked, not a
        // line invented from nowhere.
        assert!(first.contains("paragraph "), "{first:?}");

        // Every line the ring can hand back opens a turn or continues the one
        // the tail was re-marked with, whatever it is fed.
        let mut ring = TranscriptRing::default();
        for turn in 0..200 {
            ring.push_user(&format!("ask {turn}"));
            ring.push_model(&format!("a\nb\nc {turn} {}\n", "z".repeat(80)));
        }
        assert!(ring.text().len() <= TRANSCRIPT_CAP);
        let first = ring.text().lines().next().unwrap_or_default();
        assert!(
            turn_mark(first).is_some(),
            "the tail opens unattributed: {first:?}"
        );
    }

    /// One reply longer than the whole budget has no line boundary to cut at,
    /// so the fallback cut is on a character boundary. Cutting inside a
    /// character would leave a `String` that is not UTF-8, which is the one
    /// way this could panic.
    #[test]
    fn the_ring_cuts_a_runaway_line_on_a_character_boundary() {
        let mut ring = TranscriptRing::default();
        // Four bytes per character, streamed a character at a time, with no
        // newline anywhere in it.
        for _ in 0..(TRANSCRIPT_CAP / 2) {
            ring.push_model("🜁");
        }
        assert!(
            ring.text().len() <= TRANSCRIPT_CAP,
            "{} bytes",
            ring.text().len()
        );
        // Re-marked: the cut took the `ramvamp:` that opened the reply with
        // it, and a tail of unattributed characters is a tail an agent cannot
        // read.
        let tail = ring
            .text()
            .strip_prefix(MODEL)
            .unwrap_or_else(|| panic!("the tail lost its speaker: {:?}", ring.text()));
        assert!(tail.chars().all(|c| c == '🜁'));
        // Three-byte characters in whole lines take the line-boundary path
        // and must survive it just as intact.
        let mut ring = TranscriptRing::default();
        for turn in 0..500 {
            ring.push_user(&format!("{turn} {}", "日本語".repeat(20)));
        }
        assert!(ring.text().len() <= TRANSCRIPT_CAP);
        assert!(ring.text().starts_with("you: "));
        assert!(ring.text().ends_with('\n'));
    }

    /// Nothing the worker or the user can produce may upset the ring: an
    /// empty chunk, a turn that is only newlines, a cap-sized single push.
    #[test]
    fn the_ring_survives_degenerate_turns() {
        let mut ring = TranscriptRing::default();
        ring.push_model("");
        assert_eq!(ring.text(), "", "an empty chunk opens no turn");
        ring.push_user("");
        ring.push_system("");
        assert_eq!(ring.text(), "you: \nnote: \n");

        let mut ring = TranscriptRing::default();
        ring.push_user(&"z".repeat(TRANSCRIPT_CAP * 2));
        assert!(ring.text().len() <= TRANSCRIPT_CAP);

        let mut ring = TranscriptRing::default();
        ring.push_user("\n\n\n");
        assert_eq!(ring.text(), "you: \n");
    }

    /// The transcript reaches the agent through the tree, so the node has to
    /// carry the ring's tail and say that it is a tail.
    #[test]
    fn the_transcript_node_carries_the_rings_tail() {
        let mut ring = TranscriptRing::default();
        ring.push_user("hello");
        ring.push_model("hi there");
        let nodes = build_nodes(&idle_status(), &editor(""), &ring, Liveness::Live);
        let transcript = find(&nodes, "transcript").expect("a transcript node");
        assert_eq!(transcript.role, Role::Log);
        assert_eq!(transcript.value.as_deref(), Some(ring.text()));
        assert!(transcript.actions.is_empty(), "the transcript is read-only");
    }

    // -- agent input ---------------------------------------------------------

    fn act(node: &str, action: Action) -> AgentInput {
        AgentInput::act(NodeId(node.to_owned()), action, None)
    }

    fn act_with(node: &str, action: Action, value: &str) -> AgentInput {
        AgentInput::act(NodeId(node.to_owned()), action, Some(value.to_owned()))
    }

    /// The whole action vocabulary, so a test can ask what happens to the
    /// actions a node does *not* advertise as well as to the ones it does.
    fn every_action() -> Vec<Action> {
        vec![
            Action::Activate,
            Action::Focus,
            Action::Select,
            Action::Toggle,
            Action::Scroll,
            Action::SetValue,
            Action::Dismiss,
            Action::Custom("archive".to_owned()),
        ]
    }

    /// The nodes that carry numbers and prose and advertise nothing.
    const READ_ONLY: [&str; 6] = [
        "transcript",
        "phase",
        "model",
        "progress",
        "stats",
        "detail",
    ];

    /// A plan flattened into the steps it actually runs, so a test can assert
    /// over what an input can and cannot reach.
    fn steps(plan: &Plan) -> Vec<&Plan> {
        match plan {
            Plan::Sequence(inner) => inner.iter().flat_map(steps).collect(),
            other => vec![other],
        }
    }

    fn ctrl(c: char) -> Plan {
        Plan::Key(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    /// The act the tree advertises on the input line has to do the one thing
    /// the tree says it does, and nothing when the tree is not advertising it:
    /// Enter on a blank draft reprompts, so an agent acting on it would be
    /// waiting for a turn that never starts.
    #[test]
    fn activating_the_input_submits_only_a_draft_that_would_submit() {
        for (name, status) in states() {
            assert_eq!(
                plan(&act("input", Action::Activate), status.phase, "hello"),
                (Plan::Submit, InputStatus::Delivered),
                "{name}"
            );
            for blank in ["", "   ", "\t", "\n", " \n\t "] {
                assert_eq!(
                    plan(&act("input", Action::Activate), status.phase, blank),
                    (Plan::Nothing, InputStatus::Ignored),
                    "{name} with a blank draft {blank:?}"
                );
            }
        }
    }

    /// `set_value` replaces the draft the way a person does — Ctrl-U, then a
    /// paste — rather than by assigning to the buffer behind the key handler's
    /// back. A `set_value` carrying nothing to set is not a request to clear
    /// it; an agent that means that sends an empty string.
    #[test]
    fn set_value_replaces_the_draft_through_the_keyboards_own_clear() {
        assert_eq!(
            plan(
                &act_with("input", Action::SetValue, "why is prefill slow?"),
                Phase::Idle,
                "half a thought"
            ),
            (
                Plan::Sequence(vec![
                    ctrl('u'),
                    Plan::Paste("why is prefill slow?".to_owned())
                ]),
                InputStatus::Delivered
            )
        );
        // An empty value is the clear, and nothing but the clear.
        assert_eq!(
            plan(
                &act_with("input", Action::SetValue, ""),
                Phase::Idle,
                "throw this away"
            ),
            (ctrl('u'), InputStatus::Delivered)
        );
        assert_eq!(
            plan(&act("input", Action::SetValue), Phase::Idle, "keep this"),
            (Plan::Nothing, InputStatus::Ignored)
        );
    }

    /// Focus is delivered with nothing to do. The panel has one focusable
    /// surface and the keyboard never leaves it, so the move an agent asked
    /// for is already true — and since `Focus` is what the tree advertises as
    /// the way to aim `type_text`, answering it `Ignored` would report a
    /// failure immediately before the text lands correctly.
    #[test]
    fn focusing_the_input_is_delivered_because_the_keyboard_is_already_there() {
        for (name, status) in states() {
            assert_eq!(
                plan(&act("input", Action::Focus), status.phase, ""),
                (Plan::Nothing, InputStatus::Delivered),
                "{name}"
            );
        }
    }

    /// The way out of a running turn is Esc, which while a turn runs can only
    /// mean "stop this" — Ctrl-C there means stop *or* leave depending on the
    /// phase, and a stop that might end the session is not a stop. Idle there
    /// is nothing to stop, and the tree does not publish the node at all.
    #[test]
    fn stopping_is_esc_while_a_turn_runs_and_nothing_while_idle() {
        for phase in [Phase::Prefill, Phase::Decode] {
            assert_eq!(
                plan(&act("stop", Action::Activate), phase, ""),
                (
                    Plan::Key(KeyCode::Esc, KeyModifiers::NONE),
                    InputStatus::Delivered
                ),
                "{phase:?}"
            );
        }
        assert_eq!(
            plan(&act("stop", Action::Activate), Phase::Idle, ""),
            (Plan::Nothing, InputStatus::Ignored)
        );
    }

    /// And the way out of the session is Ctrl-C while idle, which is the
    /// keyboard's own exit. Mid-turn the same keystroke would interrupt rather
    /// than leave, so `quit` is not offered and not honoured there: an agent
    /// that wants out of a running turn stops it first.
    #[test]
    fn quitting_is_ctrl_c_while_idle_and_nothing_while_a_turn_runs() {
        assert_eq!(
            plan(&act("quit", Action::Activate), Phase::Idle, ""),
            (ctrl('c'), InputStatus::Delivered)
        );
        for phase in [Phase::Prefill, Phase::Decode] {
            assert_eq!(
                plan(&act("quit", Action::Activate), phase, ""),
                (Plan::Nothing, InputStatus::Ignored),
                "{phase:?}"
            );
        }
    }

    /// A node that exists and advertises nothing is not a node to act on.
    /// Answering `Delivered` there would tell an agent that acting on the
    /// transcript did something, and the next read would show it did not.
    #[test]
    fn an_act_on_a_node_that_advertises_nothing_is_ignored() {
        for (name, status) in states() {
            for node in READ_ONLY {
                for action in every_action() {
                    let input = act_with(node, action.clone(), "text");
                    assert_eq!(
                        plan(&input, status.phase, "hello"),
                        (Plan::Nothing, InputStatus::Ignored),
                        "{name}: {action:?} on {node}"
                    );
                }
            }
        }
    }

    /// An id this app never published is an agent working from a stale read or
    /// a guess, and either way there is nothing here to act on.
    #[test]
    fn an_act_on_an_unknown_node_is_ignored() {
        for node in ["", "prompt", "input ", "INPUT", "app", "task-3"] {
            for action in every_action() {
                assert_eq!(
                    plan(&act_with(node, action, "text"), Phase::Idle, "hello"),
                    (Plan::Nothing, InputStatus::Ignored),
                    "{node:?}"
                );
            }
        }
    }

    /// The raw fallback is the one input that is *supposed* to meet the
    /// bindings, so a key that parses is delivered wherever the keyboard would
    /// have landed it — including to a binding that does nothing, which is the
    /// verdict a person gets for pressing an unbound key. A key the shared
    /// grammar rejects has no keystroke to deliver, and inventing a near miss
    /// would press something nobody asked for.
    #[test]
    fn a_key_is_lowered_by_the_shared_grammar_or_ignored() {
        assert_eq!(
            plan(&AgentInput::key("ctrl+c"), Phase::Decode, ""),
            (ctrl('c'), InputStatus::Delivered)
        );
        assert_eq!(
            plan(&AgentInput::key("esc"), Phase::Decode, ""),
            (
                Plan::Key(KeyCode::Esc, KeyModifiers::NONE),
                InputStatus::Delivered
            )
        );
        // Bound to nothing here, and still delivered.
        assert_eq!(
            plan(&AgentInput::key("f5"), Phase::Idle, ""),
            (
                Plan::Key(KeyCode::F(5), KeyModifiers::NONE),
                InputStatus::Delivered
            )
        );
        for rejected in ["", "meta+x", "ctrl+", "not-a-key", "ctrl c"] {
            assert_eq!(
                plan(&AgentInput::key(rejected), Phase::Idle, ""),
                (Plan::Nothing, InputStatus::Ignored),
                "{rejected:?}"
            );
        }
    }

    /// Typed text goes to the prompt buffer through the paste path and never
    /// through the key handler one character at a time. Every binding here is
    /// Ctrl-qualified, so the key path would mostly work today and break the
    /// day a plain character is bound to something — taria's own demo lost a
    /// task to a `type_text` read as "delete" and "yes". The plan is where
    /// that is decided, so the plan is where it can be asserted.
    #[test]
    fn typed_text_reaches_the_prompt_buffer_and_never_the_bindings() {
        for text in ["dy", "cause", "ctrl+c", "aedu", "why is prefill slow?"] {
            let (plan, status) = plan(&AgentInput::text(text), Phase::Idle, "");
            assert_eq!(status, InputStatus::Delivered, "{text:?}");
            for step in steps(&plan) {
                assert!(
                    !matches!(step, Plan::Key(..)),
                    "{text:?} lowered {step:?} into the key handler"
                );
            }
            assert_eq!(plan, Plan::Paste(text.to_owned()), "{text:?}");
        }
    }

    /// Only a trailing newline submits, and it submits once.
    ///
    /// The interior ones are characters being typed, which is the answer
    /// `insert_paste` already gives a human paste — see
    /// `a_multi_line_paste_is_one_turn` in `input.rs` — and the answer
    /// `set_value` gives too, so the three ways text reaches this buffer agree
    /// about what a newline in the middle of it means. taria's
    /// `'\n'`-is-Enter lowering says in so many words that it is a convention
    /// for adapters rather than a rule, for an app whose typing surface is
    /// made of key events; this one's is a `String`.
    #[test]
    fn typed_text_submits_only_on_a_trailing_newline() {
        for (text, expected) in [
            ("hi", Plan::Paste("hi".to_owned())),
            ("hi\n", Plan::PasteThenSubmit("hi".to_owned())),
            // The interior newline is typed, not pressed: one paste, one
            // Enter, one turn — the same turn a person pasting this and
            // pressing Enter would send.
            ("a\nb\n", Plan::PasteThenSubmit("a\nb".to_owned())),
            // No trailing newline: typed and left in the buffer, exactly where
            // a person typing it would have left it.
            ("a\nb", Plan::Paste("a\nb".to_owned())),
            // A bare newline is a bare Enter — on a draft that would submit.
            ("\n", Plan::Submit),
        ] {
            let draft = if text == "\n" { "hello" } else { "" };
            assert_eq!(
                plan(&AgentInput::text(text), Phase::Idle, draft),
                (expected, InputStatus::Delivered),
                "{text:?}"
            );
        }
    }

    /// The blocker this rule exists for: one `type_text` inside the bridge's
    /// own 4,096-character limit used to expand to one `Submit` per newline,
    /// and every `Submit` is a full model turn queued on an unbounded channel
    /// and run without the keyboard being read in between. A keystroke cannot
    /// produce more than one turn; neither may an agent input.
    #[test]
    fn a_text_of_many_newlines_submits_at_most_once() {
        let submits = |plan: &Plan| {
            steps(plan)
                .iter()
                .filter(|step| matches!(step, Plan::PasteThenSubmit(_) | Plan::Submit))
                .count()
        };
        for text in [
            "a\n".repeat(2048),
            "\n".repeat(4096),
            "a\r\n".repeat(1365),
            format!("{}\n", "a\n".repeat(2048)),
        ] {
            for (name, status) in states() {
                for draft in ["", "hello"] {
                    let (plan, _) = plan(&AgentInput::text(&text), status.phase, draft);
                    assert!(
                        submits(&plan) <= 1,
                        "{name} with draft {draft:?}: {} submits from {} characters",
                        submits(&plan),
                        text.len()
                    );
                    assert!(steps(&plan).len() <= MAX_PLAN_STEPS, "{name}");
                }
            }
        }
    }

    /// `type_text("\n")` on a blank draft and `act(input, activate)` on the
    /// same draft are the same non-event: `handle_key` clears the buffer,
    /// reprompts and starts no turn. Two agent paths to one outcome must not
    /// return opposite verdicts, or an agent taking the `Delivered` one waits
    /// out a turn that is never coming.
    #[test]
    fn typed_text_and_activate_agree_about_a_draft_that_would_not_submit() {
        for (name, status) in states() {
            for blank in ["", "   ", "\t", " \t "] {
                let activated = plan(&act("input", Action::Activate), status.phase, blank);
                assert_eq!(
                    activated,
                    (Plan::Nothing, InputStatus::Ignored),
                    "{name} with draft {blank:?}"
                );
                for text in ["\n", "  \n", "\t\n"] {
                    assert_eq!(
                        plan(&AgentInput::text(text), status.phase, blank),
                        activated,
                        "{name}: {text:?} on draft {blank:?}"
                    );
                }
            }
            // And with something to submit, both deliver.
            assert_eq!(
                plan(&act("input", Action::Activate), status.phase, "hello"),
                (Plan::Submit, InputStatus::Delivered),
                "{name}"
            );
            assert_eq!(
                plan(&AgentInput::text("\n"), status.phase, "hello"),
                (Plan::Submit, InputStatus::Delivered),
                "{name}"
            );
            // Text that is only whitespace but is not submitting is still a
            // change to the buffer, so it is still delivered.
            assert_eq!(
                plan(&AgentInput::text("  "), status.phase, ""),
                (Plan::Paste("  ".to_owned()), InputStatus::Delivered),
                "{name}"
            );
        }
    }

    /// The cap is defence in depth, so it is asserted where it is enforced
    /// rather than only where it currently binds: nothing builds a long
    /// sequence today, and this is what makes that still true of whatever is
    /// written next.
    #[test]
    fn a_plan_never_runs_more_steps_than_the_cap() {
        let long = sequence(vec![Plan::Submit; 1000]);
        assert_eq!(steps(&long).len(), MAX_PLAN_STEPS);
        assert_eq!(sequence(Vec::new()), Plan::Nothing);
        assert_eq!(sequence(vec![Plan::Submit]), Plan::Submit);
    }

    /// Everything that is not a newline comes from the paste path, so agent
    /// text and a human paste normalise identically: CRLF and bare CR are line
    /// breaks, a tab is a space, and other C0 controls are dropped. Text that
    /// is nothing but those has no keystroke left in it to deliver.
    #[test]
    fn typed_text_is_normalised_by_the_paste_path_it_lands_on() {
        assert_eq!(
            plan(&AgentInput::text("a\r\nb\n"), Phase::Idle, ""),
            plan(&AgentInput::text("a\nb\n"), Phase::Idle, "")
        );
        assert_eq!(
            plan(&AgentInput::text("a\rb"), Phase::Idle, ""),
            plan(&AgentInput::text("a\nb"), Phase::Idle, "")
        );
        assert_eq!(
            plan(&AgentInput::text("a\tb"), Phase::Idle, ""),
            (Plan::Paste("a b".to_owned()), InputStatus::Delivered)
        );
        for empty in ["", "\x07", "\x1b\x00"] {
            assert_eq!(
                plan(&AgentInput::text(empty), Phase::Idle, ""),
                (Plan::Nothing, InputStatus::Ignored),
                "{empty:?}"
            );
        }
    }

    /// An input kind this build has never heard of is answered rather than
    /// dropped, so an agent hears "nothing happened" in one round trip instead
    /// of waiting out the bridge's window for an effect that cannot come.
    #[test]
    fn an_input_kind_this_build_does_not_know_is_ignored() {
        assert_eq!(
            plan(&AgentInput::Unknown, Phase::Decode, "hello"),
            (Plan::Nothing, InputStatus::Ignored)
        );
    }

    /// The invariant that keeps the tree honest: for every phase and every
    /// draft, an action the tree advertises is an action this router delivers,
    /// and an action it does not advertise is one the router ignores.
    ///
    /// Advertising an action that does nothing sends an agent off to wait for
    /// an effect that is not coming; honouring one that is not advertised puts
    /// the tree's own description of the app out of step with what the app
    /// does. Either way the read and the write sides have drifted, and this is
    /// the test that notices.
    #[test]
    fn every_advertised_action_is_delivered_and_every_other_one_is_ignored() {
        for (name, status) in states() {
            for draft in ["", "   ", "why is prefill slow?"] {
                let nodes = nodes(&status, draft);
                for node in &nodes {
                    for action in every_action() {
                        let advertised = node.actions.contains(&action);
                        // `set_value` is the only action that carries one, and
                        // without it the answer is Ignored whatever the tree
                        // says — which is its own row of the table.
                        let input = act_with(node.id.0.as_str(), action.clone(), "text");
                        let (_, earned) = plan(&input, status.phase, draft);
                        let expected = if advertised {
                            InputStatus::Delivered
                        } else {
                            InputStatus::Ignored
                        };
                        assert_eq!(
                            earned, expected,
                            "{name} with draft {draft:?}: {action:?} on {}",
                            node.id.0
                        );
                    }
                }
                // And the way out this phase is not offering is not a way out:
                // an agent holding an id from the other side of a phase change
                // is told so rather than quietly succeeding.
                let absent = if status.phase == Phase::Idle {
                    "stop"
                } else {
                    "quit"
                };
                assert!(find(&nodes, absent).is_none(), "{name} published {absent}");
                assert_eq!(
                    plan(&act(absent, Action::Activate), status.phase, draft),
                    (Plan::Nothing, InputStatus::Ignored),
                    "{name} honoured {absent}"
                );
            }
        }
    }

    /// `plan` is total: it is the first thing every agent input meets, it runs
    /// on the UI thread, and a panic there takes the terminal down with a
    /// session in it. Nothing an agent can send may do more than earn an
    /// `Ignored`.
    #[test]
    fn planning_survives_anything_an_agent_can_send() {
        let long = "x".repeat(100_000);
        let inputs = [
            AgentInput::text(long.as_str()),
            AgentInput::text("日本語\n🜁\n"),
            AgentInput::text("\n\n\n"),
            AgentInput::key(long.as_str()),
            AgentInput::key("🜁"),
            AgentInput::act(NodeId(long.clone()), Action::Activate, Some(long.clone())),
            AgentInput::act(
                NodeId("input".to_owned()),
                Action::Custom(long.clone()),
                None,
            ),
            AgentInput::act(
                NodeId("input".to_owned()),
                Action::SetValue,
                Some("🜁\n\r\t\x07".to_owned()),
            ),
            AgentInput::Unknown,
        ];
        for (name, status) in states() {
            for input in &inputs {
                for draft in ["", long.as_str(), "日本語"] {
                    let (plan, earned) = plan(input, status.phase, draft);
                    // Whatever it decided, it decided in this vocabulary.
                    for step in steps(&plan) {
                        assert!(
                            !matches!(step, Plan::Sequence(_)),
                            "{name}: a plan nested deeper than one level"
                        );
                    }
                    // And in a bounded number of steps, however long the input
                    // was: a step can be an Enter, and an Enter is a model
                    // turn nobody can take back.
                    assert!(
                        steps(&plan).len() <= MAX_PLAN_STEPS,
                        "{name}: {} steps from {input:?}",
                        steps(&plan).len()
                    );
                    if earned == InputStatus::Ignored {
                        assert_eq!(plan, Plan::Nothing, "{name}: {input:?}");
                    }
                }
            }
        }
    }
}
