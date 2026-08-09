//! The REPL's pure logic: input parsing, the transcript, the ChatML splice,
//! and the context arithmetic a turn is admitted by.
//!
//! Everything here is a total function over owned data. Nothing in this
//! module reads stdin, writes stdout, touches a `Model`, or knows what a
//! front end looks like — `chat` in `main.rs` is one caller, driving it from
//! a line-oriented prompt, and it is deliberately not the only one this could
//! have. A turn is planned ([`plan_turn`]) before the model is involved at
//! all, which is what makes the context policy testable without a 30B model
//! on the other end of it.
//!
//! The one exception is [`print_repl_help`], which writes to stderr; its text
//! is [`REPL_HELP`], so a caller that draws its own help does not have to
//! restate it.

use std::path::PathBuf;

use anyhow::bail;
use ramvamp_core::io::StreamStats;
use ramvamp_core::model::{ForwardState, StreamPhase};
use ramvamp_core::tokenizer::{ChatMessage, ContentSanitizer, Role, RvmpTokenizer};

/// The context window a run gets when nothing configures one: the v0 scope
/// cap of a single sequence at 4K (`docs/architecture.md`).
///
/// A *default*, not a constant the code may assume: everything below takes
/// the cap it is working against as an argument, because `--context` (and the
/// profile system that will set it) makes the window a runtime value. This is
/// only what `main.rs` falls back to when neither the flag nor
/// `RAMVAMP_CONTEXT` has anything to say.
pub(crate) const DEFAULT_CONTEXT: usize = 4096;

/// Every phase's streaming counters as of one instant, so a *span* of a run
/// can be reported out of a state that outlives it.
///
/// A [`ForwardState`]'s counters are cumulative from construction, and
/// [`ForwardState::reset`] keeps them on purpose — they describe the process,
/// not the sequence. `chat` builds one state for the whole session, so the
/// only way for a per-turn line to mean per-turn is to snapshot at the start
/// of the turn and subtract. [`StreamStats::since`] does the subtraction; this
/// carries it across both phases at once, because a report is per phase.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PhaseStats(pub(crate) [StreamStats; StreamPhase::ALL.len()]);

impl PhaseStats {
    /// The counters as they stand right now.
    pub(crate) fn take(state: &ForwardState) -> Self {
        let mut phases = [StreamStats::default(); StreamPhase::ALL.len()];
        for (slot, phase) in phases.iter_mut().zip(StreamPhase::ALL) {
            *slot = state.stream_stats_in(phase);
        }
        Self(phases)
    }

    /// What happened between `earlier` and this snapshot, phase by phase.
    pub(crate) fn since(&self, earlier: &Self) -> Self {
        let mut delta = *self;
        for (now, before) in delta.0.iter_mut().zip(&earlier.0) {
            *now = now.since(before);
        }
        delta
    }

    /// Each phase with its counters, in [`StreamPhase::ALL`] order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (StreamPhase, &StreamStats)> {
        StreamPhase::ALL.into_iter().zip(self.0.iter())
    }

    /// One phase's counters.
    pub(crate) fn phase(&self, want: StreamPhase) -> StreamStats {
        self.iter()
            .find(|(phase, _)| *phase == want)
            .map(|(_, stats)| *stats)
            .unwrap_or_default()
    }
}

/// One line of REPL input, already classified.
///
/// A line is a slash command only when it starts with a single `/`; `//`
/// escapes to a message whose first character is a slash, so there is no
/// input the REPL cannot send.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ReplInput {
    /// Whitespace only: reprompt, do not disturb the transcript.
    Blank,
    /// A user turn.
    Message(String),
    /// `/exit`, `/quit`, `/q`.
    Exit,
    /// `/reset`, `/clear`: back to the seed transcript.
    Reset,
    /// `/help`, `/h`, `/?`.
    Help,
    /// `/save <path>`.
    Save(PathBuf),
    /// Something slash-shaped that is not a command; the string is the
    /// message to show the user.
    BadCommand(String),
}

/// Classify one line of REPL input. Total: never fails, never panics.
pub(crate) fn parse_repl_input(line: &str) -> ReplInput {
    let text = line.trim();
    if text.is_empty() {
        return ReplInput::Blank;
    }
    if text.starts_with("//") {
        // The escape hatch: exactly one slash is dropped, so sending a
        // message that starts with `/` is always "type one more slash", and
        // `//exit` sends the text `/exit`.
        return ReplInput::Message(text[1..].to_owned());
    }
    let Some(command) = text.strip_prefix('/') else {
        return ReplInput::Message(text.to_owned());
    };
    let (name, rest) = match command.split_once(char::is_whitespace) {
        Some((name, rest)) => (name, rest.trim()),
        None => (command, ""),
    };
    match name {
        "exit" | "quit" | "q" => ReplInput::Exit,
        "reset" | "clear" => ReplInput::Reset,
        "help" | "h" | "?" => ReplInput::Help,
        "save" if !rest.is_empty() => ReplInput::Save(PathBuf::from(rest)),
        "save" => ReplInput::BadCommand("/save needs a path, e.g. /save chat.json".to_owned()),
        "" => ReplInput::BadCommand(
            "a bare / is not a command; start a message with // to send a literal slash".to_owned(),
        ),
        other => ReplInput::BadCommand(format!(
            "unknown command /{other}; /help lists them, // sends a literal slash"
        )),
    }
}

/// The conversation the REPL is holding.
///
/// **Every message stored here is sanitized on the way in**, user and
/// assistant alike ([`ContentSanitizer`]). Sanitizing the assistant turns is
/// not belt and braces: the stop set is only `{<|im_end|>, <|endoftext|>}`
/// and the stream decoder decodes with `skip_special_tokens = false`, so a
/// model that emits any other added token (151646-151668 for the pinned
/// vocabulary) puts that literal into the reply text — and the reply text is
/// re-encoded as context on the next turn, where it would become a real
/// control id. Storing the sanitized form also means `/save` writes exactly
/// what the model was shown.
///
/// Sanitizing is idempotent, so rendering through
/// [`RvmpTokenizer::encode_chat_sanitized`] on top of this is free and keeps
/// the guarantee independent of any one caller remembering it.
pub(crate) struct Transcript {
    /// What `/reset` restores: the `--system` prompt and any
    /// `--messages-file` seed, sanitized once at startup.
    seed: Vec<ChatMessage>,
    /// The live conversation, seed included.
    messages: Vec<ChatMessage>,
}

impl Transcript {
    /// A transcript seeded with `seed` (which the caller has sanitized).
    pub(crate) fn new(seed: Vec<ChatMessage>) -> Self {
        Transcript {
            messages: seed.clone(),
            seed,
        }
    }

    /// The conversation so far.
    pub(crate) fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Turns on top of the seed.
    pub(crate) fn live_turns(&self) -> usize {
        self.messages.len() - self.seed.len()
    }

    /// Drop everything back to the seed.
    pub(crate) fn reset(&mut self) {
        self.messages.clear();
        self.messages.extend_from_slice(&self.seed);
    }

    /// Append a turn, sanitizing its content.
    pub(crate) fn push(&mut self, sanitizer: &ContentSanitizer, role: Role, content: &str) {
        self.messages
            .push(ChatMessage::new(role, sanitizer.sanitize(content)));
    }

    /// Undo the last [`push`](Self::push), for a turn that was never sent.
    pub(crate) fn pop(&mut self) -> Option<ChatMessage> {
        if self.messages.len() > self.seed.len() {
            self.messages.pop()
        } else {
            None
        }
    }

    /// The transcript in the exact `--messages-file` JSON format.
    pub(crate) fn to_json(&self) -> serde_json::Result<String> {
        let mut json = serde_json::to_string_pretty(&self.messages)?;
        json.push('\n');
        Ok(json)
    }
}

/// The ChatML fragments the REPL splices onto its id history, taken from the
/// installed template rather than spelled out here.
///
/// # Why a splice and not a re-render
///
/// The KV cache holds the ids the model *generated*, and a re-render of the
/// finished turn holds a re-encoding of the reply's *text*. Those differ: the
/// `\n` that closes `<|im_start|>assistant\n` and the first characters of the
/// reply are candidates for the same BPE merge, so re-encoding can produce
/// ids the cache does not contain (`a_generation_prompt_is_not_always_a_token_
/// prefix_of_the_finished_turn`). An incremental REPL therefore extends the
/// cache with `generated_ids` and splices the *markers* around them.
///
/// The splice is exact because every fragment boundary here is an added
/// token, and the added-token trie runs before the BPE merges — so no merge
/// can span one, and encoding the pieces separately gives the ids the whole
/// render would have given (`a_spliced_turn_matches_the_whole_render`).
pub(crate) struct TurnCodec {
    /// `<|im_end|>\n`: closes the assistant turn the cache currently ends
    /// in. The model's own `<|im_end|>` is a stop token, so it is sampled but
    /// never fed — this is what the next turn feeds in its place, and it is
    /// also what closes a reply that stopped on `--max-new` instead.
    assistant_close: Vec<u32>,
}

impl TurnCodec {
    /// Derive the fragments from the tokenizer's own chat template.
    pub(crate) fn new(tokenizer: &RvmpTokenizer) -> anyhow::Result<Self> {
        // `<|im_start|>assistant\n` ...
        let opener = tokenizer.encode_chat_sanitized(&[], true)?;
        // ... and the same thing with an empty turn closed after it.
        let empty_turn =
            tokenizer.encode_chat_sanitized(&[ChatMessage::assistant(String::new())], false)?;
        let Some(close) = empty_turn.strip_prefix(opener.as_slice()) else {
            bail!(
                "chat template: an empty assistant turn ({} ids) does not start with the \
                 generation prompt ({} ids), so the REPL cannot close a turn incrementally",
                empty_turn.len(),
                opener.len(),
            );
        };
        Ok(Self {
            assistant_close: close.to_vec(),
        })
    }

    /// The ids that continue a cache already holding a reply: close the
    /// assistant turn, add the user's message, open the next reply.
    pub(crate) fn continue_with(
        &self,
        tokenizer: &RvmpTokenizer,
        message: &ChatMessage,
    ) -> anyhow::Result<Vec<u32>> {
        let mut ids = self.assistant_close.clone();
        ids.extend(tokenizer.encode_chat_sanitized(std::slice::from_ref(message), true)?);
        Ok(ids)
    }
}

/// Context accounting for one turn, against the `context_cap` this run was
/// configured with.
///
/// `Ok(room)` is how many positions are still free once `prompt_tokens` are
/// prefilled and `max_new` is reserved for the reply; `Err(total)` is what
/// the turn would have needed when that does not fit `context_cap`.
///
/// `max_new` is *reserved*, not merely hoped for: the KV cache is sized at
/// `context_cap` and a reply that reached the end of it would fail mid-token,
/// so a turn that could overrun is refused before it starts. The cap is
/// passed in rather than read from a constant because it is whatever
/// `--context` resolved to, and the caller is the only thing that knows.
pub(crate) fn context_room(
    prompt_tokens: usize,
    max_new: usize,
    context_cap: usize,
) -> Result<usize, usize> {
    match prompt_tokens.checked_add(max_new) {
        Some(total) if total <= context_cap => Ok(context_cap - total),
        Some(total) => Err(total),
        // Only reachable from absurd arguments; report it as "does not fit"
        // rather than wrapping into a number that says it does.
        None => Err(usize::MAX),
    }
}

/// The seed for turn `turn` of a session whose base seed is `base`.
///
/// `Sampler::new` re-seeds from `params.seed` on every `generate` call and
/// the RNG does not carry across calls, so a fixed seed makes every turn
/// replay the same random stream: ask the same question twice from the same
/// context — after `/reset`, or after interrupting a reply and retrying —
/// and the answer is identical, token for token. Advancing the seed by the
/// turn index keeps the whole session reproducible from `--seed` while
/// letting a repeated question be answered differently. Turn 0 uses the base
/// seed unchanged, so the first reply of a chat matches what `generate`
/// produces from the same prompt and `--seed`.
pub(crate) fn turn_seed(base: u64, turn: u64) -> u64 {
    base.wrapping_add(turn)
}

/// What [`plan_turn`] decided about a user message.
#[derive(Debug)]
pub(crate) enum TurnPlan {
    /// The turn fits. `new_ids` is what the model has *not* seen yet — the
    /// whole rendered transcript on a cold cache, the closing marker plus the
    /// user's turn on a warm one; `used` is what the conversation will occupy
    /// once they are prefilled, and `room` is what is left after the reply's
    /// reservation.
    Ready {
        new_ids: Vec<u32>,
        used: usize,
        room: usize,
    },
    /// The turn does not fit. The transcript is exactly as it was — the
    /// message is not stored, nothing older is dropped — and this is what to
    /// tell the user.
    Refused(String),
}

/// Everything a turn does before the model is involved: append the user
/// message, work out which ids the cache has not seen, and decide whether the
/// result plus `max_new` fits `context_cap`.
///
/// `history_len` is how many ids the conversation already stands at — zero
/// at startup and after `/reset`, when the whole transcript has to be
/// rendered, and the running id history otherwise, when only the new turn
/// does (see [`TurnCodec`]).
///
/// `context_cap` is what the run resolved `--context` to, and it is the same
/// number the KV cache was sized at; passing it means the refusal message
/// quotes the window the user actually configured rather than a compiled-in
/// 4096.
///
/// On refusal the appended message is rolled back, so a transcript that
/// has hit the cap is left in exactly the state `/save` should write. This
/// is the whole of the context policy: refuse, explain, change nothing.
/// Nothing here truncates, summarizes, or silently drops a turn.
pub(crate) fn plan_turn(
    tokenizer: &RvmpTokenizer,
    codec: &TurnCodec,
    transcript: &mut Transcript,
    history_len: usize,
    message: &str,
    max_new: usize,
    context_cap: usize,
) -> anyhow::Result<TurnPlan> {
    transcript.push(tokenizer.content_sanitizer(), Role::User, message);
    let new_ids = if history_len == 0 {
        tokenizer.encode_chat_sanitized(transcript.messages(), true)?
    } else {
        // The sanitized message the transcript just stored, so what is
        // encoded here is exactly what `/save` would write.
        let user = transcript
            .messages()
            .last()
            .expect("the user turn was just pushed")
            .clone();
        codec.continue_with(tokenizer, &user)?
    };
    let used = history_len + new_ids.len();
    match context_room(used, max_new, context_cap) {
        Ok(room) => Ok(TurnPlan::Ready {
            new_ids,
            used,
            room,
        }),
        Err(total) => {
            transcript.pop();
            Ok(TurnPlan::Refused(format!(
                "context: this turn needs {total} of {context_cap} tokens ({used} for the \
                 conversation + {max_new} reserved for the reply). Nothing was sent and \
                 your message was not added. Use /save <path> to keep this conversation, \
                 then /reset to start a new one — or restart with a smaller --max-new.",
            )))
        }
    }
}

/// The `/help` text: one line per command, already indented, no trailing
/// newline.
///
/// Held apart from [`print_repl_help`] because the text is the half that
/// travels — a front end that paints its own help panel needs the lines, not
/// a write to stderr — and there is to be exactly one copy of it.
pub(crate) const REPL_HELP: &str = "  /exit, /quit, /q     leave (Ctrl-D does the same)\n\
     \x20 /reset, /clear       forget the conversation, keep --system and any seed file\n\
     \x20 /save <path>         write the transcript as a --messages-file JSON array\n\
     \x20 /help, /h, /?        this list\n\
     \x20 //text               send a message starting with a literal slash\n\
     \x20 Ctrl-C               stop the reply in progress; at the prompt, exit";

/// The `/help` text, on stderr with the rest of the REPL's chrome.
pub(crate) fn print_repl_help() {
    eprintln!("{REPL_HELP}");
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn slash_commands_parse() {
        use ReplInput::*;
        for (line, want) in [
            ("", Blank),
            ("   \t \n", Blank),
            ("/exit", Exit),
            ("/quit\n", Exit),
            ("  /q  ", Exit),
            ("/reset", Reset),
            ("/clear", Reset),
            ("/help", Help),
            ("/h", Help),
            ("/?", Help),
            ("/save chat.json", Save(PathBuf::from("chat.json"))),
            (
                "/save   /tmp/a b.json  ",
                Save(PathBuf::from("/tmp/a b.json")),
            ),
            ("hello there", Message("hello there".to_owned())),
            ("  hello  ", Message("hello".to_owned())),
            // Not commands: a slash anywhere but the front.
            ("and/or", Message("and/or".to_owned())),
            (
                "what does /exit do",
                Message("what does /exit do".to_owned()),
            ),
        ] {
            assert_eq!(parse_repl_input(line), want, "{line:?}");
        }
    }

    /// Every possible line has to be sendable, including one that starts
    /// with a slash. Exactly one leading slash is dropped, so the escape is
    /// invertible: to send a message starting with `/`, type one more `/`.
    #[test]
    fn a_doubled_slash_sends_a_literal_slash() {
        for (line, want) in [("//exit", "/exit"), ("//", "/"), ("///x", "//x")] {
            assert_eq!(
                parse_repl_input(line),
                ReplInput::Message(want.to_owned()),
                "{line:?}"
            );
        }
        // Round trip: escaping any slash-leading message and parsing it back
        // yields the message.
        for message in ["/exit", "/save x", "//weird", "/"] {
            assert_eq!(
                parse_repl_input(&format!("/{message}")),
                ReplInput::Message(message.to_owned()),
            );
        }
    }

    /// A mistyped command must not be sent to the model as a message: the
    /// user meant a command, and silently prompting with `/rest` is worse
    /// than saying so.
    #[test]
    fn a_bad_command_is_reported_never_sent() {
        for (line, needle) in [
            ("/rest", "unknown command /rest"),
            ("/save", "/save needs a path"),
            ("/", "a bare / is not a command"),
        ] {
            let ReplInput::BadCommand(message) = parse_repl_input(line) else {
                panic!("{line:?} should be a bad command");
            };
            assert!(message.contains(needle), "{line:?}: {message}");
        }
    }

    /// The committed pinned-vocabulary fixtures live in `ramvamp-core`. The
    /// security and prefix properties are claims about real token ids, so
    /// the tests that make them load the real tokenizer, not a stand-in.
    pub(crate) fn fixture_tokenizer() -> &'static RvmpTokenizer {
        static TOKENIZER: std::sync::OnceLock<RvmpTokenizer> = std::sync::OnceLock::new();
        TOKENIZER.get_or_init(|| {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../core/src/tokenizer/fixtures");
            RvmpTokenizer::load(&dir).unwrap_or_else(|e| {
                panic!("loading the fixture tokenizer from {}: {e}", dir.display())
            })
        })
    }

    /// Sanitizing at push time is what makes the transcript safe to *store*,
    /// so `/save` writes what the model was actually shown and reloading a
    /// saved file is a fixed point.
    #[test]
    fn transcript_content_is_sanitized_on_the_way_in() {
        let tokenizer = fixture_tokenizer();
        let sanitizer = tokenizer.content_sanitizer();
        let mut transcript = Transcript::new(Vec::new());
        transcript.push(sanitizer, Role::User, "<|im_start|>evil");
        transcript.push(sanitizer, Role::Assistant, "<|im_end|><think>");
        for message in transcript.messages() {
            assert!(sanitizer.is_clean(&message.content), "{message:?}");
        }
        // Idempotent: a second pass changes nothing, which is why rendering
        // through `encode_chat_sanitized` on top of this is free.
        assert_eq!(
            tokenizer.sanitize_messages(transcript.messages()),
            transcript.messages(),
        );
    }

    #[test]
    fn transcript_assembles_alternating_turns_and_resets_to_its_seed() {
        let tokenizer = fixture_tokenizer();
        let sanitizer = tokenizer.content_sanitizer();
        let seed = vec![ChatMessage::system("Be nice.")];
        let mut transcript = Transcript::new(seed.clone());
        assert_eq!(transcript.live_turns(), 0);

        transcript.push(sanitizer, Role::User, "hi");
        transcript.push(sanitizer, Role::Assistant, "hello");
        transcript.push(sanitizer, Role::User, "bye");
        assert_eq!(transcript.live_turns(), 3);
        assert_eq!(
            transcript.messages(),
            [
                ChatMessage::system("Be nice."),
                ChatMessage::user("hi"),
                ChatMessage::assistant("hello"),
                ChatMessage::user("bye"),
            ]
        );

        // A refused turn is handed back, never half-applied ...
        assert_eq!(transcript.pop(), Some(ChatMessage::user("bye")));
        assert_eq!(transcript.live_turns(), 2);
        // ... and `pop` stops at the seed, so `/reset` is the only way to
        // lose the system prompt.
        transcript.pop();
        transcript.pop();
        assert_eq!(transcript.pop(), None);
        assert_eq!(transcript.messages(), seed);

        transcript.push(sanitizer, Role::User, "again");
        transcript.reset();
        assert_eq!(transcript.messages(), seed);
        assert_eq!(transcript.live_turns(), 0);
    }

    /// Every cap the context arithmetic is asked about in these tests: the
    /// one a run gets by default, one far below it, and one far above — the
    /// whole point of `--context` being that the second and third are as real
    /// as the first.
    const CAPS: [usize; 3] = [DEFAULT_CONTEXT, 512, 40_960];

    /// `--max-new` is reserved, not hoped for: the KV cache is sized at the
    /// configured context and a reply that ran into the end of it would fail
    /// mid-token, so a turn that could overrun is refused before it starts.
    ///
    /// Stated against every cap, because the cap is an argument now: an
    /// arithmetic that quietly kept using 4096 would still pass at the
    /// default and refuse nothing at 512.
    #[test]
    fn a_turn_that_could_overrun_the_context_is_refused_whole() {
        for cap in CAPS {
            assert_eq!(context_room(0, 0, cap), Ok(cap), "cap {cap}");
            assert_eq!(context_room(100, 128, cap), Ok(cap - 228), "cap {cap}");
            // Exactly full is allowed; one more is not.
            assert_eq!(context_room(cap - 128, 128, cap), Ok(0), "cap {cap}");
            assert_eq!(context_room(cap - 127, 128, cap), Err(cap + 1), "cap {cap}");
            assert_eq!(context_room(cap + 1, 0, cap), Err(cap + 1), "cap {cap}");
            // Absurd arguments report "does not fit" rather than wrapping into
            // a total that says they do.
            assert_eq!(
                context_room(usize::MAX, 1, cap),
                Err(usize::MAX),
                "cap {cap}"
            );
        }
    }

    /// The context policy, end to end: a turn that does not fit is refused
    /// whole and the transcript is left byte-identical, so nothing older is
    /// lost and the user's next move (`/save`, `/reset`) still has the
    /// complete conversation to work with.
    ///
    /// Run against every cap, so "does not fit" means "does not fit the
    /// window this run was configured with" rather than a compiled-in one.
    #[test]
    fn a_refused_turn_leaves_the_transcript_exactly_as_it_was() {
        for cap in CAPS {
            let tokenizer = fixture_tokenizer();
            let codec = TurnCodec::new(tokenizer).unwrap();
            let sanitizer = tokenizer.content_sanitizer();
            let mut transcript = Transcript::new(vec![ChatMessage::system("Be nice.")]);
            transcript.push(sanitizer, Role::User, "an earlier question");
            transcript.push(sanitizer, Role::Assistant, "an earlier answer");
            let before = transcript.messages().to_vec();

            // Fits, and on a cold cache the new ids are the whole conversation
            // plus the generation prompt.
            let TurnPlan::Ready {
                new_ids,
                used,
                room,
            } = plan_turn(
                tokenizer,
                &codec,
                &mut transcript,
                0,
                "and another",
                128,
                cap,
            )
            .unwrap()
            else {
                panic!("a short turn should fit cap {cap}");
            };
            assert_eq!(transcript.messages().len(), before.len() + 1);
            assert_eq!(
                transcript.messages().last(),
                Some(&ChatMessage::user("and another"))
            );
            assert_eq!(used, new_ids.len());
            assert_eq!(used + 128 + room, cap);
            assert_eq!(
                new_ids,
                tokenizer
                    .encode_chat_sanitized(transcript.messages(), true)
                    .unwrap(),
            );
            transcript.pop();
            assert_eq!(transcript.messages(), before);

            // Does not fit, because `--max-new` alone eats the window.
            let TurnPlan::Refused(reason) =
                plan_turn(tokenizer, &codec, &mut transcript, 0, "one more", cap, cap).unwrap()
            else {
                panic!("reserving the whole window should refuse every turn");
            };
            assert!(reason.contains("Nothing was sent"), "{reason}");
            assert!(reason.contains("/reset"), "{reason}");
            // The window it quotes is the one it was given, which is the one
            // the KV cache was actually sized at.
            assert!(
                reason.contains(&format!("of {cap} tokens")),
                "cap {cap}: {reason}"
            );
            assert_eq!(
                transcript.messages(),
                before,
                "a refused turn must not store the message or drop anything older"
            );
        }
    }

    /// A fixed seed makes every `generate` call replay the same random
    /// stream, so without this a question repeated from the same context —
    /// after `/reset`, or after interrupting a reply — is answered
    /// identically every time.
    #[test]
    fn each_turn_samples_from_its_own_seed() {
        assert_eq!(turn_seed(42, 0), 42, "turn 0 matches `generate --seed 42`");
        assert_eq!(turn_seed(42, 1), 43);
        assert_eq!(turn_seed(u64::MAX, 1), 0, "wraps rather than panicking");
    }

    // -----------------------------------------------------------------
    // the incremental REPL
    // -----------------------------------------------------------------

    /// The splice is the whole basis of the incremental REPL: closing an
    /// assistant turn and opening the next one has to produce the ids the
    /// full render would have produced, or the model sees a prompt nobody
    /// wrote.
    ///
    /// Restricted to replies that re-encode stably after a generation prompt,
    /// because for the rest the full render is the one that is wrong — that
    /// is exactly why the REPL splices `generated_ids` instead of re-encoding
    /// (`a_generation_prompt_is_not_always_a_token_prefix_of_the_finished_turn`).
    #[test]
    fn a_spliced_turn_matches_the_whole_render() {
        let tokenizer = fixture_tokenizer();
        let codec = TurnCodec::new(tokenizer).unwrap();
        let mut checked = 0;
        for system in [false, true] {
            let mut prefix: Vec<ChatMessage> = Vec::new();
            if system {
                prefix.push(ChatMessage::system("You are terse."));
            }
            prefix.push(ChatMessage::user("first question"));
            for reply in ["first answer", "42", "yes.", "a longer reply, with commas"] {
                for next in ["second question", "/help me", "  spaced  "] {
                    let prompted = tokenizer.encode_chat_sanitized(&prefix, true).unwrap();
                    let mut finished = prefix.clone();
                    finished.push(ChatMessage::assistant(reply));
                    let closed = tokenizer.encode_chat_sanitized(&finished, false).unwrap();
                    if !closed.starts_with(&prompted) {
                        continue;
                    }
                    checked += 1;

                    // What the cache holds: the prompt, then the ids the
                    // model produced (here, the stable re-encoding of them).
                    let generated =
                        &closed[prompted.len()..closed.len() - codec.assistant_close.len()];
                    let mut spliced = prompted.clone();
                    spliced.extend_from_slice(generated);
                    spliced.extend(
                        codec
                            .continue_with(tokenizer, &ChatMessage::user(next))
                            .unwrap(),
                    );

                    let mut whole = finished.clone();
                    whole.push(ChatMessage::user(next));
                    assert_eq!(
                        spliced,
                        tokenizer.encode_chat_sanitized(&whole, true).unwrap(),
                        "system={system} reply={reply:?} next={next:?}",
                    );
                }
            }
        }
        assert!(
            checked > 0,
            "no reply re-encoded stably, so nothing was compared"
        );
    }

    /// `<|im_end|>\n` and nothing else: the marker the model's own stop token
    /// stands for, which is sampled but never fed and so has to be supplied
    /// by the next turn.
    #[test]
    fn the_turn_codec_closes_a_turn_with_the_templates_own_marker() {
        use ramvamp_core::tokenizer::IM_END_TOKEN_ID;

        let tokenizer = fixture_tokenizer();
        let codec = TurnCodec::new(tokenizer).unwrap();
        assert_eq!(codec.assistant_close.first(), Some(&IM_END_TOKEN_ID));
        assert_eq!(
            tokenizer.decode(&codec.assistant_close, false).unwrap(),
            "<|im_end|>\n",
        );
    }

    /// The second turn encodes the new message and the markers around it —
    /// not the whole conversation again, which is what made the REPL
    /// quadratic.
    #[test]
    fn a_warm_turn_only_encodes_what_the_cache_has_not_seen() {
        let tokenizer = fixture_tokenizer();
        let codec = TurnCodec::new(tokenizer).unwrap();
        let sanitizer = tokenizer.content_sanitizer();
        let mut transcript = Transcript::new(vec![ChatMessage::system("Be nice.")]);

        // Turn 0, cold: the whole transcript.
        let TurnPlan::Ready { new_ids: cold, .. } = plan_turn(
            tokenizer,
            &codec,
            &mut transcript,
            0,
            "hi",
            128,
            DEFAULT_CONTEXT,
        )
        .unwrap() else {
            panic!("a short turn should fit");
        };
        assert_eq!(
            cold,
            tokenizer
                .encode_chat_sanitized(transcript.messages(), true)
                .unwrap()
        );

        // Turn 1, warm: the closer plus the new user turn, and it must be
        // strictly shorter than re-rendering everything.
        transcript.push(sanitizer, Role::Assistant, "hello");
        let history_len = cold.len() + 3;
        let TurnPlan::Ready {
            new_ids: warm,
            used,
            room,
        } = plan_turn(
            tokenizer,
            &codec,
            &mut transcript,
            history_len,
            "again",
            128,
            DEFAULT_CONTEXT,
        )
        .unwrap()
        else {
            panic!("a short turn should fit");
        };
        assert_eq!(
            warm,
            codec
                .continue_with(tokenizer, &ChatMessage::user("again"))
                .unwrap()
        );
        let whole = tokenizer
            .encode_chat_sanitized(transcript.messages(), true)
            .unwrap();
        assert!(
            warm.len() < whole.len(),
            "{} new ids vs {} for the whole render",
            warm.len(),
            whole.len()
        );
        // The accounting is over the conversation, not over the delta.
        assert_eq!(used, history_len + warm.len());
        assert_eq!(used + 128 + room, DEFAULT_CONTEXT);
    }

    /// A turn that fits 4096 but not 512 is admitted by one run and refused
    /// by the other, from the same transcript — which is the whole of what
    /// making the window a runtime value buys.
    #[test]
    fn the_same_turn_is_admitted_or_refused_by_the_configured_window() {
        let tokenizer = fixture_tokenizer();
        let codec = TurnCodec::new(tokenizer).unwrap();

        let mut roomy = Transcript::new(Vec::new());
        let TurnPlan::Ready { used, .. } = plan_turn(
            tokenizer,
            &codec,
            &mut roomy,
            0,
            "hi",
            1024,
            DEFAULT_CONTEXT,
        )
        .unwrap() else {
            panic!("1024 reserved tokens fit a 4096 window");
        };
        assert!(used + 1024 <= DEFAULT_CONTEXT);

        let mut cramped = Transcript::new(Vec::new());
        let TurnPlan::Refused(reason) =
            plan_turn(tokenizer, &codec, &mut cramped, 0, "hi", 1024, 512).unwrap()
        else {
            panic!("1024 reserved tokens cannot fit a 512 window");
        };
        assert!(reason.contains("of 512 tokens"), "{reason}");
        assert!(!reason.contains("4096"), "{reason}");
        assert_eq!(cramped.messages(), [], "a refused turn stores nothing");
    }

    /// `generate_from`'s feeding rule, which the REPL's bookkeeping rests on:
    /// every prompt id is fed, and every generated id except the last one
    /// sampled — there is nothing left to predict from it, so it is printed
    /// and never seen again by the model.
    fn fed_by(prompt: usize, generated: usize) -> usize {
        prompt + generated.saturating_sub(1)
    }

    /// The invariant `chat_turn` depends on: `state.seq_len()` is the only
    /// authority on what the cache holds, and `history[seq_len..]` is by
    /// construction exactly what the next turn must feed — across an ordinary
    /// turn, a turn cut short by Ctrl-C, and `/reset`.
    #[test]
    fn the_id_history_and_the_cache_stay_in_step_across_turns() {
        let mut history: Vec<u32> = Vec::new();
        let mut seq_len = 0usize;

        // Turn 0: a 5-id prompt, three ids generated.
        let prompt = [1u32, 2, 3, 4, 5];
        history.extend_from_slice(&prompt);
        assert_eq!(&history[seq_len..], &prompt, "the whole prompt is new");
        let generated = [10u32, 11, 12];
        seq_len = fed_by(seq_len + prompt.len(), generated.len());
        history.extend_from_slice(&generated);
        assert_eq!(seq_len, history.len() - 1, "the last id is never fed");

        // Turn 1: only the delta is new, and it trails the unfed id.
        let delta = [20u32, 21];
        history.extend_from_slice(&delta);
        assert_eq!(&history[seq_len..], &[12, 20, 21]);
        let generated = [30u32, 31];
        seq_len = fed_by(seq_len + (history.len() - seq_len), generated.len());
        history.extend_from_slice(&generated);
        assert_eq!(seq_len, history.len() - 1);

        // Turn 2, aborted after two ids. `chat_turn` keeps the ids it saw
        // through `on_token`, which have the same shape the stats would: all
        // sampled, the last one not yet fed.
        let delta = [40u32];
        history.extend_from_slice(&delta);
        let fed = seq_len;
        let spoken = [50u32, 51];
        seq_len = fed_by(fed + (history.len() - fed), spoken.len());
        history.extend_from_slice(&spoken);
        assert_eq!(seq_len, history.len() - 1, "an abort leaves the same shape");
        // So the next turn resumes from the partial reply rather than
        // re-feeding it or skipping it.
        assert_eq!(&history[seq_len..], &[51]);

        // `/reset` drops both together; the next turn is cold again.
        history.clear();
        seq_len = 0;
        assert!(history[seq_len..].is_empty());
    }
}
