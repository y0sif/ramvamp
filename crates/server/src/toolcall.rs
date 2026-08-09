//! Recovering structured `tool_calls` from the reply text the model wrote.
//!
//! The template tells the model to answer with
//!
//! ```text
//! <tool_call>
//! {"name": <function-name>, "arguments": <args-json-object>}
//! </tool_call>
//! ```
//!
//! which arrives here as ordinary generated text. A client cannot act on that;
//! it acts on `choices[0].message.tool_calls`. This module is the translation,
//! and it is pure: text in, [`Extracted`] out, no clock, no randomness, no I/O.
//! That is what lets every rule below be pinned by a unit test.
//!
//! # Finding the block boundaries
//!
//! The obvious implementation — `find("<tool_call>")`, then
//! `find("</tool_call>")` — has a bug that shows up the first time a model
//! quotes the protocol back at you:
//!
//! ```text
//! <tool_call>
//! {"name": "reply", "arguments": {"text": "wrap it in </tool_call> tags"}}
//! </tool_call>
//! ```
//!
//! The naive scan closes the block inside the JSON string and produces a
//! truncated, unparseable body. [`find_close`] therefore walks the bytes
//! tracking whether it is inside a JSON string literal (honouring `\`
//! escapes) and only accepts a `</tool_call>` found *outside* one.
//!
//! That tracking is itself a risk: a body that is not JSON at all can carry an
//! unbalanced `"`, which would make the scan believe the rest of the reply is
//! one long string and lose a terminator that is plainly there. So the
//! string-aware scan falls back to a plain search when it reaches the end
//! without a match. The fallback means the careful scan can only ever find
//! *more* blocks than the naive one, never fewer.
//!
//! Locating `name` and `arguments` inside the body uses the same idea one
//! level down: a member's value is a *span*, delimited by matching braces or
//! brackets with strings skipped, so `{"a": {"b": 1}}` and `{"a": "} }"}` both
//! end where they should. A naive `find('}')` ends both in the wrong place.
//!
//! # Bytes are preserved, never re-serialized
//!
//! [`ParsedCall::arguments`] is the raw substring the model emitted. It is not
//! parsed into a `serde_json::Value` and printed back, for two reasons. The
//! client echoes `arguments` verbatim on the next turn and
//! [`crate::prompt`] splices that string straight into the prompt as
//! `ToolArguments::Raw`, so preserving the bytes makes the round trip exact —
//! the model sees on turn *n+1* the characters it wrote on turn *n*.
//! Re-serializing would also quietly normalise key order and spacing, and
//! would turn the everyday case of a model emitting slightly invalid JSON into
//! a hard failure.
//!
//! # Malformed blocks are recovered, not dropped
//!
//! Models get this JSON wrong often enough that dropping a malformed block
//! would be a routine loss of the turn. If a `name` can be recovered the call
//! is emitted with whatever the `arguments` span held, raw and unchecked;
//! `FunctionCall::arguments` is a `String` precisely so the client can be
//! handed exactly what the model said and fail on its own terms. Only a block
//! with no recoverable name falls back to being ordinary content — and then
//! the whole block, markers included, is preserved there, so nothing the model
//! wrote is ever discarded.
//!
//! An unterminated `<tool_call>` is content for the same reason: it is what a
//! reply that ran out of `max_tokens` mid-block looks like, and no call can be
//! honestly claimed from half a block.
//!
//! # Streaming, where the whole reply is not in hand
//!
//! [`extract`] needs the finished text. A stream does not have it: the decision
//! to send a token as content has to be made before the token after it exists,
//! and once a `<` has gone out as content it cannot be recalled. Scanning the
//! text as it accumulates would also mean handling a marker split across two
//! deltas, since nothing about a byte stream promises `</tool_call>` arrives
//! whole.
//!
//! It does arrive whole, and that is the mechanism [`CallStream`] is built on.
//! Both markers are *added tokens* in the Qwen3 tokenizer —
//! [`OPEN_TAG_ID`] and [`CLOSE_TAG_ID`] — so each is a single id, and the
//! decoder does not skip them. Detection is therefore an integer compare
//! against the id the model sampled, made at the one layer that still has the
//! id, and a marker cannot be split because a token cannot be. What is left is
//! bookkeeping: buffer between the two ids, hand the body to the same
//! [`parse_block`] the buffered path uses, and give the call the same id
//! [`wire_call`] would have given it there.

use crate::request::{FunctionCall, ToolCall};
use crate::response::COMPLETION_ID_PREFIX;

/// The marker that opens a call block.
pub const OPEN_TAG: &str = "<tool_call>";

/// The marker that closes one.
pub const CLOSE_TAG: &str = "</tool_call>";

/// The single token id [`OPEN_TAG`] encodes to.
///
/// An `added_tokens` entry of the Qwen3 tokenizer, `special: false`, so the
/// streaming decoder emits it as one token carrying the literal text. Pinned as
/// a constant rather than looked up because the streaming sink has no
/// tokenizer: it sees ids and text, and this crate is meant to be testable
/// without an install on disk.
pub const OPEN_TAG_ID: u32 = 151_657;

/// The single token id [`CLOSE_TAG`] encodes to. See [`OPEN_TAG_ID`].
pub const CLOSE_TAG_ID: u32 = 151_658;

/// The `type` of every call: OpenAI has defined exactly one.
pub const CALL_KIND: &str = "function";

/// The prefix of a generated call id.
pub const CALL_ID_PREFIX: &str = "call_";

/// What a block that named no arguments contributes.
///
/// The empty object rather than the empty string: a client's first move is
/// `json.loads(arguments)`, and `""` breaks it where `{}` is the truth — a
/// call with no arguments.
const NO_ARGUMENTS: &str = "{}";

/// One call recovered from a reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCall {
    /// The function name, unescaped.
    pub name: String,
    /// The arguments, as the exact bytes the model emitted.
    pub arguments: String,
}

/// A reply split into what the client should show and what it should run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Extracted {
    /// Everything outside the blocks that became calls.
    ///
    /// When no call was extracted this is the reply unchanged, byte for byte;
    /// a plain answer must not be altered by a parser that found nothing to
    /// do. When calls *were* extracted the surrounding prose is joined and
    /// trimmed at the ends, since the whitespace that framed a removed block
    /// is framing, not text.
    pub content: String,
    /// The calls, in the order the model asked for them.
    pub calls: Vec<ParsedCall>,
}

impl Extracted {
    /// The reply as pure content, with nothing parsed out of it.
    ///
    /// For a request that declared no tools. The markers only appear because
    /// the template's tools branch instructed the model to produce them, so
    /// without tools they are ordinary text: a conversation *about* tool
    /// calling would otherwise have its example swallowed into a structured
    /// call the caller never enabled.
    pub fn none(reply: &str) -> Self {
        Extracted {
            content: reply.to_owned(),
            calls: Vec::new(),
        }
    }
}

/// Split `reply` into content and the calls it asked for.
///
/// Never fails and never panics: every malformed shape has a defined outcome,
/// documented on [the module](self).
pub fn extract(reply: &str) -> Extracted {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    while let Some(open) = reply.get(cursor..).and_then(|rest| rest.find(OPEN_TAG)) {
        let open = cursor + open;
        let body_start = open + OPEN_TAG.len();
        // No terminator: this block never finished, so it is text. Leaving
        // `cursor` where it is puts the opener and everything after it into
        // content below.
        let Some(close) = find_close(reply, body_start) else {
            break;
        };
        let body = reply.get(body_start..close).unwrap_or_default();
        match parse_block(body) {
            Some(call) => {
                content.push_str(reply.get(cursor..open).unwrap_or_default());
                calls.push(call);
            }
            // Not a call after all: keep the block verbatim, markers included.
            None => content.push_str(
                reply
                    .get(cursor..close + CLOSE_TAG.len())
                    .unwrap_or_default(),
            ),
        }
        cursor = close + CLOSE_TAG.len();
    }
    content.push_str(reply.get(cursor..).unwrap_or_default());

    if !calls.is_empty() {
        content = content.trim().to_owned();
    }
    Extracted { content, calls }
}

/// The id a call is given, derived from the completion's own id.
///
/// Deterministic on purpose. The client sends this back as `tool_call_id`, so
/// it has to be stable within a response, and a test that cannot predict an id
/// cannot pin the round trip — the same reason `id` and `created` are
/// parameters everywhere else in this crate rather than reads of a clock.
/// Uniqueness comes from the completion id, which already cannot repeat.
pub fn call_id(completion_id: &str, index: usize) -> String {
    let suffix = completion_id
        .strip_prefix(COMPLETION_ID_PREFIX)
        .unwrap_or(completion_id);
    format!("{CALL_ID_PREFIX}{suffix}_{index}")
}

/// Give one parsed call its wire shape and its id, at `index`.
///
/// The single place a [`ParsedCall`] becomes a [`ToolCall`], which is what
/// makes the buffered and streaming paths agree by construction rather than by
/// two implementations that happen to match today: the buffered path indexes a
/// finished list, the streaming one counts calls as they close, and both arrive
/// here.
pub fn wire_call(completion_id: &str, index: usize, call: &ParsedCall) -> ToolCall {
    ToolCall {
        id: call_id(completion_id, index),
        kind: CALL_KIND.to_owned(),
        function: FunctionCall {
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        },
    }
}

/// Give the parsed calls their wire shape and their ids.
pub fn wire_calls(completion_id: &str, calls: &[ParsedCall]) -> Vec<ToolCall> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| wire_call(completion_id, index, call))
        .collect()
}

/// What the sink should do with the token it just fed to a [`CallStream`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Send nothing: the token is inside a block that has not closed.
    Nothing,
    /// Send this as a content delta. Not always the token's own text — a block
    /// that closed without naming a function comes back here whole, markers
    /// included, exactly as the buffered path keeps it.
    Content(String),
    /// A block closed and named a function.
    Call {
        /// Its position in this response's `tool_calls`, which is the `index`
        /// the delta must carry: it is how a client knows which call a
        /// fragment belongs to.
        index: usize,
        /// The call.
        call: ParsedCall,
    },
}

/// [`extract`]'s rules, applied one token at a time.
///
/// Fed every generated token in order, it answers what to put on the wire for
/// it. The outcome matches what [`extract`] would have produced from the
/// concatenated reply, with one unavoidable difference: [`extract`] trims the
/// prose that framed a removed block, and a stream has already sent that
/// whitespace by the time the block it framed appears.
///
/// Pure and clock-free like the rest of the module, so the state machine is
/// pinned by unit tests rather than by a live model.
///
/// The other difference is [`find_close`]'s string awareness, which has no
/// streaming equivalent and does not need one. A `</tool_call>` written *inside*
/// an argument string reaches [`extract`] as twelve characters of a JSON
/// literal; it reaches here as [`CLOSE_TAG_ID`], the id the model chose to
/// sample, and an id is not text. Closing there is the honest reading of what
/// the model emitted, and it is also the only reading available: the fallback
/// that makes the buffered scan safe — search the *rest of the reply* — is a
/// rest that does not exist yet.
#[derive(Debug, Clone)]
pub struct CallStream {
    /// Whether calls are recovered at all.
    ///
    /// False for a request that declared no tools, and then every token is
    /// content. The model only writes these markers because the tools branch of
    /// the template told it to, so without tools they are ordinary text a user
    /// asked for — and swallowing them would delete the answer. The same gate
    /// the buffered path applies, for the same reason.
    enabled: bool,
    /// The body of the block currently open, markers excluded, or `None`
    /// outside one.
    open: Option<String>,
    /// Calls emitted so far, which is both the next `index` and the answer to
    /// "was this turn a tool call".
    emitted: usize,
}

impl CallStream {
    /// A stream that recovers calls, or one that passes everything through.
    pub fn new(enabled: bool) -> Self {
        CallStream {
            enabled,
            open: None,
            emitted: 0,
        }
    }

    /// How many calls have been emitted.
    ///
    /// Non-zero is what makes `finish_reason` `tool_calls`: the model stopped
    /// because it wanted a tool, whatever token ended the reply.
    pub fn emitted(&self) -> usize {
        self.emitted
    }

    /// Feed one generated token and its id.
    pub fn push(&mut self, id: u32, text: &str) -> Step {
        if !self.enabled {
            return Step::Content(text.to_owned());
        }
        match id {
            // A second opener inside an open block is body text, matching
            // `extract`, which looks for the terminator and not for another
            // opener.
            OPEN_TAG_ID if self.open.is_none() => {
                self.open = Some(String::new());
                Step::Nothing
            }
            // A terminator with nothing open closes nothing; it is the stray
            // marker case, and it is content.
            CLOSE_TAG_ID if self.open.is_some() => self.close(),
            _ => match &mut self.open {
                Some(body) => {
                    body.push_str(text);
                    Step::Nothing
                }
                None => Step::Content(text.to_owned()),
            },
        }
    }

    /// Generation ended: give back the text of a block that never closed.
    ///
    /// Half a block is not a call — it is what a reply that ran out of
    /// `max_tokens` mid-block looks like — but it *is* text the model wrote,
    /// and nothing the model wrote is dropped. Idempotent.
    pub fn flush(&mut self) -> Option<String> {
        self.open.take().map(|body| format!("{OPEN_TAG}{body}"))
    }

    /// Resolve the block that just closed.
    fn close(&mut self) -> Step {
        let body = self.open.take().unwrap_or_default();
        match parse_block(&body) {
            Some(call) => {
                let index = self.emitted;
                self.emitted += 1;
                Step::Call { index, call }
            }
            // No recoverable name, so it was never a call: the block survives
            // as text, markers and all.
            None => Step::Content(format!("{OPEN_TAG}{body}{CLOSE_TAG}")),
        }
    }
}

/// Find the `</tool_call>` that closes the block whose body starts at `from`.
///
/// String-aware, then naive: see [the module](self) for why it is both.
fn find_close(reply: &str, from: usize) -> Option<usize> {
    let bytes = reply.as_bytes();
    let mut i = from;
    while i < bytes.len() {
        match bytes[i] {
            // A string literal is skipped whole, so a `</tool_call>` written
            // *inside* one closes nothing.
            b'"' => i = end_of_string(bytes, i),
            b'<' if bytes[i..].starts_with(CLOSE_TAG.as_bytes()) => return Some(i),
            _ => i += 1,
        }
    }
    // An unbalanced quote in a body that was never JSON must not hide a
    // terminator that is right there in the text.
    reply
        .get(from..)
        .and_then(|rest| rest.find(CLOSE_TAG))
        .map(|at| from + at)
}

/// Recover a call from one block body, or decide it is not a call.
fn parse_block(body: &str) -> Option<ParsedCall> {
    let open = body.find('{')?;
    let mut name = None;
    let mut arguments = None;
    for (key, value) in members(body, open) {
        match key.as_str() {
            // First wins: a duplicated key is malformed, and the first is what
            // a JSON reader that stops at the first match would have seen.
            "name" if name.is_none() => name = Some(value),
            "arguments" if arguments.is_none() => arguments = Some(value),
            _ => {}
        }
    }
    let name = unquote(name?);
    if name.is_empty() {
        return None;
    }
    Some(ParsedCall {
        name,
        arguments: arguments.map_or_else(|| NO_ARGUMENTS.to_owned(), str::to_owned),
    })
}

/// The top-level members of the object starting at `open`, as (key, raw value).
///
/// Deliberately not a JSON parser: it recovers what it can and stops at the
/// first thing it does not understand, keeping every member it read up to
/// there. A strict parser would return nothing for a body with a trailing
/// comma, which is exactly the body this has to survive.
fn members(text: &str, open: usize) -> Vec<(String, &str)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = open + 1;
    loop {
        i = skip_space(bytes, i);
        // A key is quoted; anything else ends the walk, `}` included.
        if bytes.get(i) != Some(&b'"') {
            return out;
        }
        let key_end = end_of_string(bytes, i);
        let key = unquote(text.get(i..key_end).unwrap_or_default());
        i = skip_space(bytes, key_end);
        if bytes.get(i) != Some(&b':') {
            return out;
        }
        i = skip_space(bytes, i + 1);
        if i >= bytes.len() {
            return out;
        }
        let value_end = end_of_value(bytes, i);
        out.push((key, text.get(i..value_end).unwrap_or_default()));
        i = skip_space(bytes, value_end);
        if bytes.get(i) != Some(&b',') {
            return out;
        }
        i += 1;
    }
}

/// The index just past the value starting at `start`.
fn end_of_value(bytes: &[u8], start: usize) -> usize {
    match bytes.get(start) {
        Some(b'"') => end_of_string(bytes, start),
        Some(b'{') => end_of_nested(bytes, start, b'{', b'}'),
        Some(b'[') => end_of_nested(bytes, start, b'[', b']'),
        // A bare token: a number, `true`, `false`, `null`, or whatever the
        // model put there instead. It ends at the first delimiter.
        _ => {
            let mut i = start;
            while i < bytes.len() && !matches!(bytes[i], b',' | b'}' | b']') && !is_space(bytes[i])
            {
                i += 1;
            }
            i
        }
    }
}

/// The index just past the balanced `open`/`close` pair starting at `start`.
///
/// Strings are skipped whole, so a brace inside one does not count — the
/// difference between reading `{"path": "a}b"}` and reading half of it.
/// An unbalanced value ends at the end of the input, which keeps a truncated
/// block's arguments recoverable instead of empty.
fn end_of_nested(bytes: &[u8], start: usize, open: u8, close: u8) -> usize {
    let mut depth = 0usize;
    let mut i = start;
    while i < bytes.len() {
        let byte = bytes[i];
        if byte == b'"' {
            i = end_of_string(bytes, i);
            continue;
        }
        if byte == open {
            depth += 1;
        } else if byte == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return i + 1;
            }
        }
        i += 1;
    }
    bytes.len()
}

/// The index just past the string literal starting at `start`, which must be
/// its opening quote. An unterminated literal ends at the end of the input.
fn end_of_string(bytes: &[u8], start: usize) -> usize {
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            // A `\` consumes the next byte, so `\"` does not end the string.
            // Landing mid-character is harmless: this scan only ever compares
            // against ASCII, and every index it hands out is a boundary.
            b'\\' => i += 2,
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    bytes.len()
}

/// JSON whitespace.
fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

fn skip_space(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && is_space(bytes[i]) {
        i += 1;
    }
    i
}

/// The text of a quoted span, with JSON escapes resolved where they can be.
///
/// `serde_json` does the unescaping when the span is a well-formed literal, so
/// a `r` in a name is honoured. When it is not — an unterminated or badly
/// escaped literal — the quotes come off by hand rather than the value being
/// lost, which is the module's rule everywhere: recover, do not drop.
fn unquote(raw: &str) -> String {
    if let Ok(text) = serde_json::from_str::<String>(raw) {
        return text;
    }
    raw.strip_prefix('"')
        .map_or(raw, |rest| rest.strip_suffix('"').unwrap_or(rest))
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block the template asks for, around whatever `arguments` says.
    fn block(arguments: &str) -> String {
        format!("<tool_call>\n{{\"name\": \"read\", \"arguments\": {arguments}}}\n</tool_call>")
    }

    #[test]
    fn a_reply_with_no_calls_is_returned_byte_for_byte() {
        for reply in [
            "",
            "4",
            "  Sure — the answer is 4.\n\n",
            "a < b and c > d",
            "here is a tag: <tool>",
        ] {
            let out = extract(reply);
            assert!(out.calls.is_empty(), "{reply:?} produced {:?}", out.calls);
            assert_eq!(out.content, reply, "a plain reply must not be rewritten");
        }
    }

    #[test]
    fn one_call_becomes_one_parsed_call_and_empty_content() {
        let out = extract(&block(r#"{"path": "a.txt"}"#));
        assert_eq!(out.content, "");
        assert_eq!(
            out.calls,
            vec![ParsedCall {
                name: "read".to_owned(),
                arguments: r#"{"path": "a.txt"}"#.to_owned(),
            }]
        );
    }

    #[test]
    fn two_calls_in_one_turn_are_both_extracted_in_order() {
        let reply = format!(
            "{}\n{}",
            block(r#"{"path": "a"}"#),
            block(r#"{"path": "b"}"#)
        );
        let out = extract(&reply);
        assert_eq!(out.calls.len(), 2);
        assert_eq!(out.calls[0].arguments, r#"{"path": "a"}"#);
        assert_eq!(out.calls[1].arguments, r#"{"path": "b"}"#);
        assert_eq!(out.content, "");
    }

    #[test]
    fn prose_before_between_and_after_survives() {
        let reply = format!(
            "I will read both.\n{}\nand the other one\n{}\nThen I will compare them.",
            block(r#"{"path": "a"}"#),
            block(r#"{"path": "b"}"#)
        );
        let out = extract(&reply);
        assert_eq!(out.calls.len(), 2);
        assert_eq!(
            out.content,
            "I will read both.\n\nand the other one\n\nThen I will compare them."
        );
    }

    #[test]
    fn an_empty_arguments_object_is_preserved_as_written() {
        let out = extract(&block("{}"));
        assert_eq!(out.calls[0].arguments, "{}");
    }

    /// The absent-`arguments` case is the one place a value is invented, and
    /// `{}` is invented rather than `""` because the client parses it.
    #[test]
    fn a_block_with_no_arguments_key_gets_an_empty_object() {
        let out = extract("<tool_call>\n{\"name\": \"now\"}\n</tool_call>");
        assert_eq!(
            out.calls,
            vec![ParsedCall {
                name: "now".to_owned(),
                arguments: "{}".to_owned(),
            }]
        );
    }

    /// Models emit invalid JSON here routinely. A trailing comma must cost the
    /// client a parse error it can see, not a silently dropped turn.
    #[test]
    fn malformed_json_with_a_recoverable_name_still_becomes_a_call() {
        let out = extract(&block(r#"{"path": "a.txt",}"#));
        assert_eq!(out.calls[0].name, "read");
        assert_eq!(
            out.calls[0].arguments, r#"{"path": "a.txt",}"#,
            "the model's bytes reach the client unaltered, invalid or not"
        );
        assert_eq!(out.content, "");
    }

    /// A block truncated inside `arguments` still names a function, so it is
    /// still a call — with everything that was written of its arguments.
    #[test]
    fn a_block_missing_its_closing_brace_keeps_what_arguments_had() {
        let out = extract(
            "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"a\"\n</tool_call>",
        );
        assert_eq!(out.calls[0].name, "read");
        assert_eq!(out.calls[0].arguments, "{\"path\": \"a\"\n");
    }

    #[test]
    fn a_block_with_no_recoverable_name_is_content_markers_and_all() {
        for body in [
            "{\"arguments\": {\"path\": \"a\"}}",
            "{\"name\": \"\"}",
            "just some words",
            "",
        ] {
            let reply = format!("<tool_call>{body}</tool_call>");
            let out = extract(&reply);
            assert!(out.calls.is_empty(), "{body:?} became a call");
            assert_eq!(out.content, reply, "the block must survive as text");
        }
    }

    /// What a reply that hit `max_tokens` mid-block looks like.
    #[test]
    fn an_unterminated_block_is_content() {
        let reply = "Reading it.\n<tool_call>\n{\"name\": \"read\", \"argum";
        let out = extract(reply);
        assert!(out.calls.is_empty());
        assert_eq!(out.content, reply);
    }

    #[test]
    fn a_stray_closing_marker_is_content() {
        let reply = "the tag is </tool_call>, as shown";
        let out = extract(reply);
        assert!(out.calls.is_empty());
        assert_eq!(out.content, reply);
    }

    /// A naive `find('}')` ends the arguments at the inner object.
    #[test]
    fn nested_braces_inside_arguments_are_balanced_not_guessed() {
        let arguments =
            r#"{"filter": {"path": "a.txt", "range": {"from": 1, "to": 9}}, "raw": true}"#;
        let out = extract(&block(arguments));
        assert_eq!(out.calls[0].arguments, arguments);
    }

    /// A brace inside a string is text, not structure.
    #[test]
    fn a_brace_inside_a_string_argument_does_not_end_the_value() {
        let arguments = r#"{"pattern": "^\\{.*\\}$", "note": "a } on its own"}"#;
        let out = extract(&block(arguments));
        assert_eq!(out.calls[0].arguments, arguments);
    }

    /// The bug this parser exists to avoid: a terminator quoted inside the
    /// arguments must not close the block.
    #[test]
    fn a_closing_marker_inside_a_string_argument_does_not_close_the_block() {
        let arguments = r#"{"text": "wrap it in </tool_call> tags"}"#;
        let out = extract(&block(arguments));
        assert_eq!(out.calls.len(), 1, "the block closed too early");
        assert_eq!(out.calls[0].arguments, arguments);
        assert_eq!(out.content, "");
    }

    /// The same trap one escape deeper: the quote before the marker is
    /// escaped, so the string has not ended there.
    #[test]
    fn an_escaped_quote_does_not_end_the_string_scan() {
        let arguments = r#"{"text": "he said \"</tool_call>\" out loud"}"#;
        let out = extract(&block(arguments));
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].arguments, arguments);
    }

    /// And the reason the string-aware scan has a fallback: an unbalanced
    /// quote in a body that was never JSON must not swallow the rest of the
    /// reply.
    #[test]
    fn an_unbalanced_quote_does_not_hide_the_terminator() {
        let reply =
            "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path: \"a}\n</tool_call>\nDone.";
        let out = extract(reply);
        assert_eq!(out.calls.len(), 1, "the terminator was lost");
        assert_eq!(out.calls[0].name, "read");
        assert_eq!(out.content, "Done.");
    }

    #[test]
    fn an_escaped_name_is_unescaped_and_its_arguments_are_not() {
        let reply = r#"<tool_call>{"name": "read", "arguments": {"q": "a\nb"}}</tool_call>"#;
        let out = extract(reply);
        assert_eq!(out.calls[0].name, "read");
        assert_eq!(
            out.calls[0].arguments, r#"{"q": "a\nb"}"#,
            "arguments are bytes, not a decoded value"
        );
    }

    /// Multibyte text anywhere near the scanner must not split a character.
    #[test]
    fn non_ascii_text_around_and_inside_a_block_is_safe() {
        let reply = format!(
            "Je vais lire → {}\n…voilà",
            block(r#"{"path": "café.txt", "note": "naïve"}"#)
        );
        let out = extract(&reply);
        assert_eq!(
            out.calls[0].arguments,
            r#"{"path": "café.txt", "note": "naïve"}"#
        );
        // Only the ends are trimmed: the space that preceded the block is
        // prose the model wrote between two words.
        assert_eq!(out.content, "Je vais lire → \n…voilà");
    }

    /// The whitespace and key order the model chose reach the client, because
    /// the client sends them back and they are re-rendered into the prompt.
    #[test]
    fn argument_bytes_are_preserved_exactly() {
        let arguments = "{ \"b\":2,\n  \"a\" : 1 }";
        let out = extract(&block(arguments));
        assert_eq!(out.calls[0].arguments, arguments);
    }

    #[test]
    fn ids_are_derived_from_the_completion_id_and_the_index() {
        let calls = vec![
            ParsedCall {
                name: "read".to_owned(),
                arguments: "{}".to_owned(),
            },
            ParsedCall {
                name: "write".to_owned(),
                arguments: r#"{"path":"a"}"#.to_owned(),
            },
        ];
        let wire = wire_calls("chatcmpl-18f3a0001", &calls);
        assert_eq!(wire[0].id, "call_18f3a0001_0");
        assert_eq!(wire[1].id, "call_18f3a0001_1");
        assert!(wire.iter().all(|call| call.kind == "function"));
        assert_eq!(wire[1].function.name, "write");
        assert_eq!(wire[1].function.arguments, r#"{"path":"a"}"#);
        // Same input, same ids: the round trip is assertable.
        assert_eq!(wire_calls("chatcmpl-18f3a0001", &calls), wire);
        // An id without the usual prefix still yields a usable one.
        assert_eq!(call_id("x", 3), "call_x_3");
    }

    // ---- the streaming half ----

    /// The id of an ordinary token. Any value that is not a marker will do:
    /// the state machine only ever compares against the two markers.
    const TEXT: u32 = 7;

    /// Feed a token sequence through a [`CallStream`] and collect what it would
    /// have put on the wire: the concatenated content and the calls with their
    /// indices, including whatever the final flush produced.
    fn drive(enabled: bool, tokens: &[(u32, &str)]) -> (String, Vec<(usize, ParsedCall)>) {
        let mut stream = CallStream::new(enabled);
        let mut content = String::new();
        let mut calls = Vec::new();
        for (id, text) in tokens {
            match stream.push(*id, text) {
                Step::Nothing => {}
                Step::Content(text) => content.push_str(&text),
                Step::Call { index, call } => calls.push((index, call)),
            }
        }
        if let Some(rest) = stream.flush() {
            content.push_str(&rest);
        }
        // Flushing twice must not duplicate the tail.
        assert_eq!(stream.flush(), None, "flush is not idempotent");
        assert_eq!(stream.emitted(), calls.len());
        (content, calls)
    }

    /// The markers as the tokens they actually are, around one body token.
    fn streamed(body: &str) -> Vec<(u32, &str)> {
        vec![
            (OPEN_TAG_ID, OPEN_TAG),
            (TEXT, body),
            (CLOSE_TAG_ID, CLOSE_TAG),
        ]
    }

    /// The property the streaming path exists to preserve: the same reply, one
    /// token at a time, recovers the same call the buffered path recovers.
    #[test]
    fn a_streamed_block_recovers_what_the_buffered_path_recovers() {
        let body = "\n{\"name\": \"glob\", \"arguments\": {\"pattern\": \"*/.rs\"}}\n";
        let (content, calls) = drive(true, &streamed(body));
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, 0);
        assert_eq!(
            calls[0].1,
            extract(&format!("{OPEN_TAG}{body}{CLOSE_TAG}")).calls[0]
        );
        assert_eq!(calls[0].1.name, "glob");
        assert_eq!(calls[0].1.arguments, r#"{"pattern": "*/.rs"}"#);
    }

    /// A body arrives as many tokens, because a body is many tokens. Only the
    /// markers are single ids.
    #[test]
    fn a_body_split_across_tokens_is_reassembled_whole() {
        let mut tokens = vec![(OPEN_TAG_ID, OPEN_TAG)];
        for piece in [
            "\n{\"na",
            "me\": \"re",
            "ad\", \"argu",
            "ments\": {\"path\": \"a\"}}\n",
        ] {
            tokens.push((TEXT, piece));
        }
        tokens.push((CLOSE_TAG_ID, CLOSE_TAG));
        let (content, calls) = drive(true, &tokens);
        assert_eq!(content, "");
        assert_eq!(calls[0].1.name, "read");
        assert_eq!(calls[0].1.arguments, r#"{"path": "a"}"#);
    }

    #[test]
    fn prose_before_between_and_after_streams_as_content() {
        let mut tokens = vec![(TEXT, "I will "), (TEXT, "read both.\n")];
        tokens.extend(streamed(r#"{"name": "read", "arguments": {"path": "a"}}"#));
        tokens.push((TEXT, "\nand the other\n"));
        tokens.extend(streamed(r#"{"name": "read", "arguments": {"path": "b"}}"#));
        tokens.push((TEXT, "\nDone."));

        let (content, calls) = drive(true, &tokens);
        assert_eq!(content, "I will read both.\n\nand the other\n\nDone.");
        assert_eq!(calls.len(), 2);
        // The index is how a client tells two calls apart, so it counts.
        assert_eq!(calls[0].0, 0);
        assert_eq!(calls[1].0, 1);
        assert_eq!(calls[0].1.arguments, r#"{"path": "a"}"#);
        assert_eq!(calls[1].1.arguments, r#"{"path": "b"}"#);
    }

    /// What a reply that hit `max_tokens` mid-block looks like: no call, and
    /// not one byte of what the model wrote is lost.
    #[test]
    fn an_unterminated_streamed_block_flushes_as_content() {
        let tokens = [
            (TEXT, "Reading it.\n"),
            (OPEN_TAG_ID, OPEN_TAG),
            (TEXT, "\n{\"name\": \"read\", \"argum"),
        ];
        let (content, calls) = drive(true, &tokens);
        assert!(calls.is_empty());
        assert_eq!(
            content,
            "Reading it.\n<tool_call>\n{\"name\": \"read\", \"argum"
        );
    }

    /// Without tools the markers are ordinary text: the request never enabled
    /// calling, so swallowing them would delete the user's answer.
    #[test]
    fn a_streamed_block_without_tools_is_ordinary_text() {
        let body = r#"{"name": "read", "arguments": {}}"#;
        let (content, calls) = drive(false, &streamed(body));
        assert!(calls.is_empty());
        assert_eq!(content, format!("{OPEN_TAG}{body}{CLOSE_TAG}"));
    }

    /// A closed block that names no function was never a call, and the whole
    /// block — markers included — reaches the client as text.
    #[test]
    fn a_streamed_block_with_no_recoverable_name_is_content_markers_and_all() {
        for body in ["{\"arguments\": {}}", "{\"name\": \"\"}", "words", ""] {
            let (content, calls) = drive(true, &streamed(body));
            assert!(calls.is_empty(), "{body:?} became a call");
            assert_eq!(content, format!("{OPEN_TAG}{body}{CLOSE_TAG}"));
        }
    }

    #[test]
    fn a_stray_streamed_terminator_is_content() {
        let tokens = [
            (TEXT, "the tag is "),
            (CLOSE_TAG_ID, CLOSE_TAG),
            (TEXT, ", as shown"),
        ];
        let (content, calls) = drive(true, &tokens);
        assert!(calls.is_empty());
        assert_eq!(content, "the tag is </tool_call>, as shown");
    }

    /// A second opener inside an open block is body text, exactly as the
    /// buffered scan treats it.
    #[test]
    fn a_nested_opener_is_body_text() {
        let tokens = [
            (OPEN_TAG_ID, OPEN_TAG),
            (TEXT, "{\"name\": \"echo\", \"arguments\": {\"t\": \""),
            (OPEN_TAG_ID, OPEN_TAG),
            (TEXT, "\"}}"),
            (CLOSE_TAG_ID, CLOSE_TAG),
        ];
        let (content, calls) = drive(true, &tokens);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1.arguments, "{\"t\": \"<tool_call>\"}");
    }

    /// Models get this JSON wrong routinely. The call still goes out, with
    /// whatever the `arguments` span held.
    #[test]
    fn malformed_json_in_a_streamed_block_still_yields_a_call() {
        let (content, calls) = drive(
            true,
            &streamed(r#"{"name": "read", "arguments": {"p": 1,}}"#),
        );
        assert_eq!(content, "");
        assert_eq!(calls[0].1.name, "read");
        assert_eq!(calls[0].1.arguments, r#"{"p": 1,}"#);
    }

    /// The streaming and buffered paths mint the same ids for the same
    /// completion, because there is one function that mints them.
    #[test]
    fn streamed_call_ids_match_the_buffered_paths() {
        let body = r#"{"name": "read", "arguments": {}}"#;
        let mut tokens = streamed(body);
        tokens.extend(streamed(body));
        let (_, calls) = drive(true, &tokens);

        let parsed: Vec<ParsedCall> = calls.iter().map(|(_, call)| call.clone()).collect();
        let buffered = wire_calls("chatcmpl-18f3a0001", &parsed);
        let streamed: Vec<ToolCall> = calls
            .iter()
            .map(|(index, call)| wire_call("chatcmpl-18f3a0001", *index, call))
            .collect();
        assert_eq!(streamed, buffered);
        assert_eq!(streamed[0].id, "call_18f3a0001_0");
        assert_eq!(streamed[1].id, "call_18f3a0001_1");
    }

    /// The two constants are the mechanism. If they ever drift from the
    /// tokenizer's `added_tokens`, streaming silently emits raw markup again.
    #[test]
    fn the_marker_ids_are_the_qwen3_added_tokens() {
        assert_eq!(OPEN_TAG_ID, 151_657);
        assert_eq!(CLOSE_TAG_ID, 151_658);
        assert_ne!(OPEN_TAG_ID, CLOSE_TAG_ID);
    }

    /// The extracted call, re-rendered on the next turn, is the text the model
    /// wrote. That is the whole point of keeping the bytes: the renderer emits
    /// `<tool_call>\n{"name": "<name>", "arguments": <raw>}\n</tool_call>`,
    /// splicing a `ToolArguments::Raw` verbatim, so a preserved span rebuilds
    /// the block character for character.
    #[test]
    fn a_call_round_trips_back_into_the_block_the_model_emitted() {
        let reply = block(r#"{"path": "a.txt"}"#);
        let out = extract(&reply);
        let wire = wire_calls("chatcmpl-1", &out.calls);
        let rendered = format!(
            "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>",
            wire[0].function.name, wire[0].function.arguments
        );
        assert_eq!(rendered, reply);
    }
}
