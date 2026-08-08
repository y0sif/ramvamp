//! `POST /v1/chat/completions` request shapes.
//!
//! Deserialization is deliberately permissive about *fields* and strict about
//! *meaning*: unknown keys are ignored so a client sending `logprobs`,
//! `presence_penalty` or next year's parameter is served rather than refused,
//! but anything this server cannot honour honestly ([`n` > 1][ServerError::MultipleChoices],
//! a non-text content part) is a typed error rather than a silent
//! approximation.
//!
//! # The `content` trap
//!
//! Clients send message content in two forms — a bare string, and an array of
//! typed parts:
//!
//! ```json
//! {"role":"user","content":"hello"}
//! {"role":"user","content":[{"type":"text","text":"hello"}]}
//! ```
//!
//! Both are ordinary traffic; OpenCode and the OpenAI SDKs emit the array form
//! routinely. The pinned ChatML template coerces non-string content to `""`
//! (see `ramvamp_core::tokenizer::chat`), so a server that models `content` as
//! `String` and shrugs at the array form does not fail — it prefills an empty
//! user turn and the model answers a question it never received. That is the
//! worst failure shape available: confident, fluent, and about nothing.
//!
//! Hence [`Content`]: an enum over both forms, flattened by
//! [`Message::text`], with every non-text part a typed rejection. Dropping an
//! `image_url` part would be the same bug wearing a smaller hat.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ServerError;

/// The charset a tool name has to match, as a regex, for error messages.
///
/// OpenAI's own constraint, reproduced rather than invented so a client that
/// already satisfies the upstream API satisfies this one. Checked by
/// [`is_valid_tool_name`], which spells it out in code because this crate does
/// not take a regex dependency for one character class.
pub const TOOL_NAME_PATTERN: &str = "^[A-Za-z0-9_-]{1,64}$";

/// Whether `name` is a tool name this server will render.
///
/// Names are *validated*, never sanitized. Every other untrusted string in a
/// request goes through the tokenizer's sanitizer, which breaks added-token
/// literals with a zero-width marker; doing that to a name would be silent
/// corruption, because the model echoes the name back and the client matches
/// the call against the name it sent. An invisible marker inside it makes the
/// call unmatchable and the log unreadable. So a bad name is a 400.
///
/// Byte length is character length here: every accepted character is ASCII, so
/// a name with a multi-byte character fails the charset test first.
pub fn is_valid_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Check one tool name, naming the request field it came from.
fn check_tool_name(name: &str, param: &'static str) -> Result<(), ServerError> {
    if is_valid_tool_name(name) {
        return Ok(());
    }
    Err(ServerError::InvalidToolName {
        name: name.to_owned(),
        param,
    })
}

/// Check a tool name that arrived as untyped JSON.
///
/// A definition whose `name` is absent or not a string goes down the same
/// path, with the offending JSON (`null`, `123`) quoted back: a nameless
/// definition can never produce a call the client is able to match.
fn check_json_tool_name(name: &Value, param: &'static str) -> Result<(), ServerError> {
    match name.as_str() {
        Some(name) => check_tool_name(name, param),
        None => Err(ServerError::InvalidToolName {
            name: name.to_string(),
            param,
        }),
    }
}

/// The `name` of one tool definition.
///
/// Accepts both shapes the renderer accepts: the OpenAI wrapper
/// (`{"type":"function","function":{"name":…}}`) and the bare
/// `{"name":…}`, mirroring the template's own `if tool_call.function`
/// flattening. Indexing a [`Value`] with a missing key yields `Null`, which
/// [`check_json_tool_name`] refuses.
fn definition_name(tool: &Value) -> &Value {
    match tool.get("function") {
        Some(function) => &function["name"],
        None => &tool["name"],
    }
}

/// Who is speaking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// Instructions to the model.
    System,
    /// A human turn.
    User,
    /// A model turn.
    Assistant,
    /// The result of a tool call. Rendered as a `<tool_response>` block
    /// inside a `user` turn; see [`crate::prompt`].
    Tool,
}

impl MessageRole {
    /// The lowercase wire string.
    pub fn as_str(self) -> &'static str {
        match self {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        }
    }
}

/// One part of an array-form `content`.
///
/// Modelled as an open struct rather than a closed `#[serde(tag = "type")]`
/// enum on purpose: an unknown variant would surface as a serde parse failure
/// ("unknown variant `image_url`"), which is a 400 that reads like malformed
/// JSON. Keeping `kind` a string lets any part type parse and then be refused
/// with a message that names it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentPart {
    /// The part's `type`, e.g. `text` or `image_url`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The text, present on `text` parts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl ContentPart {
    /// The `type` value of a text part.
    pub const TEXT: &'static str = "text";

    /// A text part.
    pub fn text(text: impl Into<String>) -> Self {
        ContentPart {
            kind: ContentPart::TEXT.to_owned(),
            text: Some(text.into()),
        }
    }

    /// The text this part contributes, or why it cannot contribute any.
    pub fn as_text(&self) -> Result<&str, ServerError> {
        if self.kind != ContentPart::TEXT {
            return Err(ServerError::UnsupportedContentPart {
                kind: self.kind.clone(),
            });
        }
        self.text.as_deref().ok_or(ServerError::MissingPartText)
    }
}

/// Message content in either of the two forms clients send.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// The bare-string form.
    Text(String),
    /// The array-of-parts form.
    Parts(Vec<ContentPart>),
}

impl Content {
    /// The text this content contributes.
    ///
    /// Parts are joined with no separator: the split is a transport detail, so
    /// a client that sent `["hello ", "world"]` meant `hello world`, and
    /// inserting a separator would fabricate whitespace the user never typed.
    pub fn text(&self) -> Result<String, ServerError> {
        match self {
            Content::Text(text) => Ok(text.clone()),
            Content::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    out.push_str(part.as_text()?);
                }
                Ok(out)
            }
        }
    }
}

/// A function call the model asked for, echoed back by the client on the next
/// turn.
///
/// Preserved verbatim through parsing and re-rendered into the prompt as the
/// `<tool_call>` block the model originally emitted, so the conversation the
/// model sees on turn *n+1* contains the call it made on turn *n*.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Call id, matched by a `tool` message's `tool_call_id`.
    pub id: String,
    /// Always `function` today.
    #[serde(rename = "type")]
    pub kind: String,
    /// The call itself.
    pub function: FunctionCall,
}

/// The function half of a [`ToolCall`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,
    /// Arguments as a JSON *string*, exactly as OpenAI sends them. Kept a
    /// string rather than parsed: models emit invalid JSON here often enough
    /// that parsing would turn a recoverable turn into a 400, and re-encoding
    /// valid JSON would not round-trip byte for byte.
    pub arguments: String,
}

/// One chat turn as the client sent it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Who is speaking.
    pub role: MessageRole,
    /// The content. `None` for an assistant turn that is only `tool_calls`,
    /// which clients send as an explicit `"content": null`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    /// Optional participant name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Calls an assistant turn requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Which call a `tool` turn answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// A message with plain string content.
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Message {
            role,
            content: Some(Content::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// The text of this message, flattened from whichever content form
    /// arrived. Absent content is the empty string, which is what an assistant
    /// turn carrying only `tool_calls` contributes.
    pub fn text(&self) -> Result<String, ServerError> {
        match &self.content {
            Some(content) => content.text(),
            None => Ok(String::new()),
        }
    }

    /// The calls this turn carries, empty when it carries none.
    pub fn calls(&self) -> &[ToolCall] {
        self.tool_calls.as_deref().unwrap_or(&[])
    }

    /// Refuse a call whose name is outside [`TOOL_NAME_PATTERN`].
    ///
    /// A name echoed back on an assistant turn gets the same check as a
    /// definition: it is rendered into the prompt the same way, and the next
    /// `tool` turn is matched against it by the client.
    pub fn check_tool_calls(&self) -> Result<(), ServerError> {
        for call in self.calls() {
            check_tool_name(&call.function.name, "messages")?;
        }
        Ok(())
    }
}

/// A `stop` value: OpenAI accepts a single string or an array of them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StringOrArray {
    /// One sequence.
    One(String),
    /// Up to four sequences.
    Many(Vec<String>),
}

impl StringOrArray {
    /// The sequences, in order.
    pub fn as_slice(&self) -> &[String] {
        match self {
            StringOrArray::One(one) => std::slice::from_ref(one),
            StringOrArray::Many(many) => many.as_slice(),
        }
    }
}

/// `stream_options`, the only sub-object that changes the chunk sequence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Emit a final usage-only chunk before the sentinel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// A `POST /v1/chat/completions` body.
///
/// Unknown fields are ignored rather than rejected — a client that sends a
/// parameter this build does not implement gets an answer, not a 400 — so the
/// fields listed here are the ones that change behaviour, not the ones that
/// are legal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Whatever the client called the model. Echoed back unexamined: this
    /// server serves exactly one install, so validating the name could only
    /// refuse requests it is able to answer.
    pub model: String,
    /// The conversation. Must be non-empty.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Stream the reply as SSE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Streaming extras.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Softmax temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus-sampling mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k cutoff. Not an OpenAI field; accepted because llama.cpp accepts
    /// it and local tooling sends it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// PRNG seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Reply length cap, the older spelling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Reply length cap, the current spelling. Wins when both appear.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,
    /// How many completions to return. Anything above 1 is refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Tool definitions, preserved verbatim.
    ///
    /// Held as raw JSON rather than a typed schema because the renderer takes
    /// them that way: each definition is serialized into the `# Tools` system
    /// block exactly as it arrived, so a Rust shape here could only lose
    /// fields (`strict`, vendor extensions) that the reference template keeps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    /// Tool selection policy. Only `auto` and `none` are honourable here; see
    /// [`ChatCompletionRequest::check_tool_choice`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
}

impl ChatCompletionRequest {
    /// Parse a request body.
    ///
    /// Every byte here is untrusted, so serde's failure is captured as a
    /// message rather than propagated as a panic or an opaque 500.
    pub fn from_json(body: &str) -> Result<Self, ServerError> {
        serde_json::from_str(body).map_err(|e| ServerError::MalformedJson(e.to_string()))
    }

    /// Check everything that can be checked without a tokenizer.
    ///
    /// Called before any work is scheduled, so a request that cannot be served
    /// never reaches the decode loop and never occupies the single session.
    pub fn validate(&self) -> Result<(), ServerError> {
        if self.messages.is_empty() {
            return Err(ServerError::EmptyMessages);
        }
        if let Some(n) = self.n
            && n > 1
        {
            return Err(ServerError::MultipleChoices { n });
        }
        // A stop sequence has no consumer: `GenerateParams` carries no stop
        // strings and the generate entry points stop on the tokenizer's own
        // stop tokens. Accepting one would let generation run past where the
        // caller asked it to end, with nothing in the response to say so.
        if !self.stop_sequences().is_empty() {
            return Err(ServerError::StopUnsupported);
        }
        self.check_tool_choice()?;
        self.check_tools()?;
        for message in &self.messages {
            // Flattening is the check: it is the only thing that inspects
            // every part, and doing it here means a bad part is refused before
            // the session is claimed rather than mid-prefill.
            message.text()?;
            message.check_tool_calls()?;
        }
        Ok(())
    }

    /// The tool definitions, empty when none were sent.
    pub fn tools(&self) -> &[Value] {
        self.tools.as_deref().unwrap_or(&[])
    }

    /// Refuse a tool definition whose name the model could not call back.
    ///
    /// Separate from [`validate`](Self::validate) so [`crate::prompt::build`],
    /// which is the thing that actually renders these, enforces it too rather
    /// than trusting that validation ran.
    pub fn check_tools(&self) -> Result<(), ServerError> {
        for tool in self.tools() {
            check_json_tool_name(definition_name(tool), "tools")?;
        }
        Ok(())
    }

    /// Refuse a `tool_choice` this build cannot honour.
    ///
    /// `auto`, `none` and absent are accepted and all render the same prompt,
    /// because that is what the template does: it branches on `tools` alone
    /// and has no way to say "you must call something". `required` and a named
    /// function therefore cannot be expressed at all, and serving them by
    /// letting the model choose freely would drop the one constraint the
    /// request was about.
    pub fn check_tool_choice(&self) -> Result<(), ServerError> {
        let Some(choice) = &self.tool_choice else {
            return Ok(());
        };
        if matches!(choice.as_str(), Some("auto" | "none")) {
            return Ok(());
        }
        Err(ServerError::UnsupportedToolChoice {
            choice: choice.to_string(),
        })
    }

    /// Whether to stream. Absent means false.
    pub fn stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    /// Whether to emit the usage-only chunk. Absent means false.
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .and_then(|options| options.include_usage)
            .unwrap_or(false)
    }

    /// The reply length cap.
    ///
    /// `max_completion_tokens` is the current spelling and `max_tokens` the
    /// deprecated alias; when both arrive the newer one wins, because a client
    /// that sends both is an SDK filling in a compatibility field behind a
    /// caller who set the modern one.
    pub fn max_new_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    /// The stop sequences, empty when none were sent.
    ///
    /// Read by [`validate`](Self::validate) only, to refuse a non-empty one.
    /// When the runtime grows stop strings this becomes the accessor that
    /// feeds them; until then it exists so the refusal has one definition of
    /// "sent a stop sequence" across both `stop` spellings.
    pub fn stop_sequences(&self) -> &[String] {
        self.stop.as_ref().map_or(&[], StringOrArray::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> Result<ChatCompletionRequest, ServerError> {
        ChatCompletionRequest::from_json(body)
    }

    #[test]
    fn string_content_parses() {
        let request = parse(r#"{"model":"m","messages":[{"role":"user","content":"hello"}]}"#)
            .expect("valid body");
        assert_eq!(request.messages[0].text().expect("text"), "hello");
        assert_eq!(request.messages[0].role, MessageRole::User);
    }

    /// The trap this module exists for: the array form must reach the model,
    /// not be coerced to `""`.
    #[test]
    fn array_form_content_is_flattened() {
        let request = parse(
            r#"{"model":"m","messages":[
                {"role":"user","content":[{"type":"text","text":"hello"}]}
            ]}"#,
        )
        .expect("valid body");
        assert_eq!(request.messages[0].text().expect("text"), "hello");
    }

    #[test]
    fn multiple_text_parts_concatenate_without_a_separator() {
        let request = parse(
            r#"{"model":"m","messages":[
                {"role":"user","content":[
                    {"type":"text","text":"hello "},
                    {"type":"text","text":"world"}
                ]}
            ]}"#,
        )
        .expect("valid body");
        assert_eq!(request.messages[0].text().expect("text"), "hello world");
    }

    #[test]
    fn empty_parts_array_flattens_to_empty_text() {
        let request =
            parse(r#"{"model":"m","messages":[{"role":"user","content":[]}]}"#).expect("valid");
        assert_eq!(request.messages[0].text().expect("text"), "");
    }

    #[test]
    fn non_text_content_part_is_rejected_not_dropped() {
        let request = parse(
            r#"{"model":"m","messages":[
                {"role":"user","content":[
                    {"type":"text","text":"what is this"},
                    {"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}
                ]}
            ]}"#,
        )
        .expect("valid body");
        let error = request.validate().expect_err("image parts are refused");
        assert!(
            matches!(&error, ServerError::UnsupportedContentPart { kind } if kind == "image_url"),
            "{error:?}"
        );
        assert!(error.to_string().contains("image_url"));
    }

    #[test]
    fn text_part_without_text_is_rejected() {
        let request =
            parse(r#"{"model":"m","messages":[{"role":"user","content":[{"type":"text"}]}]}"#)
                .expect("valid body");
        assert!(matches!(
            request.validate().expect_err("text is required"),
            ServerError::MissingPartText
        ));
    }

    #[test]
    fn null_content_flattens_to_empty_text() {
        let request = parse(
            r#"{"model":"m","messages":[{"role":"assistant","content":null,"tool_calls":[]}]}"#,
        )
        .expect("valid body");
        assert_eq!(request.messages[0].text().expect("text"), "");
    }

    #[test]
    fn empty_messages_is_rejected() {
        let request = parse(r#"{"model":"m","messages":[]}"#).expect("valid body");
        assert!(matches!(
            request.validate().expect_err("must be non-empty"),
            ServerError::EmptyMessages
        ));
        // Absent entirely, too.
        let request = parse(r#"{"model":"m"}"#).expect("valid body");
        assert!(matches!(
            request.validate().expect_err("must be non-empty"),
            ServerError::EmptyMessages
        ));
    }

    /// The refusal this replaced answered 400 to every tools request; the
    /// renderer can carry them now, so a well-formed one is served.
    #[test]
    fn a_request_carrying_tools_is_accepted() {
        let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"read_file","parameters":{}}}]}"#;
        let req = ChatCompletionRequest::from_json(body).expect("parses");
        req.validate().expect("the renderer has a tools branch");
        assert_eq!(req.tools().len(), 1);
    }

    #[test]
    fn an_empty_tools_array_is_not_a_refusal() {
        let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[]}"#;
        let req = ChatCompletionRequest::from_json(body).expect("parses");
        req.validate().expect("an empty list asks for nothing");
        assert!(req.tools().is_empty());
    }

    #[test]
    fn the_tool_name_charset_is_openais() {
        for good in ["f", "get_weather", "read-file", "a1", &"n".repeat(64)] {
            assert!(is_valid_tool_name(good), "{good}");
        }
        for bad in [
            "",
            "get weather",
            "say\"hi",
            "read<file>",
            "a.b",
            "naïve",
            "a\u{200b}b",
            &"n".repeat(65),
        ] {
            assert!(!is_valid_tool_name(bad), "{bad}");
        }
    }

    /// A name is refused rather than repaired: the model echoes it back and
    /// the client matches on it, so a sanitizer marker inside one would make
    /// the call unmatchable and the difference invisible.
    #[test]
    fn a_tool_definition_name_outside_the_charset_is_refused_with_the_name() {
        let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"read<file>"}}]}"#;
        let error = ChatCompletionRequest::from_json(body)
            .expect("parses")
            .validate()
            .expect_err("hostile name");
        assert!(
            matches!(&error, ServerError::InvalidToolName { name, param }
                if name == "read<file>" && *param == "tools"),
            "{error:?}"
        );
        assert_eq!(error.status(), 400);
        assert!(error.to_string().contains("read<file>"));
    }

    #[test]
    fn a_definition_with_no_usable_name_is_refused_too() {
        for tools in [
            r#"[{"type":"function","function":{"parameters":{}}}]"#,
            r#"[{"type":"function","function":{"name":42}}]"#,
            r#"[{"type":"function"}]"#,
            r#"[{}]"#,
        ] {
            let body = format!(
                r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"tools":{tools}}}"#
            );
            let error = parse(&body)
                .expect("parses")
                .validate()
                .expect_err("a nameless tool cannot be called back");
            assert!(
                matches!(error, ServerError::InvalidToolName { .. }),
                "{tools}: {error:?}"
            );
        }
    }

    /// The bare shape the renderer also accepts, so the check has to see it.
    #[test]
    fn a_bare_tool_definition_is_named_by_its_own_name_field() {
        let ok = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"name":"read_file","parameters":{}}]}"#;
        parse(ok).expect("parses").validate().expect("valid name");

        let bad = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"name":"read file","parameters":{}}]}"#;
        assert!(matches!(
            parse(bad).expect("parses").validate().expect_err("refused"),
            ServerError::InvalidToolName { .. }
        ));
    }

    #[test]
    fn a_tool_name_echoed_back_on_an_assistant_turn_is_checked_against_messages() {
        let body = r#"{"model":"m","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_1","type":"function",
                 "function":{"name":"say\"hi","arguments":"{}"}}
            ]}
        ]}"#;
        let error = parse(body)
            .expect("parses")
            .validate()
            .expect_err("hostile name");
        assert!(
            matches!(&error, ServerError::InvalidToolName { name, param }
                if name == "say\"hi" && *param == "messages"),
            "{error:?}"
        );
    }

    #[test]
    fn tool_choice_accepts_only_what_the_template_can_express() {
        let base = r#""model":"m","messages":[{"role":"user","content":"hi"}]"#;
        for accepted in ["\"auto\"", "\"none\""] {
            parse(&format!("{{{base},\"tool_choice\":{accepted}}}"))
                .expect("parses")
                .validate()
                .unwrap_or_else(|e| panic!("{accepted} is renderable: {e}"));
        }
        parse(&format!("{{{base}}}"))
            .expect("parses")
            .validate()
            .expect("absent is auto");

        for refused in [
            "\"required\"",
            r#"{"type":"function","function":{"name":"read_file"}}"#,
            "\"banana\"",
        ] {
            let error = parse(&format!("{{{base},\"tool_choice\":{refused}}}"))
                .expect("parses")
                .validate()
                .expect_err("cannot be forced");
            assert!(
                matches!(error, ServerError::UnsupportedToolChoice { .. }),
                "{refused}: {error:?}"
            );
            assert_eq!(error.param(), Some("tool_choice"));
            assert!(error.to_string().contains("auto"), "{refused}");
        }
    }

    /// Nothing in the runtime applies a stop string, so accepting one lets
    /// generation run past where the caller asked it to end with nothing in
    /// the response to say so.
    #[test]
    fn a_non_empty_stop_is_refused_rather_than_dropped() {
        let base = r#""model":"m","messages":[{"role":"user","content":"hi"}]"#;
        for sent in ["\"<end>\"", r#"["a","b"]"#] {
            let error = parse(&format!("{{{base},\"stop\":{sent}}}"))
                .expect("parses")
                .validate()
                .expect_err("no consumer for stop strings");
            assert!(
                matches!(error, ServerError::StopUnsupported),
                "{sent}: {error:?}"
            );
            assert_eq!(error.status(), 400);
            assert_eq!(error.param(), Some("stop"));
        }
        // Absent and empty ask for nothing, so they are served.
        parse(&format!("{{{base}}}"))
            .expect("parses")
            .validate()
            .expect("absent asks for nothing");
        parse(&format!("{{{base},\"stop\":[]}}"))
            .expect("parses")
            .validate()
            .expect("an empty list asks for nothing");
        parse(&format!("{{{base},\"stop\":null}}"))
            .expect("parses")
            .validate()
            .expect("null is absent");
    }

    #[test]
    fn n_above_one_is_rejected() {
        let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"n":2}"#;
        let request = parse(body).expect("valid body");
        assert!(matches!(
            request.validate().expect_err("n > 1 cannot be honoured"),
            ServerError::MultipleChoices { n: 2 }
        ));
    }

    #[test]
    fn n_of_at_most_one_is_accepted() {
        for body in [
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"n":1}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"n":0}"#,
        ] {
            parse(body).expect("valid body").validate().expect("ok");
        }
    }

    #[test]
    fn both_max_token_spellings_are_read_and_the_newer_one_wins() {
        let base = r#""model":"m","messages":[{"role":"user","content":"hi"}]"#;
        let only_old = parse(&format!("{{{base},\"max_tokens\":16}}")).expect("valid");
        assert_eq!(only_old.max_new_tokens(), Some(16));

        let only_new = parse(&format!("{{{base},\"max_completion_tokens\":32}}")).expect("valid");
        assert_eq!(only_new.max_new_tokens(), Some(32));

        let both = parse(&format!(
            "{{{base},\"max_tokens\":16,\"max_completion_tokens\":32}}"
        ))
        .expect("valid");
        assert_eq!(both.max_new_tokens(), Some(32));

        let neither = parse(&format!("{{{base}}}")).expect("valid");
        assert_eq!(neither.max_new_tokens(), None);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let request = parse(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "presence_penalty":0.5,"logprobs":true,"future_knob":{"a":[1]}}"#,
        )
        .expect("unknown fields must not fail the request");
        request.validate().expect("still valid");
    }

    #[test]
    fn malformed_json_is_a_typed_error() {
        let error = parse("{not json").expect_err("malformed");
        assert!(matches!(error, ServerError::MalformedJson(_)));
        assert_eq!(error.status(), 400);
    }

    #[test]
    fn stop_accepts_a_string_or_an_array() {
        let base = r#""model":"m","messages":[{"role":"user","content":"hi"}]"#;
        let one = parse(&format!("{{{base},\"stop\":\"<end>\"}}")).expect("valid");
        assert_eq!(one.stop_sequences(), ["<end>"]);

        let many = parse(&format!("{{{base},\"stop\":[\"a\",\"b\"]}}")).expect("valid");
        assert_eq!(many.stop_sequences(), ["a", "b"]);

        let none = parse(&format!("{{{base}}}")).expect("valid");
        assert!(none.stop_sequences().is_empty());
    }

    #[test]
    fn stream_and_include_usage_default_to_false() {
        let base = r#""model":"m","messages":[{"role":"user","content":"hi"}]"#;
        let plain = parse(&format!("{{{base}}}")).expect("valid");
        assert!(!plain.stream());
        assert!(!plain.include_usage());

        let streaming = parse(&format!(
            "{{{base},\"stream\":true,\"stream_options\":{{\"include_usage\":true}}}}"
        ))
        .expect("valid");
        assert!(streaming.stream());
        assert!(streaming.include_usage());

        // Streaming without the option must not opt in to the usage chunk.
        let bare_stream = parse(&format!("{{{base},\"stream\":true}}")).expect("valid");
        assert!(bare_stream.stream());
        assert!(!bare_stream.include_usage());
    }

    /// A later wave pairs tool results with their calls by id, so both halves
    /// have to survive parsing untouched.
    #[test]
    fn tool_calls_and_tool_call_id_round_trip_without_loss() {
        let body = r#"{
            "model":"m",
            "messages":[
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_abc","type":"function",
                     "function":{"name":"get_weather","arguments":"{\"city\":\"Cairo\"}"}}
                ]},
                {"role":"tool","tool_call_id":"call_abc","content":"31C","name":"get_weather"}
            ],
            "tools":[{"type":"function","function":{"name":"get_weather","strict":true}}],
            "tool_choice":"auto"
        }"#;
        let request = ChatCompletionRequest::from_json(body).expect("valid body");

        let calls = request.messages[1]
            .tool_calls
            .as_ref()
            .expect("tool_calls preserved");
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].kind, "function");
        assert_eq!(calls[0].function.name, "get_weather");
        // Arguments stay the model's exact string, not a re-encoding of it.
        assert_eq!(calls[0].function.arguments, r#"{"city":"Cairo"}"#);

        assert_eq!(request.messages[2].role, MessageRole::Tool);
        assert_eq!(
            request.messages[2].tool_call_id.as_deref(),
            Some("call_abc")
        );
        assert_eq!(request.messages[2].name.as_deref(), Some("get_weather"));
        assert_eq!(request.messages[2].text().expect("text"), "31C");

        // `tools` keeps fields no typed schema here would know about.
        let tools = request.tools.as_ref().expect("tools preserved");
        assert_eq!(tools[0]["function"]["strict"], true);
        assert_eq!(request.tool_choice, Some(serde_json::json!("auto")));

        // And the whole thing survives a serialize/parse cycle.
        let round_tripped: ChatCompletionRequest =
            serde_json::from_str(&serde_json::to_string(&request).expect("serializes"))
                .expect("re-parses");
        assert_eq!(round_tripped, request);
    }

    #[test]
    fn every_role_parses() {
        for (wire, role) in [
            ("system", MessageRole::System),
            ("user", MessageRole::User),
            ("assistant", MessageRole::Assistant),
            ("tool", MessageRole::Tool),
        ] {
            let body = format!(r#"{{"model":"m","messages":[{{"role":"{wire}","content":"x"}}]}}"#);
            let request = parse(&body).expect("valid role");
            assert_eq!(request.messages[0].role, role);
            assert_eq!(role.as_str(), wire);
        }
        assert!(matches!(
            parse(r#"{"model":"m","messages":[{"role":"wizard","content":"x"}]}"#)
                .expect_err("unknown role"),
            ServerError::MalformedJson(_)
        ));
    }
}
