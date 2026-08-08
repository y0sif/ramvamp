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

use crate::error::ServerError;

/// Who is speaking.
///
/// `tool` parses and round-trips even though the vendored template cannot
/// render it: a later wave re-sorts tool results by `tool_call_id`, and data
/// that was never parsed cannot be re-sorted. [`crate::prompt`] is where the
/// refusal happens, not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// Instructions to the model.
    System,
    /// A human turn.
    User,
    /// A model turn.
    Assistant,
    /// The result of a tool call.
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
/// Preserved verbatim through parsing so a later wave can pair each result
/// with its call; nothing in this lane reads it.
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
    /// Held as raw JSON rather than a typed schema: this lane does not render
    /// tools, and giving them a Rust shape now would either drop fields the
    /// shape does not know about or freeze a contract before there is a
    /// consumer to hold it to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    /// Tool selection policy, preserved verbatim for the same reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
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
        // Tool definitions are parsed and preserved, but nothing renders them
        // into the prompt: the vendored ChatML template has no tool branch and
        // core's `ChatMessage` has no tool role. Answering 200 while dropping
        // them is the failure this crate refuses everywhere else, and it is
        // worse here than elsewhere because the reply looks reasonable: the
        // model simply says it cannot read files, and `prompt_tokens` is
        // identical with and without the `tools` array. Measured 2026-08-08,
        // 14 tokens either way. Refuse until the template can carry them.
        if self.tools.as_ref().is_some_and(|tools| !tools.is_empty()) {
            return Err(ServerError::ToolsUnsupported);
        }
        for message in &self.messages {
            // Flattening is the check: it is the only thing that inspects
            // every part, and doing it here means a bad part is refused before
            // the session is claimed rather than mid-prefill.
            message.text()?;
        }
        Ok(())
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

    #[test]
    fn a_request_carrying_tools_is_refused_rather_than_answered_without_them() {
        // Accepting these and dropping them returns a reply that reads fine
        // and was produced from a prompt the caller never sent.
        let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}"#;
        let req = ChatCompletionRequest::from_json(body).expect("parses");
        assert!(matches!(req.validate(), Err(ServerError::ToolsUnsupported)));
    }

    #[test]
    fn an_empty_tools_array_is_not_a_refusal() {
        let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[]}"#;
        let req = ChatCompletionRequest::from_json(body).expect("parses");
        req.validate().expect("an empty list asks for nothing");
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
