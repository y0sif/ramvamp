//! Turning a request's `messages` into the token ids the engine prefills.
//!
//! Two steps, kept separate on purpose. [`chat_messages`] maps the wire
//! messages onto `ramvamp_core`'s [`ChatMessage`] and is pure, so every rule it
//! enforces is unit-testable without a model on disk. [`build`] is the thin
//! wrapper that hands the result to the tokenizer.
//!
//! # Why the sanitized encoder
//!
//! `RvmpTokenizer::encode_chat` is reference-faithful by recorded decision: a
//! literal `<|im_start|>` inside message content encodes to the *real* control
//! id, exactly as `transformers` and `llama.cpp` do, which lets any string
//! that reaches this server fabricate a turn the model then obeys. That is
//! defensible for a local CLI where the operator types every byte. It is not
//! defensible here — the server's input is a socket. So this module uses
//! [`RvmpTokenizer::encode_chat_with_tools_sanitized`] and nothing else; the
//! sanitizer breaks added-token literals with a zero-width marker, so a user
//! asking "what does `<|im_start|>` mean?" still sees the literal in the
//! transcript but cannot forge a turn with it. With tools in play the same
//! marker covers the definitions (rewritten as JSON trees, so the serialized
//! schema cannot be corrupted) and each call's arguments.
//!
//! Tool *names* are the exception, and they are validated in
//! [`crate::request`] rather than repaired here — see
//! [`is_valid_tool_name`](crate::request::is_valid_tool_name) for why a
//! zero-width marker inside a name would be silent corruption.
//!
//! # What is not here
//!
//! Longest-common-prefix matching against the live KV cache. That needs the
//! cache, so it belongs to the engine; this module hands over the full id
//! sequence and the mapped messages and lets the engine decide how much of it
//! is already prefilled.

use ramvamp_core::tokenizer::{ChatMessage, Role, RvmpTokenizer, ToolArguments, ToolCall};

use crate::error::ServerError;
use crate::request::{ChatCompletionRequest, Message, MessageRole};

/// A prompt ready for the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    /// The rendered, sanitized ChatML prompt as token ids, ending with the
    /// generation prompt so the model's reply begins immediately.
    pub token_ids: Vec<u32>,
    /// The same conversation as core chat messages, *before* sanitization.
    ///
    /// Carried alongside the ids because the engine keeps a transcript across
    /// turns: re-deriving it from the ids would mean decoding, and decoding a
    /// prompt back into turns is not something the format supports.
    pub messages: Vec<ChatMessage>,
}

impl Prompt {
    /// How many tokens this prompt occupies.
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Whether the prompt is empty. Only reachable from an empty conversation
    /// with no generation prompt, which [`build`] never produces.
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

/// Map the wire role onto the renderer's role.
///
/// Total: every role the wire format has, the renderer has. `tool` is not a
/// ChatML marker — the template wraps tool results in a `user` turn as
/// `<tool_response>` blocks, and consecutive ones share that turn — but that
/// is the renderer's business, not this mapping's.
fn role_of(message: &Message) -> Role {
    match message.role {
        MessageRole::System => Role::System,
        MessageRole::User => Role::User,
        MessageRole::Assistant => Role::Assistant,
        MessageRole::Tool => Role::Tool,
    }
}

/// Map one wire call onto the renderer's.
///
/// The wire `arguments` is a **string** — that is what the OpenAI schema says
/// and what every client sends — so it becomes [`ToolArguments::Raw`] and is
/// spliced into the prompt verbatim. Parsing it and re-serializing would
/// rewrite the model's own bytes (key order, spacing) and would turn the
/// commonplace case of a model emitting slightly invalid JSON into a 400 on
/// the *next* turn, long after the reply the client already accepted.
///
/// `id` and `type` are dropped: the template has no slot for either, so
/// rendering them would put text in the prompt the reference never produces.
/// The client keeps them and matches results by them; nothing here needs to.
fn tool_call_of(call: &crate::request::ToolCall) -> ToolCall {
    ToolCall {
        name: call.function.name.clone(),
        arguments: ToolArguments::Raw(call.function.arguments.clone()),
    }
}

/// Map a request's messages onto core chat messages.
///
/// Flattens both content forms, refuses non-text parts, and carries tool
/// calls through. `name` and `tool_call_id` are dropped: the pinned template
/// has no slot for either, so rendering them would put text into the turn that
/// the reference template never produces. Dropping `tool_call_id` is why the
/// order of `tool` messages is the order the client sent — the template pairs
/// results with calls positionally, exactly as the reference does.
pub fn chat_messages(messages: &[Message]) -> Result<Vec<ChatMessage>, ServerError> {
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        message.check_tool_calls()?;
        out.push(ChatMessage {
            role: role_of(message),
            content: message.text()?,
            tool_calls: message.calls().iter().map(tool_call_of).collect(),
        });
    }
    Ok(out)
}

/// Build the prompt for a validated request.
///
/// Assumes [`ChatCompletionRequest::validate`] has already run; it re-checks
/// nothing it cannot cheaply re-check, but every rule it does enforce is
/// enforced here too, so calling it directly is still safe. That includes the
/// tool-name charset, checked here as well as in `validate` because this is
/// the function that renders the names.
pub fn build(
    tokenizer: &RvmpTokenizer,
    request: &ChatCompletionRequest,
) -> Result<Prompt, ServerError> {
    if request.messages.is_empty() {
        return Err(ServerError::EmptyMessages);
    }
    request.check_tools()?;
    let messages = chat_messages(&request.messages)?;
    let token_ids = tokenizer.encode_chat_with_tools_sanitized(&messages, request.tools(), true)?;
    Ok(Prompt {
        token_ids,
        messages,
    })
}

/// Check that a prompt plus its reply reservation fits `limit`, returning the
/// positions left over.
///
/// `max_new` is *reserved*, not merely hoped for: the KV cache is sized at the
/// context cap and a reply that ran into the end of it would fail mid-token, so
/// a request that could overrun is refused before the session is claimed rather
/// than halfway through a stream the client has already started rendering.
pub fn check_context(
    prompt_tokens: usize,
    max_new: usize,
    limit: usize,
) -> Result<usize, ServerError> {
    // Saturating, not wrapping: absurd arguments must report a total that
    // does not fit, never one that wraps around into a total that does.
    let requested = prompt_tokens.saturating_add(max_new);
    if requested > limit {
        return Err(ServerError::ContextOverflow { requested, limit });
    }
    Ok(limit - requested)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{Content, ContentPart, FunctionCall, ToolCall};
    use ramvamp_core::tokenizer::ToolCall as CoreToolCall;

    fn message(role: MessageRole, content: &str) -> Message {
        Message::new(role, content)
    }

    #[test]
    fn roles_map_onto_the_renderer() {
        let mapped = chat_messages(&[
            message(MessageRole::System, "be terse"),
            message(MessageRole::User, "hi"),
            message(MessageRole::Assistant, "hello"),
        ])
        .expect("supported roles");
        assert_eq!(
            mapped,
            vec![
                ChatMessage::system("be terse"),
                ChatMessage::user("hi"),
                ChatMessage::assistant("hello"),
            ]
        );
    }

    #[test]
    fn array_form_content_survives_the_mapping() {
        let mut message = message(MessageRole::User, "");
        message.content = Some(Content::Parts(vec![
            ContentPart::text("what is "),
            ContentPart::text("2+2?"),
        ]));
        let mapped = chat_messages(&[message]).expect("text parts");
        assert_eq!(mapped, vec![ChatMessage::user("what is 2+2?")]);
    }

    #[test]
    fn a_non_text_part_stops_the_mapping() {
        let mut message = message(MessageRole::User, "");
        message.content = Some(Content::Parts(vec![ContentPart {
            kind: "input_audio".to_owned(),
            text: None,
        }]));
        assert!(matches!(
            chat_messages(&[message]).expect_err("refused"),
            ServerError::UnsupportedContentPart { .. }
        ));
    }

    /// The template renders a tool result inside a `user` turn; folding it
    /// into one *here* would be the same text arrived at by lying about the
    /// role, and the renderer could no longer group consecutive results.
    #[test]
    fn a_tool_message_maps_onto_the_tool_role() {
        let mut tool = message(MessageRole::Tool, "31C");
        tool.tool_call_id = Some("call_abc".to_owned());
        assert_eq!(
            chat_messages(&[tool]).expect("the renderer has a tool branch"),
            vec![ChatMessage::tool("31C")]
        );
    }

    #[test]
    fn assistant_tool_calls_reach_the_renderer_with_their_arguments_verbatim() {
        // Deliberately not canonical JSON: spacing and key order are the
        // model's own bytes and must survive to the prompt unchanged.
        let arguments = r#"{ "city":"Cairo", "unit" : "C" }"#;
        let mut assistant = message(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![ToolCall {
            id: "call_abc".to_owned(),
            kind: "function".to_owned(),
            function: FunctionCall {
                name: "get_weather".to_owned(),
                arguments: arguments.to_owned(),
            },
        }]);
        let mapped = chat_messages(&[assistant.clone()]).expect("calls are rendered");
        assert_eq!(
            mapped,
            vec![ChatMessage::assistant_calling(
                "",
                vec![CoreToolCall::raw("get_weather", arguments)]
            )]
        );
        // `Raw`, not `Value`: a re-serialization would not round-trip these
        // bytes, and the template splices a string argument in as text.
        assert!(matches!(
            mapped[0].tool_calls[0].arguments,
            ToolArguments::Raw(_)
        ));

        assistant.tool_calls = Some(Vec::new());
        let mapped = chat_messages(&[assistant]).expect("an empty list is not a tool call");
        assert!(mapped[0].tool_calls.is_empty());
    }

    /// The one string in a tools request that is refused instead of
    /// sanitized, because the client matches the reply against it.
    #[test]
    fn a_hostile_call_name_stops_the_mapping() {
        let mut assistant = message(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![ToolCall {
            id: "call_abc".to_owned(),
            kind: "function".to_owned(),
            function: FunctionCall {
                name: "get\"weather".to_owned(),
                arguments: "{}".to_owned(),
            },
        }]);
        assert!(matches!(
            chat_messages(&[assistant]).expect_err("refused, not repaired"),
            ServerError::InvalidToolName { .. }
        ));
    }

    #[test]
    fn name_is_dropped_because_the_template_has_no_slot_for_it() {
        let mut named = message(MessageRole::User, "hi");
        named.name = Some("ada".to_owned());
        assert_eq!(
            chat_messages(&[named]).expect("valid"),
            vec![ChatMessage::user("hi")]
        );
    }

    /// The committed pinned tokenizer, reached across the workspace on
    /// purpose: the acceptance criterion for tool support is a *token count*,
    /// and only the real vocabulary produces one that means anything.
    fn fixture_tokenizer() -> RvmpTokenizer {
        let fixtures =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../core/src/tokenizer/fixtures");
        RvmpTokenizer::load(&fixtures).expect("fixture tokenizer loads")
    }

    fn request(body: &str) -> ChatCompletionRequest {
        let request = ChatCompletionRequest::from_json(body).expect("valid body");
        request.validate().expect("valid request");
        request
    }

    /// The bug this wave exists to fix, stated the only way it is observable
    /// from outside: a request that offers tools must produce a longer prompt
    /// than the same request without them. Both counts were 14 when the
    /// definitions were parsed and dropped.
    #[test]
    fn tool_definitions_lengthen_the_prompt_they_are_sent_with() {
        let tokenizer = fixture_tokenizer();
        let messages = r#""model":"m","messages":[{"role":"user","content":"read main.rs"}]"#;
        let bare = request(&format!("{{{messages}}}"));
        let armed = request(&format!(
            r#"{{{messages},"tools":[{{"type":"function","function":{{
                "name":"read_file","description":"Read a file",
                "parameters":{{"type":"object","properties":{{"path":{{"type":"string"}}}}}}
            }}}}]}}"#
        ));

        let bare = build(&tokenizer, &bare).expect("builds");
        let armed = build(&tokenizer, &armed).expect("builds");
        assert!(
            armed.len() > bare.len(),
            "tools must reach the model: {} with, {} without",
            armed.len(),
            bare.len()
        );

        let rendered = tokenizer.decode(&armed.token_ids, false).expect("decodes");
        assert!(rendered.contains("# Tools"), "{rendered}");
        assert!(rendered.contains("read_file"), "{rendered}");
    }

    /// The other half of a tool turn: the call the model made and the result
    /// the client fed back both have to be in the prompt on the next turn, or
    /// the model re-issues the call it already made.
    #[test]
    fn a_call_and_its_result_both_reach_the_prompt() {
        let tokenizer = fixture_tokenizer();
        let request = request(
            r#"{"model":"m","messages":[
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_abc","type":"function",
                     "function":{"name":"get_weather","arguments":"{\"city\": \"Cairo\"}"}}
                ]},
                {"role":"tool","tool_call_id":"call_abc","content":"31C"}
            ]}"#,
        );
        let prompt = build(&tokenizer, &request).expect("builds");
        let rendered = tokenizer.decode(&prompt.token_ids, false).expect("decodes");
        assert!(rendered.contains("<tool_call>"), "{rendered}");
        // Spliced verbatim, spacing included, rather than re-serialized.
        assert!(rendered.contains(r#"{"city": "Cairo"}"#), "{rendered}");
        assert!(
            rendered.contains("<tool_response>\n31C\n</tool_response>"),
            "{rendered}"
        );
    }

    #[test]
    fn context_check_reports_the_room_left() {
        assert_eq!(check_context(0, 0, 4096).ok(), Some(4096));
        assert_eq!(check_context(100, 128, 4096).ok(), Some(4096 - 228));
        assert_eq!(check_context(4096 - 128, 128, 4096).ok(), Some(0));
    }

    #[test]
    fn context_check_refuses_an_overrun_and_names_both_numbers() {
        let error = check_context(4000, 128, 4096).expect_err("does not fit");
        assert!(matches!(
            error,
            ServerError::ContextOverflow {
                requested: 4128,
                limit: 4096
            }
        ));
        assert_eq!(error.status(), 400);
        let message = error.to_string();
        assert!(
            message.contains("4128") && message.contains("4096"),
            "{message}"
        );
    }

    /// Absurd arguments must report "does not fit", never wrap into a number
    /// that says it does.
    #[test]
    fn context_check_does_not_wrap() {
        assert!(matches!(
            check_context(usize::MAX, 1, 4096).expect_err("cannot fit"),
            ServerError::ContextOverflow {
                requested: usize::MAX,
                ..
            }
        ));
    }
}
