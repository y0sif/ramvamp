//! Turning a request's `messages` into the token ids the engine prefills.
//!
//! Two steps, kept separate on purpose. [`chat_messages`] maps the wire
//! messages onto `ramvamp_core`'s [`ChatMessage`] and is pure, so every rule it
//! enforces is unit-testable without a model on disk. [`build`] is the thin
//! wrapper that hands the result to the tokenizer.
//!
//! # Why `encode_chat_sanitized`
//!
//! `RvmpTokenizer::encode_chat` is reference-faithful by recorded decision: a
//! literal `<|im_start|>` inside message content encodes to the *real* control
//! id, exactly as `transformers` and `llama.cpp` do, which lets any string
//! that reaches this server fabricate a turn the model then obeys. That is
//! defensible for a local CLI where the operator types every byte. It is not
//! defensible here — the server's input is a socket. So this module uses
//! [`RvmpTokenizer::encode_chat_sanitized`] and nothing else; the sanitizer
//! breaks added-token literals with a zero-width marker, so a user asking
//! "what does `<|im_start|>` mean?" still sees the literal in the transcript
//! but cannot forge a turn with it.
//!
//! # What is not here
//!
//! Longest-common-prefix matching against the live KV cache. That needs the
//! cache, so it belongs to the engine; this module hands over the full id
//! sequence and the mapped messages and lets the engine decide how much of it
//! is already prefilled.

use ramvamp_core::tokenizer::{ChatMessage, Role, RvmpTokenizer};

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
/// `tool` has no target: the vendored 2507-Instruct template deliberately
/// omits the tool branch, so there is no faithful rendering to fall back on.
/// Refused rather than folded into a user turn, which would put words in the
/// user's mouth and change what the model believes it was told.
fn role_of(message: &Message) -> Result<Role, ServerError> {
    match message.role {
        MessageRole::System => Ok(Role::System),
        MessageRole::User => Ok(Role::User),
        MessageRole::Assistant => Ok(Role::Assistant),
        MessageRole::Tool => Err(ServerError::ToolsUnsupported),
    }
}

/// Map a request's messages onto core chat messages.
///
/// Flattens both content forms, refuses non-text parts, and refuses anything
/// tool-shaped. `name` is dropped: the pinned template has no slot for it, so
/// rendering it would put text into the turn that the reference template never
/// produces.
pub fn chat_messages(messages: &[Message]) -> Result<Vec<ChatMessage>, ServerError> {
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        if message.tool_calls.as_ref().is_some_and(|c| !c.is_empty()) {
            return Err(ServerError::ToolsUnsupported);
        }
        out.push(ChatMessage::new(role_of(message)?, message.text()?));
    }
    Ok(out)
}

/// Build the prompt for a validated request.
///
/// Assumes [`ChatCompletionRequest::validate`] has already run; it re-checks
/// nothing it cannot cheaply re-check, but every rule it does enforce is
/// enforced here too, so calling it directly is still safe.
pub fn build(
    tokenizer: &RvmpTokenizer,
    request: &ChatCompletionRequest,
) -> Result<Prompt, ServerError> {
    if request.messages.is_empty() {
        return Err(ServerError::EmptyMessages);
    }
    let messages = chat_messages(&request.messages)?;
    let token_ids = tokenizer.encode_chat_sanitized(&messages, true)?;
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

    #[test]
    fn tool_messages_are_refused_rather_than_folded_into_a_user_turn() {
        let mut tool = message(MessageRole::Tool, "31C");
        tool.tool_call_id = Some("call_abc".to_owned());
        assert!(matches!(
            chat_messages(&[tool]).expect_err("no tool branch in the template"),
            ServerError::ToolsUnsupported
        ));
    }

    #[test]
    fn assistant_tool_calls_are_refused_but_an_empty_list_is_not() {
        let mut assistant = message(MessageRole::Assistant, "");
        assistant.tool_calls = Some(vec![ToolCall {
            id: "call_abc".to_owned(),
            kind: "function".to_owned(),
            function: FunctionCall {
                name: "get_weather".to_owned(),
                arguments: "{}".to_owned(),
            },
        }]);
        assert!(matches!(
            chat_messages(&[assistant.clone()]).expect_err("refused"),
            ServerError::ToolsUnsupported
        ));

        assistant.tool_calls = Some(Vec::new());
        chat_messages(&[assistant]).expect("an empty list is not a tool call");
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
