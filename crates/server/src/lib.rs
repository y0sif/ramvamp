//! OpenAI-compatible protocol layer for the ramvamp runtime.
//!
//! This crate is the *shapes and rules* of the HTTP surface and nothing else.
//! It opens no socket, reads no file, and loads no model; a transport crate
//! layers a server over it and an engine layers generation under it. The split
//! is what makes the wire contract testable: every rule below is asserted by a
//! unit test that runs in milliseconds without a 30B model on disk, and a
//! streaming sequence can be pinned byte for byte.
//!
//! # The surface
//!
//! * [`request`] — the `POST /v1/chat/completions` body, including the
//!   array-form `content` that clients send routinely and a naive server
//!   silently discards.
//! * [`response`] — `chat.completion`, `chat.completion.chunk`, the
//!   `/v1/models` listing and the `/health` body.
//! * [`sse`] — SSE payload construction: `data: {json}\n\n` and the bare
//!   `[DONE]` sentinel.
//! * [`prompt`] — mapping messages onto the vendored ChatML renderer, through
//!   the *sanitizing* encoder because the input is untrusted.
//! * [`error`] — one typed error per refusal, each knowing its status, body and
//!   headers.
//!
//! # What it is sized for
//!
//! One session, on loopback, in front of a runtime that decodes at about 2
//! tok/s. That shapes decisions that would look wrong on a hosted API:
//! `n > 1` is refused instead of approximated, a busy server answers 503 rather
//! than 429, and `Retry-After` is a small fiction rather than an honest wait.
//! Each of those has its reasoning recorded next to the code, because each of
//! them looks like a bug until you know why.

pub mod error;
pub mod prompt;
pub mod request;
pub mod response;
pub mod sse;

pub use error::{ErrorDetail, ErrorResponse, RETRY_HEADERS, ServerError};
pub use prompt::{Prompt, check_context};
pub use request::{
    ChatCompletionRequest, Content, ContentPart, Message, MessageRole, StreamOptions,
    StringOrArray, ToolCall,
};
pub use response::{
    ChatCompletion, ChatCompletionChunk, ChunkBuilder, Delta, FinishReason, Health, ModelList,
    Usage,
};
pub use sse::{DONE_SENTINEL, sse_data, sse_done, sse_event};

#[cfg(test)]
mod tests {
    use super::*;

    /// One request through every pure stage: parse, validate, map, respond.
    /// The tokenizer step is the only thing missing, and it is the only thing
    /// that needs an install on disk.
    #[test]
    fn a_request_walks_the_whole_pure_path() -> Result<(), ServerError> {
        let request = ChatCompletionRequest::from_json(
            r#"{"model":"qwen3-30b-a3b",
                "messages":[
                    {"role":"system","content":"be terse"},
                    {"role":"user","content":[{"type":"text","text":"2+2?"}]}
                ],
                "stream":true,
                "stream_options":{"include_usage":true},
                "max_completion_tokens":8}"#,
        )?;
        request.validate()?;
        assert!(request.stream());
        assert!(request.include_usage());
        assert_eq!(request.max_new_tokens(), Some(8));

        let mapped = prompt::chat_messages(&request.messages)?;
        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[1].content, "2+2?");

        check_context(64, 8, 4096)?;

        let builder = ChunkBuilder::new("chatcmpl-1", 0, &request.model);
        let stream = format!(
            "{}{}{}{}{}",
            sse_event(&builder.role())?,
            sse_event(&builder.content("4"))?,
            sse_event(&builder.finish(FinishReason::Stop))?,
            sse_event(&builder.usage(Usage::new(20, 1)))?,
            sse_done(),
        );
        assert!(stream.ends_with("data: [DONE]\n\n"));
        assert_eq!(stream.matches("\n\n").count(), 5);
        Ok(())
    }
}
