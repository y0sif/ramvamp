//! OpenAI-compatible HTTP server for the ramvamp runtime.
//!
//! The crate is layered so that the wire contract can be tested without a 30B
//! model on disk, and so that the *timing* of a stream can be tested without a
//! socket. Every rule below is asserted by a unit test that runs in
//! milliseconds, and a streaming sequence is pinned byte for byte.
//!
//! # The pure layer: shapes and rules
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
//! * [`toolcall`] — the return leg: recovering `tool_calls` from the
//!   `<tool_call>` text the model writes, preserving the argument bytes so the
//!   client's echo of them re-renders into the block the model emitted. Twice
//!   over, because a stream cannot wait for the whole reply: [`extract`] for
//!   the buffered path and [`CallStream`] for the streaming one, the latter
//!   keyed off the marker *token ids* so a boundary cannot fall between two
//!   deltas.
//! * [`error`] — one typed error per refusal, each knowing its status, body and
//!   headers.
//!
//! # The engine layer
//!
//! * [`engine`] — the [`Engine`] trait, which is the seam a stub token source
//!   substitutes into, plus [`ModelEngine`]: model, forward state and
//!   tokenizer on one thread, with the KV cache reused across requests by
//!   longest-common-prefix match. The trap in that match — and why the match
//!   is against generated ids and never against a re-render of the transcript
//!   — is documented there, because getting it wrong is silent.
//!
//! # The transport layer
//!
//! * [`wire`] — HTTP/1.1 chunked framing written by hand, because
//!   `tiny_http::Response` buffers a stream to completion and hides client
//!   disconnects.
//! * [`http`] — routing, body limits and the accept loop. Binds loopback only.
//!
//! # What it is sized for
//!
//! One session, on loopback, in front of a runtime that decodes at about 2
//! tok/s. That shapes decisions that would look wrong on a hosted API:
//! `n > 1` is refused instead of approximated, a busy server answers 503 rather
//! than 429, and `Retry-After` is a small fiction rather than an honest wait.
//! Each of those has its reasoning recorded next to the code, because each of
//! them looks like a bug until you know why.
//!
//! The same rule governs the parameters this build cannot implement rather
//! than cannot afford: a non-empty `stop` and a `tool_choice` that forces a
//! particular call are 400s, because nothing downstream applies either one and
//! a 200 that quietly ignored them would be indistinguishable from a correct
//! answer. `tools` themselves *are* served — see [`prompt`].

pub mod engine;
pub mod error;
pub mod http;
pub mod prompt;
pub mod request;
pub mod response;
pub mod sse;
pub mod toolcall;
pub mod wire;

pub use engine::{
    Completion, Engine, EngineConfig, ModelEngine, NullSink, Plan, StreamError, TokenSink,
    common_prefix, hush_stream_aborts,
};
pub use error::{ErrorDetail, ErrorResponse, RETRY_HEADERS, ServerError};
pub use http::{
    BODY_BYTES_PER_CONTEXT_TOKEN, MIN_BODY_BYTES, ServeConfig, ServeError,
    max_body_bytes_for_context, serve, serve_with,
};
pub use prompt::{Prompt, check_context};
pub use request::{
    ChatCompletionRequest, Content, ContentPart, Message, MessageRole, StreamOptions,
    StringOrArray, ToolCall,
};
pub use response::{
    ChatCompletion, ChatCompletionChunk, ChunkBuilder, Delta, FinishReason, Health, ModelList,
    ResponseMessage, ToolCallDelta, Usage,
};
pub use sse::{DONE_SENTINEL, sse_data, sse_done, sse_event};
pub use toolcall::{
    CallStream, Extracted, ParsedCall, Step, call_id, extract, wire_call, wire_calls,
};

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
