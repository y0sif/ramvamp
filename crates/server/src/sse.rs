//! Server-sent-event payload construction.
//!
//! This is the payload half only — the bytes that go *inside* the response
//! body. Chunked transfer framing, headers and flushing belong to the
//! transport lane; keeping them apart is what lets the exact bytes of a stream
//! be asserted in a unit test.
//!
//! Three details are contract, not style, and each has broken a client before:
//!
//! * **The terminator is `\n\n`, not `\r\n\r\n`.** The EventSource grammar
//!   accepts either, but a `\r` that survives into a hand-rolled line splitter
//!   ends up glued to the JSON, and the parse failure is reported against the
//!   model's output rather than against the framing.
//! * **`[DONE]` is a bare sentinel.** The literal bytes are
//!   `data: [DONE]\n\n`. It is not `"[DONE]"`, not `{"done":true}`; a client
//!   compares the data field to the five characters and hangs if they differ.
//! * **A payload never contains a literal newline.** A `\n` inside the data
//!   field ends the event, so a pretty-printed body silently splits one chunk
//!   into several malformed ones. Serialization here always goes through
//!   [`serde_json::to_string`], which escapes newlines inside strings and
//!   emits none of its own.

use serde::Serialize;

use crate::error::ServerError;

/// The sentinel data field that ends an OpenAI-compatible stream.
pub const DONE_SENTINEL: &str = "[DONE]";

/// Wrap an already-serialized JSON payload as one SSE `data:` event.
///
/// `json` must be a single line; [`sse_event`] is the way to guarantee that.
pub fn sse_data(json: &str) -> String {
    format!("data: {json}\n\n")
}

/// Serialize `value` and wrap it as one SSE event.
pub fn sse_event<T: Serialize>(value: &T) -> Result<String, ServerError> {
    Ok(sse_data(&serde_json::to_string(value)?))
}

/// The stream terminator: `data: [DONE]\n\n`.
pub fn sse_done() -> String {
    sse_data(DONE_SENTINEL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::{ChunkBuilder, FinishReason, Usage};

    #[test]
    fn an_event_ends_with_two_newlines_and_no_carriage_return() {
        let event = sse_data(r#"{"a":1}"#);
        assert_eq!(event, "data: {\"a\":1}\n\n");
        assert!(event.ends_with("\n\n"));
        assert!(!event.contains('\r'));
    }

    /// The sentinel is bare text, not JSON. Byte-for-byte.
    #[test]
    fn done_is_a_bare_sentinel() {
        assert_eq!(sse_done(), "data: [DONE]\n\n");
        assert_eq!(sse_done().as_bytes(), b"data: [DONE]\n\n");
        assert!(!sse_done().contains('"'));
        assert!(serde_json::from_str::<serde_json::Value>(DONE_SENTINEL).is_err());
    }

    /// A newline in the reply must be escaped into the JSON string, never
    /// emitted raw — a raw one would end the event early and split one chunk
    /// into two malformed ones.
    #[test]
    fn content_with_newlines_serializes_to_a_single_line() -> Result<(), ServerError> {
        let builder = ChunkBuilder::new("id", 0, "m");
        let event = sse_event(&builder.content("line one\nline two\r\nline three"))?;

        let payload = event.strip_suffix("\n\n").expect("event terminator");
        assert!(
            !payload.contains('\n') && !payload.contains('\r'),
            "payload must be one line: {payload:?}"
        );
        assert!(payload.contains(r"line one\nline two\r\nline three"));

        // And a client that splits on the terminator gets exactly one event
        // whose data field parses back to the original text.
        let events: Vec<&str> = event.split("\n\n").filter(|s| !s.is_empty()).collect();
        assert_eq!(events.len(), 1);
        let data = events[0].strip_prefix("data: ").expect("data field");
        let parsed: serde_json::Value = serde_json::from_str(data)?;
        assert_eq!(
            parsed["choices"][0]["delta"]["content"],
            "line one\nline two\r\nline three"
        );
        Ok(())
    }

    #[test]
    fn no_chunk_shape_ever_emits_a_multi_line_payload() -> Result<(), ServerError> {
        let builder = ChunkBuilder::new("chatcmpl-1", 1, "m");
        for chunk in [
            builder.role(),
            builder.content("a\nb"),
            builder.finish(FinishReason::Stop),
            builder.usage(Usage::new(1, 1)),
        ] {
            let event = sse_event(&chunk)?;
            assert_eq!(event.matches("\n\n").count(), 1, "{event:?}");
            assert!(event.ends_with("\n\n"));
            assert_eq!(
                event.trim_end_matches('\n').matches('\n').count(),
                0,
                "{event:?}"
            );
        }
        Ok(())
    }

    /// The whole wire sequence, spelled out: role, content, finish, usage,
    /// sentinel — with usage immediately before `[DONE]`, since a client stops
    /// reading at the sentinel.
    #[test]
    fn a_full_stream_reads_back_in_order() -> Result<(), ServerError> {
        let builder = ChunkBuilder::new("chatcmpl-1", 7, "m");
        let mut body = String::new();
        body.push_str(&sse_event(&builder.role())?);
        body.push_str(&sse_event(&builder.content("he"))?);
        body.push_str(&sse_event(&builder.content("llo"))?);
        body.push_str(&sse_event(&builder.finish(FinishReason::Stop))?);
        body.push_str(&sse_event(&builder.usage(Usage::new(5, 2)))?);
        body.push_str(&sse_done());

        let fields: Vec<&str> = body
            .split("\n\n")
            .filter(|s| !s.is_empty())
            .map(|event| event.strip_prefix("data: ").unwrap_or(event))
            .collect();
        assert_eq!(fields.len(), 6);
        assert_eq!(fields[5], DONE_SENTINEL);

        let chunks: Vec<serde_json::Value> = fields[..5]
            .iter()
            .map(|f| serde_json::from_str(f))
            .collect::<Result<_, _>>()?;
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "");
        assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "he");
        assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "llo");
        assert_eq!(chunks[3]["choices"][0]["delta"], serde_json::json!({}));
        assert_eq!(chunks[3]["choices"][0]["finish_reason"], "stop");
        assert_eq!(chunks[4]["choices"], serde_json::json!([]));
        assert_eq!(chunks[4]["usage"]["total_tokens"], 7);

        // Concatenating the content deltas reconstructs the reply.
        let text: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, "hello");
        Ok(())
    }
}
