//! `/v1/chat/completions` response shapes, streaming and not, plus the
//! `/v1/models` and `/health` bodies.
//!
//! The shapes are OpenAI's to the byte, including the parts that look
//! redundant. `object` is a literal discriminant some clients switch on;
//! `finish_reason` is serialized as `null` on non-final chunks rather than
//! omitted, because a client that reads `choices[0].finish_reason` positionally
//! sees a missing key as a malformed chunk. Where a field is genuinely optional
//! (`usage` on a chunk, `tool_calls` on a delta) it is omitted, which is what
//! OpenAI's own stream does for at least one of them and no client objects to.
//!
//! Nothing here reads a clock or a random source: `id` and `created` are
//! parameters. That keeps the crate free of I/O and every assertion below
//! deterministic — a test can pin the exact bytes of a chunk, which is the only
//! way to pin a streaming contract.

use serde::{Deserialize, Serialize};

/// The `object` discriminant of a non-streaming completion.
pub const OBJECT_CHAT_COMPLETION: &str = "chat.completion";

/// The prefix every completion id carries.
///
/// Lives here rather than at the mint site because it is also *read*: a call
/// id is derived from the completion id with this stripped off (see
/// [`crate::toolcall::call_id`]), and the two must not drift apart.
pub const COMPLETION_ID_PREFIX: &str = "chatcmpl-";

/// The `object` discriminant of a streaming chunk.
pub const OBJECT_CHAT_COMPLETION_CHUNK: &str = "chat.completion.chunk";

/// The `object` discriminant of a list body.
pub const OBJECT_LIST: &str = "list";

/// The `object` discriminant of a model card.
pub const OBJECT_MODEL: &str = "model";

/// Why generation stopped, in OpenAI's vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// A stop token or stop sequence ended the reply.
    Stop,
    /// The token budget ran out.
    Length,
    /// The model asked to call a tool.
    ToolCalls,
    /// Output was withheld. Never produced by this build; present because the
    /// value round-trips through client-side transcripts.
    ContentFilter,
}

/// Token accounting for one request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens prefilled, including the ChatML markers this server adds.
    pub prompt_tokens: u32,
    /// Tokens generated, stop token excluded.
    pub completion_tokens: u32,
    /// Their sum.
    pub total_tokens: u32,
}

impl Usage {
    /// Usage with the total derived, so the three numbers cannot disagree.
    ///
    /// Saturating rather than wrapping: an implausible count should report an
    /// implausible total, never a small one.
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }
}

/// The assistant message of a non-streaming completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseMessage {
    /// Always `assistant`.
    pub role: String,
    /// The reply text.
    pub content: Option<String>,
    /// Calls the model requested. Omitted when there are none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::request::ToolCall>>,
}

impl ResponseMessage {
    /// An assistant message carrying `content`.
    pub fn assistant(content: impl Into<String>) -> Self {
        ResponseMessage::calling(content, Vec::new())
    }

    /// An assistant message carrying `content` and the calls it asked for.
    ///
    /// No calls means the key is omitted rather than serialized as `[]`:
    /// clients branch on the field being *present*, and an empty array reads
    /// to some of them as "this turn wants tools" — which then waits forever
    /// for a `tool` message that is never coming.
    ///
    /// `content` stays a `Some`, empty string included, rather than becoming
    /// `null` on a pure tool-call turn. Both are legal and clients accept
    /// either; the empty string is the one that cannot be rendered as the
    /// literal text `None` by a client that formats it naively.
    pub fn calling(content: impl Into<String>, tool_calls: Vec<crate::request::ToolCall>) -> Self {
        ResponseMessage {
            role: crate::request::MessageRole::Assistant.as_str().to_owned(),
            content: Some(content.into()),
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        }
    }
}

/// One choice of a non-streaming completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Choice {
    /// Position in `choices`. Always 0 here: `n > 1` is refused.
    pub index: u32,
    /// The reply.
    pub message: ResponseMessage,
    /// Why it ended.
    pub finish_reason: Option<FinishReason>,
}

/// A non-streaming `chat.completion`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletion {
    /// Completion id, conventionally `chatcmpl-...`.
    pub id: String,
    /// Always [`OBJECT_CHAT_COMPLETION`].
    pub object: String,
    /// Unix seconds.
    pub created: u64,
    /// The model name the client asked for, echoed back.
    pub model: String,
    /// Exactly one choice.
    pub choices: Vec<Choice>,
    /// Token accounting.
    pub usage: Usage,
}

impl ChatCompletion {
    /// The single-choice completion this server always returns.
    pub fn single(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        content: impl Into<String>,
        finish_reason: FinishReason,
        usage: Usage,
    ) -> Self {
        ChatCompletion::from_message(
            id,
            created,
            model,
            ResponseMessage::assistant(content),
            finish_reason,
            usage,
        )
    }

    /// The same, around a message that was assembled elsewhere — a reply that
    /// carries tool calls, whose content and calls come out of one parse.
    pub fn from_message(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        message: ResponseMessage,
        finish_reason: FinishReason,
        usage: Usage,
    ) -> Self {
        ChatCompletion {
            id: id.into(),
            object: OBJECT_CHAT_COMPLETION.to_owned(),
            created,
            model: model.into(),
            choices: vec![Choice {
                index: 0,
                message,
                finish_reason: Some(finish_reason),
            }],
            usage,
        }
    }
}

/// The incremental half of a streaming choice.
///
/// Every field is optional and every absent field is omitted, which is what
/// makes the three chunk shapes distinguishable on the wire: `{"role":...,
/// "content":""}`, `{"content":"..."}`, and `{}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delta {
    /// Sent once, on the first chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// A slice of the reply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool-call fragments. Never produced by this lane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// One choice of a streaming chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkChoice {
    /// Always 0.
    pub index: u32,
    /// What changed.
    pub delta: Delta,
    /// `null` until the final chunk. Serialized even when null: a client
    /// reading the key positionally treats an absent one as a broken chunk.
    pub finish_reason: Option<FinishReason>,
}

/// A streaming `chat.completion.chunk`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Same id for every chunk of one response.
    pub id: String,
    /// Always [`OBJECT_CHAT_COMPLETION_CHUNK`].
    pub object: String,
    /// Unix seconds, same for every chunk of one response.
    pub created: u64,
    /// The model name the client asked for.
    pub model: String,
    /// One choice, or none at all on the usage chunk.
    pub choices: Vec<ChunkChoice>,
    /// Present only on the usage chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Builds the chunks of one streaming response.
///
/// Holds the fields that must be identical across every chunk (`id`,
/// `created`, `model`) so they cannot drift, and names the four chunk shapes
/// so the emission order is readable at the call site:
///
/// 1. [`role`](Self::role) — `delta: {"role":"assistant","content":""}`
/// 2. [`content`](Self::content) — `delta: {"content":"..."}`, repeated
/// 3. [`usage`](Self::usage) — `choices: []` plus `usage`, only when
///    `stream_options.include_usage` was set, and immediately *before* the
///    sentinel rather than after: a client stops reading at `[DONE]`, so usage
///    emitted after it is usage nobody receives
/// 4. [`finish`](Self::finish) — `delta: {}` plus `finish_reason`
///
/// The order on the wire is 1, 2*, 4, 3?, then `data: [DONE]`.
#[derive(Debug, Clone)]
pub struct ChunkBuilder {
    id: String,
    created: u64,
    model: String,
}

impl ChunkBuilder {
    /// A builder for the response identified by `id`.
    pub fn new(id: impl Into<String>, created: u64, model: impl Into<String>) -> Self {
        ChunkBuilder {
            id: id.into(),
            created,
            model: model.into(),
        }
    }

    /// The completion id every chunk carries.
    pub fn id(&self) -> &str {
        &self.id
    }

    fn chunk(&self, choices: Vec<ChunkChoice>, usage: Option<Usage>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: self.id.clone(),
            object: OBJECT_CHAT_COMPLETION_CHUNK.to_owned(),
            created: self.created,
            model: self.model.clone(),
            choices,
            usage,
        }
    }

    /// The opening chunk: role plus empty content.
    ///
    /// `content` is `""` and not omitted because clients that concatenate
    /// `delta.content` unconditionally would otherwise see their first read
    /// fail on a missing key.
    pub fn role(&self) -> ChatCompletionChunk {
        self.chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some(crate::request::MessageRole::Assistant.as_str().to_owned()),
                    content: Some(String::new()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            None,
        )
    }

    /// A content chunk.
    pub fn content(&self, text: impl Into<String>) -> ChatCompletionChunk {
        self.chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(text.into()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            None,
        )
    }

    /// The closing chunk: empty delta plus the reason.
    pub fn finish(&self, reason: FinishReason) -> ChatCompletionChunk {
        self.chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(reason),
            }],
            None,
        )
    }

    /// The usage-only chunk: no choices, one usage object.
    pub fn usage(&self, usage: Usage) -> ChatCompletionChunk {
        self.chunk(Vec::new(), Some(usage))
    }
}

/// One entry of `GET /v1/models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCard {
    /// The model name, as it must be sent back in `model`.
    pub id: String,
    /// Always [`OBJECT_MODEL`].
    pub object: String,
    /// Unix seconds; the install time is the honest value.
    pub created: u64,
    /// Owner. `ramvamp` for a local install — there is no account behind it.
    pub owned_by: String,
}

/// The `GET /v1/models` body.
///
/// A list of one: this process serves the install it was started with, and a
/// client's model picker should show exactly that.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelList {
    /// Always [`OBJECT_LIST`].
    pub object: String,
    /// The models.
    pub data: Vec<ModelCard>,
}

impl ModelList {
    /// The one-model listing.
    pub fn single(id: impl Into<String>, created: u64) -> Self {
        ModelList {
            object: OBJECT_LIST.to_owned(),
            data: vec![ModelCard {
                id: id.into(),
                object: OBJECT_MODEL.to_owned(),
                created,
                owned_by: "ramvamp".to_owned(),
            }],
        }
    }
}

/// The `GET /health` body when the model is up.
///
/// Mirrors llama.cpp's contract — `{"status":"ok"}` on 200, an error body on
/// 503 while loading — because local tooling already probes that endpoint and
/// expects that pair. See [`crate::error::ServerError::ModelLoading`] for the
/// other half.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Health {
    /// `ok`.
    pub status: String,
}

impl Health {
    /// The ready body.
    pub fn ok() -> Self {
        Health {
            status: "ok".to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ServerError;

    fn json(value: &impl Serialize) -> serde_json::Value {
        serde_json::to_value(value).expect("response types always serialize")
    }

    #[test]
    fn completion_object_is_the_literal_discriminant() {
        let completion = ChatCompletion::single(
            "chatcmpl-1",
            1_700_000_000,
            "qwen3-30b-a3b",
            "hi",
            FinishReason::Stop,
            Usage::new(10, 2),
        );
        let value = json(&completion);
        assert_eq!(value["object"], OBJECT_CHAT_COMPLETION);
        assert_eq!(value["object"], "chat.completion");
        assert_eq!(value["id"], "chatcmpl-1");
        assert_eq!(value["created"], 1_700_000_000u64);
        assert_eq!(value["model"], "qwen3-30b-a3b");
        assert_eq!(value["choices"][0]["index"], 0);
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert_eq!(value["choices"][0]["message"]["content"], "hi");
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        assert_eq!(value["usage"]["prompt_tokens"], 10);
        assert_eq!(value["usage"]["completion_tokens"], 2);
        assert_eq!(value["usage"]["total_tokens"], 12);
    }

    #[test]
    fn usage_total_is_derived_and_saturates() {
        assert_eq!(Usage::new(3, 4).total_tokens, 7);
        assert_eq!(Usage::new(u32::MAX, 1).total_tokens, u32::MAX);
    }

    #[test]
    fn finish_reasons_use_openai_spellings() {
        for (reason, wire) in [
            (FinishReason::Stop, "stop"),
            (FinishReason::Length, "length"),
            (FinishReason::ToolCalls, "tool_calls"),
            (FinishReason::ContentFilter, "content_filter"),
        ] {
            assert_eq!(json(&reason), wire);
        }
    }

    /// The whole streaming contract in one place: shape, order, and the exact
    /// keys present in each delta.
    #[test]
    fn streaming_chunk_sequence_is_exact() {
        let builder = ChunkBuilder::new("chatcmpl-x", 42, "m");

        let role = json(&builder.role());
        assert_eq!(role["object"], "chat.completion.chunk");
        assert_eq!(role["id"], "chatcmpl-x");
        assert_eq!(role["created"], 42);
        assert_eq!(role["model"], "m");
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(role["choices"][0]["delta"]["content"], "");
        assert!(role["choices"][0]["finish_reason"].is_null());
        assert!(
            role.get("usage").is_none(),
            "no usage outside the usage chunk"
        );

        let content = json(&builder.content("hel"));
        assert_eq!(content["choices"][0]["delta"]["content"], "hel");
        assert!(
            content["choices"][0]["delta"].get("role").is_none(),
            "role is sent once, on the first chunk only"
        );
        assert!(content["choices"][0]["finish_reason"].is_null());

        let finish = json(&builder.finish(FinishReason::Stop));
        assert_eq!(
            finish["choices"][0]["delta"],
            serde_json::json!({}),
            "the final delta is empty"
        );
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");

        let usage = json(&builder.usage(Usage::new(7, 3)));
        assert_eq!(
            usage["choices"],
            serde_json::json!([]),
            "the usage chunk carries no choices"
        );
        assert_eq!(usage["usage"]["prompt_tokens"], 7);
        assert_eq!(usage["usage"]["completion_tokens"], 3);
        assert_eq!(usage["usage"]["total_tokens"], 10);

        // Every chunk of one response shares id, object, created and model.
        for chunk in [role, content, finish, usage] {
            assert_eq!(chunk["id"], "chatcmpl-x");
            assert_eq!(chunk["object"], "chat.completion.chunk");
            assert_eq!(chunk["created"], 42);
            assert_eq!(chunk["model"], "m");
        }
    }

    /// `finish_reason` must be an explicit `null`, not an absent key.
    #[test]
    fn non_final_chunks_serialize_a_null_finish_reason() {
        let builder = ChunkBuilder::new("id", 0, "m");
        for chunk in [builder.role(), builder.content("x")] {
            let text = serde_json::to_string(&chunk).expect("serializes");
            assert!(text.contains(r#""finish_reason":null"#), "{text}");
        }
    }

    #[test]
    fn chunks_round_trip_through_json() -> Result<(), ServerError> {
        let builder = ChunkBuilder::new("id", 1, "m");
        for chunk in [
            builder.role(),
            builder.content("text"),
            builder.finish(FinishReason::Length),
            builder.usage(Usage::new(1, 2)),
        ] {
            let text = serde_json::to_string(&chunk)?;
            let back: ChatCompletionChunk = serde_json::from_str(&text)?;
            assert_eq!(back, chunk);
        }
        Ok(())
    }

    #[test]
    fn models_listing_is_a_list_of_one() {
        let value = json(&ModelList::single("qwen3-30b-a3b", 1_700_000_000));
        assert_eq!(value["object"], "list");
        assert_eq!(value["data"][0]["id"], "qwen3-30b-a3b");
        assert_eq!(value["data"][0]["object"], "model");
        assert_eq!(value["data"][0]["created"], 1_700_000_000u64);
        assert_eq!(value["data"][0]["owned_by"], "ramvamp");
        assert_eq!(
            value["data"].as_array().map(Vec::len),
            Some(1),
            "one process serves one install"
        );
    }

    #[test]
    fn health_ok_matches_the_llama_cpp_contract() {
        assert_eq!(json(&Health::ok()), serde_json::json!({"status": "ok"}));
        // The loading half is an error body, not a `status` field.
        assert_eq!(ServerError::ModelLoading.status(), 503);
    }
}
