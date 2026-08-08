//! Typed request/response errors and their HTTP mapping.
//!
//! Every failure this crate can report is one [`ServerError`] variant, and a
//! variant knows its own status, body and headers. The transport lane never
//! decides a status code: it asks the error.
//!
//! The body shape is OpenAI's, because that is what the clients parse:
//!
//! ```json
//! {"error":{"message":"...","type":"...","param":null,"code":"..."}}
//! ```
//!
//! `message` is a plain `String`, never `null` and never absent — at least one
//! widely used client models the field as required and fails to deserialize
//! the error before it can report it, which turns a clear 400 into a parse
//! crash. `param` and `code` are always *present* keys, `null` when they do
//! not apply, for the same reason.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use ramvamp_core::generate::GenerateError;
use ramvamp_core::model::ForwardError;
use ramvamp_core::tokenizer::TokenizerError;

/// OpenAI's `invalid_request_error`: the client sent something this server
/// cannot honour, and retrying the same bytes will fail the same way.
pub const TYPE_INVALID_REQUEST: &str = "invalid_request_error";

/// OpenAI's `server_error`: the request was fine, this process is not (yet)
/// able to serve it.
pub const TYPE_SERVER: &str = "server_error";

/// Headers a client should see on a 503, exposed as data because building
/// them is the transport lane's job and choosing them is this one's.
///
/// `Retry-After` is deliberately **1**, not the true wait.
///
/// The true wait for a busy single-session server is the remainder of the
/// in-flight reply, which at ~2 tok/s decode is routinely minutes. Sending
/// that honest number disables retrying altogether: openai-python's
/// `_should_retry` inspects `Retry-After` first and treats any value above
/// 120 seconds as "do not retry", returning the error to the caller instead
/// of waiting. A small number keeps the client in its own backoff loop, which
/// is the behaviour we actually want. Do not "fix" this to a truthful value.
pub const RETRY_HEADERS: &[(&str, &str)] = &[
    ("Retry-After", "1"),
    ("retry-after-ms", "500"),
    ("x-should-retry", "true"),
];

/// The `error` object inside an error body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Human-readable explanation. Always a string.
    pub message: String,
    /// Error class, one of [`TYPE_INVALID_REQUEST`] / [`TYPE_SERVER`].
    #[serde(rename = "type")]
    pub kind: String,
    /// The offending request field, when one can be named.
    pub param: Option<String>,
    /// Stable machine-readable code.
    pub code: Option<String>,
}

/// The whole error body: `{"error": {...}}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// The error itself.
    pub error: ErrorDetail,
}

/// Everything this crate can refuse to do.
///
/// Carries enough data to build the body without re-deriving it: an overflow
/// knows both numbers, a rejected content part knows its `type`.
#[derive(Debug, Error)]
pub enum ServerError {
    /// The body is not JSON, or not JSON of the expected shape. Carries
    /// serde's message, which names the line, column and field.
    #[error("invalid request body: {0}")]
    MalformedJson(String),

    /// `messages` was absent-but-defaulted or `[]`. There is no prompt to
    /// build, and rendering an empty conversation would make the model answer
    /// from nothing at all.
    #[error("`messages` must contain at least one message")]
    EmptyMessages,

    /// `n > 1`. A single-session server has one decode loop; returning one
    /// choice and calling it `n` would be a silent wrong answer, so it is
    /// refused loudly instead.
    #[error("`n` must be 1: this server generates a single completion per request (got {n})")]
    MultipleChoices {
        /// What the client asked for.
        n: u32,
    },

    /// A content part this build cannot render, e.g. `image_url`.
    ///
    /// Rejected rather than dropped: the pinned ChatML template coerces
    /// non-string content to `""`, so silently discarding the part makes the
    /// model answer a question it was never asked.
    #[error("unsupported content part `{kind}`: only `text` parts are supported")]
    UnsupportedContentPart {
        /// The part's `type` field, verbatim.
        kind: String,
    },

    /// A `text` content part with no `text` field.
    #[error("content part of type `text` is missing its `text` field")]
    MissingPartText,

    /// A `tool` message, or an assistant turn carrying `tool_calls`.
    ///
    /// Parsing preserves both (a later wave re-sorts tool results by
    /// `tool_call_id`), but the vendored template has no tool branch, so a
    /// prompt cannot be built from them yet.
    #[error("tool messages are not supported by this build")]
    ToolsUnsupported,

    /// The prompt plus the reservation for the reply exceeds the context.
    ///
    /// The message names the limit because a client that cannot see it has no
    /// way to trim the conversation to fit.
    #[error(
        "context: this request needs {requested} of {limit} tokens; \
         send a shorter conversation or a smaller `max_completion_tokens`"
    )]
    ContextOverflow {
        /// Prompt tokens plus the tokens reserved for the reply.
        requested: usize,
        /// The context cap this server was started with.
        limit: usize,
    },

    /// The weights are still being opened. Transient; the same request will
    /// succeed once loading finishes.
    #[error("model is still loading")]
    ModelLoading,

    /// A generation is already in flight. Transient in exactly the same way,
    /// and reported the same way.
    ///
    /// 503, not 429. Retry behaviour is identical across the clients we care
    /// about, but 429 makes openai-python raise `RateLimitError`, which
    /// surfaces to a user as a quota or billing message — nonsense from a
    /// server running on their own machine.
    #[error("server is busy: this build serves one request at a time")]
    AtCapacity,

    /// No route serves this method and path.
    ///
    /// A 404 with the same body shape as every other refusal, because a
    /// client pointed at the wrong base URL parses the error before it can
    /// report it, and an empty body or an HTML page turns "wrong path" into
    /// "the server is broken".
    #[error("invalid URL ({method} {path})")]
    NotFound {
        /// The request method, verbatim.
        method: String,
        /// The request path, query string stripped.
        path: String,
    },

    /// The request body is larger than this server will buffer.
    ///
    /// The body is read into memory to be parsed, and this process is meant
    /// to run inside a 3 GB cgroup alongside a model: an unbounded read is an
    /// OOM with a `Content-Length` header for a trigger. The cap is generous
    /// against a 4K context and small against the budget.
    #[error("request body is larger than the {limit}-byte limit")]
    BodyTooLarge {
        /// The cap, in bytes.
        limit: usize,
    },

    /// Rendering or encoding the prompt failed.
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),

    /// Generation failed inside the runtime.
    ///
    /// Reported as a 500 rather than mapped onto a 400: every input-shaped
    /// failure is caught before generation starts (see
    /// [`ServerError::ContextOverflow`] and
    /// [`ChatCompletionRequest::validate`](crate::request::ChatCompletionRequest::validate)),
    /// so anything that reaches here is this process's problem.
    #[error(transparent)]
    Generate(#[from] GenerateError),

    /// The KV cache could not be rewound to the prefix a request shares with
    /// the one before it. Same reasoning as [`ServerError::Generate`].
    #[error(transparent)]
    Forward(#[from] ForwardError),

    /// A response could not be serialized. Unreachable for the types in this
    /// crate (no maps with non-string keys, no failing `Serialize` impls);
    /// reported rather than unwrapped so that stays true by construction.
    #[error("failed to serialize response: {0}")]
    Serialize(#[from] serde_json::Error),
}

impl ServerError {
    /// The HTTP status for this error.
    pub fn status(&self) -> u16 {
        match self {
            ServerError::MalformedJson(_)
            | ServerError::EmptyMessages
            | ServerError::MultipleChoices { .. }
            | ServerError::UnsupportedContentPart { .. }
            | ServerError::MissingPartText
            | ServerError::ToolsUnsupported
            | ServerError::ContextOverflow { .. } => 400,
            ServerError::NotFound { .. } => 404,
            ServerError::BodyTooLarge { .. } => 413,
            ServerError::ModelLoading | ServerError::AtCapacity => 503,
            ServerError::Tokenizer(_)
            | ServerError::Generate(_)
            | ServerError::Forward(_)
            | ServerError::Serialize(_) => 500,
        }
    }

    /// The `error.type` string.
    ///
    /// 404 and 413 join the 400s: all three say the client sent something
    /// this server will not serve, which is what `invalid_request_error`
    /// means to the SDKs that switch on it.
    pub fn kind(&self) -> &'static str {
        match self.status() {
            400 | 404 | 413 => TYPE_INVALID_REQUEST,
            _ => TYPE_SERVER,
        }
    }

    /// The request field at fault, when one can be named.
    pub fn param(&self) -> Option<&'static str> {
        match self {
            ServerError::EmptyMessages
            | ServerError::UnsupportedContentPart { .. }
            | ServerError::MissingPartText
            | ServerError::ToolsUnsupported
            | ServerError::ContextOverflow { .. } => Some("messages"),
            ServerError::MultipleChoices { .. } => Some("n"),
            ServerError::MalformedJson(_)
            | ServerError::NotFound { .. }
            | ServerError::BodyTooLarge { .. }
            | ServerError::ModelLoading
            | ServerError::AtCapacity
            | ServerError::Tokenizer(_)
            | ServerError::Generate(_)
            | ServerError::Forward(_)
            | ServerError::Serialize(_) => None,
        }
    }

    /// The stable machine-readable code.
    pub fn code(&self) -> &'static str {
        match self {
            ServerError::MalformedJson(_) => "invalid_json",
            ServerError::EmptyMessages => "empty_messages",
            ServerError::MultipleChoices { .. } => "unsupported_value",
            ServerError::UnsupportedContentPart { .. } => "unsupported_content_part",
            ServerError::MissingPartText => "invalid_content_part",
            ServerError::ToolsUnsupported => "tools_unsupported",
            // OpenAI's own code for this case; clients special-case the
            // string, so it is worth matching exactly.
            ServerError::ContextOverflow { .. } => "context_length_exceeded",
            ServerError::NotFound { .. } => "not_found",
            ServerError::BodyTooLarge { .. } => "request_too_large",
            ServerError::ModelLoading => "model_loading",
            ServerError::AtCapacity => "server_busy",
            ServerError::Tokenizer(_) => "tokenizer_error",
            ServerError::Generate(_) => "generate_error",
            ServerError::Forward(_) => "forward_error",
            ServerError::Serialize(_) => "internal_error",
        }
    }

    /// Headers to send alongside the body.
    ///
    /// Both 503s get [`RETRY_HEADERS`]: loading and busy are the same promise
    /// to the client — come back shortly and this will work.
    pub fn headers(&self) -> &'static [(&'static str, &'static str)] {
        match self {
            ServerError::ModelLoading | ServerError::AtCapacity => RETRY_HEADERS,
            _ => &[],
        }
    }

    /// The full error body.
    pub fn body(&self) -> ErrorResponse {
        ErrorResponse {
            error: ErrorDetail {
                message: self.to_string(),
                kind: self.kind().to_owned(),
                param: self.param().map(str::to_owned),
                code: Some(self.code().to_owned()),
            },
        }
    }

    /// The full error body, serialized. Falls back to a hand-built body if
    /// serialization somehow fails, so an error path can never panic and can
    /// never return something without a `message`.
    pub fn body_json(&self) -> String {
        serde_json::to_string(&self.body()).unwrap_or_else(|_| {
            r#"{"error":{"message":"internal error","type":"server_error","param":null,"code":"internal_error"}}"#
                .to_owned()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full status/type/param/code table, pinned. A change here is a
    /// change to the wire contract, not a refactor.
    #[test]
    fn every_error_maps_to_its_status_and_code() {
        let cases: Vec<(ServerError, u16, &str, Option<&str>, &str)> = vec![
            (
                ServerError::MalformedJson("expected value at line 1".into()),
                400,
                TYPE_INVALID_REQUEST,
                None,
                "invalid_json",
            ),
            (
                ServerError::EmptyMessages,
                400,
                TYPE_INVALID_REQUEST,
                Some("messages"),
                "empty_messages",
            ),
            (
                ServerError::MultipleChoices { n: 2 },
                400,
                TYPE_INVALID_REQUEST,
                Some("n"),
                "unsupported_value",
            ),
            (
                ServerError::UnsupportedContentPart {
                    kind: "image_url".into(),
                },
                400,
                TYPE_INVALID_REQUEST,
                Some("messages"),
                "unsupported_content_part",
            ),
            (
                ServerError::MissingPartText,
                400,
                TYPE_INVALID_REQUEST,
                Some("messages"),
                "invalid_content_part",
            ),
            (
                ServerError::ToolsUnsupported,
                400,
                TYPE_INVALID_REQUEST,
                Some("messages"),
                "tools_unsupported",
            ),
            (
                ServerError::ContextOverflow {
                    requested: 5000,
                    limit: 4096,
                },
                400,
                TYPE_INVALID_REQUEST,
                Some("messages"),
                "context_length_exceeded",
            ),
            (
                ServerError::NotFound {
                    method: "GET".into(),
                    path: "/v1/embeddings".into(),
                },
                404,
                TYPE_INVALID_REQUEST,
                None,
                "not_found",
            ),
            (
                ServerError::BodyTooLarge { limit: 1_048_576 },
                413,
                TYPE_INVALID_REQUEST,
                None,
                "request_too_large",
            ),
            (
                ServerError::ModelLoading,
                503,
                TYPE_SERVER,
                None,
                "model_loading",
            ),
            (
                ServerError::AtCapacity,
                503,
                TYPE_SERVER,
                None,
                "server_busy",
            ),
            (
                ServerError::Tokenizer(TokenizerError::Encode("boom".into())),
                500,
                TYPE_SERVER,
                None,
                "tokenizer_error",
            ),
        ];

        for (error, status, kind, param, code) in cases {
            assert_eq!(error.status(), status, "{error:?}");
            assert_eq!(error.kind(), kind, "{error:?}");
            assert_eq!(error.param(), param, "{error:?}");
            assert_eq!(error.code(), code, "{error:?}");

            let body = error.body();
            assert_eq!(body.error.kind, kind);
            assert_eq!(body.error.param.as_deref(), param);
            assert_eq!(body.error.code.as_deref(), Some(code));
        }
    }

    #[test]
    fn error_message_is_always_a_non_empty_string() {
        for error in [
            ServerError::MalformedJson(String::new()),
            ServerError::EmptyMessages,
            ServerError::MultipleChoices { n: 4 },
            ServerError::UnsupportedContentPart {
                kind: String::new(),
            },
            ServerError::MissingPartText,
            ServerError::ToolsUnsupported,
            ServerError::ContextOverflow {
                requested: 1,
                limit: 0,
            },
            ServerError::NotFound {
                method: String::new(),
                path: String::new(),
            },
            ServerError::BodyTooLarge { limit: 0 },
            ServerError::ModelLoading,
            ServerError::AtCapacity,
        ] {
            let body = error.body();
            assert!(!body.error.message.is_empty(), "{error:?}");
            // Present as a JSON string, never null: a known client's schema
            // makes the field required.
            let json = error.body_json();
            assert!(
                json.contains(r#""message":""#) || json.contains(r#""message":"#),
                "{json}"
            );
            assert!(!json.contains(r#""message":null"#), "{json}");
        }
    }

    #[test]
    fn context_overflow_names_the_limit() {
        let error = ServerError::ContextOverflow {
            requested: 5000,
            limit: 4096,
        };
        let message = error.body().error.message;
        assert!(message.contains("4096"), "{message}");
        assert!(message.contains("5000"), "{message}");
    }

    #[test]
    fn only_the_503s_carry_retry_headers() {
        assert_eq!(ServerError::AtCapacity.headers(), RETRY_HEADERS);
        assert_eq!(ServerError::ModelLoading.headers(), RETRY_HEADERS);
        assert!(ServerError::EmptyMessages.headers().is_empty());
    }

    /// openai-python's `_should_retry` reads `Retry-After` first and gives up
    /// when it exceeds 120 seconds. An honest wait for this server is minutes,
    /// so an honest header would silently switch retrying off.
    #[test]
    fn retry_after_stays_under_the_client_give_up_threshold() {
        let retry_after = RETRY_HEADERS
            .iter()
            .find(|(name, _)| *name == "Retry-After")
            .map(|(_, value)| *value);
        assert_eq!(retry_after, Some("1"));
        let seconds: u64 = retry_after.and_then(|v| v.parse().ok()).unwrap_or(u64::MAX);
        assert!(seconds <= 120, "an honest Retry-After disables retrying");
    }

    /// 429 would make openai-python raise `RateLimitError`, which reads as a
    /// quota problem to a user running the server on their own machine.
    #[test]
    fn capacity_is_503_not_429() {
        assert_eq!(ServerError::AtCapacity.status(), 503);
    }

    #[test]
    fn error_body_has_the_openai_shape() {
        let json = ServerError::EmptyMessages.body_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let error = &parsed["error"];
        assert!(error["message"].is_string());
        assert_eq!(error["type"], TYPE_INVALID_REQUEST);
        assert_eq!(error["param"], "messages");
        assert_eq!(error["code"], "empty_messages");
        // `param` and `code` are always present keys, null when they do not
        // apply, rather than omitted.
        let json = ServerError::ModelLoading.body_json();
        assert!(json.contains(r#""param":null"#), "{json}");
    }
}
