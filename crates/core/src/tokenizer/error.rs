//! Typed errors for the tokenizer layer.

use std::path::{Path, PathBuf};

use thiserror::Error;

/// Errors from loading tokenizer sidecar files or from encoding, decoding,
/// and chat rendering.
///
/// Everything under `tokenizer/` in an install is untrusted until validated:
/// every failure mode is a typed, recoverable error and this layer never
/// panics on bad input.
#[derive(Debug, Error)]
pub enum TokenizerError {
    /// A filesystem operation failed.
    #[error("{}: {source}", path.display())]
    Io {
        /// Path the operation was acting on.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A tokenizer file exceeds the read size cap.
    #[error("{}: larger than the {max}-byte cap for tokenizer files", path.display())]
    FileTooLarge {
        /// Path of the oversized file.
        path: PathBuf,
        /// The cap that was exceeded.
        max: u64,
    },

    /// A JSON sidecar file exists but does not parse into its schema.
    #[error("{}: invalid JSON: {source}", path.display())]
    Json {
        /// Path of the offending file.
        path: PathBuf,
        /// Underlying parse error.
        #[source]
        source: serde_json::Error,
    },

    /// The `tokenizers` backend rejected `tokenizer.json`.
    ///
    /// The backend's error type is a boxed `dyn Error`, so only its message
    /// survives here.
    #[error("{}: tokenizer backend: {message}", path.display())]
    Backend {
        /// Path of the tokenizer definition that failed to load.
        path: PathBuf,
        /// Message from the `tokenizers` crate.
        message: String,
    },

    /// Encoding text to token ids failed.
    #[error("encode failed: {0}")]
    Encode(String),

    /// Decoding token ids to text failed.
    #[error("decode failed: {0}")]
    Decode(String),

    /// `tokenizer_config.json` has no `chat_template` string.
    #[error("tokenizer_config.json has no chat_template string")]
    MissingChatTemplate,

    /// `tokenizer_config.json` asks for a BOS token, but this model never
    /// prepends BOS (verified architecture fact).
    #[error("tokenizer_config.json sets add_bos_token=true, but this model never prepends BOS")]
    BosNotAllowed,

    /// `generation_config.json` stop tokens differ from the verified
    /// architecture facts.
    #[error("generation_config.json eos_token_id {found:?} does not match expected {expected:?}")]
    EosMismatch {
        /// Stop token ids found in the file.
        found: Vec<u32>,
        /// The stop token set the architecture pins.
        expected: [u32; 2],
    },

    /// A pinned special token is missing from the vocabulary or maps to an
    /// unexpected id.
    #[error("special token {token:?} maps to {found:?}, expected id {expected}")]
    SpecialTokenMismatch {
        /// The special token text.
        token: &'static str,
        /// The id the loaded vocabulary assigns it, if any.
        found: Option<u32>,
        /// The id the architecture pins.
        expected: u32,
    },

    /// A chat role string is not one this build renders.
    ///
    /// The upstream template also knows `tool` (and silently drops anything
    /// else); both are out of scope for v0, and rejecting them beats
    /// rendering a prompt the model was not trained on.
    #[error("unsupported chat role {0:?} (supported: system, user, assistant)")]
    UnsupportedRole(String),
}

impl TokenizerError {
    /// Attach path context to an I/O error.
    pub(crate) fn io(path: &Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}
