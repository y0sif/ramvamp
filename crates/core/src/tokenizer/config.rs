//! Serde schemas for the tokenizer sidecar files.
//!
//! Only the fields the runtime consumes are modeled; both files carry many
//! more (`transformers` bookkeeping, per-token metadata) that are ignored.
//! Parsing is schema-only — semantic validation against the pinned
//! architecture facts happens in [`RvmpTokenizer::load`].
//!
//! [`RvmpTokenizer::load`]: super::RvmpTokenizer::load

use serde::Deserialize;

/// The subset of `generation_config.json` the runtime consumes.
#[derive(Debug, Deserialize)]
pub(crate) struct GenerationConfigFile {
    /// Stop token id(s); upstream writes either a bare id or a list.
    pub eos_token_id: EosTokenIds,
    /// Default sampling temperature. Absent means the `transformers`
    /// default of 1.0.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Default nucleus-sampling mass. Absent means 1.0 (disabled).
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Default top-k cutoff. Absent means no top-k truncation.
    #[serde(default)]
    pub top_k: Option<u32>,
}

/// `eos_token_id` as upstream writes it: a single id or a list of ids.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum EosTokenIds {
    /// A single stop token id.
    One(u32),
    /// Several stop token ids, in file order.
    Many(Vec<u32>),
}

impl EosTokenIds {
    /// The stop token ids in file order.
    pub(crate) fn into_vec(self) -> Vec<u32> {
        match self {
            EosTokenIds::One(id) => vec![id],
            EosTokenIds::Many(ids) => ids,
        }
    }
}

/// The subset of `tokenizer_config.json` the runtime consumes.
#[derive(Debug, Deserialize)]
pub(crate) struct TokenizerConfigFile {
    /// The Jinja chat template. Stored for provenance and inspection only;
    /// rendering is vendored in Rust (see [`super::chat`]).
    #[serde(default)]
    pub chat_template: Option<String>,
    /// Whether encoding should prepend a BOS token. This model never does
    /// (verified architecture fact); `true` is rejected at load.
    #[serde(default)]
    pub add_bos_token: Option<bool>,
}
