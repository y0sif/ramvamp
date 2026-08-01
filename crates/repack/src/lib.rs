//! ramvamp-repack: streaming installer.
//!
//! Downloads a pinned Hugging Face checkpoint revision through bounded HTTP
//! range requests and repacks tensors directly into the ramvamp packed layout
//! as bytes arrive. The full checkpoint never exists in memory or as a second
//! copy on disk; scratch is capped at a fixed small buffer. Installs are
//! hash-verified, resumable, and promoted atomically once `manifest.json`
//! validates.

pub mod gguf;
pub mod install;
pub mod plan;
pub mod source;
pub mod tokenizer_fetch;
