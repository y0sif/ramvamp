//! The ramvamp packed model format.
//!
//! An installed model is a directory:
//!
//! ```text
//! model.rvmp/
//!   manifest.json        # architecture, quantization, file sizes, SHA-256 hashes
//!   common.bin           # embeddings/head, attention, routers, norms, shared experts
//!   tokenizer/           # HF tokenizer.json + chat template
//!   experts/
//!     layout.json        # sub-tensor offsets within each expert blob
//!     layer_00.bin ...   # one file per layer: fixed-stride, page-aligned expert blobs
//! ```
//!
//! Invariants:
//! - Expert blobs have a fixed stride and 4 KiB (page) alignment so a single
//!   O_DIRECT read fetches exactly one expert.
//! - Quantized values are copied from the source checkpoint byte-for-byte;
//!   the repacker changes layout, never numerics.
//! - `manifest.json` is written last; a directory without it is a partial
//!   install and must be rejected by the loader.
