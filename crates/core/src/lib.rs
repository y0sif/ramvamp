//! ramvamp-core: a runtime for streaming fine-grained MoE experts from NVMe.
//!
//! The design target is a 26-30B-parameter fine-grained MoE model (128 experts
//! per layer, top-8 routing, ~3-4B active parameters per token) running in a
//! ~3 GB memory budget. The always-needed "common core" (embeddings, attention,
//! routers, norms, any shared expert) stays memory-mapped; routed experts live
//! on disk in a page-aligned packed layout and are read on demand with
//! explicit parallel I/O into a small per-layer LFU cache.
//!
//! Design provenance: the streaming architecture follows the measured results
//! published by TurboFieldfare (<https://github.com/drumih/turbo-fieldfare>),
//! most importantly: explicit reads beat mmap demand paging ~8x for cold
//! experts; a 16-slot-per-layer LFU cache halves expert I/O; cross-layer
//! expert prediction does not work (~7% accuracy), so there is no speculative
//! prefetch; and I/O overlaps with compute that is guaranteed to run
//! (cache-hit experts, shared expert when the model has one).

pub mod format;
pub mod generate;
pub mod io;
pub mod kernels;
pub mod kv;
pub mod model;
pub mod tokenizer;
