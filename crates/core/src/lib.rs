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
//! most importantly: explicit reads beat mmap demand paging (3.54x per cold
//! expert read, ~8x end to end in their full-token simulator); a per-layer
//! expert cache roughly halves expert I/O; cross-layer expert prediction does
//! not work (~7% accuracy), so there is no speculative prefetch; and I/O
//! overlaps only with compute that is guaranteed to run (cache-hit experts,
//! shared expert when the model has one).
//!
//! Two claims that follow from our own measurements rather than theirs
//! (EXP-005): the cache dial is a total byte budget rather than a slot count,
//! and the eviction policy is LFU over frequency counters indexed by expert
//! id whose counts survive eviction, which is where the win over LRU actually
//! comes from.

pub mod format;
pub mod generate;
pub mod io;
pub mod kernels;
pub mod kv;
pub mod model;
pub mod threads;
pub mod tokenizer;
