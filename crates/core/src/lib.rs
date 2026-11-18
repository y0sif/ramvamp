//! ramvamp-core: a runtime for streaming fine-grained MoE experts from NVMe.
//!
//! The design target is a 26-30B-parameter fine-grained MoE model (128 experts
//! per layer, top-8 routing, ~3-4B active parameters per token) running in
//! about 4 GB of RAM. The always-needed "common core" (embeddings, attention,
//! routers, norms, any shared expert) is read once into anonymous memory at load
//! ([`io::CommonMemory`]); routed experts live on disk in a page-aligned packed
//! layout and are read on demand with explicit parallel I/O into a small
//! per-layer LFU cache.
//!
//! Design provenance: the streaming architecture follows the measured results
//! published by TurboFieldfare (<https://github.com/drumih/turbo-fieldfare>),
//! most importantly: explicit reads beat mmap demand paging (3.54x per cold
//! expert read, ~8x end to end in their full-token simulator); copying one
//! layer's routing choices predicted only ~7% of the next layer's experts, so
//! there is no speculative prefetch; and I/O overlaps only with compute that
//! is guaranteed to run (cache-hit experts, shared expert when the model has
//! one).
//!
//! Two things that follow from our own measurements rather than theirs. The
//! eviction policy is LFU over frequency counters indexed by expert id whose
//! counts survive eviction, which is where the entire win over LRU comes from
//! (EXP-005: without ghost history, per-slot LFU is worth -1.7 to 0.0 points
//! against LRU). And the cache dial is a total byte budget rather than a slot
//! count, so that one config stays meaningful across models:
//! [`model::DEFAULT_CACHE_BYTES`] is 2,485 MiB, 19 slots per layer on
//! Qwen3-30B-A3B (EXP-033).

// `chunks_exact_to_as_chunks` is newer than the toolchains some contributors
// run, hence `unknown_lints`. The flagged loops are hot kernel code, and
// rewriting them to `as_chunks` is a codegen change that gets measured, not
// slipped in to quiet a lint.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

pub mod format;
pub mod generate;
pub mod io;
pub mod kernels;
pub mod kv;
pub mod model;
pub mod threads;
pub mod tokenizer;
