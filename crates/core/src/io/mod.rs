//! Expert streaming and caching.
//!
//! - Common weights: read-only `mmap` (they are touched every token, so the
//!   page cache keeps them resident).
//! - Routed experts: explicit reads, never demand paging. On Linux with the
//!   `io-uring` feature (default), misses are fetched with io_uring +
//!   O_DIRECT into pre-allocated page-aligned slot buffers; the portable
//!   fallback uses positioned reads (`pread`) on a small thread pool.
//! - Cache: per-layer fixed slot arrays with LFU eviction (recency as
//!   tie-breaker). No cross-layer prefetch: measured predictability is too
//!   low to pay for speculative reads.
//!
//! Concurrency invariant: a slot being filled by an in-flight read, or still
//! owned by queued compute, is never reassigned.
