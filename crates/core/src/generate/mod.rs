//! Generation orchestration: chunked prefill, token-by-token decode, sampling.
//!
//! Prefill processes the prompt in bounded chunks (so one fetched expert
//! serves many rows and scratch memory stays fixed) and is layer-major.
//! Decode repeats the routed layer loop one token at a time. Sampling
//! supports greedy, temperature, top-k, top-p, and repetition penalty;
//! greedy decode must be deterministic for validation against reference
//! implementations.
