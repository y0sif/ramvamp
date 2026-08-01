# Architecture

Working design document. Grows as decisions are made; measured results move to
`docs/experiments/`.

## Goal

Run a 26-30B-parameter fine-grained MoE model coherently in about 2 GB of RAM
on an ordinary Linux machine with an NVMe SSD, CPU only, at 3+ tokens/second.

v0 target model: **Qwen3-30B-A3B** (validates token-for-token against
llama.cpp on the same checkpoint). Second model: **Gemma 4 26B-A4B** (proves
generality; adds shared-expert overlap and sliding-window KV rings).

## Memory contract (v0 budget, to be refined)

| Tenant | Budget | Notes |
| --- | --- | --- |
| Common core (mmap) | ~0.7-1.4 GB | Embeddings/head, attention, routers, norms, shared experts. Exact size depends on model and quant; derived from the manifest. |
| KV cache (FP16) | ~300 MB at 4K context | Linear for full-attention layers; ring buffers for sliding-window layers. |
| Expert cache slots | ~0.5-1.0 GB capacity | 16 page-aligned slots per layer, LFU eviction. Slot count is a runtime dial. |
| Scratch | tens of MB | Fixed-size, reused across layers and chunks. |

Benchmark honesty rule: all published numbers run inside a cgroup with
`memory.max=2.5G` and swap disabled, from a cold page cache. Warm-cache runs
are diagnostics, not results.

## Decode loop (per layer)

1. Attention + router run on resident (mmap'd) weights.
2. Read back the router's top-k expert IDs.
3. Plan against the layer's LFU slot cache: hits, misses, evictions. A slot
   owned by an in-flight read or queued compute is never reassigned.
4. Submit io_uring O_DIRECT reads for misses. While they complete, run the
   compute that is guaranteed to be needed: cache-hit experts and, when the
   model has one, the shared expert.
5. Combine routed and shared branches, apply the layer tail, continue.

Prefill is layer-major in bounded chunks (up to ~128 tokens) so one fetched
expert serves many rows and scratch stays fixed.

## Packed format (`.rvmp`)

See `crates/core/src/format/mod.rs` for the directory layout and invariants.
Key properties: fixed-stride page-aligned expert blobs (one O_DIRECT read per
expert), byte-identical quantized values from the source checkpoint, atomic
promotion gated on a validated manifest, resumable hash-verified installs.

Quantization for v0: reuse an existing 4-bit group-quantized checkpoint
layout rather than inventing one. Candidate sources and the exact quant
scheme are a design-phase decision (record it as an experiment entry).

## Open questions (design phase)

- Exact Qwen3-30B-A3B dimensions and tying: verify against the checkpoint's
  `config.json` before freezing the manifest schema.
- Source checkpoint and quant format for the repacker (GGUF Q4_K vs MLX-style
  affine INT4 vs AWQ): pick whichever makes byte-identical repacking and CPU
  dequant kernels simplest.
- Thread topology: how many io_uring submission contexts vs compute threads
  on hybrid P/E-core Intel CPUs.
- Chunked prefill size and whether AVX2 GEMM (multi-row) kernels are worth it
  for v0 or whether GEMV-per-row suffices initially.
