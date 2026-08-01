# Architecture (v0 ground truth)

Agreed 2026-08-01 after the design-phase research. This document is the source
of truth for v0 implementation. Changes to it require an experiment entry or a
recorded decision. Estimates are marked as such; everything else was verified
against checkpoints, kernel/driver behavior, or published measurements
(sources in `docs/landscape.md` and the research notes below).

## Goal

Run Qwen3-30B-A3B coherently in about 3 GB of RAM on an ordinary Linux
machine with an NVMe SSD, CPU only, at 3+ tokens/second (4-8 expected).

v0 success criterion: coherent chat, >= 3 tok/s decode, inside a cgroup with
`memory.max=3G` and `memory.swap.max=0` (zram counts as swap), cold page
cache, KL-divergence vs llama.cpp within accepted tolerance on identical
weights.

Reference hardware: Intel Core Ultra 9 185H (6P + 8E + 2 LP-E, AVX2, no
AVX-512), 16 GB LPDDR5X-7467, Micron 2450 Gen4 NVMe (~3.6 GB/s sequential
read, `max_hw_sectors_kb=128`).

## Model pin (v0)

`Qwen/Qwen3-30B-A3B-Instruct-2507`, consumed via a community GGUF Q4_K_M
(NOT unsloth UD variants, which use nonstandard per-layer types). Frozen
pin, audited 2026-08-01: repo
`bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF`, revision
`6c6e8692f43e4ca663f7ece8229a1361090d3a4c`, file
`Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`, size 18,632,183,808 B
(17.35 GiB), install size 18,626,213,888 B. The manifest records the pin
and the audited per-tensor type map.

Verified architecture facts the runtime may rely on:

| Fact | Value |
| --- | --- |
| Layers | 48, every layer MoE (`decoder_sparse_step=1`, `mlp_only_layers=[]`), no shared expert |
| Experts | 128/layer, top-8, softmax over all 128 then renormalize top-8 (`norm_topk_prob=true`) |
| Attention | Full attention all layers (no sliding window), GQA 32:4, head_dim 128, per-head QK-RMSNorm ([128] weight, no bias), NeoX-style RoPE, rope_theta 1e7 |
| Dims | hidden 2048, moe_intermediate 768, SwiGLU (silu), rms_norm_eps 1e-6 |
| Vocab | 151936, untied embeddings (separate lm_head), no BOS ever prepended |
| Stop tokens | 151645 `<\|im_end\|>` and 151643 `<\|endoftext\|>` |
| Chat format | ChatML: `<\|im_start\|>{role}\n{content}<\|im_end\|>\n`; generation prompt `<\|im_start\|>assistant\n`; 2507-Instruct template has no thinking logic |
| No biases | on any attention, router, or expert projection |

Attention inner width is 4096 (32 heads x 128): q_proj up-projects from 2048,
o_proj projects back down.

## Source quantization: GGUF Q4_K_M (decision)

Decisive criterion: llama.cpp runs the identical bytes we repack, so
validation compares arithmetic, not weights. MLX-affine and AWQ/GPTQ were
eliminated on this alone. Q4_0 (same GGUF plumbing, trivial kernel) is the
bring-up format; Q4_K_M is the shipping target.

Consequences accepted:

- Two expert dot kernels: Q4_K (144 B / 256 weights) and Q6_K (210 B / 256
  weights). Audited per-tensor types (facts, 2026-08-01, bartowski file at
  commit `6c6e8692f43e4ca663f7ece8229a1361090d3a4c`): `ffn_down_exps` is
  Q6_K on 24 of 48 layers (0-5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38,
  41-47); gate/up are Q4_K everywhere. In the common weights, `attn_k` is
  Q8_0 (all 48 layers), `attn_output` is Q5_K (all 48), `attn_v` is Q6_K on
  exactly the Q6_K-down layers and Q4_K elsewhere; `token_embd` is Q4_K,
  `output.weight` (lm_head) is Q6_K, router and norms are F32. Consequence:
  the expert-streaming kernels stay Q4_K+Q6_K, but the resident
  common-weight matmuls need Q4_K, Q5_K, Q6_K, and Q8_0 dequant paths (all
  with candle AVX2 references).
- Activations quantize to Q8_K per 256-block with per-32 `bsums`, mirroring
  ggml's `vec_dot_q4_K_q8_K` structure. Candle's Rust AVX2 k-quant ports
  (MIT/Apache-2.0) are the reference implementation to crib from.
- Per-expert slabs are contiguous byte ranges of the 3D `*_exps` tensors
  (expert index = slowest axis). For this model every Q4_K projection slab is
  884,736 B = 216 x 4 KiB pages; Q6_K down slabs are 1,290,240 B = 315 pages.
  Page alignment of expert blobs is therefore free.

## The `.rvmp` installed model

```text
model.rvmp/
  manifest.json        # see schema below; written last, atomically promoted
  common.bin           # embeddings, lm_head, attention, routers, norms; mmap'd
  tokenizer/           # tokenizer.json + chat template + generation defaults
  experts/
    layout.json        # per-layer stride, per-projection offsets and quant types
    layer_00.bin ...   # 128 fixed-stride page-aligned expert blobs per layer
```

Expert blob = gate + up + down slabs for one expert, concatenated, each
4 KiB-aligned within the blob. Stride is uniform within a layer; layers with
Q6_K down have a larger stride (~2.92 MiB) than pure-Q4_K layers
(~2.53 MiB). One blob = one O_DIRECT read.

`manifest.json` (v1 schema, frozen at implementation):

```json
{
  "rvmp_version": 1,
  "model_id": "qwen3-30b-a3b-instruct-2507",
  "source": { "hf_repo": "...", "revision": "...", "file": "...Q4_K_M.gguf",
              "sha256": "..." },
  "arch": { "n_layers": 48, "n_experts": 128, "top_k": 8, "hidden": 2048,
            "moe_intermediate": 768, "n_heads": 32, "n_kv_heads": 4,
            "head_dim": 128, "vocab": 151936, "context_length": 262144,
            "rope_theta": 1e7, "rms_eps": 1e-6, "norm_topk_prob": true,
            "tie_embeddings": false, "shared_expert": false,
            "sliding_window": null },
  "quant": { "scheme": "gguf", "tensor_types": { "...": "q4_k" } },
  "files": { "common.bin": { "size": 0, "sha256": "..." },
             "experts/layer_00.bin": { "size": 0, "sha256": "..." } }
}
```

Repacker rules (unchanged from scaffold): bounded HTTP range requests, fixed
small scratch, quantized bytes copied unchanged, resumable, `manifest.json`
validates before atomic promotion. The runtime hashes `manifest.json`,
`common.bin`, and `layout.json` at load and each layer file on first use.

## Memory contract (option B, agreed)

The expert-cache dial is a total memory budget (default ~1.3 GiB), divided by
layer count to get slots per layer. This keeps one config meaningful across
models: Qwen3 (48 layers) gets 10 slots/layer; Gemma 4 (30 layers) will get
13-14 under the same budget.

| Tenant | Qwen3 v0 | Notes |
| --- | ---: | --- |
| Common core (mmap, read-only) | 1,023.34 MiB (audited) | touched every token, page cache keeps it resident; counts toward cgroup |
| KV cache FP16 | 384 MiB @ 4K | 96 KiB/token; linear append, 48 layers |
| Expert slot pool | ~1.28 GiB | 10 slots x 48 layers, page-aligned, allocated once |
| Scratch + program + tokenizer | ~150 MiB | fixed, reused per layer/chunk |
| Total | **~2.8 GiB** | cgroup `memory.max=3G` |

Budget escape hatches, in the experiment backlog and not in v0: Q8 KV
(halves KV), 8 slots/layer, global slot pool shared across layers.

## Expert streaming and cache

- Common weights: `mmap` read-only. Routed experts: explicit reads, never
  demand paging (TurboFieldfare measured mmap ~8x slower cold; llama.cpp RFC
  measured 377 MB/s via faults vs 2.8 GB/s explicit).
- Reads: io_uring + O_DIRECT into the pre-registered slot buffers
  (registered buffers, registered files). Interrupt-driven completions; no
  SQPOLL/IOPOLL (they burn a core the GEMVs need). Queue depth 8 outstanding
  blob reads; the kernel splits each ~2.5-3 MiB read into ~20-24 concurrent
  128 KiB NVMe commands, so per-blob latency is ~1 ms when the queue is fed
  (estimate).
- O_DIRECT is also a budget-correctness requirement: buffered reads would
  charge the page cache to our cgroup and thrash it.
- Cache policy: per-layer slot arrays, LFU eviction with recency tie-break
  (TF: LFU beat LRU 72.6 -> 64.8 ms/token). No cross-layer prefetch
  (measured ~7% predictability upstream).
- Concurrency invariant: a slot owned by an in-flight read or by queued
  compute is never reassigned.
- Portable fallback behind the `io-uring` feature flag: positioned reads on a
  small thread pool, buffered; for tests and non-Linux dev only, never for
  published numbers.
- Install-time I/O note: the HF Xet CDN signs each download URL for one
  exact byte range (any other range 403s), so the installer downloads in
  large sequential windows demuxed to destination files rather than issuing
  per-tensor requests.

## Thread topology (v0)

- Compute pool: 6 threads pinned to P-cores, one per physical core, no SMT
  siblings. Spin barrier within a token step (ggml-style), condvar sleep
  between generations.
- I/O: one io_uring reactor thread pinned to an E-core. It only submits,
  reaps completions, and flips slot-ready flags.
- LP E-cores (no L3, SoC tile): never used by ramvamp threads.
- E-cores joining the compute pool for independent expert GEMVs (not
  barrier-coupled) is a backlog experiment; llama.cpp evidence says
  barrier-coupled E-cores cost 20-30%.

## Decode loop (per token, per layer)

1. Attention (QK-norm, RoPE, GQA over FP16 KV) + router on resident weights.
2. Top-8 expert IDs + renormalized weights.
3. LFU plan: hits, misses, eviction victims.
4. Submit misses to the io_uring reactor. While reads land, compute cache-hit
   experts (Qwen3 has no shared expert, so hits are the overlap work).
5. Miss blobs compute as their slots fill; weighted-sum reduction, residual,
   next layer.
6. After layer 48: final norm, lm_head, softmax/sampling (greedy path must be
   deterministic).

Sampler defaults come from the checkpoint's generation_config (temp 0.7,
top_p 0.8, top_k 20); greedy override for validation.

## Prefill (sequential sweep, our improvement over TF)

Measured coverage for this model (Layered Prefill, arXiv 2510.08055): a
128-token chunk activates 86% of each layer's experts; 512 tokens ~100%.
Chunked prefill therefore approaches "read every expert once per chunk per
layer" no matter how it is scheduled. So:

- Prefill bypasses the LFU cache entirely.
- Layer-major chunks of up to 512 tokens. Per layer: group rows by expert
  (mul_mat_id style), then stream the layer's expert file sequentially
  front-to-back through a small ring of streaming buffers, computing each
  expert against all its routed rows as it arrives. Sequential read at
  ~3.6 GB/s instead of random.
- Expected: ~4.9 s per 512-token chunk sweep (~17.6 GB of expert data),
  roughly 100 tok/s prefill (estimate; measured entry required). TF's design
  (random tile fetches through the decode cache) achieved ~28 tok/s.
- Decode cache starts cold after prefill; acceptable, first tokens warm it.

## Validation protocol vs llama.cpp

Same GGUF bytes on both sides. Gates, in order:

1. Tensor-level: repacked expert slab bytes == GGUF slice bytes (exact).
2. Kernel-level: our Q4_K/Q6_K/Q8_K dot products vs scalar reference within
   documented tolerance; alignment assumptions unit-tested.
3. Logit-level: KL divergence vs llama.cpp logits on a fixed prompt set
   (target: mean KL <= 1e-3); top-1 agreement rate reported.
4. Greedy smoke: first N tokens identical on short prompts (expected to
   diverge eventually from fp reordering; report length, do not gate).
5. Perplexity on a standard slice within noise of llama.cpp same-quant.

## Performance model (estimates, to be replaced by measurements)

- In-RAM compute ceiling on the 185H: 15-25 tok/s (bandwidth-scaled
  estimate; no direct public benchmark exists).
- Decode I/O: worst case ~1.02 GiB/token (24 Q4_K layers + 24 Q6_K-down
  layers); with 10 slots/layer LFU expect roughly 40-60% hit rate (TF
  analogy), i.e. ~420-630 MiB/token at ~3 GB/s effective random-ish read =
  ~145-220 ms/token I/O, partially overlapped with hit compute.
- Expected v0 band: 4-8 tok/s decode. Floor for success: 3.
- Anchor prior art: llama.cpp RFC #23324 (same design, pread sidecar) runs
  this model at 13 tok/s on a 16 GB M1 Pro with a much larger slot pool.

## v0 scope caps

Single sequence, 4K context, CLI chat + raw completion, greedy +
standard sampling, no server, no batching, no speculative anything, Linux
only, x86-64 with AVX2 required. Gemma 4 26B-A4B is model #2 and brings:
shared-expert overlap, SWA KV rings, per-layer attention-type mix, logit
softcap, and (if we adopt their quant source) a second quant scheme decision.

## Post-v0 direction (recorded 2026-08-01, not commitments)

- `ramvamp-server`: loopback OpenAI-compatible Chat Completions (streaming
  SSE, tool calls parsed from Qwen's native `<tool_call>` tokens). This is
  the integration path for OpenCode and anything OpenAI-speaking. Must add
  special-token sanitization for untrusted content (recorded decision:
  `encode_chat` is reference-faithful).
- KV prefix caching (prefill the system prompt once, reuse across turns);
  prerequisite for agentic clients whose prompts dominate the context.
- Larger context via Q8 KV + budget growth; unlocks the Thinking-2507
  variant (same architecture, different pin + template + think-span
  handling) for quality-over-latency users.
- Anthropic Messages API schema as a second endpoint (Claude Code path;
  until then, a translation proxy works).
- Gemma 4 26B-A4B as model #2; Vulkan backend behind the kernel trait.

## Experiment backlog (numbered entries when run)

- E-cores in compute pool for non-barrier expert GEMVs
- Global slot pool vs per-layer (hot layers steal slots)
- Q8_0 KV cache (budget escape hatch)
- Expert layout reordering by co-activation (llama.cpp #18758 measured 2.23x
  cold-I/O gain; conflicts with fixed-stride simplicity, needs data)
- `fadvise`/`readahead` tuning for the prefill sequential sweep
- Slot budget sweep: 8 vs 10 vs 12 vs 16 slots/layer hit-rate curve
- Prefill chunk size sweep: 128 vs 256 vs 512

## Deferred to implementation (not design-blocking)

- `gguf_dump` audit of the pinned file (done 2026-08-01, see "Model pin" and
  "Source quantization"); the manifest writer freezes the `tensor_types` map
- Exact Q8_K activation quantization scheme (mirror ggml's)
- Chat template rendering: vendor the 2507 template, snapshot-test against
  `transformers` reference renders
- Slot-ready signaling mechanism between reactor and compute pool (atomic
  flags vs eventfd; measure both)
