# Roadmap

This is ramvamp's plan of record: what v0 delivers, what comes next, and the
decisions still open. v0 runs Qwen3-30B-A3B Q4_K_M coherently in about 4 GB of
RAM (a 4 GiB `memory.max` cap with swap off, cold page cache), validated
against llama.cpp on byte-identical weights, from a CLI, a chat REPL, or a
loopback OpenAI-compatible HTTP server with streaming and tool calls.

Published numbers are in [benchmarks.md](benchmarks.md), the design is in
[architecture.md](architecture.md), and every `EXP-NNN` cited here is a row in
[experiments.md](experiments.md).

## Where v0 stands

Every v0 component is done and every validation gate passes.

| | status |
| --- | --- |
| Packed `.rvmp` format, streaming installer | **done**: a 17.35 GiB install with resume and verify, never holding a full tensor in memory |
| Tokenizer and vendored ChatML | **done**: token-exact against transformers on committed fixtures |
| CPU kernels | **done**: AVX2, FMA and F16C fast paths with a scalar fallback ([architecture.md](architecture.md#compute)) |
| Forward pass, KV cache, generation | **done** (EXP-003, EXP-004) |
| Expert streaming and cache | **done**: io_uring and O_DIRECT reads, ghost-LFU cache, 19 slots per layer by default (EXP-005 to EXP-014, EXP-033) |
| Sequential-sweep prefill | **done**: the default path, bit-identical to the per-token one (EXP-016) |
| Memory contract | **done**: 4 GiB with swap off (EXP-033); `--cache-bytes 1440M` fits a 3 GiB cap (EXP-031) |
| Validation gates 1 to 4 | **done**: expert bytes exact, kernels against scalar, full-vocab KL and greedy tokens against llama.cpp (EXP-003, EXP-004) |
| Gate 5: perplexity | **done**: WikiText-2 perplexity within 1% of llama.cpp at every window (EXP-035) |
| Decode throughput | **published per drive** in [benchmarks.md](benchmarks.md) |
| HTTP server | **done**: loopback OpenAI-compatible `/v1/chat/completions` (streaming and buffered), `/v1/models` and `/health`, one request at a time, KV cache reused across requests ([server.md](server.md)) |
| Tool calling | **done**: parsed from Qwen's native `<tool_call>` tokens, streaming and buffered, so agent clients work |
| Context and profiles | **done**: `--context`, profiles, and `ramvamp plan`, which refuses a configuration that cannot fit before allocating ([configuration.md](configuration.md)) |
| Distribution | **done**: `install.sh` for Linux x86_64 and the landing page |

## Next

What comes next is not decided. The candidates are the v1 items, the backlog
and the one open decision below. A second drive waits for faster hardware or a
contributor's report (the method is in [benchmarks.md](benchmarks.md#method)).

## v1

v1 has four items; the first two are set aside until their reopen condition
holds.

### 1. Gemma 4 26B-A4B as model #2: not now

A second model has to run faster or answer better than the Qwen3 pin; support
for its own sake is not a goal. Gemma 4 does not clearly clear that bar. These
are estimates from its audited header (EXP-036) and Qwen3's measured split
(EXP-034), not measurements.

- **Speed: about the same.** Roughly 1.1x decode at ctx 512, 1.2x at 4K and
  0.9x prefill: its expert reads are 0.73x Qwen3's per token, but its weights
  outside the experts are 1.469 GiB against 0.999, which costs about five
  slots under the 4 GiB cap.
- **Quality: unproven for this use.** Its published lead over Qwen3-30B-A3B is
  in thinking mode, which costs minutes per answer at this speed, and ramvamp
  ships the non-thinking Instruct pin.
- **Cost: a second model definition.** A Q4_0 kernel, `gemma4` metadata, two
  attention variants, a sliding-window KV ring, a dense MLP per layer and a
  tokenizer that wants a BOS, plus its own correctness gate.

If it is reopened, the checkpoint is Google's QAT Q4_0 GGUF, so the one new
kernel is Q4_0. **Reopen when** there is a measured reason to expect it faster
or better than the pin: non-thinking quality numbers above Qwen3's, or a live
hit-rate measurement that moves the speed estimate.

### 2. A GPU backend behind the kernel trait, Vulkan or CUDA: not now

No GPU is required, so a GPU backend can only be an added configuration, and
the CPU path stays the reference that correctness is measured against. A
Vulkan spike on the reference machine's Arc iGPU ported prefill's Q4_K batched
GEMV with matching numerics (max abs error 1.490e-6), Q4_K only, and produced
no quotable speed ratio because the CPU pool would not hold steady even at
locked clocks (EXP-026).

**Reopen when** both hold: the CPU pool stays steady on a machine in normal
use, so a ratio means something; and there is reason to expect the iGPU to
beat the best CPU kernel by 1.5x or more on this GEMV. The bar is high because
a backend costs a Q6_K shader, an upload path and a second thing to keep
correct.

### 3. Bring your own model

The `.rvmp` format and the I/O layer are model-agnostic: layer geometry comes
from the manifest at runtime. Three things are pinned to Qwen3:

1. The repacker accepts only the `qwen3moe` architecture and reads its GGUF
   metadata under that prefix.
2. The forward pass implements one attention variant (full causal GQA, no
   sliding window or logit softcap) and accepts only the audited Q4_K_M expert
   types: gate and up Q4_K, down Q4_K or Q6_K.
3. The loader requires Qwen3's QK-norm tensors (`attn_q_norm`, `attn_k_norm`).

### 4. Architectures beyond Linux x86_64

The fast kernels are AVX2, FMA and F16C, and the streamer assumes io_uring.
Both have fallbacks: the scalar reference kernels and synchronous positioned
reads. aarch64 NEON kernels are the obvious first target, and the kernel trait
plus those fallbacks are the scaffolding for them.

## Backlog

Recorded so they stop being re-derived; in no fixed order.

- **Overlap instrumentation.** Split `expert io` into submitting, waiting on a read in flight, and waiting with the queue empty, so its reducible part can be measured (EXP-025).
- **Pool per-row cost.** Decode's GEMV is memory-bound and pool workers are slower per row than the submitting thread; timebox it.
- **Decode attention past 4 kv heads.** The query-head split measured worse, and the position axis needs the forbidden rescaled softmax (EXP-020).
- **Attention blocking at long context**, where the kernel drifts memory-bound (EXP-020).
- **Prefill against prompt length.** Measured at ctx 512 and 3,961 (EXP-033); no prefill-time claim about a larger context is published until a longer rung is measured.
- **A second drive.** A mainstream TLC Gen4 report turns the published figures into a curve and tests that the design scales with the device.
- **Expert-cache policy on long multi-turn traces**, offline with `scripts/lfu_sim.py` and the committed trace; anything over about a point above the shipped policy earns a cold run (EXP-030).
- **Larger context via Q8 KV**, which also opens the Thinking-2507 variant.
- **KV prefix reuse across conversations and restarts.** The server already reuses it within one conversation.
- **An Anthropic Messages endpoint** beside the OpenAI one; a translation proxy works until then.

## Open decisions

One design decision is open.

| Decision | Status |
| --- | --- |
| Reopen per-expert progressive miss execution | **open**. TurboFieldfare measured it slower with divergent output ([comparison.md](comparison.md)). Here the staged reduction is fixed-order, so the output cannot diverge, but a compute saving on this path is partly taken back as exposed I/O and must be priced net (EXP-025) |
