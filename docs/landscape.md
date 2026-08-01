# Competitive landscape

Researched 2026-08-01, before any code was written. This document records why
ramvamp exists given what already ships.

## Differentiation statement

Given that **TurboFieldfare** (Swift/Metal, Apple-Silicon-only, single-model)
and **MoE-Infinity** (Python/CUDA, server-class) exist, and **llama.cpp has
no merged SSD expert streaming** (open feature requests plus one unmerged RFC
prototype; its mmap fallback was measured ~8x slower than explicit reads by
TurboFieldfare), ramvamp earns its existence because:

1. It brings measured, not theoretical, SSD expert streaming to Linux on
   commodity x86 hardware, with no GPU required.
2. It is class-generic over fine-grained MoE models (Qwen3-30B-A3B first,
   Gemma 4 26B-A4B second) rather than pinned to one checkpoint.
3. Rust with io_uring and O_DIRECT is a better substrate for the core trick,
   explicit parallel reads, than the macOS pread path that inspired it.

## Competitive matrix

| Project | Type / language | Approach | Status (Aug 2026) | Gap ramvamp fills |
| --- | --- | --- | --- | --- |
| [TurboFieldfare](https://github.com/drumih/turbo-fieldfare) | OSS, Swift + Metal | Explicit pread + per-layer LFU expert cache + custom quantized kernels; 5.1-6.3 tok/s in ~2 GB on an 8 GB M2 Air | Active, ~3.2k stars | Apple-only by design (Swift, Metal, unified memory); one pinned model |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | OSS, C++ | mmap demand paging when the model exceeds RAM; `--n-cpu-moe` offloads experts to RAM, not SSD | Very active | No merged SSD expert streaming; [discussion #19163](https://github.com/ggml-org/llama.cpp/discussions/19163), issues [#19825](https://github.com/ggml-org/llama.cpp/issues/19825) and [#20757](https://github.com/ggml-org/llama.cpp/issues/20757) request it; closest work is the unmerged RFC [#23324](https://github.com/ggml-org/llama.cpp/discussions/23324) (pread expert-slot prototype, 13 tok/s Qwen3-30B-A3B on a 16 GB M1 Pro) |
| [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | OSS, Python/PyTorch | Activation-aware expert cache across GPU/host/SSD tiers, prefetching | Active, academic | Server-oriented; needs CUDA and large host RAM; not a 2 GB consumer play |
| [Micro-Expert-Router](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER) | OSS, Rust | io_uring + O_DIRECT NVMe expert streaming, CPU kernels | Early-stage | Closest precedent, but its performance numbers are stated by the author to be theoretical projections, not measurements |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) / [candle](https://github.com/huggingface/candle) | OSS, Rust | Full inference engines with quantization and device offload | Active | No SSD expert streaming; useful as building blocks and kernel references |
| AirLLM / "LLM in a flash" | OSS / research paper | Layer-by-layer disk streaming / flash-aware sparsity | Dormant / never productized | Layer streaming of dense models is unusably slow; the technique needs MoE sparsity |

## Why the technique needs fine-grained MoE

| Model shape | Experts per layer | Per-expert blob (4-bit) | Worst-case reads per token | Verdict |
| --- | --- | --- | --- | --- |
| Dense 26B | none | n/a | ~13 GB | Unusable: multiple seconds per token at NVMe speeds |
| Coarse MoE (Mixtral 8x7B) | 8, top-2 | ~90 MB | ~5.6 GB | Unusable: experts too large to stream or cache |
| Fine-grained MoE (Gemma 4 26B-A4B, Qwen3-30B-A3B) | 128, top-8 | ~2.5-3.4 MB | ~0.8-1.1 GB worst case depending on model; a small LFU cache absorbs roughly half | The regime this project targets |

Fine-grained MoE is the direction the field converged on (DeepSeek-V3, Qwen3,
Gemma 4, GLM-4.5-Air), so the class of runnable models grows over time.

## Design oracle: TurboFieldfare's measured findings

Findings from their 103-entry experiment log that transfer to any
implementation of this technique:

1. Explicit reads beat mmap demand paging ~8x for cold experts
   (0.50 vs 3.97 tok/s end to end; 9.88 vs 2.79 ms per cold expert read).
2. A 16-slot-per-layer LFU cache roughly halves expert I/O
   (166 to 88 ms/token; LFU beat LRU 72.6 to 64.8 ms/token).
3. Cross-layer expert prediction fails (~7% accuracy): no speculative
   prefetch. Cache-on-reuse is the whole game.
4. Overlap I/O only with compute guaranteed to run (cache hits, shared
   expert); fine-grained overlap measured slower than coarse overlap.
5. Warm page caches fake wins. End-to-end cold measurements decide what
   ships (hence the cgroup benchmark methodology in this project).

## Known trade-offs to state honestly

- SSD-streamed MoE costs more energy per token than RAM-resident inference
  (up to ~12x per [arXiv:2508.06978](https://arxiv.org/html/2508.06978v1)).
  Reads do not wear SSDs, but battery life on laptops will be affected.
- The approach stays I/O-bound: TurboFieldfare's tuned M2 path still spends
  ~half its per-token time on expert reads. Expect roughly 2-3x slower decode
  than a fits-in-RAM engine, in exchange for ~7x less memory.
