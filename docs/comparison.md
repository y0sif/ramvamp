# Comparison

ramvamp runs a 30B MoE model on a Linux machine with no GPU in about 4 GB of
RAM by reading its routed experts from NVMe with explicit io_uring and
O_DIRECT reads, and it publishes every speed figure cold, inside a memory cap,
with the drive it ran on ([benchmarks.md](benchmarks.md)). Other projects stream or offload MoE experts; this
page says how each differs, from that project's own documentation.

## Projects

| Project | What it is | How it handles experts | How ramvamp differs |
| --- | --- | --- | --- |
| [TurboFieldfare](https://github.com/drumih/turbo-fieldfare) | Swift + Metal runtime for Apple Silicon Macs, one model (Gemma 4 26B-A4B) in about 2 GB of RAM | Bounded parallel `pread` into a 16-slot LFU cache per layer; the resident shared expert runs on the GPU while reads land. Reports 5.5 to 6.8 tok/s decode on an 8 GB M2 MacBook Air with expert prefetch on | The same technique on Linux x86 with no GPU, through io_uring and O_DIRECT |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C/C++ engine for many models and backends | Loads by memory-mapping the model by default, so a model larger than RAM is demand-paged from disk; `--n-cpu-moe N` keeps the MoE weights of the first N layers on the CPU (`--cpu-moe` keeps all of them). No merged path streams routed experts from SSD per token; [a pull request](https://github.com/ggml-org/llama.cpp/pull/25294) proposes one | Experts are never demand-paged. ramvamp validates its output against llama.cpp on the same GGUF bytes |
| [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | Python/PyTorch library for memory-constrained GPUs | An activation-aware cache keeps hot experts on the GPU and offloads the rest to host RAM and SSD, with tracing and prefetching | Needs a CUDA GPU; ramvamp needs none |
| [MER](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER) | Rust, pre-release by its own description | NVMe, host RAM and GPU VRAM as managed residency tiers; a WGPU/Vulkan GPU path plus CPU and hybrid paths | Built around GPU tiers; ramvamp is a CPU runtime |
| [AirLLM](https://github.com/lyogavin/airllm) | Python library for Hugging Face models on small GPUs, with CPU inference since v2.10.1 and Apple silicon through MLX | Splits the model into per-layer files and keeps one layer on the GPU at a time; for Kimi K3, per-expert streaming loads only the experts a token routes to | Its memory figures are GPU VRAM; ramvamp's are a CPU process's whole host memory, page cache included, inside a cgroup cap |

## Why the technique needs fine-grained MoE

Streaming works when a token touches a small, cacheable fraction of the
weights, which is what fine-grained MoE gives. Sizes are at 4-bit.

| Model shape | Experts per layer | Per-expert blob | Worst-case reads per token |
| --- | --- | --- | --- |
| Dense 26B | none | n/a | about 13 GB: every weight, every token |
| Coarse MoE (Mixtral 8x7B) | 8, top-2 | about 90 MB | about 5.6 GB |
| Fine-grained MoE (Qwen3-30B-A3B, Gemma 4 26B-A4B) | 128, top-8 | 2.5 to 3.2 MiB | 0.8 to 1.1 GB (EXP-015, EXP-036) |

## Measurements the design borrows

Three of ramvamp's design rules rest on TurboFieldfare's measurements, quoted
from its [optimization journey](https://github.com/drumih/turbo-fieldfare/blob/main/docs/OPTIMIZATION_JOURNEY.md).
They are TurboFieldfare's numbers on its machine, not ramvamp's.

- **Explicit reads, not mmap.** A cold expert read took 9.88 ms through mmap
  demand paging and 2.79 ms through `pread`, 3.54x. In their full streaming
  simulation mmap delivered about 0.50 tok/s against 3.97 for parallel
  `pread`, about 8x. These are two separate measurements.
- **No cross-layer prefetch.** Copying one layer's routing choices predicted
  only 7% of the next layer's experts. A later router-based predictor helped
  their M2 and slowed their M5 Pro, and ships off by default.
- **Coarse overlap.** Launching each expert group as its read completed cut
  their decode from 4.799 to 4.648 tok/s and changed the output, so ramvamp
  computes all cache hits, then all misses.

Streaming has an energy cost. [arXiv:2508.06978](https://arxiv.org/abs/2508.06978)
estimates that offloading MoE weights to SSD raises per-token decode energy by
up to about 12x against an HBM baseline.
