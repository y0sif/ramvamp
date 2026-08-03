# Competitive landscape

Researched 2026-08-01, before any code was written. Attributions corrected
2026-08-03 after phase-5 re-checked the sources. This document records why
ramvamp exists given what already ships.

Citation rule for this document: a number gets the machine, the quant, and
the cache size it was measured with, or it does not get quoted. Several
figures below were previously quoted without those qualifiers and read as
stronger than they are.

## Differentiation statement

Given that **TurboFieldfare** (Swift/Metal, Apple-Silicon-only, single-model)
and **MoE-Infinity** (Python/CUDA, server-class) exist, and **llama.cpp has
no merged SSD expert streaming** (open feature requests plus one unmerged RFC
prototype; TurboFieldfare measured its mmap fallback 3.54x slower per cold
expert read, and ~8x slower end to end in their full-token simulator),
ramvamp earns its existence because:

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
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | OSS, C++ | mmap demand paging when the model exceeds RAM; `--n-cpu-moe` offloads experts to RAM, not SSD | Very active | No merged SSD expert streaming; [discussion #19163](https://github.com/ggml-org/llama.cpp/discussions/19163), issues [#19825](https://github.com/ggml-org/llama.cpp/issues/19825) and [#20757](https://github.com/ggml-org/llama.cpp/issues/20757) request it; closest work is the unmerged RFC [#23324](https://github.com/ggml-org/llama.cpp/discussions/23324) (pread expert-slot prototype; 13 tok/s Qwen3-30B-A3B self-reported on a 16 GB M1 Pro, **Q6_K, 48 slots/layer, GPU-accelerated, "after warmup"**; see the caveat below) |
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

1. Explicit reads beat mmap demand paging. Two distinct measurements, which
   must not be merged into one: **3.54x** per cold expert read (9.88 vs
   2.79 ms) and **~8x** end to end in their full-token *simulator* (0.50 vs
   3.97 tok/s). Quoting "~8x" for the read comparison, as earlier drafts of
   `docs/architecture.md` did, overstates it by 2.3x.
2. A 16-slot-per-layer LFU cache roughly halves expert I/O
   (166 to 88 ms/token; LFU beat LRU 72.6 to 64.8 ms/token). Their reported
   hit rate at 16 slots is 66.6%. **Our traces do not reproduce that
   absolute level**: EXP-005 measures 58.1% at 16 slots on Qwen3-30B-A3B,
   8.5 points low, though the 16-to-24 and 16-to-32 deltas match their
   published shape. Use their curve shape, not their absolute number.
3. Cross-layer expert prediction fails (~7% accuracy): no speculative
   prefetch. Cache-on-reuse is the whole game.
4. Overlap I/O only with compute guaranteed to run (cache hits, shared
   expert); fine-grained overlap measured slower than coarse overlap. Their
   DEC-17 per-expert progressive execution measured 4.799 to 4.648 tok/s
   *with divergent output* and was disabled; their DEC-18 hit-first split
   measured a 14.4% advantage over the alternative ordering.
5. Warm page caches fake wins. End-to-end cold measurements decide what
   ships (hence the cgroup benchmark methodology in this project).
6. A dedicated I/O executor is not automatically a win: they measured 8.59
   vs 8.42 ms for one, and a 4-worker I/O pool as mixed across repeats, and
   rejected both. (Cross-project note, not theirs: flash-moe measured +4.6%
   for a persistent pool, so the evidence is genuinely mixed. ramvamp treats
   it as an open experiment rather than a premise.)

Note that TurboFieldfare's headline throughput, 5.1-6.3 tok/s in ~2 GB, is on
an 8 GB M2 Air: Apple unified memory, Metal compute, and a macOS page cache
in the read path. It is the closest comparable in the field and still not the
same machine class as a CPU-only x86 laptop with a discrete NVMe device.

## Citations corrected 2026-08-03

Phase-5 re-checked every borrowed number that load-bearing design decisions
rest on. Four were wrong or under-qualified. They are recorded here so the
bad versions do not come back.

- **"llama.cpp RFC measured 377 MB/s via faults vs 2.8 GB/s explicit."** The
  numbers are real, but they are not the RFC's. They come from
  **koren1712's Windows/CUDA fork on PCIe 3.0**, posted as a comment in
  llama.cpp discussion [#23324](https://github.com/ggml-org/llama.cpp/discussions/23324),
  not from the Metal RFC that the discussion is about. Different OS,
  different accelerator, and a PCIe generation slower than our reference
  machine. Quote it as a directional data point, with those qualifiers.
- **"13 tok/s on a 16 GB M1 Pro."** Self-reported by the prototype's author
  in the same discussion, unreplicated, and it is **Q6_K not Q4_K_M**,
  **48 slots/layer not 12**,
  GPU-accelerated, and explicitly "after warmup" with the macOS page cache
  in the read path. There is no cold counterpart. It does not bound
  ramvamp's expected band in either direction and must not be used as a
  throughput target.
- **TurboFieldfare's "~8x mmap penalty"** applied to cold reads. It is 3.54x
  for reads; ~8x is their end-to-end simulator figure. See "Design oracle"
  item 1.
- **llama.cpp #18758's "2.23x cold-I/O gain" attributed to co-activation
  layout reordering.** The 2.23x actually describes **contiguous per-expert
  interleaving of the up/gate/down projections**, which ramvamp's
  one-blob-per-expert format already provides by construction. Co-activation
  reordering itself measured roughly zero or negative on three separate
  upstream attempts. The backlog item in `docs/architecture.md` is
  downgraded accordingly.

**Needs verification:** *flash-moe* is cited in `docs/architecture.md` for
two measurements (+4.6% for a persistent I/O worker pool; waiting on all
reads before a single batched dispatch). No repository URL has been recorded
for it in this document. Add the link, or drop the citation, before either
number appears in anything published.

## Known trade-offs to state honestly

- SSD-streamed MoE costs more energy per token than RAM-resident inference
  (up to ~12x per [arXiv:2508.06978](https://arxiv.org/html/2508.06978v1)).
  Reads do not wear SSDs, but battery life on laptops will be affected.
- The approach stays I/O-bound: TurboFieldfare's tuned M2 path still spends
  ~half its per-token time on expert reads. The slowdown versus a
  fits-in-RAM engine is **drive-dependent and larger than the "2-3x" this
  document previously claimed**. On ramvamp's reference machine (Micron 2400,
  DRAM-less QLC) the measured I/O-only ceiling is 2.4-2.9 tok/s against a
  15-25 tok/s in-RAM compute estimate, so 5-10x on that device; a mainstream
  TLC Gen4 drive would roughly halve the gap. The memory saving, ~7x, is the
  part that does not depend on the drive. See the performance model in
  `docs/architecture.md` for the derivation and its provisional status.
