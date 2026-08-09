# Competitive landscape

Researched 2026-08-01, before any code was written. Attributions corrected
2026-08-03 after phase-5 re-checked the sources, and again 2026-08-09 before
publication. This document records why ramvamp exists given what already
ships.

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
| [TurboFieldfare](https://github.com/drumih/turbo-fieldfare) | OSS, Swift + Metal | Explicit pread + per-layer LFU expert cache + custom quantized kernels; 5.1-6.3 tok/s in ~2 GB on an 8 GB M2 Air | Active; **~5,495 stars and 305 forks on 2026-08-09**, repo created 2026-07-17 | Apple-only by design (Swift, Metal, unified memory); one pinned model |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | OSS, C++ | mmap demand paging when the model exceeds RAM; `--n-cpu-moe` offloads experts to RAM, not SSD | Very active | No merged SSD expert streaming; [discussion #19163](https://github.com/ggml-org/llama.cpp/discussions/19163), issues [#19825](https://github.com/ggml-org/llama.cpp/issues/19825) and [#20757](https://github.com/ggml-org/llama.cpp/issues/20757) request it; closest work is the unmerged RFC [#23324](https://github.com/ggml-org/llama.cpp/discussions/23324) (pread expert-slot prototype; 13 tok/s Qwen3-30B-A3B self-reported on a 16 GB M1 Pro, **Q6_K, 48 slots/layer, GPU-accelerated, "after warmup"**; see the caveat below), and PR [#25294](https://github.com/ggml-org/llama.cpp/pull/25294) (per-layer expert slots, async I/O worker pool, O_DIRECT), **open, not merged, as of 2026-08-09** |
| [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | OSS, Python/PyTorch | Activation-aware expert cache across GPU/host/SSD tiers, prefetching | Active, academic | Server-oriented; needs CUDA and large host RAM; not a 2 GB consumer play |
| [Micro-Expert-Router](https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER) | OSS, Rust | io_uring + O_DIRECT NVMe expert streaming, CPU kernels | Early-stage | Closest precedent, but its performance numbers are stated by the author to be theoretical projections, not measurements |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) / [candle](https://github.com/huggingface/candle) | OSS, Rust | Full inference engines with quantization and device offload | Active | No SSD expert streaming; useful as building blocks and kernel references |
| AirLLM / "LLM in a flash" | OSS / research paper | Layer-by-layer disk streaming / flash-aware sparsity | Dormant / never productized | Layer streaming of dense models is unusably slow; the technique needs MoE sparsity |

## Why the technique needs fine-grained MoE

| Model shape | Experts per layer | Per-expert blob (4-bit) | Worst-case reads per token | Verdict |
| --- | --- | --- | --- | --- |
| Dense 26B | none | n/a | ~13 GB | Unusable: multiple seconds per token at NVMe speeds |
| Coarse MoE (Mixtral 8x7B) | 8, top-2 | ~90 MB | ~5.6 GB | Unusable: experts too large to stream or cache |
| Fine-grained MoE (Gemma 4 26B-A4B, Qwen3-30B-A3B) | 128, top-8 | ~2.5-3.4 MB | ~0.8-1.1 GB worst case depending on model; a small LFU cache absorbs roughly half. **Measured live and cold at 53.0-59.3%** at the shipped 11 slots/layer, across five context rungs, no trend in context (EXP-023; reference machine, Qwen3-30B-A3B Q4_K_M, in-cgroup). Two lower figures exist and are different quantities, not disagreements: replaying the shipped cache over EXP-005's recorded routing traces gives 50.02% at 10 slots and 54.48% at 12, and `scripts/lfu_sim.py`'s sequential simulator sits about 5 points under that replay at 44.8% and 49.9% | The regime this project targets |

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
2. A 16-slot-per-layer LFU cache roughly halves expert I/O, 166 to 88
   ms/token, on their 8 GB M2 Air. Their reported hit rate at 16 slots is
   66.6%. Two caveats this document's own citation rule demands. First,
   **the quant those ms/token figures were taken at is not recorded in our
   notes**, so the pair is quotable only as "their cache halves their expert
   I/O on their machine", not as a rate. Second, an earlier version of this
   item said "LFU beat LRU 72.6 to 64.8 ms/token", which is self-contradictory
   as written: lower is better, so those numbers say LFU *lost*. Which policy
   is which cannot be recovered from our notes, so the pair is **withdrawn**
   pending a re-read of their log. The LFU-versus-LRU question is settled on
   our own traces anyway: EXP-005's simulator, replaying recorded routing
   traces rather than measuring the runtime, gives ghost-history LFU 44.8%
   against LRU's 42.6% at 10 slots/layer, and finds that per-slot LFU without
   ghost history is worth -1.7 to 0.0 points against LRU. Both are simulated
   rates, quotable for the ranking and not as the runtime's hit rate. **Our
   traces also do not
   reproduce their 66.6% absolute level**: EXP-005 gives 58.1% at 16 slots on
   Qwen3-30B-A3B, 8.5 points low, but that is not a like-for-like comparison
   either, because 58.1% is the simulator's sequential lower bound and the
   batch-pinned replay figure at 16 slots has never been computed (batch
   pinning was worth about 5 points at 10 and 12 slots). Use their curve
   shape, not their absolute number, and do not quote the size of the gap.
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
   rejected both. (Cross-project note, not theirs:
   [flash-moe](https://github.com/danveloper/flash-moe) measured a persistent
   expert `pread` pool at **4.28 against 4.09 tok/s**, which is the +4.6% this
   project quotes, and 3.81 against 3.97 ms/layer, verdict "keep". Their
   machine is a 48 GB MacBook Pro, their model is Qwen3.5-397B-A17B with
   **2-bit experts**, and there is **no slot count to state** because they run
   no expert cache of their own and read through the macOS page cache. Their
   pool dispatches `pread` through GCD, not io_uring. So the evidence is
   genuinely mixed, and ramvamp treats it as an open experiment rather than a
   premise.)

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
  **48 slots/layer not 11**,
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

**Verified 2026-08-09, was "needs verification":** *flash-moe*, cited in
`docs/architecture.md` for two measurements, is
[danveloper/flash-moe](https://github.com/danveloper/flash-moe). Both
citations check out against the repository, so both survive.

- **"+4.6% for a persistent I/O worker pool."** Its `results.tsv` records
  "Persistent expert pread pool beats dispatch_apply": 4.28 against 4.09
  tok/s, which is the +4.6%, and 3.81 against 3.97 ms/layer, output
  identical, verdict "keep". The qualifiers this document's citation rule
  demands travel with it: a 48 GB MacBook Pro, Qwen3.5-397B-A17B at **2-bit
  experts**, and **no slot count**, because flash-moe keeps no expert cache
  and reads through the macOS page cache. It is also a GCD `pread` pool, not
  io_uring, which is the axis that matters most for whether the result
  transfers here.
- **"Waits on all reads before a single batched dispatch."** Its
  `docs/plan-io-experiments.md` describes exactly that shape: `aio_suspend`
  "blocks until all 4 complete (single wait vs 4 thread joins)". This one is
  a structural claim rather than a number, so the citation rule has nothing
  to attach to it.

## Known trade-offs to state honestly

- SSD-streamed MoE costs more energy per token than RAM-resident inference
  (up to ~12x per [arXiv:2508.06978](https://arxiv.org/html/2508.06978v1)).
  Reads do not wear SSDs, but battery life on laptops will be affected.
- **Decode** stays I/O-bound: TurboFieldfare's tuned M2 path still spends
  ~half its per-token time on expert reads. That is a claim about decode, and
  it does not carry to prefill on the swept path: EXP-017 measured expert I/O
  at 1.7% of a 512-token swept prefill and 0.5% at 1891 tokens, with attention
  taking 61.3% and 85.2%. **EXP-017's own qualifier travels with those four
  figures.** They are warm, uncgrouped, and taken on a machine its Method
  records as busy, and EXP-017 states head-on that they are not publishable
  numbers: what survives is the shape of the split and ratios within a single
  run, never the absolute seconds. They are superseded directionally as well.
  On the phase-7 kernel, measured cold and in-cgroup at 4K on the reference
  machine (Qwen3-30B-A3B Q4_K_M, 11 slots/layer), the prefill split is
  **expert compute 42.1%, projections 22.8%, attention 20.4%**, elementwise
  12.2%, expert I/O 2.5% (EXP-021 Note 9, with EXP-023 reproducing 41.8 / 22.8
  / 20.8 in a different session). Attention is not the wall in prefill any
  more, it is the third largest term. Those are different entries, sessions
  and kernels, so the only licensed statement is that direction and not a
  curve drawn through both. Expert I/O staying small in prefill is confirmed
  cold on our own path: **MEASURED**
  at ctx 512, expert I/O is 44.4% of a decode token and 2.8% of a prefill token
  (EXP-023, five context rungs, cold, in-cgroup, hygiene PASS). The slowdown
  versus a fits-in-RAM engine is **drive-dependent and larger than the "2-3x"
  this document previously claimed**.

  On ramvamp's reference machine (Micron 2400, DRAM-less QLC, Qwen3-30B-A3B
  Q4_K_M at the shipped 11 slots/layer) decode **MEASURES 1.46 to 2.19 tok/s
  cold** across five context rungs (EXP-023: 2.19 / 1.91 / 1.82 / 1.75 / 1.46
  at 64 / 512 / 1,024 / 2,048 / 3,961 prompt tokens, medians of three scored
  runs at `--max-new 64`). That band is a fact about the session EXP-023
  measured rather than a property of the build on this drive: EXP-025 re-ran
  the **byte-identical** binary on the same prompt files a day later and read
  1.85 against 1.91 at ctx 512 and 1.33 against 1.46 at 3,961, and its own
  fused ladder over the same five rungs (2.16 / 1.87 / 1.74 / 1.65 / 1.43) is
  a second ladder rather than five more points on this one. Against a
  **15-25 tok/s ESTIMATED** in-RAM compute ceiling, an estimate with no
  direct public benchmark behind it, that is
  roughly **7-17x on that device**, and the width of that range is mostly the
  softness of the estimate rather than anything measured. A mainstream TLC Gen4
  drive would narrow the gap; by how much is **UNKNOWN**, since no such drive
  has been measured here.

  **The "derived I/O-only ceiling of 2.8-3.4 tok/s" this document used to build
  that ratio on is withdrawn.** It bracketed the 10- and 12-slot rows while 11
  is what ships; its hit rates were a replay of the shipped cache over recorded
  routing traces rather than a decode run (this document called them
  "simulated", which was imprecise, because the simulator is a different and
  lower figure); its bandwidth input was EXP-019's 1.55-1.69 GB/s where decode's own
  effective rate is **2.16 GB/s DERIVED** (EXP-023); and, decisively, **an
  I/O-only ceiling computed as `1 / expert_io` overstates**, because EXP-023
  Note 4 establishes that the `expert io` bucket is a residual left after hit
  compute has already covered part of the read. No replacement band is offered.

  All bandwidth figures here come from a threaded-`preadv` probe rather than
  io_uring, so they characterise the drive and not the runtime's submission
  path, and EXP-024 records that a per-file bandwidth spread on this drive is a
  fact about the session that measured it rather than a stable property. The
  memory saving, **~6x** (17.35 GiB of model bytes against a ~2.9 GiB resident
  budget), is the part that does not depend on the drive. See the performance
  model in `docs/architecture.md` for what is measured and what it bounds.
