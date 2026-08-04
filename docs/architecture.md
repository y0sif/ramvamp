# Architecture (v0 ground truth)

Agreed 2026-08-01 after the design-phase research, corrected 2026-08-03 after
the phase-5 measurement pass. This document is the source of truth for v0
implementation. Changes to it require an experiment entry or a recorded
decision. Estimates are marked as such; everything else was verified against
checkpoints, kernel/driver behavior, our own measurements, or published
measurements (sources in `docs/landscape.md` and the research notes below).

Provenance rule for this document: every number is either **measured** (with
the conditions stated) or **estimated** (marked). Numbers labelled
*provisional* were measured, but not under the experiment log's rule 2 (cold
run inside the `memory.max=3G` benchmark cgroup), so they must not be
published before being re-measured there.

## Goal

Run Qwen3-30B-A3B coherently in about 3 GB of RAM on an ordinary Linux
machine with an NVMe SSD, CPU only. The tok/s target is under review; see
"Performance model".

v0 success criterion: coherent chat, decode throughput at or above the
agreed floor (**OPEN**, currently written as >= 3 tok/s and pending the
user's decision; the derived I/O-only ceiling on the reference drive is
~2.2-2.7 tok/s, see "Performance model"), inside a cgroup with `memory.max=3G`
and `memory.swap.max=0` (zram counts as swap), cold page cache,
KL-divergence vs llama.cpp within accepted tolerance on identical weights.

Reference hardware: Intel Core Ultra 9 185H (6P + 8E + 2 LP-E, AVX2, no
AVX-512), 16 GB LPDDR5X-7467, **Micron 2400 DRAM-less QLC** NVMe
(`MTFDKBA1T0QFM-1BD1AABGB`, PCI `1344:5413`, Gen4 x4 link,
`max_hw_sectors_kb=128`). Earlier revisions of this document said "Micron
2450 Gen4 NVMe (~3.6 GB/s sequential)": wrong device, and a spec-sheet figure
rather than a measurement. Both are withdrawn.

Measured O_DIRECT random reads on that drive (all **provisional**: taken on a
machine that was not quiet, so they fail rule 2 and need re-measuring inside
the benchmark cgroup before publication):

- At the real 2.918 MiB expert stride: 1.211 GB/s at QD4, 1.349 at QD8,
  1.390 at QD16.
- Block-size sweep: 2.918 MiB ~1.35 GB/s, 8 MiB 1.86, 16 MiB 2.04, 24 MiB
  2.15.

Throughput is dominated by block size, not queue depth. Queue depth saturates
by QD4-8 (QD4 is already 87% of QD16), while raising the block size from
2.9 to 24 MiB buys +59%. This is a DRAM-less QLC drive, not a high-end Gen4
part, and the whole performance model below inherits that fact.

Two provenance facts about these numbers that the performance model has to
live with. Both are recorded in EXP-008 rather than smoothed over:

- **This table is one probe series, and the per-blob latency figures quoted
  later are another.** They were not run together and they disagree on
  level, not on shape: the QD8 latency figure implies 1.57 GB/s aggregate
  against the 1.349 GB/s tabulated here at the same depth.
- **1.59 GB/s, the constant `scripts/lfu_sim.py` and EXP-005 build the
  performance model on, does not appear in this table and cannot be
  sourced.** It is described three incompatible ways: as a sequential
  ceiling (in the simulator), as random reads at the expert stride on a
  non-quiet machine with a concurrent reader (in EXP-005), and it sits
  between this table's random-read numbers and its large-block numbers, so
  it can be neither. EXP-005 also records a quiet run of the same probe at
  1.35 GB/s while this section says the whole table came from a machine that
  was not quiet; that contradiction is unresolved and needs the probe re-run
  rather than an edit. Until then the performance model below is derived
  from the tabulated 1.211-1.349 GB/s at the chosen QD4-8 operating point.

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

## Memory contract (option B, agreed; dial revised 2026-08-03, twice)

The expert-cache dial is a total memory budget, divided by layer count to get
slots per layer. This keeps one config meaningful across models. **That is
design intent, not current fact:** `SlotPool::new(slots_per_layer,
layer_strides)` and `LayerCache::new(n_slots, n_experts)` both take slot
counts today, and wave 2 is where the configured quantity becomes bytes.

For Qwen3 (48 layers) the budget is **1,438.6 MiB of expert pool, which is
11 slots/layer**. The dial was 10 in the original design, revised up to 12 by
EXP-005 as the largest pool that fits `memory.max=3G`, and back to 11 by
EXP-012, which measured the tenant this table used to omit entirely. Gemma 4
(30 layers) gets proportionally more slots under the same byte budget; the
exact count depends on its blob stride and has not been computed.

| Tenant | Qwen3 v0 | Notes |
| --- | ---: | --- |
| Common core (mmap, read-only) | 1,023.34 MiB (audited) | touched every token, page cache keeps it resident; charged to the cgroup as `file` |
| KV cache FP16 | 384 MiB @ 4K | 96 KiB/token; linear append, 48 layers; allocated zeroed at full capacity and faulted lazily |
| Expert slot pool | 1,438.59 MiB | 11 slots x 48 layers at the real per-layer strides, page-aligned, allocated once, every page faulted at construction |
| Runtime anonymous memory | 115.1 MiB (**provisional**, EXP-012) | activations and scratch, tokenizer, thread stacks, allocator arenas; peak `anon` sampled from inside the cgroup on a live decode run whose context length, token count and page-cache state were not recorded — it fails rule 2, so it is a floor for this tenant, not a ceiling. See Open risk below |
| Subtotal | **2,961.0 MiB** (**provisional**) | 111.0 MiB headroom under `memory.max=3G` (3,072 MiB), also provisional: both cells inherit the provenance of the anon row above, and neither may be published until that row is re-measured |

The pool figures are exact arithmetic on the audited strides in
`experts/layout.json` (3,059,712 B on the 24 Q6_K-down layers, 2,654,208 B on
the other 24), not estimates. Against 1,522.44 MiB of fixed tenants
(1,023.34 + 384 + 115.1). Only the `pool MiB` column is exact: 115.1 of
those 1,522.44 MiB is provisional, so the `subtotal MiB` and
`vs 3,072 MiB` columns below are **provisional** too, including the 19.8
MiB overshoot that moved the dial.

| slots/layer | pool MiB | subtotal MiB | vs 3,072 MiB |
| ---: | ---: | ---: | ---: |
| 10 | 1,307.81 | 2,830.25 | 241.75 spare |
| 11 | 1,438.59 | 2,961.03 | 111.0 spare |
| 12 | 1,569.38 | 3,091.82 | **19.8 over** |

That 19.8 MiB is why the dial moved back: **12 slots/layer does not fit**
once anonymous runtime memory is counted, and EXP-005's fit arithmetic
counted only the mmap'd core and the KV cache.

Hit-rate provenance matters more here than the hit rates themselves.
`scripts/lfu_sim.py` resolves a routing step one expert at a time, so it
protects only the experts already fetched within that step and can evict an
expert the same token is about to request. `LayerCache::plan` receives all
`top_k` ids in one call and pins the whole step, which cannot happen.
Replaying the shipped cache over the same four traces gives **50.02% at 10
slots and 54.48% at 12** against the simulator's 44.8% and 49.9% (EXP-005
Correction), so the simulated sweep is a **lower bound**, and batch-pinned 10
slots beats sequential 12. The corrected marginal value of 12 slots over 10
is **+4.46 points for 261 MiB**, not the +5.1 the simulator suggested. There
is still no knee: marginal value falls monotonically, so the cgroup ceiling
is the binding constraint rather than diminishing returns.

**The hit rate at the shipped 11 slots/layer has not been measured.** It is
bracketed by 50.02% and 54.48%; interpolation is not a measurement, and the
sweep is re-run at 11 before any hit rate is attached to the dial.

**Open risk:** the 115.1 MiB was sampled on a decode run whose context length
and token count are not recorded, and the KV cache faults lazily, so that
figure includes only the KV rows the run actually touched. It is a floor for
this tenant, not a ceiling, and the 111.0 MiB of spare is not yet proven at
4K context with the slot pool wired in. If the real figure exceeds 226 MiB
the dial drops to 10 slots/layer, or the KV cache goes Q8. Needs the
re-measurement EXP-012 asks for before the 11-slot dial is treated as
settled.

16 slots/layer is the next real step (simulated 58.1% hit; the batch-pinned
figure at 16 has never been computed) but its 2,092.5 MiB pool puts the
subtotal at **3,614.9 MiB**. Halving the KV cache to Q8 is not enough either:
3,422.9 MiB, still 350.9 MiB over the 3,072 MiB ceiling with nothing spare.
**16 slots/layer needs a 4G cgroup.** It is therefore an experiment about
larger machines rather than a tuning step on this one, and it is filed that
way, not shipped.

Budget escape hatches, in the experiment backlog and not in v0: Q8 KV
(halves KV, and is a necessary but not sufficient condition for 16
slots/layer), fewer slots/layer. A global slot pool shared across layers is
**closed as a no**, see "Recorded decisions from phase-5 measurement".

## Expert streaming and cache

- Common weights: `mmap` read-only. Routed experts: explicit reads, never
  demand paging. Prior art, stated with the qualifiers that make it
  comparable: TurboFieldfare measured **3.54x** on cold expert reads
  (9.88 ms via mmap faults vs 2.79 ms explicit), and **~8x** end to end in
  their full-token simulator (0.50 vs 3.97 tok/s). Those are two different
  measurements, not one. A separate report of 377 MB/s through faults vs
  2.8 GB/s explicit comes from koren1712's Windows/CUDA fork on **PCIe 3.0**,
  posted as a comment in llama.cpp discussion #23324, not from the RFC
  itself. None of these are ours; see `docs/landscape.md`.
- Reads: io_uring + O_DIRECT with `register_files` and plain `opcode::Read`.
  Interrupt-driven completions; no SQPOLL/IOPOLL (they burn a core the GEMVs
  need).
- **Registered buffers (`ReadFixed`) are out.** Reasons, in order of weight:
  1. Registering the slot pool pins the whole pool with `FOLL_LONGTERM`
     (1.4-1.6 GiB at the dials under consideration; 1,438.6 MiB at the
     shipped 11 slots/layer). That memory is unreclaimable and
     uncompactable, inside a 3 GB budget whose entire point is to stay
     small. Self-defeating for this project.
  2. `RLIMIT_MEMLOCK` under systemd defaults is 8 MiB soft *and* hard, and
     raising the hard limit needs `CAP_SYS_RESOURCE`. Since kernel 6.14 the
     SQ/CQ ring memory is charged against the same limit. A design that only
     works after the user edits a systemd unit is not a design.
  3. The measured benefit is small: ~70 us of CPU per 3 MiB read at our block
     size, about 4.5% of one core out of 22 (provisional, EXP-008).
- Queue depth: small, QD4-8, now chosen on measured grounds rather than
  assumed, but on the *shape* of the throughput sweep rather than its
  absolute level. Queue depth saturates early (1.211 GB/s at QD4 vs 1.390 at
  QD16 at the expert stride, so QD4 is 87% of QD16), and deeper queues buy
  latency rather than bandwidth: measured per-blob p50 is 2.34 ms at QD1 and
  15.56 ms at QD8 (provisional). The earlier "~1 ms per blob when the queue
  is fed" figure was an estimate and is withdrawn; real per-blob latency is
  worse than that at every depth. Since the decode loop waits on all misses
  (see "Decode loop"), latency is the quantity that matters, which argues for
  the low end of the range. **Caveat, unresolved:** those latency figures
  come from a different probe series than the throughput sweep and the two
  disagree on level. 2.34 ms for one 2.918 MiB blob is 1.31 GB/s at QD1,
  which is *above* the tabulated 1.211 GB/s at QD4, so as written the latency
  data contradicts "the drive saturates by QD4" as a claim about absolute
  throughput. Both series are provisional; see EXP-008.
- O_DIRECT is a budget-correctness requirement, and this is now measured
  rather than argued: reading the same 1.4 GiB of experts inside the same
  cgroup peaked at **1,092.2 MiB** buffered versus **5.0 MiB** with O_DIRECT
  (provisional, EXP-009). Buffered reads charge the page cache to our cgroup
  and thrash it. The transition itself has not landed: the decode loop still
  reads experts with buffered `pread`, which is why no rule-2 baseline exists
  yet (EXP-006).
- **O_DIRECT can be silently downgraded to buffered I/O, so it must be
  verified at runtime.** On btrfs a read can fall back to `filemap_read()`
  with no error and a full byte count returned. Four confirmed triggers:
  1. Misaligned offset, length, or *buffer address*. btrfs requires 4096
     (its `sectorsize`), not the device's 512-byte logical block size.
  2. Compressed extents.
  3. DUP or RAID data profiles.
  4. A destination buffer whose pages have not been faulted in. btrfs runs
     direct reads with page faults disabled (`fs/btrfs/direct-io.c`) and
     falls back to the buffered path when it cannot fault the destination.
     The slot pool must therefore be touched at construction.

  Alignment cannot be probed portably either: `statx(STATX_DIOALIGN)` is
  unimplemented on btrfs, f2fs and erofs set the mask with zero values, and
  NFS fabricates an answer without asking the server. tmpfs (since 6.6) and
  loop-backed filesystems accept an O_DIRECT open and then do buffered I/O.
  **Consequence: the runtime must assert empirically at startup that expert
  reads are not populating the page cache**, rather than trusting that the
  O_DIRECT open succeeded. The exact probe is deferred to implementation.
- Cache policy: per-layer slot arrays with **LFU over frequency counters
  indexed by expert id, sized `n_experts`, whose counts survive eviction**
  (ghost history). That detail is the policy, and it is where the win comes
  from. Simulated on real routing traces at 10 slots/layer (EXP-005):
  ghost-LFU 44.8%, per-slot LFU 42.6%, LRU 42.6%, Belady offline-optimal
  55.8%. Those are the simulator's sequential lower bound; the shipped cache
  replayed over the same traces gives 50.02% at 10 slots (EXP-005
  Correction), and the policy ranking is unaffected. Per-slot LFU, which is
  what the old wording "LFU eviction with recency tie-break" describes, is
  worth **-1.7 to 0.0 points against LRU** at the slot counts measured: tied
  at 42.6% at 10 slots, and 55.4% against LRU's 57.1% at 16. It is not a
  weaker win than ghost history, it is a small loss, which is the strongest
  form of the argument for indexing counters by expert id. Counters cost
  `n_experts * u32` per layer (512 B for Qwen3, 24 KiB across 48 layers). No
  cross-layer prefetch (measured ~7% predictability upstream).
- Concurrency invariant: a slot owned by an in-flight read or by queued
  compute is never reassigned. This is not hygiene, it is a correctness
  requirement with a measured failure mode: aliasing two concurrent O_DIRECT
  reads onto one buffer makes btrfs fail checksum verification, measured at
  13-27% spurious `EIO` plus increments to the filesystem's persistent
  `corruption_errs` counter. An explicit free list measured 0 `EIO` across
  4,000 reads at QD 4/8/16/32 (provisional, EXP-007; the 13-27% is a spread
  across repeats on a machine that was not quiet, so read it as "frequently,
  not always" rather than as a rate). Index arithmetic is not sufficient,
  because completions arrive out of order.
- Expert reads must treat `EIO` as a **retryable, typed error**, not as an
  impossible outcome. Direct I/O returns real short reads and real errors;
  code that assumes success is code that corrupts a token silently.
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
- I/O: the io_uring reactor runs **inline on the coordinator thread** in v0.
  It submits, reaps completions, and flips slot-ready flags between compute
  phases. A dedicated E-core reactor thread is an **experiment**, not a
  premise. Earlier revisions of this document assumed the dedicated thread
  was free; it is not. TurboFieldfare measured a dedicated I/O executor at
  8.59 vs 8.42 ms and a 4-worker I/O pool as mixed across repeats, rejecting
  both; flash-moe measured +4.6% for a persistent pool (flash-moe citation
  needs a source link, see `docs/landscape.md`). The evidence is genuinely
  mixed, and io_uring changes the calculus (submission is cheap and batched,
  so there is less work to move off-thread than in a pread design). So it
  becomes a measurement.
- Signalling cost, measured (provisional, EXP-010), for the
  reactor-to-compute handoff: futex p50 3.1 us at 0.07 cores; atomic spin
  502 ns but 1.03 cores burned; `std::sync::mpsc` p99 237 us. The spin path
  buys ~2.6 us of latency for a whole core, which is one of the six the
  GEMVs need. The shipped pool takes both: a bounded spin then a futex wait,
  whose full publish-to-barrier round trip measures p50 1.71 us with the
  workers warm and therefore inside the spin window (provisional, EXP-010).
  That figure is a whole-pool round trip and is not comparable with the
  single wake/wait pairs above; it does not measure the futex path.
- CPU topology must be **derived at runtime from `thread_siblings_list`**,
  never inferred from core ids. On the reference 185H,
  `/sys/devices/cpu_core/cpus` is `0-11` with SMT pairs (0,5) (1,2) (3,4)
  (6,7) (8,9) (10,11), so the physical primaries are `[0, 1, 3, 6, 8, 10]`
  (not the even ids, and not the first six). E-cores 12-19 have L3; the LP
  E-cores 20-21 sit on the SoC tile with no L3 at all, so a flag handed off
  there crosses the fabric. Detection must degrade to "no pinning" on any
  machine whose topology does not match the expected shape.
- LP E-cores (no L3, SoC tile): never used by ramvamp threads.
- E-cores joining the compute pool for independent expert GEMVs (not
  barrier-coupled) is a backlog experiment; llama.cpp evidence says
  barrier-coupled E-cores cost 20-30%.

## Decode loop (per token, per layer)

1. Attention (QK-norm, RoPE, GQA over FP16 KV) + router on resident weights.
2. Top-8 expert IDs + renormalized weights.
3. Cache plan: hits, misses, eviction victims.
4. Submit all misses to io_uring as one batch. Then dispatch **all** cache-hit
   experts to the compute pool as one unit (Qwen3 has no shared expert, so
   hits are the overlap work).
5. Wait for **all** miss reads to complete, then dispatch the miss experts as
   one unit. Weighted-sum reduction, residual, next layer.
6. After layer 48: final norm, lm_head, softmax/sampling (greedy path must be
   deterministic).

Steps 4-5 are deliberately coarse. Earlier revisions said "miss blobs compute
as their slots fill", i.e. per-expert progressive execution as completions
land. That is a **measured-and-rejected** design, and three independent
implementations converged on the coarse two-phase shape instead:

- TurboFieldfare's DEC-17 implemented per-expert progressive execution and
  measured 4.799 -> 4.648 tok/s **with divergent output**, then disabled it.
  Their DEC-18 hit-first split measured a 14.4% advantage over the
  alternative ordering.
- flash-moe waits on all reads before a single batched dispatch.
- The llama.cpp pread prototype stalls its GPU on one event per layer.

This matches TurboFieldfare's general finding that fine-grained overlap
measures slower than coarse overlap. Their DEC-17 root cause is not published;
the divergent output is consistent with letting completion order drive
reduction order, but we did not verify that. Either way, our fixed-order
staged reduction keeps the result bit-exact regardless of the order experts
are actually computed in, so the coarse shape costs nothing in determinism and
buys back the scheduling.

Sampler defaults come from the checkpoint's generation_config (temp 0.7,
top_p 0.8, top_k 20); greedy override for validation.

## Prefill (sequential sweep, our improvement over TF)

Measured coverage for this model (Layered Prefill, arXiv 2510.08055): a
128-token chunk activates 86% of each layer's experts; 512 tokens ~100%.
Chunked prefill therefore approaches "read every expert once per chunk per
layer" no matter how it is scheduled. So:

- Prefill bypasses the expert cache entirely.
- Layer-major chunks of up to 512 tokens. Per layer: group rows by expert
  (mul_mat_id style), then stream the layer's expert file front-to-back
  through a small ring of streaming buffers, computing each expert against
  all its routed rows as it arrives.
- The win is read granularity, not sequentiality per se. On the reference
  drive, large blocks measured 2.04 GB/s at 16 MiB and 2.15 at 24 MiB versus
  ~1.35 at the 2.918 MiB expert stride: **+51% at 16 MiB** (provisional,
  EXP-008).
  Buffer size for the streaming ring is therefore a tuned parameter, in the
  phase-6 backlog.
- Expected: ~8.2-8.6 s per 512-token chunk sweep (~17.6 GB of expert data)
  at 2.04-2.15 GB/s, roughly 60 tok/s prefill (estimate derived from
  provisional bandwidth; measured entry required). The earlier "~4.9 s,
  roughly 100 tok/s" figure assumed the withdrawn 3.6 GB/s number and is
  superseded. TF's design (random tile fetches through the decode cache)
  achieved ~28 tok/s, so the sequential sweep is still the right call.
- Decode cache starts cold after prefill; acceptable, first tokens warm it.
  Replaying the prompt into the cache during prefill is closed as a "no",
  see "Recorded decisions from phase-5 measurement".

## Validation protocol vs llama.cpp

Same GGUF bytes on both sides. Gates, in order:

1. Tensor-level: repacked expert slab bytes == GGUF slice bytes (exact).
2. Kernel-level: our Q4_K/Q6_K/Q8_K dot products vs scalar reference within
   documented tolerance; alignment assumptions unit-tested.
3. Logit-level: KL divergence vs llama.cpp logits on a fixed set of 8
   prompts. `scripts/kl_vs_reference.py` enforces **four** conditions, and
   the gate passes only when all four hold:

   1. mean full-vocab KL(llama.cpp ‖ ramvamp) **<= 3e-2**;
   2. **every individual prompt <= 6e-2** — a mean over 8 prompts hides
      one blown prompt behind seven good ones;
   3. **top-1 agreement on every prompt**, gated rather than merely
      reported — a flipped argmax is a behavioural change at a KL the
      mean tolerates;
   4. **all 8 prompts actually scored** — a prompt dropped for a
      tokenizer mismatch or a short reference dump shrinks the gate's
      denominator, so it fails the gate instead of quietly leaving it.

   The mean is always reported next to the intra-engine scalar/AVX2 A/B
   as the noise floor.

   Revised twice, each as a recorded decision with evidence in EXP-004:
   the mean on 2026-08-03 (1e-3 -> 3e-2), and conditions 2-4 on
   2026-08-04. The original 1e-3 predates implementation and is
   unachievable without operation-identical arithmetic: EXP-004 measured
   mean 1.04e-2 against a 0.5-1.3e-2 noise floor from dot-product float
   ordering alone, flat in context depth through 3492 tokens.

   **Why 6e-2 for the per-prompt ceiling.** EXP-004's scalar/AVX2 A/B
   (`RAMVAMP_FORCE_SCALAR=1`, same binary, dot-product accumulation order
   the only difference) is the largest float-reordering perturbation this
   codebase can produce short of an algorithmic change, and it moved the
   worst per-prompt *cross-engine* KL to 3.4e-2. So 6e-2 sits 1.76x above
   the largest perturbation ever measured here and 2.2x above the worst
   status-quo prompt (2.72e-2, single_00): it catches one blown prompt,
   and reordering every dot product in the engine is not enough to trip
   it.

   **Caveat on condition 3, recorded rather than smoothed over: the
   newly-gated top-1 agreement has no measured margin behind it.** Nothing
   in this repo recorded the top1-vs-top2 logit gap on these prompts, and
   EXP-004 reports top-1 for the AVX2 side only. A prompt sitting on a
   near-tie would therefore fail condition 3 on a change that is pure
   float noise, which makes condition 3 — not the ceiling — the plausible
   flake vector here. Mitigation shipped with the decision:
   `scripts/kl_vs_reference.py` now records each prompt's top1-vs-top2 gap
   on both sides in `kl_results.json` and prints the tightest one. It is
   reported, never gated; if a future top-1 FAIL lands on a prompt whose
   recorded margin was already small, that is a near-tie and not a
   regression. First measurement (2026-08-04, phase-4 build, `--skip-longs`):
   the tightest gap is **0.228 nats** on single_05 (`def fibonacci(n):`) on
   the ramvamp side and 0.494 nats on the llama.cpp side. Nothing sits on a
   knife edge: the tightest gap is roughly 20x the mean cross-engine KL
   (a scale comparison, not a bound), so condition 3 is not currently
   fragile. That is now a recorded number instead of an assumption, which
   was the point.

   PASSED in phase 4 against llama.cpp b10217 full-vocab reference dumps
   (`models/llamacpp-ref/`, recomputable via
   `scripts/kl_vs_reference.py` with no llama.cpp install): mean
   1.039e-2, worst prompt 2.72e-2, top-1 8/8, 8/8 prompts scored.
4. Greedy smoke: first N tokens identical on short prompts (expected to
   diverge eventually from fp reordering; report length, do not gate).
   Measured in EXP-003: 3/3 prompts matched all 16 generated tokens
   character-identically, exceeding the "first tokens" bar.
5. Perplexity on a standard slice within noise of llama.cpp same-quant.
   Reference banked (EXP-004): PPL 6.3810 +/- 0.16588, llama-perplexity
   b10217, wiki.test.raw, `-c 512 --chunks 40`; corpus and log in
   `models/llamacpp-ref/`. Phase 7 implements the ramvamp side.

## Performance model (rebuilt 2026-08-03 from measured inputs)

The previous version of this section derived a "4-8 tok/s expected" band from
an assumed ~3 GB/s of drive bandwidth and an assumed 40-60% hit rate. Both
assumptions have now been measured, and both were optimistic. The band is
withdrawn.

- In-RAM compute ceiling on the 185H: 15-25 tok/s (bandwidth-scaled
  **estimate**; no direct public benchmark exists). Unchanged, still an
  estimate.
- Decode I/O volume: worst case **1,097 MB/token** (1.02 GiB: 8 experts x 48
  layers, 24 layers at a 3,059,712 B stride and 24 at 2,654,208 B). This one
  is exact arithmetic on audited strides, not a measurement.
- Decode hit rate: **simulated, not measured end to end**, and the two
  available figures are a lower bound and a replay of the shipped code, not
  one number. `scripts/lfu_sim.py` gives 44.8% at 10 slots and 49.9% at 12
  (EXP-005, 556 decode tokens of real routing traces, simulation only);
  replaying the shipped `io/cache.rs` over the identical traces the way the
  runtime actually calls it gives **50.02% at 10 and 54.48% at 12** (EXP-005
  Correction). At the shipped 11 slots/layer the hit rate has **not** been
  measured and is bracketed by those two.
- Decode I/O time, derived rather than measured. Miss bytes are estimated as
  `(1 - hit) x 1,097 MB`, which assumes misses are spread across the two
  stride classes in proportion to accesses; on the one point where EXP-005
  reports both, that assumption is within 1% of the simulator's exact byte
  count. At the tabulated QD4-8 bandwidths (1.211-1.349 GB/s):

  | dial | hit % | miss MB/token | ms/token | io-only tok/s |
  | --- | ---: | ---: | ---: | ---: |
  | 10 slots, batch-pinned | 50.02 | 548 | 406-453 | 2.21-2.46 |
  | 12 slots, batch-pinned | 54.48 | 499 | 370-412 | 2.42-2.70 |
  | no cache | 0 | 1,097 | 813-906 | 1.10-1.23 |

  So the current I/O-only band at the shipped dial is roughly **2.2-2.7
  tok/s**, bracketed by the 10- and 12-slot rows. Every input to that band is
  provisional: the bandwidths are EXP-008's non-quiet probe, and the hit
  rates are a trace replay rather than a decode run. Note this band replaces
  the earlier "2.4-2.9 tok/s", which was derived at the 1.59 GB/s constant
  that EXP-008 records as unsourced; at 1.59 GB/s these same volumes give
  2.90-3.18 tok/s. Re-derive the whole table once the bandwidth probe is
  redone under rule 2.
- **This band is drive-dependent and must never be published without the
  device.** The same design and the same ~500 MB/token on a 3.5 GB/s drive
  computes to roughly 7 tok/s. The reference machine has a DRAM-less QLC
  part; a mainstream TLC Gen4 drive would roughly double these numbers. Any
  headline tok/s figure ships next to the drive it was measured on.
- **I/O is probably not the binding constraint yet, but no single
  measurement says so.** The two figures that suggest it come from different
  machine states and must not be subtracted from each other (experiment log,
  rule 3): phase-4 decode is ~2 s/token on the 185H (EXP-004 diagnostic:
  warm cache, single thread, uncgrouped, and the expert reads in it are
  buffered `pread` served largely from that warm cache), while the 690 ms of
  uncached expert I/O is a *simulated* figure at an unsourced bandwidth
  constant (EXP-005, no cache). What can be said without combining them is
  that a single-threaded decode step measured in seconds sits an order of
  magnitude above an I/O budget estimated in hundreds of milliseconds, so
  compute-parallelism work matters at least as much as the cache does.
  Neither the cache dial nor the bandwidth probe can be evaluated against
  tok/s until the compute side is threaded and one run produces both halves
  under rule 2.
- **Floor for success: OPEN.** The criterion is currently written as 3 tok/s,
  which is above the derived I/O-only ceiling on the reference drive at the
  tabulated bandwidths (2.2-2.7 tok/s). Only the unsourced 1.59 GB/s
  constant puts 3 tok/s within reach at all, and then only at a slot count
  that no longer fits the memory budget. This needs the user's decision, not
  a silent edit. The three options are: keep 3 and accept it is unreachable on
  this drive, restate the floor per-drive, or lower it. Recorded here as
  unresolved.

Prior-art anchors, with the qualifiers that were previously missing:

- llama.cpp RFC #23324 (same design, pread sidecar) reports 13 tok/s on a
  16 GB M1 Pro. This does **not** bound our band in either direction: it is
  **Q6_K, not Q4_K_M**; **48 slots/layer, not 11**; GPU-accelerated; and the
  author explicitly qualifies it as "after warmup", with the macOS page cache
  in the read path. It is a single unreplicated self-report with no cold
  counterpart. Cite it as evidence the design works, never as a throughput
  target.
- TurboFieldfare reports 5.1-6.3 tok/s in ~2 GB on an 8 GB M2 Air, on Apple
  unified memory with Metal compute. Closer to comparable than the M1 Pro
  number, still a different machine class.
- Our traces do **not** reproduce TurboFieldfare's 66.6% hit rate at 16
  slots/layer, but the comparison is **not like-for-like** and the size of
  the gap is unknown. Our 58.1% is the simulator's sequential lower bound
  (EXP-005); the batch-pinned figure at 16 slots, which is what the shipped
  cache would actually do, has never been computed, and at 10 and 12 slots
  batch pinning was worth about 5 points. So the disagreement is at most 8.5
  points and is in the absolute level rather than the curve: the 16-to-24 and
  16-to-32 deltas match their published shape. Until the 16-slot replay is
  run, no claim of ours may lean on either their absolute number or on the
  size of the gap.

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

## Recorded decisions from phase-5 measurement (2026-08-03)

Closed with measured "no". These were open questions or backlog items; they
are neither now, and they should not come back without new evidence.

- **Global slot pool shared across layers: no.** The best possible static
  per-layer split, at the same total memory, buys **+0.53 points** of hit
  rate at the operating point (EXP-005). Not worth the loss of fixed-stride
  simplicity.
- **Replaying the prompt into the expert cache during prefill: no.** Worth
  **+0.09 points** (EXP-005). Prefill continues to bypass the cache.
- **Per-expert progressive execution as reads land: no.** See "Decode loop";
  measured and rejected upstream, with divergent output.
- **Registered io_uring buffers (`ReadFixed`): no.** See "Expert streaming
  and cache".

## Experiment backlog (numbered entries when run)

- 16 slots/layer **in a 4G cgroup**. The arithmetic does not close in 3G
  under any single saving once anonymous runtime memory is counted: 3,614.9
  MiB with the FP16 KV cache, 3,422.9 MiB with Q8 KV, against a 3,072 MiB
  ceiling (EXP-012). This is an experiment about larger machines, not a
  tuning step on the reference one. Its hit-rate value is also unquantified
  until the batch-pinned replay is run at 16 slots.
- Re-run the slot sweep at 11 slots/layer, batch-pinned, so the shipped dial
  has a hit rate of its own instead of a bracket (EXP-012).
- Re-measure peak `anon` at 4K context with the slot pool wired in, to prove
  or disprove the 111.0 MiB of headroom the 11-slot dial leaves (EXP-012).
- Dedicated E-core io_uring reactor thread vs inline on the coordinator
  (evidence upstream is mixed both ways; see "Thread topology")
- Larger read granularity for the phase-6 prefill sweep (16-24 MiB buffers;
  measured +51% at 16 MiB over the 2.918 MiB expert stride, provisional,
  EXP-008)
- E-cores in compute pool for non-barrier expert GEMVs
- `fadvise`/`readahead` tuning for the prefill sequential sweep
- Prefill chunk size sweep: 128 vs 256 vs 512
- Re-measure the O_DIRECT bandwidth probe under rule 2 (cold, inside the
  benchmark cgroup, quiet machine) and re-derive the performance model. This
  is the highest-value item on the list: the 1.59 GB/s constant the model
  was built on cannot be sourced, and the two existing probe series disagree
  by 17% at the same queue depth (EXP-008)

Dropped from the backlog:

- ~~Slot budget sweep: 8 vs 10 vs 12 vs 16 slots/layer~~: done, EXP-005 (as
  a simulated lower bound; the shipped dial's own point is back on the
  backlog above, see EXP-012).
- ~~Peak RSS of scratch + program + tokenizer against the headroom the dial
  leaves~~: superseded by EXP-012. The tenant is `anon`, program text is
  file-backed and lands with the mmap'd core, and the figure is now measured
  at 115.1 MiB. The remaining question is the 4K-context re-measurement
  listed above.
- ~~Global slot pool vs per-layer~~: closed as a no, see above.
- ~~Expert layout reordering by co-activation~~: **downgraded to inactive.**
  The 2.23x figure previously cited from llama.cpp #18758 does not describe
  co-activation reordering: it describes contiguous per-expert interleaving
  of the up/gate/down projections, which our one-blob-per-expert format
  already gives us by construction. Actual co-activation reordering measured
  roughly zero or negative three separate times upstream. Not worth revisiting
  without a new result.

## Deferred to implementation (not design-blocking)

- `gguf_dump` audit of the pinned file (done 2026-08-01, see "Model pin" and
  "Source quantization"); the manifest writer freezes the `tensor_types` map
- Exact Q8_K activation quantization scheme (mirror ggml's)
- Chat template rendering: vendor the 2507 template, snapshot-test against
  `transformers` reference renders
- Slot-ready signalling mechanism between the reactor and the compute pool.
  Measured candidates (provisional, EXP-010): futex p50 3.1 us at 0.07
  cores, atomic spin 502 ns at 1.03 cores burned, `std::sync::mpsc` p99
  237 us. The compute pool already ships the bounded-spin-then-futex
  compromise; the reactor-to-compute direction is what is still open. With
  the reactor inline in v0 this is a smaller decision than it was; it becomes
  load-bearing if the dedicated-thread experiment wins.
- The empirical O_DIRECT / page-cache-residency assertion at startup (see
  "Expert streaming and cache"); the probe mechanism is not yet specified.
