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
machine with an NVMe SSD, CPU only. **The tok/s floor is SETTLED as of
2026-08-08: it is stated per drive, not as one number.** What it resolves to
on the reference machine is in "Performance model".

v0 success criterion: coherent chat, decode throughput **published against the
drive it was measured on** rather than against a single global floor, inside a
cgroup with `memory.max=3G` and `memory.swap.max=0` (zram counts as swap),
cold page cache, KL-divergence vs llama.cpp within accepted tolerance on
identical weights.

The floor was written as `>= 3 tok/s` from phase 1 to phase 9 and was never
met at any measured rung. Three phases derived independently that it is not
reachable on this drive by code alone (phase 5 on bandwidth, phase 7 as a
stated risk, phase 9 on the post-fusion compute headroom), and the decision
was escalated three times without being taken. It is taken now, and the
reason it is stated per drive rather than lowered is that **the device is the
dominant term and this document already requires every tok/s figure to name
it** ("Performance model", the drive-dependence rule). A single number
would contradict that rule at the headline while enforcing it in the body.

What decode actually does on the reference drive is **MEASURED cold** and is
below that floor: **1.46 to 2.19 tok/s** across five context rungs at the
shipped 11 slots/layer (2.19 / 1.91 / 1.82 / 1.75 / 1.46 at 64 / 512 / 1,024 /
2,048 / 3,961 prompt tokens, medians of three scored runs, `--max-new 64`,
EXP-023), and **1.43 to 2.16 tok/s** over the same five rungs on phase 9's
fused branch in a later session (2.16 / 1.87 / 1.74 / 1.65 / 1.43, EXP-025).
Those are **two sessions and not one curve**: EXP-025 re-ran EXP-023's
byte-identical binary on the same prompts and read it 3.1% slower at 512 and
8.9% slower at 3,961, so a decode tok/s here describes a session as well as a
device. The earlier "~2.8-3.4 tok/s derived I/O-only ceiling" that stood here
is **withdrawn**; "Performance model" records why, and no replacement headline
band is offered, because an I/O-only ceiling both overstates what is reachable
and is not what the floor is written against.

Reference hardware: Intel Core Ultra 9 185H (6P + 8E + 2 LP-E, AVX2, no
AVX-512), 16 GB LPDDR5X-7467, **Micron 2400 DRAM-less QLC** NVMe
(`MTFDKBA1T0QFM-1BD1AABGB`, PCI `1344:5413`, Gen4 x4 link,
`max_hw_sectors_kb=128`). Earlier revisions of this document said "Micron
2450 Gen4 NVMe (~3.6 GB/s sequential)": wrong device, and a spec-sheet figure
rather than a measurement. Both are withdrawn.

Measured O_DIRECT reads on that drive, **cold inside the benchmark cgroup on a
quiet machine, hygiene PASS** (EXP-019, `scripts/io_probe.py`, three scored
runs over four real installed layer files spanning both stride classes). These
supersede on level the provisional EXP-008 figures this section used to carry:

| geometry | pattern | GB/s |
| --- | --- | --- |
| one expert blob (2.918 / 2.531 MiB), 8 outstanding, as decode reads | random | 1.55-1.69 |
| one expert blob, 8 outstanding | sequential | 1.58-2.37 |
| 8-expert window (23.34 / 20.25 MiB), 2 outstanding, as the prefill sweep reads | sequential | 1.60-2.37 |
| whole matrix, every cell | both | 1.54-2.37 |

**Queue depth in that probe is emulated with N OS threads issuing blocking
`preadv`, not io_uring.** So it characterises the drive and the filesystem,
not `crates/core/src/io`'s submission path, and no dial in the runtime may be
set from it without a measurement of its own.

Three findings, and the first two retire what this section used to say:

- **Block size does not dominate. It barely matters.** Going from one expert
  blob to an 8-blob block is neutral on one probed file and 15 to 16 percent
  *worse* on the other three when the read is sequential. EXP-008's "+51% at
  16 MiB", which the prefill section below leaned on, is **refuted**, and
  under either reading of what EXP-008 measured: read randomly, the same
  change gains 15 to 25 percent, which is also not +51%.
- **What predicts throughput is total bytes in flight**, which queue depth and
  block size both move and move interchangeably: at matched bytes outstanding,
  K=4/QD8 and K=8/QD4 land within noise of each other. The drive holds its
  peak up to roughly **100 MB outstanding** and gives up 15 to 18 percent past
  ~170 MB. Both shipped dials sit under that ceiling: the decode ring is at
  most 24.5 MB outstanding, the prefill sweep 49.0 MB. It is a constraint on
  *raising* them, not a reason to change them.
- **Per-file variance persists and is unexplained.** `layer_00` at 1.578 GB/s
  against `layer_20` at 2.271 in the same sweep, with byte-identical extent
  geometry (both 398 extents, mean 984,027 B, zero physically adjacent
  successor pairs), so fragmentation as `filefrag` reports it does not predict
  it. `docs/benchmark-machine.md` records a 1.46x spread between the same two
  files from an earlier measurement, so it reproduces.

Sequentiality is worth restating precisely, because it is easy to get
backwards. At the single-blob size, reading in order is worth 1.34x to 1.50x
on three of the four files. At an 8-blob block it is worth nothing (0.99x to
1.01x), because the block has already captured the same locality. The two
effects substitute for each other rather than adding.

This is a DRAM-less QLC drive, not a high-end Gen4 part, and the whole
performance model below inherits that fact.

Two provenance facts the performance model still has to live with:

- **EXP-008's two probe series disagreed with each other**, not only with this
  one: its QD8 per-blob latency figure implies 1.57 GB/s aggregate against the
  1.349 GB/s its own throughput table gives at the same depth. Both series are
  superseded on level by the table above. The per-blob latency figures quoted
  later in this document are still EXP-008's and still provisional, because
  EXP-019 timed whole blocks rather than individual blobs at the decode
  geometry.
- **1.59 GB/s, the constant `scripts/lfu_sim.py` and EXP-005 build on, is
  still unsourced.** EXP-019 does not find it either. It was described three
  incompatible ways (a sequential ceiling, in the simulator; random reads at
  the expert stride on a non-quiet machine, in EXP-005; and it sat between
  EXP-008's random-read and large-block numbers, so it could be neither), and
  nothing has since sourced it. It does fall inside EXP-019's measured range,
  which makes it a lucky guess rather than a measurement. No derived figure in
  this document uses it.

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

The expert-cache dial is a total memory budget, divided by the summed
per-layer blob strides to get slots per layer. This keeps one config
meaningful across models, and **it ships**: `DEFAULT_CACHE_BYTES`
(`crates/core/src/model/forward.rs`) is a 1,440 MiB total byte budget, and
`--cache-bytes` sets it. Earlier revisions of this section said the byte
conversion "is still owed", which was true when `SlotPool::new` and
`LayerCache::new` were the only entry points; those still take slot counts,
but they are no longer what the user configures. EXP-023 exercised the byte
dial directly at `1570M` and `1701M` and asserted the slot count each budget
bought (12 and 13, checked against every run's stderr), so the conversion is
measured as well as shipped.

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
| Runtime anonymous memory | 115.1 MiB (**provisional**, EXP-012) + 387 KiB (phase 7) + 48 KiB (phase 9) | activations and scratch, tokenizer, thread stacks, allocator arenas; peak `anon` sampled from inside the cgroup on a live decode run whose context length, token count and page-cache state were not recorded — it fails rule 2, so it is a floor for this tenant, not a ceiling. See Open risk below. The 387 KiB is decode's per-shard attention scratch, `min(n_kv_heads, shards)` carves of 132,096 B for 528,384 B total against the one 132,096 B carve that preceded it (EXP-020). The 48 KiB is phase 9's fused expert fan-out, which sizes the FFN scratch for a whole routed window instead of for one expert at a time: `gate_up` goes `2 x 8 x 768 x 4 B` = **49,152 B** against 6,144 B (**+43,008**) and `acts_q8k_moe` goes `8 x (768 / 256)` = 24 `BlockQ8K` of 292 B = **7,008 B** against 876 B (**+6,132**), so **+49,140 B = 47.99 KiB**. Both are `Vec`s in `ForwardState::with_config`, allocated once at construction and shared by all 48 layers; nothing is per token and nothing is per layer. Total added since EXP-012 sampled the row: 396,288 + 49,140 = **445,428 B (435 KiB)**. All of it is exact arithmetic on top of a provisional figure, so it does not make the row less provisional |
| Subtotal | **2,961.0 MiB** (**provisional**) | 111.0 MiB headroom under `memory.max=3G` (3,072 MiB), also provisional: both cells inherit the provenance of the anon row above, and neither may be published until that row is re-measured. **Correction (EXP-023):** measured cold at 3,961 prompt tokens plus 64 generated, the cgroup peaks at **2,929.3 MiB**, so this row overpredicts by 31.7 MiB. The prediction is not edited, because the overshoot is not yet attributed; see the Correction under the slot table below |

**The prefill sweep is not a fifth row.** Its streaming ring is a
sub-allocation of the expert slot pool above, borrowed while the pool is
idle rather than allocated (see "The prefill arena"), so it moves no cell in
this table. At the default dials, 8 experts per window and 2 windows in
flight, the ring is `2 x 8 x stride`: 46.7 MiB on a 3,059,712 B layer and
40.5 MiB on a 2,654,208 B one under the per-layer `sweep_layer` carve, and a
flat 46.7 MiB on every layer under the `PrefillSession` path, which sizes one
ring for the widest layer of the whole session (see "The prefill arena").
Either way the bytes are already counted in the 1,438.59 MiB. The prefill
driver additionally needs staging for a chunk's expert outputs, an
`[n_rows][top_k][hidden]` buffer, plus the rest of a chunk's batched
activations. **Which tenant that comes out of is decided: another
sub-allocation of the same arena, not an addition to the runtime-anonymous
row.** `ExpertStream::begin_prefill` opens a `PrefillSession`, which carves
one span laid out as `[scratch | pad | ring]` (scratch at the slab base, ring
at the next 4096 boundary past it) and hands the two halves out disjointly,
so the driver writes staging while it consumes swept experts. At the v0 dials
and the default 512-row chunk that scratch is **81,728,516 B (77.94 MiB)**, of
which 33,554,432 B is the `[n_rows][top_k][hidden]` staging proper and 792,576
B is phase 7's six per-shard attention score buffers; the
arithmetic is pinned by a test (EXP-016, EXP-020). It sits inside the 1,438.59
MiB pool row alongside the ring, so chunked prefill still moves no cell in this
table, and neither does sharding its attention.
The chunk size that sets the figure is a dial, and the 128/256/512/1024 sweep
that picks it is still owed.

The pool figures are exact arithmetic on the audited strides in
`experts/layout.json` (3,059,712 B on the 24 Q6_K-down layers, 2,654,208 B on
the other 24), not estimates. Against 1,522.44 MiB of fixed tenants
(1,023.34 + 384 + 115.1). Only the `pool MiB` column is exact: 115.1 of
those 1,522.44 MiB is provisional, so the two predicted columns below are
**provisional** too — and the 19.8 MiB overshoot that moved the dial is worse
than provisional, it is measured to be wrong in sign (EXP-023, and the
Correction below).

| slots/layer | pool MiB | predicted subtotal MiB | predicted vs 3,072 MiB (**REFUTED at 12**) | measured `memory.peak` at 3,961 + 64 tokens (EXP-023) |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 1,307.81 | 2,830.25 | 241.75 spare | not measured |
| 11 | 1,438.59 | 2,961.03 | 111.0 spare | **2,929.3 MiB**, 142.7 spare |
| 12 | 1,569.38 | 3,091.82 | ~~19.8 over~~ **wrong in sign; measured 13.6 spare** | **3,058.4 MiB**, 13.6 spare |

The prediction column is kept only as the record of what was believed; the
measurement column is what is true. **The 19.8 MiB overshoot at 12 slots was
the reason the dial moved back to 11, and it does not exist**: EXP-023
measured 12 slots/layer fitting, cold, at full context, with 13.6 MiB spare.
The prediction that moved the dial was arithmetic on a fixed-tenant sum that
is about 33 MiB high. EXP-005's fit arithmetic counted only the mmap'd core
and the KV cache, which is the separate error that first moved the dial the
other way. **The dial still does not move** — see the Correction below for why
13.6 MiB of margin in one session is not a licence to spend it.

**Correction (2026-08-06, EXP-023): the 12-slot row is wrong in sign. 12
slots/layer fits.** Measured cold inside `memory.max=3G` with
`memory.swap.max=0` at 3,961 prompt tokens plus 64 generated, hygiene PASS on
all four runs, no OOM: 3,058.4 MiB, which is 13.6 MiB **under** the cap rather
than 19.8 MiB over. The predicted column overshoots by 33.4 MiB at 12 slots
and by 31.7 MiB at 11, and two near-equal overshoots one slot apart are the
signature of a wrong constant in the fixed-tenant sum rather than of wrong
per-slot arithmetic. The per-slot arithmetic corroborates that: this table's
130.79 MiB per slot against 129.05 MiB measured between the two arms above.
**Derived**, and only partly: about 6.7 MiB of the 33.4 is the KV cache, which
is allocated at the 4,096-position capacity and faulted lazily, so at the
4,025-position high-water mark it holds 377.34 MiB of its 384 MiB row. Roughly
27 MiB is unaccounted, and the 115.1 MiB anonymous row is the only provisional
tenant in the sum, so it is where to look. That is not a licence to rewrite it
to 88 MiB: it still needs the rule-2 re-measurement EXP-012 asks for, and this
correction narrows that job rather than doing it.

**The dial does not move on this, and the shipped default stays 11
slots/layer with `--cache-bytes` unchanged.** 13.6 MiB of margin at full
context is smaller than EXP-018's unexplained 99-105 MiB residual and smaller
than the 33.4 MiB error this correction is fixing, and it is one prompt in one
session. EXP-023 also measures the throughput the extra slot buys, which is
what a future decision would weigh against that margin: **+4.7% decode at 512
tokens and +4.1% at 3,961**, with the 512 step's scored ranges overlapping and
only the 3,961 step separating at three runs each.

**13 slots/layer was measured at 512 tokens only and has no 4K run, so it is
not a candidate.** EXP-023 records it at 2,862.3 MiB and 2.06 tok/s at 512
prompt tokens, which is +7.9% decode over 11 slots and the only step in that
sweep whose ranges separate cleanly. Extrapolating its 4K peak from the
12-slot arm would put it near 3,189 MiB, over the cap, but **that is
arithmetic on a corrected constant that has not itself been re-measured** and
no 13-slot run at full context exists. Nothing may be decided about 13 until
one is taken.

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

~~**The hit rate at the shipped 11 slots/layer has not been measured.** It is
bracketed by 50.02% and 54.48%; interpolation is not a measurement, and the
sweep is re-run at 11 before any hit rate is attached to the dial.~~

**Correction (2026-08-06, EXP-023): it is measured, at 53.0% to 59.3%.** Five
cold decode runs at 64, 512, 1,024, 2,048 and 3,961 prompt tokens, hygiene
PASS, at the shipped dial with the slot count asserted per run: **59.3 / 54.0
/ 53.0 / 58.4 / 56.5 percent** of 24,192 expert requests each. The sequence is
not monotone in context, so read it as a scatter around roughly 56% rather
than a curve. **Three of the five rungs land at or above the top of the
50.02-54.48 bracket** (59.3, 58.4, 56.5), the other two land inside it near
its top (54.0, 53.0), and none falls below its floor. So the bracket is
superseded at 11 slots rather than confirmed by it.

Two provenance points survive the correction and getting them backwards would
overstate the agreement. The bracket's endpoints are **replay** figures, not
simulator output: 50.02% and 54.48% come from replaying the shipped
`io/cache.rs` over EXP-005's four routing traces. It is `scripts/lfu_sim.py`
that gives 44.8% and 49.9%, so **the simulator is the roughly 5-point
underestimate and the bracket above is already corrected for it**; that record
stands and is the reason a simulated hit rate is never quoted here directly.
And the bracket is a replay over 556 decode tokens of recorded routing while
EXP-023 is a live decode of 63 tokens per run across five different prompts,
so the two are different populations and agree in level and order rather than
like for like. The slot dial is now measured either side of 11 as well: 56.1%
at 12 slots and 57.9% at 13 at 512 tokens, and 58.4% at 12 slots at 3,961, so
one extra slot is worth roughly 2 points.

**Open risk:** the 115.1 MiB was sampled on a decode run whose context length
and token count are not recorded, and the KV cache faults lazily, so that
figure includes only the KV rows the run actually touched. It is a floor for
this tenant, not a ceiling, and the 111.0 MiB of spare is not yet proven at
4K context with the slot pool wired in. If the real figure exceeds 226 MiB
the dial drops to 10 slots/layer, or the KV cache goes Q8. Needs the
re-measurement EXP-012 asks for before the 11-slot dial is treated as
settled. Three things have since been charged against that unproven spare and
only one is large: EXP-018's unexplained 99-105 MiB residual, phase 7's
387 KiB of decode attention scratch, and phase 9's 48 KiB of fused expert
scratch. The first is the one to worry about; the other two together are
435 KiB.

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
- Queue depth: small, QD4-8, and `RING_ENTRIES` in
  `crates/core/src/io/stream.rs` is 8. **The measured justification is now
  EXP-019, not EXP-008.** The decode ring issues at most 8 single-blob reads
  at once, which is 24.5 MB outstanding, and EXP-019 puts that inside the flat
  part of the drive's curve: throughput turns on total bytes in flight, holds
  its peak to roughly 100 MB, and falls 15 to 18 percent past ~170 MB. So the
  shipped depth is at the peak rather than past it. Two limits on that
  reassurance, both worth stating: EXP-019 swept queue depth only at the
  8-blob block size, so the decode geometry has **no queue-depth curve of its
  own** and its QD2 and QD4 points are unmeasured; and its queue is threaded
  `preadv`, not io_uring, so it says nothing about `SINGLE_ISSUER` /
  `DEFER_TASKRUN` submission behaviour. An io_uring queue-depth experiment
  inside the runtime is on the backlog for exactly those two gaps.

  **Update (2026-08-06, EXP-023): the first gap is closed and the second is
  not.** EXP-023 swept queue depth at K=1, the single-blob size decode issues,
  on the same four files: the curve rises from QD 1 to QD 2 and is flat from
  QD 2 on `layer_20`, `layer_06` and `layer_21`, and from QD 4 on `layer_00`.
  Decode's own concurrency **derives** to 3.26-3.76 misses per layer step
  across five context rungs, which sits on that plateau, so the shipped depth
  is confirmed at the decode geometry and not merely at the sweep's. The queue
  is still `threaded-pread`, so the io_uring question is exactly as open as it
  was and `RING_ENTRIES` still must not move on a probe.

  **The submission queue is not the binding constraint on the decode path, and
  that half is closed by geometry rather than by measurement.** `RING_ENTRIES`
  is 8 (`crates/core/src/io/stream.rs:172`) and `top_k` is 8, so a decode layer
  can request at most 8 experts and therefore submit at most 8 miss reads;
  `ExpertStream::begin_layer` submits exactly one read per miss and **refuses
  to start a step while any earlier read is outstanding**
  (`crates/core/src/io/stream.rs:1455-1530`), so only one layer of one token
  ever has reads in flight. Every miss a decode layer can ever have therefore
  fits the ring simultaneously, and raising `RING_ENTRIES` **cannot** increase
  decode's bytes in flight. Only more concurrent misses could, and that needs
  speculative cross-layer prefetch, which is forbidden (measured ~7%
  predictability). **Derived from code geometry, not measured**, which is
  exactly why it settles the submission-side half and nothing else: it does not
  close the **drive-side** question at the decode block size. Whether 24.5 MB
  of single-blob reads at QD8 is where this drive wants to be is still
  unmeasured at that geometry, since EXP-019 swept depth only at the 8-blob
  block size. The phase-8 sweep's `scripts/io_probe.py` step
  (`scripts/phase8_decode_sweep.sh:1120`) is what addresses that half.
  Deeper queues also buy latency rather than bandwidth: measured per-blob p50
  is 2.34 ms at QD1 and 15.56 ms at QD8 (**still provisional, still EXP-008**;
  EXP-019 timed whole blocks, not individual blobs, so it does not replace
  these). The earlier "~1 ms per blob when the queue is fed" figure was an
  estimate and is withdrawn; real per-blob latency is worse than that at every
  depth. Since the decode loop waits on all misses (see "Decode loop"),
  latency is the quantity that matters, which independently argues for the low
  end of the range. EXP-008's internal contradiction on this point (2.34 ms
  for one 2.918 MiB blob is 1.31 GB/s at QD1, above its own tabulated 1.211
  GB/s at QD4) is resolved in the direction the latency series pointed:
  EXP-019 measures 1.55 to 1.69 GB/s at that geometry, above both.
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
- What the pool runs: expert and projection GEMVs, and since phase 7 attention
  as well — over **rows** in prefill and over **kv heads** in decode. Two
  things follow. `ComputePool::run` runs a job inline as a single shard when
  `rows < shards()`, so a fan-out submitted at fewer units than shards
  silently serialises; submit at `pool.shards()` and map the indices yourself.
  And the decode axis has only `n_kv_heads` = 4 units at the v0 pin, so two of
  the six pinned cores take an empty range on every decode attention call.
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
   Attention itself fans across the compute pool by **kv head** above 8 cached
   positions, so it is not a serial prologue to the expert work any more; see
   "Attention: the kernel and its fan-out".
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

**Compute and I/O trade against each other through this overlap, and that is a
property of the shape rather than an accident.** Step 4's hit compute is the
only work hiding step 5's outstanding reads, and the `expert io` bucket
measures the part of those reads that hit compute did not cover — it is a
residual, not the drive's busy time (EXP-023 Note 4). So **making the hit
compute faster does not buy its full saving: it leaves less to hide behind, and
some of the same read latency becomes visible instead.** MEASURED cold and
paired at ctx 512 (EXP-025): the fused fan-out handed back 3.32 s of GEMV over
63 tokens and `expert io` took 1.85 s of it straight back, on **identical**
request, hit, miss and byte counts in both arms — the drive did exactly the
same work. That is why a compute win of this size reached the token at 3,961
tokens of context and did not reach it at 512, and it is why the next lever at
mid context is overlap rather than more compute. It also bounds every future
compute change on this path: **price a GEMV saving net of the exposed io it
uncovers, never gross.**

Sampler defaults come from the checkpoint's generation_config (temp 0.7,
top_p 0.8, top_k 20); greedy override for validation.

### How decode's GEMVs fan out, and what phase 9 refuted about them

**This subsection carries two kinds of number and they must not be mixed.**
The *cold* figures are MEASURED cold, in-cgroup, hygiene PASS, paired against
the phase-8 reference binary in one session (EXP-025); they are results. The
*warm* figures are MEASURED WARM — ctx 512, 63 decode tokens, three runs per
arm, medians with ranges, pooled GEMV bucket = `projections + experts +
lm_head`, the router serial and excluded — and under rule 2 they stay
**diagnostics**. Artifacts: cold in
`scratch/phase9/sweep-20260807-203558/` and
`scratch/cold-bench/p9-20260807-203558-*.json`; warm in
`scratch/phase9/wave0-baseline/*.err` and
`scratch/phase9/wave1-measure/rep/*.err`.

Step 4-5's expert work fans out **once per expert phase, not once per matrix**
(`70cf304`). A phase's routed experts issue one job for every gate and up
together and one for every down; `attn_q` joins `attn_v` the same way. A layer
went from 28 fan-outs to five when its plan is all hits or all misses and seven
when it splits, and a token from 1,345 to between 241 and 337 — MEASURED cold
at **332.8 fan-outs a token at ctx 512 and 331.3 at 3,961**, inside that range,
against the pre-fusion 1,345 (DERIVED: 48 layers x 28 plus one `lm_head`).
Fusion stays strictly inside one `run_plan` call: crossing the hit/miss
boundary would make a resident expert's arithmetic wait on a missing expert's
read.

**Cold, and this is the result (EXP-025).** Decode's phase-split GEMV bucket,
`expert compute + projections`, medians of 3 scored runs with their scored
ranges, fused branch against the byte-identical phase-8 reference binary run
back to back in the same session at 11 slots/layer:

| rung | phase-8 reference | fused fan-out | ratio | ranges |
| ---: | ---: | ---: | ---: | --- |
| ctx 512 | 16.25 s (15.68-16.39) | **12.93 s** (12.49-13.96) | **1.257x** | disjoint |
| ctx 3,961 | 15.39 s (14.71-15.72) | **13.29 s** (13.18-13.31) | **1.158x** | disjoint |

The gain is carried by `expert compute` (1.347x at 512, 1.256x at 3,961,
disjoint ranges at both). **The `projections` bucket, where the `attn_q` +
`attn_v` merge lands, does not separate at either rung** (1.134x and 1.025x,
ranges overlapping), so that half of `70cf304` is measured and unattributed.
Prefill's own GEMV is unmoved — 0.997x at 512 and 0.991x at 3,961, ranges
overlapping — which is the control that says the change is decode-only.

**What that is worth on a token is a separate question and the answer depends
on context.** End to end, cold and paired: **1.011x at ctx 512 with the two
scored ranges overlapping heavily, which is not a result**, and **1.070x at ctx
3,961 with disjoint ranges** (reference 47.57-48.76 s of `decode_s` against
44.47-46.59). EXP-025 Note 1 records that even the 3,961 separation cannot be
attributed cleanly: the unchanged `attention` bucket moved 1.16-1.18x the same
way at both rungs, and subtracting it run by run leaves 1.025x. **Quote the
range 1.025x-1.070x, not the top of it.**

**Warm, and these stay diagnostics.** Same 512 workload, pooled GEMV bucket:

| arm (warm, ctx 512) | pooled GEMV bucket, median | scored range |
| --- | ---: | --- |
| pre-fusion baseline | 14.44 s | 14.35-15.06 |
| adaptive-spin policy, pre-fusion | 14.27 s | 13.99-14.48 |
| **fused fan-out** | **11.42 s** | 11.36-11.48 |
| fused + adaptive spin | 11.38 s | 11.34-11.51 |

Warm fusion is worth **1.264x on the pooled GEMV bucket** (14.44 / 11.42,
DERIVED from the medians above; the two ranges do not overlap). The cold
1.257x and this 1.264x are close, **and they are not the same bucket**: the
pooled GEMV bucket is a set of disjoint spans strictly inside the phase split's
`projections` and `expert compute`, and the cold pairing could not use it at
all because the phase-8 reference binary predates that instrument and emits no
`decode gemv split` block. Two nested measures of the same work agreeing is
corroboration, not one number measured twice (EXP-025 Note 7).

Four things the warm arms refute, and they matter more than the 1.264x:

- **EXP-001's 9.61 GB/s (the `q4_k x q8_k` 2048x2048 row of its table) is not
  a valid reference for decode, and three code comments that used it have been
  fixed.** That fixture dots a matrix small enough to sit in **L2**; decode
  streams every expert byte **once from DRAM**. They are not the same
  quantity, and the "shortfall" between decode's throughput and 9.61 GB/s was
  an artefact of pairing them rather than a gap to be closed. Nothing in this
  document or in `crates/core` may quote 9.61 GB/s as a decode reference again.
- **The pool's ~1.3 µs barrier figure does not describe decode.** It comes from
  `pool.run(6, |_| {})` — see the `ATTENTION_FANOUT_MIN_POSITIONS` docs in
  `crates/core/src/model/forward.rs`, which is where it is recorded — a hot
  loop in which **no worker ever parks**. Decode parks on essentially every
  fan-out.
  Any arithmetic that multiplies 1.3 µs by a fan-out count to price decode's
  dispatch is arithmetic on the wrong constant.
- **Six cores do not buy 6x; decode GEMV is memory-bound, not
  dispatch-bound.** Forcing a single shard puts the pooled GEMV bucket at
  **16.36 s**, of which **16.35 s is `own`** — with one shard there is nobody
  to wait for, so `own` is the serial arithmetic and 16.35 is the figure the
  ratios below and the derivations in `kernels::gemv` use. Against the
  six-shard arm run in the same session and on the same binary that is **1.16x
  pre-fusion** (16.35 against **14.11 s**, both from
  `scratch/phase9/wave0-baseline/`, single runs rather than medians). Against
  the fused arm it is **1.43x** (16.35 against 11.42 s), which is legitimate to
  pair only because fusion moves no arithmetic, so 16.35 s is the same serial
  arithmetic either side of it — **there is no single-shard control on the
  fused binary**. Against the separate three-run pre-fusion baseline of 14.44 s
  the same control gives 1.13x, which is a cross-session pairing and is the
  weaker of the two. Fused aggregate throughput is **11.00 GB/s** (DERIVED:
  1.99 GB of weights a token over 11.42 s / 63 tokens). A dispatch-bound site
  would scale with cores. This one does not.
- **The barrier wait scales with WORK, not with fan-out count.** Fusion cut
  expert scatters **6.14x** (72,576 to 11,812) and cut expert barrier wait only
  **1.47x** (6.77 s median, range 6.67-7.21, to 4.62 s median, range
  4.55-4.63). Per-scatter wait therefore went **up** about 4.2x, from 93 µs to
  391 µs (DERIVED from those medians and counts). It was never
  wake latency, and no spin policy could reach it: workers are simply slower
  per row than the submitting thread. That is why the adaptive-spin change was
  written, measured at 1.012x against a baseline whose own spread is 1.049x,
  and **reverted** (`88e3e9d`); the code is preserved unchanged and
  cherry-pickable on `feat/pool-adaptive-spin`, to be revisited only if a later
  change makes the pool latency-bound again.

Reported and not fixed: `attn_q` plus `attn_v` is 4,096 q4_k rows then 512 q6_k
rows, so an even row split leaves the last shard about 30% long. A
cost-weighted split belongs in `shard_range`. **The cold pairing does not
settle whether that half of the change is worth keeping**: it removes 48
fan-outs a token, one per layer, out of the roughly 1,012 `70cf304` removes
(MEASURED cold: the `projections` bucket reports 9,072 fan-outs over 63 tokens,
144 a token, 3 a layer where the pre-fusion path issued 4), and the bucket it
lands in does not separate at either measured rung. Settling it needs an arm
with that half reverted and the expert-phase half kept, which EXP-025 does not
run.

## Prefill (sequential sweep, our improvement over TF)

Measured coverage for this model (Layered Prefill, arXiv 2510.08055): a
128-token chunk activates 86% of each layer's experts; 512 tokens ~100%.
Chunked prefill therefore approaches "read every expert once per chunk per
layer" no matter how it is scheduled. So:

- **Prefill bypasses the expert cache entirely**, and that is what ships:
  `PrefillMode::Sweep` is the default, and `prefill_prompt` drives the sweep
  reader (`crates/core/src/io/sweep.rs`) instead of stepping the decode cache
  one token at a time (EXP-016). The token-major path is retained and
  selectable (`--prefill token-major`, `RAMVAMP_PREFILL`), because it is the
  reference the sweep is held bit-identical against; it is the only path that
  still populates the cache during prefill. EXP-013's phase-split table
  (9,600 prefill cache requests at 45.3% hit, against 24,192 decode requests
  at 52.7%) measured that older default, so read its prefill column as a
  record of the token-major path rather than of what runs today.
- Layer-major chunks of up to 512 tokens. Per layer: group rows by expert
  (mul_mat_id style), then stream the layer's expert file front-to-back
  through a ring of large window reads, computing each expert against all
  its routed rows as it arrives. The ring is a **borrow of the expert slot
  pool, not a new allocation**; see "The prefill arena".
- **The dominant win is amortization, not read granularity.** An earlier
  revision of this section said the opposite ("the win is read granularity,
  not sequentiality per se"), and it was steering design decisions. A
  512-token chunk reads each expert **once per layer instead of once per
  token**, so expert bytes per token fall from the decode worst case of
  ~1,097 MB to ~34 MB (~17.6 GB of expert data over 512 tokens). Both
  figures are exact arithmetic on the audited strides, and the ratio between
  them, ~32x, is the same reuse counted from the other end
  (`512 tokens x top-8 / 128 experts` = 32 rows per expert per layer).
  **As reads issued per expert per layer that 32x is exact and structural; as
  bytes saved it is a ceiling**, because the decode worst case assumes every
  request misses. The token-major prefill the sweep replaced ran through the
  decode cache at 45.3% hit (EXP-013), so measured against that baseline the
  bytes the sweep displaces are `(1 - 0.453) x 1,097` = ~600 MB/token and the
  realized reduction is ~17.6x. **That prediction is now measured, cold, and
  it holds**: on the same 512-token prompt in the same session, process
  `read_bytes` fell from 239.33 GB token-major to 20.72 GB swept, a factor of
  **11.55x** (EXP-018). The swept figure is 1.11x the whole installed model,
  which is what reading each expert exactly once looks like. The measured
  ratio is below the ~17.6x predicted because the token-major arm hit its
  cache harder over 512 prompt tokens than the 45.3% EXP-013 measured over 25,
  and because `read_bytes` counts the mmap-faulted common core and the decode
  tail on both arms. The same amortization is available on RAM
  bandwidth: dotting one expert weight row against all ~32 of its routed rows
  while the row is in L1 fetches that row from RAM once instead of once per
  token. That is what the batched GEMV entry points exist for (EXP-015), and
  the prefill driver is now their caller (EXP-016).
- Read granularity is a **dead effect, not a secondary one.** The claim was
  EXP-008's **+51% at 16 MiB** (2.04 GB/s at 16 MiB and 2.15 at 24 MiB against
  ~1.35 at the 2.918 MiB expert stride, provisional). EXP-017 demoted it,
  because its denominator was contradicted by EXP-013's 1.97 GB/s under the
  real access pattern at the same stride and because EXP-008's harness was
  never committed. **EXP-019 retires it.** Measured cold under rule 2 on the
  real installed files, going from one expert blob to an 8-blob block is
  neutral on one file and 15 to 16 percent *worse* on three, sequentially; the
  whole read-pattern change from the runtime's random single-blob decode read
  to a front-to-back 8-blob sweep read is 0.97x to 1.26x by file, against the
  1.51x EXP-008 implied. **Nothing in this design may be justified by read
  granularity.** Window size stays a tuned dial (the sweep exposes it, default
  8 experts per window), and EXP-019 gives that sweep an upper bound it did
  not have: window bytes times windows in flight should stay under ~100 MB
  outstanding, which the shipped 49.0 MB does with room for a doubling.
- **The sweep's ceiling is `max(I/O, compute)` per chunk, not I/O alone.**
  The I/O half, re-derived from EXP-019 where EXP-008's numbers used to sit:
  17,553,162,240 B of expert files per 512-token chunk sweep at the 1.60-2.37
  GB/s the drive gives at the sweep's own geometry is **7.4 to 11.0 s**, which
  is roughly **47 to 69 tok/s of I/O-only prefill**. That band is wide for one
  reason, the unexplained per-file variance EXP-019 records, and it happens to
  bracket the ~60 tok/s the previous revision derived from EXP-008's 2.04-2.15
  GB/s. So the estimate barely moved, but it now rests on a measurement that
  satisfies rule 2 instead of one that did not, and it is a range rather than
  a point. Earlier revisions quoted that 60 tok/s as *the* prefill figure; it
  is not one, because it says nothing about compute. **Measured, the I/O half
  is not the binding one at
  all**: expert I/O is 1.7% of a 512-token sweep prefill and 0.5% of a
  1891-token one, of which 1.76 s and 5.06 s were blocked on the drive
  (EXP-017). Those runs were warm and uncgrouped, so the shares are what to
  read and the seconds are not publishable, but no plausible correction to
  them makes a 1.7% term the constraint. The binding constraint on prefill is
  attention, which is the bullet after this one. The earlier measured anchor
  for the compute half was
  EXP-013, where decode wall was 34.20 s with 16.63 s of I/O wait: 17.57 s
  of non-I/O time over 63 decode steps, so roughly **279 ms/token** (63, not
  the 64 that entry's prose names: the step count is recovered from its own
  decode request count, `24,192 / (48 layers x top-8)` = 63. A subtraction
  inside a single entry, so rule 3 is satisfied; that run was warm and
  uncgrouped, so the figure is provisional, and a prefill row is not a
  decode token, so treat it as an order-of-magnitude anchor rather than a
  prefill number). The sweep does not change compute per token at all, it
  changes the order the bytes arrive in. What changes compute is the batched
  expert GEMV, which is **expected** to move the expert FFN from
  RAM-bandwidth-bound (one MAC per weight fetched) to MAC-bound (~32 MACs per
  weight fetched). That transition is a mechanism, not a measurement: no
  roofline, bandwidth or arithmetic-intensity figure for the expert FFN is
  recorded in this document or the experiment log, so "RAM-bandwidth-bound"
  is where the bound is expected to sit and not where it was observed. What
  *has* been observed is the wall-clock consequence: expert compute fell
  **3.27x** against the token-major path on the same 512-token prompt in the
  same session (EXP-017, 77.81 s to 23.78 s), so the batched GEMV delivered,
  and the competing hypothesis that it was thrashing on its activation side
  is refuted. That is a time, not an arithmetic intensity, so the roofline
  framing above stays a mechanism. The reuse ratio of ~32 is exact chunk
  geometry either way, and it is per *weight*, not per byte: at Q4_K/Q6_K a
  byte holds roughly 1.33 to 2 of them. **No new tok/s prediction is published
  here, and none is needed now: the measurement has been taken.** EXP-018 pairs
  the swept path against the token-major path on one 512-token prompt, cold
  inside `memory.max=3G` with swap off, hygiene PASS on both arms, and gets
  **4.23 tok/s prefill against 1.65, a 2.56x speedup**, with process
  `read_bytes` down 11.55x and `memory.peak` at 2,570.1 MiB of 3,072. That is
  what EXP-015, EXP-016 and EXP-017 each recorded as owed. It is a single run
  per arm rather than a median of five, so read the third significant figure
  with suspicion. The earlier "~4.9 s, roughly 100 tok/s" figure assumed the
  withdrawn 3.6 GB/s number and is superseded. TF's design (random tile
  fetches through the decode cache) achieved ~28 tok/s, so the sequential
  sweep is still the right call.
- **The third term the ceiling above leaves out is attention, and it was
  measured to be the largest one (EXP-017).** Phase 7 rebuilt it; what that
  build now does is "Attention: the kernel and its fan-out" below, and this
  bullet records the measurement that sent phase 7 there. On the phase-6
  build, the driver batched the projections and the expert FFN across a
  chunk's rows but ran `attention_at` row by row on the calling thread through
  one shared score buffer, so attention was neither batched nor parallel, it
  got nothing from the six pinned P-cores, and it converted K and V from f16
  to f32 per element with no vectorization. **All four of those clauses are
  false on the current build**, and none of the shares below has been
  re-measured against it — that is EXP-021's job. Measured on the swept path
  at phase 6, attention is **61.3% of a 512-token prefill and 85.2% of a
  1891-token one**, against expert I/O at
  1.7% and 0.5% over the same runs. Per-token attention cost is 148.9 ms at
  512 tokens and 555.7 ms at 1891, a ratio of 3.73 against a prompt-length
  ratio of 3.69, which is what a quadratic total looks like measured per
  token; the implied rate is roughly 0.68 GFLOP/s against a rough 51.6 GFLOP
  *estimate* for the 512-token case. **Those runs were warm, uncgrouped and
  on a machine that was not quiet, so under rule 2 none of their seconds is
  publishable**; the shape of the split and the ratios inside a single run
  are what this finding rests on. The cold cgroup end-to-end number has since
  been taken (EXP-018, prefill 2.56x), but **not** the cold phase split, so
  attention's share cold is still inferred from warm runs. EXP-018's own
  arithmetic is at least consistent with it: 17.55 GB of expert reads at
  EXP-019's 1.60-2.37 GB/s is 7.4 to 11.0 s of drive time inside a 120.96 s
  cold prefill, or 6 to 9 percent, which leaves attention as the only
  candidate for the rest. That crosses two entries, so it is a derivation and
  not a measurement, and it is only defensible at all because both batches
  were taken minutes apart in one session on one machine.
  **Both levers this bullet named have since been taken** (EXP-020): rows are
  fanned across the pool by cost, and the conversion, the dot and the V
  reduction are vectorized with AVX2 + F16C. Warm and in process the kernel
  is 6.4x to 9.1x faster at one token's worth of attention over the 64-to-4096
  context ladder, and the prefill fan-out is an **estimated** ~5.9x on the
  attention region — which Amdahl on the shares above turns into an
  **estimated** ~2.0x on a 512-token prefill and ~3.4x at 1891. **No cold
  number exists**, so nothing here may be published; EXP-021 is reserved for
  the run that produces one.
- Decode cache starts cold after prefill; acceptable, first tokens warm it.
  Replaying the prompt into the cache during prefill is closed as a "no",
  see "Recorded decisions from phase-5 measurement". **This line predicted a
  real cost and EXP-018 saw it**: taking the prefill arena invalidates every
  layer's slot occupancy (`crates/core/src/io/stream.rs`), so decode after a
  swept prefill starts with nothing resident, where the token-major path left
  the cache warmed by the whole prompt. On a 4-token generation that is
  essentially all that is measured, and decode reads 1.38 tok/s swept against
  1.88 token-major. Steady-state decode code is unchanged and bit-identical
  between the two arms, so this is a cold-start transient rather than a
  throughput regression, but that is a **hypothesis with supporting arithmetic
  and not a measurement**: EXP-005 found cold start reaching within 2 points
  of steady state only by token 48, and 725 ms/token sits just under the 649
  to 708 ms/token of I/O alone that an empty cache implies at EXP-019's
  bandwidths. A longer `--max-new` on the same paired prompt is what would
  settle it, and it is on the backlog. "Acceptable" is still the right verdict
  and it now has a price attached.

## The prefill arena (the sweep ring is borrowed, not allocated)

The streaming ring the sweep reads through is carved out of the expert slot
pool. It is not a fifth memory tenant and not a new allocation, and that is
the load-bearing half of why chunked prefill costs zero bytes of the memory
contract.

What makes it possible:

- The slot pool is **one contiguous allocation**, 4096-aligned, with every
  page faulted at construction (~1,438.6 MiB at the shipped 11 slots/layer).
- Slot addresses inside it are `base + base_offset + pitch * slot`, and
  `pitch == stride` for this model, so the slab is a **gapless run of
  blob-sized buffers**. That holds because both Qwen3 strides are exact
  4096 multiples (3,059,712 B = 747 pages, 2,654,208 B = 648 pages). It is
  an emergent property of the installed geometry rather than a contract, so
  the carve checks it per layer and refuses a padded layout instead of
  assuming.
- The pool is **idle during prefill**, because prefill bypasses the decode
  cache. That is a recorded decision rather than a convenience: replaying
  the prompt into the cache is worth +0.09 points (EXP-005).

What that buys:

- **Zero additional bytes** against the ~111 MiB of documented (and
  provisional) headroom under `memory.max=3G`. At the default dials, 8
  experts per window and 2 windows in flight, the ring is `2 x 8 x stride`,
  which is 46.7 MiB on a 3,059,712 B layer and 40.5 MiB on a 2,654,208 B
  one, and those bytes are already inside the 1,438.59 MiB pool row. Those
  two figures are the **per-layer** carve, which is what
  `ExpertStream::sweep_layer` takes: a fresh arena sized for the one layer it
  is about to sweep. A `PrefillSession` spans every layer on one carve, so
  `ring_span` sizes its ring for the **widest** layer of the model and it is
  46.7 MiB throughout; it never shrinks to 40.5 on the narrow-stride layers.
  Both fit the same pool row, so the memory contract does not care which path
  runs; a reader reconciling the two numbers against the code does.
- **One carve serves both the ring and the driver's staging.**
  `ExpertStream::begin_prefill` returns a `PrefillSession` whose span is laid
  out `[scratch | pad | ring]`: the scratch leads, so its base is the slab
  base and inherits the pool's alignment, the ring starts at the next 4096
  boundary past it so every window offset stays legal for O_DIRECT, and
  `PrefillSession::split` hands the two out as disjoint `&mut`s. That is what
  lets the layer-major driver write staging while it consumes swept experts
  without a second allocation, and at the v0 dials and a 512-row chunk the
  scratch half of that span is 81,728,516 B, 77.94 MiB (EXP-016 measured
  80,935,940 B / 77.19 MiB; phase 7 added six 132,096 B attention score
  buffers, one per compute shard, for 792,576 B — see "Attention: the kernel
  and its fan-out"). The
  scratch is deliberately **not zeroed**:
  taking it is address arithmetic over pages the pool already faulted, so the
  bytes are whatever the last expert read or the last prefill left there, and
  the driver must not assume otherwise.
- **Alignment and pre-faulting come for free.** Both are O_DIRECT
  requirements the pool already satisfies, and the second one is not
  cosmetic: btrfs runs direct reads with page faults disabled and silently
  degrades a read to the buffered path when the destination pages are not
  faulted in, with no error and a full byte count (see "Expert streaming and
  cache"). A freshly allocated prefill buffer would have had to repeat that
  work, and a bug in repeating it would have shown up as page-cache growth
  inside the cgroup rather than as a failure.

What it costs:

- **Taking the arena invalidates every layer's slot occupancy.** Sweep bytes
  land in those buffers, so no cache entry may survive claiming to hold an
  expert. The ghost-LFU frequency counters do survive, deliberately: they
  are indexed by expert id, and surviving eviction is the policy (see
  "Expert streaming and cache").
- **Known failure mode: an unreapable window read strands the arena.** If
  the ring itself fails in a way that leaves a submitted read unreapable,
  its bytes may land anywhere in the arena, so every slot the arena overlaps
  is *retired*: the buffer is leaked so a late kernel write is harmless, and
  the layer's cache is rebuilt that much smaller. The stream then refuses to
  hand out an arena for the rest of the process. It terminates and it never
  aliases, which is the point, but the arena is carved from the **head** of
  the slab, so the slots it retires are the low layers' slots. The damage at
  the shipped dials is not a boundary case, and it is exact arithmetic: the
  46.7 MiB ring is 48,955,392 B, which exceeds layer 0's entire slot row of
  11 x 3,059,712 = 33,656,832 B, and the 15,298,560 B left over is exactly 5
  slots of layer 1. So one unreapable window read leaves **layer 0 with 0
  slots and layer 1 with 6**, both under a `top_k` of 8, and both dead for
  the life of the process. Carving the arena from the *tail* instead does not
  fix this, it moves the damage to layer 47. Note the failure is a
  consequence of the borrow, not of the sweep: a separately allocated ring
  would have leaked its own pages instead of the cache's.
- **The mitigation shipped** (`4eb5f2a`), and its shape is a typed refusal on
  both edges. `SweepError::ArenaOverRetired` rejects up front any carve that
  would cover a buffer some earlier lost read may still be writing into,
  naming the layer and its slab offset. `SweepError::CacheStranded` reports a
  stranding that leaves a layer below `top_k` **at the cause**, carrying the
  first short layer, the slots it has left, `top_k`, and the read failure
  that started it as its source. What it replaces is a
  `CacheError::TooFewSlots` three decode steps later, which names a symptom
  and no cause at all. Neither error is a repair; nothing can repair this
  while the arena is the pool, which is the design.

## Attention: the kernel and its fan-out (phase 7)

EXP-017 measured attention as the binding term in prefill and phase 7 rebuilt
it in three steps. Every step is **bit-neutral**, pinned to the bit against the
kernel it replaced, and that is not a tolerance — it is the condition under
which the work was allowed at all. What follows is what the code does now.
EXP-020 records what each step was worth, warm; **no cold measurement of any of
it exists yet** (EXP-021).

**The forbidden thing first, because it has not changed.** No online,
streaming or flash-style rescaled softmax, anywhere, at any point. The causal
limit is a *length*: masked positions are **absent** from the score buffer,
from the softmax normalizer and from the V sum, rather than present with a
zero weight. A rescaled softmax reassociates the normalizer and breaks the
bit-identity gate that everything else here rests on. This is recorded in the
kernel's own module docs, in `x86.rs`'s, and here.

### Loop order: kv head outer, query head inner

GQA pairs 32 query heads with 4 kv heads at the v0 pin, so the group is 8. The
old nest put the query head outermost, which converted every K and every V
element from f16 to f32 **eight times**, once per query head in its group. The
nest is now kv-head outer with the group inner, so each element is widened once
into a `head_dim`-long f32 buffer and reused across the whole group.

Bit-neutral by construction rather than by tolerance: `h = kv_head * group + g`
visits the query heads in the same ascending order as `kv_head = h / group`;
the f16-to-f32 widening is exact, so a buffered value is bit-for-bit the value
the inline conversion produced; the dot still walks `i` ascending into one f32
with a separate multiply and add; the scale is still applied once at the end;
softmax still receives a contiguous run. The pre-change nest is kept verbatim
as a test-only reference and the two are asserted identical.

**"Once per kv head" is scoped, and the scope matters.** On the scalar path it
is literally once, for K and for V, at every group size. On the AVX2 path it is
literally once for V, and for K it is `ceil(group / 8)` — the K widening sits
*inside* `x86::qk_scores`'s chunk loop over the group. Counted: 1.00x for
groups 1-8, 2.00x for 9-16, 3.00x for 17-24, 4.00x for 25-32. **v0 is group 8,
where the ratio is exactly 1**, so nothing measured moves; the correctness
sweep carries groups 9, 12, 17 and 32, which is why an unqualified claim would
have sat next to its own disproof. Hoisting it needs the position axis outer
and the group axis inner, which makes the transposed query block hold the whole
group — a size bounded by nothing in the geometry — or re-transpose per
position block. Not worth it to remove a factor of 1.

### AVX2 + F16C, and why the obvious axis is illegal

`crates/core/src/kernels/attention/x86.rs` is gated on a runtime probe of
`avx2` **and** `f16c`. Those are separate CPUID bits, so this is not the same
probe `quants::avx2` uses, and geometries past `MAX_SIMD_HEAD_DIM = 256` fall
through to the scalar reference. Three pieces, each vectorized along an axis
that is **already independent**, so none of them reassociates a reduction:

- **The f16 widening.** `vcvtph2ps` eight at a time. Every f16 is exactly
  representable in f32, so the conversion has nothing to round; swept against
  the scalar path over all 65,536 bit patterns. Signalling NaN is the one
  scoped exception — the instruction quiets a payload the scalar path
  preserves — and a non-finite activation means the pass already failed
  upstream.
- **The QK dot.** The lanes ride the **GQA group**, not `head_dim`. The scalar
  order is one f32 accumulator per (query head, position) walking `i`
  ascending. **Eight lanes over `head_dim` would split that 128-long chain
  into eight partials plus a horizontal tree, which moves bits, so it is
  illegal here** — and it is the axis a reader reaches for first, which is why
  it is written down. The group's eight accumulators are already independent:
  transpose the group's queries once per kv head into `[i][lane]`, broadcast
  each converted K element across them, and lane `g` still walks `i` ascending
  in its own f32. Positions are blocked by `T_BLOCK = 8` purely to run eight
  *independent* chains and fill the vector add's latency; each (position, lane)
  pair still keeps a single chain over `i`.

  The position sweep is **stepped**: `T_BLOCK` while a whole block fits, then
  **at most one** `T_TAIL_BLOCK = 4`, then 1-wide. The middle rung is what
  makes widening safe rather than a trade, and the argument is exact rather
  than empirical. After the eight-loop the remainder is below 8, so a single
  `if` reaches the four-rung, and the scalar rung then runs
  `p mod 8 mod 4 = p mod 4` positions, **exactly the count `T_BLOCK = 4` ran,
  at every `p`**, while every position outside that tail sits in a block of
  four or eight rather than four. No geometry can be worse than the shipped
  code. A `const` assert pins both halves the argument needs
  (`T_BLOCK % T_TAIL_BLOCK == 0`, which collapses the scalar count, and
  `T_BLOCK <= 2 * T_TAIL_BLOCK`, which makes the single `if` enough);
  `T_TAIL_BLOCK < T_BLOCK` alone is far too weak, since 3 would pass it and
  leave a scalar tail of up to 4. Measured warm at 1.00x to 1.07x across the
  64-to-4096 ladder on the 48-layer arm, KEEP, **cold measurement owed**
  (EXP-022).

  Eight is the ceiling of the technique, read out of the linked release build
  rather than assumed: the accumulators live in `ymm8`-`ymm15` with the
  transposed query in `ymm0`, no spill and no stack store in the loop body,
  and the same probe at sixteen chains spills (17 stack moves). The cost is
  stack. `qk_scores`'s two buffers go 12 KiB to **16 KiB**, taking the frame to
  **17,144 B** of `sub` plus 48 B of pushes and crossing four guard pages, so
  the prologue emits four inline stack probes where it emitted three. That
  figure comes from the linked binary because the release profile is
  `lto = "thin"` with `codegen-units = 1`; a per-CU asm probe reports 16,936 B
  for the same function before LTO.
- **The V reduction.** Lanes over `head_dim`, which is elementwise and
  therefore exact; `t` stays strictly sequential, with the accumulator loaded
  from and stored back to `out` on every `t`, so element `i` sees exactly the
  scalar sequence.

**No FMA anywhere in that file, deliberately.** `acc += qv * kv` is a multiply
*and* an add — two roundings — and `_mm256_fmadd_ps` or `f32::mul_add`
collapses them into one and changes the result bits. The host has FMA and the
module does not enable it. This was verified load-bearing rather than assumed:
injecting an FMA at each of the two sites in an isolated copy broke four tests
each time, which also proves the vector path is genuinely taken rather than
silently falling back to scalar. **A future contributor will want to "fix"
both of these. Both are wrong.**

All loads and stores are unaligned forms. A head slice sits at a
`kv_head * head_dim` element offset inside a row, so an odd `head_dim` puts it
on a 2-byte boundary and the matching `out` sub-slice on a 4-byte one; the
claim is tested against deliberately shifted views and end to end at
`head_dim` 9.

**Miri cannot see any of this.** `is_x86_feature_detected!` is false under
Miri, so the tool takes the scalar path and never executes a single `unsafe`
block in `x86.rs`. Its soundness rests on inspection and on the bit-identity
sweep proving the *results* match — not on tooling. Treat that file
accordingly.

### Prefill fans over rows, by cost, out of the arena

`prefill::scatter_attention` submits one row per unit to the compute pool.
Rows are independent — each is a separate `attention_at_in` call reading an
immutable `q` and an immutable cache and writing only its own `[q_dim]` span of
`out` — so *any* partition reproduces the serial bytes exactly. That freedom is
spent on a **cost-balanced** split rather than an equal-row one: row `r`
attends `start + r + 1` positions, so the cumulative cost of rows `0..r` is
`r * start + r * (r + 1) / 2`, and the shard boundaries are a binary search on
that curve for `total * index / shards`. The split is a pure function of
`(rows, start, shards, index)` — no clock, no pool state — so runs stay
reproducible.

The numbers, pinned by a test: for the first 512-row chunk of a prompt over six
shards the total is **131,328** cost units, an even split would be **21,888**
each, the equal-row split's last shard carries **39,950** (a 3.3x makespan
where 6x was available), and the cost-balanced split's longest shard carries
**22,175** — 1.3% off ideal, hence an **estimated** 5.9x on the region.

**The per-shard score buffers come from the `PrefillSession` arena, so prefill
still costs zero additional bytes.** Six carves of 132,096 B took the scratch
half of the span from 80,935,940 B (77.19 MiB) to **81,728,516 B (77.94 MiB)**
of the ~1,438 MiB slab. `shards` and `max_positions` moved into `PrefillDims`
so no `scratch_bytes` call site changed, and the new term is constant in
`rows`, which is what keeps `plan_arena`'s affine assumption true.

### Decode fans over kv heads, and the query-head split was wrong

Decode splits differently, and the first attempt was measured and rejected.
`decode_attention` derives the GQA group from `q.len()`, so handing a shard a
contiguous slice of query heads silently re-maps them onto the wrong kv heads;
the split that preserves the mapping is a **strided slab** of query heads. It
works and it is bit-identical — and it fights the vectorization. `dot_block`
puts its eight lanes on the GQA group, so a one-head slab issues a full group's
vector-op count with seven lanes carrying zeros. Measured warm at 4096
positions: the whole `group = 8` call 4,293.8 µs, a `group = 1` slab 1,981.1 µs
— eight slabs are ~15.8 ms of CPU against 4.3 ms for the one call (**derived**
from those two measurements). End to end that was 1.8x at six shards for 3.1x
the CPU, taken from the cores the streamed experts need, and it **inverted** at
short context: 0.46x at 64 positions, which is where EXP-014's decode baseline
sits.

**The kv head is the axis that costs nothing.** A partition gives each kv head
to exactly one shard, so no conversion is duplicated relative to the whole
call; the group's query heads are contiguous and disjoint in both `q` and
`out`, so no gather, scatter or permutation is needed; and the group stays
whole, so the eight lanes stay full. Measured **2.4x at 4096 and 1.9x at 64**,
on a loaded machine, so a lower bound.

Three consequences worth writing down:

- **The ceiling is `n_kv_heads`, which is 4.** Six shards buy at most four.
  Going wider needs an axis that is neither the group (measured worse, above)
  nor positions (that needs the forbidden rescaled softmax).
- **The job is submitted at exactly `pool.shards()` rows**, not at
  `n_kv_heads`, and each unit slot is mapped to a kv-head range. That is a
  correctness invariant, not a tuning choice: `ComputePool::run` runs a job
  with `rows < shards()` **inline as a single shard**, so submitting 4 rows to
  a 6-shard pool would serialise the whole call — on every non-hybrid machine,
  where the pool is 12 to 32 wide against 4 kv heads, it would have serialised
  always. Surplus slots take an empty range, which the kernel documents as a
  legal no-op. The split is read from the closure's own `shard.rows` rather
  than from `pool.shards()` recomputed inside the body, so the two cannot
  disagree.
- **The fan-out is gated below 8 cached positions**, measured rather than
  assumed: from 8 upward it won every rung of every repeat by 1.66x to 2.56x;
  below 8 both arms cost 3-14 µs and the median swings either side of 1.0. The
  gate is a pure function of `kv.len(layer)`, never of a clock — a gate that
  read the wall time would make the partition, and therefore the reduction
  order, depend on how busy the machine was.

Decode has no arena, so its scratch is heap: one carve per **reachable** unit,
`min(n_kv_heads, shards)` of them, **528,384 B (516 KiB)** at the v0 pin. Sizing
it at `shards` instead left 264,192 B zero-filled at construction and never
touched. The fan-out's *new* anonymous memory is therefore **396,288 B (387
KiB)**, charged against the ~111 MiB of headroom this document already marks
**provisional** and that EXP-018's unexplained 99-105 MiB residual already eats
into.

### The aliasing lesson

The kv-range entry points originally addressed query heads absolutely, so every
shard rebuilt a `&mut` view of the **entire** output vector. The writes were
disjoint and the kernel never reads `out`, which is why it computed the right
answer and why every test passed. It was still undefined behaviour: two live
`&mut` over one allocation are UB whether or not the writes collide, because
creating the second invalidates the first.

Miri said so three ways on the pre-fix tree — Stacked Borrows against the
`Unique` held by the kernel body, Stacked Borrows against the `SharedReadOnly`
held by its validation parameter, and Tree Borrows as an outright data race
between a retag read on a worker and a non-atomic write on the submitter. `out`
is now the shard's own sub-slice, indexed from zero, while `q` stays the full
vector because the group must still be derived from it. Disjointness is
**structural**: shard windows are the image of a tiling under an injective map,
and a kernel writing outside its range would be a wrong answer rather than
unsoundness.

### Instrumentation

`decode split (forward_token):` now prints on stderr beside the prefill split,
in the same shape: a heading with token count, total and ms/token, then the six
disjoint phases (`attention`, `expert compute`, `expert io`, `projections`,
`elementwise`, `other`). It is always on when any token was decoded — no flag,
no environment variable. The heading deliberately does not begin `decode:`,
because `scripts/cold_bench.py`'s `TIMING_RE` is anchored on the prefill line
and a second match would break the harness.

It reverses a documented hot-path choice: decode's `PhaseClock` used to be left
unarmed, so a decode token paid one `Instant::now()` for the whole token. It
now pays one per phase boundary — **724 reads per token, derived by counting
the sites**, at ~27 ns each on this machine's vDSO, so ~20 µs against a token
EXP-018 puts in the hundreds of milliseconds. Order 1e-4 of the token, well
under the run-to-run noise of anything it would be read against.

**Phase 9 added a second stderr block, `decode gemv split (submitting
thread):`,** beside the first. The first block is **untouched** so that
EXP-023 stays comparable against it; the second is a finer decomposition of
two of the first block's phases and never replaces it.

It splits every pooled GEMV, as seen from the submitting thread (the pool runs
shard 0 there, so both halves are observable from one thread without any worker
touching a clock), into:

- `own` — set-up, the job descriptor, the failure-slot `Mutex`, the publish
  including its `futex` wake when workers are parked, and **this core's
  `1/shards` of the rows**;
- `wait` — the barrier: straggler shards, worker wake latency, and whatever
  the even row split costs on cores of different speeds.

Four buckets: **projections, experts, lm_head, router**. The **router is
serial on the decode thread, so its `wait` is zero by construction rather than
measured**, and the block says so on its own last line. The four buckets are
disjoint spans strictly inside the first block's `projections` and
`expert compute` phases, so their sum can never exceed those two; the gap is
the non-GEMV work those phases also cover (the softmax and top-k scan, SwiGLU,
the intermediate quantization, the expert view carves).

Two things about the counts column changed with the fused fan-out and will
mislead a reader who assumes otherwise:

- **The count is per fan-out, not per matrix.** Before fusion a job was one
  matrix and the two readings coincided. They no longer do, and the `ms` figure
  beside the count is milliseconds per *fan-out*.
- **The experts bucket's fan-out count is data-dependent.** A phase fans out
  twice — once for every gate and up together, once for every down — and a
  layer runs one phase when its plan is all hits or all misses and two when it
  splits. So the experts count is a range set by routing rather than a
  constant, and it is pinned in the tests as a range for that reason.

The accessor is `ForwardState::decode_gemv_split`, which returns four
`(name, own, wait, scatters)` tuples as plain data because `ramvamp-core` does
not print, exactly as `PrefillTiming::phases` does. It accumulates across a
run's tokens and is zeroed by the next prefill and by `ForwardState::reset`.
Cost is **4,131 clock reads per token, derived by counting the sites**, ~112 µs
against a token EXP-023 measures at 532 ms at ctx 512; zero when disarmed. No
worker reads a clock, writes shared state or touches an atomic — a worker shard
evaluates one predicted-not-taken compare — because instrumenting a barrier
must not perturb the barrier.

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
- Decode hit rate: **measured end to end as of EXP-023.** At the shipped 11
  slots/layer, five cold rungs give **53.0% to 59.3%** (59.3 / 54.0 / 53.0 /
  58.4 / 56.5 at 64 / 512 / 1,024 / 2,048 / 3,961 prompt tokens), with no
  trend in context. The dial is measured either side of 11 too: **56.1% at 12
  and 57.9% at 13** at 512 tokens, **58.4% at 12** at 3,961. The older figures
  stay on the record as the prior this corrects, and they are simulation and
  replay rather than measurement: `scripts/lfu_sim.py` gives 44.8% at 10 slots
  and 49.9% at 12 (EXP-005, 556 decode tokens of real routing traces,
  simulation only); replaying the shipped `io/cache.rs` over the identical
  traces the way the runtime actually calls it gives **50.02% at 10 and 54.48%
  at 12** (EXP-005 Correction), which is where the roughly 5-point
  simulator underestimate is recorded. **The table below is retired rather than
  re-derived**, for the reason under it: its rows are dial points from the
  replay, and mixing a measured rate into them would put two entries on one
  curve.
- Decode I/O time, derived rather than measured. Miss bytes are estimated as
  `(1 - hit) x 1,097 MB`, which assumes misses are spread across the two
  stride classes in proportion to accesses; on the one point where EXP-005
  reports both, that assumption is within 1% of the simulator's exact byte
  count. **Re-derived 2026-08-04 at EXP-019's measured bandwidth**, taken at
  the decode geometry the runtime actually issues (one expert blob, random
  order, up to 8 outstanding): **1.55-1.69 GB/s**, cold, in-cgroup, hygiene
  PASS. The table this replaces used EXP-008's 1.211-1.349 GB/s and this
  section used to say, in these words, "re-derive the whole table once the
  bandwidth probe is redone under rule 2". It has been.

  | dial | hit % | miss MB/token | ms/token | io-only tok/s |
  | --- | ---: | ---: | ---: | ---: |
  | 10 slots, batch-pinned | 50.02 | 548 | 324-354 | 2.83-3.08 |
  | 12 slots, batch-pinned | 54.48 | 499 | 295-322 | 3.11-3.39 |
  | no cache | 0 | 1,097 | 649-708 | 1.41-1.54 |

  **The "~2.8-3.4 tok/s I/O-only band" this table used to yield is RETIRED and
  must not be quoted.** The table is kept as the record of a derivation that
  measurement has overtaken; the band drawn from it is withdrawn on four counts,
  and every one of them pushes the same way:

  1. **It bracketed the 10- and 12-slot rows, and the shipped dial is 11.**
     The band never contained a row for what ships.
  2. **Its hit rates were a trace replay, not a decode.** The doc used to call
     them "simulated", which is imprecise: `scripts/lfu_sim.py` is the
     simulator and gives 44.8% / 49.9%, while 50.02% and 54.48% come from
     replaying the shipped `io/cache.rs` over EXP-005's four routing traces.
     Both are replays of 556 recorded decode tokens rather than a decode run.
     **MEASURED cold, EXP-023: 53.0-59.3% at the shipped 11 slots** over five
     live rungs.
  3. **Its bandwidth input was EXP-019's 1.55-1.69 GB/s**, a per-file
     whole-file probe. **Decode's own effective rate, DERIVED from its own
     measured counters, is 2.16 GB/s** (EXP-023: 28.3 GiB against 14.05 s of
     `io wait` at 3,961 prompt tokens).
  4. **An I/O-only ceiling computed as `1 / expert_io` OVERSTATES what decode
     can reach.** EXP-023 Note 4 establishes that the `expert io` bucket is a
     **residual**: the miss reads are already in flight during hit compute, so
     the bucket measures only the part of the read that hit compute did not
     cover. Inverting it therefore prices the drive as if it were idle during
     compute, which it is not. It is an upper bound on an upper bound.

  **What replaces it: nothing derived, and these measurements.** MEASURED cold,
  in-cgroup, hygiene PASS, medians of three scored runs at `--max-new 64`
  (EXP-023):

  | quantity | measured | what it bounds |
  | --- | --- | --- |
  | decode tok/s at 11 slots | **2.19 / 1.91 / 1.82 / 1.75 / 1.46** at ctx 64 / 512 / 1,024 / 2,048 / 3,961 | nothing — this **is** decode, end to end, and it is the number to beat |
  | decode hit rate at 11 slots | **53.0-59.3%** over the same rungs | the miss volume; no trend in context |
  | decode effective read rate (DERIVED) | **2.16 GB/s** at ctx 3,961 | an **upper** bound on the drive's average delivery rate over the window it was busy (Note 4 above) |
  | expert io share of a token | **54.1% at ctx 64 falling to 33.2% at 3,961** | a **lower** bound on drive-busy time, for the same residual reason |

  So the honest statement is: **decode measures 1.46-2.19 tok/s cold on this
  drive at the shipped dial in the session EXP-023 measured**, expert I/O is
  the largest single term below 2,048 tokens of context and is level with
  attention at 3,961, and **no I/O-only ceiling is offered**, because the only
  one this document knows how to compute overstates. The bandwidth inputs
  remain rule-2 clean but come through a threaded-`preadv` queue rather than
  io_uring (EXP-019, EXP-023 Note 10, EXP-024), so they characterise the drive
  and not the runtime's submission path. The 1,097 MB/token is still exact
  arithmetic.

  **"in the session EXP-023 measured" is load-bearing, and EXP-025 is why.**
  EXP-025 re-ran the **byte-identical** EXP-023 binary
  (sha256 `d36036b6...`) on the **same two prompt files** at the same dial a
  day later and measured **1.85 tok/s at ctx 512 against EXP-023's 1.91
  (0.969x) and 1.33 at ctx 3,961 against 1.46 (0.911x)**, MEASURED cold,
  hygiene PASS on both arms. Same bytes, same workload, 3.1% and 8.9% apart.
  **So 1.46-2.19 tok/s is a fact about one session and not a property of this
  build on this drive**, and no entry, doc line or README may quote a decode
  tok/s from one session against a decode tok/s from another. Per rule 3 that
  is a statement about the two sessions rather than about either binary; the
  most likely cause is the drive behaviour EXP-024 characterises as not
  reproducible across sessions, and EXP-025 Note 5 records that as a hypothesis
  it did not test. For the record and **not** as a second point on EXP-023's
  curve, the fused branch measured **2.16 / 1.87 / 1.74 / 1.65 / 1.43 tok/s**
  over the same five rungs in the EXP-025 session.

  The retired band's width was per-file bandwidth variance rather than
  measurement noise, which is one more reason not to resurrect it: the width
  was a property of the session that measured the spread, and EXP-024 measures
  that spread not reproducing (below).

  **Decode's own effective rate is measured (2026-08-06, EXP-023).** **DERIVED**
  from its own measured counters on one run: at 3,961 prompt tokens the streamer
  reads **28.3 GiB of experts against 14.05 s of `io wait`**, which is **2.16
  GB/s** (2.01 GiB/s). Read it as an **upper bound** rather than a point: miss
  reads are in flight during hit compute, so the drive's average delivery rate
  over the window it was actually busy is at most that (see EXP-023 Note 4 on
  why `expert io` is a residual). This figure is not merged into the retired
  table's rows and never was — that would be the curve rule 3 forbids.

  **Decode is not queue-starved.** EXP-023 derives its concurrency from the
  same counters: 10,530 misses over `63 tokens x 48 layers` is **3.48 misses
  per layer step**, and because `begin_layer` submits every miss of a step at
  once and will not open the next step while a read is outstanding, that
  average *is* decode's queue depth. The other four rungs give 3.26 to 3.76.
  EXP-023's single-blob queue-depth curve is already at plateau by QD 2 on
  three of four files and by QD 4 on the fourth, so there is no queue depth
  left to buy: more would take more concurrent misses, which needs the
  cross-layer prefetch that is closed as a no.

  **The per-file spread is NOT a lever, and EXP-023 Note 12's suggestion that
  it is has been withdrawn.** EXP-023 measured 1.568 / 3.455 / 1.654 / 3.469
  GB/s on four files at K=1, random, QD 8 — a **2.21x spread** — and pointed at
  it as the thing to attack. **EXP-024 re-ran that exact cell the next day and
  measured 1.672 / 1.607 / 1.658 / 1.601, a 1.04x spread**, with the two fast
  files roughly halving and the two slow ones unmoved; a control arm running
  the byte-identical phase-8 probe script agrees. Over all 48 expert files, the
  first time the population has been measured, the spread is **1.082x** (1.565
  to 1.694, median 1.633) and most of even that is blob size rather than any
  property of a file. Physical dispersion does not predict it (Pearson r =
  **-0.043**), and 2 MiB windows inside one file are only **1.161x** apart
  across a 7,493x median physical-span contrast. **EXP-019, EXP-023 and EXP-024
  are three sessions and must not be drawn as one curve**; the correct reading
  is that a per-file spread is a fact about the session that measured it, that
  the mechanism is drive-internal and invisible to the runtime, and that
  **there is nothing here for the runtime to pull**. All three probes are
  `threaded-pread` rather than io_uring, so none characterises the runtime's
  submission path.
- **Every tok/s figure here is drive-dependent and must never be published
  without the device.** The reference machine has a **DRAM-less QLC** part; a
  mainstream TLC Gen4 drive would materially change these numbers, and the
  arithmetic that used to be offered for how much ("~7 tok/s on a 3.5 GB/s
  drive") is **ESTIMATED** on a bandwidth constant nobody has measured on such
  a drive, so it is a direction and not a figure. Any headline tok/s figure
  ships next to the drive it was measured on. EXP-024 adds a second reason the
  device matters that is finer-grained than "which drive", and it cuts the
  opposite way from what EXP-019 and EXP-023 suggested: the same drive does not
  even give one number **across sessions**, so a published figure must name its
  session as well as its device.
- **Both halves are now measured in one run, and neither dominates.** This
  bullet used to say that no run produced both halves under rule 2 and that
  the two available figures came from different machine states. **EXP-023
  supplies the run**: five cold rungs, in-cgroup, hygiene PASS, with the phase
  split of the same tokens. MEASURED cold at ctx 512: expert io **44.4%**,
  expert compute **29.0%**, projections **17.6%**, attention **6.4%**,
  elementwise 2.6%. At ctx 3,961: expert io **33.2%**, attention **31.6%**,
  expert compute 21.1%, projections 12.1%. So expert I/O is the largest single
  term below 2,048 tokens of context, and at 3,961 it is level with attention
  and the ordering inverts run to run (EXP-023 Note 2). Everything that is
  **not** `expert io` is **45.9% of a token at ctx 64 rising to 66.8% at
  3,961** (DERIVED as `100 - expert io%` on the same rows), and what grows
  across that ladder is attention, not the GEMVs. The old phase-4 figures
  (EXP-004's ~2 s/token warm single-threaded, EXP-005's simulated 690 ms) are
  superseded and must not be quoted against these.
- **The fused decode fan-out is measured cold, and the compute win reaches the
  token only at long context.** MEASURED cold, in-cgroup, hygiene PASS, paired
  back to back against the byte-identical phase-8 binary in one session, three
  scored runs an arm, 11 slots/layer asserted on every arm (EXP-025). On the
  bucket the change targets, decode's `expert compute + projections`:
  **1.257x at ctx 512** (16.25 s, range 15.68-16.39, against 12.93 s, range
  12.49-13.96) and **1.158x at ctx 3,961** (15.39 s, 14.71-15.72, against
  13.29 s, 13.18-13.31), **disjoint ranges at both**. End to end: **1.011x at
  512 with ranges overlapping heavily, which is not a result**, and **1.070x at
  3,961 with disjoint ranges** — of which EXP-025 Note 1 attributes only
  1.025x-1.070x to the change, because the unchanged `attention` bucket moved
  1.16-1.18x the same way at both rungs. **Nothing is merged**; the branch stays
  unmerged and no dial moved.
- **The compute half is memory-bound, not dispatch-bound, and three of the
  figures previously used to reason about it are refuted.** MEASURED **warm**
  at ctx 512, so diagnostics rather than results; the cold pairing above
  corroborates the first of them and touches none of the other three:
  - **EXP-001's 9.61 GB/s is an L2-resident fixture and is not a valid
    reference for decode**, which streams every expert byte once from DRAM.
    Three code comments cited it and have been fixed.
  - **The pool's ~1.3 µs barrier figure comes from `pool.run(6, |_| {})`, a
    hot loop in which no worker parks**, and does not describe decode.
  - **Six cores buy 1.16x pre-fusion and 1.43x post-fusion, not 6x**: a forced
    single shard puts the pooled GEMV bucket at 16.35 s against 14.11 s at six
    shards in the same session, and against 11.42 s fused. Fused aggregate
    throughput is 11.00 GB/s (DERIVED).
  - **The barrier wait scales with work, not with fan-out count**: fusion cut
    expert scatters 6.14x and expert barrier wait 1.47x.

  See "How decode's GEMVs fan out, and what phase 9 refuted about them" under
  "Decode loop" for the arms, ranges and artifacts, cold and warm, and for why
  the warm 1.264x and the cold 1.257x are not the same bucket.
- **Floor for success: SETTLED 2026-08-08 as a per-drive statement.** The
  criterion was written as 3 tok/s. Every derived
  I/O-only ceiling this document has carried against it — EXP-008's 2.2-2.7,
  EXP-019's 2.8-3.4 — is **withdrawn**, for the four reasons under the retired
  table above, of which the load-bearing one is that inverting the `expert io`
  residual overstates. What exists instead is the measurement: **1.46 to 2.19
  tok/s, MEASURED cold at the shipped 11-slot dial across five context rungs
  at `--max-new 64`** (EXP-023), and **1.43 to 2.16 tok/s on the same five
  rungs on the fused branch in a different session** (EXP-025). Those are two
  sessions and rule 3 forbids one curve through them; the same binary moved
  0.969x and 0.911x between them (above), so **the floor must be judged against
  a range of sessions rather than one ladder**. 3 tok/s was not met at any
  measured rung of either, and the gap at ctx 512 was 1.57x on EXP-023's
  ladder and 1.60x on EXP-025's.

  **What ships instead.** v0 publishes decode throughput as a band attached to
  the device and the dial that produced it, never as one number:

  | | value |
  | --- | --- |
  | Drive | Micron 2400 `MTFDKBA1T0QFM-1BD1AABGB`, **DRAM-less QLC**, Gen4 x4 |
  | Decode, 11 slots/layer, ctx 64 to 3,961 | **1.46 to 2.19 tok/s** (EXP-023) and **1.43 to 2.16** (EXP-025). MEASURED cold. **Two sessions; rule 3 forbids one curve through them and they are quoted separately for that reason.** |
  | Decode, 13 slots/layer, ctx 512 | **2.06 tok/s** MEASURED cold (EXP-023) |
  | Prefill, ctx 512 | **11.25** (EXP-023), **11.07** (EXP-025) MEASURED cold |
  | `memory.peak` | **2,497 to 2,929 MiB** against the 3,072 MiB ceiling |

  The one-line summary the README carries is **"about 2 tok/s decode on a
  DRAM-less QLC drive"**. That is the honest reading of two ladders that read
  **1.91 to 2.19** and **1.87 to 2.16** over ctx 64 to 512, the lengths a chat
  turn actually uses, and it matches what the author observes running it
  himself. "About 2" is the only form in which the two sessions may be
  collapsed; **any specific figure still names its rung, its dial and its
  session.**

  **A faster drive is expected to move this and the size is UNKNOWN.** No
  mainstream TLC Gen4 part has been measured here. The arithmetic once offered
  for it ("~7 tok/s on a 3.5 GB/s drive") is **ESTIMATED on a bandwidth
  constant nobody has measured on such a drive** and stays withdrawn as a
  figure; it survives only as a direction. Measuring a second drive is the
  cheapest remaining experiment in the project and is the one thing that would
  turn this band into a curve. See `docs/roadmap.md`.

Prior-art anchors, with the qualifiers that were previously missing:

- llama.cpp RFC #23324 (same design, pread sidecar) reports 13 tok/s on a
  16 GB M1 Pro. This does **not** bound our measurements in either direction:
  it is
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
only, x86-64 with AVX2 required. Attention additionally probes **F16C**, a
separate CPUID bit from AVX2 and FMA, and falls back to the scalar reference
without it; it deliberately does not enable FMA (see "Attention: the kernel and
its fan-out"). Gemma 4 26B-A4B is model #2 and brings:
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
  **+0.09 points** (EXP-005). The build now matches the decision: the swept
  prefill is the default and goes through no cache at all (EXP-016). The
  token-major path still populates the cache, which is the behaviour this
  decision rejects, and it is retained only as the bit-identity reference.
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
- ~~Re-run the slot sweep at 11 slots/layer, batch-pinned, so the shipped dial
  has a hit rate of its own instead of a bracket (EXP-012).~~ **Done, and by
  a better instrument than this asked for: EXP-023 measures it on live cold
  decode runs rather than on a trace replay, at 53.0-59.3% across five context
  rungs, plus 56.1% at 12 slots and 57.9% at 13.** What is still not replayed
  is 16 slots, which the first bullet needs.
- Re-measure peak `anon` at 4K context with the slot pool wired in, to prove
  or disprove the 111.0 MiB of headroom the 11-slot dial leaves (EXP-012).
  **Still open, and now narrowed by EXP-023 rather than closed by it.** The
  measured 4K peak at 11 slots is 2,929.3 MiB against a predicted 2,961.03, so
  the fixed-tenant sum is about 32 MiB high, of which about 7 MiB is the
  lazily-faulted KV tail. That leaves roughly 27 MiB pointing at this row and
  it is still a `memory.peak` derivation, not a sampled `anon`. What remains
  owed is unchanged: the row itself, measured under rule 2. Older context
  below.

  EXP-018 records
  `memory.peak` at 512 tokens of context on both prefill paths, 2,576.0 MiB
  token-major and 2,570.1 MiB swept against the 3,072 MiB ceiling. That is not
  the 4K run: extrapolating the FP16 KV cache from 512 to 4096 tokens adds
  ~336 MiB and lands near 2,906 MiB, which still fits and is still
  arithmetic. It also does not explain itself cleanly, since EXP-018 sits 99
  to 105 MiB above EXP-014's 2,471.1 MiB while KV alone accounts for ~42 MiB
  of the gap. The 4K run is what closes both questions.
- Dedicated E-core io_uring reactor thread vs inline on the coordinator
  (evidence upstream is mixed both ways; see "Thread topology")
- The two sweep dials, experts per window (default 8, which is 23.3/20.3 MiB
  on the two strides) and windows in flight (default 2). The +51% at 16 MiB
  that motivated the range is **refuted** (EXP-019), so the granularity
  motivation is gone entirely and this sweep is now about finding the cost of
  the dials rather than their benefit. EXP-019 also gives it a bound worth
  respecting: the product of the two dials is bytes outstanding, the drive
  holds peak to ~100 MB and loses 15 to 18 percent past ~170 MB, so the useful
  region is `windows_in_flight` up to 4 at 8 experts, or `experts_per_window`
  up to 16 at 2 windows (EXP-008, EXP-013, EXP-015, EXP-019)
- **An io_uring queue-depth experiment inside the runtime.** **The drive-side
  half is done: EXP-023 swept queue depth at K=1, the block size decode
  actually issues.** The curve is flat from QD 2 on three of the four probed
  files and from QD 4 on the fourth, and decode's own concurrency **derives**
  to 3.26-3.76 misses per layer step, so the operating point already sits on
  the plateau and there is no depth left to buy. That sweep is still
  `threaded-pread`, so **the io_uring half of this item is untouched** and
  `RING_ENTRIES` still must not move on the strength of a probe. The original
  framing follows.

  EXP-019 measured the drive through a threaded-`preadv` queue, not io_uring,
  and swept queue depth only at the 8-blob block size, so the decode geometry
  (single blobs through `RING_ENTRIES = 8`) had no queue-depth curve of its
  own on the **drive** side. What EXP-019 does establish is that the shipped
  depth sits inside the flat part of the drive's curve at 24.5 MB outstanding,
  so this experiment is about confirming that through `SINGLE_ISSUER` /
  `DEFER_TASKRUN` rather than about an expected win. `RING_ENTRIES` must not
  be changed on the strength of EXP-019 alone. **The submission-side half of
  this item is closed**, by geometry rather than by measurement: `top_k` is 8
  and `begin_layer` refuses a step while any read is outstanding, so a decode
  layer's misses always fit the 8-entry ring and a deeper ring cannot put more
  bytes in flight. See "Expert streaming and cache", queue depth (EXP-019)
- E-cores in compute pool for non-barrier expert GEMVs
- `fadvise`/`readahead` tuning for the prefill sequential sweep
- Prefill chunk size sweep: 128 vs 256 vs 512 vs 1024. 512 is the point
  where coverage reaches ~100% of each layer's experts, so it is the
  smallest chunk that fully amortizes a sweep; the sweep is what decides
  whether the shorter chunks' lower coverage pays for their smaller
  activation working set. The dial ships as `--prefill-chunk` /
  `RAMVAMP_PREFILL_CHUNK`, so this is now a run rather than a build
  (EXP-015, EXP-016). One warm, uncgrouped run has since put 128, 256 and 512
  at 161 s, 143 s and 131 s on a 512-token prompt, all three byte-identical
  to token-major (EXP-017). That is a direction, not the sweep: it fails
  rule 2, it does not cover 1024, and it says nothing about the memory peak
- ~~**The cold rule-2 measurement of phase 7's attention work**~~: **done,
  EXP-021.** The reservation is discharged — paired cold prefill, paired cold
  decode, the 4K `memory.peak` run and the numerics gate all exist. Kept here
  only so the trail from EXP-020's warm numbers to EXP-021's cold ones is
  legible (EXP-020, EXP-021)
- ~~**The cold rule-2 measurement of phase 9's fused decode fan-out**~~:
  **done, EXP-025.** The reservation is discharged — seven cold arms, hygiene
  PASS on all of them, the branch paired back to back against the
  byte-identical phase-8 binary at ctx 512 and 3,961, the dial asserted at 11
  slots on every arm. Kept here so the trail from the warm 1.264x to the cold
  1.257x is legible. **Two follow-ups it opened and did not take**: an arm that
  isolates the `attn_q` + `attn_v` half (its bucket does not separate at either
  rung, EXP-025 Note 3), and a control that separates the 1.16-1.18x movement in
  the *unchanged* attention bucket into session drift or a second-order effect
  of fusion on pool-worker parking (EXP-025 Note 1). The second is the one that
  bounds how much of the 3,961 result the change may claim (EXP-023, EXP-025)
- **Softmax, and unfreezing `primitives`.** `primitives::softmax` is 23.5% of
  the attention kernel at 4096 positions (1,059 µs of 4,499 µs, measured,
  EXP-020) and `primitives` is frozen. Leaving it frozen caps every other
  attention lever at `1 / 0.235` = 4.26x on the kernel; perfecting it alone
  buys at most `1 / (1 - 0.235)` = 1.31x. Both figures are **derived**.
  Vectorizing it bit-neutrally means reproducing libm's `f64::exp`
  lane-for-lane and leaving the f64 normalizer serial, which is a project
  rather than a patch. Unfreezing `primitives` is a decision, not an oversight
  (EXP-020)
- **Blocking or tiling attention at long context.** After vectorization the
  kernel's cost is no longer linear in context: `max(ns/pos) / min(ns/pos)`
  over the ladder went from 1.03x to 1.40-1.57x on the 48-layer arm, because
  the arithmetic got roughly 10x cheaper and the memory traffic did not move
  at all. That is the next structural lever, and it is not more arithmetic
  (EXP-020)
- **A wider decode attention axis than the kv head.** The fan-out ceiling at
  the v0 pin is `n_kv_heads` = 4 against six pinned cores. The query-head split
  was tried and measured worse (it narrows the axis the SIMD lanes ride), and
  the position axis needs the forbidden rescaled softmax, so this is an open
  question rather than a queued task (EXP-020)
- **A longer `--max-new` on EXP-018's paired prompt**, to settle whether the
  decode drop it measured (1.88 to 1.38 tok/s) is the cold-start transient it
  looks like. EXP-018 generated 4 tokens, and the swept prefill leaves the
  expert cache empty where the token-major path left it warm, so at that
  length the transient is essentially the whole measurement. EXP-005 puts cold
  start within 2 points of steady state only by token 48, so 64 decode tokens
  on both arms of the same pair would separate the transient from steady
  state. This is the one place the sweep is currently known to cost a user
  something, so it is worth measuring rather than reasoning about (EXP-018)
- ~~The **cold phase split**~~: **done, EXP-021 and EXP-023.** Phase 7 added
  the counter half — `forward_token` emits a `decode split (forward_token):`
  block on stderr beside the prefill one, always on, at ~1e-4 of a token in
  overhead — and EXP-023 supplied the cold runs, five context rungs of both
  splits. Phase 9 added a **second** block, `decode gemv split (submitting
  thread):`, which decomposes the pooled GEMVs of the first block's
  `projections` and `expert compute` phases into `own` and barrier `wait`
  across projections / experts / lm_head / router; see "Instrumentation" for
  what changed about the counts column and why the router's `wait` is zero by
  construction. **That second block has now been read cold too** (EXP-025), but
  it **cannot be paired**: the phase-8 reference binary predates it and emits
  no such block, so every cold ratio in EXP-025 is taken on the first block's
  phase split instead. The second block's cold readings are single-arm
  characterisation — 332.8 pooled fan-outs a token at ctx 512 and 331.3 at
  3,961, and an 11.58 s pooled GEMV bucket at 512 (12.61 / 11.14 / 11.58)
  against 11.42 s warm

Dropped from the backlog:

- ~~`T_BLOCK` in `x86::dot_block`, 4 to 8~~: **done, EXP-022**, landed with a
  `T_TAIL_BLOCK = 4` rung between the wide block and the scalar tail. The
  bit-neutrality constraint that licensed it is unchanged and stays recorded:
  **more independent position chains is legal, splitting one chain over `i` is
  not**, so `T_BLOCK` is the only constant in that file that may be widened
  for parallelism. What the entry retires is the *number*: EXP-020's backlog
  carried an **estimated** "roughly 1.5-2x on the QK dot", and the whole-kernel
  measurement is 1.00x to 1.07x across the 64-to-4096 ladder, growing
  monotonically with context because attention is memory-bound at the long end
  (EXP-020 Note 3). The estimate is superseded by measurement, not merely
  unconfirmed; do not re-quote 1.5-2x. Warm, so **the cold pair is owed and
  EXP-023 is reserved for it**. See "AVX2 + F16C, and why the obvious axis is
  illegal"
- ~~Parallelize and vectorize prefill attention~~: **done, EXP-020**, and the
  measurement it produced is warm rather than cold, so the item it leaves
  behind is EXP-021 above. Three steps landed: the f16-to-f32 conversion
  hoisted out of the GQA group (2.708x arm A / 2.684x arm B, measured), an
  AVX2 + F16C path for the conversion, the dot and the V reduction (~4.2x on
  the kernel, measured in process), and fan-out across the compute pool — rows
  by cost in prefill, kv heads in decode. All bit-neutral, all pinned to the
  bit. See "Attention: the kernel and its fan-out"
- ~~Prefill throughput and peak memory for the shipped sweep, cold inside the
  3G cgroup, against the token-major path on the same prompt~~: **done,
  EXP-018.** Prefill 4.23 tok/s against 1.65, a 2.56x paired ratio, cold in
  the 3G cgroup with hygiene PASS on both arms, `memory.peak` 2,570.1 MiB.
  This item asked to be taken *after* the attention work; it was taken before
  instead, which is fine because it is a paired A/B rather than an absolute
  figure, and a ratio between two builds does not go stale when a third
  changes attention. What is still owed from it is `--repeats 5` rather than
  1, plus the decode question listed above.
- ~~Re-measure the O_DIRECT bandwidth probe under rule 2 and re-derive the
  performance model~~: **done, EXP-019.** `scripts/io_probe.py` is the
  committed harness EXP-008 never had; the drive gives 1.54-2.37 GB/s cold
  in-cgroup; EXP-008's block-size premise is refuted; and the decode I/O table
  in "Performance model" is re-derived at 1.55-1.69 GB/s. Two things it did
  **not** close: the 1.59 GB/s constant is still unsourced, since EXP-019 does
  not find it either, and the probe's queue is threaded `preadv` rather than
  io_uring, which is why the io_uring queue-depth item above exists.
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
