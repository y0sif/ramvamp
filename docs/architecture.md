# Architecture

ramvamp runs a 30B fine-grained MoE model in a 4 GiB memory budget by keeping
only the common weights and a small per-layer expert cache in RAM and reading
every other expert from NVMe on demand. This page is the design. Published
numbers and their conditions are in [benchmarks.md](benchmarks.md), every
`EXP-NNN` is a row in [experiments.md](experiments.md), prior art is in
[comparison.md](comparison.md), and the `//!` docs in `crates/core` hold the
internals.

## Goal

ramvamp's goal is to run Qwen3-30B-A3B coherently in about 4 GB of RAM on an
ordinary Linux machine with an NVMe SSD, CPU only. Success means coherent chat,
output that matches llama.cpp on identical weights (the five gates below), and
throughput measured cold inside a 4 GiB `memory.max` cgroup with
`memory.swap.max=0`. Decode is I/O-bound, so every tok/s figure is stated with
the drive it ran on rather than against one global floor.

## Model pin

v0 runs one checkpoint, `Qwen/Qwen3-30B-A3B-Instruct-2507`, as the community
GGUF Q4_K_M (not the unsloth UD variants, which use nonstandard per-layer
types): repo `bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF`, revision
`6c6e8692f43e4ca663f7ece8229a1361090d3a4c`, file
`Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`, 18,632,183,808 B (17.35 GiB).
`ramvamp-repack install` uses this pin by default, and the manifest records it.

| Fact | Value |
| --- | --- |
| Layers | 48, every layer MoE, no shared expert |
| Experts | 128 per layer, top-8: softmax over all 128, then renormalize the top 8 |
| Attention | Full attention on every layer, GQA 32:4, head_dim 128, per-head QK-RMSNorm, NeoX RoPE, rope_theta 1e7 |
| Dims | hidden 2048, attention inner width 4096, moe_intermediate 768, SwiGLU, rms_norm_eps 1e-6 |
| Vocab | 151936, untied embeddings, no BOS ever prepended |
| Stop tokens | 151645 `<\|im_end\|>` and 151643 `<\|endoftext\|>` |
| Chat format | ChatML, generation prompt `<\|im_start\|>assistant\n`, no thinking logic |
| Biases | none on any attention, router or expert projection |

## Source quantization

ramvamp runs GGUF Q4_K_M because llama.cpp runs the identical bytes, so
validation compares arithmetic rather than weights. The repacker copies the
quantized bytes unchanged and never requantizes; its GGUF parser
(`crates/repack/src/gguf`) reads only the header, bounds-checked, over HTTP
range reads.

- **Experts:** gate and up are Q4_K (144 B per 256 weights); down is Q6_K
  (210 B per 256) on 24 of 48 layers (0-5, 8, 11, 14, 17, 20, 23, 26, 29, 32,
  35, 38, 41-47) and Q4_K on the rest.
- **Common weights:** `attn_k` Q8_0, `attn_output` Q5_K, `attn_v` Q6_K on the
  Q6_K-down layers and Q4_K elsewhere, `token_embd` Q4_K, lm_head Q6_K,
  router and norms F32.
- **Activations:** Q8_K per 256-block with per-32 `bsums`, mirroring ggml's
  `vec_dot_q4_K_q8_K`, and Q8_0 for Q8_0 weights.

## The .rvmp installed model

An installed model keeps each layer's experts as fixed-stride, page-aligned
blobs, so one expert is one O_DIRECT read.

```text
model.rvmp/
  manifest.json      # pin, architecture, quant map, common_tensors, file sizes and digests
  common.bin         # embeddings, lm_head, attention, routers, norms
  tokenizer/         # tokenizer.json, chat template, generation defaults
  experts/
    layout.json      # per-layer stride, per-projection offsets and quant types
    layer_00.bin ... # 128 expert blobs per layer
```

A blob is one expert's gate, up and down slabs (884,736 B per Q4_K slab,
1,290,240 B per Q6_K slab, both whole 4 KiB pages), so the stride is
3,059,712 B (2.92 MiB) on Q6_K-down layers and 2,654,208 B (2.53 MiB) on the
rest, with no padding.

`manifest.json` is written last and promoted atomically. `common_tensors` is
the only index into `common.bin` (offsets 64-byte aligned, checked in bounds
and non-overlapping), and every manifest struct rejects unknown or missing
keys; `crates/core/src/format/manifest.rs` is the schema. At load the
manifest is schema-validated and `common.bin` and `layout.json` are hashed
against it, while each layer file is only size-checked unless
`--verify-layer-hashes` is set, because hashing 16.35 GiB through the page
cache would defeat O_DIRECT. `ramvamp-repack verify-install` is the thorough
check.

## Memory contract

The memory contract is a 4 GiB cgroup cap with swap off: at the default 19
expert slots per layer and a 4,096-token context the runtime projects
4,007.3 MiB and measured a 4,008.4 MiB peak, 88 MiB under the cap (EXP-033).

| Tenant | Qwen3 v0 | Notes |
| --- | ---: | --- |
| Common core | 1,023.34 MiB | read once into anonymous memory at load (`--common-memory resident`, the default, EXP-031); `mapped` memory-maps it |
| Expert slot pool | 2,484.84 MiB | 19 slots x 48 layers at the real strides, 130.78 MiB a slot across all layers; one allocation, every page faulted at construction |
| KV cache FP16 | 384 MiB at 4K | 96 KiB a token; allocated at full capacity, faulted lazily |
| Runtime anonymous memory | 115.1 MiB | activations, scratch, tokenizer, stacks, allocator arenas; sampled once (EXP-012), provisional, a floor rather than a ceiling |
| Total | 4,007.3 MiB | measured 4,008.4 MiB at ctx 3,961 (EXP-033); 4,016 to 4,020 MiB on a 128-token reply at that context (EXP-034) |

The expert cache is a byte budget, `--cache-bytes` (default 2,485 MiB),
divided by the summed per-layer strides to give slots per layer;
`--cache-bytes 1440M` gives 11 slots and fits a 3 GiB cap (2,962.9 MiB at
ctx 3,961, EXP-031). Resident common weights cannot be evicted by
machine-wide memory pressure from a swap-off cgroup (EXP-031). `ramvamp plan`
prints the projection without loading the model, and the runtime refuses a
configuration that does not fit before it allocates, naming every term.
Prefill adds no tenant: its 46.7 MiB ring and 77.9 MiB chunk scratch are
carved from the slot pool, which prefill leaves idle.

## Expert streaming and cache

Routed experts are read explicitly with io_uring and O_DIRECT into a fixed
per-layer slot cache, and they are never demand-paged through mmap.
TurboFieldfare measured mmap 3.54x slower per cold expert read and about 8x
slower end to end ([comparison.md](comparison.md)).

- **Reads.** Plain io_uring reads, interrupt-driven, no SQPOLL or IOPOLL
  (they burn a core the GEMVs need); ring setup steps down a flag ladder on
  `EINVAL` for older kernels. Without io_uring the streamer falls back to
  synchronous positioned reads through the same O_DIRECT handle.
- **Nothing is registered.** Registered buffers would pin the whole pool
  against an 8 MiB `RLIMIT_MEMLOCK` to save about 70 us a read (EXP-008).
- **Queue depth.** The ring has 8 entries, equal to top-k, and a layer step
  starts only after the previous step's reads complete, so a deeper ring
  cannot add bytes in flight. Eight blobs are at most 24.5 MB, inside the
  range where the drive holds its peak (EXP-019).
- **O_DIRECT is a memory requirement.** The same 1.4 GiB of reads peaked the
  cgroup at 1,092.2 MiB buffered against 5.0 MiB direct (EXP-009). A
  filesystem can silently serve O_DIRECT through the page cache, so a startup
  probe checks with `mincore` that it did not; on failure the runtime warns
  and runs buffered, outside the contract.
- **Cache policy.** Per layer, LFU over per-expert counters indexed by expert
  id that survive eviction (ghost history); without it, per-slot LFU loses
  to LRU, and a global pool across layers would add only 0.53 points (EXP-005). No online policy tried beats it by more
  than about 0.4 points (EXP-030). The live hit rate at 19 slots is 67.1% at
  ctx 512 and 68.3% at 3,961 (EXP-033), flat over longer replies (EXP-034).
- **Slot ownership.** A slot owned by a read in flight or queued compute is
  never reassigned, because two in-flight O_DIRECT reads aliased onto one
  buffer make btrfs fail checksums with spurious `EIO` (EXP-007). `EIO` and
  short reads are typed, retryable errors.
- **No cross-layer prefetch.** Copying one layer's routing predicted 7% of the
  next layer's experts upstream ([comparison.md](comparison.md)).

## Decode loop

Decode runs each layer as attention and routing, then cache-hit experts
overlapped with the miss reads, then the miss experts.

1. Attention (QK-norm, RoPE, GQA over the FP16 KV cache) and the router, on
   resident weights; attention fans out by kv head above 8 cached positions.
2. Top-8 expert ids and renormalized weights.
3. Cache plan: hits, misses, eviction victims.
4. Submit every miss to io_uring as one batch, then compute every hit as one
   unit; Qwen3 has no shared expert, so hits are the only overlap work.
5. Wait for every miss, compute the misses as one unit, then a fixed-order
   weighted sum and the residual.
6. After layer 48: final norm, lm_head, sampling (temperature 0.7, top-p 0.8,
   top-k 20 from the checkpoint; greedy for validation).

Steps 4 and 5 are deliberately coarse: per-expert execution as reads land
measured slower, with divergent output, upstream ([comparison.md](comparison.md)),
and reopening it is an open decision ([roadmap.md](roadmap.md)).
Here the reduction is fixed-order, so the result is bit-exact whatever order
experts finish in. Each expert phase issues one pool job for every gate and up
together and one for every down; merging `attn_q` with `attn_v` too did not
separate and was dropped (EXP-025).

Hit compute is the only work hiding the miss reads, so `expert io` measures
what it did not cover: a residual, not the drive's busy time. The fused
fan-out made decode's GEMV bucket 1.257x faster at ctx 512, yet `expert io`
rose about 1.85 s on identical I/O counts and the token did not separate
(EXP-025). A compute saving on this path is priced net of the I/O it exposes.

## Prefill

Prefill streams each layer's expert file front to back once per chunk of up
to 512 prompt tokens, instead of fetching experts token by token through the
cache. A 512-token chunk routes to about 100% of each layer's experts (86% at
128, per Layered Prefill, arXiv:2510.08055), so one read per expert per layer
per chunk is close to the floor.

- **Sweep.** Rows are grouped by expert, the file streams through a ring of
  8-expert windows with 2 in flight (49 MB, on the drive's plateau, EXP-019),
  and each expert is computed against its routed rows, about 32, as it lands.
- **Amortization.** Expert bytes per token fall from about 1,097 MB to about
  34 MB (EXP-015). Against the token-major path on one 512-token prompt,
  cold: 11.55x fewer bytes read and 2.56x prefill (EXP-018).
- **Cache bypass.** The borrowed ring empties the cache (the ghost counters
  survive), so decode starts cold and refills within a few tokens (EXP-034).
  Replaying the prompt into the cache was worth +0.09 points (EXP-005).
- **Bit-identical.** `--prefill token-major` keeps the per-token path as the
  reference, and both give the same bits and KV (EXP-016). Every per-row step
  runs on the pool (EXP-028); the batched Q4_K and Q6_K GEMVs dot four rows
  per pass (EXP-029, EXP-032). `--prefill-chunk` defaults to 512.

## Compute

All compute runs on a pool of one shard per physical P-core, six on the
reference machine, with the submitting thread working as one of them.

- **Placement.** `--placement shared`, the default, lets threads run on any
  P-core thread or L3 E-core; `per-core` pins one per P-core, a few percent
  faster idle and as slow as its busiest core otherwise (EXP-027); `off` sets
  no affinity ([configuration.md](configuration.md)).
- **GEMVs.** Each splits into `shards x 4` chunks claimed from a shared
  counter, so a slowed thread takes fewer. Each output row is one whole-row
  dot, so any partition is bit-identical (EXP-011, EXP-027).
- **Attention.** Prefill fans over rows, split by cost, since row `r` attends
  `start + r + 1` positions; decode fans over kv heads, at most 4 at the pin,
  because a query-head split measured worse (EXP-020).
- **Kernels.** AVX2+FMA dots for Q4_K, Q5_K, Q6_K and Q8_0, 3.2 to 5.1x a row
  over scalar (EXP-001), and AVX2+F16C attention, each probed at runtime with
  a scalar reference fallback.
- **Numerics.** Identical math must give identical bits: vector lanes only on
  already-independent axes, no split dot within a row, no FMA where the scalar
  reference rounds twice, and no rescaled softmax. Packed offsets may be only
  2-byte aligned, so kernels use unaligned loads and test shifted views.

## Validation protocol vs llama.cpp

Output is validated against llama.cpp b10217 on the same GGUF bytes through
five gates, whose reference dumps are regenerated outside the repository.

1. **Tensor.** Repacked expert slab bytes equal the GGUF slice bytes exactly.
2. **Kernel.** Dots and activation quantizers against the scalar reference
   within documented tolerance, alignment unit-tested (EXP-001, EXP-002).
3. **Logits.** Full-vocab KL(llama.cpp ‖ ramvamp) on 8 fixed prompts must
   have mean <= 3e-2, every prompt <= 6e-2, top-1 agreement on every prompt
   and all 8 scored (`scripts/kl_vs_reference.py`). Measured: mean 1.04e-2,
   worst 2.72e-2, top-1 8/8, noise floor 4.5e-3 to 1.3e-2 (EXP-004).
4. **Greedy.** First generated tokens match on short prompts, reported, not
   gated: 3/3 prompts matched all 16 tokens (EXP-003).
5. **Perplexity.** WikiText-2 raw test, 40 windows of 512 from an empty KV
   cache, positions 256 to 510 scored, by `llama-perplexity`'s protocol; the
   final and every running value within 1%. 6.3598 against 6.3810, -0.33%,
   token streams identical (EXP-035).

Three CLI subcommands serve the gates:

- `ramvamp tokenize` encodes a prompt or chat transcript, prints the ids, and
  checks the decode round-trip.
- `ramvamp logits` prints the top-N next-token logits for a raw prompt as
  JSON; gate 3 runs it with `--top` set to the vocabulary size.
- `ramvamp perplexity` scores a text file by `llama-perplexity`'s protocol and
  writes per-window scores and token ids as JSON for
  `scripts/perplexity_gate.py`.

## Performance model

Decode is I/O-bound on the reference drive: a token needs up to 1,097 MB of
experts, the cache serves about two thirds of the requests, and the rest come
from a DRAM-less QLC drive that sustains about 1.6 GB/s at decode's read
geometry (EXP-024).

- **Volume.** `8 x (24 x 3,059,712 + 24 x 2,654,208)` = 1,097,072,640 B a
  token, exact.
- **Split.** At 11 slots expert io was 44.4% of a cold decode token at ctx 512
  and 33.2% at 3,961, while attention rose from 6.4% to 31.6% (EXP-023). At
  19 slots it is still the largest term, about 204 to 212 ms a token (EXP-034).
- **No I/O-only ceiling.** `expert io` is a residual, a lower bound on drive
  time, so inverting it overstates what decode can reach.
- **Sessions.** A byte-identical binary re-read 3.1% and 8.9% slower a day
  later (EXP-025), so decode ratios are drawn only inside one session.
- **Prefill** is not I/O-bound: expert io was 2.5% of a cold 4K prefill and
  expert compute 42.1% (EXP-021). How much a faster drive moves decode is
  unmeasured.

## v0 scope caps

v0 runs one sequence at a time on Linux x86_64, CPU only, for one model,
Qwen3-30B-A3B; the repacker refuses other architectures. There is no batching
and no speculative decoding.

- Context is a dial, default 4096, refused above the trained length or when
  the footprint does not fit ([configuration.md](configuration.md)).
- `ramvamp serve` is a loopback OpenAI-compatible server, one request at a
  time, that keeps the KV cache across requests and re-prefills only the
  suffix that differs from the ids actually fed ([server.md](server.md)).
- Untrusted content reaches the model only through the sanitizing chat
  encoder; the reference encoder stays faithful to the template.
- A second model, a GPU backend and other platforms are v1 items in
  [roadmap.md](roadmap.md).
