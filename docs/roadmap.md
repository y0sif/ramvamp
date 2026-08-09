# Roadmap

ramvamp is a Rust runtime that runs fine-grained Mixture-of-Experts models
without loading the checkpoint into memory. The always-needed common weights
stay memory-mapped; the routed experts, which are most of the model, live on
NVMe in a page-aligned packed format and are fetched with io_uring and
O_DIRECT only when the router asks for them, through a small per-layer cache.
CPU only, Linux first.

**v0 delivers**: Qwen3-30B-A3B Q4_K_M generating coherent text inside a 3 GB
memory cgroup with a cold page cache, validated against llama.cpp on
byte-identical weights, driven from a CLI, a chat REPL, or a loopback
OpenAI-compatible HTTP server with streaming and tool calls.

This file is the plan of record for **what is done, what is next, and what is
open**. Three companions carry the rest, and each is authoritative on its own
subject: `docs/architecture.md` is ground truth for the design and is worth
reading before touching the runtime, `docs/experiments.md` is the record of
what was measured, and `docs/landscape.md` is why the design decisions were
made the way they were.

Numbered `EXP-NNN` citations point at `docs/experiments.md`, which records the
conditions each figure was taken under and the verdict it earned. A figure in
this file with no experiment behind it says so.

## The bar

Five rules govern every number the project publishes. They are the reason the
claims are worth anything.

1. A microbenchmark may start an experiment. End-to-end speed and output
   quality decide what ships.
2. **Publication.** Only cold, in-cgroup, hygiene-PASS numbers are publishable.
   Everything else is a diagnostic and is labelled one. **Correctness
   measurements are exempt.** *Cold* means every model file evicted with
   `posix_fadvise(POSIX_FADV_DONTNEED)`, the eviction proven with `mincore`
   rather than trusted from a return code, and non-zero block-layer
   `read_bytes` as a positive control that the run really was cold.
   *In-cgroup* means `systemd-run --user` at `MemoryMax=3G` and
   `MemorySwapMax=0`, with counters read from inside the cgroup before exit;
   zram counts as swap. *Hygiene PASS* means the reclaimers stole zero pages,
   every `memory.events` counter is zero, swap peak is zero, and both processes
   exited 0. This rule is also why a simulated or replayed cache hit rate is
   always named as a simulation or a replay and is never quoted as the
   runtime's hit rate.
3. **Comparison.** Every entry records its own baseline. Figures from different
   entries, sessions or machine states are never drawn on one curve. A ratio is
   quotable only when both arms ran back to back, on the same prompt at the
   same dials, in one run of the harness.
4. A change claiming identical math must produce identical bits. A change that
   reorders floating-point work must pass tolerance tests against reference
   output.
5. Negative results get entries too. They are the cheapest way to stop a bad
   idea coming back.

**The reference machine**, recorded in full in `docs/benchmark-machine.md`:
Intel Core Ultra 9 185H (6 P-cores, 8 E-cores, 2 LP E-cores; AVX2 and F16C, no
AVX-512 of any kind), 14.98 GiB RAM, and a **Micron 2400 DRAM-less QLC** NVMe
on a Gen4 x4 link, btrfs with `compress=zstd:3`. Cold O_DIRECT reads on the
installed model measure **1.54 to 2.37 GB/s** across the whole probe matrix
(EXP-019), but that top end comes from large sequential reads the runtime never
issues. At decode's own geometry, one expert blob at a time in random order,
the drive sustains about **1.6 GB/s**: 1.565 to 1.694, median 1.633, measured
across all 48 expert files (EXP-024). Quote the second figure when reasoning
about decode. There is no
passwordless sudo on this machine, so cold runs evict with
`posix_fadvise(POSIX_FADV_DONTNEED)` plus a `mincore` residency assertion
rather than `drop_caches` (`scripts/cold_bench.py`).

**Standing rule: decode throughput is stated per drive, not as one number.**
Decode is I/O-bound: expert reads are 54.1% of a token at ctx 64 falling to
33.2% at 3,961 (EXP-023), and the reference part is DRAM-less QLC. A single
global floor would contradict at the headline what the performance model
enforces in the body. It was written as `>= 3 tok/s` early on and was never
met at any measured rung. Three independent lines of evidence found it
unreachable on this drive by code alone: the measured read bandwidth, the
attention work's own stated risk, and the compute headroom left after the
decode fan-out was fused. It is stated per drive rather than lowered because
the device is the dominant term.

What v0 publishes, on the reference machine, Qwen3-30B-A3B Q4_K_M, cold,
inside `memory.max=3G` with swap off, at the shipped 11 slots per layer:

| | measured |
| --- | --- |
| Decode, ctx 64 to 3,961, `--max-new 64`, medians of three scored runs | **1.46 to 2.19 tok/s** (EXP-023) and **1.43 to 2.16 tok/s** (EXP-025). Two sessions, **not one curve** |
| Decode, 13 slots, ctx 512 | **2.06 tok/s** (EXP-023) |
| Prefill, ctx 512 | **11.25 tok/s** (EXP-023), **11.07** (EXP-025) |
| Expert cache hit rate, ctx 64 to 3,961 | **53.0% to 59.3%**, no trend in context (EXP-023) |
| `memory.peak` | **2,497 to 2,929 MiB** of 3,072 |
| Fidelity | mean full-vocab KL **1.04e-2** against llama.cpp, top-1 agreement 8/8 (EXP-004) |
| Model on disk | 17.35 GiB, about a 6x memory saving |

The one-line summary is **"about 2 tok/s"**. Rule 3 forbids drawing one curve
through those two decode ladders: EXP-025 re-ran EXP-023's byte-identical
binary on the same prompts and read it 3.1% slower at ctx 512 and 8.9% slower
at 3,961. On a DRAM-less QLC part a decode figure describes its session as
well as its device.

One caveat those decode figures carry, stated rather than buried: at
`--max-new 64` the run starts from a cache the prefill sweep emptied, so the
published curve describes **the first 63 tokens after a prompt**. Steady-state
decode at 256 generated tokens and beyond has not been measured cold. It is on
the list below.

A faster drive should move this and by how much is **unknown**. No mainstream
TLC Gen4 part has been measured here. That is the cheapest remaining
experiment in the project.

## Where v0 stands

| | status |
| --- | --- |
| Packed `.rvmp` format, streaming installer | **done**. Real 17.35 GiB install, resume, verify |
| Tokenizer and vendored ChatML | **done**. Token-exact against transformers on 17 committed fixtures |
| CPU kernels (AVX2 and F16C, scalar reference) | **done**. 3.2-5.1x per row over scalar (EXP-001, EXP-002) |
| Forward pass, KV cache, generation | **done** (EXP-003, EXP-004) |
| io_uring and O_DIRECT streaming, ghost-LFU expert cache | **done** (EXP-005 to EXP-014) |
| Sequential-sweep prefill, chat REPL | **done** (EXP-015 to EXP-019) |
| Attention rebuild, memory contract verified at 4K context | **done** (EXP-020, EXP-021) |
| Numerics gates 1 to 4 | **done**: tensor bit-identity 8/8, KL 1.04e-2, greedy 24/24 |
| Decode throughput | **settled**, published per drive |
| Shipping surface: `ramvamp-server` | **done**. Loopback OpenAI-compatible HTTP on 127.0.0.1: `/v1/chat/completions` streaming and buffered, `/v1/models`, `/health`, one request at a time, KV cache reused across requests |
| Tool calling | **done**. Streaming and buffered, parsed from Qwen's native `<tool_call>` tokens; both paths share one parser and one id minter. The renderer is byte-identical to 20 committed transformers fixtures. Verified against OpenCode driving real tool calls |
| Configurable context and profiles | **done**. `--context`, `RAMVAMP_CONTEXT`, a JSON profile file, `--no-config`, and a `plan` subcommand that resolves the dials and projects the memory footprint without loading the model. A configuration that cannot fit is refused before allocation instead of OOM-killed part way through prefill |
| Usable by an agent client | **done**. OpenCode drives the server with working tool calls on a 32K-context profile. That profile does **not** fit in 3 GB: it projects roughly 5.6 GiB, which is what profiles exist to express. The 3 GB contract is a property of the 4K default, not of the runtime |
| **Gate 5: perplexity** | **open**, and it is the only open v0 gate |

Four claims above have no numbered experiment behind them, which is worth
saying in a project with this file's rules. The tokenizer's 17 fixtures, the
tool-call renderer's 20 fixtures and the `plan` subcommand's behaviour are
covered by committed fixtures and `cargo test`: reproducible in a clone, just
not performance results. The 32K profile's footprint is a **projection**,
computed by `Footprint::project` rather than measured. That projection is
pinned tenant by tenant to the architecture document's memory table by a test,
and it inherits that table's caveats, including a provisional anonymous-memory
row that overpredicted the one measured 4K point by 31.7 MiB.

### Gate 5 is a good task and it is unclaimed

The llama.cpp side is banked and reproducible: **PPL 6.3810 +/- 0.16588**,
`llama-perplexity` b10217, `wiki.test.raw`, `-c 512 --chunks 40`, with the
corpus and the log in `models/llamacpp-ref/` (EXP-004). **The ramvamp side has
never been run.**

The work is to compute perplexity over the same corpus with the same chunking
on the same GGUF bytes and compare. It is self-contained. It needs an
installed model, but it is a correctness measurement, which rule 2 exempts, so
it does not need the benchmark cgroup or a quiet machine, and the reference it
is scored against already exists. The other gates are the pattern to follow,
including their shared exit-code convention: `scripts/bitident.py`,
`scripts/kl_vs_reference.py` and `scripts/greedy_regression.py`.

## v1

These are commitments to sequence, not to dates.

### 1. Gemma 4 26B-A4B as model #2

Structurally cheaper than the v0 pin. It has 30 layers against 48, and it has
a **shared expert**: unconditional per-layer compute that reads can be
overlapped against. Qwen3 has none, which is why the decode loop's only cover
for an outstanding read is cache-hit compute, and it is the mechanism
TurboFieldfare's pipeline depends on.

Worst-case expert bytes per token is the other half of the case, and the two
figures are not on the same footing:

- **Qwen3-30B-A3B: 1,097 MB per token, exact arithmetic on audited strides.**
  Every Q4_K projection slab is 884,736 B and every Q6_K down slab is
  1,290,240 B, so a layer's per-expert stride is 3,059,712 B on the 24 layers
  with a Q6_K down projection and 2,654,208 B on the other 24. At top-8 that
  is `8 x (24 x 3,059,712 + 24 x 2,654,208)` = 1,097,072,640 B.
- **Gemma 4 26B-A4B: ~816 MB per token is an ESTIMATE.** No derivation for it
  is recorded anywhere in this project, and Gemma 4's expert count, top-k,
  expert intermediate size and quantization are not recorded either. It
  becomes a fact the moment someone audits a real checkpoint's per-tensor type
  map and strides the way the Qwen3 pin was audited. Until then it is a
  plausible number and not a published one.

What it brings that v0 does not implement: sliding-window KV rings, a
per-layer attention-type mix, logit softcap, a `(1 + w)` RMSNorm variant,
query scaling folded differently from Qwen3, and possibly a second quant
scheme. Some of that is already anticipated in the design rather than
implemented. `crates/core/src/kv/mod.rs` documents bounded ring buffers for
Gemma 4's 25 sliding-window layers, which keep the KV cache flat as context
grows, and states that v0 implements the linear variant only. One format
constraint to design around: `ProjectionName`
(`crates/core/src/format/layout.rs`) is a closed `Gate | Up | Down` enum, so
an expert blob has no slot for a shared-expert slab today, even though
`shared_expert` is already expressible in the manifest.

### 2. A GPU backend behind the existing kernel trait, Vulkan or CUDA

**Standing rule: the README promises that no GPU is required, so a GPU backend
is an additional configuration and never a substitute for the CPU path.** The
kernel trait exists for exactly this, and the CPU path stays the reference
that correctness is measured against.

### 3. Bring your own model

This is closer than it sounds, which is worth saying precisely, because it is
the item most likely to attract a contributor. **The `.rvmp` format is already
model-agnostic**, and that is verified in the code rather than asserted:

- `ArchInfo` (`crates/core/src/format/manifest.rs`) carries `n_layers`,
  `n_experts`, `top_k`, `hidden`, `moe_intermediate`, `n_heads`, `n_kv_heads`,
  `head_dim`, `vocab`, `context_length`, `rope_theta`, `rms_eps`,
  `norm_topk_prob`, `tie_embeddings`, and critically `shared_expert: bool` and
  `sliding_window: Option<u32>`, which are Gemma 4 features Qwen3 does not
  have. `ArchInfo::validate` is bounds-only and arch-neutral.
- Nothing in the shipped format or I/O layer is Qwen-shaped. There is no
  architecture name, no tensor-name literal and no pinned layer or expert
  count outside `#[cfg(test)]` fixtures and doc comments. Layer geometry comes
  from the manifest at runtime.

**Three things are pinned, and all three were checked against the code:**

1. **The repacker refuses any architecture but `qwen3moe` by name.**
   `SUPPORTED_ARCH` (`crates/repack/src/plan.rs`) is checked in
   `RepackPlan::from_gguf`, which is the single choke point. The heavier cost
   is next to it and is easy to miss: **11 GGUF metadata keys are read as
   arch-prefixed literals** (`qwen3moe.block_count`, `qwen3moe.expert_count`
   and so on). GGUF namespaces metadata by architecture, so a second
   architecture needs that whole table rebuilt per arch. `norm_topk_prob`,
   `shared_expert` and `sliding_window` are hardcoded there rather than read.
2. **The forward pass implements one attention variant and one quant
   scheme.** Every entry point in `crates/core/src/kernels/attention.rs` is
   the same full causal GQA over the whole KV prefix: no sliding window, no
   per-layer attention-type mix, no logit softcap, and `KvCache` has no ring
   mode. The kernels dispatch over Q4_K, Q5_K, Q6_K and Q8_0, but
   `validate_expert_layer` (`crates/core/src/model/weights.rs`) freezes the
   allow-list to the audited Q4_K_M map: gate and up must be Q4_K, down must
   be Q4_K or Q6_K. `shared_expert` and `sliding_window` are the only two
   `ArchInfo` fields with no reader outside the manifest.
3. **The GGUF tensor-name mapping is written for Qwen3's naming.** The
   repacker's own dependence is thin: three expert-tensor suffixes plus
   `token_embd.weight` and `output.weight`, all GGUF-canonical names that
   other MoE checkpoints share. The sharper pin is on the loader side, where
   `crates/core/src/model/weights.rs` requires `attn_q_norm` and `attn_k_norm`
   unconditionally. QK-RMSNorm is a Qwen3 feature, and a checkpoint without
   those tensors fails to load.

**Gemma 4 is the forcing function that proves this rather than a separate
task.** Doing item 1 honestly means generalizing the metadata table, adding a
second attention variant, and giving the shared expert somewhere to live. What
is left after that is a much smaller job than it looks like today.

### 4. Architectures beyond linux x86_64

The fast kernels are AVX2 plus F16C and the streamer assumes io_uring. Both
have fallbacks, a scalar reference and a pread path respectively, so the
runtime runs elsewhere, slowly. **aarch64 NEON kernels are the obvious first
target.** The kernel trait and the two existing fallbacks are the whole
scaffolding for it.

### Then, in no fixed order

Recorded so they stop being re-derived.

**Throughput**

- **Expert cache policy. The largest measured lever, and it is answerable
  offline.** See "Good first contributions" below for the numbers and the
  method. One paired cold sweep confirms whatever the offline work picks.
- **Overlap instrumentation.** The `expert io` bucket is a residual, not the
  drive's busy time: miss reads are already in flight during hit compute, so
  the bucket measures only the part of a read that hit compute did not cover.
  That is load-bearing, and it moved 1.85 s between paired arms in EXP-025
  while request, hit, miss and byte counts were identical. Split it into
  submitting, waiting on a read genuinely in flight, and waiting with the
  queue empty. Until that exists nobody can say how much of it is reducible.
- **Progressive miss execution.** Currently a recorded "no", inherited from
  upstream: TurboFieldfare's DEC-17 measured it slower **with divergent
  output** and disabled it. The reason to reopen is that the divergence
  structurally cannot happen here, because ramvamp's staged reduction is
  fixed-order and bit-exact regardless of the order experts actually complete
  in. The reason to be cautious is on the record too: a compute saving on this
  path is partly taken straight back as newly exposed I/O, so it must be
  priced net rather than gross. Reopening a recorded "no" is a decision, not a
  task. The "~60 ms of a 502 ms token" figure that used to appear here has no
  surviving source and is withdrawn.
- **The slot dial.** 13 slots measured **2.06 tok/s at ctx 512** against 1.91
  at 11 slots, with separating ranges (EXP-023). Held because the 13-slot
  configuration has never been run at 4K, where the memory contract is
  tightest. A rider on the cache-policy work rather than a project of its own.
- **Why pool workers are slower per row than the submitting thread.** The
  finding that stands: decode's GEMV is memory-bound, not dispatch-bound, so
  six cores do not buy 6x and no spin policy reaches the barrier wait, which
  scales with work rather than with fan-out count. The figures behind it are
  **warm diagnostics under rule 2, not results**: the single-shard control is
  a single run on the pre-fusion binary, there is no single-shard control on
  the fused binary at all, and the 1.43x and the 11.00 GB/s aggregate derived
  from it inherit both limitations. A related structural ceiling is already
  documented and is not in dispute: decode attention fans out over kv heads,
  and the v0 pin has only 4 of them against six pinned cores, so two cores take
  an empty range on every decode attention call. Most interesting of the
  remaining levers, and most likely to consume a lot of work for nothing.
  Timebox it.
- **Prefill rate against prompt length.** Every prefill estimate in this file
  rests on EXP-023's cold ctx-512 figure of 11.25 tok/s. An uncontrolled
  observation put a roughly 32K-token conversation at about 23 minutes, some
  2x faster than that basis predicts. Either the basis is wrong for long
  prompts or prefix reuse was doing more of the work than assumed, and the two
  have very different consequences. Cheap to settle: the cold sweep already
  walks five rungs and this is one more column of what it already records.
  Until it is settled, no prefill-time claim about a large context belongs in
  a published number.
- **Steady-state decode.** Every published decode figure is a 63-token window
  starting from a cache the prefill sweep emptied. A longer `--max-new` on the
  same paired prompt is what turns that into a steady-state number, and it is
  one more arm on a sweep that already runs.
- **A second drive.** Turns the published band into a curve and tests the
  project's central claim, that the design scales with the device.

**Reach**

- **KV prefix caching beyond one conversation.** The single-conversation case
  already ships: the server keeps the KV cache across requests and re-prefills
  only the divergent suffix, matching a longest common prefix against the ids
  that were actually fed rather than against a re-render of the reply, because
  re-encoding assistant text does not reproduce the ids the model generated.
  What does not exist is reuse across conversations or across restarts.
- **Larger context via Q8 KV**, which also unlocks the Thinking-2507 variant.
- **An Anthropic Messages endpoint** alongside the OpenAI one. Small once the
  first exists; until then a translation proxy works.

## Positioning, stated plainly

At about 2 tok/s decode and about 11 tok/s prefill on the reference drive, a
4K-token prompt costs roughly six minutes to ingest and a 500-token reply
roughly four minutes. That is workable for low-volume local chat and
completion. It is **not** workable as an agentic coding backend, where one
turn is thousands of output tokens. v0 is *a local 30B endpoint for a machine
that could not otherwise run one*, not a replacement for a hosted coding
model. Prefix reuse within a conversation already ships and is why an agent
client is usable at all; a faster drive and reuse across conversations are
what would move it further, because both attack prefill and the device rather
than decode compute.

## Open decisions

These are the author's to take. Work that runs into one should stop and ask
rather than default to measuring more.

| Decision | Status |
| --- | --- |
| The tok/s floor | **settled**: stated per drive, not as one number |
| Shipping surface | **settled**: `ramvamp-server`. Building a terminal UI reimplements something that already exists in many good versions and is outside what this project contributes; the contribution is the streaming runtime, and a server is the universal interface to it |
| The `attn_q` + `attn_v` fusion | **settled**: dropped, its bucket did not separate at either rung. The expert-phase fusion stays |
| Reopen progressive miss execution | **open** |
| Move the slot dial to 12 or 13 | **held**, needs a 13-slot arm at 4K context |

The TUI (`chat --tui`) is kept rather than deleted, under the standing rule
that measured-and-set-aside work is preserved. It is genuinely useful as an
operator console for watching a cold run's hit rate move. It is not the
shipping surface, it is not on the v0 path, and it should not accrue features.
`chat` without `--tui` remains the reference behaviour, and `generate` and
`logits` keep their exact stdio, because the measurement scripts parse them.

## Good first contributions

Things a newcomer can do without a 3 GB cgroup or the reference drive. Read
`CONTRIBUTING.md` first for the gate a change has to pass.

**Offline expert-cache policy work.** This is the largest measured lever in
the project and it is answerable on a laptop. `scripts/lfu_sim.py` replays
recorded `--trace-experts` routing traces against the real per-layer strides,
sweeps slot counts, and already scores `lfu`, `lfu-aged`, `lfu-ghost`,
`lfu-window`, `lru` and `opt` (Belady's offline optimum). **LRU-K, ARC,
S3-FIFO and a layer-aware variant exploiting the per-layer slot arrays are all
untried.**

The headroom is real, and here is the evidence stated at matched conditions,
because it is easy to quote wrongly:

- Replaying the shipped `io/cache.rs` over four recorded traces (213,504
  accesses) the way the runtime actually calls it, with the whole step
  pinned, gives **50.02% at 10 slots and 54.48% at 12** (EXP-005 Correction).
  That is the shipped policy's number on this instrument.
- On the simulator's own scale, ghost-LFU against Belady's offline optimum is
  **44.8% against 55.8% at 10 slots** and **58.1% against 72.0% at 16 slots**.
  Belady at 12 slots was never computed. If you meet the pairing "49.9%
  against 72.0% at 12 slots" in older material, it is wrong twice over: 49.9%
  is the simulator rather than the replay, and 72.0% is the 16-slot row.
- The simulator understates the shipped cache by about 5 points because of how
  it models pinning, so the true oracle gap is smaller than a raw subtraction
  of those columns. It is still 11 points or more at matched slots.
- Live, cold, at the shipped 11 slots, the runtime measures **53.0% to 59.3%**
  across five context rungs (EXP-023).

Why it is worth doing beyond the size of the gap: a cache-policy change is
**bit-safe by construction**, because which experts are resident changes
nothing about what is computed, and it **costs no memory**, because it works
inside the existing slot budget rather than asking for more. It also attacks
the largest term. At ctx 512 expert I/O is 44.4% of a decode token, cold
(EXP-023), and it is the largest single term below 2,048 tokens of context.
Fewer misses is the direct lever on it, and unlike a compute win it cannot be
handed straight back as newly exposed read latency.

The loop is fast: add a policy, score it in seconds, and hand over a candidate
worth exactly one paired cold sweep. One honest caveat, which is itself a
contribution: **no routing trace is committed to the repository today.**
Capturing one needs an installed model and one run of
`ramvamp generate --trace-experts`, not a benchmark cgroup and not the
reference drive. Committing a small trace, so that this work needs no model
download at all, would be a genuinely useful first patch.

**Benchmark reports from other NVMe drives.** The single cheapest experiment
in the project, and the one that turns a published band into a curve. The
project's central claim is that the design scales with the device, and it has
been tested on exactly one device, a DRAM-less QLC part that is close to the
worst realistic case. A run on a mainstream TLC Gen4 drive would answer an
open question in this file. `CONTRIBUTING.md` carries the protocol and what a
report has to contain; `docs/benchmark-machine.md` is the template for
recording the machine so the result stays interpretable.

**Documentation.** The docs carry their own corrections and withdrawn numbers
on purpose, which makes them honest and makes them long. Worked examples, a
quickstart that survives a fresh clone, and clearer entry points are all
welcome. So is catching a figure whose source does not support it; several
have been found that way.

## What moved the numbers

A record of which work changed measured behaviour, not of who did what when.

| Work | Result |
| --- | --- |
| Repacker and the `.rvmp` format | A real 17.35 GiB install with resume and verify, never materializing a full tensor in heap |
| Tokenizer and ChatML | Token-exact against transformers on 17 committed fixtures (covered by `cargo test`, no experiment entry) |
| CPU kernels | 3.2-5.1x per row over the scalar reference (EXP-001, EXP-002) |
| Forward pass | Greedy output character-identical to llama.cpp; gate 3's tolerance re-baselined from 1e-3 to 3e-2 on measured evidence, against a 4.5e-3 to 1.3e-2 float-reordering noise floor of the same order as the cross-engine gap (EXP-003, EXP-004) |
| io_uring streaming and the expert cache | The first configuration measurable under rule 2: decode 1.88 tok/s, peak 2,471 MiB (EXP-014). The policy behind it, expert-indexed ghost LFU at 512 B per layer, came from EXP-005; the O_DIRECT requirement from EXP-009, where the same 1.4 GiB of reads peaked the cgroup at 1,092 MiB buffered against 5.0 MiB direct |
| Sequential-sweep prefill | Prefill **2.56x cold** and **11.55x fewer bytes read**, by reading each expert once per layer per chunk instead of once per token (EXP-018) |
| Attention rebuild | Prefill **6.65x cold**, decode about 1.6x. The largest verified win in the project (EXP-020 warm, EXP-021 cold) |
| Decode measurement | The per-token phase split across five context rungs, the 11-slot hit rate, and the slot dial. `T_BLOCK` 4 to 8 worth 1.014x at 4K. Closed `RING_ENTRIES`, queue depth, and "just read faster" as levers (EXP-022, EXP-023) |
| Fused decode fan-out | **1.257x on decode's GEMV bucket cold at ctx 512** with disjoint ranges, and 1.158x at 3,961 (EXP-025) |

That last row is the one most likely to be quoted wrongly, so it is stated in
full. End to end, cold and paired, the fusion measured **1.011x at ctx 512
with the two scored ranges overlapping heavily, which is neither a gain nor a
regression and must not be quoted as either**, and 1.070x at ctx 3,961 with
disjoint ranges. Even the 4K separation cannot be attributed cleanly, because
the unchanged attention bucket moved 1.16-1.18x the same way at both rungs;
subtracting it run by run leaves 1.025x. **Quote 1.025x to 1.070x at 4K, not
the top of it.** The gain is real in the bucket and mostly does not reach the
token, because a compute saving on this path leaves less work to hide
outstanding reads behind, and some of that read latency becomes visible
instead.

**On reading decode figures across the project.** At ctx 512, cold, the
project has recorded 1.17, then 1.83-1.99, then 1.91, then 1.87 tok/s. Do not
read that as a trend. Rule 3 already forbids one curve through four sessions,
and there is a larger problem: **the first two figures are at `--max-new 256`
and the last two at `--max-new 64`.** They also differ in statistics, single
runs against medians of three scored runs. Those are different quantities.
Decode slows as the generation window grows, because context grows with it and
attention's per-token cost grows with context. Measured on the same prompt,
going from `--max-new 4` to `--max-new 256` costs **0.63-0.65x on the build
before the attention rebuild** and 0.96x on the build after it (EXP-021).
Comparing a 256-token window against a 64-token window therefore flatters the
shorter one, by an amount that depends on which build is being measured. The
attention rebuild is the last work that clearly moved decode; the figures
after it are not evidence either way.
