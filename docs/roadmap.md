# Roadmap

ramvamp is a Rust runtime that runs fine-grained Mixture-of-Experts models
without loading the checkpoint into memory. The always-needed common weights
are read into memory once; the routed experts, which are most of the model, live on
NVMe in a page-aligned packed format and are fetched with io_uring and
O_DIRECT only when the router asks for them, through a small per-layer cache.
CPU only, Linux first.

**v0 delivers**: Qwen3-30B-A3B Q4_K_M generating coherent text inside a 3 GB
memory cgroup with a cold page cache, validated against llama.cpp on
byte-identical weights, driven from a CLI, a chat REPL, or a loopback
OpenAI-compatible HTTP server with streaming and tool calls.

**The memory contract is 4 GB since 2026-10-11** (EXP-033, the author's
decision): the default expert cache is 19 slots per layer, which peaks at
4,008 MiB at a 4K context and buys decode 1.26x at ctx 512 and 1.12x at 4K
over the 11 slots that fit 3 GB. `--cache-bytes 1440M` keeps the 3 GB
configuration. Everything recorded before EXP-033, v0's figures included, was
measured at 3G and says so.

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
   *In-cgroup* means `systemd-run --user` at the contract's `MemoryMax`
   (`4G` since EXP-033, `3G` before it) and
   `MemorySwapMax=0`, with counters read from inside the cgroup before exit;
   zram counts as swap. *Hygiene PASS* means no more than 1 MiB of reclaimed
   pages had to be read back, every `memory.events` counter is zero, swap peak
   is zero, and both processes exited 0. Reclaim the run never needed back is
   a note, not a failure: the machine's own memory pressure reclaims from every
   cgroup, and a run on a machine in ordinary use should not fail on pages it
   was done with (the refault amendment, EXP-031). A run recorded without the
   refault counters keeps the stricter zero-reclaim test. This rule is also why a simulated or replayed cache hit rate is
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
| `memory.peak` | **2,497 to 2,929 MiB** of 3,072 (2,497.0 is EXP-025's fused arm at ctx 64, 2,929.3 is EXP-023 at ctx 3,961) |
| Fidelity | mean full-vocab KL **1.04e-2** against llama.cpp, top-1 agreement 8/8 (EXP-004); WikiText-2 perplexity **6.3598** against llama.cpp's 6.3810, -0.33% (EXP-035) |
| Model on disk | 17.35 GiB, about a 6x memory saving |

**The current runtime at the 4 GB contract** (EXP-033; 19 slots, `memory.max=4G`,
every run clean, the machine in ordinary use): decode **2.78 to 2.81 tok/s** at
ctx 512 (`--max-new 128`) and **1.96** at ctx 3,961; prefill about **25 tok/s**
at ctx 512; peak **3,690 to 4,008 MiB** of 4,096.

**The current runtime at the 3 GB contract, measured 2026-10-11** (EXP-032; `main` at `cdad0e7`,
after the pool, prefill and memory work of EXP-027 to EXP-031), same
conditions, with the machine in ordinary use and every run hygiene PASS: at ctx
512, `--max-new 64`, **prefill 24.10 tok/s** (median of six, 23.93-24.21) and
**decode 2.21 tok/s** (2.18-2.24), `memory.peak` 2,638-2,657 MiB. One context
rung so far, and a separate session, so these sit beside the v0 figures above,
not on their curves, and no ratio is drawn between them: the quotable gains are
the paired ones inside EXP-027 to EXP-029.

The one-line summary is **"about 2 tok/s"**. Rule 3 forbids drawing one curve
through those two decode ladders: EXP-025 re-ran EXP-023's byte-identical
binary on the same prompts and read it 3.1% slower at ctx 512 and 8.9% slower
at 3,961. On a DRAM-less QLC part a decode figure describes its session as
well as its device.

Those decode figures are short windows, 64 or 128 generated tokens, that
start from a cache the prefill sweep left empty. **EXP-034 measured longer
windows cold and the cache is not the issue**: at the 4 GB default the live
hit rate is 67.1% over 128 tokens, 67.7% over 256 and 67.1% over 512, because
the cache is full within a few tokens. A long reply is slower anyway, by
attention's growth with context: decode **2.80 tok/s over 128 new tokens, 2.79
over 256 and 2.61 over 512** at ctx 512 (medians, paired in one session), and
**2.04 over 128 new tokens at ctx 3,961**.

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
| Usable by an agent client | **done**. OpenCode drives the server with working tool calls on a 32K-context profile. That profile does **not** fit in 3 GB: it projects roughly 5,649 MiB, about 5.5 GiB, which is what profiles exist to express. The 3 GB contract is a property of the 4K default, not of the runtime |
| Distribution: install script and landing page | **in tree, not yet live**. `install.sh` resolves the latest release, refuses anything that is not Linux x86_64, installs both binaries, and prints the 17.35 GiB model install rather than starting it. `site/index.html` is the one-page site, and `.github/workflows/sync-installer.yml` mirrors both onto `gh-pages`. **Pages still has to be switched on by hand once** (Settings > Pages > deploy from `gh-pages`, root), and until then `https://y0sif.github.io/ramvamp/install.sh` is a 404 and the README's one-liner does not work |
| Gate 5: perplexity | **done** (EXP-035): WikiText-2 PPL 6.3598 against llama.cpp's 6.3810 on the same GGUF, -0.33%, inside the author's 1% bar at every one of 40 windows, token streams identical |

Four claims above have no numbered experiment behind them, which is worth
saying in a project with this file's rules. The tokenizer's 17 fixtures, the
tool-call renderer's 20 fixtures and the `plan` subcommand's behaviour are
covered by committed fixtures and `cargo test`: reproducible in a clone, just
not performance results. The 32K profile's footprint is a **projection**,
computed by `Footprint::project` rather than measured. That projection is
pinned tenant by tenant to the architecture document's memory table by a test,
and it inherits that table's caveats, including a provisional anonymous-memory
row that overpredicted the one measured 4K point by 31.7 MiB.

### Gate 5 passed (EXP-035), and every v0 gate with it

The llama.cpp side was banked in EXP-004: `llama-perplexity` from llama.cpp
b10217 on the WikiText-2 raw test split (`wiki.test.raw`) at `-c 512 --chunks
40`, against the same Qwen3-30B-A3B Q4_K_M GGUF the runtime installs from,
**PPL 6.3810 +/- 0.16588**. The corpus and the per-chunk log are kept outside
the repository, and nothing about them is private: build llama.cpp at b10217,
fetch `wiki.test.raw` from the wikitext-2-raw-v1 archive, and re-run that
command on your own GGUF to regenerate the reference.

`ramvamp perplexity` applies the same protocol, read from b10217's source
rather than assumed: the file less one trailing newline, tokenized once, no
BOS (the GGUF sets `add_bos_token = false`), each 512-token window from an
empty KV cache, positions 256..510 scored. `scripts/perplexity_gate.py`
checks the token stream against `llama-tokenize` and then gates, with the
other gates' exit codes. The criterion is the author's, set before the run:
**the final and every running PPL within 1% of llama.cpp's**. ramvamp scores
**6.3598 +/- 0.16504, -0.33%**, its worst running gap is -0.69% at window 3,
and the token streams are identical. Every running value sits slightly below
the reference, a consistent -0.0033 nats per token that is a third of the bar
and is recorded rather than explained.

Prefill keeps only the last row's logits, so scoring needed a new path:
`prefill_prompt_scored` sends the scored rows through `lm_head` a block at a
time inside the scratch the chunk already holds. No `[rows][vocab]` buffer
exists and the footprint does not change.

## Now: phase 11, Gemma 4, Gate 5 and steady-state decode

Decided by the author on 2026-10-12, after phase 10 below. Three items:

1. **Done (EXP-035): Gate 5, perplexity.** PASS: 6.3598 against llama.cpp's
   6.3810, -0.33%, within the author's 1% bar at every window (above).
2. **Done (EXP-034): steady-state decode, cold.** The hit rate is flat at
   67.1-67.7% over 128, 256 and 512 new tokens, so the short published windows
   were not measuring a colder cache than a long reply sees. Decode over 512
   new tokens is 0.93x the 128-token figure, carried by attention as context
   grows.
3. **Gemma 4 26B-A4B as model #2** (v1 item 1), starting with obtaining and
   auditing a checkpoint, then the repacker, loader and forward-pass work that
   section lists, with a correctness gate against llama.cpp before any speed
   work. **Checkpoint chosen by the author on 2026-10-12: Google's QAT Q4_0
   GGUF**, which makes Q4_0 the one new kernel (Open decisions). Next: obtain
   it and audit it locally, which turns the survey in v1 item 1 into figures.

**A second drive stays deferred**: the author has no faster NVMe available, so
the cheapest experiment in the project waits for one (or for a contributor's
report). The working notes for this phase are in the local handoff,
`docs/handoff-phase11.md`.

## Done: phase 10, the CPU path on a machine in use (2026-10-10 to 2026-10-12)

Decided by the author on 2026-10-10, with the GPU backend set aside: the next
work makes the CPU path behave on a machine that is also being used for
other things, and then makes it faster. Every benchmark so far has needed a
quiet machine. That requirement is a finding in its own right: the runtime's
speed depends on what else is running.

The mechanism was in `crates/core/src/threads.rs`. `ComputePool` split every
GEMV into six fixed, equal row ranges, one per P-core, each run by a thread
pinned to that core, and the job finished only when the slowest range did.
Anything else running on one of those six cores, or on its hyperthread
sibling, slowed that range, and with it every GEMV in the token. EXP-026's
locked-clock run showed this happening: the GPU arms, driven from one
submitting thread, held steady through the same background load that made
the pool flip between two speeds.

Steps 1 to 3 are done (EXP-027) and merged (PR #4):

1. **Measured.** `cargo bench -p ramvamp-core --bench pool` runs the real
   kernels at the v0 expert shapes through five pool arms, under four loads
   (nothing added, a spinner on one pool core, one on its hyperthread sibling,
   two floating), interleaved so ordinary background load falls on every arm.
   It needs no quiet machine. On v0's pool one busy core cost the decode
   expert phase up to 2.1x and the prefill expert GEMV up to 2.3x.
2. **Work is handed out dynamically.** Every GEMV is split into
   `shards x 4` chunks claimed from a shared counter
   (`ComputePool::run_chunks`). Attention keeps its fixed split, because its
   scratch is carved per shard. Bit-identical by construction, and checked:
   all 12 end-to-end runs below produced the same text.
3. **The pinning is loosened.** `--placement shared`, now the default, lets
   every thread run on any P-core thread or L3 E-core and leaves the choice
   to the scheduler. `--placement per-core` keeps v0's layout.

End to end, warm, ctx 512, `--max-new 32`, paired against `main` over two
rounds on a machine doing its ordinary work (load average 2.5 to 5.3): the
default build prefills **1.14-1.22x** and decodes **1.13x** faster with
nothing added, and **1.34-1.38x** and **1.17-1.21x** faster with one core
busy. A busy core now costs prefill 4-5%, against 19-22% on `main`. These are
warm diagnostics under rule 2; the published figures at the top of this file
predate the change and stay until a cold paired run replaces them. The
trade-off, from the microbench on a quieter moment: when nothing else runs,
pinning per core is still about 4% faster on decode's GEMVs and 8% on
prefill's, which is why `per-core` remains an option.

The cold paired run followed on 2026-10-11, in the 3 GB cgroup at ctx 512
and `--max-new 64`: prefill **9.14 to 12.68 tok/s** and decode **2.01 to
2.12** (medians, three scored runs each), ranges disjoint, outputs identical.
Quote it as 1.13x and 1.045x, the worst new run against the best reference
run, because the reference arm drifted down during its own runs. It is **not
publishable**: five of the six scored runs lost 9-11 MiB to `kswapd` while
the cgroup sat at least 480 MiB under its limit with every `memory.events`
counter at zero, because the machine as a whole was short of memory. Rule 2
fails any reclaim, wherever the pressure came from, so whether that should
change is in "Open decisions".

### Running while the machine is busy, inside 3 GB (EXP-031)

Most cold runs since EXP-027 failed rule 2 because the machine was short of
memory, not because the runtime was: a 4 GB cap changes nothing, since `kswapd`
takes pages from any cgroup when the whole machine is short. What it could take
from the runtime was the 1 GiB of mmap'd common weights, and once it took 801
MiB of them mid-run and decode fell to 1.43 tok/s. `--common-memory resident`
reads them into anonymous memory instead, which a swap-off cgroup cannot lose:
zero refaults across every resident run, same speed, +0.75 s of load and +70
MiB of peak, and 2,963 MiB at 4K context against the 3,072 cap and `ramvamp
plan`'s own 2,961 projection. Two fixes help both modes: `tokenizer.json` (11
MiB) no longer lingers in the page cache, and btrfs's habit of caching direct
reads of compressed extents is handled. The author then settled three things:
resident is the default, model commands put themselves in a swap-off cgroup,
and rule 2 now fails a run on pages it had to read back rather than on any
reclaim. Under that rule every resident run so far passes and the mapped runs
that lost real working set still fail. EXP-027 to EXP-030 recorded no refault
counters, so their figures stay unpublishable until re-run.

### Next: three speed steps, in order

Each one is bit-identical to what it replaces, each is measured paired
against `main` as above, and each lands as its own PR.

1. **Done (EXP-028): prefill's single-threaded work is on the pool.** Cold
   and paired against `main` at ctx 512, prefill went from **11.80 to 17.49
   tok/s** (medians), **1.36x** worst run against best, with identical
   output; decode did not separate. Not publishable for the same hygiene
   reason as EXP-027. The router turned out to be about 6.5 s of a ~40 s
   prefill rather than the 9.8 s estimated below, and the per-expert glue,
   which the plan had not sized, 3.8 s. What the step was, as planned: three
   loops in `model/prefill.rs` ran on the submitting thread, one row at a
   time, while the pool waited:
   - **the router**: per row, a 128 x 2048 f32 matvec, softmax and top-k.
     Decode measures that matvec at 0.40 ms per layer-token, so at 512 rows
     and 48 layers it is roughly 9.8 s of the 42.5 s prefill above. That is
     an **estimate**: prefill charges the router to `projections` and does
     not time it separately. The first change gives it its own bucket.
   - **the elementwise rows**: the two RMSNorms, the Q8_K and Q8_0
     quantizations, QK-norm and RoPE, the residual adds and the top-k
     reduction. 7.0 s, 16% of prefill.
   - **the per-expert glue** inside the sweep: SwiGLU, the transpose and the
     re-quantization between `up` and `down`, inside `expert compute`.

   Every row is independent, so splitting rows across threads leaves each
   row's arithmetic unchanged. The router's f32 dot stays a serial sum per
   row, on purpose: vectorizing it would reorder the sum and could flip a
   near-tie in top-k. The route sink is still called in row order, after the
   parallel part. Decode's router, 128 experts per token, can split the same
   way across experts for about 4% of a token.
2. **Done (EXP-029), not as planned below.** Instead of the prototype's
   expanded scratch row, the batched Q4_K GEMV now dots four activation rows
   per pass and shares the weight-only work across them, bit-identical and
   with no scratch. Cold and paired against EXP-028's build, prefill went
   from **15.34 to 21.57 tok/s** (medians), **1.32x** worst against best.
   Q6_K `down` (24 of 48 layers) is not tiled yet. The plan was: **the Q4_K
   row hoist for prefill's batched GEMV.** It expands each weight
   row once and reuses it across every activation row routed to it, instead
   of once per activation. The prototype on `perf/q4k-row-hoist-prototype`
   is already bit-identical to production. What remains: vectorize its
   expansion, dispatch it from `gemv_q8_k_batched` for Q4_K when there are
   enough activation rows to pay for the scratch, and measure it cold. It
   covers the expert `gate` and `up` projections and every Q4_K attention
   projection.
3. **Done (EXP-030), a negative result.** On a simulator that now matches
   the runtime to 0.01 points, no classic policy beats the shipped
   ghost-LFU, and the best decaying variant gains about 0.4 points, too
   little to measure end to end. The gap to Belady is 6.6 points at 11
   slots, not 11 or more. The shipped policy stays; the simulator's
   `--pin-step` mode, five new policies and one committed trace in
   `scripts/traces/` ship. The plan was: **the expert cache policy for
   decode.** Capture a routing trace with
   `generate --trace-experts`, commit a small one so the work needs no model,
   score the untried policies in `scripts/lfu_sim.py` (LRU-K, ARC, S3-FIFO,
   and a layer-aware variant), and give the best one a single paired cold
   run. Expert reads are 44.4% of a cold decode token at ctx 512 (EXP-023), the
   largest single term, so this is the lever there,
   but the gain is bounded by the hit-rate gap (11 points or more to the
   offline optimum) and by the drive.

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
- **Gemma 4 26B-A4B: 803 MB per token, audited on the chosen checkpoint
  (EXP-036).** Google's QAT Q4_0 GGUF stores every expert as Q4_0, and one
  expert is 3,345,408 B on all 30 layers (2,230,272 B of fused gate and up,
  1,115,136 B of down), so the worst case is `8 x 30 x 3,345,408` =
  802,897,920 B, 0.73x Qwen3's. (The ~816 MB this line used to carry had no
  derivation and is withdrawn.) The weights outside the experts are larger,
  1.469 GiB against 0.999.

**A first survey (2026-10-12), since audited for the chosen file.** Read
from Google's `config.json`, the transformers implementation and the header
tables of seven GGUFs fetched remotely (no tensor data). The type map,
strides and metadata of the chosen file were then audited from its own
header (EXP-036); the architecture details below that a header cannot show
(norm form, scaling, gating) are still the survey's, from the reference
implementation. 30 layers, hidden 2,816, 128 experts at top-8 with an expert
intermediate of 704, vocabulary 262,144. llama.cpp names the architecture
`gemma4` (support since b8637). Google publishes an official
quantization-aware-trained **Q4_0** GGUF (14.44 GB, not gated) in which every
expert tensor is Q4_0: one expert is then 3,345,408 B, so the worst case is
**8 x 30 x 3,345,408 B = 803 MB per token**, against Qwen3's 1,097 MB. That
derivation is now audited on the chosen file (EXP-036). The same survey corrects this section in two places:

- **The norm is plain `x * w`**, not `(1 + w)` (llama.cpp's converter adds no
  +1 for Gemma 4), and attention scaling is 1.0 rather than a folded query
  scale.
- **The "shared expert" is an always-on dense MLP** (intermediate 2,112)
  running beside the routed experts, so it belongs with the common weights
  and the closed `ProjectionName` enum does not block it.

What it adds instead: gate and up stored fused in one expert tensor, GELU-tanh
gating, global layers (every sixth) with head dim 512, two KV heads, keys used
as values and partial RoPE, a 1,024-token window on the other 25, a router with
its own norm and scale, per-expert and per-layer output scalars, and a final
logit softcap of 30. And a down-projection row of 704 is not a whole number of
256-wide k-quant blocks, so no "Q4_K_M" file uses k-quants there: llama.cpp
falls back to Q5_0, Q5_1 or Q8_0. **Whichever checkpoint is chosen needs one
new kernel**: Q4_0 for Google's QAT file, Q5_0 for bartowski's Q4_K_M (17.04
GB), Q5_1 for unsloth's. The author chose Google's QAT Q4_0 (Open
decisions).

What it brings that v0 does not implement, as first written: sliding-window
KV rings, a per-layer attention-type mix, logit softcap, and a second quant
scheme. (A `(1 + w)` RMSNorm and a differently folded query scale were on
this list too; the survey above found neither.) Some of that is already
anticipated in the design rather than implemented. `crates/core/src/kv/mod.rs` documents bounded ring buffers for
Gemma 4's 25 sliding-window layers, which keep the KV cache flat as context
grows, and states that v0 implements the linear variant only. One format
constraint to design around: `ProjectionName`
(`crates/core/src/format/layout.rs`) is a closed `Gate | Up | Down` enum, so
an expert blob has no slot for a shared-expert slab today, even though
`shared_expert` is already expressible in the manifest. If the survey is
right, Gemma 4 does not need that slot: its shared expert is a dense MLP.

### 2. A GPU backend behind the existing kernel trait, Vulkan or CUDA

**Standing rule: the README promises that no GPU is required, so a GPU backend
is an additional configuration and never a substitute for the CPU path.** The
kernel trait exists for exactly this, and the CPU path stays the reference
that correctness is measured against.

**Decided 2026-10-10: not now.** A throwaway spike (EXP-026, preserved on
branch `exp/vulkan-vs-hoist`) ported the one kernel worth porting first,
prefill's Q4_K x Q8_K batched GEMV, to a Vulkan compute shader on the
reference machine's Arc iGPU. Two of its results stand:

- **The Q4_K arithmetic ports.** On a real expert slab the shader agrees with
  the CPU kernel to a max absolute error of 1.490e-6. It is Q4_K only, for a
  structural reason: the shader reads weights in 4-byte units, and Q6_K's
  210 B block is not a whole number of them. Q6_K is the expert `down`
  projection on 24 of 48 layers, so half the expert FFN would need a
  different shader.
- **The hardware has what the kernel needs.** The packed int8 dot product is
  hardware-accelerated, and memory is unified, so buffers are written in
  place with no staging copy.

Whether the iGPU is faster was never established, and no run produced a
quotable ratio. The last attempt locked both units' clocks (CPU 3.0 GHz, iGPU
1.4 GHz). Both GPU arms held steady across it. The CPU pool did not: its
per-round medians flipped between two levels about 1.7x apart while a browser
and a download ran in the background. The stopping rule agreed before that
run says an arm still unsteady at locked clocks means "not now", so the
measurement stops here rather than going another round. The instability is
the more useful finding, and it is what the next phase is about (see "Now"
above).

Reopen it when both of these hold: the CPU pool stays steady on a machine in
normal use, so that a ratio means something; and there is reason to expect
the iGPU to beat the best CPU kernel by 1.5x or more on this GEMV. That is the
bar agreed before the last run, and it is set high because a backend costs a
Q6_K shader, a streaming upload path and a second thing to keep correct.

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

- **Expert cache policy. Measured and closed for now (EXP-030).** The best
  online policy found beats the shipped ghost-LFU by about 0.4 points; the
  remaining 6.6 points to Belady at 11 slots is the price of not knowing the
  future. The slot dial is worth 2.2-2.4 points a slot, so for decode that
  dial, and the drive, are the levers left.
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
- **Steady-state decode. Measured (EXP-034).** The cache fills within a few
  tokens and its hit rate does not climb over longer windows; decode over 512
  new tokens is 0.93x the 128-token figure, by attention's growth with
  context. A 4K prompt with 128 new tokens peaks 12 MiB over `ramvamp plan`'s
  projection and 76 MiB under the cap, which is noted and not chased.
- **A second drive. Deferred** (2026-10-12): no faster drive is available to the
  author. Turns the published band into a curve and tests the
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
| Gate 5's pass criterion | **settled** (2026-10-12, before the run): the final and every running perplexity within 1% of llama.cpp b10217's. The reference's +/-0.16588 is its corpus sampling error, not an engine tolerance. Passed (EXP-035) |
| Reopen progressive miss execution | **open** |
| Gemma 4's checkpoint, and with it the new kernel (v1 item 1) | **settled** (2026-10-12): Google's official QAT Q4_0 GGUF (`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`), so the new kernel is Q4_0. Chosen over bartowski's Q4_K_M (Q5_0 kernel) and unsloth's UD-Q4_K_M (Q5_1): official, smallest per token, one kernel, trained for the format. See the survey in v1 item 1 |
| The memory contract: 3 GB with 11 slots, or 4 GB with up to 19 | **settled** (2026-10-11): 4 GB, 19 slots. Measured clean and paired (EXP-033): decode 1.26x at ctx 512 and 1.12x at 4K, prefill and output unchanged, 4,008 MiB peak at 4K against `ramvamp plan`'s 4,007. `--cache-bytes 1440M` keeps the 3 GB setup |
| A GPU backend (v1 item 2) | **not now** (2026-10-10): no quotable ratio, because the CPU pool would not hold steady even at locked clocks (EXP-026). Reopen conditions are in the v1 section |
| Hold the common weights resident by default | **settled** (2026-10-11): yes. Fits 3 GB at 4K context with 109 MiB spare, matches `ramvamp plan` to 2 MiB, and no resident run has had to read a page back (EXP-031). `--common-memory mapped` is the opt-out, and the hard rules say so |
| Run everyday sessions in their own swap-off cgroup | **settled** (2026-10-11): yes, automatically. Model commands re-launch once in a `systemd-run --user` scope with `MemorySwapMax=0`, skipped where swap is already off or with `RAMVAMP_NO_SCOPE` set |
| What rule 2's hygiene gate should count | **settled** (2026-10-11): reclaimed pages that had to be read back, over 1 MiB, plus `memory.events`. Other reclaim is a note (the refault amendment, EXP-031) |

The TUI (`chat --tui`) is kept rather than deleted, under the standing rule
that measured-and-set-aside work is preserved. It is genuinely useful as an
operator console for watching a cold run's hit rate move. It is not the
shipping surface, it is not on the v0 path, and it should not accrue features.
The one recorded exception is the taria integration (`feat/taria`, merged
2026-09-29): it publishes a semantic tree over a socket so an agent can drive
the TUI, and routes that input through the same key handler. The author tested
it by hand and chose to keep it.
`chat` without `--tui` remains the reference behaviour, and `generate` and
`logits` keep their exact stdio, because the measurement scripts parse them.

## Good first contributions

Things a newcomer can do without a benchmark cgroup or the reference drive. Read
`CONTRIBUTING.md` first for the gate a change has to pass.

**Offline expert-cache policy work.** Answerable on a laptop with no model
installed: `scripts/traces/` holds a recorded routing trace and the per-layer
strides, and `scripts/lfu_sim.py` replays it. With `--pin-step` the simulator
pins a whole layer-step the way the runtime does, and then it reproduces the
runtime's live hit rate to within 0.01 points (EXP-030), so a policy's
simulated rate is the number the runtime would measure.

What has been tried, pooled over four 256-token traces at 11 slots (EXP-030):
the shipped ghost-LFU at **55.74%**; S3-FIFO -0.9 points, LRU-2 -1.2, ARC
-2.2, LRU -3.1; decaying ghost counts and LRFU about **+0.4** at their best,
out of sample; Belady's optimum **62.30%**. Older material quoting a gap of 11
points or more to Belady compared simulator figures taken without step
pinning; on the matched instrument the gap is 5.1 points at 10 slots to 8.4
at 13.

So the easy part of this is done and the answer was "the shipped policy is
already close". What is genuinely open: **long multi-turn traces**, where the
topic drifts and a decaying policy should do better than it can over one
256-token reply; and policies that use more than recency and frequency, such
as a per-layer slot split or the router's own probabilities. Anything that
scores more than about a point above the shipped policy on the matched
simulator is worth one paired cold run. A cache-policy change is still
**bit-safe by construction** and **costs no memory**, which is why it is worth
a contributor's afternoon even with the bar where it is.

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
| Attention rebuild | Prefill **6.65x cold**, decode **1.69x in one session and 1.56x in a second**, which EXP-021 records as two figures and does not average. The largest verified win in the project (EXP-020 warm, EXP-021 cold) |
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
