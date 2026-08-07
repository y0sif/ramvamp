# Roadmap

**This file is the plan of record.** Handoff docs describe one phase to the
next and are written by a session that has just spent its whole context on one
narrow problem; they are not the plan. When they disagree with this file, this
file wins, and the handoff is stale.

Every phase updates this file. A phase that does not update it has not
finished.

Last updated 2026-08-08, after a cross-session audit of phases 1-9.

## Why this file exists

The phase-6 session identified that the phase plan lived only in chat, proposed
committing it, and it was never done. Every session since re-derived its own
numbering from a handoff doc. The numbering was reconciled at least twice and
never held: phase 6 ended believing phases 8-13 would be "finish attention,
decode I/O, v0 completion, server, Gemma 4, Vulkan", and none of that survived.
That is the whole reason the plan looked different every session.

## The bar

**SETTLED 2026-08-08: decode throughput is stated per drive, not as one
number.** See `docs/architecture.md` "Goal" and "Performance model".

The floor read `>= 3 tok/s` from phase 1 to phase 9 and was never met at any
measured rung. Three phases derived independently that it is not reachable on
this drive by code alone — phase 5 on bandwidth, phase 7 as a stated risk,
phase 9 on post-fusion compute headroom — and the decision was escalated three
times without being taken. Stacking every remaining lever optimistically lands
around 2.7-2.8 tok/s (DERIVED); phase 9's own research lane computed 2.83 by an
independent route before writing any code.

What v0 publishes, on the reference machine (Micron 2400, DRAM-less QLC),
Qwen3-30B-A3B Q4_K_M, cold, inside `memory.max=3G` with swap off:

| | measured |
| --- | --- |
| Decode, 11 slots, ctx 64 to 3,961 | **1.46 to 2.19 tok/s** (EXP-023) and **1.43 to 2.16** (EXP-025) — two sessions, **not one curve** |
| Decode, 13 slots, ctx 512 | **2.06 tok/s** (EXP-023) |
| Prefill, ctx 512 | **11.25** (EXP-023), **11.07** (EXP-025) |
| `memory.peak` | **2,497 to 2,929 MiB** of 3,072 |
| Fidelity | mean full-vocab KL **1.04e-2** vs llama.cpp, top-1 8/8 |

The one-line summary is **"about 2 tok/s"**. Rule 3 forbids one curve through
those two ladders: EXP-025 re-ran EXP-023's byte-identical binary and read it
3.1% slower at 512 and 8.9% slower at 3,961.

A faster drive should move this and by how much is **UNKNOWN** — no mainstream
TLC Gen4 part has been measured here. Measuring a second drive is the cheapest
remaining experiment in the project.

## Where v0 stands

| | status |
| --- | --- |
| Packed `.rvmp` format, streaming installer | **done**, phase 1 |
| Tokenizer + vendored ChatML | **done**, phase 2 |
| CPU kernels (AVX2, scalar reference) | **done**, phase 3 |
| Forward pass, KV cache, generation | **done**, phase 4 |
| io_uring + O_DIRECT streaming, ghost-LFU cache | **done**, phase 5 |
| Sequential-sweep prefill, chat REPL | **done**, phase 6 |
| Attention (GQA hoist, AVX2+F16C, fan-out) | **done**, phase 7 |
| Memory contract inside 3 GB, verified at 4K | **done**, phase 7 |
| Numerics gates 1-4 | **done** — bitident 8/8, KL 1.04e-2, greedy 24/24 |
| Decode throughput | **settled per drive**, 2026-08-08 |
| **Gate 5: perplexity** | **OPEN** — reference banked in phase 4 (llama.cpp PPL 6.3810 +/- 0.16588, wiki.test.raw `-c 512 --chunks 40`); **the ramvamp side has never been run** |
| **Shipping surface** | **OPEN** — see next section |

Everything except gate 5 and a shipping surface is finished. v0 is not blocked
on runtime work.

## Next: the shipping surface (v0 exit)

The runtime works and nothing consumes it but a REPL. TurboFieldfare ships a
CLI, an installer, a Mac app and a loopback OpenAI-compatible server; ramvamp
ships a CLI and an installer. The gap is the server.

`ramvamp-server` was recorded as post-v0 direction in the genesis session and
is the single component that unlocks every integration at once:

- **OpenAI-compatible Chat Completions on loopback, streaming SSE.** OpenCode
  and every OpenAI-speaking client work with no adapter.
- **Tool calls parsed from Qwen's native `<tool_call>` tokens.**
- **Claude Code** needs an Anthropic Messages endpoint or a translation proxy;
  the second endpoint is small once the first exists.
- **Special-token sanitization is mandatory here.** `encode_chat` is
  reference-faithful by recorded decision, so untrusted content reaching the
  server must go through `encode_chat_sanitized` (phase 6 built it).

**State the throughput reality before building against it.** At ~2 tok/s
decode and ~11 tok/s prefill, a 4K-token prompt costs ~6 minutes to ingest and
a 500-token reply costs ~4 minutes. That is workable for low-volume local
chat and completion. It is **not** workable as an agentic coding backend,
where a single turn is thousands of output tokens. The server is still the
right thing to build — it is the universal interface and it is what makes the
project consumable — but v0 should be positioned as *a local 30B endpoint for
a machine that could not otherwise run one*, not as a Claude Code replacement.

KV prefix caching (below) is what would move the agentic story, because it
attacks prefill on repeated system prompts rather than decode.

## v1 candidates, not yet committed

To be discussed and ordered before any is started. Recorded here so they stop
being re-derived.

**Throughput**
- Expert cache policy. The largest measured lever: ghost-LFU replays 49.9%
  against Belady's 72.0% at 12 slots, and nobody has tried anything between.
  Answerable **offline** against recorded traces with `scripts/lfu_sim.py` —
  candidates are LRU-K, ARC, S3-FIFO, layer-aware. One cold sweep to confirm.
  See `docs/handoff-phase9.md` item 1 for the full case.
- Overlap instrumentation. `expert io` is a residual and that is load-bearing;
  split it into submitting / waiting on a read in flight / waiting with the
  queue empty. Until it exists nobody can say how much of it is reducible.
- Progressive miss execution. **Escalated in phase 8 and never answered.**
  TurboFieldfare's DEC-17 rejection rests on divergent output, which this
  codebase structurally cannot have — the staged reduction is order-independent
  by construction. Phase 8 derived ~60 ms of a 502 ms token. Reopening a
  recorded "no" is a decision, not a task.
- The slot dial. 13 slots measured **2.06 tok/s at ctx 512** against 1.91 at
  11, ranges separating (EXP-023). Held because the 13-slot config has never
  been run at 4K. A rider on the cache-policy work, not a phase.
- Why workers are slower per row than the submitter. Six cores buy 1.43x
  post-fusion, 11.00 GB/s aggregate. Most interesting, most likely to eat a
  phase for nothing — timebox it.
- A second drive. Turns the published band into a curve and tests the project's
  central claim that the design scales with the device.

**Reach**
- KV prefix caching — prefill the system prompt once, reuse across turns. The
  prerequisite for any agentic client.
- Gemma 4 26B-A4B as model #2. **Structurally cheaper than the v0 pin**: 30
  layers against 48, ~816 MB of worst-case expert bytes per token against
  ~1,097, and it has a **shared expert** — unconditional per-layer compute to
  overlap reads with, which Qwen3 does not have and which is the mechanism
  TurboFieldfare's pipeline depends on. Brings SWA KV rings, per-layer
  attention-type mix, logit softcap, and possibly a second quant scheme.
- Larger context via Q8 KV; unlocks the Thinking-2507 variant.
- Vulkan behind the kernel trait. `README.md` promises "No GPU required" — a
  GPU backend is an additional configuration, never a substitute.

## Open decisions

Decisions only the author can take. **A phase that hits one of these should
stop and ask rather than default to measuring more** — which is what happened
for four consecutive phases.

| Decision | Raised | Status |
| --- | --- | --- |
| The tok/s floor | Phase 5, 7, 9 | **SETTLED 2026-08-08** — per drive |
| Shipping surface: server, richer CLI, or both | Genesis, reopened 2026-08-08 | **open** |
| Merge `feat/decode-compute`? | Phase 9 | **held** — 12 commits, EXP-025 KEEP, gates green |
| Keep or drop the `attn_q` + `attn_v` fusion | Phase 9, recommended dropped 3x | **open** — blocks the merge above |
| Reopen progressive miss execution | Phase 8 | **open** |
| Move the slot dial to 12 or 13 | Phase 8 ("lets hold it for now") | **held** — needs a 13-slot 4K arm |

## Phase history

| # | What | Result |
| --- | --- | --- |
| 1 | Repacker and `.rvmp` | Real 17.35 GiB install, resume, verify |
| 2 | Tokenizer and ChatML | Token-exact vs transformers on 17 fixtures |
| 3 | CPU kernels | 3.2-5.1x per row over scalar (EXP-001, EXP-002) |
| 4 | Forward pass | Greedy character-identical to llama.cpp; gate 3 re-baselined 1e-3 to 3e-2 on measured evidence (EXP-003, EXP-004) |
| 5 | io_uring streaming, LFU cache | First rule-2 measurable configuration. Decode 1.88 tok/s, peak 2,471 MiB (EXP-005..014) |
| 6 | Sweep prefill, chat REPL | Prefill **2.56x cold**, 11.55x fewer bytes (EXP-015..019) |
| 7 | Attention rebuild | Prefill **6.65x cold**, decode ~1.6x. Largest verified win (EXP-020, EXP-021) |
| 8 | Decode measurement | The decode split across five rungs; `T_BLOCK` 4->8 worth 1.014x at 4K. Closed `RING_ENTRIES`, queue depth, "faster reads" (EXP-022, EXP-023) |
| 9 | Decode compute | Fused fan-out: 1.257x on decode's GEMV bucket cold, **1.011x at 512 / 1.070x at 4K end to end**. Adaptive spin reverted. Per-file read spread refuted (EXP-024, EXP-025) |

Decode at ctx 512, cold, across the project: 1.17 (phase 5) -> **1.83-1.99
(phase 7)** -> 1.91 (phase 8) -> 1.87 (phase 9). Phase 7 is the last phase that
moved it. Rule 3 forbids drawing those on one curve; they are listed to show
the shape, not to subtract.
