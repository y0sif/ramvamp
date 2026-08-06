# Phase 8 handoff

Written 2026-08-05, at `feat/attention` = `4b39104` (phase 7 complete and
measured, not yet merged to `main`). The runtime is unchanged since `aade585`;
everything after it is harness and docs.

**Amended 2026-08-05 on `feat/decode`, on top of `d329890`**, with what phase 8
has closed and what it has found. Items struck through below landed in this
phase; "WHAT PHASE 8 HAS FOUND" and GOTCHAS 9-10 are new. The one runtime
change so far is `T_BLOCK` (EXP-022) and it is warm-measured only, so nothing
in the phase-7 summary immediately below is superseded.

**Amended again 2026-08-06 at `c78122b`, after the cold decode sweep. That
sweep is EXP-023 and it is the only thing to quote for any of it.** Read
"WHAT PHASE 8 MEASURED" below before reading anything else here: it closes
five open items, corrects one prediction in `docs/architecture.md` in sign,
discharges the `T_BLOCK` debt, and adds GOTCHA 11. Nothing in this file was
deleted to make room; the superseded text is marked where it stands.

## Where things stand

Phase 7 rebuilt attention, which EXP-017 had measured at **61.3%** of a
512-token prefill and **85.2%** of an 1891-token one. Three steps, all
bit-neutral and all pinned to the bit:

1. **Loop order** — kv head outer, query head inner, so each K and V element is
   widened from f16 once per group instead of once per query head. **2.708x**
   arm A / **2.684x** arm B, measured on a paired hardened instrument.
2. **AVX2 + F16C** for the widening, the QK dot and the V reduction. **~4.2x**
   on the kernel, measured in process by the implementing lane.
3. **Fan-out across the compute pool** — prefill over rows by cost, decode over
   kv heads. Decode measured **2.4x at 4096 positions / 1.9x at 64** on a
   loaded machine; prefill **estimated** ~5.9x on the attention region from a
   cost-balanced makespan.

Cumulatively the kernel is **6.4x to 9.1x** faster at one token's worth of
attention across the 64-to-4096 context ladder, and **~10x** on the
single-layer arm at every rung.

**None of that is publishable on its own.** Every figure above is warm, in
process and diagnostic. The cold rule-2 measurement has since been taken and
is **EXP-021**. Cold, inside `memory.max=3G` with `memory.swap.max=0`, paired
against the banked phase-5 binary on one machine in one session: prefill went
**1.68 to 11.17 tok/s, 6.65x**, over five scored runs per arm; decode at
`--max-new 256` went **1.18 to 1.99 tok/s**; a 3,961-token prompt peaks at
**2,920-2,924 MiB** of the 3,072 MiB ceiling; and all three numerics gates
pass. Quote EXP-021. The warm figures above are the mechanism, not the
result.

Prefill still costs **zero additional heap bytes** — its per-shard score
buffers come out of the `PrefillSession` arena, 77.19 to 77.94 MiB of a
1,438 MiB slab. Decode has no arena, so it added **387 KiB** of anonymous
memory.

Read before touching anything: `docs/architecture.md` (especially "Attention:
the kernel and its fan-out", then "Prefill (sequential sweep)" and "The prefill
arena"), then EXP-017, EXP-018, EXP-019 and **EXP-020** in
`docs/experiments/README.md`. The module docs on
`crates/core/src/kernels/attention.rs` and `attention/x86.rs` carry the
bit-neutrality arguments and are not optional reading before editing either
file.

## WORK TO DO

### 1. The cold measurement is done. It is EXP-021. (closed)

Taken 2026-08-05 in two unattended sessions. **Read EXP-021 before quoting any
phase-7 number.** Nothing in this list blocks phase 8 any more.

Where a future session looks for the machinery, all committed:

- `scripts/phase7_overnight.sh` — the whole thing in one go: build, warm bench,
  the three numerics gates, then the four cold steps. ~2.5-3 h, unattended,
  never aborts on the first failure. Logs and `SUMMARY.txt` land in
  `scratch/phase7/overnight-<stamp>/`.
- `scripts/phase7_rerun_cold.sh` — the cold steps only, with no model work
  before them and a settle loop that waits for `MemAvailable` >= 6,000 MiB to
  hold across four consecutive 15-second samples. ~55 min. Use this when a
  step needs repeating; the overnight script's numerics gate leaves the
  machine in a state the cold runs should not start from.
- Raw summaries: `scratch/cold-bench/p7-*.json` (session 1) and `re-*.json`
  (session 2). The runbook that specified all of it is still at
  `scratch/phase7/wave3-runbook.md`.

What it settled:

- **Prefill 6.65x cold**, 1.68 to 11.17 tok/s against the banked phase-5
  binary on a 512-token prompt, five scored runs per arm, full range 0.70% and
  1.07% about each arm's wall median. That closes EXP-018 Note 5's unbounded
  `--repeats 1` spread.
- **EXP-018's decode "regression" is explained and reversed.** Phase 5's own
  decode falls 1.81 to 1.18 tok/s as the window grows from `--max-new 4` to
  256 — its decode degrades with context, which is what EXP-020 Note 3's
  linear-in-context finding predicts. Phase 7 holds 1.83-1.99 over the same
  window and is faster at four tokens too. The 1.88-to-1.38 EXP-018 reported
  was two non-steady-state numbers compared to each other. **EXP-005's
  prompt-replay decision does not need revisiting**: there is no longer a
  deficit for it to recover.
- **4K context fits and is measured.** A 3,961-token prompt peaks at
  2,920.4 / 2,924.2 MiB of 3,072, so 148-152 MiB spare, `pgsteal` 0. The
  11-slot dial is validated at full context and EXP-014 Note 2 closes.
- **Numerics all pass** on the measured binary: `bitident.py` 8/8
  byte-identical to the phase-4 baseline, `kl_vs_reference.py --refresh` gate
  3 PASS, `greedy_regression.py` PASS against the 2026-08-03 baseline.

**One hygiene rule changed under those numbers, and it is worth judging rather
than inheriting.** Both sessions first reported most runs DIRTY. All 11 dirty
runs were **100% khugepaged** — `pgsteal_kswapd`, `_direct` and `_proactive`
all zero, `pgscan == pgsteal` exactly, on a machine 16x above the watermark
that wakes kswapd, with the flagged runs 0.39% *faster* than the clean ones.
`4b39104` excludes khugepaged from the pressure signal (still reporting it as
a SOFT note) and adds `cold_bench.py --reverdict`, which re-classifies
recorded counters without running anything. EXP-021's verdicts come from
`--reverdict`, not from a re-run. The gate was verified not to weaken:
kswapd, direct, proactive, an unknown reclaimer, a missing breakdown and
khugepaged mixed with kswapd all still fail HARD.

**`kl_vs_reference.py` requires `--refresh`.** Its per-prompt dumps are cached
at `models/llamacpp-ref/llamacpp_ref/rv_single_*.json` and **the cache is not
keyed on the binary**. Without `--refresh` it re-scores stale JSON from disk in
seconds and reports PASS having tested nothing about this build.
`greedy_regression.py` *is* keyed on binary sha256 and will recompute
correctly; `bitident.py` runs fresh. This is one script with one wrong cache
key, and it is the one that gates the logits. `phase7_overnight.sh` passes it.

### 2. Next levers, with honest ceilings (medium)

Every ceiling below is **derived or estimated**, never measured — except the
first bullet, which EXP-021 measured and which changes what the rest are worth.

- **Attention is no longer the wall in prefill, and the target should be
  re-chosen before any of the rest of this list is started.** EXP-021 Note 9,
  from the 4K run's own split: expert compute **42.1%**, projections **22.8%**,
  attention **20.4%**, elementwise **12.2%**, expert io **2.5%**. EXP-017 put
  attention at 85.2% of an 1891-token prefill on the pre-phase-7 kernel; that
  is a different entry, a different session and a different kernel, so rule 3
  forbids one curve and the only licensed reading is directional — the term
  phase 7 attacked is now third. Decode's split is different again (expert io
  42.0%, attention 29.3%), so prefill and decode no longer want the same
  lever. Measure the split on the workload you actually care about first.
  **Superseded in part by EXP-023:** that split has now been measured across
  five context rungs, and the decode half of this bullet should be read from
  "WHAT PHASE 8 MEASURED" instead. The 42.0% above is near decode's
  **minimum** expert-io share rather than its peak, which is the opposite of
  what one point invites; EXP-023 measures 54.1% at 64 tokens of context
  falling to 33.2% at 3,961. The prefill half stands and is corroborated at
  five rungs: expert compute is the largest prefill term everywhere from 512
  up.
- **Softmax is 23.5% of the kernel** at 4096 positions (1,059 µs of 4,499 µs,
  measured) and `primitives` is frozen. Leaving it frozen caps every other
  attention lever at `1 / 0.235` = **4.26x** on the kernel; perfecting softmax
  alone buys at most `1 / (1 - 0.235)` = **1.31x**. (`8bd079e`'s commit message
  attaches the 1.31x to the first claim; see EXP-020 Note 4.) Vectorizing it
  bit-neutrally is **not a patch**: `softmax` is `f64::exp` per element, so a
  lane-parallel version means reproducing libm's `exp` lane-for-lane, and the
  f64 normalizer is a serial chain that cannot be split without reassociating
  it. Only the final elementwise scale is free. **Unfreezing `primitives` is a
  decision, not an oversight.**
- **Decode's fan-out ceiling is `n_kv_heads` = 4** against six pinned cores, so
  two shards take an empty range on every call. Going wider needs an axis that
  is not the group — the query-head split was tried, measured 1.8x for 3.1x the
  CPU, and *inverted* to 0.46x at 64 positions — and not positions, which needs
  the forbidden rescaled softmax. This is an open question, not a queued task.
- ~~**`T_BLOCK = 4` in `x86::dot_block` could go to 8.**~~ **Done, EXP-022.**
  It landed at 8 with a `T_TAIL_BLOCK = 4` rung between the wide block and the
  scalar tail, and the constraint that licensed it is unchanged: more
  independent position chains is legal, splitting one chain over `i` is not.
  **The 1.5-2x estimate is superseded by measurement, not merely
  unconfirmed**: the whole kernel moves 1.00x to 1.07x across the 64-to-4096
  ladder, growing with context because attention is memory-bound at the long
  end. Do not re-quote 1.5-2x. ~~Warm, so **the cold pair is owed and EXP-023
  is reserved for it**~~ **The cold pair has been run and it is EXP-023.**
  `scripts/phase8_decode_sweep.sh` ran it against `scratch/phase7-ref/ramvamp`,
  byte-identical to the binary EXP-021 measured: **1.014x decode at 3,961
  prompt tokens and nothing distinguishable at 512**, prefill unmoved at both.
  The warm 4-7% stays correct about the kernel; what it is worth to a token is
  about 1.4% at 4K and nothing measurable at 512, which is what attention's
  31.6% and 6.4% shares of decode predict.
- **Attention is drifting memory-bound at long context.** `max(ns/pos) /
  min(ns/pos)` over the ladder went from 1.03x to 1.40-1.57x on the 48-layer
  arm — the arithmetic got roughly 10x cheaper and the memory traffic did not
  move at all. Blocking or tiling is the next *structural* lever, and it is
  not more arithmetic. That is also why the 48-layer arm's cumulative speedup
  decays 9.1x to 6.4x across the ladder while the 1-layer arm holds ~10x.

### 3. Left over from the measurement, and still carried (low)

Left over from item 1, none of it blocking:

- **The decode-256 pair and the 4K run are `--repeats 1`.** Only the prefill
  pair got five scored runs. Their only spread control is that two independent
  sessions agree, and the two phase-7 decode figures differ by 8.7% (1.99
  against 1.83 tok/s). A `--repeats 3` decode pair would cost ~35 min.
- ~~**The 512-token phase split was not captured cold.**~~ **The clobber is
  fixed; the 512-token split itself is still uncaptured.** `cold_bench.py` now
  keeps each run's stderr **verbatim in the `--json` summary**, unfiltered and
  uncapped, at a measured 607-1,111 bytes per run against summaries that are
  already 7-38 KB. A `--json` path is chosen per invocation, so that copy
  survives where the `run0N.json.stderr` sidecar does not: the sidecar path is
  still `<workdir>/runNN.json.stderr` with `NN` restarting at 0 every
  invocation, and it is still clobbered. Nothing needs copying by hand any
  more, but a 512-token cold run still has to be *taken* before its split
  exists. **Done, EXP-023.** It was taken, along with four other rungs, and
  the `--json` copy is what carried it: the splits in EXP-023 are read out of
  `runs[].stderr` in `scratch/cold-bench/p8-20260806-165322-*.json`, and the
  `runNN.json.stderr` sidecars for that sweep were clobbered exactly as
  predicted.
- **The size of the post-sweep cold-start transient is still unmeasured.**
  EXP-021 shows it no longer costs a regression at any window measured; it
  does not measure the transient itself.
- **EXP-014's discarded DIRTY runs cannot be re-checked.** Its first attempt
  scored 3 of 5 with `pgsteal` 2,817 and 2,946 pages. Those are two orders of
  magnitude larger than anything khugepaged did here and had a recorded
  external cause, so they were probably genuine pressure — but the JSONs were
  overwritten and `scratch/` is gitignored, so the classification is now
  uncheckable. EXP-014's *published* run survives at
  `scratch/cold-bench/summary.json` with `pgsteal` 0 on every run, so
  `--reverdict` leaves it PASS either way and its numbers are unaffected.
  **2,817 has since turned up again**, on the phase-8 3,961 rung's discarded
  warmup, on a different workload two days later. See GOTCHA 11; it is
  recorded as unexplained, not as a diagnosis.

Found while testing phase 7, and deliberately deferred:

- **Multi-line paste into the chat REPL submits one line per turn.** The reader
  takes a line at a time, so pasting a paragraph runs each line as its own
  prompt instead of one long one. A limitation of the REPL, not of the model or
  the runtime. Deferred on purpose: the REPL is a development affordance, and
  the project's direction is to drive ramvamp from another harness rather than
  to build one here. If it ever matters the fix is bracketed-paste mode or an
  explicit multi-line terminator. Long prompts already work through
  `generate --prompt "$(cat file)"` and through `--messages-file`, which is how
  every long-prompt measurement in EXP-020 and EXP-021 was taken, so nothing in
  the validation path depends on the REPL.

Carried over, untouched by phase 7:

- ~~`crates/core/src/io/testutil.rs` hard-codes one fixture geometry~~
  **Done.** A `Geometry` struct and `build_install_with(name, &Geometry)` now
  live in `testutil.rs`, and `prefill.rs`'s `mod wide` lost its copy of the
  install builder: **+35 / -314 lines**, net -279. The refactor is proven inert
  the only way that counts: sha256 of every file of both installs, before and
  after, byte for byte identical.
- **io_uring queue-depth curve for the decode geometry, drive side only now.**
  EXP-019 emulated depth with threaded `preadv` and swept it only at K=8, so
  single-blob decode reads through `RING_ENTRIES = 8` have no curve of their
  own on the **drive** side, and `RING_ENTRIES` must not move on EXP-019 alone.
  The phase-8 sweep's `scripts/io_probe.py` step
  (`scripts/phase8_decode_sweep.sh:1120`) is what addresses that half.
  **The submission side is closed, by geometry rather than by measurement**:
  `RING_ENTRIES` is 8 (`crates/core/src/io/stream.rs:172`), `top_k` is 8, and
  `ExpertStream::begin_layer` submits one read per miss and refuses to open a
  step while any earlier read is outstanding
  (`crates/core/src/io/stream.rs:1455-1530`). So a decode layer submits at
  most 8 reads, every miss it can ever have fits the ring at once, and raising
  `RING_ENTRIES` **cannot** increase decode's bytes in flight. Only more
  concurrent misses could, and that needs the forbidden cross-layer prefetch.
  **Derived from code geometry, not measured.**
- **Prefill chunk-size sweep** (128/256/512/1024). One warm data point exists
  (161/143/131 s at 512 tokens) but it is a direction, not the sweep, and it
  does not cover 1024 or the memory peak.
- **Kaggle or SSH-provider portability smoke.** The only path that exercises
  ext4 and the loop-device "O_DIRECT lies" fallback. Untouched since phase 4.
- **EXP-018's `memory.peak` residual, still open.** ~2,570-2,576 MiB at 516
  tokens of context, inside the 3,072 ceiling, but 99-105 MiB above EXP-014
  where KV arithmetic explains only about 42. The 4K run did **not** close it:
  EXP-021's 512-token peaks land in the same band (medians 2,571.8 and 2,577.9
  MiB, full scored range 2,569.4-2,583.5), so phase 7 added nothing to it and
  nothing identified it. What the 4K run did settle is that the contract
  survives full context, which is a different question. Closing this needs the
  provisional `anon` row in `docs/architecture.md` re-measured under rule 2.

## WHAT PHASE 8 MEASURED

One unattended cold sweep, 2026-08-06 16:53 to 19:34, `bash
scripts/phase8_decode_sweep.sh` at `c78122b`. Nine cold arms, all
`measurement hygiene: PASS`, `--warmup 1 --repeats 3 --max-new 64`, inside
`memory.max=3G` with `memory.swap.max=0`. **It is EXP-023. Quote EXP-023 and
nothing in this section, which is a summary of it.** Raw material:
`scratch/phase8/sweep-20260806-165322/SUMMARY.txt`,
`scratch/cold-bench/p8-20260806-165322-*.json` (each run record now carries
its `stderr`, so the splits survive the sidecar clobber) and
`scratch/io-probe/p8-20260806-165322-decode-qd.{json,md}`.

### What it closed

- **The decode phase split exists as a curve against context**, at 64, 512,
  1,024, 2,048 and 3,961 prompt tokens, which is what the "measure the split
  on the workload you actually care about" item below was asking for. Expert
  io is the largest single term at every rung on the first-scored-run reading,
  and its share **falls** from 54.1% to 33.2% as context grows while
  attention's **rises** from 1.5% to 31.6%. At 3,961 the two have crossed or
  are crossing: the ordering flips between that rung's own scored runs, so read
  the long end as "level", not as "expert io dominates". This replaces the
  single 4K point from EXP-021 Note 9 that the section below warns against
  choosing a lever from; it does not extend it, and the two must not be drawn
  as one curve.
- **The hit rate at the shipped 11 slots/layer is measured**, 53.0% to 59.3%
  across the ladder with no trend in context, against a `docs/architecture.md`
  that said it never had been and bracketed it 50.02-54.48%. The bracket is
  superseded at 11 slots. Three of the five rungs land at or above its top and
  none falls below its floor.
- **The slot dial is measured at 12 and at 13.** One extra slot is worth about
  2 points of hit rate: 54.0% at 11 slots, 56.1% at 12 and 57.9% at 13 at 512
  tokens; 56.5% at 11 and 58.4% at 12 at 3,961.
- **The memory contract's 12-slot prediction is corrected, in sign.** It
  predicted 3,091.82 MiB and 19.8 MiB over the cap; measured at 3,961 prompt
  tokens plus 64 generated it is **3,058.4 MiB, 13.6 MiB under**, hygiene
  PASS, no OOM. The prediction overshoots by 33.4 MiB at 12 slots and 31.7 at
  11, which is a constant error in the fixed-tenant sum rather than a slope
  error; about 7 MiB of it is the lazily-faulted KV tail and roughly 27 MiB
  points at the provisional 115.1 MiB anon row. **The shipped default stays 11
  slots and `--cache-bytes` is unchanged.**
- **The `T_BLOCK` debt is discharged.** Cold and paired against phase 7's own
  binary: 1.44 to 1.46 tok/s at 3,961 (1.014x) and nothing distinguishable at
  512, where the scored ranges overlap. Prefill unmoved, 0.999x and 0.996x.
  EXP-022's warm 4-7% remains correct **about the kernel**; attention is 6.4%
  of decode at 512 and 31.6% at 3,961, so 4-7% of those shares predicts
  0.26-0.45% and 1.3-2.2%, and that is what was measured. The 512 medians are
  **not** a regression and must not be quoted as one.
- **The 512-token cold phase split was taken**, closing the carried item
  below, along with four other rungs of prefill split.

### What is now open, and what the honest next lever is

Stated as findings, not as a plan. Nobody has decided any of this.

- **The slot dial is the cheapest measured lever and it is one measurement
  short of a decision.** One extra slot is worth **+4.7% decode at 512 and
  +4.1% at 3,961**, and 12 slots/layer fits at full context with 13.6 MiB to
  spare. Two things sit against acting on it. The 512 step's scored ranges
  overlap (1.91-1.98 against 1.96-2.03), so only the 3,961 step separates at
  three runs each. And **the 13-slot arm has no 4K run at all**: it is
  measured only at 512 tokens, where it reads 2.06 tok/s and 2,862.3 MiB, so
  there is no measurement that says whether it fits. A 13.6 MiB margin is also
  thinner than EXP-018's unexplained 99-105 MiB residual and thinner than the
  33.4 MiB prediction error just corrected.
- **Per-file bandwidth spread at the single-blob size is 2.21x, and it is a
  larger effect than anything else measured here.** At K=1, random, QD 8 the
  four probed layer files read 1.568, 3.455, 1.654 and 3.469 GB/s, with
  `layer_00` stuck at 1.59-1.60 across the entire queue-depth sweep while two
  files plateau at 3.4-3.5. Decode's own effective rate **derives** to at most
  2.16 GB/s (28.3 GiB against 14.05 s of `io wait` at 3,961), which sits
  between the two groups, and its concurrency **derives** to 3.26-3.76 misses
  per layer step, which is already on the plateau of the measured curve. So
  **decode is not queue-starved; it is dragged by the slow files.** `filefrag`
  reports byte-identical extent geometry for `layer_00` at 1.568 GB/s and
  `layer_20` at 3.455 (398 extents, mean 984,027 B, median 884,736 B, zero
  adjacent pairs on both), so fragmentation does not predict it, the obvious
  explanation is eliminated, and no other has been tested. That probe is
  `threaded-pread`, not io_uring, so it characterises the drive and the
  filesystem and not the runtime's submission path.

## WHAT PHASE 8 HAS FOUND

Four findings that are load-bearing for choosing a lever and were written down
nowhere else. Each carries its own label. None of them is a rule-2 number and
none may be published; they come from committed sources and from the one
surviving stderr sidecar, `scratch/cold-bench/run00.json.stderr`, which is
`scratch/` and therefore gitignored, so the counts are quoted here rather
than cited by path alone.

### The decode split's `expert io` bucket is a residual, and it understates the drive

**Derived from code structure, pinned by a measured coincidence.**
`stage_expert_phases` (`crates/core/src/model/forward.rs:1569-1593`) runs
`run_plan(.., stream.hits(), ..)` **before** `await_misses()` and charges it to
`expert compute`; only the `await_misses()` block is charged to `expert io`.
The miss reads were submitted by `begin_layer` and are in flight for the whole
of that hit compute, which is the entire point of the two-phase shape. So the
split's `expert io` is the **residual** wait after hit compute has already
covered part of the read, not drive-busy time, and **EXP-021 Note 9's 42.0%
decode figure understates how much of a token the drive is busy for**. The
coincidence that pins the reading: on the same run the streamer counted
**2.44 s** of io wait against the split's **2.45 s** `expert io` bucket, so the
bucket is that block and nothing else. Anyone sizing an I/O lever off that
42.0% is sizing it off a lower bound.

### EXP-021 Note 9's decode split is a 7-token post-prefill transient, not steady state

**Measured counts, derived interpretation.** Same sidecar, the session-2 4K
run. Its decode split header reads `7 tokens in 5.85s`, and the arithmetic
agrees: **2,688 expert requests / 384 per token** (48 layers x `top_k` 8) =
**7 tokens**. Of its 1,634 misses, **1,390 are cold** (85%), and its hit rate
is **39.2%** against EXP-013's measured 52.7% steady state. That is not a
property of decode; it is the arena. The default `--prefill sweep` takes the
slot pool as its arena and invalidates every layer's slot occupancy (see
`docs/architecture.md`, "The prefill arena"), so decode starts with nothing
resident and the first tokens pay for it. **A phase-8 lane must not choose a
lever from that split**: seven tokens, at 4K context, with a cold cache, is the
transient EXP-021 Note 11 says was never measured, and it is not the decode a
user spends their time in.

**Superseded by EXP-023 as a source of shares, and confirmed as a warning.**
EXP-023 measures the decode split at five context rungs over 63 tokens each
rather than 7, and its 4K rung reads expert io **33.2%** against attention
**31.6%**, where the transient read 42.0% and 29.3%. So the transient does
overstate expert io, as this finding predicted. What EXP-023 does **not** do
is escape the arena effect: `--max-new 64` still starts from a sweep-emptied
cache, so its curve is "the first 63 tokens after a prompt" and steady state
at 256 tokens and beyond is still unmeasured. Its cold-miss counts show the
same shape at a smaller scale, 2,650 to 3,149 cold misses per run.

### Decode steady state reads roughly 500 MB per token

**Derived from measured totals. Rule 3: it must not be put on a curve with
figures from other entries.** Two EXP-021 session-1 phase-7 runs, same binary,
same prompt, differing only in `--max-new`: at `--max-new 256` the process read
**146,441,695,232 B**, at `--max-new 4` it read **20,716,994,560 B** (the same
integer on all six of those runs). `--max-new N` costs `N - 1` `forward_token`
decode calls, so the difference is **252** decode tokens and
`125,724,700,672 / 252` = **498.9 MB per decode token**, over roughly the 520th
to the 770th token of context. The phase-5 arm gives **502.3 MB** by the same
subtraction (365,898,100,736 less 239,329,693,696, over the same 252), 0.7%
away, which is the agreement to expect since phase 7 moved no I/O path
(EXP-021 Note 7). At the installed mean
blob stride of **2,856,960 B** (24 layers at 3,059,712 and 24 at 2,654,208,
from `models/qwen3.rvmp/manifest.json`) that is about **175 misses of the 384
requests** a token makes, so a hit rate near **54.5%**, consistent with
EXP-013's measured 52.7% without being a re-measurement of it. The strength of
the derivation is that the subtraction cancels model load and prefill entirely;
its weakness is the uniform-stride assumption in the last step, which is why
the hit rate is "near 54.5%" and the byte figure is the one to lean on.

### The reference arm for phase 8 is phase 7's own binary

**Verified, not assumed.** `scratch/phase7-ref/ramvamp` is sha256
`d56dc034ebd3e94e83b59ad64503adf289baf22af112752593ec59068a586e66`, which is
what EXP-021 records for the binary it measured. It is the same executable
rather than a lookalike rebuild, so the phase-8 cold arms carry no toolchain or
profile drift against phase 7. `scripts/phase8_decode_sweep.sh` checks that
hash twice, against the `SHA256` file beside the binary and against a constant
pinned independently at line 191, because a re-banked reference would agree
with a regenerated sidecar and still be the wrong bytes.

## PROCESS

Same as phases 1-7 and it still works: research-first with findings reported
before planning; plan with acceptance criteria approved before implementation;
parallel lanes with strict file ownership; adversarial review per wave;
mechanical gate (`cargo fmt --check`, `cargo clippy --all-targets -- -D
warnings`, `cargo test`, `cargo test -p ramvamp-core --no-default-features`)
before anything merges; y0sif tests locally before the merge.

Carried from phase 6, both still earning their keep:

- **Docs lanes run last**, never in parallel with the code they document.
- **Run a specialist adversarial reviewer** alongside the generic code review.

One amendment earned in phase 7:

- **Do not freeze a kernel API in order to parallelise two lanes.** Phase 7 did
  exactly that so the vectorization lane and the fan-out lane could run at
  once, and the frozen entry point forced the decode fan-out onto the query
  head — the one axis that fights the vectorization it was supposed to compose
  with. It took a third commit to undo. If two lanes want the same kernel,
  either serialise them or freeze the *contract* (bit identity, error
  precedence, scratch sizing) rather than the signature.

## GOTCHAS

These are the ones earned this phase and worth the ink.

1. **The obvious SIMD axis for the QK dot is arithmetically illegal.** Eight
   lanes over `head_dim` split a 128-long f32 chain into eight partials and a
   horizontal tree, which moves bits. The legal axes are the **GQA group**
   (whose accumulators are already independent) and **position blocking**
   (which adds chains rather than splitting one). And **never FMA**: `acc +=
   qv * kv` is two roundings and `_mm256_fmadd_ps` makes it one. Both of these
   look like obvious optimizations and both break the bit-identity gate that
   the whole phase rests on. Verified load-bearing, not assumed: injecting an
   FMA at either site broke four tests.
2. **A perf bench must observe its whole output.** `sink += out[0]` left 31 of
   32 heads eliminable as dead stores under thin LTO. It did **not** fire —
   proven by paired re-measurement, ±2% with mixed signs — so the baseline is
   not retracted. The danger is entirely in the *next* run, because optimizing
   the kernel is exactly what makes it small enough to inline, and a fabricated
   speedup there is indistinguishable from a real one. `black_box` the whole
   output slice.
3. **Drift controls belong in any bench on this machine.** Re-measure the cheap
   rungs after the ladder at identical context and sample counts. They caught a
   12% transient that would have been read as signal, and they settled a
   thermal-versus-residency question by measurement rather than argument. Any
   arm whose control ratio leaves 0.99x-1.01x should be **discarded, not
   interpreted** — `scratch/phase7/attn-final.txt` is the example of a run that
   fails this and must not be read as data.
4. **`kl_vs_reference.py`'s cache is not keyed on the binary.** Always pass
   `--refresh`. See item 1 above.
5. **`ComputePool::run` runs inline when `rows < shards()`.** A fan-out
   submitted with fewer units than shards silently serialises — no error, no
   warning, correct answer. Submit at exactly `pool.shards()` rows and map the
   indices to your work axis yourself, and read the split from the closure's
   own `shard.rows` rather than recomputing `pool.shards()` inside the body, so
   the two cannot disagree.
6. **Freezing a kernel API to parallelise two lanes can freeze out the right
   design.** See PROCESS above. It cost a commit and a measured regression at
   short context.
7. **Bit-identity tests only cover the geometries they sweep.** The AVX2
   multi-chunk path (`group > 8`) was reachable from **no test at all** —
   confirmed by asserting inside it and watching all 586 tests still pass. Five
   geometries now execute it. When you add a path, add the geometry that
   reaches it, and prove the geometry reaches it by breaking the path on
   purpose.
8. **Miri never sees `crates/core/src/kernels/attention/x86.rs`.**
   `is_x86_feature_detected!` is false under Miri, so it takes the scalar path
   and executes none of that file's `unsafe`. The soundness there is by
   inspection and by the bit-identity sweep, and it is **unverified by
   tooling**. Miri is still worth running on anything that shards a buffer: on
   the pre-`65b0b3a` tree it rejected the decode fan-out three separate ways
   while every test passed.

Two more earned in phase 8, both of them the same failure in different
clothes: a check that reports success while checking nothing.

9. **A bare test name with `-- --exact` matches nothing and still exits 0.**
   `cargo test -p ramvamp-core <bare_name> -- --exact` prints
   `running 0 tests` and then `test result: ok. 0 passed; 0 failed; ...; 471
   filtered out`, and returns **exit status 0**. `--exact` compares against a
   test's **full module path**, so a bare name matches nothing and libtest
   calls an empty run a pass. It fooled two independent lanes in this phase,
   each of whom read that `ok` as "my test passes". The full path is required:
   `cargo test -p ramvamp-core
   kernels::attention::tests::dispatch_lengths_cover_every_position_block_remainder
   -- --exact` runs 1 and passes 1. Read the `N passed` count, never the word
   `ok`. Verified both ways on this tree.
10. **`scripts/phase7_overnight.sh:211` greps for a string that is never in
    the file it greps.** It runs
    `grep -a -A 9 'decode split' "$OUT/08-cold-decode-p7.log"`, but
    `cold_bench.py` spools the child's stderr to a sidecar and never echoes it
    to its own stdout, so the phase split that grep exists to surface has
    **never appeared in that log**: `grep -c 'decode split'` on phase 7's own
    `08-cold-decode-p7.log` returns 0. The split it was meant to surface lived
    only in the clobbered `run0N.json.stderr` sidecar, which is how phase 7
    came within one overwritten file of losing its single most useful finding.
    `cold_bench.py` now keeps stderr in the `--json` summary so the data
    survives, but the grep is still dead and should be pointed at the summary.
    The general form: a harness check that greps for a string is worth only as
    much as the last time someone watched it match, because a `grep` that finds
    nothing is silent.

One more earned in the cold sweep, and it is a coincidence rather than a bug.

11. **`pgsteal 2817` has now been recorded twice, a session and a workload
    apart, and nobody has explained it.** Recorded as unexplained and
    reproducible-looking. **Not** a diagnosis, and **not** a claim that the two
    events share a cause. What is measured: run 0 of
    `scratch/cold-bench/p8-20260806-165322-decode-3961.json`, the 3,961 rung's
    **discarded warmup**, is `hygiene: DIRTY` with `pgscan 2817` and `pgsteal
    2817`, **all of it `pgsteal_kswapd`** with khugepaged, direct and proactive
    at zero. That is genuine pressure under the `4b39104` rule, not the
    bookkeeping that rule exists to excuse, and it earned a hard verdict. It
    was the warmup, so it is discarded, the rung's three scored runs are CLEAN
    and the arm is PASS; no number in EXP-023 rests on it. What is odd: EXP-014
    records a discarded first attempt whose two DIRTY runs read `pgsteal`
    **2,817 and 2,946** pages, and 2,817 is the same integer. **It is not the
    same rung**, and saying so matters: EXP-014's prompt was five tokens ("The
    capital of France is") on a different binary at a different commit, where
    this is a 3,961-token prompt two days later. The two share a number and
    nothing else. EXP-014 attributed its two to the operator opening a terminal
    mid-run; this one has no such cause recorded, the sweep was unattended, and
    its settle loop had just read `MemAvailable` at 11,101 MiB. EXP-021 Note 10
    records that EXP-014's JSONs were overwritten, so its classification cannot
    be rechecked and this cannot be chased backwards. The same figure twice is
    not obviously random pressure, and what would turn it into a finding is a
    third occurrence with its counters kept, which is the reason to keep them.
    If you see it again: keep the JSON, and read `pgsteal_kswapd` rather than
    the bare total.
