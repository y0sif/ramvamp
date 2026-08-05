# Phase 8 handoff

Written 2026-08-05, at `feat/attention` = `4b39104` (phase 7 complete and
measured, not yet merged to `main`). The runtime is unchanged since `aade585`;
everything after it is harness and docs.

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
- **`T_BLOCK = 4` in `x86::dot_block` could go to 8.** The implementing lane
  called it the cheapest remaining win in that file: roughly **1.5-2x** on the
  QK dot, at 8 more live ymm registers and 16 KiB of stack. **Estimated, never
  measured.** It stays bit-neutral only because more independent position
  chains is legal; splitting one chain over `i` is not.
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
- **The 512-token phase split was not captured cold.** `cold_bench.py` writes
  the child's stderr — which carries the `prefill split` and `decode split`
  blocks — to `run0N.json.stderr` on a fixed path that the next invocation
  clobbers, so only the last session's 4K run and phase-5 512 run survive.
  Copy the sidecars alongside the `--json` summary if the split matters.
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

Carried over, untouched by phase 7:

- `crates/core/src/io/testutil.rs` hard-codes one fixture geometry and keeps
  its `TempDir` private, so `prefill.rs`'s `mod wide` had to copy the install
  builder. A `build_install_with(geometry)` there would let that copy delete
  ~360 lines.
- **io_uring queue-depth curve for the decode geometry.** EXP-019 emulated
  depth with threaded `preadv` and swept it only at K=8, so single-blob decode
  reads through `RING_ENTRIES = 8` have no curve on either the drive side or
  the submission side. `RING_ENTRIES` must not move on EXP-019 alone.
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
