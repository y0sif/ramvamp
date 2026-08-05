# Phase 8 handoff

Written 2026-08-05, at `feat/attention` = `aade585` (phase 7 complete, not yet
merged to `main`).

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

**None of that is publishable.** Every figure above is warm, in process and
diagnostic. Rule 2 wants a cold run inside `memory.max=3G` with
`memory.swap.max=0`, and nobody has taken one on this build. The honest
one-line summary of phase 7 is: *the kernel got much faster on a
microbenchmark, and nobody has measured what that did to a token.*

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

### 1. Take the cold measurement. This blocks everything. (high)

The runbook is committed at `scratch/phase7/wave3-runbook.md`. Roughly two
hours, most of it unattended, and it needs a **quiet machine**. It produces
**EXP-021**, which is already reserved by name in EXP-020 Note 9 and in the
architecture backlog.

Four things, in the runbook's order:

- **The numerics gate first**, because it is the cheapest way to find out bits
  moved: `bitident.py` against the phase-4 baseline (~85 s), `kl_vs_reference
  .py` (~3 min), `greedy_regression.py` (~50 min, unattended).
- **Paired cold prefill at 512 tokens, `--repeats 5`.** Closes EXP-018 Note 5,
  which flagged its own `--repeats 1` as leaving the spread unbounded.
  Baseline to beat: 4.23 tok/s cold.
- **Paired cold decode at `--max-new 256`.** This is what settles the EXP-018
  cold-start question. At `--max-new 4` almost everything measured was the
  transient; 256 tokens is 5x EXP-005's token-48 steady-state threshold, and
  context only grows 512 to 768 so the attention term drifts ~25% rather than
  100%. Keep the stderr: the new `decode split (forward_token):` block is what
  says whether attention's share of decode actually moved.
- **The 4K-context `memory.peak` run**, the last open item on the 11-slot
  dial, and the one that might explain EXP-018's unexplained 99-105 MiB
  residual.

**`kl_vs_reference.py` requires `--refresh`.** Its per-prompt dumps are cached
at `models/llamacpp-ref/llamacpp_ref/rv_single_*.json` and **the cache is not
keyed on the binary**. Without `--refresh` it re-scores stale JSON from disk in
seconds and reports PASS having tested nothing about this build.
`greedy_regression.py` *is* keyed on binary sha256 and will recompute
correctly; `bitident.py` runs fresh. This is one script with one wrong cache
key, and it is the one that gates the logits.

### 2. Next levers, with honest ceilings (medium)

None of these should be started before item 1. All four ceilings below are
**derived or estimated**, never measured.

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

### 3. Still carried over, untouched by phase 7 (low)

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
- **EXP-018's `memory.peak` residual.** 2,576.0 MiB, inside the 3,072 ceiling,
  but 99-105 MiB above EXP-014 where KV arithmetic explains only about 42. The
  4K run in item 1 is what might close it.

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
