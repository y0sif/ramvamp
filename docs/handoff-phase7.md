# Phase 7 handoff

Written 2026-08-04, at `main` = `d0403f2` (phase 6 merged).

## Where things stand

Phase 6 shipped chunked layer-major prefill and a CLI chat REPL. Prefill is
**2.56x faster cold** (1.65 to 4.23 tok/s, EXP-018, rule-2, hygiene PASS) and
expert bytes read fell **11.55x**. Output did not move: `bitident.py` is 8/8
byte-identical against the phase-4 baseline with the sweep in the path,
`kl_vs_reference.py` reproduces its phase-4 numbers exactly, and
`greedy_regression.py` matches every recorded metric.

Read before touching anything: `docs/architecture.md` (especially "Prefill
(sequential sweep)" and "The prefill arena"), then EXP-016, EXP-017, EXP-018
and EXP-019 in `docs/experiments/README.md`.

## WORK TO DO

### 1. Attention. This is the phase. (high)

EXP-017 measured it: attention is **61.3%** of a 512-token prefill and
**85.2%** of an 1891-token one. Its total cost is quadratic and the two
lengths confirm it (per-token 148.9 ms at 512, 555.7 ms at 1891, a ratio of
3.73 against a length ratio of 3.69). It runs **single-threaded on the decode
thread** with one shared `AttentionScratch`, so it takes nothing from the six
pinned P-cores, and it converts f16 to f32 per K/V element without
vectorization. Implied rate is roughly 0.68 GFLOP/s.

Two levers, both bit-neutral:

- **Parallelize over rows.** Rows are independent, so this cannot move a bit.
  It needs one score buffer per shard, which is **not** in the current scratch
  carve (`crates/core/src/model/prefill.rs`, the `Carver` and `scratch_bytes`),
  so the carve arithmetic and its pinning test both have to grow.
- **Vectorize the f16 to f32 conversion and the dot.** The kernel is
  `kernels/attention.rs`; `attention_at` is the prefill entry point and
  `decode_attention` delegates to the same private impl, so both benefit.

**Forbidden, and the reason is recorded:** any online, streaming or
flash-style rescaled softmax. It changes the reduction order and breaks the
bit-identity gate. The current structural mask works precisely because masked
positions are *absent* from the sum rather than zero-weighted.

Take a cold rule-2 measurement **after** this lands, not before. EXP-019's
backlog note says the same.

### 2. Settle the decode cold-start question (medium)

EXP-018 recorded decode falling 1.88 to 1.38 tok/s in the same run as the
prefill win. The hypothesis is that the sweep bypasses the cache and
invalidates occupancy, so decode now starts cold where token-major prefill
warmed it incidentally, and `--max-new 4` measures almost nothing but that
transient. Supporting but not conclusive: 725 ms/token sits just under the
re-derived 649-708 ms/token no-cache I/O-only row.

What settles it: the same paired cold run at a **much larger `--max-new`**, so
the steady state dominates the transient. If the transient is real and large,
the recorded decision "prefill does not warm the cache" (EXP-005, worth +0.09
points of hit rate) deserves revisiting, because EXP-005 measured hit *rate*,
not the cold-start cost.

### 3. Smaller open items

- **`memory.peak` residual.** EXP-018 measured 2,576.0 MiB, inside the 3,072
  ceiling, but 99-105 MiB above EXP-014 where the KV arithmetic explains only
  about 42. Unexplained.
- **4K-context `memory.peak`**, still the last open item on the 11-slot dial
  (EXP-014 Note 2). Cheaper now on the phase-6 binary.
- **io_uring queue-depth confirmation.** EXP-019 emulated depth with threaded
  `preadv`, and swept it only at K=8, so the decode geometry has no measured
  QD curve. `RING_ENTRIES = 8` is currently justified by where it sits on the
  bytes-in-flight curve (~24.5 MB, inside the plateau), which is sound but
  indirect.
- **Chunk-size sweep** (128/256/512/1024). A warm data point exists (161/143/
  131 s at 512 tokens) but it is a direction, not the sweep.
- **Kaggle or SSH-provider portability smoke.** The only path that exercises
  ext4 and the loop-device "O_DIRECT lies" fallback. Untouched since phase 4.
- `crates/core/src/io/testutil.rs` hard-codes one fixture geometry and keeps
  its `TempDir` private, so `prefill.rs`'s wide fixture had to copy the
  install builder. A `build_install_with(geometry)` there would let `mod wide`
  delete its copy.

## PROCESS

Same as phases 1-6, and it works: research-first with findings reported before
planning; plan with acceptance criteria approved before implementation;
parallel lanes with strict file ownership; adversarial review per wave;
mechanical gate (`cargo fmt --check`, `cargo clippy --all-targets -- -D
warnings`, `cargo test`, `cargo test -p ramvamp-core --no-default-features`)
before anything merges; y0sif tests locally before the merge.

Two amendments earned in phase 6:

- **Docs lanes run last**, never in parallel with the code they document.
- **Run a specialist adversarial reviewer** alongside the generic code review.
  In phase 6 the generic pass returned APPROVED with no blockers while a
  reviewer scoped to bit-identity, memory soundness and O_DIRECT found two
  silent-corruption blockers.

## GOTCHAS

- **`RAMVAMP_PREFILL=token-major`** keeps the old path. The strongest available
  numerics gate is that the two paths are byte-identical in one process
  (`sweep_and_token_major_agree_bit_for_bit`, and the wide-fixture variant).
  Do not delete the token-major path; it is the reference arm.
- **`bitident.py`'s 8 prompts are 4 to 12 tokens**, so they never cross a chunk
  seam. Seam coverage comes from the unit fixtures and from the warm A/B at
  chunk 128/256/512. A cold long-prompt baseline is still stronger and is not
  run.
- **`scratch/phase5-ref/ramvamp`** is the banked phase-5 binary (`c3572fd`).
  It is what makes rule-3 paired runs possible at any time. Do not delete it.
- **systemd rewrites `${VAR}` and `$$` in `ExecStart=`.** `cold_bench.py` now
  passes argv out of band as JSON. Any new harness that shells through
  `systemd-run` must do the same or it will silently truncate prompts while
  exiting 0.
- **Prefill scratch comes from the slot-pool arena**, not the heap. That is why
  prefill costs zero additional bytes. Anything new on that path must carve
  from `PrefillSession`, and `take_arena` refuses if a carve would cover a
  retired buffer that a kernel write may still own.
- **The trace writer needs records in position order**, but the sweep produces
  them layer-major. `RouteRecorder` in the CLI reorders them, because
  `scripts/lfu_sim.py` replays records sequentially and record order determines
  its hits and evictions.
- **Do not cite EXP-008.** Retired by EXP-019.
