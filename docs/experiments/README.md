# Experiment log

Every performance-motivated change gets a numbered entry here before it ships.
This discipline is borrowed from TurboFieldfare's 103-entry experiment record,
which is the reason their claims are credible.

## Rules

1. A microbenchmark starts an experiment; end-to-end speed and output quality
   decide whether it ships.
2. Cold measurements only for published numbers: run inside the benchmark
   cgroup (`memory.max=3G`, `memory.swap.max=0`; zram counts as swap) with a
   dropped page cache. The machine those numbers come from, and the
   constraints it puts on measurement, are recorded in
   `docs/benchmark-machine.md`.
3. Every entry records its own baseline. Entries use different machine states,
   so numbers from different entries must not be combined into one curve.
4. Changes claiming identical math must produce identical output. Changes that
   reorder floating-point operations must pass tolerance tests against
   reference outputs.
5. Negative results get entries too. They are the cheapest way to stop a bad
   idea from coming back.

## Entry template

```markdown
## EXP-NNN: short title

- Date / commit:
- Hypothesis:
- Method: (workload, machine state, how measured)
- Baseline:
- Result:
- Verdict: KEEP | REVERT | NEUTRAL
- Notes:
```

## Index

- [EXP-001: AVX2 K-quant dot kernels vs scalar reference](#exp-001-avx2-k-quant-dot-kernels-vs-scalar-reference) — KEEP
- [EXP-002: AVX2 activation quantizers](#exp-002-avx2-activation-quantizers) — KEEP
- [EXP-003: Forward-pass validation vs llama.cpp b10217 (same Q4_K_M bytes)](#exp-003-forward-pass-validation-vs-llamacpp-b10217-same-q4_k_m-bytes) — KEEP
- [EXP-004: Full-vocab KL vs llama.cpp reference dumps; scalar/AVX2 noise floor](#exp-004-full-vocab-kl-vs-llamacpp-reference-dumps-scalaravx2-noise-floor) — KEEP
- [EXP-005: Expert cache hit rate on measured routing traces (policy and slot sweep)](#exp-005-expert-cache-hit-rate-on-measured-routing-traces-policy-and-slot-sweep) — KEEP
- [EXP-006: The phase-4 baseline cannot be measured cleanly in a 3G cgroup](#exp-006-the-phase-4-baseline-cannot-be-measured-cleanly-in-a-3g-cgroup) — NEUTRAL
- [EXP-007: Slot aliasing under concurrent O_DIRECT reads](#exp-007-slot-aliasing-under-concurrent-o_direct-reads) — KEEP
- [EXP-008: Reference-drive characterisation under O_DIRECT (queue depth, block size, per-blob latency, ReadFixed)](#exp-008-reference-drive-characterisation-under-o_direct-queue-depth-block-size-per-blob-latency-readfixed) — NEUTRAL
- [EXP-009: Buffered vs O_DIRECT expert reads, page-cache charge inside the cgroup](#exp-009-buffered-vs-o_direct-expert-reads-page-cache-charge-inside-the-cgroup) — KEEP
- [EXP-010: Compute-pool signalling: bounded spin then futex](#exp-010-compute-pool-signalling-bounded-spin-then-futex) — KEEP
- [EXP-011: Row-range GEMV as the single code path](#exp-011-row-range-gemv-as-the-single-code-path) — KEEP
- [EXP-012: Anonymous runtime memory is missing from the memory contract](#exp-012-anonymous-runtime-memory-is-missing-from-the-memory-contract) — KEEP
- [EXP-013: io_uring + O_DIRECT streaming and the two-phase decode loop](#exp-013-io_uring--o_direct-streaming-and-the-two-phase-decode-loop) — KEEP
- [EXP-014: First clean cold measurement inside the 3G cgroup](#exp-014-first-clean-cold-measurement-inside-the-3g-cgroup) — KEEP
- [EXP-015: Phase-6 building blocks, landed and unmeasured](#exp-015-phase-6-building-blocks-landed-and-unmeasured) — KEEP
- [EXP-016: Chunked layer-major prefill lands as the default path](#exp-016-chunked-layer-major-prefill-lands-as-the-default-path) — KEEP
- [EXP-017: The prefill phase split, and attention is the wall](#exp-017-the-prefill-phase-split-and-attention-is-the-wall) — NEUTRAL

Entries EXP-007 through EXP-013 were measured on a machine that was not
quiet, and most are microbenchmarks rather than end-to-end runs. Under rule 2
none of their numbers is publishable; they are recorded so that the design
decisions they drove are traceable, and each states the re-measurement it
needs. EXP-013 is the exception worth naming: its **numerics** result
(byte-identical logits) is a correctness measurement that rule 2 does not
govern and that does stand as reported; only its throughput figures are
provisional.

## EXP-001: AVX2 K-quant dot kernels vs scalar reference

- Date / commit: 2026-08-02 / on top of 1663ea4 (`feat/cpu-kernels`, pre-commit)
- Hypothesis: runtime-dispatched AVX2+FMA ports of the `vec_dot_*` kernels
  (ggml's x86 recipes: `maddubs`/`madd` integer core, bsums-based min
  folding, one `fmadd` per super-block) beat the scalar reference by >= 3x
  per row at the audited GEMV shapes.
- Method: `crates/core/benches/kernels.rs` (`cargo bench -p ramvamp-core`);
  full row sweep over a 2048-row synthetic packed matrix per timed run,
  same dispatch wrapper for both sides (`force_scalar` flag), median of 31
  runs after 5 warmup. Machine: Core Ultra 9 185H, single thread,
  **warm cache — diagnostic numbers per rule 2, not publishable end-to-end
  results** (no cgroup, no cold page cache).
- Baseline: the scalar reference kernels in `kernels/quants/dot.rs`
  (same binary, forced via the dispatch escape hatch).
- Result (ns per output row; GB/s = packed row bytes / ns per row):

  | kernel      | shape (in x out) | scalar ns/row | scalar GB/s | avx2 ns/row | avx2 GB/s | speedup |
  |-------------|------------------|---------------|-------------|-------------|-----------|---------|
  | q4_k x q8_k | 2048x2048        | 456           | 2.53        | 120         | 9.61      | 3.80x   |
  | q5_k x q8_k | 2048x2048        | 449           | 3.13        | 141         | 9.97      | 3.18x   |
  | q6_k x q8_k | 2048x2048        | 533           | 3.15        | 156         | 10.79     | 3.43x   |
  | q6_k x q8_k | 768x2048         | 202           | 3.12        | 61          | 10.29     | 3.30x   |
  | q8_0 x q8_0 | 2048x2048        | 918           | 2.37        | 179         | 12.16     | 5.13x   |

  Summary: 3.2-3.8x on the k-quants, 5.1x on q8_0; ~10-12 GB/s effective
  weight bandwidth per core warm. A second run agreed within ~5% except
  q4_k avx2 (120 vs 141 ns/row across runs — treat the speedup as ~3.2-3.8x).
- Verdict: KEEP
- Notes: integer parts are bit-identical to scalar (tested); only float
  accumulation order differs (tolerance-tested at in-dims 2048/768/4096
  plus the scalar suite's adversarial blocks). All loads are `loadu`
  (1-byte alignment contract, misalignment-tested). Dispatch checks
  AVX2+FMA per call via the cached `is_x86_feature_detected!`. End-to-end
  decode impact must be re-measured cold inside the 3 GB cgroup once the
  phase-4 forward pass exists.

## EXP-002: AVX2 activation quantizers

- Date / commit: 2026-08-02 / on top of 1663ea4 (`feat/cpu-kernels`, pre-commit)
- Hypothesis: AVX2 ports of `quantize_row_q8_k` / `quantize_row_q8_0` are
  worthwhile even though quantization is once-per-token-per-row (the q8_k
  scalar path pays heavily for `round_ties_even` per element), while
  producing byte-identical blocks to the scalar reference.
- Method: same harness, machine, and caveats as EXP-001 (warm-cache,
  single-thread diagnostic): one 2048-float row quantized per timed run,
  median of 31 after 5 warmup. GB/s here is f32 *input* bytes (8 KiB/row).
- Baseline: the scalar reference quantizers in `kernels/quants/quantize.rs`
  (forced via the dispatch escape hatch).
- Result:

  | kernel        | row     | scalar ns/row | scalar GB/s | avx2 ns/row | avx2 GB/s | speedup |
  |---------------|---------|---------------|-------------|-------------|-----------|---------|
  | quantize q8_k | 2048 f32 | 11960        | 0.68        | 1065        | 7.69      | 11.23x  |
  | quantize q8_0 | 2048 f32 | 8324         | 0.98        | 1875        | 4.37      | 4.44x   |

- Verdict: KEEP
- Notes: outputs are byte-identical to the scalar reference on random,
  tie-heavy, flat, and zero rows (tested per rule 4 — this is an
  identical-math change, not a reordering). Deliberate deviation from
  ggml's own AVX2 quantizer, whose `_mm256_round_ps` ties-to-even rounding
  and `127/amax` scale differ from its scalar reference: we match the
  SCALAR reference (q8_0 rounds half away from zero via an exact tie
  fix-up; q8_k's `cvtps` under default MXCSR *is* `round_ties_even`; scale
  math stays in scalar f32). The q8_0 quantizer is slower than q8_k's
  AVX2 path because of the per-32-value tie fix-up and f16 scale rounding;
  still 4.4x over scalar.

## EXP-003: Forward-pass validation vs llama.cpp b10217 (same Q4_K_M bytes)

- Date / commit: 2026-08-03 / dd8b6ee..28c9a62 (`feat/forward-pass`)
- Hypothesis: the phase-4 forward pass (loading, KV/attention, MoE routing,
  generate/logits CLI) reproduces llama.cpp's outputs on identical Q4_K_M
  GGUF bytes.
- Method: Kaggle CPU runtime (30 GB RAM, AVX2). Model installed on-site by
  `ramvamp-repack`; source GGUF fetched at the pinned revision; llama.cpp
  prebuilt b10217. Single-position top-20 logprob comparison via
  llama-server, plus greedy raw-completion comparison via the /completion
  endpoint (`scripts/compare_llamacpp.py`).
- Baseline: llama.cpp b10217 on the same GGUF bytes.
- Result:
  - Greedy: 16-token completions character-identical on 3/3 prompts
    ("The capital of France is", "Water is composed of",
    "In Rust, ownership means").
  - Logits: top-1 agreement 2/2; top-20 overlap 20/20 and 19/20;
    union-renormalized truncated KL 0.0218 and 0.1007. Metric caveat:
    single-position, top-20-truncated, renormalized — NOT comparable to
    the full-vocab mean-KL <= 1e-3 design target.
  - ramvamp smoke on the Kaggle Xeon: 0.87 tok/s decode, single-thread,
    uncached.
- Verdict: KEEP (forward pass semantically validated)
- Notes: the full-vocab mean-KL measurement (architecture gate 3's actual
  target) remains deferred to the phase-7 benchmark rig; this entry records
  the top-20 truncated proxy only. Superseded on the KL question by EXP-004,
  which measured the full-vocab KL without waiting for phase 7.

## EXP-004: Full-vocab KL vs llama.cpp reference dumps; scalar/AVX2 noise floor

- Date / commit: 2026-08-03 / on top of 1dd48cd (`feat/forward-pass`,
  pre-commit)
- Hypothesis: the phase-4 forward pass meets architecture gate 3 as written
  (mean full-vocab KL vs llama.cpp <= 1e-3 on a fixed prompt set).
- Method: reference side captured once on the Kaggle rig — llama-server
  b10217 (ddd4ec142, prebuilt ubuntu-x64), same pinned Q4_K_M GGUF bytes,
  `/completion` with `n_predict=1, n_probs=151936, temperature=0,
  samplers=[], post_sampling_probs=false`, 8 raw-completion prompts, all
  151936 pre-sampling logprobs per prompt (f32; per-prompt prob mass
  1.0001-1.0004), stored as `single_*.npz` under `models/llamacpp-ref/`
  (local only, gitignored) with raw responses and `meta.json` provenance.
  Local side: `ramvamp logits --top 151936` per prompt (release build,
  185H). `scripts/kl_vs_reference.py` compares in f64 with both sides
  renormalized over the full vocab. Correctness measurement, not a
  performance number — cgroup/cold-cache rules do not apply. Noise-floor
  A/B: same binary rerun with `RAMVAMP_FORCE_SCALAR=1` (new env hook in
  `kernels/gemv.rs`) on 3 prompts, isolating dot-product float accumulation
  order (integer paths are bit-identical per EXP-001).
- Baseline: llama.cpp b10217 logprobs on identical GGUF bytes.
- Result (KL in nats, P = llama.cpp, Q = ramvamp-avx2):

  | prompt | KL(P‖Q) | KL(Q‖P) | TV | top-1 |
  |---|---|---|---|---|
  | The capital of France is | 2.72e-2 | 2.42e-2 | 9.0e-2 | agree |
  | Water is composed of | 2.08e-2 | 2.08e-2 | 8.8e-2 | agree |
  | In Rust, ownership means | 8.15e-3 | 7.52e-3 | 4.6e-2 | agree |
  | Once upon a time, in a village by the sea, | 2.65e-3 | 2.68e-3 | 2.7e-2 | agree |
  | The derivative of x^2 with respect to x is | 4.61e-3 | 4.38e-3 | 2.9e-2 | agree |
  | def fibonacci(n): | 7.26e-3 | 7.20e-3 | 5.6e-2 | agree |
  | The three primary colors are | 9.01e-3 | 9.72e-3 | 5.5e-2 | agree |
  | Photosynthesis is the process by which | 3.45e-3 | 3.59e-3 | 3.4e-2 | agree |

  Mean KL(P‖Q) 1.04e-2, mean KL(Q‖P) 1.00e-2, top-1 8/8 — FAILS the 1e-3
  target as written. Noise floor (first 3 prompts): ramvamp-avx2 vs
  ramvamp-scalar KL 5.7e-3 / 1.3e-2 / 4.5e-3 — the same order as the
  cross-engine gap, from reordering float accumulation alone. Direction is
  non-systematic: scalar lands CLOSER to llama.cpp than AVX2 on 2 of 3
  prompts (1.6e-2 vs 2.7e-2; 3.8e-3 vs 8.2e-3) and farther on the third
  (3.4e-2 vs 2.1e-2).

  Long-context (wikitext slices from the same capture; llama.cpp
  `/tokenize` ids == ramvamp `encode` ids exactly on all three slices and
  on all 8 short prompts):

  | ctx tokens | KL(P‖Q) | KL(Q‖P) | TV | top-1 |
  |---|---|---|---|---|
  | 512 | 8.03e-6 | 7.21e-6 | 1.0e-4 | agree |
  | 1891 | 1.52e-2 | 1.54e-2 | 6.9e-2 | agree |
  | 3492 | 5.00e-3 | 4.90e-3 | 3.6e-2 | agree |

  KL is flat in context depth — no growth from position 6 to position 3492
  (near the 4K cap). A RoPE / KV-indexing / attention bug would compound
  with position; none does. The 512-token point (8e-6) is an unusually
  low-entropy continuation, not a depth trend.
- Verdict: KEEP (measurement stands; no bug indicated; gate 3 revised, see
  the recorded decision below)
- Notes: per-position full-vocab KL of O(1e-2) is the floor for any two
  implementations of this 48-layer quantized stack that do not replicate
  arithmetic operation-for-operation; a 1e-3 mean is unachievable without
  operation-identical kernels. **Recorded decision (2026-08-03):** gate 3
  becomes mean full-vocab KL <= 3e-2, with the intra-engine scalar/AVX2 A/B
  reported alongside as the noise floor; perplexity (gate 5) remains the
  quality backstop. The 1e-3 target as written is recorded FAILED (mean
  1.04e-2) and withdrawn; the revised gate is PASSED on the same data. The
  decision is implemented as `KL_TARGET` in `scripts/kl_vs_reference.py` and
  written into gate 3 of `docs/architecture.md`.

  **Recorded decision (2026-08-04): gate 3 gains three more conditions.**
  The 2026-08-03 decision above left gate 3 as a single mean, and
  `scripts/kl_vs_reference.py` had since grown three further conditions
  that were never written down anywhere. They are now stated in gate 3 of
  `docs/architecture.md`, and recorded here as the decision that put them
  there. Gate 3 passes only when all four hold: (i) mean full-vocab KL
  <= 3e-2, unchanged; (ii) **every individual prompt <= 6e-2**
  (`KL_PROMPT_CEILING`), because a mean over 8 prompts hides one blown
  prompt behind seven good ones; (iii) **top-1 agreement on every prompt,
  gated rather than reported**, because a flipped argmax is a behavioural
  change at a KL the mean tolerates; (iv) **all 8 prompts actually
  scored**, because a prompt dropped for a tokenizer mismatch or a short
  reference dump shrinks the gate's denominator rather than the gate.

  Evidence for the 6e-2 ceiling, which the original change did not cite:
  the scalar/AVX2 A/B in this entry is the largest float-reordering
  perturbation this codebase can produce short of an algorithmic change,
  and it moved the worst per-prompt *cross-engine* KL to 3.4e-2. 6e-2 is
  therefore 1.76x above the largest perturbation ever measured here and
  2.2x above the worst status-quo prompt (2.72e-2, single_00). A single
  blown prompt trips it; reordering every dot product in the engine does
  not.

  Honest caveat, and it is condition (iii) rather than the ceiling: **the
  newly-gated top-1 agreement had no recorded margin.** This entry reports
  top-1 8/8 but never how close any prompt came to flipping, and it
  reports top-1 for the AVX2 side only, so a near-tie would fail
  condition (iii) on pure float noise. Mitigation shipped with the
  decision: `kl_vs_reference.py` now records the per-prompt top1-vs-top2
  gap on both sides in `kl_results.json` and prints the tightest. First
  measurement (2026-08-04, phase-4 build, same 8 prompts, `--skip-longs`):
  tightest gap **0.228 nats** on single_05 (`def fibonacci(n):`) on the
  ramvamp side, 0.494 nats on the llama.cpp side, with the full run at
  mean KL 1.039e-2, worst prompt 2.721e-2, top-1 8/8, 8/8 scored, PASS.
  No prompt is near a tie, so condition (iii) is not currently fragile -
  and that is now measured rather than assumed.

  Reference capture also produced per-position
  top-5000 dumps along 64-token paths (`path_*.npz`) and 128-token
  greedy texts (`greedy_texts.json`) for phase-5/6 regression fixtures.
  **Correction (2026-08-03, found while wiring these fixtures into
  `scripts/greedy_regression.py`):** this entry originally called
  `path_*.npz` *greedy* paths at context depth ~5000. Both halves are
  wrong. The paths are **sampled**: `chosen_ids` differs from the argmax
  at 5/64, 15/64 and 26/64 positions for path_00/01/02, and path_00's
  continuation begins " Barcelona" where greedy gives " Paris". And
  `depth: 5000` in `meta.json` is the top-k *dump* depth, not context
  length - actual context runs 5 to 68 tokens. Consequence: these
  fixtures must be replayed **teacher-forced on `chosen_ids`**, since a
  free-running greedy comparison would diverge at position 0 by
  construction.
  llama.cpp ran its AVX2 activation quantizer, whose rounding deliberately
  differs from the scalar reference ramvamp matches (EXP-002) — one more
  reorder-class contributor, indistinguishable in size from dot-order
  noise. Gate-5 reference also banked from the same capture:
  llama-perplexity b10217 on wiki.test.raw, `-c 512 --chunks 40` ->
  PPL 6.3810 +/- 0.16588 (full per-chunk log and the exact corpus bytes in
  `models/llamacpp-ref/`); phase 7 compares a ramvamp perplexity loop
  against this locally. Phase-4 speed observation from the long runs
  (diagnostic, warm cache, single thread): ~1.9-2.5 s/token at ctx
  512-3492 on the 185H — the baseline the phase-5/6 io and cache work
  must improve on. Kaggle is no longer needed for validation; everything
  compares against the saved dumps locally.

## EXP-005: Expert cache hit rate on measured routing traces (policy and slot sweep)

- Date / commit: 2026-08-03 / on top of d80cc84 (`feat/expert-streaming`,
  pre-commit)
- Hypothesis: the per-layer LFU expert cache reaches the 40-60% hit rate
  the performance model assumes, and ~10 slots/layer is the right dial.
- Method: `--trace-experts` records each layer's final top-k routing
  decision (after renormalization, before any expert read) to a compact
  binary trace; `scripts/lfu_sim.py` replays traces against the real
  per-layer strides from `experts/layout.json`. Four generations on the
  pinned Qwen3-30B-A3B install (factual, code/chat-template, long-context,
  greedy) totalling **556 decode tokens**. Simulation only - no cgroup or
  cold-cache rules apply; the I/O times below are derived, not measured
  end to end. Bandwidth constant 1.59 GB/s, a provisional figure from
  O_DIRECT random reads at the real expert stride on a machine that was
  NOT quiet (a concurrent reader was active); it needs re-measuring under
  rule 2 before any tok/s figure derived from it is published, and the
  same probe on a quiet run gave 1.35 GB/s, so treat the io ms/token
  column as optimistic by roughly 15%. (**The 1.59 GB/s constant is now
  recorded as unsourced.** EXP-008 tabulates the drive probes and none of
  them is 1.59; `scripts/lfu_sim.py` describes the same constant as a
  *sequential* ceiling rather than as random reads at the expert stride;
  and the value sits between EXP-008's random-read and large-block
  numbers, so it can be neither. The three descriptions cannot all be
  true. See EXP-008's Notes.)
  Policies compared on identical traces:
  per-slot LFU, expert-indexed LFU with counters surviving eviction
  ("ghost"), LRU, aged LFU, windowed LFU, and Belady offline-optimal.
- Baseline: the architecture doc's estimate of 40-60% hit rate at 10
  slots/layer, and its 4-8 tok/s expected decode band.
- Result (policy `lfu-ghost`; pool sizes use real strides, totals add the
  1023 MiB mmap'd common core and a 384 MiB FP16 KV cache at 4K):

  | slots | hit % | pool MiB | total MiB | fits 3G | io ms/token | io-only tok/s |
  |------:|------:|---------:|----------:|:-------:|------------:|--------------:|
  | 8     | 37.4  | 1046     | 2454      | yes     | 434         | 2.30          |
  | 10    | 44.8  | 1308     | 2715      | yes     | 383         | 2.61          |
  | 12    | 49.9  | 1569     | 2977      | yes     | 348         | 2.87          |
  | 16    | 58.1  | 2092     | 3500      | no      | 291         | 3.43          |
  | 24    | 70.3  | 3139     | 4546      | no      | 207         | 4.84          |
  | none  | 0     | 0        | 1407      | yes     | 690         | 1.45          |

  The `total MiB` and `fits 3G` columns are **superseded by EXP-012**: they
  add only the mmap'd common core and the KV cache to the pool, and omit the
  115.1 MiB of anonymous runtime memory that EXP-012 measured. With that
  tenant counted, 12 slots/layer does not fit. The `hit %` column is
  superseded by the Correction below.

  Policy deltas at 10 / 16 slots (hit %): ghost-LFU 44.8 / 58.1,
  windowed LFU 44.7 / 58.0, aged LFU 43.0 / 56.6, per-slot LFU 42.6 /
  55.4, LRU 42.6 / 57.1, Belady 55.8 / 72.0.

  Routing statistics: 109.3 of 128 experts touched per layer, router
  entropy 6.12 of 7.00 bits, top-8 mass 24.9%, consecutive-token reuse
  44.1%, infinite-cache ceiling 97.7% after 32 tokens. Miss mix at 10
  slots: 12.1% cold, 87.9% eviction, of which 52.5% are re-requested
  within 4 decode tokens. Per-layer hit rate spans 13.4% (layer 0) to
  66.5% (layer 31), with no early/late gradient. Cold start reaches
  within 2 points of steady state by token 48.
- Verdict: KEEP (measurement stands; three design changes follow)
- **Correction (2026-08-03, found by replaying the shipped `io/cache.rs`
  against these same traces):** the hit rates above understate the
  implementation by ~5 points, because of how the simulator models
  pinning rather than any policy difference. `scripts/lfu_sim.py`
  resolves a step one expert at a time, so it protects only experts
  *already fetched* during that step; a later miss may therefore evict a
  resident expert that the same token is about to request, and pay to
  read it straight back. The runtime's `LayerCache::plan` takes all
  `top_k` ids in one call and protects the whole step, which cannot
  happen. Replaying the shipped Rust over the identical four traces
  (213,504 accesses) reproduces the simulator **exactly** when driven
  one expert at a time - 95,626 hits at 10 slots and 106,523 at 12, with
  matching cold and eviction counts - and yields **50.02% at 10 slots
  and 54.48% at 12** when driven the way the runtime actually calls it.
  Batch-pinned 10 slots therefore beats sequential 12 slots. Read the
  slot sweep above as a lower bound; the per-slot ordering, the policy
  ranking, and the marginal-value curve are unaffected.
- Notes: three decisions come out of this entry. (1) **The dial moves to
  12 slots/layer**, the largest pool that fits `memory.max=3G`, worth
  +5.1 points and -35 io ms/token over 10 for 261 MiB; there is no knee,
  marginal value falls monotonically, so the cgroup is the binding
  constraint rather than diminishing returns. **Superseded by EXP-012:**
  that fit arithmetic omitted anonymous runtime memory, 12 slots/layer
  does not fit once it is counted, and the dial is now a pool byte budget
  of 1,438.6 MiB, which is 11 slots/layer on this model. The corrected
  marginal value of 12 slots over 10 is **+4.46 points** (50.02 to
  54.48), not +5.1; see the Correction above. (2) **The LFU win comes
  from ghost history, not from LFU.** Counters must be indexed by expert
  id over all `n_experts` and survive eviction (128 x u32 = 512 B per
  layer, 24 KiB total); per-slot LFU is worth **-1.7 to 0.0 points
  against LRU** at the slot counts measured here (tied at 42.6% at 10
  slots; 55.4% against LRU's 57.1% at 16). `docs/architecture.md` said
  "LFU eviction with recency tie-break", which reads as per-slot counters
  and is the weaker policy, so the conclusion is stronger than the
  original wording of this note made it: without ghost history, LFU is
  not a small win over LRU, it is a small loss. (An earlier version of
  this note, and of `docs/architecture.md`, also said per-slot LFU
  "loses at 48" slots. No 48-slot row was ever recorded here.
  `scripts/lfu_sim.py` can sweep 48, so the claim is checkable, but until
  the row is in this log it is withdrawn.) (3) **A global slot pool and
  prefill cache warming are both closed as "no"** - the best static
  per-layer split buys +0.53 points at the operating point, and replaying
  the prompt into the cache buys +0.09. Upstream's 66.6% at 16 slots does
  not reproduce here: this simulation gives 58.1% at 16 slots, 8.5 points
  low, but that is **not a like-for-like comparison**. 58.1% is the
  sequential lower bound described in the Correction, and the
  batch-pinned figure at 16 slots has never been computed, so the honest
  statement is that the absolute levels disagree by at most 8.5 points
  and by an unknown amount in truth. The 16->24 and 16->32 deltas match
  their published shape either way, so no published claim should lean on
  their absolute number. The doc's 4-8 tok/s band assumed 3.6 GB/s; the
  2.87 tok/s I/O-only ceiling this entry derives at 12 slots inherits the
  unsourced 1.59 GB/s constant and is re-derived in
  `docs/architecture.md`. I/O is not yet the binding constraint, but the
  two numbers that say so come from different machine states and must not
  be combined per rule 3: phase-4 decode is ~2 s/token warm-cache and
  uncgrouped (EXP-004), while the 690 ms of uncached expert I/O is a
  simulation, not a measurement. Trace capture verified
  numerically inert: `ramvamp logits --top 1000` output is SHA-256
  identical with and without `--trace-experts`.

## EXP-006: The phase-4 baseline cannot be measured cleanly in a 3G cgroup

- Date / commit: 2026-08-03 / 650b5ea (`feat/expert-streaming`)
- Hypothesis: the phase-4 decode baseline (~1.9-2.5 s/token, recorded in
  EXP-004's notes) can be re-measured under this log's rule 2 - cold page
  cache, inside `memory.max=3G` with `memory.swap.max=0` - to give phase 5
  a publishable number to beat.
- Method: new `scripts/cold_bench.py`. Evicts the model via
  `posix_fadvise(POSIX_FADV_DONTNEED)` over all 53 files the runtime
  touches and **verifies eviction with `mincore`** rather than trusting
  the return code (fadvise reports success and evicts nothing when a
  process holds the file mmap'd, so the harness also refuses to start
  while any `ramvamp` process is alive). Runs the binary under
  `systemd-run --user --wait -q --collect -p MemoryMax=3G
  -p MemorySwapMax=0 -p MemoryAccounting=yes`, re-execing an inner
  wrapper that reads `memory.peak`, `memory.events` and `memory.stat`
  from inside the cgroup before exit, plus `/proc/self/io read_bytes`
  for block-layer bytes. Workload: `generate --max-new 8` (13 tokens
  total), one discarded warmup plus one scored run.
- Baseline: none - this entry establishes whether a baseline is
  measurable at all.
- Result: **it is not.** Scored run, verified cold (3595.0 MiB resident
  evicted to 0.0 across 53 files in 2.0 s):

  | metric | value |
  |---|---|
  | wall / decode | 18.56 s, 0.81 tok/s |
  | `MemoryPeak` | 3072.0 MiB (pinned at the 3072.0 MiB ceiling) |
  | `memory.events max` | 6203 |
  | `pgscan` / `pgsteal` | 1,263,046 / 1,263,046 (~4.82 GiB reclaimed) |
  | `read_bytes` | 8,217,182,208 (7836.5 MiB) |

  A 13-token generation pulled **7.8 GiB** through the block layer and
  charged all of it to the cgroup as page cache, because phase 4 reads
  experts with buffered `pread`. The cgroup sat on its limit for the
  whole run and the kernel reclaimed continuously. Every number from
  such a run is a measurement of reclaim behaviour, not of decode.
- Verdict: NEUTRAL (no change shipped; the finding redefines the gate)
- Notes: three consequences. (1) **The "baseline to beat" of 1.9-2.5
  s/token was never a rule-2 number** - it came from warm-cache,
  uncgrouped diagnostic runs in EXP-004, and it must be cited that way
  until phase 5 can produce a clean one. (2) **`memory.events max == 0`
  is not a sufficient hygiene check.** A separate run showed `max 0`
  while `pgsteal` revealed 2.4 GiB had been silently reclaimed; clean
  page cache is dropped well before the hard limit trips. `cold_bench.py`
  therefore gates on `pgsteal == 0` and prints an explicit
  CLEAN/DIRTY verdict separate from the performance figures. `pgsteal` is
  the check this entry motivated, but quoting it alone understates what a
  CLEAN verdict now asserts. A run is CLEAN only if **all** of the
  following hold, each of them a hard problem on its own: the inner
  wrapper resolved its own cgroup; `memory.stat` exposed
  `pgscan`/`pgsteal` at all, and `pgsteal == 0`; `memory.events` was
  readable and its `max`, `oom`, `oom_kill`, `oom_group_kill` and `high`
  are all zero; `memory.max` was readable and equals the requested limit;
  `memory.swap.max` was readable and is 0; `memory.swap.peak` was readable
  and is 0; `read_bytes` from the block layer is nonzero (a zero delta
  means eviction failed and the run was warm after all); and both
  `ramvamp` and `systemd-run` exited 0. Each of those counters has three
  states rather than two, and **unknown is DIRTY** - an unreadable counter
  is exactly what an unconfined run produces, so a check that exempted its
  own missing input would make the least-confined run the cleanest one the
  harness can report. The single counter that is *systematically* missing
  rather than diagnostic is `memory.swap.peak`, which does not exist
  before Linux 6.5; on an older kernel every run would be DIRTY with no
  remedy, so `preflight()` refuses to start there (exit 2) instead of
  reporting a dirty measurement forever. (3) The
  first clean 3G measurement this project produces will be phase 5's,
  and the buffered-to-O_DIRECT transition is precisely what makes it
  possible: the same 1.4 GiB of expert reads measured a 1092.2 MiB
  cgroup peak buffered versus 5.0 MiB with O_DIRECT. That transition
  gets its own entry when it lands. (The *measurement* is now written up
  as EXP-009. **Both halves of this note have since been discharged:** the
  transition landed in EXP-013, and the clean rule-2 baseline it promised
  is EXP-014, 1.88 tok/s decode at a 2,471.1 MiB cgroup peak.)

## EXP-007: Slot aliasing under concurrent O_DIRECT reads

- Date / commit: 2026-08-03 / probe predates c7f4870
  (`feat/expert-streaming`); it is what `crates/core/src/io/slots.rs` was
  written against.
- Hypothesis: two concurrent O_DIRECT reads may share one destination
  buffer as long as their byte ranges do not overlap, so slot ownership can
  be expressed as index arithmetic rather than as an explicit free list.
- Method: ad-hoc probe on the reference 185H against the installed expert
  files on the btrfs volume, issuing concurrent O_DIRECT reads at QD
  4/8/16/32 into (a) aliased and (b) exclusively leased destination buffers,
  counting `EIO` returns and reading the volume's `corruption_errs` counter
  before and after. Counts, not timings. **The probe harness is not in the
  repo**, the machine was not quiet, and the run was not cgrouped: a
  provisional microbenchmark under rule 2, and none of its numbers is
  publishable.
- Baseline: aliased destinations, which is the index-arithmetic design.
- Result: aliasing two in-flight reads onto one buffer makes btrfs fail
  checksum verification, measured at **13-27% spurious `EIO`** across
  repeats, with matching increments to the filesystem's persistent
  `corruption_errs` counter. An explicit free list handing out exclusive
  leases measured **0 `EIO` across 4,000 reads** at QD 4/8/16/32. Second
  finding from the same session: btrfs runs direct reads with page faults
  disabled (`fs/btrfs/direct-io.c`) and silently completes through the
  buffered path when it cannot fault the destination, with no error and a
  full byte count returned, so the pool must fault every page at
  construction.
- Verdict: KEEP
- Notes: this is why `SlotPool::acquire` returns an owning `SlotGuard` and
  there is no by-index accessor: the aliasing bug is not expressible. Index
  arithmetic is not sufficient because completions arrive out of order. The
  13-27% spread is across repeats on an unquiet machine and should be read
  as "frequently, not always" rather than as a rate to quote; re-measurement
  under rule 2 is required before that rate appears anywhere outside this
  log. The qualitative result, that aliasing corrupts and exclusive leases
  do not, is what the design rests on and does not depend on the rate. The
  consequence of the second finding is untested end to end: nothing yet
  asserts at runtime that expert reads really are bypassing the page cache
  (see EXP-009's Notes).

## EXP-008: Reference-drive characterisation under O_DIRECT (queue depth, block size, per-blob latency, ReadFixed)

- Date / commit: 2026-08-03 / phase-5 design pass, before c7f4870
- Hypothesis: at the real 2.918 MiB expert stride, queue depth rather than
  block size sets throughput, and registered buffers (`ReadFixed`) are worth
  their pinning cost.
- Method: ad-hoc O_DIRECT probes on the reference machine (Core Ultra 9
  185H, Micron 2400 DRAM-less QLC, btrfs). **Harness not in the repo**,
  machine not quiet, no cgroup, page-cache state not controlled: provisional
  under rule 2 and not publishable. Two distinct probe series were run, and
  the distinction matters because they disagree: (a) a throughput series
  (queue depth swept at the expert stride, then block size swept), and (b) a
  per-blob latency series plus a `ReadFixed`-versus-`Read` CPU comparison.
  The two series were not run together.
- Baseline: none. This is characterisation, not a change.
- Result, series (a), throughput:

  | probe | value |
  |---|---|
  | 2.918 MiB blobs, QD4 | 1.211 GB/s |
  | 2.918 MiB blobs, QD8 | 1.349 GB/s |
  | 2.918 MiB blobs, QD16 | 1.390 GB/s |
  | block size 2.918 MiB | ~1.35 GB/s |
  | block size 8 MiB | 1.86 GB/s |
  | block size 16 MiB | 2.04 GB/s |
  | block size 24 MiB | 2.15 GB/s |

  Series (b), latency and CPU: per-blob **p50 2.34 ms at QD1** and **15.56
  ms at QD8**; `ReadFixed` saves **~70 us of CPU per 3 MiB read** against
  plain `Read`, about 4.5% of one core out of 22 at the decode read rate.

  Two internal contradictions, recorded rather than smoothed over:

  1. 2.34 ms for one 3,059,712 B blob is **1.31 GB/s at QD1**, above series
     (a)'s 1.211 GB/s at QD4. Taken together, the two series say the drive
     reaches series (a)'s QD4 number with no queue at all, which is not
     consistent with "the drive saturates by QD4" as a claim about absolute
     level.
  2. 15.56 ms p50 with 8 blobs in flight is **1.57 GB/s aggregate**, 17%
     above series (a)'s 1.349 GB/s at the same depth, and within 2% of the
     1.59 GB/s constant `scripts/lfu_sim.py` uses. Whatever 1.59 GB/s is, it
     is not from series (a).
- Verdict: NEUTRAL (characterisation; three decisions rest on it, and one
  widely used constant turns out to be unsourced)
- Notes: decisions leaning on this entry are QD4-8 rather than deeper (the
  *shape* of series (a), where QD4 is 87% of QD16, survives the level
  disagreement between the series, and since the decode loop waits on all
  misses the low end is preferred for latency); 16-24 MiB streaming buffers
  for the phase-6 prefill sweep (+51% at 16 MiB over the expert stride,
  **since demoted**: EXP-013 measured 1.97 GB/s under the real access
  pattern at the same 2.918 MiB stride, roughly 46% above the ~1.35 GB/s
  denominator that +51% is computed against, and the harness behind this
  entry was never committed. The dial range survives as a range to sweep;
  the +51% does not survive as a reason for it. See EXP-015 Note 5); and
  `ReadFixed` rejected, where the ~70 us is real but small against pinning
  1.4-1.6 GiB with `FOLL_LONGTERM` under an 8 MiB `RLIMIT_MEMLOCK`.
  **Unresolved, and it needs a measurement rather than an edit:** the 1.59
  GB/s constant that the whole performance model is built on has three
  mutually incompatible descriptions. `scripts/lfu_sim.py` calls it a
  measured *sequential* ceiling; EXP-005 calls it *random reads at the
  expert stride on a non-quiet machine with a concurrent reader* and says a
  quiet run of the same probe gave 1.35 GB/s; `docs/architecture.md`
  simultaneously described the whole table above, 1.35 included, as taken on
  a machine that was not quiet. The value also sits between this table's
  random-read numbers (1.211-1.390) and its large-block numbers (1.86-2.15),
  so it cannot be either. Until the probe is redone under rule 2 with a
  recorded run log, derive the performance model from the tabulated
  1.211-1.349 GB/s at the chosen QD4-8 operating point and treat 1.59 GB/s
  as unsourced.

## EXP-009: Buffered vs O_DIRECT expert reads, page-cache charge inside the cgroup

- Date / commit: 2026-08-03 / measured alongside the EXP-006 harness work
- Hypothesis: O_DIRECT is a performance preference, and buffered `pread` is
  acceptable inside a 3 GB cgroup because the expert working set per token
  is small.
- Method: the same 1.4 GiB of expert reads issued inside a `memory.max=3G`
  cgroup twice, once buffered and once with O_DIRECT, reading `memory.peak`
  from inside the cgroup. A read-volume probe, not a decode run; machine not
  quiet; provisional under rule 2. This is a memory measurement rather than
  a timing, so page-cache state matters less here than it would for a
  throughput number, but it is still uncontrolled.
- Baseline: buffered `pread`, which is what phase 4 ships.
- Result: cgroup peak **1,092.2 MiB buffered** versus **5.0 MiB with
  O_DIRECT** for the identical read volume. Buffered reads charge every
  expert byte to our cgroup as page cache, and the kernel then reclaims
  continuously against everything else in the budget. Consistent in
  direction with EXP-006, where a 13-token generation pulled 7.8 GiB through
  the block layer and held the cgroup at its 3,072 MiB ceiling for the whole
  run.
- Verdict: KEEP (O_DIRECT is a budget-correctness requirement, not a
  performance preference)
- Notes: this is the measurement EXP-006 promised an entry for. The
  **transition itself has not landed**: the decode loop still reads experts
  with buffered `pread`, so EXP-006's actual promise, a clean rule-2
  baseline once experts go through io_uring plus O_DIRECT, is still
  outstanding and gets its own entry when the loop is wired. Two things this
  entry does not measure. First, whether O_DIRECT is honoured at runtime:
  btrfs, tmpfs and loop-backed filesystems all accept the open and silently
  fall back to buffered under the conditions listed in
  `docs/architecture.md`, and `statx(STATX_DIOALIGN)` cannot be trusted to
  report it, so the runtime owes an empirical page-cache-residency assertion
  at startup that does not exist yet. Second, what O_DIRECT costs in
  throughput: the 5.0 MiB side was not timed against the 1,092.2 MiB side.

## EXP-010: Compute-pool signalling: bounded spin then futex

- Date / commit: 2026-08-03 / c7f4870 (`crates/core/src/threads.rs`)
- Hypothesis: a per-GEMV handoff between the coordinator and the compute
  workers can afford a sleeping primitive, and `std::sync::mpsc` is the
  convenient one.
- Method: two probes, which are not comparable to each other. (a) An ad-hoc
  handoff microbenchmark on the reference 185H comparing a futex wake/wait
  pair, pure atomic spinning, and `std::sync::mpsc`, recording latency
  percentiles and steady-state CPU occupancy in cores; **harness not in the
  repo**. (b) A whole-pool round-trip probe that is in the repo and
  rerunnable: `cargo test -p ramvamp-core -- --ignored wake_latency
  --nocapture` (`threads::tests::wake_latency_probe`), release build, 20,000
  timed `ComputePool::run` calls after 2,000 warmup, measuring publish ->
  wake -> all workers -> barrier. Both warm, uncgrouped, on a machine that
  was not quiet: provisional and diagnostic under rule 2, not publishable.
- Baseline: `std::sync::mpsc`, the default choice.
- Result, probe (a): a futex wake/wait pair is **p50 3.1 us** at **0.07
  cores** of steady-state overhead; pure atomic spinning is **502 ns** but
  burns **1.03 cores**; `std::sync::mpsc` is **p99 237 us**, which is
  disqualifying for a per-GEMV barrier. Spinning buys ~2.6 us of latency in
  exchange for a whole core, which is one of the six the GEMVs need.

  Probe (b): the shipped compromise, a bounded spin of 64 `pause` rounds
  followed by a futex wait, measures **p50 1.71 us** for the full round
  trip.
- Verdict: KEEP (bounded spin then futex)
- Notes: 1.71 us is a whole-pool round trip across N workers, and the 3.1 us
  and 502 ns figures are single wake/wait pairs from a different probe on a
  different harness. They must not be put on one curve (rule 3), and in
  particular 1.71 us is **not** evidence that the pool beats a bare futex:
  probe (b) warms up for 2,000 iterations precisely so that the workers are
  inside the spin window, which is the mid-token state, so it measures the
  spin path and not the futex path. What the futex path costs when the pool
  has gone idle between tokens has not been measured. Only probe (b) is
  reproducible from the repo. This cost model also feeds the open
  dedicated-E-core-reactor experiment; with the reactor inline on the
  coordinator in v0, the choice is smaller than it looks.

## EXP-011: Row-range GEMV as the single code path

- Date / commit: 2026-08-03 / c7f4870 (`crates/core/src/kernels/gemv.rs`)
- Hypothesis: GEMV can be split into contiguous row ranges for the compute
  pool without changing a single output bit, so the whole-matrix entry
  points can become thin wrappers over the row-range ones and there is one
  code path rather than two that can drift.
- Method: a rule-4 identical-output test, not a performance measurement.
  Row-partitioned results compared bit for bit (`f32::to_bits`) against the
  unpartitioned result over 8 partition schemes on 10 real model shapes, on
  both the AVX2 and the scalar path (`cargo test -p ramvamp-core`).
- Baseline: the whole-matrix GEMV entry points.
- Result: bit-identical on every shape and every scheme, on both paths. Each
  output row is an independent dot product, so partitioning by rows cannot
  reorder any accumulation; the test confirms the implementation matches
  that argument. **No speed measurement**: the pool is not wired into the
  forward pass yet, so there is no end-to-end number and none is claimed.
- Verdict: KEEP (enabler; correctness gate passed, performance not yet
  measured)
- Notes: the parallel-decode speedup this exists for is a phase-5 wave-2
  measurement, and it is the one that decides whether compute or I/O is the
  binding constraint (see EXP-005's Notes and the performance model in
  `docs/architecture.md`). Nothing in this entry supports a tok/s claim.

## EXP-012: Anonymous runtime memory is missing from the memory contract

- Date / commit: 2026-08-03 / on top of c7f4870 (`feat/expert-streaming`)
- Hypothesis: the memory contract's three tenants, the mmap'd common core,
  the KV cache and the expert slot pool, account for everything charged to
  the benchmark cgroup.
- Method: peak anonymous memory sampled from inside the cgroup during a live
  decode run. `common.bin` is mmap'd, so it is charged as `file` rather than
  `anon`; the figure is therefore everything the runtime allocates rather
  than maps, which is activations and scratch, tokenizer structures, thread
  stacks, allocator arenas, and whatever KV pages the run actually touched.
  **The run's context length, token count and page-cache state are not
  recorded**, so this is provisional under rule 2 and has to be retaken
  alongside a wired-in slot pool at the 4K context the contract budgets for.
- Baseline: the memory contract as written, which budgets zero for this
  tenant.
- Result: **115.1 MiB** of peak anonymous memory. Added to the audited
  tenants (common core 1,023.34 MiB, FP16 KV at 4K 384 MiB, and the expert
  slot pool at the real per-layer strides from `experts/layout.json`):

  | slots/layer | pool MiB | subtotal MiB | vs 3,072 MiB |
  |---:|---:|---:|---:|
  | 10 | 1,307.81 | 2,830.25 | 241.75 spare |
  | 11 | 1,438.59 | 2,961.03 | 111.0 spare |
  | 12 | 1,569.38 | 3,091.82 | **19.8 over** |
  | 16, FP16 KV | 2,092.50 | 3,614.94 | 542.9 over |
  | 16, Q8 KV | 2,092.50 | 3,422.94 | 350.9 over |

  The 12 slots/layer that EXP-005 chose does not fit. 11 slots/layer does,
  with 111.0 MiB spare.
- Verdict: KEEP (dial revised from 12 to 11 slots/layer, that is, an expert
  pool byte budget of 1,438.6 MiB on this model)
- Notes: (1) The dial `docs/architecture.md` always specified is a **byte
  budget** divided by layer count, and this is the measurement that makes
  the distinction bite: 1,438.6 MiB of pool is 11 slots/layer on
  Qwen3-30B-A3B, the byte figure is the portable one, and the slot count is
  its consequence on this model. Today's APIs,
  `SlotPool::new(slots_per_layer, layer_strides)` and
  `LayerCache::new(n_slots, n_experts)`, both take slot counts, so the byte
  budget is design intent that wave 2 implements and not current fact.
  (2) **The hit rate at 11 slots/layer has not been measured.** It is
  bracketed by the batch-pinned replay figures at 10 and 12 slots (50.02%
  and 54.48%, EXP-005 Correction); interpolating between them is not a
  measurement, and the sweep should be re-run at 11 before any hit rate is
  attached to the shipped dial. (3) 16 slots/layer no longer closes under
  any single saving: it needs a **4G cgroup** (3,614.9 MiB, or 3,422.9 MiB
  even with a Q8 KV cache). It is an experiment about larger machines rather
  than a tuning step on this one, and the backlog item is restated that way.
  (4) Double-counting caveat: the KV cache is allocated zeroed at full
  context capacity and faulted lazily, so a short decode run charges only
  the KV rows it actually touched, and the 115.1 MiB therefore includes a
  small, unquantified slice of the 384 MiB KV line. The slot pool is not
  double-counted: it faults every page at construction, but it is not yet
  wired into decode, so none of it was resident when this was sampled. Both
  facts point the same way, which is that 115.1 MiB is a floor for this
  tenant and the 111.0 MiB of spare at 11 slots/layer is not yet proven.
  (5) This partly answers and partly supersedes the backlog item "peak RSS
  of scratch + program + tokenizer against the headroom the dial leaves":
  the tenant to budget is `anon`, program text is file-backed and lands with
  the mmap'd core rather than here, and the number is now measured rather
  than guessed.
  (6) **Provenance correction (2026-08-04).** `docs/architecture.md`
  labelled the 115.1 MiB cell "(measured, EXP-012)" in its memory-contract
  table, which contradicted this entry: the Method above records that the
  run's context length, token count and page-cache state were not captured,
  so the figure fails rule 2 and is *provisional*, not measured, under that
  document's own three-way provenance rule. The cell now reads provisional,
  and so do the Subtotal, the 111.0 MiB headroom, and the `subtotal MiB` /
  `vs 3,072 MiB` columns of the slot table, all of which are arithmetic on
  it — including the 19.8 MiB overshoot that moved the dial from 12 to 11.
  The doc's Open-risk paragraph already explained this 35 lines further
  down; the headline row is the one most likely to be quoted without it.
  Nothing about the numbers changed, only what may be published.

## EXP-013: io_uring + O_DIRECT streaming and the two-phase decode loop

- Date / commit: 2026-08-04 / 477618d (`feat/expert-streaming`)
- Hypothesis: replacing the synchronous uncached per-token expert preads
  with io_uring + O_DIRECT reads into a per-layer ghost-LFU slot cache,
  overlapping miss reads with hit compute, and running every GEMV
  row-parallel across pinned P-cores, improves decode throughput **without
  changing the output at all**.
- Method: `models/qwen3.rvmp` on the reference machine
  (`docs/benchmark-machine.md`). Decode is the coarse two-phase shape:
  submit all misses, compute all cache hits row-parallel staging each
  expert's `[hidden]` output, wait for all misses, compute those, then
  reduce in fixed top-k order. Numerics gate is
  `scripts/bitident.py compare models/llamacpp-ref/phase4-baseline`, 8
  prompts at `--top 4096`, SHA-256 per prompt. Speed is
  `ramvamp generate --greedy --skip-hashes`, single run.
  **Warm page cache, no cgroup, and the machine was not quiet (review
  agents were running concurrently). Under rule 2 none of the throughput
  numbers here is publishable** - they are a direction check. The
  publishable measurement is a cold run inside `memory.max=3G`, which
  needs a quiet machine and has not been taken.
- Baseline: the phase-4 decode path at `d80cc84`, measured on the same
  machine and prompt in the same session. Its own baseline caveat stands
  (EXP-006): phase 4 cannot produce a clean 3G-cgroup number at all,
  because buffered preads charge every expert read to the cgroup as page
  cache.
- Result:

  **Numerics (the acceptance gate).** Logits **byte-identical on 8/8
  prompts**. Verified with a control first: the unmodified phase-4 binary
  was re-run against the same baseline on the same machine before the
  phase-5 build was tested, so a failure would have been attributable.
  Re-verified after the cache dial changed from 10 to 11 slots/layer,
  confirming the cache is numerically transparent.

  **Throughput**, warm, `"The capital of France is"`, greedy, 32 new
  tokens, paired on the same machine:

  | | prefill | decode |
  |---|---|---|
  | phase 4 | 0.66 tok/s | 0.75 tok/s |
  | phase 5 | 1.53 tok/s | **1.83 tok/s** |

  Generated text character-identical between the two.

  **Cache behaviour** at the shipped 11 slots/layer default (1,440 MiB
  budget), 25 prompt tokens plus 64 decode tokens, **split by phase**:

  | metric | prefill | decode |
  |---|---|---|
  | requests | 9,600 | 24,192 |
  | hit rate | 45.3% (4,348 hits) | **52.7%** (12,749 hits) |
  | pending hits | 0 | 0 |
  | misses | 5,252 (2,517 cold / 2,735 eviction) | 11,443 (809 cold / 10,634 eviction) |
  | expert bytes read | 14.0 GiB in 5,252 reads | 30.5 GiB in 11,443 reads |
  | read retries / stale completions | 0 / 0 | 0 / 0 |
  | I/O wait | 7.82 s | 16.63 s of 34.20 s decode |
  | mode | io_uring + O_DIRECT, probe `verified` | same |

  An earlier revision of this entry quoted a single blended 50.6% over
  both phases and compared it against EXP-005, which simulates decode
  records only. `StreamStats` was cumulative from `ForwardState`
  construction with no phase split, so the two were never comparable.
  `ExpertStream::stats_in(StreamPhase)` now separates them and the table
  above is a re-measurement, not an annotation. The two phases differ
  exactly as expected: prefill is cold-dominated (48% of its misses are
  first-touch) while decode is eviction-dominated (93%), which is the
  signature of a working cache on a warm working set.

- Verdict: KEEP
- Notes: four things worth carrying forward. (1) **The offline simulator
  predicted this to within half a point.** EXP-005's batch-pinned replay
  gave 50.02% at 10 slots and 54.48% at 12, so linear interpolation puts
  11 slots at ~52.3%; the running implementation measures **52.7%** on
  decode. The simulation is usable for future dial decisions rather than
  needing a full generation run each time, which matters because a real
  run costs ~50 s and a quiet machine. (2) **The drive is faster under
  the real access pattern than the synthetic probe suggested** - decode
  moved 32.75 GB in 16.63 s of I/O wait, about 1.97 GB/s, against EXP-008's
  1.211-1.390 GB/s at the same block size and queue depth. EXP-008's
  numbers were taken on a contended machine and are already flagged as
  needing re-measurement; this strengthens that. It also means the
  performance model built on 1.35-1.59 GB/s is pessimistic. (3) **I/O is
  still the wall**, 16.63 s of the 34.20 s decode, which is the expected
  shape and the reason the cache dial matters more than compute
  parallelism. (4) Zero read retries and zero stale completions across
  16,695 reads on btrfs, with the O_DIRECT capability probe reporting
  `verified`, so the page-cache bypass the 3 GB budget depends on is
  confirmed on the real path rather than assumed.

## EXP-014: First clean cold measurement inside the 3G cgroup

- Date / commit: 2026-08-04 / a58a701 (`feat/expert-streaming`)
- Hypothesis: with O_DIRECT expert reads the runtime can produce a
  measurement that satisfies rule 2 - cold page cache, inside
  `memory.max=3G` with `memory.swap.max=0`, with no reclaim - which
  EXP-006 established phase 4 could never do.
- Method: `scripts/cold_bench.py --max-new 64 --repeats 5`, run by y0sif
  on the reference machine (`docs/benchmark-machine.md`). Per scored run:
  evict all 53 files the runtime touches via
  `posix_fadvise(POSIX_FADV_DONTNEED)` and **verify** eviction with
  `mincore` (the harness refuses to start while any `ramvamp` process
  holds the model mmap'd, because `fadvise` silently evicts nothing in
  that case); launch under
  `systemd-run --user --wait -p MemoryMax=3G -p MemorySwapMax=0
  -p MemoryAccounting=yes`; read `memory.peak`, `memory.events` and
  `memory.stat` from **inside** the cgroup before exit. One warmup run
  discarded, 5 scored, medians reported. Workload:
  `generate --prompt "The capital of France is" --max-new 64 --greedy
  --skip-hashes`, 5 prompt tokens. Hygiene gate is `pgsteal == 0` plus
  readable counters, cgroup limit as requested, swap peak zero, and both
  return codes; **any** reclaim marks the run DIRTY.
- Baseline: none available. EXP-006 recorded that phase 4 cannot be
  measured cleanly here at all: buffered `pread` pulled 7.8 GiB through
  the page cache for a 13-token run, pinned the cgroup at its ceiling and
  forced ~4.82 GiB of reclaim. The phase-4 speed figures this project has
  quoted (1.9-2.5 s/token) are warm, uncgrouped diagnostics.
- Result: **5 of 5 scored runs CLEAN**, `pgsteal 0` on every one.

  | metric | median |
  |---|---:|
  | decode | **1.88 tok/s** |
  | prefill | 1.33 tok/s |
  | model load | 1.32 s |
  | wall | 39.55 s |
  | cgroup `memory.peak` | **2,471.1 MiB** of 3,072 |
  | expert bytes read | 36.9 GiB |

  A first attempt the same day scored 3 of 5 clean; the two DIRTY runs
  recorded `pgsteal` of 2,817 and 2,946 pages (11.0 and 11.5 MiB) and
  coincided with the operator opening a terminal mid-run. A second
  attempt with the machine left alone scored 5 of 5. Generated text was
  identical across clean and dirty runs.
- Verdict: KEEP
- Notes: (1) **This is the project's first number that satisfies rule 2**
  and therefore the first that may be published. It should be quoted as
  "1.88 tok/s decode, cold, inside `memory.max=3G` with swap disabled, on
  a DRAM-less Micron 2400" - the device matters, since EXP-008 measured it
  well below the 3.6 GB/s the original design assumed. (2) **The memory
  contract holds with room to spare**: 2,471.1 MiB peak against a 3,072
  MiB ceiling, 601 MiB unused. That is *lower* than EXP-012's predicted
  2,961 MiB, because this workload reaches only 69 tokens of context so
  the KV cache is a few MiB rather than its 384 MiB reservation at 4K.
  **The dial is therefore not yet validated at full context** - a 4K-context
  run should add roughly 377 MiB, landing near 2,848 MiB, which still fits
  but has not been measured. That measurement is the remaining open item
  on the 11-slots decision. (3) The DIRTY/CLEAN split is evidence the
  hygiene gate works rather than evidence of a problem: it caught operator
  activity that changed timing without changing output, which is exactly
  the contamination rule 2 exists to exclude. Note that identical output
  is not evidence of a clean measurement - reclaim distorts timing, not
  correctness. (4) Decode I/O wait is roughly half of decode wall time in
  the equivalent uncgrouped runs, so I/O and compute are now close to
  balanced; further gains need either a higher hit rate (more slots, which
  the budget does not allow) or a faster device.

## EXP-015: Phase-6 building blocks, landed and unmeasured

- Date / commit: 2026-08-04 / 867461f (`feat/prefill-sweep`); the runtime
  half is bbb0e8d on the same branch.
- Hypothesis: chunked prefill can read each expert **once per layer instead
  of once per token**, and dot each expert's weight rows against every row
  routed to it while those bytes are in L1, without changing a single output
  bit and without taking a byte from the memory contract.
- Method: **nothing was measured.** No cold run, no cgroup, no timing, no
  generated token. The entry exists because CLAUDE.md requires one for every
  performance-motivated change, and because EXP-011 set the precedent for
  logging a structural restructure before its number exists. What gates the
  change is the in-tree unit suite (rule 4, identical output) plus the
  structural argument in Note 2, not a rule-2 run.
- Baseline: EXP-014's cold 3G-cgroup medians (decode 1.88 tok/s, prefill
  1.33 tok/s, `memory.peak` 2,471.1 MiB) are still the last publishable
  numbers, and this branch is not expected to move them, because three of
  the four changes have no caller in the forward pass:

  | change | where | in the running build? |
  |---|---|---|
  | Batched GEMV entry points | `kernels/gemv.rs` | exported, **no caller** |
  | Layer-major prefill sweep over a slot-pool arena | `io/sweep.rs`, `io/stream.rs`, `io/slots.rs` | **no caller** |
  | Position-limited attention (`attention_at`) | `kernels/attention.rs` | **no caller**; `decode_attention` now delegates to the same private body |
  | Six preallocated hot-path `Vec`s | `io/stream.rs` | yes, one-off at `ExpertStream::new` |

- Result:

  **What the sweep is for, as arithmetic.** The byte figures are exact on
  the audited strides in `experts/layout.json` and the reuse counts are
  chunk geometry. Nothing in this table is a measurement:

  | quantity | decode, worst case | 512-token chunk sweep |
  |---|---:|---:|
  | expert bytes per token | ~1,097 MB | ~34 MB (~17.6 GB / 512) |
  | reads of one expert per layer | one per token | one per chunk |
  | rows dotted per weight-row fetch from RAM | 1 | ~32 |

  The ~32 is the same number twice: `512 tokens x top-8 / 128 experts`.
  That is the amortization the sweep and the batched GEMV exist for; read
  granularity is a separate and much weaker claim (Note 5).

  **The ring costs zero bytes**, because it is a borrow of the expert slot
  pool rather than an allocation. The pool is one contiguous 4096-aligned
  slab with every page faulted at construction, it is idle whenever the
  sweep runs (prefill bypasses the decode cache by design, EXP-005; today
  that bypass has no caller, so the pool is only idle in the intended
  arrangement, not in the shipped one), and `pitch == stride` on
  this model because both strides are exact 4096 multiples, so the slab is a
  gapless run of blob-sized buffers. At the shipped dials:

  | layer stride | window, 8 experts | ring, 2 windows in flight |
  |---:|---:|---:|
  | 3,059,712 B (24 layers) | 23.34 MiB | 46.7 MiB |
  | 2,654,208 B (24 layers) | 20.25 MiB | 40.5 MiB |

  All of it is already inside the 1,438.59 MiB expert-pool row of the memory
  contract. The borrow also inherits the alignment and pre-faulting that
  btrfs requires, which is not cosmetic: an un-faulted destination makes
  btrfs complete an O_DIRECT read through the buffered path with no error
  and a full byte count, which is exactly the failure the 3 GB budget cannot
  survive (EXP-009).

  That table is the **per-layer** carve, which is what
  `ExpertStream::sweep_layer` takes: one arena sized for the single layer it
  is about to sweep. The `PrefillSession` path carves once for a whole
  prefill, so `ring_span` sizes its ring for the **widest** layer of the
  model, and it is 46.7 MiB on every layer, never 40.5. Both fit the pool
  row above, so nothing in the memory contract turns on which path runs.

  **The driver's staging comes out of that same carve.** Recorded here
  because it was an open question when the sweep first landed and was settled
  in `4eb5f2a`: `ExpertStream::begin_prefill` opens a `PrefillSession` over
  one span laid out `[scratch | pad | ring]`, the scratch at the slab base
  and the ring at the next 4096 boundary past it, and
  `PrefillSession::split` hands the two
  out as disjoint `&mut`s so a layer-major driver can write a chunk's
  `[n_rows][top_k][hidden]` staging while it consumes swept experts. So the
  staging is another sub-allocation of the pool, not an addition to the
  runtime-anonymous row. The scratch is deliberately not zeroed: taking it is
  address arithmetic over pages the pool already faulted. **No byte figure is
  recorded for it**, because the driver is not written and the chunk size
  that sets it is on the sweep list in Note 4.

  **Harness finding 1: systemd rewrites `${VAR}` and `$$` inside
  `ExecStart=` arguments.** Measured on systemd 261 on the reference
  machine:

  | argument as written | what the process received |
  |---|---|
  | `A ${HOME} B` | `A /home/y0sif B` |
  | `A ${UNSET} B` | `A  B` (the token vanishes) |
  | `A $$VAR B` | `A $VAR B` |
  | `A $VAR B` | unchanged |
  | `%` specifiers, newlines, tabs, quotes, backslashes | unchanged |

  `cold_bench.py` passed the prompt to `systemd-run` this way, so any
  measurement whose prompt contained those sequences was silently truncated
  or rewritten, with `systemd-run` exiting 0 and nothing to notice.
  **No recorded measurement is invalidated**, and that was checked rather
  than assumed: `scratch/ctx4k/p4k.txt` and
  `models/llamacpp-ref/llamacpp_ref/long_00/01/02.txt` contain zero `${`,
  `$$` or `%`, and EXP-014's prompt was `The capital of France is`.
  The harness now passes argv out of band as JSON, and records the sha256 of
  both the prompt file and the delivered text.

  **Harness finding 2: `bitident.py` could report PASS while ignoring most
  of the fingerprint.** An "all" capture compared against a "singles"
  baseline reported **PASS 8/8** and silently skipped the long prompts,
  because `compare` adopted the baseline's prompt set the way it already
  adopts `--top`. A prompt-set mismatch is now **exit 2**.

- Verdict: KEEP (enablers plus two harness fixes; correctness preserved,
  **nothing measured and nothing claimed**)
- Notes:
  1. **Under rule 2 this entry publishes nothing.** There is no cold cgroup
     run behind it. Every figure above is either exact arithmetic on the
     audited strides, a restatement of an earlier entry, or a property of
     the harness measured directly (the systemd table). The measurement this
     work is for is **owed**, and it belongs to the wave-2 prefill driver,
     which is what wires the sweep and the batched GEMV into the forward
     pass. Until that lands, the sweep is code that compiles and is tested,
     not a speedup.
  2. **Bit identity is structural for the batched GEMV, and tested on top.**
     The batched path issues the same `dot()` call on the same
     `(weight_row, activation_row)` bytes as the single-vector path; only
     the loop nesting and the destination index changed, and no weight row
     is dequantized once into scratch and reused (that would break the
     per-super-block accumulation order the kernels fix, and float addition
     is not associative). The six entry points now share one `gemv_impl`
     with the non-batched four passing `n_acts == 1`, so there is one code
     path and nothing to drift, exactly as EXP-011 did for row ranges. The
     tests assert it anyway, bit for bit
     (`k_quant_batched_matches_single_vector_bitwise`,
     `q8_0_batched_matches_single_vector_bitwise`,
     `public_batched_entry_points_match_single_vector`,
     `batched_misaligned_weight_slab_is_bit_identical`). Position-limited
     attention has the same shape: masked positions are *absent* from the
     score buffer, the f64 softmax normalizer and the V sum rather than
     zero-weighted or `-inf`-biased, so the masked form is the unmasked form
     over a shorter cache, and `decode_attention` delegates to the same body
     (`attention_at_is_bit_identical_to_truncated_decode`,
     `decode_attention_matches_attention_at_at_full_length`). No online or
     flash-style rescaled softmax, which would have reordered the reduction.
  3. **The arena borrow has one failure mode worth carrying forward.** A
     sweep window read that can never be reaped may have landed anywhere in
     the arena, so every slot the arena overlaps is retired: the buffer is
     leaked, the layer's cache is rebuilt smaller, and the stream refuses to
     sweep again for the life of the process. It terminates and it never
     aliases, but the arena is carved from the **head** of the slab, so the
     retirements fall on the low layers, and the geometry is exact rather
     than marginal: the 46.7 MiB ring is 48,955,392 B against layer 0's whole
     slot row of 11 x 3,059,712 = 33,656,832 B, and the 15,298,560 B left
     over is exactly 5 slots of layer 1. One unreapable read therefore leaves
     **layer 0 with 0 slots and layer 1 with 6**, both under a `top_k` of 8
     and both dead for the life of the process. Carving from the tail instead
     only moves the damage to layer 47. **The mitigation shipped in
     `4eb5f2a`**, which an earlier revision of this note recorded as still in
     progress: `ArenaOverRetired` refuses a carve that would cover a buffer a
     lost read may still be
     writing into, and `CacheStranded` reports a stranding at the cause,
     naming the first short layer, its remaining slots and `top_k`, in place
     of a `CacheError::TooFewSlots` three decode steps later that names a
     symptom and no cause. Neither is a repair, and none is possible while
     the arena is the pool. `docs/architecture.md` records the same geometry
     and the same two errors under "The prefill arena".
  4. **Two dials ship with defaults nobody has measured**: experts per
     window (8, which divides 128 into 16 uniform windows with no ragged
     tail and lands at 23.34/20.25 MiB, inside the 16-24 MiB range EXP-008
     pointed at) and windows in flight (2, double
     buffering, so window `n + 1` is on the wire while the caller computes
     window `n`). Both are the subject of a planned sweep, alongside the
     chunk-size sweep (128 / 256 / 512 / 1024). 512 is where coverage
     reaches ~100% of a layer's experts, so it is the smallest chunk that
     fully amortizes a sweep; the shorter chunks trade coverage for a
     smaller activation working set and nothing here says which wins.
  5. **The read-granularity motivation for large windows is weaker than it
     looks, and the sweep should not be justified with it.** EXP-008's
     "+51% at 16 MiB" is measured against ~1.35 GB/s at the expert stride,
     and EXP-013 measured **1.97 GB/s** under the real access pattern at
     that same stride, roughly 46% above EXP-008's denominator. EXP-008's
     harness was also never committed. The amortization argument in the
     Result table does not depend on the drive's block-size curve at all,
     which is why it is the one this entry leans on. `scripts/io_probe.py`
     landed on this branch as a rule-2 compliant replacement harness
     (cgroup, proven page-cache eviction, `pgsteal` hygiene gate, and it
     separates granularity from sequentiality by comparing front-to-back
     against a permutation of the same blocks); **no run of it is recorded
     here**, and re-measuring EXP-008 stays open.
  6. **Why the long prompts were adopted as the chunk-seam gate**, for
     context and without overclaiming: the long reference prompts are 512,
     1891 and 3492 tokens, and 512 is exactly one chunk at the default chunk
     size. Nothing in-tree proved prefill correctness at chunked length
     before, and finding 2 meant the fingerprint could pass while covering
     only the eight short singles. Neither fact is a measurement of the
     sweep; they are what makes a future measurement of it trustworthy.
  7. **Forward pointer: the driver landed, so several statements above are
     now stale (added 2026-08-04, EXP-016).** The Baseline table says "no
     caller" of the batched GEMV, the sweep reader and `attention_at`, and
     the Result section records no byte figure for the driver's staging
     "because the driver is not written". All four were true the day this
     entry was written and none of them is true now: `37fcf81` and `32917f2`
     wired the sweep into the forward pass as the default prefill path, and
     the staging span is 80,935,940 B at the v0 dials and a 512-row chunk.
     The same goes for the aside under "The ring costs zero bytes", which
     says the pool is idle only in the intended arrangement and not in the
     shipped one; the shipped arrangement is now the intended one. The entry
     is left as written, because it records what shipped that day; EXP-016
     records what changed. Note 1's claim that the measurement is owed still
     stands, and EXP-016 did not pay it either.

## EXP-016: Chunked layer-major prefill lands as the default path

- Date / commit: 2026-08-04 / 32917f2 (`feat/prefill-sweep`); the runtime
  half is 37fcf81 on the same branch.
- Hypothesis: driving prefill through the layer-major sweep, instead of one
  `forward_token` per prompt token, reads each expert once per layer per
  chunk rather than once per token, **without changing a single output bit**
  and without taking a byte from the memory contract.
- Method: **nothing was measured.** No cold run, no cgroup, no timing, no
  tok/s, no `memory.peak`. The entry exists because CLAUDE.md requires one
  for every performance-motivated change, and because this change is now the
  default path, so what it did to correctness has to be on the record even
  while what it did to speed is not. Two correctness gates were run, one
  against the real model and one in-process; both are under Result. Rule 2
  governs neither, because neither is a throughput number.
- Baseline: EXP-014's cold 3G-cgroup medians (decode 1.88 tok/s, prefill
  1.33 tok/s, `memory.peak` 2,471.1 MiB) are still the last publishable
  numbers, and they describe the token-major prefill this change displaces.
  Unlike EXP-015, this work **is** expected to move the prefill figure.
  Nothing here says by how much, or in which direction the memory peak moves.
- Result:

  **What landed**, and what it replaces in EXP-015's "no caller" table:

  | change | where | on the default path? |
  |---|---|---|
  | Chunked layer-major prefill driver (`PrefillMode::Sweep`, `DEFAULT_PREFILL_CHUNK` 512) | `model/prefill.rs` | yes, the default |
  | Token-major prefill (one `forward_token` per prompt token) | `model/prefill.rs` | retained, selectable via `--prefill token-major` / `RAMVAMP_PREFILL` |
  | Batched GEMV entry points | `kernels/gemv.rs` | yes, per layer per chunk (EXP-015: "no caller") |
  | Sweep reader over the slot-pool arena | `io/sweep.rs`, `io/stream.rs` | yes, once per layer per chunk (EXP-015: "no caller") |
  | Position-limited attention (`attention_at`) | `kernels/attention.rs` | yes, once per row per layer (EXP-015: "no caller") |
  | `generate_from`, `GenerateStats::generated_ids`, `ForwardState::reset` | `generate/mod.rs`, `model/forward.rs` | yes, they are what lets a turn continue without rebuilding the slot pool |
  | `chat` REPL | `crates/cli` | yes |
  | `run_logits` prefilling through `prefill_prompt` | `crates/cli` | yes, and this is what puts `scripts/bitident.py` on the swept path at all |

  **Gate 1, the fingerprint against llama.cpp.**
  `python3 scripts/bitident.py compare models/llamacpp-ref/phase4-baseline
  --ramvamp target/release/ramvamp --rvmp models/qwen3.rvmp`:

  | field | value |
  |---|---|
  | result | **PASS, 8/8 byte-identical** |
  | current build | `32917f25fd074fb51f6d0d8eaf2d85b984c67630` |
  | baseline | `650b5ea7...-dirty`, captured 2026-08-03 |
  | `top` | 4096 |
  | prompt set | `singles` |
  | binary | DIFFERENT |

  **The honest caveat on gate 1: those 8 prompts are 4 to 12 tokens long.**
  Every one of them fits inside a single 512-token chunk, so not one crosses
  a chunk seam. What the gate proves is that a *one-chunk* sweep reproduces
  the token-major logits on the real model at real geometry, which is worth
  having and is not nothing. It proves nothing about the multi-chunk path.

  **Gate 2, the in-process A/B.**
  `sweep_and_token_major_agree_bit_for_bit` runs both paths through
  `prefill_prompt` and compares the final logits at `to_bits()` equality
  across 8 prompt lengths (1, 2, 3, 4, 6, 7, 8, 12) times 6 chunk sizes
  (1, 2, 3, 4, 8, 512), which covers exact multiples of the chunk, ragged
  final chunks, and prompts shorter than one chunk.
  `both_paths_leave_the_same_kv_cache` asserts the same for the stored f16
  K/V on every layer, and `decode_continues_off_a_swept_prefill` asserts
  decode picks up cleanly afterwards. All of it runs on the small synthetic
  fixture, not on the model.

  **The coverage gap between the two gates**, which an adversarial review
  identified and which matters:

  | dial | unit fixture | Qwen3 v0 |
  |---|---:|---:|
  | hidden | 256 | 2,048 |
  | `q_dim` | 256, equal to hidden | 4,096, not equal to hidden |
  | Q8_K blocks per hidden-width activation row | 1 | 8 |
  | experts per sweep window | 1 | 8 (the shipped default) |
  | chunks per gated prompt | 1 to 12 (gate 2) | 1 (gate 1) |

  So gate 2 covers multi-chunk at toy geometry and gate 1 covers real
  geometry at one chunk. **Multi-chunk at production dials on the real
  model's geometry was covered by neither.**

  **Correction (2026-08-04, found by the final review): the fixture half of
  that gap was already closed when this entry was written**, in commit
  `09ab935`, two commits earlier. The wide fixture separates every width
  (`hidden` 512, `q_dim` 768, `kv_dim` 192, `moe` 256, pairwise distinct and
  asserted by loop), gives 2 Q8_K blocks per hidden-width row, and runs at
  `experts_per_window` 1, 4 and 8 with `windows_in_flight` up to 3, over 6
  prompt lengths x 4 chunk sizes: 96 sweep runs against 6 token-major
  baselines. `plan_arena` narrows only `rows` and `windows_in_flight`, never
  `experts_per_window`, so those runs really do use the dials they name, and
  that is pinned by its own test.

  What remains outstanding is only the real-model half, which a warm A/B has
  since paid at 512 tokens: sweep prefill is byte-identical to token-major at
  chunk 128, 256 and 512, i.e. across 4, 2 and 1 chunks, at the shipped
  geometry and dials (EXP-017 Result). A cold long-prompt baseline over the
  512, 1891 and 3492 token reference prompts (EXP-015 Note 6) would still be
  stronger, and is not run.

  **Memory: prefill still costs zero additional bytes.** Both spans are
  sub-allocations of the `PrefillSession` arena, which is the head of the
  idle expert slot pool:

  | span | bytes | MiB |
  |---|---:|---:|
  | chunk scratch, 512 rows at the v0 dims | 80,935,940 | 77.19 |
  | sweep ring, 8 experts per window x 2 windows in flight | 48,955,392 | 46.69 |

  Both sit inside the 1,438.59 MiB expert-pool row of the memory contract,
  and the arithmetic is pinned by a test rather than asserted in prose: it
  checks the 512-row scratch to the byte, checks that `scratch_bytes` is
  affine in rows so no term is quietly quadratic, and checks that scratch
  plus ring fits under 1,438 MiB. The 3 GB budget therefore does not move on
  account of this change, which is a property of the arena borrow (EXP-015)
  rather than a new result.

- Verdict: KEEP (correctness gates pass at the coverage described above;
  **nothing measured and no speed claimed**)
- Notes:
  1. **Under rule 2 this entry publishes nothing.** There is no cold cgroup
     run behind it. Every figure above is either a gate result, exact
     arithmetic on the audited dims, or a restatement of an earlier entry.
     The measurement the whole phase exists for is still **owed**, and it is
     now owed by EXP-017: prefill throughput and `memory.peak` for the swept
     path, cold inside `memory.max=3G`, against the token-major path on the
     same prompt. EXP-015 said the measurement belonged to the driver's
     entry. The driver has an entry now and it is this one, and it does not
     have the number.

     **Correction (2026-08-04, on writing EXP-017): EXP-017 did not pay this
     debt either.** It is a warm, uncgrouped phase split, so it publishes
     nothing under rule 2 and carries no `memory.peak`. What it did do is
     redirect the work: attention, not I/O and not the expert GEMV, is the
     binding term. The cold measurement is best taken after the attention
     work rather than before it, and is now tracked in the backlog rather
     than assigned to a numbered successor.
  2. **Attention is the deferred half, and it bears directly on that first
     measurement.** The driver batches the projections and the expert FFN
     across a chunk's rows, but it runs `attention_at` row by row on the
     calling thread through one shared score buffer, so attention is neither
     batched nor parallel. The work is quadratic in prompt length, so at a 4K
     prompt it is a substantial term sitting outside the compute pool
     entirely, and it is a plausible reason for a first measurement to land
     below what the I/O arithmetic alone suggests. Parallelizing over rows is
     bit-neutral, because rows are independent and `shard_range` is a pure
     function, but it needs one score buffer per shard and the arena carve
     does not have one. **Measure before building it**, so that the
     measurement decides rather than the intuition.
  3. **Trace records are reordered back into position order by the CLI, and
     that is deliberate.** The sweep produces routing layer-major, but
     `scripts/lfu_sim.py` replays trace records sequentially into a simulated
     cache, so record order is what determines its hits and its evictions.
     Emitting layer-major would have silently changed what every past
     simulation measured, including the policy decisions in EXP-005, while
     still producing a file that parses. A `RouteRecorder` buffers a chunk
     and hands the writer whole records in position order, and the resulting
     file is asserted byte-identical to what token-major wrote.

     **Correction (2026-08-04, found by the final review): this note
     originally claimed records leave the buffer as each position completes,
     so a run dying mid-prefill still left every finished record readable.
     That is false under the sweep**, and the code comment added in the same
     commit says so. No position in a chunk completes until that chunk's last
     layer, so the granularity is a chunk: a run that dies at layer 30 of 48
     loses the whole chunk in progress. What does leave the buffer goes into
     `TraceWriter`'s `BufWriter`, which nothing flushes before `finish`, so
     the tail of earlier chunks can go with it. A truncated trace is still
     readable, by counting records from the file length and ignoring a
     trailing partial one.
  4. **Keeping the token-major path is not sentiment.** It is the reference
     half of gate 2; deleting it deletes the only test that covers the
     multi-chunk seams at all, and it is also the A/B arm EXP-017 needs to
     express its result as a ratio on one machine in one session (rule 3).
     Both dials are `Option`s rather than clap defaults, so a flag left unset
     does not silently override the `RAMVAMP_PREFILL` environment variables
     the state seeds itself from.

## EXP-017: The prefill phase split, and attention is the wall

- Date / commit: 2026-08-04 / on top of 4254279 (`feat/prefill-sweep`,
  pre-commit)
- Hypothesis: EXP-016 left prefill hard compute-bound and named two candidates
  for where the time goes, attention and the batched expert GEMV, with no way
  to tell them apart from the counters that existed. The estimate made before
  measuring, on a 512-token prompt, was **attention 26-52 s against expert
  GEMV 70-96 s**, so the GEMV was expected to be the larger lever and phase
  7's target. This entry measures the split instead of estimating it.
- Method: three runs of `target/release/ramvamp logits --model
  models/qwen3.rvmp --prompt "$(cat <fixture>)" --top 1 --skip-hashes` on the
  reference machine (`docs/benchmark-machine.md`), with the prefill dials
  varied: token-major at 512 tokens, sweep at 512, sweep at 1891. Fixtures are
  `models/llamacpp-ref/llamacpp_ref/long_00.txt` (512 tokens) and
  `long_01.txt` (1891 tokens).

  **Rule-2 status, stated head-on: these are NOT publishable numbers.** Warm
  page cache, no cgroup, and the machine was not quiet (a 50 minute regression
  gate had just finished and the operator was using the machine). They are a
  diagnostic split, not a throughput measurement. What survives rule 2 is the
  *shape* of the split and the ratios within a single run, not the absolute
  seconds. The cold rule-2 prefill measurement is still owed; Note 7 says so
  in the place EXP-016 promised it would be paid.

  How the split is charged: `PrefillTiming`, added in this phase, is filled by
  a `PhaseClock` that charges at region boundaries rather than opening and
  closing a pair per region, so the five phases are disjoint spans of the
  total by construction, they sum to it, and whatever no phase claims stays
  visible as `other` instead of being folded into whichever region happened to
  be open. Instrumentation overhead is bounded at roughly **2.5 ppm** for a
  512-token sweep chunk, because per-row sites are timed around the enclosing
  loop rather than per iteration. Counting the charge sites gives `16 + 2R`
  per layer, so at `R = 128` routed experts that is `48 x 272` plus two for
  the logits tail and one for the clock's construction, i.e. **13,059** reads
  per chunk at roughly 25 ns, or 2.6 ppm of a 124.27 s run. An earlier
  revision of this entry said 13,104, which is one extra read per layer; the
  conclusion is unchanged either way.
- Baseline: the token-major run from the same session at the same 512 tokens,
  which is the path EXP-016 displaced and which it deliberately kept
  selectable for exactly this comparison (EXP-016 Note 4). Both arms are in
  the table below, so every 512-token ratio quoted here is a within-session
  comparison and rule 3 is satisfied. EXP-014's cold 3G-cgroup medians remain
  the last publishable numbers this project has, and nothing here replaces
  them.
- Result:

  **The phase split**, seconds and share of that run's own total:

  | phase | token-major 512 | sweep 512 | sweep 1891 |
  | --- | ---: | ---: | ---: |
  | attention | 82.38 s (25.9%) | 76.23 s (61.3%) | 1050.90 s (85.2%) |
  | expert compute | 77.81 s (24.5%) | 23.78 s (19.1%) | 92.27 s (7.5%) |
  | expert io | 114.88 s (36.2%) | 2.10 s (1.7%) | 6.68 s (0.5%) |
  | projections | 36.65 s (11.5%) | 15.51 s (12.5%) | 59.85 s (4.8%) |
  | elementwise | 6.03 s (1.9%) | 6.64 s (5.3%) | 24.36 s (2.0%) |
  | other | 0.00 s | 0.00 s | 0.00 s |
  | **total** | **317.75 s** | **124.27 s** | **1234.05 s** |

  `other` is what no phase claimed: the arena plan, the session open and
  close, the per-block scratch carves and the chunk loop's own scaffolding.
  It came out at zero to the printed resolution on all three runs, so none of
  the prefill wall time is unattributed and the five phases can be read as the
  whole of it. That the remainder is *visible* rather than folded into
  whichever region happened to be open is the point of charging at boundaries;
  that it is zero is the result.

  Of the `expert io` figures, the **drive-blocked** subset was 113.76 s,
  1.76 s and 5.06 s respectively. That subset is the streamer's own counter,
  not a second measurement of the same interval; the remainder is the sweep's
  window bookkeeping.

  **A separate byte-comparison run** (same binary, same 512-token prompt)
  recorded sweep prefill **byte-identical to token-major at chunk sizes 128,
  256 and 512**, that is across 4, 2 and 1 chunks:

  | chunk size | chunks | windows (skipped) | wall |
  | ---: | ---: | ---: | ---: |
  | 128 | 4 | 3,072 (14) | 161 s |
  | 256 | 2 | 1,536 (3) | 143 s |
  | 512 | 1 | 768 (0) | 131 s |
  | token-major | n/a | n/a | 321 s |

  That run and the split run are **separate invocations**, so their absolute
  seconds must not be combined into one curve (rule 3). The 512-token sweep
  appears in both, at 131 s there and 124.27 s in the split table, which is
  the size of the run-to-run spread on this machine in this state and is
  another reason to read shares rather than seconds.
- Verdict: NEUTRAL (a diagnostic that redirects phase 7; no change ships from
  it, and under rule 2 it publishes nothing)
- Notes:
  1. **Comparing like for like at 512 tokens, phase 6 reduced everything it
     touched by 4.9x**: 235.37 s of non-attention work token-major against
     48.04 s swept. Per phase, expert io fell **55x** (114.88 to 2.10),
     expert compute **3.27x** (77.81 to 23.78) and projections **2.36x**
     (36.65 to 15.51). Elementwise is the exception and did not fall at all:
     6.03 s to 6.64 s, roughly 10% higher. It is the one phase the
     restructure gave nothing to, it is small either way, and this entry does
     not have the evidence to say whether the 10% is a real cost of the
     layer-major shape or the run-to-run spread the two 512-token sweep wall
     times already show.
  2. **End to end that shows up as only 2.45x**, because attention was
     untouched and is now the dominant term: 61.3% of a 512-token prefill and
     85.2% of a 1891-token one. (2.45x is 321 s to 131 s within the
     byte-comparison run; the split run's own totals give 2.56x, 317.75 s to
     124.27 s. Both are ratios inside one invocation, and the two invocations
     are not put on one curve.) Amdahl, measured: 4.9x on 74% of the work
     buys about 2.5x overall, and the remaining 26% is now 61%.
  3. **Attention's total cost is quadratic in prompt length, and the two
     lengths confirm it.** Per-token attention is **148.9 ms at 512 tokens**
     (76.23 s / 512) and **555.7 ms at 1891** (1050.90 s / 1891), a ratio of
     **3.73** against a prompt-length ratio of **3.69**. Per-token cost
     scaling linearly with length is exactly what a quadratic total looks
     like measured per token, and 3.73 against 3.69 is as clean a
     confirmation as two points can give. No other phase behaves this way:
     expert compute is 46.4 ms/token at 512 and 48.8 at 1891, essentially
     flat, which is the linear term the amortization argument predicts.
  4. **Attention runs single-threaded on the decode thread with one shared
     scratch**, so it gets nothing from the six pinned P-cores, and it
     converts K and V from f16 to f32 per element with no vectorization
     (`kernels/attention.rs` stores the cache as f16 bits and converts on the
     fly rather than keeping a dequantized plane). The implied rate is
     roughly **0.68 GFLOP/s** against a rough **51.6 GFLOP estimate** for the
     512-token case. The GFLOP figure is an **estimate** and is labelled one;
     the 76.23 s it is divided by is measured. A sub-GFLOP/s rate on a
     6-P-core AVX2 machine is the shape of the finding, and it holds even if
     the estimate is off by a factor of two in either direction. It is not a
     target, and no figure here says what the rate would become.
  5. **This corrects a prediction made before the measurement, and rule 5 is
     why it is written down.** The estimate in the Hypothesis put attention
     at 26-52 s and expert GEMV at 70-96 s on this prompt and judged the GEMV
     the larger lever. Measured, attention is **76.23 s** and expert compute
     is **23.78 s**: both estimates were wrong, in opposite directions, and
     the ordering they implied was backwards. The specific hypothesis that
     the batched GEMV was thrashing on the activation side (~75 KB streamed
     past every weight row) is **refuted**: the batched path delivers 3.27x
     against the unbatched one on the same prompt in the same session. Rule 5
     says negative results get entries because they stop bad ideas coming
     back; this one would otherwise have sent phase 7 at the wrong target and
     spent the phase optimizing a term worth 19% of prefill.
  6. **Phase 7's target is attention.** Two levers, in the order the
     measurement suggests. Parallelizing over rows is bit-neutral, since rows
     are independent and each row's softmax and V sum are self-contained, but
     it needs **one score buffer per shard** and the arena carve does not
     have one today (EXP-016 Note 2 already named this as the blocker).
     Vectorizing the f16 conversion and the dot is a second and independent
     lever, and it is subject to rule 4 like every other kernel change: a
     reordered reduction needs tolerance tests, an unreordered one needs bit
     identity. **No speedup figure is predicted here.** The headroom looks
     large and the measurement is owed, which is the same discipline Note 5
     exists to enforce.
  7. **What is still owed, and this entry does not pay it.** EXP-016 Note 1
     named EXP-017 as the entry that would produce prefill throughput and
     `memory.peak` for the swept path, cold inside `memory.max=3G`, against
     the token-major path on the same prompt. This is EXP-017 and it does
     not have that number either: every run above is warm, uncgrouped and on
     a busy machine, and the memory peak was not sampled at all. What it does
     instead is tell phase 7 where to aim, which EXP-016 Note 2 asked for in
     those words ("measure before building it"). The cold cgroup prefill
     measurement stays on the backlog in `docs/architecture.md`, and it
     should be taken after the attention work rather than before it, since
     the number it would produce today describes a term phase 7 is about to
     move.
