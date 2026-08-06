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
- [EXP-018: Cold paired prefill, the swept path against the token-major path at 512 tokens](#exp-018-cold-paired-prefill-the-swept-path-against-the-token-major-path-at-512-tokens) — KEEP
- [EXP-019: O_DIRECT bandwidth under rule 2, and EXP-008 refuted](#exp-019-o_direct-bandwidth-under-rule-2-and-exp-008-refuted) — NEUTRAL
- [EXP-020: The attention kernel rebuilt in three waves, measured warm](#exp-020-the-attention-kernel-rebuilt-in-three-waves-measured-warm) — NEUTRAL
- [EXP-021: Phase 7 measured cold: prefill, decode and 4K context under rule 2](#exp-021-phase-7-measured-cold-prefill-decode-and-4k-context-under-rule-2) — KEEP
- [EXP-022: T_BLOCK 4 to 8 with a stepped position tail, measured warm](#exp-022-t_block-4-to-8-with-a-stepped-position-tail-measured-warm) — KEEP
- [EXP-023: The cold decode sweep: the phase split against context, the 11-slot hit rate, the slot dial and T_BLOCK](#exp-023-the-cold-decode-sweep-the-phase-split-against-context-the-11-slot-hit-rate-the-slot-dial-and-t_block) — KEEP

Entries EXP-007 through EXP-013 were measured on a machine that was not
quiet, and most are microbenchmarks rather than end-to-end runs. Under rule 2
none of their numbers is publishable; they are recorded so that the design
decisions they drove are traceable, and each states the re-measurement it
needs. EXP-013 is the exception worth naming: its **numerics** result
(byte-identical logits) is a correctness measurement that rule 2 does not
govern and that does stand as reported; only its throughput figures are
provisional. EXP-008's re-measurement has since been taken: EXP-019 redoes
that characterisation under rule 2 with a committed harness and refutes the
block-size premise EXP-008 handed the design. EXP-008's own numbers stay where
they are, as the record of what was believed when phase 5 was designed.

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
- **Superseded on the throughput question (2026-08-04, EXP-019).** The
  re-measurement this entry's Notes asked for has been taken, under rule 2,
  with a committed harness (`scripts/io_probe.py`), on the real installed
  layer files. It does not agree with series (a), and the disagreement is not
  only in level:
  - **The block-size curve is refuted.** EXP-019 finds bigger blocks neutral
    on one probed file and 15 to 16 percent *worse* on the other three when
    the read is sequential. The "+51% at 16 MiB" that this entry handed the
    phase-6 window dial does not survive under either reading of what series
    (a)'s block sweep did, sequential or random.
  - **The absolute level here is too low.** EXP-019 measures 1.54 to 2.37
    GB/s across its whole matrix, above this entry's 1.211 to 1.390 and
    consistent with EXP-013's ~1.97 under the real access pattern. That is
    what this entry's own Method predicted a quiet machine would show, so it
    is a confirmation of the caveat rather than a surprise.
  - **The shape survives, restated.** Queue depth alone was never the
    variable: EXP-019 finds that throughput turns on total *bytes in flight*,
    which both queue depth and block size move, and that the drive holds its
    peak up to roughly 100 MB outstanding and loses 15 to 18 percent past it.
    "QD4-8 rather than deeper" is still the right dial at the block sizes the
    runtime uses, but for that reason and not this one.
  - **The 1.59 GB/s constant stays unsourced.** EXP-019 does not find it
    either, and nothing below should be read as sourcing it.

  This entry's Method and Result are left exactly as written. They record what
  was measured and believed when phase 5 was designed, and the corrections
  belong next to them rather than inside them.
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

## EXP-018: Cold paired prefill, the swept path against the token-major path at 512 tokens

- Date / commit: 2026-08-04 / phase-6 side `7201d6b` (`feat/prefill-sweep`),
  phase-5 side `c3572fd` (the phase-5 merge on `main`), banked as a binary at
  `scratch/phase5-ref/ramvamp` with its commit and sha256 recorded beside it.
- Hypothesis: the chunked layer-major prefill that landed in EXP-016 beats the
  token-major path it displaced, measured cold inside `memory.max=3G` on the
  same prompt in the same session, and the memory contract survives a
  512-token prompt.
- Method: `scripts/cold_bench.py --prompt-file
  models/llamacpp-ref/llamacpp_ref/long_00.txt --max-new 4 --warmup 0
  --repeats 1`, run twice back to back in one session on the reference
  machine (`docs/benchmark-machine.md`), once against the banked phase-5
  binary and once against the phase-6 release build. Per run the harness
  evicts all 53 model files with `posix_fadvise(POSIX_FADV_DONTNEED)` and
  proves the eviction with `mincore`, launches under `systemd-run --user
  --wait -p MemoryMax=3G -p MemorySwapMax=0 -p MemoryAccounting=yes`, and
  reads `memory.peak`, `memory.events`, `memory.stat` and `/proc/self/io`
  `read_bytes` from inside the cgroup before exit. Both runs report
  **hygiene PASS**: `pgsteal` 0, every `memory.events` counter 0,
  `memory.swap.peak` 0, both return codes 0. The prompt goes to `systemd-run`
  out of band as JSON (EXP-015 harness finding 1) and both runs record the
  same delivered sha256 `d1b6c407c55a...` over the same 2,002 bytes, so this
  is one workload measured twice: every ratio below is paired, on one machine,
  in one session, and rule 3 is satisfied. Raw results are
  `scratch/cold-bench/p5-512.json` and `p6-512.json`.

  Two provenance facts worth stating rather than assuming. The phase-6 build
  commit is **not** recorded in the JSON; it is inferred from
  `target/release/ramvamp`'s mtime (19:30:23) landing 31 s before `7201d6b`
  (19:30:54), so the binary is that commit's tree and nothing stronger says
  so. And `--repeats 1` means each side is a **single run**, not EXP-014's
  median of five, so run-to-run spread is unbounded here; the JSON's `median`
  block is that one run. EXP-017 saw roughly 5% between two 512-token sweeps
  on this machine warm, which is far below the prefill ratio and is the same
  size as the load-time and `memory.peak` differences below, so read those two
  rows as "unchanged" rather than as differences.
- Baseline: the phase-5 arm of the same pair, which is the token-major prefill
  EXP-016 displaced and deliberately kept selectable for exactly this
  comparison (EXP-016 Note 4). EXP-014's cold medians (decode 1.88 tok/s,
  prefill 1.33 tok/s, `memory.peak` 2,471.1 MiB) are the last published
  numbers before this entry, but they were taken at a 5-token prompt with 64
  decode tokens. That is a different workload, so they are context and not the
  baseline.
- Result:

  | metric | phase 5 | phase 6 | ratio |
  | --- | ---: | ---: | ---: |
  | prefill | 1.65 tok/s | **4.23 tok/s** | **2.56x** |
  | prefill time | 310.98 s | 120.96 s | 2.57x |
  | decode | 1.88 tok/s | 1.38 tok/s | 0.73x |
  | decode time (4 tokens) | 2.12 s | 2.89 s | |
  | model load | 1.31 s | 1.32 s | 0.99x |
  | wall | 315.05 s | 125.86 s | 2.50x |
  | cgroup `memory.peak` | 2,576.0 MiB | 2,570.1 MiB | |
  | process `read_bytes` | 239.33 GB | 20.72 GB | **11.55x fewer** |

  **Headline: prefill 2.56x, cold, inside `memory.max=3G` with swap off,
  hygiene PASS on both arms.** This is the measurement EXP-015 Note 1,
  EXP-016 Note 1 and EXP-017 Note 7 each recorded as owed, and it discharges
  that debt. It is also the **first paired A/B this project has that satisfies
  rule 2**: EXP-014 was a single arm with no baseline available to it at all.

  **The bytes moved are the mechanism, and they are the larger ratio.** Phase 6
  read **20.72 GB** against an installed model of 18,626,213,888 B, that is
  **1.11x the whole model**, which is the exact signature of a sweep that
  reads each expert once: 17,553,162,240 B of expert files plus 1,073,051,648
  B of mmap-faulted common weights is 18.63 GB of it, leaving ~2.1 GB for four
  decode tokens. Phase 5 read **239.33 GB**, 12.85x the model, or 465 MB of
  expert bytes per prompt token against the ~1,097 MB/token decode worst case,
  which implies a token-major prefill hit rate near 58% if every miss reads
  one whole blob. EXP-013 measured 45.3% on a 25-token prompt and a 512-token
  prompt should sit higher, so the two agree in direction; they were not
  measured the same way and no claim here rests on the 58%.

  **Bytes fell 11.55x and prefill time fell 2.57x**, which is the same
  statement as EXP-017's phase split from the other side: once the sweep has
  taken expert I/O out of the critical path, prefill is bounded by attention
  and the remaining bytes buy progressively less. Nothing in this entry
  measures the split; EXP-017 did, warm.

  **Decode fell, from 1.88 to 1.38 tok/s, and that is not smoothed over.**
  See Note 1: the most likely cause is a cold-start transient that the
  workload is too short to escape, it is a hypothesis and not a measurement,
  and it needs its own run.

  **Memory: both arms fit with about 500 MiB spare**, 2,576.0 MiB and 2,570.1
  MiB against the 3,072 MiB ceiling, with zero reclaim on either. The swept
  path is 5.9 MiB *below* the token-major path despite carrying an 80,935,940
  B chunk scratch and a 48,955,392 B sweep ring, which is the arena borrow
  behaving as EXP-015 and EXP-016 said it would: both spans are
  sub-allocations of the already-faulted expert slot pool, so they cost the
  contract nothing.
- Verdict: KEEP
- Notes:
  1. **The decode regression is a cold-start transient, as a hypothesis that
     needs its own measurement.** The swept prefill bypasses the expert cache
     by design (EXP-005 closed prompt replay as worth +0.09 points) and
     `crates/core/src/io/stream.rs` documents that taking the prefill arena
     **invalidates every layer's slot occupancy**, because the arena is carved
     out of the slot pool itself. So phase-6 decode begins with an empty
     cache. Phase 5's token-major prefill did the opposite: it ran 512 tokens
     of routing through the decode cache and left it warm. At `--max-new 4`
     almost everything measured is that difference. Three things support the
     reading and none of them settles it:
     - EXP-005 measured cold start reaching within 2 points of steady state
       only **by token 48**, and an infinite-cache ceiling of 97.7% after 32
       tokens. Four tokens is entirely inside the transient.
     - The arithmetic lands where an empty cache predicts. Phase-6 decode is
       725 ms/token; the no-cache row of the performance model re-derived at
       EXP-019's bandwidths is 649 to 708 ms/token of I/O alone. A measurement
       sitting just under an I/O-only ceiling is what a decode step doing
       nothing but cold misses looks like. That combines two entries, so it is
       a derivation and not a measurement; the two batches were taken minutes
       apart in one session on one machine, which is the strongest rule-3
       footing a cross-entry derivation can have and is still not a paired
       measurement.
     - Steady-state decode code is unchanged between the two arms and the
       cache, its policy and its dial are untouched, so this is **not**
       evidence of a throughput regression.

     It is still a real user-visible effect on the first tokens after a
     prompt, and it is a genuine cost of the sweep rather than an artifact of
     the harness. `docs/architecture.md` predicted it in one line ("Decode
     cache starts cold after prefill; acceptable, first tokens warm it"), and
     EXP-005 closed prompt-replay-into-cache as worth +0.09 points, but
     EXP-005 measured hit **rate** over 556 decode tokens and says nothing
     about the size of the transient at token 1. So neither settles it. **What
     would settle it is the same paired run at a longer `--max-new`** on the
     same prompt, where the transient is amortized and steady state is
     visible; that is recorded as owed in the backlog in
     `docs/architecture.md`.
  2. **EXP-014's 1.88 tok/s and this entry's 1.38 are not comparable, and the
     coincidence that phase 5 also reports 1.88 here is not evidence of
     anything.** EXP-014 measured 64 decode tokens after a 5-token prefill;
     this measures 4 decode tokens after a 512-token prefill. Different cache
     state, different context length, different token count. Rule 3 forbids
     putting them on one curve, and the phase-5 arm above is the only
     baseline this entry's decode ratio may be read against.
  3. **`memory.peak` is recorded, and the 4K question is still open.** Both
     arms carry ~2,570 to 2,576 MiB at 516 tokens of context. That is 99 to
     105 MiB above EXP-014's 2,471.1 MiB at 69 tokens, while the FP16 KV
     arithmetic (96 KiB per token across 48 layers) accounts for only ~42 MiB
     of it; the residual is **not explained here**, and since the two numbers
     come from different entries and different builds they must not be
     subtracted as though they were one series anyway. What this entry does
     settle is that the contract holds at 512 tokens with ~500 MiB spare.
     What it does not settle is 4K: EXP-014 named a 4K-context run as the
     remaining open item on the 11-slot dial, and this is not that run. From
     this starting point the same KV arithmetic puts a 4K run near 2,906 MiB,
     which still fits, and is still an extrapolation rather than a
     measurement.
  4. **Load time is unmoved**, 1.31 s against 1.32 s, which is the control
     this pair happens to carry: the two binaries mmap the same common
     weights the same way, so a ratio near 1.00 on a phase neither change
     touches is weak evidence that the two arms saw the same machine.
  5. **`--repeats 1` is the weakest part of this entry.** EXP-014 used five
     scored runs and a median for exactly this reason. The prefill ratio is
     2.56x against a run-to-run spread that EXP-017 put at roughly 5% warm,
     so the sign and the order of magnitude are safe; the third significant
     figure is not. A repeat of this pair at `--repeats 5` would cost about 40
     minutes and is worth taking before the number is quoted anywhere that
     matters.

## EXP-019: O_DIRECT bandwidth under rule 2, and EXP-008 refuted

- Date / commit: 2026-08-04 / `7201d6b` (`feat/prefill-sweep`), harness
  `scripts/io_probe.py`
- Hypothesis: EXP-008's characterisation of the reference drive is wrong in
  level because the machine was contended, and its "+51% at 16 MiB over the
  expert stride" conflates two variables, block size and read order, that can
  be separated at matched bytes. EXP-008 is the highest-value item on the
  backlog and cannot be re-run, because its harness was never committed.
- Method: `scripts/io_probe.py --repeats 3 --warmup 1`, one warmup run
  discarded and three scored, on the reference machine
  (`docs/benchmark-machine.md`), started 18:13:36 UTC and finishing
  immediately before EXP-018's two runs on the same quiet machine (result
  files timestamped 21:14:56, 21:20:12 and 21:22:20 local). That adjacency is
  why the one derivation in EXP-018 that crosses the two entries is defensible
  at all. Raw results are `scratch/io-probe/summary.json` and
  `scratch/io-probe/tables.md`.

  **Queue depth is emulated with N OS threads each issuing blocking `preadv`,
  not io_uring.** That bounds what this entry licenses: it characterises the
  drive and the filesystem, not `crates/core/src/io`'s submission path, which
  is io_uring with `SINGLE_ISSUER` and `DEFER_TASKRUN`. Every table below
  carries `queue = threaded-pread` in the JSON for that reason. A stdlib-only
  script cannot drive io_uring, and the repo's scripts staying stdlib-only was
  judged worth the label.

  What is measured: four **real installed** layer files, never a freshly
  written contiguous scratch file, spanning both stride classes (`layer_00`
  and `layer_20` at 3,059,712 B, `layer_06` and `layer_21` at 2,654,208 B).
  A block is `K` consecutive expert blobs read by one `preadv`, so every read
  lands on an exact blob boundary and every offset is a 4096 multiple. `seq`
  is a front-to-back pass; `rand` is a uniform permutation of **the same
  blocks and the same bytes**, which is what separates granularity from
  sequentiality. Two sweeps: block size `K` in 1, 2, 4, 6, 8 at QD 8, and
  queue depth in 1, 2, 4, 8, 16 at K 8. GB/s is 10^9 B/s, matching EXP-008's
  units.

  Hygiene, verified rather than assumed, all four runs **CLEAN**: `pgsteal` 0
  and every `memory.events` counter 0 inside a `memory.max=3G`,
  `memory.swap.max=0` cgroup; every probed file proven evicted by
  `mmap` + `mincore` before each run and proven **zero resident** after, with a
  positive control (one buffered 4 KiB read must then show as resident) so the
  zero means something; per-thread buffers, page-aligned and pre-faulted
  before the timer starts, never a shared pool indexed by arithmetic
  (EXP-007); `btrfs device stats` sampled around every run with
  `corruption_errs` unchanged at its 138,407 baseline and all four other
  counters at 0; zero short reads and zero errors across all 288 timed cases.
  `filefrag` reports **zero compressed extents** on all four files, closing
  one of the documented silent O_DIRECT fallback paths.
- Baseline: EXP-008 series (a). This is characterisation, so there is no
  change being gated, but unlike EXP-008 this entry does have a prior it
  contradicts.
- Result, medians of 3 scored runs. Run-to-run spread was 1.4% at the median
  case and 8.5% at the worst, so differences below ~3% are noise.

  **Block-size sweep at QD 8**

  | K | block MiB | layer_00 | layer_20 | layer_06 | layer_21 |
  |---|---|---:|---:|---:|---:|
  | 1 | 2.92 / 2.53 | 1.578 | 2.271 | 2.368 | 2.317 |
  | 2 | 5.84 / 5.06 | 1.579 | 2.308 | 2.369 | 2.349 |
  | 4 | 11.67 / 10.12 | 1.564 | 2.287 | 2.345 | 2.349 |
  | 6 | 17.51 / 15.19 | 1.570 | 2.002 | 2.095 | 2.081 |
  | 8 | 23.34 / 20.25 | 1.553 | 1.924 | 1.984 | 1.951 |

  Sequential GB/s. Paired values like `2.92 / 2.53` are the 3,059,712 B stride
  class and the 2,654,208 B stride class, in that order, here and below.

  **Queue-depth sweep at K 8**, sequential GB/s

  | QD | bytes in flight | layer_00 | layer_20 | layer_06 | layer_21 |
  |---|---|---:|---:|---:|---:|
  | 1 | 24.5 / 21.2 MB | 1.551 | 2.166 | 2.199 | 2.188 |
  | 2 | 49.0 / 42.5 MB | 1.597 | 2.294 | 2.371 | 2.321 |
  | 4 | 97.9 / 84.9 MB | 1.585 | 2.290 | 2.362 | 2.322 |
  | 8 | 195.8 / 169.9 MB | 1.553 | 1.924 | 1.984 | 1.951 |
  | 16 | 391.6 / 339.7 MB | 1.542 | 1.940 | 1.953 | 1.955 |

  **Headline decomposition at QD 8**, what a front-to-back K=8 prefill read
  buys over the runtime's random single-expert read, split into the part from
  bigger blocks and the part from order

  | file | rand K=1 | rand K=8 | seq K=8 | granularity | sequentiality | combined |
  |---|---:|---:|---:|---:|---:|---:|
  | layer_00 | 1.594 | 1.543 | 1.553 | 0.97x | 1.01x | 0.97x |
  | layer_20 | 1.688 | 1.948 | 1.924 | 1.15x | 0.99x | 1.14x |
  | layer_06 | 1.627 | 1.967 | 1.984 | 1.21x | 1.01x | 1.22x |
  | layer_21 | 1.550 | 1.942 | 1.951 | 1.25x | 1.00x | 1.26x |

  **Extent geometry** (`filefrag -v`); the installer writes each projection
  slab as its own CoW extent, so a "sequential" read is physically scattered

  | file | bytes | extents | mean extent B | adjacent pairs | compressed |
  |---|---:|---:|---:|---:|---:|
  | layer_00 | 391,643,136 | 398 | 984,027 | 0 | 0 |
  | layer_20 | 391,643,136 | 398 | 984,027 | 0 | 0 |
  | layer_06 | 339,738,624 | 398 | 853,614 | 0 | 0 |
  | layer_21 | 339,738,624 | 390 | 871,124 | 0 | 0 |

- Verdict: NEUTRAL (characterisation; it retires a premise the architecture
  doc leaned on, and rule 5 is why it gets an entry)
- Notes:
  1. **EXP-008's "+51% at 16 MiB over the expert stride" is refuted, and it
     is refuted under either reading of what EXP-008 measured.** Sequentially,
     going from one expert blob to eight leaves `layer_00` unchanged within
     noise (1.578 to 1.553, 1.6% down) and costs the other three 15 to 16
     percent (2.271 to 1.924, 2.368 to 1.984, 2.317 to 1.951). Randomly, the
     same change *gains* 15 to 25 percent on three files and loses 3% on the
     fourth. Neither is +51%, and neither is close. EXP-008's harness is gone
     so its read order cannot be recovered, which is exactly why this entry
     reports both columns.
  2. **Sequentiality and granularity trade against each other, and the
     decomposition is only the K=8 slice of that.** At the single-blob size
     order matters a great deal: `seq/rand` at K=1 is 1.34x, 1.46x and 1.50x
     on `layer_20`, `layer_06` and `layer_21` (and 0.99x on `layer_00`). By
     K=6 it is 1.00x to 1.04x and by K=8 it is 0.99x to 1.01x. So it is **not**
     true that sequentiality buys nothing; it buys a great deal at small
     blocks, and a big block has already captured the same locality, which is
     why the two columns converge. Read the decomposition table as "at the
     8-expert window the runtime's sweep actually issues, order is worth
     nothing extra", not as a statement about the drive in general.
  3. **What actually predicts throughput is bytes in flight, and both sweeps
     agree on it.** This is derived from the two tables above rather than
     swept directly, and the derivation is only sound because block size and
     queue depth move the same quantity from opposite directions and land on
     the same curve. At matched bytes outstanding the two are
     interchangeable: `layer_20` reads 2.287 at K=4/QD8 and 2.290 at K=8/QD4,
     both 97.9 MB; 2.308 at K=2/QD8 and 2.294 at K=8/QD2, both 49.0 MB. The
     curve is flat at its peak up to roughly **100 MB outstanding**, drops
     about 12% by 127 to 147 MB, and settles 15 to 18 percent down past 170
     MB. A single outstanding request is 3 to 7 percent below peak, so the
     useful range is narrow but real. `layer_00` is the exception: flat at
     1.54 to 1.60 across the entire matrix, never reaching the regime where a
     knee could appear.
  4. **The shipped dials are at the peak, not past it, on both paths.** This
     corrects the obvious first reading of the queue-depth table. The prefill
     sweep issues one read per window of 8 experts with 2 windows in flight
     (`crates/core/src/io/sweep.rs`), which is 49.0 MB outstanding (46.7 MiB,
     the ring figure EXP-016 records) and is the K=8/QD2 row: 1.597 to 2.371
     GB/s, the best cell in the matrix. The decode ring is `RING_ENTRIES = 8`
     in `crates/core/src/io/stream.rs` over single-blob reads, at most 24.5 MB
     outstanding, which is the K=1/QD8 row and also at peak. The 1.92 to 1.98
     GB/s in the QD8 and QD16 rows is a large-block-*and*-deep-queue
     combination that **neither path issues**. The honest statement is
     therefore not "20% is left on the table" but "there is a ceiling near 100
     MB outstanding, both dials sit under it, and raising either of them is
     where the 15 to 18 percent would be lost": `windows_in_flight` 4 and
     `experts_per_window` 16 both stay inside it at 97.9 MB, while
     `windows_in_flight` 8 (195.8 MB) and `experts_per_window` 24 (146.9 MB)
     do not.
  5. **What this does not measure, and what it therefore licenses.** The
     queue-depth sweep was run at K=8 only, so the decode operating point
     (single blobs) has **no queue-depth curve of its own**; its QD8 point is
     measured and its QD2 and QD4 points are not. And the queue is threaded
     `preadv`, not io_uring, so none of this transfers to the runtime's
     submission path without a measurement. Those two gaps together are what
     justifies **an io_uring queue-depth experiment inside the runtime**, and
     they are why `RING_ENTRIES` should not be changed on the strength of this
     entry. It is a reason to run something, not a constant to copy. Backlog
     item recorded in `docs/architecture.md`.
  6. **Absolute bandwidth is 1.54 to 2.37 GB/s across the whole matrix**,
     against EXP-008's tabulated 1.211 to 1.390, and consistent with EXP-013's
     ~1.97 GB/s under the real access pattern at the real block size. The
     decode-shaped cell (K=1, random, QD 8) is **1.55 to 1.69 GB/s** and the
     prefill-sweep-shaped cell (K=8, sequential, QD 2) is **1.60 to 2.37
     GB/s**; those two are what the performance model in
     `docs/architecture.md` is now re-derived from. So EXP-008 was measuring a
     contended machine, exactly as its own Method warned it might be, and the
     performance model built on 1.211 to 1.349 GB/s was pessimistic by roughly
     25 percent at the decode block size.
  7. **Per-file variance persists, exceeds run-to-run variance by a wide
     margin, and is still unexplained.** `layer_00` at 1.578 against
     `layer_20` at 2.271 in the same sweep, a 1.44x spread, with **identical**
     extent geometry: both 398 extents, both mean 984,027 B and median 884,736
     B, both with zero extents physically adjacent to their successor. So
     fragmentation as `filefrag` reports it does **not** predict it, which is
     a real finding and a negative one: the obvious hypothesis is eliminated.
     What is left is physical placement on the drive, or QLC-internal
     behaviour such as SLC-cache residency or block wear, and this probe
     cannot see either. `docs/benchmark-machine.md` already records a 1.46x
     spread between these same two files from an earlier measurement, so this
     is reproducible rather than a one-off. **Consequence for every future
     benchmark: a bandwidth aggregate that hides this spread is worse than no
     aggregate**, and any run that changes which files it touches has changed
     its own baseline.
  8. **Zero compressed extents on all four files**, so the btrfs
     `compress=zstd:3` mount option is not silently downgrading these reads to
     buffered I/O through the encoded-extent path. Combined with zero resident
     pages after every run and a passing positive control, O_DIRECT was
     honoured. That is worth recording because it is a precondition for every
     number in this entry, and because EXP-009 established that a silent
     fallback is the one failure the 3 GB budget cannot survive.

## EXP-020: The attention kernel rebuilt in three waves, measured warm

- Date / commit: 2026-08-05 / `da9034b`..`aade585` (`feat/attention`). The
  baseline arm is `da9034b` itself, materialized with `git archive` into a
  private `CARGO_TARGET_DIR` — no checkout, no branch switch, no working-tree
  change.
- Hypothesis: EXP-017 measured attention at **61.3%** of a 512-token prefill
  and **85.2%** of an 1891-token one, and named two bit-neutral levers with no
  number attached to either: parallelize over rows, and vectorize the f16
  conversion and the dot. EXP-017 Note 6 asked for them to be measured
  separately. This entry does that, and adds a third the measurement itself
  produced: hoisting the f16-to-f32 conversion out of the GQA group.
- Method: the decode-attention context sweep in
  `crates/core/benches/kernels.rs` (`cargo bench -p ramvamp-core`), two arms
  over the v0 pin (`n_layers` 48, 32 q heads : 4 kv heads, `head_dim` 128,
  `scale` 1/sqrt(128), cap 4096, so GQA `group` = 8):

  - **Arm A — 1 layer**, 8 MiB of KV planes, timed unit = one
    `decode_attention` call. The pure kernel curve, largely cache-resident.
  - **Arm B — 48 layers**, 384 MiB of planes (the KV tenant's whole v0
    budget), timed unit = a sweep of all 48 layers, i.e. one decoded token's
    worth of attention.

  Context ladder N in {64, 128, 256, 512, 1024, 2048, 4096}; the three rungs
  tabulated below are the ends and the middle. Medians of an odd number of
  timed runs after untimed warmups (arm A 31/5 at every rung; arm B
  31/31/21/15/11/7/5 with warmups 3/3/2/2/1/1/1, because one baseline arm-B
  unit at 4096 is ~2.5 s). `q`, the K/V fill and the scratch are outside the
  timed region.

  **Rule-2 status, stated head-on: none of this is publishable.** Every number
  in this entry is **warm, in process, single-threaded and diagnostic**. Rule 2
  requires published throughput to come from a cold run inside `memory.max=3G`
  with `memory.swap.max=0`; this bench allocates its KV planes in process, does
  no I/O, touches no model file and reuses a warm process. It measures the
  kernel's shape, not the runtime's behaviour. EXP-017 is the precedent and it
  is the same shape of entry: warm, NEUTRAL, explicitly not publishable.

  **The cold rule-2 measurement is owed and is scheduled.** The runbook is
  committed at `scratch/phase7/wave3-runbook.md` — paired cold prefill at
  `--repeats 5`, paired cold decode at `--max-new 256`, the 4K-context
  `memory.peak` run, and the full numerics gate. **EXP-021 is reserved for it**
  (Note 9). Nothing in this phase may be quoted as a throughput result until
  that entry exists.

  How the columns were paired, because the two halves are not equally strong:

  - The **baseline** and **wave 1** columns are one hardened measurement.
    `crates/core/benches/kernels.rs` as modified in this phase was copied
    byte-for-byte into the `da9034b` archive, so the two sides differ in the
    kernel and in nothing else; four runs in palindromic order (before, after
    ‖ after, before) so a monotonic machine-warming ramp cancels; every arm-B
    drift-control ratio inside 0.999x-1.003x. Run-to-run spread better than
    ±1.1% on every cell but one, which is disclosed as contaminated and
    replaced by its own drift repeat.
  - The **wave 2** column is medians of **3** runs taken later, on a machine
    that was **not quiet**, and its drift controls are **mixed**: of the twelve
    control ratios printed across those runs, **seven** — 1.147x, 1.020x,
    1.022x, 0.891x, 0.987x, 1.085x and 0.904x — fall outside the 0.99x-1.01x
    band the bench's own guidance says to discard on, and no single one of the
    three runs is clean on all four. Treat that column, and therefore the
    cumulative ratios, as **approximate**. Putting it on one line with the
    paired baseline is a cross-session combination inside one entry; it is done
    here because both sides are this phase's own instrument on this phase's own
    machine, and it is flagged rather than hidden. The wave-1 figure is the one
    with a real pairing behind it.

  Raw output: `scratch/phase7/attn-base-fixed.txt`,
  `attn-base-fixed-run2.txt`, `attn-w1-fixed.txt`, `attn-w1-fixed-run2.txt`,
  `attn-w2.txt`, `attn-w2-run2.txt`, `attn-w2-run3.txt`. Full working notes in
  `scratch/phase7/attention-measurements.md`. `attn-final.txt` is kept for the
  record and **must not be read as data**: all four of its drift-control
  ratios are outside the band (1.169x, 1.276x, 0.685x, 0.982x) and its arm A
  jumps 4.4x for a 2x context step between 1024 and 2048.
- Baseline: the `da9034b` kernel — head-major loop order, scalar throughout,
  one f16-to-f32 conversion per query head per element — measured on the same
  instrument in the same session as the wave-1 column. It is the kernel
  EXP-017 measured at 61.3% of prefill.
- Result:

  **Arm B, 48 layers, ns per token's worth of attention** (medians):

  | context | baseline `da9034b` | wave 1 `8fc3c4a` | wave 2 `8bd079e` | cumulative |
  | ---: | ---: | ---: | ---: | ---: |
  | 64 | 36,904,308 | 13,560,338 | 4,073,820 | **9.1x** |
  | 512 | 298,586,546 | 112,125,892 | 37,925,280 | **7.9x** |
  | 4096 | 2,445,361,068 | 911,485,605 | 382,866,372 | **6.4x** |

  **Arm A, 1 layer, ns per `decode_attention` call** (medians):

  | context | baseline `da9034b` | wave 1 `8fc3c4a` | wave 2 `8bd079e` | cumulative |
  | ---: | ---: | ---: | ---: | ---: |
  | 64 | 759,794 | 281,071 | 73,849 | **10.3x** |
  | 512 | 6,092,496 | 2,241,654 | 576,323 | **10.6x** |
  | 4096 | 49,378,095 | 18,208,422 | 5,158,669 | **9.6x** |

  Arm A holds ~10x at every rung. Arm B decays from 9.1x to 6.4x across the
  ladder, and the decay is the finding, not the noise — see Note 3.

  **What each piece was worth**, every figure labelled:

  | piece | figure | status |
  | --- | ---: | --- |
  | Wave 1, GQA conversion hoist (`8fc3c4a`) | 2.708x arm A, 2.684x arm B | **measured**, paired, hardened instrument |
  | Wave 2, AVX2+F16C on the kernel (`8bd079e`) | ~4.2x | **measured** in process by the implementing lane, not through this bench |
  | Wave 2 as this bench sees it, arm B | 3.33x at 64, 2.96x at 512, 2.38x at 4096 | **derived** from the two columns above, cross-session |
  | Decode fan-out over kv heads (`066f539`) | 2.4x at 4096, 1.9x at 64 | **measured** on a loaded machine, so a lower bound |
  | Prefill fan-out over rows (`c78981e`) | ~5.9x on the attention region | **estimated** from a cost-balanced makespan |
  | Prefill fan-out, whole prefill | ~2.0x at 512 tokens, ~3.4x at 1891 | **estimated**, Amdahl on EXP-017's shares |
  | `primitives::softmax` share at 4096 positions | 23.5% (1,059 µs of 4,499 µs) | **measured** in process |
  | AVX2 K-widening redundancy | `ceil(group / 8)`: 1.00x to group 8, 2.00x at 9, 3.00x at 17, 4.00x at 32 | **measured**; v0 is group 8, so 1.00x |

  The prefill estimate is a makespan on the row cost model, not a timing: row
  `r` attends `start + r + 1` positions, so a 512-row chunk is 131,328 cost
  units, an even six-way split would be 21,888 each, and the cost-balanced
  split's longest shard carries **22,175** — 1.3% off ideal, hence
  131,328 / 22,175 = **5.9x**. Amdahl on EXP-017's measured shares then gives
  1 / (0.387 + 0.613/5.923) = **2.04x** at 512 tokens and
  1 / (0.148 + 0.852/5.923) = **3.43x** at 1891. Both are estimates on top of
  a warm measurement; neither is a prediction this entry stands behind, and
  EXP-021 is what replaces them.

  **Memory.** Prefill costs **zero additional heap bytes**, which is the
  standing rule and is unchanged: its six per-shard score buffers are carved
  from the `PrefillSession` arena, so the scratch half of the carve went
  **80,935,940 B (77.19 MiB) to 81,728,516 B (77.94 MiB)** of the 1,438.59 MiB
  slab — six carves of 132,096 B. Decode has no arena, so its scratch is heap:
  **+387 KiB of new anonymous memory**, field total 516 KiB, charged against
  the ~111 MiB of headroom `docs/architecture.md` already marks
  **provisional** and that EXP-018's unexplained 99-105 MiB residual already
  eats into.
- Verdict: NEUTRAL (nothing here satisfies rule 2, so nothing here publishes;
  the changes ship on bit identity, and the cold measurement is owed)
- Notes:
  1. **The instrument had a real defect, it did not fire, and the baseline is
     not retracted.** The bench committed in `da9034b` timed a closure that
     consumed `out[0]` — 1 of 4096 output floats — so under `lto = "thin"` and
     `codegen-units = 1` the other 31 heads and their softmaxes were
     eliminable dead stores in any `decode_attention` cheap enough to inline.
     `47176d6` fixed it (`black_box` on the whole output slice, on `q` and on
     the cache). Whether it had been firing is answered by measurement, not by
     argument: the fixed instrument against the defective one, **same kernel on
     both sides**, moves every cell by less than ±2%, with **mixed signs**, and
     on the wave-1 side — the side that would have been inflated — the fixed
     instrument reads slightly *slower*, which is the direction a `black_box`
     barrier moves things and not the direction removing a fabrication moves
     them. So the committed 2.7x was not fabricated and the baseline column
     stands as measured. The fix is a **guard for the next run**, and the
     danger is entirely in the next run: optimizing the kernel is exactly what
     makes it small enough to inline. `47176d6` also added a drift control —
     each arm re-measures its cheap rungs after the ladder at identical
     context and sample counts — which earned itself twice, catching a 12%
     transient in one cell and settling the residency-versus-thermal question
     by measurement (all eight arm-B control ratios across the four paired runs
     sit in 0.999x-1.003x, so the ladder's own upward drift is footprint, not
     clock).
  2. **The EXP-017 cross-check is not a validation, and rule 3 is why.**
     `prefill_token_major` is `for token in prompt { forward_token(...) }`
     hitting this same `decode_attention` with `positions = t` for t = 1..N, so
     the `attention` row of a token-major prefill at N tokens is the integral
     of the arm-B curve and `Σ c·t = c·N(N+1)/2` is the right form for it.
     That mechanism was checked and is correct. EXP-017 records the token-major
     512-token attention row at **82.38 s** (the 76.23 s in the same table is
     the *sweep* row, a different path, and is not the comparable figure).
     Against `Σ_{t=1..512} t = 131,328`, the baseline arm-B curve predicts
     **76.59 s** taking the flat constant at the N=512 cell, **76.39 s** by
     piecewise-linear interpolation and **75.77 s** by least squares — all
     **derived**, and all 7.0% to 8.0% below the measured row. That comparison
     puts a synthetic in-process microbench from this session on one curve with
     a figure measured in a different session on a different (pre-`da9034b`)
     kernel on a machine EXP-017's own Method records as busy, which is exactly
     what **rule 3** forbids; EXP-017 invokes rule 3 itself to refuse a
     *smaller* combination, declining to put its own 131 s and 124.27 s
     512-token sweep runs on one curve. The gap also sits inside the **5.4%**
     run-to-run spread those two runs document, so it is not resolvable at this
     precision anyway. The only statement licensed is: **the synthetic arm-B
     curve is not inconsistent with EXP-017's measured attention row.** No lane
     may cite it as validation.
  3. **Linearity in context broke, and that is a real finding.** The baseline
     kernel's cost was linear in context: `max(ns/pos) / min(ns/pos)` over all
     seven rungs is **1.016x** on arm A and **1.031x** on arm B, and arm B's
     `ns/pos/layer` climbs only 12,027 to 12,395 across the whole ladder. After
     wave 2 the same statistic reads **1.097x / 1.253x / 1.117x** on arm A and
     **1.572x / 1.400x / 1.484x** on arm B over the three runs, with arm B's
     `ns/pos/layer` climbing 1,326 to 1,968 on the run whose controls were
     least bad. Nothing got slower — the arithmetic got roughly 10x cheaper and
     the memory traffic did not move at all, so the memory term went from a
     rounding error to a third of the cost at the long end. This is why arm B's
     cumulative speedup decays 9.1x to 6.4x while arm A holds ~10x: arm A's
     8 MiB of planes stay cache-resident and arm B's 384 MiB do not.
     **Attention is drifting memory-bound at long context**, which changes what
     the next lever should be — blocking or tiling rather than more arithmetic.
  4. **Softmax is the new wall, and the Amdahl bound in `8bd079e`'s commit
     message is stated on the wrong side.** `primitives::softmax` is **23.5%**
     of the kernel at 4096 positions (1,059 µs of 4,499 µs, measured). It is
     `f64::exp` once per (query head, position) — 32 x 4096 = 131,072 of them
     per call — plus an f64 normalizer accumulated serially, and `primitives`
     is frozen. Two Amdahl bounds follow, both **derived**, and they are not
     interchangeable: perfecting **softmax alone** buys at most
     `1 / (1 - 0.235)` = **1.31x** on the kernel, while leaving softmax frozen
     and driving **everything else** to zero buys at most `1 / 0.235` =
     **4.26x**. `8bd079e`'s message ("softmax ... is frozen, which caps
     anything further at 1.31x") attaches the first bound's number to the
     second bound's claim. No measurement moves, and softmax is still the next
     thing in the way; what changes is how much room is left behind it, and
     4.26x is the number a future lane should plan against. Vectorizing softmax
     bit-neutrally is not a patch either: it means reproducing libm's
     `f64::exp` lane-for-lane, and the normalizer is a serial f64 chain that
     cannot be lane-split without reassociating it.
  5. **Decode's fan-out ceiling is `n_kv_heads` = 4, and it is structural.**
     `066f539` split decode attention over kv heads after review measured the
     first attempt — a strided slab of query heads — fighting the
     vectorization it was meant to compose with: `x86::dot_block` puts its
     eight lanes on the GQA group, so a one-head slab issues a full group's
     vector-op count with seven lanes carrying zeros (whole `group = 8` call
     4,293.8 µs, `group = 1` slab 1,981.1 µs, so eight slabs are ~15.8 ms of
     CPU against 4.3 ms — derived from those measurements). End to end that was
     1.8x at six shards for 3.1x the CPU, and it **inverted** at short context,
     0.46x at 64 positions, which is where EXP-014's decode baseline sits. The
     kv-head split gives 2.4x at 4096 and 1.9x at 64 instead. It also cannot go
     past 4 units at the v0 pin, because there are 4 kv heads. Going wider
     needs an axis that is neither the group (measured worse) nor positions
     (that needs the forbidden rescaled softmax).
  6. **"Each element widened once per kv head" is true on the scalar path and
     `ceil(group / 8)` on the AVX2 one.** `x86::qk_scores` walks the group in
     chunks of 8 lanes and the whole position sweep, K widening included, sits
     *inside* that chunk loop. Counted directly: K is **1.00x** for group 1-8,
     **2.00x** for 9-16, **3.00x** for 17-24 and **4.00x** for 25-32, while V
     is 1.00x throughout and the scalar path is 1.00x throughout. **v0 is
     group 8, where the ratio is exactly 1**, and so is this bench's geometry,
     so no number in this entry moves. It is written down because the
     correctness sweep now carries groups 9, 12, 17 and 32, and an unqualified
     claim would have sat next to its own disproof. Not hoisted, deliberately:
     hoisting needs the position axis outer and the group axis inner, which
     makes the transposed query block hold the whole group — a size bounded by
     nothing in the geometry — or re-transpose once per position block.
  7. **`crates/core/src/kernels/attention/x86.rs` is entirely outside Miri's
     reach.** Miri takes the scalar path, because `is_x86_feature_detected!` is
     false under it, so no `unsafe` block in that file is ever executed by the
     tool. Its soundness rests on inspection (every load and store is an
     unaligned form; every slice bound is argued at the call site) and on the
     bit-identity sweep, which proves the *results* match the scalar reference
     on every geometry it covers — **not** on tooling. Miri did earn its keep
     elsewhere in this phase: on the pre-`65b0b3a` tree it rejected the decode
     fan-out three separate ways (Stacked Borrows against the `Unique` held by
     the kernel body, Stacked Borrows against the `SharedReadOnly` held by its
     validation parameter, and Tree Borrows as an outright data race), all from
     shards rebuilding a `&mut` over the whole output vector. The writes were
     disjoint and every test passed; it was still UB. `out` is now the shard's
     own sub-slice indexed from zero, so disjointness is structural.
  8. **Bit identity held at every step, which is the only reason any of this
     was allowed.** Wave 1 is pinned against the pre-restructure loop nest,
     kept verbatim as a test-only reference; wave 2 is pinned against the
     scalar path with **zero tolerance**, including a sweep of all 65,536 f16
     bit patterns through `_mm256_cvtph_ps` (signalling NaN scoped out, since a
     non-finite activation means the pass already failed upstream); the
     fan-outs are pinned by a union test asserting that disjoint kv ranges
     reproduce the whole call bit for bit. **No FMA anywhere in `x86.rs`**, and
     that is load-bearing rather than cautious: `acc += qv * kv` is two
     roundings and `_mm256_fmadd_ps` makes it one. Injecting an FMA at each of
     the two sites in an isolated copy broke four tests each time, which also
     proves the vector path is genuinely taken rather than silently falling
     back.
  9. **EXP-021 is reserved for the cold rule-2 measurement of this work**, and
     this entry does not pay it. What EXP-021 owes: paired cold prefill at 512
     tokens with `--repeats 5` (which also closes EXP-018 Note 5's unbounded
     spread), paired cold decode at `--max-new 256` (which settles the EXP-018
     cold-start question, since at `--max-new 4` almost everything measured was
     the transient), the 4K-context `memory.peak` run that is the last open
     item on the 11-slot dial, and the numerics gate. The runbook is
     `scratch/phase7/wave3-runbook.md`. Until EXP-021 exists, the honest
     summary of phase 7 is "the kernel got roughly 6 to 10x faster warm, on a
     microbenchmark, and nobody has measured what that did to a token."

## EXP-021: Phase 7 measured cold: prefill, decode and 4K context under rule 2

- Date / commit: 2026-08-05. Phase-7 arm: `target/release/ramvamp` sha256
  `d56dc034ebd3...`, the tree at `aade585` (`feat/attention`), which is the
  last commit up to `4b39104` that touches `crates/` at all — so that binary
  is the phase-7 runtime as it stands today. Session 1's harness recorded HEAD
  as `f6d8c7c` and session 2's as `9141341`; those commits add the two
  harness scripts and nothing else, and session 1 recorded the same binary
  sha256 after its `cargo build --release`, so both sessions ran that binary.
  Phase-5 arm: the binary banked by EXP-018 at `scratch/phase5-ref/ramvamp`,
  commit `c3572fd`, sha256 `e0f58f2486...`. Harness correction: `4b39104`.
- Hypothesis: EXP-020 rebuilt attention and measured 6.4x-9.1x on a warm
  in-process microbench, and EXP-020 Note 9 reserved this entry for the cold
  rule-2 measurement of what that did to a token. Four things are owed here:
  paired cold prefill at 512 tokens with `--repeats 5` (closing EXP-018
  Note 5), paired cold decode at `--max-new 256` (settling the EXP-018 Note 1
  cold-start question), the 4K-context `memory.peak` run (the last open item
  on the 11-slot dial, EXP-014 Note 2), and the numerics gate.
- Method: two unattended sessions on the reference machine
  (`docs/benchmark-machine.md`), both driven by committed harnesses so what
  ran is readable rather than reconstructed:

  - **Session 1**, `bash scripts/phase7_overnight.sh`, started 15:49:19,
    logs in `scratch/phase7/overnight-20260805-154919/`. Numerics gate, then
    the four cold steps. Raw summaries `scratch/cold-bench/p7-*.json`.
  - **Session 2**, `bash scripts/phase7_rerun_cold.sh`, started 17:34:10, logs
    in `scratch/phase7/recold-20260805-173410/`. The cold steps only, with no
    model work before them and a settle loop that waits for `MemAvailable` to
    hold above 6,000 MiB for four consecutive 15-second samples. Raw summaries
    `scratch/cold-bench/re-*.json`.

  Every cold step is `scripts/cold_bench.py`: evict all 53 model files with
  `posix_fadvise(POSIX_FADV_DONTNEED)` and prove the eviction with `mincore`,
  launch under `systemd-run --user --wait -p MemoryMax=3G -p MemorySwapMax=0
  -p MemoryAccounting=yes`, and read `memory.peak`, `memory.events`,
  `memory.stat` and `/proc/self/io` `read_bytes` from inside the cgroup before
  exit. The prompt goes out of band as JSON. The 512-token workload is
  `models/llamacpp-ref/llamacpp_ref/long_00.txt`, delivered sha256
  `d1b6c407c55a...` over 2,002 bytes on all 22 of the 512-token runs, so both
  arms are one workload. The 4K workload is
  `scratch/ctx4k/p4k.txt`, sha256 `68582aae37b9...` over 17,000 bytes.
  Prefill: `--max-new 4 --warmup 1 --repeats 5`. Decode: `--max-new 256
  --warmup 0 --repeats 1`. 4K: `--max-new 8 --warmup 0 --repeats 1`.

  **The hygiene story, head-on, because a rule changed under these numbers.**
  Both sessions initially reported most runs DIRTY: 11 runs across the two
  sessions recorded nonzero `pgsteal` and `cold_bench.py` refused to publish
  them. The reclaim was **entirely khugepaged** in all 11 — see Note 1 for the
  evidence — and `4b39104` corrected the verdict to key on the pressure
  reclaimers rather than on the bare `pgsteal` total, still reporting
  khugepaged as a SOFT note. That commit also added `--reverdict`, which
  re-classifies already-recorded counters and prints the old verdict beside
  the new one.

  **The verdicts below come from `--reverdict`, not from re-running.** Every
  counter the hygiene verdict depends on is in the summary files already, so
  correcting the rule cost no further cold runs, and the correction is visible
  as a diff:

  ```
  python3 scripts/cold_bench.py --reverdict \
      scratch/cold-bench/p7-*.json scratch/cold-bench/re-*.json
  ```

  All nine summaries re-classify to **PASS**; seven of them were recorded
  DIRTY. A reader who thinks the correction is wrong should read Note 1 and
  Note 3 and then discount this entry, which is the point of saying so here
  rather than presenting nine numbers that came back clean.

  Independent of the reclaim question, on all 24 runs across both sessions
  (21 scored, 3 discarded warmups): every `memory.events` counter 0,
  `memory.swap.peak` 0, every return code 0, and `read_bytes` non-trivial, so
  eviction did happen on every one.
- Baseline: the phase-5 arm of each pair, run back to back with the phase-7
  arm in the same session on the same prompt — the same banked binary EXP-018
  used, so the comparison is paired and rule 3 is satisfied inside this entry.
  Two figures from other entries are quoted as context and are **not** put on
  one curve with anything here: EXP-018 measured phase-6 prefill at **4.23
  tok/s** cold and decode falling 1.88 to 1.38 tok/s at `--max-new 4`, and
  EXP-014 recorded cold medians of decode 1.88 tok/s, prefill 1.33 tok/s and
  `memory.peak` 2,471.1 MiB at a 5-token prompt.
- Result:

  **Prefill, 512-token prompt, `--max-new 4`.** Medians of 5 scored runs after
  1 discarded warmup, session 1, both arms back to back:

  | metric | phase 5 | phase 7 | ratio |
  | --- | ---: | ---: | ---: |
  | prefill | 1.68 tok/s | **11.17 tok/s** | **6.65x** |
  | prefill time | 304.67 s | 45.84 s | 6.65x |
  | decode (4 tokens) | 1.81 tok/s | 2.08 tok/s | 1.15x |
  | model load | 1.30 s | 1.30 s | 1.00x |
  | wall | 308.88 s | 49.70 s | 6.21x |
  | cgroup `memory.peak` (median) | 2,571.8 MiB | 2,577.9 MiB | |
  | process `read_bytes` | 239.33 GB | 20.72 GB | **11.55x fewer** |

  **Headline: prefill 6.65x over phase 5, cold, inside `memory.max=3G` with
  swap off, over 5 scored runs per arm.** The phase-7 arm's scored spread is
  49.57 s to 49.91 s of wall and 11.13 to 11.21 tok/s of prefill; the phase-5
  arm's is 308.54 s to 311.84 s and 1.66 to 1.68 tok/s. As a full range about
  each arm's own wall median that is **0.70%** and **1.07%**, which is the
  bound EXP-018 Note 5 asked for and did not have.

  Session 2 re-ran the phase-5 arm alone (the phase-7 arm had already scored
  PASS unaided): prefill **1.68 tok/s**, wall 309.50 s, decode 1.85 tok/s,
  `memory.peak` median 2,571.0 MiB. Identical to the printed resolution on
  every field that matters.

  **Against EXP-018's phase-6 figure, 11.17 against 4.23 tok/s is 2.64x — and
  that ratio is weaker than the 6.65x above, for the reason rule 3 exists.**
  The 4.23 is one run in a different session on a different build; nothing
  here re-measures it. Two things make the comparison worth writing down
  anyway and neither makes it a paired number: the phase-5 arm reproduces
  across the two entries at 1.65 against 1.68 tok/s on the same banked binary
  and the same prompt, a 1.8% gap; and the byte counts are identical to the
  byte (Note 7). Quote 6.65x. Do not put 1.65, 4.23 and 11.17 on one curve.

  **Decode, 512-token prompt, `--max-new 256`.** One run per arm per session:

  | metric | ph 5, s1 | ph 7, s1 | ph 5, s2 | ph 7, s2 |
  | --- | ---: | ---: | ---: | ---: |
  | decode | 1.18 tok/s | **1.99 tok/s** | 1.17 tok/s | **1.83 tok/s** |
  | decode time (256 tokens) | 217.05 s | 128.82 s | 217.97 s | 140.25 s |
  | prefill | 1.66 tok/s | 11.21 tok/s | 1.68 tok/s | 11.29 tok/s |
  | wall | 526.89 s | 176.46 s | 524.01 s | 187.60 s |
  | cgroup `memory.peak` | 2,597.8 MiB | 2,608.6 MiB | 2,602.1 MiB | 2,611.6 MiB |
  | process `read_bytes` | 365.90 GB | 146.44 GB | 365.90 GB | 146.44 GB |

  Within session 1 that is **1.69x**; within session 2, **1.56x**. Each is a
  paired ratio inside one session; the two sessions are not averaged.

  **4K context, phase 7, `--max-new 8`.** `p4k.txt` tokenizes to **3,961
  prompt tokens**, inside `CONTEXT_CAP = 4096`:

  | metric | session 1 | session 2 |
  | --- | ---: | ---: |
  | prefill | 9.89 tok/s | 9.86 tok/s |
  | decode (8 tokens) | 1.47 tok/s | 1.37 tok/s |
  | wall | 408.07 s | 409.44 s |
  | cgroup `memory.peak` | **2,920.4 MiB** | **2,924.2 MiB** |
  | headroom under 3,072 MiB | 151.6 MiB | 147.8 MiB |
  | process `read_bytes` | 146.10 GB | 146.10 GB |

  **The memory contract holds at full context**, with 148 to 152 MiB spare.
  This is the run EXP-014 Note 2 named as the remaining open item on the
  11-slot dial and it closes it.

  **Numerics, session 1, on the same binary, all three gates PASS:**

  | gate | result |
  | --- | --- |
  | `bitident.py` vs the phase-4 baseline | **PASS**, 8/8 singles byte-identical |
  | `kl_vs_reference.py --refresh`, gate 3 | **PASS**, mean KL 1.039e-02, worst prompt 2.721e-02, top-1 8/8 |
  | `greedy_regression.py` | **PASS**, no metric below the 2026-08-03 baseline |

  The greedy aggregate is 407/1024 tokens (39.7%) with 1/8 prompts identical
  for all 128 tokens, and the path check is top-1 24/24 with 24/24 contexts
  verified — the same shape as the recorded baseline, which is what the gate
  compares against.
- Verdict: KEEP
- Notes:
  1. **The reclaim that dirtied 11 runs was khugepaged, and the evidence is
     four independent facts.** (a) `pgsteal_khugepaged` accounted for **100%**
     of `pgsteal` in every one of the 11, 44 to 753 pages (0.2 to 2.9 MiB),
     with `pgsteal_kswapd`, `pgsteal_direct` and `pgsteal_proactive` all zero.
     (b) `pgscan == pgsteal` **exactly** in every one — a 100% steal rate,
     which is the signature of targeted freeing, not of LRU scanning under
     pressure. (c) The machine's Normal zone sat **16x above the watermark
     that wakes kswapd**, so there was no pressure to reclaim under. (d) Wall
     times were statistically identical: over the 12 runs of the identical
     phase-5 512-token workload (6 per session, warmups included), mean wall
     was **309.14 s for the flagged runs against 310.34 s for the clean ones**
     — the flagged runs were **1.20 s, or 0.39%, faster**, which is the
     expected sign, since collapsing base pages into 2 MiB hugepages helps the
     TLB. khugepaged wakes on its own 10-second timer, scans a bounded number
     of pages and frees the base pages it collapses, and that freeing lands in
     `pgsteal_khugepaged` with nothing under memory pressure at all.

     Two provenance caveats. The free-page and watermark figures behind (c)
     were read from `/proc/zoneinfo` during the investigation (944,748 free
     pages against a high watermark of 57,965, hence 16.3x) and are **not**
     captured in any summary file; what is committed is the 16x ratio, in
     `scripts/cold_bench.py`'s module docstring and in `4b39104`'s message.
     And those same two places record the wall means as 309.13 s and 310.35 s;
     recomputed from the JSONs they are 309.14 s and 310.34 s, a 0.01 s
     rounding difference in each that changes nothing.
  2. **The dirty/clean split inverted between the sessions, and that is what
     falsifies the alternative explanation.** Session 1's phase-5 prefill went
     dirty on runs 0-2 and clean on runs 3-5; session 2's went **clean on runs
     0-2 and dirty on runs 3-5**. Same workload, same binaries, same harness
     logic. `scripts/phase7_rerun_cold.sh`'s own header states the hypothesis
     it was written on — that the ~15-minute numerics gate immediately before
     the cold runs left global memory pressure, so the early runs paid for it
     and the machine settled — and session 2 refutes that hypothesis: it ran
     no model work beforehand, its settle loop recorded `MemAvailable` at
     **11,371 / 11,354 / 11,369 / 11,362 MiB** before its four steps with swap
     use flat at 3,155-3,156 MiB, and its dirty runs were the *late* ones. An
     independent daemon on its own timer fits an arbitrary split in either
     direction; workload-driven pressure does not.
  3. **The correction was verified not to weaken the gate, and the
     verification is the reason to trust it.** `4b39104` subtracts khugepaged
     *from* the total rather than summing the reclaimers it knows about, so a
     reclaimer the script has never heard of counts as pressure instead of
     vanishing, and a `memory.stat` with no breakdown at all still counts the
     whole total — unknown stays DIRTY. Verified by test: kswapd, direct,
     proactive, an invented reclaimer name, a missing breakdown, and
     khugepaged mixed with kswapd all still fail HARD, and the mixed case
     reports the **kswapd** pages rather than the total. khugepaged is still
     reported on every run it touched, as a SOFT note, so a run is never
     silently credited as clean when something did touch its pages.
     `--reverdict` runs nothing and prints old verdict beside new, so this is
     a rule change a reader can audit rather than a number that quietly
     improved.
  4. **EXP-018's decode regression is explained, and it is reversed.** EXP-018
     measured decode falling 1.88 to 1.38 tok/s and Note 1 offered a
     cold-start transient as an unmeasured hypothesis. The measurement that
     settles it is the decode window, taken here on both arms:

     | arm | decode at `--max-new 4` | decode at `--max-new 256` | ratio |
     | --- | ---: | ---: | ---: |
     | phase 5, session 1 | 1.81 tok/s | 1.18 tok/s | 0.65x |
     | phase 5, session 2 | 1.85 tok/s | 1.17 tok/s | 0.63x |
     | phase 7, session 1 | 2.08 tok/s | 1.99 tok/s | 0.96x |

     **Phase 5's own decode degrades by a third as the window grows from 4
     tokens to 256**, on one machine in one session, which is exactly what
     EXP-020 Note 3's linear-in-context finding predicts: context grows 516 to
     768 and attention's per-token cost grows with it. Phase 7 holds 1.83-1.99
     over the same window. So EXP-018's 1.88 was a 4-token figure inflated by
     a warm cache and a short context, its 1.38 was almost entirely the
     transient, and the "regression" was a comparison of two things neither of
     which was steady state. At the window where it matters phase 7 is
     **1.56x-1.69x faster than phase 5**, and it is faster at `--max-new 4`
     too (2.08 against 1.81/1.85), so the empty-cache cost is no longer
     visible as a regression at any window measured. (Phase 7's 4-token figure
     is session 1 only; its 1.83 tok/s at 256 tokens is session 2, so the
     0.88x that pair implies crosses sessions and is not tabulated.)

     **The mechanism EXP-018 hypothesized is confirmed on the phase-5 side.**
     The surviving stderr sidecar for a scored phase-5 512-token run
     (`scratch/cold-bench/run05.json.stderr`) records its decode phase as
     1,152 expert requests, 718 hits (**62.3%**), 434 misses of which only
     **4 are cold** — the token-major prefill did leave the decode cache warm,
     which is the half of EXP-018 Note 1 that was argued from `stream.rs`
     rather than measured.

     **EXP-005's "prefill does not warm the cache" decision does not need
     revisiting, and this makes the question moot rather than answering it in
     EXP-005's favour.** EXP-005 closed prompt-replay-into-cache as worth
     **+0.09 points** of hit rate; EXP-018 Note 1 reopened it only because the
     sweep's empty cache appeared to cost real throughput. It does not: phase
     7 is at or above phase 5 at both windows measured here, so there is no
     deficit for prompt replay to recover and no reason to spend a phase on a
     +0.09-point lever. What this entry still does not do is measure the size
     of the transient itself; it measures that the transient no longer shows
     up as a loss.
  5. **4K context fits, and the `memory.peak` progression is now three
     measured points.** Within this entry, on the phase-7 arm: **2,577.9 MiB**
     median at 512 prompt tokens plus 4 decode tokens, **2,608.6-2,611.6 MiB**
     at 512 plus 256, and **2,920.4-2,924.2 MiB** at 3,961 plus 8. The last
     one leaves 148-152 MiB under the 3,072 MiB ceiling with `pgsteal` 0 and
     every `memory.events` counter 0, so the **11-slot dial is validated at
     full context** and EXP-014 Note 2 closes. For the record and not as a
     curve: EXP-012 predicted 2,961.0 MiB, EXP-014 Note 2 extrapolated ~2,848
     MiB and EXP-018 Note 3 extrapolated ~2,906 MiB; the measurement lands
     between the last two and about 40 MiB under the first. That 148-152 MiB
     of measured spare is not the same quantity as the 111.0 MiB of headroom
     `docs/architecture.md` marks **provisional** — the architecture figure is
     a budget built on a provisional `anon` row that fails rule 2, and this is
     a measured cgroup peak on one workload. They agree in sign and order of
     magnitude, which is all that should be read into it; the `anon` row still
     needs its own re-measurement before the budget can be published.
  6. **EXP-018's unexplained 99-105 MiB residual is still unexplained, and the
     4K run did not close it.** Phase-7 512-token prefill peaks land in the
     same band EXP-018 recorded: median 2,577.9 MiB on the phase-7 arm and
     2,571.8 MiB on the phase-5 arm, against EXP-018's 2,576.0 and 2,570.1,
     with a full scored range of 2,569.4-2,583.5 MiB across both arms and both
     sessions. So the residual did not move, phase 7 did not add to it, and
     nothing here identifies it. It remains a backlog item; the 4K run
     established that the contract survives full context, which is a different
     question from where those ~100 MiB go.
  7. **Phase 7 moved no bytes, and the counter says so exactly.** All six
     phase-7 512-token runs read **20,716,994,560 B** — the same integer, run
     to run, and byte-for-byte the count EXP-018 recorded for phase 6. That is
     1.11x the 18,626,213,888 B installed model, the signature of a sweep that
     reads each expert once. This is a byte identity across two entries, not a
     throughput comparison, and it is quoted only to say that the attention
     work touched no I/O path.
  8. **EXP-018's inferred 58% token-major prefill hit rate is now measured at
     58.2%.** EXP-018 derived "near 58%" from bytes on the assumption that
     every miss reads one whole blob. The phase-5 stderr sidecar records the
     prefill phase directly: **196,608 requests, 114,375 hits (58.2%), 82,233
     misses (4,343 cold / 77,890 eviction), 220.8 GiB read in 82,233 reads,
     106.46 s of io wait**. The inference was right; it is now a measurement.
     77,890 eviction misses against 4,343 cold ones is the 11-slot pool
     thrashing, which is the thing the sweep exists to avoid.
  9. **The 4K prefill phase split, which redirects phase 8's target.** From
     the session-2 4K run's stderr sidecar
     (`scratch/cold-bench/run00.json.stderr`), shares of that run's own
     401.53 s prefill: **expert compute 42.1%** (169.19 s), **projections
     22.8%** (91.61 s), **attention 20.4%** (81.84 s), **elementwise 12.2%**
     (49.01 s), **expert io 2.5%** (9.89 s, of which 7.17 s blocked on the
     drive), other 0.00 s. Its decode split over 5.85 s: expert io 42.0%,
     attention 29.3%, expert compute 17.3%, projections 9.8%, elementwise
     1.7%. **Attention is no longer the wall in prefill.** EXP-017 measured it
     at 85.2% of an 1891-token prefill on the pre-phase-7 kernel; that figure
     and this one come from different entries, different sessions and
     different kernels, so rule 3 forbids putting them on one curve and the
     only licensed statement is directional: the term EXP-017 named as the
     target is now the third largest, and expert compute is the largest.
     A phase-8 lane should measure the split on its own workload before
     choosing, not inherit this one.
  10. **EXP-014's own DIRTY runs cannot be re-checked, because the JSONs are
      gone.** EXP-014 recorded a first attempt that scored 3 of 5 clean with
      the two dirty runs at `pgsteal` 2,817 and 2,946 pages, coinciding with
      the operator opening a terminal mid-run. Those are two orders of
      magnitude larger than anything here (44-753 pages) and they have a
      recorded external cause, so they are probably genuine pressure — but
      after `4b39104` the classification is checkable, and nobody has checked
      it. It cannot be checked now: the only surviving EXP-014 artifact is
      `scratch/cold-bench/summary.json` (mtime 2026-08-04 11:47), which is the
      **clean** second attempt, 6 runs with `pgsteal` 0 on every one. The
      first attempt was written to the same default path and overwritten,
      `scratch/` is in `.gitignore` so there is no history, and a search of
      every cold-bench-shaped JSON under `scratch/` finds no run with either
      value. `--reverdict` on the surviving file returns PASS unchanged, so
      **EXP-014's published numbers are unaffected by the rule change either
      way**; what is lost is the ability to say whether its discarded runs
      were correctly discarded. Harness lesson, worth more than the lost data:
      `cold_bench.py`'s per-run sidecars (`run0N.json.stderr`) are written to a
      fixed path and clobbered by the next invocation, which is also why this
      entry has stderr for the 4K run and the phase-5 512 run and for nothing
      else. A summary that is going to be cited should be copied to a named
      file at the time, as `phase7_overnight.sh` does with `--json`.
  11. **What this entry does not settle.** The prefill pair is 5 scored runs
      per arm and its spread is bounded; the decode pair and the 4K run are
      `--repeats 1`, so EXP-018 Note 5's criticism still applies to them and
      their only spread control is that two independent sessions agree (1.99
      against 1.83 tok/s on phase-7 decode is an 8.7% gap between sessions,
      and that is the honest error bar on that figure). The phase-7 512-token
      prefill arm was measured in session 1 only. Nothing here measures the
      size of the cold-start transient itself, only that it no longer costs a
      regression. And the phase-7 512-token prefill and decode splits were not
      captured — the sidecars were overwritten — so the only phase split this
      entry carries is the 4K one in Note 9.

## EXP-022: T_BLOCK 4 to 8 with a stepped position tail, measured warm

- Date / commit: 2026-08-05 / on top of `d329890` (`feat/decode`, pre-commit).
  One runtime file moves,
  `crates/core/src/kernels/attention/x86.rs`: `T_BLOCK` 4 to 8, plus a new
  `T_TAIL_BLOCK = 4` rung, so the position sweep runs 8-wide while a whole
  block fits, then **at most one** 4-wide, then 1-wide. The dispatch sweep in
  `crates/core/src/kernels/attention.rs` grew to cover the remainder classes
  the new rung created; that is a test change and it is Note 3.
- Hypothesis: EXP-020's backlog named `T_BLOCK = 4` the cheapest remaining win
  in that file at an **estimated** "roughly 1.5-2x on the QK dot", with no
  measurement behind it. Eight positions per block runs eight independent
  accumulator chains where four ran before, which covers the vector add's
  ~4-cycle latency twice over, and eight is the last width that still fits
  AVX2's sixteen ymm registers. The arithmetic is unchanged and only
  instruction-level parallelism moves, so the question here is narrow: does it
  show up in a token's worth of attention.
- Method: two source trees differing in exactly that one file (`T_BLOCK 4`
  with no tail rung against `T_BLOCK 8` + `T_TAIL_BLOCK 4`), both otherwise
  byte-identical to `main` at `d329890` (`diff -rq` confirmed), built into
  separate `CARGO_TARGET_DIR`s and run as `cargo bench -p ramvamp-core`. The
  instrument is EXP-020's decode-attention context sweep over the v0 pin
  (`n_layers` 48, 32 q heads : 4 kv heads, `head_dim` 128, GQA `group` = 8):
  **arm A** one layer, timed unit one `decode_attention` call; **arm B** all
  48 layers, timed unit one token's worth of attention. Context ladder N in
  {64, 128, 256, 512, 1024, 2048, 4096}. **Six runs, interleaved** across
  ~25 minutes on an otherwise idle machine (loadavg 0.3-1.5) with 90-180 s of
  settle before each: three per arm, labelled A1-A3 (`T_BLOCK 4`) and B1-B3
  (`T_BLOCK 8`).

  **Rule-2 status, stated head-on: none of this is publishable.** It is warm,
  in process, single-threaded and uncgrouped; it allocates its KV planes in
  process, does no I/O and touches no model file. It exists to decide whether
  `T_BLOCK = 8` is worth carrying into a cold sweep, and for nothing else.
  EXP-017 and EXP-020 are the precedents and this is the same shape of entry.

  **The cold rule-2 measurement is owed and is specified.** The paired cold
  arms are `scripts/phase8_decode_sweep.sh`, two rungs (512 and 3,961 prompt
  tokens) run twice, once on the branch binary and once on the banked
  reference below. **EXP-023 is reserved for it** (Note 6).

  The working notes behind this entry were never committed, so the entry
  carries the whole ladder itself rather than citing a raw file.
- Baseline: the `T_BLOCK = 4` kernel, which is the code EXP-020 shipped, a
  plain `::<T_BLOCK>` loop with a `::<1>` tail. It was built from the same
  tree in the same session as the `T_BLOCK = 8` arm, so the two differ in that
  constant and in nothing else.

  For the owed cold pair the same baseline is a **banked binary**:
  `scratch/phase7-ref/ramvamp`, banked at `d329890` with its own `COMMIT` and
  `SHA256` beside it, sha256
  `d56dc034ebd3e94e83b59ad64503adf289baf22af112752593ec59068a586e66`. **Those
  are the bytes EXP-021 measured**, which records `d56dc034ebd3...` for its
  phase-7 arm at `aade585`; `d329890` is the merge that carries that same
  runtime, and EXP-021 already establishes `aade585` as the last commit
  touching `crates/`. So the reference arm is not a lookalike rebuild of
  phase 7, it is the same executable phase 7 published from, and that is
  worth stating plainly because it removes a whole class of doubt: no
  toolchain drift, no profile drift, no "it should be equivalent". The sha256
  is checked twice at run time, against the `SHA256` file beside the binary
  and against a constant pinned independently at
  `scripts/phase8_decode_sweep.sh:191`, because a re-banked reference would
  agree with a regenerated sidecar file and still not be phase 7's binary.
- Result:

  **Drift verdicts first**, because they decide what the rest is worth. The
  bench re-measures its cheap rungs after the ladder at identical context and
  sample counts; a control ratio outside 0.99x-1.01x means discard, not
  interpret (GOTCHA 3 in `docs/handoff-phase8.md`):

  | run | `T_BLOCK` | arm A ctl (64 / 512) | arm B ctl (64 / 512) | arm B usable |
  | --- | --- | --- | --- | --- |
  | A1 | 4 | 1.001x / **1.149x** | 0.996x / 1.002x | yes |
  | A2 | 4 | 0.996x / **1.093x** | **1.090x** / **1.013x** | no |
  | A3 | 4 | 0.998x / **1.096x** | 1.000x / 1.009x | yes |
  | B1 | 8 | 1.000x / 1.000x | 0.998x / 0.999x | yes |
  | B2 | 8 | 0.999x / **1.094x** | 1.000x / **0.977x** | no |
  | B3 | 8 | 1.001x / **1.095x** | 0.998x / **1.012x** | no |

  **Arm B, 48 layers, ns per token's worth of attention, every run:**

  | ctx | A1 (T4) | A2 (T4) | A3 (T4) | B1 (T8) | B2 (T8) | B3 (T8) |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 64 | 3,482,246 | 3,481,450 | 3,424,909 | 3,467,931 | 3,520,981 | 3,470,224 |
  | 128 | 7,026,379 | 6,956,720 | 6,865,715 | 6,883,976 | 7,008,652 | 7,025,442 |
  | 256 | 16,738,816 | 16,334,850 | 16,044,447 | 15,917,198 | 16,067,438 | 15,971,978 |
  | 512 | 35,731,384 | 35,518,286 | 35,034,213 | 33,979,205 | 34,751,336 | 34,172,745 |
  | 1024 | 90,242,577 | 82,110,584 | 82,200,148 | 77,785,586 | 78,092,117 | 77,310,049 |
  | 2048 | 175,050,397 | 172,405,781 | 171,806,569 | 161,788,343 | 162,078,146 | 162,137,193 |
  | 4096 | 363,785,086 | 358,728,098 | 360,529,316 | 335,757,015 | 335,632,540 | 334,116,139 |

  **Medians, and whether the two arms' full ranges overlap:**

  | ctx | T4 median | T8 median | ratio | ranges overlap? |
  | ---: | ---: | ---: | ---: | --- |
  | 64 | 3,481,450 | 3,470,224 | 1.003x | yes |
  | 128 | 6,956,720 | 7,008,652 | 0.993x | yes |
  | 256 | 16,334,850 | 15,971,978 | 1.023x | marginally |
  | 512 | 35,518,286 | 34,172,745 | 1.039x | no |
  | 1024 | 82,200,148 | 78,092,117 | 1.053x | no |
  | 2048 | 172,405,781 | 162,137,193 | 1.063x | no |
  | 4096 | 360,529,316 | 335,632,540 | 1.074x | no |

  **`T_BLOCK = 8` is neutral at 64 and 128 and worth 4 to 7 percent at 512 and
  above, on the 48-layer arm, warm.** The conclusion does not depend on the
  drift verdicts: restricted to the three runs whose arm-B controls are clean
  (A1, A3, B1) it is the same shape, and with all six included the two
  populations do not overlap at any rung from 512 up. At 64 and 128 the ranges
  interleave and no effect is claimed; the 0.7% the wrong way at 128 is inside
  the noise and is not a regression.

  **Arm A is not quotable here**, and Note 1 is why: five of its six runs fail
  its own 512 drift control. Its 4096 rung read 4,556,288 (A1) against
  4,540,599 (B1), a 0.3% difference, quoted only to say that the effect on the
  cache-resident arm is not large.
- Verdict: KEEP, and the cold measurement is **owed**, not optional. Nothing
  here satisfies rule 2, so nothing here publishes. The change ships on three
  things instead: it regresses no rung, its bit identity is pinned by the
  gates in Note 3, and the stepped tail is a strict superset of the old sweep
  by construction rather than by benchmark (Note 4). The reason to keep it is
  that it is free at short context and positive at long; the reason not to
  quote 1.07x anywhere is that this is warm and rule 2 governs what ships.

  **Superseded in part by EXP-023 (2026-08-06):** the owed cold pair has been
  run and the verdict above stands. The numbers in this entry are unchanged
  and none of them is retracted. What changes is only the status of the debt.
  Cold, paired against the banked reference, medians of 3 scored runs:
  `T_BLOCK = 8` moves decode **1.44 to 1.46 tok/s at 3,961 prompt tokens
  (1.014x)** and is **not distinguishable at 512**, where the two arms' scored
  ranges overlap (1.94-1.98 against 1.91-1.98) and the medians run 3.5% the
  wrong way, which EXP-023 Note 8 states is not a regression. Prefill is
  unmoved at both rungs, 0.999x and 0.996x. **The warm 4-7% above remains the
  correct figure for the kernel**; what it is worth to a token is about 1.4% at
  4K context and nothing measurable at 512, because EXP-023 measures attention
  at 31.6% and 6.4% of decode at those two rungs and 4-7% of those shares is
  1.3-2.2% and 0.26-0.45%. So the two entries agree, and the sentence Note 6
  left as a placeholder is now answered: somebody has measured what it did to
  a token.
- Notes:
  1. **The drift control this bench leans on is close to useless at arm A's
     512 rung, and finding that out cost a wrong conclusion first.** The
     arm-A 512 control failed in **5 of 6 runs**, on both binaries, at
     1.093x-1.149x, with only B1 at 1.000x. An earlier reading of the same
     data called that failure systematic to `T_BLOCK = 4`, on the strength of
     three failures on the `T_BLOCK 4` arm against one clean run on the other,
     which is a tempting 3-vs-1 story. **B2 and B3 refuted it**: both are `T_BLOCK = 8`
     and both fail the same control at 1.094x and 1.095x. So it is an
     instrument property, not a property of the kernel under test, and as
     banded it discards most of arm A on a signal that is not about arm A. It
     should be re-examined before any future lane leans on arm A. This is why
     the Result above rests on arm B, whose controls fail in 3 of 6 and whose
     conclusion survives dropping those three.
  2. **The backlog's estimated 1.5-2x on the QK dot did not appear end to end,
     and it is superseded by measurement rather than merely unconfirmed.**
     EXP-020's backlog item carried "roughly 1.5-2x on the QK dot" as an
     **estimate** from the implementing lane. The whole-kernel measurement
     above is 1.00x to 1.07x. Both can be true and the gap is not a
     contradiction: the QK dot is one term of the kernel, and phase 7 already
     measured attention drifting **memory-bound** at long context. EXP-020
     Note 3 records `max(ns/pos) / min(ns/pos)` on the 48-layer arm going 1.03x
     to 1.40-1.57x as the arithmetic got roughly 10x cheaper and the memory
     traffic did not move. A gain that grows monotonically with context, 0% at
     64 to 7% at 4096, is exactly what a modest arithmetic win looks like
     underneath a memory-bound ceiling. What matters for the record is the
     status change, not the size: the number in the backlog was an estimate
     with nothing behind it, it has now been measured on the quantity that
     ships, and no future lane should re-quote the 1.5-2x. `docs/architecture.md`
     and `docs/handoff-phase8.md` are updated accordingly.
  3. **The bit-identity gates were not covering the code they were meant to
     cover, and that was found by breaking it on purpose.** Which rungs of the
     stepped sweep a call runs is decided entirely by `positions % T_BLOCK`,
     and the `T_TAIL_BLOCK` rung fires only for classes 4..=7. The dispatch
     length list was `[1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 31, 61]`, whose
     union of remainders is `{0, 1, 2, 3, 4, 5, 7}`: **class 6 was absent**,
     and poisoning the tail rung for `positions % 8 == 6` left both designated
     gates green. Three changes close it. `dispatch_lengths` now carries 6, 14,
     15, 17, 23, 24 and 25, so every remainder class and every block boundary
     with one either side is swept.
     `restructured_kernel_is_bit_identical_to_head_major_reference` was
     repointed off its own private length list onto `dispatch_lengths`, so
     there is one list to keep honest rather than two. And a new guard test,
     `dispatch_lengths_cover_every_position_block_remainder`, asserts the
     coverage property directly against `x86::T_BLOCK`, so widening the
     constant again fails *that* test with the dark class named instead of
     silently unpinning a rung. This is GOTCHA 7 recurring on the same file
     that produced it.
  4. **The stepped tail's safety is an exactness argument, not a benchmark
     result.** With the middle rung in, the sweep runs `⌊p/8⌋` eight-blocks,
     then at most one four-block, then `p mod 8 mod 4 = p mod 4` one-blocks,
     **exactly the number of scalar blocks `T_BLOCK = 4` ran, at every `p`**,
     while every position outside that tail sits in a block of four or eight
     instead of four. So no geometry can be worse than the shipped code, which
     is what makes this a strict superset rather than a trade. `if` rather than
     `while` for the middle rung is deliberate: the remainder after the
     eight-loop is below 8, so a second four-block is unreachable. A `const`
     assert pins both halves the argument needs, `T_BLOCK % T_TAIL_BLOCK == 0`
     (which is what collapses the scalar count) and `T_BLOCK <= 2 *
     T_TAIL_BLOCK` (which is what makes a single `if` enough), because
     `T_TAIL_BLOCK < T_BLOCK` alone is far too weak: 3 would pass it and leave
     a scalar tail of up to 4, worse than the code being replaced. The rung
     matters at real geometries rather than at contrived ones: prefill row `r`
     attends `start + r + 1` positions so every remainder occurs, and decode's
     fan-out gate opens at 8 cached positions.
  5. **The register and stack claims are read out of the shipping binary, and
     they are what bound the technique at eight.** In the linked release build
     the inlined `head_dim` loop of `x86::qk_scores` holds the eight
     accumulators in `ymm8`-`ymm15`, the transposed query vector in `ymm0` and
     the broadcast-and-product temporary in `ymm1`, with **no spill and no
     stack store at all in the loop body**. The same probe at 16 chains spills:
     16 accumulators plus two temporaries against 16 registers, and the loop
     grows 17 stack moves. So 8 is the ceiling of this technique rather than a
     midpoint. The cost is stack: the pair of buffers in `qk_scores` goes
     **12 KiB to 16 KiB**, taking the whole frame to **17,144 B** of `sub` plus
     48 B of callee-saved pushes, which crosses four guard pages and so emits
     **four inline stack probes** where it emitted three. That figure is read
     from the **linked** binary on purpose: the workspace release profile is
     `lto = "thin"` with `codegen-units = 1`, and a per-CU
     `cargo rustc --release --lib -- --emit asm` probe reports 16,936 B for the
     same function before LTO. If a later reader measures the smaller number,
     that is why. Zero-initialization of the buffers is real rather than
     elided: two `memset(_, 0, 8192)` calls per call, four kv heads, so 64 KiB
     zeroed at ~2000 cycles against a call that does milliseconds of work at
     full context.
  6. **EXP-023 is reserved for the cold rule-2 measurement of the phase-8
     decode work**, including the paired `T_BLOCK` arms this entry owes.
     It has not been run and this entry invents no number for it. Until it
     exists, the honest summary of `T_BLOCK = 8` is "no rung got slower, the
     long end got a few percent warm, and nobody has measured what that did to
     a token."

     **Correction (2026-08-06):** EXP-023 exists and the reservation is
     discharged. The quoted summary was written before the measurement and is
     superseded by the block at the end of the Verdict above; keep it here as
     the record of what was known when this entry shipped.

## EXP-023: The cold decode sweep: the phase split against context, the 11-slot hit rate, the slot dial and T_BLOCK

- Date / commit: 2026-08-06 / `c78122b` (`feat/decode`). Branch arm:
  `target/release/ramvamp` sha256
  `d36036b6485b00e741b7448e8d963e8a0916eb0aeb36b69c48c89ea857eb8b4c`, built by
  the sweep itself and asserted newer than every tracked source under
  `crates/`. Reference arm: `scratch/phase7-ref/ramvamp`, sha256
  `d56dc034ebd3e94e83b59ad64503adf289baf22af112752593ec59068a586e66`, which is
  the same executable EXP-021 measured (EXP-022 Baseline records why that
  matters), checked against its own `SHA256` sidecar and against a constant
  pinned in the sweep script.
- Hypothesis: four things are owed and one is offered. Owed: the paired cold
  `T_BLOCK` arms EXP-022 Note 6 reserved this number for; the hit rate at the
  shipped 11 slots/layer, which `docs/architecture.md` records as never
  measured; a slot-dial measurement, since the 12-slot row of the memory
  contract is a provisional prediction that says 12 does not fit; and a
  512-token cold phase split, which `docs/handoff-phase8.md` carries as
  uncaptured. Offered: decode's phase split as a **curve against context**
  rather than the single 4K point EXP-021 Note 9 left behind, because
  `docs/handoff-phase8.md` records that point as a seven-token post-prefill
  transient and warns a phase-8 lane not to choose a lever from it.
- Method: one unattended sweep, `bash scripts/phase8_decode_sweep.sh`, started
  2026-08-06 16:53:22 and finished 19:34:11, driven end to end by a committed
  harness. Logs and `SUMMARY.txt` in
  `scratch/phase8/sweep-20260806-165322/`; per-arm summaries in
  `scratch/cold-bench/p8-20260806-165322-*.json`; the io_probe result in
  `scratch/io-probe/p8-20260806-165322-decode-qd.json` and its `.md`.

  Nine cold arms, each `scripts/cold_bench.py --warmup 1 --repeats 3
  --max-new 64 --skip-hashes --greedy`: five context rungs on the branch
  binary at the shipped dial (64, 512, 1,024, 2,048 and 3,961 prompt tokens),
  two `T_BLOCK = 4` reference arms at 512 and 3,961, and three slot-dial arms
  (512 and 3,961 at `--cache-bytes 1570M`, 512 at `1701M`). Every arm evicts
  all 53 model files with `posix_fadvise(POSIX_FADV_DONTNEED)` and proves the
  eviction with `mincore`, then launches under `systemd-run --user --wait -p
  MemoryMax=3G -p MemorySwapMax=0 -p MemoryAccounting=yes` and reads
  `memory.peak`, `memory.events`, `memory.stat` and `/proc/self/io`
  `read_bytes` from inside the cgroup before exit. Before each arm the harness
  waits for `MemAvailable` to hold above 6,000 MiB for four consecutive
  samples; it settled in 45 s every time, at 11,089 to 11,310 MiB.

  **Rule-2 status, stated head on: all nine cold arms are `measurement
  hygiene: PASS`** and every one of the 36 runs returned 0 with
  `memory.swap.peak` 0 and every `memory.events` counter 0. Of the 36 runs, 35
  are CLEAN under the `4b39104` rule and the one exception is the 3,961 rung's
  **discarded warmup**, which is Note 13. All reclaim on the 35 clean runs is
  `pgsteal_khugepaged` with `pgsteal_kswapd`, `_direct` and `_proactive` all
  zero, 0 to 716 pages, which is the pattern EXP-021 Note 1 characterised.

  **The dial each arm actually got was asserted, not assumed.** A committed
  checker (`scratch/phase8/sweep-20260806-165322/check_slots.py`, self-tested
  6 of 6 before the sweep on its own fixtures) reads the slots/layer line out
  of every run's stderr and fails the step if any of the four runs disagrees
  with the label. All ten checks pass: 11 slots on the seven default arms, 12
  at `1570M`, 13 at `1701M`. A budget buying a different dial than the label
  claims is the failure this exists to catch.

  Prompts are fixed files with recorded sha256: 64 tokens from
  `scratch/phase8/prompts/ctx64.txt`, 512 from
  `models/llamacpp-ref/llamacpp_ref/long_00.txt` (`d1b6c407c55a...`, the same
  workload EXP-018 and EXP-021 used), 1,024 and 2,048 from
  `scratch/phase8/prompts/`, and 3,961 from `scratch/ctx4k/p4k.txt`
  (`68582aae37b9...`, EXP-021's 4K workload).

  One tenth step, `scripts/io_probe.py --block-ks 1 --fixed-k 1 --fixed-qd 8
  --queue-depths 1,2,4,8,16 --patterns rand,seq --repeats 3`, is a drive-side
  probe and is scoped in Note 10.
- Baseline: three, and they are kept apart on purpose.

  1. For `T_BLOCK`, the reference arm of each rung, run back to back with the
     branch arm on the same prompt with the same dials in the same session.
     That pair is paired, and rule 3 is satisfied inside this entry.
  2. For the slot dial, the 11-slot arm of the same context rung on the same
     binary in the same session.
  3. For the hit rate and the memory contract, the **predictions** in
     `docs/architecture.md`, which are arithmetic and simulation rather than
     measurements, so what follows corrects a prediction rather than
     contradicting a measurement.

  Figures from EXP-014, EXP-018, EXP-019 and EXP-021 appear below only where
  they are named as the prior being corrected or the caveat being carried.
  **None of them is put on a curve with anything measured here.**
- Result:

  **Headline, medians of 3 scored runs per arm.** `read_bytes` is the cgroup's
  own counter, and **the three scored runs of every arm agree on it to the
  byte**, which is what a deterministic greedy workload over an evicted cache
  should do. Two warmups read slightly more than their scored runs (589,824 B
  at ctx 64 and 16,384 B at ctx 2,048); the warmups are discarded and are not
  in these medians.

  | arm | ctx | slots | binary | prefill tok/s | decode tok/s | wall s | `memory.peak` MiB | `read_bytes` MiB |
  | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
  | decode-64 | 64 | 11 | branch | 4.10 | **2.19** | 47.420 | 2,507.9 | 44,703.0 |
  | decode-512 | 512 | 11 | branch | 11.25 | **1.91** | 80.798 | 2,601.0 | 48,191.3 |
  | decode-1024 | 1,024 | 11 | branch | 11.25 | **1.82** | 128.046 | 2,653.4 | 65,617.4 |
  | decode-2048 | 2,048 | 11 | branch | 10.82 | **1.75** | 229.908 | 2,760.6 | 95,539.8 |
  | decode-3961 | 3,961 | 11 | branch | 9.87 | **1.46** | 447.710 | 2,929.3 | 163,794.9 |
  | tblock4-512 | 512 | 11 | reference | 11.26 | 1.98 | 79.824 | 2,600.0 | 48,191.3 |
  | tblock4-3961 | 3,961 | 11 | reference | 9.91 | 1.44 | 445.782 | 2,928.9 | 163,794.9 |
  | slots-512-1570M | 512 | 12 | branch | 11.36 | 2.00 | 78.926 | 2,742.5 | 46,825.5 |
  | slots-512-1701M | 512 | 13 | branch | 11.42 | 2.06 | 78.072 | 2,862.3 | 45,630.5 |
  | slots-3961-1570M | 3,961 | 12 | branch | 9.95 | 1.52 | 442.543 | **3,058.4** | 162,539.4 |

  **The decode phase split against context**, the reason this entry exists.
  Every cell is that rung's **first scored run**, warmup excluded, and Note 1
  is why that convention matters and where it costs something. 63
  `forward_token` calls per run (`--max-new 64` costs `N - 1` instrumented
  calls). `other` is 0.00 s and 0.0% at every rung and is omitted.

  | ctx | decode s | ms/token | attention | expert compute | expert io | projections | elementwise |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 64 | 32.95 | 523 | 0.48 s (**1.5%**) | 8.67 s (26.3%) | 17.84 s (**54.1%**) | 5.10 s (15.5%) | 0.86 s (2.6%) |
  | 512 | 33.55 | 532 | 2.14 s (**6.4%**) | 9.73 s (29.0%) | 14.91 s (**44.4%**) | 5.90 s (17.6%) | 0.87 s (2.6%) |
  | 1,024 | 34.42 | 546 | 3.77 s (**11.0%**) | 9.10 s (26.5%) | 15.51 s (**45.1%**) | 5.16 s (15.0%) | 0.87 s (2.5%) |
  | 2,048 | 35.30 | 560 | 6.92 s (**19.6%**) | 8.86 s (25.1%) | 13.49 s (**38.2%**) | 5.14 s (14.6%) | 0.88 s (2.5%) |
  | 3,961 | 42.82 | 680 | 13.52 s (**31.6%**) | 9.02 s (21.1%) | 14.23 s (**33.2%**) | 5.18 s (12.1%) | 0.87 s (2.0%) |

  **Expert io is the largest single term at every rung measured, and its share
  falls from 54.1% at 64 tokens of context to 33.2% at 3,961 while attention's
  rises from 1.5% to 31.6%.** In seconds, expert io is flat (17.84 down to
  14.23 s over 63 tokens) and attention is what grows (0.48 to 13.52 s, 28x
  over a 62x context increase). Note 2 bounds the "largest at every rung"
  claim, which is tighter than it looks at 3,961.

  **The prefill phase split over the same ladder**, same runs, same
  convention. This is the 512-token cold split `docs/handoff-phase8.md`
  carries as uncaptured, plus four more rungs.

  | ctx | prefill s | attention | expert compute | expert io | projections | elementwise |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 64 | 15.61 | 0.09 s (0.6%) | 5.74 s (36.7%) | 5.10 s (**32.7%**) | 3.84 s (24.6%) | 0.85 s (5.4%) |
  | 512 | 45.83 | 2.83 s (6.2%) | 21.94 s (47.9%) | 1.26 s (2.8%) | 13.47 s (29.4%) | 6.34 s (13.8%) |
  | 1,024 | 91.00 | 8.22 s (9.0%) | 42.41 s (46.6%) | 2.46 s (2.7%) | 25.29 s (27.8%) | 12.62 s (13.9%) |
  | 2,048 | 189.34 | 25.54 s (13.5%) | 85.26 s (45.0%) | 4.86 s (2.6%) | 48.43 s (25.6%) | 25.25 s (13.3%) |
  | 3,961 | 402.12 | 83.46 s (20.8%) | 168.27 s (41.8%) | 9.95 s (2.5%) | 91.51 s (22.8%) | 48.93 s (12.2%) |

  At every rung from 512 up the ordering is **expert compute, then
  projections, then attention**, which is the ordering EXP-021 Note 9 reported
  at 4K in a different session (42.1 / 22.8 / 20.4 against 41.8 / 22.8 / 20.8
  here). Attention is third everywhere on this ladder, never second. Those are
  two entries and two sessions and they are not one curve; the agreement is
  quoted as a reproduction of an ordering, not of a number. The 64-token row
  is a different regime and is Note 3.

  **Decode cache statistics**, identical across all four runs of each rung to
  the request, because `--greedy` makes routing deterministic and the arms
  share a prompt. `io wait` is the first scored run's.

  | ctx | requests | hits | hit % | misses | cold | eviction | GiB read | io wait s |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 64 | 24,192 | 14,341 | **59.3** | 9,851 | 2,905 | 6,946 | 26.5 | 17.66 |
  | 512 | 24,192 | 13,074 | **54.0** | 11,118 | 3,106 | 8,012 | 29.8 | 14.74 |
  | 1,024 | 24,192 | 12,830 | **53.0** | 11,362 | 3,149 | 8,213 | 30.5 | 15.36 |
  | 2,048 | 24,192 | 14,129 | **58.4** | 10,063 | 2,650 | 7,413 | 27.0 | 13.34 |
  | 3,961 | 24,192 | 13,662 | **56.5** | 10,530 | 2,899 | 7,631 | 28.3 | 14.05 |

  24,192 is `63 tokens x 48 layers x top_k 8` exactly, which is the arithmetic
  that says the instrumented window is the whole decode phase minus its first
  token.

  **The slot dial.** Scored range is the full range of the three scored runs.

  | ctx | slots | `--cache-bytes` | decode tok/s (median) | scored range | hit % | `read_bytes` MiB | `memory.peak` MiB | under 3,072 |
  | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
  | 512 | 11 | default | 1.91 | 1.91-1.98 | 54.0 | 48,191.3 | 2,601.0 | 471.0 |
  | 512 | 12 | 1570M | 2.00 | 1.96-2.03 | 56.1 | 46,825.5 | 2,742.5 | 329.5 |
  | 512 | 13 | 1701M | 2.06 | 2.05-2.08 | 57.9 | 45,630.5 | 2,862.3 | 209.7 |
  | 3,961 | 11 | default | 1.46 | 1.44-1.49 | 56.5 | 163,794.9 | 2,929.3 | 142.7 |
  | 3,961 | 12 | 1570M | 1.52 | 1.51-1.54 | 58.4 | 162,539.4 | **3,058.4** | **13.6** |

  **12 slots/layer fits at 3,961 tokens of context**, at 3,058.4 MiB with 13.6
  MiB spare, hygiene PASS, no OOM, on all four runs. `docs/architecture.md`
  predicts 3,091.82 MiB and **19.8 MiB over**. Note 7 works out where the
  33.4 MiB of overprediction goes and why the shipped dial still does not
  move.

  **The paired `T_BLOCK` arms**, which discharge what EXP-022 Note 6 reserved
  this entry for. Both arms of a rung ran back to back on the same prompt with
  the same dials, so the ratio is `T_BLOCK` and nothing else.

  | rung | metric | `T_BLOCK` 4 | `T_BLOCK` 8 | 8/4 |
  | ---: | --- | ---: | ---: | ---: |
  | 512 | decode tok/s | 1.98 | 1.91 | 0.965x |
  | 512 | prefill tok/s | 11.26 | 11.25 | **0.999x** |
  | 512 | wall s | 79.824 | 80.798 | 1.012x |
  | 3,961 | decode tok/s | 1.44 | **1.46** | **1.014x** |
  | 3,961 | prefill tok/s | 9.91 | 9.87 | **0.996x** |
  | 3,961 | wall s | 445.782 | 447.710 | 1.004x |

  Every scored run, because the medians alone are misleading at 512:

  | rung | `T_BLOCK` 4 scored | `T_BLOCK` 8 scored |
  | ---: | --- | --- |
  | 512 | 1.98 / 1.94 / 1.98 | 1.91 / 1.98 / 1.91 |
  | 3,961 | 1.44 / 1.44 / 1.43 | 1.49 / 1.44 / 1.46 |

  **No end-to-end effect is distinguishable at 512, and the effect at 3,961 is
  about 1.4%.** At 512 the two ranges overlap at 1.98 and the median gap runs
  the wrong way by 3.5%, which is not a regression claim and must not be
  quoted as one (Note 8). At 3,961 the ranges touch at 1.44 and every other
  `T_BLOCK = 8` run is at or above every `T_BLOCK = 4` run. Prefill is
  unmoved at both rungs, 0.999x and 0.996x, which is the control this pair
  needed: `T_BLOCK` is an attention constant, prefill runs the same attention
  kernel, and a prefill ratio that moved would mean something other than
  `T_BLOCK` moved.

  **The drive-side single-blob queue-depth curve**, K=1, which is the block
  size decode actually issues. Medians of 3 scored runs after 1 discarded
  warmup, GB/s = 10^9 B/s, hygiene PASS. **This is `threaded-pread`, not
  io_uring** (Note 10).

  | QD | layer_00 seq | layer_20 seq | layer_06 seq | layer_21 seq | layer_00 rand | layer_20 rand | layer_06 rand | layer_21 rand |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1 | 1.370 | 1.821 | 1.594 | 1.865 | 1.380 | 1.899 | 1.346 | 1.798 |
  | 2 | 1.602 | 3.387 | 2.409 | 3.163 | 1.599 | 3.265 | 1.627 | 3.198 |
  | 4 | 1.602 | 3.470 | 2.434 | 3.521 | 1.598 | 3.415 | 1.676 | 3.475 |
  | 8 | 1.591 | 3.462 | 2.417 | 3.533 | 1.568 | 3.455 | 1.654 | 3.469 |
  | 16 | 1.596 | 3.430 | 2.446 | 3.485 | 1.580 | 3.413 | 1.697 | 3.455 |

  `filefrag` reports the same extent geometry as EXP-019 recorded: 398 / 398 /
  398 / 390 extents, mean 984,027 / 984,027 / 853,614 / 871,124 B, zero
  physically adjacent pairs and **zero compressed extents** on all four files.
  `btrfs device stats` is unchanged across the session at its 138,407
  `corruption_errs` baseline with all four other counters 0.
- Verdict: KEEP. One shipped change is gated here and it stays: `T_BLOCK = 8`
  costs nothing at 512 and is worth about 1.4% at 3,961 cold, which is what
  EXP-022's warm 4-7% predicts once attention's share of a token is applied
  (Note 8). Everything else in this entry is characterisation that changes no
  dial: **the shipped default stays 11 slots/layer and `--cache-bytes` is
  unchanged.** What the entry does change is the status of four figures in
  `docs/architecture.md`, from predicted to measured, and one of them changes
  sign.
- Notes:
  1. **Every split in the Result comes from its rung's first scored run, not
     from its median run, and at ctx 64 that costs something worth naming.**
     `cold_bench.py` reports medians per metric, so no single run is "the
     median run", and a split is a decomposition of one run's wall rather than
     a set of independently medianable numbers. Taking the first scored run
     everywhere is the convention that keeps each column of the split table
     internally consistent. At ctx 64 the first scored run is **1.94 tok/s
     against a 2.19 median and is the slowest of the three** (1.94 / 2.23 /
     2.19), so its 54.1% expert io is the slow run's share and not the median
     run's: the third scored run of the same rung reads 44.1% expert io over a
     29.14 s decode. The full scored spreads and where the chosen run sits in
     each, so a reader can price the convention rather than trust it:

     | ctx | scored decode tok/s | median | spread | first scored run is |
     | ---: | --- | ---: | ---: | --- |
     | 64 | 1.94 / 2.23 / 2.19 | 2.19 | 1.149x | the **slowest** |
     | 512 | 1.91 / 1.98 / 1.91 | 1.91 | 1.037x | tied slowest |
     | 1,024 | 1.86 / 1.81 / 1.82 | 1.82 | 1.028x | the fastest |
     | 2,048 | 1.81 / 1.57 / 1.75 | 1.75 | 1.153x | the fastest |
     | 3,961 | 1.49 / 1.44 / 1.46 | 1.46 | 1.035x | the fastest |

     So two rungs carry a 15% spread (64 and 2,048) and three sit inside 4%,
     and the convention picks the slowest run at 64 and the fastest at 2,048.
     Neither of those is the median run, and no split in this entry should be
     read as one. The 2,048 outlier is its second scored run, whose `io wait`
     jumped to 17.49 s from the 13.34 s the other two recorded on identical
     byte counts, which is the same kind of variance the 64 rung shows.
     One consequence for a reader checking this entry against the raw
     data: `SUMMARY.txt` in `scratch/phase8/sweep-20260806-165322/` renders the
     **last** scored run's split, not the first, so its per-rung split blocks
     will not match the table above cell for cell. Both are in the JSONs;
     `runs[].stderr` carries all four.
  2. **"Expert io is the largest single term at every rung" is true on the
     first-scored-run reading and is inside the run-to-run spread at 3,961.**
     At 64, 512, 1,024 and 2,048 the margin over the next term is large (54.1
     against 26.3, 44.4 against 29.0, 45.1 against 26.5, 38.2 against 25.1) and
     survives every scored run of those rungs, including the slow ones Note 1
     tabulates. At 3,961 it is 33.2% against attention's
     31.6% in the first scored run, and **the ordering inverts in the other
     two**: run 2 reads attention 34.2% against expert io 31.7% and run 3 reads
     33.5% against 32.8%. So the honest statement at the long end is that the
     two terms have **crossed, or are crossing**, and no run separates them by
     more than about 2.5 points. Nothing here licenses "expert io dominates at
     4K"; what it licenses is "expert io dominates below 2,048 and is level
     with attention at 3,961".
  3. **The 64-token rung is a different regime in prefill and should not be
     read as a point on the prefill curve.** Its prefill is 4.10 tok/s against
     11.25 at 512 and its expert io is 32.7% of prefill against 2.8%, because
     the layer-major sweep reads every expert of every layer once per chunk
     regardless of how many tokens are in the chunk: 768 windows and 16.3 GiB
     at 64 tokens is the same 768 windows and 16.3 GiB as at 512 tokens. The
     fixed cost is amortized over 8x fewer tokens, which is the sweep working
     as designed and not a finding. It is tabulated because omitting a rung
     from a curve is worse than labelling it.
  4. **The `expert io` bucket is a residual and understates how long the drive
     is busy, and this sweep pins that reading again.**
     `stage_expert_phases` (`crates/core/src/model/forward.rs:1565-1591`) runs
     the hit plan, charges it to `expert compute`, and only then blocks in
     `await_misses()`, which is the sole contributor to `expert io`. The miss
     reads were submitted by `begin_layer` and are in flight throughout that
     hit compute, which is the entire point of the two-phase shape. The
     coincidence that pins it: at 3,961 the streamer's independently counted
     `io wait` is **14.05 s** against the split's **14.23 s** `expert io`
     bucket, a 1.3% gap, and the same pairing holds at every rung (14.74
     against 14.91 at 512, 15.36 against 15.51 at 1,024, 13.34 against 13.49 at
     2,048, 17.66 against 17.84 at 64). So the bucket is that block and nothing
     else. **Anyone sizing an I/O lever off the shares in this entry is sizing
     it off a lower bound on drive-busy time.** This restates a finding
     `docs/handoff-phase8.md` derived from code structure; what is new is that
     it now holds on five measured rungs rather than one.
  5. **The hit rate at the shipped 11 slots/layer is measured for the first
     time: 53.0% to 59.3% across the ladder, with no trend in context.**
     `docs/architecture.md` (lines 281-283 at the time of writing) states that
     it "has not been measured" and brackets it by 50.02% and 54.48%. **The
     measurement lands at or above the top of that bracket at three of the five
     rungs** (59.3, 58.4 and 56.5 against a 54.48% ceiling), and the other two
     land inside it and near its top (54.0 and 53.0). **No rung falls below the
     bracket's floor.** The sequence is not monotone in context, so it is a
     scatter around roughly 56% rather than a curve. Two
     provenance corrections belong with that, because getting them backwards
     would overstate the agreement. First, the bracket's endpoints are **not**
     simulator output: 50.02% at 10 slots and 54.48% at 12 come from replaying
     the shipped `io/cache.rs` over EXP-005's four routing traces. It is
     `scripts/lfu_sim.py` that gives 44.8% and 49.9%, and the documented gap of
     roughly 5 points is between the simulator and that replay, so the
     simulator is the ~5-point underestimate and the bracket is already
     corrected for it. Second, the bracket is a trace replay over 556 decode
     tokens of recorded routing and this is a live decode of 63 tokens per run
     on five different prompts, so they are different populations and the
     agreement is in level and order rather than like for like. The bracket
     should be recorded as superseded at 11 slots, not as confirmed.
  6. **One extra slot buys about 2 points of hit rate and about 4% of decode
     tok/s, and only two of the three measured steps separate at three runs
     each.** 11 to 12 is +2.1 points at 512 (54.0 to 56.1) and +1.9 at 3,961
     (56.5 to 58.4); 12 to 13 at 512 is a further +1.8 (56.1 to 57.9). In
     throughput the medians give 1.047x at 512 and 1.041x at 3,961 for 11 to
     12, and 1.079x at 512 for 11 to 13. **The 512 rung's 11-to-12 step does
     not separate**: scored ranges 1.91-1.98 against 1.96-2.03 overlap. The
     3,961 rung's does (1.44-1.49 against 1.51-1.54) and so does 512's
     11-to-13 (1.91-1.98 against 2.05-2.08). `read_bytes` falls monotonically
     with the dial at both rungs, which is the mechanism and is not subject to
     the same spread: 48,191.3 to 46,825.5 to 45,630.5 MiB at 512 and 163,794.9
     to 162,539.4 MiB at 3,961. For the record and not as a curve,
     `docs/architecture.md` puts 10 to 12 at +4.46 points from the replay,
     which is the same ~2 points per slot this measures.
  7. **The memory contract's 12-slot row is wrong in sign, and the error is a
     constant rather than a slope.** Measured at 3,961 prompt tokens plus 64
     generated, 12 slots/layer peaks at **3,058.4 MiB**, which is 13.6 MiB
     **under** the 3,072 MiB cap, against a predicted 3,091.82 MiB and 19.8 MiB
     **over**: a 33.4 MiB overprediction. The same rung at the shipped 11 slots
     peaks at 2,929.3 MiB against a predicted 2,961.03, a 31.7 MiB
     overprediction. **Two nearly equal overpredictions one slot apart is the
     signature of a wrong constant in the fixed-tenant sum, not of wrong
     per-slot arithmetic**, and the per-slot arithmetic corroborates that
     directly: the table's 130.79 MiB per slot against a measured 129.05 MiB
     between the two 3,961 arms (and 141.48 and 119.83 MiB for the two 512
     steps, which bracket it). **Derived**, and only partly: about 6.7 MiB of
     the 33.4 is the KV cache, which is allocated at the 4,096-position
     capacity and faulted lazily, so at the 4,025-position high-water mark it
     holds 377.34 MiB of its 384 MiB row. That leaves roughly 27 MiB
     unaccounted, and the only provisional row in the sum is the 115.1 MiB of
     runtime anonymous memory, which `docs/architecture.md` already records as
     failing rule 2 and as a floor rather than a ceiling. So the licensed
     statement is "the fixed-tenant sum is about 27 MiB high and the anon row
     is where to look", **not** "the anon row is 82 MiB". Its re-measurement,
     which EXP-012 asks for, is still owed and this entry does not take it.
     **The dial does not move on this.** 13.6 MiB of margin at 4K is smaller
     than EXP-018's unexplained 99-105 MiB residual, smaller than the 33.4 MiB
     this note is correcting, and measured on one prompt in one session; and
     the 13-slot arm has **no 4K run at all**, so there is no measurement that
     could support going past 12 either.
  8. **EXP-022's warm 4-7% is not contradicted by an end-to-end 1.4%; it is
     what an end-to-end 1.4% predicts once attention's share of a token is
     applied.** This entry measures attention at **6.4% of decode at 512** and
     **31.6% at 3,961**. A kernel gain of 4-7% on that term alone predicts
     0.26-0.45% end to end at 512 and 1.3-2.2% at 3,961. Measured: nothing
     separable at 512, and 1.4% at 3,961. Both rungs land where the
     decomposition says they should, so the two entries agree and neither
     needs discounting. Two things must not be read out of the 512 row. It is
     **not a regression**: the medians differ by 3.5% the wrong way, but the
     `T_BLOCK = 8` range of 1.91-1.98 sits inside the `T_BLOCK = 4` range of
     1.94-1.98 and the two share their top value, so three runs an arm cannot
     order them. And it is
     **not a null result about the kernel**: 6.4% of a token is too small a
     term for a 4-7% change in it to clear this instrument's spread, so the
     512 rung has no power to detect what EXP-022 measured. The honest summary
     that replaces EXP-022 Note 6's placeholder is: `T_BLOCK = 8` costs
     nothing anywhere measured, is worth about 1.4% of a token at 4K context,
     and the warm 4-7% remains the correct figure for the kernel rather than
     for a token.
  9. **The reference arm reproduces EXP-021's machine, which is the one
     cross-entry check this sweep licenses and it is a check rather than a
     curve.** The `T_BLOCK = 4` arm is byte-identical to EXP-021's binary, so
     a disagreement between its numbers here and EXP-021's is a statement about
     the two sessions. At 3,961 prompt tokens it reads prefill **9.91 tok/s**
     here against EXP-021's 9.89 and 9.86 across two sessions, a 0.5% spread
     over three sessions a day apart. `read_bytes` at 512 is 50,532,257,792 B
     on every one of the eight runs of both 512 arms. The machine is
     reproducing. This licenses nothing else: EXP-021's decode figures were
     taken at `--max-new 256` and `--max-new 8` against this entry's
     `--max-new 64`, and its 4K `memory.peak` of 2,920.4-2,924.2 MiB is a
     different token count from this entry's 2,929.3 MiB. Those are not one
     series.
  10. **The queue-depth curve is `threaded-pread`, not io_uring, so it
      characterises the drive and the filesystem and not the runtime's
      submission path.** Queue depth is emulated with N OS threads each issuing
      a blocking `preadv`; the runtime submits through io_uring with
      `SINGLE_ISSUER` and `DEFER_TASKRUN`. This is the same bound EXP-019 Note
      5 states and it is restated rather than inherited, because this entry
      finally supplies the missing half of what EXP-019 licensed: EXP-019 swept
      queue depth at K=8 only, so the decode operating point had **no
      queue-depth curve of its own**, and this is it. What it does not supply is
      the io_uring measurement, so `RING_ENTRIES` still must not move on the
      strength of a probe. `docs/handoff-phase8.md` separately closes the
      submission side by geometry: `RING_ENTRIES` is 8, `top_k` is 8, and a
      decode layer submits at most 8 reads, so every miss a step can have
      already fits the ring.
  11. **Per-file bandwidth spread at the single-blob size is 2.21x, and
      `layer_00` is the file that does not move.** At K=1, QD 8, random, the
      four files read 1.568, 3.455, 1.654 and 3.469 GB/s, a 2.21x spread
      (2.22x sequential). `layer_20` and `layer_21` roughly double from QD 1 to
      QD 2 (1.821 to 3.387 and 1.865 to 3.163) and plateau at 3.4-3.5 GB/s;
      `layer_06` gains half again (1.594 to 2.409) and plateaus near 2.42;
      `layer_00` gains 17% (1.370 to 1.602) and then sits at 1.59-1.60 across
      the entire sweep. Extent geometry is identical between `layer_00` and
      `layer_20` to the byte (398 extents, mean 984,027 B, median 884,736 B,
      zero adjacent pairs), so fragmentation as `filefrag` reports it still
      does not predict it, exactly as EXP-019 Note 7 found. **Two things must
      not be inferred from putting this beside EXP-019.** EXP-019's spread of
      1.44x is `layer_00` at 1.578 against `layer_20` at 2.271 at K=1 QD 8
      **sequential**, not at K=8 as it is easy to misread; and its
      decode-shaped cell (K=1, random, QD 8) read 1.55-1.69 GB/s across all
      four files where this entry reads 1.57-3.47, so **the two fast files
      roughly doubled between the two sessions on the same cell**. That is a
      large session-to-session difference on a drive whose per-file behaviour
      both entries record as unexplained. Rule 3 forbids one curve through the
      two, and the correct reading is that each entry's spread is a fact about
      its own session.
  12. **Decode is not queue-starved; it is dragged by the slow files.**
      **Derived** from the measured counters. Effective rate: 28.3 GiB of
      expert reads against 14.05 s of `io wait` at the 3,961 rung is **2.16
      GB/s** (2.01 GiB/s), and because reads are in flight during hit compute
      the drive's true average delivery rate over the window it was busy is
      **at most** that, so 2.16 GB/s is an upper bound rather than a point
      (Note 4). Concurrency: 10,530 misses over `63 tokens x 48 layers` is
      **3.48 misses per layer step**, and since `begin_layer` submits every
      miss of a step at once and refuses to open the next step while any read
      is outstanding, that average **is** decode's queue depth. The other four
      rungs give 3.26, 3.68, 3.76 and 3.33 by the same arithmetic, so the
      operating point is between QD 3 and QD 4 at every context measured. The
      curve in Note 11 is already at its plateau by QD 2 on three files and by
      QD 4 on the fourth, so **more queue depth is not available to buy**: it
      would take more concurrent misses, which needs the cross-layer prefetch
      CLAUDE.md forbids. Meanwhile 2.16 GB/s sits between the slow files'
      1.57-1.65 and the fast files' 3.46-3.47 in the same K=1, random, QD 8
      cell, which is where an aggregate over 48 files of both kinds should sit.
      The lever this points at is the per-file spread, not the queue.
  13. **The 3,961 rung's discarded warmup recorded `pgsteal_kswapd` 2,817, and
      EXP-014's discarded first attempt recorded the same integer; nobody has
      explained the coincidence.** Recorded as unexplained and
      reproducible-looking, not as a diagnosis. What is measured: run 0 of
      `p8-20260806-165322-decode-3961.json` is `hygiene: DIRTY` with
      `pgscan 2817` and `pgsteal 2817`, **all of it `pgsteal_kswapd`** and none
      of it khugepaged, so it is genuine pressure under the `4b39104` rule and
      not the bookkeeping that rule was written to excuse. It was the warmup,
      so it is discarded and no number in this entry rests on it; the arm's
      three scored runs are CLEAN and the arm is PASS. What is odd: EXP-014
      records a discarded first attempt whose two DIRTY runs read `pgsteal`
      **2,817 and 2,946** pages, and 2,817 is the same integer two days and one
      workload apart (EXP-014 is dated 2026-08-04 and this sweep 2026-08-06).
      It is **not** the same rung, and saying so matters:
      EXP-014's prompt was five tokens ("The capital of France is") on a
      different binary at a different commit, where this is a 3,961-token
      prompt. So the two share a number and nothing else. EXP-014 attributed
      its own two to the operator opening a terminal mid-run, and this one has
      no such cause recorded: the sweep was unattended and its settle loop had
      just read `MemAvailable` at 11,101 MiB. EXP-021 Note 10 already records
      that EXP-014's JSONs were overwritten, so the classification of its two
      cannot be rechecked and this coincidence cannot be chased backwards.
      What it would take to make this a finding rather than a curiosity is a
      third occurrence with its counters kept, and the exact-integer repeat is
      the reason to keep them.
  14. **What this entry does not settle.** Three scored runs per arm is a
      spread control, not an error bar, and Note 6 shows one of the five dial
      comparisons failing to separate under it. Every arm is `--max-new 64`, so
      the decode splits cover tokens 2 to 64 after a sweep-emptied cache and
      the post-prefill transient `docs/handoff-phase8.md` identifies is inside
      that window rather than excluded from it; the split curve is therefore a
      curve of "the first 63 tokens after a prompt", which is the decode a
      short reply consists of and is not steady state at 256 tokens and beyond.
      The 13-slot arm has **no 4K run**, so nothing here says whether 13 fits
      at full context, and Note 7 says why the 12-slot fit is not by itself a
      licence to move the dial. The anon-row re-measurement EXP-012 asks for is
      still owed and Note 7 narrows it rather than taking it. No numerics gate
      was run in this sweep: `T_BLOCK = 8`'s bit identity rests on EXP-022 Note
      3's tests and on EXP-021's gates, and this entry adds nothing to it.
      And the io_uring queue-depth measurement inside the runtime, which
      EXP-019 Note 5 called for, is still not taken (Note 10).
