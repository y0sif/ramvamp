# Experiment log

Every performance-motivated change gets a numbered entry here before it ships.
This discipline is borrowed from TurboFieldfare's 103-entry experiment record,
which is the reason their claims are credible.

## Rules

1. A microbenchmark starts an experiment; end-to-end speed and output quality
   decide whether it ships.
2. Cold measurements only for published numbers: run inside the benchmark
   cgroup (`memory.max=3G`, `memory.swap.max=0`; zram counts as swap) with a
   dropped page cache.
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
- Verdict: KEEP (measurement stands; no bug indicated)
- Notes: per-position full-vocab KL of O(1e-2) is the floor for any two
  implementations of this 48-layer quantized stack that do not replicate
  arithmetic operation-for-operation; a 1e-3 mean is unachievable without
  operation-identical kernels. Recommendation (decision pending): revise
  gate 3 to mean full-vocab KL <= 3e-2 with the intra-engine scalar/AVX2
  A/B recorded alongside as the noise floor; perplexity (gate 5) remains
  the quality backstop. Reference capture also produced per-position
  top-5000 dumps along 64-token greedy paths (`path_*.npz`) and 128-token
  greedy texts (`greedy_texts.json`) for phase-5/6 regression fixtures.
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
