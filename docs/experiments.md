# Experiment inventory

The public record of what ramvamp measured: one row per experiment, in ID
order. It is curated from a fuller internal log kept during development, and
every row here cites the entry it came from by ID. Those IDs are stable, so a
figure quoted anywhere in this repository can be traced to the entry that
produced it.

Raw logs, run artifacts, harness output and working notes are not part of this
record and are not published. The internal log argues with itself, withdraws
numbers and records reasoning that was later refuted; what survives that
process is here, along with the corrections that produced it.

Every performance figure comes from one reference machine and one drive: Core
Ultra 9 185H, DRAM-less Micron 2400 QLC (`MTFDKBA1T0QFM-1BD1AABGB`), btrfs.
Throughput is a property of that drive as much as of the code. The only work
taken elsewhere is the llama.cpp reference capture behind EXP-003 and EXP-004,
which is a correctness comparison and carries no timing. The v0 model is
Qwen3-30B-A3B: 48 layers, 128 experts per layer, top-8 routing, 4,096-token
context cap.

Verdicts: **Shipped** (the change, or the conclusion it drove, is in the v0
runtime), **Rejected** (measured and not adopted), **Superseded** (a later
entry replaced its number or its premise), **Withdrawn** (the entry's own claim
was retracted for want of evidence), **Open** (measured, nothing rests on it
yet).

## Measurement rules

No row below is interpretable without these.

1. A microbenchmark may start an experiment. End-to-end speed and output
   quality decide what ships.
2. **Publication.** Only cold, in-cgroup, hygiene-PASS numbers are publishable.
   Everything else is a diagnostic and is labelled one in its row. Correctness
   measurements are exempt. *Cold*: every model file evicted with
   `posix_fadvise(POSIX_FADV_DONTNEED)`, the eviction proven with `mincore`
   rather than trusted from the return code, and non-zero block-layer
   `read_bytes` as a positive control that the run really was cold.
   *In-cgroup*: `systemd-run --user` at `MemoryMax=3G` and `MemorySwapMax=0`,
   with counters read from inside the cgroup before exit; zram counts as swap.
   *Hygiene PASS*: the inner wrapper resolved its own cgroup, the pressure
   reclaimers stole zero pages, every `memory.events` counter is zero,
   `memory.max` equals the requested limit, swap peak is zero, `read_bytes` is
   non-zero, and both processes exited 0. An unreadable counter fails, which is
   why an unconfined run can never score clean.
   *khugepaged amendment*: huge-page compaction is subtracted from the reclaim
   total and reported as a soft note rather than a failure. The rule subtracts
   rather than sums, so a reclaimer nobody has heard of still counts as
   pressure and a `memory.stat` with no breakdown still fails.
3. **Comparison.** Every entry records its own baseline. Figures from different
   entries, sessions or machine states are never drawn on one curve. A ratio is
   quotable only when both arms ran back to back, on the same prompt at the
   same dials, in one session.
4. A change claiming identical math must produce identical bits. A change that
   reorders floating-point work must pass tolerance tests against reference
   outputs.
5. Negative results get entries too. They are the cheapest way to stop a bad
   idea coming back.

## Inventory

| ID | What it tested | Key evidence | Verdict |
| --- | --- | --- | --- |
| EXP-001 | AVX2 K-quant dot kernels against the scalar reference | Warm, single-thread microbenchmark on a synthetic 2048-row packed matrix, medians of 31 runs; diagnostic, not publishable. Per output row: q4_k 456 to 120 ns (3.80x), q5_k 449 to 141 (3.18x), q6_k 533 to 156 (3.43x), q8_0 918 to 179 (5.13x). A second run put q4_k AVX2 at 141 ns, so the k-quants read as 3.2-3.8x. Integer results bit-identical to scalar; only float accumulation order differs, tolerance-tested. | Shipped: runtime AVX2+FMA dispatch, 1-byte alignment contract |
| EXP-002 | AVX2 activation quantizers | Warm, single-thread, one 2048-float row per timed run; diagnostic. `quantize_row_q8_k` 11,960 to 1,065 ns/row (11.23x), `quantize_row_q8_0` 8,324 to 1,875 ns/row (4.44x). Output byte-identical to the scalar reference on random, tie-heavy, flat and zero rows. | Shipped: matches the scalar reference deliberately, not ggml's differing AVX2 rounding |
| EXP-003 | The forward pass against llama.cpp b10217 on identical Q4_K_M bytes | Correctness, exempt from rule 2. Greedy 16-token completions character-identical on 3/3 prompts; top-1 agreement 2/2; top-20 overlap 20/20 and 19/20. Its KL figures (0.0218 and 0.1007) were single-position, top-20-truncated and renormalized, and are not comparable to a full-vocab target. | Superseded: the truncated KL proxy replaced by EXP-004 |
| EXP-004 | Full-vocab KL against llama.cpp reference dumps, and the float-reordering noise floor | Correctness, exempt from rule 2. All 151,936 logprobs per prompt compared in f64 over 8 prompts: mean KL(P‖Q) 1.04e-2, worst prompt 2.72e-2, top-1 8/8. Intra-engine AVX2-against-scalar noise floor is 4.5e-3 to 1.3e-2, the same order as the cross-engine gap, and non-systematic in direction. KL is flat in context depth: 8.03e-6 at 512 tokens, 1.52e-2 at 1,891, 5.00e-3 at 3,492, so no position-compounding bug. Tightest top1-to-top2 margin 0.228 nats. | Shipped: gate 3 is now mean <= 3e-2, every prompt <= 6e-2, top-1 on every prompt, all 8 prompts scored |
| EXP-005 | Expert-cache policy and slot sweep on measured routing traces | Simulation over 556 decode tokens of recorded routing (four traces, 213,504 accesses); no cold or cgroup rules apply and no simulated hit rate is quoted directly. The LFU win comes from eviction-surviving per-expert counters, not from LFU: per-slot LFU is -1.7 to 0.0 points against LRU (tied at 42.6% at 10 slots, 55.4% against 57.1% at 16). Belady offline-optimal is 55.8% at 10 and 72.0% at 16. Replaying the shipped cache the way the runtime calls it, pinning a whole step, gives 50.02% at 10 slots and 54.48% at 12; driven one expert at a time it reproduces the simulator exactly, which is why simulator figures run about 5 points low. A global slot pool is worth +0.53 points and replaying the prompt into the cache +0.09. | Shipped: expert-indexed ghost LFU, 512 B per layer; global pool and prompt replay rejected |
| EXP-006 | Whether the buffered-`pread` decode path can be measured cold inside a 3G cgroup | It cannot. A 13-token generation pulled 7,836.5 MiB through the block layer, held the cgroup at its 3,072 MiB ceiling for the whole run and forced 1,263,046 pages of reclaim (~4.82 GiB), with `memory.events max` 6,203. Every figure from such a run measures reclaim, not decode. A separate run recorded `memory.events max` 0 while `pgsteal` showed 2.4 GiB silently reclaimed. | Shipped: the hygiene gate and its full CLEAN definition, motivated here |
| EXP-007 | Slot aliasing under concurrent O_DIRECT reads | Qualitative only: a probe on a busy machine, uncgrouped, harness never committed, so no rate from it is quotable. Aliasing two in-flight reads onto one destination buffer makes btrfs fail checksum verification, producing spurious `EIO` and matching increments to the filesystem's persistent corruption counter. An explicit free list handing out exclusive leases measured **0 `EIO` across 4,000 reads** at QD 4/8/16/32. Second finding: btrfs runs direct reads with page faults disabled and silently completes through the buffered path when it cannot fault the destination, returning a full byte count and no error. | Shipped: owning slot guards with no by-index accessor; every pool page faulted at construction |
| EXP-008 | Reference-drive characterisation under O_DIRECT: queue depth, block size, per-blob latency, `ReadFixed` | Warm, contended, uncgrouped, harness never committed; provisional throughout. At the 2.918 MiB expert stride: 1.211 GB/s at QD4, 1.349 at QD8, 1.390 at QD16. A block-size sweep read ~1.35 GB/s at the stride rising to 1.86, 2.04 and 2.15 GB/s at 8, 16 and 24 MiB. `ReadFixed` saves ~70 us of CPU per 3 MiB read. The entry records two contradictions between its own two probe series rather than smoothing them. | Superseded: level and block-size premise refuted by EXP-019. `ReadFixed` rejected against pinning 1.4-1.6 GiB under an 8 MiB `RLIMIT_MEMLOCK` |
| EXP-009 | Buffered against O_DIRECT expert reads, page-cache charge inside the cgroup | A memory measurement rather than a timing; machine not quiet. The identical 1.4 GiB of expert reads peaked the cgroup at **1,092.2 MiB buffered against 5.0 MiB with O_DIRECT**. The O_DIRECT side was not timed, so this says nothing about its throughput cost. | Shipped: O_DIRECT is a budget-correctness requirement, not a performance preference |
| EXP-010 | Compute-pool signalling: what a per-GEMV handoff can afford | Warm, uncgrouped diagnostic; two probes that are not comparable to each other. Uncommitted probe: a futex wake/wait pair p50 3.1 us at 0.07 cores of steady-state overhead, pure atomic spinning 502 ns at 1.03 cores, `std::sync::mpsc` p99 237 us. Committed whole-pool probe, 20,000 timed calls after 2,000 warmup: the shipped 64-round bounded spin then futex measures p50 1.71 us publish to barrier, which times the spin path and not the futex path. | Shipped: bounded spin then futex; `mpsc` rejected as disqualifying for a per-GEMV barrier |
| EXP-011 | Row-range GEMV as the single code path | Rule-4 identical-output test, no timing and none claimed. Row-partitioned results bit-identical (`f32::to_bits`) to the unpartitioned result over 8 partition schemes on 10 real model shapes, on both the AVX2 and the scalar path. | Shipped: enabler; whole-matrix entry points became thin wrappers |
| EXP-012 | Whether the memory contract's three tenants account for everything charged to the cgroup | They do not. Peak anonymous memory sampled inside the cgroup during a live decode: **115.1 MiB, provisional** (the run's context length, token count and page-cache state were not recorded, so it fails rule 2). Against the audited tenants (1,023.34 MiB mmap'd common core, 384 MiB FP16 KV at 4K, pool at the real per-layer strides), 11 slots/layer subtotals 2,961.03 MiB and 12 slots 3,091.82 MiB. Provisional label applies to the subtotals too, since they are arithmetic on it. | Shipped: dial moved 12 to 11 slots/layer, an expert-pool budget of 1,438.6 MiB. The 12-slot overshoot is corrected by EXP-023 |
| EXP-013 | io_uring + O_DIRECT expert streaming and the two-phase decode loop | Numerics, exempt from rule 2: logits **byte-identical on 8/8 prompts**, run against a control of the unmodified prior binary first so a failure would have been attributable, and re-verified after the dial moved to 11 slots. Throughput warm, uncgrouped, machine not quiet, a direction check only: prefill 0.66 to 1.53 tok/s, decode 0.75 to 1.83 tok/s, generated text character-identical. Cache at 11 slots/layer over 25 prompt plus 64 decode tokens, split by phase: prefill 45.3% hit rate and cold-dominated, decode 52.7% and 93% eviction-driven. Zero read retries and zero stale completions across 16,695 reads, with the O_DIRECT capability probe reporting verified. | Shipped |
| EXP-014 | The first measurement that satisfies rule 2 | Cold, in-cgroup, hygiene PASS on 5 of 5 scored runs, medians of 5, 5-token prompt with `--max-new 64`: **decode 1.88 tok/s**, prefill 1.33 tok/s, model load 1.32 s, wall 39.55 s, `memory.peak` **2,471.1 MiB of 3,072**, 36.9 GiB of expert bytes read. Context reaches only 69 tokens, so the KV tenant is a few MiB rather than its 384 MiB reservation at 4K; the dial was therefore not yet validated at full context. A first attempt scored 3 of 5 and its two flagged runs were discarded by the gate. | Shipped: the project's first publishable number |
| EXP-015 | Chunked-prefill building blocks, landed ahead of their measurement | **Nothing measured**: no cold run, no cgroup, no timing, no token. Exact arithmetic on the audited strides: decode reads ~1,097 MB of expert bytes per token worst case against ~34 MB per token under a 512-token chunk sweep, and each weight row fetched from RAM is dotted against ~32 activation rows (512 tokens x top-8 / 128 experts). The sweep ring costs zero contract bytes: 46.7 MiB borrowed from the already-faulted expert pool. Two harness defects found and fixed: systemd rewrites `${VAR}` and `$$` inside `ExecStart` arguments (no recorded measurement invalidated, checked rather than assumed), and the fingerprint comparison could report PASS while silently skipping the long prompts. | Shipped: enablers plus two harness fixes; nothing claimed |
| EXP-016 | Chunked layer-major prefill as the default path | **Nothing measured**; correctness gates only. Fingerprint against the reference baseline on the real model: PASS, 8/8 byte-identical, with the honest caveat that all 8 prompts fit inside a single 512-token chunk. In-process A/B: swept and token-major prefill agree bit for bit across 8 prompt lengths x 6 chunk sizes, and leave the same f16 KV on every layer; a wide fixture with every width pairwise distinct adds 96 sweep runs against 6 token-major baselines. Memory: chunk scratch 80,935,940 B and sweep ring 48,955,392 B are sub-allocations of the 1,438.59 MiB expert-pool row, so the contract does not move. | Shipped: default prefill path, token-major retained as the reference arm |
| EXP-017 | Where prefill time actually goes | Warm, uncgrouped, machine busy: shares and within-run ratios only, absolute seconds not quotable. Attention is 25.9% of a token-major 512-token prefill, **61.3% of a swept 512-token one and 85.2% at 1,891 tokens**; expert io falls from 36.2% of prefill to 1.7% under the sweep. Within one invocation the sweep cut non-attention work 4.9x (expert io 55x, expert compute 3.27x, projections 2.36x) for 2.45x end to end. Attention's per-token cost rises 3.73x against a 3.69x prompt-length ratio, which is what a quadratic total looks like measured per token; expert compute is flat, which is the linear term the amortization argument predicts. | Superseded: it redirected the next phase to attention, and after that work attention is third in prefill (EXP-021, EXP-023) |
| EXP-018 | The swept prefill path against the token-major path it displaced, cold and paired | Cold, in-cgroup, hygiene PASS on both arms, both run back to back in one session on one 512-token prompt with `--max-new 4`; `--repeats 1`, so run-to-run spread is unbounded. **Prefill 1.65 to 4.23 tok/s, 2.56x.** Process `read_bytes` 239.33 GB to 20.72 GB, **11.55x fewer**, the swept arm reading 1.11x the 18,626,213,888 B installed model, which is the signature of reading each expert once. `memory.peak` 2,576.0 against 2,570.1 MiB, both about 500 MiB under the ceiling, with zero reclaim. Decode fell 1.88 to 1.38 tok/s at that 4-token window. | Shipped: the first paired A/B satisfying rule 2. The decode fall was explained and reversed by EXP-021 |
| EXP-019 | O_DIRECT bandwidth under rule 2, on four real installed layer files | Cold, in-cgroup, hygiene CLEAN on all four runs with eviction proven and positive-controlled, medians of 3. Queue depth is emulated with threaded `preadv`, not io_uring, so it characterises the drive and the filesystem and not the runtime's submission path. Absolute bandwidth **1.54 to 2.37 GB/s** across the matrix; the decode-shaped cell (single blob, random, QD 8) reads 1.55 to 1.69 GB/s and the sweep-shaped cell (8 blobs, sequential, QD 2) 1.60 to 2.37. Throughput turns on **bytes in flight**, not queue depth: flat to roughly 100 MB outstanding, 15 to 18 percent down past 170 MB, and both shipped dials sit at the peak (49.0 MB prefill, at most 24.5 MB decode). Zero compressed extents on all four files, so O_DIRECT was honoured. | Shipped: the performance model is re-derived from those two cells |
| EXP-020 | The attention kernel rebuilt, measured warm | Warm, in-process, single-threaded microbenchmark: no I/O, no model file, **nothing publishable**. Wave 1, hoisting the f16-to-f32 conversion out of the GQA group, is the one paired figure on a hardened instrument, in palindromic run order with drift controls inside 0.999x-1.003x: **2.708x at one layer, 2.684x across 48**. Wave 2 (AVX2+F16C) was taken on a machine that was not quiet with 7 of 12 drift controls outside band, so its column and the cumulative ratios are approximate and not quotable. Softmax is 23.5% of the kernel at 4,096 positions and is frozen. Cost stopped being linear in context: the 48-layer arm's spread in ns per position widened from 1.031x at baseline to 1.40-1.57x once the arithmetic got cheaper, so attention drifts memory-bound at long context. Decode's fan-out ceiling is structural at 4 kv heads. Bit-identical at zero tolerance, including a sweep of all 65,536 f16 bit patterns; no FMA anywhere in the vector path. | Shipped: the waves ship on bit identity; the cold measurement is EXP-021 |
| EXP-021 | The attention work measured cold: prefill, decode, 4K context, numerics | Cold, in-cgroup, hygiene PASS, paired against the banked phase-5 binary back to back in one session. Prefill on a 512-token prompt, medians of 5 scored runs per arm: **1.68 to 11.17 tok/s, 6.65x**, with per-arm wall spread of 0.70% and 1.07%. Decode at `--max-new 256`, one run per arm: 1.18 to 1.99 tok/s in one session (1.69x) and 1.17 to 1.83 in a second (1.56x); the two sessions are not averaged. At 3,961 prompt tokens `memory.peak` is 2,920.4 and 2,924.2 MiB, **148 to 152 MiB under the 3,072 MiB ceiling**. Numerics all PASS: fingerprint 8/8 byte-identical, gate 3 at mean KL 1.039e-2 with worst prompt 2.721e-2 and top-1 8/8, greedy regression at baseline. The token-major prefill hit rate, inferred at ~58% by EXP-018, measures 58.2% here. | Shipped: 11 slots/layer validated at full context |
| EXP-022 | `T_BLOCK` 4 to 8 with a stepped position tail | Warm, in-process, uncgrouped: six interleaved runs across two trees differing in exactly one constant, so nothing publishes. On the 48-layer arm the medians are neutral at 64 and 128 positions (1.003x and 0.993x) and **1.039x, 1.053x, 1.063x and 1.074x at 512, 1,024, 2,048 and 4,096**, with the two arms' full ranges disjoint from 512 up and the conclusion surviving restriction to the runs whose controls are clean. The one-layer arm is not quotable: its own drift control failed in 5 of 6 runs, on both binaries. Register pressure bounds the technique at 8 accumulator chains; 16 spill. | Shipped: no rung regresses, the stepped tail is a strict superset by construction, and the cold value is measured in EXP-023 |
| EXP-023 | The cold decode sweep: phase split against context, the 11-slot hit rate, the slot dial, `T_BLOCK` | Cold, in-cgroup, hygiene PASS on all nine arms, medians of 3 scored runs, `--max-new 64`, five context rungs at the shipped 11 slots/layer. Decode **2.19 tok/s at 64 prompt tokens, 1.91 at 512, 1.82 at 1,024, 1.75 at 2,048, 1.46 at 3,961**. The live hit rate at the shipped dial, measured for the first time: **53.0% to 59.3% across the five rungs**, a scatter with no trend in context. Decode's expert-io share falls 54.1% to 33.2% while attention rises 1.5% to 31.6%; expert io is the largest term below 2,048 and level with attention at 3,961. In prefill the ordering at 512 tokens and above is expert compute, then projections, then attention. Slot dial: **12 slots/layer fits at 3,961 tokens, 3,058.4 MiB with 13.6 MiB spare**, and one extra slot buys about 2 points of hit rate and about 4% of decode. Paired `T_BLOCK` arms: 1.014x decode at 3,961 and nothing separable at 512. | Shipped: `T_BLOCK` 8 gated cold. The dial stays at 11 slots/layer; four predictions become measurements |
| EXP-024 | Whether the per-file read spread is a property of the files | Cold, in-cgroup, hygiene PASS on four probe arms, threaded `preadv`, no runtime binary involved. The four-file spread previously measured at 2.21x **re-read 1.044x the next day** at the same cell, and a byte-identical copy of the earlier script run 103 s later agreed within 3.5%, so the instrument is not the explanation. All 48 expert files at the decode-shaped cell span 1.565 to 1.694 GB/s, a **1.082x population spread**, most of which is blob size: the two stride classes are 1.038x apart, bandwidth correlates with block size at r = 0.835 and with physical dispersion at **r = -0.043**. Inside one file, where age and write history are identical by construction, a 7,493x median difference in physical span buys **1.161x** of bandwidth with overlapping ranges. | Withdrawn: the spread is not a stable property of a file and there is no runtime lever. The residual mechanism is drive-internal on a DRAM-less QLC part and is not addressable from `crates/core` |
| EXP-025 | Fusing the decode fan-out to once per expert phase, paired cold | Cold, in-cgroup, hygiene PASS on all 28 runs, medians of 3, `--max-new 64`, each reference arm run back to back with its branch partner on the same prompt at the same dial. Decode's GEMV bucket (expert compute plus projections) gains **1.257x at 512 prompt tokens and 1.158x at 3,961**, scored ranges disjoint at both, carried by expert compute (1.347x and 1.256x). End to end there is **no separation at 512**, where the arms' ranges overlap heavily and neither a gain nor a regression may be claimed; at 3,961 every scored run of the fused arm beats every reference run, and **between 1.025x and 1.070x** of that is attributable to the change, because the untouched attention bucket moved 1.16-1.18x in the same runs. Expert io rose about 1.85 s at 512 on byte-identical I/O counts, which is inferred to be a faster hit phase leaving less work to hide outstanding reads behind. | Open: measured, not merged. The `attn_q` and `attn_v` half was removed on this evidence; no dial moves |

## Corrections and supersessions

The project caught these itself, and each is recorded where the original claim
was made.

- **EXP-003 to EXP-004.** The single-position, top-20-truncated, renormalized KL
  in EXP-003 was never comparable to a full-vocab target. EXP-004 measured all
  151,936 logprobs per prompt and superseded it.
- **The 1e-3 KL gate.** Gate 3 as originally written required mean full-vocab
  KL <= 1e-3. Measured, it is 1.04e-2: **recorded FAILED and withdrawn**. The
  replacement, 3e-2, rests on measured evidence rather than judgement: the
  intra-engine AVX2-against-scalar noise floor is 4.5e-3 to 1.3e-2, the same
  order as the cross-engine gap, so 1e-3 is unachievable without
  operation-identical kernels. The per-prompt ceiling of 6e-2 is 1.76x above
  the largest float-reordering perturbation this codebase can produce.
- **EXP-005's simulator understated the shipped cache.** Driven one expert at a
  time the simulator reproduces exactly; driven the way the runtime calls it,
  pinning a whole step, the same traces give about 5 points more. The batch
  pinned replay is what the bracket around the shipped dial was built from, and
  a simulated hit rate is never quoted directly.
- **EXP-005's memory arithmetic superseded by EXP-012.** The fit columns of the
  slot sweep counted only the mmap'd common core and the KV cache, and omitted
  runtime anonymous memory entirely.
- **EXP-008 refuted by EXP-019.** The block-size premise does not survive: bigger
  blocks are neutral on one probed file and 15 to 16 percent *worse* on the
  other three when the read is sequential, against a claimed +51% at 16 MiB.
  The absolute level was also low, 1.211-1.390 GB/s against 1.54-2.37 measured
  on a quiet machine under rule 2, which is what EXP-008's own method warned a
  contended machine would produce. What survives is restated: throughput turns
  on bytes in flight, not on queue depth alone.
- **EXP-012's 12-slot overshoot was wrong in sign.** It predicted 12 slots/layer
  would land 19.8 MiB *over* the 3,072 MiB cap. Measured at 3,961 prompt tokens
  it lands **13.6 MiB under**, a 33.4 MiB overprediction, and the same rung at
  11 slots overpredicts by 31.7 MiB. Two nearly equal errors one slot apart
  point at a wrong constant in the fixed-tenant sum rather than wrong per-slot
  arithmetic; the provisional anonymous-memory row is where to look. The dial
  did not move on it (EXP-023).
- **EXP-013's blended hit rate.** An earlier revision quoted one figure blended
  across prefill and decode and compared it against a decode-only simulation.
  The counter was cumulative from state construction with no phase split, so the
  two were never comparable. The entry now carries a re-measurement split by
  phase, and the phases differ exactly as expected: prefill cold-dominated,
  decode eviction-dominated.
- **EXP-015's fingerprint gate could pass while ignoring most of the
  fingerprint.** A capture compared against a mismatched baseline reported PASS
  8/8 and silently skipped the long prompts. A prompt-set mismatch is now a hard
  error.
- **EXP-017 corrected its own prediction.** The estimate made before measuring
  put the expert GEMV above attention and judged it the larger lever. Measured,
  attention is 61.3% of a 512-token prefill and expert compute 19.1%: both
  estimates were wrong, in opposite directions, and the ordering they implied
  was backwards. The specific hypothesis that the batched GEMV was thrashing on
  the activation side is refuted, the batched path delivering 3.27x on the same
  prompt in the same session.
- **EXP-018's decode regression explained and reversed.** Decode falling 1.88 to
  1.38 tok/s was measured at a 4-token window. At `--max-new 256` the older
  build's own decode degrades by a third over the same window (0.65x and 0.63x
  in two sessions) while the newer holds 0.96x, so the earlier figure was a
  4-token number inflated by a warm cache and a short context and the later was
  almost entirely a cold-start transient. Neither was steady state (EXP-021).
- **The khugepaged amendment.** Eleven runs across two sessions were initially
  recorded DIRTY for non-zero `pgsteal`. All of it was `pgsteal_khugepaged`,
  with `pgscan == pgsteal` exactly (targeted freeing, not LRU scanning under
  pressure) and the flagged runs 0.39% *faster* than the clean ones on the
  identical workload. The verdict rule now keys on the pressure reclaimers and
  reports huge-page compaction as a soft note; seven of nine recorded summaries
  re-classified to PASS by re-reading counters already captured, not by
  re-running, and the reclassification prints old verdict beside new.
- **The estimated 1.5-2x on the QK dot measured 1.00-1.07x.** Carried in the
  backlog as an estimate with nothing behind it, `T_BLOCK` 4 to 8 measures
  neutral at 64 and 128 positions and 1.074x at 4,096 on the whole kernel. Both
  can be true, since the QK dot is one term under a memory-bound ceiling, but
  the estimate is superseded by measurement and is not re-quotable (EXP-022).
- **EXP-024's window result was itself corrected.** An earlier run reported
  1.104x; its timer enclosed thread start and join, and its case order was not
  interleaved. Both defects biased it **toward** the conclusion, and the
  corrected figure is 1.161x, further from it.
- **The per-file bandwidth spread is withdrawn as a property of the drive.**
  EXP-023 measured 2.21x across four expert files at decode's own read
  geometry and named it the lever worth pulling. The same cell re-read
  **1.044x** the next day, with the two fast files roughly halving and a
  byte-identical control script agreeing. Three sessions on the same files
  disagree by 2.2x on a quantity each reported as a property of a file. Any
  spread quoted anywhere describes its own session. What all three sessions do
  agree on: the slow end of the band sits at 1.55-1.69 GB/s.
- **Decode is published as two ladders, never one curve.** A byte-identical
  binary, on the same prompt files at the same dial, re-read **0.969x at 512
  prompt tokens and 0.911x at 3,961** a day after the ladder it had produced.
  Put a new build against those published figures instead of against its own
  same-session reference arm and a 1.025x-1.070x gain reads as a 0.979x
  regression. That inversion is why every ratio in this record is paired inside
  one session, and why cross-session decode figures are not subtracted (EXP-025).
