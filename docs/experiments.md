# Experiment inventory

This is ramvamp's measurement record: one row per experiment, in ID order,
each summarizing the internal log entry of the same ID. Every `EXP-NNN` cited
anywhere in this repository resolves to a row below, and the row states the
conditions that make its figures valid. The internal log, run artifacts and
harness output are not published.

Every performance figure comes from one machine and one drive, described in
[benchmarks.md](benchmarks.md#reference-machine), and throughput is a property
of that drive as much as of the code. The llama.cpp reference capture behind
EXP-003 and EXP-004 was taken elsewhere and carries no timing. The model is
Qwen3-30B-A3B: 48 layers, 128 experts per layer, top-8 routing.

Verdicts: **Shipped** (the change, or the conclusion it drove, is in the
runtime), **Rejected** (measured and not adopted), **Superseded** (a later
entry replaced its number or its premise), **Withdrawn** (its own claim was
retracted for want of evidence), **Recorded** (measured, nothing changed).

## Measurement rules

No row below is interpretable without these.

1. A microbenchmark may start an experiment. End-to-end speed and output
   quality decide what ships.
2. **Publication.** Only cold, in-cgroup, hygiene-PASS numbers are publishable.
   Everything else is a diagnostic and is labelled one in its row. Correctness
   measurements are exempt.
   - *Cold*: every model file evicted with `posix_fadvise(POSIX_FADV_DONTNEED)`,
     the eviction proven with `mincore`, and non-zero block-layer `read_bytes`
     as a positive control.
   - *In-cgroup*: `systemd-run --user` at the contract's `MemoryMax` (`4G` since
     EXP-033, `3G` for every entry before it) and `MemorySwapMax=0`, with
     counters read from inside the cgroup before exit; zram counts as swap.
   - *Hygiene PASS*: the inner wrapper resolved its own cgroup, the pressure
     reclaimers stole zero pages, every `memory.events` counter is zero,
     `memory.max` equals the requested limit, swap peak is zero, `read_bytes`
     is non-zero, and both processes exited 0. An unreadable counter fails.
   - *khugepaged amendment*: huge-page compaction is subtracted from the
     reclaim total and reported as a soft note, so an unknown reclaimer still
     counts as pressure.
   - *Refault amendment* (EXP-031): pressure reclaim fails a run only when more
     than 1 MiB of what it took had to be read back
     (`workingset_refault_file` + `workingset_refault_anon`); less is a soft
     note, because the machine's own memory pressure reclaims from every
     cgroup. A run recorded before the refault counters existed keeps the
     zero-reclaim test.
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
| EXP-001 | AVX2 K-quant dot kernels against the scalar reference | Warm single-thread microbenchmark on a synthetic 2048-row matrix, medians of 31 runs; diagnostic. Per output row over scalar: 3.2 to 3.8x for q4_k, q5_k and q6_k, 5.13x for q8_0. Integer results bit-identical; only float accumulation order differs, tolerance-tested. | Shipped: runtime AVX2+FMA dispatch with a scalar fallback, 1-byte alignment contract |
| EXP-002 | AVX2 activation quantizers | Warm single-thread diagnostic: `quantize_row_q8_k` 11.23x and `quantize_row_q8_0` 4.44x over scalar, output byte-identical to the scalar reference on random, tie-heavy, flat and zero rows. | Shipped: matches the scalar reference, not ggml's differing AVX2 rounding |
| EXP-003 | The forward pass against llama.cpp b10217 on identical Q4_K_M bytes | Correctness, exempt from rule 2. Greedy 16-token completions character-identical on 3/3 prompts. Its single-position, top-20-truncated KL is not comparable to a full-vocab target. | Superseded for KL by EXP-004; the greedy result is gate 4 |
| EXP-004 | Full-vocab KL against llama.cpp reference dumps, and the float-reordering noise floor | Correctness, exempt. All 151,936 logprobs per prompt over 8 prompts: mean KL 1.04e-2, worst prompt 2.72e-2, top-1 8/8, flat in context depth to 3,492 tokens; tightest top-1 margin 0.228 nats. The AVX2-against-scalar noise floor is 4.5e-3 to 1.3e-2, the same order as the cross-engine gap. Gate 5's reference is banked here: PPL 6.3810 +/- 0.16588 (`llama-perplexity` b10217, `wiki.test.raw`, `-c 512 --chunks 40`). | Shipped: gate 3 is mean <= 3e-2, every prompt <= 6e-2, top-1 on every prompt, all 8 scored |
| EXP-005 | Expert-cache policy and slot sweep on recorded routing traces | A replay of 556 decode tokens of routing, not a decode, so no rate here is the runtime's hit rate. Simulator: ghost LFU 44.8% against LRU 42.6% at 10 slots, and per-slot LFU without ghost history is -1.7 to 0.0 points against LRU, so the win is the eviction-surviving per-expert counters. Replaying the shipped cache, which pins a whole step, reads about 5 points higher (50.02% at 10 slots, 54.48% at 12). A global slot pool is worth +0.53 points and replaying the prompt into the cache +0.09. | Shipped: expert-indexed ghost LFU, 512 B of counters per layer; global pool and prompt replay rejected |
| EXP-006 | Whether the buffered-`pread` decode path can be measured cold inside a 3G cgroup | It cannot: a 13-token generation pulled 7,836.5 MiB through the block layer, held the cgroup at its ceiling and forced about 4.82 GiB of reclaim. A separate run showed `memory.events max` 0 while 2.4 GiB was silently reclaimed. | Shipped: the hygiene gate, motivated here |
| EXP-007 | Slot aliasing under concurrent O_DIRECT reads | Qualitative: a probe on a busy machine, so no rate is quotable. Two in-flight reads aliased onto one buffer make btrfs fail checksum verification with spurious `EIO`; exclusive leases from a free list measured 0 `EIO` across 4,000 reads at QD 4 to 32. btrfs also completes a direct read through the buffered path, silently, when it cannot fault the destination. | Shipped: owning slot guards; every pool page faulted at construction |
| EXP-008 | Reference-drive characterisation under O_DIRECT, and `ReadFixed` | Warm, contended, uncgrouped; provisional. 1.211 to 1.390 GB/s at QD4 to QD16 at the expert stride. `ReadFixed` saves about 70 us of CPU per 3 MiB read. | Superseded: level and block-size premise refuted by EXP-019. `ReadFixed` rejected: it would pin the pool under an 8 MiB `RLIMIT_MEMLOCK` |
| EXP-009 | Buffered against O_DIRECT expert reads, page-cache charge inside the cgroup | A memory measurement; the machine was not quiet. The same 1.4 GiB of expert reads peaked the cgroup at 1,092.2 MiB buffered against 5.0 MiB with O_DIRECT. Throughput was not timed. | Shipped: O_DIRECT is a memory requirement, not a speed preference |
| EXP-010 | Compute-pool signalling | Warm, uncgrouped diagnostic. A futex wake/wait pair p50 3.1 us at 0.07 cores; atomic spinning 502 ns at 1.03 cores; `std::sync::mpsc` p99 237 us. The shipped bounded spin then futex measures p50 1.71 us publish to barrier with workers warm. | Shipped: bounded spin then futex; `mpsc` rejected |
| EXP-011 | Row-range GEMV as the single code path | Rule-4 identity test, no timing: row-partitioned results bit-identical to unpartitioned over 8 partition schemes on 10 real model shapes, on the AVX2 and scalar paths. | Shipped: any row partition is bit-identical |
| EXP-012 | Whether the memory contract's tenants account for everything charged to the cgroup | They did not. Peak anonymous memory on a live decode was 115.1 MiB, provisional (context length, token count and cache state unrecorded). With it, 11 slots per layer projected 2,961.03 MiB and 12 slots 3,091.82 MiB against 3G. | Shipped: the runtime-anonymous tenant; the dial moved 12 to 11 slots (the 12-slot overshoot was measured wrong in sign by EXP-023) |
| EXP-013 | io_uring + O_DIRECT streaming and the two-phase decode loop | Numerics exempt: logits byte-identical on 8/8 prompts against the prior binary. Throughput warm and uncgrouped, a direction only. Hit rate split by phase: token-major prefill 45.3%, decode 52.7%. Zero retries and zero stale completions across 16,695 reads, with O_DIRECT verified by the startup probe. | Shipped |
| EXP-014 | The first measurement that satisfies rule 2 | Cold, in-cgroup (3G), hygiene PASS on 5 of 5, 5-token prompt, `--max-new 64`: decode 1.88 tok/s, `memory.peak` 2,471.1 MiB. Context reached only 69 tokens, so the 4K KV reservation was not exercised. | Shipped: the first publishable number |
| EXP-015 | Chunked-prefill building blocks | Nothing measured. Exact arithmetic on the audited strides: decode reads up to about 1,097 MB of experts per token against about 34 MB per token under a 512-token sweep, each weight row dotted against about 32 activation rows. Two harness defects fixed: systemd rewriting `${VAR}` in `ExecStart`, and a fingerprint check that could pass while skipping long prompts. | Shipped: enablers and harness fixes |
| EXP-016 | Chunked layer-major prefill as the default path | Correctness only. Swept and token-major prefill agree bit for bit across 8 prompt lengths x 6 chunk sizes and leave the same KV; fingerprint 8/8 byte-identical. The sweep ring and chunk scratch are carved from the expert pool, so the contract does not move. | Shipped: sweep is the default, token-major kept as the reference |
| EXP-017 | Where prefill time goes | Warm, uncgrouped, busy machine: shares and within-run ratios only. Attention was 61.3% of a swept 512-token prefill and 85.2% at 1,891 tokens; expert io fell from 36.2% to 1.7% under the sweep, and the batched expert GEMV cut expert compute 3.27x. | Superseded: it sent the next work to attention, after which attention is third in prefill (EXP-021, EXP-023) |
| EXP-018 | Swept prefill against token-major, cold and paired | Cold, in-cgroup (3G), hygiene PASS, one 512-token prompt, one run per arm: prefill 1.65 to 4.23 tok/s, 2.56x; process `read_bytes` 239.33 GB to 20.72 GB, 11.55x fewer, 1.11x the 18,626,213,888 B (17.35 GiB) installed model. Decode fell 1.88 to 1.38 tok/s at a 4-token window. | Shipped: the first paired A/B under rule 2; the decode fall is a cold-start effect (EXP-021) |
| EXP-019 | O_DIRECT bandwidth under rule 2 on four installed layer files | Cold, in-cgroup, hygiene CLEAN, medians of 3; threaded `preadv`, so it characterises the drive, not the io_uring path. 1.54 to 2.37 GB/s across the matrix; the decode-shaped cell (one blob, random, QD 8) 1.55 to 1.69. Throughput turns on bytes in flight: flat to about 100 MB, 15 to 18% down past 170 MB, and decode (at most 24.5 MB) and the prefill sweep (49.0 MB) sit on the plateau. | Shipped: the performance model rests on it |
| EXP-020 | The attention kernel rebuilt, measured warm | Warm, in-process microbenchmark, nothing publishable. Hoisting the f16-to-f32 conversion out of the GQA group: 2.708x at one layer, 2.684x across 48. AVX2+F16C vectorization and pool fan-out followed (rows by cost in prefill, kv heads in decode, a ceiling of 4 at the pin); a strided query-head split measured worse than the kv-head split. Bit-identical at zero tolerance, including all 65,536 f16 bit patterns; no FMA in the vector path. | Shipped on bit identity; measured cold in EXP-021 |
| EXP-021 | The attention work measured cold | Cold, in-cgroup (3G), hygiene PASS, paired back to back. Prefill on a 512-token prompt 1.68 to 11.17 tok/s, 6.65x (medians of 5). Decode at `--max-new 256` 1.69x in one session and 1.56x in a second, not averaged. At 3,961 prompt tokens `memory.peak` is 2,920.4 to 2,924.2 MiB of 3,072, and the prefill split is expert compute 42.1%, projections 22.8%, attention 20.4%, expert io 2.5%. Gate 3 PASS at mean KL 1.039e-2. | Shipped: 11 slots validated at full context |
| EXP-022 | `T_BLOCK` 4 to 8 with a stepped position tail | Warm, in-process, six interleaved runs: neutral at 64 and 128 positions and 1.039x to 1.074x from 512 to 4,096 on the 48-layer arm, ranges disjoint from 512 up. Sixteen accumulator chains spill. | Shipped; cold value in EXP-023 |
| EXP-023 | The cold decode sweep: phase split against context, the 11-slot hit rate, the slot dial | Cold, in-cgroup (3G), hygiene PASS, medians of 3, `--max-new 64`, 11 slots. Decode 2.19 / 1.91 / 1.82 / 1.75 / 1.46 tok/s at ctx 64 / 512 / 1,024 / 2,048 / 3,961; prefill 11.25 tok/s at 512; `memory.peak` 2,507.9 to 2,929.3 MiB; live hit rate 53.0% to 59.3%, no trend in context. Expert io is 54.1% of a token at ctx 64, 44.4% at 512 and 33.2% at 3,961, while attention rises from 1.5% (6.4% at 512) to 31.6%. 12 slots fits at 3,961 tokens (3,058.4 MiB) and 13 slots reads 2.06 tok/s at ctx 512. Single-blob queue depth plateaus by QD 2 to 4, and decode averages 3.26 to 3.76 misses per layer step. `T_BLOCK` 8 is 1.014x decode at 3,961. | Shipped: `T_BLOCK` 8; the dial stayed at 11 under 3G |
| EXP-024 | Whether the per-file read spread is a property of the files | Cold, in-cgroup, threaded `preadv`. A four-file spread measured at 2.21x re-read 1.044x the next day. All 48 expert files at the decode cell span 1.565 to 1.694 GB/s, median 1.633, a 1.082x spread that is mostly blob size; physical dispersion does not predict it (r = -0.043). | Withdrawn: the spread is not a stable property of a file, and there is no runtime lever |
| EXP-025 | Fusing the decode fan-out to once per expert phase, paired cold | Cold, in-cgroup (3G), hygiene PASS on 28 runs, medians of 3, `--max-new 64`. The fused build's own ladder: decode 2.16 / 1.87 / 1.74 / 1.65 / 1.43 tok/s at ctx 64 to 3,961, prefill 11.07 at 512, `memory.peak` 2,497.0 to 2,926.7 MiB. Decode's GEMV bucket gains 1.257x at ctx 512 and 1.158x at 3,961, but expert io rose about 1.85 s at 512 on identical I/O counts, so the token does not separate at 512; at 3,961, 1.025x to 1.070x is attributable. The byte-identical EXP-023 binary re-read 1.85 against 1.91 tok/s at 512 and 1.33 against 1.46 at 3,961. | Shipped in part: the gate/up and down fusions; the `attn_q` + `attn_v` merge was dropped on this evidence |
| EXP-026 | Whether the Arc iGPU runs prefill's Q4_K x Q8_K batched GEMV faster than the CPU pool | A Vulkan shader on a real expert slab against the production kernel. Numerics, exempt: max abs err 1.490e-6 over 24,576 outputs. Q4_K only, because Q6_K's 210 B block is not a whole number of the shader's 4-byte units. No quotable throughput ratio: with both units' clocks locked the GPU arms held steady while the CPU pool flipped between two levels about 1.7x apart. | Closed, not now: no GPU backend ([roadmap](roadmap.md), v1 item 2) |
| EXP-027 | What one busy core costs the compute pool, and whether chunked dispatch and shared placement remove it | Microbenchmark, diagnostic: one busy core cost the pinned pool 1.2 to 2.3x; chunks with every thread free across the P-cores and L3 E-cores held 1.00 to 1.19x under every load; on an idle machine per-core pinning stays about 4% faster on decode's GEMVs. Cold and paired at ctx 512: prefill 1.13x and decode 1.045x worst against best, outputs identical. Not publishable: 5 of 6 runs lost 9 to 11 MiB to machine-wide reclaim under the zero-reclaim test. | Shipped: chunked dispatch for every GEMV, `--placement shared` the default, `per-core` kept |
| EXP-028 | Moving prefill's single-threaded work onto the compute pool | The router, per-row elementwise work and per-expert glue ran on one thread while the pool waited. Split by row (the router by expert in decode), arithmetic unchanged, sweep and token-major still bit-identical. Cold and paired at ctx 512: prefill 11.80 to 17.49 tok/s, 1.36x worst against best, decode unchanged; not publishable, for the reclaim recorded in EXP-027. | Shipped |
| EXP-029 | Dotting four activation rows per pass in prefill's batched Q4_K GEMV | A four-row tile shares the weight-only work and keeps each dot's float operations unchanged; bit-identical to the single-row kernel. Kernel 1.33x at 4 rows to 1.39x at 32. Cold and paired against EXP-028's build: prefill 15.34 to 21.57 tok/s, 1.32x worst against best; not publishable, as EXP-027. | Shipped; Q6_K followed in EXP-032 |
| EXP-030 | Whether any online expert-cache policy beats the shipped ghost LFU | Instrument correctness, exempt: `scripts/lfu_sim.py --pin-step` reproduces the live decode hit rate to within 0.01 points (55.74% simulated against 55.735% live at 11 slots). On it: S3-FIFO -0.9 points, LRU-2 -1.2, ARC -2.2, LRU -3.1; the best decaying variants +0.36 to +0.40 out of sample; Belady 62.30%, a 6.6-point gap. One more slot is worth 2.2 to 2.4 points. | Rejected: about 0.4 points is below what a cold decode can resolve. The simulator modes and one committed trace ship |
| EXP-031 | Staying inside the 3 GB cap while the machine is short of memory | Cold, in-cgroup, mirrored arm order, the machine left as it was. Memory-mapped common weights lost pages to machine-wide reclaim under either cap (once 801 MiB, with decode falling to 1.43 tok/s). `--common-memory resident` reads `common.bin` into anonymous memory: 0 refaults in all 6 runs, same speed, +0.75 s load, +70 MiB peak. At ctx 3,961 resident peaks at 2,962.9 MiB, 109 MiB under 3G and within 2 MiB of `ramvamp plan`; outputs identical. | Shipped: resident is the default, model commands run in their own swap-off cgroup, and rule 2 gains the refault amendment |
| EXP-032 | The four-row tile for Q6_K, and the first clean cold runs with the machine in use | Kernel: Q6_K `down` 1.46x at 4 rows to 1.51x at 32, bit-identical. Cold, in-cgroup (3G), hygiene PASS on all 12 runs, zero refaults, A B B A at ctx 512, `--max-new 64`: the tile is 1.027x on prefill medians, decode unchanged. The build before the tile gives the first publishable figures after EXP-025: prefill 23.93 to 24.21 tok/s (median 24.10), decode 2.18 to 2.24 (median 2.21), `memory.peak` 2,638 to 2,657 MiB. | Shipped |
| EXP-033 | What a 4 GB contract buys decode: the slot dial at 11, 15 and 19 | Cold, in-cgroup at `MemoryMax=4G`, hygiene PASS, zero refaults, outputs identical, order 11 15 19 19 15 11, ctx 512, `--max-new 128`, four scored runs per arm. Decode medians 2.215 / 2.52 / 2.795 tok/s at 11 / 15 / 19 slots (2.78 to 2.81 at 19), 1.262x at 19, ranges disjoint; hit rate 53.5% / 61.7% / 67.1%; prefill unchanged (24.48 to 25.27 at 19); `memory.peak` 2,645 / 3,167 / 3,690 to 3,697 MiB. At ctx 3,961, `--max-new 64`, one scored run per arm: decode 1.75 at 11 slots and 1.96 at 19 (1.12x), prefill 18.23 at 19, hit rate 68.3%, `memory.peak` 2,960.8 MiB at 11 and 4,008.4 MiB at 19, 88 MiB under 4 GiB and within 2 MiB of `ramvamp plan`. | Shipped: the contract is 4 GB with 19 slots; `--cache-bytes 1440M` keeps the 3 GB setup |
| EXP-034 | Steady-state decode: whether the cache warms over a longer reply, at the 4 GB default | Cold, in-cgroup at 4G, hygiene PASS, zero refaults, 19 slots, ctx 512, four scored runs per window. Decode 2.80 / 2.79 / 2.61 tok/s over 128 / 256 / 512 new tokens; the live hit rate is flat at 67.1% / 67.7% / 67.1% because the cache fills within a few tokens. The 512-token window is 0.93x the 128-token one, carried by attention and expert io (about 204 to 212 ms a token) as context grows. At ctx 3,961 with 128 new tokens: decode 2.04 to 2.06 tok/s, hit rate 69.6%, `memory.peak` 4,016 to 4,020 MiB, 12 MiB over `ramvamp plan` and 76 MiB under the cap. | Recorded: the short windows were not measuring a colder cache; no code change |
| EXP-035 | Gate 5: WikiText-2 perplexity against `llama-perplexity` b10217 on the same GGUF | Correctness, exempt. The protocol read from b10217's source: the file less one trailing newline, tokenized once, no BOS, 40 windows of 512 from an empty KV cache, positions 256 to 510 scored. Token streams identical to `llama-tokenize`. Criterion set before the run: the final and every running PPL within 1%. PPL 6.3598 +/- 0.16504 against 6.3810 +/- 0.16588, -0.33%; worst running gap -0.69% at window 3. | PASS: every v0 gate closed. `ramvamp perplexity` and `scripts/perplexity_gate.py` ship |
| EXP-036 | Gemma 4 26B-A4B: auditing the chosen checkpoint's header | Audit, no run. Google's QAT Q4_0 GGUF, header read from the file; the parser reproduces Qwen3's 1,097,072,640 B per token exactly. 30 layers, 128 experts at top-8, every expert Q4_0 at 3,345,408 B, so 802,897,920 B per token worst case (0.73x Qwen3's); weights outside the experts 1.469 GiB against 0.999. | Recorded: Q4_0 would be the one new kernel; Gemma 4 is set aside ([roadmap](roadmap.md), v1 item 1) |

## Corrections and supersessions

The project caught these itself; each row above carries the corrected figure.

- EXP-003's truncated KL was never comparable to a full-vocab target; EXP-004 replaced it.
- Gate 3's original mean-KL bar of 1e-3 is unachievable against a 4.5e-3 to 1.3e-2 noise floor; it became 3e-2 plus three per-prompt conditions (EXP-004).
- EXP-005's simulator runs about 5 points under the shipped cache, which pins a whole step; a simulated rate is never quoted as the runtime's.
- EXP-005's fit arithmetic omitted runtime anonymous memory; EXP-012 added the tenant.
- EXP-008's block-size premise and bandwidth level were refuted by EXP-019.
- EXP-012's predicted 19.8 MiB overshoot at 12 slots was wrong in sign: 12 slots measured 13.6 MiB under 3G at full context (EXP-023).
- EXP-013's first hit rate blended prefill and decode; it is now reported split by phase.
- EXP-015's fingerprint gate could pass while skipping long prompts; a prompt-set mismatch is now an error.
- EXP-017's estimate put the expert GEMV above attention in prefill; measured, the order was reversed.
- EXP-018's decode fall at a 4-token window was a cold-start transient, not steady state (EXP-021).
- The khugepaged amendment reclassified runs flagged only for huge-page compaction from counters already captured, without re-running.
- The estimated 1.5 to 2x on the QK dot measured 1.00 to 1.07x on the whole kernel (EXP-022).
- EXP-024's in-file window result moved from 1.104x to 1.161x once its timer and case order were fixed.
- The per-file bandwidth spread (2.21x in EXP-023) is withdrawn as a property of the drive (EXP-024).
- Decode is published as two ladders, never one curve: a byte-identical binary re-read 0.969x and 0.911x a day later (EXP-025).
