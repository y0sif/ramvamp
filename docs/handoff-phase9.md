# Handoff: phase 9 to phase 10

Branch `feat/decode-compute`, on `8e1eee8`. **Not merged, and not
blocked either**: the cold rule-2 sweep ran and is recorded as EXP-025. The
branch is parked by the author's decision, to be landed or revisited later.
Everything below is measured; nothing is pending.

## WHAT PHASE 9 MEASURED

Phase 9 had two targets. One shipped a change worth measuring cold. The other
turned out to be a negative result, and a third finding, which nobody planned,
is probably the most useful thing in the phase.

### 1. Decode GEMV: the fan-out was the cost, and it is memory-bound underneath

The instrument came first. `12856bb` splits every pooled decode GEMV, on the
submitting thread, into `own` (set-up plus that core's own shard) and `wait`
(the barrier), across projections / experts / lm_head / router. It works
because the pool runs shard 0 inline, so both sides are visible from one
thread without touching a worker's hot path. 4,131 clock reads a token before
fusion, ~112 us against a 532 ms token.

What it found, warm, ctx 512, 63 tokens (MEASURED, diagnostics, see the
caveat below): barrier wait was 9.07 s of a 14.11 s pooled GEMV bucket, 64%
of it. Two explanations died on arithmetic before any code changed: the pool
barrier at 1.3 us a fan-out is 1.81 ms a token, and activation quantization is
0.45 ms. Together under 1%.

The control settled it. `taskset -c 0` forces one shard and drives `wait` to
zero, so `own` becomes the whole arithmetic:

    1 shard   16.35 s
    6 shards  14.11 s

**Six cores bought 1.16x.** That is the finding the phase turns on.

`70cf304` fuses matrices into one fan-out per expert phase: all of a phase's
gate and up together, all of its down together, and attn_q with attn_v. A
layer goes from 28 fan-outs to 5 when its plan is all hits or all misses and 7
when it splits; a token from 1,345 to between 241 and 337. (**The attn_q with
attn_v half was dropped on 2026-08-08**, item 3 below. As it now stands a layer
goes from 28 to 6 and 8, and a token to between 289 and 385, DERIVED. Every
measured figure in this handoff describes the binary that was measured, which
still had it.) It is bit-safe because a row of a fused space still computes one
whole-row dot on the same bytes, so the pool's tiling of the fused space
restricts to a tiling of each matrix's own rows.

Measured warm, ctx 512, 3 runs each, pooled GEMV bucket, medians with ranges:

    baseline       14.44 s  (14.35, 14.44, 15.06)
    adaptive spin  14.27 s  (13.99, 14.27, 14.48)
    fused          11.42 s  (11.36, 11.42, 11.48)
    both           11.38 s  (11.34, 11.38, 11.51)

**1.264x**, and by bucket: experts 9.94 to 7.20 (1.381x), projections 3.52 to
3.21 (1.097x), lm_head 1.05 to 0.94 on **unchanged code** (1.117x, which is
the noise floor for a bucket that size). That projections row, 1.097x under a
1.117x floor, is the reading that eventually got the attn_q with attn_v half
dropped (item 3).

**It held cold (EXP-025).** The decode GEMV bucket, medians of 3 with ranges,
paired in one session against the banked `8e1eee8` binary:

    ctx  512   ref 16.25 (15.68-16.39)  ->  fused 12.93 (12.49-13.96)   1.257x
    ctx 3961   ref 15.39 (14.71-15.72)  ->  fused 13.29 (13.18-13.31)   1.158x

Ranges disjoint at both rungs. Note the warm 1.264x and the cold 1.257x are
close but are **not the same bucket**: warm was the `own + wait` pooled GEMV
from the sub-split, cold is `expert compute + projections` from the coarse
split. Do not treat either as confirming the other's denominator.

**End to end it separates at 4K and not at 512.** Decode tok/s, medians (runs):

    ctx  512   ref 1.85 (1.78, 1.85, 1.87)  ->  fused 1.87 (1.76, 1.87, 1.97)   1.011x, overlapping
    ctx 3961   ref 1.33 (1.31, 1.33, 1.35)  ->  fused 1.43 (1.37, 1.43, 1.44)   1.075x, disjoint

512 gained nothing because **`expert io` rose ~1.85 s in the fused arm**,
systematically. Inferred, not measured: `expert io` is a residual after hit
compute, so faster hit-compute leaves less work to overlap the outstanding
reads with and more read latency becomes visible. **A compute win at mid
context partly converts into exposed io wait.** That is the single most
important thing phase 9 learned about where to go next.

### 2. The compute pool is not dispatch-bound. It is memory-bound.

`ce9b0d3` replaced the pool's fixed 64-pause-iteration spin with a budget
derived per job from the gap workers had just sat through. It measured 1.012x
against a baseline whose own spread is 1.049x. Inside the noise, and nothing
on top of the fusion either. It is reverted by `88e3e9d` and preserved
unchanged on branch **`feat/pool-adaptive-spin`**, cherry-pickable.

The reason it could not work is worth more than the change was. Fusion cut
expert scatters **6.14x** (72,576 to 11,812) and cut expert barrier wait only
**1.47x** (6.77 s to 4.62 s, medians). Per-scatter wait went *up* 4.3x. **The
wait scales with work, not with fan-out count**, so it was never wake latency
and no spin policy could reach it. Workers are slower per row than the
submitting thread.

Two references died with it, and both had been quoted as authoritative:

- **EXP-001's 9.61 GB/s is not a reference for decode.** That fixture is
  L2-resident; decode reads every expert byte once from DRAM. Three code
  comments cited it and are fixed.
- **The pool's 1.3 us barrier figure comes from `pool.run(6, |_| {})`**, a hot
  loop in which no worker ever parks. It does not describe decode.

Post-fusion, six cores buy **1.43x** over one and the aggregate is 11.00 GB/s.
Decode GEMV is bound by the memory system. **That is the open question for
phase 10** and nothing in phase 9 addresses it.

### 3. The per-file read spread was a session artifact (EXP-024, NEUTRAL)

The premise was a 2.21x spread at decode's K=1 block size, with layer_00 at
1.57 GB/s against layer_20 and layer_21 at 3.46. If the slow files could be
made to read like the fast ones, expert io drops about a third.

They cannot, because there are no fast files. Measured 2026-08-07, all hygiene
PASS:

- The same cell on the same four files: 1.60 to 1.67 GB/s. **Spread 1.04x
  where EXP-023 measured 2.21x.** The fast files became slow; the slow one did
  not move. A control with the unmodified probe from `8e1eee8` rules out this
  phase's edits to the script.
- **All 48 files, measured for the first time** (every prior entry sampled
  four): 1.565 to 1.694 GB/s, spread 1.082x. Correlation of bandwidth with
  largest-region byte fraction is **-0.043**.
- Dense vs scattered 2 MiB windows inside layer_00: **1.161x** against a
  7,493x median span contrast.

What does explain the residual spread is **blob size**, at r = 0.835: the two
stride classes sit at 1.610 and 1.671 GB/s with only ~1.045x inside either.

Per rule 3 these are three sessions and must not be drawn on one curve. The
finding is not that the drive got slower. It is that the spread is not a
stable property, so any entry quoting it describes its own session.

`docs/benchmark-machine.md`'s "It is not fragmentation" is rewritten: extent
*geometry* was ruled out, but the probe that ruled it out never computed
physical dispersion at all, and non-reproduction is the larger caveat.

## WHERE THE TOKEN GOES NOW

Fused, cold, medians. This is the map any next phase should plan against:

    ctx  512 (524 ms/token)        ctx 3961 (712 ms/token)
      expert io   275 ms  52.5%      expert io   256 ms  36.0%
      GEMV        205 ms  39.2%      attention   232 ms  32.6%
      attention    31 ms   5.8%      GEMV        211 ms  29.6%
      elementwise  13 ms   2.5%      elementwise  13 ms   1.8%

The fusion handed the crown back to I/O. At 512 expert io is over half a
token, and it **grew** when compute got faster.

## WORK TO DO, IN PRIORITY ORDER

1. **Raise the expert cache hit rate. This is the next phase.** It is the
   biggest lever, it is bit-safe by construction (which experts are resident
   changes nothing about what is computed), it costs no memory, and it attacks
   the exposed-io problem from the other side: fewer misses means less read
   latency to expose.

   The headroom is already measured and unused. EXP-005 replays the shipped
   ghost-LFU at **49.9%** against Belady's **72.0%** at 12 slots; live hit rate
   is 53-59%. Nobody has tried a policy in between.

   **And it is answerable offline**, which phase 9's questions were not.
   `scripts/lfu_sim.py` already replays real `--trace-experts` traces against
   the real per-layer strides, sweeps slots, and scores `lfu`, `lfu-aged`,
   `lfu-ghost`, `lfu-window`, `lru` and `opt`. So: capture traces, add
   candidate policies (LRU-K, ARC, S3-FIFO, a layer-aware variant exploiting
   the per-layer slot arrays), pick the winner on hit rate in seconds, and
   spend exactly one paired cold sweep confirming it. Phase 9 needed ~2 h to
   answer each question; this loop does not.

   DERIVED sketch of the prize: taking the hit rate from ~54% to ~65% is ~24%
   fewer bytes, roughly 275 ms to 209 ms at 512, about 66 ms a token. Larger
   than everything phase 9 shipped.

2. **Instrument the overlap, alongside item 1.** `expert io` is a residual and
   that is now load-bearing: it moved +/-1.85 s between paired arms and
   swallowed the fusion's win at 512. Split it into submitting, waiting on a
   read genuinely in flight, and waiting with the queue empty. Same class of
   change as phase 9's `own`/`wait` instrument, which changed the plan twice.
   Until it exists, nobody can say how much of that 275 ms is reducible.

3. ~~**Decide the `attn_q` + `attn_v` fusion.**~~ **DECIDED 2026-08-08:
   dropped.** The half is removed from `forward.rs`; the expert-phase fusion,
   which is what carries the cold 1.257x on decode's GEMV bucket, is untouched
   and stays. Read that 1.257x as a figure for **the binary EXP-025 measured**,
   which still had both halves: the shipped binary's GEMV bucket has not been
   measured. EXP-025 attributes the gain entirely to `expert compute` (1.347x
   at 512, disjoint) and records `projections` as not separating at either
   rung, so no part of it is expected to have gone with the half, but that is
   an inference and not a measurement. It was 1.097x against a 1.117x noise floor warm, its bucket did
   not separate cold at either rung (1.134x at ctx 512 and 1.025x at 3,961,
   ranges overlapping at both, both smaller than the 1.164-1.176x the
   *unchanged* attention bucket moved by in the same runs, EXP-025 Note 3), and
   it contributed 48 of the ~1,012 fan-outs a token that `70cf304` removes,
   about 4.7%. Both reviewers flagged it independently. It had been deliberately
   fenced off from every fix lane, so the removal was clean: gates and numerics
   green, `bitident.py` PASS 8/8 byte-identical to the phase-4 baseline and
   gate 3 PASS at mean KL 1.039e-02. **No performance claim attaches to the
   removal**: its bucket never separated cold, so no change is expected and
   none has been measured. Recorded in EXP-025 Note 3 (amendment),
   `docs/architecture.md` and `docs/roadmap.md`.

   **The shard-split figure this item used to quote was ~24% and the number to
   quote is ~30%** (DERIVED, and it is the figure the now-deleted code comment
   in `forward.rs` carried). Both come from the same imbalance and differ only
   in denominator. On the 24 layers where `attn_v` is Q6_K, the fused space is
   4,096 Q4_K rows then 512 Q6_K rows, and at `in_dim` 2048 a Q6_K row is
   1,680 B against a Q4_K row's 1,152 B. An even six-way split by row is 768
   rows a shard, so the last shard holds 256 Q4_K rows plus all 512 Q6_K ones:
   1,155,072 B against 884,736 B in each of the other five, **1.306x**, about
   30% long. Against the mean shard (929,792 B) the same imbalance reads
   1.242x, which is where ~24% came from. A barrier waits on the slowest shard
   against the others, so ~30% is the figure that describes the cost.

4. **Why are workers slower per row than the submitter?** The most interesting
   open problem, and the one most likely to eat a phase for nothing, so
   timebox it and do it after item 1. Six cores buy 1.43x on a site reading
   1.99 GB a token. 11.00 GB/s aggregate is far below what the memory system
   should give, so it is not obviously bandwidth saturation either. Candidates
   nobody has tested: memory-level parallelism per core, software prefetch,
   the access pattern into q4_k super-blocks, effects of the hybrid part, or
   the submitter simply starting earlier. A cost-weighted `shard_range` is a
   separate, smaller lever; it used to have item 3's mixed-format imbalance as
   a second motivation and no longer does, since dropping that half leaves no
   fused space that mixes quant formats.

5. **Attention at 4K** is 232 ms of a 712 ms token, 33%, and untouched. It is
   6% at 512, so this only buys the 4K story. Online/flash rescaled softmax is
   forbidden; blocking and tiling are not.

6. **Carried, unblocked, unchanged from phase 8**: prefill chunk-size sweep
   (128/256/512/1024, one warm point exists); EXP-018's memory.peak residual
   (note it is TWO residuals of opposite sign, see below); the pgsteal 2817
   repeat; Kaggle/SSH portability smoke; 13 slots/layer has no 4K measurement.
   The slot dial is a rider on item 1, not a phase: a better policy changes
   the hit-rate-per-slot curve and therefore the right dial.

7. **The drive reports `corruption_errs=138407`** on `/dev/nvme0n1p2`. It did
   not grow during any phase-9 run (`btrfs_session_grew: []`), so it is not
   touching these measurements, but it is unexplained on a benchmark machine
   and it sits next to a finding about that drive's read behaviour changing
   between sessions.

## PROCESS THAT WORKED

Research-first with findings reported before planning; plan with acceptance
criteria approved before any code; parallel lanes with strict file ownership;
adversarial review per wave with a specialist alongside the generic pass;
orchestrator takes the one measurement pass on a quiet machine; docs last.

Two things earned their keep this phase:

- **The specialist reviewer found the phase's only real blocker by execution**,
  not by reading: it mutated `at[j] = index * hidden` to `at[j] = j * hidden`
  and watched the fused tests stay green. Reading the diff would not have
  found it. Run one every wave.
- **Lanes were forbidden to measure.** Every timing number came from the
  orchestrator on a quiet machine, building each variant from `git archive` in
  an isolated tree. The one time a drift control failed (1.021x on wall time)
  it was caught, and the GEMV bucket it mattered for was stable at 1.004x.

## GOTCHAS

Phase 8's eleven still apply. These are new or sharpened.

1. **A median needs its spread beside it, and the noise floor needs an
   unchanged control.** `lm_head` moved 1.117x on code nothing touched. Any
   bucket-level claim smaller than that is noise. Phase 9 nearly shipped a
   spin policy on a 1.012x reading.
2. **Do not correlate two lists you extracted around a sort.** The orchestrator
   reported r = 0.159 between bandwidth and dispersion; the real figure is
   -0.043. `ag.sort()` ran between building the bandwidth list and the
   dispersion list, so it correlated sorted x against unsorted y. The
   conclusion survived; the number was garbage.
3. **A microbenchmark's cost figure does not survive a different call
   cadence.** 1.3 us of barrier measured in a hot loop became ~107 us in
   decode, because the loop never parked a worker. Ask what state the
   microbenchmark left the machine in.
4. **An L2-resident kernel fixture is not a reference for a streaming site.**
   EXP-001's 9.61 GB/s was quoted in three comments as the thing decode fell
   short of. It was never the right comparison.
5. **`git diff` is rewritten by the rtk hook into a summary and will not
   apply.** Use `rtk proxy "git diff"` when producing a patch. This cost a
   lane real time.
6. **Divide by the shard count once.** "~5 us of per-worker arithmetic" was
   32 us; the orchestrator divided by six twice and briefed a lane on it.
7. **A probe's own overhead lands where the cases are smallest.** Thread start
   and join sat inside the measurement timer: 0.22% of a 128-read case and
   ~21% of an 8-read window case. It biased the window result toward the
   conclusion being drawn. So did unvaried case ordering. Both are fixed;
   the corrected number moved *away* from the conclusion.
8. **EXP-018's residual is two residuals of opposite sign.** Phase 8 narrowed
   the fixed-tenant *over*prediction (~27 MiB, a constant). The older 99-105
   MiB *under*prediction is untouched and still unexplained. Do not merge them.
9. **`--list-regions` and friends must fail loudly.** The probe exited 0 when
   `filefrag` produced nothing at all, which a wrapper would take as success.
10. **The phase-8 sweep truncated its own decode split** to 12 lines and
    stopped at the first match, losing five of six rows at the 3,961 rung, and
    rendered the last scored run's split while EXP-023 quoted the first with
    nothing saying which. Both fixed in `scripts/phase9_decode_sweep.sh`.
11. **The prompt fixture matters.** `scratch/phase9/prompts/ctx512.txt` is a
    re-cut prompt and is NOT the file EXP-023 measured at 512
    (`long_00.txt`). Phase 9's warm A/B used it consistently across arms, which
    is fine, but its numbers must not be laid beside EXP-023's 512 rung as the
    same workload. The cold sweep uses phase 8's fixtures deliberately.
12. **A run's stderr carries two splits that share bucket names.** `prefill
    split` and `decode split (forward_token)` both have `attention`, `expert
    compute`, `expert io`, `projections`, `elementwise`. A regex over the whole
    stderr silently sums them: during EXP-025's analysis that produced a
    133 s GEMV inside a 44 s decode. Slice the block first. It was only caught
    because the number was absurd; a subtler mix would have shipped.
13. **A paired reference is not optional on this machine.** The byte-identical
    `8e1eee8` binary read 0.969x at 512 and 0.911x at 3,961 between EXP-023
    and EXP-025. Comparing a change against a published prior instead of an
    in-session baseline would have inverted the sign of the 4K result.
14. **Two preflight messages read as failures and are not.** The dirty-tree
    banner fired because `nohup` created `nohup.out` and the check asked
    `git status --porcelain`, which counts untracked files; the tell was a
    printed diff sha256 of `e3b0c442...`, sha256 of the empty string. And the
    slot self-test's five negative cases make the checker print `FAIL:` on
    purpose. Both cost a run before a single arm started. Both are fixed, but
    the general lesson stands: a preflight that cries wolf gets its whole
    output ignored, including the one line that matters.
15. **Read the per-source reclaim counters, never the bare `pgsteal`.** Every
    reclaim event in EXP-025 is `pgsteal_khugepaged` with `kswapd` and
    `direct` at zero, which is the huge-page daemon and not memory pressure.
    `pgsteal 147` recurs across two arms, the same shape as GOTCHA 11's 2,817,
    which was `kswapd`, so this does not explain it, but it does suggest a
    deterministic daemon is a likelier story than coincidence.

## STATE

    9e1c134  feat: measure physical dispersion, and read inside a file
    12856bb  feat: split decode's GEMV into own work and barrier wait
    ce9b0d3  perf: spin against the gap the pool actually sees      (reverted)
    70cf304  perf: fan out once per expert phase, not once per matrix
    88e3e9d  Revert "perf: spin against the gap the pool actually sees"
    16b30a2  fix: stop the window probe from flattering its own conclusion
    1b51978  test: pin staging to the routed index, and retire a refuted reference
    cd1547f  feat: adapt the cold sweep to pair against phase 8
    c50e514  docs: record EXP-024, and retire what phase 9 refuted
    9b11b6d  docs: hand phase 9 over
    c59f09e  fix: stop the sweep alarming its operator before step 0
    0d0cfc9  feat: tee the sweep to a transcript so it can be watched
    9744aa5  docs: record EXP-025, the fused fan-out paired cold

Branch `feat/pool-adaptive-spin` holds `ce9b0d3` for revisiting.

The cold sweep that produced EXP-025 is
`scratch/phase9/sweep-20260807-203558/`, run from `0d0cfc9`. Branch binary
sha256 `74822112a1d5...`, reference `d36036b6485b...` (`8e1eee8`, byte-identical
to EXP-023's). All 17 steps exit 0, hygiene PASS on all seven cold arms,
`slots=11 OK` asserted on all seven. Re-running it costs ~1 h 55 m:

    nohup bash scripts/phase9_decode_sweep.sh > /dev/null 2>&1 &
    tail -f scratch/phase9/sweep-*/run.log

Gate, all green on the final tree: `cargo fmt --check`, `cargo clippy
--all-targets -- -D warnings`, `cargo test` (606 passed), `cargo test -p
ramvamp-core --no-default-features` (478 passed).

Numerics, all green: `bitident.py` PASS 8/8; `greedy_regression.py` PASS with
top-1 24/24; `kl_vs_reference.py --refresh` gate 3 PASS (mean KL 1.039e-02,
worst prompt 2.721e-02); `model::prefill::tests::sweep_and_token_major_agree_
bit_for_bit` and its wide variant each confirmed `1 passed`.

The `--refresh` on the KL gate is not optional. Its cache is keyed on the
prompt name only, so without it the gate passes having tested nothing.
