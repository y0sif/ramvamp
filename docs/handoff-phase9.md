# Handoff: phase 9 to phase 10

Branch `feat/decode-compute`, 9 commits on `8e1eee8`. Not merged. The one
thing standing between it and `main` is a cold run that has not happened.

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
when it splits; a token from 1,345 to between 241 and 337. It is bit-safe
because a row of a fused space still computes one whole-row dot on the same
bytes, so the pool's tiling of the fused space restricts to a tiling of each
matrix's own rows.

Measured warm, ctx 512, 3 runs each, pooled GEMV bucket, medians with ranges:

    baseline       14.44 s  (14.35, 14.44, 15.06)
    adaptive spin  14.27 s  (13.99, 14.27, 14.48)
    fused          11.42 s  (11.36, 11.42, 11.48)
    both           11.38 s  (11.34, 11.38, 11.51)

**1.264x**, and by bucket: experts 9.94 to 7.20 (1.381x), projections 3.52 to
3.21 (1.097x), lm_head 1.05 to 0.94 on **unchanged code** (1.117x, which is
the noise floor for a bucket that size).

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

## WORK TO DO, IN PRIORITY ORDER

1. **Run the cold sweep. Nothing merges until this exists.**

       nohup bash scripts/phase9_decode_sweep.sh > /dev/null 2>&1 &

   ~1 h 55 m ESTIMATED. Five rungs on the branch binary, paired against the
   banked phase-8 reference at 512 and 3,961. The reference is
   `scratch/phase8-ref/ramvamp`, sha256 `d36036b6485b00e7…`, which is
   byte-identical to the branch binary EXP-023 published. **EXP-025 is
   reserved for the result** and the experiments index says so.
   Everything the phase claims about the fusion is warm until this runs.

2. **Decide the `attn_q` + `attn_v` fusion.** It is 1.097x against a 1.117x
   noise floor, it leaves the last shard ~24% long (4096 q4_k rows then 512
   q6_k rows on an even split), and it contributes 48 of the ~1,100 fan-outs a
   token that `70cf304` removes. Both reviewers flagged it independently. Drop
   it, or keep it and label it unattributable in EXP-025. It was deliberately
   fenced off from every fix lane, so it is clean either way. `forward.rs`
   around the `qv` buffer.

3. **Why are workers slower per row than the submitter?** This is the real
   phase-10 question. Six cores buy 1.43x on a site reading 1.99 GB a token.
   11.00 GB/s aggregate is far below what the memory system should give, so it
   is not obviously bandwidth saturation either. Candidates nobody has tested:
   memory-level parallelism per core, software prefetch, the access pattern
   into q4_k super-blocks, NUMA-ish effects of the hybrid part, or the
   submitter simply starting earlier. A cost-weighted `shard_range` is a
   separate, smaller lever and would also fix item 2's imbalance.

4. **Carried, unblocked, unchanged from phase 8**: prefill chunk-size sweep
   (128/256/512/1024, one warm point exists); EXP-018's memory.peak residual
   (note it is TWO residuals of opposite sign, see below); the pgsteal 2817
   repeat; Kaggle/SSH portability smoke; 13 slots/layer has no 4K measurement.

5. **The drive reports `corruption_errs=138407`** on `/dev/nvme0n1p2`. It did
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

Branch `feat/pool-adaptive-spin` holds `ce9b0d3` for revisiting.

Gate, all green on the final tree: `cargo fmt --check`, `cargo clippy
--all-targets -- -D warnings`, `cargo test` (606 passed), `cargo test -p
ramvamp-core --no-default-features` (478 passed).

Numerics, all green: `bitident.py` PASS 8/8; `greedy_regression.py` PASS with
top-1 24/24; `kl_vs_reference.py --refresh` gate 3 PASS (mean KL 1.039e-02,
worst prompt 2.721e-02); `model::prefill::tests::sweep_and_token_major_agree_
bit_for_bit` and its wide variant each confirmed `1 passed`.

The `--refresh` on the KL gate is not optional. Its cache is keyed on the
prompt name only, so without it the gate passes having tested nothing.
