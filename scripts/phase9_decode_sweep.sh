#!/usr/bin/env bash
# Phase 9 decode sweep: does the fused decode fan-out transfer to a cold,
# rule-2 run, and what does it do to decode tok/s across the context ladder.
#
#     nohup bash scripts/phase9_decode_sweep.sh > /dev/null 2>&1 &
#     DRY_RUN=1 bash scripts/phase9_decode_sweep.sh     # check the scaffolding
#
# Then read scratch/phase9/sweep-<stamp>/SUMMARY.txt.
#
# Adapted from scripts/phase8_decode_sweep.sh — the harness EXP-023 ran end
# to end — and through it from scripts/phase7_rerun_cold.sh. Everything those
# scripts learned the hard way is kept: the settle loop that waits for
# MemAvailable to hold above 6,000 MiB across four consecutive 15-second
# samples before any run starts, per-step exit codes captured into
# exitcodes.tsv rather than aborting on the first failure, a 60-second drain
# after each step so the page cache one run built does not contaminate the
# next, the build-then-assert-freshness pair, the self-tested slot assertion,
# the per-sweep stamping of every artifact, and a SUMMARY.txt tail.
#
# --------------------------------------- what changed from phase 8, and why --
#
#   1. The paired reference arm is re-pointed from scratch/phase7-ref
#      (T_BLOCK=4, main at d329890) to scratch/phase8-ref (8e1eee8, phase 8
#      complete). Phase 9's baseline is phase 8's result, not phase 7's.
#
#   2. Phase 8's slot-dial arms (1570M/12 slots, 1701M/13 slots) are DROPPED.
#      The dial is deliberately held at the shipped default this phase and
#      must not move, so re-measuring it is ~46 minutes spent confirming a
#      constant. The slot ASSERTION stays, and it is what proves the dial
#      really did stay put.
#
#   3. Phase 8's io_probe arms are DROPPED. Phase 9 ran the drive-side probes
#      separately and their result is final; running them again here would
#      only produce a second answer to a settled question.
#
#   4. The tail prints every stderr block WHOLE. Phase 8's printer found the
#      first line containing "split" or starting with "experts:", printed
#      twelve lines and broke. Because `experts:` matches first, the decode
#      split came out cut off after two of its six rows — see
#      scratch/phase8/sweep-20260806-165322/SUMMARY.txt lines 587-588, where
#      the 3,961-token arm's decode split ends one row in, at `attention:`,
#      and its expert-compute, expert-io, projections, elementwise and other
#      rows are absent from the artifact EXP-023 was written from. Phase 9 also emits a SECOND block, `decode
#      gemv split (submitting thread):`, which that printer would have
#      dropped entirely. The replacement takes each block's extent from
#      indentation, so it depends on no block's line count.
#
#   5. The tail names the run a quoted split came from. Phase 8 rendered the
#      LAST scored run and said so nowhere; EXP-023 quoted the FIRST. Two
#      different runs, one artifact, and nothing in it to tell them apart.
#      The split still comes from the last scored run — that is the one least
#      contaminated by warm-up — but the line above it now says which.
#
# ---------------------------------------------------------------- why 64 --
#
# Every decode step here is `--max-new 64`, deliberately. EXP-021's 4K point
# generated 8 tokens and its decode figure (1.47 tok/s) is dominated by the
# post-prefill cold-cache transient: the expert cache is empty when decode
# starts, so the first tokens pay a miss on nearly every routed expert.
# EXP-005 put the steady-state threshold around token 48. Eight tokens
# measures the transient; 64 measures decode. Repeating the 8-token shape
# across five context rungs would produce a clean-looking curve of the wrong
# quantity.
#
# ------------------------------------------------ what this script builds --
#
# Step 0 is `cargo build --release`, following scripts/phase7_overnight.sh.
# Without it the sweep measures whatever binary happens to be sitting in
# target/release, stamps it with `git rev-parse HEAD` and a sha256, and
# publishes the pair as if one had produced the other. That is not
# hypothetical: this script was written while target/release/ramvamp was a
# phase-7 build and the tree was several commits past it.
#
# The build alone is not enough, because a later edit can move it, skip it,
# or add a path that reaches the steps without it. So the build is followed
# by an assertion that the binary's mtime is not older than the newest
# tracked source under crates/ (plus Cargo.toml and Cargo.lock), and that
# assertion is FATAL. Belt and braces: the build makes the binary right, the
# assertion is what notices when it is not.
#
# SKIP_BUILD=1 skips step 0 for a binary you have just built by hand, and to
# exercise the assertion. It is safe only because the assertion still runs
# and still aborts.
#
# The freshness rule applies to exactly ONE path, target/release/ramvamp, and
# assert_binary_fresh is deliberately not parameterised. A path argument is
# how an exemption list starts, and an exemption list is how the binary under
# measurement eventually ends up on it. Every OTHER binary this sweep runs is
# identified by content hash instead, which is the stronger check anyway: a
# hash says which bytes ran, an mtime only says they are recent.
#
# ------------------------------------------- why there is a reference arm --
#
# The shipping change is 70cf304, "fan out once per expert phase, not once
# per matrix": decode's GEMV work is handed to the pinned compute pool once
# per expert phase instead of once per matrix. WARM, at ctx 512 over 63
# tokens, three runs, that moved the pooled GEMV buckets from 14.44 s to
# 11.42 s — 1.264x. Warm is a diagnostic and CLAUDE.md's rule is explicit
# that published numbers come from cold runs inside a memory.max=3G cgroup,
# so that ratio is not a result yet. This sweep is where it becomes one, or
# does not.
#
# EXP-023 is the cold baseline it has to beat, and EXP-023's decode tok/s
# ladder is (MEASURED, 11 slots, --warmup 1 --repeats 3 --max-new 64):
#
#   ctx   64   2.19   (1.94, 2.23, 2.19)
#   ctx  512   1.91   (1.91, 1.98, 1.91)
#   ctx 1024   1.82   (1.86, 1.81, 1.82)
#   ctx 2048   1.75   (1.81, 1.57, 1.75)   <- 15% spread, the widest
#   ctx 3961   1.46   (1.49, 1.44, 1.46)
#
# Those are ANOTHER session's numbers, and rule 3 forbids putting them on one
# curve with this sweep's. That is exactly why the reference arm exists: two
# rungs run twice, once on this tree's binary and once on
# scratch/phase8-ref/ramvamp, back to back in THIS session, so the comparison
# never has to cross a session boundary. EXP-023's ladder above is context
# for choosing the rungs, not a baseline to subtract.
#
# Which two rungs, and why those:
#
#   512   is the primary target. It is where the fused fan-out was measured
#         warm, so it is the one rung where a cold result can be held against
#         a warm prediction. It is also the TIGHTEST rung on EXP-023's ladder
#         (1.91, 1.98, 1.91 — under 4% spread), so an effect anywhere near
#         the warm 1.264x sits far outside that rung's own noise.
#
#   3961  is where GEMV's share is smallest, so it bounds the win from the
#         unfavourable side. EXP-023 measured attention at 33.5% of decode at
#         3,961 tokens against 5.9% at 512: the fan-out change cannot touch
#         attention, and at the top rung attention is a third of the budget.
#         If the change still wins there, it is not a short-context artifact.
#
# Not 2048, even though it is the noisiest rung (1.81, 1.57, 1.75 — 15%).
# Pairing would help it, but a pair there costs ~32 minutes to improve the
# error bar on a rung that is neither the target nor the bound, and its
# unpaired branch arm still lands on the curve. Two pairs, one at each end of
# how much room the change has, is the decisive shape.
#
# The reference binary's sha256 is
# d36036b6485b00e741b7448e8d963e8a0916eb0aeb36b69c48c89ea857eb8b4c, which is
# byte-identical to the branch binary EXP-023 measured (recorded there at
# `c78122b` as `d36036b6485b...`, docs/experiments/README.md). 8e1eee8 is the
# merge that made feat/decode main, and `git diff --name-only c78122b
# 8e1eee8` touches only docs/, so the two commits have the same runtime and
# the rebuild reproduced EXP-023's bytes exactly. The reference arm is not a
# lookalike; it is the executable phase 8 published from.
#
# The two arms of a pair run back to back, following how
# scripts/phase7_overnight.sh:145-170 pairs REF5 against RAMVAMP so both arms
# see the same machine state. Same prompt file, same --max-new, same warmup
# and repeats, nothing between them but their own settle and drain.
# docs/benchmark-machine.md is explicit that per-file bandwidth variance
# exceeds run-to-run variance, so an arm that touches a different file has
# moved its own baseline and is no longer a control.
#
# That binary is deliberately older than the tracked sources and must not
# trip the freshness assertion. It does not: the assertion is scoped to
# target/release/ramvamp alone, as above. What replaces it is
# verify_ref_binary, which re-hashes the reference in preflight and requires
# it to match both the SHA256 file banked beside it and the constant below.
#
# ------------------------------------------ why the slot count is asserted --
#
# NO arm in this sweep passes `--cache-bytes`. The slot dial is held at the
# runtime's own default for every rung, on purpose: phase 9 is measuring one
# change, and a dial that moved underneath it would be a second one. The
# assertion below is what turns "held" from an intention into a fact.
#
# `--cache-bytes` is a byte budget, not a slot count, and the runtime derives
# slots/layer from it. A slot costs the sum of every layer's page-aligned
# stride — 130.781 MiB for the shipped Qwen3-30B-A3B layout — and the default
# clears its threshold by well under 1 MiB:
#
#   1440M (ramvamp's default) -> 11 slots/layer  (needs 1438.6 MiB)
#
# A repack that changes the per-layer stride moves that threshold, and the
# run would look perfectly valid while measuring one slot fewer than it
# claims. So every cold step here is followed by a check that reads the slot
# count the runtime actually reported — `N expert slots/layer from a ...
# budget`, on the `model loaded in` line of the child stderr that cold_bench
# now keeps inside the --json summary — and fails that step if it is not the
# number the step intended. The check is a row in exitcodes.tsv like any
# other step; it never aborts the run.
#
# The check reads the summary with python3, not jq. jq is installed on the
# reference machine but no script in this repo uses it, python3 already runs
# every step here, and the slot count lives inside a JSON *string* field
# (runs[].stderr) that needs a regex either way.
#
# ------------------------------------- why every output path is per-sweep --
#
# cold_bench.py writes its `--json` summary as the very last thing main()
# does. Every failure path — including the OOM detector, "the inner run
# produced no result file (systemd-run exited N); MemoryMax may have
# OOM-killed it" — exits 2 before that write. So a failed arm leaves the
# file exactly as the previous sweep left it.
#
# That is not a corner case here. Re-running after a partial failure is the
# normal workflow, and EXP-021 measured the 3,961-token rung peaking at
# 2,920 MiB against a 3,072 MiB cap, so an OOM-killed arm at the top of the
# ladder is a live possibility rather than a surprise. With a fixed output
# path, the failed arm's slot check would read the previous sweep's summary,
# find the right slot count, and record `slots=11 OK`; the tail would then
# reprint that summary's hygiene verdict and tok/s into THIS sweep's
# SUMMARY.txt with nothing marking where they came from.
#
# Two changes, and both are load-bearing:
#
#   1. Every artifact this sweep writes carries $STAMP, exactly as $OUT
#      already did — the summaries and the per-step cold_bench workdirs.
#      Sweeps can no longer collide, and no
#      sweep destroys the evidence of an earlier one. That is why the paths
#      are stamped rather than `rm -f`'d before each arm: deleting works, but
#      it throws away the previous sweep's good arms to protect against its
#      bad ones, and the re-run-after-a-failure workflow is precisely when
#      those old arms are still the only copy.
#
#   2. Stamping alone fails quietly — the file is simply absent — and the
#      instruction here is to fail loudly. So each arm records the path it
#      expects and the second it started in $OUT/summaries.tsv, and both the
#      slot check and the tail refuse any summary that is missing, or whose
#      mtime predates the step that was supposed to write it. Missing and
#      stale are reported as failures with their own exitcodes.tsv rows and
#      their own line in the tail, not as silence.
#
# The mtime gate is redundant while the paths are stamped. It is here for the
# same reason the freshness assertion sits behind the build: the structural
# fix is one edit away from being undone, and the loud check is what notices.

set -u
set -o pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd -P)

STAMP=$(date +%Y%m%d-%H%M%S)
# Every artifact below is gated on being at least this old. Taken before any
# step so a step can never be older than the sweep that ran it.
SWEEP_T0=$(date +%s)
OUT="$ROOT/scratch/phase9/sweep-$STAMP"
mkdir -p "$OUT" "$ROOT/scratch/cold-bench" || exit 1
SUMMARY="$OUT/SUMMARY.txt"
MANIFEST="$OUT/summaries.tsv"
: > "$SUMMARY"
: > "$OUT/exitcodes.tsv"
: > "$MANIFEST"

RAMVAMP="$ROOT/target/release/ramvamp"
RVMP="$ROOT/models/qwen3.rvmp"

# The phase-8 reference arm, built from 8e1eee8 and banked beside its own
# COMMIT and SHA256, following the scratch/phase5-ref/ and scratch/phase7-ref/
# convention phase7_overnight.sh and phase8_decode_sweep.sh already use.
REF8="$ROOT/scratch/phase8-ref/ramvamp"
REF8_SHA_FILE="$ROOT/scratch/phase8-ref/SHA256"
REF8_COMMIT_FILE="$ROOT/scratch/phase8-ref/COMMIT"

# The bytes EXP-023 measured. Pinned here as a constant and not merely read
# from the SHA256 file beside the binary: if the reference is ever re-banked,
# a SHA256 regenerated alongside it agrees with itself and is still not
# phase 8's binary. Both must match.
REF8_EXPECT_SHA=d36036b6485b00e741b7448e8d963e8a0916eb0aeb36b69c48c89ea857eb8b4c

# Per-sweep output paths. See "why every output path is per-sweep" above.
CB_JSON_DIR="$ROOT/scratch/cold-bench"
CB_WORK_DIR="scratch/phase9/cold-bench/$STAMP"

# Set by run(); read by check_slots() as the earliest mtime a summary written
# by that step could possibly have.
STEP_T0=$SWEEP_T0

# Slots/layer each arm must report. 11 is ramvamp's default budget (1440M),
# which every step that does not pass --cache-bytes inherits.
SLOTS_DEFAULT=11

# The context ladder, unchanged from phase 8 — the same five files, byte for
# byte, obtained the same way. That is the point rather than an oversight:
# EXP-023 measured this ladder, and a rung measured on a different prompt is
# not the same rung. Every rung key is the prompt's true token count, so a
# label, a filename and a row can never disagree with the workload. The 512
# and 3961 rungs are the fixtures phase 7 already measured on, reused
# unchanged so the entries share a workload where they overlap; 64, 1024
# and 2048 are cut by scripts/make_ctx_prompt.py from
# models/llamacpp-ref/llamacpp_ref/long_02.txt.
#
# scratch/phase9/prompts/ctx512.txt exists and is deliberately NOT used here.
# It is phase 9's own 512-token cut (sha256 fc365369e7bfd9c5..., 1,995
# bytes), made for the warm wave-1 A/B where the only requirement was that
# the two arms of that A/B agree with each other. It is a DIFFERENT file from
# models/llamacpp-ref/llamacpp_ref/long_00.txt (d1b6c407c55aa95f...,
# 2,002 bytes), which is the 512 rung phase 7 and phase 8 both measured.
# Substituting it would move the primary target rung off its own history
# while every label in the artifact still said 512.
#
# None of the five is committed, and neither is that source text: .gitignore
# excludes /models/ (line 5) and /scratch/ (line 9). So this ladder is not a
# set of committed fixtures. It is a set of files reproducible by anyone
# holding the same source text, and the SHA-256s below are how a reader finds
# out whether they are holding it. For the three generated rungs the
# provenance is recorded in scratch/phase8/prompts/MANIFEST.json, which
# carries the source's own sha256 (52d734947d197b19..., 15,023 bytes) next to
# each fixture's; the two reused rungs are phase 7's, identified here by hash
# alone. Token counts are the tokenizer's, verified:
#
#   rung   file                                       tokens  bytes  sha256
#     64   scratch/phase8/prompts/ctx64.txt               64     264  90509897ed91916d
#    512   models/.../llamacpp_ref/long_00.txt           512   2,002  d1b6c407c55aa95f
#   1024   scratch/phase8/prompts/ctx1024.txt          1,024   4,111  643f41879677211d
#   2048   scratch/phase8/prompts/ctx2048.txt          2,048   8,752  c915b18912d2334e
#   3961   scratch/ctx4k/p4k.txt                       3,961  17,000  68582aae37b920ef
#
# The preflight re-hashes all five on every run and prints them into
# SUMMARY.txt, so a fixture that has been regenerated or swapped shows up
# next to the numbers it produced rather than inside them.
#
# The top rung is 3,961 tokens, not 4,096: CONTEXT_CAP is 4096 and
# `prompt + max-new` must fit under it, so 3,961 + 64 = 4,025 is as close to
# the cap as this sweep can sit. It is the rung docs/ calls "4K"; it is never
# 4,096 tokens, and nothing this script writes — step label, exitcodes.tsv
# row, summary filename, workdir — says 4096.
#
# None of the five files ends in a newline, so cold_bench.py's
# strip-one-trailing-newline delivers the file bytes unchanged and the file
# SHA-256 above is also the delivered SHA-256. MANIFEST.json confirms that
# for the generated three: delivered_bytes and delivered_sha256 equal
# file_bytes and file_sha256 in every entry.
CTX_RUNGS=(64 512 1024 2048 3961)

prompt_for() {
    case "$1" in
        64)   printf '%s\n' "$ROOT/scratch/phase8/prompts/ctx64.txt" ;;
        512)  printf '%s\n' "$ROOT/models/llamacpp-ref/llamacpp_ref/long_00.txt" ;;
        1024) printf '%s\n' "$ROOT/scratch/phase8/prompts/ctx1024.txt" ;;
        2048) printf '%s\n' "$ROOT/scratch/phase8/prompts/ctx2048.txt" ;;
        3961) printf '%s\n' "$ROOT/scratch/ctx4k/p4k.txt" ;;
        *)    return 1 ;;
    esac
}

# The rungs that also run a phase-8 reference arm. See "why there is a
# reference arm" above. 512 is the primary target and the tightest rung;
# 3961 is where GEMV has the least room, so it bounds the result.
PAIRED_RUNGS=(512 3961)

is_paired_rung() {
    local want=$1 r
    for r in "${PAIRED_RUNGS[@]}"; do
        if [ "$r" = "$want" ]; then
            return 0
        fi
    done
    return 1
}

# MemAvailable wanted before a cold run starts, in MiB. The 3,961-token
# workload peaks near 2,920 MiB (EXP-021), so this is real slack rather than
# just enough.
WANT_AVAIL_MIB=6000
SETTLE_SAMPLES=4
SETTLE_MAX_WAIT=900
DRAIN_S=60

say() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
avail_mib() { awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
swap_mib() { awk '/^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{print int((t-f)/1024)}' /proc/meminfo; }

settle() {
    local waited=0 stable=0 a s
    say "    settling: want MemAvailable >= ${WANT_AVAIL_MIB} MiB for ${SETTLE_SAMPLES} samples"
    while [ "$waited" -lt "$SETTLE_MAX_WAIT" ]; do
        a=$(avail_mib); s=$(swap_mib)
        if [ "$a" -ge "$WANT_AVAIL_MIB" ]; then
            stable=$((stable + 1))
        else
            stable=0
        fi
        if [ "$stable" -ge "$SETTLE_SAMPLES" ]; then
            say "    settled: MemAvailable ${a} MiB, swap in use ${s} MiB, waited ${waited}s"
            return 0
        fi
        sleep 15
        waited=$((waited + 15))
    done
    a=$(avail_mib); s=$(swap_mib)
    say "    NOT settled after ${SETTLE_MAX_WAIT}s: MemAvailable ${a} MiB, swap ${s} MiB."
    say "    running anyway; if this one comes back DIRTY, free memory and retry."
    return 0
}

# run [--no-settle] <label> <logfile> <command...>
# Records the command's OWN exit code and never aborts the script. The settle
# wait is outside the timing, so the recorded seconds are the step's work.
#
# --no-settle is for the build: compiling does not need a quiet machine, and
# blocking the build behind a 15-minute MemAvailable wait would only delay
# the heat it then has to shed. The drain after it is that heat-shed.
#
# Sets STEP_T0 to the second the step's work began, which check_slots uses as
# the floor for "a summary this step could have written".
run() {
    local settle_first=1
    if [ "$1" = "--no-settle" ]; then
        settle_first=0
        shift
    fi
    local label=$1 log=$2
    shift 2
    printf '\n>>> [%s] %s\n' "$(stamp)" "$label" | tee -a "$SUMMARY"
    local t0 t1 rc hyg
    if [ "${DRY_RUN:-0}" = "1" ]; then
        STEP_T0=$(date +%s)
        printf 'DRY_RUN, would have run:\n%s\n' "$*" > "$OUT/$log"
        printf '    would run: %s\n' "$*" | tee -a "$SUMMARY"
        printf '%s\t%s\t%s\t%s\n' "$label" "DRY" "0" "dry-run" >> "$OUT/exitcodes.tsv"
        return 0
    fi
    if [ "$settle_first" -eq 1 ]; then
        settle
    fi
    t0=$(date +%s)
    STEP_T0=$t0
    "$@" > "$OUT/$log" 2>&1
    rc=$?
    t1=$(date +%s)
    hyg=$(grep -a -o 'measurement hygiene: [A-Z]*' "$OUT/$log" | tail -1)
    printf '    exit=%d  %dm%02ds  %s  log=%s\n' \
        "$rc" $(( (t1-t0)/60 )) $(( (t1-t0)%60 )) "${hyg:-hygiene: ?}" "$log" | tee -a "$SUMMARY"
    printf '%s\t%s\t%s\t%s\n' "$label" "$rc" "$((t1-t0))" "${hyg:-?}" >> "$OUT/exitcodes.tsv"
    # Let the page cache the run just built drain before the next one.
    sleep "$DRAIN_S"
}

skip() {
    local label=$1 why=$2
    printf '\n>>> [%s] SKIPPED: %s\n' "$(stamp)" "$label" | tee -a "$SUMMARY"
    say "    $why"
    printf '%s\t%s\t%s\t%s\n' "$label (skipped)" "-" "0" "skipped" >> "$OUT/exitcodes.tsv"
}

# ------------------------------------------------- the slot-count assertion --
#
# Written to a file rather than inlined so that the preflight self-test below
# exercises the same bytes the real steps do. A self-test against a copy
# tests the copy. It is also archived with the sweep, so the assertion a
# given SUMMARY.txt was produced under can be read back later.
#
# Exit codes are distinguished on purpose:
#   0  the summary exists, is this step's, and reports the intended dial
#   1  the summary reports a DIFFERENT dial — a real, measured disagreement
#   3  there is nothing to read: missing, stale, unreadable, or no slot line
# 1 invalidates a number. 3 means the step produced no number at all.
PYCHECK="$OUT/check_slots.py"
cat > "$PYCHECK" <<'PYEOF'
"""Assert the slot dial a cold_bench summary actually reports.

usage: check_slots.py <summary.json> <want-slots> <min-mtime-epoch>

<min-mtime-epoch> is the second the step that should have written this
summary began; 0 disables the check. cold_bench.py writes --json last, after
every failure path has already exited, so a summary older than its own step
was written by an earlier sweep and describes an earlier binary.
"""
import json
import os
import re
import sys
import time

MISSING = 3
WRONG = 1

path = sys.argv[1]
want = int(sys.argv[2])
min_mtime = float(sys.argv[3])

# `model loaded in 1.30s (512 prompt tokens); 6 compute shards, 11 expert
#  slots/layer from a 1.4 GiB budget, 0 reads`
SLOTS_RE = re.compile(r"(\d+)\s+expert slots/layer from a ([^,]+) budget")


def when(epoch):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


if not os.path.isfile(path):
    print(f"    FAIL: {path} does not exist.")
    print(f"    cold_bench.py writes its --json summary as the last thing "
          f"main() does, so every failure path -- including the OOM detector "
          f"-- exits before the file appears. No file means this step "
          f"produced no measurement. An unverified dial is not a "
          f"measurement.")
    raise SystemExit(MISSING)

mtime = os.path.getmtime(path)
if min_mtime > 0 and mtime < min_mtime:
    print(f"    FAIL: {path} is STALE.")
    print(f"    Its mtime {when(mtime)} predates the start of the step that "
          f"was supposed to write it ({when(min_mtime)}), so this step wrote "
          f"nothing and the file belongs to an earlier sweep. Refusing to "
          f"certify another run's numbers as this one's.")
    raise SystemExit(MISSING)

try:
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
except (OSError, ValueError) as exc:
    print(f"    FAIL: cannot read {path}: {exc}")
    raise SystemExit(MISSING)

meta = summary.get("cache_bytes") or {}
print(f"    budget requested: {meta.get('value')} "
      f"(source: {meta.get('source')})")

seen = []
for run in summary.get("runs", []):
    match = SLOTS_RE.search(run.get("stderr") or "")
    if match:
        seen.append((run.get("label"), int(match.group(1)), match.group(2)))

if not seen:
    print(f"    FAIL: no `N expert slots/layer` line in any recorded stderr "
          f"of {os.path.basename(path)}. The count cannot be confirmed, so "
          f"neither can what this step measured. Check that cold_bench.py "
          f"still keeps the child stderr in the summary, and look at the "
          f"<workdir>/runNN.json.stderr sidecars.")
    raise SystemExit(MISSING)

for label, slots, budget in seen:
    print(f"    {label:<18} {slots} slots/layer from a {budget} budget")

wrong = [s for _, s, _ in seen if s != want]
if wrong:
    distinct = sorted(set(wrong))
    print(f"    FAIL: expected {want} slots/layer, saw {distinct}.")
    print(f"    A slot costs the sum of every layer's page-aligned stride "
          f"(130.781 MiB on the shipped layout) and these budgets clear "
          f"their thresholds by under 1 MiB, so a repack that changed the "
          f"stride would move them. This step measured a different dial "
          f"than it is labelled with. Do not publish it; recompute the "
          f"budget for {want} slots and re-run.")
    raise SystemExit(WRONG)

print(f"    OK: all {len(seen)} runs report {want} slots/layer "
      f"(summary written {when(mtime)})")
raise SystemExit(0)
PYEOF

# run_slot_check <summary.json> <expected-slots> <mtime-floor> <sink>
#
# The single place the checker is invoked. Both the real check and its
# self-test go through it, which is the point: the self-test used to call
# `python3 "$PYCHECK" ... > file` and read `$?` while the real path used
# `... | tee -a "$SUMMARY"` and read `${PIPESTATUS[0]}`. Same Python, but a
# different four lines of bash deciding the verdict — so a later edit adding
# a pipe stage here, or reverting to a bare `$?`, would make every slot check
# record exit 0 ("slots=N OK") while the self-test still printed 6/6 OK.
# This project has already shipped two false PASSes from exactly that shape
# of mistake, and `cargo test <bare_name> -- --exact` fooled two lanes on
# this branch the same way. One invocation, exercised by its own test.
run_slot_check() {
    python3 "$PYCHECK" "$1" "$2" "$3" 2>&1 | tee -a "$4"
    return "${PIPESTATUS[0]}"
}

# check_slots <label> <summary.json> <expected-slots>
#
# Reads the slot count the runtime reported and fails the check if it is not
# what the step's budget was chosen to buy, or if the summary is missing or
# older than the step that should have written it. Its own row in
# exitcodes.tsv, so a wrong dial is recorded as a failure without costing the
# rest of the run. No settle and no drain: this touches one small JSON file.
#
# Also records the arm in $MANIFEST, so the tail reports on exactly the
# summaries this sweep expected rather than globbing for whatever is lying
# around. The recording happens in both modes: a DRY_RUN manifest is how the
# scaffolding's coverage of the arms gets checked.
# The fourth argument is the second that arm's own run began, and it defaults
# to the most recent one. Pass it explicitly whenever a check does not
# immediately follow its own run: step group 1 defers both checks of a paired
# rung until after the pair, so the branch arm's check would otherwise be
# floored at the REFERENCE arm's start time and condemn a perfectly good
# summary as stale.
check_slots() {
    local label=$1 json=$2 want=$3 t0=${4:-$STEP_T0} rc verdict
    printf '%s\t%s\t%s\t%s\n' "$label" "$json" "$want" "$t0" >> "$MANIFEST"
    printf '\n>>> [%s] check: %s\n' "$(stamp)" "$label" | tee -a "$SUMMARY"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        say "    would assert $want expert slots/layer in $json"
        say "    (written no earlier than $(date -d "@$t0" '+%F %T'))"
        say "    the assertion itself was exercised by the preflight self-test"
        printf '%s\t%s\t%s\t%s\n' "$label" "DRY" "0" "dry-run" >> "$OUT/exitcodes.tsv"
        return 0
    fi
    run_slot_check "$json" "$want" "$t0" "$SUMMARY"
    rc=$?
    case "$rc" in
        0) verdict="slots=$want OK" ;;
        1) verdict="SLOT COUNT WRONG" ;;
        3) verdict="NO SUMMARY (missing/stale/unreadable)" ;;
        *) verdict="CHECK FAILED (exit $rc)" ;;
    esac
    printf '    exit=%d  %s\n' "$rc" "$verdict" | tee -a "$SUMMARY"
    printf '%s\t%s\t%s\t%s\n' "$label" "$rc" "0" "$verdict" >> "$OUT/exitcodes.tsv"
}

# selftest_check_slots
#
# Runs the assertion above against synthetic summaries with known answers.
# Six cases, ~1 second, and it runs in both modes: under DRY_RUN it is the
# only coverage the assertion gets (there are no real summaries to read), and
# in a real run it is a cheap way to find out that the checker is broken
# before two hours of measurement depend on it rather than after.
selftest_check_slots() {
    local dir="$OUT/selftest" now pyrc fails=0 total=0
    now=$(date +%s)
    mkdir -p "$dir" || { say "  self-test: cannot create $dir"; FATAL=1; return 1; }
    python3 - "$dir" <<'PYEOF'
import json
import os
import sys

d = sys.argv[1]


def loaded(slots):
    return (f"model loaded in 1.30s (512 prompt tokens); 6 compute shards, "
            f"{slots} expert slots/layer from a 1.5 GiB budget, 0 reads")


def summary(stderr):
    return {
        "cache_bytes": {"value": "1570M", "source": "argument"},
        "median": {"decode_tok_s": 1.0},
        "runs": [{"label": "scored", "stderr": stderr}],
    }


with open(os.path.join(d, "good.json"), "w", encoding="utf-8") as f:
    json.dump(summary(loaded(12)), f)
with open(os.path.join(d, "wrong.json"), "w", encoding="utf-8") as f:
    json.dump(summary(loaded(11)), f)
with open(os.path.join(d, "noline.json"), "w", encoding="utf-8") as f:
    json.dump(summary("model loaded in 1.30s, 0 reads"), f)
with open(os.path.join(d, "garbage.json"), "w", encoding="utf-8") as f:
    f.write("{not json at all")
PYEOF
    pyrc=$?
    if [ "$pyrc" -ne 0 ]; then
        say "  self-test: could not write the synthetic summaries (exit $pyrc)"
        FATAL=1
        return 1
    fi

    local spec name want exp json minm rc rest
    for spec in \
        "fresh, correct dial|12|0|$dir/good.json|$((now - 60))" \
        "wrong dial|12|1|$dir/wrong.json|0" \
        "summary missing|12|3|$dir/never-written.json|0" \
        "summary older than its step|12|3|$dir/good.json|$((now + 3600))" \
        "no slot line in stderr|12|3|$dir/noline.json|0" \
        "unreadable summary|12|3|$dir/garbage.json|0"
    do
        name=${spec%%|*}; rest=${spec#*|}
        want=${rest%%|*}; rest=${rest#*|}
        exp=${rest%%|*};  rest=${rest#*|}
        json=${rest%%|*}; minm=${rest##*|}
        total=$((total + 1))
        # Silenced at the call site, deliberately, and not by a quieter
        # variant of run_slot_check: five of these six cases are negative, so
        # making the checker print its rejection is the whole point of them.
        # On a console those lines read as the sweep collapsing rather than as
        # the checker working, and phase 9 lost a run to exactly that
        # misreading before a single arm had started. The text still lands in
        # the case's own .out file, and the mismatch branch below prints it.
        # Redirecting here keeps the one invocation the comment above
        # run_slot_check insists on: the self-test still exercises the shipped
        # path, `tee`'s file write and the `${PIPESTATUS[0]}` verdict included.
        run_slot_check "$json" "$want" "$minm" "$dir/$total.out" >/dev/null
        rc=$?
        if [ "$rc" -eq "$exp" ]; then
            say "  self-test OK   exit=$rc  $name  (rejection text in $total.out)"
        else
            say "  self-test FAIL exit=$rc want=$exp  $name"
            say "    see $dir/$total.out"
            fails=$((fails + 1))
        fi
    done

    if [ "$fails" -ne 0 ]; then
        say "MISSING: the slot assertion does not behave as specified"
        say "  ($fails of $total cases wrong). Every slot check below would be"
        say "  meaningless, so this stops the sweep rather than running it."
        FATAL=1
        printf '%s\t%s\t%s\t%s\n' "check_slots self-test" "1" "0" \
            "$fails/$total WRONG" >> "$OUT/exitcodes.tsv"
        return 1
    fi
    printf '%s\t%s\t%s\t%s\n' "check_slots self-test" "0" "0" \
        "$total/$total OK" >> "$OUT/exitcodes.tsv"
    return 0
}

# newest_tracked_source
# Prints "<mtime-epoch> <path>" for the newest tracked file that can change
# what target/release/ramvamp is. awk carries the maximum rather than
# `sort -rn | head -1`: head closes the pipe on its first line, which
# SIGPIPEs sort, which pipefail then reports as a failed pipeline.
newest_tracked_source() {
    git ls-files -z -- crates Cargo.toml Cargo.lock 2>/dev/null \
        | xargs -0 -r stat -c '%Y %n' 2>/dev/null \
        | awk 'NR == 1 || $1 > m { m = $1; p = $0 } END { if (NR) print p }'
}

# assert_binary_fresh
# Returns 0 if target/release/ramvamp is at least as new as everything that
# feeds it, non-zero otherwise. The caller decides what to do about it.
#
# Takes no path argument, on purpose. The freshness rule covers exactly the
# binary this sweep builds; every other binary it runs — today just the
# phase-8 reference, which is deliberately older than the tracked sources —
# is identified by content hash instead. Parameterising this would turn one
# rule into an exemption list, and the binary under measurement is precisely
# the thing that must never appear on such a list. The guard below is what
# keeps that true if $RAMVAMP is ever repointed.
assert_binary_fresh() {
    local newest bin_mtime src_mtime src_path
    if [ "$RAMVAMP" != "$ROOT/target/release/ramvamp" ]; then
        say "FATAL: the freshness assertion is scoped to"
        say "  $ROOT/target/release/ramvamp"
        say "  but \$RAMVAMP is now $RAMVAMP."
        say "  Either the sweep is measuring a binary it did not build, or the"
        say "  assertion has been pointed away from the one it did. Both are"
        say "  reasons to stop, not to widen the scope."
        return 1
    fi
    if [ ! -x "$RAMVAMP" ]; then
        say "FATAL: $RAMVAMP does not exist, or is not executable."
        say "  Step 0 was supposed to produce it. Read $OUT/00-build.log."
        return 1
    fi
    bin_mtime=$(stat -c %Y "$RAMVAMP" 2>/dev/null) || bin_mtime=""
    newest=$(newest_tracked_source)
    if [ -z "$bin_mtime" ] || [ -z "$newest" ]; then
        say "FATAL: cannot compare the binary against the tracked sources"
        say "  (git ls-files or stat produced nothing). This assertion is the"
        say "  only thing between a stale binary and a published number, so a"
        say "  check that cannot run is a stop, not a warning."
        return 1
    fi
    src_mtime=${newest%% *}
    src_path=${newest#* }
    say "  binary mtime : $(date -d "@$bin_mtime" '+%F %T')  target/release/ramvamp"
    say "  newest source: $(date -d "@$src_mtime" '+%F %T')  $src_path"
    if [ "$bin_mtime" -lt "$src_mtime" ]; then
        say "FATAL: target/release/ramvamp is OLDER than $src_path."
        say "  This sweep would record \`git rev-parse HEAD\` and the binary's"
        say "  sha256 side by side in SUMMARY.txt, and nothing else ties one"
        say "  to the other. Measuring this binary would stamp an older"
        say "  build's numbers with this tree's commit. Build it:"
        say "      cargo build --release"
        return 1
    fi
    say "  OK: the binary is not older than any tracked source under crates/."
    return 0
}

# verify_ref_binary
# What stands in for the freshness assertion on the phase-8 reference arm.
# The reference is meant to be old — it is 8e1eee8's build — so an mtime says
# nothing useful about it. Its identity is its bytes, and both the SHA256
# banked beside it and the constant pinned at the top of this script have to
# agree with them. FATAL on any disagreement: a reference arm that is not the
# binary it claims to be turns the paired A/B into an unlabelled comparison
# of two unknowns.
verify_ref_binary() {
    local recorded computed commit
    if [ ! -x "$REF8" ]; then
        say "MISSING: $REF8"
        say "  This is the phase-8 reference arm's binary, built from 8e1eee8."
        say "  Without it the sweep can measure a decode curve but cannot"
        say "  attribute any of it to the fused fan-out (70cf304), which is"
        say "  the change this phase exists to price and has no cold entry."
        FATAL=1
        return 1
    fi
    if [ ! -e "$REF8_SHA_FILE" ]; then
        say "MISSING: $REF8_SHA_FILE"
        say "  The reference binary is identified by its hash, so the banked"
        say "  hash is not optional bookkeeping."
        FATAL=1
        return 1
    fi
    recorded=$(awk 'NR == 1 { print $1 }' "$REF8_SHA_FILE")
    computed=$(sha256sum "$REF8" | cut -d' ' -f1)
    if [ -z "$recorded" ] || [ -z "$computed" ]; then
        say "FATAL: could not read or compute the reference binary's sha256."
        FATAL=1
        return 1
    fi
    if [ "$computed" != "$recorded" ]; then
        say "FATAL: $REF8 does not match its banked SHA256."
        say "  banked   $recorded"
        say "  computed $computed"
        say "  The file beside the binary describes a different binary. Do not"
        say "  guess which one is right; re-bank the reference from 8e1eee8."
        FATAL=1
        return 1
    fi
    if [ "$computed" != "$REF8_EXPECT_SHA" ]; then
        say "FATAL: $REF8 is not the binary EXP-023 measured."
        say "  expected $REF8_EXPECT_SHA"
        say "  computed $computed"
        say "  Its own SHA256 file agrees with it, which is exactly the case"
        say "  that a self-consistent re-bank produces. The reference arm is"
        say "  supposed to be phase 8's published bytes, not a lookalike, so"
        say "  this is a stop."
        FATAL=1
        return 1
    fi
    commit=$(head -1 "$REF8_COMMIT_FILE" 2>/dev/null)
    say "  reference binary: sha256 matches its banked SHA256 and EXP-023's"
    say "    $computed"
    say "    commit: ${commit:-<no COMMIT file>}"
    say "    exempt from the freshness assertion by design; it is 8e1eee8's"
    say "    build and is meant to be older than the tracked sources."
    return 0
}

# require_flags <what-needs-them> <script> <flag>...
# Captured into a variable and matched with `case`, not piped into `grep -q`:
# grep -q exits on the first match and would SIGPIPE the writer, which
# pipefail then reports as a failure to read the help at all.
require_flags() {
    local what=$1 script=$2
    shift 2
    local nflags=$# help rc flag missing=0
    help=$(python3 "$script" --help 2>&1)
    rc=$?
    if [ "$rc" -ne 0 ]; then
        say "MISSING: \`$(basename "$script") --help\` exited $rc; the harness is broken."
        FATAL=1
        return 1
    fi
    for flag in "$@"; do
        case "$help" in
            *"$flag"*) ;;
            *)
                say "MISSING: $(basename "$script") has no $flag flag."
                missing=$((missing + 1)) ;;
        esac
    done
    if [ "$missing" -ne 0 ]; then
        say "  $what cannot run without them."
        FATAL=1
        return 1
    fi
    say "  $(basename "$script"): all $nflags flags this sweep passes are present"
    return 0
}

say "ramvamp phase 9 — cold decode sweep: the fused fan-out against context"
say "started $(stamp)"
say "output  $OUT"
say "commit  $(git describe --always --dirty 2>/dev/null) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
# A dirty tree is the normal state when measuring before committing, so this
# records rather than refuses. It has to record something, though: the paired
# arms are two different binaries, and `git rev-parse HEAD` alone returns the
# same commit for both whenever the branch's change is uncommitted — which is
# exactly the state this sweep was written in. `--dirty` above distinguishes
# them; the diff hash below says *which* uncommitted tree, so the experiments
# entry can name a binary rather than gesture at a branch.
# `git status --porcelain` is the wrong question and phase 9 lost a run to it:
# it counts untracked files, so `nohup bash scripts/phase9_decode_sweep.sh &`
# creates nohup.out and the sweep then declares its own tree dirty. The banner
# fired with a diff sha256 of e3b0c442..., which is sha256 of nothing at all.
# What actually threatens the branch arm's provenance is a source change that
# is in the binary but in no commit, so ask that instead: tracked
# modifications, plus untracked files somewhere cargo would compile them from.
# An untracked file outside those paths cannot reach the binary and is noise.
tracked_dirty="$(git diff HEAD 2>/dev/null)"
untracked_build="$(git ls-files --others --exclude-standard \
    -- crates Cargo.toml Cargo.lock 2>/dev/null)"
if [ -n "$tracked_dirty" ] || [ -n "$untracked_build" ]; then
    say "        WORKING TREE IS DIRTY. The commit above is not sufficient"
    say "        provenance for the branch arm: its binary contains changes"
    say "        that are in no commit. Diff sha256 over \`git diff HEAD\`:"
    say "          $(printf '%s' "$tracked_dirty" | sha256sum | cut -d' ' -f1)"
    if [ -n "$untracked_build" ]; then
        say "        Untracked files under crates/ that cargo would compile:"
        printf '%s\n' "$untracked_build" | while IFS= read -r f; do
            [ -n "$f" ] && say "          $f"
        done
    fi
    say "        Record that beside the binary sha256 in the EXP entry, or"
    say "        commit before measuring and re-run. The reference arm is"
    say "        unaffected: it is pinned by hash, not by this tree."
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
    say "MODE    DRY_RUN=1 — nothing is measured, only the scaffolding runs"
fi
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    say "MODE    SKIP_BUILD=1 — step 0 is skipped; the freshness assertion is"
    say "        what decides whether the binary you already have is usable"
fi
say ""

# ------------------------------------------------------- runtime estimate --
say "--- estimated wall time (ESTIMATED, not measured) ---"
say ""
say "Derived from the phase-8 sweep's own MEASURED step walls, which ran this"
say "exact seven-arm list on this machine and this drive"
say "(scratch/phase8/sweep-20260806-165322/exitcodes.tsv):"
say ""
say "  measured  ctx   64  branch arm              202 s"
say "  measured  ctx  512  branch arm              327 s"
say "  measured  ctx  512  reference arm           325 s"
say "  measured  ctx 1024  branch arm              517 s"
say "  measured  ctx 2048  branch arm              921 s"
say "  measured  ctx 3961  branch arm            1,796 s"
say "  measured  ctx 3961  reference arm         1,787 s"
say "  measured  phase 8 end to end, 12 timed steps   2 h 41 m"
say "            (16:53:22 -> 19:34:11)"
say ""
say "  step (1 warmup + 3 scored = 4 runs)               est step"
say "  ctx   64  --max-new 64                               3 min"
say "  ctx  512  --max-new 64                               6 min"
say "  ctx  512  phase-8 reference                          6 min"
say "  ctx 1024  --max-new 64                               9 min"
say "  ctx 2048  --max-new 64                              16 min"
say "  ctx 3961  --max-new 64                              30 min"
say "  ctx 3961  phase-8 reference                         30 min"
say "                              cold-run total         ~98 min"
say "  settle (>= 45 s) + drain (60 s) x 7 steps           ~13 min"
say "  slot-count checks (7 x a few hundred ms)             <1 min"
say "  step 0, cargo build --release + 60 s heat-shed       ~4 min"
say ""
say "  TOTAL  ~1 h 55 m   ESTIMATED"
say ""
say "  Materially less than phase 8's ~2 h 49 m ESTIMATED / 2 h 41 m"
say "  MEASURED, and the difference is entirely the arms this sweep does not"
say "  run: three slot-dial arms (~46 min) and two io_probe steps (~5 min)."
say ""
say "  ~1 h 55 m is an UPPER BOUND on the branch arms. Every wall above was"
say "  measured on the phase-8 binary, which is precisely the reference arm"
say "  here — so if 70cf304 transfers cold, it is the BRANCH arms that come"
say "  in under their estimate and the reference arms that do not."
say ""
say "  Add up to 15 min per step if MemAvailable will not settle. Seven"
say "  settling steps means the pathological ceiling is another ~1 h 45 m of"
say "  waiting; the build does not settle and does not count."
say ""

# ---------------------------------------------------------------- preflight --
say "--- preflight ---"
FATAL=0

# A live ramvamp is the one failure mode that produces numbers rather than an
# error: cold_bench.py evicts with posix_fadvise(POSIX_FADV_DONTNEED), which
# returns 0 while evicting nothing if another process holds the file mmap'd.
# The run then looks clean and is warm.
if pgrep -x ramvamp > /dev/null 2>&1; then
    say "FATAL: a ramvamp process is running."
    say "  fadvise cannot evict a file another process holds mmap'd, and it"
    say "  returns 0 rather than failing, so every run below would report a"
    say "  clean cold measurement of a warm cache. Kill it and restart."
    FATAL=1
fi

# $RAMVAMP is deliberately NOT in this list: step 0 builds it, and it is
# checked, along with its freshness, immediately after the build.
[ -e "$RVMP" ] || { say "MISSING: $RVMP"; FATAL=1; }

# The reference arm is checked here, by hash, and never by mtime.
verify_ref_binary

# The harness must be able to carry every flag this sweep passes it. Checked
# up front rather than an hour in.
#
# --cache-bytes is off this list because no arm passes it any more: the slot
# dial is held at the runtime's default. io_probe is not checked at all
# because this sweep does not run it. The rule the list encodes is "every
# flag this sweep passes must exist", not "every flag that ever existed"; a
# flag the sweep no longer passes has no business being required, and a
# preflight that fails on one would abort a sweep for a reason that cannot
# affect a single number in it.
require_flags "every cold arm below" "$ROOT/scripts/cold_bench.py" \
    --ramvamp --rvmp --prompt-file --max-new --warmup --repeats \
    --workdir --json

# The slot assertion is what separates a measurement from an unverified dial,
# so it is tested before anything depends on it. See selftest_check_slots.
say "  slot assertion self-test ($PYCHECK):"
selftest_check_slots

for ctx in "${CTX_RUNGS[@]}"; do
    p=$(prompt_for "$ctx")
    if [ -e "$p" ]; then
        say "  ctx $ctx  $(sha256sum "$p" | cut -c1-16)  $(stat -c%s "$p") bytes  $p"
    else
        say "MISSING: ctx $ctx prompt $p"
        if [ "$ctx" = 64 ] || [ "$ctx" = 1024 ] || [ "$ctx" = 2048 ]; then
            say "  generate it: python3 scripts/make_ctx_prompt.py --targets $ctx"
        fi
        FATAL=1
    fi
done

if [ "$FATAL" -ne 0 ]; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
        say ""
        say "DRY_RUN: the above would have aborted a real run before any work."
    else
        say ""
        say "aborting before doing any work."
        exit 2
    fi
fi

# -------------------------------------------------- step 0: build the thing --
# The measured binary is built here, from this tree, immediately before it is
# measured. See "what this script builds" at the top for why this is not
# optional and why the assertion after it is not optional either.
say ""
say "=============================================================="
say "step 0 — build the binary this sweep will measure"
say "=============================================================="

if [ "${SKIP_BUILD:-0}" = "1" ]; then
    skip "build release binary" "SKIP_BUILD=1; the freshness assertion still decides"
else
    # --no-settle: compiling does not need a quiet machine. run()'s drain is
    # the 60-second heat-shed before the first timed step.
    run --no-settle "build release binary" "00-build.log" cargo build --release
fi

say ""
say "--- the binary under measurement ---"
if assert_binary_fresh; then
    say "binary sha256: $(sha256sum "$RAMVAMP" | cut -d' ' -f1)"
    printf '%s\t%s\t%s\t%s\n' "binary freshness assertion" "0" "0" "fresh" \
        >> "$OUT/exitcodes.tsv"
else
    printf '%s\t%s\t%s\t%s\n' "binary freshness assertion" "1" "0" \
        "STALE BINARY" >> "$OUT/exitcodes.tsv"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        say ""
        say "DRY_RUN: step 0 was skipped, so this compared whatever binary was"
        say "  already in target/. A real run builds first, which normally"
        say "  makes this pass. Reaching this line in a real run is fatal."
    else
        say ""
        say "aborting: the binary is not this tree's, so no number it produces"
        say "could honestly be attributed to this commit."
        exit 2
    fi
fi

say ""
say "load average : $(cut -d' ' -f1-3 /proc/loadavg)"
say "kernel       : $(uname -r)"
say "MemAvailable : $(avail_mib) MiB, swap in use: $(swap_mib) MiB"
say ""
say "If this sits waiting to settle, close what you can spare (a browser and"
say "Slack are usually most of it). CPU idle is not what matters here; free"
say "memory is."
say ""

# ------------------------------------ step group 1: the decode split curve --
# One cold_bench invocation per rung, plus a second one at 512 and 3961 on
# the phase-8 reference binary. --warmup 1 discards the run that pays btrfs
# extent metadata warm-up; --repeats 3 gives a median and a spread.
#
# No --cache-bytes: these are the baseline arms and inherit ramvamp's own
# 1440M default, so the curve is measured on the dial the runtime ships with.
# The check below confirms that really did buy 11 slots/layer — for the
# reference arm too, which is also how this finds out if main's default
# budget differs from the branch's.
#
# The reference arm runs IMMEDIATELY after the branch arm for the same rung,
# with the same prompt file, --max-new, warmup and repeats. Nothing separates
# them but their own settle and drain: the slot checks for both are deferred
# until after the pair, so that not even a JSON read sits between the two
# measurements. Per docs/benchmark-machine.md, per-file bandwidth variance
# exceeds run-to-run variance, so a control that reads different files is not
# a control.
#
# --workdir is per step AND per sweep. cold_bench.py writes the child's
# stderr — which is where the `decode split (forward_token)` block lives — to
# <workdir>/runNN.json.stderr with NN restarting at 0 every invocation. With
# the default shared workdir, step 2 overwrites step 1's sidecars; without
# the stamp, this sweep overwrites the last one's. Phase 7 lost every phase
# split but one exactly the first way.
say "=============================================================="
say "step group 1 — decode phase split against context length"
say "=============================================================="
say ""
say "The 512 and 3961 rungs run twice: once on this tree's binary and once on"
say "scratch/phase8-ref/ramvamp (8e1eee8, phase 8 complete), back to back in"
say "this session. 512 is the primary target — where the fused fan-out was"
say "measured warm at 1.264x, and the tightest rung on EXP-023's ladder."
say "3961 is where GEMV has the least room: EXP-023 measured attention at"
say "33.5% of decode there against 5.9% at 512, and the fan-out cannot touch"
say "attention. Between them the two pairs bound the change from both ends."

for ctx in "${CTX_RUNGS[@]}"; do
    p=$(prompt_for "$ctx")
    if [ ! -e "$p" ] && [ "${DRY_RUN:-0}" != "1" ]; then
        skip "cold decode ctx $ctx" "prompt $p not found"
        continue
    fi
    cb_json="$CB_JSON_DIR/p9-$STAMP-decode-$ctx.json"
    run "cold decode ctx $ctx, fused fan-out branch, --max-new 64" \
        "10-decode-$ctx.log" \
        python3 scripts/cold_bench.py \
            --ramvamp "$RAMVAMP" --rvmp "$RVMP" --prompt-file "$p" \
            --max-new 64 --warmup 1 --repeats 3 \
            --workdir "$CB_WORK_DIR/ctx$ctx" \
            --json "$cb_json"
    branch_t0=$STEP_T0

    ref_json=""
    ref_t0=$SWEEP_T0
    if is_paired_rung "$ctx"; then
        ref_json="$CB_JSON_DIR/p9-$STAMP-phase8ref-$ctx.json"
        run "cold decode ctx $ctx, phase-8 reference (8e1eee8), --max-new 64" \
            "11-phase8ref-$ctx.log" \
            python3 scripts/cold_bench.py \
                --ramvamp "$REF8" --rvmp "$RVMP" --prompt-file "$p" \
                --max-new 64 --warmup 1 --repeats 3 \
                --workdir "$CB_WORK_DIR/phase8ref-ctx$ctx" \
                --json "$ref_json"
        ref_t0=$STEP_T0
    fi

    check_slots "slots ctx $ctx (want $SLOTS_DEFAULT)" \
        "$cb_json" "$SLOTS_DEFAULT" "$branch_t0"
    if [ -n "$ref_json" ]; then
        check_slots "slots ctx $ctx phase-8 ref (want $SLOTS_DEFAULT)" \
            "$ref_json" "$SLOTS_DEFAULT" "$ref_t0"
    fi
done

# ------------------------------- what phase 8 ran here, and phase 9 does not --
#
# Phase 8 followed group 1 with a slot-dial group (1570M/12 slots at 512 and
# 3961, 1701M/13 slots at 512) and an io_probe group (the single-blob
# queue-depth curve). Both are gone, and neither is a candidate for quietly
# coming back:
#
#   The dial is HELD this phase. Phase 9 is pricing one change, and the whole
#   value of the paired arms below is that only one thing differs between
#   them. Re-measuring the dial would cost ~46 minutes to re-derive a number
#   that must not be acted on until the fan-out question is answered. The
#   slot ASSERTION stays on every arm, which is the part that matters: it is
#   what turns "the dial was held" into something the artifact proves rather
#   than asserts.
#
#   The io_probe curve is FINAL. Phase 9 ran the drive-side probes
#   separately and its result stands. Re-running them here would put a second
#   answer to a settled question inside an artifact about a different one,
#   and the two would eventually be quoted against each other.
#
# Their preflight checks went with them: require_flags no longer asks
# cold_bench.py for --cache-bytes and no longer inspects io_probe.py at all,
# because this sweep passes neither.

# ------------------------------------------------------------------ summary --
say ""
say "=============================================================="
say "finished $(stamp)"
say ""
say "step results (exit 0 is good):"
say ""
while IFS=$'\t' read -r label rc secs hyg; do
    case "$secs" in
        ''|*[!0-9]*) secs=0 ;;
    esac
    printf '  %-60s exit=%-4s %dm%02ds  %s\n' "$label" "$rc" $((secs/60)) $((secs%60)) "$hyg"
done < "$OUT/exitcodes.tsv" | tee -a "$SUMMARY"

if [ "${DRY_RUN:-0}" = "1" ]; then
    say ""
    say "--- no numbers: DRY_RUN ---"
    say ""
    say "Nothing was measured, so there is nothing to report. The sections"
    say "that print throughput, hygiene verdicts, the paired table and the"
    say "phase splits are skipped entirely rather than run against whatever"
    say "files an earlier sweep left behind: a real sweep's numbers printed"
    say "under a heading that says DRY_RUN is worse than no numbers at all."
    say ""
    say "The arms this sweep would have measured, and where each would have"
    say "written its summary:"
    say ""
    while IFS=$'\t' read -r label json want t0; do
        say "  want $want slots  $label"
        say "      $json"
    done < "$MANIFEST"
    say ""
    say "scaffolding checked. $OUT"
    exit 0
fi

say ""
say "--- the fused decode fan-out vs the phase-8 baseline, paired ---"
say ""
say "The two arms of each rung ran back to back on the same prompt with the"
say "same dials, in this session, so this ratio is 70cf304 and nothing else."
say "It is the whole reason the reference arm exists: without it the only"
say "baseline available is EXP-023's, from another session, which rule 3"
say "forbids putting on one curve with these numbers. A rung is printed only"
say "if BOTH its arms produced a summary of their own."
say ""
python3 - "$MANIFEST" "$CB_JSON_DIR" "$STAMP" "$SWEEP_T0" "${PAIRED_RUNGS[@]}" \
    <<'PYEOF' 2>&1 | tee -a "$SUMMARY"
import json
import os
import sys
import time

manifest, jsondir, stamp, sweep_t0 = sys.argv[1:5]
rungs = sys.argv[5:]
sweep_t0 = float(sweep_t0)

# Floor each path at its own step's start, from the manifest. Falling back to
# the sweep start rather than to 0: an unlisted path is still not allowed to
# be older than the sweep that claims it.
floors = {}
with open(manifest, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 4:
            floors[os.path.abspath(parts[1])] = float(parts[3])


def when(epoch):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def load(path):
    """Return (median dict, None) or (None, why-not)."""
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        return None, "no summary (the arm produced nothing)"
    mtime = os.path.getmtime(ap)
    floor = floors.get(ap, sweep_t0)
    if mtime < floor:
        return None, (f"stale summary, not read (mtime {when(mtime)} "
                      f"predates {when(floor)})")
    try:
        with open(ap, encoding="utf-8") as f:
            return json.load(f).get("median", {}) or {}, None
    except (OSError, ValueError) as exc:
        return None, f"unreadable: {exc}"


def ratio(new, old):
    try:
        if old and new and float(old) > 0:
            return f"{float(new) / float(old):.3f}x"
    except (TypeError, ValueError):
        pass
    return "n/a"


print(f"  {'rung':>6}  {'metric':<14} {'phase-8 ref':>12} {'branch':>12} "
      f"{'new/ref':>8}")
for rung in rungs:
    ref, ref_why = load(
        os.path.join(jsondir, f"p9-{stamp}-phase8ref-{rung}.json"))
    new, new_why = load(os.path.join(jsondir, f"p9-{stamp}-decode-{rung}.json"))
    if ref is None or new is None:
        print(f"  {rung:>6}  NOT COMPARABLE")
        if ref is None:
            print(f"          phase-8 reference: {ref_why}")
        if new is None:
            print(f"          branch           : {new_why}")
        print(f"          The fused fan-out stays unpriced at this rung: the "
              f"only baseline left for it is another session's.")
        continue
    for key in ("decode_tok_s", "prefill_tok_s", "wall_s", "load_s"):
        a, b = ref.get(key), new.get(key)
        print(f"  {rung:>6}  {key:<14} {str(a):>12} {str(b):>12} "
              f"{ratio(b, a):>8}")
print()
print("  decode_tok_s and prefill_tok_s: higher is better, so new/ref above")
print("  1.000 means the branch won. wall_s and load_s: lower is better, so")
print("  the same ratio above 1.000 means it lost. One session, one machine:")
print("  this is a paired A/B, not a published speedup, until it has an")
print("  experiments entry with a baseline, a result and a verdict.")
print()
print("  The warm prediction under test is 1.264x on the POOLED GEMV BUCKETS")
print("  at ctx 512 (14.44s -> 11.42s, three runs, diagnostic only).")
print("  decode_tok_s is a whole-token figure and GEMV is one part of a")
print("  token, so 1.264x is not what decode_tok_s is expected to show. The")
print("  gemv split block below is where the two quantities can be compared.")
PYEOF

say ""
say "--- headline throughput and hygiene verdicts ---"
for f in "$OUT"/*.log; do
    [ -e "$f" ] || continue
    printf '### %s\n' "$(basename "$f")" | tee -a "$SUMMARY"
    grep -a -E 'median (wall_s|load_s|prefill_tok_s|decode_tok_s|prefill_s|decode_s|read_bytes|MemoryPeak)|measurement hygiene|memory\.peak|pgsteal [0-9]' \
        "$f" 2>/dev/null | tee -a "$SUMMARY"
    printf '\n' | tee -a "$SUMMARY"
done

say ""
say "--- decode phase split, per step ---"
say ""
say "Pulled from the summary JSON, not from the run logs. cold_bench.py does"
say "not echo the child's stderr to its own stdout, so 'decode split' never"
say "appears in a step log; it lives in runs[].stderr of the JSON and in the"
say "<workdir>/runNN.json.stderr sidecars."
say ""
say "Driven by $MANIFEST — the arms this sweep actually launched — not by a"
say "glob. A glob prints whatever is in scratch/cold-bench, which after a"
say "failed arm is the previous sweep's answer to the same question."
say ""
python3 - "$MANIFEST" "$ROOT" "$SWEEP_T0" <<'PYEOF' 2>&1 | tee -a "$SUMMARY"
import glob
import json
import os
import re
import sys
import time

manifest, root, sweep_t0 = sys.argv[1], sys.argv[2], float(sys.argv[3])
SLOTS_RE = re.compile(r"(\d+)\s+expert slots/layer from a ([^,]+) budget")


def when(epoch):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def indent_of(line):
    return len(line) - len(line.lstrip(" \t"))


def split_blocks(lines):
    """Every block of the child's stderr worth quoting, each one WHOLE.

    A block is a line that mentions "split" or starts with "experts:", plus
    every following line indented deeper than it. The extent comes from the
    INDENTATION, never from a line count.

    Phase 8 found the first such line, printed lines[i:i + 12] and broke.
    Two things were wrong with that and both are why this exists. Because
    `experts:` matches before `decode split (forward_token):`, the twelve
    lines were spent on the experts block and the two split blocks, so the
    decode split was cut off after two of its six rows -- visible in
    scratch/phase8/sweep-20260806-165322/SUMMARY.txt, where the 3,961-token
    arm ends at `attention:` and its expert-compute, expert-io, projections,
    elementwise and other rows are simply not in the artifact EXP-023 was
    written from. And it broke after one block, so phase 9's second block,
    `decode gemv split (submitting thread):`, would never have printed at
    all.

    Counting to a bigger number would fix today's stderr and break on the
    next block anyone adds. The block boundary is already in the text.
    """
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.strip() and ("split" in line or line.startswith("experts:")):
            base = indent_of(line)
            block = [line]
            j = i + 1
            while j < n and lines[j].strip() and indent_of(lines[j]) > base:
                block.append(lines[j])
                j += 1
            blocks.append(block)
            i = j
        else:
            i += 1
    return blocks


rows = []
with open(manifest, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 4:
            rows.append((parts[0], parts[1], parts[2], float(parts[3])))

if not rows:
    print("  no arms were launched")

read = set()
for label, path, want, step_t0 in rows:
    print(f"### {os.path.basename(path)}")
    print(f"  arm: {label}")
    if not os.path.isfile(path):
        print(f"  NO SUMMARY. This arm wrote nothing, so it contributed no")
        print(f"  numbers to this sweep. cold_bench.py writes --json last,")
        print(f"  after every failure path has exited, so an absent summary")
        print(f"  means the arm failed -- check its exit code above and its")
        print(f"  log. Nothing is printed here in its place.\n")
        continue
    mtime = os.path.getmtime(path)
    if mtime < step_t0:
        print(f"  STALE SUMMARY, NOT READ. mtime {when(mtime)} predates the")
        print(f"  step that should have written it ({when(step_t0)}), so it")
        print(f"  belongs to an earlier sweep. This arm produced nothing.\n")
        continue
    read.add(os.path.abspath(path))
    try:
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"  unreadable: {exc}\n")
        continue
    med = summary.get("median", {})
    meta = summary.get("cache_bytes") or {}
    print(f"  hygiene {summary.get('hygiene')}  "
          f"decode {med.get('decode_tok_s')} tok/s  "
          f"prefill {med.get('prefill_tok_s')} tok/s  "
          f"wall {med.get('wall_s')} s")
    prompt = summary.get("prompt", {})
    print(f"  prompt {prompt.get('file')} "
          f"{prompt.get('bytes')} bytes sha256:{str(prompt.get('sha256'))[:16]}")
    scored = [(i, r) for i, r in enumerate(summary.get("runs", []))
              if r.get("label") == "scored" and r.get("stderr")]
    slots = ""
    if scored:
        match = SLOTS_RE.search(scored[-1][1]["stderr"])
        if match:
            slots = f"{match.group(1)} slots/layer from a {match.group(2)} budget"
    print(f"  budget {meta.get('value')} ({meta.get('source')})  "
          f"want {want} slots/layer  {slots}")
    if not scored:
        print("  no child stderr recorded in this summary — check the "
              "runNN.json.stderr sidecars under the step's --workdir\n")
        continue
    # WHICH run this came from, stated rather than implied. Phase 8 rendered
    # the LAST scored run and said so nowhere in the artifact; EXP-023 quoted
    # the FIRST. Two different runs behind one set of published splits, with
    # nothing written down to tell them apart. The choice here is unchanged --
    # the last scored run is the one least contaminated by warm-up -- but it
    # is now named, with its runs[] index, so a figure quoted out of this file
    # can be traced back to the run that produced it.
    idx, chosen = scored[-1]
    nruns = len(summary.get("runs", []))
    print(f"  splits below: SCORED RUN {len(scored)} of {len(scored)} "
          f"(runs[{idx}] of {nruns}, the LAST scored run). The other scored "
          f"runs' splits are in runs[].stderr of this JSON and in the "
          f"<workdir>/runNN.json.stderr sidecars.")
    blocks = split_blocks((chosen.get("stderr") or "").splitlines())
    if not blocks:
        print("  no split block in the recorded stderr\n")
        continue
    for block in blocks:
        for out in block:
            print(f"  {out}")
    print()

others = [p for p in sorted(glob.glob(
    os.path.join(root, "scratch/cold-bench/p9-*.json")))
    if os.path.abspath(p) not in read]
if others:
    print(f"  {len(others)} other p9-*.json in scratch/cold-bench belong to")
    print(f"  earlier sweeps and were NOT read: "
          f"{', '.join(os.path.basename(p) for p in others[:6])}"
          f"{' ...' if len(others) > 6 else ''}")
PYEOF

say ""
say "this sweep's summaries: scratch/cold-bench/p9-$STAMP-*.json"
say "stderr sidecars: $CB_WORK_DIR/*/runNN.json.stderr"
say "arm manifest: $MANIFEST"
say "full logs: $OUT"
say ""
say 'Any "SLOT COUNT WRONG" row above invalidates that step and only that'
say "step: the budget bought a different dial than the label claims. Recompute"
say 'from the layer stride on the "model loaded in" line and re-run it.'
say ""
say 'Any "NO SUMMARY" row means that step produced no measurement at all.'
say "Nothing was substituted for it. Re-run that arm."
say ""
say "The paired table is what CLAUDE.md's rule needs to be discharged: the"
say "fused decode fan-out (70cf304) is a performance change and owes"
say "docs/experiments a baseline, a result and a verdict. The baseline is"
say "$REF8_EXPECT_SHA"
say "(8e1eee8's build, byte-identical to EXP-023's branch binary), the result"
say "is the paired table, and the verdict is yours. The 1.264x measured warm"
say "is a diagnostic and is not the result; this is."
say ""
say "Rule 3: these are one session on one machine. They may be combined with"
say "each other into one curve and NOT with EXP-014's, EXP-018's, EXP-019's,"
say "EXP-021's or EXP-023's numbers. The estimated wall times printed at the"
say "top of this run are derived from EXP-023 and are scheduling aids, not"
say "data. EXP-023's decode ladder is quoted in this script's header for the"
say "same reason: it is why these rungs, not a baseline to subtract."
say ""
say "The one exception, and it is narrow: the phase-8 reference arm is the"
say "same bytes EXP-023 measured, so a disagreement between its numbers here"
say "and EXP-023's is a statement about the two SESSIONS, not about the"
say "change. That makes it a useful check on the machine. It is still not a"
say "licence to put EXP-023's figures on the same curve as this sweep's."
