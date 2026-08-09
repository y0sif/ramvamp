#!/usr/bin/env bash
# Phase 8 decode sweep: the cold, rule-2 decode phase-split curve against
# context length, the expert-cache slot dial, and the drive-side single-blob
# queue-depth curve.
#
#     nohup bash scripts/phase8_decode_sweep.sh > /dev/null 2>&1 &
#     DRY_RUN=1 bash scripts/phase8_decode_sweep.sh     # check the scaffolding
#
# Then read scratch/phase8/sweep-<stamp>/SUMMARY.txt.
#
# Adapted from scripts/phase7_rerun_cold.sh. Everything that script learned
# the hard way is kept: the settle loop that waits for MemAvailable to hold
# above 6,000 MiB across four consecutive 15-second samples before any run
# starts, per-step exit codes captured into exitcodes.tsv rather than
# aborting on the first failure, a 60-second drain after each step so the
# page cache one run built does not contaminate the next, and a SUMMARY.txt
# tail.
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
# feat/decode changes T_BLOCK from 4 to 8 — four QK dependency chains to
# eight. CLAUDE.md's hard rule is that every performance change gets an entry
# in docs/experiments.md with a baseline, a result and a verdict, and
# T_BLOCK has none. Left alone, phase 8 would measure its decode curve on a
# binary carrying that change and silently attribute T_BLOCK's cost to
# context length: the one confound this whole sweep exists to avoid.
#
# So two rungs, 512 and 3961, run twice — once on the branch binary and once
# on scratch/phase7-ref/ramvamp, built from d329890 (main). That pair IS the
# T_BLOCK A/B, because T_BLOCK is the only runtime change on feat/decode;
# everything else on the branch is scripts plus a test-only fixture refactor.
# Two rungs, not five: attributing T_BLOCK does not need the whole curve, and
# 3961 is the expensive end where a longer chain has the most room to matter.
#
# The reference binary's sha256 is
# d56dc034ebd3e94e83b59ad64503adf289baf22af112752593ec59068a586e66, which is
# byte-identical to the binary EXP-021 measured (recorded there as
# `d56dc034ebd3...` at `aade585`; see EXP-021 in docs/experiments.md). The
# reference arm is not a lookalike rebuild of phase 7; it is the same bytes
# phase 7 published from.
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
# `--cache-bytes` is a byte budget, not a slot count, and the runtime derives
# slots/layer from it. A slot costs the sum of every layer's page-aligned
# stride — 130.781 MiB for the shipped Qwen3-30B-A3B layout — so the arms
# below clear their thresholds by well under 1 MiB:
#
#   1440M (ramvamp's default) -> 11 slots/layer  (needs 1438.6 MiB)
#   1570M                     -> 12 slots/layer  (needs 1569.4 MiB, +0.6)
#   1701M                     -> 13 slots/layer  (needs 1700.2 MiB, +0.8)
#
# A repack that changes the per-layer stride moves those thresholds, and the
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
# normal workflow, and the 1570M/12-slot arm at full context is expected to
# sit within tens of MiB of the 3,072 MiB cap in either direction (see step
# group 2), so an OOM-killed arm is a likely outcome rather than a surprise.
# With a fixed output path, the failed arm's slot check would read the
# previous sweep's summary, find the right slot count, and record `slots=12
# OK`; the tail would then reprint that summary's hygiene verdict and tok/s
# into THIS sweep's SUMMARY.txt with nothing marking where they came from.
#
# Two changes, and both are load-bearing:
#
#   1. Every artifact this sweep writes carries $STAMP, exactly as $OUT
#      already did — the summaries, the io_probe json and markdown, and the
#      per-step cold_bench workdirs. Sweeps can no longer collide, and no
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
OUT="$ROOT/scratch/phase8/sweep-$STAMP"
mkdir -p "$OUT" "$ROOT/scratch/cold-bench" "$ROOT/scratch/io-probe" || exit 1
SUMMARY="$OUT/SUMMARY.txt"
MANIFEST="$OUT/summaries.tsv"
: > "$SUMMARY"
: > "$OUT/exitcodes.tsv"
: > "$MANIFEST"

RAMVAMP="$ROOT/target/release/ramvamp"
RVMP="$ROOT/models/qwen3.rvmp"

# The T_BLOCK=4 reference arm, built from main (d329890) and banked beside
# its own COMMIT and SHA256, following the scratch/phase5-ref/ convention
# phase7_overnight.sh already uses for REF5.
REF7="$ROOT/scratch/phase7-ref/ramvamp"
REF7_SHA_FILE="$ROOT/scratch/phase7-ref/SHA256"
REF7_COMMIT_FILE="$ROOT/scratch/phase7-ref/COMMIT"

# The bytes EXP-021 measured. Pinned here as a constant and not merely read
# from the SHA256 file beside the binary: if the reference is ever re-banked,
# a SHA256 regenerated alongside it agrees with itself and is still not
# phase 7's binary. Both must match.
REF7_EXPECT_SHA=d56dc034ebd3e94e83b59ad64503adf289baf22af112752593ec59068a586e66

# Per-sweep output paths. See "why every output path is per-sweep" above.
CB_JSON_DIR="$ROOT/scratch/cold-bench"
CB_WORK_DIR="scratch/phase8/cold-bench/$STAMP"
IO_JSON="$ROOT/scratch/io-probe/p8-$STAMP-decode-qd.json"
IO_MD="$ROOT/scratch/io-probe/p8-$STAMP-decode-qd.md"

# Set by run(); read by check_slots() as the earliest mtime a summary written
# by that step could possibly have.
STEP_T0=$SWEEP_T0

# Slots/layer each arm must report. 11 is ramvamp's default budget (1440M),
# which every step that does not pass --cache-bytes inherits.
SLOTS_DEFAULT=11

# The context ladder. Every rung key is the prompt's true token count, so a
# label, a filename and a row can never disagree with the workload. The 512
# and 3961 rungs are the fixtures phase 7 already measured on, reused
# unchanged so the two entries share a workload where they overlap; 64, 1024
# and 2048 are cut by scripts/make_ctx_prompt.py from
# models/llamacpp-ref/llamacpp_ref/long_02.txt.
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

# Resolves a rung to its prompt file. Every path here is under the gitignored
# /models/ or /scratch/ trees described above, so none of them exists in a
# fresh clone; the preflight is what turns a missing one into a clear failure
# rather than a silently short run.
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

# The rungs that also run a T_BLOCK=4 reference arm. See "why there is a
# reference arm" above. 512 is where phase 7 has the most prior art; 3961 is
# the expensive end.
TBLOCK_RUNGS=(512 3961)

is_tblock_rung() {
    local want=$1 r
    for r in "${TBLOCK_RUNGS[@]}"; do
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
        run_slot_check "$json" "$want" "$minm" "$dir/$total.out"
        rc=$?
        if [ "$rc" -eq "$exp" ]; then
            say "  self-test OK   exit=$rc  $name"
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
# T_BLOCK=4 reference, which is deliberately older than the tracked sources —
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
# What stands in for the freshness assertion on the T_BLOCK=4 reference arm.
# The reference is meant to be old — it is main's build — so an mtime says
# nothing useful about it. Its identity is its bytes, and both the SHA256
# banked beside it and the constant pinned at the top of this script have to
# agree with them. FATAL on any disagreement: a reference arm that is not the
# binary it claims to be turns the T_BLOCK A/B into an unlabelled comparison
# of two unknowns.
verify_ref_binary() {
    local recorded computed commit
    if [ ! -x "$REF7" ]; then
        say "MISSING: $REF7"
        say "  This is the T_BLOCK=4 reference arm's binary, built from main"
        say "  (d329890). Without it the sweep can measure the decode curve"
        say "  but cannot attribute any of it to T_BLOCK, which is the one"
        say "  runtime change on this branch and has no experiments entry."
        FATAL=1
        return 1
    fi
    if [ ! -e "$REF7_SHA_FILE" ]; then
        say "MISSING: $REF7_SHA_FILE"
        say "  The reference binary is identified by its hash, so the banked"
        say "  hash is not optional bookkeeping."
        FATAL=1
        return 1
    fi
    recorded=$(awk 'NR == 1 { print $1 }' "$REF7_SHA_FILE")
    computed=$(sha256sum "$REF7" | cut -d' ' -f1)
    if [ -z "$recorded" ] || [ -z "$computed" ]; then
        say "FATAL: could not read or compute the reference binary's sha256."
        FATAL=1
        return 1
    fi
    if [ "$computed" != "$recorded" ]; then
        say "FATAL: $REF7 does not match its banked SHA256."
        say "  banked   $recorded"
        say "  computed $computed"
        say "  The file beside the binary describes a different binary. Do not"
        say "  guess which one is right; re-bank the reference from main."
        FATAL=1
        return 1
    fi
    if [ "$computed" != "$REF7_EXPECT_SHA" ]; then
        say "FATAL: $REF7 is not the binary EXP-021 measured."
        say "  expected $REF7_EXPECT_SHA"
        say "  computed $computed"
        say "  Its own SHA256 file agrees with it, which is exactly the case"
        say "  that a self-consistent re-bank produces. The reference arm is"
        say "  supposed to be phase 7's published bytes, not a rebuild of the"
        say "  same source, so this is a stop."
        FATAL=1
        return 1
    fi
    commit=$(head -1 "$REF7_COMMIT_FILE" 2>/dev/null)
    say "  reference binary: sha256 matches its banked SHA256 and EXP-021's"
    say "    $computed"
    say "    commit: ${commit:-<no COMMIT file>}"
    say "    exempt from the freshness assertion by design; it is main's build"
    say "    and is meant to be older than the tracked sources."
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

say "ramvamp phase 8 — cold decode sweep against context length"
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
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    say "        WORKING TREE IS DIRTY. The commit above is not sufficient"
    say "        provenance for the branch arm: its binary contains changes"
    say "        that are in no commit. Diff sha256 over \`git diff HEAD\`:"
    say "          $(git diff HEAD 2>/dev/null | sha256sum | cut -d' ' -f1)"
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
say "Derived from EXP-021's measured phase-7 walls on this machine:"
say "  measured  model load                1.30 s"
say "  measured  prefill   512 tokens     45.84 s  (11.17 tok/s)"
say "  measured  prefill 3,961 tokens    401.53 s  ( 9.86 tok/s)"
say "  measured  decode at   512 ctx     0.503 s/token (256-token run)"
say "  measured  decode at 3,961 ctx     0.680 s/token (8-token run, so this"
say "            one is transient-dominated and is an upper bound)"
say "  measured  harness overhead per run ~1.5 s (phase-7 step walls in"
say "            scratch/phase7/*/exitcodes.tsv minus the child walls)"
say ""
say "Prefill is interpolated linearly between the two measured points"
say "(0.1028 s/token, intercept -6.8 s). At 64 tokens that fit goes negative,"
say "so the 64 rung uses a 10 s floor instead, taken from EXP-021's 20.72 GB"
say "of reads at EXP-019's 1.54-2.37 GB/s. Decode s/token is interpolated"
say "linearly in context between the two measured points."
say ""
say "  step (1 warmup + 3 scored = 4 runs)      est/run    est step"
say "  ctx   64  --max-new 64                      45 s       3 min"
say "  ctx  512  --max-new 64                      81 s       6 min"
say "  ctx 1024  --max-new 64                     133 s       9 min"
say "  ctx 2048  --max-new 64                     244 s      17 min"
say "  ctx 3961  --max-new 64                     447 s      30 min"
say "  ctx  512  T_BLOCK=4 reference               81 s       6 min"
say "  ctx 3961  T_BLOCK=4 reference              447 s      30 min"
say "  settle (>= 45 s) + drain (60 s) x 7 steps              13 min"
say "                              step group 1 total       ~114 min"
say "  group 2, 3 slot-dial arms (512+3961 at 1570M,"
say "    512 at 1701M). Upper bound: more slots means"
say "    fewer expert reads, so these run no slower"
say "    than the 11-slot walls they are derived from   ~46 min"
say "  group 3, io_probe (its own --dry-run says 80 s at"
say "    an assumed 1.40 GB/s; the QD=1 cells will run"
say "    well under that, so budget more)                ~5 min"
say "  slot-count checks (10 x a few hundred ms)         <1 min"
say "  step 0, cargo build --release + 60 s heat-shed     ~3 min"
say ""
say "  TOTAL  ~2 h 49 m   ESTIMATED"
say ""
say "  The two T_BLOCK=4 reference arms are ~40 min of that: 36 min of runs"
say "  plus their settle and drain. They are estimated at the same cost as"
say "  their branch counterparts because the estimates come from EXP-021,"
say "  which measured that very binary — so if T_BLOCK=8 changed anything,"
say "  it is the BRANCH arms whose estimate is wrong, not the reference's."
say ""
say "  Add up to 15 min per step if MemAvailable will not settle. Twelve"
say "  timed steps means the pathological ceiling is another ~3 h of waiting."
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

# Both harnesses must be able to carry every flag this sweep passes them.
# Checked up front rather than 110 minutes in, when group 2 starts.
require_flags "step group 2's slot dial" "$ROOT/scripts/cold_bench.py" \
    --ramvamp --rvmp --prompt-file --max-new --warmup --repeats \
    --workdir --json --cache-bytes
require_flags "step group 3's queue-depth curve" "$ROOT/scripts/io_probe.py" \
    --block-ks --fixed-k --fixed-qd --queue-depths --patterns --repeats \
    --json --markdown --dry-run

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
# the T_BLOCK=4 reference binary. --warmup 1 discards the run that pays btrfs
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
say "The 512 and 3961 rungs run twice: once on this branch's binary and once"
say "on scratch/phase7-ref/ramvamp (main, T_BLOCK=4), back to back. T_BLOCK"
say "4 -> 8 is the only runtime change on feat/decode and has no experiments"
say "entry, so without that pair its cost would be attributed to context"
say "length by every other number in this sweep."

for ctx in "${CTX_RUNGS[@]}"; do
    p=$(prompt_for "$ctx")
    if [ ! -e "$p" ] && [ "${DRY_RUN:-0}" != "1" ]; then
        skip "cold decode ctx $ctx" "prompt $p not found"
        continue
    fi
    cb_json="$CB_JSON_DIR/p8-$STAMP-decode-$ctx.json"
    run "cold decode ctx $ctx, T_BLOCK=8 branch, --max-new 64" \
        "10-decode-$ctx.log" \
        python3 scripts/cold_bench.py \
            --ramvamp "$RAMVAMP" --rvmp "$RVMP" --prompt-file "$p" \
            --max-new 64 --warmup 1 --repeats 3 \
            --workdir "$CB_WORK_DIR/ctx$ctx" \
            --json "$cb_json"
    branch_t0=$STEP_T0

    ref_json=""
    ref_t0=$SWEEP_T0
    if is_tblock_rung "$ctx"; then
        ref_json="$CB_JSON_DIR/p8-$STAMP-tblock4-$ctx.json"
        run "cold decode ctx $ctx, T_BLOCK=4 reference (main), --max-new 64" \
            "11-tblock4-$ctx.log" \
            python3 scripts/cold_bench.py \
                --ramvamp "$REF7" --rvmp "$RVMP" --prompt-file "$p" \
                --max-new 64 --warmup 1 --repeats 3 \
                --workdir "$CB_WORK_DIR/tblock4-ctx$ctx" \
                --json "$ref_json"
        ref_t0=$STEP_T0
    fi

    check_slots "slots ctx $ctx (want $SLOTS_DEFAULT)" \
        "$cb_json" "$SLOTS_DEFAULT" "$branch_t0"
    if [ -n "$ref_json" ]; then
        check_slots "slots ctx $ctx T_BLOCK=4 ref (want $SLOTS_DEFAULT)" \
            "$ref_json" "$SLOTS_DEFAULT" "$ref_t0"
    fi
done

# ---------------------------------------------- step group 2: the slot dial --
# The arms, and what each one is for:
#
#   1570M (12 slots) at ctx  512   — does one more slot pay at short context
#   1570M (12 slots) at ctx 3961   — and does it still fit at full context
#   1701M (13 slots) at ctx  512   — where the dial stops paying
#
# WARNING on the 3,961-token arm at 1570M. Two arithmetics disagree about
# whether 12 slots/layer fits under the 3,072 MiB cap, and they straddle it:
#
#   docs/architecture.md's static table  3,091.82 MiB   19.8 MiB OVER
#   EXP-021's measured 2,920 MiB at 11
#     slots, plus 130.8 for the twelfth  3,051    MiB   20.8 MiB UNDER
#
# 40 MiB apart on either side of the line, and architecture.md marks its own
# subtotal column provisional (115.1 of its 1,522.44 MiB of fixed tenants is
# an estimate). EXP-018 also carries an unexplained 99-105 MiB residual on
# top of its accounted memory, which is larger than either margin. So this
# arm is genuinely undecided, and running it is how it gets decided.
#
# If the cgroup OOM-kills it, that is the finding, not a harness failure:
# it is the measurement of where the memory contract stops holding.
# cold_bench.py reports it as "the inner run produced no result file
# (systemd-run exited N); MemoryMax may have OOM-killed it" and exits 2
# BEFORE writing its summary — which is exactly why the summary paths carry
# $STAMP and why the slot check refuses a missing or stale file instead of
# reading the previous sweep's.
say ""
say "=============================================================="
say "step group 2 — the expert-cache slot dial"
say "=============================================================="
say ""
say "The 3,961-token/1570M arm is undecided by ~20 MiB in both directions:"
say "architecture.md's static table puts 12 slots/layer 19.8 MiB OVER the"
say "3,072 MiB cap, EXP-021's measured 2,920 MiB at 11 slots plus 130.8 MiB"
say "puts it 20.8 MiB UNDER, and EXP-018's unexplained 99-105 MiB residual"
say "is bigger than either margin. If it is OOM-killed, that is the result."
say "Its summary will then be absent, and the slot check will say so."

for arm in "512:1570M:12" "3961:1570M:12" "512:1701M:13"; do
    ctx=${arm%%:*}
    rest=${arm#*:}
    budget=${rest%%:*}
    slots=${rest##*:}
    p=$(prompt_for "$ctx")
    if [ ! -e "$p" ] && [ "${DRY_RUN:-0}" != "1" ]; then
        skip "slot dial ctx $ctx $budget" "prompt $p not found"
        continue
    fi
    cb_json="$CB_JSON_DIR/p8-$STAMP-slots-$ctx-$budget.json"
    run "slot dial ctx $ctx, $budget (want $slots slots/layer), --max-new 64" \
        "20-slots-$ctx-$budget.log" \
        python3 scripts/cold_bench.py \
            --ramvamp "$RAMVAMP" --rvmp "$RVMP" --prompt-file "$p" \
            --max-new 64 --warmup 1 --repeats 3 \
            --cache-bytes "$budget" \
            --workdir "$CB_WORK_DIR/slots-$ctx-$budget" \
            --json "$cb_json"
    check_slots "slots ctx $ctx $budget (want $slots)" "$cb_json" "$slots"
done

# ------------------------------- step group 3: drive-side single-blob curve --
# K=1 is the point: one expert blob per read is what decode actually issues,
# and EXP-019 found total bytes in flight — not block size — predicts
# throughput, so the queue-depth axis at K=1 is the one that maps onto the
# decode path. Emulated with threads issuing blocking preadv, NOT io_uring:
# do not put these numbers on one curve with the runtime's io_uring path.
say ""
say "=============================================================="
say "step group 3 — drive-side single-blob queue-depth curve"
say "=============================================================="

IO_PROBE_ARGS=(
    --block-ks 1 --fixed-k 1 --fixed-qd 8
    --queue-depths 1,2,4,8,16
    --patterns rand,seq
    --repeats 3
    --json "$IO_JSON"
    --markdown "$IO_MD"
)

# Its own --dry-run prints the full plan and the byte totals without touching
# the drive, so the plan is on the record before the drive is.
run "io_probe plan (--dry-run, touches nothing)" "30-io-probe-dryrun.log" \
    python3 scripts/io_probe.py "${IO_PROBE_ARGS[@]}" --dry-run

run "io_probe single-blob queue-depth curve" "31-io-probe.log" \
    python3 scripts/io_probe.py "${IO_PROBE_ARGS[@]}"
IO_STEP_T0=$STEP_T0

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
    say "that print throughput, hygiene verdicts, phase splits and io_probe"
    say "tables are skipped entirely rather than run against whatever files"
    say "an earlier sweep left behind: a real sweep's numbers printed under a"
    say "heading that says DRY_RUN is worse than no numbers at all."
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
say "--- T_BLOCK 4 vs 8, paired ---"
say ""
say "The two arms of each rung ran back to back on the same prompt with the"
say "same dials, so this ratio is T_BLOCK and nothing else. It is the whole"
say "reason the reference arm exists: without it, any T_BLOCK cost shows up"
say "in the context curve above wearing context's name. A rung is printed"
say "only if BOTH its arms produced a summary of their own."
say ""
python3 - "$MANIFEST" "$CB_JSON_DIR" "$STAMP" "$SWEEP_T0" "${TBLOCK_RUNGS[@]}" \
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


print(f"  {'rung':>6}  {'metric':<14} {'T_BLOCK=4':>12} {'T_BLOCK=8':>12} "
      f"{'8/4':>8}")
for rung in rungs:
    ref, ref_why = load(os.path.join(jsondir, f"p8-{stamp}-tblock4-{rung}.json"))
    new, new_why = load(os.path.join(jsondir, f"p8-{stamp}-decode-{rung}.json"))
    if ref is None or new is None:
        print(f"  {rung:>6}  NOT COMPARABLE")
        if ref is None:
            print(f"          T_BLOCK=4 reference: {ref_why}")
        if new is None:
            print(f"          T_BLOCK=8 branch   : {new_why}")
        print(f"          T_BLOCK stays unattributed at this rung, and the "
              f"decode curve above still carries its cost.")
        continue
    for key in ("decode_tok_s", "prefill_tok_s", "wall_s", "load_s"):
        a, b = ref.get(key), new.get(key)
        print(f"  {rung:>6}  {key:<14} {str(a):>12} {str(b):>12} "
              f"{ratio(b, a):>8}")
print()
print("  decode_tok_s and prefill_tok_s: higher is better, so 8/4 above 1.000")
print("  means T_BLOCK=8 won. wall_s and load_s: lower is better, so the same")
print("  ratio above 1.000 means it lost. One session, one machine: this is a")
print("  paired A/B, not a published speedup, until it has an experiments")
print("  entry with a baseline, a result and a verdict.")
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
    scored = [r for r in summary.get("runs", [])
              if r.get("label") == "scored" and r.get("stderr")]
    slots = ""
    if scored:
        match = SLOTS_RE.search(scored[-1]["stderr"])
        if match:
            slots = f"{match.group(1)} slots/layer from a {match.group(2)} budget"
    print(f"  budget {meta.get('value')} ({meta.get('source')})  "
          f"want {want} slots/layer  {slots}")
    if not scored:
        print("  no child stderr recorded in this summary — check the "
              "runNN.json.stderr sidecars under the step's --workdir\n")
        continue
    # The last scored run: the one least contaminated by warm-up, and one
    # block is enough. The rest are in the JSON for anyone who wants them.
    lines = scored[-1]["stderr"].splitlines()
    for i, line in enumerate(lines):
        if "split" in line or line.startswith("experts:"):
            for out in lines[i:i + 12]:
                print(f"  {out}")
            print()
            break
    else:
        print("  no split block in the recorded stderr\n")

others = [p for p in sorted(glob.glob(
    os.path.join(root, "scratch/cold-bench/p8-*.json")))
    if os.path.abspath(p) not in read]
if others:
    print(f"  {len(others)} other p8-*.json in scratch/cold-bench belong to")
    print(f"  earlier sweeps and were NOT read: "
          f"{', '.join(os.path.basename(p) for p in others[:6])}"
          f"{' ...' if len(others) > 6 else ''}")
PYEOF

say ""
say "--- io_probe tables ---"
if [ -e "$IO_MD" ]; then
    io_md_mtime=$(stat -c %Y "$IO_MD" 2>/dev/null) || io_md_mtime=0
    if [ "${io_md_mtime:-0}" -ge "$IO_STEP_T0" ]; then
        tee -a "$SUMMARY" < "$IO_MD"
    else
        say "  $IO_MD predates the io_probe step; it is an earlier sweep's"
        say "  table and is NOT reprinted here. That step produced nothing."
    fi
else
    say "  $IO_MD not written; the io_probe step produced no table."
fi

say ""
say "this sweep's summaries: scratch/cold-bench/p8-$STAMP-*.json"
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
say "The T_BLOCK table is what CLAUDE.md's rule needs to be discharged:"
say "T_BLOCK 4 -> 8 is a performance change and owes docs/experiments.md a"
say "baseline, a result and a verdict. The baseline is"
say "$REF7_EXPECT_SHA"
say "(main's build, byte-identical to EXP-021's binary), the result is the"
say "paired table, and the verdict is yours. Until that entry exists, the"
say "decode curve above is a curve of context length AND T_BLOCK together."
say ""
say "Rule 3: these are one session on one machine. They may be combined with"
say "each other into one curve and NOT with EXP-014's, EXP-018's, EXP-019's"
say "or EXP-021's numbers. The estimated wall times printed at the top of"
say "this run are derived from EXP-021 and are scheduling aids, not data."
say ""
say "The one exception, and it is narrow: the T_BLOCK=4 reference arm is the"
say "same bytes EXP-021 measured, so a disagreement between its numbers here"
say "and EXP-021's is a statement about the two SESSIONS, not about T_BLOCK."
say "That makes it a useful check on the machine. It is still not a licence"
say "to put EXP-021's figures on the same curve as this sweep's."
