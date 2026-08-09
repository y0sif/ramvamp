#!/usr/bin/env bash
# Phase 7 measurement run, unattended.
#
# Takes phase 7's whole cold measurement in one go: the numerics gate, the
# paired cold rule-2 runs, the 4K context run and a quiet in-process bench.
# Roughly 2.5 to 3 hours. Nothing here is interactive.
#
# Start it and walk away:
#
#     nohup bash scripts/phase7_overnight.sh > /dev/null 2>&1 &
#
# Then read scratch/phase7/overnight-<stamp>/SUMMARY.txt in the morning.
#
# Deliberately does NOT abort on the first failure. A failed numerics gate is
# worth knowing about, but it is not a reason to waste the rest of the night, so
# every step runs and every exit code lands in the summary.
#
# The cold runs need the machine quiet. That is the whole reason this script
# exists: cold_bench.py gates on pgsteal == 0 and a single browser tab can push
# a run into reclaim and invalidate it.

set -u
set -o pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd -P)

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/scratch/phase7/overnight-$STAMP"
mkdir -p "$OUT" || exit 1
mkdir -p "$ROOT/scratch/cold-bench" || exit 1

SUMMARY="$OUT/SUMMARY.txt"
: > "$SUMMARY"

RAMVAMP="$ROOT/target/release/ramvamp"
REF5="$ROOT/scratch/phase5-ref/ramvamp"
RVMP="$ROOT/models/qwen3.rvmp"
P512="$ROOT/models/llamacpp-ref/llamacpp_ref/long_00.txt"
P4K="$ROOT/scratch/ctx4k/p4k.txt"

say() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# run <label> <logfile> <command...>
# Records wall time and the command's OWN exit code. Never aborts the script.
#
# DRY_RUN=1 prints what would run and skips it, so the scaffolding, the
# preflight and the summary can be checked in a few seconds before committing a
# night to this.
run() {
    local label=$1 log=$2
    shift 2
    printf '\n>>> [%s] %s\n' "$(stamp)" "$label" | tee -a "$SUMMARY"
    local t0 t1 rc
    t0=$(date +%s)
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf 'DRY_RUN, would have run:\n%s\n' "$*" > "$OUT/$log"
        printf '    would run: %s\n' "$*" | tee -a "$SUMMARY"
        rc=0
    else
        "$@" > "$OUT/$log" 2>&1
        rc=$?
    fi
    t1=$(date +%s)
    printf '    exit=%d  %dm%02ds  log=%s\n' "$rc" $(( (t1-t0)/60 )) $(( (t1-t0)%60 )) "$log" | tee -a "$SUMMARY"
    printf '%s\t%s\t%s\n' "$label" "$rc" "$((t1-t0))" >> "$OUT/exitcodes.tsv"
    return $rc
}

say "ramvamp phase 7 overnight measurement"
say "started $(stamp)"
say "output  $OUT"
say "commit  $(git rev-parse --short HEAD 2>/dev/null) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
say ""

# ---------------------------------------------------------------- preflight --
say "--- preflight ---"
FATAL=0
for f in "$REF5" "$RVMP" "$P512"; do
    if [ ! -e "$f" ]; then say "MISSING: $f"; FATAL=1; fi
done
[ -e "$P4K" ] || say "note: $P4K missing, the 4K step will be skipped"

if pgrep -x ramvamp > /dev/null 2>&1; then
    say "FATAL: a ramvamp process is already running."
    say "cold_bench.py cannot evict a file another process holds mmap'd, and"
    say "fadvise returns 0 while evicting nothing, so the run would look clean"
    say "and be warm. Kill it and restart."
    FATAL=1
fi

if [ "$FATAL" -ne 0 ]; then
    say ""
    say "aborting before doing any work."
    exit 2
fi

say "load average now: $(cut -d' ' -f1-3 /proc/loadavg)"
say "kernel: $(uname -r)"
say ""

# ------------------------------------------------------------------- build --
run "build release binary" "00-build.log" cargo build --release
if [ ! -x "$RAMVAMP" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    say "FATAL: $RAMVAMP was not produced. Stopping."
    exit 2
fi
[ -x "$RAMVAMP" ] && say "binary sha256: $(sha256sum "$RAMVAMP" | cut -d' ' -f1)"
say "phase-5 ref  : $(sha256sum "$REF5" | cut -d' ' -f1)"

# Let the build's heat dissipate before anything is timed.
[ "${DRY_RUN:-0}" = "1" ] || sleep 60

# ------------------------------------------------- 1. in-process bench, warm --
# First, while the machine is coldest and nothing else has run. Diagnostic only.
# Check the drift-control rows in the log before believing any cell: a ratio
# outside 0.99x-1.01x means the run was contaminated, not that the kernel moved.
run "in-process attention bench (warm, diagnostic)" "01-bench.log" \
    cargo bench -p ramvamp-core

# ------------------------------------------------------- 2. numerics gate ----
# Cheapest way to find out whether bits moved. If 2a fails, the performance
# numbers below are meaningless, but they still get collected.

run "bitident vs phase-4 baseline" "02-bitident.log" \
    python3 scripts/bitident.py compare models/llamacpp-ref/phase4-baseline \
        --ramvamp "$RAMVAMP" --rvmp "$RVMP"

# --refresh is NOT optional: the per-prompt dumps are cached and the cache is
# not keyed on the binary, so without it this re-scores stale JSON in seconds
# and reports PASS having tested nothing about this build.
run "full-vocab KL vs llama.cpp reference" "03-kl.log" \
    python3 scripts/kl_vs_reference.py --rvmp "$RVMP" \
        --ref models/llamacpp-ref/llamacpp_ref \
        --ramvamp "$RAMVAMP" --skip-longs --refresh

run "greedy regression (~50 min)" "04-greedy.log" \
    python3 scripts/greedy_regression.py --ramvamp "$RAMVAMP"

# --------------------------------------------------- 3. cold rule-2 runs -----
# memory.max=3G, memory.swap.max=0, page cache evicted, gated on pgsteal == 0.
# Each pair runs back to back so both arms see the same machine state.

run "cold prefill 512, phase-5 arm, repeats 5" "05-cold-prefill-p5.log" \
    python3 scripts/cold_bench.py \
        --ramvamp "$REF5" --rvmp "$RVMP" --prompt-file "$P512" \
        --max-new 4 --warmup 1 --repeats 5 \
        --json scratch/cold-bench/p7-p5-512.json

run "cold prefill 512, phase-7 arm, repeats 5" "06-cold-prefill-p7.log" \
    python3 scripts/cold_bench.py \
        --ramvamp "$RAMVAMP" --rvmp "$RVMP" --prompt-file "$P512" \
        --max-new 4 --warmup 1 --repeats 5 \
        --json scratch/cold-bench/p7-p7-512.json

# The decode pair is what settles EXP-018's cold-start question. At --max-new 4
# almost everything measured was the transient; 256 tokens is 5x EXP-005's
# token-48 steady-state threshold, so the transient amortizes.
run "cold decode max-new 256, phase-5 arm" "07-cold-decode-p5.log" \
    python3 scripts/cold_bench.py \
        --ramvamp "$REF5" --rvmp "$RVMP" --prompt-file "$P512" \
        --max-new 256 --warmup 0 --repeats 1 \
        --json scratch/cold-bench/p7-p5-512-n256.json

run "cold decode max-new 256, phase-7 arm" "08-cold-decode-p7.log" \
    python3 scripts/cold_bench.py \
        --ramvamp "$RAMVAMP" --rvmp "$RVMP" --prompt-file "$P512" \
        --max-new 256 --warmup 0 --repeats 1 \
        --json scratch/cold-bench/p7-p7-512-n256.json

# The last open item on the 11-slot dial (EXP-014 Note 2). p4k.txt is ~3,920
# tokens, inside CONTEXT_CAP = 4096. Do not substitute prompt.txt, which is
# ~13,800 tokens and over the cap.
if [ -e "$P4K" ]; then
    run "cold 4K context memory.peak" "09-cold-4k.log" \
        python3 scripts/cold_bench.py \
            --ramvamp "$RAMVAMP" --rvmp "$RVMP" --prompt-file "$P4K" \
            --max-new 8 --warmup 0 --repeats 1 \
            --json scratch/cold-bench/p7-4k.json
else
    say ""
    say ">>> [SKIPPED] cold 4K context: $P4K not found"
    printf '%s\t%s\t%s\n' "cold 4K context (skipped)" "-" "0" >> "$OUT/exitcodes.tsv"
fi

# ------------------------------------------------------------------ summary --
say ""
say "=============================================================="
say "finished $(stamp)"
say ""
say "step results (exit 0 is good):"
say ""
while IFS=$'\t' read -r label rc secs; do
    printf '  %-46s exit=%-4s %dm%02ds\n' "$label" "$rc" $((secs/60)) $((secs%60))
done < "$OUT/exitcodes.tsv" | tee -a "$SUMMARY"

say ""
say "--- headline lines pulled from the logs ---"
say ""
{
    for f in "$OUT"/*.log; do
        printf '### %s\n' "$(basename "$f")"
        grep -a -E 'bit-identity:|gate 3|verdict:|measurement hygiene|prefill:|decode:|memory\.peak|memory_peak|tok/s|PASS|FAIL|DIRTY|CLEAN|pgsteal' "$f" 2>/dev/null | head -18
        printf '\n'
    done
} | tee -a "$SUMMARY"

say ""
say "--- decode phase split (the new instrument) ---"
grep -a -A 9 'decode split' "$OUT/08-cold-decode-p7.log" 2>/dev/null | head -30 | tee -a "$SUMMARY"

say ""
say "--- bench drift controls: outside 0.99x-1.01x means DISCARD, not interpret ---"
grep -a -A 4 'drift control' "$OUT/01-bench.log" 2>/dev/null | tee -a "$SUMMARY"

say ""
say "raw cold-run JSON is in scratch/cold-bench/p7-*.json"
say "full logs are in $OUT"
say ""
say "Rule 3: these are one session on one machine. They may be combined with"
say "each other but NOT with EXP-014's or EXP-018's numbers into one curve."
