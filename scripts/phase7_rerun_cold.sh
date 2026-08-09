#!/usr/bin/env bash
# Re-run only the cold rule-2 steps that came back DIRTY.
#
#     bash scripts/phase7_rerun_cold.sh
#
# Why the first attempt went DIRTY, and what is different here.
#
# The reclaim was not CPU contention. It was global memory pressure: the
# numerics gate ran the model for ~15 minutes immediately before the cold runs,
# leaving several GB of page cache and pushing pages into zram, and global
# reclaim can steal pages charged to the cgroup even while the cgroup sits under
# its own 3G limit. The evidence is that the DIRTY runs were the EARLY ones and
# the run settled into cleanliness by its fourth iteration, with wall time
# unchanged (307-311 s) across both.
#
# So this script runs the cold steps ONLY, with no model work before them, and
# waits for the system to actually be settled before each one.
#
# Roughly 55 minutes, most of it the phase-5 prefill arm.

set -u
set -o pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd -P)

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/scratch/phase7/recold-$STAMP"
mkdir -p "$OUT" "$ROOT/scratch/cold-bench" || exit 1
SUMMARY="$OUT/SUMMARY.txt"
: > "$SUMMARY"

# None of the paths below is in the repository: .gitignore excludes /models/
# (line 5) and /scratch/ (line 9). They are a default layout, not a promise
# that the bytes are present. A fresh clone holds neither the installed .rvmp
# model nor the llama.cpp reference tree that supplies P512 -- that tree has
# to be re-banked from llama.cpp b10217 -- and neither the phase-5 reference
# binary nor the 4K prompt under /scratch/.
RAMVAMP="$ROOT/target/release/ramvamp"
REF5="$ROOT/scratch/phase5-ref/ramvamp"
RVMP="$ROOT/models/qwen3.rvmp"
P512="$ROOT/models/llamacpp-ref/llamacpp_ref/long_00.txt"
P4K="$ROOT/scratch/ctx4k/p4k.txt"

# MemAvailable we want before starting a cold run, in MiB. The workload peaks
# near 2,920 MiB at 4K context, so this leaves real slack rather than just
# enough.
WANT_AVAIL_MIB=6000
# Settle: MemAvailable must stay above the threshold for this many consecutive
# 15-second samples before a run starts.
SETTLE_SAMPLES=4
SETTLE_MAX_WAIT=900

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

run() {
    local label=$1 log=$2
    shift 2
    printf '\n>>> [%s] %s\n' "$(stamp)" "$label" | tee -a "$SUMMARY"
    settle
    local t0 t1 rc
    t0=$(date +%s)
    "$@" > "$OUT/$log" 2>&1
    rc=$?
    t1=$(date +%s)
    local hyg
    hyg=$(grep -a -o 'measurement hygiene: [A-Z]*' "$OUT/$log" | tail -1)
    printf '    exit=%d  %dm%02ds  %s  log=%s\n' \
        "$rc" $(( (t1-t0)/60 )) $(( (t1-t0)%60 )) "${hyg:-hygiene: ?}" "$log" | tee -a "$SUMMARY"
    printf '%s\t%s\t%s\t%s\n' "$label" "$rc" "$((t1-t0))" "${hyg:-?}" >> "$OUT/exitcodes.tsv"
    # Let the page cache the run just built drain before the next one.
    sleep 60
}

say "ramvamp phase 7 — cold rule-2 re-run"
say "started $(stamp)"
say "output  $OUT"
say "commit  $(git rev-parse --short HEAD 2>/dev/null) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
say ""

if pgrep -x ramvamp > /dev/null 2>&1; then
    say "FATAL: a ramvamp process is running; cold_bench cannot evict a file"
    say "another process holds mmap'd. Kill it and restart."
    exit 2
fi
for f in "$RAMVAMP" "$REF5" "$RVMP" "$P512"; do
    [ -e "$f" ] || { say "FATAL missing: $f"; exit 2; }
done

say "MemAvailable now: $(avail_mib) MiB, swap in use: $(swap_mib) MiB"
say ""
say "If this sits waiting to settle, close what you can spare (a browser and"
say "Slack are usually most of it). CPU idle is not what matters here; free"
say "memory is."
say ""

run "cold prefill 512, phase-5 arm, repeats 5" "01-prefill-p5.log" \
    python3 scripts/cold_bench.py \
        --ramvamp "$REF5" --rvmp "$RVMP" --prompt-file "$P512" \
        --max-new 4 --warmup 1 --repeats 5 \
        --json scratch/cold-bench/re-p5-512.json

run "cold decode max-new 256, phase-5 arm" "02-decode-p5.log" \
    python3 scripts/cold_bench.py \
        --ramvamp "$REF5" --rvmp "$RVMP" --prompt-file "$P512" \
        --max-new 256 --warmup 0 --repeats 1 \
        --json scratch/cold-bench/re-p5-512-n256.json

run "cold decode max-new 256, phase-7 arm" "03-decode-p7.log" \
    python3 scripts/cold_bench.py \
        --ramvamp "$RAMVAMP" --rvmp "$RVMP" --prompt-file "$P512" \
        --max-new 256 --warmup 0 --repeats 1 \
        --json scratch/cold-bench/re-p7-512-n256.json

if [ -e "$P4K" ]; then
    run "cold 4K context memory.peak" "04-4k.log" \
        python3 scripts/cold_bench.py \
            --ramvamp "$RAMVAMP" --rvmp "$RVMP" --prompt-file "$P4K" \
            --max-new 8 --warmup 0 --repeats 1 \
            --json scratch/cold-bench/re-p7-4k.json
else
    say ">>> [SKIPPED] 4K: $P4K not found"
fi

say ""
say "=============================================================="
say "finished $(stamp)"
say ""
while IFS=$'\t' read -r label rc secs hyg; do
    printf '  %-44s exit=%-3s %dm%02ds  %s\n' "$label" "$rc" $((secs/60)) $((secs%60)) "$hyg"
done < "$OUT/exitcodes.tsv" | tee -a "$SUMMARY"

say ""
say "--- numbers ---"
for f in "$OUT"/*.log; do
    printf '### %s\n' "$(basename "$f")" | tee -a "$SUMMARY"
    grep -a -E 'median (wall_s|prefill_tok_s|decode_tok_s|read_bytes|MemoryPeak)|measurement hygiene|pgsteal [0-9]' "$f" 2>/dev/null | tee -a "$SUMMARY"
    printf '\n' | tee -a "$SUMMARY"
done

say ""
say "Any step still DIRTY: check whether the reclaim is a few MiB with wall"
say "times unchanged across clean and dirty iterations. That is memory"
say "pressure at the margin, not a working set that did not fit, and it is"
say "worth recording as such rather than re-running forever."
