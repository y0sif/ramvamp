#!/bin/sh
# Run the Vulkan spike at unlocked CPU clocks, optionally with the CPU and the
# iGPU each locked to one fixed frequency, then put the machine back exactly
# as it was.
#
# Touches only volatile kernel state: intel_pstate/no_turbo and
# intel_pstate/max_perf_pct, each pinned CPU's scaling_governor, and with
# --lock each pinned CPU's scaling_min_freq/scaling_max_freq and the i915
# render GT's min/max/boost frequency. No TLP config file is read or written,
# tlp.service is never stopped (it is Type=oneshot and has already exited),
# and thermald keeps running throughout, so thermal and RAPL protection are
# untouched. A reboot restores every one of these regardless.
#
#   sudo sh scripts/bench-unthrottled.sh             # save, unlock, run, restore
#   sudo sh scripts/bench-unthrottled.sh --hoisted   # CPU arm runs the row-hoist
#                                                    # prototype (exp/vulkan-vs-hoist)
#   sudo sh scripts/bench-unthrottled.sh --lock      # also lock CPU and iGPU clocks
#   sudo sh scripts/bench-unthrottled.sh --lock --hoisted
#   sudo sh scripts/bench-unthrottled.sh --dry-run [--lock]  # show what would change
#   sudo sh scripts/bench-unthrottled.sh --restore   # restore from the last save
#
# --lock holds every pinned CPU at LOCK_CPU_MHZ (default 3000) and the iGPU at
# LOCK_GT_MHZ (default 1400), about 60% of each unit's peak, so neither has to
# give way to the other inside the package's shared budget. The spike checks
# the lock is in place and that the clocks it observes stay on it, and prints
# the run INVALID otherwise.
#
# The binary is built BEFORE the clocks are unlocked, then the box idles for
# COOLDOWN seconds. A `cargo run` inside the unlocked window rebuilt the whole
# tree at full turbo seconds before the timed loop, which spent the package's
# short-term power budget on the compile and left clocks settling mid-run.
#
# The restore runs from an EXIT trap, and Ctrl-C and SIGTERM exit through it,
# so it fires once on success, on failure and on interruption. Every restore
# write warns and carries on if it fails, so one failure cannot strand the
# rest.

set -eu

PSTATE=/sys/devices/system/cpu/intel_pstate
STATE=/var/tmp/ramvamp-cpu-state
LOCK_STATE=/var/tmp/ramvamp-lock-state
PINNED="0 1 3 6 8 10"
LOCK_CPU_MHZ=${LOCK_CPU_MHZ:-3000}
LOCK_GT_MHZ=${LOCK_GT_MHZ:-1400}

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }

cpufreq() { echo /sys/devices/system/cpu/cpu"$1"/cpufreq/"$2"; }

# A kernel CPU list ("0,5", "1-2") as space-separated numbers.
expand_list() {
    echo "$1" | tr ',' '\n' | while IFS=- read -r a b; do
        if [ -n "$a" ]; then seq "$a" "${b:-$a}"; fi
    done | tr '\n' ' '
}

# The pinned cpus and their hyperthread siblings. Two threads of one core
# share its clock and the core runs at the higher of their requests, so a
# lock that skips the siblings does not hold: on 2026-10-10 the locked shards
# read 2.26-4.50 GHz against 3.00 while their siblings were unlocked.
LOCKED=$(for c in $PINNED; do
    expand_list "$(cat /sys/devices/system/cpu/cpu"$c"/topology/thread_siblings_list)"
done | tr ' ' '\n' | grep . | sort -n -u | tr '\n' ' ')
# The cpus `show` prints: the pinned ones, or every locked one under --lock.
SHOWN=$PINNED

# Write a value, warning instead of aborting: used on every restore path.
put() {
    echo "$2" > "$1" 2>/dev/null || echo "  !! could not write $2 to $1" >&2
}

# Write a value that may legitimately be refused, silently. See set_limits.
try() { echo "$2" > "$1" 2>/dev/null || true; }

# Set a unit's floor and ceiling as floor, ceiling, floor. i915 refuses a
# min above the current max and a max below the current min, so the safe
# order depends on which way the new limits sit around the old ones; this
# order is safe either way. If the first floor write is refused for that
# reason, the second lands once the ceiling has moved.
set_limits() { # min_file max_file min max
    try "$1" "$3"
    put "$2" "$4"
    put "$1" "$3"
}

# The render GT's frequency files. On a part with a render GT and a media GT
# (Meteor Lake), the card-level gt_*_freq_mhz files write to every GT and read
# back aggregated across them: a lock through them half-applied on 2026-10-10,
# and the restore left the render GT pinned at 1400 MHz. So the render GT's
# own gt/gt0/rps_* files, and the card-level ones only on a single-GT part.
# Sets GT_DIR and GT_PFX; GT_DIR is empty when there is no card or several.
render_gt() {
    GT_DIR=""; GT_PFX=""; n=0
    for card in /sys/class/drm/card*; do
        case "${card##*/}" in *-*) continue ;; esac  # connectors: card1-eDP-1
        if [ -f "$card/gt/gt0/rps_min_freq_mhz" ]; then
            GT_DIR=$card/gt/gt0; GT_PFX=rps_
        elif [ -f "$card/gt_min_freq_mhz" ]; then
            GT_DIR=$card; GT_PFX=gt_
        else
            continue
        fi
        n=$((n + 1))
    done
    [ "$n" -eq 1 ] || { GT_DIR=""; GT_PFX=""; }
}

gtf() { echo "$GT_DIR/${GT_PFX}$1_freq_mhz"; }

show() {
    printf '  no_turbo=%s max_perf_pct=%s' \
        "$(cat $PSTATE/no_turbo)" "$(cat $PSTATE/max_perf_pct)"
    for c in $SHOWN; do
        printf ' cpu%s=%s,%s-%sMHz' "$c" "$(cat "$(cpufreq "$c" scaling_governor)")" \
            "$(( $(cat "$(cpufreq "$c" scaling_min_freq)") / 1000 ))" \
            "$(( $(cat "$(cpufreq "$c" scaling_max_freq)") / 1000 ))"
    done
    render_gt
    if [ -n "$GT_DIR" ]; then
        printf ' igpu=%s-%sMHz,boost%s' "$(cat "$(gtf min)")" \
            "$(cat "$(gtf max)")" "$(cat "$(gtf boost)")"
    fi
    printf '\n'
}

save() {
    : > "$STATE"
    echo "no_turbo $(cat $PSTATE/no_turbo)" >> "$STATE"
    echo "max_perf_pct $(cat $PSTATE/max_perf_pct)" >> "$STATE"
    for c in $PINNED; do
        echo "gov$c $(cat "$(cpufreq "$c" scaling_governor)")" >> "$STATE"
    done
    chmod 600 "$STATE"
    # A lock state from an earlier run must not be replayed by this one.
    rm -f "$LOCK_STATE"
}

# Saved AFTER the unlock and before the lock, so what comes back is the
# limit you run with normally, not the throttled ceiling no_turbo imposes:
# writing that ceiling back as a scaling_max_freq request would outlive this
# script and cap turbo the next time anything enabled it.
save_lock() {
    : > "$LOCK_STATE"
    for c in $LOCKED; do
        echo "cpumin$c $(cat "$(cpufreq "$c" scaling_min_freq)")" >> "$LOCK_STATE"
        echo "cpumax$c $(cat "$(cpufreq "$c" scaling_max_freq)")" >> "$LOCK_STATE"
    done
    echo "gtdir $GT_DIR" >> "$LOCK_STATE"
    echo "gtpfx $GT_PFX" >> "$LOCK_STATE"
    echo "gtmin $(cat "$(gtf min)")" >> "$LOCK_STATE"
    echo "gtmax $(cat "$(gtf max)")" >> "$LOCK_STATE"
    echo "gtboost $(cat "$(gtf boost)")" >> "$LOCK_STATE"
    chmod 600 "$LOCK_STATE"
}

saved() { sed -n "s/^$1 //p" "$2"; }

# Undo the lock.
restore_lock() {
    [ -f "$LOCK_STATE" ] || return 0
    for c in $LOCKED; do
        set_limits "$(cpufreq "$c" scaling_min_freq)" "$(cpufreq "$c" scaling_max_freq)" \
            "$(saved "cpumin$c" "$LOCK_STATE")" "$(saved "cpumax$c" "$LOCK_STATE")"
    done
    GT_DIR=$(saved gtdir "$LOCK_STATE"); GT_PFX=$(saved gtpfx "$LOCK_STATE")
    set_limits "$(gtf min)" "$(gtf max)" \
        "$(saved gtmin "$LOCK_STATE")" "$(saved gtmax "$LOCK_STATE")"
    put "$(gtf boost)" "$(saved gtboost "$LOCK_STATE")"
    rm -f "$LOCK_STATE"
}

restore() {
    [ -f "$STATE" ] || { echo "no saved state at $STATE" >&2; return 1; }
    while read -r key val; do
        case "$key" in
            no_turbo)     put $PSTATE/no_turbo "$val" ;;
            max_perf_pct) put $PSTATE/max_perf_pct "$val" ;;
            gov*)         put "$(cpufreq "${key#gov}" scaling_governor)" "$val" ;;
        esac
    done < "$STATE"
}

# The lock comes off first, while turbo is still enabled, then the turbo
# state goes back.
restore_all() {
    echo; echo "restoring..."
    restore_lock
    restore || true
    echo "restored:"; show
}

SPIKE_CPU=production
LOCK=0
DRY=0
for arg in "$@"; do
    case "$arg" in
        --hoisted) SPIKE_CPU=hoisted ;;
        --lock)    LOCK=1 ;;
        --dry-run) DRY=1 ;;
        --restore)
            echo "before:"; show
            restore_lock
            restore
            echo "restored:"; show
            exit 0
            ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ "$LOCK" -eq 1 ]; then
    SHOWN=$LOCKED
    render_gt
    [ -n "$GT_DIR" ] || { echo "--lock needs exactly one i915 card under /sys/class/drm" >&2; exit 1; }
    [ $((LOCK_CPU_MHZ % 100)) -eq 0 ] || { echo "LOCK_CPU_MHZ must be a multiple of 100" >&2; exit 1; }
    [ $((LOCK_GT_MHZ % 50)) -eq 0 ] || { echo "LOCK_GT_MHZ must be a multiple of 50" >&2; exit 1; }
    rp0=$(cat "$(gtf RP0)"); rpn=$(cat "$(gtf RPn)")
    [ "$LOCK_GT_MHZ" -le "$rp0" ] && [ "$LOCK_GT_MHZ" -ge "$rpn" ] ||
        { echo "LOCK_GT_MHZ=$LOCK_GT_MHZ is outside the render GT's ${rpn}-${rp0} MHz range" >&2; exit 1; }
    # The lock saves the current limits as "normal" and restores them after,
    # so a floor left raised by an earlier run would be saved and put back.
    gmin=$(cat "$(gtf min)")
    if [ "$gmin" -gt "$rpn" ] && [ "${ALLOW_GT_MIN:-0}" != 1 ]; then
        echo "the render GT's min is ${gmin} MHz, above its ${rpn} MHz floor; if an earlier run" >&2
        echo "left it there, reset it first:  echo $rpn | sudo tee $(gtf min)" >&2
        echo "if you set it on purpose, run with ALLOW_GT_MIN=1" >&2
        exit 1
    fi
    for c in $LOCKED; do
        hw=$(( $(cat "$(cpufreq "$c" cpuinfo_max_freq)") / 1000 ))
        # cpuinfo_max_freq reads the turbo maximum only while turbo is on;
        # with no_turbo=1 it is the base clock, so this check is skipped
        # until the unlock below has run and is repeated there.
        if [ "$(cat $PSTATE/no_turbo)" -eq 0 ] && [ "$LOCK_CPU_MHZ" -gt "$hw" ]; then
            echo "LOCK_CPU_MHZ=$LOCK_CPU_MHZ is above cpu$c's ${hw} MHz maximum" >&2; exit 1
        fi
    done
fi

if [ "$DRY" -eq 1 ]; then
    echo "current:"; show
    echo "would set: no_turbo=0 max_perf_pct=100 governor=performance on cpus $PINNED"
    if [ "$LOCK" -eq 1 ]; then
        echo "would lock: cpus $LOCKED(pinned plus hyperthread siblings) at ${LOCK_CPU_MHZ} MHz (min = max), render GT $GT_DIR at ${LOCK_GT_MHZ} MHz (min = max = boost)"
        echo "would save the lock state to: $LOCK_STATE"
    fi
    echo "would save to: $STATE"
    exit 0
fi

# Build as the invoking user, not root, so cargo uses the normal target dir
# and the normal toolchain. Resolve cargo explicitly: root's PATH does not
# have it. Built at the machine's normal (throttled) clocks, before anything
# is saved or changed, so a failed build leaves nothing to restore.
RUN_AS=${SUDO_USER:-root}
HOME_DIR=$(getent passwd "$RUN_AS" | cut -d: -f6)
CARGO="$HOME_DIR/.cargo/bin/cargo"
[ -x "$CARGO" ] || CARGO=$(command -v cargo || echo cargo)
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
COOLDOWN=${COOLDOWN:-20}

echo "building as $RUN_AS in $REPO ..."
sudo -u "$RUN_AS" HOME="$HOME_DIR" sh -c \
    "cd '$REPO' && '$CARGO' build --release -p ramvamp-vulkan-spike"
echo "cooling down for ${COOLDOWN}s before unlocking clocks ..."
sleep "$COOLDOWN"
echo

# Other work on the box is the commonest reason a run here is unsteady, so
# say what is running before spending the run. A build in the last minute,
# including the one above, counts toward the load average too.
load=$(cut -d' ' -f1 /proc/loadavg)
echo "load average $(cut -d' ' -f1-3 /proc/loadavg); busiest processes (%cpu over their lifetime):"
ps -eo pcpu,comm --sort=-pcpu | sed '1d' | grep -v ' ps$' | head -5 | sed 's/^/   /'
if awk -v l="$load" 'BEGIN { exit !(l > 2.0) }'; then
    echo "  !! the box looks busy (1-minute load $load): other builds, agents or a browser"
    echo "  !! compete with the CPU arm, and the run will likely come back unsteady"
fi
echo

save
echo "saved to $STATE:"; show

# One restore on every exit path: Ctrl-C and SIGTERM exit, and exit runs it.
trap restore_all EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo 0   > $PSTATE/no_turbo
echo 100 > $PSTATE/max_perf_pct
for c in $PINNED; do
    echo performance > "$(cpufreq "$c" scaling_governor)"
done
echo "unlocked:"; show

LOCK_ENV=""
if [ "$LOCK" -eq 1 ]; then
    for c in $LOCKED; do
        hw=$(( $(cat "$(cpufreq "$c" cpuinfo_max_freq)") / 1000 ))
        [ "$LOCK_CPU_MHZ" -le "$hw" ] || { echo "LOCK_CPU_MHZ=$LOCK_CPU_MHZ is above cpu$c's ${hw} MHz maximum" >&2; exit 1; }
    done
    save_lock
    for c in $LOCKED; do
        set_limits "$(cpufreq "$c" scaling_min_freq)" "$(cpufreq "$c" scaling_max_freq)" \
            $((LOCK_CPU_MHZ * 1000)) $((LOCK_CPU_MHZ * 1000))
    done
    set_limits "$(gtf min)" "$(gtf max)" "$LOCK_GT_MHZ" "$LOCK_GT_MHZ"
    put "$(gtf boost)" "$LOCK_GT_MHZ"
    echo "locked:"; show
    # set_limits only warns, so a refused write would leave a partial lock.
    # The spike would catch it and print INVALID, but there is no point
    # spending a run on it: stop here, and the EXIT trap puts everything back.
    for c in $LOCKED; do
        [ "$(cat "$(cpufreq "$c" scaling_min_freq)")" = "$(cat "$(cpufreq "$c" scaling_max_freq)")" ] ||
            { echo "cpu$c did not lock (min != max); not running" >&2; exit 1; }
    done
    [ "$(cat "$(gtf min)")" = "$(cat "$(gtf max)")" ] &&
        [ "$(cat "$(gtf max)")" = "$(cat "$(gtf boost)")" ] ||
        { echo "the iGPU did not lock (min, max and boost differ); not running" >&2; exit 1; }
    # Pass the values the kernel actually accepted, not the ones asked for;
    # the spike checks every pinned CPU and the iGPU against them.
    first=${PINNED%% *}
    LOCK_ENV="RAMVAMP_SPIKE_LOCK_CPU_KHZ=$(cat "$(cpufreq "$first" scaling_max_freq)") RAMVAMP_SPIKE_LOCK_GT_MHZ=$(cat "$(gtf max)")"
fi
echo

echo "running as $RUN_AS in $REPO, CPU kernel: $SPIKE_CPU${LOCK_ENV:+, $LOCK_ENV}"
echo
# A failure here must not skip the restore, hence `|| true` on top of the
# trap. $LOCK_ENV is unquoted on purpose: it is zero or two VAR=value words.
# shellcheck disable=SC2086
sudo -u "$RUN_AS" HOME="$HOME_DIR" RAMVAMP_SPIKE_CPU="$SPIKE_CPU" $LOCK_ENV sh -c \
    "cd '$REPO' && ./target/release/ramvamp-vulkan-spike" || true
