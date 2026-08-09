#!/usr/bin/env python3
"""Cold, cgroup-confined benchmark with *verified* measurement hygiene.

`docs/experiments/README.md` rule: published numbers come from cold runs
inside a `memory.max=3G` cgroup with `memory.swap.max=0`. Every part of
that sentence is easy to believe and hard to actually get, so this
harness verifies each one instead of assuming it, and prints a hygiene
verdict that is separate from the performance numbers. A fast number from
a DIRTY run is not a result.

What it checks, and why each check exists:

1. No `ramvamp` process may be running when we start.
   `posix_fadvise(POSIX_FADV_DONTNEED)` returns 0 and evicts *nothing* if
   any process still has the file mmap'd, and the runtime mmaps
   `common.bin`. A leftover process silently turns every "cold" run into
   a warm one.

2. Eviction is verified with `mincore`, never trusted from the fadvise
   return code. We mmap each file, count resident pages before and after,
   and abort if anything is still resident. (Reference measurement: 965.9
   MiB -> 0.0 MiB across 49 files in ~1.8 s.) Read-only throughout: the
   model directory is never written to.

3. The run happens inside a transient systemd *service*, not a scope:
   `systemd-run --user --wait -q -p MemoryMax=3G -p MemorySwapMax=0
   -p MemoryAccounting=yes`. `--scope` tears the cgroup down before the
   counters can be read, and `--wait` cannot be combined with
   `--remain-after-exit`, so the counters are read from *inside* the
   cgroup by an inner wrapper (this same script, `--inner`) just before
   it exits.

4. `memory.events` alone is not sufficient evidence of a clean run. A
   measured run showed `max 0` while 2.4 GiB had already been silently
   reclaimed. So the verdict keys on `pgsteal` from `memory.stat` — but on
   the *pressure* reclaimers only, not on the bare total.

   `pgsteal_khugepaged` is excluded, and that is a correction rather than
   a loosening. khugepaged is the transparent-hugepage daemon: it wakes on
   its own 10-second timer, scans a bounded number of pages, and frees the
   base pages it collapses into 2 MiB hugepages. That freeing lands in
   `pgsteal_khugepaged` while nothing about the workload is under memory
   pressure. Measured here: 11 runs across two sessions came back
   "reclaimed under pressure" with `pgsteal_khugepaged` accounting for
   100% of it and `kswapd`, `direct` and `proactive` all zero, on a
   machine whose Normal zone sat 16x above the watermark that wakes
   kswapd. `pgscan == pgsteal` exactly in every one, a 100% steal rate
   that is the signature of targeted freeing rather than LRU scanning
   under pressure. Wall times were statistically identical to the clean
   runs (mean 309.13 s dirty against 310.35 s clean over 12 runs of one
   workload), which is the expected sign: collapsing to hugepages helps
   the TLB.

   Everything that is not khugepaged still counts as pressure, including
   any reclaimer this script does not know the name of, and a `memory.stat`
   with no breakdown at all still counts the whole total — unknown stays
   DIRTY.

5. Bytes actually fetched from the block layer are reported. `read_bytes`
   in `/proc/<pid>/io` is per-task and is *not* inherited by the parent
   on exit, so the inner wrapper samples the child's `/proc/<pid>/io`
   while it runs and also takes `getrusage(RUSAGE_CHILDREN).ru_inblock`
   after reaping it — the same kernel counter, in 512-byte units, exact.
   A delta of ~0 on a supposedly cold run means eviction failed.

6. Run 1 is contaminated by btrfs metadata warm-up, so `--repeats N`
   discards `--warmup` runs (default 1) and reports the median of the
   rest.

7. Every counter has three states, not two: clean, dirty and *unknown*,
   and unknown is DIRTY. `/proc/self/cgroup` can fail to resolve, the
   memory controller can be undelegated, `memory.max` can be unreadable —
   each of which leaves the harness with no evidence at all, which is
   precisely the state an unconfined run produces. A check that exempts
   its own unreadable input is not a check, so a missing counter is
   reported as a hard problem with the counter named. The one counter that
   is systematically rather than diagnostically absent —
   `memory.swap.peak`, which does not exist before Linux 6.5 — is caught
   by `preflight()` instead, because "DIRTY forever, no remedy" is a
   broken harness rather than a failing measurement.

8. The workload argv is handed to the inner wrapper through a **file**,
   not through `systemd-run`'s command line. systemd expands `${NAME}` and
   unescapes `$$` inside `ExecStart=` arguments, and it does so silently:
   measured on systemd 261, `A ${HOME} B` arrives as `A /home/y0sif B`,
   `A ${UNSET} B` arrives as `A  B`, and `A $$VAR B` arrives as `A $VAR B`.
   (Bare `$VAR`, `%` specifiers, newlines, tabs, quotes and backslashes all
   survive, so short English prompts never tripped it.) At phase-6 prefill
   lengths the prompt is a ~17 KB document that may contain shell, LaTeX or
   template text, so `${` is not exotic and a silently shortened prompt is a
   silently wrong prefill measurement. `one_run()` therefore writes the argv
   as JSON to `<workdir>/runNN.cmd.json` and passes only `--command-file` on
   the systemd-run command line; the inner wrapper loads it and `exec`s the
   list directly, which is byte-exact. `--command` still works for ad-hoc
   use, with the same caveat it always had.

Usage:

  scripts/cold_bench.py --ramvamp target/release/ramvamp --max-new 8
  scripts/cold_bench.py --ramvamp target/release/ramvamp \\
      --max-new 64 --repeats 5 --json scratch/cold.json
  scripts/cold_bench.py --ramvamp target/release/ramvamp \\
      --prompt-file bench/prompts/4k.txt --max-new 1 --repeats 5

Exit codes, shared with the repo's other gate scripts (`bitident.py`,
`greedy_regression.py`, `kl_vs_reference.py`, `lfu_sim.py`):

  0  every scored run was CLEAN
  1  at least one scored run was DIRTY. This is a real result about the
     run, so the numbers are still printed and clearly marked. **An
     unreadable counter lands here, not on 2**: the harness ran fine, it
     simply cannot show the run was clean, and unknown is DIRTY (see 7
     above)
  2  the harness could not take the measurement at all — eviction failed,
     no cgroup v2, no `systemd-run --user`, a kernel older than 6.5, a
     `ramvamp` process was already running, the inner run produced no
     result file

## First run on phase-4 code (2026-08-03, commit 650b5ea + scripts, 185H)

`--max-new 8`, 5-token prompt, 1 warmup + 1 scored. Eviction verified at
2.9-3.6 GiB -> 0.0 MiB across 53 files in ~1.7 s each time. Every run
came back **DIRTY**, reproducibly:

  MemoryPeak  3072.0 MiB  (pinned to the 3 GiB limit)
  memory.events  max=6194   pgscan 1264371   pgsteal 1264370
  read_bytes  7841.2 MiB for 5 prefill + 8 decode tokens
  wall 17.5 s, prefill 0.67 tok/s, decode 0.87 tok/s

That is not a harness failure — it is the measurement. Phase 4 reads
experts with buffered `pread`, so ~7.8 GiB of expert bytes land in the
page cache, the cgroup pins at its 3 GiB ceiling, and ~4.82 GiB is
reclaimed *inside a 13-token run* (1,264,370 x 4 KiB pages; EXP-006's own
run reclaimed 1,263,046 pages, which is the same ~4.82 GiB — the two runs
differ, the figure does not). Nothing OOMs (oom=0, oom_kill=0) and
`memory.events max` is nonzero here, but note it would not have to be:
`pgsteal` is the check that catches reclaim `memory.events` misses. It
is not unfoolable — it is only as good as `memory.stat` being readable,
which is why an unreadable `memory.stat` is itself a hard problem (see 7
above) rather than a pgsteal of 0. So no phase-4
number from this harness is publishable under the docs rule, and phase 5
(O_DIRECT, which keeps expert bytes out of the page cache entirely)
should be the thing that first turns this verdict CLEAN. That transition
is itself a result worth recording in `docs/experiments/README.md`.

Python stdlib only. Linux **>= 6.5** + cgroup v2 + systemd --user only, by
design. The kernel floor is `memory.swap.peak`, which first appears in
6.5: rule 7 makes an unreadable counter DIRTY, and on an older kernel
that one counter is *always* unreadable, so every run would be reported
DIRTY with nothing the user could do about it. `preflight()` refuses up
front instead. (`memory.peak` arrived in the same release and is read
here, but nothing classifies on it, so only `memory.swap.peak` sets the
floor.)
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import time

PAGE = os.sysconf("SC_PAGESIZE")
PROT_READ, MAP_SHARED = 0x1, 0x01
CGROUP_ROOT = "/sys/fs/cgroup"

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_ubyte)]


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


def mib(n: float) -> str:
    return f"{n / 2**20:.1f} MiB"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------


# Linux caps a *single* argv element at MAX_ARG_STRLEN = 32 pages, separately
# from the ARG_MAX total. The prompt is one argv element of the ramvamp
# command, so a prompt past this ceiling fails with a bare E2BIG
# ("Argument list too long") that says nothing about which argument was too
# long. A ~4000-token prompt is ~17 KB, so this is headroom, not a limit.
MAX_ARG_STRLEN = 32 * PAGE


def read_prompt_file(path: str) -> tuple[str, dict]:
    """The benchmark prompt read once from `path`, and its provenance.

    Exactly one trailing newline is stripped: most editors and `printf`
    leave one behind, it is a token of its own, and prefill throughput is
    reported per token — so an invisible `\\n` would shift the very number
    the phase-6 sweep exists to measure. Exactly one, so a prompt that
    deliberately ends in a blank line can still express that with two. A
    trailing CRLF is one terminator, not two, and is removed whole rather
    than left as a dangling CR.

    Read as bytes and decoded explicitly, *not* via text-mode `open`: text
    mode applies universal-newline translation, which silently rewrites
    every CRLF in the file to LF. Tokenizers do not treat those alike, so
    that would be the same class of bug as the systemd `${}` expansion in
    item 8 — the prompt measured would not be the prompt on disk.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        fail(f"cannot read --prompt-file {path}: {e}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        fail(f"--prompt-file {path} is not valid UTF-8: {e}")
    if text.endswith("\r\n"):
        text = text[:-2]
    elif text.endswith("\n"):
        text = text[:-1]
    if not text:
        fail(f"--prompt-file {path} is empty (after stripping one trailing "
             f"newline). A zero-token prompt has no prefill to measure — "
             f"check the path.")
    size = len(text.encode("utf-8"))
    if size > MAX_ARG_STRLEN:
        fail(f"--prompt-file {path} is {size} bytes, over the "
             f"{MAX_ARG_STRLEN}-byte MAX_ARG_STRLEN ceiling on one argv "
             f"element; the workload would die with a bare E2BIG. Pass the "
             f"prompt to ramvamp another way, or shorten it.")
    # Both hashes are recorded: `file_sha256` identifies the artifact on
    # disk, `sha256` (added by the caller) identifies the bytes actually
    # handed to ramvamp. They differ by the stripped newline, and a reader
    # who cannot see both cannot tell which prompt was measured.
    return text, {"file_sha256": hashlib.sha256(raw).hexdigest(),
                  "file_bytes": len(raw)}


def display_workload(workload: list[str], max_arg: int = 72) -> str:
    """The workload as one line, with long arguments elided.

    A phase-6 prompt is ~17 KB, and pasting it into every `=== run N ===`
    header makes the run log unreadable — the reason `--prompt-file` exists.
    The elision keeps a sha256 prefix so a log line still ties back to the
    `prompt` provenance block in the JSON. Display only: the JSON records
    the argv whole.
    """
    parts = []
    for arg in workload:
        if len(arg) > max_arg:
            parts.append(f"<{len(arg)} chars sha256:{sha256_text(arg)[:12]}>")
        else:
            parts.append(arg)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# page-cache eviction, verified (adapted from the scratchpad evict.py)
# ---------------------------------------------------------------------------


def resident_bytes(path: str) -> int:
    """Resident page-cache bytes for `path`, via mmap + mincore."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            return 0
        addr = _libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value, 2**64 - 1):
            raise OSError(ctypes.get_errno(), f"mmap {path}")
        try:
            npages = (size + PAGE - 1) // PAGE
            vec = (ctypes.c_ubyte * npages)()
            if _libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                raise OSError(ctypes.get_errno(), f"mincore {path}")
            return sum(1 for b in vec if b & 1) * PAGE
        finally:
            _libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


def model_files(rvmp: str) -> list[str]:
    """Every file the runtime touches: common.bin, the 48 expert slabs,
    the manifest and the tokenizer."""
    files = []
    for pattern in ("common.bin", "manifest.json",
                    os.path.join("experts", "*.bin"),
                    os.path.join("tokenizer", "*")):
        files.extend(sorted(glob.glob(os.path.join(rvmp, pattern))))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        fail(f"no model files under {rvmp}")
    return files


def evict_verified(files: list[str], verbose: bool = True) -> dict:
    """fadvise DONTNEED every file, then prove with mincore that it worked."""
    t0 = time.time()
    before = after = 0
    stuck = []
    for path in files:
        b = resident_bytes(path)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        a = resident_bytes(path)
        before += b
        after += a
        if a:
            stuck.append((a, path))
    elapsed = time.time() - t0
    if verbose:
        print(f"  evict: {mib(before)} -> {mib(after)} across {len(files)} "
              f"files in {elapsed:.1f}s")
    if after:
        for a, path in sorted(stuck, reverse=True)[:10]:
            print(f"    STILL RESIDENT {mib(a):>12}  {path}", file=sys.stderr)
        fail(f"eviction failed: {mib(after)} still resident. posix_fadvise "
             f"silently does nothing while a process has the file mmap'd — "
             f"check for a stray ramvamp (or an editor/indexer holding it).")
    return {"resident_before": before, "resident_after": after,
            "files": len(files), "seconds": round(elapsed, 2)}


# ---------------------------------------------------------------------------
# cgroup v2 counters, read from inside the cgroup
# ---------------------------------------------------------------------------


def own_cgroup_dir() -> str | None:
    try:
        with open("/proc/self/cgroup", encoding="ascii") as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    return os.path.join(CGROUP_ROOT, parts[2].lstrip("/"))
    except OSError:
        return None
    return None


def read_kv(path: str) -> dict[str, int]:
    """Parse a `key value` table. cgroup files use a space, `/proc/*/io`
    uses `key: value`, so the trailing colon is stripped either way."""
    out: dict[str, int] = {}
    try:
        with open(path, encoding="ascii") as f:
            for line in f:
                bits = line.split()
                if len(bits) == 2:
                    try:
                        out[bits[0].rstrip(":")] = int(bits[1])
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def read_int(path: str) -> int | None:
    try:
        with open(path, encoding="ascii") as f:
            text = f.read().strip()
    except OSError:
        return None
    if text == "max":
        return -1
    try:
        return int(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# workload argv, passed to the inner wrapper out of band
# ---------------------------------------------------------------------------


def write_command_file(path: str, command: list[str]) -> None:
    """Serialize the workload argv as JSON for the inner wrapper.

    See item 8 of the module docstring: systemd expands `${NAME}` and
    unescapes `$$` in `ExecStart=` arguments, so anything routed through
    `systemd-run`'s command line is not what the workload receives. JSON
    round-trips arbitrary UTF-8 — newlines, quotes, `${...}`, `$$` — with no
    escaping rules of our own to get wrong.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(command, f)


def load_command_file(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            command = json.load(f)
    except (OSError, ValueError) as e:
        fail(f"cannot read the command file {path}: {e}")
    if (not isinstance(command, list) or not command
            or not all(isinstance(a, str) for a in command)):
        fail(f"the command file {path} is not a non-empty list of strings")
    return command


# ---------------------------------------------------------------------------
# inner: runs INSIDE the cgroup
# ---------------------------------------------------------------------------


TIMING_RE = re.compile(
    r"prefill:\s*(\d+)\s*tokens in\s*([\d.]+)s\s*\(([\d.]+) tok/s\);\s*"
    r"decode:\s*(\d+)\s*tokens in\s*([\d.]+)s\s*\(([\d.]+) tok/s\)")
LOAD_RE = re.compile(r"model loaded in\s*([\d.]+)s")


def inner(args) -> int:
    cg = own_cgroup_dir()
    result: dict = {"cgroup": cg, "argv": args.command}

    io_before = read_kv("/proc/self/io")
    # Spool the child's output to files, never to a pipe: we poll instead
    # of draining, and a pipe that fills (64 KiB) would deadlock the child.
    out_path = args.result_file + ".stdout"
    err_path = args.result_file + ".stderr"
    t0 = time.time()
    with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
        proc = subprocess.Popen(args.command, stdout=out_f, stderr=err_f)
        # `read_bytes` is per-task and vanishes with the task, so sample
        # the child while it lives; getrusage below is the exact
        # cross-check.
        peak_child_read = 0
        while proc.poll() is None:
            sample = read_kv(f"/proc/{proc.pid}/io")
            peak_child_read = max(peak_child_read, sample.get("read_bytes", 0))
            time.sleep(0.05)
    wall = time.time() - t0
    with open(out_path, encoding="utf-8", errors="replace") as f:
        stdout = f.read()
    with open(err_path, encoding="utf-8", errors="replace") as f:
        stderr = f.read()

    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    io_after = read_kv("/proc/self/io")

    # Counters must be read here: the cgroup is gone once systemd-run
    # returns, and `--wait` forbids `--remain-after-exit`.
    stat = read_kv(os.path.join(cg, "memory.stat")) if cg else {}
    result.update({
        "returncode": proc.returncode,
        "wall_s": round(wall, 3),
        "stdout": stdout,
        "stderr": stderr,
        "memory_peak": read_int(os.path.join(cg, "memory.peak")) if cg else None,
        "memory_current": read_int(os.path.join(cg, "memory.current")) if cg else None,
        "memory_max": read_int(os.path.join(cg, "memory.max")) if cg else None,
        "memory_swap_max": read_int(os.path.join(cg, "memory.swap.max")) if cg else None,
        "memory_swap_peak": read_int(os.path.join(cg, "memory.swap.peak")) if cg else None,
        "memory_events": read_kv(os.path.join(cg, "memory.events")) if cg else {},
        "pg": {k: v for k, v in stat.items()
               if k.startswith("pgscan") or k.startswith("pgsteal")},
        "read_bytes_child_sampled": peak_child_read,
        "read_bytes_rusage_children": ru.ru_inblock * 512,
        "read_bytes_self_delta": (io_after.get("read_bytes", 0)
                                  - io_before.get("read_bytes", 0)),
    })

    m = TIMING_RE.search(stderr)
    if m:
        result["prefill_tokens"] = int(m.group(1))
        result["prefill_s"] = float(m.group(2))
        result["prefill_tok_s"] = float(m.group(3))
        result["decode_tokens"] = int(m.group(4))
        result["decode_s"] = float(m.group(5))
        result["decode_tok_s"] = float(m.group(6))
    m = LOAD_RE.search(stderr)
    if m:
        result["load_s"] = float(m.group(1))

    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


# ---------------------------------------------------------------------------
# outer: orchestration
# ---------------------------------------------------------------------------


# cgroup v2 `memory.swap.peak` (and `memory.peak`) first appear in Linux
# 6.5. `classify` treats an unreadable counter as DIRTY, so below 6.5 this
# harness cannot produce a CLEAN verdict on any run, ever.
SWAP_PEAK_MIN_KERNEL = (6, 5)


def kernel_version() -> tuple[int, int] | None:
    """`(major, minor)` from `uname -r`, or None if it does not parse."""
    m = re.match(r"(\d+)\.(\d+)", os.uname().release)
    return (int(m.group(1)), int(m.group(2))) if m else None


def preflight(args) -> None:
    # Checked here rather than left to `classify`, because it is the one
    # DIRTY cause with no remedy: `memory.swap.peak` does not exist before
    # Linux 6.5, `read_int` returns None for it, unknown-is-DIRTY fires, and
    # every run on Debian 12 (6.1), Ubuntu 22.04 (5.15) or RHEL 9 (5.14)
    # comes back dirty no matter how clean it actually was. A harness that
    # cannot pass is a broken harness, not a failing measurement, so it is
    # exit 2 up front instead of exit 1 forever.
    kernel = kernel_version()
    if kernel is None:
        fail(f"cannot parse a kernel version out of {os.uname().release!r}; "
             f"this harness needs Linux >= "
             f"{SWAP_PEAK_MIN_KERNEL[0]}.{SWAP_PEAK_MIN_KERNEL[1]} for the "
             f"cgroup v2 `memory.swap.peak` counter its hygiene verdict "
             f"depends on")
    if kernel < SWAP_PEAK_MIN_KERNEL:
        fail(f"Linux {kernel[0]}.{kernel[1]} is too old: cgroup v2 "
             f"`memory.swap.peak` first appears in "
             f"{SWAP_PEAK_MIN_KERNEL[0]}.{SWAP_PEAK_MIN_KERNEL[1]}, this "
             f"harness classifies an unreadable counter as DIRTY, and that "
             f"counter is unreadable on every run here — so every run would "
             f"be DIRTY with no remedy. Debian 12 (6.1), Ubuntu 22.04 (5.15) "
             f"and RHEL 9 (5.14) are all below the floor. Take the "
             f"measurement on a >= "
             f"{SWAP_PEAK_MIN_KERNEL[0]}.{SWAP_PEAK_MIN_KERNEL[1]} kernel.")
    found = subprocess.run(["pgrep", "-a", "ramvamp"],
                           capture_output=True, text=True, check=False)
    if found.returncode == 0 and found.stdout.strip():
        print(found.stdout.strip(), file=sys.stderr)
        fail("a ramvamp process is running. posix_fadvise cannot evict a "
             "file that any process has mmap'd, and it reports success "
             "anyway — refusing to produce a fake cold run.")
    if not os.path.isdir(CGROUP_ROOT) or not os.path.exists(
            os.path.join(CGROUP_ROOT, "cgroup.controllers")):
        fail("cgroup v2 is not mounted at /sys/fs/cgroup")
    if subprocess.run(["systemd-run", "--user", "--version"],
                      capture_output=True, check=False).returncode != 0:
        fail("systemd-run --user is not available")
    if args.ramvamp and not os.path.isfile(args.ramvamp):
        fail(f"no ramvamp binary at {args.ramvamp}")


def pg_total(pg: dict[str, int], prefix: str) -> int:
    """`memory.stat` carries both `pgsteal` and its `pgsteal_*` breakdown;
    summing every matching key would double-count."""
    if prefix in pg:
        return pg[prefix]
    return sum(v for k, v in pg.items() if k.startswith(prefix + "_"))


def pg_split(pg: dict[str, int], prefix: str) -> tuple[int, int]:
    """Split `pgscan`/`pgsteal` into (pressure, khugepaged).

    khugepaged is subtracted from the total rather than the pressure
    reclaimers being summed, so a reclaimer this script has never heard of
    counts as pressure instead of vanishing. A `memory.stat` carrying no
    `<prefix>_*` breakdown cannot be split, so all of it counts as
    pressure — unknown is DIRTY, not clean.

    See item 4 in the module docstring for why khugepaged is not pressure.
    """
    total = pg_total(pg, prefix)
    if not any(k.startswith(prefix + "_") for k in pg):
        return total, 0
    khugepaged = pg.get(prefix + "_khugepaged", 0)
    return total - khugepaged, khugepaged


# Problem severities. HARD invalidates the measurement; SOFT is reported
# but survivable. The severity travels with the problem as a field, so
# rewording a message can never change a verdict — the previous version
# decided this by `str.startswith("pgscan ")`.
HARD, SOFT = "hard", "soft"


def classify(run: dict, want_max: int) -> tuple[str, list[dict]]:
    """CLEAN or DIRTY, with the reasons. Reclaim beats every other signal.

    Every counter below has three states, not two: confirmed good,
    confirmed bad, and *unknown*. Unknown is DIRTY. `read_int` and
    `read_kv` return `None`/`{}` for an unreadable file, and an unreadable
    file is the normal outcome when the memory controller is not delegated
    or the cgroup path does not resolve — i.e. exactly when the run was not
    confined at all. Exempting `None` would make an unconfined run the
    cleanest run this harness can report.
    """
    problems: list[dict] = []

    def note(severity: str, message: str) -> None:
        problems.append({"severity": severity, "message": message})

    if not run.get("cgroup"):
        note(HARD, "the inner wrapper could not resolve its own cgroup from "
                   "/proc/self/cgroup — the run was not measurably confined "
                   "and no cgroup counter below could be read")

    pg = run.get("pg") or {}
    if not pg:
        note(HARD, "memory.stat exposed no pgscan/pgsteal counters — reclaim "
                   "could not be measured, so this run is not verified "
                   "clean (this is the unknown state, not a clean one)")
        steal = scan = 0
    else:
        steal, steal_thp = pg_split(pg, "pgsteal")
        scan, scan_thp = pg_split(pg, "pgscan")
        if steal:
            note(HARD,
                 f"pgsteal {steal} pages ({mib(steal * PAGE)}) reclaimed "
                 f"under pressure — the working set did not fit, the timing "
                 f"is not a clean {want_max // 2**30} GB measurement")
        elif scan:
            # Pressure the kernel survived: worth reporting, but it did not
            # cost the run any resident page, so the timing still stands.
            note(SOFT, f"pgscan {scan} pages with no steal (pressure, no loss)")
        if steal_thp or scan_thp:
            # Not pressure, and not a defect. Reported so a run is never
            # silently credited as clean when something did touch its pages,
            # and so the exclusion stays auditable from the output alone.
            note(SOFT,
                 f"khugepaged collapsed hugepages during the run "
                 f"(pgsteal_khugepaged {steal_thp}, pgscan_khugepaged "
                 f"{scan_thp}, {mib(steal_thp * PAGE)}) — the THP daemon's "
                 f"own bookkeeping, not memory pressure, and excluded from "
                 f"the verdict")

    events = run.get("memory_events")
    if not events:
        note(HARD, "memory.events was unreadable or empty — OOM and limit "
                   "hits could not be ruled out")
    else:
        for key in ("max", "oom", "oom_kill", "oom_group_kill", "high"):
            if events.get(key):
                note(HARD, f"memory.events {key}={events[key]}")

    if run.get("memory_max") is None:
        note(HARD, f"memory.max is unreadable, so a {want_max}-byte limit "
                   f"could not be confirmed — the memory controller is "
                   f"probably not delegated to the user slice, which means "
                   f"the run was unconfined")
    elif run.get("memory_max") != want_max:
        note(HARD,
             f"memory.max is {run.get('memory_max')}, expected {want_max} — "
             f"the memory controller may not be delegated to the user slice")

    if run.get("memory_swap_max") is None:
        note(HARD, "memory.swap.max is unreadable, so swap could not be "
                   "confirmed off (zram counts as swap)")
    elif run.get("memory_swap_max") != 0:
        note(HARD, f"memory.swap.max is {run.get('memory_swap_max')}, "
                   f"expected 0 (zram counts as swap)")

    if run.get("memory_swap_peak") is None:
        note(HARD, "memory.swap.peak is unreadable, so it cannot be shown "
                   "that the run never swapped")
    elif run["memory_swap_peak"]:
        note(HARD, f"swapped {mib(run['memory_swap_peak'])}")

    read = max(run.get("read_bytes_child_sampled", 0),
               run.get("read_bytes_rusage_children", 0))
    if read == 0:
        note(HARD, "read_bytes delta is 0 — nothing came from the block "
                   "layer, so the page cache was not actually cold")

    if run.get("returncode"):
        note(HARD, f"ramvamp exited {run['returncode']}")
    if run.get("systemd_run_returncode"):
        note(HARD, f"systemd-run exited {run['systemd_run_returncode']} — the "
                   f"confined run did not complete normally, so whatever "
                   f"result file was classified is not this run's")

    hard = [p for p in problems if p["severity"] == HARD]
    return ("CLEAN" if not hard else "DIRTY"), problems


def one_run(args, index: int, label: str) -> dict:
    print(f"\n=== run {index} ({label}) ===")
    ev = evict_verified(model_files(args.rvmp))

    result_file = os.path.join(
        args.workdir, f"run{index:02d}.json")
    command_file = os.path.join(args.workdir, f"run{index:02d}.cmd.json")
    # Remove any file from an earlier invocation *before* launching. The
    # `--workdir` default is stable across runs, so a systemd-run that fails
    # to start would otherwise leave the previous run's JSON in place and
    # the harness would happily classify it as this run's result.
    for stale in (result_file, result_file + ".stdout", result_file + ".stderr",
                  command_file):
        try:
            os.unlink(stale)
        except FileNotFoundError:
            pass
        except OSError as e:
            fail(f"cannot remove the stale result file {stale}: {e}")
    unit = f"ramvamp-cold-{os.getpid()}-{index}"
    # The argv goes in a file, not on systemd-run's command line, which
    # mangles `${NAME}` and `$$` (module docstring, item 8). The inner
    # wrapper reads it before it samples `/proc/self/io`, so this read
    # cannot show up in the run's own read_bytes.
    write_command_file(command_file, args.workload)
    inner_cmd = [
        sys.executable, os.path.abspath(__file__),
        "--inner", "--result-file", result_file,
        "--command-file", command_file,
    ]
    cmd = [
        "systemd-run", "--user", "--wait", "-q", "--collect",
        f"--unit={unit}",
        "-p", f"MemoryMax={args.memory_max}",
        "-p", "MemorySwapMax=0",
        "-p", "MemoryAccounting=yes",
        "-p", f"WorkingDirectory={os.getcwd()}",
        "--",
    ] + inner_cmd
    print(f"  + {' '.join(cmd[:12])} ... {display_workload(args.workload)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    outer_wall = time.time() - t0
    if not os.path.isfile(result_file):
        print(proc.stdout, proc.stderr, file=sys.stderr)
        fail(f"the inner run produced no result file (systemd-run exited "
             f"{proc.returncode}); MemoryMax may have OOM-killed it")
    with open(result_file, encoding="utf-8") as f:
        run = json.load(f)
    run["evict"] = ev
    run["outer_wall_s"] = round(outer_wall, 3)
    run["systemd_run_returncode"] = proc.returncode
    run["label"] = label
    # A benchmark whose prompt cannot be identified from the log is not
    # reproducible, and a 4000-token prompt is not identifiable by eye.
    run["prompt"] = args.prompt_meta

    want_max = parse_size(args.memory_max)
    verdict, problems = classify(run, want_max)
    run["hygiene"] = verdict
    run["hygiene_problems"] = problems

    read = max(run.get("read_bytes_child_sampled", 0),
               run.get("read_bytes_rusage_children", 0))
    steal = pg_total(run.get("pg", {}), "pgsteal")
    scan = pg_total(run.get("pg", {}), "pgscan")
    print(f"  wall {run.get('wall_s')}s  load {run.get('load_s')}s  "
          f"prefill {run.get('prefill_tok_s')} tok/s  "
          f"decode {run.get('decode_tok_s')} tok/s")
    print(f"  MemoryPeak {mib(run.get('memory_peak') or 0)}  "
          f"(max {mib(run.get('memory_max') or 0)}, swap.max "
          f"{run.get('memory_swap_max')})")
    print(f"  memory.events {run.get('memory_events')}")
    print(f"  pgscan {scan}  pgsteal {steal}"
          f"{'  <-- RECLAIM, measurement not clean' if steal else ''}")
    print(f"  read_bytes {read} ({mib(read)}) "
          f"[sampled {run.get('read_bytes_child_sampled')}, rusage "
          f"{run.get('read_bytes_rusage_children')}]")
    print(f"  hygiene: {verdict}")
    for problem in problems:
        print(f"    - [{problem['severity']}] {problem['message']}")
    if run.get("stdout"):
        preview = run["stdout"].strip().replace("\n", " ")[:160]
        print(f"  output: {preview!r}")
    return run


def parse_size(text: str) -> int:
    units = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    text = text.strip()
    if text[-1].upper() in units:
        return int(float(text[:-1]) * units[text[-1].upper()])
    return int(text)


def median_of(runs: list[dict], key: str):
    values = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    return statistics.median(values) if values else None


def reverdict(paths: list[str]) -> int:
    """Re-classify recorded summaries against the current rules.

    Prints the old verdict beside the new one so a rule change is visible as
    a diff rather than as a number that quietly improved. Exits 1 if any
    run is still DIRTY, 0 if all are clean, 2 if a file could not be read —
    the same convention as a live run.

    "Could not read" covers a file that parses as JSON but is not a summary.
    Summaries share `--workdir` with the per-run `runNN.json` records and the
    `runNN.cmd.json` argv files, so `--reverdict scratch/cold-bench/*.json` is
    the natural thing to type and it hands this function all three. A
    `cmd.json` is a JSON *list*, and asking a list for `.get("runs")` used to
    raise an uncaught AttributeError — a traceback, from a script whose whole
    job is to be the thing you trust about a measurement. Rejected by shape,
    named, and counted as unreadable rather than silently verdicted.
    """
    worst = 0
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, ValueError) as err:
            print(f"{path}: could not read ({err})", file=sys.stderr)
            worst = max(worst, 2)
            continue

        if not isinstance(summary, dict) or not isinstance(
                summary.get("runs"), list):
            what = ("a JSON list, i.e. a runNN.cmd.json argv file"
                    if isinstance(summary, list) else
                    f"a JSON {type(summary).__name__} with no `runs` list, "
                    f"i.e. not a summary — a runNN.json holds one run, not a "
                    f"run list")
            print(f"{path}: could not read (this is {what}). --reverdict "
                  f"takes the summaries written by --json, which share a "
                  f"workdir with the per-run files; narrow the glob.",
                  file=sys.stderr)
            worst = max(worst, 2)
            continue

        runs = summary.get("runs") or []
        scored = [r for r in runs if not str(r.get("label", "")).startswith("warmup")]
        if not scored:
            scored = runs
        want_max = parse_size(summary.get("memory_max", "3G"))

        redone, dirty = [], []
        for run in scored:
            verdict, problems = classify(run, want_max)
            redone.append((run, verdict, problems))
            if verdict == "DIRTY":
                dirty.append(run)

        was = summary.get("hygiene", "?")
        now = "PASS" if not dirty else "DIRTY"
        print(f"\n{path}")
        print(f"  recorded: {was}    re-classified: {now}")
        for run, verdict, problems in redone:
            label = run.get("label", "?")
            print(f"    {label:<10} {run.get('hygiene', '?'):<6} -> {verdict}")
            for p in problems:
                print(f"      [{p['severity']}] {p['message']}")
        if now == "DIRTY":
            worst = max(worst, 1)
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inner", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    parser.add_argument("--command", nargs=argparse.REMAINDER,
                        help=argparse.SUPPRESS)
    parser.add_argument("--command-file", help=argparse.SUPPRESS)
    parser.add_argument("--rvmp", default="models/qwen3.rvmp",
                        help="installed .rvmp model dir (evicted before each run)")
    parser.add_argument("--ramvamp", default="target/release/ramvamp",
                        help="ramvamp binary to benchmark")
    # Mutually exclusive: two prompts is an ambiguous measurement, and
    # argparse's own conflict error exits 2, the same "could not measure"
    # code `fail()` uses.
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default="The capital of France is",
                              help="benchmark prompt, given literally")
    prompt_group.add_argument(
        "--prompt-file", metavar="PATH",
        help="read the benchmark prompt from PATH as UTF-8, instead of "
             "--prompt. Preferred for the long prefill prompts (512-4000 "
             "tokens) a ~17 KB command line cannot carry readably. Exactly "
             "one trailing newline is stripped (a trailing CRLF counts as "
             "one), because it would otherwise be an extra prefill token "
             "and shift the tok/s being measured; end the file with two "
             "newlines if a trailing blank line is intended. Nothing else "
             "is rewritten — interior CRLFs are preserved as authored. The "
             "path, byte length and SHA-256 of both the file and the "
             "resulting prompt are recorded in the results JSON.")
    parser.add_argument("--max-new", type=int, default=8,
                        help="tokens to generate per run")
    parser.add_argument("--sampled", action="store_true",
                        help="use the checkpoint's sampler instead of greedy "
                             "decoding (default is greedy, so repeats do the "
                             "same work)")
    parser.add_argument("--cache-bytes", metavar="BYTES", default=None,
                        help="forward `--cache-bytes BYTES` to ramvamp, the "
                             "total expert-cache budget the slot count per "
                             "layer is derived from. Passed through verbatim: "
                             "ramvamp's own `parse_bytes` is the authority on "
                             "the grammar (a plain byte count or a K/M/G/T "
                             "suffix, optionally spelled KiB/MiB/...), and it "
                             "accepts spellings this script's `parse_size` "
                             "does not, so re-parsing it here could only "
                             "reject a value ramvamp would have taken. Omitted "
                             "entirely by default, so the argv, and therefore "
                             "every number, stays comparable with runs taken "
                             "before this flag existed. A slot costs the sum "
                             "of every layer's page-aligned stride, which is "
                             "130.781 MiB for the shipped Qwen3-30B-A3B "
                             "layout, so 1440M (ramvamp's default) buys 11 "
                             "slots/layer, 1570M buys 12 and 1701M buys 13 — "
                             "but each of those clears its threshold by under "
                             "1 MiB, and a repack with a different stride "
                             "moves the thresholds. The slot count a budget "
                             "actually bought is therefore never assumed here: "
                             "read it off the `model loaded in` line of the "
                             "stderr this harness now records.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="scored runs (after the warmups)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="discarded leading runs; run 1 is contaminated "
                             "by btrfs metadata warm-up")
    parser.add_argument("--memory-max", default="3G",
                        help="cgroup MemoryMax (docs rule: 3G)")
    parser.add_argument("--workdir", default="scratch/cold-bench",
                        help="where per-run JSON lands")
    parser.add_argument("--json",
                        help="write the full summary here. Each run record "
                             "keeps the child's stderr verbatim — the phase "
                             "splits and per-phase cache stats — because the "
                             "`<workdir>/runNN.json.stderr` sidecars are "
                             "overwritten by the next invocation, and a path "
                             "chosen per experiment is not. Pass one. The "
                             "default, `<workdir>/summary.json`, is a fixed "
                             "path, so it is overwritten by the next "
                             "invocation exactly like the sidecars it exists "
                             "to outlive: the retention argument only holds "
                             "for a path you chose. Defaulting onto an "
                             "existing summary is warned about up front, not "
                             "refused — see the comment at the call site for "
                             "why refusing is worse.")
    parser.add_argument("--reverdict", metavar="SUMMARY.json", nargs="+",
                        help="re-classify already-recorded runs against the "
                             "current rules and print the verdicts, without "
                             "running anything. Every counter the verdict "
                             "depends on is already in the summary, so a rule "
                             "correction does not cost another cold run.")
    args = parser.parse_args()

    if args.reverdict:
        return reverdict(args.reverdict)
    sys.stdout.reconfigure(line_buffering=True)

    if args.inner:
        if not args.result_file:
            fail("--inner needs --result-file")
        if bool(args.command) == bool(args.command_file):
            fail("--inner needs exactly one of --command or --command-file")
        if args.command_file:
            args.command = load_command_file(args.command_file)
        return inner(args)

    if args.repeats < 1:
        fail(f"--repeats {args.repeats} scores nothing; a run that measures "
             f"zero runs cannot pass a hygiene gate. Use --repeats >= 1.")
    if args.warmup < 0:
        fail(f"--warmup {args.warmup} is negative")

    # Resolve the prompt once, up front — before preflight, before the model
    # directory is touched and long before the first eviction — so a typo'd
    # path costs nothing and reports itself, rather than surfacing as a
    # confusing ramvamp failure three cold runs in. `read_prompt_file` exits
    # 2 via `fail()`: the harness could not take the measurement.
    file_meta = {"file_sha256": None, "file_bytes": None}
    if args.prompt_file:
        args.prompt_file = os.path.abspath(args.prompt_file)
        args.prompt, file_meta = read_prompt_file(args.prompt_file)
    args.prompt_meta = {
        "source": "file" if args.prompt_file else "argument",
        "file": args.prompt_file,
        "sha256": sha256_text(args.prompt),
        "bytes": len(args.prompt.encode("utf-8")),
        **file_meta,
    }

    args.rvmp = os.path.abspath(args.rvmp)
    args.ramvamp = os.path.abspath(args.ramvamp)
    os.makedirs(args.workdir, exist_ok=True)
    args.workdir = os.path.abspath(args.workdir)
    # Resolved here rather than at the end, so the one action that can
    # silently destroy an earlier experiment's record is reported *before*
    # the cold runs start instead of after they finish. `--json` is chosen
    # per invocation; its default is not, and a `<workdir>/summary.json`
    # from an earlier session goes with no trace — the same loss, one level
    # up, that cost phase 7 its stderr sidecars and motivated keeping stderr
    # in the summary at all.
    #
    # Warned, not refused. The summary is written *after* the measurement, so
    # refusing at that point would discard runs that already cost their cold
    # time; refusing up front would make a bare `cold_bench.py` fail on a file
    # from weeks ago, and a gate script that will not run by default gets
    # worked around rather than heeded. A named path on stderr, before
    # anything is evicted, is enough for the user to Ctrl-C and pass --json.
    args.summary_json = args.json or os.path.join(args.workdir, "summary.json")
    if not args.json and os.path.exists(args.summary_json):
        print(f"warning: no --json given, so this run will overwrite the "
              f"existing {args.summary_json} when it finishes. That file is "
              f"the only surviving copy of its runs' stderr — the phase "
              f"splits and cache stats. Pass --json <path> to keep both.",
              file=sys.stderr)
    args.workload = [
        args.ramvamp,
        "generate",
        "--model", args.rvmp,
        "--prompt", args.prompt,
        "--max-new", str(args.max_new),
        "--skip-hashes",
        # A config file in the operator's home would otherwise set dials that
        # no summary records and no experiment entry mentions, so the same
        # command would measure different things on two machines. This removes
        # that layer; env vars and the explicit flags below still apply, and
        # the `model loaded in ...` line still names the budget and slot count
        # the run actually got.
        "--no-config",
    ]
    # Appended only when asked for. An always-present `--cache-bytes 1440M`
    # would be the same value ramvamp defaults to, but it would change the
    # argv recorded in every summary, and "the argv is identical" is how a
    # reader establishes that two runs measured the same workload.
    if args.cache_bytes is not None:
        args.workload += ["--cache-bytes", args.cache_bytes]
    if not args.sampled:
        args.workload.append("--greedy")
    # Recorded like the prompt is, and for the same reason: a run whose dial
    # is not in its own summary is a run nobody can interpret six weeks later.
    # `source` distinguishes "the harness set it" from "ramvamp's built-in
    # default applied", which a bare null cannot. The default is deliberately
    # not named here — this script cannot see ramvamp's default and guessing
    # it would eventually be a lie; the budget and the slot count it bought
    # are both on the `model loaded in` line of the recorded stderr.
    args.cache_meta = {
        "source": ("argument" if args.cache_bytes is not None
                   else "ramvamp-default"),
        "value": args.cache_bytes,
    }

    preflight(args)
    print(f"cold_bench: {args.warmup} warmup + {args.repeats} scored runs, "
          f"MemoryMax={args.memory_max}, MemorySwapMax=0")
    print(f"workload: {display_workload(args.workload)}")
    print(f"prompt: {args.prompt_meta['source']} "
          f"{args.prompt_meta['file'] or '(literal)'} "
          f"{args.prompt_meta['bytes']} bytes "
          f"sha256:{args.prompt_meta['sha256']}")

    runs = []
    for i in range(args.warmup + args.repeats):
        label = "warmup, discarded" if i < args.warmup else "scored"
        runs.append(one_run(args, i, label))

    scored = [r for r in runs if r["label"] == "scored"]
    if not scored:
        fail(f"no scored runs out of {len(runs)} — nothing was measured, so "
             f"there is no hygiene verdict to give")
    dirty = [r for r in scored if r["hygiene"] != "CLEAN"]

    print("\n=== summary ===")
    print(f"scored runs: {len(scored)}  clean: {len(scored) - len(dirty)}  "
          f"dirty: {len(dirty)}")
    for key, unit in (("wall_s", "s"), ("load_s", "s"),
                      ("prefill_tok_s", "tok/s"), ("decode_tok_s", "tok/s")):
        med = median_of(scored, key)
        if med is not None:
            print(f"  median {key:<14} {med:>10.3f} {unit}")
    reads = [max(r.get("read_bytes_child_sampled", 0),
                 r.get("read_bytes_rusage_children", 0)) for r in scored]
    if reads:
        print(f"  median read_bytes   {statistics.median(reads):>12.0f} "
              f"({mib(statistics.median(reads))})")
    peaks = [r.get("memory_peak") or 0 for r in scored]
    if peaks:
        print(f"  median MemoryPeak   {statistics.median(peaks):>12.0f} "
              f"({mib(statistics.median(peaks))})")

    verdict = "PASS" if not dirty else "DIRTY"
    print(f"\nmeasurement hygiene: {verdict}"
          + ("" if not dirty else
             f" — {len(dirty)}/{len(scored)} scored runs hit reclaim, hit a "
             f"cgroup limit, or left a counter the verdict depends on "
             f"unreadable (unknown is DIRTY, not clean); see the [hard] "
             f"lines above for which. Do not publish these numbers"))

    summary = {
        "workload": args.workload,
        "prompt": args.prompt_meta,
        "cache_bytes": args.cache_meta,
        "memory_max": args.memory_max,
        "warmup": args.warmup, "repeats": args.repeats,
        "hygiene": verdict,
        "median": {k: median_of(scored, k)
                   for k in ("wall_s", "load_s", "prefill_tok_s",
                             "decode_tok_s", "prefill_s", "decode_s")},
        # `stderr` is kept whole; only `stdout` is dropped. The child's stderr
        # carries the `model loaded in` line, the `prefill:`/`decode:` timing
        # line, the `experts:` cache block and the `prefill split (sweep)` /
        # `decode split (forward_token)` phase splits — the most detailed
        # diagnostic this harness produces, and the only record of *where* the
        # time went rather than how much of it there was.
        #
        # `inner()` already spools it to `<result-file>.stderr`, but that path
        # is `<workdir>/runNN.json.stderr` with `NN` restarting at 0 every
        # invocation and `--workdir` defaulting to a fixed directory, so each
        # session silently overwrites the previous session's sidecars. Phase 7
        # lost every phase split but one that way, and the single most
        # important finding of that phase nearly went unread. A `--json` path
        # is chosen per invocation, so this copy survives where the sidecar
        # does not.
        #
        # Unfiltered, and uncapped, on purpose: a filter that keeps only the
        # blocks known today drops the one a future runtime adds, and a
        # truncation drops either the load line (head) or the phase splits
        # (tail) — both of which are the point. It is also cheap; measured
        # 607-1111 bytes per run, against summaries that are already 7-38 KB.
        # `stdout` is the generated text, previewed in the run log and
        # reproducible from the recorded argv, so it stays out.
        "runs": [{k: v for k, v in r.items() if k != "stdout"}
                 for r in runs],
    }
    out = args.summary_json
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
