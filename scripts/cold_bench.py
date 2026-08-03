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
   reclaimed. So the verdict keys on `pgsteal` from `memory.stat`: any
   nonzero `pgsteal*` means the run hit reclaim, the working set did not
   fit, and the timing is not a clean 3 GB measurement.

5. Bytes actually fetched from the block layer are reported. `read_bytes`
   in `/proc/<pid>/io` is per-task and is *not* inherited by the parent
   on exit, so the inner wrapper samples the child's `/proc/<pid>/io`
   while it runs and also takes `getrusage(RUSAGE_CHILDREN).ru_inblock`
   after reaping it — the same kernel counter, in 512-byte units, exact.
   A delta of ~0 on a supposedly cold run means eviction failed.

6. Run 1 is contaminated by btrfs metadata warm-up, so `--repeats N`
   discards `--warmup` runs (default 1) and reports the median of the
   rest.

Usage:

  scripts/cold_bench.py --ramvamp target/release/ramvamp --max-new 8
  scripts/cold_bench.py --ramvamp target/release/ramvamp \\
      --max-new 64 --repeats 5 --json scratch/cold.json

Exit 0 = every scored run was CLEAN. Exit 1 = at least one scored run was
DIRTY (numbers still printed, clearly marked). Exit 2 = harness error
(eviction failed, no cgroup, a ramvamp process was running, ...).

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
page cache, the cgroup pins at its 3 GiB ceiling, and ~4.9 GiB is
reclaimed *inside a 13-token run*. Nothing OOMs (oom=0, oom_kill=0) and
`memory.events max` is nonzero here, but note it would not have to be:
the `pgsteal` check is the one that cannot be fooled. So no phase-4
number from this harness is publishable under the docs rule, and phase 5
(O_DIRECT, which keeps expert bytes out of the page cache entirely)
should be the thing that first turns this verdict CLEAN. That transition
is itself a result worth recording in `docs/experiments/README.md`.

Python stdlib only. Linux + cgroup v2 + systemd --user only, by design.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
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


def preflight(args) -> None:
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


def classify(run: dict, want_max: int) -> tuple[str, list[str]]:
    """CLEAN or DIRTY, with the reasons. Reclaim beats every other signal."""
    problems = []
    steal = pg_total(run.get("pg", {}), "pgsteal")
    scan = pg_total(run.get("pg", {}), "pgscan")
    if steal:
        problems.append(
            f"pgsteal {steal} pages ({mib(steal * PAGE)}) reclaimed under "
            f"pressure — the working set did not fit, the timing is not a "
            f"clean {want_max // 2**30} GB measurement")
    elif scan:
        problems.append(f"pgscan {scan} pages with no steal (pressure, no loss)")
    events = run.get("memory_events") or {}
    for key in ("max", "oom", "oom_kill", "oom_group_kill", "high"):
        if events.get(key):
            problems.append(f"memory.events {key}={events[key]}")
    if run.get("memory_max") not in (want_max, None):
        problems.append(
            f"memory.max is {run.get('memory_max')}, expected {want_max} — "
            f"the memory controller may not be delegated to the user slice")
    if run.get("memory_swap_max") not in (0, None):
        problems.append(f"memory.swap.max is {run.get('memory_swap_max')}, "
                        f"expected 0 (zram counts as swap)")
    if run.get("memory_swap_peak"):
        problems.append(f"swapped {mib(run['memory_swap_peak'])}")
    read = max(run.get("read_bytes_child_sampled", 0),
               run.get("read_bytes_rusage_children", 0))
    if read == 0:
        problems.append("read_bytes delta is 0 — nothing came from the block "
                        "layer, so the page cache was not actually cold")
    if run.get("returncode"):
        problems.append(f"ramvamp exited {run['returncode']}")
    # pgscan without pgsteal is pressure the kernel survived; it is worth
    # reporting but does not by itself invalidate the timing.
    hard = [p for p in problems if not p.startswith("pgscan ")]
    return ("CLEAN" if not hard else "DIRTY"), problems


def one_run(args, index: int, label: str) -> dict:
    print(f"\n=== run {index} ({label}) ===")
    ev = evict_verified(model_files(args.rvmp))

    result_file = os.path.join(
        args.workdir, f"run{index:02d}.json")
    unit = f"ramvamp-cold-{os.getpid()}-{index}"
    inner_cmd = [
        sys.executable, os.path.abspath(__file__),
        "--inner", "--result-file", result_file,
        "--command",
    ] + args.workload
    cmd = [
        "systemd-run", "--user", "--wait", "-q", "--collect",
        f"--unit={unit}",
        "-p", f"MemoryMax={args.memory_max}",
        "-p", "MemorySwapMax=0",
        "-p", "MemoryAccounting=yes",
        "-p", f"WorkingDirectory={os.getcwd()}",
        "--",
    ] + inner_cmd
    print(f"  + {' '.join(cmd[:12])} ... {' '.join(args.workload)}")
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
        print(f"    - {problem}")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inner", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    parser.add_argument("--command", nargs=argparse.REMAINDER,
                        help=argparse.SUPPRESS)
    parser.add_argument("--rvmp", default="models/qwen3.rvmp",
                        help="installed .rvmp model dir (evicted before each run)")
    parser.add_argument("--ramvamp", default="target/release/ramvamp",
                        help="ramvamp binary to benchmark")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="benchmark prompt")
    parser.add_argument("--max-new", type=int, default=8,
                        help="tokens to generate per run")
    parser.add_argument("--sampled", action="store_true",
                        help="use the checkpoint's sampler instead of greedy "
                             "decoding (default is greedy, so repeats do the "
                             "same work)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="scored runs (after the warmups)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="discarded leading runs; run 1 is contaminated "
                             "by btrfs metadata warm-up")
    parser.add_argument("--memory-max", default="3G",
                        help="cgroup MemoryMax (docs rule: 3G)")
    parser.add_argument("--workdir", default="scratch/cold-bench",
                        help="where per-run JSON lands")
    parser.add_argument("--json", help="write the full summary here")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if args.inner:
        if not args.result_file or not args.command:
            fail("--inner needs --result-file and --command")
        return inner(args)

    args.rvmp = os.path.abspath(args.rvmp)
    args.ramvamp = os.path.abspath(args.ramvamp)
    os.makedirs(args.workdir, exist_ok=True)
    args.workdir = os.path.abspath(args.workdir)
    args.workload = [
        args.ramvamp,
        "generate",
        "--model", args.rvmp,
        "--prompt", args.prompt,
        "--max-new", str(args.max_new),
        "--skip-hashes",
    ]
    if not args.sampled:
        args.workload.append("--greedy")

    preflight(args)
    print(f"cold_bench: {args.warmup} warmup + {args.repeats} scored runs, "
          f"MemoryMax={args.memory_max}, MemorySwapMax=0")
    print(f"workload: {' '.join(args.workload)}")

    runs = []
    for i in range(args.warmup + args.repeats):
        label = "warmup, discarded" if i < args.warmup else "scored"
        runs.append(one_run(args, i, label))

    scored = [r for r in runs if r["label"] == "scored"]
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
             f" — {len(dirty)}/{len(scored)} scored runs hit reclaim or a "
             f"cgroup limit; do not publish these numbers"))

    summary = {
        "workload": args.workload,
        "memory_max": args.memory_max,
        "warmup": args.warmup, "repeats": args.repeats,
        "hygiene": verdict,
        "median": {k: median_of(scored, k)
                   for k in ("wall_s", "load_s", "prefill_tok_s",
                             "decode_tok_s", "prefill_s", "decode_s")},
        "runs": [{k: v for k, v in r.items() if k not in ("stdout", "stderr")}
                 for r in runs],
    }
    out = args.json or os.path.join(args.workdir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
