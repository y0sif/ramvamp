#!/usr/bin/env python3
"""Reproducible O_DIRECT bandwidth probe over the *installed* expert files.

`docs/experiments/README.md` lists re-measuring EXP-008 as the highest-value
item on the backlog, and EXP-008 cannot be re-run: its harness was never
committed. This file is that harness, written so the entry it feeds is
reproducible, self-describing, and rule-2 compliant.

## The question

Phase 6 wants to know whether reading a layer's experts **front-to-back in
large blocks** beats reading **individual experts at the 2.918 MiB stride**,
on these actual files, and by how much. EXP-008 answered a version of that
question with two probe series that disagree with each other by 17% at the
same queue depth, and EXP-013 then measured ~1.97 GB/s at the expert stride
under the real access pattern against EXP-008's 1.211-1.390 GB/s. If EXP-013
is right, EXP-008's "+51% at 16 MiB" premise collapses to roughly +9%.

EXP-008 also conflated two different variables. "Large blocks" and
"sequential" are not the same thing, and the win could come from either.
This probe separates them by measuring, at *matched block size and matched
bytes*, a sequential front-to-back pass and a random permutation of the same
blocks. The headline is reported as a three-way decomposition:

    granularity   = random K=8  / random K=1     (bigger blocks, same disorder)
    sequentiality = sequential K=8 / random K=8  (same blocks, in order)
    combined      = sequential K=8 / random K=1  (what phase 6 would buy)

## Why the files matter more than the sweep

The installed layer files are physically fragmented, and EXP-008 could not
have known it. `filefrag` on `experts/layer_00.bin` reports **398 extents for
a 373.5 MiB file, none of them physically adjacent to its successor** — mean
extent 0.94 MiB, and the extent sizes are exactly the gate/up (884,736 B) and
down (1,290,240 B) slab sizes. The cause is in the installer:
`crates/repack/src/install/executor.rs` preallocates with `set_len` (on btrfs
that is sparse namespace only, not blocks) and then demuxes a sequential
source window into scattered destination pwrites, so every slab lands as its
own CoW extent. A "sequential" 24 MiB read is therefore ~25 physically
scattered ~1 MiB chunks.

So this probe reads **the real files**, never a freshly written contiguous
scratch file, and records `filefrag` extent statistics for every file it
touches — extent count, mean and median extent size, and how many extents are
physically adjacent to their successor. That difference may be the whole
disagreement between EXP-008 and EXP-013, and a bandwidth number without the
extent map next to it is not interpretable.

`docs/benchmark-machine.md` also records that per-file fragmentation variance
exceeds run-to-run variance (`layer_00` 1,237 MB/s vs `layer_20` 1,812 MB/s).
This probe therefore runs a **fixed, recorded** file set spanning both stride
classes and **always reports per file**. Aggregates are printed only with the
per-file min and max beside them; an aggregate that hides a 1.46x spread is a
worse answer than no aggregate.

## What "queue depth" means here, and what it does not

Queue depth is emulated with **N OS threads each issuing blocking `preadv`**
against a shared work queue. It is **not io_uring**. `os.preadv` releases the
GIL, so the requests really do overlap in the kernel, and at a ~3 MiB block
the per-call Python overhead is a few microseconds against a ~2 ms read — but
a threaded-pread queue is not the runtime's submission path, and its numbers
must not be silently compared against `crates/core/src/io`'s io_uring results.
Every table this script prints, and every JSON record it writes, carries the
`queue = threaded-pread` label for exactly that reason. Using io_uring from
Python would need a C extension or a third-party dependency; the repo's
scripts are stdlib-only and staying that way is worth more than the label.

## Measurement hygiene (rule 2), verified rather than assumed

Mechanisms are duplicated from `scripts/cold_bench.py` on purpose — that file
is owned by another lane and importing across scripts would couple two gates
that should be able to fail independently. The reasoning behind each is
documented there and only summarised here:

1. A transient systemd **service** (`systemd-run --user --wait -q`), never a
   `--scope`: a scope is torn down before its counters can be read. The
   counters are therefore read from *inside* the cgroup by an inner wrapper
   (this same script, `--inner`) just before it exits.
2. `MemoryMax=3G`, `MemorySwapMax=0`, `MemoryAccounting=yes` (zram counts as
   swap).
3. `posix_fadvise(POSIX_FADV_DONTNEED)` on every probed file, then an
   **mmap + mincore proof** that each one is actually evicted. `fadvise`
   returns 0 and evicts nothing while any process holds the file mmap'd.
4. `pgsteal` from `memory.stat`, read inside the cgroup. Any nonzero value is
   DIRTY: `memory.events max` has been observed at 0 while 2.4 GiB was
   silently reclaimed.
5. Unknown is DIRTY, never CLEAN. An unreadable counter is the normal outcome
   of an *unconfined* run, so exempting it would make an unconfined run the
   cleanest result this harness can report.

## Two checks this probe adds that `cold_bench.py` does not need

**Was O_DIRECT actually honoured?** It can be silently downgraded to buffered
I/O with no error and a full byte count returned. btrfs falls back to
`filemap_read()` on a misaligned offset, length or **buffer address**, on
compressed extents, on DUP/RAID profiles, and when the destination pages have
not been faulted in. `statx(STATX_DIOALIGN)` is unimplemented on btrfs, so
alignment cannot be probed portably. This script mirrors
`crates/core/src/io/direct.rs::probe` instead and verifies it **empirically**:

  - the destination buffer is an anonymous `mmap` (page-aligned by the
    kernel), its address is asserted to be 4096-aligned, and every page is
    written before the timer starts so the pages are faulted in;
  - `mincore` must report **zero** resident pages for every probed file after
    the run, or the reads populated the page cache and were not direct;
  - a **positive control** follows: one buffered 4 KiB read of the same file
    must then show up as resident, or the zero above proved nothing (an
    install owned by another user makes `mincore` answer "all resident" and a
    file we cannot see residency for cannot be verified either way);
  - `filefrag` flags are scanned for `encoded` extents, which are btrfs
    compressed extents and a known silent fallback path.

Any of these failing is a HARD problem: the run is DIRTY and the bandwidth
numbers describe buffered I/O, not O_DIRECT.

**Did the drive report an error?** `btrfs device stats` is sampled before and
after the whole session and again around each confined run. The reference
machine's `corruption_errs` baseline is 138,407
(`docs/benchmark-machine.md`); EXP-007 showed how easily a buffer-aliasing
bug in a benchmark manufactures an increase. **Any delta is a hard failure,
not a footnote.** This probe hands every thread its own buffer for exactly
that reason.

## Output

JSON (everything, including per-read latency percentiles), a human summary,
and a markdown block ready to paste into an experiment entry. Bandwidth is
reported in **GB/s = 10^9 bytes/s**, matching EXP-008's units.

## Exit codes

  0  measured, and every scored run was CLEAN
  1  measured, but at least one scored run was DIRTY — reclaim, a cgroup
     limit, a page cache that grew under O_DIRECT, a btrfs counter delta, or
     a counter the verdict depends on left unreadable. The numbers are still
     printed and clearly marked. **Do not publish them.**
  2  could not measure at all — no model dir, no cgroup v2, no
     `systemd-run --user`, kernel older than 6.5, eviction failed, O_DIRECT
     refused outright, a `ramvamp` process was already running, or the inner
     run produced no result file.

Python stdlib only. Linux >= 6.5 + cgroup v2 + systemd --user, by design.

Usage:

  scripts/io_probe.py --dry-run
  scripts/io_probe.py --repeats 3 --json scratch/io-probe/summary.json
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import mmap
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time

PAGE = os.sysconf("SC_PAGESIZE")
# The O_DIRECT alignment the .rvmp format guarantees: expert blob strides and
# projection offsets are 4 KiB multiples (crates/core/src/format/mod.rs).
DIO_ALIGN = 4096
PROT_READ, MAP_SHARED = 0x1, 0x01
CGROUP_ROOT = "/sys/fs/cgroup"

# cgroup v2 `memory.swap.peak` first appears in Linux 6.5, and `classify`
# treats an unreadable counter as DIRTY, so below 6.5 no run could ever be
# reported CLEAN. That is a broken harness, not a failing measurement, so
# preflight refuses up front. Same floor and same reasoning as cold_bench.py.
SWAP_PEAK_MIN_KERNEL = (6, 5)

BTRFS_COUNTERS = (
    "write_io_errs",
    "read_io_errs",
    "flush_io_errs",
    "corruption_errs",
    "generation_errs",
)

# The fixed, recorded file set. layer_00 and layer_20 are the pair
# docs/benchmark-machine.md names for per-file fragmentation variance (1,237
# vs 1,812 MB/s) and both carry the 3,059,712 B stride; layer_06 and layer_21
# carry the 2,654,208 B stride. Changing this list changes what the numbers
# mean, so it is a default rather than a computed choice.
DEFAULT_FILES = ("layer_00", "layer_20", "layer_06", "layer_21")

# Block sizes are chosen by **expert count**, not by round binary sizes, so
# every read lands on an exact blob boundary. 16 MiB is not an integer number
# of experts (16 MiB / 3,059,712 = 5.48), which is why EXP-008's "16 MiB"
# figure cannot be reproduced as an expert-aligned read at all. K=8 divides
# the 128 experts per layer evenly and gives 23.34 MiB / 20.25 MiB for the two
# stride classes. K=6 does not divide 128; the tail experts are simply not
# read and the covered byte count is recorded.
DEFAULT_KS = (1, 2, 4, 6, 8)
DEFAULT_QDS = (1, 2, 4, 8, 16)
DEFAULT_FIXED_QD = 8
DEFAULT_FIXED_K = 8
PATTERNS = ("seq", "rand")

HARD, SOFT = "hard", "soft"

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
    return f"{n / 2**20:.2f} MiB"


def gbps(byte_count: float, seconds: float) -> float:
    """GB/s in EXP-008's units: 10^9 bytes per second."""
    return (byte_count / seconds) / 1e9 if seconds > 0 else 0.0


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return ordered[index]


# ---------------------------------------------------------------------------
# page-cache residency, via mmap + mincore
# ---------------------------------------------------------------------------


def residency_is_visible(path: str) -> bool:
    """Whether `mincore` will tell the truth about `path`.

    Since Linux 4.19 `mincore` reveals page-cache residency for a file-backed
    mapping only when `inode_owner_or_capable() || file_permission(MAY_WRITE)`
    passes; otherwise it reports the whole range resident and returns success.
    A caller that does not ask this first cannot tell a warm cache from a
    refused answer. Mirrors `direct.rs::residency_is_visible`.
    """
    try:
        if os.geteuid() == 0:
            return True
        if os.stat(path).st_uid == os.geteuid():
            return True
        return os.access(path, os.W_OK, effective_ids=True)
    except (OSError, NotImplementedError, ValueError):
        return False


def resident_bytes(path: str) -> int:
    """Resident page-cache bytes for `path`. Does not fault anything in:
    `mmap` establishes the mapping and `mincore` only reads the state."""
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


def evict_verified(paths: list[str], verbose: bool = True) -> dict:
    """fadvise DONTNEED every file, then prove with mincore that it worked."""
    t0 = time.time()
    before = after = 0
    stuck = []
    for path in paths:
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
        print(f"  evict: {mib(before)} -> {mib(after)} across {len(paths)} "
              f"files in {elapsed:.1f}s")
    if after:
        for a, path in sorted(stuck, reverse=True)[:10]:
            print(f"    STILL RESIDENT {mib(a):>14}  {path}", file=sys.stderr)
        fail(f"eviction failed: {mib(after)} still resident. posix_fadvise "
             f"silently does nothing while a process has the file mmap'd — "
             f"check for a stray ramvamp (or an editor/indexer holding it).")
    return {"resident_before": before, "resident_after": after,
            "files": len(paths), "seconds": round(elapsed, 2)}


# ---------------------------------------------------------------------------
# cgroup v2 counters
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
    """Parse a `key value` table. cgroup files use a space, `/proc/*/io` uses
    `key: value`, so the trailing colon is stripped either way."""
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


def pg_total(pg: dict[str, int], prefix: str) -> int:
    """`memory.stat` carries both `pgsteal` and its `pgsteal_*` breakdown;
    summing every matching key would double-count."""
    if prefix in pg:
        return pg[prefix]
    return sum(v for k, v in pg.items() if k.startswith(prefix + "_"))


# ---------------------------------------------------------------------------
# machine provenance: device, filesystem, mount options, kernel, extents
# ---------------------------------------------------------------------------


def run_tool(argv: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              check=False)
    except (OSError, ValueError) as e:
        return 127, "", str(e)
    return proc.returncode, proc.stdout, proc.stderr


def mount_info(path: str) -> dict:
    """Source device, filesystem type, mount options and mount point."""
    info = {"path": path, "source": None, "fstype": None,
            "options": None, "target": None}
    rc, out, _ = run_tool(["findmnt", "-no", "SOURCE,FSTYPE,OPTIONS,TARGET",
                           "--target", path])
    if rc == 0 and out.strip():
        bits = out.strip().split(None, 3)
        if len(bits) == 4:
            info.update(zip(("source", "fstype", "options", "target"), bits))
    return info


BTRFS_STAT_RE = re.compile(r"^\[(?P<dev>.+)\]\.(?P<key>\w+)\s+(?P<val>\d+)$")


def btrfs_device_stats(target: str | None) -> dict:
    """`btrfs device stats <mountpoint>`, parsed per device.

    Unprivileged on the reference machine (verified: it returns the five
    persistent counters without sudo). A failure here is reported, never
    swallowed — an unreadable counter is the unknown state, and unknown is
    DIRTY when the filesystem is btrfs.
    """
    if not target:
        return {"ok": False, "reason": "no mount target", "devices": {}}
    rc, out, err = run_tool(["btrfs", "device", "stats", target])
    if rc != 0:
        return {"ok": False, "reason": (err or out).strip() or f"rc={rc}",
                "devices": {}}
    devices: dict[str, dict[str, int]] = {}
    for line in out.splitlines():
        m = BTRFS_STAT_RE.match(line.strip())
        if m:
            devices.setdefault(m.group("dev"), {})[m.group("key")] = int(
                m.group("val"))
    if not devices:
        return {"ok": False, "reason": "no counters parsed", "devices": {}}
    return {"ok": True, "reason": None, "devices": devices}


def btrfs_delta(before: dict, after: dict) -> list[str]:
    """Every counter that grew, named. Empty means the drive stayed quiet."""
    grew = []
    for dev, after_counters in (after.get("devices") or {}).items():
        before_counters = (before.get("devices") or {}).get(dev, {})
        for key in BTRFS_COUNTERS:
            b = before_counters.get(key)
            a = after_counters.get(key)
            if b is None or a is None:
                continue
            if a > b:
                grew.append(f"{dev}.{key} {b} -> {a} (+{a - b})")
    return grew


FILEFRAG_HEADER_RE = re.compile(r"\((\d+) blocks of (\d+) bytes\)")
# `ext: logical..logical: physical..physical: length: [expected:] flags`.
# The `expected` column is printed only when the extent is *not* physically
# contiguous with its predecessor, so it is optional and must be matched
# separately — folding it into the flags column would put physical block
# numbers in `flags_seen` and hide a real `encoded` flag among them.
FILEFRAG_EXTENT_RE = re.compile(
    r"^\s*(\d+):\s*(\d+)\.\.\s*(\d+):\s*(\d+)\.\.\s*(\d+):\s*(\d+):"
    r"(?:\s*(\d+):)?\s*(.*?)\s*$")


def filefrag_stats(path: str) -> dict:
    """Extent geometry for `path`: count, mean/median extent size, and how
    many extents are physically adjacent to their successor.

    This is what makes a bandwidth number interpretable. On the reference
    install `layer_00.bin` is 398 extents over 373.5 MiB with **zero**
    physically adjacent successor pairs, and the extent sizes are exactly the
    884,736 B gate/up and 1,290,240 B down slabs — so a "sequential" read is
    physically ~25 scattered ~1 MiB chunks per 24 MiB.

    `encoded` in the flags means a btrfs compressed extent, which is one of
    the documented silent O_DIRECT fallback paths, so it is surfaced rather
    than counted.
    """
    out_stats = {"ok": False, "reason": None, "extents": None,
                 "block_bytes": None, "mean_extent_bytes": None,
                 "median_extent_bytes": None, "min_extent_bytes": None,
                 "max_extent_bytes": None, "adjacent_pairs": None,
                 "adjacent_fraction": None, "filefrag_discontiguous": None,
                 "encoded_extents": 0, "unparsed_lines": 0, "flags_seen": []}
    rc, out, err = run_tool(["filefrag", "-v", path])
    if rc != 0:
        out_stats["reason"] = (err or out).strip() or f"rc={rc}"
        return out_stats
    block_bytes = None
    lengths: list[int] = []
    phys: list[tuple[int, int]] = []
    flags_seen: set[str] = set()
    encoded = 0
    unparsed = 0
    discontiguous = 0
    for line in out.splitlines():
        if block_bytes is None:
            m = FILEFRAG_HEADER_RE.search(line)
            if m:
                block_bytes = int(m.group(2))
                continue
        m = FILEFRAG_EXTENT_RE.match(line)
        if not m:
            if line.strip().startswith(("ext:", "Filesystem", "File size")):
                continue
            if "extents found" in line or not line.strip():
                continue
            unparsed += 1
            continue
        _, _lstart, _lend, pstart, pend, length, expected, flags = m.groups()
        lengths.append(int(length))
        phys.append((int(pstart), int(pend)))
        if expected is not None:
            discontiguous += 1
        for flag in (f.strip() for f in flags.split(",")):
            if flag:
                flags_seen.add(flag)
                if flag == "encoded":
                    encoded += 1
    if not lengths or block_bytes is None:
        out_stats["reason"] = "no extents parsed"
        out_stats["unparsed_lines"] = unparsed
        return out_stats
    sizes = [n * block_bytes for n in lengths]
    adjacent = sum(1 for i in range(len(phys) - 1)
                   if phys[i][1] + 1 == phys[i + 1][0])
    out_stats.update({
        "ok": True,
        "extents": len(sizes),
        "block_bytes": block_bytes,
        "mean_extent_bytes": int(statistics.fmean(sizes)),
        "median_extent_bytes": int(statistics.median(sizes)),
        "min_extent_bytes": min(sizes),
        "max_extent_bytes": max(sizes),
        "adjacent_pairs": adjacent,
        "adjacent_fraction": (round(adjacent / (len(phys) - 1), 4)
                              if len(phys) > 1 else None),
        # filefrag's own view of the same fact: it prints an `expected:`
        # column exactly when an extent is not contiguous with its
        # predecessor. It should equal (extents - 1 - adjacent_pairs); a
        # disagreement means the parse is wrong, not the disk.
        "filefrag_discontiguous": discontiguous,
        "encoded_extents": encoded,
        "unparsed_lines": unparsed,
        "flags_seen": sorted(flags_seen),
    })
    return out_stats


def kernel_version() -> tuple[int, int] | None:
    m = re.match(r"(\d+)\.(\d+)", os.uname().release)
    return (int(m.group(1)), int(m.group(2))) if m else None


# ---------------------------------------------------------------------------
# aligned, pre-faulted destination buffers
# ---------------------------------------------------------------------------


class AlignedBuffer:
    """An anonymous `mmap`, asserted 4096-aligned and fully faulted in.

    btrfs degrades an O_DIRECT read to buffered I/O — silently, with no error
    and a full byte count — when the destination buffer address is misaligned
    or its pages have not been faulted in. `os.pread` cannot be used at all
    here: it returns a fresh `bytes` object whose payload sits behind a
    PyBytes header and is therefore never page-aligned. `os.preadv` into this
    buffer is the only stdlib path that keeps the alignment contract.
    """

    def __init__(self, nbytes: int) -> None:
        if nbytes <= 0 or nbytes % DIO_ALIGN:
            raise ValueError(f"{nbytes} is not a positive multiple of "
                             f"{DIO_ALIGN}")
        self.nbytes = nbytes
        self.map = mmap.mmap(-1, nbytes)
        # The temporary ctypes view is released as soon as `addressof`
        # returns; keeping it alive would block `mmap.close()` later.
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.map))
        self.view = memoryview(self.map)

    @property
    def aligned(self) -> bool:
        return self.address % DIO_ALIGN == 0

    def prefault(self) -> None:
        """Write one byte per page so every page is present before the timer
        starts. A zero page that has never been written is not faulted in, and
        btrfs falls back to buffered rather than faulting it for us."""
        for off in range(0, self.nbytes, PAGE):
            self.map[off:off + 1] = b"\0"

    def close(self) -> None:
        self.view.release()
        self.map.close()


def open_direct(path: str) -> int:
    """Open `path` read-only with O_DIRECT, or refuse to measure.

    Unlike the runtime (`direct.rs::open`), which falls back to a buffered
    open so a machine without direct I/O can still *run*, a probe that
    silently measured buffered I/O would produce exactly the wrong number.
    """
    flag = getattr(os, "O_DIRECT", None)
    if flag is None:
        fail("this Python has no os.O_DIRECT, so the probe cannot open the "
             "expert files the way the runtime does")
    try:
        return os.open(path, os.O_RDONLY | flag)
    except OSError as e:
        fail(f"O_DIRECT open of {path} refused ({e}). The probe will not fall "
             f"back to buffered I/O: that would measure the page cache and "
             f"report it as drive bandwidth.")
        raise  # unreachable; keeps type checkers honest


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------


def case_seed(base_seed: int, repeat: int, case: dict) -> int:
    """A stable per-case seed.

    Derived with SHA-256 rather than `hash()`: Python randomizes `str.__hash__`
    per process (PYTHONHASHSEED), and each repeat runs in a *fresh* inner
    process, so a tuple hash would hand the same case a different access order
    every repeat and make `--seed` a lie. Each case still gets its own order —
    reusing one permutation everywhere would let a single unlucky layout
    decide the whole random column.
    """
    key = f"{base_seed}|{repeat}|{case['file']}|{case['pattern']}|" \
          f"{case['k']}|{case['qd']}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8],
                          "big")


def case_offsets(case: dict, rng: random.Random) -> list[int]:
    """Byte offsets of the blocks this case reads.

    Both patterns read **the same blocks and the same bytes**; only the order
    differs. That is deliberate: it isolates sequentiality from granularity,
    which EXP-008 conflated. `rand` is a uniform permutation of the K-aligned
    block starts, so every read still lands on an exact expert-blob boundary
    and every offset is a 4096 multiple.
    """
    stride = case["stride"]
    block = case["block_bytes"]
    offsets = [i * block for i in range(case["blocks"])]
    if any(off % DIO_ALIGN for off in offsets) or stride % DIO_ALIGN:
        raise ValueError("block offsets are not O_DIRECT aligned")
    if case["pattern"] == "rand":
        rng.shuffle(offsets)
    return offsets


def timed_reads(fd: int, offsets: list[int], block_bytes: int,
                qd: int) -> dict:
    """`qd` threads pulling from one work queue, each with its own buffer.

    Per-thread buffers, never a shared pool indexed by arithmetic: EXP-007
    established that two in-flight O_DIRECT reads landing in one buffer make
    btrfs verify a checksum over bytes the other read already overwrote, which
    is where `docs/benchmark-machine.md`'s 138,407 `corruption_errs` came
    from. A benchmark is not allowed to manufacture that again.

    Allocation and pre-faulting happen **before** the timer starts, so the
    reported bandwidth is drive time, not page-fault time.
    """
    buffers = [AlignedBuffer(block_bytes) for _ in range(qd)]
    misaligned = [b.address for b in buffers if not b.aligned]
    for buf in buffers:
        buf.prefault()

    lock = threading.Lock()
    cursor = iter(range(len(offsets)))
    latencies: list[float] = []
    errors: list[str] = []
    short_reads = 0

    def worker(buf: AlignedBuffer) -> None:
        nonlocal short_reads
        local: list[float] = []
        local_short = 0
        local_errors: list[str] = []
        view = buf.view
        while True:
            with lock:
                index = next(cursor, None)
            if index is None:
                break
            off = offsets[index]
            t0 = time.perf_counter()
            got = 0
            while got < block_bytes:
                try:
                    if got == 0:
                        n = os.preadv(fd, [view], off)
                    else:
                        # A short O_DIRECT read must still resume on an
                        # aligned boundary, or the retry itself is the thing
                        # that degrades to buffered.
                        if got % DIO_ALIGN:
                            local_errors.append(
                                f"short read resumed at unaligned offset "
                                f"{off + got}")
                            break
                        sub = view[got:]
                        try:
                            n = os.preadv(fd, [sub], off + got)
                        finally:
                            sub.release()
                except OSError as e:
                    local_errors.append(f"preadv at {off + got}: {e}")
                    break
                if n <= 0:
                    local_errors.append(f"preadv at {off + got} returned {n}")
                    break
                got += n
                if got < block_bytes:
                    local_short += 1
            local.append(time.perf_counter() - t0)
        with lock:
            latencies.extend(local)
            errors.extend(local_errors)
            short_reads += local_short

    threads = [threading.Thread(target=worker, args=(buf,), daemon=True)
               for buf in buffers]
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    cpu = time.process_time() - cpu0

    for buf in buffers:
        buf.close()

    total = len(offsets) * block_bytes
    return {
        "seconds": elapsed,
        "cpu_seconds": round(cpu, 4),
        "bytes": total,
        "gb_s": round(gbps(total, elapsed), 4),
        "reads": len(offsets),
        "short_reads": short_reads,
        "errors": errors,
        "buffer_addresses_misaligned": misaligned,
        "latency_ms": {
            "p50": round((pct(latencies, 0.50) or 0) * 1e3, 4),
            "p90": round((pct(latencies, 0.90) or 0) * 1e3, 4),
            "p99": round((pct(latencies, 0.99) or 0) * 1e3, 4),
            "max": round((max(latencies) if latencies else 0) * 1e3, 4),
        },
    }


def inner(args) -> int:
    """Runs INSIDE the cgroup. Does the reads, reads the counters, exits."""
    with open(args.plan_file, encoding="utf-8") as f:
        plan = json.load(f)

    cg = own_cgroup_dir()
    result: dict = {"cgroup": cg, "repeat": plan["repeat"],
                    "label": plan["label"]}
    log_lines: list[str] = []

    def log(line: str) -> None:
        log_lines.append(line)

    paths = [f["path"] for f in plan["files"]]

    # Residency must be *visible* before it can be evidence. An install owned
    # by another user makes mincore answer "everything resident" and return
    # success, which would look like a permanently degraded O_DIRECT.
    invisible = [p for p in paths if not residency_is_visible(p)]

    baseline_resident = {}
    for path in paths:
        try:
            baseline_resident[path] = resident_bytes(path)
        except OSError as e:
            baseline_resident[path] = -1
            log(f"residency baseline failed for {path}: {e}")

    btrfs_before = btrfs_device_stats(plan["mount"]["target"])
    io_before = read_kv("/proc/self/io")

    cases: list[dict] = []
    t_start = time.time()
    fds: dict[str, int] = {}
    try:
        for case in plan["cases"]:
            path = case["path"]
            if path not in fds:
                fds[path] = open_direct(path)
            rng = random.Random(case_seed(plan["seed"], plan["repeat"], case))
            offsets = case_offsets(case, rng)
            measured = timed_reads(fds[path], offsets, case["block_bytes"],
                                   case["qd"])
            record = dict(case)
            record.update(measured)
            cases.append(record)
            log(f"  {case['file']:<9} {case['pattern']:<4} K={case['k']:<2} "
                f"QD={case['qd']:<2} {case['block_bytes']:>9} B  "
                f"{measured['gb_s']:>6.3f} GB/s  "
                f"p50 {measured['latency_ms']['p50']:>7.3f} ms  "
                f"{measured['seconds']:>6.2f} s")
    finally:
        for fd in fds.values():
            os.close(fd)
    matrix_seconds = time.time() - t_start

    # O_DIRECT verification, empirical. Nothing may be resident now.
    after_resident = {}
    for path in paths:
        try:
            after_resident[path] = resident_bytes(path)
        except OSError as e:
            after_resident[path] = -1
            log(f"residency check failed for {path}: {e}")

    # Positive control: a buffered read of the same file must show up, or the
    # zeros above proved nothing at all. Mirrors direct.rs's
    # `NoResidencySignal` branch.
    control = {"ok": False, "reason": "not attempted", "resident_after": None}
    if paths:
        probe_path = paths[0]
        try:
            fd = os.open(probe_path, os.O_RDONLY)
            try:
                os.pread(fd, DIO_ALIGN, 0)
                grew = resident_bytes(probe_path)
                control = {"ok": grew > 0,
                           "reason": None if grew > 0 else
                           "a buffered read left nothing resident, so the "
                           "zero-residency result above is not evidence",
                           "resident_after": grew,
                           "path": probe_path}
                os.posix_fadvise(fd, 0, DIO_ALIGN, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError as e:
            control = {"ok": False, "reason": f"control read failed: {e}",
                       "resident_after": None, "path": probe_path}

    btrfs_after = btrfs_device_stats(plan["mount"]["target"])
    io_after = read_kv("/proc/self/io")
    stat = read_kv(os.path.join(cg, "memory.stat")) if cg else {}

    result.update({
        "matrix_seconds": round(matrix_seconds, 3),
        "cases": cases,
        "residency_visible": not invisible,
        "residency_invisible_files": invisible,
        "resident_before": baseline_resident,
        "resident_after": after_resident,
        "positive_control": control,
        "btrfs_before": btrfs_before,
        "btrfs_after": btrfs_after,
        "btrfs_grew": btrfs_delta(btrfs_before, btrfs_after),
        "memory_peak": read_int(os.path.join(cg, "memory.peak")) if cg else None,
        "memory_current": read_int(os.path.join(cg, "memory.current")) if cg else None,
        "memory_max": read_int(os.path.join(cg, "memory.max")) if cg else None,
        "memory_swap_max": read_int(os.path.join(cg, "memory.swap.max")) if cg else None,
        "memory_swap_peak": read_int(os.path.join(cg, "memory.swap.peak")) if cg else None,
        "memory_events": read_kv(os.path.join(cg, "memory.events")) if cg else {},
        "pg": {k: v for k, v in stat.items()
               if k.startswith("pgscan") or k.startswith("pgsteal")},
        "read_bytes_delta": (io_after.get("read_bytes", 0)
                             - io_before.get("read_bytes", 0)),
        "log": log_lines,
    })

    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


# ---------------------------------------------------------------------------
# hygiene verdict
# ---------------------------------------------------------------------------


def classify(run: dict, want_max: int, fstype: str | None) -> tuple[str, list[dict]]:
    """CLEAN or DIRTY, with reasons. Unknown is DIRTY, never CLEAN."""
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
                   "could not be measured, so this run is not verified clean "
                   "(this is the unknown state, not a clean one)")
    else:
        steal = pg_total(pg, "pgsteal")
        scan = pg_total(pg, "pgscan")
        if steal:
            note(HARD, f"pgsteal {steal} pages ({mib(steal * PAGE)}) reclaimed "
                       f"under pressure — the working set did not fit, so the "
                       f"timing is not a clean {want_max // 2**30} GB "
                       f"measurement")
        elif scan:
            note(SOFT, f"pgscan {scan} pages with no steal (pressure, no loss)")

    events = run.get("memory_events")
    if not events:
        note(HARD, "memory.events was unreadable or empty — OOM and limit hits "
                   "could not be ruled out")
    else:
        for key in ("max", "oom", "oom_kill", "oom_group_kill", "high"):
            if events.get(key):
                note(HARD, f"memory.events {key}={events[key]}")

    if run.get("memory_max") is None:
        note(HARD, f"memory.max is unreadable, so a {want_max}-byte limit could "
                   f"not be confirmed — the memory controller is probably not "
                   f"delegated to the user slice, which means the run was "
                   f"unconfined")
    elif run.get("memory_max") != want_max:
        note(HARD, f"memory.max is {run.get('memory_max')}, expected "
                   f"{want_max} — the memory controller may not be delegated "
                   f"to the user slice")

    if run.get("memory_swap_max") is None:
        note(HARD, "memory.swap.max is unreadable, so swap could not be "
                   "confirmed off (zram counts as swap)")
    elif run.get("memory_swap_max") != 0:
        note(HARD, f"memory.swap.max is {run.get('memory_swap_max')}, "
                   f"expected 0 (zram counts as swap)")

    if run.get("memory_swap_peak") is None:
        note(HARD, "memory.swap.peak is unreadable, so it cannot be shown that "
                   "the run never swapped")
    elif run["memory_swap_peak"]:
        note(HARD, f"swapped {mib(run['memory_swap_peak'])}")

    if not run.get("read_bytes_delta"):
        note(HARD, "read_bytes delta is 0 — nothing came from the block layer, "
                   "so these reads did not reach the drive")

    # --- O_DIRECT actually honoured ------------------------------------
    if not run.get("residency_visible", False):
        note(HARD, f"mincore will not report page-cache residency for "
                   f"{run.get('residency_invisible_files')} (not owned by this "
                   f"user and not writable), so O_DIRECT cannot be verified "
                   f"either way")
    before = run.get("resident_before") or {}
    after = run.get("resident_after") or {}
    for path, value in before.items():
        if value < 0:
            note(HARD, f"residency baseline for {path} could not be read")
        elif value:
            note(HARD, f"{path} had {mib(value)} resident before the run — the "
                       f"eviction the outer harness verified did not hold, so "
                       f"'the cache grew' cannot be told from 'the cache was "
                       f"already warm'")
    for path, value in after.items():
        if value < 0:
            note(HARD, f"residency check for {path} could not be read")
        elif value:
            note(HARD, f"{path} has {mib(value)} resident AFTER the run — the "
                       f"reads populated the page cache, so O_DIRECT was "
                       f"silently downgraded to buffered I/O and these numbers "
                       f"are page-cache bandwidth, not drive bandwidth")
    control = run.get("positive_control") or {}
    if not control.get("ok"):
        note(HARD, f"the residency positive control failed "
                   f"({control.get('reason')}) — a buffered read that leaves "
                   f"nothing resident means the zero-residency result above is "
                   f"not evidence of anything")

    # --- the drive stayed quiet ----------------------------------------
    grew = run.get("btrfs_grew") or []
    if grew:
        note(HARD, "btrfs device stats increased during the run: "
                   + "; ".join(grew)
                   + ". EXP-007: a benchmark that aliases its buffers "
                     "manufactures exactly this. Investigate before "
                     "publishing anything.")
    if fstype == "btrfs":
        for when in ("btrfs_before", "btrfs_after"):
            snapshot = run.get(when) or {}
            if not snapshot.get("ok"):
                note(HARD, f"{when} could not be read "
                           f"({snapshot.get('reason')}), so a device-error "
                           f"delta could not be ruled out")

    # --- the reads themselves ------------------------------------------
    total_short = sum(c.get("short_reads", 0) for c in run.get("cases", []))
    if total_short:
        note(HARD, f"{total_short} short O_DIRECT reads — the byte counts the "
                   f"bandwidth is computed from are not what a single read "
                   f"delivered")
    read_errors = [e for c in run.get("cases", []) for e in c.get("errors", [])]
    if read_errors:
        note(HARD, f"{len(read_errors)} read errors, first: {read_errors[0]}")
    misaligned = [a for c in run.get("cases", [])
                  for a in c.get("buffer_addresses_misaligned", [])]
    if misaligned:
        note(HARD, f"{len(misaligned)} destination buffers were not "
                   f"{DIO_ALIGN}-aligned (first address {misaligned[0]}), "
                   f"which is one of btrfs's silent buffered-fallback "
                   f"conditions")

    if run.get("systemd_run_returncode"):
        note(HARD, f"systemd-run exited {run['systemd_run_returncode']} — the "
                   f"confined run did not complete normally, so whatever "
                   f"result file was classified is not this run's")

    hard = [p for p in problems if p["severity"] == HARD]
    return ("CLEAN" if not hard else "DIRTY"), problems


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def load_layout(rvmp: str) -> list[dict]:
    path = os.path.join(rvmp, "experts", "layout.json")
    try:
        with open(path, encoding="utf-8") as f:
            layout = json.load(f)
    except (OSError, ValueError) as e:
        fail(f"cannot read {path}: {e}. This probe reads the *installed* "
             f"model, never a synthetic scratch file, because the point is "
             f"the extent geometry the installer produced.")
    layers = layout.get("layers")
    if not isinstance(layers, list) or not layers:
        fail(f"{path} has no layers array")
    return layers


def resolve_files(names: list[str], layers: list[dict], rvmp: str) -> list[dict]:
    by_stem = {}
    for index, layer in enumerate(layers):
        stem = os.path.splitext(os.path.basename(layer["file"]))[0]
        by_stem[stem] = (index, layer)
        by_stem[f"{index}"] = (index, layer)
        by_stem[f"{index:02d}"] = (index, layer)
    chosen = []
    for name in names:
        key = os.path.splitext(os.path.basename(name.strip()))[0]
        if key not in by_stem:
            fail(f"no layer file matches {name!r}; known stems look like "
                 f"'layer_00' (0..{len(layers) - 1})")
        index, layer = by_stem[key]
        path = os.path.join(rvmp, layer["file"])
        try:
            size = os.stat(path).st_size
        except OSError as e:
            fail(f"cannot stat {path}: {e}")
        expected = layer["stride"] * layer["n_experts"]
        if size != expected:
            fail(f"{path} is {size} bytes but layout.json says "
                 f"{layer['n_experts']} x {layer['stride']} = {expected}; the "
                 f"install does not match its own layout")
        if layer["stride"] % DIO_ALIGN:
            fail(f"{path} has stride {layer['stride']}, not a multiple of "
                 f"{DIO_ALIGN}; O_DIRECT reads at blob boundaries are "
                 f"impossible")
        chosen.append({
            "name": os.path.splitext(os.path.basename(layer["file"]))[0],
            "layer": index,
            "path": os.path.abspath(path),
            "stride": layer["stride"],
            "n_experts": layer["n_experts"],
            "size": size,
        })
    seen = set()
    unique = []
    for entry in chosen:
        if entry["path"] not in seen:
            seen.add(entry["path"])
            unique.append(entry)
    return unique


def build_cases(files: list[dict], ks: list[int], qds: list[int],
                fixed_qd: int, fixed_k: int,
                patterns: list[str]) -> tuple[list[dict], list[tuple[int, int]]]:
    """The (K, QD) combos, then one case per (combo, file, pattern).

    The matrix is the **union** of two sweeps sharing an operating point, not
    a full cross product: a block-size sweep at `fixed_qd` and a queue-depth
    sweep at `fixed_k`. The full 5x5 product would quadruple the wall time to
    answer questions nobody asked.

    Case order puts the combo outermost and alternates patterns on the same
    file back to back, so the seq/rand comparison — the one that carries the
    headline — is never split across a long stretch of other work.
    """
    combos: list[tuple[int, int]] = []
    for k in ks:
        combos.append((k, fixed_qd))
    for qd in qds:
        if (fixed_k, qd) not in combos:
            combos.append((fixed_k, qd))

    cases = []
    order = 0
    for k, qd in combos:
        for f in files:
            blocks = f["n_experts"] // k
            if blocks == 0:
                continue
            block_bytes = k * f["stride"]
            for pattern in patterns:
                cases.append({
                    "order": order,
                    "file": f["name"],
                    "layer": f["layer"],
                    "path": f["path"],
                    "stride": f["stride"],
                    "stride_class": f"{f['stride']}",
                    "n_experts": f["n_experts"],
                    "pattern": pattern,
                    "k": k,
                    "qd": qd,
                    "block_bytes": block_bytes,
                    "block_mib": round(block_bytes / 2**20, 4),
                    "blocks": blocks,
                    "experts_covered": blocks * k,
                    "bytes": blocks * block_bytes,
                    "queue": "threaded-pread",
                })
                order += 1
    return cases, combos


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def case_key(case: dict) -> tuple:
    return (case["file"], case["pattern"], case["k"], case["qd"])


def aggregate(runs: list[dict]) -> dict[tuple, dict]:
    """Per-case median GB/s across scored runs, with the spread kept."""
    buckets: dict[tuple, list[dict]] = {}
    for run in runs:
        for case in run.get("cases", []):
            buckets.setdefault(case_key(case), []).append(case)
    out = {}
    for key, entries in buckets.items():
        rates = [e["gb_s"] for e in entries]
        p50s = [e["latency_ms"]["p50"] for e in entries]
        out[key] = {
            "template": entries[0],
            "n": len(entries),
            "gb_s_median": round(statistics.median(rates), 4),
            "gb_s_min": round(min(rates), 4),
            "gb_s_max": round(max(rates), 4),
            "p50_ms_median": round(statistics.median(p50s), 4),
        }
    return out


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_markdown(plan: dict, agg: dict[tuple, dict], frag: dict,
                   machine: dict, verdict: str) -> str:
    files = plan["files"]
    fixed_qd, fixed_k = plan["fixed_qd"], plan["fixed_k"]
    out: list[str] = []

    out.append(f"Machine: {machine['mount']['source']} "
               f"({machine['mount']['fstype']}, "
               f"{machine['mount']['options']}), kernel "
               f"{machine['kernel_release']}. Queue depth emulated with "
               f"**N OS threads issuing blocking `preadv`** "
               f"(`{plan['queue']}`), **not io_uring** — these numbers must "
               f"not be compared against the runtime's io_uring path without "
               f"that caveat. GB/s = 10^9 B/s. Files are the installed "
               f"`{os.path.basename(plan['rvmp'])}` layer files, read cold "
               f"inside a `memory.max={plan['memory_max']}`, "
               f"`memory.swap.max=0` cgroup. Hygiene: **{verdict}**.")
    out.append("")

    def rate(name, pattern, k, qd):
        entry = agg.get((name, pattern, k, qd))
        return entry["gb_s_median"] if entry else None

    def fmt(value):
        return f"{value:.3f}" if isinstance(value, float) else "-"

    out.append(f"**Block-size sweep at QD={fixed_qd}** "
               f"(block size chosen by expert count K, so every read lands on "
               f"an exact blob boundary)")
    out.append("")
    rows = []
    for k in plan["ks"]:
        for f in files:
            block = k * f["stride"]
            if f["n_experts"] // k == 0:
                continue
            seq = rate(f["name"], "seq", k, fixed_qd)
            rnd = rate(f["name"], "rand", k, fixed_qd)
            rows.append([k, block, f"{block / 2**20:.2f}", f["name"],
                         f["stride"], fmt(seq), fmt(rnd),
                         f"{seq / rnd:.2f}x" if seq and rnd else "-"])
    out.append(md_table(
        ["K", "block bytes", "block MiB", "file", "stride", "seq GB/s",
         "rand GB/s", "seq/rand"], rows))
    out.append("")

    out.append(f"**Queue-depth sweep at K={fixed_k}**")
    out.append("")
    rows = []
    for qd in plan["qds"]:
        for f in files:
            seq = rate(f["name"], "seq", fixed_k, qd)
            rnd = rate(f["name"], "rand", fixed_k, qd)
            rows.append([qd, f["name"], f["stride"], fmt(seq), fmt(rnd),
                         f"{seq / rnd:.2f}x" if seq and rnd else "-"])
    out.append(md_table(
        ["QD", "file", "stride", "seq GB/s", "rand GB/s", "seq/rand"], rows))
    out.append("")

    out.append(f"**Headline decomposition at QD={fixed_qd}** — what a "
               f"front-to-back K={fixed_k} prefill read buys over the "
               f"runtime's random single-expert read, split into the part that "
               f"comes from bigger blocks and the part that comes from order")
    out.append("")
    rows = []
    for f in files:
        r1 = rate(f["name"], "rand", 1, fixed_qd)
        r8 = rate(f["name"], "rand", fixed_k, fixed_qd)
        s8 = rate(f["name"], "seq", fixed_k, fixed_qd)
        rows.append([
            f["name"], f["stride"], fmt(r1), fmt(r8), fmt(s8),
            f"{r8 / r1:.2f}x" if r1 and r8 else "-",
            f"{s8 / r8:.2f}x" if r8 and s8 else "-",
            f"{s8 / r1:.2f}x" if r1 and s8 else "-",
        ])
    out.append(md_table(
        ["file", "stride", "rand K=1 GB/s", f"rand K={fixed_k} GB/s",
         f"seq K={fixed_k} GB/s", "granularity", "sequentiality",
         "combined"], rows))
    out.append("")

    out.append("**Extent geometry of the probed files** (`filefrag -v`) — the "
               "installer writes each projection slab as its own CoW extent, "
               "so a 'sequential' read is physically scattered")
    out.append("")
    rows = []
    for f in files:
        s = frag.get(f["path"], {})
        rows.append([
            f["name"], f["size"],
            s.get("extents", "-"),
            s.get("mean_extent_bytes", "-"),
            s.get("median_extent_bytes", "-"),
            s.get("adjacent_pairs", "-"),
            (f"{s['adjacent_fraction'] * 100:.1f}%"
             if s.get("adjacent_fraction") is not None else "-"),
            s.get("encoded_extents", "-"),
        ])
    out.append(md_table(
        ["file", "bytes", "extents", "mean extent B", "median extent B",
         "physically adjacent pairs", "adjacent %", "compressed extents"],
        rows))
    return "\n".join(out)


def print_plan(plan: dict, assume_bw: float) -> None:
    print("=== plan (dry run: nothing is read, nothing is evicted) ===")
    print(f"model:        {plan['rvmp']}")
    print(f"memory max:   {plan['memory_max']}  swap max: 0")
    print(f"queue model:  {plan['queue']} "
          f"(NOT io_uring — see the module docstring)")
    print(f"seed:         {plan['seed']}")
    print(f"runs:         {plan['warmup']} warmup (discarded) + "
          f"{plan['repeats']} scored")
    print("\nfiles (fixed set, both stride classes):")
    for f in plan["files"]:
        print(f"  {f['name']:<10} layer {f['layer']:<3} stride {f['stride']:>9} B "
              f"x {f['n_experts']} experts = {f['size']:>12} B "
              f"({mib(f['size'])})")
    classes = sorted({f["stride"] for f in plan["files"]})
    print(f"  stride classes covered: {classes}")

    print(f"\nblock sizes by expert count K (at QD={plan['fixed_qd']}):")
    for k in plan["ks"]:
        sizes = ", ".join(
            f"{f['name']}={k * f['stride']} B ({k * f['stride'] / 2**20:.2f} MiB)"
            for f in plan["files"])
        covered = ", ".join(
            f"{f['name']}:{(f['n_experts'] // k) * k}/{f['n_experts']}"
            for f in plan["files"])
        print(f"  K={k:<2} {sizes}")
        print(f"       experts covered: {covered}")
    print(f"\nqueue depths (at K={plan['fixed_k']}): {plan['qds']}")
    print(f"patterns: {plan['patterns']}  "
          f"(rand = uniform permutation of the same K-aligned blocks, so both "
          f"patterns read identical bytes)")

    print(f"\ncombos (K, QD): {plan['combos']}")
    print(f"cases per run:  {len(plan['cases'])}")
    per_run = sum(c["bytes"] for c in plan["cases"])
    total_runs = plan["warmup"] + plan["repeats"]
    total = per_run * total_runs
    print(f"bytes per run:  {per_run} ({per_run / 2**30:.2f} GiB)")
    print(f"bytes total:    {total} ({total / 2**30:.2f} GiB) "
          f"over {total_runs} runs")

    io_s = total / assume_bw
    overhead = total_runs * (2.0 + 1.5) + len(plan["cases"]) * total_runs * 0.15
    print(f"\nestimated wall time: {io_s + overhead:.0f} s "
          f"({(io_s + overhead) / 60:.1f} min)")
    print(f"  = {io_s:.0f} s of I/O at an assumed {assume_bw / 1e9:.2f} GB/s "
          f"+ {overhead:.0f} s of eviction, systemd startup and buffer "
          f"pre-faulting")
    print("  This is an estimate from an assumed bandwidth, not a measurement.")

    print("\nper-case detail:")
    for c in plan["cases"]:
        print(f"  [{c['order']:>3}] {c['file']:<10} {c['pattern']:<4} "
              f"K={c['k']:<2} QD={c['qd']:<2} block {c['block_bytes']:>9} B "
              f"x {c['blocks']:>3} = {c['bytes']:>12} B")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def preflight(args) -> None:
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
             f"counter is unreadable on every run here — so every run would be "
             f"DIRTY with no remedy. Take the measurement on a >= "
             f"{SWAP_PEAK_MIN_KERNEL[0]}.{SWAP_PEAK_MIN_KERNEL[1]} kernel.")
    if not os.path.isdir(args.rvmp):
        fail(f"no installed model at {args.rvmp}. This probe measures the "
             f"real .rvmp files on purpose — a freshly written scratch file "
             f"would be contiguous and would not reproduce the installer's "
             f"extent fragmentation, which is the thing under test.")
    if not os.path.isdir(os.path.join(args.rvmp, "experts")):
        fail(f"{args.rvmp} has no experts/ directory")
    if getattr(os, "O_DIRECT", None) is None:
        fail("this Python has no os.O_DIRECT")
    if not hasattr(os, "preadv"):
        fail("this Python has no os.preadv, which is the only stdlib read "
             "that can target a page-aligned buffer; os.pread allocates a "
             "PyBytes whose payload is never 4096-aligned")
    found = subprocess.run(["pgrep", "-a", "ramvamp"], capture_output=True,
                           text=True, check=False)
    if found.returncode == 0 and found.stdout.strip():
        print(found.stdout.strip(), file=sys.stderr)
        fail("a ramvamp process is running. posix_fadvise cannot evict a file "
             "that any process has mmap'd, and it reports success anyway — "
             "refusing to produce a fake cold run.")
    if not os.path.isdir(CGROUP_ROOT) or not os.path.exists(
            os.path.join(CGROUP_ROOT, "cgroup.controllers")):
        fail("cgroup v2 is not mounted at /sys/fs/cgroup")
    if subprocess.run(["systemd-run", "--user", "--version"],
                      capture_output=True, check=False).returncode != 0:
        fail("systemd-run --user is not available")


def parse_size(text: str) -> int:
    units = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    text = text.strip()
    if text and text[-1].upper() in units:
        return int(float(text[:-1]) * units[text[-1].upper()])
    return int(text)


def one_run(args, plan: dict, index: int, label: str, fstype: str | None) -> dict:
    print(f"\n=== run {index} ({label}) ===")
    paths = [f["path"] for f in plan["files"]]
    ev = evict_verified(paths)

    result_file = os.path.join(args.workdir, f"run{index:02d}.json")
    plan_file = os.path.join(args.workdir, f"run{index:02d}.plan.json")
    for stale in (result_file, plan_file):
        try:
            os.unlink(stale)
        except FileNotFoundError:
            pass
        except OSError as e:
            fail(f"cannot remove the stale file {stale}: {e}")

    run_plan = dict(plan)
    run_plan["repeat"] = index
    run_plan["label"] = label
    # The plan goes through a file, never systemd-run's command line: systemd
    # expands ${NAME} and unescapes $$ inside ExecStart= arguments, silently,
    # and a model path is user-controlled text.
    with open(plan_file, "w", encoding="utf-8") as f:
        json.dump(run_plan, f)

    unit = f"ramvamp-ioprobe-{os.getpid()}-{index}"
    cmd = [
        "systemd-run", "--user", "--wait", "-q", "--collect",
        f"--unit={unit}",
        "-p", f"MemoryMax={args.memory_max}",
        "-p", "MemorySwapMax=0",
        "-p", "MemoryAccounting=yes",
        "-p", f"WorkingDirectory={os.getcwd()}",
        "--",
        sys.executable, os.path.abspath(__file__),
        "--inner", "--result-file", result_file, "--plan-file", plan_file,
    ]
    print(f"  + {' '.join(cmd[:12])} ...")
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

    for line in run.get("log", []):
        print(line)

    verdict, problems = classify(run, parse_size(args.memory_max), fstype)
    run["hygiene"] = verdict
    run["hygiene_problems"] = problems

    print(f"  matrix {run.get('matrix_seconds')}s  outer {run['outer_wall_s']}s")
    print(f"  MemoryPeak {mib(run.get('memory_peak') or 0)} "
          f"(max {mib(run.get('memory_max') or 0)}, swap.max "
          f"{run.get('memory_swap_max')})")
    print(f"  pgscan {pg_total(run.get('pg', {}), 'pgscan')}  "
          f"pgsteal {pg_total(run.get('pg', {}), 'pgsteal')}")
    print(f"  read_bytes {run.get('read_bytes_delta')} "
          f"({mib(run.get('read_bytes_delta') or 0)})")
    print(f"  resident after run: "
          f"{sum(v for v in (run.get('resident_after') or {}).values() if v > 0)} B "
          f"(must be 0 for O_DIRECT)")
    print(f"  btrfs counter deltas: {run.get('btrfs_grew') or 'none'}")
    print(f"  hygiene: {verdict}")
    for problem in problems:
        print(f"    - [{problem['severity']}] {problem['message']}")
    return run


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Bandwidth is GB/s = 10^9 bytes/s. Queue depth is emulated "
               "with OS threads issuing blocking preadv, NOT io_uring: do not "
               "compare these numbers against the runtime's io_uring path "
               "without saying so.")
    parser.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    parser.add_argument("--plan-file", help=argparse.SUPPRESS)

    parser.add_argument("--rvmp", default="models/qwen3.rvmp",
                        help="installed .rvmp model dir (default: %(default)s)")
    parser.add_argument("--files", default=",".join(DEFAULT_FILES),
                        help="comma-separated layer stems to probe. The "
                             "default set is fixed and recorded because "
                             "per-file fragmentation variance exceeds "
                             "run-to-run variance (default: %(default)s)")
    parser.add_argument("--block-ks", default=",".join(str(k) for k in DEFAULT_KS),
                        help="block sizes as expert counts, so reads land on "
                             "exact blob boundaries (default: %(default)s)")
    parser.add_argument("--queue-depths",
                        default=",".join(str(q) for q in DEFAULT_QDS),
                        help="queue depths for the depth sweep "
                             "(default: %(default)s)")
    parser.add_argument("--fixed-qd", type=int, default=DEFAULT_FIXED_QD,
                        help="queue depth held constant during the block-size "
                             "sweep (default: %(default)s)")
    parser.add_argument("--fixed-k", type=int, default=DEFAULT_FIXED_K,
                        help="expert count held constant during the "
                             "queue-depth sweep (default: %(default)s)")
    parser.add_argument("--patterns", default=",".join(PATTERNS),
                        help="access patterns to compare (default: %(default)s)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="scored runs of the whole matrix (default: "
                             "%(default)s)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="discarded leading runs; run 1 pays btrfs extent "
                             "metadata warm-up (default: %(default)s)")
    parser.add_argument("--memory-max", default="3G",
                        help="cgroup MemoryMax (docs rule: 3G)")
    parser.add_argument("--seed", type=int, default=20260804,
                        help="seed for the random access order, recorded so "
                             "the run reproduces (default: %(default)s)")
    parser.add_argument("--workdir", default="scratch/io-probe",
                        help="where per-run JSON lands (default: %(default)s)")
    parser.add_argument("--json", help="write the full summary here")
    parser.add_argument("--markdown", help="write the markdown tables here")
    parser.add_argument("--assume-bw", type=float, default=1.4e9,
                        help="bytes/s assumed only for the --dry-run wall-time "
                             "estimate (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the full plan — files, block sizes, queue "
                             "depths, byte totals, estimated wall time — and "
                             "exit without touching the drive")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if args.inner:
        if not args.result_file or not args.plan_file:
            fail("--inner needs --result-file and --plan-file")
        return inner(args)

    if args.repeats < 1:
        fail(f"--repeats {args.repeats} scores nothing; a run that measures "
             f"zero runs cannot pass a hygiene gate. Use --repeats >= 1.")
    if args.warmup < 0:
        fail(f"--warmup {args.warmup} is negative")

    def int_list(text: str, what: str) -> list[int]:
        try:
            values = [int(v) for v in text.split(",") if v.strip()]
        except ValueError:
            fail(f"--{what} must be a comma-separated list of integers, got "
                 f"{text!r}")
        if not values or any(v < 1 for v in values):
            fail(f"--{what} must be a non-empty list of positive integers")
        return values

    ks = int_list(args.block_ks, "block-ks")
    qds = int_list(args.queue_depths, "queue-depths")
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    for pattern in patterns:
        if pattern not in PATTERNS:
            fail(f"unknown pattern {pattern!r}; known: {list(PATTERNS)}")
    if args.fixed_qd < 1 or args.fixed_k < 1:
        fail("--fixed-qd and --fixed-k must be >= 1")

    args.rvmp = os.path.abspath(args.rvmp)
    preflight(args)

    layers = load_layout(args.rvmp)
    files = resolve_files(args.files.split(","), layers, args.rvmp)
    classes = sorted({f["stride"] for f in files})
    if len(classes) < 2:
        print(f"warning: the probed file set covers only stride class(es) "
              f"{classes}. The install has two, and they read differently — "
              f"the result will not generalise.", file=sys.stderr)

    cases, combos = build_cases(files, ks, qds, args.fixed_qd, args.fixed_k,
                                patterns)
    if not cases:
        fail("the requested matrix contains no cases")

    mount = mount_info(os.path.join(args.rvmp, "experts"))
    plan = {
        "rvmp": args.rvmp,
        "files": files,
        "ks": ks,
        "qds": qds,
        "fixed_qd": args.fixed_qd,
        "fixed_k": args.fixed_k,
        "patterns": patterns,
        "combos": combos,
        "cases": cases,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "memory_max": args.memory_max,
        "seed": args.seed,
        "queue": "threaded-pread",
        "mount": mount,
    }

    if args.dry_run:
        print_plan(plan, args.assume_bw)
        return 0

    os.makedirs(args.workdir, exist_ok=True)
    args.workdir = os.path.abspath(args.workdir)

    machine = {
        "kernel_release": os.uname().release,
        "uname": " ".join(os.uname()),
        "mount": mount,
        "page_size": PAGE,
        "dio_align": DIO_ALIGN,
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "command_line": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print("io_probe: O_DIRECT bandwidth over the installed expert files")
    print(f"  model      {args.rvmp}")
    print(f"  device     {mount['source']} ({mount['fstype']}) on "
          f"{mount['target']}")
    print(f"  mount opts {mount['options']}")
    print(f"  kernel     {machine['kernel_release']}")
    print(f"  queue      {plan['queue']} threads issuing blocking preadv — "
          f"NOT io_uring")
    print(f"  matrix     {len(cases)} cases x "
          f"{args.warmup} warmup + {args.repeats} scored")
    print(f"  command    {machine['command_line']}")

    frag = {}
    print("\nextent geometry (filefrag -v):")
    for f in files:
        stats = filefrag_stats(f["path"])
        frag[f["path"]] = stats
        if stats["ok"]:
            print(f"  {f['name']:<10} {stats['extents']:>5} extents  "
                  f"mean {stats['mean_extent_bytes']:>9} B  "
                  f"median {stats['median_extent_bytes']:>9} B  "
                  f"adjacent {stats['adjacent_pairs']}/"
                  f"{max(stats['extents'] - 1, 0)}  "
                  f"compressed {stats['encoded_extents']}")
        else:
            print(f"  {f['name']:<10} filefrag failed: {stats['reason']}")

    session_btrfs_before = btrfs_device_stats(mount["target"])
    if session_btrfs_before["ok"]:
        for dev, counters in session_btrfs_before["devices"].items():
            print(f"\nbtrfs device stats before: {dev} "
                  + " ".join(f"{k}={counters.get(k)}" for k in BTRFS_COUNTERS))
    elif mount["fstype"] == "btrfs":
        print(f"\nwarning: btrfs device stats unreadable "
              f"({session_btrfs_before['reason']}) — a device-error delta "
              f"cannot be ruled out and every run will be DIRTY",
              file=sys.stderr)

    runs = []
    for i in range(args.warmup + args.repeats):
        label = "warmup, discarded" if i < args.warmup else "scored"
        run = one_run(args, plan, i, label, mount["fstype"])
        run["label"] = label
        runs.append(run)

    session_btrfs_after = btrfs_device_stats(mount["target"])
    session_grew = btrfs_delta(session_btrfs_before, session_btrfs_after)

    scored = [r for r in runs if r["label"] == "scored"]
    if not scored:
        fail(f"no scored runs out of {len(runs)} — nothing was measured, so "
             f"there is no hygiene verdict to give")
    dirty = [r for r in scored if r["hygiene"] != "CLEAN"]
    verdict = "PASS" if not dirty and not session_grew else "DIRTY"

    agg = aggregate(scored)

    print("\n=== summary ===")
    print(f"scored runs: {len(scored)}  clean: {len(scored) - len(dirty)}  "
          f"dirty: {len(dirty)}")
    print(f"queue model: {plan['queue']} (threads issuing blocking preadv; "
          f"NOT io_uring — do not compare against the runtime's numbers "
          f"without this caveat)")
    if session_grew:
        print("btrfs device stats grew across the session: "
              + "; ".join(session_grew))
    else:
        print("btrfs device stats: unchanged across the session")

    print(f"\nblock-size sweep at QD={args.fixed_qd} (GB/s, median of "
          f"{len(scored)}):")
    print(f"  {'K':>3} {'bytes':>10} {'MiB':>7}  {'file':<10} "
          f"{'seq':>8} {'rand':>8}  {'seq/rand':>9}  {'rand p50 ms':>12}")
    for k in ks:
        for f in files:
            if f["n_experts"] // k == 0:
                continue
            seq = agg.get((f["name"], "seq", k, args.fixed_qd))
            rnd = agg.get((f["name"], "rand", k, args.fixed_qd))
            block = k * f["stride"]
            print(f"  {k:>3} {block:>10} {block / 2**20:>7.2f}  {f['name']:<10} "
                  f"{(seq['gb_s_median'] if seq else 0):>8.3f} "
                  f"{(rnd['gb_s_median'] if rnd else 0):>8.3f}  "
                  f"{((seq['gb_s_median'] / rnd['gb_s_median']) if seq and rnd and rnd['gb_s_median'] else 0):>8.2f}x  "
                  f"{(rnd['p50_ms_median'] if rnd else 0):>12.3f}")

    print(f"\nqueue-depth sweep at K={args.fixed_k} (GB/s, median of "
          f"{len(scored)}):")
    print(f"  {'QD':>3}  {'file':<10} {'seq':>8} {'rand':>8}  {'seq/rand':>9}")
    for qd in qds:
        for f in files:
            seq = agg.get((f["name"], "seq", args.fixed_k, qd))
            rnd = agg.get((f["name"], "rand", args.fixed_k, qd))
            print(f"  {qd:>3}  {f['name']:<10} "
                  f"{(seq['gb_s_median'] if seq else 0):>8.3f} "
                  f"{(rnd['gb_s_median'] if rnd else 0):>8.3f}  "
                  f"{((seq['gb_s_median'] / rnd['gb_s_median']) if seq and rnd and rnd['gb_s_median'] else 0):>8.2f}x")

    print(f"\nheadline, per file, at QD={args.fixed_qd} — the question phase 6 "
          f"asked:")
    for f in files:
        r1 = agg.get((f["name"], "rand", 1, args.fixed_qd))
        r8 = agg.get((f["name"], "rand", args.fixed_k, args.fixed_qd))
        s8 = agg.get((f["name"], "seq", args.fixed_k, args.fixed_qd))
        if not (r1 and r8 and s8):
            continue
        a, b, c = r1["gb_s_median"], r8["gb_s_median"], s8["gb_s_median"]
        print(f"  {f['name']:<10} stride {f['stride']:>9}  "
              f"rand K=1 {a:.3f} -> rand K={args.fixed_k} {b:.3f} "
              f"({b / a if a else 0:.2f}x granularity) -> "
              f"seq K={args.fixed_k} {c:.3f} "
              f"({c / b if b else 0:.2f}x sequentiality) = "
              f"{c / a if a else 0:.2f}x combined")

    for stride in classes:
        members = [f for f in files if f["stride"] == stride]
        rates = [agg[(f["name"], "seq", args.fixed_k, args.fixed_qd)]["gb_s_median"]
                 for f in members
                 if (f["name"], "seq", args.fixed_k, args.fixed_qd) in agg]
        if rates:
            print(f"  stride {stride}: seq K={args.fixed_k} across "
                  f"{len(rates)} files = {statistics.fmean(rates):.3f} GB/s "
                  f"[min {min(rates):.3f}, max {max(rates):.3f}] — the spread "
                  f"is the point, not the mean")

    markdown = build_markdown(plan, agg, frag, machine, verdict)
    print("\n=== markdown (paste into docs/experiments/README.md) ===\n")
    print(markdown)

    print(f"\nmeasurement hygiene: {verdict}"
          + ("" if verdict == "PASS" else
             " — reclaim, a cgroup limit, a page cache that grew under "
             "O_DIRECT, a btrfs counter delta, or a counter the verdict "
             "depends on was left unreadable (unknown is DIRTY, not clean); "
             "see the [hard] lines above. Do not publish these numbers"))

    summary = {
        "machine": machine,
        "plan": {k: v for k, v in plan.items() if k != "cases"},
        "cases": cases,
        "filefrag": frag,
        "btrfs_session_before": session_btrfs_before,
        "btrfs_session_after": session_btrfs_after,
        "btrfs_session_grew": session_grew,
        "hygiene": verdict,
        "aggregate": [
            {"file": k[0], "pattern": k[1], "k": k[2], "qd": k[3],
             "block_bytes": v["template"]["block_bytes"],
             "bytes": v["template"]["bytes"],
             "queue": v["template"]["queue"],
             "n": v["n"], "gb_s_median": v["gb_s_median"],
             "gb_s_min": v["gb_s_min"], "gb_s_max": v["gb_s_max"],
             "p50_ms_median": v["p50_ms_median"]}
            for k, v in sorted(agg.items())
        ],
        "runs": [{k: v for k, v in r.items() if k != "log"} for r in runs],
        "markdown": markdown,
    }
    out = args.json or os.path.join(args.workdir, "summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")
    if args.markdown:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.markdown)) or ".",
            exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(markdown + "\n")
        print(f"wrote {args.markdown}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
