#!/usr/bin/env python3
"""Reproducible O_DIRECT bandwidth probe over the *installed* expert files.

`docs/experiments.md` lists re-measuring EXP-008 as the highest-value
item on the backlog, and EXP-008 cannot be re-run: its harness was never
committed. This file is that harness, written so the entry it feeds is
reproducible, self-describing, and rule-2 compliant.

## The question

Phase 6 wants to know whether reading a layer's experts **front-to-back in
large blocks** beats reading **individual experts at the 2.918 MiB stride**,
on these actual files, and by how much. EXP-008 answered a version of that
question with two probe series that disagree with each other by 17% at the
same queue depth, and EXP-013 then measured ~1.97 GB/s at the expert stride
under the real access pattern against EXP-008's 1.211-1.390 GB/s. The
prediction written here before the first run was that if EXP-013 is right,
EXP-008's "+51% at 16 MiB" premise collapses to roughly +9%.

**Measured, it collapsed further than that: see EXP-019.** Sequentially,
going from one expert blob to eight is neutral on one probed file (1.578 to
1.553 GB/s) and 15 to 16 percent *worse* on the other three. So +51% does not
survive as a premise at all, and the block-size curve behind it is refuted
rather than rescaled. The level is also higher across the board, 1.54 to 2.37
GB/s, consistent with EXP-013 and not with EXP-008.

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

`docs/benchmark-machine.md` also records that per-file bandwidth variance
exceeds run-to-run variance (`layer_00` 1,237 MB/s vs `layer_20` 1,812 MB/s).
This probe therefore runs a **fixed, recorded** file set spanning both stride
classes and **always reports per file**. Aggregates are printed only with the
per-file min and max beside them; an aggregate that hides a 1.46x spread is a
worse answer than no aggregate.

Measured, that spread reproduces (1.58 vs 2.27 GB/s) and the fragmentation
hypothesis above is **eliminated as its cause**: EXP-019 finds `layer_00` and
`layer_20` byte-identical in extent geometry, 398 extents each, mean 984,027
B, zero physically adjacent successor pairs. The extent map is still worth
recording, because that is how the hypothesis got eliminated, but what is left
is physical placement or drive-internal behaviour that this probe cannot see.

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

## Phase 9: the three things the four-file matrix could not answer

EXP-023 measured a **2.21x** per-file bandwidth spread at decode's K=1 block
size, and `docs/benchmark-machine.md` concluded "it is not fragmentation" from
the fact that the fast and slow files have identical extent *geometry*. Both
statements have a hole in them, and three modes were added to close them.

**1. Geometry is not placement.** `filefrag_stats` measured extent count,
sizes and successor adjacency and nothing else. Two files can match on all
three and still differ by two orders of magnitude in how far apart their
extents sit. `dispersion_stats` adds physical span, region count under a
configurable gap, the byte-weighted fraction in the largest region, the
median inter-extent seek in read order, and the byte-weighted mean position.
Measured on this install at a 256 MiB gap: 43 of 48 files hold >= 99% of
their bytes in one region, and `layer_00` — the slow one — holds **23.0%**
across 26 regions with a 72.5 GB median seek. That does not make placement
the cause; `layer_06` reads slowly while sitting 99.6% in one region. It
makes placement a live hypothesis that the geometry columns could not see.

Every one of these numbers is a **btrfs logical bytenr**, not a device LBA
(`LOGICAL_NOT_LBA`), and every table that prints one says so.

**2. Four files are a sample; the claim is about 48.** `--all-files` runs the
decode-shaped cell across every layer file in `layout.json` and emits per-file
bandwidth in the same record as per-file dispersion, so "decode pays the
spread on all 48" can be checked and bandwidth can be regressed on dispersion
instead of eyeballed against it.

**3. A correlation across files is not a mechanism.** Files differ in age, in
write history, in when the installer wrote them and in where they landed.
`--window` reads a bounded byte range *inside one file*, which removes every
one of those file-level differences from the comparison: both windows were
written by the same `install` invocation, at the same time, into the same
file. What it does **not** remove is per-NAND-block state. Two windows at
different offsets are on different NAND blocks by construction, so their SLC
residency and read-disturb histories are *unmeasured*, not held equal. The
window test therefore isolates "physical placement plus whatever varies from
one NAND block to the next" against the file-level confounds — a real
tightening of the cross-file comparison, and not the clean placement-only
control it would be if the two windows shared cells. `--list-regions` prints
the region map and ready-made `--window` specs, so picking the windows is not
a hand-parse of `filefrag -v`.

Positions inside a run are not equivalent — the first case of a run pays
btrfs extent-tree and allocator warm-up that the later ones do not — and a
window case issues few enough reads for that to matter. Window mode therefore
**interleaves** the dense and scattered populations so neither owns the head
of the run, and **permutes the order every repeat**; the permutation is
recorded per run (`case_execution_order`) and each case records the position
it ran at (`exec_position`), so a reader can check for a position artifact
instead of trusting that there is none.

## Output

JSON (everything, including per-read latency percentiles), a human summary,
and a markdown block ready to paste into an experiment entry. Bandwidth is
reported in **GB/s = 10^9 bytes/s**, matching EXP-008's units. Every JSON key
that existed before phase 9 still exists and still means the same thing; the
dispersion, all-files and window fields are additions.

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

`--list-regions` reads no data block and has no hygiene verdict, so it maps
the same three codes onto what its metadata pass managed to collect: 0 when
every requested file's extent map was read, 1 when some were and some were
not, and 2 when **none** were — no `filefrag` on PATH, an unreadable extent
tree, or an empty report. Exiting 0 on a report with nothing in it would let a
wrapper script record "no dispersion" as a finding.

Python stdlib only. Linux >= 6.5 + cgroup v2 + systemd --user, by design.

Usage:

  scripts/io_probe.py --dry-run
  scripts/io_probe.py --repeats 3 --json scratch/io-probe/summary.json

  # the region map and ready-made window specs (metadata only, no drive I/O)
  scripts/io_probe.py --list-regions layer_00 --window-scan-bytes 2M

  # the decode-shaped cell across every layer file, joined to dispersion
  scripts/io_probe.py --all-files --fixed-k 1 --fixed-qd 8 --patterns rand \\
      --repeats 3 --json scratch/io-probe/all48.json

  # dense vs scattered windows inside one file
  scripts/io_probe.py --window layer_00:182292480:2M,layer_00:142516224:2M \\
      --window-block-bytes 262144 --fixed-qd 8 --patterns rand --repeats 3
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
# docs/benchmark-machine.md names for per-file bandwidth variance (1,237 vs
# 1,812 MB/s; EXP-019 reproduces it and rules out fragmentation as the cause,
# since the two have identical extent geometry) and both carry the 3,059,712 B
# stride; layer_06 and layer_21 carry the 2,654,208 B stride. Changing this
# list changes what the numbers mean, so it is a default rather than a
# computed choice.
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

# Two extents belong to the same physical *region* when the hole between them
# is no larger than this. 256 MiB is a quarter of a btrfs data chunk (1 GiB),
# so it never merges across two chunks, and the region count it produces is
# only weakly sensitive to the threshold: measured on this install, layer_00
# lands on 54 / 26 / 19 regions at 64 MiB / 256 MiB / 1 GiB while layer_20
# lands on 5 / 5 / 3. The signal is the ratio, not the absolute count.
DEFAULT_CLUSTER_GAP = 256 * 2**20

# Window sizes the region lister scans for dense/scattered candidates. 2 MiB
# is the size at which a window that lies wholly inside one region exists on
# this install (measured: the longest single-region logical run in layer_00 is
# 2,174,976 B); 8 MiB is the size the phase 9 brief asks for.
DEFAULT_WINDOW_SCAN = "2M,8M"

# `filefrag` prints btrfs LOGICAL bytenr, not device LBA. Every dispersion
# number this script reports therefore describes the filesystem's logical
# address space. On a single-device `single`-profile filesystem the chunk map
# is monotone per chunk, so clustering survives the translation, but the
# translation is not done here and the distinction must be stated wherever a
# number is printed. This is the command that resolves it, for a human with
# sudo; the probe never runs it (no passwordless sudo on the reference box).
LOGICAL_NOT_LBA = ("btrfs LOGICAL bytenr from filefrag, NOT device LBA")


def chunk_tree_command(source: str | None, fstype: str | None = None) -> str:
    """The command a human runs to map btrfs logical bytenr to device LBA.

    `findmnt` reports the source of a btrfs subvolume mount as
    `/dev/nvme0n1p2[/@home]`; the chunk tree lives on the device, not the
    subvolume, so the bracket is stripped rather than pasted into a command
    that would fail.

    Only btrfs interposes a logical address space between the file and the
    device, so only btrfs needs the translation. When the caller knows the
    filesystem type and it is not btrfs, handing the user a `btrfs
    inspect-internal` command would be telling them to run something that
    cannot apply to their filesystem, so a note is returned instead. Callers
    that do not know the type get the btrfs form, which is what the reference
    machine runs.
    """
    if fstype is not None and fstype != "btrfs":
        return (f"n/a on {fstype}: filefrag already reports device physical "
                f"block numbers, so there is no filesystem-internal logical "
                f"space to translate out of")
    device = (source or "/dev/<device>").split("[", 1)[0]
    return f"sudo btrfs inspect-internal dump-tree -t 3 {device}"


def physical_units_note(fstype: str | None) -> str:
    """What `filefrag`'s physical column actually means on this filesystem.

    On btrfs it is a logical bytenr that the chunk tree still has to translate
    (`LOGICAL_NOT_LBA`). On ext4, XFS and friends there is no such indirection:
    the number is a genuine device block address, and telling the user
    otherwise — which this tool did until phase 9's review — misdescribes their
    own data. Neither is the address the drive's FTL ultimately reads, which is
    the caveat that survives on every filesystem.
    """
    if fstype == "btrfs":
        return LOGICAL_NOT_LBA
    return (f"{fstype or 'unknown-fs'} physical block address from filefrag "
            f"(a device address, not a filesystem-internal logical one; still "
            f"pre-FTL, so it is not what the NAND sees either)")


def dist(value: float | None) -> str:
    """A physical distance, in the unit that makes it readable. Distances here
    run from a few kilobytes to hundreds of gigabytes, and printing 0.00 GB
    for a window that fits inside one extent hides the whole result."""
    if value is None:
        return "-"
    if value >= 1e9:
        return f"{value / 1e9:.2f} GB"
    if value >= 1e6:
        return f"{value / 1e6:.2f} MB"
    return f"{value} B"

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


def filefrag_extents(path: str) -> dict:
    """Raw `filefrag -v` parse: every extent, in logical order, in bytes.

    Split out of `filefrag_stats` so the region lister and the window scanner
    can work on the same extents the statistics are computed from, without
    either of them re-parsing or the 398-entry extent list being dragged into
    every summary JSON.

    `physical` is a btrfs **logical** bytenr (see `LOGICAL_NOT_LBA`).
    """
    out = {"ok": False, "reason": None, "block_bytes": None, "extents": [],
           "flags_seen": [], "encoded_extents": 0, "unparsed_lines": 0,
           "filefrag_discontiguous": 0, "physical_units": LOGICAL_NOT_LBA}
    rc, text, err = run_tool(["filefrag", "-v", path])
    if rc != 0:
        out["reason"] = (err or text).strip() or f"rc={rc}"
        return out
    block_bytes = None
    extents: list[dict] = []
    flags_seen: set[str] = set()
    for line in text.splitlines():
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
            out["unparsed_lines"] += 1
            continue
        _, lstart, _lend, pstart, _pend, length, expected, flags = m.groups()
        extents.append({"logical": int(lstart), "physical": int(pstart),
                        "blocks": int(length)})
        if expected is not None:
            out["filefrag_discontiguous"] += 1
        for flag in (f.strip() for f in flags.split(",")):
            if flag:
                flags_seen.add(flag)
                if flag == "encoded":
                    out["encoded_extents"] += 1
    if not extents or block_bytes is None:
        out["reason"] = "no extents parsed"
        return out
    for e in extents:
        e["logical"] *= block_bytes
        e["physical"] *= block_bytes
        e["bytes"] = e.pop("blocks") * block_bytes
    extents.sort(key=lambda e: e["logical"])
    out.update({"ok": True, "block_bytes": block_bytes, "extents": extents,
                "flags_seen": sorted(flags_seen)})
    return out


def physical_regions(extents: list[dict], gap_bytes: int) -> list[dict]:
    """Group extents into physically contiguous-ish regions.

    A region ends where the hole to the next extent (in *physical* order)
    exceeds `gap_bytes`. Regions come back sorted by byte count, largest
    first, and every input extent gains a `region` key naming its index —
    that key is what the window scanner uses to say how many regions a
    candidate window straddles.

    These are btrfs logical addresses (`LOGICAL_NOT_LBA`).
    """
    if not extents:
        return []
    ordered = sorted(extents, key=lambda e: e["physical"])
    groups: list[list[dict]] = [[ordered[0]]]
    reach = ordered[0]["physical"] + ordered[0]["bytes"]
    for e in ordered[1:]:
        if e["physical"] - reach > gap_bytes:
            groups.append([e])
            reach = e["physical"] + e["bytes"]
        else:
            groups[-1].append(e)
            reach = max(reach, e["physical"] + e["bytes"])
    regions = []
    for members in groups:
        regions.append({
            "bytes": sum(m["bytes"] for m in members),
            "extents": len(members),
            "physical_start": min(m["physical"] for m in members),
            "physical_end": max(m["physical"] + m["bytes"] for m in members),
            "members": members,
        })
    regions.sort(key=lambda r: -r["bytes"])
    for index, region in enumerate(regions):
        region["index"] = index
        for member in region["members"]:
            member["region"] = index
    return regions


def merge_logical_ranges(members: list[dict]) -> list[tuple[int, int]]:
    """The logical byte ranges a region owns, adjacent ones merged."""
    ranges: list[tuple[int, int]] = []
    for m in sorted(members, key=lambda e: e["logical"]):
        start, end = m["logical"], m["logical"] + m["bytes"]
        if ranges and ranges[-1][1] == start:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))
    return ranges


def dispersion_stats(extents: list[dict], gap_bytes: int,
                     units: str = LOGICAL_NOT_LBA) -> dict:
    """Physical dispersion, the axis `filefrag_stats` never measured.

    `docs/benchmark-machine.md` rules fragmentation out as the cause of the
    per-file bandwidth spread on the grounds that the slow and fast files have
    identical extent *geometry* — same count, same sizes, zero physically
    adjacent successor pairs. That is true and it does rule geometry out. It
    does not rule out *placement*: two files can have the same 398 extents of
    the same sizes and still differ by two orders of magnitude in how far
    apart those extents sit. These are the numbers that tell them apart.

    Every distance here is in whatever address space `filefrag` reports for
    this filesystem — btrfs logical bytes by default (`LOGICAL_NOT_LBA`), or
    the device addresses `physical_units_note` describes when the caller knows
    the filesystem is not btrfs and says so.

    `median_seek_bytes`, `mean_seek_bytes` and `max_seek_bytes` stay `None`
    for a file with a single extent: there is no inter-extent seek to take a
    median of, and inventing a 0 would put a defragmented file and a file
    whose extents happen to abut in the same bucket. Every caller that prints
    them must therefore go through `dist()`/`gb()` or its own guard — a bare
    `/ 1e9` raises `TypeError` on the one-extent file that XFS and
    `btrfs filesystem defragment` both produce.
    """
    out = {
        "cluster_gap_bytes": gap_bytes,
        "physical_units": units,
        "physical_min_bytes": None, "physical_max_bytes": None,
        "physical_span_bytes": None, "regions": None,
        "largest_region_bytes": None, "largest_region_fraction": None,
        "median_seek_bytes": None, "mean_seek_bytes": None,
        "max_seek_bytes": None, "byte_weighted_mean_physical_bytes": None,
        "region_table": [],
    }
    if not extents:
        return out
    regions = physical_regions(extents, gap_bytes)
    total = sum(e["bytes"] for e in extents)
    # Seeks are taken in **logical (read) order**, because that is the order a
    # front-to-back read issues them in and therefore the distance the drive
    # actually travels; the physical-order gap is a different quantity and is
    # already captured by the region count.
    in_read_order = sorted(extents, key=lambda e: e["logical"])
    seeks = [abs(in_read_order[i + 1]["physical"]
                 - (in_read_order[i]["physical"] + in_read_order[i]["bytes"]))
             for i in range(len(in_read_order) - 1)]
    out.update({
        "physical_min_bytes": min(e["physical"] for e in extents),
        "physical_max_bytes": max(e["physical"] + e["bytes"] for e in extents),
        "regions": len(regions),
        "largest_region_bytes": regions[0]["bytes"],
        "largest_region_fraction": (round(regions[0]["bytes"] / total, 6)
                                    if total else None),
        "byte_weighted_mean_physical_bytes": int(
            sum((e["physical"] + e["bytes"] / 2) * e["bytes"]
                for e in extents) / total) if total else None,
        "region_table": [
            {"index": r["index"], "bytes": r["bytes"], "extents": r["extents"],
             "fraction": round(r["bytes"] / total, 6) if total else None,
             "physical_start": r["physical_start"],
             "physical_end": r["physical_end"],
             "logical_ranges": len(merge_logical_ranges(r["members"]))}
            for r in regions
        ],
    })
    out["physical_span_bytes"] = (out["physical_max_bytes"]
                                  - out["physical_min_bytes"])
    if seeks:
        out.update({
            "median_seek_bytes": int(statistics.median(seeks)),
            "mean_seek_bytes": int(statistics.fmean(seeks)),
            "max_seek_bytes": max(seeks),
        })
    return out


def filefrag_stats(path: str, cluster_gap: int = DEFAULT_CLUSTER_GAP,
                   parsed: dict | None = None,
                   units: str = LOGICAL_NOT_LBA) -> dict:
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

    Since phase 9 this also carries the **physical dispersion** fields from
    `dispersion_stats`, because extent geometry alone was being read as if it
    settled a question it cannot settle. Every existing key is unchanged; the
    dispersion keys are additions.
    """
    out_stats = {"ok": False, "reason": None, "extents": None,
                 "block_bytes": None, "mean_extent_bytes": None,
                 "median_extent_bytes": None, "min_extent_bytes": None,
                 "max_extent_bytes": None, "adjacent_pairs": None,
                 "adjacent_fraction": None, "filefrag_discontiguous": None,
                 "encoded_extents": 0, "unparsed_lines": 0, "flags_seen": []}
    out_stats.update(dispersion_stats([], cluster_gap, units))
    # A caller that already has the parse (the window scanner needs the extent
    # list anyway) passes it in rather than forking `filefrag` twice per file.
    parsed = parsed if parsed is not None else filefrag_extents(path)
    if not parsed["ok"]:
        out_stats["reason"] = parsed["reason"]
        out_stats["unparsed_lines"] = parsed["unparsed_lines"]
        return out_stats
    extents = parsed["extents"]
    block_bytes = parsed["block_bytes"]
    sizes = [e["bytes"] for e in extents]
    adjacent = sum(1 for i in range(len(extents) - 1)
                   if extents[i]["physical"] + extents[i]["bytes"]
                   == extents[i + 1]["physical"])
    out_stats.update({
        "ok": True,
        "extents": len(sizes),
        "block_bytes": block_bytes,
        "mean_extent_bytes": int(statistics.fmean(sizes)),
        "median_extent_bytes": int(statistics.median(sizes)),
        "min_extent_bytes": min(sizes),
        "max_extent_bytes": max(sizes),
        "adjacent_pairs": adjacent,
        "adjacent_fraction": (round(adjacent / (len(extents) - 1), 4)
                              if len(extents) > 1 else None),
        # filefrag's own view of the same fact: it prints an `expected:`
        # column exactly when an extent is not contiguous with its
        # predecessor. It should equal (extents - 1 - adjacent_pairs); a
        # disagreement means the parse is wrong, not the disk.
        "filefrag_discontiguous": parsed["filefrag_discontiguous"],
        "encoded_extents": parsed["encoded_extents"],
        "unparsed_lines": parsed["unparsed_lines"],
        "flags_seen": parsed["flags_seen"],
    })
    out_stats.update(dispersion_stats(extents, cluster_gap, units))
    return out_stats


# ---------------------------------------------------------------------------
# byte-window locality: what a bounded read inside one file actually straddles
# ---------------------------------------------------------------------------


def window_locality(extents: list[dict], offset: int, length: int,
                    units: str = LOGICAL_NOT_LBA) -> dict:
    """How physically dispersed the byte range [offset, offset+length) is.

    This is the number the byte-window experiment turns on. A dense window and
    a scattered window inside the **same file** share that file's age, its
    write history and the single `install` run that wrote both — every
    file-level confound a cross-file comparison carries. They do **not** share
    NAND cells: two ranges at different offsets are on different blocks by
    construction, so per-block SLC residency and read-disturb state are
    uncontrolled here, merely made irrelevant to the file-level story. Read
    the result accordingly: bandwidth tracking this column means placement or
    something that varies block to block, not placement alone.

    `extents` must carry a `region` key (i.e. have been through
    `physical_regions`). Distances are in the address space `units` names,
    btrfs logical by default (`LOGICAL_NOT_LBA`).
    """
    parts: list[tuple[int, int, int]] = []  # (physical, bytes, region)
    for e in extents:
        a = max(e["logical"], offset)
        b = min(e["logical"] + e["bytes"], offset + length)
        if b <= a:
            continue
        parts.append((e["physical"] + (a - e["logical"]), b - a,
                      e.get("region", 0)))
    if not parts:
        return {"ok": False, "reason": "no extent covers the window",
                "offset": offset, "length": length}
    covered = sum(n for _, n, _ in parts)
    by_region: dict[int, int] = {}
    for _, n, r in parts:
        by_region[r] = by_region.get(r, 0) + n
    seeks = [abs(parts[i + 1][0] - (parts[i][0] + parts[i][1]))
             for i in range(len(parts) - 1)]
    return {
        "ok": True,
        "reason": None,
        "offset": offset,
        "length": length,
        "covered_bytes": covered,
        "fragments": len(parts),
        "regions_touched": len(by_region),
        "region_indices": sorted(by_region),
        "dominant_region": max(by_region, key=lambda r: by_region[r]),
        "dominant_fraction": round(max(by_region.values()) / covered, 6),
        "physical_span_bytes": (max(p + n for p, n, _ in parts)
                                - min(p for p, _, _ in parts)),
        # A window inside one extent has no inter-fragment seek, and 0 is the
        # true answer there — unlike a whole file's `median_seek_bytes`, which
        # is None for the same shape because "one extent" and "extents that
        # abut" are different facts about a file and the same fact about a
        # window's read pattern.
        "median_seek_bytes": int(statistics.median(seeks)) if seeks else 0,
        "physical_units": units,
    }


def scan_windows(extents: list[dict], size: int, length: int, stride: int,
                 count: int, units: str = LOGICAL_NOT_LBA) -> dict:
    """Rank candidate byte windows of `length` by how physically dense they are.

    Candidate offsets are the extent starts plus every `stride` multiple, not
    every 4 KiB position: a window boundary only matters where an extent
    boundary is, and stride multiples are included so a chosen window can be
    read at exact expert-blob boundaries. Ranking key is physical span, then
    region count — span is the quantity the experiment varies.

    Returns `count` mutually **non-overlapping** densest and scattered
    candidates, so several of each can be measured in one run without reading
    any byte twice.
    """
    if length <= 0 or length > size:
        return {"length": length, "candidates": 0, "dense": [], "scattered": []}
    offsets = {e["logical"] for e in extents}
    offsets.update(i * stride for i in range(size // stride + 1))
    ranked = []
    for off in sorted(offsets):
        if off % DIO_ALIGN or off < 0 or off + length > size:
            continue
        loc = window_locality(extents, off, length, units)
        if loc["ok"]:
            ranked.append(loc)
    ranked.sort(key=lambda w: (w["physical_span_bytes"], w["regions_touched"],
                               w["offset"]))

    def disjoint(pool: list[dict], seed: list[dict]) -> list[dict]:
        taken: list[dict] = []
        for w in pool:
            if all(w["offset"] + length <= t["offset"]
                   or t["offset"] + length <= w["offset"]
                   for t in taken + seed):
                taken.append(w)
            if len(taken) >= count:
                break
        return taken

    # The scattered picks are chosen against the dense picks as well as
    # against each other, so the two sets can be handed to one `--window`
    # invocation without any byte being read twice in a run.
    dense = disjoint(ranked, [])
    return {"length": length, "candidates": len(ranked), "dense": dense,
            "scattered": disjoint(list(reversed(ranked)), dense)}


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
    # Window cases are distinguished by their offset and block size, appended
    # only when they exist so that the seeds of every pre-phase-9 case — and
    # therefore every published run's access order — are bit-identical.
    # Membership, not truthiness: `scan_windows` emits offset 0 for the densest
    # window of a file whose first extent is large, and `.get(...)` being falsy
    # there would have handed that window the *whole-file* seed for the same
    # (file, pattern, K, QD). Whole-file cases carry no `offset_base` key at
    # all, so their seeds are untouched by the change.
    if "offset_base" in case:
        key += f"|{case['offset_base']}|{case['block_bytes']}"
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
    # `offset_base` is 0 for every whole-file case, so the offsets below are
    # exactly what they were before byte-window mode existed.
    base = case.get("offset_base", 0)
    offsets = [base + i * block for i in range(case["blocks"])]
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
    reported bandwidth is drive time, not page-fault time. **So does thread
    creation, and so does the join.** `threading.Thread.start()` costs on the
    order of 100 us apiece and `join()` costs a wakeup: negligible against a
    128-read whole-file case at ~250 ms, but on the order of a third of an
    8-read 2 MiB window case, which is exactly the measurement that was added
    to resolve a ~10% difference. Worse, the cost is additive and near-equal
    across cases, so leaving it in compresses every ratio toward 1.00 — it
    cannot show a difference that is not there, but it reliably hides one that
    is. The threads are therefore parked on a barrier until every one of them
    is running; the clock starts when the barrier releases and stops at the
    last read's completion, not at the last thread's exit.
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
    finished: list[float] = []
    started: list[float] = []
    # qd workers + this thread. A worker that never arrives would hang the
    # run, so both sides wait with a timeout and a broken barrier aborts the
    # case loudly instead of silently measuring a subset of the queue. The
    # barrier's action runs once, in whichever party arrives last, *before*
    # any of them returns from `wait()` — so it is the one place that can
    # stamp a start time no thread has yet read a byte after.
    ready = threading.Barrier(
        qd + 1, action=lambda: started.append(time.perf_counter()))
    BARRIER_TIMEOUT_S = 60.0

    def worker(buf: AlignedBuffer) -> None:
        nonlocal short_reads
        local: list[float] = []
        local_short = 0
        local_errors: list[str] = []
        view = buf.view
        try:
            ready.wait(BARRIER_TIMEOUT_S)
        except threading.BrokenBarrierError:
            return
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
        done = time.perf_counter()
        with lock:
            latencies.extend(local)
            errors.extend(local_errors)
            short_reads += local_short
            # Each worker stamps its own finish, so `elapsed` can end at the
            # last completed read rather than after qd joins.
            finished.append(done)

    threads = [threading.Thread(target=worker, args=(buf,), daemon=True)
               for buf in buffers]
    for t in threads:
        t.start()
    try:
        ready.wait(BARRIER_TIMEOUT_S)
    except threading.BrokenBarrierError:
        ready.abort()
        for t in threads:
            t.join(BARRIER_TIMEOUT_S)
        for buf in buffers:
            buf.close()
        raise RuntimeError(
            f"not all {qd} reader threads reached the start barrier within "
            f"{BARRIER_TIMEOUT_S:.0f}s; refusing to report a bandwidth "
            f"measured by an unknown number of threads")
    # Every thread is running and past its own barrier wait, so nothing
    # between the barrier's stamp and the last stamp in `finished` is thread
    # bookkeeping.
    t0 = started[0]
    cpu0 = time.process_time()
    for t in threads:
        t.join()
    cpu = time.process_time() - cpu0
    # A case whose every worker died before reading leaves `finished` empty;
    # falling back to the join time keeps the arithmetic total, and `errors`
    # is what makes the run DIRTY.
    elapsed = (max(finished) - t0) if finished else (time.perf_counter() - t0)

    for buf in buffers:
        buf.close()

    total = len(offsets) * block_bytes
    return {
        "seconds": elapsed,
        # Provenance for the phase-9 review fix: a summary written before it
        # has no `timer` key and its `seconds` includes thread start and join.
        "timer": "barrier-release-to-last-read",
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

    # Which case runs in which position. The outer harness computes it (it is
    # the thing being varied across repeats) and hands it over in the plan; an
    # older plan file without the key runs in built order, which is what every
    # non-window mode uses anyway.
    order = plan.get("case_execution_order")
    if order is None:
        order = list(range(len(plan["cases"])))
    if sorted(order) != list(range(len(plan["cases"]))):
        fail(f"case_execution_order is not a permutation of the "
             f"{len(plan['cases'])} planned cases, so the run would measure "
             f"some cases twice and others not at all")

    # The window label is what tells sixteen window cases apart; without it
    # they log as sixteen visually identical lines differing only in the
    # bandwidth column. The width is taken from the cases actually planned, so
    # a whole-file run's log keeps the column positions it always had.
    def case_label(case: dict) -> str:
        if case.get("window_label"):
            return f"{case['file']}@{case['window_offset']}"
        return case["file"]

    label_width = max([9] + [len(case_label(c)) for c in plan["cases"]])

    cases: list[dict] = []
    t_start = time.time()
    fds: dict[str, int] = {}
    try:
        for position, index in enumerate(order):
            case = plan["cases"][index]
            path = case["path"]
            if path not in fds:
                fds[path] = open_direct(path)
            rng = random.Random(case_seed(plan["seed"], plan["repeat"], case))
            offsets = case_offsets(case, rng)
            measured = timed_reads(fds[path], offsets, case["block_bytes"],
                                   case["qd"])
            record = dict(case)
            record.update(measured)
            # Position in *this* run, not the case's canonical index: the two
            # differ in window mode, and a position artifact is only auditable
            # if the position is on the record.
            record["exec_position"] = position
            cases.append(record)
            # `[n]` is the case's canonical index (the one --dry-run prints),
            # not its position: the log is emitted in execution order, so the
            # position is the line number and the index is the join key.
            log(f"  [{case['order']:>3}] {case_label(case):<{label_width}} "
                f"{case['pattern']:<4} K={case['k']:<2} "
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
        # The order this run actually executed in, echoed back from the plan so
        # the record is self-contained: a reader holding one run's JSON can
        # check for a position artifact without the plan file beside it.
        "case_execution_order": order,
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


def check_queue_depth(what: str, blocks: int, qd: int, hint: str) -> None:
    """Refuse a case whose queue depth it cannot actually reach.

    `timed_reads` starts `qd` threads against a queue of `blocks` reads, so
    with fewer blocks than threads the surplus threads take nothing and the
    real depth is `blocks` — while the plan, the log line, the table and the
    JSON all keep saying `qd`. That is not a small inaccuracy: `--window
    2M --window-block-bytes 1M --fixed-qd 8` runs at depth 2 and reports 8,
    and a queue-depth number that is wrong by 4x is worse than no number.

    Equality is fine (one read per thread). Whole-file cases cannot trip this
    at any sane setting — the smallest is 128/K blocks — so this fires
    essentially only for windows, which is where it is needed.
    """
    if qd > blocks:
        fail(f"{what} issues {blocks} reads but was asked for queue depth "
             f"{qd}: {qd - blocks} of the {qd} threads would get no work and "
             f"the real depth would be {blocks}, while every table and JSON "
             f"field would report {qd}. {hint}")


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
            check_queue_depth(
                f"{f['name']} at K={k}", blocks, qd,
                f"Lower the queue depth to {blocks} or below, or lower K.")
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


def build_all_files_cases(files: list[dict], k: int, qd: int,
                          patterns: list[str]) -> tuple[list[dict], list[tuple[int, int]]]:
    """One cell — (K, QD) — across **every** layer file.

    `docs/experiments.md` says decode "touches all 48 files every token
    so it pays that spread on all of them", but the spread it cites was
    measured on four files. Four is a sample, not the population, and the
    claim is about the population. This is the population: one decode-shaped
    case per file, so per-file bandwidth can be joined against per-file
    dispersion and regressed instead of asserted.

    Only the fixed cell is built. Crossing 48 files with the block-size and
    queue-depth sweeps would be 48x the wall time to answer a question nobody
    asked, and the sweeps already have their four-file answer.
    """
    cases = []
    order = 0
    for f in files:
        blocks = f["n_experts"] // k
        if blocks == 0:
            continue
        block_bytes = k * f["stride"]
        check_queue_depth(
            f"{f['name']} at K={k}", blocks, qd,
            f"Lower --fixed-qd to {blocks} or below, or lower --fixed-k.")
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
    return cases, [(k, qd)]


def parse_window_specs(text: str) -> list[tuple[str, int, int]]:
    """`file:offset:length[,file:offset:length...]`, sizes accept K/M/G."""
    specs = []
    for raw in (s.strip() for s in text.split(",")):
        if not raw:
            continue
        bits = raw.split(":")
        if len(bits) != 3:
            fail(f"--window {raw!r} is not FILE:OFFSET:LENGTH (for example "
                 f"layer_00:182292480:2M). Run --list-regions to get "
                 f"ready-made specs.")
        try:
            offset, length = parse_size(bits[1]), parse_size(bits[2])
        except ValueError:
            fail(f"--window {raw!r} has a non-numeric offset or length")
        specs.append((bits[0].strip(), offset, length))
    if not specs:
        fail("--window was given no specs")
    return specs


def build_window_cases(files: list[dict], specs: list[tuple[str, int, int]],
                       k: int, qd: int, patterns: list[str],
                       block_override: int | None, extent_map: dict,
                       units: str = LOGICAL_NOT_LBA
                       ) -> tuple[list[dict], list[tuple[int, int]]]:
    """One case per (window, pattern): a bounded read inside a single file.

    The discriminating experiment. Two windows in the same file share the
    file's age, its write history and the single `install` run that wrote
    both, so every file-level confound that a cross-file comparison carries is
    gone. They sit on different NAND blocks by construction, so per-block SLC
    residency and read-disturb state are *not* held equal — the control is
    over the file-level history, not over the cells. If a window whose bytes
    sit in one physical region reads faster than one whose bytes are scattered
    across regions, the difference is placement or something that varies from
    block to block; if they read the same, neither is moving the number and
    the per-file spread has to come from somewhere else again.

    The window's locality is computed here and stored **in the case**, so the
    result record carries the bandwidth and the geometry it is supposed to be
    explained by in one row.

    ## Case order is part of the experiment

    Positions in a run are not exchangeable: the first case pays extent-tree
    and allocator warm-up the rest do not, and in the committed
    `scratch/io-probe/p9-win2m.log` the case that happened to be built first —
    the densest window — was the slowest of all sixteen in **every** scored
    run, which moved the printed dense/scattered ratio from 1.19x to 1.10x. A
    128-read whole-file case can absorb that; an 8-read window case cannot.

    So the built order is not the given order. Windows are ranked by physical
    span and the dense half is **interleaved** with the scattered half, which
    makes it impossible for either population to own the head or the tail of
    the run. `execution_order` then permutes that list differently every
    repeat, so no case keeps a position across runs either. Both are recorded
    (`window_rank`, `window_population`, `case_order_policy`,
    `case_execution_order`, `exec_position`) so the audit does not depend on
    reading this docstring.
    """
    by_name = {f["name"]: f for f in files}
    # Reading the same bytes twice in one run means the second read is served
    # from the drive's own cache, which this harness does not control and
    # cannot evict — `posix_fadvise` reaches the page cache, not the SSD's
    # DRAM. Both ways of doing it are refused rather than warned about,
    # because the resulting number looks entirely plausible.
    for i, (a_name, a_off, a_len) in enumerate(specs):
        for b_name, b_off, b_len in specs[i + 1:]:
            if a_name == b_name and a_off < b_off + b_len and b_off < a_off + a_len:
                fail(f"windows {a_name}:{a_off}:{a_len} and "
                     f"{b_name}:{b_off}:{b_len} overlap, so the overlapping "
                     f"bytes would be read twice in one run and the second "
                     f"read would be a drive-cache hit, not a cold read. Pick "
                     f"disjoint windows — --list-regions only ever emits "
                     f"non-overlapping candidates.")
    if len(patterns) > 1:
        fail(f"--window with --patterns {','.join(patterns)} reads every "
             f"window once per pattern, a few milliseconds apart, in the same "
             f"run: the second pattern measures the drive's read cache, not "
             f"the drive. (The default is '{','.join(PATTERNS)}', so this "
             f"fires unless --patterns is given.) Run one pattern per "
             f"invocation — 'rand' is the shape decode issues, and it is what "
             f"the committed phase 9 window run used.")
    # Built in the order the specs were given, then handed to
    # `interleave_window_cases`, which decides the order they are planned in.
    built: list[dict] = []
    order = 0
    for name, offset, length in specs:
        f = by_name.get(name)
        if f is None:
            fail(f"--window names {name!r}, which is not in the probed file "
                 f"set {sorted(by_name)}")
        if offset % DIO_ALIGN or length % DIO_ALIGN:
            fail(f"window {name}:{offset}:{length} is not {DIO_ALIGN}-aligned; "
                 f"O_DIRECT would be refused or silently downgraded")
        if offset < 0 or offset + length > f["size"]:
            fail(f"window {name}:{offset}:{length} runs past the end of "
                 f"{f['name']} ({f['size']} bytes)")
        # `is None`, not `or`: --window-block-bytes 0 is falsy and would fall
        # back to K x stride while the plan, the table and the JSON all
        # reported the flag as having supplied the block size. Zero is
        # rejected up front in main(), and this keeps the two facts in
        # agreement even if that check ever moves.
        block_bytes = (k * f["stride"] if block_override is None
                       else block_override)
        if block_bytes % DIO_ALIGN:
            fail(f"--window-block-bytes {block_bytes} is not a multiple of "
                 f"{DIO_ALIGN}")
        blocks = length // block_bytes
        if blocks < 1:
            fail(f"window {name}:{offset}:{length} is smaller than one "
                 f"{block_bytes}-byte block. Either widen the window or set "
                 f"--window-block-bytes below {length}.")
        check_queue_depth(
            f"window {name}:{offset}:{length} at {block_bytes} B blocks",
            blocks, qd,
            f"Lower --fixed-qd to {blocks} or below, or lower "
            f"--window-block-bytes so the window holds at least {qd} blocks "
            f"({length // qd} B or less would give {qd}).")
        if block_override is None and offset % f["stride"]:
            print(f"warning: window offset {offset} is not a multiple of "
                  f"{f['name']}'s {f['stride']}-byte expert stride, so its "
                  f"K={k} reads do not land on blob boundaries. That is legal "
                  f"for O_DIRECT and fine for a locality test, but it is not "
                  f"the shape decode issues.", file=sys.stderr)
        extents = extent_map.get(f["path"])
        if extents:
            loc = window_locality(extents, offset, blocks * block_bytes, units)
        else:
            loc = {"ok": False,
                   "reason": "filefrag gave no extent map for this file, so "
                             "the window's physical locality is unknown and "
                             "the result cannot be regressed on it"}
        label = f"{offset}+{blocks * block_bytes}"
        for pattern in patterns:
            built.append({
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
                "block_from_k": block_override is None,
                "blocks": blocks,
                "experts_covered": None,
                "bytes": blocks * block_bytes,
                "queue": "threaded-pread",
                "offset_base": offset,
                "window_label": label,
                "window_offset": offset,
                "window_length": length,
                "window_bytes_unread": length - blocks * block_bytes,
                "window_locality": loc,
            })
            order += 1

    cases = interleave_window_cases(built)
    return cases, [(k, qd)]


def interleave_window_cases(built: list[dict]) -> list[dict]:
    """Rank window cases by physical span and alternate the two populations.

    Split out of `build_window_cases` so it can be reasoned about — and read —
    on its own, because it is the thing standing between the experiment and a
    position artifact.

    Cases are ranked by the physical span of their window, densest first. The
    denser half and the more scattered half are then taken alternately, so the
    built list reads dense, scattered, dense, scattered, ... An odd case out
    (an odd number of windows) falls in the scattered half, where it is the
    least dense of that half and therefore the least misleading place for it.
    Cases whose locality could not be computed cannot be ranked at all and are
    appended last, labelled `unknown`, so they never silently pad one
    population.

    Every case comes back carrying `window_rank` (0 = densest) and
    `window_population`, and `order` is renumbered to the position in the
    returned list — `order` is the canonical index the plan prints and the log
    joins on, while the position a case *runs* at is `exec_position` and
    changes every repeat.
    """
    def span(case: dict) -> int | None:
        loc = case.get("window_locality") or {}
        return loc.get("physical_span_bytes") if loc.get("ok") else None

    rankable = [c for c in built if span(c) is not None]
    unrankable = [c for c in built if span(c) is None]
    # Ties are certain: several dense windows can sit inside one extent and
    # span exactly `length - 1` bytes apart. Offset then order break them, so
    # the ranking is total and reproducible rather than input-order dependent.
    rankable.sort(key=lambda c: (span(c), c["window_offset"], c["order"]))
    for rank, case in enumerate(rankable):
        case["window_rank"] = rank
    # One window is not two populations. Calling the only case "scattered"
    # because it landed in the upper half of a one-element list would be the
    # table asserting a contrast that was never measured.
    half = len(rankable) // 2
    if len(rankable) < 2:
        # Still woven (as the whole of one side), just not labelled as a
        # population: there is nothing for it to contrast with.
        dense, scattered = rankable, []
        for case in rankable:
            case["window_population"] = "unpaired"
    else:
        dense, scattered = rankable[:half], rankable[half:]
        for case in dense:
            case["window_population"] = "dense"
        for case in scattered:
            case["window_population"] = "scattered"
    for case in unrankable:
        case["window_rank"] = None
        case["window_population"] = "unknown"

    woven: list[dict] = []
    for i in range(max(len(dense), len(scattered))):
        if i < len(dense):
            woven.append(dense[i])
        if i < len(scattered):
            woven.append(scattered[i])
    woven.extend(unrankable)
    for index, case in enumerate(woven):
        case["order"] = index
    return woven


def execution_order(cases: list[dict], mode: str, repeat: int) -> list[int]:
    """Case indices in the order run `repeat` should execute them.

    Identity for every whole-file mode: those cases issue 128 reads each, the
    warm-up a leading position costs is a fraction of a percent of that, and
    changing the order would change what every pre-phase-9 summary is
    comparable against for no measurable gain.

    Window mode rotates the interleaved list left by the run index, so case
    `i` runs at position `(i - repeat) mod n`. That is the property worth
    having and it is worth having exactly: **no case occupies the same
    position in two runs** while there are fewer runs than cases, so a
    per-position cost cannot accumulate onto one case. Rotation also preserves
    the interleave, so the head of every run still alternates between the two
    populations as `repeat` advances.

    Deliberately not a shuffle, and deliberately not a rotation with a
    reflection folded in. A shuffle asks the reader to trust a PRNG where a
    rotation can be checked by eye against `case_execution_order`. A
    reflection looks like it adds disorder and does the opposite: it maps case
    `i` to `n - 1 - ((i - repeat) mod n)`, which puts case 0 at position 0 in
    both run 0 and run 1, and for two cases it collapses to no permutation at
    all (`reverse(rotate([0, 1], 1))` is `[0, 1]`).

    What rotation does not break is the neighbour relation — case `i` always
    follows case `i - 1`. The interleave is what covers that: each dense case
    is preceded by a scattered one and vice versa, so any read-ahead or
    drive-cache carry-over from the previous case lands on both populations
    equally.

    The warm-up run gets index 0 and therefore the built order, so `--dry-run`
    and the first log agree with the plan.
    """
    n = len(cases)
    if mode != "window" or n < 2:
        return list(range(n))
    rotate = repeat % n
    return list(range(rotate, n)) + list(range(rotate))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def case_key(case: dict) -> tuple:
    """Whole-file cases keep their four-element key so every existing lookup
    and every previously written summary still matches. Window cases append
    their label, because two windows in one file share (file, pattern, K, QD)
    and would otherwise be averaged into each other — which is precisely the
    difference the experiment exists to see."""
    base = (case["file"], case["pattern"], case["k"], case["qd"])
    label = case.get("window_label")
    return base + (label,) if label else base


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
            # One position per scored run, in run order. Identical for every
            # case in a whole-file mode; the whole point in window mode.
            "exec_positions": [e.get("exec_position") for e in entries],
        }
    return out


def gb(value: float | None) -> str:
    """An absolute byte *position* in GB (10^9). Positions on this filesystem
    are always tens to hundreds of GB, so a fixed unit reads better than
    `dist()`'s adaptive one — and keeps a column of them comparable by eye."""
    return "-" if value is None else f"{value / 1e9:.2f}"


def pct_str(value: float | None) -> str:
    """A fraction as a percentage, or a dash. Every dispersion fraction can be
    `None` (no extents, or a file too small to have a second one), and a table
    is not allowed to raise on the file that is missing the interesting
    column."""
    return "-" if value is None else f"{value * 100:.1f}%"


def frag_row(name: str, size: int, s: dict) -> list:
    """One file's geometry *and* dispersion, in the order both tables use.

    Distance columns go through `dist()`, not `gb()`: a single-extent file — an
    XFS install, or any file after `btrfs filesystem defragment` — has a span
    of a few hundred MB and no inter-extent seek at all, and a hard GB column
    renders both as `0.00` when the whole point of the column is to tell them
    apart. The byte-weighted mean position stays in GB because it is a
    position, not a distance.
    """
    return [
        name, size, s.get("extents", "-"), s.get("mean_extent_bytes", "-"),
        s.get("median_extent_bytes", "-"), s.get("adjacent_pairs", "-"),
        pct_str(s.get("adjacent_fraction")), s.get("encoded_extents", "-"),
        dist(s.get("physical_span_bytes")), s.get("regions", "-"),
        pct_str(s.get("largest_region_fraction")),
        dist(s.get("median_seek_bytes")),
        gb(s.get("byte_weighted_mean_physical_bytes")),
    ]


FRAG_HEADER = ["file", "bytes", "extents", "mean extent B", "median extent B",
               "physically adjacent pairs", "adjacent %", "compressed extents",
               "phys span", "regions", "largest region %",
               "median seek", "byte-wtd mean pos GB"]


def dispersion_bandwidth_rows(plan: dict, agg: dict[tuple, dict],
                              frag: dict) -> list[list]:
    """The join the phase 9 question needs: one row per (file, pattern) at the
    fixed cell, bandwidth next to dispersion, ready to regress.

    Distance columns use `dist()` for the same reason `frag_row` does: a file
    with one extent has no seek to report and must not be printed as a file
    with a zero-length one.
    """
    rows = []
    for f in plan["files"]:
        s = frag.get(f["path"], {})
        for pattern in plan["patterns"]:
            entry = agg.get((f["name"], pattern, plan["fixed_k"],
                             plan["fixed_qd"]))
            if not entry:
                continue
            rows.append([
                f["name"], f["layer"], f["stride"], pattern,
                f"{entry['gb_s_median']:.3f}",
                f"{entry['gb_s_min']:.3f}-{entry['gb_s_max']:.3f}",
                f"{entry['p50_ms_median']:.3f}",
                s.get("extents", "-"), dist(s.get("physical_span_bytes")),
                s.get("regions", "-"),
                pct_str(s.get("largest_region_fraction")),
                dist(s.get("median_seek_bytes")),
            ])
    return rows


DISPERSION_BW_HEADER = ["file", "layer", "stride", "pattern", "GB/s median",
                        "GB/s min-max", "p50 ms", "extents", "phys span",
                        "regions", "largest region %", "median seek"]


def window_rows(plan: dict, agg: dict[tuple, dict]) -> list[list]:
    """One row per window case: bandwidth beside the locality it is meant to be
    explained by, in the plan's canonical (interleaved) order.

    The population and rank columns are what make a position artifact visible
    from the table alone — `dense` and `scattered` alternate down the rows, so
    a column that tracks the row number rather than the population is reading
    as a run-order effect, not a placement effect. The per-read p50 is beside
    the bandwidth for the same reason: a window this small issues single-digit
    reads, and a bandwidth ratio that the latency ratio does not corroborate
    is measuring something other than the drive.
    """
    rows = []
    for case in plan["cases"]:
        entry = agg.get(case_key(case))
        if not entry:
            continue
        loc = case.get("window_locality") or {}
        rows.append([
            case["order"], case.get("window_population", "-"),
            case["file"], case["window_offset"],
            f"{case['window_length'] / 2**20:.2f}",
            case["block_bytes"], case["blocks"], case["pattern"],
            f"{entry['gb_s_median']:.3f}",
            f"{entry['gb_s_min']:.3f}-{entry['gb_s_max']:.3f}",
            f"{entry['p50_ms_median']:.3f}",
            loc.get("regions_touched", "-"),
            # `dist`, not `gb`: a window inside one extent spans kilobytes and
            # one across regions spans hundreds of gigabytes. Forcing both
            # into a GB column prints the interesting one as 0.00.
            dist(loc.get("physical_span_bytes")),
            pct_str(loc.get("dominant_fraction")),
            dist(loc.get("median_seek_bytes")),
        ])
    return rows


WINDOW_HEADER = ["case", "population", "file", "window offset", "window MiB",
                 "block B", "blocks", "pattern", "GB/s median",
                 "GB/s min-max", "p50 ms", "regions touched", "phys span",
                 "dominant region %", "median seek"]


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
    fstype = machine["mount"]["fstype"]
    if fstype == "btrfs":
        resolve = chunk_tree_command(machine["mount"]["source"], fstype)
        out.append(f"Dispersion columns are **{LOGICAL_NOT_LBA}**. To resolve "
                   f"logical to device LBA a human with sudo runs "
                   f"`{resolve}`; this probe never does. On a single-device "
                   f"`single`-profile filesystem the chunk map is monotone "
                   f"within a chunk, so a clustering signal survives the "
                   f"translation, but an absolute LBA does not.")
    else:
        out.append(f"Dispersion columns are "
                   f"**{physical_units_note(fstype)}**. There is no "
                   f"filesystem-internal logical space to resolve here, "
                   f"unlike btrfs; the remaining indirection is the drive's "
                   f"own FTL, which no host-side tool can see.")
    out.append("")

    def frag_table(title: str) -> None:
        out.append(title)
        out.append("")
        out.append(md_table(FRAG_HEADER,
                            [frag_row(f["name"], f["size"],
                                      frag.get(f["path"], {}))
                             for f in files]))

    if plan.get("mode") == "window":
        out.append("**Byte-window locality test** — bounded reads inside a "
                   "single file. What this holds constant is the file: both "
                   "windows were written by the same `install` run, at the "
                   "same time, into the same inode, so age, write history and "
                   "install order — every confound a cross-file comparison "
                   "carries — are identical. What it does **not** hold "
                   "constant is the NAND: two ranges at different offsets are "
                   "on different blocks by construction, so their SLC "
                   "residency and read-disturb histories are unmeasured, not "
                   "equal. A dense window that reads faster therefore points "
                   "at physical placement *or* at something that varies from "
                   "block to block; no difference says neither is moving the "
                   "number, and the per-file spread stays unexplained by "
                   "anything the filesystem can see.")
        out.append("")
        out.append(f"Case order is **{plan.get('case_order_policy', '-')}**: "
                   f"the dense and scattered windows alternate in the built "
                   f"order so neither population owns the head of a run, and "
                   f"every repeat runs a different permutation "
                   f"(`case_execution_order` per run, `exec_position` per "
                   f"case, both in the JSON). Position matters here: these "
                   f"cases issue single-digit numbers of reads, and in the "
                   f"pre-fix phase 9 run the first-built case was the slowest "
                   f"of sixteen in every scored run.")
        out.append("")
        out.append(md_table(WINDOW_HEADER, window_rows(plan, agg)))
        out.append("")
        frag_table("**Whole-file extent geometry and physical dispersion of "
                   "the files the windows live in** (`filefrag -v`)")
        return "\n".join(out)

    if plan.get("mode") == "all-files":
        out.append(f"**Decode-shaped cell (K={fixed_k}, QD={fixed_qd}) across "
                   f"all {len(files)} expert layer files, joined to physical "
                   f"dispersion** — the four-file spread in EXP-023 was a "
                   f"sample; this is the population, so 'decode pays the "
                   f"spread on all 48' can be checked rather than assumed, "
                   f"and bandwidth can be regressed on dispersion.")
        out.append("")
        out.append(md_table(DISPERSION_BW_HEADER,
                            dispersion_bandwidth_rows(plan, agg, frag)))
        return "\n".join(out)

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

    frag_table("**Extent geometry and physical dispersion of the probed "
               "files** (`filefrag -v`) — the installer writes each "
               "projection slab as its own CoW extent, so a 'sequential' read "
               "is physically scattered. Identical geometry across two files "
               "rules out *fragmentation*; it does not rule out *placement*, "
               "which is what the last five columns measure")
    return "\n".join(out)


def print_plan(plan: dict, assume_bw: float) -> None:
    mode = plan.get("mode", "matrix")
    print("=== plan (dry run: nothing is read, nothing is evicted) ===")
    print(f"mode:         {mode}")
    print(f"model:        {plan['rvmp']}")
    print(f"memory max:   {plan['memory_max']}  swap max: 0")
    print(f"queue model:  {plan['queue']} "
          f"(NOT io_uring — see the module docstring)")
    print(f"seed:         {plan['seed']}")
    print(f"runs:         {plan['warmup']} warmup (discarded) + "
          f"{plan['repeats']} scored")
    print(f"\nfiles ({len(plan['files'])}):")
    for f in plan["files"]:
        print(f"  {f['name']:<10} layer {f['layer']:<3} stride {f['stride']:>9} B "
              f"x {f['n_experts']} experts = {f['size']:>12} B "
              f"({mib(f['size'])})")
    classes = sorted({f["stride"] for f in plan["files"]})
    print(f"  stride classes covered: {classes}")

    if mode == "matrix":
        print(f"\nblock sizes by expert count K (at QD={plan['fixed_qd']}):")
        for k in plan["ks"]:
            sizes = ", ".join(
                f"{f['name']}={k * f['stride']} B "
                f"({k * f['stride'] / 2**20:.2f} MiB)"
                for f in plan["files"])
            covered = ", ".join(
                f"{f['name']}:{(f['n_experts'] // k) * k}/{f['n_experts']}"
                for f in plan["files"])
            print(f"  K={k:<2} {sizes}")
            print(f"       experts covered: {covered}")
        print(f"\nqueue depths (at K={plan['fixed_k']}): {plan['qds']}")
    elif mode == "window" and any(not c.get("block_from_k", True)
                                  for c in plan["cases"]):
        print(f"\nsingle cell: QD={plan['fixed_qd']}, block size from "
              f"--window-block-bytes (K is not used; --block-ks and "
              f"--queue-depths are not swept in window mode)")
    else:
        print(f"\nsingle cell: K={plan['fixed_k']} QD={plan['fixed_qd']} "
              f"(--block-ks and --queue-depths are not swept in {mode} mode)")
    if mode == "window":
        print("\nwindows (byte ranges inside one file; the whole file is "
              "still evicted and residency-proved), in built order:")
        for c in plan["cases"]:
            loc = c.get("window_locality") or {}
            print(f"  [{c['order']:>3}] {c.get('window_population', '-'):<9} "
                  f"{c['file']:<10} offset {c['window_offset']:>12} "
                  f"len {c['window_length']:>10} "
                  f"({c['window_length'] / 2**20:.2f} MiB) "
                  f"block {c['block_bytes']:>9} x {c['blocks']:>3} "
                  f"{c['pattern']:<4}")
            if c["window_bytes_unread"]:
                print(f"       {c['window_bytes_unread']} trailing bytes are "
                      f"not read: the window is not an integer number of "
                      f"blocks")
            if loc.get("ok"):
                print(f"       locality: {loc['regions_touched']} region(s), "
                      f"span {dist(loc['physical_span_bytes'])}, "
                      f"dominant {pct_str(loc['dominant_fraction'])}, "
                      f"median seek {dist(loc['median_seek_bytes'])} "
                      f"({loc.get('physical_units', LOGICAL_NOT_LBA)})")
            else:
                print(f"       locality: UNKNOWN ({loc.get('reason')})")
        print(f"\ncase order policy: {plan.get('case_order_policy')}")
        print("  Built order alternates the dense and scattered populations, "
              "so neither owns\n  the head of a run; each repeat then runs a "
              "different permutation of it. The\n  first case of a run pays "
              "warm-up the rest do not, and an 8-read window case\n  is small "
              "enough for that to move the headline (it did, before this was "
              "fixed).")
        for repeat in range(plan["warmup"] + plan["repeats"]):
            label = "warmup" if repeat < plan["warmup"] else "scored"
            print(f"  run {repeat} ({label:<6}) executes cases in order "
                  f"{execution_order(plan['cases'], mode, repeat)}")
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
    # Per-file eviction and the four mincore residency passes each file gets
    # per run are a fixed cost the four-file matrix could ignore and a 48-file
    # sweep cannot. 0.20 s/file/run is an ESTIMATE; a single cold
    # `resident_bytes` on a 373 MiB file measured 0.013 s here.
    evict_s = total_runs * len(plan["files"]) * 0.20
    overhead = (total_runs * (2.0 + 1.5)
                + len(plan["cases"]) * total_runs * 0.15
                + evict_s)
    print(f"\nestimated wall time: {io_s + overhead:.0f} s "
          f"({(io_s + overhead) / 60:.1f} min)")
    print(f"  = {io_s:.0f} s of I/O at an assumed {assume_bw / 1e9:.2f} GB/s "
          f"+ {overhead - evict_s:.0f} s of systemd startup and buffer "
          f"pre-faulting + {evict_s:.0f} s of eviction and residency proof "
          f"over {len(plan['files'])} files x {total_runs} runs")
    print("  This is an estimate from an assumed bandwidth, not a measurement.")
    if io_s + overhead > 120:
        print(f"  START-OF-RUN WARNING: this is a "
              f"{(io_s + overhead) / 60:.1f}-minute run on a machine that "
              f"must stay otherwise idle for the numbers to mean anything. "
              f"Do not start it next to a build.")

    print("\nper-case detail:")
    for c in plan["cases"]:
        window = (f" window {c['window_offset']}+{c['window_length']}"
                  if c.get("window_label") else "")
        print(f"  [{c['order']:>3}] {c['file']:<10} {c['pattern']:<4} "
              f"K={c['k']:<2} QD={c['qd']:<2} block {c['block_bytes']:>9} B "
              f"x {c['blocks']:>3} = {c['bytes']:>12} B{window}")


def print_regions(files: list[dict], mount: dict, cluster_gap: int,
                  scan_lengths: list[int], window_count: int) -> dict:
    """The physical layout of each file, and ready-made `--window` specs.

    Metadata only: `filefrag` reads the extent tree, nothing reads a data
    block, nothing is evicted and no cgroup is entered. That is why this path
    deliberately skips `preflight` — refusing to print an extent map because
    `systemd-run --user` is missing would be a harness bug, not a measurement
    failure.

    The returned report carries `files_ok` and `files_failed` so the caller can
    pick an exit code. "filefrag is not installed" and "this file is one
    extent" produce very similar-looking output — an empty region table — and
    the difference between them is the difference between a finding and a
    missing tool.
    """
    fstype = mount.get("fstype")
    units = physical_units_note(fstype)
    report: dict = {"cluster_gap_bytes": cluster_gap,
                    "physical_units": units,
                    "fstype": fstype,
                    "logical_to_lba_command": chunk_tree_command(
                        mount.get("source"), fstype),
                    "files": [], "files_ok": 0, "files_failed": 0}
    print("=== physical dispersion of the installed expert files ===")
    print(f"device:      {mount.get('source')} ({fstype}) on "
          f"{mount.get('target')}")
    print(f"addresses:   {units}.")
    if fstype == "btrfs":
        print(f"             filefrag reports the btrfs logical address "
              f"space. On a single-device")
        print(f"             `single`-profile filesystem the chunk map is "
              f"monotone within a chunk,")
        print(f"             so clustering survives the translation to LBA, "
              f"but an absolute LBA")
        print(f"             does not. To resolve it, a human with sudo runs:")
        print(f"                 {report['logical_to_lba_command']}")
        print(f"             This probe never runs it: there is no "
              f"passwordless sudo here.")
    else:
        # Telling an ext4 or XFS user that their genuine device block numbers
        # are "NOT device LBA", and then handing them a btrfs command, is the
        # tool being wrong about the user's own filesystem.
        print(f"             filefrag reports device physical block numbers "
              f"directly on {fstype or 'this filesystem'};")
        print(f"             there is no btrfs-style logical space in the "
              f"way, so no chunk-tree")
        print(f"             translation applies. The drive's FTL is still "
              f"between these")
        print(f"             addresses and the NAND, on every filesystem.")
    print(f"region gap:  {cluster_gap} B ({cluster_gap / 2**20:.0f} MiB) — two "
          f"extents are in the same region when the hole between them is no "
          f"bigger than this")

    for f in files:
        parsed = filefrag_extents(f["path"])
        entry: dict = {"file": f["name"], "layer": f["layer"],
                       "path": f["path"], "size": f["size"],
                       "stride": f["stride"]}
        print(f"\n{f['name']}  {f['size']} B ({mib(f['size'])})  "
              f"stride {f['stride']}")
        if not parsed["ok"]:
            # Stays on stdout: it is the report's answer for this file. The
            # summary that follows, and the exit code, are what a wrapper
            # reads, and those go to stderr and to the shell respectively.
            print(f"  filefrag failed: {parsed['reason']}")
            entry["ok"] = False
            entry["reason"] = parsed["reason"]
            report["files"].append(entry)
            report["files_failed"] += 1
            continue
        extents = parsed["extents"]
        stats = dispersion_stats(extents, cluster_gap, units)
        entry.update({"ok": True, "extents": len(extents), **stats})
        report["files_ok"] += 1
        # Every distance goes through `dist()`. A single-extent file — XFS, or
        # anything after `btrfs filesystem defragment` — has no inter-extent
        # seek at all, and `median_seek_bytes` is None there; dividing it by
        # 1e9 raised TypeError on the one path in this script whose whole
        # promise is that it is safe to run.
        print(f"  extents {len(extents)}   physical span "
              f"{dist(stats['physical_span_bytes'])}   "
              f"regions {stats['regions']}   largest region "
              f"{pct_str(stats['largest_region_fraction'])} of bytes")
        print(f"  median inter-extent seek in read order "
              f"{dist(stats['median_seek_bytes'])}"
              + ("  (one extent: there is no seek to take a median of)"
                 if stats["median_seek_bytes"] is None else "")
              + f"   byte-weighted mean position "
                f"{gb(stats['byte_weighted_mean_physical_bytes'])} GB")
        regions = physical_regions(extents, cluster_gap)
        print(f"  {'#':>3} {'bytes':>12} {'%':>7} {'extents':>8} "
              f"{'phys start GB':>14} {'phys end GB':>13}  logical ranges")
        for r in regions:
            ranges = merge_logical_ranges(r["members"])
            shown = ", ".join(f"{a}..{b}" for a, b in ranges[:3])
            more = f" (+{len(ranges) - 3} more)" if len(ranges) > 3 else ""
            print(f"  {r['index']:>3} {r['bytes']:>12} "
                  f"{r['bytes'] / f['size'] * 100:>6.1f}% {r['extents']:>8} "
                  f"{r['physical_start'] / 1e9:>14.3f} "
                  f"{r['physical_end'] / 1e9:>13.3f}  {shown}{more}")

        entry["window_scans"] = []
        for length in scan_lengths:
            scan = scan_windows(extents, f["size"], length, f["stride"],
                                window_count, units)
            entry["window_scans"].append(scan)
            print(f"\n  window candidates of {length} B "
                  f"({length / 2**20:.2f} MiB), ranked by physical span "
                  f"({scan['candidates']} candidate offsets, "
                  f"non-overlapping picks):")
            for kind in ("dense", "scattered"):
                for w in scan[kind]:
                    print(f"    {kind:<9} --window "
                          f"{f['name']}:{w['offset']}:{w['length']}"
                          f"   span {dist(w['physical_span_bytes']):>10}"
                          f"   regions {w['regions_touched']:>2}"
                          f"   dominant {pct_str(w['dominant_fraction']):>6}"
                          f"   median seek "
                          f"{dist(w['median_seek_bytes']):>10}")
            if scan["dense"] and scan["scattered"]:
                d = scan["dense"][0]["physical_span_bytes"]
                s = scan["scattered"][0]["physical_span_bytes"]
                if d > 0:
                    print(f"    span contrast available at this size: "
                          f"{s / d:.1f}x")
                else:
                    print("    the densest window has zero span (one extent), "
                          "so the contrast is unbounded")
                print("    Pass both populations to ONE --window run: it "
                      "interleaves them and permutes\n"
                      "    the order every repeat, which a dense-only run "
                      "followed by a scattered-only\n"
                      "    run cannot do — that shape confounds placement "
                      "with position in the run.")
        report["files"].append(entry)

    print(f"\nWhat this does and does not show: these are host-side address "
          f"statistics ({units}). They can establish that two files, or two "
          f"windows in one file, differ in physical clustering; they cannot "
          f"by themselves establish that the drive's controller sees the same "
          f"difference. The byte-window test (--window) is what turns the "
          f"correlation into a controlled comparison — controlled for the "
          f"file's age and write history, not for the NAND blocks, which two "
          f"windows at different offsets never share.")
    if report["files_failed"]:
        print(f"\n{report['files_failed']} of "
              f"{report['files_failed'] + report['files_ok']} files have no "
              f"extent map. Without filefrag (e2fsprogs) there is no "
              f"dispersion data at all, and an empty region table must not be "
              f"read as 'this file is contiguous'.", file=sys.stderr)
    return report


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
    # Computed out here, not inside the cgroup: the order is a property of the
    # run being planned, and it goes on the record in the plan file before the
    # drive is touched.
    run_plan["case_execution_order"] = execution_order(
        plan["cases"], plan.get("mode", "matrix"), index)
    if plan.get("mode") == "window":
        print(f"  case order: {run_plan['case_execution_order']}")
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


def finish(args, plan: dict, machine: dict, cases: list[dict], frag: dict,
           agg: dict[tuple, dict], runs: list[dict], verdict: str,
           btrfs_before: dict, btrfs_after: dict, btrfs_grew: list[str],
           markdown: str) -> int:
    """Print the hygiene verdict, write the summary, pick the exit code.

    Shared by every mode so that a window run or a 48-file run cannot end up
    with a weaker record — or a more forgiving exit code — than the matrix run
    the docs were written from.
    """
    print(f"\nmeasurement hygiene: {verdict}"
          + ("" if verdict == "PASS" else
             " — reclaim, a cgroup limit, a page cache that grew under "
             "O_DIRECT, a btrfs counter delta, or a counter the verdict "
             "depends on was left unreadable (unknown is DIRTY, not clean); "
             "see the [hard] lines above. Do not publish these numbers"))

    def agg_row(key: tuple, value: dict) -> dict:
        template = value["template"]
        row = {"file": key[0], "pattern": key[1], "k": key[2], "qd": key[3],
               "block_bytes": template["block_bytes"],
               "bytes": template["bytes"],
               "queue": template["queue"],
               "n": value["n"], "gb_s_median": value["gb_s_median"],
               "gb_s_min": value["gb_s_min"], "gb_s_max": value["gb_s_max"],
               "p50_ms_median": value["p50_ms_median"]}
        # Window rows carry the geometry they are meant to be explained by, in
        # the same record as the bandwidth, so no join is needed to regress
        # one on the other.
        if template.get("window_label"):
            row.update({
                "window_offset": template["window_offset"],
                "window_length": template["window_length"],
                "window_bytes_unread": template["window_bytes_unread"],
                "window_locality": template.get("window_locality"),
                # The population split the headline ratio is taken over, and
                # the positions this case actually ran at, so a reader can
                # redo both the comparison and the position check from the
                # aggregate alone.
                "window_rank": template.get("window_rank"),
                "window_population": template.get("window_population"),
                "case_order": template.get("order"),
                "exec_positions": value["exec_positions"],
            })
        else:
            stats = frag.get(template["path"], {})
            row["dispersion"] = {k: stats.get(k) for k in (
                "extents", "physical_span_bytes", "regions",
                "largest_region_bytes", "largest_region_fraction",
                "median_seek_bytes", "mean_seek_bytes",
                "byte_weighted_mean_physical_bytes", "cluster_gap_bytes",
                "physical_units")}
        return row

    summary = {
        "machine": machine,
        "mode": plan.get("mode", "matrix"),
        "plan": {k: v for k, v in plan.items() if k != "cases"},
        "cases": cases,
        "filefrag": frag,
        "btrfs_session_before": btrfs_before,
        "btrfs_session_after": btrfs_after,
        "btrfs_session_grew": btrfs_grew,
        "hygiene": verdict,
        "aggregate": [agg_row(k, v) for k, v in sorted(agg.items())],
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
                             "per-file bandwidth variance exceeds "
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
    parser.add_argument("--all-files", action="store_true",
                        help="probe EVERY layer file in layout.json at the "
                             "single cell (--fixed-k, --fixed-qd) instead of "
                             "sweeping four files. --block-ks and "
                             "--queue-depths are ignored. This is the mode "
                             "that turns 'decode pays the per-file spread on "
                             "all 48 files' from an assumption into a "
                             "measurement, and it emits per-file bandwidth "
                             "next to per-file physical dispersion so the two "
                             "can be regressed")
    parser.add_argument("--window",
                        help="measure bounded byte windows INSIDE files: "
                             "FILE:OFFSET:LENGTH[,FILE:OFFSET:LENGTH...], "
                             "sizes accept K/M/G. Read at (--fixed-k, "
                             "--fixed-qd) unless --window-block-bytes "
                             "overrides the block size. Two windows in one "
                             "file share the file's age, its write history "
                             "and the install run that wrote it — but not "
                             "their NAND blocks, which differ by "
                             "construction, so SLC and read-disturb state are "
                             "uncontrolled rather than equal. Windows must be "
                             "disjoint and --patterns must name exactly one "
                             "pattern, or bytes get read twice in a run and "
                             "the second read is a drive-cache hit. Dense and "
                             "scattered windows are interleaved and the order "
                             "is permuted every repeat. Use --list-regions to "
                             "get specs")
    parser.add_argument("--window-block-bytes", type=parse_size,
                        help="block size for --window cases, overriding "
                             "K x stride. Must be a multiple of 4096. Needed "
                             "for windows smaller than one expert blob, which "
                             "is where the densest windows are")
    parser.add_argument("--list-regions", nargs="?", const="", default=None,
                        metavar="FILES",
                        help="print each file's physical regions and "
                             "ready-made --window specs, then exit. Optional "
                             "comma-separated stems, 'all' for every layer; "
                             "defaults to --files. Metadata only: reads no "
                             "data block, evicts nothing, enters no cgroup")
    parser.add_argument("--cluster-gap", type=parse_size,
                        default=DEFAULT_CLUSTER_GAP,
                        help="two extents are in one physical region when the "
                             "hole between them is no larger than this "
                             "(default: %(default)s = 256 MiB, a quarter of a "
                             "btrfs data chunk)")
    parser.add_argument("--window-scan-bytes", default=DEFAULT_WINDOW_SCAN,
                        help="window sizes --list-regions scans for dense and "
                             "scattered candidates (default: %(default)s)")
    parser.add_argument("--window-count", type=int, default=4,
                        help="how many non-overlapping dense and scattered "
                             "candidates --list-regions emits per size "
                             "(default: %(default)s)")
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

    if args.cluster_gap < 0:
        fail("--cluster-gap must be >= 0")
    # Rejected here rather than left to fall through: 0 is falsy, and the
    # builder used to treat "the user asked for 0-byte blocks" and "the user
    # asked for nothing" as the same request — silently reading at K x stride
    # while `block_from_k` and every table said the flag had supplied the size.
    if args.window_block_bytes is not None and args.window_block_bytes < 1:
        fail(f"--window-block-bytes {args.window_block_bytes} is not a "
             f"positive size; a block has to have bytes in it")
    if args.window_count < 1:
        fail("--window-count must be >= 1")
    scan_lengths = []
    for text in (t.strip() for t in args.window_scan_bytes.split(",")):
        if not text:
            continue
        try:
            length = parse_size(text)
        except ValueError:
            fail(f"--window-scan-bytes {text!r} is not a size")
        if length <= 0 or length % DIO_ALIGN:
            fail(f"--window-scan-bytes {text!r} is not a positive multiple of "
                 f"{DIO_ALIGN}")
        scan_lengths.append(length)

    args.rvmp = os.path.abspath(args.rvmp)
    mount = mount_info(os.path.join(args.rvmp, "experts"))

    def every_stem(layers: list[dict]) -> list[str]:
        return [os.path.splitext(os.path.basename(x["file"]))[0]
                for x in layers]

    # --- --list-regions: metadata only, so it must not need the measurement
    # --- harness to be available at all. No preflight, no drive access.
    if args.list_regions is not None:
        if not os.path.isdir(os.path.join(args.rvmp, "experts")):
            fail(f"{args.rvmp} has no experts/ directory")
        layers = load_layout(args.rvmp)
        wanted = (args.list_regions or args.files).strip()
        names = (every_stem(layers) if wanted.lower() == "all"
                 else [n for n in wanted.split(",") if n.strip()])
        listed = resolve_files(names, layers, args.rvmp)
        report = print_regions(listed, mount, args.cluster_gap, scan_lengths,
                               args.window_count)
        if args.json:
            os.makedirs(
                os.path.dirname(os.path.abspath(args.json)) or ".",
                exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nwrote {args.json}")
        # An empty report is not a clean one. `filefrag` missing (rc 127, no
        # e2fsprogs), an unreadable extent tree, or a report with no extents in
        # it all land here, and a wrapper script that only checks the exit
        # status would otherwise file "no dispersion" as the finding.
        if report["files_ok"] == 0:
            print("no extent map could be read for any requested file, so "
                  "nothing was measured. Install e2fsprogs (filefrag), or "
                  "check that the files are readable.", file=sys.stderr)
            return 2
        return 1 if report["files_failed"] else 0

    preflight(args)
    layers = load_layout(args.rvmp)

    # --- mode selection -------------------------------------------------
    if args.all_files and args.window:
        fail("--all-files and --window ask for two different experiments; "
             "run them separately so each one's numbers stand alone")
    if args.all_files:
        if args.files != ",".join(DEFAULT_FILES):
            fail("--all-files and an explicit --files contradict each other; "
                 "drop one")
        mode = "all-files"
        names = every_stem(layers)
    elif args.window:
        mode = "window"
        specs = parse_window_specs(args.window)
        names = list(dict.fromkeys(name for name, _, _ in specs))
    else:
        mode = "matrix"
        names = args.files.split(",")

    files = resolve_files(names, layers, args.rvmp)
    classes = sorted({f["stride"] for f in files})
    if len(classes) < 2 and mode != "window":
        print(f"warning: the probed file set covers only stride class(es) "
              f"{classes}. The install has two, and they read differently — "
              f"the result will not generalise.", file=sys.stderr)

    # The extent map is needed before the cases exist, because a window case
    # records the locality of the bytes it is about to read. `filefrag` reads
    # the extent tree only: no data block is touched, so this is safe under
    # --dry-run too.
    units = physical_units_note(mount.get("fstype"))
    frag: dict = {}
    extent_map: dict = {}
    for f in files:
        parsed = filefrag_extents(f["path"])
        frag[f["path"]] = filefrag_stats(f["path"], args.cluster_gap,
                                         parsed=parsed, units=units)
        if parsed["ok"]:
            physical_regions(parsed["extents"], args.cluster_gap)
            extent_map[f["path"]] = parsed["extents"]

    if mode == "all-files":
        cases, combos = build_all_files_cases(files, args.fixed_k,
                                              args.fixed_qd, patterns)
    elif mode == "window":
        cases, combos = build_window_cases(files, specs, args.fixed_k,
                                           args.fixed_qd, patterns,
                                           args.window_block_bytes,
                                           extent_map, units)
    else:
        cases, combos = build_cases(files, ks, qds, args.fixed_qd,
                                    args.fixed_k, patterns)
    if not cases:
        fail("the requested matrix contains no cases")

    plan = {
        "rvmp": args.rvmp,
        "mode": mode,
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
        "cluster_gap_bytes": args.cluster_gap,
        "physical_units": units,
        "logical_to_lba_command": chunk_tree_command(mount.get("source"),
                                                     mount.get("fstype")),
        # How the case list was built, and therefore how to read a per-run
        # `case_execution_order`. Recorded rather than described: the whole
        # point of the interleave is that a reader can check it.
        "case_order_policy": (
            "window: ranked by physical span, dense and scattered halves "
            "interleaved, then rotated left by the run index so no case "
            "repeats a position across runs"
            if mode == "window" else
            "built order, identical every run (whole-file cases issue enough "
            "reads for position not to matter)"),
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
    print(f"  mode       {mode}")
    print(f"  matrix     {len(cases)} cases x "
          f"{args.warmup} warmup + {args.repeats} scored")
    print(f"  command    {machine['command_line']}")

    print(f"\nextent geometry and physical dispersion (filefrag -v; "
          f"{units}):")
    if mount.get("fstype") == "btrfs":
        resolve = chunk_tree_command(mount.get("source"), mount.get("fstype"))
        print(f"  logical->LBA, for a human with sudo: {resolve}")
    for f in files:
        stats = frag[f["path"]]
        if stats["ok"]:
            # Distances through dist()/pct_str(): a one-extent file has no
            # median seek at all and None does not divide.
            print(f"  {f['name']:<10} {stats['extents']:>5} extents  "
                  f"mean {stats['mean_extent_bytes']:>9} B  "
                  f"median {stats['median_extent_bytes']:>9} B  "
                  f"adjacent {stats['adjacent_pairs']}/"
                  f"{max(stats['extents'] - 1, 0)}  "
                  f"compressed {stats['encoded_extents']}  "
                  f"span {dist(stats['physical_span_bytes']):>10}  "
                  f"regions {stats['regions']:>3}  "
                  f"largest {pct_str(stats['largest_region_fraction']):>6}  "
                  f"med seek {dist(stats['median_seek_bytes']):>10}")
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

    if mode == "all-files":
        print(f"\ndecode-shaped cell K={args.fixed_k} QD={args.fixed_qd} "
              f"across all {len(files)} layer files, joined to physical "
              f"dispersion (GB/s, median of {len(scored)}; dispersion is "
              f"{units}):")
        print("  " + "  ".join(DISPERSION_BW_HEADER))
        for row in dispersion_bandwidth_rows(plan, agg, frag):
            print("  " + "  ".join(str(c) for c in row))
        for pattern in patterns:
            rates = [(agg[(f["name"], pattern, args.fixed_k,
                           args.fixed_qd)]["gb_s_median"], f["name"])
                     for f in files
                     if (f["name"], pattern, args.fixed_k,
                         args.fixed_qd) in agg]
            if len(rates) > 1:
                lo, hi = min(rates), max(rates)
                print(f"  {pattern}: spread across {len(rates)} files "
                      f"{lo[0]:.3f} ({lo[1]}) .. {hi[0]:.3f} ({hi[1]}) = "
                      f"{hi[0] / lo[0] if lo[0] else 0:.2f}x — this is the "
                      f"population, not a four-file sample")

    if mode == "window":
        print(f"\nbyte-window locality test (GB/s, median of {len(scored)}; "
              f"dispersion is {units}):")
        print("  " + "  ".join(WINDOW_HEADER))
        for row in window_rows(plan, agg):
            print("  " + "  ".join(str(c) for c in row))
        print(f"  case order per run: "
              f"{[r.get('case_execution_order') for r in scored]}")
        for f in files:
            for pattern in patterns:
                pairs = []
                for case in cases:
                    if case["file"] != f["name"] or case["pattern"] != pattern:
                        continue
                    entry = agg.get(case_key(case))
                    loc = case.get("window_locality") or {}
                    if entry and loc.get("ok"):
                        pairs.append((loc["physical_span_bytes"],
                                      entry["gb_s_median"],
                                      entry["p50_ms_median"], case))
                if len(pairs) < 2:
                    continue
                # An explicit key. Span ties among dense windows are certain —
                # several can sit inside one extent — and a bare sort() falls
                # through to comparing the bandwidth, then the `case` dict,
                # which raises. Offset then canonical index break every tie.
                pairs.sort(key=lambda p: (p[0], p[3]["window_offset"],
                                          p[3]["order"]))
                # The median over every dense window and every scattered one,
                # not the single best and single worst: with a handful of
                # reads per window the extremes are the noise, and the
                # question is whether the two populations differ.
                half = len(pairs) // 2
                dense = statistics.median(r for _, r, _, _ in pairs[:half])
                scatt = statistics.median(r for _, r, _, _ in pairs[-half:])
                # The same comparison on per-read latency. Bandwidth over a
                # handful of reads is a ratio of two short intervals; the p50
                # is a property of the reads themselves. If the two disagree
                # in direction, the bandwidth ratio is measuring the harness.
                dense_p50 = statistics.median(
                    p for _, _, p, _ in pairs[:half])
                scatt_p50 = statistics.median(
                    p for _, _, p, _ in pairs[-half:])
                bw_ratio = dense / scatt if scatt else 0
                p50_ratio = scatt_p50 / dense_p50 if dense_p50 else 0
                agrees = (bw_ratio - 1) * (p50_ratio - 1) >= 0
                print(f"  {f['name']} {pattern}: {half} densest windows "
                      f"(span up to {dist(pairs[half - 1][0])}) median "
                      f"{dense:.3f} GB/s vs {half} most scattered (span from "
                      f"{dist(pairs[-half][0])}) median {scatt:.3f} GB/s = "
                      f"{bw_ratio:.2f}x.")
                print(f"    per-read p50 over the same two populations: "
                      f"{dense_p50:.3f} ms dense vs {scatt_p50:.3f} ms "
                      f"scattered = {p50_ratio:.2f}x "
                      f"({'agrees with' if agrees else 'DISAGREES with'} the "
                      f"bandwidth ratio). These windows issue "
                      f"{pairs[0][3]['blocks']} reads each, so the two must "
                      f"agree before either is believed.")
                print(f"    A ratio near 1.00 says physical placement is not "
                      f"moving the number. What that rules out is placement "
                      f"and anything else that differs between two ranges of "
                      f"one file; it does not rule out per-NAND-block state, "
                      f"which two windows never share.")

        # Position, printed rather than assumed clean. The interleave and the
        # per-repeat permutation exist to stop run position from deciding the
        # answer; this is the check that they did.
        by_position: dict[int, list[float]] = {}
        for run in scored:
            for case in run.get("cases", []):
                if case.get("exec_position") is not None:
                    by_position.setdefault(case["exec_position"],
                                           []).append(case["gb_s"])
        if by_position:
            overall = statistics.median(
                [v for values in by_position.values() for v in values])
            worst = min(by_position.items(),
                        key=lambda kv: statistics.median(kv[1]))
            first = statistics.median(by_position.get(0, [overall]))
            ratio = first / overall if overall else 0
            print(f"\n  position check (median GB/s by position in the run, "
                  f"across {len(scored)} scored runs): overall "
                  f"{overall:.3f}, position 0 {first:.3f} ({ratio:.2f}x), "
                  f"slowest position {worst[0]} at "
                  f"{statistics.median(worst[1]):.3f}. A position 0 well "
                  f"below the rest is warm-up, not placement — the interleave "
                  f"and the per-repeat rotation spread it over both "
                  f"populations, they do not remove it.")

    if mode != "matrix":
        markdown = build_markdown(plan, agg, frag, machine, verdict)
        print("\n=== markdown (summarise in docs/experiments.md) ===\n")
        print(markdown)
        return finish(args, plan, machine, cases, frag, agg, runs, verdict,
                      session_btrfs_before, session_btrfs_after, session_grew,
                      markdown)

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
    print("\n=== markdown (summarise in docs/experiments.md) ===\n")
    print(markdown)

    return finish(args, plan, machine, cases, frag, agg, runs, verdict,
                  session_btrfs_before, session_btrfs_after, session_grew,
                  markdown)


if __name__ == "__main__":
    sys.exit(main())
