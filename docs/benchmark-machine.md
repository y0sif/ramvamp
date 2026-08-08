# Reference benchmark machine

State of the machine every published number is measured on, recorded so that
results stay interpretable and so that anomalies can be attributed rather than
guessed at. Update this file when any of it changes.

## Hardware and OS

| | |
| --- | --- |
| CPU | Intel Core Ultra 9 185H: 6 P-cores (SMT), 8 E-cores, 2 LP E-cores, 22 threads |
| P-core primaries | 0, 1, 3, 6, 8, 10 (SMT pairs are 0-5, 1-2, 3-4, 6-7, 8-9, 10-11) |
| E-cores with L3 | 12-19. LP E-cores 20-21 are on the SoC tile with no L3 |
| CPU ISA | `avx`, `avx2`, `f16c`, `fma`, `avx_vnni`, `bmi1`, `bmi2`, `sha_ni`. **No AVX-512 of any flavour** — Meteor Lake ships none, and `/proc/cpuinfo` carries zero `avx512*` flags |
| RAM | 15.3 GiB (`MemTotal` reads 15,711,132 kB = **14.98 GiB**; the 15.3 figure is the older reading and the two have never been reconciled — use `MemTotal` for arithmetic) |
| Swap | zram, 7.5 GiB. Counts as swap, so benchmark runs set `memory.swap.max=0` |
| Storage | Micron 2400 `MTFDKBA1T0QFM-1BD1AABGB`, DRAM-less QLC, PCI `1344:5413`, Gen4 x4 |
| Filesystem | btrfs on `/dev/nvme0n1p2`, `compress=zstd:3,ssd,discard=async`, data profile `single` |
| Kernel | 7.1.5-arch1-1 (verified `uname -r`, 2026-08-05). This file recorded 7.0.12-arch1-1 until then, so every entry through EXP-019 was most likely taken on that kernel; the upgrade point was not recorded, which is exactly why cross-entry curves are already forbidden by rule 3 |
| `RLIMIT_MEMLOCK` | 8 MiB soft **and** hard (systemd default; the hard limit needs `CAP_SYS_RESOURCE`) |
| earlyoom | active |

**The ISA row is load-bearing from phase 7 onward.** `crates/core/src/kernels/
attention/x86.rs` is gated on `avx2` **and** `f16c` — separate CPUID bits, so it
is not the same probe `quants::avx2` uses — and it deliberately does **not**
enable `fma`, because a fused multiply-add is one rounding where the scalar
reference has two and the bit-identity gate would fail. `avx_vnni` is present
and unused. The absence of AVX-512 is why the vector width in every kernel here
is 8 f32 and why nothing is written against a 16-lane assumption; a machine with
AVX-512 would run the same code, not faster code.

The drive is not the one the design doc originally assumed. **EXP-019** is what
it actually sustains, measured cold and in-cgroup on the installed layer files:
**1.54 to 2.37 GB/s**, and the variable that predicts throughput is **total
bytes in flight**, not block size. Block size and queue depth move that one
quantity and are interchangeable at matched bytes, so neither dominates on its
own; the drive holds peak up to roughly **100 MB outstanding** and loses 15 to
18 percent past about 170 MB.

EXP-008's older reading, "block size dominates and queue depth saturates by
QD4-8", is superseded on both counts. Its block-size curve is refuted (bigger
blocks are neutral on one probed file and 15 to 16 percent worse on three
others), and its levels of 1.211 to 1.390 GB/s were measured on a contended
machine, exactly as its own Method warned. Read EXP-019 before deriving
anything from a bandwidth number on this box.

## btrfs error counters

`btrfs device stats /home` keeps five persistent per-device counters. They
survive reboots and mounts, and only `btrfs device stats --reset` clears them.

**Baseline recorded 2026-08-03: `corruption_errs` 138,407.** Deliberately not
reset, so the history is preserved. **Any increase above that number is the
signal worth investigating.**

`read_io_errs`, `write_io_errs`, `flush_io_errs` and `generation_errs` were all
**0** at the time of recording, and remain the counters to watch for genuine
device trouble: `corruption_errs` alone means btrfs caught a checksum mismatch
and returned an error rather than handing bad data to the reader.

Cause of the recorded 138,407, established and reproduced (EXP-007): a
benchmark harness indexed its buffer pool with modulo arithmetic, so two
in-flight O_DIRECT reads could target the same buffer. btrfs verifies the
checksum after DMA into the user buffer, so one read's verification ran over
bytes the other had already overwritten. Measured 13-27% spurious `EIO` with
aliased buffers versus 0 across 4,000 reads with an explicit free list. The
data on disk was never affected: the model files hash-match `manifest.json`,
the failing regions re-read clean, and an instrumented run confirmed 6,000
blocks delivered byte-correct while 256 checksum failures were logged.

This is why `crates/core/src/io/slots.rs` hands out owning `SlotGuard` leases
from a free list instead of index arithmetic, and why `chattr +C` is not used
on the model directory despite measuring +30% read and +112% write: nodatacow
disables the checksums that caught this.

To check:

```fish
btrfs device stats /home
```

## Known constraints on measurement

- **No passwordless sudo**, so `drop_caches` cannot be automated. Cold runs use
  `posix_fadvise(POSIX_FADV_DONTNEED)` plus a `mincore` residency assertion
  instead (`scripts/cold_bench.py`). Note `fadvise` silently evicts nothing
  when a process holds the file mmap'd, so the harness refuses to start while
  any `ramvamp` process is alive.
- The `io` controller is **not** delegated to user cgroups, so per-scope I/O
  accounting is unavailable. Measure `/proc/self/io` `read_bytes` in-process
  instead.
- `cgroup v2` memory accounting works rootless via
  `systemd-run --user --scope`; `memory.swap.peak` needs kernel 6.5 or newer.
- Concurrent readers cost roughly 3.5x read amplification, and **benchmarks
  must pin the same file set and run on a quiet machine**. That advice stands
  and is the durable part of this bullet. What it used to rest on does not.

  **The per-file spread is real in any one session and is not stable across
  sessions, which is the bigger caveat.** Three sessions have measured the same
  four expert files at decode's own read geometry (K=1, one expert blob,
  random order, QD 8), all cold, all in-cgroup, all hygiene PASS, and they
  disagree by more than 2x on a quantity each reported as a property of a file:

  | session | `layer_00` | `layer_06` | `layer_20` | `layer_21` | spread |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | EXP-019, 2026-08-04 | 1.594 | 1.627 | 1.688 | 1.550 | 1.09x |
  | EXP-023, 2026-08-06 | 1.568 | 1.654 | 3.455 | 3.469 | **2.21x** |
  | EXP-024, 2026-08-07 | 1.672 | 1.607 | 1.658 | 1.601 | **1.04x** |

  All figures MEASURED, cold, GB/s = 10^9 B/s, medians of three scored runs.
  **These are three sessions and rule 3 forbids drawing one curve through
  them**; the table exists to show that they disagree, not to trend them.
  Consequences for benchmarking, in order of importance: **pin the same file
  set**, because a run that changes which files it touches has changed its own
  baseline; **an aggregate that hides a spread is worse than no aggregate**;
  and **never quote a per-file spread as a fact about this drive** — it is a
  fact about the session that measured it, and it must be re-measured inside
  any entry that leans on it.

  **Extent geometry is ruled out. Physical placement is now measured, and it is
  a covariate rather than a mechanism.** This file used to attribute the spread
  to fragmentation, then to say flatly "it is not fragmentation" on the
  strength of geometry alone. The geometry half stands: `layer_00` and
  `layer_20` have byte-identical extent geometry, 398 extents each, mean
  984,027 B, median 884,736 B, zero extents physically adjacent to their
  successor, zero compressed extents (EXP-019, EXP-023). But the probe that
  supported that statement only ever computed extent count, extent sizes and
  successor adjacency — it never computed physical span or clustering — so it
  could see *fragmentation* and could not see *placement*. EXP-024 added the
  missing statistics and measured them over the whole population:

  - All 48 expert files at the same cell read **1.565 to 1.694 GB/s, median
    1.633, a 1.082x spread** (MEASURED cold). Most of even that is blob size:
    the 24 layers at a 3,059,712 B expert stride median 1.671 GB/s and the 24
    at 2,654,208 B median 1.610, so the residual spread inside a stride class
    is 1.045x and 1.047x.
  - **Pearson r between bandwidth and largest-region byte fraction is
    -0.043** over those 48 files. Bandwidth correlates with block size (r =
    0.835) and with dispersion not at all.
  - `layer_00` is the **only** physically dispersed file of the 48 — 26
    regions, 23.0% of its bytes in its largest, 72.46 GB median inter-extent
    seek — against **43 of 48 at >= 99% of bytes in one region**. It is
    **not** the slowest file: rank 17 of 48. The 99.6%-clustered `layer_06`,
    with a 0.11 GB median seek, ranks 16 — indistinguishable from `layer_00`
    in that run (1.6165 against 1.6169) and clearly slower in the four-file
    arm (1.607 against 1.672).
  - Sixteen 2 MiB windows **inside `layer_00`**, dense against scattered, are
    **1.161x** apart (1.093 vs 0.941 GB/s medians, ranges 1.024-1.284 and
    0.823-1.067) against a **7,493x** median physical-span contrast.

  Dispersion is reported as **btrfs LOGICAL bytenr from `filefrag`, not device
  LBA**; resolving it needs `sudo btrfs inspect-internal dump-tree -t 3
  /dev/nvme0n1p2`, which the probe never runs. What is left as the mechanism is
  drive-internal and invisible to any filesystem-level probe — pSLC residency
  or FTL state on this DRAM-less QLC part — and it is not addressable from the
  runtime. Vendor SMART would be the next evidence and is unavailable here:
  `nvme-cli` and `smartmontools` are not installed on this machine. See EXP-024
  for the commands, and for why a "fix" that only holds while data sits in
  pSLC must never be published as a runtime improvement.
