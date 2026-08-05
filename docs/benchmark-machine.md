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
- Concurrent readers cost roughly 3.5x read amplification, and **per-file
  bandwidth variance exceeds run-to-run variance** (`layer_00` 1,237 MB/s vs
  `layer_20` 1,812 MB/s, a 1.46x spread), so benchmarks must pin the same file
  set and run on a quiet machine. That advice stands and the spread is
  reproducible: EXP-019 measures the same two files at 1.58 and 2.27 GB/s, a
  1.44x spread, far above its 1.4% run-to-run median.

  **It is not fragmentation.** This file used to attribute the spread to
  fragmentation; EXP-019 eliminates that hypothesis. The two files have
  byte-identical extent geometry, 398 extents each, mean 984,027 B, median
  884,736 B, and zero extents physically adjacent to their successor. What is
  left is physical placement on the drive or QLC-internal behaviour such as
  SLC-cache residency or block wear, none of which a filesystem-level probe
  can see. Consequence for benchmarking: an aggregate bandwidth figure that
  hides this spread is worse than no aggregate, and any run that changes which
  files it touches has changed its own baseline.
