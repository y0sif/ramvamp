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
| RAM | 15.3 GiB |
| Swap | zram, 7.5 GiB. Counts as swap, so benchmark runs set `memory.swap.max=0` |
| Storage | Micron 2400 `MTFDKBA1T0QFM-1BD1AABGB`, DRAM-less QLC, PCI `1344:5413`, Gen4 x4 |
| Filesystem | btrfs on `/dev/nvme0n1p2`, `compress=zstd:3,ssd,discard=async`, data profile `single` |
| Kernel | 7.0.12-arch1-1 |
| `RLIMIT_MEMLOCK` | 8 MiB soft **and** hard (systemd default; the hard limit needs `CAP_SYS_RESOURCE`) |
| earlyoom | active |

The drive is not the one the design doc originally assumed. See EXP-008 for
what it actually sustains; the short version is that block size dominates and
queue depth saturates by QD4-8.

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
- Concurrent readers cost roughly 3.5x read amplification, and per-file
  fragmentation variance exceeds run-to-run variance (`layer_00` 1,237 MB/s vs
  `layer_20` 1,812 MB/s), so benchmarks must pin the same file set and run on a
  quiet machine.
