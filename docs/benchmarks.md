# Benchmarks

Covers every published throughput, memory and fidelity number for v0, the
conditions each one was measured under, and why they are stated per drive.

## Measured on the reference machine

Intel Core Ultra 9 185H (CPU only, 16 GB RAM, **Micron 2400 DRAM-less QLC**
NVMe) running Qwen3-30B-A3B Q4_K_M with a cold page cache inside
`memory.max=3G` and `memory.swap.max=0`:

| | measured |
| --- | --- |
| Decode | **about 2 tok/s**: 1.46 to 2.19 over ctx 64-3,961 (EXP-023), and 1.43 to 2.16 over the same rungs on a later branch (EXP-025) |
| Prefill | **11.25 tok/s** at ctx 512 (EXP-023) |
| Peak RAM | **2,497 to 2,929 MiB** of a 3,072 MiB ceiling: 2,497.0 at ctx 64 (EXP-025) and 2,929.3 at ctx 3,961 (EXP-023) |
| Model on disk | 17.35 GiB, a **~6x** memory saving |
| Fidelity | mean full-vocab KL **1.04e-2** vs llama.cpp, top-1 agreement 8/8 (EXP-004) |

## The conditions that make those figures valid

Every number above comes from a run that satisfies all of these. A run that
skips one is a diagnostic, not a published figure.

- **Cold.** The page cache is evicted before the run, and the harness proves
  the eviction rather than assuming it.
- **In-cgroup at `memory.max=3G` with `memory.swap.max=0`.** Swap is off
  because zram counts as swap, and a run that swaps is measuring reclaim
  rather than decode.
- **A release build** (`cargo build --release`). The debug profile is
  `opt-level = 1` and is not comparable.
- **Run as the user that owns the install.** The kernel will not report
  page-cache residency for a file this process neither owns nor may write, so
  on an install unpacked by root and run by you the startup probe cannot prove
  O_DIRECT is real and degrades to buffered reads, which changes both the
  memory figure and the throughput figure.

The full protocol, including the hygiene counters that have to be reported
next to any number, is in [../CONTRIBUTING.md](../CONTRIBUTING.md).

## Two ladders, not one curve

Those two decode ladders come from different binaries in different sessions,
and are deliberately not drawn as one curve. The drift is measured rather than
assumed: the later session also re-ran EXP-023's *byte-identical* binary at two
rungs and read **1.85 tok/s against 1.91** at ctx 512, and **1.33 against
1.46** at 3,961. That is 3.1% and 8.9% slower a day later, on the same machine,
with no code change. On a DRAM-less QLC part a decode figure describes its
session as well as its device, which is why no single number appears here
without one.

## Throughput is stated per drive on purpose

Decode is I/O-bound: expert reads are 44.4% of a decode token at ctx 512 and
33.2% at 3,961, and because that bucket is measured as a residual it is a lower
bound rather than the drive's busy time.

This reference part is DRAM-less QLC and sustains about **1.6 GB/s** at
decode's own read geometry: 1.60 to 1.67 GB/s re-measured over the four probe
files, and 1.565 to 1.694 GB/s with a median of 1.633 across all 48 expert
files.

A mainstream TLC Gen4 drive should do materially better; by how much is
unmeasured, and this project does not publish numbers it has not measured. One
report from a second drive turns a published band into a curve.

## Where the record lives

- [experiments.md](experiments.md) is the experiment record: every EXP id cited
  above, what it tested, its key evidence and its verdict.
- [benchmark-machine.md](benchmark-machine.md) is the state of the machine
  every published number is measured on.
- [roadmap.md](roadmap.md) is the plan of record and carries the same headline
  figures alongside what is still open.
