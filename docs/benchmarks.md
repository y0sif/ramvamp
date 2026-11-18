# Benchmarks

This page holds every published throughput, memory and fidelity number for
ramvamp, the conditions each was measured under, and the machine it ran on.
Each figure cites the row in [experiments.md](experiments.md) that produced it.

## The current runtime, at the 4 GB contract

On the reference machine, Qwen3-30B-A3B Q4_K_M decodes at 2.78 to 2.81 tok/s
cold inside a 4 GiB memory cap with swap off. Conditions: cold page cache,
`memory.max=4G`, `memory.swap.max=0`, the default 19 expert slots per layer,
the machine in ordinary use, every run hygiene-clean (EXP-033).

| | measured |
| --- | --- |
| Decode, ctx 512, `--max-new 128` | **2.78 to 2.81 tok/s** (median 2.795 of four scored runs) |
| Decode, ctx 3,961, `--max-new 64` | **1.96 tok/s** (one scored run) |
| Prefill, ctx 512 | **24.48 to 25.27 tok/s** |
| Prefill, ctx 3,961 | **18.23 tok/s** (one scored run) |
| Expert cache hit rate | 67.1% at ctx 512, 68.3% at ctx 3,961 |
| `memory.peak` | **3,690 to 3,697 MiB** at ctx 512, **4,008.4 MiB** at ctx 3,961, of 4,096 |
| Model on disk | 17.35 GiB (EXP-018), about **4.4x** the 4,008 MiB peak |

Longer replies decode a little slower because attention grows with context,
not because the cache is colder (EXP-034). At ctx 512 decode is 2.80, 2.79 and
2.61 tok/s over 128, 256 and 512 new tokens, with the hit rate flat at 67.1%
to 67.7%. At ctx 3,961 with 128 new tokens decode is 2.04 to 2.06 tok/s and
`memory.peak` reaches **4,016 to 4,020 MiB**, the highest clean peak recorded.

`--cache-bytes 1440M` selects 11 slots per layer, the earlier 3 GB
configuration. Paired with 19 slots in one session (EXP-033), 11 slots decodes
2.215 tok/s at ctx 512 (19 slots is 1.26x faster) and 1.75 at ctx 3,961 (1.12x),
and peaks at 2,960.8 MiB at ctx 3,961, measured inside the 4G cgroup. Inside a
3G cap it peaks at 2,962.9 MiB at ctx 3,961 (EXP-031); on the build before the
Q6_K tile, at ctx 512 and `--max-new 64`, it measured prefill 23.93 to 24.21
tok/s, decode 2.18 to 2.24 and a 2,638 to 2,657 MiB peak (EXP-032).

## Fidelity

ramvamp's output matches llama.cpp b10217 run on the same GGUF bytes.
Correctness figures are exempt from the cold and cgroup conditions.

| | measured |
| --- | --- |
| Full-vocab KL against llama.cpp, 8 prompts | mean **1.04e-2**, worst prompt 2.72e-2, top-1 agreement 8/8 (EXP-004) |
| WikiText-2 perplexity, 40 windows of 512 | **6.3598** +/- 0.16504 against llama.cpp's 6.3810 +/- 0.16588, -0.33%, token streams identical (EXP-035) |

## v0, as first published

v0 shipped at a 3 GB contract: `memory.max=3G`, swap off, 11 slots per layer,
cold. Decode measured 1.46 to 2.19 tok/s over ctx 64 to 3,961 (EXP-023) and
1.43 to 2.16 over the same rungs on a later build (EXP-025); prefill 11.25 tok/s
at ctx 512 (EXP-023); `memory.peak` 2,497.0 to 2,929.3 MiB of 3,072 (EXP-025,
EXP-023). The two decode ladders are two sessions, not one curve
([Method](#method)).

## Method

A figure on this page is published only if its run meets every condition
below; any other run is a diagnostic. The numbered rules behind them are in
[experiments.md](experiments.md#measurement-rules).

- **Cold.** Every model file is evicted with `posix_fadvise` before the run,
  the harness proves the eviction with `mincore`, and non-zero block-layer
  `read_bytes` is the positive control.
- **In-cgroup.** The run is capped at the contract's `memory.max` (4G now, 3G
  before EXP-033) with `memory.swap.max=0`, because zram counts as swap and a
  run that swaps measures reclaim, not decode.
- **Page cache counted.** `memory.peak` is the cgroup's highest memory usage,
  and cgroup v2 charges page cache as well as anonymous memory to it, so file
  pages the run reads are inside the figure, not hidden in free RAM.
- **Hygiene PASS.** The inner wrapper resolved its own cgroup, `memory.max`
  equals the request, no more than 1 MiB of pressure-reclaimed pages was read
  back (huge-page compaction excepted), every `memory.events` counter and the
  swap peak are zero, and both processes exited 0. An unreadable counter fails.
- **A release build** (`cargo build --release`), run as the user that owns the
  install. Otherwise the startup probe cannot prove O_DIRECT and falls back to
  buffered reads, which changes both memory and throughput.

Decode is I/O-bound ([architecture.md](architecture.md#performance-model)), so
throughput is stated per drive; how much a faster drive gains is unmeasured. A
decode figure also describes its session: the byte-identical EXP-023 binary
re-read 1.85 against 1.91 tok/s at ctx 512 and 1.33 against 1.46 at 3,961 a
day later (EXP-025), so every ratio here is paired within one session.

### Reproduce a run

`scripts/cold_bench.py` sets up every condition above and verifies it. After
`cargo build --release` and `ramvamp-repack install --output models/qwen3.rvmp`
(the default pin, so the weights match), run decode, then prefill at a fixed
context from a prompt file with one new token:

```bash
scripts/cold_bench.py --rvmp models/qwen3.rvmp --max-new 64 --repeats 5 --json scratch/cold-decode.json
scripts/cold_bench.py --rvmp models/qwen3.rvmp --prompt-file <prompt.txt> --max-new 1 --repeats 5 --json scratch/cold-prefill.json
```

It needs Linux 6.5+, cgroup v2 and `systemd-run --user`, and it refuses to
start while any `ramvamp` process is alive, because `posix_fadvise` cannot
evict a file another process has mapped. A build running alongside invalidates
the run. Exit 0 means every run was clean, 1 that one was dirty (report it
anyway), 2 that nothing could be measured. A reply that loops or truncates is
not a valid speed result, so check the text with a greedy run:

```bash
target/release/ramvamp generate --model models/qwen3.rvmp --prompt "The capital of France is" --max-new 64 --greedy
```

Send the summary JSON, that output and the drive, filesystem, kernel, CPU and
RAM through the benchmark report issue form.

## Reference machine

Every throughput and memory figure on this page was measured on this machine;
the llama.cpp reference outputs behind the fidelity rows were captured
elsewhere (EXP-004).

| | |
| --- | --- |
| CPU | Intel Core Ultra 9 185H: 6 P-cores (SMT), 8 E-cores, 2 LP E-cores, 22 threads |
| ISA | AVX2, F16C, FMA, AVX-VNNI; no AVX-512 |
| RAM | 16 GB (14.98 GiB `MemTotal`) |
| Swap | zram, 7.5 GiB; off inside every benchmark cgroup |
| Storage | Micron 2400 `MTFDKBA1T0QFM-1BD1AABGB`, DRAM-less QLC, PCIe Gen4 x4 |
| Drive bandwidth | about 1.6 GB/s at decode's read geometry: 1.565 to 1.694 GB/s, median 1.633, over all 48 expert files (EXP-024); 1.54 to 2.37 GB/s across the probe matrix (EXP-019) |
| Filesystem | btrfs, `compress=zstd:3,ssd,discard=async`, data profile `single` |
| OS | Arch Linux, cgroup v2 |
| `RLIMIT_MEMLOCK` | 8 MiB soft and hard (systemd default) |
