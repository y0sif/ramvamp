# Contributing to ramvamp

ramvamp streams fine-grained MoE expert weights from NVMe so a 26-30B model
runs on a CPU in about 3 GB of RAM. That constraint is the project. Most of
the rules below exist because a reasonable-looking change would quietly break
it.

Read `docs/roadmap.md` before starting anything. It is the plan of record.
`docs/architecture.md` covers the runtime and `docs/landscape.md` covers why
the design is what it is.

**The most useful contribution right now is a benchmark report from a drive
that is not the reference drive.** See
[Benchmark reports from other drives](#benchmark-reports-from-other-drives).

## The gate

Four commands. All four pass before a push, and CI runs the same four.

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build
```

Zero warnings is the standard. `-D warnings` is not advisory, and "clippy is
being picky here" is not an exception.

CI builds with `--release` where the local gate builds debug, and it also
builds with `ramvamp-core`'s `io-uring` feature turned off. If you touched
anything under `crates/core/src/io`, run
`cargo build --all-targets --no-default-features` locally as well.

None of these need the model. `cargo test` runs on a fresh clone with no
`.rvmp` install and no network; the tokenizer fixtures it needs are vendored
in `crates/core/src/tokenizer/fixtures/`. A test that only passes with a real
17 GiB install does not belong in `cargo test`.

## Hard rules

These are architectural invariants. A PR that breaks one is rejected on that
ground, and "it is faster this way" does not change it. Each has a reason.

**Experts are read explicitly. Never mmap demand paging.**
Common weights are memory-mapped. Routed experts are read with `pread` or
io_uring. TurboFieldfare measured mmap at 3.54x slower per cold expert read,
and about 8x slower end to end in their full-token simulator. Those are two
separate measurements; see `docs/landscape.md`. A demand-paged expert also
gives the runtime no control over queue depth, and total bytes in flight is
the variable that predicts read throughput on this hardware.

**No full model, shard, or tensor is ever materialized in heap memory.**
This holds in the runtime and in the repacker. The repacker streams a 17 GiB
GGUF through a bounded window and never holds a tensor. The memory ceiling is
the whole product. One `Vec` sized by a header field is how it stops being
the product.

**The repacker copies quantized bytes unchanged. It never requantizes.**
An installed `.rvmp` is the upstream quantization byte for byte, rearranged on
disk. That is what makes the fidelity comparison against llama.cpp on
identical weights mean anything. A repacker that touches the numbers turns
every numerics gate into a comparison of two different models.

**No speculative cross-layer expert prefetch.**
Cross-layer predictability measured about 7%. A prefetch at that hit rate
spends read bandwidth that decode needs, and decode is I/O-bound: expert reads
are about half of every token.

**Library code returns typed errors with `thiserror`. Binaries use `anyhow`.**
`ramvamp-core` does not panic on untrusted input. Manifests, layouts, GGUF
headers and the bytes on disk are untrusted until verified. Every count,
length and offset parsed from a file is bounds-checked before use, and no
allocation is sized directly by a parsed value. In library code that means no
`unwrap`, no `expect`, no unchecked indexing and no unchecked arithmetic on
anything that came out of a file.

**Vectorized kernels document and test the alignment they assume.**
Packed sub-tensor offsets can be as little as 2-byte aligned. A wide load that
assumes more than the format guarantees is a fault waiting for a different
layout. State the assumption in the module and cover it in a test.

**A change claiming identical math must produce identical bits.**
"Only the I/O changed" and "only the loop order changed" are testable claims,
and `scripts/bitident.py` tests them. Bit-identity catches a stale cached
expert or a reordered reduction that a KL tolerance would sit above, because
the tolerance is above the intra-engine float-reordering noise floor. If your
change is meant to be arithmetically neutral, show the fingerprint matched. If
it is not neutral, say so in the PR and expect the numerics gates to be re-run.

## Measurement rules

This is where a well-meaning change most often goes wrong. A number taken
outside these conditions is not usable, however carefully it was taken.

**Published numbers come from cold runs inside a `memory.max=3G` cgroup with
`memory.swap.max=0`.** zram counts as swap. `scripts/cold_bench.py` sets that
up and then verifies it rather than assuming it: eviction is confirmed with
`mincore` instead of trusted from the `fadvise` return code, and every run is
classified CLEAN, DIRTY or unknown. Unknown counts as DIRTY. A fast number
from a DIRTY run is not a result.

**Warm-cache runs are diagnostics.** They are useful while iterating. They are
never published and never quoted.

**Every performance change gets an entry in `docs/experiments.md`** with a
baseline, a result and a verdict. Negative results go in too. A recorded
failure is the cheapest way to stop a bad idea from coming back, and this
project has re-derived rejected ideas more than once.

**Figures from different sessions never go on one curve.** Re-running a
byte-identical binary a day later read 3.1% slower at ctx 512 and 8.9% slower
at ctx 3,961. A ratio is quotable only when both arms ran back to back, with
the same prompt and the same dials, inside one session. If your "before"
number is from last week, you do not have a before number.

## Benchmark reports from other drives

The project's central claim is that throughput scales with the device. It has
only ever been measured on one drive: a Micron 2400, DRAM-less QLC, measured
cold and in-cgroup at 1.54 to 2.37 GB/s across the whole probe matrix, whose
top end comes from large sequential reads the runtime never issues. At decode's
own geometry, one expert blob at a time in random order, the same drive
sustains about 1.6 GB/s, and that is the figure to quote when reasoning about
decode. Decode is I/O-bound, so that drive is in every published decode figure.

A mainstream TLC Gen4 part should do materially better. Nobody has measured
it, and this project does not publish numbers it has not measured. One report
from a second drive turns a published band into a curve. It is the cheapest
remaining experiment in the project and it does not require writing any code.

### Protocol

Follow it exactly. A report that skips a step is not comparable with the
reference numbers, which is the only thing that makes it worth having.

**1. Build release.**

```bash
cargo build --release
```

**2. Install the model onto the drive under test.** This is a 17.35 GiB
download.

```bash
cargo run --release --bin ramvamp-repack -- install --output models/qwen3.rvmp
cargo run --release --bin ramvamp-repack -- verify-install --input models/qwen3.rvmp
```

Do not pass `--repo`, `--revision` or `--file`. The defaults are the frozen v0
pin. A report on different weights is a different measurement.

**3. Run the harness cold and in-cgroup.** Decode:

```bash
scripts/cold_bench.py --ramvamp target/release/ramvamp --rvmp models/qwen3.rvmp \
    --max-new 64 --repeats 5 --json scratch/cold-decode.json
```

Prefill at a fixed context, using a prompt fixture and one generated token:

```bash
scripts/cold_bench.py --ramvamp target/release/ramvamp --rvmp models/qwen3.rvmp \
    --prompt-file <path-to-prompt.txt> --max-new 1 --repeats 5 \
    --json scratch/cold-prefill.json
```

The harness needs Linux 6.5 or newer, cgroup v2 and `systemd-run --user`. It
refuses to start while any `ramvamp` process is alive, because `fadvise`
evicts nothing from a file another process holds mmap'd. Exit code 0 means
every scored run was CLEAN. Exit 1 means at least one was DIRTY. Exit 2 means
the measurement could not be taken at all.

**4. Report the hygiene counters next to every number.** The harness prints
and records all of them. A throughput figure without these is not reviewable:

- the per-run hygiene verdict, and the harness exit code. Report DIRTY runs
  rather than dropping them.
- `memory.peak` and `memory.swap.peak`.
- the `pgsteal` breakdown from `memory.stat`. `pgsteal_khugepaged` is excluded
  from the verdict; everything else counts as reclaim under pressure.
- `read_bytes` for the run. A delta near zero on a supposedly cold run means
  eviction failed and you measured a warm run.
- the eviction line: MiB resident before and after, across how many files.

Attach the summary JSON. It already carries the argv, the prompt hash, the
child's stderr and every counter, so a rule correction can be re-applied
without another cold run.

**5. Check the generated text.** Decoding samples by default (the v0 pin's
temperature 0.7, top-p 0.8, top-k 20), so pass `--greedy` as below; that makes
the output deterministic for a given prompt and model, and any run reproduces
it:

```bash
target/release/ramvamp generate --model models/qwen3.rvmp \
    --prompt "The capital of France is" --max-new 64 --greedy
```

Paste the output into the report. A reply that loops, repeats a phrase, or
truncates mid-word is not a valid speed result. A broken decode is fast for
the wrong reason, and a tok/s figure taken from one is worse than no figure.

### What the report must contain

The full spec of the machine, because every one of these has moved a number on
the reference box:

- **Drive**: exact model string, capacity, interface generation and lane
  count, DRAM or DRAM-less, NAND type if you know it, and how full it was.
- **Filesystem** and mount options, verbatim.
- **Kernel**: `uname -r`.
- **CPU**: model, core layout, and the relevant ISA flags from
  `/proc/cpuinfo` (`avx2`, `f16c`, `fma`, any `avx512*`).
- **RAM** total, and whether swap or zram is configured.
- Whether the machine was otherwise idle. Concurrent readers cost roughly 3.5x
  read amplification here, so a build running in another terminal invalidates
  the run.

Open an issue titled `benchmark: <drive model>` with the report and the
summary JSON attached.

### Credit

Every contributor who sends a usable report is named in the results, in the
repository, with a link if they want one. Reports that come back DIRTY or that
refute an expectation get credited the same way.

## Scope

In scope:

- focused fixes, with a test
- documentation corrections
- benchmark reports, as above
- the v1 candidates listed in `docs/roadmap.md`

**Open an issue before starting anything large.** The roadmap has an order,
and several of its open items are decisions for the author rather than tasks
anyone can pick up; the roadmap marks which. A large PR against an item that
has not been ordered yet is work nobody can merge, and that is a bad outcome
for the person who wrote it.

Out of scope:

- **Anything that breaks the bounded-memory design.** Loading a checkpoint
  into RAM makes it a different project. So does a buffer whose size follows
  the model rather than the budget.
- **A dependency added without a stated reason.** Say what it does, what it
  replaces, and how many crates it pulls in. `ratatui` was accepted at +31
  crates against a tree of 137, and that number was part of the decision.

## Pull requests

Keep them narrow. One change per PR. A fix bundled with a refactor takes much
longer to review and often does not get merged.

The description says:

- what changed
- how it was tested, including which gate commands you ran and any measurement
- what the limitations are, and what you did not test

Commits use imperative mood with a `feat:` / `fix:` / `chore:` / `docs:` /
`perf:` prefix. Branch off `main`, named `feat/...`.

## Releases

For maintainers. Publishing to crates.io is done by hand, not by CI, because
crates.io has no undo: a version can be yanked but never republished under the
same number, so a half-finished automated run is unrecoverable.

1. Bump `version` in the root `Cargo.toml`, run the gate, commit.
2. Tag `vX.Y.Z` and push the tag. The release workflow refuses a tag whose
   version does not match the workspace, then builds and attaches the Linux
   x86_64 tarball.
3. Publish in dependency order, waiting for the sparse index between each.
   crates.io acknowledges a publish before the index serves it, so a
   back-to-back run fails when the next crate cannot resolve the one just
   uploaded.

```bash
cargo publish -p ramvamp-core
# wait for the index, then
cargo publish -p ramvamp-repack
cargo publish -p ramvamp-server
cargo publish -p ramvamp
```

`ramvamp-repack` is a leaf and nothing resolves it, so only `ramvamp-core` and
`ramvamp-server` have to be in the index before the crate after them.

## Licensing

Contributions are dual licensed MIT OR Apache-2.0, matching the project.
Opening a PR means you agree to that. There is no CLA.
