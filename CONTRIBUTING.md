# Contributing to ramvamp

ramvamp streams MoE expert weights from NVMe so that Qwen3-30B-A3B runs on a
CPU in about 4 GB of RAM.

Read [docs/roadmap.md](docs/roadmap.md) for what is planned,
[docs/architecture.md](docs/architecture.md) before touching the runtime, and
[docs/comparison.md](docs/comparison.md) for why the design is what it is.

**The most useful contribution right now is a benchmark report from a drive
other than the reference drive.** See
[Benchmark reports from other drives](#benchmark-reports-from-other-drives).

## The gate

CI runs these commands, in this order, on pull requests and on pushes to
`main`, and a change merges only when all of them pass.

```bash
./scripts/check-docs.sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
```

CI also builds the workspace in debug, builds and tests the `pread` fallback
with `ramvamp-core`'s `io-uring` feature off, and builds both on the minimum
supported Rust, 1.88:

```bash
cargo build --all-targets --workspace
cargo build --all-targets -p ramvamp-core --no-default-features
cargo test -p ramvamp-core --no-default-features
cargo +1.88 build --all-targets --workspace
cargo +1.88 build --all-targets -p ramvamp-core --no-default-features
```

Keep the `-p ramvamp-core`. At the workspace root, `--no-default-features`
leaves `io-uring` on, because the other crates depend on the core with default
features, so the `pread` path is never compiled. Run these after touching
`crates/core/src/io`.

Zero warnings is the standard: fix a clippy warning rather than allowing it.
None of the gate needs the model. `cargo test` runs on a fresh clone with no
install and no network, and a test that needs a real install does not belong
in it.

## Hard rules

The architectural invariants are listed in [AGENTS.md](AGENTS.md#hard-rules).
A PR that breaks one is rejected on that ground, whatever it gains.

## Performance changes

A performance PR reports a baseline, a result and a verdict, measured by the
[method in docs/benchmarks.md](docs/benchmarks.md#method), with both arms run
back to back in one session. Warm-cache runs are diagnostics, and negative
results count.

## Benchmark reports from other drives

Every published throughput figure comes from one drive, the
[reference machine](docs/benchmarks.md#reference-machine)'s, and decode is
I/O-bound, so a report from a second drive is the cheapest experiment left. It
needs no code.

Run the method in [docs/benchmarks.md](docs/benchmarks.md) with the repacker's
default model pin: do not pass `--repo`, `--revision` or `--file`. Check the
generated text, because a run that loops or truncates is not a speed result.
Then file the
[Benchmark report](https://github.com/y0sif/ramvamp/issues/new?template=benchmark_report.yml)
issue form, which lists every field a report needs.

Every usable report is credited by name in the results, with a link if you
want one. Reports that come back unclean or refute an expectation are credited
the same way.

## Scope

In scope:

- focused fixes, with a test
- documentation corrections
- benchmark reports, as above
- the v1 items in [docs/roadmap.md](docs/roadmap.md)

**Open an issue before starting anything large.** The roadmap has an order,
and some of its items are open decisions rather than tasks; it marks which. A
large PR against an item that has not been ordered yet is work that cannot be
merged.

Out of scope:

- **Anything that breaks the bounded-memory design.** Loading a checkpoint
  into RAM makes it a different project. So does a buffer whose size follows
  the model rather than the budget.
- **A dependency added without a stated reason.** Say what it does, what it
  replaces, and how many crates it pulls in.

## Pull requests

Keep them narrow: one change per PR. A fix bundled with a refactor takes much
longer to review and often does not get merged.

The description says:

- what changed
- how it was tested, including which gate commands you ran and any measurement
- what the limitations are, and what you did not test

Commit messages and branch names follow the
[conventions in AGENTS.md](AGENTS.md#conventions).

## Licensing

Contributions are dual licensed MIT OR Apache-2.0, matching the project.
Opening a PR means you agree to that. There is no CLA.
