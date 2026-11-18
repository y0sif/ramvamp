# ramvamp: agent guide

ramvamp: Run 26-30B MoE models in about 4 GB of RAM. No GPU required. A Rust runtime that
streams experts from NVMe, CPU-first (AVX2 with a scalar fallback) and Linux-first (io_uring
with O_DIRECT). Qwen3-30B-A3B is the only model that runs; the repacker refuses other
architectures.

## Build and check

Run these before every push. They are CI's `check` job, in its order.

```sh
./scripts/check-docs.sh                       # docs rules, see Docs below
cargo fmt --check
cargo clippy --all-targets -- -D warnings     # warnings are errors
cargo test
cargo build --release
```

- The rest of the CI gate, the `pread` fallback and MSRV legs, is in
  [CONTRIBUTING.md](CONTRIBUTING.md#the-gate). Run it after touching `crates/core/src/io`.
- The MSRV is `rust-version` in the root `Cargo.toml`. Change it and the CI job together.
- The tokenizer test fixtures are vendored in `crates/core/src/tokenizer/fixtures/`.

## Layout

| Path | Holds |
|---|---|
| `crates/core/` (`ramvamp-core`) | the runtime library; `benches/` holds the criterion benches |
| `crates/core/src/format/` | the `.rvmp` installed model: manifest, expert layout, verification |
| `crates/core/src/io/` | expert streaming: io_uring + O_DIRECT reads, slot pool, LFU cache, prefill sweep, common weights |
| `crates/core/src/kernels/` | CPU backend: quantized GEMV, attention, quant formats, primitives |
| `crates/core/src/model/` | weights, decode forward pass, chunked prefill |
| `crates/core/src/{threads.rs,kv,tokenizer,generate}` | CPU topology and compute pool; KV cache; tokenizer and chat template; generation loop |
| `crates/repack/` (`ramvamp-repack`) | streaming HF-to-.rvmp installer: `gguf/` parser, `plan.rs`, `install/` executor and resume state |
| `crates/server/` (`ramvamp-server`) | OpenAI-compatible HTTP surface, library only; `ramvamp serve` runs it |
| `crates/cli/` (`ramvamp`) | the binary: `main.rs` (subcommands, flags), `config.rs` (profiles), `scope.rs` (swap-off re-launch), `repl.rs`, `tui/` |
| `scripts/` | validation against llama.cpp, the cold benchmark harness, probes, fixture generators, `check-docs.sh` |
| `docs/` | user, design and measurement docs |
| `site/`, `install.sh` | landing page and installer, copied to gh-pages by `sync-installer.yml` |

## Hard rules

- Experts are read explicitly, with `pread` or io_uring. Never mmap demand paging.
- No speculative cross-layer expert prefetch: copying one layer's routing choices predicted 7%
  of the next layer's experts. Both rules rest on TurboFieldfare's measurements, in
  [docs/comparison.md](docs/comparison.md).
- No full model, shard or tensor is ever materialized in heap memory, in the runtime or the
  repacker, and no buffer is sized by the model rather than the budget. The one exception is
  the resident common core: one anonymous mapping of `common.bin`, read once at load with the
  page cache dropped behind it, and counted in full by the footprint projection (`CommonMemory`
  in `crates/core/src/io/mod.rs`; it is the default, `--common-memory mapped` the opt-out).
- The repacker copies quantized bytes unchanged. It never requantizes.
- Library code returns typed errors (`thiserror`); binaries use `anyhow`. `ramvamp-core` does
  not panic on untrusted input: manifests, layouts, GGUF headers and bytes on disk. Bounds-check
  every parsed count, length and offset, and size no allocation directly by a parsed value:
  no `unwrap`, `expect`, or unchecked indexing or arithmetic on anything read from a file.
- A change claiming identical math produces identical bits: `ramvamp logits` output is
  byte-identical before and after, for short prompts and one longer than a 512-token prefill
  chunk (`scripts/bitident.py` checks both). A change that is not neutral says so in its PR. The
  kernel numerics rule is in `crates/core/src/kernels/mod.rs`.
- Vectorized kernels document and test the alignment they assume. Packed sub-tensor offsets
  may be only 2-byte aligned.
- `generate` and `logits` keep their exact stdout and stderr shape: the validation and
  benchmark scripts in `scripts/` parse it.
- Every performance change gets a row in [docs/experiments.md](docs/experiments.md): baseline,
  result, verdict, negative results included. Published numbers come from cold runs inside a
  `memory.max=4G` cgroup with `memory.swap.max=0` (zram counts as swap), by the method in
  [docs/benchmarks.md](docs/benchmarks.md#method). Warm-cache runs are diagnostics.

## Conventions

- Commits: imperative mood with a `feat:`, `fix:`, `perf:`, `docs:` or `chore:` prefix
  (`test:` and `ci:` also appear).
- Branches off `main`: `feat/`, `fix/`, `perf/`, `docs/`, `chore/`, and `exp/` for measured
  work kept for the record. Changes reach `main` through a PR with green CI.
- Logging: `tracing`. No `println!` in library code.

## Releases

crates.io is published by hand, not by CI: a version can be yanked but never republished.

1. Check the README and `site/index.html` against what ships. Bump `version` in the root
   `Cargo.toml` (`[workspace.package]` and the three internal `[workspace.dependencies]`
   entries), run the gate, and merge it with `Cargo.lock` through a PR as
   `chore: release vX.Y.Z`.
2. Tag the merge commit `vX.Y.Z` and push the tag. `release.yml` refuses a tag that does not
   match the workspace version, then builds and attaches the Linux x86_64 tarball.
3. Publish in dependency order. crates.io acknowledges a publish before the sparse index
   serves it, so a back-to-back run fails to resolve the crate just uploaded.

```sh
cargo publish -p ramvamp-core      # then wait until the index serves it
cargo publish -p ramvamp-repack    # a leaf: nothing resolves it
cargo publish -p ramvamp-server    # then wait again
cargo publish -p ramvamp
```

## Docs

- One fact, one home. Link to it instead of repeating it. Usage lives in the README and
  `docs/`; how a subsystem works, and its invariants, in the owning module's `//!` comment.
- Budgets: README 220 lines, this file 150, CONTRIBUTING, SECURITY and a `docs/` page 300
  (`docs/configuration.md`, the flag hub, 500). Cut before adding.
- `docs/roadmap.md` is updated in place: rewrite its status and items, never append a phase
  section.
- No fix history (issue or PR numbers, "since #N") on any docs page. The README, CONTRIBUTING,
  SECURITY, `docs/install.md`, `docs/configuration.md` and `docs/server.md` also carry no EXP
  ids, phase or round names, or test provenance; EXP ids belong in architecture, benchmarks,
  comparison, experiments and roadmap. Help strings follow the same rule, and neither they
  nor the user docs name scripts. Review enforces these last two; the script does not.
- `scripts/check-docs.sh` enforces the rest: markdown only at its allowlisted paths (a new
  `docs/` page is added there and to `.gitignore`), `CLAUDE.md` holding only `@AGENTS.md`, no
  personal paths, emails or session links in any file, no em or en dashes in markdown, and the
  budgets and wording rules above.
- A user-visible change updates its doc in the same change. Internal files (plans, handoffs,
  raw logs) are never tracked: run `git check-ignore -v <path>` before writing one.
- Code comments quote `docs/architecture.md` headings by name ("Decode loop", "Memory
  contract"). Keep a quoted heading's text, or update the comments that quote it.

## Where to read next

- `docs/roadmap.md` (the plan), `docs/architecture.md` (before touching the runtime),
  `docs/comparison.md` (why the design is what it is), and the `//!` of the module you change.
- `CONTRIBUTING.md` for submitting a change, `docs/benchmarks.md` for the measurement method.
