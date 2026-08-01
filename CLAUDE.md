# ramvamp

Rust runtime for streaming fine-grained MoE experts from NVMe: 26-30B models
in ~3 GB RAM, CPU-first (AVX2), Linux-first (io_uring). v0 model:
Qwen3-30B-A3B. Read `docs/architecture.md` before touching the runtime and
`docs/landscape.md` for why design decisions were made.

## Commands

```bash
cargo build                                  # debug build (opt-level 1)
cargo test                                   # unit tests
cargo clippy --all-targets -- -D warnings    # lint, zero warnings
cargo fmt --check                            # format check
```

Pre-push: all four must pass.

## Workspace

- `crates/core` (`ramvamp-core`): runtime. Modules: `format` (packed .rvmp
  model), `io` (io_uring streamer + LFU cache), `kernels` (CPU backend behind
  a trait), `model` (arch config + forward pass), `kv`, `generate`.
- `crates/repack` (`ramvamp-repack`): streaming HF-to-.rvmp installer.
- `crates/cli` (`ramvamp`): user-facing binary.

## Hard rules

- Explicit reads for experts, never mmap demand paging (measured 8x slower
  upstream). Common weights are mmap'd; experts are pread/io_uring'd.
- No speculative cross-layer expert prefetch (measured ~7% predictability).
- No full model, shard, or tensor may ever be materialized in heap memory,
  in the runtime or the repacker.
- The repacker copies quantized bytes unchanged; it never requantizes.
- Library code returns typed errors (`thiserror`); binaries use `anyhow`.
  No panics in `ramvamp-core` on untrusted input.
- Every performance change gets an entry in `docs/experiments/README.md`
  (baseline, result, verdict). Published numbers come from cold runs inside
  a `memory.max=3G` cgroup with `memory.swap.max=0` (zram counts as swap);
  warm-cache runs are diagnostics.
- Vectorized kernels must document and test the alignment they assume;
  packed sub-tensor offsets may be only 2-byte aligned.

## Conventions

- Commits: imperative mood with `feat:` / `fix:` / `chore:` / `docs:` /
  `perf:` prefixes.
- `main` is the integration branch; feature branches as `feat/...`.
- Structured logging via `tracing`; no `println!` in library code.
- User's shell is fish: no `&&` chaining, use `;`. Rust toolchain via rustup.
