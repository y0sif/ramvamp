# Third-party notices

A dated review of the third-party code ramvamp depends on.

**Reviewed 2026-08-09**, against the workspace at that date.

This file is an attribution aid. It is not legal advice, and it is not a
compliance artifact. If you are shipping ramvamp somewhere that needs one, do
your own review.

**Scope: direct dependencies only.** Everything below is declared in
`Cargo.toml`'s `[workspace.dependencies]` block or in a crate's own
`[dependencies]`. The transitive graph is not covered here; `cargo tree` and
`Cargo.lock` are the source for that. Licence strings are the crates' own
`license` fields as reported by `cargo metadata` on the review date, not a
reading of their licence texts.

ramvamp itself is licensed **MIT OR Apache-2.0**. See `LICENSE-MIT` and
`LICENSE-APACHE`.

## Internal crates

`ramvamp-core`, `ramvamp-repack` and `ramvamp-server` are path dependencies
inside this workspace. They carry the workspace licence, MIT OR Apache-2.0.

## Direct dependencies

Versions are the ones resolved in `Cargo.lock` on the review date.

| Crate | Version | Licence | Used by |
| --- | --- | --- | --- |
| `anyhow` | 1.0.104 | MIT OR Apache-2.0 | repack, cli |
| `clap` | 4.6.5 | MIT OR Apache-2.0 | repack, cli |
| `io-uring` | 0.7.13 | MIT OR Apache-2.0 | core |
| `libc` | 0.2.189 | MIT OR Apache-2.0 | core, cli |
| `memmap2` | 0.9.11 | MIT OR Apache-2.0 | core |
| `ratatui` | 0.30.2 | MIT | cli |
| `rustix` | 1.1.4 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | core |
| `serde` | 1.0.229 | MIT OR Apache-2.0 | core, repack, server |
| `serde_json` | 1.0.151 | MIT OR Apache-2.0 | core, repack, server, cli |
| `sha2` | 0.10.9 | MIT OR Apache-2.0 | core, repack |
| `thiserror` | 2.0.19 | MIT OR Apache-2.0 | core, repack, server |
| `tiny_http` | 0.12.0 | MIT OR Apache-2.0 | server |
| `tokenizers` | 0.21.4 | Apache-2.0 | core |
| `tracing` | 0.1.44 | MIT | core, repack, server |
| `tracing-subscriber` | 0.3.23 | MIT | repack, cli |
| `ureq` | 3.3.0 | MIT OR Apache-2.0 | repack |

Every licence above was established from crate metadata. There is no entry
whose licence could not be determined.

Three entries are worth calling out. `tokenizers` is Apache-2.0 only, so it is
the one dependency that does not offer an MIT option. `rustix` offers three
alternatives including Apache-2.0 with the LLVM exception. `tracing`,
`tracing-subscriber` and `ratatui` are MIT only.

`tiny_http` and `ratatui` are declared in `crates/server/Cargo.toml` and
`crates/cli/Cargo.toml` rather than in the workspace block. Everything else
comes from `[workspace.dependencies]`.

## What each one is used for

**`anyhow`**: the error type in the binaries. Library crates use `thiserror`
instead, per the workspace rule.

**`clap`**: argument parsing for `ramvamp` and `ramvamp-repack`, derive API.

**`io-uring`**: the submission and completion queues behind O_DIRECT expert
reads on Linux. Optional: it sits behind `ramvamp-core`'s default `io-uring`
feature, which is turned off for the portable `pread` fallback used by tests
and non-Linux development.

**`libc`**: the syscalls `rustix` does not expose. `mincore` and
`posix_fadvise` for the O_DIRECT capability probe, and the `SIGINT` handler in
`chat`. It was already in the tree through `memmap2`, which is why it was
preferred over `ctrlc` or `signal-hook`.

**`memmap2`**: memory-maps the always-resident common weights. Experts are
never mapped; they are read explicitly, which is a hard rule of the project.

**`ratatui`**: draws the pinned status panel in `chat --tui`. Default
features are off; only the `crossterm` backend is enabled. `crossterm` is not
listed separately anywhere, so the tree cannot end up with two versions of it.

**`rustix`**: safe syscall wrappers, with the `thread`, `process` and `mm`
features. Used for CPU affinity pinning, thread and process control, and
memory syscalls.

**`serde`**: derive-based serialization for the `.rvmp` manifest and layout,
config profiles, and the server's wire types.

**`serde_json`**: JSON everywhere it appears: the manifest, config profiles,
message files, chat-template tool schemas, and the OpenAI-compatible request
and response bodies. Built with the `preserve_order` feature, which swaps
`Value`'s object map to `IndexMap` so key order round-trips. Chat templates
render tool schemas the way Python's `json.dumps` does, in insertion order,
and sorting them would silently change the prompt the model sees.

**`sha2`**: SHA-256 for install integrity. Per-file hashes are written at
install time and re-checked by `verify-install`.

**`thiserror`**: derives the typed error enums in every library crate.

**`tiny_http`**: the server's only transport dependency, chosen for being
small. It brings `ascii`, `chunked_transfer` and `httpdate` and nothing else.
Its `Response` type is used for the buffered endpoints; the SSE path writes
chunked framing by hand through `Request::into_writer`.

**`tokenizers`**: the Hugging Face tokenizer, loading the pinned
`tokenizer.json`. The chat template itself is vendored in-repo
(`crates/core/src/tokenizer/chat.rs`) rather than taken from this crate.

**`tracing`**: structured logging in libraries and binaries. `println!` is
not used in library code.

**`tracing-subscriber`**: the subscriber the binaries install, with the
`env-filter` feature so `RUST_LOG` works.

**`ureq`**: HTTP client for the installer's ranged downloads, with the
`rustls` feature so there is no OpenSSL dependency.

## Model weights

**No model weights are in this repository, and none are redistributed with a
release.** The installer downloads them at install time, directly from Hugging
Face. They remain governed by their own licence and by the terms of the
repositories they come from. Read those terms yourself before using a model.

The v0 default is a frozen pin. `ramvamp-repack install` defaults to:

| | |
| --- | --- |
| GGUF repo | `bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF` |
| Revision | `6c6e8692f43e4ca663f7ece8229a1361090d3a4c` |
| File | `Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` |

The tokenizer is pinned separately, and to the upstream model repo rather than
to the quantization repo:

| | |
| --- | --- |
| Tokenizer repo | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Revision | `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` |

Both pins are constants in the source: `PIN_REPO`, `PIN_REVISION` and
`PIN_FILE` in `crates/repack/src/main.rs`, and `TOKENIZER_REPO` and
`TOKENIZER_REVISION` in `crates/repack/src/tokenizer_fetch.rs`. The chat
template replicated in `crates/core/src/tokenizer/chat.rs` comes from that
same tokenizer revision, and the test fixtures record it.

The repacker copies quantized bytes unchanged and never requantizes, so an
installed `.rvmp` directory holds the upstream quantization rearranged on
disk. The upstream terms apply to it unchanged.

## Regenerating this file

```bash
cargo metadata --format-version 1 --offline
```

Filter the `packages` array to the names declared in
`[workspace.dependencies]` plus each crate's own `[dependencies]`, and read
`license` and `version` off each. Update the review date at the top when you
do.
