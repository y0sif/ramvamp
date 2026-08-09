# Install

Covers what ramvamp needs from a machine, the three I/O modes it can end up
in, and every `ramvamp-repack` subcommand beyond the one-line install.

## Requirements

- **Linux on x86_64.** io_uring and O_DIRECT are the point of the design.
- **Rust 1.85 or newer** (edition 2024).
- **An NVMe SSD** with about **17.35 GiB free** for the installed model. The
  installer streams the source GGUF into place: no doubling, no separate copy
  of the download kept on disk.
- **AVX2 and FMA are optional.** They are detected at runtime (F16C too, for
  attention) and there is a scalar fallback, so it runs without them, just
  slower.

## The three I/O modes, and the banner that names yours

Two degradations are automatic, and the load banner tells you which of the
three modes you got: `io_uring+O_DIRECT`, `io_uring+buffered`, or `pread`.
io_uring falls back to `pread`, and O_DIRECT falls back to buffered reads if
the filesystem refuses the flag or the startup probe cannot prove it is real.

**The ~3 GB memory contract turns on O_DIRECT, not on io_uring.** Buffered
reads are charged to the page cache and expert traffic is most of the model, so
`io_uring+buffered` will not hold the budget. `pread` is the slow mode, but it
still bypasses the page cache when O_DIRECT was verified, so the contract
survives losing io_uring and does not survive losing O_DIRECT.

## Run as the user that owns the install

The kernel will not report page-cache residency for a file this process
neither owns nor may write, so on an install unpacked by root and run by you,
the startup probe cannot prove O_DIRECT is real, and it degrades to buffered
rather than assume.

## Build

```bash
git clone https://github.com/y0sif/ramvamp
cd ramvamp
cargo build --release
```

Two binaries come out of a build: `ramvamp` and `ramvamp-repack`.

## The repacker

The repacker copies quantized bytes unchanged. It never requantizes, so
llama.cpp runs the identical bytes and validation compares arithmetic rather
than weights.

```bash
# Peek at the pinned source first. This parses the GGUF header over HTTP range
# requests only, so it costs nothing before a 17 GiB download.
./target/release/ramvamp-repack inspect --remote

# Download and repack in one streaming pass. --output is the only flag you
# need: repo, revision and file all default to the pinned Qwen3-30B-A3B source.
./target/release/ramvamp-repack install --output ~/models/qwen3-30b-a3b.rvmp

# Optional, and thorough: full hash verification of the install.
# Note --input here, --output above.
./target/release/ramvamp-repack verify-install --input ~/models/qwen3-30b-a3b.rvmp
```

| subcommand | what it does | flags |
| --- | --- | --- |
| `inspect` | Parse a GGUF header, build the repack plan, and print a report. No data section is ever read | Exactly one of `--local <PATH>` or `--remote`; with `--remote`, `--repo`, `--revision` and `--file` override the defaults |
| `install` | Stream a GGUF source into an installed `.rvmp` directory | `--output <DIR>` (required); `--local <PATH>` for an offline source; `--repo`, `--revision`, `--file` for a different remote; `--window-mib <N>` (default 32, one HTTP range request per window); `--resume`; `--overwrite`; `--skip-verify`; `--skip-tokenizer`; `--tokenizer-dir <PATH>` |
| `fetch-tokenizer` | Fetch the pinned tokenizer files into an existing completed install and record them in its manifest | `--output <DIR>` (required); `--tokenizer-dir <PATH>` to copy from a local directory instead of downloading |
| `verify-install` | Re-verify an installed directory: manifest, per-file size and hash, layout cross-check | `--input <DIR>` (required) |
| `discard-partial` | Delete the `.partial` staging directory of an interrupted install | `--output <DIR>` (required) |

`install` fetches the tokenizer itself once the model is durable. A tokenizer
failure only prints a warning pointing at `fetch-tokenizer`, because the model
install has already succeeded by then.

Do not pass `--repo`, `--revision` or `--file` when reporting a benchmark: the
defaults are the frozen v0 pin, and a report on different weights is a
different measurement. See [benchmarks.md](benchmarks.md).

## After installing

`plan` prices a configuration before you run it, and refuses an impossible one
before anything is allocated. See [configuration.md](configuration.md) for the
dials it resolves, and [server.md](server.md) for the HTTP surface.
