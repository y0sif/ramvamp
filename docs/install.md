# Install

Covers how to get the two binaries, what ramvamp needs from a machine, the
three I/O modes it can end up in, and every `ramvamp-repack` subcommand beyond
the one-line install.

## The one-liner

```bash
curl -sSL https://y0sif.github.io/ramvamp/install.sh | bash
```

It resolves the latest release, downloads `ramvamp-linux-x86_64.tar.gz`, checks
that the binary it extracted reports the version the tag claims, and installs
`ramvamp` and `ramvamp-repack`. Then it stops: it prints the model install
command and does **not** run it, because that is a 17.35 GiB download and it is
not a script's decision to start.

What it refuses, and what it only warns about:

| check | on failure |
| --- | --- |
| Linux on x86_64 | **refuses**, installs nothing |
| `curl` and `tar` on PATH | **refuses** |
| download is a real tarball holding both binaries | **refuses** |
| kernel 5.1 or newer | warns: io_uring falls back to `pread` |
| AVX2 and F16C | warns: scalar kernels, much slower |
| an NVMe device exists | warns: the drive is the throughput |
| binary runs and its version matches the tag | warns |
| install directory is on `PATH` | warns, and says how to fix it |

| variable | default | meaning |
| --- | --- | --- |
| `RAMVAMP_VERSION` | latest release | pin a tag, for example `v0.1.0` |
| `RAMVAMP_INSTALL_DIR` | `/usr/local/bin`, or `~/.local/bin` with neither root nor `sudo` | where the two binaries go |

Re-running it upgrades in place, and says which version it replaced. If another
`ramvamp` is already on `PATH` somewhere else, it says that too rather than
letting the two shadow each other quietly.

The script is [`install.sh`](../install.sh) at the repository root. The copy
served from `y0sif.github.io/ramvamp/install.sh` is mirrored from it by
`.github/workflows/sync-installer.yml`, so the two cannot drift.

Prefer to do it by hand:

```bash
curl -sSL -o ramvamp.tar.gz \
  https://github.com/y0sif/ramvamp/releases/latest/download/ramvamp-linux-x86_64.tar.gz
tar xzf ramvamp.tar.gz
sudo install -m755 ramvamp ramvamp-repack /usr/local/bin/
```

## Requirements

- **Linux on x86_64.** io_uring and O_DIRECT are the point of the design.
- **An NVMe SSD** with about **17.35 GiB free** for the installed model. The
  installer streams the source GGUF into place: no doubling, no separate copy
  of the download kept on disk. Other storage works and is far slower, because
  decode reads up to about 1.1 GB of expert weights per token.
- **AVX2 and FMA are optional.** They are detected at runtime (F16C too, for
  attention) and there is a scalar fallback, so it runs without them, just
  slower.
- **Rust 1.88 or newer, only to build from source.** A prebuilt binary needs no
  toolchain at all. Edition 2024 itself needs only 1.85, but `ratatui`, which
  draws the `chat --tui` panel, needs 1.88, and cargo resolves the floor across
  the whole workspace.

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

## Build from source

Only needed for an unreleased commit, a local change, or a machine the prebuilt
binary will not run on. Needs Rust 1.88 or newer and nothing else: the tree has
no C dependencies, and `ureq` uses rustls rather than OpenSSL.

```bash
git clone https://github.com/y0sif/ramvamp
cd ramvamp
cargo build --release
```

Two binaries come out of a build: `target/release/ramvamp` and
`target/release/ramvamp-repack`. Every example below writes them as bare
commands, which is what they are after an install; from a source tree, prefix
them with `./target/release/`.

## The repacker

The repacker copies quantized bytes unchanged. It never requantizes, so
llama.cpp runs the identical bytes and validation compares arithmetic rather
than weights.

```bash
# Peek at the pinned source first. This parses the GGUF header over HTTP range
# requests only, so it costs nothing before a 17 GiB download.
ramvamp-repack inspect --remote

# Download and repack in one streaming pass. --output is the only flag you
# need: repo, revision and file all default to the pinned Qwen3-30B-A3B source.
ramvamp-repack install --output ~/models/qwen3-30b-a3b.rvmp

# Optional, and thorough: full hash verification of the install.
# Note --input here, --output above.
ramvamp-repack verify-install --input ~/models/qwen3-30b-a3b.rvmp
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
