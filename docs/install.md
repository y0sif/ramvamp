# Install

This page covers requirements, the installer, the model install, paths,
building from source, uninstalling and troubleshooting. The install command
itself is in the [README](../README.md#install).

## Requirements

- **Linux on x86_64.** Expert reads use io_uring (kernel 5.1 or newer) and
  O_DIRECT, both Linux interfaces.
- **An NVMe SSD with 17.35 GiB free.** The repacker streams the download into
  place and keeps no second copy. Other storage works far more slowly, because
  decode reads up to about 1.1 GB of expert weights per token.
- **About 4 GB of free memory** at the default 4K context. `ramvamp plan`
  prints the figure for any setting; see
  [configuration.md](configuration.md#check-a-configuration-with-plan).
- **AVX2, FMA and F16C are optional.** They are detected at runtime, and a
  scalar fallback runs without them, much more slowly.
- **A systemd user session is optional.** With one, model commands run with
  swap off; see [configuration.md](configuration.md#the-swap-off-scope).

## What the installer checks

The [installer](../install.sh) refuses a machine it cannot support and warns
about one that will run slowly. Re-running it upgrades in place and reports the
version it replaced.

| check | on failure |
| --- | --- |
| Linux on x86_64 | **refuses**, installs nothing |
| `curl` and `tar` on PATH | **refuses** |
| download is a real tarball holding both binaries | **refuses** |
| kernel 5.1 or newer | warns: io_uring falls back to `pread` |
| AVX2 and F16C | warns: scalar kernels, much slower |
| an NVMe device exists | warns: the drive sets the speed |
| binary runs and its version matches the tag | warns |
| install directory is on `PATH` | warns, and says how to fix it |
| no other `ramvamp` is already on `PATH` | warns: whichever comes first on `PATH` runs |

| variable | default | meaning |
| --- | --- | --- |
| `RAMVAMP_VERSION` | latest release | pin a tag, for example `curl ... \| RAMVAMP_VERSION=v0.1.1 bash` |
| `RAMVAMP_INSTALL_DIR` | `/usr/local/bin`, or `~/.local/bin` with neither root nor `sudo` | where the two binaries go |

## Install the model

`ramvamp-repack` downloads the GGUF and repacks it in one streaming pass,
copying the quantized bytes unchanged.

```bash
# Download and repack. Repo, revision and file default to a pinned commit,
# which `ramvamp-repack install --help` prints.
ramvamp-repack install --output ~/models/qwen3-30b-a3b.rvmp

# Optional: full hash verification. Note --input here.
ramvamp-repack verify-install --input ~/models/qwen3-30b-a3b.rvmp
```

| subcommand | what it does | flags |
| --- | --- | --- |
| `inspect` | Parse a GGUF header, build the repack plan, and print a report. No data section is read | Exactly one of `--local <PATH>` or `--remote`; with `--remote`, `--repo`, `--revision` (default `main`, not the pin) and `--file` |
| `install` | Stream a GGUF source into an installed `.rvmp` directory | `--output <DIR>` (required); `--local <PATH>` for an offline source; `--repo`, `--revision`, `--file` for a different remote; `--window-mib <N>` (default 32, one HTTP range request per window); `--resume`; `--overwrite`; `--skip-verify`; `--skip-tokenizer`; `--tokenizer-dir <PATH>` |
| `fetch-tokenizer` | Fetch the pinned tokenizer files into a completed install | `--output <DIR>` (required); `--tokenizer-dir <PATH>` to copy from a local directory |
| `verify-install` | Re-verify an install: manifest, per-file size and hash, layout | `--input <DIR>` (required) |
| `discard-partial` | Delete the staging directory of an interrupted install | `--output <DIR>` (required) |

`install` fetches the tokenizer after the model. If that step fails, the model
install stands, and `fetch-tokenizer` adds the tokenizer later.

## Paths

| what | where |
| --- | --- |
| `ramvamp`, `ramvamp-repack` | `/usr/local/bin`, `~/.local/bin`, or `RAMVAMP_INSTALL_DIR` |
| the model | the `--output` directory; `<output>.partial` while an install runs |
| config file | `~/.config/ramvamp/config.json`, only if you create it ([configuration.md](configuration.md#named-profiles)) |

## From source

Building needs Rust 1.88 or newer and a C and C++ compiler (`build-essential`
on Debian or Ubuntu, `base-devel` on Arch), because `ring`, Oniguruma and
esaxx build native code. OpenSSL is not needed.

```bash
git clone https://github.com/y0sif/ramvamp
cd ramvamp
cargo build --release
```

The binaries land in `target/release/`. Copy them onto your `PATH`, or run them
as `./target/release/ramvamp`.

## Uninstall

Uninstalling is deleting the files in [Paths](#paths).

```bash
sudo rm "$(command -v ramvamp)" "$(command -v ramvamp-repack)"  # drop sudo for ~/.local/bin
rm -r ~/models/qwen3-30b-a3b.rvmp   # the model
rm -r ~/.config/ramvamp             # only if you created a config file
```

## Troubleshooting

The load banner on stderr ends with the I/O mode: `io_uring+O_DIRECT`,
`io_uring+buffered`, or `pread`. io_uring falls back to `pread` when the kernel
cannot create a ring. O_DIRECT falls back to buffered reads when the filesystem
refuses it or a startup probe cannot prove it bypasses the page cache.

- **Memory depends on O_DIRECT, not on io_uring.** Buffered reads fill the page
  cache, so `io_uring+buffered` does not hold about 4 GB. `pread` with O_DIRECT
  is slower and still bypasses the page cache.
- **Run as the user that owns the install.** For a file the process neither
  owns nor may write, the kernel hides page-cache residency, so the probe
  degrades to buffered.
- **`RUST_LOG=warn` shows why it degraded.** ramvamp logs only errors by default.
- **An interrupted install** continues with `install --resume`.
- **A configuration that does not fit** is refused at startup. Lower `--context`
  or `--cache-bytes`; see [configuration.md](configuration.md).
