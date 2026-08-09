```
 ||    _ __ __ _ _ __ _____   ____ _ _ __ ___  _ __     ||
 ||   | '__/ _` | '_ ` _ \ \ / / _` | '_ ` _ \| '_ \    ||
 ||:  | | | (_| | | | | | \ V / (_| | | | | | | |_) |  :||
 ||   |_|  \__,_|_| |_| |_|\_/ \__,_|_| |_| |_| .__/    ||
 ||                                           |_|       ||

```

<p align="center">
  <a href="https://github.com/y0sif/ramvamp/actions/workflows/ci.yml"><img alt="CI status" src="https://github.com/y0sif/ramvamp/actions/workflows/ci.yml/badge.svg"></a> <a href="#license"><img alt="License: MIT OR Apache-2.0" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg"></a> <img alt="Rust 1.85 or newer" src="https://img.shields.io/badge/rust-1.85%2B-orange.svg">
</p>

<p align="center">
  <strong>Run 26-30B MoE models in about 3 GB of RAM. No GPU required.</strong><br>A Rust runtime that streams experts from NVMe instead of holding them hostage in memory.
</p>

> **Status: v0, and it runs.** Qwen3-30B-A3B generates coherent text on CPU inside a 3 GB cgroup
> with a cold page cache, validated against llama.cpp on identical weights. The shipping surface
> is `ramvamp serve`: the OpenAI Chat Completions API on loopback, with streaming SSE, tool
> calls, and named configuration profiles.

ramvamp is a CPU-first local LLM inference runtime, written in Rust, for fine-grained
Mixture-of-Experts models. It runs a 30B-parameter quantized model on an ordinary Linux x86_64
machine with an NVMe SSD, no GPU and no 32 GB of RAM, by never loading the full checkpoint into
memory. The always-needed common weights stay memory-mapped; the routed experts, which are most
of the model, live on disk in a page-aligned packed format and are fetched with io_uring and
O_DIRECT only when the router asks for them, through a small per-layer LFU cache.
Qwen3-30B-A3B is the v0 model; Gemma 4 26B-A4B is the intended second model and is **roadmap,
not present tense**, since the repacker accepts `qwen3moe` architecture GGUFs and refuses
anything else by name.

## At a glance

Qwen3-30B-A3B Q4_K_M, cold, inside `memory.max=3G` with `memory.swap.max=0`, on the reference
machine (Intel Core Ultra 9 185H, CPU only, **Micron 2400 DRAM-less QLC** NVMe):

| | measured |
| --- | --- |
| Decode | **about 2 tok/s** on this DRAM-less QLC drive: 1.46 to 2.19 over ctx 64-3,961 (EXP-023), and 1.43 to 2.16 over the same rungs on a later branch (EXP-025) |
| Prefill | **11.25 tok/s** at ctx 512 (EXP-023) |
| Peak RAM | **2,497 to 2,929 MiB** of a 3,072 MiB ceiling: 2,497.0 at ctx 64 (EXP-025), 2,929.3 at ctx 3,961 (EXP-023) |
| Model on disk | 17.35 GiB, a **~6x** memory saving |
| Fidelity | mean full-vocab KL **1.04e-2** vs llama.cpp, top-1 agreement 8/8 (EXP-004) |

**Throughput is stated per drive on purpose**: decode is I/O-bound and this part sustains about
1.6 GB/s at decode's own read geometry, so no figure appears without its drive. Full table,
conditions and drive detail: [docs/benchmarks.md](docs/benchmarks.md).

## Quickstart

```bash
git clone https://github.com/y0sif/ramvamp
cd ramvamp
cargo build --release

# Download and repack the pinned Qwen3-30B-A3B in one streaming pass (17.35 GiB on disk).
./target/release/ramvamp-repack install --output ~/models/qwen3-30b-a3b.rvmp

# Generate. Text streams to stdout; timing and expert stats go to stderr.
./target/release/ramvamp generate --model ~/models/qwen3-30b-a3b.rvmp \
    --prompt "Explain io_uring in two sentences."
```

`chat` is the terminal REPL and `serve` the HTTP endpoint. Requirements, the three I/O modes and
the repacker's other subcommands: [docs/install.md](docs/install.md).

## Serve and configure

```bash
./target/release/ramvamp serve --model ~/models/qwen3-30b-a3b.rvmp --port 8080
```

That is `/v1/chat/completions` (streaming SSE and buffered), `/v1/models` and `/health`, on
`127.0.0.1` with no `--host` flag, on purpose. It serves one request at a time, reuses the KV cache
across requests by longest common prefix, and refuses fields it cannot honour exactly as typed
`400`s rather than accepting them quietly. Routes, refusals and positioning: [docs/server.md](docs/server.md).

`--context` (default **4096**) is the dial that trades conversation length against resident
bytes, and `plan` prices any value of it without loading the model; a 32K profile projects about
5,649 MiB and needs a bigger budget than 3 GB, which is what named profiles exist to express.
Profiles, `--profile` / `--config` / `--no-config`, the four-layer precedence chain and the three
environment variables: [docs/configuration.md](docs/configuration.md).

## How it works

1. **Install repacks, it does not convert.** The GGUF is streamed into the `.rvmp` layout: a
   `common.bin` for what every token needs, one file per layer holding 128 fixed-stride,
   page-aligned expert blobs. Nothing is requantized and no shard is materialized in heap memory.
2. **Common weights are mmap'd.** Embeddings, attention, routers, norms and the lm_head are
   about 1,023 MiB, touched every token, and the page cache keeps them resident.
3. **Experts are read explicitly, never demand-paged.** Per layer the router picks 8 of 128.
   Hits in that layer's LFU cache dispatch to the compute pool immediately; the misses go to
   io_uring as one batch of O_DIRECT reads, so they bypass the page cache and cost the memory
   budget nothing, and the hit compute hides their latency.
4. **The cache is small on purpose.** At the default 1,440 MiB budget that is **11 slots per layer
   out of 128 experts**. Worst case is **1,097 MB of expert weights per decode token**, and the
   cache absorbs 53.0% to 59.3% of requests across five measured context rungs (EXP-023).

Explicit reads rather than mmap demand paging is a measured choice, and so is having no speculative
cross-layer expert prefetch; [docs/landscape.md](docs/landscape.md) carries both measurements.
[docs/architecture.md](docs/architecture.md) is the design and [docs/roadmap.md](docs/roadmap.md) the plan of record.

## Workspace

| Crate | Purpose |
| --- | --- |
| `ramvamp-core` | Runtime library: packed format, expert streaming, LFU cache, CPU kernels, KV cache, generation |
| `ramvamp-repack` | Streaming installer: ranged Hugging Face downloads repacked directly into the `.rvmp` layout |
| `ramvamp-server` | OpenAI-compatible HTTP layer: request validation, SSE framing, tool-call extraction, KV prefix reuse |
| `ramvamp` | CLI: `serve`, `chat`, `generate`, `plan` |

## Development

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

All four must pass before a push. [CONTRIBUTING.md](CONTRIBUTING.md) covers the rest, including
the measurement rules a performance change has to follow; security issues go through
[SECURITY.md](SECURITY.md).

## Prior art

Inspired by [TurboFieldfare](https://github.com/drumih/turbo-fieldfare), a Swift + Metal runtime
that proved the physics on Apple Silicon: Gemma 4 26B-A4B in about 2 GB at 5.1 to 6.3 tok/s on an
8 GB M2 Air. ramvamp exists because that proof is locked to Macs with M-series chips.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at
your option. Model weights are not included and remain governed by their own terms.
