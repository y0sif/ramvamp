<h1 align="center">ramvamp</h1>

<p align="center">
  <a href="https://github.com/y0sif/ramvamp/actions/workflows/ci.yml"><img alt="CI status" src="https://github.com/y0sif/ramvamp/actions/workflows/ci.yml/badge.svg"></a> <a href="#license"><img alt="License: MIT OR Apache-2.0" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg"></a> <img alt="Rust 1.88 or newer" src="https://img.shields.io/badge/rust-1.88%2B-orange.svg">
</p>

<p align="center">
  <strong>Run 26-30B MoE models in about 4 GB of RAM. No GPU required.</strong><br>A Rust runtime that streams experts from NVMe.
</p>

ramvamp is a CPU-first local LLM inference runtime, written in Rust, for fine-grained
Mixture-of-Experts models. It loads about 1 GiB of common weights into memory once and fetches each
routed expert from NVMe with io_uring and O_DIRECT only when the router picks it. Its output is
validated against llama.cpp on identical weights, and `ramvamp serve` speaks the OpenAI Chat
Completions API on loopback.

It runs one model, Qwen3-30B-A3B (Q4_K_M, 17.35 GiB on disk), on Linux x86_64 with an NVMe SSD and
the CPU alone. The repacker refuses other architectures by name.

The trade is speed for memory. Decode runs at about 2.8 tokens per second, because each token can
read up to about 1.1 GB of expert weights from the SSD. On a machine with the RAM to hold all
17.35 GiB, a runtime that keeps the whole model in memory, such as llama.cpp, skips those reads.

## At a glance

Speed and memory are measured cold, with the page cache dropped first, inside a 4 GiB `memory.max`
cgroup with `memory.swap.max=0`, so neither free RAM nor swap can hold the model. Qwen3-30B-A3B
Q4_K_M with the default expert cache, on a DRAM-less QLC NVMe drive:

| | measured |
| --- | --- |
| Speed | decode **2.78 to 2.81 tok/s** and prefill **about 25 tok/s** at ctx 512; decode 1.96 tok/s at ctx 3,961 |
| Peak memory | **3,690 to 4,020 MiB** of cgroup `memory.peak`, page cache included: about 4.4x below the model's 17.35 GiB |
| Fidelity | mean full-vocab KL **1.04e-2** against llama.cpp, top-1 agreement 8/8, perplexity 6.36 against its 6.38 |

`--cache-bytes 1440M` fits a 3 GiB cap at about 2.2 tok/s decode. Full tables and the method:
[docs/benchmarks.md](docs/benchmarks.md).

## Install

The installer supports Linux x86_64 only. It puts `ramvamp` and `ramvamp-repack` in
`/usr/local/bin`, or in `~/.local/bin` without root or sudo.

```bash
curl -sSL https://y0sif.github.io/ramvamp/install.sh | bash
```

Re-run it to upgrade. To pin a release, end the command with `| RAMVAMP_VERSION=v0.1.1 bash`.
It stops before the model, a 17.35 GiB download that belongs on an NVMe SSD:

```bash
ramvamp-repack install --output ~/models/qwen3-30b-a3b.rvmp
```

<details>
<summary>Prebuilt tarball, by hand</summary>

Each [release](https://github.com/y0sif/ramvamp/releases/latest) has `ramvamp-linux-x86_64.tar.gz`,
holding both binaries, this README and both licences:

```bash
curl -sSLO https://github.com/y0sif/ramvamp/releases/latest/download/ramvamp-linux-x86_64.tar.gz
tar xzf ramvamp-linux-x86_64.tar.gz
sudo install -m755 ramvamp ramvamp-repack /usr/local/bin/
```

</details>

<details>
<summary>Build from source</summary>

Building needs Rust 1.88 or newer and a C and C++ compiler. The steps are in
[docs/install.md](docs/install.md#from-source).

</details>

## Chat in the terminal

`ramvamp chat` is a multi-turn chat: Ctrl-D or `/exit` leaves, and Ctrl-C stops the reply.
`/reset` forgets the conversation, `/save <path>` writes it as JSON, and `/help` lists the rest.

```bash
ramvamp chat --model ~/models/qwen3-30b-a3b.rvmp --tui
```

`--tui` adds a panel with prefill progress, the token rate, context use and the expert cache hit
rate, and needs a terminal on stdout. Without it, `chat` is plain text and works redirected.

## Serve and configure

`ramvamp serve` is an OpenAI-compatible HTTP server on 127.0.0.1, with no `--host` flag. It serves
`/v1/chat/completions` (streaming and buffered), `/v1/models` and `/health`, one request at a time,
and reuses the KV cache across requests that share a prefix. It answers a `400` to a value it cannot
honour and ignores fields it does not implement: [docs/server.md](docs/server.md).

```bash
ramvamp serve --model ~/models/qwen3-30b-a3b.rvmp --port 8080
```

`--context` (default 4096) sets the context length, and `ramvamp plan` prints the memory a setting
needs without loading the model. Profiles, flags, environment variables and their precedence:
[docs/configuration.md](docs/configuration.md).

## How it works

Each token needs about 1 GiB of common weights plus 8 of the 128 experts in each of 48 layers.

1. **Install repacks, it does not convert.** The GGUF becomes a `common.bin` and one file per layer
   of 128 page-aligned expert blobs, with quantized bytes copied unchanged.
2. **Common weights are read in once.** Embeddings, attention, routers, norms and the lm_head go
   into anonymous memory at load. `--common-memory mapped` maps them instead.
3. **Experts are read explicitly, never demand-paged.** Cache hits compute at once while the misses
   go to io_uring as one batch of O_DIRECT reads per layer, which bypass the page cache.
4. **The cache is small.** `--cache-bytes` defaults to 2485M, which holds 19 of each layer's 128
   experts and serves about two thirds of expert requests.

Design: [docs/architecture.md](docs/architecture.md). Why not mmap or cross-layer prefetch:
[docs/comparison.md](docs/comparison.md).

## Documentation

- [docs/install.md](docs/install.md): requirements, install paths, building from source, uninstalling
- [docs/configuration.md](docs/configuration.md): flags, environment variables and profile keys, with defaults
- [docs/server.md](docs/server.md): HTTP endpoints, honoured request fields and refusals
- [docs/benchmarks.md](docs/benchmarks.md): every published number, its conditions, the reference machine
- [docs/architecture.md](docs/architecture.md): packed format, streaming, cache, kernels and validation
- [docs/comparison.md](docs/comparison.md): how five related projects handle experts, and the measurements the design borrows
- [docs/experiments.md](docs/experiments.md): one line per experiment
- [docs/roadmap.md](docs/roadmap.md): the plan

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers building, testing and submitting a change, including what
a performance change must measure. Report security issues through [SECURITY.md](SECURITY.md).

## Prior art

ramvamp is inspired by [TurboFieldfare](https://github.com/drumih/turbo-fieldfare), a Swift and
Metal runtime that streams MoE experts from SSD on Apple Silicon.
[docs/comparison.md](docs/comparison.md) compares the two.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at
your option. Model weights are not included and remain governed by their own terms.
