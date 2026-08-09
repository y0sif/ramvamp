<h1 align="center">ramvamp</h1>

<p align="center">
  <a href="https://github.com/y0sif/ramvamp/actions/workflows/ci.yml"><img alt="CI status" src="https://github.com/y0sif/ramvamp/actions/workflows/ci.yml/badge.svg"></a>
  <a href="#license"><img alt="License: MIT OR Apache-2.0" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg"></a>
  <img alt="Rust 1.85 or newer" src="https://img.shields.io/badge/rust-1.85%2B-orange.svg">
</p>

<p align="center">
  <strong>Run 26-30B MoE models in about 3 GB of RAM. No GPU required.</strong><br>
  A Rust runtime that streams experts from NVMe instead of holding them hostage in memory.
</p>

> **Status: v0, and it runs.** Qwen3-30B-A3B generates coherent text on CPU
> inside a 3 GB cgroup with a cold page cache, validated against llama.cpp on
> identical weights. The shipping surface is here: `ramvamp serve` speaks the
> OpenAI Chat Completions API on loopback, with streaming SSE, tool calls, and
> named configuration profiles.

ramvamp is a CPU-first local LLM inference runtime, written in Rust, for
fine-grained Mixture-of-Experts models. It runs a 30B-parameter quantized model
on an ordinary Linux x86_64 machine with an NVMe SSD, no GPU and no 32 GB of
RAM, by never loading the full checkpoint into memory. The always-needed common
weights stay memory-mapped; the routed experts, which are most of the model,
live on disk in a page-aligned packed format and are fetched with io_uring and
O_DIRECT only when the router asks for them, through a small per-layer LFU
cache.

Qwen3-30B-A3B is the v0 model. Gemma 4 26B-A4B is the intended second model and
is **roadmap, not present tense**: the repacker accepts `qwen3moe` architecture
GGUFs and refuses anything else by name.

## Measured on the reference machine

Intel Core Ultra 9 185H (CPU only, 16 GB RAM, **Micron 2400 DRAM-less QLC**
NVMe) running Qwen3-30B-A3B Q4_K_M with a cold page cache inside
`memory.max=3G` and `memory.swap.max=0`:

| | measured |
| --- | --- |
| Decode | **about 2 tok/s**: 1.46 to 2.19 over ctx 64-3,961 (EXP-023), and 1.43 to 2.16 over the same rungs on a later branch (EXP-025) |
| Prefill | **11.25 tok/s** at ctx 512 (EXP-023) |
| Peak RAM | **2,497 to 2,929 MiB** of a 3,072 MiB ceiling: 2,497.0 at ctx 64 (EXP-025) and 2,929.3 at ctx 3,961 (EXP-023) |
| Model on disk | 17.35 GiB, a **~6x** memory saving |
| Fidelity | mean full-vocab KL **1.04e-2** vs llama.cpp, top-1 agreement 8/8 |

Those two ladders come from different binaries in different sessions, and are
deliberately not drawn as one curve. The drift is measured rather than assumed:
the later session also re-ran EXP-023's *byte-identical* binary at two rungs and
read **1.85 tok/s against 1.91** at ctx 512, and **1.33 against 1.46** at 3,961.
That is 3.1% and 8.9% slower a day later, on the same machine, with no code
change. On a DRAM-less QLC part a decode figure describes its session as well as
its device, which is why no single number appears here without one.

**Throughput is stated per drive on purpose.** Decode is I/O-bound: expert
reads are 44.4% of a decode token at ctx 512 and 33.2% at 3,961, and because
that bucket is measured as a residual it is a lower bound rather than the
drive's busy time. This reference part is DRAM-less QLC and sustains about
**1.6 GB/s** at decode's own read geometry: 1.60 to 1.67 GB/s re-measured over
the four probe files, and 1.565 to 1.694 GB/s with a median of 1.633 across all
48 expert files. A mainstream TLC Gen4 drive should do materially better; by how
much is unmeasured, and this project does not publish numbers it has not
measured. Every figure above comes from a cold run inside the benchmark cgroup;
the method and the experiment record are in
[docs/experiments.md](docs/experiments.md).

## Requirements

- **Linux on x86_64.** io_uring and O_DIRECT are the point of the design.
- **Rust 1.85 or newer** (edition 2024).
- **An NVMe SSD** with about **17.35 GiB free** for the installed model. The
  installer streams the source GGUF into place: no doubling, no separate copy
  of the download kept on disk.
- **AVX2 and FMA are optional.** They are detected at runtime (F16C too, for
  attention) and there is a scalar fallback, so it runs without them, just
  slower.

Two degradations are automatic, and the load banner tells you which of the
three modes you got: `io_uring+O_DIRECT`, `io_uring+buffered`, or `pread`.
io_uring falls back to `pread`, and O_DIRECT falls back to buffered reads if
the filesystem refuses the flag or the startup probe cannot prove it is real.

**The ~3 GB memory contract turns on O_DIRECT, not on io_uring.** Buffered
reads are charged to the page cache and expert traffic is most of the model, so
`io_uring+buffered` will not hold the budget. `pread` is the slow mode, but it
still bypasses the page cache when O_DIRECT was verified, so the contract
survives losing io_uring and does not survive losing O_DIRECT.

Run ramvamp as the user that owns the install. The kernel will not report
page-cache residency for a file this process neither owns nor may write, so on
an install unpacked by root and run by you, the startup probe cannot prove
O_DIRECT is real, and it degrades to buffered rather than assume.

## Install

```bash
git clone https://github.com/y0sif/ramvamp
cd ramvamp
cargo build --release
```

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

The repacker copies quantized bytes unchanged. It never requantizes, so
llama.cpp runs the identical bytes and validation compares arithmetic rather
than weights.

## Run it

```bash
# Price the configuration before running it. Reads the manifest and the expert
# layout only, so it answers without loading a single weight.
# Exits nonzero if the configuration will not fit, naming every term.
./target/release/ramvamp plan --model ~/models/qwen3-30b-a3b.rvmp

# Chat in the terminal.
./target/release/ramvamp chat --model ~/models/qwen3-30b-a3b.rvmp

# Or serve it. This is the shipping surface.
./target/release/ramvamp serve --model ~/models/qwen3-30b-a3b.rvmp --port 8080
```

| command | what it is |
| --- | --- |
| `serve` | OpenAI-compatible HTTP endpoint on loopback. Streaming SSE, tool calls, KV reuse across requests |
| `chat` | Terminal REPL, one turn at a time |
| `generate` | One-shot completion; requires exactly one of `--prompt` or `--messages-file` |
| `plan` | Prices a configuration without loading the model, and refuses an impossible one before anything is allocated |

`tokenize` and `logits` also exist. They are development and validation
affordances, not user features, and their stdio is frozen because the
measurement scripts parse it.

## The server

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-30b-a3b",
       "messages": [{"role": "user", "content": "Explain io_uring in two sentences."}],
       "stream": true}'
```

| route | method | what it does |
| --- | --- | --- |
| `/v1/chat/completions` | `POST` | Chat Completions, streaming SSE when `"stream": true` and buffered otherwise |
| `/v1/models` | `GET` | The single served model |
| `/health` | `GET` | Liveness |

Three things worth knowing before you point a client at it:

- **It binds `127.0.0.1` and there is no `--host` flag, on purpose.** The HTTP
  layer has no header-size cap, so an endless header line is a memory hazard.
  Anything that needs to be reachable from elsewhere should sit behind a
  reverse proxy, which is also where TLS, authentication and request limits
  belong. There is no auth and no CORS in ramvamp itself.
- **One request at a time, structurally.** There is no worker pool: one thread
  accepts and handles inline, and a request arriving mid-generation waits in
  the accept queue rather than being rejected. At these speeds an immediate 503
  would just burn a client's retries against a request that holds the model for
  minutes.
- **The KV cache is reused across requests** by a longest-common-prefix match
  against the ids actually fed, so a repeated system prompt is not re-prefilled
  every turn.

Tool calls work in both the streaming and the buffered path. The two share one
parser and one id minter, and the renderer is byte-identical to 20 transformers
fixtures.

<details>
<summary>What the server refuses, and why it refuses rather than approximates</summary>

Each of these is a typed `400` with an OpenAI-shaped error body, chosen over
accepting the field and quietly not honouring it:

| field | why |
| --- | --- |
| `n > 1` | One completion per request; there is one KV cache |
| `stop` | Generation ends on the model's own stop tokens, so a stop sequence would be accepted and never applied |
| forced `tool_choice` | The vendored chat template branches on `tools` alone and cannot force a named call. `auto`, `none` and omitting it are all fine |
| non-text content parts | The pinned ChatML template coerces non-string content to `""`, so dropping an `image_url` part would make the model answer a question it was never asked |

Prompt plus reserved output over the context cap is a `400` with OpenAI's own
`context_length_exceeded` code. Unknown and unimplemented fields
(`logprobs`, `presence_penalty` and friends) are accepted and ignored.

</details>

**Position it honestly.** At about 2 tok/s decode and about 11 tok/s prefill at
ctx 512 on the reference drive, this is a local 30B endpoint for a machine that could not
otherwise run one. It is workable for low-volume local chat and completion. It
is not an agentic coding backend, where a single turn is thousands of output
tokens.

## Configure it

`--context` (default **4096**) is the dial that trades conversation length
against resident bytes, and `plan` will price any value of it for you. A 32K
profile projects about 5,649 MiB and needs a bigger budget than 3 GB, which is
what profiles exist to express.

Named profiles live in `~/.config/ramvamp/config.json` and are selected with
`--profile`. `--config` points at a different file; `--no-config` ignores the
file layer entirely.

Precedence, lowest to highest: **built-in default, then the profile file, then
the environment variable, then the explicit flag.**

## How it works

1. **Install repacks, it does not convert.** The GGUF is streamed into the
   `.rvmp` layout: a `common.bin` for everything needed on every token, and one
   file per layer holding 128 fixed-stride, page-aligned expert blobs. Nothing
   is requantized and no shard is ever materialized in heap memory.
2. **Common weights are mmap'd.** Embeddings, attention, routers, norms and the
   lm_head are about 1,023 MiB, touched every token, and the page cache keeps
   them resident.
3. **Experts are read explicitly, never demand-paged.** Per layer, the router
   picks 8 of 128 experts. Hits in that layer's LFU cache dispatch to the
   compute pool immediately; the misses are submitted to io_uring as one batch
   of O_DIRECT reads, so they bypass the page cache and cost the memory budget
   nothing, and the hit compute is what hides their latency.
4. **The cache is small on purpose.** At the default 1,440 MiB budget that is
   **11 slots per layer out of 128 experts**, with frequency counters that
   survive eviction. Worst case is **1,097 MB of expert weights per decode
   token**, and the cache absorbs roughly half of the requests (53.0% to 59.3%
   measured across five context rungs), which is why the drive is the pacing
   item and why every throughput number here names the drive it was measured
   on.

Explicit reads rather than mmap demand paging is a measured choice, not a
preference: TurboFieldfare measured mmap 3.54x slower per cold expert read, and
about 8x slower end to end in their full-token simulator. There is no
speculative cross-layer expert prefetch either, because cross-layer routing
measured about 7% predictable.

See [docs/architecture.md](docs/architecture.md) for the design and
[docs/landscape.md](docs/landscape.md) for why the decisions went the way they
did. [docs/roadmap.md](docs/roadmap.md) is the plan of record.

## Workspace

| Crate | Purpose |
| --- | --- |
| `ramvamp-core` | Runtime library: packed format, expert streaming, LFU cache, CPU kernels, KV cache, generation |
| `ramvamp-repack` | Streaming installer: ranged Hugging Face downloads repacked directly into the `.rvmp` layout |
| `ramvamp-server` | OpenAI-compatible HTTP layer: request validation, SSE framing, tool-call extraction, KV prefix reuse |
| `ramvamp` | CLI: `serve`, `chat`, `generate`, `plan` |

Two binaries come out of a build: `ramvamp` and `ramvamp-repack`.

## Development

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

All four must pass before a push. [CONTRIBUTING.md](CONTRIBUTING.md) covers the
rest, including the measurement rules any performance change has to follow.
Security issues go through [SECURITY.md](SECURITY.md).

## The name

In music, *vamping* is holding a repeating riff while the soloist gets ready.
That is the runtime's whole trick: keep computing on the weights already in RAM
while the experts stream in from disk. Also, it revamps what your RAM can hold.

## Prior art

The approach is inspired by
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare), a Swift + Metal
runtime that proved the physics on Apple Silicon: Gemma 4 26B-A4B in about 2 GB
at 5.1 to 6.3 tok/s on an 8 GB M2 Air. ramvamp exists because that proof is
locked to Macs with M-series chips.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option. Model weights are not included and
remain governed by their own terms.
