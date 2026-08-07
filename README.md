<h1 align="center">ramvamp</h1>

<p align="center">
  <strong>Run 26-30B MoE models in about 3 GB of RAM. No GPU required.</strong><br>
  A Rust runtime that streams experts from NVMe instead of holding them hostage in memory.
</p>

> **Status: pre-v0, and it runs.** Qwen3-30B-A3B generates coherent text on
> CPU inside a 3 GB cgroup with a cold page cache, validated against
> llama.cpp on identical weights. What is left before this goes public is a
> shipping surface, not a working runtime.

Measured on the reference machine — Intel Core Ultra 9 185H, CPU only, 16 GB
RAM, **Micron 2400 DRAM-less QLC** NVMe — running Qwen3-30B-A3B Q4_K_M with a
cold page cache inside `memory.max=3G` and `memory.swap.max=0`:

| | measured |
| --- | --- |
| Decode | **about 2 tok/s** — 1.46 to 2.19 over ctx 64–3,961 in one session, 1.43 to 2.16 over the same rungs in another |
| Prefill | **11.25 tok/s** at ctx 512 |
| Peak RAM | **2.5–2.9 GiB** of a 3.0 GiB ceiling |
| Model on disk | 17.35 GiB — a **~6x** memory saving |
| Fidelity | mean full-vocab KL **1.04e-2** vs llama.cpp, top-1 agreement 8/8 |

Those are two separate sessions and deliberately not merged into one curve:
re-running the *byte-identical* binary a day later read 3.1% slower at ctx 512
and 8.9% slower at 3,961. On a DRAM-less QLC part, a decode figure describes
its session as well as its device.

**Throughput is stated per drive on purpose.** Decode is I/O-bound — expert
reads are about half of every token — and this reference part is DRAM-less
QLC measuring 1.54–2.37 GB/s. A mainstream TLC Gen4 drive should do
materially better; by how much is unmeasured, and this project does not
publish numbers it has not measured. Every figure above comes from a cold run
inside the benchmark cgroup; the method and the full experiment record are in
[docs/experiments](docs/experiments/README.md).

ramvamp runs fine-grained Mixture-of-Experts models, Qwen3-30B-A3B first and
Gemma 4 26B-A4B next, without loading the full checkpoint into memory. The
always-needed common weights stay memory-mapped; the routed experts, most of
the model, live on SSD in a page-aligned packed format and are fetched with
io_uring + O_DIRECT only when the router asks for them, through a small
per-layer LFU cache.

The name: in music, *vamping* is holding a repeating riff while the soloist
gets ready. That is the runtime's whole trick, keep computing on the weights
already in RAM while the experts stream in from disk. Also, it revamps what
your RAM can hold.

The approach is inspired by
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare), a Swift + Metal
runtime that proved the physics on Apple Silicon (26B in ~2 GB at 5-6 tok/s on
an 8 GB M2 Air). ramvamp exists because that proof is locked to Macs with
M-series chips. See [docs/landscape.md](docs/landscape.md) for the full
competitive picture and [docs/architecture.md](docs/architecture.md) for the
design.

## Workspace

| Crate | Purpose |
| --- | --- |
| `ramvamp-core` | Runtime library: packed format, expert streaming, LFU cache, CPU kernels, KV cache, generation |
| `ramvamp-repack` | Streaming installer: ranged Hugging Face downloads repacked directly into the `.rvmp` layout |
| `ramvamp` | CLI: chat and raw completion |

## Development

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option. Model weights are not included and
remain governed by their own terms.
