<h1 align="center">ramvamp</h1>

<p align="center">
  <strong>Run 26-30B MoE models in about 2 GB of RAM. No GPU required.</strong><br>
  A Rust runtime that streams experts from NVMe instead of holding them hostage in memory.
</p>

> **Status: pre-v0.** Nothing runs yet. This repository goes public when the
> first honest benchmark exists: coherent Qwen3-30B-A3B chat at 3+ tok/s
> inside a 2.5 GB cgroup, token-validated against llama.cpp.

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
