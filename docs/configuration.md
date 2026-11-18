# Configuration

ramvamp is configured with command-line flags, and a named profile saves a set
of them under one name. With no flags and no profile, every dial takes its
built-in default, which is the published 4K configuration.

## Check a configuration with `plan`

`ramvamp plan` prints every dial, where its value came from, and the memory the
configuration needs, without loading the model. It reads two small metadata
files and answers in well under a second.

```bash
ramvamp plan --model ~/models/qwen3-30b-a3b.rvmp --context 32768
```

| configuration | projected memory |
| --- | --- |
| defaults: `--context 4096`, 19 expert slots per layer | about 4,007 MiB |
| `--context 32768` | about 6,695 MiB |
| `--cache-bytes 1440M`, 11 slots per layer | about 2,961 MiB, fits a 3 GiB cap |

The projection is judged against total RAM or the tightest cgroup
`memory.max`, whichever is smaller. A configuration that does not fit makes
`plan` exit nonzero, naming every term, so a script can ask without parsing the
output. A model command refuses the same configuration before allocating its
caches.

## Named profiles

A profile names a combination of dials, so a small chat setup and a
wide-context agent setup become `--profile chat` and `--profile agent`.

```json
{
  "version": 1,
  "default_profile": "chat",
  "profiles": {
    "chat":  { "context": 4096,  "cache_bytes": "1440M" },
    "agent": { "context": 16384, "cache_bytes": "1440M", "placement": "per-core" }
  }
}
```

The file is read from `--config <PATH>` if given, else
`$XDG_CONFIG_HOME/ramvamp/config.json`, else `~/.config/ramvamp/config.json`.
Nothing creates it. `version` is required and must be `1`.

| flag | what it does |
| --- | --- |
| `--profile <NAME>` | Select a profile. Unset: the file's `default_profile`, or the built-in defaults. A name the file does not define is an error listing the ones it does |
| `--config <PATH>` | Read this file instead of the search path. A file named here must exist |
| `--no-config` | Ignore any config file. Environment variables and flags still apply |

A missing file means the built-in defaults. A file that does not parse is an
error naming the path, and so is an unknown key at the root or in a profile.
A profile may carry six keys: `cache_bytes`, `context`, `placement`,
`prefill`, `prefill_chunk` and `threads`. Each is checked exactly as its flag
is, so `"1440M"` means what `--cache-bytes 1440M` means.

## Precedence

Lowest to highest: **built-in default, then the profile, then the environment
variable, then the flag.** `plan` shows which of the four set each dial.

## Every dial

These flags are shared by `generate`, `chat`, `serve`, `logits`, `perplexity`
and `plan`. The integrity flags have no profile key, so a config file cannot
turn checking off.

| flag | default | profile key | variable | what it sets |
| --- | --- | --- | --- | --- |
| `--context <TOKENS>` | 4096 | `context` | `RAMVAMP_CONTEXT` | context window |
| `--cache-bytes <BYTES>` | `2485M`, 19 slots per layer | `cache_bytes` | none | expert-cache budget; binary suffixes `K`, `M`, `G`, `T`, for example `1440M` or `2.4G` |
| `--threads <N>` | one per physical performance core on a hybrid CPU, else one per usable CPU | `threads` | none | compute threads, counting the decode thread; 1 to 256 |
| `--placement <MODE>` | `shared` | `placement` | `RAMVAMP_PLACEMENT` | where threads run: `shared` (any performance core, or any efficiency core with an L3, so the scheduler can dodge a busy one), `per-core` (one pinned per performance core, faster only on an idle machine), `off` (no affinity) |
| `--prefill <MODE>` | `sweep` | `prefill` | `RAMVAMP_PREFILL` | `sweep` passes over each layer's experts once per chunk of prompt positions; `token-major` runs one forward pass per prompt token through the expert cache |
| `--prefill-chunk <N>` | 512 | `prefill_chunk` | `RAMVAMP_PREFILL_CHUNK` | prompt positions per sweep; lowered automatically when the cache cannot stage that many |
| `--common-memory <MODE>` | `resident` | none | none | `resident` reads the common weights into memory once; `mapped` memory-maps them, so the kernel may drop and re-read them |
| `--skip-hashes` | off | none | none | skip SHA-256 checks at load; size checks still run |
| `--verify-layer-hashes` | off | none | none | also hash every expert file on first open, a full read of the expert set on each start |

`--context` sizes the KV cache, so it trades conversation length against
memory. A prompt plus the tokens reserved for its reply must fit inside it, and
a value above the model's trained length is refused, because positions past it
produce fluent nonsense rather than an error.

`generate`, `chat` and `serve` also take `--max-new` (128, or 512 for `serve`),
`--greedy`, `--temperature`, `--top-k`, `--top-p` and `--seed`. Sampling
defaults come from the model; for the pinned one, temperature 0.7, top-p 0.8,
top-k 20. `serve`'s own flags are in [server.md](server.md#flags), and
`ramvamp <command> --help` lists everything.

## Environment variables

| variable | effect |
| --- | --- |
| `RAMVAMP_CONTEXT`, `RAMVAMP_PLACEMENT`, `RAMVAMP_PREFILL`, `RAMVAMP_PREFILL_CHUNK` | the dials in the table above |
| `RAMVAMP_NO_SCOPE` | set to any value: run without the [swap-off scope](#the-swap-off-scope) |
| `RAMVAMP_ASCII` | any value but `0`: ASCII glyphs in `chat --tui`. `TERM=dumb` or a non-UTF-8 locale does the same |
| `RUST_LOG` | log filter. ramvamp logs only errors by default; `RUST_LOG=warn` or `info` shows more. `ramvamp-repack` defaults to `info` |

An unusable value in a dial variable is ignored, so an old shell setting
cannot stop the binary from starting. `RAMVAMP_CONTEXT` and `RAMVAMP_PLACEMENT`
print a warning to stderr when ignored. `RAMVAMP_PREFILL` and
`RAMVAMP_PREFILL_CHUNK` warn only in the log, which the default level hides:
set `RUST_LOG=warn`, or run `plan`, whose source column then does not say
`env`.

## The swap-off scope

Model commands run with swap turned off for their own process, so nothing
else on the machine can push the common weights out to swap. `generate`,
`chat`, `serve`, `logits` and `perplexity` re-launch themselves once inside a
systemd user scope with `MemorySwapMax=0`. No root is needed, and the terminal,
signals and exit status stay the command's own. The scope sets no memory cap.

It is skipped when swap is already off for the process, when
`systemd-run --user --scope` does not work (no systemd user session), or when
`RAMVAMP_NO_SCOPE` is set to any value. `plan` and `tokenize` never re-launch.

```bash
RAMVAMP_NO_SCOPE=1 ramvamp chat --model ~/models/qwen3-30b-a3b.rvmp
```

Throughput and memory by configuration are in [benchmarks.md](benchmarks.md).
The memory contract behind the projection is in
[architecture.md](architecture.md#memory-contract).
