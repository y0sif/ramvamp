# Configuration

Covers the `--context` dial, named profile files, and the four-layer
precedence chain that decides which value of a dial actually runs.

## `--context`, the dial that costs memory

`--context` (default **4096**) is the dial that trades conversation length
against resident bytes. The KV cache is sized at it, and a prompt plus the
tokens reserved for its reply must fit inside it. It is refused above what the
model was trained for, because positions past that generate fluent nonsense
rather than an error.

`plan` will price any value of it for you without loading a single weight:

```bash
./target/release/ramvamp plan --model ~/models/qwen3-30b-a3b.rvmp --context 32768
```

A 32K profile projects about **5,649 MiB**, about 5.5 GiB, and needs a bigger
budget than 3 GB. That is what profiles exist to express: the 3 GB contract is
a property of the 4K default, not of the runtime. `plan` exits nonzero when a
configuration will not fit, naming every term, so a script can ask the question
without parsing the answer.

## Named profiles

A profile is a *name* for a combination of dials, so that a small chat setup
and a wide-context agent setup are `--profile chat` and `--profile agent`
rather than four flags each, remembered by hand.

```json
{
  "version": 1,
  "default_profile": "chat",
  "profiles": {
    "chat":  { "context": 4096,  "cache_bytes": "1440M" },
    "agent": { "context": 16384, "cache_bytes": "1440M" }
  }
}
```

The file is read from `--config <PATH>` if given, else
`$XDG_CONFIG_HOME/ramvamp/config.json`, else `~/.config/ramvamp/config.json`.

| flag | what it does |
| --- | --- |
| `--profile <NAME>` | Select a named profile. Unset: the file's `default_profile`, or the built-in defaults when there is neither. A name the file does not define is an error listing the ones it does |
| `--config <PATH>` | Read this file instead of the search path. A file named here has to exist; one merely searched for does not |
| `--no-config` | Ignore any config file and use the built-in defaults |

A missing file is not an error: it means the built-in defaults, which are the
measured, published 4K configuration. Nothing ships a file, writes one, or
creates the directory. A file that exists and does not parse *is* an error
naming the path, and so is an unrecognized key, at the root and inside every
profile: `"contxt": 16384` would otherwise widen nothing and say nothing.

The five keys a profile may carry are `cache_bytes`, `context`, `prefill`,
`prefill_chunk` and `threads`, spelled as their flags with `-` written `_`.
`cache_bytes` goes through the same parser `--cache-bytes` uses, so `"1440M"`
in the file means exactly what `--cache-bytes 1440M` means.

**`--skip-hashes` and `--verify-layer-hashes` are deliberately not profile
keys.** They are per-invocation decisions about what to check, not a
configuration worth naming, and a config file that could turn integrity
checking off is a config file worth not having.

## Precedence

Lowest to highest: **built-in default, then the profile file, then the
environment variable, then the explicit flag.**

A file the user edited months ago must not beat a variable they exported in
this shell, and neither may beat what they typed on this command line.

| variable | dial |
| --- | --- |
| `RAMVAMP_CONTEXT` | `--context` |
| `RAMVAMP_PREFILL` | `--prefill` (`sweep` or `token-major`) |
| `RAMVAMP_PREFILL_CHUNK` | `--prefill-chunk` |

An unusable value in any of them is ignored rather than fatal, so a shell
profile set years ago cannot stop the binary from starting. It is not ignored
silently: the warning goes to stderr, because a variable that was meant to
widen the window and did nothing is exactly the failure a user would otherwise
attribute to the model.

What each configuration costs in throughput and resident bytes is in
[benchmarks.md](benchmarks.md); the memory contract behind the projection is in
[architecture.md](architecture.md).
