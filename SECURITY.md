# Security policy

## Reporting a vulnerability

Use GitHub's private security advisory. Go to the repository's **Security**
tab and choose **Report a vulnerability**. That keeps the report private while
it is being fixed.

Do not open a public issue for a suspected vulnerability.

Include what you did, what happened, and what you expected instead. A file or
a script that reproduces it is worth more than a description. If the bug is in
parsing, attach the input that triggers it.

There is no bounty. Expect a first reply within a week. This is a pre-v0
project maintained by one person, so a fix can take considerably longer than
the reply.

## Supported versions

ramvamp is pre-v0. Only `main` is supported and there are no backports.

## Deployment posture

Read this before deciding whether something is a vulnerability here.

`ramvamp serve` binds `127.0.0.1` and nothing else. **There is deliberately no
flag to change it.** The reason is specific: the HTTP layer has neither a
header-size cap nor a read timeout, so a single connection sending an endless
header line grows a buffer until the process dies. This process is meant to
run inside a `memory.max=3G` cgroup next to a 30B model, where "grows a
buffer" means the OOM killer takes the model with it. Loopback is the
mitigation, and it is documented at the top of `crates/server/src/http.rs`.

There is no authentication and no TLS. Anyone who needs the server reachable
from another host should put a real reverse proxy in front of it. Request size
limits, timeouts, TLS and authentication belong there.

ramvamp is not intended for production, multi-user, or security-critical
deployments. It is a local endpoint for one user on one machine.

## Security properties

These are claims the code makes. A demonstrated break of one is a valid
report.

**`ramvamp-core` does not panic on untrusted input.** Manifests, layouts, the
packed `.rvmp` bytes on disk and GGUF headers are untrusted until verified.
Every count, length and offset parsed from a file is bounds-checked before
use, string, array and tensor counts are capped, arithmetic on parsed values
is checked, and failures are typed errors. A panic, an abort, or an allocation
whose size is driven by a file's own header field is a bug. Reachable from
input is enough; it does not need to be reachable over the network. The same
holds for the repacker's GGUF parser.

**Installed bytes are hash-verified.** `ramvamp-repack verify-install`
re-checks sizes, hashes and layout against the manifest. A way to get a
modified or truncated file past that check is in scope.

**The sanitizing prompt paths cannot fabricate a turn.** See
[the tokenization boundary](#the-tokenization-boundary) below, which is where
the exceptions are written down.

## In scope

Specific to this system:

- **`.rvmp` manifest and layout parsing.** Offsets, lengths and counts that
  drive a read or an allocation.
- **Malformed or hostile GGUF input**, to `ramvamp-repack install` or
  `ramvamp-repack inspect`, local or over the network.
- **Path traversal in the repacker.** Anything that writes outside the install
  directory, including through a repo name, revision or file name that is
  carried into a path or a URL.
- **Verification bypass.** A model file that has been changed and that
  `verify-install` still accepts.
- **io_uring submission and completion handling.** A completion attributed to
  the wrong slot, a buffer reused while a read is still in flight, a slot
  lease returned twice. Aliased buffers previously produced spurious btrfs
  checksum errors on this machine, which is why slots are handed out as owning
  leases from a free list instead of by index arithmetic.
- **Alignment assumptions in the AVX2 kernels.** Packed sub-tensor offsets can
  be only 2-byte aligned. A wide load on a layout the format permits is in
  scope.
- **Special-token injection through the chat template** on the sanitizing
  paths (`chat` and `serve`).
- **Credential or file-access surprises in the installer.** The installer
  makes anonymous HTTPS range requests and reads no credentials today. Any
  path that would send credentials, read them off the machine, or touch a file
  outside the install directory and its lock is in scope.
- **Input-driven memory exhaustion** on a path the loopback restriction does
  not already cover, in particular an unbounded download or an unbounded
  allocation during install.

## Out of scope

Named here so nobody spends time writing them up:

- **Model quality.** Wrong, biased or offensive generated text is a property
  of the weights.
- **Incorrect generated output**, including hallucination, refusals, and
  disagreement with another runtime within the recorded numerics tolerances.
- **Expected resource use.** The runtime reads gigabytes per token by design
  and runs the CPU flat out. "It used 2.9 GiB of a 3.0 GiB budget" is the
  design working.
- **Performance regressions.** Those are real bugs and belong in a normal
  issue with a cold, in-cgroup measurement attached.
- **Denial of service against the server from a host that already has loopback
  access.** The missing header cap and read timeout are known, documented, and
  the reason the listener is loopback-only.
- **Anything that requires the attacker to already control the machine, the
  model directory, or the config file.**

## The tokenization boundary

This is subtle and it is deliberate, so it is written down rather than left to
be discovered.

The chat template renders reference-faithfully on the validation paths and
sanitizes on the interactive ones.

**`generate --messages-file` and `tokenize --messages-file` are
reference-faithful.** They go through `RvmpTokenizer::encode_chat`, where a
literal `<|im_start|>` in message content encodes to the real control id,
exactly as transformers and llama.cpp do. That is the point of those paths:
their output is what `scripts/compare_llamacpp.py` compares against llama.cpp,
and sanitizing there would change the bytes under comparison and quietly
invalidate it. The consequence is not silent. Content that will encode to
control ids is named on stderr, with the message index and the role, before
the model sees it.

Treat `--messages-file` as a trusted local file that the invoking user wrote.
Do not point it at content from somewhere else. A control token that reaches
the model through that flag is the documented behaviour of that flag, not a
vulnerability.

**`chat` and `serve` sanitize instead.** Both read input they do not control,
and both feed their own output back into the next prompt, so a control-token
literal must not be able to fabricate a turn on the next round. They use
`encode_chat_sanitized` and `encode_chat_with_tools_sanitized`, which break
added-token literals with a zero-width marker, on user turns and assistant
turns alike. In `chat` the transcript is sanitized once on the way in, so
`/reset` cannot restore an unsanitized seed. In the server, `prompt` uses the
sanitizing encoder and nothing else.

A way to get a real control id into the prompt through `chat` or `serve` is a
vulnerability. Report it.
