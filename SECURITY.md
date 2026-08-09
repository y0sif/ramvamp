# Security

## Reporting

Use GitHub's [private vulnerability reporting](https://github.com/y0sif/ramvamp/security/advisories/new)
rather than a public issue. Expect a first reply within a week.

## What is worth reporting

ramvamp parses files downloaded from the internet and serves HTTP, so the
interesting surface is:

- `.rvmp` manifest and layout parsing, and the offset and length arithmetic
  that follows it
- GGUF parsing in the repacker, on malformed or hostile input
- path traversal in the installer
- verification bypasses, where a corrupted install passes its hash checks
- alignment assumptions in the AVX2 kernels, which operate on packed
  sub-tensor offsets that may be only 2-byte aligned
- special-token injection through the chat template

`ramvamp-core` is not supposed to panic on untrusted input. A panic reachable
from a crafted model file or a crafted request is a bug worth reporting even
if you cannot get further than a crash.

Not security issues: model output quality, expected high resource use, and
performance regressions.

## Deployment posture

`ramvamp serve` binds `127.0.0.1` and there is deliberately no flag to change
that. It has no authentication and no TLS, and the HTTP layer has no
header-size cap or read timeout, so one connection sending an endless header
line can exhaust memory next to a 30B model in a 3 GB budget. Put a reverse
proxy in front of it if it needs to be reachable from anywhere else.

This is not built for production, multi-user, or security-critical use.
