# Security

## Reporting

Open a [GitHub issue](https://github.com/y0sif/ramvamp/issues/new/choose); there is no private
channel. Expect a first reply within a week.

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

`ramvamp serve` has no authentication, no TLS, and no header-size cap or read
timeout: one connection sending an endless header line can grow memory until
the process is killed. That is why it binds `127.0.0.1` and no flag changes it.
The body cap grows with `--context` ([server.md](docs/server.md#refusals)).
A server reachable from elsewhere belongs behind a reverse proxy, which is
where TLS, authentication and request limits go.

This is not built for production, multi-user, or security-critical use.
