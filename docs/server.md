# The server

Covers `ramvamp serve`: its routes, its concurrency model, what it refuses and
why, and what it is honestly workable for.

## A first request

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-30b-a3b",
       "messages": [{"role": "user", "content": "Explain io_uring in two sentences."}],
       "stream": true}'
```

## Routes

| route | method | what it does |
| --- | --- | --- |
| `/v1/chat/completions` | `POST` | Chat Completions, streaming SSE when `"stream": true` and buffered otherwise |
| `/v1/models` | `GET` | The single served model |
| `/health` | `GET` | Liveness |

## Three things worth knowing before you point a client at it

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

## Tool calls

Tool calls work in both the streaming and the buffered path. The two share one
parser and one id minter, and the renderer is byte-identical to 20 transformers
fixtures.

## What the server refuses, and why it refuses rather than approximates

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

## Position it honestly

At about 2 tok/s decode and about 11 tok/s prefill at ctx 512 on the reference
drive, this is a local 30B endpoint for a machine that could not otherwise run
one. It is workable for low-volume local chat and completion. It is not an
agentic coding backend, where a single turn is thousands of output tokens.

Those figures, and the conditions that make them valid, are in
[benchmarks.md](benchmarks.md). The context window a client can fill is a dial:
see [configuration.md](configuration.md).
