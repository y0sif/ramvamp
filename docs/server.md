# The server

`ramvamp serve` is an OpenAI-compatible Chat Completions server on
`127.0.0.1`. It serves one request at a time and keeps the KV cache between
requests. This page covers its flags, endpoints, the request fields it honours
and what it refuses.

```bash
ramvamp serve --model ~/models/qwen3-30b-a3b.rvmp

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-30b-a3b.rvmp",
       "messages": [{"role": "user", "content": "Explain io_uring in two sentences."}],
       "stream": true}'
```

The port opens only once the model has loaded, so a client gets connection
refused until it is ready.

## Flags

| flag | default | what it does |
| --- | --- | --- |
| `--port <N>` | 8080 | TCP port on `127.0.0.1`. `0` picks a free port, which is logged only with `RUST_LOG=info` |
| `--served-model-name <NAME>` | the install directory's name | the id `/v1/models` lists |
| `--max-new <N>` | 512 | reply cap for a request that sends neither `max_completion_tokens` nor `max_tokens`. It is reserved against the context, so a large value shortens what a client can send |
| `--keepalive-secs <SECONDS>` | 10 | silence before a stream gets an SSE keep-alive comment. Keep it under 300, where common clients give up |
| `--greedy`, `--temperature`, `--top-k`, `--top-p`, `--seed` | off, then the model's sampling settings | defaults a request can override |

There is no `--host`; [SECURITY.md](../SECURITY.md#deployment-posture) says
why. The model dials (`--context`, `--cache-bytes`, `--profile` and the rest)
are in [configuration.md](configuration.md).

## Endpoints

| route | method | what it does |
| --- | --- | --- |
| `/v1/chat/completions` | `POST` | Chat Completions, streamed as SSE when `"stream": true`, buffered otherwise |
| `/v1/models` | `GET` | the single served model |
| `/health` | `GET` | liveness |

Any other path or method is a `404`. A query string is ignored when routing.

## Request fields

| field | effect |
| --- | --- |
| `messages` | required and non-empty. Roles `system`, `user`, `assistant`, `tool`. Content is a string or an array of `text` parts |
| `model` | required, echoed back in the reply, not checked against the served id |
| `stream` | stream the reply as SSE |
| `stream_options.include_usage` | add a final usage chunk to a stream |
| `temperature`, `top_p`, `top_k`, `seed` | override the server's defaults. `temperature` 0 or below is greedy. Without `seed`, each request gets the server's seed plus the request number |
| `max_completion_tokens`, `max_tokens` | reply cap; `max_completion_tokens` wins when both are sent |
| `tools`, `tool_choice` | tool calls, in streamed and buffered replies alike. `tool_choice` may be `auto`, `none` or absent |

Other fields, such as `logprobs` and `presence_penalty`, are accepted and
ignored.

## How requests are served

- **One at a time.** A request that arrives mid-generation waits in the accept
  queue until the current one finishes. It is not rejected.
- **The KV cache is reused** across requests by longest common prefix, so a
  repeated system prompt is not prefilled again.
- **Use `"stream": true` for long replies.** Only a stream gets keep-alive
  comments, and a buffered client may time out before its reply is complete.

Decode and prefill speeds are in [benchmarks.md](benchmarks.md). This is a
local endpoint for low-volume chat and completion, not an agentic coding
backend, where one turn is thousands of output tokens.

## Refusals

Each refusal carries an OpenAI-shaped error body with the `code` below. The
400s cover fields the server cannot honour as asked, where accepting them would
mean a wrong answer.

| status | `code` | cause |
| --- | --- | --- |
| 400 | `invalid_json` | the body is not JSON of the expected shape |
| 400 | `empty_messages` | `messages` is empty |
| 400 | `unsupported_value` | `n` above 1; a non-empty `stop`, since generation stops on the model's own stop tokens; `tool_choice` other than `auto` or `none`, which the chat template cannot force |
| 400 | `unsupported_content_part`, `invalid_content_part` | a content part that is not text, which the model would otherwise never see; a `text` part with no `text` |
| 400 | `invalid_tool_name` | a tool name in `tools` or in an assistant's `tool_calls` outside `^[A-Za-z0-9_-]{1,64}$` |
| 400 | `context_length_exceeded` | the prompt plus the reply cap is over `--context` |
| 404 | `not_found` | any other path or method |
| 413 | `request_too_large` | a body over 256 bytes per token of `--context`, with a 64 KiB floor (1 MiB at the default 4096) |
