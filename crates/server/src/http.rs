//! The HTTP transport: one thread, one socket, one request at a time.
//!
//! # Loopback only, and not configurable
//!
//! The listener binds `127.0.0.1` and there is no flag to change it. That is a
//! security decision, not an oversight: `tiny_http`'s `read_next_line` has
//! neither a header-size cap nor a read timeout, so a single connection
//! sending an endless header line grows a `Vec` until the process dies — and
//! this process is meant to run inside a `memory.max=3G` cgroup next to a 30B
//! model, where "grows a `Vec`" means the OOM killer takes the model with it.
//! Loopback is the mitigation. Anyone who wants this reachable from elsewhere
//! should put a real reverse proxy in front of it, which is also where request
//! limits, TLS and authentication belong.
//!
//! # Concurrency is one, structurally
//!
//! There is no semaphore and no worker pool. One thread calls `Server::recv`
//! in a loop and handles each request inline, which *is* the concurrency
//! limit: a request that arrives mid-generation waits in the accept queue
//! until the current one finishes. That is the desired behaviour, and it is
//! why [`ServerError::AtCapacity`] is never returned here.
//!
//! Rejecting on arrival would be much worse than it looks. The OpenAI SDKs
//! default to `max_retries = 2` with 0.5 s and 1 s backoff, so a client burns
//! all three attempts in about 1.5 s — against a request that occupies the
//! model for minutes. An immediate 503 therefore guarantees a hard failure,
//! where queueing guarantees an answer. `AtCapacity` stays in the error enum
//! for a future build that adds a *bounded* queue and can overflow it.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use thiserror::Error;
use tiny_http::{Header, Method, Request, Response, Server};

use crate::engine::{Engine, NullSink, Plan, StreamError, TokenSink, hush_stream_aborts, unix_now};
use crate::error::ServerError;
use crate::request::ChatCompletionRequest;
use crate::response::{
    COMPLETION_ID_PREFIX, ChatCompletion, ChunkBuilder, FinishReason, Health, ModelList,
    ResponseMessage,
};
use crate::sse::{sse_data, sse_done, sse_event};
use crate::wire::{KEEPALIVE, StreamState};

/// `POST` this to generate.
pub const PATH_CHAT_COMPLETIONS: &str = "/v1/chat/completions";
/// `GET` this for the one-model listing.
pub const PATH_MODELS: &str = "/v1/models";
/// `GET` this for a liveness probe.
pub const PATH_HEALTH: &str = "/health";

/// How the server was started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServeConfig {
    /// TCP port on `127.0.0.1`. `0` binds an ephemeral port.
    pub port: u16,
    /// How long a stream may go silent before a keep-alive comment is sent.
    ///
    /// Every client that matters gives up after 300 s of no bytes — undici's
    /// `bodyTimeout`, the `openai` provider's `headerTimeout`, and
    /// `CLAUDE_CODE_BYTE_STREAM_IDLE_TIMEOUT_MS` all land on the same number —
    /// and a 4K prompt takes about six minutes to prefill. Ten seconds is
    /// thirty times under the wall and costs fourteen bytes a tick.
    pub keepalive: Duration,
    /// Largest request body this server will buffer, in bytes.
    pub max_body_bytes: usize,
}

impl Default for ServeConfig {
    fn default() -> Self {
        ServeConfig {
            port: 8080,
            keepalive: Duration::from_secs(10),
            // Derived from the default 4K context, which is what it has always
            // worked out to: 1 MiB. `ramvamp serve` overrides this from the
            // window it actually resolved `--context` to.
            max_body_bytes: max_body_bytes_for_context(DEFAULT_CONTEXT),
        }
    }
}

/// The context window [`ServeConfig::default`] sizes itself against.
///
/// The CLI owns the real default and passes the resolved window in; this is
/// only what a library caller who sets nothing gets, and it is the same 4096.
pub const DEFAULT_CONTEXT: usize = 4096;

/// Request-body bytes allowed per token of context.
///
/// A prompt runs about 4 bytes of text per token, so 256 is sixty-four times
/// the text a completely full window can hold — headroom for JSON escaping,
/// for the `tools` array, and for the whitespace a pretty-printing client
/// sends. It is also small against a 3 GB budget at any window that budget can
/// host: a 32K context allows 8 MiB.
pub const BODY_BYTES_PER_CONTEXT_TOKEN: usize = 256;

/// Floor under [`max_body_bytes_for_context`].
///
/// The JSON envelope — role names, tool-call ids, the `model` string, the
/// framing — does not shrink with the context, so a deliberately tiny window
/// must not produce a cap that a well-formed request cannot fit inside.
pub const MIN_BODY_BYTES: usize = 64 << 10;

/// The request-body cap for a server configured with a `context`-token window.
///
/// This used to be a flat 1 MiB, justified by "a 4K context's whole prompt is
/// on the order of 16 KB of text". That reasoning does not survive `--context`
/// being a dial: an agent client that sends tool schemas against a 16K window
/// writes a request that approaches the old constant, and would have been
/// answered with a 413 for asking for exactly what the server was configured
/// to allow. Deriving it means the cap moves with the window instead.
///
/// Saturating, and floored at [`MIN_BODY_BYTES`], so no window produces a cap
/// that wraps or one too small to hold an envelope.
#[must_use]
pub fn max_body_bytes_for_context(context: usize) -> usize {
    context
        .saturating_mul(BODY_BYTES_PER_CONTEXT_TOKEN)
        .max(MIN_BODY_BYTES)
}

/// Why the server could not run. Per-request failures are [`ServerError`]s and
/// are answered, not returned.
#[derive(Debug, Error)]
pub enum ServeError {
    /// The port could not be bound.
    #[error("binding 127.0.0.1:{port}: {message}")]
    Bind {
        /// The port that was asked for.
        port: u16,
        /// What the listener reported.
        message: String,
    },
    /// The accept loop broke.
    #[error("accepting connections: {0}")]
    Accept(#[from] std::io::Error),
}

/// Serve until the process ends.
///
/// # Errors
///
/// [`ServeError::Bind`] if the port is taken, [`ServeError::Accept`] if the
/// listener dies. Everything a client can cause is answered as HTTP.
pub fn serve(engine: &mut dyn Engine, config: ServeConfig) -> Result<(), ServeError> {
    serve_with(engine, config, |_| {})
}

/// [`serve`], reporting the bound address before the first accept.
///
/// The hook exists so a test can bind port 0 and find out where it landed;
/// nothing in production passes anything but a no-op.
///
/// # Errors
///
/// Exactly [`serve`]'s.
pub fn serve_with(
    engine: &mut dyn Engine,
    config: ServeConfig,
    ready: impl FnOnce(SocketAddr),
) -> Result<(), ServeError> {
    hush_stream_aborts();
    // Loopback, always. See the module docs: there is no flag for this and
    // there must not be one until the header reader has a cap.
    let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, config.port));
    let server = Server::http(bind).map_err(|e| ServeError::Bind {
        port: config.port,
        message: e.to_string(),
    })?;
    let address = match server.server_addr().to_ip() {
        Some(address) => address,
        None => bind,
    };
    tracing::info!(%address, model = engine.model_id(), "listening");
    ready(address);

    loop {
        let request = server.recv()?;
        handle(engine, request, &config);
    }
}

/// Route one request. Never panics on anything a client can send.
fn handle(engine: &mut dyn Engine, request: Request, config: &ServeConfig) {
    // The query string is not part of the route: a client appending
    // `?api-version=...` (Azure-flavoured SDKs do) must still be served.
    let path = request
        .url()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_owned();
    let method = request.method().clone();
    tracing::debug!(%method, %path, "request");

    match (&method, path.as_str()) {
        (Method::Post, PATH_CHAT_COMPLETIONS) => chat_completions(engine, request, config),
        (Method::Get, PATH_MODELS) => {
            let body = ModelList::single(engine.model_id(), engine.created());
            respond_json(request, 200, &body);
        }
        (Method::Get, PATH_HEALTH) => respond_json(request, 200, &Health::ok()),
        _ => respond_error(
            request,
            &ServerError::NotFound {
                method: method.as_str().to_owned(),
                path,
            },
        ),
    }
}

/// `POST /v1/chat/completions`, streaming or not.
fn chat_completions(engine: &mut dyn Engine, mut request: Request, config: &ServeConfig) {
    let body = match read_body(&mut request, config.max_body_bytes) {
        Ok(body) => body,
        Err(e) => return respond_error(request, &e),
    };
    let parsed = match ChatCompletionRequest::from_json(&body) {
        Ok(parsed) => parsed,
        Err(e) => return respond_error(request, &e),
    };
    // Which optional capabilities this client actually asked for, logged
    // before anything is refused. Kept because the refusals above it are only
    // defensible against evidence: `stop` and a forced `tool_choice` are 400s
    // here, and whether that makes the server unusable depends entirely on
    // what the clients pointed at it send on every request.
    tracing::debug!(
        stop = parsed.stop_sequences().len(),
        tools = parsed.tools().len(),
        tool_choice = ?parsed.tool_choice,
        stream = parsed.stream(),
        "chat request"
    );
    // Everything that can be refused is refused here, before a status is
    // committed: once a stream's head is out, 200 is the answer whatever
    // happens next.
    let plan = match engine.prepare(&parsed) {
        Ok(plan) => plan,
        Err(e) => return respond_error(request, &e),
    };

    let id = completion_id();
    let created = unix_now();
    if parsed.stream() {
        stream_into(
            engine,
            &parsed,
            plan,
            request.into_writer(),
            &id,
            created,
            config.keepalive,
        );
    } else {
        // No head to write early and nothing to keep alive: a non-streaming
        // client asked for silence until the reply is complete, and at this
        // runtime's speeds that is very likely to hit its own timeout. That
        // is the client's choice to make, and `stream: true` is the fix.
        let mut sink = NullSink;
        match engine.run(plan, &mut sink) {
            Ok(completion) => {
                // The whole reply is in hand, so the `<tool_call>` blocks in it
                // can be turned into calls before anything is sent. The
                // streaming path cannot do this, having already sent the text,
                // which is why only this half is wired up.
                //
                // Gated on the request having declared tools. Without that, a
                // conversation *about* tool calls ("show me what a tool call
                // looks like") gets its example swallowed: the prose vanishes
                // into a structured call the client never enabled, and the
                // reply comes back `finish_reason: "tool_calls"` for a request
                // that offered none. The model only emits these blocks because
                // the tools branch of the template told it to, so when no tools
                // were sent the markers are ordinary text and stay that way.
                let extracted = if parsed.tools().is_empty() {
                    crate::toolcall::Extracted::none(&completion.text)
                } else {
                    crate::toolcall::extract(&completion.text)
                };
                let calls = crate::toolcall::wire_calls(&id, &extracted.calls);
                // A recovered call overrides the stop reason: the model
                // stopped because it wanted a tool, whatever token ended it.
                // Nothing recovered leaves the reason alone, so a block cut off
                // by the token budget still reports `length`.
                let finish_reason = if calls.is_empty() {
                    completion.finish_reason
                } else {
                    FinishReason::ToolCalls
                };
                let body = ChatCompletion::from_message(
                    id,
                    created,
                    &parsed.model,
                    ResponseMessage::calling(extracted.content, calls),
                    finish_reason,
                    completion.usage,
                );
                respond_json(request, 200, &body);
            }
            Err(e) => respond_error(request, &e),
        }
    }
}

/// Run `plan` and write an SSE stream into `writer`.
///
/// Split out from the `tiny_http` plumbing on purpose: `writer` is the only
/// thing this needs from the connection, so a test can hand it a recorder that
/// timestamps every flush and assert what a buffered implementation could
/// never pass — that the events arrive *when* the engine produced them.
pub fn stream_into<W: Write + Send + 'static>(
    engine: &mut dyn Engine,
    request: &ChatCompletionRequest,
    plan: Plan,
    writer: W,
    id: &str,
    created: u64,
    keepalive: Duration,
) {
    let state = Arc::new(Mutex::new(StreamState::new(writer)));

    // The head goes out before a single expert is read. A 4K prompt is about
    // six minutes of prefill and the clients give up after five, so the order
    // of these two statements is the difference between a working server and
    // one that is dead on arrival.
    if let Err(e) = lock(&state).head() {
        tracing::debug!(?e, "client left before the response head");
        return;
    }

    let builder = ChunkBuilder::new(id, created, &request.model);
    // The role chunk rides out with the head, so a client sees a well-formed
    // stream begin before the model has done any work at all.
    if emit(&state, &sse_or_empty(&builder.role())).is_err() {
        return;
    }

    let keeper = KeepAlive::spawn(Arc::clone(&state), keepalive);
    let mut sink = SseSink {
        state: Arc::clone(&state),
        builder: builder.clone(),
        keepalive,
    };
    let include_usage = request.include_usage();
    let outcome = engine.run(plan, &mut sink);
    drop(keeper);

    match outcome {
        Ok(completion) if completion.aborted => {
            tracing::debug!("client left mid-reply; the stream is abandoned unterminated");
        }
        Ok(completion) => {
            let _ = emit(
                &state,
                &sse_or_empty(&builder.finish(completion.finish_reason)),
            );
            // Usage goes immediately *before* the sentinel: a client stops
            // reading at `[DONE]`, so usage after it is usage nobody gets.
            if include_usage {
                let _ = emit(&state, &sse_or_empty(&builder.usage(completion.usage)));
            }
            let _ = emit(&state, &sse_done());
            let _ = lock(&state).finish();
        }
        Err(e) => {
            // The status is already spent — the head said 200 minutes ago —
            // so the only honest place left to report this is inside the
            // stream, in the same body shape a 4xx would have had.
            tracing::error!(error = %e, "generation failed mid-stream");
            let _ = emit(&state, &sse_data(&e.body_json()));
            let _ = emit(&state, &sse_done());
            let _ = lock(&state).finish();
        }
    }
}

/// A [`TokenSink`] that frames each token as an SSE chunk on the socket.
struct SseSink<W: Write> {
    state: Arc<Mutex<StreamState<W>>>,
    builder: ChunkBuilder,
    keepalive: Duration,
}

impl<W: Write> TokenSink for SseSink<W> {
    fn prefill(&mut self, done: usize, total: usize) -> Result<(), StreamError> {
        let mut state = lock(&self.state);
        if let Some(failed) = state.failure() {
            return Err(failed);
        }
        // Progress has no OpenAI chunk shape, so nothing is reported as
        // progress. What it is good for is a second, cheaper source of
        // keep-alives: for a 4K prompt these fire roughly every 45 s, which
        // is under the 300 s wall on its own but far too sparse to be the
        // only source — hence [`KeepAlive`] as well, and hence the shared
        // idle clock, so the two do not double up.
        if state.idle_for() >= self.keepalive {
            state.emit(KEEPALIVE)?;
            tracing::trace!(done, total, "keep-alive on prefill progress");
        }
        Ok(())
    }

    fn token(&mut self, text: &str) -> Result<(), StreamError> {
        // A token whose bytes are still an incomplete UTF-8 character decodes
        // to nothing; sending an empty delta would be a chunk that says
        // nothing, and the next token carries the character anyway.
        if text.is_empty() {
            return Ok(());
        }
        let event = sse_or_empty(&self.builder.content(text));
        lock(&self.state).emit(&event)
    }
}

/// Serialize one chunk, or the empty string.
///
/// [`sse_event`] can only fail if `serde_json` cannot serialize a struct of
/// `String`s and numbers, which it can. The failure branch is logged rather
/// than unwrapped so that stays true by construction, and an empty payload is
/// dropped by the framer rather than sent as a zero-length chunk.
fn sse_or_empty(chunk: &crate::response::ChatCompletionChunk) -> String {
    match sse_event(chunk) {
        Ok(event) => event,
        Err(e) => {
            tracing::error!(error = %e, "serializing a chunk failed; dropping it");
            String::new()
        }
    }
}

/// A thread that writes an SSE comment whenever the stream has gone quiet.
///
/// It exists because prefill progress is not a clock. `GenerateProgress`
/// fires once per prefill chunk — about eight events for a 4K prompt — so the
/// gap between two of them is tens of seconds and depends on the chunk width,
/// the cache state and the drive. A timer does not.
///
/// It shares the writer through the same mutex the generation thread uses, so
/// a comment can never interleave with half a chunk, and it reads the shared
/// idle clock so it stays quiet while content is flowing.
struct KeepAlive {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl KeepAlive {
    fn spawn<W: Write + Send + 'static>(
        state: Arc<Mutex<StreamState<W>>>,
        interval: Duration,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        // Quarter of the interval, so a comment is at worst 25% late, and
        // bounded either way so a tiny interval does not spin and a huge one
        // does not delay the join.
        let tick = (interval / 4).clamp(Duration::from_millis(5), Duration::from_secs(1));
        let handle = std::thread::Builder::new()
            .name("rvmp-keepalive".to_owned())
            .spawn(move || {
                while !flag.load(Ordering::Relaxed) {
                    std::thread::sleep(tick);
                    if flag.load(Ordering::Relaxed) {
                        return;
                    }
                    let mut state = lock(&state);
                    if state.failure().is_some() {
                        return;
                    }
                    if state.idle_for() >= interval && state.emit(KEEPALIVE).is_err() {
                        // The generation loop will see the same sticky
                        // failure at its next token and unwind there, where
                        // leaving is safe.
                        return;
                    }
                }
            });
        match handle {
            Ok(handle) => KeepAlive {
                stop,
                handle: Some(handle),
            },
            Err(e) => {
                // A stream without keep-alives still works for short prompts,
                // so this is degraded rather than fatal.
                tracing::warn!(error = %e, "no keep-alive thread; long prefills may time out");
                KeepAlive { stop, handle: None }
            }
        }
    }
}

impl Drop for KeepAlive {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            // Joined, not detached: the thread holds a writer clone, and the
            // terminating chunk must not race a comment.
            let _ = handle.join();
        }
    }
}

/// Lock the shared stream state, ignoring poisoning.
///
/// A poisoned mutex here means a panic while a write was in progress. The
/// state behind it is a writer and two flags — there is no invariant a panic
/// could have broken halfway — and refusing to write would turn one failed
/// request into a hung one.
fn lock<W: Write>(state: &Mutex<StreamState<W>>) -> MutexGuard<'_, StreamState<W>> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

fn emit<W: Write>(state: &Mutex<StreamState<W>>, payload: &str) -> Result<(), StreamError> {
    lock(state).emit(payload)
}

/// Read a request body, refusing anything over `limit`.
///
/// Checked twice: the declared `Content-Length` first, so an oversized body is
/// refused without reading it, and the actual byte count second, because
/// `Content-Length` is a claim by the client and a chunked request does not
/// make one at all.
fn read_body(request: &mut Request, limit: usize) -> Result<String, ServerError> {
    if request.body_length().is_some_and(|len| len > limit) {
        return Err(ServerError::BodyTooLarge { limit });
    }
    let mut body = Vec::new();
    // `limit + 1`, so "exactly at the limit" and "over it" are
    // distinguishable without reading the whole thing.
    let capped = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
    request
        .as_reader()
        .take(capped)
        .read_to_end(&mut body)
        .map_err(|e| ServerError::MalformedJson(format!("reading the request body: {e}")))?;
    if body.len() > limit {
        return Err(ServerError::BodyTooLarge { limit });
    }
    String::from_utf8(body)
        .map_err(|_| ServerError::MalformedJson("request body is not valid UTF-8".to_owned()))
}

/// A completion id: `chatcmpl-` plus something that does not repeat.
///
/// Not random, because a random source is a dependency and this is an
/// identifier, not a secret. The clock plus a counter cannot collide within a
/// process and cannot collide across two processes unless they start in the
/// same nanosecond.
fn completion_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{COMPLETION_ID_PREFIX}{:x}{:04x}", unix_now(), n & 0xffff)
}

/// Build a header from ASCII.
///
/// `Header::from_bytes` fails only on non-ASCII, and every call site here
/// passes a literal, so the failure branch is unreachable — reported rather
/// than unwrapped so it stays that way by construction.
fn header(name: &str, value: &str) -> Option<Header> {
    match Header::from_bytes(name.as_bytes(), value.as_bytes()) {
        Ok(header) => Some(header),
        Err(()) => {
            tracing::error!(name, value, "non-ASCII header; dropped");
            None
        }
    }
}

/// Answer with a buffered JSON body.
///
/// The buffered path is correct for everything that is not a stream: it is one
/// short body, written once, and `tiny_http`'s own encoder handles the
/// framing.
fn respond_json<T: serde::Serialize>(request: Request, status: u16, body: &T) {
    match serde_json::to_string(body) {
        Ok(json) => send(request, status, json, &[]),
        Err(e) => {
            let error = ServerError::Serialize(e);
            respond_error(request, &error);
        }
    }
}

/// Answer with an error body, its status and its headers.
fn respond_error(request: Request, error: &ServerError) {
    tracing::debug!(status = error.status(), error = %error, "refusing a request");
    send(request, error.status(), error.body_json(), error.headers());
}

fn send(request: Request, status: u16, body: String, extra: &[(&str, &str)]) {
    let mut response = Response::from_string(body).with_status_code(status);
    if let Some(header) = header("Content-Type", "application/json") {
        response = response.with_header(header);
    }
    for (name, value) in extra {
        if let Some(header) = header(name, value) {
            response = response.with_header(header);
        }
    }
    if let Err(e) = request.respond(response) {
        // `respond` maps the client-gone kinds to `Ok`, so anything reported
        // here is a real write failure.
        tracing::debug!(error = %e, "failed to send a response");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Completion;
    use crate::response::{FinishReason, Usage};
    use crate::wire::{SSE_HEAD, TERMINATOR};
    use std::sync::atomic::AtomicUsize;
    use std::time::Instant;

    /// A token source with no model in it: known tokens, at known intervals.
    ///
    /// This is what makes the streaming contract testable at all. A buffered
    /// stream and a live one produce byte-identical transcripts and differ
    /// only in *when* the bytes appear, so the assertion has to be about time,
    /// and an assertion about time needs a source whose timing is known.
    struct StubEngine {
        prefill_steps: usize,
        prefill_delay: Duration,
        tokens: Vec<String>,
        token_delay: Duration,
        /// Why the stub says it stopped. The tool-call lane overrides this, so
        /// what it overrides has to be settable.
        finish_reason: FinishReason,
        /// How many tokens the engine got as far as producing. The evidence
        /// that a hung-up client actually stops generation.
        produced: Arc<AtomicUsize>,
    }

    impl StubEngine {
        fn new(tokens: &[&str]) -> Self {
            StubEngine {
                prefill_steps: 0,
                prefill_delay: Duration::ZERO,
                tokens: tokens.iter().map(|t| (*t).to_owned()).collect(),
                token_delay: Duration::ZERO,
                finish_reason: FinishReason::Stop,
                produced: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl Engine for StubEngine {
        fn model_id(&self) -> &str {
            "stub-model"
        }

        fn created(&self) -> u64 {
            1_700_000_000
        }

        fn prepare(&mut self, request: &ChatCompletionRequest) -> Result<Plan, ServerError> {
            request.validate()?;
            Ok(Plan {
                prompt_tokens: 11,
                max_new: self.tokens.len(),
                cached_prefix: 0,
                prompt_ids: vec![1, 2, 3],
                params: ramvamp_core::generate::GenerateParams {
                    max_new: self.tokens.len(),
                    temperature: 0.7,
                    top_k: Some(20),
                    top_p: 0.8,
                    seed: 42,
                    greedy: false,
                },
            })
        }

        fn run(&mut self, plan: Plan, sink: &mut dyn TokenSink) -> Result<Completion, ServerError> {
            let mut text = String::new();
            for step in 1..=self.prefill_steps {
                std::thread::sleep(self.prefill_delay);
                if sink.prefill(step, self.prefill_steps).is_err() {
                    return Ok(aborted(text, plan.prompt_tokens, 0));
                }
            }
            for (index, token) in self.tokens.clone().iter().enumerate() {
                std::thread::sleep(self.token_delay);
                self.produced.store(index + 1, Ordering::Relaxed);
                text.push_str(token);
                if sink.token(token).is_err() {
                    return Ok(aborted(text, plan.prompt_tokens, index + 1));
                }
            }
            let generated = self.tokens.len();
            Ok(Completion {
                text,
                usage: Usage::new(plan.prompt_tokens as u32, generated as u32),
                finish_reason: self.finish_reason,
                aborted: false,
            })
        }
    }

    fn aborted(text: String, prompt: usize, generated: usize) -> Completion {
        Completion {
            text,
            usage: Usage::new(prompt as u32, generated as u32),
            finish_reason: FinishReason::Stop,
            aborted: true,
        }
    }

    /// One flushed group: when it went out, and what it was.
    type Flushed = Arc<Mutex<Vec<(Duration, Vec<u8>)>>>;

    /// A writer that timestamps every flush and can be told to die.
    #[derive(Clone)]
    struct Recorder {
        events: Flushed,
        pending: Arc<Mutex<Vec<u8>>>,
        flushes: Arc<AtomicUsize>,
        start: Instant,
        die_after_flushes: Option<usize>,
    }

    impl Recorder {
        fn new() -> Self {
            Recorder {
                events: Arc::new(Mutex::new(Vec::new())),
                pending: Arc::new(Mutex::new(Vec::new())),
                flushes: Arc::new(AtomicUsize::new(0)),
                start: Instant::now(),
                die_after_flushes: None,
            }
        }

        fn dying_after(mut self, flushes: usize) -> Self {
            self.die_after_flushes = Some(flushes);
            self
        }

        /// Every flushed group, as (elapsed, text).
        fn events(&self) -> Vec<(Duration, String)> {
            self.events
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .iter()
                .map(|(at, bytes)| (*at, String::from_utf8_lossy(bytes).into_owned()))
                .collect()
        }

        fn body(&self) -> String {
            self.events().into_iter().map(|(_, text)| text).collect()
        }
    }

    impl Write for Recorder {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.pending
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            let bytes =
                std::mem::take(&mut *self.pending.lock().unwrap_or_else(PoisonError::into_inner));
            let n = self.flushes.fetch_add(1, Ordering::Relaxed) + 1;
            if self.die_after_flushes.is_some_and(|limit| n > limit) {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            self.events
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((self.start.elapsed(), bytes));
            Ok(())
        }
    }

    fn request(stream: bool) -> ChatCompletionRequest {
        let body = format!(
            r#"{{"model":"stub-model","messages":[{{"role":"user","content":"hi"}}],"stream":{stream}}}"#
        );
        ChatCompletionRequest::from_json(&body).expect("valid body")
    }

    fn run_stream(engine: &mut StubEngine, recorder: Recorder, keepalive: Duration) {
        let request = request(true);
        let plan = engine.prepare(&request).expect("a plan");
        stream_into(
            engine,
            &request,
            plan,
            recorder,
            "chatcmpl-test",
            1_700_000_000,
            keepalive,
        );
    }

    /// The assertion a buffered implementation cannot pass: the content
    /// events are separated on the wire by the interval the source produced
    /// them at, rather than arriving together at the end.
    #[test]
    fn events_reach_the_client_at_the_sources_pace() {
        let mut engine = StubEngine::new(&["a", "b", "c", "d", "e"]);
        engine.token_delay = Duration::from_millis(60);
        let recorder = Recorder::new();
        run_stream(&mut engine, recorder.clone(), Duration::from_secs(60));

        let content: Vec<Duration> = recorder
            .events()
            .into_iter()
            .filter(|(_, text)| text.contains(r#""content":"#) && !text.contains(r#""role""#))
            .map(|(at, _)| at)
            .collect();
        assert_eq!(content.len(), 5, "{:?}", recorder.events());
        for pair in content.windows(2) {
            assert!(
                pair[1] - pair[0] >= Duration::from_millis(40),
                "events arrived in a burst: {content:?}"
            );
        }
        assert!(
            content[4] - content[0] >= Duration::from_millis(200),
            "{content:?}"
        );
    }

    /// The head is committed before the engine does any work, which is what
    /// keeps a six-minute prefill under a five-minute client timeout.
    #[test]
    fn the_head_is_written_before_the_engine_runs() {
        let mut engine = StubEngine::new(&["x"]);
        engine.prefill_steps = 1;
        engine.prefill_delay = Duration::from_millis(250);
        let recorder = Recorder::new();
        run_stream(&mut engine, recorder.clone(), Duration::from_secs(60));

        let events = recorder.events();
        assert_eq!(events[0].1, SSE_HEAD);
        assert!(
            events[0].0 < Duration::from_millis(100),
            "the head waited for the model: {:?}",
            events[0].0
        );
        // And the role chunk rides out with it.
        assert!(events[1].1.contains(r#""role":"assistant""#));
        assert!(events[1].0 < Duration::from_millis(100));
    }

    /// A prompt that takes minutes to prefill produces no content events, so
    /// something else has to keep the byte stream alive.
    #[test]
    fn a_slow_prefill_emits_keep_alive_comments() {
        let mut engine = StubEngine::new(&["done"]);
        engine.prefill_steps = 1;
        engine.prefill_delay = Duration::from_millis(400);
        let recorder = Recorder::new();
        run_stream(&mut engine, recorder.clone(), Duration::from_millis(50));

        let comments: Vec<Duration> = recorder
            .events()
            .into_iter()
            .filter(|(_, text)| text.contains(KEEPALIVE))
            .map(|(at, _)| at)
            .collect();
        assert!(
            comments.len() >= 3,
            "expected keep-alives during a 400ms prefill, got {comments:?}"
        );
        // They are comments, so a client discards them: the reply is intact.
        assert!(recorder.body().contains(r#""content":"done""#));
    }

    /// A client that hangs up surfaces as a write error, and the engine stops
    /// generating instead of running for minutes with nobody listening.
    #[test]
    fn a_dropped_client_surfaces_as_a_write_error_and_aborts_generation() {
        let mut engine = StubEngine::new(&["a", "b", "c", "d", "e", "f"]);
        let produced = Arc::clone(&engine.produced);
        // head, role, then one content event; the next flush is a broken pipe.
        let recorder = Recorder::new().dying_after(3);
        run_stream(&mut engine, recorder.clone(), Duration::from_secs(60));

        let got = produced.load(Ordering::Relaxed);
        assert!(
            (1..=2).contains(&got),
            "generation should have stopped at the hangup, produced {got}"
        );
        // Nothing is appended after the client is gone: no sentinel, no
        // terminator, because there is nowhere to put them.
        assert!(!recorder.body().contains("[DONE]"), "{}", recorder.body());
    }

    /// The whole wire sequence: head, framing, order, sentinel, terminator.
    #[test]
    fn the_sentinel_precedes_the_zero_length_chunk() {
        let mut engine = StubEngine::new(&["he", "llo"]);
        let recorder = Recorder::new();
        run_stream(&mut engine, recorder.clone(), Duration::from_secs(60));

        let body = recorder.body();
        let rest = body.strip_prefix(SSE_HEAD).expect("head first");
        assert!(rest.ends_with(TERMINATOR), "{rest:?}");
        let done = rest.find("data: [DONE]").expect("sentinel");
        let end = rest.rfind(TERMINATOR).expect("terminator");
        assert!(done < end, "the sentinel must precede the terminator");
    }

    /// Every chunk's declared length is the byte length of its payload, and
    /// the reassembled deltas are the reply. This is the framing check a
    /// client actually performs.
    #[test]
    fn the_chunk_framing_decodes_back_to_the_reply() {
        let mut engine = StubEngine::new(&["Hé", "llo", "→!"]);
        let recorder = Recorder::new();
        run_stream(&mut engine, recorder.clone(), Duration::from_secs(60));

        let body = recorder.body();
        let mut rest = body.strip_prefix(SSE_HEAD).expect("head first");
        let mut payloads = Vec::new();
        loop {
            let (head, tail) = rest.split_once("\r\n").expect("a chunk header");
            let len = usize::from_str_radix(head, 16).expect("hex length");
            if len == 0 {
                assert_eq!(tail, "\r\n", "the terminator ends the body");
                break;
            }
            assert!(tail.len() >= len + 2, "chunk shorter than its header says");
            payloads.push(tail[..len].to_owned());
            assert_eq!(&tail[len..len + 2], "\r\n", "chunk not CRLF-terminated");
            rest = &tail[len + 2..];
        }

        let text: String = payloads
            .iter()
            .filter_map(|event| event.strip_prefix("data: "))
            .filter_map(|json| serde_json::from_str::<serde_json::Value>(json.trim_end()).ok())
            .filter_map(|chunk| {
                chunk["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_owned)
            })
            .collect();
        assert_eq!(text, "Héllo→!");
    }

    /// `include_usage` puts the usage chunk between the finish chunk and the
    /// sentinel, and leaves it out entirely when it was not asked for.
    #[test]
    fn usage_is_emitted_only_when_asked_for_and_before_the_sentinel() {
        let mut engine = StubEngine::new(&["x"]);
        let recorder = Recorder::new();
        run_stream(&mut engine, recorder.clone(), Duration::from_secs(60));
        assert!(!recorder.body().contains(r#""usage""#));

        let asked = ChatCompletionRequest::from_json(
            r#"{"model":"stub-model","messages":[{"role":"user","content":"hi"}],
                "stream":true,"stream_options":{"include_usage":true}}"#,
        )
        .expect("valid body");
        let plan = engine.prepare(&asked).expect("a plan");
        let recorder = Recorder::new();
        stream_into(
            &mut engine,
            &asked,
            plan,
            recorder.clone(),
            "chatcmpl-test",
            1_700_000_000,
            Duration::from_secs(60),
        );
        let body = recorder.body();
        let usage = body.find(r#""usage""#).expect("a usage chunk");
        let finish = body.find(r#""finish_reason":"stop""#).expect("a finish");
        let done = body.find("[DONE]").expect("the sentinel");
        assert!(finish < usage && usage < done, "{body}");
        assert!(body.contains(r#""prompt_tokens":11"#), "{body}");
    }

    /// A runtime failure after the head is out cannot be a status code, so it
    /// is an error event inside the stream followed by a clean terminator.
    #[test]
    fn a_mid_stream_failure_becomes_an_error_event() {
        struct Failing;
        impl Engine for Failing {
            fn model_id(&self) -> &str {
                "stub-model"
            }
            fn created(&self) -> u64 {
                0
            }
            fn prepare(&mut self, _: &ChatCompletionRequest) -> Result<Plan, ServerError> {
                unreachable!("the test builds the plan itself")
            }
            fn run(&mut self, _: Plan, _: &mut dyn TokenSink) -> Result<Completion, ServerError> {
                Err(ServerError::Generate(
                    ramvamp_core::generate::GenerateError::EmptyPrompt,
                ))
            }
        }
        let mut stub = StubEngine::new(&[]);
        let request = request(true);
        let plan = stub.prepare(&request).expect("a plan");
        let recorder = Recorder::new();
        stream_into(
            &mut Failing,
            &request,
            plan,
            recorder.clone(),
            "chatcmpl-test",
            0,
            Duration::from_secs(60),
        );
        let body = recorder.body();
        assert!(body.starts_with(SSE_HEAD));
        assert!(body.contains(r#""code":"generate_error""#), "{body}");
        assert!(body.contains("[DONE]"));
        assert!(body.ends_with(TERMINATOR));
    }

    // ---- the buffered endpoints, over a real socket ----

    fn spawn_server() -> SocketAddr {
        spawn_server_with(&["ok"], FinishReason::Stop)
    }

    fn spawn_server_with(tokens: &[&str], finish_reason: FinishReason) -> SocketAddr {
        let tokens: Vec<String> = tokens.iter().map(|t| (*t).to_owned()).collect();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let borrowed: Vec<&str> = tokens.iter().map(String::as_str).collect();
            let mut engine = StubEngine::new(&borrowed);
            engine.finish_reason = finish_reason;
            let _ = serve_with(
                &mut engine,
                ServeConfig {
                    port: 0,
                    ..ServeConfig::default()
                },
                |address| {
                    let _ = tx.send(address);
                },
            );
        });
        rx.recv_timeout(Duration::from_secs(10))
            .expect("the server binds")
    }

    fn roundtrip(address: SocketAddr, raw: &str) -> String {
        let mut stream = std::net::TcpStream::connect(address).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("timeout");
        stream.write_all(raw.as_bytes()).expect("write");
        stream.flush().expect("flush");
        let mut out = Vec::new();
        let mut buffer = [0u8; 4096];
        loop {
            match stream.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(n) => out.extend_from_slice(&buffer[..n]),
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn get(address: SocketAddr, path: &str) -> String {
        roundtrip(
            address,
            &format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"),
        )
    }

    fn post(address: SocketAddr, path: &str, body: &str) -> String {
        roundtrip(
            address,
            &format!(
                "POST {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
                 Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            ),
        )
    }

    #[test]
    fn the_buffered_routes_answer_over_a_socket() {
        let address = spawn_server();
        // Loopback only, and the bound address proves it.
        assert!(address.ip().is_loopback(), "{address}");

        let health = get(address, PATH_HEALTH);
        assert!(health.starts_with("HTTP/1.1 200"), "{health}");
        assert!(health.contains(r#"{"status":"ok"}"#), "{health}");

        let models = get(address, PATH_MODELS);
        assert!(models.starts_with("HTTP/1.1 200"), "{models}");
        assert!(models.contains(r#""id":"stub-model""#), "{models}");
        assert!(models.contains(r#""object":"list""#), "{models}");

        // A query string is not part of the route.
        let versioned = get(address, "/v1/models?api-version=2024-02-01");
        assert!(versioned.contains(r#""id":"stub-model""#), "{versioned}");

        let missing = get(address, "/v1/embeddings");
        assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");
        assert!(
            missing.contains(r#""type":"invalid_request_error""#),
            "{missing}"
        );
        assert!(missing.contains(r#""code":"not_found""#), "{missing}");

        // Right path, wrong method: still a 404 with a parseable body, never
        // an empty one.
        let wrong = get(address, PATH_CHAT_COMPLETIONS);
        assert!(wrong.starts_with("HTTP/1.1 404"), "{wrong}");
        assert!(wrong.contains(r#""error""#), "{wrong}");
    }

    #[test]
    fn a_non_streaming_completion_comes_back_as_one_json_body() {
        let address = spawn_server();
        let response = post(
            address,
            PATH_CHAT_COMPLETIONS,
            r#"{"model":"stub-model","messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(
            response.contains(r#""object":"chat.completion""#),
            "{response}"
        );
        assert!(response.contains(r#""content":"ok""#), "{response}");
        assert!(response.contains(r#""finish_reason":"stop""#), "{response}");
    }

    /// What the model actually writes when it wants a tool: prose, then the
    /// block the template asked for.
    const TOOL_CALL_REPLY: &str = "Let me look.\n<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"a.txt\"}}\n</tool_call>";

    /// The JSON body of an HTTP response, headers stripped.
    fn body_json(response: &str) -> serde_json::Value {
        let body = response
            .split_once("\r\n\r\n")
            .map_or(response, |(_, body)| body);
        serde_json::from_str(body).unwrap_or_else(|e| panic!("a JSON body: {e}\n{response}"))
    }

    /// The end of this wave: a client can act on the reply. The text the model
    /// wrote is gone from `content` and present as structure, with an id the
    /// client can put on its `tool` message.
    #[test]
    fn a_tool_call_reply_comes_back_as_structured_calls() {
        let address = spawn_server_with(&[TOOL_CALL_REPLY], FinishReason::Stop);
        let response = post(
            address,
            PATH_CHAT_COMPLETIONS,
            r#"{"model":"stub-model","messages":[{"role":"user","content":"read a.txt"}],
                "tools":[{"type":"function","function":{"name":"read","parameters":{}}}]}"#,
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let json = body_json(&response);
        let choice = &json["choices"][0];

        // The stop reason the engine reported is overridden: the model stopped
        // because it wants a tool run.
        assert_eq!(choice["finish_reason"], "tool_calls");

        let calls = choice["message"]["tool_calls"]
            .as_array()
            .unwrap_or_else(|| panic!("a tool_calls array: {json}"));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "read");
        // `arguments` is a JSON *string*, byte for byte what the model wrote.
        assert_eq!(calls[0]["function"]["arguments"], r#"{"path": "a.txt"}"#);
        let call_id = calls[0]["id"].as_str().expect("an id");
        assert!(call_id.starts_with("call_"), "{call_id}");
        assert!(call_id.ends_with("_0"), "{call_id}");
        assert!(
            call_id.contains(
                json["id"]
                    .as_str()
                    .and_then(|id| id.strip_prefix("chatcmpl-"))
                    .expect("a completion id")
            ),
            "the call id is derived from the completion id: {call_id}"
        );

        // And the prose survives without the markup.
        assert_eq!(choice["message"]["content"], "Let me look.");
        assert!(
            !response.contains("<tool_call>"),
            "the raw block is still in the body: {response}"
        );
    }

    /// The same reply, from a request that declared no tools, stays text.
    ///
    /// A conversation *about* tool calling is the case this protects: asked
    /// what a call looks like, the model writes one, and extracting it would
    /// delete the answer from `content` and hand back a call the caller never
    /// enabled, on a request that offered no tools to call.
    #[test]
    fn a_block_in_a_reply_to_a_request_without_tools_stays_text() {
        let address = spawn_server_with(&[TOOL_CALL_REPLY], FinishReason::Stop);
        let response = post(
            address,
            PATH_CHAT_COMPLETIONS,
            r#"{"model":"stub-model","messages":[{"role":"user","content":"show me a tool call"}]}"#,
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let json = body_json(&response);
        let choice = &json["choices"][0];

        assert_eq!(choice["finish_reason"], "stop");
        assert!(
            choice["message"]["tool_calls"].is_null(),
            "a request without tools got tool_calls back: {json}"
        );
        // Byte for byte, markers and all.
        assert_eq!(
            choice["message"]["content"].as_str().expect("content"),
            TOOL_CALL_REPLY
        );
    }

    /// A reply with no block is untouched, and the key stays absent rather
    /// than becoming an empty array a client would read as "tools wanted".
    #[test]
    fn a_plain_reply_carries_no_tool_calls_key() {
        let address = spawn_server();
        let response = post(
            address,
            PATH_CHAT_COMPLETIONS,
            r#"{"model":"stub-model","messages":[{"role":"user","content":"hi"}]}"#,
        );
        let json = body_json(&response);
        assert_eq!(json["choices"][0]["message"]["content"], "ok");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert!(
            json["choices"][0]["message"].get("tool_calls").is_none(),
            "{json}"
        );
    }

    /// A block cut off by the token budget yields no call, so the reason stays
    /// `length` — the client's cue to raise `max_tokens`, not to run something
    /// that was never fully asked for.
    #[test]
    fn a_block_truncated_by_the_token_budget_keeps_the_length_reason() {
        let address = spawn_server_with(
            &["<tool_call>\n{\"name\": \"read\", \"argum"],
            FinishReason::Length,
        );
        let response = post(
            address,
            PATH_CHAT_COMPLETIONS,
            r#"{"model":"stub-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8}"#,
        );
        let json = body_json(&response);
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert!(
            json["choices"][0]["message"].get("tool_calls").is_none(),
            "{json}"
        );
        assert_eq!(
            json["choices"][0]["message"]["content"], "<tool_call>\n{\"name\": \"read\", \"argum",
            "half a block is text, and text is not dropped"
        );
    }

    /// Wave 4's boundary, asserted rather than assumed: the streaming path
    /// still sends the model's text exactly as it comes, blocks included.
    #[test]
    fn the_streaming_path_still_emits_the_raw_block() {
        let mut engine = StubEngine::new(&["<tool_call>", "{\"name\": \"read\"}", "</tool_call>"]);
        let recorder = Recorder::new();
        run_stream(&mut engine, recorder.clone(), Duration::from_secs(60));

        let body = recorder.body();
        assert!(body.contains(r#""content":"<tool_call>""#), "{body}");
        assert!(body.contains(r#""content":"</tool_call>""#), "{body}");
        assert!(body.contains(r#""finish_reason":"stop""#), "{body}");
        assert!(
            !body.contains(r#""tool_calls""#),
            "streaming tool calls are wave 4: {body}"
        );
    }

    /// The array-form `content` every SDK sends, end to end. A server that
    /// models `content` as a `String` answers a question it never received.
    #[test]
    fn the_array_form_content_reaches_the_engine() {
        let address = spawn_server();
        let response = post(
            address,
            PATH_CHAT_COMPLETIONS,
            r#"{"model":"stub-model","messages":[
                {"role":"user","content":[{"type":"text","text":"hi"}]}]}"#,
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains(r#""content":"ok""#), "{response}");
    }

    #[test]
    fn a_malformed_body_is_a_400_with_the_openai_shape() {
        let address = spawn_server();
        let response = post(address, PATH_CHAT_COMPLETIONS, "{not json");
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
        assert!(response.contains(r#""code":"invalid_json""#), "{response}");
        assert!(!response.contains(r#""message":null"#), "{response}");
    }

    /// Every byte off the socket is untrusted, and the body is read into
    /// memory to be parsed. Inside a 3 GB cgroup an uncapped read is an OOM.
    #[test]
    fn an_oversized_body_is_refused_by_its_declared_length() {
        const BODY: &str = "0123456789abcdef0123456789abcdef";
        let mut request = tiny_http::TestRequest::new()
            .with_method(Method::Post)
            .with_path(PATH_CHAT_COMPLETIONS)
            .with_body(BODY)
            .into();
        let error = read_body(&mut request, 16).expect_err("over the cap");
        assert!(matches!(error, ServerError::BodyTooLarge { limit: 16 }));
        assert_eq!(error.status(), 413);
    }

    #[test]
    fn a_body_at_the_cap_is_accepted() {
        const BODY: &str = "0123456789abcdef";
        let mut request = tiny_http::TestRequest::new()
            .with_method(Method::Post)
            .with_path(PATH_CHAT_COMPLETIONS)
            .with_body(BODY)
            .into();
        assert_eq!(
            read_body(&mut request, BODY.len()).expect("at the cap"),
            BODY
        );
    }

    #[test]
    fn completion_ids_do_not_repeat() {
        let ids: std::collections::HashSet<String> = (0..64).map(|_| completion_id()).collect();
        assert_eq!(ids.len(), 64);
        assert!(ids.iter().all(|id| id.starts_with("chatcmpl-")));
    }
}
