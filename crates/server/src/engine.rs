//! The thing that turns a request into tokens, and the seam that lets the
//! transport be tested without one.
//!
//! # Why there is a trait here
//!
//! Streaming behaviour is not verifiable by reading code. A response that is
//! buffered to completion and one that is delivered live produce *byte
//! identical* transcripts; they differ only in when the bytes arrive. The only
//! way to assert the difference is to drive the transport from a source that
//! emits known tokens at known intervals — which means the transport cannot be
//! wired straight to a 30B model. Hence [`Engine`]: `ModelEngine` is one
//! implementation, and the tests use another that sleeps.
//!
//! # The two-step shape
//!
//! [`Engine::prepare`] does everything that can fail *before* a byte of
//! response is committed — parsing the conversation, encoding it, checking it
//! against the context cap — and [`Engine::run`] does the part that cannot be
//! taken back. That split is what lets a context overflow be a clean HTTP 400
//! while a mid-generation failure is reported inside an already-200 stream.
//!
//! # Prefix caching, and the trap in it
//!
//! A Chat Completions client is stateless: every request carries the whole
//! conversation, so a naive server re-prefills all of it every turn, which at
//! this runtime's speeds is minutes of work to reach a state it already had.
//! The fix is to keep the KV cache across requests and re-prefill only the
//! part that changed — [`ForwardState::truncate`] plus a longest-common-prefix
//! match.
//!
//! The trap is *what to match against*, and it fails silently. The client
//! sends previous assistant turns back as **text**, and re-encoding that text
//! does not reproduce the ids the model generated: the `\n` that closes
//! `<|im_start|>assistant\n` and the first characters of the reply are
//! candidates for the same BPE merge, so a generation-prompt render is not
//! always a token prefix of the finished turn (measured; the CLI has a test
//! named after it). Comparing against a re-render therefore produces a *wrong*
//! cache hit — the positions are reused, the ids at them are not the ids the
//! cache holds, `generate_from` cannot detect it because the arithmetic still
//! adds up, and the model answers fluently from a context that never existed.
//!
//! So the comparison is against [`ModelEngine::history`], which holds the ids
//! that were actually fed: prompt renders as encoded, and assistant turns as
//! [`GenerateStats::generated_ids`], never as a re-encoding of the reply.
//! Where the client's re-render diverges from what the model said, the common
//! prefix simply ends there and the suffix is re-prefilled. Correct by
//! construction rather than by luck.

use std::cell::RefCell;
use std::io::ErrorKind;
use std::sync::Once;
use std::time::{SystemTime, UNIX_EPOCH};

use ramvamp_core::generate::{
    GenerateParams, GenerateProgress, StopReason, generate_from_with_progress,
};
use ramvamp_core::model::{ForwardState, Model};
use ramvamp_core::tokenizer::RvmpTokenizer;
use thiserror::Error;

use crate::error::ServerError;
use crate::prompt;
use crate::request::ChatCompletionRequest;
use crate::response::{FinishReason, Usage};

/// A failure writing to whoever is consuming the stream.
///
/// Deliberately `Copy`: the transport shares one of these between the thread
/// running the model and the keep-alive thread, and the `io::Error` payload
/// carries nothing worth the `Arc` — the kind is the whole story, and the
/// response to every kind is the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum StreamError {
    /// The client hung up. The ordinary way a stream ends early, and not an
    /// error worth logging above `debug`.
    #[error("the client closed the connection")]
    Disconnected,
    /// Any other write failure.
    #[error("writing to the client failed: {0:?}")]
    Io(ErrorKind),
}

impl From<std::io::Error> for StreamError {
    /// Classify a write failure. The four kinds `tiny_http` itself treats as
    /// "client went away" are the four treated as a disconnect here — the
    /// difference being that here it is reported rather than swallowed.
    fn from(error: std::io::Error) -> Self {
        match error.kind() {
            ErrorKind::BrokenPipe
            | ErrorKind::ConnectionReset
            | ErrorKind::ConnectionAborted
            | ErrorKind::ConnectionRefused => StreamError::Disconnected,
            other => StreamError::Io(other),
        }
    }
}

/// Where a running generation reports to.
///
/// Both methods return a `Result` because the usual implementation is a
/// socket, and a socket that has gone away is the signal to stop generating —
/// there is nobody left to generate for, and the session is worth minutes.
pub trait TokenSink {
    /// Prompt positions committed so far, out of the whole prompt.
    ///
    /// Counted against the *whole* prompt including the part served from
    /// cache, so a cache hit shows up as progress already made rather than as
    /// a prefill that starts at zero and finishes suspiciously fast.
    fn prefill(&mut self, positions_done: usize, positions_total: usize)
    -> Result<(), StreamError>;

    /// One token's worth of newly-decoded text.
    fn token(&mut self, text: &str) -> Result<(), StreamError>;
}

/// A sink that discards everything, for the non-streaming path.
///
/// The reply text comes back in [`Completion::text`] either way, so the
/// buffered endpoint needs nothing from the sink but a place to not send it.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl TokenSink for NullSink {
    fn prefill(&mut self, _: usize, _: usize) -> Result<(), StreamError> {
        Ok(())
    }

    fn token(&mut self, _: &str) -> Result<(), StreamError> {
        Ok(())
    }
}

/// One accepted request, ready to run.
///
/// Everything that could have been refused already has been, which is the
/// point: holding a `Plan` means the response head can be committed.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// The whole prompt, cached prefix included. This is the number the
    /// client is billed in `usage.prompt_tokens`, not the suffix actually
    /// prefilled — a cache hit is this server's business, not a discount the
    /// client can observe and come to depend on.
    pub prompt_tokens: usize,
    /// Reply cap for this turn, after clamping to the context.
    pub max_new: usize,
    /// Leading positions the KV cache already holds and this turn reuses.
    pub cached_prefix: usize,
    /// The full sanitized prompt id sequence.
    pub prompt_ids: Vec<u32>,
    /// Sampling dials, merged from the request over the checkpoint defaults.
    pub params: GenerateParams,
}

impl Plan {
    /// How many positions this turn actually has to prefill.
    pub fn to_prefill(&self) -> usize {
        self.prompt_ids.len().saturating_sub(self.cached_prefix)
    }
}

/// What a finished (or abandoned) generation produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    /// The reply text, concatenated from what the sink was shown.
    pub text: String,
    /// Token accounting for the whole request.
    pub usage: Usage,
    /// Why it ended.
    pub finish_reason: FinishReason,
    /// The sink asked to stop — in practice, the client hung up. Nothing more
    /// should be written; there is nowhere to write it.
    pub aborted: bool,
}

/// The generator behind the HTTP surface.
pub trait Engine {
    /// The model id `/v1/models` advertises and every response echoes.
    fn model_id(&self) -> &str;

    /// Unix seconds to report as the model's creation time.
    fn created(&self) -> u64;

    /// Everything that can be refused before the response is committed.
    ///
    /// # Errors
    ///
    /// Any [`ServerError`] whose status is a 4xx: this is where a bad request
    /// becomes an HTTP status rather than a broken stream.
    fn prepare(&mut self, request: &ChatCompletionRequest) -> Result<Plan, ServerError>;

    /// Run the plan, reporting through `sink`.
    ///
    /// A sink failure is not an error here: it produces a [`Completion`] with
    /// [`Completion::aborted`] set, because "the client left" is an ordinary
    /// outcome and the engine must be ready for the next request either way.
    ///
    /// # Errors
    ///
    /// [`ServerError::Generate`] or [`ServerError::Forward`] when the runtime
    /// itself fails. The engine is left able to serve the next request.
    fn run(&mut self, plan: Plan, sink: &mut dyn TokenSink) -> Result<Completion, ServerError>;
}

/// Payload of the deliberate unwind that stops a generation whose client has
/// gone away.
///
/// The same device, for the same reason, as `ChatAbort` in the CLI:
/// `generate_from`'s `on_token` returns `()`, so it has no way to report that
/// it wants out, and an unwind is the only exit from a decode loop in
/// progress. It is raised at a token boundary — between two forward passes,
/// with no expert read in flight and the KV cache whole — which is what makes
/// the state reusable afterwards.
#[derive(Debug)]
pub struct ServerAbort;

/// Install, once per process, a panic hook that prints nothing for
/// [`ServerAbort`] and forwards every other panic to the hook already in
/// place.
///
/// Without it a client pressing Ctrl-C dumps a "thread panicked" backtrace
/// into the server's log for what is an entirely routine event. Real panics,
/// on any thread, still print exactly as before.
pub fn hush_stream_aborts() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if info.payload().downcast_ref::<ServerAbort>().is_none() {
                previous(info);
            }
        }));
    });
}

/// Length of the longest common prefix of two id sequences.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Unix seconds, or 0 if the clock is before the epoch.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// How a [`ModelEngine`] was configured.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// The id to advertise and echo.
    pub model_id: String,
    /// The context cap the [`ForwardState`] was built with.
    pub context_limit: usize,
    /// Reply cap for a request that does not send one.
    pub default_max_new: usize,
    /// Sampling dials to start from; a request overrides field by field.
    pub params: GenerateParams,
}

/// The real engine: a model, its state and its tokenizer, on one thread.
///
/// [`ForwardState`] is `Send` but not `Sync` — it owns the expert slot pool,
/// the io_uring ring and the pinned compute pool — so it is moved onto the
/// thread that serves requests and never shared. That is not a limitation
/// being worked around: concurrency here is one, structurally, because a
/// second decode loop would halve the token rate of both and there is exactly
/// one KV cache.
pub struct ModelEngine {
    model: Model,
    state: ForwardState,
    tokenizer: RvmpTokenizer,
    config: EngineConfig,
    created: u64,
    /// Every id the cached conversation consists of: prompt renders as
    /// encoded, assistant turns as the ids the model actually generated.
    ///
    /// Runs *ahead* of the KV cache by design — the last id sampled is
    /// emitted and never fed back, because there is nothing left to predict
    /// from it — so [`ForwardState::seq_len`] is the authority on how much of
    /// it the cache holds, never a count kept alongside. See the module docs
    /// for why this is not a re-render of the transcript.
    history: Vec<u32>,
    /// Requests served, used to vary the seed when the client does not pin
    /// one. Without it every repeat of a question gets a byte-identical
    /// answer, which reads as a broken model rather than as determinism.
    turn: u64,
}

impl ModelEngine {
    /// Build an engine over an already-loaded model and state.
    ///
    /// The loading is the caller's, deliberately: `state` carries the cache
    /// budget, thread pinning and prefill dials, and those belong to the
    /// command line that set them rather than to this crate.
    pub fn new(
        model: Model,
        state: ForwardState,
        tokenizer: RvmpTokenizer,
        config: EngineConfig,
    ) -> Self {
        ModelEngine {
            model,
            state,
            tokenizer,
            config,
            created: unix_now(),
            history: Vec::new(),
            turn: 0,
        }
    }

    /// The ids the cached conversation consists of.
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Merge a request's sampling fields over the configured defaults.
    fn sampling(&self, request: &ChatCompletionRequest, max_new: usize) -> GenerateParams {
        let mut params = self.config.params;
        params.max_new = max_new;
        if let Some(temperature) = request.temperature {
            // <= 0 is greedy, which `generate` already implements; passing it
            // through rather than translating keeps one rule in one place.
            params.temperature = temperature;
        }
        if let Some(top_p) = request.top_p {
            params.top_p = top_p;
        }
        if let Some(top_k) = request.top_k {
            params.top_k = Some(top_k);
        }
        // An unpinned seed advances per request; a pinned one is honoured
        // exactly, which is what `seed` means to a client.
        params.seed = request
            .seed
            .unwrap_or_else(|| self.config.params.seed.wrapping_add(self.turn));
        params
    }

    /// Recover from a runtime failure that may have left the cache ragged.
    ///
    /// Cheap — every allocation survives — and the alternative is an engine
    /// that answers `PositionMismatch` to every request for the rest of its
    /// life.
    fn recover(&mut self) {
        self.state.reset();
        self.history.clear();
    }
}

impl Engine for ModelEngine {
    fn model_id(&self) -> &str {
        &self.config.model_id
    }

    fn created(&self) -> u64 {
        self.created
    }

    fn prepare(&mut self, request: &ChatCompletionRequest) -> Result<Plan, ServerError> {
        request.validate()?;
        // Sanitized here, before anything is compared: the ids that get
        // prefilled are the sanitized ones, so a prefix match against the
        // unsanitized stream would be a match against a sequence this server
        // never feeds the model.
        let built = prompt::build(&self.tokenizer, request)?;
        if built.token_ids.is_empty() {
            return Err(ServerError::EmptyMessages);
        }

        let max_new = request
            .max_new_tokens()
            .map_or(self.config.default_max_new, |n| n as usize)
            .max(1);
        prompt::check_context(built.token_ids.len(), max_new, self.config.context_limit)?;

        // The cache says where it is; `history` says what is in it. The
        // smaller of the two bounds what can be reused, because `history`
        // runs one id ahead of the cache after every reply.
        let fed = self.state.seq_len()?;
        let shared = common_prefix(&self.history, &built.token_ids);
        // At least one token has to be prefilled: `generate_with_stops`
        // refuses an empty prompt with `EmptyPrompt`, and an identical resend
        // (the incoming ids being a prefix of the history) would otherwise
        // produce exactly that. Rewinding one position is the cheapest legal
        // suffix.
        let cached_prefix = shared.min(fed).min(built.token_ids.len() - 1);

        tracing::debug!(
            prompt_tokens = built.token_ids.len(),
            cached = cached_prefix,
            prefill = built.token_ids.len() - cached_prefix,
            max_new,
            "prepared a request"
        );

        Ok(Plan {
            prompt_tokens: built.token_ids.len(),
            max_new,
            cached_prefix,
            params: self.sampling(request, max_new),
            prompt_ids: built.token_ids,
        })
    }

    fn run(&mut self, plan: Plan, sink: &mut dyn TokenSink) -> Result<Completion, ServerError> {
        // A truncate that does nothing still fails loudly downstream:
        // `forward_token` re-validates `position == seq_len()` and raises
        // `PositionMismatch`. So this is not papered over on any path.
        if let Err(e) = self.state.truncate(plan.cached_prefix) {
            self.recover();
            return Err(ServerError::Forward(e));
        }
        self.turn = self.turn.wrapping_add(1);

        let prompt_total = plan.prompt_ids.len();
        let cached_prefix = plan.cached_prefix;
        let new_ids = &plan.prompt_ids[cached_prefix..];

        // One cell for everything both callbacks touch. They are never live
        // at the same instant — `generate` calls one or the other, never both
        // — but the borrow checker cannot see that through two closures held
        // simultaneously, and a `RefCell` is the honest way to say so.
        let live = RefCell::new(Live {
            sink,
            reply: String::new(),
            spoken: Vec::new(),
            abort: None,
        });

        let outcome = {
            let mut on_progress = |event: GenerateProgress| {
                let GenerateProgress::PrefillChunk { positions_done, .. } = event else {
                    return;
                };
                let mut live = live.borrow_mut();
                if live.abort.is_some() {
                    return;
                }
                // Reported against the whole prompt, cached prefix included.
                let done = cached_prefix + positions_done;
                if let Err(e) = live.sink.prefill(done, prompt_total) {
                    // Recorded, *not* unwound. A prefill chunk boundary is
                    // inside `prefill_sweep`, which still owns a
                    // `PrefillSession` over the streamer's staging arena and
                    // has not called `finish()` on it; unwinding through that
                    // abandons the arena rather than releasing it. The decode
                    // loop's token boundary is the one place the runtime
                    // documents as safe to leave, so the abort waits for it.
                    // The cost is finishing a prefill nobody will read, which
                    // is bounded by the prompt.
                    live.abort = Some(e);
                }
            };

            let mut on_token = |id: u32, text: &str| {
                let mut live = live.borrow_mut();
                live.reply.push_str(text);
                live.spoken.push(id);
                if live.abort.is_none()
                    && let Err(e) = live.sink.token(text)
                {
                    live.abort = Some(e);
                }
                if live.abort.is_some() {
                    // Release the borrow before unwinding: the guard would
                    // drop anyway, but leaving it to the unwinder makes the
                    // catch site's `into_inner` depend on drop order.
                    drop(live);
                    std::panic::panic_any(ServerAbort);
                }
            };

            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                generate_from_with_progress(
                    &self.model,
                    &mut self.state,
                    &self.tokenizer,
                    new_ids,
                    cached_prefix,
                    &plan.params,
                    Some(&mut on_progress),
                    &mut on_token,
                )
            }))
        };

        let live = live.into_inner();
        match outcome {
            Ok(Ok(stats)) => {
                // The cache now holds this prompt; the history is it plus what
                // the model said, in the ids it actually produced.
                self.history.clear();
                self.history.extend_from_slice(&plan.prompt_ids);
                self.history.extend_from_slice(&stats.generated_ids);
                let finish_reason = match stats.stop {
                    StopReason::StopToken(_) => FinishReason::Stop,
                    StopReason::MaxNew => FinishReason::Length,
                };
                tracing::info!(
                    prompt_tokens = plan.prompt_tokens,
                    prefilled = new_ids.len(),
                    generated = stats.generated,
                    prefill_s = stats.prefill.as_secs_f64(),
                    decode_s = stats.decode.as_secs_f64(),
                    "served a completion"
                );
                Ok(Completion {
                    text: live.reply,
                    usage: Usage::new(
                        u32::try_from(plan.prompt_tokens).unwrap_or(u32::MAX),
                        u32::try_from(stats.generated).unwrap_or(u32::MAX),
                    ),
                    finish_reason,
                    aborted: false,
                })
            }
            Ok(Err(e)) => {
                // Mid-pass failure: the cache may be ragged, and a ragged one
                // reports `PositionMismatch` forever.
                self.recover();
                Err(ServerError::Generate(e))
            }
            Err(payload) => {
                if payload.downcast_ref::<ServerAbort>().is_none() {
                    // Somebody else's panic. Re-raise it untouched.
                    std::panic::resume_unwind(payload);
                }
                // The partial reply is in the cache up to its second-to-last
                // id, so it has to be in the history too — otherwise the next
                // turn would prefill from a position the model reached by a
                // route the history no longer records. `spoken` is exactly
                // what `generated_ids` would have held: the unwind leaves
                // `on_token` before the id is fed, and the trailing flush
                // call that would have repeated an id never happens.
                self.history.clear();
                self.history.extend_from_slice(&plan.prompt_ids);
                self.history.extend_from_slice(&live.spoken);
                tracing::debug!(
                    generated = live.spoken.len(),
                    reason = ?live.abort,
                    "client left; abandoned the reply"
                );
                Ok(Completion {
                    usage: Usage::new(
                        u32::try_from(plan.prompt_tokens).unwrap_or(u32::MAX),
                        u32::try_from(live.spoken.len()).unwrap_or(u32::MAX),
                    ),
                    text: live.reply,
                    finish_reason: FinishReason::Stop,
                    aborted: true,
                })
            }
        }
    }
}

/// The mutable half of a run, shared by the two callbacks.
struct Live<'a> {
    sink: &'a mut dyn TokenSink,
    reply: String,
    spoken: Vec<u32>,
    abort: Option<StreamError>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_common_prefix_stops_at_the_first_divergent_id() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 9, 3]), 2);
        assert_eq!(common_prefix(&[1, 2, 3], &[]), 0);
        assert_eq!(common_prefix(&[], &[1]), 0);
        assert_eq!(common_prefix(&[1, 2], &[1, 2, 3, 4]), 2);
    }

    /// The identical-resend edge: the incoming ids are a prefix of the
    /// history, so the naive suffix is empty and `generate_with_stops` answers
    /// `EmptyPrompt`. Rewinding one position is the fix, and it costs one
    /// forward pass.
    #[test]
    fn an_identical_resend_still_leaves_one_token_to_prefill() {
        let history = [1u32, 2, 3, 4, 5];
        let incoming = [1u32, 2, 3];
        let fed = history.len();
        let shared = common_prefix(&history, &incoming);
        assert_eq!(shared, incoming.len());
        let cached = shared.min(fed).min(incoming.len() - 1);
        assert_eq!(cached, 2);
        assert_eq!(incoming.len() - cached, 1);
    }

    /// The cache, not the history, bounds what can be reused: the history
    /// runs one id ahead after every reply, because the last token sampled is
    /// emitted and never fed back.
    #[test]
    fn the_cache_bounds_the_reusable_prefix_not_the_history() {
        let history = [1u32, 2, 3, 4];
        let incoming = [1u32, 2, 3, 4, 5, 6];
        let fed = history.len() - 1;
        let shared = common_prefix(&history, &incoming);
        assert_eq!(shared, 4);
        assert_eq!(shared.min(fed).min(incoming.len() - 1), 3);
    }

    /// The whole reason the history is not a re-render. A client sends the
    /// assistant turn back as text; re-encoding it produces different ids
    /// than the model generated, so the prefix ends at the divergence and the
    /// rest is re-prefilled — instead of silently reusing positions whose ids
    /// the cache does not hold.
    #[test]
    fn a_re_encoded_assistant_turn_diverges_and_shortens_the_prefix() {
        // Ids up to and including the generation prompt, then the reply.
        let generated_first = 5000u32;
        let history = [
            10,
            11,
            12,
            /* <|im_start|>assistant\n */ 13,
            generated_first,
            77,
        ];
        // The client's re-render merges the closing newline with the reply's
        // first characters, so position 3 differs.
        let re_encoded = [10, 11, 12, 4242, 77, 30, 31];
        assert_eq!(common_prefix(&history, &re_encoded), 3);
        // Position 3 onwards is re-prefilled rather than wrongly reused.
        assert_ne!(history[3], re_encoded[3]);
    }

    /// The abort payload round-trips through `catch_unwind` and is
    /// distinguishable from a real panic, which is what keeps the engine
    /// alive across a client hangup.
    #[test]
    fn a_stream_abort_unwinds_and_is_caught() {
        hush_stream_aborts();
        let outcome = std::panic::catch_unwind(|| std::panic::panic_any(ServerAbort));
        let payload = outcome.expect_err("the abort unwinds");
        assert!(payload.downcast_ref::<ServerAbort>().is_some());
    }

    #[test]
    fn every_client_gone_io_kind_classifies_as_a_disconnect() {
        for kind in [
            ErrorKind::BrokenPipe,
            ErrorKind::ConnectionReset,
            ErrorKind::ConnectionAborted,
            ErrorKind::ConnectionRefused,
        ] {
            assert_eq!(
                StreamError::from(std::io::Error::from(kind)),
                StreamError::Disconnected
            );
        }
        assert_eq!(
            StreamError::from(std::io::Error::from(ErrorKind::WouldBlock)),
            StreamError::Io(ErrorKind::WouldBlock)
        );
    }

    #[test]
    fn a_plan_knows_how_much_it_has_to_prefill() {
        let plan = Plan {
            prompt_tokens: 6,
            max_new: 8,
            cached_prefix: 4,
            prompt_ids: vec![1, 2, 3, 4, 5, 6],
            params: GenerateParams {
                max_new: 8,
                temperature: 0.7,
                top_k: Some(20),
                top_p: 0.8,
                seed: 42,
                greedy: false,
            },
        };
        assert_eq!(plan.to_prefill(), 2);
    }
}
