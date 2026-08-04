//! Generation orchestration: prefill, token-by-token decode, sampling.
//!
//! **Prefill today is decode**: [`forward_token`] in a loop over the prompt,
//! one token at a time, through the same [`ForwardState`] and therefore the
//! same expert cache, with logits requested only for the last prompt token.
//! Decode then repeats that loop one generated token at a time. Sampling
//! supports greedy, temperature, top-k, and top-p (repetition penalty is
//! not implemented yet); greedy decode must be deterministic for
//! validation against reference implementations.
//!
//! The layer-major chunked prefill `docs/architecture.md` specifies — a
//! bounded chunk of positions swept per layer, so one fetched expert serves
//! many rows and the cache is bypassed entirely — is phase 6, and is not
//! this code. Nothing here bypasses or bounds the cache differently for
//! prompt tokens, and the streaming counters reflect that: a pure-prefill
//! run reports cache hits, because prompt positions reuse each other's
//! experts exactly the way decode positions do.
//!
//! # Sampling
//!
//! Greedy is a pure argmax over the raw logits (first index wins ties) and
//! is what the llama.cpp validation gates use. The stochastic path applies
//! filters in llama.cpp's default sampler-chain order: **top-k, then
//! top-p, then temperature**, then one draw from the final softmax. Top-p
//! measures probability mass on the *unscaled* logits (temperature applies
//! after the truncations, as in llama.cpp's chain). Randomness is an
//! inline seeded xorshift64* PRNG — deterministic for a fixed seed, no
//! external dependency.
//!
//! # Streaming detokenization
//!
//! Decoding each token id alone is wrong for byte-level BPE (one Unicode
//! character can span tokens), so [`generate`] decodes a bounded trailing
//! window of the generated ids and emits the new suffix, withholding any
//! trailing U+FFFD replacement characters until the bytes that complete
//! them arrive (they are flushed verbatim at end of generation if the model
//! stops mid-character).

use std::time::{Duration, Instant};

use thiserror::Error;

use crate::model::{
    ForwardError, ForwardState, Model, StreamPhase, forward_token, forward_token_traced,
};
use crate::tokenizer::{RvmpTokenizer, SamplingDefaults, TokenizerError};

/// Which pass a routing record came from.
///
/// Both passes go through the same [`ForwardState`] and the same expert
/// cache in this build (see the module docs), so the tag is not a statement
/// about how the experts were fetched. It exists because the two passes are
/// not comparable *workloads*: prefill walks a prompt whose positions the
/// caller chose, decode walks the model's own output, and a cache-hit rate
/// quoted over both at once is a different number from either. The offline
/// simulator (`scripts/lfu_sim.py`) filters on it and models decode only, so
/// anything measured against that simulation has to filter the same way.
///
/// `docs/architecture.md` ("Prefill") specifies a cache-bypassing
/// layer-major prefill sweep for phase 6. When that lands the tag will
/// additionally mean "fetched differently"; today it does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracePhase {
    /// A prompt token.
    Prefill,
    /// A generated token.
    Decode,
}

impl From<TracePhase> for StreamPhase {
    /// The two enums name the same split from opposite ends — the trace
    /// format's phase byte and the streamer's counter bucket — and
    /// [`generate`] is what keeps them in step, declaring the streamer's
    /// phase for every token whether or not anything is tracing.
    fn from(phase: TracePhase) -> Self {
        match phase {
            TracePhase::Prefill => Self::Prefill,
            TracePhase::Decode => Self::Decode,
        }
    }
}

/// Observer for [`generate_traced`], called once per layer per token with
/// `(phase, position, layer, top_k)`. `top_k` is the final
/// `(expert, weight)` selection in routed order (descending router
/// probability).
pub type RouteSink<'a> = &'a mut dyn FnMut(TracePhase, usize, u32, &[(u32, f32)]);

/// Knobs for one [`generate`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerateParams {
    /// Maximum tokens to generate.
    pub max_new: usize,
    /// Softmax temperature (applied after top-k/top-p). Values <= 0 fall
    /// back to greedy.
    pub temperature: f32,
    /// Top-k cutoff; `None` disables top-k truncation.
    pub top_k: Option<u32>,
    /// Nucleus-sampling probability mass; `>= 1.0` disables top-p.
    pub top_p: f32,
    /// PRNG seed for the stochastic path.
    pub seed: u64,
    /// Deterministic argmax decoding; ignores the sampling knobs.
    pub greedy: bool,
}

impl GenerateParams {
    /// Parameters seeded from a checkpoint's sampling defaults (for the v0
    /// pin: temperature 0.7, top-p 0.8, top-k 20), with `max_new` 128,
    /// seed 42, and greedy off.
    pub fn from_defaults(defaults: SamplingDefaults) -> Self {
        Self {
            max_new: 128,
            temperature: defaults.temperature,
            top_k: defaults.top_k,
            top_p: defaults.top_p,
            seed: 42,
            greedy: false,
        }
    }
}

/// Why a [`generate`] call stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// A stop token was sampled (it is not emitted or streamed).
    StopToken(u32),
    /// [`GenerateParams::max_new`] tokens were generated.
    MaxNew,
}

/// Counters and timings from one [`generate`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerateStats {
    /// Prompt tokens prefilled.
    pub prompt_tokens: usize,
    /// Tokens generated (stop token excluded).
    pub generated: usize,
    /// Why generation stopped.
    pub stop: StopReason,
    /// Wall time of the prefill loop.
    pub prefill: Duration,
    /// Wall time of the decode loop (sampling + forward passes).
    pub decode: Duration,
}

/// Typed errors from generation.
#[derive(Debug, Error)]
pub enum GenerateError {
    /// The forward pass failed.
    #[error(transparent)]
    Forward(#[from] ForwardError),

    /// Detokenization failed.
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),

    /// The prompt has no tokens: there are no logits to sample from.
    #[error("generate: empty prompt")]
    EmptyPrompt,

    /// Internal invariant: a `want_logits` pass returned no logits.
    /// Unreachable through the public API; reported instead of asserted.
    #[error("generate: forward pass returned no logits")]
    MissingLogits,
}

/// xorshift64* PRNG: deterministic, seedable, dependency-free.
#[derive(Debug)]
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        // State must be nonzero; splash a constant for seed 0.
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1) with 53 random bits.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Reusable sampler: owns the candidate scratch and the PRNG, so the
/// decode loop allocates only once.
#[derive(Debug)]
struct Sampler {
    params: GenerateParams,
    rng: Xorshift,
    /// `(token_id, logit)` candidates, filtered in place.
    candidates: Vec<(u32, f32)>,
}

impl Sampler {
    fn new(params: &GenerateParams, vocab: usize) -> Self {
        Self {
            params: *params,
            rng: Xorshift::new(params.seed),
            candidates: Vec::with_capacity(vocab),
        }
    }

    /// Argmax over raw logits; the first index wins ties.
    fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best as u32
    }

    /// Sample the next token id from `logits`.
    fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.params.greedy || self.params.temperature <= 0.0 || logits.len() < 2 {
            return Self::argmax(logits);
        }
        let cands = &mut self.candidates;
        cands.clear();
        cands.extend(logits.iter().enumerate().map(|(i, &l)| (i as u32, l)));

        // 1. Top-k: keep the k largest logits.
        if let Some(k) = self.params.top_k {
            let k = k as usize;
            if k > 0 && k < cands.len() {
                cands.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
                cands.truncate(k);
            }
        }
        cands.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // 2. Top-p: keep the smallest prefix whose (unscaled) softmax mass
        //    reaches top_p, always including the crossing token.
        if self.params.top_p < 1.0 {
            let max = f64::from(cands[0].1);
            let total: f64 = cands.iter().map(|c| (f64::from(c.1) - max).exp()).sum();
            let mut cum = 0.0f64;
            let mut keep = cands.len();
            for (i, c) in cands.iter().enumerate() {
                cum += (f64::from(c.1) - max).exp() / total;
                if cum >= f64::from(self.params.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            cands.truncate(keep.max(1));
        }

        // 3. Temperature, final softmax, one draw.
        let inv_t = 1.0 / f64::from(self.params.temperature);
        let max = f64::from(cands[0].1) * inv_t;
        let total: f64 = cands
            .iter()
            .map(|c| (f64::from(c.1) * inv_t - max).exp())
            .sum();
        let mut r = self.rng.next_f64() * total;
        for c in cands.iter() {
            r -= (f64::from(c.1) * inv_t - max).exp();
            if r < 0.0 {
                return c.0;
            }
        }
        // Rounding remainder: fall back to the least likely kept token.
        cands.last().map_or(0, |c| c.0)
    }
}

/// Ids [`StreamDecoder`] keeps once their text has been emitted.
///
/// The window only has to span the ids that can still be holding the
/// UTF-8 bytes of one unfinished character. A Unicode scalar is at most
/// four UTF-8 bytes and every byte-level BPE token carries at least one
/// byte, so at most four ids can split a character; eight leaves slack
/// for added tokens and for a decoder that is less byte-exact than this
/// vocabulary's plain `ByteLevel` one, at a bounded cost per push.
const STREAM_WINDOW_KEEP: usize = 8;

/// Hard cap on [`StreamDecoder`]'s window.
///
/// Reaching it means the window decode has ended in U+FFFD — or refused to
/// split cleanly — for this many ids in a row, which well-formed UTF-8
/// cannot do (see [`STREAM_WINDOW_KEEP`]); only a run of undecodable bytes
/// gets here. The decoder then stops withholding, emits the replacement
/// characters verbatim, and restarts with an empty window: visibly broken
/// text for input that is already broken, instead of a window that grows
/// with the generation. Well above the four ids real text needs, so no
/// legitimate sequence trips it.
const STREAM_WINDOW_MAX: usize = 64;

/// Incremental detokenizer: decodes a bounded trailing window of the
/// generated ids each push and returns only the newly-safe suffix. "Safe"
/// excludes trailing U+FFFD replacement characters, which mark a Unicode
/// character whose UTF-8 bytes are still split across future tokens
/// (byte-level BPE decode is byte-prefix-stable except for that tail).
///
/// # Cost
///
/// [`push`](Self::push) decodes the window once and then searches for a
/// clean cut (see [`trim`](Self::trim)), all of it bounded by
/// [`STREAM_WINDOW_MAX`] ids and in practice a little over
/// [`STREAM_WINDOW_KEEP`]. Nothing it does — decoded ids, bytes allocated,
/// bytes copied — depends on how many tokens have already been generated,
/// so a whole generation is O(n) rather than the O(n^2) of re-decoding the
/// accumulated sequence once per token.
///
/// # Why sliding is sound
///
/// `tokenizers`' decode of this vocabulary is
/// `String::from_utf8_lossy(concat(bytes(id) for id in ids))`: the bytes
/// concatenate and the lossy read is taken once over the whole run. So if
/// the ids dropped off the front decode on their own to complete, valid
/// UTF-8 that is a prefix of the window's decode, the rest of the window
/// decodes to exactly the suffix that [`emitted`](Self::emitted) indexes
/// into. [`trim`](Self::trim) drops ids only after checking exactly that,
/// so an added token or a split character straddling the window edge
/// simply stays in the window until it is whole and emitted.
#[derive(Debug, Default)]
struct StreamDecoder {
    /// Trailing ids of the generation, at most [`STREAM_WINDOW_MAX`].
    window: Vec<u32>,
    /// Bytes of the *window's* decode already emitted.
    emitted: usize,
    /// Last id pushed, independent of what the window still holds.
    last: Option<u32>,
    /// Ids handed to [`RvmpTokenizer::decode`] since construction; the
    /// scaling test asserts this grows per push by a bounded amount
    /// rather than with the sequence length.
    #[cfg(test)]
    decoded_ids: usize,
}

impl StreamDecoder {
    fn new() -> Self {
        Self::default()
    }

    /// Decode `self.window[start..end]`, counting the work for tests.
    fn decode_range(
        &mut self,
        tokenizer: &RvmpTokenizer,
        start: usize,
        end: usize,
    ) -> Result<String, TokenizerError> {
        #[cfg(test)]
        {
            self.decoded_ids += end - start;
        }
        tokenizer.decode(&self.window[start..end], false)
    }

    /// Append one id and return the newly emittable text (possibly empty).
    fn push(&mut self, tokenizer: &RvmpTokenizer, id: u32) -> Result<String, TokenizerError> {
        self.window.push(id);
        self.last = Some(id);
        let len = self.window.len();
        let text = self.decode_range(tokenizer, 0, len)?;

        // Withhold a trailing run of replacement characters: those are a
        // character whose remaining bytes live in ids not pushed yet. At
        // the cap that withholding is abandoned (see STREAM_WINDOW_MAX).
        let capped = len >= STREAM_WINDOW_MAX;
        let mut safe = text.len();
        if !capped {
            while text[..safe].ends_with('\u{FFFD}') {
                safe -= '\u{FFFD}'.len_utf8();
            }
        }
        let mut out = String::new();
        if safe > self.emitted {
            // In range by construction; `get` keeps a hostile tokenizer
            // from panicking us if decode were ever not prefix-stable.
            out = text.get(self.emitted..safe).unwrap_or("").to_owned();
            self.emitted = safe;
        }

        if capped {
            // The whole window has now been emitted, replacement
            // characters and all, so it carries no debt: start over
            // rather than keep growing.
            self.window.clear();
            self.emitted = 0;
        } else {
            self.trim(tokenizer, &text)?;
        }
        Ok(out)
    }

    /// Drop the longest leading run of ids that splits cleanly, back down
    /// to [`STREAM_WINDOW_KEEP`].
    ///
    /// `text` is the decode of the current window. A cut is clean when the
    /// dropped run decodes to complete UTF-8 (no trailing U+FFFD) that is
    /// already inside the emitted prefix and matches `text` there — the
    /// conditions that make the rest of the window decode to exactly the
    /// suffix `emitted` indexes into (see the type docs).
    ///
    /// Cuts have to be searched for rather than taken one id at a time:
    /// byte-level BPE routinely ends an id in the middle of a character
    /// (the rest of it merged into the next id), and no such id is ever a
    /// clean cut on its own. So the widest useful cut is tried first and
    /// narrowed — at most `len - STREAM_WINDOW_KEEP` decodes of at most
    /// that many ids. When nothing splits, the window grows by one and the
    /// next push searches one wider, bounded by [`STREAM_WINDOW_MAX`].
    fn trim(&mut self, tokenizer: &RvmpTokenizer, text: &str) -> Result<(), TokenizerError> {
        let len = self.window.len();
        if len <= STREAM_WINDOW_KEEP {
            return Ok(());
        }
        let mut cut = len - STREAM_WINDOW_KEEP;
        while cut > 0 {
            let head = self.decode_range(tokenizer, 0, cut)?;
            if !head.is_empty()
                && !head.ends_with('\u{FFFD}')
                && head.len() <= self.emitted
                && text.starts_with(&head)
            {
                self.window.drain(..cut);
                self.emitted -= head.len();
                return Ok(());
            }
            cut -= 1;
        }
        Ok(())
    }

    /// Everything still withheld (a trailing incomplete character, decoded
    /// with replacement characters), emptying the decoder's debt.
    fn flush(&mut self, tokenizer: &RvmpTokenizer) -> Result<String, TokenizerError> {
        if self.window.is_empty() {
            return Ok(String::new());
        }
        let len = self.window.len();
        let text = self.decode_range(tokenizer, 0, len)?;
        let out = text.get(self.emitted..).unwrap_or("").to_owned();
        self.emitted = text.len();
        Ok(out)
    }

    fn last_id(&self) -> Option<u32> {
        self.last
    }
}

/// Prefill `prompt_ids` and decode up to `max_new` tokens, streaming each
/// token id and its newly-decoded text through `on_token`.
///
/// Prefill runs [`forward_token`] per prompt token (logits only for the
/// last). Decode samples per [`GenerateParams`], stops on any of the
/// tokenizer's stop tokens (the stop token is neither counted nor
/// streamed) or after `max_new` tokens, and streams text via incremental
/// detokenization (see the module docs). `state` must be fresh (empty KV
/// cache); positions continue from the prompt.
///
/// # Errors
///
/// [`GenerateError::EmptyPrompt`] on an empty prompt; forward-pass and
/// tokenizer failures pass through typed. On error the state is mid-token
/// and must be discarded.
pub fn generate(
    model: &Model,
    state: &mut ForwardState,
    tokenizer: &RvmpTokenizer,
    prompt_ids: &[u32],
    params: &GenerateParams,
    mut on_token: impl FnMut(u32, &str),
) -> Result<GenerateStats, GenerateError> {
    generate_with_stops(
        model,
        state,
        tokenizer,
        prompt_ids,
        params,
        tokenizer.stop_tokens(),
        &mut on_token,
        None,
    )
}

/// [`generate`] with a [`RouteSink`] recording every layer's routing
/// decision for both the prefill and the decode passes.
///
/// Numerically identical to [`generate`] (see [`forward_token_traced`]).
///
/// # Errors
///
/// Exactly [`generate`]'s.
pub fn generate_traced(
    model: &Model,
    state: &mut ForwardState,
    tokenizer: &RvmpTokenizer,
    prompt_ids: &[u32],
    params: &GenerateParams,
    mut on_token: impl FnMut(u32, &str),
    trace: RouteSink<'_>,
) -> Result<GenerateStats, GenerateError> {
    generate_with_stops(
        model,
        state,
        tokenizer,
        prompt_ids,
        params,
        tokenizer.stop_tokens(),
        &mut on_token,
        Some(trace),
    )
}

/// One [`forward_token`], tagging any routing it reports with `phase` and
/// `position` before handing it to `trace`. Untraced runs take the plain
/// [`forward_token`] path.
///
/// `phase` is also declared to the expert streamer here — every forward pass
/// this module runs goes through this function, so the streamer's per-phase
/// counters cannot drift out of step with the trace's phase byte. It costs
/// one store per token, traced or not.
fn traced_step<'s>(
    model: &Model,
    state: &'s mut ForwardState,
    token_id: u32,
    position: usize,
    want_logits: bool,
    phase: TracePhase,
    trace: &mut Option<RouteSink<'_>>,
) -> Result<Option<&'s [f32]>, ForwardError> {
    state.set_stream_phase(StreamPhase::from(phase));
    match trace.as_deref_mut() {
        Some(sink) => {
            let mut on_route = |layer: u32, topk: &[(u32, f32)]| sink(phase, position, layer, topk);
            forward_token_traced(
                model,
                state,
                token_id,
                position,
                want_logits,
                Some(&mut on_route),
            )
        }
        None => forward_token(model, state, token_id, position, want_logits),
    }
}

/// [`generate`] with an explicit stop set (unit tests drive this with
/// synthetic stop tokens a tiny fixture model can actually emit) and an
/// optional routing trace.
#[allow(clippy::too_many_arguments)]
fn generate_with_stops(
    model: &Model,
    state: &mut ForwardState,
    tokenizer: &RvmpTokenizer,
    prompt_ids: &[u32],
    params: &GenerateParams,
    stop_tokens: &[u32],
    on_token: &mut dyn FnMut(u32, &str),
    mut trace: Option<RouteSink<'_>>,
) -> Result<GenerateStats, GenerateError> {
    let (&last, rest) = prompt_ids.split_last().ok_or(GenerateError::EmptyPrompt)?;

    let prefill_start = Instant::now();
    for (pos, &id) in rest.iter().enumerate() {
        traced_step(
            model,
            state,
            id,
            pos,
            false,
            TracePhase::Prefill,
            &mut trace,
        )?;
    }
    let mut logits = traced_step(
        model,
        state,
        last,
        rest.len(),
        true,
        TracePhase::Prefill,
        &mut trace,
    )?
    .ok_or(GenerateError::MissingLogits)?;
    let prefill = prefill_start.elapsed();

    let decode_start = Instant::now();
    let mut sampler = Sampler::new(params, logits.len());
    let mut stream = StreamDecoder::new();
    let mut position = prompt_ids.len();
    let mut generated = 0usize;
    let mut stop = StopReason::MaxNew;
    while generated < params.max_new {
        let next = sampler.sample(logits);
        if stop_tokens.contains(&next) {
            stop = StopReason::StopToken(next);
            break;
        }
        let text = stream.push(tokenizer, next)?;
        on_token(next, &text);
        generated += 1;
        if generated == params.max_new {
            break;
        }
        logits = traced_step(
            model,
            state,
            next,
            position,
            true,
            TracePhase::Decode,
            &mut trace,
        )?
        .ok_or(GenerateError::MissingLogits)?;
        position += 1;
    }
    // Flush a trailing incomplete character (attributed to the last id).
    let tail = stream.flush(tokenizer)?;
    if !tail.is_empty() {
        if let Some(id) = stream.last_id() {
            on_token(id, &tail);
        }
    }

    Ok(GenerateStats {
        prompt_tokens: prompt_ids.len(),
        generated,
        stop,
        prefill,
        decode: decode_start.elapsed(),
    })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::io::LoadOptions;
    use crate::io::testutil::build_install;
    use crate::model::{ForwardState, RuntimeConfig};

    /// Fixture-sized runtime dials: an unpinned two-shard pool and a small
    /// expert budget, so a test suite that runs many states in parallel
    /// neither pins every thread to one core nor reserves the production
    /// 1,438 MiB per state.
    fn small(model: &Model, context_cap: usize) -> ForwardState {
        ForwardState::with_config(model, context_cap, RuntimeConfig::testing()).unwrap()
    }

    /// The committed real-tokenizer fixtures (pinned Qwen3 vocabulary).
    fn fixtures_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tokenizer/fixtures")
    }

    fn fixture_tokenizer() -> RvmpTokenizer {
        RvmpTokenizer::load(&fixtures_dir()).expect("fixture tokenizer loads")
    }

    fn greedy_params(max_new: usize) -> GenerateParams {
        GenerateParams {
            max_new,
            temperature: 0.7,
            top_k: Some(20),
            top_p: 0.8,
            seed: 42,
            greedy: true,
        }
    }

    // --- Sampler ---

    fn sampled_params(temperature: f32, top_k: Option<u32>, top_p: f32, seed: u64) -> Sampler {
        Sampler::new(
            &GenerateParams {
                max_new: 1,
                temperature,
                top_k,
                top_p,
                seed,
                greedy: false,
            },
            8,
        )
    }

    #[test]
    fn greedy_is_argmax_and_first_index_wins_ties() {
        let logits = [0.5f32, 3.0, -1.0, 3.0];
        let mut sampler = Sampler::new(
            &GenerateParams {
                greedy: true,
                ..greedy_params(1)
            },
            logits.len(),
        );
        for _ in 0..4 {
            assert_eq!(sampler.sample(&logits), 1);
        }
        // Temperature <= 0 also degrades to greedy.
        let mut cold = sampled_params(0.0, None, 1.0, 7);
        assert_eq!(cold.sample(&logits), 1);
    }

    #[test]
    fn top_k_one_is_deterministic_argmax() {
        let logits = [0.1f32, 0.2, 5.0, 0.3, 4.9];
        let mut sampler = sampled_params(1.0, Some(1), 1.0, 123);
        for _ in 0..16 {
            assert_eq!(sampler.sample(&logits), 2);
        }
    }

    #[test]
    fn top_k_filters_to_the_k_largest() {
        // Only ids 2 and 4 survive top_k = 2; both must appear over draws.
        let logits = [0.0f32, 0.0, 5.0, 0.0, 4.9];
        let mut sampler = sampled_params(1.0, Some(2), 1.0, 99);
        let mut seen = [0usize; 5];
        for _ in 0..200 {
            seen[sampler.sample(&logits) as usize] += 1;
        }
        assert_eq!(seen[0] + seen[1] + seen[3], 0, "{seen:?}");
        assert!(seen[2] > 0 && seen[4] > 0, "{seen:?}");
    }

    #[test]
    fn top_p_keeps_the_smallest_covering_prefix() {
        // Softmax of [ln 6, ln 3, ln 1] = [0.6, 0.3, 0.1]. top_p = 0.7
        // keeps {0, 1} (0.6 < 0.7, then 0.9 >= 0.7 includes the crossing
        // token); id 2 must never appear.
        let logits = [6.0f32.ln(), 3.0f32.ln(), 1.0f32.ln()];
        let mut sampler = sampled_params(1.0, None, 0.7, 0xF00);
        let mut seen = [0usize; 3];
        for _ in 0..300 {
            seen[sampler.sample(&logits) as usize] += 1;
        }
        assert_eq!(seen[2], 0, "{seen:?}");
        assert!(seen[0] > 0 && seen[1] > 0, "{seen:?}");
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let logits: Vec<f32> = (0..64).map(|i| ((i * 37) % 11) as f32 * 0.3).collect();
        let draw = |seed: u64| -> Vec<u32> {
            let mut sampler = sampled_params(0.9, Some(8), 0.95, seed);
            (0..32).map(|_| sampler.sample(&logits)).collect()
        };
        assert_eq!(draw(1234), draw(1234));
        assert_ne!(draw(1234), draw(4321), "different seeds should diverge");
    }

    // --- StreamDecoder ---

    use crate::tokenizer::{ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID, IM_START_TOKEN_ID};

    /// A deliberately mixed sequence — ASCII, CJK, an emoji ZWJ cluster,
    /// a ChatML marker (which encodes to its single added-token id),
    /// astral-plane characters — repeated `reps` times, so a run over it
    /// slides the window many times with every awkward case recurring on
    /// both sides of a slide.
    fn mixed_ids(tokenizer: &RvmpTokenizer, reps: usize) -> Vec<u32> {
        const UNIT: &str = "hi \u{4F60}\u{597D}\u{4E16}\u{754C} \u{1F30D}\u{1F469}\u{200D}\u{1F4BB}\
                            <|im_end|>\n\u{13000}\u{1D518} ok ";
        tokenizer.encode(&UNIT.repeat(reps)).unwrap()
    }

    /// Drive a decoder over `ids`, asserting no push leaks a replacement
    /// character and the window stays bounded. Returns the assembled text
    /// (every push plus `flush`) and the ids each push handed to the
    /// tokenizer.
    fn drive_stream(tokenizer: &RvmpTokenizer, ids: &[u32]) -> (String, Vec<usize>) {
        let mut stream = StreamDecoder::new();
        let mut assembled = String::new();
        let mut per_push = Vec::with_capacity(ids.len());
        let mut counted = 0usize;
        for (i, &id) in ids.iter().enumerate() {
            let piece = stream.push(tokenizer, id).unwrap();
            assert!(
                !piece.contains('\u{FFFD}'),
                "push {i} leaked a replacement char: {piece:?}"
            );
            assert!(
                stream.window.len() <= STREAM_WINDOW_MAX,
                "window grew past its cap at push {i}"
            );
            assembled.push_str(&piece);
            per_push.push(stream.decoded_ids - counted);
            counted = stream.decoded_ids;
        }
        assembled.push_str(&stream.flush(tokenizer).unwrap());
        (assembled, per_push)
    }

    /// The property that actually matters: streaming is a partition of the
    /// one-shot decode. Whatever the window does, every push concatenated
    /// with the final flush must equal `decode(all_ids, false)`.
    #[test]
    fn stream_decoder_matches_the_full_decode() {
        let tokenizer = fixture_tokenizer();
        let enc = |s: &str| tokenizer.encode(s).unwrap();
        let mut raw_specials = vec![IM_START_TOKEN_ID];
        raw_specials.extend(enc("user\n"));
        raw_specials.extend([IM_END_TOKEN_ID, ENDOFTEXT_TOKEN_ID, 151_668]);

        let cases: Vec<(&str, Vec<u32>)> = vec![
            ("empty", Vec::new()),
            ("single ascii id", enc("hello")[..1].to_vec()),
            ("single added-token id", vec![IM_END_TOKEN_ID]),
            ("ascii", enc("hello world, 1 + 2 = 3. done!")),
            (
                "cjk",
                enc("\u{4F60}\u{597D}\u{4E16}\u{754C}\u{3002}\u{6D4B}\u{8BD5}\u{4E00}\u{4E0B}"),
            ),
            (
                "emoji",
                enc("ok \u{1F30D}\u{1F680}\u{1F469}\u{200D}\u{1F4BB} fine"),
            ),
            ("astral planes", enc("\u{13000}\u{1D518}\u{10348}")),
            (
                "chatml round trip",
                enc("<|im_start|>user\nhi \u{1F600}<|im_end|>\n<|im_start|>assistant\n"),
            ),
            ("raw added-token ids", raw_specials),
            ("long mixed, many slides", mixed_ids(&tokenizer, 24)),
        ];
        for (name, ids) in cases {
            let (assembled, _) = drive_stream(&tokenizer, &ids);
            assert_eq!(
                assembled,
                tokenizer.decode(&ids, false).unwrap(),
                "case {name}"
            );
        }
    }

    /// The window-boundary case the bounded tail introduces: a character
    /// whose bytes are split across ids, arriving after the window has
    /// already started sliding.
    #[test]
    fn stream_decoder_carries_a_character_across_a_window_slide() {
        let tokenizer = fixture_tokenizer();
        let split = ["\u{13000}", "\u{1D518}", "\u{10348}", "\u{1F30D}"]
            .iter()
            .map(|s| tokenizer.encode(s).unwrap())
            .find(|ids| ids.len() >= 2)
            .expect("some exotic character splits into multiple ids");
        assert!(split.len() <= STREAM_WINDOW_KEEP);
        let prefix = tokenizer
            .encode(&"the quick brown fox jumps over the lazy dog ".repeat(3))
            .unwrap();
        assert!(
            prefix.len() > STREAM_WINDOW_KEEP * 2,
            "{} ids",
            prefix.len()
        );

        let mut stream = StreamDecoder::new();
        let mut assembled = String::new();
        for &id in &prefix {
            assembled.push_str(&stream.push(&tokenizer, id).unwrap());
        }
        // Trimmed to the keep bound, so the character below starts at a
        // slid window edge rather than at the start of the sequence.
        assert_eq!(stream.window.len(), STREAM_WINDOW_KEEP);
        assert_eq!(assembled, tokenizer.decode(&prefix, false).unwrap());

        for (i, &id) in split.iter().enumerate() {
            let piece = stream.push(&tokenizer, id).unwrap();
            assert!(!piece.contains('\u{FFFD}'), "push {i}: {piece:?}");
            // The trim refuses to drop an id whose bytes are unfinished or
            // still withheld, so a slide cannot cut the character in half.
            assert!(stream.window.ends_with(&split[..=i]), "push {i}");
            assembled.push_str(&piece);
        }
        assembled.push_str(&stream.flush(&tokenizer).unwrap());

        let all: Vec<u32> = prefix.iter().chain(split.iter()).copied().collect();
        assert_eq!(assembled, tokenizer.decode(&all, false).unwrap());
    }

    /// `generate` calls `on_token` once per generated token, plus one more
    /// only when `flush` still owes text — `generated` or `generated + 1`,
    /// never more. That contract lives in these two shapes.
    #[test]
    fn stream_decoder_flush_owes_text_at_most_once() {
        let tokenizer = fixture_tokenizer();

        // Nothing withheld: no extra event.
        let mut stream = StreamDecoder::new();
        for &id in &tokenizer.encode("all done.").unwrap() {
            stream.push(&tokenizer, id).unwrap();
        }
        assert_eq!(stream.flush(&tokenizer).unwrap(), "");
        assert_eq!(stream.flush(&tokenizer).unwrap(), "");

        // Stopped mid-character: exactly one extra event, attributed to the
        // last id pushed, and nothing after it.
        let split = tokenizer.encode("\u{13000}").unwrap();
        assert!(split.len() >= 2);
        let head = &split[..split.len() - 1];
        let mut stream = StreamDecoder::new();
        for &id in head {
            assert_eq!(stream.push(&tokenizer, id).unwrap(), "");
        }
        let tail = stream.flush(&tokenizer).unwrap();
        assert!(tail.contains('\u{FFFD}'));
        assert_eq!(stream.last_id(), head.last().copied());
        assert_eq!(stream.flush(&tokenizer).unwrap(), "");
    }

    #[test]
    fn stream_decoder_handles_empty_and_single_id_sequences() {
        let tokenizer = fixture_tokenizer();
        let mut empty = StreamDecoder::new();
        assert_eq!(empty.flush(&tokenizer).unwrap(), "");
        assert_eq!(empty.last_id(), None);

        for id in [
            tokenizer.encode("hi").unwrap()[0],
            IM_START_TOKEN_ID,
            IM_END_TOKEN_ID,
        ] {
            let mut stream = StreamDecoder::new();
            let piece = stream.push(&tokenizer, id).unwrap();
            assert_eq!(piece, tokenizer.decode(&[id], false).unwrap(), "id {id}");
            assert_eq!(stream.flush(&tokenizer).unwrap(), "");
            assert_eq!(stream.last_id(), Some(id));
        }
    }

    /// The review finding this decoder closed (2026-08-02): re-decoding the
    /// accumulated ids per token is O(n^2). Counted decode work — not wall
    /// clock, which would flake on a loaded machine — pins that the cost of
    /// a push is set by the window, not by how much has been generated.
    #[test]
    fn stream_decoder_decode_work_per_push_does_not_grow_with_length() {
        let tokenizer = fixture_tokenizer();
        let short = mixed_ids(&tokenizer, 3);
        let long = mixed_ids(&tokenizer, 24);
        assert!(long.len() > short.len() * 4, "{} ids", long.len());

        let (short_text, short_push) = drive_stream(&tokenizer, &short);
        let (long_text, long_push) = drive_stream(&tokenizer, &long);
        assert_eq!(short_text, tokenizer.decode(&short, false).unwrap());
        assert_eq!(long_text, tokenizer.decode(&long, false).unwrap());

        // Same content, 8x the length, same worst-case push. The second
        // bound is deliberately far below `STREAM_WINDOW_MAX`: a trim that
        // stalls (an earlier draft cut one id at a time, which byte-level
        // BPE blocks whenever an id ends mid-character) still terminates
        // and still decodes correctly, but drifts the window up to the cap.
        let short_worst = short_push.iter().copied().max().unwrap();
        let long_worst = long_push.iter().copied().max().unwrap();
        assert!(
            long_worst <= short_worst && long_worst <= STREAM_WINDOW_KEEP * 3,
            "per-push decode grew with length: {short_worst} -> {long_worst}"
        );

        // And therefore the whole run is linear, not quadratic: the old
        // full re-decode would have cost n(n+1)/2 ids.
        let n = long.len();
        let total: usize = long_push.iter().sum();
        assert!(total <= n * STREAM_WINDOW_KEEP * 3, "total {total}, n {n}");
        assert!(total * 4 < n * (n + 1) / 2, "total {total}, n {n}");
    }

    #[test]
    fn stream_decoder_withholds_split_utf8() {
        let tokenizer = fixture_tokenizer();
        // Find a character the byte-level BPE splits across tokens, so the
        // partial decode really produces trailing U+FFFD.
        let ids = ["\u{13000}", "\u{1D518}", "\u{1F30D}", "\u{10348}"]
            .iter()
            .map(|s| tokenizer.encode(s).unwrap())
            .find(|ids| ids.len() >= 2)
            .expect("some exotic character splits into multiple tokens");
        let full = tokenizer.decode(&ids, false).unwrap();

        let mut stream = StreamDecoder::new();
        let mut assembled = String::new();
        for (i, &id) in ids.iter().enumerate() {
            let piece = stream.push(&tokenizer, id).unwrap();
            assert!(
                !piece.contains('\u{FFFD}'),
                "piece {i} leaked a replacement char: {piece:?}"
            );
            assembled.push_str(&piece);
        }
        assembled.push_str(&stream.flush(&tokenizer).unwrap());
        assert_eq!(assembled, full);
        // Flushing again yields nothing.
        assert_eq!(stream.flush(&tokenizer).unwrap(), "");
    }

    #[test]
    fn stream_decoder_flushes_incomplete_tail_at_stop() {
        let tokenizer = fixture_tokenizer();
        let ids = tokenizer.encode("\u{13000}").unwrap();
        assert!(ids.len() >= 2);
        let mut stream = StreamDecoder::new();
        // Push all but the final byte-carrying token: text stays withheld.
        let text = stream.push(&tokenizer, ids[0]).unwrap();
        assert_eq!(text, "");
        // A mid-character stop flushes the replacement-rendered tail.
        let tail = stream.flush(&tokenizer).unwrap();
        assert!(tail.contains('\u{FFFD}'));
    }

    #[test]
    fn stream_decoder_ascii_is_immediate() {
        let tokenizer = fixture_tokenizer();
        let ids = tokenizer.encode("hello world").unwrap();
        let mut stream = StreamDecoder::new();
        let mut assembled = String::new();
        for &id in &ids {
            assembled.push_str(&stream.push(&tokenizer, id).unwrap());
        }
        assert_eq!(assembled, "hello world");
        assert_eq!(stream.flush(&tokenizer).unwrap(), "");
    }

    // --- Generation loop over the synthetic fixture ---

    struct Harness {
        _fx: crate::io::testutil::Fixture,
        model: Model,
        tokenizer: RvmpTokenizer,
    }

    fn harness(tag: &str) -> Harness {
        let fx = build_install(tag);
        crate::model::testsupport::temper_install(&fx);
        let model = Model::load(
            &fx.root,
            LoadOptions {
                skip_hashes: true,
                ..LoadOptions::default()
            },
        )
        .unwrap();
        Harness {
            _fx: fx,
            model,
            tokenizer: fixture_tokenizer(),
        }
    }

    fn run_greedy(
        h: &Harness,
        prompt: &[u32],
        max_new: usize,
        stops: &[u32],
    ) -> (Vec<u32>, String, GenerateStats) {
        let mut state = small(&h.model, 16);
        let mut events: Vec<(u32, String)> = Vec::new();
        let mut on_token = |id: u32, piece: &str| {
            events.push((id, piece.to_owned()));
        };
        let stats = generate_with_stops(
            &h.model,
            &mut state,
            &h.tokenizer,
            prompt,
            &greedy_params(max_new),
            stops,
            &mut on_token,
            None,
        )
        .unwrap();
        // One event per generated token, plus at most one flush event that
        // re-reports the last id with the withheld tail text.
        assert!(events.len() == stats.generated || events.len() == stats.generated + 1);
        let ids: Vec<u32> = events.iter().take(stats.generated).map(|e| e.0).collect();
        let text: String = events.iter().map(|e| e.1.as_str()).collect();
        (ids, text, stats)
    }

    #[test]
    fn greedy_generation_is_deterministic_and_counts_match() {
        let h = harness("gen-greedy");
        let (ids_a, text_a, stats) = run_greedy(&h, &[1, 2, 3], 4, &[]);
        assert_eq!(stats.prompt_tokens, 3);
        assert_eq!(stats.generated, 4);
        assert_eq!(stats.stop, StopReason::MaxNew);
        assert_eq!(ids_a.len(), 4);
        // Every generated id is inside the fixture vocab.
        assert!(ids_a.iter().all(|&id| (id as usize) < 32));
        // The streamed text is the decode of the generated ids.
        assert_eq!(text_a, h.tokenizer.decode(&ids_a, false).unwrap());

        let (ids_b, text_b, _) = run_greedy(&h, &[1, 2, 3], 4, &[]);
        assert_eq!(ids_a, ids_b);
        assert_eq!(text_a, text_b);
    }

    #[test]
    fn stop_token_halts_generation_and_is_not_streamed() {
        let h = harness("gen-stop");
        let (ids, _, _) = run_greedy(&h, &[1, 2, 3], 4, &[]);
        assert!(ids.len() >= 2);
        // Pick the first position whose token has not occurred earlier in
        // the greedy stream (so the stop cannot fire prematurely), then
        // rerun with it as the stop set: generation must emit exactly the
        // prefix before it and report the stop id.
        let j = (1..ids.len())
            .find(|&j| !ids[..j].contains(&ids[j]))
            .unwrap_or(0);
        let stop = ids[j];
        let (stopped_ids, text, stats) = run_greedy(&h, &[1, 2, 3], 4, &[stop]);
        assert_eq!(stopped_ids, ids[..j].to_vec());
        assert_eq!(stats.generated, j);
        assert_eq!(stats.stop, StopReason::StopToken(stop));
        assert_eq!(text, h.tokenizer.decode(&ids[..j], false).unwrap());
    }

    #[test]
    fn max_new_zero_generates_nothing() {
        let h = harness("gen-zero");
        let (ids, text, stats) = run_greedy(&h, &[1], 0, &[]);
        assert!(ids.is_empty());
        assert!(text.is_empty());
        assert_eq!(stats.generated, 0);
        assert_eq!(stats.stop, StopReason::MaxNew);
        assert_eq!(stats.prompt_tokens, 1);
    }

    #[test]
    fn tracing_records_every_layer_and_changes_nothing() {
        let h = harness("gen-trace");
        let prompt = [1u32, 2, 3];
        let (want_ids, want_text, want_stats) = run_greedy(&h, &prompt, 4, &[]);

        let arch = h.model.arch();
        let (n_layers, top_k) = (arch.n_layers, arch.top_k as usize);
        let mut records: Vec<(TracePhase, usize, u32, Vec<u32>)> = Vec::new();
        let mut sink = |phase: TracePhase, pos: usize, layer: u32, topk: &[(u32, f32)]| {
            records.push((phase, pos, layer, topk.iter().map(|&(e, _)| e).collect()));
        };
        let mut state = small(&h.model, 16);
        let mut ids = Vec::new();
        let mut text = String::new();
        let stats = generate_traced(
            &h.model,
            &mut state,
            &h.tokenizer,
            &prompt,
            &greedy_params(4),
            |id, piece| {
                ids.push(id);
                text.push_str(piece);
            },
            &mut sink,
        )
        .unwrap();

        // Identical output: the sink observes, it does not perturb.
        assert_eq!(stats.stop, StopReason::MaxNew);
        assert_eq!(stats.generated, want_stats.generated);
        assert_eq!(ids[..want_ids.len()], want_ids[..]);
        assert_eq!(text, want_text);

        // One record per layer per token, layers in order, phases split at
        // the prompt boundary, and every routed id inside the expert count.
        let tokens = prompt.len() + stats.generated - 1;
        assert_eq!(records.len(), tokens * n_layers as usize);
        for (i, (phase, pos, layer, experts)) in records.iter().enumerate() {
            let token = i / n_layers as usize;
            assert_eq!(*layer as usize, i % n_layers as usize);
            assert_eq!(*pos, token);
            let want_phase = if token < prompt.len() {
                TracePhase::Prefill
            } else {
                TracePhase::Decode
            };
            assert_eq!(*phase, want_phase, "record {i}");
            assert_eq!(experts.len(), top_k);
            assert!(experts.iter().all(|&e| e < arch.n_experts), "{experts:?}");
            // torch.topk semantics: no expert is selected twice.
            let mut unique = experts.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), top_k, "{experts:?}");
        }
    }

    /// F3. The streamer's counters are cumulative from construction, so a
    /// footer that quotes them reports prefill folded into what reads as a
    /// decode number: a five-token prompt and `--max-new 4` is eight forward
    /// passes, five of them prefill. `generate` declares the phase to the
    /// streamer per token; this pins that the split is real, exhaustive, and
    /// lands exactly where the pass counts say it should.
    #[test]
    fn streaming_counters_split_prefill_from_decode() {
        let h = harness("gen-phase-split");
        let prompt = [1u32, 2, 3];
        let max_new = 4;
        let mut state = small(&h.model, 16);
        let stats = generate_with_stops(
            &h.model,
            &mut state,
            &h.tokenizer,
            &prompt,
            &greedy_params(max_new),
            &[],
            &mut |_, _| {},
            None,
        )
        .unwrap();
        assert_eq!(stats.generated, max_new);

        let arch = h.model.arch();
        // One request per routed expert per layer per forward pass.
        let per_pass = u64::from(arch.n_layers) * u64::from(arch.top_k);
        let prefill = state.stream_stats_in(StreamPhase::Prefill);
        let decode = state.stream_stats_in(StreamPhase::Decode);
        let total = state.stream_stats();

        // Prefill runs one pass per prompt token. Decode runs one per
        // generated token *after* the first, which is sampled from the
        // prompt's logits.
        assert_eq!(prefill.accesses(), prompt.len() as u64 * per_pass);
        assert_eq!(decode.accesses(), (max_new as u64 - 1) * per_pass);

        // Exhaustive and disjoint: every request lands in exactly one phase.
        assert_eq!(prefill.accesses() + decode.accesses(), total.accesses());
        assert_eq!(prefill.misses + decode.misses, total.misses);
        assert_eq!(prefill.hits + decode.hits, total.hits);
        assert_eq!(prefill.bytes_read + decode.bytes_read, total.bytes_read);

        // And the point of all of it: the cumulative figure is not the decode
        // figure, and quoting it as one overstates the work by the prompt.
        assert_ne!(total.accesses(), decode.accesses());
    }

    #[test]
    fn empty_prompt_is_a_typed_error() {
        let h = harness("gen-empty");
        let mut state = small(&h.model, 16);
        let err = generate(
            &h.model,
            &mut state,
            &h.tokenizer,
            &[],
            &greedy_params(4),
            |_, _| {},
        )
        .unwrap_err();
        assert!(matches!(err, GenerateError::EmptyPrompt));
    }

    #[test]
    fn public_generate_uses_the_pinned_stop_tokens() {
        // The fixture model's 32-token vocab can never emit 151645/151643,
        // so the public wrapper runs to max_new.
        let h = harness("gen-public");
        let mut state = small(&h.model, 16);
        let mut count = 0usize;
        let stats = generate(
            &h.model,
            &mut state,
            &h.tokenizer,
            &[4, 5],
            &greedy_params(3),
            |_, _| count += 1,
        )
        .unwrap();
        assert_eq!(stats.stop, StopReason::MaxNew);
        assert_eq!(stats.generated, 3);
        assert!(count >= 3);
    }
}
