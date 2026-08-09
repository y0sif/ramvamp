//! Generation orchestration: prefill, token-by-token decode, sampling.
//!
//! **Prefill is the chunked layer-major sweep**
//! ([`prefill_prompt`](crate::model::prefill_prompt)): up to
//! [`DEFAULT_PREFILL_CHUNK`](crate::model::DEFAULT_PREFILL_CHUNK) prompt
//! positions are carried through the model together and each layer's expert
//! file is streamed once per chunk, bypassing the decode cache entirely.
//! Decode then runs [`forward_token`] one generated token at a time through
//! that cache. Sampling supports greedy, temperature, top-k, and top-p
//! (repetition penalty is not implemented yet); greedy decode must be
//! deterministic for validation against reference implementations.
//!
//! The streaming counters reflect the split: a pure-prefill run reports **no
//! cache accesses at all** (its bytes land in `sweep_bytes_read` and its
//! windows in `sweep_windows_read`), because prompt positions no longer
//! request experts through the slot cache. Selecting
//! [`PrefillMode::TokenMajor`](crate::model::PrefillMode) restores the
//! phase-5 behaviour, cache accounting included; it exists for the
//! byte-identical-logits A/B and is not the default.
//!
//! # Continuing a sequence
//!
//! [`generate_from`] takes a starting position, so a second turn prefills
//! only the new tokens against a KV cache the first turn left behind. This
//! is the *only* correct way to continue a conversation: a ChatML
//! generation-prompt render is not always an id-prefix of the finished
//! assistant turn (the `\n` that ends `<|im_start|>assistant\n` and the head
//! of the reply are candidates for the same BPE merge), so a cache cannot be
//! extended by re-encoding the reply's text. The ids that were actually
//! generated are reported in [`GenerateStats::generated_ids`] for exactly
//! that reason.
//!
//! # Watching a call from outside it
//!
//! A generate call holds `&mut ForwardState` from the first prefill chunk to
//! the last token, and every telemetry accessor on that state takes `&self`,
//! so a driver cannot read one while a call is in flight. That leaves a long
//! prompt looking like a hang. [`generate_from_with_progress`] is the way
//! through: an optional [`GenerateProgress`] callback fired from inside the
//! loops, once per prefill chunk and once per sampled token. It is additive
//! and off by default — [`generate_from`] is that function with `None` — and
//! it is observation only, never a hook that can change what is computed.
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

use crate::io::StreamStats;
use crate::model::{
    ForwardError, ForwardState, Model, PrefillProgressSink, PrefillRouteSink, StreamPhase,
    forward_token, forward_token_traced, prefill_prompt_with_progress,
};
use crate::tokenizer::{RvmpTokenizer, SamplingDefaults, TokenizerError};

/// Which pass a routing record came from.
///
/// The two passes are not comparable *workloads*: prefill walks a prompt
/// whose positions the caller chose, decode walks the model's own output,
/// and a cache-hit rate quoted over both at once is a different number from
/// either. The offline simulator (`scripts/lfu_sim.py`) filters on this tag
/// and models decode only, so anything measured against that simulation has
/// to filter the same way.
///
/// Under the default [`PrefillMode::Sweep`](crate::model::PrefillMode) the
/// tag additionally means "fetched differently": prefill bypasses the slot
/// cache and streams each layer front to back. It also means "emitted in a
/// different order" — the sweep reports `(layer, row)`, the token-major path
/// reports `(token, layer)` — because that is the order the work happens in.
/// The set of `(position, layer)` pairs and the selections at each are
/// identical either way.
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

/// A progress event reported from inside a generate call.
///
/// [`generate_from_with_progress`] exists because a generate call is opaque
/// from the outside for as long as it runs: it holds `&mut ForwardState` for
/// the whole call, so the telemetry accessors on that state — every one of
/// which takes `&self` — are unreachable from another thread until it
/// returns. A long prompt is therefore a multi-second stall with nothing to
/// show, and `on_token` says nothing at all until the first token is sampled.
/// This enum is the inside of the call talking to whatever is driving it.
///
/// It is instrumentation and nothing else. Numerics, ordering and scheduling
/// are identical whether a callback is installed or not, which
/// `progress_does_not_perturb_generation` pins.
#[derive(Clone, Copy, Debug)]
pub enum GenerateProgress {
    /// Fired once after each prefill chunk completes.
    ///
    /// Both counts are relative to this call's prompt, so a turn continuing a
    /// cache from position 4,000 still reports `0..=prompt_ids.len()`.
    /// `positions_done` increases strictly to `positions_total`; the number
    /// of events is not fixed, because the sweep's chunk width is narrowed to
    /// whatever the expert slot slab can host and
    /// [`PrefillMode::TokenMajor`](crate::model::PrefillMode) reports per
    /// token rather than per chunk (see
    /// [`PrefillProgressSink`](crate::model::PrefillProgressSink)).
    PrefillChunk {
        /// Prompt positions committed so far, counted from this call's start.
        positions_done: usize,
        /// Prompt positions this call will consume in total.
        positions_total: usize,
    },
    /// Fired once per decoded token, before `on_token` for that token.
    ///
    /// `index` is 0-based over the tokens actually generated, so it always
    /// ends one short of [`GenerateStats::generated`]. Two things do **not**
    /// produce an event: a sampled stop token (which is never streamed and
    /// never counted), and the trailing `on_token` call that flushes a
    /// partial UTF-8 character at end of generation — that flush re-reports
    /// an id already seen and samples nothing, so counting it would make
    /// `index` disagree with the stats. One `DecodeToken` per sampled token,
    /// exactly.
    DecodeToken {
        /// 0-based index of this token within the generated sequence.
        index: usize,
        /// The streamer's counters for the decode phase as of this token,
        /// i.e. [`ForwardState::stream_stats_in`] at
        /// [`StreamPhase::Decode`]. Decode-only on purpose: the cumulative
        /// figure folds the prompt in and is not a decode number (see
        /// [`ForwardState::stream_stats`]).
        stats: StreamStats,
    },
}

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

/// Counters, timings and output ids from one [`generate`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateStats {
    /// Prompt tokens prefilled.
    pub prompt_tokens: usize,
    /// Tokens generated (stop token excluded). Always
    /// `generated_ids.len()`.
    pub generated: usize,
    /// The ids that were generated, in order, stop token excluded.
    ///
    /// The only sound way to continue a conversation: a KV cache is extended
    /// by appending the ids that were actually produced, never by re-encoding
    /// the reply's text (see the module docs). `on_token` cannot substitute
    /// for this, because its final flush event repeats the last id to carry
    /// the withheld tail of a split character.
    pub generated_ids: Vec<u32>,
    /// Why generation stopped.
    pub stop: StopReason,
    /// Wall time of the prefill pass.
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

    /// `start_position` does not continue the state's KV cache.
    ///
    /// Validated rather than trusted: a caller continuing a conversation owns
    /// the bookkeeping, and a silent off-by-one would prefill the new turn at
    /// the wrong RoPE positions and produce plausible nonsense.
    #[error("generate: start_position {position}, but the state holds {expected} positions")]
    PositionMismatch {
        /// The position the caller asked to continue from.
        position: usize,
        /// The position the state actually expects next.
        expected: usize,
    },

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
/// Reaching it means [`trim`](StreamDecoder::trim) has refused to split for
/// `STREAM_WINDOW_MAX - STREAM_WINDOW_KEEP` pushes in a row, which
/// well-formed UTF-8 cannot do (see [`STREAM_WINDOW_KEEP`]): only a run of
/// undecodable bytes, or a token stream whose every id boundary lands
/// strictly inside a character, gets this far. The push that reaches the cap
/// therefore stops waiting for a clean split and forces one — see
/// [`force_cut`](StreamDecoder::force_cut), which drops straight back to
/// [`STREAM_WINDOW_KEEP`] ids **without emitting anything
/// `decode(all_ids, false)` does not**, so the window is bounded here
/// unconditionally and the decoder's contract survives it. Well above the
/// four ids real text needs, so no legitimate sequence trips it.
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
/// clean cut (see [`trim`](Self::trim)); the push that reaches
/// [`STREAM_WINDOW_MAX`] decodes the retained tail once more. Everything is
/// bounded by the window, never by the sequence. The **bound** is the trim
/// search, which decodes `cut` ids for every cut it tries and so costs at
/// most `(MAX - KEEP)(MAX - KEEP + 1) / 2` = 1,596 decoded ids in a push, on
/// top of the window's own decode. Approaching that takes a pathological id
/// run — one where no leading run ever splits cleanly, which is what drives
/// the window to the cap in the first place. Separately **measured** on
/// mixed real text (ASCII, CJK, an emoji ZWJ cluster, added tokens): a worst
/// case of 14 decoded ids per push, and the scaling test pins it under
/// `STREAM_WINDOW_KEEP * 3`. Either way nothing here — decoded ids, bytes
/// allocated, bytes copied — depends on how many tokens have already been
/// generated, so a whole generation is O(n) rather than the O(n^2) of
/// re-decoding the accumulated sequence once per token.
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

        // At the cap the search for a clean split has failed for long
        // enough; cut regardless (see STREAM_WINDOW_MAX).
        if len >= STREAM_WINDOW_MAX {
            return self.force_cut(tokenizer, &text);
        }

        // Withhold a trailing run of replacement characters: those are a
        // character whose remaining bytes live in ids not pushed yet.
        let mut safe = text.len();
        while text[..safe].ends_with('\u{FFFD}') {
            safe -= '\u{FFFD}'.len_utf8();
        }
        let mut out = String::new();
        if safe > self.emitted {
            // In range by construction; `get` keeps a hostile tokenizer
            // from panicking us if decode were ever not prefix-stable.
            out = text.get(self.emitted..safe).unwrap_or("").to_owned();
            self.emitted = safe;
        }

        self.trim(tokenizer, &text)?;
        Ok(out)
    }

    /// Force the window back down to [`STREAM_WINDOW_KEEP`] ids, emitting
    /// everything in `text` that can no longer change.
    ///
    /// Runs when the window reaches [`STREAM_WINDOW_MAX`], i.e. when
    /// [`trim`](Self::trim) has found no clean split for
    /// `MAX - KEEP` pushes running. It has to cut anyway, and it must do so
    /// without breaking the contract the rest of the decoder keeps: every
    /// `push` concatenated with the final [`flush`](Self::flush) equals
    /// `decode(all_ids, false)`. Two facts make that possible.
    ///
    /// **Only the last replacement character is provisional.** Lossy UTF-8
    /// decoding is greedy and left to right: each maximal invalid subpart is
    /// terminated by a byte that cannot extend it, and appending bytes never
    /// revisits that decision. The one exception is a subpart still open at
    /// the end of the input — the split character this decoder exists to
    /// withhold — and an open subpart renders as exactly one U+FFFD. So the
    /// whole-trailing-run withholding [`push`](Self::push) does is merely
    /// conservative: holding back the final replacement character alone is
    /// sufficient, and unlike the run it always leaves something emittable.
    /// That is what breaks the deadlock, because the id runs that reach the
    /// cap are exactly the ones whose decode is *all* replacement characters.
    ///
    /// **The cut need not be a character boundary.** The retained tail is
    /// re-decoded on its own, so its leading bytes may be the back half of a
    /// character whose front half is being dropped and decode to replacement
    /// characters the whole window's decode did not have. That is harmless:
    /// those bytes were already emitted correctly, out of the whole window's
    /// decode, where the character was intact. So [`emitted`](Self::emitted)
    /// is re-anchored from the *back* — the tail's last `withheld` bytes are
    /// the same open subpart the whole window ended on (an open subpart is at
    /// most four bytes and every id carries at least one, so it cannot reach
    /// past the retained ids), and everything before them counts as emitted.
    /// Dropping a prefix can only make the tail's decode *begin* with more
    /// replacement characters, and those come from permanently invalid bytes,
    /// so the tail's decode stays prefix-stable as the window grows again and
    /// `emitted` keeps indexing what it did before the cut.
    fn force_cut(
        &mut self,
        tokenizer: &RvmpTokenizer,
        text: &str,
    ) -> Result<String, TokenizerError> {
        let mut safe = text.len();
        if text.ends_with('\u{FFFD}') {
            safe -= '\u{FFFD}'.len_utf8();
        }
        let withheld = text.len() - safe;
        // `get`, as in `push`: never panic on a decode that is not
        // prefix-stable, just emit nothing this push.
        let out = text.get(self.emitted..safe).unwrap_or("").to_owned();

        let len = self.window.len();
        let cut = len.saturating_sub(STREAM_WINDOW_KEEP);
        let tail = self.decode_range(tokenizer, cut, len)?;
        self.window.drain(..cut);
        self.emitted = tail.len().saturating_sub(withheld);
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
/// Prefill runs [`prefill_prompt`](crate::model::prefill_prompt) over the
/// whole prompt (logits only for the last position). Decode samples per
/// [`GenerateParams`], stops on any of the tokenizer's stop tokens (the stop
/// token is neither counted nor streamed) or after `max_new` tokens, and
/// streams text via incremental detokenization (see the module docs). `state`
/// must be fresh (empty KV cache); use [`generate_from`] to continue one.
///
/// Nothing here reports progress; [`generate_from_with_progress`] is the
/// entry point that does.
///
/// # Errors
///
/// [`GenerateError::EmptyPrompt`] on an empty prompt;
/// [`GenerateError::PositionMismatch`] when `state` is not fresh;
/// forward-pass and tokenizer failures pass through typed. On error the
/// state is mid-pass and must be discarded or [`ForwardState::reset`].
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
        0,
        params,
        tokenizer.stop_tokens(),
        &mut on_token,
        None,
        None,
    )
}

/// [`generate`] continuing a KV cache that already holds `start_position`
/// positions.
///
/// `prompt_ids` is only the **new** tokens — the next turn's render minus
/// everything the state has already seen — and `start_position` must equal
/// [`ForwardState::seq_len`]. This is what makes a multi-turn REPL cost one
/// turn's prefill instead of one whole conversation's, and one
/// [`ForwardState`] instead of one per turn: constructing a state reserves
/// the ~1,438 MiB expert slot pool, opens io_uring and spawns the pinned
/// compute pool.
///
/// The new tokens must be the ids that actually continue the sequence. For an
/// assistant turn that means the ids in
/// [`GenerateStats::generated_ids`], never a re-encoding of the reply's text:
/// a generation-prompt render is not always an id-prefix of the finished
/// turn, because the `\n` closing `<|im_start|>assistant\n` and the first
/// characters of the reply can merge (measured, across `"hello"`,
/// `" hello"`, `"\nhello"`, `"```rust"` and `"    indented"`).
///
/// # Which ids are "new"
///
/// Not every generated id reaches the cache: the last one sampled is emitted
/// and never fed back (there is nothing left to predict from it), and a
/// sampled stop token is not fed either. So a caller keeps the full id
/// history and lets the cache say where it is:
///
/// ```text
/// history.extend(&stats.generated_ids);      // turn N's reply
/// history.extend(render_next_user_turn());   // turn N+1's prompt
/// let fed = state.seq_len()?;                // what the model has seen
/// generate_from(model, state, tok, &history[fed..], fed, params, on_token)?;
/// ```
///
/// `state.seq_len()` is authoritative and `history[fed..]` is by construction
/// exactly the suffix it has not consumed.
///
/// # Errors
///
/// [`GenerateError::PositionMismatch`] when `start_position` disagrees with
/// the cache — including when a previous pass failed and left it ragged —
/// plus everything [`generate`] returns.
pub fn generate_from(
    model: &Model,
    state: &mut ForwardState,
    tokenizer: &RvmpTokenizer,
    prompt_ids: &[u32],
    start_position: usize,
    params: &GenerateParams,
    on_token: impl FnMut(u32, &str),
) -> Result<GenerateStats, GenerateError> {
    generate_from_with_progress(
        model,
        state,
        tokenizer,
        prompt_ids,
        start_position,
        params,
        None,
        on_token,
    )
}

/// [`generate_from`] reporting prefill and decode progress as it goes.
///
/// `on_progress` sees a [`GenerateProgress::PrefillChunk`] after each prefill
/// chunk commits and a [`GenerateProgress::DecodeToken`] before each
/// `on_token`, which is what lets a UI show a prompt filling and live
/// streaming counters during a call that holds `&mut ForwardState` from start
/// to finish. `on_token` is unchanged and still carries the text.
///
/// Passing `None` is [`generate_from`] exactly — the same code path, the same
/// arithmetic in the same order, the same bytes out. Nothing between the two
/// entry points reads the callback except an `if let` at each of the two fire
/// sites.
///
/// # Errors
///
/// Exactly [`generate_from`]'s. A callback cannot fail: it returns `()`, so
/// there is no way for an observer to abort a run, and equally no way for one
/// to invent an error the uninstrumented path would not have produced.
#[allow(clippy::too_many_arguments)]
pub fn generate_from_with_progress(
    model: &Model,
    state: &mut ForwardState,
    tokenizer: &RvmpTokenizer,
    prompt_ids: &[u32],
    start_position: usize,
    params: &GenerateParams,
    on_progress: Option<&mut dyn FnMut(GenerateProgress)>,
    mut on_token: impl FnMut(u32, &str),
) -> Result<GenerateStats, GenerateError> {
    generate_with_stops(
        model,
        state,
        tokenizer,
        prompt_ids,
        start_position,
        params,
        tokenizer.stop_tokens(),
        &mut on_token,
        None,
        on_progress,
    )
}

/// [`generate`] with a [`RouteSink`] recording every layer's routing
/// decision for both the prefill and the decode passes.
///
/// Numerically identical to [`generate`] (see [`forward_token_traced`]).
///
/// Starts at position 0 and so needs a fresh `state`: unlike
/// [`generate_from`] there is no way to trace a turn that continues a cache.
/// That is a gap, not a design decision — the trace format's positions are
/// absolute, so the plumbing is a `start_position` parameter and nothing else.
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
        0,
        params,
        tokenizer.stop_tokens(),
        &mut on_token,
        Some(trace),
        None,
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
/// synthetic stop tokens a tiny fixture model can actually emit), a starting
/// position, an optional routing trace, and an optional progress observer.
#[allow(clippy::too_many_arguments)]
fn generate_with_stops(
    model: &Model,
    state: &mut ForwardState,
    tokenizer: &RvmpTokenizer,
    prompt_ids: &[u32],
    start_position: usize,
    params: &GenerateParams,
    stop_tokens: &[u32],
    on_token: &mut dyn FnMut(u32, &str),
    mut trace: Option<RouteSink<'_>>,
    mut on_progress: Option<&mut dyn FnMut(GenerateProgress)>,
) -> Result<GenerateStats, GenerateError> {
    if prompt_ids.is_empty() {
        return Err(GenerateError::EmptyPrompt);
    }
    // The cache decides where the sequence actually is; `start_position` is
    // the caller's claim about it, and the two have to agree before a single
    // RoPE position is computed. A ragged cache (a previous pass abandoned
    // mid-token) reports as a mismatch rather than as an opaque kv error.
    let expected = state
        .seq_len()
        .map_err(|_| GenerateError::PositionMismatch {
            position: start_position,
            expected: usize::MAX,
        })?;
    if start_position != expected {
        return Err(GenerateError::PositionMismatch {
            position: start_position,
            expected,
        });
    }

    let prefill_start = Instant::now();
    state.set_stream_phase(StreamPhase::Prefill);
    let mut logits = {
        // Both sinks borrow their observer for the length of the prefill
        // only; the decode loop below needs `on_progress` back.
        let mut tagged;
        let sink: Option<PrefillRouteSink<'_>> = match trace.as_deref_mut() {
            Some(inner) => {
                tagged = |position: usize, layer: u32, topk: &[(u32, f32)]| {
                    inner(TracePhase::Prefill, position, layer, topk);
                };
                Some(&mut tagged)
            }
            None => None,
        };
        // The prefill layer counts positions; naming them is this layer's
        // job, so the pair is widened into the event here rather than there.
        let mut counted;
        let progress: Option<PrefillProgressSink<'_>> = match on_progress.as_deref_mut() {
            Some(inner) => {
                counted = |positions_done: usize, positions_total: usize| {
                    inner(GenerateProgress::PrefillChunk {
                        positions_done,
                        positions_total,
                    });
                };
                Some(&mut counted)
            }
            None => None,
        };
        prefill_prompt_with_progress(model, state, prompt_ids, sink, progress)?
    };
    let prefill = prefill_start.elapsed();

    let decode_start = Instant::now();
    let mut sampler = Sampler::new(params, logits.len());
    let mut stream = StreamDecoder::new();
    let mut position = start_position + prompt_ids.len();
    let mut generated_ids: Vec<u32> = Vec::with_capacity(params.max_new);
    let mut stop = StopReason::MaxNew;
    while generated_ids.len() < params.max_new {
        let next = sampler.sample(logits);
        if stop_tokens.contains(&next) {
            stop = StopReason::StopToken(next);
            break;
        }
        let text = stream.push(tokenizer, next)?;
        // Before `on_token`, and only for a token that was actually sampled:
        // the trailing flush below calls `on_token` a second time for an id
        // already reported, and giving that an index would put the event
        // stream one ahead of `GenerateStats::generated`. The counters are
        // read here rather than after the pass because a caller watching a
        // 128-token generation wants them per token, and `stream_stats_in`
        // is a cheap copy of already-summed fields.
        if let Some(progress) = on_progress.as_deref_mut() {
            progress(GenerateProgress::DecodeToken {
                index: generated_ids.len(),
                stats: state.stream_stats_in(StreamPhase::Decode),
            });
        }
        on_token(next, &text);
        generated_ids.push(next);
        if generated_ids.len() == params.max_new {
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
    if !tail.is_empty()
        && let Some(id) = stream.last_id()
    {
        on_token(id, &tail);
    }

    Ok(GenerateStats {
        prompt_tokens: prompt_ids.len(),
        generated: generated_ids.len(),
        generated_ids,
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
    use crate::model::{ForwardState, PrefillConfig, PrefillMode, RuntimeConfig, prefill_prompt};

    /// Fixture-sized runtime dials: an unpinned two-shard pool and a small
    /// expert budget, so a test suite that runs many states in parallel
    /// neither pins every thread to one core nor reserves the production
    /// 1,438 MiB per state.
    fn small(model: &Model, context_cap: usize) -> ForwardState {
        ForwardState::with_config(model, context_cap, RuntimeConfig::testing()).unwrap()
    }

    /// Sweep prefill with a ring narrow enough that the fixture's sub-1 MiB
    /// slot pool still leaves room for a chunk wider than any prompt here, so
    /// every test prompt is prefilled as exactly one chunk.
    fn one_chunk_sweep() -> PrefillConfig {
        PrefillConfig {
            mode: PrefillMode::Sweep,
            chunk: 512,
            experts_per_window: 1,
            windows_in_flight: 1,
        }
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

    /// An id whose decode is a lone U+FFFD, and stays one U+FFFD per id when
    /// repeated. Byte-level BPE has a single-byte token for every byte, so
    /// the byte alphabet (ids 0..256 in this vocabulary) holds stray
    /// continuation bytes; a run of them is a real id sequence that `trim`
    /// can never cut (every candidate head ends in U+FFFD) and `push` can
    /// never emit (every window decode is nothing but withheld replacement
    /// characters), so the window grows one id per push to the cap.
    fn undecodable_filler(tokenizer: &RvmpTokenizer) -> u32 {
        const FFFD: &str = "\u{FFFD}";
        (0u32..256)
            .find(|&id| {
                let one = tokenizer.decode(&[id], false).unwrap_or_default();
                let two = tokenizer.decode(&[id, id], false).unwrap_or_default();
                one == FFFD && two == FFFD.repeat(2)
            })
            .expect("the byte alphabet has a stray continuation byte")
    }

    /// Drive a decoder over `ids` without `drive_stream`'s "no push leaks a
    /// replacement character" rule: the sequences below really are
    /// undecodable in part, so replacement characters are the *correct*
    /// output and withholding them forever is not an option. Returns the
    /// assembled text and the window length after each push.
    fn drive_past_the_cap(tokenizer: &RvmpTokenizer, ids: &[u32]) -> (String, Vec<usize>) {
        let mut stream = StreamDecoder::new();
        let mut assembled = String::new();
        let mut lens = Vec::with_capacity(ids.len());
        for (i, &id) in ids.iter().enumerate() {
            assembled.push_str(&stream.push(tokenizer, id).unwrap());
            assert!(
                stream.window.len() <= STREAM_WINDOW_MAX,
                "window past the cap at push {i}: {}",
                stream.window.len()
            );
            lens.push(stream.window.len());
        }
        assembled.push_str(&stream.flush(tokenizer).unwrap());
        (assembled, lens)
    }

    /// The cap branch owes the same promise as every other path through the
    /// decoder: push-by-push output plus `flush` is `decode(all_ids, false)`,
    /// byte for byte. Nothing else in this module reaches
    /// [`STREAM_WINDOW_MAX`] — `drive_stream` only asserts the window stays
    /// under it, and the scaling test deliberately stays far below — so both
    /// sequences here are built to land exactly on it, from the two
    /// directions that can:
    ///
    /// 1. **Undecodable bytes, then a character split across ids.** 63 stray
    ///    continuation bytes grow the window to one short of the cap with
    ///    nothing emitted, and the 64th push is the *first* id of a multi-id
    ///    character, so the cap lands with that character half arrived. The
    ///    branch this replaced emitted the whole window verbatim — including
    ///    the half character's U+FFFD — and cleared, so the ids completing
    ///    the character then decoded to a second U+FFFD: two replacement
    ///    characters where `decode` yields the character itself.
    /// 2. **A token whose bytes are a rotation of a character's.** Repeated,
    ///    it puts *every* id boundary strictly inside a character, so no cut
    ///    is a clean split at all. The forced cut has to sever a character
    ///    and re-anchor across it, and the character still has to come out
    ///    whole and exactly once.
    #[test]
    fn stream_decoder_cap_matches_the_full_decode() {
        let tokenizer = fixture_tokenizer();

        let filler = undecodable_filler(&tokenizer);
        let split = ["\u{13000}", "\u{1D518}", "\u{10348}", "\u{1F30D}"]
            .iter()
            .map(|s| tokenizer.encode(s).unwrap())
            .find(|ids| ids.len() >= 2)
            .expect("some exotic character splits into multiple ids");
        let mut half_char = vec![filler; STREAM_WINDOW_MAX - 1];
        half_char.extend(split.iter().copied());

        // Bytes `92 E1 9E`, and `E1 9E 92` is U+17B2: the only token in the
        // pinned vocabulary whose own repetition never lands an id boundary
        // on a character boundary. Pinned by assertion, so a vocabulary
        // change fails loudly instead of quietly weakening the case.
        const ROTATION: u32 = 72_496;
        let once = tokenizer.decode(&[ROTATION], false).unwrap();
        let twice = tokenizer.decode(&[ROTATION, ROTATION], false).unwrap();
        assert_ne!(
            twice,
            once.repeat(2),
            "id {ROTATION} no longer straddles a character boundary"
        );
        let rotated = vec![ROTATION; STREAM_WINDOW_MAX + STREAM_WINDOW_KEEP];

        for (name, ids) in [
            ("split character at the cap", half_char),
            ("rotated token, no clean cut anywhere", rotated),
        ] {
            let (assembled, lens) = drive_past_the_cap(&tokenizer, &ids);
            assert_eq!(
                assembled,
                tokenizer.decode(&ids, false).unwrap(),
                "case {name}"
            );
            // And the cap really is what was exercised: the window grew to
            // one id short of it, and the next push dropped it straight back
            // to the keep bound instead of clearing or growing.
            assert_eq!(
                lens[STREAM_WINDOW_MAX - 2],
                STREAM_WINDOW_MAX - 1,
                "case {name}: window did not reach the cap"
            );
            assert_eq!(
                lens[STREAM_WINDOW_MAX - 1],
                STREAM_WINDOW_KEEP,
                "case {name}: forced cut did not land on the keep bound"
            );
        }
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
            0,
            &greedy_params(max_new),
            stops,
            &mut on_token,
            None,
            None,
        )
        .unwrap();
        // One event per generated token, plus at most one flush event that
        // re-reports the last id with the withheld tail text. That repeat is
        // exactly why `GenerateStats::generated_ids` exists: `on_token`'s ids
        // are a stream, not a list, and only the stats carry the sequence a
        // caller can append to a KV cache.
        assert!(events.len() == stats.generated || events.len() == stats.generated + 1);
        assert_eq!(stats.generated_ids.len(), stats.generated);
        let ids: Vec<u32> = events.iter().take(stats.generated).map(|e| e.0).collect();
        assert_eq!(
            ids, stats.generated_ids,
            "the stream and the stats disagree"
        );
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
        // One chunk for the whole prompt, so the layer-major emission order
        // asserted below is not at the mercy of how many rows the fixture's
        // sub-1 MiB slot pool happens to leave room for.
        state.set_prefill_config(one_chunk_sweep()).unwrap();
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

        // One record per layer per token, phases split at the prompt
        // boundary, and every routed id inside the expert count.
        let tokens = prompt.len() + stats.generated - 1;
        assert_eq!(records.len(), tokens * n_layers as usize);
        for (phase, pos, layer, experts) in &records {
            let want_phase = if *pos < prompt.len() {
                TracePhase::Prefill
            } else {
                TracePhase::Decode
            };
            assert_eq!(*phase, want_phase, "position {pos}");
            assert!(*layer < n_layers);
            assert_eq!(experts.len(), top_k);
            assert!(experts.iter().all(|&e| e < arch.n_experts), "{experts:?}");
            // torch.topk semantics: no expert is selected twice.
            let mut unique = experts.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), top_k, "{experts:?}");
        }

        // Every (position, layer) pair exactly once, across both phases.
        let mut pairs: Vec<(usize, u32)> = records.iter().map(|r| (r.1, r.2)).collect();
        pairs.sort_unstable();
        let want: Vec<(usize, u32)> = (0..tokens)
            .flat_map(|pos| (0..n_layers).map(move |layer| (pos, layer)))
            .collect();
        assert_eq!(pairs, want);

        // Emission *order* differs by phase, and that is the documented
        // contract on `TracePhase`: the sweep reports (layer, row) because
        // that is the order it does the work in, while decode — still one
        // `forward_token` per token — reports (token, layer).
        let decode: Vec<_> = records
            .iter()
            .filter(|r| r.0 == TracePhase::Decode)
            .collect();
        for (i, record) in decode.iter().enumerate() {
            assert_eq!(
                record.2 as usize,
                i % n_layers as usize,
                "decode record {i}"
            );
            assert_eq!(record.1, prompt.len() + i / n_layers as usize);
        }
        let prefill: Vec<_> = records
            .iter()
            .filter(|r| r.0 == TracePhase::Prefill)
            .collect();
        for (i, record) in prefill.iter().enumerate() {
            assert_eq!(
                record.2 as usize,
                i / prompt.len(),
                "prefill record {i} is not layer-major"
            );
            assert_eq!(record.1, i % prompt.len());
        }
    }

    /// F3. The streamer's counters are cumulative from construction, so a
    /// footer that quotes them reports prefill folded into what reads as a
    /// decode number. `generate` declares the phase to the streamer for every
    /// pass; this pins that the split is real, exhaustive, and lands exactly
    /// where the pass counts say it should — in **both** prefill modes, which
    /// now account for their expert bytes in different counters entirely.
    #[test]
    fn streaming_counters_split_prefill_from_decode() {
        let h = harness("gen-phase-split");
        let prompt = [1u32, 2, 3];
        let max_new = 4;
        let arch = h.model.arch();
        // One cache request per routed expert per layer per forward pass.
        let per_pass = u64::from(arch.n_layers) * u64::from(arch.top_k);

        let run = |config: PrefillConfig| {
            let mut state = small(&h.model, 16);
            state.set_prefill_config(config).unwrap();
            let stats = generate_with_stops(
                &h.model,
                &mut state,
                &h.tokenizer,
                &prompt,
                0,
                &greedy_params(max_new),
                &[],
                &mut |_, _| {},
                None,
                None,
            )
            .unwrap();
            assert_eq!(stats.generated, max_new);
            state
        };

        // Token-major prefill: one cache pass per prompt token, exactly as in
        // phase 5. Decode runs one pass per generated token *after* the
        // first, which is sampled from the prompt's logits.
        let state = run(PrefillConfig {
            mode: PrefillMode::TokenMajor,
            ..one_chunk_sweep()
        });
        let prefill = state.stream_stats_in(StreamPhase::Prefill);
        let decode = state.stream_stats_in(StreamPhase::Decode);
        let total = state.stream_stats();
        assert_eq!(prefill.accesses(), prompt.len() as u64 * per_pass);
        assert_eq!(decode.accesses(), (max_new as u64 - 1) * per_pass);
        // Exhaustive and disjoint: every request lands in exactly one phase.
        assert_eq!(prefill.accesses() + decode.accesses(), total.accesses());
        assert_eq!(prefill.misses + decode.misses, total.misses);
        assert_eq!(prefill.hits + decode.hits, total.hits);
        assert_eq!(prefill.bytes_read + decode.bytes_read, total.bytes_read);
        // The point of all of it: the cumulative figure is not the decode
        // figure, and quoting it as one overstates the work by the prompt.
        assert_ne!(total.accesses(), decode.accesses());

        // The sweep: prefill makes **no cache requests at all**. Its bytes
        // are in the sweep counters, which is the whole reason those exist
        // separately — a prefill hit rate is no longer a number that means
        // anything, and a decode one is finally clean of the prompt.
        let state = run(one_chunk_sweep());
        let prefill = state.stream_stats_in(StreamPhase::Prefill);
        let decode = state.stream_stats_in(StreamPhase::Decode);
        let total = state.stream_stats();
        assert_eq!(prefill.accesses(), 0, "the sweep bypasses the slot cache");
        assert_eq!(prefill.hits, 0);
        assert_eq!(prefill.misses, 0);
        assert_eq!(prefill.bytes_read, 0);
        assert!(prefill.sweep_windows_read > 0, "no window was ever read");
        assert!(prefill.sweep_bytes_read > 0, "no expert bytes were swept");
        assert_eq!(decode.accesses(), (max_new as u64 - 1) * per_pass);
        assert_eq!(total.accesses(), decode.accesses());
        assert_eq!(decode.sweep_windows_read, 0, "decode never sweeps");
    }

    /// A multi-turn conversation on one [`ForwardState`] must produce exactly
    /// what a fresh state fed the whole history produces. This is what makes
    /// a REPL affordable: a second turn prefills only its new tokens instead
    /// of the whole conversation, and reuses the ~1,438 MiB expert slot pool,
    /// the io_uring ring and the pinned compute pool rather than rebuilding
    /// them.
    ///
    /// It also pins the "which ids are new" rule from `generate_from`'s docs:
    /// the last sampled id is emitted but never fed, so the un-fed suffix is
    /// `history[state.seq_len()..]` and nothing else.
    #[test]
    fn a_second_turn_continues_an_existing_cache() {
        let h = harness("gen-continue");
        let params = greedy_params(3);
        let mut history: Vec<u32> = vec![1, 2, 3];

        let mut state = small(&h.model, 24);
        state.set_prefill_config(one_chunk_sweep()).unwrap();
        let turn1 = generate_with_stops(
            &h.model,
            &mut state,
            &h.tokenizer,
            &history.clone(),
            0,
            &params,
            &[],
            &mut |_, _| {},
            None,
            None,
        )
        .unwrap();
        assert_eq!(turn1.generated_ids.len(), 3);
        history.extend(&turn1.generated_ids);

        // The last generated id was emitted but never fed, so the cache is
        // one short of the history.
        let fed = state.seq_len().unwrap();
        assert_eq!(fed, history.len() - 1);

        // Turn two: the un-fed suffix, which is that trailing id plus the new
        // user tokens.
        history.extend([7u32, 8]);
        let segment = history[fed..].to_vec();
        let turn2 = generate_with_stops(
            &h.model,
            &mut state,
            &h.tokenizer,
            &segment,
            fed,
            &params,
            &[],
            &mut |_, _| {},
            None,
            None,
        )
        .unwrap();
        assert_eq!(turn2.prompt_tokens, segment.len());

        // The reference: a fresh state fed the whole history at once.
        let mut fresh = small(&h.model, 24);
        fresh.set_prefill_config(one_chunk_sweep()).unwrap();
        let want = generate_with_stops(
            &h.model,
            &mut fresh,
            &h.tokenizer,
            &history,
            0,
            &params,
            &[],
            &mut |_, _| {},
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            turn2.generated_ids, want.generated_ids,
            "a continued cache diverged from a rebuilt one"
        );
        assert_eq!(state.seq_len().unwrap(), fresh.seq_len().unwrap());
    }

    /// A stateless chat request resends the whole conversation, so the runtime
    /// keeps the KV cache up to the longest common prefix and re-prefills only
    /// the divergent suffix. [`ForwardState::truncate`] is that rewind, and
    /// this is the test that earns it: a cache cut back to `n` positions must
    /// be **indistinguishable** from one that only ever held `n`.
    ///
    /// The claim is proved at two levels, in both prefill modes:
    ///
    /// 1. **Bits.** The next forward pass off the rewound cache is
    ///    `to_bits()`-identical to the same pass off the cache that never
    ///    grew, with zero tolerance — the same shape as
    ///    `attention_at_is_bit_identical_to_truncated_decode`, one level up.
    /// 2. **Ids.** And it stays identical through a whole generation, which is
    ///    the level the server ships.
    ///
    /// The rewound state genuinely writes positions past the seam first — the
    /// `assert_ne!` on the two prefills' logits is what makes that non-vacuous
    /// — so a stale row surviving the truncate, or a position off by one,
    /// fails here rather than passing invisibly.
    #[test]
    fn a_truncated_cache_is_bit_identical_to_one_that_never_grew() {
        let h = harness("gen-truncate");
        let prefix = [1u32, 2, 3, 4];
        let divergent = [5u32, 6, 7];
        let next = 8u32;

        for mode in [PrefillMode::Sweep, PrefillMode::TokenMajor] {
            let config = PrefillConfig {
                mode,
                ..one_chunk_sweep()
            };

            // The state that overshoots: prefill the common prefix *and* a
            // suffix the next request will turn out not to share, so the
            // positions past the seam hold real f16 bits.
            let mut whole: Vec<u32> = prefix.to_vec();
            whole.extend(divergent);
            let mut rewound = small(&h.model, 24);
            rewound.set_prefill_config(config).unwrap();
            let long_logits = prefill_prompt(&h.model, &mut rewound, &whole, None)
                .unwrap()
                .to_vec();
            assert_eq!(rewound.seq_len().unwrap(), whole.len());

            // The reference: a state that only ever saw the prefix.
            let mut short = small(&h.model, 24);
            short.set_prefill_config(config).unwrap();
            let short_logits = prefill_prompt(&h.model, &mut short, &prefix, None)
                .unwrap()
                .to_vec();

            // Non-vacuity: the suffix has to have moved the computation, or
            // "identical after the rewind" would have been true before it too.
            assert_ne!(
                long_logits, short_logits,
                "{mode:?}: the divergent suffix changed nothing, so this test \
                 would pass with a truncate that did nothing"
            );

            rewound.truncate(prefix.len()).unwrap();
            assert_eq!(rewound.seq_len().unwrap(), prefix.len());
            assert_eq!(rewound.seq_len().unwrap(), short.seq_len().unwrap());

            // Level 1, the proof.
            let got = forward_token(&h.model, &mut rewound, next, prefix.len(), true)
                .unwrap()
                .expect("logits were requested");
            let got = got.to_vec();
            let want = forward_token(&h.model, &mut short, next, prefix.len(), true)
                .unwrap()
                .expect("logits were requested");
            let want = want.to_vec();
            assert_eq!(got.len(), want.len());
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{mode:?} logit {i}: off a truncated cache {g:e} vs off one \
                     that never grew {w:e}"
                );
            }

            // Level 2: the same through a whole turn, sampled ids included.
            let params = greedy_params(3);
            let segment = [11u32, 12];
            let at = prefix.len() + 1;
            let turn = |state: &mut ForwardState| {
                generate_with_stops(
                    &h.model,
                    state,
                    &h.tokenizer,
                    &segment,
                    at,
                    &params,
                    &[],
                    &mut |_, _| {},
                    None,
                    None,
                )
                .unwrap()
            };
            let continued = turn(&mut rewound);
            let reference = turn(&mut short);
            assert_eq!(
                continued.generated_ids, reference.generated_ids,
                "{mode:?}: a rewound cache diverged from one that never grew"
            );
            assert_eq!(rewound.seq_len().unwrap(), short.seq_len().unwrap());
        }
    }

    /// The rewind is refused, not fudged: past the cached length there is
    /// nothing to keep, only stale bits from a longer sequence. It comes back
    /// typed through [`ForwardError::Kv`], the cursor does not move, and the
    /// state is still usable afterwards. (The other refusal, a ragged cache,
    /// has no public way to be produced here and is pinned at the cache level
    /// by `truncate_refuses_ragged_and_upward`.)
    #[test]
    fn truncate_refuses_what_it_cannot_honour() {
        let h = harness("gen-truncate-refuse");
        let mut state = small(&h.model, 16);
        state.set_prefill_config(one_chunk_sweep()).unwrap();
        prefill_prompt(&h.model, &mut state, &[1u32, 2, 3], None).unwrap();

        let err = state.truncate(4).unwrap_err();
        assert!(
            matches!(
                err,
                ForwardError::Kv(crate::kv::KvError::TruncateBeyondLength {
                    requested: 4,
                    len: 3
                })
            ),
            "unexpected error: {err}"
        );
        assert_eq!(state.seq_len().unwrap(), 3, "a refused truncate rewound");

        // And the cache still works: truncating to a real prefix succeeds and
        // the next token is accepted at the seam.
        state.truncate(2).unwrap();
        assert_eq!(state.seq_len().unwrap(), 2);
        forward_token(&h.model, &mut state, 5, 2, true).unwrap();
        assert_eq!(state.seq_len().unwrap(), 3);
    }

    /// `start_position` is validated, not trusted: an off-by-one would prefill
    /// the new turn at the wrong RoPE positions and produce plausible
    /// nonsense.
    #[test]
    fn a_wrong_start_position_is_typed() {
        let h = harness("gen-start-position");
        let mut state = small(&h.model, 16);
        state.set_prefill_config(one_chunk_sweep()).unwrap();

        // A fresh state is at 0, so anything else is refused, both through
        // the public wrapper and before any work happens.
        let err = generate_from(
            &h.model,
            &mut state,
            &h.tokenizer,
            &[1, 2],
            1,
            &greedy_params(1),
            |_, _| {},
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                GenerateError::PositionMismatch {
                    position: 1,
                    expected: 0
                }
            ),
            "unexpected error: {err}"
        );
        assert_eq!(state.seq_len().unwrap(), 0, "nothing was consumed");

        // And `generate` is `generate_from(.., 0, ..)`, so a state that is
        // *not* fresh is refused rather than silently mis-positioned.
        generate_from(
            &h.model,
            &mut state,
            &h.tokenizer,
            &[1, 2],
            0,
            &greedy_params(2),
            |_, _| {},
        )
        .unwrap();
        let held = state.seq_len().unwrap();
        assert!(held > 0);
        let err = generate(
            &h.model,
            &mut state,
            &h.tokenizer,
            &[3],
            &greedy_params(1),
            |_, _| {},
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                GenerateError::PositionMismatch {
                    position: 0,
                    expected,
                } if expected == held
            ),
            "unexpected error: {err}"
        );
    }

    /// `reset` starts a fresh sequence on an existing state, which must be
    /// indistinguishable from a brand-new one — without paying for a new
    /// expert slot pool.
    #[test]
    fn reset_starts_a_fresh_sequence() {
        let h = harness("gen-reset");
        let prompt = [4u32, 5, 6];
        let params = greedy_params(3);

        let mut reused = small(&h.model, 16);
        reused.set_prefill_config(one_chunk_sweep()).unwrap();
        let first = generate_with_stops(
            &h.model,
            &mut reused,
            &h.tokenizer,
            &prompt,
            0,
            &params,
            &[],
            &mut |_, _| {},
            None,
            None,
        )
        .unwrap();
        assert!(reused.seq_len().unwrap() > 0);

        reused.reset();
        assert_eq!(reused.seq_len().unwrap(), 0);
        for layer in 0..h.model.n_layers() as usize {
            assert_eq!(reused.kv_len(layer).unwrap(), 0);
        }
        let second = generate_with_stops(
            &h.model,
            &mut reused,
            &h.tokenizer,
            &prompt,
            0,
            &params,
            &[],
            &mut |_, _| {},
            None,
            None,
        )
        .unwrap();
        assert_eq!(second.generated_ids, first.generated_ids);
    }

    // --- Progress reporting ---

    /// One instrumented run: every progress event, every `on_token` call, and
    /// the stats, so a test can hold the three against each other.
    fn run_with_progress(
        h: &Harness,
        config: PrefillConfig,
        prompt: &[u32],
        max_new: usize,
    ) -> (Vec<GenerateProgress>, Vec<(u32, String)>, GenerateStats) {
        let mut state = small(&h.model, 24);
        state.set_prefill_config(config).unwrap();
        let mut progress: Vec<GenerateProgress> = Vec::new();
        let mut tokens: Vec<(u32, String)> = Vec::new();
        let stats = {
            let mut on_progress = |event: GenerateProgress| progress.push(event);
            generate_from_with_progress(
                &h.model,
                &mut state,
                &h.tokenizer,
                prompt,
                0,
                &greedy_params(max_new),
                Some(&mut on_progress),
                |id, piece| tokens.push((id, piece.to_owned())),
            )
            .unwrap()
        };
        (progress, tokens, stats)
    }

    fn prefill_events(events: &[GenerateProgress]) -> Vec<(usize, usize)> {
        events
            .iter()
            .filter_map(|event| match *event {
                GenerateProgress::PrefillChunk {
                    positions_done,
                    positions_total,
                } => Some((positions_done, positions_total)),
                GenerateProgress::DecodeToken { .. } => None,
            })
            .collect()
    }

    fn decode_events(events: &[GenerateProgress]) -> Vec<usize> {
        events
            .iter()
            .filter_map(|event| match *event {
                GenerateProgress::DecodeToken { index, .. } => Some(index),
                GenerateProgress::PrefillChunk { .. } => None,
            })
            .collect()
    }

    /// The same run without any instrumentation, through the public
    /// `generate_from`. The reference every "the callback changes nothing"
    /// assertion is made against.
    fn run_plain(
        h: &Harness,
        config: PrefillConfig,
        prompt: &[u32],
        max_new: usize,
    ) -> (Vec<(u32, String)>, GenerateStats) {
        let mut state = small(&h.model, 24);
        state.set_prefill_config(config).unwrap();
        let mut tokens: Vec<(u32, String)> = Vec::new();
        let stats = generate_from(
            &h.model,
            &mut state,
            &h.tokenizer,
            prompt,
            0,
            &greedy_params(max_new),
            |id, piece| tokens.push((id, piece.to_owned())),
        )
        .unwrap();
        (tokens, stats)
    }

    /// A progress bar's whole contract: one event per chunk, counted from
    /// this call's start, strictly increasing, landing exactly on the prompt
    /// length. The chunk *width* is the sweep's business — it is narrowed to
    /// whatever the expert slot slab can host — so the counts below are
    /// pinned at widths the fixture's pool comfortably fits, and the shape
    /// properties are asserted for every width.
    #[test]
    fn prefill_progress_counts_every_chunk_to_the_total() {
        let h = harness("gen-progress-prefill");
        let prompt = [1u32, 2, 3, 4, 5, 6];
        let n = prompt.len();

        // Sweep, per chunk width. `chunk: 1` is the interesting one: it is
        // the only width where "one event per chunk" and "one event per
        // token" differ from each other and from a single event.
        for (chunk, want) in [
            (1usize, vec![1usize, 2, 3, 4, 5, 6]),
            (2, vec![2, 4, 6]),
            (3, vec![3, 6]),
            (512, vec![6]),
        ] {
            let config = PrefillConfig {
                chunk,
                ..one_chunk_sweep()
            };
            let (events, _, stats) = run_with_progress(&h, config, &prompt, 2);
            let prefill = prefill_events(&events);
            assert_eq!(
                prefill.iter().map(|e| e.0).collect::<Vec<_>>(),
                want,
                "chunk {chunk}"
            );
            assert!(
                prefill.iter().all(|e| e.1 == n),
                "chunk {chunk}: total is not the prompt length: {prefill:?}"
            );
            assert_eq!(stats.prompt_tokens, n);
        }

        // Token-major has no chunks, so its unit is the token. Same
        // guarantees, finer granularity — documented on `PrefillProgressSink`
        // rather than papered over with a synthetic chunk width.
        let (events, _, _) = run_with_progress(
            &h,
            PrefillConfig {
                mode: PrefillMode::TokenMajor,
                ..one_chunk_sweep()
            },
            &prompt,
            2,
        );
        assert_eq!(
            prefill_events(&events),
            (1..=n).map(|done| (done, n)).collect::<Vec<_>>()
        );

        // The shape properties, for every configuration above: strictly
        // increasing, never past the total, and the last event is the total.
        for config in [
            PrefillConfig {
                chunk: 1,
                ..one_chunk_sweep()
            },
            PrefillConfig {
                chunk: 4,
                ..one_chunk_sweep()
            },
            one_chunk_sweep(),
            PrefillConfig {
                mode: PrefillMode::TokenMajor,
                ..one_chunk_sweep()
            },
        ] {
            let (events, _, _) = run_with_progress(&h, config, &prompt, 2);
            let prefill = prefill_events(&events);
            assert!(!prefill.is_empty(), "{config:?} reported nothing");
            let mut previous = 0usize;
            for &(done, total) in &prefill {
                assert!(done > previous, "{config:?} went backwards: {prefill:?}");
                assert!(done <= total, "{config:?} overshot: {prefill:?}");
                previous = done;
            }
            assert_eq!(prefill.last().unwrap(), &(n, n), "{config:?}");
        }
    }

    /// The trap this callback had to be threaded around: `on_token` fires one
    /// **extra** trailing time to flush a partial UTF-8 character, repeating
    /// an id that was already reported. A `DecodeToken` there would put the
    /// event stream one ahead of [`GenerateStats::generated`] and give a UI a
    /// token count its own stats contradict, so the flush reports nothing.
    ///
    /// Both halves are exercised: a run whose output is plain ASCII (no
    /// flush) and one that ends mid-character (flush), the second on a
    /// byte-alphabet fixture because a 32-id vocabulary of ASCII punctuation
    /// can never reach the branch at all.
    #[test]
    fn decode_progress_fires_once_per_token_and_not_for_the_flush() {
        let h = harness("gen-progress-decode");
        let (events, tokens, stats) = run_with_progress(&h, one_chunk_sweep(), &[1, 2, 3], 5);
        assert_eq!(stats.generated, 5);
        assert_eq!(
            decode_events(&events),
            (0..stats.generated).collect::<Vec<_>>()
        );
        assert_eq!(
            tokens.len(),
            stats.generated,
            "this fixture's vocabulary is ASCII punctuation; nothing can be withheld"
        );

        // Every event precedes its own `on_token` call, and the last prefill
        // event precedes every decode event: the ordering a UI relies on to
        // switch from a progress bar to a token stream.
        let first_decode = events
            .iter()
            .position(|e| matches!(e, GenerateProgress::DecodeToken { .. }))
            .expect("decode reported");
        assert!(
            events[..first_decode]
                .iter()
                .all(|e| matches!(e, GenerateProgress::PrefillChunk { .. })),
            "a decode event arrived before prefill finished"
        );

        // The flush case. Ids 106..=255 of the pinned vocabulary are raw
        // bytes 0xAE..=0xFF, so a greedy run that ends on one leaves the
        // stream decoder holding an incomplete character.
        let w = byte_harness("gen-progress-flush");
        let (events, tokens, stats) = run_with_progress(&w, one_chunk_sweep(), &[3, 15], 3);
        assert_eq!(
            stats.generated_ids,
            vec![134, 134, 134],
            "the fixture's greedy output moved; pick a new prompt that ends mid-character"
        );
        assert_eq!(
            tokens.len(),
            stats.generated + 1,
            "this run was supposed to reach the trailing flush"
        );
        assert_eq!(
            tokens.last().unwrap().0,
            *stats.generated_ids.last().unwrap(),
            "the flush repeats the last id rather than reporting a new one"
        );
        // The point: one event per *sampled* token, flush or no flush.
        assert_eq!(decode_events(&events), vec![0, 1, 2]);
    }

    /// The one that matters. `generate_from` is `generate_from_with_progress`
    /// with `None`, and an installed callback observes without perturbing:
    /// same ids, same text, byte for byte, in both prefill modes and on the
    /// fixture whose output reaches the trailing flush. Instrumentation that
    /// moves a number is not instrumentation.
    #[test]
    fn progress_does_not_perturb_generation() {
        for tag in ["gen-progress-noop", "gen-progress-noop-wide"] {
            let wide = tag.ends_with("-wide");
            let h = if wide {
                byte_harness(tag)
            } else {
                harness(tag)
            };
            let prompt: &[u32] = if wide { &[3, 15] } else { &[1, 2, 3] };
            for config in [
                one_chunk_sweep(),
                PrefillConfig {
                    chunk: 2,
                    ..one_chunk_sweep()
                },
                PrefillConfig {
                    mode: PrefillMode::TokenMajor,
                    ..one_chunk_sweep()
                },
            ] {
                let (want_tokens, want) = run_plain(&h, config, prompt, 4);
                let (events, tokens, stats) = run_with_progress(&h, config, prompt, 4);

                assert_eq!(
                    tokens, want_tokens,
                    "{tag} {config:?}: the token stream moved"
                );
                assert_eq!(stats.generated_ids, want.generated_ids, "{tag} {config:?}");
                assert_eq!(stats.generated, want.generated, "{tag} {config:?}");
                assert_eq!(stats.prompt_tokens, want.prompt_tokens, "{tag} {config:?}");
                assert_eq!(stats.stop, want.stop, "{tag} {config:?}");

                // And the observer really did observe, so the equality above
                // is not the equality of two uninstrumented runs.
                assert!(!prefill_events(&events).is_empty(), "{tag} {config:?}");
                assert_eq!(
                    decode_events(&events).len(),
                    want.generated,
                    "{tag} {config:?}"
                );
            }
        }
    }

    /// A fixture whose vocabulary spans the tokenizer's whole byte alphabet,
    /// so greedy decode can land on a raw byte that is not a character on its
    /// own — the only way to reach the trailing-flush branch with a synthetic
    /// model.
    fn byte_harness(tag: &str) -> Harness {
        let fx = crate::io::testutil::build_install_with(
            tag,
            &crate::io::testutil::Geometry {
                vocab: 256,
                ..crate::io::testutil::NARROW
            },
        );
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
