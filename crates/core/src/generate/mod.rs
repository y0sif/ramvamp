//! Generation orchestration: chunked prefill, token-by-token decode, sampling.
//!
//! Prefill processes the prompt in bounded chunks (so one fetched expert
//! serves many rows and scratch memory stays fixed) and is layer-major.
//! Decode repeats the routed layer loop one token at a time. Sampling
//! supports greedy, temperature, top-k, top-p, and repetition penalty;
//! greedy decode must be deterministic for validation against reference
//! implementations.
//!
//! # What exists today
//!
//! [`generate`] is the token-at-a-time baseline: prefill is
//! [`forward_token`] in a loop (logits requested only for the last prompt
//! token — the layer-major chunked sweep replaces this later), then the
//! decode loop samples, streams, and stops on a stop token or
//! [`GenerateParams::max_new`].
//!
//! ## Sampling
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
//! ## Streaming detokenization
//!
//! Decoding each token id alone is wrong for byte-level BPE (one Unicode
//! character can span tokens), so [`generate`] decodes the accumulated ids
//! and emits the new suffix, withholding any trailing U+FFFD replacement
//! characters until the bytes that complete them arrive (they are flushed
//! verbatim at end of generation if the model stops mid-character).

use std::time::{Duration, Instant};

use thiserror::Error;

use crate::model::{ForwardError, ForwardState, Model, forward_token};
use crate::tokenizer::{RvmpTokenizer, SamplingDefaults, TokenizerError};

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

/// Incremental detokenizer: accumulates generated ids, decodes the whole
/// sequence each push, and returns only the newly-safe suffix. "Safe"
/// excludes trailing U+FFFD replacement characters, which mark a Unicode
/// character whose UTF-8 bytes are still split across future tokens
/// (byte-level BPE decode is byte-prefix-stable except for that tail).
#[derive(Debug, Default)]
struct StreamDecoder {
    ids: Vec<u32>,
    /// Bytes of the accumulated decode already emitted.
    emitted: usize,
}

impl StreamDecoder {
    fn new() -> Self {
        Self::default()
    }

    /// Append one id and return the newly emittable text (possibly empty).
    fn push(&mut self, tokenizer: &RvmpTokenizer, id: u32) -> Result<String, TokenizerError> {
        self.ids.push(id);
        let full = tokenizer.decode(&self.ids, false)?;
        let mut safe = full.len();
        while full[..safe].ends_with('\u{FFFD}') {
            safe -= '\u{FFFD}'.len_utf8();
        }
        if safe <= self.emitted {
            return Ok(String::new());
        }
        // In range by construction; `get` keeps a hostile tokenizer from
        // panicking us if decode were ever not prefix-stable.
        let out = full.get(self.emitted..safe).unwrap_or("").to_owned();
        self.emitted = safe;
        Ok(out)
    }

    /// Everything still withheld (a trailing incomplete character, decoded
    /// with replacement characters), emptying the decoder's debt.
    fn flush(&mut self, tokenizer: &RvmpTokenizer) -> Result<String, TokenizerError> {
        if self.ids.is_empty() {
            return Ok(String::new());
        }
        let full = tokenizer.decode(&self.ids, false)?;
        let out = full.get(self.emitted..).unwrap_or("").to_owned();
        self.emitted = full.len();
        Ok(out)
    }

    fn last_id(&self) -> Option<u32> {
        self.ids.last().copied()
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
    )
}

/// [`generate`] with an explicit stop set (unit tests drive this with
/// synthetic stop tokens a tiny fixture model can actually emit).
fn generate_with_stops(
    model: &Model,
    state: &mut ForwardState,
    tokenizer: &RvmpTokenizer,
    prompt_ids: &[u32],
    params: &GenerateParams,
    stop_tokens: &[u32],
    on_token: &mut dyn FnMut(u32, &str),
) -> Result<GenerateStats, GenerateError> {
    let (&last, rest) = prompt_ids.split_last().ok_or(GenerateError::EmptyPrompt)?;

    let prefill_start = Instant::now();
    for (pos, &id) in rest.iter().enumerate() {
        forward_token(model, state, id, pos, false)?;
    }
    let mut logits =
        forward_token(model, state, last, rest.len(), true)?.ok_or(GenerateError::MissingLogits)?;
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
        logits = forward_token(model, state, next, position, true)?
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
    use crate::model::ForwardState;

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
        let model = Model::load(&fx.root, LoadOptions { skip_hashes: true }).unwrap();
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
        let mut state = ForwardState::new(&h.model, 16).unwrap();
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
    fn empty_prompt_is_a_typed_error() {
        let h = harness("gen-empty");
        let mut state = ForwardState::new(&h.model, 16).unwrap();
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
        let mut state = ForwardState::new(&h.model, 16).unwrap();
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
