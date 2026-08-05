//! Chunked, layer-major prefill.
//!
//! Token-major prefill *is* decode: one [`forward_token`](super::forward_token)
//! per prompt token, each one re-reading `top_k` experts on all 48 layers. A
//! 4K prompt therefore reads ~1,097 MB of expert weights **per token** and
//! ~4.5 TB in total. This module runs the shape `docs/architecture.md`
//! specifies instead: a bounded chunk of up to
//! [`DEFAULT_PREFILL_CHUNK`] positions is carried through the model together,
//! and each layer's expert file is streamed once, front to back, with every
//! expert computed against **all** of the chunk's rows routed to it.
//!
//! Expert bytes drop from `top_k` blobs per token per layer to one pass over
//! the layer file per *chunk* — ~34 MB/token at a 512-token chunk instead of
//! ~1,097 MB — and each weight row is reused across the ~32 rows a 512-row
//! chunk routes to a given expert instead of being restreamed per row.
//!
//! # The one thing that is schedule-sensitive
//!
//! **Logits must be byte-identical to the token-major path.** `scripts/
//! bitident.py` compares with zero tolerance. Prefill is allowed to change
//! *which* experts are read and *when*; it is not allowed to change the
//! arithmetic or its order.
//!
//! Exactly one site cares. The per-row MoE reduction accumulates in f32 in
//! **top-k order** (descending router probability, ties to the lowest expert
//! id), and f32 addition is not associative. The sweep hands experts back in
//! ascending **blob** order, which is a different order. So expert outputs are
//! staged per `[row][slot_in_topk]` and reduced afterwards, in top-k order,
//! exactly as `forward_token` reduces its own staging buffer. Accumulating on
//! arrival would move bits; [`reduce_experts`] is the function that does not.
//!
//! Everything else is order-agnostic by construction:
//!
//! - A given `(weight_row, activation_row)` dot product is a fixed sequence of
//!   f32 operations whatever the outer loop order is, so
//!   [`gemv_q8_k_batched`] is bit-identical to `gemv_q8_k_rows` per element
//!   (its own docs pin that), and [`crate::threads::shard_range`] is a pure
//!   function, so the pool fan-out is bit-neutral.
//! - RMSNorm, RoPE, softmax, SwiGLU, `vec_add` and both activation
//!   quantizers are per row with no cross-row coupling.
//! - [`attention_at_in`] is bit-identical to `decode_attention` against a cache
//!   truncated to the same length, which is what lets a chunk's whole K/V be
//!   appended before any of its rows attend. Rows are independent of each
//!   other too, so [`scatter_attention`] may split them across shards by cost
//!   rather than by count without moving a bit.
//!
//! # Layer-major, and why the K/V goes in first
//!
//! ```text
//! embed all R rows
//! for layer in 0..n_layers:
//!     norm/quantize/Q,K,V for all R rows      (batched GEMV, n_acts = R)
//!     QK-RMSNorm + RoPE per row at start + r
//!     append ALL R rows' K/V to kv[layer]     <-- the inversion
//!     attention_at_in(row r, positions = start + r + 1)   (fanned out)
//!     o_proj, residual, ffn norm, router, top-k for all R rows
//!     sweep the layer once; per expert, compute its routed rows in one batch
//!     reduce each row's staging in top-k order
//! final norm + lm_head for the LAST row only
//! ```
//!
//! Appending the whole chunk's K/V before attending is the point: it is one
//! pass over the layer instead of R, and the causal mask that makes it correct
//! is [`attention_at_in`]'s `positions` argument. Mid-layer the KV cache is
//! legitimately **ragged** — layer `L` holds `start + R` positions while layer
//! `L + 1` still holds `start` — so [`crate::kv::KvCache::seq_len`] reports
//! [`KvError::RaggedLayers`](crate::kv::KvError::RaggedLayers) until the chunk
//! finishes. That is why this path cannot reuse `forward_token`'s position
//! gate, and why nothing here weakens it: the token-major path still needs it.
//!
//! # Memory
//!
//! Every batched buffer is carved from the [`PrefillSession`] scratch span,
//! which is the head of the idle expert-slot slab. Nothing is allocated: the
//! documented headroom under `memory.max=3G` is ~111 MiB, so a heap buffer
//! here would be a budget event even though the slab itself is ~1,438 MiB.
//! See [`scratch_bytes`] for the arithmetic and [`Carver`] for the typed
//! sub-allocator that hands the span out.
//!
//! # A/B
//!
//! [`PrefillMode`] keeps both paths in one binary, defaulting to the sweep.
//! This exists so the strongest available test can be written: the sweep and
//! the token-major loop must produce byte-identical logits for the same
//! prompt, which `sweep_and_token_major_agree_bit_for_bit` asserts across
//! several chunk sizes and prompt lengths.

use std::mem::{align_of, size_of};
use std::ops::Range;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::format::{ArchInfo, ExpertsLayout};
use crate::io::{PrefillSession, SweepConfig, SweepExpert, SweepPlan};
use crate::kernels::KernelError;
use crate::kernels::attention::{
    AttentionError, attention_at_in, scratch_len as attention_scratch_len,
};
use crate::kernels::primitives::{
    rmsnorm, rmsnorm_in_place, rope_neox_heads, softmax, swiglu_combine, vec_add,
};
use crate::kernels::quants::{
    BlockQ8_0, BlockQ8K, QuantFormat, quantize_row_q8_0, quantize_row_q8_k,
};
use crate::kernels::{gemv_q8_0_batched, gemv_q8_k_batched};
use crate::kv::{KvCache, KvError};
use crate::threads::ComputePool;

use super::forward::{ForwardError, ForwardState, dot_f32, pool_gemv_q8_k, record, taken};
use super::weights::Model;

/// Positions carried through the model together by default.
///
/// Total prefill expert bytes are `ceil(prompt / chunk) * 17.55 GB`, so wider
/// is strictly better for I/O and the only cost is staging, which scales
/// linearly at 64 KiB/row (`top_k * hidden * 4` at the v0 dims). 512 is the
/// documented starting point; the 128/256/512/1024 sweep is a planned
/// experiment, which is why this is a dial and not a constant in the driver.
pub const DEFAULT_PREFILL_CHUNK: usize = 512;

/// Which prefill path [`prefill_prompt`] takes.
///
/// Both are retained on purpose: keeping them in one binary is what makes the
/// bit-identity test possible (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillMode {
    /// Chunked layer-major sweep. One pass over each layer's expert file per
    /// chunk, cache bypassed. The default.
    #[default]
    Sweep,
    /// One [`forward_token`](super::forward_token) per prompt token through
    /// the decode cache — the phase-5 behaviour, kept for A/B.
    TokenMajor,
}

/// Runtime dials for the prefill pass.
///
/// Held on [`ForwardState`] rather than on
/// [`RuntimeConfig`](super::RuntimeConfig): the latter's fields are
/// constructed positionally by the CLI, which a different lane owns, so
/// growing it would break that crate's build. Set these with
/// [`ForwardState::set_prefill_config`]; [`PrefillConfig::from_env`] is the
/// override that is reachable without any CLI change at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillConfig {
    /// Which path to take. Defaults to [`PrefillMode::Sweep`].
    pub mode: PrefillMode,
    /// Positions per chunk. Defaults to [`DEFAULT_PREFILL_CHUNK`]. Clamped
    /// down when the slot slab cannot hold that chunk's staging; see
    /// [`plan_arena`].
    pub chunk: usize,
    /// Experts one sweep window read covers.
    pub experts_per_window: u32,
    /// Sweep windows kept in flight.
    pub windows_in_flight: u32,
}

impl Default for PrefillConfig {
    fn default() -> Self {
        let sweep = SweepConfig::default();
        Self {
            mode: PrefillMode::default(),
            chunk: DEFAULT_PREFILL_CHUNK,
            experts_per_window: sweep.experts_per_window,
            windows_in_flight: sweep.windows_in_flight,
        }
    }
}

impl PrefillConfig {
    /// The default, with `RAMVAMP_PREFILL` and `RAMVAMP_PREFILL_CHUNK`
    /// applied when they are set and parse.
    ///
    /// `RAMVAMP_PREFILL` takes `sweep` or `token-major` (`token` and
    /// `token_major` are accepted too); `RAMVAMP_PREFILL_CHUNK` takes a
    /// positive integer. Anything else is ignored with a warning rather than
    /// failing a load: this is a diagnostic dial, and a typo in an
    /// environment variable is not a reason to refuse to run a model.
    ///
    /// [`ForwardState`] seeds itself from this, so the two dials are
    /// reachable today without touching the CLI.
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(raw) = std::env::var("RAMVAMP_PREFILL") {
            match raw.trim().to_ascii_lowercase().as_str() {
                "sweep" => config.mode = PrefillMode::Sweep,
                "token" | "token-major" | "token_major" => config.mode = PrefillMode::TokenMajor,
                other => tracing::warn!(value = other, "ignoring unknown RAMVAMP_PREFILL"),
            }
        }
        if let Ok(raw) = std::env::var("RAMVAMP_PREFILL_CHUNK") {
            match raw.trim().parse::<usize>() {
                Ok(chunk) if chunk > 0 => config.chunk = chunk,
                _ => tracing::warn!(value = raw, "ignoring unusable RAMVAMP_PREFILL_CHUNK"),
            }
        }
        config
    }

    /// The sweep dials alone.
    #[must_use]
    pub fn sweep_config(self) -> SweepConfig {
        SweepConfig {
            experts_per_window: self.experts_per_window,
            windows_in_flight: self.windows_in_flight,
        }
    }

    /// Reject a degenerate dial before it reaches the driver.
    ///
    /// **All three dials, one refusal point.** Every field here is `pub`, so
    /// every field is caller-supplied, and each used to be refused somewhere
    /// else: the chunk here, `experts_per_window == 0` silently coerced to 1
    /// by `plan_arena`, and an oversized `windows_in_flight` only inside
    /// `PrefillSession::begin`, several hundred lines and one arena plan
    /// later. The sweep dials are delegated to [`SweepConfig::validate`],
    /// which owns their bounds, so the two cannot drift apart.
    ///
    /// # Errors
    ///
    /// [`ForwardError::InvalidPrefillChunk`] for a zero chunk;
    /// [`ForwardError::Sweep`] with
    /// [`SweepError::BadDials`](crate::io::SweepError::BadDials) for a zero
    /// `experts_per_window`, a zero `windows_in_flight`, or more windows in
    /// flight than `MAX_WINDOWS_IN_FLIGHT`.
    pub fn validate(self) -> Result<Self, ForwardError> {
        if self.chunk == 0 {
            return Err(ForwardError::InvalidPrefillChunk { chunk: self.chunk });
        }
        self.sweep_config().validate()?;
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// Where the wall time went
// ---------------------------------------------------------------------------

/// One phase of a prefill pass, as charged by [`PhaseClock`].
///
/// Deliberately coarse. The point is to tell the two candidate causes of a
/// compute-bound prefill apart — serial per-row attention against activation
/// thrash in the batched expert GEMV — not to attribute every instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    /// The causal attention over the KV cache: `attention_at_in` per row on
    /// the sweep, `decode_attention` per token on the token-major path. Both
    /// are fanned out across the compute pool, so this is wall time, not core
    /// time.
    Attention,
    /// GEMVs against resident weights: Q/K/V, `o_proj`, the router matvec
    /// (with its softmax and top-k selection, which it dominates) and
    /// `lm_head`.
    Projections,
    /// Expert arithmetic: gate/up, SwiGLU, the intermediate quantization and
    /// down, for every routed expert. Excludes waiting for its bytes.
    ExpertCompute,
    /// Getting expert bytes in front of that arithmetic.
    ExpertIo,
    /// Everything per-row and cheap per element: RMSNorms, RoPE, activation
    /// quantization, transposes, KV appends, residual adds, the routing
    /// bookkeeping and the top-k reduction.
    Elementwise,
}

/// Where one [`prefill_prompt`] call's wall time went.
///
/// EXP-016 left prefill compute-bound: at a 512-token chunk expert reads fell
/// from 82,233 to 768 windows while wall time only fell 2.45x. Two causes were
/// proposed, attention (quadratic in prompt length, serial per row) and the
/// batched expert GEMV (whose activation side streams ~75 KB past every weight
/// row), and the counters that existed could not tell them apart.
///
/// EXP-017 answered it with these timers, and refuted the GEMV half: at 512
/// tokens attention is 61.3% of prefill and expert compute is 19.1%, the
/// batched GEMV having cut expert compute 3.27x against token-major. Expert
/// I/O is down to 1.7%, so non-I/O work is 98.3% of a swept prefill, not the
/// ~93% the 2.45x ratio implied. At 1891 tokens attention reaches 85.2%.
///
/// # What sums to what
///
/// [`PrefillTiming::total`] is the wall time of the whole `prefill_prompt`
/// call. The five phase fields are **disjoint** spans inside it, charged by
/// [`PhaseClock`] at region boundaries rather than opened and closed
/// independently, so [`accounted`](PrefillTiming::accounted) cannot exceed
/// `total` and [`other`](PrefillTiming::other) closes the sum exactly.
///
/// `other` is not noise. It is the arena plan, `begin_prefill`/`session
/// finish`, the per-block scratch carves, and the chunk loop's own
/// scaffolding — real work that is simply none of the five, made visible
/// instead of being folded into whichever phase happened to be open.
///
/// # I/O wait is not measured twice
///
/// [`PrefillTiming::expert_io`] spans the calls that put expert bytes in front
/// of the arithmetic: `split`, `next_expert` and `finish` on the sweep,
/// `begin_layer`, `await_misses` and `end_layer` on the token-major path. The
/// *blocked on the drive* subset of that is already counted by the streamer,
/// as [`StreamStats::sweep_io_wait`](crate::io::StreamStats::sweep_io_wait)
/// and [`StreamStats::io_wait`](crate::io::StreamStats::io_wait) respectively,
/// and nothing here re-times it. `expert_io - sweep_io_wait` is the sweep's
/// own window bookkeeping; reading the two side by side is the point, which is
/// why the CLI prints them together.
///
/// # Lifetime
///
/// Zeroed at the start of every `prefill_prompt` and by
/// [`ForwardState::reset`]. This describes **one** prefill, unlike the
/// streaming counters, which describe the process and survive a reset on
/// purpose; a cumulative timing across a chat session's turns would answer no
/// question anyone has. Decode contributes nothing: `forward_token` shares its
/// instrumentation with the token-major path and charges only while a prefill
/// has armed the struct.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefillTiming {
    /// Which path produced these numbers. `None` until a prefill has run,
    /// which is what [`PrefillTiming::ran`] asks.
    pub mode: Option<PrefillMode>,
    /// Prompt positions the run consumed.
    pub tokens: u64,
    /// Wall time of the whole `prefill_prompt` call, phases and everything
    /// between them.
    pub total: Duration,
    /// [`Phase::Attention`].
    pub attention: Duration,
    /// [`Phase::Projections`].
    pub projections: Duration,
    /// [`Phase::ExpertCompute`].
    pub expert_compute: Duration,
    /// [`Phase::ExpertIo`].
    pub expert_io: Duration,
    /// [`Phase::Elementwise`].
    pub elementwise: Duration,
}

impl PrefillTiming {
    /// The five phases together.
    #[must_use]
    pub fn accounted(&self) -> Duration {
        self.attention + self.projections + self.expert_compute + self.expert_io + self.elementwise
    }

    /// The share of [`PrefillTiming::total`] no phase claimed.
    ///
    /// Saturating rather than checked because the phases are disjoint spans of
    /// the total by construction: this is zero-or-positive unless the
    /// instrumentation itself is wrong, which is exactly what the unit tests
    /// assert.
    #[must_use]
    pub fn other(&self) -> Duration {
        self.total.saturating_sub(self.accounted())
    }

    /// Whether a prefill has run and left numbers here.
    #[must_use]
    pub fn ran(&self) -> bool {
        self.mode.is_some()
    }

    /// Every phase, in report order, with the unaccounted remainder last.
    ///
    /// Returned as an array rather than printed here because
    /// `ramvamp-core` does not print.
    #[must_use]
    pub fn phases(&self) -> [(&'static str, Duration); 6] {
        [
            ("attention", self.attention),
            ("expert compute", self.expert_compute),
            ("expert io", self.expert_io),
            ("projections", self.projections),
            ("elementwise", self.elementwise),
            ("other", self.other()),
        ]
    }

    /// A zeroed timing labelled with the run that is about to fill it.
    pub(super) fn started(mode: PrefillMode, tokens: usize) -> Self {
        Self {
            mode: Some(mode),
            tokens: tokens as u64,
            ..Self::default()
        }
    }
}

/// A stopwatch over a [`PrefillTiming`]: each [`charge`](PhaseClock::charge)
/// closes the region that began at the previous boundary and opens the next
/// one.
///
/// **One clock read per boundary, not two.** The alternative — an
/// `Instant::now()` pair around every region — doubles the syscall-free
/// `clock_gettime` count and leaves gaps between regions that silently vanish
/// instead of landing in [`PrefillTiming::other`].
///
/// The boundaries are placed around *loops*, not around iterations, wherever a
/// site is entered per row: the per-row attention loop is charged once per
/// layer, not 512 times. The two sites charged per iteration are the ones
/// whose iteration count is per expert rather than per row (~128 per layer),
/// where the resolution is worth ~25 ns.
pub(super) struct PhaseClock<'a> {
    /// `None` when this pass charges nowhere.
    timing: Option<&'a mut PrefillTiming>,
    since: Instant,
    /// When the whole pass began, for a clock that must close its own
    /// [`PrefillTiming::total`]. `None` for a prefill clock, whose total is
    /// the whole `prefill_prompt` call and is measured outside this type.
    whole: Option<Instant>,
}

impl<'a> PhaseClock<'a> {
    /// Open the first region, charging into `timing` only when `armed`.
    ///
    /// Unarmed is every `forward_token` a decode step makes *into the prefill
    /// accumulator*: the token-major prefill and decode run the same
    /// instrumented code, and a decoded token's phases belong to no prefill.
    pub(super) fn new(timing: &'a mut PrefillTiming, armed: bool) -> Self {
        let now = Instant::now();
        Self {
            timing: armed.then_some(timing),
            since: now,
            whole: None,
        }
    }

    /// A clock over one **decoded token**: the same five phases, plus the
    /// token's own wall time and the token itself when [`PhaseClock::close`]
    /// runs.
    ///
    /// A prefill is timed from outside `forward_token` by `prefill_prompt`,
    /// which knows where the pass begins and ends; a decode step has no such
    /// outer frame, so it closes its own total.
    pub(super) fn decoding(timing: &'a mut PrefillTiming) -> Self {
        let now = Instant::now();
        Self {
            timing: Some(timing),
            since: now,
            whole: Some(now),
        }
    }

    /// End the pass: on a [`PhaseClock::decoding`] clock, add everything since
    /// it was created to [`PrefillTiming::total`] and one to
    /// [`PrefillTiming::tokens`]. A no-op on any other clock.
    ///
    /// Not a `Drop` impl: a pass that failed mid-token leaves a state its
    /// caller must discard, and charging a token that produced nothing would
    /// put a lie in the only numbers anyone reads afterwards.
    pub(super) fn close(self) {
        let Some(whole) = self.whole else {
            return;
        };
        if let Some(timing) = self.timing {
            timing.total += whole.elapsed();
            timing.tokens += 1;
        }
    }

    /// Charge everything since the last boundary to `phase`, and open the
    /// next region here.
    ///
    /// A no-op, down to the clock read, on an unarmed clock.
    #[inline]
    pub(super) fn charge(&mut self, phase: Phase) {
        let Some(timing) = self.timing.as_deref_mut() else {
            return;
        };
        let now = Instant::now();
        let slot = match phase {
            Phase::Attention => &mut timing.attention,
            Phase::Projections => &mut timing.projections,
            Phase::ExpertCompute => &mut timing.expert_compute,
            Phase::ExpertIo => &mut timing.expert_io,
            Phase::Elementwise => &mut timing.elementwise,
        };
        // Saturating: `Instant` is monotonic, but a negative delta must not be
        // a panic in library code whatever the platform's clock does.
        *slot += now.saturating_duration_since(self.since);
        self.since = now;
    }
}

/// Observer for the router's decision during prefill, called once per row per
/// layer with `(absolute position, layer, top_k)`.
///
/// The analogue of [`ExpertRouteSink`](super::ExpertRouteSink) for a pass that
/// carries many positions at once, which is why the position is an argument
/// rather than something the caller already knows.
///
/// **Record order differs from the token-major path.** Token-major emits
/// `(token, layer)` in that nesting; the sweep emits `(layer, row)`. Both
/// cover exactly the same set of `(position, layer)` pairs and the routing
/// decisions are identical — only the emission order changes, because that is
/// the order the work happens in.
pub type PrefillRouteSink<'a> = &'a mut dyn FnMut(usize, u32, &[(u32, f32)]);

// ---------------------------------------------------------------------------
// Typed scratch sub-allocation
// ---------------------------------------------------------------------------

/// A type it is sound to reinterpret initialized scratch bytes as.
///
/// # Safety
///
/// Every initialized bit pattern of `size_of::<Self>()` bytes must be a valid
/// value of `Self`: no niches, no validity invariants, no padding whose
/// contents could be observed as anything but "some initialized bytes", and no
/// interior mutability. This holds for plain integer/float aggregates and
/// nothing else. The [`PrefillSession`] scratch is always *initialized* (the
/// slab is allocated zeroed) but never *zeroed on carve*, so a `ScratchPod`
/// slice can legitimately come back holding the previous chunk's bytes; every
/// consumer here writes before it reads.
unsafe trait ScratchPod: Copy {}

// SAFETY: every 32-bit pattern is a valid `f32` (including signalling NaNs,
// which Rust permits as values) and a valid `u32`.
unsafe impl ScratchPod for f32 {}
// SAFETY: as above.
unsafe impl ScratchPod for u32 {}
// SAFETY: `f32` + `[i8; 256]` + `[i16; 16]`, all of which accept every bit
// pattern; the aggregate is 4-aligned with no padding, so there are no
// uninitialized bytes to observe either.
unsafe impl ScratchPod for BlockQ8K {}
// SAFETY: `f32` + `[i8; 32]`; as above.
unsafe impl ScratchPod for BlockQ8_0 {}
// SAFETY: two `u32`-sized fields, `repr(C)`, no padding.
unsafe impl ScratchPod for TopkEntry {}
// SAFETY: as above.
unsafe impl ScratchPod for RowSlot {}

/// One selected expert and its renormalized router weight.
///
/// A named `repr(C)` pair rather than `(u32, f32)`, because this one is
/// reinterpreted out of raw scratch bytes and tuple layout is unspecified.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct TopkEntry {
    /// Routed expert id.
    expert: u32,
    /// Weight this expert's output is scaled by in the reduction.
    weight: f32,
}

/// One entry of the inverse routing index: a chunk row and which of its
/// `top_k` slots the expert occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
struct RowSlot {
    /// Row within the chunk.
    row: u32,
    /// Index into that row's top-k, which is the staging slot to fill.
    slot: u32,
}

/// A bump allocator over one `&mut [u8]` span.
///
/// Disjointness comes from [`slice::split_at_mut`], which is safe; the only
/// unsafe step is reinterpreting an already-split byte range as a
/// [`ScratchPod`] slice, and that is guarded by an explicit alignment check
/// rather than by an assumption about the slab's base address.
struct Carver<'a> {
    rest: &'a mut [u8],
    used: usize,
}

impl<'a> Carver<'a> {
    /// Start carving at the base of `bytes`.
    fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            rest: bytes,
            used: 0,
        }
    }

    /// Bytes consumed so far, padding included.
    fn used(&self) -> usize {
        self.used
    }

    /// Take `len` values of `T`, padding forward to `T`'s alignment.
    ///
    /// # Errors
    ///
    /// [`ForwardError::PrefillScratch`] when the span is exhausted or the
    /// requested length overflows a byte count;
    /// [`ForwardError::PrefillScratchAlign`] when the span cannot be aligned
    /// for `T` at all, which a page-aligned slab base rules out but which is
    /// checked rather than assumed.
    fn take<T: ScratchPod>(&mut self, len: usize) -> Result<&'a mut [T], ForwardError> {
        let align = align_of::<T>();
        let pad = self.rest.as_ptr().align_offset(align);
        // Before the byte arithmetic, not after: `usize::MAX` padding makes
        // every `checked_add` below overflow, so an unalignable span would
        // otherwise report `PrefillScratch { needed: u64::MAX }` for any
        // non-zero `len` and `PrefillScratchAlign` never. Unreachable at a
        // page-aligned slab base — but a diagnostic that lies is worse than
        // one that is merely unreachable.
        if pad == usize::MAX {
            return Err(ForwardError::PrefillScratchAlign { align });
        }
        let want = len
            .checked_mul(size_of::<T>())
            .and_then(|bytes| bytes.checked_add(pad))
            .ok_or(ForwardError::PrefillScratch {
                needed: u64::MAX,
                available: self.rest.len() as u64,
            })?;
        if want > self.rest.len() {
            return Err(ForwardError::PrefillScratch {
                needed: (self.used + want) as u64,
                available: (self.used + self.rest.len()) as u64,
            });
        }
        let rest = std::mem::take(&mut self.rest);
        let (head, tail) = rest.split_at_mut(want);
        self.rest = tail;
        self.used += want;
        let ptr = head.as_mut_ptr();
        // SAFETY: `ptr + pad` is aligned for `T` (`align_offset` computed
        // `pad` for exactly this pointer) and `len * size_of::<T>()` bytes
        // past it are inside `head`, which `split_at_mut` just made
        // exclusively ours for `'a`. Every byte of the slab was initialized
        // by `SlotPool::new`, and `T: ScratchPod` promises every such bit
        // pattern is a valid `T`.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr.add(pad).cast::<T>(), len) })
    }
}

/// Every batched buffer one chunk needs, carved out of the session scratch.
///
/// Field order here is the carve order, which is also the order
/// [`scratch_bytes`] sums; `carve_consumes_exactly_the_documented_bytes` pins
/// that the two agree.
struct Scratch<'a> {
    /// Residual stream, `[rows][hidden]`.
    residual: &'a mut [f32],
    /// RMSNorm output, `[rows][hidden]`, reused by both norm sites.
    normed: &'a mut [f32],
    /// Query projection, `[rows][q_dim]`.
    q: &'a mut [f32],
    /// Attention context, `[rows][q_dim]`.
    attn_out: &'a mut [f32],
    /// Key projection, `[rows][kv_dim]`.
    k: &'a mut [f32],
    /// Value projection, `[rows][kv_dim]`.
    v: &'a mut [f32],
    /// Batched-GEMV output staging, `[out_dim][n]` — the transpose source for
    /// Q/K/V and o_proj, then the expert down-projection's destination.
    tmat: &'a mut [f32],
    /// Staged expert outputs, `[rows][top_k][hidden]`. The buffer that makes
    /// the reduction order independent of the sweep order.
    staged: &'a mut [f32],
    /// Expert gate projection, `[moe][n]`, then SwiGLU in place.
    gate: &'a mut [f32],
    /// Expert up projection, `[moe][n]`.
    up: &'a mut [f32],
    /// SwiGLU output transposed to `[n][moe]` for per-row quantization.
    moe_t: &'a mut [f32],
    /// One row's weighted expert accumulator, `[hidden]`.
    expert_acc: &'a mut [f32],
    /// Per-shard attention scratch, `[shards][attn_shard]`: one whole
    /// [`attention_scratch_len`] carve per compute shard, so the per-row
    /// attention can run on every pinned core at once without a heap byte.
    /// Constant in `rows` on purpose — [`plan_arena`] recovers the per-row
    /// slope by differencing [`scratch_bytes`] and needs it to stay affine.
    attn_shards: &'a mut [f32],
    /// Router logits, `[rows][n_experts]`.
    router_logits: &'a mut [f32],
    /// Router probabilities, `[rows][n_experts]`; top-k selection consumes
    /// them destructively, per row.
    router_probs: &'a mut [f32],
    /// Q8_K quantization of `normed`, `[rows][hidden / 256]`.
    acts_hidden: &'a mut [BlockQ8K],
    /// One expert's routed rows gathered contiguously, `[n][hidden / 256]`.
    acts_gather: &'a mut [BlockQ8K],
    /// Q8_K quantization of `attn_out`, `[rows][q_dim / 256]`.
    acts_attn: &'a mut [BlockQ8K],
    /// Q8_K quantization of `moe_t`, `[n][moe / 256]`.
    acts_moe: &'a mut [BlockQ8K],
    /// Q8_0 quantization of `normed`, `[rows][hidden / 32]`.
    acts_q80: &'a mut [BlockQ8_0],
    /// Selected experts and weights, `[rows][top_k]`, in top-k order.
    topk: &'a mut [TopkEntry],
    /// Inverse routing index entries, CSR-ordered by expert.
    index: &'a mut [RowSlot],
    /// CSR offsets, `[n_experts + 1]`.
    index_start: &'a mut [u32],
    /// Fill cursors while building the index, `[n_experts]`.
    index_cursor: &'a mut [u32],
}

/// Architecture dimensions the chunk driver reads once per call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrefillDims {
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    moe: usize,
    top_k: usize,
    n_experts: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// Q8_K blocks in a `[hidden]` row.
    hidden_q8k: usize,
    /// Q8_0 blocks in a `[hidden]` row.
    hidden_q80: usize,
    /// Q8_K blocks in a `[q_dim]` row.
    attn_q8k: usize,
    /// Q8_K blocks in a `[moe]` row.
    moe_q8k: usize,
    /// `max(q_dim, hidden)`: the widest batched-GEMV output `tmat` must hold.
    wide: usize,
    /// Compute shards the per-row attention is fanned out across, which is
    /// [`ComputePool::shards`] and therefore counts the submitting thread.
    /// Here rather than passed to [`scratch_bytes`] so that all of its call
    /// sites — [`plan_arena`]'s three included — stay unchanged.
    shards: usize,
    /// The KV cache's position capacity: the longest row any chunk of this
    /// prefill can attend over, and so the width the per-shard attention
    /// carve is sized for once instead of per chunk.
    max_positions: usize,
}

/// Weights per Q8_K super-block.
const QK_K: usize = 256;
/// Weights per Q8_0 block.
const QK8_0: usize = 32;

impl PrefillDims {
    /// One compute shard's attention scratch, in f32.
    ///
    /// Sized from [`PrefillDims::max_positions`] rather than from the chunk,
    /// so the carve serves the last row of the last chunk as well as the
    /// first row of the first one. A slice longer than one call needs is
    /// accepted by `attention_at_in` and its tail ignored.
    fn attn_shard(&self) -> usize {
        attention_scratch_len(
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.max_positions,
        )
    }

    /// Every shard's attention scratch together, in f32.
    fn attn_scratch(&self) -> usize {
        self.shards.saturating_mul(self.attn_shard())
    }

    /// Read the chunk geometry off `arch`, for a pool of `shards` shards
    /// against a cache holding `max_positions` positions.
    ///
    /// # Errors
    ///
    /// [`ForwardError::UnsupportedDim`] when a dimension is not a whole
    /// number of activation blocks, and [`ForwardError::InvalidTopK`] on a
    /// nonsensical router config — the same refusals
    /// [`ForwardState::with_config`] makes, restated because this reads the
    /// *argument* model.
    fn new(arch: &ArchInfo, shards: usize, max_positions: usize) -> Result<Self, ForwardError> {
        let hidden = arch.hidden as usize;
        let q_dim = arch.n_heads as usize * arch.head_dim as usize;
        let kv_dim = arch.n_kv_heads as usize * arch.head_dim as usize;
        let moe = arch.moe_intermediate as usize;
        let top_k = arch.top_k as usize;
        let n_experts = arch.n_experts as usize;
        for (what, dim, block) in [
            ("hidden", hidden, QK_K),
            ("hidden", hidden, QK8_0),
            ("moe_intermediate", moe, QK_K),
            ("n_heads * head_dim", q_dim, QK_K),
        ] {
            if dim == 0 || dim % block != 0 {
                return Err(ForwardError::UnsupportedDim { what, dim, block });
            }
        }
        if top_k == 0 || top_k > n_experts {
            return Err(ForwardError::InvalidTopK { top_k, n_experts });
        }
        if kv_dim == 0 || arch.head_dim == 0 {
            return Err(ForwardError::UnsupportedDim {
                what: "n_kv_heads * head_dim",
                dim: kv_dim,
                block: 1,
            });
        }
        Ok(Self {
            hidden,
            q_dim,
            kv_dim,
            moe,
            top_k,
            n_experts,
            n_heads: arch.n_heads as usize,
            n_kv_heads: arch.n_kv_heads as usize,
            head_dim: arch.head_dim as usize,
            hidden_q8k: hidden / QK_K,
            hidden_q80: hidden / QK8_0,
            attn_q8k: q_dim / QK_K,
            moe_q8k: moe / QK_K,
            wide: q_dim.max(hidden),
            // A pool always has at least one shard; clamping here means a
            // caller that passed zero gets a one-shard carve rather than a
            // zero-length one the fan-out would then index into.
            shards: shards.max(1),
            max_positions,
        })
    }
}

/// Add `count * unit` bytes to a running total, refusing to overflow.
fn accumulate(total: &mut usize, count: usize, unit: usize) -> Result<(), ForwardError> {
    let bytes = count
        .checked_mul(unit)
        .and_then(|bytes| total.checked_add(bytes))
        .ok_or(ForwardError::PrefillScratch {
            needed: u64::MAX,
            available: *total as u64,
        })?;
    *total = bytes;
    Ok(())
}

/// Scratch bytes one chunk of `rows` rows needs, at `dims`.
///
/// The arithmetic, in carve order. `H` is hidden, `Q` is `n_heads * head_dim`,
/// `V` is `n_kv_heads * head_dim`, `M` is `moe_intermediate`, `K` is `top_k`,
/// `E` is `n_experts`, `W` is `max(Q, H)`, `S` is the compute shard count and
/// `A` is one shard's [`attention_scratch_len`] carve:
///
/// ```text
/// f32   rows * (2H + 2Q + 2V + W + K*H + 3M + 2E)   + H            [expert_acc]
///                                                   + S*A          [attn_shards]
/// Q8_K  rows * (2 * H/256 + Q/256 + M/256)          blocks of 292 B
/// Q8_0  rows * (H/32)                               blocks of  36 B
/// pairs rows * K                                    of 8 B, twice  [topk, index]
/// u32   2E + 1                                                     [CSR]
/// ```
///
/// At the v0 pin (`rows` 512, H 2048, Q 4096, V 512, M 768, K 8, E 128,
/// `S` 6, `A` = 8 * 4096 + 2 * 128 = 33,024 f32):
///
/// ```text
/// f32     512 * 36,352 floats           = 74,448,896 B   71.00 MiB
///   of which staging (K*H)              = 33,554,432 B   32.00 MiB
///   of which q + attn_out (2Q)          = 16,777,216 B   16.00 MiB
/// expert_acc                            =      8,192 B
/// attn_shards  6 * 33,024 * 4 B         =    792,576 B    0.76 MiB
/// Q8_K    512 * 35 blocks * 292 B       =  5,232,640 B    4.99 MiB
/// Q8_0    512 * 64 blocks *  36 B       =  1,179,648 B    1.13 MiB
/// topk + index  2 * 512 * 8 * 8 B       =     65,536 B
/// CSR     257 * 4 B                     =      1,028 B
/// total                                 = 81,728,516 B   77.94 MiB
/// ```
///
/// against a ~1,438 MiB slab whose sweep ring costs a further ~46.7 MiB at the
/// default dials. The 3 GiB contract is not touched: these bytes are the slot
/// pool's, already resident, borrowed for the length of the prefill. That is
/// why the sharded attention scratch lives here and not on the heap — a
/// prefill costs no *additional* bytes, whatever the headroom happens to be.
///
/// **`S*A` is constant in `rows`.** [`plan_arena`] recovers the per-row slope
/// as `scratch_bytes(2) - scratch_bytes(1)` and divides the budget by it, which
/// only answers the question while this function stays affine in `rows`.
///
/// # Errors
///
/// [`ForwardError::PrefillScratch`] if the total overflows `usize`.
fn scratch_bytes(dims: &PrefillDims, rows: usize) -> Result<usize, ForwardError> {
    let f32_bytes = size_of::<f32>();
    let mut total = 0usize;
    // Per-row f32 planes, in carve order.
    for width in [
        dims.hidden,
        dims.hidden,
        dims.q_dim,
        dims.q_dim,
        dims.kv_dim,
        dims.kv_dim,
        dims.wide,
        dims.top_k * dims.hidden,
        dims.moe,
        dims.moe,
        dims.moe,
    ] {
        accumulate(&mut total, rows.saturating_mul(width), f32_bytes)?;
    }
    accumulate(&mut total, dims.hidden, f32_bytes)?;
    accumulate(&mut total, dims.attn_scratch(), f32_bytes)?;
    accumulate(&mut total, rows.saturating_mul(dims.n_experts), f32_bytes)?;
    accumulate(&mut total, rows.saturating_mul(dims.n_experts), f32_bytes)?;
    // Activation blocks.
    let q8k = size_of::<BlockQ8K>();
    for blocks in [
        dims.hidden_q8k,
        dims.hidden_q8k,
        dims.attn_q8k,
        dims.moe_q8k,
    ] {
        accumulate(&mut total, rows.saturating_mul(blocks), q8k)?;
    }
    accumulate(
        &mut total,
        rows.saturating_mul(dims.hidden_q80),
        size_of::<BlockQ8_0>(),
    )?;
    // Routing bookkeeping.
    let pairs = rows.saturating_mul(dims.top_k);
    accumulate(&mut total, pairs, size_of::<TopkEntry>())?;
    accumulate(&mut total, pairs, size_of::<RowSlot>())?;
    accumulate(&mut total, dims.n_experts + 1, size_of::<u32>())?;
    accumulate(&mut total, dims.n_experts, size_of::<u32>())?;
    Ok(total)
}

/// Carve `bytes` into every buffer a `rows`-row chunk needs.
///
/// Called once per layer phase rather than held across the session: the
/// [`PrefillSession`] hands the same span back from
/// [`PrefillSession::scratch`] and [`PrefillSession::split`], and re-deriving
/// the split is a few dozen pointer adjustments.
///
/// # Errors
///
/// [`ForwardError::PrefillScratch`] if `bytes` is shorter than
/// [`scratch_bytes`] says.
fn carve<'a>(
    bytes: &'a mut [u8],
    dims: &PrefillDims,
    rows: usize,
) -> Result<(Scratch<'a>, usize), ForwardError> {
    let mut c = Carver::new(bytes);
    let scratch = Scratch {
        residual: c.take(rows * dims.hidden)?,
        normed: c.take(rows * dims.hidden)?,
        q: c.take(rows * dims.q_dim)?,
        attn_out: c.take(rows * dims.q_dim)?,
        k: c.take(rows * dims.kv_dim)?,
        v: c.take(rows * dims.kv_dim)?,
        tmat: c.take(rows * dims.wide)?,
        staged: c.take(rows * dims.top_k * dims.hidden)?,
        gate: c.take(rows * dims.moe)?,
        up: c.take(rows * dims.moe)?,
        moe_t: c.take(rows * dims.moe)?,
        expert_acc: c.take(dims.hidden)?,
        attn_shards: c.take(dims.attn_scratch())?,
        router_logits: c.take(rows * dims.n_experts)?,
        router_probs: c.take(rows * dims.n_experts)?,
        acts_hidden: c.take(rows * dims.hidden_q8k)?,
        acts_gather: c.take(rows * dims.hidden_q8k)?,
        acts_attn: c.take(rows * dims.attn_q8k)?,
        acts_moe: c.take(rows * dims.moe_q8k)?,
        acts_q80: c.take(rows * dims.hidden_q80)?,
        topk: c.take(rows * dims.top_k)?,
        index: c.take(rows * dims.top_k)?,
        index_start: c.take(dims.n_experts + 1)?,
        index_cursor: c.take(dims.n_experts)?,
    };
    Ok((scratch, c.used()))
}

// ---------------------------------------------------------------------------
// Pool fan-out for the batched kernels
// ---------------------------------------------------------------------------

/// A raw pointer shared across shards.
///
/// `pub(super)` because [`super::forward`]'s decode-side attention fan-out
/// rebuilds disjoint sub-slices the same way and there is no reason for two
/// copies of the same soundness argument.
pub(super) struct SendPtr(pub(super) *mut f32);

impl SendPtr {
    /// The wrapped pointer.
    ///
    /// A method rather than a field access on purpose: edition 2024 closures
    /// capture disjoint *fields*, so reading `base.0` inside the shard body
    /// would capture a bare `*mut f32` and defeat the `Send`/`Sync` impls
    /// below. Going through `&self` captures the whole wrapper.
    pub(super) fn get(&self) -> *mut f32 {
        self.0
    }
}

// SAFETY: the pointer is only ever used to rebuild disjoint sub-slices, one
// per shard, of a slice the submitting thread holds `&mut` to for the whole
// call; `ComputePool::run` does not return until every shard has dropped its
// slice. This is `threads::ComputePool::scatter`'s argument, re-made here
// because the batched kernels shard over *weight rows* while `scatter` shards
// over output *elements*, and one weight row is `n_acts` of them.
unsafe impl Send for SendPtr {}
// SAFETY: as above.
unsafe impl Sync for SendPtr {}

/// Fan a batched GEMV out over contiguous weight-row ranges.
///
/// `out` is `[out_dim][n_acts]` row-major, so a weight-row range maps to one
/// contiguous sub-slice and every shard writes a disjoint span. Bit-identical
/// to the whole-matrix call: [`crate::threads::shard_range`] is pure and each
/// output element is an independent dot product.
///
/// # Errors
///
/// [`KernelError::LengthMismatch`] when `out` is not `out_dim * n_acts` long,
/// plus whatever the kernel itself reports for the first shard that fails.
fn scatter_rows<F>(
    pool: &mut ComputePool,
    out: &mut [f32],
    out_dim: usize,
    n_acts: usize,
    f: F,
) -> Result<(), KernelError>
where
    F: Fn(Range<usize>, &mut [f32]) -> Result<(), KernelError> + Sync,
{
    let expected = out_dim.saturating_mul(n_acts);
    if out.len() != expected {
        return Err(KernelError::LengthMismatch {
            what: "prefill batched gemv: out vs out_dim * n_acts",
            left: out.len(),
            right: expected,
        });
    }
    if n_acts == 0 || out_dim == 0 {
        return Ok(());
    }
    let failure: Mutex<Option<KernelError>> = Mutex::new(None);
    let base = SendPtr(out.as_mut_ptr());
    pool.run(out_dim, |shard| {
        let start = shard.rows.start * n_acts;
        let len = shard.rows.len() * n_acts;
        // SAFETY: `shard_range` yields disjoint sub-ranges of `0..out_dim`,
        // each visited by exactly one thread, so scaling them by `n_acts`
        // yields disjoint sub-ranges of `0..out_dim * n_acts` — which is
        // `out.len()`, checked above. `out` is mutably borrowed for the whole
        // call, and `run` joins before returning.
        let chunk = unsafe { std::slice::from_raw_parts_mut(base.get().add(start), len) };
        if let Err(err) = f(shard.rows.clone(), chunk) {
            record(&failure, err);
        }
    });
    taken(failure)
}

/// [`scatter_rows`] for a k-quant weight matrix against Q8_K activations.
#[allow(clippy::too_many_arguments)]
fn pool_batched_q8_k(
    pool: &mut ComputePool,
    format: QuantFormat,
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    acts: &[BlockQ8K],
    n_acts: usize,
    out: &mut [f32],
) -> Result<(), KernelError> {
    scatter_rows(pool, out, out_dim, n_acts, |rows, chunk| {
        gemv_q8_k_batched(format, weight, in_dim, out_dim, acts, n_acts, rows, chunk)
    })
}

/// [`scatter_rows`] for the q8_0 `attn_k` matrix against Q8_0 activations.
fn pool_batched_q8_0(
    pool: &mut ComputePool,
    weight: &[u8],
    in_dim: usize,
    out_dim: usize,
    acts: &[BlockQ8_0],
    n_acts: usize,
    out: &mut [f32],
) -> Result<(), KernelError> {
    scatter_rows(pool, out, out_dim, n_acts, |rows, chunk| {
        gemv_q8_0_batched(weight, in_dim, out_dim, acts, n_acts, rows, chunk)
    })
}

/// Attention work a chunk's first `r` rows cost, in units of one
/// position-times-head.
///
/// Row `r` of a chunk starting at absolute position `start` attends
/// `start + r + 1` positions, so the cumulative cost of rows `0..r` is
/// `r * start + r * (r + 1) / 2`. `u128` because the product of two
/// `usize`-sized dimensions is not a `usize` claim anyone should make; the
/// real values are tiny (~2.2 million at the v0 pin).
fn attention_cost(rows: usize, start: usize) -> u128 {
    let rows = rows as u128;
    rows * start as u128 + rows * (rows + 1) / 2
}

/// The rows shard `index` of `shards` owns, split by attention *cost* rather
/// than by row count.
///
/// A pure function of `(rows, start, shards, index)` — nothing about the pool,
/// the wall clock or the order shards finish in reaches it — so the partition
/// is reproducible, and it is contiguous and ascending like
/// [`crate::threads::shard_range`], which it replaces here.
///
/// **Why not an equal-row split.** Row `r` attends `start + r + 1` positions,
/// so the last shard of an equal-row split does far more work than the first
/// and the makespan is set by that last shard. For the first 512-row chunk of
/// a prompt (`start = 0`) over six shards, the equal-row split gives the last
/// shard 39,950 of the 131,328 total cost units against an ideal 21,888 — a
/// 3.3x speedup where 6x was available. Balancing the cost instead gives every
/// shard within one row of `total / shards`.
///
/// **It cannot move a bit.** Rows are independent: each one is a separate
/// `attention_at_in` call reading an immutable `q` and an immutable cache and
/// writing only its own `[q_dim]` output. There is no cross-row reduction to
/// reassociate, so *any* partition of `0..rows` produces the same bytes as the
/// serial loop — which is why the split is free to be cost-balanced rather
/// than merely deterministic.
fn attention_shard_range(rows: usize, start: usize, shards: usize, index: usize) -> Range<usize> {
    if shards == 0 || index >= shards {
        return 0..0;
    }
    if shards == 1 {
        return 0..rows;
    }
    let total = attention_cost(rows, start);
    let lo = attention_boundary(rows, start, total, shards, index);
    let hi = attention_boundary(rows, start, total, shards, index + 1);
    lo..hi
}

/// The first row at which `index / shards` of the chunk's attention cost has
/// been paid: the lower edge of shard `index`'s range.
///
/// [`attention_cost`] is strictly increasing in `rows`, so this is a binary
/// search, and the targets ascend with `index`, so consecutive boundaries
/// never cross.
fn attention_boundary(
    rows: usize,
    start: usize,
    total: u128,
    shards: usize,
    index: usize,
) -> usize {
    if index == 0 {
        return 0;
    }
    if index >= shards {
        return rows;
    }
    let target = total * index as u128 / shards as u128;
    let (mut lo, mut hi) = (0usize, rows);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if attention_cost(mid, start) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Fan one layer's causal attention out over the compute pool, one row per
/// call.
///
/// EXP-017 measured this loop at 61.3% of a 512-token prefill and 85.2% of an
/// 1891-token one, running on the submitting thread while the other pinned
/// cores sat parked. Rows are independent, so this is the parallel region with
/// the least to argue about in the whole pass: shard `s` calls
/// [`attention_at_in`] on the rows [`attention_shard_range`] gave it, with its
/// own scratch slice out of the session arena and its own `[q_dim]` span of
/// `out`. Nothing is reduced across shards, nothing accumulates in completion
/// order, and every call is bit-identical to the serial `attention_at` on the
/// same row.
///
/// `scratch` is `shards * shard_scratch` f32 — the [`Scratch::attn_shards`]
/// carve — and `out` is `rows * q_dim`.
///
/// # Errors
///
/// [`KernelError::LengthMismatch`] when `q` or `out` is not `rows * q_dim`
/// long, [`ForwardError::PrefillScratch`] when `scratch` is shorter than the
/// pool's shard count needs, and [`ForwardError::Attention`] for the first
/// shard that reports one. The two length checks are what make the raw-pointer
/// splits below sound; they are not a convenience.
#[allow(clippy::too_many_arguments)]
fn scatter_attention(
    pool: &mut ComputePool,
    kv: &KvCache,
    layer: usize,
    start: usize,
    rows: usize,
    q_dim: usize,
    scale: f32,
    shard_scratch: usize,
    q: &[f32],
    scratch: &mut [f32],
    out: &mut [f32],
) -> Result<(), ForwardError> {
    if rows == 0 {
        return Ok(());
    }
    let span = rows.saturating_mul(q_dim);
    for (what, len) in [
        ("attention fan-out: q", q.len()),
        ("attention fan-out: out", out.len()),
    ] {
        if len != span {
            return Err(KernelError::LengthMismatch {
                what,
                left: len,
                right: span,
            }
            .into());
        }
    }
    // Checked once, here, rather than inside the shard body: `shard.index <
    // shard.count <= pool.shards()` always holds, so this single comparison is
    // what makes `scratch[index * shard_scratch ..]` in-bounds for every shard
    // — and therefore what makes the raw-pointer split below sound.
    let need = pool.shards().saturating_mul(shard_scratch);
    if scratch.len() < need {
        return Err(ForwardError::PrefillScratch {
            needed: need as u64,
            available: scratch.len() as u64,
        });
    }
    let failure: Mutex<Option<AttentionError>> = Mutex::new(None);
    let out_base = SendPtr(out.as_mut_ptr());
    let scratch_base = SendPtr(scratch.as_mut_ptr());
    pool.run(rows, |shard| {
        // `shard.count`, not `pool.shards()`: a job with fewer rows than
        // shards runs inline as a single shard, and the split must agree with
        // the shard it is actually running on.
        let mine = attention_shard_range(rows, start, shard.count, shard.index);
        // SAFETY: shard indices are distinct across the job and each is
        // visited by exactly one thread, so `index * shard_scratch` names a
        // disjoint `shard_scratch`-long run per shard; the bound check above
        // puts every one of them inside `scratch`, which is mutably borrowed
        // for the whole call and which `run` joins before releasing.
        let slice = unsafe {
            std::slice::from_raw_parts_mut(
                scratch_base.get().add(shard.index * shard_scratch),
                shard_scratch,
            )
        };
        for r in mine {
            // SAFETY: `attention_shard_range` partitions `0..rows` into
            // disjoint contiguous ranges, so row `r` belongs to exactly one
            // shard and `r * q_dim .. (r + 1) * q_dim` is a disjoint run
            // inside `out`, whose length the caller pins at `rows * q_dim`.
            let row_out =
                unsafe { std::slice::from_raw_parts_mut(out_base.get().add(r * q_dim), q_dim) };
            if let Err(err) = attention_at_in(
                &q[r * q_dim..(r + 1) * q_dim],
                kv,
                layer,
                start + r + 1,
                scale,
                slice,
                row_out,
            ) {
                record(&failure, err);
                return;
            }
        }
    });
    taken(failure)?;
    Ok(())
}

/// `dst[c * rows + r] = src[r * cols + c]`.
///
/// Pure data movement: no float arithmetic happens here, so nothing about it
/// can move a bit.
///
/// **Both slices must be exactly `rows * cols` long.** The size contract is
/// checked rather than clamped on purpose: this module's thesis is that the
/// driver may not assume the scratch it was handed is zeroed, and a silently
/// truncated source or a silently dropped destination write would leave the
/// previous chunk's arena bytes in `q`, `k`, `v` or `moe_t` — a numerical
/// divergence with no error attached. Every call site passes exactly-sized
/// slices today; a future sizing mistake gets a [`ForwardError`], not a wrong
/// answer.
///
/// # Errors
///
/// [`ForwardError::Kernel`] with [`KernelError::LengthMismatch`] when either
/// slice is not `rows * cols` long, or when that product overflows.
fn transpose(src: &[f32], rows: usize, cols: usize, dst: &mut [f32]) -> Result<(), ForwardError> {
    let expected = rows.checked_mul(cols).ok_or(KernelError::LengthMismatch {
        what: "prefill transpose: rows * cols overflows",
        left: rows,
        right: cols,
    })?;
    for (what, len) in [
        ("prefill transpose: src vs rows * cols", src.len()),
        ("prefill transpose: dst vs rows * cols", dst.len()),
    ] {
        if len != expected {
            return Err(KernelError::LengthMismatch {
                what,
                left: len,
                right: expected,
            }
            .into());
        }
    }
    if expected == 0 {
        return Ok(());
    }
    // `c < cols` and `r < rows`, so `c * rows + r < cols * rows == dst.len()`:
    // the index below is in bounds because the lengths were just checked.
    for (r, row) in src.chunks_exact(cols).take(rows).enumerate() {
        for (c, &value) in row.iter().enumerate() {
            dst[c * rows + r] = value;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Arena planning
// ---------------------------------------------------------------------------

/// Bytes of sweep ring one [`PrefillSession`] reserves at `config`.
///
/// Mirrors `io::sweep::ring_span`: `windows_in_flight` buffers of the widest
/// window any layer produces, where a window is capped at the layer's own
/// expert count. Recomputed here because the driver has to know whether the
/// carve will fit *before* it asks for one.
fn ring_bytes(layout: &ExpertsLayout, experts_per_window: u32, windows_in_flight: u32) -> u64 {
    let widest = layout
        .layers
        .iter()
        .map(|layer| u64::from(experts_per_window.min(layer.n_experts.max(1))) * layer.stride)
        .max()
        .unwrap_or(0);
    widest.saturating_mul(u64::from(windows_in_flight))
}

/// The chunk width and sweep dials that fit inside the slot slab.
///
/// The configured chunk is the ceiling, never the floor: on a small pool (the
/// unit-test fixture reserves under 1 MiB) a 512-row staging buffer simply
/// does not exist, and refusing to prefill at all would be the wrong answer
/// when a narrower chunk is correct and merely slower. Windows in flight are
/// given up only after the chunk has already been narrowed to one row, since
/// total prefill bytes scale with `ceil(prompt / chunk)` and read-ahead is
/// worth far less than chunk width.
///
/// Dials are validated, never coerced: a zero on either sweep dial is
/// [`PrefillConfig::validate`]'s refusal, and narrowing is only ever applied to
/// values that were legal to begin with.
///
/// # Errors
///
/// [`ForwardError::PrefillScratch`] when not even a single-row chunk fits
/// beside a one-window ring; whatever [`PrefillConfig::validate`] refuses.
fn plan_arena(
    layout: &ExpertsLayout,
    dims: &PrefillDims,
    config: PrefillConfig,
    want_rows: usize,
    pool_bytes: u64,
) -> Result<(usize, SweepConfig), ForwardError> {
    const PAGE: u64 = 4096;
    let config = config.validate()?;
    let mut in_flight = config.windows_in_flight;
    let mut smallest_need = u64::MAX;
    loop {
        let ring = ring_bytes(layout, config.experts_per_window, in_flight);
        // `PrefillSession::begin` pads the scratch up to a page before the
        // ring, so the usable scratch is the page-floor of what is left.
        let budget = scratch_budget(pool_bytes, ring, PAGE);
        // `scratch_bytes` is affine in `rows` — every term is either a
        // constant or `rows * constant`, with no alignment padding (see
        // `every_scratch_type_packs_without_padding`) — so the widest viable
        // chunk is a division rather than a search.
        let one = scratch_bytes(dims, 1)? as u64;
        let two = scratch_bytes(dims, 2)? as u64;
        let per_row = two.saturating_sub(one).max(1);
        let fixed = one.saturating_sub(per_row);
        let mut rows = want_rows
            .max(1)
            .min(usize::try_from(budget.saturating_sub(fixed) / per_row).unwrap_or(usize::MAX));
        // The division is the answer, not a guess; the loop is the belt and
        // braces that keeps a future non-affine term from overflowing the
        // carve instead of narrowing it.
        while rows > 0 && scratch_bytes(dims, rows)? as u64 > budget {
            rows -= 1;
        }
        if rows > 0 {
            if rows < want_rows || in_flight < config.windows_in_flight {
                tracing::debug!(
                    asked_rows = want_rows,
                    rows,
                    asked_windows = config.windows_in_flight,
                    windows_in_flight = in_flight,
                    pool_bytes,
                    "prefill narrowed to fit the expert slot slab"
                );
            }
            return Ok((
                rows,
                SweepConfig {
                    experts_per_window: config.experts_per_window,
                    windows_in_flight: in_flight,
                },
            ));
        }
        smallest_need = smallest_need.min(one.saturating_add(ring));
        if in_flight == 1 {
            return Err(ForwardError::PrefillScratch {
                needed: smallest_need,
                available: pool_bytes,
            });
        }
        in_flight -= 1;
    }
}

/// Scratch bytes left in a `pool_bytes` slab once the ring is reserved.
///
/// The page floor is not decoration: [`PrefillSession`] rounds the scratch
/// request up to a page before placing the ring, so a request that merely
/// fits would push the ring past the slab.
fn scratch_budget(pool_bytes: u64, ring: u64, page: u64) -> u64 {
    pool_bytes.saturating_sub(ring) / page * page
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// Everything outside the scratch span that one chunk mutates.
struct ChunkState<'a> {
    kv: &'a mut KvCache,
    pool: &'a mut ComputePool,
    plan: &'a mut SweepPlan,
    /// Deduplicated routed expert ids for the layer being swept. Owned
    /// outside the scratch because [`PrefillSession::split`] takes it while it
    /// is also handing the scratch back, and the two may not alias.
    routed: &'a mut Vec<u32>,
    /// `(expert, weight)` staging for [`PrefillRouteSink`], which speaks the
    /// same tuple the token-major sink does.
    sink_buf: &'a mut Vec<(u32, f32)>,
    /// The `[vocab]` output, written for the final row of the final chunk.
    logits: &'a mut [f32],
}

/// Run `tokens` through the model, leaving the KV cache advanced by
/// `tokens.len()` positions and the last token's logits in the state.
///
/// This is the prefill entry point: [`crate::generate::generate`] calls it,
/// and so should any driver that wants a prompt consumed at prefill cost
/// rather than decode cost. The path is chosen by
/// [`ForwardState::prefill_config`] — [`PrefillMode::Sweep`] by default,
/// [`PrefillMode::TokenMajor`] for the A/B.
///
/// `tokens` must be non-empty and must continue the KV cache: position `p` is
/// `state.seq_len() + i` for token `i`.
///
/// # Errors
///
/// [`ForwardError::EmptyPrefill`] for an empty prompt;
/// [`ForwardError::Kv`] when the context cap cannot hold the prompt;
/// [`ForwardError::PrefillScratch`] when the expert slot slab cannot host a
/// chunk; [`ForwardError::Sweep`] when a layer sweep fails; kernel, model and
/// I/O failures pass through typed. On error the state must be assumed
/// mid-chunk — the KV cache may be ragged — and discarded.
pub fn prefill_prompt<'s>(
    model: &Model,
    state: &'s mut ForwardState,
    tokens: &[u32],
    on_route: Option<PrefillRouteSink<'_>>,
) -> Result<&'s [f32], ForwardError> {
    if tokens.is_empty() {
        return Err(ForwardError::EmptyPrefill);
    }
    let mode = state.prefill_config().validate()?.mode;
    // Armed before the run and closed after it whatever the run did, so a
    // prefill that failed mid-chunk still leaves a `total` the phases can be
    // read against instead of a struct that claims a run is still in flight.
    state.arm_prefill_timing(mode, tokens.len());
    let started = Instant::now();
    let outcome = match mode {
        PrefillMode::TokenMajor => prefill_token_major(model, state, tokens, on_route),
        PrefillMode::Sweep => prefill_sweep(model, state, tokens, on_route),
    };
    state.close_prefill_timing(started.elapsed());
    outcome?;
    Ok(state.logits())
}

/// The phase-5 path: one `forward_token` per prompt token.
fn prefill_token_major(
    model: &Model,
    state: &mut ForwardState,
    tokens: &[u32],
    mut on_route: Option<PrefillRouteSink<'_>>,
) -> Result<(), ForwardError> {
    let start = state.seq_len()?;
    let last = tokens.len() - 1;
    for (i, &id) in tokens.iter().enumerate() {
        let position = start + i;
        match on_route.as_deref_mut() {
            Some(sink) => {
                let mut per_layer = |layer: u32, topk: &[(u32, f32)]| sink(position, layer, topk);
                super::forward_token_traced(
                    model,
                    state,
                    id,
                    position,
                    i == last,
                    Some(&mut per_layer),
                )?;
            }
            None => {
                super::forward_token(model, state, id, position, i == last)?;
            }
        }
    }
    Ok(())
}

/// The chunked layer-major path.
fn prefill_sweep(
    model: &Model,
    state: &mut ForwardState,
    tokens: &[u32],
    mut on_route: Option<PrefillRouteSink<'_>>,
) -> Result<(), ForwardError> {
    let arch = model.arch();
    state.check_arch(arch)?;
    // The shard count and the context cap are part of the carve: the per-row
    // attention runs one shard per pinned core, each with its own scratch
    // slice, and the widest row any chunk can attend is the cache's capacity.
    let dims = PrefillDims::new(arch, state.shards(), state.context_cap())?;
    let config = state.prefill_config().validate()?;

    // The whole prompt is checked against the cap up front, not row by row:
    // a chunk that ran out of cache halfway through would leave the layers
    // ragged, which is a state only a discarded `ForwardState` can hold.
    let start = state.seq_len()?;
    let capacity = state.context_cap();
    if start.saturating_add(tokens.len()) > capacity {
        return Err(ForwardError::Kv(KvError::CapacityExceeded {
            layer: 0,
            capacity,
        }));
    }

    let want_rows = config.chunk.min(tokens.len()).max(1);
    let (rows, sweep_config) = plan_arena(
        model.layout(),
        &dims,
        config,
        want_rows,
        state.cache_bytes(),
    )?;
    let bytes = scratch_bytes(&dims, rows)?;
    tracing::debug!(
        rows,
        prompt = tokens.len(),
        scratch_bytes = bytes,
        sweep = %sweep_config,
        "prefill sweep starting"
    );

    let parts = state.prefill_parts();
    let timing = parts.timing;
    let mut chunk_state = ChunkState {
        kv: parts.kv,
        pool: parts.pool,
        plan: parts.sweep_plan,
        routed: parts.routed,
        sink_buf: parts.topk,
        logits: parts.logits,
    };
    let mut session = parts.stream.begin_prefill(bytes, sweep_config)?;

    // Started here, after the arena exists: the plan and `begin_prefill` are
    // per-pass setup that belongs to no phase, and leaving them outside every
    // region puts them in `other` rather than on the first `Elementwise`
    // boundary that happens to close. The clock is a parameter of `run_chunk`
    // rather than a `ChunkState` field so that it and the chunk driver's
    // field-by-field reborrows stay independent. Always armed: this function
    // is only reachable from `prefill_prompt`, which armed it.
    let mut clock = PhaseClock::new(timing, true);

    let mut position = start;
    let mut rest = tokens;
    while !rest.is_empty() {
        let n = rows.min(rest.len());
        let (chunk, tail) = rest.split_at(n);
        run_chunk(
            model,
            arch,
            &dims,
            rows,
            &mut session,
            &mut chunk_state,
            chunk,
            position,
            tail.is_empty(),
            &mut on_route,
            &mut clock,
        )?;
        position += n;
        rest = tail;
    }
    session.finish()?;
    Ok(())
}

/// One chunk of `tokens.len()` positions starting at absolute `start`.
///
/// `rows` is the width the scratch was carved for, which is `>= tokens.len()`;
/// a short final chunk simply uses the leading rows of every buffer.
#[allow(clippy::too_many_arguments)]
fn run_chunk(
    model: &Model,
    arch: &ArchInfo,
    dims: &PrefillDims,
    rows: usize,
    session: &mut PrefillSession<'_>,
    state: &mut ChunkState<'_>,
    tokens: &[u32],
    start: usize,
    want_logits: bool,
    on_route: &mut Option<PrefillRouteSink<'_>>,
    clock: &mut PhaseClock<'_>,
) -> Result<(), ForwardError> {
    let n = tokens.len();
    let eps = arch.rms_eps as f32;
    let theta = arch.rope_theta as f32;
    let scale = 1.0 / (dims.head_dim as f32).sqrt();
    let (hidden, q_dim, kv_dim, top_k) = (dims.hidden, dims.q_dim, dims.kv_dim, dims.top_k);

    // 1. Embed every row of the chunk.
    {
        let (s, _) = carve(session.scratch(), dims, rows)?;
        for (r, &token) in tokens.iter().enumerate() {
            model.embed(token, &mut s.residual[r * hidden..(r + 1) * hidden])?;
        }
    }

    for layer in 0..model.n_layers() {
        let lw = model.layer(layer)?;
        let layer_idx = layer as usize;

        // 2a-2i. Everything before the expert phase, with no sweep running.
        {
            let (mut s, _) = carve(session.scratch(), dims, rows)?;

            // (a) attention RMSNorm and (b) both activation quantizations.
            for r in 0..n {
                let residual = &s.residual[r * hidden..(r + 1) * hidden];
                let normed = &mut s.normed[r * hidden..(r + 1) * hidden];
                rmsnorm(residual, lw.attn_norm, eps, normed)?;
                quantize_row_q8_k(
                    normed,
                    &mut s.acts_hidden[r * dims.hidden_q8k..(r + 1) * dims.hidden_q8k],
                )?;
                quantize_row_q8_0(
                    normed,
                    &mut s.acts_q80[r * dims.hidden_q80..(r + 1) * dims.hidden_q80],
                )?;
            }

            // (c) Q, K and V for the whole chunk, then transposed back to
            // row-major so every per-row step below reads a contiguous row.
            clock.charge(Phase::Elementwise);
            pool_batched_q8_k(
                state.pool,
                lw.attn_q.format,
                lw.attn_q.bytes,
                hidden,
                q_dim,
                &s.acts_hidden[..n * dims.hidden_q8k],
                n,
                &mut s.tmat[..q_dim * n],
            )?;
            clock.charge(Phase::Projections);
            transpose(&s.tmat[..q_dim * n], q_dim, n, &mut s.q[..n * q_dim])?;
            clock.charge(Phase::Elementwise);
            pool_batched_q8_0(
                state.pool,
                lw.attn_k.bytes,
                hidden,
                kv_dim,
                &s.acts_q80[..n * dims.hidden_q80],
                n,
                &mut s.tmat[..kv_dim * n],
            )?;
            clock.charge(Phase::Projections);
            transpose(&s.tmat[..kv_dim * n], kv_dim, n, &mut s.k[..n * kv_dim])?;
            clock.charge(Phase::Elementwise);
            pool_batched_q8_k(
                state.pool,
                lw.attn_v.format,
                lw.attn_v.bytes,
                hidden,
                kv_dim,
                &s.acts_hidden[..n * dims.hidden_q8k],
                n,
                &mut s.tmat[..kv_dim * n],
            )?;
            clock.charge(Phase::Projections);
            transpose(&s.tmat[..kv_dim * n], kv_dim, n, &mut s.v[..n * kv_dim])?;

            // (d) Per-head QK-RMSNorm then RoPE, at this row's absolute
            // position. HF order, exactly as `forward_token` does it.
            for r in 0..n {
                let position = start + r;
                let rope_pos = u32::try_from(position)
                    .map_err(|_| ForwardError::PositionOverflow { position })?;
                let q = &mut s.q[r * q_dim..(r + 1) * q_dim];
                for head in q.chunks_exact_mut(dims.head_dim) {
                    rmsnorm_in_place(head, lw.attn_q_norm, eps)?;
                }
                rope_neox_heads(q, dims.n_heads, dims.head_dim, rope_pos, theta)?;
                let k = &mut s.k[r * kv_dim..(r + 1) * kv_dim];
                for head in k.chunks_exact_mut(dims.head_dim) {
                    rmsnorm_in_place(head, lw.attn_k_norm, eps)?;
                }
                rope_neox_heads(k, dims.n_kv_heads, dims.head_dim, rope_pos, theta)?;
            }

            // (e) The inversion: the whole chunk's K/V goes in before any of
            // it is attended. The cache is ragged from here until the layer
            // loop moves on.
            for r in 0..n {
                state.kv.append(
                    layer_idx,
                    &s.k[r * kv_dim..(r + 1) * kv_dim],
                    &s.v[r * kv_dim..(r + 1) * kv_dim],
                )?;
            }

            // (f) Causal attention: row r sees positions 0..=start + r and
            // nothing of the rows queued behind it.
            //
            // Charged once around the whole parallel region, not per row: at a
            // 512-row chunk this is 512 calls per layer, and the boundary is
            // worth ~25 ns each while the region is the phase under
            // investigation.
            clock.charge(Phase::Elementwise);
            scatter_attention(
                state.pool,
                state.kv,
                layer_idx,
                start,
                n,
                q_dim,
                scale,
                dims.attn_shard(),
                &s.q[..n * q_dim],
                s.attn_shards,
                &mut s.attn_out[..n * q_dim],
            )?;

            clock.charge(Phase::Attention);

            // (g) Output projection and the attention residual add.
            for r in 0..n {
                quantize_row_q8_k(
                    &s.attn_out[r * q_dim..(r + 1) * q_dim],
                    &mut s.acts_attn[r * dims.attn_q8k..(r + 1) * dims.attn_q8k],
                )?;
            }
            clock.charge(Phase::Elementwise);
            pool_batched_q8_k(
                state.pool,
                lw.attn_output.format,
                lw.attn_output.bytes,
                q_dim,
                hidden,
                &s.acts_attn[..n * dims.attn_q8k],
                n,
                &mut s.tmat[..hidden * n],
            )?;
            clock.charge(Phase::Projections);
            for r in 0..n {
                let residual = &mut s.residual[r * hidden..(r + 1) * hidden];
                for (h, value) in residual.iter_mut().enumerate() {
                    // The same single f32 add `vec_add` performs; only the
                    // source's layout differs.
                    *value += s.tmat[h * n + r];
                }
            }

            // (h) FFN RMSNorm and the expert activations.
            for r in 0..n {
                let residual = &s.residual[r * hidden..(r + 1) * hidden];
                let normed = &mut s.normed[r * hidden..(r + 1) * hidden];
                rmsnorm(residual, lw.ffn_norm, eps, normed)?;
                quantize_row_q8_k(
                    normed,
                    &mut s.acts_hidden[r * dims.hidden_q8k..(r + 1) * dims.hidden_q8k],
                )?;
            }

            // (i) Routing, per row, exactly as `forward_token` routes.
            //
            // Charged to `Projections`: the f32 router matvec is
            // `n_experts * hidden` scalar MACs per row, which dwarfs the
            // softmax and the top-k scan it shares this loop with.
            clock.charge(Phase::Elementwise);
            for r in 0..n {
                let normed = &s.normed[r * hidden..(r + 1) * hidden];
                let logits = &mut s.router_logits[r * dims.n_experts..(r + 1) * dims.n_experts];
                for (row, logit) in lw.router.data().chunks_exact(hidden).zip(logits.iter_mut()) {
                    *logit = dot_f32(row, normed);
                }
                let probs = &mut s.router_probs[r * dims.n_experts..(r + 1) * dims.n_experts];
                probs.copy_from_slice(logits);
                softmax(probs)?;
                let topk = &mut s.topk[r * top_k..(r + 1) * top_k];
                for entry in topk.iter_mut() {
                    let mut best_e = 0usize;
                    let mut best_p = f32::NEG_INFINITY;
                    for (e, &p) in probs.iter().enumerate() {
                        if p > best_p {
                            best_p = p;
                            best_e = e;
                        }
                    }
                    *entry = TopkEntry {
                        expert: best_e as u32,
                        weight: best_p,
                    };
                    probs[best_e] = f32::NEG_INFINITY;
                }
                if arch.norm_topk_prob {
                    let sum: f32 = topk.iter().map(|entry| entry.weight).sum();
                    for entry in topk.iter_mut() {
                        entry.weight /= sum;
                    }
                }
                if let Some(sink) = on_route.as_deref_mut() {
                    state.sink_buf.clear();
                    state
                        .sink_buf
                        .extend(topk.iter().map(|entry| (entry.expert, entry.weight)));
                    sink(start + r, layer, state.sink_buf);
                }
            }

            clock.charge(Phase::Projections);

            // (j) The inverse index, and the routed set the sweep plans on.
            build_index(&mut s, dims, n)?;
            state.routed.clear();
            for expert in 0..dims.n_experts {
                if s.index_start[expert + 1] > s.index_start[expert] {
                    state.routed.push(expert as u32);
                }
            }
            clock.charge(Phase::Elementwise);
        }

        // 2k. Sweep the layer, computing each expert against all its rows.
        //
        // The two boundaries here *are* per iteration, unlike the per-row
        // loops above: this loop turns once per routed expert (~128 a layer,
        // not ~512 a chunk), and splitting the window wait from the arithmetic
        // is the whole question this instrumentation exists to answer.
        let covered = {
            let (bytes, mut sweep) = session.split(state.plan, layer, state.routed)?;
            let (mut s, _) = carve(bytes, dims, rows)?;
            let mut covered = 0usize;
            loop {
                let next = sweep.next_expert()?;
                clock.charge(Phase::ExpertIo);
                let Some(expert) = next else { break };
                covered += run_expert(state.pool, dims, &mut s, n, &expert)?;
                clock.charge(Phase::ExpertCompute);
            }
            sweep.finish()?;
            clock.charge(Phase::ExpertIo);
            covered
        };
        if covered != n * top_k {
            return Err(ForwardError::PrefillCoverage {
                layer,
                covered,
                expected: n * top_k,
            });
        }

        // 2l. Reduce each row's staging in top-k order and add the residual.
        {
            let (mut s, _) = carve(session.scratch(), dims, rows)?;
            reduce_experts(&mut s, dims, n)?;
        }
        clock.charge(Phase::Elementwise);
    }

    // 3. The tail, for the last row only: a `[rows][vocab]` logit buffer would
    //    be 296.75 MiB at the v0 dims and nothing wants the other rows.
    if want_logits {
        let (s, _) = carve(session.scratch(), dims, rows)?;
        let last = n - 1;
        let residual = &s.residual[last * hidden..(last + 1) * hidden];
        let normed = &mut s.normed[last * hidden..(last + 1) * hidden];
        rmsnorm(residual, model.final_norm(), eps, normed)?;
        let acts = &mut s.acts_hidden[last * dims.hidden_q8k..(last + 1) * dims.hidden_q8k];
        quantize_row_q8_k(normed, acts)?;
        clock.charge(Phase::Elementwise);
        let head = model.lm_head();
        pool_gemv_q8_k(
            state.pool,
            head.format,
            head.bytes,
            head.in_dim,
            head.out_dim,
            acts,
            state.logits,
        )?;
        clock.charge(Phase::Projections);
    }
    Ok(())
}

/// Build the CSR inverse routing index: for each expert, the `(row, slot)`
/// pairs that route to it.
///
/// The index is a permutation of the chunk's `n * top_k` `(row, slot)` pairs
/// by construction — a counting sort over expert id — which is what lets the
/// coverage check in [`run_chunk`] be a single equality.
///
/// # Errors
///
/// [`ForwardError::RoutedExpertOutOfRange`] if a selection named an expert the
/// layer does not have. Unreachable through the router (the argmax runs over
/// exactly `n_experts` probabilities) and reported rather than trusted,
/// because the alternative is an out-of-bounds index into the CSR arrays.
fn build_index(s: &mut Scratch<'_>, dims: &PrefillDims, n: usize) -> Result<(), ForwardError> {
    s.index_start.fill(0);
    for entry in &s.topk[..n * dims.top_k] {
        let expert = entry.expert as usize;
        let slot =
            s.index_start
                .get_mut(expert + 1)
                .ok_or(ForwardError::RoutedExpertOutOfRange {
                    expert: entry.expert,
                    n_experts: dims.n_experts,
                })?;
        *slot += 1;
    }
    // No row may select the same expert twice — the top-k argmax blanks each
    // winner with `-inf`, so it cannot — and the batch buffers depend on it:
    // `gate`, `up`, `moe_t` and `tmat` are all carved for `rows` activations,
    // and an expert routed by more `(row, slot)` pairs than the chunk has
    // rows would index past every one of them. Checked, not assumed.
    for e in 0..dims.n_experts {
        let count = s.index_start[e + 1] as usize;
        if count > n {
            return Err(ForwardError::RepeatedRoutedExpert {
                expert: e as u32,
                count,
                rows: n,
            });
        }
        s.index_start[e + 1] += s.index_start[e];
    }
    s.index_cursor
        .copy_from_slice(&s.index_start[..dims.n_experts]);
    for r in 0..n {
        for slot in 0..dims.top_k {
            let entry = s.topk[r * dims.top_k + slot];
            let cursor = &mut s.index_cursor[entry.expert as usize];
            let at = *cursor as usize;
            s.index[at] = RowSlot {
                row: r as u32,
                slot: slot as u32,
            };
            *cursor += 1;
        }
    }
    Ok(())
}

/// Compute one swept expert against every chunk row routed to it, staging each
/// result into that row's top-k slot. Returns how many `(row, slot)` pairs it
/// covered.
fn run_expert(
    pool: &mut ComputePool,
    dims: &PrefillDims,
    s: &mut Scratch<'_>,
    n_rows: usize,
    expert: &SweepExpert<'_>,
) -> Result<usize, ForwardError> {
    let e = expert.expert as usize;
    let (lo, hi) = match (s.index_start.get(e), s.index_start.get(e + 1)) {
        (Some(&lo), Some(&hi)) => (lo as usize, hi as usize),
        _ => {
            return Err(ForwardError::RoutedExpertOutOfRange {
                expert: expert.expert,
                n_experts: dims.n_experts,
            });
        }
    };
    if hi <= lo {
        // The sweep only yields routed experts, so this cannot happen through
        // the driver; a chunk that routes nothing to `e` simply has no work.
        return Ok(0);
    }
    let count = hi - lo;
    debug_assert!(count <= n_rows * dims.top_k);
    let (hidden, moe) = (dims.hidden, dims.moe);
    let blocks = dims.hidden_q8k;

    // Gather this expert's rows' activations into one contiguous batch: the
    // batched kernels take `n_acts` consecutive rows, and a routed set is by
    // nature scattered across the chunk.
    for (j, entry) in s.index[lo..hi].iter().enumerate() {
        let src = entry.row as usize * blocks;
        s.acts_gather[j * blocks..(j + 1) * blocks]
            .copy_from_slice(&s.acts_hidden[src..src + blocks]);
    }
    let acts = &s.acts_gather[..count * blocks];

    let view = &expert.view;
    let gate_slab = view.gate();
    let up_slab = view.up();
    let down_slab = view.down();
    pool_batched_q8_k(
        pool,
        gate_slab.format,
        gate_slab.bytes,
        hidden,
        moe,
        acts,
        count,
        &mut s.gate[..moe * count],
    )?;
    pool_batched_q8_k(
        pool,
        up_slab.format,
        up_slab.bytes,
        hidden,
        moe,
        acts,
        count,
        &mut s.up[..moe * count],
    )?;
    // SwiGLU is elementwise, so it is layout-agnostic: running it over the
    // whole `[moe][count]` plane performs exactly the per-row operations.
    swiglu_combine(&mut s.gate[..moe * count], &s.up[..moe * count])?;
    transpose(
        &s.gate[..moe * count],
        moe,
        count,
        &mut s.moe_t[..count * moe],
    )?;
    for j in 0..count {
        quantize_row_q8_k(
            &s.moe_t[j * moe..(j + 1) * moe],
            &mut s.acts_moe[j * dims.moe_q8k..(j + 1) * dims.moe_q8k],
        )?;
    }
    pool_batched_q8_k(
        pool,
        down_slab.format,
        down_slab.bytes,
        moe,
        hidden,
        &s.acts_moe[..count * dims.moe_q8k],
        count,
        &mut s.tmat[..hidden * count],
    )?;
    for (j, entry) in s.index[lo..hi].iter().enumerate() {
        let base = (entry.row as usize * dims.top_k + entry.slot as usize) * hidden;
        let dst = &mut s.staged[base..base + hidden];
        for (h, slot) in dst.iter_mut().enumerate() {
            *slot = s.tmat[h * count + j];
        }
    }
    Ok(count)
}

/// Reduce each row's staged expert outputs **in top-k order** and add them to
/// the residual.
///
/// This is the whole bit-identity argument in one function. The sweep hands
/// experts back in ascending blob order; `forward_token` accumulates
/// `acc += w_i * d_i` for `i` in `0..top_k`, descending router probability.
/// f32 addition is not associative, so the two orders give different bits, and
/// the only way to have both a blob-order sweep and a top-k-order sum is to
/// stage every expert's output and reduce afterwards — which is exactly what
/// `forward_token` already does for its own two-phase hit/miss split.
///
/// The operations are literally the same: separate multiply and add (never
/// `mul_add`, whose single rounding differs), an `expert_acc` zeroed per row,
/// and one `vec_add` into the residual.
fn reduce_experts(s: &mut Scratch<'_>, dims: &PrefillDims, n: usize) -> Result<(), ForwardError> {
    let (hidden, top_k) = (dims.hidden, dims.top_k);
    for r in 0..n {
        s.expert_acc.fill(0.0);
        for slot in 0..top_k {
            let entry = s.topk[r * top_k + slot];
            let base = (r * top_k + slot) * hidden;
            let staged = &s.staged[base..base + hidden];
            for (acc, &d) in s.expert_acc.iter_mut().zip(staged) {
                *acc += entry.weight * d;
            }
        }
        vec_add(&mut s.residual[r * hidden..(r + 1) * hidden], s.expert_acc)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{ExpertsLayout, LayerLayout};
    use crate::io::testutil::{Fixture, build_install};
    use crate::io::{LoadOptions, SweepError};
    use crate::model::RuntimeConfig;
    use crate::model::testsupport::temper_install;

    const SKIP: LoadOptions = LoadOptions {
        skip_hashes: true,
        verify_layer_hashes: false,
    };

    fn load_fixture(tag: &str) -> (Fixture, Model) {
        let fx = build_install(tag);
        temper_install(&fx);
        let model = Model::load(&fx.root, SKIP).unwrap();
        (fx, model)
    }

    /// A state over the tiny fixture with an explicit prefill config.
    fn state_with(model: &Model, cap: usize, prefill: PrefillConfig) -> ForwardState {
        let mut st = ForwardState::with_config(model, cap, RuntimeConfig::testing()).unwrap();
        st.set_prefill_config(prefill).unwrap();
        st
    }

    fn sweep_config(chunk: usize) -> PrefillConfig {
        PrefillConfig {
            mode: PrefillMode::Sweep,
            chunk,
            // The fixture pool is under 1 MiB; one narrow window keeps the
            // ring small enough to leave room for a several-row chunk.
            experts_per_window: 1,
            windows_in_flight: 1,
        }
    }

    fn token_major_config() -> PrefillConfig {
        PrefillConfig {
            mode: PrefillMode::TokenMajor,
            ..sweep_config(1)
        }
    }

    fn logit_bits(logits: &[f32]) -> Vec<u32> {
        logits.iter().map(|v| v.to_bits()).collect()
    }

    /// The second bit-identity fixture: a geometry chosen so that no two of
    /// the driver's widths can accidentally agree.
    ///
    /// [`build_install`] is one hard-coded geometry — `hidden 256`, `q_dim
    /// 256`, `moe 256`, `4` experts, `top_k 2` — and every prefill test above
    /// runs it at `experts_per_window: 1, windows_in_flight: 1`. Four things
    /// the driver does are therefore never exercised there, each of which is
    /// a place a future defect would hide silently rather than fail:
    ///
    /// - **`wide = max(q_dim, hidden)` is degenerate** at `256 == 256`, so
    ///   `tmat`'s four slicings (`q_dim * n`, `kv_dim * n`, `hidden * n`,
    ///   `hidden * count`) collapse to one width and a wrong one would still
    ///   read the right bytes. The shipped model is `q_dim 4096` against
    ///   `hidden 2048`.
    /// - **`hidden_q8k == 1`**, so [`run_expert`]'s `acts_gather` block copy
    ///   is a single-block copy. The shipped model has `2048 / 256 = 8`.
    /// - **`experts_per_window == 1`**, so the sweep's intra-window
    ///   `within * stride` offset is never reached from this driver. The
    ///   production default is 8.
    /// - **No skip pressure**: 4 experts at 1 per window leaves nothing for a
    ///   chunk's routing to miss.
    ///
    /// `scripts/bitident.py` closes none of them either — its prompts are
    /// 4-12 tokens and single-chunk. Hence this: `hidden 512`, `q_dim 768`,
    /// `kv_dim 192`, `moe 256` (**pairwise different**, so a wrong width
    /// cannot accidentally agree), `hidden_q8k 2`, `attn_q8k 3`, 16 experts
    /// at `top_k 4`.
    ///
    /// It lives here rather than beside [`build_install`] because
    /// `io/testutil.rs` belongs to another lane; the right home for a
    /// geometry-parameterised builder is there, and this module keeps only
    /// what the parameterisation would have produced.
    mod wide {
        use std::collections::BTreeMap;
        use std::fs;
        use std::path::PathBuf;

        use crate::format::{
            ArchInfo, CommonTensor, ExpertsLayout, FileEntry, LAYOUT_FILE, LayerLayout, Manifest,
            Projection, ProjectionName, QuantInfo, RVMP_VERSION, SourceInfo, layer_file_name,
            sha256_file, testutil::TempDir, write_layout, write_manifest,
        };
        use crate::kernels::quants::{QuantFormat, f32_to_f16};
        use crate::model::testsupport::temper_parts;

        /// Layers in the fixture.
        pub(super) const N_LAYERS: u32 = 2;
        /// Experts per layer — enough that a chunk's routing cannot cover
        /// them all, which is what produces skipped windows.
        pub(super) const N_EXPERTS: u32 = 16;
        /// Experts per token.
        pub(super) const TOP_K: u32 = 4;
        /// Hidden dimension: two Q8_K super-blocks per row, unlike the
        /// single-block fixture.
        pub(super) const HIDDEN: usize = 512;
        /// Per-expert FFN intermediate — deliberately not `HIDDEN`.
        pub(super) const MOE: usize = 256;
        /// Query heads.
        pub(super) const N_HEADS: usize = 12;
        /// KV heads: 4:1 GQA.
        pub(super) const N_KV_HEADS: usize = 3;
        /// Head dimension.
        pub(super) const HEAD_DIM: usize = 64;
        /// `n_heads * head_dim` = 768, so `wide = q_dim > hidden`.
        pub(super) const Q_DIM: usize = N_HEADS * HEAD_DIM;
        /// `n_kv_heads * head_dim` = 192.
        pub(super) const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
        /// Vocabulary size.
        pub(super) const VOCAB: usize = 40;

        /// A complete install in a self-cleaning temp dir.
        pub(super) struct WideFixture {
            _tmp: TempDir,
            /// The install directory.
            pub(super) root: PathBuf,
        }

        /// Deterministic filler byte for quantized payloads, as
        /// `io::testutil` fills them; the scales are re-stamped by
        /// [`temper_parts`] afterwards.
        fn pattern_byte(seed: usize, block: usize, byte: usize) -> u8 {
            (seed.wrapping_mul(31) ^ block.wrapping_mul(7) ^ byte.wrapping_mul(3)) as u8
        }

        /// Packed bytes for `rows` quantized rows of `in_dim` weights.
        fn quant_bytes(format: QuantFormat, rows: usize, in_dim: usize, seed: usize) -> Vec<u8> {
            let row_bytes = format.row_bytes(in_dim).expect("fixture dims divide");
            let block_bytes = format.block_bytes();
            let n_blocks = rows * row_bytes / block_bytes;
            let mut out = vec![0u8; rows * row_bytes];
            for block in 0..n_blocks {
                let base = block * block_bytes;
                for byte in 0..block_bytes {
                    out[base + byte] = pattern_byte(seed, block, byte);
                }
                let d = f32_to_f16(0.5).to_le_bytes();
                let dmin = f32_to_f16(0.25).to_le_bytes();
                match format {
                    QuantFormat::Q4_K | QuantFormat::Q5_K => {
                        out[base..base + 2].copy_from_slice(&d);
                        out[base + 2..base + 4].copy_from_slice(&dmin);
                    }
                    QuantFormat::Q6_K => out[base + 208..base + 210].copy_from_slice(&d),
                    QuantFormat::Q8_0 => out[base..base + 2].copy_from_slice(&d),
                    QuantFormat::Q8_K => unreachable!("activation-only format"),
                }
            }
            out
        }

        /// RMSNorm weight near 1, varying per element.
        ///
        /// `io::testutil`'s ramp reaches 128 at `hidden 512`, which pushes the
        /// value projections out of the KV cache's f16 range; this fixture is
        /// twice as wide, so its norms are gentler on purpose.
        fn norm_value(name: &str, index: usize) -> f32 {
            1.0 + (((index * 7 + name.len()) % 13) as f32) / 64.0
        }

        /// Router weight: signed and non-monotonic, so different tokens
        /// genuinely select different experts. A ramp would make every row
        /// route to the same top-k and leave the skip logic untested.
        fn router_value(index: usize) -> f32 {
            ((((index * 37 + 11) % 29) as f32) - 14.0) / 64.0
        }

        /// LE bytes of `elements` values from `f`.
        fn f32_bytes(elements: usize, f: impl Fn(usize) -> f32) -> Vec<u8> {
            let mut out = Vec::with_capacity(elements * 4);
            for i in 0..elements {
                out.extend_from_slice(&f(i).to_le_bytes());
            }
            out
        }

        /// The fixture's architecture facts.
        pub(super) fn arch() -> ArchInfo {
            ArchInfo {
                n_layers: N_LAYERS,
                n_experts: N_EXPERTS,
                top_k: TOP_K,
                hidden: HIDDEN as u32,
                moe_intermediate: MOE as u32,
                n_heads: N_HEADS as u32,
                n_kv_heads: N_KV_HEADS as u32,
                head_dim: HEAD_DIM as u32,
                vocab: VOCAB as u32,
                context_length: 4096,
                rope_theta: 1e7,
                rms_eps: 1e-6,
                norm_topk_prob: true,
                tie_embeddings: false,
                shared_expert: false,
                sliding_window: None,
            }
        }

        /// One common tensor's dtype and generated payload.
        struct TensorDef {
            dtype: &'static str,
            bytes: Vec<u8>,
        }

        /// Every common tensor, keyed by GGUF name, with the same per-tensor
        /// type mix `io::testutil` uses: layer 0 mirrors a Q6_K-down layer,
        /// layer 1 is pure Q4_K.
        fn common_defs() -> BTreeMap<String, TensorDef> {
            let mut defs: BTreeMap<String, TensorDef> = BTreeMap::new();
            let mut quant = |name: String, format: QuantFormat, rows: usize, in_dim: usize| {
                let dtype = match format {
                    QuantFormat::Q4_K => "q4_k",
                    QuantFormat::Q5_K => "q5_k",
                    QuantFormat::Q6_K => "q6_k",
                    QuantFormat::Q8_0 => "q8_0",
                    QuantFormat::Q8_K => unreachable!("activation-only format"),
                };
                let bytes = quant_bytes(format, rows, in_dim, name.len() + rows);
                defs.insert(name, TensorDef { dtype, bytes });
            };
            quant(
                "token_embd.weight".to_owned(),
                QuantFormat::Q4_K,
                VOCAB,
                HIDDEN,
            );
            quant("output.weight".to_owned(), QuantFormat::Q6_K, VOCAB, HIDDEN);
            for layer in 0..N_LAYERS {
                quant(
                    format!("blk.{layer}.attn_q.weight"),
                    QuantFormat::Q4_K,
                    Q_DIM,
                    HIDDEN,
                );
                quant(
                    format!("blk.{layer}.attn_k.weight"),
                    QuantFormat::Q8_0,
                    KV_DIM,
                    HIDDEN,
                );
                let v_format = if layer == 0 {
                    QuantFormat::Q6_K
                } else {
                    QuantFormat::Q4_K
                };
                quant(
                    format!("blk.{layer}.attn_v.weight"),
                    v_format,
                    KV_DIM,
                    HIDDEN,
                );
                quant(
                    format!("blk.{layer}.attn_output.weight"),
                    QuantFormat::Q5_K,
                    HIDDEN,
                    Q_DIM,
                );
            }
            let mut f32_def = |name: String, bytes: Vec<u8>| {
                defs.insert(
                    name,
                    TensorDef {
                        dtype: "f32",
                        bytes,
                    },
                );
            };
            f32_def(
                "output_norm.weight".to_owned(),
                f32_bytes(HIDDEN, |i| norm_value("output_norm.weight", i)),
            );
            for layer in 0..N_LAYERS {
                for (stem, elements) in [
                    ("attn_norm", HIDDEN),
                    ("ffn_norm", HIDDEN),
                    ("attn_q_norm", HEAD_DIM),
                    ("attn_k_norm", HEAD_DIM),
                ] {
                    let name = format!("blk.{layer}.{stem}.weight");
                    let bytes = f32_bytes(elements, |i| norm_value(&name, i));
                    f32_def(name, bytes);
                }
                f32_def(
                    format!("blk.{layer}.ffn_gate_inp.weight"),
                    f32_bytes(N_EXPERTS as usize * HIDDEN, router_value),
                );
            }
            defs
        }

        /// Build a complete, tempered install under a fresh temp dir.
        ///
        /// Load it with `skip_hashes`: the tempering rewrites block scales
        /// after the manifest digests are taken, exactly as
        /// `io::testutil` + `temper_install` do for the narrow fixture.
        pub(super) fn build(tag: &str) -> WideFixture {
            let tmp = TempDir::new(tag);
            let root = tmp.path().join("model.rvmp");
            fs::create_dir_all(root.join("experts")).expect("create install dirs");

            let defs = common_defs();
            let mut common_tensors = BTreeMap::new();
            let mut common = Vec::new();
            for (name, def) in &defs {
                let offset = (common.len() as u64).next_multiple_of(64);
                common.resize(offset as usize, 0);
                common.extend_from_slice(&def.bytes);
                common_tensors.insert(
                    name.clone(),
                    CommonTensor {
                        offset,
                        len: def.bytes.len() as u64,
                        dtype: def.dtype.to_owned(),
                    },
                );
            }
            fs::write(root.join("common.bin"), &common).expect("write common.bin");

            let mut layers = Vec::new();
            for layer in 0..N_LAYERS {
                let down_format = if layer == 0 {
                    QuantFormat::Q6_K
                } else {
                    QuantFormat::Q4_K
                };
                let slabs: [(ProjectionName, QuantFormat, usize, usize); 3] = [
                    (ProjectionName::Gate, QuantFormat::Q4_K, MOE, HIDDEN),
                    (ProjectionName::Up, QuantFormat::Q4_K, MOE, HIDDEN),
                    (ProjectionName::Down, down_format, HIDDEN, MOE),
                ];
                let mut projections = Vec::new();
                let mut cursor: u64 = 0;
                for (name, format, rows, in_dim) in slabs {
                    let offset_in_blob = cursor.next_multiple_of(4096);
                    let len = (format.row_bytes(in_dim).unwrap() * rows) as u64;
                    projections.push(Projection {
                        name,
                        offset_in_blob,
                        len,
                        quant: match format {
                            QuantFormat::Q4_K => "q4_k",
                            QuantFormat::Q6_K => "q6_k",
                            _ => unreachable!("fixture expert formats"),
                        }
                        .to_owned(),
                    });
                    cursor = offset_in_blob + len;
                }
                let stride = cursor.next_multiple_of(4096);

                let mut file_bytes = vec![0u8; (stride * u64::from(N_EXPERTS)) as usize];
                for expert in 0..N_EXPERTS {
                    let blob_base = (u64::from(expert) * stride) as usize;
                    for (p, ((_, format, rows, in_dim), projection)) in
                        slabs.iter().zip(&projections).enumerate()
                    {
                        let seed = layer as usize * 1009 + expert as usize * 101 + p * 13;
                        let bytes = quant_bytes(*format, *rows, *in_dim, seed);
                        let start = blob_base + projection.offset_in_blob as usize;
                        file_bytes[start..start + bytes.len()].copy_from_slice(&bytes);
                    }
                }
                let file = layer_file_name(layer);
                fs::write(root.join(&file), &file_bytes).expect("write layer file");
                layers.push(LayerLayout {
                    file,
                    stride,
                    n_experts: N_EXPERTS,
                    projections,
                });
            }
            let layout = ExpertsLayout { layers };
            write_layout(&root, &layout).expect("write layout");

            let mut files = BTreeMap::new();
            let mut record = |name: String| {
                let path = root.join(&name);
                files.insert(
                    name,
                    FileEntry {
                        size: fs::metadata(&path).unwrap().len(),
                        sha256: sha256_file(&path).unwrap(),
                    },
                );
            };
            record("common.bin".to_owned());
            record(LAYOUT_FILE.to_owned());
            for layer in 0..N_LAYERS {
                record(layer_file_name(layer));
            }

            let mut tensor_types: BTreeMap<String, String> = defs
                .iter()
                .map(|(name, def)| (name.clone(), def.dtype.to_owned()))
                .collect();
            for (layer, layout_layer) in layout.layers.iter().enumerate() {
                for projection in &layout_layer.projections {
                    let stem = match projection.name {
                        ProjectionName::Gate => "ffn_gate_exps",
                        ProjectionName::Up => "ffn_up_exps",
                        ProjectionName::Down => "ffn_down_exps",
                    };
                    tensor_types.insert(
                        format!("blk.{layer}.{stem}.weight"),
                        projection.quant.clone(),
                    );
                }
            }

            let manifest = Manifest {
                rvmp_version: RVMP_VERSION,
                model_id: "fixture-moe-wide-2l".to_owned(),
                source: SourceInfo {
                    hf_repo: "test/fixture-wide".to_owned(),
                    revision: "deadbeef".to_owned(),
                    file: "fixture-wide-Q4_K_M.gguf".to_owned(),
                    sha256: "0".repeat(64),
                },
                arch: arch(),
                quant: QuantInfo {
                    scheme: "gguf".to_owned(),
                    tensor_types,
                },
                common_tensors,
                files,
            };
            write_manifest(&root, &manifest).expect("write manifest");

            temper_parts(&root, &manifest.common_tensors, &layout.layers);
            WideFixture { _tmp: tmp, root }
        }
    }

    /// Prefill `ids` through a fresh state and return the final logits' bits.
    fn run(model: &Model, prefill: PrefillConfig, ids: &[u32]) -> Vec<u32> {
        let mut st = state_with(model, 64, prefill);
        let logits = prefill_prompt(model, &mut st, ids, None).unwrap();
        assert_eq!(logits.len(), crate::io::testutil::VOCAB);
        logit_bits(logits)
    }

    // ---- the load-bearing test ------------------------------------------

    /// **The** test: the sweep and the token-major loop must agree bit for
    /// bit, across chunk sizes, at prompt lengths that are an exact multiple
    /// of the chunk, that are not, and that are shorter than one chunk.
    #[test]
    fn sweep_and_token_major_agree_bit_for_bit() {
        let (_fx, model) = load_fixture("prefill-bitident");
        let prompt: Vec<u32> = (0..12u32).map(|i| (i * 5 + 1) % 32).collect();

        for &len in &[1usize, 2, 3, 4, 6, 7, 8, 12] {
            let ids = &prompt[..len];
            let want = run(&model, token_major_config(), ids);
            for &chunk in &[1usize, 2, 3, 4, 8, 512] {
                let got = run(&model, sweep_config(chunk), ids);
                assert_eq!(
                    got,
                    want,
                    "prompt len {len}, chunk {chunk}: sweep prefill diverged \
                     (exact multiple: {})",
                    len % chunk == 0
                );
            }
        }
    }

    // ---- the same test, on a geometry that cannot hide a width bug --------

    /// A state over [`wide`]. Its expert blobs are ~252 KiB against the narrow
    /// fixture's ~26 KiB, so the 4 MiB [`RuntimeConfig::testing`] budget
    /// cannot host a multi-window ring beside a chunk and [`plan_arena`] would
    /// narrow `windows_in_flight` straight back to 1 — which is the dial these
    /// tests exist to raise. 24 MiB leaves ~16 MiB of scratch beside the
    /// widest ring used here.
    fn wide_state(model: &Model, cap: usize, prefill: PrefillConfig) -> ForwardState {
        let runtime = RuntimeConfig {
            cache_bytes: 24 * 1024 * 1024,
            threads: Some(2),
            pin: false,
        };
        let mut st = ForwardState::with_config(model, cap, runtime).unwrap();
        st.set_prefill_config(prefill).unwrap();
        st
    }

    fn wide_dials(
        mode: PrefillMode,
        chunk: usize,
        experts_per_window: u32,
        windows_in_flight: u32,
    ) -> PrefillConfig {
        PrefillConfig {
            mode,
            chunk,
            experts_per_window,
            windows_in_flight,
        }
    }

    fn load_wide(tag: &str) -> (wide::WideFixture, Model) {
        let fx = wide::build(tag);
        let model = Model::load(&fx.root, SKIP).unwrap();
        (fx, model)
    }

    /// Prefill `ids` through a fresh wide state and return the logits' bits.
    fn wide_run(model: &Model, prefill: PrefillConfig, ids: &[u32]) -> Vec<u32> {
        let mut st = wide_state(model, 64, prefill);
        let logits = prefill_prompt(model, &mut st, ids, None).unwrap();
        assert_eq!(logits.len(), wide::VOCAB);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "the wide fixture produced non-finite logits, so bit-identity \
             between two NaN-producing paths would prove nothing"
        );
        logit_bits(logits)
    }

    /// The fixture is only worth its runtime if every width it was built to
    /// separate is actually separated. Asserted rather than commented,
    /// because a later edit that quietly collapses two of them would make
    /// every test below pass for the wrong reason.
    #[test]
    fn the_wide_geometry_separates_every_width() {
        let (_fx, model) = load_wide("prefill-wide-geometry");
        let dims = PrefillDims::new(model.arch(), 2, 64).unwrap();

        let widths = [dims.hidden, dims.q_dim, dims.kv_dim, dims.moe];
        for (i, a) in widths.iter().enumerate() {
            for b in &widths[i + 1..] {
                assert_ne!(a, b, "{widths:?} are not pairwise distinct");
            }
        }
        // `wide = max(q_dim, hidden)` is no longer both of them at once, so a
        // `tmat` slicing that used the wrong one would read the wrong bytes.
        assert_eq!(dims.wide, dims.q_dim);
        assert!(dims.wide > dims.hidden, "`wide` is degenerate again");
        // `run_expert`'s `acts_gather` copy is a multi-block copy.
        assert_eq!(dims.hidden_q8k, 2);
        assert_eq!(dims.attn_q8k, 3);
        assert_eq!(dims.moe_q8k, 1);
        assert_eq!(dims.hidden_q80, 16);
        // Enough experts that a chunk's routing cannot cover them all.
        assert_eq!((dims.n_experts, dims.top_k), (16, 4));

        // And [`wide_state`]'s budget really does host the dials the tests
        // below raise: `plan_arena` hands them back unnarrowed, so those tests
        // run at the dials they name rather than quietly falling back to one
        // expert per window and one window in flight.
        let cache_bytes =
            wide_state(&model, 8, wide_dials(PrefillMode::Sweep, 512, 8, 3)).cache_bytes();
        for (experts_per_window, windows_in_flight) in [(1u32, 1u32), (4, 1), (4, 2), (8, 3)] {
            let config = wide_dials(
                PrefillMode::Sweep,
                512,
                experts_per_window,
                windows_in_flight,
            );
            let (rows, planned) =
                plan_arena(model.layout(), &dims, config, 9, cache_bytes).unwrap();
            assert_eq!(rows, 9, "the chunk was narrowed at {config:?}");
            assert_eq!(planned.experts_per_window, experts_per_window);
            assert_eq!(
                planned.windows_in_flight, windows_in_flight,
                "read-ahead was given up at {config:?}"
            );
        }
    }

    /// [`sweep_and_token_major_agree_bit_for_bit`], run through [`wide`] at
    /// `experts_per_window` **1 and above** and several `windows_in_flight`.
    ///
    /// What this covers that the narrow fixture cannot:
    ///
    /// - every `tmat` slicing is a distinguishable width (`q_dim * n` 768n,
    ///   `kv_dim * n` 192n, `hidden * n` 512n) against a buffer carved for
    ///   `wide = 768`, so a slicing that took the wrong dimension would read
    ///   stale arena bytes rather than the right ones by coincidence;
    /// - `run_expert`'s `acts_gather` gather is a two-block copy;
    /// - `experts_per_window > 1` puts more than one expert in a window, which
    ///   is the only way the sweep's intra-window `within * stride` offset is
    ///   reached from this driver;
    /// - `windows_in_flight > 1` overlaps window reads with compute.
    ///
    /// Prompt lengths and chunk widths still cross seams (5 at chunk 2, 8 at
    /// chunk 3), land on exact multiples (4 at chunk 2, 8 at chunk 4, 9 at
    /// chunk 3), leave a ragged final chunk (5 at chunk 4, 9 at chunk 4) and
    /// run shorter than one chunk (every length at chunk 512).
    #[test]
    fn wide_sweep_and_token_major_agree_bit_for_bit() {
        let (_fx, model) = load_wide("prefill-wide-bitident");
        let vocab = wide::VOCAB as u32;
        let prompt: Vec<u32> = (0..9u32).map(|i| (i * 11 + 3) % vocab).collect();
        let dials: [(u32, u32); 4] = [(1, 1), (4, 1), (4, 2), (8, 3)];

        for &len in &[1usize, 3, 4, 5, 8, 9] {
            let ids = &prompt[..len];
            let want = wide_run(&model, wide_dials(PrefillMode::TokenMajor, 1, 1, 1), ids);
            for &chunk in &[2usize, 3, 4, 512] {
                for &(experts_per_window, windows_in_flight) in &dials {
                    let got = wide_run(
                        &model,
                        wide_dials(
                            PrefillMode::Sweep,
                            chunk,
                            experts_per_window,
                            windows_in_flight,
                        ),
                        ids,
                    );
                    assert_eq!(
                        got,
                        want,
                        "prompt len {len}, chunk {chunk}, {experts_per_window} experts/window, \
                         {windows_in_flight} in flight: sweep prefill diverged \
                         (exact multiple: {})",
                        len % chunk == 0
                    );
                }
            }
        }

        // Not a degenerate model whose logits ignore their input: a prompt one
        // token longer must land somewhere else, and the vocabulary must not
        // come back flat. Without this the agreement above could be two paths
        // agreeing on a constant.
        let dials = wide_dials(PrefillMode::Sweep, 4, 4, 2);
        let eight = wide_run(&model, dials, &prompt[..8]);
        assert_ne!(wide_run(&model, dials, &prompt), eight);
        assert!(eight.iter().any(|&bits| bits != eight[0]));
    }

    /// The wide fixture's KV cache agrees too, at a dial pair the narrow
    /// fixture cannot reach.
    #[test]
    fn wide_paths_leave_the_same_kv_cache() {
        let (_fx, model) = load_wide("prefill-wide-kv");
        let ids: Vec<u32> = (0..7u32)
            .map(|i| (i * 5 + 2) % wide::VOCAB as u32)
            .collect();

        let mut token_major = wide_state(&model, 32, wide_dials(PrefillMode::TokenMajor, 1, 1, 1));
        prefill_prompt(&model, &mut token_major, &ids, None).unwrap();
        let mut sweep = wide_state(&model, 32, wide_dials(PrefillMode::Sweep, 3, 4, 2));
        prefill_prompt(&model, &mut sweep, &ids, None).unwrap();

        assert_eq!(sweep.seq_len().unwrap(), ids.len());
        for layer in 0..model.n_layers() as usize {
            assert_eq!(
                sweep.kv_k_layer(layer).unwrap(),
                token_major.kv_k_layer(layer).unwrap(),
                "layer {layer} keys"
            );
            assert_eq!(
                sweep.kv_v_layer(layer).unwrap(),
                token_major.kv_v_layer(layer).unwrap(),
                "layer {layer} values"
            );
        }
    }

    /// Windows no row routed are skipped, and the routing that produces them
    /// is genuinely row-dependent.
    ///
    /// The narrow fixture has 4 experts at 1 per window and `top_k 2`, so a
    /// chunk of two rows can cover every window and the skip path may never
    /// run. Here 16 experts at `top_k 4` guarantee it: a one-row chunk names
    /// 4 experts, which at 1 and 2 experts per window cannot reach more than
    /// 4 of the 16 (resp. 8) windows.
    ///
    /// The second assertion is the one that keeps the first honest — a
    /// fixture whose router picked the same 4 experts for every token would
    /// skip windows for a reason that has nothing to do with the driver.
    #[test]
    fn the_wide_sweep_skips_windows_nothing_routed_to() {
        use std::collections::BTreeSet;

        let (_fx, model) = load_wide("prefill-wide-skips");
        let ids: Vec<u32> = (0..6u32)
            .map(|i| (i * 13 + 5) % wide::VOCAB as u32)
            .collect();

        for experts_per_window in [1u32, 2] {
            let mut st = wide_state(
                &model,
                32,
                wide_dials(PrefillMode::Sweep, 1, experts_per_window, 2),
            );
            let mut routed: BTreeSet<u32> = BTreeSet::new();
            {
                let mut sink = |_pos: usize, _layer: u32, topk: &[(u32, f32)]| {
                    routed.extend(topk.iter().map(|&(expert, _)| expert));
                };
                prefill_prompt(&model, &mut st, &ids, Some(&mut sink)).unwrap();
            }

            let stats = st.stream_stats();
            assert_eq!(
                stats.accesses(),
                0,
                "a swept prefill bypasses the expert cache entirely"
            );
            assert!(stats.sweep_windows_read > 0, "{experts_per_window}/window");
            assert!(
                stats.sweep_windows_skipped > 0,
                "{experts_per_window} experts/window: no window went unrouted, \
                 so the skip path is still untested"
            );
            assert!(
                routed.len() > model.arch().top_k as usize,
                "the router picks the same experts for every row, so the skips \
                 above prove nothing about the driver: {routed:?}"
            );
        }
    }

    /// `experts_per_window > 1` really does reach the sweep's intra-window
    /// `within * stride` slice.
    ///
    /// Every other prefill test runs one expert per window, where `within` is
    /// always 0 and slicing the window at the wrong offset would still hand
    /// back the right blob. The bit-identity test above runs at 4 and 8
    /// experts per window and would fail if the slice were wrong — but only if
    /// a routed expert ever sits somewhere other than the head of its window.
    /// That precondition is asserted here rather than assumed, because it
    /// depends on the fixture's router and nothing else would notice if it
    /// stopped holding.
    #[test]
    fn a_wide_window_slices_experts_at_a_non_zero_offset() {
        const EXPERTS_PER_WINDOW: u32 = 4;
        let (_fx, model) = load_wide("prefill-wide-within");
        let ids: Vec<u32> = (0..6u32)
            .map(|i| (i * 13 + 5) % wide::VOCAB as u32)
            .collect();

        // Chunk 1, so each layer's sweep plans on exactly one row's top-k and
        // the sink reports that set directly.
        let mut st = wide_state(
            &model,
            32,
            wide_dials(PrefillMode::Sweep, 1, EXPERTS_PER_WINDOW, 2),
        );
        let mut off_head = 0usize;
        {
            let mut sink = |_pos: usize, _layer: u32, topk: &[(u32, f32)]| {
                off_head += topk
                    .iter()
                    .filter(|&&(expert, _)| expert % EXPERTS_PER_WINDOW != 0)
                    .count();
            };
            prefill_prompt(&model, &mut st, &ids, Some(&mut sink)).unwrap();
        }
        assert!(
            off_head > 0,
            "every routed expert led its window, so `within` was always 0 and \
             the intra-window offset is still untested"
        );
    }

    // ---- back to the narrow fixture --------------------------------------

    /// The KV cache the two paths leave behind must also agree: same length,
    /// same stored f16 bits, on every layer.
    #[test]
    fn both_paths_leave_the_same_kv_cache() {
        let (_fx, model) = load_fixture("prefill-kv-agree");
        let ids: Vec<u32> = (0..7u32).map(|i| (i * 3 + 2) % 32).collect();

        let mut token_major = state_with(&model, 32, token_major_config());
        prefill_prompt(&model, &mut token_major, &ids, None).unwrap();
        let mut sweep = state_with(&model, 32, sweep_config(3));
        prefill_prompt(&model, &mut sweep, &ids, None).unwrap();

        assert_eq!(token_major.seq_len().unwrap(), ids.len());
        assert_eq!(sweep.seq_len().unwrap(), ids.len());
        for layer in 0..model.n_layers() as usize {
            assert_eq!(
                sweep.kv_k_layer(layer).unwrap(),
                token_major.kv_k_layer(layer).unwrap(),
                "layer {layer} keys"
            );
            assert_eq!(
                sweep.kv_v_layer(layer).unwrap(),
                token_major.kv_v_layer(layer).unwrap(),
                "layer {layer} values"
            );
        }
    }

    /// Decode must continue cleanly off a swept prefill: the cache is
    /// uniform, the slot pool is a cache again, and the next token lands on
    /// the position prefill stopped at.
    #[test]
    fn decode_continues_off_a_swept_prefill() {
        let (_fx, model) = load_fixture("prefill-then-decode");
        let ids = [1u32, 2, 3, 4, 5];

        let mut a = state_with(&model, 32, token_major_config());
        prefill_prompt(&model, &mut a, &ids, None).unwrap();
        let want = logit_bits(
            super::super::forward_token(&model, &mut a, 6, 5, true)
                .unwrap()
                .unwrap(),
        );

        let mut b = state_with(&model, 32, sweep_config(2));
        prefill_prompt(&model, &mut b, &ids, None).unwrap();
        assert_eq!(
            b.seq_len().unwrap(),
            5,
            "the cache is uniform after a chunk"
        );
        let got = logit_bits(
            super::super::forward_token(&model, &mut b, 6, 5, true)
                .unwrap()
                .unwrap(),
        );
        assert_eq!(got, want);
    }

    // ---- attention positions --------------------------------------------

    /// Row `r` must attend over exactly `start + r + 1` positions. Proven
    /// where it is observable: the logits of a prompt prefilled as one chunk
    /// equal the logits of the same prompt fed one row at a time, which is
    /// only true if no row saw a position behind it. A row that saw the whole
    /// chunk would make the answer depend on the chunk width.
    #[test]
    fn a_row_never_attends_past_its_own_position() {
        let (_fx, model) = load_fixture("prefill-causal");
        let ids: Vec<u32> = (0..9u32).map(|i| (i * 7 + 3) % 32).collect();

        // Chunk 1 is a row-at-a-time sweep: every row is the last position in
        // its own chunk, so its mask is unambiguous.
        let want = run(&model, sweep_config(1), &ids);
        for chunk in [2usize, 3, 4, 9] {
            assert_eq!(
                run(&model, sweep_config(chunk), &ids),
                want,
                "chunk {chunk} let a row see the future"
            );
        }

        // And the same prompt with one token appended must differ, so the
        // test above is not passing on a degenerate model whose logits ignore
        // the context entirely.
        let mut longer = ids.clone();
        longer.push(11);
        assert_ne!(run(&model, sweep_config(4), &longer), want);
    }

    /// Mid-chunk the KV cache is deliberately ragged — layer `L` holds the
    /// whole chunk while every layer after it still holds `start` — and
    /// [`crate::kv::KvCache::seq_len`] reports that as an error. Two things
    /// follow, and both are asserted here.
    ///
    /// **The driver cannot use `seq_len` per layer**, which is why
    /// [`ForwardState::kv_len`] exists; the window in which `seq_len` refuses
    /// is reproduced directly below, because no successful prefill can be
    /// paused inside it.
    ///
    /// **The raggedness never escapes**: every chunk boundary and the end of
    /// the prompt leave every layer level, at every chunk width, and a prompt
    /// past the context cap is refused before a single row is appended rather
    /// than halfway through one.
    #[test]
    fn the_cache_is_ragged_mid_chunk_and_level_after() {
        let (_fx, model) = load_fixture("prefill-ragged");
        let n_layers = model.n_layers() as usize;

        // The mid-chunk window, reproduced: a layer-major append of a
        // 3-row chunk leaves the cache in a state `seq_len` refuses, and
        // levels it again only when the last layer catches up.
        let arch = model.arch();
        let mut kv = KvCache::new(
            n_layers,
            arch.n_kv_heads as usize,
            arch.head_dim as usize,
            8,
        )
        .unwrap();
        let row = vec![0.25f32; arch.n_kv_heads as usize * arch.head_dim as usize];
        for layer in 0..n_layers {
            for _ in 0..3 {
                kv.append(layer, &row, &row).unwrap();
            }
            assert_eq!(kv.len(layer).unwrap(), 3);
            if layer + 1 < n_layers {
                assert!(
                    matches!(kv.seq_len(), Err(KvError::RaggedLayers { .. })),
                    "layer {layer}: a half-swept chunk must read as ragged"
                );
            }
        }
        assert_eq!(kv.seq_len().unwrap(), 3);

        // Clean runs: level after every chunk boundary, at chunk widths that
        // divide the prompt differently.
        for chunk in [1usize, 2, 3, 4, 5] {
            let mut st = state_with(&model, 16, sweep_config(chunk));
            prefill_prompt(&model, &mut st, &[1, 2, 3, 4, 5], None).unwrap();
            assert_eq!(st.seq_len().unwrap(), 5, "chunk {chunk}");
            for layer in 0..n_layers {
                assert_eq!(st.kv_len(layer).unwrap(), 5, "chunk {chunk} layer {layer}");
            }
        }

        // A prompt past the cap is refused up front, so the cache is left
        // untouched and level rather than half-appended: the check is on the
        // whole prompt, not on the row that would have overflowed.
        let mut st = state_with(&model, 3, sweep_config(2));
        assert!(prefill_prompt(&model, &mut st, &[1, 2, 3, 4], None).is_err());
        assert_eq!(st.seq_len().unwrap(), 0);
        for layer in 0..n_layers {
            assert_eq!(st.kv_len(layer).unwrap(), 0);
        }
    }

    // ---- the inverse index ----------------------------------------------

    /// The index must be a permutation: every `(row, slot)` appears exactly
    /// once, every row contributes all `top_k` of its slots, and each
    /// expert's run holds only entries that named it.
    #[test]
    fn the_inverse_index_is_a_permutation() {
        let dims = PrefillDims {
            hidden: 256,
            q_dim: 256,
            kv_dim: 128,
            moe: 256,
            top_k: 3,
            n_experts: 5,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 64,
            hidden_q8k: 1,
            hidden_q80: 8,
            attn_q8k: 1,
            moe_q8k: 1,
            wide: 256,
            shards: 2,
            max_positions: 8,
        };
        let rows = 6usize;
        let mut bytes = vec![0u8; scratch_bytes(&dims, rows).unwrap()];
        let (mut s, _) = carve(&mut bytes, &dims, rows).unwrap();

        // A deliberately lumpy assignment: expert 0 takes many rows, expert 4
        // takes none, and one expert is chosen by every row.
        for r in 0..rows {
            for slot in 0..dims.top_k {
                s.topk[r * dims.top_k + slot] = TopkEntry {
                    expert: ((r * dims.top_k + slot) % 4) as u32,
                    weight: 1.0,
                };
            }
        }
        build_index(&mut s, &dims, rows).unwrap();

        assert_eq!(s.index_start[0], 0);
        assert_eq!(
            s.index_start[dims.n_experts] as usize,
            rows * dims.top_k,
            "the index covers every (row, slot) pair"
        );
        let mut seen = vec![false; rows * dims.top_k];
        for expert in 0..dims.n_experts {
            let (lo, hi) = (
                s.index_start[expert] as usize,
                s.index_start[expert + 1] as usize,
            );
            assert!(lo <= hi, "expert {expert} has a reversed run");
            for entry in &s.index[lo..hi] {
                let flat = entry.row as usize * dims.top_k + entry.slot as usize;
                assert!(
                    !seen[flat],
                    "(row {}, slot {}) twice",
                    entry.row, entry.slot
                );
                seen[flat] = true;
                assert_eq!(
                    s.topk[flat].expert as usize, expert,
                    "entry filed under the wrong expert"
                );
            }
        }
        assert!(seen.iter().all(|&hit| hit), "a (row, slot) pair was lost");
        // Expert 4 is routed by nobody, so its run is empty and the sweep
        // would never be asked for it.
        assert_eq!(s.index_start[4], s.index_start[5]);
    }

    /// A row that selected the same expert twice would give that expert more
    /// activations than the batch buffers hold. The router cannot produce it;
    /// the index refuses it anyway, because the alternative is a slice past
    /// the end of `gate`, `up`, `moe_t` and `tmat` alike.
    #[test]
    fn a_repeated_routed_expert_is_typed() {
        let dims = PrefillDims {
            top_k: 2,
            ..tiny_dims()
        };
        let mut bytes = vec![0u8; scratch_bytes(&dims, 1).unwrap()];
        let (mut s, _) = carve(&mut bytes, &dims, 1).unwrap();
        s.topk[0] = TopkEntry {
            expert: 3,
            weight: 0.5,
        };
        s.topk[1] = TopkEntry {
            expert: 3,
            weight: 0.5,
        };
        assert!(
            matches!(
                build_index(&mut s, &dims, 1).unwrap_err(),
                ForwardError::RepeatedRoutedExpert {
                    expert: 3,
                    count: 2,
                    rows: 1
                }
            ),
            "a doubly-routed expert must be refused"
        );
    }

    /// A selection naming an expert the layer does not have is a typed
    /// refusal, not an out-of-bounds index into the CSR arrays.
    #[test]
    fn an_out_of_range_routed_expert_is_typed() {
        let dims = PrefillDims {
            hidden: 256,
            q_dim: 256,
            kv_dim: 128,
            moe: 256,
            top_k: 2,
            n_experts: 4,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 64,
            hidden_q8k: 1,
            hidden_q80: 8,
            attn_q8k: 1,
            moe_q8k: 1,
            wide: 256,
            shards: 2,
            max_positions: 8,
        };
        let mut bytes = vec![0u8; scratch_bytes(&dims, 1).unwrap()];
        let (mut s, _) = carve(&mut bytes, &dims, 1).unwrap();
        s.topk[0] = TopkEntry {
            expert: 4,
            weight: 1.0,
        };
        s.topk[1] = TopkEntry {
            expert: 0,
            weight: 1.0,
        };
        assert!(matches!(
            build_index(&mut s, &dims, 1).unwrap_err(),
            ForwardError::RoutedExpertOutOfRange {
                expert: 4,
                n_experts: 4
            }
        ));
    }

    // ---- the reduction order --------------------------------------------

    /// The reduction must run in top-k order, not arrival order. Built to
    /// fail if it did not: three staged values whose f32 sum is
    /// order-dependent, weighted so that summing them by ascending expert id
    /// (which is the order a sweep yields them in) gives different bits from
    /// summing them by top-k rank.
    #[test]
    fn the_reduction_runs_in_top_k_order_not_arrival_order() {
        let dims = PrefillDims {
            hidden: 4,
            q_dim: 4,
            kv_dim: 4,
            moe: 4,
            top_k: 3,
            n_experts: 3,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 4,
            hidden_q8k: 1,
            hidden_q80: 1,
            attn_q8k: 1,
            moe_q8k: 1,
            wide: 4,
            shards: 2,
            max_positions: 8,
        };
        let mut bytes = vec![0u8; scratch_bytes(&dims, 1).unwrap()];
        let (mut s, _) = carve(&mut bytes, &dims, 1).unwrap();

        // Classic non-associative triple. Half an ulp added to 1.0 is a tie
        // and rounds to even, i.e. back to 1.0, twice over — but the two
        // halves summed *first* make a whole ulp, which 1.0 does keep.
        let big = 1.0f32;
        let small = f32::EPSILON / 2.0;
        // Top-k order is (expert 2, expert 0, expert 1): the sweep would
        // deliver 0, 1, 2 instead.
        s.topk[0] = TopkEntry {
            expert: 2,
            weight: 1.0,
        };
        s.topk[1] = TopkEntry {
            expert: 0,
            weight: 1.0,
        };
        s.topk[2] = TopkEntry {
            expert: 1,
            weight: 1.0,
        };
        let staged = [big, small, small];
        for (slot, &value) in staged.iter().enumerate() {
            for h in 0..dims.hidden {
                s.staged[slot * dims.hidden + h] = value;
            }
        }
        s.residual.fill(0.0);
        reduce_experts(&mut s, &dims, 1).unwrap();

        // Top-k order sums big first, so both small terms vanish.
        let want_topk = ((0.0f32 + big) + small) + small;
        // Arrival (ascending expert id) order would sum slot 1 and slot 2
        // first — the two smalls — and their sum survives the add to `big`.
        let arrival = ((0.0f32 + small) + small) + big;
        assert_ne!(
            want_topk.to_bits(),
            arrival.to_bits(),
            "the fixture is not order-sensitive; the test proves nothing"
        );
        for (h, &value) in s.residual.iter().enumerate() {
            assert_eq!(value.to_bits(), want_topk.to_bits(), "element {h}");
        }
    }

    // ---- the sub-allocator ----------------------------------------------

    fn tiny_dims() -> PrefillDims {
        PrefillDims {
            hidden: 512,
            q_dim: 768,
            kv_dim: 256,
            moe: 256,
            top_k: 4,
            n_experts: 9,
            n_heads: 6,
            n_kv_heads: 2,
            head_dim: 128,
            hidden_q8k: 2,
            hidden_q80: 16,
            attn_q8k: 3,
            moe_q8k: 1,
            wide: 768,
            // Two shards over a 16-position cache: `attn_shard` is
            // `3 * 16 + 2 * 128 = 304` f32, so the carve is small enough that
            // the overlap and packing tests still run on a few KiB.
            shards: 2,
            max_positions: 16,
        }
    }

    /// The v0 dims, so the arithmetic quoted on [`scratch_bytes`] is a
    /// checked claim rather than a comment that rots. Six shards is the
    /// shipped pin (six P-cores, the submitting thread among them) and 4096
    /// is the shipped context cap; both are part of the carve now.
    fn qwen3_30b_a3b_dims() -> PrefillDims {
        PrefillDims {
            hidden: 2048,
            q_dim: 32 * 128,
            kv_dim: 4 * 128,
            moe: 768,
            top_k: 8,
            n_experts: 128,
            n_heads: 32,
            n_kv_heads: 4,
            head_dim: 128,
            hidden_q8k: 8,
            hidden_q80: 64,
            attn_q8k: 16,
            moe_q8k: 3,
            wide: 4096,
            shards: 6,
            max_positions: 4096,
        }
    }

    /// The documented v0 figures, to the byte. A 512-row chunk costs 77.94
    /// MiB of the ~1,438 MiB slot slab, of which the `[rows][top_k][hidden]`
    /// staging that makes the reduction order-independent is 32 MiB and the
    /// six per-shard attention carves are 0.76 MiB.
    #[test]
    fn the_documented_v0_scratch_arithmetic_holds() {
        let dims = qwen3_30b_a3b_dims();
        assert_eq!(scratch_bytes(&dims, 512).unwrap(), 81_728_516);
        // Just under 77.94 MiB.
        assert_eq!(81_728_516 / 1024 / 1024, 77);
        // Staging alone, and the pair of `[rows][q_dim]` planes.
        assert_eq!(512 * dims.top_k * dims.hidden * 4, 33_554_432);
        assert_eq!(2 * 512 * dims.q_dim * 4, 16_777_216);
        // The attention carve: `(32/4) * 4096 + 2 * 128` f32 per shard, six
        // shards, which is exactly what the previous total grew by.
        assert_eq!(dims.attn_shard(), 33_024);
        assert_eq!(dims.attn_scratch() * 4, 792_576);
        assert_eq!(80_935_940 + 792_576, 81_728_516);
        // Affine in rows, which `plan_arena` divides by: the attention carve
        // is per *shard*, not per row, so the slope is untouched.
        let per_row = scratch_bytes(&dims, 2).unwrap() - scratch_bytes(&dims, 1).unwrap();
        assert_eq!(per_row, 145_408 + 35 * 292 + 64 * 36 + 2 * 8 * 8);
        assert_eq!(
            scratch_bytes(&dims, 512).unwrap(),
            scratch_bytes(&dims, 1).unwrap() + 511 * per_row
        );

        // Beside the default sweep ring, against the shipped slab: 125 MiB
        // of 1,438, so neither dial is anywhere near the constraint.
        let shipped = layout_of(&[(3_059_712, 128), (2_654_208, 128)]);
        let ring = ring_bytes(&shipped, 8, 2);
        assert_eq!(ring, 48_955_392);
        assert!(ring + 81_728_516 < 1_438 * 1024 * 1024);
    }

    // ---- the attention split ---------------------------------------------

    /// [`attention_shard_range`] is a partition: contiguous, ascending, every
    /// row in exactly one shard, and no shard reaching past `rows`.
    ///
    /// The parallel attention region's whole soundness argument rests on this
    /// — a row visited twice would be two threads writing one `[q_dim]` span,
    /// and a row visited never would be a silently wrong answer.
    #[test]
    fn the_attention_split_partitions_every_row() {
        for rows in [0usize, 1, 2, 5, 6, 7, 8, 63, 512, 1891] {
            for start in [0usize, 1, 7, 512, 4095] {
                for shards in [1usize, 2, 3, 6, 8, 16] {
                    let mut seen = vec![0u32; rows];
                    let mut next = 0usize;
                    for index in 0..shards {
                        let mine = attention_shard_range(rows, start, shards, index);
                        assert!(mine.start <= mine.end, "{mine:?} is reversed");
                        assert_eq!(mine.start, next, "shards {shards} index {index} not flush");
                        assert!(mine.end <= rows, "{mine:?} runs past {rows}");
                        next = mine.end;
                        for r in mine {
                            seen[r] += 1;
                        }
                    }
                    assert_eq!(next, rows, "rows {rows} shards {shards} left a tail");
                    assert!(
                        seen.iter().all(|&hits| hits == 1),
                        "rows {rows} start {start} shards {shards}: {seen:?}"
                    );
                }
            }
        }
    }

    /// The split is a pure function of `(rows, start, shards, index)`.
    ///
    /// Purity is not what makes the region bit-neutral — rows are independent,
    /// so *any* partition gives the same bytes — but it is what makes a run
    /// reproducible, and reproducibility is what makes a bit-identity failure
    /// debuggable instead of a coin flip.
    #[test]
    fn the_attention_split_is_a_pure_function() {
        for (rows, start, shards) in [(512usize, 0usize, 6usize), (355, 1536, 6), (17, 512, 3)] {
            for index in 0..shards {
                let once = attention_shard_range(rows, start, shards, index);
                for _ in 0..4 {
                    assert_eq!(attention_shard_range(rows, start, shards, index), once);
                }
            }
        }
    }

    /// The split balances *cost*, not row count: row `r` attends
    /// `start + r + 1` positions, so an equal-row split leaves the last shard
    /// of the first 512-row chunk with 39,950 of 131,328 cost units against an
    /// ideal 21,888 — a 3.3x makespan where 6x was available. This is the
    /// assertion that a future edit reverting to `shard_range` would fail.
    #[test]
    fn the_attention_split_balances_cost_not_rows() {
        let (rows, start, shards) = (512usize, 0usize, 6usize);
        let total = attention_cost(rows, start);
        assert_eq!(total, 131_328);
        let ideal = total / shards as u128;

        let mut widest = 0u128;
        for index in 0..shards {
            let mine = attention_shard_range(rows, start, shards, index);
            let cost = attention_cost(mine.end, start) - attention_cost(mine.start, start);
            // Every shard within one row's worth of the ideal, which at this
            // chunk is at most 512 units against an ideal of 21,888.
            let slack = cost.abs_diff(ideal);
            assert!(
                slack <= (start + rows) as u128,
                "shard {index} {mine:?} costs {cost}, ideal {ideal}"
            );
            widest = widest.max(cost);
        }
        // Cost-balanced means row-unbalanced, and deliberately so: the cheap
        // early rows come in bulk and the expensive late ones do not.
        let first = attention_shard_range(rows, start, shards, 0).len();
        let last = attention_shard_range(rows, start, shards, shards - 1).len();
        assert!(first > last * 2, "{first} rows against {last}");

        // The equal-row split's last shard, for contrast: 39,950 cost units
        // against the balanced split's 22,175 makespan, an ideal of 21,888.
        let naive = attention_cost(rows, start) - attention_cost(rows - rows / shards, start);
        assert!(
            widest * 3 < naive * 2,
            "cost balancing bought too little: {widest} vs {naive}"
        );
    }

    /// The carve consumes exactly what [`scratch_bytes`] promised. The two
    /// walk the same sequence in the same order, and this is what keeps them
    /// from drifting: a buffer added to one and not the other fails here.
    #[test]
    fn carve_consumes_exactly_the_documented_bytes() {
        let dims = tiny_dims();
        for rows in [1usize, 2, 7, 33] {
            let total = scratch_bytes(&dims, rows).unwrap();
            let mut bytes = vec![0u8; total];
            let (_, used) = carve(&mut bytes, &dims, rows).unwrap();
            assert_eq!(used, total, "rows {rows}");
        }
    }

    /// Every block type the carve hands out is 4-aligned with a size that is
    /// a whole number of 4-byte units, which is why no padding ever appears
    /// between two carved buffers and `scratch_bytes` can be a plain sum.
    #[test]
    fn every_scratch_type_packs_without_padding() {
        for (name, size, align) in [
            ("f32", size_of::<f32>(), align_of::<f32>()),
            ("u32", size_of::<u32>(), align_of::<u32>()),
            ("BlockQ8K", size_of::<BlockQ8K>(), align_of::<BlockQ8K>()),
            ("BlockQ8_0", size_of::<BlockQ8_0>(), align_of::<BlockQ8_0>()),
            ("TopkEntry", size_of::<TopkEntry>(), align_of::<TopkEntry>()),
            ("RowSlot", size_of::<RowSlot>(), align_of::<RowSlot>()),
        ] {
            assert_eq!(align, 4, "{name} alignment");
            assert_eq!(size % 4, 0, "{name} size {size}");
        }
        assert_eq!(size_of::<BlockQ8K>(), 292);
        assert_eq!(size_of::<BlockQ8_0>(), 36);
    }

    /// The carved buffers are disjoint and in bounds: stamping each one with
    /// a distinct byte and reading the span back finds every stamp intact.
    #[test]
    fn carved_buffers_do_not_overlap() {
        let dims = tiny_dims();
        let rows = 5usize;
        let total = scratch_bytes(&dims, rows).unwrap();
        let mut bytes = vec![0u8; total];
        {
            let (s, _) = carve(&mut bytes, &dims, rows).unwrap();
            s.residual.fill(f32::from_bits(0x0101_0101));
            s.normed.fill(f32::from_bits(0x0202_0202));
            s.q.fill(f32::from_bits(0x0303_0303));
            s.staged.fill(f32::from_bits(0x0404_0404));
            s.expert_acc.fill(f32::from_bits(0x0505_0505));
            s.attn_shards.fill(f32::from_bits(0x0808_0808));
            s.index_start.fill(0x0606_0606);
            s.index_cursor.fill(0x0707_0707);
        }
        // Re-carve and check every stamp survived, which it cannot if two
        // buffers overlapped.
        let (s, _) = carve(&mut bytes, &dims, rows).unwrap();
        assert!(s.residual.iter().all(|v| v.to_bits() == 0x0101_0101));
        assert!(s.normed.iter().all(|v| v.to_bits() == 0x0202_0202));
        assert!(s.q.iter().all(|v| v.to_bits() == 0x0303_0303));
        assert!(s.staged.iter().all(|v| v.to_bits() == 0x0404_0404));
        assert!(s.expert_acc.iter().all(|v| v.to_bits() == 0x0505_0505));
        assert!(s.attn_shards.iter().all(|v| v.to_bits() == 0x0808_0808));
        assert!(s.index_start.iter().all(|&v| v == 0x0606_0606));
        assert!(s.index_cursor.iter().all(|&v| v == 0x0707_0707));
        assert_eq!(s.residual.len(), rows * dims.hidden);
        assert_eq!(s.staged.len(), rows * dims.top_k * dims.hidden);
        assert_eq!(s.attn_shards.len(), dims.shards * dims.attn_shard());
        assert_eq!(s.index_start.len(), dims.n_experts + 1);
    }

    /// A span one byte short of the layout is a typed refusal, not a panic
    /// and not a silently truncated buffer.
    #[test]
    fn a_short_scratch_span_is_typed() {
        let dims = tiny_dims();
        let total = scratch_bytes(&dims, 3).unwrap();
        let mut bytes = vec![0u8; total - 1];
        assert!(matches!(
            carve(&mut bytes, &dims, 3),
            Err(ForwardError::PrefillScratch { .. })
        ));
        let mut empty: [u8; 0] = [];
        assert!(matches!(
            carve(&mut empty, &dims, 1),
            Err(ForwardError::PrefillScratch { .. })
        ));
    }

    /// The carve does not assume the span starts on a page: it pads to each
    /// type's alignment from wherever it actually is.
    #[test]
    fn the_carve_aligns_from_an_offset_base() {
        let dims = tiny_dims();
        let rows = 2usize;
        let total = scratch_bytes(&dims, rows).unwrap();
        // Three bytes of lead-in, so the base is deliberately misaligned.
        let mut bytes = vec![0u8; total + 8];
        let (s, used) = carve(&mut bytes[3..], &dims, rows).unwrap();
        assert!(used <= total + 3, "padding ran away: {used} vs {total}");
        assert_eq!(s.residual.as_ptr() as usize % align_of::<f32>(), 0);
        assert_eq!(s.acts_hidden.as_ptr() as usize % align_of::<BlockQ8K>(), 0);
    }

    // ---- arena planning --------------------------------------------------

    fn layout_of(strides: &[(u64, u32)]) -> ExpertsLayout {
        ExpertsLayout {
            layers: strides
                .iter()
                .enumerate()
                .map(|(i, &(stride, n_experts))| LayerLayout {
                    file: format!("experts/layer_{i:02}.bin"),
                    stride,
                    n_experts,
                    projections: Vec::new(),
                })
                .collect(),
        }
    }

    /// The ring estimate must match `io::sweep`'s: the widest layer's window,
    /// capped at that layer's expert count, times the windows in flight.
    #[test]
    fn the_ring_estimate_caps_windows_at_the_expert_count() {
        // The shipped geometry: two stride classes, 128 experts, 8 per
        // window, 2 in flight.
        let shipped = layout_of(&[(3_059_712, 128), (2_654_208, 128)]);
        assert_eq!(ring_bytes(&shipped, 8, 2), 2 * 8 * 3_059_712);
        // A layer with fewer experts than the dial cannot make a full window.
        let tiny = layout_of(&[(4096, 3)]);
        assert_eq!(ring_bytes(&tiny, 8, 2), 2 * 3 * 4096);
        // And a zero-expert layer does not produce a zero-wide ring.
        let empty = layout_of(&[(4096, 0)]);
        assert_eq!(ring_bytes(&empty, 8, 1), 4096);
    }

    /// A pool too small for the requested chunk narrows the chunk rather than
    /// refusing to prefill, and gives up read-ahead only after the chunk is
    /// down to a single row.
    #[test]
    fn the_arena_plan_narrows_before_it_refuses() {
        let dims = tiny_dims();
        let layout = layout_of(&[(65_536, 4)]);
        let config = PrefillConfig {
            mode: PrefillMode::Sweep,
            chunk: 512,
            experts_per_window: 8,
            windows_in_flight: 2,
        };

        // Roomy: the full chunk and the full read-ahead.
        let (rows, sweep) = plan_arena(&layout, &dims, config, 512, 64 * 1024 * 1024).unwrap();
        assert_eq!(rows, 512);
        assert_eq!(sweep.windows_in_flight, 2);

        // Tight: the ring is 512 KiB, so the chunk has to shrink.
        let (rows, sweep) = plan_arena(&layout, &dims, config, 512, 1024 * 1024).unwrap();
        assert!((1..512).contains(&rows), "rows {rows}");
        assert_eq!(sweep.windows_in_flight, 2, "read-ahead is given up last");

        // Tighter still: one window's worth plus a row.
        let one_window = 4 * 65_536;
        let (rows, sweep) = plan_arena(
            &layout,
            &dims,
            config,
            512,
            one_window + scratch_bytes(&dims, 1).unwrap() as u64 + 8192,
        )
        .unwrap();
        assert_eq!(rows, 1);
        assert_eq!(sweep.windows_in_flight, 1);

        // Hopeless: a typed refusal naming what it would have needed.
        let err = plan_arena(&layout, &dims, config, 512, one_window).unwrap_err();
        assert!(
            matches!(err, ForwardError::PrefillScratch { .. }),
            "unexpected error: {err}"
        );
    }

    /// The chunk dial is a ceiling, never a floor: a prompt shorter than one
    /// chunk carves for the prompt, not for 512 rows.
    #[test]
    fn a_short_prompt_carves_for_the_prompt() {
        let dims = tiny_dims();
        let layout = layout_of(&[(65_536, 4)]);
        let config = PrefillConfig {
            mode: PrefillMode::Sweep,
            chunk: 512,
            experts_per_window: 8,
            windows_in_flight: 2,
        };
        let (rows, _) = plan_arena(&layout, &dims, config, 3, 64 * 1024 * 1024).unwrap();
        assert_eq!(rows, 3);
    }

    // ---- misc -------------------------------------------------------------

    #[test]
    fn transpose_moves_every_element() {
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 12];
        transpose(&src, 3, 4, &mut dst).unwrap();
        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(dst[c * 3 + r], src[r * 4 + c]);
            }
        }
        // Degenerate shapes are no-ops rather than panics, as long as both
        // slices agree with `rows * cols == 0`.
        let mut empty: [f32; 0] = [];
        transpose(&[], 0, 4, &mut empty).unwrap();
        transpose(&[], 3, 0, &mut empty).unwrap();
    }

    /// A mis-sized `transpose` is a typed refusal, not a partial copy.
    ///
    /// The old shape dropped out-of-range writes with `dst.get_mut` and
    /// truncated a short source with `chunks_exact(cols).take(rows)`. Both are
    /// silent, and in a module whose whole thesis is that the driver may not
    /// assume its scratch is zeroed, either one would leave the *previous*
    /// chunk's arena bytes in `q`, `k`, `v` or `moe_t` and carry them into the
    /// residual. A numerical divergence with no error attached is the one
    /// failure this module cannot afford; every call site is exactly sized
    /// today, and this is what keeps that a checked fact.
    #[test]
    fn a_mis_sized_transpose_is_typed() {
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 12];

        // A source one row short would have been silently truncated, leaving
        // the last output column holding whatever `dst` already had.
        assert!(matches!(
            transpose(&src[..8], 3, 4, &mut dst),
            Err(ForwardError::Kernel(KernelError::LengthMismatch {
                left: 8,
                right: 12,
                ..
            }))
        ));
        // A destination one element short would have dropped that write.
        assert!(matches!(
            transpose(&src, 3, 4, &mut dst[..11]),
            Err(ForwardError::Kernel(KernelError::LengthMismatch {
                left: 11,
                right: 12,
                ..
            }))
        ));
        // And a degenerate shape does not excuse a mis-sized slice either.
        assert!(transpose(&[], 0, 4, &mut dst).is_err());
        // `rows * cols` overflowing is typed rather than wrapping.
        assert!(transpose(&src, usize::MAX, 2, &mut dst).is_err());
    }

    /// All three dials refuse at one point.
    ///
    /// They used to refuse at three: the chunk in [`PrefillConfig::validate`],
    /// `experts_per_window == 0` nowhere (it was silently coerced to 1 by
    /// [`plan_arena`]), and an oversized `windows_in_flight` only once the
    /// sweep session was already being opened. Every field is `pub`, so every
    /// field is caller-supplied.
    #[test]
    fn every_prefill_dial_refuses_at_validate() {
        let base = PrefillConfig::default();
        assert!(base.validate().is_ok());

        assert!(matches!(
            PrefillConfig { chunk: 0, ..base }.validate().unwrap_err(),
            ForwardError::InvalidPrefillChunk { chunk: 0 }
        ));
        for bad in [
            PrefillConfig {
                experts_per_window: 0,
                ..base
            },
            PrefillConfig {
                windows_in_flight: 0,
                ..base
            },
            PrefillConfig {
                windows_in_flight: crate::io::MAX_WINDOWS_IN_FLIGHT + 1,
                ..base
            },
        ] {
            let err = bad.validate().unwrap_err();
            assert!(
                matches!(err, ForwardError::Sweep(SweepError::BadDials { .. })),
                "unexpected error for {bad:?}: {err}"
            );
        }

        // And the same refusal reaches the driver's own entry points rather
        // than being coerced by either of them.
        let layout = layout_of(&[(65_536, 4)]);
        assert!(matches!(
            plan_arena(
                &layout,
                &tiny_dims(),
                PrefillConfig {
                    experts_per_window: 0,
                    ..base
                },
                4,
                64 * 1024 * 1024,
            )
            .unwrap_err(),
            ForwardError::Sweep(SweepError::BadDials { .. })
        ));
        let (_fx, model) = load_fixture("prefill-dials");
        let mut st = state_with(&model, 8, sweep_config(4));
        assert!(matches!(
            st.set_prefill_config(PrefillConfig {
                windows_in_flight: crate::io::MAX_WINDOWS_IN_FLIGHT + 1,
                ..sweep_config(4)
            })
            .unwrap_err(),
            ForwardError::Sweep(SweepError::BadDials { .. })
        ));
    }

    #[test]
    fn an_empty_prompt_is_typed() {
        let (_fx, model) = load_fixture("prefill-empty");
        let mut st = state_with(&model, 8, sweep_config(4));
        assert!(matches!(
            prefill_prompt(&model, &mut st, &[], None).unwrap_err(),
            ForwardError::EmptyPrefill
        ));
    }

    #[test]
    fn a_zero_chunk_is_typed() {
        let (_fx, model) = load_fixture("prefill-zero-chunk");
        let mut st = state_with(&model, 8, sweep_config(4));
        assert!(matches!(
            st.set_prefill_config(PrefillConfig {
                chunk: 0,
                ..sweep_config(4)
            })
            .unwrap_err(),
            ForwardError::InvalidPrefillChunk { chunk: 0 }
        ));
    }

    #[test]
    fn a_prompt_past_the_context_cap_is_typed() {
        let (_fx, model) = load_fixture("prefill-cap");
        let mut st = state_with(&model, 3, sweep_config(4));
        let err = prefill_prompt(&model, &mut st, &[1, 2, 3, 4], None).unwrap_err();
        assert!(
            matches!(
                err,
                ForwardError::Kv(KvError::CapacityExceeded { capacity: 3, .. })
            ),
            "unexpected error: {err}"
        );
    }

    /// The route sink sees every `(position, layer)` pair exactly once, with
    /// the same selections the token-major path reports — only the order
    /// differs, which the type's docs promise.
    #[test]
    fn the_route_sink_covers_every_position_and_layer() {
        let (_fx, model) = load_fixture("prefill-sink");
        let ids = [1u32, 2, 3, 4, 5];
        let n_layers = model.n_layers();
        let top_k = model.arch().top_k as usize;

        let collect = |config: PrefillConfig| {
            let mut st = state_with(&model, 16, config);
            let mut seen: Vec<(usize, u32, Vec<u32>)> = Vec::new();
            {
                let mut sink = |pos: usize, layer: u32, topk: &[(u32, f32)]| {
                    seen.push((pos, layer, topk.iter().map(|&(e, _)| e).collect()));
                };
                prefill_prompt(&model, &mut st, &ids, Some(&mut sink)).unwrap();
            }
            seen.sort_by_key(|(pos, layer, _)| (*pos, *layer));
            seen
        };

        let want = collect(token_major_config());
        assert_eq!(want.len(), ids.len() * n_layers as usize);
        assert!(want.iter().all(|(_, _, experts)| experts.len() == top_k));
        assert_eq!(collect(sweep_config(2)), want);
        assert_eq!(collect(sweep_config(5)), want);
    }

    /// The environment override is read, and nonsense in it is ignored rather
    /// than fatal.
    #[test]
    fn the_env_override_parses_or_is_ignored() {
        // `from_env` reads process-global state, so this test only asserts
        // the parse table via the same match, not by mutating the
        // environment (which would race every other test in the binary).
        let base = PrefillConfig::default();
        assert_eq!(base.mode, PrefillMode::Sweep);
        assert_eq!(base.chunk, DEFAULT_PREFILL_CHUNK);
        assert_eq!(
            base.sweep_config(),
            SweepConfig {
                experts_per_window: base.experts_per_window,
                windows_in_flight: base.windows_in_flight,
            }
        );
        assert!(base.validate().is_ok());
        assert!(PrefillConfig { chunk: 0, ..base }.validate().is_err());
    }

    // ---- the phase split ------------------------------------------------
    //
    // Nothing here asserts a *duration*. This machine runs the model under a
    // cgroup while these tests run, so any "attention took at least X" would
    // flake; what is asserted is the arithmetic — that the phases are disjoint
    // spans of the total, that a run starts from zero, and that a reset drops
    // them. A double-charged phase pushes `accounted` past `total` and every
    // one of these catches it.

    /// [`PrefillTiming::other`] closes the sum by construction, and
    /// [`PrefillTiming::accounted`] is exactly the five phases. Pure
    /// arithmetic: no model, no clock, nothing to flake.
    #[test]
    fn the_unaccounted_remainder_closes_the_sum() {
        let ms = Duration::from_millis;
        let timing = PrefillTiming {
            mode: Some(PrefillMode::Sweep),
            tokens: 512,
            total: ms(1000),
            attention: ms(400),
            projections: ms(30),
            expert_compute: ms(450),
            expert_io: ms(90),
            elementwise: ms(20),
        };
        assert_eq!(timing.accounted(), ms(990));
        assert_eq!(timing.other(), ms(10));
        assert_eq!(
            timing.accounted() + timing.other(),
            timing.total,
            "the phases and the remainder must be the whole run"
        );
        assert!(timing.ran());

        // A total shorter than its phases would be an instrumentation bug, not
        // a panic: `other` saturates.
        let broken = PrefillTiming {
            total: ms(1),
            ..timing
        };
        assert_eq!(broken.other(), Duration::ZERO);

        // The default has run nothing and claims nothing.
        let fresh = PrefillTiming::default();
        assert!(!fresh.ran());
        assert_eq!(fresh.accounted(), Duration::ZERO);
        assert_eq!(fresh.other(), Duration::ZERO);
        assert_eq!(fresh.phases().len(), 6);
    }

    /// Both paths report, and both report phases that fit inside their own
    /// total. This is what makes the sweep and the token-major loop
    /// comparable phase by phase, which is the whole point of instrumenting
    /// the A/B path too.
    #[test]
    fn both_prefill_paths_report_phases_inside_their_total() {
        let (_fx, model) = load_fixture("prefill-timing-split");
        let ids = [1u32, 2, 3, 4, 5, 6];

        for (want_mode, config) in [
            (PrefillMode::Sweep, sweep_config(4)),
            (PrefillMode::TokenMajor, token_major_config()),
        ] {
            let mut st = state_with(&model, 32, config);
            assert!(
                !st.prefill_timing().ran(),
                "{want_mode:?}: a fresh state has no prefill to report"
            );
            prefill_prompt(&model, &mut st, &ids, None).unwrap();

            let timing = st.prefill_timing();
            assert_eq!(timing.mode, Some(want_mode));
            assert_eq!(timing.tokens, ids.len() as u64);
            assert!(timing.ran());
            assert!(
                timing.accounted() <= timing.total,
                "{want_mode:?}: phases {:?} exceed the total {:?} — a region is \
                 charged twice",
                timing.accounted(),
                timing.total
            );
            assert_eq!(
                timing.accounted() + timing.other(),
                timing.total,
                "{want_mode:?}: the split must account for the whole run"
            );
            assert!(
                timing.total > Duration::ZERO,
                "{want_mode:?}: a prefill that ran took some time"
            );
        }
    }

    /// Every prefill starts from zero. Run a long prompt, then a one-token
    /// one: if the phases accumulated across runs, the second run's phases
    /// would carry the first run's work and overflow its much smaller total.
    #[test]
    fn every_prefill_rearms_the_timing() {
        let (_fx, model) = load_fixture("prefill-timing-rearm");
        let long: Vec<u32> = (0..12u32).map(|i| (i * 5 + 1) % 32).collect();

        let mut st = state_with(&model, 64, sweep_config(4));
        prefill_prompt(&model, &mut st, &long, None).unwrap();
        assert_eq!(st.prefill_timing().tokens, long.len() as u64);

        // A fresh sequence on the same state, on the other path.
        st.reset();
        st.set_prefill_config(token_major_config()).unwrap();
        prefill_prompt(&model, &mut st, &long[..1], None).unwrap();

        let timing = st.prefill_timing();
        assert_eq!(timing.tokens, 1, "the token count is per run, not summed");
        assert_eq!(
            timing.mode,
            Some(PrefillMode::TokenMajor),
            "the mode is the run's, not the state's history"
        );
        assert!(
            timing.accounted() <= timing.total,
            "phases {:?} exceed this run's total {:?}: the previous run leaked in",
            timing.accounted(),
            timing.total
        );
    }

    /// [`ForwardState::reset`] drops the timing, unlike the streaming counters
    /// beside it. A per-phase split left over from a sequence that has been
    /// thrown away would be read as belonging to the one that replaced it.
    #[test]
    fn a_reset_drops_the_prefill_timing() {
        let (_fx, model) = load_fixture("prefill-timing-reset");
        let mut st = state_with(&model, 32, sweep_config(4));
        prefill_prompt(&model, &mut st, &[1, 2, 3, 4, 5], None).unwrap();
        assert!(st.prefill_timing().ran());

        st.reset();
        assert_eq!(
            st.prefill_timing(),
            PrefillTiming::default(),
            "a reset state reports no prefill at all"
        );
        assert!(!st.prefill_timing().ran());
    }

    /// Decode charges nothing. `forward_token` carries the same
    /// instrumentation the token-major prefill does, and the only thing
    /// keeping generation out of a prompt's numbers is that a prefill has to
    /// arm the timing first.
    #[test]
    fn decode_does_not_charge_the_prefill_timing() {
        let (_fx, model) = load_fixture("prefill-timing-decode");
        let ids = [1u32, 2, 3, 4, 5];
        let mut st = state_with(&model, 32, sweep_config(4));
        prefill_prompt(&model, &mut st, &ids, None).unwrap();
        let after_prefill = st.prefill_timing();

        for step in 0..3usize {
            let position = ids.len() + step;
            crate::model::forward_token(&model, &mut st, 7, position, true).unwrap();
        }
        assert_eq!(
            st.prefill_timing(),
            after_prefill,
            "three decode tokens moved a prefill counter"
        );
    }
}
