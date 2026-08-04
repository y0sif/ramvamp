//! The prefill sweep: one layer's expert file, front to back, in big reads.
//!
//! Decode and prefill want opposite things from the drive. Decode routes
//! `top_k` experts per token per layer and cannot know which until the router
//! has run, so it reads one blob at a time into a cache. Prefill has the whole
//! chunk's routing in hand before a single byte is read, and at a 512-token
//! chunk that routing touches ~100% of every layer's experts — so reading them
//! in file order, once, is strictly better than reading them in routing order,
//! 4096 times.
//!
//! ```text
//! let mut plan = SweepPlan::new();                       // once per process
//! for layer in 0..n_layers {
//!     let routed = chunk.routed_experts(layer);          // ascending ids
//!     let mut sweep = stream.sweep_layer(&mut plan, layer, routed, cfg)?;
//!     while let Some(expert) = sweep.next_expert()? {
//!         compute(expert.expert, &expert.view, rows_for(expert.expert));
//!     }
//!     sweep.finish()?;
//! }
//! ```
//!
//! # What it does not do
//!
//! **It does not touch the decode cache.** Replaying the prompt into the LFU
//! measured +0.09 points (EXP-005), which does not pay for the slot pressure,
//! so prefill deliberately leaves the cache cold. Taking the arena empties
//! every layer's slot occupancy for the same reason, from the other direction:
//! sweep bytes land in those buffers, so no entry may survive claiming to hold
//! an expert.
//!
//! **It does not allocate.** The window buffers are carved from the idle slot
//! pool ([`ExpertStream::take_arena`]), whose every byte is already resident
//! and pre-faulted; the in-flight window table is a fixed array; and
//! [`SweepPlan`] is the caller's to reuse across layers, so after the first
//! layer the geometry costs nothing either.
//!
//! # Geometry
//!
//! An expert file is `n_experts * stride` bytes with no header and no padding,
//! so expert `e` lives at `e * stride` and a run of `k` experts starting at `f`
//! is one read of `k * stride` bytes at `f * stride`. A *window* is such a run.
//!
//! Two dials, both the subject of a planned experiment:
//!
//! - **Experts per window** ([`DEFAULT_EXPERTS_PER_WINDOW`], 8). On the two
//!   shipped Qwen3-30B-A3B strides that is 23.34 MiB and 20.25 MiB, and
//!   `128 / 8 = 16` uniform windows per layer with no ragged tail. The size is
//!   *not* chosen because the drive prefers big blocks: EXP-019 refutes that,
//!   finding larger blocks neutral on one probed file and 15 to 16 percent
//!   worse on the other three, which retires the "16-24 MiB the drive wants"
//!   band this comment used to cite from EXP-008. What the drive tracks is
//!   total bytes in flight, and 8 experts at the default 2 windows in flight
//!   is 49.0 MB outstanding, which measured as the **best cell** in EXP-019's
//!   matrix (1.60 to 2.37 GB/s) and sits well inside the peak plateau that
//!   ends near 100 MB. So the dial is unchanged and its justification is not.
//! - **Windows in flight** ([`DEFAULT_WINDOWS_IN_FLIGHT`], 2). Double
//!   buffering: window `n + 1` is on the wire while the caller computes window
//!   `n`. [`LayerSweep::next_expert`] waits for one named window rather than
//!   for everything outstanding, which is what keeps the overlap real.
//!
//! # Skipping
//!
//! A window whose experts the chunk routes *none* of is never read. At 512
//! tokens that essentially never fires; at 128 tokens coverage is about 86%,
//! and skipping whole windows is how the reads stay large instead of
//! degenerating into a scatter of holes.
//!
//! # Alignment
//!
//! O_DIRECT needs 4096 on the file offset, the transfer length **and** the
//! destination address, and [`is_retryable`](super::stream) deliberately
//! excludes `EINVAL` — so an unaligned read is not a retry, it is a dead
//! prefill. Every one of the three is guaranteed by construction here:
//!
//! - [`ExpertStream::take_arena`] refuses unless every layer's slot pitch
//!   equals its stride, which is the same statement as "every stride is a 4096
//!   multiple". Offsets (`f * stride`) and lengths (`k * stride`) follow.
//! - The arena base is the slab base, allocated with
//!   [`SLOT_ALIGN`](crate::io::SLOT_ALIGN), and every buffer within it is a
//!   whole number of windows from that base.
//!
//! It is checked anyway, on every read, because "guaranteed by construction"
//! is a claim about code that changes.

use std::fmt;
use std::ptr::NonNull;

use thiserror::Error;

use super::direct;
use super::{ExpertStream, ExpertView, IoError};

/// Experts one window covers by default.
///
/// 23.34 MiB / 20.25 MiB on the two shipped strides, and an exact divisor of
/// the 128 experts per layer.
pub const DEFAULT_EXPERTS_PER_WINDOW: u32 = 8;

/// Windows the sweep keeps in flight by default: enough to overlap the read of
/// the next window with the compute of this one.
pub const DEFAULT_WINDOWS_IN_FLIGHT: u32 = 2;

/// Most windows the sweep will keep in flight.
///
/// The in-flight table is a fixed array so a sweep allocates nothing, and this
/// is its width. Well above anything useful: past a couple of windows the
/// arena is the cost, and extra bytes in flight stop paying. EXP-019 puts the
/// drive's peak plateau at roughly 100 MB outstanding, which 4 windows of 8
/// experts (97.9 MB) already reaches, and measures 15 to 18 percent below peak
/// past about 170 MB, which 8 windows (195.8 MB) is. This replaces an earlier
/// "the drive saturates at QD4 (EXP-008)" note whose premise EXP-019 retires.
pub const MAX_WINDOWS_IN_FLIGHT: u32 = 8;

/// Failures from planning or driving a prefill sweep.
///
/// Geometry reaches this module from a validated layout, but the dials and the
/// routed set come from the caller, so every rejection is a typed error.
/// [`SweepError::Io`] carries everything the shared read path raises.
#[derive(Debug, Error)]
pub enum SweepError {
    /// The read path failed: a window read errored terminally, the layer file
    /// could not be opened or verified, or a submission was refused.
    #[error(transparent)]
    Io(#[from] IoError),

    /// A dial was zero, or windows-in-flight exceeded
    /// [`MAX_WINDOWS_IN_FLIGHT`].
    #[error(
        "sweep dials out of range: {experts_per_window} experts/window and \
         {windows_in_flight} windows in flight (both must be >= 1, and windows \
         in flight at most {max})"
    )]
    BadDials {
        /// Experts per window the caller asked for.
        experts_per_window: u32,
        /// Windows in flight the caller asked for.
        windows_in_flight: u32,
        /// The in-flight ceiling.
        max: u32,
    },

    /// A routed expert id is at or past the layer's expert count.
    #[error("routed expert {expert} out of range ({n_experts} experts)")]
    ExpertOutOfRange {
        /// The offending id.
        expert: u32,
        /// Experts the layer has.
        n_experts: u32,
    },

    /// One window would be a read longer than a single SQE can carry.
    #[error(
        "a sweep window of {experts} experts x {stride} B is {bytes} B, past \
         the {max} B one read can carry; lower experts-per-window"
    )]
    WindowTooLarge {
        /// Experts the window covers.
        experts: u32,
        /// The layer's blob stride.
        stride: u64,
        /// Bytes the window would read, computed in 128 bits so that a stride
        /// past `u64::MAX / experts_per_window` is *reported* rather than an
        /// overflow panic in a `pub` function over `pub` fields.
        bytes: u128,
        /// The per-read ceiling.
        max: u64,
    },

    /// The arena cannot be taken while the decode cache has reads outstanding.
    #[error("the prefill arena cannot be taken with {outstanding} read(s) in flight")]
    ReadsInFlight {
        /// Reads the decode path has not awaited.
        outstanding: usize,
    },

    /// The arena cannot be taken with a decode step open.
    #[error("the prefill arena cannot be taken with a step open on layer {layer}")]
    StepOpen {
        /// Layer whose step is open.
        layer: u32,
    },

    /// Some layer's slots are padded, so the slot pool is not a gapless run of
    /// blob-sized buffers and cannot be carved into an arena.
    ///
    /// Equivalently: that layer's blob stride is not a multiple of 4096, which
    /// also rules out O_DIRECT for the whole install. Both shipped Qwen3
    /// strides are exact multiples; this is the guard against a future layout
    /// that is not.
    #[error(
        "layer {layer}: blob stride {stride} B pads to a {pitch} B slot pitch, \
         so the slot pool is not a gapless prefill arena"
    )]
    PaddedSlots {
        /// The offending layer.
        layer: u32,
        /// Its declared blob stride.
        stride: usize,
        /// The padded distance between slot bases.
        pitch: usize,
    },

    /// The slot pool is smaller than one window buffer.
    #[error(
        "the prefill arena needs {needed} B but the slot pool is {available} B; \
         lower experts-per-window or raise the expert cache budget"
    )]
    ArenaTooSmall {
        /// Bytes one window buffer needs.
        needed: u64,
        /// Bytes the pool has.
        available: u64,
    },

    /// An arena is already out, so a second carve would name the same bytes
    /// twice.
    ///
    /// Not reachable from ordinary control flow — every path that takes the
    /// arena parks the stream's `&mut` and gives it back on `Drop`, and both
    /// `?` and panics run `Drop`. Leaking is what gets past that, and
    /// [`std::mem::forget`] is safe Rust: a forgotten
    /// [`PrefillSession`](super::PrefillSession) ends its borrow without ever
    /// releasing, so the next
    /// [`begin_prefill`](crate::io::ExpertStream::begin_prefill) would hand out
    /// a second `&mut [u8]` over the first one's scratch.
    #[error("a prefill arena of {bytes} B is already out")]
    ArenaOut {
        /// Bytes the live arena covers.
        bytes: usize,
    },

    /// A window read was lost, so the arena's bytes can never be handed out
    /// again in this process.
    #[error(
        "the prefill arena was given up by a window read that could not be \
         reaped; this process cannot sweep again"
    )]
    ArenaPoisoned,

    /// A sweep operation ran with no arena out. A bug in this module, or the
    /// arena being taken away mid-sweep by [`SweepError::ArenaPoisoned`]'s
    /// cause.
    #[error("no prefill arena is out")]
    NoArena,

    /// The carve would cover a slot buffer that was given up to protect a read
    /// which could never be reaped, so the kernel may still be writing there.
    ///
    /// Retiring a slot leaks its buffer *inside* the slab rather than freeing
    /// it — that is the whole point, a late kernel write has to land somewhere
    /// harmless — so the bytes stay where they were and an arena carved over
    /// them would put a window read on a live DMA destination.
    #[error(
        "the prefill arena of {bytes} B would cover layer {layer}'s retired \
         slot buffer at slab offset {offset}; a read that could not be reaped \
         may still be writing there"
    )]
    ArenaOverRetired {
        /// Layer the retired buffer belonged to.
        layer: u32,
        /// Its offset from the slab base.
        offset: usize,
        /// Bytes the carve asked for.
        bytes: usize,
    },

    /// A lost window read forced every slot the arena covered to be retired,
    /// and that left a layer with fewer slots than the model routes experts
    /// per step.
    ///
    /// The decode cache for that layer can no longer serve a single step. Said
    /// here, at the cause, rather than as a
    /// [`CacheError::TooFewSlots`](crate::io::CacheError::TooFewSlots) several
    /// forward passes later. See [`ExpertStream::sweep_layer`]'s docs for the
    /// geometry that makes this reachable at the shipped dials.
    #[error(
        "a prefill window read could not be reaped: every slot the arena \
         covered is retired, which leaves layer {layer} with {slots} slot(s) \
         against a top_k of {top_k}, so that layer can no longer decode"
    )]
    CacheStranded {
        /// First layer left short.
        layer: u32,
        /// Slots it has left.
        slots: u32,
        /// Experts the model routes per step.
        top_k: u32,
        /// The read failure that started it.
        #[source]
        source: IoError,
    },

    /// The sweep already failed and cannot be resumed.
    ///
    /// [`LayerSweep::next_expert`]'s first failure is latched and returned as
    /// it was raised; every *later* call reports this instead of walking a
    /// cursor that has already moved, because the window it would hand out was
    /// never filled, so its bytes are a previous window's or the zeros the slab
    /// was born with — silently wrong expert weights with no error and no log
    /// line. [`LayerSweep::finish`] reports the latched error rather than this,
    /// so nothing that only looks there loses an errno.
    #[error("this prefill sweep already failed and cannot be resumed: {reason}")]
    Aborted {
        /// What the original failure said.
        reason: String,
    },

    /// A read was about to be submitted with an offset, length, or destination
    /// direct I/O would answer `EINVAL` for.
    #[error("{what} is {value}, not a multiple of {align}")]
    Misaligned {
        /// Which of the three quantities.
        what: &'static str,
        /// The value that failed.
        value: u64,
        /// The alignment required.
        align: u64,
    },
}

/// Reject a quantity direct I/O would refuse.
///
/// `EINVAL` is not in [`is_retryable`](super::stream)'s list on purpose, so an
/// unaligned read fails the whole prefill with no second attempt. Every caller
/// establishes alignment from the geometry first; this is the assertion that
/// the geometry still says what it used to.
pub(super) fn check_aligned(what: &'static str, value: u64) -> Result<(), SweepError> {
    if direct::is_aligned(value) {
        Ok(())
    } else {
        Err(SweepError::Misaligned {
            what,
            value,
            align: direct::DIO_ALIGN,
        })
    }
}

/// How wide the sweep reads and how far ahead it reads.
///
/// Both dials are the subject of a planned experiment; the defaults are the
/// reasoned starting point, not a measured optimum. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SweepConfig {
    /// Experts one window read covers.
    pub experts_per_window: u32,
    /// Windows kept in flight, and therefore buffers carved from the arena.
    pub windows_in_flight: u32,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            experts_per_window: DEFAULT_EXPERTS_PER_WINDOW,
            windows_in_flight: DEFAULT_WINDOWS_IN_FLIGHT,
        }
    }
}

impl SweepConfig {
    /// Reject a degenerate or oversized dial pair.
    ///
    /// # Errors
    ///
    /// [`SweepError::BadDials`] for a zero on either dial, or more windows in
    /// flight than [`MAX_WINDOWS_IN_FLIGHT`].
    pub fn validate(self) -> Result<Self, SweepError> {
        if self.experts_per_window == 0
            || self.windows_in_flight == 0
            || self.windows_in_flight > MAX_WINDOWS_IN_FLIGHT
        {
            return Err(SweepError::BadDials {
                experts_per_window: self.experts_per_window,
                windows_in_flight: self.windows_in_flight,
                max: MAX_WINDOWS_IN_FLIGHT,
            });
        }
        Ok(self)
    }

    /// Bytes one window buffer occupies, at this layer's stride.
    ///
    /// The full width even for a ragged last window: buffers are uniform so the
    /// address arithmetic is.
    ///
    /// # Errors
    ///
    /// [`SweepError::WindowTooLarge`] when the product does not fit a `u64`.
    /// Unreachable at any real geometry — a blob stride is narrowed to `u32`
    /// when the stream opens — but both fields are `pub` on a `pub` struct, so
    /// this is a value the caller can choose and an overflow panic in a
    /// library is not an answer.
    pub fn window_bytes(self, stride: u64) -> Result<u64, SweepError> {
        let bytes = u128::from(self.experts_per_window) * u128::from(stride);
        u64::try_from(bytes).map_err(|_| SweepError::WindowTooLarge {
            experts: self.experts_per_window,
            stride,
            bytes,
            max: u64::from(u32::MAX),
        })
    }
}

impl fmt::Display for SweepConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} experts/window x {} in flight",
            self.experts_per_window, self.windows_in_flight
        )
    }
}

/// One front-to-back read of a run of consecutive experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SweepWindow {
    /// Lowest expert id the window covers.
    pub first_expert: u32,
    /// Experts covered. The full dial except for a ragged last window.
    pub n_experts: u32,
    /// Byte offset of the window in the layer file: `first_expert * stride`.
    pub file_offset: u64,
    /// Bytes the window reads: `n_experts * stride`.
    pub len: u64,
    /// Whether the chunk routes any of these experts. A window with none is
    /// never read.
    pub routed: bool,
}

/// A layer's window geometry and routed mask, sized once and refilled per
/// layer.
///
/// Hand the same value to every [`ExpertStream::sweep_layer`] of a chunk:
/// [`SweepPlan::build`] clears and refills it, so only the first layer of the
/// first chunk allocates.
#[derive(Debug, Clone, Default)]
pub struct SweepPlan {
    /// Every window, skipped ones included, in ascending expert order.
    windows: Vec<SweepWindow>,
    /// Whether each expert of the layer is routed by the chunk.
    routed: Vec<bool>,
    /// Windows with at least one routed expert.
    n_read: usize,
    /// The layer's blob stride, for the byte arithmetic.
    stride: u64,
}

impl SweepPlan {
    /// An empty plan that will grow into whatever the first layer needs.
    pub fn new() -> Self {
        Self::default()
    }

    /// Lay out `n_experts` experts of `stride` bytes into windows, marking each
    /// as routed or skippable against `routed`.
    ///
    /// `routed` is a set of expert ids in any order; duplicates are harmless.
    /// An empty set marks every window skippable, which is a legal (if
    /// pointless) sweep.
    ///
    /// # Errors
    ///
    /// [`SweepError::BadDials`] for a degenerate dial;
    /// [`SweepError::ExpertOutOfRange`] for a routed id past `n_experts`;
    /// [`SweepError::WindowTooLarge`] when one window would exceed what a
    /// single read can carry.
    pub fn build(
        &mut self,
        n_experts: u32,
        stride: u64,
        config: SweepConfig,
        routed: &[u32],
    ) -> Result<(), SweepError> {
        let config = config.validate()?;
        let bytes = config.window_bytes(stride)?;
        if bytes > u64::from(u32::MAX) {
            return Err(SweepError::WindowTooLarge {
                experts: config.experts_per_window,
                stride,
                bytes: u128::from(bytes),
                max: u64::from(u32::MAX),
            });
        }

        self.stride = stride;
        self.routed.clear();
        self.routed.resize(n_experts as usize, false);
        for &expert in routed {
            let flag = self
                .routed
                .get_mut(expert as usize)
                .ok_or(SweepError::ExpertOutOfRange { expert, n_experts })?;
            *flag = true;
        }

        self.windows.clear();
        self.n_read = 0;
        let per = config.experts_per_window;
        for first in (0..n_experts).step_by(per as usize) {
            // The last window is short whenever `per` does not divide
            // `n_experts`; 128/8 leaves no tail on the shipped model.
            let covered = per.min(n_experts - first);
            let is_routed = self.routed[first as usize..(first + covered) as usize]
                .iter()
                .any(|&flag| flag);
            if is_routed {
                self.n_read += 1;
            }
            self.windows.push(SweepWindow {
                first_expert: first,
                n_experts: covered,
                file_offset: u64::from(first) * stride,
                len: u64::from(covered) * stride,
                routed: is_routed,
            });
        }
        Ok(())
    }

    /// Every window of the layer, skipped ones included.
    pub fn windows(&self) -> &[SweepWindow] {
        &self.windows
    }

    /// Whether the chunk routes `expert`.
    pub fn is_routed(&self, expert: u32) -> bool {
        self.routed.get(expert as usize).copied().unwrap_or(false)
    }

    /// Windows that will be read.
    pub fn windows_to_read(&self) -> usize {
        self.n_read
    }

    /// Windows the routed set lets the sweep skip entirely.
    pub fn windows_skipped(&self) -> usize {
        self.windows.len() - self.n_read
    }

    /// The blob stride the plan was built against.
    pub fn stride(&self) -> u64 {
        self.stride
    }
}

/// One expert's bytes, valid until the sweep advances.
///
/// The lifetime is the sweep's `&mut` borrow, so the borrow checker is what
/// enforces "until the caller advances": calling
/// [`next_expert`](LayerSweep::next_expert) again requires this value to be
/// dead.
#[derive(Debug)]
pub struct SweepExpert<'a> {
    /// The expert's id within the layer.
    pub expert: u32,
    /// The whole blob, exactly the layer's stride long.
    pub bytes: &'a [u8],
    /// The blob carved into its gate/up/down projection slabs.
    pub view: ExpertView<'a>,
}

/// A window buffer that is filling or filled.
#[derive(Debug, Clone, Copy)]
struct Pending {
    /// Index into the sweep's window list.
    window: usize,
    /// Index into the stream's in-flight table.
    read: usize,
}

/// A layer's expert file, streamed front to back.
///
/// Holds the [`ExpertStream`] exclusively for its whole life, which is what
/// makes borrowing the slot pool as scratch sound: no cache operation is
/// reachable while a sweep is running. See `io::stream`'s module docs.
///
/// Dropping finishes the sweep. [`LayerSweep::finish`] does the same thing and
/// reports what went wrong.
pub struct LayerSweep<'a> {
    stream: &'a mut ExpertStream,
    plan: &'a SweepPlan,
    layer: u32,
    /// The layer's blob stride, narrowed once.
    stride: usize,
    /// Bytes one arena buffer occupies.
    window_bytes: usize,
    /// Byte offset of buffer 0 within the arena.
    ///
    /// Zero for a sweep that took the arena itself. A sweep running inside a
    /// [`PrefillSession`] is handed the ring *behind* the session's scratch
    /// span, and every offset it computes is relative to the arena base, so
    /// the two spans stay disjoint by arithmetic rather than by convention.
    ring_base: usize,
    /// Buffers carved from the arena, `<= MAX_WINDOWS_IN_FLIGHT` of them.
    in_flight: [Option<Pending>; MAX_WINDOWS_IN_FLIGHT as usize],
    /// Buffers actually carved.
    buffers: usize,
    /// Next window index to submit; skipped windows are stepped over.
    next_submit: usize,
    /// Next window index to hand to the caller.
    next_consume: usize,
    /// The window being handed out, as `(window index, buffer)`.
    current: Option<(usize, usize)>,
    /// Next expert offset within the current window.
    cursor: u32,
    /// Windows accounted for in the stream's read/skipped counters. Only for
    /// the teardown log line; the counters themselves live on the stream.
    counted: usize,
    /// The first failure, latched.
    ///
    /// Typed, not rendered: a driver that reaches the failure through
    /// [`finish`](Self::finish) rather than through the call that raised it
    /// still gets a `SweepError::Io` it can tell `ENOSPC` from `EIO` in.
    ///
    /// A `SweepError` is not `Clone` — it carries an `io::Error` — and the
    /// original leaves by value on the call that raised it, so what is latched
    /// is [`latchable`]'s reconstruction of it. See [`SweepError::Aborted`] for
    /// why anything at all has to be kept.
    failed: Option<SweepError>,
    /// Whether teardown gives the arena back. False inside a
    /// [`PrefillSession`], which owns the arena across every layer.
    owns_arena: bool,
    /// Whether the teardown has run.
    finished: bool,
}

impl fmt::Debug for LayerSweep<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayerSweep")
            .field("layer", &self.layer)
            .field("windows", &self.plan.windows().len())
            .field("to_read", &self.plan.windows_to_read())
            .field("buffers", &self.buffers)
            .field("consumed", &self.next_consume)
            .finish()
    }
}

impl<'a> LayerSweep<'a> {
    /// Plan `layer`, take the arena, and put the first windows on the wire.
    ///
    /// Reached through [`ExpertStream::sweep_layer`], which is the entry point
    /// the driver uses.
    pub(super) fn begin(
        stream: &'a mut ExpertStream,
        plan: &'a mut SweepPlan,
        layer: u32,
        routed: &[u32],
        config: SweepConfig,
    ) -> Result<Self, SweepError> {
        let config = config.validate()?;
        let (stride, n_experts) = stream.layer_geometry(layer)?;
        plan.build(n_experts, stride, config, routed)?;
        let window_bytes = window_span(config, stride, n_experts)?;
        let available = stream.cache_bytes();
        let buffers = fit_buffers(
            layer,
            config,
            window_bytes,
            available,
            plan.windows_to_read(),
        )?;

        // The file is opened and verified through the same first-use path
        // `begin_layer` takes, before the arena is out: an error here leaves
        // the cache untouched.
        stream.open_layer(layer)?;
        stream.take_arena(buffers * window_bytes as usize)?;
        Self::start(stream, plan, layer, stride, window_bytes, buffers, 0, true)
    }

    /// Sweep `layer` into a ring the caller already carved.
    ///
    /// The [`PrefillSession`] path: the arena is out for the whole prefill and
    /// its head is the driver's scratch, so this claims neither and works
    /// `ring_bytes` in from `ring_base`. The arena is *not* given back at
    /// teardown; the session owns it.
    ///
    /// # The guard this does not inherit
    ///
    /// [`begin`](Self::begin) reaches [`ExpertStream::take_arena`]'s checks by
    /// taking the arena. This one deliberately does not take it, so it
    /// inherits none of them — including the one that matters here, that
    /// nothing is still writing into the pool. The window ring is reused layer
    /// after layer, and every sweep starts by submitting into buffer 0, so a
    /// previous layer's windows still in flight would be a second O_DIRECT read
    /// onto a live DMA destination: the corruption EXP-007 measured as 13-27%
    /// spurious btrfs `EIO` plus permanent `corruption_errs`.
    ///
    /// A `LayerSweep` drains its own windows in `teardown`, which both `?` and
    /// a panic run — but [`std::mem::forget`] is safe Rust and skips it, and
    /// this is a `pub` API two lanes up. So the check is made here rather than
    /// argued from the caller.
    pub(super) fn begin_within(
        stream: &'a mut ExpertStream,
        plan: &'a mut SweepPlan,
        layer: u32,
        routed: &[u32],
        config: SweepConfig,
        ring_base: usize,
        ring_bytes: u64,
    ) -> Result<Self, SweepError> {
        let outstanding = stream.reads_outstanding();
        if outstanding > 0 {
            tracing::error!(
                layer,
                outstanding,
                "refusing a sweep into a ring a previous sweep's windows are \
                 still writing into"
            );
            return Err(SweepError::ReadsInFlight { outstanding });
        }
        let config = config.validate()?;
        let (stride, n_experts) = stream.layer_geometry(layer)?;
        plan.build(n_experts, stride, config, routed)?;
        let window_bytes = window_span(config, stride, n_experts)?;
        let buffers = fit_buffers(
            layer,
            config,
            window_bytes,
            ring_bytes,
            plan.windows_to_read(),
        )?;
        stream.open_layer(layer)?;
        Self::start(
            stream,
            plan,
            layer,
            stride,
            window_bytes,
            buffers,
            ring_base,
            false,
        )
    }

    /// Assemble the sweep and put the first windows on the wire.
    #[allow(clippy::too_many_arguments)]
    fn start(
        stream: &'a mut ExpertStream,
        plan: &'a SweepPlan,
        layer: u32,
        stride: u64,
        window_bytes: u64,
        buffers: usize,
        ring_base: usize,
        owns_arena: bool,
    ) -> Result<Self, SweepError> {
        let mut sweep = Self {
            stream,
            plan,
            layer,
            // Narrowing cannot fail: `window_bytes >= stride` was just
            // compared against a `usize` byte count.
            stride: stride as usize,
            window_bytes: window_bytes as usize,
            ring_base,
            in_flight: [None; MAX_WINDOWS_IN_FLIGHT as usize],
            buffers,
            next_submit: 0,
            next_consume: 0,
            current: None,
            cursor: 0,
            counted: 0,
            failed: None,
            owns_arena,
            finished: false,
        };
        if let Err(error) = sweep.pump() {
            // `teardown` drains whatever did reach the kernel and gives the
            // arena back, so a failed prime does not strand the pool.
            let _ = sweep.teardown();
            return Err(error);
        }
        Ok(sweep)
    }

    /// The layer being swept.
    pub fn layer(&self) -> u32 {
        self.layer
    }

    /// The window geometry this sweep is running.
    pub fn plan(&self) -> &SweepPlan {
        self.plan
    }

    /// Buffers carved from the arena, which is the windows actually in flight.
    ///
    /// At most the [`SweepConfig::windows_in_flight`] dial, and less when the
    /// layer has fewer windows than that or the pool cannot hold them.
    pub fn buffers(&self) -> usize {
        self.buffers
    }

    /// The next routed expert, in ascending id, blocking for its window's read.
    ///
    /// Returns `None` once every routed expert of the layer has been handed
    /// out. Experts the chunk does not route are skipped: they have no rows to
    /// compute against, and a window with none of them is never read at all.
    ///
    /// The returned bytes live in the arena and stay valid until this is called
    /// again — which the borrow checker enforces, since the result borrows the
    /// sweep.
    ///
    /// # Errors
    ///
    /// [`SweepError::Io`] when a window read fails terminally, after its one
    /// retry; [`SweepError::Misaligned`] if a window ever came out
    /// unaligned, which the geometry rules out; [`SweepError::Aborted`] on
    /// every call after the first failure.
    ///
    /// **The first failure is final.** Nothing in the signature forbids
    /// calling this again — it takes `&mut self`, and only
    /// [`finish`](Self::finish) consumes — but a retry would walk a cursor
    /// that has already moved, over a window whose read never filled it. So
    /// the failure is latched and every later call reports
    /// [`SweepError::Aborted`] rather than handing back arena bytes that hold
    /// a previous window, or the zeros the slab was born with.
    pub fn next_expert(&mut self) -> Result<Option<SweepExpert<'_>>, SweepError> {
        if let Some(error) = &self.failed {
            return Err(SweepError::Aborted {
                reason: error.to_string(),
            });
        }
        let position = match self.advance() {
            Ok(Some(position)) => position,
            Ok(None) => return Ok(None),
            Err(error) => return Err(self.latch(error)),
        };
        let (expert, offset) = position;
        // SAFETY: `advance` returned this offset only after awaiting the read
        // that filled it, so no kernel write is outstanding into the range;
        // and the arena is exclusive to this sweep for as long as it holds the
        // stream's `&mut`, which is `'_` here.
        let Some(bytes) = (unsafe { self.stream.arena_bytes(offset, self.stride) }) else {
            // The arena went away under a live sweep, which is `strand_arena`
            // and nothing else. The cursor has already stepped past this
            // expert, so a retry would *skip* it rather than repeat it —
            // latch, exactly as for a failed read.
            self.failed = Some(SweepError::NoArena);
            return Err(SweepError::NoArena);
        };
        // The latch is written field by field rather than through `latch`:
        // `bytes` borrows `self.stream`, and a `&mut self` method would
        // conflict with a borrow the success path still needs.
        let view = match self.stream.expert_reader().view_over(self.layer, bytes) {
            Ok(view) => view,
            Err(error) => {
                let error = SweepError::Io(error);
                self.failed = Some(latchable(&error));
                return Err(error);
            }
        };
        Ok(Some(SweepExpert {
            expert,
            bytes,
            view,
        }))
    }

    /// Whether this sweep has failed and will refuse every further expert.
    pub fn is_aborted(&self) -> bool {
        self.failed.is_some()
    }

    /// Finish the sweep: drain every window still on the wire and give the
    /// arena back.
    ///
    /// Dropping does the same thing; this reports what went wrong instead of
    /// logging it. Idempotent.
    ///
    /// # Errors
    ///
    /// [`SweepError::Io`] when a window still in flight failed; otherwise the
    /// first failure an earlier [`next_expert`](Self::next_expert) latched,
    /// reported as it was raised — a sweep that stopped short of the layer did
    /// not do what the caller asked, and saying so is the point of calling this
    /// rather than dropping. The latched error keeps its type, so a driver that
    /// only looks here can still tell `ENOSPC` from `EIO`.
    pub fn finish(mut self) -> Result<(), SweepError> {
        self.teardown()
    }

    /// Record the first failure and hand it back unchanged.
    ///
    /// The caller gets the error it caused, by value and unaltered; the latch
    /// keeps [`latchable`]'s reconstruction of it, which is the closest a
    /// non-`Clone` error gets to being in two places at once.
    fn latch(&mut self, error: SweepError) -> SweepError {
        if self.failed.is_none() {
            self.failed = Some(latchable(&error));
        }
        error
    }

    /// Position the sweep on the next routed expert, returning its id and its
    /// byte offset into the arena.
    ///
    /// Split out of [`LayerSweep::next_expert`] so that no borrow of `self`
    /// escapes a loop that keeps mutating it.
    fn advance(&mut self) -> Result<Option<(u32, usize)>, SweepError> {
        loop {
            if let Some((window, buffer)) = self.current {
                let win = self.plan.windows()[window];
                while self.cursor < win.n_experts {
                    let within = self.cursor;
                    self.cursor += 1;
                    let expert = win.first_expert + within;
                    if self.plan.is_routed(expert) {
                        let offset = self.ring_base
                            + buffer * self.window_bytes
                            + within as usize * self.stride;
                        return Ok(Some((expert, offset)));
                    }
                }
                // Window exhausted. Free its buffer and refill it *before*
                // blocking on the next window, so the drive keeps working
                // while the caller computes what comes back.
                self.in_flight[buffer] = None;
                self.current = None;
                self.next_consume = window + 1;
                self.pump()?;
                continue;
            }
            // Step over windows the chunk routes nothing in; they were never
            // submitted. Counted here, as they are stepped over, rather than
            // all at once when the sweep starts: `next_consume` is the one
            // monotone walk over every window, so counting both halves on it
            // keeps `read + skipped` a description of what the sweep actually
            // did. Counting the skips up front and the reads one at a time
            // made an aborted sweep report a coverage it never achieved.
            while self
                .plan
                .windows()
                .get(self.next_consume)
                .is_some_and(|win| !win.routed)
            {
                self.stream.count_window(false);
                self.counted += 1;
                self.next_consume += 1;
            }
            if self.next_consume >= self.plan.windows().len() {
                return Ok(None);
            }
            // `pump` submits routed windows in order and frees buffers in the
            // same order, so the one the caller is owed is always in flight.
            let Some(buffer) = self.buffer_of(self.next_consume) else {
                return Err(SweepError::NoArena);
            };
            let read = self.in_flight[buffer].map(|pending| pending.read);
            self.stream.sweep_await(read)?;
            self.stream.count_window(true);
            self.counted += 1;
            self.current = Some((self.next_consume, buffer));
            self.cursor = 0;
        }
    }

    /// Submit routed windows into every free buffer.
    fn pump(&mut self) -> Result<(), SweepError> {
        loop {
            // Step over skipped windows without submitting anything.
            while self
                .plan
                .windows()
                .get(self.next_submit)
                .is_some_and(|win| !win.routed)
            {
                self.next_submit += 1;
            }
            let Some(&win) = self.plan.windows().get(self.next_submit) else {
                return Ok(());
            };
            let Some(buffer) = self.free_buffer() else {
                return Ok(());
            };
            let arena_offset = self.ring_base + buffer * self.window_bytes;
            check_aligned("sweep window file offset", win.file_offset)?;
            check_aligned("sweep window length", win.len)?;
            check_aligned("sweep window arena offset", arena_offset as u64)?;
            // Narrowing is safe: `SweepPlan::build` refused a window wider than
            // `u32::MAX`, and `len <= window_bytes`.
            let len = u32::try_from(win.len).map_err(|_| SweepError::WindowTooLarge {
                experts: win.n_experts,
                stride: self.stride as u64,
                bytes: u128::from(win.len),
                max: u64::from(u32::MAX),
            })?;
            let read = self.stream.sweep_submit(
                self.layer,
                buffer as u32,
                win.first_expert,
                win.file_offset,
                len,
                arena_offset,
            )?;
            self.in_flight[buffer] = Some(Pending {
                window: self.next_submit,
                read,
            });
            self.next_submit += 1;
        }
    }

    /// A buffer holding nothing.
    fn free_buffer(&self) -> Option<usize> {
        (0..self.buffers).find(|&buffer| self.in_flight[buffer].is_none())
    }

    /// The buffer window `window` was submitted into.
    fn buffer_of(&self, window: usize) -> Option<usize> {
        (0..self.buffers).find(|&buffer| self.in_flight[buffer].is_some_and(|p| p.window == window))
    }

    /// Drain and release, once.
    fn teardown(&mut self) -> Result<(), SweepError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let windows = self.plan.windows().len();
        if self.counted < windows {
            tracing::warn!(
                layer = self.layer,
                covered = self.counted,
                windows,
                "prefill sweep torn down before it covered the layer; its \
                 window counters describe what it did, not what it planned"
            );
        }
        // Every window still on the wire is writing into the arena, so it has
        // to be reaped before the pool goes back to being a cache.
        let result = self.stream.sweep_await(None);
        self.stream.sweep_clear_reads();
        if self.owns_arena {
            self.stream.release_arena();
        }
        match (result, self.failed.take()) {
            // The drain's own failure first: it is the newer fact, and the
            // latched one has already been reported to whoever caused it.
            (Err(error), _) => Err(error),
            // The latched one moves out as it was, rather than as a message: a
            // driver that reaches the failure here — because it discarded what
            // `next_expert` returned — still gets the errno.
            (Ok(()), Some(error)) => Err(error),
            (Ok(()), None) => Ok(()),
        }
    }
}

impl Drop for LayerSweep<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.teardown() {
            tracing::error!(
                layer = self.layer,
                %error,
                "prefill sweep torn down without completing the layer"
            );
        }
    }
}

/// A copy of a failure, good enough to latch, keeping the errno where there is
/// one.
///
/// [`SweepError`] is not `Clone` — `io::Error` is not — and the original has to
/// leave by value on the call that raised it, so the latch keeps this instead.
/// The reconstruction is exact for the one shape that carries an errno; every
/// other variant is plain data whose own message is all it ever had, and
/// becomes the [`SweepError::Aborted`] a later `next_expert` would have made of
/// it anyway.
///
/// Rendering *everything* — which is what this replaced — turned `ENOSPC` into
/// a string, the same loss `ExpertStream`'s `flatten_io` exists to prevent on
/// the decode path.
fn latchable(error: &SweepError) -> SweepError {
    match error {
        SweepError::Io(IoError::Io { path, source }) => SweepError::Io(IoError::Io {
            path: path.clone(),
            source: dup_os(source),
        }),
        SweepError::CacheStranded {
            layer,
            slots,
            top_k,
            source: IoError::Io { path, source },
        } => SweepError::CacheStranded {
            layer: *layer,
            slots: *slots,
            top_k: *top_k,
            source: IoError::Io {
                path: path.clone(),
                source: dup_os(source),
            },
        },
        other => SweepError::Aborted {
            reason: other.to_string(),
        },
    }
}

/// An `io::Error` rebuilt from its errno, or from its kind and message when it
/// never had one (an `UnexpectedEof` this module synthesised, say).
fn dup_os(error: &std::io::Error) -> std::io::Error {
    error.raw_os_error().map_or_else(
        || std::io::Error::new(error.kind(), error.to_string()),
        std::io::Error::from_raw_os_error,
    )
}

/// Bytes one window buffer occupies for a layer of `n_experts` experts.
///
/// Uniform, so a ragged last window still lands at a buffer base. Never wider
/// than the layer — a window cannot cover experts that do not exist, and
/// reserving for them would refuse arenas that are in fact big enough.
fn window_span(config: SweepConfig, stride: u64, n_experts: u32) -> Result<u64, SweepError> {
    SweepConfig {
        experts_per_window: config.experts_per_window.min(n_experts.max(1)),
        ..config
    }
    .window_bytes(stride)
}

/// Buffers of `window_bytes` that fit in `available` bytes of ring, clamped to
/// the dial and to the windows there are to put in them.
///
/// # Errors
///
/// [`SweepError::ArenaTooSmall`] when not even one buffer fits.
fn fit_buffers(
    layer: u32,
    config: SweepConfig,
    window_bytes: u64,
    available: u64,
    windows_to_read: usize,
) -> Result<usize, SweepError> {
    if window_bytes > available {
        return Err(SweepError::ArenaTooSmall {
            needed: window_bytes,
            available,
        });
    }
    let buffers = (config.windows_in_flight as usize)
        .min(windows_to_read.max(1))
        .min((available / window_bytes) as usize)
        .max(1);
    if buffers < config.windows_in_flight as usize {
        tracing::debug!(
            layer,
            asked = config.windows_in_flight,
            buffers,
            window_bytes,
            ring_bytes = available,
            "fewer prefill windows in flight than asked for"
        );
    }
    Ok(buffers)
}

/// One chunked prefill: the slot-pool slab carved into driver scratch and a
/// sweep ring, for as long as the driver needs both at once.
///
/// # Why this exists
///
/// A layer-major prefill has to do two things *simultaneously*: consume swept
/// experts, and write each expert's output into an `[n_rows][top_k][hidden]`
/// f32 staging buffer. Both want to come out of the slot pool, because the
/// whole point is that prefill costs no bytes against the memory budget beyond
/// what the decode cache already made resident.
///
/// [`ExpertStream::sweep_layer`] cannot serve that: it parks the
/// `&mut ExpertStream` for the sweep's whole life, so the driver can hold
/// either the sweep or a scratch span, never both. This owns the `&mut`
/// instead and hands out the two as a split borrow — [`PrefillSession::split`]
/// returns a `&mut [u8]` and a [`LayerSweep`] whose byte ranges are disjoint by
/// construction, which is what makes two `&mut` into one allocation sound.
///
/// ```text
/// let mut session = stream.begin_prefill(staging_bytes, cfg)?;
/// for layer in 0..n_layers {
///     let (staging, mut sweep) = session.split(&mut plan, layer, routed)?;
///     while let Some(expert) = sweep.next_expert()? {
///         compute(&expert.view, staging);      // both borrows live at once
///     }
///     sweep.finish()?;
/// }
/// session.finish()?;
/// ```
///
/// # The carve
///
/// ```text
/// slab: [ scratch .. | pad to 4096 | ring: windows_in_flight buffers | unused ]
///       ^ arena base, 4096-aligned
/// ```
///
/// The scratch leads, so its base is the slab base and therefore
/// [`SLOT_ALIGN`](crate::io::SLOT_ALIGN)-aligned; the ring starts at the next
/// 4096 boundary past it, so every window offset stays legal for O_DIRECT. The
/// ring is sized for the *widest* layer, since one session spans all of them.
/// At the shipped geometry the slab is ~1,438 MiB against a ~47 MiB ring, so
/// this is an ownership problem and not a space one.
///
/// # The scratch is not zeroed
///
/// Taking it is address arithmetic over pages [`SlotPool`](crate::io::SlotPool)
/// already faulted in; memsetting 1.4 GiB to hand back zeros nobody asked for
/// would be a real cost on the measured path. **The driver must not assume
/// zeros**: the bytes are whatever the last expert read or the last prefill
/// left there. They are always *initialized* — the slab is allocated zeroed —
/// so the slice is sound, just not blank.
pub struct PrefillSession<'a> {
    stream: &'a mut ExpertStream,
    /// Base of the scratch span. The arena base, which is the slab base.
    scratch: NonNull<u8>,
    /// Bytes of scratch the caller asked for.
    scratch_len: usize,
    /// Byte offset of the sweep ring within the arena.
    ring_base: usize,
    /// Bytes of ring, enough for `windows_in_flight` buffers of the widest
    /// layer.
    ring_bytes: u64,
    /// Dials every layer of this prefill runs with.
    config: SweepConfig,
    /// Whether the teardown has run.
    finished: bool,
}

impl fmt::Debug for PrefillSession<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefillSession")
            .field("scratch_bytes", &self.scratch_len)
            .field("ring_base", &self.ring_base)
            .field("ring_bytes", &self.ring_bytes)
            .field("config", &self.config)
            .finish()
    }
}

impl<'a> PrefillSession<'a> {
    /// Carve the arena and open the session.
    ///
    /// Reached through [`ExpertStream::begin_prefill`].
    pub(super) fn begin(
        stream: &'a mut ExpertStream,
        scratch_bytes: usize,
        config: SweepConfig,
    ) -> Result<Self, SweepError> {
        let config = config.validate()?;
        let ring_bytes = ring_span(stream, config)?;
        let align = direct::DIO_ALIGN as usize;
        // The ring starts at the next page past the scratch, so the two spans
        // cannot overlap and every window offset stays 4096-aligned.
        let ring_base =
            scratch_bytes
                .checked_next_multiple_of(align)
                .ok_or(SweepError::ArenaTooSmall {
                    needed: u64::MAX,
                    available: stream.cache_bytes(),
                })?;
        let available = stream.cache_bytes();
        let total = (ring_base as u64)
            .checked_add(ring_bytes)
            .filter(|total| *total <= available)
            .ok_or(SweepError::ArenaTooSmall {
                needed: (ring_base as u64).saturating_add(ring_bytes),
                available,
            })?;
        // Narrowing cannot fail: `total <= available`, itself a `usize`.
        stream.take_arena(total as usize)?;
        let scratch = stream.arena_base().ok_or(SweepError::NoArena)?;
        debug_assert!(
            scratch_bytes <= ring_base,
            "the scratch span and the sweep ring must not overlap"
        );
        tracing::debug!(
            scratch_bytes,
            ring_base,
            ring_bytes,
            pool_bytes = available,
            "prefill session open"
        );
        Ok(Self {
            stream,
            scratch,
            scratch_len: scratch_bytes,
            ring_base,
            ring_bytes,
            config,
            finished: false,
        })
    }

    /// Bytes of scratch this session carved.
    pub fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// Byte offset of the sweep ring within the arena, which is the first byte
    /// past the scratch span that a window read may ever touch.
    pub fn ring_offset(&self) -> usize {
        self.ring_base
    }

    /// The dials every layer of this prefill runs with.
    pub fn config(&self) -> SweepConfig {
        self.config
    }

    /// The scratch span, with no sweep running.
    ///
    /// For the parts of a chunk that happen between layers. Use
    /// [`split`](Self::split) to hold it *and* a sweep at the same time.
    ///
    /// The contents are whatever was there before; see the type's docs.
    pub fn scratch(&mut self) -> &mut [u8] {
        // SAFETY: the span is the head of the arena, which is out for this
        // session's whole life — so no slot guard is reachable and no window
        // read can target it, the ring starting a page past its end. Every
        // byte was initialized by `SlotPool::new`. `&mut self` is what rules
        // out a second view.
        unsafe { std::slice::from_raw_parts_mut(self.scratch.as_ptr(), self.scratch_len) }
    }

    /// The split borrow: the scratch span and a sweep of `layer`, live at once.
    ///
    /// The two `&mut` name disjoint halves of one allocation — scratch at the
    /// arena base, window buffers a page past its end — which the borrow
    /// checker cannot see and [`PrefillSession::begin`] establishes by
    /// arithmetic.
    ///
    /// `plan` is the caller's to reuse across layers, exactly as for
    /// [`ExpertStream::sweep_layer`]. The returned sweep does **not** give the
    /// arena back when it is dropped; this session does, at
    /// [`finish`](Self::finish).
    ///
    /// # Errors
    ///
    /// Everything [`ExpertStream::sweep_layer`] raises except the arena-taking
    /// ones, which happened when the session opened:
    /// [`SweepError::ArenaTooSmall`] when the ring cannot hold one window of
    /// this layer, [`SweepError::BadDials`] or
    /// [`SweepError::ExpertOutOfRange`] for bad input, [`SweepError::Io`] when
    /// the layer file cannot be opened or the first reads cannot be submitted.
    pub fn split<'s>(
        &'s mut self,
        plan: &'s mut SweepPlan,
        layer: u32,
        routed: &[u32],
    ) -> Result<(&'s mut [u8], LayerSweep<'s>), SweepError> {
        let (scratch, scratch_len) = (self.scratch, self.scratch_len);
        let (ring_base, ring_bytes, config) = (self.ring_base, self.ring_bytes, self.config);
        let sweep = LayerSweep::begin_within(
            &mut *self.stream,
            plan,
            layer,
            routed,
            config,
            ring_base,
            ring_bytes,
        )?;
        // SAFETY: as `scratch`, plus the disjointness the split rests on — the
        // sweep's every destination is at `ring_base + ..`, and `ring_base` is
        // `scratch_len` rounded up to a page, so no window read and no
        // `SweepExpert` can name a byte of this slice.
        let staging = unsafe { std::slice::from_raw_parts_mut(scratch.as_ptr(), scratch_len) };
        Ok((staging, sweep))
    }

    /// Drain anything still on the wire and give the arena back.
    ///
    /// Dropping does the same thing; this reports what went wrong instead of
    /// logging it. Idempotent.
    ///
    /// # Errors
    ///
    /// [`SweepError::Io`] when a window read that outlived its sweep failed.
    pub fn finish(mut self) -> Result<(), SweepError> {
        self.teardown()
    }

    /// Drain and release, once.
    fn teardown(&mut self) -> Result<(), SweepError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        // Belt and braces: every `LayerSweep` drains its own windows, and this
        // catches one that was leaked rather than dropped.
        let result = self.stream.sweep_await(None);
        self.stream.sweep_clear_reads();
        self.stream.release_arena();
        result
    }
}

impl Drop for PrefillSession<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.teardown() {
            tracing::error!(
                %error,
                "prefill session torn down with a window read still failing"
            );
        }
    }
}

/// Bytes of ring one session needs: `windows_in_flight` buffers of the widest
/// window any layer of this model produces.
///
/// One session spans every layer and the strides differ (the shipped model has
/// two classes, 3,059,712 B and 2,654,208 B), so the ring is sized for the
/// worst of them rather than re-carved per layer.
fn ring_span(stream: &ExpertStream, config: SweepConfig) -> Result<u64, SweepError> {
    let mut widest = 0u64;
    for layer in 0..stream.n_layers() {
        let (stride, n_experts) = stream.layer_geometry(layer)?;
        widest = widest.max(window_span(config, stride, n_experts)?);
    }
    widest
        .checked_mul(u64::from(config.windows_in_flight))
        .ok_or(SweepError::ArenaTooSmall {
            needed: u64::MAX,
            available: stream.cache_bytes(),
        })
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Fixture, N_EXPERTS, N_LAYERS, build_install};
    use super::*;
    use crate::io::{ExpertReader, LoadOptions, SLOT_ALIGN};

    /// Bytes one slot per layer costs across the whole fixture.
    fn slot_row(fx: &Fixture) -> u64 {
        fx.layout
            .layers
            .iter()
            .map(|layer| layer.stride.next_multiple_of(SLOT_ALIGN as u64))
            .sum()
    }

    fn open(fx: &Fixture, slots: u32) -> ExpertStream {
        ExpertStream::new(
            &fx.root,
            &fx.manifest,
            &fx.layout,
            slot_row(fx) * u64::from(slots),
            LoadOptions::default(),
        )
        .expect("stream opens")
    }

    /// Every expert id of a fixture layer.
    fn all_experts(fx: &Fixture, layer: u32) -> Vec<u32> {
        (0..fx.layout.layers[layer as usize].n_experts).collect()
    }

    /// Sweep one layer and collect `(expert, blob)` in the order yielded.
    fn sweep_all(
        stream: &mut ExpertStream,
        layer: u32,
        routed: &[u32],
        config: SweepConfig,
    ) -> Vec<(u32, Vec<u8>)> {
        let mut plan = SweepPlan::new();
        let mut sweep = stream
            .sweep_layer(&mut plan, layer, routed, config)
            .expect("sweep begins");
        let mut out = Vec::new();
        while let Some(expert) = sweep.next_expert().expect("window reads") {
            out.push((expert.expert, expert.bytes.to_vec()));
        }
        sweep.finish().expect("sweep finishes");
        out
    }

    // ---- window arithmetic -------------------------------------------

    /// Both shipped stride classes, laid out into windows: 128 experts at 8
    /// per window is 16 uniform windows with no ragged tail, and every offset
    /// and length stays 4096-aligned.
    #[test]
    fn window_arithmetic_covers_both_stride_classes() {
        const QWEN_STRIDES: [u64; 2] = [3_059_712, 2_654_208];
        let config = SweepConfig::default();
        let routed: Vec<u32> = (0..128).collect();
        for stride in QWEN_STRIDES {
            let mut plan = SweepPlan::new();
            plan.build(128, stride, config, &routed).unwrap();
            assert_eq!(plan.windows().len(), 16, "128 / 8 is 16 whole windows");
            assert_eq!(plan.windows_to_read(), 16);
            assert_eq!(plan.windows_skipped(), 0);

            let mut expected_offset = 0;
            for (index, window) in plan.windows().iter().enumerate() {
                assert_eq!(window.first_expert, index as u32 * 8);
                assert_eq!(window.n_experts, 8, "no ragged tail at 128/8");
                assert_eq!(window.file_offset, expected_offset);
                assert_eq!(window.len, 8 * stride);
                assert!(window.routed);
                assert!(direct::is_aligned(window.file_offset));
                assert!(direct::is_aligned(window.len));
                expected_offset += window.len;
            }
            // The windows tile the file exactly.
            assert_eq!(expected_offset, 128 * stride);
            // 16-24 MiB. Not a band the drive requires (EXP-019 refutes the
            // block-size premise EXP-008 handed this dial); this pins the
            // window size the shipped dial actually produces, so a change to
            // either the dial or a stride has to be deliberate.
            let mib = config.window_bytes(stride).unwrap() as f64 / (1024.0 * 1024.0);
            assert!((16.0..=24.0).contains(&mib), "{mib} MiB out of band");
        }
    }

    /// A count the dial does not divide leaves exactly one short window, at
    /// the end, and it still tiles the file.
    #[test]
    fn the_last_window_is_short_when_the_dial_does_not_divide() {
        let stride = 8192;
        let config = SweepConfig {
            experts_per_window: 8,
            windows_in_flight: 2,
        };
        let routed: Vec<u32> = (0..130).collect();
        let mut plan = SweepPlan::new();
        plan.build(130, stride, config, &routed).unwrap();
        assert_eq!(plan.windows().len(), 17);
        let last = *plan.windows().last().unwrap();
        assert_eq!(last.first_expert, 128);
        assert_eq!(last.n_experts, 2, "130 - 16 * 8");
        assert_eq!(last.len, 2 * stride);
        assert_eq!(last.file_offset, 128 * stride);
        for window in &plan.windows()[..16] {
            assert_eq!(window.n_experts, 8);
        }
        assert_eq!(
            last.file_offset + last.len,
            130 * stride,
            "the windows must tile the whole file"
        );

        // And a single window shorter than the dial is legal.
        plan.build(3, stride, config, &[0, 1, 2]).unwrap();
        assert_eq!(plan.windows().len(), 1);
        assert_eq!(plan.windows()[0].n_experts, 3);
    }

    /// A window with no routed expert is marked skippable; one with a single
    /// routed expert is not.
    #[test]
    fn a_window_routing_nothing_is_skippable() {
        let stride = 4096;
        let config = SweepConfig {
            experts_per_window: 4,
            windows_in_flight: 2,
        };
        let mut plan = SweepPlan::new();
        // 16 experts, 4 windows; route one expert in window 0 and the last
        // expert of window 3.
        plan.build(16, stride, config, &[1, 15]).unwrap();
        let routed: Vec<bool> = plan.windows().iter().map(|w| w.routed).collect();
        assert_eq!(routed, vec![true, false, false, true]);
        assert_eq!(plan.windows_to_read(), 2);
        assert_eq!(plan.windows_skipped(), 2);

        // Nothing routed at all: every window is skippable.
        plan.build(16, stride, config, &[]).unwrap();
        assert_eq!(plan.windows_to_read(), 0);
        assert_eq!(plan.windows_skipped(), 4);

        // Everything routed: nothing is.
        let all: Vec<u32> = (0..16).collect();
        plan.build(16, stride, config, &all).unwrap();
        assert_eq!(plan.windows_to_read(), 4);
        assert_eq!(plan.windows_skipped(), 0);
    }

    #[test]
    fn a_routed_id_past_the_layer_is_a_typed_error() {
        let mut plan = SweepPlan::new();
        let err = plan
            .build(4, 4096, SweepConfig::default(), &[0, 4])
            .unwrap_err();
        assert!(
            matches!(
                err,
                SweepError::ExpertOutOfRange {
                    expert: 4,
                    n_experts: 4
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn degenerate_dials_are_refused() {
        for config in [
            SweepConfig {
                experts_per_window: 0,
                windows_in_flight: 2,
            },
            SweepConfig {
                experts_per_window: 8,
                windows_in_flight: 0,
            },
            SweepConfig {
                experts_per_window: 8,
                windows_in_flight: MAX_WINDOWS_IN_FLIGHT + 1,
            },
        ] {
            let err = config.validate().unwrap_err();
            assert!(
                matches!(err, SweepError::BadDials { .. }),
                "unexpected error: {err}"
            );
        }
        SweepConfig::default().validate().unwrap();
    }

    /// One window has to fit in a single SQE's `len`.
    #[test]
    fn a_window_past_u32_is_refused() {
        let mut plan = SweepPlan::new();
        let err = plan
            .build(
                64,
                1 << 30,
                SweepConfig {
                    experts_per_window: 8,
                    windows_in_flight: 2,
                },
                &[0],
            )
            .unwrap_err();
        assert!(
            matches!(err, SweepError::WindowTooLarge { .. }),
            "unexpected error: {err}"
        );
    }

    // ---- alignment ---------------------------------------------------

    /// Offsets, lengths and destinations are all checked, and the error names
    /// which one failed. `EINVAL` is not retryable, so this is the only place
    /// a misaligned sweep read can be caught.
    #[test]
    fn alignment_checks_reject_each_quantity() {
        check_aligned("sweep read offset", 0).unwrap();
        check_aligned("sweep read offset", 4096).unwrap();
        check_aligned("sweep read length", 3_059_712).unwrap();
        for (what, value) in [
            ("sweep read offset", 4095u64),
            ("sweep read length", 1),
            ("sweep read destination", 0x1_0000_0800),
        ] {
            let err = check_aligned(what, value).unwrap_err();
            match err {
                SweepError::Misaligned {
                    what: named,
                    value: got,
                    align,
                } => {
                    assert_eq!(named, what);
                    assert_eq!(got, value);
                    assert_eq!(align, direct::DIO_ALIGN);
                }
                other => panic!("unexpected error: {other}"),
            }
        }
    }

    // ---- end to end --------------------------------------------------

    /// The bytes a sweep hands back are the bytes on disk, expert for expert,
    /// and they arrive in ascending id.
    #[test]
    fn a_sweep_yields_the_same_bytes_as_read_expert() {
        let fx = build_install("sweep-bytes");
        let mut stream = open(&fx, 4);
        let reader = ExpertReader::new(&fx.root, &fx.manifest, &fx.layout, LoadOptions::default())
            .expect("reader opens");

        for layer in 0..fx.layout.layers.len() as u32 {
            let routed = all_experts(&fx, layer);
            let swept = sweep_all(&mut stream, layer, &routed, SweepConfig::default());
            assert_eq!(swept.len(), routed.len(), "layer {layer} expert count");

            let mut buf = Vec::new();
            for (index, (expert, bytes)) in swept.iter().enumerate() {
                assert_eq!(*expert, index as u32, "layer {layer} out of order");
                let view = reader.read_expert(layer, *expert, &mut buf).unwrap();
                assert_eq!(
                    bytes.as_slice(),
                    view.blob(),
                    "layer {layer} expert {expert} blob mismatch"
                );
            }
        }
    }

    /// The gate/up/down carve of a swept blob matches the layout, so the
    /// consumer needs nothing but the view.
    #[test]
    fn a_swept_view_carves_the_layout_slabs() {
        let fx = build_install("sweep-slabs");
        let mut stream = open(&fx, 4);
        let mut plan = SweepPlan::new();
        let routed = all_experts(&fx, 0);
        let mut sweep = stream
            .sweep_layer(&mut plan, 0, &routed, SweepConfig::default())
            .unwrap();
        let file = std::fs::read(fx.root.join(&fx.layout.layers[0].file)).unwrap();
        let stride = fx.layout.layers[0].stride as usize;
        while let Some(expert) = sweep.next_expert().unwrap() {
            let base = expert.expert as usize * stride;
            for projection in &fx.layout.layers[0].projections {
                let from = base + projection.offset_in_blob as usize;
                let to = from + projection.len as usize;
                assert_eq!(
                    expert.view.slab(projection.name).bytes,
                    &file[from..to],
                    "expert {} {:?} slab",
                    expert.expert,
                    projection.name
                );
            }
        }
        sweep.finish().unwrap();
    }

    /// A sparsely routed chunk sees only the experts it routes, and the
    /// windows with none of them are never read.
    #[test]
    fn a_sparse_chunk_skips_whole_windows() {
        let fx = build_install("sweep-skip");
        let mut stream = open(&fx, 4);
        // Two experts per window over four experts is two windows; route only
        // the second window's experts.
        let config = SweepConfig {
            experts_per_window: 2,
            windows_in_flight: 2,
        };
        let before = stream.stats();
        let swept = sweep_all(&mut stream, 1, &[2, 3], config);
        assert_eq!(
            swept.iter().map(|(e, _)| *e).collect::<Vec<_>>(),
            vec![2, 3]
        );
        let delta = stream.stats().since(&before);
        assert_eq!(delta.sweep_windows_read, 1);
        assert_eq!(delta.sweep_windows_skipped, 1);
        assert_eq!(
            delta.sweep_bytes_read,
            2 * fx.layout.layers[1].stride,
            "the skipped window's bytes were read anyway"
        );

        // And the bytes are still right.
        let reader = ExpertReader::new(&fx.root, &fx.manifest, &fx.layout, LoadOptions::default())
            .expect("reader opens");
        let mut buf = Vec::new();
        for (expert, bytes) in &swept {
            let view = reader.read_expert(1, *expert, &mut buf).unwrap();
            assert_eq!(bytes.as_slice(), view.blob());
        }
    }

    /// A routed set that covers nothing reads nothing, and still returns the
    /// arena.
    #[test]
    fn a_chunk_routing_nothing_reads_nothing() {
        let fx = build_install("sweep-empty");
        let mut stream = open(&fx, 4);
        let before = stream.stats();
        assert!(sweep_all(&mut stream, 0, &[], SweepConfig::default()).is_empty());
        let delta = stream.stats().since(&before);
        assert_eq!(delta.sweep_windows_read, 0);
        assert_eq!(delta.sweep_bytes_read, 0);
        assert!(delta.sweep_windows_skipped > 0);
        // The cache is usable again immediately.
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
    }

    /// Every window is read exactly once, whatever the dials, and the sweep
    /// works with one buffer as well as with several.
    #[test]
    fn every_dial_pair_reads_the_file_exactly_once() {
        let fx = build_install("sweep-dials");
        let stride = fx.layout.layers[0].stride;
        let routed = all_experts(&fx, 0);
        for experts_per_window in 1..=5u32 {
            for windows_in_flight in 1..=3u32 {
                let mut stream = open(&fx, 4);
                let config = SweepConfig {
                    experts_per_window,
                    windows_in_flight,
                };
                let swept = sweep_all(&mut stream, 0, &routed, config);
                assert_eq!(
                    swept.iter().map(|(e, _)| *e).collect::<Vec<_>>(),
                    routed,
                    "dials {config}"
                );
                let stats = stream.stats();
                assert_eq!(
                    stats.sweep_bytes_read,
                    u64::from(fx.layout.layers[0].n_experts) * stride,
                    "dials {config} read the file a number of times other than once"
                );
                assert_eq!(stats.sweep_windows_skipped, 0);
                assert_eq!(
                    stats.sweep_windows_read as usize,
                    (fx.layout.layers[0].n_experts as usize).div_ceil(experts_per_window as usize)
                );
                // Nothing landed in the cache half of the counters.
                assert_eq!(stats.accesses(), 0);
                assert_eq!(stats.bytes_read, 0);
            }
        }
    }

    /// The next window is on the wire before the caller sees the first byte of
    /// this one — which is the entire point of `windows_in_flight`.
    ///
    /// Guarded because it is easy to lose: waiting for *everything*
    /// outstanding, as the decode path does, would submit both windows and
    /// then block on both, and the read of window `n + 1` would no longer
    /// overlap the compute of window `n`.
    #[test]
    fn the_next_window_is_in_flight_before_this_one_is_consumed() {
        let fx = build_install("sweep-overlap");
        let mut stream = open(&fx, 4);
        let mut plan = SweepPlan::new();
        let routed = all_experts(&fx, 0);
        // Four experts, one per window, two buffers.
        let mut sweep = stream
            .sweep_layer(
                &mut plan,
                0,
                &routed,
                SweepConfig {
                    experts_per_window: 1,
                    windows_in_flight: 2,
                },
            )
            .unwrap();
        assert_eq!(sweep.buffers(), 2);
        assert_eq!(sweep.plan().windows().len(), 4);

        // Scoped rather than dropped: what ends the borrow is the scope, and
        // the borrow is what the sweep's "valid until you advance" rests on.
        {
            let first = sweep.next_expert().unwrap().unwrap();
            assert_eq!(first.expert, 0);
        }
        // Window 0 has been awaited and window 1 is tracked in the other
        // buffer, so the drive is busy while the caller computes.
        assert_eq!(sweep.stream.stats().sweep_windows_read, 1);
        assert_eq!(
            sweep.in_flight[1].map(|pending| pending.window),
            Some(1),
            "window 1 was not started before window 0 was handed over"
        );

        // Consuming window 1 puts window 2 in the buffer window 0 freed,
        // before it blocks on window 1.
        {
            let second = sweep.next_expert().unwrap().unwrap();
            assert_eq!(second.expert, 1);
        }
        assert_eq!(sweep.stream.stats().sweep_windows_read, 2);
        assert_eq!(
            sweep.in_flight[0].map(|pending| pending.window),
            Some(2),
            "the freed buffer was not refilled"
        );
        // With a ring, the prefetch is a real submission rather than a
        // deferred one; the `pread` fallback submits inside the await and has
        // no overlap to measure, which its own docs say.
        if sweep.stream.mode() != crate::io::StreamMode::Pread {
            assert_eq!(sweep.stream.stats().sweep_reads_submitted, 3);
        }
        sweep.finish().unwrap();
    }

    /// The dial can ask for more windows in flight than the layer has, or than
    /// the pool can hold; both are clamped rather than refused.
    #[test]
    fn buffers_are_clamped_to_the_layer_and_the_pool() {
        let fx = build_install("sweep-clamp");
        let mut stream = open(&fx, 4);
        let mut plan = SweepPlan::new();
        let routed = all_experts(&fx, 0);
        // One window covers the whole layer, so only one buffer is useful.
        let sweep = stream
            .sweep_layer(
                &mut plan,
                0,
                &routed,
                SweepConfig {
                    experts_per_window: 8,
                    windows_in_flight: 4,
                },
            )
            .unwrap();
        assert_eq!(sweep.buffers(), 1);
        sweep.finish().unwrap();
    }

    /// A window wider than the whole slot pool is a typed refusal, not a
    /// truncated read.
    #[test]
    fn a_window_bigger_than_the_pool_is_refused() {
        let fx = build_install("sweep-arena-small");
        // Two slots per layer is 2 blobs of headroom per layer; a 4-expert
        // window needs more than the whole pool.
        let mut stream = open(&fx, 2);
        let mut plan = SweepPlan::new();
        let routed = all_experts(&fx, 0);
        let err = stream
            .sweep_layer(
                &mut plan,
                0,
                &routed,
                SweepConfig {
                    experts_per_window: 8,
                    windows_in_flight: 1,
                },
            )
            .unwrap_err();
        assert!(
            matches!(err, SweepError::ArenaTooSmall { .. }),
            "unexpected error: {err}"
        );
        // The stream is untouched: the cache still works.
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
    }

    /// Taking the arena empties the cache, so an expert made resident before a
    /// sweep is a miss after it — its bytes are gone, whatever the slot says.
    #[test]
    fn a_sweep_invalidates_the_decode_cache() {
        let fx = build_install("sweep-invalidates");
        let mut stream = open(&fx, 4);
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
        // Resident: the same request hits.
        stream.begin_layer(0, &[0, 1]).unwrap();
        assert_eq!(stream.hits().len(), 2);
        stream.end_layer(0);

        let routed = all_experts(&fx, 0);
        sweep_all(&mut stream, 0, &routed, SweepConfig::default());

        // And now it misses again, because the slots hold sweep bytes.
        stream.begin_layer(0, &[0, 1]).unwrap();
        assert!(stream.hits().is_empty(), "a slot survived the sweep");
        assert_eq!(stream.misses().len(), 2);
        stream.await_misses().unwrap();
        for &(index, slot) in stream.misses().to_vec().iter() {
            let expert = [0u32, 1][index];
            let mut buf = Vec::new();
            let reader =
                ExpertReader::new(&fx.root, &fx.manifest, &fx.layout, LoadOptions::default())
                    .unwrap();
            let expected = reader.read_expert(0, expert, &mut buf).unwrap();
            assert_eq!(stream.view(0, slot).unwrap().blob(), expected.blob());
        }
        stream.end_layer(0);
    }

    /// The counters are all in the sweep half, and the phase split still adds
    /// up.
    #[test]
    fn sweep_counters_split_by_phase() {
        use crate::io::StreamPhase;

        let fx = build_install("sweep-phases");
        let mut stream = open(&fx, 4);
        let stride = fx.layout.layers[0].stride;
        let routed = all_experts(&fx, 0);
        sweep_all(&mut stream, 0, &routed, SweepConfig::default());

        stream.set_phase(StreamPhase::Decode);
        stream.begin_layer(1, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(1);

        let prefill = stream.stats_in(StreamPhase::Prefill);
        let decode = stream.stats_in(StreamPhase::Decode);
        let total = stream.stats();

        // Prefill did sweep work and no cache work.
        assert_eq!(prefill.accesses(), 0);
        assert!(!prefill.is_idle(), "the sweep phase must not look idle");
        assert_eq!(
            prefill.sweep_bytes_read,
            u64::from(fx.layout.layers[0].n_experts) * stride
        );
        assert!(prefill.sweep_reads_submitted >= 1);
        assert_eq!(prefill.bytes_read, 0);

        // Decode did cache work and no sweep work.
        assert_eq!(decode.accesses(), 2);
        assert_eq!(decode.sweep_bytes_read, 0);
        assert_eq!(decode.sweep_windows(), 0);
        assert_eq!(decode.bytes_read, 2 * fx.layout.layers[1].stride);

        // And the documented invariant holds field by field.
        assert_eq!(prefill.plus(&decode), total);
    }

    /// Two sweeps in a row over the same layer read the same bytes: the arena
    /// is properly given back.
    #[test]
    fn the_arena_is_reusable_across_layers_and_sweeps() {
        let fx = build_install("sweep-reuse");
        let mut stream = open(&fx, 4);
        let routed = all_experts(&fx, 0);
        let first = sweep_all(&mut stream, 0, &routed, SweepConfig::default());
        let second = sweep_all(&mut stream, 0, &routed, SweepConfig::default());
        assert_eq!(first, second);
        // A different layer, with a different stride class, straight after.
        let other = sweep_all(&mut stream, 1, &all_experts(&fx, 1), SweepConfig::default());
        assert_ne!(other[0].1, first[0].1, "layer 1 returned layer 0's bytes");
    }

    /// Dropping a sweep without finishing it still drains and releases.
    #[test]
    fn dropping_a_sweep_releases_the_arena() {
        let fx = build_install("sweep-drop");
        let mut stream = open(&fx, 4);
        let mut plan = SweepPlan::new();
        let routed = all_experts(&fx, 0);
        {
            let mut sweep = stream
                .sweep_layer(&mut plan, 0, &routed, SweepConfig::default())
                .unwrap();
            // Consume one expert and walk away with windows still in flight.
            assert!(sweep.next_expert().unwrap().is_some());
        }
        // The cache is usable again, which it would not be if the arena were
        // still out or a read still outstanding.
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
    }

    /// A sweep refuses to start while the decode path has reads outstanding —
    /// those reads are writing into the very buffers the arena would cover.
    #[test]
    fn a_sweep_is_refused_while_reads_are_in_flight() {
        let fx = build_install("sweep-inflight");
        let mut stream = open(&fx, 4);
        let mut plan = SweepPlan::new();
        stream.begin_layer(0, &[0, 1]).unwrap();
        assert_eq!(stream.misses().len(), 2, "the fixture must miss cold");
        let routed = all_experts(&fx, 0);
        let err = stream
            .sweep_layer(&mut plan, 0, &routed, SweepConfig::default())
            .unwrap_err();
        assert!(
            matches!(
                err,
                SweepError::ReadsInFlight { .. } | SweepError::StepOpen { .. }
            ),
            "unexpected error: {err}"
        );
        stream.await_misses().unwrap();

        // With the reads awaited but the step still open, it is the step that
        // refuses.
        let err = stream
            .sweep_layer(&mut plan, 0, &routed, SweepConfig::default())
            .unwrap_err();
        assert!(
            matches!(err, SweepError::StepOpen { layer: 0 }),
            "unexpected error: {err}"
        );
        stream.end_layer(0);
        stream
            .sweep_layer(&mut plan, 0, &routed, SweepConfig::default())
            .expect("a closed step lets the sweep start")
            .finish()
            .unwrap();
    }

    // ---- failure latching ------------------------------------------------

    /// A window read that fails terminally is final: a retried `next_expert`
    /// must never hand back arena bytes no read ever filled.
    #[test]
    fn a_failed_window_read_latches_instead_of_handing_out_unfilled_bytes() {
        // The failure this guards: `LayerSweep` latched nothing, so after a
        // terminal read failure `current` was still `None` and the window was
        // still in `in_flight`. The next call walked to the same window, asked
        // `sweep_await` for a read that had already resolved, got `Ok` because
        // nothing was outstanding any more, and handed out a `SweepExpert`
        // over bytes holding a *previous* window — or the zeros the slab was
        // born with. Wrong weights, wrong logits, no error, no log line.
        let fx = build_install("sweep-latch-read");
        let mut stream = open(&fx, 4);
        // Opened and verified before the truncation, so it is the read itself
        // that has to notice.
        stream.begin_layer(0, &[0]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);

        let path = fx.root.join(&fx.layout.layers[0].file);
        let whole = std::fs::read(&path).unwrap();
        std::fs::write(&path, &whole[..whole.len() / 2]).unwrap();

        let routed = all_experts(&fx, 0);
        // One window in flight, so which read the failure surfaces on is the
        // geometry rather than the drive's completion order.
        let config = SweepConfig {
            experts_per_window: 1,
            windows_in_flight: 1,
        };
        {
            let mut plan = SweepPlan::new();
            let mut sweep = stream.sweep_layer(&mut plan, 0, &routed, config).unwrap();
            // The half of the file that is still there reads fine.
            for expert in 0..2u32 {
                assert_eq!(sweep.next_expert().unwrap().unwrap().expert, expert);
            }
            // Expert 2's window is past the truncation: EOF, no retry.
            let first = sweep.next_expert().unwrap_err();
            assert!(
                matches!(first, SweepError::Io(_)),
                "unexpected error: {first}"
            );
            assert!(sweep.is_aborted());
            // And every later call reports it rather than a blob.
            for _ in 0..3 {
                match sweep.next_expert().unwrap_err() {
                    SweepError::Aborted { reason } => assert_eq!(reason, first.to_string()),
                    other => panic!("unexpected error: {other}"),
                }
            }
            // The drain is clean — the failed read reached a terminal state —
            // so what `finish` has left to report is that the sweep never
            // covered the layer, as the error that stopped it rather than as a
            // rendering of one.
            match sweep.finish().unwrap_err() {
                SweepError::Io(IoError::Io { source, .. }) => {
                    assert_eq!(source.kind(), std::io::ErrorKind::UnexpectedEof);
                }
                other => panic!("unexpected error: {other}"),
            }
        }

        // The arena went back, so the stream is a cache again.
        std::fs::write(&path, &whole).unwrap();
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
    }

    /// A sweep whose arena is taken away mid-flight latches too — and there
    /// the retry would *skip* an expert rather than repeat one.
    #[test]
    fn a_lost_arena_latches_instead_of_skipping_an_expert() {
        let fx = build_install("sweep-latch-arena");
        let mut stream = open(&fx, 4);
        let routed = all_experts(&fx, 0);
        let mut plan = SweepPlan::new();
        // Every window submitted up front, so the walk reaches the byte view
        // rather than stopping at a submission.
        let config = SweepConfig {
            experts_per_window: 1,
            windows_in_flight: 4,
        };
        let mut sweep = stream.sweep_layer(&mut plan, 0, &routed, config).unwrap();
        assert_eq!(sweep.next_expert().unwrap().unwrap().expert, 0);

        // Exactly what `strand_arena` leaves behind: no arena, the windows
        // abandoned, every slot it covered retired.
        sweep.stream.strand_open_arena();
        let err = sweep.next_expert().unwrap_err();
        assert!(
            matches!(err, SweepError::NoArena),
            "unexpected error: {err}"
        );

        // The cursor has already stepped over expert 1. Unlatched, this call
        // returned expert *2* and the layer quietly lost one.
        match sweep.next_expert().unwrap_err() {
            SweepError::Aborted { reason } => assert_eq!(reason, SweepError::NoArena.to_string()),
            other => panic!("unexpected error: {other}"),
        }
        // A clean drain still reports that the sweep did not finish the layer,
        // as the error that stopped it rather than as a rendering of one.
        let err = sweep.finish().unwrap_err();
        assert!(
            matches!(err, SweepError::NoArena),
            "unexpected error: {err}"
        );
    }

    /// The window counters describe what the sweep did, not what it planned.
    #[test]
    fn window_counters_follow_the_sweep_rather_than_the_plan() {
        // Skipped windows used to be counted all at once when the sweep
        // started and read windows one at a time as they were consumed, so an
        // aborted sweep reported every skip against a fraction of the reads —
        // under-reporting coverage on exactly the runs worth investigating.
        let fx = build_install("sweep-counters-abort");
        let mut stream = open(&fx, 4);
        // Four windows of one expert; route the first and the last, so the two
        // skippable ones sit between them.
        let config = SweepConfig {
            experts_per_window: 1,
            windows_in_flight: 1,
        };

        let before = stream.stats();
        {
            let mut plan = SweepPlan::new();
            let mut sweep = stream.sweep_layer(&mut plan, 0, &[0, 3], config).unwrap();
            assert_eq!(sweep.next_expert().unwrap().unwrap().expert, 0);
            // Walk away with three windows unvisited.
        }
        let delta = stream.stats().since(&before);
        assert_eq!(delta.sweep_windows_read, 1);
        assert_eq!(
            delta.sweep_windows_skipped, 0,
            "windows the sweep never reached were counted as skipped"
        );

        // Run to the end and every window is accounted for exactly once.
        let before = stream.stats();
        assert_eq!(sweep_all(&mut stream, 0, &[0, 3], config).len(), 2);
        let delta = stream.stats().since(&before);
        assert_eq!(delta.sweep_windows_read, 2);
        assert_eq!(delta.sweep_windows_skipped, 2);
        assert_eq!(delta.sweep_windows(), 4);
    }

    /// `window_bytes` is `pub` over `pub` fields, so an absurd stride is a
    /// typed refusal rather than an overflow panic in a debug build.
    #[test]
    fn an_overflowing_window_is_refused_rather_than_panicking() {
        let config = SweepConfig {
            experts_per_window: 1 << 20,
            windows_in_flight: 1,
        };
        match config.window_bytes(u64::MAX).unwrap_err() {
            SweepError::WindowTooLarge {
                experts,
                stride,
                bytes,
                max,
            } => {
                assert_eq!(experts, 1 << 20);
                assert_eq!(stride, u64::MAX);
                assert_eq!(bytes, u128::from(u64::MAX) << 20);
                assert_eq!(max, u64::from(u32::MAX));
            }
            other => panic!("unexpected error: {other}"),
        }
        // The ordinary case still answers a number.
        assert_eq!(
            SweepConfig::default().window_bytes(3_059_712).unwrap(),
            8 * 3_059_712
        );
        // And the plan refuses rather than propagating a half-built geometry.
        let mut plan = SweepPlan::new();
        assert!(matches!(
            plan.build(4, u64::MAX, config, &[0]).unwrap_err(),
            SweepError::WindowTooLarge { .. }
        ));
    }

    // ---- the prefill session ---------------------------------------------

    /// The split borrow: a staging span and a live sweep at the same time,
    /// both out of the slot pool, neither aliasing the other.
    #[test]
    fn a_prefill_session_hands_out_scratch_and_a_sweep_at_once() {
        let fx = build_install("prefill-session");
        let mut stream = open(&fx, 4);
        let reader = ExpertReader::new(&fx.root, &fx.manifest, &fx.layout, LoadOptions::default())
            .expect("reader opens");
        let pool_bytes = stream.cache_bytes();
        const SCRATCH: usize = 8192;
        let config = SweepConfig {
            experts_per_window: 2,
            windows_in_flight: 2,
        };

        let mut plan = SweepPlan::new();
        let mut session = stream
            .begin_prefill(SCRATCH, config)
            .expect("session opens");
        assert_eq!(session.scratch_len(), SCRATCH);
        assert!(
            session.ring_offset() >= SCRATCH,
            "the sweep ring overlaps the scratch span"
        );
        assert!(direct::is_aligned(session.ring_offset() as u64));

        // Stamp the staging span and let a whole prefill's window reads run
        // over it: this is the test that says the two spans are disjoint.
        session.scratch().fill(0xa5);
        for layer in 0..N_LAYERS {
            let routed = all_experts(&fx, layer);
            let (staging, mut sweep) = session
                .split(&mut plan, layer, &routed)
                .expect("the layer sweeps");
            let mut seen = 0u32;
            let mut buf = Vec::new();
            while let Some(expert) = sweep.next_expert().expect("window reads") {
                // Both borrows live at once, which is the whole point.
                let expected = reader.read_expert(layer, expert.expert, &mut buf).unwrap();
                assert_eq!(expert.bytes, expected.blob(), "layer {layer}");
                staging[expert.expert as usize] = expert.expert as u8;
                seen += 1;
            }
            sweep.finish().expect("the sweep finishes");
            assert_eq!(seen, fx.layout.layers[layer as usize].n_experts);
        }

        let scratch = session.scratch();
        for expert in 0..N_EXPERTS {
            assert_eq!(scratch[expert as usize], expert as u8);
        }
        assert!(
            scratch[N_EXPERTS as usize..].iter().all(|&b| b == 0xa5),
            "a window read landed in the staging span"
        );
        session.finish().expect("the session finishes");

        // The pool is a cache again, exactly as big as it was.
        assert_eq!(stream.cache_bytes(), pool_bytes);
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
    }

    /// A carve past the slab is refused, and an unaligned scratch request
    /// still starts the ring on a page boundary.
    #[test]
    fn a_prefill_session_carve_is_bounded_and_page_aligned() {
        let fx = build_install("prefill-session-size");
        let mut stream = open(&fx, 4);
        let pool_bytes = stream.cache_bytes() as usize;
        let config = SweepConfig {
            experts_per_window: 1,
            windows_in_flight: 1,
        };

        // Scratch alone fills the slab, so there is no room for a ring.
        let err = stream.begin_prefill(pool_bytes, config).unwrap_err();
        assert!(
            matches!(err, SweepError::ArenaTooSmall { .. }),
            "unexpected error: {err}"
        );
        // Nothing was taken: the cache still works.
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);

        let widest = fx
            .layout
            .layers
            .iter()
            .map(|layer| layer.stride)
            .max()
            .unwrap();
        let session = stream.begin_prefill(4097, config).expect("session opens");
        assert_eq!(session.scratch_len(), 4097);
        assert_eq!(session.ring_offset(), 8192, "the ring must start on a page");
        assert_eq!(session.config(), config);
        session.finish().unwrap();

        // The ring is sized for the widest layer, not the first one.
        let session = stream.begin_prefill(0, config).expect("session opens");
        assert_eq!(session.ring_offset(), 0);
        drop(session);
        let too_big = pool_bytes - widest as usize + 4096;
        let err = stream.begin_prefill(too_big, config).unwrap_err();
        assert!(
            matches!(err, SweepError::ArenaTooSmall { .. }),
            "unexpected error: {err}"
        );
    }

    /// A sweep that was leaked rather than dropped never drained its windows,
    /// and the next sweep of the session carves the same ring from buffer 0
    /// up.
    #[test]
    fn a_leaked_sweep_cannot_start_a_second_one_over_its_ring() {
        // The failure this guards: `begin_within` deliberately does not take
        // the arena, so it inherited none of `take_arena`'s checks — nothing on
        // that path looked at `outstanding` at all. `mem::forget(sweep)` is
        // safe Rust, skips the teardown that drains the windows, and ends the
        // borrow that made the session unusable; the next `split` then computed
        // `arena_offset = ring_base + 0`, the exact address layer 0's still-live
        // window read was writing into. Two O_DIRECT reads on one destination,
        // which EXP-007 measured as 13-27% spurious btrfs EIO.
        let fx = build_install("prefill-leaked-sweep");
        let mut stream = open(&fx, 4);
        let config = SweepConfig {
            experts_per_window: 1,
            windows_in_flight: 2,
        };
        let mut plan = SweepPlan::new();
        let routed0 = all_experts(&fx, 0);
        let routed1 = all_experts(&fx, 1);
        let mut session = stream.begin_prefill(0, config).expect("session opens");

        let (_staging, sweep) = session
            .split(&mut plan, 0, &routed0)
            .expect("layer 0 sweeps");
        std::mem::forget(sweep);

        // Both backends leave a submitted window outstanding until someone
        // reaps it, and the leak is precisely nobody reaping.
        let outstanding = session.stream.reads_outstanding();
        assert!(outstanding > 0, "the leak left no windows on the wire");
        match session.split(&mut plan, 1, &routed1) {
            Err(SweepError::ReadsInFlight { outstanding: seen }) => assert_eq!(seen, outstanding),
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => panic!("a leaked sweep's live windows started a second sweep"),
        }

        // The session is still the arena's owner and still drains the leak.
        session.finish().expect("the session drains what was left");
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
    }

    /// A session that was leaked rather than dropped never gave the arena back,
    /// so a second carve would be a second `&mut [u8]` over the same scratch.
    #[test]
    fn a_leaked_prefill_session_cannot_take_a_second_arena() {
        let fx = build_install("prefill-leaked-session");
        let mut stream = open(&fx, 4);
        let config = SweepConfig {
            experts_per_window: 1,
            windows_in_flight: 1,
        };
        let session = stream.begin_prefill(4096, config).expect("session opens");
        let scratch = session.scratch_len();
        std::mem::forget(session);

        match stream.begin_prefill(scratch, config).unwrap_err() {
            SweepError::ArenaOut { bytes } => assert!(bytes >= scratch),
            other => panic!("unexpected error: {other}"),
        }
        // And a plain sweep, which takes the arena through the same door.
        let mut plan = SweepPlan::new();
        let routed = all_experts(&fx, 0);
        match stream
            .sweep_layer(&mut plan, 0, &routed, config)
            .unwrap_err()
        {
            SweepError::ArenaOut { .. } => {}
            other => panic!("unexpected error: {other}"),
        }

        // Giving the leaked carve back by hand puts the pool back to a cache,
        // which is the only thing left that can.
        stream.release_arena();
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
    }

    /// The latch keeps the failure rather than a rendering of it, so a driver
    /// that only reads `finish` can still tell `ENOSPC` from `EIO`.
    #[test]
    fn a_latched_failure_keeps_its_errno() {
        let path = std::path::Path::new("layer_00.bin");
        let enospc = SweepError::Io(IoError::io(
            path,
            std::io::Error::from_raw_os_error(libc::ENOSPC),
        ));
        match latchable(&enospc) {
            SweepError::Io(IoError::Io { source, .. }) => {
                assert_eq!(source.raw_os_error(), Some(libc::ENOSPC));
            }
            other => panic!("unexpected error: {other}"),
        }

        // Through a stranding, which is the shape that also names dead layers.
        let stranded = SweepError::CacheStranded {
            layer: 3,
            slots: 0,
            top_k: 8,
            source: IoError::io(path, std::io::Error::from_raw_os_error(libc::EIO)),
        };
        match latchable(&stranded) {
            SweepError::CacheStranded {
                layer,
                slots,
                top_k,
                source: IoError::Io { source, .. },
            } => {
                assert_eq!((layer, slots, top_k), (3, 0, 8));
                assert_eq!(source.raw_os_error(), Some(libc::EIO));
            }
            other => panic!("unexpected error: {other}"),
        }

        // A synthesised error has no errno, so its kind is what survives.
        let eof = SweepError::Io(IoError::io(
            path,
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read"),
        ));
        match latchable(&eof) {
            SweepError::Io(IoError::Io { source, .. }) => {
                assert_eq!(source.kind(), std::io::ErrorKind::UnexpectedEof);
                assert_eq!(source.to_string(), "short read");
            }
            other => panic!("unexpected error: {other}"),
        }

        // A variant with nothing but data in it renders, which is all it had.
        let plain = SweepError::NoArena;
        match latchable(&plain) {
            SweepError::Aborted { reason } => assert_eq!(reason, plain.to_string()),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn a_layer_past_the_layout_is_a_typed_error() {
        let fx = build_install("sweep-layer-range");
        let mut stream = open(&fx, 4);
        let mut plan = SweepPlan::new();
        let err = stream
            .sweep_layer(&mut plan, 9, &[0], SweepConfig::default())
            .unwrap_err();
        assert!(
            matches!(
                err,
                SweepError::Io(IoError::LayerOutOfRange {
                    layer: 9,
                    n_layers: 2
                })
            ),
            "unexpected error: {err}"
        );
    }
}
