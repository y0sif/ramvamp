//! The expert read path, from routing decision to resident blob.
//!
//! [`ExpertStream`] is what the decode loop drives. One call per layer per
//! token:
//!
//! ```text
//! begin_layer(layer, top_k)   plan the step, submit every miss, return
//!   -> hits()                 compute these now; the reads are in flight
//!   -> misses()
//! await_misses()              block until every submitted read has landed
//!   -> view(layer, slot)      byte view over a filled slot's slabs
//! end_layer(layer)            hand every slot this step protected back
//! ```
//!
//! That shape is the measured one, not a convenience: three independent
//! implementations converged on waiting for *all* misses before dispatching
//! them as one unit, and TurboFieldfare measured per-expert progressive
//! execution as both slower and divergent (see `docs/architecture.md`,
//! "Decode loop").
//!
//! # What it owns
//!
//! - A [`SlotPool`], leased **once** at construction: every slot's guard is
//!   held for the life of the stream, so a slot's address never moves and the
//!   pool can never hand the same buffer to two readers. Which slot may be
//!   written when is decided by [`LayerCache`], the component that knows
//!   about in-flight reads and queued compute.
//! - One [`LayerCache`] per layer (ghost-history LFU), sized from the byte
//!   budget.
//! - The ring, or the synchronous fallback.
//!
//! # Placement
//!
//! The ring runs **inline on the calling thread**: `begin_layer` submits,
//! `await_misses` reaps. There is no reactor thread in v0 — a dedicated
//! E-core reactor is a recorded experiment, not the shipped design, because
//! the evidence for one is mixed and io_uring submission is cheap enough that
//! there is little to move off-thread. Nothing here assumes the placement,
//! though: submission and reaping are separate methods over an explicit
//! in-flight table keyed by `user_data`, which is what a reactor would need.
//!
//! # What can go wrong, and what happens then
//!
//! - **Completions arrive out of order.** Dispatch is on `user_data`, which
//!   packs a per-read tag with the in-flight index; a completion whose tag no
//!   longer matches is stale and is dropped rather than credited to whatever
//!   read now occupies that slot.
//! - **`EIO` is a real, retryable outcome**, not an impossible one. Direct
//!   I/O returns real errors and real short reads. A failed blob read is
//!   retried exactly once and then reported as a typed error; `result == 0`
//!   is EOF, never success.
//! - **A short read is continued, or restarted.** A short read that stopped on
//!   a block boundary is reissued from where it stopped. One that did not
//!   cannot be: O_DIRECT rejects an unaligned offset with `EINVAL`, which is
//!   not a retryable errno, so the blob is reissued from its base instead.
//! - **The SQ fills.** `PushError` means submit-then-retry, never drop. If the
//!   submit answers `EBUSY` the completion queue is drained first, because
//!   `EBUSY` means "the CQ needs reaping", not "this read is lost".
//! - **`submit_and_wait` returns `EINTR`.** Retried, up to a bound, so a
//!   signal storm cannot park the decode thread forever.
//! - **A failed fill leaves a slot un-owned.** The step invalidates it (the
//!   completion has been reaped, which is that call's safety requirement) and
//!   releases the rest, so a failed layer does not strand slots.
//! - **A read that can never be reaped** (the ring itself failed) has its slot
//!   *retired*: the buffer is leaked so a late kernel write is harmless, and
//!   the layer is rebuilt one slot smaller so nothing can ever be handed that
//!   slot again. The layer keeps serving steps on what is left, and once too
//!   little is left it says so ([`super::CacheError::TooFewSlots`]) rather
//!   than failing every read forever.
//!
//! # The prefill arena
//!
//! Prefill does not use this cache at all. [`super::sweep`] streams a layer's
//! expert file front to back in large windows and computes each expert against
//! every routed row as it arrives, which needs one big scratch buffer rather
//! than a slot per expert — and the pool is *idle* at that moment, because the
//! decode cache is cold by design (replaying the prompt into it measured +0.09
//! points, EXP-005).
//!
//! So the pool doubles as that scratch: [`ExpertStream::take_arena`] carves the
//! head of the slab into a [`Arena`] and hands it to the sweep.
//!
//! **The soundness argument is a borrow, not a comment.** [`SlotTable`]
//! believes it owns every slot, so an arena that coexisted with live slot views
//! would alias `&mut`. It cannot:
//!
//! - `take_arena` takes `&mut self`, and every caller of it
//!   ([`super::sweep::LayerSweep`]) parks that `&mut ExpertStream` for the
//!   whole life of the arena. No cache operation — `begin_layer`, `view`,
//!   `end_layer`, or anything else that reaches a [`SlotGuard`] — is callable
//!   while the arena exists, because they all need the stream and the stream is
//!   exclusively borrowed.
//! - The pointer is derived from [`SlotPool::slab_base`], the same raw
//!   allocation base every guard's address is derived from, so no `&mut` slice
//!   spanning the slab ever has to be conjured out of a guard.
//! - `take_arena` refuses while anything is in flight, **and** refuses to
//!   cover any buffer a lost read was allowed to keep writing into. The second
//!   half is not implied by the first: `Inflight::abandon` drains `outstanding`
//!   for reads that can never be reaped, so a zero counter says the table is
//!   empty, not that the drive is idle. What those reads leave is a *retired*
//!   slot — a buffer leaked in place, still inside the slab — and the carve is
//!   checked against exactly those.
//! - Taking the arena **invalidates every layer's slot occupancy**. Sweep bytes
//!   land in the pool, so any cache entry that survived would name a buffer
//!   holding somebody else's expert. Only occupancy is dropped; the ghost-LFU
//!   frequency counters are indexed by expert id and survive on purpose.
//! - Every layer's pitch must equal its stride, or the arena is refused: a
//!   padded layer makes the slab something other than a gapless run of
//!   blob-sized buffers.
//!
//! A sweep read that can never be reaped is the one case that outlives the
//! borrow. Its bytes may land anywhere in the arena, so every slot the arena
//! overlaps is retired and the stream refuses to hand out an arena ever again
//! ([`ExpertStream::strand_arena`]).

use std::fmt;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

#[cfg(feature = "io-uring")]
use std::os::fd::AsRawFd;

#[cfg(feature = "io-uring")]
use io_uring::{IoUring, opcode, types};

use crate::format::{ExpertsLayout, FormatError, Manifest, sha256_file};

use super::direct::{self, DirectFault, DirectSupport};
use super::sweep::{SweepError, check_aligned};
use super::{
    CachePlan, CacheStats, ExpertReader, ExpertView, IoError, LayerCache, LoadOptions, MAX_SLOTS,
    SLOT_ALIGN, SlotError, SlotGuard, SlotPool,
};

/// Submission queue depth.
///
/// What predicts throughput on the reference drive is **total bytes in
/// flight**, not queue depth on its own. Block size and queue depth move that
/// same quantity and are interchangeable at matched bytes, the drive holds its
/// peak up to roughly 100 MB outstanding, and it gives back 15 to 18 percent
/// past about 170 MB (EXP-019, cold and in-cgroup on the installed layer
/// files). Decode reads **one expert blob per miss**, so 8 outstanding is at
/// most 24.5 MB in flight, which is inside that plateau. The slow 1.92 to 1.98
/// GB/s cells in EXP-019's matrix are the large-block *and* deep-queue corner,
/// which this path never issues. Hence 8.
///
/// This supersedes the previous justification, which read "the drive saturates
/// early, 1.211 GB/s at QD4 against 1.390 at QD16" from EXP-008. EXP-019
/// retires that on level (it measures 1.54 to 2.37 GB/s on a quiet machine)
/// and on mechanism (queue depth was never the variable). The conclusion is
/// unchanged; the reason for it is not. EXP-008's per-blob latency series
/// (p50 2.34 ms at QD1 against 15.56 ms at QD8) was never re-taken, so the
/// "decode waits on all misses, prefer the low end for latency" argument is
/// context here rather than a second measurement.
///
/// **Two gaps this does not close**, both open work in `docs/architecture.md`.
/// EXP-019 swept queue depth only at 8 experts per read, so the decode
/// geometry has no measured queue-depth curve of its own: its QD8 point is
/// measured, its QD2 and QD4 points are not. And the probe emulated depth with
/// threaded `preadv`, not io_uring, so it characterises the drive and the
/// filesystem rather than this submission path. An io_uring confirmation
/// inside the runtime is still owed, and this constant should not move before
/// that lands.
///
/// The ring is also charged against `RLIMIT_MEMLOCK`, which is 8 MiB soft
/// *and* hard under systemd defaults since kernel 6.14.
#[cfg(feature = "io-uring")]
const RING_ENTRIES: u32 = 8;

/// Passes of the completion loop with no progress before it is called a
/// livelock rather than a slow drive.
#[cfg(feature = "io-uring")]
const MAX_IDLE_PASSES: u32 = 1024;

/// Pushes of one SQE, each separated by a submit and a completion-queue
/// drain, before a full submission queue is called wedged.
///
/// Three is the worst case the shape of the retry implies — push fails, the
/// submit answers `EBUSY` so the CQ is drained, push fails again because the
/// SQ still holds the entry the kernel refused, the submit now succeeds and
/// frees it — and this is comfortably above it.
#[cfg(feature = "io-uring")]
const MAX_PUSH_ATTEMPTS: u32 = 8;

/// `io_uring_enter` calls interrupted by a signal before it is called a
/// signal storm rather than a signal.
///
/// `EINTR` must be retried — propagating it would fail a step for a stray
/// `SIGWINCH` — but an unbounded retry parks the decode thread with no way
/// out, so the loop is bounded and reports a typed error at the end of it.
#[cfg(feature = "io-uring")]
const MAX_EINTR_RETRIES: u32 = 1024;

/// How expert reads are actually being served.
///
/// This is what the runtime *got*, not what it asked for. Every downgrade is
/// logged when the stream opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// io_uring, with page-cache bypass empirically verified. The only mode
    /// the published memory numbers apply to.
    ODirect,
    /// io_uring, but reads are (or may be) charged to the page cache: the
    /// filesystem downgraded O_DIRECT, or it could not be shown that it did
    /// not. Correct, but outside the memory contract.
    Buffered,
    /// Synchronous positioned reads: the `io-uring` feature is off, or the
    /// ring could not be created on this kernel. Says nothing about direct
    /// I/O, which is reported separately by
    /// [`ExpertStream::direct_io`].
    Pread,
}

impl fmt::Display for StreamMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ODirect => "io_uring+O_DIRECT",
            Self::Buffered => "io_uring+buffered",
            Self::Pread => "pread",
        })
    }
}

/// Which half of a generation the stream is serving.
///
/// Prefill and decode have different miss profiles — prefill touches every
/// layer with a cold cache, decode reuses what prefill left resident — so a
/// hit rate summed over both understates the steady state. EXP-013 quoted one
/// such figure: 33,792 requests is 88 forward passes, about 24 of them
/// prefill, and prefill carries a much higher cold-miss share.
///
/// The stream cannot infer the phase (it sees one layer at a time, with no
/// notion of a token boundary), so the caller declares it with
/// [`ExpertStream::set_phase`]. A caller that never does attributes everything
/// to [`StreamPhase::Prefill`], which is where a stream starts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum StreamPhase {
    /// The prompt pass: every layer, cold cache, all positions at once.
    #[default]
    Prefill,
    /// Token-by-token generation.
    Decode,
}

impl StreamPhase {
    /// Every phase, in the order a generation goes through them.
    pub const ALL: [Self; 2] = [Self::Prefill, Self::Decode];

    /// Index into a per-phase array.
    fn index(self) -> usize {
        match self {
            Self::Prefill => 0,
            Self::Decode => 1,
        }
    }
}

impl fmt::Display for StreamPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        })
    }
}

/// Cumulative streaming telemetry, summed across every layer.
///
/// The cache half comes from the per-layer [`LayerCache`]s; the I/O half is
/// counted here. `hits + pending_hits + misses` is the number of routed
/// experts resolved.
///
/// [`ExpertStream::stats`] returns this summed over the whole run;
/// [`ExpertStream::stats_in`] returns one phase's share of it, which is what a
/// steady-state decode hit rate has to be quoted from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamStats {
    /// Routed experts served from a slot whose bytes were already valid.
    pub hits: u64,
    /// Routed experts that landed on a slot still filling from an earlier
    /// step's read. No second read is issued, but no read is avoided either.
    pub pending_hits: u64,
    /// Routed experts that had to be read.
    pub misses: u64,
    /// Misses for an expert the layer had never successfully fetched.
    pub cold_misses: u64,
    /// Misses for an expert the layer had fetched and evicted. The number
    /// that says whether the slot budget is wrong.
    pub eviction_misses: u64,
    /// Bytes actually transferred into slots.
    pub bytes_read: u64,
    /// Reads handed to the kernel, including retries and short-read
    /// continuations. Exceeds `misses` by exactly those.
    pub reads_submitted: u64,
    /// Reads that came back failed or short and were reissued.
    pub read_retries: u64,
    /// Wall time blocked in [`ExpertStream::await_misses`].
    pub io_wait: Duration,

    /// Bytes transferred into the prefill arena by [`super::sweep`].
    ///
    /// Counted apart from [`StreamStats::bytes_read`] rather than added to it,
    /// because the sweep bypasses the cache entirely: mixing the two would
    /// make "bytes read" mean "bytes read into slots" for one phase and
    /// something else for the other.
    pub sweep_bytes_read: u64,
    /// Window reads the sweep handed to the kernel, retries and short-read
    /// continuations included. A 23 MiB O_DIRECT read coming back in pieces is
    /// ordinary, so this comfortably exceeds `sweep_windows_read`.
    pub sweep_reads_submitted: u64,
    /// Sweep reads that came back failed or short and were reissued.
    pub sweep_read_retries: u64,
    /// Wall time the sweep spent blocked on a window read.
    pub sweep_io_wait: Duration,
    /// Windows the sweep actually read.
    pub sweep_windows_read: u64,
    /// Windows the sweep skipped because the chunk routed none of their
    /// experts. Zero at 512-token chunks (coverage is ~100%); the dial that
    /// matters at 128.
    pub sweep_windows_skipped: u64,
}

impl StreamStats {
    /// Routed experts resolved: hits, pending hits, and misses.
    ///
    /// **Cache traffic only.** A phase served entirely by [`super::sweep`]
    /// reports zero here and is still a phase that did work; ask
    /// [`StreamStats::sweep_windows`] before concluding a phase was idle.
    pub fn accesses(&self) -> u64 {
        self.hits + self.pending_hits + self.misses
    }

    /// Windows the sweep looked at, read and skipped together.
    ///
    /// The "did this phase do anything?" question for a cache-bypassing
    /// prefill, which [`StreamStats::accesses`] cannot answer.
    pub fn sweep_windows(&self) -> u64 {
        self.sweep_windows_read + self.sweep_windows_skipped
    }

    /// Whether this phase did any work at all, through either path.
    pub fn is_idle(&self) -> bool {
        self.accesses() == 0 && self.sweep_windows() == 0
    }

    /// Fraction of routed experts served without waiting on a read; `0.0`
    /// before any request.
    ///
    /// Pending hits are in the denominator and not the numerator, matching
    /// [`CacheStats::hit_rate`](crate::io::CacheStats::hit_rate).
    pub fn hit_rate(&self) -> f64 {
        let accesses = self.accesses();
        if accesses == 0 {
            0.0
        } else {
            self.hits as f64 / accesses as f64
        }
    }

    /// What happened between an earlier snapshot of the same stream and this
    /// one.
    ///
    /// Saturating field by field. Every counter here is monotonic, so a
    /// non-zero floor would only ever hide a bug in the caller's snapshot
    /// order; it is not worth an error path in a telemetry accessor.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            hits: self.hits.saturating_sub(earlier.hits),
            pending_hits: self.pending_hits.saturating_sub(earlier.pending_hits),
            misses: self.misses.saturating_sub(earlier.misses),
            cold_misses: self.cold_misses.saturating_sub(earlier.cold_misses),
            eviction_misses: self.eviction_misses.saturating_sub(earlier.eviction_misses),
            bytes_read: self.bytes_read.saturating_sub(earlier.bytes_read),
            reads_submitted: self.reads_submitted.saturating_sub(earlier.reads_submitted),
            read_retries: self.read_retries.saturating_sub(earlier.read_retries),
            io_wait: self.io_wait.saturating_sub(earlier.io_wait),
            sweep_bytes_read: self
                .sweep_bytes_read
                .saturating_sub(earlier.sweep_bytes_read),
            sweep_reads_submitted: self
                .sweep_reads_submitted
                .saturating_sub(earlier.sweep_reads_submitted),
            sweep_read_retries: self
                .sweep_read_retries
                .saturating_sub(earlier.sweep_read_retries),
            sweep_io_wait: self.sweep_io_wait.saturating_sub(earlier.sweep_io_wait),
            sweep_windows_read: self
                .sweep_windows_read
                .saturating_sub(earlier.sweep_windows_read),
            sweep_windows_skipped: self
                .sweep_windows_skipped
                .saturating_sub(earlier.sweep_windows_skipped),
        }
    }

    /// Two disjoint spans of the same stream, added together.
    pub(super) fn plus(&self, other: &Self) -> Self {
        Self {
            hits: self.hits + other.hits,
            pending_hits: self.pending_hits + other.pending_hits,
            misses: self.misses + other.misses,
            cold_misses: self.cold_misses + other.cold_misses,
            eviction_misses: self.eviction_misses + other.eviction_misses,
            bytes_read: self.bytes_read + other.bytes_read,
            reads_submitted: self.reads_submitted + other.reads_submitted,
            read_retries: self.read_retries + other.read_retries,
            io_wait: self.io_wait + other.io_wait,
            sweep_bytes_read: self.sweep_bytes_read + other.sweep_bytes_read,
            sweep_reads_submitted: self.sweep_reads_submitted + other.sweep_reads_submitted,
            sweep_read_retries: self.sweep_read_retries + other.sweep_read_retries,
            sweep_io_wait: self.sweep_io_wait + other.sweep_io_wait,
            sweep_windows_read: self.sweep_windows_read + other.sweep_windows_read,
            sweep_windows_skipped: self.sweep_windows_skipped + other.sweep_windows_skipped,
        }
    }
}

/// I/O counters kept by the stream itself.
///
/// Every read-path counter has a cache half and a sweep half, and which one a
/// read lands in is decided by `in_sweep` rather than per call site: the two
/// paths are mutually exclusive in time (an arena is refused while anything is
/// in flight, and the sweep holds the stream's `&mut` for its whole life), so
/// a mode flag cannot mis-attribute a read.
#[derive(Debug, Clone, Copy, Default)]
struct IoStats {
    bytes_read: u64,
    reads_submitted: u64,
    read_retries: u64,
    io_wait: Duration,
    stale_completions: u64,
    sweep_bytes_read: u64,
    sweep_reads_submitted: u64,
    sweep_read_retries: u64,
    sweep_io_wait: Duration,
    sweep_windows_read: u64,
    sweep_windows_skipped: u64,
    /// Whether a prefill arena is currently out, which is what decides the
    /// bucket every counter below goes into.
    in_sweep: bool,
}

impl IoStats {
    /// Credit transferred bytes to the phase that is running.
    fn add_bytes(&mut self, bytes: u64) {
        if self.in_sweep {
            self.sweep_bytes_read += bytes;
        } else {
            self.bytes_read += bytes;
        }
    }

    /// Credit one read handed to the kernel.
    fn add_submit(&mut self) {
        if self.in_sweep {
            self.sweep_reads_submitted += 1;
        } else {
            self.reads_submitted += 1;
        }
    }

    /// Credit one reissue: a short read continued, or a failure retried.
    fn add_retry(&mut self) {
        if self.in_sweep {
            self.sweep_read_retries += 1;
        } else {
            self.read_retries += 1;
        }
    }

    /// Credit time spent blocked on the drive.
    fn add_wait(&mut self, elapsed: Duration) {
        if self.in_sweep {
            self.sweep_io_wait += elapsed;
        } else {
            self.io_wait += elapsed;
        }
    }
}

/// Per-layer file state: geometry resolved at construction, handle opened and
/// verified on first use.
#[derive(Debug)]
struct LayerState {
    path: PathBuf,
    /// Install-relative name, for error messages.
    name: String,
    stride: u64,
    /// `stride` narrowed for the SQE's `len` field.
    len: u32,
    n_experts: u32,
    expected_size: u64,
    expected_sha256: String,
    file: Option<File>,
}

/// Where a read's bytes land.
///
/// The one thing [`Inflight`] would otherwise have to know about destinations.
/// Deliberately an offset rather than a pointer: it keeps [`Read`] `Copy` and
/// `Send`, so the whole table stays as trivially movable as it was when the
/// cache was the only consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dest {
    /// The cache slot named by [`Read::layer`] and [`Read::slot`].
    Slot,
    /// A byte offset into the prefill arena, for a [`super::sweep`] window.
    Arena(usize),
}

/// One outstanding blob read.
#[derive(Debug, Clone, Copy)]
struct Read {
    /// Token this read was submitted with. A completion carrying anything
    /// else names a read that no longer exists.
    user_data: u64,
    /// Where the bytes go.
    dst: Dest,
    layer: u32,
    slot: u32,
    expert: u32,
    /// File offset of the blob's first byte.
    base: u64,
    /// Bytes already in the slot.
    filled: u32,
    /// Bytes still owed.
    remaining: u32,
    /// Failed attempts already retried. The budget is one.
    attempts: u8,
    /// Times this blob was restarted from its base after a short read that
    /// stopped off a block boundary. The budget is one.
    restarts: u8,
    /// Whether this read has reached a terminal state.
    done: bool,
}

/// What a completion meant.
#[derive(Debug)]
enum Reap {
    /// Not a live read of ours: a duplicate, or a completion for a read that
    /// was already retried under a new tag. Ignored.
    Stale,
    /// The blob is complete.
    Done(usize),
    /// More bytes are owed (short read) or the failure is retryable. The
    /// caller must reissue.
    Again(usize),
    /// Terminal failure.
    Failed(usize, io::Error),
}

/// The in-flight table: every read submitted and not yet resolved.
///
/// Separated from [`ExpertStream`] so that completion bookkeeping — the part
/// with out-of-order arrival, stale tags and the retry budget in it — is
/// reachable without a device that can be made to fail.
#[derive(Debug, Default)]
struct Inflight {
    reads: Vec<Read>,
    /// Reads submitted and not yet in a terminal state.
    outstanding: usize,
    /// Rolls once per submitted read, so a reissue invalidates the token of
    /// the attempt it replaces.
    next_tag: u32,
    /// Alignment a reissue's file offset must satisfy, or `0` when reads are
    /// buffered and any offset is legal. [`direct::DIO_ALIGN`] once layer
    /// files are opened `O_DIRECT`: the kernel answers `EINVAL` — which
    /// [`is_retryable`] deliberately excludes — for an unaligned offset,
    /// length or destination, so a short read that stopped mid-block cannot be
    /// continued and the blob is restarted instead.
    align: u32,
}

impl Inflight {
    /// An empty table preallocated for `top_k` reads per step, which is the
    /// most one layer of one token can have outstanding.
    fn with_capacity(top_k: usize) -> Self {
        Self {
            reads: Vec::with_capacity(top_k),
            ..Self::default()
        }
    }

    /// Record a read into a cache slot and return its index.
    fn track(&mut self, layer: u32, slot: u32, expert: u32, base: u64, len: u32) -> usize {
        self.push(Dest::Slot, layer, slot, expert, base, len)
    }

    /// Record a read into the prefill arena at `offset` and return its index.
    ///
    /// `expert` is the window's first expert and `slot` its buffer index;
    /// neither means anything to this table, they are payload for the log
    /// lines the retry paths emit.
    fn track_arena(
        &mut self,
        layer: u32,
        buffer: u32,
        first_expert: u32,
        base: u64,
        len: u32,
        offset: usize,
    ) -> usize {
        self.push(Dest::Arena(offset), layer, buffer, first_expert, base, len)
    }

    /// Record a read about to be submitted and return its index.
    fn push(
        &mut self,
        dst: Dest,
        layer: u32,
        slot: u32,
        expert: u32,
        base: u64,
        len: u32,
    ) -> usize {
        let index = self.reads.len();
        let user_data = self.token(index);
        self.reads.push(Read {
            user_data,
            dst,
            layer,
            slot,
            expert,
            base,
            filled: 0,
            remaining: len,
            attempts: 0,
            restarts: 0,
            done: false,
        });
        self.outstanding += 1;
        index
    }

    /// Give read `index` a fresh token, so the completion of the attempt it
    /// replaces is recognisable as stale.
    fn retag(&mut self, index: usize) -> u64 {
        let user_data = self.token(index);
        self.reads[index].user_data = user_data;
        user_data
    }

    /// `tag:index`, the value carried in `user_data`.
    fn token(&mut self, index: usize) -> u64 {
        self.next_tag = self.next_tag.wrapping_add(1);
        (u64::from(self.next_tag) << 32) | (index as u64 & 0xffff_ffff)
    }

    /// Fold one completion into the table.
    ///
    /// `result` is the kernel's convention: bytes transferred, `0` for EOF,
    /// or a negative errno. Nothing here panics on an unexpected value —
    /// completions are the one input this layer cannot validate in advance.
    fn resolve(&mut self, user_data: u64, result: i32, io: &mut IoStats) -> Reap {
        let index = (user_data & 0xffff_ffff) as usize;
        let align = self.align;
        let Some(read) = self.reads.get_mut(index) else {
            return Reap::Stale;
        };
        if read.user_data != user_data || read.done {
            return Reap::Stale;
        }
        if result > 0 {
            // A short read is a real O_DIRECT outcome: continue from where it
            // stopped rather than trusting the count.
            let got = (result as u32).min(read.remaining);
            read.filled += got;
            read.remaining -= got;
            io.add_bytes(u64::from(got));
            if read.remaining == 0 {
                read.done = true;
                self.outstanding -= 1;
                return Reap::Done(index);
            }
            io.add_retry();
            if align != 0 && read.filled % align != 0 {
                // Off a block boundary, so `base + filled` is an offset
                // O_DIRECT will refuse. Start the blob again rather than
                // submit a read that is guaranteed `EINVAL`.
                if read.restarts > 0 {
                    read.done = true;
                    self.outstanding -= 1;
                    let (base, filled) = (read.base, read.filled);
                    return Reap::Failed(
                        index,
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "expert blob at {base}: short reads keep stopping off a \
                                 {align}-byte boundary (last at +{filled}), which direct \
                                 I/O cannot resume from"
                            ),
                        ),
                    );
                }
                read.restarts += 1;
                tracing::warn!(
                    layer = read.layer,
                    expert = read.expert,
                    slot = read.slot,
                    filled = read.filled,
                    "short read stopped off a block boundary; restarting the blob"
                );
                read.remaining += read.filled;
                read.filled = 0;
                // A restart is a fresh attempt at the whole blob, so it gets a
                // fresh retry budget. Without this, a blob that spent its one
                // retry on an `EIO`, made progress, and then restarted would
                // have no attempt left for the next transient error — the
                // failure mode `is_retryable` exists to survive. Still bounded:
                // one restart x one retry each is two reissues per blob.
                read.attempts = 0;
                return Reap::Again(index);
            }
            return Reap::Again(index);
        }
        if result == 0 {
            // Zero bytes with bytes still owed is EOF, never success.
            read.done = true;
            self.outstanding -= 1;
            let (base, filled, remaining) = (read.base, read.filled, read.remaining);
            return Reap::Failed(
                index,
                io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "expert blob at {base}: {remaining} of {} bytes past end of file",
                        filled + remaining
                    ),
                ),
            );
        }
        let errno = -result;
        let error = io::Error::from_raw_os_error(errno);
        if is_retryable(errno) && read.attempts == 0 {
            read.attempts += 1;
            io.add_retry();
            tracing::warn!(
                layer = read.layer,
                expert = read.expert,
                slot = read.slot,
                %error,
                "expert blob read failed, retrying once"
            );
            return Reap::Again(index);
        }
        read.done = true;
        self.outstanding -= 1;
        Reap::Failed(index, error)
    }

    /// Give up on a read that will never complete, so `outstanding` can
    /// still drain. The caller owes the slot whatever protection the reason
    /// demands.
    fn abandon(&mut self, index: usize) {
        if let Some(read) = self.reads.get_mut(index)
            && !read.done
        {
            read.done = true;
            self.outstanding -= 1;
        }
    }

    /// Drop every entry. Only valid once nothing is outstanding.
    fn clear(&mut self) {
        debug_assert_eq!(self.outstanding, 0, "clearing the table under live reads");
        self.reads.clear();
    }
}

/// Whether an errno is worth one more attempt.
///
/// `EIO` leads the list deliberately: it is what a transient media or
/// checksum failure surfaces as, and code that treats it as impossible is
/// code that corrupts a token silently.
fn is_retryable(errno: i32) -> bool {
    matches!(
        errno,
        libc::EIO | libc::EAGAIN | libc::EINTR | libc::ECANCELED | libc::ENOMEM
    )
}

/// The whole slot pool, leased once and held for the life of the stream.
///
/// Two allocators would otherwise be racing to name the same buffers:
/// [`LayerCache`] decides which slot an expert goes in, and [`SlotPool`]
/// decides which slot a lease gets. So the pool is drained at construction —
/// every slot leased, none ever returned — and the cache's slot index becomes
/// the only name. A slot's address is then fixed for the process, which is
/// what a cache entry that outlives a step needs.
///
/// **Drop order is load-bearing.** Every guard borrows the pool, so all of
/// them are released before the slab they point into is freed.
/// (`SlotPool::drop` would refuse to free a slab with a live lease and leak it
/// instead — correct, but 1.4 GiB of correct.) That ordering is spelled out in
/// [`Drop for SlotTable`](SlotTable#impl-Drop-for-SlotTable) rather than left
/// to field declaration order, because the pool is not dropped by a field at
/// all.
///
/// # Why the pool is a raw pointer
///
/// The guards hold `&'static SlotPool` borrows of a heap allocation stored in
/// the same value: a self-referential struct, which no lifetime can express.
/// The obvious spelling — keep a `Box<SlotPool>` and hand out `&*box` — is the
/// one shape that is *not* obviously sound: `Box` is the only non-reference
/// type rustc gives `noalias` to, so under Stacked and Tree Borrows a `Unique`
/// retag of the box (which happens every time it is moved, and on every
/// `&mut` through it) invalidates references previously derived from it. It is
/// why `ouroboros`, `yoke` and `self_cell` all reach for `AliasableBox` or a
/// raw pointer here. So there is no `Box` and no `transmute`: the allocation
/// is `Box::into_raw`'d once, every borrow is derived from the raw pointer,
/// and `Drop` gives it back to a `Box` after the last guard is gone.
///
/// # Retirement
///
/// A slot whose read can never be reaped is *retired*: its guard is leaked, so
/// the kernel may write into that buffer forever without aliasing anything,
/// and it is struck from `live` so no later step can name it. Cache slot
/// numbers index `live`, so retiring one renumbers the layer — which is safe
/// only because the caller rebuilds that layer's [`LayerCache`] from scratch
/// at the same moment. See [`ExpertStream::retire_slot`].
struct SlotTable {
    /// Flat `layer * slots_per_layer + pool slot`. `None` where a guard was
    /// leaked to protect a read that can never be reaped.
    guards: Box<[Option<SlotGuard<'static>>]>,
    /// Per layer, the pool slots still usable, indexed by *cache* slot. Starts
    /// as the identity and only ever shrinks.
    live: Box<[Vec<u32>]>,
    /// The pool every guard borrows, owned by this value and freed in `Drop`.
    pool: NonNull<SlotPool>,
    /// Slots the pool was built with, which is the stride of `guards` and the
    /// count before any retirement.
    slots_per_layer: u32,
}

// SAFETY: the only field that is not already `Send` is `pool`, a uniquely
// owned pointer to a `SlotPool`, and `SlotPool` is itself `Send`. Moving a
// `SlotTable` to another thread moves the pool with it, exactly as the
// `Box<SlotPool>` this replaced did.
unsafe impl Send for SlotTable {}

// SAFETY: `&SlotTable` grants no more than `&SlotPool` and `&SlotGuard`, both
// of which are `Sync`. `write_ptr`, the one method that hands out a writable
// address, takes `&mut self`.
unsafe impl Sync for SlotTable {}

impl fmt::Debug for SlotTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlotTable")
            .field("slots_per_layer", &self.slots_per_layer)
            .field("total_bytes", &self.pool().total_bytes())
            .finish()
    }
}

impl Drop for SlotTable {
    fn drop(&mut self) {
        // Every lease first: `SlotPool::drop` leaks the whole slab rather than
        // free memory a guard still names.
        self.guards = Box::default();
        // SAFETY: `pool` came from `Box::into_raw` in `new` and has not been
        // freed. Every guard derived from it has just been dropped, and
        // `SlotTable` hands no borrow of the pool to anything that outlives
        // it, so this is the last use of the allocation.
        drop(unsafe { Box::from_raw(self.pool.as_ptr()) });
    }
}

impl SlotTable {
    /// Allocate the pool and take every lease in it.
    fn new(slots_per_layer: u32, strides: &[u64]) -> Result<Self, SlotError> {
        let raw = Box::into_raw(Box::new(SlotPool::new(slots_per_layer, strides)?));
        // SAFETY: `raw` is a live, uniquely owned allocation this function is
        // about to take ownership of, and the borrow is only extended to
        // `'static` because the guards derived from it are dropped by
        // `Drop for SlotTable` before the allocation is freed. Deriving from
        // the raw pointer rather than from a `Box` is what keeps the guards
        // valid across every later move of this value: see the type's docs.
        let pool: &'static SlotPool = unsafe { &*raw };
        let n_layers = pool.n_layers();
        let mut guards: Vec<Option<SlotGuard<'static>>> = (0..strides.len()
            * slots_per_layer as usize)
            .map(|_| None)
            .collect();
        let mut failure = None;
        'lease: for layer in 0..n_layers {
            for _ in 0..slots_per_layer {
                match pool.acquire(layer) {
                    Ok(guard) => {
                        let flat =
                            layer as usize * slots_per_layer as usize + guard.index() as usize;
                        guards[flat] = Some(guard);
                    }
                    Err(error) => {
                        failure = Some(error);
                        break 'lease;
                    }
                }
            }
        }
        if let Some(error) = failure {
            drop(guards);
            // SAFETY: every lease taken above has just been dropped, so the
            // allocation is unborrowed and this is its last use.
            drop(unsafe { Box::from_raw(raw) });
            return Err(error);
        }
        Ok(Self {
            guards: guards.into_boxed_slice(),
            live: (0..n_layers)
                .map(|_| (0..slots_per_layer).collect())
                .collect(),
            // SAFETY: `Box::into_raw` never returns null.
            pool: unsafe { NonNull::new_unchecked(raw) },
            slots_per_layer,
        })
    }

    /// The pool this table owns.
    fn pool(&self) -> &SlotPool {
        // SAFETY: the allocation is live for the whole life of this value —
        // `Drop` is the only thing that frees it — and nothing hands out a
        // `&mut SlotPool`, so a shared borrow is always legal.
        unsafe { self.pool.as_ref() }
    }

    /// Pool slots layer `layer` can still use.
    fn usable(&self, layer: u32) -> u32 {
        self.live
            .get(layer as usize)
            .map_or(0, |live| live.len() as u32)
    }

    /// Flat index of `(layer, cache slot)`, or `None` if the layer is out of
    /// range, the slot is out of range, or the slot has been retired.
    fn flat(&self, layer: u32, slot: u32) -> Option<usize> {
        let pool_slot = *self.live.get(layer as usize)?.get(slot as usize)? as usize;
        Some(layer as usize * self.slots_per_layer as usize + pool_slot)
    }

    /// Read-only view of a slot's bytes. Only valid when no read is in
    /// flight into it, which the cache's `Filling` state is what proves.
    fn bytes(&self, layer: u32, slot: u32) -> Option<&[u8]> {
        let flat = self.flat(layer, slot)?;
        self.guards[flat].as_ref().map(SlotGuard::as_slice)
    }

    /// Mutable view, for the synchronous read path.
    fn bytes_mut(&mut self, layer: u32, slot: u32) -> Option<&mut [u8]> {
        let flat = self.flat(layer, slot)?;
        self.guards[flat].as_mut().map(SlotGuard::as_mut_slice)
    }

    /// Destination pointer for an asynchronous read.
    ///
    /// # Safety
    ///
    /// The caller inherits [`SlotGuard::as_mut_ptr`]'s contract in full:
    /// until the completion has been reaped, no slice view of this slot may
    /// be created from any thread, no second read may target it, and the
    /// guard must not be dropped. [`SlotTable::leak`] discharges the last of
    /// those on a teardown that cannot reap.
    #[cfg(feature = "io-uring")]
    unsafe fn write_ptr(&mut self, layer: u32, slot: u32) -> Option<*mut u8> {
        let flat = self.flat(layer, slot)?;
        // SAFETY: forwarded to the caller by this function's own contract.
        self.guards[flat]
            .as_mut()
            .map(|guard| unsafe { guard.as_mut_ptr() })
    }

    /// Give up a slot permanently, so nothing can be handed its address
    /// again. For a read whose completion can no longer be reaped.
    ///
    /// The buffer is leaked — a late kernel write into it then lands in memory
    /// nothing else will ever own — and the slot is struck from the layer, so
    /// [`SlotTable::flat`] can never resolve to it again however the caller's
    /// cache renumbers itself. Returns whether anything was retired.
    ///
    /// Retiring renumbers the layer's cache slots: everything above `slot`
    /// shifts down one. The caller must rebuild that layer's [`LayerCache`] to
    /// match, which is why this is not public beyond
    /// [`ExpertStream::retire_slot`]. Retiring several slots of one layer must
    /// be done in *descending* slot order for the same reason.
    fn retire(&mut self, layer: u32, slot: u32) -> bool {
        let Some(flat) = self.flat(layer, slot) else {
            return false;
        };
        if let Some(guard) = self.guards[flat].take() {
            guard.leak();
        }
        self.live[layer as usize].remove(slot as usize);
        true
    }

    /// Resident bytes of the whole pool.
    ///
    /// What was allocated, retirements included: a retired slot's pages are
    /// leaked, not returned, so they still count against the process.
    fn total_bytes(&self) -> u64 {
        self.pool().total_bytes() as u64
    }

    /// Byte offset of `(layer, cache slot)` from the slab base, or `None` when
    /// either index is out of range or the slot has been retired.
    ///
    /// Goes through `live`, so it answers for the layer as it is *now*: a
    /// retirement renumbers cache slots and this follows the renumbering.
    fn slot_offset(&self, layer: u32, slot: u32) -> Option<usize> {
        let pool_slot = *self.live.get(layer as usize)?.get(slot as usize)?;
        self.pool().slot_offset(layer, pool_slot)
    }

    /// The first layer whose slots are padded, as `(layer, stride, pitch)`.
    ///
    /// `None` means the slab is a gapless run of blob-sized buffers, which is
    /// what an arena carve needs. See [`SlotPool::padded_layer`].
    fn padded_layer(&self) -> Option<(u32, usize, usize)> {
        self.pool().padded_layer()
    }

    /// The first retired slot buffer lying inside the first `bytes` of the
    /// slab, as `(layer, offset from the slab base)`.
    ///
    /// A retired slot's guard was leaked *precisely because* a read into it can
    /// never be reaped, so the kernel may still be writing there — and the
    /// buffer stays inside the slab, because leaking it in place is what makes
    /// that write harmless. Retiring strikes the slot from `live`, so
    /// [`SlotTable::slot_offset`] can no longer name it; `guards` is what still
    /// records where it was, as the `None` it was replaced by.
    fn retired_within(&self, bytes: usize) -> Option<(u32, usize)> {
        let per_layer = self.slots_per_layer as usize;
        if per_layer == 0 {
            return None;
        }
        self.guards
            .iter()
            .enumerate()
            .filter(|(_, guard)| guard.is_none())
            .find_map(|(flat, _)| {
                let layer = (flat / per_layer) as u32;
                let pool_slot = (flat % per_layer) as u32;
                self.pool()
                    .slot_offset(layer, pool_slot)
                    .filter(|offset| *offset < bytes)
                    .map(|offset| (layer, offset))
            })
    }

    /// Base of the whole slab, for the prefill arena.
    ///
    /// # Safety
    ///
    /// Inherits [`SlotPool::slab_base`]'s contract: no guard of this pool may
    /// be read, written, or handed to the kernel while the returned pointer or
    /// anything derived from it is live. `&mut self` proves it for Rust
    /// references; the caller owes it for the kernel.
    unsafe fn slab_base(&mut self) -> NonNull<u8> {
        // SAFETY: forwarded to this function's own caller.
        unsafe { self.pool().slab_base() }
    }
}

/// The head of the idle slot-pool slab, borrowed as prefill scratch.
///
/// Not an allocation and not a copy: every byte is already resident and
/// pre-faulted by [`SlotPool::new`], and carving this touches none of them.
///
/// Deliberately lifetime-free. The natural spelling — `Arena<'a>` tied to the
/// `&'a mut ExpertStream` it came from — cannot be stored *beside* that same
/// `&mut` in [`super::sweep::LayerSweep`], so the exclusivity is carried by the
/// borrow the sweep already holds and this type stays a plain address. It is
/// private to the crate's io layer and never escapes a `&mut ExpertStream`; see
/// the module docs for the full argument.
#[derive(Debug, Clone, Copy)]
struct Arena {
    /// Slab base. [`SLOT_ALIGN`]-aligned, because the allocation is.
    base: NonNull<u8>,
    /// Bytes carved. A multiple of [`SLOT_ALIGN`].
    len: usize,
}

// SAFETY: an `Arena` is an address into a slab the `ExpertStream` that made it
// owns, and it never leaves that stream. Moving the stream to another thread
// moves the slab and the arena together, exactly as `SlotTable` already does.
unsafe impl Send for Arena {}

/// The step [`ExpertStream::begin_layer`] opened.
#[derive(Debug, Default)]
struct Step {
    active: bool,
    layer: u32,
    /// `(index into the request, slot)` for resident experts.
    hits: Vec<(usize, u32)>,
    /// `(index into the request, slot)` for experts being read.
    misses: Vec<(usize, u32)>,
    /// Every slot the step protected, to be released by `end_layer`.
    protected: Vec<u32>,
}

impl Step {
    /// An idle step preallocated for `top_k` experts, which is the widest one
    /// the model can plan.
    fn with_capacity(top_k: usize) -> Self {
        Self {
            hits: Vec::with_capacity(top_k),
            misses: Vec::with_capacity(top_k),
            protected: Vec::with_capacity(top_k),
            ..Self::default()
        }
    }

    fn reset(&mut self, layer: u32) {
        self.active = true;
        self.layer = layer;
        self.hits.clear();
        self.misses.clear();
        self.protected.clear();
    }
}

/// The expert read path: slot pool, per-layer caches, and the ring or the
/// fallback beneath them.
///
/// One decode thread drives one stream. See the module docs for the call
/// sequence and for what happens when a read fails.
pub struct ExpertStream {
    /// Blob geometry and the slab specs every view is carved with.
    reader: ExpertReader,
    layers: Box<[LayerState]>,
    caches: Box<[LayerCache]>,
    /// Reused across steps so the decode loop does not allocate.
    plan: CachePlan,
    step: Step,
    inflight: Inflight,
    /// Completions drained from the ring before they are acted on, so that
    /// reissuing does not need the completion queue borrowed. Bounded by the
    /// reads one step can have outstanding, twice over (a retagged read can
    /// leave a stale completion behind), and preallocated to that.
    #[cfg(feature = "io-uring")]
    completions: Vec<(u64, i32)>,
    /// The batch of completions currently being acted on, swapped out of
    /// `completions` so that a reissue may drain the ring into it meanwhile.
    #[cfg(feature = "io-uring")]
    reaping: Vec<(u64, i32)>,
    /// Declared before `slots`: the ring must be gone before the buffers its
    /// reads target are freed.
    #[cfg(feature = "io-uring")]
    ring: Option<IoUring>,
    /// Thread the ring was created on. `IORING_SETUP_SINGLE_ISSUER` binds a
    /// ring to its creator and answers `EEXIST` to any other task, and this
    /// type is `Send` on purpose, so the owner is tracked and the ring is
    /// rebuilt if the stream is driven from somewhere else.
    #[cfg(feature = "io-uring")]
    ring_owner: std::thread::ThreadId,
    /// How many times that has happened. Past a couple, the flags that make
    /// a ring thread-bound are dropped rather than paid for again.
    #[cfg(feature = "io-uring")]
    ring_rebinds: u32,
    slots: SlotTable,
    /// The prefill arena, while [`super::sweep`] holds this stream's `&mut`.
    /// `None` at every other moment, which is what makes a cache operation and
    /// an arena read unable to coexist.
    arena: Option<Arena>,
    /// Set once a sweep read became unreapable. The arena's bytes are then
    /// permanently unsafe to hand out again, so every later take is refused.
    arena_poisoned: bool,
    /// Experts the model routes per step, floored at 1.
    ///
    /// The slot count below which a layer can no longer serve a step, which is
    /// what makes a retirement's damage reportable at its cause; see
    /// [`ExpertStream::strand_arena`].
    top_k: u32,
    mode: StreamMode,
    support: DirectSupport,
    /// Whether layer files are opened with `O_DIRECT`.
    direct_open: bool,
    /// Whether first use of a layer file hashes it in full: the opt-in
    /// [`LoadOptions::verify_layer_hashes`], never under
    /// [`LoadOptions::skip_hashes`].
    hash_layers: bool,
    io: IoStats,
    /// Cache counters carried over from layers rebuilt by
    /// [`ExpertStream::retire_slot`], so [`ExpertStream::stats`] stays
    /// monotonic across a retirement.
    retired: CacheStats,
    /// Which half of the generation the caller says it is in.
    phase: StreamPhase,
    /// Cumulative totals at the moment `phase` began.
    phase_start: StreamStats,
    /// Totals accumulated by each phase, not counting the one now open.
    phase_totals: [StreamStats; 2],
    /// Test-only: submits still to be answered as `EBUSY` would be answered,
    /// which no kernel new enough to run this can be made to do for real.
    #[cfg(all(test, feature = "io-uring"))]
    stalled_enters: u32,
}

impl fmt::Debug for ExpertStream {
    /// Hand-written because `IoUring` is not `Debug`, and because dumping
    /// every layer's state would bury what a log line wants.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExpertStream")
            .field("mode", &self.mode)
            .field("direct_io", &self.support)
            .field("n_layers", &self.layers.len())
            .field("slots_per_layer", &self.slots.slots_per_layer)
            .field("cache_bytes", &self.slots.total_bytes())
            .field("outstanding", &self.inflight.outstanding)
            .finish()
    }
}

impl ExpertStream {
    /// Open every layer's expert file and size the cache from a byte budget.
    ///
    /// `cache_bytes` is a **total** budget for the slot pool across all
    /// layers, which is the dial `docs/architecture.md` specifies so that one
    /// configuration stays meaningful across models. It buys
    /// `cache_bytes / sum(page-aligned stride of every layer)` slots per
    /// layer: on the audited Qwen3-30B-A3B geometry (24 layers at 3,059,712 B
    /// and 24 at 2,654,208 B) 1,438.59 MiB buys exactly 11, which is the
    /// shipped dial. The result is clamped to a layer's expert count — slots
    /// past that can never be filled — and to
    /// [`MAX_SLOTS`](crate::io::MAX_SLOTS).
    ///
    /// Construction allocates and pre-faults the whole pool (the dominant
    /// cost of this call), leases every slot in it, probes whether O_DIRECT
    /// actually bypasses the page cache on this filesystem, and creates the
    /// ring. Layer files open lazily, verified on first use.
    ///
    /// The floor is `top_k` slots per layer, not one: a step routes `top_k`
    /// distinct experts and every one of them needs a slot at the same time,
    /// so anything less fails at the first forward pass rather than at
    /// startup. That is rejected here, naming the budget that would work —
    /// after the pool has been allocated and pre-faulted and the tokenizer has
    /// come up is far too late to learn it.
    ///
    /// # Errors
    ///
    /// [`IoError::Format`] when the layout does not validate against the
    /// manifest; [`IoError::CacheBudgetTooSmall`] when the budget buys fewer
    /// than `top_k` slots per layer; [`IoError::Slots`] when the budget cannot
    /// fit one slot per layer ([`SlotError::ZeroSlotsPerLayer`]) or the pool
    /// does not fit in memory; [`IoError::Cache`] when a layer's expert count
    /// is out of range; [`IoError::TooLarge`] for a blob stride past `u32`.
    pub fn new(
        dir: &Path,
        manifest: &Manifest,
        layout: &ExpertsLayout,
        cache_bytes: u64,
        options: LoadOptions,
    ) -> Result<Self, IoError> {
        // Validates the layout against the manifest and resolves every
        // layer's gate/up/down slab geometry.
        let reader = ExpertReader::new(dir, manifest, layout, options)?;

        let mut layers = Vec::with_capacity(layout.layers.len());
        let mut strides = Vec::with_capacity(layout.layers.len());
        for layer in &layout.layers {
            let entry = manifest
                .files
                .get(&layer.file)
                .ok_or_else(|| FormatError::MissingFileEntry(layer.file.clone()))?;
            layers.push(LayerState {
                path: dir.join(&layer.file),
                name: layer.file.clone(),
                stride: layer.stride,
                len: u32::try_from(layer.stride).map_err(|_| IoError::TooLarge {
                    what: "expert blob stride",
                    value: layer.stride,
                })?,
                n_experts: layer.n_experts,
                expected_size: entry.size,
                expected_sha256: entry.sha256.clone(),
                file: None,
            });
            strides.push(layer.stride);
        }

        let slots_per_layer =
            slots_for_budget(cache_bytes, &strides, &layers, manifest.arch.top_k)?;
        let slots = SlotTable::new(slots_per_layer, &strides)?;
        let mut caches = Vec::with_capacity(layers.len());
        for layer in &layers {
            caches.push(LayerCache::new(slots_per_layer, layer.n_experts)?);
        }

        // Every per-step buffer is sized for `top_k` here rather than grown on
        // token 0. Six `Vec`s reach exactly `top_k` and never more — the plan's
        // hit/miss lists, the open step's hit/miss/protected lists, and the
        // in-flight table — and they are per *process*, not per layer, so this
        // is six allocations once instead of six growths inside the first
        // token's layer loop.
        let top_k = manifest.arch.top_k.max(1) as usize;

        // Direct I/O needs a 4096 multiple on the offset *and* the length,
        // and both come from the layout's stride. A layout that does not
        // supply one cannot use O_DIRECT at all.
        let mut stream = Self {
            reader,
            layers: layers.into_boxed_slice(),
            caches: caches.into_boxed_slice(),
            plan: CachePlan::with_capacity(top_k),
            step: Step::with_capacity(top_k),
            inflight: Inflight::with_capacity(top_k),
            #[cfg(feature = "io-uring")]
            completions: Vec::with_capacity(completion_capacity(slots_per_layer)),
            #[cfg(feature = "io-uring")]
            reaping: Vec::with_capacity(completion_capacity(slots_per_layer)),
            #[cfg(feature = "io-uring")]
            ring: None,
            #[cfg(feature = "io-uring")]
            ring_owner: std::thread::current().id(),
            #[cfg(feature = "io-uring")]
            ring_rebinds: 0,
            slots,
            arena: None,
            arena_poisoned: false,
            top_k: top_k as u32,
            mode: StreamMode::Pread,
            support: DirectSupport::Unusable(DirectFault::Unsupported),
            direct_open: false,
            hash_layers: options.verify_layer_hashes && !options.skip_hashes,
            io: IoStats::default(),
            retired: CacheStats::default(),
            phase: StreamPhase::default(),
            phase_start: StreamStats::default(),
            phase_totals: [StreamStats::default(); 2],
            #[cfg(all(test, feature = "io-uring"))]
            stalled_enters: 0,
        };
        stream.support = stream.probe_direct_io(&strides);
        stream.direct_open = stream.support.is_usable();
        // A short read cannot be continued from an unaligned offset through an
        // O_DIRECT handle; see `Inflight::align`.
        stream.inflight.align = if stream.direct_open {
            direct::DIO_ALIGN as u32
        } else {
            0
        };

        #[cfg(feature = "io-uring")]
        {
            stream.ring = build_ring(true);
            stream.ring_owner = std::thread::current().id();
        }
        stream.mode = stream.pick_mode();

        let bytes = stream.slots.total_bytes();
        tracing::info!(
            mode = %stream.mode,
            direct_io = %stream.support,
            slots_per_layer,
            pool_bytes = bytes,
            budget_bytes = cache_bytes,
            n_layers = stream.layers.len(),
            "expert stream ready"
        );
        // The memory contract is about the page cache, not about the ring:
        // `Pread` with a verified O_DIRECT handle still bypasses it, which is
        // what `StreamMode::Pread`'s own docs say. So the warning follows the
        // probe, not the mode.
        if !stream.support.is_verified() {
            let measured = stream.support.fault() == Some(DirectFault::PageCacheGrew);
            tracing::warn!(
                mode = %stream.mode,
                direct_io = %stream.support,
                verdict = if measured { "measured" } else { "unmeasurable" },
                "{}; the 3 GB memory contract does not hold in this mode",
                if measured {
                    "expert reads are charged to the page cache"
                } else {
                    "expert reads could not be shown to bypass the page cache"
                }
            );
        } else if stream.mode != StreamMode::ODirect {
            tracing::info!(
                mode = %stream.mode,
                "no io_uring on this host; expert reads are synchronous but \
                 still bypass the page cache, so the memory contract holds"
            );
        }
        Ok(stream)
    }

    /// Slots each layer's cache was built with, as bought by the byte budget.
    ///
    /// A layer that has had a slot retired holds fewer than this; ask
    /// [`ExpertStream::usable_slots`] for the current figure.
    pub fn slots_per_layer(&self) -> u32 {
        self.slots.slots_per_layer
    }

    /// Slots layer `layer` can still use, which is
    /// [`ExpertStream::slots_per_layer`] less anything retired by a read that
    /// could not be reaped. `0` for a layer out of range.
    pub fn usable_slots(&self, layer: u32) -> u32 {
        self.slots.usable(layer)
    }

    /// Bytes the slot pool actually made resident.
    ///
    /// At most the `cache_bytes` the stream was built with, and less whenever
    /// the budget bought a fraction of a slot or was clamped.
    pub fn cache_bytes(&self) -> u64 {
        self.slots.total_bytes()
    }

    /// How reads are being served.
    pub fn mode(&self) -> StreamMode {
        self.mode
    }

    /// Whether expert reads are verified to bypass the page cache.
    ///
    /// Orthogonal to [`ExpertStream::mode`]: the synchronous fallback can
    /// still be reading with a verified O_DIRECT handle.
    pub fn direct_io(&self) -> bool {
        self.support.is_verified()
    }

    /// What the startup probe concluded about direct I/O, verbatim.
    pub fn direct_support(&self) -> DirectSupport {
        self.support
    }

    /// Layers the stream serves.
    pub fn n_layers(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Plan one routing step and submit a read for every miss.
    ///
    /// `experts` is the top-k the router selected, in routed order. Does not
    /// block: the reads are in flight when this returns, so the caller should
    /// compute [`ExpertStream::hits`] before calling
    /// [`ExpertStream::await_misses`]. Exactly one
    /// [`ExpertStream::end_layer`] must follow.
    ///
    /// A duplicate expert id inside one request resolves once and is listed
    /// once, at its first position.
    ///
    /// # Errors
    ///
    /// [`IoError::LayerOutOfRange`]; [`IoError::Cache`] when an expert id is
    /// out of range, the step needs more slots than the layer has, or every
    /// slot is protected; [`IoError::Format`] when the layer file fails its
    /// first-use size or hash check; [`IoError::Io`] when the file cannot be
    /// opened, a read cannot be submitted, or a step is already open.
    pub fn begin_layer(&mut self, layer: u32, experts: &[u32]) -> Result<(), IoError> {
        let index = self.layer_index(layer)?;
        if self.step.active {
            return Err(self.protocol_error(
                layer,
                "begin_layer called with a step already open; end_layer first",
            ));
        }
        if self.inflight.outstanding > 0 {
            // Those reads are still writing into slots the previous step
            // assigned. Forgetting them here would recycle a live DMA
            // destination, so the step is refused rather than started.
            return Err(self.protocol_error(
                layer,
                "reads from an earlier step were never awaited; call \
                 await_misses before beginning another layer",
            ));
        }
        self.ensure_open(index)?;
        #[cfg(feature = "io-uring")]
        self.rebind_ring();

        {
            let plan = &mut self.plan;
            let cache = &mut self.caches[index];
            cache.plan(experts, plan)?;
        }
        self.step.reset(layer);
        for &(expert, slot) in self.plan.hits() {
            let request = position_of(experts, expert);
            self.step.hits.push((request, slot));
            self.step.protected.push(slot);
        }
        for &(expert, slot) in self.plan.misses() {
            let request = position_of(experts, expert);
            self.step.misses.push((request, slot));
            self.step.protected.push(slot);
        }

        self.inflight.clear();
        #[cfg(feature = "io-uring")]
        {
            // Anything left here names a read of a step that is over; its tag
            // can only resolve as stale, so it is dropped rather than walked.
            self.completions.clear();
        }
        let stride = self.layers[index].stride;
        let len = self.layers[index].len;
        let mut failure = None;
        for i in 0..self.plan.misses().len() {
            let (expert, slot) = self.plan.misses()[i];
            let read = self
                .inflight
                .track(layer, slot, expert, u64::from(expert) * stride, len);
            if let Err(error) = self.submit(read) {
                // This one never reached the kernel, so nothing will ever
                // complete it: resolve it here or the table never drains.
                self.inflight.abandon(read);
                failure = Some(self.fail(read, flatten_io(error)));
                break;
            }
        }
        if let Some(error) = failure {
            // Earlier misses of the same step may already be in the ring, and
            // they own their slots until their completions land, so they are
            // drained before the step unwinds.
            let _ = self.await_misses();
            self.abandon_step();
            return Err(error);
        }
        if let Err(error) = self.flush() {
            self.strand_step();
            return Err(error);
        }
        Ok(())
    }

    /// Resident experts of the open step, as `(index into the request the
    /// step was planned from, slot)`.
    pub fn hits(&self) -> &[(usize, u32)] {
        &self.step.hits
    }

    /// Experts being read for the open step, same encoding as
    /// [`ExpertStream::hits`]. Their slots hold nothing usable until
    /// [`ExpertStream::await_misses`] returns.
    pub fn misses(&self) -> &[(usize, u32)] {
        &self.step.misses
    }

    /// Block until every read [`ExpertStream::begin_layer`] submitted has
    /// completed.
    ///
    /// A failed blob read is retried once. On a terminal failure every other
    /// read of the step is still drained first — their slots are live DMA
    /// destinations until their completions are reaped — the step is then
    /// cleaned up (failed slots invalidated, the rest released) and the first
    /// error is returned. [`ExpertStream::end_layer`] after that is a no-op.
    ///
    /// # Errors
    ///
    /// [`IoError::Io`] wrapping the read's errno after the retry, an
    /// unexpected EOF, or a submission failure.
    pub fn await_misses(&mut self) -> Result<(), IoError> {
        if self.inflight.outstanding == 0 {
            return Ok(());
        }
        let mut failure = None;
        if let Err(error) = self.drive_reads(None, &mut failure) {
            // The ring itself failed, so nothing will ever reap what is still
            // in flight: those slots are given up rather than recycled under a
            // write we cannot see.
            self.strand_step();
            return Err(failure.unwrap_or(error));
        }
        match failure {
            None => Ok(()),
            Some(error) => {
                self.abandon_step();
                Err(error)
            }
        }
    }

    /// Byte view over a resident slot's gate/up/down slabs.
    ///
    /// # Errors
    ///
    /// [`IoError::LayerOutOfRange`]; [`IoError::Io`] when the slot is out of
    /// range for the layer, or holds no valid bytes because its read has not
    /// completed. That check is also what makes this safe to call while other
    /// reads of the same step are in flight.
    pub fn view(&self, layer: u32, slot: u32) -> Result<ExpertView<'_>, IoError> {
        let index = self.layer_index(layer)?;
        if !self.caches[index].is_ready(slot) {
            return Err(self.protocol_error(
                layer,
                &format!("slot {slot} holds no completed expert blob"),
            ));
        }
        let bytes = self.slots.bytes(layer, slot).ok_or_else(|| {
            self.protocol_error(layer, &format!("slot {slot} is not in the pool"))
        })?;
        self.reader.view_over(layer, bytes)
    }

    // ---------------------------------------------------------------------
    // The prefill arena. See the module docs for why borrowing the pool is
    // sound; everything below is `pub(super)` because `io::sweep` is the only
    // legitimate caller and the borrow it holds is half the argument.
    // ---------------------------------------------------------------------

    /// Stream one layer's expert file front to back, bypassing the cache.
    ///
    /// This is prefill. The returned [`LayerSweep`] holds this stream
    /// exclusively until it is finished or dropped, which is what lets it
    /// borrow the idle slot pool as scratch for its large reads; see
    /// [`super::sweep`] for the shape of the loop and `io::stream`'s module
    /// docs for why the borrow is sound.
    ///
    /// `plan` is the caller's to reuse across layers and chunks — it is
    /// rebuilt here, so passing the same one every time means only the first
    /// layer allocates. `routed` is the set of experts the chunk sends any row
    /// to, in any order; a window none of them fall in is never read.
    ///
    /// **Taking a sweep empties the decode cache**, deliberately: its buffers
    /// are what the sweep reads into. The ghost-LFU frequency counters survive.
    ///
    /// # Errors
    ///
    /// [`SweepError::ReadsInFlight`] or [`SweepError::StepOpen`] when the
    /// decode path is mid-step; [`SweepError::ArenaTooSmall`] when the expert
    /// cache budget cannot hold one window; [`SweepError::PaddedSlots`] for a
    /// layout whose blob stride is not 4096-aligned; [`SweepError::BadDials`]
    /// or [`SweepError::ExpertOutOfRange`] for bad input; [`SweepError::Io`]
    /// when the layer file cannot be opened or the first reads cannot be
    /// submitted.
    pub fn sweep_layer<'a>(
        &'a mut self,
        plan: &'a mut super::SweepPlan,
        layer: u32,
        routed: &[u32],
        config: super::SweepConfig,
    ) -> Result<super::LayerSweep<'a>, SweepError> {
        super::LayerSweep::begin(self, plan, layer, routed, config)
    }

    /// Open a chunked prefill: `scratch_bytes` of driver staging plus a sweep
    /// ring, both carved from the idle slot pool.
    ///
    /// What [`ExpertStream::sweep_layer`] cannot do. A layer-major prefill
    /// writes each expert's output into an `[n_rows][top_k][hidden]` staging
    /// buffer *while* it consumes the sweep, and both have to come out of the
    /// pool or prefill costs bytes the 3 GB budget does not have. A
    /// `LayerSweep` parks this stream's `&mut` for its whole life, so a driver
    /// holding one cannot hold a scratch span too;
    /// [`PrefillSession::split`](super::PrefillSession::split) hands out both
    /// at once as disjoint halves of one carve.
    ///
    /// The scratch is **not zeroed** — see
    /// [`PrefillSession`](super::PrefillSession) — and the session owns the
    /// arena until it is finished or dropped.
    ///
    /// # Errors
    ///
    /// Everything [`ExpertStream::sweep_layer`] raises for the carve:
    /// [`SweepError::ReadsInFlight`], [`SweepError::StepOpen`],
    /// [`SweepError::PaddedSlots`], [`SweepError::ArenaOverRetired`],
    /// [`SweepError::ArenaPoisoned`], and [`SweepError::ArenaTooSmall`] when
    /// the scratch plus a ring for the widest layer does not fit the pool.
    pub fn begin_prefill(
        &mut self,
        scratch_bytes: usize,
        config: super::SweepConfig,
    ) -> Result<super::PrefillSession<'_>, SweepError> {
        super::PrefillSession::begin(self, scratch_bytes, config)
    }

    /// Borrow the head of the idle slot-pool slab as `bytes` of prefill
    /// scratch, invalidating the decode cache's slot occupancy.
    ///
    /// Nothing is allocated and nothing is faulted: every byte was made
    /// resident by [`SlotPool::new`] and the carve is address arithmetic. The
    /// occupancy reset is not a side effect but the point — sweep bytes land in
    /// those buffers, so any surviving cache entry would name an expert that is
    /// no longer there. The ghost-LFU frequency counters and the fetched flags
    /// are indexed by expert id and survive, exactly as they survive an
    /// eviction.
    ///
    /// # Errors
    ///
    /// [`SweepError::ReadsInFlight`] / [`SweepError::StepOpen`] when the cache
    /// is mid-step; [`SweepError::PaddedSlots`] when some layer's pitch exceeds
    /// its stride, so the slab is not a gapless run of blob-sized buffers;
    /// [`SweepError::ArenaTooSmall`] when the pool is smaller than the carve;
    /// [`SweepError::ArenaOverRetired`] when the carve would cover a buffer
    /// some earlier lost read may still be writing into;
    /// [`SweepError::ArenaPoisoned`] once a sweep read has been lost;
    /// [`SweepError::ArenaOut`] when one is out already.
    pub(super) fn take_arena(&mut self, bytes: usize) -> Result<(), SweepError> {
        if self.arena_poisoned {
            return Err(SweepError::ArenaPoisoned);
        }
        // A second carve over the first one's bytes. Every holder gives the
        // arena back on `Drop`, which both `?` and a panic run — but leaking is
        // safe Rust, and a forgotten `PrefillSession` ends its borrow of this
        // stream without ever releasing. The next carve would then hand out a
        // second `&mut [u8]` over the first one's scratch, which is UB reached
        // from two safe `pub` calls.
        if let Some(arena) = self.arena {
            tracing::error!(
                live_bytes = arena.len,
                bytes,
                "refusing a second prefill arena over a live one"
            );
            return Err(SweepError::ArenaOut { bytes: arena.len });
        }
        if self.inflight.outstanding > 0 {
            return Err(SweepError::ReadsInFlight {
                outstanding: self.inflight.outstanding,
            });
        }
        if self.step.active {
            return Err(SweepError::StepOpen {
                layer: self.step.layer,
            });
        }
        if let Some((layer, stride, pitch)) = self.slots.padded_layer() {
            return Err(SweepError::PaddedSlots {
                layer,
                stride,
                pitch,
            });
        }
        let available = self.slots.total_bytes();
        if bytes as u64 > available {
            return Err(SweepError::ArenaTooSmall {
                needed: bytes as u64,
                available,
            });
        }
        // `outstanding == 0` is *not* the same statement as "no kernel write is
        // outstanding into the slab". `strand_step` and `strand_arena` both
        // drop reads out of the table with `Inflight::abandon`, whose whole
        // reason for existing is that those reads can never be reaped — the
        // counter drains because the read is unknowable, not because it
        // finished. What they leave behind is a *retired* slot: a buffer leaked
        // in place, still inside the slab, which the kernel may write into
        // forever. So the carve is checked against those buffers rather than
        // against the counter.
        if let Some((layer, offset)) = self.slots.retired_within(bytes) {
            tracing::error!(
                layer,
                offset,
                bytes,
                "refusing a prefill arena over a retired slot buffer"
            );
            return Err(SweepError::ArenaOverRetired {
                layer,
                offset,
                bytes,
            });
        }
        // SAFETY: the caller of this method parks the `&mut ExpertStream` it
        // borrowed for the whole life of the arena, so no slot guard of this
        // pool is reachable — from safe code or otherwise — until
        // `release_arena` runs. For the kernel half: nothing is outstanding
        // *and* nothing the carve covers was ever abandoned, both checked
        // above, which together are what rule out a live DMA into these bytes.
        let base = unsafe { self.slots.slab_base() };
        debug_assert!(
            base.as_ptr() as usize as u64 % direct::DIO_ALIGN == 0,
            "the slot pool slab is allocated 4096-aligned"
        );
        self.invalidate_all_slots();
        self.inflight.clear();
        #[cfg(feature = "io-uring")]
        self.completions.clear();
        self.arena = Some(Arena { base, len: bytes });
        self.io.in_sweep = true;
        tracing::debug!(
            arena_bytes = bytes,
            pool_bytes = available,
            "prefill arena taken; the expert cache is now empty"
        );
        Ok(())
    }

    /// Give the arena back and leave the cache empty.
    ///
    /// Invalidates occupancy a second time. Redundant while the borrow holds —
    /// nothing could have populated a slot with the stream exclusively borrowed
    /// — and it is one pass over a few hundred slot records, which is the
    /// cheapest possible way to keep "no slot ever reports ready with sweep
    /// bytes in it" true by construction rather than by argument.
    pub(super) fn release_arena(&mut self) {
        if self.arena.take().is_none() {
            return;
        }
        // Both callers drain before releasing, so this is always taken — but
        // `invalidate_all_slots` unprotects every slot of every layer without
        // proving anything about the kernel, and "the caller drained first" is
        // an argument, not a check. The check is one comparison and it is
        // already made on the next line for the read table.
        if self.inflight.outstanding == 0 {
            self.invalidate_all_slots();
            self.inflight.clear();
        } else {
            tracing::error!(
                outstanding = self.inflight.outstanding,
                "the prefill arena was released with window reads still in \
                 flight; leaving slot occupancy alone, since it is already \
                 empty from the carve"
            );
        }
        self.io.in_sweep = false;
    }

    /// Bytes currently carved, or `0` when no arena is out.
    pub(super) fn arena_len(&self) -> usize {
        self.arena.map_or(0, |arena| arena.len)
    }

    /// Base address of the arena, or `None` when none is out.
    ///
    /// For [`super::PrefillSession`], which needs an address rather than a
    /// slice: the scratch span and the sweep ring are two `&mut` into one
    /// carve, so the split has to be made from a pointer. The exclusivity is
    /// the same borrow every other arena method rests on — the holder parks
    /// this stream's `&mut` — and the disjointness is the session's own
    /// arithmetic.
    pub(super) fn arena_base(&self) -> Option<NonNull<u8>> {
        self.arena.map(|arena| arena.base)
    }

    /// Read-only view of `len` arena bytes at `offset`.
    ///
    /// # Safety
    ///
    /// Every read targeting this range must have reached a terminal state —
    /// its completion reaped, or its `pread` returned. The type system cannot
    /// see a kernel write, so this is the caller's to prove;
    /// [`super::sweep::LayerSweep`] proves it by only calling this for a window
    /// it has already awaited.
    pub(super) unsafe fn arena_bytes(&self, offset: usize, len: usize) -> Option<&[u8]> {
        let arena = self.arena?;
        if offset.checked_add(len)? > arena.len {
            return None;
        }
        // SAFETY: the range is inside the slab, every byte of which was
        // initialized by `SlotPool::new`, and the arena only exists while its
        // holder has this stream exclusively borrowed — so no `&mut` view of
        // the same bytes can exist. The kernel half is the caller's promise.
        Some(unsafe { std::slice::from_raw_parts(arena.base.as_ptr().add(offset), len) })
    }

    /// Submit one arena-destined window read and return its in-flight index.
    ///
    /// Alignment is the caller's to establish — `sweep::check_window` does it
    /// from the geometry — and asserted here as well, because
    /// [`is_retryable`] deliberately excludes `EINVAL`, so an unaligned
    /// O_DIRECT read is a hard mid-prefill failure with no retry behind it.
    ///
    /// # Errors
    ///
    /// [`SweepError::Misaligned`] when offset, length or destination is not a
    /// [`direct::DIO_ALIGN`] multiple; [`SweepError::Io`] when the read cannot
    /// be handed to the kernel.
    pub(super) fn sweep_submit(
        &mut self,
        layer: u32,
        buffer: u32,
        first_expert: u32,
        file_offset: u64,
        len: u32,
        arena_offset: usize,
    ) -> Result<usize, SweepError> {
        let Some(arena) = self.arena else {
            return Err(SweepError::NoArena);
        };
        // Unconditionally, not only when the handle is O_DIRECT: the startup
        // probe answers for the geometry as it was at open time, and this is
        // the read that would die of an un-retryable `EINVAL`.
        check_aligned("sweep read offset", file_offset)?;
        check_aligned("sweep read length", u64::from(len))?;
        check_aligned(
            "sweep read destination",
            arena.base.as_ptr() as usize as u64 + arena_offset as u64,
        )?;
        self.ensure_open(self.layer_index(layer)?)?;
        let index =
            self.inflight
                .track_arena(layer, buffer, first_expert, file_offset, len, arena_offset);
        if let Err(error) = self.submit(index) {
            // Never reached the kernel, so nothing will ever write into the
            // arena for it: resolve it here or the table never drains.
            self.inflight.abandon(index);
            return Err(error.into());
        }
        if let Err(error) = self.flush() {
            // Whether the kernel took the reads already pushed is unknowable,
            // and they target the arena rather than one slot.
            return Err(self.strand_arena_error(error));
        }
        Ok(index)
    }

    /// Block until the window read `until` has landed, or until every
    /// outstanding window has if it is `None`.
    ///
    /// Stopping at one window is what keeps the double buffer double: the
    /// windows behind it stay in flight against the caller's compute.
    ///
    /// # Errors
    ///
    /// [`SweepError::Io`] wrapping the read's errno after its retry, an
    /// unexpected EOF, or a submission failure. A failure that leaves reads
    /// unreapable retires every slot the arena covers first.
    pub(super) fn sweep_await(&mut self, until: Option<usize>) -> Result<(), SweepError> {
        let mut failure = None;
        if let Err(error) = self.drive_reads(until, &mut failure) {
            let source = failure.unwrap_or(error);
            return Err(self.strand_arena_error(source));
        }
        match failure {
            None => Ok(()),
            Some(error) => Err(error.into()),
        }
    }

    /// Reads this stream has handed to the kernel and not yet reaped.
    ///
    /// For [`LayerSweep::begin_within`](super::sweep::LayerSweep), which reuses
    /// a ring the session already carved and so reaches none of
    /// [`take_arena`](Self::take_arena)'s checks. Note what this does *not*
    /// say: `Inflight::abandon` drains the counter for reads that can never be
    /// reaped, so `0` means "nothing is awaitable", not "nothing is writing".
    /// The second statement is `SlotTable::retired_within`'s to make.
    pub(super) fn reads_outstanding(&self) -> usize {
        self.inflight.outstanding
    }

    /// Forget every resolved read, so one sweep's window indices do not
    /// accumulate across layers. Only valid with nothing outstanding.
    pub(super) fn sweep_clear_reads(&mut self) {
        if self.inflight.outstanding == 0 {
            self.inflight.clear();
        }
    }

    /// Count one window the sweep looked at.
    pub(super) fn count_window(&mut self, read: bool) {
        if read {
            self.io.sweep_windows_read += 1;
        } else {
            self.io.sweep_windows_skipped += 1;
        }
    }

    /// Open and verify a layer's expert file, the same first-use path
    /// `begin_layer` takes. Shared rather than duplicated: one O_DIRECT open,
    /// one size check, one optional hash, per layer per process.
    ///
    /// # Errors
    ///
    /// [`IoError::LayerOutOfRange`]; [`IoError::Format`] on the size or hash
    /// check; [`IoError::Io`] when the file cannot be opened.
    pub(super) fn open_layer(&mut self, layer: u32) -> Result<(), IoError> {
        let index = self.layer_index(layer)?;
        self.ensure_open(index)
    }

    /// A layer's blob stride and expert count.
    ///
    /// # Errors
    ///
    /// [`IoError::LayerOutOfRange`] past the last layer.
    pub(super) fn layer_geometry(&self, layer: u32) -> Result<(u64, u32), IoError> {
        let index = self.layer_index(layer)?;
        let state = &self.layers[index];
        Ok((state.stride, state.n_experts))
    }

    /// The resolved gate/up/down geometry, for slicing arena bytes.
    pub(super) fn expert_reader(&self) -> &ExpertReader {
        &self.reader
    }

    /// Drop every layer's slot occupancy, keeping the ghost history.
    ///
    /// `invalidate` is `unsafe` because it unprotects a slot without proving
    /// its read is over; here that proof is the caller's — every path into this
    /// runs with nothing in flight.
    fn invalidate_all_slots(&mut self) {
        for cache in &mut self.caches {
            // SAFETY: both call sites check `inflight.outstanding == 0` first,
            // and both also rule out a read that was *abandoned* rather than
            // reaped — `take_arena` by refusing a carve over any retired slot
            // buffer, `release_arena` because `strand_arena` takes the arena
            // away, so the release is a no-op after one. So no read is writing
            // into any slot of any layer at either call site, which is exactly
            // `reset_occupancy`'s obligation — one statement about one moment,
            // made once per layer rather than once per slot.
            unsafe { cache.reset_occupancy() };
        }
    }

    /// Give up the arena for good, after a window read that can never be
    /// reaped.
    ///
    /// The one failure the borrow cannot contain. A lost sweep read may land
    /// anywhere in the arena, and the arena is the head of the slot pool, so
    /// every slot it overlaps is **retired**: the buffer is leaked, and the
    /// layer's cache is rebuilt that much smaller so nothing can ever be handed
    /// the address again. The stream then refuses to sweep for the rest of the
    /// process — a second arena over the same bytes would be a second read into
    /// a live DMA destination.
    ///
    /// Loud and expensive, but it terminates and it never aliases.
    ///
    /// # The damage, at the shipped dials
    ///
    /// The arena is carved from the **head** of the slab, which is where layer
    /// 0 lives, and it is bigger than one layer's whole slot row: 2 windows of
    /// 8 experts at the 3,059,712 B stride is 46.7 MiB against layer 0's 11
    /// slots x 2.92 MiB = 32.1 MiB. So one unreapable window read leaves layer
    /// 0 with **0** slots and layer 1 with 6 — both under a `top_k` of 8, and
    /// both dead for the rest of the process.
    ///
    /// Carving the arena from the *tail* instead does not fix this; it moves
    /// the damage to layer 47. Nothing can fix it while the arena is the pool,
    /// which is the design. What this does instead is say so **here**, at the
    /// cause, as [`SweepError::CacheStranded`] — the alternative is a
    /// [`CacheError::TooFewSlots`](crate::io::CacheError::TooFewSlots) three
    /// decode steps later, which names a symptom and no cause at all.
    ///
    /// Returns the first layer left below `top_k` and how many slots it has,
    /// or `None` when every layer can still serve a step.
    #[must_use = "a layer left below top_k has to be reported at the cause"]
    fn strand_arena(&mut self) -> Option<(u32, u32)> {
        let len = self.arena_len();
        for index in 0..self.inflight.reads.len() {
            let read = self.inflight.reads[index];
            if read.done {
                continue;
            }
            tracing::error!(
                layer = read.layer,
                window = read.slot,
                first_expert = read.expert,
                "prefill window read cannot be reaped; the arena is given up"
            );
            self.inflight.abandon(index);
        }
        self.arena = None;
        self.arena_poisoned = true;
        self.io.in_sweep = false;
        // Descending in both indices: retiring one slot renumbers every higher
        // slot of the same layer.
        for layer in (0..self.n_layers()).rev() {
            for slot in (0..self.slots.usable(layer)).rev() {
                if self
                    .slots
                    .slot_offset(layer, slot)
                    .is_some_and(|offset| offset < len)
                {
                    self.retire_slot(layer, slot);
                }
            }
        }
        tracing::error!(
            arena_bytes = len,
            "the prefill arena was lost to an unreapable read; every slot it \
             covered is retired and this process will not sweep again"
        );
        let stranded = (0..self.n_layers())
            .map(|layer| (layer, self.slots.usable(layer)))
            .find(|&(_, slots)| slots < self.top_k);
        if let Some((layer, slots)) = stranded {
            tracing::error!(
                layer,
                slots,
                top_k = self.top_k,
                "the retirement left a layer below top_k; that layer can no \
                 longer serve a decode step"
            );
        }
        stranded
    }

    /// Give up the arena and name the damage: the read failure when the cache
    /// survived it, [`SweepError::CacheStranded`] when it did not.
    fn strand_arena_error(&mut self, source: IoError) -> SweepError {
        match self.strand_arena() {
            Some((layer, slots)) => SweepError::CacheStranded {
                layer,
                slots,
                top_k: self.top_k,
                source,
            },
            None => SweepError::Io(source),
        }
    }

    /// Release every slot the step protected, hits included.
    ///
    /// Exactly one call per [`ExpertStream::begin_layer`]. Releasing only the
    /// misses would leave every hit slot protected forever, which
    /// [`LayerCache::stuck_protected_slot`] detects and `plan` asserts
    /// against in debug builds.
    pub fn end_layer(&mut self, layer: u32) {
        if !self.step.active && self.inflight.outstanding == 0 {
            tracing::warn!(layer, "end_layer without an open step");
            return;
        }
        if self.step.active && self.step.layer != layer {
            tracing::error!(
                asked = layer,
                open = self.step.layer,
                "end_layer for a different layer than the open step; \
                 releasing the open one"
            );
        }
        if self.inflight.outstanding > 0 {
            // The caller skipped `await_misses`, so those reads are still
            // live DMA into their slots and nothing here can prove otherwise.
            // Leaving them in the table would refuse every later
            // `begin_layer`, so they are retired instead: the buffers are
            // leaked, the layers shrink, and the stream keeps working on what
            // is left. Expensive and loud, but it terminates.
            tracing::error!(
                layer = self.step.layer,
                outstanding = self.inflight.outstanding,
                "end_layer with reads still in flight; await_misses was \
                 skipped, so their slots are retired and the layer shrinks"
            );
            self.strand_step();
            return;
        }
        let index = self.step.layer as usize;
        if let Some(cache) = self.caches.get_mut(index) {
            for slot in self.step.protected.drain(..) {
                cache.release(slot);
            }
        }
        self.step.active = false;
        if self.inflight.outstanding == 0 {
            self.inflight.clear();
        }
    }

    /// Cumulative telemetry: the cache half summed across layers, the I/O
    /// half counted by the stream.
    ///
    /// Summed over the whole run, prefill and decode together. Quoting this as
    /// a decode figure understates the steady state, because prefill's share
    /// of cold misses is much the larger; use [`ExpertStream::stats_in`].
    pub fn stats(&self) -> StreamStats {
        let mut stats = StreamStats {
            bytes_read: self.io.bytes_read,
            reads_submitted: self.io.reads_submitted,
            read_retries: self.io.read_retries,
            io_wait: self.io.io_wait,
            sweep_bytes_read: self.io.sweep_bytes_read,
            sweep_reads_submitted: self.io.sweep_reads_submitted,
            sweep_read_retries: self.io.sweep_read_retries,
            sweep_io_wait: self.io.sweep_io_wait,
            sweep_windows_read: self.io.sweep_windows_read,
            sweep_windows_skipped: self.io.sweep_windows_skipped,
            ..StreamStats::default()
        };
        let mut fold = |cache: CacheStats| {
            stats.hits += cache.hits;
            stats.pending_hits += cache.pending_hits;
            stats.misses += cache.misses;
            stats.cold_misses += cache.cold_misses;
            stats.eviction_misses += cache.eviction_misses;
        };
        for cache in &self.caches {
            fold(cache.stats());
        }
        // Layers rebuilt by a retirement lost their counters with their cache.
        fold(self.retired);
        stats
    }

    /// Telemetry for one phase of the generation only.
    ///
    /// The decode-only figure is the one a steady-state hit rate is quoted
    /// from. Requires the caller to have declared its phase transitions
    /// through [`ExpertStream::set_phase`]; a caller that never does gets
    /// everything under [`StreamPhase::Prefill`] and nothing under
    /// [`StreamPhase::Decode`].
    ///
    /// `stats_in(Prefill) + stats_in(Decode) == stats()`, field by field.
    pub fn stats_in(&self, phase: StreamPhase) -> StreamStats {
        let closed = self.phase_totals[phase.index()];
        if phase == self.phase {
            closed.plus(&self.stats().since(&self.phase_start))
        } else {
            closed
        }
    }

    /// Which half of the generation the stream is counting against.
    pub fn phase(&self) -> StreamPhase {
        self.phase
    }

    /// Declare which half of the generation the stream is now serving.
    ///
    /// Everything counted from here on is attributed to `phase`. Call it
    /// before the prompt pass and again before the first decode token; calling
    /// it with the phase already open does nothing, so it is safe to call once
    /// per forward pass. Switching back and forth is allowed and accumulates:
    /// a second prefill (a second prompt on the same stream) adds to the
    /// prefill totals rather than replacing them.
    pub fn set_phase(&mut self, phase: StreamPhase) {
        if phase == self.phase {
            return;
        }
        let now = self.stats();
        let closed = &mut self.phase_totals[self.phase.index()];
        *closed = closed.plus(&now.since(&self.phase_start));
        self.phase_start = now;
        self.phase = phase;
    }

    /// Completions that named no live read. Always zero in a healthy run.
    pub fn stale_completions(&self) -> u64 {
        self.io.stale_completions
    }

    /// Validate a layer index against the layout.
    fn layer_index(&self, layer: u32) -> Result<usize, IoError> {
        if (layer as usize) < self.layers.len() {
            Ok(layer as usize)
        } else {
            Err(IoError::LayerOutOfRange {
                layer,
                n_layers: self.n_layers(),
            })
        }
    }

    /// Wrap an I/O failure against the layer the open step is reading.
    ///
    /// Only the ring raises errors that are about the step rather than about a
    /// named call; the synchronous path always has a read to blame.
    #[cfg(feature = "io-uring")]
    fn layer_error(&self, source: io::Error) -> IoError {
        IoError::Io {
            path: self
                .layers
                .get(self.step.layer as usize)
                .map(|state| state.path.clone())
                .unwrap_or_default(),
            source,
        }
    }

    /// An error that names the layer file the call was about.
    fn protocol_error(&self, layer: u32, message: &str) -> IoError {
        IoError::Io {
            path: self
                .layers
                .get(layer as usize)
                .map(|state| state.path.clone())
                .unwrap_or_default(),
            source: io::Error::new(io::ErrorKind::InvalidInput, message.to_owned()),
        }
    }

    /// Open and verify a layer file on first use.
    ///
    /// Size always, SHA-256 only under [`LoadOptions::verify_layer_hashes`] —
    /// the same gate [`ExpertReader`] applies, because this path never goes
    /// through it. That hash pass streams the whole file through the page
    /// cache, so what it pulled in is dropped afterwards: leaving gigabytes
    /// of layer file resident would defeat the O_DIRECT budget before the
    /// first token.
    fn ensure_open(&mut self, index: usize) -> Result<(), IoError> {
        if self.layers[index].file.is_some() {
            return Ok(());
        }
        let path = self.layers[index].path.clone();
        let (file, applied) =
            direct::open(&path, self.direct_open).map_err(|e| IoError::io(&path, e))?;
        let size = file.metadata().map_err(|e| IoError::io(&path, e))?.len();
        let state = &self.layers[index];
        if size != state.expected_size {
            return Err(FormatError::SizeMismatch {
                name: state.name.clone(),
                expected: state.expected_size,
                actual: size,
            }
            .into());
        }
        if self.hash_layers {
            let actual = sha256_file(&path)?;
            let state = &self.layers[index];
            if !actual.eq_ignore_ascii_case(&state.expected_sha256) {
                return Err(FormatError::HashMismatch {
                    name: state.name.clone(),
                    expected: state.expected_sha256.clone(),
                    actual,
                }
                .into());
            }
        }
        if self.hash_layers {
            // Only what this call pulled in. `POSIX_FADV_DONTNEED` is
            // process-global — it evicts pages any other reader of the same
            // install is using — so it is issued only when this process is the
            // one that warmed the file, which is exactly the hash pass above.
            // Without the hash pass nothing here has touched the file, and
            // dropping a shared install's cache on every open would be a side
            // effect on other processes rather than a cleanup of our own.
            if let Err(error) = direct::fadvise_dontneed(&file, 0, 0) {
                tracing::debug!(path = %path.display(), %error, "could not drop the layer file's page cache");
            }
        }
        if self.direct_open && !applied {
            tracing::warn!(path = %path.display(), "O_DIRECT refused for this layer file");
        }
        tracing::debug!(path = %path.display(), bytes = size, direct = applied, "layer file open");
        self.layers[index].file = Some(file);
        Ok(())
    }

    /// Measure whether O_DIRECT reads on this install bypass the page cache.
    ///
    /// Uses a real slot as the destination, so the pre-fault that btrfs's
    /// fault-disabled direct path requires is on the measured path. The probe
    /// opens layer 0's file on its own, before that file's size and hash
    /// checks have run, and trusts nothing it reads: 4096 bytes land in a
    /// scratch slot that no cache entry owns, and only the page-cache
    /// bookkeeping around them is looked at.
    fn probe_direct_io(&mut self, strides: &[u64]) -> DirectSupport {
        if !strides.iter().all(|stride| direct::is_aligned(*stride)) {
            // Every read is `stride` bytes at `expert * stride`; if that is
            // not 4096-aligned the kernel would reject each one.
            return DirectSupport::Unusable(DirectFault::UnalignedGeometry);
        }
        let Some(path) = self.layers.first().map(|state| state.path.clone()) else {
            return DirectSupport::Unusable(DirectFault::Unsupported);
        };
        let Some(buf) = self.slots.bytes_mut(0, 0) else {
            return DirectSupport::Unusable(DirectFault::BadProbeBuffer);
        };
        direct::probe(&path, buf)
    }

    /// The mode the probe and the ring add up to.
    fn pick_mode(&self) -> StreamMode {
        #[cfg(feature = "io-uring")]
        if self.ring.is_some() {
            return if self.support.is_verified() {
                StreamMode::ODirect
            } else {
                StreamMode::Buffered
            };
        }
        StreamMode::Pread
    }

    /// Hand read `index` to the kernel, or hold it for the synchronous path.
    fn submit(&mut self, index: usize) -> Result<(), IoError> {
        #[cfg(feature = "io-uring")]
        if self.ring.is_some() {
            return self.push_sqe(index);
        }
        let _ = index;
        Ok(())
    }

    /// Flush whatever submission the ring has queued. A no-op elsewhere.
    fn flush(&mut self) -> Result<(), IoError> {
        #[cfg(feature = "io-uring")]
        if self.ring.is_some() && self.inflight.outstanding > 0 {
            self.enter(0)?;
        }
        Ok(())
    }

    /// Drive every in-flight read, or just enough of them to finish `until`.
    ///
    /// The shared body of [`ExpertStream::await_misses`] and the sweep's
    /// window wait; the two differ only in what they do with a failure, and in
    /// that the sweep stops as soon as the window it wants has landed so the
    /// windows behind it keep overlapping compute.
    ///
    /// `Err` means the submission path itself failed and nothing still in
    /// flight can ever be reaped; the caller owes those destinations whatever
    /// protection they need. A read that failed terminally is reported through
    /// `failure` instead, and is safe to clean up.
    fn drive_reads(
        &mut self,
        until: Option<usize>,
        failure: &mut Option<IoError>,
    ) -> Result<(), IoError> {
        if self.inflight.outstanding == 0 {
            return Ok(());
        }
        #[cfg(feature = "io-uring")]
        if self.ring.is_some() {
            self.await_ring(until, failure)?;
        } else {
            self.await_pread(until, failure);
        }
        #[cfg(not(feature = "io-uring"))]
        self.await_pread(until, failure);
        Ok(())
    }

    /// Whether the read the caller is waiting for has reached a terminal
    /// state. `false` for `None`, which means "wait for everything".
    fn reached(&self, until: Option<usize>) -> bool {
        until.is_some_and(|index| self.inflight.reads[index].done)
    }

    /// Drive the synchronous read path to completion.
    fn await_pread(&mut self, until: Option<usize>, failure: &mut Option<IoError>) {
        let start = Instant::now();
        for index in 0..self.inflight.reads.len() {
            if self.reached(until) {
                break;
            }
            while !self.inflight.reads[index].done {
                let result = self.pread_once(index);
                let user_data = self.inflight.reads[index].user_data;
                match self.inflight.resolve(user_data, result, &mut self.io) {
                    Reap::Done(index) => self.mark_ready(index),
                    Reap::Again(index) => {
                        self.inflight.retag(index);
                    }
                    Reap::Failed(index, error) => {
                        let error = self.fail(index, error);
                        failure.get_or_insert(error);
                    }
                    Reap::Stale => {
                        // Unreachable: the token came straight from the entry
                        // and the loop only runs while that entry is live.
                        // Resolved anyway, so a bug here cannot wedge the
                        // table into a step that can never be begun again.
                        self.io.stale_completions += 1;
                        self.inflight.abandon(index);
                        break;
                    }
                }
            }
        }
        self.io.add_wait(start.elapsed());
    }

    /// One positioned read of read `index`'s remaining bytes, in the
    /// kernel's `result` convention.
    fn pread_once(&mut self, index: usize) -> i32 {
        let read = self.inflight.reads[index];
        let Some(file) = self.layers[read.layer as usize].file.as_ref() else {
            return -libc::EBADF;
        };
        let from = read.filled as usize;
        let to = from + read.remaining as usize;
        let offset = read.base + u64::from(read.filled);
        match read.dst {
            Dest::Slot => {
                let Some(slot) = self.slots.bytes_mut(read.layer, read.slot) else {
                    return -libc::EFAULT;
                };
                let Some(dst) = slot.get_mut(from..to) else {
                    return -libc::EFAULT;
                };
                self.io.add_submit();
                pread_result(file, dst, offset)
            }
            Dest::Arena(base) => {
                let Some(arena) = self.arena else {
                    return -libc::EFAULT;
                };
                if base.checked_add(to).is_none_or(|end| end > arena.len) {
                    return -libc::EFAULT;
                }
                // SAFETY: `base + filled + remaining <= arena.len` was just
                // checked, so the window stays inside the slab the arena was
                // carved from. The arena is only `Some` while the sweep holds
                // this stream's `&mut`, which rules out every other view of
                // those bytes — no slot guard is reachable meanwhile.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        arena.base.as_ptr().add(base + from),
                        read.remaining as usize,
                    )
                };
                self.io.add_submit();
                pread_result(file, dst, offset)
            }
        }
    }

    /// A read completed: its bytes are valid and its expert is resident.
    ///
    /// A sweep window has no cache entry to promote — that is the point of the
    /// sweep — so it is only the arithmetic in [`Inflight::resolve`] that
    /// mattered for it.
    fn mark_ready(&mut self, index: usize) {
        let read = self.inflight.reads[index];
        if read.dst != Dest::Slot {
            return;
        }
        if let Some(cache) = self.caches.get_mut(read.layer as usize) {
            cache.mark_ready(read.slot);
        }
    }

    /// A read failed terminally: drop the slot's assignment and name the
    /// file in the error.
    fn fail(&mut self, index: usize, error: io::Error) -> IoError {
        let read = self.inflight.reads[index];
        tracing::error!(
            layer = read.layer,
            expert = read.expert,
            slot = read.slot,
            %error,
            "expert blob read failed"
        );
        if read.dst == Dest::Slot {
            if let Some(cache) = self.caches.get_mut(read.layer as usize) {
                // SAFETY: this read reached a terminal state, so its completion
                // has been reaped (ring) or the call has returned (pread), and
                // no reissue was made. Nothing is writing into the slot.
                unsafe { cache.invalidate(read.slot) };
            }
            // An invalidated slot is unassigned, so it is not this step's to
            // release any more.
            self.step.protected.retain(|slot| *slot != read.slot);
        }
        let path = self
            .layers
            .get(read.layer as usize)
            .map(|state| state.path.clone())
            .unwrap_or_default();
        IoError::Io {
            path,
            source: error,
        }
    }

    /// Give up every read of the step that can no longer be reaped.
    ///
    /// For the one case the retry logic cannot cover: the ring refused a
    /// submission, so whether the kernel took any of the reads already pushed
    /// is unknowable. Their slots are **retired** — the buffer is leaked, so a
    /// late kernel write into it aliases nothing, and the layer is rebuilt one
    /// slot smaller so the slot can never be handed out again. The layer loses
    /// that much cache capacity for good, which is the price of not aliasing a
    /// live DMA destination.
    ///
    /// Dropping the cache entry instead — which is what `invalidate` does —
    /// would be worse than useless: an `Empty` entry is the *first* thing
    /// `choose_victim` reaches for, so the next miss on that layer would be
    /// assigned the leaked slot, fail to find a buffer, invalidate it back to
    /// `Empty`, and be assigned it again for as long as the process lives.
    fn strand_step(&mut self) {
        if self.arena.is_some() {
            // A lost read into the arena is not one slot's problem; see
            // `strand_arena`. Unreachable in practice — an arena is out only
            // while `io::sweep` holds this stream's `&mut`, so no step exists
            // — but the two recovery paths must not be mixed up if that ever
            // changes.
            let _ = self.strand_arena();
            return;
        }
        let mut retire: Vec<(u32, u32)> = Vec::new();
        for index in 0..self.inflight.reads.len() {
            let read = self.inflight.reads[index];
            if read.done {
                continue;
            }
            tracing::error!(
                layer = read.layer,
                expert = read.expert,
                slot = read.slot,
                "expert read cannot be reaped; retiring its slot"
            );
            self.inflight.abandon(index);
            retire.push((read.layer, read.slot));
            self.step.protected.retain(|slot| *slot != read.slot);
        }
        // The step's surviving slots go back to the cache the way they were
        // planned against, before any renumbering.
        self.abandon_step();
        // Descending, because retiring a slot shifts every higher slot of that
        // layer down one.
        retire.sort_unstable_by(|a, b| b.cmp(a));
        for (layer, slot) in retire {
            self.retire_slot(layer, slot);
        }
    }

    /// Take a slot away from a layer for good.
    ///
    /// The buffer is leaked and the layer's cache is rebuilt one slot smaller.
    /// The rebuild is what makes the retirement stick: [`LayerCache`] has no
    /// terminal state for a slot, and every non-terminal one it does have is
    /// either a victim candidate (`Empty`, `Idle`) or a leak the stuck-slot
    /// detector is entitled to panic on in debug builds (`Filling`, `Ready`).
    /// A smaller cache has no entry to reach for at all.
    ///
    /// The cost is the layer's ghost history: a fresh [`LayerCache`] starts
    /// with zeroed frequency counters, so the layer re-learns its hot experts
    /// over the next few steps. That is a real regression and it is the right
    /// trade against a layer that can never serve another read. The counters
    /// it had are folded into [`ExpertStream::stats`] so the totals stay
    /// monotonic.
    ///
    /// A layer retired down to fewer than `top_k` slots reports
    /// [`CacheError::TooFewSlots`](crate::io::CacheError::TooFewSlots) on its
    /// next step — a terminal error that names the real cause, not a wedge.
    ///
    /// # Why the failure path poisons the arena
    ///
    /// By the time this runs, `strand_step` has already called
    /// [`Inflight::abandon`] on the read, so `outstanding` has drained *for a
    /// read that may still be writing*. Retiring is what makes that buffer
    /// visible again, to `SlotTable::retired_within` and so to
    /// [`take_arena`](Self::take_arena). A path out of here that retires
    /// nothing would leave the buffer abandoned-but-invisible: every
    /// `take_arena` guard would pass and the next sweep would carve a window
    /// over pages a decode read is still writing into.
    ///
    /// So the one way out that does not retire says so the only other way it
    /// can — by poisoning the arena, which is the exact permission
    /// `retired_within` would have withdrawn. The invariant then reads off this
    /// function alone: **`retire_slot` either retires the slot or refuses the
    /// process another arena.**
    ///
    /// The alternative — retire first and rebuild the cache after — makes the
    /// buffer visible unconditionally but leaves the layer's live
    /// [`LayerCache`] describing a slot numbering that no longer exists, which
    /// is silently wrong weights rather than a loud refusal. Loud wins.
    fn retire_slot(&mut self, layer: u32, slot: u32) {
        if self.slots.flat(layer, slot).is_none() {
            // Already retired — so `retired_within` already sees the buffer —
            // or it never existed and there is no buffer to see.
            return;
        }
        let usable = self.slots.usable(layer).saturating_sub(1);
        let Some(cache) = self.caches.get(layer as usize) else {
            self.poison_arena_unretired(layer, slot, "the layer has no cache");
            return;
        };
        let (carried, n_experts) = (cache.stats(), cache.n_experts());
        // Built *before* anything is renumbered: retiring the slot without
        // replacing the cache would leave the surviving cache entries pointing
        // at the wrong buffers. `usable` only shrinks and `n_experts` came from
        // a cache already built with it, so this cannot actually fail.
        let fresh = match LayerCache::new(usable, n_experts) {
            Ok(fresh) => fresh,
            Err(error) => {
                debug_assert!(false, "rebuilding a shrunken layer cache: {error}");
                let reason = error.to_string();
                self.poison_arena_unretired(layer, slot, &reason);
                return;
            }
        };
        self.slots.retire(layer, slot);
        self.caches[layer as usize] = fresh;
        self.retired = add_cache_stats(self.retired, carried);
        tracing::error!(
            layer,
            slot,
            usable,
            of = self.slots.slots_per_layer,
            "expert slot retired; the layer's cache is rebuilt smaller and \
             loses its ghost history"
        );
    }

    /// [`retire_slot`](Self::retire_slot) could not retire, so the buffer stays
    /// leased and invisible to `SlotTable::retired_within`: refuse this process
    /// another arena instead, which is the permission the retirement would have
    /// withdrawn.
    fn poison_arena_unretired(&mut self, layer: u32, slot: u32, reason: &str) {
        self.arena_poisoned = true;
        tracing::error!(
            layer,
            slot,
            reason,
            "could not retire a slot after a read that cannot be reaped; the \
             slot stays leased and the layer keeps its old capacity, so this \
             process will not sweep again"
        );
    }

    /// Close a step that failed, so a caller that propagates the error
    /// upwards does not strand the slots it protected.
    fn abandon_step(&mut self) {
        if !self.step.active {
            return;
        }
        let layer = self.step.layer;
        self.end_layer(layer);
    }

    /// Rebuild the ring if the stream has moved to another thread.
    ///
    /// `IORING_SETUP_SINGLE_ISSUER` — and `DEFER_TASKRUN`, which requires it —
    /// bind a ring to the task that created it; every `io_uring_enter` from
    /// another one answers `EEXIST`. Opening a model on the loading thread
    /// and decoding on a worker is the obvious shape, and the type is `Send`
    /// so that it is allowed, so the mismatch is repaired here instead of
    /// being a puzzling failure at the first miss. Safe only because nothing
    /// is in flight at the start of a step.
    ///
    /// A stream that keeps changing threads gets a ring without the
    /// thread-bound flags rather than a rebuild per step.
    #[cfg(feature = "io-uring")]
    fn rebind_ring(&mut self) {
        let current = std::thread::current().id();
        if self.ring.is_none() || self.ring_owner == current {
            return;
        }
        if self.inflight.outstanding > 0 {
            tracing::error!(
                outstanding = self.inflight.outstanding,
                "the driving thread changed with reads in flight; the ring \
                 cannot be rebuilt under live DMA"
            );
            return;
        }
        self.ring_rebinds += 1;
        let single_issuer = self.ring_rebinds < 2;
        tracing::info!(
            rebinds = self.ring_rebinds,
            single_issuer,
            "expert stream moved to another thread; rebuilding the ring"
        );
        // Dropped before the replacement is created so only one ring exists.
        self.ring = None;
        self.ring = build_ring(single_issuer);
        self.ring_owner = current;
        self.mode = self.pick_mode();
    }

    /// Give up the open step's reads as if the ring had failed, which is the
    /// one thing that reaches [`ExpertStream::strand_step`] without a device
    /// that can be made to fail. Test-only.
    #[cfg(test)]
    fn strand_open_step(&mut self) {
        self.strand_step();
    }

    /// Give up the open arena as if a window read had become unreapable, which
    /// needs a ring that can be made to fail. Returns the first layer left
    /// below `top_k`, as [`ExpertStream::strand_arena`] does. Test-only.
    #[cfg(test)]
    pub(super) fn strand_open_arena(&mut self) -> Option<(u32, u32)> {
        self.strand_arena()
    }

    /// Turn a read failure into the error a lost window read would report,
    /// without a device that can be made to fail. Test-only.
    #[cfg(test)]
    pub(super) fn strand_open_arena_error(&mut self, source: IoError) -> SweepError {
        self.strand_arena_error(source)
    }

    /// Replace the slot pool with a padded one, so the arena's
    /// gapless-slab contract can be exercised.
    ///
    /// Every real layout the runtime targets is 4096-aligned throughout — both
    /// shipped Qwen3 strides are — so a padded pool cannot be built from a
    /// fixture install; it is substituted instead. The stream cannot serve a
    /// read afterwards, which is fine for the one thing this is for.
    /// Test-only.
    #[cfg(test)]
    fn force_padded_slots(&mut self) {
        self.slots = SlotTable::new(2, &[4096, 5000]).expect("a tiny pool builds");
    }

    /// Answer the next `count` submits the way `enter` answers `EBUSY`:
    /// submitted nothing, reap and come back. Test-only.
    ///
    /// Real `EBUSY` needs a kernel that refuses to submit while completions
    /// are overflowed. Since 6.x the overflow list is unbounded and the
    /// refusal is gone — measured on this machine, a one-entry ring took six
    /// no-ops with the completion queue full throughout and answered `Ok` to
    /// every submit — but `EBUSY` is still in `io_uring_enter(2)`'s contract
    /// and still reachable on the older kernels this runtime supports. So the
    /// answer is injected rather than provoked.
    #[cfg(all(test, feature = "io-uring"))]
    fn stall_enters(&mut self, count: u32) {
        self.stalled_enters = count;
    }

    /// Leave the completion queue full, with entries spilled into the kernel's
    /// overflow list — the state `EBUSY` used to be reported for, and the one
    /// the drain has to cope with either way. No-ops are used because they
    /// complete without touching a slot, and their completions read as stale.
    /// Test-only.
    #[cfg(all(test, feature = "io-uring"))]
    fn force_cq_backlog(&mut self, count: usize) {
        let ring = self.ring.as_mut().expect("ring");
        for i in 0..count {
            let nop = opcode::Nop::new()
                .build()
                .user_data(0xdead_beef_0000_0000 | i as u64);
            // SAFETY: a no-op SQE borrows nothing at all.
            while unsafe { ring.submission().push(&nop) }.is_err() {
                let _ = ring.submitter().submit();
            }
            let _ = ring.submitter().submit();
        }
        // The kernel completes a no-op immediately, but "immediately" is still
        // asynchronous.
        std::thread::sleep(Duration::from_millis(5));
    }

    /// Replace the ring with the smallest one the kernel will build, so that
    /// a step with more misses than submission slots exercises the
    /// queue-full path. Test-only.
    #[cfg(all(test, feature = "io-uring"))]
    fn force_tiny_ring(&mut self) {
        self.ring = None;
        self.ring = IoUring::builder().build(1).ok();
        self.ring_owner = std::thread::current().id();
        assert!(self.ring.is_some(), "a one-entry ring should always build");
    }

    /// Push read `index` as a `Read` SQE, submitting first if the queue is
    /// full.
    #[cfg(feature = "io-uring")]
    fn push_sqe(&mut self, index: usize) -> Result<(), IoError> {
        let read = self.inflight.reads[index];
        let Some(file) = self.layers[read.layer as usize].file.as_ref() else {
            return Err(self.protocol_error(read.layer, "layer file is not open"));
        };
        let fd = file.as_raw_fd();
        let base = match read.dst {
            // SAFETY: the read is parked in the in-flight table until its
            // completion is reaped, the guard behind this pointer is held for
            // the life of the stream, no other read targets this slot (the
            // cache keeps it `Filling`), and no slice view of it is taken
            // meanwhile: `view` refuses a slot that is not ready. `Drop` leaks
            // the slot if a teardown cannot reap.
            Dest::Slot => match unsafe { self.slots.write_ptr(read.layer, read.slot) } {
                Some(base) => base,
                None => return Err(self.protocol_error(read.layer, "slot is not in the pool")),
            },
            Dest::Arena(offset) => {
                let Some(arena) = self.arena else {
                    return Err(self.protocol_error(read.layer, "no prefill arena is out"));
                };
                if offset
                    .checked_add(read.filled as usize + read.remaining as usize)
                    .is_none_or(|end| end > arena.len)
                {
                    return Err(self.protocol_error(read.layer, "window is outside the arena"));
                }
                // SAFETY: bounds-checked above, so the window stays inside the
                // slab. The arena exists only while the sweep holds this
                // stream's `&mut`, so no slot guard is reachable meanwhile and
                // no second read can target these bytes; `strand_arena` retires
                // every slot the arena covers if a completion is ever lost.
                unsafe { arena.base.as_ptr().add(offset) }
            }
        };
        // SAFETY: `filled + remaining` is the length checked against the slot
        // stride or the arena length above, so this stays inside it.
        let dst = unsafe { base.add(read.filled as usize) };
        debug_assert!(
            read.dst == Dest::Slot
                || (dst as usize as u64 % direct::DIO_ALIGN == 0
                    && (read.base + u64::from(read.filled)) % direct::DIO_ALIGN == 0
                    && u64::from(read.remaining) % direct::DIO_ALIGN == 0),
            "a sweep read must be 4096-aligned in offset, length and destination"
        );
        let entry = opcode::Read::new(types::Fd(fd), dst, read.remaining)
            .offset(read.base + u64::from(read.filled))
            .build()
            .user_data(read.user_data);

        for _ in 0..MAX_PUSH_ATTEMPTS {
            let pushed = match self.ring.as_mut() {
                // SAFETY: the destination stays alive and untouched until the
                // completion is reaped, as argued above; `entry` is a plain
                // `Read` with no borrowed state beyond that pointer.
                Some(ring) => unsafe { ring.submission().push(&entry) }.is_ok(),
                None => return Err(self.protocol_error(read.layer, "the io_uring ring is gone")),
            };
            if pushed {
                self.io.add_submit();
                return Ok(());
            }
            // A full SQ is a submit-and-retry, never a dropped read. The
            // submit may answer `EBUSY`, which `enter` reports as "submitted
            // nothing" and which means the *completion* queue needs draining
            // before the kernel will take anything else — so drain it and push
            // again, rather than calling a transient CQ overflow a fatal step
            // failure.
            if self.enter(0)? == 0 {
                self.drain_cq();
            }
        }
        Err(self.protocol_error(
            read.layer,
            "submission queue stayed full after submitting and draining completions",
        ))
    }

    /// Move every completion the ring has ready into `self.completions`.
    ///
    /// Separate from acting on them so that a reissue can reap without the
    /// completion queue borrowed, and so that a `push_sqe` blocked on a full
    /// SQ can make room without losing what it drained.
    #[cfg(feature = "io-uring")]
    fn drain_cq(&mut self) {
        let Some(ring) = self.ring.as_mut() else {
            return;
        };
        let sink = &mut self.completions;
        let mut cq = ring.completion();
        cq.sync();
        for cqe in &mut cq {
            sink.push((cqe.user_data(), cqe.result()));
        }
    }

    /// `io_uring_enter`, retrying `EINTR` rather than propagating it.
    ///
    /// `EBUSY` means the completion queue needs draining first; it is
    /// reported as "submitted nothing" so the caller reaps and comes back.
    /// Both callers honour that: [`ExpertStream::await_ring`] reaps on its
    /// next pass, [`ExpertStream::push_sqe`] reaps before it pushes again.
    #[cfg(feature = "io-uring")]
    fn enter(&mut self, want: usize) -> Result<usize, IoError> {
        #[cfg(test)]
        if self.stalled_enters > 0 {
            self.stalled_enters -= 1;
            return Ok(0);
        }
        let Some(ring) = self.ring.as_ref() else {
            return Err(self.protocol_error(self.step.layer, "the io_uring ring is gone"));
        };
        let mut interrupted = 0u32;
        loop {
            match ring.submitter().submit_and_wait(want) {
                Ok(submitted) => return Ok(submitted),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                    // Bounded: a signal every caller is expected to ignore
                    // must not become a decode thread that never returns.
                    interrupted += 1;
                    if interrupted > MAX_EINTR_RETRIES {
                        tracing::error!(
                            retries = interrupted,
                            "io_uring_enter interrupted on every attempt"
                        );
                        return Err(self.layer_error(error));
                    }
                    continue;
                }
                Err(error) if error.raw_os_error() == Some(libc::EBUSY) => return Ok(0),
                Err(error) if error.raw_os_error() == Some(libc::EEXIST) => {
                    // A thread-bound ring reached from another task. The step
                    // boundary repairs this (see `rebind_ring`); mid-step it
                    // cannot be, because reads are already in flight.
                    tracing::error!(
                        "io_uring rejected a submission from a thread other than \
                         the one that created the ring; one thread must drive a \
                         step from begin_layer to end_layer"
                    );
                    return Err(self.layer_error(error));
                }
                Err(error) => {
                    return Err(self.layer_error(error));
                }
            }
        }
    }

    /// Wait for every submitted read, reissuing short reads and one retry per
    /// failed blob.
    #[cfg(feature = "io-uring")]
    fn await_ring(
        &mut self,
        until: Option<usize>,
        failure: &mut Option<IoError>,
    ) -> Result<(), IoError> {
        let mut idle = 0;
        while self.inflight.outstanding > 0 && !self.reached(until) {
            // `push_sqe` may already have reaped some of what is owed while
            // making room in the submission queue; waiting on the ring before
            // acting on those would block for completions that have already
            // arrived.
            if self.completions.is_empty() {
                let start = Instant::now();
                self.enter(1)?;
                self.io.add_wait(start.elapsed());
                self.drain_cq();
            }
            if self.completions.is_empty() {
                idle += 1;
                if idle > MAX_IDLE_PASSES {
                    return Err(self.protocol_error(
                        self.step.layer,
                        "no completions after repeated waits; the ring is wedged",
                    ));
                }
                continue;
            }
            idle = 0;
            // Swapped out of the sink, because reissuing a read below may
            // drain the ring again and must not append to the batch being
            // walked.
            std::mem::swap(&mut self.completions, &mut self.reaping);
            for i in 0..self.reaping.len() {
                let (user_data, result) = self.reaping[i];
                match self.inflight.resolve(user_data, result, &mut self.io) {
                    Reap::Done(index) => self.mark_ready(index),
                    Reap::Again(index) => {
                        self.inflight.retag(index);
                        if let Err(error) = self.push_sqe(index) {
                            // The read cannot be reissued, so it will never
                            // complete: resolve it here or `outstanding`
                            // never drains. Its completion *was* reaped, so
                            // the slot is safe to invalidate.
                            self.inflight.abandon(index);
                            let error = self.fail(index, flatten_io(error));
                            failure.get_or_insert(error);
                        }
                    }
                    Reap::Failed(index, error) => {
                        let error = self.fail(index, error);
                        failure.get_or_insert(error);
                    }
                    Reap::Stale => {
                        self.io.stale_completions += 1;
                        tracing::error!(
                            user_data,
                            result,
                            "stale io_uring completion: no live read carries this tag"
                        );
                    }
                }
            }
            self.reaping.clear();
            self.flush()?;
        }
        Ok(())
    }
}

impl Drop for ExpertStream {
    fn drop(&mut self) {
        if self.inflight.outstanding == 0 {
            return;
        }
        // An unwind tore the stream down mid-read. The kernel may still write
        // into those buffers after the ring's descriptor closes, so their
        // leases are given up rather than returned: `SlotPool::drop` then
        // leaks the slab instead of freeing memory under a live DMA.
        tracing::error!(
            outstanding = self.inflight.outstanding,
            "expert stream dropped with reads in flight; leaking their slots"
        );
        // A window read lands somewhere in the arena rather than in one slot,
        // so every slot the arena covers is given up.
        let arena = self.arena_len();
        let sweeping = arena > 0
            && self
                .inflight
                .reads
                .iter()
                .any(|read| !read.done && matches!(read.dst, Dest::Arena(_)));
        // Descending, so retiring one slot does not renumber the next.
        let mut stranded: Vec<(u32, u32)> = self
            .inflight
            .reads
            .iter()
            .filter(|read| !read.done && read.dst == Dest::Slot)
            .map(|read| (read.layer, read.slot))
            .collect();
        if sweeping {
            for layer in 0..self.slots.live.len() as u32 {
                for slot in 0..self.slots.usable(layer) {
                    if self
                        .slots
                        .slot_offset(layer, slot)
                        .is_some_and(|offset| offset < arena)
                    {
                        stranded.push((layer, slot));
                    }
                }
            }
        }
        stranded.sort_unstable_by(|a, b| b.cmp(a));
        stranded.dedup();
        for (layer, slot) in stranded {
            self.slots.retire(layer, slot);
        }
    }
}

/// Unwrap an [`IoError`] back into the `io::Error` it carries, so a
/// submission failure that is about to be re-reported as a read failure keeps
/// its errno.
///
/// The path context is dropped, not the cause: [`ExpertStream::fail`] puts the
/// layer file's path back on immediately. Flattening through `to_string`
/// instead — which is what this replaced — turned every errno into a message,
/// so a caller could no longer tell `ENOSPC` from `EEXIST` in a failure that
/// still had one.
fn flatten_io(error: IoError) -> io::Error {
    match error {
        IoError::Io { source, .. } => source,
        other => io::Error::other(other.to_string()),
    }
}

/// One positioned read, in the kernel's `result` convention: bytes
/// transferred, `0` for EOF, or a negative errno.
fn pread_result(file: &File, dst: &mut [u8], offset: u64) -> i32 {
    match file.read_at(dst, offset) {
        Ok(got) => i32::try_from(got).unwrap_or(i32::MAX),
        Err(error) => -error.raw_os_error().unwrap_or(libc::EIO),
    }
}

/// Cache counters added together.
///
/// [`CacheStats`] is a plain counter bag with no arithmetic of its own, and
/// this file only ever needs the one operation.
fn add_cache_stats(a: CacheStats, b: CacheStats) -> CacheStats {
    CacheStats {
        hits: a.hits + b.hits,
        pending_hits: a.pending_hits + b.pending_hits,
        misses: a.misses + b.misses,
        cold_misses: a.cold_misses + b.cold_misses,
        eviction_misses: a.eviction_misses + b.eviction_misses,
        evictions: a.evictions + b.evictions,
    }
}

/// Completions one pass of the reap loop can hold.
///
/// A completion exists only for a submitted read, and a step submits at most
/// one read per slot, so the live count is bounded by the slots of one layer —
/// doubled, because a retagged read can leave the completion of the attempt it
/// replaced behind. Preallocated to that so the decode loop never allocates
/// inside the completion path, however badly the completion queue overflows.
#[cfg(feature = "io-uring")]
fn completion_capacity(slots_per_layer: u32) -> usize {
    (2 * RING_ENTRIES as usize).max(2 * slots_per_layer as usize)
}

/// Slots per layer bought by a total byte budget.
///
/// The pool charges each layer the page-aligned pitch of its own stride, so a
/// slot costs the sum of those across all layers. Clamped to what a layer can
/// use (its expert count) and to [`MAX_SLOTS`], and floored at `top_k`, which
/// is the smallest number of slots any step of this model can be served from.
fn slots_for_budget(
    cache_bytes: u64,
    strides: &[u64],
    layers: &[LayerState],
    top_k: u32,
) -> Result<u32, IoError> {
    if strides.is_empty() {
        return Err(SlotError::NoLayers.into());
    }
    let mut per_slot: u128 = 0;
    for &stride in strides {
        let pitch =
            stride
                .checked_next_multiple_of(SLOT_ALIGN as u64)
                .ok_or(IoError::TooLarge {
                    what: "expert blob stride",
                    value: stride,
                })?;
        per_slot += u128::from(pitch);
    }
    if per_slot == 0 {
        return Err(SlotError::ZeroStride { layer: 0 }.into());
    }
    let slots = u32::try_from(u128::from(cache_bytes) / per_slot).unwrap_or(u32::MAX);
    if slots == 0 {
        tracing::error!(
            cache_bytes,
            bytes_per_slot = %per_slot,
            n_layers = strides.len(),
            "expert cache budget cannot fit one slot per layer"
        );
        return Err(SlotError::ZeroSlotsPerLayer.into());
    }
    let usable = layers
        .iter()
        .map(|layer| layer.n_experts)
        .min()
        .unwrap_or(MAX_SLOTS)
        .clamp(1, MAX_SLOTS);
    let chosen = if slots > usable {
        tracing::info!(
            budget_slots = slots,
            slots_per_layer = usable,
            "expert cache budget buys more slots than a layer can use; clamping"
        );
        usable
    } else {
        slots
    };

    // One step routes `top_k` distinct experts and needs all of them resident
    // at once, so a layer with fewer slots than that cannot serve a single
    // forward pass. Caught here rather than mid-pass: by then the pool has
    // been allocated and pre-faulted, the layer files are open and the
    // tokenizer is up, and the error the cache raises names slots rather than
    // the dial the operator actually set.
    let floor = top_k.max(1);
    if chosen < floor {
        let needed = per_slot * u128::from(floor);
        tracing::error!(
            cache_bytes,
            bytes_per_slot = %per_slot,
            slots = chosen,
            top_k = floor,
            minimum_bytes = %needed,
            "expert cache budget cannot fit one slot per routed expert"
        );
        return Err(IoError::CacheBudgetTooSmall {
            given: cache_bytes,
            // Saturating: a model whose `top_k` slots do not fit in 64 bits of
            // budget cannot be run at all, and the number is for a message.
            needed: u64::try_from(needed).unwrap_or(u64::MAX),
            slots: chosen,
            top_k: floor,
        });
    }
    Ok(chosen)
}

/// Where `expert` first appears in the request the step was planned from.
///
/// `LayerCache::plan` resolves a duplicated id once, at its first position,
/// so this cannot fail for an id the plan reported — but it is program state,
/// not untrusted input, and a stray id maps to 0 rather than panicking.
fn position_of(experts: &[u32], expert: u32) -> usize {
    experts.iter().position(|&id| id == expert).unwrap_or(0)
}

/// Create the ring, dropping setup flags the kernel refuses.
///
/// The flag matrix is a kernel-version matrix — `defer_taskrun` wants 6.1 and
/// `single_issuer`, `single_issuer` wants 6.0, `coop_taskrun` and
/// `taskrun_flag` want 5.19 — and `build()` answers `EINVAL` for anything the
/// running kernel does not know. Since expert streaming is default-on, an
/// older kernel gets a plainer ring rather than a failed load. No SQPOLL (it
/// burns a core the GEMVs need) and no IOPOLL (`EOPNOTSUPP` on btrfs anyway).
///
/// `single_issuer` also decides whether the two thread-bound levels are tried
/// at all: see [`ExpertStream::rebind_ring`].
#[cfg(feature = "io-uring")]
fn build_ring(single_issuer: bool) -> Option<IoUring> {
    let first = if single_issuer { 0 } else { 2 };
    for level in first..4 {
        let mut builder = IoUring::builder();
        let flags = match level {
            0 => {
                builder.setup_single_issuer().setup_defer_taskrun();
                "single_issuer+defer_taskrun"
            }
            1 => {
                builder
                    .setup_single_issuer()
                    .setup_coop_taskrun()
                    .setup_taskrun_flag();
                "single_issuer+coop_taskrun+taskrun_flag"
            }
            2 => {
                builder.setup_coop_taskrun().setup_taskrun_flag();
                "coop_taskrun+taskrun_flag"
            }
            _ => "none",
        };
        match builder.build(RING_ENTRIES) {
            Ok(ring) => {
                tracing::info!(entries = RING_ENTRIES, flags, "io_uring ready");
                return Some(ring);
            }
            Err(error) => {
                tracing::debug!(flags, %error, "io_uring setup refused, dropping flags");
            }
        }
    }
    tracing::warn!("io_uring unavailable; expert reads fall back to pread");
    None
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Fixture, build_install};
    use super::*;
    use crate::io::CacheError;

    /// Bytes one slot per layer costs across the whole fixture.
    fn slot_row(fx: &Fixture) -> u64 {
        fx.layout
            .layers
            .iter()
            .map(|layer| layer.stride.next_multiple_of(SLOT_ALIGN as u64))
            .sum()
    }

    fn open(fx: &Fixture, slots: u32) -> ExpertStream {
        open_with(fx, slots, LoadOptions::default()).expect("stream opens")
    }

    fn open_with(fx: &Fixture, slots: u32, options: LoadOptions) -> Result<ExpertStream, IoError> {
        ExpertStream::new(
            &fx.root,
            &fx.manifest,
            &fx.layout,
            slot_row(fx) * u64::from(slots),
            options,
        )
    }

    /// The blob of `expert` as it is on disk.
    fn blob_on_disk(fx: &Fixture, layer: u32, expert: u32) -> Vec<u8> {
        let layout = &fx.layout.layers[layer as usize];
        let bytes = std::fs::read(fx.root.join(&layout.file)).unwrap();
        let base = (u64::from(expert) * layout.stride) as usize;
        bytes[base..base + layout.stride as usize].to_vec()
    }

    #[test]
    fn stream_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ExpertStream>();
    }

    #[test]
    fn budget_dial_converts_bytes_to_slots() {
        let fx = build_install("stream-budget");
        let row = slot_row(&fx);

        // Exactly n rows buys n slots; a fraction over buys no more. Starts at
        // `top_k`, which is the floor `new` enforces.
        for slots in fx.manifest.arch.top_k..=3u32 {
            let stream = open(&fx, slots);
            assert_eq!(stream.slots_per_layer(), slots);
            let extra = ExpertStream::new(
                &fx.root,
                &fx.manifest,
                &fx.layout,
                row * u64::from(slots) + row / 2,
                LoadOptions::default(),
            )
            .unwrap();
            assert_eq!(extra.slots_per_layer(), slots);
            // The pool is what the slots actually cost, never more than asked.
            assert_eq!(stream.cache_bytes(), row * u64::from(slots));
            assert!(stream.cache_bytes() <= row * u64::from(slots));
        }

        // A budget short of one slot per layer is an error, not a rounding.
        let err = ExpertStream::new(
            &fx.root,
            &fx.manifest,
            &fx.layout,
            row - 1,
            LoadOptions::default(),
        )
        .unwrap_err();
        assert!(
            matches!(err, IoError::Slots(SlotError::ZeroSlotsPerLayer)),
            "unexpected error: {err}"
        );

        // More slots than a layer has experts is capped: they could never
        // hold anything.
        let huge = open(&fx, 64);
        assert_eq!(huge.slots_per_layer(), fx.layout.layers[0].n_experts);
    }

    #[test]
    fn a_budget_below_top_k_is_rejected_at_construction() {
        // `--cache-bytes 700M` on the shipped model buys 5 slots against a
        // top_k of 8, and used to load the whole model, allocate and pre-fault
        // the pool and start the tokenizer before dying in the first forward
        // pass with an error about slots. The dial the operator set is the one
        // named, and it is named before any of that work happens.
        let fx = build_install("stream-budget-floor");
        let row = slot_row(&fx);
        let top_k = fx.manifest.arch.top_k;
        assert!(top_k > 1, "fixture must route more than one expert");

        for slots in 1..top_k {
            let err = ExpertStream::new(
                &fx.root,
                &fx.manifest,
                &fx.layout,
                row * u64::from(slots),
                LoadOptions::default(),
            )
            .unwrap_err();
            match err {
                IoError::CacheBudgetTooSmall {
                    given,
                    needed,
                    slots: bought,
                    top_k: k,
                } => {
                    assert_eq!(given, row * u64::from(slots));
                    assert_eq!(bought, slots);
                    assert_eq!(k, top_k);
                    // The minimum it names has to be a budget that works.
                    assert_eq!(needed, row * u64::from(top_k));
                    assert!(
                        err.to_string().contains(&needed.to_string()),
                        "the error must name the minimum budget: {err}"
                    );
                }
                other => panic!("unexpected error: {other}"),
            }
        }

        // And the floor itself opens.
        let ok = open(&fx, top_k);
        assert_eq!(ok.slots_per_layer(), top_k);

        // A budget under one whole slot is still the zero-slot error: there is
        // no meaningful "minimum for top_k" to quote when the pool cannot hold
        // a single row.
        let err = ExpertStream::new(
            &fx.root,
            &fx.manifest,
            &fx.layout,
            row - 1,
            LoadOptions::default(),
        )
        .unwrap_err();
        assert!(
            matches!(err, IoError::Slots(SlotError::ZeroSlotsPerLayer)),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn miss_reads_the_blob_and_a_second_step_hits() {
        let fx = build_install("stream-miss-hit");
        let mut stream = open(&fx, 2);
        let request = [1u32, 3];

        stream.begin_layer(0, &request).unwrap();
        assert!(stream.hits().is_empty(), "cold cache reported a hit");
        let misses = stream.misses().to_vec();
        assert_eq!(misses.len(), 2);
        stream.await_misses().unwrap();

        for &(index, slot) in &misses {
            let expert = request[index];
            let view = stream.view(0, slot).unwrap();
            let expected = blob_on_disk(&fx, 0, expert);
            assert_eq!(view.blob(), expected, "expert {expert} blob mismatch");
            // And the slabs land where the layout says they do.
            for projection in &fx.layout.layers[0].projections {
                let from = projection.offset_in_blob as usize;
                let to = from + projection.len as usize;
                assert_eq!(
                    view.slab(projection.name).bytes,
                    &expected[from..to],
                    "expert {expert} {:?} slab",
                    projection.name
                );
            }
        }
        stream.end_layer(0);

        let after_first = stream.stats();
        assert_eq!(after_first.misses, 2);
        assert_eq!(after_first.cold_misses, 2);
        assert_eq!(
            after_first.bytes_read,
            2 * fx.layout.layers[0].stride,
            "read more or fewer bytes than the two blobs"
        );
        assert_eq!(after_first.reads_submitted, 2);
        assert_eq!(after_first.read_retries, 0);

        // Same experts again: resident, so no read at all.
        stream.begin_layer(0, &request).unwrap();
        assert_eq!(stream.misses().len(), 0, "a resident expert was re-read");
        assert_eq!(stream.hits().len(), 2);
        stream.await_misses().unwrap();
        for &(index, slot) in stream.hits().to_vec().iter() {
            let view = stream.view(0, slot).unwrap();
            assert_eq!(view.blob(), blob_on_disk(&fx, 0, request[index]));
        }
        stream.end_layer(0);

        let after_second = stream.stats();
        assert_eq!(after_second.hits, 2);
        assert_eq!(after_second.misses, 2, "a hit was counted as a miss");
        assert_eq!(after_second.bytes_read, after_first.bytes_read);
        assert_eq!(after_second.reads_submitted, after_first.reads_submitted);
    }

    #[test]
    fn every_slot_of_a_batch_gets_its_own_blob() {
        // Four misses in one step: completions land in whatever order the
        // kernel gives them, so this is the end-to-end form of "out-of-order
        // completions map to the right slots".
        let fx = build_install("stream-batch");
        let mut stream = open(&fx, 4);
        let request = [3u32, 0, 2, 1];
        stream.begin_layer(1, &request).unwrap();
        assert_eq!(stream.misses().len(), 4);
        let misses = stream.misses().to_vec();
        stream.await_misses().unwrap();

        let mut slots_seen = Vec::new();
        for &(index, slot) in &misses {
            let expert = request[index];
            assert_eq!(
                stream.view(1, slot).unwrap().blob(),
                blob_on_disk(&fx, 1, expert)
            );
            slots_seen.push(slot);
        }
        slots_seen.sort_unstable();
        slots_seen.dedup();
        assert_eq!(slots_seen.len(), 4, "two experts shared a slot");
        stream.end_layer(1);
    }

    #[cfg(feature = "io-uring")]
    #[test]
    fn a_step_wider_than_the_submission_queue_still_completes() {
        // A full SQ is a submit-then-retry, never a dropped read. One
        // submission slot and two completion slots against four misses makes
        // the queue full on every push after the first, and overflows the
        // completion queue too.
        let fx = build_install("stream-tiny-ring");
        let mut stream = open(&fx, 4);
        stream.force_tiny_ring();
        let request = [0u32, 1, 2, 3];
        stream.begin_layer(1, &request).unwrap();
        assert_eq!(stream.misses().len(), 4);
        let misses = stream.misses().to_vec();
        stream.await_misses().unwrap();
        for &(index, slot) in &misses {
            assert_eq!(
                stream.view(1, slot).unwrap().blob(),
                blob_on_disk(&fx, 1, request[index])
            );
        }
        stream.end_layer(1);
        assert_eq!(stream.stats().reads_submitted, 4);
        assert_eq!(stream.stale_completions(), 0);
    }

    #[cfg(feature = "io-uring")]
    #[test]
    fn a_full_completion_queue_is_drained_rather_than_reported_as_fatal() {
        // `EBUSY` from `io_uring_enter` means "reap the completion queue and
        // come back", which is what the call's own doc says and what
        // `await_ring` honours. `push_sqe` used to answer it by retrying the
        // push against a submission queue the kernel had just refused to
        // drain, and calling the second failure fatal — so a transient
        // completion-queue overflow killed a forward pass.
        //
        // The backlog below leaves the CQ full with entries spilled into the
        // overflow list, which is exactly the state that makes every
        // `io_uring_enter` answer `EBUSY`, and a one-entry ring makes the SQ
        // full on every push after the first.
        const BACKLOG: usize = 6;
        const STALLS: u32 = 3;
        let fx = build_install("stream-cq-overflow");
        let mut stream = open(&fx, 4);
        stream.force_tiny_ring();
        stream.force_cq_backlog(BACKLOG);
        stream.stall_enters(STALLS);

        let request = [0u32, 1, 2, 3];
        stream.begin_layer(1, &request).expect("step submits");
        assert_eq!(stream.misses().len(), 4);
        let misses = stream.misses().to_vec();
        stream.await_misses().expect("step completes");
        for &(index, slot) in &misses {
            assert_eq!(
                stream.view(1, slot).unwrap().blob(),
                blob_on_disk(&fx, 1, request[index]),
                "expert {} came back wrong through a full CQ",
                request[index]
            );
        }
        stream.end_layer(1);
        assert_eq!(stream.stats().reads_submitted, 4);
        // Every no-op of the backlog was reaped and recognised as naming no
        // live read, which is what draining the CQ costs and all it costs.
        assert_eq!(stream.stale_completions(), BACKLOG as u64);

        // The layer is intact: nothing was retired and the next step hits.
        assert_eq!(stream.usable_slots(1), 4);
        stream.begin_layer(1, &request).unwrap();
        assert_eq!(stream.hits().len(), 4);
        stream.await_misses().unwrap();
        stream.end_layer(1);
    }

    #[test]
    fn a_retired_slot_is_never_handed_out_again() {
        // The failure this guards: a slot whose read can never be reaped used
        // to be leaked *and* invalidated, and `choose_victim` prefers `Empty`
        // slots. So the next miss on that layer was assigned the leaked slot,
        // found no buffer, invalidated it back to `Empty`, and was assigned it
        // again — every `begin_layer` on that layer failing for the life of
        // the process.
        let fx = build_install("stream-retire");
        let mut stream = open(&fx, 4);
        let before = stream.usable_slots(0);
        assert_eq!(before, 4);

        stream.begin_layer(0, &[3]).unwrap();
        assert_eq!(stream.misses().len(), 1);
        stream.strand_open_step();

        // The layer lost exactly that much capacity, and only that layer.
        assert_eq!(stream.usable_slots(0), before - 1);
        assert_eq!(stream.usable_slots(1), before);

        // And it keeps working on what is left, for as long as it is driven.
        for step in 0..64u32 {
            let request = [step % 4, (step / 4 + 1) % 4];
            let request: Vec<u32> = if request[0] == request[1] {
                vec![request[0]]
            } else {
                request.to_vec()
            };
            stream
                .begin_layer(0, &request)
                .unwrap_or_else(|e| panic!("step {step} refused after a retirement: {e}"));
            stream.await_misses().unwrap();
            for &(index, slot) in stream.misses().to_vec().iter() {
                assert_eq!(
                    stream.view(0, slot).unwrap().blob(),
                    blob_on_disk(&fx, 0, request[index]),
                    "step {step} served the wrong blob after a retirement"
                );
            }
            for &(index, slot) in stream.hits().to_vec().iter() {
                assert_eq!(
                    stream.view(0, slot).unwrap().blob(),
                    blob_on_disk(&fx, 0, request[index]),
                    "step {step} hit the wrong blob after a retirement"
                );
            }
            stream.end_layer(0);
        }
        // The counters the rebuilt cache lost are still in the totals.
        let stats = stream.stats();
        assert!(stats.misses >= 1, "the stranded step's miss was dropped");
        assert_eq!(
            stats.hits + stats.pending_hits + stats.misses,
            1 + 64 * 2 - 16
        );
    }

    #[test]
    fn retiring_a_layer_down_to_nothing_is_a_typed_error_not_a_wedge() {
        // Retirement shrinks a layer, and a layer below `top_k` slots can no
        // longer serve a step. It has to say so, once, in terms of slots —
        // never fail every read forever with "slot is not in the pool".
        let fx = build_install("stream-retire-all");
        let mut stream = open(&fx, 4);
        for expert in 0..4u32 {
            if stream.usable_slots(0) == 0 {
                break;
            }
            stream.begin_layer(0, &[expert]).unwrap();
            stream.strand_open_step();
        }
        assert_eq!(stream.usable_slots(0), 0);
        let err = stream.begin_layer(0, &[0]).unwrap_err();
        assert!(
            matches!(
                err,
                IoError::Cache(CacheError::TooFewSlots { n_slots: 0, .. })
            ),
            "unexpected error: {err}"
        );
        // Twice, so the state is stable rather than degrading further.
        assert!(matches!(
            stream.begin_layer(0, &[0]).unwrap_err(),
            IoError::Cache(CacheError::TooFewSlots { n_slots: 0, .. })
        ));
        // The other layer is untouched.
        stream.begin_layer(1, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(1);
    }

    #[test]
    fn end_layer_without_awaiting_retires_rather_than_wedging() {
        // Skipping `await_misses` used to leave the in-flight table populated
        // for good, and `begin_layer` refuses to start a step over live reads
        // — so every later step failed with no path out. Those reads are
        // genuinely unreapable, so their slots are retired and the stream
        // keeps going one slot lighter.
        let fx = build_install("stream-end-without-await");
        let mut stream = open(&fx, 4);
        stream.begin_layer(1, &[2]).unwrap();
        stream.end_layer(1); // no await_misses

        assert_eq!(stream.usable_slots(1), 3);
        for expert in [0u32, 1, 3] {
            stream
                .begin_layer(1, &[expert])
                .expect("layer still serves");
            stream.await_misses().unwrap();
            let slot = stream.misses()[0].1;
            assert_eq!(
                stream.view(1, slot).unwrap().blob(),
                blob_on_disk(&fx, 1, expert)
            );
            stream.end_layer(1);
        }
    }

    #[test]
    fn phase_split_separates_prefill_from_decode() {
        let fx = build_install("stream-phases");
        let mut stream = open(&fx, 4);
        assert_eq!(stream.phase(), StreamPhase::Prefill);

        // Prefill: two cold misses.
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);

        let prefill = stream.stats_in(StreamPhase::Prefill);
        assert_eq!(prefill.misses, 2);
        assert_eq!(prefill.cold_misses, 2);
        assert_eq!(prefill.hits, 0);
        assert_eq!(stream.stats_in(StreamPhase::Decode), StreamStats::default());

        stream.set_phase(StreamPhase::Decode);
        assert_eq!(stream.phase(), StreamPhase::Decode);
        // Idempotent.
        stream.set_phase(StreamPhase::Decode);

        // Decode: the same two experts are resident, so both hit.
        for _ in 0..3 {
            stream.begin_layer(0, &[0, 1]).unwrap();
            stream.await_misses().unwrap();
            stream.end_layer(0);
        }

        let decode = stream.stats_in(StreamPhase::Decode);
        assert_eq!(decode.hits, 6);
        assert_eq!(decode.misses, 0);
        assert_eq!(decode.hit_rate(), 1.0);
        assert_eq!(decode.bytes_read, 0, "decode read a resident expert");

        // Prefill's share did not move, and the two add up to the whole.
        assert_eq!(stream.stats_in(StreamPhase::Prefill), prefill);
        let total = stream.stats();
        assert_eq!(prefill.plus(&decode), total);
        // The cumulative rate understates decode, which is the whole point.
        assert!(
            total.hit_rate() < decode.hit_rate(),
            "cumulative {} vs decode {}",
            total.hit_rate(),
            decode.hit_rate()
        );

        // Going back to prefill accumulates rather than restarts.
        stream.set_phase(StreamPhase::Prefill);
        stream.begin_layer(1, &[2]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(1);
        assert_eq!(stream.stats_in(StreamPhase::Prefill).misses, 3);
        assert_eq!(stream.stats_in(StreamPhase::Decode), decode);
    }

    #[test]
    fn layers_are_independent() {
        let fx = build_install("stream-layers");
        let mut stream = open(&fx, 2);
        for layer in 0..2u32 {
            stream.begin_layer(layer, &[0, 1]).unwrap();
            stream.await_misses().unwrap();
            for &(index, slot) in stream.misses().to_vec().iter() {
                let expert = [0u32, 1][index];
                assert_eq!(
                    stream.view(layer, slot).unwrap().blob(),
                    blob_on_disk(&fx, layer, expert)
                );
            }
            stream.end_layer(layer);
        }
        // Layer 0's cache is untouched by layer 1's traffic.
        stream.begin_layer(0, &[0, 1]).unwrap();
        assert_eq!(stream.misses().len(), 0);
        stream.end_layer(0);
    }

    #[test]
    fn end_layer_releases_everything_over_a_long_run() {
        // A slot that stays protected is unevictable, and a layer that leaks
        // them degrades to permanent AllSlotsBusy. `LayerCache::plan` asserts
        // against it in debug builds after STUCK_PROTECTED_PLANS attempts, so
        // a run this long is the check.
        let fx = build_install("stream-longrun");
        let mut stream = open(&fx, 2);
        for step in 0..200u32 {
            let a = step % 4;
            let b = (step / 4 + 1) % 4;
            let request = if a == b { vec![a] } else { vec![a, b] };
            stream.begin_layer(step % 2, &request).unwrap();
            stream.await_misses().unwrap();
            for &(_, slot) in stream.misses().to_vec().iter() {
                stream.view(step % 2, slot).unwrap();
            }
            stream.end_layer(step % 2);
        }
        let stats = stream.stats();
        assert_eq!(stats.hits + stats.misses + stats.pending_hits, 350);
        assert!(stats.hits > 0, "no hit in 200 steps over 4 experts");
        assert_eq!(stats.pending_hits, 0, "a step left a read in flight");
        assert_eq!(stats.read_retries, 0);
        // Every miss moved exactly one blob, and the two layers' blobs are
        // different sizes, so the total is bracketed rather than exact.
        let small = fx.layout.layers[0].stride.min(fx.layout.layers[1].stride);
        let large = fx.layout.layers[0].stride.max(fx.layout.layers[1].stride);
        assert!((stats.misses * small..=stats.misses * large).contains(&stats.bytes_read));
        assert_eq!(stats.reads_submitted, stats.misses);
    }

    #[test]
    fn rejects_bad_layers_experts_and_call_order() {
        let fx = build_install("stream-bounds");
        let mut stream = open(&fx, 2);

        assert!(matches!(
            stream.begin_layer(7, &[0]).unwrap_err(),
            IoError::LayerOutOfRange { layer: 7, .. }
        ));
        assert!(matches!(
            stream.begin_layer(0, &[9]).unwrap_err(),
            IoError::Cache(CacheError::ExpertOutOfRange { expert: 9, .. })
        ));
        // More distinct experts than the layer has slots.
        assert!(matches!(
            stream.begin_layer(0, &[0, 1, 2]).unwrap_err(),
            IoError::Cache(CacheError::TooFewSlots { .. })
        ));

        stream.begin_layer(0, &[0]).unwrap();
        let err = stream.begin_layer(0, &[1]).unwrap_err();
        assert!(matches!(err, IoError::Io { .. }), "unexpected error: {err}");
        stream.await_misses().unwrap();
        stream.end_layer(0);

        // A slot that never held a completed read has no view.
        assert!(stream.view(0, 1).is_err());
        assert!(stream.view(9, 0).is_err());
    }

    #[test]
    fn duplicate_ids_resolve_once() {
        let fx = build_install("stream-dupes");
        let mut stream = open(&fx, 2);
        stream.begin_layer(0, &[2, 2]).unwrap();
        assert_eq!(stream.misses().len(), 1);
        assert_eq!(stream.misses()[0].0, 0, "the first occurrence should win");
        stream.await_misses().unwrap();
        stream.end_layer(0);
        assert_eq!(stream.stats().misses, 1);
    }

    #[test]
    fn short_file_is_an_error_not_a_silent_short_read() {
        // Truncate the layer file *after* it has been opened and verified, so
        // the read itself has to notice. The first blob still fits, the last
        // one does not.
        let fx = build_install("stream-eof");
        let mut stream = open(&fx, 2);
        stream.begin_layer(0, &[0]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);

        let path = fx.root.join(&fx.layout.layers[0].file);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();

        stream.begin_layer(0, &[3]).unwrap();
        let err = stream.await_misses().unwrap_err();
        match &err {
            IoError::Io { source, .. } => assert!(
                source.kind() == io::ErrorKind::UnexpectedEof
                    || source.raw_os_error().is_some_and(|e| e == libc::EINVAL),
                "unexpected source: {source:?}"
            ),
            other => panic!("unexpected error: {other}"),
        }
        // The failed step cleaned up after itself: the layer still works.
        stream.end_layer(0);
        std::fs::write(&path, &bytes).unwrap();
        stream.begin_layer(0, &[0]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
    }

    #[test]
    fn layer_hashes_are_checked_only_when_asked_for() {
        let fx = build_install("stream-hash");
        let victim = fx.root.join(&fx.layout.layers[1].file);
        let mut bytes = std::fs::read(&victim).unwrap();
        bytes[4096] ^= 0xff;
        std::fs::write(&victim, &bytes).unwrap();

        // Opt in: the corruption is caught when the file is first used.
        let mut strict = open_with(
            &fx,
            2,
            LoadOptions {
                skip_hashes: false,
                verify_layer_hashes: true,
            },
        )
        .unwrap();
        assert!(matches!(
            strict.begin_layer(1, &[0]).unwrap_err(),
            IoError::Format(FormatError::HashMismatch { .. })
        ));
        // Layer 0 is untouched and still streams.
        strict.begin_layer(0, &[0]).unwrap();
        strict.await_misses().unwrap();
        strict.end_layer(0);

        // Default: layer files are size-checked only, so this reads happily.
        let mut lax = open(&fx, 2);
        lax.begin_layer(1, &[0]).unwrap();
        lax.await_misses().unwrap();
        lax.end_layer(1);

        // The size check is not optional, whatever the hash policy.
        std::fs::write(&victim, &bytes[..bytes.len() - 4096]).unwrap();
        let mut short = open(&fx, 2);
        assert!(matches!(
            short.begin_layer(1, &[0]).unwrap_err(),
            IoError::Format(FormatError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn a_stream_can_be_driven_from_the_thread_it_was_moved_to() {
        // `ExpertStream` is `Send` because the natural shape is "open the
        // model here, decode over there". io_uring's `SINGLE_ISSUER` binds a
        // ring to the task that created it and answers `EEXIST` to anyone
        // else, so this is the test that says whether the mode flags and the
        // `Send` bound can both be true at once.
        let fx = build_install("stream-moved");
        let mut stream = open(&fx, 2);
        // Three different threads in turn, which is also what drives the
        // rebind counter past the point where thread-bound flags are kept.
        for expert in [2u32, 1, 3] {
            let expected = blob_on_disk(&fx, 1, expert);
            let handle = std::thread::spawn(move || {
                let mut stream = stream;
                stream.begin_layer(1, &[expert]).unwrap();
                stream.await_misses().unwrap();
                let slot = stream.misses()[0].1;
                let blob = stream.view(1, slot).unwrap().blob().to_vec();
                stream.end_layer(1);
                (stream, blob)
            });
            let (moved, blob) = handle.join().unwrap();
            assert_eq!(blob, expected, "expert {expert} after a thread move");
            stream = moved;
        }
        assert_eq!(stream.stats().misses, 3);
        assert_eq!(stream.stale_completions(), 0);
    }

    #[test]
    fn probe_reports_a_sane_mode() {
        let fx = build_install("stream-mode");
        let stream = open(&fx, 2);
        let mode = stream.mode();
        println!(
            "mode={mode} direct_io={} support={}",
            stream.direct_io(),
            stream.direct_support()
        );
        // O_DIRECT is claimed only when the probe verified it, and only the
        // ring can claim a non-pread mode.
        assert_eq!(
            mode == StreamMode::ODirect,
            stream.direct_io() && cfg!(feature = "io-uring")
        );
        if cfg!(not(feature = "io-uring")) {
            assert_eq!(mode, StreamMode::Pread);
        }
        assert_eq!(stream.direct_io(), stream.direct_support().is_verified());
        assert_eq!(stream.stale_completions(), 0);
    }

    /// Every per-step buffer is sized at construction, not grown on token 0.
    ///
    /// Six `Vec`s used to reach their working size the first time a layer was
    /// planned — `CachePlan`'s two lists, the open step's three, and the
    /// in-flight table — and they are per *process*, not per layer, so it was
    /// six growths inside the first token's layer loop and none after.
    #[test]
    fn the_per_step_buffers_are_preallocated() {
        let fx = build_install("stream-prealloc");
        let stream = open(&fx, 4);
        let top_k = fx.manifest.arch.top_k as usize;
        assert!(top_k >= 2, "the fixture must route more than one expert");

        assert!(stream.step.hits.capacity() >= top_k);
        assert!(stream.step.misses.capacity() >= top_k);
        assert!(stream.step.protected.capacity() >= top_k);
        assert!(stream.inflight.reads.capacity() >= top_k);

        // `CachePlan` keeps its lists private, so it answers for its own
        // capacity: the smaller of the two, which is what `with_capacity`
        // promises. Compared against a plan that has never allocated, so this
        // says "preallocated for this step" rather than "non-zero".
        assert!(stream.plan.capacity() >= top_k);
        assert_eq!(CachePlan::new().capacity(), 0);

        // And a full-width step does not disturb any of it.
        let request: Vec<u32> = (0..top_k as u32).collect();
        let mut stream = stream;
        let before = (
            stream.step.hits.capacity(),
            stream.step.misses.capacity(),
            stream.step.protected.capacity(),
            stream.inflight.reads.capacity(),
        );
        let plan_before = stream.plan.capacity();
        stream.begin_layer(0, &request).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
        assert_eq!(
            before,
            (
                stream.step.hits.capacity(),
                stream.step.misses.capacity(),
                stream.step.protected.capacity(),
                stream.inflight.reads.capacity(),
            ),
            "a top-k step reallocated a preallocated buffer"
        );
        assert_eq!(
            stream.plan.capacity(),
            plan_before,
            "a top-k step regrew the cache plan"
        );
    }

    // ---- the prefill arena -----------------------------------------------

    #[test]
    fn the_arena_carve_is_page_aligned_and_allocates_nothing() {
        let fx = build_install("stream-arena-carve");
        let mut stream = open(&fx, 4);
        let pool_bytes = stream.cache_bytes();
        let want = 2 * fx.layout.layers[0].stride as usize;

        stream.take_arena(want).unwrap();
        let arena = stream.arena.expect("the arena is out");
        assert_eq!(arena.len, want);
        assert!(direct::is_aligned(arena.base.as_ptr() as usize as u64));
        assert!(direct::is_aligned(arena.len as u64));
        assert!(arena.len as u64 <= pool_bytes, "the carve left the slab");
        // The slab is what it was: the carve is arithmetic, not an allocation.
        assert_eq!(stream.cache_bytes(), pool_bytes);
        assert_eq!(stream.arena_len(), want);

        stream.release_arena();
        assert!(stream.arena.is_none());
        assert_eq!(stream.arena_len(), 0);
        // And the cache is usable the moment the arena is back.
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);

        // A carve past the slab is a typed refusal.
        let err = stream.take_arena(pool_bytes as usize + 4096).unwrap_err();
        assert!(
            matches!(err, SweepError::ArenaTooSmall { .. }),
            "unexpected error: {err}"
        );
    }

    /// The arena may not be taken while the cache is mid-step: those reads are
    /// writing into the very buffers it would hand out.
    #[test]
    fn the_arena_is_refused_while_the_cache_is_busy() {
        let fx = build_install("stream-arena-busy");
        let mut stream = open(&fx, 4);
        stream.begin_layer(0, &[0, 1]).unwrap();
        assert_eq!(stream.misses().len(), 2, "the fixture must miss cold");
        let err = stream.take_arena(4096).unwrap_err();
        assert!(
            matches!(err, SweepError::ReadsInFlight { outstanding: 2 }),
            "unexpected error: {err}"
        );

        stream.await_misses().unwrap();
        let err = stream.take_arena(4096).unwrap_err();
        assert!(
            matches!(err, SweepError::StepOpen { layer: 0 }),
            "unexpected error: {err}"
        );

        stream.end_layer(0);
        stream
            .take_arena(4096)
            .expect("a closed step lets it through");
        stream.release_arena();
    }

    /// Taking the arena empties every layer's occupancy, because sweep bytes
    /// land in those buffers — and keeps the ghost history, because that is
    /// indexed by expert id and survives eviction on purpose.
    #[test]
    fn taking_the_arena_empties_every_layers_occupancy() {
        let fx = build_install("stream-arena-invalidate");
        let mut stream = open(&fx, 4);
        for layer in 0..2u32 {
            stream.begin_layer(layer, &[0, 1]).unwrap();
            stream.await_misses().unwrap();
            stream.end_layer(layer);
        }
        for layer in 0..2u32 {
            assert!(
                (0..stream.usable_slots(layer))
                    .any(|slot| stream.caches[layer as usize].is_ready(slot)),
                "layer {layer} should hold something before the arena"
            );
        }
        let hits_before = stream.stats().hits;

        // One page is enough: the invalidation is not scoped to the carve.
        stream.take_arena(4096).unwrap();
        for layer in 0..2u32 {
            for slot in 0..stream.usable_slots(layer) {
                assert!(
                    !stream.caches[layer as usize].is_ready(slot),
                    "layer {layer} slot {slot} survived the arena"
                );
            }
        }
        stream.release_arena();

        // The counters are untouched, so the ghost history and the totals both
        // carried across.
        assert_eq!(stream.stats().hits, hits_before);
        // Every expert now misses, and it is an *eviction* miss: the layer had
        // fetched it, which is what the surviving `fetched` flags record.
        stream.begin_layer(0, &[0, 1]).unwrap();
        assert!(stream.hits().is_empty());
        stream.await_misses().unwrap();
        stream.end_layer(0);
        assert_eq!(stream.stats().eviction_misses, 2);
    }

    /// A layout whose blob stride is not 4096-aligned pads every slot, so the
    /// slab is not a gapless run of blob-sized buffers and the arena is
    /// refused rather than silently handed out with holes in it.
    #[test]
    fn a_padded_slot_pool_refuses_the_arena() {
        let fx = build_install("stream-arena-padded");
        let mut stream = open(&fx, 4);
        stream.take_arena(4096).expect("the fixture is gapless");
        stream.release_arena();

        stream.force_padded_slots();
        let err = stream.take_arena(4096).unwrap_err();
        match err {
            SweepError::PaddedSlots {
                layer,
                stride,
                pitch,
            } => {
                assert_eq!((layer, stride, pitch), (1, 5000, 8192));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    /// A window read that can never be reaped gives up every slot the arena
    /// covered — the bytes may land anywhere in it — and the process never
    /// sweeps again.
    #[test]
    fn a_lost_window_read_retires_the_slots_the_arena_covered() {
        let fx = build_install("stream-arena-strand");
        let mut stream = open(&fx, 4);
        let stride = fx.layout.layers[0].stride as usize;
        assert_eq!(stream.usable_slots(0), 4);
        assert_eq!(stream.usable_slots(1), 4);

        // Two of layer 0's four slots, and none of layer 1's, which start
        // past the whole of layer 0's run.
        stream.take_arena(2 * stride).unwrap();
        assert_eq!(
            stream.strand_arena(),
            None,
            "two of four slots is still top_k of 2"
        );

        assert_eq!(stream.usable_slots(0), 2, "the arena's slots stayed live");
        assert_eq!(stream.usable_slots(1), 4, "an untouched layer shrank");
        assert!(stream.arena.is_none());

        let err = stream.take_arena(4096).unwrap_err();
        assert!(
            matches!(err, SweepError::ArenaPoisoned),
            "unexpected error: {err}"
        );
        // What is left still serves steps.
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
    }

    /// The arena may not be carved over a slot buffer that was leaked to
    /// protect a read which can never be reaped.
    #[test]
    fn the_arena_refuses_to_cover_a_retired_slot_buffer() {
        // The failure this guards: `strand_step` gives a read up through
        // `Inflight::abandon`, which drains `outstanding` *because the read
        // can never be reaped* — not because it finished — leaks the
        // destination slot in place, and (unlike `strand_arena`) does not
        // poison the arena. So every guard `take_arena` had still passed, and
        // the next sweep issued a 23 MiB O_DIRECT read into pages an abandoned
        // decode read was still writing: two O_DIRECT reads aliased onto one
        // buffer, which EXP-007 measured as 13-27% spurious btrfs EIO.
        let fx = build_install("stream-arena-over-retired");
        let mut stream = open(&fx, 4);
        let stride0 = fx.layout.layers[0].stride as usize;

        // Layer 1's slots start past the whole of layer 0's run, so a carve
        // that stops inside layer 0 is still legal: the refusal is about the
        // bytes the carve covers, not about "something somewhere was retired".
        stream.begin_layer(1, &[3]).unwrap();
        stream.strand_open_step();
        assert_eq!(stream.usable_slots(1), 3);
        stream
            .take_arena(2 * stride0)
            .expect("layer 1's retirement is past this carve");
        stream.release_arena();

        // A retirement inside the carve is not.
        stream.begin_layer(0, &[3]).unwrap();
        stream.strand_open_step();
        match stream.take_arena(2 * stride0).unwrap_err() {
            SweepError::ArenaOverRetired {
                layer,
                offset,
                bytes,
            } => {
                assert_eq!(layer, 0);
                assert_eq!(bytes, 2 * stride0);
                assert!(offset < bytes, "the refusal named a buffer outside it");
            }
            other => panic!("unexpected error: {other}"),
        }
        // Permanent — a leaked buffer never becomes safe again — and the two
        // prefill entry points are refused through the same check.
        assert!(matches!(
            stream.take_arena(4096).unwrap_err(),
            SweepError::ArenaOverRetired { .. }
        ));
        let mut plan = crate::io::SweepPlan::new();
        assert!(matches!(
            stream
                .sweep_layer(&mut plan, 0, &[0], super::super::SweepConfig::default())
                .unwrap_err(),
            SweepError::ArenaOverRetired { .. }
        ));
        let small = super::super::SweepConfig {
            experts_per_window: 1,
            windows_in_flight: 1,
        };
        assert!(matches!(
            stream.begin_prefill(4096, small).unwrap_err(),
            SweepError::ArenaOverRetired { .. }
        ));
        // And decode keeps working on what is left, which is the whole reason
        // retirement is preferred to giving up.
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
    }

    /// A stranding that drops a layer below `top_k` says so at the cause.
    #[test]
    fn stranding_the_arena_below_top_k_is_named_at_its_cause() {
        // The arena is carved from the head of the slab, where layer 0 lives,
        // and at the shipped dials it is wider than layer 0's whole slot row
        // (46.7 MiB against 32.1 MiB). So one unreapable window read leaves
        // layer 0 with nothing. That used to surface three decode steps later
        // as `TooFewSlots`, which names a symptom and no cause at all.
        let fx = build_install("stream-arena-strand-top-k");
        let top_k = fx.manifest.arch.top_k;
        let stride0 = fx.layout.layers[0].stride as usize;
        let lost = || IoError::io(Path::new("layer_00.bin"), io::Error::other("ring failed"));

        let mut stream = open(&fx, 4);
        stream.take_arena(4 * stride0).unwrap();
        match stream.strand_open_arena_error(lost()) {
            SweepError::CacheStranded {
                layer,
                slots,
                top_k: k,
                ..
            } => assert_eq!((layer, slots, k), (0, 0, top_k)),
            other => panic!("unexpected error: {other}"),
        }
        assert_eq!(stream.usable_slots(0), 0);
        assert_eq!(stream.usable_slots(1), 4, "an untouched layer shrank");
        // The old symptom is still what a later step sees; the point is that
        // it is no longer the first thing anybody hears about.
        assert!(matches!(
            stream.begin_layer(0, &[0]).unwrap_err(),
            IoError::Cache(CacheError::TooFewSlots { .. })
        ));

        // A stranding every layer survives still reports the read failure.
        let mut stream = open(&fx, 4);
        stream.take_arena(2 * stride0).unwrap();
        let err = stream.strand_open_arena_error(lost());
        assert!(matches!(err, SweepError::Io(_)), "unexpected error: {err}");
        assert_eq!(stream.usable_slots(0), 4 - 2);
        stream.begin_layer(0, &[0, 1]).unwrap();
        stream.await_misses().unwrap();
        stream.end_layer(0);
    }

    // ---- in-flight bookkeeping -------------------------------------------
    //
    // Completion handling is where out-of-order arrival, stale tags, short
    // reads and the retry budget live. None of that can be provoked from a
    // filesystem without root, so it is driven directly here.

    fn tracked(inflight: &mut Inflight, slots: &[u32]) -> Vec<u64> {
        slots
            .iter()
            .enumerate()
            .map(|(i, &slot)| {
                let index = inflight.track(0, slot, i as u32, u64::from(slot) * 4096, 4096);
                inflight.reads[index].user_data
            })
            .collect()
    }

    #[test]
    fn completions_are_dispatched_out_of_order_by_user_data() {
        let mut io = IoStats::default();
        let mut inflight = Inflight::default();
        let tokens = tracked(&mut inflight, &[5, 2, 7]);
        assert_eq!(inflight.outstanding, 3);

        // Arrive 2, 0, 1 — each must credit its own read.
        for &i in &[2usize, 0, 1] {
            match inflight.resolve(tokens[i], 4096, &mut io) {
                Reap::Done(index) => {
                    assert_eq!(index, i, "completion credited to the wrong read");
                    assert_eq!(inflight.reads[index].slot, [5u32, 2, 7][i]);
                }
                other => panic!("expected Done, got {other:?}"),
            }
        }
        assert_eq!(inflight.outstanding, 0);
        assert_eq!(io.bytes_read, 3 * 4096);
    }

    #[test]
    fn stale_completions_are_rejected() {
        let mut io = IoStats::default();
        let mut inflight = Inflight::default();
        let tokens = tracked(&mut inflight, &[1, 4]);

        // An index nothing was ever submitted for.
        assert!(matches!(inflight.resolve(9, 4096, &mut io), Reap::Stale));
        // The right index carrying a tag from another life.
        let forged = (tokens[0] & 0xffff_ffff) | (0xdead_u64 << 32);
        assert!(matches!(
            inflight.resolve(forged, 4096, &mut io),
            Reap::Stale
        ));
        // A duplicate of a completion already consumed.
        assert!(matches!(
            inflight.resolve(tokens[1], 4096, &mut io),
            Reap::Done(1)
        ));
        assert!(matches!(
            inflight.resolve(tokens[1], 4096, &mut io),
            Reap::Stale
        ));
        // And the token a retry replaced.
        assert!(matches!(
            inflight.resolve(tokens[0], -libc::EIO, &mut io),
            Reap::Again(0)
        ));
        let reissued = inflight.retag(0);
        assert_ne!(reissued, tokens[0]);
        assert!(matches!(
            inflight.resolve(tokens[0], 4096, &mut io),
            Reap::Stale
        ));
        assert_eq!(io.bytes_read, 4096, "a stale completion moved the counters");
        assert_eq!(inflight.outstanding, 1);
        // The reissued attempt still completes normally.
        assert!(matches!(
            inflight.resolve(reissued, 4096, &mut io),
            Reap::Done(0)
        ));
        assert_eq!(inflight.outstanding, 0);
    }

    #[test]
    fn eio_is_retried_exactly_once_then_reported() {
        let mut io = IoStats::default();
        let mut inflight = Inflight::default();
        let tokens = tracked(&mut inflight, &[0]);

        assert!(matches!(
            inflight.resolve(tokens[0], -libc::EIO, &mut io),
            Reap::Again(0)
        ));
        assert_eq!(io.read_retries, 1);
        let second = inflight.retag(0);
        match inflight.resolve(second, -libc::EIO, &mut io) {
            Reap::Failed(0, error) => {
                assert_eq!(error.raw_os_error(), Some(libc::EIO));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
        assert_eq!(io.read_retries, 1, "the second failure was retried too");
        assert_eq!(inflight.outstanding, 0);
        assert_eq!(io.bytes_read, 0);
    }

    /// A sweep window goes through the same table as a cache miss — the short
    /// read, the one `EIO` retry, the restart rule — and lands in the sweep
    /// half of the counters rather than the cache half.
    ///
    /// This is the whole reason [`Inflight`] is destination-agnostic: the
    /// retry logic is the part that is hard to get right, and it is not worth
    /// having two of.
    #[test]
    fn a_window_read_retries_like_a_blob_and_counts_in_the_sweep_half() {
        let mut io = IoStats {
            in_sweep: true,
            ..IoStats::default()
        };
        let mut inflight = Inflight {
            align: direct::DIO_ALIGN as u32,
            ..Inflight::default()
        };
        // One 4-page window of layer 0, buffer 1, starting at expert 8.
        let index = inflight.track_arena(0, 1, 8, 8 * 4096, 4 * 4096, 3 * 4096);
        assert_eq!(inflight.reads[index].dst, Dest::Arena(3 * 4096));
        let token = inflight.reads[index].user_data;

        // A short read that stopped on a block boundary continues.
        assert!(matches!(
            inflight.resolve(token, 2 * 4096, &mut io),
            Reap::Again(0)
        ));
        assert_eq!(inflight.reads[0].filled, 2 * 4096);
        assert_eq!(inflight.reads[0].remaining, 2 * 4096);

        // `EIO` on the continuation is retried exactly once.
        let second = inflight.retag(0);
        assert!(matches!(
            inflight.resolve(second, -libc::EIO, &mut io),
            Reap::Again(0)
        ));
        let third = inflight.retag(0);
        assert!(matches!(
            inflight.resolve(third, 2 * 4096, &mut io),
            Reap::Done(0)
        ));
        assert_eq!(inflight.outstanding, 0);

        assert_eq!(io.sweep_bytes_read, 4 * 4096);
        assert_eq!(io.sweep_read_retries, 2, "short read plus one EIO");
        assert_eq!(io.bytes_read, 0, "sweep bytes leaked into the cache half");
        assert_eq!(io.read_retries, 0);

        // A short read that stopped *off* a boundary restarts the window,
        // because O_DIRECT cannot resume from an unaligned offset.
        let index = inflight.track_arena(0, 0, 0, 0, 4 * 4096, 0);
        let token = inflight.reads[index].user_data;
        assert!(matches!(
            inflight.resolve(token, 4096 + 512, &mut io),
            Reap::Again(1)
        ));
        assert_eq!(inflight.reads[1].filled, 0, "the window was not restarted");
        assert_eq!(inflight.reads[1].remaining, 4 * 4096);
    }

    /// A restarted blob gets its retry budget back: it is being read again
    /// from the beginning, so the attempt the previous try spent is not its.
    #[test]
    fn a_restart_gives_the_blob_a_fresh_retry_budget() {
        let mut io = IoStats::default();
        let mut inflight = Inflight {
            align: direct::DIO_ALIGN as u32,
            ..Inflight::default()
        };
        let index = inflight.track(0, 0, 0, 0, 8192);
        let token = inflight.reads[index].user_data;

        // One transient failure, retried.
        assert!(matches!(
            inflight.resolve(token, -libc::EIO, &mut io),
            Reap::Again(0)
        ));
        assert_eq!(inflight.reads[0].attempts, 1);

        // A short read that stopped off a block boundary restarts the blob.
        let token = inflight.retag(0);
        assert!(matches!(
            inflight.resolve(token, 1536, &mut io),
            Reap::Again(0)
        ));
        assert_eq!(inflight.reads[0].restarts, 1);
        assert_eq!(inflight.reads[0].filled, 0);
        assert_eq!(
            inflight.reads[0].attempts, 0,
            "a restarted blob kept a retry budget it had already spent"
        );

        // So the next transient error is still worth one more attempt...
        let token = inflight.retag(0);
        assert!(matches!(
            inflight.resolve(token, -libc::EIO, &mut io),
            Reap::Again(0)
        ));
        // ...and it is still bounded: one restart times one retry each.
        let token = inflight.retag(0);
        match inflight.resolve(token, -libc::EIO, &mut io) {
            Reap::Failed(0, error) => assert_eq!(error.raw_os_error(), Some(libc::EIO)),
            other => panic!("expected Failed, got {other:?}"),
        }
        assert_eq!(inflight.outstanding, 0);
    }

    /// A submission failure re-reported as a read failure keeps its errno.
    #[test]
    fn a_flattened_submission_failure_keeps_its_errno() {
        let wrapped = IoError::io(
            Path::new("experts/layer_00.bin"),
            io::Error::from_raw_os_error(libc::ENOSPC),
        );
        let flat = flatten_io(wrapped);
        assert_eq!(flat.raw_os_error(), Some(libc::ENOSPC));
        // Anything without an errno still says exactly what it said.
        let other = IoError::LayerOutOfRange {
            layer: 9,
            n_layers: 2,
        };
        let text = other.to_string();
        assert_eq!(flatten_io(other).to_string(), text);
    }

    #[test]
    fn permanent_errors_are_not_retried() {
        let mut io = IoStats::default();
        let mut inflight = Inflight::default();
        let tokens = tracked(&mut inflight, &[0]);
        match inflight.resolve(tokens[0], -libc::EINVAL, &mut io) {
            Reap::Failed(0, error) => assert_eq!(error.raw_os_error(), Some(libc::EINVAL)),
            other => panic!("expected Failed, got {other:?}"),
        }
        assert_eq!(io.read_retries, 0);
    }

    #[test]
    fn short_reads_continue_and_eof_fails() {
        let mut io = IoStats::default();
        let mut inflight = Inflight::default();
        let tokens = tracked(&mut inflight, &[0, 1]);

        // Half a blob: continue from where it stopped, at the right offset.
        assert!(matches!(
            inflight.resolve(tokens[0], 2048, &mut io),
            Reap::Again(0)
        ));
        assert_eq!(inflight.reads[0].filled, 2048);
        assert_eq!(inflight.reads[0].remaining, 2048);
        let next = inflight.retag(0);
        assert!(matches!(
            inflight.resolve(next, 2048, &mut io),
            Reap::Done(0)
        ));
        assert_eq!(io.bytes_read, 4096);

        // Zero bytes with bytes still owed is EOF, not success.
        match inflight.resolve(tokens[1], 0, &mut io) {
            Reap::Failed(1, error) => assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof),
            other => panic!("expected Failed, got {other:?}"),
        }
        assert_eq!(inflight.outstanding, 0);
    }

    #[test]
    fn an_unaligned_short_read_restarts_the_blob_instead_of_being_resumed() {
        // Under O_DIRECT the reissue offset is `base + filled`, and the kernel
        // answers `EINVAL` to an offset that is not a multiple of the block
        // size — an errno `is_retryable` deliberately excludes, so continuing
        // from an unaligned stop is a guaranteed terminal failure. The blob is
        // started again instead.
        let mut io = IoStats::default();
        let mut inflight = Inflight {
            align: direct::DIO_ALIGN as u32,
            ..Inflight::default()
        };
        let tokens = tracked(&mut inflight, &[0]);

        assert!(matches!(
            inflight.resolve(tokens[0], 1536, &mut io),
            Reap::Again(0)
        ));
        assert_eq!(
            inflight.reads[0].filled, 0,
            "reissue offset must be aligned"
        );
        assert_eq!(inflight.reads[0].remaining, 4096);
        assert_eq!(inflight.reads[0].restarts, 1);
        // The bytes really did move, even though they are being re-read.
        assert_eq!(io.bytes_read, 1536);
        assert_eq!(io.read_retries, 1);

        // The restart budget is one: a device that keeps stopping mid-block is
        // a terminal error, not a loop.
        let next = inflight.retag(0);
        match inflight.resolve(next, 1536, &mut io) {
            Reap::Failed(0, error) => assert_eq!(error.kind(), io::ErrorKind::InvalidData),
            other => panic!("expected Failed, got {other:?}"),
        }
        assert_eq!(inflight.outstanding, 0);

        // A short read that *did* stop on a block boundary still continues
        // from where it stopped, which is the common case.
        let mut aligned = Inflight {
            align: direct::DIO_ALIGN as u32,
            ..Inflight::default()
        };
        let index = aligned.track(0, 0, 0, 0, 8192);
        let token = aligned.reads[index].user_data;
        assert!(matches!(
            aligned.resolve(token, 4096, &mut io),
            Reap::Again(0)
        ));
        assert_eq!(aligned.reads[0].filled, 4096);
        assert_eq!(aligned.reads[0].restarts, 0);

        // And with no alignment requirement nothing is restarted at all.
        let mut buffered = Inflight::default();
        let index = buffered.track(0, 0, 0, 0, 4096);
        let token = buffered.reads[index].user_data;
        assert!(matches!(
            buffered.resolve(token, 1536, &mut io),
            Reap::Again(0)
        ));
        assert_eq!(buffered.reads[0].filled, 1536);
        assert_eq!(buffered.reads[0].restarts, 0);
    }
}
