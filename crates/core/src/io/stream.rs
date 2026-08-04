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
//!   retried exactly once and then reported as a typed error; a short read
//!   continues from where it stopped; `result == 0` is EOF, never success.
//! - **The SQ fills.** `PushError` means submit-then-retry, never drop.
//! - **`submit_and_wait` returns `EINTR`.** Retried, not propagated.
//! - **A failed fill leaves a slot un-owned.** The step invalidates it (the
//!   completion has been reaped, which is that call's safety requirement) and
//!   releases the rest, so a failed layer does not strand slots.

use std::fmt;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(feature = "io-uring")]
use std::os::fd::AsRawFd;

#[cfg(feature = "io-uring")]
use io_uring::{IoUring, opcode, types};

use crate::format::{ExpertsLayout, FormatError, Manifest, sha256_file};

use super::direct::{self, DirectFault, DirectSupport};
use super::{
    CachePlan, ExpertReader, ExpertView, IoError, LayerCache, LoadOptions, MAX_SLOTS, SLOT_ALIGN,
    SlotError, SlotGuard, SlotPool,
};

/// Submission queue depth.
///
/// The drive saturates early — 1.211 GB/s at QD4 against 1.390 at QD16 at the
/// expert stride, so QD4 is 87% of QD16 — and deeper queues buy latency
/// rather than bandwidth (per-blob p50 2.34 ms at QD1 against 15.56 ms at
/// QD8, provisional, EXP-008). The decode loop waits on *all* misses, so
/// latency is the quantity that matters and the queue stays at the low end of
/// the useful range. It is also charged against `RLIMIT_MEMLOCK`, which is
/// 8 MiB soft *and* hard under systemd defaults since kernel 6.14.
#[cfg(feature = "io-uring")]
const RING_ENTRIES: u32 = 8;

/// Passes of the completion loop with no progress before it is called a
/// livelock rather than a slow drive.
#[cfg(feature = "io-uring")]
const MAX_IDLE_PASSES: u32 = 1024;

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

/// Cumulative streaming telemetry, summed across every layer.
///
/// The cache half comes from the per-layer [`LayerCache`]s; the I/O half is
/// counted here. `hits + pending_hits + misses` is the number of routed
/// experts resolved.
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
}

/// I/O counters kept by the stream itself.
#[derive(Debug, Clone, Copy, Default)]
struct IoStats {
    bytes_read: u64,
    reads_submitted: u64,
    read_retries: u64,
    io_wait: Duration,
    stale_completions: u64,
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

/// One outstanding blob read.
#[derive(Debug, Clone, Copy)]
struct Read {
    /// Token this read was submitted with. A completion carrying anything
    /// else names a read that no longer exists.
    user_data: u64,
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
}

impl Inflight {
    /// Record a read about to be submitted and return its index.
    fn track(&mut self, layer: u32, slot: u32, expert: u32, base: u64, len: u32) -> usize {
        let index = self.reads.len();
        let user_data = self.token(index);
        self.reads.push(Read {
            user_data,
            layer,
            slot,
            expert,
            base,
            filled: 0,
            remaining: len,
            attempts: 0,
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
            io.bytes_read += u64::from(got);
            if read.remaining == 0 {
                read.done = true;
                self.outstanding -= 1;
                return Reap::Done(index);
            }
            io.read_retries += 1;
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
            io.read_retries += 1;
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
/// **Field order is load-bearing.** `guards` borrow `pool`; struct fields
/// drop in declaration order, so the guards are released before the slab they
/// point into is freed. (`SlotPool::drop` would refuse to free a slab with a
/// live lease and leak it instead — correct, but 1.4 GiB of correct.)
struct SlotTable {
    /// Flat `layer * slots_per_layer + slot`. `None` only where a guard was
    /// leaked to protect an unreaped read.
    guards: Box<[Option<SlotGuard<'static>>]>,
    /// The pool every guard borrows. Boxed so its address is stable, and
    /// declared last so it outlives them.
    pool: Box<SlotPool>,
    slots_per_layer: u32,
}

impl fmt::Debug for SlotTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlotTable")
            .field("slots_per_layer", &self.slots_per_layer)
            .field("total_bytes", &self.pool.total_bytes())
            .finish()
    }
}

impl SlotTable {
    /// Allocate the pool and take every lease in it.
    fn new(slots_per_layer: u32, strides: &[u64]) -> Result<Self, SlotError> {
        let pool = Box::new(SlotPool::new(slots_per_layer, strides)?);
        let n_layers = pool.n_layers();
        let mut guards: Vec<Option<SlotGuard<'static>>> = (0..strides.len()
            * slots_per_layer as usize)
            .map(|_| None)
            .collect();
        for layer in 0..n_layers {
            for _ in 0..slots_per_layer {
                let guard = pool.acquire(layer)?;
                // SAFETY: the lease is being extended to `'static` because it
                // is stored beside the pool it borrows, which no lifetime can
                // express. Three things make it sound, and all three are
                // structural rather than conventional: `pool` is boxed, so
                // its address does not change when this value moves; the
                // guards are private to `SlotTable` and never handed out, so
                // none can outlive it; and `guards` is declared before `pool`
                // so every lease is dropped before the slab is freed.
                let guard =
                    unsafe { std::mem::transmute::<SlotGuard<'_>, SlotGuard<'static>>(guard) };
                let flat = layer as usize * slots_per_layer as usize + guard.index() as usize;
                guards[flat] = Some(guard);
            }
        }
        Ok(Self {
            guards: guards.into_boxed_slice(),
            pool,
            slots_per_layer,
        })
    }

    /// Flat index of `(layer, slot)`, or `None` if either is out of range.
    fn flat(&self, layer: u32, slot: u32) -> Option<usize> {
        if slot >= self.slots_per_layer || layer >= self.pool.n_layers() {
            return None;
        }
        Some(layer as usize * self.slots_per_layer as usize + slot as usize)
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
    fn leak(&mut self, layer: u32, slot: u32) {
        if let Some(flat) = self.flat(layer, slot)
            && let Some(guard) = self.guards[flat].take()
        {
            guard.leak();
        }
    }

    /// Resident bytes of the whole pool.
    fn total_bytes(&self) -> u64 {
        self.pool.total_bytes() as u64
    }
}

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
    /// reissuing does not need the completion queue borrowed.
    #[cfg(feature = "io-uring")]
    completions: Vec<(u64, i32)>,
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
    mode: StreamMode,
    support: DirectSupport,
    /// Whether layer files are opened with `O_DIRECT`.
    direct_open: bool,
    /// Whether first use of a layer file hashes it in full: the opt-in
    /// [`LoadOptions::verify_layer_hashes`], never under
    /// [`LoadOptions::skip_hashes`].
    hash_layers: bool,
    io: IoStats,
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
    /// # Errors
    ///
    /// [`IoError::Format`] when the layout does not validate against the
    /// manifest; [`IoError::Slots`] when the budget cannot fit one slot per
    /// layer ([`SlotError::ZeroSlotsPerLayer`]) or the pool does not fit in
    /// memory; [`IoError::Cache`] when a layer's expert count is out of
    /// range; [`IoError::TooLarge`] for a blob stride past `u32`.
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

        let slots_per_layer = slots_for_budget(cache_bytes, &strides, &layers)?;
        let slots = SlotTable::new(slots_per_layer, &strides)?;
        let mut caches = Vec::with_capacity(layers.len());
        for layer in &layers {
            caches.push(LayerCache::new(slots_per_layer, layer.n_experts)?);
        }

        // Direct I/O needs a 4096 multiple on the offset *and* the length,
        // and both come from the layout's stride. A layout that does not
        // supply one cannot use O_DIRECT at all.
        let mut stream = Self {
            reader,
            layers: layers.into_boxed_slice(),
            caches: caches.into_boxed_slice(),
            plan: CachePlan::new(),
            step: Step::default(),
            inflight: Inflight::default(),
            #[cfg(feature = "io-uring")]
            completions: Vec::with_capacity(2 * RING_ENTRIES as usize),
            #[cfg(feature = "io-uring")]
            ring: None,
            #[cfg(feature = "io-uring")]
            ring_owner: std::thread::current().id(),
            #[cfg(feature = "io-uring")]
            ring_rebinds: 0,
            slots,
            mode: StreamMode::Pread,
            support: DirectSupport::Unusable(DirectFault::Unsupported),
            direct_open: false,
            hash_layers: options.verify_layer_hashes && !options.skip_hashes,
            io: IoStats::default(),
        };
        stream.support = stream.probe_direct_io(&strides);
        stream.direct_open = stream.support.is_usable();

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
        if stream.mode != StreamMode::ODirect {
            tracing::warn!(
                mode = %stream.mode,
                direct_io = %stream.support,
                "expert reads are not verified to bypass the page cache; the \
                 3 GB memory contract does not hold in this mode"
            );
        }
        Ok(stream)
    }

    /// Slots each layer's cache holds, as bought by the byte budget.
    pub fn slots_per_layer(&self) -> u32 {
        self.slots.slots_per_layer
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
                failure = Some(self.fail(read, io::Error::other(error.to_string())));
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

        #[cfg(feature = "io-uring")]
        if self.ring.is_some() {
            if let Err(error) = self.await_ring(&mut failure) {
                // The ring itself failed, so nothing will ever reap what is
                // still in flight: those slots are given up rather than
                // recycled under a write we cannot see.
                self.strand_step();
                return Err(failure.unwrap_or(error));
            }
        } else {
            self.await_pread(&mut failure);
        }
        #[cfg(not(feature = "io-uring"))]
        self.await_pread(&mut failure);

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

    /// Release every slot the step protected, hits included.
    ///
    /// Exactly one call per [`ExpertStream::begin_layer`]. Releasing only the
    /// misses would leave every hit slot protected forever, which
    /// [`LayerCache::stuck_protected_slot`] detects and `plan` asserts
    /// against in debug builds.
    pub fn end_layer(&mut self, layer: u32) {
        if !self.step.active {
            tracing::warn!(layer, "end_layer without an open step");
            return;
        }
        if self.step.layer != layer {
            tracing::error!(
                asked = layer,
                open = self.step.layer,
                "end_layer for a different layer than the open step; \
                 releasing the open one"
            );
        }
        if self.inflight.outstanding > 0 {
            // Releasing a Filling slot is refused by the cache anyway; say so
            // rather than leaving it silent.
            tracing::error!(
                layer = self.step.layer,
                outstanding = self.inflight.outstanding,
                "end_layer with reads still in flight; await_misses was skipped"
            );
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
    pub fn stats(&self) -> StreamStats {
        let mut stats = StreamStats {
            bytes_read: self.io.bytes_read,
            reads_submitted: self.io.reads_submitted,
            read_retries: self.io.read_retries,
            io_wait: self.io.io_wait,
            ..StreamStats::default()
        };
        for cache in &self.caches {
            let cache = cache.stats();
            stats.hits += cache.hits;
            stats.pending_hits += cache.pending_hits;
            stats.misses += cache.misses;
            stats.cold_misses += cache.cold_misses;
            stats.eviction_misses += cache.eviction_misses;
        }
        stats
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
        if let Err(error) = direct::fadvise_dontneed(&file, 0, 0) {
            tracing::debug!(path = %path.display(), %error, "could not drop the layer file's page cache");
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

    /// Drive the synchronous read path to completion.
    fn await_pread(&mut self, failure: &mut Option<IoError>) {
        let start = Instant::now();
        for index in 0..self.inflight.reads.len() {
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
        self.io.io_wait += start.elapsed();
    }

    /// One positioned read of read `index`'s remaining bytes, in the
    /// kernel's `result` convention.
    fn pread_once(&mut self, index: usize) -> i32 {
        let read = self.inflight.reads[index];
        let Some(file) = self.layers[read.layer as usize].file.as_ref() else {
            return -libc::EBADF;
        };
        let Some(slot) = self.slots.bytes_mut(read.layer, read.slot) else {
            return -libc::EFAULT;
        };
        let from = read.filled as usize;
        let to = from + read.remaining as usize;
        let Some(dst) = slot.get_mut(from..to) else {
            return -libc::EFAULT;
        };
        self.io.reads_submitted += 1;
        match file.read_at(dst, read.base + u64::from(read.filled)) {
            Ok(got) => i32::try_from(got).unwrap_or(i32::MAX),
            Err(error) => -error.raw_os_error().unwrap_or(libc::EIO),
        }
    }

    /// A read completed: its bytes are valid and its expert is resident.
    fn mark_ready(&mut self, index: usize) {
        let read = self.inflight.reads[index];
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
        if let Some(cache) = self.caches.get_mut(read.layer as usize) {
            // SAFETY: this read reached a terminal state, so its completion
            // has been reaped (ring) or the call has returned (pread), and no
            // reissue was made. Nothing is writing into the slot.
            unsafe { cache.invalidate(read.slot) };
        }
        // An invalidated slot is unassigned, so it is not this step's to
        // release any more.
        self.step.protected.retain(|slot| *slot != read.slot);
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
    /// is unknowable. Their slots are **leaked** — the buffer is never handed
    /// out again, which is what makes a late kernel write harmless — and
    /// their cache entries are dropped so the layer keeps planning steps.
    /// The layer loses that much cache capacity for good, which is the price
    /// of not aliasing a live DMA destination.
    fn strand_step(&mut self) {
        for index in 0..self.inflight.reads.len() {
            let read = self.inflight.reads[index];
            if read.done {
                continue;
            }
            tracing::error!(
                layer = read.layer,
                expert = read.expert,
                slot = read.slot,
                "expert read cannot be reaped; leaking its slot"
            );
            self.inflight.abandon(index);
            self.slots.leak(read.layer, read.slot);
            if let Some(cache) = self.caches.get_mut(read.layer as usize) {
                // SAFETY: the slot's buffer has just been leaked, so nothing
                // can ever be handed that address again. Re-assigning the
                // slot *index* is then harmless: a fill into it fails with a
                // typed error rather than aliasing whatever the kernel may
                // still write.
                unsafe { cache.invalidate(read.slot) };
            }
            self.step.protected.retain(|slot| *slot != read.slot);
        }
        self.abandon_step();
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
        // SAFETY: the read is parked in the in-flight table until its
        // completion is reaped, the guard behind this pointer is held for the
        // life of the stream, no other read targets this slot (the cache
        // keeps it `Filling`), and no slice view of it is taken meanwhile:
        // `view` refuses a slot that is not ready. `Drop` leaks the slot if a
        // teardown cannot reap.
        let Some(base) = (unsafe { self.slots.write_ptr(read.layer, read.slot) }) else {
            return Err(self.protocol_error(read.layer, "slot is not in the pool"));
        };
        // SAFETY: `filled < stride` and the slot is `stride` bytes, so this
        // stays inside the slot.
        let dst = unsafe { base.add(read.filled as usize) };
        let entry = opcode::Read::new(types::Fd(fd), dst, read.remaining)
            .offset(read.base + u64::from(read.filled))
            .build()
            .user_data(read.user_data);

        for attempt in 0..2 {
            let ring = self.ring.as_mut().expect("ring");
            // SAFETY: the destination stays alive and untouched until the
            // completion is reaped, as argued above; `entry` is a plain
            // `Read` with no borrowed state beyond that pointer.
            let pushed = unsafe { ring.submission().push(&entry) };
            if pushed.is_ok() {
                self.io.reads_submitted += 1;
                return Ok(());
            }
            if attempt == 0 {
                // A full SQ is a submit-and-retry, never a dropped read.
                self.enter(0)?;
            }
        }
        Err(self.protocol_error(read.layer, "submission queue stayed full after a submit"))
    }

    /// `io_uring_enter`, retrying `EINTR` rather than propagating it.
    ///
    /// `EBUSY` means the completion queue needs draining first; it is
    /// reported as "submitted nothing" so the caller reaps and comes back.
    #[cfg(feature = "io-uring")]
    fn enter(&mut self, want: usize) -> Result<usize, IoError> {
        let ring = self.ring.as_ref().expect("ring");
        loop {
            match ring.submitter().submit_and_wait(want) {
                Ok(submitted) => return Ok(submitted),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
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
    fn await_ring(&mut self, failure: &mut Option<IoError>) -> Result<(), IoError> {
        let mut idle = 0;
        while self.inflight.outstanding > 0 {
            let start = Instant::now();
            self.enter(1)?;
            self.io.io_wait += start.elapsed();

            self.completions.clear();
            {
                let ring = self.ring.as_mut().expect("ring");
                let sink = &mut self.completions;
                let mut cq = ring.completion();
                cq.sync();
                for cqe in &mut cq {
                    sink.push((cqe.user_data(), cqe.result()));
                }
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
            for i in 0..self.completions.len() {
                let (user_data, result) = self.completions[i];
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
                            let error = self.fail(index, io::Error::other(error.to_string()));
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
        // into those slots after the ring's descriptor closes, so their
        // leases are given up rather than returned: `SlotPool::drop` then
        // leaks the slab instead of freeing memory under a live DMA.
        tracing::error!(
            outstanding = self.inflight.outstanding,
            "expert stream dropped with reads in flight; leaking their slots"
        );
        for index in 0..self.inflight.reads.len() {
            let read = self.inflight.reads[index];
            if !read.done {
                self.slots.leak(read.layer, read.slot);
            }
        }
    }
}

/// Slots per layer bought by a total byte budget.
///
/// The pool charges each layer the page-aligned pitch of its own stride, so a
/// slot costs the sum of those across all layers. Clamped to what a layer can
/// use (its expert count) and to [`MAX_SLOTS`].
fn slots_for_budget(
    cache_bytes: u64,
    strides: &[u64],
    layers: &[LayerState],
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
    if slots > usable {
        tracing::info!(
            budget_slots = slots,
            slots_per_layer = usable,
            "expert cache budget buys more slots than a layer can use; clamping"
        );
        return Ok(usable);
    }
    Ok(slots)
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

        // Exactly n rows buys n slots; a fraction over buys no more.
        for slots in 1..=3u32 {
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
}
