//! Per-layer expert slot cache.
//!
//! Owned by wave-1 lane B. Policy is LFU with recency as tie-breaker and
//! **frequency counters indexed by expert id, not by slot**, so a count
//! survives its expert's eviction. That detail is the policy: measured on
//! real routing traces (EXP-005), ghost-history LFU beats per-slot LFU by
//! +2.2 points and LRU by +2.2 at 10 slots/layer, while per-slot LFU beats
//! LRU by only 0.0-1.7 points and loses outright at 48 slots.
//!
//! The per-expert state is two arrays, both sized by the id space: the `u32`
//! ghost counters, and the `bool` "has ever been fetched" flags behind the
//! cold/eviction split. For Qwen3's 128 experts that is 512 B + 128 B per
//! layer, 30 KiB across 48 layers.
//!
//! A slot that is filling from an in-flight read, or still owned by queued
//! compute, is never selected as an eviction victim.
//!
//! # The policy, exactly
//!
//! One [`LayerCache`] per layer. [`LayerCache::plan`] takes the top-k expert
//! ids the router selected for one token in that layer and splits them into
//! hits (the expert already owns a slot, so the caller issues no read) and
//! misses (slot assigned, caller must read into it), evicting as needed:
//!
//! 1. Every requested expert bumps its frequency counter once per routing
//!    decision, hit or miss. Counters are monotonic: never aged, never
//!    decayed, never cleared on eviction. Aging was measured too
//!    (`lfu-aged`, halving every 256 accesses) and lost 1.8 points.
//! 2. Empty slots are filled before anything is evicted.
//! 3. Otherwise the victim is the unprotected slot with the lowest frequency
//!    count, ties broken by least recent use through a logical clock that
//!    ticks once per *distinct* requested expert — a duplicate id inside one
//!    step resolves once and does not tick it again.
//! 4. Slots that are filling, owned by queued compute, or holding an expert
//!    requested by *this* step are never victims. If that leaves no victim,
//!    `plan` returns [`CacheError`] and changes nothing: the upstream
//!    prototype aborts here, and this crate does not panic on such input.
//!
//! # Why counters outlive their expert
//!
//! Routing is bursty: at 10 slots/layer, 87.9% of misses are for experts the
//! layer has already fetched and thrown out, and half of those are
//! re-requested within 4 decode tokens. Per-slot counters reset a
//! re-admitted expert to 1 and it is immediately the cheapest thing to
//! evict, so the cache thrashes between the same few experts. With ghost
//! history a re-admitted expert keeps everything it earned and is hard to
//! evict again, which is where the entire LFU-over-LRU margin comes from
//! (EXP-005, 556 decode tokens, 10 slots/layer: ghost-history LFU 44.8%,
//! windowed 44.7%, aged 43.0%, per-slot 42.6%, LRU 42.6%, Belady 55.8%).
//!
//! # Lifecycle
//!
//! ```text
//! plan()          -> miss: slot is Filling  (protected)
//! mark_ready(s)   -> read completed:  Ready (protected, bytes valid)
//! release(s)      -> compute done:    Idle  (evictable)
//! invalidate(s)   -> fill failed:     Empty (unassigned)   [unsafe]
//! ```
//!
//! `invalidate` is `unsafe` because it is the one transition that unprotects
//! a slot whose read may still be live; see [`LayerCache::invalidate`].
//! `release` refuses to unprotect a `Filling` slot for the same reason, and
//! [`LayerCache::stuck_protected_slot`] catches the opposite mistake, a slot
//! that is protected and never handed back. Both protected states count: a
//! `Ready` slot whose compute never released it and a `Filling` slot whose
//! completion was never reaped nor invalidated are the same bug from the
//! layer's point of view, and both end it in permanent
//! [`CacheError::AllSlotsBusy`].
//!
//! # What the counters mean
//!
//! [`CacheStats::hits`] counts requests served from bytes that are already
//! valid. A request that lands on a slot whose fill is still in flight is not
//! one of those: no second read is issued for it, but it does wait for a
//! read, so it is counted separately as [`CacheStats::pending_hits`] and left
//! out of [`CacheStats::hit_rate`]'s numerator. An expert counts as "fetched"
//! for the cold/eviction split only once a fill for it has actually completed
//! ([`LayerCache::mark_ready`]), so a failed read followed by `invalidate`
//! leaves the retry a cold miss instead of inflating the eviction share.
//!
//! Slot lookup is a linear scan over the layer's slots. The dial is a byte
//! budget — 1438 MiB today, which is 11 slots/layer on Qwen3-30B-A3B — and
//! [`MAX_SLOTS`] bounds it well above anything the memory contract allows, so
//! the scan is a handful of comparisons per lookup. Whether a reverse index
//! would earn back its invalidation cost at that size has not been measured;
//! this is reasoning, not a benchmark.

use thiserror::Error;

use crate::format::MAX_EXPERTS;

/// Largest slot count one layer may be built with.
///
/// The dial is a byte budget — 1438 MiB buys 11 slots/layer on
/// Qwen3-30B-A3B, and the largest count EXP-005 swept was 24 — so this cap is
/// nearly two orders above the operating point and never binds a real
/// configuration. It exists so that `n_slots` cannot make this module's
/// per-step linear scans, or its allocation, unbounded.
pub const MAX_SLOTS: u32 = 1024;

/// How many [`LayerCache::plan`] calls a slot may stay protected across
/// before it is treated as leaked by its owner.
///
/// A protected slot is one nothing may evict: `Ready` means queued compute
/// owns the buffer, `Filling` means a read is landing in it. The decode loop
/// discharges both within the same step. Surviving this many further steps of
/// the same layer means a caller stopped releasing, or stopped reaping; see
/// [`LayerCache::stuck_protected_slot`].
///
/// The count is of *attempted* plans, not successful ones. That is the whole
/// point: the leak's terminal state is a layer that can no longer serve any
/// step, so a clock that only ticked on success would stop exactly when the
/// detector is needed.
pub const STUCK_PROTECTED_PLANS: u64 = 64;

/// Errors from building a layer cache, or from planning one routing step
/// against its slots.
///
/// Every planning variant means the request could not be served as asked;
/// the cache is left exactly as it was, so the caller can drop the step or
/// retry after releasing slots. Nothing here panics.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CacheError {
    /// An empty expert id space: no request could ever name a valid expert.
    #[error("layer cache needs at least one expert")]
    NoExperts,

    /// The declared id space is past [`MAX_EXPERTS`].
    #[error("layer cache expert space {n_experts} is over the {limit} cap")]
    ExpertSpaceTooLarge {
        /// Experts the caller asked for.
        n_experts: u32,
        /// The cap that was exceeded.
        limit: u32,
    },

    /// The requested slot count is past [`MAX_SLOTS`].
    #[error("layer cache slot count {n_slots} is over the {limit} cap")]
    TooManySlots {
        /// Slots the caller asked for.
        n_slots: u32,
        /// The cap that was exceeded.
        limit: u32,
    },

    /// A routed expert id is at or past the layer's expert count.
    #[error("expert {expert} out of range ({n_experts} experts)")]
    ExpertOutOfRange {
        /// The offending id.
        expert: u32,
        /// Experts per layer this cache was built for.
        n_experts: u32,
    },

    /// One routing step asked for more distinct experts than the layer has
    /// slots, so they cannot be resident at the same time. Raise
    /// `slots_per_layer` to at least the model's `top_k`.
    #[error("{requested} experts requested in one step, layer has {n_slots} slots")]
    TooFewSlots {
        /// Slots in this layer.
        n_slots: u32,
        /// Distinct experts the step asked for.
        requested: usize,
    },

    /// Room exists in principle but every candidate slot is protected: it is
    /// filling from an in-flight read, still owned by queued compute, or
    /// holds another expert this same step needs. Evicting one would alias a
    /// buffer under a live O_DIRECT read, so the step fails instead.
    ///
    /// (What aliasing costs on btrfs — spurious EIO from checksum failure —
    /// was measured during io bring-up but is not yet an entry in
    /// `docs/experiments/README.md`; treat the figure quoted on
    /// [`CachePlan::hits`] as provisional.)
    #[error(
        "layer needs {needed} more slot(s) but only {free} of {n_slots} are free \
         (the rest are filling or in use)"
    )]
    AllSlotsBusy {
        /// Slots in this layer.
        n_slots: u32,
        /// Slots that could have been filled or evicted.
        free: usize,
        /// Slots the step needed on top of its hits.
        needed: usize,
    },
}

/// Where one slot stands between a planned read and the compute that uses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    /// No expert assigned. First choice for a fill, never a victim to evict.
    Empty,
    /// Assigned, with a read in flight into the buffer. Protected.
    Filling,
    /// Assigned and readable, still owned by queued compute. Protected.
    Ready,
    /// Assigned and readable, owned by nobody. The only eviction candidate.
    Idle,
}

/// One entry of a layer's slot array. Buffers live in the slot pool; this is
/// only the bookkeeping that decides who owns which buffer.
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// Expert resident in (or being read into) this slot. `None` iff
    /// `state == Empty`.
    expert: Option<u32>,
    /// Logical clock of the last routing decision that named this slot.
    last_used: u64,
    /// Fill and ownership state.
    state: SlotState,
    /// `plan` attempt at which this slot last *entered* a protected state,
    /// `Filling` or `Ready`. Deliberately not refreshed by a hit on a slot
    /// that is already `Ready`, so a slot that is requested every step but
    /// never released still ages out; see
    /// [`LayerCache::stuck_protected_slot`]. Each protected phase gets its
    /// own budget: a `Filling` slot whose completion is reaped restarts the
    /// clock as it enters `Ready`, because the read really did finish and
    /// what owns the buffer now is different. Meaningless while `Idle` or
    /// `Empty`.
    protected_at: u64,
}

impl Slot {
    /// An unassigned slot.
    const EMPTY: Self = Self {
        expert: None,
        last_used: 0,
        state: SlotState::Empty,
        protected_at: 0,
    };
}

/// What one routing step must do: which experts are already in which slots,
/// and which slots the caller has to read into.
///
/// Reuse one buffer across the decode loop. [`LayerCache::plan`] clears it
/// and refills it, so after the first few steps it allocates nothing.
#[derive(Debug, Clone, Default)]
pub struct CachePlan {
    /// `(expert, slot)` for experts already resident.
    hits: Vec<(u32, u32)>,
    /// `(expert, slot)` for experts the caller must read.
    misses: Vec<(u32, u32)>,
}

impl CachePlan {
    /// An empty plan that will grow into whatever the first step needs.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty plan preallocated for `top_k` experts per step.
    pub fn with_capacity(top_k: usize) -> Self {
        Self {
            hits: Vec::with_capacity(top_k),
            misses: Vec::with_capacity(top_k),
        }
    }

    /// Resident experts, as `(expert, slot)` in request order.
    ///
    /// A hit may name a slot whose fill is still in flight: `plan` never
    /// issues a second read for an expert that already owns a slot, because
    /// two O_DIRECT reads aliasing one buffer make btrfs report spurious EIO
    /// (provisional: 13-27% during io bring-up, not yet recorded in
    /// `docs/experiments/README.md`). Callers that plan more than once per
    /// step must therefore wait for outstanding completions (see
    /// [`LayerCache::is_ready`]) before computing on a hit. Those requests
    /// are counted as [`CacheStats::pending_hits`], not as hits.
    pub fn hits(&self) -> &[(u32, u32)] {
        &self.hits
    }

    /// Experts to read, as `(expert, slot)` in request order. Each slot is
    /// `Filling` and protected until [`LayerCache::invalidate`] or a
    /// [`LayerCache::mark_ready`] + [`LayerCache::release`] pair.
    pub fn misses(&self) -> &[(u32, u32)] {
        &self.misses
    }

    /// Distinct experts the step resolved.
    pub fn len(&self) -> usize {
        self.hits.len() + self.misses.len()
    }

    /// Whether the step resolved nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop both lists, keeping their capacity.
    pub fn clear(&mut self) {
        self.hits.clear();
        self.misses.clear();
    }

    /// Entries either list can hold before it reallocates.
    ///
    /// The *smaller* of the two, which is what
    /// [`with_capacity`](Self::with_capacity) actually guarantees: a plan
    /// preallocated for `top_k` reports at least `top_k` here whatever the
    /// hit/miss split of the steps it has served. Exists so that "the decode
    /// loop does not allocate" is assertable directly rather than inferred
    /// from the addresses the two `Vec`s happen to hold.
    pub fn capacity(&self) -> usize {
        self.hits.capacity().min(self.misses.capacity())
    }
}

/// Cumulative cache telemetry for one layer.
///
/// The cold/eviction split is the number that says whether the slot budget
/// is wrong: cold misses are unavoidable at any cache size, eviction misses
/// are what more slots would buy. At 10 slots/layer, 87.9% of misses were
/// eviction misses (EXP-005), and `docs/architecture.md` argues from that
/// share, so the split has to be honest in both directions. It is only
/// meaningful if the caller reports completions through
/// [`LayerCache::mark_ready`]: an expert becomes "fetched" when a read for it
/// lands, never merely because one was planned. A caller that never marks
/// anything ready counts every miss as cold.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Requested experts found resident with bytes already valid.
    pub hits: u64,
    /// Requested experts whose slot was still filling from a read an earlier
    /// step issued. No second read is issued for these, but they are not
    /// served without one either, so they are neither hits nor misses.
    pub pending_hits: u64,
    /// Requested experts that had to be read.
    pub misses: u64,
    /// Misses for an expert this layer had never successfully fetched.
    pub cold_misses: u64,
    /// Misses for an expert this layer had fetched and evicted.
    pub eviction_misses: u64,
    /// Occupied slots reclaimed to make room.
    pub evictions: u64,
}

impl CacheStats {
    /// Requested experts resolved: hits, pending hits, and misses.
    pub fn accesses(&self) -> u64 {
        self.hits + self.pending_hits + self.misses
    }

    /// Fraction of requests served without waiting on a read; `0.0` before
    /// any request.
    ///
    /// Pending hits are in the denominator and not the numerator: they wait
    /// on a read like a miss does, they just do not add one. This is the
    /// number compared against `scripts/lfu_sim.py`, which has no in-flight
    /// state and therefore never produces a pending hit — so on the runtime's
    /// own one-plan-per-step usage the two accountings agree exactly.
    pub fn hit_rate(&self) -> f64 {
        let accesses = self.accesses();
        if accesses == 0 {
            0.0
        } else {
            self.hits as f64 / accesses as f64
        }
    }
}

/// One layer's expert slot array plus its ghost-history frequency counters.
///
/// See the module docs for the policy. Construction allocates
/// `n_slots` slot records and `n_experts` counters and nothing after that:
/// planning a step is a linear scan over the slots.
#[derive(Debug, Clone)]
pub struct LayerCache {
    /// Slot bookkeeping, indexed by slot number.
    slots: Vec<Slot>,
    /// Ghost history: request count per expert id, surviving eviction.
    freq: Vec<u32>,
    /// Whether a read for each expert has ever *completed* in this layer, for
    /// the cold split. Set by [`LayerCache::mark_ready`], never by planning a
    /// fill that might still fail.
    fetched: Vec<bool>,
    /// Ticks once per distinct requested expert; breaks frequency ties by
    /// recency.
    clock: u64,
    /// [`LayerCache::plan`] calls *attempted*, for the stuck-slot check.
    /// Counts rejected steps too; see [`STUCK_PROTECTED_PLANS`].
    plan_attempts: u64,
    /// Cumulative telemetry.
    stats: CacheStats,
}

impl LayerCache {
    /// A cache of `n_slots` slots over an id space of `n_experts` experts.
    ///
    /// `n_slots` must be at least the model's `top_k` for any step to be
    /// satisfiable; a smaller value, zero included, is accepted here and
    /// reported as [`CacheError::TooFewSlots`] when a step arrives.
    ///
    /// Both sizes are bounded here rather than assumed. `n_experts` reaches
    /// this module from `manifest.json`, whose validation already caps it at
    /// [`MAX_EXPERTS`] — but that is an upstream invariant this type cannot
    /// see, and sizing allocations from an unchecked `u32` with no error path
    /// is exactly what [`SlotPool::new`](crate::io::SlotPool::new) refuses to
    /// do for the same class of input.
    ///
    /// # Errors
    ///
    /// [`CacheError::NoExperts`] for an empty id space,
    /// [`CacheError::ExpertSpaceTooLarge`] past [`MAX_EXPERTS`], and
    /// [`CacheError::TooManySlots`] past [`MAX_SLOTS`].
    pub fn new(n_slots: u32, n_experts: u32) -> Result<Self, CacheError> {
        if n_experts == 0 {
            return Err(CacheError::NoExperts);
        }
        if n_experts > MAX_EXPERTS {
            return Err(CacheError::ExpertSpaceTooLarge {
                n_experts,
                limit: MAX_EXPERTS,
            });
        }
        if n_slots > MAX_SLOTS {
            return Err(CacheError::TooManySlots {
                n_slots,
                limit: MAX_SLOTS,
            });
        }
        Ok(Self {
            slots: vec![Slot::EMPTY; n_slots as usize],
            freq: vec![0; n_experts as usize],
            fetched: vec![false; n_experts as usize],
            clock: 0,
            plan_attempts: 0,
            stats: CacheStats::default(),
        })
    }

    /// Slots in this layer.
    pub fn n_slots(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Size of the expert id space.
    pub fn n_experts(&self) -> u32 {
        self.freq.len() as u32
    }

    /// Cumulative telemetry since construction.
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Whether `slot` holds bytes a kernel may read. False for a slot that
    /// is empty, still filling, or out of range.
    pub fn is_ready(&self, slot: u32) -> bool {
        self.slots
            .get(slot as usize)
            .is_some_and(|s| matches!(s.state, SlotState::Ready | SlotState::Idle))
    }

    /// The expert resident in (or being read into) `slot`, if any.
    pub fn resident(&self, slot: u32) -> Option<u32> {
        self.slots.get(slot as usize).and_then(|s| s.expert)
    }

    /// Resolve one routing decision into hits and slot assignments.
    ///
    /// `experts` is the top-k the router selected for this layer and token,
    /// in routed order. `out` is cleared first and refilled; reuse one
    /// buffer to keep the decode loop allocation-free. Every distinct id
    /// bumps its frequency counter once, hit or miss; duplicates within one
    /// call resolve once and are listed once.
    ///
    /// Each entry of [`CachePlan::misses`] names a slot the caller must read
    /// the expert's blob into. The slot is protected from eviction until it
    /// is released, so misses from a later step can never land on it.
    ///
    /// # Errors
    ///
    /// [`CacheError::ExpertOutOfRange`] for an id outside the id space,
    /// [`CacheError::TooFewSlots`] when one step needs more slots than the
    /// layer has, and [`CacheError::AllSlotsBusy`] when every slot that
    /// could have taken a miss is filling or in use. All three are checked
    /// before anything is evicted: on error no slot, counter, or statistic
    /// changes and `out` is empty.
    ///
    /// The single exception is the staleness clock behind
    /// [`stuck_protected_slot`](Self::stuck_protected_slot), which a rejected
    /// step advances like any other. It has to: a layer whose slots have all
    /// leaked answers `AllSlotsBusy` to *every* step, so a clock that only
    /// ran on servable steps would freeze in exactly the state the detector
    /// exists to report. Nothing else reads it, and it is not cache content.
    ///
    /// # Panics
    ///
    /// In debug builds only, if a slot has been protected across more than
    /// [`STUCK_PROTECTED_PLANS`] attempts — the caller-side leak
    /// [`stuck_protected_slot`](Self::stuck_protected_slot) describes. The
    /// check is compiled out of release builds, where the same condition is
    /// a query rather than a panic.
    pub fn plan(&mut self, experts: &[u32], out: &mut CachePlan) -> Result<(), CacheError> {
        self.debug_assert_nothing_stuck_protected();
        out.clear();
        // Before `check`, so the clock keeps running once the layer stops
        // being able to serve anything. Wrapping for the same reason the
        // logical clock below wraps.
        self.plan_attempts = self.plan_attempts.wrapping_add(1);
        self.check(experts)?;

        for (i, &expert) in experts.iter().enumerate() {
            if experts[..i].contains(&expert) {
                // The router should not route one expert twice in a step; if
                // it does, the first occurrence already resolved it.
                continue;
            }
            // Wrapping: 2^32 routing decisions for one expert in one layer is
            // unreachable, and a panic here would be worse than a wrap.
            self.clock = self.clock.wrapping_add(1);
            let counter = &mut self.freq[expert as usize];
            *counter = counter.wrapping_add(1);

            match self.find(expert) {
                Some(slot) => {
                    let clock = self.clock;
                    let plan_attempts = self.plan_attempts;
                    let entry = &mut self.slots[slot];
                    entry.last_used = clock;
                    // Recorded before the promotion below, which would
                    // otherwise hide it.
                    let still_filling = entry.state == SlotState::Filling;
                    if entry.state == SlotState::Idle {
                        // Queued compute owns it again for this step.
                        entry.state = SlotState::Ready;
                        entry.protected_at = plan_attempts;
                    }
                    out.hits.push((expert, slot as u32));
                    if still_filling {
                        // The expert owns a slot, so no second read is issued
                        // for it — but its bytes are not there yet, so this
                        // request was not served without a read either.
                        self.stats.pending_hits += 1;
                    } else {
                        self.stats.hits += 1;
                    }
                }
                None => {
                    // `check` proved a victim exists; the error keeps the
                    // promise that this crate never panics on bad input.
                    let slot = self
                        .choose_victim(experts)
                        .ok_or(CacheError::AllSlotsBusy {
                            n_slots: self.n_slots(),
                            free: 0,
                            needed: 1,
                        })?;
                    let clock = self.clock;
                    let plan_attempts = self.plan_attempts;
                    let entry = &mut self.slots[slot];
                    if entry.expert.is_some() {
                        self.stats.evictions += 1;
                    }
                    *entry = Slot {
                        expert: Some(expert),
                        last_used: clock,
                        state: SlotState::Filling,
                        // `Filling` is protected too, so the clock starts
                        // here rather than at `mark_ready`: a completion that
                        // is never reaped strands the slot just as a hit that
                        // is never released does.
                        protected_at: plan_attempts,
                    };
                    out.misses.push((expert, slot as u32));
                    self.stats.misses += 1;
                    // Classified against reads that *completed*, not against
                    // reads that were planned. A fill that failed and was
                    // invalidated leaves its expert un-fetched, so the retry
                    // is another cold miss rather than a phantom eviction
                    // miss arguing for a bigger slot budget.
                    if self.fetched.get(expert as usize).copied().unwrap_or(false) {
                        self.stats.eviction_misses += 1;
                    } else {
                        self.stats.cold_misses += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Record that the read into `slot` completed and its bytes are valid.
    ///
    /// This is also what makes the slot's expert count as fetched for the
    /// cold/eviction split, so the caller owes one call per completed fill if
    /// [`CacheStats`] is to mean anything.
    ///
    /// The slot stays protected: queued compute still owns it until
    /// [`release`](Self::release). Out-of-range or unassigned slots are
    /// ignored.
    pub fn mark_ready(&mut self, slot: u32) {
        let plan_attempts = self.plan_attempts;
        let Some(entry) = self.slots.get_mut(slot as usize) else {
            tracing::warn!(slot, "mark_ready for a slot outside the layer");
            return;
        };
        let completed = match entry.state {
            SlotState::Filling => {
                entry.state = SlotState::Ready;
                // A new protected phase, with its own budget: the read is
                // over and what protects the slot now is queued compute.
                entry.protected_at = plan_attempts;
                entry.expert
            }
            // Already readable: a duplicate completion, harmless.
            SlotState::Ready | SlotState::Idle => None,
            SlotState::Empty => {
                tracing::warn!(slot, "mark_ready for an unassigned slot");
                None
            }
        };
        // `get_mut` rather than indexing: resident ids were range-checked by
        // `check`, but this method is public and must not panic if that ever
        // stops being true.
        if let Some(expert) = completed
            && let Some(fetched) = self.fetched.get_mut(expert as usize)
        {
            *fetched = true;
        }
    }

    /// Hand `slot` back: its expert stays resident but the slot becomes an
    /// eviction candidate.
    ///
    /// Call once compute for the step has finished with the buffer, for the
    /// plan's hits as well as its misses. Releasing only the misses leaks
    /// every hit slot into permanent protection, which
    /// [`stuck_protected_slot`](Self::stuck_protected_slot) detects and
    /// `plan` asserts against in debug builds.
    ///
    /// A slot with a read still in flight is *not* released - dropping that
    /// protection is what corrupts O_DIRECT reads - so a failed fill must go
    /// through [`invalidate`](Self::invalidate) instead. Out-of-range and
    /// empty slots are ignored.
    pub fn release(&mut self, slot: u32) {
        let Some(entry) = self.slots.get_mut(slot as usize) else {
            tracing::warn!(slot, "release for a slot outside the layer");
            return;
        };
        match entry.state {
            SlotState::Ready => entry.state = SlotState::Idle,
            SlotState::Filling => {
                tracing::warn!(
                    slot,
                    "release with a fill still in flight, keeping it protected"
                );
            }
            SlotState::Idle | SlotState::Empty => {}
        }
    }

    /// Drop `slot`'s assignment and hand it back unassigned.
    ///
    /// For a fill that failed: the buffer holds nothing usable, so the expert
    /// must not stay marked resident, and the slot's expert is left
    /// un-fetched so a retry counts as the cold miss it is. Out-of-range
    /// slots are ignored.
    ///
    /// # Safety
    ///
    /// The read into this slot's buffer must have completed, or been
    /// cancelled and reaped, before this is called.
    ///
    /// This is the one transition that unprotects a slot without proof that
    /// its read is over, and it resets the slot to `Empty`, which makes it
    /// the *first* pick in the next step's victim search. It is also the
    /// natural error path for a failed fill — precisely when the read's state
    /// is least certain. If the kernel may still write into the buffer,
    /// handing that slot to the next miss puts two O_DIRECT reads on one
    /// destination, which is the corruption the whole slot discipline exists
    /// to prevent. Callers that cannot prove the read is over must cancel it
    /// first, or leak the slot
    /// ([`SlotGuard::leak`](crate::io::SlotGuard::leak)) and never call this.
    pub unsafe fn invalidate(&mut self, slot: u32) {
        let Some(entry) = self.slots.get_mut(slot as usize) else {
            tracing::warn!(slot, "invalidate for a slot outside the layer");
            return;
        };
        *entry = Slot::EMPTY;
    }

    /// Drop every slot's assignment at once, keeping the ghost history.
    ///
    /// [`invalidate`](Self::invalidate) for the whole layer: every slot goes
    /// back to `Empty`, and `freq`/`fetched` — which are indexed by expert id
    /// and survive eviction on purpose — are untouched, so the next miss on an
    /// expert this layer had fetched is still counted as the eviction miss it
    /// is.
    ///
    /// For the prefill sweep, which borrows every buffer of the pool at once
    /// ([`ExpertStream::sweep_layer`](crate::io::ExpertStream::sweep_layer)):
    /// the obligation below is one statement about one moment, and stating it
    /// once per layer is both cheaper and more honest than restating it per
    /// slot.
    ///
    /// # Safety
    ///
    /// [`invalidate`](Self::invalidate)'s contract, for every slot of the
    /// layer: no read may be in flight into *any* of this layer's buffers.
    /// Unprotecting a slot the kernel is still writing into makes it the first
    /// pick of the next victim search, which puts two O_DIRECT reads on one
    /// destination — the corruption the slot discipline exists to prevent.
    pub unsafe fn reset_occupancy(&mut self) {
        self.slots.fill(Slot::EMPTY);
    }

    /// The first slot that has been protected — `Filling` or `Ready` —
    /// across more than [`STUCK_PROTECTED_PLANS`] attempted
    /// [`LayerCache::plan`] calls, if any.
    ///
    /// This is the leak `release`'s docs warn about, made detectable, in both
    /// of the shapes it takes:
    ///
    /// - `Ready`: a hit on an `Idle` slot promotes it back to `Ready`, so a
    ///   caller that releases [`CachePlan::misses`] but not
    ///   [`CachePlan::hits`] protects that slot forever. The stamp is only
    ///   refreshed when a slot *enters* a protected state, so a slot that is
    ///   hit every step and never released still ages out and is reported.
    /// - `Filling`: a completion that is never reported through
    ///   [`mark_ready`](Self::mark_ready) and never dropped through
    ///   [`invalidate`](Self::invalidate) leaves the slot filling forever.
    ///   That is equally unevictable and equally fatal, so it is equally
    ///   reported.
    ///
    /// Either way the layer degrades to permanent
    /// [`CacheError::AllSlotsBusy`] — which is the state this has to keep
    /// working in, and the reason the clock counts attempted plans rather
    /// than servable ones. Once the leak is total, every step fails; a
    /// success-only clock would stop at that exact moment and leave a dead
    /// layer undiagnosed.
    ///
    /// `plan` checks this in debug builds and panics; in release it is a
    /// query the caller can poll.
    pub fn stuck_protected_slot(&self) -> Option<u32> {
        self.slots
            .iter()
            .position(|slot| {
                matches!(slot.state, SlotState::Filling | SlotState::Ready)
                    && self.plan_attempts.wrapping_sub(slot.protected_at) > STUCK_PROTECTED_PLANS
            })
            // Bounded by MAX_SLOTS.
            .map(|slot| slot as u32)
    }

    /// Debug-only guard against the leak
    /// [`LayerCache::stuck_protected_slot`] describes. Compiled out of
    /// release builds.
    fn debug_assert_nothing_stuck_protected(&self) {
        debug_assert!(
            self.stuck_protected_slot().is_none(),
            "slot {:?} has been protected across more than {} plan attempts: a \
             caller is releasing misses but not hits, or is not reaping its \
             completions, and this layer will degrade to permanent AllSlotsBusy",
            self.stuck_protected_slot(),
            STUCK_PROTECTED_PLANS,
        );
    }

    /// Validate a request and prove it can be served without touching a
    /// protected slot, before any state changes.
    fn check(&self, experts: &[u32]) -> Result<(), CacheError> {
        let n_experts = self.n_experts();
        for &expert in experts {
            if expert >= n_experts {
                return Err(CacheError::ExpertOutOfRange { expert, n_experts });
            }
        }

        let mut needed = 0usize;
        let mut distinct = 0usize;
        for (i, &expert) in experts.iter().enumerate() {
            if experts[..i].contains(&expert) {
                continue;
            }
            distinct += 1;
            if self.find(expert).is_none() {
                needed += 1;
            }
        }
        if distinct > self.slots.len() {
            return Err(CacheError::TooFewSlots {
                n_slots: self.n_slots(),
                requested: distinct,
            });
        }

        let free = self
            .slots
            .iter()
            .filter(|s| self.is_evictable(s, experts))
            .count();
        if needed > free {
            return Err(CacheError::AllSlotsBusy {
                n_slots: self.n_slots(),
                free,
                needed,
            });
        }
        Ok(())
    }

    /// Whether a miss may take this slot: empty, or idle and not holding an
    /// expert the same step needs.
    fn is_evictable(&self, slot: &Slot, requested: &[u32]) -> bool {
        match slot.state {
            SlotState::Empty => true,
            SlotState::Idle => slot.expert.is_none_or(|e| !requested.contains(&e)),
            SlotState::Filling | SlotState::Ready => false,
        }
    }

    /// Index of the slot holding `expert`, resident or filling.
    fn find(&self, expert: u32) -> Option<usize> {
        self.slots.iter().position(|s| s.expert == Some(expert))
    }

    /// The slot a miss should take: the first empty one, else the evictable
    /// slot with the lowest ghost-history count, ties going to the least
    /// recently used and then to the lowest index.
    fn choose_victim(&self, requested: &[u32]) -> Option<usize> {
        if let Some(empty) = self
            .slots
            .iter()
            .position(|s| s.state == SlotState::Empty && s.expert.is_none())
        {
            return Some(empty);
        }
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| self.is_evictable(s, requested))
            .min_by_key(|(i, s)| {
                let freq = s.expert.map_or(0, |e| self.freq[e as usize]);
                (freq, s.last_used, *i)
            })
            .map(|(i, _)| i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache whose geometry is in range; the bounds themselves are tested
    /// in `construction_bounds_the_slot_count_and_the_id_space`.
    fn new_cache(n_slots: u32, n_experts: u32) -> LayerCache {
        LayerCache::new(n_slots, n_experts).expect("test geometry is in range")
    }

    /// Drive one step and hand every slot it touched straight back, the way
    /// the decode loop does once the layer's FFN has run.
    fn step(cache: &mut LayerCache, experts: &[u32]) -> CachePlan {
        let mut plan = CachePlan::with_capacity(experts.len());
        cache.plan(experts, &mut plan).expect("step is satisfiable");
        for &(_, slot) in plan.misses() {
            cache.mark_ready(slot);
        }
        for &(_, slot) in plan.hits().iter().chain(plan.misses()) {
            cache.release(slot);
        }
        plan
    }

    /// Slot each expert landed in, whether it hit or missed.
    fn slot_of(plan: &CachePlan, expert: u32) -> u32 {
        plan.hits()
            .iter()
            .chain(plan.misses())
            .find(|(e, _)| *e == expert)
            .map(|(_, s)| *s)
            .expect("expert is in the plan")
    }

    /// Whether any slot currently holds `expert`.
    fn holds(cache: &LayerCache, expert: u32) -> bool {
        (0..cache.n_slots()).any(|slot| cache.resident(slot) == Some(expert))
    }

    #[test]
    fn hit_returns_the_resident_slot_and_leaves_residency_alone() {
        let mut cache = new_cache(4, 8);
        let first = step(&mut cache, &[3]);
        assert_eq!(first.misses(), [(3, 0)]);
        assert!(first.hits().is_empty());

        let second = step(&mut cache, &[3]);
        assert_eq!(second.hits(), [(3, 0)]);
        assert!(second.misses().is_empty());
        assert_eq!(cache.resident(0), Some(3));
        assert_eq!(cache.stats().evictions, 0);
        // Nothing else was disturbed.
        for slot in 1..4 {
            assert_eq!(cache.resident(slot), None);
        }
    }

    #[test]
    fn cold_misses_fill_empty_slots_before_evicting() {
        let mut cache = new_cache(4, 8);
        let plan = step(&mut cache, &[5, 1, 7]);
        assert_eq!(plan.misses(), [(5, 0), (1, 1), (7, 2)]);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().cold_misses, 3);
        assert_eq!(cache.resident(3), None);
    }

    #[test]
    fn eviction_picks_the_lowest_frequency() {
        let mut cache = new_cache(3, 8);
        // Counts after this: 0 -> 3, 1 -> 2, 2 -> 1.
        for _ in 0..3 {
            step(&mut cache, &[0]);
        }
        for _ in 0..2 {
            step(&mut cache, &[1]);
        }
        step(&mut cache, &[2]);

        let plan = step(&mut cache, &[4]);
        // Expert 2 is the coldest, and it is not the least recently used
        // (that is expert 0), so this is frequency deciding, not recency.
        assert_eq!(plan.misses(), [(4, 2)]);
        assert_eq!(cache.resident(0), Some(0));
        assert_eq!(cache.resident(1), Some(1));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn frequency_ties_break_on_recency() {
        let mut cache = new_cache(3, 8);
        // All three end on a count of 2; expert 0 was used longest ago.
        for expert in [0, 1, 2] {
            step(&mut cache, &[expert]);
        }
        for expert in [0, 1, 2] {
            step(&mut cache, &[expert]);
        }
        // Counts are 2 across the board and the logical clock ticked in
        // request order, so expert 0 is the least recently used.
        let plan = step(&mut cache, &[6]);
        assert_eq!(plan.misses(), [(6, 0)], "least recently used loses the tie");
        assert_eq!(cache.resident(1), Some(1));
        assert_eq!(cache.resident(2), Some(2));
    }

    /// The justification for the whole file: counters indexed by expert, not
    /// by slot, pick a different victim than per-slot counters would.
    ///
    /// Script (3 slots), each step released before the next:
    ///
    /// ```text
    /// step   ghost counters          per-slot counters      residency
    /// 0 x5   0:5                     0:5                    s0=0
    /// 1 x3   0:5 1:3                 0:5 1:3                s1=1
    /// 2 x3   0:5 1:3 2:3             0:5 1:3 2:3            s2=2
    /// 3      evict 1: count 3 ties   same                   s1=3
    ///        with 2, and 1 is LRU
    /// 1      evict 3: count 1        same                   s1=1
    ///        1 comes back at 4       1 is reset to 1
    /// 4      lowest is 2, at 3       lowest is 1, at 1
    ///        -> evicts expert 2      -> would evict expert 1
    /// ```
    #[test]
    fn ghost_history_changes_the_victim() {
        let mut cache = new_cache(3, 8);
        for _ in 0..5 {
            step(&mut cache, &[0]);
        }
        for _ in 0..2 {
            step(&mut cache, &[1]);
        }
        for _ in 0..2 {
            step(&mut cache, &[2]);
        }
        let slot_1 = slot_of(&step(&mut cache, &[1]), 1);
        let slot_2 = slot_of(&step(&mut cache, &[2]), 2);

        // Expert 1 is evicted, then immediately re-requested.
        let evicted = step(&mut cache, &[3]);
        assert_eq!(evicted.misses(), [(3, slot_1)], "coldest and LRU loses");
        let readmitted = step(&mut cache, &[1]);
        assert_eq!(
            readmitted.misses(),
            [(1, slot_1)],
            "expert 3 is now coldest"
        );

        // Ghost history put expert 1 back at 3 requests; a per-slot counter
        // would have reset it to 1 and made it the obvious victim here.
        let plan = step(&mut cache, &[4]);
        assert_eq!(
            plan.misses(),
            [(4, slot_2)],
            "expert 2 (count 3) is evicted, not the re-admitted expert 1 (count 4)"
        );
        assert_eq!(
            cache.resident(slot_1),
            Some(1),
            "re-admitted expert survives"
        );
    }

    #[test]
    fn filling_and_in_use_slots_are_never_victims() {
        let mut cache = new_cache(2, 8);
        let mut plan = CachePlan::new();
        cache
            .plan(&[0, 1], &mut plan)
            .expect("two slots, two experts");
        assert_eq!(plan.misses(), [(0, 0), (1, 1)]);

        // Both slots are filling: nothing may be evicted.
        let err = cache.plan(&[2], &mut plan).unwrap_err();
        assert!(matches!(err, CacheError::AllSlotsBusy { free: 0, .. }));

        // Bytes have landed but compute still owns them: still protected.
        cache.mark_ready(0);
        cache.mark_ready(1);
        let err = cache.plan(&[2], &mut plan).unwrap_err();
        assert!(matches!(err, CacheError::AllSlotsBusy { free: 0, .. }));

        // One release is enough to make progress.
        cache.release(0);
        cache.plan(&[2], &mut plan).expect("slot 0 is idle now");
        assert_eq!(plan.misses(), [(2, 0)]);
        assert_eq!(cache.resident(1), Some(1));
    }

    #[test]
    fn a_slot_holding_an_expert_this_step_needs_is_not_a_victim() {
        let mut cache = new_cache(2, 8);
        // Expert 0 is hot, expert 1 is cold and resident.
        for _ in 0..3 {
            step(&mut cache, &[0]);
        }
        step(&mut cache, &[1]);

        // Expert 1 is the cheapest thing to evict, but this step needs it.
        let plan = step(&mut cache, &[2, 1]);
        assert_eq!(plan.hits(), [(1, 1)]);
        assert_eq!(
            plan.misses(),
            [(2, 0)],
            "the hot but unneeded expert 0 goes"
        );
    }

    #[test]
    fn release_of_a_filling_slot_keeps_it_protected() {
        let mut cache = new_cache(1, 8);
        let mut plan = CachePlan::new();
        cache.plan(&[0], &mut plan).expect("empty cache");
        cache.release(0);
        assert!(!cache.is_ready(0));
        let err = cache.plan(&[1], &mut plan).unwrap_err();
        assert!(matches!(err, CacheError::AllSlotsBusy { .. }));

        cache.mark_ready(0);
        assert!(cache.is_ready(0));
        cache.release(0);
        cache.plan(&[1], &mut plan).expect("slot is idle");
        assert_eq!(plan.misses(), [(1, 0)]);
    }

    #[test]
    fn invalidate_frees_a_failed_fill() {
        let mut cache = new_cache(1, 8);
        let mut plan = CachePlan::new();
        cache.plan(&[0], &mut plan).expect("empty cache");
        // SAFETY: no read was ever submitted for this slot in a unit test, so
        // the buffer is not a live I/O destination.
        unsafe { cache.invalidate(0) };
        assert_eq!(cache.resident(0), None);
        assert!(!cache.is_ready(0));

        cache.plan(&[1], &mut plan).expect("slot is free again");
        assert_eq!(plan.misses(), [(1, 0)]);
        assert_eq!(
            cache.stats().evictions,
            0,
            "an empty slot is not an eviction"
        );
    }

    #[test]
    fn too_few_slots_is_a_typed_error() {
        let mut cache = new_cache(2, 8);
        let mut plan = CachePlan::new();
        let err = cache.plan(&[0, 1, 2], &mut plan).unwrap_err();
        assert_eq!(
            err,
            CacheError::TooFewSlots {
                n_slots: 2,
                requested: 3
            }
        );
        assert!(plan.is_empty());
        assert_eq!(cache.stats(), CacheStats::default(), "nothing was touched");
    }

    #[test]
    fn an_out_of_range_expert_is_a_typed_error() {
        let mut cache = new_cache(4, 8);
        let mut plan = CachePlan::new();
        let err = cache.plan(&[1, 8], &mut plan).unwrap_err();
        assert_eq!(
            err,
            CacheError::ExpertOutOfRange {
                expert: 8,
                n_experts: 8
            }
        );
        assert!(plan.is_empty());
        assert_eq!(
            cache.resident(0),
            None,
            "the valid id was not admitted either"
        );
    }

    #[test]
    fn a_failed_step_changes_nothing() {
        let mut cache = new_cache(2, 8);
        let mut plan = CachePlan::new();
        cache
            .plan(&[0, 1], &mut plan)
            .expect("two slots, two experts");
        let before = cache.stats();

        assert!(cache.plan(&[2], &mut plan).is_err());
        assert!(plan.is_empty());
        assert_eq!(cache.stats(), before);
        assert_eq!(cache.resident(0), Some(0));
        assert_eq!(cache.resident(1), Some(1));
        // "Nothing" is slots, counters, and statistics. The staleness clock
        // is the documented exception and it did advance; see
        // `the_detector_still_fires_once_the_leak_wedges_the_whole_layer`
        // for why it has to.
    }

    #[test]
    fn duplicate_ids_in_one_step_resolve_once() {
        let mut cache = new_cache(4, 8);
        let plan = step(&mut cache, &[2, 2, 5]);
        assert_eq!(plan.misses(), [(2, 0), (5, 1)]);
        assert_eq!(plan.len(), 2);
        assert_eq!(cache.stats().misses, 2);
    }

    #[test]
    fn stats_track_the_cold_and_eviction_split() {
        let mut cache = new_cache(2, 8);
        step(&mut cache, &[0, 1]); // 2 cold misses
        step(&mut cache, &[0, 1]); // 2 hits
        step(&mut cache, &[2]); // 1 cold miss, evicts one of {0, 1}
        step(&mut cache, &[0, 1]); // 1 hit + 1 eviction miss, evicts expert 2
        step(&mut cache, &[2]); // 1 eviction miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 5);
        assert_eq!(stats.cold_misses, 3);
        assert_eq!(stats.eviction_misses, 2);
        assert_eq!(stats.accesses(), 8);
        assert_eq!(stats.evictions, 3);
        assert!((stats.hit_rate() - 0.375).abs() < 1e-12);
        assert_eq!(CacheStats::default().hit_rate(), 0.0);
        // One plan per step with every completion reported: the shape the
        // simulator models, and it produces no pending hits at all.
        assert_eq!(stats.pending_hits, 0);
        assert_eq!(stats.hits + stats.misses, stats.accesses());
    }

    /// A read that failed and was invalidated must not leave its expert
    /// marked as fetched: the retry is a *cold* miss, because the layer never
    /// held those bytes. Counting it as an eviction miss inflates the share
    /// `docs/architecture.md` uses to argue for buying more slots.
    #[test]
    fn a_failed_fill_leaves_the_retry_a_cold_miss() {
        let mut cache = new_cache(1, 8);
        let mut plan = CachePlan::new();

        cache.plan(&[5], &mut plan).expect("empty cache");
        assert_eq!(cache.stats().cold_misses, 1);
        // The read fails, so the fill is never marked ready.
        // SAFETY: the failed read has been reaped by the caller by this point.
        unsafe { cache.invalidate(0) };

        cache.plan(&[5], &mut plan).expect("slot is free again");
        let stats = cache.stats();
        assert_eq!(
            stats.cold_misses, 2,
            "expert 5 was never resident, so this is still a cold miss"
        );
        assert_eq!(
            stats.eviction_misses, 0,
            "an expert that was never fetched cannot have been evicted"
        );

        // Once a fill actually completes, the split flips over for real.
        cache.mark_ready(0);
        cache.release(0);
        step(&mut cache, &[6]); // evicts expert 5
        assert!(!holds(&cache, 5));
        step(&mut cache, &[5]);
        let stats = cache.stats();
        assert_eq!(stats.cold_misses, 3, "expert 6 was the third cold miss");
        assert_eq!(
            stats.eviction_misses, 1,
            "expert 5 had been resident this time, so its retry is an eviction miss"
        );
    }

    /// A request that lands on a slot whose read is still in flight issues no
    /// second read - but it was not served without one either, so it is not a
    /// hit. `hit_rate` is compared against the simulator's, which has no
    /// in-flight state, so the two must not disagree on what a hit is.
    #[test]
    fn a_request_on_an_in_flight_fill_is_not_counted_as_a_hit() {
        let mut cache = new_cache(2, 8);
        let mut plan = CachePlan::new();

        cache.plan(&[0], &mut plan).expect("empty cache");
        assert_eq!(plan.misses(), [(0, 0)]);

        // Same expert again before the fill completes.
        cache.plan(&[0], &mut plan).expect("expert 0 owns a slot");
        assert_eq!(
            plan.hits(),
            [(0, 0)],
            "the plan still reports it as resident, so no second read is issued"
        );
        assert!(
            plan.misses().is_empty(),
            "a second read would alias the buffer"
        );
        assert!(!cache.is_ready(0), "the bytes are not valid yet");

        let stats = cache.stats();
        assert_eq!(stats.hits, 0, "nothing has been served without a read");
        assert_eq!(stats.pending_hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.accesses(), 2);
        assert_eq!(stats.hit_rate(), 0.0);

        // Once the read lands, the same request is a real hit.
        cache.mark_ready(0);
        cache.plan(&[0], &mut plan).expect("expert 0 is resident");
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.pending_hits, 1);
        assert_eq!(stats.accesses(), 3);
        assert!((stats.hit_rate() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn construction_bounds_the_slot_count_and_the_id_space() {
        assert_eq!(LayerCache::new(12, 0).unwrap_err(), CacheError::NoExperts);
        assert_eq!(
            LayerCache::new(12, MAX_EXPERTS + 1).unwrap_err(),
            CacheError::ExpertSpaceTooLarge {
                n_experts: MAX_EXPERTS + 1,
                limit: MAX_EXPERTS,
            }
        );
        assert_eq!(
            LayerCache::new(u32::MAX, 128).unwrap_err(),
            CacheError::TooManySlots {
                n_slots: u32::MAX,
                limit: MAX_SLOTS,
            }
        );
        // The caps themselves are allowed, and so is a zero-slot layer, which
        // reports at `plan` time rather than at construction.
        LayerCache::new(MAX_SLOTS, MAX_EXPERTS).expect("the caps are inclusive");
        LayerCache::new(0, 8).expect("a zero-slot layer is legal");
        // The real dial, for scale: 11 slots over a 128-expert layer.
        let cache = LayerCache::new(11, 128).expect("the operating point");
        assert_eq!(cache.n_slots(), 11);
        assert_eq!(cache.n_experts(), 128);
    }

    /// C5, the leak `release` warns about: a caller that releases misses but
    /// not hits keeps promoting a slot back to `Ready` and never hands it
    /// back, so the layer runs out of victims for good.
    #[test]
    fn a_hit_slot_that_is_never_released_is_detected() {
        let mut cache = new_cache(2, 8);
        let mut plan = CachePlan::new();

        // Plan 1: expert 0 misses, the read lands, and the caller releases
        // it exactly once.
        cache.plan(&[0], &mut plan).expect("empty cache");
        cache.mark_ready(0);
        cache.release(0);

        // From here the caller only ever releases misses. Each step hits
        // expert 0, promoting its Idle slot back to Ready and leaving it
        // there. `plan` checks at entry, so stepping only while nothing is
        // stuck keeps this out of the debug assertion's way.
        let mut plans = 0u64;
        while cache.stuck_protected_slot().is_none() {
            cache.plan(&[0], &mut plan).expect("expert 0 is resident");
            assert!(plan.misses().is_empty(), "expert 0 stays resident");
            plans += 1;
            assert!(
                plans <= STUCK_PROTECTED_PLANS + 8,
                "the leaked Ready slot was never reported"
            );
        }
        assert_eq!(cache.stuck_protected_slot(), Some(0));
        assert!(
            plans > STUCK_PROTECTED_PLANS,
            "reported after only {plans} plans, before the slot could be called leaked"
        );
    }

    /// The state that leak ends in, which is the one the detector exists for:
    /// every slot `Ready`, nothing released, so every step that routes a
    /// non-resident expert fails and the layer is dead.
    ///
    /// This is the regression test for a clock that only advanced on a
    /// servable step. Under such a clock `plan_attempts` freezes at the first
    /// [`CacheError::AllSlotsBusy`], the stamps stay a handful of plans
    /// behind it forever, and the detector returns `None` for the rest of the
    /// process — silent at exactly the moment it is most needed.
    ///
    /// The shape is the operating point: 11 slots, 128 experts, top-8, and a
    /// caller that releases `misses()` but not `hits()`.
    #[test]
    fn the_detector_still_fires_once_the_leak_wedges_the_whole_layer() {
        const SLOTS: u32 = 11;
        let mut cache = new_cache(SLOTS, 128);
        let mut plan = CachePlan::new();

        // Four top-8 tokens are enough: each step's hits promote their Idle
        // slots back to Ready and strand them there, and the misses fill the
        // rest. Experts 0..11 land in slots 0..11 and never leave.
        let tokens: [[u32; 8]; 4] = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 0, 1, 2, 3, 4],
            [8, 9, 10, 0, 1, 2, 3, 4],
        ];
        for experts in tokens {
            cache.plan(&experts, &mut plan).expect("11 slots, top-8");
            for &(_, slot) in plan.misses() {
                cache.mark_ready(slot);
                cache.release(slot);
            }
        }
        assert_eq!(
            (0..SLOTS).filter(|&s| cache.resident(s).is_some()).count(),
            SLOTS as usize,
            "every slot is occupied"
        );
        assert_eq!(
            cache.stuck_protected_slot(),
            None,
            "four plans in, nothing has been protected long enough to be a leak"
        );

        // The layer is now wedged: expert 11 is not resident and no slot is
        // evictable, so this step - and every step like it - fails.
        let mut wedged = 0u64;
        while cache.stuck_protected_slot().is_none() {
            let err = cache.plan(&[11], &mut plan).unwrap_err();
            assert_eq!(
                err,
                CacheError::AllSlotsBusy {
                    n_slots: SLOTS,
                    free: 0,
                    needed: 1,
                },
                "the layer should be wedged, not merely busy"
            );
            wedged += 1;
            assert!(
                wedged <= STUCK_PROTECTED_PLANS + 8,
                "the layer has answered AllSlotsBusy for {wedged} consecutive \
                 steps and the detector has still not reported the leak"
            );
        }
        // The setup plans are on the same clock, and the oldest stamp was set
        // during them, so the wedged loop still has to carry nearly the whole
        // budget. Without this the loop above would also be satisfied by a
        // detector that fired immediately, for entirely the wrong reason.
        assert!(
            wedged >= STUCK_PROTECTED_PLANS - tokens.len() as u64,
            "reported after only {wedged} wedged steps"
        );
    }

    /// The other half of "protected and never handed back": a fill whose
    /// completion is never reported and never invalidated. The slot stays
    /// `Filling`, which is exactly as unevictable as a stranded `Ready` slot
    /// and exactly as fatal, so it has to be reported too.
    #[test]
    fn a_fill_whose_completion_is_never_reaped_is_detected() {
        let mut cache = new_cache(2, 8);
        let mut plan = CachePlan::new();

        cache.plan(&[0], &mut plan).expect("empty cache");
        assert_eq!(plan.misses(), [(0, 0)]);
        // Slot 0 is left Filling: no mark_ready, no invalidate. Nothing will
        // ever evict it again.

        // Meanwhile the caller keeps working on the layer's other slot, so
        // the clock has an honest reason to advance.
        let mut plans = 0u64;
        while cache.stuck_protected_slot().is_none() {
            cache.plan(&[1], &mut plan).expect("slot 1 serves expert 1");
            plans += 1;
            assert!(
                plans <= STUCK_PROTECTED_PLANS + 8,
                "the unreaped fill was never reported"
            );
        }
        assert_eq!(
            cache.stuck_protected_slot(),
            Some(0),
            "the Filling slot is the leaked one"
        );
    }

    #[test]
    fn a_well_behaved_caller_never_looks_stuck() {
        let mut cache = new_cache(4, 8);
        for i in 0..(STUCK_PROTECTED_PLANS * 3) {
            // Marks ready and releases hits and misses alike, as the decode
            // loop does, so no slot is ever left Filling or Ready between
            // steps.
            step(&mut cache, &[(i % 8) as u32]);
            assert_eq!(
                cache.stuck_protected_slot(),
                None,
                "false positive at step {i}"
            );
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "releasing misses but not hits")]
    fn plan_asserts_on_a_stuck_protected_slot_in_debug_builds() {
        let mut cache = new_cache(2, 8);
        let mut plan = CachePlan::new();
        cache.plan(&[0], &mut plan).expect("empty cache");
        cache.mark_ready(0);
        cache.release(0);
        // Keep hitting expert 0 and never release it again; the entry check
        // fires once the slot has been Ready across enough plans.
        for _ in 0..(STUCK_PROTECTED_PLANS + 8) {
            cache.plan(&[0], &mut plan).expect("expert 0 is resident");
        }
    }

    #[test]
    fn the_plan_buffer_is_reused_without_reallocating() {
        let mut cache = new_cache(12, 128);
        let mut plan = CachePlan::with_capacity(8);
        let steps: [[u32; 8]; 3] = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 9, 2, 11, 4, 13, 6, 15],
            [16, 1, 18, 3, 20, 5, 22, 7],
        ];
        for experts in steps {
            cache.plan(&experts, &mut plan).expect("12 slots, top-8");
            for &(_, slot) in plan.misses() {
                cache.mark_ready(slot);
            }
            for &(_, slot) in plan.hits().iter().chain(plan.misses()) {
                cache.release(slot);
            }
            assert_eq!(plan.len(), 8);
            assert_eq!(plan.hits.capacity(), 8);
            assert_eq!(plan.misses.capacity(), 8);
        }
    }

    #[test]
    fn counters_survive_but_never_decay() {
        let mut cache = new_cache(2, 4);
        // Expert 0 banks history, is evicted, and comes back cheaper to keep
        // than a newcomer even after a long gap.
        for _ in 0..6 {
            step(&mut cache, &[0]);
        }
        for _ in 0..40 {
            step(&mut cache, &[1, 2]);
        }
        assert!(!holds(&cache, 0), "expert 0 was pushed out");

        step(&mut cache, &[0]);
        // Expert 0 is back with 7 requests of history against 40 each for
        // experts 1 and 2, so the next newcomer takes expert 0's slot.
        let plan = step(&mut cache, &[3]);
        assert_eq!(plan.misses().len(), 1);
        assert_eq!(cache.resident(plan.misses()[0].1), Some(3));
        assert_ne!(
            cache.stats().eviction_misses,
            0,
            "the thrash is counted as eviction misses, not cold ones"
        );
    }

    #[test]
    fn a_zero_slot_layer_reports_instead_of_panicking() {
        let mut cache = new_cache(0, 8);
        let mut plan = CachePlan::new();
        let err = cache.plan(&[0], &mut plan).unwrap_err();
        assert_eq!(
            err,
            CacheError::TooFewSlots {
                n_slots: 0,
                requested: 1
            }
        );
        cache
            .plan(&[], &mut plan)
            .expect("an empty step is trivially fine");
        assert!(plan.is_empty());
        // Out-of-range slot bookkeeping is ignored, not fatal.
        cache.mark_ready(9);
        cache.release(9);
        // SAFETY: slot 9 does not exist, so no read can target it.
        unsafe { cache.invalidate(9) };
        assert!(!cache.is_ready(9));
        assert_eq!(cache.resident(9), None);
    }
}
