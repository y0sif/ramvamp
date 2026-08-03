//! Per-layer expert slot cache.
//!
//! Owned by wave-1 lane B. Policy is LFU with recency as tie-breaker and
//! **frequency counters indexed by expert id, not by slot**, so a count
//! survives its expert's eviction. That detail is the policy: measured on
//! real routing traces (EXP-005), ghost-history LFU beats per-slot LFU by
//! +2.2 points and LRU by +2.2 at 10 slots/layer, while per-slot LFU beats
//! LRU by only 0.0-1.7 points and loses outright at 48 slots.
//!
//! The counter array costs `n_experts * u32` per layer (512 B for Qwen3,
//! 24 KiB across 48 layers).
//!
//! A slot that is filling from an in-flight read, or still owned by queued
//! compute, is never selected as an eviction victim.
//!
//! # The policy, exactly
//!
//! One [`LayerCache`] per layer. [`LayerCache::plan`] takes the top-k expert
//! ids the router selected for one token in that layer and splits them into
//! hits (already resident) and misses (slot assigned, caller must read into
//! it), evicting as needed:
//!
//! 1. Every requested expert bumps its frequency counter once per routing
//!    decision, hit or miss. Counters are monotonic: never aged, never
//!    decayed, never cleared on eviction. Aging was measured too
//!    (`lfu-aged`, halving every 256 accesses) and lost 1.8 points.
//! 2. Empty slots are filled before anything is evicted.
//! 3. Otherwise the victim is the unprotected slot with the lowest frequency
//!    count, ties broken by least recent use through a logical clock that
//!    ticks once per requested expert.
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
//! invalidate(s)   -> fill failed:     Empty (unassigned)
//! ```
//!
//! Slot lookup is a linear scan. At the target 12 slots/layer a reverse
//! index costs more than it saves.


use thiserror::Error;

/// Errors from planning one routing step against a layer's slots.
///
/// Every variant means the request could not be served as asked; the cache
/// is left exactly as it was, so the caller can drop the step or retry after
/// releasing slots. Nothing here panics.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CacheError {
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
}

impl Slot {
    /// An unassigned slot.
    const EMPTY: Self = Self {
        expert: None,
        last_used: 0,
        state: SlotState::Empty,
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
    /// two O_DIRECT reads aliasing one buffer make btrfs report spurious
    /// EIO. Callers that plan more than once per step must therefore wait
    /// for outstanding completions (see [`LayerCache::is_ready`]) before
    /// computing on a hit.
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
}

/// Cumulative cache telemetry for one layer.
///
/// The cold/eviction split is the number that says whether the slot budget
/// is wrong: cold misses are unavoidable at any cache size, eviction misses
/// are what more slots would buy. At 10 slots/layer, 87.9% of misses were
/// eviction misses (EXP-005).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Requested experts found resident.
    pub hits: u64,
    /// Requested experts that had to be read.
    pub misses: u64,
    /// Misses for an expert this layer had never fetched.
    pub cold_misses: u64,
    /// Misses for an expert this layer had fetched and evicted.
    pub eviction_misses: u64,
    /// Occupied slots reclaimed to make room.
    pub evictions: u64,
}

impl CacheStats {
    /// Requested experts resolved, hits plus misses.
    pub fn accesses(&self) -> u64 {
        self.hits + self.misses
    }

    /// Fraction of requests served without a read; `0.0` before any request.
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
    /// Whether this layer has ever fetched each expert, for the cold split.
    fetched: Vec<bool>,
    /// Ticks once per requested expert; breaks frequency ties by recency.
    clock: u64,
    /// Cumulative telemetry.
    stats: CacheStats,
}

impl LayerCache {
    /// A cache of `n_slots` slots over an id space of `n_experts` experts.
    ///
    /// `n_slots` must be at least the model's `top_k` for any step to be
    /// satisfiable; a smaller value is accepted here and reported as
    /// [`CacheError::TooFewSlots`] when a step arrives.
    pub fn new(n_slots: u32, n_experts: u32) -> Self {
        Self {
            slots: vec![Slot::EMPTY; n_slots as usize],
            freq: vec![0; n_experts as usize],
            fetched: vec![false; n_experts as usize],
            clock: 0,
            stats: CacheStats::default(),
        }
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
    /// before anything is evicted: on error the cache is unchanged and `out`
    /// is empty.
    pub fn plan(&mut self, experts: &[u32], out: &mut CachePlan) -> Result<(), CacheError> {
        out.clear();
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
                    let entry = &mut self.slots[slot];
                    entry.last_used = clock;
                    if entry.state == SlotState::Idle {
                        // Queued compute owns it again for this step.
                        entry.state = SlotState::Ready;
                    }
                    out.hits.push((expert, slot as u32));
                    self.stats.hits += 1;
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
                    let entry = &mut self.slots[slot];
                    if entry.expert.is_some() {
                        self.stats.evictions += 1;
                    }
                    *entry = Slot {
                        expert: Some(expert),
                        last_used: clock,
                        state: SlotState::Filling,
                    };
                    out.misses.push((expert, slot as u32));
                    self.stats.misses += 1;
                    let fetched = &mut self.fetched[expert as usize];
                    if *fetched {
                        self.stats.eviction_misses += 1;
                    } else {
                        *fetched = true;
                        self.stats.cold_misses += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Record that the read into `slot` completed and its bytes are valid.
    ///
    /// The slot stays protected: queued compute still owns it until
    /// [`release`](Self::release). Out-of-range or unassigned slots are
    /// ignored.
    pub fn mark_ready(&mut self, slot: u32) {
        let Some(entry) = self.slots.get_mut(slot as usize) else {
            tracing::warn!(slot, "mark_ready for a slot outside the layer");
            return;
        };
        match entry.state {
            SlotState::Filling => entry.state = SlotState::Ready,
            // Already readable: a duplicate completion, harmless.
            SlotState::Ready | SlotState::Idle => {}
            SlotState::Empty => tracing::warn!(slot, "mark_ready for an unassigned slot"),
        }
    }

    /// Hand `slot` back: its expert stays resident but the slot becomes an
    /// eviction candidate.
    ///
    /// Call once compute for the step has finished with the buffer, for the
    /// plan's hits as well as its misses. A slot with a read still in flight
    /// is *not* released - dropping that protection is what corrupts
    /// O_DIRECT reads - so a failed fill must go through
    /// [`invalidate`](Self::invalidate) instead. Out-of-range and empty
    /// slots are ignored.
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

    /// Drop `slot`'s assignment and return it to the free pool.
    ///
    /// For a fill that failed: the buffer holds nothing usable, so the
    /// expert must not stay marked resident. Only safe once the read has
    /// definitively completed or been reaped - the slot buffer must not
    /// still be a live I/O destination. Out-of-range slots are ignored.
    pub fn invalidate(&mut self, slot: u32) {
        let Some(entry) = self.slots.get_mut(slot as usize) else {
            tracing::warn!(slot, "invalidate for a slot outside the layer");
            return;
        };
        *entry = Slot::EMPTY;
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
        let mut cache = LayerCache::new(4, 8);
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
        let mut cache = LayerCache::new(4, 8);
        let plan = step(&mut cache, &[5, 1, 7]);
        assert_eq!(plan.misses(), [(5, 0), (1, 1), (7, 2)]);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().cold_misses, 3);
        assert_eq!(cache.resident(3), None);
    }

    #[test]
    fn eviction_picks_the_lowest_frequency() {
        let mut cache = LayerCache::new(3, 8);
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
        let mut cache = LayerCache::new(3, 8);
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
        let mut cache = LayerCache::new(3, 8);
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
            "expert 2 (count 2) is evicted, not the re-admitted expert 1 (count 3)"
        );
        assert_eq!(
            cache.resident(slot_1),
            Some(1),
            "re-admitted expert survives"
        );
    }

    #[test]
    fn filling_and_in_use_slots_are_never_victims() {
        let mut cache = LayerCache::new(2, 8);
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
        let mut cache = LayerCache::new(2, 8);
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
        let mut cache = LayerCache::new(1, 8);
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
        let mut cache = LayerCache::new(1, 8);
        let mut plan = CachePlan::new();
        cache.plan(&[0], &mut plan).expect("empty cache");
        cache.invalidate(0);
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
        let mut cache = LayerCache::new(2, 8);
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
        let mut cache = LayerCache::new(4, 8);
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
        let mut cache = LayerCache::new(2, 8);
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
    }

    #[test]
    fn duplicate_ids_in_one_step_resolve_once() {
        let mut cache = LayerCache::new(4, 8);
        let plan = step(&mut cache, &[2, 2, 5]);
        assert_eq!(plan.misses(), [(2, 0), (5, 1)]);
        assert_eq!(plan.len(), 2);
        assert_eq!(cache.stats().misses, 2);
    }

    #[test]
    fn stats_track_the_cold_and_eviction_split() {
        let mut cache = LayerCache::new(2, 8);
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
    }

    #[test]
    fn the_plan_buffer_is_reused_without_reallocating() {
        let mut cache = LayerCache::new(12, 128);
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
        let mut cache = LayerCache::new(2, 4);
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
        // Expert 0 is back with 7 requests of history against 41 each for
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
        let mut cache = LayerCache::new(0, 8);
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
        cache.invalidate(9);
        assert!(!cache.is_ready(9));
        assert_eq!(cache.resident(9), None);
    }
}
