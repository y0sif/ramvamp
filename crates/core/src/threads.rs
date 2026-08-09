//! Pinned compute pool and CPU topology.
//!
//! Owned by wave-1 lane D. Six compute shards, one per physical P-core with
//! no SMT sibling, joined by a barrier for the duration of a token step.
//! With the default [`PoolConfig::inline_caller`] those six shards are five
//! spawned worker threads plus the submitting decode thread, which runs
//! shard 0 itself; with `inline_caller: false` they are six spawned workers
//! and the decode thread only waits.
//!
//! Topology on the reference machine (Core Ultra 9 185H) is not guessable
//! from core ids: `/sys/devices/cpu_core/cpus` is `0-11` and
//! `/sys/devices/cpu_atom/cpus` is `12-21`, and the P-core SMT pairs are
//! (0,5) (1,2) (3,4) (6,7) (8,9) (10,11) — so the primaries are
//! `[0, 1, 3, 6, 8, 10]`. E-cores 12-19 have L3; the LP E-cores 20-21 sit
//! on the SoC tile with no L3 at all, so a flag handed off there crosses
//! the fabric. Detection must be derived from `thread_siblings_list` at
//! runtime rather than hardcoded, and must degrade to "no pinning" on any
//! machine whose topology does not match the expected shape.
//!
//! # Layout
//!
//! * [`Topology`] — inspectable, `Display`-able view of what was detected.
//!   Parsed from a sysfs root so tests can point it at a fixture tree.
//! * [`pin_current_thread`] / [`CpuMask`] — affinity primitives. Every pin is
//!   intersected with `sched_getaffinity`, so a restricted cpuset can never
//!   be escaped and a pin failure only ever degrades to "unpinned".
//! * [`ComputePool`] — N shards of a row range executed in parallel and
//!   rendezvoused at a barrier, cheap enough for ~1000 GEMVs per token.
//!
//! # Handoff cost
//!
//! Recorded as EXP-010 in `docs/experiments.md`, and provisional under
//! that log's rule 2 (measured on a machine that was not quiet, and a
//! microbenchmark rather than an end-to-end run): treat the figures as
//! ordering evidence, not as published numbers.
//!
//! What was measured on the reference machine is the round trip this pool
//! actually performs — submitter publishes, workers wake, workers run, the
//! barrier retires — for three handoff primitives. A futex wake/wait pair is
//! p50 3.1 µs at 0.07 cores of steady-state overhead; pure atomic spinning
//! is 502 ns but burns 1.03 cores; `std::sync::mpsc` is p99 237 µs, which is
//! disqualifying for a per-GEMV barrier. The figures predate the decision to
//! drive the io_uring reactor inline on the coordinator, so they are *not* a
//! reactor-to-worker measurement — this pool performs no such handoff.
//!
//! The pool therefore uses a bounded spin ([`SPIN_ROUNDS`]) followed by a
//! futex wait, which behaves like the spinner while the decode loop is hot
//! and like the futex when it goes idle. Re-measure with
//! `cargo test -p ramvamp-core -- --ignored wake_latency --nocapture`.
//!
//! # Borrowed work on persistent threads
//!
//! A GEMV job must borrow the weight slice, the activation slice and a
//! disjoint mutable sub-slice of the output — none of which is `'static`.
//! `std::thread::scope` expresses that, but re-creating a scope per GEMV
//! would spawn thousands of threads per token. Instead the pool keeps its
//! threads forever and re-establishes the scope invariant per job:
//!
//! 1. [`ComputePool::run`] takes `&mut self`, so only one job can be in
//!    flight and no job can be submitted from inside another.
//! 2. The closure is type-erased to a thin `*const ()` plus a monomorphised
//!    `unsafe fn(*const (), Shard)` trampoline. No lifetime is transmuted;
//!    the pointer is simply `&F` with its type forgotten.
//! 3. The pointer is published to the workers *before* the job sequence
//!    number is bumped, and every worker reads it only after observing the
//!    new sequence number, so the publication is properly ordered.
//! 4. `run` does not return — on the normal path or the unwind path — until
//!    `pending` has reached zero, i.e. until every worker has finished
//!    calling the closure and can no longer observe the pointer. The wait is
//!    performed by a `Drop` guard, exactly as `std::thread::scope` does, so
//!    a panic in the caller's own shard cannot let the borrow escape. The
//!    guard is constructed *before* the job is published, so there is no
//!    instant at which a worker can see the pointer without the guard being
//!    live.
//! 5. Worker closures run under `catch_unwind`, so a panicking closure can
//!    neither skip its `pending` decrement (which would deadlock the
//!    barrier) nor unwind out of the worker thread (which would silently
//!    shrink the pool). The payload is re-raised on the submitting thread.
//!    The decrement is performed by a `Drop` guard rather than by a tail
//!    call, so it survives an unwind raised *while* the payload is being
//!    recorded — the one remaining way a `catch_unwind` body can still
//!    unwind is a panic payload whose own `Drop` panics, and that is
//!    contained by a second `catch_unwind` around the recording step (see
//!    `worker_loop`), so it can neither strand the barrier nor kill the
//!    thread.
//!
//! `F: Sync` is what makes `&F` sendable to the workers; `T: Send` is what
//! makes the disjoint output sub-slices sendable in [`ComputePool::scatter`].

use std::cell::UnsafeCell;
use std::collections::BTreeSet;
use std::fmt::{self, Write as _};
use std::fs;
use std::io::ErrorKind;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use thiserror::Error;
use tracing::{debug, info, warn};

/// Default sysfs mount point.
const SYSFS_ROOT: &str = "/sys";

/// Largest CPU id an affinity mask can hold.
///
/// `rustix`'s `CpuSet` is a fixed-size `[u64; CPU_SETSIZE / 64]` and
/// `CpuSet::set` indexes it directly (`rustix-1.1.4`,
/// `src/backend/linux_raw/thread/cpu_set.rs:9-13`). That is an ordinary Rust
/// array index, so an out-of-range id is a bounds-check *panic*, not
/// undefined behaviour — but `ramvamp-core` must not panic on data it read
/// out of sysfs, so every CPU id is range-checked against this constant
/// before it reaches the syscall wrapper and reported as a typed error
/// instead.
#[cfg(target_os = "linux")]
pub const MAX_CPUS: usize = rustix::thread::CpuSet::MAX_CPU;
/// Largest CPU id an affinity mask can hold.
#[cfg(not(target_os = "linux"))]
pub const MAX_CPUS: usize = 1024;

/// Hard ceiling on [`PoolConfig::shards`].
///
/// A shard exists to occupy one CPU, so more shards than an affinity mask
/// can even name is meaningless — and it is the difference between a caller
/// typo and `Vec::with_capacity(usize::MAX - 1)`, which aborts the process
/// instead of unwinding. Values above this are clamped, not rejected,
/// because the constructors are documented as infallible.
pub const MAX_SHARDS: usize = MAX_CPUS;

/// Bounded spin, in `pause` iterations, before a thread parks on a futex.
///
/// 64 `pause` instructions is roughly 2-3 µs on the reference machine —
/// the same order as the futex round trip it replaces, so a hot decode
/// loop essentially never enters the kernel while an idle pool still
/// parks promptly.
pub const SPIN_ROUNDS: u32 = 64;

/// "Wake everyone" for `FUTEX_WAKE`.
///
/// The kernel reads the count as a signed `int`, so `u32::MAX` arrives as
/// `-1` and the wake loop terminates after a *single* waiter. `i32::MAX` is
/// the largest value that means what it looks like it means.
const WAKE_ALL: u32 = i32::MAX.unsigned_abs();

// ---------------------------------------------------------------------------
// CPU masks
// ---------------------------------------------------------------------------

/// A set of logical CPU ids.
///
/// Ordered and deduplicated; iteration is always ascending, which is what
/// makes SMT-primary selection deterministic.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuMask(BTreeSet<usize>);

impl CpuMask {
    /// An empty mask.
    #[must_use]
    pub fn new() -> Self {
        Self(BTreeSet::new())
    }

    /// A mask holding `0..n`, clamped to [`MAX_CPUS`].
    #[must_use]
    pub fn up_to(n: usize) -> Self {
        Self((0..n.min(MAX_CPUS)).collect())
    }

    /// Add `cpu`. Ids at or beyond [`MAX_CPUS`] are rejected and return
    /// `false` rather than panicking, because ids come from sysfs.
    pub fn insert(&mut self, cpu: usize) -> bool {
        if cpu >= MAX_CPUS {
            return false;
        }
        self.0.insert(cpu)
    }

    /// Whether `cpu` is in the mask.
    #[must_use]
    pub fn contains(&self, cpu: usize) -> bool {
        self.0.contains(&cpu)
    }

    /// Number of CPUs in the mask.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the mask holds no CPUs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Ascending iterator over the mask.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().copied()
    }

    /// The CPUs present in both masks.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        Self(self.0.intersection(&other.0).copied().collect())
    }

    /// The mask as an ascending vector.
    #[must_use]
    pub fn to_vec(&self) -> Vec<usize> {
        self.0.iter().copied().collect()
    }

    /// This thread's current affinity mask, or `None` if it cannot be read.
    ///
    /// Never fails loudly: an unreadable mask degrades to "assume every CPU
    /// `available_parallelism` reports".
    #[must_use]
    pub fn current() -> Option<Self> {
        current_affinity()
    }
}

impl FromIterator<usize> for CpuMask {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut mask = Self::new();
        for cpu in iter {
            mask.insert(cpu);
        }
        mask
    }
}

impl fmt::Display for CpuMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_cpu_list(&self.to_vec()))
    }
}

#[cfg(target_os = "linux")]
fn current_affinity() -> Option<CpuMask> {
    match rustix::thread::sched_getaffinity(None) {
        Ok(set) => {
            let mut mask = CpuMask::new();
            for cpu in 0..MAX_CPUS {
                if set.is_set(cpu) {
                    mask.insert(cpu);
                }
            }
            if mask.is_empty() { None } else { Some(mask) }
        }
        Err(err) => {
            debug!(error = %err, "sched_getaffinity failed; assuming all cpus");
            None
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn current_affinity() -> Option<CpuMask> {
    None
}

/// Best-effort affinity mask: the real one, else `0..available_parallelism`.
fn affinity_or_default() -> CpuMask {
    current_affinity().unwrap_or_else(|| CpuMask::up_to(available_parallelism()))
}

fn available_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

// ---------------------------------------------------------------------------
// Pinning
// ---------------------------------------------------------------------------

/// Errors from setting thread affinity.
///
/// Callers are expected to treat every variant as "run unpinned": a user in
/// a restricted cpuset, a container, or a non-Linux dev box must still be
/// able to start the runtime.
#[derive(Debug, Error)]
pub enum PinError {
    /// The requested CPU is outside the process's own affinity mask.
    #[error("cpu {cpu} is outside this process's affinity mask")]
    NotAllowed {
        /// The rejected CPU id.
        cpu: usize,
    },

    /// The requested CPU id does not fit an affinity mask.
    #[error("cpu {cpu} is beyond the {max}-cpu affinity mask limit")]
    OutOfRange {
        /// The rejected CPU id.
        cpu: usize,
        /// Largest representable CPU id plus one.
        max: usize,
    },

    /// A pin was requested with no CPUs in it.
    #[error("refusing to pin to an empty cpu mask")]
    EmptyMask,

    /// `sched_setaffinity` refused the mask.
    #[error("sched_setaffinity failed: {0}")]
    SetAffinity(#[source] std::io::Error),

    /// This platform has no affinity API.
    #[error("cpu pinning is not supported on this platform")]
    Unsupported,
}

/// Pin the calling thread to exactly `cpu`.
///
/// The CPU is checked against the caller's current affinity mask first, so
/// this can never widen a cpuset restriction.
pub fn pin_current_thread(cpu: usize) -> Result<(), PinError> {
    if cpu >= MAX_CPUS {
        return Err(PinError::OutOfRange { cpu, max: MAX_CPUS });
    }
    if let Some(allowed) = current_affinity()
        && !allowed.contains(cpu)
    {
        return Err(PinError::NotAllowed { cpu });
    }
    let mut mask = CpuMask::new();
    mask.insert(cpu);
    set_affinity(&mask)
}

/// Pin the calling thread to every CPU in `mask`.
pub fn pin_current_thread_to(mask: &CpuMask) -> Result<(), PinError> {
    if mask.is_empty() {
        return Err(PinError::EmptyMask);
    }
    set_affinity(mask)
}

#[cfg(target_os = "linux")]
fn set_affinity(mask: &CpuMask) -> Result<(), PinError> {
    let mut set = rustix::thread::CpuSet::new();
    for cpu in mask.iter() {
        if cpu >= MAX_CPUS {
            return Err(PinError::OutOfRange { cpu, max: MAX_CPUS });
        }
        set.set(cpu);
    }
    // `None` targets the calling thread.
    rustix::thread::sched_setaffinity(None, &set)
        .map_err(|err| PinError::SetAffinity(std::io::Error::from(err)))
}

#[cfg(not(target_os = "linux"))]
fn set_affinity(_mask: &CpuMask) -> Result<(), PinError> {
    Err(PinError::Unsupported)
}

/// Pin, or log and carry on unpinned. Returns whether the pin took effect.
fn pin_or_warn(cpu: usize, what: &'static str) -> bool {
    match pin_current_thread(cpu) {
        Ok(()) => {
            debug!(cpu, thread = what, "pinned thread");
            true
        }
        Err(err) => {
            warn!(cpu, thread = what, error = %err, "cpu pin failed; running unpinned");
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

/// Errors from parsing CPU topology out of sysfs.
///
/// Never fatal: [`Topology::detect`] and [`Topology::detect_at`] swallow
/// these and fall back to an unpinned topology.
#[derive(Debug, Error)]
pub enum TopologyError {
    /// A sysfs file could not be read.
    #[error("{}: {source}", path.display())]
    Io {
        /// The file that could not be read.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// A sysfs file did not hold a CPU list.
    #[error("{}: malformed cpu list {value:?} ({reason})", path.display())]
    Malformed {
        /// The offending file.
        path: PathBuf,
        /// Its contents, trimmed.
        value: String,
        /// Why the parse failed.
        reason: &'static str,
    },

    /// The process may not run on any CPU at all.
    #[error("the process affinity mask is empty")]
    NoCpus,
}

/// What the runtime learned about this machine's CPUs.
///
/// Construct with [`Topology::detect`] in production and
/// [`Topology::detect_at`] / [`Topology::parse_at`] against a fixture tree
/// in tests. Every CPU id in every accessor is guaranteed to be inside the
/// process's own affinity mask.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    allowed: CpuMask,
    performance: Vec<usize>,
    efficiency: Vec<usize>,
    l3_efficiency: Vec<usize>,
    compute: Vec<usize>,
    reactor: Option<usize>,
    hybrid: bool,
    pins: bool,
    shards: usize,
}

impl Topology {
    /// Detect from the real `/sys` and the calling thread's affinity mask.
    ///
    /// Infallible by construction: anything unexpected logs and degrades to
    /// [`Topology::unpinned`].
    #[must_use]
    pub fn detect() -> Self {
        Self::detect_at(Path::new(SYSFS_ROOT), affinity_or_default())
    }

    /// Detect against an arbitrary sysfs root, degrading on any failure.
    #[must_use]
    pub fn detect_at(sysfs: &Path, allowed: CpuMask) -> Self {
        match Self::parse_at(sysfs, allowed.clone()) {
            Ok(topo) => {
                if topo.pins {
                    info!(
                        compute = %format_cpu_list(&topo.compute),
                        reactor = ?topo.reactor,
                        "hybrid cpu topology detected; pinning compute threads"
                    );
                } else {
                    info!(
                        shards = topo.shards,
                        "no pinnable hybrid topology; running unpinned"
                    );
                }
                topo
            }
            Err(err) => {
                debug!(error = %err, "cpu topology detection failed; running unpinned");
                Self::unpinned(allowed)
            }
        }
    }

    /// A topology that pins nothing and parallelises over available CPUs.
    #[must_use]
    pub fn unpinned(allowed: CpuMask) -> Self {
        let shards = fallback_shards(&allowed);
        Self {
            allowed,
            performance: Vec::new(),
            efficiency: Vec::new(),
            l3_efficiency: Vec::new(),
            compute: Vec::new(),
            reactor: None,
            hybrid: false,
            pins: false,
            shards,
        }
    }

    /// Parse a sysfs tree, reporting why detection failed.
    ///
    /// `allowed` bounds everything: no returned CPU is outside it. A tree
    /// with no hybrid PMU directories parses successfully but yields
    /// `pins() == false`, because non-hybrid pinning is not something this
    /// runtime has evidence for.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError`] if a sysfs file exists but cannot be read
    /// or does not hold a CPU list, or if `allowed` is empty.
    pub fn parse_at(sysfs: &Path, allowed: CpuMask) -> Result<Self, TopologyError> {
        if allowed.is_empty() {
            return Err(TopologyError::NoCpus);
        }

        let perf = read_cpu_list(&sysfs.join("devices/cpu_core/cpus"))?;
        let eff = read_cpu_list(&sysfs.join("devices/cpu_atom/cpus"))?;
        let hybrid = matches!((&perf, &eff), (Some(p), Some(e)) if !p.is_empty() && !e.is_empty());

        let performance: Vec<usize> = perf
            .unwrap_or_default()
            .into_iter()
            .filter(|c| allowed.contains(*c))
            .collect();
        let efficiency: Vec<usize> = eff
            .unwrap_or_default()
            .into_iter()
            .filter(|c| allowed.contains(*c))
            .collect();

        // Candidate compute CPUs: P-cores on a hybrid part, otherwise every
        // CPU the kernel says is online (intersected with our own mask).
        let candidates: Vec<usize> = if hybrid {
            performance.clone()
        } else {
            let online = read_cpu_list(&sysfs.join("devices/system/cpu/online"))?;
            match online {
                Some(list) => list.into_iter().filter(|c| allowed.contains(*c)).collect(),
                None => allowed.to_vec(),
            }
        };

        // `None` means sysfs would not say what the SMT layout is for some
        // candidate (see `read_siblings`). That disables pinning for the
        // whole machine; every other field is still reported.
        let compute = smt_primaries(sysfs, &candidates).unwrap_or_default();

        // An io_uring reactor wants an E-core that shares L3 with the
        // compute cores, so completion flags hand off through L3 rather than
        // across the fabric. LP E-cores (no L3 at all) are never used.
        let l3_efficiency: Vec<usize> = efficiency
            .iter()
            .copied()
            .filter(|&cpu| cpu_has_l3(sysfs, cpu))
            .collect();
        let reactor = if hybrid {
            l3_efficiency.first().copied()
        } else {
            None
        };

        // Only pin on a shape we recognise and that buys parallelism.
        let pins = hybrid && compute.len() >= 2;
        let shards = if pins {
            compute.len()
        } else {
            fallback_shards(&allowed)
        };

        Ok(Self {
            allowed,
            performance,
            efficiency,
            l3_efficiency,
            compute: if pins { compute } else { Vec::new() },
            reactor,
            hybrid,
            pins,
            shards,
        })
    }

    /// Whether the kernel reported a hybrid (P-core/E-core) part.
    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        self.hybrid
    }

    /// Whether compute threads should be pinned.
    #[must_use]
    pub fn pins(&self) -> bool {
        self.pins
    }

    /// The process's affinity mask at detection time.
    #[must_use]
    pub fn allowed(&self) -> &CpuMask {
        &self.allowed
    }

    /// Allowed logical CPUs on performance cores (empty when not hybrid).
    #[must_use]
    pub fn performance_cpus(&self) -> &[usize] {
        &self.performance
    }

    /// Allowed logical CPUs on efficiency cores (empty when not hybrid).
    #[must_use]
    pub fn efficiency_cpus(&self) -> &[usize] {
        &self.efficiency
    }

    /// Efficiency cores that have an L3 cache; the LP tile is excluded.
    #[must_use]
    pub fn l3_efficiency_cpus(&self) -> &[usize] {
        &self.l3_efficiency
    }

    /// One logical CPU per physical compute core, ascending.
    ///
    /// Empty when [`Topology::pins`] is false.
    #[must_use]
    pub fn compute_cpus(&self) -> &[usize] {
        &self.compute
    }

    /// Preferred CPU for the io_uring reactor thread.
    #[must_use]
    pub fn reactor_cpu(&self) -> Option<usize> {
        self.reactor
    }

    /// How many shards a parallel job should be split into.
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.max(1)
    }
}

impl fmt::Display for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} cpus allowed {}",
            if self.hybrid { "hybrid" } else { "uniform" },
            format_cpu_list(&self.allowed.to_vec())
        )?;
        if self.hybrid {
            write!(
                f,
                "; P {}; E {} (L3 {})",
                format_cpu_list(&self.performance),
                format_cpu_list(&self.efficiency),
                format_cpu_list(&self.l3_efficiency)
            )?;
        }
        write!(f, "; compute {}", format_cpu_list(&self.compute))?;
        match self.reactor {
            Some(cpu) => write!(f, "; reactor {cpu}")?,
            None => write!(f, "; reactor unpinned")?,
        }
        write!(
            f,
            "; {} shards, pinning {}",
            self.shard_count(),
            if self.pins { "on" } else { "off" }
        )
    }
}

/// How many shards to use when nothing is pinned.
fn fallback_shards(allowed: &CpuMask) -> usize {
    // `available_parallelism` accounts for cgroup CPU quota, which the
    // affinity mask does not; the mask accounts for cpuset restrictions,
    // which the quota does not. Take the tighter of the two.
    allowed.len().min(available_parallelism()).max(1)
}

/// Read a sysfs cpu-list file. `Ok(None)` means the file is absent.
fn read_cpu_list(path: &Path) -> Result<Option<Vec<usize>>, TopologyError> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(TopologyError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    parse_cpu_list(text.trim())
        .map(Some)
        .map_err(|reason| TopologyError::Malformed {
            path: path.to_path_buf(),
            value: text.trim().to_owned(),
            reason,
        })
}

/// Parse a Linux cpu list (`0-11`, `0,5`, `0-3,8,10-11`).
///
/// Returns an ascending, deduplicated vector. The error is a static reason
/// string; callers attach the path. Ids at or beyond [`MAX_CPUS`] are
/// rejected rather than silently truncated, because indexing the `CpuSet`
/// bit array with one would panic downstream (see [`MAX_CPUS`]).
fn parse_cpu_list(text: &str) -> Result<Vec<usize>, &'static str> {
    let mut cpus = BTreeSet::new();
    let text = text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err("empty range element");
        }
        let (lo, hi) = match part.split_once('-') {
            Some((lo, hi)) => (lo.trim(), hi.trim()),
            None => (part, part),
        };
        let lo: usize = lo.parse().map_err(|_| "not a cpu id")?;
        let hi: usize = hi.parse().map_err(|_| "not a cpu id")?;
        if hi < lo {
            return Err("descending range");
        }
        if hi >= MAX_CPUS {
            return Err("cpu id beyond the affinity mask limit");
        }
        for cpu in lo..=hi {
            cpus.insert(cpu);
        }
    }
    Ok(cpus.into_iter().collect())
}

/// Render an ascending cpu list the way sysfs does.
fn format_cpu_list(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return "-".to_owned();
    }
    let mut out = String::new();
    let mut i = 0;
    while i < cpus.len() {
        let start = cpus[i];
        let mut end = start;
        let mut j = i + 1;
        while j < cpus.len() && cpus[j] == end + 1 {
            end = cpus[j];
            j += 1;
        }
        if !out.is_empty() {
            out.push(',');
        }
        let _ = if end == start {
            write!(out, "{start}")
        } else {
            write!(out, "{start}-{end}")
        };
        i = j;
    }
    out
}

/// Pick one logical CPU per physical core, in ascending order.
///
/// `candidates` must be ascending. Siblings outside `candidates` still
/// retire their group — a cpuset that exposes only one half of an SMT pair
/// still yields exactly one primary for that physical core.
///
/// `None` means sysfs would not say what the SMT layout is for at least one
/// candidate, in which case *no* primaries are returned rather than a
/// partial answer; see [`read_siblings`] for the policy and for why it is
/// not "assume that CPU is alone on its core".
fn smt_primaries(sysfs: &Path, candidates: &[usize]) -> Option<Vec<usize>> {
    let mut claimed: BTreeSet<usize> = BTreeSet::new();
    let mut primaries = Vec::new();
    for &cpu in candidates {
        if claimed.contains(&cpu) {
            continue;
        }
        let siblings = read_siblings(sysfs, cpu)?;
        primaries.push(cpu);
        claimed.insert(cpu);
        claimed.extend(siblings);
    }
    Some(primaries)
}

/// Sibling logical CPUs of `cpu`, or `None` when sysfs will not say.
///
/// # Failure policy
///
/// Sibling data is evidence *for* pinning in exactly the way [`cpu_has_l3`]
/// is evidence for the reactor hint, and both obey the same two rules: a
/// file this cannot read is never fatal to detection, and is never guessed
/// at either. Concretely:
///
/// * a readable, non-empty list is authoritative;
/// * both spellings are tried independently, so a malformed
///   `thread_siblings_list` still falls through to `core_cpus_list` instead
///   of discarding the machine on the strength of one bad file;
/// * an absent, empty, unreadable or malformed pair of files yields `None`,
///   never `[cpu]`. "I could not read the sibling list" is not evidence that
///   `cpu` has no SMT sibling. The old `[cpu]` default meant that on a
///   container exposing `devices/cpu_core/cpus` but no `cpuN/topology/`,
///   *both* halves of every SMT pair were promoted to primaries — twelve
///   pinned threads on six physical cores, announced only by a `debug!`;
/// * `None` propagates to [`smt_primaries`] and disables pinning for the
///   whole machine. Running unpinned is a measurable slowdown that shows up
///   in `docs/experiments.md`; double-booking physical cores is a silent
///   violation of this module's one-thread-per-physical-core invariant, so
///   the two are not symmetric and the tie goes to unpinned.
fn read_siblings(sysfs: &Path, cpu: usize) -> Option<Vec<usize>> {
    let base = sysfs.join(format!("devices/system/cpu/cpu{cpu}/topology"));
    // `thread_siblings_list` is the classic name; `core_cpus_list` is the
    // post-5.3 spelling. Either is authoritative.
    for name in ["thread_siblings_list", "core_cpus_list"] {
        match read_cpu_list(&base.join(name)) {
            Ok(Some(list)) if !list.is_empty() => return Some(list),
            // Absent or empty: try the other spelling.
            Ok(_) => {}
            Err(err) => warn!(cpu, file = name, error = %err, "unusable smt sibling list"),
        }
    }
    warn!(
        cpu,
        topology = %base.display(),
        "sysfs reports no usable smt sibling list; refusing to pin, because \
         treating an unknown sibling list as 'no sibling' would put two \
         compute threads on one physical core"
    );
    None
}

/// Whether `cpu` has any level-3 cache.
///
/// The LP E-cores on Meteor Lake sit on the SoC tile and report no L3 index
/// at all, which is how they are told apart from the L3-sharing E-cores.
///
/// Deliberately infallible: this only picks a *hint* for the io_uring
/// reactor, so an unreadable cache node degrades to "no L3 here" rather than
/// discarding the whole (pinnable) topology. Note that sysfs puts a plain
/// `uevent` file alongside the `index*` directories, so entries must be
/// filtered by name — joining `level` onto `uevent` yields `ENOTDIR`, not
/// `ENOENT`.
fn cpu_has_l3(sysfs: &Path, cpu: usize) -> bool {
    let dir = sysfs.join(format!("devices/system/cpu/cpu{cpu}/cache"));
    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(err) => {
            if err.kind() != ErrorKind::NotFound {
                debug!(cpu, error = %err, "cannot list cache nodes; assuming no L3");
            }
            return false;
        }
    };
    for entry in entries.flatten() {
        if !entry.file_name().as_encoded_bytes().starts_with(b"index") {
            continue;
        }
        if let Ok(text) = fs::read_to_string(entry.path().join("level"))
            && text.trim() == "3"
        {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Job partitioning
// ---------------------------------------------------------------------------

/// One shard of a parallel job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shard {
    /// Zero-based shard index.
    pub index: usize,
    /// Total shards this job was split into.
    pub count: usize,
    /// Half-open row range this shard owns.
    pub rows: Range<usize>,
}

impl Shard {
    /// Number of rows in this shard.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether this shard has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// The row range shard `index` of `shards` owns when splitting `rows`.
///
/// A pure function of `(rows, shards, index)` — the pool never partitions a
/// job any other way, so a given `(rows, shards)` always produces the same
/// split and therefore bit-identical reduction order per shard.
///
/// Shards are contiguous and ascending; the first `rows % shards` shards get
/// one extra row. Every row in `0..rows` lands in exactly one shard.
#[must_use]
pub fn shard_range(rows: usize, shards: usize, index: usize) -> Range<usize> {
    if shards == 0 || index >= shards {
        return 0..0;
    }
    let base = rows / shards;
    let rem = rows % shards;
    let start = index * base + index.min(rem);
    let len = base + usize::from(index < rem);
    start..start + len
}

// ---------------------------------------------------------------------------
// Parking
// ---------------------------------------------------------------------------

/// Longest a single futex sleep in [`Shared::await_barrier`] may last.
///
/// Every futex sleep in this module is bounded, so that a wakeup lost for
/// *any* reason — a protocol bug, a kernel oddity, a future edit to the wake
/// path — degrades to a slow poll instead of an unrecoverable hang. This is
/// defence in depth, not the mechanism the barrier relies on: the protocol
/// documented on [`Shared`] is what makes lost wakeups impossible, and this
/// is what makes being wrong about that survivable and observable.
///
/// It is free in the hot path. A parked submitter is by definition waiting
/// on a job that is already running; the decode loop usually never reaches
/// the park at all ([`SPIN_ROUNDS`] covers the common barrier); and one
/// extra wakeup per millisecond of a job that already costs milliseconds is
/// noise.
const BARRIER_POLL: Duration = Duration::from_millis(1);

/// First bound on a worker's futex sleep in [`Shared::await_seq`].
///
/// A worker waits here for the *next* job, so unlike the barrier it can
/// legitimately sleep for a long time — between tokens, or indefinitely once
/// the process goes idle. Polling at [`BARRIER_POLL`] forever would cost
/// five threads x 1000 wakeups/s on an idle pool, which is exactly what the
/// futex is there to avoid. So the bound starts here, where a lost job
/// wakeup costs a millisecond, and doubles up to [`JOB_POLL_MAX`].
const JOB_POLL_MIN: Duration = Duration::from_millis(1);

/// Ceiling on the [`JOB_POLL_MIN`] backoff.
///
/// An idle worker settles at ~4 wakeups/s, and the worst case for a wakeup
/// lost on an already-idle pool is that the next job starts a quarter second
/// late rather than never.
const JOB_POLL_MAX: Duration = Duration::from_millis(256);

#[cfg(target_os = "linux")]
mod park {
    use rustix::thread::futex;
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    /// Sleep until `word` stops being `expected`, until woken, or until
    /// `timeout` elapses.
    ///
    /// `EAGAIN` (the value already changed), `EINTR` and `ETIMEDOUT` are all
    /// normal and indistinguishable from a real wakeup here: every caller
    /// re-reads the word in a loop, so a spurious return is free.
    ///
    /// `FUTEX_WAIT` takes a *relative* timeout, which is what a `Duration`
    /// is; the bitset variants would need an absolute one.
    pub(super) fn wait(word: &AtomicU32, expected: u32, timeout: Duration) {
        let timeout = futex::Timespec {
            // `Duration` seconds outrange `Secs`; saturating keeps the
            // timeout bounded either way, which is all this needs.
            tv_sec: futex::Secs::try_from(timeout.as_secs()).unwrap_or(futex::Secs::MAX),
            // `subsec_nanos()` is under 1e9, which is exact in every `Nsecs`
            // Linux uses (`i64` on linux-raw, `c_long` on the libc backend).
            tv_nsec: timeout.subsec_nanos() as futex::Nsecs,
        };
        let _ = futex::wait(word, futex::Flags::PRIVATE, expected, Some(&timeout));
    }

    /// Wake up to `count` threads parked on `word`.
    pub(super) fn wake(word: &AtomicU32, count: u32) {
        let _ = futex::wake(word, futex::Flags::PRIVATE, count);
    }
}

#[cfg(not(target_os = "linux"))]
mod park {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    /// Portable stand-in for the futex path: yield until the word changes or
    /// `timeout` elapses.
    ///
    /// Only used for non-Linux development builds; it burns a core while
    /// waiting, which is exactly why the Linux path exists.
    pub(super) fn wait(word: &AtomicU32, expected: u32, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        while word.load(Ordering::Acquire) == expected && Instant::now() < deadline {
            std::thread::yield_now();
        }
    }

    /// No-op: the portable `wait` polls rather than sleeps.
    pub(super) fn wake(_word: &AtomicU32, _count: u32) {}
}

// ---------------------------------------------------------------------------
// Compute pool
// ---------------------------------------------------------------------------

type PanicPayload = Box<dyn std::any::Any + Send + 'static>;

/// A type-erased reference to the job closure, plus its partitioning.
///
/// `data` is `&F` with its type forgotten and `call` is the monomorphised
/// trampoline that restores it. No lifetime is transmuted; validity is
/// enforced entirely by the barrier protocol documented on [`Shared`].
#[derive(Clone, Copy)]
struct JobRef {
    data: *const (),
    call: unsafe fn(*const (), Shard),
    rows: usize,
    shards: usize,
}

/// Restore `&F` from an erased pointer and invoke it.
///
/// # Safety
///
/// `data` must be a live `*const F` published by [`ComputePool::run`] for a
/// job that has not yet completed its barrier.
unsafe fn call_job<F: Fn(Shard) + Sync>(data: *const (), shard: Shard) {
    // SAFETY: the caller guarantees `data` came from `&F` and that the
    // referent outlives this call (the submitter blocks until every worker
    // has returned from here).
    let f = unsafe { &*data.cast::<F>() };
    f(shard);
}

/// State shared between the submitting thread and the workers.
///
/// # Protocol
///
/// * `seq` is bumped once per job and once at shutdown. Workers wait for it
///   to change; it is the futex word they park on.
/// * `pending` counts workers that have not yet finished the current job.
///   It is the futex word the submitter parks on, and reaching zero is the
///   barrier.
/// * `job` is written only by the submitter, only while `pending == 0`, and
///   only before the `seq` store that publishes it. Workers read it only
///   after observing the new `seq`, and strictly before the `fetch_sub` that
///   retires their shard. Those facts are what make the `UnsafeCell` sound;
///   see [`Shared::publish`].
/// * `workers_parked` / `lead_parked` let each side skip the `futex` syscall
///   when nobody is actually asleep. Both are updated with `SeqCst` and
///   paired against a `SeqCst` access of the corresponding futex word, which
///   is the standard Dekker interleaving: whichever side commits second
///   necessarily observes the other, so a wakeup can never be lost.
///
/// # Why a worker may read the park flag but never take it
///
/// The Dekker argument above holds *within* one job. `lead_parked` carries
/// no job identity, so a flag set for job N+1 is indistinguishable from one
/// set for job N — and a worker of job N can still be between its
/// `fetch_sub` and its inspection of the flag long after job N's barrier has
/// retired, because the barrier is satisfied by the `fetch_sub` itself, not
/// by the worker leaving `finish_shard`.
///
/// If that worker *took* the flag (`swap(0)`) it would consume a flag that
/// belongs to the next job and wake nobody, because the submitter has not
/// enqueued on the futex yet. The submitter would then park with
/// `lead_parked == 0`, and the eventual last decrement of job N+1 would find
/// no flag and issue no wake: a permanent deadlock. Nothing about that
/// depends on how narrow the window is, which is why it is fixed by
/// construction rather than by narrowing.
///
/// The invariant that removes the class is: **`lead_parked` is written only
/// by the thread it belongs to.** Workers only ever `load` it, so no worker
/// of any generation can consume any other generation's flag; the submitter
/// sets it once on entering its parked phase and clears it once on leaving,
/// both in [`Shared::await_barrier`]. A worker of job N that observes job
/// N+1's flag issues one wake nobody is waiting for, which costs a syscall
/// and is otherwise inert — that is the entire price, and it is paid only in
/// the interleaving that used to deadlock.
///
/// What remains is a single-generation obligation, and `SeqCst` discharges
/// it. Worker: `fetch_sub(pending)` then `load(lead_parked)`. Submitter:
/// `store(lead_parked, 1)` then `load(pending)`. In the total order, if the
/// worker's load misses the submitter's store then the submitter's load must
/// come after the worker's decrement — and a submitter that reads
/// `pending == 0` never parks. A submitter that does park is guaranteed
/// either a wake or an `EAGAIN`, because `park::wait` only sleeps while
/// `pending` still equals the value just read, and `pending` decreases
/// monotonically within a job (`run` takes `&mut self`, so the submitter
/// cannot republish while it is parked).
struct Shared {
    seq: AtomicU32,
    pending: AtomicU32,
    workers_parked: AtomicU32,
    /// Set by the submitter while it is parked on `pending`; read, never
    /// written, by workers. See the type docs for why the asymmetry is
    /// load-bearing.
    lead_parked: AtomicU32,
    quit: AtomicU32,
    job: UnsafeCell<Option<JobRef>>,
    panic: Mutex<Option<PanicPayload>>,
    /// Test-only interleaving control; not present in any other build.
    #[cfg(test)]
    probe: tests::BarrierProbe,
}

// SAFETY: `job` is the only interior-mutable non-atomic field. The protocol
// above gives it a strict single-writer / quiesced-readers discipline, and
// the raw pointer it holds is only dereferenced between the publishing `seq`
// store and the `pending` drain, a window the submitter provably outlives.
unsafe impl Send for Shared {}
// SAFETY: see `Send`.
unsafe impl Sync for Shared {}

impl Shared {
    fn new() -> Self {
        Self {
            seq: AtomicU32::new(0),
            pending: AtomicU32::new(0),
            workers_parked: AtomicU32::new(0),
            lead_parked: AtomicU32::new(0),
            quit: AtomicU32::new(0),
            job: UnsafeCell::new(None),
            panic: Mutex::new(None),
            #[cfg(test)]
            probe: tests::BarrierProbe::new(),
        }
    }

    fn record_panic(&self, payload: PanicPayload) {
        let mut slot = self.panic.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(payload);
        }
    }

    fn take_panic(&self) -> Option<PanicPayload> {
        self.panic
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// Publish a job and release the workers.
    fn publish(&self, job: JobRef, workers: u32) {
        // SAFETY: no worker can be *reading* the slot here.
        //
        // Note that this does not follow from `pending == 0` on its own: a
        // worker is still inside `finish_shard` for a moment after the
        // `fetch_sub` that drives `pending` to zero, so "the barrier
        // retired" is not "every worker has left `finish_shard`". What makes
        // the write sound is the ordering *inside* the worker: `worker_loop`
        // reads this slot strictly before the `fetch_sub` that retires its
        // shard, and does not touch it again. So the previous barrier's
        // observation of `pending == 0` — an `Acquire`/`SeqCst` load reading
        // the value written by that release `fetch_sub` — happens-after
        // every worker's read of the slot, and this write happens-after that
        // observation. `run` takes `&mut self`, so there is no second
        // submitter. The `SeqCst` `seq` store below releases this write to
        // the workers that are about to read it.
        unsafe {
            *self.job.get() = Some(job);
        }
        self.pending.store(workers, Ordering::Relaxed);
        let next = self.seq.load(Ordering::Relaxed).wrapping_add(1);
        self.seq.store(next, Ordering::SeqCst);
        if self.workers_parked.load(Ordering::SeqCst) > 0 {
            park::wake(&self.seq, workers.min(WAKE_ALL));
        }
    }

    /// Retire this worker's shard; wake the submitter if it was the last.
    ///
    /// The flag is **read, not taken**. Taking it (`swap(0)`) is what let a
    /// worker still inside this function from the *previous* job consume the
    /// flag the submitter had just set for the current one, wake nobody, and
    /// strand the submitter forever. Only the submitter clears its own flag;
    /// see the `Shared` type docs for the full argument.
    fn finish_shard(&self) {
        if self.pending.fetch_sub(1, Ordering::SeqCst) == 1 {
            // The window the cross-generation regression test forces open.
            #[cfg(test)]
            let stalled = self.probe.enter_wake_window();
            let parked = self.lead_parked.load(Ordering::SeqCst) != 0;
            if parked {
                park::wake(&self.pending, 1);
            }
            #[cfg(test)]
            if stalled {
                self.probe.leave_wake_window(parked);
            }
        }
    }

    /// Block until every worker has retired its shard.
    ///
    /// `lead_parked` is set once on entering the parked phase and cleared
    /// once on leaving it, both from this thread. Nothing else ever writes
    /// it, which is what keeps the flag tied to this call rather than to
    /// whichever job happens to be in flight when some worker gets around to
    /// looking at it.
    fn await_barrier(&self) {
        for _ in 0..SPIN_ROUNDS {
            if self.pending.load(Ordering::Acquire) == 0 {
                return;
            }
            std::hint::spin_loop();
        }
        // Dekker: this store is ordered before the `SeqCst` load below, and
        // a worker's `fetch_sub` is ordered before its load of the flag.
        self.lead_parked.store(1, Ordering::SeqCst);
        loop {
            let outstanding = self.pending.load(Ordering::SeqCst);
            if outstanding == 0 {
                break;
            }
            #[cfg(test)]
            self.probe.enter_park_window(&self.lead_parked);
            // Bounded: a lost wake costs a millisecond, not the process.
            park::wait(&self.pending, outstanding, BARRIER_POLL);
        }
        self.lead_parked.store(0, Ordering::SeqCst);
    }

    /// Block until `seq` differs from `last`, returning the new value.
    fn await_seq(&self, last: u32) -> u32 {
        for _ in 0..SPIN_ROUNDS {
            let seq = self.seq.load(Ordering::Acquire);
            if seq != last {
                return seq;
            }
            std::hint::spin_loop();
        }
        let mut poll = JOB_POLL_MIN;
        loop {
            self.workers_parked.fetch_add(1, Ordering::SeqCst);
            let seq = self.seq.load(Ordering::SeqCst);
            if seq != last {
                self.workers_parked.fetch_sub(1, Ordering::SeqCst);
                return seq;
            }
            park::wait(&self.seq, last, poll);
            poll = (poll * 2).min(JOB_POLL_MAX);
            self.workers_parked.fetch_sub(1, Ordering::SeqCst);
            let seq = self.seq.load(Ordering::Acquire);
            if seq != last {
                return seq;
            }
        }
    }
}

/// Waits out the barrier no matter how `run` leaves its body.
///
/// This is the whole reason borrowed data can be handed to threads that
/// outlive it: `run` cannot return, break, or unwind past this guard while
/// a worker might still hold the erased closure pointer.
struct BarrierGuard<'a> {
    shared: &'a Shared,
}

impl Drop for BarrierGuard<'_> {
    fn drop(&mut self) {
        self.shared.await_barrier();
        // SAFETY: `pending == 0`, so no worker holds the erased pointer any
        // more and the submitter is the only live accessor of the slot.
        unsafe {
            *self.shared.job.get() = None;
        }
        if std::thread::panicking() {
            // The caller's own panic wins; drop any worker payload rather
            // than double-panicking out of a `Drop`.
            //
            // Dropping it here does run the payload's own `Drop` while this
            // thread is unwinding, so a payload that panics in `Drop` aborts
            // the process. That is not a barrier hazard - `await_barrier()`
            // above has already returned, so `pending == 0` and no worker is
            // stranded either way - but it is a real abort path, and this
            // module enumerates those rather than leaving them implicit. It
            // is accepted: a panic payload is whatever `panic!` was handed,
            // in practice a `String` or `&str`, and a type whose `Drop`
            // panics has no non-aborting disposal anywhere in a `Drop` body.
            if self.shared.take_panic().is_some() {
                warn!("discarding worker panic while the submitting thread unwinds");
            }
        }
    }
}

/// How to build a [`ComputePool`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolConfig {
    /// Total shards per job, counting the submitting thread when
    /// `inline_caller` is set. `None` takes [`Topology::shard_count`].
    ///
    /// Clamped to `1..=`[`MAX_SHARDS`]. This is a caller-supplied number
    /// rather than untrusted input, but `Some(usize::MAX)` would otherwise
    /// reach `Vec::with_capacity`, and a capacity overflow aborts the
    /// process rather than unwinding — which is not something a `#[must_use]`
    /// constructor documented as infallible is allowed to do.
    pub shards: Option<usize>,
    /// Attempt to pin worker threads. Failures degrade to unpinned.
    pub pin: bool,
    /// Also pin the thread that constructs the pool to the lead compute CPU.
    ///
    /// Only meaningful with `inline_caller`; the constructing thread is
    /// assumed to be the decode thread. Use
    /// [`ComputePool::pin_calling_thread`] if it is not.
    pub pin_caller: bool,
    /// Run shard 0 on the submitting thread instead of a worker.
    ///
    /// This is the ggml/llama.cpp shape: N shards means N-1 spawned threads
    /// plus the decode thread, one per physical P-core, and it removes one
    /// wake and one barrier hop from every GEMV.
    pub inline_caller: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            shards: None,
            pin: true,
            pin_caller: true,
            inline_caller: true,
        }
    }
}

/// A pool of persistent, optionally pinned worker threads.
///
/// Workers park between jobs and wake to run a closure over a contiguous row
/// range. See the module docs for the cost model and for why handing
/// non-`'static` borrows to threads that outlive the borrow is sound.
///
/// `run` takes `&mut self` on purpose: it is what statically rules out two
/// jobs in flight at once and re-entrant submission from inside a job, both
/// of which would break the borrow argument.
pub struct ComputePool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    topology: Topology,
    shards: usize,
    inline_caller: bool,
    lead_cpu: Option<usize>,
}

impl ComputePool {
    /// Detect the topology and build a pool with the default config.
    ///
    /// Pins the calling thread to the lead compute CPU (see
    /// [`PoolConfig::pin_caller`]). Never fails: an undetectable topology, a
    /// restricted cpuset, or a failed `pthread_create` all degrade to a
    /// smaller, unpinned pool.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(PoolConfig::default())
    }

    /// Detect the topology and build a pool with `config`.
    ///
    /// Never fails, and — unlike "never returns an error" — never aborts
    /// either: [`PoolConfig::shards`] is clamped to [`MAX_SHARDS`] before it
    /// reaches an allocation or a spawn loop, and a spawn that fails part
    /// way through shrinks the pool with a warning instead of propagating.
    #[must_use]
    pub fn with_config(config: PoolConfig) -> Self {
        let topology = Topology::detect();
        Self::with_topology(topology, config)
    }

    /// Build a pool over an already-detected topology.
    ///
    /// See [`ComputePool::with_config`] for the clamping and degradation
    /// rules; this is the same constructor with detection already done.
    #[must_use]
    pub fn with_topology(topology: Topology, config: PoolConfig) -> Self {
        let shards = config
            .shards
            .unwrap_or_else(|| topology.shard_count())
            .clamp(1, MAX_SHARDS);
        let inline_caller = config.inline_caller;
        let pin = config.pin && topology.pins();
        let compute = topology.compute_cpus().to_vec();

        let offset = usize::from(inline_caller);
        let worker_count = shards.saturating_sub(offset);
        let lead_cpu = if pin && inline_caller {
            compute.first().copied()
        } else {
            None
        };

        let shared = Arc::new(Shared::new());
        let mut workers = Vec::with_capacity(worker_count);
        for i in 0..worker_count {
            let shard_index = i + offset;
            let cpu = if pin {
                compute.get(shard_index).copied()
            } else {
                None
            };
            let shared = Arc::clone(&shared);
            let spawned = std::thread::Builder::new()
                .name(format!("rvmp-compute-{shard_index}"))
                .spawn(move || worker_loop(&shared, shard_index, cpu));
            match spawned {
                Ok(handle) => workers.push(handle),
                Err(err) => {
                    warn!(
                        error = %err,
                        wanted = worker_count,
                        got = workers.len(),
                        "could not spawn compute worker; shrinking the pool"
                    );
                    break;
                }
            }
        }

        let shards = workers.len() + offset;
        let pool = Self {
            shared,
            workers,
            topology,
            shards: shards.max(1),
            inline_caller,
            lead_cpu,
        };

        if config.pin_caller
            && let Some(cpu) = pool.lead_cpu
        {
            pin_or_warn(cpu, "compute-lead");
        }
        debug!(
            shards = pool.shards,
            workers = pool.workers.len(),
            inline_caller,
            "compute pool ready"
        );
        pool
    }

    /// Total shards a job is split into, counting the submitting thread.
    #[must_use]
    pub fn shards(&self) -> usize {
        self.shards
    }

    /// The topology this pool was built from.
    #[must_use]
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// The CPU the submitting thread should run on, if pinning is in effect.
    #[must_use]
    pub fn lead_cpu(&self) -> Option<usize> {
        self.lead_cpu
    }

    /// Pin the calling thread to [`ComputePool::lead_cpu`].
    ///
    /// For the case where the pool is built on a different thread than the
    /// one that will drive the decode loop. Returns whether a pin happened.
    pub fn pin_calling_thread(&self) -> bool {
        match self.lead_cpu {
            Some(cpu) => pin_or_warn(cpu, "compute-lead"),
            None => false,
        }
    }

    /// Run `f` once per shard of `0..rows` and return when all have finished.
    ///
    /// Shards are the [`shard_range`] split, so the row partition is a pure
    /// function of `(rows, shards())` and results are reproducible.
    /// Empty shards are skipped; `rows == 0` calls `f` not at all.
    ///
    /// Jobs smaller than one row per shard run entirely on the calling
    /// thread as a single shard, which is both cheaper and still a
    /// deterministic function of `(rows, shards())`.
    ///
    /// # Panics
    ///
    /// If `f` panics on any shard the panic is re-raised on this thread
    /// after the barrier completes, so the pool is never left poisoned or
    /// deadlocked. A panic in the caller's own shard wins; a worker's is
    /// then discarded with a warning.
    pub fn run<F>(&mut self, rows: usize, f: F)
    where
        F: Fn(Shard) + Sync,
    {
        if rows == 0 {
            return;
        }
        let shards = self.shards;
        let inline_caller = self.inline_caller;
        let workers = self.workers.len() as u32;
        if workers == 0 || rows < shards {
            f(Shard {
                index: 0,
                count: 1,
                rows: 0..rows,
            });
            return;
        }

        let job = JobRef {
            data: std::ptr::from_ref(&f).cast::<()>(),
            call: call_job::<F>,
            rows,
            shards,
        };

        let shared: &Shared = &self.shared;
        // The guard is armed *before* the job is published, not after.
        // Nothing in `publish` can panic today, so the erased `&f` could not
        // actually escape through that gap — but the entire soundness
        // argument for the erased pointer is "the guard exists from the
        // moment the workers can see it", and an argument that depends on
        // auditing the callee for panics is one edit away from being wrong.
        // Draining a barrier that was never published is a no-op
        // (`pending == 0`), so arming early is free.
        let guard = BarrierGuard { shared };
        shared.publish(job, workers);
        if inline_caller {
            f(Shard {
                index: 0,
                count: shards,
                rows: shard_range(rows, shards, 0),
            });
        }
        drop(guard);

        if let Some(payload) = shared.take_panic() {
            resume_unwind(payload);
        }
    }

    /// Split `out` into disjoint contiguous chunks and run `f` on each.
    ///
    /// The chunk handed to a shard is exactly `out[shard.rows]`, so this is
    /// the natural shape for a row-per-output-element GEMV.
    ///
    /// # Panics
    ///
    /// Propagates a closure panic exactly as [`ComputePool::run`] does.
    pub fn scatter<T, F>(&mut self, out: &mut [T], f: F)
    where
        T: Send,
        F: Fn(Shard, &mut [T]) + Sync,
    {
        let rows = out.len();
        let base = SendPtr(out.as_mut_ptr());
        self.run(rows, |shard| {
            let start = shard.rows.start;
            let len = shard.rows.len();
            // SAFETY: `shard_range` yields disjoint sub-ranges of `0..rows`
            // and each shard is visited by exactly one thread, so the
            // reconstructed `&mut [T]` never aliases another shard's. `out`
            // is mutably borrowed for the whole call, so nothing outside can
            // alias it either, and `run` does not return until every shard
            // has dropped its slice.
            let chunk = unsafe { std::slice::from_raw_parts_mut(base.get().add(start), len) };
            f(shard, chunk);
        });
    }
}

impl Default for ComputePool {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ComputePool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputePool")
            .field("shards", &self.shards)
            .field("workers", &self.workers.len())
            .field("inline_caller", &self.inline_caller)
            .field("lead_cpu", &self.lead_cpu)
            .finish()
    }
}

impl Drop for ComputePool {
    fn drop(&mut self) {
        // `&mut self` means no job is in flight, so the workers are parked
        // or spinning on `seq` and nothing reads the job slot.
        self.shared.quit.store(1, Ordering::Release);
        let next = self.shared.seq.load(Ordering::Relaxed).wrapping_add(1);
        self.shared.seq.store(next, Ordering::SeqCst);
        park::wake(&self.shared.seq, WAKE_ALL);
        for handle in self.workers.drain(..) {
            if handle.join().is_err() {
                warn!("compute worker thread terminated abnormally");
            }
        }
    }
}

/// A raw pointer that is safe to share because the pool guarantees the
/// sub-slices derived from it are disjoint and outlived by the borrow.
struct SendPtr<T>(*mut T);

impl<T> SendPtr<T> {
    /// The wrapped pointer.
    ///
    /// Taking it through a method rather than the tuple field is deliberate:
    /// it makes the closure in `scatter` capture the whole `SendPtr` (which
    /// is `Sync`) instead of precise-capturing the bare `*mut T` (which is
    /// not).
    fn get(&self) -> *mut T {
        self.0
    }
}

// SAFETY: the pointer is only ever turned into disjoint `&mut [T]` chunks by
// `ComputePool::scatter`, under a `&mut [T]` borrow that outlives the job.
unsafe impl<T: Send> Send for SendPtr<T> {}
// SAFETY: see `Send`.
unsafe impl<T: Send> Sync for SendPtr<T> {}

/// Retires one shard on every exit path, the way [`BarrierGuard`] retires
/// the job.
///
/// A tail call to `finish_shard` is correct only as long as nothing above it
/// can unwind, which is a property of the code as written rather than of the
/// protocol. This makes the decrement structural.
struct ShardGuard<'a> {
    shared: &'a Shared,
}

impl Drop for ShardGuard<'_> {
    fn drop(&mut self) {
        self.shared.finish_shard();
    }
}

fn worker_loop(shared: &Shared, shard_index: usize, cpu: Option<usize>) {
    if let Some(cpu) = cpu {
        pin_or_warn(cpu, "compute-worker");
    }
    let mut last = 0u32;
    loop {
        last = shared.await_seq(last);
        if shared.quit.load(Ordering::Acquire) != 0 {
            break;
        }
        // SAFETY: the submitter wrote this slot before the `seq` store we
        // just observed with `Acquire`, and cannot write it again until the
        // `finish_shard` below drives `pending` to zero. This read is
        // sequenced before that decrement, which is precisely what
        // `Shared::publish` relies on.
        let job = unsafe { *shared.job.get() };
        // From here on the decrement happens no matter how this iteration
        // ends.
        let _retire = ShardGuard { shared };
        if let Some(job) = job {
            let rows = shard_range(job.rows, job.shards, shard_index);
            if !rows.is_empty() {
                let shard = Shard {
                    index: shard_index,
                    count: job.shards,
                    rows,
                };
                // `AssertUnwindSafe`: the closure's captures are the
                // caller's, and a panic here is re-raised on the caller's
                // thread, so no torn state is ever observed by anyone who
                // did not also unwind.
                let result = catch_unwind(AssertUnwindSafe(|| {
                    // SAFETY: `job.data` came from `&F` in `run`, which is
                    // blocked on the barrier until this call returns.
                    unsafe { (job.call)(job.data, shard) }
                }));
                if let Err(payload) = result {
                    // Recording can itself unwind, in exactly one way: the
                    // payload's own `Drop` panics when `record_panic` drops
                    // it (the slot was already occupied). `ShardGuard` would
                    // still retire the shard, but the unwind would leave
                    // `worker_loop` and kill the thread, and the pool sizes
                    // its `pending` from the number of join handles, not
                    // from the number of live threads — so the *next* job
                    // would never retire. Contain it here instead.
                    if let Err(second) =
                        catch_unwind(AssertUnwindSafe(|| shared.record_panic(payload)))
                    {
                        // Deliberately leaked, not dropped: dropping it
                        // would re-enter the same panicking destructor with
                        // no `catch_unwind` left to catch it.
                        std::mem::forget(second);
                        warn!("panic while recording a worker panic; payload leaked");
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, AtomicUsize};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    /// How long a probe hand-off waits before giving up.
    ///
    /// Only reached if the scenario failed to set itself up; the assertions
    /// then report *that* rather than letting the test hang.
    const PROBE_DEADLINE: Duration = Duration::from_secs(10);

    /// Test-only control over the `finish_shard` / `await_barrier`
    /// interleaving, forcing the cross-generation window open on demand.
    ///
    /// It lives inside [`Shared`] rather than in a `static` so that two
    /// tests running in parallel — which is cargo's default — cannot arm
    /// each other's pool. Every field is inert until explicitly armed: an
    /// unarmed probe costs one uncontended swap per retired job and two
    /// loads per park iteration, and non-test builds do not have the field
    /// at all.
    #[derive(Debug, Default)]
    pub(super) struct BarrierProbe {
        /// One-shot: the next worker to drive `pending` to zero stalls
        /// between that decrement and its look at `lead_parked`.
        stall_worker: AtomicU32,
        /// Set by that worker once it is inside the window.
        worker_waiting: AtomicU32,
        /// Set by the submitter to let the stalled worker proceed.
        resume_worker: AtomicU32,
        /// Set by the worker once it has left the window.
        worker_left: AtomicU32,
        /// What the stalled worker saw in `lead_parked`. `1` means the
        /// submitter had already armed for the *next* job — the interleaving
        /// the regression test exists to force.
        worker_saw_flag: AtomicU32,
        /// One-shot: the submitter holds its arming window open until the
        /// stalled worker has been all the way through the wake window.
        stall_lead: AtomicU32,
        /// `lead_parked`, read by the submitter after the straggler has been
        /// all the way through the wake window. `1` means the straggler left
        /// the flag alone; `0` means it consumed it. This is the assertion
        /// that does not depend on `BARRIER_POLL` rescuing the process.
        flag_after_straggler: AtomicU32,
        /// Whether `flag_after_straggler` was ever written.
        flag_recorded: AtomicU32,
    }

    impl BarrierProbe {
        pub(super) fn new() -> Self {
            Self::default()
        }

        /// Stall the next worker that retires a job's last shard.
        fn arm_worker(&self) {
            self.stall_worker.store(1, Ordering::SeqCst);
        }

        /// Hold the next park of the submitter open until that worker has
        /// inspected `lead_parked`.
        fn arm_lead(&self) {
            self.stall_lead.store(1, Ordering::SeqCst);
        }

        /// Whether a straggler was actually held in the wake window and
        /// released again, i.e. whether the interleaving was exercised.
        fn straggler_went_through_window(&self) -> bool {
            self.worker_left.load(Ordering::SeqCst) != 0
        }

        fn worker_saw_flag(&self) -> bool {
            self.worker_saw_flag.load(Ordering::SeqCst) != 0
        }

        /// Whether the submitter's flag survived the straggler, and whether
        /// it was ever looked at.
        fn flag_survived_straggler(&self) -> (bool, bool) {
            (
                self.flag_recorded.load(Ordering::SeqCst) != 0,
                self.flag_after_straggler.load(Ordering::SeqCst) != 0,
            )
        }

        /// Called from `finish_shard`, between the decrement that retired
        /// the job and the look at `lead_parked`. Returns whether this
        /// thread is the stalled one.
        pub(super) fn enter_wake_window(&self) -> bool {
            if self.stall_worker.swap(0, Ordering::SeqCst) == 0 {
                return false;
            }
            self.worker_waiting.store(1, Ordering::SeqCst);
            wait_for(&self.resume_worker);
            true
        }

        /// Called from `finish_shard` once the flag has been inspected.
        pub(super) fn leave_wake_window(&self, saw_flag: bool) {
            self.worker_saw_flag
                .store(u32::from(saw_flag), Ordering::SeqCst);
            self.worker_left.store(1, Ordering::SeqCst);
        }

        /// Called from `await_barrier` after `lead_parked` is armed and a
        /// non-zero `pending` has been read, i.e. in the few hundred
        /// nanoseconds before the submitter is actually enqueued on the
        /// futex. Releases the stalled worker into exactly that window, then
        /// records whether its own flag is still armed afterwards.
        pub(super) fn enter_park_window(&self, lead_parked: &AtomicU32) {
            if self.stall_lead.load(Ordering::SeqCst) == 0
                || self.worker_waiting.load(Ordering::SeqCst) == 0
            {
                return;
            }
            self.resume_worker.store(1, Ordering::SeqCst);
            wait_for(&self.worker_left);
            self.flag_after_straggler
                .store(lead_parked.load(Ordering::SeqCst), Ordering::SeqCst);
            self.flag_recorded.store(1, Ordering::SeqCst);
            self.stall_lead.store(0, Ordering::SeqCst);
        }
    }

    /// Sleep (never spin: the stress runs use a three-CPU cpuset) until
    /// `flag` is set or [`PROBE_DEADLINE`] passes.
    fn wait_for(flag: &AtomicU32) {
        let deadline = Instant::now() + PROBE_DEADLINE;
        while flag.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// Unique directory under the system temp dir, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "ramvamp-threads-{tag}-{}-{n}-{nanos}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create test temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_file(root: &Path, rel: &str, contents: &str) {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create fixture dir");
        }
        fs::write(&path, contents).expect("write fixture file");
    }

    /// A cpu with a sibling list and an optional cache level tree.
    ///
    /// Mirrors real sysfs, including the plain `uevent` file that sits
    /// alongside the `index*` directories — joining `level` onto it yields
    /// `ENOTDIR`, which a naive cache walk turns into a hard error.
    fn write_cpu(root: &Path, cpu: usize, siblings: &str, cache_levels: &[u32]) {
        write_file(
            root,
            &format!("devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"),
            siblings,
        );
        write_file(
            root,
            &format!("devices/system/cpu/cpu{cpu}/cache/uevent"),
            "MAJOR=0\n",
        );
        for (i, level) in cache_levels.iter().enumerate() {
            write_file(
                root,
                &format!("devices/system/cpu/cpu{cpu}/cache/index{i}/level"),
                &format!("{level}\n"),
            );
        }
    }

    /// The reference machine: Core Ultra 9 185H.
    ///
    /// P-cores 0-11 in SMT pairs (0,5) (1,2) (3,4) (6,7) (8,9) (10,11);
    /// E-cores 12-19 sharing L3; LP E-cores 20-21 with no L3 index at all.
    fn reference_sysfs() -> TempDir {
        let tmp = TempDir::new("185h");
        let root = tmp.path();
        write_file(root, "devices/cpu_core/cpus", "0-11\n");
        write_file(root, "devices/cpu_atom/cpus", "12-21\n");
        write_file(root, "devices/system/cpu/online", "0-21\n");

        const PAIRS: [(usize, usize); 6] = [(0, 5), (1, 2), (3, 4), (6, 7), (8, 9), (10, 11)];
        for (a, b) in PAIRS {
            let list = format!("{a},{b}\n");
            write_cpu(root, a, &list, &[1, 1, 2, 3]);
            write_cpu(root, b, &list, &[1, 1, 2, 3]);
        }
        for cpu in 12..=19 {
            write_cpu(root, cpu, &format!("{cpu}\n"), &[1, 1, 2, 3]);
        }
        // LP E-cores: L1 and L2 only, no L3 index.
        for cpu in 20..=21 {
            write_cpu(root, cpu, &format!("{cpu}\n"), &[1, 1, 2]);
        }
        tmp
    }

    /// A plain 4-core-8-thread non-hybrid part.
    fn uniform_sysfs() -> TempDir {
        let tmp = TempDir::new("uniform");
        let root = tmp.path();
        write_file(root, "devices/system/cpu/online", "0-7\n");
        for core in 0..4 {
            let (a, b) = (core, core + 4);
            let list = format!("{a},{b}\n");
            write_cpu(root, a, &list, &[1, 1, 2, 3]);
            write_cpu(root, b, &list, &[1, 1, 2, 3]);
        }
        tmp
    }

    // --- cpu list parsing ---------------------------------------------------

    #[test]
    fn cpu_lists_parse_every_sysfs_spelling() {
        assert_eq!(parse_cpu_list("0-11"), Ok((0..=11).collect::<Vec<_>>()));
        assert_eq!(parse_cpu_list("0,5"), Ok(vec![0, 5]));
        assert_eq!(
            parse_cpu_list("0-3,8,10-11"),
            Ok(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list(" 12 "), Ok(vec![12]));
        assert_eq!(parse_cpu_list(""), Ok(Vec::new()));
        assert_eq!(parse_cpu_list("  \n"), Ok(Vec::new()));
        // Duplicates collapse and order is normalised.
        assert_eq!(parse_cpu_list("5,1-2,1"), Ok(vec![1, 2, 5]));
    }

    #[test]
    fn malformed_cpu_lists_are_rejected_not_panicked_on() {
        assert!(parse_cpu_list("abc").is_err());
        assert!(parse_cpu_list("0-").is_err());
        assert!(parse_cpu_list("-3").is_err());
        assert!(parse_cpu_list("3-1").is_err());
        assert!(parse_cpu_list("0,,3").is_err());
        assert!(parse_cpu_list("0-11 12").is_err());
        assert!(parse_cpu_list("99999999999999999999").is_err());
        // Beyond the CpuSet bit array, which would panic `CpuSet::set`.
        assert!(parse_cpu_list(&format!("{}", MAX_CPUS)).is_err());
        assert!(parse_cpu_list(&format!("0-{}", MAX_CPUS)).is_err());
        assert!(parse_cpu_list(&format!("{}", MAX_CPUS - 1)).is_ok());
    }

    #[test]
    fn cpu_lists_round_trip_through_the_formatter() {
        for text in ["0-11", "0,5", "0-3,8,10-11", "12"] {
            let parsed = parse_cpu_list(text).expect("parse");
            assert_eq!(format_cpu_list(&parsed), text, "round trip of {text:?}");
        }
        assert_eq!(format_cpu_list(&[]), "-");
    }

    // --- topology parsing ---------------------------------------------------

    #[test]
    fn reference_machine_yields_the_expected_smt_primaries() {
        let tmp = reference_sysfs();
        let topo = Topology::parse_at(tmp.path(), CpuMask::up_to(22)).expect("parse");

        assert!(topo.is_hybrid());
        assert!(topo.pins());
        assert_eq!(topo.performance_cpus(), &(0..=11).collect::<Vec<_>>()[..]);
        assert_eq!(topo.efficiency_cpus(), &(12..=21).collect::<Vec<_>>()[..]);
        // 20 and 21 are the LP tile: no L3 index in sysfs.
        assert_eq!(
            topo.l3_efficiency_cpus(),
            &(12..=19).collect::<Vec<_>>()[..]
        );
        assert_eq!(topo.compute_cpus(), [0, 1, 3, 6, 8, 10]);
        assert_eq!(topo.reactor_cpu(), Some(12));
        assert_eq!(topo.shard_count(), 6);
    }

    #[test]
    fn topology_display_is_readable_for_the_cli() {
        let tmp = reference_sysfs();
        let topo = Topology::parse_at(tmp.path(), CpuMask::up_to(22)).expect("parse");
        let text = topo.to_string();
        assert!(text.contains("hybrid"), "{text}");
        assert!(text.contains("P 0-11"), "{text}");
        assert!(text.contains("E 12-21 (L3 12-19)"), "{text}");
        assert!(text.contains("compute 0-1,3,6,8,10"), "{text}");
        assert!(text.contains("reactor 12"), "{text}");
        assert!(text.contains("pinning on"), "{text}");
    }

    #[test]
    fn non_hybrid_machines_degrade_to_no_pinning() {
        let tmp = uniform_sysfs();
        let topo = Topology::parse_at(tmp.path(), CpuMask::up_to(8)).expect("parse");

        assert!(!topo.is_hybrid());
        assert!(!topo.pins());
        assert!(topo.compute_cpus().is_empty());
        assert!(topo.performance_cpus().is_empty());
        assert!(topo.efficiency_cpus().is_empty());
        assert_eq!(topo.reactor_cpu(), None);
        assert!(topo.shard_count() >= 1);
        assert!(topo.shard_count() <= 8);
    }

    #[test]
    fn missing_sysfs_degrades_instead_of_failing() {
        let tmp = TempDir::new("absent");
        let root = tmp.path().join("does-not-exist");
        let topo = Topology::detect_at(&root, CpuMask::up_to(4));
        assert!(!topo.pins());
        assert!(topo.compute_cpus().is_empty());
        assert_eq!(topo.reactor_cpu(), None);
        assert!(topo.shard_count() >= 1);
    }

    #[test]
    fn a_malformed_sibling_list_falls_through_to_the_other_spelling() {
        // One bad `thread_siblings_list` must not discard a machine whose
        // `core_cpus_list` says the same thing: the failure policy is "no
        // single unreadable file decides the topology".
        let tmp = reference_sysfs();
        write_file(
            tmp.path(),
            "devices/system/cpu/cpu3/topology/thread_siblings_list",
            "three-and-four\n",
        );
        write_file(
            tmp.path(),
            "devices/system/cpu/cpu3/topology/core_cpus_list",
            "3,4\n",
        );

        let topo = Topology::parse_at(tmp.path(), CpuMask::up_to(22)).expect("parse");
        assert!(topo.pins());
        assert_eq!(topo.compute_cpus(), [0, 1, 3, 6, 8, 10]);
    }

    #[test]
    fn unknowable_smt_siblings_disable_pinning_rather_than_double_booking() {
        // The container shape: `devices/cpu_core/cpus` is there, but
        // `cpuN/topology/` is not. Guessing "each cpu is its own core" would
        // promote both halves of every SMT pair and put twelve pinned
        // threads on six physical cores.
        for broken in ["three-and-four\n", ""] {
            let tmp = reference_sysfs();
            for name in ["thread_siblings_list", "core_cpus_list"] {
                write_file(
                    tmp.path(),
                    &format!("devices/system/cpu/cpu3/topology/{name}"),
                    broken,
                );
            }

            let topo = Topology::parse_at(tmp.path(), CpuMask::up_to(22)).expect("parse");
            assert!(!topo.pins(), "{broken:?} still pinned: {topo}");
            assert!(topo.compute_cpus().is_empty(), "{topo}");
            // Everything that does not depend on the SMT layout survives.
            assert!(topo.is_hybrid());
            assert_eq!(topo.performance_cpus(), &(0..=11).collect::<Vec<_>>()[..]);
            assert_eq!(topo.reactor_cpu(), Some(12));
            assert!(topo.shard_count() >= 1);
        }
    }

    #[test]
    fn a_missing_topology_directory_disables_pinning() {
        let tmp = reference_sysfs();
        fs::remove_dir_all(tmp.path().join("devices/system/cpu/cpu0/topology"))
            .expect("drop topology dir");
        let topo = Topology::parse_at(tmp.path(), CpuMask::up_to(22)).expect("parse");
        assert!(!topo.pins(), "{topo}");
        assert!(topo.compute_cpus().is_empty(), "{topo}");
        assert!(!Topology::detect_at(tmp.path(), CpuMask::up_to(22)).pins());
    }

    #[test]
    fn malformed_pmu_cpu_list_fails_parse_and_degrades_detect() {
        let tmp = reference_sysfs();
        write_file(tmp.path(), "devices/cpu_core/cpus", "0-11,oops\n");
        assert!(Topology::parse_at(tmp.path(), CpuMask::up_to(22)).is_err());
        assert!(!Topology::detect_at(tmp.path(), CpuMask::up_to(22)).pins());
    }

    #[test]
    fn topology_never_leaves_the_affinity_mask() {
        let tmp = reference_sysfs();
        // A cpuset holding only half of some SMT pairs.
        let allowed: CpuMask = [0, 1, 2, 3, 13].into_iter().collect();
        let topo = Topology::parse_at(tmp.path(), allowed.clone()).expect("parse");

        // 5 is 0's sibling and 4 is 3's, but neither is allowed, so each
        // physical core still contributes exactly one allowed primary.
        assert_eq!(topo.compute_cpus(), [0, 1, 3]);
        assert_eq!(topo.performance_cpus(), [0, 1, 2, 3]);
        assert_eq!(topo.efficiency_cpus(), [13]);
        assert_eq!(topo.reactor_cpu(), Some(13));
        for cpu in topo
            .compute_cpus()
            .iter()
            .chain(topo.performance_cpus())
            .chain(topo.efficiency_cpus())
            .chain(topo.l3_efficiency_cpus())
            .chain(topo.reactor_cpu().iter())
        {
            assert!(allowed.contains(*cpu), "cpu {cpu} escaped the mask");
        }
    }

    #[test]
    fn a_single_allowed_cpu_disables_pinning() {
        let tmp = reference_sysfs();
        let topo = Topology::parse_at(tmp.path(), [3].into_iter().collect()).expect("parse");
        assert!(topo.is_hybrid());
        assert!(!topo.pins());
        assert!(topo.compute_cpus().is_empty());
        assert_eq!(topo.shard_count(), 1);
    }

    #[test]
    fn an_empty_affinity_mask_is_an_error_not_a_panic() {
        let tmp = reference_sysfs();
        let err = Topology::parse_at(tmp.path(), CpuMask::new()).expect_err("must fail");
        assert!(matches!(err, TopologyError::NoCpus), "{err:?}");
    }

    #[test]
    fn hybrid_without_any_l3_ecore_leaves_the_reactor_unpinned() {
        let tmp = reference_sysfs();
        // Demote every E-core to the LP tile: drop its L3 cache index.
        for cpu in 12..=19 {
            let dir = tmp
                .path()
                .join(format!("devices/system/cpu/cpu{cpu}/cache/index3"));
            fs::remove_dir_all(&dir).expect("drop l3 index");
        }
        let topo = Topology::parse_at(tmp.path(), CpuMask::up_to(22)).expect("parse");
        assert!(topo.l3_efficiency_cpus().is_empty());
        assert_eq!(topo.reactor_cpu(), None);
        // Compute pinning is unaffected.
        assert_eq!(topo.compute_cpus(), [0, 1, 3, 6, 8, 10]);
    }

    #[test]
    fn detect_on_this_host_never_panics() {
        let topo = Topology::detect();
        assert!(topo.shard_count() >= 1);
        for cpu in topo.compute_cpus() {
            assert!(topo.allowed().contains(*cpu));
        }
        if let Some(cpu) = topo.reactor_cpu() {
            assert!(topo.allowed().contains(cpu));
        }

        // Host-conditional, and deliberately so: on a machine whose kernel
        // does expose a hybrid PMU and where we may use at least two
        // physical P-cores, a silent fall back to "unpinned" is a detection
        // bug, not graceful degradation. Anywhere else this is a no-op.
        let host_p_cores: Vec<usize> = fs::read_to_string("/sys/devices/cpu_core/cpus")
            .ok()
            .and_then(|text| parse_cpu_list(text.trim()).ok())
            .unwrap_or_default();
        let hybrid_host =
            !host_p_cores.is_empty() && Path::new("/sys/devices/cpu_atom/cpus").exists();
        let usable_p = host_p_cores
            .iter()
            .filter(|cpu| topo.allowed().contains(**cpu))
            .count();
        // Four allowed P-core threads is at least two physical cores.
        if hybrid_host && usable_p >= 4 {
            assert!(topo.is_hybrid(), "hybrid host reported as uniform: {topo}");
            assert!(topo.pins(), "hybrid host left unpinned: {topo}");
            assert!(topo.compute_cpus().len() >= 2, "{topo}");
            assert!(topo.reactor_cpu().is_some(), "{topo}");
        }
    }

    // --- masks and pinning --------------------------------------------------

    #[test]
    fn cpu_masks_intersect() {
        let a: CpuMask = [0, 1, 3, 6, 8, 10].into_iter().collect();
        let b: CpuMask = [1, 2, 3, 4, 10].into_iter().collect();
        assert_eq!(a.intersect(&b).to_vec(), [1, 3, 10]);
        assert_eq!(b.intersect(&a), a.intersect(&b));
        assert!(a.intersect(&CpuMask::new()).is_empty());
        assert_eq!(a.len(), 6);
        assert!(a.contains(8));
        assert!(!a.contains(9));
    }

    #[test]
    fn cpu_masks_refuse_ids_that_would_overflow_the_bit_array() {
        let mut mask = CpuMask::new();
        assert!(mask.insert(MAX_CPUS - 1));
        assert!(!mask.insert(MAX_CPUS));
        assert!(!mask.insert(usize::MAX));
        assert_eq!(mask.to_vec(), [MAX_CPUS - 1]);
        assert_eq!(CpuMask::up_to(usize::MAX).len(), MAX_CPUS);
    }

    #[test]
    fn pinning_out_of_range_is_a_typed_error() {
        let err = pin_current_thread(MAX_CPUS).expect_err("must fail");
        assert!(matches!(err, PinError::OutOfRange { .. }), "{err:?}");
        let err = pin_current_thread_to(&CpuMask::new()).expect_err("must fail");
        assert!(matches!(err, PinError::EmptyMask), "{err:?}");
    }

    #[test]
    fn pinning_outside_the_affinity_mask_is_refused() {
        let Some(allowed) = CpuMask::current() else {
            return; // No affinity API on this platform.
        };
        let outside = (0..MAX_CPUS).find(|c| !allowed.contains(*c));
        if let Some(cpu) = outside {
            let err = pin_current_thread(cpu).expect_err("must fail");
            assert!(matches!(err, PinError::NotAllowed { .. }), "{err:?}");
        }
    }

    // --- partitioning -------------------------------------------------------

    #[test]
    fn shard_ranges_tile_the_row_space_exactly_once() {
        for rows in [0usize, 1, 2, 3, 7, 8, 64, 127, 1000, 4096] {
            for shards in [1usize, 2, 3, 5, 6, 8, 16] {
                let mut next = 0;
                let mut seen = 0;
                for i in 0..shards {
                    let r = shard_range(rows, shards, i);
                    assert_eq!(r.start, next, "rows={rows} shards={shards} i={i}");
                    assert!(r.end >= r.start);
                    next = r.end;
                    seen += r.len();
                }
                assert_eq!(next, rows, "rows={rows} shards={shards}");
                assert_eq!(seen, rows, "rows={rows} shards={shards}");
            }
        }
    }

    #[test]
    fn shard_ranges_are_balanced_and_deterministic() {
        // 10 rows over 4 shards: 3,3,2,2 — never 4,2,2,2.
        let got: Vec<_> = (0..4).map(|i| shard_range(10, 4, i)).collect();
        assert_eq!(got, vec![0..3, 3..6, 6..8, 8..10]);
        // Same inputs, same answer, forever.
        for _ in 0..8 {
            assert_eq!(shard_range(10, 4, 1), 3..6);
        }
        assert_eq!(shard_range(10, 0, 0), 0..0);
        assert_eq!(shard_range(10, 4, 4), 0..0);
    }

    // --- pool ---------------------------------------------------------------

    /// A pool with a fixed shard count and no pinning, so tests behave the
    /// same on a 185H, a CI container, and a single-core box.
    fn test_pool(shards: usize) -> ComputePool {
        ComputePool::with_topology(
            Topology::unpinned(CpuMask::up_to(shards.max(1))),
            PoolConfig {
                shards: Some(shards),
                pin: false,
                pin_caller: false,
                inline_caller: true,
            },
        )
    }

    #[test]
    fn every_row_is_visited_exactly_once() {
        let mut pool = test_pool(4);
        assert_eq!(pool.shards(), 4);
        let rows = 1000;
        let visits: Vec<AtomicU32> = (0..rows).map(|_| AtomicU32::new(0)).collect();
        pool.run(rows, |shard| {
            for row in shard.rows.clone() {
                visits[row].fetch_add(1, Ordering::Relaxed);
            }
        });
        for (row, v) in visits.iter().enumerate() {
            assert_eq!(v.load(Ordering::Relaxed), 1, "row {row}");
        }
    }

    #[test]
    fn shards_are_disjoint_and_cover_the_job() {
        let mut pool = test_pool(6);
        let seen: Mutex<Vec<Range<usize>>> = Mutex::new(Vec::new());
        pool.run(97, |shard| {
            assert_eq!(shard.count, 6);
            assert!(!shard.is_empty());
            assert_eq!(shard.len(), shard.rows.len());
            seen.lock().expect("lock").push(shard.rows.clone());
        });
        let mut ranges = seen.into_inner().expect("into_inner");
        ranges.sort_by_key(|r| r.start);
        assert_eq!(ranges.len(), 6);
        let mut next = 0;
        for r in &ranges {
            assert_eq!(r.start, next);
            next = r.end;
        }
        assert_eq!(next, 97);
    }

    #[test]
    fn parallel_results_match_the_single_threaded_reference() {
        let rows = 2003;
        let weights: Vec<f32> = (0..rows).map(|i| (i as f32).mul_add(0.25, -3.0)).collect();

        let mut expected = vec![0.0f32; rows];
        for (i, out) in expected.iter_mut().enumerate() {
            *out = weights[i].mul_add(weights[i], weights[i]);
        }

        for shards in [1usize, 2, 3, 7] {
            let mut pool = test_pool(shards);
            let mut got = vec![0.0f32; rows];
            pool.scatter(&mut got, |shard, chunk| {
                for (k, slot) in chunk.iter_mut().enumerate() {
                    let w = weights[shard.rows.start + k];
                    *slot = w.mul_add(w, w);
                }
            });
            assert_eq!(got, expected, "shards={shards}");
        }
    }

    #[test]
    fn repeated_jobs_reuse_the_same_threads() {
        // Exercises the park/wake protocol thousands of times, which is the
        // per-GEMV cadence the decode loop runs at.
        let mut pool = test_pool(4);
        let rows = 128;
        let total = AtomicUsize::new(0);
        for _ in 0..2000 {
            pool.run(rows, |shard| {
                total.fetch_add(shard.rows.len(), Ordering::Relaxed);
            });
        }
        assert_eq!(total.load(Ordering::Relaxed), 2000 * rows);
    }

    /// Regression, cross-generation lost wakeup.
    ///
    /// A worker that is still inside `finish_shard` for job N must not be
    /// able to consume the park flag the submitter armed for job N+1.
    ///
    /// The interleaving is *forced*, not raced. [`BarrierProbe`] stalls the
    /// worker that drives `pending` to zero in the two-instruction window
    /// between its decrement and its look at `lead_parked`, and holds the
    /// submitter between arming that flag and enqueueing on the futex, so
    /// the straggler's inspection lands inside the submitter's arming window
    /// on every run rather than once in a few million barriers.
    ///
    /// Against the original `lead_parked.swap(0, SeqCst)` this deadlocks
    /// deterministically: the straggler takes job N+1's flag, its wake finds
    /// nobody enqueued, the submitter then parks with `lead_parked == 0`,
    /// and the last decrement of job N+1 finds no flag and issues no wake.
    /// With the flag read instead of taken, the straggler's inspection is
    /// inert and the real last decrement still sees the flag.
    ///
    /// The bounded [`BARRIER_POLL`] would *also* rescue the old code here,
    /// in about a millisecond — that is exactly what it is for. So progress
    /// alone is not the assertion. The load-bearing one is the protocol
    /// fact, checked with the submitter still holding its arming window
    /// open: after a straggler from the *previous* job has been all the way
    /// through the wake window, the submitter's flag is still armed. That
    /// fails against `swap(0)` whether or not the timeout is present.
    #[test]
    fn a_straggler_cannot_consume_the_next_jobs_park_flag() {
        let finished = Arc::new(AtomicU32::new(0));
        let stalled = Arc::new(AtomicU32::new(0));
        let saw_flag = Arc::new(AtomicU32::new(0));
        let flag_recorded = Arc::new(AtomicU32::new(0));
        let flag_survived = Arc::new(AtomicU32::new(0));
        let scenario = {
            let (finished, stalled, saw_flag, flag_recorded, flag_survived) = (
                Arc::clone(&finished),
                Arc::clone(&stalled),
                Arc::clone(&saw_flag),
                Arc::clone(&flag_recorded),
                Arc::clone(&flag_survived),
            );
            std::thread::spawn(move || {
                // Three shards, inline caller: two spawned workers.
                let mut pool = test_pool(3);

                // Job N. The submitter's own shard is slow, so both workers
                // have retired long before it reaches the barrier and it
                // resolves on the spin path — it never arms `lead_parked`
                // for job N at all, which is exactly what leaves the
                // straggler's flag write with nothing of its own to clear.
                pool.shared.probe.arm_worker();
                pool.run(300, |shard| {
                    if shard.index == 0 {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                });

                // Job N+1, submitted immediately, as the decode loop does.
                // The workers' shards are slow and the submitter's is not,
                // so the submitter is genuinely parked before any decrement
                // of this job can land.
                pool.shared.probe.arm_lead();
                pool.run(300, |shard| {
                    if shard.index != 0 {
                        std::thread::sleep(Duration::from_millis(150));
                    }
                });

                stalled.store(
                    u32::from(pool.shared.probe.straggler_went_through_window()),
                    Ordering::SeqCst,
                );
                saw_flag.store(
                    u32::from(pool.shared.probe.worker_saw_flag()),
                    Ordering::SeqCst,
                );
                let (recorded, survived) = pool.shared.probe.flag_survived_straggler();
                flag_recorded.store(u32::from(recorded), Ordering::SeqCst);
                flag_survived.store(u32::from(survived), Ordering::SeqCst);
                finished.store(1, Ordering::SeqCst);
            })
        };

        // Watchdog: a deadlocked pool must fail the test, not hang the run.
        let deadline = Instant::now() + Duration::from_secs(30);
        while finished.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            finished.load(Ordering::SeqCst),
            1,
            "the compute pool deadlocked: a straggler from the previous job \
             consumed the park flag armed for the current one"
        );
        scenario.join().expect("scenario thread");

        assert_eq!(
            stalled.load(Ordering::SeqCst),
            1,
            "the probe never stalled a worker in the wake window; the \
             interleaving under test was not exercised"
        );
        assert_eq!(
            saw_flag.load(Ordering::SeqCst),
            1,
            "the straggler did not observe the submitter's park flag, so it \
             was never in a position to consume it; the interleaving under \
             test was not exercised"
        );
        assert_eq!(
            flag_recorded.load(Ordering::SeqCst),
            1,
            "the submitter never reached its park window with a straggler \
             held in the wake window; the interleaving under test was not \
             exercised"
        );
        assert_eq!(
            flag_survived.load(Ordering::SeqCst),
            1,
            "a straggler from the previous job cleared the park flag armed \
             for the current one; the submitter is about to park with \
             `lead_parked == 0` and only `BARRIER_POLL` will get it out"
        );
    }

    #[test]
    fn tiny_and_empty_jobs_run_inline() {
        let mut pool = test_pool(6);
        let calls = AtomicUsize::new(0);
        pool.run(0, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        let shards = Mutex::new(Vec::new());
        pool.run(3, |shard| {
            shards.lock().expect("lock").push(shard);
        });
        let shards = shards.into_inner().expect("into_inner");
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].count, 1);
        assert_eq!(shards[0].rows, 0..3);
    }

    #[test]
    fn a_pool_with_one_shard_still_runs_the_job() {
        let mut pool = test_pool(1);
        assert_eq!(pool.shards(), 1);
        let sum = AtomicUsize::new(0);
        pool.run(10, |shard| {
            sum.fetch_add(shard.rows.len(), Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn borrowed_non_static_data_is_usable_from_the_pool() {
        // The point of the whole design: locals, not `'static`.
        let mut pool = test_pool(4);
        let weights = vec![2.0f32; 512];
        let bias = 0.5f32;
        let mut out = vec![0.0f32; 512];
        pool.scatter(&mut out, |shard, chunk| {
            for (k, slot) in chunk.iter_mut().enumerate() {
                *slot = weights[shard.rows.start + k] + bias;
            }
        });
        assert!(out.iter().all(|v| (*v - 2.5).abs() < 1e-6));
        drop(weights);
    }

    #[test]
    fn a_panicking_shard_neither_deadlocks_nor_poisons_the_pool() {
        let mut pool = test_pool(4);
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        // Panic on a shard that a worker owns, not the caller's shard 0.
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.run(400, |shard| {
                assert!(shard.index != 2, "boom");
            });
        }));
        // And on the caller's own shard.
        let caller = catch_unwind(AssertUnwindSafe(|| {
            pool.run(400, |shard| {
                assert!(shard.index != 0, "boom-lead");
            });
        }));
        std::panic::set_hook(hook);

        assert!(result.is_err(), "worker panic must propagate");
        assert!(caller.is_err(), "caller panic must propagate");

        // The pool is still fully usable and no stale payload is left over.
        let sum = AtomicUsize::new(0);
        pool.run(400, |shard| {
            sum.fetch_add(shard.rows.len(), Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 400);
    }

    #[test]
    fn dropping_the_pool_joins_every_worker() {
        let pool = test_pool(6);
        let watch = Arc::clone(&pool.shared);
        assert_eq!(Arc::strong_count(&watch), 7); // pool + 5 workers + watch
        drop(pool);
        // Every worker dropped its `Arc`, so every worker thread returned.
        assert_eq!(Arc::strong_count(&watch), 1);
    }

    #[test]
    fn pools_can_be_built_and_torn_down_repeatedly() {
        for shards in 1..=8 {
            let mut pool = test_pool(shards);
            let sum = AtomicUsize::new(0);
            pool.run(64, |shard| {
                sum.fetch_add(shard.rows.len(), Ordering::Relaxed);
            });
            assert_eq!(sum.load(Ordering::Relaxed), 64);
        }
    }

    #[test]
    fn a_pool_without_an_inline_caller_uses_only_workers() {
        let mut pool = ComputePool::with_topology(
            Topology::unpinned(CpuMask::up_to(4)),
            PoolConfig {
                shards: Some(4),
                pin: false,
                pin_caller: false,
                inline_caller: false,
            },
        );
        assert_eq!(pool.shards(), 4);
        let caller = std::thread::current().id();
        let elsewhere = AtomicUsize::new(0);
        pool.run(400, |shard| {
            if std::thread::current().id() != caller {
                elsewhere.fetch_add(shard.rows.len(), Ordering::Relaxed);
            }
        });
        assert_eq!(elsewhere.load(Ordering::Relaxed), 400);
    }

    #[test]
    fn debug_output_names_the_pool_shape() {
        let pool = test_pool(3);
        let text = format!("{pool:?}");
        assert!(text.contains("shards: 3"), "{text}");
        assert!(text.contains("workers: 2"), "{text}");
    }

    /// Handoff-latency probe. Not part of the gate; run with
    /// `cargo test -p ramvamp-core -- --ignored wake_latency --nocapture`.
    #[test]
    #[ignore = "timing probe, not a correctness gate"]
    fn wake_latency_probe() {
        use std::time::Instant;

        let mut pool = ComputePool::new();
        let shards = pool.shards();
        let rows = shards * 8;
        let mut samples = Vec::with_capacity(20_000);
        // Warm up so the workers are spinning, as they would mid-token.
        for _ in 0..2_000 {
            pool.run(rows, |_| {});
        }
        for _ in 0..20_000 {
            let start = Instant::now();
            pool.run(rows, |_| {});
            samples.push(start.elapsed().as_nanos() as u64);
        }
        samples.sort_unstable();
        let pct = |p: usize| samples[samples.len() * p / 100];
        eprintln!(
            "shards={shards} barrier ns: p50={} p90={} p99={} max={}",
            pct(50),
            pct(90),
            pct(99),
            samples[samples.len() - 1]
        );
    }
}
