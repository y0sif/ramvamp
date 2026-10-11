//! The compute pool on a machine that is doing other things.
//!
//! Every GEMV the runtime runs goes through [`ComputePool`], and the question
//! here is what one busy core costs it. Five pool arms run the same real
//! kernels at the v0 expert shapes, six threads each:
//!
//! - `per-core/fixed`: v0's pool. One thread pinned per P-core, each owning a
//!   fixed sixth of every job ([`ComputePool::scatter`]).
//! - `per-core/chunks`: the same threads claiming small chunks from a shared
//!   counter ([`ComputePool::scatter_chunks`]).
//! - `shared`: chunks, with every thread free on the P-cores and the E-cores
//!   that share the L3 ([`Placement::Shared`]).
//! - `only P`: chunks, free on the P-cores alone. Not a runtime option; it
//!   shows whether the E-cores in the shared set help or hurt.
//! - `off`: chunks, with no affinity at all ([`Placement::Off`]).
//!
//! under four loads, each a thread that spins flat out:
//!
//! - `none`: nothing added. Whatever else the machine is doing still runs,
//!   and falls on every arm alike, because the arms are interleaved.
//! - `busy pool core`: one spinner pinned to a CPU the pinned pool computes on
//!   (not the submitting thread's).
//! - `busy sibling`: one spinner on that CPU's hyperthread sibling.
//! - `2 floating`: two unpinned spinners, placed by the scheduler.
//!
//! Each round runs every load, and under each load every arm, in an order
//! that rotates per round. A slowdown is taken per round against the same arm
//! with no load in the same round, then the median over rounds is reported, so
//! drift in the machine's background cancels instead of landing on one arm.
//!
//! Warm, in-process, and not in a cgroup: every figure here is a diagnostic
//! under rule 2 of `docs/experiments.md`, never a published number.
//!
//! Run with `cargo bench -p ramvamp-core --bench pool`. `POOL_ROUNDS`
//! (default 12) and `POOL_BLOCK` (timed units per block, default 15) scale it.

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ramvamp_core::kernels::gemv::{gemv_q8_k_batched, gemv_q8_k_rows};
use ramvamp_core::kernels::quants::{BlockQ8K, QuantFormat, f32_to_f16, quantize_row_q8_k};
use ramvamp_core::threads::{
    ComputePool, CpuMask, Placement, PoolConfig, Topology, pin_current_thread_to,
};

/// v0 expert geometry (Qwen3-30B-A3B).
const HIDDEN: usize = 2048;
const MOE: usize = 768;
const TOP_K: usize = 8;
/// Prefill's batched GEMV width: the chunk rows routed to one expert.
const N_ACTS: usize = 32;
/// Distinct expert sets cycled through, so the weights a unit reads are not
/// still in L3 from the unit before. Four sets of a decode layer's eight
/// experts is about 98 MB.
const SETS: usize = 4;
const WARMUP_UNITS: usize = 3;
/// Time for a spinner to start or stop before a block is timed.
const SETTLE: Duration = Duration::from_millis(30);

/// Deterministic 64-bit LCG (Knuth MMIX constants), as in `kernels.rs`.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Random packed blocks with sane f16 scales, as in `kernels.rs`.
fn synth_matrix(format: QuantFormat, in_dim: usize, out_dim: usize, rng: &mut Lcg) -> Vec<u8> {
    let scale_offs: &[usize] = match format {
        QuantFormat::Q4_K | QuantFormat::Q5_K => &[0, 2],
        QuantFormat::Q6_K => &[208],
        _ => &[0],
    };
    let block_bytes = format.block_bytes();
    let blocks = in_dim / format.block_weights() * out_dim;
    let mut w = vec![0u8; blocks * block_bytes];
    for block in w.chunks_exact_mut(block_bytes) {
        for b in block.iter_mut() {
            *b = (rng.next_u64() >> 32) as u8;
        }
        for &off in scale_offs {
            let d = f32_to_f16(0.01 + 0.02 * (rng.next_f32() + 1.0));
            block[off..off + 2].copy_from_slice(&d.to_le_bytes());
        }
    }
    w
}

fn synth_acts(dim: usize, n: usize, rng: &mut Lcg) -> Vec<BlockQ8K> {
    let mut acts = vec![BlockQ8K::default(); dim / 256 * n];
    for row in acts.chunks_exact_mut(dim / 256) {
        let x: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
        quantize_row_q8_k(&x, row).expect("quantize");
    }
    acts
}

// ---------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dispatch {
    Fixed,
    Chunks,
}

/// Where an arm's threads may run.
enum Where {
    /// A placement the runtime offers.
    Runtime(Placement),
    /// Every thread, workers and submitter alike, free within this set. Not a
    /// runtime option; here to show which part of the shared set matters.
    Within(CpuMask),
}

struct Arm {
    name: String,
    pool: ComputePool,
    place: Where,
    dispatch: Dispatch,
    /// The process's own mask, restored before each placement so that one
    /// arm's pin never leaks into the next.
    allowed: CpuMask,
}

impl Arm {
    fn new(name: &str, place: Where, dispatch: Dispatch, allowed: &CpuMask) -> Self {
        // A worker inherits the affinity of the thread that spawns it.
        let mask = match &place {
            Where::Runtime(_) => allowed,
            Where::Within(mask) => mask,
        };
        let _ = pin_current_thread_to(mask);
        let placement = match &place {
            Where::Runtime(placement) => *placement,
            Where::Within(_) => Placement::Off,
        };
        let pool = ComputePool::with_config(PoolConfig {
            placement,
            pin_caller: false,
            ..PoolConfig::default()
        });
        let _ = pin_current_thread_to(allowed);
        Self {
            name: name.to_owned(),
            pool,
            place,
            dispatch,
            allowed: allowed.clone(),
        }
    }

    /// Place the calling thread for this arm and hand back its pool.
    fn enter(&mut self) -> &mut ComputePool {
        let _ = pin_current_thread_to(&self.allowed);
        match &self.place {
            Where::Runtime(_) => {
                self.pool.pin_calling_thread();
            }
            Where::Within(mask) => {
                let _ = pin_current_thread_to(mask);
            }
        }
        &mut self.pool
    }
}

/// `out[rows] = W[rows] . x` for one activation row, through the pool.
fn gemv(
    pool: &mut ComputePool,
    dispatch: Dispatch,
    format: QuantFormat,
    weight: &[u8],
    in_dim: usize,
    acts: &[BlockQ8K],
    out: &mut [f32],
) {
    let out_dim = out.len();
    match dispatch {
        Dispatch::Fixed => pool.scatter(out, |shard, piece| {
            gemv_q8_k_rows(format, weight, in_dim, out_dim, acts, shard.rows, piece).expect("gemv");
        }),
        Dispatch::Chunks => pool.scatter_chunks(out, |chunk, piece| {
            gemv_q8_k_rows(format, weight, in_dim, out_dim, acts, chunk.rows, piece).expect("gemv");
        }),
    }
}

/// Prefill's batched GEMV: `N_ACTS` outputs per weight row, row-major.
fn gemv_batched(
    pool: &mut ComputePool,
    dispatch: Dispatch,
    weight: &[u8],
    acts: &[BlockQ8K],
    out: &mut [[f32; N_ACTS]],
) {
    let out_dim = out.len();
    let run = |rows: std::ops::Range<usize>, piece: &mut [[f32; N_ACTS]]| {
        gemv_q8_k_batched(
            QuantFormat::Q4_K,
            weight,
            HIDDEN,
            out_dim,
            acts,
            N_ACTS,
            rows,
            piece.as_flattened_mut(),
        )
        .expect("batched gemv");
    };
    match dispatch {
        Dispatch::Fixed => pool.scatter(out, |shard, piece| run(shard.rows, piece)),
        Dispatch::Chunks => pool.scatter_chunks(out, |chunk, piece| run(chunk.rows, piece)),
    }
}

// ---------------------------------------------------------------------------
// Workloads
// ---------------------------------------------------------------------------

struct Expert {
    /// Gate and up as one 1536-row Q4_K matrix, as the fused decode path
    /// runs them.
    gate_up: Vec<u8>,
    /// Q6_K, the worse of the two `down` formats in the v0 pin.
    down: Vec<u8>,
}

struct Workloads {
    sets: Vec<Vec<Expert>>,
    x_hidden: Vec<BlockQ8K>,
    x_moe: Vec<BlockQ8K>,
    x_batch: Vec<BlockQ8K>,
    gate_up_out: Vec<f32>,
    down_out: Vec<f32>,
    batch_out: Vec<[f32; N_ACTS]>,
    next: usize,
}

impl Workloads {
    fn new() -> Self {
        let mut rng = Lcg(0x5eed_cafe_f00d_d00d);
        let sets = (0..SETS)
            .map(|_| {
                (0..TOP_K)
                    .map(|_| Expert {
                        gate_up: synth_matrix(QuantFormat::Q4_K, HIDDEN, 2 * MOE, &mut rng),
                        down: synth_matrix(QuantFormat::Q6_K, MOE, HIDDEN, &mut rng),
                    })
                    .collect()
            })
            .collect();
        Self {
            sets,
            x_hidden: synth_acts(HIDDEN, 1, &mut rng),
            x_moe: synth_acts(MOE, 1, &mut rng),
            x_batch: synth_acts(HIDDEN, N_ACTS, &mut rng),
            gate_up_out: vec![0.0; 2 * MOE],
            down_out: vec![0.0; HIDDEN],
            batch_out: vec![[0.0; N_ACTS]; MOE],
            next: 0,
        }
    }

    /// One decode layer's expert phase: eight experts, each a gate/up job
    /// and a down job. Sixteen pool jobs.
    fn decode_layer(&mut self, pool: &mut ComputePool, dispatch: Dispatch) {
        let set = &self.sets[self.next % SETS];
        self.next += 1;
        for e in set {
            gemv(
                pool,
                dispatch,
                QuantFormat::Q4_K,
                &e.gate_up,
                HIDDEN,
                &self.x_hidden,
                &mut self.gate_up_out,
            );
            gemv(
                pool,
                dispatch,
                QuantFormat::Q6_K,
                &e.down,
                MOE,
                &self.x_moe,
                &mut self.down_out,
            );
        }
        black_box(&self.down_out);
    }

    /// Prefill's expert GEMV: eight experts' gate halves, each against the
    /// 32 chunk rows routed to it. Eight pool jobs.
    fn prefill_experts(&mut self, pool: &mut ComputePool, dispatch: Dispatch) {
        let set = &self.sets[self.next % SETS];
        self.next += 1;
        for e in set {
            // The first 768 rows of the fused matrix are a whole Q4_K gate.
            let gate = &e.gate_up[..e.gate_up.len() / 2];
            gemv_batched(pool, dispatch, gate, &self.x_batch, &mut self.batch_out);
        }
        black_box(&self.batch_out);
    }
}

#[derive(Clone, Copy)]
enum Workload {
    Decode,
    Prefill,
}

impl Workload {
    fn title(self) -> &'static str {
        match self {
            Self::Decode => {
                "decode: one layer's expert phase, 8 x (Q4_K 1536x2048 gate/up + Q6_K 2048x768 down), 1 activation row, 16 pool jobs"
            }
            Self::Prefill => {
                "prefill: 8 x Q4_K 768x2048 gate, 32 activation rows each, 8 pool jobs"
            }
        }
    }

    fn run(self, w: &mut Workloads, pool: &mut ComputePool, dispatch: Dispatch) {
        match self {
            Self::Decode => w.decode_layer(pool, dispatch),
            Self::Prefill => w.prefill_experts(pool, dispatch),
        }
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// A thread that spins flat out while its flag is set and naps otherwise.
struct Spinner {
    on: Arc<AtomicBool>,
    quit: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Spinner {
    /// A spinner on `cpu`, or anywhere in `allowed` if `None`.
    ///
    /// The mask is set explicitly either way: a thread inherits its parent's
    /// affinity, and the parent here is the submitting thread, which a pinned
    /// pool has pinned to one CPU.
    fn new(cpu: Option<usize>, allowed: &CpuMask) -> Self {
        let on = Arc::new(AtomicBool::new(false));
        let quit = Arc::new(AtomicBool::new(false));
        let (on2, quit2) = (Arc::clone(&on), Arc::clone(&quit));
        let mask = match cpu {
            Some(cpu) => std::iter::once(cpu).collect(),
            None => allowed.clone(),
        };
        let handle = std::thread::spawn(move || {
            pin_current_thread_to(&mask).expect("place spinner");
            let mut x = 1u64;
            while !quit2.load(Ordering::Relaxed) {
                if on2.load(Ordering::Relaxed) {
                    for _ in 0..10_000 {
                        x = black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
                    }
                } else {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            black_box(x);
        });
        Self {
            on,
            quit,
            handle: Some(handle),
        }
    }

    fn set(&self, on: bool) {
        self.on.store(on, Ordering::Relaxed);
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.quit.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

struct Load {
    name: String,
    spinners: Vec<Spinner>,
}

/// The sibling hyperthread of `cpu`, from sysfs, if it has one.
fn sibling_of(cpu: usize) -> Option<usize> {
    let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
    let text = std::fs::read_to_string(path).ok()?;
    let mut ids = Vec::new();
    for part in text.trim().split(',') {
        match part.split_once('-') {
            Some((a, b)) => ids.extend(a.parse::<usize>().ok()?..=b.parse::<usize>().ok()?),
            None => ids.push(part.parse().ok()?),
        }
    }
    ids.into_iter().find(|&c| c != cpu)
}

fn read_trim(path: &str) -> String {
    std::fs::read_to_string(path)
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|_| "?".to_owned())
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return f64::NAN;
    }
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

/// Per-unit times in ms, indexed `[load][arm][round][unit]`.
type Samples = Vec<Vec<Vec<Vec<f64>>>>;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn main() {
    let rounds = env_usize("POOL_ROUNDS", 12);
    let block = env_usize("POOL_BLOCK", 15);

    let allowed = CpuMask::current().unwrap_or_else(|| CpuMask::up_to(1));
    let topo = Topology::detect();
    let compute = topo.compute_cpus().to_vec();
    let pcores: CpuMask = topo.performance_cpus().iter().copied().collect();
    let shared = topo.shared_cpus();
    let mut arms = [
        Arm::new(
            "per-core/fixed",
            Where::Runtime(Placement::PerCore),
            Dispatch::Fixed,
            &allowed,
        ),
        Arm::new(
            "per-core/chunks",
            Where::Runtime(Placement::PerCore),
            Dispatch::Chunks,
            &allowed,
        ),
        Arm::new(
            &format!("shared {shared}"),
            Where::Runtime(Placement::Shared),
            Dispatch::Chunks,
            &allowed,
        ),
        Arm::new(
            &format!("only P {pcores}"),
            Where::Within(pcores),
            Dispatch::Chunks,
            &allowed,
        ),
        Arm::new(
            &format!("off {allowed}"),
            Where::Runtime(Placement::Off),
            Dispatch::Chunks,
            &allowed,
        ),
    ];
    let _ = pin_current_thread_to(&allowed);
    println!("topology      : {topo}");
    println!(
        "pools         : {} threads each; per-core/* pinned on {:?}; the last three run chunks, free within the set named; {} chunks per job of 768 rows",
        arms[0].pool.shards(),
        compute,
        arms[1].pool.chunks(MOE)
    );
    println!(
        "intel_pstate  : no_turbo {}, max_perf_pct {}; cpu0 governor {}, max {} kHz",
        read_trim("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        read_trim("/sys/devices/system/cpu/intel_pstate/max_perf_pct"),
        read_trim("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        read_trim("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
    );
    println!("load average  : {} (start)", read_trim("/proc/loadavg"));
    if !topo.pins() {
        println!("!! this machine's topology did not pin, so the pinned arms are not pinned");
    }

    // A worker's CPU, not the submitting thread's.
    let victim = compute.get(2).or(compute.last()).copied();
    let mut loads = vec![Load {
        name: "none".to_owned(),
        spinners: Vec::new(),
    }];
    if let Some(cpu) = victim {
        loads.push(Load {
            name: format!("busy pool core (cpu{cpu})"),
            spinners: vec![Spinner::new(Some(cpu), &allowed)],
        });
        if let Some(sib) = sibling_of(cpu) {
            loads.push(Load {
                name: format!("busy sibling (cpu{sib})"),
                spinners: vec![Spinner::new(Some(sib), &allowed)],
            });
        }
    }
    loads.push(Load {
        name: "2 floating".to_owned(),
        spinners: vec![Spinner::new(None, &allowed), Spinner::new(None, &allowed)],
    });

    let mut work = Workloads::new();

    for workload in [Workload::Decode, Workload::Prefill] {
        let mut samples: Samples = vec![vec![Vec::new(); arms.len()]; loads.len()];
        for round in 0..rounds {
            for (li, load) in loads.iter().enumerate() {
                for s in &load.spinners {
                    s.set(true);
                }
                std::thread::sleep(SETTLE);
                for k in 0..arms.len() {
                    let ai = (k + round) % arms.len();
                    let arm = &mut arms[ai];
                    let dispatch = arm.dispatch;
                    let pool = arm.enter();
                    for _ in 0..WARMUP_UNITS {
                        workload.run(&mut work, pool, dispatch);
                    }
                    let mut times = Vec::with_capacity(block);
                    for _ in 0..block {
                        let t = Instant::now();
                        workload.run(&mut work, pool, dispatch);
                        times.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                    samples[li][ai].push(times);
                }
                for s in &load.spinners {
                    s.set(false);
                }
            }
            eprint!(
                "\r{} round {}/{rounds}   ",
                workload.title().split(':').next().unwrap_or(""),
                round + 1
            );
        }
        eprintln!();
        let names: Vec<&str> = arms.iter().map(|a| a.name.as_str()).collect();
        report(workload, &names, &loads, &samples);
    }
    println!("load average  : {} (end)", read_trim("/proc/loadavg"));
}

fn report(workload: Workload, arms: &[&str], loads: &[Load], samples: &Samples) {
    println!();
    println!("== {} ==", workload.title());
    println!(
        "   per unit: median of round medians in ms, then the median over rounds of (this load / no load) for the same arm"
    );
    print!("{:<28}", "load");
    for arm in arms {
        print!("{arm:>18}");
    }
    println!();
    for (li, (load, per_arm)) in loads.iter().zip(samples).enumerate() {
        print!("{:<28}", load.name);
        for (rounds, base) in per_arm.iter().zip(&samples[0]) {
            let mut meds: Vec<f64> = rounds.iter().map(|r| median(&mut r.clone())).collect();
            let typical = median(&mut meds);
            if li == 0 {
                print!("{:>18}", format!("{typical:.3}"));
            } else {
                let mut ratios: Vec<f64> = rounds
                    .iter()
                    .zip(base)
                    .map(|(r, base)| median(&mut r.clone()) / median(&mut base.clone()))
                    .collect();
                let slow = median(&mut ratios);
                print!("{:>18}", format!("{typical:.3} ({slow:.2}x)"));
            }
        }
        println!();
    }
    println!("   p90 of every unit, ms:");
    for (load, per_arm) in loads.iter().zip(samples) {
        print!("{:<28}", load.name);
        for rounds in per_arm {
            let mut all: Vec<f64> = rounds.iter().flatten().copied().collect();
            print!("{:>18}", format!("{:.3}", quantile(&mut all, 0.9)));
        }
        println!();
    }
}
