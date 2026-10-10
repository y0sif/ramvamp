//! Spike: can the Intel Arc iGPU (Xe-LPG, Meteor Lake) beat the pinned CPU
//! P-core pool on a Q4_K x Q8_K batched GEMV, the shape that dominates
//! ramvamp's prefill?
//!
//! Prefill reads each routed expert's weights once per 512-token chunk and
//! applies them to the ~32 chunk rows routed to that expert. That batched
//! GEMV is ~64% of prefill's arithmetic, and prefill is compute-bound with
//! I/O headroom, so this one kernel is the whole target.
//!
//! Both arms run the same real bytes: one expert's packed Q4_K gate slab out
//! of an installed `.rvmp`, and one deterministic batch of Q8_K activation
//! rows. Arm A is `gemv_q8_k_batched` under the production `ComputePool`.
//! Arm B is a Vulkan compute dispatch that reproduces the reference dot's
//! structure (packed weights, integer accumulation per sub-block, sub-block
//! scale applied in integers, mins folded in through `bsums`), not a naive
//! dequantize-then-multiply.
//!
//! **The shader is Q4_K-only, and not just arithmetically.** It reads the
//! weight slab as `uint w[]` and indexes it in 4-byte units, so the packed
//! block size has to be a whole number of uints. Q4_K's 144 B is; **Q6_K's
//! 210 B is not**, and `ffn_down_exps` is Q6_K on 24 of the v0 model's 48
//! layers (`docs/architecture.md`). So this read pattern covers the expert
//! `gate` and `up` projections and cannot be reused for `down` on half the
//! layers without a byte-granular path. That is a limit on what a positive
//! result here would license, not a detail of the benchmark:
//! [`gpu::check_shader_format`] refuses anything else rather than letting
//! the assumption stay implicit.
//!
//! What this binary will and will not claim. It gates itself on four
//! conditions and exits non-zero if any fails: the two arms must agree
//! numerically inside named tolerances *including* a signed-bias check,
//! every arm must hold one steady state across a fixed sample (its per-round
//! medians must agree across the run's windows and must not spread wide),
//! the machine's clock policy must not be capping the CPU arm, and the
//! weights must be a real expert slab rather than the synthetic fallback. A
//! ratio measured against a throttled CPU is a fact about the power daemon,
//! a ratio between arms that changed state mid-run is a fact about which
//! state each arm happened to be in, and a ratio measured on invented bytes
//! is a fact about nothing, so none of them
//! is printed as a result. Every one of those reasons lands *in the verdict
//! sentence*, which also carries the weights' provenance inline, so the one
//! line most likely to be quoted out of context cannot be quoted without it.
//!
//! The gates are decided the moment the sampling loop ends, before a single
//! timing is printed, because the unit of quotation is the *line* and not the
//! run. When any gate fails, no ratio is computed anywhere -- not per round,
//! not per arm, not in the verdict line -- and every millisecond and GFLOP/s
//! figure that is still printed is stamped [`INVALID_TAG`]. A screenshot of
//! any one line of an invalid run must not read as a result.

mod clock;
mod cpu;
mod data;
mod gpu;
mod vk;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ramvamp_core::threads::{ComputePool, PoolConfig};

/// Hidden size of the v0 model: one expert weight row is this many weights.
const IN_DIM: usize = 2048;
/// MoE intermediate size: rows in one expert's gate slab.
const OUT_DIM: usize = 768;
/// Chunk rows routed to one expert in a 512-token prefill chunk.
const N_ACTS: usize = 32;
/// Layer and expert the slab is read from. Any would do; these are pinned so
/// the numbers are reproducible.
const LAYER: usize = 0;
/// See [`LAYER`].
const EXPERT: usize = 0;

/// Untimed iterations run once per arm before any sample is recorded.
///
/// The CPU arm needs this much and not a token less. `intel_pstate` samples
/// on the order of 10 ms, so a block of a dozen iterations finishes inside a
/// handful of sample periods and records the ramp rather than the plateau: an
/// earlier revision of this file used 5 warm-up and 15 timed iterations and
/// its headline ratio moved 47% across three consecutive runs, entirely on
/// the CPU side. 200 iterations at roughly 0.5 ms is ~100 ms of continuous
/// load, which is enough for the governor to settle.
const WARMUP: usize = 200;
/// Untimed iterations at the top of every block, after the *other* arm has
/// had the machine. Cheaper than [`WARMUP`] because the unit was in steady
/// state a few milliseconds ago; enough to walk back up after the handover.
const REWARM: usize = 20;
/// Timed iterations per arm per round.
const BLOCK: usize = 50;
/// Rounds per run, and every one of them runs. The sample size is fixed
/// before anything is measured: stopping once the numbers look settled is
/// optional stopping, and the minimum-based rule that did it never stopped at
/// all with turbo on. A round is ~70 ms, so this is ~4 s of timed work.
const ROUNDS: usize = 60;
/// The run is split into this many consecutive windows, and each arm's window
/// medians must agree. A drift, or a slow episode confined to part of the
/// run, shows up as windows that disagree.
const WINDOWS: usize = 3;
/// How far apart an arm's window medians may be, as `max / min - 1`. Set
/// well under the effects this spike decides on (15-30%), so instability
/// inside it cannot flip a verdict.
const TOL_WINDOW_SPREAD: f64 = 0.05;
/// How wide an arm's per-round medians may spread, as the interquartile range
/// over the median. An arm flipping between two states throughout the run has
/// windows that agree and a wide IQR, and no single typical value to compare.
const TOL_IQR: f64 = 0.10;

/// Numerics gate: outputs below this fraction of the reference's largest
/// magnitude are cancellation artefacts and carry a meaningless relative
/// error, so the relative-error gates are scored over the rest.
const SIGNIFICANT: f64 = 1e-3;
/// Numerics gate: max relative error over the significant outputs.
const TOL_MAX_REL_BIG: f64 = 1e-3;
/// Numerics gate: mean unsigned relative error over every output.
const TOL_MEAN_REL: f64 = 1e-4;
/// Numerics gate: max absolute error, as a fraction of the reference's
/// largest magnitude.
const TOL_MAX_ABS_REL_SCALE: f64 = 1e-4;
/// Numerics gate: mean SIGNED relative error over the significant outputs.
///
/// This is the gate the unsigned metrics cannot replace. Symmetric rounding
/// noise cancels in a signed mean and a systematic scale error does not, so a
/// shader that multiplied every output by 1.0001 would sit comfortably inside
/// every threshold above -- its max relative error over the significant
/// outputs would read about 1.0e-4 against the 9.5e-5 an honest kernel
/// already produces -- and would miss this one by an order of magnitude.
const TOL_SIGNED_BIAS: f64 = 1e-5;

/// Stamped in place of the leading indent on every timing line of a run whose
/// verdict is INVALID.
///
/// The unit of quotation is the line, not the run. A per-round ratio, a
/// GFLOP/s figure or an arm's sampled minimum is a screenshot on its own, and
/// a reader who sees one has not seen the verdict twenty lines below it. So
/// on an invalid run the ratios are not printed at all and everything that is
/// still printed carries this.
const INVALID_TAG: &str = "[INVALID] ";

fn main() -> Result<()> {
    // `run` owns every Vulkan handle, so returning through it runs the
    // destructors before the process exits.
    if run()? {
        Ok(())
    } else {
        std::process::exit(1)
    }
}

/// The whole spike. Returns whether every gate passed.
#[allow(clippy::too_many_lines)]
fn run() -> Result<bool> {
    let install = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("models/qwen3.rvmp"), PathBuf::from);

    // ---- inputs -------------------------------------------------------
    let slab = match data::load_gate_slab(&install, LAYER, EXPERT, IN_DIM) {
        Ok(slab) => slab,
        Err(err) => {
            println!(
                "!! could not read a real expert slab from {}: {err:#}",
                install.display()
            );
            println!(
                "!! FALLING BACK TO SYNTHETIC WEIGHT BYTES -- the timing is still\n\
                 !! honest (same instruction stream, same byte count) but the numbers\n\
                 !! were not measured on real quantized data, so the verdict below is\n\
                 !! printed INVALID and this binary exits non-zero.\n\
                 !! NOTE: the default path is RELATIVE, so running this binary from any\n\
                 !! directory but the repo root lands here. Pass the install path as\n\
                 !! argv[1] to measure on a real slab.\n"
            );
            data::synthetic_gate_slab(IN_DIM, OUT_DIM)
        }
    };
    let out_dim = slab.out_dim;
    let in_dim = slab.in_dim;
    // A zero-row slab would divide by zero three lines below and would then
    // ask Vulkan for a zero-size buffer, which VUID-VkBufferCreateInfo-size-
    // 00912 forbids. `load_gate_slab` already refuses a zero-length slab;
    // this is the check that makes the arithmetic here provably safe for any
    // slab, including the synthetic one.
    if out_dim == 0 || slab.bytes.is_empty() {
        bail!(
            "the gate slab is empty ({} B, {out_dim} rows): there is nothing to measure",
            slab.bytes.len()
        );
    }
    let n_blocks = in_dim / data::QK_K;
    let acts = data::build_acts(N_ACTS, IN_DIM, 0xA11C_E5EE_D000_0001)?;

    println!("== problem ==");
    println!("  weights   : {}", slab.provenance);
    println!(
        "  format    : {:?}, {} B per 256-weight block. The shader indexes its weight\n              \
         buffer as `uint w[]`, so it can only address a format whose block is a whole\n              \
         number of uints. Q6_K (210 B, the expert `down` projection on 24 of 48\n              \
         layers) is not one and is refused, not silently misread.",
        gpu::SHADER_FORMAT,
        gpu::SHADER_FORMAT.block_bytes()
    );
    println!(
        "  shape     : {out_dim} rows x {in_dim} in-dim, {} B ({} B/row)",
        slab.bytes.len(),
        slab.bytes.len() / out_dim
    );
    println!("  acts      : {N_ACTS} rows x {n_blocks} q8_k blocks (292 B/block)");
    let flops = 2.0 * (N_ACTS * out_dim * in_dim) as f64;
    println!(
        "  arithmetic: {:.2} MFLOP per pass (1 MAC = 2 FLOP)\n",
        flops / 1e6
    );

    // ---- device -------------------------------------------------------
    let ctx = vk::VkContext::new(true)?;
    ctx.caps.report();

    // ---- arm A: the production CPU path -------------------------------
    let mut pool = ComputePool::with_config(PoolConfig::default());
    println!("== CPU pool ==");
    println!(
        "  shards {} over compute cpus {:?} (P-cores {:?})",
        pool.shards(),
        pool.topology().compute_cpus(),
        pool.topology().performance_cpus()
    );
    let cpus: Vec<usize> = pool.topology().compute_cpus().to_vec();
    let lock = clock::requested().map_err(anyhow::Error::msg)?;
    let throttles = report_cpu_clock_caps(&cpus, lock.is_some());
    let gt = clock::render_gt();
    // Every way the clock lock failed: not in place before sampling, or did
    // not hold during it. Only ever non-empty on a locked run.
    let mut lock_problems = Vec::new();
    if let Some(lock) = &lock {
        println!();
        println!("== clock lock ==");
        println!(
            "  requested: CPU {:.2} GHz on cpus {cpus:?}, iGPU {} MHz ({})",
            lock.cpu_khz as f64 / 1e6,
            lock.gt_mhz,
            gt.as_ref().map_or_else(
                || "no single i915 render GT found".to_owned(),
                clock::Gt::describe
            )
        );
        lock_problems = clock::check_in_place(lock, &clock::read_settings(&cpus, gt.as_ref()));
        if lock_problems.is_empty() {
            println!(
                "  in place: PASS (min = max on every pinned cpu; min = max = boost on the iGPU)"
            );
        }
        for problem in &lock_problems {
            println!("  !! {problem}");
        }
    }
    let mut clocks = clock::ClockLog::default();
    println!();

    let mut cpu_out = vec![0f32; out_dim * N_ACTS];

    // `RAMVAMP_SPIKE_CPU=hoisted` swaps the CPU arm to the row-hoist
    // prototype. Anything else, or unset, is the production kernel.
    let kernel = match std::env::var("RAMVAMP_SPIKE_CPU").as_deref() {
        Ok("hoisted") => cpu::CpuKernel::Hoisted,
        _ => cpu::CpuKernel::Production,
    };
    let cpu_label = match kernel {
        cpu::CpuKernel::Production => "CPU  (pool, avx2)",
        cpu::CpuKernel::Hoisted => "CPU  (pool, avx2, hoisted)",
    };
    println!("== CPU kernel ==\n  {kernel:?} (set RAMVAMP_SPIKE_CPU=hoisted for the prototype)");
    let hoist_identical = if kernel == cpu::CpuKernel::Hoisted {
        let mut prod = vec![0f32; out_dim * N_ACTS];
        cpu::gemv_batched_pooled(
            cpu::CpuKernel::Production,
            &mut pool,
            &slab.bytes,
            in_dim,
            out_dim,
            &acts,
            N_ACTS,
            &mut prod,
        )?;
        cpu::gemv_batched_pooled(
            kernel,
            &mut pool,
            &slab.bytes,
            in_dim,
            out_dim,
            &acts,
            N_ACTS,
            &mut cpu_out,
        )?;
        let differ = prod
            .iter()
            .zip(&cpu_out)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "  hoisted vs production on this slab: {} of {} outputs differ in bits [{}]",
            differ,
            prod.len(),
            if differ == 0 { "PASS" } else { "FAIL" }
        );
        differ == 0
    } else {
        true
    };
    println!();

    // ---- arm B: the Vulkan dispatch -----------------------------------
    let gemv = gpu::GpuGemv::new(&ctx, gpu::SHADER_FORMAT, in_dim, out_dim, N_ACTS)?;
    gemv.upload_weights(&slab.bytes)
        .context("uploading the weight slab")?;
    gemv.upload_acts(&acts, N_ACTS, n_blocks)
        .context("uploading the activation batch")?;

    // Each arm runs as a BLOCK of back-to-back iterations, and the blocks
    // alternate over several rounds.
    //
    // Interleaving one iteration at a time was tried first and is worse for
    // both arms: the pool's workers park during the ~0.3 ms GPU submit and
    // pay a wake on every CPU iteration, and the iGPU never holds its clocks
    // through the CPU block. Neither reaches a steady state, which is the
    // state prefill actually runs in -- a chunk drives many consecutive
    // expert GEMVs on whichever unit owns them. Alternating blocks keeps
    // each arm in steady state while still sampling both across the whole
    // wall-clock window, so a thermal or contention drift shows up as
    // disagreement between rounds rather than as a silent bias toward
    // whichever arm ran first.
    //
    // Rounds are a fixed count, [`ROUNDS`], and every one of them runs.
    // Whether the sample describes one steady state is judged afterwards,
    // per arm, by [`Steadiness`].
    let mut cpu_iter = || {
        let t = Instant::now();
        cpu::gemv_batched_pooled(
            kernel,
            &mut pool,
            &slab.bytes,
            in_dim,
            out_dim,
            &acts,
            N_ACTS,
            &mut cpu_out,
        )?;
        Ok::<f64, anyhow::Error>(t.elapsed().as_secs_f64())
    };

    print!("warming up ({WARMUP} untimed iterations per arm) ... ");
    for _ in 0..WARMUP {
        cpu_iter()?;
    }
    for _ in 0..WARMUP {
        gemv.run()?;
    }
    for _ in 0..WARMUP {
        gemv.run_streamed(&slab.bytes)?;
    }
    println!("done");

    let mut cpu_times = Vec::new();
    let mut gpu_resident = Vec::new();
    let mut gpu_streamed = Vec::new();
    let mut gpu_device = Vec::new();
    let mut rounds = Vec::new();
    let mut cpu_steady = Steadiness::new(cpu_label);
    let mut res_steady = Steadiness::new("GPU  (weights resident)");
    let mut str_steady = Steadiness::new("GPU  (weights streamed)");
    for _ in 0..ROUNDS {
        let mut cpu_round = Vec::with_capacity(BLOCK);
        for _ in 0..REWARM {
            cpu_iter()?;
        }
        for i in 0..BLOCK {
            cpu_round.push(cpu_iter()?);
            // Between iterations, never inside one: the read is outside every
            // timed region.
            if i == BLOCK / 2 {
                clocks.sample_cpus(&cpus);
            }
        }
        clocks.sample_cpus(&cpus);

        let mut res_round = Vec::with_capacity(BLOCK);
        for _ in 0..REWARM {
            gemv.run()?;
        }
        for i in 0..BLOCK {
            let (host, device) = gemv.run()?;
            res_round.push(host);
            gpu_device.push(device);
            if i == BLOCK / 2 {
                clocks.sample_gt(gt.as_ref());
            }
        }
        clocks.sample_gt(gt.as_ref());

        let mut str_round = Vec::with_capacity(BLOCK);
        for _ in 0..REWARM {
            gemv.run_streamed(&slab.bytes)?;
        }
        for i in 0..BLOCK {
            let (host, _) = gemv.run_streamed(&slab.bytes)?;
            str_round.push(host);
            if i == BLOCK / 2 {
                clocks.sample_gt(gt.as_ref());
            }
        }
        clocks.sample_gt(gt.as_ref());

        cpu_steady.observe(&cpu_round);
        res_steady.observe(&res_round);
        str_steady.observe(&str_round);
        rounds.push((
            Stats::of(&mut cpu_round.clone()),
            Stats::of(&mut res_round.clone()),
            Stats::of(&mut str_round.clone()),
        ));
        cpu_times.extend_from_slice(&cpu_round);
        gpu_resident.extend_from_slice(&res_round);
        gpu_streamed.extend_from_slice(&str_round);
    }
    let mut gpu_out = vec![0f32; out_dim * N_ACTS];
    gemv.read_out(&mut gpu_out)?;

    // ---- numerical agreement ------------------------------------------
    let numerics_ok = report_agreement(&cpu_out, &gpu_out) && hoist_identical;

    // ---- performance ---------------------------------------------------
    let cpu = Stats::of(&mut cpu_times);
    let gpu_r = Stats::of(&mut gpu_resident);
    let gpu_s = Stats::of(&mut gpu_streamed);
    let gpu_d = Stats::of(&mut gpu_device);

    if let Some(lock) = &lock {
        lock_problems.extend(clocks.violations(lock));
    }

    let arms = [&cpu_steady, &res_steady, &str_steady];
    let unsteady: Vec<String> = arms
        .iter()
        .filter(|a| !a.steady())
        .map(|a| a.why_unsteady())
        .collect();

    // Every reason this run's timings are not a result, computed *here*
    // rather than after the figures it disqualifies have already been
    // printed. The invalidation set is knowable at this point -- numerics,
    // provenance and the clock policy were settled before the loop, and
    // steadiness the moment it ended -- so every line below can
    // carry it. It used to be computed at the very bottom, which is how
    // 10-20 clean unqualified ratio and GFLOP/s lines came to be printed
    // above a verdict that said none of them counted.
    let invalid = invalidations(
        numerics_ok,
        slab.synthetic,
        &slab.provenance,
        &throttles,
        &lock_problems,
        &unsteady,
    );
    // Two spaces when the run is valid, so the layout is unchanged; the
    // marker otherwise. Nothing below prints a timing without it.
    let tag: &str = if invalid.is_empty() {
        "  "
    } else {
        INVALID_TAG
    };

    println!("== clocks observed during sampling ==");
    for line in clocks.summary() {
        println!("  {line}");
    }
    if lock.is_none() {
        println!("  (unlocked run: informational, not gated)");
    }
    println!();

    println!(
        "== sampling ({} rounds x {BLOCK} timed runs per arm, {WARMUP} + {REWARM}/round warm-up, no early stop) ==",
        rounds.len()
    );
    for a in arms {
        println!("{tag}{}", a.report());
    }
    if !unsteady.is_empty() {
        println!(
            "  !! An arm is steady when its per-round medians agree across the run's\n  \
             !! {WINDOWS} windows to within {:.0}% and their interquartile range is within\n  \
             !! {:.0}% of their median. Windows that disagree mean the arm changed state\n  \
             !! part-way through (drift, or a slow episode); a wide IQR means it flipped\n  \
             !! between states throughout. Either way it has no single typical value,\n  \
             !! and a ratio against it depends on which state each arm happened to be in.",
            TOL_WINDOW_SPREAD * 100.0,
            TOL_IQR * 100.0
        );
    }
    println!();

    println!("== performance ==");
    if !invalid.is_empty() {
        println!(
            "  !! Every figure in this section is stamped {} and NO ratio is printed. A\n  \
             !! per-round ratio line is the single most quotable thing this binary emits,\n  \
             !! and one lifted out of an invalid run reads exactly like a measurement of\n  \
             !! the two units. This run is not one, because:",
            INVALID_TAG.trim()
        );
        for reason in &invalid {
            println!("  !!   - {reason}");
        }
    }
    for (i, (c, r, s)) in rounds.iter().enumerate() {
        let head = format!(
            "{tag}round {:2}: CPU median {:7.3} min {:6.3} ms   GPU resident {:7.3} min {:6.3} ms   GPU streamed {:7.3} min {:6.3} ms",
            i + 1,
            c.median * 1e3,
            c.min * 1e3,
            r.median * 1e3,
            r.min * 1e3,
            s.median * 1e3,
            s.min * 1e3,
        );
        if invalid.is_empty() {
            println!(
                "{head}   ({:.2}x / {:.2}x)",
                c.median / r.median,
                c.median / s.median
            );
        } else {
            println!("{head}");
        }
    }
    println!();
    cpu.report(cpu_label, flops, tag);
    gpu_r.report("GPU  (weights resident)", flops, tag);
    gpu_s.report("GPU  (weights streamed)", flops, tag);
    if gpu_d.median > 0.0 {
        gpu_d.report("GPU  (device timestamps)", flops, tag);
    }
    println!("  1-minute load average at exit: {}", loadavg());
    println!();

    report_gpu_timing_scope(
        &gemv,
        slab.bytes.len(),
        ctx.caps.host_cached_available,
        ctx.caps.subgroup_size,
    );
    if kernel == cpu::CpuKernel::Hoisted {
        println!(
            "== the two arms run the same algorithm, up to the expansion ==\n  \
             This run's CPU arm is the row-hoist prototype: it unpacks each weight\n  \
             row once and reuses it across all {N_ACTS} activations, as the shader\n  \
             does. Its expansion is deliberately scalar where the shader's is not,\n  \
             so the CPU figure is still a floor on what the hoist is worth.\n"
        );
    } else {
        report_arm_asymmetry();
    }

    // The verdict compares each arm's typical round, the median of its
    // per-round medians, because that is the statistic the steadiness gate
    // vouches for. Best-vs-best minima were the headline before, and the
    // 2026-10-09 re-runs showed why they cannot be: an arm's best block moved
    // by 10-20% between rounds while its median held within 3%. The device
    // timestamps are the dispatch alone, with no submit, fence or transfer, so
    // their ratio is a ceiling for a backend that batches many experts into
    // one submit, not a figure any single dispatch reaches. Everything is
    // computed inside the valid branch only: on an invalid run no ratio is
    // formatted at all.
    if invalid.is_empty() {
        let cpu_t = cpu_steady.typical();
        let res_t = res_steady.typical();
        let str_t = str_steady.typical();
        let resident = cpu_t / res_t;
        let streamed = cpu_t / str_t;
        let word = |r: f64| if r > 1.0 { "FASTER" } else { "SLOWER" };
        // The provenance is inside the sentence, not on a line above it: a
        // ratio measured on invented bytes and a ratio measured on a real
        // expert slab are different claims, and the quotable line has to say
        // which one it is.
        let mut verdict = format!(
            "VERDICT: on {}, at {}, against the {}-shard pinned CPU pool running the \
             {kernel:?} kernel at {:.3} ms per pass, the iGPU is {resident:.2}x {} with the weights \
             resident ({:.3} ms) and {streamed:.2}x {} with the weight slab streamed \
             ({:.3} ms), comparing each arm's median of {ROUNDS} round medians.",
            slab.provenance,
            lock.map_or_else(|| "unlocked clocks".to_owned(), |l| l.describe(cpus.len())),
            pool.shards(),
            cpu_t * 1e3,
            word(resident),
            res_t * 1e3,
            word(streamed),
            str_t * 1e3,
        );
        if gpu_d.median > 0.0 {
            verdict.push_str(&format!(
                " Dispatch alone, without submit, fence or transfer, is {:.3} ms ({:.2}x): \
                 the ceiling for a backend that batches many experts into one submit.",
                gpu_d.median * 1e3,
                cpu_t / gpu_d.median
            ));
        }
        println!("{verdict}");
    } else {
        // No ratio, not even one carrying its own invalidation. This is the
        // line a screenshot lands on, and a number in it survives being
        // separated from the words around it in a way the words do not.
        println!(
            "VERDICT: INVALID -- {}. On {}, no ratio is computed or printed anywhere in this \
             run's output.",
            invalid.join("; "),
            slab.provenance
        );
    }
    println!(
        "GATES: numerics {}, weights {}, machine {}, sampling {}",
        pass_fail(numerics_ok),
        pass_fail(!slab.synthetic),
        pass_fail(throttles.is_empty() && lock_problems.is_empty()),
        pass_fail(unsteady.is_empty())
    );
    // Every gate now has a reason on `invalid`, so this is the whole exit
    // condition rather than one gate plus a list.
    Ok(invalid.is_empty())
}

/// `PASS` or `FAIL`.
fn pass_fail(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

/// Every reason this run's timings are not a result, in the order they are
/// reported.
///
/// Pure, and separate from [`run`], for one reason: the two conditions that
/// matter most here are the two that cannot be reached without a Vulkan
/// device, so as long as this logic lived inline in `run` nothing held
/// "a numerics FAIL invalidates the verdict" or "synthetic weights invalidate
/// the verdict" -- the exact pair of defects the correction pass exists for.
/// An empty return is the *only* thing that lets a ratio be printed, so a
/// regression here is a regression to publishing a number that is not a
/// measurement of the two units.
///
/// Numerics comes first because it is the gate that says the two arms
/// computed the same thing at all. A ratio between an arm and a *different
/// computation* is not faster or slower than anything.
fn invalidations(
    numerics_ok: bool,
    synthetic: bool,
    provenance: &str,
    throttles: &[String],
    lock: &[String],
    unsteady: &[String],
) -> Vec<String> {
    let mut invalid = Vec::new();
    if !numerics_ok {
        invalid.push(
            "the shader does not reproduce the CPU kernel inside tolerance, so the two arms \
             did not compute the same thing"
                .to_owned(),
        );
    }
    if synthetic {
        invalid.push(format!(
            "weights are SYNTHETIC, not a real expert slab ({provenance})"
        ));
    }
    if !throttles.is_empty() {
        invalid.push(format!("CPU throttled ({})", throttles.join(", ")));
    }
    if !lock.is_empty() {
        invalid.push(format!(
            "clock lock not in place or did not hold ({})",
            lock.join("; ")
        ));
    }
    if !unsteady.is_empty() {
        invalid.push(format!("not a steady state: {}", unsteady.join("; ")));
    }
    invalid
}

/// One arm's per-round block medians, and whether they describe one steady
/// state.
///
/// This replaced a stopping rule on each arm's running minimum: sample until
/// the minimum stops improving by 0.5% for 3 rounds, and treat a round whose
/// minimum rose as drift that restarts the count. That rule converged at
/// locked, throttled clocks, and with turbo on it never did. On the
/// 2026-10-09 re-runs each arm counted as settled in 1 to 11 of 60 rounds,
/// never in the same round as the others, because one block's minimum moved
/// by more than 0.5% from round to round while the arm's median held within
/// 3%. On this box the minimum is too noisy to gate on.
///
/// So the sample size is fixed and the gate reads medians, for the two ways
/// an arm can lack one typical value: it changed state part-way through, so
/// its windows disagree, or it flipped between states throughout, so its
/// interquartile range is wide. The minimum rule's drift check is covered by
/// the first: a climbing arm's windows disagree.
struct Steadiness {
    label: &'static str,
    medians: Vec<f64>,
}

impl Steadiness {
    fn new(label: &'static str) -> Self {
        Self {
            label,
            medians: Vec::new(),
        }
    }

    /// Record one round's block of timings.
    fn observe(&mut self, block: &[f64]) {
        self.medians.push(median(block));
    }

    /// The arm's figure: the median of its per-round medians.
    fn typical(&self) -> f64 {
        median(&self.medians)
    }

    /// The median of each of [`WINDOWS`] consecutive windows, in order, with
    /// the last absorbing any remainder. Empty when there are fewer rounds
    /// than windows.
    fn window_medians(&self) -> Vec<f64> {
        let n = self.medians.len();
        if n < WINDOWS {
            return Vec::new();
        }
        let size = n / WINDOWS;
        (0..WINDOWS)
            .map(|w| {
                let end = if w + 1 == WINDOWS { n } else { (w + 1) * size };
                median(&self.medians[w * size..end])
            })
            .collect()
    }

    /// `max / min - 1` over the window medians; infinite with no windows.
    fn window_spread(&self) -> f64 {
        let w = self.window_medians();
        if w.is_empty() {
            return f64::INFINITY;
        }
        let hi = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lo = w.iter().copied().fold(f64::INFINITY, f64::min);
        hi / lo - 1.0
    }

    /// The interquartile range of the per-round medians over their median;
    /// infinite with no rounds.
    fn iqr_ratio(&self) -> f64 {
        if self.medians.is_empty() {
            return f64::INFINITY;
        }
        (quantile(&self.medians, 0.75) - quantile(&self.medians, 0.25)) / self.typical()
    }

    fn steady(&self) -> bool {
        self.window_spread() <= TOL_WINDOW_SPREAD && self.iqr_ratio() <= TOL_IQR
    }

    /// The gate's evidence: the window medians and how far apart they are,
    /// and the IQR, each against its limit.
    fn figures(&self) -> String {
        let w = self.window_medians();
        let windows = if w.is_empty() {
            format!("fewer than {WINDOWS} rounds")
        } else {
            let ms: Vec<String> = w.iter().map(|m| format!("{:.3}", m * 1e3)).collect();
            format!(
                "windows {} ms, {:.1}% apart (limit {:.0}%)",
                ms.join(" / "),
                self.window_spread() * 100.0,
                TOL_WINDOW_SPREAD * 100.0
            )
        };
        format!(
            "{windows}; IQR {:.1}% of median (limit {:.0}%)",
            self.iqr_ratio() * 100.0,
            TOL_IQR * 100.0
        )
    }

    /// Why this arm fails the gate, naming the arm and its numbers, for the
    /// verdict line.
    fn why_unsteady(&self) -> String {
        format!("{} ({})", self.label, self.figures())
    }

    fn report(&self) -> String {
        let verdict = if self.steady() { "STEADY" } else { "UNSTEADY" };
        format!(
            "{:24}: {} rounds, median of round medians {:.3} ms | {} | {verdict}",
            self.label,
            self.medians.len(),
            self.typical() * 1e3,
            self.figures()
        )
    }
}

/// The median of `xs`, averaging the middle pair on an even count; NaN when
/// empty.
fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n.is_multiple_of(2) {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    } else {
        s[n / 2]
    }
}

/// The `p` quantile of a non-empty `xs`, interpolating linearly between order
/// statistics (numpy's default method).
fn quantile(xs: &[f64], p: f64) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    let k = (s.len() - 1) as f64 * p;
    let lo = k.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (s[hi] - s[lo]) * (k - lo as f64)
}

/// State plainly what each GPU timing does and does not contain.
///
/// The resident number is the one that is easy to quote and the one that
/// flatters the GPU most: everything that is not the dispatch itself happens
/// outside its loop. The streamed number puts the weight slab back inside.
fn report_gpu_timing_scope(
    gemv: &gpu::GpuGemv<'_>,
    weight_bytes: usize,
    host_cached: bool,
    default_subgroup: u32,
) {
    println!("== what the GPU timings include ==");
    println!(
        "  weights resident: `reset_fences` + `queue_submit` + `wait_for_fences`, and\n    \
         nothing else. The {weight_bytes} B weight slab, the activation repack and\n    \
         upload, every descriptor write and the command-buffer recording all happen\n    \
         once before the loop, and the output readback happens after it. A backend\n    \
         only pays this little when it dispatches the SAME expert again."
    );
    println!(
        "  weights streamed: the same submit, plus the whole {weight_bytes} B slab written\n    \
         into the mapped buffer inside the timed region on every iteration. Prefill\n    \
         visits a different expert on every dispatch, so this is the transfer a real\n    \
         backend cannot avoid. The activations are still uploaded once, which is\n    \
         also what a backend would do: one batch of chunk rows feeds many experts.\n    \
         The CPU arm reads its weights straight from the heap and copies nothing."
    );
    println!(
        "  device timestamps: the dispatch alone, taken from the resident block. It\n    \
         excludes submit and fence latency as well as every transfer."
    );
    // The memory type is whatever `Buffer::new` actually landed in, which is
    // not always what it asked for: it prefers `DEVICE_LOCAL | HOST_VISIBLE |
    // HOST_COHERENT` and falls back to plain `HOST_VISIBLE | HOST_COHERENT`
    // when no unified type exists. The two have different transfer costs, so
    // this reports the flags that came back rather than describing the
    // request.
    let props = gemv.memory_properties();
    let device_local = props.contains(ash::vk::MemoryPropertyFlags::DEVICE_LOCAL);
    let cached = props.contains(ash::vk::MemoryPropertyFlags::HOST_CACHED);
    println!(
        "  memory type      : {props:?}\n    \
         (`src/gpu.rs` takes the FIRST type matching its flags, so this is what the\n    \
         driver listed first and not a tuned choice.)"
    );
    match (device_local, cached) {
        (true, false) => println!(
            "    Device-local, host-visible and NOT host-cached, i.e. write-combined: host\n    \
             writes stream into it, host READS of the output buffer are slow, and that\n    \
             readback sits outside every timed region above."
        ),
        (true, true) => println!(
            "    Device-local, host-visible and HOST_CACHED, so it is NOT write-combined:\n    \
             host reads of the mapping go through the cache and the readback is cheap. The\n    \
             readback still sits outside every timed region above."
        ),
        (false, _) => println!(
            "    !! NOT device-local. `Buffer::new` found no DEVICE_LOCAL host-visible type\n    \
             !! and fell back to plain host-visible memory, so the weight slab and the\n    \
             !! activations live in host memory the GPU reads across the bus. The GPU arm's\n    \
             !! timings are of that arrangement, not of a unified heap, and the\n    \
             !! write-combined reasoning above does not apply."
        ),
    }
    if host_cached && !cached {
        println!(
            "    This device also exposes a HOST_CACHED host-visible type, which is where\n    \
             an output buffer would belong if a readback were ever put inside a timed\n    \
             region. The spike does not use it."
        );
    }
    if gemv.subgroup_pinned() {
        println!(
            "  subgroup width   : REQUIRED of the pipeline at {} lanes via\n    \
             `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`, so the 32-consecutive-uint\n    \
             activation read the packing is built for is the pattern that ran and therefore\n    \
             the pattern that was timed.",
            gpu::ASSUMED_SUBGROUP
        );
    } else {
        println!(
            "  subgroup width   : NOT required of the pipeline. This device cannot pin a\n    \
             compute subgroup size, so the width is the driver's per-pipeline choice; its\n    \
             reported default is {default_subgroup}. The arithmetic is correct at any width\n    \
             -- the shader synchronizes with `barrier()` and shared memory, not with\n    \
             subgroup ops -- but the activation-minor packing assumes 32 lanes share a\n    \
             super-block, and that assumption is UNVERIFIED here."
        );
    }
    println!();
}

/// State the algorithmic asymmetry between the arms, next to the ratio.
///
/// The two arms compute the same thing and do not do the same work to get
/// there, and a reader who does not know that will attribute all of the gap
/// to the hardware.
fn report_arm_asymmetry() {
    println!("== the two arms do not run the same algorithm ==");
    println!(
        "  The shader unpacks each weight row ONCE: it converts d/dmin, expands the\n  \
         eight 6-bit (scale, min) pairs and splits the nibbles into shared memory,\n  \
         then reuses that across all {N_ACTS} activations. The production CPU kernel does\n  \
         not: `crates/core/src/kernels/gemv.rs` loops weight rows outer and\n  \
         activations inner, so `vec_dot_q4_k_q8_k` redoes the f16 conversion, the\n  \
         scale/min unpack and the nibble split on all {N_ACTS} passes over the row."
    );
    println!(
        "  A hoisted CPU prototype, with a deliberately scalar expansion, measured\n  \
         0.482-0.495 ms against production's 0.551-0.568 ms in the same session:\n  \
         **1.15-1.18x, and that figure is a WARM DIAGNOSTIC under rule 2 of\n  \
         `docs/experiments.md`, not a publishable number.** Its conditions: the same\n  \
         throttled box (no_turbo=1, max_perf_pct=60, governor powersave), warm, NOT\n  \
         in a cgroup, and taken before this harness had convergence gating, so the\n  \
         sample was not shown to be a steady state. Nor was that prototype ever\n  \
         checked for bit-identity against production, so the figure is not a\n  \
         verified same-math A/B; and its expansion is scalar where production's\n  \
         inner loop is AVX2, which makes 1.15-1.18x a FLOOR on the asymmetry rather\n  \
         than a measurement of it. The prototype is preserved on branch\n  \
         `perf/q4k-row-hoist-prototype`, where the reconstruction IS bit-identical\n  \
         to production on the shapes tested -- a property of the reconstruction, not\n  \
         of the arm this figure came from -- and it is not a merge candidate.\n  \
         Closing the asymmetry is a change to the CPU kernel in `crates/core`, not\n  \
         to the GPU path.\n"
    );
}

/// Score the two arms against each other and gate on the result.
///
/// Returns whether every threshold held. Printing the numbers is not enough:
/// an unchecked metric is one nobody reads, and the shape of error this
/// kernel is most likely to grow -- a wrong scale conversion applied
/// uniformly -- moves the headline relative error by less than the noise
/// already in it. The signed mean is what separates the two.
fn report_agreement(cpu: &[f32], gpu: &[f32]) -> bool {
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut sum_rel = 0f64;
    let mut max_abs_at = 0usize;
    let mut max_rel_at = 0usize;
    let mut scale = 0f64;
    for &c in cpu {
        scale = scale.max(f64::from(c).abs());
    }
    // Outputs whose reference is a rounding artefact of cancellation carry a
    // meaningless relative error, so they get their own line rather than
    // being dropped or being allowed to define the headline number.
    let floor = scale * SIGNIFICANT;
    let mut max_rel_big = 0f64;
    let mut signed_sum_big = 0f64;
    let mut big = 0usize;
    for (i, (&c, &g)) in cpu.iter().zip(gpu).enumerate() {
        let (c, g) = (f64::from(c), f64::from(g));
        let abs = (c - g).abs();
        // Relative to the reference magnitude; a zero reference contributes
        // its absolute error, which is the only honest reading there.
        let rel = if c == 0.0 { abs } else { abs / c.abs() };
        if abs > max_abs {
            max_abs = abs;
            max_abs_at = i;
        }
        if rel > max_rel {
            max_rel = rel;
            max_rel_at = i;
        }
        if c.abs() >= floor {
            max_rel_big = max_rel_big.max(rel);
            // Signed, and signed against the reference's own sign, so a
            // uniform scale factor adds up instead of cancelling between
            // positive and negative outputs.
            signed_sum_big += (g - c) / c;
            big += 1;
        }
        sum_rel += rel;
    }
    let n = cpu.len();
    let mean_rel = sum_rel / n as f64;
    let signed_bias = if big == 0 {
        0.0
    } else {
        signed_sum_big / big as f64
    };
    let max_abs_tol = scale * TOL_MAX_ABS_REL_SCALE;

    println!("== numerical agreement ({n} outputs) ==");
    println!("  reference magnitude (max |cpu|): {scale:.6}");
    println!(
        "  max abs err : {max_abs:.3e}  [tol {max_abs_tol:.3e}] {}  at [{}][{}] (cpu {:.6} gpu {:.6})",
        pass_fail(max_abs <= max_abs_tol),
        max_abs_at / N_ACTS,
        max_abs_at % N_ACTS,
        cpu[max_abs_at],
        gpu[max_abs_at]
    );
    println!(
        "  max rel err : {max_rel:.3e}  [ungated: dominated by cancellation]  at [{}][{}] (cpu {:.6} gpu {:.6})",
        max_rel_at / N_ACTS,
        max_rel_at % N_ACTS,
        cpu[max_rel_at],
        gpu[max_rel_at]
    );
    println!(
        "  mean rel err: {mean_rel:.3e}  [tol {TOL_MEAN_REL:.3e}] {}",
        pass_fail(mean_rel <= TOL_MEAN_REL)
    );
    println!(
        "  max rel err over the {big} outputs with |cpu| >= {floor:.3e} ({}% of scale): \
         {max_rel_big:.3e}  [tol {TOL_MAX_REL_BIG:.3e}] {}",
        SIGNIFICANT * 100.0,
        pass_fail(max_rel_big <= TOL_MAX_REL_BIG)
    );
    println!(
        "  mean SIGNED rel err over the same {big}: {signed_bias:+.3e}  \
         [tol +/-{TOL_SIGNED_BIAS:.3e}] {}",
        pass_fail(signed_bias.abs() <= TOL_SIGNED_BIAS)
    );
    let exact = cpu.iter().zip(gpu).filter(|(a, b)| a == b).count();
    println!("  bit-identical outputs: {exact}/{n}");
    let ok = max_abs <= max_abs_tol
        && mean_rel <= TOL_MEAN_REL
        && max_rel_big <= TOL_MAX_REL_BIG
        && signed_bias.abs() <= TOL_SIGNED_BIAS;
    println!("  NUMERICS: {}", pass_fail(ok));
    if !ok {
        println!(
            "  !! the shader does not reproduce the CPU kernel inside tolerance. No timing\n  \
             !! below describes the same computation on both arms."
        );
    }
    println!();
    ok
}

/// Read a one-line sysfs file, trimmed.
fn sysfs(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_owned())
}

/// Read one cpufreq attribute for every CPU in `cpus`, grouped by value.
///
/// Returns `(value, cpus with that value)` pairs in first-seen order. A CPU
/// whose file is missing or unreadable groups under `unknown`, which is a
/// distinct answer from `performance` and is treated as one.
fn per_cpu_attr(cpus: &[usize], attr: &str) -> Vec<(String, Vec<usize>)> {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for &cpu in cpus {
        let value = sysfs(&format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{attr}"))
            .unwrap_or_else(|| "unknown".to_owned());
        if let Some(slot) = groups.iter_mut().find(|(v, _)| *v == value) {
            slot.1.push(cpu);
        } else {
            groups.push((value, vec![cpu]));
        }
    }
    groups
}

/// Read one cpufreq attribute for one CPU as a kHz number.
///
/// `None` covers both "the file is not there" and "it did not parse", which
/// are the same answer for gating purposes: the value is unknown, so it
/// cannot be compared.
fn cpu_khz(cpu: usize, attr: &str) -> Option<u64> {
    sysfs(&format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{attr}"))?
        .parse::<u64>()
        .ok()
}

/// Print the clock ceiling of every CPU the pool pins, and return every
/// reason arm A is not free to run at full speed.
///
/// This is not decoration. Arm A is only as fast as the clock it is allowed
/// to run at, and on a laptop under a power daemon that ceiling can sit at
/// well under half the part's turbo frequency. A GPU-vs-CPU ratio measured
/// against a throttled CPU is a ratio about the power policy, not about the
/// two units, so a non-empty return here invalidates the verdict outright
/// rather than printing a warning beside a clean-looking number.
///
/// Four knobs are gated, not three. `no_turbo`, `max_perf_pct` and the
/// governor are the ones a reader thinks of, and clearing them is what the
/// unthrottled re-run does; `scaling_max_freq` is the one a TLP or `cpupower`
/// profile pins directly, underneath all three. A box with `no_turbo=0`,
/// `max_perf_pct=100`, governor `performance` and `scaling_max_freq` held
/// below `cpuinfo_max_freq` is still a throttled box, and gating on the first
/// three alone would print PASS on it and emit the clean quotable verdict
/// this whole harness exists to withhold. So the per-CPU ceiling is compared
/// against that CPU's own hardware maximum.
///
/// Every pinned CPU is read, not `cpu0`. `cpufreq` policies are per-policy
/// and a policy need not cover the whole package: on a hybrid part the
/// P-cores and E-cores are separate policies, and reading `cpu0` alone would
/// have reported `performance` for a pool half of whose shards were pinned
/// to a capped policy. `intel_pstate`'s `no_turbo` and `max_perf_pct` are
/// package-wide and are read once.
///
/// On a `locked` run the `scaling_max_freq` ceiling is deliberate, so that
/// one comparison is skipped and [`clock`] gates it instead, more strictly:
/// min must equal max at the locked value, and the observed clock must stay
/// on it for the whole run. The other three knobs are still gated.
fn report_cpu_clock_caps(cpus: &[usize], locked: bool) -> Vec<String> {
    let governors = per_cpu_attr(cpus, "scaling_governor");
    let max_freqs = per_cpu_attr(cpus, "scaling_max_freq");
    let hw_max_freqs = per_cpu_attr(cpus, "cpuinfo_max_freq");
    let no_turbo = sysfs("/sys/devices/system/cpu/intel_pstate/no_turbo");
    let max_pct = sysfs("/sys/devices/system/cpu/intel_pstate/max_perf_pct");

    println!(
        "  clock policy over the {} pinned cpus {cpus:?}:",
        cpus.len()
    );
    for (governor, on) in &governors {
        println!("    governor {governor} on cpus {on:?}");
    }
    for (label, groups) in [
        ("scaling_max_freq", &max_freqs),
        ("cpuinfo_max_freq", &hw_max_freqs),
    ] {
        for (khz, on) in groups {
            match khz.parse::<u64>() {
                Ok(khz) => println!("    {label} {:.2} GHz on cpus {on:?}", khz as f64 / 1e6),
                Err(_) => println!("    {label} {khz} on cpus {on:?}"),
            }
        }
    }
    print!("    intel_pstate:");
    match &no_turbo {
        Some(v) => print!(" no_turbo {v}"),
        None => print!(" no_turbo unreadable"),
    }
    match &max_pct {
        Some(v) => print!(", max_perf_pct {v}"),
        None => print!(", max_perf_pct unreadable"),
    }
    println!();

    let mut throttles = Vec::new();
    if no_turbo.as_deref() == Some("1") {
        throttles.push("no_turbo=1".to_owned());
    }
    if let Some(pct) = &max_pct
        && pct.parse::<u32>().is_ok_and(|p| p < 100)
    {
        throttles.push(format!("max_perf_pct={pct}"));
    }
    for (governor, on) in &governors {
        if governor != "performance" {
            throttles.push(format!("governor={governor} on cpus {on:?}"));
        }
    }
    // The knob the other three do not cover: a profile can pin
    // `scaling_max_freq` under `cpuinfo_max_freq` with turbo enabled, the
    // percentage at 100 and the governor on `performance`. Compared per CPU,
    // because the ceiling is a per-policy value.
    let mut capped = Vec::new();
    let mut unreadable = Vec::new();
    if locked {
        println!("    scaling_max_freq ceiling: deliberate, checked by the clock lock instead");
    }
    for &cpu in cpus.iter().filter(|_| !locked) {
        match (
            cpu_khz(cpu, "scaling_max_freq"),
            cpu_khz(cpu, "cpuinfo_max_freq"),
        ) {
            (Some(scaling), Some(hw)) if scaling < hw => capped.push(format!(
                "cpu{cpu} {:.2} of {:.2} GHz",
                scaling as f64 / 1e6,
                hw as f64 / 1e6
            )),
            (Some(_), Some(_)) => {}
            _ => unreadable.push(cpu),
        }
    }
    if !capped.is_empty() {
        throttles.push(format!(
            "scaling_max_freq below cpuinfo_max_freq on {}",
            capped.join(", ")
        ));
    }
    if !unreadable.is_empty() {
        // Not a throttle: an unknown ceiling is not evidence of a capped one.
        // Printed, because it is the one gate that then has nothing behind it.
        println!(
            "    !! scaling_max_freq/cpuinfo_max_freq unreadable on cpus {unreadable:?}; the \
             per-cpu clock-ceiling gate could not be evaluated there"
        );
    }
    if !throttles.is_empty() {
        println!(
            "  !! THE CPU ARM IS NOT FREE TO RUN AT FULL SPEED ({}). Arm A is capped\n  \
             !! below this part's turbo frequency, so any CPU-vs-GPU ratio measured here\n  \
             !! understates the CPU by roughly that much. The verdict below is printed\n  \
             !! INVALID and this binary exits non-zero; unlock the clocks and re-run\n  \
             !! before quoting a speedup as a property of the two units.",
            throttles.join(", ")
        );
    }
    throttles
}

/// The kernel's 1-minute load average, as a string.
///
/// Printed with the timings because this box is shared with the author's
/// other work: a CPU arm measured under load average 3 is not the same
/// claim as one measured on an idle machine, and the reader needs to see
/// which they are looking at.
fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Median, min, max over a set of timings. The author's box is noisy; a mean
/// would hide that and a single run would be a coin flip.
struct Stats {
    median: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn of(samples: &mut [f64]) -> Self {
        samples.sort_by(f64::total_cmp);
        Self {
            median: samples[samples.len() / 2],
            min: samples[0],
            max: samples[samples.len() - 1],
        }
    }

    /// One line per arm. `tag` is the caller's leading indent, which is
    /// [`INVALID_TAG`] on a run whose verdict is INVALID: a GFLOP/s figure
    /// is a throughput claim and must not be printable without one.
    fn report(&self, label: &str, flops: f64, tag: &str) {
        println!(
            "{tag}{label:24}: median {:8.3} ms ({:7.2} GFLOP/s)  range [{:.3}, {:.3}] ms",
            self.median * 1e3,
            flops / self.median / 1e9,
            self.min * 1e3,
            self.max * 1e3
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run with every gate satisfied. The one input shape that lets a ratio
    /// be printed, so every test below is a departure from it.
    fn clean() -> Vec<String> {
        invalidations(true, false, "REAL slab", &[], &[], &[])
    }

    /// The only way a ratio ever reaches stdout.
    #[test]
    fn a_clean_run_has_no_invalidations() {
        assert!(clean().is_empty(), "got {:?}", clean());
    }

    /// The first of the two defects the correction pass exists for, and the
    /// one no device-free test could reach while this logic lived inline in
    /// `run`: a numerics FAIL means the two arms did not compute the same
    /// thing, so their ratio is not a comparison at all.
    #[test]
    fn a_numerics_failure_invalidates_the_verdict() {
        let v = invalidations(false, false, "REAL slab", &[], &[], &[]);
        assert_eq!(v.len(), 1, "got {v:?}");
        assert!(
            v[0].contains("did not compute the same thing"),
            "the reason must say what a numerics FAIL means: {}",
            v[0]
        );
    }

    /// Numerics is reported first even when other gates also failed, because
    /// it is the gate that says the comparison is a comparison.
    #[test]
    fn numerics_is_reported_first() {
        let throttles = ["no_turbo=1".to_owned()];
        let unsteady = ["CPU (windows apart)".to_owned()];
        let lock = ["CPU ran at 2.40-3.00 GHz".to_owned()];
        let v = invalidations(false, true, "SYNTHETIC bytes", &throttles, &lock, &unsteady);
        assert_eq!(v.len(), 5, "every failing gate must contribute: {v:?}");
        assert!(v[0].contains("did not compute the same thing"), "{v:?}");
    }

    /// The second defect: a run on invented bytes is not a run on a real
    /// expert slab, and the provenance travels with the reason.
    #[test]
    fn synthetic_weights_invalidate_the_verdict() {
        let v = invalidations(true, true, "SYNTHETIC q4_k bytes (768x2048)", &[], &[], &[]);
        assert_eq!(v.len(), 1, "got {v:?}");
        assert!(v[0].contains("SYNTHETIC"), "{}", v[0]);
        assert!(
            v[0].contains("768x2048"),
            "the provenance must travel with the reason: {}",
            v[0]
        );
    }

    /// A throttled CPU arm makes the ratio a fact about the power daemon.
    /// Every reason the clock gate found is named, including the per-cpu
    /// `scaling_max_freq` cap that the package-wide knobs do not show.
    #[test]
    fn a_throttled_cpu_invalidates_the_verdict() {
        let throttles = [
            "no_turbo=1".to_owned(),
            "scaling_max_freq below cpuinfo_max_freq on cpu0 2.30 of 4.80 GHz".to_owned(),
        ];
        let v = invalidations(true, false, "REAL slab", &throttles, &[], &[]);
        assert_eq!(v.len(), 1, "got {v:?}");
        assert!(v[0].contains("no_turbo=1"), "{}", v[0]);
        assert!(v[0].contains("scaling_max_freq"), "{}", v[0]);
    }

    /// An arm that is not one steady state invalidates the verdict, and the
    /// reason names the arm and carries its numbers.
    #[test]
    fn an_unsteady_arm_invalidates_the_verdict() {
        let a = arm("GPU  (weights streamed)", &PROD_STREAMED);
        let v = invalidations(true, false, "REAL slab", &[], &[], &[a.why_unsteady()]);
        assert_eq!(v.len(), 1, "got {v:?}");
        assert!(v[0].contains("not a steady state"), "{}", v[0]);
        assert!(
            v[0].contains("GPU  (weights streamed)"),
            "the arm must be named: {}",
            v[0]
        );
        assert!(
            v[0].contains("windows"),
            "the evidence must travel: {}",
            v[0]
        );
    }

    /// A lock that was not in place, or did not hold, invalidates the
    /// verdict: the ratio would be about clocks nobody ran at.
    #[test]
    fn a_failed_clock_lock_invalidates_the_verdict() {
        let lock = ["iGPU ran at 900-1400 MHz against 1400 locked".to_owned()];
        let v = invalidations(true, false, "REAL slab", &[], &lock, &[]);
        assert_eq!(v.len(), 1, "got {v:?}");
        assert!(v[0].contains("clock lock"), "{}", v[0]);
        assert!(
            v[0].contains("900-1400 MHz"),
            "the evidence must travel: {}",
            v[0]
        );
    }

    /// An arm fed one-sample blocks, so each round's median is the value.
    fn arm(label: &'static str, medians: &[f64]) -> Steadiness {
        let mut a = Steadiness::new(label);
        for &m in medians {
            a.observe(&[m]);
        }
        a
    }

    /// Round-to-round noise well inside both tolerances is one steady state.
    #[test]
    fn noise_inside_tolerance_is_steady() {
        let m: Vec<f64> = (0..ROUNDS)
            .map(|r| if r % 2 == 0 { 1.02 } else { 0.98 })
            .collect();
        let a = arm("t", &m);
        assert!(a.steady(), "{}", a.report());
    }

    /// A step part-way through is two states, however flat each side is. The
    /// minimum-based rule called the flat tail after a step converged.
    #[test]
    fn a_step_change_is_unsteady() {
        let m: Vec<f64> = (0..ROUNDS)
            .map(|r| if r < ROUNDS / 2 { 1.0 } else { 1.2 })
            .collect();
        let a = arm("t", &m);
        assert!(a.window_spread() > TOL_WINDOW_SPREAD, "{}", a.report());
        assert!(!a.steady());
    }

    /// A steady climb, as from a part thermally derating, is not a steady
    /// state: its windows disagree.
    #[test]
    fn drift_is_unsteady() {
        let m: Vec<f64> = (0..ROUNDS).map(|r| 1.0 + 0.005 * r as f64).collect();
        let a = arm("t", &m);
        assert!(a.window_spread() > TOL_WINDOW_SPREAD, "{}", a.report());
        assert!(!a.steady());
    }

    /// An arm flipping between two states throughout has windows that agree
    /// and still no single typical value. The IQR is what catches it.
    #[test]
    fn flipping_between_two_states_is_unsteady() {
        let m: Vec<f64> = (0..ROUNDS)
            .map(|r| if r % 2 == 0 { 1.0 } else { 1.7 })
            .collect();
        let a = arm("t", &m);
        assert!(a.window_spread() <= TOL_WINDOW_SPREAD, "{}", a.report());
        assert!(a.iqr_ratio() > TOL_IQR, "{}", a.report());
        assert!(!a.steady());
    }

    /// Fewer rounds than windows is no evidence of a steady state.
    #[test]
    fn too_few_rounds_is_unsteady() {
        assert!(!arm("t", &[1.0, 1.0]).steady());
        assert!(!arm("t", &[]).steady());
    }

    /// The median averages the middle pair; the quantile interpolates.
    #[test]
    fn median_and_quantile_match_the_textbook() {
        assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < 1e-12);
        assert!((median(&[4.0, 1.0, 3.0, 2.0]) - 2.5).abs() < 1e-12);
        assert!((quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.25) - 2.0).abs() < 1e-12);
        assert!((quantile(&[1.0, 2.0, 3.0, 4.0], 0.75) - 3.25).abs() < 1e-12);
    }

    // Per-round medians in microseconds from the two unthrottled runs of
    // 2026-10-09: Intel Arc (MTL) iGPU against 6 pinned P-cores, turbo on,
    // built before the clocks were unlocked. The tolerances were set from the
    // size of the effects being decided, not fitted to these runs.
    const PROD_CPU: [f64; 60] = [
        279.0, 264.0, 276.0, 270.0, 267.0, 267.0, 271.0, 271.0, 271.0, 272.0, 278.0, 267.0, 272.0,
        267.0, 261.0, 267.0, 273.0, 269.0, 270.0, 271.0, 267.0, 267.0, 270.0, 271.0, 267.0, 269.0,
        265.0, 267.0, 266.0, 272.0, 273.0, 273.0, 271.0, 272.0, 272.0, 278.0, 449.0, 276.0, 275.0,
        277.0, 479.0, 282.0, 314.0, 296.0, 276.0, 276.0, 276.0, 281.0, 271.0, 276.0, 279.0, 287.0,
        285.0, 276.0, 274.0, 278.0, 276.0, 280.0, 285.0, 285.0,
    ];
    const PROD_RESIDENT: [f64; 60] = [
        251.0, 386.0, 314.0, 283.0, 281.0, 274.0, 278.0, 277.0, 278.0, 278.0, 270.0, 258.0, 272.0,
        269.0, 248.0, 264.0, 279.0, 271.0, 272.0, 272.0, 268.0, 271.0, 270.0, 270.0, 281.0, 274.0,
        264.0, 275.0, 264.0, 279.0, 275.0, 256.0, 291.0, 279.0, 274.0, 284.0, 278.0, 286.0, 283.0,
        295.0, 280.0, 313.0, 311.0, 284.0, 304.0, 308.0, 314.0, 297.0, 272.0, 278.0, 277.0, 271.0,
        272.0, 271.0, 265.0, 273.0, 273.0, 255.0, 270.0, 279.0,
    ];
    const PROD_STREAMED: [f64; 60] = [
        289.0, 324.0, 274.0, 272.0, 299.0, 258.0, 298.0, 292.0, 270.0, 291.0, 276.0, 299.0, 296.0,
        287.0, 264.0, 293.0, 310.0, 272.0, 288.0, 301.0, 303.0, 267.0, 286.0, 413.0, 433.0, 312.0,
        405.0, 394.0, 427.0, 420.0, 405.0, 392.0, 413.0, 447.0, 413.0, 429.0, 421.0, 420.0, 412.0,
        410.0, 407.0, 445.0, 464.0, 436.0, 298.0, 314.0, 292.0, 311.0, 277.0, 270.0, 246.0, 306.0,
        290.0, 300.0, 283.0, 287.0, 267.0, 270.0, 287.0, 269.0,
    ];
    const HOIST_CPU: [f64; 60] = [
        238.0, 213.0, 234.0, 214.0, 212.0, 210.0, 228.0, 225.0, 231.0, 224.0, 226.0, 213.0, 212.0,
        390.0, 362.0, 425.0, 494.0, 366.0, 345.0, 388.0, 349.0, 328.0, 374.0, 369.0, 374.0, 331.0,
        385.0, 384.0, 364.0, 348.0, 352.0, 335.0, 370.0, 335.0, 364.0, 374.0, 366.0, 217.0, 216.0,
        216.0, 213.0, 213.0, 214.0, 217.0, 215.0, 237.0, 216.0, 215.0, 214.0, 375.0, 222.0, 384.0,
        431.0, 383.0, 376.0, 372.0, 381.0, 216.0, 394.0, 373.0,
    ];
    const HOIST_RESIDENT: [f64; 60] = [
        269.0, 284.0, 269.0, 280.0, 274.0, 267.0, 271.0, 278.0, 272.0, 270.0, 269.0, 271.0, 275.0,
        300.0, 301.0, 312.0, 284.0, 249.0, 267.0, 281.0, 278.0, 249.0, 289.0, 269.0, 288.0, 257.0,
        273.0, 299.0, 263.0, 256.0, 286.0, 256.0, 279.0, 268.0, 265.0, 266.0, 281.0, 267.0, 443.0,
        461.0, 459.0, 445.0, 385.0, 360.0, 344.0, 315.0, 305.0, 284.0, 285.0, 244.0, 313.0, 286.0,
        292.0, 299.0, 280.0, 271.0, 287.0, 277.0, 272.0, 293.0,
    ];
    const HOIST_STREAMED: [f64; 60] = [
        247.0, 292.0, 260.0, 274.0, 308.0, 256.0, 292.0, 298.0, 264.0, 278.0, 301.0, 293.0, 353.0,
        331.0, 270.0, 332.0, 310.0, 323.0, 279.0, 303.0, 275.0, 277.0, 275.0, 273.0, 278.0, 300.0,
        297.0, 307.0, 416.0, 409.0, 431.0, 417.0, 418.0, 404.0, 426.0, 419.0, 423.0, 420.0, 477.0,
        465.0, 461.0, 408.0, 349.0, 323.0, 384.0, 345.0, 303.0, 317.0, 301.0, 284.0, 321.0, 323.0,
        326.0, 327.0, 266.0, 321.0, 289.0, 428.0, 439.0, 425.0,
    ];

    /// The production-kernel run: the CPU and resident GPU held one state,
    /// and the streamed GPU had a slow episode in the middle third.
    #[test]
    fn the_production_run_is_steady_except_the_streamed_arm() {
        let cpu = arm("cpu", &PROD_CPU);
        let res = arm("res", &PROD_RESIDENT);
        let st = arm("str", &PROD_STREAMED);
        assert!(cpu.steady(), "{}", cpu.report());
        assert!(res.steady(), "{}", res.report());
        assert!(!st.steady(), "{}", st.report());
        let w = st.window_medians();
        assert!(w[1] > 1.3 * w[0] && w[1] > 1.3 * w[2], "{w:?}");
    }

    /// The hoisted run: the CPU flipped between ~0.21 and ~0.36 ms, and both
    /// GPU arms had their own slow episodes. No arm is one steady state.
    #[test]
    fn the_hoisted_run_is_unsteady_on_every_arm() {
        for (name, m) in [
            ("cpu", &HOIST_CPU),
            ("res", &HOIST_RESIDENT),
            ("str", &HOIST_STREAMED),
        ] {
            let a = arm(name, m);
            assert!(!a.steady(), "{}", a.report());
        }
    }

    /// Alternating-sign reference outputs, so a uniform scale error cannot
    /// cancel itself between positive and negative entries.
    fn signed_reference() -> Vec<f32> {
        (0..256)
            .map(|i| {
                let v = 1.0 + i as f32;
                if i % 2 == 0 { v } else { -v }
            })
            .collect()
    }

    /// The gate that motivates the signed metric: a uniform 1.00003x scale
    /// error sits inside every unsigned threshold and must still fail.
    #[test]
    fn a_uniform_scale_error_fails_the_signed_gate() {
        let cpu = signed_reference();
        let gpu: Vec<f32> = cpu.iter().map(|&c| c * 1.000_03).collect();
        assert!(
            !report_agreement(&cpu, &gpu),
            "a uniform 1.00003x scale error must fail the numerics gate"
        );
    }

    /// And symmetric rounding noise of the same per-output magnitude must
    /// pass it, or the gate would fail every honest run.
    #[test]
    fn symmetric_rounding_noise_passes_the_signed_gate() {
        let cpu = signed_reference();
        let gpu: Vec<f32> = cpu
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                if i % 2 == 0 {
                    c * 1.000_03
                } else {
                    c * 0.999_97
                }
            })
            .collect();
        assert!(
            report_agreement(&cpu, &gpu),
            "symmetric noise must pass the numerics gate"
        );
    }
}
