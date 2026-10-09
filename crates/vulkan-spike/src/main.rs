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
//! numerically inside named tolerances *including* a signed-bias check, each
//! arm's sampled minimum must stop improving (and must not be drifting
//! upward) before the sample is called a steady state, the machine's clock
//! policy must not be capping the CPU arm, and the weights must be a real
//! expert slab rather than the synthetic fallback. A ratio measured against
//! a throttled CPU is a fact about the power daemon, a ratio measured before
//! either arm reaches steady state is a fact about the sample size, and a
//! ratio measured on invented bytes is a fact about nothing, so none of them
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
/// Rounds that always run, before convergence is even tested.
const MIN_ROUNDS: usize = 3;
/// Hard cap on rounds. Reaching it without converging is reported, not
/// hidden. A round is ~70 ms, so 60 rounds is ~4 s of timed work. 20 was
/// enough at locked throttled clocks (EXP-026 converged in 10 and 13 rounds)
/// but not with turbo on, where round-to-round jitter in the block minimum
/// restarts patience far more often.
const MAX_ROUNDS: usize = 60;
/// Consecutive rounds that must fail to improve an arm's running minimum
/// before that minimum counts as converged.
const PATIENCE: usize = 3;
/// Relative improvement in the running minimum that still counts as progress.
/// A round that shaves less than this off the best sample is noise, not a
/// unit still climbing.
const MIN_IMPROVEMENT: f64 = 0.005;

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
    let throttles = report_cpu_clock_caps(pool.topology().compute_cpus());
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
    // Rounds are not a fixed count. Every arm keeps sampling until its
    // running minimum has stopped improving for [`PATIENCE`] rounds, so the
    // sample size is set by the slowest arm to settle rather than by a
    // constant chosen before anything was measured.
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
    let mut cpu_conv = Convergence::new(cpu_label);
    let mut res_conv = Convergence::new("GPU  (weights resident)");
    let mut str_conv = Convergence::new("GPU  (weights streamed)");
    for round in 1..=MAX_ROUNDS {
        let mut cpu_round = Vec::with_capacity(BLOCK);
        for _ in 0..REWARM {
            cpu_iter()?;
        }
        for _ in 0..BLOCK {
            cpu_round.push(cpu_iter()?);
        }

        let mut res_round = Vec::with_capacity(BLOCK);
        for _ in 0..REWARM {
            gemv.run()?;
        }
        for _ in 0..BLOCK {
            let (host, device) = gemv.run()?;
            res_round.push(host);
            gpu_device.push(device);
        }

        let mut str_round = Vec::with_capacity(BLOCK);
        for _ in 0..REWARM {
            gemv.run_streamed(&slab.bytes)?;
        }
        for _ in 0..BLOCK {
            let (host, _) = gemv.run_streamed(&slab.bytes)?;
            str_round.push(host);
        }

        cpu_conv.observe(round, &cpu_round);
        res_conv.observe(round, &res_round);
        str_conv.observe(round, &str_round);
        rounds.push((
            Stats::of(&mut cpu_round.clone()),
            Stats::of(&mut res_round.clone()),
            Stats::of(&mut str_round.clone()),
        ));
        cpu_times.extend_from_slice(&cpu_round);
        gpu_resident.extend_from_slice(&res_round);
        gpu_streamed.extend_from_slice(&str_round);

        if round >= MIN_ROUNDS
            && cpu_conv.converged()
            && res_conv.converged()
            && str_conv.converged()
        {
            break;
        }
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

    let arms = [&cpu_conv, &res_conv, &str_conv];
    let converged = arms.iter().all(|c| c.converged());
    let drifting: Vec<&str> = arms
        .iter()
        .filter(|c| c.drifting())
        .map(|c| c.label)
        .collect();

    // Every reason this run's timings are not a result, computed *here*
    // rather than after the figures it disqualifies have already been
    // printed. The invalidation set is knowable at this point -- numerics,
    // provenance and the clock policy were settled before the loop, and
    // convergence and drift the moment it ended -- so every line below can
    // carry it. It used to be computed at the very bottom, which is how
    // 10-20 clean unqualified ratio and GFLOP/s lines came to be printed
    // above a verdict that said none of them counted.
    let invalid = invalidations(
        numerics_ok,
        slab.synthetic,
        &slab.provenance,
        &throttles,
        converged,
        &drifting,
    );
    // Two spaces when the run is valid, so the layout is unchanged; the
    // marker otherwise. Nothing below prints a timing without it.
    let tag: &str = if invalid.is_empty() {
        "  "
    } else {
        INVALID_TAG
    };

    println!(
        "== sampling ({} rounds x {BLOCK} timed runs per arm, {WARMUP} + {REWARM}/round warm-up) ==",
        rounds.len()
    );
    for c in arms {
        println!("{tag}{}", c.report());
    }
    if !converged {
        println!(
            "  !! at least one arm's minimum had not settled when the {MAX_ROUNDS}-round\n  \
             !! cap was hit. Every figure below is an upper bound on that arm and the\n  \
             !! ratio is not a stable quantity."
        );
    }
    if !drifting.is_empty() {
        println!(
            "  !! {} got monotonically WORSE across rounds. That is drift -- something\n  \
             !! else started on this box, or the part is thermally derating -- and not a\n  \
             !! steady state. It is reported separately from staleness on purpose: an\n  \
             !! arm whose minimum keeps rising also never improves, and counting that\n  \
             !! toward patience would print CONVERGED over data that shows the opposite.",
            drifting.join(" and ")
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

    // Two ratios, because they answer different questions. The median-vs-
    // median one is what this box delivers as it is actually loaded. The
    // best-vs-best one uses each arm's least-disturbed sample, which is the
    // closest this machine gets to an uncontended measurement -- and it is
    // the conservative reading, because the CPU arm is the one that
    // contention hurts. Both are computed inside the valid branch only: on
    // an invalid run no ratio is formatted at all.
    if invalid.is_empty() {
        let median_ratio = cpu.median / gpu_r.median;
        let best_ratio = cpu.min / gpu_r.min;
        let streamed_ratio = cpu.min / gpu_s.min;
        let verdict = if best_ratio > 1.0 { "FASTER" } else { "SLOWER" };
        // The provenance is inside the sentence, not on a line above it: a
        // ratio measured on invented bytes and a ratio measured on a real
        // expert slab are different claims, and the quotable line has to say
        // which one it is.
        println!(
            "VERDICT: on {}, iGPU is {best_ratio:.2}x {verdict} than the {}-shard pinned CPU \
             pool with the weights already resident, and {streamed_ratio:.2}x with the weight \
             slab streamed, on best-vs-best ({:.3} ms resident / {:.3} ms streamed GPU vs \
             {:.3} ms CPU); {median_ratio:.2}x resident on median-vs-median ({:.3} ms vs \
             {:.3} ms).",
            slab.provenance,
            pool.shards(),
            gpu_r.min * 1e3,
            gpu_s.min * 1e3,
            cpu.min * 1e3,
            gpu_r.median * 1e3,
            cpu.median * 1e3
        );
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
        pass_fail(throttles.is_empty()),
        pass_fail(converged && drifting.is_empty())
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
    converged: bool,
    drifting: &[&str],
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
    if !converged {
        invalid.push(format!(
            "sampled minimum not converged at the {MAX_ROUNDS}-round cap"
        ));
    }
    if !drifting.is_empty() {
        invalid.push(format!(
            "sampled minimum drifting upward, not a steady state ({})",
            drifting.join(", ")
        ));
    }
    invalid
}

/// Tracks one arm's running minimum across rounds, and whether it has settled.
///
/// The question a fixed iteration count cannot answer is whether the unit was
/// still speeding up while it was being measured. A minimum that keeps
/// falling round after round says it was.
///
/// A minimum that keeps *rising* is the other failure, and it is not the same
/// one. The running minimum is monotone by construction, so an arm getting
/// steadily worse stops improving it -- which, counted naively, looks exactly
/// like an arm that has settled. A build starting on the box mid-run would
/// therefore have printed `CONVERGED` over data showing drift. So each
/// round's own block minimum is compared against the previous round's, and a
/// round that is materially worse is recorded as drift and *restarts*
/// patience rather than counting toward it: a round that got worse is
/// positive evidence against a steady state, not neutral evidence for one.
struct Convergence {
    label: &'static str,
    min: f64,
    samples: usize,
    rounds: usize,
    /// Round in which the minimum last improved by more than
    /// [`MIN_IMPROVEMENT`].
    last_improved: usize,
    stale: usize,
    /// The previous round's own block minimum, which is what drift is
    /// measured against. The *running* minimum cannot detect drift: it never
    /// rises.
    prev_block_min: Option<f64>,
    /// Consecutive rounds whose block minimum rose by more than
    /// [`MIN_IMPROVEMENT`] over the round before.
    rising: usize,
    /// The worst run of consecutive rising rounds seen, so a drift that ends
    /// before the cap is still reported.
    worst_rising: usize,
}

impl Convergence {
    fn new(label: &'static str) -> Self {
        Self {
            label,
            min: f64::INFINITY,
            samples: 0,
            rounds: 0,
            last_improved: 0,
            stale: 0,
            prev_block_min: None,
            rising: 0,
            worst_rising: 0,
        }
    }

    fn observe(&mut self, round: usize, block: &[f64]) {
        let block_min = block.iter().copied().fold(f64::INFINITY, f64::min);
        self.samples += block.len();
        self.rounds = round;
        let improved = block_min < self.min * (1.0 - MIN_IMPROVEMENT);
        let rose = self
            .prev_block_min
            .is_some_and(|prev| block_min > prev * (1.0 + MIN_IMPROVEMENT));
        if improved {
            self.last_improved = round;
            self.stale = 0;
            self.rising = 0;
        } else if rose {
            self.rising += 1;
            self.worst_rising = self.worst_rising.max(self.rising);
            // Not `stale += 1`: this round is evidence the arm is moving,
            // just in the wrong direction.
            self.stale = 0;
        } else {
            self.stale += 1;
            self.rising = 0;
        }
        self.prev_block_min = Some(block_min);
        self.min = self.min.min(block_min);
    }

    fn converged(&self) -> bool {
        self.stale >= PATIENCE && !self.drifting()
    }

    /// Whether this arm's block minimum rose for [`PATIENCE`] rounds running
    /// at any point in the sample.
    fn drifting(&self) -> bool {
        self.worst_rising >= PATIENCE
    }

    fn report(&self) -> String {
        let label = self.label;
        let (n, r) = (self.samples, self.rounds);
        let min = self.min * 1e3;
        let head = format!("{label:24}: {n} timed samples over {r} rounds, min {min:.3} ms");
        if self.drifting() {
            format!(
                "{head} DRIFTING (block minimum rose for {} consecutive rounds; a rising \
                 minimum is not a steady state and does not count as patience)",
                self.worst_rising
            )
        } else if self.converged() {
            format!(
                "{head} CONVERGED (no better min in the last {} rounds)",
                self.stale
            )
        } else {
            format!(
                "{head} NOT CONVERGED (last improved in round {})",
                self.last_improved
            )
        }
    }
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
fn report_cpu_clock_caps(cpus: &[usize]) -> Vec<String> {
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
    for &cpu in cpus {
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
        invalidations(true, false, "REAL slab", &[], true, &[])
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
        let v = invalidations(false, false, "REAL slab", &[], true, &[]);
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
        let v = invalidations(false, true, "SYNTHETIC bytes", &throttles, false, &["CPU"]);
        assert_eq!(v.len(), 5, "every failing gate must contribute: {v:?}");
        assert!(v[0].contains("did not compute the same thing"), "{v:?}");
    }

    /// The second defect: a run on invented bytes is not a run on a real
    /// expert slab, and the provenance travels with the reason.
    #[test]
    fn synthetic_weights_invalidate_the_verdict() {
        let v = invalidations(
            true,
            true,
            "SYNTHETIC q4_k bytes (768x2048)",
            &[],
            true,
            &[],
        );
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
        let v = invalidations(true, false, "REAL slab", &throttles, true, &[]);
        assert_eq!(v.len(), 1, "got {v:?}");
        assert!(v[0].contains("no_turbo=1"), "{}", v[0]);
        assert!(v[0].contains("scaling_max_freq"), "{}", v[0]);
    }

    /// A minimum still falling at the cap is a fact about the sample size.
    #[test]
    fn a_sample_that_never_converged_invalidates_the_verdict() {
        let v = invalidations(true, false, "REAL slab", &[], false, &[]);
        assert_eq!(v.len(), 1, "got {v:?}");
        assert!(v[0].contains("not converged"), "{}", v[0]);
    }

    /// And a minimum rising through the run is not a steady state, whichever
    /// arm it was.
    #[test]
    fn a_drifting_arm_invalidates_the_verdict() {
        let v = invalidations(
            true,
            false,
            "REAL slab",
            &[],
            true,
            &["GPU  (weights resident)"],
        );
        assert_eq!(v.len(), 1, "got {v:?}");
        assert!(v[0].contains("drifting"), "{}", v[0]);
        assert!(
            v[0].contains("GPU  (weights resident)"),
            "the arm must be named: {}",
            v[0]
        );
    }

    /// A minimum that keeps falling is a unit that was still speeding up, and
    /// the sampler must not call that a steady state.
    #[test]
    fn a_falling_minimum_never_converges() {
        let mut c = Convergence::new("test");
        let mut best = 1.0;
        for round in 1..=10 {
            // Each round beats the last by well over `MIN_IMPROVEMENT`.
            best *= 0.9;
            c.observe(round, &[best, best * 1.1]);
            assert!(!c.converged(), "converged in round {round} while improving");
        }
        assert_eq!(c.samples, 20);
    }

    /// A minimum that stops improving converges after exactly `PATIENCE`
    /// stale rounds, and noise inside `MIN_IMPROVEMENT` does not reset it.
    #[test]
    fn a_settled_minimum_converges_after_patience_rounds() {
        let mut c = Convergence::new("test");
        c.observe(1, &[1.0]);
        for round in 2..=(1 + PATIENCE) {
            // Better by 0.1%, which is under `MIN_IMPROVEMENT` and must not
            // count as progress.
            c.observe(round, &[1.0 - 0.001 * (round - 1) as f64]);
        }
        assert!(
            c.converged(),
            "should have converged after {PATIENCE} rounds"
        );
        assert_eq!(c.last_improved, 1);
    }

    /// The defect this drift tracking exists for: an arm that gets
    /// monotonically WORSE never improves its running minimum either, so a
    /// sampler that only counts non-improvement would call the drift a
    /// steady state and print CONVERGED.
    #[test]
    fn a_rising_minimum_is_drift_and_never_converges() {
        let mut c = Convergence::new("test");
        let mut worst = 1.0;
        c.observe(1, &[worst]);
        for round in 2..=10 {
            // Each round is materially worse than the last, as if a build
            // had started on the box after round 1.
            worst *= 1.1;
            c.observe(round, &[worst]);
            assert!(
                !c.converged(),
                "converged in round {round} on a monotonically degrading arm"
            );
        }
        assert!(
            c.drifting(),
            "a monotonically rising minimum must report drift"
        );
        assert!(
            c.report().contains("DRIFTING"),
            "the report must say so: {}",
            c.report()
        );
    }

    /// Drift is reported even when it stops: an arm that degrades for
    /// `PATIENCE` rounds and then holds still has not delivered a sample of
    /// one steady state, and the tail must not launder the head.
    #[test]
    fn drift_that_ends_is_still_reported() {
        let mut c = Convergence::new("test");
        let mut worst = 1.0;
        c.observe(1, &[worst]);
        for round in 2..=(1 + PATIENCE) {
            worst *= 1.1;
            c.observe(round, &[worst]);
        }
        assert!(c.drifting());
        // Now hold flat for long enough that staleness alone would have
        // called it converged.
        for round in (2 + PATIENCE)..=(2 + 3 * PATIENCE) {
            c.observe(round, &[worst]);
        }
        assert!(
            c.stale >= PATIENCE,
            "the flat tail must accumulate patience"
        );
        assert!(
            !c.converged(),
            "a run containing sustained drift is not a converged sample"
        );
    }

    /// Ordinary round-to-round noise inside `MIN_IMPROVEMENT` is not drift,
    /// or every honest run would be invalidated.
    #[test]
    fn noise_is_not_drift() {
        let mut c = Convergence::new("test");
        c.observe(1, &[1.0]);
        for round in 2..=12 {
            // Alternating +/-0.2%, well inside `MIN_IMPROVEMENT`.
            let v = if round % 2 == 0 { 1.002 } else { 0.998 };
            c.observe(round, &[v]);
        }
        assert!(!c.drifting(), "sub-threshold noise must not read as drift");
        assert!(c.converged(), "a settled noisy arm must still converge");
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
