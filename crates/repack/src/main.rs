//! ramvamp-repack CLI.
//!
//! `inspect` parses a GGUF source — a local file or a Hugging Face repo
//! over ranged HTTP (header only, never the data section) — builds the
//! repack plan, and prints a human-readable report of what an install
//! would look like.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::{Context, bail};
use clap::{Args, Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use ramvamp_repack::gguf::GgufFile;
use ramvamp_repack::plan::RepackPlan;
use ramvamp_repack::source::{LocalFile, RangeRead, RemoteFile};

/// Tensor types the audited Q4_K_M qwen3moe pin contains (see
/// docs/architecture.md, audited 2026-08-01); anything else is flagged in
/// the report.
const EXPECTED_TYPES: &[&str] = &["q4_k", "q5_k", "q6_k", "q8_0", "f32"];

#[derive(Parser)]
#[command(
    name = "ramvamp-repack",
    version,
    about = "Streaming GGUF-to-.rvmp model installer"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse a GGUF header, build the repack plan, and print a report.
    Inspect(InspectArgs),
}

#[derive(Args)]
struct InspectArgs {
    /// Path to a local GGUF file.
    #[arg(long, value_name = "PATH", conflicts_with = "remote")]
    local: Option<PathBuf>,
    /// Inspect a Hugging Face GGUF over HTTP range requests.
    #[arg(long)]
    remote: bool,
    /// Hugging Face repo (with --remote).
    #[arg(
        long,
        default_value = "bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF"
    )]
    repo: String,
    /// Repo revision: branch, tag, or commit (with --remote).
    #[arg(long, default_value = "main")]
    revision: String,
    /// File name within the repo (with --remote).
    #[arg(long, default_value = "Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")]
    file: String,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect(args) => inspect(&args),
    }
}

fn inspect(args: &InspectArgs) -> anyhow::Result<()> {
    let source: Box<dyn RangeRead> = match (&args.local, args.remote) {
        (Some(path), false) => {
            println!("source: {}", path.display());
            Box::new(
                LocalFile::open(path)
                    .with_context(|| format!("open local file {}", path.display()))?,
            )
        }
        (None, true) => {
            let url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                args.repo, args.revision, args.file
            );
            println!("source: {url}");
            Box::new(RemoteFile::open(&url).with_context(|| format!("open remote {url}"))?)
        }
        _ => bail!("pass exactly one of --local <path> or --remote"),
    };
    println!("source size: {}", human_bytes(source.len()));

    let gguf = GgufFile::parse(source.as_ref()).context("parse GGUF header")?;
    let plan = RepackPlan::from_gguf(&gguf).context("build repack plan")?;
    print_report(&gguf, &plan);
    Ok(())
}

fn print_report(gguf: &GgufFile, plan: &RepackPlan) {
    let a = &plan.arch;
    println!(
        "gguf header (metadata + tensor index): {}",
        human_bytes(gguf.data_section_offset())
    );
    println!();
    println!("== architecture ==");
    println!("arch: qwen3moe (GGUF v{})", gguf.version());
    println!(
        "layers: {}   experts: {} (top-{})   shared expert: {}",
        a.n_layers, a.n_experts, a.top_k, a.shared_expert
    );
    println!(
        "hidden: {}   moe_intermediate: {}   vocab: {}",
        a.hidden, a.moe_intermediate, a.vocab
    );
    println!(
        "heads: {} q / {} kv   head_dim: {}   sliding_window: {}",
        a.n_heads,
        a.n_kv_heads,
        a.head_dim,
        a.sliding_window
            .map_or_else(|| "none".to_owned(), |w| w.to_string())
    );
    println!(
        "rope_theta: {}   rms_eps: {}   context_length: {}",
        a.rope_theta,
        a.rms_eps,
        plan.context_length()
    );
    println!(
        "norm_topk_prob: {}   tie_embeddings: {}",
        a.norm_topk_prob, a.tie_embeddings
    );

    println!();
    println!("== tensor types ==");
    let total_tensors: u64 = plan.totals.tensor_type_counts.values().sum();
    println!("{total_tensors} tensors");
    for (ty, count) in &plan.totals.tensor_type_counts {
        println!("  {ty:<6} {count:>5}");
    }

    println!();
    println!("== quantization ==");
    let q6k = &plan.totals.q6k_down_layers;
    if q6k.is_empty() {
        println!("ffn_down_exps: q4_k on all layers");
    } else {
        println!(
            "ffn_down_exps is not q4_k on {} of {} layers: {}",
            q6k.len(),
            a.n_layers,
            compress_layers(q6k)
        );
    }

    println!();
    println!("== planned install ==");
    println!(
        "common.bin: {} ({} tensors)",
        human_bytes(plan.common_size),
        plan.common_tensors.len()
    );
    // Group layers into classes by identical stride.
    let mut classes: std::collections::BTreeMap<u64, Vec<u32>> = std::collections::BTreeMap::new();
    for (n, layer) in plan.layout.layers.iter().enumerate() {
        classes.entry(layer.stride).or_default().push(n as u32);
    }
    for (stride, layers) in &classes {
        println!(
            "expert stride {} ({}): {} layers [{}]",
            stride,
            human_bytes(*stride),
            layers.len(),
            compress_layers(layers)
        );
    }
    println!("copy ops: {}", plan.copy_ops.len());
    println!(
        "download (tensor bytes): {}",
        human_bytes(plan.totals.download_bytes)
    );
    println!(
        "installed (data files):  {}",
        human_bytes(plan.totals.installed_bytes)
    );

    let unexpected: Vec<(&String, &String)> = plan
        .quant
        .tensor_types
        .iter()
        .filter(|(_, ty)| !EXPECTED_TYPES.contains(&ty.as_str()))
        .collect();
    if !unexpected.is_empty() {
        println!();
        println!("== warnings ==");
        for (name, ty) in unexpected {
            println!("unexpected tensor type {ty}: {name}");
        }
    }
}

/// Human-readable byte count alongside the exact number.
fn human_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    let b = bytes as f64;
    if b >= KIB * KIB * KIB {
        format!("{bytes} B ({:.2} GiB)", b / (KIB * KIB * KIB))
    } else if b >= KIB * KIB {
        format!("{bytes} B ({:.2} MiB)", b / (KIB * KIB))
    } else if b >= KIB {
        format!("{bytes} B ({:.2} KiB)", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

/// Compress a sorted layer list into `0-5,7,42-47` form.
fn compress_layers(layers: &[u32]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < layers.len() {
        let start = layers[i];
        let mut end = start;
        while i + 1 < layers.len() && layers[i + 1] == end + 1 {
            i += 1;
            end = layers[i];
        }
        if !out.is_empty() {
            out.push(',');
        }
        if start == end {
            let _ = write!(out, "{start}");
        } else {
            let _ = write!(out, "{start}-{end}");
        }
        i += 1;
    }
    out
}
