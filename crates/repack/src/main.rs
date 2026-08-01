//! ramvamp-repack CLI.
//!
//! `inspect` parses a GGUF source — a local file or a Hugging Face repo
//! over ranged HTTP (header only, never the data section) — builds the
//! repack plan, and prints a human-readable report of what an install
//! would look like. `install` executes the plan into a `.rvmp` directory
//! (streaming, resumable, hash-verified) and then fetches the pinned
//! tokenizer files into it; `fetch-tokenizer` amends an existing completed
//! install with those files; `verify-install` re-checks an installed
//! directory; `discard-partial` deletes an abandoned staging directory.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::{Args, Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use ramvamp_core::format;
use ramvamp_repack::gguf::GgufFile;
use ramvamp_repack::install::{self, InstallOptions, SourcePin};
use ramvamp_repack::plan::RepackPlan;
use ramvamp_repack::source::{LocalFile, RangeRead, RemoteFile};
use ramvamp_repack::tokenizer_fetch::{self, TokenizerSource};

/// Frozen v0 model pin (docs/architecture.md "Model pin", audited
/// 2026-08-01): the exact revision `install` defaults to.
const PIN_REPO: &str = "bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF";
const PIN_REVISION: &str = "6c6e8692f43e4ca663f7ece8229a1361090d3a4c";
const PIN_FILE: &str = "Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf";

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
    /// Stream a GGUF source into an installed .rvmp directory.
    Install(InstallArgs),
    /// Fetch the pinned tokenizer files into an existing completed
    /// install and record them in its manifest.
    FetchTokenizer(FetchTokenizerArgs),
    /// Re-verify an installed .rvmp directory (sizes, hashes, layout).
    VerifyInstall(VerifyInstallArgs),
    /// Delete the .partial staging directory of an interrupted install.
    DiscardPartial(DiscardPartialArgs),
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

#[derive(Args)]
struct InstallArgs {
    /// Destination install directory, e.g. /models/qwen3.rvmp.
    #[arg(long, value_name = "DIR")]
    output: PathBuf,
    /// Install from a local GGUF file instead of the Hugging Face pin.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["repo", "revision", "file"]
    )]
    local: Option<PathBuf>,
    /// Hugging Face repo (defaults to the frozen v0 pin).
    #[arg(long, default_value = PIN_REPO)]
    repo: String,
    /// Repo revision: commit hash of the frozen pin by default.
    #[arg(long, default_value = PIN_REVISION)]
    revision: String,
    /// File name within the repo.
    #[arg(long, default_value = PIN_FILE)]
    file: String,
    /// Transfer window size in MiB (one HTTP range request per window).
    #[arg(long, value_name = "N", default_value_t = 32,
          value_parser = clap::value_parser!(u64).range(1..=1024))]
    window_mib: u64,
    /// Continue a matching interrupted install from its last durable
    /// window.
    #[arg(long, conflicts_with = "overwrite")]
    resume: bool,
    /// Replace an existing install or partial install from scratch.
    #[arg(long)]
    overwrite: bool,
    /// Skip the post-promotion hash self-check.
    #[arg(long)]
    skip_verify: bool,
    /// Skip fetching the tokenizer files after the model install (add
    /// them later with `fetch-tokenizer`).
    #[arg(long)]
    skip_tokenizer: bool,
    /// Copy the tokenizer files from a local directory instead of
    /// downloading the pinned tokenizer revision (offline installs).
    #[arg(long, value_name = "PATH", conflicts_with = "skip_tokenizer")]
    tokenizer_dir: Option<PathBuf>,
}

#[derive(Args)]
struct FetchTokenizerArgs {
    /// Installed .rvmp directory to amend (must be a complete install;
    /// partials are refused).
    #[arg(long, value_name = "DIR")]
    output: PathBuf,
    /// Copy the tokenizer files from a local directory instead of
    /// downloading the pinned tokenizer revision.
    #[arg(long, value_name = "PATH")]
    tokenizer_dir: Option<PathBuf>,
}

#[derive(Args)]
struct VerifyInstallArgs {
    /// Installed .rvmp directory to verify.
    #[arg(long, value_name = "DIR")]
    input: PathBuf,
}

#[derive(Args)]
struct DiscardPartialArgs {
    /// Install target whose .partial staging directory should be deleted.
    #[arg(long, value_name = "DIR")]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    // Honor RUST_LOG when set, but default to `info` so progress lines
    // are visible (`from_default_env` alone would default to ERROR and
    // suppress them).
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect(args) => inspect(&args),
        Command::Install(args) => install_cmd(&args),
        Command::FetchTokenizer(args) => fetch_tokenizer_cmd(&args),
        Command::VerifyInstall(args) => verify_install_cmd(&args),
        Command::DiscardPartial(args) => discard_partial_cmd(&args),
    }
}

/// Open the install source and derive its pin (identity + manifest
/// `source` fields).
fn open_install_source(args: &InstallArgs) -> anyhow::Result<(Box<dyn RangeRead>, SourcePin)> {
    match &args.local {
        Some(path) => {
            let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.clone());
            let file = canonical
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "local.gguf".to_owned());
            let source = LocalFile::open(path)
                .with_context(|| format!("open local file {}", path.display()))?;
            Ok((
                Box::new(source),
                SourcePin {
                    url: canonical.display().to_string(),
                    hf_repo: "(local)".to_owned(),
                    revision: "(local)".to_owned(),
                    file,
                },
            ))
        }
        None => {
            let url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                args.repo, args.revision, args.file
            );
            let source = RemoteFile::open(&url).with_context(|| format!("open remote {url}"))?;
            Ok((
                Box::new(source),
                SourcePin {
                    url,
                    hf_repo: args.repo.clone(),
                    revision: args.revision.clone(),
                    file: args.file.clone(),
                },
            ))
        }
    }
}

fn install_cmd(args: &InstallArgs) -> anyhow::Result<()> {
    let (source, pin) = open_install_source(args)?;
    println!("source: {}", pin.url);
    println!("source size: {}", human_bytes(source.len()));

    let gguf = GgufFile::parse(source.as_ref()).context("parse GGUF header")?;
    let plan = RepackPlan::from_gguf(&gguf).context("build repack plan")?;
    println!(
        "plan: {} layers, {} experts, {} copy ops",
        plan.arch.n_layers,
        plan.arch.n_experts,
        plan.copy_ops.len()
    );
    println!(
        "download (tensor bytes): {}",
        human_bytes(plan.totals.download_bytes)
    );
    println!(
        "installed (data files):  {}",
        human_bytes(plan.totals.installed_bytes)
    );
    println!(
        "window: {} MiB   target: {}",
        args.window_mib,
        args.output.display()
    );

    let opts = InstallOptions {
        window_bytes: args.window_mib * 1024 * 1024,
        resume: args.resume,
        overwrite: args.overwrite,
        skip_verify: args.skip_verify,
        ..InstallOptions::default()
    };
    let report = install::install(source.as_ref(), &plan, &pin, &args.output, &opts)
        .context("install failed")?;

    println!();
    println!("installed: {}", report.final_dir.display());
    if report.windows_resumed > 0 {
        println!(
            "windows: {} total, {} resumed from a previous run",
            report.windows_total, report.windows_resumed
        );
    } else {
        println!("windows: {}", report.windows_total);
    }
    println!(
        "bytes copied this run: {}",
        human_bytes(report.bytes_copied)
    );
    println!(
        "source sha256 (digest-of-digests over {} windows): {}",
        report.windows_total, report.source_sha256
    );
    println!(
        "note: source.sha256 is a per-window digest-of-digests, not the plain \
         file hash (see the install module docs)"
    );
    println!(
        "self-check: {}",
        if report.verified {
            "PASS"
        } else {
            "skipped (--skip-verify)"
        }
    );

    if args.skip_tokenizer {
        println!(
            "tokenizer: skipped (--skip-tokenizer); add it later with \
             `ramvamp-repack fetch-tokenizer --output {}`",
            report.final_dir.display()
        );
    } else {
        fetch_tokenizer_into(&report.final_dir, args.tokenizer_dir.as_deref());
    }
    Ok(())
}

/// Fetch the tokenizer into a just-promoted install. Never fails the
/// caller: the model install is already complete and durable, so a
/// tokenizer failure only prints a warning pointing at the
/// `fetch-tokenizer` subcommand.
fn fetch_tokenizer_into(dir: &Path, tokenizer_dir: Option<&Path>) {
    match tokenizer_fetch::fetch_and_amend(dir, tokenizer_source(tokenizer_dir)) {
        Ok(entries) => {
            println!(
                "tokenizer: {} files recorded in the manifest",
                entries.len()
            );
            for (name, entry) in &entries {
                println!("  {name}: {}", human_bytes(entry.size));
            }
        }
        Err(e) => {
            let e = anyhow::Error::from(e);
            eprintln!("warning: tokenizer fetch failed: {e:#}");
            eprintln!(
                "warning: the model install itself succeeded; run \
                 `ramvamp-repack fetch-tokenizer --output {}` to add the \
                 tokenizer later",
                dir.display()
            );
        }
    }
}

/// Map the optional `--tokenizer-dir` flag to a tokenizer source.
fn tokenizer_source(dir: Option<&Path>) -> TokenizerSource {
    match dir {
        Some(path) => TokenizerSource::LocalDir(path.to_path_buf()),
        None => TokenizerSource::Pinned,
    }
}

fn fetch_tokenizer_cmd(args: &FetchTokenizerArgs) -> anyhow::Result<()> {
    let entries = tokenizer_fetch::fetch_and_amend(
        &args.output,
        tokenizer_source(args.tokenizer_dir.as_deref()),
    )
    .context("fetch tokenizer")?;
    for (name, entry) in &entries {
        println!(
            "{name}: {}  sha256 {}",
            human_bytes(entry.size),
            entry.sha256
        );
    }
    println!(
        "amended: {} ({} tokenizer entries verified; run verify-install \
         for a full re-check)",
        args.output.display(),
        entries.len()
    );
    Ok(())
}

fn verify_install_cmd(args: &VerifyInstallArgs) -> anyhow::Result<()> {
    match check_install(&args.input) {
        Ok((files, bytes)) => {
            println!(
                "PASS: {} ({files} files, {} verified)",
                args.input.display(),
                human_bytes(bytes)
            );
            Ok(())
        }
        Err(e) => {
            println!("FAIL: {}: {e:#}", args.input.display());
            bail!("verification failed");
        }
    }
}

/// Manifest validation, per-file size+hash verification, and layout
/// cross-check. Returns (file count, verified bytes).
fn check_install(dir: &Path) -> anyhow::Result<(usize, u64)> {
    let manifest = format::load_manifest(dir).context("load manifest.json")?;
    manifest.validate().context("validate manifest")?;
    let layout = format::load_layout(dir).context("load experts/layout.json")?;
    layout
        .validate_against(&manifest)
        .context("cross-check layout against manifest")?;
    format::verify_files(dir, &manifest).context("verify file hashes")?;
    let bytes = manifest.files.values().map(|f| f.size).sum();
    Ok((manifest.files.len(), bytes))
}

fn discard_partial_cmd(args: &DiscardPartialArgs) -> anyhow::Result<()> {
    let removed = install::discard_partial(&args.output).context("discard partial install")?;
    println!("removed: {}", removed.display());
    Ok(())
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
