//! ramvamp: the user-facing CLI.
//!
//! - `tokenize`: tokenizer + vendored chat template smoke test.
//! - `generate`: run the forward pass end to end and stream text to
//!   stdout (timing footer on stderr).
//! - `logits`: raw-encode a prompt, run one forward pass, and print the
//!   top-N next-token logits as JSON — the llama.cpp comparison hook
//!   consumed by `scripts/compare_llamacpp.py`.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, bail};
use clap::{ArgGroup, Parser, Subcommand};
use ramvamp_core::generate::{GenerateParams, StopReason, generate};
use ramvamp_core::model::{ForwardState, LoadOptions, Model, forward_token};
use ramvamp_core::tokenizer::{ChatMessage, RvmpTokenizer};

/// v0 scope cap: single sequence, 4K context (`docs/architecture.md`).
const CONTEXT_CAP: usize = 4096;

#[derive(Parser)]
#[command(
    name = "ramvamp",
    version,
    about = "Run 26-30B fine-grained MoE models in ~3 GB of RAM by streaming experts from NVMe"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Tokenizer smoke test: encode a prompt or a chat transcript with an
    /// installed model's tokenizer and verify the decode round-trip.
    #[command(group(
        ArgGroup::new("input")
            .required(true)
            .args(["prompt", "messages_file"])
    ))]
    Tokenize {
        /// Installed model directory (the .rvmp dir).
        #[arg(long, value_name = "DIR")]
        model: PathBuf,

        /// Raw text to encode directly (no chat template).
        #[arg(long, value_name = "TEXT")]
        prompt: Option<String>,

        /// JSON file with a conversation to render through the chat
        /// template (with generation prompt) and encode:
        /// [{"role": "user", "content": "..."}, ...]
        #[arg(long, value_name = "FILE")]
        messages_file: Option<PathBuf>,
    },

    /// Generate text: raw completion from --prompt, or chat completion
    /// from a --messages-file rendered through the chat template. Streams
    /// to stdout; timing goes to stderr.
    #[command(group(
        ArgGroup::new("input")
            .required(true)
            .args(["prompt", "messages_file"])
    ))]
    Generate {
        /// Installed model directory (the .rvmp dir).
        #[arg(long, value_name = "DIR")]
        model: PathBuf,

        /// Raw text prompt (no chat template).
        #[arg(long, value_name = "TEXT")]
        prompt: Option<String>,

        /// JSON conversation file, rendered with the generation prompt.
        #[arg(long, value_name = "FILE")]
        messages_file: Option<PathBuf>,

        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 128)]
        max_new: usize,

        /// Deterministic argmax decoding (the validation mode).
        #[arg(long)]
        greedy: bool,

        /// Sampling temperature (default: the checkpoint's).
        #[arg(long)]
        temperature: Option<f32>,

        /// Top-k cutoff (default: the checkpoint's).
        #[arg(long)]
        top_k: Option<u32>,

        /// Top-p nucleus mass (default: the checkpoint's).
        #[arg(long)]
        top_p: Option<f32>,

        /// PRNG seed for sampled decoding.
        #[arg(long)]
        seed: Option<u64>,

        /// Skip SHA-256 integrity checks (fast dev loads).
        #[arg(long)]
        skip_hashes: bool,
    },

    /// Print the top-N next-token logits for a raw prompt as JSON (the
    /// llama.cpp logit-comparison hook).
    Logits {
        /// Installed model directory (the .rvmp dir).
        #[arg(long, value_name = "DIR")]
        model: PathBuf,

        /// Raw text prompt (no chat template).
        #[arg(long, value_name = "TEXT")]
        prompt: String,

        /// How many top tokens to print.
        #[arg(long, default_value_t = 20)]
        top: usize,

        /// Skip SHA-256 integrity checks (fast dev loads).
        #[arg(long)]
        skip_hashes: bool,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    match Cli::parse().command {
        Command::Tokenize {
            model,
            prompt,
            messages_file,
        } => tokenize(&model, prompt, messages_file),
        Command::Generate {
            model,
            prompt,
            messages_file,
            max_new,
            greedy,
            temperature,
            top_k,
            top_p,
            seed,
            skip_hashes,
        } => run_generate(
            &model,
            prompt,
            messages_file,
            max_new,
            greedy,
            temperature,
            top_k,
            top_p,
            seed,
            skip_hashes,
        ),
        Command::Logits {
            model,
            prompt,
            top,
            skip_hashes,
        } => run_logits(&model, &prompt, top, skip_hashes),
    }
}

/// Encode the input, print what the model would actually see, and check
/// that decoding the ids reproduces the exact input string.
fn tokenize(
    model: &Path,
    prompt: Option<String>,
    messages_file: Option<PathBuf>,
) -> anyhow::Result<()> {
    let tokenizer = load_tokenizer(model)?;
    let (rendered, ids) = encode_input(&tokenizer, prompt, messages_file)?;

    println!("rendered: {rendered:?}");
    println!("tokens: {}", ids.len());
    let preview: Vec<u32> = ids.iter().copied().take(8).collect();
    println!("first ids: {preview:?}");
    let tail_start = ids.len().saturating_sub(8);
    println!("last ids: {:?}", &ids[tail_start..]);
    println!(
        "stop tokens: {:?}  sampling defaults: {:?}",
        tokenizer.stop_tokens(),
        tokenizer.sampling_defaults()
    );

    let decoded = tokenizer.decode(&ids, false)?;
    if decoded == rendered {
        println!("round-trip: ok");
        Ok(())
    } else {
        bail!("round-trip mismatch: decoded {decoded:?}");
    }
}

/// Generate and stream a completion.
#[allow(clippy::too_many_arguments)]
fn run_generate(
    model_dir: &Path,
    prompt: Option<String>,
    messages_file: Option<PathBuf>,
    max_new: usize,
    greedy: bool,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    skip_hashes: bool,
) -> anyhow::Result<()> {
    let tokenizer = load_tokenizer(model_dir)?;
    let (_, prompt_ids) = encode_input(&tokenizer, prompt, messages_file)?;
    if prompt_ids.is_empty() {
        bail!("prompt encodes to zero tokens");
    }
    if prompt_ids.len() + max_new > CONTEXT_CAP {
        bail!(
            "prompt ({}) + max-new ({max_new}) exceeds the v0 context cap of {CONTEXT_CAP}",
            prompt_ids.len()
        );
    }

    let load_start = Instant::now();
    let model = Model::load(model_dir, LoadOptions { skip_hashes })
        .with_context(|| format!("loading model from {}", model_dir.display()))?;
    let mut state = ForwardState::new(&model, CONTEXT_CAP)?;
    eprintln!(
        "model loaded in {:.2}s ({} prompt tokens)",
        load_start.elapsed().as_secs_f64(),
        prompt_ids.len()
    );

    let mut params = GenerateParams::from_defaults(tokenizer.sampling_defaults());
    params.max_new = max_new;
    params.greedy = greedy;
    if let Some(t) = temperature {
        params.temperature = t;
    }
    if let Some(k) = top_k {
        params.top_k = Some(k);
    }
    if let Some(p) = top_p {
        params.top_p = p;
    }
    if let Some(s) = seed {
        params.seed = s;
    }

    let mut stdout = std::io::stdout();
    let stats = generate(
        &model,
        &mut state,
        &tokenizer,
        &prompt_ids,
        &params,
        |_, text| {
            let _ = stdout.write_all(text.as_bytes());
            let _ = stdout.flush();
        },
    )?;
    println!();

    let prefill_s = stats.prefill.as_secs_f64();
    let decode_s = stats.decode.as_secs_f64();
    let decode_rate = if decode_s > 0.0 {
        stats.generated as f64 / decode_s
    } else {
        0.0
    };
    let stop = match stats.stop {
        StopReason::StopToken(id) => format!("stop token {id}"),
        StopReason::MaxNew => "max-new".to_owned(),
    };
    eprintln!(
        "prefill: {} tokens in {prefill_s:.2}s ({:.2} tok/s); decode: {} tokens in \
         {decode_s:.2}s ({decode_rate:.2} tok/s); stopped by {stop}",
        stats.prompt_tokens,
        if prefill_s > 0.0 {
            stats.prompt_tokens as f64 / prefill_s
        } else {
            0.0
        },
        stats.generated,
    );
    Ok(())
}

/// One forward pass over a raw prompt; top-N logits as JSON on stdout.
fn run_logits(model_dir: &Path, prompt: &str, top: usize, skip_hashes: bool) -> anyhow::Result<()> {
    let tokenizer = load_tokenizer(model_dir)?;
    let ids = tokenizer.encode(prompt)?;
    if ids.is_empty() {
        bail!("prompt encodes to zero tokens");
    }
    if ids.len() > CONTEXT_CAP {
        bail!(
            "prompt ({}) exceeds the v0 context cap of {CONTEXT_CAP}",
            ids.len()
        );
    }

    let model = Model::load(model_dir, LoadOptions { skip_hashes })
        .with_context(|| format!("loading model from {}", model_dir.display()))?;
    let mut state = ForwardState::new(&model, CONTEXT_CAP)?;

    let last = ids.len() - 1;
    for (pos, &id) in ids[..last].iter().enumerate() {
        forward_token(&model, &mut state, id, pos, false)?;
    }
    let logits = forward_token(&model, &mut state, ids[last], last, true)?
        .context("final forward pass returned no logits")?;

    // log-softmax normalizer in f64: lse = max + ln(sum(exp(l - max))).
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits
        .iter()
        .map(|&l| (f64::from(l) - f64::from(max)).exp())
        .sum();
    let lse = f64::from(max) + sum.ln();

    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    order.truncate(top);

    let top_entries: Vec<serde_json::Value> = order
        .iter()
        .map(|&id| {
            let logit = logits[id as usize];
            serde_json::json!({
                "token_id": id,
                "logit": logit,
                "logprob": f64::from(logit) - lse,
                "text": tokenizer.decode(&[id], false).unwrap_or_default(),
            })
        })
        .collect();
    let out = serde_json::json!({
        "prompt": prompt,
        "prompt_tokens": ids.len(),
        "prompt_ids": ids,
        "top": top_entries,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn load_tokenizer(model: &Path) -> anyhow::Result<RvmpTokenizer> {
    RvmpTokenizer::load(model)
        .with_context(|| format!("loading tokenizer from {}", model.display()))
}

/// Encode either a raw prompt (plain encode, no template) or a chat
/// transcript (rendered with the generation prompt). Returns the rendered
/// text and its token ids.
fn encode_input(
    tokenizer: &RvmpTokenizer,
    prompt: Option<String>,
    messages_file: Option<PathBuf>,
) -> anyhow::Result<(String, Vec<u32>)> {
    match (prompt, messages_file) {
        (Some(text), None) => {
            let ids = tokenizer.encode(&text)?;
            Ok((text, ids))
        }
        (None, Some(path)) => {
            let data = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            let messages: Vec<ChatMessage> = serde_json::from_str(&data)
                .with_context(|| format!("parsing messages from {}", path.display()))?;
            let rendered = tokenizer.render_chat(&messages, true);
            let ids = tokenizer.encode_chat(&messages, true)?;
            Ok((rendered, ids))
        }
        // clap's arg group guarantees exactly one input.
        _ => unreachable!("clap enforces exactly one of --prompt/--messages-file"),
    }
}
