//! ramvamp: the user-facing CLI.
//!
//! Chat and raw-completion modes land once the core runtime can execute a
//! forward pass. Until then, `tokenize` is the maintainer's smoke test for
//! the tokenizer + vendored chat template against an installed model.

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::{ArgGroup, Parser, Subcommand};
use ramvamp_core::tokenizer::{ChatMessage, RvmpTokenizer};

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
    }
}

/// Encode the input, print what the model would actually see, and check
/// that decoding the ids reproduces the exact input string.
fn tokenize(
    model: &Path,
    prompt: Option<String>,
    messages_file: Option<PathBuf>,
) -> anyhow::Result<()> {
    let tokenizer = RvmpTokenizer::load(model)
        .with_context(|| format!("loading tokenizer from {}", model.display()))?;

    let (rendered, ids) = match (prompt, messages_file) {
        (Some(text), None) => {
            let ids = tokenizer.encode(&text)?;
            (text, ids)
        }
        (None, Some(path)) => {
            let data = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            let messages: Vec<ChatMessage> = serde_json::from_str(&data)
                .with_context(|| format!("parsing messages from {}", path.display()))?;
            let rendered = tokenizer.render_chat(&messages, true);
            let ids = tokenizer.encode_chat(&messages, true)?;
            (rendered, ids)
        }
        // clap's arg group guarantees exactly one input.
        _ => unreachable!("clap enforces exactly one of --prompt/--messages-file"),
    };

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
