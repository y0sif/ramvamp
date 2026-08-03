//! ramvamp: the user-facing CLI.
//!
//! - `tokenize`: tokenizer + vendored chat template smoke test.
//! - `generate`: run the forward pass end to end and stream text to
//!   stdout (timing footer on stderr).
//! - `logits`: raw-encode a prompt, run one forward pass, and print the
//!   top-N next-token logits as JSON — the llama.cpp comparison hook
//!   consumed by `scripts/compare_llamacpp.py`.
//!
//! Both `generate` and `logits` can dump the router's per-layer expert
//! selection with `--trace-experts <PATH>`; see [`TraceWriter`] for the
//! file format and `scripts/lfu_sim.py` for the consumer.

use std::io::{Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, bail};
use clap::{ArgGroup, Parser, Subcommand};
use ramvamp_core::generate::{GenerateParams, StopReason, TracePhase, generate, generate_traced};
use ramvamp_core::model::{ForwardState, LoadOptions, Model, forward_token, forward_token_traced};
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

        /// Dump the per-layer routed expert ids for every prefill and
        /// decode position to a binary trace file (see the TraceWriter
        /// docs; read by scripts/lfu_sim.py).
        #[arg(long, value_name = "FILE")]
        trace_experts: Option<PathBuf>,

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

        /// Dump the per-layer routed expert ids for every prompt position
        /// (all prefill) to a binary trace file.
        #[arg(long, value_name = "FILE")]
        trace_experts: Option<PathBuf>,

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
            trace_experts,
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
            trace_experts.as_deref(),
            skip_hashes,
        ),
        Command::Logits {
            model,
            prompt,
            top,
            trace_experts,
            skip_hashes,
        } => run_logits(&model, &prompt, top, trace_experts.as_deref(), skip_hashes),
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

/// Magic of the `--trace-experts` binary routing trace.
const TRACE_MAGIC: &[u8; 8] = b"RVMPTRC1";
/// Version of the trace format this build writes.
const TRACE_VERSION: u32 = 1;
/// Byte offset of the header's `n_records` field, patched on close.
const TRACE_N_RECORDS_OFFSET: u64 = 24;

/// Writer for the binary expert-routing trace (`--trace-experts`).
///
/// Format `RVMPTRC1`, every integer little-endian:
///
/// ```text
/// header, 28 bytes
///   magic      [u8; 8]  b"RVMPTRC1"
///   version    u32      1
///   n_layers   u32      layers per record
///   n_experts  u32      routed experts per layer (the id space)
///   top_k      u32      routed experts per layer per token
///   n_records  u32      complete records; written 0, patched on close
/// record, 8 + n_layers * top_k * 4 bytes, repeated n_records times
///   phase      u8       0 = prefill, 1 = decode
///   _pad       [u8; 3]  zero
///   position   u32      sequence position of the token
///   experts    [u32; n_layers * top_k]
///                       layer-major, layer 0 first; within a layer the
///                       top_k ids in routed order, i.e. descending
///                       router probability
/// ```
///
/// One record per token, in stream order (a generation writes its prefill
/// records then its decode records). The phase byte exists because the
/// design bypasses the expert cache during prefill
/// (`docs/architecture.md`, "Prefill"), so a consumer must be able to
/// filter. The reader is `scripts/lfu_sim.py`, which documents the same
/// layout.
struct TraceWriter {
    out: std::io::BufWriter<std::fs::File>,
    n_layers: u32,
    top_k: u32,
    /// Expert ids written into the record currently in progress.
    filled: u32,
    /// Records completed so far.
    records: u32,
}

impl TraceWriter {
    /// Create `path` and write the header (with a placeholder count).
    fn create(path: &Path, n_layers: u32, n_experts: u32, top_k: u32) -> anyhow::Result<Self> {
        if n_layers == 0 || top_k == 0 {
            bail!("trace: n_layers ({n_layers}) and top_k ({top_k}) must be nonzero");
        }
        let file = std::fs::File::create(path)
            .with_context(|| format!("creating expert trace {}", path.display()))?;
        let mut out = std::io::BufWriter::new(file);
        out.write_all(TRACE_MAGIC)?;
        for field in [TRACE_VERSION, n_layers, n_experts, top_k, 0] {
            out.write_all(&field.to_le_bytes())?;
        }
        Ok(Self {
            out,
            n_layers,
            top_k,
            filled: 0,
            records: 0,
        })
    }

    /// Append one layer's routing decision. Layers must arrive in order
    /// 0..n_layers for each token; `phase` and `position` are read only
    /// when a new record starts (at layer 0).
    fn push(
        &mut self,
        phase: TracePhase,
        position: usize,
        layer: u32,
        topk: &[(u32, f32)],
    ) -> anyhow::Result<()> {
        if topk.len() as u32 != self.top_k {
            bail!(
                "trace: layer {layer} routed {} experts, want {}",
                topk.len(),
                self.top_k
            );
        }
        let expected = self.filled / self.top_k;
        if layer != expected {
            bail!("trace: layer {layer} out of order, expected {expected}");
        }
        if self.filled == 0 {
            let mut head = [0u8; 8];
            head[0] = match phase {
                TracePhase::Prefill => 0,
                TracePhase::Decode => 1,
            };
            let pos = u32::try_from(position)
                .with_context(|| format!("trace: position {position} exceeds u32"))?;
            head[4..8].copy_from_slice(&pos.to_le_bytes());
            self.out.write_all(&head)?;
        }
        for &(expert, _) in topk {
            self.out.write_all(&expert.to_le_bytes())?;
        }
        self.filled += self.top_k;
        if self.filled == self.n_layers * self.top_k {
            self.filled = 0;
            self.records += 1;
        }
        Ok(())
    }

    /// Flush, patch the header's record count, and report it. A record
    /// left half-written (a run that failed mid-token) is not counted, so
    /// readers see only whole records.
    fn finish(mut self) -> anyhow::Result<u32> {
        self.out.flush()?;
        self.out.seek(SeekFrom::Start(TRACE_N_RECORDS_OFFSET))?;
        self.out.write_all(&self.records.to_le_bytes())?;
        self.out.flush()?;
        Ok(self.records)
    }
}

/// One [`forward_token`], optionally recording its routing as prefill.
fn forward_traced<'s>(
    model: &Model,
    state: &'s mut ForwardState,
    token_id: u32,
    position: usize,
    want_logits: bool,
    trace: Option<&mut TraceWriter>,
) -> anyhow::Result<Option<&'s [f32]>> {
    let Some(writer) = trace else {
        return Ok(forward_token(
            model,
            state,
            token_id,
            position,
            want_logits,
        )?);
    };
    let mut failure: Option<anyhow::Error> = None;
    let mut sink = |layer: u32, topk: &[(u32, f32)]| {
        if failure.is_none() {
            if let Err(e) = writer.push(TracePhase::Prefill, position, layer, topk) {
                failure = Some(e);
            }
        }
    };
    let out = forward_token_traced(
        model,
        state,
        token_id,
        position,
        want_logits,
        Some(&mut sink),
    )?;
    match failure {
        Some(e) => Err(e),
        None => Ok(out),
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
    trace_experts: Option<&Path>,
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
    let mut on_token = |_: u32, text: &str| {
        let _ = stdout.write_all(text.as_bytes());
        let _ = stdout.flush();
    };
    let stats = match trace_experts {
        None => generate(
            &model,
            &mut state,
            &tokenizer,
            &prompt_ids,
            &params,
            &mut on_token,
        )?,
        Some(path) => {
            let arch = model.arch();
            let mut writer = TraceWriter::create(path, arch.n_layers, arch.n_experts, arch.top_k)?;
            let mut failure: Option<anyhow::Error> = None;
            let mut sink = |phase: TracePhase, pos: usize, layer: u32, topk: &[(u32, f32)]| {
                if failure.is_none() {
                    if let Err(e) = writer.push(phase, pos, layer, topk) {
                        failure = Some(e);
                    }
                }
            };
            let stats = generate_traced(
                &model,
                &mut state,
                &tokenizer,
                &prompt_ids,
                &params,
                &mut on_token,
                &mut sink,
            )?;
            if let Some(e) = failure {
                return Err(e).context("writing the expert trace");
            }
            let records = writer.finish()?;
            eprintln!("expert trace: {records} records -> {}", path.display());
            stats
        }
    };
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
fn run_logits(
    model_dir: &Path,
    prompt: &str,
    top: usize,
    trace_experts: Option<&Path>,
    skip_hashes: bool,
) -> anyhow::Result<()> {
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

    let arch = model.arch();
    let mut writer = trace_experts
        .map(|path| TraceWriter::create(path, arch.n_layers, arch.n_experts, arch.top_k))
        .transpose()?;

    let last = ids.len() - 1;
    for (pos, &id) in ids[..last].iter().enumerate() {
        forward_traced(&model, &mut state, id, pos, false, writer.as_mut())?;
    }
    let logits = forward_traced(&model, &mut state, ids[last], last, true, writer.as_mut())?
        .context("final forward pass returned no logits")?;
    if let Some(writer) = writer {
        let records = writer.finish()?;
        eprintln!("expert trace: {records} records");
    }

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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    /// A trace file path unique to this process and call.
    fn temp_trace(tag: &str) -> PathBuf {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "ramvamp-trace-{tag}-{}-{n}.bin",
            std::process::id()
        ))
    }

    /// One decoded trace record.
    #[derive(Debug, PartialEq, Eq)]
    struct Record {
        phase: u8,
        position: u32,
        experts: Vec<u32>,
    }

    /// Reference reader for the `RVMPTRC1` format documented on
    /// [`TraceWriter`] (the production reader is `scripts/lfu_sim.py`;
    /// this one exists to prove the writer round-trips).
    ///
    /// Returns `(n_layers, n_experts, top_k, records)`.
    fn read_trace(path: &Path) -> (u32, u32, u32, Vec<Record>) {
        let bytes = std::fs::read(path).expect("trace file readable");
        assert!(bytes.len() >= 28, "short header: {} bytes", bytes.len());
        assert_eq!(&bytes[0..8], TRACE_MAGIC);
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        assert_eq!(u32_at(8), TRACE_VERSION);
        let (n_layers, n_experts, top_k) = (u32_at(12), u32_at(16), u32_at(20));
        let n_records = u32_at(24) as usize;

        let ids_per_record = (n_layers * top_k) as usize;
        let record_bytes = 8 + ids_per_record * 4;
        assert_eq!(bytes.len(), 28 + n_records * record_bytes, "trailing bytes");

        let records = (0..n_records)
            .map(|r| {
                let base = 28 + r * record_bytes;
                assert_eq!(&bytes[base + 1..base + 4], &[0, 0, 0], "pad must be zero");
                Record {
                    phase: bytes[base],
                    position: u32_at(base + 4),
                    experts: (0..ids_per_record)
                        .map(|i| u32_at(base + 8 + i * 4))
                        .collect(),
                }
            })
            .collect();
        (n_layers, n_experts, top_k, records)
    }

    /// `top_k` routed pairs whose ids start at `first` (weights are not
    /// part of the trace, so their values only have to be plausible).
    fn routed(first: u32, top_k: u32) -> Vec<(u32, f32)> {
        (0..top_k)
            .map(|i| (first + i, 1.0 / (i + 1) as f32))
            .collect()
    }

    #[test]
    fn trace_writer_round_trips() {
        let path = temp_trace("round-trip");
        let (n_layers, n_experts, top_k) = (3u32, 8u32, 2u32);
        let mut w = TraceWriter::create(&path, n_layers, n_experts, top_k).unwrap();

        let phases = [
            (TracePhase::Prefill, 0usize),
            (TracePhase::Prefill, 1),
            (TracePhase::Decode, 2),
        ];
        for (token, &(phase, position)) in phases.iter().enumerate() {
            for layer in 0..n_layers {
                let first = (token as u32 * 10) + layer;
                w.push(phase, position, layer, &routed(first, top_k))
                    .unwrap();
            }
        }
        assert_eq!(w.finish().unwrap(), 3);

        let (got_layers, got_experts, got_top_k, records) = read_trace(&path);
        assert_eq!(
            (got_layers, got_experts, got_top_k),
            (n_layers, n_experts, top_k)
        );
        assert_eq!(records.len(), 3);
        for (token, (&(phase, position), record)) in phases.iter().zip(&records).enumerate() {
            assert_eq!(record.phase, u8::from(phase == TracePhase::Decode));
            assert_eq!(record.position as usize, position);
            // Layer-major, routed order preserved.
            let want: Vec<u32> = (0..n_layers)
                .flat_map(|layer| {
                    let first = (token as u32 * 10) + layer;
                    (0..top_k).map(move |i| first + i)
                })
                .collect();
            assert_eq!(record.experts, want);
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn trace_writer_counts_only_complete_records() {
        // A run that dies mid-token must not advertise a partial record.
        let path = temp_trace("partial");
        let mut w = TraceWriter::create(&path, 4, 8, 2).unwrap();
        for layer in 0..4 {
            w.push(TracePhase::Decode, 7, layer, &routed(layer, 2))
                .unwrap();
        }
        w.push(TracePhase::Decode, 8, 0, &routed(0, 2)).unwrap();
        assert_eq!(w.finish().unwrap(), 1);

        let bytes = std::fs::read(&path).unwrap();
        // Header claims one record; the half-written second one trails it.
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 1);
        assert_eq!(bytes.len(), 28 + (8 + 4 * 2 * 4) + (8 + 2 * 4));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn trace_writer_rejects_out_of_order_layers() {
        let path = temp_trace("order");
        let mut w = TraceWriter::create(&path, 3, 8, 2).unwrap();
        w.push(TracePhase::Prefill, 0, 0, &routed(0, 2)).unwrap();
        let err = w
            .push(TracePhase::Prefill, 0, 2, &routed(0, 2))
            .unwrap_err()
            .to_string();
        assert!(err.contains("out of order"), "{err}");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn trace_writer_rejects_wrong_top_k() {
        let path = temp_trace("topk");
        let mut w = TraceWriter::create(&path, 3, 8, 4).unwrap();
        let err = w
            .push(TracePhase::Prefill, 0, 0, &routed(0, 3))
            .unwrap_err()
            .to_string();
        assert!(err.contains("routed 3 experts"), "{err}");
        assert!(TraceWriter::create(&path, 0, 8, 4).is_err());
        assert!(TraceWriter::create(&path, 3, 8, 0).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
