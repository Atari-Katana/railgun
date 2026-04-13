//! `railgun generate` — offline text generation.
//!
//! Loads a model from a local directory, tokenizes the prompt,
//! runs greedy decoding, and prints the generated text to stdout.
//!
//! # Example
//!
//! ```text
//! railgun generate \
//!     --model /models/Llama-3.2-1B \
//!     --prompt "The capital of France is" \
//!     --max-tokens 32 \
//!     --temperature 0.0
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use tracing::info;

use vllm_core::Device;
use vllm_models::llama::model::LlamaModel;
use vllm_models::{CausalLM, RailgunTokenizer};

/// Arguments for the `generate` subcommand.
#[derive(Args, Debug)]
pub struct GenerateArgs {
    /// Path to the model directory (must contain config.json + *.safetensors).
    #[arg(short, long)]
    pub model: PathBuf,

    /// Text prompt to generate from.
    #[arg(short, long)]
    pub prompt: String,

    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 128)]
    pub max_tokens: usize,

    /// Sampling temperature (0.0 = greedy).
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// CUDA device ordinal. Omit to use CPU.
    #[arg(long)]
    pub cuda_device: Option<u32>,

    /// Print token IDs alongside generated text (useful for validation).
    #[arg(long, default_value_t = false)]
    pub show_token_ids: bool,
}

/// Entry point for the `generate` subcommand.
pub fn run(args: GenerateArgs) -> Result<()> {
    let device = match args.cuda_device {
        Some(ordinal) => Device::Cuda(ordinal),
        None => Device::Cpu,
    };
    info!(device = %device, model = %args.model.display(), "generate command start");

    // ── Load tokenizer ─────────────────────────────────────────────────────
    let tokenizer = RailgunTokenizer::load(&args.model)
        .with_context(|| format!("loading tokenizer from {}", args.model.display()))?;

    // ── Load model ─────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let mut model = LlamaModel::load(&args.model, device, None)
        .with_context(|| format!("loading model from {}", args.model.display()))?;
    let load_ms = t0.elapsed().as_millis();
    info!(load_ms, "model loaded");

    // ── Tokenize prompt ────────────────────────────────────────────────────
    let prompt_ids = tokenizer.encode(&args.prompt, true)
        .with_context(|| "tokenizing prompt")?;
    info!(
        prompt_len = prompt_ids.len(),
        prompt_ids = ?&prompt_ids[..prompt_ids.len().min(16)],
        "prompt tokenized"
    );

    let eos_id = tokenizer
        .eos_token_id()
        .or_else(|| vllm_models::RailgunTokenizer::eos_from_config(model.config()));

    // ── Run greedy generation ──────────────────────────────────────────────
    let gen_start = Instant::now();
    let output_ids = model
        .generate_greedy(&prompt_ids, args.max_tokens, eos_id)
        .with_context(|| "during generation")?;
    let elapsed = gen_start.elapsed();
    let tps = output_ids.len() as f64 / elapsed.as_secs_f64();

    // ── Decode and print ───────────────────────────────────────────────────
    let output_text = tokenizer.decode(&output_ids, true)
        .with_context(|| "decoding output tokens")?;

    println!("\n{}{}", args.prompt, output_text);

    if args.show_token_ids {
        println!("\n[prompt token ids] {:?}", &prompt_ids);
        println!("[output token ids] {:?}", &output_ids);
    }

    eprintln!(
        "\n[stats] generated {} tokens in {:.1}s ({:.1} tok/s), load {load_ms}ms",
        output_ids.len(),
        elapsed.as_secs_f64(),
        tps
    );

    Ok(())
}
