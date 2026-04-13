//! `railgun serve` — OpenAI-compatible HTTP server.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::post,
    Json, Router,
};
use clap::Args;
use futures::Stream;
use serde::Deserialize;
use tracing::info;

use vllm_core::{Device, DType};
use vllm_engine::RailgunEngine;
use vllm_models::llama::model::LlamaModel;
use vllm_models::{RailgunTokenizer, CausalLM};
use vllm_scheduler::{Scheduler, SchedulerConfig, SamplingParams};

/// Arguments for the `serve` subcommand.
#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Path to the model directory.
    #[arg(short, long)]
    pub model: PathBuf,

    /// Host to bind to.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// CUDA device ordinal. Omit to use CPU.
    #[arg(long)]
    pub cuda_device: Option<u32>,

    /// Data type for model weights and activations.
    /// Options: f16, bf16, f32.
    #[arg(long)]
    pub dtype: Option<String>,

    /// KV cache data type. Defaults to the model's dtype.
    #[arg(long)]
    pub kv_dtype: Option<String>,

    // --- Scheduler Configuration ---

    /// Maximum number of requests in the running batch simultaneously.
    #[arg(long, default_value_t = 256, help_heading = "Scheduler Configuration")]
    pub max_num_seqs: usize,

    /// Maximum tokens processed per step (prefill + decode combined).
    #[arg(long, default_value_t = 4096, help_heading = "Scheduler Configuration")]
    pub max_num_batched_tokens: usize,

    /// KV cache block size (tokens per block).
    #[arg(long, default_value_t = 16, help_heading = "Scheduler Configuration")]
    pub block_size: usize,

    /// Total KV cache blocks available on GPU.
    #[arg(long, default_value_t = 1000, help_heading = "Scheduler Configuration")]
    pub num_gpu_blocks: usize,

    /// Total KV cache blocks available on CPU (host memory).
    #[arg(long, default_value_t = 512, help_heading = "Scheduler Configuration")]
    pub num_cpu_blocks: usize,

    /// Maximum sequence length (prompt + generation).
    #[arg(long, default_value_t = 4096, help_heading = "Scheduler Configuration")]
    pub max_model_len: usize,
}

/// Shared application state.
struct AppState {
    engine: Arc<RailgunEngine>,
}

/// OpenAI-compatible Chat Completion Request.
#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[allow(dead_code)]
    model: String,
    messages: Vec<ChatMessage>,
    
    // Sampling parameters
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    #[allow(dead_code)]
    stream: Option<bool>,
    stop: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

fn parse_dtype(s: &str) -> Result<DType> {
    match s.to_lowercase().as_str() {
        "f32" | "float32" => Ok(DType::F32),
        "f16" | "float16" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        _ => anyhow::bail!("Unsupported dtype: {}", s),
    }
}

/// Entry point for the `serve` subcommand.
pub fn run(args: ServeArgs) -> Result<()> {
    let device = match args.cuda_device {
        Some(ordinal) => Device::Cuda(ordinal),
        None => Device::Cpu,
    };

    let dtype_override = args.dtype.as_deref().map(parse_dtype).transpose()?;
    let kv_dtype = args.kv_dtype.as_deref().map(parse_dtype).transpose()?
        .unwrap_or(dtype_override.unwrap_or(DType::F16));

    // 1. Load model & tokenizer
    let tokenizer = RailgunTokenizer::load(&args.model)?;
    let model = LlamaModel::load(&args.model, device, dtype_override)?;
    
    // 2. Initialize Scheduler
    // We use values from the model's config
    let config = model.config();
    let sched_config = SchedulerConfig {
        max_num_batched_tokens: args.max_num_batched_tokens,
        max_num_seqs: args.max_num_seqs,
        block_size: args.block_size,
        num_gpu_blocks: args.num_gpu_blocks,
        num_cpu_blocks: args.num_cpu_blocks,
        max_model_len: args.max_model_len,
    };
    
    let scheduler = Scheduler::new(
        sched_config,
        device,
        kv_dtype,
        config.num_kv_heads(),
        config.head_dim(),
    )?;

    // 3. Start Engine
    let engine = RailgunEngine::new(
        model,
        tokenizer,
        scheduler,
    );

    let state = Arc::new(AppState {
        engine,
    });

    // 4. Setup Routes
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    info!(addr = %addr, "Railgun server listening");

    tokio::runtime::Runtime::new()?.block_on(async {
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    Ok(())
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    // Combine messages into a simple prompt (naive for Phase 4)
    let mut prompt = String::new();
    for msg in req.messages {
        prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }
    prompt.push_str("assistant: ");

    let sampling_params = SamplingParams {
        max_new_tokens: req.max_tokens.unwrap_or(128),
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        ..Default::default()
    };

    let mut rx = state.engine.generate(prompt, sampling_params).await;

    let stream = async_stream::stream! {
        while let Some(res) = rx.recv().await {
            // Simplified OpenAI streaming format
            let chunk = serde_json::json!({
                "id": format!("chatcmpl-{}", res.request_id),
                "object": "chat.completion.chunk",
                "created": 123456789,
                "model": "railgun-llama",
                "choices": [{
                    "index": 0,
                    "delta": { "content": res.text },
                    "finish_reason": res.finish_reason,
                }]
            });
            yield Ok(Event::default().data(chunk.to_string()));
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}
