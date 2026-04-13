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
use serde::{Deserialize, Serialize};
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
}

/// Shared application state.
struct AppState {
    engine: Arc<RailgunEngine>,
    tokenizer: Arc<RailgunTokenizer>,
}

/// OpenAI-compatible Chat Completion Request.
#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[allow(dead_code)]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// OpenAI-compatible Chat Completion Response (non-stream).
#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatMessageResponse {
    role: String,
    content: String,
}

/// Entry point for the `serve` subcommand.
pub fn run(args: ServeArgs) -> Result<()> {
    let device = match args.cuda_device {
        Some(ordinal) => Device::Cuda(ordinal),
        None => Device::Cpu,
    };

    // 1. Load model & tokenizer
    let tokenizer = RailgunTokenizer::load(&args.model)?;
    let model = LlamaModel::load(&args.model, device)?;
    
    // 2. Initialize Scheduler
    // We use values from the model's config
    let config = model.config();
    let sched_config = SchedulerConfig::default(); // TODO: tune from args
    
    // For Phase 4, we use CDType::F32/F16 for KV depending on model
    let kv_dtype = DType::F16; 
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
        tokenizer.clone(), // This is actually cheap or we wrap in Arc
        scheduler,
    );

    let state = Arc::new(AppState {
        engine,
        tokenizer: Arc::new(tokenizer),
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
        max_new_tokens: req.max_tokens.unwrap_or(128) as u32,
        temperature: req.temperature.unwrap_or(0.7),
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
