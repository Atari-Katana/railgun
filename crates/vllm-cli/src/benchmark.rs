//! `railgun benchmark` — Measure throughput and latency.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Args;
use futures::future::join_all;
use tracing::info;

use vllm_core::{Device, DType};
use vllm_engine::RailgunEngine;
use vllm_models::llama::model::LlamaModel;
use vllm_models::{RailgunTokenizer, CausalLM};
use vllm_scheduler::{Scheduler, SchedulerConfig, SamplingParams};

/// Arguments for the `benchmark` subcommand.
#[derive(Args, Debug)]
pub struct BenchmarkArgs {
    /// Path to the model directory.
    #[arg(short, long)]
    pub model: PathBuf,

    /// Number of concurrent requests to simulate.
    #[arg(short, long, default_value_t = 32)]
    pub num_requests: usize,

    /// Fixed prompt to use for all requests.
    #[arg(short, long, default_value = "Explain the importance of Rust in systems programming.")]
    pub prompt: String,

    /// Tokens to generate per request.
    #[arg(long, default_value_t = 128)]
    pub max_tokens: usize,

    /// CUDA device ordinal.
    #[arg(long)]
    pub cuda_device: Option<u32>,
}

pub fn run(args: BenchmarkArgs) -> Result<()> {
    let device = match args.cuda_device {
        Some(ordinal) => Device::Cuda(ordinal),
        None => Device::Cpu,
    };

    info!(num_requests = args.num_requests, model = %args.model.display(), "Starting benchmark");

    let tokenizer = RailgunTokenizer::load(&args.model)?;
    let model = LlamaModel::load(&args.model, device, None)?;
    
    let config = model.config();
    let sched_config = SchedulerConfig {
        max_num_seqs: args.num_requests + 1,
        max_num_batched_tokens: 4096,
        ..Default::default()
    };
    
    let scheduler = Scheduler::new(
        sched_config,
        device,
        DType::F16, 
        config.num_kv_heads(),
        config.head_dim(),
    )?;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let engine = RailgunEngine::new(model, tokenizer, scheduler);
        let start = Instant::now();
        let mut futures = Vec::new();

        for _ in 0..args.num_requests {
            let engine = engine.clone();
            let prompt = args.prompt.clone();
            let params = SamplingParams {
                max_new_tokens: args.max_tokens as u32,
                temperature: 0.0, // greedy for consistency
                ..Default::default()
            };

            futures.push(tokio::spawn(async move {
                let mut rx = engine.generate(prompt, params).await;
                let mut t_count = 0;
                while let Some(_) = rx.recv().await {
                    t_count += 1;
                }
                t_count
            }));
        }

        let results = join_all(futures).await;
        let elapsed = start.elapsed();
        
        let mut total_tokens = 0;
        for res in results {
            total_tokens += res.unwrap_or(0);
        }

        let tps = total_tokens as f64 / elapsed.as_secs_f64();
        
        println!("\nBenchmark Results:");
        println!("-----------------");
        println!("Concurrent Requests: {}", args.num_requests);
        println!("Tokens Generated:    {}", total_tokens);
        println!("Total Time:          {:.2}s", elapsed.as_secs_f64());
        println!("Throughput:          {:.2} tokens/sec", tps);
        println!("Avg Latency/Token:   {:.2}ms", (elapsed.as_secs_f64() * 1000.0) / total_tokens as f64);
    });

    Ok(())
}
