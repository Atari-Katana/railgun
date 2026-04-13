//! Railgun CLI — `generate` and `serve` subcommands.
//!
//! # Usage
//!
//! ```text
//! railgun generate --model /path/to/model --prompt "Hello" --max-tokens 50
//! railgun serve    --model /path/to/model --port 8080
//! ```

mod generate;
mod serve;
mod benchmark;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name        = "railgun",
    about       = "High-throughput LLM inference — pure Rust",
    version,
    long_about  = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a prompt (offline, single request).
    Generate(generate::GenerateArgs),
    /// Start the OpenAI-compatible HTTP server (Phase 4).
    Serve(serve::ServeArgs),
    /// Perform a throughput benchmark.
    Benchmark(benchmark::BenchmarkArgs),
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .compact()
        .init();
}

fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Commands::Generate(args) => generate::run(args),
        Commands::Serve(args)    => serve::run(args),
        Commands::Benchmark(args) => benchmark::run(args),
    }
}
