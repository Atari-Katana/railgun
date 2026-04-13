# Railgun ⚡

**Railgun** is a high-performance, pure-Rust implementation of an LLM inference engine, reimagined from the ground up for robustness and throughput.

It implements the core breakthroughs of modern LLM serving—**PagedAttention** and **Continuous Batching**—entirely in Rust and hand-optimized CUDA kernels.

## Key Features

- **PagedAttention v1**: Memory-efficient KV cache management using paged blocks. Reduces VRAM fragmentation and allows for significantly larger batch sizes.
- **Continuous Batching**: Dynamically schedules incoming requests into a single "packed" GPU batch, maximizing utilization and minimizing latency.
- **Native CUDA Kernels**: Custom-written kernels for PagedAttention, KV cache reshaping, and Rotary Positional Embeddings (RoPE) for asynchronous, packed sequences.
- **Pure Rust Engine**: Built on `candle-core` for tensor arithmetic, but uses a native architecture for the forward pass to support high-throughput serving.
- **OpenAI Compatible**: Includes a streaming HTTP server out of the box.

## Architecture

Railgun is organized into several modular crates:

- `vllm-cuda`: Hand-optimized PTX kernels and their Rust launchers.
- `vllm-paged-attention`: The physical memory pool and block allocator.
- `vllm-scheduler`: Continuous batching logic and request lifecycle management.
- `vllm-models`: Native Llama architecture supporting packed-sequence processing.
- `vllm-engine`: Async orchestrator that keeps the GPU saturated.
- `vllm-cli`: The user-facing tool for serving and testing.

## Getting Started

### Prerequisites

- **Rust**: Latest stable version.
- **CUDA Toolkit**: v12.0 or newer.
- **Compiler**: `gcc-14` and `g++-14` are required for kernel compilation (configured via the `bin/nvcc` wrapper).

### Installation

```bash
# Clone the repository
git clone https://github.com/Atari-Katana/railgun
cd railgun

# Build the project (enforcing CUDA support and custom host compiler)
PATH=$PWD/bin:$PATH CUDAHOSTCXX=/usr/bin/gcc-14 cargo build --release --features cuda
```

### Usage

#### 1. Quick Generate (Smoke Test)
Run a single request through the single-threaded standalone path:
```bash
cargo run --release --features cuda -- generate \
    --model /path/to/Llama-3.2-1B \
    --prompt "The capital of France is" \
    --max-tokens 50
```

#### 2. Start the Inference Server
Launch the multi-request continuous batching engine:
```bash
cargo run --release --features cuda -- serve \
    --model /path/to/Llama-3.2-1B \
    --port 8080
```

The server is compatible with any OpenAI client:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "railgun-llama",
    "messages": [{"role": "user", "content": "Tell me a joke."}],
    "stream": true
  }'
```

## License

MIT
