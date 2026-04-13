# Railgun CLI Server Configuration

This document describes the CLI configuration options for the `railgun serve` command. These options allow you to tune the performance and resource usage of the inference engine.

## General Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Path to the model directory (must contain `config.json` and `*.safetensors`). | (Required) |
| `--host` | Host to bind the server to. | `0.0.0.0` |
| `--port` | Port to listen on. | `8080` |
| `--cuda-device` | CUDA device ordinal (e.g., `0`, `1`). Omit to use CPU. | `None` |
| `--dtype` | Data type for model weights and activations (`f16`, `bf16`, `f32`). | Loaded from config |
| `--kv-dtype` | Data type for the KV cache. | Matches `--dtype` |

## Scheduler Configuration

These options control the continuous batching scheduler, which manages how requests are packed together.

| Flag | Description | Default |
|------|-------------|---------|
| `--max-num-seqs` | Maximum number of concurrent requests in the running batch. | `256` |
| `--max-num-batched-tokens` | Maximum tokens (prefill + decode) processed in a single model step. | `4096` |
| `--block-size` | KV cache block size (tokens per block). | `16` |
| `--num-gpu-blocks` | Total number of KV cache blocks to allocate on the GPU. | `1000` |
| `--num-cpu-blocks` | Total number of KV cache blocks to allocate on host memory (for swapping). | `512` |
| `--max-model-len` | Maximum sequence length (prompt + generation) supported. | `4096` |

### Tuning Tips

- **GPU Memory**: If you encounter Out-Of-Memory (OOM) errors, try reducing `--num-gpu-blocks` or `--max-num-batched-tokens`.
- **Throughput**: For higher throughput with many concurrent users, increase `--max-num-seqs` and `--max-num-batched-tokens`, provided you have enough GPU memory.
- **Latency**: For lower latency on individual requests, keep `--max-num-batched-tokens` small (e.g., 512 or 1024), but this will reduce overall throughput.
- **Context Length**: Ensure `--max-model-len` is set correctly for your use case and model capabilities.

## OpenAI Compatibility

The server exposes an OpenAI-compatible endpoint at `/v1/chat/completions`.

### Supported Request Fields

- `model`: String (currently ignored, defaults to loaded model).
- `messages`: List of message objects with `role` and `content`.
- `max_tokens`: Maximum new tokens to generate.
- `temperature`: Sampling temperature (0.0 for greedy).
- `top_p`: Nucleus sampling parameter.
- `stream`: Boolean to enable streaming (defaults to `true`).
- `stop`: List of stop sequences (strings). *Note: Currently stop sequences as strings are being integrated.*

### Example Usage

```bash
railgun serve \
    --model /models/Llama-3.2-1B \
    --cuda-device 0 \
    --dtype f16 \
    --num-gpu-blocks 2000 \
    --max-num-seqs 128
```
