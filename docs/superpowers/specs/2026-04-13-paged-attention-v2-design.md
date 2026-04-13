# Design: PagedAttention V2 (Hybrid Dispatch)

**Status:** Draft
**Date:** 2026-04-13
**Topic:** CUDA Kernels / PagedAttention

## 1. Overview
Railgun's current `paged_attention_v1` kernel is a slow, two-pass baseline. To support modern LLMs (e.g., Llama 3.1) with long context (up to 128k) while maintaining peak performance for short sequences, we will implement a **Hybrid Dispatch** architecture.

## 2. Architecture

### 2.1 Dispatch Logic
The `PagedAttentionOp` in `vllm-cuda` will choose between two execution paths based on the sequence length (`context_len`):

| Context Length | Kernel Path | Description |
| :--- | :--- | :--- |
| **<= 4096** | **Optimized V1+** | Single-pass online softmax. One thread block per head. Optimized for latency. |
| **> 4096** | **Partitioned V2** | Two-phase execution: Partitions the sequence into 256-token chunks, processes in parallel, then reduces. |

### 2.2 Kernel Specifications

#### Optimized V1+ (Single-Pass)
*   **Algorithm:** Online Softmax (FlashAttention-style).
*   **Parallelism:** Each thread block handles one attention head (`batch_size * num_heads` blocks).
*   **Optimization:** Threads within a block collaborate to compute dot products (Q*K) and weighted sums (V) in a single loop over KV blocks.
*   **Shared Memory:** Used for cross-thread `max` and `sum` reductions.

#### Partitioned V2 (Two-Phase)
*   **Phase 1 (Partition):** 
    *   Sequence is split into $N$ partitions of size 256.
    *   Each thread block processes one partition.
    *   Outputs: `partial_sums`, `partial_maxes`, and `partial_values` to global temporary buffer.
*   **Phase 2 (Reduction):**
    *   A single reduction kernel reads the partial results and computes the final weighted sum.

## 3. Implementation Plan

### Phase 1: CUDA Kernel Development
1.  **`paged_attention_v1_plus.cu`**: Implement the single-pass online softmax kernel.
2.  **`paged_attention_v2.cu`**: Implement the partitioned and reduction kernels.
3.  **Build Script Update**: Modify `crates/vllm-cuda/build.rs` to compile the new `.cu` files.

### Phase 2: Rust Integration
1.  **`PagedAttentionKernels`**: Update the kernel launcher in `crates/vllm-cuda/src/kernels/mod.rs` to load and launch the new functions.
2.  **`PagedAttentionOp`**: Implement the dispatch logic in `crates/vllm-cuda/src/kernels/paged_attn_op.rs`.
3.  **Temporary Buffers**: Add support for the scratchpad memory required by the V2 reduction phase (using `DeviceBuffer`).

## 4. Testing & Validation
*   **Numerical Parity:** Verify that V1+, Partitioned V2, and the original CPU fallback produce identical results (within float precision).
*   **Throughput Benchmarking:** Measure tokens/sec at 512, 4096, and 16k context lengths using `railgun benchmark`.
*   **Memory Safety:** Ensure temporary buffers for V2 are correctly freed and do not cause leaks during high-concurrency serving.

## 5. Success Criteria
*   **V1+ Performance:** >2x speedup over current `paged_attention_v1` for short sequences.
*   **Long Context:** Successful generation at 32k+ context length without kernel timeout.
