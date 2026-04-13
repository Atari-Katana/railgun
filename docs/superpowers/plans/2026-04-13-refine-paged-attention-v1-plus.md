# Refine PagedAttention V1+ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix shared memory reduction logic in PagedAttention V1+ and optimize kernel performance using warp-level primitives, while improving Rust error handling.

**Architecture:**
- Optimize `paged_attention_v1_plus.cu` by using `__shfl_xor_sync` for warp-level reduction, reducing `__syncthreads()` calls from $O(\log N)$ to $O(1)$ or $O(2)$.
- Ensure robustness for non-power-of-2 `head_dim`.
- Replace `.unwrap()` with `map_err` in `launch_v1_plus`.

**Tech Stack:**
- CUDA, Rust (candle-core, cudarc)

---

### Task 1: Optimize and Fix CUDA Kernel

**Files:**
- Modify: `crates/vllm-cuda/src/kernels/paged_attention_v1_plus.cu`

- [ ] **Step 1: Implement optimized reduction in `paged_attention_v1_plus.cu`**

Update the kernel to use warp-level primitives.

```cpp
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

extern "C" __global__ void paged_attention_v1_plus(
    const float* __restrict__ query,           // [batch, num_heads, head_dim]
    const float* __restrict__ key_cache,       // [num_blocks, num_kv_heads, block_size, head_dim]
    const float* __restrict__ value_cache,     // [num_blocks, num_kv_heads, block_size, head_dim]
    const int32_t* __restrict__ block_table,   // [batch, max_blocks]
    const int32_t* __restrict__ context_lens,  // [batch]
    float* __restrict__ output,                // [batch, num_heads, head_dim]
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq
) {
    const int head_idx = blockIdx.x % num_heads;
    const int seq_idx = blockIdx.x / num_heads;
    const int tid = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const float* q_ptr = query + (seq_idx * num_heads * head_dim) + (head_idx * head_dim);

    // Load Q into registers. Assuming head_dim <= blockDim.x.
    float q_val = (tid < head_dim) ? q_ptr[tid] : 0.0f;

    float m = -INFINITY;
    float s = 0.0f;
    float res = 0.0f;

    // Shared memory for block reduction (dot product)
    extern __shared__ float s_reduce[];

    for (int token_idx = 0; token_idx < context_len; ++token_idx) {
        int block_idx_in_table = token_idx / block_size;
        int token_offset_in_block = token_idx % block_size;
        int physical_block_id = block_table[seq_idx * max_blocks_per_seq + block_idx_in_table];

        const float* k_ptr = key_cache + (physical_block_id * num_kv_heads * block_size * head_dim) + (kv_head_idx * block_size * head_dim) + (token_offset_in_block * head_dim);
        const float* v_ptr = value_cache + (physical_block_id * num_kv_heads * block_size * head_dim) + (kv_head_idx * block_size * head_dim) + (token_offset_in_block * head_dim);

        float k_val = (tid < head_dim) ? k_ptr[tid] : 0.0f;
        float v_val = (tid < head_dim) ? v_ptr[tid] : 0.0f;

        // Dot product Q * K
        float qk = q_val * k_val;

        // Warp-level reduction
        unsigned int mask = __activemask();
        for (int offset = 16; offset > 0; offset >>= 1) {
            qk += __shfl_xor_sync(mask, qk, offset);
        }

        float score;
        if (head_dim <= 32) {
            score = __shfl_sync(mask, qk, 0) * scale;
        } else {
            __shared__ float s_score;
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            if (lane_id == 0) {
                s_reduce[warp_id] = qk;
            }
            __syncthreads();

            if (tid == 0) {
                float sum = 0;
                for (int i = 0; i < (head_dim + 31) / 32; ++i) {
                    sum += s_reduce[i];
                }
                s_score = sum * scale;
            }
            __syncthreads();
            score = s_score;
        }

        // Online Softmax (FlashAttention-style)
        float m_old = m;
        m = fmaxf(m, score);
        float exp_m_diff = expf(m_old - m);
        float exp_score_diff = expf(score - m);

        s = s * exp_m_diff + exp_score_diff;
        res = res * exp_m_diff + exp_score_diff * v_val;
    }

    // Final output calculation
    if (tid < head_dim) {
        output[(seq_idx * num_heads * head_dim) + (head_idx * head_dim) + tid] = res / s;
    }
}
```

---

### Task 2: Refine Rust Bindings

**Files:**
- Modify: `crates/vllm-cuda/src/kernels/mod.rs`

- [ ] **Step 1: Replace `.unwrap()` and update shared memory size in `launch_v1_plus`**

```rust
    pub unsafe fn launch_v1_plus(
        &self,
        query: &CudaSlice<f32>,
        key_cache: &CudaSlice<f32>,
        value_cache: &CudaSlice<f32>,
        block_table: &CudaSlice<i32>,
        context_lens: &CudaSlice<i32>,
        output: &mut CudaSlice<f32>,
        scale: f32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        max_blocks_per_seq: i32,
        batch_size: i32,
    ) -> CoreResult<()> {
        let stream = self.candle_dev.cuda_stream();
        let func = self.paged_attn_v1_plus_module
            .load_function("paged_attention_v1_plus")
            .map_err(|e| CoreError::Tensor(format!("Failed to load paged_attention_v1_plus function: {e}")))?;

        let cfg = LaunchConfig {
            grid_dim: ((batch_size * num_heads) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (((head_dim + 31) / 32) * 4) as u32, // Enough for warp partial sums
        };

        let mut builder = stream.launch_builder(&func);
        builder.arg(query).arg(key_cache).arg(value_cache).arg(block_table).arg(context_lens)
            .arg(output).arg(&scale).arg(&num_heads).arg(&num_kv_heads).arg(&head_dim)
            .arg(&block_size).arg(&max_blocks_per_seq);
        builder.launch(cfg)
            .map_err(|e| CoreError::Tensor(format!("CUDA launch error: {e}")))?;

        Ok(())
    }
```

---

### Task 3: Verification and Commitment

- [ ] **Step 1: Build the project**

Run: `cargo build -p vllm-cuda --features cuda`
Expected: Successful build.

- [ ] **Step 2: Commit all changes**

```bash
git add crates/vllm-cuda/src/kernels/paged_attention_v1_plus.cu crates/vllm-cuda/src/kernels/mod.rs
git commit -m "fix: refine PagedAttention V1+ kernel reduction logic and error handling"
```
