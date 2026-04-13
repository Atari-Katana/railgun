# PagedAttention V2 Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize PagedAttention V2 reduction rescale factors by pre-computing them in shared memory and unify constants in a shared header.

**Architecture:** Create a common `.cuh` header for constants. Update V2 kernels to use it. Optimize the reduction kernel to avoid redundant `expf` calls by using thread 0 to pre-compute factors for all partitions in shared memory.

**Tech Stack:** CUDA (C++), Rust (Cargo for verification).

---

### Task 1: Create `paged_attention_utils.cuh`

**Files:**
- Create: `crates/vllm-cuda/src/kernels/paged_attention_utils.cuh`

- [ ] **Step 1: Create the header file**

```cpp
#ifndef PAGED_ATTENTION_UTILS_CUH
#define PAGED_ATTENTION_UTILS_CUH

#include <cuda_runtime.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

#define PARTITION_SIZE 256

#endif // PAGED_ATTENTION_UTILS_CUH
```

- [ ] **Step 2: Commit**

```bash
git add crates/vllm-cuda/src/kernels/paged_attention_utils.cuh
git commit -m "feat: add paged_attention_utils.cuh for unified constants"
```

---

### Task 2: Update `paged_attention_v2_partition.cu`

**Files:**
- Modify: `crates/vllm-cuda/src/kernels/paged_attention_v2_partition.cu`

- [ ] **Step 1: Refactor to use the new header**

```cpp
#include <cuda_runtime.h>
#include <stdint.h>
#include "paged_attention_utils.cuh"

// Remove local INFINITY and PARTITION_SIZE definitions

extern "C" __global__ void paged_attention_v2_partition(
...
```

- [ ] **Step 2: Commit**

```bash
git add crates/vllm-cuda/src/kernels/paged_attention_v2_partition.cu
git commit -m "refactor: use paged_attention_utils.cuh in v2 partition kernel"
```

---

### Task 3: Optimize `paged_attention_v2_reduce.cu`

**Files:**
- Modify: `crates/vllm-cuda/src/kernels/paged_attention_v2_reduce.cu`

- [ ] **Step 1: Implement shared memory optimization and bounds checking**

```cpp
#include <cuda_runtime.h>
#include <stdint.h>
#include "paged_attention_utils.cuh"

extern "C" __global__ void paged_attention_v2_reduce(
    float* __restrict__ output,            // [batch, num_heads, head_dim]
    const float* __restrict__ exp_sums,    // [batch, num_heads, num_partitions]
    const float* __restrict__ max_logits,  // [batch, num_heads, num_partitions]
    const float* __restrict__ tmp_out,     // [batch, num_heads, num_partitions, head_dim]
    const int32_t* __restrict__ context_lens, // [batch]
    const int num_heads,
    const int head_dim,
    const int num_partitions
) {
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    const int num_partitions_for_seq = (context_len + (PARTITION_SIZE - 1)) / PARTITION_SIZE;

    if (num_partitions_for_seq <= 0) {
        if (tid < head_dim) {
            const int out_idx = (seq_idx * num_heads * head_dim) + (head_idx * head_dim) + tid;
            output[out_idx] = 0.0f;
        }
        return;
    }

    // Base indices
    const int stat_base_idx = (seq_idx * num_heads * num_partitions) + (head_idx * num_partitions);
    const int tmp_out_base_idx = (seq_idx * num_heads * num_partitions * head_dim) + (head_idx * num_partitions * head_dim);

    // Shared memory for broadcasting global statistics and pre-computed factors
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_rescale[512]; // Max 512 partitions for 128k context

    // Thread 0 computes global statistics and pre-computes rescale factors
    if (tid == 0) {
        float m_final = -INFINITY;
        int p_limit = min(num_partitions_for_seq, num_partitions);
        
        for (int p = 0; p < p_limit; ++p) {
            m_final = fmaxf(m_final, max_logits[stat_base_idx + p]);
        }
        s_max = m_final;

        float s_final = 0.0f;
        for (int p = 0; p < p_limit; ++p) {
            float m_p = max_logits[stat_base_idx + p];
            float rescale = expf(m_p - m_final);
            s_rescale[p] = rescale;
            s_final += rescale * exp_sums[stat_base_idx + p];
        }
        s_sum = s_final;
    }
    __syncthreads();

    if (tid >= head_dim) return;

    float m_final = s_max;
    float s_final = s_sum;
    int p_limit = min(num_partitions_for_seq, num_partitions);

    // 3. Compute weighted sum using pre-computed rescale factors
    float out_val = 0.0f;
    for (int p = 0; p < p_limit; ++p) {
        out_val += s_rescale[p] * tmp_out[tmp_out_base_idx + p * head_dim + tid];
    }

    // 4. Normalize and write output
    const int out_idx = (seq_idx * num_heads * head_dim) + (head_idx * head_dim) + tid;
    output[out_idx] = out_val / s_final;
}
```

- [ ] **Step 2: Commit**

```bash
git add crates/vllm-cuda/src/kernels/paged_attention_v2_reduce.cu
git commit -m "fix: optimize PagedAttention V2 reduction rescale factors and shared memory"
```

---

### Task 4: Verification

**Files:**
- N/A

- [ ] **Step 1: Build the crate**

Run: `cargo build -p vllm-cuda --features cuda`
Expected: SUCCESS

- [ ] **Step 2: Final Commit (if needed for any fixups)**

```bash
# If any fixes were needed
git add .
git commit -m "chore: final fixups for paged attention refinement"
```
