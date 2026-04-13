# PagedAttention V2 Reduction Optimization and Constants Unification

## 1. Overview
Refine the PagedAttention V2 implementation by unifying constants in a shared header and optimizing the reduction kernel's performance by pre-computing rescale factors in shared memory.

## 2. Technical Design

### 2.1. Shared Header: `paged_attention_utils.cuh`
- **Path:** `crates/vllm-cuda/src/kernels/paged_attention_utils.cuh`
- **Content:**
  - Guard against multiple inclusions.
  - Define `INFINITY` if not defined.
  - Define `PARTITION_SIZE` as 256.

### 2.2. Kernel Refinement: `paged_attention_v2_partition.cu`
- Remove local `INFINITY` and `PARTITION_SIZE` definitions.
- Include `paged_attention_utils.cuh`.

### 2.3. Kernel Optimization: `paged_attention_v2_reduce.cu`
- Include `paged_attention_utils.cuh`.
- Remove local `INFINITY` and `PARTITION_SIZE` definitions.
- Add `__shared__ float s_rescale[512];` for rescale factors.
- **Thread 0 Logic Update:**
  1. Find `m_final` across `min(num_partitions_for_seq, num_partitions)`.
  2. Compute `s_rescale[p] = expf(m_p - m_final)` and accumulate `s_final` in one loop.
  3. Store `s_rescale[p]` in shared memory.
- **Main Loop Update:**
  - Read `s_rescale[p]` from shared memory instead of calling `expf`.
  - Add `p < num_partitions` bounds checking.

## 3. Implementation Plan

### Step 1: Create `paged_attention_utils.cuh`
Define common constants.

### Step 2: Update `paged_attention_v2_partition.cu`
Refactor to use the new header.

### Step 3: Update `paged_attention_v2_reduce.cu`
Implement the shared memory optimization and bounds checking.

### Step 4: Verification
Run `cargo build -p vllm-cuda --features cuda`.

### Step 5: Commit
Commit changes with message: "fix: optimize PagedAttention V2 reduction rescale factors and unify constants"
