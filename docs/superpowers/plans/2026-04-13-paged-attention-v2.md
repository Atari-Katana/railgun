# PagedAttention V2 (Hybrid Dispatch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a high-performance PagedAttention engine with a single-pass V1+ kernel for short sequences and a partitioned V2 kernel for long contexts.

**Architecture:** Hybrid dispatch in `PagedAttentionOp`. Sequence lengths <= 4096 use Optimized V1+ (online softmax, one-pass). Sequences > 4096 use Partitioned V2 (parallel partitions + final reduction).

**Tech Stack:** Rust, CUDA (PTX), `cudarc`, `candle-core`.

---

### Task 1: Build Infrastructure & Placeholder Kernels

**Files:**
- Modify: `crates/vllm-cuda/build.rs`
- Create: `crates/vllm-cuda/src/kernels/paged_attention_v1_plus.cu`
- Create: `crates/vllm-cuda/src/kernels/paged_attention_v2_partition.cu`
- Create: `crates/vllm-cuda/src/kernels/paged_attention_v2_reduce.cu`

- [ ] **Step 1: Update `build.rs` to compile new kernels**

```rust
// In crates/vllm-cuda/build.rs
// Add to the kernels vector:
let kernels = vec!["paged_attention", "paged_attention_v1_plus", "paged_attention_v2_partition", "paged_attention_v2_reduce", "rope"];
```

- [ ] **Step 2: Create placeholder `.cu` files**

Write empty `extern "C"` kernels to ensure the build pipeline works.

- [ ] **Step 3: Run build to verify compilation**

Run: `cargo build -p vllm-cuda --features cuda`
Expected: `Compiled ... to PTX` for all 5 kernels.

- [ ] **Step 4: Commit**

```bash
git add crates/vllm-cuda/build.rs crates/vllm-cuda/src/kernels/*.cu
git commit -m "build: setup infrastructure for V1+ and V2 kernels"
```

---

### Task 2: Implement Optimized V1+ (Single-Pass)

**Files:**
- Modify: `crates/vllm-cuda/src/kernels/paged_attention_v1_plus.cu`
- Modify: `crates/vllm-cuda/src/kernels/mod.rs`

- [ ] **Step 1: Implement the V1+ Kernel**

Implement online softmax with shared memory for local reduction.

```cuda
extern "C" __global__ void paged_attention_v1_plus(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens,
    float* __restrict__ output,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq
) {
    // Single-pass implementation using Online Softmax logic
    // ...
}
```

- [ ] **Step 2: Add launcher to `mod.rs`**

```rust
pub unsafe fn launch_v1_plus(...) -> CoreResult<()> {
    // ... load "paged_attention_v1_plus" from paged_attn_v1_plus_module
}
```

- [ ] **Step 3: Update `PagedAttentionKernels::new` to load the new PTX**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: implement single-pass PagedAttention V1+ kernel"
```

---

### Task 3: Implement Partitioned V2 - Phase 1 (Partition)

**Files:**
- Modify: `crates/vllm-cuda/src/kernels/paged_attention_v2_partition.cu`
- Modify: `crates/vllm-cuda/src/kernels/mod.rs`

- [ ] **Step 1: Implement Partition Kernel**

Calculates partial maxes, sums, and weighted values for 256-token chunks.

- [ ] **Step 2: Define partial result buffer structure**

Define how partial results are stored in global memory for the reduction kernel.

- [ ] **Step 3: Add launcher to `mod.rs`**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: implement PagedAttention V2 partition kernel"
```

---

### Task 4: Implement Partitioned V2 - Phase 2 (Reduction)

**Files:**
- Modify: `crates/vllm-cuda/src/kernels/paged_attention_v2_reduce.cu`
- Modify: `crates/vllm-cuda/src/kernels/mod.rs`

- [ ] **Step 1: Implement Reduction Kernel**

Combines partial results from V2 Partition.

- [ ] **Step 2: Add launcher to `mod.rs`**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: implement PagedAttention V2 reduction kernel"
```

---

### Task 5: Hybrid Dispatch Integration

**Files:**
- Modify: `crates/vllm-cuda/src/kernels/paged_attn_op.rs`

- [ ] **Step 1: Implement dispatch logic**

```rust
// In PagedAttentionOp::execute
let max_context_len = // query context_lens tensor max
if max_context_len <= 4096 {
    self.kernels.launch_v1_plus(...)
} else {
    // Allocate temporary scratchpad
    // self.kernels.launch_v2_partition(...)
    // self.kernels.launch_v2_reduce(...)
}
```

- [ ] **Step 2: Handle scratchpad memory**

Use `DeviceBuffer` for temporary storage during V2 execution.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: implement hybrid dispatch logic in PagedAttentionOp"
```

---

### Task 6: Testing & Benchmarking

**Files:**
- Create: `crates/vllm-cuda/tests/test_paged_attention.rs`

- [ ] **Step 1: Write numerical parity test**

Compare V1+, V2, and baseline `paged_attention_v1` on identical inputs.

- [ ] **Step 2: Run benchmarks**

Run: `cargo run --release --benchmark -- --num-requests 1 --max-tokens 512`
Verify performance improvement.

- [ ] **Step 3: Final Housekeeping**

Run linters and workspace tests.

- [ ] **Step 4: Commit**

```bash
git commit -m "test: verify PagedAttention V2 numerical accuracy and performance"
```
