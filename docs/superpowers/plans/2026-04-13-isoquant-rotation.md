# IsoQuant Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement blockwise SO(4) isoclinic rotations (quaternions) for the KV cache to decorrelate features.

**Architecture:** Extended `GpuBlockPool` with rotation metadata. Modified `reshape_and_cache` to derive and apply rotations. Updated `paged_attention` to rotate $Q$ and un-rotate $V$ partial sums.

**Tech Stack:** Rust, CUDA (PTX), `candle-core`.

---

### Task 1: Update PagedAttention Infrastructure

**Files:**
- Modify: `crates/vllm-paged-attention/src/block.rs`
- Modify: `crates/vllm-paged-attention/src/allocator.rs`

- [ ] **Step 1: Add `rotation_metadata` to `GpuBlockPool`**

Update `GpuBlockPool::new` to allocate a tensor of shape `[num_blocks, num_kv_heads, head_dim / 4, 8]` with `DType::F32`.

- [ ] **Step 2: Add accessor for metadata in `GpuBlockPool`**

- [ ] **Step 3: Update `BlockAllocator` tests to reflect new pool structure**

- [ ] **Step 4: Commit**

```bash
git add crates/vllm-paged-attention/src/
git commit -m "feat: add rotation metadata storage to GpuBlockPool"
```

---

### Task 2: Implement IsoQuant Math Utilities

**Files:**
- Create: `crates/vllm-cuda/src/kernels/isoquant_utils.cuh`

- [ ] **Step 1: Implement quaternion multiplication and SO(4) transform**

```cpp
// Left-isoclinic: qL * v
// Right-isoclinic: v * conj(qR)
// SO(4) transform: qL * v * conj(qR)
```

- [ ] **Step 2: Implement basic rotation derivation**

Implement a simplified logic to compute $(q_L, q_R)$ from a 4D vector (e.g., rotating it towards the $[1, 1, 1, 1]$ direction).

- [ ] **Step 3: Commit**

```bash
git add crates/vllm-cuda/src/kernels/isoquant_utils.cuh
git commit -m "feat: implement IsoQuant SO(4) math utilities"
```

---

### Task 3: Implement IsoQuant Write Kernel

**Files:**
- Modify: `crates/vllm-cuda/src/kernels/paged_attention.cu`
- Modify: `crates/vllm-cuda/src/kernels/mod.rs`

- [ ] **Step 1: Implement `reshape_and_cache_isoquant` kernel**

It should:
1. Check if it's the first token in the block.
2. If yes, derive $(q_L, q_R)$ and store in `rotation_metadata`.
3. If no, load existing metadata.
4. Rotate $K$ and $V$ using `apply_isoquant`.
5. Store in cache.

- [ ] **Step 2: Update Rust launcher in `mod.rs`**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: implement reshape_and_cache with IsoQuant rotation"
```

---

### Task 4: Update Attention Kernels for IsoQuant

**Files:**
- Modify: `crates/vllm-cuda/src/kernels/paged_attention_v1_plus.cu`
- Modify: `crates/vllm-cuda/src/kernels/paged_attention_v2_partition.cu`
- Modify: `crates/vllm-cuda/src/kernels/paged_attention_v2_reduce.cu`

- [ ] **Step 1: Update V1+ kernel**

Rotate $Q$ sub-vectors before dot product. Un-rotate the weighted sum of $V$ before final write.

- [ ] **Step 2: Update V2 kernels**

V2 Partition should handle $Q$ rotation and $V$ rotation. V2 Reduce should handle the final un-rotation if needed (or keep it in Partition).

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: integrate IsoQuant rotation into PagedAttention kernels"
```

---

### Task 5: Integration & Custom Op Update

**Files:**
- Modify: `crates/vllm-cuda/src/kernels/paged_attn_op.rs`

- [ ] **Step 1: Update `PagedAttentionOp` to accept `rotation_metadata` tensor**

- [ ] **Step 2: Add `use_isoquant` flag to `PagedAttentionOp`**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: enable IsoQuant dispatch in PagedAttentionOp"
```

---

### Task 6: Verification & Metrics

**Files:**
- Create: `crates/vllm-cuda/tests/test_isoquant.rs`

- [ ] **Step 1: Implement numerical parity test**

- [ ] **Step 2: Measure decorrelation**

Add a test that prints the Kurtosis of the KV cache with and without IsoQuant.

- [ ] **Step 3: Commit**

```bash
git commit -m "test: verify IsoQuant numerical parity and decorrelation"
```
