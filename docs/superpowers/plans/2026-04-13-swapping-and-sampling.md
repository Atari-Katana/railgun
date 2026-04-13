# Refinement Plan: Host Swapping and Top-K Sampling

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement KV block swapping to host memory and full `top_k` sampling support.

**Architecture:**
1.  **Host Swapping**: Extend `vllm-paged-attention` with a `CpuBlockPool` and implement `swap_in`/`swap_out` in the `Scheduler`.
2.  **Top-K Sampling**: Implement logit masking in `vllm-engine/src/sampling.rs`.

**Tech Stack:** Rust, CUDA, `candle-core`.

---

### Task 1: CPU Block Pool & Memory Management

**Files:**
- Modify: `crates/vllm-paged-attention/src/block.rs`
- Modify: `crates/vllm-paged-attention/src/allocator.rs`

- [ ] **Step 1: Add `CpuBlockPool` to `block.rs`**
  Implement a pool that lives on the CPU. It should have the same structure as `BlockPool` but use `Device::Cpu`.

- [ ] **Step 2: Add `swap` methods to `BlockAllocator`**
  Implement `swap_out(gpu_id, cpu_id)` and `swap_in(cpu_id, gpu_id)` using `candle` tensor operations.

- [ ] **Step 3: Commit**
```bash
git add crates/vllm-paged-attention/src/block.rs crates/vllm-paged-attention/src/allocator.rs
git commit -m "feat: implement CPU block pool and swap primitives"
```

---

### Task 2: Scheduler Swapping Logic

**Files:**
- Modify: `crates/vllm-scheduler/src/scheduler.rs`
- Modify: `crates/vllm-scheduler/src/request.rs`

- [ ] **Step 1: Track Swapped Requests**
  Add `swapped: VecDeque<Request>` to `Scheduler`. Update `RequestStatus` or add a flag to track if a request is on CPU.

- [ ] **Step 2: Implement `swap_out` logic in `schedule()`**
  When GPU blocks are low, move the least-recently-used running request to CPU instead of just preempting to `waiting`.

- [ ] **Step 3: Implement `swap_in` logic in `schedule()`**
  When GPU blocks are available, move requests from `swapped` back to `running`.

- [ ] **Step 4: Commit**
```bash
git commit -m "feat: integrate host swapping into the scheduler"
```

---

### Task 3: Implement Top-K Sampling

**Files:**
- Modify: `crates/vllm-engine/src/sampling.rs`

- [ ] **Step 1: Implement `top_k` masking**
  In `Sampler::sample`, if `top_k > 0`, sort logits, keep the top K, and mask the rest to `-infinity`.

- [ ] **Step 2: Run verification**
  Ensure it doesn't break existing `top_p` or `temperature` logic.

- [ ] **Step 3: Commit**
```bash
git commit -m "feat: implement top_k sampling logic"
```

---

### Task 4: Testing & Validation

**Files:**
- Create: `crates/vllm-engine/tests/test_sampling.rs`
- Create: `crates/vllm-scheduler/tests/test_swapping.rs`

- [ ] **Step 1: Verify Top-K**
  Test that `top_k=1` always returns the same token (greedy) and `top_k=5` limits the diversity.

- [ ] **Step 2: Verify Swapping**
  Simulate memory pressure and verify requests are moved to CPU and back without losing state.

- [ ] **Step 3: Commit**
```bash
git commit -m "test: verify host swapping and top_k sampling"
```
