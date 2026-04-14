# Design: IsoQuant (Blockwise SO(4) Rotation)

**Status:** Draft
**Date:** 2026-04-13
**Topic:** CUDA Kernels / KV Cache Compression

## 1. Overview
IsoQuant is a blockwise rotation framework designed to decorrelate features in LLM KV caches, reducing quantization error. It leverages the isoclinic decomposition of the SO(4) rotation group, representing 4D feature blocks as quaternions and applying the transform $T(v) = q_L v \bar{q}_R$.

We will implement **Approach 1: Per-Block Semi-Dynamic IsoQuant Rotation**. Rotations are derived and stored at the granularity of one physical KV block (16 tokens).

## 2. Architecture

### 2.1 Metadata Storage
The `GpuBlockPool` will be extended with a metadata tensor to store the isoclinic rotation parameters for each block.

*   **Tensor**: `rotation_metadata`
*   **Shape**: `[num_blocks, num_kv_heads, head_dim / 4, 8]`
*   **Data Type**: `FP32` (8 floats: $q_L$ and $q_R$ quaternions)
*   **Granularity**: 1 set of rotations per 16 tokens.

### 2.2 IsoQuant Kernels

#### A. `reshape_and_cache_isoquant`
This kernel handles the writing of new tokens into the paged cache.
1.  **Rotation Derivation**: If the token being written is the *first* token in a block (offset 0), derive the optimal $(q_L, q_R)$ to minimize feature variance within that 4D sub-block. For Phase 1, we will use a simplified Procrustes-style derivation or a fixed-point iteration as described in the paper.
2.  **Rotation Application**: Apply $T(v) = q_L v \bar{q}_R$ to the $K$ and $V$ features of the incoming token using the block's metadata.
3.  **Storage**: Store rotated $K, V$ in the cache and $q_L, q_R$ in metadata.

#### B. `paged_attention_isoquant` (Update to V1+/V2)
The attention kernels must be modified to maintain numerical invariance:
1.  **Query Rotation**: For each block loaded from memory, the corresponding sub-vectors of the Query ($Q$) must be rotated by $R_b$ before the dot product: $(R_b Q) \cdot (R_b K) = Q \cdot K$.
2.  **Value Un-rotation**: The weighted sum of $V$ for a block will be in the rotated space. Before adding the block's contribution to the global head accumulator, the partial sum must be rotated by the inverse transform $R_b^{-1}$ (using conjugate quaternions $\bar{q}_L, q_R$).

### 2.3 Mathematical Invariance
The inner product is preserved by the orthogonal rotation:
$$\langle R Q, R K \rangle = \langle Q, R^T R K \rangle = \langle Q, K \rangle$$
The value space is restored by:
$$\sum \alpha_i (R V_i) = R \sum \alpha_i V_i \implies R^{-1} (R \sum \alpha_i V_i) = \sum \alpha_i V_i$$

## 3. Implementation Plan

### Task 1: Metadata Infrastructure
*   Modify `vllm-paged-attention` to allocate the `rotation_metadata` tensor in `GpuBlockPool`.
*   Update `BlockAllocator` to handle metadata initialization.

### Task 2: IsoQuant Math Utilities
*   Create `isoquant_utils.cuh` with:
    *   Quaternion multiplication (`q_mul`).
    *   Isoclinic SO(4) transform (`apply_isoquant`).
    *   Simplified dynamic rotation derivation logic.

### Task 3: IsoQuant Write Kernel
*   Implement `reshape_and_cache_isoquant` in `paged_attention.cu`.

### Task 4: IsoQuant Attention Kernels
*   Modify `paged_attention_v1_plus.cu` and `paged_attention_v2` to support the rotation/un-rotation logic.

### Task 5: Integration & Verification
*   Update `PagedAttentionOp` to dispatch to IsoQuant kernels.
*   Add integration tests to verify numerical parity with the non-rotated baseline.

## 4. Success Criteria
*   **Numerical Parity**: Output features match baseline within $10^{-5}$ epsilon.
*   **Decorrelation**: Measurable reduction in feature Kurtosis or variance in the rotated KV cache.
*   **Latency**: $<15\%$ overhead for the rotation logic (Phase 1 goal).
