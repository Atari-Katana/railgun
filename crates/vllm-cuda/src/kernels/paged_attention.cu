/*
 * Railgun PagedAttention Kernel V1
 *
 * This kernel performs the attention score calculation and weighted sum 
 * for a decoded token across paged KV blocks.
 *
 * Layout:
 * - Query: [num_seqs, num_heads, head_dim]
 * - Key Cache: [num_blocks, num_kv_heads, block_size, head_dim]
 * - Value Cache: [num_blocks, num_kv_heads, block_size, head_dim]
 * - Block Table: [num_seqs, max_num_blocks_per_seq]
 * - Context Lens: [num_seqs]
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void paged_attention_v1(
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
    const int seq_idx = blockIdx.x; // One block per sequence in batch
    const int head_idx = threadIdx.y; // num_heads threads in y
    const int tid = threadIdx.x; // head_dim or smaller

    if (seq_idx >= gridDim.x || head_idx >= num_heads) return;

    const int context_len = context_lens[seq_idx];
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    
    // Pointer to this sequence's query
    const float* q_ptr = query + (seq_idx * num_heads * head_dim) + (head_idx * head_dim);

    // Shared memory for intermediate sums (attention scores)
    // For simplicity in V1, we'll do direct accumulation if head_dim is small.
    // In a production kernel, we would use shared memory and shuffles for reduction.
    
    float acc = 0.0f;
    float l_max = -1e20f;
    float l_sum = 0.0f;
    
    // We'll iterate over all blocks for this sequence
    for (int token_idx = 0; token_idx < context_len; ++token_idx) {
        int block_idx_in_table = token_idx / block_size;
        int token_offset_in_block = token_idx % block_size;
        
        int physical_block_id = block_table[seq_idx * max_blocks_per_seq + block_idx_in_table];
        
        // Compute dot product Q * K
        float score = 0.0f;
        const float* k_ptr = key_cache + (physical_block_id * num_kv_heads * block_size * head_dim) + (kv_head_idx * block_size * head_dim) + (token_offset_in_block * head_dim);

        // Dot product over head_dim
        // V1: Assuming thread per head_dim for now or looping.
        for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;

        // Softmax online update (FlashAttention style)
        float old_l_max = l_max;
        l_max = fmaxf(l_max, score);
        l_sum = l_sum * expf(old_l_max - l_max) + expf(score - l_max);
        
        // We'll need another pass or a different loop structure to accumulate V correctly.
        // For V1, let's keep it simple: two passes.
    }

    // Pass 2: Weighted sum of Values
    // Actually, to avoid two passes, we need a buffer or enough shared memory.
    // Given we are in Phase 5 baseline, I'll implement the "simple but correct" version first.
    
    for (int d = 0; d < head_dim; ++d) {
        float out_val = 0.0f;
        float current_max = -1e20f;
        float current_sum = 0.0f;
        float res = 0.0f;

        for (int token_idx = 0; token_idx < context_len; ++token_idx) {
            int block_idx_in_table = token_idx / block_size;
            int token_offset_in_block = token_idx % block_size;
            int physical_block_id = block_table[seq_idx * max_blocks_per_seq + block_idx_in_table];

            // Recompute score (or load from shared if we had space)
            float score = 0.0f;
            const float* k_ptr = key_cache + (physical_block_id * num_kv_heads * block_size * head_dim) + (kv_head_idx * block_size * head_dim) + (token_offset_in_block * head_dim);
            for (int k_d = 0; k_d < head_dim; ++k_d) score += q_ptr[k_d] * k_ptr[k_d];
            score *= scale;

            const float* v_ptr = value_cache + (physical_block_id * num_kv_heads * block_size * head_dim) + (kv_head_idx * block_size * head_dim) + (token_offset_in_block * head_dim);

            float exp_score = expf(score - l_max);
            res += exp_score * v_ptr[d];
        }
        
        output[(seq_idx * num_heads * head_dim) + (head_idx * head_dim) + d] = res / l_sum;
    }
}

extern "C" __global__ void reshape_and_cache(
    const float* __restrict__ k,             // [batch, num_kv_heads, head_dim]
    const float* __restrict__ v,             // [batch, num_kv_heads, head_dim]
    float* __restrict__ k_cache,             // [num_blocks, num_kv_heads, block_size, head_dim]
    float* __restrict__ v_cache,             // [num_blocks, num_kv_heads, block_size, head_dim]
    const int32_t* __restrict__ slot_mapping, // [batch]
    const int num_kv_heads,
    const int head_dim,
    const int block_size
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = threadIdx.y;
    const int dim_idx = threadIdx.x;

    if (batch_idx >= gridDim.x || head_idx >= num_kv_heads || dim_idx >= head_dim) return;

    const int slot_idx = slot_mapping[batch_idx];
    if (slot_idx < 0) return;

    const int block_idx = slot_idx / block_size;
    const int token_idx = slot_idx % block_size;

    const int src_idx = (batch_idx * num_kv_heads * head_dim) + (head_idx * head_dim) + dim_idx;
    const int dst_idx = (block_idx * num_kv_heads * block_size * head_dim) + (head_idx * block_size * head_dim) + (token_idx * head_dim) + dim_idx;

    k_cache[dst_idx] = k[src_idx];
    v_cache[dst_idx] = v[src_idx];
}
