#include <cuda_runtime.h>
#include <stdint.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

#define PARTITION_SIZE 256

extern "C" __global__ void paged_attention_v2_partition(
    const float* __restrict__ query,           // [batch, num_heads, head_dim]
    const float* __restrict__ key_cache,       // [num_blocks, num_kv_heads, block_size, head_dim]
    const float* __restrict__ value_cache,     // [num_blocks, num_kv_heads, block_size, head_dim]
    const int32_t* __restrict__ block_table,   // [batch, max_blocks]
    const int32_t* __restrict__ context_lens,  // [batch]
    float* __restrict__ tmp_out,               // [batch, num_heads, num_partitions, head_dim]
    float* __restrict__ exp_sums,              // [batch, num_heads, num_partitions]
    float* __restrict__ max_logits,            // [batch, num_heads, num_partitions]
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq,
    const int num_partitions
) {
    const int partition_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    const int start_token_idx = partition_idx * PARTITION_SIZE;
    
    if (start_token_idx >= context_len) return;
    
    const int end_token_idx = min(start_token_idx + PARTITION_SIZE, context_len);

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const float* q_ptr = query + (seq_idx * num_heads * head_dim) + (head_idx * head_dim);

    // Load Q into registers.
    float q_val = (tid < head_dim) ? q_ptr[tid] : 0.0f;

    float m = -INFINITY;
    float s = 0.0f;
    float res = 0.0f;

    // Shared memory for block reduction (dot product)
    extern __shared__ float s_reduce[];

    for (int token_idx = start_token_idx; token_idx < end_token_idx; ++token_idx) {
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
        for (int offset = 16; offset > 0; offset >>= 1) {
            qk += __shfl_xor_sync(0xffffffff, qk, offset);
        }

        float score;
        if (head_dim <= 32) {
            score = __shfl_sync(0xffffffff, qk, 0) * scale;
        } else {
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            if (lane_id == 0) {
                s_reduce[warp_id] = qk;
            }
            __syncthreads();

            float sum = 0;
            for (int i = 0; i < (head_dim + 31) / 32; ++i) {
                sum += s_reduce[i];
            }
            score = sum * scale;
            __syncthreads();
        }

        // Online Softmax (FlashAttention-style)
        float m_old = m;
        m = fmaxf(m, score);
        float exp_m_diff = expf(m_old - m);
        float exp_score_diff = expf(score - m);

        s = s * exp_m_diff + exp_score_diff;
        res = res * exp_m_diff + exp_score_diff * v_val;
    }

    // Output partial results
    if (tid < head_dim) {
        int out_base_idx = (seq_idx * num_heads * num_partitions * head_dim) + 
                           (head_idx * num_partitions * head_dim) + 
                           (partition_idx * head_dim);
        tmp_out[out_base_idx + tid] = res;
    }
    
    if (tid == 0) {
        int stat_base_idx = (seq_idx * num_heads * num_partitions) + 
                            (head_idx * num_partitions) + 
                            (partition_idx);
        exp_sums[stat_base_idx] = s;
        max_logits[stat_base_idx] = m;
    }
}
