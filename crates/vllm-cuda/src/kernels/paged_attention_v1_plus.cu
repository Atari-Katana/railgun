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
    // For head_dim=64, we'll use blockDim.x=64.
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
        s_reduce[tid] = qk;
        __syncthreads();

        // Block reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_reduce[tid] += s_reduce[tid + stride];
            }
            __syncthreads();
        }
        float score = s_reduce[0] * scale;

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
