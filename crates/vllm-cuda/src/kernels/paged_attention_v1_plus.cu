#include <cuda_runtime.h>
#include <stdint.h>
#include "isoquant_utils.cuh"

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

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
    const int max_blocks_per_seq,
    const float* __restrict__ rotation_metadata
) {
    // Note: The safety check for head_dim % 4 should ideally be in the C++ launcher
    // for better performance, but is included here as a fallback.
    if (head_dim % 4 != 0) {
        return;
    }

    const int head_idx = blockIdx.x % num_heads;
    const int seq_idx = blockIdx.x / num_heads;
    const int tid = threadIdx.x;
    const int num_threads = head_dim / 4;

    if (tid >= num_threads) return;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    // Load IsoQuant rotations for the current head/sequence
    const int metadata_offset = (seq_idx * num_heads + head_idx) * 8; // 8 floats per head (qL, qR)
    const float4 qL = *reinterpret_cast<const float4*>(rotation_metadata + metadata_offset);
    const float4 qR = *reinterpret_cast<const float4*>(rotation_metadata + metadata_offset + 4);

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const float* q_ptr = query + (seq_idx * num_heads * head_dim) + (head_idx * head_dim);

    // Load and rotate Q
    float4 q_vec = *reinterpret_cast<const float4*>(q_ptr + tid * 4);
    q_vec = apply_isoquant(q_vec, qL, qR);

    float m = -INFINITY;
    float s = 0.0f;
    float4 o_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int token_idx = 0; token_idx < context_len; ++token_idx) {
        int block_idx_in_table = token_idx / block_size;
        int token_offset_in_block = token_idx % block_size;
        int physical_block_id = block_table[seq_idx * max_blocks_per_seq + block_idx_in_table];

        const float* k_ptr_base = key_cache + (physical_block_id * num_kv_heads * block_size * head_dim) + (kv_head_idx * block_size * head_dim) + (token_offset_in_block * head_dim);
        const float* v_ptr_base = value_cache + (physical_block_id * num_kv_heads * block_size * head_dim) + (kv_head_idx * block_size * head_dim) + (token_offset_in_block * head_dim);

        // Load and rotate K and V
        float4 k_vec = *reinterpret_cast<const float4*>(k_ptr_base + tid * 4);
        float4 v_vec = *reinterpret_cast<const float4*>(v_ptr_base + tid * 4);

        k_vec = apply_isoquant(k_vec, qL, qR);
        v_vec = apply_isoquant(v_vec, qL, qR);

        // Dot product
        float qk_dot = q_vec.x * k_vec.x + q_vec.y * k_vec.y + q_vec.z * k_vec.z + q_vec.w * k_vec.w;
        
        for (int offset = num_threads / 2; offset > 0; offset >>= 1) {
            qk_dot += __shfl_down_sync(0xffffffff, qk_dot, offset);
        }

        float score = (tid == 0) ? __shfl_sync(0xffffffff, qk_dot, 0) * scale : 0.0f;
        score = __shfl_sync(0xffffffff, score, 0);

        // Update step
        float m_old = m;
        m = fmaxf(m, score);
        float exp_m_diff = expf(m_old - m);
        float exp_score_diff = expf(score - m);

        s = s * exp_m_diff + exp_score_diff;
        o_vec.x = o_vec.x * exp_m_diff + exp_score_diff * v_vec.x;
        o_vec.y = o_vec.y * exp_m_diff + exp_score_diff * v_vec.y;
        o_vec.z = o_vec.z * exp_m_diff + exp_score_diff * v_vec.z;
        o_vec.w = o_vec.w * exp_m_diff + exp_score_diff * v_vec.w;
    }
    
    // Final normalization and inverse rotation
    if (s != 0.0f) {
        o_vec.x /= s;
        o_vec.y /= s;
        o_vec.z /= s;
        o_vec.w /= s;
    }

    o_vec = apply_inverse_isoquant(o_vec, qL, qR);

    float* output_ptr = output + (seq_idx * num_heads * head_dim) + (head_idx * head_dim);
    *reinterpret_cast<float4*>(output_ptr + tid * 4) = o_vec;
}
