#include <cuda_runtime.h>
#include <stdint.h>
#include "paged_attention_utils.cuh"
#include "isoquant_utils.cuh"

extern "C" __global__ void paged_attention_v2_reduce(
    float* __restrict__ output,
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const float* __restrict__ tmp_out,
    const int32_t* __restrict__ context_lens,
    const int num_heads,
    const int head_dim,
    const int num_partitions,
    const float* __restrict__ rotation_metadata
) {
    // Note: The safety check for head_dim % 4 should ideally be in the C++ launcher.
    if (head_dim % 4 != 0) {
        return;
    }

    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = head_dim / 4;

    const int context_len = context_lens[seq_idx];
    const int num_partitions_for_seq = (context_len + (PARTITION_SIZE - 1)) / PARTITION_SIZE;

    if (num_partitions_for_seq <= 0) {
        if (tid < num_threads) {
            const int out_idx = (seq_idx * num_heads * head_dim) + (head_idx * head_dim) + tid * 4;
            *reinterpret_cast<float4*>(output + out_idx) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }

    // Load IsoQuant rotations for the current head
    const int metadata_offset = head_idx * 8;
    const float4 qL = *reinterpret_cast<const float4*>(rotation_metadata + metadata_offset);
    const float4 qR = *reinterpret_cast<const float4*>(rotation_metadata + metadata_offset + 4);

    const int stat_base_idx = (seq_idx * num_heads * num_partitions) + (head_idx * num_partitions);
    const int tmp_out_base_idx = (seq_idx * num_heads * num_partitions * head_dim) + (head_idx * num_partitions * head_dim);

    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_rescale[512];

    if (tid == 0) {
        float m_final = -INFINITY;
        int p_limit = min(num_partitions_for_seq, 512);
        
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

    if (tid >= num_threads) return;

    float s_final = s_sum;
    int p_limit = min(num_partitions_for_seq, 512);

    float4 out_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int p = 0; p < p_limit; ++p) {
        float4 tmp_vec = *reinterpret_cast<const float4*>(tmp_out + tmp_out_base_idx + p * head_dim + tid * 4);
        float rescale = s_rescale[p];
        out_vec.x += rescale * tmp_vec.x;
        out_vec.y += rescale * tmp_vec.y;
        out_vec.z += rescale * tmp_vec.z;
        out_vec.w += rescale * tmp_vec.w;
    }

    if (s_final != 0.0f) {
        out_vec.x /= s_final;
        out_vec.y /= s_final;
        out_vec.z /= s_final;
        out_vec.w /= s_final;
    }

    out_vec = apply_inverse_isoquant(out_vec, qL, qR);

    const int out_idx = (seq_idx * num_heads * head_dim) + (head_idx * head_dim) + tid * 4;
    *reinterpret_cast<float4*>(output + out_idx) = out_vec;
}
