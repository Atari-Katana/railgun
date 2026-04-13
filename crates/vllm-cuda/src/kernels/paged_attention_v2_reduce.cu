#include <cuda_runtime.h>
#include <stdint.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

extern "C" __global__ void paged_attention_v2_reduce(
    float* __restrict__ output,            // [batch, num_heads, head_dim]
    const float* __restrict__ exp_sums,    // [batch, num_heads, num_partitions]
    const float* __restrict__ max_logits,  // [batch, num_heads, num_partitions]
    const float* __restrict__ tmp_out,     // [batch, num_heads, num_partitions, head_dim]
    const int32_t* __restrict__ context_lens, // [batch]
    const int num_heads,
    const int head_dim,
    const int num_partitions
) {
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // PARTITION_SIZE = 256 must match the value used in paged_attention_v2_partition.cu
    const int PARTITION_SIZE = 256;
    const int context_len = context_lens[seq_idx];
    const int num_partitions_for_seq = (context_len + (PARTITION_SIZE - 1)) / PARTITION_SIZE;

    if (num_partitions_for_seq <= 0) {
        if (tid < head_dim) {
            const int out_idx = (seq_idx * num_heads * head_dim) + (head_idx * head_dim) + tid;
            output[out_idx] = 0.0f;
        }
        return;
    }

    // Base indices
    const int stat_base_idx = (seq_idx * num_heads * num_partitions) + (head_idx * num_partitions);
    const int tmp_out_base_idx = (seq_idx * num_heads * num_partitions * head_dim) + (head_idx * num_partitions * head_dim);

    // Shared memory for broadcasting global statistics
    __shared__ float s_max;
    __shared__ float s_sum;

    // Thread 0 computes global statistics once per block (head)
    if (tid == 0) {
        // 1. Find global max logit across partitions
        float m_final = -INFINITY;
        for (int p = 0; p < num_partitions_for_seq; ++p) {
            m_final = fmaxf(m_final, max_logits[stat_base_idx + p]);
        }
        s_max = m_final;

        // 2. Compute global sum of exponentials
        float s_final = 0.0f;
        for (int p = 0; p < num_partitions_for_seq; ++p) {
            float m_p = max_logits[stat_base_idx + p];
            float s_p = exp_sums[stat_base_idx + p];
            s_final += expf(m_p - m_final) * s_p;
        }
        s_sum = s_final;
    }
    __syncthreads();

    if (tid >= head_dim) return;

    float m_final = s_max;
    float s_final = s_sum;

    // 3. Compute weighted sum of partial outputs for this thread's head_dim element
    float out_val = 0.0f;
    for (int p = 0; p < num_partitions_for_seq; ++p) {
        float m_p = max_logits[stat_base_idx + p];
        float res_p = tmp_out[tmp_out_base_idx + p * head_dim + tid];
        out_val += expf(m_p - m_final) * res_p;
    }

    // 4. Normalize and write output
    const int out_idx = (seq_idx * num_heads * head_dim) + (head_idx * head_dim) + tid;
    output[out_idx] = out_val / s_final;
}
