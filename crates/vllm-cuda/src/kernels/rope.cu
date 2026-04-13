#include <cuda_runtime.h>

// CUDA Kernel for Rotary Positional Embeddings (RoPE)
// 
// Parameters:
// - x: [num_tokens, num_heads, head_dim] (float)
// - cos_sin: [max_seq_len, head_dim] (float, prepopulated table)
// - positions: [num_tokens] (int32)
// - num_heads: int
// - head_dim: int
extern "C" __global__ void rotary_embedding_kernel(
    float* __restrict__ x,
    const float* __restrict__ cos_sin,
    const int* __restrict__ positions,
    const int num_heads,
    const int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int d = threadIdx.x * 2; // Process 2 elements at a time (d and d + head_dim/2)

    if (d >= head_dim) return;

    const int pos = positions[token_idx];
    const int half_dim = head_dim / 2;
    
    // Offset for this head/token
    const int x_offset = token_idx * num_heads * head_dim + head_idx * head_dim;
    
    // RoPE math:
    // x_new[d] = x[d] * cos - x[d + half] * sin
    // x_new[d + half] = x[d + half] * cos + x[d] * sin
    
    const float cos_val = cos_sin[pos * head_dim + d / 2];
    const float sin_val = cos_sin[pos * head_dim + half_dim + d / 2];
    
    const float v_low = x[x_offset + d / 2];
    const float v_high = x[x_offset + half_dim + d / 2];
    
    x[x_offset + d / 2] = v_low * cos_val - v_high * sin_val;
    x[x_offset + half_dim + d / 2] = v_high * cos_val + v_low * sin_val;
}
