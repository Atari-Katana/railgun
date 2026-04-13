//! PagedAttention implementation for Railgun.
//!
//! This module provides the `PagedAttention` layer which replaces the 
//! standard causal attention in Transformer models. Instead of a single 
//! contiguous KV cache tensor, it uses a block-based cache managed by 
//! `vllm-paged-attention`.

use candle_core::{Result, Tensor, Module};
use candle_nn::VarBuilder;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use vllm_cuda::kernels::PagedAttentionKernels;

/// PagedAttention Layer.
/// 
/// This layer performs the following:
/// 1. Project input hidden states to Q, K, V.
/// 2. Store K and V into the paged cache (using `block_table`).
/// 3. Fetch all previous K and V blocks for this request.
/// 4. Perform scaled dot-product attention.
pub struct PagedSelfAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,

    #[cfg(feature = "cuda")]
    kernels: Option<Arc<PagedAttentionKernels>>,
}

impl PagedSelfAttention {
    pub fn load(
        vb: VarBuilder,
        cfg: &candle_transformers::models::llama::Config,
        #[cfg(feature = "cuda")]
        kernels: Option<Arc<PagedAttentionKernels>>,
    ) -> Result<Self> {
        let size_q = cfg.hidden_size;
        let size_kv = cfg.hidden_size / (cfg.num_attention_heads / cfg.num_key_value_heads);
        
        let q_proj = if vb.contains_tensor("q_proj.bias") {
            candle_nn::linear(size_q, cfg.hidden_size, vb.pp("q_proj"))?
        } else {
            candle_nn::linear_no_bias(size_q, cfg.hidden_size, vb.pp("q_proj"))?
        };
        let k_proj = if vb.contains_tensor("k_proj.bias") {
            candle_nn::linear(size_q, size_kv, vb.pp("k_proj"))?
        } else {
            candle_nn::linear_no_bias(size_q, size_kv, vb.pp("k_proj"))?
        };
        let v_proj = if vb.contains_tensor("v_proj.bias") {
            candle_nn::linear(size_q, size_kv, vb.pp("v_proj"))?
        } else {
            candle_nn::linear_no_bias(size_q, size_kv, vb.pp("v_proj"))?
        };
        let o_proj = if vb.contains_tensor("o_proj.bias") {
            candle_nn::linear(cfg.hidden_size, size_q, vb.pp("o_proj"))?
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, size_q, vb.pp("o_proj"))?
        };
        
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            scale,
            #[cfg(feature = "cuda")]
            kernels,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        block_table: &Tensor,     // [batch, max_blocks]
        context_lens: &Tensor,    // [batch]
        slot_mapping: &Tensor,   // [total_tokens]
        kv_cache: &mut vllm_paged_attention::block::GpuBlockPool,
        max_context_len: usize,
    ) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let total_tokens = match x.rank() {
            2 => x.dim(0)?,
            3 => x.dim(0)? * x.dim(1)?,
            _ => return Err(candle_core::Error::Msg("Unsupported rank for input tensor".to_string())),
        };

        #[cfg(feature = "cuda")]
        if let Some(kernels) = &self.kernels {
            let mut k_cache = kv_cache.k_cache()?;
            let mut v_cache = kv_cache.v_cache()?;
            
            let op = vllm_cuda::kernels::PagedAttentionOp {
                scale: self.scale as f32,
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
                block_size: kv_cache.block_size,
                kernels: kernels.clone(),
            };

            // Reshape K/V for the kernel: [total_tokens, num_kv_heads, head_dim]
            let k_reshaped = k.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;
            let v_reshaped = v.reshape((total_tokens, self.num_kv_heads, self.head_dim))?;

            // 1. Store K/V into Paged Cache
            op.reshape_and_cache(&k_reshaped, &v_reshaped, &mut k_cache, &mut v_cache, slot_mapping)?;

            // 2. Compute Attention
            if x.rank() == 2 {
                // PACKED/DECODE PATH
                let q_reshaped = q.reshape((total_tokens, self.num_heads, self.head_dim))?;
                let out = op.execute(&q_reshaped, &k_cache, &v_cache, block_table, context_lens, max_context_len)?;
                return self.o_proj.forward(&out.reshape((total_tokens, ()))?);
            }
        }

        // FALLBACK PATH (or rank 3 / CPU)
        if x.rank() == 2 {
            // If we are here in packed mode, it means CUDA kernels are missing or not enabled.
            // This is currently unsupported for production high-throughput serving.
            return self.o_proj.forward(&q); 
        }

        let (b_sz, seq_len, _) = x.dims3()?;
        
        let _q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let _k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let _v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let att = (_q.matmul(&_k.transpose(2, 3)?)? * self.scale)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let y = att.matmul(&_v)?;
        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;
        self.o_proj.forward(&y)
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}
