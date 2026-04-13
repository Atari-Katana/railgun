//! Native Llama implementation with PagedAttention support.

use candle_core::{Device, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module, RMSNorm, VarBuilder};
use crate::config::ModelConfig;
use crate::llama::attention::PagedSelfAttention;

pub struct LlamaLayer {
    attention: PagedSelfAttention,
    mlp: LlamaMLP,
    attention_norm: RMSNorm,
    ffn_norm: RMSNorm,
}

impl LlamaLayer {
    pub fn load(
        vb: VarBuilder,
        cfg: &candle_transformers::models::llama::Config,
    ) -> Result<Self> {
        let attention = PagedSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = LlamaMLP::load(vb.pp("mlp"), cfg)?;
        let attention_norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ffn_norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        
        Ok(Self {
            attention,
            mlp,
            attention_norm,
            ffn_norm,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
        slot_mapping: &Tensor,
        kv_cache: &mut vllm_paged_attention::block::BlockPool,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.attention_norm.forward(x)?;
        
        // Paged attention step
        // In Phase 5, this will call the custom kernel via PagedAttentionOp
        let x = self.attention.forward(&x, block_table, context_lens, kv_cache)?;
        let x = (x + residual)?;
        
        let residual = &x;
        let x = self.ffn_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        (x + residual)
    }
}

pub struct LlamaMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl LlamaMLP {
    pub fn load(
        vb: VarBuilder, 
        cfg: &candle_transformers::models::llama::Config,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = candle_nn::linear(h, i, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear(h, i, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear(i, h, vb.pp("down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

pub struct RailgunLlama {
    embed_tokens: Embedding,
    layers: Vec<LlamaLayer>,
    norm: RMSNorm,
    lm_head: Linear,
}

impl RailgunLlama {
    pub fn load(
        vb: VarBuilder,
        cfg: &candle_transformers::models::llama::Config,
    ) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(LlamaLayer::load(vb_layers.pp(i), cfg)?);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        
        Ok(Self { embed_tokens, layers, norm, lm_head })
    }

    pub fn forward(
        &self,
        tokens: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
        slot_mapping: &Tensor,
        kv_cache: &mut vllm_paged_attention::block::BlockPool,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = tokens.dims2()?;
        let mut x = self.embed_tokens.forward(tokens)?;
        for layer in &self.layers {
            x = layer.forward(&x, block_table, context_lens, slot_mapping, kv_cache)?;
        }
        x = self.norm.forward(&x)?;
        let last_token = x.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        self.lm_head.forward(&last_token)
    }

    pub fn forward_packed(
        &self,
        input_ids: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
        slot_mapping: &Tensor,
        kv_cache: &mut vllm_paged_attention::block::BlockPool,
    ) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, block_table, context_lens, slot_mapping, kv_cache)?;
        }
        x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }
}
