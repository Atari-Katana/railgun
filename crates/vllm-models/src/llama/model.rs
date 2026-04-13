//! Llama model — wraps `candle_transformers::models::llama::Llama`.
//!
//! # Design decision: wrap vs re-implement
//!
//! `candle-transformers` already ships a production-quality Llama implementation
//! with GQA, Llama-3 RoPE scaling, RMSNorm, and SwiGLU MLP — all tested against
//! the HuggingFace reference. Re-implementing these from scratch would add weeks
//! of work and risk numerical divergence. Instead we:
//!
//! 1. Delegate all forward-pass arithmetic to `candle_transformers::models::llama::Llama`.
//! 2. Handle weight loading (safetensors → `VarBuilder`) in this module.
//! 3. Expose the result through our own `CausalLM` trait so the rest of Railgun
//!    stays decoupled from candle's internal types.
//!
//! The KV cache is managed by `candle_transformers`' `Cache` type internally.
//! In Phase 5 we will replace this with our own `BlockAllocator`-backed
//! paged-attention cache, which requires a custom attention kernel.

use std::path::Path;

use candle_core::{DType as CDType, Device as CDevice, Tensor as CTensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config as LlamaInnerConfig, LlamaEosToks};
use tracing::info;

use vllm_core::{CoreError, Device, Result, Tensor};

use crate::config::ModelConfig;
use crate::traits::CausalLM;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert our `ModelConfig` into the candle-transformers `Config`.
fn to_candle_config(cfg: &ModelConfig) -> LlamaInnerConfig {
    LlamaInnerConfig {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size: cfg.vocab_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_kv_heads(),
        use_flash_attn: false,
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_theta as f32,
        bos_token_id: cfg.bos_token_id,
        eos_token_id: cfg.eos_token_id.map(LlamaEosToks::Single),
        rope_scaling: None,
        max_position_embeddings: cfg.max_position_embeddings,
        tie_word_embeddings: false,
    }
}

/// Map a HuggingFace `torch_dtype` string to a candle `DType`.
fn parse_dtype(s: Option<&str>) -> CDType {
    match s {
        Some("bfloat16") => CDType::BF16,
        Some("float16") => CDType::F16,
        Some("float32") => CDType::F32,
        _ => CDType::F32,
    }
}

/// Detect all `*.safetensors` files in a directory and return them sorted.
fn find_safetensors(model_dir: &Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    let mut files: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    files.sort();
    Ok(files)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public type
// ─────────────────────────────────────────────────────────────────────────────

use crate::llama::architecture::RailgunLlama;

/// A Llama-family causal language model using the native Railgun PagedAttention architecture.
pub struct LlamaModel {
    inner: RailgunLlama,
    config: ModelConfig,
    device: Device,
    dtype: CDType,
}

impl LlamaModel {
    /// Load a Llama model using the native Railgun architecture.
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config = ModelConfig::from_file(&config_path)?;

        let dtype = parse_dtype(config.torch_dtype.as_deref());
        let candle_config = to_candle_config(&config);
        let candle_device = CDevice::try_from(device)?;

        info!(
            model_dir = %model_dir.display(),
            architecture = "RailgunLlama",
            device = %device,
            "loading native Llama model"
        );

        let st_files = find_safetensors(model_dir)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&st_files, dtype, &candle_device)
                .map_err(CoreError::from)?
        };

        let inner = RailgunLlama::load(vb, &candle_config).map_err(CoreError::from)?;

        Ok(Self {
            inner,
            config,
            device,
            dtype,
        })
    }

    /// Access the native RailgunLlama architecture.
    pub fn architecture(&self) -> &RailgunLlama {
        &self.inner
    }

    /// Reset the model's local KV cache.
    pub fn reset(&mut self) {
        // In native mode, we'll just clear the local pool and state if we had one.
        // For simplicity, we'll recreate them on demand in prefill.
    }

    /// Sequential prefill (standalone mode).
    pub fn prefill(&mut self, token_ids: &[u32]) -> Result<CTensor> {
        let device = CDevice::try_from(self.device)?;
        let input_ids = CTensor::new(token_ids, &device)?;
        
        // Single-request standalone execution
        let num_tokens = token_ids.len();
        let block_size = 16;
        let num_blocks = (num_tokens + block_size - 1) / block_size + 1;

        let mut pool = vllm_paged_attention::block::BlockPool::new(
            num_blocks, 
            block_size, 
            self.inner.num_kv_heads(), 
            self.inner.head_dim(), 
            self.dtype.try_into().unwrap(), 
            self.device
        )?;

        let mut block_ids = Vec::new();
        let mut slot_mapping = Vec::new();
        for i in 0..num_tokens {
            let b_idx = i / block_size;
            let t_idx = i % block_size;
            if t_idx == 0 { block_ids.push(b_idx as u32); }
            slot_mapping.push((b_idx * block_size + t_idx) as i32);
        }

        let block_table = CTensor::new(block_ids.as_slice(), &device)?.reshape((1, block_ids.len()))?;
        let context_lens = CTensor::new(&[num_tokens as i32], &device)?;
        let slot_mapping_t = CTensor::new(slot_mapping.as_slice(), &device)?;

        let logits = self.inner.forward_packed(
            &input_ids,
            &block_table,
            &context_lens,
            &slot_mapping_t,
            &mut pool,
        )?;

        // Return last token's logits
        logits.get(num_tokens - 1).map_err(CoreError::from)
    }

    /// Sequential decode step (standalone mode).
    /// Note: This is an expensive simulation for offline use.
    pub fn decode_step(&mut self, last_token_id: u32) -> Result<CTensor> {
        self.prefill(&[last_token_id])
    }

    /// Greedy decode: argmax of last logits tensor.
    pub fn greedy_token(logits: &CTensor) -> Result<u32> {
        let next_token = logits
            .argmax(candle_core::D::Minus1)
            .map_err(CoreError::from)?;
        let val = next_token
            .to_scalar::<u32>()
            .map_err(CoreError::from)?;
        Ok(val)
    }

    /// Run a full greedy generation sequence.
    pub fn generate_greedy(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        let mut current_ids = prompt_ids.to_vec();
        let mut output = Vec::new();

        for _ in 0..max_new_tokens {
            let logits = self.prefill(&current_ids)?;
            let next_token = Self::greedy_token(&logits)?;
            
            if eos_token_id == Some(next_token) {
                break;
            }
            
            output.push(next_token);
            current_ids.push(next_token);
        }

        Ok(output)
    }
}

impl CausalLM for LlamaModel {
    /// Run one forward step.
    ///
    /// For the purposes of Phase 3, `input_ids` should be a 1-D tensor of
    /// token IDs. This calls `prefill` if the position is 0, else `decode_step`.
    ///
    /// Phase 5 will replace this with a proper packed-batch forward pass.
    fn forward(
        &self,
        _input_ids: &Tensor,
        _positions: &Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // The CausalLM trait's stateless forward() doesn't fit LlamaModel's
        // stateful cache well. Use generate_greedy() or prefill()/decode_step()
        // directly in Phase 3. Phase 5 will reconcile these with the scheduler.
        Err(CoreError::Tensor(
            "Use LlamaModel::generate_greedy() directly in Phase 3. \
             Phase 5 integrates with the scheduler."
                .into(),
        ))
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn device(&self) -> Device {
        self.device
    }
}
