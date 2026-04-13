use std::sync::Arc;
use vllm_core::{Device as VllmDevice, Result as CoreResult, CoreError};

use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, CudaModule, CudaSlice, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::Ptx;

const PAGED_ATTENTION_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/paged_attention.ptx"));
const PAGED_ATTENTION_V1_PLUS_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/paged_attention_v1_plus.ptx"));
const ROPE_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope.ptx"));

/// Manages the custom CUDA kernels for PagedAttention and RoPE.
#[derive(Debug, Clone)]
pub struct PagedAttentionKernels {
    candle_dev: candle_core::CudaDevice,
    paged_attn_module: Arc<CudaModule>,
    paged_attn_v1_plus_module: Arc<CudaModule>,
    rope_module: Arc<CudaModule>,
}

impl PagedAttentionKernels {
    pub fn new(candle_dev: &candle_core::CudaDevice, ordinal: usize) -> CoreResult<Self> {
        let stream = candle_dev.cuda_stream();
        let context = stream.context();

        let paged_attn_src = std::str::from_utf8(PAGED_ATTENTION_PTX).unwrap();
        let paged_attn_v1_plus_src = std::str::from_utf8(PAGED_ATTENTION_V1_PLUS_PTX).unwrap();
        let rope_src = std::str::from_utf8(ROPE_PTX).unwrap();

        let paged_attn_module = context
            .load_module(Ptx::from_src(paged_attn_src))
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load PagedAttention group: {e}"),
            })?;

        let paged_attn_v1_plus_module = context
            .load_module(Ptx::from_src(paged_attn_v1_plus_src))
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load PagedAttention V1+ group: {e}"),
            })?;

        let rope_module = context
            .load_module(Ptx::from_src(rope_src))
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load RoPE group: {e}"),
            })?;

        Ok(Self {
            candle_dev: candle_dev.clone(),
            paged_attn_module,
            paged_attn_v1_plus_module,
            rope_module,
        })
    }

    pub fn candle_device(&self) -> &candle_core::CudaDevice {
        &self.candle_dev
    }

    pub unsafe fn launch_rope(
        &self,
        x: &mut CudaSlice<f32>,
        cos_sin: &CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        num_heads: i32,
        head_dim: i32,
        num_tokens: i32,
    ) -> CoreResult<()> {
        let stream = self.candle_dev.cuda_stream();
        let func = self.rope_module.load_function("rotary_embedding_kernel").unwrap();
        
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, num_heads as u32, 1),
            block_dim: ((head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&func);
        builder.arg(x).arg(cos_sin).arg(positions).arg(&num_heads).arg(&head_dim);
        builder.launch(cfg)
            .map_err(|e| CoreError::Tensor(format!("RoPE launch error: {e}")))?;

        Ok(())
    }

    pub unsafe fn launch_reshape_and_cache(
        &self,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
        slot_mapping: &CudaSlice<i32>,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        batch_size: i32,
    ) -> CoreResult<()> {
        let stream = self.candle_dev.cuda_stream();
        let func = self.paged_attn_module.load_function("reshape_and_cache").unwrap();
        
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, num_kv_heads as u32, head_dim as u32),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&func);
        builder.arg(k).arg(v).arg(k_cache).arg(v_cache).arg(slot_mapping)
            .arg(&num_kv_heads).arg(&head_dim).arg(&block_size);
        builder.launch(cfg)
            .map_err(|e| CoreError::Tensor(format!("CUDA launch error: {e}")))?;

        Ok(())
    }

    pub unsafe fn launch_v1(
        &self,
        query: &CudaSlice<f32>,
        key_cache: &CudaSlice<f32>,
        value_cache: &CudaSlice<f32>,
        block_table: &CudaSlice<i32>,
        context_lens: &CudaSlice<i32>,
        output: &mut CudaSlice<f32>,
        scale: f32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        max_blocks_per_seq: i32,
        batch_size: i32,
    ) -> CoreResult<()> {
        let stream = self.candle_dev.cuda_stream();
        let func = self.paged_attn_module.load_function("paged_attention_v1").unwrap();

        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, num_heads as u32, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&func);
        builder.arg(query).arg(key_cache).arg(value_cache).arg(block_table).arg(context_lens)
            .arg(output).arg(&scale).arg(&num_heads).arg(&num_kv_heads).arg(&head_dim)
            .arg(&block_size).arg(&max_blocks_per_seq);
        builder.launch(cfg)
            .map_err(|e| CoreError::Tensor(format!("CUDA launch error: {e}")))?;

        Ok(())
    }

    pub unsafe fn launch_v1_plus(
        &self,
        query: &CudaSlice<f32>,
        key_cache: &CudaSlice<f32>,
        value_cache: &CudaSlice<f32>,
        block_table: &CudaSlice<i32>,
        context_lens: &CudaSlice<i32>,
        output: &mut CudaSlice<f32>,
        scale: f32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        max_blocks_per_seq: i32,
        batch_size: i32,
    ) -> CoreResult<()> {
        let stream = self.candle_dev.cuda_stream();
        let func = self.paged_attn_v1_plus_module.load_function("paged_attention_v1_plus").unwrap();

        let cfg = LaunchConfig {
            grid_dim: ((batch_size * num_heads) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (head_dim as u32 * 4), // 4 bytes per float for block reduction
        };

        let mut builder = stream.launch_builder(&func);
        builder.arg(query).arg(key_cache).arg(value_cache).arg(block_table).arg(context_lens)
            .arg(output).arg(&scale).arg(&num_heads).arg(&num_kv_heads).arg(&head_dim)
            .arg(&block_size).arg(&max_blocks_per_seq);
        builder.launch(cfg)
            .map_err(|e| CoreError::Tensor(format!("CUDA launch error: {e}")))?;

        Ok(())
    }
}
pub mod paged_attn_op;
pub use paged_attn_op::PagedAttentionOp;
