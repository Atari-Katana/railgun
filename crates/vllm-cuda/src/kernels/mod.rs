//! CUDA kernel loaders and launchers.
//!
//! Hand-written CUDA kernels (compiled to PTX at build time) are loaded 
//! into the current [`CudaContext`] and executed via `cudarc`.

use std::sync::Arc;
use cudarc::driver::{CudaDevice, LaunchConfig};
use vllm_core::{CoreError, Result, Device as VllmDevice};

/// The PTX content for the PagedAttention kernels, compiled by build.rs.
pub const PAGED_ATTENTION_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/paged_attention.ptx"));
pub const ROPE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/rope.ptx"));

/// Interface to optimized CUDA kernels for Railgun.
pub struct PagedAttentionKernels {
    device: Arc<CudaDevice>,
}

impl PagedAttentionKernels {
    pub fn new(device: Arc<CudaDevice>, ordinal: usize) -> Result<Self> {
        // Load the Paged Attention PTX
        device
            .load_ptx(PAGED_ATTENTION_PTX.into(), "paged_attn", &[
                "paged_attention_v1",
                "reshape_and_cache"
            ])
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load PagedAttention PTX: {e}"),
            })?;

        // Load the RoPE PTX
        device
            .load_ptx(ROPE_PTX.into(), "rope_kernels", &["rotary_embedding_kernel"])
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load RoPE PTX: {e}"),
            })?;

        Ok(Self { device })
    }

    /// Launch the RoPE kernel.
    pub unsafe fn launch_rope(
        &self,
        x: &mut cudarc::driver::CudaSlice<f32>,
        cos_sin: &cudarc::driver::CudaSlice<f32>,
        positions: &cudarc::driver::CudaSlice<i32>,
        num_heads: i32,
        head_dim: i32,
        num_tokens: i32,
    ) -> Result<()> {
        let func = self.device.get_func("rope_kernels", "rotary_embedding_kernel").unwrap();
        
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, num_heads as u32, 1),
            block_dim: ((head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let params = (x, cos_sin, positions, num_heads, head_dim);
        func.launch(cfg, params)
            .map_err(|e| CoreError::Tensor(format!("RoPE launch error: {e}")))?;

        Ok(())
    }

    /// Launch the reshape_and_cache kernel.
    pub unsafe fn launch_reshape_and_cache(
        &self,
        k: &cudarc::driver::CudaSlice<f32>,
        v: &cudarc::driver::CudaSlice<f32>,
        k_cache: &mut cudarc::driver::CudaSlice<f32>,
        v_cache: &mut cudarc::driver::CudaSlice<f32>,
        slot_mapping: &cudarc::driver::CudaSlice<i32>,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        batch_size: i32,
    ) -> Result<()> {
        let func = self.device.get_func("paged_attn", "reshape_and_cache").unwrap();
        
        // grid = (batch_size, 1, 1)
        // block = (1, num_kv_heads, head_dim) - Simplify: one thread per dimension
        // Note: For large head_dim, this should be tuned.
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, num_kv_heads as u32, head_dim as u32),
            shared_mem_bytes: 0,
        };

        let params = (
            k, v, k_cache, v_cache, slot_mapping,
            num_kv_heads, head_dim, block_size
        );

        func.launch(cfg, params)
            .map_err(|e| CoreError::Tensor(format!("CUDA launch error: {e}")))?;

        Ok(())
    }

    /// Launch the paged_attention_v1 kernel.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_v1(
        &self,
        query: &cudarc::driver::CudaSlice<f32>,
        key_cache: &cudarc::driver::CudaSlice<f32>,
        value_cache: &cudarc::driver::CudaSlice<f32>,
        block_table: &cudarc::driver::CudaSlice<i32>,
        context_lens: &cudarc::driver::CudaSlice<i32>,
        output: &mut cudarc::driver::CudaSlice<f32>,
        scale: f32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        max_blocks_per_seq: i32,
        batch_size: i32,
    ) -> Result<()> {
        let func = self.device.get_func("paged_attn", "paged_attention_v1").unwrap();

        // grid = (batch_size, 1, 1)
        // block = (1, num_heads, 1) - One thread per head in V1
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, num_heads as u32, 1),
            shared_mem_bytes: 0,
        };

        let params = (
            query,
            key_cache,
            value_cache,
            block_table,
            context_lens,
            output,
            scale,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
        );

        func.launch(cfg, params)
            .map_err(|e| CoreError::Tensor(format!("CUDA launch error: {e}")))?;

        Ok(())
    }
}
pub mod paged_attn_op;
pub use paged_attn_op::PagedAttentionOp;
