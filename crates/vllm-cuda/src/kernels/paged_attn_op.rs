//! Custom Operator for PagedAttention.
//!
//! This module implements the `candle_core::op::CustomOp` trait, allowing
//! Railgun to plug its hand-optimized CUDA kernels directly into the 
//! Candle computation graph.

use candle_core::{CudaDevice, CudaStorage, Device, Layout, Result, Shape, Tensor, CpuStorage};
use crate::kernels::PagedAttentionKernels;

/// The PagedAttention Custom Operation.
pub struct PagedAttentionOp {
    pub scale: f32,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub kernels: std::sync::Arc<PagedAttentionKernels>,
}

impl candle_core::CustomOp1 for PagedAttentionOp {
    fn name(&self) -> &'static str {
        "paged_attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg("paged_attention is only implemented for CUDA".to_string()))
    }

    fn cuda_fwd(&self, _: &CudaStorage, _: &Layout) -> Result<(CudaStorage, Shape)> {
        // We use PagedAttentionOp::execute instead for better multi-input support
        Err(candle_core::Error::Msg("Use PagedAttentionOp::execute instead".to_string()))
    }
}

impl PagedAttentionOp {
    /// Execute the paged attention kernel.
    /// 
    /// # Arguments
    /// * `q` - Query tensor [batch, num_heads, head_dim]
    /// * `k_cache` - Key cache [num_blocks, num_kv_heads, block_size, head_dim]
    /// * `v_cache` - Value cache [num_blocks, num_kv_heads, block_size, head_dim]
    /// * `block_table` - Block indices [batch, max_blocks]
    /// * `context_lens` - Sequence lengths [batch]
    pub fn execute(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
    ) -> Result<Tensor> {
        let device = q.device();
        let (batch_size, num_heads, head_dim) = q.dims3()?;
        let (_, num_kv_heads, block_size, _) = k_cache.dims4()?;
        let max_blocks = block_table.dim(1)?;

        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, _) = k_cache.storage_and_layout();
        let (v_storage, _) = v_cache.storage_and_layout();
        let (bt_storage, _) = block_table.storage_and_layout();
        let (cl_storage, _) = context_lens.storage_and_layout();

        match (&*q_storage, &*k_storage, &*v_storage, &*bt_storage, &*cl_storage) {
            (
                candle_core::Storage::Cuda(q_cuda),
                candle_core::Storage::Cuda(k_cuda),
                candle_core::Storage::Cuda(v_cuda),
                candle_core::Storage::Cuda(bt_cuda),
                candle_core::Storage::Cuda(cl_cuda),
            ) => {
                let dev = self.kernels.device();
                let out_shape = Shape::from((batch_size, num_heads, head_dim));
                let out_size = out_shape.elem_count();
                let mut out_cuda = dev.alloc_zeros::<f32>(out_size).map_err(candle_core::Error::from)?;

                unsafe {
                    self.kernels.launch_v1(
                        q_cuda.as_cuda_slice::<f32>()?,
                        k_cuda.as_cuda_slice::<f32>()?,
                        v_cuda.as_cuda_slice::<f32>()?,
                        bt_cuda.as_cuda_slice::<i32>()?,
                        cl_cuda.as_cuda_slice::<i32>()?,
                        &mut out_cuda,
                        self.scale,
                        num_heads as i32,
                        self.num_kv_heads as i32,
                        head_dim as i32,
                        block_size as i32,
                        max_blocks as i32,
                        batch_size as i32,
                    ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                }

                let new_storage = candle_core::CudaStorage::wrap_cuda_slice(out_cuda, dev.clone());
                Ok(Tensor::from_storage(candle_core::Storage::Cuda(new_storage), out_shape, candle_core::op::BackpropOp::none(), false))
            }
            _ => Err(candle_core::Error::Msg("PagedAttentionOp requires CUDA tensors".to_string())),
        }
    }
    /// Store new K and V tokens into the block pool.
    pub fn reshape_and_cache(
        &self,
        k: &Tensor,
        v: &Tensor,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        let (batch_size, num_kv_heads, head_dim) = k.dims3()?;
        let block_size = self.block_size;

        let (k_storage, _) = k.storage_and_layout();
        let (v_storage, _) = v.storage_and_layout();
        let (kc_storage, _) = k_cache.storage_and_layout();
        let (vc_storage, _) = v_cache.storage_and_layout();
        let (slot_storage, _) = slot_mapping.storage_and_layout();

        match (&*k_storage, &*v_storage, &*kc_storage, &*vc_storage, &*slot_storage) {
            (
                candle_core::Storage::Cuda(k_cuda),
                candle_core::Storage::Cuda(v_cuda),
                candle_core::Storage::Cuda(kc_cuda),
                candle_core::Storage::Cuda(vc_cuda),
                candle_core::Storage::Cuda(slot_cuda),
            ) => {
                let mut kc_slice = kc_cuda.as_cuda_slice::<f32>()?.clone();
                let mut vc_slice = vc_cuda.as_cuda_slice::<f32>()?.clone();
                
                unsafe {
                    self.kernels.launch_reshape_and_cache(
                        k_cuda.as_cuda_slice::<f32>()?,
                        v_cuda.as_cuda_slice::<f32>()?,
                        &mut kc_slice,
                        &mut vc_slice,
                        slot_cuda.as_cuda_slice::<i32>()?,
                        num_kv_heads as i32,
                        head_dim as i32,
                        block_size as i32,
                        batch_size as i32,
                    ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                }
                Ok(())
            }
            _ => Err(candle_core::Error::Msg("reshape_and_cache requires CUDA tensors".to_string())),
        }
    }

    /// Apply Rotary Positional Embeddings to the given tensor.
    pub fn apply_rope(
        &self,
        x: &mut Tensor,
        cos_sin: &Tensor,
        positions: &Tensor,
    ) -> Result<()> {
        let (num_tokens, num_heads, head_dim) = x.dims3()?;
        let (x_storage, _) = x.storage_and_layout();
        let (cs_storage, _) = cos_sin.storage_and_layout();
        let (pos_storage, _) = positions.storage_and_layout();

        match (&*x_storage, &*cs_storage, &*pos_storage) {
            (
                candle_core::Storage::Cuda(x_cuda),
                candle_core::Storage::Cuda(cs_cuda),
                candle_core::Storage::Cuda(pos_cuda),
            ) => {
                let mut x_slice = x_cuda.as_cuda_slice::<f32>()?.clone();
                unsafe {
                    self.kernels.launch_rope(
                        &mut x_slice,
                        cs_cuda.as_cuda_slice::<f32>()?,
                        pos_cuda.as_cuda_slice::<i32>()?,
                        num_heads as i32,
                        head_dim as i32,
                        num_tokens as i32,
                    ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                }
                Ok(())
            }
            _ => Err(candle_core::Error::Msg("apply_rope requires CUDA tensors".to_string())),
        }
    }
}
