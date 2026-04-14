use std::sync::Arc;
use vllm_core::{CoreError, Device as VllmDevice, Result as CoreResult};

use candle_core::cuda_backend::cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg,
};
use candle_core::cuda_backend::cudarc::nvrtc::Ptx;

const PAGED_ATTENTION_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/paged_attention.ptx"));
const PAGED_ATTENTION_V1_PLUS_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/paged_attention_v1_plus.ptx"));
const PAGED_ATTENTION_V2_PARTITION_PTX: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/paged_attention_v2_partition.ptx"
));
const PAGED_ATTENTION_V2_REDUCE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/paged_attention_v2_reduce.ptx"));
const ROPE_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope.ptx"));

const MAX_HEAD_DIM_PAGED_ATTENTION: i32 = 256;
const MAX_HEAD_DIM_V2_REDUCE: i32 = 1024;

/// Manages the custom CUDA kernels for PagedAttention and RoPE.
#[derive(Debug, Clone)]
pub struct PagedAttentionKernels {
    candle_dev: candle_core::CudaDevice,
    _paged_attn_module: Arc<CudaModule>,
    _paged_attn_v1_plus_module: Arc<CudaModule>,
    _paged_attn_v2_partition_module: Arc<CudaModule>,
    _paged_attn_v2_reduce_module: Arc<CudaModule>,
    _rope_module: Arc<CudaModule>,
    // Cached functions
    paged_attn_v1_func: CudaFunction,
    paged_attn_v1_plus_func: CudaFunction,
    paged_attn_v2_partition_func: CudaFunction,
    paged_attn_v2_reduce_func: CudaFunction,
    reshape_and_cache_func: CudaFunction,
    reshape_and_cache_isoquant_func: CudaFunction,
    rope_func: CudaFunction,
}

fn check_paged_attention_head_dim(head_dim: i32) -> CoreResult<()> {
    if head_dim % 4 != 0 || head_dim > MAX_HEAD_DIM_PAGED_ATTENTION {
        Err(CoreError::NotSupported {
            feature: "PagedAttention",
            reason: format!(
                "head_dim must be a multiple of 4 and <= {}, but got {}",
                MAX_HEAD_DIM_PAGED_ATTENTION, head_dim
            ),
        })
    } else {
        Ok(())
    }
}

impl PagedAttentionKernels {
    pub fn new(candle_dev: &candle_core::CudaDevice, ordinal: usize) -> CoreResult<Self> {
        let stream = candle_dev.cuda_stream();
        let context = stream.context();

        let paged_attn_src =
            std::str::from_utf8(PAGED_ATTENTION_PTX).map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load PagedAttention PTX: {e}"),
            })?;
        let paged_attn_v1_plus_src = std::str::from_utf8(PAGED_ATTENTION_V1_PLUS_PTX).map_err(
            |e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load PagedAttention V1+ PTX: {e}"),
            },
        )?;
        let paged_attn_v2_partition_src =
            std::str::from_utf8(PAGED_ATTENTION_V2_PARTITION_PTX).map_err(|e| {
                CoreError::DeviceInit {
                    device: VllmDevice::Cuda(ordinal as u32),
                    reason: format!("Failed to load PagedAttention V2 Partition PTX: {e}"),
                }
            })?;
        let paged_attn_v2_reduce_src = std::str::from_utf8(PAGED_ATTENTION_V2_REDUCE_PTX)
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load PagedAttention V2 Reduce PTX: {e}"),
            })?;
        let rope_src = std::str::from_utf8(ROPE_PTX).map_err(|e| CoreError::DeviceInit {
            device: VllmDevice::Cuda(ordinal as u32),
            reason: format!("Failed to load RoPE PTX: {e}"),
        })?;

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

        let paged_attn_v2_partition_module = context
            .load_module(Ptx::from_src(paged_attn_v2_partition_src))
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load PagedAttention V2 partition group: {e}"),
            })?;

        let paged_attn_v2_reduce_module = context
            .load_module(Ptx::from_src(paged_attn_v2_reduce_src))
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load PagedAttention V2 reduce group: {e}"),
            })?;

        let rope_module =
            context
                .load_module(Ptx::from_src(rope_src))
                .map_err(|e| CoreError::DeviceInit {
                    device: VllmDevice::Cuda(ordinal as u32),
                    reason: format!("Failed to load RoPE group: {e}"),
                })?;

        let paged_attn_v1_func = paged_attn_module
            .load_function("paged_attention_v1")
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load paged_attention_v1 function: {e}"),
            })?;

        let paged_attn_v1_plus_func = paged_attn_v1_plus_module
            .load_function("paged_attention_v1_plus")
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load paged_attention_v1_plus function: {e}"),
            })?;

        let paged_attn_v2_partition_func = paged_attn_v2_partition_module
            .load_function("paged_attention_v2_partition")
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load paged_attention_v2_partition function: {e}"),
            })?;

        let paged_attn_v2_reduce_func = paged_attn_v2_reduce_module
            .load_function("paged_attention_v2_reduce")
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load paged_attention_v2_reduce function: {e}"),
            })?;

        let reshape_and_cache_func = paged_attn_module
            .load_function("reshape_and_cache")
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load reshape_and_cache function: {e}"),
            })?;

        let reshape_and_cache_isoquant_func = paged_attn_module
            .load_function("reshape_and_cache_isoquant")
            .map_err(|e| CoreError::DeviceInit {
                device: VllmDevice::Cuda(ordinal as u32),
                reason: format!("Failed to load reshape_and_cache_isoquant function: {e}"),
            })?;

        let rope_func =
            rope_module
                .load_function("rotary_embedding_kernel")
                .map_err(|e| CoreError::DeviceInit {
                    device: VllmDevice::Cuda(ordinal as u32),
                    reason: format!("Failed to load rotary_embedding_kernel function: {e}"),
                })?;

        Ok(Self {
            candle_dev: candle_dev.clone(),
            _paged_attn_module: paged_attn_module,
            _paged_attn_v1_plus_module: paged_attn_v1_plus_module,
            _paged_attn_v2_partition_module: paged_attn_v2_partition_module,
            _paged_attn_v2_reduce_module: paged_attn_v2_reduce_module,
            _rope_module: rope_module,
            paged_attn_v1_func,
            paged_attn_v1_plus_func,
            paged_attn_v2_partition_func,
            paged_attn_v2_reduce_func,
            reshape_and_cache_func,
            reshape_and_cache_isoquant_func,
            rope_func,
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

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, num_heads as u32, 1),
            block_dim: ((head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&self.rope_func);
        builder
            .arg(x)
            .arg(cos_sin)
            .arg(positions)
            .arg(&num_heads)
            .arg(&head_dim);
        builder
            .launch(cfg)
            .map_err(|e| CoreError::Tensor(format!("RoPE kernel launch failed: {e}")))?;

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

        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, num_kv_heads as u32, head_dim as u32),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&self.reshape_and_cache_func);
        builder
            .arg(k)
            .arg(v)
            .arg(k_cache)
            .arg(v_cache)
            .arg(slot_mapping)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .arg(&block_size);
        builder.launch(cfg).map_err(|e| {
            CoreError::Tensor(format!("reshape_and_cache kernel launch failed: {e}"))
        })?;

        Ok(())
    }

    pub unsafe fn launch_reshape_and_cache_isoquant(
        &self,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
        slot_mapping: &CudaSlice<i32>,
        rotation_metadata: &mut CudaSlice<f32>,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        batch_size: i32,
    ) -> CoreResult<()> {
        let stream = self.candle_dev.cuda_stream();

        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: ((head_dim / 4) as u32, num_kv_heads as u32, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&self.reshape_and_cache_isoquant_func);
        builder
            .arg(k)
            .arg(v)
            .arg(k_cache)
            .arg(v_cache)
            .arg(slot_mapping)
            .arg(rotation_metadata)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .arg(&block_size);
        builder.launch(cfg).map_err(|e| {
            CoreError::Tensor(format!(
                "reshape_and_cache_isoquant kernel launch failed: {e}"
            ))
        })?;

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

        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, num_heads as u32, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&self.paged_attn_v1_func);
        builder
            .arg(query)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_table)
            .arg(context_lens)
            .arg(output)
            .arg(&scale)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .arg(&block_size)
            .arg(&max_blocks_per_seq);
        builder.launch(cfg).map_err(|e| {
            CoreError::Tensor(format!("paged_attention_v1 kernel launch failed: {e}"))
        })?;

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
        rotation_metadata: &CudaSlice<f32>,
        scale: f32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        max_blocks_per_seq: i32,
        batch_size: i32,
    ) -> CoreResult<()> {
        check_paged_attention_head_dim(head_dim)?;
        let stream = self.candle_dev.cuda_stream();

        let block_size_x = ((head_dim + 31) / 32 * 32) as u32;
        let cfg = LaunchConfig {
            grid_dim: ((batch_size * num_heads) as u32, 1, 1),
            block_dim: (block_size_x, 1, 1),
            shared_mem_bytes: (((head_dim + 31) / 32) * 4) as u32, // Enough for warp partial sums
        };

        let mut builder = stream.launch_builder(&self.paged_attn_v1_plus_func);
        builder
            .arg(query)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_table)
            .arg(context_lens)
            .arg(output)
            .arg(rotation_metadata)
            .arg(&scale)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .arg(&block_size)
            .arg(&max_blocks_per_seq);
        builder.launch(cfg).map_err(|e| {
            CoreError::Tensor(format!(
                "paged_attention_v1_plus kernel launch failed: {e}"
            ))
        })?;

        Ok(())
    }

    pub unsafe fn launch_v2_partition(
        &self,
        query: &CudaSlice<f32>,
        key_cache: &CudaSlice<f32>,
        value_cache: &CudaSlice<f32>,
        block_table: &CudaSlice<i32>,
        context_lens: &CudaSlice<i32>,
        tmp_out: &mut CudaSlice<f32>,
        exp_sums: &mut CudaSlice<f32>,
        max_logits: &mut CudaSlice<f32>,
        rotation_metadata: &CudaSlice<f32>,
        scale: f32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        max_blocks_per_seq: i32,
        num_partitions: i32,
        batch_size: i32,
    ) -> CoreResult<()> {
        check_paged_attention_head_dim(head_dim)?;
        let stream = self.candle_dev.cuda_stream();

        let block_size_x = ((head_dim + 31) / 32 * 32) as u32;
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, num_heads as u32, num_partitions as u32),
            block_dim: (block_size_x, 1, 1),
            shared_mem_bytes: (((head_dim + 31) / 32) * 4) as u32,
        };

        let mut builder = stream.launch_builder(&self.paged_attn_v2_partition_func);
        builder
            .arg(query)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_table)
            .arg(context_lens)
            .arg(tmp_out)
            .arg(exp_sums)
            .arg(max_logits)
            .arg(rotation_metadata)
            .arg(&scale)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .arg(&block_size)
            .arg(&max_blocks_per_seq)
            .arg(&num_partitions);
        builder.launch(cfg).map_err(|e| {
            CoreError::Tensor(format!(
                "paged_attention_v2_partition kernel launch failed: {e}"
            ))
        })?;

        Ok(())
    }

    pub unsafe fn launch_v2_reduce(
        &self,
        output: &mut CudaSlice<f32>,
        exp_sums: &CudaSlice<f32>,
        max_logits: &CudaSlice<f32>,
        tmp_out: &CudaSlice<f32>,
        context_lens: &CudaSlice<i32>,
        rotation_metadata: &CudaSlice<f32>,
        num_heads: i32,
        head_dim: i32,
        num_partitions: i32,
        batch_size: i32,
    ) -> CoreResult<()> {
        let stream = self.candle_dev.cuda_stream();

        // PARTITION_SIZE = 256 must match the value used in the CUDA kernels
        // paged_attention_v2_partition.cu and paged_attention_v2_reduce.cu
        const _PARTITION_SIZE: i32 = 256;

        if head_dim > MAX_HEAD_DIM_V2_REDUCE {
            return Err(CoreError::Tensor(format!(
                "head_dim {} exceeds maximum block size {} for reduction kernel",
                head_dim, MAX_HEAD_DIM_V2_REDUCE
            )));
        }

        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, num_heads as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&self.paged_attn_v2_reduce_func);
        builder
            .arg(output)
            .arg(exp_sums)
            .arg(max_logits)
            .arg(tmp_out)
            .arg(context_lens)
            .arg(rotation_metadata)
            .arg(&num_heads)
            .arg(&head_dim)
            .arg(&num_partitions);
        builder.launch(cfg).map_err(|e| {
            CoreError::Tensor(format!(
                "paged_attention_v2_reduce kernel launch failed: {e}"
            ))
        })?;

        Ok(())
    }
}
pub mod paged_attn_op;
pub use paged_attn_op::PagedAttentionOp;
