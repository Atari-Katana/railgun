//! Block types for the PagedAttention KV cache.
//!
//! A `BlockId` is an opaque handle to a fixed-size "page" of KV cache memory.
//! `BlockPool` pre-allocates all pages at startup and owns the backing tensor.

use candle_core::Tensor as CTensor;
use vllm_core::{Device, DType};

/// Opaque identifier for a KV cache block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Mask for the CPU block flag (highest bit).
    pub const CPU_FLAG: u32 = 0x8000_0000;

    /// Create a new GPU block ID.
    pub fn gpu(index: u32) -> Self {
        Self(index & !Self::CPU_FLAG)
    }

    /// Create a new CPU block ID.
    pub fn cpu(index: u32) -> Self {
        Self(index | Self::CPU_FLAG)
    }

    /// Returns true if this is a CPU block.
    pub fn is_cpu(&self) -> bool {
        (self.0 & Self::CPU_FLAG) != 0
    }

    /// Returns the raw index within its pool.
    pub fn index(&self) -> u32 {
        self.0 & !Self::CPU_FLAG
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_cpu() {
            write!(f, "CpuBlock({})", self.index())
        } else {
            write!(f, "GpuBlock({})", self.index())
        }
    }
}

/// A shared, pre-allocated pool of all KV cache pages for one GPU.
pub struct GpuBlockPool {
    /// `[num_blocks, 2, num_kv_heads, block_size, head_dim]`
    pub storage: CTensor,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_blocks: usize,
    pub device: Device,
}

impl GpuBlockPool {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: Device,
    ) -> vllm_core::Result<Self> {
        let shape = [num_blocks, 2, num_kv_heads, block_size, head_dim];
        let storage = vllm_core::Tensor::zeros(&shape, dtype, device.clone())?;
        Ok(Self {
            storage: storage.into_inner(),
            block_size,
            num_kv_heads,
            head_dim,
            num_blocks,
            device,
        })
    }

    pub fn k_cache(&self) -> candle_core::Result<CTensor> {
        self.storage.narrow(1, 0, 1)?.squeeze(1)
    }

    pub fn v_cache(&self) -> candle_core::Result<CTensor> {
        self.storage.narrow(1, 1, 1)?.squeeze(1)
    }
}

/// A shared, pre-allocated pool of all KV cache pages in host (CPU) memory.
pub struct CpuBlockPool {
    pub storage: CTensor,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_blocks: usize,
}

impl CpuBlockPool {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> vllm_core::Result<Self> {
        let shape = [num_blocks, 2, num_kv_heads, block_size, head_dim];
        let storage = vllm_core::Tensor::zeros(&shape, dtype, Device::Cpu)?;
        Ok(Self {
            storage: storage.into_inner(),
            block_size,
            num_kv_heads,
            head_dim,
            num_blocks,
        })
    }
}

/// A lightweight reference to a managed block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHandle(pub BlockId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_id_display() {
        assert_eq!(BlockId::gpu(42).to_string(), "GpuBlock(42)");
        assert_eq!(BlockId::cpu(42).to_string(), "CpuBlock(42)");
    }
}
