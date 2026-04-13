//! Block types for the PagedAttention KV cache.
//!
//! A `BlockId` is an opaque handle to a fixed-size "page" of KV cache memory.
//! `BlockPool` pre-allocates all pages at startup and owns the backing tensor.

use candle_core::Tensor as CTensor;
use vllm_core::{Device, DType};

/// Opaque identifier for a KV cache block.
///
/// IDs are stable within a single process lifetime.
/// Do not serialise or compare across processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Block({})", self.0)
    }
}

/// A shared, pre-allocated pool of all KV cache pages for one GPU.
///
/// The backing storage is a single contiguous tensor with shape
/// `[num_blocks, 2, num_kv_heads, block_size, head_dim]` where the
/// second dimension indexes K or V.
pub struct BlockPool {
    /// `[num_blocks, 2, num_kv_heads, block_size, head_dim]`
    pub storage: CTensor,
    /// Number of token slots per page.
    pub block_size: usize,
    /// Number of KV attention head groups.
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Total number of pages.
    pub num_blocks: usize,
    /// Device where the pool lives.
    pub device: Device,
}

impl BlockPool {
    /// Allocate the KV cache pool.
    ///
    /// # Arguments
    ///
    /// - `num_blocks` – Total pages in the pool.
    /// - `block_size` – Tokens per page (default 16 in production).
    /// - `num_kv_heads` – Number of KV attention heads.
    /// - `head_dim` – Dimension of each attention head.
    /// - `dtype` – dtype for KV tensors (typically BF16 or FP16).
    /// - `device` – Target device.
    ///
    /// # Errors
    ///
    /// Returns `candle_core::Error` if the allocation fails.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: Device,
    ) -> vllm_core::Result<Self> {
        let shape = [num_blocks, 2, num_kv_heads, block_size, head_dim];
        let storage = vllm_core::Tensor::zeros(&shape, dtype, device)?;
        Ok(Self {
            storage: storage.into_inner(),
            block_size,
            num_kv_heads,
            head_dim,
            num_blocks,
            device,
        })
    }

    /// Bytes allocated by the pool.
    pub fn byte_size(&self) -> usize {
        self.num_blocks * 2 * self.num_kv_heads * self.block_size * self.head_dim * 2
        // assumes BF16/F16 (2 bytes); for correctness callers should track dtype
    }

    /// Returns a view of the Key cache.
    pub fn k_cache(&self) -> candle_core::Result<CTensor> {
        self.storage.narrow(1, 0, 1)?.squeeze(1)
    }

    /// Returns a view of the Value cache.
    pub fn v_cache(&self) -> candle_core::Result<CTensor> {
        self.storage.narrow(1, 1, 1)?.squeeze(1)
    }
}

/// A lightweight reference to a managed block.
///
/// Returned by [`BlockAllocator::allocate`]. The allocator tracks live
/// blocks separately; `BlockHandle` is just a typed ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHandle(pub BlockId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_id_display() {
        assert_eq!(BlockId(42).to_string(), "Block(42)");
    }

    #[test]
    fn pool_creation_cpu() {
        let pool = BlockPool::new(
            /*num_blocks=*/ 128,
            /*block_size=*/ 16,
            /*num_kv_heads=*/ 8,
            /*head_dim=*/ 128,
            DType::F32,
            Device::Cpu,
        )
        .unwrap();
        assert_eq!(pool.num_blocks, 128);
        assert_eq!(pool.block_size, 16);
        assert_eq!(pool.storage.dims(), &[128, 2, 8, 16, 128]);
    }
}
