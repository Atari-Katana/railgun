//! # vllm-paged-attention
//!
//! PagedAttention block allocator and KV cache management for Railgun.

pub mod allocator;
pub mod block;
pub mod kv_cache;

pub use allocator::BlockAllocator;
pub use block::{BlockHandle, BlockId, BlockPool};
pub use kv_cache::KVCache;
