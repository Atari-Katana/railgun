//! KV cache — provides a per-request view into the shared block pool.

use super::block::BlockId;

/// A per-request handle to KV cache blocks.
///
/// Each request owns a sequence of [`BlockId`]s — one per `ceil(num_tokens /
/// block_size)` blocks. Attention reads and writes go through this view.
#[derive(Debug, Default)]
pub struct KVCache {
    /// Ordered list of block IDs assigned to this request.
    /// Grows as the sequence length increases.
    pub block_table: Vec<BlockId>,
}

impl KVCache {
    /// Create an empty KV cache for a new request.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a newly allocated block to this request's block table.
    pub fn push_block(&mut self, id: BlockId) {
        self.block_table.push(id);
    }

    /// Number of blocks currently assigned.
    pub fn num_blocks(&self) -> usize {
        self.block_table.len()
    }

    /// Capacity in tokens given the pool's block size.
    pub fn token_capacity(&self, block_size: usize) -> usize {
        self.block_table.len() * block_size
    }

    /// Return the block table as a flat slice for kernel dispatch.
    pub fn as_block_ids(&self) -> &[BlockId] {
        &self.block_table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_query() {
        let mut kv = KVCache::new();
        assert_eq!(kv.num_blocks(), 0);
        kv.push_block(BlockId(0));
        kv.push_block(BlockId(1));
        assert_eq!(kv.num_blocks(), 2);
        assert_eq!(kv.token_capacity(16), 32);
        assert_eq!(kv.as_block_ids(), &[BlockId(0), BlockId(1)]);
    }
}
