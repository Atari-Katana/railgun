//! Block allocator for the KV cache.
//!
//! [`BlockAllocator`] manages a pool of fixed-size blocks.
//! It hands out [`BlockId`]s to callers and reclaims them on demand.
//! Reference-counting supports copy-on-write prefix caching.

use std::collections::VecDeque;
use std::sync::Arc;

use thiserror::Error;
use tracing::trace;

use super::block::{BlockHandle, BlockId, BlockPool};

/// Errors from the block allocator.
#[derive(Debug, Error)]
pub enum AllocError {
    /// No free blocks remain in the pool.
    #[error("KV cache pool exhausted: all {total} blocks are in use")]
    OutOfBlocks { total: usize },

    /// An invalid block ID was passed to a method.
    #[error("invalid block id {id:?}")]
    InvalidBlock { id: BlockId },
}

/// Manages allocation and reclamation of [`BlockId`]s.
///
/// All methods take `&mut self` to ensure single-threaded access within
/// the scheduler loop. The scheduler itself runs on a single Tokio task
/// for the hot path, so this is appropriate.
pub struct BlockAllocator {
    pool: Arc<BlockPool>,
    free: VecDeque<BlockId>,
    ref_counts: Vec<u32>,
}

impl BlockAllocator {
    /// Create an allocator over the given pool.
    ///
    /// Initially all blocks are free.
    pub fn new(pool: Arc<BlockPool>) -> Self {
        let num_blocks = pool.num_blocks;
        let free = (0..num_blocks as u32).map(BlockId).collect();
        let ref_counts = vec![0u32; num_blocks];
        Self {
            pool,
            free,
            ref_counts,
        }
    }

    /// Allocate one block, returning its ID.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError::OutOfBlocks`] when the pool is exhausted.
    pub fn allocate(&mut self) -> Result<BlockHandle, AllocError> {
        let id = self.free.pop_front().ok_or(AllocError::OutOfBlocks {
            total: self.pool.num_blocks,
        })?;
        debug_assert_eq!(self.ref_counts[id.0 as usize], 0);
        self.ref_counts[id.0 as usize] = 1;
        trace!(%id, "allocated block");
        Ok(BlockHandle(id))
    }

    /// Decrement the reference count of a block.
    ///
    /// When the count reaches zero, the block is returned to the free pool.
    ///
    /// # Panics
    ///
    /// Panics if `id` has a zero reference count (double-free), as this
    /// indicates an invariant violation in the calling code.
    pub fn free(&mut self, id: BlockId) {
        let rc = &mut self.ref_counts[id.0 as usize];
        assert!(*rc > 0, "double-free of {id}");
        *rc -= 1;
        if *rc == 0 {
            self.free.push_back(id);
            trace!(%id, "freed block");
        }
    }

    /// Increment the reference count (copy-on-write share).
    ///
    /// Used for prefix caching: two requests point to the same immutable
    /// block until one writes to it, at which point it must fork.
    ///
    /// # Panics
    ///
    /// Panics if the block is currently free (ref_count == 0).
    pub fn fork(&mut self, id: BlockId) -> BlockHandle {
        let rc = &mut self.ref_counts[id.0 as usize];
        assert!(*rc > 0, "cannot fork free block {id}");
        *rc += 1;
        BlockHandle(id)
    }

    /// Number of free blocks remaining.
    pub fn num_free(&self) -> usize {
        self.free.len()
    }

    /// Number of allocated (in-use) blocks.
    pub fn num_used(&self) -> usize {
        self.pool.num_blocks - self.free.len()
    }

    /// Total capacity.
    pub fn capacity(&self) -> usize {
        self.pool.num_blocks
    }

    /// Reference to the backing pool.
    pub fn pool(&self) -> &Arc<BlockPool> {
        &self.pool
    }

    /// Mutably access the backing pool for model updates.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other thread is accessing the pool's
    /// memory (especially on the GPU side) during the model step.
    pub unsafe fn pool_mut(&self) -> &mut BlockPool {
        let ptr = Arc::as_ptr(&self.pool) as *mut BlockPool;
        &mut *ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockPool;
    use vllm_core::{Device, DType};

    fn make_allocator(n: usize) -> BlockAllocator {
        let pool = BlockPool::new(n, 16, 4, 64, DType::F32, Device::Cpu).unwrap();
        BlockAllocator::new(Arc::new(pool))
    }

    #[test]
    fn initial_state() {
        let alloc = make_allocator(64);
        assert_eq!(alloc.num_free(), 64);
        assert_eq!(alloc.num_used(), 0);
        assert_eq!(alloc.capacity(), 64);
    }

    #[test]
    fn allocate_and_free() {
        let mut alloc = make_allocator(4);
        let h0 = alloc.allocate().unwrap();
        let h1 = alloc.allocate().unwrap();
        assert_eq!(alloc.num_free(), 2);
        alloc.free(h0.0);
        assert_eq!(alloc.num_free(), 3);
        alloc.free(h1.0);
        assert_eq!(alloc.num_free(), 4);
    }

    #[test]
    fn exhaustion_returns_error() {
        let mut alloc = make_allocator(2);
        let _a = alloc.allocate().unwrap();
        let _b = alloc.allocate().unwrap();
        let err = alloc.allocate().unwrap_err();
        assert!(matches!(err, AllocError::OutOfBlocks { total: 2 }));
    }

    #[test]
    fn fork_increases_ref_count() {
        let mut alloc = make_allocator(4);
        let h = alloc.allocate().unwrap();
        let h2 = alloc.fork(h.0);
        // Freeing once should not return to pool
        alloc.free(h.0);
        assert_eq!(alloc.num_free(), 3); // only 3 free, h still has ref_count=1
        alloc.free(h2.0);
        assert_eq!(alloc.num_free(), 4); // now all free
    }

    #[test]
    #[should_panic(expected = "double-free")]
    fn double_free_panics() {
        let mut alloc = make_allocator(4);
        let h = alloc.allocate().unwrap();
        alloc.free(h.0);
        alloc.free(h.0); // should panic
    }

    #[test]
    fn reuse_after_free() {
        let mut alloc = make_allocator(2);
        let h0 = alloc.allocate().unwrap();
        let h1 = alloc.allocate().unwrap();
        alloc.free(h0.0);
        // Should be able to allocate again
        let h2 = alloc.allocate().unwrap();
        assert_eq!(alloc.num_free(), 0);
        alloc.free(h1.0);
        alloc.free(h2.0);
        assert_eq!(alloc.num_free(), 2);
    }
}
