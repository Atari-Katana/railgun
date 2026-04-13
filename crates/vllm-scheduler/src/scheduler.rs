//! Continuous batching scheduler.
//!
//! [`Scheduler`] maintains a waiting queue and a running batch.
//! Each call to [`schedule`] produces a [`SchedulerOutput`] describing
//! exactly what the model runner should execute this step.
//!
//! # Algorithm
//!
//! The scheduler uses **FIFO with token budget** (matching vLLM V1):
//!
//! 1. Promote waiting requests into the running batch if:
//!    - There are free KV cache blocks to cover the prompt.
//!    - The step's token budget (`max_num_batched_tokens`) is not exceeded.
//! 2. Schedule one decode token per running request.
//! 3. If KV blocks are exhausted, preempt the most-recently-added
//!    running request (swap it back to waiting).

use std::collections::VecDeque;
use std::sync::Arc;

use tracing::{debug, info, warn};

use vllm_paged_attention::allocator::BlockAllocator;
use vllm_paged_attention::block::{GpuBlockPool, CpuBlockPool};

use super::batch::{DecodeSlot, PrefillChunk, SchedulerOutput};
use super::request::{FinishReason, Request, RequestId, RequestStatus, SamplingParams};
use vllm_core::{Device, DType};

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum tokens processed per step (prefill + decode combined).
    pub max_num_batched_tokens: usize,
    /// Maximum number of requests in the running batch simultaneously.
    pub max_num_seqs: usize,
    /// KV cache block size (tokens per block).
    pub block_size: usize,
    /// Total KV cache blocks available on GPU.
    pub num_gpu_blocks: usize,
    /// Total KV cache blocks available on CPU (host memory).
    pub num_cpu_blocks: usize,
    /// Maximum sequence length (prompt + generation).
    pub max_model_len: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_batched_tokens: 4096,
            max_num_seqs: 256,
            block_size: 16,
            num_gpu_blocks: 1000,
            num_cpu_blocks: 512,
            max_model_len: 4096,
        }
    }
}

/// The result of feeding a sampled token back to the scheduler.
#[derive(Debug)]
pub struct StepOutput {
    /// One (request_id, token_id) pair per request that was in decode this step.
    pub decode_tokens: Vec<(RequestId, u32)>,
    /// Requests that finished prefill and ready to start decoding.
    pub prefill_done: Vec<RequestId>,
}

/// Continuous batching scheduler.
///
/// # Thread Safety
///
/// `Scheduler` is **not** `Sync`. The engine loop must call `schedule` and
/// `update_from_output` from a single thread (or a single tokio task).
pub struct Scheduler {
    config: SchedulerConfig,
    allocator: BlockAllocator,
    waiting: VecDeque<Request>,
    running: Vec<Request>,
    swapped: VecDeque<Request>,
    finished: Vec<Request>,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(
        config: SchedulerConfig,
        device: Device,
        kv_dtype: DType,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> vllm_core::Result<Self> {
        let gpu_pool = GpuBlockPool::new(
            config.num_gpu_blocks,
            config.block_size,
            num_kv_heads,
            head_dim,
            kv_dtype,
            device,
        )?;
        let cpu_pool = CpuBlockPool::new(
            config.num_cpu_blocks,
            config.block_size,
            num_kv_heads,
            head_dim,
            kv_dtype,
        )?;
        let allocator = BlockAllocator::new(Arc::new(gpu_pool), Arc::new(cpu_pool));
        info!(
            gpu_blocks = config.num_gpu_blocks,
            cpu_blocks = config.num_cpu_blocks,
            block_size = config.block_size,
            "scheduler KV cache initialised with host swapping"
        );
        Ok(Self {
            config,
            allocator,
            waiting: VecDeque::new(),
            running: Vec::new(),
            swapped: VecDeque::new(),
            finished: Vec::new(),
        })
    }

    /// Add an incoming request to the waiting queue.
    pub fn add_request(&mut self, prompt_token_ids: Vec<u32>, params: SamplingParams) -> RequestId {
        let req = Request::new(prompt_token_ids, params);
        let id = req.id;
        debug!(request_id = %id, "enqueued");
        self.waiting.push_back(req);
        id
    }


    /// Compute and return the batch to execute this step.
    ///
    /// Must be followed by a call to [`update_from_output`] after the model
    /// runs, to apply the sampled tokens.
    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut out = SchedulerOutput::default();
        let mut token_budget = self.config.max_num_batched_tokens;

        // ── 1. Try to swap in requests from host ───────────────────────────
        while let Some(req) = self.swapped.front() {
            if self.running.len() >= self.config.max_num_seqs {
                break;
            }
            let blocks_needed = req.kv_cache.num_blocks();
            if self.allocator.num_free_gpu() < blocks_needed {
                break;
            }

            let mut req = self.swapped.pop_front().unwrap();
            let gpu_blocks = self.allocator.swap_in(req.kv_cache.as_block_ids()).expect("swap_in failed");
            req.kv_cache.block_table = gpu_blocks;
            req.status = RequestStatus::Decoding;
            debug!(request_id = %req.id, "swapped in from host");
            self.running.push(req);
        }

        // ── 2. If GPU is full and we have waiting requests, swap out LRU ──
        while !self.waiting.is_empty() && self.allocator.num_free_gpu() < 1 && !self.running.is_empty() {
            // Simple LRU: the request at the front of `running` is the oldest
            let mut req = self.running.remove(0);
            match self.allocator.swap_out(req.kv_cache.as_block_ids()) {
                Ok(cpu_blocks) => {
                    req.kv_cache.block_table = cpu_blocks;
                    req.status = RequestStatus::Swapped;
                    self.swapped.push_back(req);
                    info!(request_id = %self.swapped.back().unwrap().id, "swapped out to host to make room for waiting");
                }
                Err(_) => {
                    // CPU pool also full, fallback to preemption (back to waiting)
                    warn!(request_id = %req.id, "CPU pool full during swap out — preempting");
                    self.free_kv_blocks(&mut req);
                    req.status = RequestStatus::Waiting;
                    req.output_token_ids.clear();
                    self.waiting.push_front(req);
                }
            }
        }

        // ── 3. Try to promote waiting requests ─────────────────────────────
        while let Some(req) = self.waiting.front() {
            if self.running.len() >= self.config.max_num_seqs {
                break;
            }

            let blocks_needed =
                (req.prompt_len() + self.config.block_size - 1) / self.config.block_size;

            // +1 for the first decode step
            if self.allocator.num_free_gpu() < blocks_needed + 1 {
                warn!(
                    free  = self.allocator.num_free_gpu(),
                    need  = blocks_needed + 1,
                    "KV cache full; cannot admit new request"
                );
                break;
            }

            let chunk_size = req.prompt_len().min(token_budget);
            if chunk_size == 0 {
                break;
            }

            let mut req = self.waiting.pop_front().unwrap();

            // Allocate initial KV blocks
            for _ in 0..blocks_needed {
                let handle = self.allocator.allocate().unwrap();
                req.kv_cache.push_block(handle.0);
            }

            let token_ids: Vec<u32> = req.prompt_token_ids[..chunk_size].to_vec();
            let block_table = req.kv_cache.as_block_ids().to_vec();

            let num_processed = req.num_processed();
            
            if chunk_size < req.prompt_len() {
                req.status = RequestStatus::Prefilling {
                    next_chunk_start: num_processed + chunk_size,
                };
            } else {
                req.status = RequestStatus::Prefilling {
                    next_chunk_start: req.prompt_len(),
                };
            }

            for (i, _) in token_ids.iter().enumerate() {
                let block_idx = (num_processed + i) / self.config.block_size;
                let token_idx = (num_processed + i) % self.config.block_size;
                let block_id = block_table[block_idx];
                out.slot_mapping.push((block_id.0 * self.config.block_size as u32 + token_idx as u32) as i32);
            }

            out.prefill_chunks.push(PrefillChunk {
                request_id: req.id,
                token_ids,
                position_start: num_processed,
                block_table,
                sampling_params: req.params.clone(),
            });

            token_budget -= chunk_size;
            self.running.push(req);
        }

        // ── 2. Decode step for all running requests ─────────────────────────
        for req in self.running.iter() {
            if matches!(req.status, RequestStatus::Decoding) && token_budget > 0 {
                let last_token = req
                    .output_token_ids
                    .last()
                    .copied()
                    .unwrap_or_else(|| *req.prompt_token_ids.last().unwrap());

                let seq_len = req.seq_len();
                let block_table = req.kv_cache.as_block_ids();
                let block_idx = (seq_len - 1) / self.config.block_size;
                let token_idx = (seq_len - 1) % self.config.block_size;
                let block_id = block_table[block_idx];
                out.slot_mapping.push((block_id.0 * self.config.block_size as u32 + token_idx as u32) as i32);

                out.decode_slots.push(DecodeSlot {
                    request_id: req.id,
                    last_token_id: last_token,
                    seq_len,
                    block_table: block_table.to_vec(),
                    sampling_params: req.params.clone(),
                });
                token_budget -= 1;
            }
        }

        debug!(
            prefill = out.prefill_chunks.len(),
            decode = out.decode_slots.len(),
            tokens = out.num_tokens(),
            "scheduled"
        );
        out
    }

    /// Apply sampled tokens from the model step back into request state.
    ///
    /// Requests that finish are moved to `self.finished`.
    pub fn update_from_output(&mut self, output: &StepOutput) {
        // Mark prefill-done requests as decoding
        for id in &output.prefill_done {
            if let Some(req) = self.running.iter_mut().find(|r| &r.id == id) {
                req.status = RequestStatus::Decoding;
            }
        }

        // Append decoded tokens
        let mut done_ids: Vec<usize> = Vec::new();
        let mut preempted_ids: Vec<usize> = Vec::new();
        for (id, token_id) in &output.decode_tokens {
            if let Some((idx, req)) = self
                .running
                .iter_mut()
                .enumerate()
                .find(|(_, r)| &r.id == id)
            {
                if let Some(reason) = req.append_token(*token_id) {
                    req.status = RequestStatus::Finished(reason);
                    done_ids.push(idx);
                } else {
                    // Ensure we have a block for the next token slot
                    let tokens_in_use = req.seq_len();
                    let blocks_needed =
                        (tokens_in_use + self.config.block_size - 1) / self.config.block_size;
                    while req.kv_cache.num_blocks() < blocks_needed {
                        match self.allocator.allocate() {
                            Ok(h) => req.kv_cache.push_block(h.0),
                            Err(_) => {
                                // Try to swap out the request being processed if it hit OOM
                                if self.allocator.num_free_cpu() >= req.kv_cache.num_blocks() {
                                    warn!(request_id = %id, "OOM during decode — swapping to host");
                                    preempted_ids.push(idx); // We'll handle the actual swap in the removal loop
                                } else {
                                    warn!(request_id = %id, "OOM during decode & CPU pool full — aborting");
                                    req.status = RequestStatus::Finished(FinishReason::Error(
                                        "out of KV cache blocks (GPU & CPU)".into(),
                                    ));
                                    done_ids.push(idx);
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        let mut all_remove_ids = done_ids.clone();
        all_remove_ids.extend(preempted_ids.clone());
        all_remove_ids.sort_unstable();
        all_remove_ids.dedup();

        for &idx in all_remove_ids.iter().rev() {
            let mut req = self.running.remove(idx);
            if preempted_ids.contains(&idx) {
                match self.allocator.swap_out(req.kv_cache.as_block_ids()) {
                    Ok(cpu_blocks) => {
                        req.kv_cache.block_table = cpu_blocks;
                        req.status = RequestStatus::Swapped;
                        self.swapped.push_back(req);
                    }
                    Err(_) => {
                        // This shouldn't happen if we checked num_free_cpu above, but safety first
                        self.free_kv_blocks(&mut req);
                        req.status = RequestStatus::Waiting;
                        req.output_token_ids.clear();
                        self.waiting.push_front(req);
                    }
                }
            } else {
                self.free_kv_blocks(&mut req);
                self.finished.push(req);
            }
        }
    }

    /// Abort a request proactively (e.g. client disconnected).
    pub fn abort(&mut self, id: RequestId) {
        if let Some(pos) = self.running.iter().position(|r| r.id == id) {
            let mut req = self.running.remove(pos);
            self.free_kv_blocks(&mut req);
            req.status = RequestStatus::Finished(FinishReason::Cancelled);
            self.finished.push(req);
        } else if let Some(pos) = self.waiting.iter().position(|r| r.id == id) {
            let mut req = self.waiting.remove(pos).unwrap();
            req.status = RequestStatus::Finished(FinishReason::Cancelled);
            self.finished.push(req);
        }
    }

    /// Returns a reference to an active request.
    pub fn find_request(&self, id: RequestId) -> Option<&Request> {
        self.running
            .iter()
            .find(|r| r.id == id)
            .or_else(|| self.waiting.iter().find(|r| r.id == id))
            .or_else(|| self.finished.iter().find(|r| r.id == id))
    }


    /// Take all finished requests.
    pub fn drain_finished(&mut self) -> Vec<Request> {
        std::mem::take(&mut self.finished)
    }

    /// Return all KV cache blocks held by a request.
    fn free_kv_blocks(&mut self, req: &mut Request) {
        for &block_id in req.kv_cache.as_block_ids() {
            self.allocator.free(block_id);
        }
        req.kv_cache.block_table.clear();
    }

    // ── Metrics helpers ─────────────────────────────────────────────────────

    /// Number of requests waiting to be scheduled.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Number of requests currently in-flight.
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Number of free KV cache blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.allocator.num_free_gpu()
    }

    /// Access the KV cache block pool for model inference.
    pub fn block_pool_mut(&mut self) -> &mut GpuBlockPool {
        unsafe { self.allocator.gpu_pool_mut() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vllm_core::DType;

    fn tiny_scheduler() -> Scheduler {
        let config = SchedulerConfig {
            max_num_batched_tokens: 64,
            max_num_seqs: 4,
            block_size: 4,
            num_gpu_blocks: 32,
            num_cpu_blocks: 16,
            max_model_len: 128,
        };
        Scheduler::new(config, Device::Cpu, DType::F32, 2, 8).unwrap()
    }

    use vllm_core::Device;

    #[test]
    fn add_and_schedule_prefill() {
        let mut sched = tiny_scheduler();
        sched.add_request(vec![1, 2, 3, 4], SamplingParams::default());
        let out = sched.schedule();
        assert_eq!(out.prefill_chunks.len(), 1);
        assert_eq!(out.decode_slots.len(), 0);
        assert_eq!(out.prefill_chunks[0].token_ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn decode_after_prefill_done() {
        let mut sched = tiny_scheduler();
        let id = sched.add_request(vec![1, 2], SamplingParams::default());

        // Schedule prefill
        let _ = sched.schedule();

        // Simulate model output: prefill done
        sched.update_from_output(&StepOutput {
            decode_tokens: vec![],
            prefill_done: vec![id],
        });

        // Next step should have a decode slot
        let out2 = sched.schedule();
        assert_eq!(out2.decode_slots.len(), 1);
    }

    #[test]
    fn request_finishes_on_max_tokens() {
        let mut sched = tiny_scheduler();
        let params = SamplingParams {
            max_new_tokens: 1,
            ..Default::default()
        };
        let id = sched.add_request(vec![1], params);
        let _ = sched.schedule();

        // Prefill done
        sched.update_from_output(&StepOutput {
            decode_tokens: vec![],
            prefill_done: vec![id],
        });
        let out = sched.schedule();
        assert_eq!(out.decode_slots.len(), 1);

        // Apply 1 decode token → should finish
        sched.update_from_output(&StepOutput {
            decode_tokens: vec![(id, 42)],
            prefill_done: vec![],
        });

        assert_eq!(sched.num_running(), 0);
        let finished = sched.drain_finished();
        assert_eq!(finished.len(), 1);
        assert!(matches!(
            finished[0].status,
            RequestStatus::Finished(FinishReason::MaxTokens)
        ));
    }

    #[test]
    fn multiple_requests_scheduled_together() {
        let mut sched = tiny_scheduler();
        for _ in 0..3 {
            sched.add_request(vec![1, 2], SamplingParams::default());
        }
        let out = sched.schedule();
        // All 3 should be prefilled (3×2 = 6 tokens, budget = 64)
        assert_eq!(out.prefill_chunks.len(), 3);
    }
}
