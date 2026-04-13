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
use vllm_paged_attention::block::BlockPool;

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
    /// Total KV cache blocks available.
    pub num_gpu_blocks: usize,
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
    finished: Vec<Request>,
}

impl Scheduler {
    /// Create a new scheduler.
    ///
    /// # Arguments
    ///
    /// * `config` – Scheduler configuration.
    /// * `device` – The GPU device (used to allocate the KV pool).
    /// * `kv_dtype` – dtype for KV tensors (typically BF16).
    /// * `num_kv_heads` – Number of KV attention head groups in the model.
    /// * `head_dim` – Attention head dimension.
    pub fn new(
        config: SchedulerConfig,
        device: Device,
        kv_dtype: DType,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> vllm_core::Result<Self> {
        let pool = BlockPool::new(
            config.num_gpu_blocks,
            config.block_size,
            num_kv_heads,
            head_dim,
            kv_dtype,
            device,
        )?;
        let allocator = BlockAllocator::new(Arc::new(pool));
        info!(
            num_blocks = config.num_gpu_blocks,
            block_size = config.block_size,
            "scheduler KV cache initialised"
        );
        Ok(Self {
            config,
            allocator,
            waiting: VecDeque::new(),
            running: Vec::new(),
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

        // ── 1. Try to promote waiting requests ─────────────────────────────
        while let Some(req) = self.waiting.front() {
            if self.running.len() >= self.config.max_num_seqs {
                break;
            }

            let blocks_needed =
                (req.prompt_len() + self.config.block_size - 1) / self.config.block_size;

            // +1 for the first decode step
            if self.allocator.num_free() < blocks_needed + 1 {
                warn!(
                    free  = self.allocator.num_free(),
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
                                warn!(request_id = %id, "OOM during decode — aborting request");
                                req.status = RequestStatus::Finished(FinishReason::Error(
                                    "out of KV cache blocks".into(),
                                ));
                                done_ids.push(idx);
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Remove finished requests (in reverse order to keep indices valid)
        done_ids.sort_unstable();
        done_ids.dedup();
        for &idx in done_ids.iter().rev() {
            let mut req = self.running.remove(idx);
            self.free_kv_blocks(&mut req);
            self.finished.push(req);
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
        self.allocator.num_free()
    }

    /// Access the KV cache block pool for model inference.
    pub fn block_pool_mut(&mut self) -> &mut BlockPool {
        unsafe { self.allocator.pool_mut() }
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
