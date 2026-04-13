//! Scheduler output — describes what the model runner should execute this step.

use super::request::{RequestId, SamplingParams};
use vllm_paged_attention::block::BlockId;

/// A slot in the decode batch: one request generating its next token.
#[derive(Debug, Clone)]
pub struct DecodeSlot {
    /// The request being decoded.
    pub request_id: RequestId,
    /// The last generated token ID (input for this decode step).
    pub last_token_id: u32,
    /// Current total sequence length (prompt + generated so far).
    pub seq_len: usize,
    /// Block table for this request (passed to paged attention kernel).
    pub block_table: Vec<BlockId>,
    /// Sampling parameters for this request.
    pub sampling_params: SamplingParams,
}

/// A chunk of prompt tokens being prefilled.
#[derive(Debug, Clone)]
pub struct PrefillChunk {
    /// The request this chunk belongs to.
    pub request_id: RequestId,
    /// Token IDs for this chunk.
    pub token_ids: Vec<u32>,
    /// Starting position in the full sequence (for positional embeddings).
    pub position_start: usize,
    /// Block table at the time of scheduling.
    pub block_table: Vec<BlockId>,
    /// Sampling parameters for this request.
    pub sampling_params: SamplingParams,
}

/// Output of one scheduler step — the batch the GPU should execute.
///
/// The model runner receives this, runs attention + FFN, and returns
/// logits that the sampler converts to token selections.
#[derive(Debug, Default)]
pub struct SchedulerOutput {
    /// Prompt chunks to process this step.
    pub prefill_chunks: Vec<PrefillChunk>,
    /// Decode slots to process this step.
    pub decode_slots: Vec<DecodeSlot>,
    /// IDs of requests preempted (evicted) this step.
    pub preempted: Vec<RequestId>,
    /// Mapping of each token in the batch to its KV cache slot.
    /// Shape: [num_tokens]
    pub slot_mapping: Vec<i32>,
}

impl SchedulerOutput {
    /// Total number of tokens scheduled this step.
    pub fn num_tokens(&self) -> usize {
        let prefill: usize = self.prefill_chunks.iter().map(|c| c.token_ids.len()).sum();
        let decode = self.decode_slots.len(); // 1 token per decode slot
        prefill + decode
    }

    /// Returns `true` if no work is scheduled.
    pub fn is_empty(&self) -> bool {
        self.prefill_chunks.is_empty() && self.decode_slots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::RequestId;

    #[test]
    fn empty_output() {
        let out = SchedulerOutput::default();
        assert!(out.is_empty());
        assert_eq!(out.num_tokens(), 0);
    }

    #[test]
    fn num_tokens_counts_correctly() {
        let mut out = SchedulerOutput::default();
        out.prefill_chunks.push(PrefillChunk {
            request_id: RequestId::new(),
            token_ids: vec![1, 2, 3, 4],
            position_start: 0,
            block_table: vec![],
            sampling_params: SamplingParams::default(),
        });
        out.decode_slots.push(DecodeSlot {
            request_id: RequestId::new(),
            last_token_id: 99,
            seq_len: 10,
            block_table: vec![],
            sampling_params: SamplingParams::default(),
        });
        assert_eq!(out.num_tokens(), 5); // 4 prefill + 1 decode
    }
}
