//! Request types for the scheduler.
//!
//! A [`Request`] encapsulates everything the scheduler needs to manage one
//! inference request from arrival to completion.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use vllm_paged_attention::KVCache;

/// Unique identifier for a request, scoped to this process lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub Uuid);

impl RequestId {
    /// Generate a fresh random request ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Display only the first 8 hex chars for brevity in logs
        write!(f, "{}", &self.0.to_string()[..8])
    }
}

/// Parameters controlling text generation for one request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Maximum number of new tokens to generate.
    ///
    /// Must be ≥ 1.
    pub max_new_tokens: u32,

    /// Sampling temperature.
    ///
    /// 0.0 = greedy decoding (argmax).
    /// Higher values increase randomness.
    /// Valid range: [0.0, ∞).
    pub temperature: f32,

    /// Nucleus (top-p) sampling parameter.
    ///
    /// The model samples from the smallest set of tokens whose cumulative
    /// probability exceeds `top_p`. 1.0 disables top-p filtering.
    /// Valid range: (0.0, 1.0].
    pub top_p: f32,

    /// Top-k sampling: only the `top_k` highest-probability tokens are
    /// considered. 0 disables top-k filtering.
    pub top_k: u32,

    /// Optional list of token IDs that terminate generation early.
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            stop_token_ids: vec![],
        }
    }
}

/// Why a request finished generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// `max_new_tokens` was reached.
    MaxTokens,
    /// A stop token from `sampling_params.stop_token_ids` was generated.
    StopToken,
    /// The request was explicitly cancelled by the caller.
    Cancelled,
    /// An unrecoverable error occurred during generation.
    Error(String),
}

/// The current lifecycle stage of a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestStatus {
    /// Waiting in the queue; not yet assigned KV cache blocks.
    Waiting,
    /// In the prefill phase; processing prompt tokens.
    ///
    /// `next_chunk_start` tracks progress through chunked prefills.
    Prefilling { next_chunk_start: usize },
    /// Prompt has been processed; generating one new token per step.
    Decoding,
    /// KV cache blocks have been moved to host memory to free GPU space.
    Swapped,
    /// Generation is complete.
    Finished(FinishReason),
}

impl RequestStatus {
    /// Returns `true` if the request is no longer active.
    pub fn is_done(&self) -> bool {
        matches!(self, RequestStatus::Finished(_))
    }
}

/// An inference request tracked by the scheduler.
pub struct Request {
    /// Unique request identifier.
    pub id: RequestId,
    /// Tokenised prompt (input).
    pub prompt_token_ids: Vec<u32>,
    /// Generation parameters.
    pub params: SamplingParams,
    /// Current lifecycle status.
    pub status: RequestStatus,
    /// Tokens generated so far (excluding the prompt).
    pub output_token_ids: Vec<u32>,
    /// KV cache state for this request.
    pub kv_cache: KVCache,
}

impl Request {
    /// Create a new waiting request.
    ///
    /// # Arguments
    ///
    /// * `prompt_token_ids` – Tokenised input. Must be non-empty.
    /// * `params` – Sampling configuration.
    ///
    /// # Panics
    ///
    /// Panics if `prompt_token_ids` is empty.
    pub fn new(prompt_token_ids: Vec<u32>, params: SamplingParams) -> Self {
        assert!(!prompt_token_ids.is_empty(), "prompt must be non-empty");
        Self {
            id: RequestId::new(),
            prompt_token_ids,
            params,
            status: RequestStatus::Waiting,
            output_token_ids: Vec::new(),
            kv_cache: KVCache::new(),
        }
    }

    /// Total sequence length = prompt + generated tokens so far.
    pub fn seq_len(&self) -> usize {
        self.prompt_token_ids.len() + self.output_token_ids.len()
    }

    /// Number of prompt tokens.
    pub fn prompt_len(&self) -> usize {
        self.prompt_token_ids.len()
    }

    /// Number of tokens already processed (KV cached) for this request.
    pub fn num_processed(&self) -> usize {
        match self.status {
            RequestStatus::Waiting => 0,
            RequestStatus::Prefilling { next_chunk_start } => next_chunk_start,
            RequestStatus::Decoding | RequestStatus::Swapped => {
                self.prompt_len() + self.output_token_ids.len()
            }
            RequestStatus::Finished(_) => self.seq_len(),
        }
    }

    /// Append a generated token and check stop conditions.
    ///
    /// Returns the finish reason if generation should stop, or `None` to
    /// continue generating.
    pub fn append_token(&mut self, token_id: u32) -> Option<FinishReason> {
        self.output_token_ids.push(token_id);

        // Check stop tokens
        if self.params.stop_token_ids.contains(&token_id) {
            return Some(FinishReason::StopToken);
        }
        // Check max tokens
        if self.output_token_ids.len() >= self.params.max_new_tokens as usize {
            return Some(FinishReason::MaxTokens);
        }
        None
    }
}

impl std::fmt::Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Request")
            .field("id", &self.id)
            .field("prompt_len", &self.prompt_len())
            .field("output_len", &self.output_token_ids.len())
            .field("status", &self.status)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_lifecycle() {
        let params = SamplingParams {
            max_new_tokens: 3,
            stop_token_ids: vec![2], // EOS
            ..Default::default()
        };
        let mut req = Request::new(vec![1, 2, 3], params);
        assert_eq!(req.prompt_len(), 3);
        assert_eq!(req.seq_len(), 3);
        assert_eq!(req.status, RequestStatus::Waiting);

        // First token — continues
        assert!(req.append_token(42).is_none());
        assert_eq!(req.seq_len(), 4);

        // Stop token
        let reason = req.append_token(2).unwrap();
        assert_eq!(reason, FinishReason::StopToken);
    }

    #[test]
    fn max_tokens_stop() {
        let params = SamplingParams {
            max_new_tokens: 2,
            ..Default::default()
        };
        let mut req = Request::new(vec![1], params);
        assert!(req.append_token(10).is_none());
        let reason = req.append_token(20).unwrap();
        assert_eq!(reason, FinishReason::MaxTokens);
    }

    #[test]
    fn request_id_is_unique() {
        let id1 = RequestId::new();
        let id2 = RequestId::new();
        assert_ne!(id1, id2);
    }
}
