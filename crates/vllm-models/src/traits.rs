//! Trait defining the interface for causal language models in Railgun.

use vllm_core::{Device, Result, Tensor};

use crate::config::ModelConfig;

/// A generative language model that can produce the next-token logits.
///
/// Implementors must be `Send + Sync` so they can be shared between the
/// async engine and the blocking GPU thread.
///
/// # Example
///
/// ```ignore
/// let logits = model.forward(&input_ids, &positions, None)?;
/// let next_token = logits.argmax(..)?; // greedy
/// ```
pub trait CausalLM: Send + Sync {
    /// Run a forward pass and return per-position logits.
    ///
    /// # Arguments
    ///
    /// * `input_ids` – Token IDs tensor of shape `[total_tokens]` (packed
    ///   batch: all sequences concatenated). Non-negative integers in
    ///   `[0, vocab_size)`.
    /// * `positions` – Position IDs for RoPE, same shape as `input_ids`.
    ///   For a decode step this is `[seq_len]` for each request.
    /// * `attention_mask` – Optional causal mask (if `None`, assume standard
    ///   causal masking based on positions).
    ///
    /// # Returns
    ///
    /// Logits tensor of shape `[total_tokens, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns any model-level error (invalid shape, CUDA OOM, etc.).
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor>;

    /// Returns the model's configuration.
    fn config(&self) -> &ModelConfig;

    /// Returns the device this model's weights live on.
    fn device(&self) -> Device;

    /// Returns the effective vocabulary size.
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
}
