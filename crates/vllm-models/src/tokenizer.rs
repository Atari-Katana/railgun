//! Tokenizer wrapper for Railgun.
//!
//! Loads HuggingFace tokenizers from `tokenizer.json` and provides
//! encode/decode helpers used by the CLI and engine.

use std::path::Path;

use tokenizers::Tokenizer;
use tracing::debug;

use crate::config::ModelConfig;
use vllm_core::{CoreError, Result};

/// A loaded HuggingFace tokenizer.
///
/// Wraps the `tokenizers` crate to provide a Railgun-typed API.
#[derive(Clone)]
pub struct RailgunTokenizer {
    inner: Tokenizer,
}

impl RailgunTokenizer {
    /// Load from `tokenizer.json` in a model directory.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Io`] if the file doesn't exist.
    /// Returns [`CoreError::Tensor`] if the tokenizer fails to parse.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&path).map_err(|e| {
            CoreError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("tokenizer load error: {e}"),
            ))
        })?;
        debug!(path = %path.display(), "tokenizer loaded");
        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Tensor`] if encoding fails.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| CoreError::Tensor(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a sequence of token IDs back to text.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Tensor`] if decoding fails.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| CoreError::Tensor(e.to_string()))
    }

    /// BOS token ID from the tokenizer vocab, if available.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner
            .token_to_id("<|begin_of_text|>")
            .or_else(|| self.inner.token_to_id("<s>"))
    }

    /// EOS token IDs — returns the first found among known EOS tokens.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner
            .token_to_id("<|eot_id|>")
            .or_else(|| self.inner.token_to_id("<|end_of_text|>"))
            .or_else(|| self.inner.token_to_id("</s>"))
    }

    /// Retrieve the EOS token from the model config as a fallback.
    pub fn eos_from_config(config: &ModelConfig) -> Option<u32> {
        config.eos_token_id
    }
}

#[cfg(test)]
mod tests {
    // Integration tests require a real tokenizer.json file.
    // Run with: cargo test -p vllm-models -- --ignored
    // after downloading a model to /tmp/test-model/.
}
