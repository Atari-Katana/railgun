//! Sampling logic for token generation.
//!
//! This module provides functions to transform logits into token IDs 
//! based on sampling parameters (temperature, top-p, top-k).

use candle_core::{Result, Tensor};
use candle_transformers::generation::LogitsProcessor;

use vllm_scheduler::SamplingParams;

/// A sampler that applies sampling parameters to logits.
pub struct Sampler {
    processor: LogitsProcessor,
}

impl Sampler {
    pub fn new(params: &SamplingParams) -> Self {
        // Map SamplingParams to candle's LogitsProcessor
        // Note: Llama-3 usually uses a high seed or random seed.
        let seed = 42; 
        
        let processor = LogitsProcessor::new(
            seed,
            Some(params.temperature),
            Some(params.top_p),
        );

        Self { processor }
    }

    /// Run sampling on the provided logits.
    /// 
    /// # Arguments
    /// * `logits` - Probabilities or raw scores [vocab_size] or [1, vocab_size]
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = if logits.rank() == 2 {
            logits.get(0)?
        } else {
            logits.clone()
        };
        
        self.processor.sample(&logits)
    }
}
