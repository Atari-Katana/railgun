//! Sampling logic for token generation.
//!
//! This module provides functions to transform logits into token IDs 
//! based on sampling parameters (temperature, top-p, top-k).

use candle_core::{Result, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::deepseek2::TopKLastDimOp;

use vllm_scheduler::SamplingParams;

/// A sampler that applies sampling parameters to logits.
pub struct Sampler {
    processor: LogitsProcessor,
    top_k: u32,
}

impl Sampler {
    pub fn new(params: &SamplingParams) -> Self {
        // Map SamplingParams to candle's LogitsProcessor
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let processor = LogitsProcessor::new(
            seed,
            Some(params.temperature as f64),
            Some(params.top_p as f64),
        );

        Self { 
            processor,
            top_k: params.top_k,
        }
    }

    /// Run sampling on the provided logits.
    /// 
    /// # Arguments
    /// * `logits` - Probabilities or raw scores [vocab_size] or [1, vocab_size]
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let mut logits = if logits.rank() == 2 {
            logits.get(0)?
        } else {
            logits.clone()
        };

        // Top-k masking logic
        if self.top_k > 0 {
            let vocab_size = logits.dims1()?;
            let k = self.top_k as usize;
            if k < vocab_size {
                // Find the k-th largest value to use as a threshold
                let top_k_values = logits.topk(k)?.values;
                let min_top_k = top_k_values.get(k - 1)?
                    .broadcast_as(logits.shape())?;
                let mask = logits.ge(&min_top_k)?;
                let inf_tensor = Tensor::full(f32::NEG_INFINITY, logits.shape(), logits.device())?;
                logits = mask.where_cond(&logits, &inf_tensor)?;
            }
        }

        self.processor.sample(&logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_top_k_sampling() -> Result<()> {
        let device = Device::Cpu;
        // Logits: [0.1, 0.5, 0.2, 0.8, 0.4]
        // Indices:  0    1    2    3    4
        // Sorted:   0.8(3), 0.5(1), 0.4(4), 0.2(2), 0.1(0)
        let logits = Tensor::new(&[0.1f32, 0.5, 0.2, 0.8, 0.4], &device)?;
        
        // top_k = 1 should always give index 3 (greedy)
        let params = SamplingParams {
            top_k: 1,
            temperature: 1.0,
            top_p: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&params);
        let token = sampler.sample(&logits)?;
        assert_eq!(token, 3);

        // top_k = 2 should only give index 3 or 1
        let params = SamplingParams {
            top_k: 2,
            temperature: 1.0,
            top_p: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&params);
        for _ in 0..20 {
            let token = sampler.sample(&logits)?;
            assert!(token == 3 || token == 1, "Token {} is not in top-2", token);
        }

        // top_k = 10 (greater than vocab) should work as no-op
        let params = SamplingParams {
            top_k: 10,
            temperature: 1.0,
            top_p: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&params);
        let _ = sampler.sample(&logits)?;

        Ok(())
    }
}
