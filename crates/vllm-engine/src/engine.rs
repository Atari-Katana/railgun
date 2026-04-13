//! Railgun inference engine.
//!
//! The Engine is the central orchestrator that connects the Scheduler,
//! the Model, and the Sampler. It provides an async interface for clients
//! (like the HTTP server) to submit requests and stream results.

use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, error};

use vllm_core::Result;
use vllm_models::llama::model::LlamaModel;
use vllm_models::RailgunTokenizer;
use vllm_scheduler::{Scheduler, SamplingParams, SchedulerOutput, RequestId};
use crate::sampling::Sampler;

/// Command sent to the background engine loop.
enum EngineCommand {
    Submit {
        prompt: String,
        sampling_params: SamplingParams,
        tx: mpsc::UnboundedSender<EngineStepResponse>,
    },
}

/// Response sent back for each generated token or status update.
#[derive(Debug, Clone)]
pub struct EngineStepResponse {
    pub request_id: RequestId,
    pub text: String,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
}

/// The top-level inference engine.
pub struct RailgunEngine {
    cmd_tx: mpsc::UnboundedSender<EngineCommand>,
}

impl RailgunEngine {
    /// Initialise the engine with a model and scheduler.
    pub fn new(
        model: LlamaModel,
        tokenizer: RailgunTokenizer,
        scheduler: Scheduler,
    ) -> Arc<Self> {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        
        let engine = Arc::new(Self { cmd_tx });
        
        // Start the background engine loop
        tokio::spawn(async move {
            if let Err(e) = run_engine_loop(cmd_rx, model, tokenizer, scheduler).await {
                error!(error = %e, "Engine loop fatal error");
            }
        });

        engine
    }

    /// Submit a new prompt for generation.
    ///
    /// Returns a stream (mpsc receiver) of responses.
    pub async fn generate(
        &self,
        prompt: String,
        sampling_params: SamplingParams,
    ) -> mpsc::UnboundedReceiver<EngineStepResponse> {
        let (tx, rx) = mpsc::unbounded_channel();
        let _ = self.cmd_tx.send(EngineCommand::Submit {
            prompt,
            sampling_params,
            tx,
        });
        rx
    }
}

/// The main engine loop.
/// 
/// This loop runs indefinitely, processing commands and executing inference steps.
async fn run_engine_loop(
    mut cmd_rx: mpsc::UnboundedReceiver<EngineCommand>,
    mut model: LlamaModel,
    mut scheduler: Scheduler,
    tokenizer: RailgunTokenizer,
) -> Result<()> {
    info!("Engine loop started with packed batching");

    let mut response_channels = std::collections::HashMap::new();

    loop {
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                EngineCommand::Submit { prompt, sampling_params, tx } => {
                    let prompt_ids = tokenizer.encode(&prompt, true)?;
                    let request_id = scheduler.add_request(prompt_ids, sampling_params);
                    response_channels.insert(request_id, tx);
                }
            }
        }

        let output = scheduler.schedule();
        if output.is_empty() {
            if let Some(cmd) = cmd_rx.recv().await {
                match cmd {
                    EngineCommand::Submit { prompt, sampling_params, tx } => {
                        let prompt_ids = tokenizer.encode(&prompt, true)?;
                        let request_id = scheduler.add_request(prompt_ids, sampling_params);
                        response_channels.insert(request_id, tx);
                    }
                }
            }
            continue;
        }

        // Execute Model Step (Packed)
        let step_results = execute_model_step(&mut model, &tokenizer, &output, &mut scheduler).await?;

        let mut decode_tokens = Vec::new();
        let mut prefill_done = Vec::new();

        for res in step_results {
            if let Some(token_id) = res.token_id {
                decode_tokens.push((res.request_id, token_id));
            }
            if res.is_prefill {
                prefill_done.push(res.request_id);
            }

            if let Some(tx) = response_channels.get(&res.request_id) {
                if tx.send(res.engine_res.clone()).is_err() {
                    scheduler.abort(res.request_id);
                    response_channels.remove(&res.request_id);
                }
            }
            if res.engine_res.is_finished {
                response_channels.remove(&res.request_id);
            }
        }
        
        scheduler.update_from_output(&vllm_scheduler::StepOutput {
            decode_tokens,
            prefill_done,
        });

        tokio::task::yield_now().await;
    }
}

/// Internal struct for model step results
struct InternalStepResult {
    request_id: RequestId,
    token_id: Option<u32>,
    is_prefill: bool,
    engine_res: EngineStepResponse,
}

/// Executes a single inference step for the currently scheduled batch.
async fn execute_model_step(
    model: &mut LlamaModel,
    tokenizer: &RailgunTokenizer,
    batch: &SchedulerOutput,
    scheduler: &mut Scheduler,
) -> Result<Vec<InternalStepResult>> {
    let mut results = Vec::new();
    let device = candle_core::Device::try_from(model.device())?;
    
    // 1. Prepare packed tensors
    let total_tokens = batch.num_tokens();
    let num_requests = batch.prefill_chunks.len() + batch.decode_slots.len();

    // Flattened input IDs
    let mut flat_tokens = Vec::with_capacity(total_tokens);
    for p in &batch.prefill_chunks { flat_tokens.extend_from_slice(&p.token_ids); }
    for d in &batch.decode_slots { flat_tokens.push(d.last_token_id); }
    let input_ids = candle_core::Tensor::new(flat_tokens, &device)?;
    
    // Slot mapping
    let slot_mapping = candle_core::Tensor::new(
        batch.slot_mapping.as_slice(),
        &device
    )?;

    // Context lens and Block table (Padded for batching)
    let mut context_lens_vec = Vec::with_capacity(num_requests);
    let mut max_blocks = 0;
    for p in &batch.prefill_chunks {
        context_lens_vec.push((p.position_start + p.token_ids.len()) as i32);
        max_blocks = max_blocks.max(p.block_table.len());
    }
    for d in &batch.decode_slots {
        context_lens_vec.push(d.seq_len as i32);
        max_blocks = max_blocks.max(d.block_table.len());
    }
    let context_lens = candle_core::Tensor::new(context_lens_vec, &device)?;

    let mut flat_block_table = Vec::with_capacity(num_requests * max_blocks);
    for p in &batch.prefill_chunks {
        flat_block_table.extend_from_slice(&p.block_table);
        flat_block_table.extend(std::iter::repeat(0).take(max_blocks - p.block_table.len()));
    }
    for d in &batch.decode_slots {
        flat_block_table.extend_from_slice(&d.block_table);
        flat_block_table.extend(std::iter::repeat(0).take(max_blocks - d.block_table.len()));
    }
    let block_table = candle_core::Tensor::new(flat_block_table, &device)?
        .reshape((num_requests, max_blocks))?;

    // 2. Execute GPU Forward Pass
    let logits = model.architecture().forward_packed(
        &input_ids,
        &block_table,
        &context_lens,
        &slot_mapping,
        scheduler.block_pool_mut(),
    )?;

    // 3. Extract and Sample Logits
    // Each request gets its last token's logits
    let mut current_offset = 0;
    for (i, p) in batch.prefill_chunks.iter().enumerate() {
        let last_token_idx = current_offset + p.token_ids.len() - 1;
        let req_logits = logits.get(last_token_idx)?;
        
        let mut sampler = Sampler::new(&p.sampling_params);
        let token_id = sampler.sample(&req_logits)?;
        let text = tokenizer.decode(&[token_id], true)?;
        
        results.push(InternalStepResult {
            request_id: p.request_id,
            token_id: Some(token_id),
            is_prefill: true,
            engine_res: EngineStepResponse {
                request_id: p.request_id,
                text,
                is_finished: false,
                finish_reason: None,
            },
        });
        current_offset += p.token_ids.len();
    }

    for (i, d) in batch.decode_slots.iter().enumerate() {
        let logits_idx = current_offset + i;
        let req_logits = logits.get(logits_idx)?;
        
        let mut sampler = Sampler::new(&d.sampling_params);
        let token_id = sampler.sample(&req_logits)?;
        let text = tokenizer.decode(&[token_id], true)?;
        
        results.push(InternalStepResult {
            request_id: d.request_id,
            token_id: Some(token_id),
            is_prefill: false,
            engine_res: EngineStepResponse {
                request_id: d.request_id,
                text,
                is_finished: false,
                finish_reason: None,
            },
        });
    }

    Ok(results)
}
