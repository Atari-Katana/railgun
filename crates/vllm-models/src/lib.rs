//! # vllm-models
//! Model implementations (Llama, Mistral, Qwen) for Railgun.

pub mod config;
pub mod registry;
pub mod tokenizer;
pub mod traits;
pub mod llama;

pub use config::ModelConfig;
pub use registry::{ModelMetadata, ModelRegistry};
pub use tokenizer::RailgunTokenizer;
pub use traits::CausalLM;

/// Register all built-in model architectures in the global registry.
///
/// This should be called once at application startup (e.g., in `main`).
pub fn register_builtin_models() {
    ModelRegistry::register_provider::<llama::model::LlamaModel>();
}
