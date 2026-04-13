//! Model registry — maps architecture strings to factory functions.

use std::collections::HashMap;

use vllm_core::{Device, Result};

use crate::config::ModelConfig;
use crate::traits::CausalLM;

/// Type alias for a model factory function.
pub type ModelFactory =
    fn(config: &ModelConfig, device: Device) -> Result<Box<dyn CausalLM>>;

/// Registry mapping architecture identifiers to factory functions.
///
/// # Example
///
/// ```
/// use vllm_models::registry::ModelRegistry;
/// let registry = ModelRegistry::default();
/// assert!(!registry.has("LlamaForCausalLM")); // empty until populated
/// ```
#[derive(Default)]
pub struct ModelRegistry {
    factories: HashMap<&'static str, ModelFactory>,
}

impl ModelRegistry {
    /// Register a new factory.
    pub fn register(&mut self, architecture: &'static str, factory: ModelFactory) {
        self.factories.insert(architecture, factory);
    }

    /// Look up a factory by architecture string.
    pub fn get(&self, architecture: &str) -> Option<ModelFactory> {
        self.factories.get(architecture).copied()
    }

    /// Returns `true` if the architecture is registered.
    pub fn has(&self, architecture: &str) -> bool {
        self.factories.contains_key(architecture)
    }

    /// List all registered architecture strings.
    pub fn architectures(&self) -> Vec<&&'static str> {
        self.factories.keys().collect()
    }

    /// Build a model from a config using the registered factory.
    ///
    /// # Errors
    ///
    /// Returns [`vllm_core::CoreError`] or a model-specific error if
    /// construction fails. Returns a string error if the architecture is
    /// not registered.
    pub fn build(
        &self,
        config: &ModelConfig,
        device: Device,
    ) -> std::result::Result<Box<dyn CausalLM>, String> {
        let arch = config.architecture();
        let factory = self
            .get(arch)
            .ok_or_else(|| format!("unknown architecture: {arch}"))?;
        factory(config, device).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_has_no_architectures() {
        let reg = ModelRegistry::default();
        assert!(!reg.has("LlamaForCausalLM"));
    }

    #[test]
    fn register_and_has() {
        let mut reg = ModelRegistry::default();
        fn dummy_factory(_: &ModelConfig, _: Device) -> Result<Box<dyn CausalLM>> {
            panic!("not implemented")
        }
        reg.register("LlamaForCausalLM", dummy_factory);
        assert!(reg.has("LlamaForCausalLM"));
        assert!(!reg.has("MistralForCausalLM"));
    }
}
