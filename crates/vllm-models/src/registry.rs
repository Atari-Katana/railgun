//! Model registry — maps architecture strings to factory functions.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{OnceLock, RwLock};

use vllm_core::{CoreError, DType, Device, Result};

use crate::config::ModelConfig;
use crate::traits::CausalLM;

/// Type alias for a model factory function.
pub type ModelFactory =
    fn(model_dir: &Path, config: &ModelConfig, device: Device, dtype: Option<DType>) -> Result<Box<dyn CausalLM>>;

/// Metadata for a registered model architecture.
#[derive(Clone)]
pub struct ModelMetadata {
    /// The architecture identifier (e.g., "LlamaForCausalLM").
    pub architecture: &'static str,
    /// The factory function to instantiate the model.
    pub factory: ModelFactory,
    /// A human-readable description of the model family.
    pub description: &'static str,
    /// The model family (e.g., "Llama", "Mistral").
    pub family: &'static str,
    /// List of supported data types for this architecture.
    pub supported_dtypes: &'static [DType],
    /// Whether this architecture supports Grouped Query Attention (GQA).
    pub supports_gqa: bool,
    /// Whether this architecture supports PagedAttention v1/v2.
    pub supports_paged_attention: bool,
}

impl ModelMetadata {
    /// Create a new metadata object with default values for new fields (Legacy compatibility).
    pub fn new(architecture: &'static str, factory: ModelFactory, description: &'static str) -> Self {
        Self {
            architecture,
            factory,
            description,
            family: "Unknown",
            supported_dtypes: &[DType::F16, DType::BF16, DType::F32],
            supports_gqa: false,
            supports_paged_attention: false,
        }
    }

    /// Builder-style method to set the family.
    pub fn with_family(mut self, family: &'static str) -> Self {
        self.family = family;
        self
    }

    /// Builder-style method to set supported dtypes.
    pub fn with_dtypes(mut self, dtypes: &'static [DType]) -> Self {
        self.supported_dtypes = dtypes;
        self
    }

    /// Builder-style method to set GQA support.
    pub fn with_gqa(mut self, supports_gqa: bool) -> Self {
        self.supports_gqa = supports_gqa;
        self
    }

    /// Builder-style method to set PagedAttention support.
    pub fn with_paged_attention(mut self, supports_pa: bool) -> Self {
        self.supports_paged_attention = supports_pa;
        self
    }
}

/// A trait that models can implement to be easily registered.
///
/// This provides "strict interface enforcement" by ensuring that every
/// registered model provides all required metadata and a factory.
pub trait ModelProvider {
    /// Returns the metadata for this model.
    fn metadata() -> ModelMetadata;
}

static REGISTRY: OnceLock<RwLock<HashMap<String, ModelMetadata>>> = OnceLock::new();

fn get_registry() -> &'static RwLock<HashMap<String, ModelMetadata>> {
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Global model registry for Railgun.
pub struct ModelRegistry;

impl ModelRegistry {
    /// Register a new model architecture.
    pub fn register(metadata: ModelMetadata) {
        let mut registry = get_registry().write().unwrap();
        registry.insert(metadata.architecture.to_string(), metadata);
    }

    /// Register a model that implements the `ModelProvider` trait.
    pub fn register_provider<P: ModelProvider>() {
        Self::register(P::metadata());
    }

    /// Look up metadata for an architecture string.
    pub fn get(architecture: &str) -> Option<ModelMetadata> {
        let registry = get_registry().read().unwrap();
        registry.get(architecture).cloned()
    }

    /// Returns `true` if the architecture is registered.
    pub fn has(architecture: &str) -> bool {
        let registry = get_registry().read().unwrap();
        registry.contains_key(architecture)
    }

    /// List all registered architecture strings.
    pub fn architectures() -> Vec<String> {
        let registry = get_registry().read().unwrap();
        registry.keys().cloned().collect()
    }

    /// List metadata for all registered architectures.
    pub fn all_metadata() -> Vec<ModelMetadata> {
        let registry = get_registry().read().unwrap();
        registry.values().cloned().collect()
    }

    /// Load a model from a directory.
    ///
    /// This helper reads the `config.json`, determines the architecture,
    /// and calls the appropriate factory.
    pub fn load(model_dir: &Path, device: Device, dtype: Option<DType>) -> Result<Box<dyn CausalLM>> {
        let config_path = model_dir.join("config.json");
        let config = ModelConfig::from_file(&config_path).map_err(|e| CoreError::Io(e))?;

        let arch = config.architecture();
        let metadata = Self::get(arch)
            .ok_or_else(|| CoreError::Tensor(format!("unknown architecture: {arch}")))?;

        (metadata.factory)(model_dir, &config, device, dtype)
    }

    /// Build a model from a config and directory using the registered factory.
    pub fn build(
        model_dir: &Path,
        config: &ModelConfig,
        device: Device,
        dtype: Option<DType>,
    ) -> Result<Box<dyn CausalLM>> {
        let arch = config.architecture();
        let metadata = Self::get(arch)
            .ok_or_else(|| CoreError::Tensor(format!("unknown architecture: {arch}")))?;
        (metadata.factory)(model_dir, config, device, dtype)
    }
}

/// Macro to register a model architecture at initialization.
///
/// Since Rust doesn't have static constructors, this is a helper to
/// manually register models in a crate's init function or at the
/// start of `main`.
#[macro_export]
macro_rules! register_model {
    // Legacy style (3 args)
    ($arch:expr, $factory:expr, $desc:expr) => {
        $crate::ModelRegistry::register($crate::ModelMetadata::new($arch, $factory, $desc));
    };
    // Provider style
    ($provider:ty) => {
        $crate::ModelRegistry::register_provider::<$provider>();
    };
    // Advanced style (multiple fields via builder)
    ($arch:expr, $factory:expr, $desc:expr, family: $family:expr, dtypes: $dtypes:expr, gqa: $gqa:expr, pa: $pa:expr) => {
        $crate::ModelRegistry::register(
            $crate::ModelMetadata::new($arch, $factory, $desc)
                .with_family($family)
                .with_dtypes($dtypes)
                .with_gqa($gqa)
                .with_paged_attention($pa),
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_get() {
        fn dummy_factory(_: &Path, _: &ModelConfig, _: Device, _: Option<DType>) -> Result<Box<dyn CausalLM>> {
            Err(CoreError::Tensor("not implemented".into()))
        }

        ModelRegistry::register(ModelMetadata::new(
            "TestArch",
            dummy_factory,
            "A test architecture",
        ));

        assert!(ModelRegistry::has("TestArch"));
        let meta = ModelRegistry::get("TestArch").unwrap();
        assert_eq!(meta.architecture, "TestArch");
        assert_eq!(meta.description, "A test architecture");
        assert_eq!(meta.family, "Unknown");
    }

    #[test]
    fn builder_pattern() {
        fn dummy_factory(_: &Path, _: &ModelConfig, _: Device, _: Option<DType>) -> Result<Box<dyn CausalLM>> {
            Err(CoreError::Tensor("not implemented".into()))
        }

        let meta = ModelMetadata::new("TestBuilder", dummy_factory, "desc")
            .with_family("TestFamily")
            .with_gqa(true);

        assert_eq!(meta.family, "TestFamily");
        assert!(meta.supports_gqa);
        assert!(!meta.supports_paged_attention);
    }
}
