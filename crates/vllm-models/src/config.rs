//! Model configuration, deserialised from HuggingFace `config.json`.

use serde::{Deserialize, Serialize};

/// Configuration for a transformer language model.
///
/// All fields correspond directly to keys in the HuggingFace `config.json`
/// format. Optional fields default to values that are correct for most
/// Llama-family models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model family architectures, e.g. `["LlamaForCausalLM"]`.
    #[serde(default)]
    pub architectures: Vec<String>,

    /// Hidden state dimensionality.
    pub hidden_size: usize,

    /// Number of query attention heads.
    pub num_attention_heads: usize,

    /// Number of key-value head groups (< `num_attention_heads` for GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,

    /// Intermediate (FFN) dimensionality.
    pub intermediate_size: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Maximum number of position embeddings.
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,

    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// Layer normalisation epsilon.
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,

    /// BOS (beginning-of-sequence) token ID.
    #[serde(default)]
    pub bos_token_id: Option<u32>,

    /// EOS (end-of-sequence) token ID.
    #[serde(default)]
    pub eos_token_id: Option<u32>,

    /// Torch dtype string (e.g. `"bfloat16"`).
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

fn default_max_pos() -> usize {
    4096
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_rms_eps() -> f64 {
    1e-5
}

impl ModelConfig {
    /// Effective number of KV heads, falling back to `num_attention_heads`
    /// when the config does not specify GQA.
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
            .unwrap_or(self.num_attention_heads)
    }

    /// Dimension of each attention head: `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Primary architecture string, or `"Unknown"` if empty.
    pub fn architecture(&self) -> &str {
        self.architectures.first().map(|s| s.as_str()).unwrap_or("Unknown")
    }

    /// Load a config from a JSON string.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Load a config from a file path.
    pub fn from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_json(&s).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA_CONFIG: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 16,
        "intermediate_size": 8192,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5
    }"#;

    #[test]
    fn parse_llama_config() {
        let cfg = ModelConfig::from_json(LLAMA_CONFIG).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_kv_heads(), 8);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.architecture(), "LlamaForCausalLM");
    }

    #[test]
    fn defaults_when_kv_heads_absent() {
        let cfg: ModelConfig = serde_json::from_str(r#"{
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "intermediate_size": 1024,
            "vocab_size": 1000
        }"#).unwrap();
        assert_eq!(cfg.num_kv_heads(), 8); // falls back to num_attention_heads
    }
}
