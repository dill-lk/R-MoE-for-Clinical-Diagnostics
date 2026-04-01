//! Configuration management for R-MoE.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::models::{InferenceParams, Modality, HITLMode};
use crate::error::{RMoEError, RMoEResult};

/// Main configuration for R-MoE framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMoEConfig {
    /// Model paths
    #[serde(default)]
    pub models: ModelPaths,

    /// Inference parameters
    #[serde(default)]
    pub inference: InferenceParams,

    /// Pipeline parameters
    #[serde(default)]
    pub pipeline: PipelineConfig,

    /// API configurations
    #[serde(default)]
    pub api: ApiConfig,

    /// Prompt directory
    #[serde(default = "default_prompt_dir")]
    pub prompts_dir: String,
}

fn default_prompt_dir() -> String {
    "prompts".to_string()
}

impl Default for RMoEConfig {
    fn default() -> Self {
        Self {
            models: ModelPaths::default(),
            inference: InferenceParams::default(),
            pipeline: PipelineConfig::default(),
            api: ApiConfig::default(),
            prompts_dir: default_prompt_dir(),
        }
    }
}

impl RMoEConfig {
    /// Load configuration from a JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> RMoEResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| RMoEError::ConfigError(format!("Failed to read config: {}", e)))?;
        
        serde_json::from_str(&content)
            .map_err(|e| RMoEError::ConfigError(format!("Failed to parse config: {}", e)))
    }

    /// Load configuration from a TOML file.
    pub fn from_toml(path: impl AsRef<Path>) -> RMoEResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| RMoEError::ConfigError(format!("Failed to read config: {}", e)))?;
        
        toml::from_str(&content)
            .map_err(|e| RMoEError::ConfigError(format!("Failed to parse TOML: {}", e)))
    }

    /// Save configuration to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> RMoEResult<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Merge with CLI arguments (CLI takes precedence).
    pub fn merge_cli(&mut self, cli: &CliOverrides) {
        if let Some(ref vision_proj) = cli.vision_proj {
            self.models.vision_proj = vision_proj.clone();
        }
        if let Some(ref vision_text) = cli.vision_text {
            self.models.vision_text = vision_text.clone();
        }
        if let Some(ref reasoning) = cli.reasoning {
            self.models.reasoning = reasoning.clone();
        }
        if let Some(ref clinical) = cli.clinical {
            self.models.clinical = clinical.clone();
        }
        if let Some(temp) = cli.temperature {
            self.inference.temperature = temp;
        }
        if let Some(n) = cli.n_predict {
            self.inference.max_new_tokens = n;
        }
        if let Some(n) = cli.n_gpu_layers {
            self.inference.n_gpu_layers = n;
        }
        if let Some(t) = cli.threshold {
            self.pipeline.confidence_threshold = t;
        }
        if let Some(m) = cli.max_iter {
            self.pipeline.max_iterations = m;
        }
        if let Some(mode) = cli.hitl {
            self.pipeline.hitl_mode = mode;
        }
    }
}

/// Model file paths configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPaths {
    /// CLIP mmproj for vision
    pub vision_proj: String,
    /// Vision text backbone
    pub vision_text: String,
    /// Reasoning model (ARLL)
    pub reasoning: String,
    /// Clinical model (CSR)
    pub clinical: String,
}

impl Default for ModelPaths {
    fn default() -> Self {
        Self {
            vision_proj: "models/vision_proj.gguf".to_string(),
            vision_text: "models/vision_text.gguf".to_string(),
            reasoning: "models/reasoning_expert.gguf".to_string(),
            clinical: "models/clinical_expert.gguf".to_string(),
        }
    }
}

/// Pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Confidence threshold θ for #wanna# gate
    pub confidence_threshold: f64,
    /// Maximum recursive iterations
    pub max_iterations: usize,
    /// HITL interaction mode
    pub hitl_mode: HITLMode,
    /// Current imaging modality
    pub modality: Modality,
    /// Enable bias detection
    pub enable_bias_detection: bool,
    /// Enable temporal comparison
    pub enable_temporal: bool,
    /// Enable RAG retrieval
    pub enable_rag: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.90,
            max_iterations: 3,
            hitl_mode: HITLMode::Auto,
            modality: Modality::CXR,
            enable_bias_detection: true,
            enable_temporal: true,
            enable_rag: true,
        }
    }
}

/// API provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// OpenAI API configuration
    #[serde(default)]
    pub openai: Option<OpenAIConfig>,
    /// Anthropic API configuration
    #[serde(default)]
    pub anthropic: Option<AnthropicConfig>,
    /// Custom API endpoints
    #[serde(default)]
    pub custom: Vec<CustomApiConfig>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            openai: None,
            anthropic: None,
            custom: Vec::new(),
        }
    }
}

/// OpenAI API configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// API key (prefer env var OPENAI_API_KEY)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Base URL (for Azure or proxies)
    #[serde(default = "default_openai_url")]
    pub base_url: String,
    /// Default model
    #[serde(default = "default_openai_model")]
    pub model: String,
    /// Organization ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub organization: Option<String>,
}

fn default_openai_url() -> String {
    "https://api.openai.com/v1".to_string()
}

fn default_openai_model() -> String {
    "gpt-4o".to_string()
}

/// Anthropic API configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    /// API key (prefer env var ANTHROPIC_API_KEY)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Base URL
    #[serde(default = "default_anthropic_url")]
    pub base_url: String,
    /// Default model
    #[serde(default = "default_anthropic_model")]
    pub model: String,
}

fn default_anthropic_url() -> String {
    "https://api.anthropic.com".to_string()
}

fn default_anthropic_model() -> String {
    "claude-sonnet-4-20250514".to_string()
}

/// Custom API endpoint configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomApiConfig {
    /// Endpoint name/identifier
    pub name: String,
    /// Base URL
    pub base_url: String,
    /// API key (if required)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Request timeout (seconds)
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Headers to include
    #[serde(default)]
    pub headers: std::collections::HashMap<String, String>,
}

fn default_timeout() -> u64 {
    60
}

/// CLI argument overrides (for merging with config file).
#[derive(Debug, Clone, Default)]
pub struct CliOverrides {
    pub vision_proj: Option<String>,
    pub vision_text: Option<String>,
    pub reasoning: Option<String>,
    pub clinical: Option<String>,
    pub temperature: Option<f32>,
    pub n_predict: Option<usize>,
    pub n_gpu_layers: Option<i32>,
    pub threshold: Option<f64>,
    pub max_iter: Option<usize>,
    pub hitl: Option<HITLMode>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RMoEConfig::default();
        assert_eq!(config.pipeline.confidence_threshold, 0.90);
        assert_eq!(config.pipeline.max_iterations, 3);
        assert_eq!(config.inference.temperature, 0.2);
    }

    #[test]
    fn test_config_serialization() {
        let config = RMoEConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: RMoEConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.pipeline.confidence_threshold, config.pipeline.confidence_threshold);
    }
}
