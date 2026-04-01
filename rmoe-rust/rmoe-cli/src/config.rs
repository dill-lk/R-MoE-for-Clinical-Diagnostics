//! CLI configuration management.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// CLI configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Default vision model
    pub default_vision_model: String,
    /// Default reasoning model
    pub default_reasoning_model: String,
    /// Default clinical model
    pub default_clinical_model: String,
    /// Confidence threshold
    pub confidence_threshold: f64,
    /// Maximum recursive iterations
    pub max_iterations: usize,
    /// API configurations
    pub apis: HashMap<String, ApiConfig>,
    /// Model paths
    pub model_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub provider: String,
    pub api_key_env: String,
    pub base_url: Option<String>,
    pub default_model: String,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            default_vision_model: "openai:gpt-4o".to_string(),
            default_reasoning_model: "anthropic:claude-sonnet-4-20250514".to_string(),
            default_clinical_model: "openai:gpt-4o".to_string(),
            confidence_threshold: 0.90,
            max_iterations: 3,
            apis: HashMap::new(),
            model_paths: vec![],
        }
    }
}

impl CliConfig {
    /// Load configuration from file.
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;
        
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: CliConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }

    /// Save configuration to file.
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;
        
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, content)?;
        
        Ok(())
    }

    /// Get configuration file path.
    pub fn config_path() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
        Ok(home.join(".rmoe").join("config.toml"))
    }

    /// Get models directory path.
    pub fn models_dir() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
        Ok(home.join(".rmoe").join("models"))
    }
}
