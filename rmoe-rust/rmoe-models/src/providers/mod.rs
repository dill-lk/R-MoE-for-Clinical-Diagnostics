//! Provider implementations for all major AI APIs.
//!
//! Supported providers:
//! - OpenAI (GPT-4, GPT-4V, GPT-4o)
//! - Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
//! - Google AI (Gemini Pro, Gemini Pro Vision)
//! - Azure OpenAI
//! - Groq (fast inference)
//! - Together AI
//! - Mistral AI
//! - Ollama (local server)
//! - OpenRouter (multi-provider)
//! - Custom OpenAI-compatible endpoints

pub mod openai;
pub mod anthropic;
pub mod google;
pub mod azure;
pub mod groq;
pub mod together;
pub mod mistral;
pub mod ollama;
pub mod openrouter;

use async_trait::async_trait;
use rmoe_core::{InferenceParams, RMoEError, RMoEResult};
use serde::{Deserialize, Serialize};

pub use openai::*;
pub use anthropic::*;
pub use google::*;
pub use azure::*;
pub use groq::*;
pub use together::*;
pub use mistral::*;
pub use ollama::*;
pub use openrouter::*;

/// Supported API providers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    OpenAI,
    Anthropic,
    Google,
    Azure,
    Groq,
    Together,
    Mistral,
    Ollama,
    OpenRouter,
    Custom(String),
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::OpenAI => write!(f, "OpenAI"),
            Provider::Anthropic => write!(f, "Anthropic"),
            Provider::Google => write!(f, "Google AI"),
            Provider::Azure => write!(f, "Azure OpenAI"),
            Provider::Groq => write!(f, "Groq"),
            Provider::Together => write!(f, "Together AI"),
            Provider::Mistral => write!(f, "Mistral AI"),
            Provider::Ollama => write!(f, "Ollama"),
            Provider::OpenRouter => write!(f, "OpenRouter"),
            Provider::Custom(name) => write!(f, "Custom ({})", name),
        }
    }
}

impl Provider {
    /// Get the default base URL for this provider.
    pub fn default_base_url(&self) -> &str {
        match self {
            Provider::OpenAI => "https://api.openai.com/v1",
            Provider::Anthropic => "https://api.anthropic.com/v1",
            Provider::Google => "https://generativelanguage.googleapis.com/v1beta",
            Provider::Azure => "", // Requires custom endpoint
            Provider::Groq => "https://api.groq.com/openai/v1",
            Provider::Together => "https://api.together.xyz/v1",
            Provider::Mistral => "https://api.mistral.ai/v1",
            Provider::Ollama => "http://localhost:11434/api",
            Provider::OpenRouter => "https://openrouter.ai/api/v1",
            Provider::Custom(_) => "",
        }
    }

    /// Get the environment variable name for API key.
    pub fn api_key_env_var(&self) -> &str {
        match self {
            Provider::OpenAI => "OPENAI_API_KEY",
            Provider::Anthropic => "ANTHROPIC_API_KEY",
            Provider::Google => "GOOGLE_API_KEY",
            Provider::Azure => "AZURE_OPENAI_API_KEY",
            Provider::Groq => "GROQ_API_KEY",
            Provider::Together => "TOGETHER_API_KEY",
            Provider::Mistral => "MISTRAL_API_KEY",
            Provider::Ollama => "", // No key needed
            Provider::OpenRouter => "OPENROUTER_API_KEY",
            Provider::Custom(_) => "CUSTOM_API_KEY",
        }
    }

    /// Get recommended models for clinical diagnostics.
    pub fn recommended_models(&self) -> Vec<&str> {
        match self {
            Provider::OpenAI => vec!["gpt-4o", "gpt-4-turbo", "gpt-4-vision-preview"],
            Provider::Anthropic => vec!["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
            Provider::Google => vec!["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro-vision"],
            Provider::Azure => vec!["gpt-4o", "gpt-4-turbo"],
            Provider::Groq => vec!["llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
            Provider::Together => vec!["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
            Provider::Mistral => vec!["mistral-large-latest", "mistral-medium-latest"],
            Provider::Ollama => vec!["llama3.1:70b", "mixtral:8x7b", "medllama2"],
            Provider::OpenRouter => vec!["anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
            Provider::Custom(_) => vec![],
        }
    }
}

/// Configuration for an API provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub provider: Provider,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub organization: Option<String>,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub headers: std::collections::HashMap<String, String>,
}

impl ProviderConfig {
    /// Create config from provider with defaults.
    pub fn new(provider: Provider, model: impl Into<String>) -> Self {
        Self {
            base_url: provider.default_base_url().to_string(),
            provider,
            api_key: None,
            model: model.into(),
            organization: None,
            timeout_secs: 120,
            max_retries: 3,
            headers: std::collections::HashMap::new(),
        }
    }

    /// Load API key from environment variable.
    pub fn with_env_key(mut self) -> Self {
        let env_var = self.provider.api_key_env_var();
        if !env_var.is_empty() {
            self.api_key = std::env::var(env_var).ok();
        }
        self
    }

    /// Set API key explicitly.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> RMoEResult<()> {
        // Ollama doesn't need an API key
        if !matches!(self.provider, Provider::Ollama) && self.api_key.is_none() {
            return Err(RMoEError::ConfigError(format!(
                "API key required for {}. Set {} environment variable.",
                self.provider,
                self.provider.api_key_env_var()
            )));
        }
        if self.base_url.is_empty() {
            return Err(RMoEError::ConfigError(
                "Base URL is required".to_string()
            ));
        }
        if self.model.is_empty() {
            return Err(RMoEError::ConfigError(
                "Model name is required".to_string()
            ));
        }
        Ok(())
    }
}

/// Trait for provider-specific request building.
#[async_trait]
pub trait ProviderClient: Send + Sync {
    /// Send a chat completion request.
    async fn chat_completion(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<String>;

    /// Send a streaming chat completion request.
    async fn chat_completion_stream(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<tokio::sync::mpsc::Receiver<String>>;

    /// Send a vision request (for multimodal models).
    async fn vision_completion(
        &self,
        messages: &[ChatCompletionMessage],
        image_data: &[u8],
        params: &InferenceParams,
    ) -> RMoEResult<String>;

    /// Get provider name.
    fn provider_name(&self) -> &str;

    /// Get model name.
    fn model_name(&self) -> &str;

    /// Check if provider supports vision.
    fn supports_vision(&self) -> bool;
}

/// Standard chat message format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: MessageContent,
}

/// Message content (text or multimodal).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Multimodal(Vec<ContentPart>),
}

/// Content part for multimodal messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlContent },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrlContent {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ChatCompletionMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: MessageContent::Text(content.into()),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: MessageContent::Text(content.into()),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: MessageContent::Text(content.into()),
        }
    }

    pub fn user_with_image(text: impl Into<String>, image_base64: impl Into<String>, mime_type: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: MessageContent::Multimodal(vec![
                ContentPart::Text { text: text.into() },
                ContentPart::ImageUrl {
                    image_url: ImageUrlContent {
                        url: format!("data:{};base64,{}", mime_type, image_base64.into()),
                        detail: Some("high".to_string()),
                    },
                },
            ]),
        }
    }
}

/// Create a provider client from configuration.
pub fn create_client(config: ProviderConfig) -> RMoEResult<Box<dyn ProviderClient>> {
    config.validate()?;
    
    match config.provider {
        Provider::OpenAI => Ok(Box::new(OpenAIClient::new(config))),
        Provider::Anthropic => Ok(Box::new(AnthropicClient::new(config))),
        Provider::Google => Ok(Box::new(GoogleClient::new(config))),
        Provider::Azure => Ok(Box::new(AzureClient::new(config))),
        Provider::Groq => Ok(Box::new(GroqClient::new(config))),
        Provider::Together => Ok(Box::new(TogetherClient::new(config))),
        Provider::Mistral => Ok(Box::new(MistralClient::new(config))),
        Provider::Ollama => Ok(Box::new(OllamaClient::new(config))),
        Provider::OpenRouter => Ok(Box::new(OpenRouterClient::new(config))),
        Provider::Custom(_) => Ok(Box::new(OpenAIClient::new(config))), // Use OpenAI-compatible
    }
}
