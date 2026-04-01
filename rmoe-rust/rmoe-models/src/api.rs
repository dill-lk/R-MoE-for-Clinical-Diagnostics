//! API client implementations for external LLM providers.
//!
//! Supports:
//! - OpenAI (GPT-4, GPT-4V, etc.)
//! - Anthropic (Claude)
//! - Custom REST endpoints

use async_trait::async_trait;
use rmoe_core::{
    ChatMessage, ChatRole, InferenceParams, RMoEError, RMoEResult,
    TextModel, ChatModel,
};
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn};

// ═══════════════════════════════════════════════════════════════════════════════
//  OpenAI Client
// ═══════════════════════════════════════════════════════════════════════════════

/// OpenAI API client.
#[derive(Debug, Clone)]
pub struct OpenAIClient {
    api_key: String,
    base_url: String,
    model: String,
    organization: Option<String>,
    client: reqwest::Client,
}

impl OpenAIClient {
    /// Create a new OpenAI client.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: model.into(),
            organization: None,
            client: reqwest::Client::new(),
        }
    }

    /// Create from environment variable.
    pub fn from_env(model: impl Into<String>) -> RMoEResult<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| RMoEError::ConfigError("OPENAI_API_KEY not set".to_string()))?;
        Ok(Self::new(api_key, model))
    }

    /// Set custom base URL (for Azure or proxies).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set organization ID.
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Send chat completion request.
    async fn chat_completion(&self, messages: &[OpenAIMessage], params: &InferenceParams) -> RMoEResult<String> {
        let request = OpenAIChatRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            temperature: Some(params.temperature),
            max_tokens: Some(params.max_new_tokens as u32),
            top_p: Some(params.top_p),
            frequency_penalty: Some(params.repeat_penalty - 1.0), // Adjust for OpenAI format
            stream: Some(false),
        };

        let mut req = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");

        if let Some(ref org) = self.organization {
            req = req.header("OpenAI-Organization", org);
        }

        debug!(model = self.model, "Sending OpenAI chat completion request");

        let response = req
            .json(&request)
            .send()
            .await
            .map_err(|e| RMoEError::ApiError {
                status: 0,
                message: e.to_string(),
            })?;

        let status = response.status().as_u16();
        if status == 429 {
            return Err(RMoEError::RateLimitExceeded { retry_after_ms: 60000 });
        }
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(RMoEError::ApiError { status, message: body });
        }

        let body: OpenAIChatResponse = response.json().await.map_err(|e| {
            RMoEError::ParseError(format!("Failed to parse OpenAI response: {}", e))
        })?;

        body.choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| RMoEError::ParseError("No choices in response".to_string()))
    }
}

#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

impl From<&ChatMessage> for OpenAIMessage {
    fn from(msg: &ChatMessage) -> Self {
        Self {
            role: match msg.role {
                ChatRole::System => "system".to_string(),
                ChatRole::User => "user".to_string(),
                ChatRole::Assistant => "assistant".to_string(),
            },
            content: msg.content.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[async_trait]
impl TextModel for OpenAIClient {
    async fn generate(
        &self,
        system_prompt: &str,
        user_input: &str,
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        let messages = vec![
            OpenAIMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            OpenAIMessage {
                role: "user".to_string(),
                content: user_input.to_string(),
            },
        ];
        self.chat_completion(&messages, params).await
    }

    async fn generate_stream(
        &self,
        system_prompt: &str,
        user_input: &str,
        params: &InferenceParams,
    ) -> RMoEResult<tokio::sync::mpsc::Receiver<String>> {
        // For simplicity, use non-streaming and emit all at once
        // Real implementation would use SSE streaming
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let response = self.generate(system_prompt, user_input, params).await?;
        
        tokio::spawn(async move {
            let _ = tx.send(response).await;
        });

        Ok(rx)
    }

    fn name(&self) -> &str {
        &self.model
    }

    fn is_ready(&self) -> bool {
        true
    }
}

#[async_trait]
impl ChatModel for OpenAIClient {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        let oai_messages: Vec<OpenAIMessage> = messages.iter().map(|m| m.into()).collect();
        self.chat_completion(&oai_messages, params).await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        params: &InferenceParams,
    ) -> RMoEResult<tokio::sync::mpsc::Receiver<String>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let response = self.chat(messages, params).await?;
        
        tokio::spawn(async move {
            let _ = tx.send(response).await;
        });

        Ok(rx)
    }

    fn name(&self) -> &str {
        &self.model
    }

    fn is_ready(&self) -> bool {
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Anthropic Client
// ═══════════════════════════════════════════════════════════════════════════════

/// Anthropic Claude API client.
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl AnthropicClient {
    /// Create a new Anthropic client.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Create from environment variable.
    pub fn from_env(model: impl Into<String>) -> RMoEResult<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| RMoEError::ConfigError("ANTHROPIC_API_KEY not set".to_string()))?;
        Ok(Self::new(api_key, model))
    }

    /// Send messages request.
    async fn send_message(&self, system: &str, messages: &[AnthropicMessage], params: &InferenceParams) -> RMoEResult<String> {
        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: params.max_new_tokens as u32,
            system: Some(system.to_string()),
            messages: messages.to_vec(),
            temperature: Some(params.temperature),
        };

        debug!(model = self.model, "Sending Anthropic message request");

        let response = self.client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| RMoEError::ApiError {
                status: 0,
                message: e.to_string(),
            })?;

        let status = response.status().as_u16();
        if status == 429 {
            return Err(RMoEError::RateLimitExceeded { retry_after_ms: 60000 });
        }
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(RMoEError::ApiError { status, message: body });
        }

        let body: AnthropicResponse = response.json().await.map_err(|e| {
            RMoEError::ParseError(format!("Failed to parse Anthropic response: {}", e))
        })?;

        body.content
            .first()
            .map(|c| c.text.clone())
            .ok_or_else(|| RMoEError::ParseError("No content in response".to_string()))
    }
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: String,
}

#[async_trait]
impl TextModel for AnthropicClient {
    async fn generate(
        &self,
        system_prompt: &str,
        user_input: &str,
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        let messages = vec![AnthropicMessage {
            role: "user".to_string(),
            content: user_input.to_string(),
        }];
        self.send_message(system_prompt, &messages, params).await
    }

    async fn generate_stream(
        &self,
        system_prompt: &str,
        user_input: &str,
        params: &InferenceParams,
    ) -> RMoEResult<tokio::sync::mpsc::Receiver<String>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let response = self.generate(system_prompt, user_input, params).await?;
        
        tokio::spawn(async move {
            let _ = tx.send(response).await;
        });

        Ok(rx)
    }

    fn name(&self) -> &str {
        &self.model
    }

    fn is_ready(&self) -> bool {
        true
    }
}

#[async_trait]
impl ChatModel for AnthropicClient {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        let system = messages.iter()
            .find(|m| matches!(m.role, ChatRole::System))
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .filter(|m| !matches!(m.role, ChatRole::System))
            .map(|m| AnthropicMessage {
                role: match m.role {
                    ChatRole::User => "user".to_string(),
                    ChatRole::Assistant => "assistant".to_string(),
                    ChatRole::System => "user".to_string(), // Shouldn't happen
                },
                content: m.content.clone(),
            })
            .collect();

        self.send_message(system, &anthropic_messages, params).await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        params: &InferenceParams,
    ) -> RMoEResult<tokio::sync::mpsc::Receiver<String>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let response = self.chat(messages, params).await?;
        
        tokio::spawn(async move {
            let _ = tx.send(response).await;
        });

        Ok(rx)
    }

    fn name(&self) -> &str {
        &self.model
    }

    fn is_ready(&self) -> bool {
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Unified API Backend
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified API backend that can switch between providers.
pub enum ApiBackend {
    OpenAI(OpenAIClient),
    Anthropic(AnthropicClient),
}

impl ApiBackend {
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::OpenAI(OpenAIClient::new(api_key, model))
    }

    pub fn anthropic(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::Anthropic(AnthropicClient::new(api_key, model))
    }

    pub fn name(&self) -> &str {
        match self {
            Self::OpenAI(c) => c.name(),
            Self::Anthropic(c) => c.name(),
        }
    }
}

#[async_trait]
impl TextModel for ApiBackend {
    async fn generate(
        &self,
        system_prompt: &str,
        user_input: &str,
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        match self {
            Self::OpenAI(c) => c.generate(system_prompt, user_input, params).await,
            Self::Anthropic(c) => c.generate(system_prompt, user_input, params).await,
        }
    }

    async fn generate_stream(
        &self,
        system_prompt: &str,
        user_input: &str,
        params: &InferenceParams,
    ) -> RMoEResult<tokio::sync::mpsc::Receiver<String>> {
        match self {
            Self::OpenAI(c) => c.generate_stream(system_prompt, user_input, params).await,
            Self::Anthropic(c) => c.generate_stream(system_prompt, user_input, params).await,
        }
    }

    fn name(&self) -> &str {
        self.name()
    }

    fn is_ready(&self) -> bool {
        true
    }
}
