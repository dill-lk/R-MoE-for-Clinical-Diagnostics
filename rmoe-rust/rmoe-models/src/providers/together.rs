//! Together AI API client.

use async_trait::async_trait;
use rmoe_core::{InferenceParams, RMoEError, RMoEResult};
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::{ChatCompletionMessage, MessageContent, ProviderClient, ProviderConfig};

/// Together AI API client - uses OpenAI-compatible API.
pub struct TogetherClient {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl TogetherClient {
    pub fn new(config: ProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }
}

#[derive(Debug, Serialize)]
struct TogetherRequest {
    model: String,
    messages: Vec<TogetherMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TogetherMessage {
    role: String,
    content: String,
}

impl From<&ChatCompletionMessage> for TogetherMessage {
    fn from(msg: &ChatCompletionMessage) -> Self {
        let content = match &msg.content {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Multimodal(_) => "[Multimodal content]".to_string(),
        };
        Self {
            role: msg.role.clone(),
            content,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TogetherResponse {
    choices: Vec<TogetherChoice>,
}

#[derive(Debug, Deserialize)]
struct TogetherChoice {
    message: TogetherResponseMessage,
}

#[derive(Debug, Deserialize)]
struct TogetherResponseMessage {
    content: Option<String>,
}

#[async_trait]
impl ProviderClient for TogetherClient {
    async fn chat_completion(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        let request = TogetherRequest {
            model: self.config.model.clone(),
            messages: messages.iter().map(TogetherMessage::from).collect(),
            temperature: Some(params.temperature),
            max_tokens: Some(params.max_new_tokens as u32),
            stream: Some(false),
        };

        debug!(model = %self.config.model, "Sending Together AI request");

        let response = self.client
            .post(format!("{}/chat/completions", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_key.as_deref().unwrap_or("")))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| RMoEError::ApiError {
                status: 0,
                message: format!("Request failed: {}", e),
            })?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(RMoEError::ApiError { status, message: body });
        }

        let body: TogetherResponse = response.json().await.map_err(|e| {
            RMoEError::ParseError(format!("Failed to parse Together response: {}", e))
        })?;

        body.choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| RMoEError::ParseError("No content in response".to_string()))
    }

    async fn chat_completion_stream(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<tokio::sync::mpsc::Receiver<String>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let response = self.chat_completion(messages, params).await?;
        
        tokio::spawn(async move {
            let _ = tx.send(response).await;
        });

        Ok(rx)
    }

    async fn vision_completion(
        &self,
        messages: &[ChatCompletionMessage],
        _image_data: &[u8],
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        self.chat_completion(messages, params).await
    }

    fn provider_name(&self) -> &str {
        "Together AI"
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn supports_vision(&self) -> bool {
        self.config.model.contains("vision") || self.config.model.contains("llava")
    }
}
