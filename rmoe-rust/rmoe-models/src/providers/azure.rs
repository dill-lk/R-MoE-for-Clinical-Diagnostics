//! Azure OpenAI API client implementation.

use async_trait::async_trait;
use rmoe_core::{InferenceParams, RMoEError, RMoEResult};
use serde::{Deserialize, Serialize};
use tracing::debug;
use futures::StreamExt;

use super::{ChatCompletionMessage, MessageContent, ProviderClient, ProviderConfig};

/// Azure OpenAI API client.
pub struct AzureClient {
    config: ProviderConfig,
    client: reqwest::Client,
    api_version: String,
}

impl AzureClient {
    pub fn new(config: ProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            client,
            api_version: "2024-02-01".to_string(),
        }
    }

    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }
}

#[derive(Debug, Serialize)]
struct AzureRequest {
    messages: Vec<AzureMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AzureMessage {
    role: String,
    content: serde_json::Value,
}

impl From<ChatCompletionMessage> for AzureMessage {
    fn from(msg: ChatCompletionMessage) -> Self {
        let content = match msg.content {
            MessageContent::Text(text) => serde_json::Value::String(text),
            MessageContent::Multimodal(parts) => {
                serde_json::to_value(parts).unwrap_or(serde_json::Value::Null)
            }
        };
        Self {
            role: msg.role,
            content,
        }
    }
}

#[derive(Debug, Deserialize)]
struct AzureResponse {
    choices: Vec<AzureChoice>,
}

#[derive(Debug, Deserialize)]
struct AzureChoice {
    message: AzureResponseMessage,
}

#[derive(Debug, Deserialize)]
struct AzureResponseMessage {
    content: Option<String>,
}

#[async_trait]
impl ProviderClient for AzureClient {
    async fn chat_completion(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        let request = AzureRequest {
            messages: messages.iter().map(|m| AzureMessage::from(m.clone())).collect(),
            temperature: Some(params.temperature),
            max_tokens: Some(params.max_new_tokens as u32),
            stream: Some(false),
        };

        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.config.base_url,
            self.config.model,
            self.api_version
        );

        debug!(model = %self.config.model, "Sending Azure OpenAI request");

        let response = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("api-key", self.config.api_key.as_deref().unwrap_or(""))
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

        let body: AzureResponse = response.json().await.map_err(|e| {
            RMoEError::ParseError(format!("Failed to parse Azure response: {}", e))
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
        "Azure OpenAI"
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn supports_vision(&self) -> bool {
        self.config.model.contains("vision") || self.config.model.contains("gpt-4o")
    }
}
