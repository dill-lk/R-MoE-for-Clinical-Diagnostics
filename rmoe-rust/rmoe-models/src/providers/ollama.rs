//! Ollama local server API client.

use async_trait::async_trait;
use rmoe_core::{InferenceParams, RMoEError, RMoEResult};
use serde::{Deserialize, Serialize};
use tracing::debug;
use futures::StreamExt;

use super::{ChatCompletionMessage, MessageContent, ContentPart, ProviderClient, ProviderConfig};

/// Ollama local server API client.
pub struct OllamaClient {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl OllamaClient {
    pub fn new(config: ProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }
}

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

impl From<&ChatCompletionMessage> for OllamaMessage {
    fn from(msg: &ChatCompletionMessage) -> Self {
        match &msg.content {
            MessageContent::Text(text) => Self {
                role: msg.role.clone(),
                content: text.clone(),
                images: None,
            },
            MessageContent::Multimodal(parts) => {
                let mut text_parts = Vec::new();
                let mut images = Vec::new();

                for part in parts {
                    match part {
                        ContentPart::Text { text } => text_parts.push(text.clone()),
                        ContentPart::ImageUrl { image_url } => {
                            // Extract base64 data
                            let url = &image_url.url;
                            if url.starts_with("data:") {
                                if let Some(pos) = url.find(",") {
                                    images.push(url[pos + 1..].to_string());
                                }
                            }
                        }
                    }
                }

                Self {
                    role: msg.role.clone(),
                    content: text_parts.join(" "),
                    images: if images.is_empty() { None } else { Some(images) },
                }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: Option<OllamaResponseMessage>,
    done: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaResponseMessage {
    content: String,
}

#[async_trait]
impl ProviderClient for OllamaClient {
    async fn chat_completion(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        let request = OllamaRequest {
            model: self.config.model.clone(),
            messages: messages.iter().map(OllamaMessage::from).collect(),
            options: Some(OllamaOptions {
                temperature: Some(params.temperature),
                num_predict: Some(params.max_new_tokens as u32),
                top_p: Some(params.top_p),
                top_k: Some(params.top_k as u32),
            }),
            stream: false,
        };

        debug!(model = %self.config.model, "Sending Ollama request");

        let response = self.client
            .post(format!("{}/chat", self.config.base_url))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| RMoEError::ApiError {
                status: 0,
                message: format!("Ollama request failed (is Ollama running?): {}", e),
            })?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(RMoEError::ApiError { status, message: body });
        }

        let body: OllamaResponse = response.json().await.map_err(|e| {
            RMoEError::ParseError(format!("Failed to parse Ollama response: {}", e))
        })?;

        body.message
            .map(|m| m.content)
            .ok_or_else(|| RMoEError::ParseError("No message in response".to_string()))
    }

    async fn chat_completion_stream(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<tokio::sync::mpsc::Receiver<String>> {
        let request = OllamaRequest {
            model: self.config.model.clone(),
            messages: messages.iter().map(OllamaMessage::from).collect(),
            options: Some(OllamaOptions {
                temperature: Some(params.temperature),
                num_predict: Some(params.max_new_tokens as u32),
                top_p: Some(params.top_p),
                top_k: Some(params.top_k as u32),
            }),
            stream: true,
        };

        let (tx, rx) = tokio::sync::mpsc::channel(100);

        let response = self.client
            .post(format!("{}/chat", self.config.base_url))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| RMoEError::ApiError {
                status: 0,
                message: format!("Request failed: {}", e),
            })?;

        tokio::spawn(async move {
            let mut stream = response.bytes_stream();

            while let Some(chunk) = stream.next().await {
                if let Ok(bytes) = chunk {
                    if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                        for line in text.lines() {
                            if let Ok(response) = serde_json::from_str::<OllamaResponse>(line) {
                                if let Some(msg) = response.message {
                                    if tx.send(msg.content).await.is_err() {
                                        break;
                                    }
                                }
                                if response.done {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    async fn vision_completion(
        &self,
        messages: &[ChatCompletionMessage],
        _image_data: &[u8],
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        // Ollama vision uses the same endpoint with images in message
        self.chat_completion(messages, params).await
    }

    fn provider_name(&self) -> &str {
        "Ollama"
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn supports_vision(&self) -> bool {
        self.config.model.contains("llava") || 
        self.config.model.contains("bakllava") ||
        self.config.model.contains("vision")
    }
}
