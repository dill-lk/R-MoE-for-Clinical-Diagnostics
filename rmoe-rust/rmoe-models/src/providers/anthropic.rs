//! Anthropic Claude API client implementation.

use async_trait::async_trait;
use rmoe_core::{InferenceParams, RMoEError, RMoEResult};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};
use futures::StreamExt;

use super::{ChatCompletionMessage, MessageContent, ContentPart, ProviderClient, ProviderConfig};

/// Anthropic Claude API client.
pub struct AnthropicClient {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl AnthropicClient {
    pub fn new(config: ProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    fn build_request(&self, messages: &[ChatCompletionMessage], params: &InferenceParams, stream: bool) -> AnthropicRequest {
        // Extract system message if present
        let system = messages.iter()
            .find(|m| m.role == "system")
            .and_then(|m| match &m.content {
                MessageContent::Text(t) => Some(t.clone()),
                _ => None,
            });

        // Convert non-system messages to Anthropic format
        let anthropic_messages: Vec<AnthropicMessage> = messages.iter()
            .filter(|m| m.role != "system")
            .map(|m| AnthropicMessage::from(m.clone()))
            .collect();

        AnthropicRequest {
            model: self.config.model.clone(),
            max_tokens: params.max_new_tokens as u32,
            system,
            messages: anthropic_messages,
            temperature: Some(params.temperature),
            stream: Some(stream),
        }
    }

    async fn send_request(&self, request: &AnthropicRequest) -> RMoEResult<reqwest::Response> {
        let mut req_builder = self.client
            .post(format!("{}/messages", self.config.base_url))
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01");

        if let Some(ref key) = self.config.api_key {
            req_builder = req_builder.header("x-api-key", key);
        }

        for (key, value) in &self.config.headers {
            req_builder = req_builder.header(key, value);
        }

        let response = req_builder
            .json(request)
            .send()
            .await
            .map_err(|e| RMoEError::ApiError {
                status: 0,
                message: format!("Request failed: {}", e),
            })?;

        let status = response.status().as_u16();

        if status == 429 {
            return Err(RMoEError::RateLimitExceeded {
                retry_after_ms: 60000,
            });
        }

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(RMoEError::ApiError {
                status,
                message: body,
            });
        }

        Ok(response)
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
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

impl From<ChatCompletionMessage> for AnthropicMessage {
    fn from(msg: ChatCompletionMessage) -> Self {
        let role = if msg.role == "assistant" {
            "assistant"
        } else {
            "user"
        }.to_string();

        let content = match msg.content {
            MessageContent::Text(text) => vec![AnthropicContent::Text { text }],
            MessageContent::Multimodal(parts) => {
                parts.into_iter().map(|part| match part {
                    ContentPart::Text { text } => AnthropicContent::Text { text },
                    ContentPart::ImageUrl { image_url } => {
                        // Parse data URL
                        let url = &image_url.url;
                        if url.starts_with("data:") {
                            let parts: Vec<&str> = url.splitn(2, ",").collect();
                            if parts.len() == 2 {
                                let media_type = parts[0]
                                    .strip_prefix("data:")
                                    .unwrap_or("image/png")
                                    .split(';')
                                    .next()
                                    .unwrap_or("image/png")
                                    .to_string();
                                AnthropicContent::Image {
                                    source: AnthropicImageSource {
                                        source_type: "base64".to_string(),
                                        media_type,
                                        data: parts[1].to_string(),
                                    },
                                }
                            } else {
                                AnthropicContent::Text { text: "[Image]".to_string() }
                            }
                        } else {
                            AnthropicContent::Text { text: format!("[Image: {}]", url) }
                        }
                    }
                }).collect()
            }
        };

        Self { role, content }
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicResponseContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponseContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<AnthropicDelta>,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: Option<String>,
    text: Option<String>,
}

#[async_trait]
impl ProviderClient for AnthropicClient {
    async fn chat_completion(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        let request = self.build_request(messages, params, false);
        
        debug!(
            model = %self.config.model,
            messages = messages.len(),
            "Sending Anthropic chat completion request"
        );

        let response = self.send_request(&request).await?;
        let body: AnthropicResponse = response.json().await.map_err(|e| {
            RMoEError::ParseError(format!("Failed to parse Anthropic response: {}", e))
        })?;

        Ok(body.content
            .iter()
            .filter(|c| c.content_type == "text")
            .filter_map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join(""))
    }

    async fn chat_completion_stream(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<tokio::sync::mpsc::Receiver<String>> {
        let request = self.build_request(messages, params, true);
        
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        let response = self.send_request(&request).await?;
        
        tokio::spawn(async move {
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        
                        while let Some(pos) = buffer.find("\n\n") {
                            let line = buffer[..pos].trim().to_string();
                            buffer = buffer[pos + 2..].to_string();

                            if line.starts_with("data: ") {
                                let data = &line[6..];
                                if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                                    if event.event_type == "content_block_delta" {
                                        if let Some(delta) = event.delta {
                                            if let Some(text) = delta.text {
                                                if tx.send(text).await.is_err() {
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Stream error: {}", e);
                        break;
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
        // Anthropic vision uses the same endpoint
        self.chat_completion(messages, params).await
    }

    fn provider_name(&self) -> &str {
        "Anthropic"
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn supports_vision(&self) -> bool {
        self.config.model.contains("claude-3") || self.config.model.contains("claude-sonnet-4")
    }
}
