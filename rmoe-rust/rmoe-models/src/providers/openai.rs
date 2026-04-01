//! OpenAI API client implementation.

use async_trait::async_trait;
use rmoe_core::{InferenceParams, RMoEError, RMoEResult};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use futures::StreamExt;

use super::{ChatCompletionMessage, MessageContent, ProviderClient, ProviderConfig};

/// OpenAI API client.
pub struct OpenAIClient {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl OpenAIClient {
    pub fn new(config: ProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    fn build_request(&self, messages: &[ChatCompletionMessage], params: &InferenceParams, stream: bool) -> OpenAIRequest {
        OpenAIRequest {
            model: self.config.model.clone(),
            messages: messages.iter().map(|m| OpenAIMessage::from(m.clone())).collect(),
            temperature: Some(params.temperature),
            max_tokens: Some(params.max_new_tokens as u32),
            top_p: Some(params.top_p),
            frequency_penalty: Some((params.repeat_penalty - 1.0).clamp(-2.0, 2.0)),
            stream: Some(stream),
        }
    }

    async fn send_request(&self, request: &OpenAIRequest) -> RMoEResult<reqwest::Response> {
        let mut req_builder = self.client
            .post(format!("{}/chat/completions", self.config.base_url))
            .header("Content-Type", "application/json");

        if let Some(ref key) = self.config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        if let Some(ref org) = self.config.organization {
            req_builder = req_builder.header("OpenAI-Organization", org);
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
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(60);
            return Err(RMoEError::RateLimitExceeded {
                retry_after_ms: retry_after * 1000,
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
struct OpenAIRequest {
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
    content: serde_json::Value,
}

impl From<ChatCompletionMessage> for OpenAIMessage {
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
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamResponse {
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIDelta,
}

#[derive(Debug, Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
}

#[async_trait]
impl ProviderClient for OpenAIClient {
    async fn chat_completion(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        let request = self.build_request(messages, params, false);
        
        debug!(
            model = %self.config.model,
            messages = messages.len(),
            "Sending OpenAI chat completion request"
        );

        let response = self.send_request(&request).await?;
        let body: OpenAIResponse = response.json().await.map_err(|e| {
            RMoEError::ParseError(format!("Failed to parse OpenAI response: {}", e))
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
                        
                        // Process complete SSE lines
                        while let Some(pos) = buffer.find("\n\n") {
                            let line = buffer[..pos].trim();
                            buffer = buffer[pos + 2..].to_string();

                            if line.starts_with("data: ") {
                                let data = &line[6..];
                                if data == "[DONE]" {
                                    break;
                                }
                                if let Ok(parsed) = serde_json::from_str::<OpenAIStreamResponse>(data) {
                                    if let Some(content) = parsed.choices.first()
                                        .and_then(|c| c.delta.content.clone())
                                    {
                                        if tx.send(content).await.is_err() {
                                            break;
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
        // OpenAI vision uses the same endpoint, image is embedded in messages
        self.chat_completion(messages, params).await
    }

    fn provider_name(&self) -> &str {
        "OpenAI"
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn supports_vision(&self) -> bool {
        self.config.model.contains("vision") || 
        self.config.model.contains("gpt-4o") ||
        self.config.model.contains("gpt-4-turbo")
    }
}
