//! Google Gemini API client implementation.

use async_trait::async_trait;
use rmoe_core::{InferenceParams, RMoEError, RMoEResult};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use super::{ChatCompletionMessage, MessageContent, ContentPart, ProviderClient, ProviderConfig};

/// Google Gemini API client.
pub struct GoogleClient {
    config: ProviderConfig,
    client: reqwest::Client,
}

impl GoogleClient {
    pub fn new(config: ProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    fn build_request(&self, messages: &[ChatCompletionMessage], params: &InferenceParams) -> GeminiRequest {
        let contents: Vec<GeminiContent> = messages.iter()
            .filter(|m| m.role != "system")
            .map(|m| GeminiContent::from(m.clone()))
            .collect();

        let system_instruction = messages.iter()
            .find(|m| m.role == "system")
            .and_then(|m| match &m.content {
                MessageContent::Text(t) => Some(GeminiSystemInstruction {
                    parts: vec![GeminiPart::Text { text: t.clone() }],
                }),
                _ => None,
            });

        GeminiRequest {
            contents,
            system_instruction,
            generation_config: Some(GeminiGenerationConfig {
                temperature: Some(params.temperature),
                max_output_tokens: Some(params.max_new_tokens as u32),
                top_p: Some(params.top_p),
                top_k: Some(params.top_k as u32),
            }),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    InlineData { inline_data: GeminiInlineData },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

impl From<ChatCompletionMessage> for GeminiContent {
    fn from(msg: ChatCompletionMessage) -> Self {
        let role = if msg.role == "assistant" { "model" } else { "user" }.to_string();

        let parts = match msg.content {
            MessageContent::Text(text) => vec![GeminiPart::Text { text }],
            MessageContent::Multimodal(content_parts) => {
                content_parts.into_iter().map(|part| match part {
                    ContentPart::Text { text } => GeminiPart::Text { text },
                    ContentPart::ImageUrl { image_url } => {
                        let url = &image_url.url;
                        if url.starts_with("data:") {
                            let parts: Vec<&str> = url.splitn(2, ",").collect();
                            if parts.len() == 2 {
                                let mime_type = parts[0]
                                    .strip_prefix("data:")
                                    .unwrap_or("image/png")
                                    .split(';')
                                    .next()
                                    .unwrap_or("image/png")
                                    .to_string();
                                GeminiPart::InlineData {
                                    inline_data: GeminiInlineData {
                                        mime_type,
                                        data: parts[1].to_string(),
                                    },
                                }
                            } else {
                                GeminiPart::Text { text: "[Image]".to_string() }
                            }
                        } else {
                            GeminiPart::Text { text: format!("[Image: {}]", url) }
                        }
                    }
                }).collect()
            }
        };

        Self { role, parts }
    }
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiResponseContent,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponsePart {
    text: Option<String>,
}

#[async_trait]
impl ProviderClient for GoogleClient {
    async fn chat_completion(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<String> {
        let request = self.build_request(messages, params);
        
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.config.base_url,
            self.config.model,
            self.config.api_key.as_deref().unwrap_or("")
        );

        debug!(
            model = %self.config.model,
            "Sending Google Gemini request"
        );

        let response = self.client
            .post(&url)
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

        let body: GeminiResponse = response.json().await.map_err(|e| {
            RMoEError::ParseError(format!("Failed to parse Gemini response: {}", e))
        })?;

        body.candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .and_then(|p| p.text.clone())
            .ok_or_else(|| RMoEError::ParseError("No content in response".to_string()))
    }

    async fn chat_completion_stream(
        &self,
        messages: &[ChatCompletionMessage],
        params: &InferenceParams,
    ) -> RMoEResult<tokio::sync::mpsc::Receiver<String>> {
        // Gemini streaming uses SSE at streamGenerateContent endpoint
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
        "Google"
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn supports_vision(&self) -> bool {
        self.config.model.contains("vision") || 
        self.config.model.contains("gemini-1.5") ||
        self.config.model.contains("gemini-pro-vision")
    }
}
