//! OpenAI-compatible API endpoints.

use axum::{
    extract::State,
    Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{AppState, ErrorResponse};

/// OpenAI-compatible chat completion request.
#[derive(Debug, Deserialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
}

/// OpenAI message format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Serialize)]
pub struct OpenAIChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: OpenAIUsage,
}

#[derive(Debug, Serialize)]
pub struct OpenAIChoice {
    pub index: u32,
    pub message: OpenAIMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI-compatible model list.
#[derive(Debug, Serialize)]
pub struct OpenAIModelList {
    pub object: String,
    pub data: Vec<OpenAIModel>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Handle OpenAI-compatible chat completions.
pub async fn openai_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OpenAIChatRequest>,
) -> Result<Json<OpenAIChatResponse>, (StatusCode, Json<ErrorResponse>)> {
    state.request_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    // Get the last user message
    let last_message = request.messages.last()
        .map(|m| m.content.clone())
        .unwrap_or_default();

    // Placeholder response - would call actual model
    let response_content = format!(
        "This is a placeholder response from R-MoE API. Your message: '{}'",
        if last_message.len() > 50 { &last_message[..50] } else { &last_message }
    );

    let response = OpenAIChatResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: request.model,
        choices: vec![OpenAIChoice {
            index: 0,
            message: OpenAIMessage {
                role: "assistant".to_string(),
                content: response_content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: OpenAIUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    Ok(Json(response))
}

/// List available models (OpenAI-compatible).
pub async fn openai_list_models(
    State(state): State<Arc<AppState>>,
) -> Json<OpenAIModelList> {
    Json(OpenAIModelList {
        object: "list".to_string(),
        data: vec![
            OpenAIModel {
                id: "rmoe-diagnostic".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "rmoe".to_string(),
            },
            OpenAIModel {
                id: "rmoe-chat".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "rmoe".to_string(),
            },
            OpenAIModel {
                id: "rmoe-vision".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "rmoe".to_string(),
            },
        ],
    })
}
