//! # rmoe-api
//!
//! REST and WebSocket API server for R-MoE framework.
//!
//! Features:
//! - RESTful endpoints for inference
//! - WebSocket streaming
//! - OpenAI-compatible API
//! - Health checks and metrics

use axum::{
    routing::{get, post},
    Router,
    Json,
    Extension,
    extract::{State, WebSocketUpgrade, ws::{Message, WebSocket}},
    response::IntoResponse,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::{info, error};

pub mod handlers;
pub mod routes;
pub mod openai_compat;
pub mod websocket;

pub use handlers::*;
pub use routes::*;
pub use openai_compat::*;

/// API server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Host to bind to
    pub host: String,
    /// Port to bind to
    pub port: u16,
    /// Enable CORS
    pub cors: bool,
    /// API key for authentication (optional)
    pub api_key: Option<String>,
    /// Maximum request body size
    pub max_body_size: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            cors: true,
            api_key: None,
            max_body_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

/// Application state shared across handlers.
pub struct AppState {
    /// Configuration
    pub config: ApiConfig,
    /// Request counter
    pub request_count: std::sync::atomic::AtomicU64,
}

impl AppState {
    pub fn new(config: ApiConfig) -> Self {
        Self {
            config,
            request_count: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_secs: u64,
}

/// Error response.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: u16,
}

/// Start the API server.
pub async fn start_server(config: ApiConfig) -> anyhow::Result<()> {
    let state = Arc::new(AppState::new(config.clone()));

    let app = create_router(state.clone());

    let addr = format!("{}:{}", config.host, config.port);
    info!("Starting R-MoE API server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Create the API router.
pub fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::permissive();

    Router::new()
        // Health and status
        .route("/health", get(health_handler))
        .route("/version", get(version_handler))
        
        // Diagnostic endpoints
        .route("/api/v1/diagnose", post(diagnose_handler))
        .route("/api/v1/chat", post(chat_handler))
        
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/models", get(openai_list_models))
        
        // WebSocket
        .route("/ws", get(websocket_handler))
        
        .layer(cors)
        .with_state(state)
}

/// Health check handler.
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: 0, // Would track actual uptime
    })
}

/// Version handler.
async fn version_handler() -> &'static str {
    concat!("R-MoE API v", env!("CARGO_PKG_VERSION"))
}

/// Diagnose handler placeholder.
async fn diagnose_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DiagnoseRequest>,
) -> Result<Json<DiagnoseResponse>, (StatusCode, Json<ErrorResponse>)> {
    state.request_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    // Placeholder - would call actual diagnostic engine
    Ok(Json(DiagnoseResponse {
        success: true,
        message: "Diagnostic processing not yet implemented".to_string(),
        data: None,
    }))
}

/// Chat handler placeholder.
async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, Json<ErrorResponse>)> {
    state.request_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    Ok(Json(ChatResponse {
        message: "Chat endpoint placeholder".to_string(),
    }))
}

/// WebSocket handler.
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

async fn handle_websocket(mut socket: WebSocket, state: Arc<AppState>) {
    while let Some(msg) = socket.recv().await {
        if let Ok(Message::Text(text)) = msg {
            // Echo for now - would process diagnostic requests
            if socket.send(Message::Text(format!("Received: {}", text).into())).await.is_err() {
                break;
            }
        }
    }
}

/// Diagnose request.
#[derive(Debug, Deserialize)]
pub struct DiagnoseRequest {
    pub symptoms: Option<String>,
    pub image_base64: Option<String>,
    pub patient_history: Option<String>,
}

/// Diagnose response.
#[derive(Debug, Serialize)]
pub struct DiagnoseResponse {
    pub success: bool,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

/// Chat request.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    pub model: Option<String>,
}

/// Chat response.
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub message: String,
}
