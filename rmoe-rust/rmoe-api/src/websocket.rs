//! WebSocket handling for streaming.

use axum::extract::ws::{Message, WebSocket};
use serde::{Deserialize, Serialize};
use tracing::{info, error};

/// WebSocket message types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    /// Start a diagnostic session
    StartDiagnose {
        symptoms: Option<String>,
        image_base64: Option<String>,
    },
    /// Chat message
    Chat {
        message: String,
        model: Option<String>,
    },
    /// Token stream chunk
    Token {
        content: String,
    },
    /// Error
    Error {
        message: String,
    },
    /// Session complete
    Complete {
        result: serde_json::Value,
    },
}

/// Handle WebSocket connection.
pub async fn handle_ws_connection(mut socket: WebSocket) {
    info!("New WebSocket connection");

    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                    match ws_msg {
                        WsMessage::StartDiagnose { symptoms, image_base64 } => {
                            // Would start diagnostic pipeline
                            let response = WsMessage::Token {
                                content: "Starting diagnostic analysis...".to_string(),
                            };
                            if let Ok(json) = serde_json::to_string(&response) {
                                let _ = socket.send(Message::Text(json.into())).await;
                            }
                        }
                        WsMessage::Chat { message, model } => {
                            // Would process chat
                            let response = WsMessage::Token {
                                content: format!("Received: {}", message),
                            };
                            if let Ok(json) = serde_json::to_string(&response) {
                                let _ = socket.send(Message::Text(json.into())).await;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket connection closed");
                break;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
}
