//! Error types for the R-MoE framework.

use thiserror::Error;

/// Main error type for R-MoE operations.
#[derive(Error, Debug)]
pub enum RMoEError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("API error: {status} - {message}")]
    ApiError { status: u16, message: String },

    #[error("Rate limit exceeded: retry after {retry_after_ms}ms")]
    RateLimitExceeded { retry_after_ms: u64 },

    #[error("Timeout after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },

    #[error("Image processing error: {0}")]
    ImageError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Agent error in {agent}: {message}")]
    AgentError { agent: String, message: String },

    #[error("Router error: {0}")]
    RouterError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("RAG retrieval error: {0}")]
    RetrievalError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Confidence below threshold: {sc:.4} < {threshold:.2}")]
    LowConfidence { sc: f64, threshold: f64 },

    #[error("Max iterations ({max}) reached without convergence")]
    MaxIterationsReached { max: usize },

    #[error("Escalation required: {reason}")]
    EscalationRequired { reason: String },

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl RMoEError {
    /// Check if this error is recoverable (can retry).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            RMoEError::RateLimitExceeded { .. }
                | RMoEError::Timeout { .. }
                | RMoEError::ApiError { status: 429 | 500 | 502 | 503 | 504, .. }
        )
    }

    /// Check if this error requires human escalation.
    pub fn requires_escalation(&self) -> bool {
        matches!(
            self,
            RMoEError::LowConfidence { .. }
                | RMoEError::MaxIterationsReached { .. }
                | RMoEError::EscalationRequired { .. }
        )
    }
}

/// Result type alias for R-MoE operations.
pub type RMoEResult<T> = Result<T, RMoEError>;
