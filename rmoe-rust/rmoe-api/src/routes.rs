//! API route definitions.

/// API version prefix.
pub const API_V1: &str = "/api/v1";

/// Route paths.
pub mod paths {
    pub const HEALTH: &str = "/health";
    pub const VERSION: &str = "/version";
    pub const DIAGNOSE: &str = "/api/v1/diagnose";
    pub const CHAT: &str = "/api/v1/chat";
    pub const MODELS: &str = "/api/v1/models";
    pub const AGENTS: &str = "/api/v1/agents";
    pub const WEBSOCKET: &str = "/ws";
    
    // OpenAI-compatible
    pub const OPENAI_CHAT: &str = "/v1/chat/completions";
    pub const OPENAI_MODELS: &str = "/v1/models";
}
