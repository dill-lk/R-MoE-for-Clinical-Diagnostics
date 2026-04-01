//! Core traits for the R-MoE framework.
//!
//! These traits define the interfaces for models, agents, and routing,
//! allowing pluggable implementations (local GGUF, cloud APIs, etc.).

use async_trait::async_trait;
use crate::error::RMoEError;
use crate::models::*;

// ═══════════════════════════════════════════════════════════════════════════════
//  Model Traits
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for text-to-text inference models.
#[async_trait]
pub trait TextModel: Send + Sync {
    /// Generate text from a prompt with system context.
    async fn generate(
        &self,
        system_prompt: &str,
        user_input: &str,
        params: &InferenceParams,
    ) -> Result<String, RMoEError>;

    /// Generate with streaming output.
    async fn generate_stream(
        &self,
        system_prompt: &str,
        user_input: &str,
        params: &InferenceParams,
    ) -> Result<tokio::sync::mpsc::Receiver<String>, RMoEError>;

    /// Get model name/identifier.
    fn name(&self) -> &str;

    /// Check if model is loaded and ready.
    fn is_ready(&self) -> bool;
}

/// Trait for vision-language models that can process images.
#[async_trait]
pub trait VisionModel: Send + Sync {
    /// Generate text from image and text prompt.
    async fn generate_with_image(
        &self,
        system_prompt: &str,
        image_path: &str,
        user_text: &str,
        params: &InferenceParams,
    ) -> Result<String, RMoEError>;

    /// Get model name/identifier.
    fn name(&self) -> &str;

    /// Check if model is loaded and ready.
    fn is_ready(&self) -> bool;
}

/// Trait for embedding models (for RAG).
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Generate embedding vector for text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, RMoEError>;

    /// Embedding dimension.
    fn dimension(&self) -> usize;
}

/// Trait for chat-style models with message history.
#[async_trait]
pub trait ChatModel: Send + Sync {
    /// Generate response given message history.
    async fn chat(
        &self,
        messages: &[ChatMessage],
        params: &InferenceParams,
    ) -> Result<String, RMoEError>;

    /// Chat with streaming output.
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        params: &InferenceParams,
    ) -> Result<tokio::sync::mpsc::Receiver<String>, RMoEError>;

    /// Get model name/identifier.
    fn name(&self) -> &str;

    /// Check if model is loaded and ready.
    fn is_ready(&self) -> bool;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Agent Traits
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for R-MoE pipeline agents (MPE, ARLL, CSR).
#[async_trait]
pub trait Agent: Send + Sync {
    /// Agent identifier/name.
    fn name(&self) -> &str;

    /// Agent role description.
    fn role(&self) -> &str;

    /// Execute the agent's task.
    async fn execute(&self, input: AgentInput) -> Result<AgentOutput, RMoEError>;
}

/// Input to an agent.
#[derive(Debug, Clone)]
pub struct AgentInput {
    /// Primary input text or context
    pub context: String,
    /// Optional image path
    pub image_path: Option<String>,
    /// Prior context from previous iteration
    pub prior_context: Option<String>,
    /// Feedback from #wanna# protocol
    pub wanna_feedback: Option<FeedbackTensor>,
    /// RAG references
    pub rag_references: Vec<String>,
    /// Current iteration number
    pub iteration: usize,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl Default for AgentInput {
    fn default() -> Self {
        Self {
            context: String::new(),
            image_path: None,
            prior_context: None,
            wanna_feedback: None,
            rag_references: Vec::new(),
            iteration: 1,
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Output from an agent.
#[derive(Debug, Clone)]
pub enum AgentOutput {
    /// Perception evidence from MPE
    Perception(PerceptionEvidence),
    /// Reasoning output from ARLL
    Reasoning(ReasoningOutput),
    /// Clinical report from CSR
    Report(ClinicalReport),
    /// Raw text output
    Text(String),
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Router Traits
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for routing requests to appropriate models/agents.
#[async_trait]
pub trait Router: Send + Sync {
    /// Select the best model for a given task.
    async fn route(&self, request: &RouterRequest) -> Result<RouterDecision, RMoEError>;

    /// Get available models/endpoints.
    fn available_targets(&self) -> Vec<String>;
}

/// Request to the router.
#[derive(Debug, Clone)]
pub struct RouterRequest {
    /// Task type: "vision" | "reasoning" | "clinical" | "chat"
    pub task_type: String,
    /// Input text/prompt
    pub input: String,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Preferred model (if any)
    pub preferred_model: Option<String>,
    /// Whether to prefer local models
    pub prefer_local: bool,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: Option<u64>,
}

/// Router's decision on which target to use.
#[derive(Debug, Clone)]
pub struct RouterDecision {
    /// Selected target identifier
    pub target: String,
    /// Target type: "local" | "api"
    pub target_type: String,
    /// Confidence in the decision
    pub confidence: f64,
    /// Reason for selection
    pub reason: String,
    /// Fallback targets if primary fails
    pub fallbacks: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Model Provider Trait
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for model providers (GGUF loader, API client, etc.).
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Load a model by path or identifier.
    async fn load(&mut self, model_id: &str) -> Result<(), RMoEError>;

    /// Unload the current model.
    async fn unload(&mut self) -> Result<(), RMoEError>;

    /// Check if a model is currently loaded.
    fn is_loaded(&self) -> bool;

    /// Get the current model's name.
    fn current_model(&self) -> Option<&str>;

    /// Provider name.
    fn provider_name(&self) -> &str;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Memory / Context Traits
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for conversation memory management.
#[async_trait]
pub trait Memory: Send + Sync {
    /// Add a message to memory.
    async fn add(&mut self, message: ChatMessage) -> Result<(), RMoEError>;

    /// Get recent messages (up to limit).
    async fn recent(&self, limit: usize) -> Result<Vec<ChatMessage>, RMoEError>;

    /// Clear all messages.
    async fn clear(&mut self) -> Result<(), RMoEError>;

    /// Get total message count.
    async fn count(&self) -> usize;

    /// Summarize conversation history.
    async fn summarize(&self) -> Result<String, RMoEError>;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  RAG Traits
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for RAG retrieval engines.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Retrieve relevant documents for a query.
    async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievedDocument>, RMoEError>;

    /// Add a document to the index.
    async fn add_document(&mut self, doc: Document) -> Result<(), RMoEError>;

    /// Get index statistics.
    fn stats(&self) -> IndexStats;
}

/// A document in the RAG index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: std::collections::HashMap<String, String>,
}

/// A retrieved document with relevance score.
#[derive(Debug, Clone)]
pub struct RetrievedDocument {
    pub document: Document,
    pub score: f64,
}

/// Index statistics.
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub total_documents: usize,
    pub total_tokens: usize,
}

use serde::{Serialize, Deserialize};
