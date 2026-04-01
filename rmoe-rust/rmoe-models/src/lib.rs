//! # rmoe-models
//!
//! Production model backends for R-MoE clinical diagnostics.
//!
//! Supports:
//! - Local GGUF models via llama.cpp bindings
//! - OpenAI API (GPT-4o, GPT-4V)
//! - Anthropic API (Claude 3.5, Claude 3)
//! - Google AI (Gemini Pro, Gemini Pro Vision)
//! - Azure OpenAI
//! - Groq (fast inference)
//! - Together AI
//! - Ollama (local server)
//! - Custom OpenAI-compatible endpoints

pub mod gguf;
pub mod api;
pub mod providers;

pub use gguf::*;
pub use api::ApiBackend;
pub use providers::*;
