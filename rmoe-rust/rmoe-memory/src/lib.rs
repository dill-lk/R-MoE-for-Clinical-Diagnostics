//! # rmoe-memory
//!
//! Context and conversation memory management for R-MoE framework.
//!
//! Features:
//! - Conversation history with sliding window
//! - Semantic memory (long-term storage)
//! - Context compression
//! - Memory-efficient token management

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

pub mod conversation;
pub mod semantic;
pub mod compression;

pub use conversation::*;
pub use semantic::*;
pub use compression::*;

/// A single memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: String,
    /// Entry type
    pub entry_type: MemoryEntryType,
    /// Content
    pub content: String,
    /// Timestamp (Unix epoch)
    pub timestamp: u64,
    /// Metadata
    pub metadata: std::collections::HashMap<String, String>,
    /// Importance score (0.0-1.0)
    pub importance: f64,
}

/// Types of memory entries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryEntryType {
    /// User message
    UserMessage,
    /// Assistant response
    AssistantMessage,
    /// System message
    SystemMessage,
    /// Diagnostic result
    DiagnosticResult,
    /// Image analysis
    ImageAnalysis,
    /// Clinical finding
    ClinicalFinding,
    /// Summary
    Summary,
}

/// Memory manager for managing conversation and semantic memory.
pub struct MemoryManager {
    /// Short-term conversation memory
    pub conversation: ConversationMemory,
    /// Long-term semantic memory
    pub semantic: SemanticMemory,
    /// Maximum context tokens
    pub max_tokens: usize,
}

impl MemoryManager {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            conversation: ConversationMemory::new(100),
            semantic: SemanticMemory::new(1000),
            max_tokens,
        }
    }

    /// Add a user message.
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.conversation.add(MemoryEntry {
            id: uuid::Uuid::new_v4().to_string(),
            entry_type: MemoryEntryType::UserMessage,
            content: content.into(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            metadata: std::collections::HashMap::new(),
            importance: 0.5,
        });
    }

    /// Add an assistant message.
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.conversation.add(MemoryEntry {
            id: uuid::Uuid::new_v4().to_string(),
            entry_type: MemoryEntryType::AssistantMessage,
            content: content.into(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            metadata: std::collections::HashMap::new(),
            importance: 0.5,
        });
    }

    /// Get context for prompt building.
    pub fn get_context(&self, max_entries: usize) -> Vec<&MemoryEntry> {
        self.conversation.recent(max_entries)
    }

    /// Clear all memory.
    pub fn clear(&mut self) {
        self.conversation.clear();
    }

    /// Get memory statistics.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            conversation_entries: self.conversation.len(),
            semantic_entries: self.semantic.len(),
            estimated_tokens: self.estimate_tokens(),
        }
    }

    fn estimate_tokens(&self) -> usize {
        // Rough estimation: ~4 characters per token
        self.conversation.total_chars() / 4
    }
}

/// Memory statistics.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub conversation_entries: usize,
    pub semantic_entries: usize,
    pub estimated_tokens: usize,
}
