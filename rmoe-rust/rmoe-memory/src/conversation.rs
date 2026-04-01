//! Conversation memory with sliding window.

use std::collections::VecDeque;
use super::{MemoryEntry, MemoryEntryType};

/// Conversation memory with configurable window size.
pub struct ConversationMemory {
    /// Memory entries
    entries: VecDeque<MemoryEntry>,
    /// Maximum entries to keep
    max_entries: usize,
}

impl ConversationMemory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries),
            max_entries,
        }
    }

    /// Add an entry to memory.
    pub fn add(&mut self, entry: MemoryEntry) {
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Get recent entries.
    pub fn recent(&self, count: usize) -> Vec<&MemoryEntry> {
        self.entries.iter().rev().take(count).rev().collect()
    }

    /// Get all entries.
    pub fn all(&self) -> Vec<&MemoryEntry> {
        self.entries.iter().collect()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get total character count.
    pub fn total_chars(&self) -> usize {
        self.entries.iter().map(|e| e.content.len()).sum()
    }

    /// Get entries by type.
    pub fn by_type(&self, entry_type: MemoryEntryType) -> Vec<&MemoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.entry_type == entry_type)
            .collect()
    }

    /// Summarize old entries to save space.
    pub fn compress(&mut self, keep_recent: usize) {
        if self.entries.len() <= keep_recent {
            return;
        }

        let to_compress = self.entries.len() - keep_recent;
        let old_entries: Vec<_> = self.entries.drain(..to_compress).collect();
        
        // Create summary entry
        let summary = MemoryEntry {
            id: uuid::Uuid::new_v4().to_string(),
            entry_type: MemoryEntryType::Summary,
            content: format!("[Compressed {} previous messages]", old_entries.len()),
            timestamp: chrono::Utc::now().timestamp() as u64,
            metadata: std::collections::HashMap::new(),
            importance: 0.3,
        };

        self.entries.push_front(summary);
    }
}

impl Default for ConversationMemory {
    fn default() -> Self {
        Self::new(100)
    }
}
