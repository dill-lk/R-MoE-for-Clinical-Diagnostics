//! Semantic (long-term) memory storage.

use std::collections::HashMap;
use super::MemoryEntry;

/// Semantic memory for long-term storage.
pub struct SemanticMemory {
    /// Stored entries indexed by ID
    entries: HashMap<String, MemoryEntry>,
    /// Maximum entries
    max_entries: usize,
}

impl SemanticMemory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(max_entries),
            max_entries,
        }
    }

    /// Store an entry.
    pub fn store(&mut self, entry: MemoryEntry) {
        // If at capacity, remove lowest importance entry
        if self.entries.len() >= self.max_entries {
            if let Some(min_key) = self.entries
                .iter()
                .min_by(|a, b| a.1.importance.partial_cmp(&b.1.importance).unwrap())
                .map(|(k, _)| k.clone())
            {
                self.entries.remove(&min_key);
            }
        }
        self.entries.insert(entry.id.clone(), entry);
    }

    /// Retrieve an entry by ID.
    pub fn get(&self, id: &str) -> Option<&MemoryEntry> {
        self.entries.get(id)
    }

    /// Search entries by content similarity (simple keyword match).
    pub fn search(&self, query: &str, limit: usize) -> Vec<&MemoryEntry> {
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let mut results: Vec<(&MemoryEntry, usize)> = self.entries
            .values()
            .map(|entry| {
                let content_lower = entry.content.to_lowercase();
                let matches = keywords.iter()
                    .filter(|kw| content_lower.contains(*kw))
                    .count();
                (entry, matches)
            })
            .filter(|(_, matches)| *matches > 0)
            .collect();

        results.sort_by(|a, b| b.1.cmp(&a.1));
        results.into_iter().take(limit).map(|(e, _)| e).collect()
    }

    /// Get all entries sorted by importance.
    pub fn all_by_importance(&self) -> Vec<&MemoryEntry> {
        let mut entries: Vec<_> = self.entries.values().collect();
        entries.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        entries
    }

    /// Get number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove an entry.
    pub fn remove(&mut self, id: &str) -> Option<MemoryEntry> {
        self.entries.remove(id)
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for SemanticMemory {
    fn default() -> Self {
        Self::new(1000)
    }
}
