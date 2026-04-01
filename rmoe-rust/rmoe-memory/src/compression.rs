//! Memory compression utilities.

/// Compression strategy for memory.
#[derive(Debug, Clone, Copy)]
pub enum CompressionStrategy {
    /// Keep only recent N entries
    SlidingWindow(usize),
    /// Summarize old entries
    Summarize,
    /// Remove by importance threshold
    ImportanceThreshold(f64),
    /// Remove by age (seconds)
    AgeThreshold(u64),
}

/// Compress text content.
pub fn compress_text(text: &str, max_length: usize) -> String {
    if text.len() <= max_length {
        return text.to_string();
    }

    // Simple truncation with ellipsis
    let truncated = &text[..max_length.saturating_sub(3)];
    format!("{}...", truncated)
}

/// Estimate token count (rough approximation).
pub fn estimate_tokens(text: &str) -> usize {
    // Average ~4 characters per token for English text
    text.len() / 4
}

/// Split text into chunks for processing.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    
    let mut i = 0;
    while i < words.len() {
        let end = (i + chunk_size).min(words.len());
        let chunk = words[i..end].join(" ");
        chunks.push(chunk);
        
        if end >= words.len() {
            break;
        }
        i += chunk_size - overlap;
    }
    
    chunks
}
