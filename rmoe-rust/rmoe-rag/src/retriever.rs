//! Document retriever interface.

use super::{Document, RetrievalResult};

/// Trait for document retrieval.
pub trait Retriever: Send + Sync {
    /// Retrieve relevant documents for a query.
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<RetrievalResult>;
    
    /// Add a document to the retriever.
    fn add_document(&mut self, doc: Document);
    
    /// Get number of documents.
    fn len(&self) -> usize;
    
    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Build retrieval context for prompt augmentation.
pub fn build_retrieval_context(results: &[RetrievalResult], max_tokens: usize) -> String {
    let mut context = String::new();
    let mut estimated_tokens = 0;

    for result in results {
        let doc_tokens = result.document.content.len() / 4; // Rough estimate
        
        if estimated_tokens + doc_tokens > max_tokens {
            break;
        }

        if !context.is_empty() {
            context.push_str("\n\n---\n\n");
        }

        if let Some(title) = &result.document.title {
            context.push_str(&format!("**{}**\n\n", title));
        }

        context.push_str(&result.document.content);
        estimated_tokens += doc_tokens;
    }

    context
}

/// Format retrieval results for display.
pub fn format_results(results: &[RetrievalResult]) -> String {
    results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let title = r.document.title.as_deref().unwrap_or("Untitled");
            let preview = if r.document.content.len() > 100 {
                format!("{}...", &r.document.content[..100])
            } else {
                r.document.content.clone()
            };
            format!(
                "{}. [Score: {:.2}] {}\n   {}",
                i + 1,
                r.score,
                title,
                preview
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}
