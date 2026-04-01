//! # rmoe-rag
//!
//! Retrieval-Augmented Generation for clinical knowledge.
//!
//! Features:
//! - BM25 keyword search
//! - Vector similarity search
//! - Hybrid retrieval
//! - Document chunking

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

pub mod bm25;
pub mod vector;
pub mod chunker;
pub mod retriever;

pub use bm25::*;
pub use vector::*;
pub use chunker::*;
pub use retriever::*;

/// A document in the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier
    pub id: String,
    /// Document content
    pub content: String,
    /// Document title
    pub title: Option<String>,
    /// Source (file path, URL, etc.)
    pub source: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// A retrieved result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// The document
    pub document: Document,
    /// Relevance score
    pub score: f64,
    /// Matched keywords (for BM25)
    pub matched_keywords: Vec<String>,
}

/// RAG configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Maximum documents to retrieve
    pub top_k: usize,
    /// Minimum relevance score
    pub min_score: f64,
    /// Retrieval strategy
    pub strategy: RetrievalStrategy,
    /// Chunk size for documents
    pub chunk_size: usize,
    /// Chunk overlap
    pub chunk_overlap: usize,
}

/// Retrieval strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetrievalStrategy {
    /// BM25 keyword search
    BM25,
    /// Vector similarity search
    Vector,
    /// Hybrid (BM25 + Vector)
    Hybrid,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.3,
            strategy: RetrievalStrategy::BM25,
            chunk_size: 512,
            chunk_overlap: 64,
        }
    }
}

/// Knowledge base for RAG.
pub struct KnowledgeBase {
    /// Stored documents
    documents: Vec<Document>,
    /// BM25 index
    bm25_index: BM25Index,
    /// Configuration
    config: RagConfig,
}

impl KnowledgeBase {
    pub fn new(config: RagConfig) -> Self {
        Self {
            documents: Vec::new(),
            bm25_index: BM25Index::new(),
            config,
        }
    }

    /// Add a document to the knowledge base.
    pub fn add_document(&mut self, doc: Document) {
        self.bm25_index.add_document(&doc.id, &doc.content);
        self.documents.push(doc);
    }

    /// Add multiple documents.
    pub fn add_documents(&mut self, docs: Vec<Document>) {
        for doc in docs {
            self.add_document(doc);
        }
    }

    /// Retrieve relevant documents.
    pub fn retrieve(&self, query: &str) -> Vec<RetrievalResult> {
        match self.config.strategy {
            RetrievalStrategy::BM25 => self.retrieve_bm25(query),
            RetrievalStrategy::Vector => self.retrieve_vector(query),
            RetrievalStrategy::Hybrid => self.retrieve_hybrid(query),
        }
    }

    fn retrieve_bm25(&self, query: &str) -> Vec<RetrievalResult> {
        let scores = self.bm25_index.search(query, self.config.top_k);
        
        scores.into_iter()
            .filter_map(|(doc_id, score)| {
                if score < self.config.min_score {
                    return None;
                }
                self.documents.iter()
                    .find(|d| d.id == doc_id)
                    .map(|doc| RetrievalResult {
                        document: doc.clone(),
                        score,
                        matched_keywords: vec![],
                    })
            })
            .collect()
    }

    fn retrieve_vector(&self, query: &str) -> Vec<RetrievalResult> {
        // Placeholder for vector search
        // Would require embedding model integration
        self.retrieve_bm25(query)
    }

    fn retrieve_hybrid(&self, query: &str) -> Vec<RetrievalResult> {
        // Combine BM25 and vector results
        self.retrieve_bm25(query)
    }

    /// Get number of documents.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Clear all documents.
    pub fn clear(&mut self) {
        self.documents.clear();
        self.bm25_index = BM25Index::new();
    }
}

impl Default for KnowledgeBase {
    fn default() -> Self {
        Self::new(RagConfig::default())
    }
}

/// Load clinical knowledge from files.
pub fn load_clinical_knowledge(_path: &str) -> Vec<Document> {
    // Implementation would load from various formats
    vec![]
}
