//! Vector similarity search (placeholder for embedding-based search).

use std::collections::HashMap;

/// Vector store for similarity search.
pub struct VectorStore {
    /// Stored vectors: id -> embedding
    vectors: HashMap<String, Vec<f64>>,
    /// Embedding dimension
    dimension: usize,
}

impl VectorStore {
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: HashMap::new(),
            dimension,
        }
    }

    /// Add a vector to the store.
    pub fn add(&mut self, id: &str, vector: Vec<f64>) {
        if vector.len() == self.dimension {
            self.vectors.insert(id.to_string(), vector);
        }
    }

    /// Search for similar vectors.
    pub fn search(&self, query: &[f64], top_k: usize) -> Vec<(String, f64)> {
        if query.len() != self.dimension {
            return vec![];
        }

        let mut scores: Vec<(String, f64)> = self.vectors
            .iter()
            .map(|(id, vec)| (id.clone(), cosine_similarity(query, vec)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);
        scores
    }

    /// Get number of vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Calculate cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Calculate Euclidean distance between two vectors.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new(768) // Common embedding dimension
    }
}
