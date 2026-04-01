//! BM25 (Best Matching 25) algorithm implementation.

use std::collections::HashMap;

/// BM25 index for keyword search.
pub struct BM25Index {
    /// Document term frequencies: doc_id -> term -> frequency
    doc_terms: HashMap<String, HashMap<String, usize>>,
    /// Document lengths
    doc_lengths: HashMap<String, usize>,
    /// Inverse document frequencies
    idf: HashMap<String, f64>,
    /// Average document length
    avg_doc_length: f64,
    /// BM25 parameters
    k1: f64,
    b: f64,
}

impl BM25Index {
    pub fn new() -> Self {
        Self {
            doc_terms: HashMap::new(),
            doc_lengths: HashMap::new(),
            idf: HashMap::new(),
            avg_doc_length: 0.0,
            k1: 1.5,
            b: 0.75,
        }
    }

    /// Add a document to the index.
    pub fn add_document(&mut self, doc_id: &str, content: &str) {
        let terms = self.tokenize(content);
        let doc_length = terms.len();

        let mut term_freq: HashMap<String, usize> = HashMap::new();
        for term in &terms {
            *term_freq.entry(term.clone()).or_insert(0) += 1;
        }

        self.doc_terms.insert(doc_id.to_string(), term_freq);
        self.doc_lengths.insert(doc_id.to_string(), doc_length);

        self.update_statistics();
    }

    /// Search for documents matching the query.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(String, f64)> {
        let query_terms = self.tokenize(query);
        let mut scores: HashMap<String, f64> = HashMap::new();

        for term in &query_terms {
            if let Some(idf) = self.idf.get(term) {
                for (doc_id, term_freqs) in &self.doc_terms {
                    if let Some(tf) = term_freqs.get(term) {
                        let doc_length = *self.doc_lengths.get(doc_id).unwrap_or(&1) as f64;
                        let score = self.calculate_bm25_score(
                            *tf as f64,
                            *idf,
                            doc_length,
                        );
                        *scores.entry(doc_id.clone()).or_insert(0.0) += score;
                    }
                }
            }
        }

        let mut sorted: Vec<(String, f64)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.truncate(top_k);
        sorted
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect()
    }

    fn update_statistics(&mut self) {
        let num_docs = self.doc_lengths.len();
        if num_docs == 0 {
            return;
        }

        // Calculate average document length
        let total_length: usize = self.doc_lengths.values().sum();
        self.avg_doc_length = total_length as f64 / num_docs as f64;

        // Calculate IDF for all terms
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        for term_freqs in self.doc_terms.values() {
            for term in term_freqs.keys() {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
        }

        for (term, df) in doc_freq {
            let idf = ((num_docs as f64 - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln();
            self.idf.insert(term, idf);
        }
    }

    fn calculate_bm25_score(&self, tf: f64, idf: f64, doc_length: f64) -> f64 {
        let numerator = tf * (self.k1 + 1.0);
        let denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_length / self.avg_doc_length);
        idf * numerator / denominator
    }
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_search() {
        let mut index = BM25Index::new();
        index.add_document("doc1", "Chest pain is a common symptom of myocardial infarction");
        index.add_document("doc2", "Headache can be caused by various conditions");
        index.add_document("doc3", "Cardiac arrest requires immediate intervention");

        let results = index.search("chest pain cardiac", 3);
        assert!(!results.is_empty());
        
        // doc1 should rank higher due to matching "chest" and "pain"
        assert_eq!(results[0].0, "doc1");
    }
}
