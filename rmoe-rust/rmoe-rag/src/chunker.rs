//! Document chunking utilities.

/// Chunking strategy.
#[derive(Debug, Clone, Copy)]
pub enum ChunkingStrategy {
    /// Fixed size chunks with overlap
    FixedSize { size: usize, overlap: usize },
    /// Sentence-based chunking
    Sentence { max_sentences: usize },
    /// Paragraph-based chunking
    Paragraph,
    /// Semantic chunking (requires model)
    Semantic,
}

/// A document chunk.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chunk content
    pub content: String,
    /// Start position in original document
    pub start: usize,
    /// End position in original document
    pub end: usize,
    /// Chunk index
    pub index: usize,
}

/// Document chunker.
pub struct Chunker {
    strategy: ChunkingStrategy,
}

impl Chunker {
    pub fn new(strategy: ChunkingStrategy) -> Self {
        Self { strategy }
    }

    /// Chunk a document.
    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        match self.strategy {
            ChunkingStrategy::FixedSize { size, overlap } => {
                self.chunk_fixed_size(text, size, overlap)
            }
            ChunkingStrategy::Sentence { max_sentences } => {
                self.chunk_sentences(text, max_sentences)
            }
            ChunkingStrategy::Paragraph => self.chunk_paragraphs(text),
            ChunkingStrategy::Semantic => self.chunk_fixed_size(text, 512, 64),
        }
    }

    fn chunk_fixed_size(&self, text: &str, size: usize, overlap: usize) -> Vec<Chunk> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();

        let mut i = 0;
        let mut index = 0;

        while i < words.len() {
            let end = (i + size).min(words.len());
            let content = words[i..end].join(" ");

            let start_char = words[..i].iter().map(|w| w.len() + 1).sum::<usize>();
            let end_char = start_char + content.len();

            chunks.push(Chunk {
                content,
                start: start_char,
                end: end_char,
                index,
            });

            if end >= words.len() {
                break;
            }

            i += size - overlap;
            index += 1;
        }

        chunks
    }

    fn chunk_sentences(&self, text: &str, max_sentences: usize) -> Vec<Chunk> {
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut i = 0;
        let mut index = 0;

        while i < sentences.len() {
            let end = (i + max_sentences).min(sentences.len());
            let content = sentences[i..end].join(". ");

            chunks.push(Chunk {
                content: format!("{}.", content),
                start: 0,
                end: 0,
                index,
            });

            i = end;
            index += 1;
        }

        chunks
    }

    fn chunk_paragraphs(&self, text: &str) -> Vec<Chunk> {
        text.split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .enumerate()
            .map(|(index, content)| Chunk {
                content: content.to_string(),
                start: 0,
                end: 0,
                index,
            })
            .collect()
    }
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new(ChunkingStrategy::FixedSize {
            size: 256,
            overlap: 32,
        })
    }
}
