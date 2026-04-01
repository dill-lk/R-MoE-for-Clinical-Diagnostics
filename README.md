<div align="center">

# 🧠 R-MoE Engine

**Run Any AI Model. Anywhere. With Intelligence.**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)](#installation)
[![License](https://img.shields.io/badge/license-Apache-green)](#license)
[![Paper](https://img.shields.io/badge/paper-PDF-blue)](paper/R_MoE.pdf)

*High-performance Recursive Multi-Agent Mixture-of-Experts Framework for Clinical Diagnostics*

</div>

---

> ⚠️ **MEDICAL DISCLAIMER**
>
> This system is designed for **research and educational purposes only**.
> It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.
> Always consult qualified healthcare professionals for medical decisions.
> The developers assume no liability for clinical use of this software.

---

## What is R-MoE?

R-MoE is a high-performance, modular AI framework capable of:

- 🔥 Running **local models** (GGUF via llama.cpp, ONNX)
- ☁️ Connecting to **external APIs** (OpenAI, Anthropic, Google, Azure, Groq, Together, Mistral, Ollama, OpenRouter)
- 🤖 Orchestrating **multi-agent Mixture-of-Experts** pipelines
- ⚡ Delivering **fast, reliable, and developer-friendly inference**

R-MoE addresses *diagnostic hallucinations* in medical AI by replacing monolithic
vision-language models with a **three-phase recursive agent pipeline** that mimics
the dual-process cognitive workflow of human radiologists.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          R-MoE Engine                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │     MPE      │───▶│     ARLL     │───▶│     CSR      │      │
│  │  Perception  │    │  Reasoning   │    │  Clinical    │      │
│  │  (Vision)    │    │  (Logic)     │    │  (Synthesis) │      │
│  └──────────────┘    └──────┬───────┘    └──────────────┘      │
│                             │                                   │
│                      ┌──────┴───────┐                          │
│                      │ #wanna#      │                          │
│                      │ Protocol     │                          │
│                      │ Sc ≥ 0.90?   │                          │
│                      └──────────────┘                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  🔌 Providers: OpenAI │ Anthropic │ Google │ Azure │ Groq     │
│                Together │ Mistral │ Ollama │ OpenRouter        │
└─────────────────────────────────────────────────────────────────┘
```

### The Three-Phase Pipeline

1. **MPE (Multi-modal Perception Engine)** - Vision processing with Qwen2-VL / Moondream2
2. **ARLL (Agentic Reasoning & Logic Layer)** - Chain-of-thought reasoning with DeepSeek-R1
3. **CSR (Clinical Synthesis & Reporting)** - Final report with MedGemma / clinical models

### #wanna# Protocol

Recursive confidence gating: `Sc = 1 - σ²` (where σ² = variance of DDx probabilities)

- Threshold: **θ = 0.90**
- Max iterations: **3** before human escalation
- Feedback types: High-res crop, Alternate view, Modality escalation

### Key Results (Paper §5)

| System | F1 | Type I Err % | ECE | Latency (s) |
|--------|-----|--------------|-----|-------------|
| **R-MoE (ours)** | **0.92** | **5.2** | **0.08** | 45 |
| GPT-4V | 0.85 | 7.8 | 0.15 | 32 |
| Gemini 1.5 Pro | 0.87 | 7.1 | 0.13 | 38 |

**25% false-positive reduction** · **47% ECE improvement** · **18% better temporal tracking**

---

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) 1.75 or later
- (Optional) CUDA toolkit for GPU acceleration
- (Optional) Ollama for local model serving

### Build from Source

```bash
git clone https://github.com/your-repo/R-MoE-for-Clinical-Diagnostics
cd R-MoE-for-Clinical-Diagnostics/rmoe-rust

# Build release binary
cargo build --release

# Install to path
cargo install --path rmoe-cli
```

### Verify Installation

```bash
rmoe --version
rmoe --help
```

---

## Quick Start

### 1. Configure API Providers

```bash
# Set environment variables for your providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Or use the CLI to configure
rmoe config init
rmoe api add openai --env
rmoe api add anthropic --env
```

### 2. Interactive Chat

```bash
# Start chat with default model
rmoe chat

# Use specific model
rmoe chat --model anthropic:claude-sonnet-4-20250514

# With custom system prompt
rmoe chat --system "You are a medical expert assistant."
```

### 3. Run Clinical Diagnosis

```bash
# Full diagnostic pipeline
rmoe diagnose --symptoms "chest pain, shortness of breath" \
              --vision-model openai:gpt-4o \
              --reasoning-model anthropic:claude-sonnet-4-20250514 \
              --clinical-model openai:gpt-4o

# With medical image
rmoe diagnose --image chest_xray.png \
              --symptoms "persistent cough for 2 weeks"

# With patient history
rmoe diagnose --image scan.dcm \
              --symptoms "acute abdominal pain" \
              --history "Previous appendectomy, hypertension"
```

### 4. Direct Model Inference

```bash
# Run single inference
rmoe run openai:gpt-4o --prompt "Explain myocardial infarction"

# With file input
rmoe run anthropic:claude-sonnet-4-20250514 --file query.txt

# Local model via Ollama
rmoe run ollama:llama3.1:70b --prompt "Medical query..."
```

---

## Supported Providers

| Provider | Environment Variable | Models |
|----------|---------------------|--------|
| OpenAI | `OPENAI_API_KEY` | gpt-4o, gpt-4-turbo, gpt-4-vision |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514, claude-3-5-sonnet, claude-3-opus |
| Google | `GOOGLE_API_KEY` | gemini-1.5-pro, gemini-1.5-flash |
| Azure | `AZURE_OPENAI_API_KEY` | gpt-4o (deployment-based) |
| Groq | `GROQ_API_KEY` | llama-3.1-70b, mixtral-8x7b |
| Together | `TOGETHER_API_KEY` | Meta-Llama-3.1-70B-Instruct |
| Mistral | `MISTRAL_API_KEY` | mistral-large-latest |
| Ollama | (none needed) | Any local model |
| OpenRouter | `OPENROUTER_API_KEY` | Multi-provider routing |

---

## CLI Reference

```
USAGE:
    rmoe <COMMAND>

COMMANDS:
    run        Run inference on a model
    chat       Interactive chat mode
    diagnose   Run clinical diagnostic pipeline
    api        Manage API providers
    model      Manage local models
    list       List available providers and models
    config     Configuration management
    bench      Run benchmarks

OPTIONS:
    -c, --config <FILE>    Path to configuration file
    -v, --verbose          Verbose output
    -h, --help             Print help
    -V, --version          Print version
```

### Examples

```bash
# List all configured providers
rmoe api list

# Test API connection
rmoe api test openai

# List recommended models
rmoe list --models

# Run benchmark
rmoe bench openai:gpt-4o --iterations 10

# Show current config
rmoe config show
```

---

## API Server

R-MoE includes a REST/WebSocket API server compatible with OpenAI's API format:

```bash
# Start the server
cargo run --release -p rmoe-api

# Or after installation
rmoe-api --host 0.0.0.0 --port 8080
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/models` | GET | List available models |
| `/api/v1/diagnose` | POST | Clinical diagnosis |
| `/ws` | WebSocket | Streaming interface |

### Example Request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rmoe-diagnostic",
    "messages": [{"role": "user", "content": "Analyze chest pain symptoms"}]
  }'
```

---

## Project Structure

```
rmoe-rust/
├── Cargo.toml              # Workspace configuration
├── rmoe-core/              # Core engine, types, traits
│   ├── src/
│   │   ├── lib.rs          # Main exports
│   │   ├── models.rs       # Data structures (DDxEnsemble, WannaState)
│   │   ├── traits.rs       # Model trait definitions
│   │   ├── engine.rs       # DiagnosticEngine, WannaStateMachine
│   │   ├── config.rs       # Configuration management
│   │   └── error.rs        # Error types
├── rmoe-models/            # Model backends
│   ├── src/
│   │   ├── gguf.rs         # Local GGUF models (llama.cpp)
│   │   ├── api.rs          # API model wrapper
│   │   └── providers/      # Provider implementations
│   │       ├── openai.rs
│   │       ├── anthropic.rs
│   │       ├── google.rs
│   │       ├── azure.rs
│   │       ├── groq.rs
│   │       ├── together.rs
│   │       ├── mistral.rs
│   │       ├── ollama.rs
│   │       └── openrouter.rs
├── rmoe-agents/            # Agent implementations
│   ├── src/
│   │   ├── mpe.rs          # Multi-modal Perception Engine
│   │   ├── arll.rs         # Agentic Reasoning & Logic Layer
│   │   └── csr.rs          # Clinical Synthesis & Reporting
├── rmoe-router/            # MoE routing logic
├── rmoe-memory/            # Context & conversation memory
├── rmoe-rag/               # RAG engine (BM25, vector search)
├── rmoe-api/               # REST/WebSocket server
└── rmoe-cli/               # CLI interface
```

---

## Configuration

### Configuration File

Default location: `~/.rmoe/config.toml`

```toml
# Default models for diagnostic pipeline
default_vision_model = "openai:gpt-4o"
default_reasoning_model = "anthropic:claude-sonnet-4-20250514"
default_clinical_model = "openai:gpt-4o"

# #wanna# protocol settings
confidence_threshold = 0.90
max_iterations = 3

# API configurations
[apis.openai]
provider = "openai"
api_key_env = "OPENAI_API_KEY"
default_model = "gpt-4o"

[apis.anthropic]
provider = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-sonnet-4-20250514"
```

### Inference Parameters

```toml
[inference]
temperature = 0.2
max_new_tokens = 512
top_p = 0.95
top_k = 40
n_ctx = 2048
```

---

## Research Notebook

See [`research/advanced_rmoe_demo.ipynb`](research/advanced_rmoe_demo.ipynb) for:

- Multi-agent simulation
- Routing visualization
- Confidence scoring analysis
- API + local hybrid inference
- Benchmarking (latency + accuracy)
- Comparative evaluation

---

## Paper

Full research paper: [`paper/R_MoE.pdf`](paper/R_MoE.pdf)

The paper details:
- Theoretical framework for recursive MoE
- #wanna# protocol formalization
- Experimental methodology
- Ablation studies
- Clinical validation results

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Run tests
cargo test --workspace

# Run with debug logging
RUST_LOG=debug cargo run -p rmoe-cli -- chat

# Format code
cargo fmt --all

# Run lints
cargo clippy --workspace
```

---

## License

Apache 2.0 License. See [LICENSE](LICENSE).

---

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF inference
- [tokio](https://tokio.rs/) for async runtime
- [axum](https://github.com/tokio-rs/axum) for web framework

---

<div align="center">

**"Run Any Model. Anywhere. With Intelligence."**

</div>
