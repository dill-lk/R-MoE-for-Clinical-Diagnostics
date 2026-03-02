
## R-MoE Clinical Engine (Research-Oriented Core)
This repository contains a runnable C++17 framework for a **Recursive Multi-Agent Mixture-of-Experts (R-MoE)** clinical pipeline with uncertainty-aware gating, model swap orchestration, prompt-driven phases, and configurable runtime settings.

all model gguf files can be access from here - https://drive.google.com/drive/folders/1NbTL4BFFrySVmFt05wEh-B1q3mqLE3C5?usp=sharing

## Architecture Flow
```text
 _________________________________________________________
|                                                         |
|          RECURSIVE MULTI-AGENT MoE (R-MoE)              |
|        "Expert-Level Autonomous Medical Logic"          |
|_________________________________________________________|
                |
  [ INPUT ] ----+----> (Raw DICOM / Medical Volumetric Data)
                |
  ______________|______________          _______________________
 |                             |        |                       |
 |  PHASE 1: MPE (Perception)  |        |  MEDICAL KNOWLEDGE    |
 |  [Qwen2-VL-72B - Adv Vision]|<-------|  (Anatomy, Pathology) |
 |   |                         |        |_______________________|
 |   +--> Dyn. Res. Adaptation  |
 |   +--> Visual Token Merger   | <--- (Reduces Latency)
 |   +--> Saliency-Aware Cropping|
 |_____________________________|
                |
         { Structured Visual Evidence }
                |
  ______________|______________          _______________________
 |                             |        |                       |
 |   PHASE 2: ARLL (Reasoning) |<-------|  VECTOR RAG ENGINE    |
 |   [DeepSeek-R1 - CoT Agent] |        |  (Gold-Std Benchmarks)|
 |   |                         |        |_______________________|
 |   +--> Chain-of-Thought (CoT)|
 |   +--> Conflict Resolution   | <--- (Agent Consensus)
 |   +--> Confidence Scoring(Sc)|
 |_____________________________|
                |
      [ CONFIDENCE GATE (Sc) ]
      Threshold Sc >= 0.90 ? ---------+
                |                     |
      (NO)      |               (YES) |          _______________
+---------------+                     |         |               |
|      [ #wanna# PROTOCOL ]           +-------->| PHASE 3: CSR  |
|      Recursive Logic Loop           |         | (Synthesis)   |
|                                     |         | [Llama-3-Med] |
|  1. Attention Refocus (MPE)         |         |_______________|
|  2. Hypothesis Re-evaluation (ARLL) |                 |
|  3. Dynamic Region Zoom             |         { CLINICAL DOC }
+---------------+                     |                 |
                |                     |        [ DOCTOR AUDIT ]
    [ RE-SCAN WITH NEW CONTEXT ]      |                 |
                |                     |        (Human-Validate)
                +---------------------+--------> [ FINAL REPORT ]
```

### Why this flow is powerful

| Expert | Model | Key Innovation |
|---|---|---|
| **MPE** (Vision) | Qwen2-VL-72B | Dynamic Resolution Adaptation · Visual Token Merger · Saliency-Aware Cropping |
| **ARLL** (Reasoning) | DeepSeek-R1 | Chain-of-Thought (CoT) · Conflict Resolution · DDx variance confidence `Sc = 1 − σ²` |
| **CSR** (Medical) | Llama-3-Medius | ICD-11 / SNOMED CT · TIRADS / BI-RADS · HITL escalation |

- **MPE** adjusts resolution dynamically so even micro-lesions are detected, and merges redundant visual tokens to keep inference fast.
- **ARLL** never guesses — it reasons step-by-step and computes `Sc = 1 − σ²` (ensemble DDx variance). If `Sc < 0.90`, the `#wanna#` token triggers a recursive zoom or alternate-view request instead of forcing a diagnosis.
- **CSR** generates a fully coded clinical document (ICD-11, SNOMED CT, risk stratification) and flags cases for radiologist review via the HITL mechanism.

## llama_decode status
- **Yes, there is a concrete `llama_decode` generation scaffold** in `ExpertSwapper::infer_text()`.
- Current implementation is token-level text generation scaffold (tokenize -> decode prompt -> greedy next-token loop).
- Full multimodal handoff (real visual token tensor transport across experts) is still a v2 task; current handoff is represented via structured text/feedback tensors.

## What's improved
- Confidence formula `Sc = 1 − σ²` implemented from DDx ensemble variance (paper Section 3.1).
- `ddx_probabilities` field on `DiagnosticData`; `ddx_variance` (σ²) field on `UncertaintyMetrics`.
- Configurable runtime model routing via CLI and JSON settings.
- Runtime image path input (`--image`) instead of hardcoding in code.
- Interactive post-diagnosis chat with Reasoning or Clinical expert.
- Uncertainty metrics per iteration (`uncertainty`, `predictive_entropy`, `ddx_variance`) for traceability.
- Iteration audit trail (decision + Sc + σ² + entropy) returned in `RunSummary`.
- Recursive abstention control with configurable gate (`max_iterations`, `confidence_threshold`).
- CSR report includes SNOMED CT, risk stratification (TIRADS/BI-RADS), treatment recommendations, and HITL flag.
- EOF-safe interactive chat loop — exits cleanly when launched from a subprocess.

## Build & Run
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

# Full runtime argument mode (new-style flags)
./build/rmoe_engine \
  --model-vision   models/vision_text.gguf \
  --model-proj     models/vision_proj.gguf \
  --model-reasoning models/reasoning_expert.gguf \
  --model-clinical  models/clinical_expert.gguf \
  --image          models/test_patient.png \
  --temp 0.6 --n-predict 512

# With settings override
./build/rmoe_engine \
  --settings settings/rmoe_settings.json \
  --image models/test_patient.png
```

## Google Colab quick-start
```python
# 1. Clone and build
!git clone --recurse-submodules https://github.com/dill-lk/Mr.ToM /content/Mr.ToM
!cmake -S /content/Mr.ToM -B /content/Mr.ToM/build -DCMAKE_BUILD_TYPE=Release
!cmake --build /content/Mr.ToM/build --config Release -j4

# 2. Put your GGUF files in MyDrive/Medical_MoE_Models/ then run:
%run /content/Mr.ToM/colab_runner.py
```

The launcher (`colab_runner.py`) automatically mounts Google Drive, copies the
model files into `/content/models/`, and runs the engine with the right flags.
See the script's docstring for custom temperature, token-budget, and image-path
options.

## CLI flags

| Flag | Alias | Description |
|---|---|---|
| `--model-vision <path>` | `--vision-text` | Vision LLM (`vision_text.gguf`) |
| `--model-proj <path>` | `--vision-proj` | CLIP mmproj (`vision_proj.gguf`) |
| `--model-reasoning <path>` | `--reasoning` | ARLL model (`reasoning_expert.gguf`) |
| `--model-clinical <path>` | `--clinical` | CSR model (`clinical_expert.gguf`) |
| `--image <path>` | | Patient image (**required**) |
| `--temp <f>` | `--temperature` | Sampling temperature (default 0.2) |
| `--n-predict <n>` | `--n_predict` | Max tokens to generate (default 128) |
| `--settings <json>` | | JSON settings file (loaded before CLI flags) |
| `--chat-target reasoning\|clinical` | | Post-diagnosis chat expert (default: reasoning) |

## Interactive doctor chat
After diagnosis completes, terminal enters chat mode:
- Type a question to query chosen expert.
- Type `exit` to quit.
- Sending EOF (e.g. from a subprocess with no stdin) exits cleanly.

## Settings schema
```json
{
  "vision_proj_model": "models/vision_proj.gguf",
  "vision_text_model": "models/vision_text.gguf",
  "reasoning_model": "models/reasoning_expert.gguf",
  "clinical_model": "models/clinical_expert.gguf",
  "max_iterations": 3,
  "confidence_threshold": 0.90
}
```
