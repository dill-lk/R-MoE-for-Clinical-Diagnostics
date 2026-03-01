# Mr.ToM
The Theory Of Mind

## R-MoE Clinical Engine (Research-Oriented Core)
This repository contains a runnable C++17 framework for a **Recursive Multi-Agent Mixture-of-Experts (R-MoE)** clinical pipeline with uncertainty-aware gating, model swap orchestration, prompt-driven phases, and configurable runtime settings.

## Architecture Flow (ASCII)
```text
================================================================================
             Mr.ToM (R-MoE) : CLINICAL DIAGNOSTIC PIPELINE
       [ Recursive Multi-Agent Mixture-of-Experts Architecture ]
================================================================================

       [ INPUT: PATIENT RADIOGRAPH + CLINICAL NOTES ]
                        |
                        v
+-----------------------+-----------------------+
| PHASE 1: MPE (Perception)                      |
|-----------------------------------------------|
| MODELS: vision_proj.gguf + vision_text.gguf   |
| TASK: Visual embedding extraction + alignment  |
+-----------------------+-----------------------+
                        |
            [ SWAP: UNLOAD PHASE 1 -> LOAD PHASE 2 ]
                        |
                        v
+-----------------------+-----------------------+
| PHASE 2: ARLL (Reasoning)                      |
|-----------------------------------------------|
| MODEL: reasoning_expert.gguf                   |
| TASK: CoT diagnostic logic + Sc estimation     |
+-----------------------+-----------------------+
                        |
           [ CONFIDENCE GATE (Sc Threshold: 0.90) ]
                        |
         IF Sc < 0.90   |           IF Sc >= 0.90
       +----------------+----------------+
       |                                 |
 [ #wanna# PROTOCOL ]                    v
       |                  +-----------------------+-----------------------+
 [ LOOP BACK TO P1 ]      | PHASE 3: CSR (Reporting)                      |
       |                  |-----------------------------------------------|
 (Max 3 Iterations)       | MODEL: clinical_expert.gguf                   |
       |                  | TASK: Final ICD-11 clinical synthesis         |
       +----------------> +-----------------------+-----------------------+
                                                  |
                                                  v
                                     [ FINAL OUTPUT / HITL ESCALATION ]
================================================================================
```

## llama_decode status
- **Yes, there is a concrete `llama_decode` generation scaffold** in `ExpertSwapper::infer_text()`.
- Current implementation is token-level text generation scaffold (tokenize -> decode prompt -> greedy next-token loop).
- Full multimodal handoff (real visual token tensor transport across experts) is still a v2 task; current handoff is represented via structured text/feedback tensors.

## What’s improved
- Configurable runtime model routing via CLI and JSON settings.
- Runtime image path input (`--image`) instead of hardcoding in code.
- Interactive post-diagnosis chat with Reasoning or Clinical expert.
- Uncertainty metrics per iteration (`uncertainty`, `predictive_entropy`) for traceability.
- Iteration audit trail (decision + confidence + entropy) returned in `RunSummary`.
- Recursive abstention control with configurable gate (`max_iterations`, `confidence_threshold`).

## Build & Run
```bash
cmake -S . -B build
cmake --build build -j4

# Full runtime argument mode
./build/rmoe_engine \
  --vision-proj models/vision_proj.gguf \
  --vision-text models/vision_text.gguf \
  --reasoning models/reasoning_expert.gguf \
  --clinical models/clinical_expert.gguf \
  --image patient_intake/chest_xray_ap_view.png

# With settings override
./build/rmoe_engine \
  --settings settings/rmoe_settings.json \
  --image patient_intake/chest_xray_ap_view.png
```

## CLI flags
- `--vision-proj <path>`
- `--vision-text <path>`
- `--reasoning <path>`
- `--clinical <path>`
- `--image <path>` (**required**)
- `--settings <json>`
- `--chat-target reasoning|clinical` (default: `reasoning`)

## Interactive doctor chat
After diagnosis completes, terminal enters chat mode:
- Type a question to query chosen expert.
- Type `exit` to quit.

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
