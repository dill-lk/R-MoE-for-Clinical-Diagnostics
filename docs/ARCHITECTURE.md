# R-MoE Architecture (Current Implementation)

## Triple-Expert Pipeline
1. **MPE (Perception)**
   - Loads `vision_proj.gguf`, then `vision_text.gguf`.
   - Produces embedding-style visual summary.
2. **ARLL (Reasoning)**
   - Loads `reasoning_expert.gguf`.
   - Produces CoT-style reasoning summary and confidence `Sc`.
3. **CSR (Reporting)**
   - Loads `clinical_expert.gguf`.
   - Produces structured ICD-11 JSON report.

## #wanna# Recursive Gate
- Hard limit: configurable (`max_iterations`, default `3`).
- Gate: if `Sc < confidence_threshold` (default `0.90`), engine triggers `#wanna#`.
- Feedback tensor requests:
  - `High-Res Crop`, or
  - `Alternate View`.
- Exit strategy:
  - if still below threshold at hard limit -> `EscalateToHuman()`.

## Research-Style Runtime Extensions
- **Uncertainty metrics** captured each iteration:
  - confidence,
  - uncertainty (`1 - Sc`),
  - predictive entropy (binary entropy approximation).
- **Iteration trace** captured for auditability:
  - perception summary,
  - reasoning summary,
  - gate decision,
  - uncertainty metrics.
- **RunSummary output** returns:
  - success/escalation flags,
  - number of iterations executed,
  - final report (if successful),
  - complete iteration trace.

## Expert Swapper (RAM Safety)
- Only one expert model stays loaded at a time.
- Before each phase load, prior phase model/context is unloaded.
- Prevents multi-model RAM pressure on constrained edge hardware.

## System Prompts
Prompt templates are stored under `prompts/`:
- `mpe_system_prompt.txt`
- `arll_system_prompt.txt`
- `csr_system_prompt.txt`

These prompts are prepended to inference requests for each phase.
