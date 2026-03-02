# R-MoE Collaboration Guide

> **Recursive Multi-Agent Mixture-of-Experts (R-MoE) for Autonomous Clinical Diagnostics**
> This guide explains what the project is, how it is structured, and how to contribute.

---

## What Is This Project?

R-MoE (Recursive Mixture-of-Experts) is an open-source AI system that performs
autonomous clinical image diagnostics by chaining three specialist language models
in sequence.  It is designed to reduce *diagnostic hallucinations* that occur when a
single large vision-language model is asked to handle perception, reasoning, and
clinical reporting all at once.

The pipeline runs entirely on a consumer GPU (T4 / Colab free tier) using
quantised GGUF models loaded one at a time to stay within VRAM limits.

---

## Three-Phase Pipeline at a Glance

| Phase | Agent | Model file | Role |
|-------|-------|-----------|------|
| 1 – MPE  | `VisionExpert`     | `vision_proj.gguf` + `vision_text.gguf` | Perceive the image; crop salient regions; produce visual evidence |
| 2 – ARLL | `ReasoningExpert`  | `reasoning_expert.gguf`                 | Build differential diagnoses (DDx); compute confidence score Sc = 1−σ² |
| 3 – CSR  | `ReportingExpert`  | `clinical_expert.gguf`                  | Generate structured ICD-11/SNOMED report; apply dual-layer safety check |

If `Sc < 0.90` after Phase 2 the `#wanna#` protocol triggers up to three
additional passes (zoom crop → alternate view → modality escalation) before
handing off to a human radiologist.

---

## Repository Layout

```
R-MoE-for-Clinical-Diagnostics/
├── engine.py              CLI entry-point
├── colab_runner.py        Google Colab launcher (mount Drive, stage models, run)
├── requirements.txt       Python dependencies
├── settings/
│   ├── rmoe_settings.json              T4 / Colab config
│   └── rmoe_settings_research.json     8×A100 research config
├── prompts/               System-prompt templates (MPE / ARLL / CSR)
├── data/
│   └── benchmark_cases.csv            20 annotated evaluation cases
├── paper/
│   └── rmoe_paper.tex                 Full LaTeX manuscript
├── docs/
│   ├── ARCHITECTURE.md                Technical architecture notes
│   └── COLLAB_GUIDE.md                ← you are here
├── rmoe/                  Python package (see below)
└── tests/                 119 pytest unit tests (no GPU required)
```

### `rmoe/` package modules

| File | Purpose |
|------|---------|
| `models.py`      | Dataclasses: `DDxEnsemble`, `RunSummary`, `WannaState`, … |
| `core.py`        | `DiagnosticEngine`, `WannaStateMachine`, `MrTom` public API |
| `agents.py`      | `VisionExpert`, `ReasoningExpert`, `ReportingExpert`, `ExpertSwapper` |
| `hitl.py`        | `HITLCoordinator` — human-in-the-loop prompts and query routing |
| `rag.py`         | `VectorRAGEngine` — BM25 retrieval over clinical guidelines |
| `ontology.py`    | ICD-11 / SNOMED CT tables; TIRADS / BI-RADS / Lung-RADS risk scales |
| `calibration.py` | `CalibrationTracker`, ECE, Brier score |
| `audit.py`       | `AuditLogger`, `SessionReportGenerator` (HIPAA audit trail) |
| `bias.py`        | `CognitiveBiasDetector` (anchoring, conflicting evidence, …) |
| `mcv.py`         | `MCVBuilder` / `MCVInjector` — Multi-Modal Contextual Vectors |
| `safety.py`      | `CSRSafetyValidator` — semantic parser + clinical rule checker |
| `modality.py`    | `ModalityEscalationRouter` (CXR → CT → MRI → PET-CT) |
| `temporal.py`    | `TemporalComparator` — Fleischner threshold, Sc adjustment |
| `saliency.py`    | `SaliencyProcessor` — crop + zoom sub-patches |
| `dicom.py`       | `DICOMProcessor` — lung / bone / brain / soft-tissue windowing |
| `ensemble.py`    | `MultiTemperatureEnsemble` — cross-temperature σ² for Sc |
| `eval.py`        | `BenchmarkRunner` — F1, AUC, ECE, Brier, Type-I/II errors |
| `charts.py`      | ASCII charts: Sc progression, DDx evolution, reliability diagram |
| `ui.py`          | ANSI terminal UI helpers |
| `mock.py`        | Realistic mock experts for CI / offline development |

---

## Quick Start (no GPU, no model files)

```bash
git clone https://github.com/dill-lk/Mr.ToM
cd Mr.ToM
pip install -r requirements.txt

# Run in mock mode — no model files needed
python engine.py --image test_patient.png
```

The engine automatically falls back to `rmoe/mock.py` when no `.gguf` model
files are present, so the full pipeline (including charts) runs offline.

---

## Running the Test Suite

All 119 tests run without model files or a GPU:

```bash
python -m pytest tests/ -v
```

---

## How to Contribute

1. **Fork** the repository and create a feature branch.
2. **Install dev dependencies**: `pip install -r requirements.txt pytest`.
3. **Write tests** in `tests/` that cover your change.  New tests must pass
   without model files (use `rmoe/mock.py` helpers where needed).
4. **Run the full test suite** (`python -m pytest tests/ -v`) and confirm all
   119 tests still pass.
5. **Open a pull request** against the `main` branch with a clear description
   of *what* changed and *why*.

### Adding a new module

* Place the file under `rmoe/` and follow the docstring style used in existing
  modules (see `rmoe/bias.py` for a minimal example).
* Export public symbols from `rmoe/__init__.py` and add them to `__all__`.
* Add at least one pytest test in `tests/`.

### Changing inference behaviour

* Core pipeline logic lives in `rmoe/core.py` (`DiagnosticEngine`) and
  `rmoe/agents.py`.
* The confidence formula `Sc = 1 − σ²` is computed in `rmoe/models.py`
  (`DDxEnsemble.compute_sc`).
* Any change that touches `Sc` or the `#wanna#` gate should include a new test
  in `tests/test_core.py`.

---

## Configuring the Pipeline

`settings/rmoe_settings.json` is the active config file:

```json
{
  "vision_proj_model":  "models/vision_proj.gguf",
  "vision_text_model":  "models/vision_text.gguf",
  "reasoning_model":    "models/reasoning_expert.gguf",
  "clinical_model":     "models/clinical_expert.gguf",
  "modality":           "CXR",
  "confidence_threshold": 0.90,
  "max_iterations": 3,
  "inference": {
    "n_ctx": 2048,
    "n_gpu_layers": -1,
    "temperature": 0.2,
    "max_new_tokens": 512
  }
}
```

CLI flags override any JSON value — see `python engine.py --help` for the full
list.

---

## Colab Usage

See [RUN.md](../RUN.md) for the complete cell-by-cell Colab guide, including
Drive setup, model download links, and troubleshooting tips.

---

## Citation

```bibtex
@article{rmoe2025,
  title  = {Recursive Multi-Agent Mixture-of-Experts (RMoE)
             for Autonomous Clinical Diagnostics},
  year   = {2026},
  note   = {Open-source implementation at github.com/dill-lk/Mr.ToM}
}
```
