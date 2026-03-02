"""
R-MoE Clinical Engine — Google Colab Launcher (v2.0)
=====================================================
Primary entry-point for running the **Python** R-MoE engine in Google Colab.
Model files are staged from Google Drive into Colab's fast local storage, then
the engine is invoked directly via the `engine.py` Python API (no C++ binary
required).

Quick-start (run each cell in order)
─────────────────────────────────────
Cell 1 — Install dependencies (GPU build):
    !CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python
    # CPU-only (no CUDA):
    # !pip install llama-cpp-python

Cell 2 — Mount Drive and launch:
    %run colab_runner.py

Or import and call directly:
    from colab_runner import run_python_engine
    run_python_engine()

Expected Drive layout
─────────────────────
  MyDrive/
  └── Medical_MoE_Models/
      ├── vision_text.gguf       ← Qwen2-VL backbone
      ├── vision_proj.gguf       ← CLIP mmproj
      ├── reasoning_expert.gguf  ← DeepSeek-R1
      ├── clinical_expert.gguf   ← Llama-3-Medius
      └── test_patient.png       ← sample patient image

Legacy C++ engine
─────────────────
The pre-built C++ binary lives in legacy/.  To build it:
    !cmake -S /content/Mr.ToM/legacy -B /content/Mr.ToM/legacy/build \\
           -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
    !cmake --build /content/Mr.ToM/legacy/build --config Release -j4

Then run the legacy engine with run_legacy_engine().
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

# ── Path configuration ────────────────────────────────────────────────────────
DRIVE_DIR        = "/content/drive/MyDrive/Medical_MoE_Models"
LOCAL_MODELS_DIR = "/content/models"
REPO_DIR         = "/content/Mr.ToM"
LEGACY_ENGINE    = os.path.join(REPO_DIR, "legacy", "build", "rmoe_engine")

_MODEL_FILES = [
    "vision_text.gguf",
    "vision_proj.gguf",
    "reasoning_expert.gguf",
    "clinical_expert.gguf",
    "test_patient.png",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment setup
# ═══════════════════════════════════════════════════════════════════════════════

def setup_environment() -> bool:
    """
    Stage model files from Google Drive into LOCAL_MODELS_DIR.

    Returns:
        True if all required files are ready, False if any Drive file is missing.
    """
    print("📂 Setting up environment...")

    if not os.path.exists("/content"):
        print("⚠️  /content not found — are you running inside Google Colab?")
        return False

    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

    all_ready = True
    for fname in _MODEL_FILES:
        drive_path = os.path.join(DRIVE_DIR, fname)
        local_path = os.path.join(LOCAL_MODELS_DIR, fname)

        if not os.path.exists(drive_path):
            print(f"❌ MISSING ON DRIVE: {drive_path}")
            all_ready = False
            continue

        if os.path.exists(local_path):
            print(f"⚡ {fname} already staged.")
        else:
            print(f"🚚 Copying {fname} …")
            shutil.copy2(drive_path, local_path)
            print(f"   ✅ {fname} ready.")

    return all_ready


# ═══════════════════════════════════════════════════════════════════════════════
#  Primary path: Python engine
# ═══════════════════════════════════════════════════════════════════════════════

def run_python_engine(
    image: str = "models/test_patient.png",
    temperature: float = 0.2,
    n_predict: int = 512,
    n_gpu_layers: int = -1,
    prior_image: str | None = None,
    audit_log: str | None = "audit_trail.json",
    chat_target: str = "reasoning",
    eval_mode: bool = False,
) -> None:
    """
    Run the R-MoE v2.0 Python engine directly in this process.

    Args:
        image:        Patient image path (relative to /content or absolute).
        temperature:  Sampling temperature (default 0.2 — clinical precision).
        n_predict:    Max tokens per inference step (default 512).
        n_gpu_layers: GPU layers to offload (-1 = all, 0 = CPU only).
        prior_image:  Optional prior scan for temporal comparison.
        audit_log:    Path to write JSON audit trail (None = skip).
        chat_target:  Post-diagnosis Q&A expert: "reasoning" or "clinical".
        eval_mode:    Print ECE / calibration block after run.
    """
    if not setup_environment():
        print("🛑 Setup failed — check that all model files are present on Drive.")
        return

    # Add repo to Python path so engine.py is importable
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    try:
        from engine import MrTom, WannaStateMachine, ExpertTarget  # type: ignore
        from engine import (  # type: ignore
            print_banner,
            _print_input_info,
            _print_run_summary,
            _print_clinical_report,
            _print_eval_summary,
            _print_rule,
            _GREEN, _BOLD, _CYAN, _RESET, _DIM,
        )
    except ImportError as exc:
        print(f"❌ Cannot import engine.py: {exc}")
        print(f"   Make sure {REPO_DIR}/engine.py exists.")
        return

    _HARD_LIMIT = 3
    _THRESHOLD  = 0.90

    print_banner()

    mr_tom = MrTom(WannaStateMachine(_HARD_LIMIT, _THRESHOLD))

    # Load settings if available
    settings_path = os.path.join(REPO_DIR, "settings", "rmoe_settings.json")
    if os.path.exists(settings_path):
        mr_tom.load_settings(settings_path)

    # Apply run-time overrides
    mr_tom.set_vision_model(
        os.path.join(LOCAL_MODELS_DIR, "vision_proj.gguf"),
        os.path.join(LOCAL_MODELS_DIR, "vision_text.gguf"),
    )
    mr_tom.set_reasoning_model(os.path.join(LOCAL_MODELS_DIR, "reasoning_expert.gguf"))
    mr_tom.set_clinical_model(os.path.join(LOCAL_MODELS_DIR, "clinical_expert.gguf"))
    mr_tom.set_temperature(temperature)
    mr_tom.set_max_tokens(n_predict)
    mr_tom.set_gpu_layers(n_gpu_layers)

    # Resolve image path
    image_path = image if os.path.isabs(image) else os.path.join("/content", image)
    _print_input_info(image_path, _THRESHOLD, _HARD_LIMIT, prior_image)

    summary = mr_tom.process_patient_case(
        image_path,
        audit_log_path=audit_log,
        prior_image=prior_image,
    )

    _print_run_summary(summary, _HARD_LIMIT)

    if summary.final_report_json:
        _print_clinical_report(summary.final_report_json)
    else:
        print()
        _print_rule("=")

    if eval_mode:
        _print_eval_summary(summary)

    # ── Interactive doctor session ────────────────────────────────────────────
    target = (ExpertTarget.Clinical if chat_target == "clinical"
              else ExpertTarget.Reasoning)
    expert_label = ("CSR (clinical report)" if target == ExpertTarget.Clinical
                    else "ARLL (diagnostic reasoning)")
    print(
        f"\n{_DIM}"
        f"  Follow-up questions available  |  Expert: {expert_label}"
        f"  |  Type 'exit' to quit\n"
        f"{_RESET}"
    )
    _print_rule()

    try:
        while True:
            print(f"\n{_GREEN}{_BOLD}  [DOCTOR]  {_RESET}", end="", flush=True)
            try:
                query = input()
            except EOFError:
                break
            query = query.strip()
            if query.lower() == "exit":
                break
            if not query:
                continue
            response = mr_tom.ask_expert(query, target)
            print(f"{_CYAN}  [Mr.ToM]  {_RESET}{response}")
    except KeyboardInterrupt:
        pass

    print()
    _print_rule()
    print(f"{_DIM}  Session closed.\n{_RESET}", end="")
    _print_rule()
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Legacy path: pre-built C++ binary
# ═══════════════════════════════════════════════════════════════════════════════

def run_legacy_engine(
    image: str = "models/test_patient.png",
    temperature: float = 0.6,
    n_predict: int = 512,
    n_gpu_layers: int = 99,
) -> None:
    """
    Run the legacy pre-built C++ rmoe_engine binary.

    Build it first:
        !cmake -S /content/Mr.ToM/legacy -B /content/Mr.ToM/legacy/build \\
               -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
        !cmake --build /content/Mr.ToM/legacy/build --config Release -j4

    Args:
        image:        Patient image path (relative to /content).
        temperature:  Sampling temperature.
        n_predict:    Max tokens per inference step.
        n_gpu_layers: GPU layers (99 = full GPU offload).
    """
    if not setup_environment():
        print("🛑 Setup failed.")
        return

    if not os.path.exists(LEGACY_ENGINE):
        print(
            f"🛑 Legacy engine binary not found at {LEGACY_ENGINE}\n"
            "   Build it with:\n"
            "     !cmake -S /content/Mr.ToM/legacy -B /content/Mr.ToM/legacy/build"
            " -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON\n"
            "     !cmake --build /content/Mr.ToM/legacy/build --config Release -j4"
        )
        return

    cmd = [
        LEGACY_ENGINE,
        "--model-vision",    "models/vision_text.gguf",
        "--model-proj",      "models/vision_proj.gguf",
        "--model-reasoning", "models/reasoning_expert.gguf",
        "--model-clinical",  "models/clinical_expert.gguf",
        "--image",           image,
        "--temp",            str(temperature),
        "--n-predict",       str(n_predict),
        "--n-gpu-layers",    str(n_gpu_layers),
    ]

    print("\n🚀 Launching legacy C++ R-MoE Engine…")
    print("-" * 60)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/content",
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
    except Exception as exc:
        print(f"\n🔥 Error launching legacy engine: {exc}")

    print("-" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry-point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Mount Google Drive if not already mounted
    if not os.path.exists("/content/drive"):
        try:
            from google.colab import drive  # type: ignore[import-untyped]
            drive.mount("/content/drive")
        except ImportError:
            print("⚠️  Not running inside Colab — skipping Drive mount.")
        except Exception as exc:
            print(f"⚠️  Drive mount failed: {exc}")
            print("   Ensure you are in Colab and authorise Drive access when prompted.")

    # Use Python engine by default
    run_python_engine()
