"""
R-MoE Clinical Engine — Google Colab launcher
==============================================
Copies GGUF model files from Google Drive into Colab's fast /content/models/
directory, then launches the pre-built rmoe_engine binary with the correct
CLI flags.

Usage (run each cell in order):
  1. Mount Drive and run this script:
       %run colab_runner.py
  2. Or call run_engine() directly after mounting Drive manually:
       from colab_runner import run_engine
       run_engine()

Expected Drive layout:
  MyDrive/
  └── Medical_MoE_Models/
      ├── vision_text.gguf
      ├── vision_proj.gguf
      ├── reasoning_expert.gguf
      ├── clinical_expert.gguf
      └── test_patient.png

The engine binary is expected at:
  /content/Mr.ToM/build/rmoe_engine
Build it first with:
  !cmake -S /content/Mr.ToM -B /content/Mr.ToM/build -DCMAKE_BUILD_TYPE=Release
  !cmake --build /content/Mr.ToM/build --config Release -j4
"""

import os
import shutil
import subprocess

# ── Path configuration ────────────────────────────────────────────────────────
DRIVE_DIR       = "/content/drive/MyDrive/Medical_MoE_Models"
LOCAL_MODELS_DIR = "/content/models"
ENGINE_PATH     = "/content/Mr.ToM/build/rmoe_engine"

# Models to stage from Drive → local fast storage
_MODEL_FILES = [
    "vision_text.gguf",
    "vision_proj.gguf",
    "reasoning_expert.gguf",
    "clinical_expert.gguf",
    "test_patient.png",
]


def setup_environment() -> bool:
    """Stage model files from Drive into LOCAL_MODELS_DIR.

    Returns:
        True if all files are ready, False if any Drive file is missing.
    """
    print("📂 Setting up environment...")

    if not os.path.exists("/content"):
        print("⚠️  /content does not exist — are you running inside Google Colab?")
        return False

    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

    for fname in _MODEL_FILES:
        drive_path = os.path.join(DRIVE_DIR, fname)
        local_path = os.path.join(LOCAL_MODELS_DIR, fname)

        if not os.path.exists(drive_path):
            print(f"❌ MISSING ON DRIVE: {drive_path}")
            return False

        if os.path.exists(local_path):
            print(f"⚡ {fname} ready.")
        else:
            print(f"🚚 Copying {fname}...")
            shutil.copy2(drive_path, local_path)

    return True


def run_engine(
    temperature: float = 0.6,
    n_predict: int = 512,
    image: str = "models/test_patient.png",
) -> None:
    """Run the R-MoE diagnostic engine.

    Args:
        temperature: Sampling temperature (default 0.6).
        n_predict:   Maximum tokens to generate per inference step (default 512).
        image:       Path to the patient image, relative to /content (default
                     "models/test_patient.png").
    """
    if not setup_environment():
        print("🛑 Setup failed — check that all model files are on Drive.")
        return

    if not os.path.exists(ENGINE_PATH):
        print(
            f"🛑 Engine binary not found at {ENGINE_PATH}\n"
            "   Build it with:\n"
            "     !cmake -S /content/Mr.ToM -B /content/Mr.ToM/build "
            "-DCMAKE_BUILD_TYPE=Release\n"
            "     !cmake --build /content/Mr.ToM/build --config Release -j4"
        )
        return

    cmd = [
        ENGINE_PATH,
        "--model-vision",    "models/vision_text.gguf",
        "--model-proj",      "models/vision_proj.gguf",
        "--model-reasoning", "models/reasoning_expert.gguf",
        "--model-clinical",  "models/clinical_expert.gguf",
        "--image",           image,
        "--temp",            str(temperature),
        "--n-predict",       str(n_predict),
    ]

    print("\n🚀 Launching R-MoE Engine...")
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
        print(f"\n🔥 Error launching engine: {exc}")

    print("-" * 60)


if __name__ == "__main__":
    if not os.path.exists("/content/drive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except ImportError:
            print("⚠️  Not running inside Colab — skipping Drive mount.")
        except Exception as exc:
            print(f"⚠️  Drive mount failed: {exc}")
            print("   Ensure you are running in Colab and authorise Drive access when prompted.")

    run_engine()
