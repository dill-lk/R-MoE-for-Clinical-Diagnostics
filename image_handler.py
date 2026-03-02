"""
image_handler.py — Patient image upload helper for R-MoE v2.0.

Provides a single public function ``upload_patient_image()`` that obtains a
patient scan before the diagnostic pipeline starts.  Three modes are supported:

1. **Google Colab** — opens the native Colab file-upload widget so the doctor
   can click and select any PNG / JPEG / DICOM file from their computer.  The
   uploaded file is saved to ``/content/models/`` and its path is returned.

2. **Interactive CLI** — when running locally in a TTY the function prompts the
   doctor to enter a file path.

3. **Fallback** — if neither a Colab upload nor a CLI path is provided (e.g. in
   automated / scripting mode), the existing ``test_patient.png`` inside
   ``dest_dir`` is used if present, otherwise ``None`` is returned so the caller
   can handle the missing-image case.

Typical Colab usage::

    from image_handler import upload_patient_image
    image_path = upload_patient_image()

    from colab_runner import run_python_engine
    run_python_engine(image=image_path, ...)

Typical CLI usage::

    python engine.py --image "$(python -c \
        'from image_handler import upload_patient_image; print(upload_patient_image())')"

Or simply pass ``--image <path>`` directly to ``engine.py``.
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Optional


# Default directory that mirrors where colab_runner.py stages model files.
_DEFAULT_DEST_DIR = "/content/models"
_FALLBACK_FILENAME = "test_patient.png"

# Supported image extensions (DICOM handled by rmoe/dicom.py at inference time)
_SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".dcm"}


def _is_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # type: ignore[import-untyped]  # noqa: F401
        return True
    except ImportError:
        return False


def _colab_upload(dest_dir: str) -> Optional[str]:
    """
    Open the Colab file-upload widget, save the first uploaded file to
    ``dest_dir``, and return its path.  Returns ``None`` if the upload is
    cancelled or fails.
    """
    try:
        from google.colab import files as colab_files  # type: ignore[import-untyped]
    except ImportError:
        return None

    print("\n📂  Please upload your patient image (PNG / JPEG / DICOM):")
    try:
        uploaded = colab_files.upload()
    except Exception as exc:
        print(f"  ⚠️  Upload widget error: {exc}")
        return None

    if not uploaded:
        print("  ⚠️  No file was uploaded.")
        return None

    # Take the first uploaded file (doctors typically upload one scan at a time)
    filename = next(iter(uploaded))
    ext = os.path.splitext(filename)[1].lower()

    if ext not in _SUPPORTED_EXT:
        print(
            f"  ⚠️  '{filename}' has an unsupported extension '{ext}'.\n"
            f"      Supported formats: {', '.join(sorted(_SUPPORTED_EXT))}"
        )
        return None

    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    # The Colab upload widget writes the file to the current working directory;
    # move it to dest_dir when the directories differ.
    cwd_path = os.path.join(os.getcwd(), filename)
    if os.path.abspath(cwd_path) != os.path.abspath(dest_path) and os.path.exists(cwd_path):
        shutil.move(cwd_path, dest_path)

    if os.path.exists(dest_path):
        size_kb = os.path.getsize(dest_path) // 1024
        print(f"  ✅  Uploaded '{filename}' ({size_kb} KB) → {dest_path}")
        return dest_path

    print(f"  ❌  Could not locate uploaded file at '{dest_path}'.")
    return None


def _cli_prompt(dest_dir: str) -> Optional[str]:
    """
    Ask the operator to type an image path in a TTY session.
    Returns the path if the file exists, otherwise ``None``.
    """
    if not sys.stdin.isatty():
        return None

    print("\n📂  Enter the path to the patient image (or press Enter to use default):")
    print("    Supported formats: PNG, JPEG, BMP, WebP, DICOM (.dcm)")
    print("    > ", end="", flush=True)
    try:
        raw = input().strip()
    except EOFError:
        return None

    if not raw:
        return None

    path = os.path.expanduser(raw)
    if not os.path.isfile(path):
        print(f"  ❌  File not found: {path}")
        return None

    ext = os.path.splitext(path)[1].lower()
    if ext not in _SUPPORTED_EXT:
        print(
            f"  ⚠️  '{path}' has an unsupported extension '{ext}'.\n"
            f"      Supported formats: {', '.join(sorted(_SUPPORTED_EXT))}"
        )
        return None

    # Optionally copy to dest_dir so everything is co-located with the models
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(path))
    if os.path.abspath(path) != os.path.abspath(dest_path):
        shutil.copy2(path, dest_path)
        print(f"  ✅  Copied to {dest_path}")
        return dest_path

    return path


def upload_patient_image(
    dest_dir: Optional[str] = None,
    fallback_path: Optional[str] = None,
) -> Optional[str]:
    """
    Obtain a patient image path before the R-MoE pipeline starts.

    Priority:
      1. Colab upload widget (if running inside Google Colab)
      2. Interactive CLI prompt (if stdin is a TTY)
      3. ``fallback_path`` if supplied
      4. ``<dest_dir>/test_patient.png`` if it exists

    Args:
        dest_dir:      Directory where the uploaded image is stored.
                       Defaults to ``/content/models`` (Colab) or the
                       current working directory (local).
        fallback_path: Explicit path to use when interactive upload is
                       unavailable and no default is found.

    Returns:
        Absolute path to the image, or ``None`` if no image could be obtained.
    """
    # Resolve dest_dir
    if dest_dir is None:
        dest_dir = _DEFAULT_DEST_DIR if _is_colab() else os.getcwd()

    # 1. Colab widget upload
    if _is_colab():
        path = _colab_upload(dest_dir)
        if path:
            return path
        # Fall through to fallback if upload was cancelled

    # 2. CLI prompt
    path = _cli_prompt(dest_dir)
    if path:
        return path

    # 3. Explicit fallback supplied by caller
    if fallback_path and os.path.isfile(fallback_path):
        print(f"  ℹ️   Using supplied fallback image: {fallback_path}")
        return fallback_path

    # 4. Default test_patient.png in dest_dir
    default = os.path.join(dest_dir, _FALLBACK_FILENAME)
    if os.path.isfile(default):
        print(f"  ℹ️   No image uploaded — using default: {default}")
        return default

    print(
        "  ⚠️   No patient image found.\n"
        f"      Upload via Colab widget, supply --image <path> on the CLI,\n"
        f"      or place a '{_FALLBACK_FILENAME}' in '{dest_dir}'."
    )
    return None
