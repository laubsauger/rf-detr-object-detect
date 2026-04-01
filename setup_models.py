#!/usr/bin/env python3
"""Download and export RF-DETR ONNX models for the web app.

Automatically creates a Python venv and installs dependencies if needed.

Usage:
    python setup_models.py              # export all models in manifest
    python setup_models.py nano small   # export specific models only
"""
import json
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "web" / "public" / "models"
MANIFEST = MODELS_DIR / "manifest.json"
VENV_DIR = ROOT / "venv"
REQUIREMENTS = ROOT / "requirements.txt"

IS_WIN = sys.platform == "win32"
VENV_PYTHON = VENV_DIR / ("Scripts" / "python.exe" if IS_WIN else "bin" / "python")


def ensure_venv():
    """Create venv and install deps if not already set up."""
    if VENV_PYTHON.exists():
        # Check if rfdetr is installed
        result = subprocess.run(
            [str(VENV_PYTHON), "-c", "import rfdetr"],
            capture_output=True,
        )
        if result.returncode == 0:
            return  # venv exists and has rfdetr

    if not VENV_PYTHON.exists():
        print("Creating Python venv...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])

    print("Installing Python dependencies (torch, rfdetr, etc.)...")
    print("This may take a few minutes on first run.\n")
    pip = VENV_DIR / ("Scripts" / "pip.exe" if IS_WIN else "bin" / "pip")
    subprocess.check_call([str(pip), "install", "-r", str(REQUIREMENTS)])


def run_in_venv():
    """Re-exec this script inside the venv if we're not already in it."""
    # If rfdetr imports fine, we're good
    try:
        import rfdetr  # noqa: F401
        return False  # no re-exec needed
    except ImportError:
        pass

    # Re-run this script with the venv python
    args = [str(VENV_PYTHON), __file__] + sys.argv[1:]
    result = subprocess.run(args)
    sys.exit(result.returncode)


def models_from_manifest():
    with open(MANIFEST) as f:
        return [m["id"] for m in json.load(f)]


def get_model_class(model_id):
    from rfdetr import (
        RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase,
    )
    from rfdetr import (
        RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge,
    )

    REGISTRY = {
        "nano":       (RFDETRNano,      384),
        "small":      (RFDETRSmall,     512),
        "medium":     (RFDETRMedium,    576),
        "large":      (RFDETRLarge,     704),
        "base":       (RFDETRBase,      560),
        "seg-nano":   (RFDETRSegNano,   312),
        "seg-small":  (RFDETRSegSmall,  384),
        "seg-medium": (RFDETRSegMedium, 432),
        "seg-large":  (RFDETRSegLarge,  504),
    }

    if model_id not in REGISTRY:
        print(f"Unknown model: {model_id}")
        print(f"Available: {', '.join(REGISTRY.keys())}")
        sys.exit(1)

    return REGISTRY[model_id]


def main():
    ensure_venv()
    run_in_venv()

    # If we get here, rfdetr is available
    model_ids = sys.argv[1:] if sys.argv[1:] else models_from_manifest()

    missing = []
    for mid in model_ids:
        onnx_path = MODELS_DIR / mid / "inference_model.onnx"
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"  {mid}: already exists ({size_mb:.0f} MB), skipping")
        else:
            missing.append(mid)

    if not missing:
        print("All models already exported.")
        return

    print(f"\nExporting {len(missing)} model(s): {', '.join(missing)}")
    print("This will download PyTorch weights and convert to ONNX.\n")

    for mid in missing:
        cls, resolution = get_model_class(mid)
        output_dir = str(MODELS_DIR / mid)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Exporting {mid} ({cls.__name__}) at {resolution}x{resolution}...")
        model = cls()
        model.export(
            output_dir=output_dir,
            shape=(resolution, resolution),
            batch_size=1,
            opset_version=17,
            verbose=True,
        )
        print(f"  -> {output_dir}/inference_model.onnx\n")

    print("Done. All models exported.")


if __name__ == "__main__":
    main()
