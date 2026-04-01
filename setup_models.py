#!/usr/bin/env python3
"""Download and export RF-DETR ONNX models for the web app.

Usage:
    python setup_models.py              # export default models (all in manifest)
    python setup_models.py nano small   # export specific models only
"""
import json
import sys
import os
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "web" / "public" / "models"
MANIFEST = MODELS_DIR / "manifest.json"


def get_model_class(model_id):
    """Lazy-import rfdetr and return (ModelClass, resolution) for a model id."""
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


def models_from_manifest():
    """Read model IDs from the web manifest."""
    with open(MANIFEST) as f:
        return [m["id"] for m in json.load(f)]


def main():
    if sys.argv[1:]:
        model_ids = sys.argv[1:]
    else:
        model_ids = models_from_manifest()

    # Check which models already exist
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
