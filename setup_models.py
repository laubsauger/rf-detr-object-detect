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
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "web" / "public" / "models"
MANIFEST = MODELS_DIR / "manifest.json"
VENV_DIR = ROOT / "venv"
REQUIREMENTS = ROOT / "requirements.txt"

IS_WIN = sys.platform == "win32"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe" if IS_WIN else VENV_DIR / "bin" / "python"


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
    pip = VENV_DIR / "Scripts" / "pip.exe" if IS_WIN else VENV_DIR / "bin" / "pip"
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


# Family + resolution per id. RF-DETR ids map to rfdetr classes; YOLO26 ids
# map to ultralytics weights. Web decode keys off `family` in manifest.json.
SPECS = {
    "nano":         dict(family="rfdetr", rf="RFDETRNano",      resolution=384),
    "small":        dict(family="rfdetr", rf="RFDETRSmall",     resolution=512),
    "medium":       dict(family="rfdetr", rf="RFDETRMedium",    resolution=576),
    "large":        dict(family="rfdetr", rf="RFDETRLarge",     resolution=704),
    "base":         dict(family="rfdetr", rf="RFDETRBase",      resolution=560),
    "seg-nano":     dict(family="rfdetr", rf="RFDETRSegNano",   resolution=312),
    "seg-small":    dict(family="rfdetr", rf="RFDETRSegSmall",  resolution=384),
    "seg-medium":   dict(family="rfdetr", rf="RFDETRSegMedium", resolution=432),
    "seg-large":    dict(family="rfdetr", rf="RFDETRSegLarge",  resolution=504),
    "yolo26n":      dict(family="yolo", weights="yolo26n.pt",     resolution=640),
    "yolo26s":      dict(family="yolo", weights="yolo26s.pt",     resolution=640),
    "yolo26n-seg":  dict(family="yolo", weights="yolo26n-seg.pt", resolution=640),
    "yolo26s-seg":  dict(family="yolo", weights="yolo26s-seg.pt", resolution=640),
}

# RTMW whole-body pose (rtmlib prebuilt onnx, zip contains end2end.onnx). Too
# large for git (123-219MB) -> downloaded here so clones reproduce. fp16 variants
# (inference_model_fp16.onnx) are converted locally (keep_io_types -> fp32 IO).
RTMLIB = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk"
RTMW = {
    "rtmw-m":     f"{RTMLIB}/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
    "rtmw-l":     f"{RTMLIB}/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
    "rtmw-l-384": f"{RTMLIB}/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
}
# Extra detector input resolutions for the pose top-down crop (640 = base export).
# The web pose UI lets you pick these; lower res = faster person boxes.
DET_RES = [320, 384, 512]

# RTMW3D whole-body 3D pose (single .onnx, ~352MB, from HuggingFace). Top-down,
# same crop path as 2D RTMW but 3 SimCC axes (x,y,z).
RTMW3D = {
    "rtmw3d-x": "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/"
                "rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx",
}


def download_url(url, dst):
    print(f"  downloading {os.path.basename(dst)} (~350MB)...")
    urllib.request.urlretrieve(url, dst)


def download_rtmw(mid, output_dir):
    url = RTMW[mid]
    print(f"  downloading RTMW {mid} from rtmlib (~120-220MB)...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        with zipfile.ZipFile(tmp.name) as z, \
                z.open("end2end.onnx") as src, \
                open(os.path.join(output_dir, "inference_model.onnx"), "wb") as dst:
            shutil.copyfileobj(src, dst)
    os.unlink(tmp.name)


def convert_fp16(src, dst):
    import onnx
    from onnxconverter_common import float16
    # keep_io_types=True: inputs/outputs stay fp32 (internal fp16), so the web
    # preprocess + SimCC decode need no dtype changes.
    m16 = float16.convert_float_to_float16(onnx.load(src), keep_io_types=True)
    onnx.save(m16, dst)


def export_yolo_res(res, dst):
    from ultralytics import YOLO
    p = YOLO(str(ROOT / "yolo26n.pt")).export(format="onnx", imgsz=res, opset=17, verbose=False)
    shutil.move(str(p), dst)


def get_spec(model_id):
    if model_id not in SPECS:
        print(f"Unknown model: {model_id}")
        print(f"Available: {', '.join(SPECS)}")
        sys.exit(1)
    return SPECS[model_id]


def export_rfdetr(spec, output_dir):
    import rfdetr

    cls = getattr(rfdetr, spec["rf"])
    res = spec["resolution"]
    print(f"  RF-DETR {spec['rf']} at {res}x{res}...")
    model = cls()
    model.export(
        output_dir=output_dir,
        shape=(res, res),
        batch_size=1,
        opset_version=17,
        verbose=True,
    )


def export_yolo(spec, output_dir):
    import shutil
    from ultralytics import YOLO

    res = spec["resolution"]
    print(f"  YOLO26 {spec['weights']} at {res}x{res}...")
    model = YOLO(spec["weights"])
    onnx_path = model.export(format="onnx", imgsz=res, opset=17, verbose=False)
    # Ultralytics writes <weights>.onnx next to the weights; normalise the name.
    shutil.move(str(onnx_path), os.path.join(output_dir, "inference_model.onnx"))


def ensure_model(mid):
    """Produce every artifact a model needs, skipping ones already on disk.
    Returns True if any work was done."""
    out_dir = MODELS_DIR / mid
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "inference_model.onnx"
    did = False

    # --- RTMW pose: download base + convert fp16 ---
    if mid in RTMW:
        if not base.exists():
            download_rtmw(mid, str(out_dir))
            print(f"  -> {base} ({base.stat().st_size // (1024*1024)} MB)")
            did = True
        fp16 = out_dir / "inference_model_fp16.onnx"
        if not fp16.exists():
            print(f"  converting {mid} -> fp16...")
            convert_fp16(str(base), str(fp16))
            did = True
        return did

    # --- RTMW3D pose: download single onnx (no zip, no fp16) ---
    if mid in RTMW3D:
        if not base.exists():
            download_url(RTMW3D[mid], str(base))
            print(f"  -> {base} ({base.stat().st_size // (1024*1024)} MB)")
            did = True
        return did

    # --- RF-DETR / YOLO base export ---
    spec = get_spec(mid)
    if not base.exists():
        print(f"Exporting {mid} ({spec['family']})...")
        if spec["family"] == "rfdetr":
            export_rfdetr(spec, str(out_dir))
        else:
            export_yolo(spec, str(out_dir))
        print(f"  -> {base}")
        did = True

    # --- yolo26n is the pose detector: also emit lower-res variants ---
    if mid == "yolo26n":
        for res in DET_RES:
            dst = out_dir / f"inference_model_{res}.onnx"
            if not dst.exists():
                print(f"  detector variant yolo26n @ {res}...")
                export_yolo_res(res, str(dst))
                did = True

    return did


def main():
    ensure_venv()
    run_in_venv()

    model_ids = sys.argv[1:] if sys.argv[1:] else models_from_manifest()
    any_work = False
    for mid in model_ids:
        if ensure_model(mid):
            any_work = True
        else:
            print(f"  {mid}: all artifacts present, skipping")

    print("Done." if any_work else "All models already present.")


if __name__ == "__main__":
    main()
