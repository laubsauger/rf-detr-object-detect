"""Benchmark any model in the zoo on its real backend.

    python benchmark.py                  # default 3-way seg: RF-DETR vs YOLO26(torch) vs YOLO26(MLX)
    python benchmark.py yolo26n yolo26n-mlx       # any ids

Three backends are comparable head-to-head:
    *-seg / nano / ...   RF-DETR + YOLO26 via PyTorch on MPS (or CUDA/CPU)
    *-mlx                YOLO26 native on MLX (Apple Silicon Metal, no torch)

Times end-to-end predict() with warmup. For RF-DETR ids it also times the
exported ONNX across available ORT providers.
"""
import sys
import time

import numpy as np

import models

ITERS = 50
WARMUP = 5

ids = sys.argv[1:] or ["seg-nano", "yolo26n-seg", "yolo26n-seg-mlx"]
for mid in ids:
    if mid not in models.REGISTRY:
        print(f"Unknown model: {mid}. Available: {', '.join(models.REGISTRY)}")
        sys.exit(1)

import cv2
from ultralytics.utils import ASSETS

# Real image with objects — random noise finds 0 detections, which skips all
# mask/NMS postprocess and undercounts. Resized square to each model's native res.
_BASE = cv2.imread(str(ASSETS / "bus.jpg"))


def native_input(res):
    return cv2.resize(_BASE, (res, res))


def print_stats(label, times):
    print(f"  {label:<22s} median {np.median(times):6.1f}ms | mean {np.mean(times):6.1f}ms | min {np.min(times):6.1f}ms | max {np.max(times):6.1f}ms")


def bench(label, fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    print_stats(label, times)


print(f"=== Zoo benchmark | torch device: {models.DEVICE} | {WARMUP} warmup + {ITERS} iters ===\n")

for mid in ids:
    spec = models.REGISTRY[mid]
    print(f"--- {mid}  ({spec['family']} {spec['task']} @ {spec['resolution']}px) ---")

    # 1. End-to-end predict on the model's real backend (torch/MPS or MLX/Metal).
    #    Fed at native resolution (res x res) so no resize cost skews the number.
    dummy = native_input(spec["resolution"])
    try:
        model = models.build(mid)
        bench(f"predict [{model.backend}/{model.device}]", lambda: model.predict(dummy, threshold=0.5))
        sp = model.last_speed
        if sp:
            print(f"    split  pre {sp['preprocess']:4.1f} | inf {sp['inference']:5.1f} | post {sp['postprocess']:5.1f} ms (ultralytics profiler)")
        else:
            print("    split  not available for this backend (bundled)")
    except Exception as e:
        print(f"  predict skipped: {e}")

    # 2. ONNX Runtime — only RF-DETR ids have exported web models.
    onnx_path = f"web/public/models/{mid}/inference_model.onnx"
    try:
        import onnxruntime as ort
        import os

        if not os.path.exists(onnx_path):
            print(f"  ONNX skipped: {onnx_path} not found (run setup_models.py)")
        else:
            res = spec["resolution"]
            input_np = np.random.randn(1, 3, res, res).astype(np.float32)
            onnx_input = "images" if spec["family"] == "yolo" else "input"
            available = ort.get_available_providers()
            providers_to_try = []
            if "CUDAExecutionProvider" in available:
                providers_to_try.append(("CUDA", ["CUDAExecutionProvider"]))
            if "CoreMLExecutionProvider" in available:
                providers_to_try.append(("CoreML", ["CoreMLExecutionProvider"]))
            providers_to_try.append(("CPU", ["CPUExecutionProvider"]))

            for plabel, providers in providers_to_try:
                try:
                    sess = ort.InferenceSession(onnx_path, providers=providers)
                    bench(f"onnx {plabel}", lambda s=sess: s.run(None, {onnx_input: input_np}))
                except Exception as e:
                    print(f"  onnx {plabel} failed: {e}")
    except ImportError:
        print("  ONNX skipped: onnxruntime not installed")

    print()
