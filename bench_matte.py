"""Background-removal (person matte) benchmark — the real use case.

Throw a frame in, get back the person pixels only (everything else removed),
no boxes / no contours / no per-instance UI. Measures the leanest pipeline:

    inference -> union of person masks -> apply alpha to frame

    python bench_matte.py                       # default backends
    python bench_matte.py yolo26n-seg yolo26n-seg-mlx seg-nano

Reports the median ms + fps for each stage so you can see where time goes for
the matte use case specifically (masks only, person class only).
"""
import sys
import time

import cv2
import numpy as np

import models

# person class id differs by family: YOLO contiguous 0, RF-DETR COCO cat-id 1.
PERSON = {"yolo": 0, "mlx": 0, "rfdetr": 1}

ITERS = 40
WARMUP = 10


def med(xs):
    return float(np.median(xs))


def person_union_fast(model, rgb, threshold):
    """YOLO torch fast path: masks already at proto res (160). Returns
    (full_res_alpha uint8 HxW, stage_ms dict)."""
    h, w = rgb.shape[:2]
    t0 = time.perf_counter()
    r = model.predict_fast(rgb, threshold)
    t1 = time.perf_counter()
    pid = PERSON[model.backend if model.backend == "mlx" else model.family]
    masks, mh, mw = r["masks"], r["mh"], r["mw"]
    sel = (r["classes"] == pid)
    if masks is None or not sel.any():
        union_small = np.zeros((mh or 160, mw or 160), np.uint8)
    else:
        union_small = (masks[sel].max(axis=0) > 0).astype(np.uint8)
    alpha = cv2.resize(union_small * 255, (w, h), interpolation=cv2.INTER_LINEAR)
    t2 = time.perf_counter()
    cutout = cv2.bitwise_and(rgb, rgb, mask=(alpha > 127).astype(np.uint8))
    t3 = time.perf_counter()
    return cutout, {
        "infer": (t1 - t0) * 1000,
        "union+upscale": (t2 - t1) * 1000,
        "apply": (t3 - t2) * 1000,
        "total": (t3 - t0) * 1000,
    }


def person_union_sv(model, bgr, threshold):
    """MLX / RF-DETR path via sv.Detections (full-res masks)."""
    h, w = bgr.shape[:2]
    t0 = time.perf_counter()
    det = model.predict(bgr, threshold)
    t1 = time.perf_counter()
    pid = PERSON[model.backend if model.backend == "mlx" else model.family]
    if det.mask is None or len(det) == 0:
        alpha = np.zeros((h, w), np.uint8)
    else:
        sel = det.class_id == pid
        alpha = (det.mask[sel].max(axis=0) if sel.any() else np.zeros((h, w), bool)).astype(np.uint8)
    t2 = time.perf_counter()
    cutout = cv2.bitwise_and(bgr, bgr, mask=alpha)
    t3 = time.perf_counter()
    return cutout, {
        "infer": (t1 - t0) * 1000,
        "union": (t2 - t1) * 1000,
        "apply": (t3 - t2) * 1000,
        "total": (t3 - t0) * 1000,
    }


def main():
    ids = sys.argv[1:] or ["yolo26n-seg", "yolo26n-seg-mlx", "seg-nano"]
    # Use a real frame with a person; bus.jpg has several.
    from ultralytics.utils import ASSETS
    bgr = cv2.resize(cv2.imread(str(ASSETS / "bus.jpg")), (640, 640))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print(f"=== person matte benchmark | device {models.DEVICE} | {ITERS} iters ===\n")
    for mid in ids:
        m = models.build(mid)
        fast = hasattr(m, "predict_fast")
        run = (lambda: person_union_fast(m, rgb, 0.5)) if fast else (lambda: person_union_sv(m, bgr, 0.5))
        for _ in range(WARMUP):
            run()
        stages = {}
        for _ in range(ITERS):
            _, s = run()
            for k, v in s.items():
                stages.setdefault(k, []).append(v)
        tot = med(stages["total"])
        parts = "  ".join(f"{k} {med(v):5.1f}" for k, v in stages.items() if k != "total")
        print(f"{mid:16s} [{m.backend}/{m.device}]  total {tot:5.1f}ms ({1000/tot:4.1f} fps)   | {parts}")


if __name__ == "__main__":
    main()
