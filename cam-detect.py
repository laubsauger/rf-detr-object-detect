"""Webcam object detection / segmentation across the model zoo.

Switch models live with [ and ] (or n / p). Press q to quit.

    python cam-detect.py                      # default compare set
    python cam-detect.py --model yolo26n-seg  # single model
    python cam-detect.py --model seg-nano,yolo26n-seg,seg-small  # custom cycle
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import supervision as sv

import models


_color_cache = {}


def color_for(class_id):
    cid = int(class_id)
    if cid not in _color_cache:
        hsv = np.uint8([[[(cid * 37) % 180, 200, 255]]])
        b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        _color_cache[cid] = (int(b), int(g), int(r))
    return _color_cache[cid]


def draw_masks_fast(frame, detections, alpha=0.45):
    """Blend each mask only inside its own bounding box — cost scales with the
    sum of box areas, not the whole frame. Avoids the full-frame addWeighted /
    per-mask full-res assignment that made the naive version ~18ms."""
    if detections.mask is None or len(detections) == 0:
        return
    H, W = frame.shape[:2]
    for m, cid, box in zip(detections.mask, detections.class_id, detections.xyxy):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        sub = frame[y1:y2, x1:x2]
        sm = m[y1:y2, x1:x2]
        color = np.array(color_for(cid), np.float32)
        sub[sm] = (sub[sm] * (1 - alpha) + color * alpha).astype(np.uint8)


def fit(frame, long_side):
    """Downscale so the longest side <= long_side (keeps aspect). No upscaling."""
    h, w = frame.shape[:2]
    if max(h, w) <= long_side:
        return frame
    s = long_side / max(h, w)
    return cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)

# Spans all three backends so the default run compares them head-to-head.
DEFAULT_CYCLE = ["seg-nano", "yolo26n-seg", "yolo26n-seg-mlx", "yolo26n-mlx"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default=",".join(DEFAULT_CYCLE),
    help="comma-separated model ids to cycle through (see models.REGISTRY)",
)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--camera", type=int, default=0)
parser.add_argument(
    "--size", type=int, default=960,
    help="cap processing/display long side (px). Lower = cheaper mask draw.",
)
args = parser.parse_args()

cycle = [m.strip() for m in args.model.split(",") if m.strip()]
for mid in cycle:
    if mid not in models.REGISTRY:
        raise SystemExit(
            f"Unknown model '{mid}'. Available: {', '.join(models.REGISTRY)}"
        )

print(f"Device: {models.DEVICE}")
print(f"Models: {', '.join(cycle)}  (switch with [ ] or n p, quit with q)")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

_cache = {}


def load(mid):
    if mid not in _cache:
        print(f"Loading {mid} on {models.DEVICE}...")
        _cache[mid] = models.build(mid)
    return _cache[mid]


idx = 0
model = load(cycle[idx])

cap = cv2.VideoCapture(args.camera)

# Per-model rolling window of (predict_ms, draw_ms, total_ms) — kept across
# switches so each model accumulates its own stats for comparison.
stats = {}


def window(mid):
    if mid not in stats:
        stats[mid] = deque(maxlen=60)
    return stats[mid]


def median(xs, i):
    vals = sorted(x[i] for x in xs)
    return vals[len(vals) // 2] if vals else 0.0


while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)      # mirror, like the CSS-flipped web video
    frame = fit(frame, args.size)   # cap processing/display resolution

    t0 = time.perf_counter()
    detections = model.predict(frame, threshold=args.threshold)
    t_pred = time.perf_counter()

    labels = [
        model.label(class_id, conf)
        for class_id, conf in zip(detections.class_id, detections.confidence)
    ]

    annotated = frame.copy()
    if model.task == "seg":
        draw_masks_fast(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels)
    t_draw = time.perf_counter()

    # Stage split. Same names as the web app (pre / inf / post / draw). For
    # torch/YOLO these come from ultralytics' own profiler; "conv" is the
    # remaining wall time in predict() (sv mask materialization etc.).
    predict_ms = (t_pred - t0) * 1000
    draw_ms = (t_draw - t_pred) * 1000
    total_ms = (t_draw - t0) * 1000
    sp = model.last_speed
    if sp:
        pre, inf, post = sp["preprocess"], sp["inference"], sp["postprocess"]
        conv = max(0.0, predict_ms - (pre + inf + post))
    else:
        pre, inf, post, conv = 0.0, predict_ms, 0.0, 0.0  # MLX / RF-DETR: no split

    w = window(model.model_id)
    w.append((pre, inf, post, conv, draw_ms, total_ms))
    m_pre, m_inf, m_post = median(w, 0), median(w, 1), median(w, 2)
    m_conv, m_draw, m_total = median(w, 3), median(w, 4), median(w, 5)
    inf_fps = 1000 / m_inf if m_inf else 0.0
    total_fps = 1000 / m_total if m_total else 0.0

    line1 = f"{model.model_id} [{model.task}] {model.backend}/{model.device} | {frame.shape[1]}x{frame.shape[0]} | {len(detections)} obj"
    line2 = (
        f"pre {m_pre:4.1f}  inf {m_inf:5.1f} ({inf_fps:4.1f}fps)  post {m_post:4.1f}"
        f"  conv {m_conv:4.1f}  draw {m_draw:4.1f}  | total {m_total:5.1f}ms ({total_fps:4.1f}fps) n={len(w)}"
    )
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(annotated, line1, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(annotated, line2, (8, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 255), 1, cv2.LINE_AA)

    cv2.imshow("Webcam", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key in (ord("]"), ord("n")):
        idx = (idx + 1) % len(cycle)
        model = load(cycle[idx])
    elif key in (ord("["), ord("p")):
        idx = (idx - 1) % len(cycle)
        model = load(cycle[idx])

cap.release()
cv2.destroyAllWindows()
