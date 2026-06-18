"""Unified model zoo: RF-DETR + YOLO26, detection + segmentation.

Device-aware (MPS on Mac, CUDA, else CPU). Every model exposes the same
interface: `predict(bgr_frame, threshold) -> supervision.Detections`, plus
`class_names`, `task` ("det"/"seg") and `family` ("rfdetr"/"yolo").

Input is always a native OpenCV **BGR** uint8 frame — each wrapper does its own
colour conversion (RF-DETR wants RGB, ultralytics wants BGR).
"""
import numpy as np
import torch
import supervision as sv

# Contiguous 80-class COCO names (YOLO indexing). The MLX backend returns
# generic "class0".."class79" so we attach these ourselves.
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


# ----------------------------------------------------------------------------
# Registry — id -> spec. Nothing is loaded until build() is called for an id.
# ----------------------------------------------------------------------------
# RF-DETR ids match web/public/models/manifest.json. YOLO26 ids are new.
REGISTRY = {
    # --- RF-DETR detection ---
    "nano":         dict(family="rfdetr", task="det", rf="RFDETRNano",   resolution=384),
    "small":        dict(family="rfdetr", task="det", rf="RFDETRSmall",  resolution=512),
    "medium":       dict(family="rfdetr", task="det", rf="RFDETRMedium", resolution=576),
    "large":        dict(family="rfdetr", task="det", rf="RFDETRLarge",  resolution=704),
    # --- RF-DETR segmentation ---
    "seg-nano":     dict(family="rfdetr", task="seg", rf="RFDETRSegNano",   resolution=312),
    "seg-small":    dict(family="rfdetr", task="seg", rf="RFDETRSegSmall",  resolution=384),
    "seg-medium":   dict(family="rfdetr", task="seg", rf="RFDETRSegMedium", resolution=432),
    "seg-large":    dict(family="rfdetr", task="seg", rf="RFDETRSegLarge",  resolution=504),
    # --- YOLO26 detection ---
    "yolo26n":      dict(family="yolo", task="det", weights="yolo26n.pt", resolution=640),
    "yolo26s":      dict(family="yolo", task="det", weights="yolo26s.pt", resolution=640),
    # --- YOLO26 segmentation ---
    "yolo26n-seg":  dict(family="yolo", task="seg", weights="yolo26n-seg.pt", resolution=640),
    "yolo26s-seg":  dict(family="yolo", task="seg", weights="yolo26s-seg.pt", resolution=640),
    # --- YOLO26 on MLX (native Apple Silicon, no torch at runtime) ---
    "yolo26n-mlx":      dict(family="mlx", task="det", weights="weights_mlx/yolo26n.npz",     resolution=640),
    "yolo26s-mlx":      dict(family="mlx", task="det", weights="weights_mlx/yolo26s.npz",     resolution=640),
    "yolo26n-seg-mlx":  dict(family="mlx", task="seg", weights="weights_mlx/yolo26n-seg.npz", resolution=640),
    "yolo26s-seg-mlx":  dict(family="mlx", task="seg", weights="weights_mlx/yolo26s-seg.npz", resolution=640),
}


class Model:
    """Common wrapper. Subclasses implement `_predict`."""

    def __init__(self, model_id, spec, class_names, backend="torch", device=None):
        self.model_id = model_id
        self.family = spec["family"]
        self.task = spec["task"]
        self.resolution = spec["resolution"]
        self.backend = backend           # "torch" | "mlx"
        self.device = device or DEVICE
        self.class_names = class_names  # {int id: str name}
        # Per-call stage timings in ms: {"preprocess","inference","postprocess"}.
        # None for backends that don't expose a split (RF-DETR, MLX).
        self.last_speed = None

    def predict(self, bgr, threshold):
        raise NotImplementedError

    def label(self, class_id, conf):
        name = self.class_names.get(int(class_id), str(class_id))
        return f"{name} {conf:.2f}"


class RFDETRModel(Model):
    def __init__(self, model_id, spec, optimize=True):
        import rfdetr
        from rfdetr.assets.coco_classes import COCO_CLASSES

        cls = getattr(rfdetr, spec["rf"])
        self.model = cls(device=DEVICE)
        if optimize:
            self.model.optimize_for_inference()
        # COCO_CLASSES is a dict keyed by COCO category-id (1..90), which is
        # exactly what RF-DETR emits as class_id — use it directly.
        super().__init__(model_id, spec, dict(COCO_CLASSES))

    def predict(self, bgr, threshold):
        # RF-DETR wants RGB; .copy() avoids negative-stride tensor errors.
        rgb = bgr[:, :, ::-1].copy()
        return self.model.predict(rgb, threshold=threshold)


class YOLOModel(Model):
    def __init__(self, model_id, spec):
        from ultralytics import YOLO

        self.model = YOLO(spec["weights"])
        self.model.to(DEVICE)
        super().__init__(model_id, spec, dict(self.model.names))

    def predict(self, bgr, threshold):
        # ultralytics consumes BGR numpy directly (OpenCV convention).
        res = self.model.predict(
            bgr,
            device=DEVICE,
            imgsz=self.resolution,
            conf=threshold,
            verbose=False,
        )[0]
        self.last_speed = dict(res.speed)  # {"preprocess","inference","postprocess"} ms
        return sv.Detections.from_ultralytics(res)

    # --- fast path for the bridge: forward via ultralytics' optimized backend,
    #     decode masks at proto res (no 640 upsample, no sv conversion). ~2x. ---
    def _backend(self):
        if getattr(self, "_be", None) is None:
            import numpy as np
            warm = np.zeros((self.resolution, self.resolution, 3), np.uint8)
            self.model.predict(warm, imgsz=self.resolution, device=DEVICE, verbose=False)
            self._be = self.model.predictor.model  # AutoBackend (warmed, fused, fp16-aware)
        return self._be

    def predict_fast(self, rgb, threshold):
        """rgb: HWC uint8 RGB. Returns {xyxyn, scores, classes,
        masks(uint8 N,mh,mw|None), mh, mw} — boxes normalized 0..1 in model
        space, masks at proto res cropped to box. Mirrors the ONNX/web decode;
        avoids ultralytics' full-res mask postprocess."""
        import time
        import cv2
        import numpy as np
        import torch

        be = self._backend()
        R = self.resolution
        if rgb.shape[0] != R or rgb.shape[1] != R:
            rgb = cv2.resize(rgb, (R, R))

        t0 = time.perf_counter()
        # Single CPU copy: HWC view -> contiguous CHW (writable, so no read-only
        # warning and no extra GPU .contiguous()).
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        x = torch.from_numpy(chw).to(DEVICE).unsqueeze(0).float().div_(255)
        if getattr(be, "fp16", False):
            x = x.half()
        t1 = time.perf_counter()
        with torch.inference_mode():
            out = be(x)
        if DEVICE == "mps":
            torch.mps.synchronize()
        t2 = time.perf_counter()

        preds = out[0]
        if isinstance(preds, (list, tuple)):           # seg: (dets, proto)
            dets, proto = preds[0][0].float().cpu().numpy(), preds[1][0].float().cpu().numpy()
        else:                                          # det: just dets
            dets, proto = preds[0].float().cpu().numpy(), None

        keep = dets[:, 4] >= threshold
        kept = dets[keep]
        xyxyn = kept[:, :4] / R
        scores = kept[:, 4].copy()
        classes = kept[:, 5].astype(int)

        masks = None
        mh = mw = 0
        if proto is not None and len(kept):
            nm, mh, mw = proto.shape
            logits = (kept[:, 6:6 + nm] @ proto.reshape(nm, -1)).reshape(-1, mh, mw)
            keepm = logits > 0                          # sigmoid > 0.5
            bx1 = (xyxyn[:, 0] * mw)[:, None, None]; bx2 = (xyxyn[:, 2] * mw)[:, None, None]
            by1 = (xyxyn[:, 1] * mh)[:, None, None]; by2 = (xyxyn[:, 3] * mh)[:, None, None]
            xs = np.arange(mw)[None, None, :]; ys = np.arange(mh)[None, :, None]
            masks = (keepm & (xs >= bx1) & (xs < bx2) & (ys >= by1) & (ys < by2)).astype(np.uint8)

        self.last_speed = {
            "preprocess": round((t1 - t0) * 1000, 2),
            "inference": round((t2 - t1) * 1000, 2),
            "postprocess": round((time.perf_counter() - t2) * 1000, 2),
        }
        return {"xyxyn": xyxyn, "scores": scores, "classes": classes,
                "masks": masks, "mh": mh, "mw": mw}


class MLXYOLOModel(Model):
    """YOLO26 running natively on MLX (Apple Silicon Metal) — no torch at runtime."""

    def __init__(self, model_id, spec):
        from yolo26mlx import YOLO

        ul_task = "segment" if spec["task"] == "seg" else "detect"
        self.model = YOLO(spec["weights"], task=ul_task)
        names = {i: n for i, n in enumerate(COCO80)}
        super().__init__(model_id, spec, names, backend="mlx", device="mlx")

    def predict(self, bgr, threshold):
        # Same BGR-numpy convention as ultralytics.
        res = self.model.predict(bgr, conf=threshold, imgsz=self.resolution)[0]
        b = res.boxes
        xyxy = np.asarray(b.xyxy, dtype=np.float32).reshape(-1, 4)
        if len(xyxy) == 0:
            return sv.Detections.empty()
        mask = None
        if self.task == "seg" and res.masks is not None:
            mask = np.asarray(res.masks.data).astype(bool)
        return sv.Detections(
            xyxy=xyxy,
            confidence=np.asarray(b.conf, dtype=np.float32).ravel(),
            class_id=np.asarray(b.cls).astype(int).ravel(),
            mask=mask,
        )


def build(model_id, **kwargs):
    if model_id not in REGISTRY:
        raise KeyError(
            f"Unknown model '{model_id}'. Available: {', '.join(REGISTRY)}"
        )
    spec = REGISTRY[model_id]
    if spec["family"] == "rfdetr":
        return RFDETRModel(model_id, spec, **kwargs)
    if spec["family"] == "yolo":
        return YOLOModel(model_id, spec, **kwargs)
    if spec["family"] == "mlx":
        return MLXYOLOModel(model_id, spec, **kwargs)
    raise ValueError(f"Unknown family {spec['family']}")
