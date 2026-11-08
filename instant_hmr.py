"""InstantHMR adapter and binary mesh transport for the existing Python bridge."""
import json
from copy import copy
import math
import struct
import sys
import time

import numpy as np

from setup_instanthmr import REVISION, SOURCE, resolve_assets, validate_fit

VARIANTS = ("nano", "tiny", "small", "medium", "medium-commercial")
DEVICES = {"cpu": "CPUExecutionProvider", "coreml": "CoreMLExecutionProvider",
           "cuda": "CUDAExecutionProvider"}
MAX_PERSONS = 4


def mesh_options(options):
    """Validate complete session settings before replacing any filter history."""
    if not isinstance(options, dict) or set(options) - {"smoothing", "betaScale", "maxPersons"}:
        raise ValueError("Unknown mesh options")
    result = {"smoothing": "one-euro", "betaScale": 1.0, "maxPersons": MAX_PERSONS, **options}
    if result["smoothing"] not in ("off", "one-euro", "alpha-beta", "output"):
        raise ValueError("Unknown smoothing mode")
    beta = result["betaScale"]
    if type(beta) not in (float, int) or not math.isfinite(beta) or not 0.25 <= beta <= 4:
        raise ValueError("Output responsiveness must be between 0.25 and 4")
    limit = result["maxPersons"]
    if type(limit) is not int or not 1 <= limit <= MAX_PERSONS:
        raise ValueError(f"Person limit must be between 1 and {MAX_PERSONS}")
    return result


class MeshModel:
    family = "instanthmr"
    task = "mesh"
    backend = "onnxruntime"
    last_speed = None

    def __init__(self, model_id, device, fit=1):
        self.fit = validate_fit(fit)
        variant = model_id.removeprefix("instanthmr-")
        if variant not in VARIANTS or device not in DEVICES:
            raise ValueError("Unknown InstantHMR model or device")
        marker = SOURCE / "REVISION"
        if not marker.exists() or marker.read_text().strip() != REVISION:
            raise RuntimeError("Install InstantHMR: pip install -r requirements-instanthmr.txt; "
                               "python setup_instanthmr.py")
        import onnxruntime as ort
        if DEVICES[device] not in ort.get_available_providers():
            raise RuntimeError(f"{device} is unavailable in this ONNX Runtime installation")
        sys.path.insert(0, str(SOURCE))
        from instanthmr import InstantHMR, SKELETON_EDGES
        import models

        self.model_id = model_id
        self.device = device
        network, body = resolve_assets(variant, fit)
        self.hmr = InstantHMR(network, body_path=body, device=device, max_batch=MAX_PERSONS)
        if self.hmr.provider != DEVICES[device]:
            raise RuntimeError(f"Requested {device}, but InstantHMR initialized {self.hmr.provider}")
        self.edges = SKELETON_EDGES
        self.detector = models.build("yolo26n")
        self.hmr.warmup(1)
        self.configure({})

    def for_session(self, options):
        # Share expensive ONNX sessions; keep pose filters and output filters
        # private to this WebSocket. The bridge executes inference serially.
        session = copy(self)
        session.hmr = copy(self.hmr)
        session.configure(options)
        return session

    def configure(self, options):
        from instanthmr.smoothing import PoseFilter, TemporalSmoother

        options = mesh_options(options)
        mode = options["smoothing"]
        pose_filter = PoseFilter(mode) if mode in ("one-euro", "alpha-beta") else None
        smoother = TemporalSmoother(beta_scale=options["betaScale"]) if mode == "output" else None
        self.options = options
        self.hmr.pose_filter = pose_filter
        self.smoother = smoother

    def reset_tracking(self):
        self.configure(self.options)

    def metadata(self):
        return {"faces": self.hmr.faces.reshape(-1).tolist(), "edges": self.edges,
                "coordinates": "camera-metres", "fit": self.fit, "meshOptions": self.options,
                "detector": f"yolo26n/{self.detector.device}"}

    def predict_mesh(self, rgb, threshold):
        t0 = time.perf_counter()
        # Zoo wrappers consume BGR. HMR consumes contiguous RGB.
        det = self.detector.predict(rgb[:, :, ::-1].copy(), threshold)
        indices = np.flatnonzero(det.class_id == 0)
        indices = indices[np.argsort(-det.confidence[indices])][:self.options["maxPersons"]]
        t1 = time.perf_counter()
        persons = self.hmr.predict(np.ascontiguousarray(rgb), det.xyxy[indices], det.confidence[indices],
                                   timestamp=t0)
        # Upstream returns early on empty boxes. Age its pose tracks explicitly
        # so a later person cannot inherit a departed person's history.
        if not len(indices) and self.hmr.pose_filter is not None:
            self.hmr.pose_filter([], np.empty((0, 204), np.float32), timestamp=t0)
        if self.smoother is not None:
            persons = self.smoother(persons, t0)
        t2 = time.perf_counter()
        return persons, {"detector": round((t1 - t0) * 1000, 2),
                         "hmr": round((t2 - t1) * 1000, 2)}


def encode_mesh(persons, w, h, server):
    """JSON descriptor (4-byte padded) + float32 camera-space arrays.

    Each person's payload is 2D joints, 3D joints, then mesh vertices. Faces are
    sent once in the ready message, rather than repeated in every frame.
    """
    t0 = time.perf_counter()
    descriptors, chunks = [], []
    for p in persons:
        descriptors.append({"bbox": (p.bbox / [w, h, w, h]).tolist(),
                            "score": p.confidence, "joints": len(p.joints_2d),
                            "vertices": len(p.vertices_local)})
        chunks.extend(np.asarray(a, dtype="<f4").tobytes() for a in
                      (p.joints_2d / [w, h], p.joints_3d_cam, p.vertices_cam))
    payload = b"".join(chunks)
    server["encode"] = round((time.perf_counter() - t0) * 1000, 2)
    header = json.dumps({"mesh": True, "persons": descriptors, "server": server},
                        allow_nan=False).encode()
    header += b" " * (-len(header) % 4)
    return struct.pack("<I", len(header)) + header + payload
