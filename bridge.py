"""WebSocket inference bridge — lets the web app use the native Python zoo
(RF-DETR / YOLO26 on MPS, YOLO26 on MLX) instead of in-browser ONNX.

    python bridge.py                 # ws://0.0.0.0:8765

Wire protocol (one WebSocket, binary + text):
  client -> server  text JSON  : {"type":"model","id":"yolo26n-seg-mlx","threshold":0.5}
                                  {"type":"threshold","value":0.4}
  client -> server  binary     : [u16 w][u16 h][u8 channels][...raw pixels RGBA/RGB]
  server -> client  text JSON  : {"type":"ready", id, family, task, backend, device}
  server -> client  binary     : [u32 jsonLen][json][mask bytes]
       json = {dets:[{x1,y1,x2,y2,score,cls,q}], mw, mh, seg, speed}
       coords are NORMALIZED 0..1 and UN-mirrored (model space); the browser
       applies the same mirror/scale it already uses for the ONNX path, so the
       draw code and a future WebRTC transport are unchanged.
  Masks are downsampled to mw x mh (default 160) uint8 0/1 — proto-ish res, so
  the browser upscales on the GPU canvas (cheap) exactly like the ONNX path.
"""
import argparse
import asyncio
import json
import struct
import time

import cv2
import numpy as np
import websockets

import models

MASK_RES = 160  # mask grid sent to the browser (it upscales on the canvas)

_cache = {}


def get_model(model_id):
    if model_id not in _cache:
        print(f"[bridge] loading {model_id} ...")
        _cache[model_id] = models.build(model_id)
        print(f"[bridge] ready {model_id} ({_cache[model_id].backend}/{_cache[model_id].device})")
    return _cache[model_id]


def encode(boxes_norm, scores, classes, masks, mh, mw, seg, speed, server):
    """Pack normalized boxes + low-res uint8 masks into the wire frame.
    boxes_norm: (N,4) xyxy in 0..1, masks: (N,mh,mw) uint8 or None."""
    te0 = time.perf_counter()
    dets = [
        {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
         "score": float(s), "cls": int(c), "q": i}
        for i, (b, s, c) in enumerate(zip(boxes_norm, scores, classes))
    ]
    mbytes = masks.tobytes() if masks is not None and len(masks) else b""
    server["encode"] = round((time.perf_counter() - te0) * 1000, 2)
    header = json.dumps({
        "dets": dets, "mw": mw, "mh": mh, "seg": seg,
        "speed": speed,    # pre/inf/post (torch yolo only)
        "server": server,  # parse / predict / encode (ms)
    }).encode()
    return struct.pack("<I", len(header)) + header + mbytes


def person_id(model):
    return 1 if model.family == "rfdetr" else 0   # RF-DETR COCO cat-id 1, YOLO 0


def run_matte(model, frame, threshold):
    """Least-back-and-forth path: return ONE person-union mask at proto res
    (no boxes, no per-instance). ~25KB vs ~128KB."""
    pid = person_id(model)
    if hasattr(model, "predict_fast"):
        r = model.predict_fast(frame[:, :, :3], threshold)
        mh, mw = r["mh"] or MASK_RES, r["mw"] or MASK_RES
        masks = r["masks"]
        sel = r["classes"] == pid
        if masks is None or not sel.any():
            return np.zeros((mh, mw), np.uint8), mh, mw
        return (masks[sel].max(0) > 0).astype(np.uint8), mh, mw

    bgr = frame[:, :, 2::-1].copy()
    det = model.predict(bgr, threshold=threshold)
    if det.mask is None or len(det) == 0 or not (det.class_id == pid).any():
        return np.zeros((MASK_RES, MASK_RES), np.uint8), MASK_RES, MASK_RES
    full = det.mask[det.class_id == pid].max(0).astype(np.uint8)
    small = cv2.resize(full, (MASK_RES, MASK_RES), interpolation=cv2.INTER_NEAREST)
    return small, MASK_RES, MASK_RES


def encode_matte(union, mh, mw, speed, server):
    te0 = time.perf_counter()
    mbytes = union.tobytes()
    server["encode"] = round((time.perf_counter() - te0) * 1000, 2)
    header = json.dumps({
        "matte": True, "mw": mw, "mh": mh, "seg": True,
        "speed": speed, "server": server,
    }).encode()
    return struct.pack("<I", len(header)) + header + mbytes


def run_inference(model, frame, w, h, threshold):
    """frame: HWC uint8 RGB(A). Fast path (predict_fast: forward + proto-res
    masks) when available, else sv.Detections downsampled to MASK_RES."""
    if hasattr(model, "predict_fast"):
        r = model.predict_fast(frame[:, :, :3], threshold)   # RGB straight through
        return (r["xyxyn"], r["scores"], r["classes"], r["masks"],
                r["mh"] or MASK_RES, r["mw"] or MASK_RES, model.task == "seg")

    bgr = frame[:, :, 2::-1].copy()                          # RGB(A) -> BGR for sv path
    det = model.predict(bgr, threshold=threshold)
    boxes = det.xyxy / np.array([w, h, w, h], dtype=np.float32)
    masks = None
    if det.mask is not None and len(det):
        masks = np.stack([
            cv2.resize(det.mask[i].astype(np.uint8), (MASK_RES, MASK_RES),
                       interpolation=cv2.INTER_NEAREST)
            for i in range(len(det))
        ])
    return (boxes, det.confidence, det.class_id, masks,
            MASK_RES, MASK_RES, det.mask is not None)


def disable_nagle(ws):
    # Large frames + request/response trigger Nagle/delayed-ACK ~40ms stalls.
    sock = ws.transport.get_extra_info("socket")
    if sock is not None:
        import socket as _s
        sock.setsockopt(_s.IPPROTO_TCP, _s.TCP_NODELAY, 1)


async def handler(ws):
    model = None
    threshold = 0.5
    matte = False
    disable_nagle(ws)
    print("[bridge] client connected")
    async for msg in ws:
        if isinstance(msg, str):
            req = json.loads(msg)
            if req.get("type") == "model":
                model = get_model(req["id"])
                threshold = float(req.get("threshold", 0.5))
                await ws.send(json.dumps({
                    "type": "ready", "id": model.model_id, "family": model.family,
                    "task": model.task, "backend": model.backend, "device": model.device,
                }))
            elif req.get("type") == "threshold":
                threshold = float(req["value"])
            elif req.get("type") == "matte":
                matte = bool(req["value"])   # person-union mask only
            continue

        # binary frame
        if model is None:
            continue
        t0 = time.perf_counter()
        w = struct.unpack_from("<H", msg, 0)[0]
        h = struct.unpack_from("<H", msg, 2)[0]
        ch = msg[4]
        frame = np.frombuffer(msg, dtype=np.uint8, offset=5).reshape(h, w, ch)
        t1 = time.perf_counter()
        if matte:
            union, mh, mw = run_matte(model, frame, threshold)
            server = {"parse": round((t1 - t0) * 1000, 2),
                      "predict": round((time.perf_counter() - t1) * 1000, 2)}
            out = encode_matte(union, mh, mw, model.last_speed, server)
        else:
            boxes, scores, classes, masks, mh, mw, seg = run_inference(model, frame, w, h, threshold)
            server = {"parse": round((t1 - t0) * 1000, 2),
                      "predict": round((time.perf_counter() - t1) * 1000, 2)}
            out = encode(boxes, scores, classes, masks, mh, mw, seg, model.last_speed, server)
        await ws.send(out)
    print("[bridge] client disconnected")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"[bridge] device={models.DEVICE}  serving ws://{args.host}:{args.port}")
    # compression=None is critical: permessage-deflate on the ~128KB mask
    # payloads adds ~45ms/frame. Disabling it cuts round-trip 58ms -> 11ms.
    async with websockets.serve(handler, args.host, args.port, max_size=None, compression=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
