# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Real-time object **detection + segmentation** zoo (RF-DETR and YOLO26, COCO
classes) with three ways to run the same models:

1. **Python webcam** (`cam-detect.py`) — native OpenCV window, PyTorch inference (MPS/CUDA/CPU), live model switching.
2. **Web app** (`web/`) — browser + webcam, ONNX Runtime Web (WASM/WebGPU).
3. **Python bridge** (`bridge.py`) — browser UI, native inference (MPS/MLX) over a WebSocket; faster than in-browser ONNX on Apple Silicon.

`README.md` is the source of truth for setup, benchmark numbers, and the bridge
wire protocol. This file is the orientation map — read it first, then the README
section for the part you're touching.

## Commands

```bash
source venv/bin/activate           # Python 3.11 venv in ./venv

# Python webcam (q quit, [ ] or p n cycle the --model list live)
python cam-detect.py
python cam-detect.py --model yolo26n-seg,yolo26n-seg-mlx,seg-nano

# Benchmarks (these are the perf-check scripts — there is no unit test suite)
python benchmark.py [<id> ...]     # end-to-end predict() per backend, incl. ONNX providers
python bench_matte.py [<id> ...]   # person-union matte pipeline

# Python inference bridge for the web app
python bridge.py                   # ws://0.0.0.0:8765

# Web app (separate terminal)
cd web && npm install              # postinstall runs ../setup_models.py -> exports ONNX
cd web && npm run dev              # https://localhost:5173 (self-signed cert, HTTPS required)

# Export / re-export ONNX models for the web app (already-exported ids are skipped)
python setup_models.py [<id> ...]  # canonical exporter, bootstraps venv, both families
```

## Architecture

- **`models.py` is the single registry and the heart of the project.** `REGISTRY`
  maps every model id → spec; `build(id)` returns a wrapper. All wrappers expose
  one interface: `predict(bgr_frame, threshold) -> supervision.Detections`, plus
  `class_names`, `task` (`det`/`seg`), `family` (`rfdetr`/`yolo`/`mlx`), `backend`,
  `device`, `last_speed`. Add a model = add a `REGISTRY` entry (+ a manifest entry
  if it needs a web ONNX export). Nothing loads until `build()`.
- **Three consumers share that interface:** `cam-detect.py` (native window),
  `bridge.py` (WebSocket server), and the ONNX path in `web/src/main.js`.
- **`bridge.py` and the web ONNX path produce the same decoded shape** (normalized
  boxes + proto-res 160² masks), so `web/src/main.js` draw code is identical
  whether dets come from ONNX or the bridge. A future WebRTC transport can reuse
  `decodeBridgeResult()` in `web/src/bridge.js`.
- **`web/src/main.js`** (~1150 lines, no framework) is the whole front-end: model
  loading w/ progress, `preprocess`/`postprocess` per family, mask decode +
  contour tracing, sidebar, cutout mode, the `detectLoop`. It branches on
  `family` (different input name, normalization, output layout per family — see
  Gotchas).
- **`setup_models.py`** is the canonical ONNX exporter (both families, auto-creates
  venv, re-execs inside it). `export_onnx.py` is a legacy RF-DETR-only variant —
  prefer `setup_models.py`.
- **Two fast paths matter:** `YOLOModel.predict_fast` (forward via ultralytics'
  warmed `AutoBackend`, decode masks at proto res — skips full-res mask
  postprocess + sv conversion, ~2×) and `bridge.py` matte mode (one person-union
  mask, ~25KB vs ~128KB). RF-DETR has no fast path; it stays on the sv route.

## Key invariants (don't break these)

- **Model resolutions are fixed** (`REGISTRY[...]["resolution"]`). RF-DETR ties
  positional embeddings to a fixed grid; YOLO26 reports COCO mAP at 640 (imgsz
  must be ×32). Do not "make them configurable."
- **Channel order:** `models.py` always takes a native OpenCV **BGR** frame. Each
  wrapper converts internally — RF-DETR wants RGB (`frame[:, :, ::-1].copy()`;
  `.copy()` is required to avoid negative-stride PyTorch errors), ultralytics/MLX
  consume BGR directly. Don't add a global convert.
- **Class indexing differs by family:** RF-DETR emits 91 COCO category-ids (0-90,
  via `rfdetr.assets.coco_classes.COCO_CLASSES`); YOLO/MLX emit contiguous 80
  (0-79, the `COCO80` table for MLX which returns generic names). The web app
  keeps a name+colour table per family. `person` is cat-id 1 (RF-DETR) vs 0 (YOLO)
  — see `person_id()` in `bridge.py`.
- **`bridge.py` sets `compression=None`** — permessage-deflate on ~128KB mask
  payloads adds ~45ms/frame. Also `TCP_NODELAY` (Nagle/delayed-ACK stalls). Don't
  re-enable compression.
- **Web dev server requires HTTPS** (`@vitejs/plugin-basic-ssl`) for webcam +
  cross-origin isolation (COOP/COEP in `vite.config.js`) so multithreaded WASM
  works. Vite proxies `wss://<host>/bridge` → `ws://localhost:8765` to avoid
  mixed-content blocking.

## Dependencies & weights

- Python deps in `requirements.txt`. **MLX backend (`yolo-mlx`) is NOT in it** —
  Apple-Silicon only; install separately and convert weights to `weights_mlx/*.npz`
  (see README "Native MLX backend"). `mlx` ids fail to build without it.
- Weights (`*.pth`/`*.pt`/`*.npz`) and exported `*.onnx` are gitignored and
  auto-downloaded/exported on first use — they are large, do not commit them.
