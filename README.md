# object-detect

Real-time object detection + segmentation model zoo (**RF-DETR** and **YOLO26**)
over COCO classes. Two ways to run:

1. **Python webcam** (`cam-detect.py`) — native OpenCV window, PyTorch inference
   on the active device (MPS on Mac / CUDA / CPU). Switch models live.
2. **Web app** (`web/`) — browser, webcam, ONNX Runtime Web (WASM/WebGPU).
3. **Web app + Python bridge** (`bridge.py`) — browser UI, native inference. The
   web app can run any zoo model on MPS/MLX over a WebSocket (see below).

Python 3.11, venv lives in `./venv`.

## Model zoo

`models.py` is the single registry. Same interface for both families
(`predict(bgr, threshold) -> supervision.Detections`):

| family | backend | det ids | seg ids | native res |
|--------|---------|---------|---------|-----------|
| RF-DETR | torch (MPS/CUDA/CPU) | nano / small / medium / large | seg-nano / seg-small / seg-medium / seg-large | 384/512/576/704, seg 312/384/432/504 |
| YOLO26 | torch (MPS/CUDA/CPU) | yolo26n / yolo26s | yolo26n-seg / yolo26s-seg | 640 (all) |
| YOLO26 | **MLX** (Apple Silicon, no torch) | yolo26n-mlx / yolo26s-mlx | yolo26n-seg-mlx / yolo26s-seg-mlx | 640 (all) |

Resolutions are model-native and must not be changed — RF-DETR ties positional
embeddings to a fixed grid, YOLO26 reports COCO mAP at 640 (imgsz must be ×32).

### Native MLX backend (Apple Silicon)

[`yolo-mlx`](https://github.com/thewebAI/yolo-mlx) runs YOLO26 directly on Metal
via MLX — no PyTorch at runtime, fastest on Mac. One-time install + weight
convert (already done if `weights_mlx/*.npz` exist):

```bash
pip install "yolo-mlx[convert,segment] @ git+https://github.com/thewebAI/yolo-mlx"
mkdir -p weights_mlx
for w in yolo26n yolo26s yolo26n-seg yolo26s-seg; do
  yolo-mlx converters convert $w.pt -o weights_mlx/$w.npz --verify
done
```

Not in `requirements.txt` — MLX is macOS/Apple-Silicon only.

---

## 1. Python webcam

```bash
source venv/bin/activate
python cam-detect.py                      # default compare cycle
python cam-detect.py --model yolo26n-seg  # single model
# compare torch/MPS vs native MLX live:
python cam-detect.py --model yolo26n-seg,yolo26n-seg-mlx,seg-nano
```

Press `q` to quit; `[` / `]` (or `p` / `n`) cycle through the `--model` list
live. Box + label always; masks for seg models. HUD shows two lines: model /
backend / device / obj count, and `predict / draw / total ms | median ms +
fps`. Each model keeps its own rolling median (n=60) across switches, so you can
flip between backends and read stable comparable numbers. Weights auto-download
on first run.

First-time setup (if `venv` missing):

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Benchmark

```bash
source venv/bin/activate
python benchmark.py                              # default 3-way seg comparison
python benchmark.py yolo26n yolo26n-mlx          # any zoo ids
```

Times end-to-end `predict()` (with warmup) on each model's real backend. For
RF-DETR / YOLO26 ids it also times the exported ONNX across available providers
(CUDA / CoreML / CPU). MLX ids run on Metal.

### Apples-to-apples staging

Both the python cam HUD and the web stats line report the **same stages** as a
rolling median (n=60), so they can be compared directly:

| stage | meaning | python source | web source |
|-------|---------|---------------|------------|
| `pre`  | resize → tensor → device | ultralytics profiler | `preprocess()` canvas+normalize |
| `inf`  | model forward | ultralytics profiler | `session.run` |
| `post` | NMS-free decode + mask assembly | ultralytics profiler | decode loop / `buildYoloMasks` |
| `conv` | sv mask materialization (python only) | wall − (pre+inf+post) | — |
| `draw` | overlay masks + boxes | `draw_masks_fast` | `drawDetections` (contours) |

`inf` is the apples-to-apples model number. Both feed a 640px input for YOLO26.

yolo26n-seg, real image with objects, M-series Mac (median):

| backend | pre | inf | post | total predict |
|---------|----:|----:|-----:|--------------:|
| **MLX / Metal** | — | — | — | **8.8 ms** (bundled) |
| torch / MPS | 0.7 | **10.1** | 7.9 | 19.5 ms |
| onnx CoreML (in browser) | — | 15.2 | — | — |
| onnx CPU/WASM (in browser) | — | 22.7 | — | — |

Ranking holds: **MLX > torch/MPS > onnx-CoreML > onnx-CPU**. The earlier
"python is 110 ms" was never inference — it was `sv.MaskAnnotator` filling masks
at full webcam res. cam-detect now mirrors the frame, caps processing res
(`--size`, default 960) and draws masks via a single low-res overlay blend.

**python↔web bridge basis:** native MLX inference (~9 ms) beats in-browser ONNX
(~15–23 ms). A local Python↔web bridge serving MLX to the browser UI would give
native-Metal speed behind the web front-end.

---

## 2. Web app

```bash
cd web
npm install        # postinstall runs setup_models.py -> exports ONNX models
npm run dev        # https://localhost:5173  (self-signed cert, HTTPS required)
```

Open the printed `https://` URL. Accept the self-signed cert warning. Allow
webcam access.

Build / preview:

```bash
npm run build
npm run preview
```

### Notes

- Dev server forces **HTTPS** (`@vitejs/plugin-basic-ssl`) — needed for webcam +
  cross-origin isolation (COOP/COEP headers set in `vite.config.js`) so
  multithreaded WASM works.
- ONNX `.wasm` files are copied from `onnxruntime-web` into the build by
  `vite-plugin-static-copy`.

### Models

ONNX models live in `web/public/models/<id>/inference_model.onnx`, listed in
`web/public/models/manifest.json`. Each manifest entry carries `family`
(`rfdetr` / `yolo`) and `task` (`det` / `seg`) — the web decoder branches on
`family` (different input name, normalization, and output layout per family).

| id | resolution | family / type |
|----|-----------|------|
| nano / small / medium / large | 384 / 512 / 576 / 704 | RF-DETR detection |
| seg-nano / seg-small / seg-medium / seg-large | 312 / 384 / 432 / 504 | RF-DETR segmentation |
| yolo26n / yolo26s | 640 | YOLO26 detection |
| yolo26n-seg / yolo26s-seg | 640 | YOLO26 segmentation |

Export / re-export (needs the Python venv; auto-created if absent):

```bash
python setup_models.py                    # all models in manifest
python setup_models.py yolo26n-seg nano   # specific ids only
```

Already-exported models are skipped. RF-DETR exports via `rfdetr.export()`;
YOLO26 via `ultralytics` `export(format="onnx")` (NMS-free / end-to-end,
top-300 output).

---

## 3. Python inference bridge

Run native inference (MPS / MLX) behind the web UI — faster than in-browser ONNX
on Apple Silicon. Python does inference only; the browser draws (cheap).

```bash
source venv/bin/activate
python bridge.py                 # ws://0.0.0.0:8765

# separate terminal
cd web && npm run dev            # then pick a "Python ·" backend in the UI
```

In the web app's **Backend** dropdown choose `Python · torch/MPS` or
`Python · MLX`. Vite proxies `wss://<host>/bridge` → the python ws (so no
mixed-content block on the HTTPS page); `python-mlx` requires a `yolo26*` model.

**Transport:** raw RGBA frame up (no codec — loopback is ~GB/s), normalized dets
+ proto-res (160²) masks down, one request in flight (backpressure). Same
decoded shape as the ONNX path, so all draw code is shared. A WebRTC transport
for a *remote* GPU box can drop into `web/src/bridge.js` reusing
`decodeBridgeResult()`.

> **Critical:** the server sets `compression=None`. Default permessage-deflate on
> the ~128KB mask payloads adds ~45ms/frame — disabling it cut round-trip
> 58ms → 11ms.

**Measured round-trip (bus.jpg, M-series, median), instrumented end to end:**

| backend | rtt | model fwd | decode+pack | transport |
|---------|----:|----------:|------------:|----------:|
| yolo26n-seg · torch/MPS | **12.5 ms** | 9.8 | ~1.5 | ~1 |
| yolo26n-seg · MLX | 10.7 ms | bundled | — | ~1 |
| yolo26n (det) · torch/MPS | 9.8 ms | 8.9 | ~0 | ~1 |
| seg-nano (RF-DETR) · MPS | ~38 ms | model-bound | sv path | ~1 |

The torch/YOLO path uses **`YOLOModel.predict_fast`**: forward through
ultralytics' warmed `AutoBackend`, then decode masks at proto res (160²) on the
CPU — exactly like the ONNX/web path. This skips ultralytics' full-res mask
postprocess + `sv.Detections` conversion that previously cost ~10 ms
(`20.5 → 12.5 ms`). Transport itself is <2 ms (parse + encode + wire); the
websocket was never the bottleneck. RF-DETR stays on the sv path (model-bound,
different arch). Both the web stats line and `bridge.py` payload expose the
split (`rtt = model + decode + transport`).

> Earlier debugging milestones baked into the code: `compression=None`
> (permessage-deflate added ~45 ms on mask payloads), raw RGBA frames (no codec
> on loopback), and `predict_fast` (skip full-res masks). End state: every YOLO
> path is within ~3 ms of its raw inference floor.

See [`docs/realtime-python-js-bridge.md`](docs/realtime-python-js-bridge.md) for
the portable recipe (reusable in other projects).

---

## 4. Background removal (person matte)

The "throw a frame in, get the person cut out" use case — person class only, no
boxes / contours / sidebar, leanest pipeline.

**Web:** toggle **Cutout (person only, no UI)** in the controls. Works on any
backend (WebGPU / WASM / Python bridge); composites `video × person-mask` on the
GPU (`drawImage` + `destination-in`) — no full-res per-pixel JS. The stats line
shows fps per backend so you can compare directly.

**Bridge matte mode (least back-and-forth):** when cutout is on with a Python
backend, the web sends `{type:"matte"}` and the server returns a **single
person-union mask** (~25 KB, 160²) instead of all per-instance masks (~128 KB) —
no boxes, no per-instance. Latency on loopback is the same (inference-bound) but
the payload is 5× smaller and **flat regardless of how many people** are in
frame (`run_matte` / `encode_matte` in `bridge.py`).

**Benchmark** the raw pipeline (inference → person-union mask → apply):

```bash
python bench_matte.py                       # default backends
python bench_matte.py yolo26n-seg-mlx       # specific
```

| backend | person matte | infer | union+apply |
|---------|------------:|------:|------------:|
| yolo26n-seg · MLX | **114 fps** (8.8 ms) | 8.6 | 0.2 |
| yolo26n-seg · torch/MPS | **88 fps** (11.3 ms) | 11.2 | 0.1 |
| seg-nano (RF-DETR) · MPS | 24 fps (42 ms) | 41.8 | 0.1 |

**The matte itself is ~free (~0.2 ms)** — it's 100% inference-bound. So for
background removal: pick the fastest seg model, skip all UI. This is also why
WebGPU ≈ Python-bridge for full-UI seg in the browser — the cost was the
mask-drawing/contour UI, not inference. Strip the UI (this mode) and you're
reading raw model speed.

---

## 5. InstantHMR: live 3D body meshes

The dashboard also supports [InstantHMR](https://github.com/Naturalpad-studio/InstantHMR)
through the existing Python bridge. It uses our YOLO26n detector to find up to
four people, then runs the InstantHMR ONNX network and the MHR body graph.
Each person has 70 keypoints and an 18,439-vertex mesh.

One-time setup (Python 3.10–3.12; use the existing project venv):

```bash
source venv/bin/activate
pip install -r requirements-instanthmr.txt
python setup_instanthmr.py
```

Setup downloads the upstream runtime and licences into ignored
`vendor/InstantHMR/`, and caches ONNX weights in the Hugging Face cache.
The source revision and weight revision are pinned in `setup_instanthmr.py`.
It does not install the upstream Rerun demo or run any Git commands.

```bash
python bridge.py
# separate terminal:
cd web && npm run dev
```

Select **InstantHMR medium-commercial · mesh**, then **Start Camera**. The
dashboard selects **Python · InstantHMR** automatically. The **Mesh device**
control explicitly selects CPU, CoreML (Mac), or CUDA (NVIDIA); unavailable
providers produce an error. CPU is the initial selection. CoreML may partition
unsupported ONNX operations onto CPU, as in upstream; the body graph uses CPU
on Mac. CUDA requires the GPU ONNX Runtime package installed by the optional
requirements file on Linux/Windows.

Drag the shaded 3D view to orbit, use **Wireframe mesh**, or download the current
frame as an **OBJ** (all people, triangle faces, camera-space metres: X right,
Y down, Z forward). The viewer rotates the coordinate axes for an upright
display and centers the group, preserving relative positions. The 2D overlay
matches the mirrored webcam; the 3D scene and export preserve body handedness.
Timing shows detector and HMR + mesh separately. Frames preserve their aspect
ratio and are capped at 640 pixels on the long side before transport. Background
cutout and segment controls are disabled for mesh models.

The mesh controls also expose:

- **Smoothing:** upstream's **Pose · One Euro** preset is on by default. It
  filters body angles before mesh generation; the 2D keypoints and camera
  translation remain raw. **Pose · alpha-beta** is the other upstream preset.
  **Off** provides an unsmoothed comparison.
- **Output · One Euro:** smooths the resulting 2D/3D keypoints, vertices and
  camera translation instead. Its **Response** slider scales upstream's beta
  values from 0.25× to 4× (default 1×). Lower is steadier; higher follows motion
  faster. This mode can introduce lag and does not preserve the exact pose/body
  relationship in the way pose filtering does. Pose and output filters are
  mutually exclusive.
- **Fit quality:** off (network only), one crop (default, faster), or two crops
  (higher accuracy). Changing fit loads a different ONNX graph; each downloads
  once. Prefetch with `python setup_instanthmr.py --fit 2`.
- **People:** cap inference at 1–4 people. Use one for a faster single-person
  webcam test. The confidence slider remains the detector threshold.
- **Reset smoothing history:** clears temporal state without reloading the
  network. Changing smoothing/response/person limit also clears history. Filter
  state belongs to each WebSocket, resets on camera restart, and ages out after
  six missed inference frames. Filters use actual monotonic frame timestamps,
  rather than assuming a 30 fps inference rate.

Smoothing, response, and person limit changes apply live without downloading or
reloading model weights. Pose filters use the pinned upstream coefficients:
One Euro cutoff 12 Hz, beta 0.5, derivative cutoff 1 Hz; alpha-beta alpha 0.8,
beta 0.005. Their internal coefficients are kept as presets.

The default weights are `medium-commercial`. The other four model choices are
marked **non-commercial**, matching the upstream model licences. Each variant
downloads on first use; prefetch another with
`python setup_instanthmr.py --model small`. Weights remain subject to the
[upstream SAM licence](https://github.com/Naturalpad-studio/InstantHMR/blob/main/LICENSE-SAM).

Focused checks with the optional integration installed (no weights, camera, or
GPU used):

```bash
venv/bin/python -m unittest test_instant_hmr.py
node --test web/src/mesh.test.js
```

## Dependencies

- **Python** (`requirements.txt`): rfdetr, ultralytics, supervision,
  opencv-python, torch, torchvision, onnx, onnxruntime, websockets. Plus
  optional `yolo-mlx` (Apple Silicon) for the MLX backend.
- **Web** (`web/package.json`): onnxruntime-web, vite.

## Gotchas

- **Channel order.** RF-DETR wants RGB; ultralytics/YOLO wants BGR. `models.py`
  takes a native OpenCV BGR frame and each wrapper converts as needed
  (RF-DETR's `frame[:, :, ::-1].copy()` — `.copy()` avoids negative-stride
  PyTorch errors).
- **Class indexing differs.** RF-DETR emits 91 COCO category-ids (0-90); YOLO26
  emits contiguous 80 (0-79). The web app keeps a name+colour table per family.
- **Web preprocessing.** RF-DETR uses ImageNet mean/std + input `input`; YOLO26
  uses plain 0-1 + input `images`. Both stretch the frame to a square (no
  letterbox) — minor box shift on very non-square inputs, consistent across
  families.
