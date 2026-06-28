# SPEC — pose detection (RTMW whole-body) in web via ONNX

caveman encoded. `?` = unconfirmed, verify vs real exported model. ids monotonic.

## §G — goal

Add whole-body pose (RTMW, 133 keypoints) as new model family in web app.
Run in-browser via onnxruntime-web, WebGPU primary / WASM fallback. Draw skeleton
overlay. Python export = later stage, out of scope now (but onnx artifact must
exist for web to load).

## §C — constraints

- C1 web/JS first. python side (export, bridge, cam-detect) deferred — user not care now.
- C2 reuse existing `web/` stack: `onnxruntime-web@^1.24.3` + `backendSelect` EP path.
  NOT add `@microsoft/onnxruntime-web` (old scope, dup) nor node `canvas` (handoff wrong, browser has canvas).
- C3 RTMW = **top-down** pose. needs person bbox → crop → pose per crop. no "one-stage RTMW"
  exists (handoff claim wrong). bound loop = cap max persons/frame.
- C4 reuse existing YOLO26 det (person class) already in web as bbox source. no new detector.
- C5 no new hardcoded webgpu→wasm fallback block — use existing EP/backend mechanism (CLAUDE.md: no brittle fallbacks).
- C6 target M3 Max / RTX 4090. budget: pose inference <10ms webgpu (per person crop).
- C7 fp32 model first (target GPUs strong). fp16/quant only if bandwidth-bound. defer.
- C8 onnx = **rtmlib prebuilt** (hosted known-good), not self-export. download artifacts.
- C9 offer multiple variants in zoo (like existing nano/small flow); default = fastest.
- C10 pose is **top-down: detector REQUIRED** for correct results (RTMW wants a tight person crop,
   not a full frame). DEFAULT = yolo26n detector → per-person crop → pose. modes: all / single / cap-N.
   whole-frame (no detector) kept only as explicit DEBUG option, never default. detector-load failure =
   error + draw nothing, NOT silent whole-frame fallback (global: no brittle fallbacks).
- C11 3D pose = **RTMW3D-x** (rtmw3d-x onnx, 352MB, from HuggingFace Soykaf/RTMW3D-x). single .onnx (no zip).
   heavy: ~10-15fps webgpu. same top-down crop path (288x384) + yolo26n detector. adds **three.js** dep for a
   3D skeleton viewer. only published variant (no small/fast 3D). 2D RTMW family stays as-is.

## §I — surfaces

- I.onnx   ! `web/public/models/<id>/inference_model.onnx` — RTMW artifact
- I.manif  ! `web/public/models/manifest.json` — new entry, `family:"rtmw"` `task:"pose"`
- I.main   ! `web/src/main.js` — new family branch: preprocess / postprocess / draw / sidebar
- I.id     ! zoo ids (rtmlib prebuilt): `rtmw-m` (256x192 dw-l-m, FASTEST/default), `rtmw-l` (256x192 dw-x-l),
            `rtmw-l-384` (384x288 dw-x-l). no separate rtmw-x onnx exists in rtmlib.
- I.det    ! existing yolo26 det path (person class_id 0) reused for bbox
- I.ep     ! existing `backendSelect` (webgpu / wasm) — pose plugs in, no separate session config
- I.ctrl   ! new UI controls (parallel to existing threshold/maskOpacity): person mode (single/cap-N/all);
            overlay on-demand toggle + per-group toggles (body / hands / face); kpt-score threshold slider
- I.id3d   ! `rtmw3d-x` (288x384, family `rtmw3d`, task `pose3d`) — real 3D whole-body
- I.three  ! `web/src/pose3d.js` + `#pose3d` canvas — three.js scene, skeleton (joints+bones), auto-orbit

## §V — invariants

- V1 CONFIRMED: input name `input`, shape `[batch,3,H,W]` NCHW. rtmw-m/rtmw-l=256x192 (H256,W192),
     rtmw-l-384=384x288 (H384,W288). read dims per loaded id from manifest, not hardcoded.
- V2 preprocess = **bbox affine warp** (person center+scale → model res), NOT naive full-frame
     letterbox. pad bbox to model aspect + 1.25, keep rect M to invert. ONE drawImage(video, srcRect →
     poseW×poseH) does crop+scale to exact model input in a single op — no full-frame copy, no 2nd resize.
- V3 CONFIRMED (pipeline.json, to_rgb=True): normalize **RGB, 0-255 scale**, mean=`[123.675,116.28,103.53]`
     std=`[58.395,57.12,57.375]`. NOT baked in onnx → apply in JS. getImageData = RGBA already RGB-order.
     (handoff "0-1 ImageNet" wrong on scale; spec earlier "BGR" wrong on order.)
- V4 CONFIRMED: output `simcc_x[batch,133,Wx]` `simcc_y[batch,133,Hy]`. 256x192→Wx384,Hy512; 384x288→Wx576,Hy768.
     split_ratio r=2.0 (Wx=W*r, Hy=H*r). decode: argmax per axis → /r → (x,y) input-space px; score=max simcc val.
- V5 keypoint topology = COCO-WholeBody 133: `0-16` body(17), `17-22` foot(6), `23-90` face(68),
     `91-111` L-hand(21), `112-132` R-hand(21). handoff slices WRONG — use this.
- V6 decoded kpts in model-input space → apply inverse affine M⁻¹ → source-frame coords.
     respect existing video mirror/flip (same as yolo/rfdetr draw path).
- V7 EP via existing backendSelect; webgpu primary, wasm fallback through existing code, no new block (C5).
- V8 web decode branches on manifest `family` ("rtmw") parallel to "rfdetr"/"yolo".
- V9 per-keypoint score threshold; kpt below thresh not drawn; bone drawn only if both ends pass.
- V10 pose top-down (DEFAULT). yolo26n detector → person boxes → per-box tight crop (pad 1.25) → pose.
     modes: all / cap3 / cap2 / single(top-1 conf). loop cost = N_selected × pose-infer. detector load fail =
     show error + draw nothing (NO silent whole-frame fallback). whole-frame = explicit DEBUG option only.
- V11 skeleton = **joints + bones**. joints = filled circles at kpts passing V9. bones = lines over
     defined edge sets per group: body (COCO-17 limb pairs), foot (heel/toe links to ankles),
     each hand (5 finger chains: wrist→[4 joints] ×5), face (iBUG-68 contour polylines: jaw/brows/
     eyes/nose/lips; or dots if contours overkill). bone drawn iff both endpoints pass V9.
- V12 per-group color: body, L-hand, R-hand, face each distinct (reuse hsl scheme like existing COLORS).
- V13 sizing: joint radius + bone width scale w/ box/frame size (not fixed px) so far targets still readable.
- V14 overlay = **on-demand**: drawn only when pose model active AND overlay toggle on. per-group
     toggles (body/hands/face) gate their joints+bones independently. default all on.
- V15 CONFIRMED rtmw3d-x I/O: input `input[1,3,384,288]`; outputs `output[1,133,576]`=simcc_X,
     `1554[1,133,768]`=simcc_Y, `1556[1,133,576]`=simcc_Z. split_ratio 2.0 all axes. (order resolved
     empirically: `output` decodes to real image-x; `1556` is flat-in-x = depth.)
- V16 3D decode: x=argmax(output)/2, y=argmax(1554)/2 (input px → invert crop M for 2D overlay). z needs the
     RTMPose3d transform: z_px=argmax(1556)/2; **z=(z_px/(poseW/2) - 1) × z_range** (z_range=2.1744869) — NOT
     raw px (raw makes depth look flat). For the 3D scene normalize x,y by poseH/2 (aspect-preserving) so x,y,z
     share scale. score=0.5*(maxX+maxY); z maxima low, don't gate on z. (verified: depth≈41% body height,
     nose fwd of ears, feet back.)
- V17 3D viewer (three.js): joints+bones in a 3D scene, reuse POSE_GROUPS edges/colors. center each person
     on pelvis (mid hips 11,12), flip Y (image down → 3D up), mirror x to match flipped video.
     shown only for family `rtmw3d`. 2D overlay still drawn from x,y. OrbitControls: drag rotate / right-drag
     pan / wheel zoom; auto-rotate default, any manual grab disables it (UI checkbox synced); Reset restores
     view + re-enables auto. Depth-scale slider (0.5–4×, default 1.5) exaggerates z for perception.

## §T — tasks

| id | st | task | cites |
|----|----|------|-------|
| T1 | x | download rtmlib prebuilt onnx (rtmw-m/l 256, rtmw-l 384) → web/public/models/<id>/; ? resolved | I.onnx,C8 |
| T2 | x | manifest entries (family=rtmw task=pose, resW/resH); default order rtmw-m first (fastest) | I.manif,V8,C9 |
| T3 | x | detector path (DEFAULT, yolo26n): per-person tight crop→pose; modes all/single/cap-N; load-fail=error not fallback | C3,C4,C10,V10 |
| T3b | x | whole-frame path (DEBUG option only, not default): single full-frame bbox, no detector | C10,V10 |
| T4 | x | preprocess: bbox affine (aspect-pad+1.25) drawImage crop→input + RGB normalize, keep rect M | V1,V2,V3 |
| T5 | x | inference: ort session via existing EP path (reused create), feed [1,3,H,W] tensor | I.ep,V7 |
| T6 | x | postprocess: SimCC argmax/split_ratio → kpts+scores → invert rect M to source coords | V4,V6,V9 |
| T7a | x | edge sets: body COCO-17, foot links, hand finger chains ×2; face = dots (V11 allows) | V5,V11 |
| T7b | x | drawPose: joints (circles) + bones (lines), per-group colour, size scaled to box, mirror x | V11,V12,V13 |
| T7c | x | gate overlay + per-group (body/hands/face) via toggles | V14,I.ctrl |
| T8 | x | UI: pose ids auto in modelSelect; pose-controls (mode/kpt-thresh/group/overlay); detectLoop branch | I.main,I.ctrl,V8,V14 |
| T9 | ~ | LIVE-verified via CDP (fake-cam person): WebGPU boot ✓, detector→pose top-down ✓, 1ppl, skeleton on person no-drift ✓, overlay 13.7k px ✓, toggles ✓. CAVEAT: inf 35ms (det+pose combined) > 10ms target C6; pre 16.7ms = 2× getImageData readback. mem-5k soak not run. perf optimize = future | V1-V14,C6 |
| T10 | x | download rtmw3d-x onnx (352MB HF) → web/public/models/rtmw3d-x/; ? I/O resolved (V15) | I.id3d,C11 |
| T11 | x | manifest rtmw3d-x (family=rtmw3d task=pose3d resW288 resH384); setup_models.py downloads onnx (reproduce) | I.manif,I.id3d |
| T12 | x | 3D decode: 3-axis SimCC argmax/2 → x,y (invert crop) + z depth; reuses detector+crop path | V15,V16 |
| T13 | x | three.js viewer pose3d.js + #pose3d canvas: joints+bones vtx-colored, pelvis-center, auto-orbit+drag | I.three,V17 |
| T14 | x | LIVE via CDP: rtmw3d-x WebGPU ✓, 1ppl, 3D skeleton renders w/ real depth (3/4 orbit) ✓, 2D overlay ✓. poseInf 69ms ~12fps (matches C11 heavy) | I.main,V16,V17 |

## §B — bugs

| id | date | cause | fix |
|----|------|-------|-----|
