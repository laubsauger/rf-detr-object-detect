import * as ort from "onnxruntime-web";
import { createBridge, decodeBridgeResult } from "./bridge.js";
import { initPose3d, updatePose3d, setPose3dRunning, setAutoRotate, resetView, onPose3dInteract } from "./pose3d.js";

ort.env.logLevel = "error";
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

const threadsOk = typeof SharedArrayBuffer !== "undefined";
console.log(`[ORT] Threading: ${threadsOk ? "enabled" : "DISABLED (no SharedArrayBuffer)"}, cores: ${navigator.hardwareConcurrency}`);

// ONNX model outputs 91 classes (COCO category IDs 0-90, index 0 is background)
const COCO_CLASSES = new Array(91).fill(null);
COCO_CLASSES[1]="person";COCO_CLASSES[2]="bicycle";COCO_CLASSES[3]="car";COCO_CLASSES[4]="motorcycle";
COCO_CLASSES[5]="airplane";COCO_CLASSES[6]="bus";COCO_CLASSES[7]="train";COCO_CLASSES[8]="truck";
COCO_CLASSES[9]="boat";COCO_CLASSES[10]="traffic light";COCO_CLASSES[11]="fire hydrant";
COCO_CLASSES[13]="stop sign";COCO_CLASSES[14]="parking meter";COCO_CLASSES[15]="bench";
COCO_CLASSES[16]="bird";COCO_CLASSES[17]="cat";COCO_CLASSES[18]="dog";COCO_CLASSES[19]="horse";
COCO_CLASSES[20]="sheep";COCO_CLASSES[21]="cow";COCO_CLASSES[22]="elephant";COCO_CLASSES[23]="bear";
COCO_CLASSES[24]="zebra";COCO_CLASSES[25]="giraffe";COCO_CLASSES[27]="backpack";
COCO_CLASSES[28]="umbrella";COCO_CLASSES[31]="handbag";COCO_CLASSES[32]="tie";
COCO_CLASSES[33]="suitcase";COCO_CLASSES[34]="frisbee";COCO_CLASSES[35]="skis";
COCO_CLASSES[36]="snowboard";COCO_CLASSES[37]="sports ball";COCO_CLASSES[38]="kite";
COCO_CLASSES[39]="baseball bat";COCO_CLASSES[40]="baseball glove";COCO_CLASSES[41]="skateboard";
COCO_CLASSES[42]="surfboard";COCO_CLASSES[43]="tennis racket";COCO_CLASSES[44]="bottle";
COCO_CLASSES[46]="wine glass";COCO_CLASSES[47]="cup";COCO_CLASSES[48]="fork";
COCO_CLASSES[49]="knife";COCO_CLASSES[50]="spoon";COCO_CLASSES[51]="bowl";
COCO_CLASSES[52]="banana";COCO_CLASSES[53]="apple";COCO_CLASSES[54]="sandwich";
COCO_CLASSES[55]="orange";COCO_CLASSES[56]="broccoli";COCO_CLASSES[57]="carrot";
COCO_CLASSES[58]="hot dog";COCO_CLASSES[59]="pizza";COCO_CLASSES[60]="donut";
COCO_CLASSES[61]="cake";COCO_CLASSES[62]="chair";COCO_CLASSES[63]="couch";
COCO_CLASSES[64]="potted plant";COCO_CLASSES[65]="bed";COCO_CLASSES[67]="dining table";
COCO_CLASSES[70]="toilet";COCO_CLASSES[72]="tv";COCO_CLASSES[73]="laptop";
COCO_CLASSES[74]="mouse";COCO_CLASSES[75]="remote";COCO_CLASSES[76]="keyboard";
COCO_CLASSES[77]="cell phone";COCO_CLASSES[78]="microwave";COCO_CLASSES[79]="oven";
COCO_CLASSES[80]="toaster";COCO_CLASSES[81]="sink";COCO_CLASSES[82]="refrigerator";
COCO_CLASSES[84]="book";COCO_CLASSES[85]="clock";COCO_CLASSES[86]="vase";
COCO_CLASSES[87]="scissors";COCO_CLASSES[88]="teddy bear";COCO_CLASSES[89]="hair drier";
COCO_CLASSES[90]="toothbrush";

const NUM_CLASSES = 91;

const COLORS = COCO_CLASSES.map((_, i) => {
  const h = (i * 37) % 360;
  return `hsl(${h}, 70%, 55%)`;
});

function hslToRgba(hslStr, alpha) {
  const tmp = document.createElement("canvas");
  tmp.width = tmp.height = 1;
  const c = tmp.getContext("2d");
  c.fillStyle = hslStr;
  c.fillRect(0, 0, 1, 1);
  const [r, g, b] = c.getImageData(0, 0, 1, 1).data;
  return [r, g, b, alpha];
}
const COLOR_RGBA = COLORS.map(c => hslToRgba(c, 100));

// YOLO26 uses contiguous 80-class COCO indexing (0-79), unlike RF-DETR's
// 91 category-id scheme above. Separate name + colour tables per family.
const YOLO_CLASSES = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
  "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
  "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
  "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
  "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
  "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
  "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
  "remote","keyboard","cell phone","microwave","oven","toaster","sink",
  "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
  "toothbrush",
];
const YOLO_COLORS = YOLO_CLASSES.map((_, i) => `hsl(${(i * 37) % 360}, 70%, 55%)`);

// --- RTMW whole-body pose (COCO-WholeBody 133) — V5 topology ---
// 0-16 body, 17-22 foot, 23-90 face, 91-111 L-hand, 112-132 R-hand.
const NUM_KPTS = 133;
const BODY_EDGES = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],   // shoulders + arms
  [5, 11], [6, 12], [11, 12],                // torso
  [11, 13], [13, 15], [12, 14], [14, 16],    // legs
  [0, 1], [0, 2], [1, 3], [2, 4],            // nose-eyes-ears
  [3, 5], [4, 6],                            // ears-shoulders
];
const FOOT_EDGES = [
  [15, 17], [15, 18], [15, 19],   // L ankle -> big toe / small toe / heel
  [16, 20], [16, 21], [16, 22],   // R ankle -> ...
];
// 21-pt hand: wrist=base, then 5 fingers × 4 joints (thumb,index,middle,ring,pinky).
function handEdges(base) {
  const e = [];
  for (const off of [1, 5, 9, 13, 17]) {
    e.push([base, base + off], [base + off, base + off + 1],
           [base + off + 1, base + off + 2], [base + off + 2, base + off + 3]);
  }
  return e;
}
// group -> kpt index range [lo,hi), distinct colour (V12), bone edges (face = dots only, V11).
const POSE_GROUPS = [
  { key: "body",  lo: 0,   hi: 23,  color: "#00e5ff", edges: BODY_EDGES.concat(FOOT_EDGES) },
  { key: "face",  lo: 23,  hi: 91,  color: "#ff5cf0", edges: [] },
  { key: "lhand", lo: 91,  hi: 112, color: "#ffa726", edges: handEdges(91) },
  { key: "rhand", lo: 112, hi: 133, color: "#ffd54f", edges: handEdges(112) },
];
// RTMW preprocessing (pipeline.json, to_rgb=True) — V3. RGB, 0-255 scale.
const POSE_MEAN = [123.675, 116.28, 103.53];
const POSE_STD = [58.395, 57.12, 57.375];
const POSE_PADDING = 1.25;   // pipeline.json TopDownGetBboxCenterScale
const Z_RANGE = 2.1744869;   // RTMW3D depth scale (rtmlib RTMPose3d default) — V16

// Active lookup tables — swapped on model load based on family.
let activeNames = COCO_CLASSES;
let activeColors = COLORS;

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// DOM
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const videoPane = document.getElementById("video-pane");
const placeholder = document.getElementById("video-placeholder");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const startBtn = document.getElementById("startBtn");
const startBtnLarge = document.getElementById("startBtnLarge");
const thresholdInput = document.getElementById("threshold");
const threshValEl = document.getElementById("threshVal");
const badgeEl = document.getElementById("backend-badge");
const modelSelect = document.getElementById("modelSelect");
const backendSelect = document.getElementById("backendSelect");
const fullscreenBtn = document.getElementById("fullscreenBtn");
const showSegmentsCheckbox = document.getElementById("showSegments");
const cutoutCheckbox = document.getElementById("cutoutMode");
const segmentPane = document.getElementById("segment-pane");
const segmentGrid = document.getElementById("segment-grid");
const progressBar = document.getElementById("progress-bar");
const progressFill = document.getElementById("progress-fill");
const maskOpacityInput = document.getElementById("maskOpacity");
const maskOpacityVal = document.getElementById("maskOpacityVal");
const outlineOpacityInput = document.getElementById("outlineOpacity");
const outlineOpacityValEl = document.getElementById("outlineOpacityVal");
const controlPanel = document.getElementById("control-panel");
const panelToggle = document.getElementById("panel-toggle");
const panelClose = document.getElementById("panelClose");
// Pose (RTMW) controls
const poseControls = document.getElementById("pose-controls");
const poseModeSelect = document.getElementById("poseMode");
const kptThreshInput = document.getElementById("kptThresh");
const kptThreshVal = document.getElementById("kptThreshVal");
const showBodyCb = document.getElementById("showBody");
const showHandsCb = document.getElementById("showHands");
const showFaceCb = document.getElementById("showFace");
const poseOverlayCb = document.getElementById("poseOverlay");
const precisionSelect = document.getElementById("posePrecision");
const detResSelect = document.getElementById("detRes");
const pose3dCanvas = document.getElementById("pose3d");
const pose3dControls = document.getElementById("pose3d-controls");
const autoRotateCb = document.getElementById("autoRotate");
const resetViewBtn = document.getElementById("resetView");
const depthScaleInput = document.getElementById("depthScale");
const depthScaleVal = document.getElementById("depthScaleVal");
let pose3dReady = false;   // three.js scene initialized lazily on first 3D model load

// State
let session = null;
let running = false;
let threshold = 0.5;
let activeBackend = null;
let hasSegmentation = false;
let family = "rfdetr";      // "rfdetr" | "yolo"
let inputName = "input";    // ONNX graph input name (rfdetr:"input", yolo:"images")
let inputSize = 384;
let bridge = null;          // python inference bridge (null = local ONNX)
let usingBridge = false;
let loading = false;
let manifest = [];
let currentModelId = null;
let frameCount = 0;
let showSegments = false;
let cutoutMode = false;
let maskOpacity = 0.4;
let outlineOpacity = 0.85;

// --- RTMW pose state ---
let poseW = 192, poseH = 256;            // model input W,H per variant (V1)
let poseInputBuf = null, poseCanvas = null, poseCtx = null;
let detSession = null, detBuf = null, detCanvas = null, detCtx = null;  // optional person detector
let detRes = 512;                        // detector input res (UI: 640/512/384/320) — 512 default
let posePrecision = "fp32";              // fp32 | fp16 (UI; fp16 ~halves download, ~8% faster on webgpu)
let poseMode = "all";                    // all | cap3 | cap2 | single | exclusive (debug) — V10
let detLoadFailed = false;               // detector load errored — stop retrying every frame
let kptThresh = 0.3;                      // per-keypoint score gate (V9)
let showBody = true, showHands = true, showFace = true, poseOverlayOn = true;  // V14
let depthScale = 1.5;                     // 3D viewer depth exaggeration (V16/V17)
const poseProf = {};                      // per-stage timing (ms), read via window.__poseProf
if (typeof window !== "undefined") window.__poseProf = poseProf;

let inputBuf = null;
let prepCanvas = null;
let prepCtx = null;


const segmentPool = [];
let activeSegments = 0;

let frameCanvas = null;
let frameCtx = null;

// Model blob cache — downloaded once, reused across backend switches
const modelCache = new Map();

function getSegmentEl(index) {
  if (index < segmentPool.length) return segmentPool[index];
  const el = document.createElement("div");
  el.className = "segment";
  const canvas = document.createElement("canvas");
  const label = document.createElement("div");
  label.className = "seg-label";
  el.appendChild(canvas);
  el.appendChild(label);
  segmentPool.push(el);
  return el;
}

// --- Validation ---

function validateOnnx(buf, url) {
  if (buf.byteLength < 16) {
    throw new Error(`Model file too small (${buf.byteLength} bytes) — likely missing: ${url}`);
  }
  // ONNX protobuf starts with field tag 0x08; HTML starts with 0x3C (<)
  const first = new Uint8Array(buf, 0, 4);
  if (first[0] === 0x3C) {
    throw new Error(
      `Model file not found at ${url} (got HTML instead of ONNX). ` +
      `Run export_onnx.py to generate models, or copy them to web/public/models/.`
    );
  }
}

// --- Progress bar ---

function showProgress(pct) {
  progressBar.classList.add("active");
  progressFill.style.width = pct + "%";
}

function hideProgress() {
  progressBar.classList.remove("active");
  progressFill.style.width = "0%";
}

async function fetchModelWithProgress(url) {
  if (modelCache.has(url)) return modelCache.get(url);

  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${url}`);

  const contentType = response.headers.get("Content-Type") || "";
  if (contentType.includes("text/html")) {
    throw new Error(
      `Model file not found at ${url} (server returned HTML). ` +
      `Run export_onnx.py on a machine with PyTorch to generate the .onnx files.`
    );
  }

  const contentLength = response.headers.get("Content-Length");
  const total = contentLength ? parseInt(contentLength, 10) : 0;

  if (!response.body || !total) {
    // No streaming support or unknown size — just download
    const buf = await response.arrayBuffer();
    validateOnnx(buf, url);
    modelCache.set(url, buf);
    return buf;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    showProgress((received / total) * 100);
  }

  const buf = new ArrayBuffer(received);
  const view = new Uint8Array(buf);
  let offset = 0;
  for (const chunk of chunks) {
    view.set(chunk, offset);
    offset += chunk.length;
  }

  validateOnnx(buf, url);
  modelCache.set(url, buf);
  return buf;
}

// --- Controls ---

thresholdInput.addEventListener("input", () => {
  threshold = parseFloat(thresholdInput.value);
  threshValEl.textContent = threshold.toFixed(2);
  if (bridge) bridge.setThreshold(threshold);
});

modelSelect.addEventListener("change", () => {
  loadModel(modelSelect.value);
});

backendSelect.addEventListener("change", () => {
  if (currentModelId) loadModel(currentModelId);
});

panelToggle.addEventListener("click", () => {
  controlPanel.classList.toggle("collapsed");
  panelToggle.style.display = controlPanel.classList.contains("collapsed") ? "" : "none";
});

panelClose.addEventListener("click", () => {
  controlPanel.classList.add("collapsed");
  panelToggle.style.display = "";
});

outlineOpacityInput.addEventListener("input", () => {
  outlineOpacity = parseFloat(outlineOpacityInput.value);
  outlineOpacityValEl.textContent = outlineOpacity.toFixed(2);
});

maskOpacityInput.addEventListener("input", () => {
  maskOpacity = parseFloat(maskOpacityInput.value);
  maskOpacityVal.textContent = maskOpacity.toFixed(2);
});

// --- Pose controls (V10 mode, V9 threshold, V14 per-group toggles) ---
poseModeSelect.addEventListener("change", () => {
  poseMode = poseModeSelect.value;
  if (poseMode !== "exclusive") ensureDetector();   // lazy-load yolo26n on first detector use
});
kptThreshInput.addEventListener("input", () => {
  kptThresh = parseFloat(kptThreshInput.value);
  kptThreshVal.textContent = kptThresh.toFixed(2);
});
showBodyCb.addEventListener("change", () => { showBody = showBodyCb.checked; });
showHandsCb.addEventListener("change", () => { showHands = showHandsCb.checked; });
showFaceCb.addEventListener("change", () => { showFace = showFaceCb.checked; });
poseOverlayCb.addEventListener("change", () => { poseOverlayOn = poseOverlayCb.checked; });
precisionSelect.addEventListener("change", () => {
  posePrecision = precisionSelect.value;
  if (currentModelId && family === "rtmw") loadModel(currentModelId);   // reload pose at new precision
});
detResSelect.addEventListener("change", () => {
  detRes = parseInt(detResSelect.value, 10);
  detSession = null; detLoadFailed = false;        // force detector reload at the new res
  if (poseMode !== "exclusive") ensureDetector();
});

// 3D viewer orbit controls
autoRotateCb.addEventListener("change", () => setAutoRotate(autoRotateCb.checked));
resetViewBtn.addEventListener("click", () => { resetView(); autoRotateCb.checked = true; });
depthScaleInput.addEventListener("input", () => {
  depthScale = parseFloat(depthScaleInput.value);
  depthScaleVal.textContent = depthScale.toFixed(1) + "×";
});

showSegmentsCheckbox.addEventListener("change", () => {
  showSegments = showSegmentsCheckbox.checked;
  segmentPane.classList.toggle("visible", showSegments);
  if (!showSegments) {
    for (let i = 0; i < activeSegments; i++) segmentPool[i].style.display = "none";
    activeSegments = 0;
  }
  deferredSyncOverlay();
});

cutoutCheckbox.addEventListener("change", () => {
  cutoutMode = cutoutCheckbox.checked;
  // Hide the live video so the transparent overlay shows person-on-black.
  video.style.opacity = cutoutMode ? "0" : "1";
  videoPane.style.background = cutoutMode ? "#000" : "";
  // Bridge: switch to person-union-mask-only payload (least back-and-forth).
  if (bridge) bridge.setMatte(cutoutMode);
});

fullscreenBtn.addEventListener("click", () => {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    videoPane.requestFullscreen();
  }
});

document.addEventListener("fullscreenchange", () => {
  videoPane.classList.toggle("fullscreen", !!document.fullscreenElement);
  deferredSyncOverlay();
});

// --- Model loading ---

async function loadModel(modelId) {
  if (loading) return;
  loading = true;
  modelSelect.disabled = true;
  startBtn.disabled = true;
  startBtnLarge.disabled = true;
  currentModelId = modelId;

  const meta = manifest.find(m => m.id === modelId);
  family = meta.family || "rfdetr";
  const is3d = family === "rtmw3d";
  const isPose = family === "rtmw" || is3d;
  if (isPose) { poseW = meta.resW; poseH = meta.resH; inputSize = meta.resW; }
  else { inputSize = meta.resolution; }
  inputName = family === "yolo" ? "images" : "input";
  activeNames = family === "yolo" ? YOLO_CLASSES : COCO_CLASSES;
  activeColors = family === "yolo" ? YOLO_COLORS : COLORS;
  // Pose controls only apply to the rtmw family.
  poseControls.style.display = isPose ? "" : "none";

  const preferredBackend = backendSelect.value;

  // RTMW runs in-browser only — no pose model on the python bridge server.
  if (isPose && preferredBackend.startsWith("python")) {
    statusEl.childNodes[0].textContent = "RTMW pose runs in-browser — pick WebGPU or WASM.";
    loading = false;
    modelSelect.disabled = false;
    return;
  }

  // --- Python bridge backend (native MPS / MLX over WebSocket) ---
  if (preferredBackend.startsWith("python")) {
    const ok = await loadBridgeModel(modelId, meta, preferredBackend);
    loading = false;
    modelSelect.disabled = false;
    if (ok) { startBtn.disabled = false; startBtnLarge.disabled = false; }
    return;
  }
  if (bridge) { bridge.close(); bridge = null; usingBridge = false; }

  // RTMW (2D) fp16 variant keeps fp32 IO, so decode/preprocess are unchanged.
  // rtmw3d has no fp16 variant — always fp32.
  const fileName = (family === "rtmw" && posePrecision === "fp16") ? "inference_model_fp16.onnx" : "inference_model.onnx";
  const modelUrl = `/models/${modelId}/${fileName}`;
  const backends = preferredBackend === "auto"
    ? ["webgpu", "wasm"]
    : [preferredBackend];

  session = null;
  activeBackend = null;

  // Download model with progress (cached after first download)
  let modelData;
  try {
    statusEl.childNodes[0].textContent = `Downloading ${meta.label}...`;
    modelData = await fetchModelWithProgress(modelUrl);
    hideProgress();
  } catch (e) {
    console.error("Model download error:", e);
    statusEl.childNodes[0].textContent = e.message.includes("not found")
      ? `Model not found — run export_onnx.py or copy .onnx files to web/public/models/`
      : `Download failed: ${e.message}`;
    hideProgress();
    loading = false;
    modelSelect.disabled = false;
    return;
  }

  for (const backend of backends) {
    try {
      statusEl.childNodes[0].textContent = `Initializing ${meta.label} (${backend})...`;
      session = await ort.InferenceSession.create(modelData, {
        executionProviders: [backend],
        graphOptimizationLevel: "all",
        enableCpuMemArena: true,
        enableMemPattern: true,
      });
      activeBackend = backend;
      break;
    } catch (e) {
      console.warn(`${backend} failed:`, e.message);
    }
  }

  if (!session) {
    statusEl.childNodes[0].textContent = "Failed to load model.";
    loading = false;
    modelSelect.disabled = false;
    return;
  }

  hasSegmentation = meta.task === "seg";

  if (isPose) {
    poseInputBuf = new Float32Array(3 * poseW * poseH);
    poseCanvas = new OffscreenCanvas(poseW, poseH);
    poseCtx = poseCanvas.getContext("2d", { willReadFrequently: true });
    // Reset detector so it (re)loads on the current EP; preload unless debug whole-frame.
    detSession = null;
    detLoadFailed = false;
    if (poseMode !== "exclusive") await ensureDetector();
    // 3D skeleton viewer — only for rtmw3d.
    if (is3d) {
      pose3dCanvas.style.display = "block";
      pose3dControls.style.display = "flex";
      if (!pose3dReady) {
        initPose3d(pose3dCanvas);
        // Manual grab takes over: reflect it in the auto-rotate checkbox.
        onPose3dInteract(() => { autoRotateCb.checked = false; setAutoRotate(false); });
        pose3dReady = true;
      }
      setPose3dRunning(true);
    } else {
      setPose3dRunning(false);
      pose3dCanvas.style.display = "none";
      pose3dControls.style.display = "none";
    }
  } else {
    inputBuf = new Float32Array(1 * 3 * inputSize * inputSize);
    prepCanvas = new OffscreenCanvas(inputSize, inputSize);
    prepCtx = prepCanvas.getContext("2d", { willReadFrequently: true });
  }

  const modelType = isPose ? "Pose" : hasSegmentation ? "Seg" : "Det";
  badgeEl.textContent = activeBackend.toUpperCase();
  badgeEl.className = activeBackend;
  const dims = isPose ? `${poseW}x${poseH}` : `${inputSize}px`;
  statusEl.childNodes[0].textContent = `${meta.label} (${dims}, ${modelType}) `;
  startBtn.disabled = false;
  startBtnLarge.disabled = false;
  modelSelect.disabled = false;
  loading = false;
}

async function loadBridgeModel(modelId, meta, backend) {
  if (bridge) { bridge.close(); bridge = null; }
  usingBridge = false;
  session = null;

  // python-mps serves the manifest id directly; python-mlx needs the -mlx zoo
  // variant, which only exists for YOLO26.
  let serverId = modelId;
  if (backend === "python-mlx") {
    if (!modelId.startsWith("yolo26")) {
      statusEl.childNodes[0].textContent = "MLX backend supports YOLO26 models only — pick a yolo26* model.";
      return false;
    }
    serverId = `${modelId}-mlx`;
  }

  try {
    statusEl.childNodes[0].textContent = `Connecting to Python bridge (${serverId})...`;
    bridge = createBridge();
    const m = await bridge.connect(serverId, threshold);
    if (cutoutMode) bridge.setMatte(true);
    family = m.family;
    activeNames = family === "yolo" ? YOLO_CLASSES : COCO_CLASSES;
    activeColors = family === "yolo" ? YOLO_COLORS : COLORS;
    hasSegmentation = m.task === "seg";
    usingBridge = true;
    activeBackend = backend;

    prepCanvas = new OffscreenCanvas(inputSize, inputSize);
    prepCtx = prepCanvas.getContext("2d", { willReadFrequently: true });

    badgeEl.textContent = backend === "python-mlx" ? "PY·MLX" : "PY·MPS";
    badgeEl.className = backend;
    const modelType = hasSegmentation ? "Seg" : "Det";
    statusEl.childNodes[0].textContent = `${meta.label} (${m.backend}/${m.device}, ${modelType}) `;
    return true;
  } catch (e) {
    console.error("bridge load error:", e);
    statusEl.childNodes[0].textContent = e.message;
    bridge = null;
    usingBridge = false;
    return false;
  }
}

async function loadManifest() {
  const res = await fetch("/models/manifest.json");
  manifest = await res.json();
  for (const m of manifest) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    modelSelect.appendChild(opt);
  }
  modelSelect.disabled = false;
  return manifest[0].id;
}

// --- Inference ---

function preprocess(videoEl) {
  prepCtx.drawImage(videoEl, 0, 0, inputSize, inputSize);
  const imgData = prepCtx.getImageData(0, 0, inputSize, inputSize).data;

  const hw = inputSize * inputSize;
  // RF-DETR expects ImageNet mean/std; YOLO26 expects plain 0-1. Both NCHW RGB.
  const yolo = family === "yolo";
  for (let i = 0; i < hw; i++) {
    const r = imgData[i * 4] / 255;
    const g = imgData[i * 4 + 1] / 255;
    const b = imgData[i * 4 + 2] / 255;
    if (yolo) {
      inputBuf[i] = r;
      inputBuf[hw + i] = g;
      inputBuf[2 * hw + i] = b;
    } else {
      inputBuf[i] = (r - MEAN[0]) / STD[0];
      inputBuf[hw + i] = (g - MEAN[1]) / STD[1];
      inputBuf[2 * hw + i] = (b - MEAN[2]) / STD[2];
    }
  }
  return new ort.Tensor("float32", inputBuf, [1, 3, inputSize, inputSize]);
}

function postprocess(dets, labels, vidW, vidH) {
  const nq = dets.length / 4;
  const results = [];

  for (let q = 0; q < nq; q++) {
    let bestScore = -Infinity;
    let bestClass = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      if (!COCO_CLASSES[c]) continue;
      const logit = labels[q * NUM_CLASSES + c];
      const score = 1 / (1 + Math.exp(-logit));
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }

    if (bestScore < threshold) continue;

    const cx = dets[q * 4];
    const cy = dets[q * 4 + 1];
    const w = dets[q * 4 + 2];
    const h = dets[q * 4 + 3];

    results.push({
      x1: (1 - (cx + w / 2)) * vidW,
      y1: (cy - h / 2) * vidH,
      x2: (1 - (cx - w / 2)) * vidW,
      y2: (cy + h / 2) * vidH,
      score: bestScore,
      classId: bestClass,
      queryIdx: q,
    });
  }

  return results;
}

// YOLO26 ONNX is NMS-free / end-to-end: output0 = [1, 300, 6] (det) or
// [1, 300, 38] (seg) where each row is [x1,y1,x2,y2,conf,cls, ...32 coeffs].
// Coords are pixels in the (stretched) inputSize square. We mirror x to match
// the CSS-flipped video, exactly like the RF-DETR path.
function postprocessYolo(data, stride, vidW, vidH) {
  const nrows = data.length / stride;
  const results = [];
  for (let q = 0; q < nrows; q++) {
    const o = q * stride;
    const score = data[o + 4];
    if (score < threshold) continue;  // rows are score-sorted; could break, but cheap

    const nx1 = data[o] / inputSize;
    const ny1 = data[o + 1] / inputSize;
    const nx2 = data[o + 2] / inputSize;
    const ny2 = data[o + 3] / inputSize;

    results.push({
      x1: (1 - nx2) * vidW,   // mirror: left/right swap
      y1: ny1 * vidH,
      x2: (1 - nx1) * vidW,
      y2: ny2 * vidH,
      score,
      classId: Math.round(data[o + 5]),
      queryIdx: results.length,  // index into the compacted mask buffer below
      row: q,
      // original (un-mirrored) normalized box — mask grid is mirrored at draw time
      _nx1: nx1, _ny1: ny1, _nx2: nx2, _ny2: ny2,
    });
  }
  return results;
}

// Build an RF-DETR-shaped mask buffer from YOLO seg outputs so the existing
// drawMasks / updateSidebar code works unchanged. masksData is [n, mh, mw]
// indexed by det.queryIdx; value > 0 means inside the mask.
//   output0 row tail (offset 6..37) = 32 mask coefficients
//   proto = output1 [1, 32, mh, mw];  mask = coeffs · proto, cropped to the box
function buildYoloMasks(detections, output0, stride, proto, protoDims) {
  const nMask = protoDims[1];        // 32
  const mh = protoDims[2];
  const mw = protoDims[3];
  const maskSize = mh * mw;
  const n = detections.length;
  const masksData = new Float32Array(n * maskSize);
  masksData.fill(-1);                // -1 => outside (drawMasks treats > 0 as in-mask)

  for (let j = 0; j < n; j++) {
    const det = detections[j];
    const coefBase = det.row * stride + 6;

    // Box in proto grid coords (proto spans the inputSize square).
    const bx1 = Math.max(0, Math.floor((det._nx1) * mw));
    const bx2 = Math.min(mw, Math.ceil((det._nx2) * mw));
    const by1 = Math.max(0, Math.floor((det._ny1) * mh));
    const by2 = Math.min(mh, Math.ceil((det._ny2) * mh));

    const out = j * maskSize;
    for (let y = by1; y < by2; y++) {
      for (let x = bx1; x < bx2; x++) {
        const p = y * mw + x;
        let acc = 0;
        for (let c = 0; c < nMask; c++) {
          acc += output0[coefBase + c] * proto[c * maskSize + p];
        }
        // sigmoid > 0.5  <=>  logit > 0; store logit, drawMasks checks > 0.
        masksData[out + p] = acc;
      }
    }
  }
  return { masksData, maskDims: [1, n, mh, mw] };
}

// --- Drawing ---

// --- Mask-to-polygon contour tracing (marching squares) ---

function traceContours(mask, w, h) {
  // Returns array of polygons, each polygon is array of [x, y] points
  // Uses simple contour tracing: scan for boundary pixels and follow edges
  const visited = new Uint8Array(w * h);
  const contours = [];

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x] || visited[y * w + x]) continue;
      // Check if boundary pixel (has at least one non-mask neighbor)
      const isBoundary =
        x === 0 || x === w - 1 || y === 0 || y === h - 1 ||
        !mask[(y - 1) * w + x] || !mask[(y + 1) * w + x] ||
        !mask[y * w + (x - 1)] || !mask[y * w + (x + 1)];
      if (!isBoundary) continue;

      // Trace boundary using Moore neighborhood
      const contour = [];
      const dirs = [[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]];
      let cx = x, cy = y, dir = 0;
      const startKey = cy * w + cx;
      let steps = 0;
      const maxSteps = w * h;

      do {
        contour.push([cx, cy]);
        visited[cy * w + cx] = 1;
        let found = false;

        // Search clockwise from (dir + 5) % 8 for next boundary pixel
        const startDir = (dir + 5) % 8;
        for (let i = 0; i < 8; i++) {
          const d = (startDir + i) % 8;
          const nx = cx + dirs[d][1];
          const ny = cy + dirs[d][0];
          if (nx >= 0 && nx < w && ny >= 0 && ny < h && mask[ny * w + nx]) {
            cx = nx;
            cy = ny;
            dir = d;
            found = true;
            break;
          }
        }
        if (!found) break;
        steps++;
      } while ((cy * w + cx !== startKey) && steps < maxSteps);

      if (contour.length >= 3) {
        contours.push(contour);
      }
    }
  }
  return contours;
}

// Douglas-Peucker polygon simplification
function simplifyPolygon(points, epsilon) {
  if (points.length <= 3) return points;

  let maxDist = 0, maxIdx = 0;
  const start = points[0], end = points[points.length - 1];
  const dx = end[0] - start[0], dy = end[1] - start[1];
  const lenSq = dx * dx + dy * dy;

  for (let i = 1; i < points.length - 1; i++) {
    let dist;
    if (lenSq === 0) {
      const ex = points[i][0] - start[0], ey = points[i][1] - start[1];
      dist = Math.sqrt(ex * ex + ey * ey);
    } else {
      const t = Math.max(0, Math.min(1, ((points[i][0] - start[0]) * dx + (points[i][1] - start[1]) * dy) / lenSq));
      const px = start[0] + t * dx - points[i][0];
      const py = start[1] + t * dy - points[i][1];
      dist = Math.sqrt(px * px + py * py);
    }
    if (dist > maxDist) { maxDist = dist; maxIdx = i; }
  }

  if (maxDist > epsilon) {
    const left = simplifyPolygon(points.slice(0, maxIdx + 1), epsilon);
    const right = simplifyPolygon(points.slice(maxIdx), epsilon);
    return left.slice(0, -1).concat(right);
  }
  return [start, end];
}

function drawMasks(detections, masksData, maskDims, vidW, vidH) {
  const maskH = maskDims[2];
  const maskW = maskDims[3];
  const maskSize = maskH * maskW;

  if (maskOpacity === 0 && outlineOpacity === 0) return;

  // Scale factors from mask coords to video coords (mirrored)
  const sx = vidW / maskW;
  const sy = vidH / maskH;

  // Build a binary mask per detection and extract contours
  const binaryMask = new Uint8Array(maskSize);

  for (const det of detections) {
    const offset = det.queryIdx * maskSize;

    // Build binary mask for this detection
    binaryMask.fill(0);
    for (let i = 0; i < maskSize; i++) {
      if (masksData[offset + i] > 0) binaryMask[i] = 1;
    }

    // Extract contours and simplify
    const contours = traceContours(binaryMask, maskW, maskH);

    for (const contour of contours) {
      // Simplify polygon (epsilon=1 in mask space)
      const simplified = simplifyPolygon(contour, 1.0);
      if (simplified.length < 3) continue;

      // Build canvas path — mirror x coords to match CSS-flipped video
      ctx.beginPath();
      const x0 = (maskW - 1 - simplified[0][0]) * sx;
      const y0 = simplified[0][1] * sy;
      ctx.moveTo(x0, y0);
      for (let i = 1; i < simplified.length; i++) {
        const px = (maskW - 1 - simplified[i][0]) * sx;
        const py = simplified[i][1] * sy;
        ctx.lineTo(px, py);
      }
      ctx.closePath();

      // Fill
      if (maskOpacity > 0) {
        ctx.globalAlpha = maskOpacity;
        ctx.fillStyle = activeColors[det.classId];
        ctx.fill();
      }

      // Outline
      if (outlineOpacity > 0) {
        ctx.globalAlpha = outlineOpacity;
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.stroke();
      }
    }
  }

  ctx.globalAlpha = 1;
}

// Person-only background removal. Builds a person-union alpha at proto res
// (cheap), then composites the INFERENCE-TIME frame snapshot × mask on the GPU
// (drawImage + destination-in). Compositing the snapshot — not the live video —
// keeps the mask aligned with the pixels it was computed from (the video has
// advanced ~inference-latency ms by the time the mask arrives).
let cutMask = null, cutMaskCtx = null;
let snapCanvas = null, snapCtx = null;

function snapshotFrame(w, h) {
  if (!snapCanvas || snapCanvas.width !== w || snapCanvas.height !== h) {
    snapCanvas = new OffscreenCanvas(w, h);
    snapCtx = snapCanvas.getContext("2d");
  }
  snapCtx.drawImage(video, 0, 0, w, h);   // un-mirrored frame at inference time
}

function drawCutout(detections, masksData, maskDims, w, h) {
  ctx.clearRect(0, 0, w, h);
  if (!masksData || !maskDims) return;
  const mh = maskDims[2], mw = maskDims[3], maskSize = mh * mw;
  const personId = family === "yolo" ? 0 : 1;   // YOLO 0 / RF-DETR COCO cat-id 1

  if (!cutMask || cutMask.width !== mw || cutMask.height !== mh) {
    cutMask = new OffscreenCanvas(mw, mh);
    cutMaskCtx = cutMask.getContext("2d");
  }
  const img = cutMaskCtx.createImageData(mw, mh);
  const a = img.data;
  let any = false;
  for (const det of detections) {
    if (det.classId !== personId) continue;
    const off = det.queryIdx * maskSize;
    for (let i = 0; i < maskSize; i++) {
      if (masksData[off + i] > 0) { a[i * 4 + 3] = 255; any = true; }
    }
  }
  if (!any) return;
  cutMaskCtx.putImageData(img, 0, 0);

  // snapshot drawn mirrored (CSS-flipped feed); mask is un-mirrored model space,
  // so mirror it too when used as the alpha.
  ctx.save();
  ctx.translate(w, 0); ctx.scale(-1, 1);
  ctx.drawImage(snapCanvas, 0, 0, w, h);   // the frame inference saw, not live video
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(cutMask, 0, 0, w, h);      // GPU upscale 160² -> w×h
  ctx.restore();
  ctx.globalCompositeOperation = "source-over";
}

function drawDetections(detections, w, h, masksData, maskDims) {
  ctx.clearRect(0, 0, w, h);

  if (masksData && maskDims) {
    drawMasks(detections, masksData, maskDims, w, h);
  }

  for (const det of detections) {
    const color = activeColors[det.classId];
    const label = `${activeNames[det.classId]} ${det.score.toFixed(2)}`;
    const bx = det.x1;
    const by = det.y1;
    const bw = det.x2 - det.x1;
    const bh = det.y2 - det.y1;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(bx, by, bw, bh);

    ctx.font = "bold 14px system-ui";
    const textW = ctx.measureText(label).width + 8;
    ctx.fillStyle = color;
    ctx.fillRect(bx, by - 20, textW, 20);

    ctx.fillStyle = "#000";
    ctx.fillText(label, bx + 4, by - 5);
  }
}

// --- RTMW pose pipeline (top-down) ---

// Lazy-load yolo26n as the optional person detector (V10 detector modes, C4).
// Reuses the existing ONNX/EP path (V7) — same backend the pose model loaded on.
async function ensureDetector() {
  if (detSession) return true;
  if (detLoadFailed) return false;   // already errored — don't refetch every frame
  try {
    // 640 is the canonical export; lower res served as inference_model_<res>.onnx.
    const detUrl = `/models/yolo26n/inference_model${detRes === 640 ? "" : "_" + detRes}.onnx`;
    const buf = await fetchModelWithProgress(detUrl);
    hideProgress();
    detSession = await ort.InferenceSession.create(buf, {
      executionProviders: [activeBackend || "wasm"],
      graphOptimizationLevel: "all",
      enableCpuMemArena: true,
      enableMemPattern: true,
    });
    detBuf = new Float32Array(3 * detRes * detRes);
    detCanvas = new OffscreenCanvas(detRes, detRes);
    detCtx = detCanvas.getContext("2d", { willReadFrequently: true });
    return true;
  } catch (e) {
    console.error("detector load failed:", e);
    detLoadFailed = true;
    statusEl.childNodes[0].textContent = `Person detector load failed: ${e.message}`;
    return false;
  }
}

// Run yolo26n, return person boxes in un-mirrored source px, score-sorted desc.
async function detPersonBoxes(vidW, vidH) {
  const R = detRes;
  const a = performance.now();
  detCtx.drawImage(video, 0, 0, R, R);
  const d = detCtx.getImageData(0, 0, R, R).data;
  const hw = R * R;
  for (let i = 0; i < hw; i++) {
    detBuf[i] = d[i * 4] / 255;
    detBuf[hw + i] = d[i * 4 + 1] / 255;
    detBuf[2 * hw + i] = d[i * 4 + 2] / 255;
  }
  const t = new ort.Tensor("float32", detBuf, [1, 3, R, R]);
  const b = performance.now();
  const out = await detSession.run({ images: t });
  t.dispose();
  const c = performance.now();
  const o0 = out.output0;
  const data = o0.data;
  const stride = o0.dims[2];
  const boxes = [];
  for (let q = 0; q < data.length / stride; q++) {
    const o = q * stride;
    const score = data[o + 4];
    if (score < threshold) break;                   // rows score-sorted desc → done
    if (Math.round(data[o + 5]) !== 0) continue;    // YOLO person = class 0
    const x1 = (data[o] / R) * vidW;
    const y1 = (data[o + 1] / R) * vidH;
    const x2 = (data[o + 2] / R) * vidW;
    const y2 = (data[o + 3] / R) * vidH;
    boxes.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1, score });
  }
  for (const k in out) out[k].dispose();
  boxes.sort((a, b) => b.score - a.score);
  poseProf.detPre = b - a; poseProf.detInf = c - b; poseProf.detPost = performance.now() - c;
  return boxes;
}

// V2: bbox affine warp. Pad box to model aspect + padding 1.25, crop source into
// the poseW×poseH input via an axis-aligned drawImage (RTMPose rotation=0, so the
// affine is just scale+translate → invertible by the same rect). Normalize (V3).
function posePreprocess(box) {
  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;
  let w = box.w, h = box.h;
  const aspect = poseW / poseH;
  if (w > h * aspect) h = w / aspect; else w = h * aspect;
  w *= POSE_PADDING; h *= POSE_PADDING;
  const rect = { sx: cx - w / 2, sy: cy - h / 2, sw: w, sh: h };

  // One op: crop the person box straight from the video AND scale to RTMW's exact
  // input (poseW×poseH) — no full-frame copy, no second resize, tensor dims == model
  // input (V1). drawImage ignores the CSS mirror, so this is un-mirrored model space (V6).
  poseCtx.clearRect(0, 0, poseW, poseH);
  poseCtx.drawImage(video, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, poseW, poseH);
  const d = poseCtx.getImageData(0, 0, poseW, poseH).data;
  const hw = poseW * poseH;
  for (let i = 0; i < hw; i++) {
    poseInputBuf[i] = (d[i * 4] - POSE_MEAN[0]) / POSE_STD[0];
    poseInputBuf[hw + i] = (d[i * 4 + 1] - POSE_MEAN[1]) / POSE_STD[1];
    poseInputBuf[2 * hw + i] = (d[i * 4 + 2] - POSE_MEAN[2]) / POSE_STD[2];
  }
  const tensor = new ort.Tensor("float32", poseInputBuf, [1, 3, poseH, poseW]);
  return { tensor, rect };
}

// V4: SimCC decode. argmax per axis / split_ratio → model-input px, then invert
// the crop rect → source-frame coords (V6). score = mean of the two axis maxima.
function decodeSimcc(out, rect) {
  const sx = out.simcc_x.data, sy = out.simcc_y.data;
  const K = out.simcc_x.dims[1];
  const Wx = out.simcc_x.dims[2];
  const Hy = out.simcc_y.dims[2];
  const rX = Wx / poseW, rY = Hy / poseH;   // split_ratio ≈ 2
  const kpts = new Array(K);
  for (let k = 0; k < K; k++) {
    let bx = 0, bvx = -Infinity;
    const ox = k * Wx;
    for (let i = 0; i < Wx; i++) { const v = sx[ox + i]; if (v > bvx) { bvx = v; bx = i; } }
    let by = 0, bvy = -Infinity;
    const oy = k * Hy;
    for (let i = 0; i < Hy; i++) { const v = sy[oy + i]; if (v > bvy) { bvy = v; by = i; } }
    const mx = bx / rX, my = by / rY;       // model-input px
    kpts[k] = {
      x: rect.sx + (mx / poseW) * rect.sw,
      y: rect.sy + (my / poseH) * rect.sh,
      score: 0.5 * (bvx + bvy),
    };
  }
  return kpts;
}

// RTMW3D: 3-axis SimCC (V15/V16). output=X[576], "1554"=Y[768], "1556"=Z[576].
// Returns 2D keypoints (source-frame px, for the overlay) + 3D (model px incl depth).
function decode3d(out, rect) {
  const sx = out.output.data, sy = out["1554"].data, sz = out["1556"].data;
  const K = out.output.dims[1];
  const Wx = out.output.dims[2], Hy = out["1554"].dims[2], Wz = out["1556"].dims[2];
  const rX = Wx / poseW, rY = Hy / poseH, rZ = Wz / poseW;   // split_ratio ≈ 2
  const k2d = new Array(K), k3d = new Array(K);
  for (let k = 0; k < K; k++) {
    let bx = 0, vx = -1e9; const ox = k * Wx; for (let i = 0; i < Wx; i++) { const v = sx[ox + i]; if (v > vx) { vx = v; bx = i; } }
    let by = 0, vy = -1e9; const oy = k * Hy; for (let i = 0; i < Hy; i++) { const v = sy[oy + i]; if (v > vy) { vy = v; by = i; } }
    let bz = 0; const oz = k * Wz; let vz = -1e9; for (let i = 0; i < Wz; i++) { const v = sz[oz + i]; if (v > vz) { vz = v; bz = i; } }
    const mx = bx / rX, my = by / rY, mz = bz / rZ;          // model-input px (x,y) + depth px (z, 0..poseW)
    const score = 0.5 * (vx + vy);                            // z maxima low; gate on x,y only
    k2d[k] = { x: rect.sx + (mx / poseW) * rect.sw, y: rect.sy + (my / poseH) * rect.sh, score };
    // 3D in a consistent normalized space (V16): x,y ÷ (poseH/2) keep aspect; z gets the
    // RTMPose3d transform (z_px → root-relative depth × z_range), else depth looks flat.
    k3d[k] = {
      x: mx / (poseH / 2),
      y: my / (poseH / 2),
      z: (mz / (poseW / 2) - 1) * Z_RANGE,
      score,
    };
  }
  return { k2d, k3d };
}

function groupOn(key) {
  if (key === "body") return showBody;
  if (key === "face") return showFace;
  return showHands;   // lhand / rhand
}

// V11/V12/V13: joints (circles) + bones (lines), per-group colour, size scaled to
// the person box. Mirror x to match the CSS-flipped video (V6).
function drawPose(persons, vidW, vidH) {
  ctx.clearRect(0, 0, vidW, vidH);
  if (!poseOverlayOn) return;
  for (const p of persons) {
    const kpts = p.kpts;
    const r = Math.max(2, Math.min(8, Math.round((p.box.h || vidH) / 110)));
    const lw = Math.max(1.5, r * 0.7);
    for (const g of POSE_GROUPS) {
      if (!groupOn(g.key)) continue;
      ctx.strokeStyle = g.color;
      ctx.lineWidth = lw;
      ctx.lineCap = "round";
      ctx.beginPath();
      for (const [a, b] of g.edges) {
        const ka = kpts[a], kb = kpts[b];
        if (!ka || !kb || ka.score < kptThresh || kb.score < kptThresh) continue;
        ctx.moveTo(vidW - ka.x, ka.y);
        ctx.lineTo(vidW - kb.x, kb.y);
      }
      ctx.stroke();
      ctx.fillStyle = g.color;
      const rr = g.key === "face" ? Math.max(1.2, r * 0.45) : r;
      for (let i = g.lo; i < g.hi; i++) {
        const k = kpts[i];
        if (!k || k.score < kptThresh) continue;
        ctx.beginPath();
        ctx.arc(vidW - k.x, k.y, rr, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}

// --- Sidebar segments ---

function updateSidebar(detections, masksData, maskDims, vidW, vidH) {
  if (!masksData || !maskDims) {
    for (let i = 0; i < activeSegments; i++) segmentPool[i].style.display = "none";
    activeSegments = 0;
    return;
  }

  const maskH = maskDims[2];
  const maskW = maskDims[3];
  const maskSize = maskH * maskW;

  const withArea = detections.map(det => {
    let area = 0;
    const offset = det.queryIdx * maskSize;
    for (let i = 0; i < maskSize; i++) {
      if (masksData[offset + i] > 0) area++;
    }
    return { ...det, maskArea: area };
  });

  withArea.sort((a, b) => b.maskArea - a.maskArea || b.score - a.score);

  if (!frameCanvas || frameCanvas.width !== vidW || frameCanvas.height !== vidH) {
    frameCanvas = new OffscreenCanvas(vidW, vidH);
    frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });
  }
  frameCtx.save();
  frameCtx.translate(vidW, 0);
  frameCtx.scale(-1, 1);
  frameCtx.drawImage(video, 0, 0, vidW, vidH);
  frameCtx.restore();

  const count = withArea.length;

  for (let i = 0; i < count; i++) {
    const det = withArea[i];
    const el = getSegmentEl(i);

    const bx = Math.max(0, Math.floor(det.x1));
    const by = Math.max(0, Math.floor(det.y1));
    const bw = Math.min(vidW - bx, Math.ceil(det.x2 - det.x1));
    const bh = Math.min(vidH - by, Math.ceil(det.y2 - det.y1));

    if (bw <= 0 || bh <= 0) {
      el.style.display = "none";
      continue;
    }

    const canvas = el.children[0];
    const label = el.children[1];

    canvas.width = bw;
    canvas.height = bh;
    const cctx = canvas.getContext("2d", { willReadFrequently: true });

    cctx.drawImage(frameCanvas, bx, by, bw, bh, 0, 0, bw, bh);

    const imgData = cctx.getImageData(0, 0, bw, bh);
    const px = imgData.data;

    for (let py = 0; py < bh; py++) {
      const vidY = by + py;
      const my = Math.min(Math.floor((vidY / vidH) * maskH), maskH - 1);

      for (let pxx = 0; pxx < bw; pxx++) {
        const vidX = bx + pxx;
        const origX = vidW - 1 - vidX;
        const mx = Math.min(Math.floor((origX / vidW) * maskW), maskW - 1);
        const maskIdx = my * maskW + mx;

        const val = masksData[det.queryIdx * maskSize + maskIdx];
        let occluded = false;
        if (val > 0) {
          for (let j = 0; j < count; j++) {
            if (j === i) continue;
            const other = withArea[j];
            if (masksData[other.queryIdx * maskSize + maskIdx] > 0 && other.score > det.score) {
              occluded = true;
              break;
            }
          }
        }

        if (val <= 0 || occluded) {
          const idx = (py * bw + pxx) * 4;
          px[idx + 3] = 0;
        }
      }
    }

    cctx.putImageData(imgData, 0, 0);

    const areaPercent = ((det.maskArea / maskSize) * 100).toFixed(1);
    label.textContent = `${activeNames[det.classId]} ${det.score.toFixed(2)} (${areaPercent}%)`;
    label.style.color = activeColors[det.classId];

    el.style.display = "";
    if (!el.parentNode) segmentGrid.appendChild(el);
  }

  for (let i = count; i < activeSegments; i++) segmentPool[i].style.display = "none";
  activeSegments = count;
}

// --- Overlay positioning ---
// Canvas has the same CSS sizing (inset:0, 100%x100%, object-fit:contain) as the video,
// so the browser handles positioning identically. We just set the pixel resolution.
function syncOverlay() {
  const nativeW = video.videoWidth;
  const nativeH = video.videoHeight;
  if (!nativeW || !nativeH) return;
  overlay.width = nativeW;
  overlay.height = nativeH;
}

let syncPending = false;
function deferredSyncOverlay() {
  if (!running || syncPending) return;
  syncPending = true;
  requestAnimationFrame(() => {
    syncOverlay();
    syncPending = false;
  });
}

const resizeObserver = new ResizeObserver(deferredSyncOverlay);
resizeObserver.observe(videoPane);
window.addEventListener("resize", deferredSyncOverlay);

// --- Main loop ---

async function detectLoop() {
  if (!running) return;

  const t0 = performance.now();

  const vidW = video.videoWidth;
  const vidH = video.videoHeight;
  if (overlay.width !== vidW || overlay.height !== vidH) {
    syncOverlay();
  }

  // Snapshot the frame inference is about to consume, so the cutout composites
  // the mask against the matching pixels (not the video advanced during await).
  if (cutoutMode) snapshotFrame(vidW, vidH);

  let detections, masksData = null, maskDims = null;
  let tPreprocess, tInference, tPostprocess;
  let bridgeServer = null, bridgeSpeed = null;
  let posePersons = null;

  const isPoseFam = family === "rtmw" || family === "rtmw3d";

  if (isPoseFam) {
    // Top-down: pick person box(es), crop each straight from video at the model's
    // exact input res, run RTMW, decode keypoints. Exclusive = whole frame (V10).
    let boxes;
    if (poseMode === "exclusive") {
      boxes = [{ x: 0, y: 0, w: vidW, h: vidH }];   // DEBUG only: whole frame, no detector
    } else if (await ensureDetector()) {
      boxes = await detPersonBoxes(vidW, vidH);     // top-down: tight person crops (default)
      if (poseMode === "single") boxes = boxes.slice(0, 1);
      else if (poseMode.startsWith("cap")) boxes = boxes.slice(0, parseInt(poseMode.slice(3), 10));
    } else {
      boxes = [];   // detector required but failed to load — error shown, draw nothing
    }
    tPreprocess = performance.now();

    posePersons = [];
    let pPre = 0, pInf = 0, pPost = 0;
    for (const box of boxes) {
      const a = performance.now();
      const { tensor, rect } = posePreprocess(box);
      const b = performance.now();
      const out = await session.run({ input: tensor });
      tensor.dispose();
      const c = performance.now();
      let kpts, kpts3d = null;
      if (family === "rtmw3d") { const d = decode3d(out, rect); kpts = d.k2d; kpts3d = d.k3d; }
      else kpts = decodeSimcc(out, rect);
      posePersons.push({ kpts, kpts3d, box });
      for (const k in out) out[k].dispose();
      pPre += b - a; pInf += c - b; pPost += performance.now() - c;
    }
    poseProf.posePre = pPre; poseProf.poseInf = pInf; poseProf.posePost = pPost;
    poseProf.nppl = boxes.length;
    tInference = performance.now();
    tPostprocess = tInference;
    detections = posePersons;
  } else if (usingBridge) {
    // Native Python inference over the WebSocket. Send the frame, await decoded
    // dets + masks (same shape as the ONNX path), draw with the shared code.
    prepCtx.drawImage(video, 0, 0, inputSize, inputSize);
    const imageData = prepCtx.getImageData(0, 0, inputSize, inputSize);
    tPreprocess = performance.now();
    const buf = await bridge.infer(imageData);
    tInference = performance.now();
    if (!buf) {                       // disconnected / no result
      statusEl.childNodes[0].textContent = "Python bridge disconnected.";
      running = false;
      return;
    }
    const r = decodeBridgeResult(buf, vidW, vidH);
    if (r.matte) {
      // One person-union mask. Wrap as a single synthetic person detection so
      // drawCutout works unchanged (it filters person + unions).
      detections = [{ classId: family === "yolo" ? 0 : 1, queryIdx: 0, score: 1 }];
      masksData = r.alpha;
      maskDims = [1, 1, r.mh, r.mw];
    } else {
      detections = r.detections;
      masksData = r.masksData;
      maskDims = r.maskDims;
    }
    bridgeServer = r.server;          // {parse, predict, encode}
    bridgeSpeed = r.speed;            // {preprocess, inference, postprocess} (torch yolo)
    tPostprocess = performance.now();
  } else {
    const inputTensor = preprocess(video);
    tPreprocess = performance.now();
    const output = await session.run({ [inputName]: inputTensor });
    tInference = performance.now();
    inputTensor.dispose();

    if (family === "yolo") {
      const out0 = output.output0;
      const stride = out0.dims[2];             // 6 (det) or 38 (seg)
      detections = postprocessYolo(out0.data, stride, vidW, vidH);
      if (hasSegmentation && detections.length) {
        const proto = output.output1;
        const built = buildYoloMasks(detections, out0.data, stride, proto.data, proto.dims);
        masksData = built.masksData;
        maskDims = built.maskDims;
      }
      for (const k in output) output[k].dispose();
    } else {
      detections = postprocess(output.dets.data, output.labels.data, vidW, vidH);
      masksData = hasSegmentation ? output.masks.data : null;
      maskDims = hasSegmentation ? output.masks.dims : null;
      output.dets.dispose();
      output.labels.dispose();
      if (output.masks) output.masks.dispose();
    }
    tPostprocess = performance.now();
  }

  if (isPoseFam) {
    drawPose(posePersons, vidW, vidH);
    if (family === "rtmw3d") {
      const groups = POSE_GROUPS.map(g => ({ lo: g.lo, hi: g.hi, color: g.color, edges: g.edges, enabled: groupOn(g.key) }));
      updatePose3d(posePersons, { groups, kptThresh, depthScale });
    }
  } else if (cutoutMode) {
    drawCutout(detections, masksData, maskDims, vidW, vidH);
  } else {
    drawDetections(detections, vidW, vidH, masksData, maskDims);
  }
  const tDraw = performance.now();

  let tSidebar = tDraw;
  frameCount++;
  if (showSegments && !cutoutMode && !isPoseFam && frameCount % 3 === 0) {
    updateSidebar(detections, masksData, maskDims, vidW, vidH);
    tSidebar = performance.now();
  }

  const mode = isPoseFam ? (family === "rtmw3d" ? "pose3d" : "pose") : hasSegmentation ? "seg" : "det";
  // Rolling median per stage (n=60) — same as the python cam HUD, so the two
  // are directly comparable instead of reading jittery instantaneous values.
  const stat = {
    pre: tPreprocess - t0,            // browser: getImageData / preprocess
    inf: tInference - tPreprocess,    // bridge: full round-trip; onnx: session.run
    post: tPostprocess - tInference,  // browser decode
    draw: tDraw - tPostprocess,
    total: tSidebar - t0,
  };
  if (usingBridge && bridgeServer) {
    // Split the round-trip to expose overhead-over-raw-inference.
    const sParse = bridgeServer.parse || 0;
    const sPredict = bridgeServer.predict || 0;   // full predict wall (incl sv mask conv)
    const sEncode = bridgeServer.encode || 0;     // serialize + mask resize
    const sModel = bridgeSpeed ? bridgeSpeed.inference : sPredict;  // pure model fwd
    stat.srv = sParse + sPredict + sEncode;        // total server time
    stat.wire = (tInference - tPreprocess) - stat.srv;  // websocket + tunnel
    stat.model = sModel;
    stat.sParse = sParse;
    stat.sEnc = sEncode + (bridgeServer.maskresize || 0);
    stat.sConv = sPredict - sModel;                // sv mask materialization (predict - pure)
  }
  pushStat(stat);
  const m = medianStats();

  const base = [
    `pre:${m.pre.toFixed(1)}`,
    `inf:${m.inf.toFixed(1)} (${(1000 / m.inf).toFixed(0)}fps)`,
    `post:${m.post.toFixed(1)}`,
    `draw:${m.draw.toFixed(1)}`,
    `total:${m.total.toFixed(1)}ms (${(1000 / m.total).toFixed(0)}fps)`,
    `${detections.length}${isPoseFam ? "ppl" : "obj"} ${mode} ${isPoseFam ? `${poseW}x${poseH}` : inputSize + "px"} ${family}`,
  ];
  if (usingBridge && bridgeServer) {
    base.push(
      `‖ rtt:${m.inf.toFixed(1)} = model:${m.model.toFixed(1)} + conv:${m.sConv.toFixed(1)} + ` +
      `enc:${m.sEnc.toFixed(1)} + parse:${m.sParse.toFixed(1)} + wire:${m.wire.toFixed(1)}`
    );
  }
  statsEl.textContent = base.join(" | ");

  requestAnimationFrame(detectLoop);
}

// --- Rolling-median stats (matches python cam-detect HUD) ---
let statHist = [];
let statKey = null;

function pushStat(s) {
  const key = `${currentModelId}|${activeBackend}`;
  if (statKey !== key) {  // reset window on model OR backend switch
    statHist = [];
    statKey = key;
  }
  statHist.push(s);
  if (statHist.length > 60) statHist.shift();
}

function medianStats() {
  const out = {};
  if (!statHist.length) return out;
  for (const k of Object.keys(statHist[statHist.length - 1])) {
    const vals = statHist.map(s => s[k]).filter(v => v != null).sort((a, b) => a - b);
    out[k] = vals.length ? vals[vals.length >> 1] : 0;
  }
  return out;
}

// --- Camera start/stop ---

async function startCamera() {
  if (running) {
    running = false;
    startBtn.textContent = "Start Camera";
    startBtnLarge.textContent = "Start Camera";
    video.srcObject?.getTracks().forEach(t => t.stop());
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    statsEl.textContent = "";
    placeholder.classList.remove("hidden");
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  video.srcObject = stream;
  await video.play();

  placeholder.classList.add("hidden");
  // Wait a frame for layout to settle before measuring video rect
  await new Promise(r => requestAnimationFrame(r));
  syncOverlay();

  running = true;
  startBtn.textContent = "Stop";
  startBtnLarge.textContent = "Stop";
  detectLoop();
}

startBtn.addEventListener("click", startCamera);
startBtnLarge.addEventListener("click", startCamera);

// --- Init ---

// Panel starts open, hide the gear toggle
panelToggle.style.display = "none";

loadManifest().then(defaultModel => loadModel(defaultModel));
