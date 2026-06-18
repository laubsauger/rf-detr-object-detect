import * as ort from "onnxruntime-web";
import { createBridge, decodeBridgeResult } from "./bridge.js";

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
  inputSize = meta.resolution;
  family = meta.family || "rfdetr";
  inputName = family === "yolo" ? "images" : "input";
  activeNames = family === "yolo" ? YOLO_CLASSES : COCO_CLASSES;
  activeColors = family === "yolo" ? YOLO_COLORS : COLORS;

  const preferredBackend = backendSelect.value;

  // --- Python bridge backend (native MPS / MLX over WebSocket) ---
  if (preferredBackend.startsWith("python")) {
    const ok = await loadBridgeModel(modelId, meta, preferredBackend);
    loading = false;
    modelSelect.disabled = false;
    if (ok) { startBtn.disabled = false; startBtnLarge.disabled = false; }
    return;
  }
  if (bridge) { bridge.close(); bridge = null; usingBridge = false; }

  const modelUrl = `/models/${modelId}/inference_model.onnx`;
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

  inputBuf = new Float32Array(1 * 3 * inputSize * inputSize);
  prepCanvas = new OffscreenCanvas(inputSize, inputSize);
  prepCtx = prepCanvas.getContext("2d", { willReadFrequently: true });

  const modelType = hasSegmentation ? "Seg" : "Det";
  badgeEl.textContent = activeBackend.toUpperCase();
  badgeEl.className = activeBackend;
  statusEl.childNodes[0].textContent = `${meta.label} (${inputSize}px, ${modelType}) `;
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
// (cheap), then composites video × mask on the GPU (drawImage + destination-in)
// — no full-res per-pixel JS, no boxes/contours/sidebar.
let cutMask = null, cutMaskCtx = null;
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

  // video drawn mirrored (CSS-flipped feed); mask is un-mirrored model space, so
  // mirror it too when used as the alpha.
  ctx.save();
  ctx.translate(w, 0); ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(cutMask, 0, 0, w, h);   // GPU upscale 160² -> w×h
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

  let detections, masksData = null, maskDims = null;
  let tPreprocess, tInference, tPostprocess;
  let bridgeServer = null, bridgeSpeed = null;

  if (usingBridge) {
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

  if (cutoutMode) {
    drawCutout(detections, masksData, maskDims, vidW, vidH);
  } else {
    drawDetections(detections, vidW, vidH, masksData, maskDims);
  }
  const tDraw = performance.now();

  let tSidebar = tDraw;
  frameCount++;
  if (showSegments && !cutoutMode && frameCount % 3 === 0) {
    updateSidebar(detections, masksData, maskDims, vidW, vidH);
    tSidebar = performance.now();
  }

  const mode = hasSegmentation ? "seg" : "det";
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
    `${detections.length}obj ${mode} ${inputSize}px ${family}`,
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
