import * as ort from "onnxruntime-web";

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
let inputSize = 384;
let loading = false;
let manifest = [];
let currentModelId = null;
let frameCount = 0;
let showSegments = false;
let maskOpacity = 0.4;
let outlineOpacity = 0.85;

let inputBuf = null;
let prepCanvas = null;
let prepCtx = null;

let maskCanvas = null;
let maskCtx = null;
let maskImgData = null;

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
  const contentLength = response.headers.get("Content-Length");
  const total = contentLength ? parseInt(contentLength, 10) : 0;

  if (!response.body || !total) {
    // No streaming support or unknown size — just download
    const buf = await response.arrayBuffer();
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

  modelCache.set(url, buf);
  return buf;
}

// --- Controls ---

thresholdInput.addEventListener("input", () => {
  threshold = parseFloat(thresholdInput.value);
  threshValEl.textContent = threshold.toFixed(2);
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

  const modelUrl = `/models/${modelId}/inference_model.onnx`;
  const preferredBackend = backendSelect.value;
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
    statusEl.childNodes[0].textContent = `Download failed: ${e.message}`;
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

  hasSegmentation = session.outputNames.includes("masks");

  inputBuf = new Float32Array(1 * 3 * inputSize * inputSize);
  prepCanvas = new OffscreenCanvas(inputSize, inputSize);
  prepCtx = prepCanvas.getContext("2d", { willReadFrequently: true });

  maskCanvas = null;
  maskImgData = null;

  const modelType = hasSegmentation ? "Seg" : "Det";
  badgeEl.textContent = activeBackend.toUpperCase();
  badgeEl.className = activeBackend;
  statusEl.childNodes[0].textContent = `${meta.label} (${inputSize}px, ${modelType}) `;
  startBtn.disabled = false;
  startBtnLarge.disabled = false;
  modelSelect.disabled = false;
  loading = false;
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
  for (let i = 0; i < hw; i++) {
    const r = imgData[i * 4] / 255;
    const g = imgData[i * 4 + 1] / 255;
    const b = imgData[i * 4 + 2] / 255;
    inputBuf[i] = (r - MEAN[0]) / STD[0];
    inputBuf[hw + i] = (g - MEAN[1]) / STD[1];
    inputBuf[2 * hw + i] = (b - MEAN[2]) / STD[2];
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

// --- Drawing ---

// Reusable owner map buffer — avoids allocation per frame
let ownerBuf = null;
let ownerBufSize = 0;

function drawMasks(detections, masksData, maskDims, vidW, vidH) {
  const maskH = maskDims[2];
  const maskW = maskDims[3];
  const maskSize = maskH * maskW;
  const fillAlpha = Math.round(maskOpacity * 255);
  const edgeAlpha = Math.round(outlineOpacity * 255);

  if (fillAlpha === 0 && edgeAlpha === 0) return;

  if (!maskCanvas || maskCanvas.width !== maskW || maskCanvas.height !== maskH) {
    maskCanvas = new OffscreenCanvas(maskW, maskH);
    maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    maskImgData = maskCtx.createImageData(maskW, maskH);
  }

  // Reuse owner buffer
  if (ownerBufSize !== maskSize) {
    ownerBuf = new Int8Array(maskSize);
    ownerBufSize = maskSize;
  }
  ownerBuf.fill(-1);

  const pixels = maskImgData.data;
  pixels.fill(0);

  const nd = detections.length;

  // Single pass: build owner map
  for (let i = 0; i < maskSize; i++) {
    for (let d = 0; d < nd; d++) {
      if (masksData[detections[d].queryIdx * maskSize + i] > 0) {
        ownerBuf[i] = d;
        break;
      }
    }
  }

  // Single pass: fill pixels + edge detect
  for (let my = 0; my < maskH; my++) {
    const rowOff = my * maskW;
    for (let mx = 0; mx < maskW; mx++) {
      const o = ownerBuf[rowOff + mx];
      if (o < 0) continue;

      const outX = maskW - 1 - mx;
      const idx = (rowOff + outX) << 2;

      const isEdge = edgeAlpha > 0 && (
        mx === 0 || mx === maskW - 1 || my === 0 || my === maskH - 1 ||
        ownerBuf[rowOff - maskW + mx] !== o ||
        ownerBuf[rowOff + maskW + mx] !== o ||
        ownerBuf[rowOff + mx - 1] !== o ||
        ownerBuf[rowOff + mx + 1] !== o
      );

      if (isEdge) {
        pixels[idx] = 255;
        pixels[idx + 1] = 255;
        pixels[idx + 2] = 255;
        pixels[idx + 3] = edgeAlpha;
      } else if (fillAlpha > 0) {
        const [r, g, b] = COLOR_RGBA[detections[o].classId];
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = fillAlpha;
      }
    }
  }

  maskCtx.putImageData(maskImgData, 0, 0);
  ctx.drawImage(maskCanvas, 0, 0, vidW, vidH);
}

function drawDetections(detections, w, h, masksData, maskDims) {
  ctx.clearRect(0, 0, w, h);

  if (masksData && maskDims) {
    drawMasks(detections, masksData, maskDims, w, h);
  }

  for (const det of detections) {
    const color = COLORS[det.classId];
    const label = `${COCO_CLASSES[det.classId]} ${det.score.toFixed(2)}`;
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
    label.textContent = `${COCO_CLASSES[det.classId]} ${det.score.toFixed(2)} (${areaPercent}%)`;
    label.style.color = COLORS[det.classId];

    el.style.display = "";
    if (!el.parentNode) segmentGrid.appendChild(el);
  }

  for (let i = count; i < activeSegments; i++) segmentPool[i].style.display = "none";
  activeSegments = count;
}

// --- Overlay positioning ---
// object-fit:contain means the video element has black bars.
// We compute the actual rendered content rect from the native aspect ratio.
function syncOverlay() {
  const elemRect = video.getBoundingClientRect();
  const nativeW = video.videoWidth;
  const nativeH = video.videoHeight;
  if (!nativeW || !nativeH) return;

  const elemW = elemRect.width;
  const elemH = elemRect.height;
  const scale = Math.min(elemW / nativeW, elemH / nativeH);
  const renderedW = nativeW * scale;
  const renderedH = nativeH * scale;
  // Center offset within the element
  const offsetX = (elemW - renderedW) / 2;
  const offsetY = (elemH - renderedH) / 2;

  const paneRect = videoPane.getBoundingClientRect();

  overlay.width = nativeW;
  overlay.height = nativeH;
  overlay.style.width = renderedW + "px";
  overlay.style.height = renderedH + "px";
  overlay.style.left = (elemRect.left - paneRect.left + offsetX) + "px";
  overlay.style.top = (elemRect.top - paneRect.top + offsetY) + "px";
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

  const inputTensor = preprocess(video);
  const output = await session.run({ input: inputTensor });

  const detsData = output.dets.data;
  const labelsData = output.labels.data;
  const masksData = hasSegmentation ? output.masks.data : null;
  const maskDims = hasSegmentation ? output.masks.dims : null;

  inputTensor.dispose();
  output.dets.dispose();
  output.labels.dispose();
  if (output.masks) output.masks.dispose();

  const vidW = video.videoWidth;
  const vidH = video.videoHeight;

  const detections = postprocess(detsData, labelsData, vidW, vidH);
  drawDetections(detections, overlay.width, overlay.height, masksData, maskDims);

  frameCount++;
  if (showSegments && frameCount % 3 === 0) {
    updateSidebar(detections, masksData, maskDims, vidW, vidH);
  }

  const elapsed = performance.now() - t0;
  const mode = hasSegmentation ? "seg" : "det";
  statsEl.textContent = `${elapsed.toFixed(0)}ms | ${(1000 / elapsed).toFixed(1)} FPS | ${detections.length} objects | ${mode}`;

  requestAnimationFrame(detectLoop);
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
