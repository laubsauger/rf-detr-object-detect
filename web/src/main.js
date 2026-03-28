import * as ort from "onnxruntime-web";

ort.env.logLevel = "error";

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
const container = document.getElementById("container");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const startBtn = document.getElementById("startBtn");
const thresholdInput = document.getElementById("threshold");
const threshValEl = document.getElementById("threshVal");
const badgeEl = document.getElementById("backend-badge");
const modelSelect = document.getElementById("modelSelect");
const backendSelect = document.getElementById("backendSelect");
const fullscreenBtn = document.getElementById("fullscreenBtn");

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

let inputBuf = null;
let prepCanvas = null;
let prepCtx = null;
let maskCanvas = null;
let maskCtx = null;
let maskImgData = null;

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

fullscreenBtn.addEventListener("click", () => {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    container.requestFullscreen();
  }
});

document.addEventListener("fullscreenchange", () => {
  container.classList.toggle("fullscreen", !!document.fullscreenElement);
});

// --- Model loading ---

async function loadModel(modelId) {
  if (loading) return;
  loading = true;
  modelSelect.disabled = true;
  startBtn.disabled = true;
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

  for (const backend of backends) {
    try {
      statusEl.childNodes[0].textContent = `Loading ${meta.label} (${backend})...`;
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: [backend],
      });
      activeBackend = backend;
      break;
    } catch (e) {
      console.warn(`${backend} failed:`, e.message);
    }
  }

  if (!session) {
    statusEl.childNodes[0].textContent = "Failed to load model with selected backend.";
    loading = false;
    modelSelect.disabled = false;
    return;
  }

  hasSegmentation = session.outputNames.includes("masks");

  inputBuf = new Float32Array(1 * 3 * inputSize * inputSize);
  prepCanvas = new OffscreenCanvas(inputSize, inputSize);
  prepCtx = prepCanvas.getContext("2d", { willReadFrequently: true });

  const modelType = hasSegmentation ? "Seg" : "Det";
  badgeEl.textContent = activeBackend.toUpperCase();
  badgeEl.className = activeBackend;
  statusEl.childNodes[0].textContent = `${meta.label} (${inputSize}px, ${modelType}) `;
  startBtn.disabled = false;
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

function drawMasks(detections, masksData, maskDims, vidW, vidH) {
  const maskH = maskDims[2];
  const maskW = maskDims[3];
  const maskSize = maskH * maskW;

  if (!maskCanvas || maskCanvas.width !== vidW || maskCanvas.height !== vidH) {
    maskCanvas = new OffscreenCanvas(vidW, vidH);
    maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    maskImgData = maskCtx.createImageData(vidW, vidH);
  }

  const pixels = maskImgData.data;
  pixels.fill(0);

  // Iterate over output pixels and sample from the mask (not the other way around)
  // This avoids gaps from upscaling a small mask (78x78) to video resolution
  for (let py = 0; py < vidH; py++) {
    const my = (py / vidH) * maskH;
    const myFloor = Math.min(Math.floor(my), maskH - 1);

    for (let px = 0; px < vidW; px++) {
      // Mirror x: video is CSS-flipped, so sample from the right side for left pixels
      const srcX = (vidW - 1 - px) / vidW;
      const mx = Math.min(Math.floor(srcX * maskW), maskW - 1);

      const idx = (py * vidW + px) * 4;

      for (const det of detections) {
        const val = masksData[det.queryIdx * maskSize + myFloor * maskW + mx];
        if (val <= 0) continue;

        const [r, g, b, a] = COLOR_RGBA[det.classId];
        if (pixels[idx + 3] === 0) {
          pixels[idx] = r;
          pixels[idx + 1] = g;
          pixels[idx + 2] = b;
          pixels[idx + 3] = a;
        } else {
          pixels[idx] = (pixels[idx] + r) >> 1;
          pixels[idx + 1] = (pixels[idx + 1] + g) >> 1;
          pixels[idx + 2] = (pixels[idx + 2] + b) >> 1;
          pixels[idx + 3] = Math.min(255, pixels[idx + 3] + a);
        }
        break; // top detection wins this pixel
      }
    }
  }

  maskCtx.putImageData(maskImgData, 0, 0);
  ctx.drawImage(maskCanvas, 0, 0);
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

// --- Main loop ---

async function detectLoop() {
  if (!running) return;

  const t0 = performance.now();

  const inputTensor = preprocess(video);
  const output = await session.run({ input: inputTensor });

  // Copy data out before releasing tensors
  const detsData = output.dets.data;
  const labelsData = output.labels.data;
  const masksData = hasSegmentation ? output.masks.data : null;
  const maskDims = hasSegmentation ? output.masks.dims : null;

  // Release ORT GPU/WASM tensor memory to prevent leaks
  inputTensor.dispose();
  output.dets.dispose();
  output.labels.dispose();
  if (output.masks) output.masks.dispose();

  const vidW = video.videoWidth;
  const vidH = video.videoHeight;

  const detections = postprocess(detsData, labelsData, vidW, vidH);
  drawDetections(detections, overlay.width, overlay.height, masksData, maskDims);

  const elapsed = performance.now() - t0;
  const mode = hasSegmentation ? "seg" : "det";
  statsEl.textContent = `${elapsed.toFixed(0)}ms | ${(1000 / elapsed).toFixed(1)} FPS | ${detections.length} objects | ${mode}`;

  requestAnimationFrame(detectLoop);
}

startBtn.addEventListener("click", async () => {
  if (running) {
    running = false;
    startBtn.textContent = "Start Camera";
    video.srcObject?.getTracks().forEach(t => t.stop());
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    statsEl.textContent = "";
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
  });
  video.srcObject = stream;
  await video.play();

  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  container.style.width = video.videoWidth + "px";
  container.style.height = video.videoHeight + "px";

  running = true;
  startBtn.textContent = "Stop";
  detectLoop();
});

// --- Init ---

loadManifest().then(defaultModel => loadModel(defaultModel));
