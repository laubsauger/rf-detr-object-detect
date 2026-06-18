// Inference bridge client — lets the web app run inference on the native Python
// zoo (RF-DETR/YOLO26 on MPS, YOLO26 on MLX) over a WebSocket.
//
// Transport-agnostic by design: decodeBridgeResult() turns the wire payload into
// the same {detections, masksData, maskDims} the ONNX path produces, so the draw
// code is shared. A future WebRTC transport only needs to swap connect()/infer()
// and can reuse decodeBridgeResult() untouched.

// Same-origin path proxied to the python bridge by vite (see vite.config.js).
// Same origin + wss on an https page => no mixed-content blocking.
const BRIDGE_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/bridge`;

export function createBridge(url = BRIDGE_URL) {
  let ws = null;
  let pending = null;   // resolver for the in-flight infer()
  let meta = null;      // {family, task, backend, device} from the server

  function connect(modelId, threshold) {
    return new Promise((resolve, reject) => {
      ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => ws.send(JSON.stringify({ type: "model", id: modelId, threshold }));
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          const msg = JSON.parse(ev.data);
          if (msg.type === "ready") { meta = msg; resolve(msg); }
          return;
        }
        if (pending) { const p = pending; pending = null; p(ev.data); }
      };
      ws.onerror = () => reject(new Error(`bridge connection failed (${url}) — is bridge.py running?`));
      ws.onclose = () => { if (pending) { const p = pending; pending = null; p(null); } };
    });
  }

  // One request in flight (backpressure): caller awaits before sending the next.
  function infer(imageData) {
    return new Promise((resolve) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) { resolve(null); return; }
      const { data, width, height } = imageData;  // RGBA Uint8ClampedArray
      const buf = new Uint8Array(5 + data.length);
      const dv = new DataView(buf.buffer);
      dv.setUint16(0, width, true);
      dv.setUint16(2, height, true);
      buf[4] = 4;  // channels (RGBA)
      buf.set(data, 5);
      pending = resolve;
      ws.send(buf.buffer);
    });
  }

  function setThreshold(value) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "threshold", value }));
    }
  }

  // Matte mode: server returns one person-union mask instead of all per-instance
  // masks (least back-and-forth for background removal).
  function setMatte(value) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "matte", value }));
    }
  }

  function close() { if (ws) { ws.onclose = null; ws.close(); ws = null; } }

  return { connect, infer, setThreshold, setMatte, close, get meta() { return meta; } };
}

// Wire payload -> {detections, masksData, maskDims, speed}. Coords arrive
// normalized + un-mirrored (model space); we mirror x and scale to video exactly
// like the ONNX postprocess, so drawDetections/drawMasks work unchanged.
export function decodeBridgeResult(buf, vidW, vidH) {
  const dv = new DataView(buf);
  const jsonLen = dv.getUint32(0, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jsonLen)));

  // Matte mode: single person-union mask, no boxes.
  if (json.matte) {
    return {
      matte: true,
      alpha: new Uint8Array(buf, 4 + jsonLen),   // [mh*mw] uint8 0/1
      mw: json.mw, mh: json.mh,
      speed: json.speed, server: json.server,
    };
  }

  const detections = json.dets.map((d) => ({
    x1: (1 - d.x2) * vidW, y1: d.y1 * vidH,   // mirror x (CSS-flipped video)
    x2: (1 - d.x1) * vidW, y2: d.y2 * vidH,
    score: d.score, classId: d.cls, queryIdx: d.q,
  }));

  let masksData = null, maskDims = null;
  if (json.seg && detections.length) {
    // uint8 0/1 masks at [n, mh, mw]; drawMasks treats > 0 as in-mask and
    // mirrors x itself, same as the ONNX mask buffer.
    masksData = new Uint8Array(buf, 4 + jsonLen);
    maskDims = [1, detections.length, json.mh, json.mw];
  }
  return { detections, masksData, maskDims, speed: json.speed, server: json.server };
}
