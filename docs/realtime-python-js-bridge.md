# Real-time browser ↔ Python inference bridge (reusable recipe)

Battle-tested setup for streaming webcam frames from a browser to a local Python
process for native inference (PyTorch/MPS, MLX, CUDA…) and getting results back
at 60–100 fps. Every number below was measured on Apple Silicon; the *design*
rules are portable.

## TL;DR — the rules that matter

1. **Transport = WebSocket, binary, one request in flight.** On localhost the
   loopback is ~GB/s; you do not need WebRTC (see "When WebRTC" below).
2. **Disable permessage-deflate.** `websockets.serve(..., compression=None)`.
   Compressing a ~128 KB payload costs **~45 ms/frame**. This is the single
   biggest footgun. Round-trip dropped 58 ms → 11 ms when disabled.
3. **Send raw frames, not JPEG/PNG.** Reuse the browser's `getImageData` RGBA
   buffer. A 640² frame is 1.6 MB; loopback ships it in <1 ms with zero codec
   CPU. Encoding would add 3–5 ms and buy nothing locally.
4. **Send small results.** Masks at *proto resolution* (e.g. 160²) as `uint8`,
   not full-frame. The browser upscales on the GPU canvas for free.
5. **Backpressure: one frame in flight.** Send the next frame only after the
   previous result returns; drop the rest. No queue buildup, always-fresh frame.
6. **Decode is transport-agnostic.** The function that turns the wire payload
   into your draw structures should not know whether it came from WebSocket,
   WebRTC, or local ONNX. Swap transports without touching draw code.
7. **TCP_NODELAY** on both sockets — cheap insurance against Nagle stalls.
8. **Don't reimplement the model's forward.** Use the framework's *warmed*
   optimized path (ultralytics `AutoBackend`), then do only the cheap decode
   yourself. A hand-rolled `model.model(x)` forward was 22 ms vs 9.8 ms for the
   library's backend — it applies fusion/fp16/inference_mode you'd otherwise miss.

## Architecture

```
browser                                  python (one process, model zoo)
────────                                  ──────────────────────────────
capture → drawImage(640²) → getImageData
  └ RGBA Uint8 (no encode)
  └─ ws.send(binary) ───────────────────▶ np.frombuffer → forward (MPS/MLX)
                                           decode masks @160 (no full-res upsample)
  ◀── ws.send([u32 len][json][masks]) ──┘ json: dets normalized + un-mirrored
 decodeResult() → {dets, masks}            masks: uint8 N×160×160
 draw (shared with local-inference path)
```

### Wire format

- **Up (binary):** `[u16 width][u16 height][u8 channels][...RGBA bytes]`.
- **Down (binary):** `[u32 jsonLen][json][mask bytes]`.
  - `json = {dets:[{x1,y1,x2,y2,score,cls,q}], mw, mh, seg, speed, server}`
  - coords **normalized 0..1 and un-mirrored** (model space). The browser applies
    its own mirror + scale, so the same draw code serves local and bridged paths.
  - masks: `uint8` 0/1, shape `[n, mh, mw]`, indexed by `q`.
- **Control (text JSON):** `{type:"model", id, threshold}` / `{type:"threshold", value}`.

### Server (Python) — essentials

```python
import asyncio, json, struct, numpy as np, websockets

async def handler(ws):
    sock = ws.transport.get_extra_info("socket")
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    async for msg in ws:
        if isinstance(msg, str): ...        # control: set model / threshold
        w, h = struct.unpack_from("<H", msg, 0)[0], struct.unpack_from("<H", msg, 2)[0]
        frame = np.frombuffer(msg, np.uint8, offset=5).reshape(h, w, msg[4])
        boxes, masks = infer(frame[:, :, :3])      # RGB straight through
        await ws.send(pack(boxes, masks))

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765,
                                max_size=None, compression=None):  # <- critical
        await asyncio.Future()
```

### Browser — essentials

```js
const ws = new WebSocket(`${location.protocol==="https:"?"wss":"ws"}//${location.host}/bridge`);
ws.binaryType = "arraybuffer";

// one in flight:
function infer(imageData) {
  return new Promise(res => {
    pending = res;
    const buf = new Uint8Array(5 + imageData.data.length);
    new DataView(buf.buffer).setUint16(0, imageData.width, true);
    new DataView(buf.buffer).setUint16(2, imageData.height, true);
    buf[4] = 4; buf.set(imageData.data, 5);
    ws.send(buf.buffer);
  });
}
ws.onmessage = e => { const p = pending; pending = null; p(e.data); };
```

### HTTPS dev origin → proxy the WebSocket

A `wss://` page cannot open a raw `ws://` (mixed content). Proxy it through the
dev server so the browser uses `wss://<host>/bridge` (same origin). Vite:

```js
server: { proxy: { "/bridge": { target: "ws://localhost:8765", ws: true } } }
```

`ws:true` makes it a passthrough tunnel, so the server's `compression=None` and
extension negotiation apply end-to-end (the proxy doesn't re-compress).

## Where the time goes (yolo26n-seg, 640², MPS)

| stage | ms | note |
|-------|---:|------|
| frame parse (frombuffer) | ~0 | view, no copy |
| **model forward** | 9.8 | the floor; use the framework's warmed backend |
| mask decode @160 (coeffs·proto, crop) | ~1.5 | CPU numpy; GPU is *slower* here (small-kernel overhead) |
| encode + pack | ~0.1 | json + `tobytes()` |
| wire (up 1.6 MB + down 128 KB) | ~1 | loopback |
| **round-trip total** | **~12.5** | ~80 fps |

The transport is <2 ms. Overhead lives in mask handling — keep masks at proto
res and never materialize them full-frame.

## Pitfalls that cost real time

- **permessage-deflate** (see rule 2) — 45 ms.
- **Full-res masks.** `sv.Detections.from_ultralytics` upsamples masks to frame
  res (~10 ms). Decode from proto coeffs instead.
- **GPU-side mask decode on MPS.** Many tiny ops → per-kernel launch overhead;
  CPU numpy at 160² won 23 ms → 11 ms. Measure, don't assume GPU is faster.
- **Non-contiguous / read-only tensors.** `torch.from_numpy` on a `frombuffer`
  view warns + can be slow; do one `np.ascontiguousarray(rgb.transpose(2,0,1))`.
- **Reimplementing the forward.** Lost fusion/fp16 → 22 ms vs 9.8 ms.

## When WebRTC instead

Only when Python is on a **different machine** over a real network. Then WebRTC
(aiortc) buys hardware video encode + congestion control; results go over a
DataChannel. On localhost it *adds* codec + jitter-buffer latency and loses
frame-exactness for nothing — WebSocket + raw frames is strictly better.

Keep the decode transport-agnostic and a WebRTC `infer()` drops in beside the
WebSocket one without touching draw code.

## Honest caveat: the bridge isn't always faster

If the model exports to ONNX and runs well on **WebGPU in-browser**, that path
has *no round trip* and often matches or beats the bridge:

| model | WebGPU (in-browser) | Python bridge |
|-------|--------------------:|--------------:|
| YOLO26-seg | ~47 fps | ~46 fps (MLX) / ~80 (det) |
| RF-DETR-seg | ~20 fps | ~21 fps (MPS) |

The bridge wins decisively when: WebGPU is unavailable, the model has no good
ONNX/WebGPU path, you need a framework-only model (MLX, custom), or you want to
keep weights/compute server-side. Benchmark your actual model before assuming
the bridge is the win.
