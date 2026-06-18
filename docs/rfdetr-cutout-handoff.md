# Handoff: cheap RF-DETR person cutout (background removal)

Goal: frame in → person pixels out, transparent bg. Fast roundtrip, minimal data.

## The pattern (3 moving parts)

1. **Server inference-only** (`bridge.py`). Do NOT draw or send the image back.
   Return a single mask.
2. **Person-union mask at proto res** — one `uint8` 160² mask (~25 KB), not
   per-instance, not full-res.
3. **Browser composites on GPU** — `drawImage(video)` + `destination-in(mask)`.
   No per-pixel JS.

## Server (RF-DETR specifics)

```python
det = rfdetr_model.predict(bgr, threshold)        # sv.Detections, masks full-res
person = det.class_id == 1                         # RF-DETR uses COCO cat-id 1 (YOLO=0)
union = det.mask[person].max(0).astype("uint8")    # OR all person masks -> one mask
small = cv2.resize(union, (160, 160), cv2.INTER_NEAREST)  # proto res, cheap to ship
ws.send(struct.pack("<I", len(hdr)) + hdr + small.tobytes())   # hdr=json{mw,mh,...}
```

See `run_matte` / `encode_matte` in `bridge.py` for the exact code (handles the
fast YOLO path too; RF-DETR takes the `sv.Detections` branch).

## Transport (non-negotiable)

- WebSocket, **`compression=None`** (permessage-deflate = +45 ms/frame).
- Up: raw RGBA frame `[u16 w][u16 h][u8 ch][bytes]` (no JPEG; loopback is fast).
- One request in flight (send next frame only after prev result).
- HTTPS page → proxy `wss://host/bridge` (vite `proxy:{ "/bridge":{ws:true} }`).

## Browser composite (+ latency sync)

The mask is for a frame from ~inference-ms ago. **Snapshot that frame at send
time and composite against the snapshot, not the live video** — else the mask
trails a moving person.

```js
// 1. at inference time (top of loop, before await): freeze the frame
snapCtx.drawImage(video, 0, 0, w, h);              // snapCanvas = what inference saw

// 2. when the mask returns: build alpha at proto res, then GPU-scale as matte
maskCtx.putImageData(alphaImageData, 0, 0);        // 160², a[i*4+3]=255 where mask>0
ctx.save(); ctx.translate(w,0); ctx.scale(-1,1);   // mirror to match CSS-flipped video
ctx.drawImage(snapCanvas, 0, 0, w, h);             // <- snapshot, NOT live video
ctx.globalCompositeOperation = "destination-in";
ctx.drawImage(maskCanvas, 0, 0, w, h);             // upscale 160² -> w×h on GPU
ctx.restore(); ctx.globalCompositeOperation = "source-over";
```

Hide the live `<video>` (`opacity:0`, black pane bg) so the transparent overlay
reads as person-on-black. The cutout lags real-time by inference latency but
stays aligned. See `snapshotFrame` + `drawCutout` in `web/src/main.js`.

## Numbers (M-series, seg-nano)

~24 fps, **inference-bound** — the matte (union+resize+composite) is ~0.2 ms.
RF-DETR is the slow part; swap to YOLO26 (MLX 114 fps / torch 88 fps) if you want
speed. The roundtrip/transport itself is <2 ms.

## Gotchas

- RF-DETR person class = **1** (COCO cat-id), not 0.
- Mask is in un-mirrored model space; mirror it at draw time (video is CSS-flipped).
- Don't materialize full-res masks per frame — union first, ship at 160².
- **Latency sync:** composite against the snapshot frame, not live video (see above).
