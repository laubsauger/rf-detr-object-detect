import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createBridge, decodeBridgeResult } from "./bridge.js";
import { meshToObj } from "./mesh.js";

test("Python mesh packet decodes in JS with separate people, metres and normalized 2D", () => {
  const root = fileURLToPath(new URL("../../", import.meta.url));
  const result = spawnSync(`${root}venv/bin/python`, ["-c", `
from types import SimpleNamespace
import base64, json
import numpy as np
from instant_hmr import encode_mesh
p = SimpleNamespace(bbox=np.array([20, 10, 120, 90]), confidence=.8,
    joints_2d=np.array([[40, 20], [80, 60]]), joints_3d_cam=np.array([[1, 2, 3], [4, 5, 6]]),
    vertices_local=np.zeros((3,3)), vertices_cam=np.arange(9).reshape(3,3))
print(json.dumps([base64.b64encode(encode_mesh(ps, 200, 100, {})).decode() for ps in ([p,p], [])]))
`], { cwd: root, encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  const [full, empty] = JSON.parse(result.stdout).map(base64 => {
    const b = Buffer.from(base64, "base64");
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  });
  const decoded = decodeBridgeResult(full, 1000, 500);
  assert.equal(decoded.persons.length, 2);
  const person = decoded.persons[0];
  assert.equal(person.x1, 400);
  assert.equal(person.y2, 450);
  assert.deepEqual([...person.joints3d], [1, 2, 3, 4, 5, 6]);
  assert.ok(Math.abs(person.joints2d[0] - .2) < 1e-6);
  assert.deepEqual([...person.vertices], [0, 1, 2, 3, 4, 5, 6, 7, 8]);
  assert.deepEqual([...decoded.persons[1].vertices], [...person.vertices]);
  assert.deepEqual(decodeBridgeResult(empty, 1000, 500).persons, []);
  assert.throws(() => decodeBridgeResult(full.slice(0, -4), 1000, 500));
});

test("OBJ indexes multiple people without changing camera coordinates", () => {
  const person = { vertices: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]) };
  const obj = meshToObj([person, person], new Uint32Array([0, 1, 2]));
  assert.match(obj, /v 1 2 3\n/);
  assert.match(obj, /f 1 2 3\n/);
  assert.match(obj, /f 4 5 6\n/);
  assert.equal(obj.split("\n").filter(line => line.startsWith("v ")).length, 6);
});

test("existing detection and matte packets retain mirror and mask decoding", () => {
  const pack = (json, payload) => {
    const header = new TextEncoder().encode(JSON.stringify(json));
    const buf = new ArrayBuffer(4 + header.length + payload.length);
    new DataView(buf).setUint32(0, header.length, true);
    new Uint8Array(buf, 4, header.length).set(header);
    new Uint8Array(buf, 4 + header.length).set(payload);
    return buf;
  };
  const det = decodeBridgeResult(pack({ dets: [{ x1: .1, x2: .4, y1: .2, y2: .8, cls: 0, q: 0, score: .9 }], seg: true, mh: 1, mw: 2 }, [1, 0]), 100, 200);
  assert.equal(det.detections[0].x1, 60);
  assert.deepEqual(det.maskDims, [1, 1, 1, 2]);
  assert.deepEqual([...det.masksData], [1, 0]);
  const matte = decodeBridgeResult(pack({ matte: true, mh: 1, mw: 2 }, [0, 1]), 100, 200);
  assert.equal(matte.matte, true);
  assert.deepEqual([...matte.alpha], [0, 1]);
});

test("bridge surfaces model and inference errors, and settles pending work on close", async () => {
  const OriginalWebSocket = globalThis.WebSocket;
  let socket;
  class FakeSocket {
    static OPEN = 1;
    readyState = 1;
    constructor() { socket = this; }
    send() {}
    close() { this.onclose?.(); }
    message(data) { this.onmessage({ data: JSON.stringify(data) }); }
  }
  globalThis.WebSocket = FakeSocket;
  try {
    const bridge = createBridge("ws://test");
    const failure = bridge.connect("bad-model", .5);
    socket.message({ type: "error", message: "Install InstantHMR" });
    await assert.rejects(failure, /Install InstantHMR/);
    bridge.close();
    const ready = bridge.connect("mesh", .5);
    socket.message({ type: "ready", family: "instanthmr" });
    await ready;
    const configuring = bridge.configureMesh({ smoothing: "output", betaScale: .5, maxPersons: 1 });
    socket.message({ type: "mesh-configured", options: { smoothing: "output", betaScale: .5, maxPersons: 1 } });
    assert.equal((await configuring).smoothing, "output");
    assert.equal(bridge.meta.meshOptions.maxPersons, 1);
    const badConfiguration = bridge.configureMesh({ smoothing: "invalid" });
    socket.message({ type: "error", request: "mesh-options", message: "Unknown smoothing mode" });
    await assert.rejects(badConfiguration, /Unknown smoothing mode/);
    const frame = { data: new Uint8Array(4), width: 1, height: 1 };
    const pending = bridge.infer(frame);
    socket.message({ type: "error", message: "inference failed" });
    await assert.rejects(pending, /inference failed/);
    const closing = bridge.infer(frame);
    const closingConfig = bridge.configureMesh({}, true);
    bridge.close();
    await assert.rejects(closingConfig, /closed/);
    assert.equal(await closing, null);
    const disconnected = bridge.connect("mesh", .5);
    socket.close();
    await assert.rejects(disconnected, /disconnected/);
  } finally { globalThis.WebSocket = OriginalWebSocket; }
});
