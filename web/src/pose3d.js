// Three.js 3D skeleton viewer for RTMW3D. Joints = Points, bones = LineSegments,
// vertex-coloured per group. Each person is pelvis-centered (mid-hips), Y flipped
// (image-down → world-up), X mirrored to match the CSS-flipped video. Coords come
// in as model-input px (x:0..W, y:0..H, z:depth ~px); centered + scaled to world.
//
// Camera = OrbitControls: drag rotate, right-drag pan, wheel zoom. Auto-rotates
// until the user grabs it; a Reset restores the view and re-enables auto-rotate.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const VIEW = 1.6;        // normalized coords → world units
const PERSON_GAP = 1.8;  // x-offset between multiple people

let renderer, scene, camera, group, points, lines, controls, meshGroup;
let meshMode = false;
let wireframe = false;
let meshNeedsFit = true;
let latest = { persons: [], opts: null };
let running = false, raf = 0;
let onGrab = null;       // callback when user starts interacting (to flip UI off auto)

function pelvisOf(k, thr) {
  const lh = k[11], rh = k[12];
  if (lh && rh) return { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2, z: (lh.z + rh.z) / 2 };
  let n = 0, sx = 0, sy = 0, sz = 0;
  for (const p of k) { if (p && p.score >= thr) { sx += p.x; sy += p.y; sz += p.z; n++; } }
  return n ? { x: sx / n, y: sy / n, z: sz / n } : { x: 0, y: 0, z: 0 };
}

export function initPose3d(canvas) {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  renderer.setClearColor(0x0a0a0f, 1);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
  camera.position.set(0, 0, 6);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.6;
  controls.enablePan = true;
  controls.screenSpacePanning = true;
  controls.minDistance = 1.2;
  controls.maxDistance = 40;
  controls.target.set(0, 0, 0);
  controls.saveState();                 // snapshot for resetView()
  // Any manual interaction takes over: stop auto-rotate (UI checkbox synced via cb).
  controls.addEventListener("start", () => { if (onGrab) onGrab(); });

  group = new THREE.Group();
  scene.add(group);
  meshGroup = new THREE.Group();
  scene.add(meshGroup);
  scene.add(new THREE.HemisphereLight(0xd9ecff, 0x374151, 2.5));
  const keyLight = new THREE.DirectionalLight(0xffffff, 3);
  keyLight.position.set(2, 4, 5);
  scene.add(keyLight);

  const grid = new THREE.GridHelper(6, 12, 0x335577, 0x223344);
  grid.position.y = -2.2;
  group.add(grid);

  points = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ size: 0.09, vertexColors: true, sizeAttenuation: true })
  );
  lines = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ vertexColors: true })
  );
  group.add(points, lines);
  new ResizeObserver(() => {
    const { clientWidth: width, clientHeight: height } = canvas;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }).observe(canvas);
}

function rebuild() {
  const { persons, opts } = latest;
  const pos = [], col = [], lpos = [], lcol = [];
  if (opts) {
    persons.forEach((p, i) => {
      const k = p.kpts3d;
      if (!k) return;
      const piv = pelvisOf(k, opts.kptThresh);
      const dz = opts.depthScale ?? 1.5;   // depth exaggeration (UI slider)
      const tf = j => {
        const v = k[j];
        return [-(v.x - piv.x) * VIEW + i * PERSON_GAP, -(v.y - piv.y) * VIEW, -(v.z - piv.z) * VIEW * dz];
      };
      for (const g of opts.groups) {
        if (!g.enabled) continue;
        const c = new THREE.Color(g.color);
        for (let j = g.lo; j < g.hi; j++) {
          const v = k[j];
          if (!v || v.score < opts.kptThresh) continue;
          pos.push(...tf(j)); col.push(c.r, c.g, c.b);
        }
        for (const [a, b] of g.edges) {
          const ka = k[a], kb = k[b];
          if (!ka || !kb || ka.score < opts.kptThresh || kb.score < opts.kptThresh) continue;
          lpos.push(...tf(a), ...tf(b)); lcol.push(c.r, c.g, c.b, c.r, c.g, c.b);
        }
      }
    });
  }
  const pg = points.geometry, lg = lines.geometry;
  pg.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  pg.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  lg.setAttribute("position", new THREE.Float32BufferAttribute(lpos, 3));
  lg.setAttribute("color", new THREE.Float32BufferAttribute(lcol, 3));
}

function loop() {
  if (!running) return;
  raf = requestAnimationFrame(loop);
  if (!meshMode) rebuild();
  controls.update();           // damping + auto-rotate
  renderer.render(scene, camera);
}

export function updatePose3d(persons, opts) {
  if (meshMode) {
    controls.target.set(0, 0, 0);
    camera.position.set(0, 0, 6);
    controls.update();
    controls.saveState();
  }
  meshMode = false;
  if (meshGroup) meshGroup.visible = false;
  if (points) { points.visible = true; lines.visible = true; }
  latest = { persons, opts };
}

// Camera-space metres -> right-handed world, without reflecting the mesh.
// Rotate 180 degrees about X: image-down becomes up, forward becomes -Z.
export function updateMesh3d(persons, faces) {
  meshMode = true;
  points.visible = lines.visible = false;
  meshGroup.visible = true;
  if (!persons.length) meshNeedsFit = true;
  while (meshGroup.children.length > persons.length) {
    const mesh = meshGroup.children[meshGroup.children.length - 1];
    meshGroup.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  persons.forEach((p, i) => {
    let mesh = meshGroup.children[i];
    if (!mesh) {
      const geometry = new THREE.BufferGeometry();
      geometry.setIndex(new THREE.BufferAttribute(faces, 1));
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(p.vertices.length), 3));
      mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
        color: new THREE.Color().setHSL((0.52 + i * 0.17) % 1, 0.65, 0.6),
        roughness: 0.65, metalness: 0.1, side: THREE.DoubleSide, wireframe,
      }));
      mesh.rotation.x = Math.PI;
      meshGroup.add(mesh);
    }
    mesh.geometry.attributes.position.array.set(p.vertices);
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
    mesh.geometry.computeBoundingSphere();
  });
  if (persons.length) {
    const bounds = new THREE.Box3().setFromObject(meshGroup);
    const center = bounds.getCenter(new THREE.Vector3());
    // Center the scene using one shared translation; retain relative positions.
    meshGroup.position.sub(center);
    if (meshNeedsFit) {
      // Fit the full group using the narrower field of view (portrait canvas).
      const radius = bounds.getBoundingSphere(new THREE.Sphere()).radius;
      const vertical = THREE.MathUtils.degToRad(camera.fov / 2);
      const horizontal = Math.atan(Math.tan(vertical) * camera.aspect);
      const distance = radius / Math.sin(Math.min(vertical, horizontal)) * 1.1;
      controls.target.set(0, 0, 0);
      camera.position.set(0, 0, distance);
      controls.update();
      controls.saveState();
      meshNeedsFit = false;
    }
  }
}

export function setMeshWireframe(on) {
  wireframe = on;
  if (meshGroup) for (const mesh of meshGroup.children) mesh.material.wireframe = on;
}

export function setPose3dRunning(on) {
  if (on === running) return;
  running = on;
  if (on) loop(); else cancelAnimationFrame(raf);
}

export function setAutoRotate(on) {
  if (controls) controls.autoRotate = on;
}

export function resetView() {
  if (!controls) return;
  controls.reset();            // restores saved camera pos/target/zoom
  controls.autoRotate = true;
}

export function onPose3dInteract(cb) {
  onGrab = cb;
}
