// Bridge-only entries stay out of the ONNX export manifest.
export const MESH_MODELS = ["medium-commercial", "nano", "tiny", "small", "medium"].map(variant => ({
  id: `instanthmr-${variant}`, family: "instanthmr", task: "mesh", resolution: 640,
  label: `InstantHMR ${variant} · mesh${variant === "medium-commercial" ? "" : " (non-commercial)"}`,
}));

export function drawMeshOverlay(ctx, persons, edges, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#00e5ff";
  ctx.fillStyle = "#00e5ff";
  ctx.lineWidth = 2;
  for (const p of persons) {
    ctx.strokeRect(p.x1, p.y1, p.x2 - p.x1, p.y2 - p.y1);
    const k = p.joints2d;
    ctx.beginPath();
    for (const [a, b] of edges) {
      ctx.moveTo((1 - k[a * 2]) * width, k[a * 2 + 1] * height);
      ctx.lineTo((1 - k[b * 2]) * width, k[b * 2 + 1] * height);
    }
    ctx.stroke();
    for (let j = 0; j < k.length; j += 2) {
      ctx.beginPath();
      ctx.arc((1 - k[j]) * width, k[j + 1] * height, 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

// OBJ preserves camera coordinates and metres, including positions between people.
export function meshToObj(persons, faces) {
  const lines = ["# InstantHMR MHR mesh; metres; camera axes: x right, y down, z forward"];
  let base = 1;
  persons.forEach((p, index) => {
    lines.push(`o person_${index + 1}`);
    for (let i = 0; i < p.vertices.length; i += 3) {
      lines.push(`v ${p.vertices[i]} ${p.vertices[i + 1]} ${p.vertices[i + 2]}`);
    }
    for (let i = 0; i < faces.length; i += 3) {
      lines.push(`f ${faces[i] + base} ${faces[i + 1] + base} ${faces[i + 2] + base}`);
    }
    base += p.vertices.length / 3;
  });
  return lines.join("\n") + "\n";
}
