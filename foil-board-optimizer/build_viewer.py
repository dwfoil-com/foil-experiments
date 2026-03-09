"""Build an interactive HTML viewer from optimization results.

Usage:
    python build_viewer.py /tmp/foilopt-hires

Expects density.bin, meta.json in the output directory.
Produces viewer.html in the project directory.
"""

import base64, json, sys, os
import numpy as np


def build(result_dir: str, output_path: str = "viewer.html"):
    with open(os.path.join(result_dir, "meta.json")) as f:
        meta = json.load(f)

    density_raw = np.fromfile(os.path.join(result_dir, "density.bin"), dtype=np.float32)
    density_b64 = base64.b64encode(density_raw.tobytes()).decode()

    meta_json = json.dumps(meta)

    html = f'''<!DOCTYPE html>
<html>
<head>
<title>Foil Board Topology Optimization</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0a1a; color: #ddd; font-family: system-ui, -apple-system, sans-serif; overflow: hidden; }}
  canvas {{ display: block; }}
  #panel {{
    position: absolute; top: 12px; left: 12px; z-index: 10;
    background: rgba(10,10,30,0.85); backdrop-filter: blur(10px);
    padding: 16px; border-radius: 10px; width: 280px;
    border: 1px solid rgba(255,255,255,0.08);
  }}
  #panel h2 {{ font-size: 15px; margin-bottom: 10px; color: #fff; }}
  .control {{ margin: 8px 0; }}
  .control label {{ font-size: 12px; display: block; margin-bottom: 3px; opacity: 0.7; }}
  .control input[type=range] {{ width: 100%; accent-color: #e85d04; }}
  .stat {{ font-size: 12px; opacity: 0.6; margin: 2px 0; }}
  .legend {{ display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 11px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}
  #hint {{ position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
    font-size: 12px; opacity: 0.3; }}
  .btn {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
    color: #ddd; padding: 5px 10px; border-radius: 5px; font-size: 11px; cursor: pointer; margin: 2px; }}
  .btn:hover {{ background: rgba(255,255,255,0.15); }}
  .btn.active {{ background: rgba(232,93,4,0.3); border-color: #e85d04; }}
  .btn-row {{ display: flex; flex-wrap: wrap; gap: 2px; margin: 6px 0; }}
</style>
</head>
<body>

<div id="panel">
  <h2>Foil Board Internal Structure</h2>
  <div class="stat" id="stats"></div>

  <div class="control">
    <label>Density threshold: <span id="thresh-val">0.30</span></label>
    <input type="range" id="threshold" min="0.05" max="0.95" step="0.05" value="0.30">
  </div>

  <div class="control">
    <label>X-axis clip (length): <span id="clip-val">100%</span></label>
    <input type="range" id="clipX" min="0" max="100" step="1" value="100">
  </div>

  <div class="control">
    <label>Z-axis clip (thickness): <span id="clipZ-val">100%</span></label>
    <input type="range" id="clipZ" min="0" max="100" step="1" value="100">
  </div>

  <div class="btn-row">
    <button class="btn active" id="btn-structure">Structure</button>
    <button class="btn active" id="btn-loads">Loads</button>
    <button class="btn active" id="btn-board">Board outline</button>
    <button class="btn" id="btn-xray">X-ray</button>
  </div>

  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#e85d04"></div>Dense material</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ffd166"></div>Light material</div>
    <div class="legend-item"><div class="legend-dot" style="background:#4488ff"></div>Board outline</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ff3366"></div>Foot load zone</div>
    <div class="legend-item"><div class="legend-dot" style="background:#33ff88"></div>Mast mount (fixed)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ff33ff"></div>Mast forces</div>
  </div>
</div>

<div id="hint">Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const meta = {meta_json};
const {{ nelx, nely, nelz, lx, ly, lz }} = meta;

// Decode density
const b64 = "{density_b64}";
const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
const density = new Float32Array(raw.buffer);

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a1a);

const camera = new THREE.PerspectiveCamera(40, innerWidth/innerHeight, 0.01, 100);
camera.position.set(2.0, 1.2, 1.0);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(3, 4, 2);
sun.castShadow = true;
scene.add(sun);
const fill = new THREE.DirectionalLight(0x6688cc, 0.3);
fill.position.set(-2, -1, 3);
scene.add(fill);

// Groups
const structureGroup = new THREE.Group();
const loadsGroup = new THREE.Group();
const boardGroup = new THREE.Group();
scene.add(structureGroup);
scene.add(loadsGroup);
scene.add(boardGroup);

// Center offset (so board is centered at origin)
const cx = lx/2, cy = ly/2, cz = lz/2;

// === BOARD OUTLINE ===
const boardGeo = new THREE.BoxGeometry(lx, ly, lz);
const boardEdges = new THREE.LineSegments(
  new THREE.EdgesGeometry(boardGeo),
  new THREE.LineBasicMaterial({{ color: 0x4488ff, opacity: 0.4, transparent: true }})
);
boardGroup.add(boardEdges);

// === LOAD VISUALIZATION ===
function addZone(bounds, color, y, label) {{
  const [xmin, xmax, ymin, ymax] = bounds;
  const w = xmax - xmin, h = ymax - ymin;
  const geo = new THREE.PlaneGeometry(w, h);
  const mat = new THREE.MeshBasicMaterial({{
    color, opacity: 0.3, transparent: true, side: THREE.DoubleSide
  }});
  const plane = new THREE.Mesh(geo, mat);
  plane.position.set((xmin+xmax)/2 - cx, (ymin+ymax)/2 - cy, y - cz);
  loadsGroup.add(plane);

  // Border
  const edge = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({{ color, linewidth: 2 }})
  );
  edge.position.copy(plane.position);
  loadsGroup.add(edge);
}}

// Foot zone on deck (top face)
addZone(meta.foot_bounds, 0xff3366, lz, 'Foot zone');

// Mast mount on bottom
addZone(meta.mast_bounds, 0x33ff88, 0, 'Mast mount');

// Mast mount on deck (where mast forces are applied)
addZone(meta.mast_bounds, 0xff33ff, lz, 'Mast force zone');

// Force arrows
function addArrow(origin, dir, length, color) {{
  const d = new THREE.Vector3(...dir).normalize();
  const arrow = new THREE.ArrowHelper(d, new THREE.Vector3(...origin), length, color, length*0.25, length*0.12);
  loadsGroup.add(arrow);
}}

// Show forces for each load case (just riding_normal as representative)
const lc = meta.load_cases[0];
const mf = lc.mast_force;
const mb = meta.mast_bounds;
const mastCenter = [(mb[0]+mb[1])/2 - cx, (mb[2]+mb[3])/2 - cy, lz - cz];

// Mast force arrow (scale for visibility)
const fMag = Math.sqrt(mf[0]**2 + mf[1]**2 + mf[2]**2);
addArrow(mastCenter, mf, 0.15, 0xff33ff);

// Deck load arrows (downward over foot zone)
const fb = meta.foot_bounds;
for (let xi = 0; xi < 3; xi++) {{
  for (let yi = 0; yi < 2; yi++) {{
    const x = fb[0] + (fb[1]-fb[0]) * (xi+0.5)/3 - cx;
    const y = fb[2] + (fb[3]-fb[2]) * (yi+0.5)/2 - cy;
    addArrow([x, y, lz - cz + 0.06], [0,0,-1], 0.06, 0xff3366);
  }}
}}

// Fixed BC indicators (small cubes at mast mount bottom)
for (let xi = 0; xi < 3; xi++) {{
  for (let yi = 0; yi < 2; yi++) {{
    const x = mb[0] + (mb[1]-mb[0]) * (xi+0.5)/3 - cx;
    const y = mb[2] + (mb[3]-mb[2]) * (yi+0.5)/2 - cy;
    const cube = new THREE.Mesh(
      new THREE.BoxGeometry(0.015, 0.015, 0.015),
      new THREE.MeshBasicMaterial({{ color: 0x33ff88 }})
    );
    cube.position.set(x, y, -cz);
    loadsGroup.add(cube);
  }}
}}

// === VOXEL STRUCTURE ===
const dx = lx/nelx, dy = ly/nely, dz_el = lz/nelz;

// Instanced mesh for voxels
const voxelGeo = new THREE.BoxGeometry(dx*0.95, dy*0.95, dz_el*0.95);
const voxelMat = new THREE.MeshPhysicalMaterial({{
  vertexColors: true, metalness: 0.05, roughness: 0.6, clearcoat: 0.2,
}});

let instancedMesh = null;
let xrayMode = false;

function buildStructure(threshold, clipXPct, clipZPct) {{
  // Remove old
  if (instancedMesh) structureGroup.remove(instancedMesh);

  const clipX = (clipXPct / 100) * nelx;
  const clipZ = (clipZPct / 100) * nelz;

  // Count visible voxels
  let count = 0;
  // Elements ordered: k(z) outermost, j(y), i(x) innermost
  for (let k = 0; k < nelz; k++) {{
    for (let j = 0; j < nely; j++) {{
      for (let i = 0; i < nelx; i++) {{
        const idx = k * nely * nelx + j * nelx + i;
        if (density[idx] >= threshold && i < clipX && k < clipZ) count++;
      }}
    }}
  }}

  if (count === 0) return;

  const iMesh = new THREE.InstancedMesh(voxelGeo, voxelMat, count);
  iMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  const colorAttr = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);

  const matrix = new THREE.Matrix4();
  let vi = 0;

  for (let k = 0; k < nelz; k++) {{
    for (let j = 0; j < nely; j++) {{
      for (let i = 0; i < nelx; i++) {{
        const idx = k * nely * nelx + j * nelx + i;
        const d = density[idx];
        if (d < threshold || i >= clipX || k >= clipZ) continue;

        const x = (i + 0.5) * dx - cx;
        const y = (j + 0.5) * dy - cy;
        const z = (k + 0.5) * dz_el - cz;

        matrix.makeTranslation(x, y, z);
        iMesh.setMatrixAt(vi, matrix);

        // Color: low density = yellow, high = deep orange/red
        const t = Math.max(0, Math.min(1, (d - threshold) / (1 - threshold)));
        colorAttr.setXYZ(vi,
          0.9 + 0.1 * t,            // R
          0.82 - 0.6 * t,           // G
          0.39 - 0.35 * t            // B
        );
        vi++;
      }}
    }}
  }}

  iMesh.geometry.setAttribute('color', colorAttr);
  iMesh.instanceMatrix.needsUpdate = true;
  iMesh.castShadow = true;
  iMesh.receiveShadow = true;

  instancedMesh = iMesh;
  structureGroup.add(iMesh);
}}

// Initial build
buildStructure(0.3, 100, 100);

// Stats
document.getElementById('stats').innerHTML =
  `${{nelx}}×${{nely}}×${{nelz}} mesh (${{nelx*nely*nelz}} elements)<br>`+
  `Volume: ${{(meta.volfrac*100).toFixed(1)}}% &middot; `+
  `Compliance: ${{meta.compliance.toFixed(4)}} &middot; `+
  `${{meta.iterations}} iterations`;

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(0, 0, 0);

// UI handlers
let currentThresh = 0.3, currentClipX = 100, currentClipZ = 100;

document.getElementById('threshold').addEventListener('input', e => {{
  currentThresh = parseFloat(e.target.value);
  document.getElementById('thresh-val').textContent = currentThresh.toFixed(2);
  buildStructure(currentThresh, currentClipX, currentClipZ);
}});

document.getElementById('clipX').addEventListener('input', e => {{
  currentClipX = parseInt(e.target.value);
  document.getElementById('clip-val').textContent = currentClipX + '%';
  buildStructure(currentThresh, currentClipX, currentClipZ);
}});

document.getElementById('clipZ').addEventListener('input', e => {{
  currentClipZ = parseInt(e.target.value);
  document.getElementById('clipZ-val').textContent = currentClipZ + '%';
  buildStructure(currentThresh, currentClipX, currentClipZ);
}});

document.getElementById('btn-structure').addEventListener('click', e => {{
  e.target.classList.toggle('active');
  structureGroup.visible = e.target.classList.contains('active');
}});

document.getElementById('btn-loads').addEventListener('click', e => {{
  e.target.classList.toggle('active');
  loadsGroup.visible = e.target.classList.contains('active');
}});

document.getElementById('btn-board').addEventListener('click', e => {{
  e.target.classList.toggle('active');
  boardGroup.visible = e.target.classList.contains('active');
}});

document.getElementById('btn-xray').addEventListener('click', e => {{
  xrayMode = !xrayMode;
  e.target.classList.toggle('active');
  if (instancedMesh) {{
    instancedMesh.material.transparent = xrayMode;
    instancedMesh.material.opacity = xrayMode ? 0.3 : 1.0;
    instancedMesh.material.depthWrite = !xrayMode;
    instancedMesh.material.needsUpdate = true;
  }}
}});

// Animate
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

addEventListener('resize', () => {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});
</script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Viewer written to {output_path}")


if __name__ == "__main__":
    result_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/foilopt-hires"
    build(result_dir)
