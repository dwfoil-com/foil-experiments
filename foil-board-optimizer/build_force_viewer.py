#!/usr/bin/env python3
"""Build interactive force visualization viewer from Phase 1 results.

Shows deck/mast forces with sliders to scale each load case independently.
Displays strain energy heatmap overlay.

Usage:
    python build_force_viewer.py results/modal_bulkhead5 -o force_viewer.html
"""

import json
import base64
import sys
import os
import argparse
import numpy as np
from pathlib import Path

def load_phase1_results(phase1_dir):
    """Load Phase 1 metadata and strain energy data."""
    meta_path = os.path.join(phase1_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Load per-load-case strain energy
    se_dir = os.path.join(phase1_dir, "strain_energy")
    load_cases = {}

    for lc_file in sorted(Path(se_dir).glob("*.bin")):
        lc_name = lc_file.stem
        se = np.fromfile(lc_file, dtype=np.float32)
        load_cases[lc_name] = {
            "se": se.tolist(),
            "scale": 1.0
        }

    return meta, load_cases


def get_load_case_forces(lc_name):
    """Return typical force magnitudes for each load case."""
    forces = {
        "riding_normal": {
            "deck_fy": 0.0,
            "deck_fz": -850,  # 85kg rider
            "mast_fx": 0.0,
            "mast_fy": 0.0,
            "mast_fz": -785
        },
        "pumping": {
            "deck_fy": 0.0,
            "deck_fz": -1500,  # large vertical pump
            "mast_fx": 0.0,
            "mast_fy": 0.0,
            "mast_fz": -1500
        },
        "jump_landing": {
            "deck_fy": 0.0,
            "deck_fz": -3000,  # high impact
            "mast_fx": 0.0,
            "mast_fy": 0.0,
            "mast_fz": -3000
        },
        "carving": {
            "deck_fy": -800,  # lateral
            "deck_fz": -850,
            "mast_fx": 0.0,
            "mast_fy": -400,  # mast experiences lateral load
            "mast_fz": -785
        },
        "front_foot_drive": {
            "deck_fy": 0.0,
            "deck_fz": -600,  # front foot pressed
            "mast_fx": 100,   # forward thrust
            "mast_fy": 0.0,
            "mast_fz": -400
        },
        "back_foot_drive": {
            "deck_fy": 0.0,
            "deck_fz": -600,  # back foot pressed
            "mast_fx": -100,  # backward thrust
            "mast_fy": 0.0,
            "mast_fz": -400
        }
    }
    return forces.get(lc_name, forces["riding_normal"])


def build_html(meta, load_cases, output_path):
    """Generate interactive HTML viewer with force sliders."""

    # Prepare load case data
    lc_data = {}
    for lc_name, lc_info in load_cases.items():
        forces = get_load_case_forces(lc_name)
        lc_data[lc_name] = {
            "forces": forces,
            "se": lc_info["se"]
        }

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Force Visualization — Foil Board Topology</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; overflow: hidden; }}
        #canvas {{ width: 100%; height: 80vh; display: block; }}
        #controls {{
            width: 100%; height: 20vh;
            background: #f5f5f5;
            padding: 20px;
            overflow-y: auto;
            box-sizing: border-box;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            align-content: start;
        }}
        .load-case {{
            background: white;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .load-case label {{
            font-weight: bold;
            display: block;
            margin-bottom: 8px;
        }}
        .slider-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        input[type="range"] {{
            flex: 1;
            min-width: 100px;
        }}
        .value-display {{
            min-width: 40px;
            text-align: right;
            font-size: 12px;
            color: #666;
        }}
        .legend {{
            font-size: 12px;
            color: #666;
            margin-top: 8px;
        }}
        h1 {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 10;
            background: rgba(255,255,255,0.9);
            padding: 10px;
            border-radius: 4px;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <h1>Load Case Forces & Strain Energy</h1>
    <canvas id="canvas"></canvas>
    <div id="controls"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Data from Phase 1
        const meta = {meta};
        const loadCaseData = {json.dumps(lc_data)};

        // Scale factors (user-adjustable)
        const scales = {{}};
        Object.keys(loadCaseData).forEach(lc => scales[lc] = 1.0);

        // Three.js scene
        const canvas = document.getElementById('canvas');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xcccccc);

        const camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
        camera.position.set(0.8, 0.3, 0.8);
        camera.lookAt(0.8, 0, 0.1);

        const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        renderer.shadowMap.enabled = true;

        // Lighting
        const ambLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(1, 1, 1);
        dirLight.castShadow = true;
        scene.add(dirLight);

        // Board geometry (simplified box for now)
        const boardGeo = new THREE.BoxGeometry(1.64, 0.45, 0.12);
        const boardMat = new THREE.MeshPhongMaterial({{ color: 0x4488ff, shininess: 100 }});
        const board = new THREE.Mesh(boardGeo, boardMat);
        board.castShadow = true;
        board.receiveShadow = true;
        scene.add(board);

        // Mast mount (small box at bottom center)
        const mastGeo = new THREE.BoxGeometry(0.08, 0.12, 0.05);
        const mastMat = new THREE.MeshPhongMaterial({{ color: 0x333333 }});
        const mast = new THREE.Mesh(mastGeo, mastMat);
        mast.position.set(0.73, 0, -0.08);
        mast.castShadow = true;
        scene.add(mast);

        // Helper: draw force arrow
        function drawForce(origin, force, color, scale = 0.001) {{
            const magnitude = Math.sqrt(force.x**2 + force.y**2 + force.z**2);
            if (magnitude < 1) return; // skip tiny forces

            const dir = new THREE.Vector3(force.x, force.y, force.z).normalize();
            const length = magnitude * scale;

            const arrowHelper = new THREE.ArrowHelper(dir, origin, length, color, 0.05, 0.05);
            scene.add(arrowHelper);
        }}

        // Helper: update forces for current load case
        function updateForces(lcName) {{
            // Remove old arrows
            scene.children.forEach(obj => {{
                if (obj.isArrowHelper || obj.userData.isForceArrow) scene.remove(obj);
            }});

            const lc = loadCaseData[lcName];
            const forces = lc.forces;
            const scale = scales[lcName];

            // Deck force (center top)
            const deckPos = new THREE.Vector3(0.73, 0, 0.08);
            drawForce(deckPos,
                {{ x: forces.deck_fy * scale, y: 0, z: forces.deck_fz * scale }},
                0xff4444, 0.0005);

            // Mast force (bottom center)
            const mastPos = new THREE.Vector3(0.73, 0, -0.08);
            drawForce(mastPos,
                {{ x: forces.mast_fx * scale, y: forces.mast_fy * scale, z: forces.mast_fz * scale }},
                0x44ff44, 0.0005);

            renderer.render(scene, camera);
        }}

        // Build controls
        const controlsDiv = document.getElementById('controls');
        Object.keys(loadCaseData).forEach(lcName => {{
            const container = document.createElement('div');
            container.className = 'load-case';

            const label = document.createElement('label');
            label.textContent = lcName.replace(/_/g, ' ').toUpperCase();
            container.appendChild(label);

            const sliderGroup = document.createElement('div');
            sliderGroup.className = 'slider-group';

            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = '0';
            slider.max = '200';
            slider.value = '100';
            slider.step = '10';

            const display = document.createElement('span');
            display.className = 'value-display';
            display.textContent = '100%';

            slider.addEventListener('input', (e) => {{
                const val = parseInt(e.target.value);
                scales[lcName] = val / 100.0;
                display.textContent = val + '%';
                updateForces(lcName);
            }});

            sliderGroup.appendChild(slider);
            sliderGroup.appendChild(display);
            container.appendChild(sliderGroup);

            const legend = document.createElement('div');
            legend.className = 'legend';
            legend.innerHTML = `
                <div>🔴 Deck force</div>
                <div>🟢 Mast force</div>
            `;
            container.appendChild(legend);

            controlsDiv.appendChild(container);
        }});

        // Show first load case
        updateForces(Object.keys(loadCaseData)[0]);

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}
        animate();

        // Handle resize
        window.addEventListener('resize', () => {{
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        }});
    </script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Force viewer: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build interactive force visualization viewer")
    parser.add_argument("phase1_dir", help="Phase 1 results directory")
    parser.add_argument("-o", "--output", default="force_viewer.html", help="Output HTML file")
    args = parser.parse_args()

    meta, load_cases = load_phase1_results(args.phase1_dir)
    build_html(meta, load_cases, args.output)


if __name__ == "__main__":
    main()
