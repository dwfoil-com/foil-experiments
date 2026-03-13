#!/usr/bin/env python3
"""Build a comparison HTML viewer showing multiple optimization result sets.

Loads Phase 1 (coarse SIMP), fullboard assembled volume, and optionally hires
cross-sections, and generates a single interactive viewer with a dataset switcher.

Usage:
    python build_comparison_viewer.py
    python build_comparison_viewer.py -o compare.html
"""

import base64, json, os, sys, glob
import numpy as np
from pathlib import Path


def load_phase1(result_dir="results/modal_bulkhead5"):
    """Load Phase 1 coarse SIMP result."""
    meta_path = os.path.join(result_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    density = np.fromfile(os.path.join(result_dir, "density.bin"), dtype=np.float32)

    # Load strain energy
    se_data = {}
    se_dir = os.path.join(result_dir, "strain_energy")
    if os.path.isdir(se_dir):
        for lc in meta.get("load_cases", []):
            se_path = os.path.join(se_dir, f"{lc['name']}.bin")
            if os.path.exists(se_path):
                se_data[lc["name"]] = np.fromfile(se_path, dtype=np.float32)

    nelx, nely, nelz = meta["nelx"], meta["nely"], meta["nelz"]
    lx, ly, lz = meta["lx"], meta["ly"], meta["lz"]
    volfrac = (density > 0.3).mean()
    V_elem = (lx / nelx) * (ly / nely) * (lz / nelz)
    core_mass_kg = volfrac * nelx * nely * nelz * V_elem * 1100  # printed core density
    shell_mass_kg = 3.32  # estimated carbon shell (planform-corrected)
    total_mass_kg = core_mass_kg + shell_mass_kg

    info_lines = [
        f"<b>Purpose:</b> Find where material should go (coarse placement)",
        f"<b>Pipeline:</b> Phase 1 only (3D SIMP topology optimization)",
        f"<b>Mesh:</b> {nelx}×{nely}×{nelz} = {nelx*nely*nelz:,} elements",
        f"<b>Solid fraction:</b> {volfrac*100:.1f}% (threshold 0.3)",
        f"<b>Compliance:</b> {meta.get('compliance', 0):.4f} J (lower = stiffer)",
        f"<b>Weight est:</b> {total_mass_kg:.1f} kg (core {core_mass_kg:.1f} + shell {shell_mass_kg:.1f})",
        f"<b>Load cases:</b> {len(meta.get('load_cases', []))} ({', '.join(lc['name'] for lc in meta.get('load_cases', []))})",
        f"<b>Iterations:</b> {meta.get('iterations', '?')} in {meta.get('time_seconds', 0)/60:.0f} min",
    ]

    return {
        "name": "Phase 1: Coarse SIMP (56×20×12)",
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "lx": lx, "ly": ly, "lz": lz,
        "density": density,
        "strain_energy": se_data,
        "meta": meta,
        "info": "<br>".join(info_lines),
    }


def load_fullboard_volume(vol_path="results/structure_fullboard.npy",
                          target_nx=80):
    """Load assembled fullboard volume, downsampled for browser rendering."""
    volume = np.load(vol_path)  # (nx, nz, ny) in build convention
    nx, nz, ny = volume.shape

    # Board dimensions (from FoilBoard defaults)
    lx, ly, lz = 1.641, 0.495, 0.117

    # Downsample in X for browser performance
    step_x = max(1, nx // target_nx)
    vol_ds = volume[::step_x, :, :]
    nx_ds = vol_ds.shape[0]

    # Convert (nx, nz, ny) → flat density in (k, j, i) = (nelz, nely, nelx) order
    # The viewer expects: idx = k * nely * nelx + j * nelx + i
    nelx, nely, nelz = nx_ds, ny, nz
    density = np.zeros(nelx * nely * nelz, dtype=np.float32)
    for k in range(nelz):
        for j in range(nely):
            for i in range(nelx):
                density[k * nely * nelx + j * nelx + i] = vol_ds[i, k, j]

    volfrac = (density > 0.5).mean()
    V_elem = (lx / nelx) * (ly / nely) * (lz / nelz)
    core_mass_kg = volfrac * nelx * nely * nelz * V_elem * 1100
    shell_mass_kg = 3.32
    total_mass_kg = core_mass_kg + shell_mass_kg

    info_lines = [
        f"<b>Purpose:</b> Continuous internal structure interpolated between cross-sections",
        f"<b>Pipeline:</b> Phases 1→2→3 (SIMP → 2D refinement → 3D interpolation)",
        f"<b>Mesh:</b> {nelx}×{nely}×{nelz} = {nelx*nely*nelz:,} voxels (from {nx}×{nz}×{ny})",
        f"<b>Solid fraction:</b> {volfrac*100:.1f}% (threshold 0.5)",
        f"<b>Weight est:</b> {total_mass_kg:.1f} kg (core {core_mass_kg:.1f} + shell {shell_mass_kg:.1f})",
        f"<b>Note:</b> Includes shell — use 'Internal Only' to see ribs without shell",
    ]

    return {
        "name": f"Fullboard Assembled ({nelx}×{nely}×{nelz})",
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "lx": lx, "ly": ly, "lz": lz,
        "density": density,
        "strain_energy": {},
        "meta": None,
        "info": "<br>".join(info_lines),
    }


def load_fullboard_at_phase1_res(vol_path="results/structure_fullboard.npy",
                                  phase1_meta=None):
    """Resample fullboard volume to Phase 1 mesh resolution for direct comparison."""
    volume = np.load(vol_path)  # (nx, nz, ny)
    nx_v, nz_v, ny_v = volume.shape
    lx, ly, lz = 1.641, 0.495, 0.117

    nelx = phase1_meta["nelx"]
    nely = phase1_meta["nely"]
    nelz = phase1_meta["nelz"]

    # Resample via nearest-neighbor
    density = np.zeros(nelx * nely * nelz, dtype=np.float32)
    for k in range(nelz):
        for j in range(nely):
            for i in range(nelx):
                # Map to volume indices
                ix = min(int((i + 0.5) / nelx * nx_v), nx_v - 1)
                iz = min(int((k + 0.5) / nelz * nz_v), nz_v - 1)
                iy = min(int((j + 0.5) / nely * ny_v), ny_v - 1)
                density[k * nely * nelx + j * nelx + i] = volume[ix, iz, iy]

    volfrac = (density > 0.5).mean()
    V_elem = (lx / nelx) * (ly / nely) * (lz / nelz)
    core_mass_kg = volfrac * nelx * nely * nelz * V_elem * 1100
    shell_mass_kg = 3.32
    total_mass_kg = core_mass_kg + shell_mass_kg

    info_lines = [
        f"<b>Purpose:</b> Direct comparison — fullboard result resampled onto Phase 1 mesh",
        f"<b>Pipeline:</b> Phases 1→2→3, then resampled back to Phase 1 grid",
        f"<b>Mesh:</b> {nelx}×{nely}×{nelz} = {nelx*nely*nelz:,} elements (from {nx_v}×{nz_v}×{ny_v})",
        f"<b>Solid fraction:</b> {volfrac*100:.1f}% (threshold 0.5)",
        f"<b>Weight est:</b> {total_mass_kg:.1f} kg (core {core_mass_kg:.1f} + shell {shell_mass_kg:.1f})",
        f"<b>Note:</b> Same mesh as Phase 1 — compare directly to see what refinement changed",
    ]

    return {
        "name": f"Fullboard → Phase 1 Mesh ({nelx}×{nely}×{nelz})",
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "lx": lx, "ly": ly, "lz": lz,
        "density": density,
        "strain_energy": {},
        "meta": None,
        "info": "<br>".join(info_lines),
    }


def load_hires_cross_sections(cs_dir="results/cross_sections_hires"):
    """Load hires cross-sections and assemble into 3D volume."""
    summary_path = os.path.join(cs_dir, "summary.json")
    if not os.path.exists(summary_path):
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    bhs = summary.get("bulkheads", [])
    if not bhs:
        return None

    lx, ly, lz = 1.641, 0.495, 0.117

    # Load each cross-section
    slices = []
    for bh in bhs:
        npy_path = os.path.join(cs_dir, f"density_x{bh['x_pos']:.3f}.npy")
        if os.path.exists(npy_path):
            cs = np.load(npy_path)  # (nelz_2d, nely_2d)
            slices.append((bh["x_pos"], cs))

    if not slices:
        return None

    slices.sort(key=lambda s: s[0])
    x_positions = [s[0] for s in slices]
    cs_shape = slices[0][1].shape  # (nelz_2d, nely_2d)
    nelz_2d, nely_2d = cs_shape

    # Downsample if too large
    step_y = max(1, nely_2d // 50)
    step_z = max(1, nelz_2d // 30)

    nelx = len(slices)
    nely = nely_2d // step_y
    nelz = nelz_2d // step_z

    density = np.zeros(nelx * nely * nelz, dtype=np.float32)
    for i, (xp, cs) in enumerate(slices):
        cs_ds = cs[::step_z, ::step_y]
        for k in range(min(nelz, cs_ds.shape[0])):
            for j in range(min(nely, cs_ds.shape[1])):
                density[k * nely * nelx + j * nelx + i] = cs_ds[k, j]

    x_min, x_max = x_positions[0], x_positions[-1]
    x_span = x_max - x_min

    volfrac = (density > 0.5).mean()
    V_elem = (x_span / nelx) * (ly / nely) * (lz / nelz)
    core_mass_kg = volfrac * nelx * nely * nelz * V_elem * 1100
    elem_dy = ly / nely_2d * 1000
    elem_dz = lz / nelz_2d * 1000

    info_lines = [
        f"<b>Purpose:</b> High-resolution under-foot zone — finest rib detail available",
        f"<b>Pipeline:</b> Phases 1→2 (2D SIMP at 200×60 resolution)",
        f"<b>Mesh:</b> {nelx} slices × {nely}×{nelz} (from {nely_2d}×{nelz_2d}, {elem_dy:.1f}×{elem_dz:.1f}mm elements)",
        f"<b>Coverage:</b> X=[{x_min:.3f}, {x_max:.3f}]m (under-foot zone only)",
        f"<b>Solid fraction:</b> {volfrac*100:.1f}% (target 15%)",
        f"<b>Weight est:</b> {core_mass_kg:.1f} kg core (partial board — under-foot only)",
        f"<b>Note:</b> 4× finer than fullboard cross-sections — see individual rib features",
    ]

    return {
        "name": f"Hires Cross-Sections ({nelx}×{nely}×{nelz})",
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "lx": x_span, "ly": ly, "lz": lz,
        "density": density,
        "strain_energy": {},
        "meta": None,
        "info": "<br>".join(info_lines),
        "x_offset": x_min,
    }


def load_fullboard_cross_sections(cs_dir="results/cross_sections_fullboard"):
    """Load fullboard cross-sections and assemble into 3D volume."""
    summary_path = os.path.join(cs_dir, "summary.json")
    if not os.path.exists(summary_path):
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    bhs = summary.get("bulkheads", [])
    if not bhs:
        return None

    lx, ly, lz = 1.641, 0.495, 0.117

    slices = []
    for bh in bhs:
        npy_path = os.path.join(cs_dir, f"density_x{bh['x_pos']:.3f}.npy")
        if os.path.exists(npy_path):
            cs = np.load(npy_path)
            slices.append((bh["x_pos"], cs))

    if not slices:
        return None

    slices.sort(key=lambda s: s[0])
    cs_shape = slices[0][1].shape
    nelz_2d, nely_2d = cs_shape

    # Downsample cross-sections for browser
    step_y = max(1, nely_2d // 50)
    step_z = max(1, nelz_2d // 15)

    nelx = len(slices)
    nely = nely_2d // step_y
    nelz = nelz_2d // step_z

    density = np.zeros(nelx * nely * nelz, dtype=np.float32)
    for i, (xp, cs) in enumerate(slices):
        cs_ds = cs[::step_z, ::step_y]
        for k in range(min(nelz, cs_ds.shape[0])):
            for j in range(min(nely, cs_ds.shape[1])):
                density[k * nely * nelx + j * nelx + i] = cs_ds[k, j]

    volfrac = (density > 0.5).mean()
    V_elem = (lx / nelx) * (ly / nely) * (lz / nelz)
    core_mass_kg = volfrac * nelx * nely * nelz * V_elem * 1100
    shell_mass_kg = 3.32
    total_mass_kg = core_mass_kg + shell_mass_kg
    elem_dy = ly / nely_2d * 1000  # mm
    elem_dz = lz / nelz_2d * 1000  # mm

    info_lines = [
        f"<b>Purpose:</b> High-res 2D rib/web patterns at each X station (Phase 2 raw output)",
        f"<b>Pipeline:</b> Phases 1→2 (56 independent 2D SIMP optimizations)",
        f"<b>Mesh:</b> {nelx} slices × {nely}×{nelz} (from {nely_2d}×{nelz_2d}, {elem_dy:.1f}×{elem_dz:.1f}mm elements)",
        f"<b>Solid fraction:</b> {volfrac*100:.1f}% (target 15%)",
        f"<b>Weight est:</b> {total_mass_kg:.1f} kg (core {core_mass_kg:.1f} + shell {shell_mass_kg:.1f})",
        f"<b>Note:</b> Each slice optimized independently — no X-continuity between slices",
    ]

    return {
        "name": f"Fullboard Cross-Sections ({nelx}×{nely}×{nelz})",
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "lx": lx, "ly": ly, "lz": lz,
        "density": density,
        "strain_energy": {},
        "meta": None,
        "info": "<br>".join(info_lines),
    }


def load_internal_only_volume(vol_path="results/board_continuous_volume.npy",
                              target_nx=80):
    """Load continuous volume with shell stripped — internal ribs only."""
    import json as _json
    sys.path.insert(0, os.path.dirname(__file__))
    from foilopt.geometry.board import FoilBoard, load_board_shape

    volume = np.load(vol_path)  # (nx, nz, ny)
    nx, nz, ny = volume.shape
    board = FoilBoard()
    bs = load_board_shape()
    if bs is None:
        return None

    lx, ly, lz = board.length, board.width, board.thickness

    # Infer x_coords from summary if available
    summary_path = os.path.join(os.path.dirname(vol_path), "cross_sections_fullboard", "summary.json")
    if not os.path.exists(summary_path):
        summary_path = os.path.join("results", "cross_sections_fullboard", "summary.json")
    if os.path.exists(summary_path):
        summary = _json.load(open(summary_path))
        bhs = summary.get("bulkheads", [])
        if bhs:
            x_min, x_max = bhs[0]["x_pos"], bhs[-1]["x_pos"]
        else:
            x_min, x_max = 0.0, lx
    else:
        x_min, x_max = 0.0, lx

    x_coords = np.linspace(x_min, x_max, nx)

    # Vectorized shell stripping using board shape superellipse
    x_3d = np.broadcast_to(x_coords[:, None, None], (nx, nz, ny))
    iy_3d = np.broadcast_to(np.arange(ny)[None, None, :], (nx, nz, ny))
    iz_3d = np.broadcast_to(np.arange(nz)[None, :, None], (nx, nz, ny))

    y_phys = iy_3d / ny * board.width
    z_opt = iz_3d / nz * board.thickness

    hw = bs.half_width_at(x_3d.ravel()).reshape(nx, nz, ny)
    bot_z = bs.bot_z_at(x_3d.ravel()).reshape(nx, nz, ny)
    deck_z = bs.deck_z_at(x_3d.ravel()).reshape(nx, nz, ny)

    z_s3d = bs.z_min + (z_opt / board.thickness) * (bs.z_max - bs.z_min)
    center_y = board.width / 2.0
    y_norm = np.abs(y_phys - center_y) / np.maximum(hw, 1e-6)
    z_mid = (bot_z + deck_z) / 2.0
    z_half = (deck_z - bot_z) / 2.0
    z_norm = np.abs(z_s3d - z_mid) / np.maximum(z_half, 1e-6)

    r = np.power(y_norm, 2.5) + np.power(z_norm, 2.5)

    vol_internal = volume.copy()
    vol_internal[r > 0.72] = 0.0  # strip shell

    # Downsample for browser
    step_x = max(1, nx // target_nx)
    vol_ds = vol_internal[::step_x, :, :]
    nx_ds = vol_ds.shape[0]

    # Convert (nx, nz, ny) → flat density in viewer order
    nelx, nely, nelz = nx_ds, ny, nz
    density = np.zeros(nelx * nely * nelz, dtype=np.float32)
    for k in range(nelz):
        for j in range(nely):
            for i in range(nelx):
                density[k * nely * nelx + j * nelx + i] = vol_ds[i, k, j]

    volfrac = (density > 0.5).mean()
    V_elem = (lx / nelx) * (ly / nely) * (lz / nelz)
    core_mass_kg = volfrac * nelx * nely * nelz * V_elem * 1100  # all core material
    # No shell in this view — weight is core only

    info_lines = [
        f"<b>Purpose:</b> Internal ribs/webs only — what gets 3D printed (shell removed)",
        f"<b>Pipeline:</b> Phases 1→2→3, then shell stripped via superellipse mask (r>0.72)",
        f"<b>Mesh:</b> {nelx}×{nely}×{nelz} = {nelx*nely*nelz:,} voxels (from {nx}×{nz}×{ny})",
        f"<b>Solid fraction:</b> {volfrac*100:.1f}% (internal ribs only)",
        f"<b>Weight est:</b> {core_mass_kg:.1f} kg (printed core only, no shell)",
        f"<b>STL available:</b> board_internal_only.stl (viewable in Preview.app)",
        f"<b>Note:</b> This is what gets 3D printed — carbon shell is laid up separately",
    ]

    return {
        "name": f"Internal Only — No Shell ({nelx}×{nely}×{nelz})",
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "lx": lx, "ly": ly, "lz": lz,
        "density": density,
        "strain_energy": {},
        "meta": None,
        "info": "<br>".join(info_lines),
    }


def load_discrete_bulkhead_volume(vol_path="results/board_volume.npy",
                                   target_nx=80):
    """Load discrete bulkhead volume (plates, not continuous)."""
    if not os.path.exists(vol_path):
        return None

    volume = np.load(vol_path)
    nx, nz, ny = volume.shape
    lx, ly, lz = 1.641, 0.495, 0.117

    step_x = max(1, nx // target_nx)
    vol_ds = volume[::step_x, :, :]
    nx_ds = vol_ds.shape[0]

    nelx, nely, nelz = nx_ds, ny, nz
    density = np.zeros(nelx * nely * nelz, dtype=np.float32)
    for k in range(nelz):
        for j in range(nely):
            for i in range(nelx):
                density[k * nely * nelx + j * nelx + i] = vol_ds[i, k, j]

    volfrac = (density > 0.5).mean()
    V_elem = (lx / nelx) * (ly / nely) * (lz / nelz)
    core_mass_kg = volfrac * nelx * nely * nelz * V_elem * 1100
    shell_mass_kg = 3.32
    total_mass_kg = core_mass_kg + shell_mass_kg

    info_lines = [
        f"<b>Purpose:</b> Traditional construction — full-height transverse plates at fixed X stations",
        f"<b>Pipeline:</b> Phases 1→2→4 (discrete bulkhead mode, not continuous)",
        f"<b>Mesh:</b> {nelx}×{nely}×{nelz} = {nelx*nely*nelz:,} voxels (from {nx}×{nz}×{ny})",
        f"<b>Solid fraction:</b> {volfrac*100:.1f}%",
        f"<b>Weight est:</b> {total_mass_kg:.1f} kg (core {core_mass_kg:.1f} + shell {shell_mass_kg:.1f})",
        f"<b>Note:</b> Simpler to build than continuous — CNC-cuttable plywood/foam plates",
    ]

    return {
        "name": f"Discrete Bulkheads ({nelx}×{nely}×{nelz})",
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "lx": lx, "ly": ly, "lz": lz,
        "density": density,
        "strain_energy": {},
        "meta": None,
        "info": "<br>".join(info_lines),
    }


def parse_s3dx_for_viewer(s3dx_path=None):
    """Import and use the parse_s3dx from build_viewer.py."""
    sys.path.insert(0, os.path.dirname(__file__))
    from build_viewer import parse_s3dx

    if s3dx_path and os.path.exists(s3dx_path):
        return parse_s3dx(s3dx_path)
    default = os.path.expanduser("~/Downloads/TT60 - Ken Adgate Inspired.s3dx")
    if os.path.exists(default):
        return parse_s3dx(default)
    return None


def build_comparison_html(datasets, board_shape, output_path):
    """Generate the comparison HTML viewer."""
    # Encode each dataset's density as base64
    encoded = []
    for ds in datasets:
        d_b64 = base64.b64encode(ds["density"].tobytes()).decode()

        # Encode strain energy
        se_b64 = {}
        for name, se in ds.get("strain_energy", {}).items():
            se_b64[name] = base64.b64encode(se.tobytes()).decode()

        encoded.append({
            "name": ds["name"],
            "nelx": ds["nelx"], "nely": ds["nely"], "nelz": ds["nelz"],
            "lx": ds["lx"], "ly": ds["ly"], "lz": ds["lz"],
            "density_b64": d_b64,
            "se_b64": se_b64,
            "info": ds["info"],
            "has_meta": ds.get("meta") is not None,
        })

    # Use Phase 1 meta for load zones / force arrows (always available)
    phase1_meta = datasets[0].get("meta", {})
    meta_json = json.dumps(phase1_meta) if phase1_meta else "{}"
    board_shape_json = json.dumps(board_shape) if board_shape else "null"
    datasets_json = json.dumps(encoded)

    html = f'''<!DOCTYPE html>
<html>
<head>
<title>Foil Board — Structure Comparison</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0a1a; color: #ddd; font-family: system-ui, -apple-system, sans-serif; overflow: hidden; }}
  canvas {{ display: block; }}
  #panel {{
    position: absolute; top: 12px; left: 12px; z-index: 10;
    background: rgba(10,10,30,0.85); backdrop-filter: blur(10px);
    padding: 16px; border-radius: 10px; width: 340px;
    border: 1px solid rgba(255,255,255,0.08);
    max-height: calc(100vh - 24px); overflow-y: auto;
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
  select {{
    width: 100%; padding: 8px; border-radius: 6px; font-size: 13px;
    background: rgba(255,255,255,0.1); color: #fff; border: 1px solid rgba(255,255,255,0.2);
    cursor: pointer; margin: 4px 0;
  }}
  select option {{ background: #1a1a2e; color: #fff; }}
  #dataset-info {{
    font-size: 11px; opacity: 0.7; margin: 6px 0; line-height: 1.6;
    padding: 8px 10px; background: rgba(255,255,255,0.05); border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.06);
  }}
  #dataset-info b {{
    color: #ffd166; opacity: 1;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none; width: 16px; height: 16px;
    background: #fff; border-radius: 50%; border: 2px solid #e85d04;
    cursor: pointer; box-shadow: 0 1px 4px rgba(0,0,0,0.4);
  }}
  .separator {{
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 10px 0;
  }}
</style>
</head>
<body>

<div id="panel">
  <h2>Structure Comparison</h2>

  <div class="control">
    <label>Dataset:</label>
    <select id="dataset-select"></select>
    <div id="dataset-info"></div>
  </div>

  <div class="separator"></div>

  <div class="control">
    <label>Density threshold: <span id="thresh-val" style="font-weight:bold">0.30</span></label>
    <div style="position:relative;height:22px;margin:4px 0">
      <div style="position:absolute;top:6px;left:0;right:0;height:10px;border-radius:5px;
        background:linear-gradient(to right, #ddeeff 0%, #ffd166 25%, #e85d04 55%, #c1121f 85%, #6b0f1a 100%);
        opacity:0.7"></div>
      <input type="range" id="threshold" min="0.02" max="0.95" step="0.01" value="0.30"
        style="position:absolute;top:0;width:100%;-webkit-appearance:none;background:transparent;height:22px">
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;opacity:0.4;margin-top:2px">
      <span>foam</span><span>reinforcement</span><span>primary</span>
    </div>
  </div>

  <div class="control">
    <label>X-axis clip (length): <span id="clip-val">100%</span></label>
    <input type="range" id="clipX" min="0" max="100" step="1" value="100">
  </div>

  <div class="control">
    <label>Z-axis clip (thickness): <span id="clipZ-val">100%</span></label>
    <input type="range" id="clipZ" min="0" max="100" step="1" value="100">
  </div>

  <div class="control">
    <label>Color mode:</label>
    <div class="btn-row">
      <button class="btn active" id="btn-density">Density</button>
      <button class="btn" id="btn-lc-riding">Riding</button>
      <button class="btn" id="btn-lc-pumping">Pumping</button>
      <button class="btn" id="btn-lc-landing">Landing</button>
      <button class="btn" id="btn-lc-carving">Carving</button>
    </div>
  </div>

  <div class="btn-row">
    <button class="btn active" id="btn-foam">Foam core</button>
    <button class="btn active" id="btn-loads">Loads</button>
    <button class="btn active" id="btn-board">Board shape</button>
    <button class="btn" id="btn-xray">X-ray</button>
  </div>

  <div class="separator"></div>

  <div id="voxel-stats" class="stat"></div>

  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#6b0f1a"></div>Primary</div>
    <div class="legend-item"><div class="legend-dot" style="background:#e85d04"></div>Secondary</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ffd166"></div>Reinforcement</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ddeeff;opacity:0.4"></div>Foam</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ff3366"></div>Front foot</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ff6633"></div>Back foot</div>
    <div class="legend-item"><div class="legend-dot" style="background:#33ff88"></div>Mast mount</div>
  </div>
</div>

<div id="hint">Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan &middot; Switch datasets above</div>

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

// All datasets
const allDatasets = {datasets_json};
const meta = {meta_json};
const boardData = {board_shape_json};

// Decode datasets
const decoded = allDatasets.map(ds => {{
  const raw = Uint8Array.from(atob(ds.density_b64), c => c.charCodeAt(0));
  const density = new Float32Array(raw.buffer);

  // Decode strain energy
  const strainEnergy = {{}};
  for (const [name, b] of Object.entries(ds.se_b64 || {{}})) {{
    const seRaw = Uint8Array.from(atob(b), c => c.charCodeAt(0));
    const se = new Float32Array(seRaw.buffer);
    let mx = 0;
    for (let i = 0; i < se.length; i++) if (se[i] > mx) mx = se[i];
    se._max = mx;
    strainEnergy[name] = se;
  }}

  return {{
    name: ds.name,
    nelx: ds.nelx, nely: ds.nely, nelz: ds.nelz,
    lx: ds.lx, ly: ds.ly, lz: ds.lz,
    density, strainEnergy, info: ds.info,
    has_meta: ds.has_meta,
  }};
}});

let currentDS = decoded[0];
let activeColorMode = 'density';

// Scene
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
scene.add(new THREE.DirectionalLight(0x6688cc, 0.3).translateX(-2).translateY(-1).translateZ(3));

const structureGroup = new THREE.Group();
const loadsGroup = new THREE.Group();
const boardGroup = new THREE.Group();
const foamGroup = new THREE.Group();
scene.add(structureGroup);
scene.add(loadsGroup);
scene.add(boardGroup);
scene.add(foamGroup);

// Board shape
function buildBoardShape() {{
  // Clear old
  while (boardGroup.children.length) boardGroup.remove(boardGroup.children[0]);
  if (!boardData) return;

  const {{ lx, ly, lz }} = currentDS;
  const cx = lx/2, cy = ly/2;

  const bv = boardData.board_verts;
  const bt = boardData.board_tris;

  let zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < bv.length; i++) {{
    if (bv[i][2] < zMin) zMin = bv[i][2];
    if (bv[i][2] > zMax) zMax = bv[i][2];
  }}
  const boardZCenter = (zMin + zMax) / 2;

  const positions = new Float32Array(bv.length * 3);
  for (let i = 0; i < bv.length; i++) {{
    positions[i*3]   = bv[i][0] - cx;
    positions[i*3+1] = bv[i][1] - cy;
    positions[i*3+2] = bv[i][2] - boardZCenter;
  }}

  const indices = new Uint32Array(bt.length * 3);
  for (let i = 0; i < bt.length; i++) {{
    indices[i*3] = bt[i][0]; indices[i*3+1] = bt[i][1]; indices[i*3+2] = bt[i][2];
  }}

  const surfGeo = new THREE.BufferGeometry();
  surfGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  surfGeo.setIndex(new THREE.BufferAttribute(indices, 1));
  surfGeo.computeVertexNormals();

  boardGroup.add(new THREE.Mesh(surfGeo, new THREE.MeshPhysicalMaterial({{
    color: 0x4488ff, opacity: 0.12, transparent: true, side: THREE.DoubleSide,
    depthWrite: false, roughness: 0.8,
  }})));

  boardGroup.add(new THREE.LineSegments(
    new THREE.WireframeGeometry(surfGeo),
    new THREE.LineBasicMaterial({{ color: 0x4488ff, opacity: 0.08, transparent: true }})
  ));

  // Planform outlines
  const pX = boardData.planform_x, pY = boardData.planform_y;
  const outlineMat = new THREE.LineBasicMaterial({{ color: 0x4488ff, opacity: 0.7, transparent: true }});
  function makeOutlineLoop(zOff) {{
    const geo = new THREE.BufferGeometry();
    const v = new Float32Array(pX.length * 3);
    for (let i = 0; i < pX.length; i++) {{
      v[i*3] = pX[i] - cx; v[i*3+1] = pY[i] - cy; v[i*3+2] = zOff;
    }}
    geo.setAttribute('position', new THREE.BufferAttribute(v, 3));
    return new THREE.LineLoop(geo, outlineMat);
  }}
  boardGroup.add(makeOutlineLoop(lz/2));
  boardGroup.add(makeOutlineLoop(-lz/2));
}}

// Load zones
function buildLoadZones() {{
  while (loadsGroup.children.length) loadsGroup.remove(loadsGroup.children[0]);
  if (!meta || !meta.mast_bounds) return;

  const {{ lx, ly, lz }} = currentDS;
  const cx = lx/2, cy = ly/2, cz = lz/2;

  function addZone(bounds, color, z) {{
    const [xmin, xmax, ymin, ymax] = bounds;
    const w = xmax - xmin, h = ymax - ymin;
    const geo = new THREE.PlaneGeometry(w, h);
    const mat = new THREE.MeshBasicMaterial({{ color, opacity: 0.3, transparent: true, side: THREE.DoubleSide }});
    const plane = new THREE.Mesh(geo, mat);
    plane.position.set((xmin+xmax)/2 - cx, (ymin+ymax)/2 - cy, z - cz);
    loadsGroup.add(plane);
    const edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({{ color, linewidth: 2 }})
    );
    edge.position.copy(plane.position);
    loadsGroup.add(edge);
  }}

  const ffb = meta.front_foot_bounds || meta.foot_bounds;
  const bfb = meta.back_foot_bounds || meta.foot_bounds;
  addZone(ffb, 0xff3366, lz);
  addZone(bfb, 0xff6633, lz);
  addZone(meta.mast_bounds, 0x33ff88, 0);

  // Force arrows
  function addArrow(origin, dir, length, color) {{
    const d = new THREE.Vector3(...dir).normalize();
    const arrow = new THREE.ArrowHelper(d, new THREE.Vector3(...origin), length, color, length*0.25, length*0.12);
    loadsGroup.add(arrow);
  }}

  const mb = meta.mast_bounds;
  const mastCenter = [(mb[0]+mb[1])/2 - cx, (mb[2]+mb[3])/2 - cy, lz - cz];
  if (meta.load_cases && meta.load_cases[0]) {{
    const mf = meta.load_cases[0].mast_force;
    addArrow(mastCenter, mf, 0.15, 0xff33ff);
  }}

  function addFootArrows(bounds, color) {{
    for (let xi = 0; xi < 2; xi++) {{
      const x = bounds[0] + (bounds[1]-bounds[0]) * (xi+0.5)/2 - cx;
      const y = (bounds[2]+bounds[3])/2 - cy;
      addArrow([x, y, lz - cz + 0.06], [0,0,-1], 0.06, color);
    }}
  }}
  addFootArrows(ffb, 0xff3366);
  addFootArrows(bfb, 0xff6633);

  // Fixed BC cubes
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
}}

// Voxel rendering
let instancedMesh = null;
let foamMesh = null;
let xrayMode = false;
let showFoam = true;

function buildStructure(threshold, clipXPct, clipZPct) {{
  if (instancedMesh) structureGroup.remove(instancedMesh);
  if (foamMesh) foamGroup.remove(foamMesh);
  instancedMesh = null;
  foamMesh = null;

  const {{ nelx, nely, nelz, lx, ly, lz, density, strainEnergy }} = currentDS;
  const dx = lx/nelx, dy = ly/nely, dz = lz/nelz;
  const cx = lx/2, cy = ly/2, cz = lz/2;
  const clipX = (clipXPct / 100) * nelx;
  const clipZ = (clipZPct / 100) * nelz;

  const VOID_THRESH = 0.01;
  let count = 0, foamCount = 0;
  for (let k = 0; k < nelz; k++) {{
    for (let j = 0; j < nely; j++) {{
      for (let i = 0; i < nelx; i++) {{
        if (i >= clipX || k >= clipZ) continue;
        const d = density[k * nely * nelx + j * nelx + i];
        if (d >= threshold) count++;
        else if (d >= VOID_THRESH) foamCount++;
      }}
    }}
  }}

  // Update stats
  const totalVisible = count + foamCount;
  document.getElementById('voxel-stats').innerHTML =
    `${{count.toLocaleString()}} structural + ${{foamCount.toLocaleString()}} foam voxels visible`;

  // Structural voxels
  if (count > 0) {{
    const voxelGeo = new THREE.BoxGeometry(dx*0.95, dy*0.95, dz*0.95);
    const voxelMat = new THREE.MeshPhysicalMaterial({{
      vertexColors: true, metalness: 0.05, roughness: 0.6, clearcoat: 0.2,
      transparent: xrayMode, opacity: xrayMode ? 0.3 : 1.0,
      depthWrite: !xrayMode,
    }});
    const iMesh = new THREE.InstancedMesh(voxelGeo, voxelMat, count);
    iMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const colorAttr = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    const matrix = new THREE.Matrix4();
    let vi = 0;

    for (let k = 0; k < nelz; k++) {{
      for (let j = 0; j < nely; j++) {{
        for (let i = 0; i < nelx; i++) {{
          if (i >= clipX || k >= clipZ) continue;
          const idx = k * nely * nelx + j * nelx + i;
          const d = density[idx];
          if (d < threshold) continue;

          matrix.makeTranslation((i+0.5)*dx - cx, (j+0.5)*dy - cy, (k+0.5)*dz - cz);
          iMesh.setMatrixAt(vi, matrix);

          if (activeColorMode === 'density') {{
            const t = Math.max(0, Math.min(1, (d - threshold) / (1 - threshold)));
            colorAttr.setXYZ(vi, 0.9 + 0.1*t, 0.82 - 0.6*t, 0.39 - 0.35*t);
          }} else {{
            // Strain energy heatmap
            const seArr = strainEnergy[activeColorMode];
            const se = seArr ? seArr[idx] : 0;
            const seMax = seArr ? seArr._max || 1 : 1;
            const s = Math.max(0, Math.min(1, Math.sqrt(se / seMax)));
            if (s < 0.25) {{
              colorAttr.setXYZ(vi, 0.1, 0.2 + 2.4*s, 0.6 + 1.6*s);
            }} else if (s < 0.5) {{
              const t2 = (s - 0.25) * 4;
              colorAttr.setXYZ(vi, 0.1 + 0.8*t2, 0.8, 1.0 - 0.6*t2);
            }} else if (s < 0.75) {{
              const t2 = (s - 0.5) * 4;
              colorAttr.setXYZ(vi, 0.9, 0.8 - 0.4*t2, 0.4 - 0.3*t2);
            }} else {{
              const t2 = (s - 0.75) * 4;
              colorAttr.setXYZ(vi, 0.9 + 0.1*t2, 0.4 - 0.35*t2, 0.1 - 0.05*t2);
            }}
          }}
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

  // Foam voxels
  if (foamCount > 0) {{
    const foamGeo = new THREE.BoxGeometry(dx*0.95, dy*0.95, dz*0.95);
    const foamMat = new THREE.MeshPhysicalMaterial({{
      color: 0xddeeff, opacity: 0.08, transparent: true, depthWrite: false, roughness: 0.95,
    }});
    const fMesh = new THREE.InstancedMesh(foamGeo, foamMat, foamCount);
    fMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const matrix = new THREE.Matrix4();
    let fi = 0;

    for (let k = 0; k < nelz; k++) {{
      for (let j = 0; j < nely; j++) {{
        for (let i = 0; i < nelx; i++) {{
          if (i >= clipX || k >= clipZ) continue;
          const idx = k * nely * nelx + j * nelx + i;
          const d = density[idx];
          if (d >= threshold || d < VOID_THRESH) continue;

          matrix.makeTranslation((i+0.5)*dx - cx, (j+0.5)*dy - cy, (k+0.5)*dz - cz);
          fMesh.setMatrixAt(fi, matrix);
          fi++;
        }}
      }}
    }}
    fMesh.instanceMatrix.needsUpdate = true;
    foamMesh = fMesh;
    foamGroup.add(fMesh);
  }}

  foamGroup.visible = showFoam;
}}

// Switch dataset
function switchDataset(index) {{
  currentDS = decoded[index];
  document.getElementById('dataset-info').innerHTML = currentDS.info;

  // Update color mode buttons: disable strain energy buttons if no SE data
  const seNames = ['riding_normal', 'pumping', 'jump_landing', 'carving'];
  const seBtns = ['btn-lc-riding', 'btn-lc-pumping', 'btn-lc-landing', 'btn-lc-carving'];
  for (let i = 0; i < seNames.length; i++) {{
    const btn = document.getElementById(seBtns[i]);
    if (btn) {{
      const hasSE = currentDS.strainEnergy[seNames[i]] !== undefined;
      btn.style.opacity = hasSE ? '1' : '0.3';
      btn.disabled = !hasSE;
    }}
  }}

  // Reset to density mode if current mode has no data
  if (activeColorMode !== 'density' && !currentDS.strainEnergy[activeColorMode]) {{
    activeColorMode = 'density';
    document.querySelectorAll('.btn-row .btn').forEach(b => b.classList.remove('active'));
    document.getElementById('btn-density').classList.add('active');
  }}

  buildBoardShape();
  buildLoadZones();
  buildStructure(currentThresh, currentClipX, currentClipZ);
}}

// Build dataset selector
const select = document.getElementById('dataset-select');
decoded.forEach((ds, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = ds.name;
  select.appendChild(opt);
}});
select.addEventListener('change', () => switchDataset(parseInt(select.value)));

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(0, 0, 0);

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

// Toggle buttons
document.getElementById('btn-loads').addEventListener('click', e => {{
  e.target.classList.toggle('active');
  loadsGroup.visible = e.target.classList.contains('active');
}});
document.getElementById('btn-board').addEventListener('click', e => {{
  e.target.classList.toggle('active');
  boardGroup.visible = e.target.classList.contains('active');
}});
document.getElementById('btn-foam').addEventListener('click', e => {{
  showFoam = !showFoam;
  e.target.classList.toggle('active');
  foamGroup.visible = showFoam;
}});
document.getElementById('btn-xray').addEventListener('click', e => {{
  xrayMode = !xrayMode;
  e.target.classList.toggle('active');
  buildStructure(currentThresh, currentClipX, currentClipZ);
}});

// Color mode buttons
const lcMap = {{
  'btn-density': 'density',
  'btn-lc-riding': 'riding_normal',
  'btn-lc-pumping': 'pumping',
  'btn-lc-landing': 'jump_landing',
  'btn-lc-carving': 'carving',
}};
for (const [btnId, lcName] of Object.entries(lcMap)) {{
  const btn = document.getElementById(btnId);
  if (!btn) continue;
  btn.addEventListener('click', () => {{
    if (btn.disabled) return;
    for (const id of Object.keys(lcMap)) {{
      const b = document.getElementById(id);
      if (b) b.classList.remove('active');
    }}
    btn.classList.add('active');
    activeColorMode = lcName;
    buildStructure(currentThresh, currentClipX, currentClipZ);
  }});
}}

// Keyboard shortcuts for dataset switching
document.addEventListener('keydown', e => {{
  if (e.key >= '1' && e.key <= '9') {{
    const idx = parseInt(e.key) - 1;
    if (idx < decoded.length) {{
      select.value = idx;
      switchDataset(idx);
    }}
  }}
}});

// Initial build
buildBoardShape();
buildLoadZones();
buildStructure(0.3, 100, 100);
document.getElementById('dataset-info').innerHTML = currentDS.info;

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
    print(f"Comparison viewer: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build structure comparison viewer")
    parser.add_argument("-o", "--output", default="comparison.html")
    args = parser.parse_args()

    datasets = []

    # 1. Phase 1 coarse SIMP
    p1_dir = "results/modal_bulkhead5"
    if os.path.exists(os.path.join(p1_dir, "density.bin")):
        print("Loading Phase 1...")
        datasets.append(load_phase1(p1_dir))
    else:
        print(f"Warning: {p1_dir}/density.bin not found, skipping Phase 1")

    # 2. Fullboard assembled volume (fine detail)
    vol_path = "results/structure_fullboard.npy"
    if os.path.exists(vol_path):
        print("Loading fullboard volume (fine)...")
        datasets.append(load_fullboard_volume(vol_path, target_nx=80))

        # 3. Fullboard resampled to Phase 1 mesh (for direct comparison)
        if datasets and datasets[0].get("meta"):
            print("Loading fullboard → Phase 1 mesh...")
            datasets.append(load_fullboard_at_phase1_res(vol_path, datasets[0]["meta"]))

    # 4. Internal only (shell stripped) — the "3D printed core" view — FULL RES
    cont_vol = "results/board_continuous_volume.npy"
    if os.path.exists(cont_vol):
        print("Loading internal-only (shell stripped, full resolution)...")
        internal = load_internal_only_volume(cont_vol, target_nx=999)  # no downsampling
        if internal:
            datasets.append(internal)

    # 5. Discrete bulkheads (plate mode) — full res
    bh_vol = "results/board_volume.npy"
    if os.path.exists(bh_vol):
        print("Loading discrete bulkheads (full resolution)...")
        discrete = load_discrete_bulkhead_volume(bh_vol, target_nx=999)
        if discrete:
            datasets.append(discrete)

    # 6. Fullboard cross-sections (assembled slices)
    fb_cs_dir = "results/cross_sections_fullboard"
    if os.path.exists(os.path.join(fb_cs_dir, "summary.json")):
        print("Loading fullboard cross-sections...")
        fb_cs = load_fullboard_cross_sections(fb_cs_dir)
        if fb_cs:
            datasets.append(fb_cs)

    # 7. Hires cross-sections
    hr_dir = "results/cross_sections_hires"
    if os.path.exists(os.path.join(hr_dir, "summary.json")):
        print("Loading hires cross-sections...")
        hr = load_hires_cross_sections(hr_dir)
        if hr:
            datasets.append(hr)

    if not datasets:
        print("No datasets found!")
        sys.exit(1)

    print(f"\nBuilding comparison viewer with {len(datasets)} datasets...")
    for i, ds in enumerate(datasets):
        print(f"  [{i+1}] {ds['name']}: {ds['info']}")

    board_shape = parse_s3dx_for_viewer()
    build_comparison_html(datasets, board_shape, args.output)
    print(f"\nOpen {args.output} in browser. Press 1-{len(datasets)} to switch datasets.")


if __name__ == "__main__":
    main()
