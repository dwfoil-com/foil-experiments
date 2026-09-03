"""
Stance analysis for pump foiling: pro vs beginner foot placement.

Runs MediaPipe Pose on a clip and extracts stance metrics that separate
"back-leg-only" pumping from a balanced, trimmed stance:

  - stance width           ankle-to-ankle distance (world metres, and as a
                           fraction of leg length)
  - front / back knee angle 3D knee angle per leg, per frame
  - knee asymmetry         front knee angle minus back knee angle; large
                           positive = front leg locked straight while the back
                           leg stays bent (the classic beginner tell)
  - leg drive ratio        back-knee range of motion / front-knee range of
                           motion over the pump cycle; >1.5 = pumping mostly
                           with the back leg
  - front knee over ankle  horizontal offset of the front knee ahead of the
                           front ankle along the stance line; negative = foot
                           is in front of the knee (too far forward)
  - hip fraction           where the hips sit between the back ankle (0) and
                           the front ankle (1); a proxy for weight split
  - pump frequency         from hip vertical oscillation in image space

All angular/width metrics use MediaPipe *world* landmarks (metres, hip-centred)
so they are view-independent; a frontal clip and a side clip are comparable.

The mast is usually not visible, so mast offset is not measured directly. Use
--mast-x / --nose-x / --tail-x on a reference frame to record foot positions
as a fraction of board length for clips where the board is visible side-on.

Outputs (in --output dir, prefixed by clip stem):
  <stem>_stance.mp4        skeleton + HUD overlay
  <stem>_stance.json       summary metrics + time series
  <stem>_contact.jpg       6-frame contact sheet of the overlay
"""
import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

L = dict(nose=0, l_sh=11, r_sh=12, l_hip=23, r_hip=24, l_knee=25, r_knee=26,
         l_ank=27, r_ank=28, l_heel=29, r_heel=30, l_toe=31, r_toe=32)

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (27, 31),
            (28, 30), (28, 32), (29, 31), (30, 32)]


def angle_3d(a, b, c):
    """Angle at b (degrees) between vectors b->a and b->c."""
    v1, v2 = a - b, c - b
    cosang = np.sum(v1 * v2, axis=-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


class CropTracker:
    """Keeps a crop window around the rider so small figures get enough pixels."""

    def __init__(self, width, height, target_h=720, margin=1.8):
        self.w, self.h = width, height
        self.target_h = target_h
        self.margin = margin
        self.box = None  # (cx, cy, size) in full-frame pixels

    def window(self):
        if self.box is None:
            return None
        cx, cy, size = self.box
        # A rider who already fills a third of the frame gains nothing from cropping,
        # and the moving window makes the tracking flicker, so only crop small figures.
        if size > 0.2 * self.h:
            return None
        half = size * self.margin / 2
        x0 = int(max(0, cx - half)); y0 = int(max(0, cy - half))
        x1 = int(min(self.w, cx + half)); y1 = int(min(self.h, cy + half))
        if x1 - x0 < 64 or y1 - y0 < 64:
            return None
        return x0, y0, x1, y1

    def update(self, px):
        xs, ys = px[:, 0], px[:, 1]
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        size = max(xs.max() - xs.min(), ys.max() - ys.min())
        if self.box is None:
            self.box = (cx, cy, size)
            return
        # Sticky window: only move when the rider drifts well off centre or changes
        # size a lot, so the crop stays fixed and MediaPipe's tracking is not upset.
        ocx, ocy, osz = self.box
        if abs(cx - ocx) > 0.2 * osz or abs(cy - ocy) > 0.2 * osz or size > 1.3 * osz or size < 0.7 * osz:
            self.box = (cx, cy, size)

    def lost(self):
        self.box = None


def run_pose(video_path, model_path, start=0.0, end=None, autocrop=True):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(start * fps); f1 = int(end * fps) if end else n_total
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path), delegate=mp_python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4)
    lm = vision.PoseLandmarker.create_from_options(options)
    tracker = CropTracker(W, H)

    frames, px_all, world_all, vis_all = [], [], [], []
    idx = f0
    while idx < f1:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        win = tracker.window() if autocrop else None
        if win:
            x0, y0, x1, y1 = win
            crop = rgb[y0:y1, x0:x1]
            scale = tracker.target_h / crop.shape[0]
            if scale > 1.05:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                scale = 1.0
            img = np.ascontiguousarray(crop)
        else:
            x0 = y0 = 0; scale = 1.0; img = rgb
        ts = int((idx - f0) * 1000 / fps)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), ts)
        if not res.pose_landmarks and win:
            # Lost inside the crop: retry on the full frame before giving up on it.
            x0 = y0 = 0; scale = 1.0; img = rgb
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), ts + 1)
        if res.pose_landmarks:
            p = res.pose_landmarks[0]; w = res.pose_world_landmarks[0]
            px = np.array([[q.x * img.shape[1] / scale + x0, q.y * img.shape[0] / scale + y0] for q in p])
            world = np.array([[q.x, q.y, q.z] for q in w])
            vis = np.array([q.visibility for q in p])
            tracker.update(px)
        else:
            px = np.full((33, 2), np.nan); world = np.full((33, 3), np.nan); vis = np.zeros(33)
            tracker.lost()
        frames.append(frame); px_all.append(px); world_all.append(world); vis_all.append(vis)
        idx += 1
        if (idx - f0) % 100 == 0:
            print(f"  frame {idx - f0}/{f1 - f0}")
    cap.release(); lm.close()
    return fps, W, H, frames, np.array(px_all), np.array(world_all), np.array(vis_all)


def smooth(arr, fps):
    """Savitzky-Golay along axis 0, NaN-tolerant via interpolation."""
    out = arr.copy()
    n = len(arr)
    win = max(5, int(fps * 0.15) | 1)
    if n < win + 2:
        return out
    flat = out.reshape(n, -1)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        good = ~np.isnan(col)
        if good.sum() < win:
            continue
        col_i = np.interp(np.arange(n), np.flatnonzero(good), col[good])
        flat[:, j] = savgol_filter(col_i, win, 3)
        flat[~good, j] = np.nan
    return flat.reshape(arr.shape)


def compute_metrics(fps, px, world, vis, front_override="auto"):
    n = len(px)
    valid = ~np.isnan(world[:, 0, 0])
    world = smooth(world, fps); px = smooth(px, fps)

    la, ra = world[:, L["l_ank"]], world[:, L["r_ank"]]
    lk, rk = world[:, L["l_knee"]], world[:, L["r_knee"]]
    lh, rh = world[:, L["l_hip"]], world[:, L["r_hip"]]
    hip_c = (lh + rh) / 2
    nose = world[:, L["nose"]]
    sh_c = (world[:, L["l_sh"]] + world[:, L["r_sh"]]) / 2

    leg_len = (np.linalg.norm(lk - lh, axis=1) + np.linalg.norm(la - lk, axis=1) +
               np.linalg.norm(rk - rh, axis=1) + np.linalg.norm(ra - rk, axis=1)) / 2
    shoulder_w = np.linalg.norm(world[:, L["l_sh"]] - world[:, L["r_sh"]], axis=1)

    # Stance axis: horizontal (x,z) direction from one ankle to the other.
    horiz = np.array([1, 0, 1])
    axis = (la - ra) * horiz
    axis_len = np.linalg.norm(axis, axis=1) + 1e-9
    axis_u = axis / axis_len[:, None]

    # Which foot is front? Nose (head) leads hips in the direction of travel.
    head_lead = np.nansum(((nose - hip_c) * horiz) * axis_u, axis=1)
    if front_override == "left":
        left_front = True
    elif front_override == "right":
        left_front = False
    else:
        left_front = np.nanmedian(head_lead) > 0
    if left_front:
        fa, fk, fh, ba, bk, bh = la, lk, lh, ra, rk, rh
    else:
        fa, fk, fh, ba, bk, bh = ra, rk, rh, la, lk, lh
    fwd = ((fa - ba) * horiz); fwd /= (np.linalg.norm(fwd, axis=1)[:, None] + 1e-9)

    stance_w = np.linalg.norm((fa - ba) * horiz, axis=1)
    front_knee = angle_3d(fh, fk, fa)
    back_knee = angle_3d(bh, bk, ba)
    knee_over_ankle = np.sum(((fk - fa) * horiz) * fwd, axis=1)
    hip_frac = np.sum(((hip_c - ba) * horiz) * fwd, axis=1) / (stance_w + 1e-9)
    torso_lean = np.degrees(np.arctan2(np.sum(((sh_c - hip_c) * horiz) * fwd, axis=1),
                                       -(sh_c - hip_c)[:, 1]))

    # Pump cycle from image-space hip height (world coords are hip-centred).
    hip_y_px = (px[:, L["l_hip"], 1] + px[:, L["r_hip"], 1]) / 2
    leg_px = (np.linalg.norm(px[:, L["l_hip"]] - px[:, L["l_ank"]], axis=1) +
              np.linalg.norm(px[:, L["r_hip"]] - px[:, L["r_ank"]], axis=1)) / 2
    hip_y_rel = -(hip_y_px - np.nanmean(hip_y_px)) / (np.nanmean(leg_px) + 1e-9)
    hy = np.where(np.isnan(hip_y_rel), 0, hip_y_rel)
    peaks, _ = find_peaks(hy, distance=int(fps * 0.35), prominence=0.03)
    freq = (len(peaks) - 1) / ((peaks[-1] - peaks[0]) / fps) if len(peaks) > 2 else 0.0

    def rom(sig):
        s = sig[~np.isnan(sig)]
        return float(np.percentile(s, 90) - np.percentile(s, 10)) if len(s) > 10 else float("nan")

    def med(sig):
        return float(np.nanmedian(sig))

    front_rom, back_rom = rom(front_knee), rom(back_knee)
    summary = dict(
        frames_tracked=int(valid.sum()), frames_total=int(n),
        front_foot="left" if left_front else "right",
        stance_width_m=med(stance_w),
        stance_width_over_leg=med(stance_w / leg_len),
        stance_width_over_shoulder=med(stance_w / shoulder_w),
        front_knee_deg=med(front_knee), back_knee_deg=med(back_knee),
        knee_asymmetry_deg=med(front_knee - back_knee),
        front_knee_rom_deg=front_rom, back_knee_rom_deg=back_rom,
        leg_drive_ratio=float(back_rom / front_rom) if front_rom and front_rom > 1 else float("nan"),
        front_knee_over_ankle_m=med(knee_over_ankle),
        hip_fraction=med(hip_frac),
        torso_lean_deg=med(torso_lean),
        pump_freq_hz=float(freq), pump_cycles=int(len(peaks)),
        hip_amplitude_over_leg=rom(hip_y_rel),
    )
    series = dict(
        t=(np.arange(n) / fps).round(3).tolist(),
        front_knee=np.round(front_knee, 1).tolist(), back_knee=np.round(back_knee, 1).tolist(),
        stance_width=np.round(stance_w, 3).tolist(), hip_fraction=np.round(hip_frac, 3).tolist(),
        knee_over_ankle=np.round(knee_over_ankle, 3).tolist(), hip_y_rel=np.round(hip_y_rel, 3).tolist(),
        tracked=valid.tolist(),
    )
    per_frame = dict(front_knee=front_knee, back_knee=back_knee, stance_w=stance_w, hip_frac=hip_frac,
                     knee_over_ankle=knee_over_ankle, left_front=left_front)
    return summary, series, per_frame, px


def render(frames, px, per_frame, fps, out_path, label, contact_path):
    H, W = frames[0].shape[:2]
    scale = max(1.0, 720 / H)
    OW, OH = int(W * scale), int(H * scale)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OW, OH))
    trace_f, trace_b = deque(maxlen=int(fps * 2)), deque(maxlen=int(fps * 2))
    lf = per_frame["left_front"]
    fi, bi = (L["l_ank"], L["r_ank"]) if lf else (L["r_ank"], L["l_ank"])
    fk, bk = (L["l_knee"], L["r_knee"]) if lf else (L["r_knee"], L["l_knee"])
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, OH / 1400)
    contact_idx = set(np.linspace(0, len(frames) - 1, 6).astype(int).tolist())
    contact = []
    for i, frame in enumerate(frames):
        img = cv2.resize(frame, (OW, OH)) if scale != 1.0 else frame.copy()
        p = px[i] * scale
        if not np.isnan(p[0, 0]):
            for a, b in SKELETON:
                cv2.line(img, tuple(p[a].astype(int)), tuple(p[b].astype(int)), (0, 255, 0), 2)
            for j in range(33):
                cv2.circle(img, tuple(p[j].astype(int)), 3, (0, 0, 255), -1)
            # Front leg red, back leg cyan
            for hip, knee, ank, col in ((L["l_hip"] if lf else L["r_hip"], fk, fi, (0, 0, 255)),
                                        (L["r_hip"] if lf else L["l_hip"], bk, bi, (255, 255, 0))):
                cv2.line(img, tuple(p[hip].astype(int)), tuple(p[knee].astype(int)), col, 4)
                cv2.line(img, tuple(p[knee].astype(int)), tuple(p[ank].astype(int)), col, 4)
            trace_f.append(tuple(p[fi].astype(int))); trace_b.append(tuple(p[bi].astype(int)))
            for tr, col in ((trace_f, (0, 0, 255)), (trace_b, (255, 255, 0))):
                pts = list(tr)
                for k in range(1, len(pts)):
                    cv2.line(img, pts[k - 1], pts[k], col, 1)
            fkd, bkd = per_frame["front_knee"][i], per_frame["back_knee"][i]
            sw, hf, ko = per_frame["stance_w"][i], per_frame["hip_frac"][i], per_frame["knee_over_ankle"][i]
            lines = [f"front knee {fkd:5.0f}   back knee {bkd:5.0f}",
                     f"stance {sw*100:4.0f} cm   hips {hf:4.2f} (0=back 1=front)",
                     f"front knee ahead of ankle {ko*100:+4.0f} cm"]
        else:
            lines = ["no pose"]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (OW, int(OH * 0.11) + 10), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        cv2.putText(img, label, (10, int(28 * fs) + 4), font, fs * 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        for k, t in enumerate(lines):
            cv2.putText(img, t, (10, int((28 * fs) * (k + 2)) + 4), font, fs * 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
        if i in contact_idx:
            contact.append(cv2.resize(img, (int(OW * 360 / OH), 360)))
    vw.release()
    if contact:
        cv2.imwrite(str(contact_path), np.hstack(contact))
    # Re-encode to h264 so browsers can play it.
    import subprocess
    tmp = out_path.with_suffix(".tmp.mp4")
    out_path.rename(tmp)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-movflags", "+faststart", str(out_path)], check=True)
    tmp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default=str(Path(__file__).parent / "pose_landmarker_full.task"))
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--group", default="unknown", help="pro | beginner | reference")
    ap.add_argument("--front", default="auto", choices=["auto", "left", "right"])
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--source", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-autocrop", action="store_true")
    args = ap.parse_args()

    video = Path(args.video)
    out = Path(args.output) if args.output else video.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    label = args.label or stem
    print(f"Processing {video.name}")
    fps, W, H, frames, px, world, vis = run_pose(video, args.model, args.start, args.end, not args.no_autocrop)
    summary, series, per_frame, px_s = compute_metrics(fps, px, world, vis, args.front)
    summary.update(dict(label=label, group=args.group, source=args.source, note=args.note, clip=video.name,
                        fps=float(fps), width=W, height=H, start=args.start, end=args.end))
    render(frames, px_s, per_frame, fps, out / f"{stem}_stance.mp4", label, out / f"{stem}_contact.jpg")
    with open(out / f"{stem}_stance.json", "w") as f:
        json.dump(dict(summary=summary, series=series), f)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
