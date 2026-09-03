"""
Compile the key clips into one shareable video: for each clip, the tracked
overlay on the left and the hips-pinned skeleton on the right, with a title
strip, beginners first, then pros. Output: stance/export/pump_stance_review.mp4

    python export_reel.py [--max-seconds 8] [--height 720]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_stance_report import derived

HERE = Path(__file__).parent
OUT = HERE / "stance" / "output"
EXPORT = HERE / "stance" / "export"
REVIEW = HERE / "stance" / "review.json"
ORDER = {"beginner": 0, "pro": 1, "reference": 2}
FONT = "/System/Library/Fonts/Helvetica.ttc"


def best_window(series, seconds):
    """Start time of the window with the most tracked frames, so no segment opens on 'no pose'."""
    t = series["t"]
    ok = [1 if (a is not None and b is not None and a == a and b == b) else 0
          for a, b in zip(series["front_knee"], series["back_knee"])]
    n = len(t)
    if n < 2:
        return 0.0
    fps = (n - 1) / max(1e-6, t[-1] - t[0])
    w = min(int(seconds * fps), max(1, n // 2))
    run = sum(ok[:w]); best, best_i = run, 0
    for i in range(1, n - w):
        run += ok[i + w - 1] - ok[i - 1]
        if run > best:
            best, best_i = run, i
    return t[best_i] - t[0]


def esc(t):
    return t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=8)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--include-reference", action="store_true")
    args = ap.parse_args()
    EXPORT.mkdir(exist_ok=True)
    review = json.load(open(REVIEW)) if REVIEW.exists() else {}
    clips = []
    for p in sorted(OUT.glob("*_stance.json")):
        s = json.load(open(p))["summary"]; stem = p.name.replace("_stance.json", "")
        rv = review.get(stem, {})
        if rv.get("relevant") in ("no", "unsure"):
            continue
        if s["group"] == "reference" and not args.include_reference:
            continue
        if 100 * s["frames_tracked"] / max(1, s["frames_total"]) < 60:
            continue
        if not (OUT / f"{stem}_skel.mp4").exists():
            continue
        clips.append((ORDER.get(s["group"], 9), s["label"], stem, s))
    clips.sort()
    H = args.height
    W = int(H * 16 / 9) // 2 * 2
    segs = []
    for _, label, stem, s in clips:
        seg = EXPORT / f"seg_{stem}.mp4"
        d = json.load(open(OUT / f"{stem}_stance.json"))
        x = derived(d["series"])
        asym, drive = x["asym_at_bottom_deg"], s.get("leg_drive_ratio") or float("nan")
        ss = best_window(d["series"], args.max_seconds)
        strip = f"{s['group'].upper()}  {label}   back knee deeper by {asym:.0f} deg at bottom of pump   back/front knee movement {drive:.2f}x"
        # Scale onto the final canvas before the strip, so the text is never cut off by a narrow segment.
        vf = (f"[0:v]scale=-2:{H}[a];[1:v]scale=-2:{H}[b];[a][b]hstack=inputs=2[v];"
              f"[v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a19,setsar=1[p];"
              f"[p]pad={W}:{H + 56}:0:56:color=0x1a1a19,drawtext=fontfile={FONT}:text='{esc(strip)}':fontcolor=white:fontsize=22:x=16:y=16[out]")
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{ss:.2f}", "-i", str(OUT / f"{stem}_stance.mp4"),
               "-ss", f"{ss:.2f}", "-i", str(OUT / f"{stem}_skel.mp4"),
               "-filter_complex", vf, "-map", "[out]", "-t", str(args.max_seconds), "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-an", str(seg)]
        subprocess.run(cmd, check=True)
        segs.append(seg)
        print(f"segment {len(segs):2d} {s['group']:<9} {label:<34} from {ss:5.1f} s")
    inputs = []; filt = []
    for i, seg in enumerate(segs):
        inputs += ["-i", str(seg)]
        filt.append(f"[{i}:v]setsar=1[s{i}]")
    filt.append("".join(f"[s{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[out]")
    final = EXPORT / "pump_stance_review.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filt), "-map", "[out]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", str(final)], check=True)
    for seg in segs:
        seg.unlink()
    print(f"wrote {final} ({len(segs)} clips, up to {args.max_seconds:.0f} s each)")


if __name__ == "__main__":
    main()
