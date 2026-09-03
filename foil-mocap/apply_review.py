"""
Re-run stance_analysis.py according to stance/review.json.

For each reviewed clip:
  relevant == "no"      -> its output JSON is moved to stance/output/excluded/ so the
                           report and the browser app stop using it
  relevant == "yes"     -> re-analysed on the chosen start/end window (seconds on the
                           overlay video plus its window_offset) with the chosen group, keeping the original label,
                           source and note
  "unsure" or unmarked  -> left as is

Then rebuilds the report and the browser app's reference data.

    python apply_review.py            # apply everything
    python apply_review.py --dry-run  # just print what would happen
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "stance" / "output"
REVIEW = HERE / "stance" / "review.json"
EXCLUDED = OUT / "excluded"


def main():
    dry = "--dry-run" in sys.argv
    review = json.load(open(REVIEW)) if REVIEW.exists() else {}
    if not review:
        print("no review.json yet"); return
    for stem, r in review.items():
        jpath = OUT / f"{stem}_stance.json"
        if not jpath.exists():
            print(f"skip {stem}: no analysed output"); continue
        s = json.load(open(jpath))["summary"]
        clip = HERE / "samples" / "input" / s["clip"] if stem == "slowmo_pump" else HERE / "stance" / "input" / s["clip"]
        if r.get("relevant") == "no":
            print(f"exclude {stem}")
            if not dry:
                EXCLUDED.mkdir(exist_ok=True)
                for f in OUT.glob(f"{stem}_*"):
                    shutil.move(str(f), EXCLUDED / f.name)
            continue
        if r.get("relevant") != "yes":
            print(f"leave  {stem} ({r.get('relevant') or 'unmarked'})"); continue
        off = r.get("window_offset") or 0
        start = None if r.get("start") is None else r["start"] + off
        end = None if r.get("end") is None else r["end"] + off
        if start is None and off: start = off
        if end is None and s.get("end"): end = s["end"]
        group = r.get("group") or s["group"]
        same_window = (start is None or abs(start - (s.get("start") or 0)) < 0.05) and (end is None or (s.get("end") and abs(end - s["end"]) < 0.05))
        if same_window and group == s["group"]:
            print(f"keep   {stem} (window and group unchanged)")
            if r.get("note") and not dry:
                d = json.load(open(jpath)); d["summary"]["note"] = (s.get("note", "") + " Review: " + r["note"]).strip(); json.dump(d, open(jpath, "w"))
            continue
        cmd = [sys.executable, str(HERE / "stance_analysis.py"), str(clip), "-o", str(OUT), "--label", s["label"], "--group", group,
               "--source", s.get("source", ""), "--note", (s.get("note", "") + (" Review: " + r["note"] if r.get("note") else "")).strip()]
        if start is not None: cmd += ["--start", str(start)]
        if end is not None: cmd += ["--end", str(end)]
        if s.get("front_foot") and r.get("front"): cmd += ["--front", r["front"]]
        print(f"rerun  {stem}: {start} to {end} s as {group}")
        if not dry:
            subprocess.run(cmd, check=True)
            # The overlay video now starts at the new window, so re-base the review entry onto it.
            r["window_offset"] = start or 0
            r["start"] = 0.0 if start is not None else None
            r["end"] = (end - (start or 0)) if end is not None else None
            REVIEW.write_text(json.dumps(review, indent=1))
    if not dry:
        subprocess.run([sys.executable, str(HERE / "build_stance_report.py"), "--web"], check=True)


if __name__ == "__main__":
    main()
