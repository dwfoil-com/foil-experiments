"""
Build stance/report.html from the *_stance.json files produced by
stance_analysis.py. Self-contained HTML; videos and contact sheets are linked
relatively so the page works from file://.
"""
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent / "stance"
OUT = ROOT / "output"
GROUP_ORDER = {"beginner": 0, "pro": 1, "reference": 2}
GROUP_COL = {"beginner": "var(--s2)", "pro": "var(--s1)", "reference": "var(--s3)"}

METRICS = [
    ("asym_at_bottom_deg", "Knee asymmetry at bottom of pump", "deg", "front knee minus back knee in the frames where the back knee is most bent. This is the moment the forum screenshots show: back leg squatting, front leg still straight."),
    ("leg_drive_ratio", "Leg drive ratio", "x", "back knee range of motion divided by front knee range of motion across the pump cycle. 1.0 = both legs working equally. Above 1.5 = mostly back leg."),
    ("front_knee_p10", "Front knee, deepest bend", "deg", "10th percentile front knee angle. If this stays above about 150 the front leg never really loads."),
    ("knee_asymmetry_deg", "Knee asymmetry", "deg", "front knee angle minus back knee angle. Positive = front leg straighter than back. The forum tell for back-leg pumping is a locked front leg over a bent back leg."),
    ("stance_width_over_leg", "Stance width", "leg lengths", "ankle to ankle, as a fraction of leg length. Forum guidance is shoulder width or narrower."),
    ("front_knee_over_ankle_m", "Front knee ahead of ankle", "m", "how far the front knee sits ahead of the front ankle along the stance line. Negative = the foot is out in front of the knee."),
    ("hip_fraction", "Hip position", "0=back 1=front", "where the hips sit between the back ankle and the front ankle. A proxy for weight split between the feet."),
    ("front_knee_deg", "Front knee angle", "deg", "median front knee angle. 180 = locked straight."),
    ("back_knee_deg", "Back knee angle", "deg", "median back knee angle."),
    ("pump_freq_hz", "Pump frequency", "Hz", "from hip vertical oscillation. Camera motion leaks into this on handheld clips."),
]


# Manual pixel reads from side-on frames where board and mast are visible.
# (clip stem, time s, board length px, front foot px ahead of mast, back foot px ahead of mast)
MANUAL_BOARD = [
    ("bad_race_fatigue_ig", "Beginner B (race fatigue)", 9.0, 330, 85, 0),
    ("bad_race_fatigue_ig", "Beginner B (race fatigue)", 14.0, 245, 55, 5),
    ("good_kane_trimmed_yt", "Kane (fully trimmed)", 20.0, 165, 35, 5),
    ("good_kane_trimmed_yt", "Kane (fully trimmed)", 40.0, 140, 40, 5),
    ("good_edo_tanas_ig", "Edo Tanas (pro)", 6.5, 485, 125, 20),
]


def manual_board_table():
    out = ["<table><thead><tr><th>clip</th><th>time</th><th>front foot ahead of mast</th><th>back foot ahead of mast</th><th>stance</th></tr></thead><tbody>"]
    for stem, label, t, L, f, b in MANUAL_BOARD:
        out.append(f"<tr><td><a href='#{stem}'>{html.escape(label)}</a></td><td>{t:.0f}s</td><td>{100*f/L:.0f}% of board length</td>"
                   f"<td>{100*b/L:.0f}% of board length</td><td>{100*(f-b)/L:.0f}% of board length</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def tickfmt(v, span):
    if span >= 20:
        return f"{v:.0f}"
    if span >= 2:
        return f"{v:.1f}"
    return f"{v:.2f}"


def fmt(v, nd=1):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return f"{v:.{nd}f}"


def derived(series):
    """Extra metrics computed from the per-frame knee angles."""
    f = [x for x in series["front_knee"] if x is not None and not math.isnan(x)]
    b = [x for x in series["back_knee"] if x is not None and not math.isnan(x)]
    pairs = [(ff, bb) for ff, bb in zip(series["front_knee"], series["back_knee"])
             if ff is not None and bb is not None and not math.isnan(ff) and not math.isnan(bb)]
    nan = float("nan")
    if len(pairs) < 20:
        return dict(front_knee_p10=nan, back_knee_p10=nan, asym_at_bottom_deg=nan)

    def pct(arr, q):
        a = sorted(arr); return a[min(len(a) - 1, int(q * len(a)))]

    b_cut = pct([bb for _, bb in pairs], 0.2)
    bottom = [ff - bb for ff, bb in pairs if bb <= b_cut]
    return dict(front_knee_p10=pct(f, 0.1), back_knee_p10=pct(b, 0.1),
                asym_at_bottom_deg=pct(bottom, 0.5) if bottom else nan)


REVIEW_PATH = ROOT / "review.json"


REVIEW = {}


def load_review():
    try:
        return json.load(open(REVIEW_PATH))
    except Exception:
        return {}


def load():
    clips = []
    for p in sorted(OUT.glob("*_stance.json")):
        d = json.load(open(p))
        s = d["summary"]
        stem = p.name.replace("_stance.json", "")
        s["stem"] = stem
        s["video_rel"] = f"output/{stem}_stance.mp4"
        s["contact_rel"] = f"output/{stem}_contact.jpg"
        s["tracked_pct"] = 100 * s["frames_tracked"] / max(1, s["frames_total"])
        s.update(derived(d["series"]))
        clips.append((s, d["series"]))
    clips.sort(key=lambda c: (GROUP_ORDER.get(c[0]["group"], 9), c[0]["label"]))
    return clips


def dot_plot(clips, key, title, unit):
    """One-axis dot plot, one row per group, direct-labelled dots."""
    vals = [(s["group"], s["label"], s.get(key)) for s, _ in clips
            if s.get(key) is not None and not (isinstance(s.get(key), float) and math.isnan(s.get(key)))]
    if not vals:
        return ""
    lo = min(v for _, _, v in vals); hi = max(v for _, _, v in vals)
    if key == "leg_drive_ratio":
        lo, hi = min(lo, 0.5), max(hi, 2.0)
    if key == "knee_asymmetry_deg":
        lo, hi = min(lo, -10), max(hi, 10)
    if key == "hip_fraction":
        lo, hi = min(lo, 0.3), max(hi, 0.7)
    pad = (hi - lo) * 0.12 or 1
    lo -= pad; hi += pad
    W, H, left, right = 640, 150, 90, 20
    rows = ["beginner", "pro", "reference"]
    ry = {g: 32 + i * 34 for i, g in enumerate(rows)}

    def sx(v):
        return left + (v - lo) / (hi - lo) * (W - left - right)

    out = [f'<svg class="dot" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    # ticks
    nt = 5
    for i in range(nt + 1):
        v = lo + (hi - lo) * i / nt
        x = sx(v)
        out.append(f'<line x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="{H-30}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{H-12}" class="tick" text-anchor="middle">{tickfmt(v, hi - lo)}</text>')
    if lo < 0 < hi and key in ("knee_asymmetry_deg", "front_knee_over_ankle_m"):
        x = sx(0)
        out.append(f'<line x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="{H-30}" class="zero"/>')
    if key == "leg_drive_ratio":
        x = sx(1.0)
        out.append(f'<line x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="{H-30}" class="zero"/>')
    for g in rows:
        out.append(f'<text x="{left-8}" y="{ry[g]+4}" class="rowlab" text-anchor="end">{g}</text>')
    for g, label, v in vals:
        x = sx(v); y = ry[g]
        out.append(f'<circle cx="{x:.1f}" cy="{y}" r="7" fill="{GROUP_COL.get(g, "#888")}" stroke="var(--surface)" stroke-width="2">'
                   f'<title>{html.escape(label)}: {v:.2f} {unit}</title></circle>')
    out.append(f'<text x="{left}" y="12" class="axis">{html.escape(unit)}</text>')
    out.append("</svg>")
    return "".join(out)


def line_chart(series, cid):
    t = series["t"]; f = series["front_knee"]; b = series["back_knee"]
    pts = [(tt, ff, bb) for tt, ff, bb in zip(t, f, b) if ff is not None and bb is not None
           and not (isinstance(ff, float) and math.isnan(ff))]
    if len(pts) < 5:
        return "<p class='muted'>Not enough tracked frames for a time series.</p>"
    W, H, left, right, top, bot = 640, 200, 44, 12, 14, 26
    t0, t1 = pts[0][0], pts[-1][0]
    ylo, yhi = 60, 185

    def sx(v): return left + (v - t0) / max(1e-6, t1 - t0) * (W - left - right)
    def sy(v): return top + (yhi - min(max(v, ylo), yhi)) / (yhi - ylo) * (H - top - bot)

    def path(idx):
        d = []; pen = False
        prev_t = None
        for p in pts:
            if prev_t is not None and p[0] - prev_t > 0.5:
                pen = False
            d.append(("L" if pen else "M") + f"{sx(p[0]):.1f},{sy(p[idx]):.1f}")
            pen = True; prev_t = p[0]
        return " ".join(d)

    out = [f'<svg class="line" id="{cid}" viewBox="0 0 {W} {H}" data-t="{",".join(f"{p[0]:.2f}" for p in pts)}" '
           f'data-f="{",".join(f"{p[1]:.0f}" for p in pts)}" data-b="{",".join(f"{p[2]:.0f}" for p in pts)}">']
    for yv in (90, 120, 150, 180):
        out.append(f'<line x1="{left}" y1="{sy(yv):.1f}" x2="{W-right}" y2="{sy(yv):.1f}" class="grid"/>'
                   f'<text x="{left-6}" y="{sy(yv)+4:.1f}" class="tick" text-anchor="end">{yv}</text>')
    for i in range(5):
        tv = t0 + (t1 - t0) * i / 4
        out.append(f'<text x="{sx(tv):.1f}" y="{H-8}" class="tick" text-anchor="middle">{tv:.0f}s</text>')
    out.append(f'<path d="{path(1)}" class="front"/>')
    out.append(f'<path d="{path(2)}" class="back"/>')
    out.append(f'<line class="cross" x1="0" y1="{top}" x2="0" y2="{H-bot}" style="display:none"/>')
    out.append(f'<circle class="mf" r="5" style="display:none"/><circle class="mb" r="5" style="display:none"/>')
    out.append(f'<rect class="hit" x="{left}" y="{top}" width="{W-left-right}" height="{H-top-bot}" fill="transparent"/>')
    out.append("</svg>")
    out.append(f'<div class="legend"><span><i class="sw front"></i>front knee</span><span><i class="sw back"></i>back knee</span>'
               f'<span class="tip" id="{cid}-tip"></span></div>')
    return "".join(out)


def group_summary(clips):
    rows = []
    for g in ("beginner", "pro"):
        gs = [s for s, _ in clips if s["group"] == g and s["tracked_pct"] > 40]
        if not gs:
            continue
        def med(k):
            v = sorted(x[k] for x in gs if x.get(k) is not None and not math.isnan(x[k]))
            return v[len(v) // 2] if v else float("nan")
        rows.append((g, len(gs), med("asym_at_bottom_deg"), med("leg_drive_ratio"), med("stance_width_over_leg"),
                     med("front_knee_over_ankle_m") * 100, med("hip_fraction"), med("front_knee_deg"), med("back_knee_deg")))
    out = ["<table><thead><tr><th>group</th><th>clips</th><th>knee asymmetry at bottom (deg)</th><th>leg drive ratio</th>"
           "<th>stance / leg</th><th>front knee ahead of ankle (cm)</th><th>hip fraction</th><th>front knee (deg)</th><th>back knee (deg)</th></tr></thead><tbody>"]
    for r in rows:
        out.append(f"<tr><td><b style='color:{GROUP_COL[r[0]]}'>{r[0]}</b></td><td>{r[1]}</td><td>{fmt(r[2])}</td><td>{fmt(r[3], 2)}</td>"
                   f"<td>{fmt(r[4], 2)}</td><td>{fmt(r[5], 0)}</td><td>{fmt(r[6], 2)}</td><td>{fmt(r[7], 0)}</td><td>{fmt(r[8], 0)}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def clip_table(clips):
    out = ["<table><thead><tr><th>clip</th><th>group</th><th>tracked</th><th>front foot</th><th>stance (cm)</th><th>stance / leg</th>"
           "<th>front knee</th><th>back knee</th><th>asym</th><th>asym at bottom</th><th>front min</th><th>back min</th><th>front ROM</th><th>back ROM</th><th>drive ratio</th>"
           "<th>knee ahead of ankle (cm)</th><th>hips</th><th>pump Hz</th></tr></thead><tbody>"]
    for s, _ in clips:
        out.append(f"<tr><td><a href='#{s['stem']}'>{html.escape(s['label'])}</a></td><td style='color:{GROUP_COL.get(s['group'], '#888')}'><b>{s['group']}</b></td>"
                   f"<td>{s['tracked_pct']:.0f}%</td><td>{s['front_foot']}</td><td>{fmt(s['stance_width_m']*100, 0)}</td><td>{fmt(s['stance_width_over_leg'], 2)}</td>"
                   f"<td>{fmt(s['front_knee_deg'], 0)}</td><td>{fmt(s['back_knee_deg'], 0)}</td><td>{fmt(s['knee_asymmetry_deg'], 0)}</td>"
                   f"<td>{fmt(s['asym_at_bottom_deg'], 0)}</td><td>{fmt(s['front_knee_p10'], 0)}</td><td>{fmt(s['back_knee_p10'], 0)}</td>"
                   f"<td>{fmt(s['front_knee_rom_deg'], 0)}</td><td>{fmt(s['back_knee_rom_deg'], 0)}</td><td>{fmt(s['leg_drive_ratio'], 2)}</td>"
                   f"<td>{fmt(s['front_knee_over_ankle_m']*100, 0)}</td><td>{fmt(s['hip_fraction'], 2)}</td><td>{fmt(s['pump_freq_hz'], 2)}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def is_key(s):
    """Key clips: reviewed as relevant (or unreviewed) and tracked well enough to trust."""
    rv = REVIEW.get(s["stem"], {})
    if rv.get("relevant") == "no" or rv.get("relevant") == "unsure":
        return False
    return s["tracked_pct"] >= 60


def card(s, series, i):
    cid = f"c{i}"
    src = f'<a href="{html.escape(s["source"])}" target="_blank">source</a>' if s.get("source") else ""
    note = html.escape(s.get("note", ""))
    win = f"{fmt(s.get('start') or 0, 1)} to {fmt(s['end'], 1)} s" if s.get("end") else "whole clip"
    return f"""
<section class="card" id="{s['stem']}">
  <div class="row">
    <div class="pair">
      <video src="{s['video_rel']}" controls muted preload="metadata"></video>
      <video src="output/{s['stem']}_skel.mp4" controls muted preload="metadata" class="skel"></video>
    </div>
    <div>
      <h3><span class="badge" style="background:{GROUP_COL.get(s['group'], '#888')}">{s['group']}</span> {html.escape(s['label'])}</h3>
      <p class="muted">{note} {src} &middot; {win} &middot; tracked {s['tracked_pct']:.0f}% &middot; front foot {s['front_foot']}</p>
      <div class="stats key">
        <div class="stat"><b>{fmt(s['asym_at_bottom_deg'], 0)}&deg;</b><span>back knee deeper than front at bottom of pump</span></div>
        <div class="stat"><b>{fmt(s['leg_drive_ratio'], 2)}x</b><span>back knee movement / front knee movement</span></div>
        <div class="stat"><b>{fmt(s['front_knee_p10'], 0)}&deg; / {fmt(s['back_knee_p10'], 0)}&deg;</b><span>deepest front / back knee bend</span></div>
      </div>
      <div class="review" data-stem="{s['stem']}" data-offset="{s.get('start') or 0}">
        <div class="rv-row">
          <label><input type="radio" name="rel-{cid}" value="yes"> use it</label>
          <label><input type="radio" name="rel-{cid}" value="no"> drop</label>
          <label><input type="radio" name="rel-{cid}" value="unsure"> unsure</label>
          <select class="rv-group"><option value="pro">pro</option><option value="beginner">beginner</option><option value="reference">reference</option></select>
          <button type="button" class="rv-btn rv-setstart">Start = now</button>
          <input type="number" class="rv-start" step="0.1" min="0" placeholder="start">
          <input type="number" class="rv-end" step="0.1" min="0" placeholder="end">
          <button type="button" class="rv-btn rv-setend">End = now</button>
          <button type="button" class="rv-btn rv-play">Play window</button>
          <span class="rv-now muted">0.0 s</span>
        </div>
        <div class="rv-row">
          <input type="text" class="rv-note" placeholder="note">
          <span class="rv-state muted"></span>
        </div>
      </div>
    </div>
  </div>
  <details>
    <summary>Detail: contact sheet, all numbers, knee angles over time</summary>
    <img src="{s['contact_rel']}" alt="contact sheet" class="contact">
    <div class="stats">
      <div class="stat"><b>{fmt(s['stance_width_m']*100, 0)} cm</b><span>stance width ({fmt(s['stance_width_over_leg'], 2)} leg lengths, {fmt(s['stance_width_over_shoulder'], 2)} shoulder widths)</span></div>
      <div class="stat"><b>{fmt(s['front_knee_over_ankle_m']*100, 0)} cm</b><span>front knee ahead of front ankle</span></div>
      <div class="stat"><b>{fmt(s['hip_fraction'], 2)}</b><span>hips between back ankle (0) and front ankle (1)</span></div>
      <div class="stat"><b>{fmt(s['front_knee_deg'], 0)}&deg; / {fmt(s['back_knee_deg'], 0)}&deg;</b><span>median front / back knee angle</span></div>
      <div class="stat"><b>{fmt(s['front_knee_rom_deg'], 0)}&deg; / {fmt(s['back_knee_rom_deg'], 0)}&deg;</b><span>front / back knee range of motion</span></div>
      <div class="stat"><b>{fmt(s['pump_freq_hz'], 2)} Hz</b><span>pump frequency ({s['pump_cycles']} cycles)</span></div>
    </div>
    {line_chart(series, cid)}
  </details>
</section>"""


def build():
    global REVIEW
    REVIEW = load_review()
    clips = load()
    key = [(s, ser) for s, ser in clips if is_key(s)]
    other = [(s, ser) for s, ser in clips if not is_key(s)]
    key_cards = "".join(card(s, ser, i) for i, (s, ser) in enumerate(key))
    other_cards = "".join(card(s, ser, 1000 + i) for i, (s, ser) in enumerate(other))
    top_dots = "".join(
        f'<figure><figcaption><b>{title}</b> <span class="muted">{html.escape(desc)}</span></figcaption>{dot_plot(clips, k, title, unit)}</figure>'
        for k, title, unit, desc in METRICS[:2])
    all_dots = "".join(
        f'<figure><figcaption><b>{title}</b> <span class="muted">{html.escape(desc)}</span></figcaption>{dot_plot(clips, k, title, unit)}</figure>'
        for k, title, unit, desc in METRICS[2:])
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pump Stance Review</title>
<style>
:root{{color-scheme:light;--surface:#fcfcfb;--surface-2:#f1f0ec;--text:#0b0b0b;--text-2:#52514e;--muted:#7a7873;--grid:#e3e1db;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--front:#2a78d6;--back:#eb6834}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;--surface:#1a1a19;--surface-2:#242422;--text:#fff;--text-2:#c3c2b7;--muted:#8f8e87;--grid:#33332f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--front:#3987e5;--back:#d95926}}}}
:root[data-theme=dark]{{color-scheme:dark;--surface:#1a1a19;--surface-2:#242422;--text:#fff;--text-2:#c3c2b7;--muted:#8f8e87;--grid:#33332f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--front:#3987e5;--back:#d95926}}
body{{margin:0;background:var(--surface);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:0 20px 80px}}
h1{{font-size:24px;margin:16px 0 4px}} h2{{font-size:18px;margin:28px 0 8px}} h3{{font-size:16px;margin:0 0 2px}}
p{{margin:6px 0}} .muted{{color:var(--muted);font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}} th,td{{padding:6px 8px;text-align:left;border-bottom:1px solid var(--grid);white-space:nowrap}} th{{color:var(--text-2);font-weight:600}}
.tablewrap{{overflow-x:auto}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:18px}}
figure{{margin:0;background:var(--surface-2);border-radius:8px;padding:10px}} figcaption{{font-size:13px;margin-bottom:4px}}
svg.dot,svg.line{{width:100%;height:auto;display:block}}
.grid{{stroke:var(--grid);stroke-width:1}} .zero{{stroke:var(--text-2);stroke-width:1;stroke-dasharray:3 3}}
.tick,.rowlab,.axis{{font-size:11px;fill:var(--text-2)}} .rowlab{{font-size:12px}}
.card{{background:var(--surface-2);border-radius:10px;padding:14px;margin:14px 0}}
.badge{{display:inline-block;color:#fff;font-size:11px;padding:2px 8px;border-radius:10px;vertical-align:middle;margin-right:4px}}
.contact{{width:100%;border-radius:6px;display:block;margin:8px 0}}
.row{{display:grid;grid-template-columns:minmax(300px,520px) 1fr;gap:14px;align-items:start}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
video{{width:100%;max-height:360px;background:#000;border-radius:6px}} video.skel{{background:#f6f4f0}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;margin:8px 0}} .stat{{background:var(--surface);border-radius:6px;padding:8px 10px}} .stat b{{font-size:20px;display:block}} .stat span{{font-size:12px;color:var(--text-2)}}
.stats.key .stat b{{font-size:24px}}
path.front{{fill:none;stroke:var(--front);stroke-width:2}} path.back{{fill:none;stroke:var(--back);stroke-width:2}}
.cross{{stroke:var(--text-2);stroke-width:1}} .mf{{fill:var(--front);stroke:var(--surface);stroke-width:2}} .mb{{fill:var(--back);stroke:var(--surface);stroke-width:2}}
.legend{{display:flex;gap:16px;font-size:12px;color:var(--text-2);margin-top:4px}} .sw{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}} .sw.front{{background:var(--front)}} .sw.back{{background:var(--back)}}
.tip{{margin-left:auto;font-variant-numeric:tabular-nums}}
ul{{padding-left:20px}} li{{margin:3px 0}}
details{{margin-top:8px}} summary{{cursor:pointer;color:var(--text-2);font-size:13px}} details[open] summary{{margin-bottom:8px}}
details.section{{background:var(--surface-2);border-radius:10px;padding:10px 14px;margin:14px 0}} details.section summary{{font-size:16px;font-weight:600;color:var(--text)}}
.review{{background:var(--surface);border-radius:8px;padding:8px 10px;margin:8px 0 0;border-left:4px solid var(--grid)}}
.review.yes{{border-left-color:var(--s1)}} .review.no{{border-left-color:var(--s2)}} .review.unsure{{border-left-color:var(--s3)}}
.rv-row{{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin:3px 0;font-size:13px}}
.rv-btn{{font:inherit;font-size:12px;padding:3px 8px;border-radius:6px;border:1px solid var(--grid);background:var(--surface-2);color:var(--text);cursor:pointer}}
.review input[type=number]{{width:64px;font:inherit;font-size:13px;padding:3px 5px;border-radius:5px;border:1px solid var(--grid);background:var(--surface-2);color:var(--text)}}
.review input[type=text]{{flex:1;min-width:160px;font:inherit;font-size:13px;padding:3px 6px;border-radius:5px;border:1px solid var(--grid);background:var(--surface-2);color:var(--text)}}
.review select{{font:inherit;font-size:13px;padding:3px 5px;border-radius:5px;border:1px solid var(--grid);background:var(--surface-2);color:var(--text)}}
.rvbar{{position:sticky;top:0;z-index:5;background:var(--surface-2);border-bottom:1px solid var(--grid);padding:8px 12px;margin:0 -20px 12px;display:flex;gap:14px;align-items:center;font-size:13px;flex-wrap:wrap}}
.rvbar b{{font-variant-numeric:tabular-nums}}
@media (max-width:760px){{.row{{grid-template-columns:1fr}} .grid2{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="rvbar"><span>Review: <b id="rv-count">0</b> of <b>{len(clips)}</b> clips marked</span><span id="rv-server" class="muted">checking save server…</span><button type="button" class="rv-btn" id="rv-copy">Copy review JSON</button><button type="button" class="rv-btn" id="rv-next">Jump to next unreviewed</button></div>
<h1>Pump Stance Review</h1>
<p class="muted">Each clip shows the tracked overlay and, beside it, the same movement as a hips-pinned side view built from the 3D landmarks, so every clip is in the same frame whatever the camera angle. Does the front leg share the pump? Two numbers per clip: how much deeper the back knee bends than the front at the bottom of each pump (0 means equal), and how much more the back knee moves than the front over the cycle (1.0 means equal). Front leg is red in the overlays, back leg cyan.</p>

<div class="grid2">{top_dots}</div>

<h2>Key clips</h2>
{key_cards}

<details class="section"><summary>Other clips ({len(other)}: dropped, unsure or poorly tracked)</summary>
{other_cards}
</details>

<details class="section"><summary>All the numbers</summary>
<h2>Group medians</h2>
<div class="tablewrap">{group_summary(clips)}</div>
<p class="muted">Clips with under 40% tracked frames are excluded from the medians. Reference clips are shown but not pooled.</p>
<h2>Other metrics</h2>
<div class="grid2">{all_dots}</div>
<h2>All clips</h2>
<div class="tablewrap">{clip_table(clips)}</div>
<h2>Feet relative to the mast (manual reads)</h2>
<p class="muted">Read by eye off gridded frames, as a fraction of the visible board length, so perspective and board size are not corrected. Every rider here stands with the back foot roughly over the mast. The front foot lands 20 to 30 percent of board length ahead in both groups, so foot-to-mast distance on its own does not separate them. The difference shows up in the knees and hips.</p>
<div class="tablewrap">{manual_board_table()}</div>
</details>

<details class="section"><summary>How it is measured, and the limits</summary>
<ul>
<li><b>Knee asymmetry at bottom of pump</b>: front knee angle minus back knee angle, taken only in the frames where the back knee is most bent. A big positive number means the back leg squats while the front leg stays straight.</li>
<li><b>Leg drive ratio</b>: back knee range of motion divided by front knee range of motion. Near 1.0 means both legs pump. Well above 1.0 means the back leg pumps and the front leg is a strut.</li>
<li><b>Deepest knee bend</b>: the 10th percentile knee angle over the clip. 180 is locked straight.</li>
<li><b>Stance width</b>: ankle to ankle in metres, also as a fraction of leg length and of shoulder width. MediaPipe world scale is estimated, so treat centimetres as approximate and compare the ratios.</li>
<li><b>Hip position</b>: 0 is over the back ankle, 1 is over the front ankle. A proxy for where the weight sits between the feet.</li>
<li><b>Mast offset is not measured.</b> The mast is under water or out of frame in almost every clip. The contact sheets show feet on the board for a manual read.</li>
<li><b>Pump frequency</b> comes from image space, so handheld or drone camera motion contaminates it.</li>
<li><b>Front foot detection</b> uses the head leading the hips along the stance line and is listed per clip so it can be checked against the overlay.</li>
<li>Small riders are cropped and upscaled before tracking. Tracked percentage tells you how much to trust a clip.</li>
</ul>
</details>
</main>
<script>
const SAVE_URL = "http://127.0.0.1:8767/review.json";
const REVIEW = {json.dumps(load_review())};
const TOTAL = {len(clips)};
let serverOk = false;
const stateOf = box => {{
  const rel = box.querySelector('input[type=radio]:checked');
  const num = el => el.value === '' ? null : +el.value;
  const off = +box.dataset.offset || 0;
  return {{ relevant: rel ? rel.value : null, group: box.querySelector('.rv-group').value,
           start: num(box.querySelector('.rv-start')), end: num(box.querySelector('.rv-end')), window_offset: off,
           note: box.querySelector('.rv-note').value, updated: new Date().toISOString() }};
}};
const allState = () => {{ const o = {{}}; document.querySelectorAll('.review').forEach(b => {{ const st = stateOf(b); if (st.relevant || st.note || st.start != null || st.end != null) o[b.dataset.stem] = st; }}); return o; }};
function paint(box) {{
  const st = stateOf(box); box.classList.remove('yes', 'no', 'unsure'); if (st.relevant) box.classList.add(st.relevant);
  document.getElementById('rv-count').textContent = Object.values(allState()).filter(x => x.relevant).length;
}}
let saveTimer = null;
function save(box) {{
  paint(box);
  const data = allState();
  try {{ localStorage.setItem('pump-stance-review', JSON.stringify(data)); }} catch (e) {{}}
  const state = box.querySelector('.rv-state'); state.textContent = 'saving…';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => fetch(SAVE_URL, {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(data) }})
    .then(r => {{ state.textContent = r.ok ? 'saved to stance/review.json' : 'server error, kept in browser'; }})
    .catch(() => {{ state.textContent = 'save server off, kept in browser (use Copy review JSON)'; }}), 250);
}}
function apply(box, st) {{
  if (!st) return;
  if (st.relevant) {{ const r = box.querySelector(`input[type=radio][value=${{st.relevant}}]`); if (r) r.checked = true; }}
  if (st.group) box.querySelector('.rv-group').value = st.group;
  if (st.start != null) box.querySelector('.rv-start').value = st.start;
  if (st.end != null) box.querySelector('.rv-end').value = st.end;
  if (st.note) box.querySelector('.rv-note').value = st.note;
  paint(box);
}}
document.querySelectorAll('.review').forEach(box => {{
  const card = box.closest('.card'), video = card.querySelector('video'), off = +box.dataset.offset || 0;
  const now = box.querySelector('.rv-now'), startIn = box.querySelector('.rv-start'), endIn = box.querySelector('.rv-end');
  const grp = card.querySelector('.badge'); if (grp) box.querySelector('.rv-group').value = grp.textContent.trim();
  video.addEventListener('timeupdate', () => {{ now.textContent = video.currentTime.toFixed(1) + ' s' + (off ? ` (source ${{(video.currentTime + off).toFixed(1)}} s)` : ''); }});
  box.querySelector('.rv-setstart').onclick = () => {{ startIn.value = video.currentTime.toFixed(1); save(box); }};
  box.querySelector('.rv-setend').onclick = () => {{ endIn.value = video.currentTime.toFixed(1); save(box); }};
  box.querySelector('.rv-play').onclick = () => {{
    const a = startIn.value === '' ? 0 : +startIn.value, b = endIn.value === '' ? video.duration : +endIn.value;
    video.currentTime = Math.max(0, a); video.play();
    const stop = () => {{ if (video.currentTime >= b) {{ video.pause(); video.removeEventListener('timeupdate', stop); }} }};
    video.addEventListener('timeupdate', stop);
  }};
  box.querySelectorAll('input, select').forEach(el => el.addEventListener('change', () => save(box)));
  apply(box, REVIEW[box.dataset.stem]);
}});
try {{ const local = JSON.parse(localStorage.getItem('pump-stance-review') || '{{}}'); document.querySelectorAll('.review').forEach(b => {{ const st = local[b.dataset.stem]; if (st && (!REVIEW[b.dataset.stem] || st.updated > REVIEW[b.dataset.stem].updated)) apply(b, st); }}); }} catch (e) {{}}
fetch(SAVE_URL).then(r => r.json()).then(d => {{ serverOk = true; document.getElementById('rv-server').textContent = 'auto-saving to stance/review.json'; document.querySelectorAll('.review').forEach(b => {{ const st = d[b.dataset.stem]; const cur = stateOf(b); if (st && (!cur.relevant || st.updated > (cur.updated || ''))) apply(b, st); }}); }})
  .catch(() => {{ document.getElementById('rv-server').textContent = 'save server not running: choices stay in this browser, use Copy review JSON to hand them over'; }});
document.getElementById('rv-copy').onclick = () => {{ const txt = JSON.stringify(allState(), null, 1); navigator.clipboard.writeText(txt).then(() => {{ document.getElementById('rv-copy').textContent = 'Copied'; setTimeout(() => document.getElementById('rv-copy').textContent = 'Copy review JSON', 1500); }}); }};
document.getElementById('rv-next').onclick = () => {{ const b = [...document.querySelectorAll('.review')].find(x => !stateOf(x).relevant); if (b) b.closest('.card').scrollIntoView({{ behavior: 'smooth', block: 'start' }}); }};
document.querySelectorAll('svg.line').forEach(svg=>{{
  const t=svg.dataset.t.split(',').map(Number), f=svg.dataset.f.split(',').map(Number), b=svg.dataset.b.split(',').map(Number);
  const hit=svg.querySelector('.hit'), cross=svg.querySelector('.cross'), mf=svg.querySelector('.mf'), mb=svg.querySelector('.mb');
  const tip=document.getElementById(svg.id+'-tip');
  const left=44,right=12,top=14,bot=26,W=640,H=200,ylo=60,yhi=185, t0=t[0], t1=t[t.length-1];
  const sx=v=>left+(v-t0)/Math.max(1e-6,t1-t0)*(W-left-right), sy=v=>top+(yhi-Math.min(Math.max(v,ylo),yhi))/(yhi-ylo)*(H-top-bot);
  hit.addEventListener('mousemove',e=>{{
    const pt=svg.createSVGPoint(); pt.x=e.clientX; pt.y=e.clientY; const p=pt.matrixTransform(svg.getScreenCTM().inverse());
    const tv=t0+(p.x-left)/(W-left-right)*(t1-t0); let i=0; for(let k=0;k<t.length;k++){{ if(Math.abs(t[k]-tv)<Math.abs(t[i]-tv)) i=k; }}
    cross.style.display=''; cross.setAttribute('x1',sx(t[i])); cross.setAttribute('x2',sx(t[i]));
    mf.style.display=''; mf.setAttribute('cx',sx(t[i])); mf.setAttribute('cy',sy(f[i]));
    mb.style.display=''; mb.setAttribute('cx',sx(t[i])); mb.setAttribute('cy',sy(b[i]));
    tip.textContent=`${{t[i].toFixed(1)}}s  front ${{f[i]}}°  back ${{b[i]}}°  diff ${{f[i]-b[i]}}°`;
  }});
  hit.addEventListener('mouseleave',()=>{{cross.style.display='none';mf.style.display='none';mb.style.display='none';tip.textContent='';}});
}});
</script>
</body></html>"""
    (ROOT / "report.html").write_text(page)
    print(f"wrote {ROOT / 'report.html'} with {len(clips)} clips ({len(key)} key)")


WEB_APP = Path(__file__).parent.parent / "site" / "apps" / "pump-stance" / "index.html"
KEYS = ["label", "group", "tracked_pct", "asym_at_bottom_deg", "front_knee_p10", "back_knee_p10", "leg_drive_ratio",
        "stance_width_over_leg", "hip_fraction", "front_knee_over_ankle_m", "knee_asymmetry_deg", "front_knee_deg", "back_knee_deg"]


def write_web_reference():
    """Embed the reference-clip metrics into the browser app between the REFERENCE markers."""
    clips = load()
    rows = []
    for s, _ in clips:
        if s["group"] not in ("pro", "beginner"):
            continue
        row = {k: (None if isinstance(s.get(k), float) and math.isnan(s[k]) else (round(s[k], 3) if isinstance(s.get(k), float) else s.get(k))) for k in KEYS}
        rows.append(row)
    src = WEB_APP.read_text()
    a = src.index("// REFERENCE:BEGIN"); a = src.index("\n", a) + 1
    b = src.index("// REFERENCE:END")
    src = src[:a] + "const REFERENCE = " + json.dumps(rows) + ";\n" + src[b:]
    WEB_APP.write_text(src)
    print(f"wrote {len(rows)} reference clips into {WEB_APP}")


if __name__ == "__main__":
    import sys
    build()
    if "--web" in sys.argv:
        write_web_reference()
