"""Standalone LIGHTWEIGHT SVG gallery (vector, ~30KB) for the Artifact — same
scenes as the heavy PNG version but tiny, so it loads on mobile."""
import json
import math
from pathlib import Path

D = Path(__file__).resolve().parent
refa = {s["ep"]: s for s in json.loads((D / "refa_coords.json").read_text())}
flag = {s["ep"]: s for s in json.loads((D / "flagship_coords.json").read_text())}
RA_STEP = refa[3]["step"]
FL_STEP = flag[3]["step"]

ORDER = [(148, "Straight · 18 m/s"), (271, "Sharp left"), (239, "Sharp right"),
         (3, "Sharp left"), (28, "Gentle · 18 m/s"), (166, "Gentle · slow"),
         (47, "Straight · slow"), (31, "Straight · 36 m/s")]
GT_C, RA_C, FL_C = "#1baf7a", "#eb6834", "#2a78d6"
W, H, PAD = 340, 224, 12


def poly(pts, px):
    return " ".join(f"{px(f, l)[0]:.1f},{px(f, l)[1]:.1f}" for f, l in pts)


def dots(pts, px, col):
    return "".join(
        f'<circle cx="{px(pts[i][0], pts[i][1])[0]:.1f}" '
        f'cy="{px(pts[i][0], pts[i][1])[1]:.1f}" r="2.3" fill="{col}"/>'
        for i in range(4, len(pts), 5))


def scene_svg(ep):
    ra, fl = refa[ep], flag[ep]
    gt = ra["gt"]
    allp = [(0.0, 0.0)] + [tuple(p) for p in gt + ra["pred"] + fl["pred"]]
    fs = [p[0] for p in allp]; ls = [p[1] for p in allp]
    fmin, fmax, lmin, lmax = min(fs), max(fs), min(ls), max(ls)
    fr, lr = max(fmax - fmin, 1.0), max(lmax - lmin, 1.0)
    fmin -= fr * 0.10; fmax += fr * 0.10; lmin -= lr * 0.12; lmax += lr * 0.12
    span_f, span_l = max(fmax - fmin, 2.0), max(lmax - lmin, 2.0)
    pw, ph = W - 2 * PAD, H - 2 * PAD
    scale = min(pw / span_l, ph / span_f)
    fc, lc = (fmin + fmax) / 2, (lmin + lmax) / 2
    cx, cy = W / 2, H / 2

    def px(f, l):
        return (cx - (l - lc) * scale, cy - (f - fc) * scale)

    grid = []
    step = next((s for s in (1, 2, 5, 10, 20, 50) if span_f / s <= 6), 100)
    g = math.ceil(fmin / step) * step
    while g <= fmax:
        _, y = px(g, lc)
        grid.append(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W-PAD}" y2="{y:.1f}" '
                    f'stroke="rgba(136,135,129,.20)" stroke-width="1"/>')
        g += step
    ox, oy = px(0, 0)
    return (
        f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
        f'aria-label="Episode {ep} trajectories">'
        f'<rect x="{PAD}" y="{PAD}" width="{W-2*PAD}" height="{H-2*PAD}" rx="8" '
        f'fill="none" stroke="rgba(136,135,129,.28)" stroke-width="1"/>'
        + "".join(grid) +
        f'<polyline points="{poly(fl["pred"], px)}" fill="none" stroke="{FL_C}" '
        f'stroke-width="2" stroke-dasharray="5 4" stroke-linejoin="round"/>'
        f'<polyline points="{poly(ra["pred"], px)}" fill="none" stroke="{RA_C}" '
        f'stroke-width="2" stroke-dasharray="5 4" stroke-linejoin="round"/>'
        f'<polyline points="{poly(gt, px)}" fill="none" stroke="{GT_C}" '
        f'stroke-width="2.6" stroke-linejoin="round"/>'
        + dots(gt, px, GT_C) + dots(ra["pred"], px, RA_C) + dots(fl["pred"], px, FL_C) +
        f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="3.5" fill="#7a7a75"/></svg>')


cards = []
for ep, desc in ORDER:
    ra_ade, fl_ade = refa[ep]["ade"], flag[ep]["ade"]
    ra_w = "600" if ra_ade < fl_ade else "400"
    fl_w = "400" if ra_ade < fl_ade else "600"
    cards.append(f'''
    <section class="card">
      <div class="card-h">
        <span class="ep">ep {ep:05d}</span><span class="ty">{desc}</span>
      </div>
      <div class="ade">
        <span style="color:{RA_C};font-weight:{ra_w}">REF-A {ra_ade:.2f} m</span>
        <span style="color:{FL_C};font-weight:{fl_w}">flagship {fl_ade:.2f} m</span>
      </div>
      {scene_svg(ep)}
    </section>''')

HTML = f'''<title>TanitAD — Trajectory Overlays</title>
<style>
  :root{{--bg:#f5f7fa;--card:#fff;--line:#e4e8ee;--soft:#eef1f6;
    --ink:#161a1f;--mut:#5b6570;--faint:#8a94a0;}}
  @media (prefers-color-scheme:dark){{:root{{--bg:#0e1116;--card:#171b22;
    --line:#262c35;--soft:#1c222b;--ink:#e7eaef;--mut:#9aa4b0;--faint:#6b7480;}}}}
  :root[data-theme=light]{{--bg:#f5f7fa;--card:#fff;--line:#e4e8ee;--soft:#eef1f6;
    --ink:#161a1f;--mut:#5b6570;--faint:#8a94a0;}}
  :root[data-theme=dark]{{--bg:#0e1116;--card:#171b22;--line:#262c35;--soft:#1c222b;
    --ink:#e7eaef;--mut:#9aa4b0;--faint:#6b7480;}}
  *{{box-sizing:border-box;}}
  body{{margin:0;background:var(--bg);color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5;}}
  .wrap{{max-width:640px;margin:0 auto;padding:22px 16px 60px;}}
  h1{{font-size:1.35rem;margin:0 0 6px;letter-spacing:-.01em;}}
  .lede{{color:var(--mut);font-size:.92rem;margin:0 0 14px;}}
  .legend{{display:flex;flex-wrap:wrap;gap:10px 16px;font-size:.8rem;
    color:var(--mut);padding:10px 12px;background:var(--card);
    border:1px solid var(--line);border-radius:12px;margin-bottom:16px;}}
  .legend i{{display:inline-block;width:20px;height:0;vertical-align:middle;
    margin-right:6px;}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:11px 13px;margin-bottom:12px;}}
  .card-h{{display:flex;justify-content:space-between;align-items:baseline;
    margin-bottom:3px;}}
  .ep{{font-weight:600;font-size:.95rem;}}
  .ty{{font-size:.8rem;color:var(--mut);}}
  .ade{{display:flex;gap:14px;font-size:.8rem;margin-bottom:6px;
    font-variant-numeric:tabular-nums;}}
  .notes{{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:14px 16px 4px;margin-top:20px;}}
  .notes li{{color:var(--mut);font-size:.88rem;margin:6px 0;}}
  .notes b{{color:var(--ink);font-weight:600;}}
  footer{{color:var(--faint);font-size:.78rem;text-align:center;margin-top:20px;}}
</style>
<div class="wrap">
  <h1>Trajectory overlays — flagship vs REF-A</h1>
  <p class="lede">Bird's-eye view, ego frame, up = ahead. Same 8 held-out scenes
    for both arms; dots every 0.5 s; ADE@2s = mean waypoint error (lower is
    tighter to ground truth).</p>
  <div class="legend">
    <span><i style="border-top:2.6px solid {GT_C}"></i>ground truth</span>
    <span><i style="border-top:2px dashed {RA_C}"></i>REF-A · frozen DINO · {RA_STEP//1000}k</span>
    <span><i style="border-top:2px dashed {FL_C}"></i>flagship · trained · {FL_STEP//1000}k</span>
  </div>
  {''.join(cards)}
  <div class="notes">
    <ul>
      <li><b>Flagship tracks tighter on 5 of 8 scenes</b> despite ~⅐ the training
        — biggest gap on the fast straight ep00148 (1.31 m vs 4.04 m): REF-A's
        frozen encoder overshoots forward at speed.</li>
      <li><b>REF-A holds the gentle-curve cases</b> (ep00028, ep00003) where the
        3k flagship still under-bends.</li>
      <li>Mid-training snapshots — both arms train on to 30k; paths will tighten.</li>
    </ul>
  </div>
  <footer>TanitAD Phase-0 · grounded 2 s rollout · REF-A {RA_STEP//1000}k / flagship {FL_STEP//1000}k</footer>
</div>'''

out = D / "trajectory_gallery.html"
out.write_text(HTML, encoding="utf-8")
print("wrote", out, len(HTML), "bytes")
