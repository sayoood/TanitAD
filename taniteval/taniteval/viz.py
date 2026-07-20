"""TanitEval — BEV trajectory visualization.

Self-contained SVGs (no deps, artifact-safe): GT vs prediction vs CV in the
ego frame (x forward, y left) for selected windows — best / median / worst by
model ADE — plus optional second-model overlay for A/B panels."""
from __future__ import annotations

import torch

C = {"gt": "#8a93a6", "pred": "#2dd4bf", "cv": "#e0b34d", "b": "#a78bfa"}


def _path(way, sx, sy, ox, oy):
    pts = [(ox, oy)] + [(ox - float(w[1]) * sx, oy - float(w[0]) * sy)
                        for w in way]
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def bev_svg(gt, pred, cv, pred_b=None, title="", w=250, h=210, max_m=None):
    """One BEV panel. gt/pred/cv: [4, 2] waypoints (x fwd, y left), metres."""
    m = max_m or max(1.0, float(torch.cat(
        [t.abs().max().reshape(1) for t in (gt, pred, cv)]).max()) * 1.15)
    sx = sy = (min(w, h) * 0.82) / m
    ox, oy = w * 0.5, h * 0.88
    rows = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="ui-monospace,monospace">',
            f'<rect width="{w}" height="{h}" rx="10" fill="#0d141b" '
            f'stroke="#1e2833"/>']
    for gm in range(5, int(m) + 1, 5):                     # range rings
        rows.append(f'<circle cx="{ox}" cy="{oy}" r="{gm * sy:.1f}" fill="none" '
                    f'stroke="#182029" stroke-width="1"/>')
    for key, way, wd in (("cv", cv, 1.6), ("gt", gt, 2.6), ("pred", pred, 2.6)):
        dash = ' stroke-dasharray="5,4"' if key == "cv" else ""
        rows.append(f'<path d="{_path(way, sx, sy, ox, oy)}" fill="none" '
                    f'stroke="{C[key]}" stroke-width="{wd}"{dash} '
                    f'stroke-linecap="round"/>')
        for pt in way:
            rows.append(f'<circle cx="{ox - float(pt[1]) * sx:.1f}" '
                        f'cy="{oy - float(pt[0]) * sy:.1f}" r="2.6" '
                        f'fill="{C[key]}"/>')
    if pred_b is not None:
        rows.append(f'<path d="{_path(pred_b, sx, sy, ox, oy)}" fill="none" '
                    f'stroke="{C["b"]}" stroke-width="2.2" '
                    f'stroke-linecap="round"/>')
    rows.append(f'<circle cx="{ox}" cy="{oy}" r="4" fill="#e9edf3"/>')  # ego
    rows.append(f'<text x="10" y="17" font-size="10.5" fill="#a3afbf">{title}'
                f'</text>')
    rows.append(f'<text x="10" y="{h - 9}" font-size="9" fill="#6b7789">'
                f'{m:.0f} m scale · gt grey · pred teal · cv dashed</text>')
    rows.append("</svg>")
    return "".join(rows)


def gallery(data, model_name, k_each=2, data_b=None, name_b=None):
    """Best/median/worst panels by window ADE. Returns list of SVG strings."""
    de = torch.linalg.norm(data["pred"] - data["gt"], dim=-1).mean(dim=1)
    order = torch.argsort(de)
    n = len(order)
    picks = ([("best", int(order[i])) for i in range(k_each)]
             + [("median", int(order[n // 2 + i])) for i in range(k_each)]
             + [("worst", int(order[-1 - i])) for i in range(k_each)])
    out = []
    for tag, i in picks:
        pb = data_b["pred"][i] if data_b is not None else None
        t = (f"{model_name} · {tag} · ade {float(de[i]):.2f} m"
             + (f" · vs {name_b}" if name_b else ""))
        out.append(bev_svg(data["gt"][i], data["pred"][i], data["cv"][i],
                           pred_b=pb, title=t))
    return out
