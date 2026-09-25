#!/usr/bin/env python3
"""Two-surface leaderboard: planners that must choose vs world models handed the expert's controls.
ADE@2s full-set mean + episode-cluster bootstrap CI95, 881 val windows / 40 episodes.
Sources: MODEL_REGISTRY.md §6 and §1.4b; taniteval/results/driving_*.json;
…/incoming/2026-07-26-closedloop-artifact-rerun/closedloop_flagship-30k.CORRECTED.json (R1 §1.1)."""
import math
import sys

OUT = sys.argv[1]
GROUPS = [
    ("Planners: the model chooses", "plan", [
        ("v1.6 · REF-C decoder on v1 WM", 0.4375, 0.3423, 0.5501),
        ("REF-C-XL · flat, 252M", 0.4714, 0.3896, 0.5556),
        ("REF-C-base · flat, 104M", 0.4728, 0.3835, 0.5699),
        ("REF-C-small · flat, 55M", 0.5261, 0.4295, 0.6262),
        ("REF-B v2 · hierarchical BC, no WM", 0.5913, 0.4766, 0.7131),
        ("v4.1 @10k · WM + planner head", 0.8522, 0.7468, 0.9800),
        ("v1 hierarchy plan → tracker", 1.9028, 1.732, 2.0745),
        ("v1 tactical head, direct", 3.3839, 2.8336, 3.9722),
    ]),
    ("Given the expert's controls", "given", [
        ("flagship v1 world model (headline)", 0.4271, 0.3675, 0.4871),
        ("kinematic bicycle, 0 parameters", 0.4518, 0.3097, 0.6174),
        ("v3enc @10k (RESTART)", 1.9654, 1.6556, 2.2859),
        ("REF-A · frozen DINOv2", 2.1675, 1.9081, 2.4212),
        ("flagship no-speed control", 3.0175, 2.5450, 3.5444),
    ]),
]
FLOORS = [  # value, label, row, anchor
    (0.5005, "kinematic best-of-3 0.501", 0, "end"),
    (0.5735, "ego-status, no vision 0.574", 0, "start"),
    (0.523, "CTRV oracle 0.523", 1, "start"),
    (0.8377, "constant velocity 0.838", 2, "start"),
]
W, X0, X1 = 840, 262, 812
VMIN, VMAX = 0.30, 4.2
ROW, GAP = 28, 30
TOP = 70


def x(v):
    return X0 + (math.log(v) - math.log(VMIN)) / (math.log(VMAX) - math.log(VMIN)) * (X1 - X0)


n_rows = sum(len(g[2]) for g in GROUPS)
H = TOP + len(GROUPS) * GAP + n_rows * ROW + 52
s = [f'<figure class="chart"><svg viewBox="0 0 {W} {H}" width="100%" style="min-width:680px;max-width:{W}px" '
     f'role="img" aria-labelledby="lbt lbd" font-family="Barlow, system-ui, sans-serif">',
     '<title id="lbt">ADE at 2 s on two different surfaces: planners that choose, and world models handed the expert\'s future controls</title>',
     '<desc id="lbd">Flat REF-C planners score about 0.47 m. The hierarchy\'s own decision paths score 1.9 and 3.4 m. '
     'The flagship headline of 0.43 m is a world model handed the expert\'s controls and matches a zero-parameter bicycle model given the same controls.</desc>']
grid_bottom = H - 44
for t in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
    xx = x(t)
    s.append(f'<line x1="{xx:.1f}" y1="{TOP-6}" x2="{xx:.1f}" y2="{grid_bottom}" stroke="var(--rule)" stroke-width="1"/>')
    s.append(f'<text x="{xx:.1f}" y="{grid_bottom+16}" text-anchor="middle" font-size="12" fill="var(--muted)" '
             f'style="font-variant-numeric:tabular-nums">{t:g}</text>')
s.append(f'<text x="{(X0+X1)/2:.0f}" y="{H-8}" text-anchor="middle" font-size="12" fill="var(--muted)">'
         f'ADE@2s in metres, log scale · lower is better</text>')
for v, lab, row, anc in FLOORS:
    xx = x(v)
    s.append(f'<line x1="{xx:.1f}" y1="{TOP-8}" x2="{xx:.1f}" y2="{grid_bottom}" stroke="var(--muted)" '
             f'stroke-width="1.5" stroke-dasharray="4 4"/>')
    ty = 14 + row * 17
    dx = -5 if anc == "end" else 5
    s.append(f'<text x="{xx+dx:.1f}" y="{ty}" text-anchor="{anc}" font-size="12" fill="var(--muted)">{lab}</text>')
y = TOP
for title, kind, rows in GROUPS:
    y += GAP
    s.append(f'<text x="8" y="{y-9}" font-size="12.5" font-weight="600" fill="var(--ink)" '
             f'letter-spacing=".02em">{title}</text>')
    for lab, m, lo, hi in rows:
        cy = y + ROW / 2
        col = "var(--accent)" if kind == "plan" else "var(--muted)"
        fill = col if kind == "plan" else "var(--surface)"
        s.append(f'<g><title>{lab}: {m:.4f} m, CI95 [{lo:.4f}, {hi:.4f}]; full-set mean, episode-cluster bootstrap, 881 windows / 40 episodes'
                 + (' (scored with the expert\'s future controls)' if kind == "given" else '') + '</title>')
        s.append(f'<rect x="0" y="{y:.1f}" width="{W}" height="{ROW}" fill="transparent"/>')
        s.append(f'<text x="{X0-12}" y="{cy+4:.1f}" text-anchor="end" font-size="13" fill="var(--ink)">{lab}</text>')
        s.append(f'<line x1="{x(lo):.1f}" y1="{cy:.1f}" x2="{x(hi):.1f}" y2="{cy:.1f}" stroke="{col}" stroke-width="2" stroke-linecap="round"/>')
        for e in (lo, hi):
            s.append(f'<line x1="{x(e):.1f}" y1="{cy-5:.1f}" x2="{x(e):.1f}" y2="{cy+5:.1f}" stroke="{col}" stroke-width="2" stroke-linecap="round"/>')
        s.append(f'<circle cx="{x(m):.1f}" cy="{cy:.1f}" r="5" fill="{fill}" stroke="{col}" stroke-width="2"/>')
        if x(hi) + 44 > X1 + 20:
            s.append(f'<text x="{x(lo)-8:.1f}" y="{cy+4:.1f}" text-anchor="end" font-size="12" fill="var(--muted)" style="font-variant-numeric:tabular-nums">{m:.3f}</text>')
        else:
            s.append(f'<text x="{x(hi)+8:.1f}" y="{cy+4:.1f}" font-size="12" fill="var(--muted)" style="font-variant-numeric:tabular-nums">{m:.3f}</text>')
        s.append('</g>')
        y += ROW
s.append('</svg><figcaption>Filled points: planners that must choose a trajectory from frames and ego speed. Hollow points: '
         'models handed the expert\'s logged future steering and acceleration, so nothing is chosen. '
         'ADE@2s full-set mean with 95 % episode-cluster bootstrap interval, 881 val windows / 40 episodes '
         '(MODEL_REGISTRY §6 and §1.4b; the bicycle and hierarchy-path values from the 2026-07-26 closed-loop artifact rerun). '
         'Dashed lines: trivial floors on the same windows. Off-scale: flagship v2 5.94 (given controls). '
         'P2 CEM over v1 (0.893) is omitted: legacy estimator, not recomputable. '
         'Overlapping intervals are not an ordering; orderings come only from paired tests.</figcaption></figure>')
open(OUT, "w", encoding="utf-8").write("\n".join(s))
print("wrote", OUT, H)
