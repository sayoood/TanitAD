"""TanitEval — REF-C PLAN-FAN: the anchored-diffusion proposal set, scored, in BEV.

THE ASK (Sayed, 2026-07-20): show the different planned possibilities, COLORED BY
SCORE, highlighting the SELECTED one, in metric BEV. This module renders exactly
what REF-C's decoder computes — no invented modes, no re-scoring, no re-ranking.

WHAT THE DECODER ACTUALLY COMPUTES (read off tanitad/refs/refc.py
``AnchoredDiffusionDecoder.forward``; the claims below are code facts, not
paper facts — DiffusionDrive's own recipe differs on point 2):

  1. ONE classifier pass over the FULL anchor vocabulary. The N anchor
     trajectories (N = 256 for REF-C-XL, a data-built FPS vocabulary carried in
     the ``decoder.anchors`` buffer) become queries, cross-attend the 8x8xF conv
     map under a FiLM(condition), and emit
        conf   [B, N]        per-anchor confidence LOGITS (pre-softmax)
        offset [B, N, S, 2]  per-anchor trajectory offset
     giving x = anchors + offset — every anchor gets a refinement, not just a
     winner.

  2. ALL N ANCHORS ARE DENOISED — not a top-K subset. The truncated-diffusion
     loop (``for i in range(steps)``) calls ``_decode`` on the full [B, N, S, 2]
     tensor every step; there is no top-K gather anywhere in the module. At eval
     ``model.eval()`` zeroes the noise, so the 2 refinement steps are
     deterministic. CONSEQUENCE FOR THE VIZ: the scored fan is all 256 refined
     proposals; the shadow layer is the raw pre-refinement vocabulary. (Had only
     top-K been denoised, the spec would have been raw anchor+offset as shadow
     and the denoised top-K as the fan — it is not the case here.)

  3. THE SCORE IS COMPUTED ON THE ANCHOR, THE GEOMETRY IS THE DENOISED ONE. The
     denoise passes return ``_, off`` — their confidence output is DISCARDED. The
     selection score is the t=0 classifier-pass confidence over the ORIGINAL
     anchors, while the trajectory that gets selected is the 2-step-refined one.
     Scoring and refinement are therefore decoupled inside the model itself; the
     HUD's oracle-in-fan gap is the direct read on what that costs.

  4. H19 maneuver prior is IN the logits. ``graft_maneuver=True`` for this run:
     conf <- conf + maneuver_to_anchor(log_softmax(maneuver_logits)). The
     returned ``anchor_logits`` are therefore the POST-reweight logits — the
     softmax of them IS the distribution the model selects by. The HUD's tactical
     maneuver is the same head that produced this prior.

  5. ``grounded_selector=False`` for this run (verified against the run's
     config.json), so score == conf and ``sel_idx == anchor_logits.argmax``. The
     renderer ASSERTS this per batch and fails loud if a future ckpt breaks it.

LAYERS (back -> front), all in the metric ego BEV, ego at bottom-centre, heading
up, ISOTROPIC metres (a circle is a circle):
  1. vocabulary shadow  — all N RAW anchors (pre-refinement), faint grey
  2. scored fan         — all N REFINED proposals, colour = softmax confidence on
                          a FIXED log scale (viridis LUT, hardcoded — no
                          matplotlib on the eval pod), alpha + linewidth scale
                          with the score, drawn in ASCENDING score order
  3. top-8 emphasis     — thicker + waypoint dots
  4. SELECTED plan      — colour halo under a white core, 4 labelled waypoints
  5. GT                 — dashed green
  6. HUD + colorbar + 5 m grid

ORACLE-IN-FAN (the coverage-vs-scoring diagnostic, the point of the whole panel):
  ade_sel    = ADE of the proposal the model PICKED
  ade_oracle = min over all N refined proposals of that proposal's ADE
  ade_vocab  = min over all N RAW anchors (what the vocabulary alone could reach)
A large (ade_sel - ade_oracle) gap means the fan CONTAINED a good plan and the
model failed to SCORE it -> the failure is scoring, not coverage, and the fix is
a planning cost, not a bigger vocabulary. A small gap with both large means the
vocabulary/refinement cannot reach the manoeuvre at all -> coverage.

TRAJECTORY SURFACE (honest note, same as direct_overlay): every proposal is 4
TIME waypoints at WP_STEPS 5/10/15/20 (0.5/1/1.5/2 s, ego frame of the last
window pose). Polylines are drawn ego-origin -> wp1 -> wp2 -> wp3 -> wp4 as
STRAIGHT segments: a drawing device only, no curvature is invented, and every
number in the HUD is computed from the 4 waypoints alone.

BEV RANGE is derived from the clip's speed profile (cover v0*2 s + margin) and
held FIXED for the whole clip — a per-frame rescale would make the video jitter
and would make the fan's apparent spread meaningless frame-to-frame. The active
range is printed in the HUD.

REUSE: the same renderer is the v3/P2 CEM-candidate view — feed candidate
rollouts + planning COST instead of anchors + confidence and invert the
normalisation (low cost = bright). See
"TanitAD Research Hub/Benchmarks & Eval/PLANNER_VIZ_CONCEPT.md".

Usage (eval pod):
  PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
    python3 -m taniteval.plan_fan --model refc-xl-30k
  ... --clips 3:sharpturn,31:highspeed-straight
  ... --stills 3                 # + top-entropy PNG stills per clip
  ... --stills-only              # no video
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import loaders                                       # noqa: E402
from taniteval.cam_overlay import ego_future_path                   # noqa: E402
from taniteval.corpus_overlay import (HORIZON, FlatProjector,       # noqa: E402
                                      pretty_man, pretty_route)
from taniteval.direct_overlay import OUT, WP_STEPS, _fit            # noqa: E402
from taniteval.flagship_overlay import (COL_GT, HUD_BG,             # noqa: E402
                                        HUD_DIM, HUD_FG, K, WP_IDX, _font)
from taniteval.registry import CORPORA, MODELS                      # noqa: E402

# ---- canvas -----------------------------------------------------------------
CW, CH = 1280, 800                        # both even (libx264 yuv420p)
HUD_H = 96
BEV_XY, BEV_WH = (8, 100), (860, 692)     # the star
CAM_XY, CAM_WH = (896, 100), (356, 356)   # small context panel
BAR_XY, BAR_WH = (892, 508), (364, 26)    # score colorbar
LEG_X, LEG_Y = 876, 560

F_TOP, F_HUD, F_SUB, F_TINY = _font(16), _font(14), _font(13), _font(11)

BEV_BG = (8, 11, 15)
GRID = (30, 37, 46)
GRID_LBL = (96, 106, 120)
SHADOW = (74, 82, 96)                     # raw-anchor vocabulary
COL_SEL_CORE = (255, 255, 255)
PANEL_EDGE = (58, 68, 80)

# Fixed log-probability colour scale. FIXED (not per-frame renormalised) so a
# colour means the same probability in every frame of every clip; the uniform
# level 1/N is marked on the bar, so "brighter than the uniform tick" reads
# directly as "the model prefers this proposal".
P_FLOOR = 1e-4
LOG_FLOOR = math.log10(P_FLOOR)
BAR_TICKS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)

# viridis, 10 canonical stops (matplotlib is NOT installed on the eval pod).
VIRIDIS = ((68, 1, 84), (72, 40, 120), (62, 73, 137), (49, 104, 142),
           (38, 130, 142), (31, 158, 137), (53, 183, 121), (110, 206, 88),
           (181, 222, 43), (253, 231, 37))

DEFAULT_CLIPS = ("3:sharpturn,31:highspeed-straight,"
                 "28:highspeed-curve,11:failure-worstwindow")
TOP_K_EMPH = 8
MODE_THRESH = 0.01                        # "n_modes>1%" HUD counter


# ============================================================================
# colour / geometry helpers
# ============================================================================

def viridis(t: float) -> tuple[int, int, int]:
    """Perceptually-uniform LUT lookup with linear interpolation, t in [0, 1]."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    x = t * (len(VIRIDIS) - 1)
    i = min(int(x), len(VIRIDIS) - 2)
    f = x - i
    a, b = VIRIDIS[i], VIRIDIS[i + 1]
    return tuple(int(round(a[j] + f * (b[j] - a[j]))) for j in range(3))


def p_to_t(p: float) -> float:
    """softmax probability -> colour parameter on the FIXED log scale."""
    if p <= P_FLOOR:
        return 0.0
    return (math.log10(p) - LOG_FLOOR) / (0.0 - LOG_FLOOR)


def _clean(pts):
    """Finite, de-duplicated point list.

    PIL's WIDE-line renderer normalises the segment normal by the segment
    length (`dx * width / hypot`). A zero- or near-zero-length segment with
    width > 1 therefore divides by ~0, and the resulting non-finite polygon
    makes the C rasteriser try to fill a ~1e16-wide box — it does not crash, it
    GRINDS (one core, minutes per frame, no output). Trajectory proposals
    legitimately contain repeated waypoints (any plan for a stopped vehicle), so
    every data-driven polyline is cleaned before it reaches PIL.
    """
    out = []
    for p in pts:
        x, y = float(p[0]), float(p[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if not out or math.dist(out[-1], (x, y)) > 1e-3:
            out.append((x, y))
    return out


def _line(d, pts, fill, width=1):
    """Polyline draw hardened by :func:`_clean`."""
    pts = _clean(pts)
    if len(pts) >= 2:
        d.line(pts, fill=fill, width=width)


def _dashed(d, pts, fill, width=3, dash=10, gap=7):
    """Dashed polyline (PIL has no dash support) — used for the GT path.

    Walks the polyline from a SINGLE cumulative arc-length parameter. The
    obvious incremental implementation (carry the leftover dash across
    segments) accumulates float error until the residual span underflows to
    ~1e-15, which emits exactly the degenerate wide line described in
    :func:`_clean` and hangs the renderer on one frame. Do not reintroduce an
    incremental carry: the loop bound here is closed-form.
    """
    pts = _clean(pts)
    if len(pts) < 2:
        return
    seglen = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total, period = sum(seglen), float(dash + gap)
    if total < 0.5 or period <= 0.0:
        return

    def at(s):                                  # arc length -> point on the line
        for i, ln in enumerate(seglen):
            if s <= ln or i == len(seglen) - 1:
                f = min(max(s / ln, 0.0), 1.0) if ln > 1e-9 else 0.0
                return (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f,
                        pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f)
            s -= ln
        return pts[-1]

    for k in range(int(total / period) + 1):
        s0 = k * period
        s1 = min(s0 + dash, total)
        if s1 - s0 >= 0.5:                      # never emit a degenerate dash
            d.line([at(s0), at(s1)], fill=fill, width=width)


def poly(traj) -> list[tuple[float, float]]:
    """[S,2] waypoints -> the ego-origin-anchored knot list (drawing device)."""
    return [(0.0, 0.0)] + [(float(p[0]), float(p[1])) for p in traj]


# ============================================================================
# BEV panel — THE STAR
# ============================================================================

def draw_bev(w, h, anchors, fan, probs, sel, gt_wp, xmax, oracle=None):
    """Metric ego BEV, ego origin bottom-centre, heading up, ISOTROPIC.

    anchors [N,S,2] raw vocabulary · fan [N,S,2] refined proposals ·
    probs [N] softmax confidences · sel int · gt_wp [S,2] · xmax metres forward.
    Rendered into its OWN image so every layer is clipped to the panel.
    """
    im = Image.new("RGB", (w, h), BEV_BG)
    d = ImageDraw.Draw(im, "RGBA")
    pad_b, pad_t = 30, 16
    ppm = (h - pad_b - pad_t) / xmax             # px per metre (both axes)
    cx, by = w / 2.0, h - pad_b
    ymax = (w / 2.0) / ppm

    def m2px(X, Y):
        return cx - Y * ppm, by - X * ppm

    # --- metric grid (5 m) ---------------------------------------------------
    lbl_every = 5 if xmax <= 40 else 10
    r = 5
    while r <= xmax + 1e-6:
        _, py = m2px(r, 0.0)
        d.line([(0, py), (w, py)], fill=GRID + (255,))
        if r % lbl_every == 0 and py > 30:      # keep clear of the panel title
            d.text((4, py - 13), f"{r} m", fill=GRID_LBL, font=F_TINY)
        r += 5
    r = 5
    while r <= ymax + 1e-6:
        for sgn in (-1, 1):
            px, _ = m2px(0.0, sgn * r)
            d.line([(px, pad_t), (px, by)], fill=GRID + (255,))
        r += 5
    d.line([(cx, pad_t), (cx, by)], fill=(48, 58, 70, 255))

    # --- layer 1: vocabulary shadow (RAW anchors, pre-refinement) ------------
    for a in anchors:
        _line(d, [m2px(x, y) for x, y in poly(a)], SHADOW + (46,), 1)

    # --- layer 2: the scored fan (ALL refined proposals, ascending score) ----
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    top = set(sorted(range(len(probs)), key=lambda i: -probs[i])[:TOP_K_EMPH])
    for i in order:
        if i == sel:
            continue
        t = p_to_t(probs[i])
        col = viridis(t)
        emph = i in top
        _line(d, [m2px(x, y) for x, y in poly(fan[i])],
              col + ((255,) if emph else (int(18 + 216 * t),)),
              (1 + int(round(2.6 * t))) + (2 if emph else 0))
        if emph:                                  # layer 3: top-8 waypoint dots
            for p in fan[i]:
                px, py = m2px(float(p[0]), float(p[1]))
                d.ellipse([px - 2.5, py - 2.5, px + 2.5, py + 2.5], fill=col)

    # --- the oracle proposal (best available in the fan), thin cyan outline --
    if oracle is not None and oracle != sel:
        _line(d, [m2px(x, y) for x, y in poly(fan[oracle])],
              (90, 220, 255, 190), 2)
        px, py = m2px(float(fan[oracle][-1][0]), float(fan[oracle][-1][1]))
        d.ellipse([px - 5, py - 5, px + 5, py + 5], outline=(90, 220, 255),
                  width=2)
        d.text((px + 8, py - 7), "oracle", fill=(90, 220, 255), font=F_TINY)

    # --- layer 4: THE SELECTED PLAN — colour halo under a white core ---------
    scol = viridis(p_to_t(probs[sel]))
    spts = [m2px(x, y) for x, y in poly(fan[sel])]
    _line(d, spts, scol + (235,), 11)
    _line(d, spts, COL_SEL_CORE + (255,), 4)
    for j, p in enumerate(fan[sel]):
        px, py = m2px(float(p[0]), float(p[1]))
        d.ellipse([px - 7, py - 7, px + 7, py + 7], fill=scol,
                  outline=COL_SEL_CORE, width=2)
        d.text((px + 10, py - 6), f"{WP_STEPS[j] / 10:.1f}s",
               fill=(226, 232, 240), font=F_TINY)

    # --- layer 5: GT, dashed, ON TOP (the reference must never be occluded) --
    # Per-horizon ERROR TIE-LINES first: REF-C's dominant error is LONGITUDINAL
    # (right speed profile, wrong distance) and a longitudinal miss along a
    # near-straight path is invisible in BEV — the plan and the GT trace almost
    # the same line and differ only in WHERE the 0.5/1/1.5/2 s marks sit. Each
    # tie-line is literally one of the four terms the ADE averages.
    # They are drawn as OFFSET dimension lines (one lateral lane per horizon):
    # laid on the path itself they would be exactly collinear with it and
    # invisible under the selected plan.
    for j, (a, b) in enumerate(zip(fan[sel], gt_wp)):
        pa, pb = m2px(float(a[0]), float(a[1])), m2px(float(b[0]), float(b[1]))
        off = 14.0 + 10.0 * j
        qa, qb = (pa[0] + off, pa[1]), (pb[0] + off, pb[1])
        d.line([pa, qa], fill=(255, 110, 100, 110), width=1)
        d.line([pb, qb], fill=(255, 110, 100, 110), width=1)
        _dashed(d, [qa, qb], (255, 96, 86, 240), width=3, dash=5, gap=4)
        err = math.dist((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        if abs(qa[1] - qb[1]) > 16:
            d.text((qa[0] + 4, (qa[1] + qb[1]) / 2 - 6), f"{err:.1f}",
                   fill=(255, 140, 130), font=F_TINY)
    _dashed(d, [m2px(x, y) for x, y in poly(gt_wp)], COL_GT + (255,), width=4)
    for p in gt_wp:
        px, py = m2px(float(p[0]), float(p[1]))
        d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=COL_GT,
                  outline=(10, 30, 14), width=2)
    ex, ey = m2px(float(gt_wp[-1][0]), float(gt_wp[-1][1]))
    d.text((ex - 64, ey - 7), "GT 2.0s", fill=COL_GT, font=F_TINY)

    # --- ego ----------------------------------------------------------------
    d.polygon([(cx - 7, by + 6), (cx + 7, by + 6), (cx, by - 10)],
              fill=(236, 240, 246))
    d.rectangle([0, 0, w - 1, h - 1], outline=PANEL_EDGE, width=1)
    d.rectangle([1, 1, w - 2, 20], fill=(8, 11, 15, 225))
    d.text((8, 5), f"PLAN FAN — metric BEV (ego frame, heading up) · "
                   f"0-{xmax:.0f} m fwd · +-{ymax:.0f} m lat · 5 m grid",
           fill=HUD_DIM, font=F_TINY)
    return im


# ============================================================================
# camera context panel + colorbar + legend
# ============================================================================

def draw_cam(w, h, rgb_hwc, proj, fan, probs, sel, gt_path):
    """Small camera-projection context panel (Sayed's standard keeps camera and
    BEV together). GT + the top-8 fan + the selected plan, flat-ground pinhole."""
    im = Image.fromarray(rgb_hwc).resize((w, h), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    up = w / 256.0
    g = proj(gt_path, up=up)
    if len(g) >= 2:
        _dashed(d, g, COL_GT + (255,), width=4, dash=9, gap=6)
    for i in sorted(range(len(probs)), key=lambda i: probs[i])[-TOP_K_EMPH:]:
        if i == sel:
            continue
        _line(d, proj(poly(fan[i]), up=up),
              viridis(p_to_t(probs[i])) + (170,), 2)
    s = proj(poly(fan[sel]), up=up)
    _line(d, s, viridis(p_to_t(probs[sel])) + (235,), 7)
    _line(d, s, COL_SEL_CORE + (255,), 3)
    for x, y in proj(fan[sel], up=up):
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=COL_SEL_CORE, width=2)
    d.rectangle([0, 0, w - 1, 18], fill=HUD_BG)
    d.text((6, 3), "camera context (flat pinhole) · GT dashed · top-8 + pick",
           fill=HUD_DIM, font=F_TINY)
    d.rectangle([0, 0, w - 1, h - 1], outline=PANEL_EDGE, width=1)
    return im


def draw_colorbar(d, n_anchors):
    """Horizontal viridis bar on the FIXED log-probability scale + the uniform
    (1/N) reference tick — the semantics of the fan's colours."""
    x, y = BAR_XY
    w, h = BAR_WH
    d.text((x, y - 42), "proposal score = softmax confidence (log scale)",
           fill=HUD_FG, font=F_SUB)
    for i in range(w):
        d.line([(x + i, y), (x + i, y + h)], fill=viridis(i / (w - 1.0)))
    d.rectangle([x, y, x + w, y + h], outline=PANEL_EDGE, width=1)
    for p in BAR_TICKS:
        px = x + p_to_t(p) * w
        d.line([(px, y + h), (px, y + h + 5)], fill=GRID_LBL)
        lab = "1" if p >= 1.0 else f"1e{int(round(math.log10(p)))}"
        d.text((px - 9, y + h + 7), lab, fill=GRID_LBL, font=F_TINY)
    ux = x + p_to_t(1.0 / n_anchors) * w
    d.line([(ux, y - 8), (ux, y + h + 2)], fill=(255, 120, 110), width=2)
    d.text((min(ux - 24, x + w - 62), y - 22), f"uniform 1/{n_anchors}",
           fill=(255, 120, 110), font=F_TINY)


LEGEND = (
    ("layers, back to front", None),
    ("raw anchor vocabulary (pre-refinement)", SHADOW),
    ("refined proposals, colour = score", (49, 104, 142)),
    ("top-8 by score (thicker + waypoint dots)", (110, 206, 88)),
    ("SELECTED plan (white core + score halo)", COL_SEL_CORE),
    ("best-available proposal (oracle-in-fan)", (90, 220, 255)),
    ("ground truth, dashed", COL_GT),
    ("per-horizon error (the 4 ADE terms)", (255, 110, 100)),
)


def draw_legend(d, note_lines):
    y = LEG_Y
    for text, col in LEGEND:
        if col is None:
            d.text((LEG_X, y), text, fill=HUD_FG, font=F_SUB)
        else:
            d.line([(LEG_X, y + 8), (LEG_X + 22, y + 8)], fill=col, width=4)
            d.text((LEG_X + 30, y + 1), text, fill=HUD_DIM, font=F_TINY)
        y += 19
    y += 6
    for ln in note_lines:
        d.text((LEG_X, y), ln, fill=(120, 130, 146), font=F_TINY)
        y += 13


# ============================================================================
# frame assembly
# ============================================================================

def compose(rec, ctx):
    """One rendered frame [CW, CH] from a per-frame record + clip context."""
    im = Image.new("RGB", (CW, CH), HUD_BG)
    d = ImageDraw.Draw(im, "RGBA")
    # plain-python once per frame: 256 proposals x 4 waypoints of tensor
    # element access dominates the render otherwise.
    fan = rec["fan"].tolist()
    gt_wp, gt_path = rec["gt_wp"].tolist(), rec["gt_path"].tolist()
    bev = draw_bev(BEV_WH[0], BEV_WH[1], ctx["anchors"], fan,
                   rec["probs"], rec["sel"], gt_wp, ctx["xmax"],
                   oracle=rec["oracle"])
    im.paste(bev, BEV_XY)
    if ctx["proj"] is not None and rec["rgb"] is not None:
        im.paste(draw_cam(CAM_WH[0], CAM_WH[1], rec["rgb"], ctx["proj"],
                          fan, rec["probs"], rec["sel"], gt_path), CAM_XY)
    draw_colorbar(d, ctx["n_anchors"])
    draw_legend(d, ctx["legend_notes"])

    lim = CW - 16
    gap = rec["ade"] - rec["oracle_ade"]
    l0 = _fit(f"{ctx['title']} · step {ctx['step']} · {ctx['corpus']} "
              f"ep{ctx['ep']:02d} {ctx['tag']} · frame {rec['t']:03d}", F_TOP, lim)
    l1 = _fit(f"tactical: {rec['man']}    strategic: route {rec['route']}    "
              f"v0 {rec['v0']:.1f} m/s", F_HUD, lim)
    l2 = _fit(f"ADE(selected) {rec['ade']:.2f} m   oracle-in-fan "
              f"{rec['oracle_ade']:.2f} m   gap {gap:+.2f} m   "
              f"vocab-oracle {rec['vocab_ade']:.2f} m   |   clip mean: sel "
              f"{ctx['mean_ade']:.2f} / oracle {ctx['mean_oracle']:.2f} m",
              F_SUB, lim)
    l3 = _fit(f"top-1 p {rec['top1']:.3f}   entropy {rec['H']:.2f}/"
              f"{ctx['h_max']:.2f} nats   modes>1% {rec['n_modes']}   |   "
              f"{ctx['n_anchors']} anchors, ALL refined by {ctx['steps']} "
              f"truncated-denoise steps (no top-K gate)", F_SUB, lim)
    d.rectangle([0, 0, CW, HUD_H], fill=HUD_BG)
    d.text((8, 6), l0, fill=HUD_FG, font=F_TOP)
    d.text((8, 30), l1, fill=HUD_FG, font=F_HUD)
    d.text((8, 52), l2, fill=(255, 190, 120) if gap > 0.5 else HUD_DIM,
           font=F_SUB)
    d.text((8, 74), l3, fill=HUD_DIM, font=F_SUB)
    return im


# ============================================================================
# inference
# ============================================================================

@torch.no_grad()
def episode_planfan(model, ep, device, window, steps, batch=4, max_frames=400):
    """Stride-1 REF-C decode keeping the FULL proposal set per frame.

    Calls the model exactly as taniteval.refc_eval.collect does (nav=follow, v0
    through the measurement encoder, ``steps`` truncated-denoise steps) so every
    number here is the same quantity the leaderboard row reports. Returns
    t -> record; t = window end (the pose the ego frame is anchored to).
    """
    frames, poses = ep.frames, ep.poses.float()
    T = min(frames.shape[0], poses.shape[0])
    starts = list(range(0, T - window - K))[:max_frames]
    out = {}
    for i in range(0, len(starts), batch):
        ch = starts[i:i + batch]
        last = torch.tensor([s + window - 1 for s in ch])
        fw = torch.stack([torch.as_tensor(frames[s:s + window])
                          for s in ch]).to(device).float().div_(255.0)
        v0 = poses[last, 3].to(device)
        o = model(fw, nav_cmd=None, v0=v0, steps=steps)     # follow-command eval
        logits = o["anchor_logits"].float().cpu()           # [b, N] POST-H19
        fan = o["anchor_traj"].float().cpu()                # [b, N, S, 2] refined
        sel = o["sel_idx"].cpu()
        # Faithfulness check: with grounded_selector off the selection IS the
        # argmax of the returned logits. Fail loud if a ckpt breaks that.
        assert torch.equal(sel, logits.argmax(dim=1)), (
            "sel_idx != argmax(anchor_logits) — the decoder is scoring with "
            "something other than the returned logits (grounded_selector?); the "
            "fan colours would not be the selection score. Refusing to render.")
        probs = torch.softmax(logits, dim=1)
        man = o["maneuver_logits"].argmax(-1).cpu().tolist()
        route = o["route_logits"].argmax(-1).cpu().tolist()
        for j, s in enumerate(ch):
            t = s + window - 1
            gt_path = ego_future_path(poses, t, K)
            gt_wp = gt_path[WP_IDX]
            de = torch.linalg.norm(fan[j] - gt_wp[None], dim=-1).mean(-1)  # [N]
            k = int(de.argmin())
            p = probs[j]
            out[t] = dict(
                t=t, fan=fan[j], probs=p.tolist(), sel=int(sel[j]), oracle=k,
                gt_wp=gt_wp, gt_path=gt_path,
                ade=float(de[int(sel[j])]), oracle_ade=float(de[k]),
                top1=float(p.max()),
                H=float(-(p * (p.clamp_min(1e-12)).log()).sum()),
                n_modes=int((p > MODE_THRESH).sum()),
                v0=float(poses[t, 3]), man=pretty_man(man[j]),
                route=pretty_route(route[j]), rgb=None)
    return out


# ============================================================================
# clip driver
# ============================================================================

def render_clip(model, ep, ep_idx, tag, ctx0, device, args):
    anchors = model.decoder.anchors.detach().float().cpu()        # [N, S, 2]
    recs = episode_planfan(model, ep, device, ctx0["window"], ctx0["steps"],
                           batch=args.batch, max_frames=args.max_frames)
    ts = sorted(recs)
    if not ts:
        print(f"[skip] ep{ep_idx:02d} {tag}: too few frames", flush=True)
        return None
    # vocabulary-only oracle (what the RAW anchors could reach) — coverage floor
    for t in ts:
        r = recs[t]
        dv = torch.linalg.norm(anchors - r["gt_wp"][None], dim=-1).mean(-1)
        r["vocab_ade"] = float(dv.min())

    mean_ade = sum(recs[t]["ade"] for t in ts) / len(ts)
    mean_or = sum(recs[t]["oracle_ade"] for t in ts) / len(ts)
    mean_vo = sum(recs[t]["vocab_ade"] for t in ts) / len(ts)
    # Clip-stable isotropic BEV range: cover v0*2 s + margin, and never clip the
    # GT or the selected plan. Held fixed for the clip (a per-frame rescale makes
    # the fan's apparent spread meaningless frame-to-frame).
    need = max([max(r["v0"] for r in recs.values()) * 2.0 + 8.0, 20.0]
               + [float(r["gt_wp"][:, 0].max()) + 5.0 for r in recs.values()]
               + [float(r["fan"][r["sel"]][:, 0].max()) + 5.0
                  for r in recs.values()])
    xmax = min(90.0, 5.0 * math.ceil(need / 5.0))

    ctx = dict(ctx0, ep=ep_idx, tag=tag, xmax=xmax, anchors=anchors.tolist(),
               mean_ade=mean_ade, mean_oracle=mean_or)
    name = (f"refc-planfan_step{ctx0['step']}_ep{ep_idx:02d}_{tag}")
    worst = max(ts, key=lambda t: recs[t]["ade"])
    multimodal = sorted(ts, key=lambda t: -recs[t]["H"])[:args.stills]

    print(f"[clip] ep{ep_idx:02d} {tag}: frames={len(ts)} BEV 0-{xmax:.0f} m  "
          f"selADE {mean_ade:.3f}  oracleADE {mean_or:.3f}  "
          f"vocabOracle {mean_vo:.3f}  gap {mean_ade - mean_or:+.3f}  "
          f"worst f{worst} sel {recs[worst]['ade']:.3f} vs oracle "
          f"{recs[worst]['oracle_ade']:.3f}", flush=True)

    # per-frame evidence table (the ep11 analysis reads from this)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    jpath = OUT.parent / f"planfan_{name}.json"
    jpath.write_text(json.dumps(dict(
        clip=name, ep=ep_idx, tag=tag, step=ctx0["step"], model=ctx0["model_key"],
        n_anchors=ctx0["n_anchors"], denoise_steps=ctx0["steps"],
        all_anchors_denoised=True, xmax_m=xmax, frames=len(ts),
        mean_ade_selected=mean_ade, mean_ade_oracle_in_fan=mean_or,
        mean_ade_oracle_vocab=mean_vo,
        worst_frame=dict(t=worst, ade=recs[worst]["ade"],
                         oracle=recs[worst]["oracle_ade"],
                         vocab=recs[worst]["vocab_ade"],
                         top1=recs[worst]["top1"], H=recs[worst]["H"],
                         n_modes=recs[worst]["n_modes"], v0=recs[worst]["v0"]),
        per_frame=[dict(t=t, ade=recs[t]["ade"], oracle=recs[t]["oracle_ade"],
                        vocab=recs[t]["vocab_ade"], top1=recs[t]["top1"],
                        H=recs[t]["H"], n_modes=recs[t]["n_modes"],
                        v0=recs[t]["v0"], sel=recs[t]["sel"],
                        oracle_idx=recs[t]["oracle"]) for t in ts]),
        indent=1))

    # stills first (worst window + the most multimodal moments)
    stills = []
    for t, why in [(worst, "worstwindow")] + [(t, "multimodal")
                                              for t in multimodal]:
        r = recs[t]
        r["rgb"] = ep.frames[t, -3:].permute(1, 2, 0).numpy()
        p = OUT / f"{name}_f{t:03d}_{why}.png"
        compose(r, ctx).save(p)
        r["rgb"] = None
        stills.append(str(p))
        print(f"[still] {p}  ADE {r['ade']:.2f} oracle {r['oracle_ade']:.2f} "
              f"H {r['H']:.2f} modes {r['n_modes']}", flush=True)
    if args.stills_only:
        return dict(name=name, mp4=None, stills=stills, json=str(jpath),
                    frames=len(ts), mean_ade=mean_ade, mean_oracle=mean_or,
                    mean_vocab=mean_vo, worst=worst,
                    worst_ade=recs[worst]["ade"],
                    worst_oracle=recs[worst]["oracle_ade"])

    fdir = OUT / f"_frames_{name}"
    fdir.mkdir(parents=True, exist_ok=True)
    for n, t in enumerate(ts):
        r = recs[t]
        r["rgb"] = ep.frames[t, -3:].permute(1, 2, 0).numpy()
        compose(r, ctx).save(fdir / f"f{n:04d}.png")
        r["rgb"] = None
    mp4 = OUT / f"{name}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-r", str(args.fps), "-i", str(fdir / "f%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", str(mp4)], check=True, capture_output=True)
    shutil.rmtree(fdir)
    print(f"[video] {mp4}  frames={len(ts)}", flush=True)
    return dict(name=name, mp4=str(mp4), stills=stills, json=str(jpath),
                frames=len(ts), mean_ade=mean_ade, mean_oracle=mean_or,
                mean_vocab=mean_vo, worst=worst, worst_ade=recs[worst]["ade"],
                worst_oracle=recs[worst]["oracle_ade"])


def main():
    ap = argparse.ArgumentParser("plan_fan")
    ap.add_argument("--model", default="refc-xl-30k")
    ap.add_argument("--corpus", default="physicalai",
                    choices=[c["key"] for c in CORPORA])
    ap.add_argument("--clips", default=DEFAULT_CLIPS)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--stills", type=int, default=2,
                    help="top-entropy (most multimodal) PNG stills per clip")
    ap.add_argument("--stills-only", action="store_true")
    ap.add_argument("--horizon", type=float, default=None)
    args = ap.parse_args()
    device = "cuda"

    entry = [m for m in MODELS if m["key"] == args.model][0]
    assert entry["arch"] == "refc", (
        f"{args.model} is arch={entry['arch']}: the plan fan is REF-C's anchored "
        "proposal set. For CEM/MPC candidates (v3/P2) feed this renderer the "
        "candidate rollouts + planning COST (see PLANNER_VIZ_CONCEPT.md).")
    corp = [c for c in CORPORA if c["key"] == args.corpus][0]
    files = sorted(Path(corp["root"]).glob("ep_*.pt"))
    assert files, f"no episodes under {corp['root']}"
    clips = [(int(a.split(":")[0]), a.split(":")[1])
             for a in args.clips.split(",") if a.strip()]

    L = loaders.load(entry, device)
    assert L["step_readout"] is None, "REF-C must have no grounded step_readout"
    model, step = L["model"], L["step"]
    assert not model.cfg.refc1, "refc1 ckpt: horizons are distances, not times"
    assert tuple(model.cfg.trajectory.horizons) == WP_STEPS
    assert not model.cfg.grounded_selector, (
        "grounded_selector=True: the selection score is conf + a progress proxy, "
        "so the fan colours (softmax of conf) are NOT the selection score. Add "
        "the proxy to the colour before rendering.")
    steps = model.cfg.decoder.diffusion_steps if \
        entry.get("mode", "diffusion") == "diffusion" else 0
    n_anchors = int(model.decoder.anchors.shape[0])
    window = int(model.cfg.window)
    cy = args.horizon if args.horizon is not None else HORIZON[args.corpus]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[load] {args.model} step={step} anchors={n_anchors} "
          f"denoise_steps={steps} window={window} "
          f"graft_maneuver={model.cfg.graft_maneuver} "
          f"grounded_selector={model.cfg.grounded_selector}", flush=True)

    ctx0 = dict(model_key=args.model, step=step, corpus=args.corpus,
                title="REF-C-XL anchored diffusion · plan fan",
                n_anchors=n_anchors, steps=steps, window=window,
                h_max=math.log(n_anchors), proj=FlatProjector(cy),
                legend_notes=[
                    f"score = softmax over ALL {n_anchors} anchor logits (H19",
                    "maneuver prior included) from the t=0 classifier pass;",
                    f"geometry is post-{steps}-denoise-step — those passes' own",
                    "confidences are discarded by the decoder.",
                    "polylines: ego -> the 4 waypoints, straight segments",
                    "(drawing device); all metrics use the waypoints only."])

    from tanitad.data.mixing import load_episode
    done = []
    for idx, tag in clips:
        ep = load_episode(str(files[idx]), mmap=True)
        r = render_clip(model, ep, idx, tag, ctx0, device, args)
        if r:
            done.append(r)
        del ep
        torch.cuda.empty_cache()

    print("PLAN_FAN_DONE", flush=True)
    for r in done:
        print(f"  {r['name']}: frames={r['frames']} selADE {r['mean_ade']:.3f} "
              f"oracleADE {r['mean_oracle']:.3f} "
              f"gap {r['mean_ade'] - r['mean_oracle']:+.3f} "
              f"worst f{r['worst']} {r['worst_ade']:.3f} vs "
              f"{r['worst_oracle']:.3f} -> {r['mp4']}")


if __name__ == "__main__":
    main()
