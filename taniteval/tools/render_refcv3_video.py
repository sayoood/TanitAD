#!/usr/bin/env python3
"""render_refcv3_video.py — the FIVE-PANEL refcv3 planner reel (PI, 2026-09-03).

⛔⛔ READ ``taniteval/tools/REFCV3_ARM.md`` §2 FIRST. It is the DEFINITION of
what refcv3's arm IS, and every claim this renderer draws on a frame is bounded
by it. This file draws pixels; it invents no number.

WHAT IS ON SCREEN, AND WHAT IT MEANS
====================================
The PI asked for five things together. Each is a declared ``VizElement``
(``tanitad.viz_standard``), so a silently missing panel is impossible — an
element that cannot be computed renders its REASON, never a blank.

1. **CAMERA** — the front image with the GROUND-TRUTH future path (green) and
   refcv3's SELECTED trajectory (orange) projected into it.
2. **METRIC BEV** — refcv3's whole fan: ``anchor_traj`` [128, 8, 2], every
   anchor coloured by its SELECTION SCORE, the selected one drawn on top, GT in
   the same metric frame.
3. **TACTICAL** — the FACTORED heads, ``lat_logits_tac`` and ``lon_logits_tac``,
   as two separate 8-class distributions against the v7.2 labels. ⛔ They are
   NEVER collapsed into one 5-way row: the whole point of the factored head is
   that lateral and longitudinal are separate decisions, and the 5-way collapse
   is the programme's single largest known tactical defect.
4. **STRATEGIC** — ``route_logits`` against the v2.1 per-window route label,
   beside the v7.2 nav token that was FED. ⛔ The fed token is drawn in its own
   ``strategic_input`` slot and marked GIVEN INPUT — putting it in the
   ``strategic`` slot is the nav-echo defect and ``viz_standard.check_frame``
   REFUSES it.
5. **HUD** — the measured ``v0`` at t0, the nav token, the checkpoint step.

⛔ THE THREE THINGS THIS FILE REFUSES TO GET WRONG
==================================================
**(a) THE SELECTION IS ``out["traj"]``, NEVER ``a_star``.** ``a_star`` is the
anchor nearest the GROUND TRUTH (``refc_v3_train.py:460``) — an ORACLE. Drawing
it as "the model's choice" would be a fabricated result. It is rendered only
under ``--with-oracle``, in its own colour, with the word ORACLE in the legend
and on the frame. The deployed selection is ``sel_score_v3``-ranked
(``refc_v3.py:509-527`` on the hier arm; ``refc.py:1531-1534`` on the flat arm)
and is read straight off ``out["traj"]``.

**(b) FRAMES ARE BRIDGED LOCALLY (C79).** The overlay is drawn on
``item["frames"][-1][-3:]`` — the exact uint8 tensor ``frames_to_device`` hands
the encoder, upscaled with nearest-neighbour so no pixel is invented. Nothing is
re-decoded, re-fetched or re-cropped between scoring and drawing.

**(c) THE PROJECTION IS THE CLIP'S OWN, AND IT IS CYLINDRICAL.** The B1 cache is
``256x640, f_ref 305.5775, projection="cylindrical"`` — read from the
``*.v2ep.pt`` payload's own ``frame`` field, never assumed. On an
equidistant-azimuth raster the column is LINEAR IN AZIMUTH; applying the pinhole
formula ``u = cx - f*Y/X`` there is wrong and looks entirely plausible (it
implies 92.6 deg where the rig is a 120 deg wide). :class:`CylProjector` inverts
``calib.cylindrical_rays`` exactly, and REFUSES a projection it cannot express
rather than approximating it with the nearest one it can.
The extrinsics are **per-clip and MEASURED**, from the dataset's own
``calibration/sensor_extrinsics`` parquet: camera height on PhysicalAI is
1.245-1.607 m and the three constants circulating in this repo (1.22 / 1.43 /
1.5) are all wrong as a constant. Without ``--extrinsics`` the camera overlay is
DISABLED and says so on the frame — the BEV is calibration-independent and
carries the comparison either way.

USAGE
=====
    python taniteval/tools/render_refcv3_video.py \
        --ckpt   <run>/ckpt_30000.pt --config <run>/config.json \
        --episodes <dir with the chosen *.v2ep.pt> \
        --labels   <.../s2_labels_v7.2_eval.jsonl.gz> \
        --extrinsics <extrinsics.json>  --with-oracle \
        --out <out>/refcv3_five_panel.mp4

⛔ CORRECTED 2026-09-04 — THIS DOCSTRING USED TO SAY *"re-render against the
FINAL checkpoint with ``--ckpt <run>/ckpt_40284_FINAL.pt``"*. **THAT FILE DOES
NOT EXIST AND NOTHING WILL EVER WRITE ONE**, and the same file said so 590 lines
below in ``--expect-step``'s own help: ``refc_v3_train.py:103`` sets
``MILESTONES = (5000, 15000, 20000, 30000)`` and the only milestone write is
``ckpt_{step}.pt`` gated on membership, so 40 284 is not a milestone. **The
final checkpoint is the ROLLING ``ckpt.pt``** — a file whose NAME never changes
while its CONTENTS do, which is exactly what makes a stale copy
indistinguishable from the real one. An agent following the old line would have
gone looking for a file that is not there and, worse, could have manufactured it
by copying whatever ``ckpt.pt`` held at that moment without reading its step.
*(Class: a true instruction that implies a wrong next action — the same family
as the stale-blocker sweep.)*

THE FINAL RE-RENDER, AS ACTUALLY PERFORMED (MEASURED 2026-09-04)
===============================================================
Freeze the rolling file under an immutable name, **verify the step from the
checkpoint's own ``step`` field, never from the filename**, and md5 both ends::

    # pod-side, only after `ps -eo args | grep -c 'sup[_]refcv3'` reads 0
    python - <<'EOF'
    import torch; ck = torch.load("…/refcv3-b1-v72-30k/ckpt.pt", map_location="cpu")
    assert ck["step"] == 40284, ck["step"]          # ⛔ the whole point
    torch.save({k: v for k, v in ck.items() if k != "opt"},
               "…/refcv3-b1-v72-30k/ckpt_step40284_frozen.pt")
    EOF
    # -> 428,518,255 B  md5 b1ed7075ff730d0993d2eaa3c86f6b56  (opt state dropped:
    #    the rolling ckpt.pt is 1,284,991,701 B and the optimiser never crosses
    #    the wire for a render)

then the IDENTICAL render command with two arguments changed::

    --ckpt <run>/ckpt_step40284_frozen.pt  --expect-step 40284 \
    --out  <out>/refcv3_five_panel_step40284.mp4

``--expect-step`` reads ``ck["step"]`` and REFUSES before any GPU is spent if it
disagrees. Nothing else changes: the step is read from the checkpoint and burned
into the banner and the sidecar, so two reels can never be confused.

⭐ RUN IT FROM THE CODE THAT TRAINED THE CHECKPOINT, NOT FROM ``HEAD``.
MEASURED 2026-09-04: the off-Drive clone used for this render carried
``stack/tanitad/refs/refc_v3.py``, ``refs/refc.py`` and
``scripts/refc_v3_train.py`` **md5-identical to the TRAINING POD's**, while the
G: worktree had already diverged on all three (the concurrent REF-C v4 stream,
E11'/E14/X15, landed 23:59-00:15 — after the run ended at 22:43). Rendering from
the worktree would have scored the checkpoint under code it was never trained
with. The STRICT load (0 missing / 0 unexpected keys) is the check that settles
it, and it is recorded in the sidecar.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE_PARENT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_TE_PARENT)


def _load_by_path(name: str, path: str):
    import importlib.util
    if not os.path.exists(path):
        sys.exit(f"[render_refcv3] required sibling {path} is missing")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: ⭐ THE ARM ADAPTER, IMPORTED — never re-implemented. ``refcv3_arm`` owns the
#: STRICT model rebuild, the eval-time seam neutralisation, the trainer's own
#: window dataset with the v7.2 label join and the nav source, and the grid
#: index-select. Importing it is what keeps ONE definition of "refcv3's arm" in
#: the programme; a renderer that rebuilt the model its own way could show a
#: different model from the one the eval scores.
arm = _load_by_path("refcv3_arm_for_render", os.path.join(_HERE, "refcv3_arm.py"))

import torch                                                       # noqa: E402
from PIL import Image, ImageDraw, ImageFont                        # noqa: E402

from tanitad.data import v7_labels as v7l                          # noqa: E402
from tanitad.models.v6 import (tactical_lat_actions,               # noqa: E402
                               tactical_lon_actions_v)
from tanitad.refs.refb import NAV_COMMANDS, ROUTE_CLASSES          # noqa: E402
from tanitad.viz_standard import VizElement, check_frame           # noqa: E402

DT_FRAME = 0.1

# --------------------------------------------------------------------------- #
# COLOUR SEMANTICS — one meaning per colour, in EVERY panel                    #
# --------------------------------------------------------------------------- #
C_GT = (110, 231, 138)          # GROUND TRUTH, everywhere, always green
C_SEL = (255, 158, 61)          # refcv3's OWN selection, everywhere, orange
C_ORACLE = (186, 128, 255)      # the a_star ORACLE ceiling — violet, always labelled
C_GIVEN = (240, 190, 90)        # a GIVEN INPUT (the nav token) — amber
C_FAN_LO = (46, 62, 84)         # fan anchor, low selection score
C_FAN_HI = (56, 189, 248)       # fan anchor, high selection score
C_DEAD = (58, 44, 52)           # an anchor the reachability guard killed
C_BG = (9, 12, 17)
C_PANEL = (16, 21, 29)
C_BAR = (34, 44, 58)
C_GRID = (34, 42, 53)
C_FG = (233, 238, 245)
C_DIM = (140, 152, 168)
C_WARN = (245, 180, 90)
C_OK = (110, 231, 138)
C_BAD = (248, 113, 113)

# --------------------------------------------------------------------------- #
# LAYOUT                                                                       #
# --------------------------------------------------------------------------- #
PAD = 12
CAM_UP = 2                                   # 256x640 -> 512x1280
W_LEFT, H_CAM = 1280, 512
W_RIGHT = 604
W_TOT = PAD + W_LEFT + PAD + W_RIGHT + PAD   # 1920
Y_BAN, H_BAN = 0, 74
Y_CAM = Y_BAN + H_BAN + 8
Y_TAC, H_TAC = Y_CAM + H_CAM + 12, 254
Y_STR, H_STR = Y_TAC + H_TAC + 12, 152
H_BEV = (Y_STR + H_STR) - Y_CAM
Y_LEG, H_LEG = Y_STR + H_STR + 12, 0         # legend rides in the BEV column
Y_HUD = Y_STR + H_STR + 12
H_HUD = 74
H_TOT = Y_HUD + H_HUD + PAD


def _p(*a):
    print(*a, flush=True)


_FONT_SRC = [None]


def font(size: int, bold: bool = False):
    """A REAL truetype face at video scale, or a loud refusal.

    ``taniteval.flagship_overlay._font`` silently falls back to PIL's 11 px
    bitmap default when DejaVu is absent — which is every Windows box. At 1920 px
    that is unreadable, and it fails by looking merely ugly rather than by
    raising, so it is exactly the class of defect that ships."""
    cands = (["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else
             ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for c in cands:
        if os.path.exists(c):
            _FONT_SRC[0] = _FONT_SRC[0] or c
            return ImageFont.truetype(c, size)
    raise SystemExit(
        "[render_refcv3] no scalable font found (tried %s). PIL's bitmap "
        "default is 11 px and is illegible at 1920 px, so this refuses rather "
        "than rendering a reel nobody can read." % ", ".join(cands))


def fit(d, text, f, max_w):
    """One line, ELLIPSISED rather than clipped — PIL draws past the edge with
    no error, and a cut word reads as data (MEASURED once: the whole caveat
    'NOT a hierarchy result' was absent from every frame of a reel)."""
    if d.textlength(text, font=f) <= max_w:
        return text
    while text and d.textlength(text + "…", font=f) > max_w:
        text = text[:-1]
    return text + "…"


def wrap(d, text, f, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if d.textlength(t, font=f) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def lerp(a, b, t):
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


# =========================================================================== #
# THE CAMERA MODEL                                                            #
# =========================================================================== #
class ProjectionRefused(ValueError):
    """The frame's projection is one this module cannot express — refuse.

    Approximating a cylinder with a pinhole is the failure this class exists to
    make impossible: it produces a plausible-looking overlay that is wrong, and
    an overlay is the one artefact a reader believes without checking.
    """


class CylProjector:
    """Ego-frame ground point -> output pixel, for the cache's OWN frame.

    THE FORWARD MODEL, inverted exactly from ``tanitad.data.calib``:

    ``cylindrical_rays`` (``calib.py:906``) gives, for output pixel (u, v),
    the CAMERA-frame ray ``(sin phi, y_n, cos phi)`` with
    ``phi = (u - (W-1)/2)/f_ref`` and ``y_n = (v - (H-1)/2)/f_ref``; camera axes
    are ``+x right, +y DOWN, +z boresight`` (``ftheta_project_ray``). Inverting,
    a camera-frame direction ``d`` lands at::

        phi = atan2(d.x, d.z)
        u   = (W-1)/2 + f_ref * phi
        v   = (H-1)/2 + f_ref * d.y / hypot(d.x, d.z)

    ⭐ The denominator is the RADIAL distance ``hypot(x, z)``, not the forward
    distance ``z``. That is the whole difference from the pinhole, and it is why
    a pinhole overlay on this raster bends the wrong way at the edges.

    ``cylindrical_rectify`` centres the ray fan on the clip's own principal
    point, so the camera frame here IS the clip's optical frame and the
    extrinsic rotation below is the only transform needed. The pinhole branch is
    kept so a legacy 256x256/266 cache still renders through ONE projector.

    EXTRINSICS ARE PER-CLIP AND MEASURED (``calibration/sensor_extrinsics``):
    ``t`` is the camera position in the vehicle frame and ``R`` its orientation,
    so ``p_cam = R^T (p_veh - t)``. The camera sits ~2.0-2.1 m FORWARD of the
    vehicle origin and 1.29-1.58 m above the ground on the clips rendered here;
    both terms matter at the near field, where the trajectory is.
    """

    def __init__(self, frame: dict, extr: dict | None):
        self.h = int(frame["height"])
        self.w = int(frame["width"])
        self.f = float(frame["f_ref"])
        self.proj = str(frame["projection"])
        if self.proj not in ("cylindrical", "pinhole"):
            raise ProjectionRefused(
                f"projection {self.proj!r} is not one this renderer can "
                f"express (known: cylindrical, pinhole). REFUSING rather than "
                f"approximating it with the nearest one.")
        self.cx = (self.w - 1) / 2.0
        self.cy = (self.h - 1) / 2.0
        self.enabled = extr is not None
        if extr is None:
            self.R = self.t = None
            self.source = "DISABLED (no per-clip extrinsics)"
            return
        q = (float(extr["qx"]), float(extr["qy"]),
             float(extr["qz"]), float(extr["qw"]))
        self.R = _quat_to_R(*q)                       # camera -> vehicle
        self.t = np.array([float(extr["x"]), float(extr["y"]),
                           float(extr["z"])], dtype=np.float64)
        self.source = extr.get("source", "sensor_extrinsics parquet")
        axis = self.R @ np.array([0.0, 0.0, 1.0])
        self.pitch_down_deg = math.degrees(
            math.atan2(-axis[2], math.hypot(axis[0], axis[1])))

    def label(self) -> str:
        if not self.enabled:
            return (f"camera overlay DISABLED — no per-clip extrinsics "
                    f"({self.h}x{self.w} {self.proj} f_ref {self.f:.2f})")
        return (f"{self.h}x{self.w} {self.proj} f_ref {self.f:.3f} · "
                f"cam h {self.t[2]:.3f} m, fwd {self.t[0]:.3f} m, pitch-down "
                f"{self.pitch_down_deg:+.2f}° (MEASURED per-clip)")

    def __call__(self, pts_xy, up: int = CAM_UP):
        """[N, 2] ego-frame ground points (x fwd, y left) -> list of (px, py).

        A point behind the image plane, outside the raster, or nearer than
        ``0.5 m`` radially is DROPPED rather than clamped: a clamped point draws
        a line to a place the model never predicted."""
        if not self.enabled:
            return []
        out = []
        for p in np.asarray(pts_xy, dtype=np.float64):
            pv = np.array([float(p[0]), float(p[1]), 0.0])
            pc = self.R.T @ (pv - self.t)             # vehicle -> camera
            x, y, z = float(pc[0]), float(pc[1]), float(pc[2])
            if self.proj == "cylindrical":
                rho = math.hypot(x, z)
                if rho < 0.5 or z <= 0.0:
                    out.append(None)
                    continue
                u = self.cx + self.f * math.atan2(x, z)
                v = self.cy + self.f * y / rho
            else:
                if z < 0.5:
                    out.append(None)
                    continue
                u = self.cx + self.f * x / z
                v = self.cy + self.f * y / z
            if not (-self.w <= u <= 2 * self.w and -self.h <= v <= 2 * self.h):
                out.append(None)
                continue
            out.append((u * up, v * up))
        return out


def _quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def polyline(d, pts, fill, width):
    """Draw a polyline that may contain None (dropped) points, as SEGMENTS.

    ⛔ Never bridge across a dropped point: the bridge is a line the model did
    not predict, drawn through a region the projection refused."""
    run = []
    for p in pts:
        if p is None:
            if len(run) >= 2:
                d.line(run, fill=fill, width=width, joint="curve")
            run = []
        else:
            run.append(p)
    if len(run) >= 2:
        d.line(run, fill=fill, width=width, joint="curve")


def densify(path_xy, n=64):
    """[S, 2] -> [n, 2] by arc-length-parameterised LINEAR interpolation, with
    the ego origin (0, 0) prepended.

    ⚠️ DECLARED ON THE FRAME: refcv3 emits EIGHT points (0.5/1/1.5/2/3/4/5/6 s).
    The smooth curve on screen is those eight points joined, not a denser
    prediction. The emitted slots are drawn as explicit markers on top so the
    real resolution of the output is never hidden by the interpolation."""
    p = np.concatenate([np.zeros((1, 2)), np.asarray(path_xy, dtype=np.float64)])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-6:
        return p
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1)


# =========================================================================== #
# PANELS                                                                      #
# =========================================================================== #
def draw_bev(size, gt, sel, fan, score, keep, oracle, goal_pt, past,
             xmax, ymax, F, slot_s, bank_note="", gt_trunc_s=None):
    """The metric BEV: THE FAN, scored — and the one calibration-independent panel.

    ⭐ Why the fan is the point. refcv3 is an ANCHOR model: its output is a
    choice among 128 pre-computed trajectories, and the thing that decides
    whether it drives is the RANKING, not the shape of any one path. A BEV that
    shows only the winner shows the least informative half of the model. Every
    anchor is drawn, coloured by its own ``sel_score_v3``; the ranking is
    therefore visible as a shape.

    ⛔ ``keep`` is ``out["reach_keep"]``: anchors the reachability guard killed
    before the argmax. They are drawn in a DIFFERENT colour, not omitted — an
    anchor that was excluded is evidence about the guard, and hiding it would
    make the guard invisible.

    This panel depends on no calibration at all. If the camera overlay were
    wrong, this one would still be right — which is why it is 604 px wide here
    and not a 152 px inset."""
    w, h = size
    im = Image.new("RGB", (w, h), (7, 10, 14))
    d = ImageDraw.Draw(im, "RGBA")
    pad, top = 40, 88
    cx, by = w // 2, h - pad

    def m2px(X, Y):
        return (cx - (Y / ymax) * ((w / 2) - pad),
                by - (max(X, 0.0) / xmax) * (by - top))

    step = 20 if xmax > 60 else (10 if xmax > 25 else 5)
    r = step
    while r <= xmax + 0.1:
        _, py = m2px(r, 0)
        d.line([(10, py), (w - 10, py)], fill=C_GRID)
        if py - 15 > top - 4:        # a label the header would swallow is not
            d.text((13, py - 15), f"{r} m",      # drawn: a half-covered "60 m"
                   fill=(104, 116, 132), font=F["tiny"])   # reads as "6 m"
        r += step
    for lat in (-ymax / 2, ymax / 2):
        px, _ = m2px(0, lat)
        d.line([(px, top), (px, by)], fill=(26, 33, 42))
    d.line([(cx, top), (cx, by)], fill=(44, 54, 67))

    # ---- THE FAN: every anchor, coloured by its selection RANK -------------- #
    # ⚠️ RANK, not the raw score, and the reason is MEASURED on this checkpoint:
    # `sel_score_v3` spans roughly [-30, +2] because a handful of anchors are
    # driven far negative, so a min-max ramp puts ~120 of the 128 in the top
    # decile of the colour scale and the picture reads "every anchor is a
    # winner". The rank ramp is monotone in the same quantity and shows the
    # ORDERING, which is what the argmax actually consumes.
    if fan is not None:
        order = np.argsort(score)                    # best drawn LAST, on top
        rank = np.empty(len(score), dtype=np.float64)
        rank[order] = np.arange(len(score)) / max(len(score) - 1, 1)
        for i in order:
            t = float(rank[i])
            alive = bool(keep[i]) if keep is not None else True
            col = lerp(C_FAN_LO, C_FAN_HI, t) if alive else C_DEAD
            a = int(34 + 170 * (t ** 2.2)) if alive else 60
            pts = [m2px(float(q[0]), float(q[1])) for q in fan[i]]
            d.line([m2px(0, 0)] + pts, fill=col + (a,),
                   width=2 if t > 0.93 else 1)

    if past is not None and len(past) >= 2:
        d.line([m2px(float(a_), float(b_)) for a_, b_ in past],
               fill=(74, 86, 100), width=2)

    if oracle is not None:                            # the CEILING, labelled
        d.line([m2px(0, 0)] + [m2px(float(q[0]), float(q[1])) for q in oracle],
               fill=C_ORACLE + (235,), width=3)

    d.line([m2px(0, 0)] + [m2px(float(q[0]), float(q[1])) for q in gt],
           fill=C_GT, width=7)
    d.line([m2px(0, 0)] + [m2px(float(q[0]), float(q[1])) for q in sel],
           fill=C_SEL, width=4)
    for j, q in enumerate(sel):                       # the EIGHT emitted slots
        x, y = m2px(float(q[0]), float(q[1]))
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=C_SEL, width=2,
                  fill=(11, 15, 21, 255))
    for q in gt:
        x, y = m2px(float(q[0]), float(q[1]))
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=C_GT)
    if goal_pt is not None:                           # E9's predicted 2 s goal
        x, y = m2px(float(goal_pt[0]), float(goal_pt[1]))
        d.line([(x - 9, y), (x + 9, y)], fill=C_SEL + (220,), width=2)
        d.line([(x, y - 9), (x, y + 9)], fill=C_SEL + (220,), width=2)
        d.text((max(x - 78, 6), y - 9), "goal_point_tac", fill=C_SEL,
               font=F["micro"])
    d.polygon([(cx - 7, by), (cx + 7, by), (cx, by - 14)], fill=(232, 236, 242))
    d.rectangle([0, 0, w, 80 if gt_trunc_s is not None else 66],
                fill=(7, 10, 14))
    d.text((12, 3), "METRIC BEV · top-down · metres · "
                    "CALIBRATION-INDEPENDENT", fill=C_FG, font=F["sub"])
    d.text((12, 21), f"the whole fan: {0 if fan is None else len(fan)} anchors "
                     f"× 8 slots ({slot_s}), brightness = selection RANK",
           fill=C_DIM, font=F["tiny"])
    # ⚠️ STATED, because it changes how the picture reads: the two axes are
    # scaled INDEPENDENTLY so the lateral spread of the fan is visible at all.
    # At an equal aspect the whole fan would be a single vertical line and the
    # panel would show nothing.
    ratio = (xmax / max(2 * ymax, 1e-6)) * ((2 * (w / 2 - pad)) / max(by - top, 1))
    d.text((12, 36), f"x 0..{xmax:g} m, y ±{ymax:g} m — axes INDEPENDENT "
                     f"(lateral ≈{ratio:.1f}×)", fill=(112, 124, 140),
           font=F["micro"])
    d.text((12, 50), bank_note, fill=(112, 124, 140), font=F["micro"])
    if gt_trunc_s is not None:
        # ⛔ SAID, NEVER SILENT. `future_poses_ext` is clamped at the clip's
        # last pose, so an unmarked short GT would look like a stop the ego
        # never made.
        d.text((12, 64), f"⚠ GT drawn to {gt_trunc_s:g} s only — the clip ends "
                         f"before the model's 6 s horizon", fill=C_WARN,
               font=F["micro"])
    return im


def draw_scorebar(size, score, keep, sel_idx, a_star, F):
    """The selection surface as a 1-D strip: one column per anchor, in ANCHOR
    ORDER, height = score. The selected column is orange; the oracle's is violet.

    ⭐ This is the degeneracy gate, on the frame. REFCV3_ARM.md §4 records a
    random-init model that selected ONE anchor on 42/42 windows while the
    trivial-profile instrument read 0.0000 — a completely degenerate arm that a
    family table would have reported as scene understanding. A flat strip, or a
    strip whose orange column never moves, says that in one glance."""
    w, h = size
    im = Image.new("RGB", (w, h), C_PANEL)
    d = ImageDraw.Draw(im, "RGBA")
    n = len(score)
    x0, y0, y1 = 12, 50, h - 26
    bw = (w - 24) / n
    hi = float(np.max(score))
    # ⚠️ THE AXIS IS CLIPPED AT THE 5th PERCENTILE AND THE FRAME SAYS SO. A few
    # anchors are driven to ~-30 while the rest live in a ~2-wide band, so a
    # min-max axis flattens the whole surface into one indistinguishable block.
    lo = float(np.percentile(score, 5.0))
    true_lo = float(np.min(score))
    rng = max(hi - lo, 1e-9)
    rank = np.empty(n, dtype=np.float64)
    rank[np.argsort(score)] = np.arange(n) / max(n - 1, 1)
    for i in range(n):
        t = min(max((float(score[i]) - lo) / rng, 0.0), 1.0)
        alive = bool(keep[i]) if keep is not None else True
        col = lerp(C_FAN_LO, C_FAN_HI, float(rank[i])) if alive else C_DEAD
        x = x0 + i * bw
        d.rectangle([x, y1 - max(t * (y1 - y0), 1.0), x + max(bw - 0.6, 0.7), y1],
                    fill=col)
    marks = [(a_star, C_ORACLE, "a_star ORACLE"), (sel_idx, C_SEL, "SELECTED")]
    if a_star is not None and a_star == sel_idx:
        # ⭐ Worth saying explicitly: the deployed selection AGREEING with the
        # GT-nearest anchor is the good case, and it is not the same statement
        # as "the drawn path is the oracle". Collapsing the two markers stops
        # them overprinting into an unreadable smear.
        marks = [(sel_idx, C_SEL, "SELECTED = a_star (oracle agrees)")]
    for idx, col, tag in marks:
        if idx is None or idx < 0:
            continue
        x = x0 + idx * bw + bw / 2
        d.line([(x, y0 - 8), (x, y1)], fill=col + (230,), width=2)
        tw = d.textlength(tag, font=F["micro"])
        d.text((min(max(x + 5, 2), w - tw - 4), y0 - 22), tag, fill=col,
               font=F["micro"])
    n_dead = 0 if keep is None else int((~np.asarray(keep)).sum())
    d.text((12, 6), "SELECTION SURFACE · sel_score_v3 per anchor "
                    "(anchor index →)", fill=C_FG, font=F["sub"])
    d.text((12, h - 22),
           f"argmax #{sel_idx}   max {hi:+.2f}   axis floor = 5th pct "
           f"{lo:+.2f} (true min {true_lo:+.2f})   killed by reach_keep: "
           f"{n_dead}/{n}", fill=C_DIM, font=F["micro"])
    return im


#: Axis labels for the 8-wide factored heads. ⚠️ ABBREVIATION ONLY — the FULL
#: token is always printed in the `pred`/`GT` line beneath, so nothing on this
#: frame can be read as a class name that does not exist in the vocabulary.
SHORT = {
    "LANE_KEEP": "LANE_KEEP", "LANE_CHANGE_L": "LC_L", "LANE_CHANGE_R": "LC_R",
    "ABORT_LC": "ABORT", "NUDGE_L": "NUDGE_L", "NUDGE_R": "NUDGE_R",
    "TURN_L": "TURN_L", "TURN_R": "TURN_R",
    "FOLLOW": "FOLLOW", "CRUISE": "CRUISE", "YIELD_MERGE": "YIELD",
    "BRAKE_TO": "BRAKE", "CREEP": "CREEP", "HOLD": "HOLD",
    "ADAPT_SPEED_FOR_CURVE": "ADAPT_CRV", "ACCELERATE": "ACCEL",
}


def draw_factor(d, x, y, w, title, classes, probs, pred, gt, F,
                gt_note: str | None):
    """ONE factored head as a labelled bar row: probs, prediction, GT.

    ⛔ LAT and LON get one of these EACH and are never merged. The 5-way
    collapse (``refc_tactical.COLLAPSE_TABLE``) destroys the longitudinal
    decision on every turn row — a turn absorbs it entirely — so a single 5-way
    bar chart CANNOT show a braking-into-a-turn error at all. That defect is the
    reason the factored head exists, and rendering it collapsed would hide
    exactly what this panel was asked to show."""
    d.text((x, y), title, fill=C_FG, font=F["sub"])
    n = len(classes)
    bw = (w - 8) / n
    top, bot = y + 22, y + 108
    d.line([(x, bot), (x + w - 8, bot)], fill=C_GRID)
    for i, cname in enumerate(classes):
        bx = x + i * bw
        p = float(probs[i])
        hgt = p * (bot - top)
        is_pred, is_gt = (i == pred), (gt is not None and i == gt)
        col = C_SEL if is_pred else C_BAR
        d.rectangle([bx + 2, bot - hgt, bx + bw - 4, bot], fill=col)
        if is_gt:                                   # GT: a green cap + underline
            d.rectangle([bx + 2, bot - max(hgt, 3) - 4, bx + bw - 4,
                         bot - max(hgt, 3)], fill=C_GT)
            d.rectangle([bx + 2, bot + 2, bx + bw - 4, bot + 5], fill=C_GT)
        short = SHORT.get(cname, cname)
        tw = d.textlength(short, font=F["micro"])
        d.text((bx + (bw - 6 - tw) / 2, bot + 8), short,
               fill=(C_FG if (is_pred or is_gt) else C_DIM), font=F["micro"])
        if p >= 0.06:
            pt = f"{p:.2f}"
            d.text((bx + (bw - 6 - d.textlength(pt, font=F["micro"])) / 2,
                    max(top - 2, bot - hgt - 14)), pt, fill=C_DIM,
                   font=F["micro"])
    ok = (gt is not None and pred == gt)
    d.text((x, y + 134), f"model: {classes[pred]}   p={float(probs[pred]):.2f}",
           fill=C_SEL, font=F["sub"])
    if gt is None:
        d.text((x, y + 154), fit(d, f"v7.2 GT: — {gt_note}", F["sub"], w - 10),
               fill=C_WARN, font=F["sub"])
    else:
        d.text((x, y + 154),
               f"v7.2 GT: {classes[gt]}   "
               f"{'MATCH' if ok else '≠  MISMATCH'}",
               fill=(C_OK if ok else C_BAD), font=F["sub"])


def draw_hbars(d, x, y, w, classes, probs, pred, gt, F, gt_note):
    """The 3-class route head as horizontal bars (few classes, long names)."""
    rh = 26
    for i, cname in enumerate(classes):
        yy = y + i * rh
        p = float(probs[i])
        is_pred, is_gt = (i == pred), (gt is not None and i == gt)
        d.rectangle([x + 106, yy + 4, x + 106 + (w - 170) * p, yy + rh - 8],
                    fill=C_SEL if is_pred else C_BAR)
        d.text((x, yy + 3), cname.replace("route_", ""),
               fill=(C_FG if (is_pred or is_gt) else C_DIM), font=F["sub"])
        d.text((x + w - 56, yy + 3), f"{p:.2f}", fill=C_DIM, font=F["sub"])
        if is_gt:
            d.rectangle([x + 100, yy + 2, x + 104, yy + rh - 6], fill=C_GT)
    ok = (gt is not None and pred == gt)
    if gt is None:
        d.text((x, y + 3 * rh + 6), f"GT — {gt_note}", fill=C_WARN,
               font=F["sub"])
    else:
        d.text((x, y + 3 * rh + 6),
               f"GT {ROUTE_CLASSES[gt].replace('route_', '')}   "
               f"{'MATCH' if ok else 'MISMATCH'}",
               fill=(C_OK if ok else C_BAD), font=F["sub"])


# =========================================================================== #
def load_extrinsics(path: str | None, clip_ids) -> dict:
    """clip_id -> the MEASURED per-clip camera extrinsic, or {} (overlay off).

    ⚠️ NOT a constant, and not derivable from the rig label. MEASURED over 40
    PhysicalAI clips: camera height 1.245-1.607 m (CV 7.4 %, 37 distinct values
    in 40), while the rig-A/rig-B medians differ by 1.5 % — so conditioning on
    the rig cannot substitute for reading the clip. A missing entry disables the
    camera overlay for that clip and says so ON THE FRAME."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        tab = json.load(fh)
    out = {}
    for cid in clip_ids:
        e = tab.get(str(cid))
        if e:
            out[str(cid)] = e
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", required=True,
                    help="a dir of *.v2ep.pt — the clips to render")
    ap.add_argument("--labels", required=True, help="the v7.2 EVAL blob")
    ap.add_argument("--extrinsics", default=None,
                    help="JSON: clip_id -> {qx,qy,qz,qw,x,y,z}. WITHOUT it the "
                         "camera overlay is DISABLED and says so on the frame.")
    ap.add_argument("--nav-source", default="v72", choices=["v72", "none", "auto"])
    ap.add_argument("--grid", default="2s", choices=["2s", "6s"])
    ap.add_argument("--action-units", default="steer", choices=["steer", "kappa"])
    ap.add_argument("--out", required=True, help="output .mp4 FILE")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames-per-ep", type=int, default=0)
    ap.add_argument("--clips", default="",
                    help="comma-separated clip_ids, rendered IN THIS ORDER. "
                         "Default: every clip under --episodes, sorted. ⛔ The "
                         "order and the selection are written into the sidecar "
                         "— a hand-picked reel must never be quotable as a "
                         "representative one.")
    ap.add_argument("--with-oracle", action="store_true",
                    help="also draw the a_star (GT-nearest) anchor. ALWAYS "
                         "labelled ORACLE on the frame; never the model's choice.")
    ap.add_argument("--stills", default="",
                    help="comma-separated frame indices to also save as PNG")
    ap.add_argument("--stills-dir", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lru", type=int, default=4)
    ap.add_argument("--episodes-n", type=int, default=0)
    ap.add_argument("--allow-nonstrict", action="store_true")
    ap.add_argument("--expect-step", type=int, default=0,
                    help="⛔ REFUSE a checkpoint whose OWN `step` differs. There "
                         "is no `ckpt_<N>_FINAL.pt`: `refc_v3_train.py:103` sets "
                         "MILESTONES = (5000, 15000, 20000, 30000) and the only "
                         "milestone write is `ckpt_{step}.pt` (:1197-1199), so the "
                         "FINAL checkpoint is the rolling `ckpt.pt` (:1090/:1192) "
                         "— a file whose NAME never changes while its contents do. "
                         "Copy it to an immutable name, then pass the step you "
                         "believe you copied. VERIFIED BY CONTENT, from ck['step'].")
    ap.add_argument("--keep-frames", action="store_true")
    a = ap.parse_args(argv)
    if os.path.isdir(a.out):
        sys.exit(f"--out must be a FILE, got a directory: {a.out}")

    # ⛔ THE ENCODER IS RESOLVED BEFORE THE RENDER, NEVER AFTER IT. A missing
    # ffmpeg discovered at the encode step throws away every frame just
    # rendered — on a 3-clip reel that is ~500 frames of GPU.
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            sys.exit("no ffmpeg on PATH and imageio-ffmpeg not installed. "
                     "Refusing to render frames that could not be encoded.")

    import refb_labels
    tr = arm.trainer()
    dev = a.device
    model, cfg, targs, prov = arm.load_model(a.ckpt, a.config, dev,
                                             a.allow_nonstrict)
    if a.expect_step and int(prov.get("step") or -1) != int(a.expect_step):
        sys.exit(f"[render_refcv3] ⛔ STEP MISMATCH: --expect-step "
                 f"{a.expect_step} but {a.ckpt} carries step "
                 f"{prov.get('step')!r}. The final checkpoint is the ROLLING "
                 f"`ckpt.pt`, whose name never changes while its contents do — "
                 f"so 'the final reel' rendered from a stale copy is "
                 f"indistinguishable from the real one except by this check. "
                 f"Refusing before the GPU is spent.")
    hier = bool(cfg.hier)
    grid = arm.grid_slots(cfg.core.trajectory.horizons, a.grid)
    eps, files, clip_ids, ds, lman, join, nav_src, raw_off = arm.build_corpus(
        a, cfg, prov)
    steps = int(prov["decoder_steps"])
    W = int(cfg.core.window)
    horizons = list(cfg.core.trajectory.horizons)
    slot_s = [round(h * DT_FRAME, 1) for h in horizons]
    LAT = list(tactical_lat_actions(cfg.tac_vocab_version))
    LON = list(tactical_lon_actions_v(cfg.tac_vocab_version))
    _p(f"[model] {a.ckpt} step={prov['step']} arm={prov['arm']} "
       f"anchors={prov['n_anchors']} slots={slot_s}s window={W} "
       f"decoder={prov['decoder_mode']}/{steps}")
    _p(f"[vocab] tac_vocab={cfg.tac_vocab_version} lat={len(LAT)} lon={len(LON)} "
       f"route={list(ROUTE_CLASSES)}")

    extr = load_extrinsics(a.extrinsics, clip_ids)
    _p(f"[calib] per-clip extrinsics for {len(extr)}/{len(clip_ids)} clips"
       + ("" if extr else "  ⚠ CAMERA OVERLAY WILL BE DISABLED"))

    F = {"ban": font(24, True), "hud": font(17), "sub": font(15),
         "tiny": font(13), "micro": font(11), "big": font(20, True)}
    _p(f"[font] {_FONT_SRC[0]}")

    anchors_bank = model.core.decoder.anchors.detach().float().cpu()
    by_ep: dict[int, list] = {}
    for wi, (e_i, t) in enumerate(ds.index):
        if wi % max(1, a.stride) == 0:
            by_ep.setdefault(e_i, []).append((wi, t))
    if a.clips:
        want = [c.strip() for c in a.clips.split(",") if c.strip()]
        pos = {str(c): i for i, c in enumerate(clip_ids)}
        bad = [c for c in want if c not in pos]
        if bad:
            sys.exit(f"[render_refcv3] --clips names {bad} which are not under "
                     f"--episodes ({len(clip_ids)} clips there)")
        ep_order = [pos[c] for c in want if pos[c] in by_ep]
    else:
        ep_order = sorted(by_ep)

    frames_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)),
                              "_frames_" + os.path.basename(a.out).rsplit(".", 1)[0])
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    stills = {int(x) for x in a.stills.split(",") if x.strip()} if a.stills else set()
    stills_dir = a.stills_dir or frames_dir
    if stills:
        os.makedirs(stills_dir, exist_ok=True)

    n_out, t0w, per_ep = 0, time.time(), []
    viz_rows = None
    for e_i in ep_order:
        ep = ds.episodes[e_i]
        cid = str(clip_ids[e_i])
        pay = torch.load(os.path.join(a.episodes, files[e_i]), map_location="cpu",
                         weights_only=False)
        # ⭐ THE FRAME IS THE CACHE'S OWN, read from the payload — never assumed.
        fr_dict = pay.get("frame") or {"height": int(pay["image_h"]),
                                       "width": int(pay["image_w"]),
                                       "f_ref": 266.0, "projection": "pinhole"}
        del pay
        proj = CylProjector(fr_dict, extr.get(cid))
        rec = lman.by_clip.get(cid) if hasattr(lman, "by_clip") else None
        _p(f"[clip] {cid} frame={fr_dict} extrinsics={'yes' if proj.enabled else 'NO'}")
        _p(f"[calib] {proj.label()}")

        wins = by_ep[e_i]
        if a.max_frames_per_ep:
            wins = wins[:a.max_frames_per_ep]
        # ---- fixed BEV extent for the whole clip (temporal stability) ------ #
        # ⛔ FIXED PER CLIP, NEVER PER FRAME. A per-frame autoscale makes the
        # ego appear to move when only the axis moved, and it silently rescales
        # the error the panel exists to show.
        # ⛔ THE EXTENT IS FITTED TO THE SCENE, NOT TO THE ANCHOR BANK, AND THE
        # FAN IS DELIBERATELY LET RUN OFF THE PANEL. MEASURED on this
        # checkpoint's bank: at the 6 s slot the anchors reach 224 m forward and
        # ±166 m lateral (p90 |y| = 116 m). An extent that contained them would
        # compress the GT-vs-selected comparison — the thing the panel is FOR —
        # into a few pixels, and would still not "show the whole fan" in any
        # useful sense. Rays leaving the frame are the honest picture: the bank
        # spans a far wider space than any one scene needs, and the interesting
        # question is which of them the score ranks, not where they all end.
        # The bank's true extent is printed on the panel so the clipping is
        # declared rather than discovered.
        xmax, ymax = 20.0, 6.0
        for wi, t in wins[::5]:
            it = ds[wi]
            fvx = it["future_valid_ext"]
            vs = [j for j, h in enumerate(horizons) if bool(fvx[h - 1])]
            if not vs:
                continue
            g = refb_labels.waypoint_targets(
                it["pose_last"].float()[None],
                it["future_poses_ext"].float()[None], horizons)[0].numpy()[vs]
            xmax = max(xmax, float(g[:, 0].max()))
            ymax = max(ymax, float(np.abs(g[:, 1]).max()))
        xmax = min(120.0, 10 * (int(xmax // 10) + 1))
        ymax = round(max(8.0, 1.35 * ymax), 1)
        anc = anchors_bank.numpy()
        bank_note = (f"bank reaches {anc[..., 0].max():.0f} m fwd / "
                     f"±{np.abs(anc[..., 1]).max():.0f} m lat at 6 s — most "
                     f"anchors leave this frame")
        ades = []
        for k_i, (wi, t) in enumerate(wins):
            item = ds[wi]
            t0 = t + W - 1
            fv = item["future_valid_ext"]
            # ⛔ THE ADMISSION TEST IS THE **GRID**, NOT ALL EIGHT SLOTS, AND
            # THE DIFFERENCE IS WHERE THE MANOEUVRES ARE. `future_poses_ext`
            # is CLAMPED at the clip's last pose, so an out-of-range slot
            # silently repeats it — scoring or drawing that would be inventing
            # ground truth. Requiring all eight (6 s) valid would be safe but
            # costs the last 6 s of every clip: on ca11a2a2 that is the entire
            # left turn (13.0-19.0 s of a 20.5 s clip), i.e. the test would
            # have deleted exactly the event the reel exists to show.
            # ⇒ admit on the SCORED grid, and draw GT only up to the last
            # genuinely valid slot, with the truncation printed on the panel.
            valid_slots = [j for j, h in enumerate(horizons) if bool(fv[h - 1])]
            if not all(bool(fv[horizons[j] - 1]) for j in grid["slots"]):
                continue
            gt_trunc_s = (None if len(valid_slots) == len(horizons)
                          else round(horizons[valid_slots[-1]] * DT_FRAME, 1))
            pose_last = item["pose_last"].float()
            v0 = float(pose_last[3])
            nav_id = int(item["nav_cmd"]) if "nav_cmd" in item else 0
            nav_ok = bool(item["nav_valid"]) if "nav_valid" in item else False
            traj_tgt = refb_labels.waypoint_targets(
                pose_last[None], item["future_poses_ext"].float()[None],
                horizons)                                        # [1, S, 2]
            gt8 = traj_tgt[0].numpy()
            gt_draw = gt8[valid_slots]        # only slots the clip really has

            # ---- the model: ONE forward, exactly the arm's ----------------- #
            frames_u8 = item["frames"]                           # [W, C, H, Wpx]
            fr = tr.frames_to_device(frames_u8[None], dev)
            nav_t = (torch.tensor([nav_id], dtype=torch.long, device=dev)
                     if nav_src == "v72" else None)
            v0_t = torch.full((1,), v0, dtype=torch.float32, device=dev)
            with torch.no_grad():
                out = model(fr, nav_cmd=nav_t, v0=v0_t, steps=steps)
            sel = out["traj"][0].float().cpu().numpy()           # THE selection
            fan = out["anchor_traj"][0].float().cpu().numpy()    # [128, 8, 2]
            rank_key = "sel_score_v3" if "sel_score_v3" in out else "sel_score"
            score = out[rank_key][0].float().cpu().numpy()
            keep = (out["reach_keep"][0].cpu().numpy()
                    if "reach_keep" in out else None)
            sel_idx = int(out["sel_idx"][0])
            # the ORACLE, computed exactly as the trainer does — a CEILING only
            sv = torch.stack([fv[h - 1] for h in horizons]).float().cpu()
            dist = (((traj_tgt.float().cpu()[:, None] - anchors_bank[None])
                     ** 2).sum(-1) * sv[None, None]).sum(-1)
            a_star = int(dist.argmin(dim=1)[0])
            oracle = fan[a_star] if a.with_oracle else None
            goal_pt = (out["goal_point_tac"][0].float().cpu().numpy()
                       if "goal_point_tac" in out else None)

            lat_key = "lat_logits_tac" if hier else "lat_decision"
            lon_key = "lon_logits_tac" if hier else "lon_decision"
            lat_p = torch.softmax(out[lat_key][0].float(), -1).cpu().numpy()
            lon_p = torch.softmax(out[lon_key][0].float(), -1).cpu().numpy()
            rte_p = torch.softmax(out["route_logits"][0].float(), -1).cpu().numpy()
            lat_pred, lon_pred = int(lat_p.argmax()), int(lon_p.argmax())
            rte_pred = int(rte_p.argmax())
            lat_gt = int(item["lat_v7"]) if "lat_v7" in item else v7l.IGNORE_ID
            lon_gt = int(item["lon_v7"]) if "lon_v7" in item else v7l.IGNORE_ID
            lat_gt = None if lat_gt == v7l.IGNORE_ID else lat_gt
            lon_gt = None if lon_gt == v7l.IGNORE_ID else lon_gt
            rte_gt = (int(item["route_target"])
                      if bool(item.get("route_valid", False)) else None)
            if rte_gt is not None and not (0 <= rte_gt < len(ROUTE_CLASSES)):
                rte_gt = None

            # ADE of the DEPLOYED selection on the dump grid (the arm's own
            # instants), recomputed from the path — never converted from the
            # trainer's mean-L1 `eval_traj`.
            gsel = grid["slots"]
            ade = float(np.linalg.norm(sel[gsel] - gt8[gsel], axis=-1).mean())
            ades.append(ade)
            k3 = kin3_row(sel[gsel], gt8[gsel], grid["dt_s"])

            # ---- THE VIZ STANDARD: declare, or do not render ---------------- #
            band = "in band" if lat_gt is not None else \
                "outside the v7.2 record's ±2.0 s band"
            els = [
                VizElement.present(
                    "camera", "GT + selected trajectory" if proj.enabled
                    else "disabled",
                    source="CylProjector(v2ep['frame'], per-clip "
                           "sensor_extrinsics) on item['frames'][-1][-3:]",
                    kind="derived"),
                VizElement.present(
                    "bev", f"fan 128×8 + selection (anchor #{sel_idx})",
                    source="out['anchor_traj'], out['sel_score_v3'], "
                           "out['traj']", kind="model_output"),
                VizElement.present(
                    "tactical", f"lat={LAT[lat_pred]} lon={LON[lon_pred]}",
                    source="softmax(out['lat_logits_tac']) / "
                           "softmax(out['lon_logits_tac'])",
                    kind="model_output", conditioned_on=("nav_cmd",)),
                VizElement.present(
                    "strategic", ROUTE_CLASSES[rte_pred],
                    source="out['route_logits'].argmax(-1)",
                    kind="model_output", conditioned_on=("nav_cmd",)),
                VizElement.present(
                    "ade", f"{ade:.3f} m",
                    source="mean L2 of out['traj'] vs "
                           "refb_labels.waypoint_targets on the 2 s grid",
                    kind="derived"),
                VizElement.present(
                    "strategic_input", NAV_COMMANDS[nav_id],
                    source="V3Dataset.enable_nav_from_v7 (v7.2 nav_command, "
                           "provenance ego-future)", kind="given_input"),
                (VizElement.present(
                    "tactical_gt", f"lat={LAT[lat_gt]} lon={LON[lon_gt]}",
                    source="v7_labels.tactical_class_ids(record, t_now_s)",
                    kind="gt_label")
                 if lat_gt is not None and lon_gt is not None else
                 VizElement.unavailable(
                     "tactical_gt",
                     f"window {band}; v7_labels.tactical_class_ids returns "
                     f"IGNORE_ID and is NEVER clamped to a neutral class")),
            ]
            els = check_frame(els, where=f"refcv3 @ {cid}/w{wi}")
            if viz_rows is None:
                viz_rows = [e.to_dict() for e in els]

            # =============== CANVAS ======================================== #
            canvas = Image.new("RGB", (W_TOT, H_TOT), C_BG)
            d = ImageDraw.Draw(canvas)

            # ---- camera ---------------------------------------------------- #
            rgb = frames_u8[-1, -3:].permute(1, 2, 0).contiguous().numpy()
            cam = Image.fromarray(rgb).resize(
                (fr_dict["width"] * CAM_UP, fr_dict["height"] * CAM_UP),
                Image.NEAREST)          # NEAREST: no pixel invented (C79)
            cd = ImageDraw.Draw(cam)
            gt_dense = _ego_future(ep.poses, t0, 60)
            if proj.enabled:
                # ⭐ THE CALIBRATION SELF-CHECK, ON THE FRAME. The horizon row
                # is predicted from the clip's OWN extrinsics and drawn; if it
                # does not sit where the road meets the sky, the projection is
                # wrong and the viewer can see that without trusting a caption.
                # An overlay is the one artefact a reader believes on sight, so
                # it carries its own falsifier.
                hz = proj([[1e5, 0.0]])
                if hz and hz[0] is not None:
                    yh = hz[0][1]
                    cd.line([(0, yh), (W_LEFT, yh)], fill=(150, 132, 62),
                            width=1)
                    cd.text((10, yh - 16), "horizon predicted from the clip's "
                            "own extrinsics", fill=(178, 158, 78), font=F["micro"])
                polyline(cd, proj(gt_dense), C_GT, 9)
                if a.with_oracle:
                    polyline(cd, proj(densify(oracle)), C_ORACLE, 3)
                polyline(cd, proj(densify(sel)), C_SEL, 5)
                for q in proj(sel):
                    if q is None:
                        continue
                    cd.ellipse([q[0] - 7, q[1] - 7, q[0] + 7, q[1] + 7],
                               outline=C_SEL, width=3)
                for q in proj(gt_draw):
                    if q is None:
                        continue
                    cd.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4],
                               fill=C_GT)
            else:
                cd.rectangle([0, H_CAM // 2 - 22, W_LEFT, H_CAM // 2 + 22],
                             fill=(12, 16, 22))
                cd.text((16, H_CAM // 2 - 10),
                        "CAMERA OVERLAY DISABLED — no per-clip extrinsics "
                        "for this clip; the BEV panel carries the comparison",
                        fill=C_WARN, font=F["hud"])
            canvas.paste(cam, (PAD, Y_CAM))
            d.rectangle([PAD, Y_CAM, PAD + W_LEFT, Y_CAM + 22], fill=(9, 12, 17))
            d.text((PAD + 8, Y_CAM + 3),
                   fit(d, f"FRONT CAMERA — the exact bytes the model "
                          f"scored · {proj.label()}", F["tiny"], W_LEFT - 20),
                   fill=C_DIM, font=F["tiny"])

            # ---- BEV + selection surface ----------------------------------- #
            past = _past_path(ep.poses, t0, 40)
            bev_h = H_BEV - 142
            bev = draw_bev((W_RIGHT, bev_h), gt_draw, sel, fan, score, keep,
                           oracle, goal_pt, past, xmax, ymax, F,
                           "/".join(f"{s:g}" for s in slot_s) + " s",
                           bank_note, gt_trunc_s)
            canvas.paste(bev, (PAD + W_LEFT + PAD, Y_CAM))
            canvas.paste(draw_scorebar((W_RIGHT, 130), score, keep, sel_idx,
                                       a_star if a.with_oracle else None, F),
                         (PAD + W_LEFT + PAD, Y_CAM + bev_h + 12))

            # ---- tactical --------------------------------------------------- #
            d.rectangle([PAD, Y_TAC, PAD + W_LEFT, Y_TAC + H_TAC], fill=C_PANEL)
            d.text((PAD + 12, Y_TAC + 8),
                   "TACTICAL — the FACTORED heads, lateral and longitudinal "
                   "kept SEPARATE (never collapsed to one 5-way label)",
                   fill=C_FG, font=F["sub"])
            hw = (W_LEFT - 48) // 2
            draw_factor(d, PAD + 16, Y_TAC + 30, hw,
                        f"LATERAL · lat_logits_tac ({len(LAT)}-way, "
                        f"{cfg.tac_vocab_version})", LAT, lat_p, lat_pred,
                        lat_gt, F, band)
            draw_factor(d, PAD + 32 + hw, Y_TAC + 30, hw,
                        f"LONGITUDINAL · lon_logits_tac ({len(LON)}-way, "
                        f"{cfg.tac_vocab_version})", LON, lon_p, lon_pred,
                        lon_gt, F, band)
            # ⭐ THE PER-FRAME TACTICAL COMPARISON, and why it is here beside
            # the v7.2 row rather than instead of it. The v7.2 record covers
            # only a ±2.0 s band around the clip's own t0_s, so on most frames
            # the declared-head row has NO ground truth at all. The kinematic
            # factorisation is defined on EVERY window: it reads the same
            # (dyaw, dv, v0, v1) off the model's path and off the GT path
            # through ONE imported instrument (four_families.maneuver_kinematics
            # -> refc_tactical.factor_from_kinematics), so the two are
            # commensurable by construction.
            # ⚠️ It is a MANY-TO-ONE PROJECTION FOR METRICS, never a label
            # source: kin3 destroys exactly the distinctions the v7 vocabulary
            # exists to keep (NUDGE vs LANE_CHANGE, CREEP vs BRAKE_TO). Both
            # rows are shown because neither one alone is the tactical answer.
            lat_ok = k3["lat_sel"] == k3["lat_gt"]
            lon_ok = k3["lon_sel"] == k3["lon_gt"]
            d.text((PAD + 16, Y_TAC + 212),
                   "PATH-DERIVED (every frame; the SAME kinematic factoriser "
                   "the four-families eval uses, on the 2 s grid) — a "
                   "projection for METRICS, never a label:",
                   fill=C_DIM, font=F["tiny"])
            d.text((PAD + 16, Y_TAC + 230),
                   f"model path  lat={k3['lat_sel_name']:<11s} "
                   f"lon={k3['lon_sel_name']}", fill=C_SEL, font=F["sub"])
            d.text((PAD + 396, Y_TAC + 230),
                   f"GT path  lat={k3['lat_gt_name']:<11s} "
                   f"lon={k3['lon_gt_name']}", fill=C_GT, font=F["sub"])
            d.text((PAD + 790, Y_TAC + 230),
                   f"lateral {'MATCH' if lat_ok else '≠ MISMATCH'}   ·   "
                   f"longitudinal {'MATCH' if lon_ok else '≠ MISMATCH'}",
                   fill=(C_OK if (lat_ok and lon_ok) else C_BAD), font=F["sub"])

            # ---- strategic --------------------------------------------------- #
            d.rectangle([PAD, Y_STR, PAD + W_LEFT, Y_STR + H_STR], fill=C_PANEL)
            d.text((PAD + 12, Y_STR + 8),
                   "STRATEGIC — route head vs the v2.1 per-window route "
                   "label", fill=C_FG, font=F["sub"])
            draw_hbars(d, PAD + 16, Y_STR + 30, 420, list(ROUTE_CLASSES), rte_p,
                       rte_pred, rte_gt, F,
                       "route_valid False (no route event in the lookahead)")
            # --- column 2: the FED nav token, in its own declared slot ------- #
            # ⛔ THIS IS NOT THE STRATEGIC PREDICTION AND IS NEVER DRAWN AS ONE.
            # `viz_standard.check_frame` REFUSES a privileged kind in the
            # `strategic` slot, so the fed token lives in `strategic_input` and
            # is drawn in the GIVEN-INPUT colour with the words on the frame.
            # Flagship v1's route head scored 1.0000 as an exact bijection of
            # the nav it was fed; a viewer must never have to guess which of
            # these two boxes is the model's own decision.
            xs = PAD + 466
            d.text((xs, Y_STR + 30), "nav token FED to the model",
                   fill=C_DIM, font=F["sub"])
            d.text((xs, Y_STR + 50), NAV_COMMANDS[nav_id].upper(),
                   fill=C_GIVEN, font=F["big"])
            # ⛔ THE CAPTION MUST NOT IMPLY THE TOKEN DRIVES THE ROUTE HEAD.
            # "both heads are conditioned on it (E13)" is true of the WIRING and
            # is, on its own, misleading about the BEHAVIOUR: the route head is
            # nav-insensitive BY CONSTRUCTION. `route_logits = route_head(pooled)`
            # and `pooled` is the encoder's output over `frames` alone, while
            # `nav_cmd` is one-hot'd into the SIBLING `measurement(...)` branch
            # and never reaches `pooled` (`refs/refc.py`; the v3 `nav_inject`
            # path touches only `z_tac_raw` and `ctx`). A viewer shown only the
            # wiring would credit an oracle input for a decision it does not
            # drive, which is the nav-echo defect read backwards.
            #
            # ⚠️ THE STRUCTURE IS THE CLAIM; THE MEASUREMENT IS ITS WITNESS, AND
            # THE WITNESS BELONGS TO A DIFFERENT STEP. `paired_true_minus_
            # shuffled_accuracy = 0.0 [0.0, 0.0]` (paired episode-cluster
            # bootstrap, n = 3 622 windows / 128 episodes) was measured at step
            # **30 000** — `taniteval/results/refcv3-30k-openloop-20260903-
            # 2004.json`. Printing it bare on a step-40 284 frame would read as
            # measured HERE, which it was not; the step travels with it. The
            # architectural argument is what carries to this checkpoint, and it
            # is stronger than any single measurement because it is a file:line.
            #
            # ⚠️ AND IT IS NOT A DEFECT. On the 1 736 changed-nav windows the
            # head follows the LABEL at 0.7414 and the WRONG fed nav at 0.2224 —
            # vision-derived route skill, the opposite of flagship v1's 1.0000
            # echo of its own input. The caption says INSENSITIVE, never "broken".
            #
            # ⛔ AND THE SLICE IS ASSERTED, NOT TRUSTED. MEASURED 2026-09-04:
            # the first draft of this caption wrapped to SEVEN lines under a
            # `[:6]` cap, and the line the cap ate was *"MEASURED @ step
            # 30 000"* — the qualifier the caption was rewritten to add. A
            # silent truncation does not merely shorten a caption; it deletes
            # the part that bounds the claim, and the frame still looks
            # finished. So the cap now REFUSES rather than trims.
            NAV_CAP = 6
            nav_lines = wrap(
                d, "GIVEN INPUT, not a prediction — the clip's v7.2 "
                   "nav_command (ego-future); an ORACLE absent at deployment. "
                   "Both heads are WIRED to it (E13), but the route head is "
                   "nav-INSENSITIVE BY CONSTRUCTION: it reads pooled VISION, "
                   "not this token. (+0.0000 true−shuffled, MEASURED @ step "
                   "30 000.)",
                F["tiny"], 330)
            if len(nav_lines) > NAV_CAP:
                sys.exit(f"[render_refcv3] ⛔ the nav caption wraps to "
                         f"{len(nav_lines)} lines but only {NAV_CAP} fit. "
                         f"Trimming it silently would drop the LAST line, "
                         f"which is where the claim's bound lives "
                         f"({nav_lines[-1]!r}). Shorten the text or raise the "
                         f"panel; do not let the cap decide what the frame "
                         f"stops saying.")
            for li, ln in enumerate(nav_lines):
                d.text((xs, Y_STR + 72 + li * 13), ln, fill=C_GIVEN
                       if li == 0 else C_DIM, font=F["tiny"])
            if not nav_ok:
                d.text((xs, Y_STR + 74 + len(nav_lines) * 13),
                       "nav_valid = False — index 0 fed by "
                       "convention", fill=C_WARN, font=F["tiny"])
            # --- column 3: the strategic GOAL, a model output with no GT here - #
            if "g_str" in out:
                g_str = out["g_str"][0].float().cpu().numpy()
                xg = PAD + 830
                d.text((xg, Y_STR + 30), "strategic goal g_str (model output)",
                       fill=C_DIM, font=F["sub"])
                d.text((xg, Y_STR + 50),
                       f"bearing {math.degrees(math.atan2(g_str[1], g_str[0])):+.1f}°"
                       f"   (cos {g_str[0]:+.3f}, sin {g_str[1]:+.3f})",
                       fill=C_FG, font=F["sub"])
                d.text((xg, Y_STR + 70),
                       f"along-track preference {g_str[2]:+.3f}  "
                       f"(−1 near … +1 far)", fill=C_FG, font=F["sub"])
                for li, ln in enumerate(wrap(
                        d, "No GT beside it: g_str's training label is the LAN "
                           "corridor, which is TRAIN-ONLY and is not fed at "
                           "inference — so this frame shows the decision, not "
                           "its score.", F["tiny"], 400)[:3]):
                    d.text((xg, Y_STR + 94 + li * 15), ln, fill=C_WARN,
                           font=F["tiny"])

            # ---- banner ------------------------------------------------------ #
            d.rectangle([0, Y_BAN, W_TOT, Y_BAN + H_BAN], fill=(15, 20, 28))
            d.text((PAD, Y_BAN + 6),
                   fit(d, f"refcv3 (REF-C v3, {prov['arm']}) · ckpt step "
                          f"{prov['step']} · clip {cid} · frame {t0:03d} "
                          f"= t {t0 * DT_FRAME:.1f} s  ({k_i + 1}/{len(wins)})",
                       F["ban"], W_TOT - 2 * PAD), fill=C_FG, font=F["ban"])
            for li, ln in enumerate(wrap(
                    d, "ONE-SHOT PLANNER, NOT A ROLLOUT: one forward pass at "
                       "the window origin consumes the observed frames, the "
                       "v7.2 nav token and the measured v0 — no actions, "
                       "no loop, no future information. The drawn path is the "
                       "model's OWN sel_score_v3 choice among its 128 anchors "
                       "(out['traj']), NEVER the GT-nearest anchor.",
                    F["tiny"], W_TOT - 2 * PAD)[:2]):
                d.text((PAD, Y_BAN + 38 + li * 16), ln, fill=C_WARN,
                       font=F["tiny"])

            # ---- HUD --------------------------------------------------------- #
            d.rectangle([0, Y_HUD, W_TOT, Y_HUD + H_HUD], fill=(15, 20, 28))
            d.line([(0, Y_HUD), (W_TOT, Y_HUD)], fill=(44, 54, 67))
            d.text((PAD, Y_HUD + 6),
                   f"v0 {v0:5.2f} m/s (MEASURED at t0, the only ego input)"
                   f"     nav {NAV_COMMANDS[nav_id]}"
                   f"     ckpt step {prov['step']}"
                   f"     ADE(sel, {a.grid}) {ade:.3f} m"
                   f"     clip-mean {np.mean(ades):.3f} m",
                   fill=C_FG, font=F["hud"])
            leg = [("GT (green)", C_GT), ("refcv3 SELECTION (orange)", C_SEL)]
            if a.with_oracle:
                leg.append(("a_star ORACLE ceiling, NOT the model's choice "
                            "(violet)", C_ORACLE))
            leg.append(("GIVEN INPUT (amber)", C_GIVEN))
            leg.append(("the 128-anchor fan: dim→cyan by sel_score_v3 RANK; "
                        "dull red = killed by reach_keep", C_FAN_HI))
            xx = PAD
            for txt, col in leg:
                d.rectangle([xx, Y_HUD + 32, xx + 22, Y_HUD + 44], fill=col)
                d.text((xx + 28, Y_HUD + 30), txt, fill=C_DIM, font=F["tiny"])
                xx += 28 + int(d.textlength(txt, font=F["tiny"])) + 26
            d.text((PAD, Y_HUD + 50),
                   "camera path = the model's 8 emitted slots "
                   f"({'/'.join(f'{s:g}' for s in slot_s)} s, circles) joined by "
                   "linear interpolation — the circles are the real output "
                   "resolution.   BEV is calibration-independent.",
                   fill=(112, 124, 140), font=F["tiny"])

            canvas.save(os.path.join(frames_dir, f"f{n_out:06d}.png"))
            if n_out in stills:
                canvas.save(os.path.join(stills_dir, f"still_f{n_out:06d}.png"))
            n_out += 1
        per_ep.append({"clip_id": cid, "episode_index": int(e_i),
                       "frames": len(ades),
                       "mean_ade_sel_m": round(float(np.mean(ades)), 4),
                       "max_ade_sel_m": round(float(np.max(ades)), 4),
                       "camera_overlay": bool(proj.enabled),
                       "frame": fr_dict,
                       "extrinsics": extr.get(cid)})
        _p(f"  [clip {cid}] {len(ades)} frames  mean ADE(sel) "
           f"{np.mean(ades):.3f} m  [{time.time() - t0w:.0f}s, {n_out} total]")

    if not n_out:
        sys.exit("[render_refcv3] no frames rendered")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-r", str(a.fps),
                    "-i", os.path.join(frames_dir, "f%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-preset", "slow", "-movflags", "+faststart", a.out],
                   check=True, capture_output=True)
    if not a.keep_frames:
        shutil.rmtree(frames_dir)
    side = {
        "renderer": "taniteval/tools/render_refcv3_video.py",
        "model": {"ckpt": a.ckpt, "step": prov["step"], "arm": prov["arm"],
                  "n_anchors": prov["n_anchors"], "horizons": prov["horizons"],
                  "window": prov["window"], "decoder": prov["decoder_mode"],
                  "decoder_steps": prov["decoder_steps"],
                  "tac_vocab_version": prov["tac_vocab_version"],
                  "state_dict_load": prov["state_dict_load"]},
        "grid": grid, "nav": join["nav"], "labels": join["labels"],
        "fps": a.fps, "frames": n_out, "resolution": f"{W_TOT}x{H_TOT}",
        "duration_s": round(n_out / a.fps, 1),
        "with_oracle": bool(a.with_oracle),
        "clip_selection": (f"explicit --clips, in this order: {a.clips}"
                           if a.clips else
                           f"every clip under {a.episodes}, sorted"),
        "window_stride": int(a.stride),
        "max_frames_per_ep": int(a.max_frames_per_ep) or "all",
        "per_clip": per_ep,
        "viz_standard_elements": viz_rows,
        "what_is_drawn": (
            "The path is out['traj'] — refcv3's OWN sel_score_v3-ranked choice "
            "among its 128 anchors (refc_v3.py:509-527). It is NOT a_star, the "
            "GT-nearest anchor, which appears only under --with-oracle and is "
            "labelled ORACLE wherever it is drawn."),
        "what_this_is_not": (
            "NOT a rollout and NOT closed-loop driving: refcv3 consumes no "
            "actions and has no loop to close (REFCV3_ARM.md §2.4). One forward "
            "pass emits the whole 6 s path. The ADE shown is the deployed "
            "selection's, on the arm's own grid, recomputed from the path — "
            "never converted from the trainer's mean-L1 eval_traj."),
        "nav_is_an_oracle": (
            "The nav token is the clip's v7.2 nav_command, provenance "
            "ego-future. It will not exist at deployment; the tactical and "
            "route heads are conditioned on it (E13, refc_v3.py:437-441). "
            "⚠️ THAT IS THE WIRING, NOT THE BEHAVIOUR, and the two must be "
            "quoted together. The ROUTE head is nav-insensitive BY "
            "CONSTRUCTION: `route_logits = route_head(pooled)` and `pooled` is "
            "the encoder's output over `frames` alone, while `nav_cmd` is "
            "one-hot'd into the SIBLING `measurement(...)` branch and never "
            "reaches it (refs/refc.py; the v3 nav_inject path touches only "
            "z_tac_raw and ctx). The witness: paired_true_minus_shuffled_"
            "accuracy = 0.0 [0.0, 0.0], paired episode-cluster bootstrap, "
            "n = 3622 windows / 128 episodes — ⚠️ MEASURED AT STEP 30000 "
            "(taniteval/results/refcv3-30k-openloop-20260903-2004.json), NOT "
            "at this checkpoint; the architectural argument is what carries "
            "here. ⚠️ It is NOT a defect: on the 1736 changed-nav windows the "
            "head follows the LABEL at 0.7414 and the WRONG fed nav at 0.2224 "
            "— vision-derived route skill, the opposite of flagship v1's "
            "1.0000 echo of its own input. The TACTICAL heads are a different "
            "story and DO move with nav (lat +0.0233 [-0.0026,+0.0537] not "
            "separated; lon +0.0225 [+0.0009,+0.0417] separated, same run)."),
    }
    with open(a.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(side, fh, indent=2, ensure_ascii=False)
    _p(f"\n[video] {a.out}  {n_out} frames  {n_out / a.fps:.0f}s  "
       f"{W_TOT}x{H_TOT}")
    _p(f"[video] sidecar {a.out}.json")


def kin3_row(sel_grid, gt_grid, dt):
    """The path-derived 3-way lat/lon decision for the model path AND the GT path.

    ⭐ ONE INSTRUMENT, IMPORTED — ``four_families.maneuver_kinematics`` (the
    four inputs) into ``refc_tactical.factor_from_kinematics`` (the classifier).
    A second copy of either would be a second rule, and this programme has paid
    for that before. Both arms go through the identical call on the identical
    grid, so a disagreement here is a disagreement about the DECISION, not about
    two labellers.

    ⚠️ Two approximations travel with it and are the honest cost of reading a
    decision off a path: ``dyaw`` is the path TANGENT at the horizon (not the
    vehicle yaw — they diverge at low speed) and ``v0``/``v1`` are one-step
    CHORD speeds (~1 % under the recorded speed on a curve). Both are the
    four-families eval's own caveats, unchanged.
    """
    from taniteval.four_families import maneuver_kinematics
    from tanitad.refs.refc_tactical import (LAT_CLASSES, LON_CLASSES,
                                            factor_from_kinematics)
    wp = torch.as_tensor(np.stack([sel_grid, gt_grid]), dtype=torch.float32)
    dyaw, dv, v0, v1, _prov = maneuver_kinematics(wp, float(dt))
    lat, lon = factor_from_kinematics(dyaw, dv, v0, v1)
    ls, lg = int(lat[0]), int(lat[1])
    ns, ng = int(lon[0]), int(lon[1])
    return {"lat_sel": ls, "lat_gt": lg, "lon_sel": ns, "lon_gt": ng,
            "lat_sel_name": LAT_CLASSES[ls], "lat_gt_name": LAT_CLASSES[lg],
            "lon_sel_name": LON_CLASSES[ns], "lon_gt_name": LON_CLASSES[ng]}


def _ego_future(poses, t, k):
    """Future GT positions t+1..t+k in the ego frame of pose t. [<=k, 2]."""
    x0, y0, yaw = float(poses[t, 0]), float(poses[t, 1]), float(poses[t, 2])
    c, s = math.cos(yaw), math.sin(yaw)
    fut = poses[t + 1:t + 1 + k, :2].float().numpy() - np.array([x0, y0])
    return np.stack([c * fut[:, 0] + s * fut[:, 1],
                     -s * fut[:, 0] + c * fut[:, 1]], 1)


def _past_path(poses, t, k):
    """Where the ego CAME from, in the ego frame at t (context, not a claim)."""
    p = poses[max(0, t - k):t + 1, :2].float().numpy()
    if len(p) < 2:
        return None
    c, s = math.cos(float(poses[t, 2])), math.sin(float(poses[t, 2]))
    d = p - p[-1]
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], 1)


if __name__ == "__main__":
    main()
